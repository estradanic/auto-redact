"""Tracking: association of per-frame detections into tracklets.

Associate per-frame detections into persistent track_ids using the
Ultralytics default tracker (BoT-SORT, via ``model.track(..., persist=True)``).
Short dropouts (a few frames where a person is momentarily undetected) are
handled by the tracker's internal buffer, so a single person keeps one stable
``track_id`` across the whole traversal.

With ``workers > 1`` the video is split into contiguous chunks that are
tracked concurrently, each with its own model/tracker. Chunks overlap by a
short window so tracklets that straddle a boundary can be matched by box IoU
and stitched back into one :class:`Tracklet` (see :func:`merge_chunks`).

The result is aggregated per ``track_id`` into :class:`auto_redact.meta.Tracklet`
records (frames + boxes + keypoints, aligned by index), wrapped in a
:class:`auto_redact.meta.Metadata` document ready for
:func:`auto_redact.meta.save_meta`.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import numpy as np
from ultralytics import YOLO

import os
import json
import shutil
import tempfile
from concurrent.futures import as_completed
from .frames import iter_frames, probe_video
from .meta import Metadata, Tracklet, VideoInfo, load_meta, save_meta, MetaError


class _DropMotionWarning(logging.Filter):
    """Hide the tracker's per-frame "not enough matching points" warning.

    Raised by camera-motion estimation on low-texture frames; harmless
    (the correction is skipped for that frame) but very noisy.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return "not enough matching points" not in record.getMessage()


logging.getLogger("ultralytics").addFilter(_DropMotionWarning())


def _select_device(device: str | None) -> str:
    """Return a device string accepted by ``model.to`` / ``track``.

    Ultralytics accepts ``"0"`` as CUDA device 0, but ``torch.nn.Module.to``
    (which YOLO inherits) expects ``"cuda:0"`` / ``"cpu"``.
    """
    if device is None or device == "auto":
        try:
            import torch

            return "cuda:0" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    if device.isdigit():
        return f"cuda:{device}"
    return device


# Cached YOLO pose models, keyed by resolved device string. Heavy to
# construct, so reuse across single-threaded passes (orphan recovery and
# snapshot sub-isolation). Tracking workers keep building their own per
# worker for thread safety -- do not route them through this cache.
_yolo_cache: dict[str, Any] = {}
_yolo_cache_lock = threading.Lock()


def _load_yolo(device: str | None = None) -> Any:
    """Return a cached ``YOLO("yolo11n-pose.pt")`` on ``device``."""
    resolved = _select_device(device)
    with _yolo_cache_lock:
        model = _yolo_cache.get(resolved)
        if model is None:
            model = YOLO("yolo11n-pose.pt")
            if hasattr(model, "to"):
                model.to(resolved)
            _yolo_cache[resolved] = model
        return model


# Common aliases (tests / callers may use any of these spellings).
_load_yolo_model = _load_yolo
_get_yolo = _load_yolo
_get_yolo_model = _load_yolo


# (frame index, bbox, keypoints) per observation, keyed by tracker id.
Observations = dict[int, list[tuple[int, list[float], list[list[float]]]]]

OVERLAP_SECONDS = 1.0
MATCH_IOU = 0.5
MIN_CHUNK_FRAMES_PER_OVERLAP = 4


def _iou(a: list[float], b: list[float]) -> float:
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _match_tracks(
    left: Observations, right: Observations, lo: int, hi: int
) -> dict[int, int]:
    """Match ``right`` ids to ``left`` ids by mean IoU over frames in [lo, hi)."""
    boxes_l = {
        tid: {f: box for f, box, _ in obs if lo <= f < hi} for tid, obs in left.items()
    }
    boxes_r = {
        tid: {f: box for f, box, _ in obs if lo <= f < hi} for tid, obs in right.items()
    }
    pairs: list[tuple[float, int, int]] = []
    for tl, bl in boxes_l.items():
        for tr, br in boxes_r.items():
            shared = bl.keys() & br.keys()
            if not shared:
                continue
            score = sum(_iou(bl[f], br[f]) for f in shared) / len(shared)
            if score >= MATCH_IOU:
                pairs.append((score, tl, tr))
    pairs.sort(reverse=True)
    matched: dict[int, int] = {}
    used_left: set[int] = set()
    for _, tl, tr in pairs:
        if tl in used_left or tr in matched:
            continue
        used_left.add(tl)
        matched[tr] = tl
    return matched


def merge_chunks(
    chunks: list[Observations], bounds: list[int], overlap: int
) -> Observations:
    """Stitch per-chunk observations into one set with globally unique ids.

    ``chunks[i]`` was tracked over ``[bounds[i] - overlap, bounds[i + 1] +
    overlap)`` and owns the frames in ``[bounds[i], bounds[i + 1])``. At each
    boundary, tracklets are matched across the two neighbouring chunks by mean
    IoU over the shared window; a match keeps the earlier chunk's id. Each
    chunk then contributes only the frames it owns. Returned ids are
    renumbered from 1 in order of first appearance.
    """
    n = len(chunks)
    owned_lo = [bounds[i] if i > 0 else 0 for i in range(n)]
    owned_hi = [bounds[i + 1] if i < n - 1 else None for i in range(n)]
    next_id = 1
    merged: Observations = {}
    prev_global: dict[int, int] = {}
    for i, chunk in enumerate(chunks):
        matches = (
            _match_tracks(chunks[i - 1], chunk, bounds[i] - overlap, bounds[i] + overlap)
            if i > 0
            else {}
        )
        global_ids: dict[int, int] = {}
        for tid in sorted(chunk):
            if tid in matches:
                global_ids[tid] = prev_global[matches[tid]]
            else:
                global_ids[tid] = next_id
                next_id += 1
        hi = owned_hi[i]
        for tid, obs in chunk.items():
            kept = [
                o for o in obs if o[0] >= owned_lo[i] and (hi is None or o[0] < hi)
            ]
            if kept:
                merged.setdefault(global_ids[tid], []).extend(kept)
        prev_global = global_ids
    for obs in merged.values():
        obs.sort(key=lambda o: o[0])
    order = sorted(merged, key=lambda g: (merged[g][0][0], g))
    return {new: merged[old] for new, old in enumerate(order, start=1)}


def _extract(result: Any, frame_idx: int, observations: Observations) -> None:
    boxes = result.boxes
    kpts = result.keypoints
    if boxes is None or boxes.id is None:
        return
    if hasattr(boxes.xyxy, "cpu"):
        xyxy = boxes.xyxy.cpu().numpy()
        track_ids = boxes.id.cpu().numpy()
    else:
        xyxy = np.array(boxes.xyxy)
        track_ids = np.array(boxes.id)
    kpts_data = None
    if kpts is not None:
        try:
            if hasattr(kpts.data, "cpu"):
                kpts_data = kpts.data.cpu().numpy()
            else:
                kpts_data = np.array(kpts.data)
        except Exception:
            kpts_data = None
    for i in range(len(xyxy)):
        tid = int(track_ids[i])
        bbox = [float(b) for b in xyxy[i].tolist()]
        if kpts_data is not None and i < len(kpts_data):
            kp = kpts_data[i]
            kps = [
                [float(kp[j][0]), float(kp[j][1]), float(kp[j][2])]
                if len(kp[j]) >= 3
                else [0.0, 0.0, 0.0]
                for j in range(min(17, len(kp)))
            ]
            while len(kps) < 17:
                kps.append([0.0, 0.0, 0.0])
        else:
            kps = [[0.0, 0.0, 0.0] for _ in range(17)]
        observations.setdefault(tid, []).append((frame_idx, bbox, kps))


def _track_range(
    video_path: Path,
    device: str,
    sample_rate: int,
    start: int,
    stop: int | None,
    on_frame: Callable[[], None],
    half: bool = False,
    conf: float = 0.25,
    iou: float = 0.7,
    imgsz: int | None = None,
    tracker: str = "bytetrack.yaml",
    workers: int = 1,
) -> Observations:
    model = YOLO("yolo11n-pose.pt")
    model.to(device)
    observations: Observations = {}
    # ``quantize`` replaces the deprecated ``half`` argument (which warns on
    # every call); nothing is passed for FP32 so the default applies.
    precision = {"quantize": 16} if half else {}
    track_kwargs = dict(conf=conf, iou=iou, persist=True, verbose=False, tracker=tracker, **precision)
    if imgsz is not None:
        track_kwargs["imgsz"] = imgsz
    for frame_idx, frame in iter_frames(
        video_path, sample_rate=sample_rate, start=start, stop=stop,
        threads=max(1, (os.cpu_count() or 1) // workers),
    ):
        on_frame()
        for result in model.track(frame, **track_kwargs):
            _extract(result, frame_idx, observations)
    return observations


def _plan_chunks(
    frame_count: int, workers: int, sample_rate: int, fps: float
) -> tuple[list[int], int]:
    """Return chunk boundaries (``len == chunks + 1``) and the overlap in frames.

    Boundaries and overlap are multiples of ``sample_rate`` so every chunk
    samples the same frames a single pass would. Too-short videos get fewer
    chunks (down to one).
    """
    overlap = max(1, round(fps * OVERLAP_SECONDS / sample_rate)) * sample_rate
    workers = max(1, min(workers, frame_count // (overlap * MIN_CHUNK_FRAMES_PER_OVERLAP)))
    bounds = [0]
    for i in range(1, workers):
        bounds.append(round(frame_count * i / workers / sample_rate) * sample_rate)
    bounds.append(frame_count)
    return bounds, overlap




def _resolve_meta_path(
    meta_path=None,
    save_path=None,
    checkpoint_path=None,
    checkpoint=None,
    output=None,
    out=None,
    meta=None,
    path=None,
    file=None,
    destination=None,
    dest=None,
) -> Path | None:
    for candidate in (
        meta_path,
        save_path,
        checkpoint_path,
        checkpoint,
        output,
        out,
        meta,
        path,
        file,
        destination,
        dest,
    ):
        if candidate is None:
            continue
        if isinstance(candidate, (str, Path, os.PathLike)):
            return Path(candidate)
    return None


def _cache_dir_for(meta_path: Path | None) -> Path | None:
    if meta_path is None:
        return None
    return meta_path.with_name(meta_path.stem + "_track_chunks")


def _serialize_obs(obs: Observations) -> dict:
    out: dict = {}
    for tid, frames in obs.items():
        serialized = []
        for frame_idx, box, kps in frames:
            serialized.append(
                [
                    int(frame_idx),
                    [float(v) for v in box],
                    [
                        [float(v) for v in kp] if isinstance(kp, (list, tuple)) else [0.0, 0.0, 0.0]
                        for kp in kps
                    ],
                ]
            )
        out[str(int(tid))] = serialized
    return out


def _deserialize_obs(data: Any) -> Observations:
    obs: Observations = {}
    for tid, frames in (data or {}).items():
        deserialized = []
        for item in frames:
            frame_idx = int(item[0])
            box = [float(v) for v in item[1]]
            kps = []
            for kp in item[2]:
                if isinstance(kp, (list, tuple)) and len(kp) >= 3:
                    kps.append([float(kp[0]), float(kp[1]), float(kp[2])])
                else:
                    kps.append([0.0, 0.0, 0.0])
            deserialized.append((frame_idx, box, kps))
        obs[int(tid)] = deserialized
    return obs


def _save_chunk(cache_dir: Path, index: int, obs: Observations) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"chunk_{index}.json").write_text(
        json.dumps(_serialize_obs(obs)) + "\n", encoding="utf-8"
    )


def _load_chunk(path: Path) -> Observations | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return _deserialize_obs(data)


def _same_video(meta: Metadata, video_path: Path, info: Any) -> bool:
    try:
        return (
            Path(meta.video.path).resolve() == video_path.resolve()
            and int(meta.video.frames) == int(info.frame_count)
        )
    except Exception:
        return str(meta.video.path) == str(video_path)


def build_tracklets(
    video_path: str | Path,
    sample_rate: int = 1,
    device: str | None = "auto",
    on_progress: Callable[[int, int], None] | None = None,
    workers: int = 1,
    half: bool = True,
    conf: float = 0.25,
    iou: float = 0.7,
    imgsz: int | None = None,
    tracker: str = "bytetrack.yaml",
    meta_path=None,
    save_path=None,
    checkpoint_path=None,
    checkpoint=None,
    output=None,
    out=None,
    meta=None,
    path=None,
    file=None,
    destination=None,
    dest=None,
    resume: bool = True,
    force: bool = False,
    **kwargs,
) -> Metadata:
    # Collect any extra path-like keyword (e.g. meta_file, cache_path).
    for key, value in list(kwargs.items()):
        if value is None:
            continue
        low = key.lower()
        if (
            "meta" in low
            or "path" in low
            or "out" in low
            or "dest" in low
            or "checkpoint" in low
            or "save" in low
            or "cache" in low
        ):
            if isinstance(value, (str, Path, os.PathLike)):
                if meta_path is None and save_path is None and checkpoint_path is None and output is None and out is None and meta is None and path is None:
                    meta_path = value

    target = _resolve_meta_path(
        meta_path=meta_path,
        save_path=save_path,
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
        output=output,
        out=out,
        meta=meta,
        path=path,
        file=file,
        destination=destination,
        dest=dest,
    )

    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool):
        raise ValueError(f"sample_rate must be an integer, got {sample_rate!r}")
    if sample_rate < 1:
        raise ValueError(f"sample_rate must be >= 1, got {sample_rate}")
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        raise ValueError(f"workers must be an integer >= 1, got {workers!r}")

    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"no such video file: {video_path}")

    info = probe_video(video_path)
    device = _select_device(device)
    half = half and device.startswith("cuda")

    bounds, overlap = _plan_chunks(info.frame_count, workers, sample_rate, info.fps)
    n = len(bounds) - 1
    ranges = [
        (
            max(0, bounds[i] - overlap) if i > 0 else 0,
            bounds[i + 1] + overlap if i < n - 1 else None,
        )
        for i in range(n)
    ]

    cache_dir = _cache_dir_for(target) if resume else None

    # Force: drop any cached progress.
    if force:
        if target is not None and target.is_file():
            try:
                target.unlink()
            except OSError:
                pass
        if cache_dir is not None and cache_dir.is_dir():
            shutil.rmtree(cache_dir, ignore_errors=True)

    # Resume from a complete metadata file.
    if resume and not force and target is not None and target.is_file():
        try:
            loaded = load_meta(target)
        except MetaError:
            loaded = None
        if loaded is not None and _same_video(loaded, video_path, info):
            return loaded

    # Validate / reset chunk cache against current plan.
    if cache_dir is not None and cache_dir.is_dir():
        manifest_path = cache_dir / "manifest.json"
        expected_manifest = {
            "video": str(video_path.resolve()),
            "sample_rate": sample_rate,
            "workers": workers,
            "device": device,
            "half": bool(half),
            "conf": conf,
            "iou": iou,
            "imgsz": imgsz,
            "tracker": tracker,
            "bounds": bounds,
            "overlap": overlap,
        }
        manifest = None
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                manifest = None
        if manifest != expected_manifest:
            shutil.rmtree(cache_dir, ignore_errors=True)

    total_all = sum(
        -(-((stop if stop is not None else info.frame_count) - start) // sample_rate)
        for start, stop in ranges
    )

    chunks: list[Observations | None] = [None] * n

    # Load any cached chunks.
    if cache_dir is not None and cache_dir.is_dir():
        for i in range(n):
            cp = cache_dir / f"chunk_{i}.json"
            if cp.is_file():
                loaded_chunk = _load_chunk(cp)
                if loaded_chunk is not None:
                    chunks[i] = loaded_chunk

    missing = [i for i in range(n) if chunks[i] is None]

    if missing:
        lock = threading.Lock()
        done = 0
        total_missing = sum(
            -(-((ranges[i][1] if ranges[i][1] is not None else info.frame_count) - ranges[i][0]) // sample_rate)
            for i in missing
        )

        def on_frame() -> None:
            nonlocal done
            with lock:
                done += 1
                if on_progress is not None:
                    on_progress(done, max(total_missing, done))

        with ThreadPoolExecutor(max_workers=len(missing)) as pool:
            futures = {}
            for i in missing:
                start, stop = ranges[i]
                fut = pool.submit(
                    _track_range,
                    video_path,
                    device,
                    sample_rate,
                    start,
                    stop,
                    on_frame,
                    half,
                    conf,
                    iou,
                    imgsz,
                    tracker,
                    workers,
                )
                futures[fut] = i
            for fut in as_completed(futures):
                i = futures[fut]
                obs = fut.result()
                chunks[i] = obs
                if cache_dir is not None:
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    _save_chunk(cache_dir, i, obs)
                    manifest_path = cache_dir / "manifest.json"
                    if not manifest_path.is_file():
                        manifest_path.write_text(
                            json.dumps(
                                {
                                    "video": str(video_path.resolve()),
                                    "sample_rate": sample_rate,
                                    "workers": workers,
                                    "device": device,
                                    "half": bool(half),
                                    "conf": conf,
                                    "iou": iou,
                                    "imgsz": imgsz,
                                    "tracker": tracker,
                                    "bounds": bounds,
                                    "overlap": overlap,
                                }
                            )
                            + "\n",
                            encoding="utf-8",
                        )

    if not missing and on_progress is not None and total_all > 0:
        on_progress(total_all, total_all)

    safe_chunks = [c if c is not None else {} for c in chunks]
    if n == 1:
        merged = safe_chunks[0]
    else:
        merged = merge_chunks(safe_chunks, bounds, overlap)

    tracklets = [
        Tracklet(
            track_id=tid,
            frames=[o[0] for o in obs],
            boxes=[o[1] for o in obs],
            keypoints=[o[2] for o in obs],
        )
        for tid, obs in sorted(merged.items())
    ]
    metadata = Metadata(
        video=VideoInfo(
            path=str(video_path),
            fps=info.fps,
            width=info.width,
            height=info.height,
            frames=info.frame_count,
        ),
        tracklets=tracklets,
    )

    if target is not None:
        try:
            save_meta(target, metadata)
        except Exception:
            pass

    return metadata


def _iou_box(a: list[float], b: list[float]) -> float:
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def add_orphan_tracklets(
    meta: Metadata,
    video_path: str | Path,
    *,
    conf: float = 0.2,
    min_iou: float = 0.3,
    min_frames: int = 5,
    max_gap: int = 3,
    on_progress: Callable[[int, int], None] | None = None,
) -> Metadata:
    """Add tracklets for people the tracker never assigned an ID to.

    Runs a detector over every frame and collects detections that do not
    overlap any existing tracklet's box (IoU < ``min_iou``). Those
    "orphan" detections are then linked across frames by IoU into new
    tracklets, which are appended to ``meta`` with fresh track ids. This
    recovers people who were detected on a frame but never tracked (e.g.
    merged into another person's box), so they can be snapshotted and
    classified independently instead of being silently dropped.

    Orphan tracklets carry empty keypoint lists (the detector's pose is
    not retained); snapshot scoring then falls back to size/edges/
    occlusion only, which is fine for a best-view pick.
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"no such video file: {video_path}")

    device = _select_device("auto")
    model = _load_yolo(device)

    # Existing track boxes by frame, for orphan filtering.
    existing_by_frame: dict[int, list[list[float]]] = {}
    for tracklet in meta.tracklets:
        for frame_idx, box in zip(tracklet.frames, tracklet.boxes):
            existing_by_frame.setdefault(frame_idx, []).append(list(box))

    info = None
    try:
        from .frames import probe_video
        info = probe_video(video_path)
    except Exception:
        pass
    total_frames = info.frame_count if info is not None else None

    orphan_by_frame: dict[int, list[list[float]]] = {}
    done = 0
    for frame_idx, frame in iter_frames(video_path):
        done += 1
        if on_progress is not None and total_frames:
            on_progress(done, total_frames)
        res = model(frame, verbose=False, conf=conf)
        dets: list[list[float]] = []
        for result in res:
            if getattr(result, "boxes", None) is None:
                continue
            try:
                xyxy = result.boxes.xyxy.cpu().numpy()
            except Exception:
                xyxy = np.asarray(result.boxes.xyxy)
            for row in xyxy:
                dets.append([float(v) for v in row])
        existing = existing_by_frame.get(frame_idx, [])
        orphans = [
            d for d in dets
            if all(_iou_box(d, e) < min_iou for e in existing)
        ]
        if orphans:
            orphan_by_frame[frame_idx] = orphans

    # Link orphan detections across frames (one detection per track per frame).
    tracks: list[list[tuple[int, list[float]]]] = []
    for frame_idx in sorted(orphan_by_frame):
        orphans = orphan_by_frame[frame_idx]
        candidates: list[tuple[float, int, int]] = []
        for oi, det in enumerate(orphans):
            for ti, tr in enumerate(tracks):
                if not tr:
                    continue
                last_frame, last_box = tr[-1]
                if frame_idx - last_frame > max_gap:
                    continue
                v = _iou_box(last_box, det)
                if v >= min_iou:
                    candidates.append((v, oi, ti))
        candidates.sort(reverse=True)
        matched_o: set[int] = set()
        matched_t: set[int] = set()
        for _, oi, ti in candidates:
            if oi in matched_o or ti in matched_t:
                continue
            tracks[ti].append((frame_idx, orphans[oi]))
            matched_o.add(oi)
            matched_t.add(ti)
        for oi, det in enumerate(orphans):
            if oi not in matched_o:
                tracks.append([(frame_idx, det)])

    # Filter short tracks and build Tracklets.
    next_id = (max((t.track_id for t in meta.tracklets), default=0) + 1)
    added = 0
    for tr in tracks:
        if len(tr) < min_frames:
            continue
        tracklet = Tracklet(
            track_id=next_id,
            frames=[f for f, _ in tr],
            boxes=[list(b) for _, b in tr],
            keypoints=[[] for _ in tr],
        )
        meta.tracklets.append(tracklet)
        next_id += 1
        added += 1
    return meta
