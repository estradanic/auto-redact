"""Streaming single-pass pipeline (Option A).

Reads the source video once. Tracks, scores best-view snapshots, detects scene
cuts and classifies people as they leave the frame, and writes redacted frames
to the output as soon as every person visible in them is decided. Frames whose
subjects are not yet decided are buffered (spilled to disk past a threshold)
and shipped later. Worst case (one subject present for the whole video) is
identical to the classic pipeline, but with one decode instead of several and
partial early output for short-lived subjects.

This module deliberately keeps the classic ``pipeline.run_pipeline`` untouched
so the two can be compared.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import subprocess
import tempfile
import threading
import queue as _queue
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, ContextManager

import cv2
import numpy as np
from PIL import Image

from .classify import (
    DEFAULT_MAX_IMAGE_SIZE,
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_URL,
    DEFAULT_TIMEOUT,
    DEFAULT_WORKERS,
    VlmClassifier,
    classify_snapshots,
)
from .frames import iter_frames, probe_video
from .meta import Metadata, MetaError, Tracklet, VideoInfo, load_meta, save_meta
from .render import (
    DEFAULT_BRIDGE_IOU,
    DEFAULT_CRF,
    DEFAULT_ENCODER,
    DEFAULT_MAX_GAP,
    DEFAULT_MIN_TRACK_FRAMES,
    DEFAULT_PRESET,
    DEFAULT_SMOOTH_WINDOW,
    ENCODERS,
    _box_iou,
    _clamp_box,
    _fill_tracklet_gaps,
    _lerp_box,
    _smooth_boxes,
    redaction_boxes,
    _Segment,
    _copy_segment,
    _encode_segment,
    _concat_video,
    _mux_audio,
    _keyframe_table,
)
from .pipeline import _classify_key
from .scene import _frame_hist
from .siglip import DEFAULT_SIGLIP_MODEL, DEFAULT_THRESHOLD, SiglipClassifier
from .snapshot import (
    CLOSEUP_AREA_FRACTION,
    DEFAULT_CONTACT_SHEET,
    DEFAULT_PADDING,
    DISTRACTOR_WEIGHT,
    EDGE_PENALTY,
    EDGE_TOLERANCE,
    HEAD_WEIGHT,
    BODY_WEIGHT,
    ISOLATE_DILATE,
    SAME_PERSON_IOU,
    _clamped_area,
    _padded_crop_window,
    crop_snapshot,
    observation_score,
    redact_others,
    sub_isolate_people,
    write_contact_sheet,
)
try:
    import torch as _torch
except Exception:
    _torch = None
from .track import _select_device, _load_yolo, _extract, _iou, _same_video


# Reuse the same defaults as the classic pipeline where sensible.
DEFAULT_SPILL_THRESHOLD = 128


@dataclass
class _Active:
    internal_id: int
    tracker_id: int
    frames: list[int] = field(default_factory=list)
    boxes: list[list[float]] = field(default_factory=list)
    keypoints: list[list[list[float]]] = field(default_factory=list)
    first_frame: int = 0
    last_frame: int = 0
    best_frame: int | None = None
    best_box: list[float] | None = None
    best_score: float = -1.0
    best_others: list[tuple[int, list[float]]] = field(default_factory=list)
    closed: bool = False


def _select_tracker_device(device):
    return _select_device(device)


def _bgr(frame_rgb):
    return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)


def run_pipeline_streaming(
    video,
    prompt,
    out,
    *,
    meta_path=None,
    thumbs_dir=None,
    force=False,
    sample_rate=1,
    device="auto",
    track_workers=1,  # accepted for compat; streaming tracks in one pass
    half=True,
    track_conf=0.25,
    track_iou=0.7,
    track_imgsz=None,
    track_tracker="bytetrack.yaml",
    orphans=False,
    resume=True,
    padding=DEFAULT_PADDING,
    isolate=True,
    sub_isolate=False,
    closeup_area_fraction=CLOSEUP_AREA_FRACTION,
    head_weight=HEAD_WEIGHT,
    body_weight=BODY_WEIGHT,
    edge_penalty=EDGE_PENALTY,
    edge_tolerance=EDGE_TOLERANCE,
    distractor_weight=DISTRACTOR_WEIGHT,
    same_person_iou=SAME_PERSON_IOU,
    isolate_dilate=ISOLATE_DILATE,
    classifier="vlm",
    model=None,
    threshold=DEFAULT_THRESHOLD,
    classify_device="auto",
    ollama_url=DEFAULT_OLLAMA_URL,
    timeout=DEFAULT_TIMEOUT,
    classify_workers=DEFAULT_WORKERS,
    think=False,
    max_image_size=DEFAULT_MAX_IMAGE_SIZE,
    keep_alive="24h",
    crf=18,
    preset="medium",
    selective=True,  # accepted; streaming uses GOP append
    encoder="libx264",
    max_gap=DEFAULT_MAX_GAP,
    smooth_window=DEFAULT_SMOOTH_WINDOW,
    bridge_iou=DEFAULT_BRIDGE_IOU,  # accepted; bridging skipped in streaming
    min_track_frames=DEFAULT_MIN_TRACK_FRAMES,
    scene_cuts=True,
    scene_cut_threshold=0.45,
    spill_threshold=DEFAULT_SPILL_THRESHOLD,
    bookends=True,
    progress=lambda message: None,
    bar=lambda desc: contextlib.nullcontext(),
) -> dict:
    if classifier not in ("vlm", "siglip"):
        raise ValueError(f"unknown classifier {classifier!r}")
    if model is None:
        model = DEFAULT_SIGLIP_MODEL if classifier == "siglip" else DEFAULT_MODEL

    # Defer rendering (and frame buffering) when features that need a full pass
    # are requested: orphan recovery and selective GOP copy. Decision-making
    # (track + snapshot + classify) still happens in one pass.
    defer_render = bool(orphans)

    video = Path(video)
    out = Path(out)
    meta_file = Path(meta_path) if meta_path else out.with_name(out.stem + ".meta.json")
    thumbs = Path(thumbs_dir) if thumbs_dir else meta_file.with_name(meta_file.stem + "_thumbs")
    thumbs.mkdir(parents=True, exist_ok=True)
    sidecar = meta_file.with_name(meta_file.name + ".classify.json")
    ckpt_path = meta_file.with_name(meta_file.name + ".stream.ckpt.json")

    info = probe_video(video)
    device = _select_device(device)

    _gpu_lock = threading.Lock()
    _state_lock = threading.Lock()

    def _manage_vram(tag=""):
        if _torch is not None and _torch.cuda.is_available():
            try:
                alloc = _torch.cuda.memory_allocated() / (1024**3)
                total = _torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if alloc > 0.85 * total:
                    progress(f"      VRAM high ({tag}): {alloc:.1f}/{total:.1f} GB")
                _torch.cuda.synchronize()
            except Exception:
                pass

    # Reuse cached result if available (mirrors classic pipeline).
    if (not force) and meta_file.is_file() and out.is_file():
        try:
            loaded = load_meta(meta_file)
        except MetaError:
            loaded = None
        if loaded is not None and _same_video(loaded, video, info):
            key = _classify_key(prompt, classifier, model, threshold, think)
            try:
                stored = json.loads(sidecar.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                stored = None
            if stored == key and all(t.redact is not None for t in loaded.tracklets):
                progress("reusing cached result")
                redacted = sum(1 for t in loaded.tracklets if t.redact)
                return {"out": out, "meta_path": meta_file, "tracklets": len(loaded.tracklets), "redacted": redacted}

    def _ser_active(at):
        return {
            "internal_id": at.internal_id,
            "tracker_id": at.tracker_id,
            "frames": list(at.frames),
            "boxes": [list(b) for b in at.boxes],
            "keypoints": [list(k) for k in at.keypoints],
            "first_frame": at.first_frame,
            "last_frame": at.last_frame,
            "best_frame": at.best_frame,
            "best_box": (None if at.best_box is None else list(at.best_box)),
            "best_score": at.best_score,
            "best_others": [(oid, list(ob)) for oid, ob in at.best_others],
            "closed": at.closed,
        }

    def _deser_active(d):
        at = _Active(internal_id=d["internal_id"], tracker_id=d["tracker_id"],
                     first_frame=d["first_frame"])
        at.frames = list(d["frames"])
        at.boxes = [list(b) for b in d["boxes"]]
        at.keypoints = [list(k) for k in d["keypoints"]]
        at.last_frame = d["last_frame"]
        at.best_frame = d["best_frame"]
        at.best_box = (None if d["best_box"] is None else list(d["best_box"]))
        at.best_score = d["best_score"]
        at.best_others = [(oid, list(ob)) for oid, ob in d["best_others"]]
        at.closed = d["closed"]
        return at

    checkpoint = None
    resuming = False
    if resume and not force and ckpt_path.is_file():
        try:
            checkpoint = json.loads(ckpt_path.read_text(encoding="utf-8"))
            resuming = True
        except (OSError, ValueError):
            checkpoint = None
            resuming = False

    start_frame = 0
    if resuming and checkpoint is not None:
        start_frame = int(checkpoint.get("last_frame", -1)) + 1

    # Fresh run cleanup.
    if not resuming:
        try:
            growing_ts.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            ckpt_path.unlink(missing_ok=True)
        except Exception:
            pass

    selective_stream = False
    gops = []
    gop_frames = {}
    gop_blockers = {}
    shipped_gops = set()
    segments = []
    keyframes = {}
    is_open_gop = False
    if selective and not defer_render:

        keyframes, is_open_gop = _keyframe_table(video)
        if keyframes and not is_open_gop and 0 in keyframes:
            selective_stream = True
            boundaries = sorted({0, info.frame_count} | set(keyframes))
            gops = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1) if boundaries[i + 1] > boundaries[i]]
        else:
            selective_stream = False
    half = half and device.startswith("cuda")

    # classifier instance
    siglip = None
    if classifier == "siglip":
        progress("      initializing classifier model")
        siglip = SiglipClassifier(model, threshold, classify_device)
    else:
        progress("      waiting for ollama to load the model")

    def make_vlm():
        return VlmClassifier(model, ollama_url, timeout, think, keep_alive, max_image_size)

    yolo = _load_yolo(device if device != "auto" else None)
    # ensure on device
    try:
        yolo.to(device)
    except Exception:
        pass

    w, h = info.width, info.height
    fps = info.fps

    # state
    next_internal_id = 1
    active: dict[int, _Active] = {}  # tracker_id -> active
    closed_tracklets: list[_Active] = []
    redact_by_id: dict[int, bool | None] = {}
    score_by_id: dict[int, float | None] = {}
    snapshot_path_by_id: dict[int, str | None] = {}
    isolated_by_id: dict[int, list[int]] = {}
    best_frame_by_id: dict[int, int | None] = {}

    # render boxes accumulated as tracklets close
    render_boxes: dict[int, list[tuple[int, int, int, int]]] = {}
    closed_redacted: list[_Active] = []
    unblock_at: dict[int, set[int]] = {}

    # frame buffer: idx -> ndarray or Path (spilled)
    frame_buffer: dict[int, Any] = {}
    spill_dir = Path(tempfile.mkdtemp(prefix="auto-redact-stream-"))
    growing_ts = out.with_suffix(".ts")
    in_memory_count = [0]

    frame_blockers: dict[int, set[int]] = {}

    next_ship = [0]

    # Restore from checkpoint if resuming.
    if resuming and checkpoint is not None:
        next_internal_id = int(checkpoint.get("next_internal_id", next_internal_id))
        # closed tracklets
        for d in checkpoint.get("closed", []):
            at = _deser_active(d["active"])
            closed_tracklets.append(at)
            redact_by_id[at.internal_id] = d.get("redact")
            score_by_id[at.internal_id] = d.get("score")
            snapshot_path_by_id[at.internal_id] = d.get("snapshot_path")
            isolated_by_id[at.internal_id] = list(d.get("isolated") or [])
            best_frame_by_id[at.internal_id] = d.get("best_frame")
            if d.get("redact") is True:
                closed_redacted.append(at)
        # active tracklets
        for d in checkpoint.get("active", []):
            at = _deser_active(d)
            active[at.tracker_id] = at
        # render boxes
        for k, v in (checkpoint.get("render_boxes") or {}).items():
            render_boxes[int(k)] = [tuple(b) for b in v]
        # shipped gops
        for gs in checkpoint.get("shipped_gops", []):
            shipped_gops.add(int(gs))
        # gop frames (re-decode)
        for gs, idxs in (checkpoint.get("gop_frames") or {}).items():
            gop_frames[int(gs)] = []
            for idx in idxs:
                fr = None
                for fi, f in iter_frames(video, start=idx, stop=idx + 1):
                    fr = f
                    break
                if fr is not None:
                    gop_frames[int(gs)].append((idx, fr))
        # gop blockers
        for gs, ids in (checkpoint.get("gop_blockers") or {}).items():
            gop_blockers[int(gs)] = set(ids)
        # frame buffer (re-decode)
        for idx in checkpoint.get("frame_buffer_indices", []):
            fr = None
            for fi, f in iter_frames(video, start=idx, stop=idx + 1):
                fr = f
                break
            if fr is not None:
                frame_buffer[idx] = fr
                in_memory_count[0] += 1
        next_ship[0] = int(checkpoint.get("next_ship", next_ship[0]))

    # ffmpeg
    out.parent.mkdir(parents=True, exist_ok=True)
    has_audio = info.has_audio

    out.parent.mkdir(parents=True, exist_ok=True)
    has_audio = info.has_audio
    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", f"{w}x{h}",
        "-framerate", f"{int(fps)}/{1}" if float(fps).is_integer() else str(fps),
        "-i", "pipe:0",
    ]
    # use fps arg helper-ish
    from fractions import Fraction
    rate = Fraction(fps).limit_denominator(1001)
    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", f"{w}x{h}",
        "-framerate", f"{rate.numerator}/{rate.denominator}",
        "-i", "pipe:0",
    ]
    if has_audio:
        cmd += ["-i", str(video)]
    cmd += ["-map", "0:v:0"]
    if has_audio:
        cmd += ["-map", "1:a:0", "-c:a", "copy"]
    cmd += ["-c:v", "libx264", "-crf", str(crf), "-preset", preset,
            "-pix_fmt", "yuv420p", "-movflags", "frag_keyframe+empty_moov", str(out)]
    proc = None
    if not defer_render and not selective_stream:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    sub_model = None
    if sub_isolate:
        try:
            sub_model = _load_yolo("cpu")
        except Exception:
            sub_model = None

    precision = {"quantize": 16} if half else {}
    track_kwargs = dict(conf=track_conf, iou=track_iou, persist=True, verbose=False, tracker=track_tracker, **precision)
    if track_imgsz is not None:
        track_kwargs["imgsz"] = track_imgsz

    close_threshold = max_gap if max_gap > 0 else 30

    def spill_one():
        # spill the earliest in-memory frame that is blocked (not shippable now)
        candidates = [idx for idx, v in frame_buffer.items() if not isinstance(v, Path)]
        if not candidates:
            return
        candidates.sort()
        # prefer one that can't ship now
        target = None
        for idx in candidates:
            if idx != next_ship[0]:
                target = idx
                break
        if target is None:
            target = candidates[0]
        arr = frame_buffer.pop(target)
        path = spill_dir / f"frame_{target}.npy"
        np.save(path, arr)
        frame_buffer[target] = path
        in_memory_count[0] -= 1

    def store_frame(idx, arr):
        if in_memory_count[0] >= spill_threshold:
            spill_one()
        frame_buffer[idx] = arr
        in_memory_count[0] += 1

    def load_frame(idx):
        v = frame_buffer.get(idx)
        if isinstance(v, Path):
            return np.load(v)
        return v

    # Async classification setup.
    classify_queue = _queue.Queue()
    classified_ids = set()
    tracklet_by_id = {}
    pending_results = {}
    _finalized_ids = set()

    def _classify_one(at):
        """Worker: decode best frame, crop, isolate, classify. Runs in background."""
        redact = None
        score = None
        snap_path = None
        isolated_ids = []
        try:
            if at.best_box is not None and at.best_frame is not None:
                best_arr = None
                for fi, fr in iter_frames(video, start=at.best_frame, stop=at.best_frame + 1):
                    best_arr = fr
                    break
                if best_arr is not None:
                    try:
                        crop = crop_snapshot(best_arr, at.best_box, padding)
                    except Exception:
                        crop = None
                    if crop is not None:
                        from .snapshot import _padded_crop_window, _clamped_area
                        window = _padded_crop_window(at.best_box, w, h, padding)
                        area_fraction = _clamped_area(at.best_box, w, h) / max(1.0, float(w) * float(h))
                        closeup = area_fraction >= closeup_area_fraction
                        if isolate and not closeup:
                            crop, painted = redact_others(
                                crop, window, at.best_box, at.best_others,
                                same_person_iou=same_person_iou, dilate=isolate_dilate,
                            )
                            isolated_ids = painted
                        if sub_isolate and sub_model is not None and not closeup:
                            with _gpu_lock:
                                sub_isolate_people(crop, window, at.best_box, sub_model, dilate=isolate_dilate)
                                _manage_vram("sub_isolate")
                        image = Image.fromarray(crop)
                        thumb_path = thumbs / f"track_{at.internal_id}.jpg"
                        image.save(thumb_path, format="JPEG", quality=90)
                        snap_path = str(thumb_path)

                        # Classify (GPU lock, since ollama/VLM or SigLIP use GPU).
                        with _gpu_lock:
                            if classifier == "siglip":
                                dec = siglip.classify(snap_path, prompt)
                            else:
                                clf = make_vlm()
                                dec = clf.classify(snap_path, prompt)
                            _manage_vram("classify")
                        redact = dec.redact
                        score = dec.score
        except Exception:
            redact = None
            score = None

        with _state_lock:
            pending_results[at.internal_id] = (redact, score, snap_path, isolated_ids)
            classified_ids.add(at.internal_id)

    def _classify_worker():
        while True:
            at = classify_queue.get()
            if at is None:
                classify_queue.task_done()
                break
            try:
                _classify_one(at)
            except Exception:
                with _state_lock:
                    pending_results[at.internal_id] = (None, None, None, [])
                    classified_ids.add(at.internal_id)
            finally:
                classify_queue.task_done()

    _worker_thread = threading.Thread(target=_classify_worker, daemon=True)
    _worker_thread.start()

    def close_tracklet(at: _Active):
        nonlocal next_internal_id
        if at.closed:
            return
        at.closed = True
        tracklet_by_id[at.internal_id] = at
        closed_tracklets.append(at)

        # Schedule unblock (hold for bridging window).
        if max_gap > 0:
            unblock_frame = at.last_frame + max_gap + 1
            unblock_at.setdefault(unblock_frame, set()).add(at.internal_id)
        else:
            for idx in list(frame_blockers.keys()):
                if at.internal_id in frame_blockers[idx]:
                    frame_blockers[idx].discard(at.internal_id)
            for gs in list(gop_blockers.keys()):
                gop_blockers[gs].discard(at.internal_id)

        # Push to classification queue (async).
        classify_queue.put(at)

    def _finalize_classified():
        """Process any newly classified tracklets: render boxes, bridging, meta write."""
        newly = []
        with _state_lock:
            for tid in list(classified_ids):
                if tid in _finalized_ids:
                    continue
                if tid not in pending_results:
                    continue
                newly.append(tid)
        for tid in newly:
            with _state_lock:
                redact, score, snap_path, isolated_ids = pending_results[tid]
                at = tracklet_by_id[tid]
            # compute render boxes if redacted
            if redact is True:
                filled_frames, filled_boxes = _fill_tracklet_gaps(at.frames, at.boxes, max_gap)
                if smooth_window > 1 and len(filled_boxes) > 1:
                    filled_boxes = _smooth_boxes(filled_frames, filled_boxes, smooth_window)
                for f, box in zip(filled_frames, filled_boxes):
                    pb = _clamp_box(box, w, h)
                    if pb is not None:
                        render_boxes.setdefault(f, []).append(pb)
                # bridging
                if bridge_iou > 0 and max_gap > 0:
                    with _state_lock:
                        for prev in closed_redacted:
                            gap = at.first_frame - prev.last_frame - 1
                            if not (0 < gap <= max_gap):
                                continue
                            if _box_iou(prev.boxes[-1], at.boxes[0]) < bridge_iou:
                                continue
                            for step in range(1, gap + 1):
                                t = step / (gap + 1)
                                box = _lerp_box(prev.boxes[-1], at.boxes[0], t)
                                frame_idx = prev.last_frame + step
                                pb = _clamp_box(box, w, h)
                                if pb is not None:
                                    render_boxes.setdefault(frame_idx, []).append(pb)
                        closed_redacted.append(at)
            # store results
            redact_by_id[tid] = redact
            score_by_id[tid] = score
            snapshot_path_by_id[tid] = snap_path
            isolated_by_id[tid] = isolated_ids
            best_frame_by_id[tid] = at.best_frame
            # periodic meta write
            try:
                _tmp_tracklets = []
                with _state_lock:
                    _closed = list(closed_tracklets)
                for _at in sorted(_closed, key=lambda a: a.internal_id):
                    _tmp_tracklets.append(Tracklet(
                        track_id=_at.internal_id,
                        frames=list(_at.frames),
                        boxes=[list(b) for b in _at.boxes],
                        keypoints=[list(k) for k in _at.keypoints],
                        best_frame=best_frame_by_id.get(_at.internal_id),
                        best_snapshot_path=snapshot_path_by_id.get(_at.internal_id),
                        isolated_track_ids=list(isolated_by_id.get(_at.internal_id, [])),
                        redact=redact_by_id.get(_at.internal_id),
                        score=score_by_id.get(_at.internal_id),
                    ))
                save_meta(meta_file, Metadata(video=VideoInfo(path=str(video), fps=fps, width=w, height=h, frames=info.frame_count), tracklets=_tmp_tracklets))
            except Exception:
                pass
            with _state_lock:
                _finalized_ids.add(tid)

    # Main pass
    def save_checkpoint(last_frame):
        try:
            data = {
                "last_frame": last_frame,
                "next_internal_id": next_internal_id,
                "shipped_gops": sorted(shipped_gops),
                "closed": [
                    {"active": _ser_active(at),
                     "redact": redact_by_id.get(at.internal_id),
                     "score": score_by_id.get(at.internal_id),
                     "snapshot_path": snapshot_path_by_id.get(at.internal_id),
                     "isolated": isolated_by_id.get(at.internal_id, []),
                     "best_frame": best_frame_by_id.get(at.internal_id)}
                    for at in closed_tracklets
                ],
                "active": [_ser_active(at) for at in active.values()],
                "render_boxes": {str(k): [list(b) for b in v] for k, v in render_boxes.items()},
                "gop_frames": {str(gs): [fi for fi, _ in frames] for gs, frames in gop_frames.items()},
                "gop_blockers": {str(gs): sorted(bs) for gs, bs in gop_blockers.items()},
                "next_ship": next_ship[0],
                "frame_buffer_indices": sorted(frame_buffer.keys()),
            }
            ckpt_path.write_text(json.dumps(data))
        except Exception:
            pass

    prev_hist = None
    total = info.frame_count
    progress(f"[stream] tracking+rendering {video}")

    # We'll iterate all frames
    frame_iter = iter_frames(video, sample_rate=1, start=start_frame)
    # track sampled frames separately; but iter_frames sample_rate=1 yields all.
    last_seen_global = 0

    _stream_bar_ctx = None
    _stream_on_progress = None
    try:
        _stream_bar_ctx = bar("stream")
        _stream_on_progress = _stream_bar_ctx.__enter__()
    except Exception:
        _stream_bar_ctx = None
        _stream_on_progress = None

    for frame_idx, frame_rgb in frame_iter:
        if _stream_on_progress is not None:
            try:
                _stream_on_progress(frame_idx + 1, total)
            except Exception:
                pass
        # checkpoint previous frame's state
        if frame_idx > start_frame and (frame_idx - 1) % 50 == 0:
            try:
                save_checkpoint(frame_idx - 1)
            except Exception:
                pass
        # progress
        if frame_idx % 50 == 0:
            progress(f"      frame {frame_idx}/{total}")

        # scene cut
        bgr = _bgr(frame_rgb)
        hist = _frame_hist(bgr)
        if prev_hist is not None and scene_cuts:
            dist = float(__import__("cv2").compareHist(prev_hist, hist, __import__("cv2").HISTCMP_BHATTACHARYYA))
            if dist >= scene_cut_threshold:
                for tid in list(active.keys()):
                    close_tracklet(active.pop(tid))
        prev_hist = hist

        # tracking on sampled frames
        if frame_idx % sample_rate == 0:
            # Use model.track returns list; _extract fills obs dict
            obs: dict[int, list] = {}
            with _gpu_lock:
                _track_results = list(yolo.track(frame_rgb, **track_kwargs))
                _manage_vram("track")
            for result in _track_results:
                _extract(result, frame_idx, obs)
            # obs: tid -> [(frame_idx, box, kps)]
            for tid, frames_obs in obs.items():
                for (fi, box, kps) in frames_obs:
                    if tid not in active:
                        at = _Active(internal_id=next_internal_id, tracker_id=tid, first_frame=fi)
                        next_internal_id += 1
                        active[tid] = at
                    at = active[tid]
                    if at.closed:
                        # reopen as new tracklet
                        at = _Active(internal_id=next_internal_id, tracker_id=tid, first_frame=fi)
                        next_internal_id += 1
                        active[tid] = at
                    at.frames.append(fi)
                    at.boxes.append(list(box))
                    at.keypoints.append(kps)
                    at.last_frame = fi
                    # scoring
                    # build frame obs list for this frame from active tracklets observed this frame
                    # We'll score after collecting all tids for this frame; but we need all boxes.
                    pass
            # Now score best snapshots for tracklets observed this frame.
            # Build list of (internal_id, box) for tracklets observed in this frame.
            frame_obs_list = []
            for tid, frames_obs in obs.items():
                at = active.get(tid)
                if at is None or at.closed:
                    continue
                # find the box for this frame (last appended)
                frame_obs_list.append((at.internal_id, list(at.boxes[-1]), at.keypoints[-1] if at.keypoints else []))
            # for each, score
            for internal_id, box, kps in frame_obs_list:
                others = [(oid, obox) for (oid, obox, _) in [(x[0], x[1], None) for x in frame_obs_list] if oid != internal_id]
                # others need boxes
                other_boxes = [obox for (oid, obox, _) in [(x[0], x[1], None) for x in frame_obs_list] if oid != internal_id]
                # reconstruct other_boxes properly
                other_boxes = [x[1] for x in frame_obs_list if x[0] != internal_id]
                score = observation_score(
                    box, kps, w, h, other_boxes,
                    padding=padding, head_weight=head_weight, body_weight=body_weight,
                    edge_penalty=edge_penalty, edge_tolerance=edge_tolerance,
                    distractor_weight=distractor_weight, same_person_iou=same_person_iou,
                )
                # find active at
                at = None
                for a in active.values():
                    if a.internal_id == internal_id:
                        at = a
                        break
                if at is None:
                    continue
                if score > at.best_score:
                    at.best_score = score
                    at.best_box = list(box)
                    at.best_frame = frame_idx
                    at.best_others = [(oid, list(obox)) for (oid, obox, _) in [(x[0], x[1], None) for x in frame_obs_list] if oid != internal_id]
                    # fix best_others to use boxes
                    at.best_others = [(oid, list(obox)) for oid, obox in [(x[0], x[1]) for x in frame_obs_list] if oid != internal_id]

        # update last_seen for active tracklets observed this frame
        observed_ids = set()
        if frame_idx % sample_rate == 0:
            for tid in obs.keys():
                observed_ids.add(tid)
        for tid, at in list(active.items()):
            if at.closed:
                continue
            if tid in observed_ids:
                at.last_frame = frame_idx
            else:
                # dropout close
                if frame_idx - at.last_frame > close_threshold:
                    close_tracklet(active.pop(tid))

        def _gop_start_of(fidx):
            for gs, ge in gops:
                if gs <= fidx < ge:
                    return gs
            return gops[-1][0] if gops else 0

        # process scheduled unblocks (bridging hold)
        if frame_idx in unblock_at:
            for tid in list(unblock_at.pop(frame_idx)):
                with _state_lock:
                    is_classified = tid in classified_ids
                if is_classified:
                    for idx in list(frame_blockers.keys()):
                        if tid in frame_blockers[idx]:
                            frame_blockers[idx].discard(tid)
                    for gs in list(gop_blockers.keys()):
                        gop_blockers[gs].discard(tid)
                else:
                    # reschedule
                    unblock_at.setdefault(frame_idx + 1, set()).add(tid)

        # Process newly classified tracklets (render boxes, bridging, meta).
        try:
            _finalize_classified()
        except Exception:
            pass

        if selective_stream:
            gs = _gop_start_of(frame_idx)
            gop_frames.setdefault(gs, []).append((frame_idx, frame_rgb))
            # track blockers for this GOP
            blockers = gop_blockers.setdefault(gs, set())
            for tid, at in active.items():
                if at.closed:
                    continue
                if at.first_frame <= frame_idx <= at.last_frame + max_gap:
                    blockers.add(at.internal_id)
            # try to ship fully-buffered GOPs in order
            for gstart, gend in gops:
                if gstart in shipped_gops:
                    continue
                if gend > frame_idx + 1:
                    break
                if gop_blockers.get(gstart):
                    break
                # ship this GOP
                frames = gop_frames.pop(gstart, [])
                edited = any(idx in render_boxes for idx, _ in frames)
                seg = _Segment(gstart, gend, edited)
                seg_path = Path(tempfile.mkdtemp(prefix="auto-redact-gop-")) / f"gop_{gstart:06d}.mp4"
                # reuse a single temp dir? use spill_dir
                seg_path = spill_dir / f"gop_{gstart:06d}.ts"
                if edited:
                    _encode_segment(video, seg_path, render_boxes, keyframes, seg, info, crf, preset, encoder)
                else:
                    _copy_segment(video, seg_path, keyframes, seg)
                # Append this GOP to the growing TS file for mid-run playability.
                with open(growing_ts, "ab") as _f:
                    _f.write(seg_path.read_bytes())
                try:
                    seg_path.unlink(missing_ok=True)
                except Exception:
                    pass
                segments.append(seg_path)
                shipped_gops.add(gstart)
        elif not defer_render:
            # blockers for this frame
            blockers = set()
            for tid, at in active.items():
                if at.closed:
                    continue
                if at.first_frame <= frame_idx <= at.last_frame + max_gap:
                    blockers.add(at.internal_id)
            frame_blockers[frame_idx] = blockers

            # store frame
            store_frame(frame_idx, frame_rgb)

            # ship eligible
            while next_ship[0] in frame_buffer:
                idx = next_ship[0]
                if frame_blockers.get(idx):
                    break
                arr = load_frame(idx)
                if arr is None:
                    break
                # paint
                for box in render_boxes.get(idx, []):
                    cv2.rectangle(arr, (box[0], box[1]), (box[2], box[3]), (0, 0, 0), thickness=-1)
                try:
                    proc.stdin.write(arr.tobytes())
                except BrokenPipeError:
                    break
                # remove from buffer
                v = frame_buffer.pop(idx, None)
                if isinstance(v, Path):
                    try:
                        v.unlink(missing_ok=True)
                    except Exception:
                        pass
                in_memory_count[0] = max(0, in_memory_count[0] - (0 if isinstance(v, Path) else 1))
                next_ship[0] += 1

    if _stream_bar_ctx is not None:
        try:
            _stream_bar_ctx.__exit__(None, None, None)
        except Exception:
            pass

    # end: close all remaining active
    for tid in list(active.keys()):
        close_tracklet(active.pop(tid))

    if (not defer_render) and (not selective_stream):
        # ship remaining buffered frames
        while next_ship[0] < info.frame_count:
            idx = next_ship[0]
            if idx in frame_buffer or idx < info.frame_count:
                if idx in frame_buffer:
                    arr = load_frame(idx)
                    if arr is not None:
                        for box in render_boxes.get(idx, []):
                            cv2.rectangle(arr, (box[0], box[1]), (box[2], box[3]), (0, 0, 0), thickness=-1)
                        try:
                            proc.stdin.write(arr.tobytes())
                        except BrokenPipeError:
                            pass
                        v = frame_buffer.pop(idx, None)
                        if isinstance(v, Path):
                            try: v.unlink(missing_ok=True)
                            except Exception: pass
            next_ship[0] += 1

        try:
            proc.stdin.close()
        except Exception:
            pass
        if proc is not None:
            proc.wait()

    # Drain classification queue.
    classify_queue.put(None)
    try:
        _worker_thread.join(timeout=300)
    except Exception:
        pass
    try:
        _finalize_classified()
    except Exception:
        pass

    # release any remaining scheduled unblocks
    for tids in unblock_at.values():
        for tid in tids:
            for idx in list(frame_blockers.keys()):
                if tid in frame_blockers[idx]:
                    frame_blockers[idx].discard(tid)
            for gs in list(gop_blockers.keys()):
                gop_blockers[gs].discard(tid)
    unblock_at.clear()

    if selective_stream:
        # ship any remaining GOPs
        for gstart, gend in gops:
            if gstart in shipped_gops:
                continue
            gop_blockers.get(gstart, set()).clear()
            frames = gop_frames.pop(gstart, [])
            edited = any(idx in render_boxes for idx, _ in frames)
            seg = _Segment(gstart, gend, edited)
            seg_path = spill_dir / f"gop_{gstart:06d}.ts"
            if edited:
                _encode_segment(video, seg_path, render_boxes, keyframes, seg, info, crf, preset, encoder)
            else:
                _copy_segment(video, seg_path, keyframes, seg)
            # Append this GOP to the growing TS file for mid-run playability.
            with open(growing_ts, "ab") as _f:
                _f.write(seg_path.read_bytes())
            try:
                seg_path.unlink(missing_ok=True)
            except Exception:
                pass
            segments.append(seg_path)
            shipped_gops.add(gstart)
        # Remux the growing TS into the final MP4 (faststart for playability).
        import subprocess as _sp
        _sp.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(growing_ts), "-i", str(video),
             "-map", "0:v:0", "-map", "1:a?", "-c", "copy",
             "-movflags", "+faststart", str(out)],
            check=True,
        )
        try:
            growing_ts.unlink(missing_ok=True)
        except Exception:
            pass

    # build metadata
    tracklets = []
    for at in sorted(closed_tracklets, key=lambda a: a.internal_id):
        t = Tracklet(
            track_id=at.internal_id,
            frames=list(at.frames),
            boxes=[list(b) for b in at.boxes],
            keypoints=[list(k) for k in at.keypoints],
            best_frame=best_frame_by_id.get(at.internal_id),
            best_snapshot_path=snapshot_path_by_id.get(at.internal_id),
            isolated_track_ids=list(isolated_by_id.get(at.internal_id, [])),
            redact=redact_by_id.get(at.internal_id),
            score=score_by_id.get(at.internal_id),
        )
        tracklets.append(t)
    meta = Metadata(video=VideoInfo(path=str(video), fps=fps, width=w, height=h, frames=info.frame_count), tracklets=tracklets)
    try:
        save_meta(meta_file, meta)
    except Exception:
        pass

    # Orphan recovery + deferred render (selective / orphans) support.
    if orphans:
        from .track import add_orphan_tracklets

        meta = add_orphan_tracklets(meta, video, on_progress=None)
        from .snapshot import select_snapshots

        select_snapshots(
            meta,
            thumbs_dir=thumbs,
            padding=padding,
            isolate=isolate,
            sub_isolate=sub_isolate,
            closeup_area_fraction=closeup_area_fraction,
            head_weight=head_weight,
            body_weight=body_weight,
            edge_penalty=edge_penalty,
            edge_tolerance=edge_tolerance,
            distractor_weight=distractor_weight,
            same_person_iou=same_person_iou,
            isolate_dilate=isolate_dilate,
            contact_sheet=DEFAULT_CONTACT_SHEET,
            resume=True,
            meta_path=meta_file,
        )
        from .classify import classify_snapshots as _classify

        if classifier == "siglip":
            _classify(meta, prompt, classifier=siglip, resume=True, meta_path=meta_file)
        else:
            _classify(
                meta, prompt, model=model, ollama_url=ollama_url, timeout=timeout,
                workers=classify_workers, think=think, max_image_size=max_image_size,
                resume=True, meta_path=meta_file,
            )
        try:
            save_meta(meta_file, meta)
        except Exception:
            pass

    if defer_render:
        from .render import render_redacted

        render_redacted(
            meta, out, crf=crf, preset=preset, selective=selective, encoder=encoder,
            max_gap=max_gap, smooth_window=smooth_window, bridge_iou=bridge_iou,
            min_track_frames=min_track_frames, on_progress=None,
        )

    # sidecar
    key = {"prompt": prompt.strip(), "model": model}
    if classifier == "siglip":
        key.update(classifier=classifier, threshold=threshold)
    sidecar.write_text(json.dumps(key) + "\n", encoding="utf-8")

    # contact sheet
    try:
        saved = []
        for at in closed_tracklets:
            sp = snapshot_path_by_id.get(at.internal_id)
            if sp and Path(sp).is_file():
                img = Image.open(sp).convert("RGB")
                img.thumbnail((192, 256), Image.Resampling.LANCZOS)
                saved.append((at.internal_id, img))
        saved.sort(key=lambda x: x[0])
        if saved:
            write_contact_sheet(saved, thumbs / DEFAULT_CONTACT_SHEET)
    except Exception:
        pass

    # cleanup spill dir
    try:
        import shutil
        shutil.rmtree(spill_dir, ignore_errors=True)
    except Exception:
        pass
    try:
        ckpt_path.unlink(missing_ok=True)
    except Exception:
        pass

    redacted = sum(1 for t in meta.tracklets if t.redact)
    progress(f"{len(meta.tracklets)} tracklet(s), {redacted} redacted; wrote {out}")
    return {"out": out, "meta_path": meta_file, "tracklets": len(meta.tracklets), "redacted": redacted}
