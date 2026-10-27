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

from pathlib import Path
from typing import Callable

import cv2
from .scene import _frame_hist, split_tracklets_at_cuts

import os
from .config import TrackConfig, resolve
from .detectors import make_detector, make_tracker
from .frames import iter_frames, probe_video
from .geometry import box_iou
from .meta import Metadata, Tracklet, VideoInfo, save_meta
from .protocols import PersonDetector, PersonTracker


# (frame index, bbox, keypoints) per observation, keyed by tracker id.
Observations = dict[int, list[tuple[int, list[float], list[list[float]]]]]


def _track_range(
    video_path: Path,
    tracker: PersonTracker,
    sample_rate: int,
    on_frame: Callable[[], None],
    scene_cuts: bool = False,
    scene_cut_threshold: float = TrackConfig.scene_cut_threshold,
) -> tuple[Observations, list[int]]:
    observations: Observations = {}
    cuts: list[int] = []
    prev_hist = None
    for frame_idx, frame in iter_frames(
        video_path, sample_rate=1, threads=os.cpu_count(),
    ):
        on_frame()
        if scene_cuts:
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            hist = _frame_hist(bgr)
            if prev_hist is not None:
                dist = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
                if dist >= scene_cut_threshold:
                    cuts.append(frame_idx)
            prev_hist = hist
        if sample_rate > 1 and frame_idx % sample_rate != 0:
            continue
        for detection in tracker.update(frame):
            observations.setdefault(detection.track_id, []).append(
                (frame_idx, detection.box, detection.keypoints)
            )
    return observations, cuts


def build_tracklets(
    video_path: str | Path,
    config: TrackConfig | None = None,
    *,
    on_progress: Callable[[int, int], None] | None = None,
    meta_path: str | Path | None = None,
    tracker: PersonTracker | None = None,
    **overrides,
) -> Metadata:
    """Detect and track people through ``video_path`` into tracklets.

    ``tracker`` replaces the default ultralytics YOLO + ByteTrack backend
    (see :mod:`auto_redact.detectors`).

    ``config`` (a :class:`~auto_redact.config.TrackConfig`, with keyword
    ``overrides`` applied on top) sets the stride, detector thresholds,
    tracker and scene-cut handling. With ``meta_path`` the result is also
    written there.
    """
    cfg = resolve(TrackConfig, config, overrides)
    sample_rate = cfg.stride
    scene_cuts = cfg.scene_cuts
    target = Path(meta_path) if meta_path is not None else None

    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool):
        raise ValueError(f"stride must be an integer, got {sample_rate!r}")
    if sample_rate < 1:
        raise ValueError(f"stride must be >= 1, got {sample_rate}")
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"no such video file: {video_path}")

    info = probe_video(video_path)
    if tracker is None:
        tracker = make_tracker(cfg)

    obs_container = {"done": 0}

    def on_frame():
        obs_container["done"] += 1
        if on_progress is not None:
            on_progress(obs_container["done"], max(info.frame_count, obs_container["done"]))

    observations, cuts = _track_range(
        video_path, tracker, sample_rate, on_frame,
        scene_cuts=scene_cuts, scene_cut_threshold=cfg.scene_cut_threshold,
    )

    if on_progress is not None:
        on_progress(info.frame_count, info.frame_count)

    merged = observations


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

    if scene_cuts and cuts:
        metadata, n_splits = split_tracklets_at_cuts(metadata, cuts)
    else:
        n_splits = 0
    metadata.scene_cuts = list(cuts) if scene_cuts else []
    metadata.scene_cut_splits = n_splits

    if target is not None:
        try:
            save_meta(target, metadata)
        except Exception:
            pass

    return metadata


def add_orphan_tracklets(
    meta: Metadata,
    video_path: str | Path,
    *,
    conf: float = 0.2,
    min_iou: float = 0.3,
    min_frames: int = 5,
    max_gap: int = 3,
    on_progress: Callable[[int, int], None] | None = None,
    detector: PersonDetector | None = None,
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

    if detector is None:
        detector = make_detector()

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
        dets = detector.detect(frame, conf)
        existing = existing_by_frame.get(frame_idx, [])
        orphans = [
            d for d in dets
            if all(box_iou(d, e) < min_iou for e in existing)
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
                v = box_iou(last_box, det)
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
