"""Scene-cut detection and tracklet splitting.

ByteTrack associates by motion/IoU and will happily continue a ``track_id``
across a hard cut when the next shot happens to have a person in a
similar place. Classification is per tracklet, so the pre-cut person's
``redact`` flag then paints the wrong person after the cut.

:func:`detect_scene_cuts` finds hard cuts via HSV-histogram distance.
:func:`split_tracklets_at_cuts` breaks any tracklet that straddles a cut
into separate tracklets so each shot is classified on its own.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import av
import cv2
import numpy as np

from .meta import Metadata, Tracklet

# Bhattacharyya distance on a 32x32 HSV hist; hard cuts land near 0.5-0.9,
# within-shot motion stays well below ~0.25 on typical drama footage.
DEFAULT_CUT_THRESHOLD = 0.45
_HIST_SIZE = (32, 32)
_HIST_RANGES = [0, 180, 0, 256]
_DETECT_WIDTH = 320


class SceneError(RuntimeError):
    """Raised when scene-cut detection cannot run."""


def detect_scene_cuts(
    video_path: str | Path,
    *,
    threshold: float = DEFAULT_CUT_THRESHOLD,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[int]:
    """Return frame indices that start a new shot (hard cuts only).

    Decodes ``video_path`` once, comparing each frame's HSV histogram to
    the previous. Indices where the Bhattacharyya distance is at least
    ``threshold`` are returned in ascending order. Soft transitions
    (dissolves) are not detected.
    """
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise SceneError(f"threshold must be a number, got {threshold!r}")
    if not 0 < float(threshold) < 1:
        raise SceneError(f"threshold must be in (0, 1), got {threshold}")

    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"no such video file: {path}")

    cuts: list[int] = []
    prev_hist: np.ndarray | None = None
    prev_index: int | None = None
    total = 0

    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            # Best-effort total for progress; some containers omit it.
            if stream.frames and stream.frames > 0:
                total = int(stream.frames)
            time_base = stream.time_base
            rate = stream.average_rate
            index = 0
            for frame in container.decode(video=0):
                if rate is not None and time_base is not None and frame.pts is not None:
                    index = round(float(frame.pts * time_base * rate))
                image = frame.to_ndarray(format="bgr24")
                hist = _frame_hist(image)
                if prev_hist is not None and prev_index is not None:
                    distance = float(
                        cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
                    )
                    if distance >= threshold:
                        cuts.append(index)
                prev_hist = hist
                prev_index = index
                if on_progress is not None:
                    done = index + 1
                    on_progress(done, total if total > 0 else max(done, 1))
                index += 1
    except av.AVError as error:
        raise SceneError(f"failed to decode {path}: {error}") from error

    return cuts


def split_tracklets_at_cuts(
    meta: Metadata,
    cuts: list[int],
) -> tuple[Metadata, int]:
    """Break tracklets that span a scene cut into per-shot pieces.

    Each cut frame is treated as the first frame of a new shot: observations
    with ``frame < cut`` stay on one piece, ``frame >= cut`` on the next.
    The earliest piece keeps the original ``track_id``; later pieces get
    fresh ids. Snapshot/classification fields are kept only on the piece
    that contains ``best_frame`` (when set); every other piece is cleared
    so it can be re-snapshotted and re-classified. Tracklets that do not
    cross any cut are left unchanged.

    Returns ``(updated_meta, n_splits)`` where ``n_splits`` counts how many
    extra tracklets were created.
    """
    if not cuts:
        return meta, 0

    cut_frames = sorted({int(c) for c in cuts})
    next_id = max((t.track_id for t in meta.tracklets), default=0) + 1
    updated: list[Tracklet] = []
    splits = 0

    for tracklet in meta.tracklets:
        pieces = _partition_at_cuts(tracklet, cut_frames)
        if len(pieces) == 1:
            updated.append(tracklet)
            continue
        for i, (frames, boxes, keypoints) in enumerate(pieces):
            keep_id = tracklet.track_id if i == 0 else next_id
            if i > 0:
                next_id += 1
                splits += 1
            updated.append(
                _piece_from(
                    tracklet,
                    track_id=keep_id,
                    frames=frames,
                    boxes=boxes,
                    keypoints=keypoints,
                )
            )

    meta.tracklets = updated
    return meta, splits


def apply_scene_cuts(
    meta: Metadata,
    *,
    threshold: float = DEFAULT_CUT_THRESHOLD,
    cuts: list[int] | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[Metadata, list[int], int]:
    """Detect cuts (unless ``cuts`` is given) and split tracklets across them.

    Returns ``(meta, cuts, n_splits)``.
    """
    if cuts is None:
        cuts = detect_scene_cuts(
            meta.video.path, threshold=threshold, on_progress=on_progress
        )
    meta, n_splits = split_tracklets_at_cuts(meta, cuts)
    return meta, cuts, n_splits


def _frame_hist(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    if width > _DETECT_WIDTH:
        scale = _DETECT_WIDTH / width
        image = cv2.resize(
            image,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, _HIST_SIZE, _HIST_RANGES)
    cv2.normalize(hist, hist)
    return hist


def _partition_at_cuts(
    tracklet: Tracklet,
    cuts: list[int],
) -> list[tuple[list[int], list[list[float]], list[list[list[float]]]]]:
    """Split one tracklet's observations into cut-separated pieces."""
    if not tracklet.frames:
        return [([], [], [])]

    pieces: list[tuple[list[int], list[list[float]], list[list[list[float]]]]] = []
    cur_frames: list[int] = []
    cur_boxes: list[list[float]] = []
    cur_keypoints: list[list[list[float]]] = []
    cut_i = 0

    for frame, box, keypoints in zip(
        tracklet.frames, tracklet.boxes, tracklet.keypoints
    ):
        while cut_i < len(cuts) and frame >= cuts[cut_i]:
            if cur_frames:
                pieces.append((cur_frames, cur_boxes, cur_keypoints))
                cur_frames, cur_boxes, cur_keypoints = [], [], []
            cut_i += 1
        cur_frames.append(frame)
        cur_boxes.append(box)
        cur_keypoints.append(keypoints)

    if cur_frames:
        pieces.append((cur_frames, cur_boxes, cur_keypoints))
    return pieces or [([], [], [])]


def _piece_from(
    source: Tracklet,
    *,
    track_id: int,
    frames: list[int],
    boxes: list[list[float]],
    keypoints: list[list[list[float]]],
) -> Tracklet:
    """Build a piece, keeping classify fields only if ``best_frame`` is in it."""
    keep_classify = (
        source.best_frame is not None and source.best_frame in frames
    )
    return Tracklet(
        track_id=track_id,
        frames=frames,
        boxes=boxes,
        keypoints=keypoints,
        best_frame=source.best_frame if keep_classify else None,
        best_snapshot_path=source.best_snapshot_path if keep_classify else None,
        isolated_track_ids=(
            list(source.isolated_track_ids) if keep_classify else []
        ),
        redact=source.redact if keep_classify else None,
        score=source.score if keep_classify else None,
    )
