"""Which boxes to paint on which frames.

Turns the tracklets flagged ``redact = true`` into a ``{frame: [boxes]}``
index. Short detection gaps inside a tracklet are interpolated, boxes are
lightly smoothed, and nearby redacted tracklets with overlapping endpoints
are bridged so the black box does not flicker when the detector or tracker
drops. Boxes are clamped to the frame and rounded *outward*, so a sloppy
bbox may hide a pixel of background but never leaks redacted pixels.

Pure functions only: no video, ffmpeg or models are involved.
"""

from __future__ import annotations

from .config import RenderConfig, resolve
from .errors import RenderError
from .geometry import PixelBox, box_iou, lerp_box, to_pixel_box
from .meta import Metadata

FloatBox = list[float]  # x1, y1, x2, y2


def redaction_boxes(
    meta: Metadata,
    config: RenderConfig | None = None,
    **overrides,
) -> dict[int, list[PixelBox]]:
    """Index redacted tracklets' boxes by source frame number.

    Returns ``{frame_idx: [(x1, y1, x2, y2), ...]}`` in pixels, merged
    across tracklets, for tracklets with ``redact = true`` only (false
    and null are left untouched in the output). Boxes are clamped to the
    frame and rounded outward; degenerate boxes (no area after clamping)
    are dropped.

    Tracklets often miss frames when the detector drops out. Gaps of at
    most ``max_gap`` frames inside a tracklet are filled by linear
    interpolation so the black box stays continuous. A centered moving
    average of width ``smooth_window`` (odd; ``1`` disables) settles
    per-frame jitter. When one redacted tracklet ends and another starts
    within ``max_gap`` frames with endpoint IoU of at least
    ``bridge_iou``, the seam between them is interpolated too (ByteTrack
    ID switches); ``bridge_iou <= 0`` disables bridging. Tracklets with
    fewer than ``min_track_frames`` observations are skipped (they are
    usually false merges that flash a huge box); pass ``1`` to keep them.
    Settings come from ``config`` (a :class:`RenderConfig`) with keyword
    ``overrides`` applied on top.
    """
    cfg = resolve(RenderConfig, config, overrides)
    max_gap, smooth_window = cfg.max_gap, cfg.smooth_window
    bridge_iou, min_track_frames = cfg.bridge_iou, cfg.min_track_frames
    if isinstance(max_gap, bool) or not isinstance(max_gap, int) or max_gap < 0:
        raise RenderError(f"max_gap must be an integer >= 0, got {max_gap!r}")
    if (
        isinstance(smooth_window, bool)
        or not isinstance(smooth_window, int)
        or smooth_window < 1
        or smooth_window % 2 == 0
    ):
        raise RenderError(
            f"smooth_window must be an odd integer >= 1, got {smooth_window!r}"
        )
    if isinstance(bridge_iou, bool) or not isinstance(bridge_iou, (int, float)):
        raise RenderError(f"bridge_iou must be a number, got {bridge_iou!r}")
    if (
        isinstance(min_track_frames, bool)
        or not isinstance(min_track_frames, int)
        or min_track_frames < 1
    ):
        raise RenderError(
            f"min_track_frames must be an integer >= 1, got {min_track_frames!r}"
        )

    width, height = meta.video.width, meta.video.height
    boxes: dict[int, list[PixelBox]] = {}
    series: list[tuple[list[int], list[FloatBox]]] = []

    for tracklet in meta.tracklets:
        if tracklet.effective_redact is not True:
            continue
        if len(tracklet.frames) < min_track_frames:
            continue
        filled_frames, filled_boxes = fill_gaps(
            tracklet.frames, tracklet.boxes, max_gap
        )
        if smooth_window > 1 and len(filled_boxes) > 1:
            filled_boxes = smooth_boxes(filled_frames, filled_boxes, smooth_window)
        series.append((filled_frames, filled_boxes))
        for frame_idx, box in zip(filled_frames, filled_boxes):
            pixel_box = to_pixel_box(box, width, height)
            if pixel_box is not None:
                boxes.setdefault(frame_idx, []).append(pixel_box)

    if bridge_iou > 0 and max_gap > 0 and len(series) > 1:
        for frame_idx, box in bridge_seams(series, max_gap, bridge_iou):
            pixel_box = to_pixel_box(box, width, height)
            if pixel_box is not None:
                boxes.setdefault(frame_idx, []).append(pixel_box)

    return boxes


def fill_gaps(
    frames: list[int],
    boxes: list[FloatBox],
    max_gap: int,
) -> tuple[list[int], list[FloatBox]]:
    """Insert linearly interpolated boxes across short missing spans."""
    if not frames:
        return [], []
    out_frames = [frames[0]]
    out_boxes = [list(boxes[0])]
    for prev_frame, prev_box, frame, box in zip(
        frames, boxes, frames[1:], boxes[1:]
    ):
        gap = frame - prev_frame - 1
        if 0 < gap <= max_gap:
            for step in range(1, gap + 1):
                t = step / (gap + 1)
                out_frames.append(prev_frame + step)
                out_boxes.append(lerp_box(prev_box, box, t))
        out_frames.append(frame)
        out_boxes.append(list(box))
    return out_frames, out_boxes


def smooth_boxes(
    frames: list[int],
    boxes: list[FloatBox],
    window: int,
) -> list[FloatBox]:
    """Centered moving average over consecutive-frame runs only.

    A new run starts after any temporal hole so a cut or unfilled gap
    does not pull the average across unrelated poses.
    """
    if len(boxes) <= 1 or window <= 1:
        return [list(b) for b in boxes]

    radius = window // 2
    smoothed: list[FloatBox] = [list(b) for b in boxes]
    run_start = 0
    for i in range(1, len(frames) + 1):
        at_end = i == len(frames)
        broken = not at_end and frames[i] != frames[i - 1] + 1
        if not (at_end or broken):
            continue
        run_end = i  # exclusive
        for j in range(run_start, run_end):
            lo = max(run_start, j - radius)
            hi = min(run_end, j + radius + 1)
            n = hi - lo
            smoothed[j] = [
                sum(boxes[k][c] for k in range(lo, hi)) / n for c in range(4)
            ]
        run_start = i
    return smoothed


def bridge_seams(
    series: list[tuple[list[int], list[FloatBox]]],
    max_gap: int,
    min_iou: float,
) -> list[tuple[int, FloatBox]]:
    """Interpolate short seams between overlapping redacted tracklets."""
    endpoints: list[tuple[int, FloatBox, int, FloatBox]] = []
    for frames, boxes in series:
        if not frames:
            continue
        endpoints.append((frames[0], boxes[0], frames[-1], boxes[-1]))

    extra: list[tuple[int, FloatBox]] = []
    for i, (_a0, _ab0, a_end, a_box) in enumerate(endpoints):
        for j, (b_start, b_box, _b1, _bb1) in enumerate(endpoints):
            if i == j:
                continue
            gap = b_start - a_end - 1
            if not (0 < gap <= max_gap):
                continue
            if box_iou(a_box, b_box) < min_iou:
                continue
            for step in range(1, gap + 1):
                t = step / (gap + 1)
                extra.append((a_end + step, lerp_box(a_box, b_box, t)))
    return extra
