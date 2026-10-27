"""Pure box geometry shared by tracking, snapshots and rendering.

Boxes are ``[x1, y1, x2, y2]`` in pixels. Nothing here touches I/O, so it
is cheap to test and safe to import from anywhere.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

PixelBox = tuple[int, int, int, int]


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection-over-union of two boxes (0 when either is empty)."""
    inter_w = max(0.0, min(float(a[2]), float(b[2])) - max(float(a[0]), float(b[0])))
    inter_h = max(0.0, min(float(a[3]), float(b[3])) - max(float(a[1]), float(b[1])))
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    union = (
        max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
        + max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
        - inter
    )
    return inter / union if union > 0 else 0.0


def lerp_box(a: Sequence[float], b: Sequence[float], t: float) -> list[float]:
    """Linear interpolation between two boxes."""
    return [a[i] + (b[i] - a[i]) * t for i in range(4)]


def clamped_area(box: Sequence[float], width: int, height: int) -> float:
    """Visible area of ``box`` after clamping it to the frame, in px²."""
    x1, y1, x2, y2 = (float(v) for v in box)
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def overlap_fraction(
    box: Sequence[float], other: Sequence[float], width: int, height: int
) -> float:
    """Fraction of ``box``'s visible area covered by ``other``."""
    own = clamped_area(box, width, height)
    if own <= 0:
        return 0.0
    inter = clamped_area(
        [max(box[0], other[0]), max(box[1], other[1]),
         min(box[2], other[2]), min(box[3], other[3])],
        width,
        height,
    )
    return min(1.0, inter / own)


def clamp01(value: float) -> float:
    value = float(value)
    return min(1.0, max(0.0, value)) if math.isfinite(value) else 0.0


def to_pixel_box(box: Sequence[float], width: int, height: int) -> PixelBox | None:
    """Round ``box`` outward to whole pixels inside the frame (None if empty)."""
    x1, y1, x2, y2 = box
    x1 = max(0, min(width, math.floor(x1)))
    y1 = max(0, min(height, math.floor(y1)))
    x2 = max(0, min(width, math.ceil(x2)))
    y2 = max(0, min(height, math.ceil(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)
