"""Framing-change detection inside a tracklet.

A tracklet gets one best snapshot and one verdict. When the camera pans or
zooms *within* a shot (no hard cut), the same tracklet can show very different
parts of a person: hips and thighs first, head and shoulders after the pan.
The snapshot scorer then picks one of them and the other is judged by proxy.

:class:`ViewTracker` watches which body regions the pose model can see
(head, shoulders, hips, legs) and says when that set changes, so the tracklet
can be split into per-view pieces, each with its own snapshot and verdict.
Pure and model-free; the staged and streaming pipelines both use it.
"""

from __future__ import annotations

from typing import Sequence

# COCO-17 keypoint indices per body region.
REGIONS: dict[str, tuple[int, ...]] = {
    "head": (0, 1, 2, 3, 4),
    "shoulders": (5, 6),
    "hips": (11, 12),
    "legs": (13, 14, 15, 16),
}
_NAMES = tuple(REGIONS)

KEYPOINT_CONF = 0.5  # a keypoint counts as seen at or above this confidence
SMOOTHING = 0.3      # EMA weight of the newest observation
ON, OFF = 0.65, 0.35  # hysteresis on a region's smoothed visibility
MAX_LOOKBACK = 8      # a split may start at most this many observations back


def region_visibility(keypoints: Sequence[Sequence[float]]) -> list[float]:
    """Fraction of each region's keypoints seen with enough confidence."""
    out = []
    for name in _NAMES:
        idx = [i for i in REGIONS[name] if i < len(keypoints)]
        seen = sum(1 for i in idx if keypoints[i][2] >= KEYPOINT_CONF)
        out.append(seen / len(idx) if idx else 0.0)
    return out


class ViewTracker:
    """Online detector: feed one tracklet's observations in order.

    ``update`` returns 0 normally. When at least ``min_regions`` body regions
    have appeared or disappeared relative to the current view (and the old
    view lasted at least ``min_frames`` observations) it returns how many of
    the latest observations, this one included, belong to the new view; the
    split goes just before them. Smoothing and hysteresis keep single-frame
    keypoint dropouts from counting, and the lookback finds where the change
    actually started rather than when it was noticed (at most
    ``MAX_LOOKBACK`` back). The new view then becomes the reference.
    """

    def __init__(self, min_frames: int = 12, min_regions: int = 2) -> None:
        self.min_frames = max(1, int(min_frames))
        self.min_regions = max(1, int(min_regions))
        self._ema: list[float] | None = None
        self._present: list[bool] = []
        self._reference: list[bool] = []
        self._age = 0
        self._off_reference = 0  # trailing observations that disagree with the reference

    @property
    def view(self) -> tuple[str, ...]:
        """Names of the regions currently in view."""
        return tuple(n for n, p in zip(_NAMES, self._present) if p)

    def update(self, keypoints: Sequence[Sequence[float]]) -> int:
        vis = region_visibility(keypoints)
        if self._ema is None:
            self._ema = list(vis)
            self._present = [v >= 0.5 for v in vis]
            self._reference = list(self._present)
            self._age = 1
            return 0
        for i, v in enumerate(vis):
            self._ema[i] += SMOOTHING * (v - self._ema[i])
            if self._ema[i] > ON:
                self._present[i] = True
            elif self._ema[i] < OFF:
                self._present[i] = False
        self._age += 1
        if self._age <= self.min_frames:
            # Still settling: the reference follows the evidence.
            self._reference = list(self._present)
            self._off_reference = 0
            return 0
        raw_differs = sum(1 for v, r in zip(vis, self._reference) if (v >= 0.5) != r)
        self._off_reference = self._off_reference + 1 if raw_differs >= self.min_regions else 0
        changed = sum(1 for a, b in zip(self._present, self._reference) if a != b)
        if changed < self.min_regions:
            return 0
        new_view = max(1, min(self._off_reference, MAX_LOOKBACK))
        if self._age - new_view < self.min_frames:
            return 0  # the old view was too short to stand on its own
        self._reference = list(self._present)
        self._age = new_view
        self._off_reference = 0
        return new_view


def split_points(
    keypoints: Sequence[Sequence[Sequence[float]]],
    min_frames: int = 12,
    min_regions: int = 2,
) -> list[int]:
    """Indices (into the observation list) where a new view begins."""
    tracker = ViewTracker(min_frames, min_regions)
    points = []
    for i, kps in enumerate(keypoints):
        n = tracker.update(kps)
        if n:
            points.append(i - n + 1)
    return points


def split_tracklets_at_view_changes(
    meta,
    min_frames: int = 12,
    min_regions: int = 2,
) -> tuple[object, int]:
    """Break tracklets into one piece per view (staged pipeline).

    Mirrors :func:`auto_redact.scene.split_tracklets_at_cuts`: the first piece
    keeps the tracklet's id, later pieces get fresh ids, and classify fields
    stay only on the piece holding ``best_frame``. A tail shorter than
    ``min_frames`` is merged back into the piece before it (a few leftover
    frames would be dropped by the renderer, leaking them).

    Returns ``(meta, n_splits)``.
    """
    from .scene import _piece_from

    next_id = max((t.track_id for t in meta.tracklets), default=0) + 1
    updated, splits = [], 0
    for tracklet in meta.tracklets:
        points = split_points(tracklet.keypoints, min_frames, min_regions)
        while points and len(tracklet.frames) - points[-1] < min_frames:
            points.pop()
        if not points:
            updated.append(tracklet)
            continue
        bounds = [0, *points, len(tracklet.frames)]
        for i, (a, b) in enumerate(zip(bounds, bounds[1:])):
            keep_id = tracklet.track_id if i == 0 else next_id
            if i > 0:
                next_id += 1
                splits += 1
            updated.append(
                _piece_from(
                    tracklet,
                    track_id=keep_id,
                    frames=tracklet.frames[a:b],
                    boxes=tracklet.boxes[a:b],
                    keypoints=tracklet.keypoints[a:b],
                )
            )
    meta.tracklets = updated
    return meta, splits
