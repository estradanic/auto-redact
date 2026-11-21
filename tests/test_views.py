"""Framing-change detection inside a tracklet (pan/zoom without a cut)."""

from auto_redact.meta import Metadata, Tracklet, VideoInfo
from auto_redact.views import (
    ViewTracker,
    region_visibility,
    split_points,
    split_tracklets_at_view_changes,
)


def pose(*regions: str, conf: float = 0.9) -> list[list[float]]:
    """17 COCO keypoints with the named regions seen and the rest not."""
    from auto_redact.views import REGIONS

    kps = [[0.0, 0.0, 0.0] for _ in range(17)]
    for name in regions:
        for i in REGIONS[name]:
            kps[i] = [1.0, 1.0, conf]
    return kps


LOWER = pose("hips", "legs")
UPPER = pose("head", "shoulders")


def test_region_visibility_counts_confident_keypoints_per_region() -> None:
    vis = region_visibility(pose("head", "hips"))
    assert vis == [1.0, 0.0, 1.0, 0.0]  # head, shoulders, hips, legs
    assert region_visibility(pose("head", conf=0.4)) == [0.0, 0.0, 0.0, 0.0]


def test_pan_from_hips_to_face_starts_a_new_view() -> None:
    # Smoothing notices it a few frames late, but the split is placed where
    # the change actually began.
    assert split_points([LOWER] * 30 + [UPPER] * 30) == [30]


def test_steady_framing_never_splits() -> None:
    assert split_points([UPPER] * 100) == []


def test_single_frame_keypoint_dropouts_are_ignored() -> None:
    noisy = [UPPER if i % 7 else LOWER for i in range(100)]
    assert split_points(noisy) == []


def test_one_region_changing_is_not_enough_by_default() -> None:
    assert split_points([pose("head", "shoulders", "hips")] * 30
                        + [pose("head", "shoulders")] * 30) == []
    assert split_points([pose("head", "shoulders", "hips")] * 30
                        + [pose("head", "shoulders")] * 30, min_regions=1) != []


def test_a_view_must_be_old_enough_to_split() -> None:
    tracker = ViewTracker(min_frames=12)
    fired = [tracker.update(LOWER if i < 4 else UPPER) for i in range(40)]
    assert sum(fired) <= 1
    assert not any(fired[:12])


def test_view_after_a_split_becomes_the_reference() -> None:
    assert split_points([LOWER] * 30 + [UPPER] * 40 + [LOWER] * 40) == [30, 70]


def _meta(n_frames: int, change_at: int) -> Metadata:
    keypoints = [LOWER if i < change_at else UPPER for i in range(n_frames)]
    tl = Tracklet(
        track_id=7,
        frames=list(range(n_frames)),
        boxes=[[0.0, 0.0, 10.0, 10.0]] * n_frames,
        keypoints=keypoints,
        best_frame=n_frames - 1,
        best_snapshot_path="t.jpg",
        redact=False,
        score=0.1,
        detail="d",
    )
    return Metadata(video=VideoInfo(path="x", fps=25.0, width=64, height=48, frames=n_frames), tracklets=[tl])


def test_staged_split_keeps_classification_only_on_the_best_frame_piece() -> None:
    meta, splits = split_tracklets_at_view_changes(_meta(80, 30))
    assert splits == 1
    first, second = meta.tracklets
    assert (first.track_id, second.track_id) == (7, 8)
    assert first.frames + second.frames == list(range(80))
    assert first.redact is None and first.best_frame is None  # to be re-snapshotted
    assert second.redact is False and second.best_frame == 79  # best frame lives here


def test_staged_split_merges_a_too_short_tail_back() -> None:
    meta, splits = split_tracklets_at_view_changes(_meta(80, 72))
    assert splits == 0
    assert len(meta.tracklets) == 1 and len(meta.tracklets[0].frames) == 80


def test_cut_detection_histogram_is_unchanged_by_shrinking_first() -> None:
    import cv2
    import numpy as np

    from auto_redact.scene import _frame_hist, rgb_frame_hist

    rng = np.random.default_rng(0)
    for shape in ((1080, 1920, 3), (720, 1280, 3), (240, 320, 3), (200, 301, 3)):
        frame = rng.integers(0, 256, size=shape, dtype=np.uint8)
        old = _frame_hist(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        assert np.array_equal(rgb_frame_hist(frame), old), shape
