"""Tests for auto_redact.snapshot: best-frame selection + thumbnails (#12).

Synthetic clips come from ffmpeg lavfi (no checked-in media). Thumbnail
content is verified against the decoded source frame with a tolerance
(JPEG is lossy); crop geometry is checked exactly via array shape.
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from auto_redact.cli import main
from auto_redact.frames import iter_frames, probe_video
from auto_redact.meta import Metadata, Tracklet, VideoInfo, load_meta, save_meta
from auto_redact.snapshot import (
    CONTACT_SHEET_CELL,
    CONTACT_SHEET_LABEL_HEIGHT,
    SnapshotError,
    best_observation,
    crop_snapshot,
    observation_score,
    redact_others,
    select_snapshots,
    write_contact_sheet,
)

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")

pytestmark = pytest.mark.skipif(
    not (FFMPEG and FFPROBE), reason="ffmpeg/ffprobe not installed"
)

TESTSRC = "testsrc=duration=2:size=320x240:rate=25"  # 50 frames


def make_clip(path: Path, source: str = TESTSRC) -> Path:
    """Render a lavfi test source to an h264 mp4 at ``path``."""
    subprocess.run(
        [FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", source,
         "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )
    return path


def make_tracklet(track_id, frames, boxes) -> Tracklet:
    return Tracklet(
        track_id=track_id,
        frames=list(frames),
        boxes=[list(box) for box in boxes],
        keypoints=[[] for _ in frames],
    )


def make_meta(video: Path, tracklets) -> Metadata:
    info = probe_video(video)
    return Metadata(
        video=VideoInfo(
            path=str(video),
            fps=info.fps,
            width=info.width,
            height=info.height,
            frames=info.frame_count,
        ),
        tracklets=tracklets,
    )


def decode_frame(video: Path, frame_idx: int) -> np.ndarray:
    for idx, frame in iter_frames(video):
        if idx == frame_idx:
            return frame
    raise AssertionError(f"frame {frame_idx} not in {video}")


def load_rgb(path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


# --- best_observation (unit) -------------------------------------------------


def test_best_observation_picks_largest_bbox() -> None:
    tracklet = make_tracklet(
        1,
        frames=[5, 6, 7],
        boxes=[[0, 0, 20, 20], [0, 100, 25, 120], [40, 40, 140, 200]],
    )

    frame_idx, box = best_observation(tracklet, width=320, height=240)

    assert (frame_idx, box) == (7, [40, 40, 140, 200])


def test_best_observation_measures_area_clamped_to_frame() -> None:
    tracklet = make_tracklet(
        1,
        frames=[0, 1],
        boxes=[[300, 200, 400, 300], [10, 10, 60, 60]],  # 20x40 visible
    )

    frame_idx, _ = best_observation(tracklet, width=320, height=240)

    assert frame_idx == 1  # not the raw-larger box sticking out of frame


def test_best_observation_ties_keep_earliest_frame() -> None:
    tracklet = make_tracklet(1, frames=[3, 9], boxes=[[0, 0, 10, 10]] * 2)

    frame_idx, _ = best_observation(tracklet, width=320, height=240)

    assert frame_idx == 3


def test_best_observation_rejects_empty_tracklet() -> None:
    with pytest.raises(SnapshotError, match="tracklet 1 has no observations"):
        best_observation(make_tracklet(1, [], []), width=320, height=240)


def test_best_observation_rejects_misaligned_tracklet() -> None:
    tracklet = Tracklet(track_id=1, frames=[0, 1], boxes=[[0, 0, 10, 10]], keypoints=[])

    with pytest.raises(SnapshotError, match="not aligned"):
        best_observation(tracklet, width=320, height=240)


def test_best_observation_rejects_degenerate_boxes() -> None:
    tracklet = make_tracklet(
        1, frames=[0, 1], boxes=[[10, 10, 10, 50], [80, 80, 40, 90]]
    )

    with pytest.raises(SnapshotError, match="positive area"):
        best_observation(tracklet, width=320, height=240)


def kps(conf, head=None):
    head = conf if head is None else head
    return [[0, 0, head]] * 5 + [[0, 0, conf]] * 12


def test_best_observation_prefers_visible_keypoints_over_bigger_box() -> None:
    tracklet = make_tracklet(
        1, frames=[0, 1], boxes=[[0, 0, 100, 100], [0, 0, 80, 80]]
    )
    tracklet.keypoints = [kps(0.1), kps(0.9)]

    frame_idx, _ = best_observation(tracklet, width=320, height=240)

    assert frame_idx == 1


def test_best_observation_prefers_visible_head() -> None:
    tracklet = make_tracklet(1, frames=[0, 1], boxes=[[0, 0, 100, 100]] * 2)
    tracklet.keypoints = [kps(0.8, head=0.0), kps(0.8, head=0.9)]

    frame_idx, _ = best_observation(tracklet, width=320, height=240)

    assert frame_idx == 1


def test_best_observation_avoids_occluded_frames() -> None:
    tracklet = make_tracklet(
        1, frames=[0, 1], boxes=[[0, 0, 100, 100], [0, 0, 90, 90]]
    )
    blocker = make_tracklet(2, frames=[0], boxes=[[0, 0, 100, 100]])

    frame_idx, _ = best_observation(
        tracklet, width=320, height=240, others=[tracklet, blocker]
    )

    assert frame_idx == 1


def test_observation_score_without_keypoints_is_sqrt_area() -> None:
    assert observation_score([10, 10, 20, 30], [], 320, 240) == pytest.approx(200**0.5)


def test_best_observation_prefers_clear_medium_shot_over_blurry_closeup() -> None:
    tracklet = make_tracklet(
        1, frames=[0, 1], boxes=[[20, 20, 220, 220], [20, 20, 120, 120]]
    )
    tracklet.keypoints = [kps(0.5), kps(0.95)]

    frame_idx, _ = best_observation(tracklet, width=320, height=240)

    assert frame_idx == 1


def test_best_observation_prefers_full_body_view() -> None:
    boxes = [[20, 20, 120, 220]] * 2
    tracklet = make_tracklet(1, frames=[0, 1], boxes=boxes)
    cropped = [[0, 0, 0.9]] * 11 + [[0, 0, 0.05]] * 6  # legs not visible
    tracklet.keypoints = [cropped, kps(0.7)]

    frame_idx, _ = best_observation(tracklet, width=320, height=240)

    assert frame_idx == 1


def test_best_observation_penalises_boxes_touching_frame_edges() -> None:
    tracklet = make_tracklet(
        1, frames=[0, 1], boxes=[[0, 20, 100, 200], [20, 20, 118, 200]]
    )

    frame_idx, _ = best_observation(tracklet, width=320, height=240)

    assert frame_idx == 1


def test_observation_score_deranks_bystanders_in_the_padded_crop() -> None:
    # [100, 50, 200, 150] + 20% padding -> crop window [80, 30, 220, 170];
    # a bystander standing in the left padding strip overlaps the box not
    # at all (no occlusion) but would appear in the thumbnail.
    box = [100.0, 50.0, 200.0, 150.0]
    bystander = [80.0, 30.0, 100.0, 170.0]

    clean = observation_score(box, [], 320, 240, [], padding=0.2)
    dirty = observation_score(box, [], 320, 240, [bystander], padding=0.2)

    assert dirty < clean
    # padding=0 turns the term off: the crop would be the box itself.
    assert observation_score(box, [], 320, 240, [bystander], padding=0.0) == clean


def test_best_observation_prefers_frames_without_bystanders() -> None:
    # Frame 0 has the bigger box, but two bystanders crowd its padded
    # crop (top and bottom padding strips); frame 1 is slightly smaller
    # and clean, so it should win.
    tracklet = make_tracklet(
        1,
        frames=[0, 1],
        boxes=[[60.0, 40.0, 260.0, 200.0], [75.0, 50.0, 265.0, 200.0]],
    )
    bystanders = [
        make_tracklet(2, frames=[0], boxes=[[20.0, 8.0, 300.0, 40.0]]),
        make_tracklet(3, frames=[0], boxes=[[20.0, 200.0, 300.0, 232.0]]),
    ]

    frame_idx, _ = best_observation(
        tracklet, width=320, height=240, others=[tracklet, *bystanders]
    )

    assert frame_idx == 1
    # Control: without the bystanders the bigger frame 0 wins again.
    frame_idx, _ = best_observation(tracklet, width=320, height=240)
    assert frame_idx == 0


# --- crop_snapshot (unit) ----------------------------------------------------


def test_crop_snapshot_pads_and_stays_inside_frame() -> None:
    frame = np.arange(240 * 320 * 3, dtype=np.uint8).reshape(240, 320, 3)

    crop = crop_snapshot(frame, [40.0, 40.0, 140.0, 200.0], padding=0.2)

    # 100x160 box + 20/32px margin per side -> 140x224 window at (20, 8).
    assert crop.shape == (224, 140, 3)
    np.testing.assert_array_equal(crop, frame[8:232, 20:160])


def test_crop_snapshot_clamps_padding_at_frame_edge() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    crop = crop_snapshot(frame, [0.0, 0.0, 50.0, 50.0], padding=0.2)

    assert crop.shape == (60, 60, 3)  # margin runs off two edges, clamped


def test_crop_snapshot_tight_crop_without_padding() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    crop = crop_snapshot(frame, [10.5, 20.5, 30.0, 40.0], padding=0.0)

    assert crop.shape == (20, 20, 3)  # floor/ceil outward: frame[20:40, 10:30]


@pytest.mark.parametrize("padding", [-0.1, float("nan"), float("inf"), True, "0.2"])
def test_crop_snapshot_rejects_invalid_padding(padding) -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    with pytest.raises(SnapshotError, match="padding"):
        crop_snapshot(frame, [10.0, 10.0, 50.0, 50.0], padding=padding)


def test_crop_snapshot_rejects_degenerate_box() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    with pytest.raises(SnapshotError, match="no area"):
        crop_snapshot(frame, [50.0, 10.0, 50.0, 50.0], padding=0.2)


# --- redact_others (unit) ----------------------------------------------------


def test_redact_others_blacks_out_bystanders_inside_the_crop() -> None:
    crop = np.full((100, 100, 3), 200, dtype=np.uint8)
    # The crop covers frame rect [100, 50, 200, 150].
    window = (100, 50, 200, 150)
    target = [120.0, 70.0, 180.0, 130.0]
    bystander = (2, [185.0, 55.0, 240.0, 100.0])  # enters at the right edge

    painted_crop, painted = redact_others(crop, window, target, [bystander])

    assert painted == [2]
    assert painted_crop is crop
    # Dilated box [179, 50, 246, 105] in frame space -> crop [79, 0, 100, 55],
    # but the subject's own region is protected, so only the parts outside the
    # target bbox ([20, 20, 80, 80] in crop coords) are painted.
    assert (crop[0:20, 79:100] == 0).all()
    assert (crop[20:55, 80:100] == 0).all()
    assert (crop[60:90, 10:70] == 200).all()  # rest of the crop untouched


def test_redact_others_skips_the_subject_and_people_outside_the_crop() -> None:
    crop = np.full((100, 100, 3), 200, dtype=np.uint8)
    window = (100, 50, 200, 150)
    target = [120.0, 70.0, 180.0, 130.0]
    fragment = (3, [122.0, 72.0, 178.0, 128.0])  # IoU ~0.87: same person
    outside = (4, [300.0, 50.0, 340.0, 90.0])  # far right of the crop

    _, painted = redact_others(crop, window, target, [fragment, outside])

    assert painted == []
    assert (crop == 200).all()


def test_redact_others_dilates_so_edges_do_not_leak() -> None:
    crop = np.full((40, 40, 3), 200, dtype=np.uint8)
    window = (0, 0, 40, 40)
    target = [0.0, 0.0, 10.0, 10.0]
    bystander = (2, [20.0, 20.0, 30.0, 30.0])

    redact_others(crop, window, target, [bystander], dilate=0.5)

    # Dilated to [15, 15, 35, 35]; one pixel outside stays unpainted.
    assert (crop[15:35, 15:35] == 0).all()
    assert (crop[14, 14] == 200).all()


def test_redact_others_paints_solid_black() -> None:
    # Black, not blur/mosaic: the point is that nothing about the
    # bystander, not even their shirt colour, reaches the classifier.
    crop = np.full((50, 50, 3), 128, dtype=np.uint8)
    window = (0, 0, 50, 50)
    target = [0.0, 0.0, 20.0, 20.0]

    redact_others(crop, window, target, [(9, [20.0, 20.0, 50.0, 50.0])])

    assert (crop[20:50, 17:50] == 0).all()  # dilated by 3 px
    assert (crop[:20, :20] == 128).all()  # the subject's own crop area is protected


# --- select_snapshots (integration) ------------------------------------------


@pytest.fixture
def clip(tmp_path: Path) -> Path:
    return make_clip(tmp_path / "clip.mp4")


@pytest.fixture
def two_person_meta(clip: Path) -> Metadata:
    # Track 1 grows toward frame 40 (its largest bbox), track 2's largest
    # bbox is its first observation at frame 10.
    return make_meta(
        clip,
        [
            make_tracklet(
                1,
                frames=[0, 25, 40],
                boxes=[[0, 0, 20, 20], [0, 100, 25, 120], [40, 40, 140, 200]],
            ),
            make_tracklet(2, frames=[10, 30], boxes=[[200, 20, 300, 120], [0, 0, 50, 50]]),
        ],
    )


@pytest.fixture
def shared_frame_meta(clip: Path) -> Metadata:
    # Track 1's only observation is frame 40; track 2 also has an
    # observation there ([150, 20, 210, 80], inside track 1's padded
    # crop window [20, 8, 160, 232]) but its own best frame is 10.
    return make_meta(
        clip,
        [
            make_tracklet(1, frames=[40], boxes=[[40, 40, 140, 200]]),
            make_tracklet(
                2, frames=[40, 10], boxes=[[150, 20, 210, 80], [0, 0, 80, 100]]
            ),
        ],
    )


def test_select_snapshots_isolates_bystanders(
    shared_frame_meta: Metadata, tmp_path: Path
) -> None:
    thumbs = tmp_path / "thumbs"

    select_snapshots(shared_frame_meta, thumbs_dir=thumbs)

    track1, track2 = shared_frame_meta.tracklets
    assert track1.best_frame == 40
    assert track1.isolated_track_ids == [2]
    thumb = load_rgb(thumbs / "track_1.jpg")
    assert thumb.shape == (224, 140, 3)
    # Track 2's dilated box [144, 14, 216, 86] clamps to crop [124:140, 6:66].
    assert thumb[6:66, 124:140].mean() < 8
    assert thumb[60:200, 0:100].mean() > 8  # the subject is untouched
    # Track 2's own best frame (10) has nobody else in its crop.
    assert track2.best_frame == 10
    assert track2.isolated_track_ids == []


def test_select_snapshots_no_isolate_keeps_raw_crops(
    shared_frame_meta: Metadata, tmp_path: Path
) -> None:
    thumbs = tmp_path / "thumbs"

    select_snapshots(shared_frame_meta, thumbs_dir=thumbs, isolate=False)

    for tracklet in shared_frame_meta.tracklets:
        assert tracklet.isolated_track_ids == []
    expected = decode_frame(Path(shared_frame_meta.video.path), 40)[8:232, 20:160]
    thumb = load_rgb(thumbs / "track_1.jpg")
    assert np.abs(thumb.astype(int) - expected.astype(int)).mean() < 8


def test_select_snapshots_writes_one_thumb_per_tracklet(
    two_person_meta: Metadata, tmp_path: Path
) -> None:
    thumbs = tmp_path / "thumbs"

    returned = select_snapshots(two_person_meta, thumbs_dir=thumbs)

    assert returned is two_person_meta
    assert sorted(p.name for p in thumbs.glob("*.jpg")) == [
        "contact_sheet.jpg",
        "track_1.jpg",
        "track_2.jpg",
    ]
    for tracklet in two_person_meta.tracklets:
        assert tracklet.best_frame == {1: 40, 2: 10}[tracklet.track_id]
        assert tracklet.best_snapshot_path == str(thumbs / f"track_{tracklet.track_id}.jpg")
        assert Path(tracklet.best_snapshot_path).is_file()


def test_select_snapshots_crops_the_best_frame_with_padding(
    two_person_meta: Metadata, tmp_path: Path
) -> None:
    thumbs = tmp_path / "thumbs"

    select_snapshots(two_person_meta, thumbs_dir=thumbs, padding=0.2)

    # Padded crop of [40, 40, 140, 200] at 20% -> frame[8:232, 20:160].
    expected = decode_frame(Path(two_person_meta.video.path), 40)[8:232, 20:160]
    thumb = load_rgb(thumbs / "track_1.jpg")
    assert thumb.shape == expected.shape
    # JPEG + yuv420p round trip: close, not bit-exact.
    assert np.abs(thumb.astype(int) - expected.astype(int)).mean() < 8


def test_select_snapshots_writes_contact_sheet(
    two_person_meta: Metadata, tmp_path: Path
) -> None:
    thumbs = tmp_path / "thumbs"

    select_snapshots(two_person_meta, thumbs_dir=thumbs)

    with Image.open(thumbs / "contact_sheet.jpg") as sheet:
        cell_w, cell_h = CONTACT_SHEET_CELL
        # 2 thumbnails -> 2 columns, 1 row of cell + label band.
        assert sheet.size == (2 * cell_w, cell_h + CONTACT_SHEET_LABEL_HEIGHT)


def test_select_snapshots_can_skip_contact_sheet(
    two_person_meta: Metadata, tmp_path: Path
) -> None:
    thumbs = tmp_path / "thumbs"

    select_snapshots(two_person_meta, thumbs_dir=thumbs, contact_sheet=None)

    assert not (thumbs / "contact_sheet.jpg").exists()
    assert (thumbs / "track_1.jpg").is_file()


def test_select_snapshots_without_tracklets_writes_nothing(
    clip: Path, tmp_path: Path
) -> None:
    thumbs = tmp_path / "thumbs"

    select_snapshots(make_meta(clip, []), thumbs_dir=thumbs)

    assert not thumbs.exists()


def test_select_snapshots_missing_video_raises(two_person_meta, tmp_path: Path) -> None:
    two_person_meta.video.path = str(tmp_path / "nope.mp4")

    with pytest.raises(FileNotFoundError, match="no such video file"):
        select_snapshots(two_person_meta, thumbs_dir=tmp_path / "thumbs")


def test_select_snapshots_unknown_frame_raises(clip: Path, tmp_path: Path) -> None:
    meta = make_meta(clip, [make_tracklet(1, [999], [[10, 10, 100, 100]])])

    with pytest.raises(SnapshotError, match=r"frame\(s\) \[999\] not found"):
        select_snapshots(meta, thumbs_dir=tmp_path / "thumbs")


@pytest.mark.parametrize("padding", [-0.5, float("nan"), True])
def test_select_snapshots_rejects_invalid_padding(
    two_person_meta, tmp_path: Path, padding
) -> None:
    with pytest.raises(SnapshotError, match="padding"):
        select_snapshots(
            two_person_meta, thumbs_dir=tmp_path / "thumbs", padding=padding
        )


def test_select_snapshots_rejects_tracklet_without_observations(
    clip: Path, tmp_path: Path
) -> None:
    meta = make_meta(clip, [make_tracklet(7, [], [])])

    with pytest.raises(SnapshotError, match="tracklet 7 has no observations"):
        select_snapshots(meta, thumbs_dir=tmp_path / "thumbs")


def test_write_contact_sheet_requires_thumbnails(tmp_path: Path) -> None:
    with pytest.raises(SnapshotError, match="no thumbnails"):
        write_contact_sheet([], tmp_path / "sheet.jpg")


def test_write_contact_sheet_downscales_large_thumbnails(tmp_path: Path) -> None:
    cell_w, cell_h = CONTACT_SHEET_CELL
    big = Image.new("RGB", (cell_w * 3, cell_h * 3), (200, 30, 30))

    out = write_contact_sheet([(1, big)], tmp_path / "sheet.png")

    with Image.open(out) as sheet:
        assert sheet.size == (cell_w, cell_h + CONTACT_SHEET_LABEL_HEIGHT)


# --- CLI --------------------------------------------------------------------


def test_cli_snapshot_updates_meta_and_writes_thumbs(
    two_person_meta: Metadata, tmp_path: Path
) -> None:
    meta_path = tmp_path / "meta.json"
    save_meta(meta_path, two_person_meta)
    thumbs = tmp_path / "thumbs"

    assert main(["snapshot", "--meta", str(meta_path), "--thumbs", str(thumbs)]) == 0

    reloaded = load_meta(meta_path)
    assert [t.best_frame for t in reloaded.tracklets] == [40, 10]
    assert reloaded.tracklets[0].best_snapshot_path == str(thumbs / "track_1.jpg")
    assert (thumbs / "track_2.jpg").is_file()
    assert (thumbs / "contact_sheet.jpg").is_file()


def test_cli_snapshot_no_contact_sheet(
    two_person_meta: Metadata, tmp_path: Path
) -> None:
    meta_path = tmp_path / "meta.json"
    save_meta(meta_path, two_person_meta)
    thumbs = tmp_path / "thumbs"

    result = main(
        ["snapshot", "--meta", str(meta_path), "--thumbs", str(thumbs),
         "--no-contact-sheet"]
    )

    assert result == 0
    assert (thumbs / "track_1.jpg").is_file()
    assert not (thumbs / "contact_sheet.jpg").exists()


def test_cli_snapshot_no_isolate(
    shared_frame_meta: Metadata, tmp_path: Path
) -> None:
    meta_path = tmp_path / "meta.json"
    save_meta(meta_path, shared_frame_meta)
    thumbs = tmp_path / "thumbs"

    result = main(
        ["snapshot", "--meta", str(meta_path), "--thumbs", str(thumbs),
         "--no-isolate"]
    )

    assert result == 0
    reloaded = load_meta(meta_path)
    assert all(t.isolated_track_ids == [] for t in reloaded.tracklets)


def test_cli_snapshot_reports_errors(tmp_path: Path) -> None:
    result = main(
        ["snapshot", "--meta", str(tmp_path / "nope.json"), "--thumbs", "thumbs"]
    )

    assert result == 1


def test_cli_snapshot_requires_meta() -> None:
    with pytest.raises(SystemExit):
        main(["snapshot"])
