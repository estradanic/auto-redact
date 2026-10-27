"""Tests for auto_redact.render (issue #15).

Synthetic clips come from ffmpeg lavfi (no checked-in media). Boxes are
verified pixel-wise: solid-color sources make "black inside the box,
original color outside" a crisp assertion, with a 2px inset to absorb
yuv420p chroma subsampling at edges.
"""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from auto_redact.cli import main
from auto_redact.frames import iter_frames, probe_video
from auto_redact.meta import Metadata, Tracklet, VideoInfo, save_meta
from auto_redact.boxes import fill_gaps, redaction_boxes, smooth_boxes
from auto_redact.render import RenderError, render_redacted
from auto_redact.segments import Segment, build_segments, keyframe_table

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")

pytestmark = pytest.mark.skipif(
    not (FFMPEG and FFPROBE), reason="ffmpeg/ffprobe not installed"
)

RED = "color=c=red:size=64x48:rate=10:duration=2"  # 20 frames
TESTSRC = "testsrc=duration=2:size=320x240:rate=25"  # 50 frames


def make_clip(path: Path, source: str = TESTSRC, audio: bool = False, gop: int | None = None) -> Path:
    """Render a lavfi test source to an h264 mp4, optionally with AAC audio."""
    command = [FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", source]
    if audio:
        duration = source.split("duration=")[1].split(":")[0]
        command += ["-f", "lavfi", "-i", f"sine=duration={duration}:frequency=440"]
    if gop is not None:
        command += ["-g", str(gop)]
    command += ["-pix_fmt", "yuv420p"]
    if audio:
        command += ["-c:a", "aac"]
    command += [str(path)]
    subprocess.run(command, check=True)
    return path


def make_tracklet(track_id, frames, boxes, redact) -> Tracklet:
    return Tracklet(
        track_id=track_id,
        frames=frames,
        boxes=boxes,
        keypoints=[[] for _ in frames],
        redact=redact,
    )


def make_meta(video: Path, tracklets, width=None, height=None, fps=25.0) -> Metadata:
    info = probe_video(video)
    return Metadata(
        video=VideoInfo(
            path=str(video),
            fps=fps,
            width=width if width is not None else info.width,
            height=height if height is not None else info.height,
            frames=info.frame_count,
        ),
        tracklets=tracklets,
    )


def probe_streams(path: Path) -> list[dict]:
    proc = subprocess.run(
        [FFPROBE, "-v", "error", "-print_format", "json", "-show_streams", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout)["streams"]


@pytest.fixture
def red_clip(tmp_path: Path) -> Path:
    return make_clip(tmp_path / "red.mp4", source=RED)


@pytest.fixture
def clip(tmp_path: Path) -> Path:
    return make_clip(tmp_path / "clip.mp4")


@pytest.fixture
def clip_with_audio(tmp_path: Path) -> Path:
    return make_clip(tmp_path / "clip_audio.mp4", audio=True)


@pytest.fixture
def gop_clip(tmp_path: Path) -> Path:
    """320x240 testsrc, 50 frames, keyframes every 10 frames (0,10,20,30,40)."""
    return make_clip(tmp_path / "gop.mp4", gop=10)


@pytest.fixture
def gop_red_clip(tmp_path: Path) -> Path:
    """Solid red, 50 frames, keyframes every 10 frames."""
    source = "color=c=red:size=64x48:rate=25:duration=2"
    return make_clip(tmp_path / "gop_red.mp4", source=source, gop=10)


@pytest.fixture
def gop_audio_clip(tmp_path: Path) -> Path:
    return make_clip(tmp_path / "gop_audio.mp4", audio=True, gop=10)


# --- redaction_boxes (unit) -------------------------------------------------


def test_redaction_boxes_indexes_only_redacted_tracklets(red_clip: Path) -> None:
    meta = make_meta(
        red_clip,
        [
            make_tracklet(1, [0, 1], [[10, 10, 20, 20], [30.2, 30.8, 40.1, 40.9]], True),
            make_tracklet(2, [0], [[1, 1, 5, 5]], False),
            make_tracklet(3, [0, 1], [[2, 2, 6, 6], [3, 3, 7, 7]], None),
        ],
    )

    # Floats round outward; false/null tracklets are excluded.
    # Disable temporal fill/smooth/min-length so this stays a pure index check.
    assert redaction_boxes(
        meta, max_gap=0, smooth_window=1, bridge_iou=0, min_track_frames=1
    ) == {
        0: [(10, 10, 20, 20)],
        1: [(30, 30, 41, 41)],
    }


def test_redaction_boxes_merges_tracklets_per_frame(red_clip: Path) -> None:
    meta = make_meta(
        red_clip,
        [
            make_tracklet(1, [0], [[10, 10, 20, 20]], True),
            make_tracklet(2, [0, 1], [[30, 30, 40, 40], [4, 4, 8, 8]], True),
        ],
    )

    assert redaction_boxes(
        meta, max_gap=0, smooth_window=1, bridge_iou=0, min_track_frames=1
    ) == {
        0: [(10, 10, 20, 20), (30, 30, 40, 40)],
        1: [(4, 4, 8, 8)],
    }


def test_redaction_boxes_clamps_and_drops_degenerate(red_clip: Path) -> None:
    meta = make_meta(
        red_clip,
        [
            make_tracklet(
                1,
                [0, 1, 2, 3],
                [
                    [-5, -5, 10, 10],  # partially outside -> clamped
                    [60, 10, 100, 20],  # past the right edge -> clamped
                    [500, 500, 600, 600],  # fully outside -> dropped
                    [20, 20, 10, 10],  # reversed -> dropped
                ],
                True,
            ),
        ],
    )

    assert redaction_boxes(
        meta, max_gap=0, smooth_window=1, bridge_iou=0, min_track_frames=1
    ) == {
        0: [(0, 0, 10, 10)],
        1: [(60, 10, 64, 20)],
    }


def test_fill_tracklet_gaps_interpolates_short_holes() -> None:
    frames, boxes = fill_gaps(
        [0, 3],
        [[0.0, 0.0, 10.0, 10.0], [30.0, 0.0, 40.0, 10.0]],
        max_gap=5,
    )
    assert frames == [0, 1, 2, 3]
    assert boxes[1] == pytest.approx([10.0, 0.0, 20.0, 10.0])
    assert boxes[2] == pytest.approx([20.0, 0.0, 30.0, 10.0])


def test_fill_tracklet_gaps_skips_holes_over_max_gap() -> None:
    frames, boxes = fill_gaps(
        [0, 5],
        [[0.0, 0.0, 10.0, 10.0], [50.0, 0.0, 60.0, 10.0]],
        max_gap=2,
    )
    assert frames == [0, 5]
    assert len(boxes) == 2


def test_smooth_boxes_averages_within_contiguous_runs() -> None:
    frames = [0, 1, 2]
    boxes = [
        [0.0, 0.0, 10.0, 10.0],
        [30.0, 0.0, 40.0, 10.0],
        [60.0, 0.0, 70.0, 10.0],
    ]
    smoothed = smooth_boxes(frames, boxes, window=3)
    assert smoothed[1] == pytest.approx([30.0, 0.0, 40.0, 10.0])
    # Endpoints average with their one neighbour only.
    assert smoothed[0] == pytest.approx([15.0, 0.0, 25.0, 10.0])


def test_redaction_boxes_fills_gaps_and_bridges_id_switch(red_clip: Path) -> None:
    # Track A ends at frame 2; track B starts at 5 with overlapping boxes.
    # Gap of 2 frames should be bridged; within-track gap of 1 should fill.
    meta = make_meta(
        red_clip,
        [
            make_tracklet(
                1,
                [0, 2],
                [[10.0, 10.0, 30.0, 30.0], [10.0, 10.0, 30.0, 30.0]],
                True,
            ),
            make_tracklet(
                2,
                [5, 6],
                [[12.0, 12.0, 32.0, 32.0], [12.0, 12.0, 32.0, 32.0]],
                True,
            ),
        ],
    )

    boxes = redaction_boxes(
        meta, max_gap=5, smooth_window=1, bridge_iou=0.3, min_track_frames=1
    )
    assert set(boxes) == {0, 1, 2, 3, 4, 5, 6}
    # Mid-gap fill inside track 1.
    assert boxes[1] == [(10, 10, 30, 30)]
    # Bridged seam between tracks.
    assert len(boxes[3]) == 1
    assert len(boxes[4]) == 1


def test_redaction_boxes_skips_short_tracklets(red_clip: Path) -> None:
    meta = make_meta(
        red_clip,
        [
            make_tracklet(1, [0, 1], [[10, 10, 20, 20], [10, 10, 20, 20]], True),
            make_tracklet(
                2,
                list(range(10)),
                [[30, 30, 40, 40]] * 10,
                True,
            ),
        ],
    )
    boxes = redaction_boxes(
        meta, max_gap=0, smooth_window=1, bridge_iou=0, min_track_frames=5
    )
    assert boxes[0] == [(30, 30, 40, 40)]
    assert 1 in boxes and boxes[1] == [(30, 30, 40, 40)]


def test_redaction_boxes_rejects_bad_stability_args(red_clip: Path) -> None:
    meta = make_meta(red_clip, [])
    with pytest.raises(RenderError, match="max_gap"):
        redaction_boxes(meta, max_gap=-1)
    with pytest.raises(RenderError, match="smooth_window"):
        redaction_boxes(meta, smooth_window=4)
    with pytest.raises(RenderError, match="min_track_frames"):
        redaction_boxes(meta, min_track_frames=0)


# --- render_redacted (integration) ------------------------------------------


def test_render_blacks_out_redacted_boxes(red_clip: Path, tmp_path: Path) -> None:
    frames = list(range(10))  # redact only the first second
    meta = make_meta(
        red_clip,
        [
            make_tracklet(1, frames, [[10, 10, 30, 30]] * len(frames), True),
            make_tracklet(2, frames, [[35, 35, 50, 50]] * len(frames), False),
        ],
    )
    out = tmp_path / "out.mp4"

    render_redacted(meta, out)

    decoded = list(iter_frames(out))
    assert len(decoded) == 20
    assert decoded[0][1].shape == (48, 64, 3)
    redacted_frame = decoded[0][1]
    untouched_frame = decoded[15][1]
    # Inside the redacted box: black. Elsewhere the red source survives
    # (checked on the red channel; green/blue are ~0 in a red frame).
    assert redacted_frame[12:28, 12:28].max() < 10
    assert redacted_frame[37:47, 37:48, 0].min() > 200
    assert redacted_frame[2:10, 40:60, 0].min() > 200
    assert untouched_frame[12:28, 12:28, 0].min() > 200


def test_render_preserves_audio(clip_with_audio: Path, tmp_path: Path) -> None:
    meta = make_meta(clip_with_audio, tracklets=[])
    out = tmp_path / "out.mp4"

    render_redacted(meta, out)

    source_streams = probe_streams(clip_with_audio)
    out_streams = probe_streams(out)
    source_audio = next(s for s in source_streams if s["codec_type"] == "audio")
    out_audio = next(s for s in out_streams if s["codec_type"] == "audio")
    # -c:a copy: same codec, sample rate and channels as the source.
    assert out_audio["codec_name"] == source_audio["codec_name"] == "aac"
    assert out_audio["sample_rate"] == source_audio["sample_rate"]
    assert out_audio["channels"] == source_audio["channels"]
    assert probe_video(out).frame_count == 50


def test_render_video_only_input_has_no_audio(clip: Path, tmp_path: Path) -> None:
    meta = make_meta(clip, tracklets=[])

    out = render_redacted(meta, tmp_path / "out.mp4")

    streams = probe_streams(out)
    assert [s["codec_type"] for s in streams] == ["video"]
    assert probe_video(out).frame_count == 50


def test_render_missing_source_raises(red_clip: Path, tmp_path: Path) -> None:
    meta = make_meta(red_clip, tracklets=[])
    meta.video.path = str(tmp_path / "nope.mp4")

    with pytest.raises(FileNotFoundError, match="no such video file"):
        render_redacted(meta, tmp_path / "out.mp4")


def test_render_dimension_mismatch_raises(red_clip: Path, tmp_path: Path) -> None:
    meta = make_meta(red_clip, tracklets=[], width=640, height=480)

    with pytest.raises(RenderError, match="dimensions disagree"):
        render_redacted(meta, tmp_path / "out.mp4")


@pytest.mark.parametrize("crf", [99, -1, 2.5, True])
def test_render_invalid_crf_raises(red_clip: Path, tmp_path: Path, crf) -> None:
    meta = make_meta(red_clip, tracklets=[])

    with pytest.raises(RenderError, match="crf"):
        render_redacted(meta, tmp_path / "out.mp4", crf=crf)


# --- CLI --------------------------------------------------------------------


def test_cli_render_writes_output(red_clip: Path, tmp_path: Path) -> None:
    frames = list(range(20))
    meta = make_meta(
        red_clip, [make_tracklet(1, frames, [[10, 10, 30, 30]] * len(frames), True)]
    )
    meta_path = tmp_path / "meta.json"
    save_meta(meta_path, meta)
    out = tmp_path / "out.mp4"

    assert main(["render", "--meta", str(meta_path), "--out", str(out)]) == 0

    assert out.is_file()
    assert probe_video(out).frame_count == 20


def test_cli_render_reports_errors(red_clip: Path, tmp_path: Path) -> None:
    result = main(
        ["render", "--meta", str(tmp_path / "nope.json"), "--out", "out.mp4"]
    )

    assert result == 1


def test_cli_requires_a_command() -> None:
    with pytest.raises(SystemExit):
        main([])


def test_cli_render_requires_meta_and_out() -> None:
    with pytest.raises(SystemExit):
        main(["render"])


# --- selective rendering (issue #18) ----------------------------------------


def test_keyframe_table_reports_keyframe_indices(gop_clip: Path) -> None:
    keyframes, _ = keyframe_table(gop_clip)
    assert sorted(keyframes) == [0, 10, 20, 30, 40]


def test_build_segments_aligns_to_keyframes_and_merges() -> None:
    kf = {0: 0, 10: 1, 20: 2, 30: 3, 40: 4}
    # A redaction inside one GOP re-encodes just that GOP; the rest is copied
    # and adjacent copy chunks merge.
    assert build_segments({12, 13}, kf, 50) == [
        Segment(0, 10, False),
        Segment(10, 20, True),
        Segment(20, 50, False),
    ]
    # A redaction spanning two GOPs merges them into one edited chunk.
    assert build_segments({15, 25}, kf, 50) == [
        Segment(0, 10, False),
        Segment(10, 30, True),
        Segment(30, 50, False),
    ]
    # No redactions -> a single copied chunk; redaction at frame 0 edits the
    # leading GOP.
    assert build_segments(set(), kf, 50) == [Segment(0, 50, False)]
    assert build_segments({0}, kf, 50) == [
        Segment(0, 10, True),
        Segment(10, 50, False),
    ]


def test_selective_blacks_out_redactions_and_copies_rest(
    gop_red_clip: Path, tmp_path: Path
) -> None:
    redacted = [10, 11, 12]
    meta = make_meta(
        gop_red_clip,
        [make_tracklet(1, redacted, [[10, 10, 30, 30]] * len(redacted), True)],
    )
    source = list(iter_frames(gop_red_clip))
    out = render_redacted(
        meta, tmp_path / "sel.mp4", selective=True, min_track_frames=1
    )
    rendered = dict(iter_frames(out))

    assert len(rendered) == len(source) == 50
    # Redacted frames have a black box...
    assert rendered[10][12:28, 12:28].max() < 10
    assert rendered[11][12:28, 12:28].max() < 10
    assert rendered[12][12:28, 12:28].max() < 10
    # ...and untouched frames outside the edited GOP [10, 20) are bit-identical
    # to the source (they were stream-copied, not re-encoded).
    for index in (0, 5, 25, 30, 49):
        np.testing.assert_array_equal(rendered[index], source[index][1])


def test_selective_visually_matches_full_reencode(
    gop_red_clip: Path, tmp_path: Path
) -> None:
    redacted = list(range(20))  # every frame redacted -> falls back to re-encode
    meta = make_meta(
        gop_red_clip,
        [make_tracklet(1, redacted, [[8, 8, 40, 40]] * len(redacted), True)],
    )
    full = render_redacted(meta, tmp_path / "full.mp4")
    selective = render_redacted(meta, tmp_path / "sel.mp4", selective=True)

    full_frames = dict(iter_frames(full))
    sel_frames = dict(iter_frames(selective))
    assert sorted(full_frames) == sorted(sel_frames)
    # Both black out the box; the box region matches between the two paths.
    for index in (0, 10, 19):
        assert full_frames[index][12:28, 12:28].max() < 10
        assert sel_frames[index][12:28, 12:28].max() < 10


def test_selective_with_no_redactions_is_a_pure_copy(
    gop_clip: Path, tmp_path: Path
) -> None:
    meta = make_meta(gop_clip, tracklets=[])
    source = dict(iter_frames(gop_clip))

    out = render_redacted(meta, tmp_path / "sel.mp4", selective=True)
    rendered = dict(iter_frames(out))

    assert len(rendered) == len(source) == 50
    for index in source:
        np.testing.assert_array_equal(rendered[index], source[index])


def test_selective_preserves_audio(
    gop_audio_clip: Path, tmp_path: Path
) -> None:
    redacted = [10, 11]
    meta = make_meta(
        gop_audio_clip,
        [make_tracklet(1, redacted, [[10, 10, 30, 30]] * len(redacted), True)],
    )
    out = render_redacted(
        meta, tmp_path / "sel.mp4", selective=True, min_track_frames=1
    )

    source_streams = probe_streams(gop_audio_clip)
    out_streams = probe_streams(out)
    source_audio = next(s for s in source_streams if s["codec_type"] == "audio")
    out_audio = next(s for s in out_streams if s["codec_type"] == "audio")
    assert out_audio["codec_name"] == source_audio["codec_name"] == "aac"
    assert out_audio["sample_rate"] == source_audio["sample_rate"]
    assert out_audio["channels"] == source_audio["channels"]
    # Video frame count and duration line up with the source (A/V stays in sync).
    assert probe_video(out).frame_count == probe_video(gop_audio_clip).frame_count
    assert probe_video(out).duration == pytest.approx(
        probe_video(gop_audio_clip).duration, abs=0.05
    )


def test_selective_dimension_mismatch_raises(gop_clip: Path, tmp_path: Path) -> None:
    meta = make_meta(gop_clip, tracklets=[], width=640, height=480)

    with pytest.raises(RenderError, match="dimensions disagree"):
        render_redacted(meta, tmp_path / "sel.mp4", selective=True)


def test_cli_render_selective_flag(gop_red_clip: Path, tmp_path: Path) -> None:
    redacted = [10]
    meta = make_meta(
        gop_red_clip, [make_tracklet(1, redacted, [[10, 10, 30, 30]] * 1, True)]
    )
    meta_path = tmp_path / "meta.json"
    save_meta(meta_path, meta)
    out = tmp_path / "out.mp4"

    assert main(["render", "--meta", str(meta_path), "--out", str(out), "--selective"]) == 0

    assert out.is_file()
    assert probe_video(out).frame_count == 50


def _nvenc_works(tmp_path: Path) -> bool:
    probe = subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=size=256x256:duration=0.2",
         "-c:v", "h264_nvenc", "-f", "null", "-"],
        capture_output=True,
    )
    return probe.returncode == 0


@pytest.mark.parametrize("selective", [False, True])
def test_nvenc_encoder_redacts(tmp_path: Path, selective: bool) -> None:
    if not _nvenc_works(tmp_path):
        pytest.skip("h264_nvenc unavailable")
    # NVENC rejects very small frames, so don't use the 64x48 fixture.
    gop_red_clip = make_clip(
        tmp_path / "big_red.mp4",
        source="color=c=red:size=320x240:rate=25:duration=2",
        gop=10,
    )
    redacted = [10, 11, 12]
    meta = make_meta(
        gop_red_clip,
        [make_tracklet(1, redacted, [[10, 10, 30, 30]] * len(redacted), True)],
    )
    out = render_redacted(
        meta,
        tmp_path / "nv.mp4",
        selective=selective,
        encoder="h264_nvenc",
        min_track_frames=1,
    )
    rendered = dict(iter_frames(out))
    assert len(rendered) == 50
    assert rendered[11][12:28, 12:28].max() < 30


def test_unknown_encoder_rejected(gop_red_clip: Path, tmp_path: Path) -> None:
    meta = make_meta(gop_red_clip, [])
    with pytest.raises(RenderError, match="encoder"):
        render_redacted(meta, tmp_path / "x.mp4", encoder="nope")


def test_a_human_override_beats_the_classifier_in_the_rendered_boxes(red_clip: Path) -> None:
    keep = make_tracklet(1, [0, 1], [[10, 10, 20, 20]] * 2, False)
    keep.override = True  # reviewer says: redact after all
    drop = make_tracklet(2, [0, 1], [[30, 30, 40, 40]] * 2, True)
    drop.override = False  # reviewer says: leave visible
    boxes = redaction_boxes(
        make_meta(red_clip, [keep, drop]),
        max_gap=0, smooth_window=1, bridge_iou=0, min_track_frames=1,
    )
    assert boxes == {0: [(10, 10, 20, 20)], 1: [(10, 10, 20, 20)]}
