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
from auto_redact.render import (
    RenderError,
    redaction_boxes,
    render_redacted,
)

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")

pytestmark = pytest.mark.skipif(
    not (FFMPEG and FFPROBE), reason="ffmpeg/ffprobe not installed"
)

RED = "color=c=red:size=64x48:rate=10:duration=2"  # 20 frames
TESTSRC = "testsrc=duration=2:size=320x240:rate=25"  # 50 frames


def make_clip(path: Path, source: str = TESTSRC, audio: bool = False) -> Path:
    """Render a lavfi test source to an h264 mp4, optionally with AAC audio."""
    command = [FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", source]
    if audio:
        duration = source.split("duration=")[1].split(":")[0]
        command += ["-f", "lavfi", "-i", f"sine=duration={duration}:frequency=440"]
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
    assert redaction_boxes(meta) == {0: [(10, 10, 20, 20)], 1: [(30, 30, 41, 41)]}


def test_redaction_boxes_merges_tracklets_per_frame(red_clip: Path) -> None:
    meta = make_meta(
        red_clip,
        [
            make_tracklet(1, [0], [[10, 10, 20, 20]], True),
            make_tracklet(2, [0, 1], [[30, 30, 40, 40], [4, 4, 8, 8]], True),
        ],
    )

    assert redaction_boxes(meta) == {
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

    assert redaction_boxes(meta) == {0: [(0, 0, 10, 10)], 1: [(60, 10, 64, 20)]}


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
