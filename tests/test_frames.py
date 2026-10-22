"""Tests for auto_redact.frames (issue #8).

Tiny synthetic clips are generated with ffmpeg's lavfi sources, so no
sample media needs to be checked in. The 10s / 25 fps / 320x240 clip
matches the issue's acceptance check: decode a 10s clip and report
correct fps/dimensions/frame count.
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from auto_redact.frames import VideoInfo, iter_frames, probe_video

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")

pytestmark = pytest.mark.skipif(
    not (FFMPEG and FFPROBE), reason="ffmpeg/ffprobe not installed"
)

TESTSRC = "testsrc=duration=10:size=320x240:rate=25"


def make_clip(path: Path, source: str = TESTSRC) -> Path:
    """Render a lavfi test source to an h264 mp4 at ``path``."""
    subprocess.run(
        [
            FFMPEG,
            "-v", "error",
            "-y",
            "-f", "lavfi",
            "-i", source,
            "-pix_fmt", "yuv420p",
            str(path),
        ],
        check=True,
    )
    return path


@pytest.fixture
def clip(tmp_path: Path) -> Path:
    return make_clip(tmp_path / "clip.mp4")


def test_probe_reports_metadata(clip: Path) -> None:
    info = probe_video(clip)

    assert isinstance(info, VideoInfo)
    assert info.fps == pytest.approx(25.0)
    assert info.width == 320
    assert info.height == 240
    assert info.frame_count == 250
    assert info.duration == pytest.approx(10.0, abs=0.1)
    assert info.frame_shape == (240, 320)


def test_iter_frames_decodes_all_frames_sequentially(clip: Path) -> None:
    frames = list(iter_frames(clip))

    assert len(frames) == 250
    assert [idx for idx, _ in frames] == list(range(250))
    for _, frame in frames[::50]:  # spot-check shapes along the way
        assert frame.shape == (240, 320, 3)
        assert frame.dtype == np.uint8


def test_sample_rate_yields_every_nth_source_frame(clip: Path) -> None:
    frames = list(iter_frames(clip, sample_rate=2))

    # frame_idx stays a *source* index: 0, 2, 4, ...
    assert [idx for idx, _ in frames] == list(range(0, 250, 2))
    assert len(frames) == 125


def test_frames_are_rgb_ordered(tmp_path: Path) -> None:
    clip = make_clip(
        tmp_path / "red.mp4", source="color=c=red:size=64x48:rate=10:duration=1"
    )

    frame = next(iter_frames(clip))[1]

    assert frame.shape == (48, 64, 3)
    red, green, blue = frame.mean(axis=(0, 1))
    assert red > 240  # would be ~0 if the channels were BGR-ordered
    assert green < 10
    assert blue < 10


@pytest.mark.parametrize("sample_rate", [0, -1, 2.5, True, "2"])
def test_sample_rate_is_a_positive_int(sample_rate) -> None:
    with pytest.raises(ValueError):
        iter_frames("ignored.mp4", sample_rate=sample_rate)


def test_probe_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        probe_video(tmp_path / "nope.mp4")


def test_iter_frames_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        list(iter_frames(tmp_path / "nope.mp4"))


def test_probe_undecodable_file_raises(tmp_path: Path) -> None:
    text_file = tmp_path / "notes.txt"
    text_file.write_text("not a video\n")

    with pytest.raises(RuntimeError, match="ffprobe failed"):
        probe_video(text_file)


def test_probe_audio_only_file_raises(tmp_path: Path) -> None:
    wav = tmp_path / "tone.wav"
    subprocess.run(
        [
            FFMPEG,
            "-v", "error",
            "-y",
            "-f", "lavfi",
            "-i", "sine=duration=1:frequency=440",
            str(wav),
        ],
        check=True,
    )

    with pytest.raises(ValueError, match="no video stream"):
        probe_video(wav)
