"""Tests comparing the streaming pipeline to the classic one.

Uses a short clip cut from a real video if available; otherwise skips.
"""

import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest

from auto_redact.frames import probe_video
from auto_redact.meta import load_meta
from auto_redact.pipeline import run_pipeline
from auto_redact.streaming import run_pipeline_streaming

CHUCK = Path("/home/nic/Videos/Chuck Redacted/S04E16 The Masquerade.mp4")
CLIP = Path("/tmp/opencode/chuck_stream_test.mp4")


@pytest.fixture(scope="module")
def clip():
    if not CHUCK.is_file():
        pytest.skip("Chuck clip not available")
    if not CLIP.is_file():
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-ss", "0", "-t", "8",
             "-i", str(CHUCK), "-c:v", "libx264", "-preset", "ultrafast",
             "-crf", "28", "-pix_fmt", "yuv420p", "-an", str(CLIP)],
            check=True,
        )
    return CLIP


def test_streaming_matches_classic_metadata(clip, tmp_path):
    classic_dir = tmp_path / "classic"
    stream_dir = tmp_path / "stream"
    classic_dir.mkdir()
    stream_dir.mkdir()

    run_pipeline(
        clip, "person in a suit", classic_dir / "out.mp4",
        thumbs_dir=classic_dir / "thumbs", ollama_url="http://localhost:11434",
        classify_workers=1, track_workers=1, scene_cuts=True,
        progress=lambda m: None,
    )
    run_pipeline_streaming(
        clip, "person in a suit", stream_dir / "out.mp4",
        thumbs_dir=stream_dir / "thumbs", ollama_url="http://localhost:11434",
        classify_workers=1, scene_cuts=True, progress=lambda m: None,
    )

    mc = load_meta(classic_dir / "out.meta.json")
    ms = load_meta(stream_dir / "out.meta.json")
    assert len(mc.tracklets) == len(ms.tracklets)
    for a, b in zip(mc.tracklets, ms.tracklets):
        assert a.frames == b.frames, (a.track_id, "frames differ")
        assert a.best_frame == b.best_frame
        assert a.redact == b.redact
        assert a.score == b.score


def test_streaming_frame_count_matches(clip, tmp_path):
    classic_dir = tmp_path / "classic"
    stream_dir = tmp_path / "stream"
    classic_dir.mkdir()
    stream_dir.mkdir()

    run_pipeline(
        clip, "person in a suit", classic_dir / "out.mp4",
        thumbs_dir=classic_dir / "thumbs", ollama_url="http://localhost:11434",
        classify_workers=1, track_workers=1, scene_cuts=True,
        progress=lambda m: None,
    )
    run_pipeline_streaming(
        clip, "person in a suit", stream_dir / "out.mp4",
        thumbs_dir=stream_dir / "thumbs", ollama_url="http://localhost:11434",
        classify_workers=1, scene_cuts=True, progress=lambda m: None,
    )
    assert probe_video(classic_dir / "out.mp4").frame_count == probe_video(stream_dir / "out.mp4").frame_count
    assert probe_video(stream_dir / "out.mp4").frame_count == probe_video(clip).frame_count
