"""Video frame source: ffprobe metadata + sequential PyAV decode.

The frame source is the foundation every later pipeline stage builds on
(issue #8): probe a video once for its metadata, then decode frames
sequentially as ``(frame_idx, ndarray)`` pairs.

Frames are decoded in presentation order with PyAV and exposed as RGB
``uint8`` arrays of shape ``(height, width, 3)``. ``frame_idx`` is always
the frame's index in the *source* video, even when sampling, so
downstream stages can refer back to source frames unambiguously.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    """Container metadata for a video file, as reported by ffprobe."""

    path: Path
    fps: float
    width: int
    height: int
    frame_count: int
    duration: float

    @property
    def frame_shape(self) -> tuple[int, int]:
        """Shape ``(height, width)`` of a decoded frame (no channel axis)."""
        return (self.height, self.width)


def probe_video(path: str | Path) -> VideoInfo:
    """Probe a video file with ffprobe and return its metadata.

    Reports fps (average frame rate), pixel dimensions, total frame
    count, and duration. The frame count comes from the container's
    ``nb_frames``; when a container omits it (e.g. some MKV/WebM files)
    it is estimated as ``duration * fps``.

    Raises:
        FileNotFoundError: ``path`` does not exist.
        ValueError: the file contains no video stream.
        RuntimeError: ffprobe failed (e.g. the file is not decodable).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no such video file: {path}")

    proc = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffprobe failed for {path}: {proc.stderr.strip() or 'unknown error'}"
        )

    probe = json.loads(proc.stdout)
    stream = _pick_video_stream(probe.get("streams", []))
    if stream is None:
        raise ValueError(f"no video stream found in {path}")

    fps = _stream_fps(stream)
    duration = _stream_duration(probe, stream)
    frame_count = int(stream.get("nb_frames") or 0)
    if frame_count <= 0 and duration > 0 and fps > 0:
        # Container doesn't record a frame count; estimate it.
        frame_count = round(duration * fps)

    return VideoInfo(
        path=path,
        fps=fps,
        width=int(stream["width"]),
        height=int(stream["height"]),
        frame_count=frame_count,
        duration=duration,
    )


def iter_frames(
    path: str | Path, sample_rate: int = 1
) -> Iterator[tuple[int, np.ndarray]]:
    """Decode a video sequentially, yielding ``(frame_idx, frame)`` pairs.

    ``frame`` is an RGB ``uint8`` array of shape ``(height, width, 3)``
    (see ``probe_video`` for the dimensions). With ``sample_rate = N``
    every Nth frame is yielded (indices 0, N, 2N, ...); skipped frames
    are still decoded, because inter-frame codecs cannot be seeked
    frame-accurately without it.

    Arguments are validated eagerly; decode errors surface from PyAV as
    the iteration proceeds.
    """
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool):
        raise ValueError(f"sample_rate must be an integer, got {sample_rate!r}")
    if sample_rate < 1:
        raise ValueError(f"sample_rate must be >= 1, got {sample_rate}")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no such video file: {path}")
    return _decode_frames(path, sample_rate)


def _decode_frames(path: Path, sample_rate: int) -> Iterator[tuple[int, np.ndarray]]:
    with av.open(str(path)) as container:
        for frame_idx, frame in enumerate(container.decode(video=0)):
            if frame_idx % sample_rate == 0:
                yield frame_idx, frame.to_ndarray(format="rgb24")


def _pick_video_stream(streams: list[dict]) -> dict | None:
    for stream in streams:
        if stream.get("codec_type") != "video":
            continue
        # Skip embedded cover art (e.g. album art in an mp3/mp4).
        if stream.get("disposition", {}).get("attached_pic"):
            continue
        return stream
    return None


def _stream_fps(stream: dict) -> float:
    # avg_frame_rate is the true rate; r_frame_rate is the nominal one
    # and can be wildly off for variable-frame-rate files.
    for key in ("avg_frame_rate", "r_frame_rate"):
        rate = stream.get(key)
        if not rate or rate == "0/0":
            continue
        try:
            fps = float(Fraction(rate))
        except (ValueError, ZeroDivisionError):
            continue
        if fps > 0:
            return fps
    return 0.0


def _stream_duration(probe: dict, stream: dict) -> float:
    # format.duration is seconds; a stream's duration is in time_base units.
    fmt_duration = probe.get("format", {}).get("duration")
    if fmt_duration:
        return float(fmt_duration)
    if stream.get("duration") and stream.get("time_base"):
        return float(stream["duration"]) * float(Fraction(stream["time_base"]))
    return 0.0
