"""Rendering: draw black boxes over redacted people and write the video.

First correct renderer (issue #15); optimizing it is a separate issue
(#18 skips re-encoding frames without redactions). The strategy is a
plain full re-encode:

1. Index the metadata's boxes by frame, keeping only tracklets with
   ``redact = true`` (:func:`redaction_boxes`).
2. Decode the source video sequentially with PyAV (``frames.py``) and
   paint a filled black box over each redacted bbox on its frame.
3. Pipe the raw RGB frames into an ``ffmpeg`` subprocess that encodes
   with ``libx264`` and copies any audio stream untouched (``-c:a
   copy``), so audio stays bit-identical and video is a normal playable
   H.264 track.

Boxes are clamped to the frame and rounded *outward*, so a sloppy bbox
may hide a pixel of background but never leaks redacted pixels. If the
source video's dimensions disagree with the metadata's, rendering
refuses to run: the boxes were computed in the metadata's coordinate
space and would misalign silently.
"""

from __future__ import annotations

import math
import subprocess
from fractions import Fraction
from pathlib import Path

import av
import cv2

from .frames import VideoInfo, iter_frames, probe_video
from .meta import Metadata

DEFAULT_CRF = 18
DEFAULT_PRESET = "medium"

PixelBox = tuple[int, int, int, int]  # x1, y1, x2, y2


class RenderError(RuntimeError):
    """Raised when the re-encode fails or inputs are inconsistent."""


def redaction_boxes(meta: Metadata) -> dict[int, list[PixelBox]]:
    """Index redacted tracklets' boxes by source frame number.

    Returns ``{frame_idx: [(x1, y1, x2, y2), ...]}`` in pixels, merged
    across tracklets, for tracklets with ``redact = true`` only (false
    and null are left untouched in the output). Boxes are clamped to the
    frame and rounded outward; degenerate boxes (no area after clamping)
    are dropped.
    """
    width, height = meta.video.width, meta.video.height
    boxes: dict[int, list[PixelBox]] = {}
    for tracklet in meta.tracklets:
        if tracklet.redact is not True:
            continue
        for frame_idx, box in zip(tracklet.frames, tracklet.boxes):
            pixel_box = _clamp_box(box, width, height)
            if pixel_box is not None:
                boxes.setdefault(frame_idx, []).append(pixel_box)
    return boxes


def render_redacted(
    meta: Metadata,
    out_path: str | Path,
    *,
    crf: int = DEFAULT_CRF,
    preset: str = DEFAULT_PRESET,
) -> Path:
    """Re-encode ``meta.video.path`` with black boxes over redacted people.

    Every frame is decoded, redacted boxes for that frame are painted
    black, and the result is piped to ffmpeg (``libx264``, ``crf``,
    ``yuv420p``); audio is copied verbatim when present. Tracklets whose
    ``redact`` flag is false or null are drawn normally. The source
    video is read from ``meta.video.path`` (relative paths resolve
    against the current directory).

    Returns the output path. Raises :class:`RenderError` when the
    metadata and source video disagree or ffmpeg fails, and
    ``FileNotFoundError`` when the source video is missing.
    """
    _check_crf(crf)
    if not isinstance(preset, str) or not preset:
        raise RenderError(f"preset must be a non-empty string, got {preset!r}")

    source = Path(meta.video.path)
    if not source.is_file():
        raise FileNotFoundError(f"no such video file: {source}")

    probe = probe_video(source)
    if (probe.width, probe.height) != (meta.video.width, meta.video.height):
        raise RenderError(
            f"video dimensions disagree: {source} is {probe.width}x{probe.height}, "
            f"metadata says {meta.video.width}x{meta.video.height}; "
            "the boxes would misalign"
        )

    boxes = redaction_boxes(meta)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    _encode(source, out, boxes, probe, crf, preset)
    return out


def _clamp_box(box: list[float], width: int, height: int) -> PixelBox | None:
    x1, y1, x2, y2 = box
    x1 = max(0, min(width, math.floor(x1)))
    y1 = max(0, min(height, math.floor(y1)))
    x2 = max(0, min(width, math.ceil(x2)))
    y2 = max(0, min(height, math.ceil(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _encode(
    source: Path,
    out: Path,
    boxes: dict[int, list[PixelBox]],
    probe: VideoInfo,
    crf: int,
    preset: str,
) -> None:
    command = [
        "ffmpeg",
        "-v", "error",
        "-y",
        "-f", "rawvideo",
        "-pixel_format", "rgb24",
        "-video_size", f"{probe.width}x{probe.height}",
        # Exact rational rate so e.g. 29.97 stays 30000/1001, not a drift.
        "-framerate", _fps_arg(probe.fps),
        "-i", "pipe:0",
    ]
    has_audio = _has_audio(source)
    if has_audio:
        command += ["-i", str(source)]
    command += ["-map", "0:v:0"]
    if has_audio:
        command += ["-map", "1:a:0", "-c:a", "copy"]
    command += [
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", preset,
        "-pix_fmt", "yuv420p",
        str(out),
    ]

    proc = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    broken_pipe = False
    try:
        for frame_idx, frame in iter_frames(source):
            for x1, y1, x2, y2 in boxes.get(frame_idx, ()):
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), thickness=-1)
            proc.stdin.write(frame.tobytes())
    except BrokenPipeError:
        # ffmpeg already exited; surfaced via its return code below.
        broken_pipe = True
    except BaseException:
        proc.kill()
        proc.wait()
        raise
    finally:
        try:
            proc.stdin.close()
        except (BrokenPipeError, ValueError):
            pass

    returncode = proc.wait()
    stderr = proc.stderr.read().decode("utf-8", errors="replace").strip()
    if returncode != 0 or broken_pipe:
        raise RenderError(
            f"ffmpeg failed for {source} (exit {returncode}): "
            f"{stderr or 'unknown error'}"
        )


def _fps_arg(fps: float) -> str:
    rate = Fraction(fps).limit_denominator(1001)
    return f"{rate.numerator}/{rate.denominator}"


def _has_audio(path: Path) -> bool:
    with av.open(str(path)) as container:
        return bool(container.streams.audio)


def _check_crf(crf: int) -> None:
    if isinstance(crf, bool) or not isinstance(crf, int):
        raise RenderError(f"crf must be an integer, got {crf!r}")
    if not 0 <= crf <= 51:
        raise RenderError(f"crf must be in 0..51, got {crf}")
