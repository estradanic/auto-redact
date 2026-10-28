"""Encoding backends and the ffmpeg plumbing around them.

An :class:`~auto_redact.protocols.Encoder` says how ffmpeg is told to write
H.264 (CPU ``libx264`` or NVIDIA ``h264_nvenc``) and what to fall back to
when it fails; everything else here is encoder-agnostic: painting frames
into an ffmpeg pipe, concatenating segments and muxing the original audio.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path
from typing import Callable, Iterator

import av
import cv2

from .config import ENCODERS, RenderConfig
from .errors import RenderError
from .frames import VideoInfo, iter_frames
from .geometry import PixelBox
from .protocols import Encoder

logger = logging.getLogger(__name__)

# libx264 preset names -> nearest NVENC preset (p1 fastest .. p7 slowest);
# names not listed (e.g. "p5") pass through unchanged.
_NVENC_PRESETS = {
    "ultrafast": "p1", "superfast": "p1", "veryfast": "p2", "faster": "p3",
    "fast": "p3", "medium": "p4", "slow": "p5", "slower": "p6", "veryslow": "p7",
}


class Libx264Encoder:
    name = "libx264"

    def prepare(self) -> None:
        pass

    def codec_args(self, crf: int, preset: str) -> list[str]:
        return ["-c:v", "libx264", "-crf", str(crf), "-preset", preset]

    def fallback(self, preset: str, error: BaseException) -> tuple[Encoder, str] | None:
        return None


class NvencEncoder:
    name = "h264_nvenc"

    def prepare(self) -> None:
        """Free cached CUDA memory so NVENC can create its own context."""
        import gc

        gc.collect()
        torch = sys.modules.get("torch")
        try:
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def codec_args(self, crf: int, preset: str) -> list[str]:
        return [
            "-c:v", "h264_nvenc",
            "-rc", "vbr",
            "-cq", str(crf),
            "-b:v", "0",
            "-preset", _NVENC_PRESETS.get(preset, preset),
        ]

    def fallback(self, preset: str, error: BaseException) -> tuple[Encoder, str] | None:
        """Retry on libx264; NVENC is flaky on small GPUs."""
        logger.warning(
            "h264_nvenc failed (%s); falling back to libx264. On small GPUs "
            "NVENC needs free VRAM for its own CUDA context.",
            str(error).splitlines()[0][:200],
        )
        if preset in _NVENC_PRESETS.values():
            preset = RenderConfig.preset
        return Libx264Encoder(), preset


_ENCODER_CLASSES = {"libx264": Libx264Encoder, "h264_nvenc": NvencEncoder}
assert set(_ENCODER_CLASSES) == set(ENCODERS)


def make_encoder(name: str) -> Encoder:
    try:
        return _ENCODER_CLASSES[name]()
    except KeyError:
        raise RenderError(f"encoder must be one of {ENCODERS}, got {name!r}") from None


def run_with_fallback(
    encoder: Encoder, preset: str, run: Callable[[Encoder, str], None]
) -> tuple[Encoder, str]:
    """``run(encoder, preset)``, retrying once on the encoder's fallback.

    Returns the encoder/preset that succeeded, so callers doing several
    encodes can stay on the fallback.
    """
    try:
        run(encoder, preset)
        return encoder, preset
    except RenderError as error:
        fallback = encoder.fallback(preset, error)
        if fallback is None:
            raise
        run(*fallback)
        return fallback


def check_crf(crf: int) -> None:
    if isinstance(crf, bool) or not isinstance(crf, int):
        raise RenderError(f"crf must be an integer, got {crf!r}")
    if not 0 <= crf <= 51:
        raise RenderError(f"crf must be in 0..51, got {crf}")


def fps_arg(fps: float) -> str:
    # Exact rational rate so e.g. 29.97 stays 30000/1001, not a drift.
    rate = Fraction(fps).limit_denominator(1001)
    return f"{rate.numerator}/{rate.denominator}"


def has_audio(path: Path) -> bool:
    with av.open(str(path)) as container:
        return bool(container.streams.audio)


def raw_input_args(probe: VideoInfo) -> list[str]:
    """ffmpeg arguments for reading raw RGB frames from stdin."""
    return [
        "-f", "rawvideo",
        "-pixel_format", "rgb24",
        "-video_size", f"{probe.width}x{probe.height}",
        "-framerate", fps_arg(probe.fps),
        "-i", "pipe:0",
    ]


@contextmanager
def ffmpeg_stdin(command: list[str], source: Path) -> Iterator:
    """Run ``command`` and yield its stdin for writing raw frames.

    On exit, waits for ffmpeg and raises :class:`RenderError` if it failed
    (including when it quit early and the pipe broke).
    """
    proc = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    broken_pipe = False
    try:
        yield proc.stdin
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


def paint(frame, boxes: list[PixelBox] | tuple = ()) -> None:
    """Fill each box solid black, in place."""
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), thickness=-1)


def encode_full(
    source: Path,
    out: Path,
    boxes: dict[int, list[PixelBox]],
    probe: VideoInfo,
    crf: int,
    preset: str,
    encoder: Encoder,
    on_progress: Callable[[int, int], None] | None = None,
) -> None:
    """Decode every frame, paint redactions and re-encode the whole video.

    Audio is copied unchanged when the source has any.
    """
    command = ["ffmpeg", "-v", "error", "-y", *raw_input_args(probe)]
    audio = getattr(probe, "has_audio", None)
    if audio is None:
        audio = has_audio(source)
    if audio:
        command += ["-i", str(source)]
    command += ["-map", "0:v:0"]
    if audio:
        command += ["-map", "1:a:0", "-c:a", "copy"]
    command += [*encoder.codec_args(crf, preset), "-pix_fmt", "yuv420p",
               "-movflags", "frag_keyframe+empty_moov", str(out)]

    with ffmpeg_stdin(command, source) as stdin:
        for frame_idx, frame in iter_frames(source):
            paint(frame, boxes.get(frame_idx, ()))
            stdin.write(frame)
            if on_progress is not None:
                on_progress(frame_idx + 1, probe.frame_count)


def run_ffmpeg(arguments: list[str]) -> None:
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", *arguments], capture_output=True
    )
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RenderError(f"ffmpeg failed: {stderr or 'unknown error'}")


def concat_video(parts: list[Path], joined: Path) -> None:
    """Concatenate the per-chunk video files into ``joined`` without re-encoding."""
    listing = joined.with_suffix(".txt")
    listing.write_text(
        "".join(f"file '{part.as_posix()}'\n" for part in parts), encoding="utf-8"
    )
    run_ffmpeg(
        ["-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(joined)]
    )


def mux_audio(video: Path, source: Path, out: Path) -> None:
    """Copy the joined video and the source's (whole, untouched) audio to ``out``."""
    run_ffmpeg(
        [
            "-i", str(video),
            "-i", str(source),
            "-map", "0:v:0",
            "-map", "1:a?",
            "-c", "copy",
            str(out),
        ]
    )
