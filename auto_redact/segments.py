"""Selective rendering (issue #18): copy untouched GOPs, re-encode the rest.

Encoding dominates render time (~90%), so untouched stretches of the
timeline are copied as whole GOPs instead of being decoded and re-encoded:

1. Find every keyframe (a demux-only ffprobe pass) and partition the
   timeline at keyframe boundaries; a partition chunk that contains any
   redacted frame is marked *edited*, the rest *copied*. Adjacent chunks
   of the same kind are merged (:func:`build_segments`).
2. Copied chunks are written by packet passthrough (PyAV ``demux`` ->
   ``mux``), so their bits reach the output untouched. Edited chunks are
   decoded, painted and re-encoded at GOP granularity.
3. The per-chunk files are concatenated (``-c copy``) and the original
   audio track is muxed back in one piece, so audio stays continuous and
   bit-identical.
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import av

from .encode import (
    concat_video,
    encode_full,
    ffmpeg_stdin,
    mux_audio,
    paint,
    raw_input_args,
)
from .frames import VideoInfo
from .geometry import PixelBox
from .protocols import Encoder


@dataclass
class Segment:
    """A keyframe-aligned chunk of the timeline ``[start, end)``."""

    start: int  # first frame index (inclusive)
    end: int  # one past the last frame index (exclusive)
    edited: bool  # True when any frame in the chunk is redacted


def render_selective(
    source: Path,
    out: Path,
    boxes: dict[int, list[PixelBox]],
    probe: VideoInfo,
    crf: int,
    preset: str,
    encoder: Encoder,
    on_progress: Callable[[int, int], None] | None = None,
) -> None:
    """Write ``out`` by copying untouched GOPs and re-encoding edited ones.

    Falls back to the full re-encode when the source exposes no keyframe
    index (copy can't cut safely) or when every frame is redacted.
    """
    keyframes, is_open_gop = keyframe_table(source)
    if not keyframes or is_open_gop:
        # Open GOPs (trailing B-frames that reference the next keyframe)
        # can't be split into independently decodable segments without
        # re-encoding the boundaries; fall back to the full re-encode to
        # avoid dropped frames / frozen video.
        encode_full(source, out, boxes, probe, crf, preset, encoder, on_progress)
        return

    segments = build_segments(set(boxes), keyframes, probe.frame_count)
    if 0 not in keyframes:
        encode_full(source, out, boxes, probe, crf, preset, encoder, on_progress)
        return
    if len(segments) == 1 and segments[0].edited:
        # Everything is redacted; identical to a full re-encode.
        encode_full(source, out, boxes, probe, crf, preset, encoder, on_progress)
        return

    with tempfile.TemporaryDirectory(prefix="auto-redact-") as tmp:
        tmpdir = Path(tmp)
        parts: list[Path] = []
        for number, segment in enumerate(segments):
            part = tmpdir / f"part_{number:04d}.mp4"
            if segment.edited:
                encode_segment(
                    source, part, boxes, keyframes, segment, probe, crf, preset, encoder
                )
            else:
                copy_segment(source, part, keyframes, segment)
            parts.append(part)
            if on_progress is not None:
                on_progress(segment.end, probe.frame_count)
        joined = tmpdir / "joined.mp4"
        concat_video(parts, joined)
        mux_audio(joined, source, out)


def is_open_gop(source: Path) -> bool:
    """Return True when the source uses open GOPs.

    Open GOPs have trailing B-frames that are decoded *after* the next
    keyframe but presented *before* it. Such frames cannot be cleanly
    attributed to a single segment, so selective copying drops or
    misplaces them (frozen video / artifacts). Detection is folded into
    :func:`keyframe_table` (same ffprobe pass), so this just reads that
    result.
    """
    try:
        return keyframe_table(source)[1]
    except Exception:
        return False


def keyframe_table(source: Path) -> tuple[dict[int, int], bool]:
    """Map presentation frame index -> pts for every keyframe in ``source``.

    Also detects open GOPs in the same pass. Uses a single ffprobe packet
    pass (``packet=pts,flags``), so no frames are decoded and the video is
    traversed only once. ``pts`` is in the video stream's time base; the
    frame index is derived relative to the first keyframe.

    Returns ``(table, is_open_gop)``. ``table`` is ``{}`` when the container
    reports no keyframes or the stream lacks timing information.
    """
    proc = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "packet=pts,flags",
            "-of", "csv=print_section=0",
            str(source),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return {}, False
    try:
        with av.open(str(source)) as container:
            stream = container.streams.video[0]
            time_base = stream.time_base
            rate = stream.average_rate
    except Exception:
        return {}, False
    if not time_base or not rate:
        return {}, False

    table: dict[int, int] = {}
    pts0: int | None = None
    last_keyframe_pts: int | None = None
    is_open_gop = False
    for line in proc.stdout.splitlines():
        parts = line.split(",")
        if len(parts) < 2:
            continue
        try:
            pts = int(parts[0])
        except ValueError:
            continue
        if "K" in parts[1]:
            if pts0 is None:
                pts0 = pts
            index = round(float((pts - pts0) * time_base * rate))
            table[index] = pts
            last_keyframe_pts = pts
        else:
            if last_keyframe_pts is not None and pts < last_keyframe_pts:
                is_open_gop = True
    return table, is_open_gop


def build_segments(
    altered: set[int], keyframes: dict[int, int], frame_count: int
) -> list[Segment]:
    """Partition ``[0, frame_count)`` at keyframes, flagging edited chunks.

    A chunk is *edited* when it contains any altered (redacted) frame, so
    redactions are re-encoded at whole-GOP granularity and everything else is
    copied. Adjacent chunks of the same kind are merged.
    """
    boundaries = sorted(
        boundary
        for boundary in {0, frame_count, *keyframes}
        if 0 <= boundary <= frame_count
    )
    segments: list[Segment] = []
    for start, end in zip(boundaries, boundaries[1:]):
        if end <= start:
            continue
        edited = any(start <= frame < end for frame in altered)
        if segments and segments[-1].edited == edited:
            segments[-1].end = end
        else:
            segments.append(Segment(start, end, edited))
    return segments


def copy_segment(
    source: Path, part: Path, keyframes: dict[int, int], segment: Segment
) -> None:
    """Write the packets of ``segment`` to ``part`` untouched (stream copy).

    Only whole GOPs are taken (``[segment.start, segment.end)`` are keyframe
    boundaries), so the result is independently decodable and tiles exactly
    with its neighbours. Packets are selected by pts, which is robust to B-frame
    DTS/PTS reordering.
    """
    low = keyframes[segment.start]
    high = keyframes.get(segment.end)  # None when the chunk runs to end of stream
    with av.open(str(source)) as input_container:
        stream = input_container.streams.video[0]
        with av.open(str(part), "w") as output_container:
            output_stream = output_container.add_stream_from_template(stream)
            try:
                # Jump to the keyframe at ``low`` instead of demuxing from the start.
                input_container.seek(low, stream=stream, backward=True, any_frame=False)
            except Exception:
                pass
            for packet in input_container.demux(stream):
                if packet.pts is None:
                    continue
                if packet.pts < low:
                    continue
                if high is not None and packet.pts >= high:
                    break
                packet.stream = output_stream
                output_container.mux(packet)


def encode_segment(
    source: Path,
    part: Path,
    boxes: dict[int, list[PixelBox]],
    keyframes: dict[int, int],
    segment: Segment,
    probe: VideoInfo,
    crf: int,
    preset: str,
    encoder: Encoder,
) -> None:
    """Decode ``segment``'s frames, paint redactions and encode them to ``part``.

    Seeks straight to the chunk's first keyframe, so earlier untouched frames
    are never decoded.
    """
    command = [
        "ffmpeg", "-v", "error", "-y",
        *raw_input_args(probe),
        *encoder.codec_args(crf, preset),
        "-pix_fmt", "yuv420p",
        str(part),
    ]
    with ffmpeg_stdin(command, source) as stdin, av.open(str(source)) as container:
        stream = container.streams.video[0]
        container.seek(keyframes[segment.start], stream=stream)
        index = segment.start
        for frame in container.decode(video=0):
            if index >= segment.end:
                break
            array = frame.to_ndarray(format="rgb24")
            paint(array, boxes.get(index, ()))
            stdin.write(array)
            index += 1
