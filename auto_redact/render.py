"""Rendering: draw black boxes over redacted people and write the video.

The default renderer (issue #15) is a plain full re-encode:

1. Index the metadata's boxes by frame, keeping only tracklets with
   ``redact = true`` (:func:`redaction_boxes`). Short detection gaps
   inside a tracklet are interpolated, boxes are lightly smoothed, and
   nearby redacted tracklets with overlapping endpoints are bridged so
   the black box does not flicker when the detector or tracker drops.
2. Decode the source video sequentially with PyAV (``frames.py``) and
   paint a filled black box over each redacted bbox on its frame.
3. Pipe the raw RGB frames into an ``ffmpeg`` subprocess that encodes
   with ``libx264`` and copies any audio stream untouched (``-c:a
   copy``), so audio stays bit-identical and video is a normal playable
   H.264 track.

Issue #18 adds an *optional* selective renderer (``selective=True``,
behind the CLI's ``--selective`` flag; the default remains the full
re-encode until the optimization is proven). Encoding dominates render
time (~90%), so untouched stretches of the timeline are copied as whole
GOPs instead of being decoded and re-encoded:

1. Find every keyframe (a demux-only ffprobe pass) and partition the
   timeline at keyframe boundaries; a partition chunk that contains any
   redacted frame is marked *edited*, the rest *copied*. Adjacent chunks
   of the same kind are merged (:func:`_build_segments`).
2. Copied chunks are written by packet passthrough (PyAV ``demux`` ->
   ``mux``), so their bits reach the output untouched. Edited chunks are
   decoded, painted and re-encoded at GOP granularity.
3. The per-chunk files are concatenated (``-c copy``) and the original
   audio track is muxed back in one piece, so audio stays continuous and
   bit-identical.

Boxes are clamped to the frame and rounded *outward*, so a sloppy bbox
may hide a pixel of background but never leaks redacted pixels. If the
source video's dimensions disagree with the metadata's, rendering
refuses to run: the boxes were computed in the metadata's coordinate
space and would misalign silently.
"""

from __future__ import annotations

import math
import subprocess
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable

import av
import cv2

from .frames import VideoInfo, iter_frames, probe_video
from .meta import Metadata

DEFAULT_CRF = 18
DEFAULT_PRESET = "medium"
DEFAULT_ENCODER = "libx264"
ENCODERS = ("libx264", "h264_nvenc")

# Detection dropouts leave holes in tracklets; without filling them the
# black box blinks on and off. Defaults cover ~1.25s at 24 fps with a
# wider moving average for jittery tracking and more aggressive bridging.
DEFAULT_MAX_GAP = 30
DEFAULT_SMOOTH_WINDOW = 9
DEFAULT_BRIDGE_IOU = 0.5
# One- and few-frame "tracklets" are often detector merges that paint a
# huge box for a flash; longer tracks already cover the person after gap
# fill. Set to 1 to keep every redacted observation.
DEFAULT_MIN_TRACK_FRAMES = 5

# libx264 preset names -> nearest NVENC preset (p1 fastest .. p7 slowest);
# names not listed (e.g. "p5") pass through unchanged.
_NVENC_PRESETS = {
    "ultrafast": "p1", "superfast": "p1", "veryfast": "p2", "faster": "p3",
    "fast": "p3", "medium": "p4", "slow": "p5", "slower": "p6", "veryslow": "p7",
}

PixelBox = tuple[int, int, int, int]  # x1, y1, x2, y2
FloatBox = list[float]  # x1, y1, x2, y2


class RenderError(RuntimeError):
    """Raised when the re-encode fails or inputs are inconsistent."""


def redaction_boxes(
    meta: Metadata,
    *,
    max_gap: int = DEFAULT_MAX_GAP,
    smooth_window: int = DEFAULT_SMOOTH_WINDOW,
    bridge_iou: float = DEFAULT_BRIDGE_IOU,
    min_track_frames: int = DEFAULT_MIN_TRACK_FRAMES,
) -> dict[int, list[PixelBox]]:
    """Index redacted tracklets' boxes by source frame number.

    Returns ``{frame_idx: [(x1, y1, x2, y2), ...]}`` in pixels, merged
    across tracklets, for tracklets with ``redact = true`` only (false
    and null are left untouched in the output). Boxes are clamped to the
    frame and rounded outward; degenerate boxes (no area after clamping)
    are dropped.

    Tracklets often miss frames when the detector drops out. Gaps of at
    most ``max_gap`` frames inside a tracklet are filled by linear
    interpolation so the black box stays continuous. A centered moving
    average of width ``smooth_window`` (odd; ``1`` disables) settles
    per-frame jitter. When one redacted tracklet ends and another starts
    within ``max_gap`` frames with endpoint IoU of at least
    ``bridge_iou``, the seam between them is interpolated too (ByteTrack
    ID switches); ``bridge_iou <= 0`` disables bridging. Tracklets with
    fewer than ``min_track_frames`` observations are skipped (they are
    usually false merges that flash a huge box); pass ``1`` to keep them.
    """
    if isinstance(max_gap, bool) or not isinstance(max_gap, int) or max_gap < 0:
        raise RenderError(f"max_gap must be an integer >= 0, got {max_gap!r}")
    if (
        isinstance(smooth_window, bool)
        or not isinstance(smooth_window, int)
        or smooth_window < 1
        or smooth_window % 2 == 0
    ):
        raise RenderError(
            f"smooth_window must be an odd integer >= 1, got {smooth_window!r}"
        )
    if isinstance(bridge_iou, bool) or not isinstance(bridge_iou, (int, float)):
        raise RenderError(f"bridge_iou must be a number, got {bridge_iou!r}")
    if (
        isinstance(min_track_frames, bool)
        or not isinstance(min_track_frames, int)
        or min_track_frames < 1
    ):
        raise RenderError(
            f"min_track_frames must be an integer >= 1, got {min_track_frames!r}"
        )

    width, height = meta.video.width, meta.video.height
    boxes: dict[int, list[PixelBox]] = {}
    series: list[tuple[list[int], list[FloatBox]]] = []

    for tracklet in meta.tracklets:
        if tracklet.redact is not True:
            continue
        if len(tracklet.frames) < min_track_frames:
            continue
        filled_frames, filled_boxes = _fill_tracklet_gaps(
            tracklet.frames, tracklet.boxes, max_gap
        )
        if smooth_window > 1 and len(filled_boxes) > 1:
            filled_boxes = _smooth_boxes(filled_frames, filled_boxes, smooth_window)
        series.append((filled_frames, filled_boxes))
        for frame_idx, box in zip(filled_frames, filled_boxes):
            pixel_box = _clamp_box(box, width, height)
            if pixel_box is not None:
                boxes.setdefault(frame_idx, []).append(pixel_box)

    if bridge_iou > 0 and max_gap > 0 and len(series) > 1:
        for frame_idx, box in _bridge_tracklet_seams(series, max_gap, bridge_iou):
            pixel_box = _clamp_box(box, width, height)
            if pixel_box is not None:
                boxes.setdefault(frame_idx, []).append(pixel_box)

    return boxes


def _fill_tracklet_gaps(
    frames: list[int],
    boxes: list[FloatBox],
    max_gap: int,
) -> tuple[list[int], list[FloatBox]]:
    """Insert linearly interpolated boxes across short missing spans."""
    if not frames:
        return [], []
    out_frames = [frames[0]]
    out_boxes = [list(boxes[0])]
    for prev_frame, prev_box, frame, box in zip(
        frames, boxes, frames[1:], boxes[1:]
    ):
        gap = frame - prev_frame - 1
        if 0 < gap <= max_gap:
            for step in range(1, gap + 1):
                t = step / (gap + 1)
                out_frames.append(prev_frame + step)
                out_boxes.append(_lerp_box(prev_box, box, t))
        out_frames.append(frame)
        out_boxes.append(list(box))
    return out_frames, out_boxes


def _lerp_box(a: FloatBox, b: FloatBox, t: float) -> FloatBox:
    return [a[i] + (b[i] - a[i]) * t for i in range(4)]


def _smooth_boxes(
    frames: list[int],
    boxes: list[FloatBox],
    window: int,
) -> list[FloatBox]:
    """Centered moving average over consecutive-frame runs only.

    A new run starts after any temporal hole so a cut or unfilled gap
    does not pull the average across unrelated poses.
    """
    if len(boxes) <= 1 or window <= 1:
        return [list(b) for b in boxes]

    radius = window // 2
    smoothed: list[FloatBox] = [list(b) for b in boxes]
    run_start = 0
    for i in range(1, len(frames) + 1):
        at_end = i == len(frames)
        broken = not at_end and frames[i] != frames[i - 1] + 1
        if not (at_end or broken):
            continue
        run_end = i  # exclusive
        for j in range(run_start, run_end):
            lo = max(run_start, j - radius)
            hi = min(run_end, j + radius + 1)
            n = hi - lo
            smoothed[j] = [
                sum(boxes[k][c] for k in range(lo, hi)) / n for c in range(4)
            ]
        run_start = i
    return smoothed


def _box_iou(a: FloatBox, b: FloatBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _bridge_tracklet_seams(
    series: list[tuple[list[int], list[FloatBox]]],
    max_gap: int,
    min_iou: float,
) -> list[tuple[int, FloatBox]]:
    """Interpolate short seams between overlapping redacted tracklets."""
    endpoints: list[tuple[int, FloatBox, int, FloatBox]] = []
    for frames, boxes in series:
        if not frames:
            continue
        endpoints.append((frames[0], boxes[0], frames[-1], boxes[-1]))

    extra: list[tuple[int, FloatBox]] = []
    for i, (_a0, _ab0, a_end, a_box) in enumerate(endpoints):
        for j, (b_start, b_box, _b1, _bb1) in enumerate(endpoints):
            if i == j:
                continue
            gap = b_start - a_end - 1
            if not (0 < gap <= max_gap):
                continue
            if _box_iou(a_box, b_box) < min_iou:
                continue
            for step in range(1, gap + 1):
                t = step / (gap + 1)
                extra.append((a_end + step, _lerp_box(a_box, b_box, t)))
    return extra


def render_redacted(
    meta: Metadata,
    out_path: str | Path,
    *,
    crf: int = DEFAULT_CRF,
    preset: str = DEFAULT_PRESET,
    selective: bool = False,
    encoder: str = DEFAULT_ENCODER,
    max_gap: int = DEFAULT_MAX_GAP,
    smooth_window: int = DEFAULT_SMOOTH_WINDOW,
    bridge_iou: float = DEFAULT_BRIDGE_IOU,
    min_track_frames: int = DEFAULT_MIN_TRACK_FRAMES,
    on_progress: Callable[[int, int], None] | None = None,
) -> Path:
    """Re-encode ``meta.video.path`` with black boxes over redacted people.

    Every frame is decoded, redacted boxes for that frame are painted
    black, and the result is piped to ffmpeg (``libx264``, ``crf``,
    ``yuv420p``); audio is copied verbatim when present. Tracklets whose
    ``redact`` flag is false or null are drawn normally. The source
    video is read from ``meta.video.path`` (relative paths resolve
    against the current directory).

    ``max_gap``, ``smooth_window``, ``bridge_iou``, and
    ``min_track_frames`` are forwarded to :func:`redaction_boxes` so
    detection dropouts and short ID switches do not flicker the black
    box.

    With ``selective=True`` (issue #18) the untouched stretches of the
    timeline are stream-copied as whole GOPs and only chunks containing
    redacted frames are re-encoded; the result is visually identical but
    meaningfully faster when little of the frame is redacted. It falls
    back to the full re-encode when no keyframe index is available.

    ``encoder`` is ``"libx264"`` (CPU) or ``"h264_nvenc"`` (NVIDIA GPU;
    ``crf`` becomes its constant-quality value and ``preset`` is mapped to
    the nearest ``p1``-``p7``).

    ``on_progress(done, total)`` is called with frame counts as rendering
    advances (selective mode reports per segment).

    Returns the output path. Raises :class:`RenderError` when the
    metadata and source video disagree or ffmpeg fails, and
    ``FileNotFoundError`` when the source video is missing.
    """
    _check_crf(crf)
    if not isinstance(preset, str) or not preset:
        raise RenderError(f"preset must be a non-empty string, got {preset!r}")
    if encoder not in ENCODERS:
        raise RenderError(f"encoder must be one of {ENCODERS}, got {encoder!r}")

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

    boxes = redaction_boxes(
        meta,
        max_gap=max_gap,
        smooth_window=smooth_window,
        bridge_iou=bridge_iou,
        min_track_frames=min_track_frames,
    )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if selective:
        _render_selective(source, out, boxes, probe, crf, preset, encoder, on_progress)
    else:
        _encode(source, out, boxes, probe, crf, preset, encoder, on_progress)
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
    encoder: str,
    on_progress: Callable[[int, int], None] | None = None,
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
    has_audio = getattr(probe, "has_audio", None)
    if has_audio is None:
        has_audio = _has_audio(source)
    if has_audio:
        command += ["-i", str(source)]
    command += ["-map", "0:v:0"]
    if has_audio:
        command += ["-map", "1:a:0", "-c:a", "copy"]
    command += [
        *_video_codec_args(encoder, crf, preset),
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
            proc.stdin.write(frame)
            if on_progress is not None:
                on_progress(frame_idx + 1, probe.frame_count)
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


def _video_codec_args(encoder: str, crf: int, preset: str) -> list[str]:
    if encoder == "h264_nvenc":
        return [
            "-c:v", "h264_nvenc",
            "-rc", "vbr",
            "-cq", str(crf),
            "-b:v", "0",
            "-preset", _NVENC_PRESETS.get(preset, preset),
        ]
    return ["-c:v", "libx264", "-crf", str(crf), "-preset", preset]


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


# --- selective rendering (issue #18) ----------------------------------------


@dataclass
class _Segment:
    """A keyframe-aligned chunk of the timeline ``[start, end)``."""

    start: int  # first frame index (inclusive)
    end: int  # one past the last frame index (exclusive)
    edited: bool  # True when any frame in the chunk is redacted


def _render_selective(
    source: Path,
    out: Path,
    boxes: dict[int, list[PixelBox]],
    probe: VideoInfo,
    crf: int,
    preset: str,
    encoder: str,
    on_progress: Callable[[int, int], None] | None = None,
) -> None:
    """Write ``out`` by copying untouched GOPs and re-encoding edited ones.

    Falls back to the full re-encode when the source exposes no keyframe
    index (copy can't cut safely) or when every frame is redacted.
    """
    keyframes, is_open_gop = _keyframe_table(source)
    if not keyframes or is_open_gop:
        # Open GOPs (trailing B-frames that reference the next keyframe)
        # can't be split into independently decodable segments without
        # re-encoding the boundaries; fall back to the full re-encode to
        # avoid dropped frames / frozen video.
        _encode(source, out, boxes, probe, crf, preset, encoder, on_progress)
        return

    segments = _build_segments(set(boxes), keyframes, probe.frame_count)
    if 0 not in keyframes:
        _encode(source, out, boxes, probe, crf, preset, encoder, on_progress)
        return
    if len(segments) == 1 and segments[0].edited:
        # Everything is redacted; identical to a full re-encode.
        _encode(source, out, boxes, probe, crf, preset, encoder, on_progress)
        return

    with tempfile.TemporaryDirectory(prefix="auto-redact-") as tmp:
        tmpdir = Path(tmp)
        parts: list[Path] = []
        for number, segment in enumerate(segments):
            part = tmpdir / f"part_{number:04d}.mp4"
            if segment.edited:
                _encode_segment(
                    source, part, boxes, keyframes, segment, probe, crf, preset, encoder
                )
            else:
                _copy_segment(source, part, keyframes, segment)
            parts.append(part)
            if on_progress is not None:
                on_progress(segment.end, probe.frame_count)
        joined = tmpdir / "joined.mp4"
        _concat_video(parts, joined)
        _mux_audio(joined, source, out)


def _is_open_gop(source: Path) -> bool:
    """Return True when the source uses open GOPs.

    Open GOPs have trailing B-frames that are decoded *after* the next
    keyframe but presented *before* it. Such frames cannot be cleanly
    attributed to a single segment, so selective copying drops or
    misplaces them (frozen video / artifacts). Detection is folded into
    :func:`_keyframe_table` (same ffprobe pass), so this just reads that
    result.
    """
    try:
        return _keyframe_table(source)[1]
    except Exception:
        return False


def _keyframe_table(source: Path) -> tuple[dict[int, int], bool]:
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


def _build_segments(
    altered: set[int], keyframes: dict[int, int], frame_count: int
) -> list[_Segment]:
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
    segments: list[_Segment] = []
    for start, end in zip(boundaries, boundaries[1:]):
        if end <= start:
            continue
        edited = any(start <= frame < end for frame in altered)
        if segments and segments[-1].edited == edited:
            segments[-1].end = end
        else:
            segments.append(_Segment(start, end, edited))
    return segments


def _copy_segment(
    source: Path, part: Path, keyframes: dict[int, int], segment: _Segment
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
            for packet in input_container.demux(stream):
                if packet.pts is None:
                    continue
                if packet.pts < low:
                    continue
                if high is not None and packet.pts >= high:
                    break
                packet.stream = output_stream
                output_container.mux(packet)


def _encode_segment(
    source: Path,
    part: Path,
    boxes: dict[int, list[PixelBox]],
    keyframes: dict[int, int],
    segment: _Segment,
    probe: VideoInfo,
    crf: int,
    preset: str,
    encoder: str,
) -> None:
    """Decode ``segment``'s frames, paint redactions and encode them to ``part``.

    Seeks straight to the chunk's first keyframe, so earlier untouched frames
    are never decoded.
    """
    command = [
        "ffmpeg",
        "-v", "error",
        "-y",
        "-f", "rawvideo",
        "-pixel_format", "rgb24",
        "-video_size", f"{probe.width}x{probe.height}",
        "-framerate", _fps_arg(probe.fps),
        "-i", "pipe:0",
        *_video_codec_args(encoder, crf, preset),
        "-pix_fmt", "yuv420p",
        str(part),
    ]
    proc = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    broken_pipe = False
    try:
        with av.open(str(source)) as container:
            stream = container.streams.video[0]
            container.seek(keyframes[segment.start], stream=stream)
            index = segment.start
            for frame in container.decode(video=0):
                if index >= segment.end:
                    break
                array = frame.to_ndarray(format="rgb24")
                for x1, y1, x2, y2 in boxes.get(index, ()):
                    cv2.rectangle(array, (x1, y1), (x2, y2), (0, 0, 0), thickness=-1)
                proc.stdin.write(array)
                index += 1
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


def _concat_video(parts: list[Path], joined: Path) -> None:
    """Concatenate the per-chunk video files into ``joined`` without re-encoding."""
    listing = joined.with_suffix(".txt")
    listing.write_text(
        "".join(f"file '{part.as_posix()}'\n" for part in parts), encoding="utf-8"
    )
    _run_ffmpeg(
        ["-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(joined)]
    )


def _mux_audio(video: Path, source: Path, out: Path) -> None:
    """Copy the joined video and the source's (whole, untouched) audio to ``out``."""
    _run_ffmpeg(
        [
            "-i", str(video),
            "-i", str(source),
            "-map", "0:v:0",
            "-map", "1:a?",
            "-c", "copy",
            "-movflags", "frag_keyframe+empty_moov",
            str(out),
        ]
    )


def _run_ffmpeg(arguments: list[str]) -> None:
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", *arguments], capture_output=True
    )
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RenderError(f"ffmpeg failed: {stderr or 'unknown error'}")
