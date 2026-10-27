"""Rendering: draw black boxes over redacted people and write the video.

:func:`render_redacted` ties the pieces together:

1. :mod:`auto_redact.boxes` indexes the metadata's boxes by frame, keeping
   only tracklets with ``redact = true`` (gap-filled, smoothed, bridged).
2. :mod:`auto_redact.segments` (the default, ``selective=True``) copies the
   untouched GOPs of the source and re-encodes only the chunks containing
   redactions; otherwise :func:`auto_redact.encode.encode_full` decodes every
   frame, paints it and pipes it through ffmpeg. Audio is copied unchanged.
3. The video is written by the configured :class:`~auto_redact.protocols.Encoder`
   (``libx264`` or ``h264_nvenc``, which falls back to libx264 on failure).

If the source video's dimensions disagree with the metadata's, rendering
refuses to run: the boxes were computed in the metadata's coordinate space
and would misalign silently.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from .boxes import redaction_boxes
from .config import RenderConfig, resolve
from .encode import check_crf, encode_full, make_encoder, run_with_fallback
from .errors import RenderError
from .frames import probe_video
from .meta import Metadata
from .segments import render_selective

__all__ = ["RenderError", "redaction_boxes", "render_redacted"]


def render_redacted(
    meta: Metadata,
    out_path: str | Path,
    config: RenderConfig | None = None,
    *,
    on_progress: Callable[[int, int], None] | None = None,
    **overrides,
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
    cfg = resolve(RenderConfig, config, overrides)
    crf, preset, encoder, selective = cfg.crf, cfg.preset, cfg.encoder, cfg.selective
    check_crf(crf)
    if not isinstance(preset, str) or not preset:
        raise RenderError(f"preset must be a non-empty string, got {preset!r}")
    backend = make_encoder(encoder)

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

    boxes = redaction_boxes(meta, cfg)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    backend.prepare()

    def run(backend, preset: str) -> None:
        if selective:
            render_selective(source, out, boxes, probe, crf, preset, backend, on_progress)
        else:
            encode_full(source, out, boxes, probe, crf, preset, backend, on_progress)

    run_with_fallback(backend, preset, run)
    return out
