"""Snapshot selection: one best-view thumbnail per tracklet (issue #12).

Classification (#13/#14) asks a vision model a single yes/no question
per person, so each tracklet first needs one representative image. This
is heuristic v1 (keypoint-aware scoring comes later in #17): pick the
frame where the tracklet's bounding box is largest — the subject at
their biggest/closest — crop it around the box with some padding, and
save it as ``<thumbs_dir>/track_<id>.jpg``.

For every tracklet the pass records ``best_frame`` and
``best_snapshot_path`` in the metadata (schema v1, see
:mod:`auto_redact.meta`) and lays all thumbnails into a contact sheet
(``<thumbs_dir>/contact_sheet.jpg`` by default) so the choice can be
eyeballed.

Only the picked frames are decoded: frames are read sequentially (an
inter-frame codec cannot seek frame-accurately) and iteration stops as
soon as every needed frame has been captured.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .frames import iter_frames
from .meta import Metadata, Tracklet

DEFAULT_THUMBS_DIR = "thumbs"
DEFAULT_CONTACT_SHEET = "contact_sheet.jpg"
DEFAULT_PADDING = 0.2  # extra margin around the bbox, per side, as a fraction
JPEG_QUALITY = 90

CONTACT_SHEET_CELL = (192, 256)  # per-thumbnail cell, max (width, height) px
CONTACT_SHEET_LABEL_HEIGHT = 24  # px reserved below each cell for "track <id>"
CONTACT_SHEET_BACKGROUND = (240, 240, 240)


class SnapshotError(RuntimeError):
    """Raised when snapshots cannot be selected, cropped or written."""


def best_observation(
    tracklet: Tracklet, width: int, height: int
) -> tuple[int, list[float]]:
    """Pick the tracklet's largest bbox: ``(frame index into the video, box)``.

    The area of each box is measured *clamped to the frame*, so a bbox
    mostly outside the picture doesn't win on paper. Ties keep the
    earliest observation (deterministic). Raises :class:`SnapshotError`
    for a tracklet without observations, misaligned frame/box lists, or
    boxes that are all degenerate (no area after clamping).
    """
    if len(tracklet.frames) != len(tracklet.boxes):
        raise SnapshotError(
            f"tracklet {tracklet.track_id}: frames and boxes are not aligned: "
            f"got {len(tracklet.frames)} frames and {len(tracklet.boxes)} boxes"
        )
    if not tracklet.boxes:
        raise SnapshotError(f"tracklet {tracklet.track_id} has no observations")

    best_idx = 0
    best_area = -1.0
    for i, box in enumerate(tracklet.boxes):
        area = _clamped_area(box, width, height)
        if area > best_area:  # strict >: earliest observation wins ties
            best_area = area
            best_idx = i
    if best_area <= 0:
        raise SnapshotError(
            f"tracklet {tracklet.track_id} has no bbox with positive area "
            f"inside the {width}x{height} frame"
        )
    return tracklet.frames[best_idx], tracklet.boxes[best_idx]


def crop_snapshot(frame: np.ndarray, box: list[float], padding: float) -> np.ndarray:
    """Crop ``frame`` around ``box``, padded and clamped to the frame.

    ``padding`` expands the box per side as a fraction of its own width
    and height (0 = tight crop). The result is a contiguous RGB array;
    raises :class:`SnapshotError` when the padded box leaves nothing of
    the frame to crop.
    """
    _check_padding(padding)
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in box)
    if not (math.isfinite(x1) and math.isfinite(y1)
            and math.isfinite(x2) and math.isfinite(y2)):
        raise SnapshotError(f"bbox has non-finite coordinates: {box!r}")
    if x2 <= x1 or y2 <= y1:
        raise SnapshotError(f"bbox has no area: {box!r}")

    pad_x = (x2 - x1) * padding
    pad_y = (y2 - y1) * padding
    ix1 = max(0, min(width, math.floor(x1 - pad_x)))
    iy1 = max(0, min(height, math.floor(y1 - pad_y)))
    ix2 = max(0, min(width, math.ceil(x2 + pad_x)))
    iy2 = max(0, min(height, math.ceil(y2 + pad_y)))
    if ix2 <= ix1 or iy2 <= iy1:
        raise SnapshotError(f"bbox {box!r} leaves no pixels inside the frame")
    return np.ascontiguousarray(frame[iy1:iy2, ix1:ix2])


def select_snapshots(
    meta: Metadata,
    thumbs_dir: str | Path = DEFAULT_THUMBS_DIR,
    *,
    padding: float = DEFAULT_PADDING,
    contact_sheet: str | Path | None = DEFAULT_CONTACT_SHEET,
) -> Metadata:
    """Give every tracklet its best-frame thumbnail and record it in ``meta``.

    For each tracklet: pick the largest bbox (:func:`best_observation`),
    decode that frame from ``meta.video.path``, crop it
    (:func:`crop_snapshot`), and save ``<thumbs_dir>/track_<id>.jpg``.
    The tracklet's ``best_frame`` / ``best_snapshot_path`` are updated in
    place (and the returned ``meta`` is the same object, ready for
    :func:`auto_redact.meta.save_meta`).

    Args:
        meta: Tracklet metadata whose ``video.path`` is read as the source.
        thumbs_dir: Directory for the thumbnails (created if missing).
        padding: Crop margin around each bbox, per side, as a fraction of
            the bbox size (must be a finite number >= 0).
        contact_sheet: Contact sheet filename; a relative path resolves
            inside ``thumbs_dir``. ``None`` skips the contact sheet.

    Raises:
        FileNotFoundError: the source video is missing.
        SnapshotError: invalid padding, a tracklet without observations
            or usable boxes, or a needed frame absent from the video.
    """
    _check_padding(padding)
    source = Path(meta.video.path)
    if not source.is_file():
        raise FileNotFoundError(f"no such video file: {source}")

    width, height = meta.video.width, meta.video.height
    # frame_idx -> tracklets whose best snapshot lives on that frame
    by_frame: dict[int, list[tuple[Tracklet, list[float]]]] = {}
    best_frames: dict[int, int] = {}
    for tracklet in meta.tracklets:
        frame_idx, box = best_observation(tracklet, width, height)
        by_frame.setdefault(frame_idx, []).append((tracklet, box))
        best_frames[tracklet.track_id] = frame_idx

    crops: dict[int, np.ndarray] = {}
    pending = set(by_frame)
    for frame_idx, frame in iter_frames(source):
        for tracklet, box in by_frame.get(frame_idx, ()):
            crops[tracklet.track_id] = crop_snapshot(frame, box, padding)
        pending.discard(frame_idx)
        if not pending:
            break
    if pending:
        raise SnapshotError(
            f"{source}: frame(s) {sorted(pending)} not found; "
            f"the video has fewer frames than the metadata claims"
        )

    thumbs = Path(thumbs_dir)
    saved: list[tuple[int, Image.Image]] = []
    if meta.tracklets:
        thumbs.mkdir(parents=True, exist_ok=True)
    for tracklet in sorted(meta.tracklets, key=lambda t: t.track_id):
        crop = crops[tracklet.track_id]
        image = Image.fromarray(crop)
        thumb_path = thumbs / f"track_{tracklet.track_id}.jpg"
        image.save(thumb_path, format="JPEG", quality=JPEG_QUALITY)
        tracklet.best_frame = best_frames[tracklet.track_id]
        tracklet.best_snapshot_path = str(thumb_path)
        saved.append((tracklet.track_id, image))

    if contact_sheet is not None and saved:
        sheet_path = Path(contact_sheet)
        if not sheet_path.is_absolute():
            sheet_path = thumbs / sheet_path
        write_contact_sheet(saved, sheet_path)
    return meta


def write_contact_sheet(
    thumbnails: list[tuple[int, Image.Image]], out_path: str | Path
) -> Path:
    """Lay ``thumbnails`` (``(track_id, image)``) into a grid image.

    Each thumbnail is scaled down (never up) into a fixed-size cell and
    labeled ``track <id>`` underneath; the grid is roughly square. The
    sheet exists so a human can eyeball that every tracklet's "largest
    bbox" frame really shows the person prominently.
    """
    if not thumbnails:
        raise SnapshotError("no thumbnails to lay out")

    cell_w, cell_h = CONTACT_SHEET_CELL
    columns = math.ceil(math.sqrt(len(thumbnails)))
    rows = math.ceil(len(thumbnails) / columns)
    sheet = Image.new(
        "RGB",
        (columns * cell_w, rows * (cell_h + CONTACT_SHEET_LABEL_HEIGHT)),
        CONTACT_SHEET_BACKGROUND,
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for i, (track_id, image) in enumerate(thumbnails):
        cell_x = (i % columns) * cell_w
        cell_y = (i // columns) * (cell_h + CONTACT_SHEET_LABEL_HEIGHT)
        thumb = image
        scale = min(1.0, cell_w / thumb.width, cell_h / thumb.height)
        if scale < 1.0:
            thumb = thumb.resize(
                (max(1, round(thumb.width * scale)),
                 max(1, round(thumb.height * scale))),
                Image.Resampling.LANCZOS,
            )
        sheet.paste(
            thumb,
            (cell_x + (cell_w - thumb.width) // 2,
             cell_y + (cell_h - thumb.height) // 2),
        )
        draw.text(
            (cell_x + 4, cell_y + cell_h + 4),
            f"track {track_id}",
            fill=(20, 20, 20),
            font=font,
        )

    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)  # format follows the suffix (.jpg/.png/...)
    return destination


def _clamped_area(box: list[float], width: int, height: int) -> float:
    """Visible area of ``box`` after clamping it to the frame, in px²."""
    x1, y1, x2, y2 = (float(v) for v in box)
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _check_padding(padding: float) -> None:
    if isinstance(padding, bool) or not isinstance(padding, (int, float)):
        raise SnapshotError(f"padding must be a number, got {padding!r}")
    if not math.isfinite(float(padding)) or padding < 0:
        raise SnapshotError(f"padding must be finite and >= 0, got {padding!r}")
