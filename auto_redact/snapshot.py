"""Snapshot selection: one best-view thumbnail per tracklet (issue #12).

Classification (#13/#14) asks a vision model a single yes/no question
per person, so each tracklet first needs one representative image. This
scores every observation (#17) as the square root of the clamped bbox
area — bigger/closer is better, but with diminishing returns so extreme
close-ups don't beat a clear medium shot — weighted by keypoint quality:
overall keypoint confidence (visibility), confidence of the head keypoints
(nose, eyes, ears), confidence of the lower-body keypoints (hips, knees,
ankles), a penalty for each frame edge the box touches (the person is
likely cut off), how much of the box other tracklets cover in that
frame (occlusion), and how much of the *padded crop* other people
intrude into (distractors). Tracklets without keypoints are scored by
size, edges and occlusion only.
The best frame is cropped around the box with some padding and saved as
``<thumbs_dir>/track_<id>.jpg``.

By default each thumbnail is then *isolated*: every other tracklet
visible on the chosen frame gets its pixels blacked out of the crop
(:func:`redact_others`), so by the time the image reaches the classifier
the only possible subject is the tracklet the snapshot was taken for.
Bystanders in a crowded frame can otherwise match the prompt themselves
and get the wrong person flagged for redaction. Boxes that look like
the subject itself (high bbox IoU — a tracker fragment of the same
person) are never painted.

For every tracklet the pass records ``best_frame``,
``best_snapshot_path`` and ``isolated_track_ids`` in the metadata
(schema v1, see :mod:`auto_redact.meta`) and lays all thumbnails into a
contact sheet (``<thumbs_dir>/contact_sheet.jpg`` by default) so the
choice can be eyeballed.

Only the picked frames are decoded: frames are read sequentially (an
inter-frame codec cannot seek frame-accurately) and iteration stops as
soon as every needed frame has been captured.
"""

from __future__ import annotations

import json
import math
from typing import Any, Callable
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import os

from .frames import iter_frames
from .meta import Metadata, Tracklet, save_meta, save_meta_fast
from .track import _resolve_meta_path
from . import track as _track_mod


def _load_yolo(device=None):
    """Delegate to the cached loader in :mod:`auto_redact.track`."""
    return _track_mod._load_yolo(device)

DEFAULT_THUMBS_DIR = "thumbs"
DEFAULT_CONTACT_SHEET = "contact_sheet.jpg"
DEFAULT_PADDING = 0.2  # extra margin around the bbox, per side, as a fraction
JPEG_QUALITY = 90

HEAD_KEYPOINTS = 5  # COCO order: nose, left/right eye, left/right ear
HEAD_WEIGHT = 0.5  # share of the score that depends on head visibility
BODY_KEYPOINTS = slice(11, 17)  # COCO: hips, knees, ankles
BODY_WEIGHT = 0.5  # share of the score that depends on lower-body visibility
EDGE_PENALTY = 0.85  # score multiplier per frame edge the box touches
EDGE_TOLERANCE = 1.0  # px

SAME_PERSON_IOU = 0.5  # bbox IoU >= this: probably the subject, not a bystander
# When the subject's bbox covers at least this fraction of the frame, treat
# the observation as a close-up: blacking out bystanders (tracklet or
# sub-isolation) paints large bars that confuse the classifier, and the
# subject already dominates the crop, so isolation is skipped.
CLOSEUP_AREA_FRACTION = 0.35
DISTRACTOR_WEIGHT = 0.5  # max share of the score removed by bystanders in the crop
ISOLATE_DILATE = 0.1  # painted bystander boxes grow per side, fraction of their size

CONTACT_SHEET_CELL = (192, 256)  # per-thumbnail cell, max (width, height) px
CONTACT_SHEET_LABEL_HEIGHT = 24  # px reserved below each cell for "track <id>"
CONTACT_SHEET_BACKGROUND = (240, 240, 240)


class SnapshotError(RuntimeError):
    """Raised when snapshots cannot be selected, cropped or written."""


def best_observation(
    tracklet: Tracklet,
    width: int,
    height: int,
    others: Sequence[Tracklet] = (),
    frame_boxes: dict[int, list[tuple[int, list[float]]]] | None = None,
    *,
    padding: float = DEFAULT_PADDING,
    head_weight: float = HEAD_WEIGHT,
    body_weight: float = BODY_WEIGHT,
    edge_penalty: float = EDGE_PENALTY,
    edge_tolerance: float = EDGE_TOLERANCE,
    distractor_weight: float = DISTRACTOR_WEIGHT,
    same_person_iou: float = SAME_PERSON_IOU,
) -> tuple[int, list[float]]:
    """Pick the tracklet's best observation: ``(frame index, box)``.

    Each observation scores ``sqrt(area) * visibility * head * body *
    edges * (1 - occlusion) * (1 - distractors)`` (see
    :func:`observation_score`); ``others`` are the other tracklets used
    to measure occlusion and crop intrusion. Callers scoring many
    tracklets can pass a prebuilt ``frame_boxes`` index (see
    :func:`index_boxes_by_frame`) to avoid rebuilding it per tracklet;
    it then replaces ``others``. ``padding`` is the crop margin used
    later, so frames where bystanders would intrude into the padded
    crop are mildly deranked (see :func:`observation_score`); pass 0 to
    score on the raw bbox only. The area is measured *clamped to the
    frame*, so a bbox mostly outside the picture doesn't win on paper.
    Ties keep the earliest observation (deterministic). Raises
    :class:`SnapshotError` for a tracklet without observations,
    misaligned frame/box/keypoint lists, or boxes that are all degenerate
    (no area after clamping).
    """
    if len(tracklet.frames) != len(tracklet.boxes):
        raise SnapshotError(
            f"tracklet {tracklet.track_id}: frames and boxes are not aligned: "
            f"got {len(tracklet.frames)} frames and {len(tracklet.boxes)} boxes"
        )
    if not tracklet.boxes:
        raise SnapshotError(f"tracklet {tracklet.track_id} has no observations")
    if tracklet.keypoints and len(tracklet.keypoints) != len(tracklet.boxes):
        raise SnapshotError(
            f"tracklet {tracklet.track_id}: keypoints and boxes are not aligned: "
            f"got {len(tracklet.keypoints)} keypoint sets and "
            f"{len(tracklet.boxes)} boxes"
        )

    if frame_boxes is None:
        frame_boxes = index_boxes_by_frame(others)
    own_id = tracklet.track_id
    best_idx = 0
    best_score = -1.0
    any_area = False
    for i, box in enumerate(tracklet.boxes):
        if _clamped_area(box, width, height) <= 0:
            continue
        any_area = True
        keypoints = tracklet.keypoints[i] if tracklet.keypoints else []
        frame_idx = tracklet.frames[i]
        score = observation_score(
            box,
            keypoints,
            width,
            height,
            [
                other
                for owner, other in frame_boxes.get(frame_idx, ())
                if owner != own_id
            ],
            padding=padding,
            head_weight=head_weight,
            body_weight=body_weight,
            edge_penalty=edge_penalty,
            edge_tolerance=edge_tolerance,
            distractor_weight=distractor_weight,
            same_person_iou=same_person_iou,
        )
        if score > best_score:  # strict >: earliest observation wins ties
            best_score = score
            best_idx = i
    if not any_area:
        raise SnapshotError(
            f"tracklet {tracklet.track_id} has no bbox with positive area "
            f"inside the {width}x{height} frame"
        )
    return tracklet.frames[best_idx], tracklet.boxes[best_idx]


def index_boxes_by_frame(
    tracklets: Sequence[Tracklet],
) -> dict[int, list[tuple[int, list[float]]]]:
    """Map frame index -> ``(track_id, box)`` for every observation."""
    index: dict[int, list[tuple[int, list[float]]]] = {}
    for tracklet in tracklets:
        for frame_idx, box in zip(tracklet.frames, tracklet.boxes):
            index.setdefault(frame_idx, []).append((tracklet.track_id, box))
    return index


def observation_score(
    box: list[float],
    keypoints: Sequence[Sequence[float]],
    width: int,
    height: int,
    other_boxes: Sequence[Sequence[float]] = (),
    *,
    padding: float = DEFAULT_PADDING,
    head_weight: float = HEAD_WEIGHT,
    body_weight: float = BODY_WEIGHT,
    edge_penalty: float = EDGE_PENALTY,
    edge_tolerance: float = EDGE_TOLERANCE,
    distractor_weight: float = DISTRACTOR_WEIGHT,
    same_person_iou: float = SAME_PERSON_IOU,
) -> float:
    """Score one observation: higher means a better snapshot candidate.

    ``sqrt(area) * visibility * head * body * edges * (1 - occlusion)
    * (1 - distractors)``
    where

    * ``area``: bbox area clamped to the frame;
    * ``visibility``: mean keypoint confidence (1 without keypoints);
    * ``head``: ``1 - HEAD_WEIGHT * (1 - mean confidence of the head
      keypoints)``, so a hidden head halves the score but a body
      without a face isn't discarded outright (1 without keypoints);
    * ``body``: like ``head`` but for the hip/knee/ankle keypoints, so a
      full-body view beats a cropped one (1 without keypoints);
    * ``edges``: ``EDGE_PENALTY`` for every frame edge the box touches;
    * ``occlusion``: largest fraction of the box covered by one of
      ``other_boxes`` (the other people in the same frame);
    * ``distractors``: ``DISTRACTOR_WEIGHT`` times the fraction of the
      *padded crop window* (the thumbnail this observation will become,
      see :func:`crop_snapshot`) intruded on by other people. This
      mildly prefers moments where nobody else would appear in the
      thumbnail; isolation (see :func:`redact_others`) still removes
      them when no clean moment exists. ``padding`` is that crop
      margin (0 disables the term); boxes that mostly coincide with
      ``box`` (``SAME_PERSON_IOU``) count as the subject, not
      bystanders.
    """
    area = _clamped_area(box, width, height)
    if area <= 0:
        return 0.0
    visibility = 1.0
    head = 1.0
    body = 1.0
    if keypoints:
        confs = [_clamp01(kp[2]) for kp in keypoints]
        visibility = sum(confs) / len(confs)
        head_confs = confs[:HEAD_KEYPOINTS]
        head = 1.0 - head_weight * (1.0 - sum(head_confs) / len(head_confs))
        body_confs = confs[BODY_KEYPOINTS]
        if body_confs:
            body = 1.0 - body_weight * (1.0 - sum(body_confs) / len(body_confs))
    touched = sum(
        (
            box[0] <= edge_tolerance,
            box[1] <= edge_tolerance,
            box[2] >= width - edge_tolerance,
            box[3] >= height - edge_tolerance,
        )
    )
    occlusion = max(
        (_overlap_fraction(box, other, width, height) for other in other_boxes),
        default=0.0,
    )
    distractor = (
        _distractor_fraction(box, other_boxes, width, height, padding, same_person_iou)
        if padding > 0
        else 0.0
    )
    return (
        math.sqrt(area) * visibility * head * body
        * edge_penalty**touched * (1.0 - occlusion)
        * (1.0 - distractor_weight * distractor)
    )


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

    ix1, iy1, ix2, iy2 = _padded_crop_window(box, width, height, padding)
    if ix2 <= ix1 or iy2 <= iy1:
        raise SnapshotError(f"bbox {box!r} leaves no pixels inside the frame")
    return np.ascontiguousarray(frame[iy1:iy2, ix1:ix2])


def redact_others(
    crop: np.ndarray,
    window: tuple[int, int, int, int],
    target_box: Sequence[float],
    others: Sequence[tuple[int, Sequence[float]]],
    *,
    dilate: float = ISOLATE_DILATE,
    same_person_iou: float = SAME_PERSON_IOU,
) -> tuple[np.ndarray, list[int]]:
    """Black out every bystander inside ``crop`` (painted in place).

    ``crop`` is the padded crop of ``target_box``; ``window`` is the
    frame-space ``(x1, y1, x2, y2)`` rect it was cut from (see
    :func:`_padded_crop_window`), which lets bystander ``others`` —
    ``(track_id, box)`` pairs in frame coordinates — be translated into
    crop coordinates and painted. Each box is dilated by ``dilate``
    (per side, as a fraction of its own size) so edges don't leak, then
    clamped to the crop and filled solid black. Black beats blur or
    mosaic here: it also destroys colour and silhouette cues, which a
    clothing-based prompt ("person in a red shirt") would otherwise
    still match on a blurred bystander.

    Boxes whose IoU with ``target_box`` reaches ``same_person_iou`` are
    left alone: they are most likely a tracker fragment of the subject
    itself, and painting them would erase the very person the snapshot
    exists for. Returns the crop (the same object, painted) and the
    painted track ids, sorted.
    """
    height, width = crop.shape[:2]
    wx1, wy1, wx2, wy2 = (int(v) for v in window)
    painted: list[int] = []
    # Paint every bystander, but never over the subject itself: a bystander
    # whose (dilated) box reaches the subject must not erase the very person
    # the snapshot exists for. This is what produced the all-black thumbnails
    # when two bystanders flanked the subject. Tracker fragments of the
    # subject (high IoU) are skipped entirely, as before.
    mask = np.zeros((height, width), dtype=bool)
    for track_id, box in others:
        if _box_iou(box, target_box) >= same_person_iou:
            continue
        x1, y1, x2, y2 = (float(v) for v in box)
        pad_x = max(2.0, (x2 - x1) * dilate)
        pad_y = max(2.0, (y2 - y1) * dilate)
        cx1 = max(0, max(wx1, math.floor(x1 - pad_x)) - wx1)
        cy1 = max(0, max(wy1, math.floor(y1 - pad_y)) - wy1)
        cx2 = min(width, min(wx2, math.ceil(x2 + pad_x)) - wx1)
        cy2 = min(height, min(wy2, math.ceil(y2 + pad_y)) - wy1)
        if cx2 <= cx1 or cy2 <= cy1:
            continue  # this bystander is outside the crop window
        mask[cy1:cy2, cx1:cx2] = True
        painted.append(track_id)
    # Protect the subject's own region from every bystander, including the
    # dilation halo around them.
    tx1 = max(0, math.floor(float(target_box[0])) - wx1)
    ty1 = max(0, math.floor(float(target_box[1])) - wy1)
    tx2 = min(width, math.ceil(float(target_box[2])) - wx1)
    ty2 = min(height, math.ceil(float(target_box[3])) - wy1)
    if tx2 > tx1 and ty2 > ty1:
        mask[ty1:ty2, tx1:tx2] = False
    crop[mask] = 0
    return crop, sorted(painted)


def sub_isolate_people(
    crop: np.ndarray,
    window: tuple[int, int, int, int],
    target_box: Sequence[float],
    model: Any,
    *,
    dilate: float = ISOLATE_DILATE,
    conf: float = 0.25,
) -> int:
    """Black out extra people detected inside ``crop`` (beyond the subject).

    Runs a person detector (``model``) on the already-cropped thumbnail and,
    when more than one person is found, keeps the detection that best overlaps
    the subject's box and paints the others solid black. This recovers cases
    where the original tracker merged two people into a single tracklet, so
    there was no separate tracklet to isolate. Returns the number of extra
    people painted (0 when the detector finds at most one person or is
    unavailable).
    """
    if model is None:
        return 0
    try:
        results = model(crop, verbose=False)
    except Exception:
        return 0
    boxes: list[list[float]] = []
    for result in results:
        if getattr(result, "boxes", None) is None:
            continue
        try:
            xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
        except Exception:
            try:
                xyxy = np.asarray(result.boxes.xyxy)
                confs = np.asarray(result.boxes.conf)
            except Exception:
                continue
        for i in range(len(xyxy)):
            if i < len(confs) and float(confs[i]) < conf:
                continue
            boxes.append([float(v) for v in xyxy[i]])
    if len(boxes) <= 1:
        return 0

    height, width = crop.shape[:2]
    wx1, wy1, _, _ = (int(v) for v in window)
    target_crop = [
        float(target_box[0]) - wx1,
        float(target_box[1]) - wy1,
        float(target_box[2]) - wx1,
        float(target_box[3]) - wy1,
    ]
    best_idx = max(range(len(boxes)), key=lambda i: _box_iou(boxes[i], target_crop))

    mask = np.zeros((height, width), dtype=bool)
    painted = 0
    for i, box in enumerate(boxes):
        if i == best_idx:
            continue
        x1, y1, x2, y2 = (float(v) for v in box)
        pad_x = max(2.0, (x2 - x1) * dilate)
        pad_y = max(2.0, (y2 - y1) * dilate)
        cx1 = max(0, math.floor(x1 - pad_x))
        cy1 = max(0, math.floor(y1 - pad_y))
        cx2 = min(width, math.ceil(x2 + pad_x))
        cy2 = min(height, math.ceil(y2 + pad_y))
        if cx2 <= cx1 or cy2 <= cy1:
            continue
        mask[cy1:cy2, cx1:cx2] = True
        painted += 1
    # Never paint over the subject itself.
    tx1 = max(0, math.floor(float(target_box[0])) - wx1)
    ty1 = max(0, math.floor(float(target_box[1])) - wy1)
    tx2 = min(width, math.ceil(float(target_box[2])) - wx1)
    ty2 = min(height, math.ceil(float(target_box[3])) - wy1)
    if tx2 > tx1 and ty2 > ty1:
        mask[ty1:ty2, tx1:tx2] = False
    crop[mask] = 0
    return painted


    crop[mask] = 0
    return crop, sorted(painted)


def select_snapshots(
    meta: Metadata,
    thumbs_dir: str | Path = DEFAULT_THUMBS_DIR,
    *,
    padding: float = DEFAULT_PADDING,
    isolate: bool = True,
    sub_isolate: bool = False,
    closeup_area_fraction: float = CLOSEUP_AREA_FRACTION,
    head_weight: float = HEAD_WEIGHT,
    body_weight: float = BODY_WEIGHT,
    edge_penalty: float = EDGE_PENALTY,
    edge_tolerance: float = EDGE_TOLERANCE,
    distractor_weight: float = DISTRACTOR_WEIGHT,
    same_person_iou: float = SAME_PERSON_IOU,
    isolate_dilate: float = ISOLATE_DILATE,
    contact_sheet: str | Path | None = DEFAULT_CONTACT_SHEET,
    on_progress: Callable[[int, int], None] | None = None,
    resume: bool = True,
    meta_path=None,
    save_path=None,
    checkpoint_path=None,
    checkpoint=None,
    output=None,
    out=None,
    path=None,
    file=None,
    destination=None,
    dest=None,
    **kwargs,
) -> Metadata:
    """Give every tracklet its best-frame thumbnail and record it in ``meta``.

    For each tracklet: pick the largest bbox (:func:`best_observation`),
    decode that frame from ``meta.video.path``, crop it
    (:func:`crop_snapshot`), black out the other people visible in the
    crop (:func:`redact_others`, unless ``isolate`` is false), and save
    ``<thumbs_dir>/track_<id>.jpg``. The tracklet's ``best_frame`` /
    ``best_snapshot_path`` are updated in place (and the returned
    ``meta`` is the same object, ready for
    :func:`auto_redact.meta.save_meta`).

    Args:
        meta: Tracklet metadata whose ``video.path`` is read as the source.
        thumbs_dir: Directory for the thumbnails (created if missing).
        padding: Crop margin around each bbox, per side, as a fraction of
            the bbox size (must be a finite number >= 0).
        isolate: Black out bystanders in each thumbnail so the classifier
            can only judge the tracklet's own subject; their track ids
            land in ``Tracklet.isolated_track_ids``. ``False`` keeps the
            raw crop.
        contact_sheet: Contact sheet filename; a relative path resolves
            inside ``thumbs_dir``. ``None`` skips the contact sheet.
        on_progress: Optional ``(done, total)`` callback, in decoded frames
            (the scan stops once the last needed frame is reached).

    Raises:
        FileNotFoundError: the source video is missing.
        SnapshotError: invalid padding, a tracklet without observations
            or usable boxes, or a needed frame absent from the video.
    """
    _check_padding(padding)
    # Collect any extra path-like keyword.
    for key, value in list(kwargs.items()):
        if value is None:
            continue
        low = key.lower()
        if isinstance(value, (str, Path, os.PathLike)) and (
            "meta" in low or "path" in low or "out" in low
            or "dest" in low or "save" in low or "checkpoint" in low
        ):
            if meta_path is None and save_path is None and checkpoint_path is None and output is None and out is None and path is None:
                meta_path = value
    target = _resolve_meta_path(
        meta_path=meta_path,
        save_path=save_path,
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
        output=output,
        out=out,
        path=path,
        file=file,
        destination=destination,
        dest=dest,
    )

    if not resume:
        for tracklet in meta.tracklets:
            tracklet.best_frame = None
            tracklet.best_snapshot_path = None
            tracklet.isolated_track_ids = []

    source = Path(meta.video.path)
    if not source.is_file():
        raise FileNotFoundError(f"no such video file: {source}")

    ckpt_path = (
        target.with_name(target.name + ".snapshot.ckpt.json")
        if target is not None else None
    )

    width, height = meta.video.width, meta.video.height

    def _already_done(tracklet: Tracklet) -> bool:
        return (
            resume
            and tracklet.best_snapshot_path is not None
            and Path(tracklet.best_snapshot_path).is_file()
        )

    # Resume from an incremental checkpoint (if any) so a crash mid-stage
    # only loses the frames since the last checkpoint, not the whole stage.
    if resume and ckpt_path is not None and ckpt_path.is_file():
        try:
            records = json.loads(ckpt_path.read_text(encoding="utf-8")).get(
                "completed", []
            )
            by_id = {t.track_id: t for t in meta.tracklets}
            for record in records:
                tracklet = by_id.get(record.get("track_id"))
                if tracklet is None or tracklet.best_snapshot_path is not None:
                    continue
                tracklet.best_frame = record.get("best_frame")
                tracklet.best_snapshot_path = record.get("best_snapshot_path")
                tracklet.isolated_track_ids = list(
                    record.get("isolated_track_ids") or []
                )
        except (OSError, ValueError, TypeError):
            pass

    to_process = [t for t in meta.tracklets if not _already_done(t)]
    pre_done = [t for t in meta.tracklets if _already_done(t)]

    # frame_idx -> tracklets whose best snapshot lives on that frame
    by_frame: dict[int, list[tuple[Tracklet, list[float]]]] = {}
    best_frames: dict[int, int] = {}
    frame_boxes = index_boxes_by_frame(meta.tracklets)
    for tracklet in to_process:
        frame_idx, box = best_observation(
            tracklet, width, height, frame_boxes=frame_boxes, padding=padding,
            head_weight=head_weight, body_weight=body_weight,
            edge_penalty=edge_penalty, edge_tolerance=edge_tolerance,
            distractor_weight=distractor_weight,
        )
        by_frame.setdefault(frame_idx, []).append((tracklet, box))
        best_frames[tracklet.track_id] = frame_idx

    # Optional second-stage detector for merged tracklets (see
    # :func:`sub_isolate_people`). Loaded lazily so the dependency is only
    # needed when the feature is requested.
    sub_model = None
    if sub_isolate:
        try:
            sub_model = _load_yolo("cpu")
        except Exception:
            sub_model = None

    # Each crop is written to disk as soon as its frame is decoded, and only a
    # cell-sized copy is kept for the contact sheet: holding every full-size
    # crop at once needs gigabytes on a long video.
    thumbs = Path(thumbs_dir)
    if meta.tracklets:
        thumbs.mkdir(parents=True, exist_ok=True)
    saved: list[tuple[int, Image.Image]] = []
    if to_process:
        pending = set(by_frame)
        last_needed = max(by_frame, default=-1) + 1
        save_counter = 0

        def _process_frame(frame_idx: int, frame: np.ndarray) -> None:
            nonlocal save_counter
            if on_progress is not None:
                on_progress(frame_idx + 1, last_needed)
            group = by_frame.get(frame_idx, ())
            if group:
                for tracklet, box in group:
                    crop = crop_snapshot(frame, box, padding)
                    tracklet.isolated_track_ids = []
                    # Close-ups dominate the frame: isolating bystanders paints
                    # large black bars that confuse the classifier, and the
                    # subject already dominates the crop, so skip isolation
                    # (and sub-isolation) for them.
                    _area_fraction = _clamped_area(box, width, height) / max(
                        1.0, float(width) * float(height)
                    )
                    _closeup = _area_fraction >= closeup_area_fraction
                    if isolate and not _closeup:
                        crop, painted = redact_others(
                            crop,
                            _padded_crop_window(box, width, height, padding),
                            box,
                            [
                                (other_id, other_box)
                                for other_id, other_box in frame_boxes.get(frame_idx, ())
                                if other_id != tracklet.track_id
                            ],
                            same_person_iou=same_person_iou,
                            dilate=isolate_dilate,
                        )
                        tracklet.isolated_track_ids = painted
                    if sub_isolate and sub_model is not None and not _closeup:
                        sub_isolate_people(
                            crop,
                            _padded_crop_window(box, width, height, padding),
                            box,
                            sub_model,
                            dilate=isolate_dilate,
                        )
                    image = Image.fromarray(crop)
                    thumb_path = thumbs / f"track_{tracklet.track_id}.jpg"
                    image.save(thumb_path, format="JPEG", quality=JPEG_QUALITY)
                    tracklet.best_frame = best_frames[tracklet.track_id]
                    tracklet.best_snapshot_path = str(thumb_path)
                    image.thumbnail(CONTACT_SHEET_CELL, Image.Resampling.LANCZOS)
                    saved.append((tracklet.track_id, image))
                # Persist progress to a small incremental checkpoint (only
                # the completed tracklets) every 10 distinct frames.
                save_counter += 1
                if ckpt_path is not None and save_counter % 10 == 0:
                    try:
                        completed = [
                            {
                                "track_id": t.track_id,
                                "best_frame": t.best_frame,
                                "best_snapshot_path": t.best_snapshot_path,
                                "isolated_track_ids": list(t.isolated_track_ids),
                            }
                            for t in meta.tracklets
                            if t.best_snapshot_path is not None
                        ]
                        ckpt_path.write_text(
                            json.dumps({"completed": completed}, separators=(",", ":")),
                            encoding="utf-8",
                        )
                    except Exception:
                        pass

        total_frames = meta.video.frames
        distinct_frames = sorted(by_frame)
        use_seek = len(distinct_frames) <= max(10, total_frames // 500)

        if use_seek:
            for frame in distinct_frames:
                found = False
                for frame_idx, decoded in iter_frames(
                    source, start=frame, stop=frame + 1
                ):
                    found = True
                    _process_frame(frame_idx, decoded)
                    pending.discard(frame)
                    pending.discard(frame_idx)
                if not found:
                    pass
        else:
            for frame_idx, frame in iter_frames(source):
                _process_frame(frame_idx, frame)
                pending.discard(frame_idx)
                if not pending:
                    break
        if target is not None:
            try:
                save_meta(target, meta)
            except Exception:
                pass
        if ckpt_path is not None:
            try:
                ckpt_path.unlink(missing_ok=True)
            except Exception:
                pass
        if pending:
            raise SnapshotError(
                f"{source}: frame(s) {sorted(pending)} not found; "
                f"the video has fewer frames than the metadata claims"
            )

    # Include already-completed thumbnails so the contact sheet still covers
    # every tracklet when resuming.
    for tracklet in pre_done:
        if tracklet.best_snapshot_path is not None:
            p = Path(tracklet.best_snapshot_path)
            try:
                img = Image.open(p).convert("RGB")
                img.thumbnail(CONTACT_SHEET_CELL, Image.Resampling.LANCZOS)
                saved.append((tracklet.track_id, img))
            except Exception:
                pass
    saved.sort(key=lambda item: item[0])

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


def _padded_crop_window(
    box: Sequence[float], width: int, height: int, padding: float
) -> tuple[int, int, int, int]:
    """Padded ``box`` as a clamped integer crop window ``(x1, y1, x2, y2)``.

    Frame coordinates, the rect :func:`crop_snapshot` cuts out. Not
    validating: callers must have checked the box is finite (and the
    window non-empty where that matters).
    """
    x1, y1, x2, y2 = (float(v) for v in box)
    pad_x = (x2 - x1) * padding
    pad_y = (y2 - y1) * padding
    return (
        max(0, min(width, math.floor(x1 - pad_x))),
        max(0, min(height, math.floor(y1 - pad_y))),
        max(0, min(width, math.ceil(x2 + pad_x))),
        max(0, min(height, math.ceil(y2 + pad_y))),
    )


def _distractor_fraction(
    box: Sequence[float],
    other_boxes: Sequence[Sequence[float]],
    width: int,
    height: int,
    padding: float,
    same_person_iou: float = SAME_PERSON_IOU,
) -> float:
    """Share of the padded crop window of ``box`` intruded on by others.

    Bystanders overlapping each other may double-count; the result is
    capped at 1. Boxes mostly coinciding with ``box``
    (``SAME_PERSON_IOU``) are the subject, not distractors. Degenerate
    boxes score 0.
    """
    coords = [float(v) for v in box]
    if not all(math.isfinite(coord) for coord in coords):
        return 0.0
    window = _padded_crop_window(coords, width, height, padding)
    window_area = (window[2] - window[0]) * (window[3] - window[1])
    if window_area <= 0:
        return 0.0
    covered = 0.0
    for other in other_boxes:
        if _box_iou(coords, other) >= same_person_iou:
            continue
        covered += _clamped_area(
            [max(window[0], other[0]), max(window[1], other[1]),
             min(window[2], other[2]), min(window[3], other[3])],
            width,
            height,
        )
    return min(1.0, covered / window_area)


def _box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection-over-union of two boxes (0 when either is empty)."""
    inter_w = max(0.0, min(float(a[2]), float(b[2])) - max(float(a[0]), float(b[0])))
    inter_h = max(0.0, min(float(a[3]), float(b[3])) - max(float(a[1]), float(b[1])))
    inter = inter_w * inter_h
    union = (
        max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
        + max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
        - inter
    )
    return inter / union if union > 0 else 0.0


def _clamped_area(box: list[float], width: int, height: int) -> float:
    """Visible area of ``box`` after clamping it to the frame, in px²."""
    x1, y1, x2, y2 = (float(v) for v in box)
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _clamp01(value: float) -> float:
    value = float(value)
    return min(1.0, max(0.0, value)) if math.isfinite(value) else 0.0


def _overlap_fraction(
    box: Sequence[float], other: Sequence[float], width: int, height: int
) -> float:
    """Fraction of ``box``'s visible area covered by ``other``."""
    own = _clamped_area(list(box), width, height)
    if own <= 0:
        return 0.0
    inter = _clamped_area(
        [max(box[0], other[0]), max(box[1], other[1]),
         min(box[2], other[2]), min(box[3], other[3])],
        width,
        height,
    )
    return min(1.0, inter / own)


def _check_padding(padding: float) -> None:
    if isinstance(padding, bool) or not isinstance(padding, (int, float)):
        raise SnapshotError(f"padding must be a number, got {padding!r}")
    if not math.isfinite(float(padding)) or padding < 0:
        raise SnapshotError(f"padding must be finite and >= 0, got {padding!r}")
