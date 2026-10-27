"""Metadata: the hand-off between pipeline stages (schema v2).

The tracking pass writes a metadata file with :func:`save_meta`; later
stages read it back with :func:`load_meta`, so classification and rendering
can re-run without re-tracking (the CLI's ``--meta`` flag points at it).

On disk the document is split in two:

``<name>.json``
    A small, human-readable (and hand-editable) summary: video info, scene
    cuts and one record per tracklet with its snapshot, isolation and
    classification fields (``redact`` is what ``render`` acts on)::

        {"version": 2,
         "video": {"path": "...", "fps": 30, "width": 1280, "height": 720, "frames": 900},
         "scene_cuts": [], "scene_cut_splits": 0,
         "arrays": "<name>.npz",
         "tracklets": [{"track_id": 1, "observations": 3, "best_frame": 13,
                        "best_snapshot_path": "thumbs/track_1.jpg",
                        "isolated_track_ids": [2], "redact": null, "score": null}]}

``<name>.npz``
    The bulk per-frame data, as compressed numpy arrays per tracklet:
    ``t<id>_frames`` (n,), ``t<id>_boxes`` (n, 4) as ``[x1, y1, x2, y2]`` and
    ``t<id>_keypoints`` (n, k, 3) as ``[x, y, confidence]`` triples (k = 17
    for COCO-pose models).

Schema v1 documents (everything inline in one JSON file) still load.
Documents with an unsupported version or any structural defect raise
:class:`MetaError` rather than loading half-trusted data. The in-memory
types below are plain dataclasses; validation lives at the file boundary.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

import msgspec
import numpy as np

SCHEMA_VERSION = 2
LEGACY_VERSIONS = (1,)

_Positive = msgspec.Meta(gt=0)
_NonNegative = msgspec.Meta(ge=0)


class MetaError(ValueError):
    """Raised when a metadata document is invalid or unsupported."""


@dataclass
class VideoInfo:
    """Probe results for the input video (issue #8)."""

    path: str
    fps: float
    width: int
    height: int
    frames: int

    @classmethod
    def from_probe(cls, probe: Any) -> VideoInfo:
        """Convert a :func:`auto_redact.frames.probe_video` result.

        Duck-typed on ``path``/``fps``/``width``/``height``/
        ``frame_count`` so importing this schema module never pulls in
        PyAV, and names map to the on-disk schema (``frame_count`` ->
        ``frames``).
        """
        required = ("path", "fps", "width", "height", "frame_count")
        missing = [name for name in required if not hasattr(probe, name)]
        if missing:
            raise MetaError(
                f"probe result is missing attribute(s): {', '.join(missing)}"
            )
        return cls(
            path=str(probe.path),
            fps=probe.fps,
            width=probe.width,
            height=probe.height,
            frames=probe.frame_count,
        )


@dataclass
class Tracklet:
    """One tracked person: per-frame boxes and keypoints (issue #10).

    ``frames``, ``boxes`` and ``keypoints`` are aligned by index.
    ``best_frame`` / ``best_snapshot_path`` are set by snapshot selection
    (#12), ``isolated_track_ids`` records whose pixels were blacked out of
    the snapshot, and ``redact`` / ``score`` / ``detail`` (the classifier's
    reason, e.g. the VLM reply) by classification (#13); ``override`` is a
    human correction that rendering prefers and re-classification never
    touches. All start empty/null.
    """

    track_id: int
    frames: list[int]
    boxes: list[list[float]]
    keypoints: list[list[list[float]]]
    best_frame: int | None = None
    best_snapshot_path: str | None = None
    isolated_track_ids: list[int] = field(default_factory=list)
    redact: bool | None = None
    score: float | None = None
    detail: str | None = None
    override: bool | None = None

    @property
    def effective_redact(self) -> bool | None:
        """The decision rendering acts on: a human ``override`` beats ``redact``."""
        return self.override if self.override is not None else self.redact


@dataclass
class Metadata:
    """The whole metadata document: video info, tracklets, scene cuts."""

    video: VideoInfo
    tracklets: list[Tracklet] = field(default_factory=list)
    scene_cuts: list[int] = field(default_factory=list)
    scene_cut_splits: int = 0


# --- document shape (what the JSON file may contain) -------------------------


class _VideoDoc(msgspec.Struct, forbid_unknown_fields=True):
    path: str
    fps: Annotated[float, _Positive]
    width: Annotated[int, _Positive]
    height: Annotated[int, _Positive]
    frames: Annotated[int, _Positive]


class _TrackletDoc(msgspec.Struct, forbid_unknown_fields=True):
    track_id: int
    best_frame: Annotated[int, _NonNegative] | None = None
    best_snapshot_path: str | None = None
    isolated_track_ids: list[Annotated[int, _NonNegative]] = []
    redact: bool | None = None
    score: float | None = None
    detail: str | None = None
    override: bool | None = None
    observations: Annotated[int, _NonNegative] | None = None
    # Schema v1 only: the arrays inline instead of in the .npz file.
    frames: list[Any] | None = None
    boxes: list[Any] | None = None
    keypoints: list[Any] | None = None


class _MetaDoc(msgspec.Struct, forbid_unknown_fields=True):
    version: int
    video: _VideoDoc
    tracklets: list[_TrackletDoc]
    scene_cuts: list[Annotated[int, _NonNegative]] = []
    scene_cut_splits: Annotated[int, _NonNegative] = 0
    arrays: str | None = None


def _plain(value: Any) -> Any:
    """numpy scalars (from detectors) to plain Python numbers."""
    return value.item() if isinstance(value, np.generic) else value


def _decode(data: Any, where: str) -> _MetaDoc:
    if not isinstance(data, dict):
        raise MetaError(f"{where}: expected object, got {type(data).__name__}")
    if "version" not in data:
        raise MetaError(f"{where}: missing key(s): version")
    version = data["version"]
    if isinstance(version, bool) or not isinstance(version, int):
        raise MetaError(
            f"{where}.version: expected integer, got {type(version).__name__}"
        )
    if version != SCHEMA_VERSION and version not in LEGACY_VERSIONS:
        raise MetaError(
            f"{where}.version: unsupported schema version {version}, "
            f"expected {SCHEMA_VERSION}"
        )
    try:
        return msgspec.convert(data, _MetaDoc)
    except msgspec.ValidationError as error:
        raise MetaError(f"{where}: {error}") from error


# --- arrays -------------------------------------------------------------------


def _arrays(tracklet: Tracklet) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate one tracklet's per-frame data and convert it to arrays."""
    where = f"tracklet {tracklet.track_id}"
    raw_frames, raw_boxes, raw_keypoints = (
        value.tolist() if hasattr(value, "tolist") else value
        for value in (tracklet.frames, tracklet.boxes, tracklet.keypoints)
    )
    count = len(raw_frames)
    if not (len(raw_boxes) == count == len(raw_keypoints)):
        raise MetaError(
            f"{where}: frames, boxes and keypoints must be aligned: got "
            f"{count} frames, {len(raw_boxes)} boxes, "
            f"{len(raw_keypoints)} keypoint sets"
        )

    frames = np.asarray(raw_frames) if count else np.zeros(0, dtype=np.int64)
    if frames.ndim != 1 or not np.issubdtype(frames.dtype, np.integer):
        raise MetaError(f"{where}.frames: expected a list of integers")
    if (frames < 0).any():
        raise MetaError(f"{where}.frames: expected non-negative integer indexes")

    try:
        boxes = np.asarray(raw_boxes, dtype=np.float64).reshape(count, -1)
    except (TypeError, ValueError):
        boxes = None
    if boxes is None or boxes.shape != (count, 4):
        raise MetaError(
            f"{where}.boxes: expected 4 coordinates [x1, y1, x2, y2] per box"
        )

    try:
        keypoints = np.asarray(raw_keypoints, dtype=np.float64)
    except ValueError:
        counts = sorted({len(frame) for frame in raw_keypoints})
        raise MetaError(
            f"{where}: keypoints per frame must match, got {counts}"
        ) from None
    except TypeError:
        keypoints = None
    if keypoints is not None and keypoints.size == 0:
        keypoints = keypoints.reshape(count, 0, 3)
    if keypoints is None or keypoints.ndim != 3 or keypoints.shape[2] != 3:
        raise MetaError(
            f"{where}.keypoints: expected [x, y, confidence] per keypoint"
        )

    if not (np.isfinite(boxes).all() and np.isfinite(keypoints).all()):
        raise MetaError(f"{where}: boxes and keypoints must be finite numbers")
    return frames, boxes, keypoints


def _array_path(path: Path) -> Path:
    return path.with_suffix(".npz")


def _write_atomic(path: Path, write) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


# --- public API ---------------------------------------------------------------


def _document(meta: Metadata, arrays_file: Path) -> dict[str, Any]:
    """The validated JSON summary for ``meta`` (arrays are checked elsewhere)."""
    try:
        document = msgspec.to_builtins(
            _MetaDoc(
                version=SCHEMA_VERSION,
                video=_VideoDoc(**{k: _plain(v) for k, v in vars(meta.video).items()}),
                tracklets=[
                    _TrackletDoc(
                        track_id=_plain(t.track_id),
                        best_frame=_plain(t.best_frame),
                        best_snapshot_path=t.best_snapshot_path,
                        isolated_track_ids=[_plain(i) for i in t.isolated_track_ids],
                        redact=_plain(t.redact),
                        score=_plain(t.score),
                        detail=t.detail,
                        override=_plain(t.override),
                        observations=len(t.frames),
                    )
                    for t in meta.tracklets
                ],
                scene_cuts=[_plain(c) for c in meta.scene_cuts],
                scene_cut_splits=_plain(meta.scene_cut_splits),
                arrays=arrays_file.name,
            )
        )
        for record in document["tracklets"]:
            for legacy in ("frames", "boxes", "keypoints"):
                record.pop(legacy, None)
        # Run the document through the strict decoder so bad values (a
        # string fps, a non-positive size, ...) are rejected before
        # anything is written.
        _decode(document, "metadata")
    except (TypeError, msgspec.ValidationError) as error:
        raise MetaError(f"metadata: {error}") from error
    return document


def _write_summary(destination: Path, document: dict[str, Any]) -> None:
    text = json.dumps(document, indent=2) + "\n"
    _write_atomic(destination, lambda tmp: tmp.write_text(text, encoding="utf-8"))


def save_meta(path: str | Path, meta: Metadata) -> Path:
    """Validate ``meta`` and write it to ``path`` (JSON) plus a sibling ``.npz``.

    Everything is checked before anything is written, so a document that
    saves can be loaded back identically. Both files are replaced
    atomically (arrays first, so a crash never leaves a summary pointing
    at missing data).
    """
    if not isinstance(meta, Metadata):
        raise MetaError(f"expected a Metadata instance, got {type(meta).__name__}")
    destination = Path(path)
    arrays_file = _array_path(destination)

    arrays: dict[str, np.ndarray] = {}
    for tracklet in meta.tracklets:
        frames, boxes, keypoints = _arrays(tracklet)
        key = f"t{tracklet.track_id}"
        arrays[f"{key}_frames"] = frames
        arrays[f"{key}_boxes"] = boxes
        arrays[f"{key}_keypoints"] = keypoints

    document = _document(meta, arrays_file)
    _write_atomic(arrays_file, lambda tmp: _savez(tmp, arrays))
    _write_summary(destination, document)
    return destination


def save_meta_summary(path: str | Path, meta: Metadata) -> Path:
    """Rewrite only the JSON summary, for changes to the small per-tracklet fields.

    Decisions, overrides and snapshot paths live in the summary, so changing
    them does not need the (large) ``.npz`` rewritten. Raises
    :class:`MetaError` if the existing arrays file does not hold exactly the
    tracklets in ``meta`` (use :func:`save_meta` after adding, removing or
    re-tracking tracklets).
    """
    if not isinstance(meta, Metadata):
        raise MetaError(f"expected a Metadata instance, got {type(meta).__name__}")
    destination = Path(path)
    arrays_file = _array_path(destination)
    try:
        with np.load(arrays_file, allow_pickle=False) as store:
            stored = {name.rsplit("_", 1)[0] for name in store.files}
    except (OSError, ValueError) as error:
        raise MetaError(f"{arrays_file}: cannot update the summary without its arrays: {error}") from error
    if stored != {f"t{t.track_id}" for t in meta.tracklets}:
        raise MetaError(f"{arrays_file} does not match these tracklets; use save_meta")
    _write_summary(destination, _document(meta, arrays_file))
    return destination


def _savez(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with open(path, "wb") as handle:
        np.savez_compressed(handle, **arrays)


def load_meta(path: str | Path) -> Metadata:
    """Load and validate a metadata document (schema v2, or legacy v1)."""
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise MetaError(f"{source}: not valid JSON: {error}") from error
    return meta_from_document(data, base_dir=source.parent)


def meta_from_document(data: Any, base_dir: Path | None = None) -> Metadata:
    """Build a :class:`Metadata` from a decoded JSON document."""
    doc = _decode(data, "metadata")
    store = None
    if doc.arrays is not None:
        store = _open_arrays(Path(base_dir or ".") / doc.arrays)
    try:
        tracklets = [_tracklet_from_doc(t, store) for t in doc.tracklets]
    finally:
        if store is not None:
            store.close()
    return Metadata(
        video=VideoInfo(**{f: getattr(doc.video, f) for f in doc.video.__struct_fields__}),
        tracklets=tracklets,
        scene_cuts=list(doc.scene_cuts),
        scene_cut_splits=doc.scene_cut_splits,
    )


def _open_arrays(path: Path):
    try:
        return np.load(path, allow_pickle=False)
    except FileNotFoundError:
        raise MetaError(f"{path}: arrays file referenced by the metadata is missing") from None
    except (OSError, ValueError) as error:
        raise MetaError(f"{path}: unreadable arrays file: {error}") from error


def _tracklet_from_doc(doc: _TrackletDoc, store) -> Tracklet:
    inline = doc.frames is not None
    if store is not None:
        key = f"t{doc.track_id}"
        try:
            frames = store[f"{key}_frames"]
            boxes = store[f"{key}_boxes"]
            keypoints = store[f"{key}_keypoints"]
        except KeyError:
            raise MetaError(
                f"tracklet {doc.track_id}: no arrays stored for it in the arrays file"
            ) from None
        if doc.observations is not None and len(frames) != doc.observations:
            raise MetaError(
                f"tracklet {doc.track_id}: summary says {doc.observations} "
                f"observations but the arrays hold {len(frames)}"
            )
        raw: Tracklet | None = Tracklet(
            doc.track_id, frames.tolist(), boxes.tolist(), keypoints.tolist()
        )
    elif inline and doc.boxes is not None and doc.keypoints is not None:
        raw = Tracklet(doc.track_id, doc.frames, doc.boxes, doc.keypoints)
    else:
        raise MetaError(
            f"tracklet {doc.track_id}: missing frames, boxes and keypoints"
        )
    frames, boxes, keypoints = _arrays(raw)  # validates inline (v1) data too
    return Tracklet(
        track_id=doc.track_id,
        frames=frames.tolist(),
        boxes=boxes.tolist(),
        keypoints=keypoints.tolist(),
        best_frame=doc.best_frame,
        best_snapshot_path=doc.best_snapshot_path,
        isolated_track_ids=list(doc.isolated_track_ids),
        redact=doc.redact,
        score=doc.score,
        detail=doc.detail,
        override=doc.override,
    )
