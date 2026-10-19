"""Metadata schema v1: save/load JSON (video info + tracklets).

The metadata document is the hand-off between pipeline stages: the
tracking pass writes it with :func:`save_meta`, and later stages read it
back with :func:`load_meta` so classification and rendering can re-run
without re-tracking (the CLI's ``--meta`` flag points at this file).
Schema version 1::

    {
      "version": 1,
      "video": {"path": "...", "fps": 30, "width": 1280, "height": 720, "frames": 900},
      "tracklets": [
        {"track_id": 1,
         "frames": [12, 13],
         "boxes": [[x1, y1, x2, y2], ...],
         "keypoints": [[[x, y, conf], ...], ...],
         "best_frame": 34,
         "best_snapshot_path": "thumbs/track_1.jpg",
         "redact": null}
      ]
    }

``frames``, ``boxes`` and ``keypoints`` are aligned by index: one entry
per observed frame, each frame's keypoints a fixed-length list of
``[x, y, confidence]`` triples (17 for COCO-pose models). ``best_frame``
and ``best_snapshot_path`` are filled in by snapshot selection (#12) and
``redact`` by classification (#13); all three are null until then.
Documents with an unsupported version or any structural defect raise
:class:`MetaError` rather than loading half-trusted data.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from numbers import Integral, Real
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


class MetaError(ValueError):
    """Raised when a metadata document is invalid or unsupported."""


def _type_name(value: Any) -> str:
    return type(value).__name__


def _require_object(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MetaError(f"{where}: expected object, got {_type_name(value)}")
    return value


def _check_keys(
    data: dict[str, Any],
    required: set[str],
    optional: set[str],
    where: str,
) -> None:
    missing = sorted(key for key in required if key not in data)
    if missing:
        raise MetaError(f"{where}: missing key(s): {', '.join(missing)}")
    unknown = sorted(
        str(key) for key in data if key not in required and key not in optional
    )
    if unknown:
        raise MetaError(f"{where}: unknown key(s): {', '.join(unknown)}")


def _as_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise MetaError(f"{where}: expected integer, got {_type_name(value)}")
    return int(value)


def _as_non_negative_int(value: Any, where: str) -> int:
    result = _as_int(value, where)
    if result < 0:
        raise MetaError(f"{where}: expected a non-negative integer, got {result}")
    return result


def _as_float(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise MetaError(f"{where}: expected number, got {_type_name(value)}")
    result = float(value)
    if not math.isfinite(result):
        raise MetaError(f"{where}: expected a finite number, got {result!r}")
    return result


def _as_positive_float(value: Any, where: str) -> float:
    result = _as_float(value, where)
    if result <= 0:
        raise MetaError(f"{where}: expected a positive number, got {result}")
    return result


def _as_positive_int(value: Any, where: str) -> int:
    result = _as_int(value, where)
    if result <= 0:
        raise MetaError(f"{where}: expected a positive integer, got {result}")
    return result


def _as_str(value: Any, where: str) -> str:
    if not isinstance(value, str):
        raise MetaError(f"{where}: expected string, got {_type_name(value)}")
    return value


def _as_bool_or_none(value: Any, where: str) -> bool | None:
    if value is not None and not isinstance(value, bool):
        raise MetaError(f"{where}: expected boolean or null, got {_type_name(value)}")
    return value


def _as_sequence(value: Any, where: str) -> list | tuple:
    tolist = getattr(value, "tolist", None)  # numpy/torch output from #9/#10
    if callable(tolist):
        value = tolist()
    if not isinstance(value, (list, tuple)):
        raise MetaError(f"{where}: expected array, got {_type_name(value)}")
    return value


def _as_box(value: Any, where: str) -> list[float]:
    seq = _as_sequence(value, where)
    if len(seq) != 4:
        raise MetaError(
            f"{where}: expected 4 coordinates [x1, y1, x2, y2], got {len(seq)}"
        )
    return [_as_float(coord, f"{where}[{i}]") for i, coord in enumerate(seq)]


def _as_keypoint(value: Any, where: str) -> list[float]:
    seq = _as_sequence(value, where)
    if len(seq) != 3:
        raise MetaError(
            f"{where}: expected [x, y, confidence], got {len(seq)} values"
        )
    return [_as_float(component, f"{where}[{i}]") for i, component in enumerate(seq)]


def _as_keypoints(value: Any, where: str) -> list[list[float]]:
    seq = _as_sequence(value, where)
    return [_as_keypoint(keypoint, f"{where}[{i}]") for i, keypoint in enumerate(seq)]


def _check_aligned(
    frames: list[int],
    boxes: list[list[float]],
    keypoints: list[list[list[float]]],
    where: str,
) -> None:
    if not (len(frames) == len(boxes) == len(keypoints)):
        raise MetaError(
            f"{where}: frames, boxes and keypoints must be aligned: got "
            f"{len(frames)} frames, {len(boxes)} boxes, "
            f"{len(keypoints)} keypoint sets"
        )
    counts = {len(frame) for frame in keypoints}
    if len(counts) > 1:
        raise MetaError(
            f"{where}: keypoints per frame must match, got {sorted(counts)}"
        )


@dataclass
class VideoInfo:
    """Probe results for the input video (issue #8)."""

    path: str
    fps: float
    width: int
    height: int
    frames: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": _as_str(self.path, "video.path"),
            "fps": _as_positive_float(self.fps, "video.fps"),
            "width": _as_positive_int(self.width, "video.width"),
            "height": _as_positive_int(self.height, "video.height"),
            "frames": _as_positive_int(self.frames, "video.frames"),
        }

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

    @classmethod
    def from_dict(cls, data: Any) -> VideoInfo:
        data = _require_object(data, "video")
        _check_keys(
            data,
            required={"path", "fps", "width", "height", "frames"},
            optional=set(),
            where="video",
        )
        # Keys were checked above, so data holds exactly the dataclass fields;
        # to_dict() validates and coerces the raw JSON values.
        raw = cls(**data)
        return cls(**raw.to_dict())


@dataclass
class Tracklet:
    """One tracked person: per-frame boxes and keypoints (issue #10).

    ``best_frame`` / ``best_snapshot_path`` are set by snapshot selection
    (#12) and ``redact`` by classification (#13); both start as null.
    """

    track_id: int
    frames: list[int]
    boxes: list[list[float]]
    keypoints: list[list[list[float]]]
    best_frame: int | None = None
    best_snapshot_path: str | None = None
    redact: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        track_id = _as_int(self.track_id, "tracklet.track_id")
        where = f"tracklet {track_id}"
        frames = [
            _as_non_negative_int(frame, f"{where}.frames[{i}]")
            for i, frame in enumerate(_as_sequence(self.frames, f"{where}.frames"))
        ]
        boxes = [
            _as_box(box, f"{where}.boxes[{i}]")
            for i, box in enumerate(_as_sequence(self.boxes, f"{where}.boxes"))
        ]
        keypoints = [
            _as_keypoints(frame_keypoints, f"{where}.keypoints[{i}]")
            for i, frame_keypoints in enumerate(
                _as_sequence(self.keypoints, f"{where}.keypoints")
            )
        ]
        _check_aligned(frames, boxes, keypoints, where)
        return {
            "track_id": track_id,
            "frames": frames,
            "boxes": boxes,
            "keypoints": keypoints,
            "best_frame": (
                None
                if self.best_frame is None
                else _as_non_negative_int(self.best_frame, f"{where}.best_frame")
            ),
            "best_snapshot_path": (
                None
                if self.best_snapshot_path is None
                else _as_str(self.best_snapshot_path, f"{where}.best_snapshot_path")
            ),
            "redact": _as_bool_or_none(self.redact, f"{where}.redact"),
        }

    @classmethod
    def from_dict(cls, data: Any) -> Tracklet:
        data = _require_object(data, "tracklet")
        _check_keys(
            data,
            required={"track_id", "frames", "boxes", "keypoints"},
            optional={"best_frame", "best_snapshot_path", "redact"},
            where="tracklet",
        )
        raw = cls(
            track_id=data["track_id"],
            frames=data["frames"],
            boxes=data["boxes"],
            keypoints=data["keypoints"],
            best_frame=data.get("best_frame"),
            best_snapshot_path=data.get("best_snapshot_path"),
            redact=data.get("redact"),
        )
        return cls(**raw.to_dict())


@dataclass
class Metadata:
    """The whole metadata document: video info + tracklets."""

    video: VideoInfo
    tracklets: list[Tracklet] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": SCHEMA_VERSION,
            "video": self.video.to_dict(),
            "tracklets": [
                tracklet.to_dict()
                for tracklet in _as_sequence(self.tracklets, "tracklets")
            ],
        }

    @classmethod
    def from_dict(cls, data: Any) -> Metadata:
        data = _require_object(data, "metadata")
        _check_keys(
            data,
            required={"version", "video", "tracklets"},
            optional=set(),
            where="metadata",
        )
        version = _as_int(data["version"], "metadata.version")
        if version != SCHEMA_VERSION:
            raise MetaError(
                f"metadata.version: unsupported schema version {version}, "
                f"expected {SCHEMA_VERSION}"
            )
        tracklets = _as_sequence(data["tracklets"], "tracklets")
        return cls(
            video=VideoInfo.from_dict(data["video"]),
            tracklets=[Tracklet.from_dict(tracklet) for tracklet in tracklets],
        )


def save_meta(path: str | Path, meta: Metadata) -> Path:
    """Write ``meta`` to ``path`` as schema v1 JSON, creating parents.

    The document is validated and coerced to JSON-safe types before
    writing, so anything that can be saved can be loaded back identically.
    """
    if not isinstance(meta, Metadata):
        raise MetaError(f"expected a Metadata instance, got {_type_name(meta)}")
    payload = meta.to_dict()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return destination


def load_meta(path: str | Path) -> Metadata:
    """Load and validate a metadata document written by :func:`save_meta`."""
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise MetaError(f"{source}: not valid JSON: {error}") from error
    return Metadata.from_dict(data)
