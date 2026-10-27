"""Human review labels for one video: the feedback the tuner learns from.

Stored beside the metadata (``labels.json`` in the video's workspace) and
never touched by re-tracking or re-classification, so reviewing work is not
lost. Two kinds of feedback:

* **track labels**: "this person should / should not be redacted"
  (``redact`` true/false), with an optional note;
* **frame labels**: a spot where the output is wrong: ``leak`` (a person is
  visible without a black box, optionally with a ``box`` around them) or
  ``over`` (a black box is there that should not be).

``track_key`` is the tracking fingerprint (see :mod:`auto_redact.cache`) the
labels were made against. Track ids only mean something for that tracking,
so labels made for different tracking are reported as :meth:`Labels.stale`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal

import msgspec

VERSION = 1
KINDS = ("leak", "over")
_NonNegative = msgspec.Meta(ge=0)


class LabelsError(ValueError):
    """Raised when a labels file is invalid."""


class _TrackDoc(msgspec.Struct, forbid_unknown_fields=True):
    redact: bool
    note: str = ""


class _FrameDoc(msgspec.Struct, forbid_unknown_fields=True):
    frame: Annotated[int, _NonNegative]
    kind: Literal["leak", "over"]
    box: list[float] | None = None
    note: str = ""


class _LabelsDoc(msgspec.Struct, forbid_unknown_fields=True):
    version: int
    track_key: str | None = None
    tracks: dict[str, _TrackDoc] = {}
    frames: list[_FrameDoc] = []


@dataclass
class FrameLabel:
    frame: int
    kind: str
    box: list[float] | None = None
    note: str = ""


@dataclass
class TrackLabel:
    redact: bool
    note: str = ""


@dataclass
class Labels:
    track_key: str | None = None
    tracks: dict[int, TrackLabel] = field(default_factory=dict)
    frames: list[FrameLabel] = field(default_factory=list)

    def stale(self, current_track_key: str | None) -> bool:
        """True when the labels were made for different tracking."""
        return (
            self.track_key is not None
            and current_track_key is not None
            and self.track_key != current_track_key
        )

    def set_track(self, track_id: int, redact: bool | None, note: str = "") -> None:
        """Record (or, with ``None``, remove) the verdict for a track."""
        if redact is None:
            self.tracks.pop(track_id, None)
        else:
            self.tracks[track_id] = TrackLabel(redact, note)

    def add_frame(
        self, frame: int, kind: str, box: list[float] | None = None, note: str = ""
    ) -> FrameLabel:
        if kind not in KINDS:
            raise LabelsError(f"kind must be one of {KINDS}, got {kind!r}")
        if box is not None and (len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]):
            raise LabelsError(f"box must be [x1, y1, x2, y2] with positive size, got {box!r}")
        if frame < 0:
            raise LabelsError(f"frame must be >= 0, got {frame}")
        label = FrameLabel(int(frame), kind, [float(v) for v in box] if box else None, note)
        self.frames.append(label)
        return label

    def remove_frame(self, index: int) -> None:
        if not 0 <= index < len(self.frames):
            raise LabelsError(f"no frame label #{index}")
        del self.frames[index]

    def leaks(self) -> list[FrameLabel]:
        return [f for f in self.frames if f.kind == "leak"]

    def overs(self) -> list[FrameLabel]:
        return [f for f in self.frames if f.kind == "over"]


def load_labels(path: str | Path) -> Labels:
    """Load ``path``; a missing file is an empty :class:`Labels`."""
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except FileNotFoundError:
        return Labels()
    try:
        doc = msgspec.convert(json.loads(text), _LabelsDoc)
    except (ValueError, msgspec.ValidationError) as error:
        raise LabelsError(f"{source}: {error}") from error
    if doc.version != VERSION:
        raise LabelsError(f"{source}: unsupported labels version {doc.version}")
    try:
        tracks = {int(k): TrackLabel(v.redact, v.note) for k, v in doc.tracks.items()}
    except ValueError:
        raise LabelsError(f"{source}: track ids must be integers") from None
    return Labels(
        track_key=doc.track_key,
        tracks=tracks,
        frames=[FrameLabel(f.frame, f.kind, f.box, f.note) for f in doc.frames],
    )


def save_labels(path: str | Path, labels: Labels) -> Path:
    """Write ``labels`` atomically."""
    destination = Path(path)
    document: dict[str, Any] = {
        "version": VERSION,
        "track_key": labels.track_key,
        "tracks": {
            str(i): {"redact": t.redact, "note": t.note}
            for i, t in sorted(labels.tracks.items())
        },
        "frames": [
            {"frame": f.frame, "kind": f.kind, "box": f.box, "note": f.note}
            for f in labels.frames
        ],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.name + ".tmp")
    tmp.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, destination)
    return destination


def apply_labels(meta: Any, labels: Labels) -> int:
    """Set ``override`` on every tracklet that has a track verdict; returns how many."""
    applied = 0
    for tracklet in meta.tracklets:
        verdict = labels.tracks.get(tracklet.track_id)
        if verdict is not None and tracklet.override != verdict.redact:
            tracklet.override = verdict.redact
            applied += 1
    return applied
