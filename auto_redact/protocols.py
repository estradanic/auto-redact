"""Interfaces for the swappable parts of the pipeline.

The stages depend on these shapes rather than on concrete backends, so a
fake can stand in for a model or ffmpeg in tests, and a new backend only
has to implement the protocol and be registered in its factory
(:func:`auto_redact.classify.make_classifier`,
:func:`auto_redact.encode.make_encoder`).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .classify import Decision


@runtime_checkable
class Classifier(Protocol):
    """Judge one thumbnail against the prompt.

    A classifier may set ``thread_safe = True`` to let the stage call it
    from several worker threads at once.
    """

    def classify(self, image_path: str | Path, prompt: str) -> Decision: ...


class Encoder(Protocol):
    """A video encoder backend: how ffmpeg is told to write H.264."""

    name: str

    def prepare(self) -> None:
        """Get ready to encode (e.g. free GPU memory for the encoder)."""
        ...

    def codec_args(self, crf: int, preset: str) -> list[str]:
        """ffmpeg output arguments selecting this encoder and its quality."""
        ...

    def fallback(
        self, preset: str, error: BaseException
    ) -> tuple[Encoder, str] | None:
        """A more reliable ``(encoder, preset)`` to retry with after ``error``."""
        ...


class Detection(NamedTuple):
    """One tracked person in one frame."""

    track_id: int
    box: list[float]  # x1, y1, x2, y2 in pixels
    keypoints: list[list[float]]  # [x, y, confidence] per COCO keypoint


class PersonTracker(Protocol):
    """Turns frames, fed in order, into detections with stable track ids."""

    lost_buffer: int
    """Steps after which an id that stopped appearing is gone for good."""

    def update(self, frame) -> list[Detection]: ...


class PersonDetector(Protocol):
    """Finds people in a single RGB image, without identities."""

    def detect(self, frame, conf: float) -> list[list[float]]:
        """``[x1, y1, x2, y2]`` boxes of people scoring at least ``conf``."""
        ...
