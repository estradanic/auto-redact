"""Person detection and tracking behind swappable interfaces.

The pipeline needs two things from a vision backend:

* a :class:`~auto_redact.protocols.PersonTracker` that turns video frames,
  in order, into per-person detections carrying a stable track id (what
  tracking and the streaming pipeline use);
* a :class:`~auto_redact.protocols.PersonDetector` that finds people in a
  single image with no identity (orphan recovery and thumbnail
  sub-isolation).

Ultralytics YOLO11n-pose with ByteTrack (or BoT-SORT, via
``TrackConfig.tracker``) implements both; it is the only module that
imports ``ultralytics``. Tests and other backends supply their own.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np

from .config import TrackConfig
from .protocols import Detection

_WEIGHTS_DIR = Path(os.path.expanduser("~/.cache/auto-redact/weights"))
_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
_WEIGHTS = str(_WEIGHTS_DIR / "yolo11n-pose.pt")
_KEYPOINTS = 17  # COCO pose


class _DropMotionWarning(logging.Filter):
    """Hide the tracker's per-frame "not enough matching points" warning.

    Raised by camera-motion estimation on low-texture frames; harmless
    (the correction is skipped for that frame) but very noisy.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return "not enough matching points" not in record.getMessage()


logging.getLogger("ultralytics").addFilter(_DropMotionWarning())


def select_device(device: str | None) -> str:
    """Return a device string accepted by ``model.to`` / ``track``.

    Ultralytics accepts ``"0"`` as CUDA device 0, but ``torch.nn.Module.to``
    (which YOLO inherits) expects ``"cuda:0"`` / ``"cpu"``.
    """
    if device is None or device == "auto":
        try:
            import torch

            return "cuda:0" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    if device.isdigit():
        return f"cuda:{device}"
    return device


def _new_model(device: str) -> Any:
    from ultralytics import YOLO

    model = YOLO(_WEIGHTS)
    if hasattr(model, "to"):
        model.to(device)
    return model


# Cached models for single-threaded detection passes. Trackers keep their
# own model, because tracking state lives inside it.
_model_cache: dict[str, Any] = {}
_model_cache_lock = threading.Lock()


def _cached_model(device: str) -> Any:
    with _model_cache_lock:
        model = _model_cache.get(device)
        if model is None:
            model = _model_cache[device] = _new_model(device)
        return model


def _to_numpy(value: Any) -> np.ndarray:
    return value.cpu().numpy() if hasattr(value, "cpu") else np.asarray(value)


class UltralyticsDetector:
    """YOLO person detection on single images (no identities)."""

    def __init__(self, device: str = "auto") -> None:
        self._device = select_device(device)

    def detect(self, frame: np.ndarray, conf: float) -> list[list[float]]:
        results = _cached_model(self._device)(frame, verbose=False, conf=conf)
        boxes: list[list[float]] = []
        for result in results:
            if getattr(result, "boxes", None) is None:
                continue
            for row in _to_numpy(result.boxes.xyxy):
                boxes.append([float(v) for v in row])
        return boxes


class UltralyticsTracker:
    """YOLO pose detection plus ultralytics' ByteTrack/BoT-SORT association."""

    def __init__(
        self, config: TrackConfig | None = None, device: str | None = None, half: bool | None = None
    ) -> None:
        config = config if config is not None else TrackConfig()
        self.device = select_device(config.device if device is None else device)
        half = config.half if half is None else half
        self._model = _new_model(self.device)
        # ``quantize`` replaces the deprecated ``half`` argument (which warns
        # on every call); nothing is passed for FP32 so the default applies.
        self._kwargs: dict[str, Any] = dict(
            conf=config.conf, iou=config.iou, persist=True, verbose=False,
            tracker=config.tracker,
        )
        if half and self.device.startswith("cuda"):
            self._kwargs["quantize"] = 16
        if config.imgsz is not None:
            self._kwargs["imgsz"] = config.imgsz
        self.lost_buffer = _track_buffer(config.tracker)

    def close(self) -> None:
        """Release the underlying model and any GPU memory it holds."""
        self._model = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def update(self, frame: np.ndarray) -> list[Detection]:
        detections: list[Detection] = []
        for result in self._model.track(frame, **self._kwargs):
            detections.extend(_detections(result))
        return detections


def _track_buffer(tracker: str) -> int:
    """Frames the tracker keeps a lost id alive before dropping it for good."""
    try:
        import yaml
        from ultralytics.utils.checks import check_yaml

        with open(check_yaml(tracker), encoding="utf-8") as handle:
            return int(yaml.safe_load(handle)["track_buffer"])
    except Exception:
        return 30


def _detections(result: Any) -> list[Detection]:
    boxes = result.boxes
    kpts = result.keypoints
    if boxes is None or boxes.id is None:
        return []
    xyxy = _to_numpy(boxes.xyxy)
    track_ids = _to_numpy(boxes.id)
    kpts_data = None
    if kpts is not None:
        try:
            kpts_data = _to_numpy(kpts.data)
        except Exception:
            kpts_data = None
    detections = []
    for i in range(len(xyxy)):
        if kpts_data is not None and i < len(kpts_data):
            kp = kpts_data[i]
            keypoints = [
                [float(kp[j][0]), float(kp[j][1]), float(kp[j][2])]
                if len(kp[j]) >= 3
                else [0.0, 0.0, 0.0]
                for j in range(min(_KEYPOINTS, len(kp)))
            ]
            while len(keypoints) < _KEYPOINTS:
                keypoints.append([0.0, 0.0, 0.0])
        else:
            keypoints = [[0.0, 0.0, 0.0] for _ in range(_KEYPOINTS)]
        detections.append(
            Detection(int(track_ids[i]), [float(b) for b in xyxy[i].tolist()], keypoints)
        )
    return detections


def make_tracker(
    config: TrackConfig | None = None, device: str | None = None, half: bool | None = None
):
    """The tracking backend (ultralytics YOLO + ByteTrack by default).

    ``device`` and ``half`` default to the config's values.
    """
    return UltralyticsTracker(config, device, half)


def make_detector(device: str = "auto"):
    """The single-image person detector."""
    return UltralyticsDetector(device)
