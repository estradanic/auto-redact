"""Person detection: YOLO11n-pose per-frame boxes + keypoints.

Run Ultralytics ``yolo11n-pose.pt`` over frames to detect people with
bounding boxes and 17 COCO keypoints.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO

from .frames import iter_frames


def _select_device(device: str | None) -> str:
    """Return a device string accepted by ``model.to`` / ``predict``.

    Ultralytics accepts ``"0"`` as CUDA device 0, but ``torch.nn.Module.to``
    (which YOLO inherits) expects ``"cuda:0"`` / ``"cpu"``. Normalise here so
    callers can use either notation.
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


def _draw_overlay(frame: np.ndarray, detections: list[dict[str, Any]]) -> Image.Image:
    """Draw detection boxes and keypoints on a copy of ``frame``."""
    image = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(image)
    for idx, det in enumerate(detections):
        color = _PALETTE[idx % len(_PALETTE)]
        x1, y1, x2, y2 = det["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        for x, y, conf in det["keypoints"]:
            if conf > 0.1:
                draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=color)
    return image


_PALETTE = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
]


def detect_people(
    video_path: str | Path,
    sample_rate: int = 1,
    device: str | None = "auto",
    overlay_path: str | Path | None = None,
) -> Iterator[tuple[int, list[dict[str, Any]]]]:
    """Detect people in ``video_path``.

    Args:
        video_path: Path to the input video.
        sample_rate: Process every Nth frame (passed through to frame reader).
        device: Device to run inference on ("auto", "cpu", or "0"/"cuda").
        overlay_path: Optional file path; when set, a debug overlay image
            (boxes + keypoints drawn on the first frame with detections)
            is saved here for eyeballing.

    Yields:
        ``(frame_idx, detections)`` where each detection is a dict with:
          - ``bbox``: ``[x1, y1, x2, y2]`` (float)
          - ``keypoints``: ``[[x, y, conf], ...]`` for 17 COCO keypoints
          - ``det_conf``: detection confidence
    """
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool):
        raise ValueError(f"sample_rate must be an integer, got {sample_rate!r}")
    if sample_rate < 1:
        raise ValueError(f"sample_rate must be >= 1, got {sample_rate}")

    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"no such video file: {video_path}")

    model = YOLO("yolo11n-pose.pt")
    model.to(_select_device(device))

    overlay_saved = False

    for frame_idx, frame in iter_frames(video_path, sample_rate=sample_rate):
        results = model(frame, verbose=False)
        frame_dets: list[dict[str, Any]] = []
        for result in results:
            boxes = result.boxes
            kpts = result.keypoints
            if boxes is None:
                continue
            # Extract data
            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.array(boxes.xyxy)
            confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.array(boxes.conf)
            cls = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.array(boxes.cls)
            kpts_data = None
            if kpts is not None:
                try:
                    kpts_data = kpts.data.cpu().numpy()  # (n, 17, 3)
                except Exception:
                    kpts_data = np.array(kpts.data)
            for i in range(len(xyxy)):
                c = int(cls[i]) if i < len(cls) else 0
                if c == 0:  # COCO class 0 = person
                    bbox = xyxy[i].tolist()  # [x1,y1,x2,y2]
                    det_conf = float(confs[i]) if i < len(confs) else 0.0
                    kps = []
                    if kpts_data is not None and i < len(kpts_data):
                        kp = kpts_data[i]  # 17x3
                        for j in range(17):
                            if j < len(kp):
                                x = float(kp[j][0]) if len(kp[j]) > 0 else 0.0
                                y = float(kp[j][1]) if len(kp[j]) > 1 else 0.0
                                conf = float(kp[j][2]) if len(kp[j]) > 2 else 0.0
                            else:
                                x, y, conf = 0.0, 0.0, 0.0
                            kps.append([x, y, conf])
                    frame_dets.append(
                        {
                            "bbox": [float(b) for b in bbox],
                            "keypoints": kps,
                            "det_conf": det_conf,
                        }
                    )
        if overlay_path is not None and not overlay_saved and frame_dets:
            overlay = _draw_overlay(frame, frame_dets)
            destination = Path(overlay_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            overlay.save(destination)
            overlay_saved = True
        yield frame_idx, frame_dets
