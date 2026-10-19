"""Tracking: ByteTrack association of per-frame detections into tracklets.

Associate per-frame detections into persistent track_ids using Ultralytics
ByteTrack (via ``model.track(..., persist=True)``). Short dropouts (a few
frames where a person is momentarily undetected) are handled by ByteTrack's
internal track buffer, so a single person keeps one stable ``track_id``
across the whole traversal.

The result is aggregated per ``track_id`` into :class:`auto_redact.meta.Tracklet`
records (frames + boxes + keypoints, aligned by index), wrapped in a
:class:`auto_redact.meta.Metadata` document ready for
:func:`auto_redact.meta.save_meta`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO

from .frames import iter_frames, probe_video
from .meta import Metadata, Tracklet, VideoInfo


def _select_device(device: str | None) -> str:
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


def build_tracklets(
    video_path: str | Path,
    sample_rate: int = 1,
    device: str | None = "auto",
) -> Metadata:
    """Run ByteTrack over ``video_path`` and return aggregated tracklets.

    Args:
        video_path: Path to the input video.
        sample_rate: Process every Nth frame (stride; passed to the reader).
        device: Device to run inference on ("auto", "cpu", or "0"/"cuda").

    Returns:
        A :class:`Metadata` document with one :class:`Tracklet` per
        persistent ``track_id``, each holding aligned ``frames`` / ``boxes``
        / ``keypoints`` lists (ready for :func:`save_meta`).
    """
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool):
        raise ValueError(f"sample_rate must be an integer, got {sample_rate!r}")
    if sample_rate < 1:
        raise ValueError(f"sample_rate must be >= 1, got {sample_rate}")

    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"no such video file: {video_path}")

    info = probe_video(video_path)
    model = YOLO("yolo11n-pose.pt")
    model.to(_select_device(device))

    # track_id -> per-observation lists (frame index, box, keypoints)
    observations: dict[int, tuple[list[int], list[list[float]], list[Any]]] = {}

    for frame_idx, frame in iter_frames(video_path, sample_rate=sample_rate):
        results = model.track(frame, persist=True, verbose=False)
        for result in results:
            boxes = result.boxes
            kpts = result.keypoints
            if boxes is None or boxes.id is None:
                continue
            if hasattr(boxes.xyxy, "cpu"):
                xyxy = boxes.xyxy.cpu().numpy()
                track_ids = boxes.id.cpu().numpy()
            else:
                xyxy = np.array(boxes.xyxy)
                track_ids = np.array(boxes.id)
            kpts_data = None
            if kpts is not None:
                try:
                    if hasattr(kpts.data, "cpu"):
                        kpts_data = kpts.data.cpu().numpy()
                    else:
                        kpts_data = np.array(kpts.data)
                except Exception:
                    kpts_data = None
            for i in range(len(xyxy)):
                tid = int(track_ids[i])
                bbox = [float(b) for b in xyxy[i].tolist()]
                if kpts_data is not None and i < len(kpts_data):
                    kp = kpts_data[i]
                    kps = [
                        [float(kp[j][0]), float(kp[j][1]), float(kp[j][2])]
                        if len(kp[j]) >= 3
                        else [0.0, 0.0, 0.0]
                        for j in range(min(17, len(kp)))
                    ]
                    while len(kps) < 17:
                        kps.append([0.0, 0.0, 0.0])
                else:
                    kps = [[0.0, 0.0, 0.0] for _ in range(17)]
                if tid not in observations:
                    observations[tid] = ([], [], [])
                frames_list, boxes_list, kpts_list = observations[tid]
                frames_list.append(frame_idx)
                boxes_list.append(bbox)
                kpts_list.append(kps)

    tracklets = [
        Tracklet(
            track_id=tid,
            frames=obs[0],
            boxes=obs[1],
            keypoints=obs[2],
        )
        for tid, obs in sorted(observations.items())
    ]
    return Metadata(
        video=VideoInfo(
            path=str(video_path),
            fps=info.fps,
            width=info.width,
            height=info.height,
            frames=info.frame_count,
        ),
        tracklets=tracklets,
    )
