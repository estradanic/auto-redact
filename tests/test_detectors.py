"""Tracker/detector seams: adapter parsing and stage use of injected backends."""

import numpy as np

from auto_redact.detectors import _detections
from auto_redact.protocols import Detection
from auto_redact.track import add_orphan_tracklets, build_tracklets

from .test_snapshot import make_clip, make_meta, make_tracklet


class Tensor:
    """Minimal stand-in for a torch tensor."""

    def __init__(self, data):
        self.data = np.array(data)

    def cpu(self):
        return self

    def numpy(self):
        return self.data


class Result:
    def __init__(self, ids, boxes, keypoints):
        self.boxes = type("B", (), {"id": Tensor(ids) if ids is not None else None,
                                    "xyxy": Tensor(boxes)})()
        self.keypoints = type("K", (), {"data": Tensor(keypoints)})() if keypoints is not None else None


def test_detections_parse_ids_boxes_and_keypoints():
    kp = [[[1.0, 2.0, 0.5]] * 17]
    found = _detections(Result([7], [[1, 2, 3, 4]], kp))
    assert found == [Detection(7, [1.0, 2.0, 3.0, 4.0], [[1.0, 2.0, 0.5]] * 17)]


def test_detections_without_ids_or_keypoints():
    assert _detections(Result(None, [[1, 2, 3, 4]], None)) == []
    found = _detections(Result([1], [[1, 2, 3, 4]], None))
    assert found[0].keypoints == [[0.0, 0.0, 0.0]] * 17


def test_short_keypoint_sets_are_padded_to_seventeen():
    found = _detections(Result([1], [[0, 0, 1, 1]], [[[1.0, 1.0, 1.0]] * 5]))
    assert len(found[0].keypoints) == 17 and found[0].keypoints[10] == [0.0, 0.0, 0.0]


class Scripted:
    lost_buffer = 5

    def __init__(self):
        self.seen = 0

    def update(self, frame):
        self.seen += 1
        return [Detection(3, [1.0, 1.0, 9.0, 9.0], [[0.0, 0.0, 0.0]] * 17)]


def test_build_tracklets_uses_the_injected_tracker_and_stride(tmp_path):
    video = make_clip(tmp_path / "v.mp4")
    tracker = Scripted()
    meta = build_tracklets(video, tracker=tracker, stride=2, scene_cuts=False)
    assert [t.track_id for t in meta.tracklets] == [3]
    assert all(f % 2 == 0 for f in meta.tracklets[0].frames)
    assert tracker.seen == len(meta.tracklets[0].frames)


class FakeDetector:
    def detect(self, frame, conf):
        return [[100.0, 100.0, 160.0, 200.0]]  # nobody tracked here


def test_orphan_recovery_uses_the_injected_detector(tmp_path):
    video = make_clip(tmp_path / "v.mp4")
    meta = make_meta(video, [make_tracklet(1, range(0, 10), [[5, 5, 40, 80]] * 10)])
    out = add_orphan_tracklets(meta, video, detector=FakeDetector())
    assert len(out.tracklets) == 2
    assert out.tracklets[1].boxes[0] == [100.0, 100.0, 160.0, 200.0]


def test_tracker_honours_the_configured_device_and_precision(monkeypatch):
    from auto_redact import detectors
    from auto_redact.config import TrackConfig

    seen = []
    monkeypatch.setattr(detectors, "_new_model", lambda device: seen.append(device) or object())
    monkeypatch.setattr(detectors, "_track_buffer", lambda tracker: 30)

    cpu = detectors.make_tracker(TrackConfig(device="cpu", half=True))
    assert seen == ["cpu"] and cpu.device == "cpu"
    assert "quantize" not in cpu._kwargs  # FP16 only applies on CUDA

    gpu = detectors.make_tracker(TrackConfig(device="0", half=True))
    assert gpu.device == "cuda:0" and gpu._kwargs["quantize"] == 16
    assert "quantize" not in detectors.make_tracker(TrackConfig(device="0", half=False))._kwargs
    assert gpu._kwargs["conf"] == 0.25 and gpu._kwargs["tracker"] == "bytetrack.yaml"
