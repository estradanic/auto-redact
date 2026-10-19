"""Tests for metadata schema v1: save/load round-trip and validation (#11)."""

import json
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path

from auto_redact.meta import (
    SCHEMA_VERSION,
    MetaError,
    Metadata,
    Tracklet,
    VideoInfo,
    load_meta,
    save_meta,
)


def pose_frame(offset=0.0, count=17):
    """One frame of COCO-pose keypoints: [x, y, confidence] triples."""
    return [[offset + i, offset + i + 1.0, 0.9] for i in range(count)]


def sample_metadata():
    """A document as the tracking (#10) + snapshot (#12) passes write it."""
    track1 = Tracklet(
        track_id=1,
        frames=[12, 13, 14],
        boxes=[
            [100.0, 50.0, 300.0, 640.0],
            [104.5, 50.5, 305.5, 645.5],
            [109.0, 51.0, 311.0, 651.0],
        ],
        keypoints=[pose_frame(0.0), pose_frame(0.5), pose_frame(1.0)],
        best_frame=13,
        best_snapshot_path="thumbs/track_1.jpg",
        redact=None,
    )
    track2 = Tracklet(  # snapshot/classification fields still null
        track_id=2,
        frames=[0, 1],
        boxes=[[640.0, 120.0, 980.0, 719.0], [636.0, 121.0, 976.0, 719.0]],
        keypoints=[pose_frame(10.0), pose_frame(10.5)],
    )
    video = VideoInfo(path="clip.mp4", fps=30.0, width=1280, height=720, frames=900)
    return Metadata(video=video, tracklets=[track1, track2])


def valid_document():
    """A minimal schema v1 document, as raw JSON data."""
    return {
        "version": SCHEMA_VERSION,
        "video": {"path": "clip.mp4", "fps": 30, "width": 1280, "height": 720, "frames": 900},
        "tracklets": [
            {
                "track_id": 1,
                "frames": [0],
                "boxes": [[0.0, 0.0, 10.0, 10.0]],
                "keypoints": [[[1.0, 2.0, 0.9]]],
            }
        ],
    }


class RoundTripTests(unittest.TestCase):
    """The issue #11 acceptance bar: save, load, get identical data back."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def test_save_then_load_returns_identical_data(self):
        meta = sample_metadata()
        path = self.dir / "out.json"
        save_meta(path, meta)
        loaded = load_meta(path)
        self.assertEqual(loaded, meta)
        # ... and loading what was loaded again changes nothing either
        again = self.dir / "out.json"
        save_meta(again, loaded)
        self.assertEqual(load_meta(again), loaded)

    def test_saved_document_matches_schema_shape(self):
        path = save_meta(self.dir / "out.json", sample_metadata())
        data = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(data["version"], SCHEMA_VERSION)
        self.assertEqual(set(data), {"version", "video", "tracklets"})
        self.assertEqual(
            set(data["video"]), {"path", "fps", "width", "height", "frames"}
        )

        track = data["tracklets"][0]
        self.assertEqual(
            set(track),
            {
                "track_id",
                "frames",
                "boxes",
                "keypoints",
                "best_frame",
                "best_snapshot_path",
                "redact",
            },
        )
        self.assertIsNone(track["redact"])  # classification (#13) fills this in
        self.assertEqual(len(track["frames"]), len(track["boxes"]))
        self.assertEqual(len(track["frames"]), len(track["keypoints"]))
        self.assertEqual(len(track["keypoints"][0]), 17)
        self.assertEqual(len(track["keypoints"][0][0]), 3)

    def test_redact_round_trips_once_classification_fills_it(self):
        meta = sample_metadata()
        meta.tracklets[0].redact = True
        meta.tracklets[1].redact = False
        path = self.dir / "out.json"
        save_meta(path, meta)
        loaded = load_meta(path)
        self.assertIs(loaded.tracklets[0].redact, True)
        self.assertIs(loaded.tracklets[1].redact, False)

    def test_missing_optional_fields_load_as_null(self):
        # A tracking-only pass may omit fields #12/#13 fill in later.
        doc = valid_document()
        meta = Metadata.from_dict(doc)
        track = meta.tracklets[0]
        self.assertIsNone(track.best_frame)
        self.assertIsNone(track.best_snapshot_path)
        self.assertIsNone(track.redact)

        # Saving adds them back as null, so the file shape is stable.
        path = save_meta(self.dir / "out.json", meta)
        saved = json.loads(path.read_text(encoding="utf-8"))
        self.assertIsNone(saved["tracklets"][0]["best_frame"])
        self.assertIsNone(saved["tracklets"][0]["best_snapshot_path"])
        self.assertIsNone(saved["tracklets"][0]["redact"])

    def test_save_creates_parent_directories(self):
        path = self.dir / "nested" / "thumbs" / "meta.json"
        save_meta(path, sample_metadata())
        self.assertTrue(path.is_file())

    def test_array_like_values_from_detection_are_accepted(self):
        # Tracking (#10) may hand over numpy/torch arrays, not lists.
        class ArrayLike:
            def __init__(self, data):
                self._data = data

            def tolist(self):
                return self._data

        meta = Metadata(
            video=VideoInfo("clip.mp4", 30.0, 1280, 720, 2),
            tracklets=[
                Tracklet(
                    track_id=1,
                    frames=ArrayLike([0, 1]),
                    boxes=ArrayLike(
                        [[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0]]
                    ),
                    keypoints=[pose_frame(), pose_frame()],
                )
            ],
        )
        path = save_meta(self.dir / "out.json", meta)
        expected = Metadata(
            video=VideoInfo("clip.mp4", 30.0, 1280, 720, 2),
            tracklets=[
                Tracklet(
                    track_id=1,
                    frames=[0, 1],
                    boxes=[[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0]],
                    keypoints=[pose_frame(), pose_frame()],
                )
            ],
        )
        self.assertEqual(load_meta(path), expected)

    def test_video_info_from_probe_converts_the_frames_probe(self):
        # Bridge from #8: probe_video() -> metadata document video info.
        class Probe:  # duck-typed stand-in for frames.VideoInfo
            path = Path("clip.mp4")
            fps = 30.0
            width = 1280
            height = 720
            frame_count = 900
            duration = 30.0

        info = VideoInfo.from_probe(Probe())
        self.assertEqual(info, VideoInfo("clip.mp4", 30.0, 1280, 720, 900))
        meta = Metadata(video=info, tracklets=[])
        path = save_meta(self.dir / "out.json", meta)
        self.assertEqual(load_meta(path), meta)

    def test_video_info_from_probe_accepts_the_real_probe_video_info(self):
        try:
            from auto_redact.frames import VideoInfo as ProbeVideoInfo
        except ImportError:
            self.skipTest("PyAV/numpy not installed")
        probe = ProbeVideoInfo(
            path=Path("clip.mp4"), fps=30.0, width=1280, height=720,
            frame_count=900, duration=30.0,
        )
        self.assertEqual(
            VideoInfo.from_probe(probe), VideoInfo("clip.mp4", 30.0, 1280, 720, 900)
        )

    def test_video_info_from_probe_rejects_non_probe_objects(self):
        with self.assertRaisesRegex(MetaError, r"missing attribute\(s\)"):
            VideoInfo.from_probe(object())

    def test_to_dict_serializes_every_dataclass_field(self):
        # Guard: a field added later must not silently go missing on disk.
        meta = sample_metadata()
        self.assertEqual(
            set(meta.video.to_dict()), {f.name for f in fields(VideoInfo)}
        )
        self.assertEqual(
            set(meta.tracklets[0].to_dict()), {f.name for f in fields(Tracklet)}
        )
        self.assertEqual(
            set(meta.to_dict()), {f.name for f in fields(Metadata)} | {"version"}
        )


class ValidationTests(unittest.TestCase):
    """Defective documents are rejected with a clear MetaError."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def assert_load_fails(self, doc, message):
        with self.assertRaisesRegex(MetaError, message):
            Metadata.from_dict(doc)

    def test_unsupported_version_is_rejected(self):
        doc = valid_document()
        doc["version"] = SCHEMA_VERSION + 1
        self.assert_load_fails(doc, r"unsupported schema version 2")

    def test_missing_version_is_rejected(self):
        doc = valid_document()
        del doc["version"]
        self.assert_load_fails(doc, r"metadata: missing key\(s\): version")

    def test_unknown_keys_are_rejected(self):
        doc = valid_document()
        doc["trackletz"] = []
        self.assert_load_fails(doc, r"metadata: unknown key\(s\): trackletz")

    def test_non_object_document_is_rejected(self):
        self.assert_load_fails([1, 2, 3], r"metadata: expected object, got list")

    def test_wrong_version_type_is_rejected(self):
        doc = valid_document()
        doc["version"] = True
        self.assert_load_fails(doc, r"metadata\.version: expected integer, got bool")

    def test_misaligned_tracklet_is_rejected(self):
        doc = valid_document()
        doc["tracklets"][0]["frames"] = [0, 1]
        self.assert_load_fails(doc, r"frames, boxes and keypoints must be aligned")

    def test_wrong_box_arity_is_rejected(self):
        doc = valid_document()
        doc["tracklets"][0]["boxes"] = [[0.0, 0.0, 10.0]]
        self.assert_load_fails(doc, r"expected 4 coordinates \[x1, y1, x2, y2\]")

    def test_wrong_keypoint_shape_is_rejected(self):
        doc = valid_document()
        doc["tracklets"][0]["keypoints"] = [[[1.0, 2.0]]]
        self.assert_load_fails(doc, r"expected \[x, y, confidence\]")

    def test_uneven_keypoint_counts_are_rejected(self):
        doc = valid_document()
        doc["tracklets"][0]["frames"] = [0, 1]
        doc["tracklets"][0]["boxes"] = [[0.0, 0.0, 10.0, 10.0]] * 2
        doc["tracklets"][0]["keypoints"] = [pose_frame(0.0), pose_frame(0.0, 16)]
        self.assert_load_fails(doc, r"keypoints per frame must match")

    def test_negative_frame_index_is_rejected(self):
        doc = valid_document()
        doc["tracklets"][0]["frames"] = [-1]
        self.assert_load_fails(doc, r"non-negative integer")

    def test_wrong_video_fps_type_is_rejected(self):
        doc = valid_document()
        doc["video"]["fps"] = "30"
        self.assert_load_fails(doc, r"video\.fps: expected number, got str")

    def test_non_positive_video_fps_is_rejected(self):
        doc = valid_document()
        doc["video"]["fps"] = -30
        self.assert_load_fails(doc, r"video\.fps: expected a positive number")

    def test_non_boolean_redact_is_rejected(self):
        doc = valid_document()
        doc["tracklets"][0]["redact"] = "yes"
        self.assert_load_fails(doc, r"tracklet 1\.redact: expected boolean or null")

    def test_non_array_tracklets_is_rejected(self):
        doc = valid_document()
        doc["tracklets"] = {}
        self.assert_load_fails(doc, r"tracklets: expected array, got dict")

    def test_save_refuses_to_write_an_invalid_document(self):
        meta = Metadata(
            video=VideoInfo("clip.mp4", 30.0, 1280, 720, 900),
            tracklets=[
                Tracklet(
                    track_id=1,
                    frames=[0, 1],
                    boxes=[[0.0, 0.0, 10.0, 10.0]],
                    keypoints=[pose_frame()],
                )
            ],
        )
        path = self.dir / "out.json"
        with self.assertRaisesRegex(MetaError, r"must be aligned"):
            save_meta(path, meta)
        self.assertFalse(path.exists())

    def test_save_rejects_non_metadata_values(self):
        with self.assertRaisesRegex(MetaError, r"expected a Metadata instance"):
            save_meta(self.dir / "out.json", {"version": SCHEMA_VERSION})

    def test_load_rejects_invalid_json(self):
        path = self.dir / "broken.json"
        path.write_text("{not json", encoding="utf-8")
        with self.assertRaisesRegex(MetaError, r"not valid JSON"):
            load_meta(path)

    def test_load_missing_file_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_meta(self.dir / "does-not-exist.json")


if __name__ == "__main__":
    unittest.main()
