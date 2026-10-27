"""Tests for the metadata schema: save/load round-trip, legacy v1, validation (#11)."""

import json
from pathlib import Path

import numpy as np
import pytest

from auto_redact.meta import (
    SCHEMA_VERSION,
    MetaError,
    Metadata,
    Tracklet,
    VideoInfo,
    load_meta,
    meta_from_document,
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
    )
    track2 = Tracklet(  # snapshot/classification fields still null
        track_id=2,
        frames=[0, 1],
        boxes=[[640.0, 120.0, 980.0, 719.0], [636.0, 121.0, 976.0, 719.0]],
        keypoints=[pose_frame(10.0), pose_frame(10.5)],
    )
    video = VideoInfo(path="clip.mp4", fps=30.0, width=1280, height=720, frames=900)
    return Metadata(video=video, tracklets=[track1, track2])


def legacy_document():
    """A minimal schema v1 document: everything inline in one JSON file."""
    return {
        "version": 1,
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


def load_document(doc):
    return meta_from_document(doc)


# --- round trip -----------------------------------------------------------------


def test_save_then_load_returns_identical_data(tmp_path):
    meta = sample_metadata()
    path = tmp_path / "out.json"
    save_meta(path, meta)
    loaded = load_meta(path)
    assert loaded == meta
    save_meta(path, loaded)  # and again, unchanged
    assert load_meta(path) == loaded


def test_summary_json_is_small_and_arrays_live_in_the_npz(tmp_path):
    path = save_meta(tmp_path / "out.meta.json", sample_metadata())
    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["version"] == SCHEMA_VERSION
    assert data["arrays"] == "out.meta.npz"
    assert set(data) == {
        "version", "video", "tracklets", "scene_cuts", "scene_cut_splits", "arrays"
    }
    assert set(data["video"]) == {"path", "fps", "width", "height", "frames"}
    track = data["tracklets"][0]
    assert set(track) == {
        "track_id", "observations", "best_frame", "best_snapshot_path",
        "isolated_track_ids", "redact", "score", "detail", "override",
    }
    assert track["observations"] == 3
    assert track["redact"] is None  # classification (#13) fills this in

    with np.load(tmp_path / "out.meta.npz") as arrays:
        assert arrays["t1_frames"].shape == (3,)
        assert arrays["t1_boxes"].shape == (3, 4)
        assert arrays["t1_keypoints"].shape == (3, 17, 3)


def test_classification_and_snapshot_fields_round_trip(tmp_path):
    meta = sample_metadata()
    meta.tracklets[0].redact = True
    meta.tracklets[0].score = 0.75
    meta.tracklets[0].isolated_track_ids = [2, 5]
    meta.tracklets[1].redact = False
    meta.scene_cuts = [40, 90]
    meta.scene_cut_splits = 1
    loaded = load_meta(save_meta(tmp_path / "out.json", meta))
    assert loaded == meta
    assert loaded.tracklets[0].redact is True
    assert loaded.tracklets[1].redact is False
    assert loaded.tracklets[1].score is None
    assert loaded.tracklets[1].isolated_track_ids == []


def test_detail_and_override_round_trip_and_override_wins(tmp_path):
    meta = sample_metadata()
    meta.tracklets[0].redact = False
    meta.tracklets[0].detail = "reply: 'No.'"
    meta.tracklets[0].override = True
    loaded = load_meta(save_meta(tmp_path / "out.json", meta))
    assert loaded == meta
    assert loaded.tracklets[0].effective_redact is True  # human beats classifier
    assert loaded.tracklets[1].effective_redact is None  # neither set
    loaded.tracklets[0].override = None
    assert loaded.tracklets[0].effective_redact is False


def test_hand_edited_redact_flag_is_honoured(tmp_path):
    path = save_meta(tmp_path / "out.json", sample_metadata())
    data = json.loads(path.read_text(encoding="utf-8"))
    data["tracklets"][1]["redact"] = True
    path.write_text(json.dumps(data), encoding="utf-8")
    assert load_meta(path).tracklets[1].redact is True


def test_save_creates_parent_directories(tmp_path):
    path = tmp_path / "nested" / "thumbs" / "meta.json"
    save_meta(path, sample_metadata())
    assert path.is_file() and path.with_suffix(".npz").is_file()


def test_numpy_values_from_detection_are_accepted(tmp_path):
    meta = Metadata(
        video=VideoInfo("clip.mp4", np.float64(30.0), np.int64(1280), 720, 2),
        tracklets=[
            Tracklet(
                track_id=np.int64(1),
                frames=np.array([0, 1]),
                boxes=np.array([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0]]),
                keypoints=np.array([pose_frame(), pose_frame()]),
            )
        ],
    )
    loaded = load_meta(save_meta(tmp_path / "out.json", meta))
    assert loaded.tracklets[0].frames == [0, 1]
    assert loaded.tracklets[0].boxes[1] == [1.0, 1.0, 11.0, 11.0]
    assert loaded.video.width == 1280


def test_tracklets_without_keypoints_round_trip(tmp_path):
    meta = Metadata(
        video=VideoInfo("clip.mp4", 30.0, 1280, 720, 2),
        tracklets=[Tracklet(1, [0, 1], [[0.0, 0.0, 5.0, 5.0]] * 2, [[], []])],
    )
    assert load_meta(save_meta(tmp_path / "out.json", meta)) == meta


def test_empty_metadata_round_trips(tmp_path):
    meta = Metadata(video=VideoInfo("clip.mp4", 30.0, 640, 480, 1))
    assert load_meta(save_meta(tmp_path / "out.json", meta)) == meta


def test_video_info_from_probe_converts_the_frames_probe():
    class Probe:  # duck-typed stand-in for frames.VideoInfo
        path = Path("clip.mp4")
        fps = 30.0
        width = 1280
        height = 720
        frame_count = 900

    assert VideoInfo.from_probe(Probe()) == VideoInfo("clip.mp4", 30.0, 1280, 720, 900)


def test_video_info_from_probe_rejects_non_probe_objects():
    with pytest.raises(MetaError, match=r"missing attribute\(s\)"):
        VideoInfo.from_probe(object())


# --- legacy schema v1 -------------------------------------------------------------


def test_legacy_v1_documents_still_load(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(json.dumps(legacy_document()), encoding="utf-8")
    loaded = load_meta(path)
    assert loaded.video.fps == 30
    assert loaded.tracklets[0].boxes == [[0.0, 0.0, 10.0, 10.0]]
    assert loaded.tracklets[0].redact is None
    assert loaded.tracklets[0].isolated_track_ids == []
    # saving upgrades it to the current layout
    save_meta(path, loaded)
    assert json.loads(path.read_text())["version"] == SCHEMA_VERSION
    assert load_meta(path) == loaded


# --- validation ---------------------------------------------------------------------


def assert_load_fails(doc, message):
    with pytest.raises(MetaError, match=message):
        load_document(doc)


def test_unsupported_version_is_rejected():
    doc = legacy_document()
    doc["version"] = SCHEMA_VERSION + 1
    assert_load_fails(doc, rf"unsupported schema version {SCHEMA_VERSION + 1}")


def test_missing_version_is_rejected():
    doc = legacy_document()
    del doc["version"]
    assert_load_fails(doc, r"missing key\(s\): version")


def test_unknown_keys_are_rejected():
    doc = legacy_document()
    doc["trackletz"] = []
    assert_load_fails(doc, "trackletz")


def test_non_object_document_is_rejected():
    assert_load_fails([1, 2, 3], r"expected object, got list")


def test_wrong_version_type_is_rejected():
    doc = legacy_document()
    doc["version"] = True
    assert_load_fails(doc, r"version: expected integer, got bool")


def test_misaligned_tracklet_is_rejected():
    doc = legacy_document()
    doc["tracklets"][0]["frames"] = [0, 1]
    assert_load_fails(doc, r"frames, boxes and keypoints must be aligned")


def test_wrong_box_arity_is_rejected():
    doc = legacy_document()
    doc["tracklets"][0]["boxes"] = [[0.0, 0.0, 10.0]]
    assert_load_fails(doc, r"expected 4 coordinates \[x1, y1, x2, y2\]")


def test_wrong_keypoint_shape_is_rejected():
    doc = legacy_document()
    doc["tracklets"][0]["keypoints"] = [[[1.0, 2.0]]]
    assert_load_fails(doc, r"expected \[x, y, confidence\]")


def test_uneven_keypoint_counts_are_rejected():
    doc = legacy_document()
    track = doc["tracklets"][0]
    track["frames"] = [0, 1]
    track["boxes"] = [[0.0, 0.0, 10.0, 10.0]] * 2
    track["keypoints"] = [pose_frame(0.0), pose_frame(0.0, 16)]
    assert_load_fails(doc, r"keypoints per frame must match")


def test_negative_frame_index_is_rejected():
    doc = legacy_document()
    doc["tracklets"][0]["frames"] = [-1]
    assert_load_fails(doc, r"non-negative integer")


def test_fractional_frame_index_is_rejected():
    doc = legacy_document()
    doc["tracklets"][0]["frames"] = [0.5]
    assert_load_fails(doc, r"list of integers")


def test_non_finite_coordinates_are_rejected(tmp_path):
    meta = sample_metadata()
    meta.tracklets[0].boxes[0][0] = float("nan")
    with pytest.raises(MetaError, match="finite"):
        save_meta(tmp_path / "out.json", meta)


def test_wrong_video_fps_type_is_rejected():
    doc = legacy_document()
    doc["video"]["fps"] = "30"
    assert_load_fails(doc, r"fps")


def test_non_positive_video_fps_is_rejected():
    doc = legacy_document()
    doc["video"]["fps"] = -30
    assert_load_fails(doc, r"fps")


def test_non_boolean_redact_is_rejected():
    doc = legacy_document()
    doc["tracklets"][0]["redact"] = "yes"
    assert_load_fails(doc, r"redact")


def test_non_array_tracklets_is_rejected():
    doc = legacy_document()
    doc["tracklets"] = {}
    assert_load_fails(doc, r"tracklets")


def test_save_refuses_to_write_an_invalid_document(tmp_path):
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
    path = tmp_path / "out.json"
    with pytest.raises(MetaError, match="must be aligned"):
        save_meta(path, meta)
    assert not path.exists()
    assert not path.with_suffix(".npz").exists()


def test_save_refuses_a_bad_video_before_writing(tmp_path):
    meta = sample_metadata()
    meta.video.fps = "fast"
    with pytest.raises(MetaError, match="fps"):
        save_meta(tmp_path / "out.json", meta)
    assert list(tmp_path.iterdir()) == []


def test_save_rejects_non_metadata_values(tmp_path):
    with pytest.raises(MetaError, match="expected a Metadata instance"):
        save_meta(tmp_path / "out.json", {"version": SCHEMA_VERSION})


def test_load_rejects_invalid_json(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(MetaError, match="not valid JSON"):
        load_meta(path)


def test_load_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_meta(tmp_path / "does-not-exist.json")


def test_missing_arrays_file_is_reported(tmp_path):
    path = save_meta(tmp_path / "out.json", sample_metadata())
    path.with_suffix(".npz").unlink()
    with pytest.raises(MetaError, match="arrays file"):
        load_meta(path)


def test_arrays_that_disagree_with_the_summary_are_rejected(tmp_path):
    path = save_meta(tmp_path / "out.json", sample_metadata())
    data = json.loads(path.read_text(encoding="utf-8"))
    data["tracklets"][0]["observations"] = 99
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(MetaError, match="99 observations"):
        load_meta(path)


def test_tracklet_missing_from_arrays_is_rejected(tmp_path):
    path = save_meta(tmp_path / "out.json", sample_metadata())
    data = json.loads(path.read_text(encoding="utf-8"))
    data["tracklets"][0]["track_id"] = 77
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(MetaError, match="no arrays stored"):
        load_meta(path)


def test_summary_save_updates_decisions_without_touching_the_arrays(tmp_path):
    from auto_redact.meta import save_meta_summary

    path = save_meta(tmp_path / "out.json", sample_metadata())
    arrays = path.with_suffix(".npz")
    before = arrays.stat().st_mtime_ns, arrays.read_bytes()

    meta = load_meta(path)
    meta.tracklets[0].override = True
    meta.tracklets[1].redact = False
    save_meta_summary(path, meta)

    assert (arrays.stat().st_mtime_ns, arrays.read_bytes()) == before
    again = load_meta(path)
    assert again == meta and again.tracklets[0].override is True


def test_summary_save_refuses_when_the_tracklets_changed(tmp_path):
    from auto_redact.meta import save_meta_summary

    path = save_meta(tmp_path / "out.json", sample_metadata())
    meta = load_meta(path)
    meta.tracklets.pop()
    with pytest.raises(MetaError, match="use save_meta"):
        save_meta_summary(path, meta)


def test_summary_save_needs_existing_arrays(tmp_path):
    from auto_redact.meta import save_meta_summary

    with pytest.raises(MetaError, match="without its arrays"):
        save_meta_summary(tmp_path / "nothing.json", sample_metadata())
