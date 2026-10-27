"""The streaming pipeline must match the staged one.

Neither needs real models here: a scripted tracker stands in for YOLO +
ByteTrack and a fake SigLIP stands in for the classifier, so the tests run
anywhere ffmpeg does.
"""

import shutil
import subprocess

import pytest

from auto_redact.classify import Decision
from auto_redact.config import ClassifyConfig, Config, SnapshotConfig, TrackConfig
from auto_redact.frames import iter_frames, probe_video
from auto_redact.meta import load_meta
from auto_redact.pipeline import run_pipeline
from auto_redact.protocols import Detection
from auto_redact.streaming import run_pipeline_streaming
from auto_redact.workspace import workspace_for

FFMPEG = shutil.which("ffmpeg")
pytestmark = pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not installed")

FRAMES = 80


class ScriptedTracker:
    """Two people crossing the frame; ids are stable like a real tracker's."""

    lost_buffer = 5

    def __init__(self):
        self.frame = -1

    def update(self, frame):
        self.frame += 1
        i = self.frame
        found = []
        if i < 60:
            found.append(Detection(1, [10 + i, 20, 90 + i, 200], [[0.0, 0.0, 0.0]] * 17))
        if 20 <= i < FRAMES:
            found.append(Detection(2, [200 - i, 30, 280 - i, 220], [[0.0, 0.0, 0.0]] * 17))
        return found


class FakeSiglip:
    """Redacts whichever person's snapshot has more red than blue-ish content."""

    def __init__(self, model, threshold, device):
        pass

    def classify(self, path, prompt):
        name = str(path)
        return Decision(name.endswith("track_1.jpg"), 0.9, "fake")


@pytest.fixture
def clip(tmp_path):
    video = tmp_path / "in.mp4"
    subprocess.run(
        [FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i",
         f"testsrc=size=320x240:rate=25:duration={FRAMES / 25}",
         "-c:v", "libx264", "-g", "10", "-bf", "0", "-pix_fmt", "yuv420p", str(video)],
        check=True,
    )
    return video


@pytest.fixture
def config(monkeypatch):
    monkeypatch.setattr("auto_redact.siglip.SiglipClassifier", FakeSiglip)
    return Config(
        track=TrackConfig(scene_cuts=False),
        snapshot=SnapshotConfig(isolate=False),
        classify=ClassifyConfig(classifier="siglip", device="cpu"),
    )


@pytest.fixture
def both(clip, tmp_path, config, monkeypatch):
    monkeypatch.setattr(
        "auto_redact.track.make_tracker", lambda *a, **k: ScriptedTracker()
    )
    staged, streamed = tmp_path / "staged", tmp_path / "streamed"
    staged.mkdir()
    streamed.mkdir()
    run_pipeline(clip, "person", staged / "out.mp4", config,
                 meta_path=staged / "out.meta.json", thumbs_dir=staged / "thumbs")
    run_pipeline_streaming(clip, "person", streamed / "out.mp4", config,
                           meta_path=streamed / "out.meta.json",
                           thumbs_dir=streamed / "thumbs",
                           tracker=ScriptedTracker())
    return clip, staged, streamed


def test_streaming_matches_staged_metadata(both):
    _, staged, streamed = both
    a = load_meta(staged / "out.meta.json")
    b = load_meta(streamed / "out.meta.json")
    assert [t.track_id for t in a.tracklets] == [1, 2]
    assert len(a.tracklets) == len(b.tracklets)
    for x, y in zip(a.tracklets, b.tracklets):
        assert x.track_id == y.track_id
        assert x.frames == y.frames
        assert x.boxes == y.boxes
        assert x.best_frame == y.best_frame
        assert x.redact == y.redact
        assert x.score == y.score
    assert [t.redact for t in a.tracklets] == [True, False]


def test_streaming_output_matches_staged_frame_for_frame(both):
    clip, staged, streamed = both
    assert probe_video(staged / "out.mp4").frame_count == probe_video(clip).frame_count
    assert probe_video(streamed / "out.mp4").frame_count == probe_video(clip).frame_count
    for (i, a), (_, b) in zip(iter_frames(staged / "out.mp4"), iter_frames(streamed / "out.mp4")):
        assert abs(a.astype(int) - b.astype(int)).mean() < 2.0, f"frame {i} differs"


def test_redacted_person_is_actually_blacked_out(both):
    _, staged, _ = both
    frames = dict(iter_frames(staged / "out.mp4"))
    inside = frames[10][60:150, 30:80]  # person 1's box at frame 10 (smoothed, so inner part)
    assert inside.max() < 30


def test_staged_run_reuses_what_streaming_cached(clip, tmp_path, config):
    run_pipeline_streaming(clip, "person", tmp_path / "out.mp4", config,
                           tracker=ScriptedTracker())
    summary = run_pipeline(clip, "person", tmp_path / "out.mp4", config)
    assert summary.reused == ["track", "snapshot", "classify"]
    assert (summary.tracklets, summary.redacted) == (2, 1)


class UndecidedSiglip:
    def __init__(self, model, threshold, device):
        pass

    def classify(self, path, prompt):
        return Decision(None, None, "could not tell")


@pytest.fixture
def undecided(config, monkeypatch):
    import dataclasses

    monkeypatch.setattr("auto_redact.siglip.SiglipClassifier", UndecidedSiglip)
    return lambda policy: dataclasses.replace(
        config, classify=dataclasses.replace(config.classify, on_undecided=policy)
    )


@pytest.mark.parametrize("runner", ["staged", "streaming"])
def test_undecided_people_are_redacted_by_default(clip, tmp_path, undecided, monkeypatch, runner):
    monkeypatch.setattr("auto_redact.track.make_tracker", lambda *a, **k: ScriptedTracker())
    cfg = undecided("redact")
    if runner == "staged":
        run_pipeline(clip, "person", tmp_path / "out.mp4", cfg)
    else:
        run_pipeline_streaming(clip, "person", tmp_path / "out.mp4", cfg, tracker=ScriptedTracker())
    meta = load_meta(workspace_for(clip) / "meta.json")
    assert [t.redact for t in meta.tracklets] == [True, True]
    frames = dict(iter_frames(tmp_path / "out.mp4"))
    assert frames[10][60:150, 30:80].max() < 30  # person 1 is blacked out


def test_streaming_error_policy_writes_no_video(clip, tmp_path, undecided):
    from auto_redact.classify import ClassifyError

    with pytest.raises(ClassifyError, match="could not be classified"):
        run_pipeline_streaming(clip, "person", tmp_path / "out.mp4", undecided("error"),
                               tracker=ScriptedTracker())
    assert not (tmp_path / "out.mp4").exists()
    assert [t.redact for t in load_meta(workspace_for(clip) / "meta.json").tracklets] == [None, None]
