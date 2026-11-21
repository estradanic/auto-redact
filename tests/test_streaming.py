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


def test_growing_ts_parts_carry_tiling_audio(tmp_path):
    """Each shipped part holds its slice of the source audio, so the
    concatenated growing file plays with sound all the way through."""
    from auto_redact.segments import Segment, add_audio_to_part, copy_segment, encode_segment, keyframe_table
    from auto_redact.encode import make_encoder

    source = tmp_path / "av.mp4"
    subprocess.run(
        [FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=25:duration=3",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=3", "-c:v", "libx264", "-g", "10",
         "-bf", "0", "-pix_fmt", "yuv420p", "-c:a", "aac", str(source)],
        check=True,
    )
    info = probe_video(source)
    keyframes, _ = keyframe_table(source)
    bounds = sorted({0, info.frame_count, *keyframes})
    encoder = make_encoder("libx264")
    growing = tmp_path / "growing.ts"
    for n, (lo, hi) in enumerate(zip(bounds, bounds[1:])):
        edited = n == 1
        segment = Segment(lo, hi, edited)
        video_part, part = tmp_path / f"v{n}.ts", tmp_path / f"p{n}.ts"
        if edited:
            encode_segment(source, video_part, {}, keyframes, segment, info, 23, "ultrafast", encoder)
        else:
            copy_segment(source, video_part, keyframes, segment)
        assert add_audio_to_part(video_part, part, source, keyframes, segment, retime=edited)
        with open(growing, "ab") as f:
            f.write(part.read_bytes())

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-count_packets", "-show_entries",
         "stream=codec_type,nb_read_packets", "-of", "csv=p=0", str(growing)],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    counts = dict(line.split(",") for line in probe)
    assert int(counts["video"]) == info.frame_count
    assert int(counts["audio"]) >= 0.95 * 3 * 44100 / 1024


def test_shipping_stays_in_order_when_a_later_gop_is_final_first():
    """A closed-but-undecided tracklet blocks only its own GOP; the GOPs after
    it must still wait, or the growing file is appended out of order."""
    from auto_redact.meta import Tracklet, VideoInfo
    from auto_redact.streaming import _Frontier, _Track

    info = VideoInfo(path="x.mp4", fps=25.0, width=64, height=48, frames=40)
    frontier = _Frontier(info, max_gap=0, smooth_window=1, bridge_iou=0.5, min_track_frames=1)
    gops = [(0, 10), (10, 20), (20, 30), (30, 40)]
    tl = Tracklet(track_id=1, frames=[12, 15, 18], boxes=[[0, 0, 5, 5]] * 3, keypoints=[[]] * 3)
    frontier.add(_Track(uid=1, tracker_id=1, tl=tl, closed=True))
    frontier.advance(40)
    frontier.finish_tracking()

    assert frontier.ready_ranges(gops, set()) == [0]
    track = next(iter(frontier.undecided.values()))
    frontier.decide(track)
    assert frontier.ready_ranges(gops, {0}) == [1, 2, 3]


def _fake_vlc(tmp_path, monkeypatch):
    """A `vlc` that records the bytes piped to it instead of opening a window."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    captured = tmp_path / "captured.ts"
    script = bindir / "vlc"
    script.write_text(f"#!/bin/sh\ncat > {captured}\n")
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}:{__import__('os').environ['PATH']}")
    return captured


def _counts(path):
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-count_packets", "-show_entries",
         "stream=codec_type,nb_read_packets", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    return {k: int(v) for k, v in (line.split(",") for line in probe)}


@pytest.mark.parametrize("selective", [True, False])
def test_vlc_is_fed_the_whole_video_as_mpegts(tmp_path, config, monkeypatch, selective):
    from dataclasses import replace

    video = tmp_path / "in.mp4"
    subprocess.run(
        [FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i",
         f"testsrc=size=320x240:rate=25:duration={FRAMES / 25}",
         "-f", "lavfi", "-i", f"sine=frequency=440:duration={FRAMES / 25}",
         "-c:v", "libx264", "-g", "10", "-bf", "0", "-pix_fmt", "yuv420p",
         "-c:a", "aac", str(video)],
        check=True,
    )
    captured = _fake_vlc(tmp_path, monkeypatch)
    config = replace(config, render=replace(config.render, selective=selective))
    out = tmp_path / "out.mp4"
    run_pipeline_streaming(video, "person", out, config, meta_path=tmp_path / "m.json",
                           thumbs_dir=tmp_path / "thumbs", tracker=ScriptedTracker(), vlc=True)

    counts = _counts(captured)
    assert counts["video"] == FRAMES
    assert counts["audio"] > 0
    assert not out.with_suffix(".ts").exists()


def test_vlc_without_streaming_fails_immediately(capsys):
    from auto_redact.cli import main

    with pytest.raises(SystemExit) as exit_info:
        main(["run", "--video", "x.mp4", "--prompt", "person", "--out", "o.mp4",
              "--vlc", "--no-stats", "--no-inhibit"])
    assert exit_info.value.code == 2
    assert "--vlc requires --streaming" in capsys.readouterr().err


def test_vlc_does_not_inherit_opencvs_qt_plugin_paths(tmp_path, monkeypatch):
    """Importing cv2 points QT_QPA_* at its own Qt plugins, which abort a child
    VLC's Qt interface: the process must not see them."""
    from auto_redact.vlc import LivePlayer

    seen = tmp_path / "env.txt"
    bindir = tmp_path / "bin"
    bindir.mkdir()
    script = bindir / "vlc"
    script.write_text(f"#!/bin/sh\nenv > {seen}\ncat > /dev/null\n")
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}:{__import__('os').environ['PATH']}")
    monkeypatch.setenv("QT_QPA_PLATFORM_PLUGIN_PATH", "/opencv/qt/plugins")
    monkeypatch.setenv("QT_QPA_FONTDIR", "/opencv/qt/fonts")
    source = tmp_path / "s.ts"
    source.write_bytes(b"")
    player = LivePlayer(source)
    player.start()
    player.finish()
    player.wait()
    assert "QT_QPA" not in seen.read_text()


def test_vlc_dying_early_is_reported(tmp_path, monkeypatch):
    from auto_redact.vlc import LivePlayer

    bindir = tmp_path / "bin"
    bindir.mkdir()
    script = bindir / "vlc"
    script.write_text("#!/bin/sh\necho 'qt plugin exploded' >&2\nexit 134\n")
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}:{__import__('os').environ['PATH']}")
    messages = []
    source = tmp_path / "s.ts"
    source.write_bytes(b"")
    player = LivePlayer(source, messages.append)
    player.start()
    player.wait()
    text = "\n".join(messages)
    assert "exited unexpectedly (code 134)" in text
    assert "qt plugin exploded" in text


def test_snapshot_is_rebuilt_from_the_video_for_a_resumed_track(clip, tmp_path):
    """A tracklet restored from a checkpoint has its box and best frame but no
    crop in memory; it must be re-read from the video, not left undecided."""
    from pathlib import Path

    from auto_redact.meta import Tracklet, VideoInfo
    from auto_redact.streaming import _Decider, _Frontier, _Track

    info = probe_video(clip)
    frontier = _Frontier(
        VideoInfo(path=str(clip), fps=info.fps, width=info.width, height=info.height,
                  frames=info.frame_count),
        max_gap=0, smooth_window=1, bridge_iou=0.5, min_track_frames=1,
    )
    decider = _Decider(
        frontier=frontier, info=info, thumbs=tmp_path, prompt="person", padding=0.2,
        isolate=False, sub_isolate=False, closeup_area_fraction=0.3, same_person_iou=0.5,
        isolate_dilate=0.1, classifier=None, prefilter_clf=None, prefilter="none", workers=1,
    )
    tl = Tracklet(track_id=1, frames=[30], boxes=[[10, 20, 90, 200]], keypoints=[[]])
    tl.best_frame = 30
    track = _Track(uid=1, tracker_id=1, tl=tl, closed=True, best_box=[10, 20, 90, 200])

    path = decider._snapshot(track)

    assert path is not None and Path(path).is_file()
    assert tl.best_snapshot_path == path


def test_resumed_run_classifies_tracklets_that_were_pending_when_interrupted(
    clip, tmp_path, config, monkeypatch
):
    """Interrupt while a closed tracklet still awaits classification, then resume:
    its saved crop must be reloaded, not left 'no usable snapshot' (redacted by policy)."""
    import json
    import threading

    release = threading.Event()

    class Blocking(FakeSiglip):
        def classify(self, path, prompt):
            release.wait(10)
            return super().classify(path, prompt)

    monkeypatch.setattr("auto_redact.siglip.SiglipClassifier", Blocking)

    class Interrupted(ScriptedTracker):
        def update(self, frame):
            if self.frame >= 70:
                raise KeyboardInterrupt
            return super().update(frame)

    kw = dict(meta_path=tmp_path / "m.json", thumbs_dir=tmp_path / "thumbs", checkpoint_interval=10)
    try:
        with pytest.raises(KeyboardInterrupt):
            run_pipeline_streaming(clip, "person", tmp_path / "out.mp4", config,
                                   tracker=Interrupted(), **kw)
    finally:
        release.set()
    ckpt = json.loads((tmp_path / "m.json.stream.track.ckpt.json").read_text())
    pending = [t for t in ckpt["closed"] if t["redact"] is None]
    assert pending and (tmp_path / "m.json.stream.track.ckpt.json.crops" / f"{pending[0]['uid']}.png").is_file()

    monkeypatch.setattr("auto_redact.siglip.SiglipClassifier", FakeSiglip)

    class Resumed(ScriptedTracker):
        def __init__(self):
            super().__init__()
            self.frame = ckpt["last_frame"]

    from auto_redact.streaming import _Decider

    crop_ready = {}
    original = _Decider._snapshot

    def spy(self, track):
        crop_ready[track.uid] = track.best_crop is not None
        return original(self, track)

    monkeypatch.setattr(_Decider, "_snapshot", spy)
    run_pipeline_streaming(clip, "person", tmp_path / "out.mp4", config, tracker=Resumed(), **kw)
    assert crop_ready[pending[0]["uid"]], "crop should be restored from disk, not re-decoded"
    meta = load_meta(tmp_path / "m.json")
    assert all("no usable snapshot" not in (t.detail or "") for t in meta.tracklets)
    assert all(t.redact is not None for t in meta.tracklets)
    assert not (tmp_path / "m.json.stream.track.ckpt.json.crops").exists()


def test_vlc_closing_early_is_reported_even_when_it_exits_cleanly(tmp_path, monkeypatch):
    from auto_redact.vlc import LivePlayer

    bindir = tmp_path / "bin"
    bindir.mkdir()
    script = bindir / "vlc"
    script.write_text("#!/bin/sh\nexit 0\n")
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}:{__import__('os').environ['PATH']}")
    messages = []
    source = tmp_path / "s.ts"
    source.write_bytes(b"")
    player = LivePlayer(source, messages.append)
    player.start()
    player.wait()
    assert any("closed early" in m for m in messages)


def test_vlc_seek_hotkeys_are_unbound(tmp_path, monkeypatch):
    from auto_redact.vlc import LivePlayer

    seen = tmp_path / "args.txt"
    bindir = tmp_path / "bin"
    bindir.mkdir()
    script = bindir / "vlc"
    script.write_text(f'#!/bin/sh\necho "$@" > {seen}\ncat > /dev/null\n')
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}:{__import__('os').environ['PATH']}")
    source = tmp_path / "s.ts"
    source.write_bytes(b"")
    player = LivePlayer(source)
    player.start()
    player.finish()
    player.wait()
    args = seen.read_text()
    assert "--key-jump-short=Unset" in args and "--key-jump+long=Unset" in args


class PanningTracker:
    """One person, one tracker id, filmed from the hips up: the camera pans
    from the lower body to the face at ``change_at`` with no cut."""

    lost_buffer = 5

    def __init__(self, change_at):
        self.frame, self.change_at = -1, change_at

    def update(self, frame):
        from tests.test_views import LOWER, UPPER

        self.frame += 1
        kps = LOWER if self.frame < self.change_at else UPPER
        return [Detection(1, [10 + self.frame, 20, 90 + self.frame, 200], kps)]


@pytest.fixture
def view_config(config):
    import dataclasses

    return dataclasses.replace(
        config, track=dataclasses.replace(config.track, view_split=True)
    )


def test_staged_and_streaming_split_a_pan_into_the_same_views(clip, tmp_path, view_config, monkeypatch):
    monkeypatch.setattr("auto_redact.track.make_tracker", lambda *a, **k: PanningTracker(30))
    staged, streamed = tmp_path / "staged", tmp_path / "streamed"
    staged.mkdir()
    streamed.mkdir()
    run_pipeline(clip, "person", staged / "out.mp4", view_config,
                 meta_path=staged / "m.json", thumbs_dir=staged / "thumbs")
    run_pipeline_streaming(clip, "person", streamed / "out.mp4", view_config,
                           meta_path=streamed / "m.json", thumbs_dir=streamed / "thumbs",
                           tracker=PanningTracker(30))
    a, b = load_meta(staged / "m.json"), load_meta(streamed / "m.json")
    assert len(a.tracklets) == len(b.tracklets) == 2
    for x, y in zip(a.tracklets, b.tracklets):
        assert x.frames == y.frames
        assert x.best_frame == y.best_frame
    assert a.tracklets[0].frames[-1] + 1 == a.tracklets[1].frames[0]  # no gap at the seam
    # Each view was snapshotted from its own frames: the first piece's best
    # frame is from the lower-body view, not the later face view.
    assert b.tracklets[0].best_frame < 30 <= b.tracklets[1].best_frame


def test_view_split_is_off_by_default(clip, tmp_path, config):
    run_pipeline_streaming(clip, "person", tmp_path / "out.mp4", config,
                           tracker=PanningTracker(30))
    assert len(load_meta(workspace_for(clip) / "meta.json").tracklets) == 1


def test_a_short_tail_inherits_the_verdict_before_it(clip, tmp_path, view_config):
    # The pan happens 6 frames before the person leaves: too short a view to
    # snapshot, and under min_track_frames (it would be dropped from the video).
    run_pipeline_streaming(clip, "person", tmp_path / "out.mp4", view_config,
                           tracker=PanningTracker(74))
    meta = load_meta(workspace_for(clip) / "meta.json")
    assert len(meta.tracklets) == 2
    head, tail = meta.tracklets
    assert len(tail.frames) < view_config.track.view_min_frames
    assert head.redact is True  # FakeSiglip redacts track_1
    assert tail.redact is True  # ...and the tail did not get its own (False) verdict
    assert "inherited" in (tail.detail or "")
    frames = dict(iter_frames(tmp_path / "out.mp4"))
    assert frames[79][60:150, 100:160].max() < 30  # the tail's frames are blacked out too


def test_crop_writer_thread_is_stopped_and_the_checkpoint_cleaned_up(clip, tmp_path, config):
    import threading

    run_pipeline_streaming(clip, "person", tmp_path / "out.mp4", config,
                           meta_path=tmp_path / "m.json", thumbs_dir=tmp_path / "thumbs",
                           tracker=ScriptedTracker())
    assert not [t for t in threading.enumerate() if t.name.startswith("crop-save")]
    assert not (tmp_path / "m.json.stream.track.ckpt.json.crops").exists()
    assert not (tmp_path / "m.json.stream.track.ckpt.json").exists()
