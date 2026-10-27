"""Tests for the end-to-end ``run`` command (#16).

Tracking is stubbed (no YOLO weights needed); ollama is the stub server
from test_classify.
"""

import pytest

from auto_redact import pipeline
from auto_redact.cli import main
from auto_redact.config import ClassifyConfig, Config, TrackConfig
from auto_redact.meta import load_meta
from auto_redact.frames import probe_video
from auto_redact.workspace import workspace_for

from .test_classify import StubOllama
from .test_snapshot import make_clip, make_meta, make_tracklet


@pytest.fixture
def setup(tmp_path, monkeypatch):
    video = make_clip(tmp_path / "in.mp4")
    calls = {"track": 0}

    def fake_track(path, config=None, **kwargs):
        calls["track"] += 1
        return make_meta(
            video,
            [
                make_tracklet(1, range(0, 10), [[20, 20, 120, 200]] * 10),
                make_tracklet(2, range(5, 15), [[150, 30, 250, 220]] * 10),
            ],
        )

    monkeypatch.setattr(pipeline, "build_tracklets", fake_track)
    stub = StubOllama(["yes", "no"])
    yield video, tmp_path, calls, stub
    stub.close()


def run(video, tmp_path, stub, *extra, prompt="red shirt"):
    return main(
        ["run", "--video", str(video), "--prompt", prompt,
         "--out", str(tmp_path / "out.mp4"), "--ollama-url", stub.url,
         "--classify-workers", "1", "--no-scene-cuts", *extra]
    )


def test_run_produces_video_and_reuses_cache(setup, capsys):
    video, tmp_path, calls, stub = setup
    assert run(video, tmp_path, stub) == 0
    out = tmp_path / "out.mp4"
    assert probe_video(out).frame_count == probe_video(video).frame_count
    assert "2 tracklet(s), 1 redacted" in capsys.readouterr().out
    meta = load_meta(workspace_for(video) / "meta.json")
    assert [t.redact for t in meta.tracklets] == [True, False]
    requests = len(stub.requests)

    assert run(video, tmp_path, stub) == 0
    assert calls["track"] == 1
    assert len(stub.requests) == requests
    assert "reused: track, snapshot, classify" in capsys.readouterr().out


def test_new_prompt_reclassifies_only(setup):
    video, tmp_path, calls, stub = setup
    run(video, tmp_path, stub)
    requests = len(stub.requests)
    run(video, tmp_path, stub, prompt="blue hat")
    assert calls["track"] == 1
    assert len(stub.requests) == requests + 2


def test_force_redoes_everything(setup):
    video, tmp_path, calls, stub = setup
    run(video, tmp_path, stub)
    run(video, tmp_path, stub, "--force")
    assert calls["track"] == 2


def test_siglip_run_and_cache_key(setup, monkeypatch, capsys):
    video, tmp_path, calls, stub = setup
    seen = []

    class FakeSiglip:
        def __init__(self, model, threshold, device):
            seen.append((model, threshold, device))
            self.threshold = threshold

        def classify(self, path, prompt):
            from auto_redact.classify import Decision

            return Decision(self.threshold < 0.5, 0.4, "fake")

    monkeypatch.setattr("auto_redact.siglip.SiglipClassifier", FakeSiglip)
    args = ("--classifier", "siglip", "--classify-device", "cpu")

    assert run(video, tmp_path, stub, *args, "--threshold", "0.3") == 0
    meta = load_meta(workspace_for(video) / "meta.json")
    assert [t.redact for t in meta.tracklets] == [True, True]
    assert [t.score for t in meta.tracklets] == [0.4, 0.4]
    assert seen == [("google/siglip2-base-patch16-224", 0.3, "cpu")]
    assert stub.requests == []

    run(video, tmp_path, stub, *args, "--threshold", "0.3")  # cached
    assert len(seen) == 1
    run(video, tmp_path, stub, *args, "--threshold", "0.9")  # new threshold
    assert len(seen) == 2
    run(video, tmp_path, stub)  # back to the VLM: must not reuse siglip flags
    assert len(stub.requests) == 2


def test_run_drives_progress_bars(setup):
    from contextlib import contextmanager

    video, tmp_path, calls, stub = setup
    seen = []

    @contextmanager
    def bar(desc):
        seen.append(desc)
        yield None

    config = Config(
        track=TrackConfig(scene_cuts=False),
        classify=ClassifyConfig(ollama_url=stub.url, workers=1),
    )
    pipeline.run_pipeline(video, "red shirt", tmp_path / "out.mp4", config, bar=bar)
    assert seen == ["track", "snapshot", "classify", "render"]


def test_changed_snapshot_settings_resnapshot_and_reclassify(setup):
    video, tmp_path, calls, stub = setup
    run(video, tmp_path, stub)
    requests = len(stub.requests)
    run(video, tmp_path, stub, "--padding", "0.4")
    assert calls["track"] == 1  # tracking settings did not change
    assert len(stub.requests) == requests + 2  # new thumbnails, new decisions


def test_changed_track_settings_retrack(setup):
    video, tmp_path, calls, stub = setup
    run(video, tmp_path, stub)
    run(video, tmp_path, stub, "--stride", "2")
    assert calls["track"] == 2


def test_execution_only_settings_keep_the_cache(setup, capsys):
    video, tmp_path, calls, stub = setup
    run(video, tmp_path, stub)
    requests = len(stub.requests)
    run(video, tmp_path, stub, "--timeout", "30", "--device", "cpu", "--keep-alive", "1m")
    assert calls["track"] == 1
    assert len(stub.requests) == requests
    assert "reused: track, snapshot, classify" in capsys.readouterr().out


def test_metadata_from_before_the_manifest_is_adopted_once(setup, capsys):
    video, tmp_path, calls, stub = setup
    run(video, tmp_path, stub)
    manifest = workspace_for(video) / "meta.json.cache.json"
    legacy_sidecar = workspace_for(video) / "meta.json.classify.json"
    legacy_sidecar.write_text('{"prompt": "red shirt", "model": "gemma3:4b"}\n')
    requests = len(stub.requests)
    manifest.unlink()
    capsys.readouterr()

    run(video, tmp_path, stub)
    assert calls["track"] == 1
    assert len(stub.requests) == requests
    assert "predates the stage cache" in capsys.readouterr().out
    assert manifest.is_file()  # from now on the cache is checked normally
    run(video, tmp_path, stub, "--stride", "2")
    assert calls["track"] == 2


def test_run_data_stays_out_of_the_output_folder(setup):
    video, tmp_path, calls, stub = setup
    run(video, tmp_path, stub)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["in.mp4", "out.mp4"]
    workspace = workspace_for(video)
    assert (workspace / "meta.json").is_file()
    assert (workspace / "meta.npz").is_file()
    assert (workspace / "meta.json.cache.json").is_file()
    assert (workspace / "thumbs" / "contact_sheet.jpg").is_file()


def test_different_outputs_share_one_videos_tracking(setup):
    video, tmp_path, calls, stub = setup
    run(video, tmp_path, stub)
    main(["run", "--video", str(video), "--prompt", "red shirt",
          "--out", str(tmp_path / "second.mp4"), "--ollama-url", stub.url,
          "--classify-workers", "1", "--no-scene-cuts"])
    assert calls["track"] == 1
    assert (tmp_path / "second.mp4").is_file()


def test_metadata_next_to_the_output_from_older_versions_is_used_in_place(setup, capsys):
    video, tmp_path, calls, stub = setup
    # What older versions left behind: metadata beside the output video.
    run(video, tmp_path, stub, "--meta", str(tmp_path / "out.meta.json"))
    capsys.readouterr()

    run(video, tmp_path, stub)  # no --meta: finds and keeps using it
    out = capsys.readouterr().out
    assert "using existing metadata next to the output" in out
    assert "reused: track, snapshot, classify" in out
    assert calls["track"] == 1
    assert not workspace_for(video).exists()


def test_explicit_meta_path_wins(setup):
    video, tmp_path, calls, stub = setup
    meta = tmp_path / "custom" / "m.json"
    run(video, tmp_path, stub, "--meta", str(meta))
    assert meta.is_file() and (tmp_path / "custom" / "m_thumbs").is_dir()
    assert not workspace_for(video).exists()


def write_labels(video, track_key, verdicts):
    from auto_redact.labels import Labels, save_labels
    from auto_redact.workspace import labels_path_for

    labels = Labels(track_key=track_key)
    for tid, redact in verdicts.items():
        labels.set_track(tid, redact)
    save_labels(labels_path_for(workspace_for(video) / "meta.json"), labels)


def stored_track_key(video):
    from auto_redact import cache

    return cache.StageCache(workspace_for(video) / "meta.json").stages["track"]


def test_review_labels_are_applied_by_the_next_run(setup, capsys):
    video, tmp_path, calls, stub = setup
    run(video, tmp_path, stub)
    assert "1 redacted" in capsys.readouterr().out  # stub says yes, no

    write_labels(video, stored_track_key(video), {2: True})  # reviewer: redact the second person too
    run(video, tmp_path, stub)
    out = capsys.readouterr().out
    assert "[labels] applying 1 review verdict" in out and "2 redacted" in out
    assert calls["track"] == 1 and "reused: track, snapshot, classify" in out

    meta = load_meta(workspace_for(video) / "meta.json")
    assert [(t.redact, t.override) for t in meta.tracklets] == [(True, None), (False, True)]


def test_a_reviewer_can_keep_someone_the_classifier_flagged(setup, capsys):
    video, tmp_path, calls, stub = setup
    run(video, tmp_path, stub)
    write_labels(video, stored_track_key(video), {1: False})
    capsys.readouterr()
    run(video, tmp_path, stub)
    assert "0 redacted" in capsys.readouterr().out


def test_labels_for_other_tracking_are_ignored_with_a_warning(setup, capsys):
    video, tmp_path, calls, stub = setup
    run(video, tmp_path, stub)
    write_labels(video, "tracking-from-another-run", {1: False, 2: True})
    capsys.readouterr()
    run(video, tmp_path, stub)
    out = capsys.readouterr().out
    assert "different tracking; ignoring" in out and "1 redacted" in out


def test_reclassifying_never_discards_a_review_override(setup):
    video, tmp_path, calls, stub = setup
    run(video, tmp_path, stub)
    write_labels(video, stored_track_key(video), {2: True})
    run(video, tmp_path, stub)
    run(video, tmp_path, stub, prompt="something else")  # new prompt: re-classifies everyone
    meta = load_meta(workspace_for(video) / "meta.json")
    assert meta.tracklets[1].override is True and meta.tracklets[1].effective_redact is True


def test_review_command_needs_stored_tracking(tmp_path, capsys):
    missing = tmp_path / "never_run.mkv"
    missing.write_bytes(b"x")
    assert main(["review", "--video", str(missing), "--no-stats", "--no-browser"]) == 1
    assert "run `auto-redact run` on it first" in capsys.readouterr().err
