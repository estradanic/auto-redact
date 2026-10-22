"""Tests for the end-to-end ``run`` command (#16).

Tracking is stubbed (no YOLO weights needed); ollama is the stub server
from test_classify.
"""

import pytest

from auto_redact import pipeline
from auto_redact.cli import main
from auto_redact.meta import load_meta
from auto_redact.frames import probe_video

from .test_classify import StubOllama
from .test_snapshot import make_clip, make_meta, make_tracklet


@pytest.fixture
def setup(tmp_path, monkeypatch):
    video = make_clip(tmp_path / "in.mp4")
    calls = {"track": 0}

    def fake_track(path, **kwargs):
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
    meta = load_meta(tmp_path / "out.meta.json")
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

    monkeypatch.setattr(pipeline, "SiglipClassifier", FakeSiglip)
    args = ("--classifier", "siglip", "--classify-device", "cpu")

    assert run(video, tmp_path, stub, *args, "--threshold", "0.3") == 0
    meta = load_meta(tmp_path / "out.meta.json")
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

    pipeline.run_pipeline(
        video, "red shirt", tmp_path / "out.mp4", ollama_url=stub.url, classify_workers=1, scene_cuts=False, bar=bar
    )
    assert seen == ["track", "snapshot", "classify", "render"]
