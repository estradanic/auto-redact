"""Tests for auto_redact.classify: VLM yes/no via ollama (#13).

A stub HTTP server stands in for ollama, so the real request path is
exercised without a model or network.
"""

import base64
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
from PIL import Image

from auto_redact.classify import (
    ClassifyError,
    Decision,
    build_question,
    classify_snapshots,
    parse_yes_no,
)
from auto_redact.cli import main
from auto_redact.meta import Metadata, Tracklet, VideoInfo, load_meta, save_meta


class StubOllama:
    """Replies per request from ``replies`` (cycled), recording requests."""

    def __init__(self, replies, status=200):
        self.replies = list(replies)
        self.status = status
        self.requests = []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers["Content-Length"])
                stub.requests.append(
                    {"path": self.path, "body": json.loads(self.rfile.read(length))}
                )
                reply = stub.replies[(len(stub.requests) - 1) % len(stub.replies)]
                payload = json.dumps(
                    {"response": reply} if stub.status == 200 else {"error": reply}
                ).encode()
                self.send_response(stub.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def ollama():
    stubs = []

    def make(replies, status=200):
        stub = StubOllama(replies, status)
        stubs.append(stub)
        return stub

    yield make
    for stub in stubs:
        stub.close()


@pytest.fixture
def meta(tmp_path: Path) -> Metadata:
    tracklets = []
    for track_id in (1, 2):
        thumb = tmp_path / f"track_{track_id}.jpg"
        Image.new("RGB", (8, 8), (track_id * 50, 0, 0)).save(thumb)
        tracklets.append(
            Tracklet(
                track_id=track_id,
                frames=[0],
                boxes=[[0, 0, 8, 8]],
                keypoints=[[]],
                best_frame=0,
                best_snapshot_path=str(thumb),
            )
        )
    return Metadata(
        video=VideoInfo(path="v.mp4", fps=25, width=8, height=8, frames=1),
        tracklets=tracklets,
    )


@pytest.mark.parametrize(
    "reply, expected",
    [
        ("Yes", True),
        ("yes.", True),
        ("  No\n", False),
        ("NO, she is wearing blue.", False),
        ("Yes, the person wears a red shirt.", True),
        ("I cannot tell", None),
        ("", None),
        ("nobody here", None),  # "no" inside a word doesn't count
    ],
)
def test_parse_yes_no(reply, expected) -> None:
    assert parse_yes_no(reply) is expected


def test_build_question_embeds_prompt() -> None:
    assert build_question("person in a red shirt") == (
        "Does this image show a person matching: 'person in a red shirt'? "
        "Answer yes or no."
    )


def test_classify_sets_redact_flags_and_sends_image(meta, ollama) -> None:
    stub = ollama(["Yes", "No"])

    result = classify_snapshots(
        meta, "person in a red shirt", model="mymodel", ollama_url=stub.url,
        workers=1,
    )

    assert result is meta
    assert [t.redact for t in meta.tracklets] == [True, False]
    first = stub.requests[0]
    assert first["path"] == "/api/generate"
    assert first["body"]["model"] == "mymodel"
    assert first["body"]["stream"] is False
    assert "person in a red shirt" in first["body"]["prompt"]
    expected_image = Path(meta.tracklets[0].best_snapshot_path).read_bytes()
    assert base64.b64decode(first["body"]["images"][0]) == expected_image


def test_classify_logs_decision_and_raw_reply(meta, ollama, caplog) -> None:
    stub = ollama(["Yes, definitely", "no"])

    with caplog.at_level(logging.INFO, logger="auto_redact.classify"):
        classify_snapshots(meta, "x", ollama_url=stub.url, workers=1)

    assert "track 1: redact=True" in caplog.text
    assert "Yes, definitely" in caplog.text
    assert "track 2: redact=False" in caplog.text


def test_classify_unparseable_reply_is_redacted_by_default(meta, ollama, caplog) -> None:
    stub = ollama(["hmm"])

    with caplog.at_level(logging.INFO, logger="auto_redact.classify"):
        classify_snapshots(meta, "x", ollama_url=stub.url)

    assert [t.redact for t in meta.tracklets] == [True, True]
    assert "undecided, redacting by policy" in caplog.text
    assert "hmm" in caplog.text  # the raw reply is still in the log


def test_classification_records_the_reason(meta, ollama) -> None:
    stub = ollama(["Yes, red shirt", "hmm"])
    classify_snapshots(meta, "x", ollama_url=stub.url)
    assert "Yes, red shirt" in meta.tracklets[0].detail
    assert "undecided" in meta.tracklets[1].detail and "hmm" in meta.tracklets[1].detail


def test_reclassifying_keeps_human_overrides(meta, ollama) -> None:
    meta.tracklets[0].override = False
    stub = ollama(["yes", "yes"])
    classify_snapshots(meta, "x", ollama_url=stub.url, resume=False)
    assert meta.tracklets[0].redact is True and meta.tracklets[0].override is False
    assert meta.tracklets[0].effective_redact is False


def test_on_undecided_keep_leaves_redact_unset(meta, ollama, caplog) -> None:
    stub = ollama(["hmm"])

    with caplog.at_level(logging.WARNING, logger="auto_redact.classify"):
        classify_snapshots(meta, "x", ollama_url=stub.url, on_undecided="keep")

    assert [t.redact for t in meta.tracklets] == [None, None]
    assert "unparseable" in caplog.text


def test_on_undecided_error_fails_after_saving_the_rest(meta, ollama, tmp_path) -> None:
    from auto_redact.meta import load_meta, save_meta

    stub = ollama(["yes", "hmm"])
    path = save_meta(tmp_path / "m.json", meta)

    with pytest.raises(ClassifyError, match=r"1 tracklet\(s\) could not be classified: \[2\]"):
        classify_snapshots(meta, "x", ollama_url=stub.url, on_undecided="error",
                           meta_path=path)

    saved = load_meta(path)
    assert [t.redact for t in saved.tracklets] == [True, None]


def test_undecided_policy_keeps_clear_answers_untouched(meta, ollama) -> None:
    stub = ollama(["no", "hmm"])
    classify_snapshots(meta, "x", ollama_url=stub.url)
    assert [t.redact for t in meta.tracklets] == [False, True]


def test_classifier_exceptions_in_a_prefilter_are_undecided_too() -> None:
    from auto_redact.classify import resolve_undecided

    undecided = Decision(None, None, "siglip error")
    assert resolve_undecided(undecided, "redact").redact is True
    assert resolve_undecided(undecided, "keep").redact is None
    assert resolve_undecided(Decision(False, 0.1, "x"), "redact").redact is False


def test_classify_requires_snapshots(meta, ollama) -> None:
    meta.tracklets[1].best_snapshot_path = None

    with pytest.raises(ClassifyError, match=r"\[2\].*snapshot"):
        classify_snapshots(meta, "x", ollama_url=ollama(["yes"]).url)


def test_classify_missing_snapshot_file(meta, ollama) -> None:
    Path(meta.tracklets[0].best_snapshot_path).unlink()

    with pytest.raises(FileNotFoundError, match="no such snapshot"):
        classify_snapshots(meta, "x", ollama_url=ollama(["yes"]).url)


def test_classify_rejects_empty_prompt(meta) -> None:
    with pytest.raises(ClassifyError, match="prompt"):
        classify_snapshots(meta, "  ")


def test_classify_http_error(meta, ollama) -> None:
    stub = ollama(["model 'nope' not found"], status=404)

    with pytest.raises(ClassifyError, match="HTTP 404.*not found"):
        classify_snapshots(meta, "x", model="nope", ollama_url=stub.url)


def test_classify_unreachable_server(meta) -> None:
    with pytest.raises(ClassifyError, match="could not reach ollama"):
        classify_snapshots(meta, "x", ollama_url="http://127.0.0.1:1")


def test_cli_classify_updates_meta(meta, ollama, tmp_path: Path, capsys) -> None:
    stub = ollama(["yes", "no"])
    meta_path = tmp_path / "meta.json"
    save_meta(meta_path, meta)

    result = main(
        ["classify", "--meta", str(meta_path), "--prompt", "red shirt",
         "--ollama-url", stub.url, "--model", "m", "--classify-workers", "1"]
    )

    assert result == 0
    assert [t.redact for t in load_meta(meta_path).tracklets] == [True, False]
    assert stub.requests[0]["body"]["model"] == "m"
    assert "1 to redact" in capsys.readouterr().out


def test_cli_classify_reports_errors(meta, tmp_path: Path) -> None:
    meta_path = tmp_path / "meta.json"
    save_meta(meta_path, meta)

    result = main(
        ["classify", "--meta", str(meta_path), "--prompt", "x",
         "--ollama-url", "http://127.0.0.1:1"]
    )

    assert result == 1


def test_cli_classify_requires_prompt() -> None:
    with pytest.raises(SystemExit):
        main(["classify", "--meta", "m.json"])


def test_pluggable_classifier_stores_score(meta) -> None:
    class Fake:
        def classify(self, image_path, prompt):
            return Decision(True, 0.9, "fake")

    classify_snapshots(meta, "x", classifier=Fake())

    assert [t.redact for t in meta.tracklets] == [True, True]
    assert [t.score for t in meta.tracklets] == [0.9, 0.9]


def test_classify_reports_progress(meta) -> None:
    class Fake:
        def classify(self, image_path, prompt):
            return Decision(True)

    calls = []
    classify_snapshots(
        meta, "x", classifier=Fake(), on_progress=lambda d, t: calls.append((d, t))
    )

    assert calls == [(0, 2), (1, 2), (2, 2)]


def test_concurrent_classification_matches_sequential(meta) -> None:
    class Fake:
        thread_safe = True

        def classify(self, image_path, prompt):
            return Decision("1" in Path(image_path).name, None, "fake")

    seen = []
    classify_snapshots(
        meta, "x", classifier=Fake(), workers=4,
        on_progress=lambda d, t: seen.append((d, t)),
    )

    expected = ["1" in Path(t.best_snapshot_path).name for t in meta.tracklets]
    assert [t.redact for t in meta.tracklets] == expected
    assert seen[-1] == (2, 2)


def test_invalid_workers_rejected(meta) -> None:
    with pytest.raises(ValueError, match="workers"):
        classify_snapshots(meta, "x", workers=0)


def test_vram_reserve_explicit_value_skips_detection(monkeypatch) -> None:
    from auto_redact import classify
    from auto_redact.config import ClassifyConfig

    monkeypatch.setattr(
        classify, "vlm_shares_gpu", lambda cfg: pytest.fail("should not probe")
    )
    assert classify.resolve_vlm_vram_reserve(ClassifyConfig(vlm_vram_reserve_mb=123)) == 123
    assert classify.resolve_vlm_vram_reserve(ClassifyConfig(vlm_vram_reserve_mb=0)) == 0


@pytest.mark.parametrize(
    "shares, expected",
    [(True, 2500), (None, 2500), (False, 0)],
)
def test_vram_reserve_auto(monkeypatch, shares, expected) -> None:
    from auto_redact import classify
    from auto_redact.config import ClassifyConfig

    monkeypatch.setattr(classify, "vlm_shares_gpu", lambda cfg: shares)
    assert classify.resolve_vlm_vram_reserve(ClassifyConfig()) == expected


def test_remote_ollama_never_shares_this_gpu() -> None:
    from auto_redact.classify import vlm_shares_gpu
    from auto_redact.config import ClassifyConfig

    assert vlm_shares_gpu(ClassifyConfig(ollama_url="http://192.0.2.1:11434")) is False


@pytest.mark.parametrize("pids, expected", [({4242}, True), (set(), False), (None, None)])
def test_local_ollama_sharing_depends_on_gpu_processes(ollama, monkeypatch, pids, expected) -> None:
    from auto_redact import classify
    from auto_redact.config import ClassifyConfig

    stub = ollama([""])
    monkeypatch.setattr(classify, "_ollama_gpu_pids", lambda: pids)
    config = ClassifyConfig(ollama_url=stub.url, model="m")

    assert classify.vlm_shares_gpu(config) is expected
    assert stub.requests[0]["body"]["model"] == "m"  # model preloaded first


def test_unreachable_local_ollama_is_undetermined(monkeypatch) -> None:
    from auto_redact import classify
    from auto_redact.config import ClassifyConfig

    monkeypatch.setattr(classify, "_ollama_gpu_pids", lambda: set())
    assert classify.vlm_shares_gpu(ClassifyConfig(ollama_url="http://127.0.0.1:9")) is None


def test_decision_metrics_record_scores_routing_and_verdicts(tmp_path, monkeypatch) -> None:
    from auto_redact import telemetry
    from auto_redact.classify import record_decision_metrics

    monkeypatch.setenv("AUTO_REDACT_STATS", "1")
    monkeypatch.setenv("AUTO_REDACT_STATS_DIR", str(tmp_path))
    run = telemetry.begin("test", {})
    assert run is not None
    try:
        settled = Decision(False, 0.12, "score 0.120 vs threshold 0.3")
        record_decision_metrics(1, settled, settled, "positives")
        asked = Decision(True, 0.42, "score 0.420 vs threshold 0.3")
        record_decision_metrics(2, asked, Decision(False, None, "No."), "positives")
        record_decision_metrics(3, None, Decision(True, None, "Yes."), "none")

        stats = run.stats
        assert stats["dists"]["prefilter.score"]["count"] == 2
        assert stats["dists"]["prefilter.score"]["bins"][1] == 1  # 0.12 -> 0.1-0.2
        assert stats["dists"]["prefilter.score"]["bins"][4] == 1  # 0.42 -> 0.4-0.5
        assert stats["dists"]["prefilter.score[sent_to_vlm]"]["count"] == 1
        assert stats["counts"]["prefilter.routed"] == {"prefilter": 1, "vlm": 1}
        assert stats["counts"]["decisions[prefilter]"] == {"keep": 1}
        assert stats["counts"]["decisions[vlm]"] == {"keep": 1}
        assert stats["counts"]["decisions[classifier]"] == {"redact": 1}
        assert stats["counts"]["vlm.verdict_by_prefilter_band"] == {"0.4-0.5:keep": 1}
        lines = [json.loads(x) for x in (run.dir / "decisions.jsonl").read_text().splitlines()]
        assert [(x["track"], x["route"], x["verdict"]) for x in lines] == [
            (1, "prefilter", "keep"), (2, "vlm", "keep"), (3, "classifier", "redact"),
        ]
        assert lines[1]["prefilter_score"] == 0.42
    finally:
        telemetry.end(0)


def test_prompt_expression_asks_one_question_per_leaf_through_the_stage(meta, ollama) -> None:
    # Replies cycle: track 1 gets yes then no then yes, track 2 gets no.
    stub = ollama(["Yes", "No", "Yes", "No"])
    prompt = '"a woman" and ("bare thighs" or "lingerie")'

    classify_snapshots(meta, prompt, ollama_url=stub.url, workers=1)

    asked = [r["body"]["prompt"] for r in stub.requests]
    assert ["a woman" in q for q in asked[:3]] == [True, False, False]
    assert "bare thighs" in asked[1] and "lingerie" in asked[2]
    assert "a woman" in asked[3]  # track 2 stops at the first leaf ("No")
    assert len(asked) == 4
    assert [t.redact for t in meta.tracklets] == [True, False]
    assert "'bare thighs': no" in meta.tracklets[0].detail
    assert meta.tracklets[0].detail.endswith("-> yes")
