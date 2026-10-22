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


def test_classify_unparseable_reply_leaves_redact_unset(meta, ollama, caplog) -> None:
    stub = ollama(["hmm"])

    with caplog.at_level(logging.WARNING, logger="auto_redact.classify"):
        classify_snapshots(meta, "x", ollama_url=stub.url)

    assert [t.redact for t in meta.tracklets] == [None, None]
    assert "unparseable" in caplog.text


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
