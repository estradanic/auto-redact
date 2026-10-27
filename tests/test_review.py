"""The review session (logic) and its HTTP adapter."""

import http.client
import json
import shutil
import subprocess
import threading
import time

import cv2
import numpy as np
import pytest
from PIL import Image

from auto_redact import cache
from auto_redact.classify import Decision
from auto_redact.config import ClassifyConfig, RenderConfig
from auto_redact.frames import probe_video
from auto_redact.labels import Labels, load_labels, save_labels
from auto_redact.meta import Metadata, Tracklet, VideoInfo, load_meta, save_meta
from auto_redact.review import Review, ReviewError, make_server
from auto_redact.workspace import labels_path_for

FFMPEG = shutil.which("ffmpeg")
pytestmark = pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not installed")

FRAMES = 60
CONFIG = RenderConfig(smooth_window=1, bridge_iou=0.0, min_track_frames=1, max_gap=3)


def make_track(tid, frames, box, redact, score=None, detail=None):
    return Tracklet(
        tid, list(frames), [list(box) for _ in frames], [[] for _ in frames],
        redact=redact, score=score, detail=detail, best_frame=frames[0],
    )


@pytest.fixture
def workspace(tmp_path):
    video = tmp_path / "v.mp4"
    subprocess.run(
        [FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i",
         f"color=c=white:size=320x240:rate=25:duration={FRAMES / 25}",
         "-pix_fmt", "yuv420p", str(video)],
        check=True,
    )
    info = probe_video(video)
    ws = tmp_path / "ws"
    ws.mkdir()
    thumbs = ws / "thumbs"
    thumbs.mkdir()
    tracks = [
        make_track(1, range(0, 30), (20, 20, 100, 200), True, 0.9, "reply: 'Yes'"),
        make_track(2, range(10, 50), (150, 30, 250, 220), False, 0.2, "reply: 'No'"),
        make_track(3, range(0, 5), (200, 10, 230, 60), None, None, "undecided <script>x</script>"),
    ]
    for t in tracks:
        path = thumbs / f"track_{t.track_id}.jpg"
        Image.new("RGB", (30, 40), (t.track_id * 60, 0, 0)).save(path)
        t.best_snapshot_path = str(path)
    meta = Metadata(VideoInfo(str(video), info.fps, info.width, info.height, info.frame_count), tracks)
    meta_path = ws / "meta.json"
    save_meta(meta_path, meta)
    cache.StageCache(meta_path).record("track", "trackkey")
    return meta_path


@pytest.fixture
def review(workspace):
    return Review(workspace, CONFIG, ClassifyConfig(), out_path=workspace.parent / "out.mp4")


# --- state ---------------------------------------------------------------------


def test_state_summarizes_people_and_decisions(review):
    s = review.state()
    assert s["video"]["frames"] == FRAMES
    assert [t["id"] for t in s["tracks"]] == [1, 2, 3]
    assert s["counts"] == {"tracks": 3, "redact": 1, "keep": 1, "undecided": 1, "overrides": 0}
    t1 = s["tracks"][0]
    assert (t1["first"], t1["last"], t1["n"], t1["score"]) == (0, 29, 30, 0.9)
    assert t1["has_thumb"] and t1["detail"] == "reply: 'Yes'"
    assert s["settings"] == {"max_gap": 3, "smooth_window": 1, "bridge_iou": 0.0, "min_track_frames": 1}
    assert s["has_scores"] and s["out"].endswith("out.mp4")


def test_overriding_a_track_saves_to_meta_and_labels(review, workspace):
    summary = review.set_track(2, True, "that's him")
    assert (summary["override"], summary["effective"], summary["labeled"]) == (True, True, True)
    assert review.state()["counts"]["redact"] == 2 and review.state()["counts"]["overrides"] == 1

    assert load_meta(workspace).tracklets[1].override is True
    labels = load_labels(labels_path_for(workspace))
    assert labels.tracks[2].redact is True and labels.tracks[2].note == "that's him"
    assert labels.track_key == "trackkey"


def test_clearing_an_override_restores_the_classifiers_decision(review, workspace):
    review.set_track(2, True)
    summary = review.set_track(2, None)
    assert (summary["override"], summary["effective"]) == (None, False)
    assert load_meta(workspace).tracklets[1].override is None
    assert 2 not in load_labels(labels_path_for(workspace)).tracks


def test_override_changes_the_painted_boxes(review):
    config = review.config_from(None)
    assert 25 not in {f for f, b in review.boxes_for(config).items() if any(x[0] == 150 for x in b)}
    review.set_track(2, True)
    assert any(x[0] == 150 for x in review.boxes_for(config)[25])


def test_unknown_track_and_bad_values_are_rejected(review):
    with pytest.raises(ReviewError, match="no track"):
        review.set_track(99, True)
    with pytest.raises(ReviewError, match="true, false or null"):
        review.set_track(1, "yes")


def test_frame_labels_round_trip_and_validate(review, workspace):
    i = review.add_frame_label(12, "leak", [10, 10, 60, 120], "arm")
    review.add_frame_label(40, "over")
    labels = load_labels(labels_path_for(workspace))
    assert [(f.frame, f.kind) for f in labels.frames] == [(12, "leak"), (40, "over")]
    assert review.state()["labels"]["frames"][0]["box"] == [10, 10, 60, 120]
    review.delete_frame_label(i)
    assert len(load_labels(labels_path_for(workspace)).frames) == 1
    for bad in ((FRAMES, "leak"), (-1, "leak"), (5, "bogus"), (True, "leak")):
        with pytest.raises(ReviewError):
            review.add_frame_label(*bad)
    with pytest.raises(ReviewError, match="no frame label"):
        review.delete_frame_label(9)


def test_labels_made_for_other_tracking_are_set_aside(workspace):
    old = Labels(track_key="some-older-tracking")
    old.set_track(1, False)
    old.add_frame(3, "leak")
    save_labels(labels_path_for(workspace), old)
    review = Review(workspace, CONFIG)
    state = review.state()
    assert state["labels"]["ignored"] == 2 and state["labels"]["frames"] == []
    assert review.state()["tracks"][0]["labeled"] is None


def test_existing_labels_for_this_tracking_are_loaded(workspace):
    mine = Labels(track_key="trackkey")
    mine.set_track(3, True)
    save_labels(labels_path_for(workspace), mine)
    assert Review(workspace, CONFIG).state()["tracks"][2]["labeled"] is True


# --- settings and tuning ------------------------------------------------------------


def test_settings_are_validated(review):
    assert review.config_from({"max_gap": "10", "bridge_iou": "0.4"}).max_gap == 10
    for bad in ({"nope": 1}, {"max_gap": -1}, {"smooth_window": 4}, {"bridge_iou": 2}, {"min_track_frames": 0}, {"max_gap": "x"}):
        with pytest.raises(ReviewError):
            review.config_from(bad)


def test_preview_compares_labels_under_trial_and_current_settings(review):
    review.add_frame_label(35, "leak", [20, 20, 100, 200])  # track 1 ended at 29: gap of 6
    p = review.preview({"max_gap": 3})
    assert p["trial"]["leaks_covered"] == 0 and p["area_change_pct"] == 0
    # track 1 simply ends before frame 35: no max_gap fixes that, and the diagnosis says so
    a = review.analyze()
    assert a["settings"]["diagnoses"][0]["cause"] == "track_edge"


def test_analyze_finds_a_fixable_gap(workspace):
    meta = load_meta(workspace)
    gappy = make_track(4, [*range(0, 10), *range(20, 30)], (20, 20, 100, 200), True)
    gappy.best_snapshot_path = meta.tracklets[0].best_snapshot_path
    meta.tracklets = [gappy]
    save_meta(workspace, meta)
    review = Review(workspace, CONFIG)
    review.add_frame_label(15, "leak", [20, 20, 100, 200])
    s = review.analyze()["settings"]
    assert s["changes"] == {"max_gap": 10}
    assert s["best_eval"]["leaks_covered"] == 1 and s["base_eval"]["leaks_covered"] == 0
    assert review.preview({"max_gap": 10})["trial"]["leaks_covered"] == 1


def test_try_classifier_scores_only_labeled_people(review, monkeypatch):
    class Fake:
        def classify(self, path, prompt):
            return Decision(path.name == "track_1.jpg", None, "fake")

    seen = {}
    monkeypatch.setattr("auto_redact.review.make_classifier", lambda cfg: seen.setdefault("cfg", cfg) and Fake())
    with pytest.raises(ReviewError, match="label some tracks"):
        review.try_classifier("red shirt")
    review.set_track(1, True)
    review.set_track(2, False)
    out = review.try_classifier("red shirt", {"classifier": "siglip", "threshold": "0.3", "bogus": 1})
    assert out["labeled"] == 2 and out["tried"]["tp"] == 1 and out["tried"]["tn"] == 1
    assert out["current"]["tp"] == 1  # the stored decisions: track 1 redact, track 2 keep
    assert seen["cfg"].classifier == "siglip"
    with pytest.raises(ReviewError, match="prompt"):
        review.try_classifier("  ")


# --- frames ------------------------------------------------------------------------------


def decode(jpeg):
    return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)


def test_frame_jpeg_draws_the_painted_boxes(review):
    raw, shown = decode(review.frame_jpeg(5, overlay=False)), decode(review.frame_jpeg(5))
    assert raw.shape == shown.shape == (240, 320, 3)
    inside = shown[100, 60]  # track 1's painted box (BGR): white tinted red
    assert inside[2] > inside[1] + 20 and raw[100, 60].min() > 240
    assert (shown[230, 300] == raw[230, 300]).all()  # outside any box: untouched


def test_frame_jpeg_follows_the_settings_being_previewed(review):
    review.add_frame_label(35, "leak")
    near = decode(review.frame_jpeg(32, {"max_gap": 3}))
    # frame 32 is past track 1's end: nothing red at its old position either way
    assert near[100, 60][2] - near[100, 60][1] < 20


def test_frame_jpeg_rejects_out_of_range(review):
    with pytest.raises(ReviewError):
        review.frame_jpeg(FRAMES)


def test_thumbnails(review):
    assert review.thumb_path(1).name == "track_1.jpg"
    review.meta.tracklets[2].best_snapshot_path = None
    with pytest.raises(ReviewError, match="no thumbnail"):
        review.thumb_path(3)


# --- rendering ----------------------------------------------------------------------------


def wait_for(review, state, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        status = review.render_status()
        if status["state"] == state:
            return status
        time.sleep(0.1)
    raise AssertionError(review.render_status())


def test_render_uses_overrides_and_the_chosen_settings(review, workspace):
    review.set_track(2, True)
    review.start_render({"max_gap": 3})
    status = wait_for(review, "done")
    out = review.out_path
    assert out.is_file() and probe_video(out).frame_count == FRAMES
    frame = {i: f for i, f in __import__("auto_redact.frames", fromlist=["iter_frames"]).iter_frames(out)}[20]
    assert frame[100, 60].max() < 30 and frame[100, 200].max() < 30  # both people painted black
    assert "wrote" in status["message"]


def test_render_needs_an_output_path(workspace):
    with pytest.raises(ReviewError, match="--out"):
        Review(workspace, CONFIG).start_render()


def test_render_failure_is_reported_not_raised(review):
    review.meta.video.path = "/nonexistent.mp4"
    review.start_render()
    status = wait_for(review, "error")
    assert "FileNotFoundError" in status["message"]


# --- HTTP ------------------------------------------------------------------------------------


@pytest.fixture
def server(review):
    httpd = make_server(review)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd.server_address[1], review
    httpd.shutdown()
    httpd.server_close()


def call(port, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    hdrs = dict(headers or {})
    payload = None
    if body is not None:
        payload = json.dumps(body)
        hdrs.setdefault("Content-Type", "application/json")
    conn.request(method, path, payload, hdrs)
    response = conn.getresponse()
    return response.status, response.getheader("Content-Type"), response.read()


def test_http_serves_the_page_state_thumbs_and_frames(server):
    port, _ = server
    status, kind, body = call(port, "GET", "/")
    assert status == 200 and kind.startswith("text/html") and b"auto-redact review" in body
    status, _, body = call(port, "GET", "/api/state")
    assert status == 200 and json.loads(body)["counts"]["tracks"] == 3
    status, kind, body = call(port, "GET", "/api/thumb/1")
    assert (status, kind) == (200, "image/jpeg") and body[:2] == b"\xff\xd8"
    status, kind, body = call(port, "GET", "/api/frame/5?max_gap=3&smooth_window=1&overlay=1")
    assert (status, kind) == (200, "image/jpeg") and decode(body).shape == (240, 320, 3)


def test_http_frame_ignores_the_pages_cache_busting_parameter(server):
    port, _ = server
    status, kind, _ = call(port, "GET", "/api/frame/5?overlay=1&max_gap=30&smooth_window=9&bridge_iou=0.5&min_track_frames=5&_=1793072906285")
    assert (status, kind) == (200, "image/jpeg")
    assert call(port, "GET", "/api/frame/5?bogus=1")[0] == 200  # not a setting: ignored, not an error


def test_http_edits_and_errors(server):
    port, review = server
    status, _, body = call(port, "POST", "/api/track", {"track_id": 2, "redact": True})
    assert status == 200 and json.loads(body)["effective"] is True
    status, _, body = call(port, "POST", "/api/frame-label", {"frame": 9, "kind": "leak", "box": [1, 1, 50, 50]})
    assert status == 200 and json.loads(body) == {"index": 0}
    status, _, body = call(port, "POST", "/api/analyze", {})
    assert status == 200 and json.loads(body)["labels"]["leaks"] == 1
    status, _, body = call(port, "POST", "/api/preview", {"settings": {"max_gap": 7}})
    assert status == 200 and "trial" in json.loads(body)

    status, _, body = call(port, "POST", "/api/track", {"track_id": 99, "redact": True})
    assert status == 400 and "no track" in json.loads(body)["error"]
    assert call(port, "POST", "/api/preview", {"settings": {"max_gap": -5}})[0] == 400
    assert call(port, "GET", "/api/frame/9999")[0] == 400
    assert call(port, "GET", "/api/nothing")[0] == 404
    assert call(port, "POST", "/api/nothing", {})[0] == 404


def test_http_requires_a_json_body_for_writes(server):
    port, _ = server
    status, _, body = call(port, "POST", "/api/track", None, {"Content-Type": "text/plain"})
    assert status == 400 and "JSON" in json.loads(body)["error"]


def test_http_refuses_foreign_host_headers(server):
    port, _ = server
    status, _, _ = call(port, "GET", "/api/state", headers={"Host": "evil.example"})
    assert status == 403
    assert call(port, "GET", "/api/state", headers={"Host": f"localhost:{port}"})[0] == 200


def test_http_render_round_trip(server):
    port, review = server
    status, _, body = call(port, "POST", "/api/render", {"settings": {"max_gap": 3}})
    assert status == 200 and json.loads(body)["state"] == "running"
    wait_for(review, "done")
    status, _, body = call(port, "GET", "/api/render-status")
    assert json.loads(body)["state"] == "done"


def test_the_page_escapes_untrusted_text():
    page = open("auto_redact/review_page.html", encoding="utf-8").read()
    assert "esc(t.detail)" in page and "t.detail}" not in page.replace("esc(t.detail)", "")


def test_a_busy_port_is_a_clean_error(review):
    from auto_redact.review import serve

    first = make_server(review)
    try:
        with pytest.raises(ReviewError, match="cannot listen on port"):
            serve(review, port=first.server_address[1], open_browser=False)
    finally:
        first.server_close()
