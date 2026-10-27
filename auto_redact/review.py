"""Review server: look at a run's decisions, correct them, and tune from the feedback.

``auto-redact review --video v.mkv`` starts a small local web page (stdlib
HTTP server, one static page, bound to 127.0.0.1) over a video's stored
tracking results. In it you can

* see each tracked person with the classifier's decision, score and reason,
  and overrule it (saved as a human ``override`` on the tracklet and as a
  track label in ``labels.json``; re-running never discards either);
* scrub the video with the redaction boxes drawn on top, and flag frames where
  a person is visible without a box (``leak``) or a box is where it should not
  be (``over``);
* ask for suggestions: :mod:`auto_redact.tune` explains each flagged frame,
  recommends render settings and a SigLIP threshold that fit your labels, and
  lets you preview the effect of any setting on them before rendering.

:class:`Review` holds all the logic and returns plain dicts, so it is tested
without HTTP; the handler below is a thin JSON adapter.
"""

from __future__ import annotations

import dataclasses
import json
import re
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import cv2

from . import cache, tune
from .classify import make_classifier
from .config import ClassifyConfig, RenderConfig
from .frames import iter_frames
from .labels import Labels, LabelsError, load_labels, save_labels
from .meta import load_meta, save_meta_summary
from .render import render_redacted
from .snapshot import index_boxes_by_frame
from .workspace import labels_path_for

PAGE = Path(__file__).with_name("review_page.html")

_KNOB_TYPES = {"max_gap": int, "smooth_window": int, "bridge_iou": float, "min_track_frames": int}
_CLASSIFIER_KEYS = ("classifier", "model", "threshold", "think")

RED, GREEN, YELLOW, MAGENTA = (60, 60, 230), (80, 200, 90), (30, 210, 240), (200, 60, 200)  # BGR


class ReviewError(ValueError):
    """A bad request (unknown track, out-of-range frame, invalid setting)."""


class Review:
    def __init__(
        self,
        meta_path: str | Path,
        render_config: RenderConfig | None = None,
        classify_config: ClassifyConfig | None = None,
        out_path: str | Path | None = None,
    ) -> None:
        self.meta_path = Path(meta_path)
        self.labels_path = labels_path_for(self.meta_path)
        self.render_config = render_config or RenderConfig()
        self.classify_config = classify_config or ClassifyConfig()
        self.out_path = Path(out_path) if out_path else None
        self.lock = threading.RLock()

        self.meta = load_meta(self.meta_path)
        self.video = Path(self.meta.video.path)
        self.track_key = cache.StageCache(self.meta_path).stages.get("track")
        labels = load_labels(self.labels_path)
        self.ignored_labels = 0
        if labels.stale(self.track_key):
            self.ignored_labels = len(labels.tracks) + len(labels.frames)
            labels = Labels()
        labels.track_key = labels.track_key or self.track_key
        self.labels = labels

        self._boxes_cache: dict[tuple, Any] = {}
        self._frame_index = index_boxes_by_frame(self.meta.tracklets)
        self._by_id = {t.track_id: t for t in self.meta.tracklets}
        self._render = {"state": "idle", "done": 0, "total": 0, "message": ""}

    # --- reading ---------------------------------------------------------------

    def state(self) -> dict[str, Any]:
        with self.lock:
            tracks = [self._track_summary(t) for t in self.meta.tracklets]
            return {
                "video": dataclasses.asdict(self.meta.video),
                "tracks": tracks,
                "counts": {
                    "tracks": len(tracks),
                    "redact": sum(t["effective"] is True for t in tracks),
                    "keep": sum(t["effective"] is False for t in tracks),
                    "undecided": sum(t["effective"] is None for t in tracks),
                    "overrides": sum(t["override"] is not None for t in tracks),
                },
                "labels": {
                    "frames": [
                        {"index": i, **dataclasses.asdict(f)}
                        for i, f in enumerate(self.labels.frames)
                    ],
                    "ignored": self.ignored_labels,
                },
                "settings": tune.settings_of(self.render_config),
                "classifier": {
                    "name": self.classify_config.classifier,
                    "model": self.classify_config.resolved_model,
                    "threshold": self.classify_config.threshold,
                },
                "has_scores": any(t.score is not None for t in self.meta.tracklets),
                "scene_cuts": list(self.meta.scene_cuts),
                "out": str(self.out_path) if self.out_path else None,
                "video_exists": self.video.is_file(),
            }

    def _track_summary(self, t) -> dict[str, Any]:
        verdict = self.labels.tracks.get(t.track_id)
        return {
            "id": t.track_id,
            "first": t.frames[0] if t.frames else None,
            "last": t.frames[-1] if t.frames else None,
            "n": len(t.frames),
            "best_frame": t.best_frame,
            "score": t.score,
            "detail": t.detail,
            "redact": t.redact,
            "override": t.override,
            "effective": t.effective_redact,
            "labeled": verdict.redact if verdict else None,
            "has_thumb": bool(t.best_snapshot_path) and Path(t.best_snapshot_path).is_file(),
        }

    def thumb_path(self, track_id: int) -> Path:
        tracklet = self._track(track_id)
        path = Path(tracklet.best_snapshot_path or "")
        if not path.is_file() or path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            raise ReviewError(f"track {track_id} has no thumbnail")
        return path

    # --- decisions and labels -----------------------------------------------------

    def _track(self, track_id: int):
        try:
            return self._by_id[int(track_id)]
        except (KeyError, TypeError, ValueError):
            raise ReviewError(f"no track {track_id!r}") from None

    def set_track(self, track_id: int, redact: bool | None, note: str = "") -> dict[str, Any]:
        """Overrule the classifier for a track (``None`` clears your verdict)."""
        if redact is not None and not isinstance(redact, bool):
            raise ReviewError("redact must be true, false or null")
        with self.lock:
            tracklet = self._track(track_id)
            tracklet.override = redact
            self.labels.set_track(tracklet.track_id, redact, note)
            self._boxes_cache.clear()
            save_meta_summary(self.meta_path, self.meta)
            save_labels(self.labels_path, self.labels)
            return self._track_summary(tracklet)

    def add_frame_label(
        self, frame: int, kind: str, box: list[float] | None = None, note: str = ""
    ) -> int:
        with self.lock:
            if not isinstance(frame, int) or isinstance(frame, bool) or not 0 <= frame < self.meta.video.frames:
                raise ReviewError(f"frame must be an integer in 0..{self.meta.video.frames - 1}")
            try:
                self.labels.add_frame(frame, kind, box, note)
            except LabelsError as error:
                raise ReviewError(str(error)) from None
            save_labels(self.labels_path, self.labels)
            return len(self.labels.frames) - 1

    def delete_frame_label(self, index: int) -> None:
        with self.lock:
            try:
                self.labels.remove_frame(index)
            except LabelsError as error:
                raise ReviewError(str(error)) from None
            save_labels(self.labels_path, self.labels)

    # --- settings and suggestions ----------------------------------------------------

    def config_from(self, params: dict[str, Any] | None) -> RenderConfig:
        """The render config with the four tunable knobs overridden from ``params``."""
        changes: dict[str, Any] = {}
        for key, value in (params or {}).items():
            if key not in _KNOB_TYPES:
                raise ReviewError(f"unknown setting {key!r}")
            try:
                changes[key] = _KNOB_TYPES[key](value)
            except (TypeError, ValueError):
                raise ReviewError(f"{key}: expected a number, got {value!r}") from None
        config = dataclasses.replace(self.render_config, **changes)
        if config.max_gap < 0 or config.min_track_frames < 1:
            raise ReviewError("max_gap must be >= 0 and min_track_frames >= 1")
        if config.smooth_window < 1 or config.smooth_window % 2 == 0:
            raise ReviewError("smooth_window must be an odd number >= 1")
        if not 0 <= config.bridge_iou <= 1:
            raise ReviewError("bridge_iou must be between 0 and 1")
        return config

    def boxes_for(self, config: RenderConfig):
        key = (config.max_gap, config.smooth_window, config.bridge_iou, config.min_track_frames)
        with self.lock:
            if key not in self._boxes_cache:
                self._boxes_cache[key] = tune.redaction_boxes(self.meta, config)
            return self._boxes_cache[key]

    def analyze(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        with self.lock:
            return tune.analyze(
                self.meta, self.labels, self.config_from(params), self.classify_config.threshold
            )

    def preview(self, params: dict[str, Any]) -> dict[str, Any]:
        """How the labels fare under ``params``, next to the current settings."""
        with self.lock:
            trial, _ = tune.evaluate_settings(self.meta, self.labels, self.config_from(params))
            base, _ = tune.evaluate_settings(self.meta, self.labels, self.render_config)
            growth = (trial.area - base.area) / base.area * 100 if base.area else 0.0
            return {"trial": trial.to_dict(), "base": base.to_dict(), "area_change_pct": round(growth, 2)}

    def try_classifier(self, prompt: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Score a prompt/model on just the labeled tracks (slow: it runs the classifier)."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ReviewError("prompt must not be empty")
        options = {k: v for k, v in (options or {}).items() if k in _CLASSIFIER_KEYS and v not in (None, "")}
        with self.lock:
            labeled = [t for t in self.meta.tracklets if t.track_id in self.labels.tracks]
            if not labeled:
                raise ReviewError("label some tracks first")
            try:
                config = dataclasses.replace(self.classify_config, **options)
            except TypeError as error:
                raise ReviewError(str(error)) from None
            report = tune.evaluate_classifier(
                self.meta, self.labels, make_classifier(config), prompt.strip(), config.on_undecided
            )
            current = tune.ClassifierReport()
            for t in labeled:
                want, said = self.labels.tracks[t.track_id].redact, bool(t.redact)
                current.tp += said and want
                current.fp += said and not want
                current.fn += (not said) and want
                current.tn += (not said) and not want
            return {"tried": report.to_dict(), "current": current.to_dict(), "labeled": len(labeled)}

    # --- frames -------------------------------------------------------------------------

    def frame_jpeg(self, frame: int, params: dict[str, Any] | None = None, overlay: bool = True) -> bytes:
        if not 0 <= frame < self.meta.video.frames:
            raise ReviewError(f"frame must be in 0..{self.meta.video.frames - 1}")
        if not self.video.is_file():
            raise ReviewError(f"source video not found: {self.video}")
        image = None
        for _, rgb in iter_frames(self.video, start=frame, stop=frame + 1):
            image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if image is None:
            raise ReviewError(f"could not decode frame {frame}")
        if overlay:
            self._draw(image, frame, self.config_from(params))
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise ReviewError("could not encode frame")
        return encoded.tobytes()

    def _draw(self, image, frame: int, config: RenderConfig) -> None:
        painted = self.boxes_for(config).get(frame, [])
        shade = image.copy()
        for x1, y1, x2, y2 in painted:
            cv2.rectangle(shade, (x1, y1), (x2, y2), RED, thickness=-1)
        cv2.addWeighted(shade, 0.35, image, 0.65, 0, dst=image)
        for x1, y1, x2, y2 in painted:
            cv2.rectangle(image, (x1, y1), (x2, y2), RED, 2)
        with self.lock:
            observed = list(self._frame_index.get(frame, ()))
            labels = [f for f in self.labels.frames if f.frame == frame]
        for track_id, box in observed:
            x1, y1, x2, y2 = (int(v) for v in box)
            cv2.rectangle(image, (x1, y1), (x2, y2), GREEN, 1)
            cv2.putText(image, f"#{track_id}", (x1 + 3, y1 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1, cv2.LINE_AA)
        for label in labels:
            if label.box is not None:
                x1, y1, x2, y2 = (int(v) for v in label.box)
                color = YELLOW if label.kind == "leak" else MAGENTA
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, label.kind, (x1 + 3, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    # --- rendering --------------------------------------------------------------------------

    def start_render(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if self.out_path is None:
            raise ReviewError("start the review with --out to enable rendering")
        config = self.config_from(params)
        with self.lock:
            if self._render["state"] == "running":
                raise ReviewError("a render is already running")
            self._render = {"state": "running", "done": 0, "total": 0, "message": "", "out": str(self.out_path)}

        def progress(done: int, total: int) -> None:
            self._render.update(done=done, total=total)

        def work() -> None:
            try:
                render_redacted(self.meta, self.out_path, config, on_progress=progress)
                self._render.update(state="done", message=f"wrote {self.out_path}")
            except Exception as error:  # surfaced to the page, not swallowed
                self._render.update(state="error", message=f"{type(error).__name__}: {error}")

        threading.Thread(target=work, name="review-render", daemon=True).start()
        return dict(self._render)

    def render_status(self) -> dict[str, Any]:
        return dict(self._render)


# --- HTTP ---------------------------------------------------------------------------------


def make_server(review: Review, host: str = "127.0.0.1", port: int = 0) -> ThreadingHTTPServer:
    """An HTTP server for ``review``; ``port=0`` picks a free port."""

    class Handler(BaseHTTPRequestHandler):
        server_version = "auto-redact-review"

        def log_message(self, *args: Any) -> None:  # keep the terminal quiet
            pass

        # -- plumbing
        def _allowed_host(self) -> bool:
            name = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]")
            return name in ("127.0.0.1", "localhost", "::1")  # blocks DNS-rebinding

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload: Any, status: int = 200) -> None:
            self._send(status, json.dumps(payload).encode("utf-8"), "application/json")

        def _body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                raise ReviewError("expected a JSON body")
            try:
                data = json.loads(self.rfile.read(length) or b"{}")
            except ValueError:
                raise ReviewError("invalid JSON") from None
            if not isinstance(data, dict):
                raise ReviewError("expected a JSON object")
            return data

        def _dispatch(self, method: str) -> None:
            if not self._allowed_host():
                return self._json({"error": "forbidden host"}, 403)
            url = urlparse(self.path)
            query = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                self._route(method, url.path, query)
            except ReviewError as error:
                self._json({"error": str(error)}, 400)
            except Exception as error:  # a bug: tell the page, keep serving
                self._json({"error": f"{type(error).__name__}: {error}"}, 500)

        def do_GET(self) -> None:
            self._dispatch("GET")

        def do_POST(self) -> None:
            self._dispatch("POST")

        # -- routes
        def _route(self, method: str, path: str, query: dict[str, str]) -> None:
            if method == "GET":
                if path == "/":
                    return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
                if path == "/api/state":
                    return self._json(review.state())
                if path == "/api/render-status":
                    return self._json(review.render_status())
                match = re.fullmatch(r"/api/thumb/(\d+)", path)
                if match:
                    thumb = review.thumb_path(int(match.group(1)))
                    kind = "image/png" if thumb.suffix.lower() == ".png" else "image/jpeg"
                    return self._send(200, thumb.read_bytes(), kind)
                match = re.fullmatch(r"/api/frame/(\d+)", path)
                if match:
                    overlay = query.get("overlay", "1") != "0"
                    knobs = {k: v for k, v in query.items() if k in _KNOB_TYPES}  # ignore cache-busters
                    jpeg = review.frame_jpeg(int(match.group(1)), knobs, overlay)
                    return self._send(200, jpeg, "image/jpeg")
                return self._json({"error": "not found"}, 404)

            data = self._body()
            if path == "/api/track":
                return self._json(review.set_track(data.get("track_id"), data.get("redact"), data.get("note", "")))
            if path == "/api/frame-label":
                index = review.add_frame_label(data.get("frame"), data.get("kind"), data.get("box"), data.get("note", ""))
                return self._json({"index": index})
            if path == "/api/frame-label/delete":
                review.delete_frame_label(data.get("index"))
                return self._json({"ok": True})
            if path == "/api/analyze":
                return self._json(review.analyze(data.get("settings")))
            if path == "/api/preview":
                return self._json(review.preview(data.get("settings") or {}))
            if path == "/api/try-classifier":
                return self._json(review.try_classifier(data.get("prompt"), data.get("options")))
            if path == "/api/render":
                return self._json(review.start_render(data.get("settings")))
            return self._json({"error": "not found"}, 404)

    return ThreadingHTTPServer((host, port), Handler)


def serve(review: Review, port: int = 0, open_browser: bool = True) -> None:
    try:
        server = make_server(review, port=port)
    except OSError as error:
        raise ReviewError(f"cannot listen on port {port}: {error.strerror or error}") from None
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"reviewing {review.video.name}: {url}   (Ctrl-C to stop)", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
