"""Streaming single-pass pipeline.

Same four stages as :func:`auto_redact.pipeline.run_pipeline` (track, snapshot,
classify, render), but overlapped instead of run back to back:

* a **decoder thread** reads the source once and flags scene cuts;
* the **main thread** runs YOLO/ByteTrack, assembles tracklets, and keeps the
  best-view crop of each person as it goes (so no frame is ever re-decoded or
  buffered for snapshotting);
* a **classifier pool** turns a closed tracklet into a thumbnail and a decision
  while tracking continues;
* a **shipper thread** renders output as soon as it is final: GOPs (selective
  mode) are copied or re-encoded independently and concatenated at the end, or
  frames are streamed into one encoder when selective copy is unavailable.

Output is identical to the staged pipeline because the boxes for any frame are
produced by the very same :func:`auto_redact.render.redaction_boxes`, applied
only once no undecided tracklet can still change that frame (see
:class:`_Frontier`). A person present for the whole video therefore blocks
output until they are classified, the same as the staged worst case.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import queue
import shutil
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from . import telemetry
from .classify import (
    DEFAULT_MAX_IMAGE_SIZE,
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_URL,
    DEFAULT_TIMEOUT,
    DEFAULT_WORKERS,
    Decision,
    VlmClassifier,
)
from .frames import iter_frames, probe_video
from .meta import Metadata, MetaError, Tracklet, VideoInfo, load_meta, save_meta
from .pipeline import _classify_key, run_pipeline
from .render import (
    DEFAULT_BRIDGE_IOU,
    DEFAULT_CRF,
    DEFAULT_ENCODER,
    DEFAULT_MAX_GAP,
    DEFAULT_MIN_TRACK_FRAMES,
    DEFAULT_PRESET,
    DEFAULT_SMOOTH_WINDOW,
    ENCODERS,
    RenderError,
    _check_crf,
    _concat_video,
    _copy_segment,
    _encode,
    _encode_segment,
    _keyframe_table,
    _mux_audio,
    nvenc_fallback_preset,
    _Segment,
    redaction_boxes,
    render_redacted,
)
from .scene import _frame_hist
from .siglip import DEFAULT_SIGLIP_MODEL, DEFAULT_THRESHOLD, SiglipClassifier
from .snapshot import (
    BODY_WEIGHT,
    CLOSEUP_AREA_FRACTION,
    CONTACT_SHEET_CELL,
    DEFAULT_CONTACT_SHEET,
    DEFAULT_PADDING,
    DISTRACTOR_WEIGHT,
    EDGE_PENALTY,
    EDGE_TOLERANCE,
    HEAD_WEIGHT,
    ISOLATE_DILATE,
    JPEG_QUALITY,
    SAME_PERSON_IOU,
    _clamped_area,
    _load_yolo,
    _padded_crop_window,
    crop_snapshot,
    observation_score,
    redact_others,
    sub_isolate_people,
    write_contact_sheet,
)
from .track import _extract, _same_video, _select_device

logger = logging.getLogger(__name__)

_LAZY_CHUNK = 128  # frames per box computation in full re-encode mode


@dataclass
class _Track:
    """A tracklet under construction plus its snapshot scratch state."""

    uid: int
    tracker_id: int
    tl: Tracklet
    closed: bool = False
    best_score: float = -1.0
    best_box: list[float] | None = None
    best_crop: np.ndarray | None = None
    best_others: list[tuple[int, list[float]]] = field(default_factory=list)
    thumb: Image.Image | None = None

    @property
    def first(self) -> int:
        return self.tl.frames[0]

    @property
    def last(self) -> int:
        return self.tl.frames[-1]


class _Frontier:
    """Shared tracking/decision state that decides which frames are final.

    Frame ``f`` is final once nothing that is still unknown can add a box to
    it. A tracklet influences frames from ``first - margin`` (seam bridging
    reaches back ``max_gap`` frames) to ``last + margin``; while it is open
    its end is unknown, and while it is closed but undecided its verdict is.
    Tracklets not yet seen start at or after ``consumed``.
    """

    def __init__(self, info: VideoInfo, max_gap: int, smooth_window: int,
                 bridge_iou: float, min_track_frames: int) -> None:
        self.cv = threading.Condition()
        self.info = info
        self.margin = max_gap + 1
        self.box_args = dict(
            max_gap=max_gap, smooth_window=smooth_window,
            bridge_iou=bridge_iou, min_track_frames=min_track_frames,
        )
        self.consumed = 0
        self.done = False
        self.fatal: BaseException | None = None
        self.undecided: dict[int, _Track] = {}
        self.redacted: list[Tracklet] = []

    def advance(self, consumed: int) -> None:
        with self.cv:
            self.consumed = consumed
            self.cv.notify_all()

    def finish_tracking(self) -> None:
        with self.cv:
            self.done = True
            self.cv.notify_all()

    def fail(self, error: BaseException) -> None:
        with self.cv:
            if self.fatal is None:
                self.fatal = error
            self.cv.notify_all()

    def add(self, track: _Track) -> None:
        with self.cv:
            self.undecided[track.uid] = track

    def decide(self, track: _Track) -> None:
        with self.cv:
            self.undecided.pop(track.uid, None)
            if track.tl.redact is True:
                self.redacted.append(track.tl)
            self.cv.notify_all()

    def _final_locked(self, lo: int, hi: int) -> bool:
        if not self.done and hi > self.consumed - self.margin:
            return False
        for t in self.undecided.values():
            end = t.last + self.margin if t.closed else float("inf")
            if t.first - self.margin < hi and end >= lo:
                return False
        return True

    def wait_final(self, lo: int, hi: int) -> None:
        with self.cv:
            while not self._final_locked(lo, hi):
                if self.fatal is not None:
                    raise self.fatal
                self.cv.wait(0.5)
            if self.fatal is not None:
                raise self.fatal

    def ready_ranges(self, ranges: list[tuple[int, int]], taken: set[int]) -> list[int]:
        """Block until some untaken range is final; return their indexes."""
        with self.cv:
            while True:
                if self.fatal is not None:
                    raise self.fatal
                ready = [
                    i for i, (lo, hi) in enumerate(ranges)
                    if i not in taken and self._final_locked(lo, hi)
                ]
                if ready or len(taken) == len(ranges):
                    return ready
                self.cv.wait(0.5)

    def boxes_for(self, lo: int, hi: int) -> dict[int, list]:
        """Redaction boxes for frames ``[lo, hi)``; the range must be final."""
        with self.cv:
            subset = [
                tl for tl in self.redacted
                if tl.frames[-1] >= lo - self.margin and tl.frames[0] < hi + self.margin
            ]
        meta = Metadata(video=self.info, tracklets=subset)
        return redaction_boxes(meta, **self.box_args)


class _LazyBoxes:
    """``boxes.get(frame)`` that blocks until the frame is final.

    Lets the staged ``_encode`` loop run unchanged while it is fed by an
    ongoing analysis: it decodes frame N, asks for its boxes, and waits here
    until they cannot change any more.
    """

    def __init__(self, frontier: _Frontier, frame_count: int) -> None:
        self.frontier = frontier
        self.frame_count = frame_count
        self.chunk = -1
        self.boxes: dict[int, list] = {}
        self.shipped = 0

    def get(self, idx: int, default=()):
        chunk = idx // _LAZY_CHUNK
        if chunk != self.chunk:
            lo = chunk * _LAZY_CHUNK
            hi = lo + _LAZY_CHUNK
            with telemetry.timed("ship.wait_final"):
                self.frontier.wait_final(lo, hi)
            self.boxes = self.frontier.boxes_for(lo, hi)
            self.chunk = chunk
        self.shipped = idx + 1
        return self.boxes.get(idx, default)


def _track_buffer(tracker: str) -> int:
    try:
        import yaml
        from ultralytics.utils.checks import check_yaml

        with open(check_yaml(tracker), encoding="utf-8") as handle:
            return int(yaml.safe_load(handle)["track_buffer"])
    except Exception:
        return 30


def _decode_thread(video: Path, out: queue.Queue, abort: threading.Event,
                   scene_cuts: bool, threshold: float) -> None:
    def put(item) -> bool:
        while not abort.is_set():
            try:
                out.put(item, timeout=0.5)
                return True
            except queue.Full:
                pass
        return False

    try:
        prev = None
        for idx, frame in iter_frames(video, sample_rate=1, threads=os.cpu_count()):
            cut = False
            if scene_cuts:
                hist = _frame_hist(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                if prev is not None:
                    dist = float(cv2.compareHist(prev, hist, cv2.HISTCMP_BHATTACHARYYA))
                    cut = dist >= threshold
                prev = hist
            if not put((idx, frame, cut)):
                return
        put(None)
    except BaseException as error:  # surfaced on the consumer side
        put(error)


class _Decider:
    """Turns closed tracklets into snapshots and verdicts on a worker pool."""

    def __init__(self, *, frontier, info, thumbs, prompt, padding, isolate,
                 sub_isolate, closeup_area_fraction, same_person_iou,
                 isolate_dilate, classifier, prefilter_clf, prefilter,
                 workers) -> None:
        self.frontier = frontier
        self.w, self.h = info.width, info.height
        self.thumbs = thumbs
        self.prompt = prompt
        self.padding = padding
        self.isolate = isolate
        self.closeup = closeup_area_fraction
        self.same_person_iou = same_person_iou
        self.dilate = isolate_dilate
        self.clf = classifier
        self.pre_clf = prefilter_clf
        self.prefilter = prefilter
        self.sub_isolate = sub_isolate
        self.sub_model = None
        self.sub_lock = threading.Lock()
        if sub_isolate:
            try:
                self.sub_model = _load_yolo("cpu")
            except Exception:
                self.sub_model = None
        self.pool = ThreadPoolExecutor(max_workers=max(1, int(workers)), thread_name_prefix="decide")
        self.submitted = 0
        self.completed = 0
        self.count_lock = threading.Lock()

    def submit(self, track: _Track) -> None:
        with self.count_lock:
            self.submitted += 1
        self.pool.submit(self._run, track)

    def _run(self, track: _Track) -> None:
        try:
            with telemetry.timed("decide.total"):
                decision = self._decide(track)
        except BaseException as error:
            logger.warning("track %s: decision failed (%s: %s)", track.uid, type(error).__name__, error)
            logger.debug("decision traceback", exc_info=True)
            decision = Decision(None, None, f"error: {error}")
        tl = track.tl
        tl.redact, tl.score = decision.redact, decision.score
        if decision.redact is None:
            logger.warning("track %s: unparseable or undecided, redact left unset (%s)", track.uid, decision.detail)
        else:
            logger.info("track %s: redact=%s (%s)", track.uid, decision.redact, decision.detail)
        track.best_crop = None
        with self.count_lock:
            self.completed += 1
        self.frontier.decide(track)

    def _snapshot(self, track: _Track) -> str | None:
        if track.best_box is None or track.best_crop is None:
            return None
        box, crop, tl = track.best_box, track.best_crop, track.tl
        window = _padded_crop_window(box, self.w, self.h, self.padding)
        fraction = _clamped_area(box, self.w, self.h) / max(1.0, float(self.w) * float(self.h))
        closeup = fraction >= self.closeup
        if self.isolate and not closeup:
            crop, painted = redact_others(
                crop, window, box, track.best_others,
                same_person_iou=self.same_person_iou, dilate=self.dilate,
            )
            tl.isolated_track_ids = painted
        if self.sub_isolate and self.isolate and self.sub_model is not None and not closeup:
            with self.sub_lock:
                sub_isolate_people(crop, window, box, self.sub_model, dilate=self.dilate)
        image = Image.fromarray(crop)
        path = self.thumbs / f"track_{track.uid}.jpg"
        image.save(path, format="JPEG", quality=JPEG_QUALITY)
        image.thumbnail(CONTACT_SHEET_CELL, Image.Resampling.LANCZOS)
        track.thumb = image
        tl.best_snapshot_path = str(path)
        return str(path)

    def _decide(self, track: _Track) -> Decision:
        path = self._snapshot(track)
        if path is None:
            return Decision(None, None, "no usable snapshot")
        if self.pre_clf is not None:
            try:
                early = self.pre_clf.classify(path, self.prompt)
            except Exception as error:
                logger.warning("track %s: siglip prefilter error: %s", track.uid, error)
                return Decision(None, None, "siglip error")
            needs_vlm = early.redact is (False if self.prefilter == "negatives" else True)
            if not needs_vlm:
                return early
        return self.clf.classify(path, self.prompt)

    def drain(self) -> None:
        self.pool.shutdown(wait=True)


class _SelectiveShipper(threading.Thread):
    """Copies or re-encodes each GOP as soon as it is final."""

    def __init__(self, frontier, video, info, keyframes, work_dir, crf, preset, encoder):
        super().__init__(name="ship", daemon=True)
        self.frontier, self.video, self.info = frontier, video, info
        self.keyframes, self.work_dir = keyframes, work_dir
        self.crf, self.preset, self.encoder = crf, preset, encoder
        bounds = sorted({0, info.frame_count, *keyframes})
        self.gops = [
            (a, b) for a, b in zip(bounds, bounds[1:]) if b > a
        ]
        self.parts: list[tuple[int, Path]] = []
        self.shipped_frames = 0

    def run(self) -> None:
        try:
            taken: set[int] = set()
            while len(taken) < len(self.gops):
                ready = self.frontier.ready_ranges(self.gops, taken)
                for run in _consecutive(ready):
                    lo, hi = self.gops[run[0]][0], self.gops[run[-1]][1]
                    boxes = self.frontier.boxes_for(lo, hi)
                    for edited, group in _split_by_edit(run, self.gops, boxes):
                        self._ship(edited, self.gops[group[0]][0], self.gops[group[-1]][1], boxes)
                    taken.update(run)
        except BaseException as error:
            self.frontier.fail(error)

    def _ship(self, edited: bool, lo: int, hi: int, boxes) -> None:
        part = self.work_dir / f"part_{lo:08d}.mp4"
        segment = _Segment(lo, hi, edited)
        with telemetry.timed("ship.encode" if edited else "ship.copy"):
            if edited:
                try:
                    _encode_segment(self.video, part, boxes, self.keyframes, segment,
                                    self.info, self.crf, self.preset, self.encoder)
                except RenderError as error:
                    if self.encoder != "h264_nvenc":
                        raise
                    self.preset = nvenc_fallback_preset(self.preset, error)
                    self.encoder = "libx264"
                    _encode_segment(self.video, part, boxes, self.keyframes, segment,
                                    self.info, self.crf, self.preset, self.encoder)
            else:
                _copy_segment(self.video, part, self.keyframes, segment)
        self.parts.append((lo, part))
        self.shipped_frames += hi - lo


def _consecutive(indexes: list[int]) -> list[list[int]]:
    runs: list[list[int]] = []
    for i in indexes:
        if runs and runs[-1][-1] == i - 1:
            runs[-1].append(i)
        else:
            runs.append([i])
    return runs


def _split_by_edit(run, gops, boxes):
    groups: list[tuple[bool, list[int]]] = []
    for i in run:
        lo, hi = gops[i]
        edited = any(f in boxes for f in range(lo, hi))
        if groups and groups[-1][0] == edited:
            groups[-1][1].append(i)
        else:
            groups.append((edited, [i]))
    return groups


class _FullShipper(threading.Thread):
    """Streams every frame through one encoder, painting as boxes finalize."""

    def __init__(self, frontier, video, out, info, crf, preset, encoder):
        super().__init__(name="ship", daemon=True)
        self.args = (video, out, info, crf, preset, encoder)
        self.boxes = _LazyBoxes(frontier, info.frame_count)
        self.frontier = frontier

    @property
    def shipped_frames(self) -> int:
        return self.boxes.shipped

    def run(self) -> None:
        video, out, info, crf, preset, encoder = self.args
        try:
            try:
                _encode(video, out, self.boxes, info, crf, preset, encoder)
            except RenderError as error:
                if encoder != "h264_nvenc":
                    raise
                _encode(video, out, self.boxes, info, crf,
                        nvenc_fallback_preset(preset, error), "libx264")
        except BaseException as error:
            self.frontier.fail(error)


def run_pipeline_streaming(
    video,
    prompt,
    out,
    *,
    meta_path=None,
    thumbs_dir=None,
    force=False,
    sample_rate=1,
    device="auto",
    half=True,
    track_conf=0.25,
    track_iou=0.7,
    track_imgsz=None,
    track_tracker="bytetrack.yaml",
    orphans=False,
    resume=True,
    padding=DEFAULT_PADDING,
    isolate=True,
    sub_isolate=False,
    closeup_area_fraction=CLOSEUP_AREA_FRACTION,
    head_weight=HEAD_WEIGHT,
    body_weight=BODY_WEIGHT,
    edge_penalty=EDGE_PENALTY,
    edge_tolerance=EDGE_TOLERANCE,
    distractor_weight=DISTRACTOR_WEIGHT,
    same_person_iou=SAME_PERSON_IOU,
    isolate_dilate=ISOLATE_DILATE,
    classifier="vlm",
    model=None,
    threshold=DEFAULT_THRESHOLD,
    classify_device="auto",
    ollama_url=DEFAULT_OLLAMA_URL,
    timeout=DEFAULT_TIMEOUT,
    classify_workers=DEFAULT_WORKERS,
    think=False,
    max_image_size=DEFAULT_MAX_IMAGE_SIZE,
    prefilter="none",
    prefilter_model=None,
    prefilter_threshold=None,
    prefilter_device="auto",
    keep_alive="3h",
    crf=DEFAULT_CRF,
    preset=DEFAULT_PRESET,
    selective=True,
    encoder=DEFAULT_ENCODER,
    max_gap=DEFAULT_MAX_GAP,
    smooth_window=DEFAULT_SMOOTH_WINDOW,
    bridge_iou=DEFAULT_BRIDGE_IOU,
    min_track_frames=DEFAULT_MIN_TRACK_FRAMES,
    scene_cuts=True,
    scene_cut_threshold=0.45,
    progress=lambda message: None,
    bar=lambda desc, **_: contextlib.nullcontext(),
) -> dict:
    if classifier not in ("vlm", "siglip"):
        raise ValueError(f"unknown classifier {classifier!r}; use 'vlm' or 'siglip'")
    if model is None:
        model = DEFAULT_SIGLIP_MODEL if classifier == "siglip" else DEFAULT_MODEL
    _check_crf(crf)
    if encoder not in ENCODERS:
        raise RenderError(f"encoder must be one of {ENCODERS}, got {encoder!r}")
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate < 1:
        raise ValueError(f"sample_rate must be an integer >= 1, got {sample_rate!r}")

    video, out = Path(video), Path(out)
    if not video.is_file():
        raise FileNotFoundError(f"no such video file: {video}")
    meta_file = Path(meta_path) if meta_path else out.with_name(out.stem + ".meta.json")
    thumbs = Path(thumbs_dir) if thumbs_dir else meta_file.with_name(meta_file.stem + "_thumbs")
    sidecar = meta_file.with_name(meta_file.name + ".classify.json")
    info = probe_video(video)

    # With tracking and snapshots already cached there is nothing left to
    # overlap; the staged pipeline reuses them and only redoes what changed.
    if not force and meta_file.is_file():
        try:
            cached = load_meta(meta_file)
        except MetaError:
            cached = None
        if (
            cached is not None
            and _same_video(cached, video, info)
            and cached.tracklets
            and all(t.best_snapshot_path and Path(t.best_snapshot_path).is_file() for t in cached.tracklets)
        ):
            progress("[stream] tracking and snapshots cached; using the staged pipeline")
            summary = run_pipeline(
                video, prompt, out, meta_path=meta_file, thumbs_dir=thumbs, force=False,
                sample_rate=sample_rate, device=device, half=half, resume=resume,
                classifier=classifier, model=model, threshold=threshold,
                classify_device=classify_device, ollama_url=ollama_url, timeout=timeout,
                classify_workers=classify_workers, think=think, max_image_size=max_image_size,
                prefilter=prefilter, prefilter_model=prefilter_model,
                prefilter_threshold=prefilter_threshold, prefilter_device=prefilter_device,
                crf=crf, preset=preset, selective=selective, encoder=encoder,
                max_gap=max_gap, smooth_window=smooth_window, bridge_iou=bridge_iou,
                min_track_frames=min_track_frames, scene_cuts=scene_cuts,
                scene_cut_threshold=scene_cut_threshold, progress=progress, bar=bar,
            )
            return {"out": summary.out, "meta_path": summary.meta_path,
                    "tracklets": summary.tracklets, "redacted": summary.redacted}

    device = _select_device(device)
    half = half and device.startswith("cuda")
    w, h = info.width, info.height
    thumbs.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Classifiers.
    if classifier == "siglip":
        progress("      initializing classifier model")
        clf = SiglipClassifier(model, threshold, classify_device)
    else:
        progress("      waiting for ollama to load the model")
        clf = VlmClassifier(model, ollama_url, timeout, think, keep_alive, max_image_size)
    pre_clf, prefilter = None, (prefilter or "none").lower()
    if prefilter in ("negatives", "positives") and classifier == "vlm":
        pre_clf = SiglipClassifier(
            prefilter_model or DEFAULT_SIGLIP_MODEL,
            prefilter_threshold if prefilter_threshold is not None else DEFAULT_THRESHOLD,
            prefilter_device,
        )

    # Rendering mode.
    work_dir = Path(tempfile.mkdtemp(prefix="auto-redact-stream-"))
    keyframes: dict[int, int] = {}
    if selective and not orphans:
        keyframes, open_gop = _keyframe_table(video)
        if open_gop or 0 not in keyframes:
            keyframes = {}
    frontier = _Frontier(
        VideoInfo(path=str(video), fps=info.fps, width=w, height=h, frames=info.frame_count),
        max_gap, smooth_window, bridge_iou, min_track_frames,
    )
    shipper = None
    if not orphans:
        shipper = (
            _SelectiveShipper(frontier, video, info, keyframes, work_dir, crf, preset, encoder)
            if keyframes else _FullShipper(frontier, video, out, info, crf, preset, encoder)
        )
    telemetry.note("stream.render_mode", "none (orphans)" if shipper is None else
                   "selective" if keyframes else "full")

    decider = _Decider(
        frontier=frontier, info=info, thumbs=thumbs, prompt=prompt.strip(), padding=padding,
        isolate=isolate, sub_isolate=sub_isolate, closeup_area_fraction=closeup_area_fraction,
        same_person_iou=same_person_iou, isolate_dilate=isolate_dilate, classifier=clf,
        prefilter_clf=pre_clf, prefilter=prefilter, workers=classify_workers,
    )

    from ultralytics import YOLO

    yolo = YOLO("yolo11n-pose.pt")
    yolo.to(device)
    track_kwargs = dict(conf=track_conf, iou=track_iou, persist=True, verbose=False, tracker=track_tracker)
    if half:
        track_kwargs["quantize"] = 16
    if track_imgsz is not None:
        track_kwargs["imgsz"] = track_imgsz
    # ByteTrack drops a lost id after ``track_buffer`` tracker steps and never
    # reuses it, so a tracklet silent for longer than that is complete.
    close_gap = (_track_buffer(track_tracker) + 2) * sample_rate

    tracks: list[_Track] = []
    open_tracks: dict[int, _Track] = {}
    cuts: list[int] = []
    next_uid = 1

    def close(track: _Track) -> None:
        track.closed = True
        decider.submit(track)

    abort = threading.Event()
    frames_q: queue.Queue = queue.Queue(maxsize=6)
    decoder = threading.Thread(
        target=_decode_thread, name="decode", daemon=True,
        args=(video, frames_q, abort, scene_cuts, scene_cut_threshold),
    )

    score_kwargs = dict(
        padding=padding, head_weight=head_weight, body_weight=body_weight,
        edge_penalty=edge_penalty, edge_tolerance=edge_tolerance,
        distractor_weight=distractor_weight,
    )
    progress(f"[stream] tracking, classifying and rendering {video}")
    bars = contextlib.ExitStack()
    cbs: dict[str, Any] = {}
    for position, desc in enumerate(("track", "classify", "ship")):
        try:
            ctx = bar(desc, position=position)
        except TypeError:
            ctx = bar(desc)
        cbs[desc] = bars.enter_context(ctx)

    def report(frame_idx: int) -> None:
        if cbs["track"]:
            cbs["track"](frame_idx + 1, max(info.frame_count, frame_idx + 1))
        if cbs["classify"]:
            cbs["classify"](decider.completed, max(1, decider.submitted))
        if cbs["ship"] and shipper is not None:
            cbs["ship"](shipper.shipped_frames, max(1, info.frame_count))

    try:
        with bars, telemetry.stage("stream"):
            if shipper is not None:
                shipper.start()
            decoder.start()
            last_idx = -1
            while True:
                with telemetry.timed("stream.decode_wait"):
                    item = frames_q.get()
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise item
                frame_idx, frame, cut = item
                if frontier.fatal is not None:
                    raise frontier.fatal
                last_idx = frame_idx
                if cut:
                    cuts.append(frame_idx)
                    for track in list(open_tracks.values()):
                        close(track)
                    open_tracks.clear()

                if frame_idx % sample_rate == 0:
                    obs: dict[int, list] = {}
                    with telemetry.timed("stream.yolo"):
                        for result in yolo.track(frame, **track_kwargs):
                            _extract(result, frame_idx, obs)
                    frame_obs: list[tuple[_Track, list[float], list]] = []
                    for tid, observations in obs.items():
                        for _fi, box, kps in observations:
                            track = open_tracks.get(tid)
                            fresh = track is None
                            if fresh:
                                tl = Tracklet(track_id=next_uid, frames=[], boxes=[], keypoints=[])
                                track = _Track(uid=next_uid, tracker_id=tid, tl=tl)
                                next_uid += 1
                                open_tracks[tid] = track
                                tracks.append(track)
                            track.tl.frames.append(frame_idx)
                            track.tl.boxes.append(box)
                            track.tl.keypoints.append(kps)
                            if fresh:
                                frontier.add(track)
                            frame_obs.append((track, box, kps))
                    with telemetry.timed("stream.score"):
                        for track, box, kps in frame_obs:
                            if _clamped_area(box, w, h) <= 0:
                                continue
                            others = [(o.uid, ob) for o, ob, _ in frame_obs if o is not track]
                            score = observation_score(
                                box, kps, w, h, [ob for _, ob in others], **score_kwargs
                            )
                            if score > track.best_score:  # strict: earliest wins ties
                                track.best_score = score
                                track.tl.best_frame = frame_idx
                                track.best_box = box
                                track.best_others = [(uid, list(ob)) for uid, ob in others]
                                track.best_crop = crop_snapshot(frame, box, padding).copy()

                for tid, track in list(open_tracks.items()):
                    if frame_idx - track.last > close_gap:
                        del open_tracks[tid]
                        close(track)
                frontier.advance(frame_idx + 1)
                if frame_idx % 8 == 0:
                    report(frame_idx)

            for track in open_tracks.values():
                close(track)
            open_tracks.clear()
            frontier.finish_tracking()
            report(max(last_idx, 0))
            decoder.join()

            if shipper is not None:
                while shipper.is_alive():
                    shipper.join(0.5)
                    report(max(last_idx, 0))
            if frontier.fatal is not None:
                raise frontier.fatal
            decider.drain()
            report(max(last_idx, 0))
    except BaseException as error:
        abort.set()
        frontier.fail(error)
        decider.pool.shutdown(wait=False, cancel_futures=True)
        shutil.rmtree(work_dir, ignore_errors=True)
        raise

    # Final metadata, numbered exactly as the staged pipeline would.
    meta = _build_meta(video, info, tracks, cuts if scene_cuts else [], thumbs, scene_cuts)

    if orphans:
        meta = _finish_with_orphans(
            meta, video, out, meta_file, thumbs, prompt, clf,
            dict(
                padding=padding, isolate=isolate, sub_isolate=sub_isolate,
                closeup_area_fraction=closeup_area_fraction, head_weight=head_weight,
                body_weight=body_weight, edge_penalty=edge_penalty,
                edge_tolerance=edge_tolerance, distractor_weight=distractor_weight,
                same_person_iou=same_person_iou, isolate_dilate=isolate_dilate,
                classify_workers=classify_workers, crf=crf, preset=preset,
                selective=selective, encoder=encoder, max_gap=max_gap,
                smooth_window=smooth_window, bridge_iou=bridge_iou,
                min_track_frames=min_track_frames,
            ),
        )
    elif keyframes:
        parts = [p for _, p in sorted(shipper.parts)]
        joined = work_dir / "joined.mp4"
        with telemetry.timed("ship.finalize"):
            _concat_video(parts, joined)
            _mux_audio(joined, video, out)
    shutil.rmtree(work_dir, ignore_errors=True)

    save_meta(meta_file, meta)
    sidecar.write_text(json.dumps(_classify_key(prompt, classifier, model, threshold, think)) + "\n", encoding="utf-8")
    sheet = [(t.track_id, t_thumb) for t, t_thumb in _thumb_pairs(meta, tracks)]
    if sheet:
        write_contact_sheet(sorted(sheet, key=lambda p: p[0]), thumbs / DEFAULT_CONTACT_SHEET)

    redacted = sum(1 for t in meta.tracklets if t.redact)
    progress(f"{len(meta.tracklets)} tracklet(s), {redacted} redacted; wrote {out}")
    return {"out": out, "meta_path": meta_file, "tracklets": len(meta.tracklets), "redacted": redacted}


def _build_meta(video, info, tracks, cuts, thumbs, scene_cuts) -> Metadata:
    """Renumber like ``build_tracklets`` + ``split_tracklets_at_cuts`` do."""
    ordered = sorted(tracks, key=lambda t: (t.tracker_id, t.first))
    next_id = max((t.tracker_id for t in ordered), default=0) + 1
    splits, seen, renames = 0, set(), {}
    for t in ordered:
        if t.tracker_id in seen:
            new = next_id
            next_id += 1
            splits += 1
        else:
            seen.add(t.tracker_id)
            new = t.tracker_id
        renames[t.uid] = new
    # Two-phase rename so ids that swap places never overwrite each other.
    moved = []
    for t in ordered:
        src = thumbs / f"track_{t.uid}.jpg"
        if src.is_file() and renames[t.uid] != t.uid:
            tmp = src.with_suffix(".tmp")
            src.rename(tmp)
            moved.append((tmp, thumbs / f"track_{renames[t.uid]}.jpg"))
    for tmp, dst in moved:
        tmp.replace(dst)
    for t in ordered:
        tl = t.tl
        tl.track_id = renames[t.uid]
        tl.isolated_track_ids = sorted(renames.get(i, i) for i in tl.isolated_track_ids)
        if tl.best_snapshot_path:
            tl.best_snapshot_path = str(thumbs / f"track_{tl.track_id}.jpg")
    meta = Metadata(
        video=VideoInfo(path=str(video), fps=info.fps, width=info.width, height=info.height, frames=info.frame_count),
        tracklets=[t.tl for t in ordered],
    )
    meta.scene_cuts = list(cuts) if scene_cuts else []
    meta.scene_cut_splits = splits if scene_cuts else 0
    return meta


def _thumb_pairs(meta, tracks):
    by_final = {t.tl.track_id: t for t in tracks}
    for tl in meta.tracklets:
        t = by_final.get(tl.track_id)
        if t is not None and t.thumb is not None:
            yield tl, t.thumb


def _finish_with_orphans(meta, video, out, meta_file, thumbs, prompt, clf, opts) -> Metadata:
    """Orphan recovery needs a full extra pass, so rendering stays staged."""
    from .classify import classify_snapshots
    from .snapshot import select_snapshots
    from .track import add_orphan_tracklets

    meta = add_orphan_tracklets(meta, video, on_progress=None)
    select_snapshots(
        meta, thumbs_dir=thumbs, padding=opts["padding"], isolate=opts["isolate"],
        sub_isolate=opts["sub_isolate"], closeup_area_fraction=opts["closeup_area_fraction"],
        head_weight=opts["head_weight"], body_weight=opts["body_weight"],
        edge_penalty=opts["edge_penalty"], edge_tolerance=opts["edge_tolerance"],
        distractor_weight=opts["distractor_weight"], same_person_iou=opts["same_person_iou"],
        isolate_dilate=opts["isolate_dilate"], contact_sheet=DEFAULT_CONTACT_SHEET,
        resume=True, meta_path=meta_file,
    )
    classify_snapshots(meta, prompt, classifier=clf, resume=True, meta_path=meta_file,
                       workers=opts["classify_workers"])
    render_redacted(
        meta, out, crf=opts["crf"], preset=opts["preset"], selective=opts["selective"],
        encoder=opts["encoder"], max_gap=opts["max_gap"], smooth_window=opts["smooth_window"],
        bridge_iou=opts["bridge_iou"], min_track_frames=opts["min_track_frames"],
    )
    return meta
