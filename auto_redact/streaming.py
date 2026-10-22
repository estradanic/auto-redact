"""Streaming single-pass pipeline (Option A).

Reads the source video once. Tracks, scores best-view snapshots, detects scene
cuts and classifies people as they leave the frame, and writes redacted frames
to the output as soon as every person visible in them is decided. Frames whose
subjects are not yet decided are buffered (spilled to disk past a threshold)
and shipped later. Worst case (one subject present for the whole video) is
identical to the classic pipeline, but with one decode instead of several and
partial early output for short-lived subjects.

This module deliberately keeps the classic ``pipeline.run_pipeline`` untouched
so the two can be compared.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import subprocess
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, ContextManager

import cv2
import numpy as np
from PIL import Image

from .classify import (
    DEFAULT_MAX_IMAGE_SIZE,
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_URL,
    DEFAULT_TIMEOUT,
    DEFAULT_WORKERS,
    VlmClassifier,
    classify_snapshots,
)
from .frames import iter_frames, probe_video
from .meta import Metadata, MetaError, Tracklet, VideoInfo, load_meta, save_meta
from .render import (
    DEFAULT_BRIDGE_IOU,
    DEFAULT_CRF,
    DEFAULT_ENCODER,
    DEFAULT_MAX_GAP,
    DEFAULT_MIN_TRACK_FRAMES,
    DEFAULT_PRESET,
    DEFAULT_SMOOTH_WINDOW,
    ENCODERS,
    _clamp_box,
    _fill_tracklet_gaps,
    _smooth_boxes,
    redaction_boxes,
    _Segment,
    _copy_segment,
    _encode_segment,
    _concat_video,
    _mux_audio,
    _keyframe_table,
)
from .pipeline import _classify_key
from .scene import _frame_hist
from .siglip import DEFAULT_SIGLIP_MODEL, DEFAULT_THRESHOLD, SiglipClassifier
from .snapshot import (
    CLOSEUP_AREA_FRACTION,
    DEFAULT_CONTACT_SHEET,
    DEFAULT_PADDING,
    DISTRACTOR_WEIGHT,
    EDGE_PENALTY,
    EDGE_TOLERANCE,
    HEAD_WEIGHT,
    BODY_WEIGHT,
    ISOLATE_DILATE,
    SAME_PERSON_IOU,
    _clamped_area,
    _padded_crop_window,
    crop_snapshot,
    observation_score,
    redact_others,
    sub_isolate_people,
    write_contact_sheet,
)
from .track import _select_device, _load_yolo, _extract, _iou, _same_video


# Reuse the same defaults as the classic pipeline where sensible.
DEFAULT_SPILL_THRESHOLD = 128


@dataclass
class _Active:
    internal_id: int
    tracker_id: int
    frames: list[int] = field(default_factory=list)
    boxes: list[list[float]] = field(default_factory=list)
    keypoints: list[list[list[float]]] = field(default_factory=list)
    first_frame: int = 0
    last_frame: int = 0
    best_frame: int | None = None
    best_box: list[float] | None = None
    best_score: float = -1.0
    best_others: list[tuple[int, list[float]]] = field(default_factory=list)
    closed: bool = False


def _select_tracker_device(device):
    return _select_device(device)


def _bgr(frame_rgb):
    return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)


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
    track_workers=1,  # accepted for compat; streaming tracks in one pass
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
    crf=18,
    preset="medium",
    selective=True,  # accepted; streaming uses full re-encode
    encoder="libx264",
    max_gap=DEFAULT_MAX_GAP,
    smooth_window=DEFAULT_SMOOTH_WINDOW,
    bridge_iou=DEFAULT_BRIDGE_IOU,  # accepted; bridging skipped in streaming
    min_track_frames=DEFAULT_MIN_TRACK_FRAMES,
    scene_cuts=True,
    scene_cut_threshold=0.45,
    spill_threshold=DEFAULT_SPILL_THRESHOLD,
    bookends=True,
    progress=lambda message: None,
    bar=lambda desc: contextlib.nullcontext(),
) -> dict:
    if classifier not in ("vlm", "siglip"):
        raise ValueError(f"unknown classifier {classifier!r}")
    if model is None:
        model = DEFAULT_SIGLIP_MODEL if classifier == "siglip" else DEFAULT_MODEL

    # Defer rendering (and frame buffering) when features that need a full pass
    # are requested: orphan recovery and selective GOP copy. Decision-making
    # (track + snapshot + classify) still happens in one pass.
    defer_render = bool(orphans)

    video = Path(video)
    out = Path(out)
    meta_file = Path(meta_path) if meta_path else out.with_name(out.stem + ".meta.json")
    thumbs = Path(thumbs_dir) if thumbs_dir else meta_file.with_name(meta_file.stem + "_thumbs")
    thumbs.mkdir(parents=True, exist_ok=True)
    sidecar = meta_file.with_name(meta_file.name + ".classify.json")

    info = probe_video(video)
    device = _select_device(device)

    # Reuse cached result if available (mirrors classic pipeline).
    if (not force) and meta_file.is_file() and out.is_file():
        try:
            loaded = load_meta(meta_file)
        except MetaError:
            loaded = None
        if loaded is not None and _same_video(loaded, video, info):
            key = _classify_key(prompt, classifier, model, threshold, think)
            try:
                stored = json.loads(sidecar.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                stored = None
            if stored == key and all(t.redact is not None for t in loaded.tracklets):
                progress("reusing cached result")
                redacted = sum(1 for t in loaded.tracklets if t.redact)
                return {"out": out, "meta_path": meta_file, "tracklets": len(loaded.tracklets), "redacted": redacted}

    selective_stream = False
    gops = []
    gop_frames = {}
    gop_blockers = {}
    shipped_gops = set()
    segments = []
    keyframes = {}
    is_open_gop = False
    if selective and not defer_render:
        keyframes, is_open_gop = _keyframe_table(video)
        if keyframes and not is_open_gop and 0 in keyframes:
            selective_stream = True
            boundaries = sorted({0, info.frame_count} | set(keyframes))
            gops = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1) if boundaries[i + 1] > boundaries[i]]
        else:
            selective_stream = False
    half = half and device.startswith("cuda")

    # classifier instance
    siglip = None
    if classifier == "siglip":
        progress("      initializing classifier model")
        siglip = SiglipClassifier(model, threshold, classify_device)
    else:
        progress("      waiting for ollama to load the model")

    def make_vlm():
        return VlmClassifier(model, ollama_url, timeout, think, "10m", max_image_size)

    yolo = _load_yolo(device if device != "auto" else None)
    # ensure on device
    try:
        yolo.to(device)
    except Exception:
        pass

    w, h = info.width, info.height
    fps = info.fps

    # state
    next_internal_id = 1
    active: dict[int, _Active] = {}  # tracker_id -> active
    closed_tracklets: list[_Active] = []
    redact_by_id: dict[int, bool | None] = {}
    score_by_id: dict[int, float | None] = {}
    snapshot_path_by_id: dict[int, str | None] = {}
    isolated_by_id: dict[int, list[int]] = {}
    best_frame_by_id: dict[int, int | None] = {}

    # render boxes accumulated as tracklets close
    render_boxes: dict[int, list[tuple[int, int, int, int]]] = {}

    # frame buffer: idx -> ndarray or Path (spilled)
    frame_buffer: dict[int, Any] = {}
    spill_dir = Path(tempfile.mkdtemp(prefix="auto-redact-stream-"))
    in_memory_count = [0]

    frame_blockers: dict[int, set[int]] = {}

    next_ship = [0]
    # ffmpeg
    out.parent.mkdir(parents=True, exist_ok=True)
    has_audio = info.has_audio
    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", f"{w}x{h}",
        "-framerate", f"{int(fps)}/{1}" if float(fps).is_integer() else str(fps),
        "-i", "pipe:0",
    ]
    # use fps arg helper-ish
    from fractions import Fraction
    rate = Fraction(fps).limit_denominator(1001)
    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", f"{w}x{h}",
        "-framerate", f"{rate.numerator}/{rate.denominator}",
        "-i", "pipe:0",
    ]
    if has_audio:
        cmd += ["-i", str(video)]
    cmd += ["-map", "0:v:0"]
    if has_audio:
        cmd += ["-map", "1:a:0", "-c:a", "copy"]
    cmd += ["-c:v", "libx264", "-crf", str(crf), "-preset", preset,
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    proc = None
    if not defer_render and not selective_stream:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    sub_model = None
    if sub_isolate:
        try:
            sub_model = _load_yolo("cpu")
        except Exception:
            sub_model = None

    precision = {"quantize": 16} if half else {}
    track_kwargs = dict(conf=track_conf, iou=track_iou, persist=True, verbose=False, tracker=track_tracker, **precision)
    if track_imgsz is not None:
        track_kwargs["imgsz"] = track_imgsz

    close_threshold = max_gap if max_gap > 0 else 30

    def spill_one():
        # spill the earliest in-memory frame that is blocked (not shippable now)
        candidates = [idx for idx, v in frame_buffer.items() if not isinstance(v, Path)]
        if not candidates:
            return
        candidates.sort()
        # prefer one that can't ship now
        target = None
        for idx in candidates:
            if idx != next_ship[0]:
                target = idx
                break
        if target is None:
            target = candidates[0]
        arr = frame_buffer.pop(target)
        path = spill_dir / f"frame_{target}.npy"
        np.save(path, arr)
        frame_buffer[target] = path
        in_memory_count[0] -= 1

    def store_frame(idx, arr):
        if in_memory_count[0] >= spill_threshold:
            spill_one()
        frame_buffer[idx] = arr
        in_memory_count[0] += 1

    def load_frame(idx):
        v = frame_buffer.get(idx)
        if isinstance(v, Path):
            return np.load(v)
        return v

    def close_and_classify(at: _Active):
        nonlocal next_internal_id
        if at.closed:
            return
        at.closed = True
        redact = None
        score = None
        snap_path = None
        isolated_ids: list[int] = []

        if at.best_box is not None and at.best_frame is not None:
            best_arr = None
            for fi, fr in iter_frames(video, start=at.best_frame, stop=at.best_frame + 1):
                best_arr = fr
                break
            if best_arr is not None:
                try:
                    crop = crop_snapshot(best_arr, at.best_box, padding)
                except Exception:
                    crop = None
                if crop is not None:
                    from .snapshot import _padded_crop_window, _clamped_area
                    window = _padded_crop_window(at.best_box, w, h, padding)
                    area_fraction = _clamped_area(at.best_box, w, h) / max(1.0, float(w) * float(h))
                    closeup = area_fraction >= closeup_area_fraction
                    if isolate and not closeup:
                        crop, painted = redact_others(
                            crop, window, at.best_box, at.best_others,
                            same_person_iou=same_person_iou, dilate=isolate_dilate,
                        )
                        isolated_ids = painted
                    if sub_isolate and sub_model is not None and not closeup:
                        sub_isolate_people(crop, window, at.best_box, sub_model, dilate=isolate_dilate)
                    image = Image.fromarray(crop)
                    thumb_path = thumbs / f"track_{at.internal_id}.jpg"
                    image.save(thumb_path, format="JPEG", quality=90)
                    snap_path = str(thumb_path)

                    # classify
                    if classifier == "siglip":
                        dec = siglip.classify(snap_path, prompt)
                    else:
                        clf = make_vlm()
                        dec = clf.classify(snap_path, prompt)
                    redact = dec.redact
                    score = dec.score

        # compute render boxes if redacted
        if redact is True:
            filled_frames, filled_boxes = _fill_tracklet_gaps(at.frames, at.boxes, max_gap)
            if smooth_window > 1 and len(filled_boxes) > 1:
                filled_boxes = _smooth_boxes(filled_frames, filled_boxes, smooth_window)
            for f, box in zip(filled_frames, filled_boxes):
                pb = _clamp_box(box, w, h)
                if pb is not None:
                    render_boxes.setdefault(f, []).append(pb)

        closed_tracklets.append(at)
        redact_by_id[at.internal_id] = redact
        score_by_id[at.internal_id] = score
        snapshot_path_by_id[at.internal_id] = snap_path
        isolated_by_id[at.internal_id] = isolated_ids
        best_frame_by_id[at.internal_id] = at.best_frame

        # unblock frames
        for idx in list(frame_blockers.keys()):
            if at.internal_id in frame_blockers[idx]:
                frame_blockers[idx].discard(at.internal_id)
        # unblock GOPs (selective streaming)
        for gs in list(gop_blockers.keys()):
            gop_blockers[gs].discard(at.internal_id)

    # bookend pass: track first and last chunks quickly to classify boundary tracklets early.
    # We implement by simply running the main loop over the whole video; bookends toggle
    # only affects nothing here (classification happens at close). For simplicity, bookends
    # flag is accepted but main loop already classifies at close. (No-op kept for compat.)
    # Main pass
    prev_hist = None
    total = info.frame_count
    progress(f"[stream] tracking+rendering {video}")

    # We'll iterate all frames
    frame_iter = iter_frames(video, sample_rate=1)
    # track sampled frames separately; but iter_frames sample_rate=1 yields all.
    last_seen_global = 0

    for frame_idx, frame_rgb in frame_iter:
        # progress
        if frame_idx % 50 == 0:
            progress(f"      frame {frame_idx}/{total}")

        # scene cut
        bgr = _bgr(frame_rgb)
        hist = _frame_hist(bgr)
        if prev_hist is not None and scene_cuts:
            dist = float(__import__("cv2").compareHist(prev_hist, hist, __import__("cv2").HISTCMP_BHATTACHARYYA))
            if dist >= scene_cut_threshold:
                for tid in list(active.keys()):
                    close_and_classify(active.pop(tid))
        prev_hist = hist

        # tracking on sampled frames
        if frame_idx % sample_rate == 0:
            # Use model.track returns list; _extract fills obs dict
            obs: dict[int, list] = {}
            for result in yolo.track(frame_rgb, **track_kwargs):
                _extract(result, frame_idx, obs)
            # obs: tid -> [(frame_idx, box, kps)]
            for tid, frames_obs in obs.items():
                for (fi, box, kps) in frames_obs:
                    if tid not in active:
                        at = _Active(internal_id=next_internal_id, tracker_id=tid, first_frame=fi)
                        next_internal_id += 1
                        active[tid] = at
                    at = active[tid]
                    if at.closed:
                        # reopen as new tracklet
                        at = _Active(internal_id=next_internal_id, tracker_id=tid, first_frame=fi)
                        next_internal_id += 1
                        active[tid] = at
                    at.frames.append(fi)
                    at.boxes.append(list(box))
                    at.keypoints.append(kps)
                    at.last_frame = fi
                    # scoring
                    # build frame obs list for this frame from active tracklets observed this frame
                    # We'll score after collecting all tids for this frame; but we need all boxes.
                    pass
            # Now score best snapshots for tracklets observed this frame.
            # Build list of (internal_id, box) for tracklets observed in this frame.
            frame_obs_list = []
            for tid, frames_obs in obs.items():
                at = active.get(tid)
                if at is None or at.closed:
                    continue
                # find the box for this frame (last appended)
                frame_obs_list.append((at.internal_id, list(at.boxes[-1]), at.keypoints[-1] if at.keypoints else []))
            # for each, score
            for internal_id, box, kps in frame_obs_list:
                others = [(oid, obox) for (oid, obox, _) in [(x[0], x[1], None) for x in frame_obs_list] if oid != internal_id]
                # others need boxes
                other_boxes = [obox for (oid, obox, _) in [(x[0], x[1], None) for x in frame_obs_list] if oid != internal_id]
                # reconstruct other_boxes properly
                other_boxes = [x[1] for x in frame_obs_list if x[0] != internal_id]
                score = observation_score(
                    box, kps, w, h, other_boxes,
                    padding=padding, head_weight=head_weight, body_weight=body_weight,
                    edge_penalty=edge_penalty, edge_tolerance=edge_tolerance,
                    distractor_weight=distractor_weight, same_person_iou=same_person_iou,
                )
                # find active at
                at = None
                for a in active.values():
                    if a.internal_id == internal_id:
                        at = a
                        break
                if at is None:
                    continue
                if score > at.best_score:
                    at.best_score = score
                    at.best_box = list(box)
                    at.best_frame = frame_idx
                    at.best_others = [(oid, list(obox)) for (oid, obox, _) in [(x[0], x[1], None) for x in frame_obs_list] if oid != internal_id]
                    # fix best_others to use boxes
                    at.best_others = [(oid, list(obox)) for oid, obox in [(x[0], x[1]) for x in frame_obs_list] if oid != internal_id]

        # update last_seen for active tracklets observed this frame
        observed_ids = set()
        if frame_idx % sample_rate == 0:
            for tid in obs.keys():
                observed_ids.add(tid)
        for tid, at in list(active.items()):
            if at.closed:
                continue
            if tid in observed_ids:
                at.last_frame = frame_idx
            else:
                # dropout close
                if frame_idx - at.last_frame > close_threshold:
                    close_and_classify(active.pop(tid))

        def _gop_start_of(fidx):
            for gs, ge in gops:
                if gs <= fidx < ge:
                    return gs
            return gops[-1][0] if gops else 0

        if selective_stream:
            gs = _gop_start_of(frame_idx)
            gop_frames.setdefault(gs, []).append((frame_idx, frame_rgb))
            # track blockers for this GOP
            blockers = gop_blockers.setdefault(gs, set())
            for tid, at in active.items():
                if at.closed:
                    continue
                if at.first_frame <= frame_idx <= at.last_frame + max_gap:
                    blockers.add(at.internal_id)
            # try to ship fully-buffered GOPs in order
            for gstart, gend in gops:
                if gstart in shipped_gops:
                    continue
                if gend > frame_idx + 1:
                    break
                if gop_blockers.get(gstart):
                    break
                # ship this GOP
                frames = gop_frames.pop(gstart, [])
                edited = any(idx in render_boxes for idx, _ in frames)
                seg = _Segment(gstart, gend, edited)
                seg_path = Path(tempfile.mkdtemp(prefix="auto-redact-gop-")) / f"gop_{gstart:06d}.mp4"
                # reuse a single temp dir? use spill_dir
                seg_path = spill_dir / f"gop_{gstart:06d}.mp4"
                if edited:
                    _encode_segment(video, seg_path, render_boxes, keyframes, seg, info, crf, preset, encoder)
                else:
                    _copy_segment(video, seg_path, keyframes, seg)
                segments.append(seg_path)
                shipped_gops.add(gstart)
        elif not defer_render:
            # blockers for this frame
            blockers = set()
            for tid, at in active.items():
                if at.closed:
                    continue
                if at.first_frame <= frame_idx <= at.last_frame + max_gap:
                    blockers.add(at.internal_id)
            frame_blockers[frame_idx] = blockers

            # store frame
            store_frame(frame_idx, frame_rgb)

            # ship eligible
            while next_ship[0] in frame_buffer:
                idx = next_ship[0]
                if frame_blockers.get(idx):
                    break
                arr = load_frame(idx)
                if arr is None:
                    break
                # paint
                for box in render_boxes.get(idx, []):
                    cv2.rectangle(arr, (box[0], box[1]), (box[2], box[3]), (0, 0, 0), thickness=-1)
                try:
                    proc.stdin.write(arr.tobytes())
                except BrokenPipeError:
                    break
                # remove from buffer
                v = frame_buffer.pop(idx, None)
                if isinstance(v, Path):
                    try:
                        v.unlink(missing_ok=True)
                    except Exception:
                        pass
                in_memory_count[0] = max(0, in_memory_count[0] - (0 if isinstance(v, Path) else 1))
                next_ship[0] += 1

    # end: close all remaining active
    for tid in list(active.keys()):
        close_and_classify(active.pop(tid))

    if (not defer_render) and (not selective_stream):
        # ship remaining buffered frames
        while next_ship[0] < info.frame_count:
            idx = next_ship[0]
            if idx in frame_buffer or idx < info.frame_count:
                if idx in frame_buffer:
                    arr = load_frame(idx)
                    if arr is not None:
                        for box in render_boxes.get(idx, []):
                            cv2.rectangle(arr, (box[0], box[1]), (box[2], box[3]), (0, 0, 0), thickness=-1)
                        try:
                            proc.stdin.write(arr.tobytes())
                        except BrokenPipeError:
                            pass
                        v = frame_buffer.pop(idx, None)
                        if isinstance(v, Path):
                            try: v.unlink(missing_ok=True)
                            except Exception: pass
            next_ship[0] += 1

        try:
            proc.stdin.close()
        except Exception:
            pass
        if proc is not None:
            proc.wait()

    if selective_stream:
        # ship any remaining GOPs
        for gstart, gend in gops:
            if gstart in shipped_gops:
                continue
            gop_blockers.get(gstart, set()).clear()
            frames = gop_frames.pop(gstart, [])
            edited = any(idx in render_boxes for idx, _ in frames)
            seg = _Segment(gstart, gend, edited)
            seg_path = spill_dir / f"gop_{gstart:06d}.mp4"
            if edited:
                _encode_segment(video, seg_path, render_boxes, keyframes, seg, info, crf, preset, encoder)
            else:
                _copy_segment(video, seg_path, keyframes, seg)
            segments.append(seg_path)
            shipped_gops.add(gstart)
        joined = spill_dir / "joined.mp4"
        _concat_video(segments, joined)
        _mux_audio(joined, video, out)

    # build metadata
    tracklets = []
    for at in sorted(closed_tracklets, key=lambda a: a.internal_id):
        t = Tracklet(
            track_id=at.internal_id,
            frames=list(at.frames),
            boxes=[list(b) for b in at.boxes],
            keypoints=[list(k) for k in at.keypoints],
            best_frame=best_frame_by_id.get(at.internal_id),
            best_snapshot_path=snapshot_path_by_id.get(at.internal_id),
            isolated_track_ids=list(isolated_by_id.get(at.internal_id, [])),
            redact=redact_by_id.get(at.internal_id),
            score=score_by_id.get(at.internal_id),
        )
        tracklets.append(t)
    meta = Metadata(video=VideoInfo(path=str(video), fps=fps, width=w, height=h, frames=info.frame_count), tracklets=tracklets)
    try:
        save_meta(meta_file, meta)
    except Exception:
        pass

    # Orphan recovery + deferred render (selective / orphans) support.
    if orphans:
        from .track import add_orphan_tracklets

        meta = add_orphan_tracklets(meta, video, on_progress=None)
        from .snapshot import select_snapshots

        select_snapshots(
            meta,
            thumbs_dir=thumbs,
            padding=padding,
            isolate=isolate,
            sub_isolate=sub_isolate,
            closeup_area_fraction=closeup_area_fraction,
            head_weight=head_weight,
            body_weight=body_weight,
            edge_penalty=edge_penalty,
            edge_tolerance=edge_tolerance,
            distractor_weight=distractor_weight,
            same_person_iou=same_person_iou,
            isolate_dilate=isolate_dilate,
            contact_sheet=DEFAULT_CONTACT_SHEET,
            resume=True,
            meta_path=meta_file,
        )
        from .classify import classify_snapshots as _classify

        if classifier == "siglip":
            _classify(meta, prompt, classifier=siglip, resume=True, meta_path=meta_file)
        else:
            _classify(
                meta, prompt, model=model, ollama_url=ollama_url, timeout=timeout,
                workers=classify_workers, think=think, max_image_size=max_image_size,
                resume=True, meta_path=meta_file,
            )
        try:
            save_meta(meta_file, meta)
        except Exception:
            pass

    if defer_render:
        from .render import render_redacted

        render_redacted(
            meta, out, crf=crf, preset=preset, selective=selective, encoder=encoder,
            max_gap=max_gap, smooth_window=smooth_window, bridge_iou=bridge_iou,
            min_track_frames=min_track_frames, on_progress=None,
        )

    # sidecar
    key = {"prompt": prompt.strip(), "model": model}
    if classifier == "siglip":
        key.update(classifier=classifier, threshold=threshold)
    sidecar.write_text(json.dumps(key) + "\n", encoding="utf-8")

    # contact sheet
    try:
        saved = []
        for at in closed_tracklets:
            sp = snapshot_path_by_id.get(at.internal_id)
            if sp and Path(sp).is_file():
                img = Image.open(sp).convert("RGB")
                img.thumbnail((192, 256), Image.Resampling.LANCZOS)
                saved.append((at.internal_id, img))
        saved.sort(key=lambda x: x[0])
        if saved:
            write_contact_sheet(saved, thumbs / DEFAULT_CONTACT_SHEET)
    except Exception:
        pass

    # cleanup spill dir
    try:
        import shutil
        shutil.rmtree(spill_dir, ignore_errors=True)
    except Exception:
        pass

    redacted = sum(1 for t in meta.tracklets if t.redact)
    progress(f"{len(meta.tracklets)} tracklet(s), {redacted} redacted; wrote {out}")
    return {"out": out, "meta_path": meta_file, "tracklets": len(meta.tracklets), "redacted": redacted}
