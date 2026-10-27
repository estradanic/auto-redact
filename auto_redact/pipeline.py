"""End-to-end pipeline (issue #16): track -> snapshot -> classify -> render.

Stages are cached through the metadata JSON: a re-run reuses whatever the
existing metadata already holds. The classify stage additionally records
the prompt/classifier settings it ran with in a sidecar ``<meta>.classify.json`` so that
changing the prompt re-classifies instead of silently reusing stale flags.
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, ContextManager

from .classify import (
    DEFAULT_MAX_IMAGE_SIZE,
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_URL,
    DEFAULT_TIMEOUT,
    DEFAULT_WORKERS,
    classify_snapshots,
)
from .meta import Metadata, MetaError, load_meta, save_meta
from .render import (
    DEFAULT_BRIDGE_IOU,
    DEFAULT_CRF,
    DEFAULT_ENCODER,
    DEFAULT_MAX_GAP,
    DEFAULT_MIN_TRACK_FRAMES,
    DEFAULT_PRESET,
    DEFAULT_SMOOTH_WINDOW,
    render_redacted,
)
from .scene import DEFAULT_CUT_THRESHOLD, apply_scene_cuts
from . import telemetry
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
    select_snapshots,
)
from .track import build_tracklets


@dataclass
class RunSummary:
    out: Path
    meta_path: Path
    tracklets: int
    redacted: int
    reused: list[str]


def default_meta_path(out: str | Path) -> Path:
    out = Path(out)
    return out.with_name(out.stem + ".meta.json")


def _same_file(a: str | Path, b: str | Path) -> bool:
    return Path(a).resolve() == Path(b).resolve()


def _snapshots_done(meta: Metadata) -> bool:
    return all(
        t.best_snapshot_path is not None and Path(t.best_snapshot_path).is_file()
        for t in meta.tracklets
    )


def _classify_key(
    prompt: str, classifier: str, model: str, threshold: float, think: bool = False
) -> dict[str, object]:
    key: dict[str, object] = {"prompt": prompt.strip(), "model": model}
    if classifier == "siglip":
        key.update(classifier=classifier, threshold=threshold)
    elif think:
        key["think"] = True
    return key


def run_pipeline(
    video: str | Path,
    prompt: str,
    out: str | Path,
    *,
    meta_path: str | Path | None = None,
    thumbs_dir: str | Path | None = None,
    force: bool = False,
    sample_rate: int = 1,
    device: str = "auto",
    half: bool = True,
    track_conf: float = 0.25,
    track_iou: float = 0.7,
    track_imgsz: int | None = None,
    track_tracker: str = "bytetrack.yaml",
    orphans: bool = False,
    resume: bool = True,
    padding: float = DEFAULT_PADDING,
    isolate: bool = True,
    sub_isolate: bool = False,
    closeup_area_fraction: float = CLOSEUP_AREA_FRACTION,
    head_weight: float = HEAD_WEIGHT,
    body_weight: float = BODY_WEIGHT,
    edge_penalty: float = EDGE_PENALTY,
    edge_tolerance: float = EDGE_TOLERANCE,
    distractor_weight: float = DISTRACTOR_WEIGHT,
    same_person_iou: float = SAME_PERSON_IOU,
    isolate_dilate: float = ISOLATE_DILATE,
    classifier: str = "vlm",
    model: str | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    classify_device: str = "auto",
    ollama_url: str = DEFAULT_OLLAMA_URL,
    timeout: float = DEFAULT_TIMEOUT,
    classify_workers: int = DEFAULT_WORKERS,
    think: bool = False,
    max_image_size: int = DEFAULT_MAX_IMAGE_SIZE,
    prefilter: str = "none",
    prefilter_model: str | None = None,
    prefilter_threshold: float | None = None,
    prefilter_device: str = "auto",
    crf: int = DEFAULT_CRF,
    preset: str = DEFAULT_PRESET,
    selective: bool = True,
    encoder: str = DEFAULT_ENCODER,
    max_gap: int = DEFAULT_MAX_GAP,
    smooth_window: int = DEFAULT_SMOOTH_WINDOW,
    bridge_iou: float = DEFAULT_BRIDGE_IOU,
    min_track_frames: int = DEFAULT_MIN_TRACK_FRAMES,
    scene_cuts: bool = True,
    scene_cut_threshold: float = DEFAULT_CUT_THRESHOLD,
    progress: Callable[[str], None] = lambda message: None,
    bar: Callable[[str], ContextManager[Callable[[int, int], None] | None]] = (
        lambda desc: contextlib.nullcontext()
    ),
) -> RunSummary:
    if classifier not in ("vlm", "siglip"):
        raise ValueError(f"unknown classifier {classifier!r}; use 'vlm' or 'siglip'")
    if model is None:
        model = DEFAULT_SIGLIP_MODEL if classifier == "siglip" else DEFAULT_MODEL
    meta_file = Path(meta_path) if meta_path else default_meta_path(out)
    thumbs = Path(thumbs_dir) if thumbs_dir else meta_file.with_name(
        meta_file.stem + "_thumbs"
    )
    sidecar = meta_file.with_name(meta_file.name + ".classify.json")
    reused: list[str] = []

    meta: Metadata | None = None
    if meta_file.is_file() and not force:
        try:
            meta = load_meta(meta_file)
        except MetaError as error:
            progress(f"[track] ignoring unreadable metadata ({error})")
        else:
            if not _same_file(meta.video.path, video):
                progress(
                    f"[track] cached metadata is for {meta.video.path}, "
                    "not this video; re-tracking"
                )
                meta = None

    if meta is None:
        progress(f"[1/4 track] tracking people in {video}")
        with telemetry.stage("track"), bar("track") as on_progress:
            meta = build_tracklets(
                video,
                sample_rate=sample_rate,
                device=device,
                on_progress=on_progress,
                half=half,
                conf=track_conf,
                iou=track_iou,
                imgsz=track_imgsz,
                tracker=track_tracker,
                scene_cuts=scene_cuts,
                scene_cut_threshold=scene_cut_threshold,
                meta_path=meta_file,
                force=force,
                resume=resume,
            )
        if orphans:
            from auto_redact.track import add_orphan_tracklets
            with telemetry.stage("orphan"), bar("orphan") as orphan_progress:
                meta = add_orphan_tracklets(
                    meta, video, on_progress=orphan_progress
                )
        if scene_cuts:
            cuts = getattr(meta, "scene_cuts", [])
            n_splits = getattr(meta, "scene_cut_splits", 0)
            progress(
                f"      scene cuts: {len(cuts)} cut(s), {n_splits} tracklet split(s)"
            )
        save_meta(meta_file, meta)
        sidecar.unlink(missing_ok=True)
        snapshots_cached = classified_cached = False
    else:
        reused.append("track")
        progress(f"[1/4 track] reusing {meta_file}")
        snapshots_cached = _snapshots_done(meta)

    progress(f"      {len(meta.tracklets)} tracklet(s)")

    if snapshots_cached:
        reused.append("snapshot")
        progress("[2/4 snapshot] reusing existing snapshots")
    else:
        progress("[2/4 snapshot] selecting best frames")
        with telemetry.stage("snapshot"), bar("snapshot") as on_progress:
            if not resume:
                for tracklet in meta.tracklets:
                    tracklet.best_frame = None
                    tracklet.best_snapshot_path = None
                    tracklet.isolated_track_ids = []
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
                on_progress=on_progress,
                resume=resume,
                meta_path=meta_file,
            )
        for tracklet in meta.tracklets:
            tracklet.redact = None
        sidecar.unlink(missing_ok=True)
        save_meta(meta_file, meta)

    key = _classify_key(prompt, classifier, model, threshold, think)
    try:
        stored = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        stored = None
    classified_cached = (
        not force
        and stored == key
        and all(t.redact is not None for t in meta.tracklets)
    )
    if classified_cached:
        reused.append("classify")
        progress("[3/4 classify] reusing existing decisions")
    else:
        if stored is not None and stored != key:
            for tracklet in meta.tracklets:
                tracklet.redact = None
                tracklet.score = None
        progress(f"[3/4 classify] asking {model}: {prompt!r}")
        siglip = None
        if classifier == "siglip":
            progress("      initializing classifier model")
            siglip = SiglipClassifier(model, threshold, classify_device)
        else:
            progress("      waiting for ollama to load the model")
        with telemetry.stage("classify"), bar("classify") as on_progress, (
            bar("prefilter") if prefilter in ("negatives", "positives") and siglip is None
            else contextlib.nullcontext()
        ) as on_prefilter:
            if siglip is not None:
                classify_snapshots(
                    meta,
                    prompt,
                    classifier=siglip,
                    on_progress=on_progress,
                    resume=resume,
                    meta_path=meta_file,
                )
            else:
                classify_snapshots(
                    meta,
                    prompt,
                    model=model,
                    ollama_url=ollama_url,
                    timeout=timeout,
                    workers=classify_workers,
                    think=think,
                    max_image_size=max_image_size,
                    prefilter=prefilter,
                    prefilter_model=prefilter_model,
                    prefilter_threshold=prefilter_threshold,
                    prefilter_device=prefilter_device,
                    on_progress=on_progress,
                    on_prefilter_progress=on_prefilter,
                    resume=resume,
                    meta_path=meta_file,
                )
        siglip = None  # let the render release its GPU memory
        save_meta(meta_file, meta)
        sidecar.write_text(json.dumps(key) + "\n", encoding="utf-8")

    redacted = sum(1 for t in meta.tracklets if t.redact)
    progress(f"[4/4 render] redacting {redacted} of {len(meta.tracklets)}")
    with telemetry.stage("render"), bar("render") as on_progress:
        result = render_redacted(
            meta,
            out,
            crf=crf,
            preset=preset,
            selective=selective,
            encoder=encoder,
            max_gap=max_gap,
            smooth_window=smooth_window,
            bridge_iou=bridge_iou,
            min_track_frames=min_track_frames,
            on_progress=on_progress,
        )
    return RunSummary(
        out=Path(result),
        meta_path=meta_file,
        tracklets=len(meta.tracklets),
        redacted=redacted,
        reused=reused,
    )
