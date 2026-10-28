"""End-to-end pipeline (issue #16): track -> snapshot -> classify -> render.

Stages are cached through the metadata JSON plus a stage-key manifest (see
:mod:`auto_redact.cache`): a re-run reuses a stage only when the settings
that produced it (and everything upstream of it) are unchanged. So
changing the prompt re-classifies, changing the snapshot padding
re-snapshots and re-classifies, and changing the stride re-tracks.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, ContextManager

from . import cache, telemetry
from .classify import classify_snapshots
from .config import Config
from .labels import apply_labels, load_labels
from .meta import Metadata, MetaError, load_meta, save_meta, save_meta_summary
from .render import render_redacted
from .snapshot import select_snapshots
from .workspace import labels_path_for, resolve_run_paths
from .track import add_orphan_tracklets, build_tracklets


@dataclass
class RunSummary:
    out: Path
    meta_path: Path
    tracklets: int
    redacted: int
    reused: list[str]


def snapshots_done(meta: Metadata) -> bool:
    return all(
        t.best_snapshot_path is not None and Path(t.best_snapshot_path).is_file()
        for t in meta.tracklets
    )


def run_pipeline(
    video: str | Path,
    prompt: str,
    out: str | Path,
    config: Config | None = None,
    *,
    meta_path: str | Path | None = None,
    thumbs_dir: str | Path | None = None,
    force: bool = False,
    progress: Callable[[str], None] = lambda message: None,
    bar: Callable[[str], ContextManager[Callable[[int, int], None] | None]] = (
        lambda desc: contextlib.nullcontext()
    ),
) -> RunSummary:
    cfg = config if config is not None else Config()
    resume = cfg.resume
    meta_file, thumbs = resolve_run_paths(video, out, meta_path, thumbs_dir, progress)
    stages = cache.StageCache(meta_file)
    reused: list[str] = []

    track_key = cache.track_key(video, cfg.track)
    snapshot_key = cache.snapshot_key(track_key, cfg.snapshot, thumbs)
    classify_key = cache.classify_key(snapshot_key, prompt, cfg.classify)

    meta: Metadata | None = None
    if meta_file.is_file() and not force:
        legacy = not stages.path.exists()
        if not legacy and not stages.matches("track", track_key):
            progress("[track] cached metadata was made with other settings; re-tracking")
        else:
            try:
                meta = load_meta(meta_file)
            except MetaError as error:
                progress(f"[track] ignoring unreadable metadata ({error})")
            else:
                if legacy:
                    adopted = cache.adopt_legacy(
                        stages, meta_file, meta,
                        {"track": track_key, "snapshot": snapshot_key, "classify": classify_key},
                        prompt, cfg.classify,
                    )
                    progress(
                        "[track] metadata predates the stage cache; trusting its "
                        f"{', '.join(adopted)} results"
                    )

    if meta is None:
        progress(f"[1/4 track] tracking people in {video}")
        stages.begin("track", track_key)
        with telemetry.stage("track"), bar("track") as on_progress:
            meta = build_tracklets(
                video, cfg.track, on_progress=on_progress, meta_path=meta_file,
                resume=resume and not force,
            )
        if cfg.track.orphans:
            with telemetry.stage("orphan"), bar("orphan") as orphan_progress:
                meta = add_orphan_tracklets(meta, video, on_progress=orphan_progress)
        if cfg.track.scene_cuts:
            progress(
                f"      scene cuts: {len(meta.scene_cuts)} cut(s), "
                f"{meta.scene_cut_splits} tracklet split(s)"
            )
        save_meta(meta_file, meta)
        stages.record("track", track_key)
    else:
        reused.append("track")
        progress(f"[1/4 track] reusing {meta_file}")

    progress(f"      {len(meta.tracklets)} tracklet(s)")

    if (
        "track" in reused
        and stages.matches("snapshot", snapshot_key)
        and snapshots_done(meta)
    ):
        reused.append("snapshot")
        progress("[2/4 snapshot] reusing existing snapshots")
    else:
        progress("[2/4 snapshot] selecting best frames")
        resume_snapshots = resume and stages.begin("snapshot", snapshot_key)
        with telemetry.stage("snapshot"), bar("snapshot") as on_progress:
            for tracklet in meta.tracklets:
                tracklet.best_frame = None
                tracklet.best_snapshot_path = None
                tracklet.isolated_track_ids = []
                tracklet.redact = None
                tracklet.score = None
                tracklet.detail = None
            select_snapshots(
                meta,
                thumbs,
                cfg.snapshot,
                on_progress=on_progress,
                resume=resume_snapshots,
                meta_path=meta_file,
            )
        save_meta(meta_file, meta)
        stages.record("snapshot", snapshot_key)

    classify = cfg.classify
    if (
        stages.matches("classify", classify_key)
        and all(t.redact is not None for t in meta.tracklets)
    ):
        reused.append("classify")
        progress("[3/4 classify] reusing existing decisions")
    else:
        resume_classify = resume and stages.begin("classify", classify_key)
        for tracklet in meta.tracklets:
            tracklet.redact = None
            tracklet.score = None
            tracklet.detail = None
        progress(f"[3/4 classify] asking {classify.resolved_model}: {prompt!r}")
        progress(
            "      initializing classifier model"
            if classify.classifier == "siglip"
            else "      waiting for ollama to load the model"
        )
        uses_prefilter = (
            classify.classifier == "vlm"
            and classify.prefilter in ("negatives", "positives")
        )
        with telemetry.stage("classify"), bar("classify") as on_progress, (
            bar("prefilter") if uses_prefilter else contextlib.nullcontext()
        ) as on_prefilter:
            classify_snapshots(
                meta,
                prompt,
                classify,
                on_progress=on_progress,
                on_prefilter_progress=on_prefilter,
                resume=resume_classify,
                meta_path=meta_file,
            )
        save_meta(meta_file, meta)
        stages.record("classify", classify_key)

    labels = load_labels(labels_path_for(meta_file))
    if labels.tracks:
        if labels.stale(track_key):
            progress("[labels] review labels were made for different tracking; ignoring them")
        elif apply_labels(meta, labels):
            progress(f"[labels] applying {len(labels.tracks)} review verdict(s)")
            save_meta_summary(meta_file, meta)

    redacted = sum(1 for t in meta.tracklets if t.effective_redact)
    progress(f"[4/4 render] redacting {redacted} of {len(meta.tracklets)}")
    with telemetry.stage("render"), bar("render") as on_progress:
        result = render_redacted(meta, out, cfg.render, on_progress=on_progress)
    return RunSummary(
        out=Path(result),
        meta_path=meta_file,
        tracklets=len(meta.tracklets),
        redacted=redacted,
        reused=reused,
    )
