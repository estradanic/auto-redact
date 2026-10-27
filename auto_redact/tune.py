"""Learn settings from review feedback.

Everything here is a pure function of the metadata, the human labels
(:mod:`auto_redact.labels`) and the render settings: no models, video or
UI. Three questions are answered:

**Why is this frame wrong?**  :func:`diagnose_leak` takes a spot where a
person is visible without a black box and works out which stage let them
through (never detected, tracked but not flagged, dropped as too short, in a
gap, lagging smoothing, or off the end of their track) and which setting, if
any, would have caught it. :func:`diagnose_over` does the same for a box that
should not be there.

**Which render settings fit my labels?**  :func:`suggest_settings` tries the
fixes the diagnoses point to (and small combinations of them), re-computes
the redaction boxes for each, and picks the one that covers the most
labeled leaks, then shows the fewest labeled over-redactions, then adds the
least black area.

**Which classifier setting fits my labels?**  :func:`sweep_threshold` finds
the SigLIP threshold that matches the labeled tracks (recall first: a missed
person is worse than an extra box), and :func:`evaluate_classifier` scores any
classifier/prompt on just the labeled tracks.

Suggestions are evidence for a human, never applied automatically.
"""

from __future__ import annotations

import bisect
import dataclasses
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .boxes import redaction_boxes
from .config import RenderConfig
from .geometry import PixelBox, box_iou, clamped_area
from .labels import FrameLabel, Labels
from .meta import Metadata, Tracklet

RENDER_KNOBS = ("max_gap", "smooth_window", "bridge_iou", "min_track_frames")
COVERED = 0.8  # a leak box counts as covered when this fraction is painted over


# --- coverage ------------------------------------------------------------------


def covered_fraction(box: list[float], painted: list[PixelBox] | tuple) -> float:
    """Fraction of ``box`` that the union of ``painted`` boxes covers."""
    x1, y1 = max(0, math.floor(box[0])), max(0, math.floor(box[1]))
    x2, y2 = math.ceil(box[2]), math.ceil(box[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    mask = np.zeros((y2 - y1, x2 - x1), dtype=bool)
    for px1, py1, px2, py2 in painted:
        ix1, iy1, ix2, iy2 = max(px1, x1), max(py1, y1), min(px2, x2), min(py2, y2)
        if ix2 > ix1 and iy2 > iy1:
            mask[iy1 - y1 : iy2 - y1, ix1 - x1 : ix2 - x1] = True
    return float(mask.mean())


def label_covered(label: FrameLabel, painted: list[PixelBox] | tuple) -> float:
    """Coverage of a frame label: of its box, or (no box) 1/0 for any box at all."""
    if label.box is not None:
        return covered_fraction(label.box, painted)
    return 1.0 if painted else 0.0


def total_area(boxes: dict[int, list[PixelBox]]) -> int:
    return sum((x2 - x1) * (y2 - y1) for frame in boxes.values() for x1, y1, x2, y2 in frame)


# --- diagnosing a frame ----------------------------------------------------------


@dataclass
class Diagnosis:
    """What went wrong at one labeled frame, and what might fix it."""

    frame: int
    kind: str  # "leak" or "over"
    cause: str
    message: str
    track_id: int | None = None
    fix: dict[str, Any] = field(default_factory=dict)  # render settings that would help
    advice: str = ""  # a fix that is not a render setting

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _nearest(frames: list[int], frame: int) -> tuple[int, int]:
    """Index and distance of the observation closest to ``frame``."""
    i = bisect.bisect_left(frames, frame)
    options = [j for j in (i - 1, i) if 0 <= j < len(frames)]
    best = min(options, key=lambda j: abs(frames[j] - frame))
    return best, abs(frames[best] - frame)


def _candidate(meta: Metadata, label: FrameLabel, reach: int) -> Tracklet | None:
    """The tracklet most plausibly meant by ``label``: one seen near it, overlapping it."""
    best, best_score = None, 0.0
    for tracklet in meta.tracklets:
        if not tracklet.frames:
            continue
        i, distance = _nearest(tracklet.frames, label.frame)
        if distance > reach:
            continue
        if label.box is None:
            score = 1.0 / (1 + distance)
        else:
            observed = tracklet.boxes[i]
            area = clamped_area(label.box, meta.video.width, meta.video.height)
            inter = clamped_area(
                [max(label.box[0], observed[0]), max(label.box[1], observed[1]),
                 min(label.box[2], observed[2]), min(label.box[3], observed[3])],
                meta.video.width, meta.video.height,
            )
            overlap = inter / area if area > 0 else 0.0
            score = max(box_iou(label.box, observed), overlap * 0.8) / (1 + distance / 30)
            if score < 0.1:
                continue
        if score > best_score:
            best, best_score = tracklet, score
    return best


def diagnose_leak(
    meta: Metadata,
    label: FrameLabel,
    config: RenderConfig,
    boxes: dict[int, list[PixelBox]] | None = None,
) -> Diagnosis:
    """Explain why a person visible at ``label.frame`` has no black box."""
    boxes = redaction_boxes(meta, config) if boxes is None else boxes
    frame = label.frame
    if label_covered(label, boxes.get(frame, ())) >= COVERED:
        return Diagnosis(frame, "leak", "covered", "this frame is covered with the current settings")

    reach = max(config.max_gap, 30) * 3
    tracklet = _candidate(meta, label, reach)
    if tracklet is None:
        return Diagnosis(
            frame, "leak", "not_detected",
            "no tracklet was near this person: the detector never found them here",
            advice="lower --track-conf, raise imgsz (--set track.imgsz=1280) or try --orphans, then re-track",
        )

    tid = tracklet.track_id
    decision = tracklet.effective_redact
    if decision is not True:
        score = f" (score {tracklet.score:.2f})" if tracklet.score is not None else ""
        return Diagnosis(
            frame, "leak", "not_flagged",
            f"track {tid} follows this person but was not flagged for redaction{score}",
            track_id=tid,
            advice=(
                (f"a SigLIP threshold at or below {tracklet.score:.2f} would flag it; "
                 if tracklet.score is not None else "")
                + "or mark the track 'redact' in the review page"
            ),
        )

    if len(tracklet.frames) < config.min_track_frames:
        return Diagnosis(
            frame, "leak", "dropped_short",
            f"track {tid} is flagged but has only {len(tracklet.frames)} observations "
            f"(min_track_frames is {config.min_track_frames})",
            track_id=tid, fix={"min_track_frames": len(tracklet.frames)},
        )

    first, last = tracklet.frames[0], tracklet.frames[-1]
    if frame < first or frame > last:
        edge = first - frame if frame < first else frame - last
        return Diagnosis(
            frame, "leak", "track_edge",
            f"track {tid} had not started / had already ended here (by {edge} frames): "
            "the tracker lost this person",
            track_id=tid,
            advice="no render setting extends a track; try a lower --track-conf, a different "
                   "--track-tracker (botsort.yaml) or --orphans, then re-track",
        )

    i, distance = _nearest(tracklet.frames, frame)
    if distance == 0:
        return Diagnosis(
            frame, "leak", "box_lag",
            f"track {tid} sees them here but the painted box misses part of them "
            "(smoothing lags fast movement)",
            track_id=tid, fix={"smooth_window": 1},
        )
    j = bisect.bisect_left(tracklet.frames, frame)
    gap = tracklet.frames[j] - tracklet.frames[j - 1] - 1
    return Diagnosis(
        frame, "leak", "gap",
        f"track {tid} has a {gap}-frame detection gap here (max_gap is {config.max_gap})",
        track_id=tid, fix={"max_gap": gap},
    )


def diagnose_over(
    meta: Metadata,
    label: FrameLabel,
    config: RenderConfig,
    boxes: dict[int, list[PixelBox]] | None = None,
) -> Diagnosis:
    """Explain why a black box is painted at ``label.frame``."""
    boxes = redaction_boxes(meta, config) if boxes is None else boxes
    frame = label.frame
    if label_covered(label, boxes.get(frame, ())) == 0.0:
        return Diagnosis(frame, "over", "clear", "no black box is painted here with the current settings")

    plain = redaction_boxes(meta, dataclasses.replace(config, bridge_iou=0.0))
    if len(plain.get(frame, [])) < len(boxes.get(frame, [])):
        return Diagnosis(
            frame, "over", "bridged",
            "this box only exists because two tracklets were bridged across a gap",
            fix={"bridge_iou": 0.0},
        )
    contributing = []
    for tracklet in meta.tracklets:
        if tracklet.effective_redact is not True or len(tracklet.frames) < config.min_track_frames:
            continue
        if tracklet.frames and tracklet.frames[0] <= frame <= tracklet.frames[-1]:
            i, distance = _nearest(tracklet.frames, frame)
            observed = tracklet.boxes[i]
            if label.box is None or box_iou(label.box, observed) > 0 or distance:
                contributing.append((tracklet, distance))
    if contributing:
        tracklet, distance = min(contributing, key=lambda item: item[1])
        how = "observed here" if distance == 0 else f"filled across a gap ({distance} frames from a detection)"
        return Diagnosis(
            frame, "over", "interpolated" if distance else "classified",
            f"track {tracklet.track_id} is flagged and {how}",
            track_id=tracklet.track_id,
            fix={"max_gap": max(0, distance - 1)} if distance else {},
            advice="mark the track 'keep' in the review page if this person should stay visible",
        )
    return Diagnosis(frame, "over", "smoothing", "the smoothed box extends past where the person was")


# --- render settings -----------------------------------------------------------------


@dataclass
class Evaluation:
    leaks_covered: int
    leaks_total: int
    overs_present: int
    overs_total: int
    area: int

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def evaluate_settings(
    meta: Metadata, labels: Labels, config: RenderConfig
) -> tuple[Evaluation, dict[int, list[PixelBox]]]:
    boxes = redaction_boxes(meta, config)
    leaks = labels.leaks()
    overs = labels.overs()
    return (
        Evaluation(
            leaks_covered=sum(label_covered(l, boxes.get(l.frame, ())) >= COVERED for l in leaks),
            leaks_total=len(leaks),
            overs_present=sum(label_covered(o, boxes.get(o.frame, ())) > 0 for o in overs),
            overs_total=len(overs),
            area=total_area(boxes),
        ),
        boxes,
    )


def settings_of(config: RenderConfig) -> dict[str, Any]:
    return {k: getattr(config, k) for k in RENDER_KNOBS}


@dataclass
class SettingsSuggestion:
    base: dict[str, Any]
    best: dict[str, Any]
    changes: dict[str, Any]
    base_eval: Evaluation
    best_eval: Evaluation
    area_change_pct: float
    tried: int
    diagnoses: list[Diagnosis]

    def flags(self) -> str:
        return " ".join(f"--{k.replace('_', '-')} {v}" for k, v in self.changes.items())

    def yaml(self) -> str:
        return "\n".join(f"{k}: {v}" for k, v in self.changes.items())

    def to_dict(self) -> dict[str, Any]:
        out = dataclasses.asdict(self)
        out["flags"], out["yaml"] = self.flags(), self.yaml()
        return out


def suggest_settings(
    meta: Metadata, labels: Labels, config: RenderConfig | None = None
) -> SettingsSuggestion:
    """Find render settings that fit the labeled leaks and over-redactions."""
    config = config if config is not None else RenderConfig()
    base_eval, base_boxes = evaluate_settings(meta, labels, config)

    diagnoses = [diagnose_leak(meta, l, config, base_boxes) for l in labels.leaks()]
    diagnoses += [diagnose_over(meta, o, config, base_boxes) for o in labels.overs()]
    leak_fixes = [d.fix for d in diagnoses if d.kind == "leak" and d.fix]
    over_fixes = [d.fix for d in diagnoses if d.kind == "over" and d.fix]

    def merge(fixes: list[dict[str, Any]]) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        for fix in fixes:
            for key, value in fix.items():
                if key in ("max_gap",):
                    merged[key] = max(merged.get(key, value), value)
                else:  # smaller is more permissive for the others
                    merged[key] = min(merged.get(key, value), value)
        return merged

    candidates: list[dict[str, Any]] = [{}]
    candidates += [dict(f) for f in leak_fixes]
    combined = merge(leak_fixes)
    if combined:
        candidates.append(combined)
        candidates += [{**combined, "smooth_window": w} for w in (1, 3, 5)
                       if w <= config.smooth_window]
    if over_fixes:
        candidates += [{**c, **merge(over_fixes)} for c in list(candidates)]

    seen, scored = set(), []
    for changes in candidates:
        changes = {k: v for k, v in changes.items() if getattr(config, k) != v}
        key = tuple(sorted(changes.items()))
        if key in seen:
            continue
        seen.add(key)
        trial = dataclasses.replace(config, **changes)
        evaluation = base_eval if not changes else evaluate_settings(meta, labels, trial)[0]
        rank = (-evaluation.leaks_covered, evaluation.overs_present, evaluation.area, len(changes))
        scored.append((rank, changes, evaluation))
    _, best_changes, best_eval = min(scored, key=lambda item: item[0])

    best = {**settings_of(config), **best_changes}
    growth = (best_eval.area - base_eval.area) / base_eval.area * 100 if base_eval.area else 0.0
    return SettingsSuggestion(
        base=settings_of(config), best=best, changes=best_changes,
        base_eval=base_eval, best_eval=best_eval,
        area_change_pct=round(growth, 2), tried=len(scored), diagnoses=diagnoses,
    )


# --- classifier threshold ------------------------------------------------------------


@dataclass
class ThresholdPoint:
    threshold: float
    tp: int  # labeled "redact", classifier says redact
    fp: int  # labeled "keep", classifier says redact (extra box)
    fn: int  # labeled "redact", classifier says keep (a leak)
    tn: int

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class ThresholdSuggestion:
    best: ThresholdPoint
    current: ThresholdPoint | None
    labeled: int
    positives: int
    points: list[ThresholdPoint]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _point(threshold: float, scored: list[tuple[float, bool]]) -> ThresholdPoint:
    tp = sum(s >= threshold and want for s, want in scored)
    fp = sum(s >= threshold and not want for s, want in scored)
    fn = sum(s < threshold and want for s, want in scored)
    tn = sum(s < threshold and not want for s, want in scored)
    return ThresholdPoint(round(threshold, 4), tp, fp, fn, tn)


def sweep_threshold(
    meta: Metadata, labels: Labels, current: float | None = None
) -> ThresholdSuggestion | None:
    """Best SigLIP threshold for the labeled tracks that have a score.

    Misses (``fn``) are weighed before extra boxes (``fp``). Returns None
    unless the labeled, scored tracks include both someone to redact and
    someone to keep: with one class every threshold on one side of the
    scores ties, so there is nothing to learn. When several thresholds tie,
    the middle of the tie is chosen: the most robust choice.
    """
    scored = [
        (t.score, labels.tracks[t.track_id].redact)
        for t in meta.tracklets
        if t.score is not None and t.track_id in labels.tracks
    ]
    if not any(want for _, want in scored) or all(want for _, want in scored):
        return None
    values = sorted({s for s, _ in scored})
    cuts = [0.0] + [(a + b) / 2 for a, b in zip(values, values[1:])] + [min(1.0, values[-1] + 1e-6)]
    points = [_point(c, scored) for c in cuts]
    top = min((p.fn, p.fp) for p in points)
    tied = [p for p in points if (p.fn, p.fp) == top]
    return ThresholdSuggestion(
        best=tied[len(tied) // 2],
        current=_point(current, scored) if current is not None else None,
        labeled=len(scored),
        positives=sum(want for _, want in scored),
        points=points,
    )


# --- trying a classifier on the labeled tracks ------------------------------------------


@dataclass
class ClassifierReport:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    undecided: int = 0
    mistakes: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def evaluate_classifier(
    meta: Metadata,
    labels: Labels,
    classifier: Any,
    prompt: str,
    on_undecided: str = "redact",
) -> ClassifierReport:
    """Run ``classifier`` on the labeled tracks' snapshots and score it."""
    report = ClassifierReport()
    for tracklet in meta.tracklets:
        label = labels.tracks.get(tracklet.track_id)
        if label is None or not tracklet.best_snapshot_path:
            continue
        decision = classifier.classify(Path(tracklet.best_snapshot_path), prompt)
        said = decision.redact
        if said is None:
            report.undecided += 1
            said = on_undecided == "redact"
        if said and label.redact:
            report.tp += 1
        elif said and not label.redact:
            report.fp += 1
        elif not said and label.redact:
            report.fn += 1
        else:
            report.tn += 1
        if said != label.redact:
            report.mistakes.append(
                {"track_id": tracklet.track_id, "said": said, "wanted": label.redact,
                 "detail": decision.detail}
            )
    return report


def analyze(
    meta: Metadata,
    labels: Labels,
    config: RenderConfig | None = None,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Everything the review page's "suggestions" panel shows."""
    settings = suggest_settings(meta, labels, config)
    sweep = sweep_threshold(meta, labels, threshold)
    return {
        "settings": settings.to_dict(),
        "threshold": sweep.to_dict() if sweep else None,
        "labels": {"tracks": len(labels.tracks), "leaks": len(labels.leaks()),
                   "overs": len(labels.overs())},
    }
