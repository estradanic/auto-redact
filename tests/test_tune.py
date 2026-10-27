"""The tuner is pure: synthetic metadata in, diagnoses and suggestions out."""

import pytest

from auto_redact.config import RenderConfig
from auto_redact.labels import Labels
from auto_redact.meta import Metadata, Tracklet, VideoInfo
from auto_redact.tune import (
    analyze,
    covered_fraction,
    diagnose_leak,
    diagnose_over,
    evaluate_classifier,
    suggest_settings,
    sweep_threshold,
)

NO_SMOOTH = RenderConfig(smooth_window=1, bridge_iou=0.0, min_track_frames=1, max_gap=5)


def make_meta(*tracklets):
    return Metadata(VideoInfo("v.mp4", 25.0, 1000, 500, 400), list(tracklets))


def track(tid, frames, box=(100, 100, 140, 200), redact=True, score=None, dx=0):
    return Tracklet(
        tid, list(frames),
        [[box[0] + dx * i, box[1], box[2] + dx * i, box[3]] for i, _ in enumerate(frames)],
        [[] for _ in frames], redact=redact, score=score,
    )


def leak(frame, box=(100, 100, 140, 200)):
    labels = Labels()
    labels.add_frame(frame, "leak", list(box))
    return labels.frames[0]


# --- coverage -----------------------------------------------------------------


def test_covered_fraction_uses_the_union_of_boxes():
    box = [0, 0, 10, 10]
    assert covered_fraction(box, []) == 0.0
    assert covered_fraction(box, [(0, 0, 10, 10)]) == 1.0
    assert covered_fraction(box, [(0, 0, 5, 10), (5, 0, 10, 10)]) == 1.0
    assert covered_fraction(box, [(0, 0, 5, 10), (0, 0, 5, 10)]) == 0.5
    assert covered_fraction(box, [(50, 50, 60, 60)]) == 0.0


# --- diagnosing leaks ---------------------------------------------------------------


def test_a_covered_frame_is_not_a_leak():
    meta = make_meta(track(1, range(10)))
    assert diagnose_leak(meta, leak(5), NO_SMOOTH).cause == "covered"


def test_nobody_near_means_the_detector_missed_them():
    meta = make_meta(track(1, range(10), box=(800, 100, 840, 200)))
    d = diagnose_leak(meta, leak(5), NO_SMOOTH)
    assert d.cause == "not_detected" and "track-conf" in d.advice and not d.fix


def test_a_tracklet_far_away_in_time_does_not_count():
    meta = make_meta(track(1, range(300, 310)))
    assert diagnose_leak(meta, leak(5), NO_SMOOTH).cause == "not_detected"


def test_unflagged_track_points_at_classification_and_its_score():
    meta = make_meta(track(7, range(10), redact=False, score=0.31))
    d = diagnose_leak(meta, leak(5), NO_SMOOTH)
    assert (d.cause, d.track_id) == ("not_flagged", 7)
    assert "0.31" in d.message and "0.31" in d.advice


def test_undecided_track_is_also_not_flagged():
    meta = make_meta(track(7, range(10), redact=None))
    assert diagnose_leak(meta, leak(5), NO_SMOOTH).cause == "not_flagged"


def test_a_human_override_counts_as_flagged():
    t = track(7, range(10), redact=False)
    t.override = True
    assert diagnose_leak(make_meta(t), leak(5), NO_SMOOTH).cause == "covered"


def test_short_flagged_track_is_dropped_by_min_track_frames():
    meta = make_meta(track(2, range(3)))
    config = RenderConfig(smooth_window=1, bridge_iou=0.0, max_gap=5, min_track_frames=5)
    d = diagnose_leak(meta, leak(1), config)
    assert (d.cause, d.fix) == ("dropped_short", {"min_track_frames": 3})


def test_detection_gap_wider_than_max_gap():
    meta = make_meta(track(3, [*range(0, 10), *range(30, 40)]))
    d = diagnose_leak(meta, leak(20), NO_SMOOTH)
    assert (d.cause, d.fix) == ("gap", {"max_gap": 20})


def test_before_or_after_a_track_no_setting_helps():
    meta = make_meta(track(3, range(50, 60)))
    d = diagnose_leak(meta, leak(30), NO_SMOOTH)
    assert d.cause == "track_edge" and not d.fix and "re-track" in d.advice


def test_smoothing_can_lag_a_fast_mover():
    fast = track(4, range(0, 20), dx=20)
    config = RenderConfig(smooth_window=9, bridge_iou=0.0, min_track_frames=1, max_gap=5)
    d = diagnose_leak(make_meta(fast), leak(0, box=(100, 100, 140, 200)), config)
    assert (d.cause, d.fix) == ("box_lag", {"smooth_window": 1})


# --- diagnosing over-redaction ---------------------------------------------------------


def over(frame, box=None):
    labels = Labels()
    labels.add_frame(frame, "over", box)
    return labels.frames[0]


def test_over_clear_when_nothing_is_painted():
    assert diagnose_over(make_meta(track(1, range(5))), over(100), NO_SMOOTH).cause == "clear"


def test_over_from_an_observed_flagged_track():
    d = diagnose_over(make_meta(track(1, range(10))), over(5), NO_SMOOTH)
    assert (d.cause, d.track_id) == ("classified", 1)
    assert "keep" in d.advice


def test_over_from_interpolation_across_a_gap():
    meta = make_meta(track(1, [*range(0, 5), *range(10, 15)]))
    d = diagnose_over(meta, over(7), NO_SMOOTH)
    assert d.cause == "interpolated" and "max_gap" in d.fix


def test_over_from_bridging_two_tracklets():
    a, b = track(1, range(0, 10)), track(2, range(14, 24))
    bridging = RenderConfig(smooth_window=1, bridge_iou=0.3, min_track_frames=1, max_gap=5)
    d = diagnose_over(make_meta(a, b), over(11), bridging)
    assert (d.cause, d.fix) == ("bridged", {"bridge_iou": 0.0})


# --- suggesting render settings ------------------------------------------------------------


def test_suggestion_widens_max_gap_to_cover_a_labeled_leak():
    meta = make_meta(track(3, [*range(0, 10), *range(30, 40)]))
    labels = Labels()
    labels.add_frame(20, "leak", [100, 100, 140, 200])
    s = suggest_settings(meta, labels, NO_SMOOTH)
    assert s.changes == {"max_gap": 20}
    assert (s.base_eval.leaks_covered, s.best_eval.leaks_covered) == (0, 1)
    assert s.area_change_pct > 0
    assert s.flags() == "--max-gap 20" and s.yaml() == "max_gap: 20"


def test_suggestion_lowers_min_track_frames():
    meta = make_meta(track(2, range(3)))
    labels = Labels()
    labels.add_frame(1, "leak", [100, 100, 140, 200])
    config = RenderConfig(smooth_window=1, bridge_iou=0.0, max_gap=5, min_track_frames=5)
    s = suggest_settings(meta, labels, config)
    assert s.changes == {"min_track_frames": 3} and s.best_eval.leaks_covered == 1


def test_suggestion_leaves_alone_what_it_cannot_fix():
    meta = make_meta(track(3, range(50, 60)))
    labels = Labels()
    labels.add_frame(30, "leak", [100, 100, 140, 200])
    s = suggest_settings(meta, labels, NO_SMOOTH)
    assert s.changes == {} and s.best_eval.leaks_covered == 0
    assert s.diagnoses[0].cause == "track_edge"


def test_suggestion_with_no_labels_changes_nothing():
    s = suggest_settings(make_meta(track(1, range(10))), Labels(), NO_SMOOTH)
    assert s.changes == {} and s.tried == 1


def test_suggestion_clears_an_over_redaction_by_dropping_bridging():
    a, b = track(1, range(0, 10)), track(2, range(14, 24))
    bridging = RenderConfig(smooth_window=1, bridge_iou=0.3, min_track_frames=1, max_gap=5)
    labels = Labels()
    labels.add_frame(11, "over")
    s = suggest_settings(make_meta(a, b), labels, bridging)
    assert s.changes == {"bridge_iou": 0.0}
    assert (s.base_eval.overs_present, s.best_eval.overs_present) == (1, 0)


def test_covering_leaks_beats_avoiding_black_area():
    meta = make_meta(track(3, [*range(0, 10), *range(30, 40)]))
    labels = Labels()
    labels.add_frame(20, "leak", [100, 100, 140, 200])
    labels.add_frame(21, "over")  # contradictory: same gap
    s = suggest_settings(meta, labels, NO_SMOOTH)
    assert s.best_eval.leaks_covered == 1


# --- threshold ----------------------------------------------------------------------------------


def scored_meta(scores):
    return make_meta(*[track(i, range(5), redact=None, score=s) for i, s in scores.items()])


def test_threshold_sweep_catches_every_wanted_person_with_the_fewest_extras():
    meta = scored_meta({1: 0.9, 2: 0.6, 3: 0.4, 4: 0.3, 5: 0.2, 6: 0.55})
    labels = Labels()
    for tid, want in {1: True, 2: True, 3: True, 4: False, 5: False, 6: False}.items():
        labels.set_track(tid, want)
    s = sweep_threshold(meta, labels, current=0.5)
    assert (s.best.fn, s.best.fp) == (0, 1)
    assert 0.3 < s.best.threshold <= 0.4
    assert (s.current.fn, s.current.fp) == (1, 1)  # today's 0.5 misses track 3 AND boxes track 6
    assert (s.labeled, s.positives) == (6, 3)


def test_threshold_prefers_recall_over_extras():
    meta = scored_meta({1: 0.2, 2: 0.8, 3: 0.7})
    labels = Labels()
    for tid, want in {1: True, 2: False, 3: False}.items():
        labels.set_track(tid, want)
    s = sweep_threshold(meta, labels)
    assert s.best.fn == 0 and s.best.fp == 2  # catching the 0.2 person costs two extras


def test_threshold_ties_pick_the_middle():
    meta = scored_meta({1: 0.9, 2: 0.1})
    labels = Labels()
    labels.set_track(1, True)
    labels.set_track(2, False)
    assert sweep_threshold(meta, labels).best.threshold == pytest.approx(0.5)


def test_threshold_needs_both_classes_labeled():
    meta = scored_meta({1: 0.9, 2: 0.1})
    only_keep, only_redact = Labels(), Labels()
    only_keep.set_track(1, False)
    only_keep.set_track(2, False)
    only_redact.set_track(1, True)
    only_redact.set_track(2, True)
    assert sweep_threshold(meta, only_keep) is None
    assert sweep_threshold(meta, only_redact) is None


def test_threshold_needs_scored_labeled_tracks():
    meta = scored_meta({1: 0.9})
    assert sweep_threshold(meta, Labels()) is None
    meta.tracklets[0].score = None
    labels = Labels()
    labels.set_track(1, True)
    assert sweep_threshold(meta, labels) is None


# --- trying a classifier -----------------------------------------------------------------------------


def test_evaluate_classifier_on_labeled_tracks_only(tmp_path):
    from auto_redact.classify import Decision

    tracklets = []
    for tid in (1, 2, 3, 4):
        t = track(tid, range(3))
        t.best_snapshot_path = str(tmp_path / f"t{tid}.jpg")
        tracklets.append(t)
    meta = make_meta(*tracklets)
    labels = Labels()
    for tid, want in {1: True, 2: True, 3: False}.items():  # track 4 unlabeled
        labels.set_track(tid, want)

    answers = {"t1.jpg": True, "t2.jpg": False, "t3.jpg": None, "t4.jpg": True}

    class Fake:
        def classify(self, path, prompt):
            return Decision(answers[path.name], None, "fake")

    r = evaluate_classifier(meta, labels, Fake(), "p")
    assert (r.tp, r.fn, r.fp, r.tn, r.undecided) == (1, 1, 1, 0, 1)  # undecided redacts
    assert {m["track_id"] for m in r.mistakes} == {2, 3}
    keep = evaluate_classifier(meta, labels, Fake(), "p", on_undecided="keep")
    assert (keep.fp, keep.tn) == (0, 1)


def test_analyze_bundles_everything_the_page_needs():
    meta = make_meta(track(1, range(10), score=0.7), track(2, range(10), box=(500, 100, 540, 200), score=0.2))
    labels = Labels()
    labels.set_track(1, True)
    labels.set_track(2, False)
    labels.add_frame(5, "leak", [100, 100, 140, 200])
    out = analyze(meta, labels, NO_SMOOTH, threshold=0.5)
    assert out["labels"] == {"tracks": 2, "leaks": 1, "overs": 0}
    assert out["settings"]["best_eval"]["leaks_total"] == 1
    assert out["threshold"]["best"]["fn"] == 0
