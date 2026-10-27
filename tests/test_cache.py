"""Stage keys: what invalidates which stage."""

from auto_redact import cache
from auto_redact.config import ClassifyConfig, SnapshotConfig, TrackConfig


def make_keys(tmp_path, track=TrackConfig(), snapshot=SnapshotConfig(),
              classify=ClassifyConfig(), prompt="red shirt"):
    video = tmp_path / "v.mp4"
    if not video.exists():
        video.write_bytes(b"x")
    t = cache.track_key(video, track)
    s = cache.snapshot_key(t, snapshot, tmp_path / "thumbs")
    c = cache.classify_key(s, prompt, classify)
    return t, s, c


def test_keys_are_stable(tmp_path):
    assert make_keys(tmp_path) == make_keys(tmp_path)


def test_track_setting_invalidates_every_stage(tmp_path):
    base = make_keys(tmp_path)
    changed = make_keys(tmp_path, track=TrackConfig(stride=2))
    assert all(a != b for a, b in zip(base, changed))


def test_snapshot_setting_keeps_tracking(tmp_path):
    t0, s0, c0 = make_keys(tmp_path)
    t1, s1, c1 = make_keys(tmp_path, snapshot=SnapshotConfig(padding=0.5))
    assert t0 == t1 and s0 != s1 and c0 != c1


def test_prompt_only_invalidates_classification(tmp_path):
    t0, s0, c0 = make_keys(tmp_path)
    t1, s1, c1 = make_keys(tmp_path, prompt="blue hat")
    assert (t0, s0) == (t1, s1) and c0 != c1


def test_prompt_whitespace_is_ignored(tmp_path):
    assert make_keys(tmp_path)[2] == make_keys(tmp_path, prompt="  red shirt\n")[2]


def test_execution_only_settings_do_not_matter(tmp_path):
    base = make_keys(tmp_path)
    same = make_keys(
        tmp_path,
        track=TrackConfig(device="cpu", half=False),
        classify=ClassifyConfig(workers=4, timeout=5.0, keep_alive="1m"),
    )
    assert base == same


def test_threshold_matters_for_siglip_only(tmp_path):
    vlm_a = make_keys(tmp_path, classify=ClassifyConfig(threshold=0.2))[2]
    vlm_b = make_keys(tmp_path, classify=ClassifyConfig(threshold=0.9))[2]
    assert vlm_a == vlm_b
    sig_a = make_keys(tmp_path, classify=ClassifyConfig(classifier="siglip", threshold=0.2))[2]
    sig_b = make_keys(tmp_path, classify=ClassifyConfig(classifier="siglip", threshold=0.9))[2]
    assert sig_a != sig_b


def test_vlm_prefilter_and_image_size_matter(tmp_path):
    base = make_keys(tmp_path)[2]
    assert make_keys(tmp_path, classify=ClassifyConfig(prefilter="negatives"))[2] != base
    assert make_keys(tmp_path, classify=ClassifyConfig(max_image_size=512))[2] != base


def test_video_change_invalidates_tracking(tmp_path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    before = cache.track_key(video, TrackConfig())
    video.write_bytes(b"xy")
    assert cache.track_key(video, TrackConfig()) != before


def test_manifest_records_and_invalidates_downstream(tmp_path):
    meta = tmp_path / "m.json"
    stages = cache.StageCache(meta)
    for name in cache.STAGES:
        stages.record(name, name + "-key")
    again = cache.StageCache(meta)
    assert again.matches("classify", "classify-key")
    again.invalidate("snapshot")
    assert again.matches("track", "track-key")
    assert not again.matches("snapshot", "snapshot-key")
    assert not cache.StageCache(meta).matches("classify", "classify-key")


def test_begin_reports_whether_an_interrupted_attempt_can_resume(tmp_path):
    meta = tmp_path / "m.json"
    stages = cache.StageCache(meta)
    assert stages.begin("snapshot", "k1") is False  # fresh
    # ... crash; a new process with the same settings may resume
    assert cache.StageCache(meta).begin("snapshot", "k1") is True
    # different settings must not pick up the old checkpoint
    assert cache.StageCache(meta).begin("snapshot", "k2") is False
    stages = cache.StageCache(meta)
    stages.record("snapshot", "k2")
    assert stages.begin("snapshot", "k3") is False


def test_corrupt_manifest_is_treated_as_empty(tmp_path):
    meta = tmp_path / "m.json"
    (tmp_path / "m.json.cache.json").write_text("{nope")
    assert cache.StageCache(meta).stages == {}


def test_undecided_policy_changes_classification_results(tmp_path):
    base = make_keys(tmp_path)[2]
    assert make_keys(tmp_path, classify=ClassifyConfig(on_undecided="keep"))[2] != base
