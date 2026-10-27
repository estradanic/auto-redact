import pytest

from auto_redact.labels import Labels, LabelsError, load_labels, save_labels


def test_missing_file_is_empty(tmp_path):
    assert load_labels(tmp_path / "labels.json") == Labels()


def test_round_trip(tmp_path):
    labels = Labels(track_key="abc")
    labels.set_track(3, True, "red shirt")
    labels.set_track(5, False)
    labels.add_frame(120, "leak", [10, 20, 60, 200], "shoulder visible")
    labels.add_frame(300, "over")
    path = save_labels(tmp_path / "w" / "labels.json", labels)
    assert load_labels(path) == labels


def test_set_track_none_removes_the_verdict():
    labels = Labels()
    labels.set_track(1, True)
    labels.set_track(1, None)
    assert labels.tracks == {}


def test_leaks_and_overs_are_separated():
    labels = Labels()
    labels.add_frame(1, "leak")
    labels.add_frame(2, "over")
    assert [f.frame for f in labels.leaks()] == [1]
    assert [f.frame for f in labels.overs()] == [2]


def test_labels_for_other_tracking_are_stale():
    assert Labels(track_key="a").stale("b")
    assert not Labels(track_key="a").stale("a")
    assert not Labels().stale("b")  # nothing recorded: nothing to disagree with
    assert not Labels(track_key="a").stale(None)


@pytest.mark.parametrize(
    "call, message",
    [
        (lambda l: l.add_frame(1, "bogus"), "kind must be one of"),
        (lambda l: l.add_frame(1, "leak", [1, 2, 3]), "box must be"),
        (lambda l: l.add_frame(1, "leak", [5, 5, 5, 9]), "positive size"),
        (lambda l: l.add_frame(-1, "leak"), ">= 0"),
        (lambda l: l.remove_frame(0), "no frame label"),
    ],
)
def test_invalid_edits_are_rejected(call, message):
    with pytest.raises(LabelsError, match=message):
        call(Labels())


def test_invalid_files_are_rejected(tmp_path):
    path = tmp_path / "labels.json"
    for text in ("{nope", '{"version": 2}', '{"version": 1, "extra": 1}',
                 '{"version": 1, "frames": [{"frame": 1, "kind": "huh"}]}',
                 '{"version": 1, "tracks": {"x": {"redact": true}}}'):
        path.write_text(text)
        with pytest.raises(LabelsError):
            load_labels(path)
