"""`auto-redact clean`: listing and deleting stored run data."""

import json
import os
import time

import pytest

from auto_redact import workspace
from auto_redact.cli import main


def make_workspace(video, age_days=0, labeled=False, size=100, with_meta=True):
    ws = workspace.workspace_for(video)
    ws.mkdir(parents=True)
    if with_meta:
        (ws / "meta.json").write_text(json.dumps({"video": {"path": str(video)}}))
    (ws / "blob.bin").write_bytes(b"x" * size)
    if labeled:
        (ws / workspace.LABELS_NAME).write_text("{}")
    old = time.time() - age_days * 86400
    os.utime(ws, (old, old))
    return ws


@pytest.fixture
def videos(tmp_path):
    present = tmp_path / "present.mkv"
    present.write_bytes(b"v")
    return present, tmp_path / "gone.mkv", tmp_path / "labeled.mkv"


def run_clean(*args):
    return main(["clean", "--no-stats", *args])


def test_listing_shows_workspaces_and_changes_nothing(videos, capsys):
    present, gone, _ = videos
    a, b = make_workspace(present), make_workspace(gone)
    assert run_clean() == 0
    out = capsys.readouterr().out
    assert present.name in out and gone.name in out
    assert "[video missing]" in out
    assert a.exists() and b.exists()


def test_missing_removes_only_workspaces_whose_video_is_gone(videos, capsys):
    present, gone, _ = videos
    keep, drop = make_workspace(present), make_workspace(gone)
    assert run_clean("--missing", "--yes") == 0
    assert keep.exists() and not drop.exists()
    assert "deleted 1 entry" in capsys.readouterr().out


def test_dry_run_deletes_nothing(videos, capsys):
    present, gone, _ = videos
    ws = make_workspace(gone)
    assert run_clean("--all", "--dry-run") == 0
    assert ws.exists() and "would delete" in capsys.readouterr().out


def test_without_yes_a_non_terminal_refuses(videos, capsys):
    present, *_ = videos
    ws = make_workspace(present)
    assert run_clean("--all") == 1  # pytest's stdin is not a tty
    assert ws.exists() and "--yes" in capsys.readouterr().err


def test_labeled_workspaces_are_protected_unless_asked(videos, capsys):
    _, _, labeled = videos
    ws = make_workspace(labeled, labeled=True)
    run_clean("--all", "--yes")
    assert ws.exists()
    assert "skipping (has review labels" in capsys.readouterr().out
    run_clean("--all", "--yes", "--labeled")
    assert not ws.exists()


def test_older_than_alone_selects_by_age(videos):
    present, gone, _ = videos
    old, new = make_workspace(gone, age_days=40), make_workspace(present, age_days=1)
    run_clean("--older-than", "30", "--yes")
    assert not old.exists() and new.exists()


def test_older_than_narrows_other_selectors(videos):
    present, gone, _ = videos
    old_gone = make_workspace(gone, age_days=40)
    new_present = make_workspace(present, age_days=1)
    run_clean("--all", "--older-than", "30", "--yes")
    assert not old_gone.exists() and new_present.exists()


def test_video_selects_that_videos_workspace(videos):
    present, gone, _ = videos
    a, b = make_workspace(present), make_workspace(gone)
    run_clean("--video", str(gone), "--yes")
    assert a.exists() and not b.exists()


def test_runs_are_cleaned_only_when_asked(tmp_path, monkeypatch, videos):
    from auto_redact import telemetry

    root = tmp_path / "runs"
    monkeypatch.setenv("AUTO_REDACT_STATS_DIR", str(root))
    old, new = root / "2025-old-run", root / "2026-new-run"
    for d, age in ((old, 60), (new, 1)):
        d.mkdir(parents=True)
        (d / "run.log").write_text("log")
        t = time.time() - age * 86400
        os.utime(d, (t, t))
    present, *_ = videos
    ws = make_workspace(present)
    assert telemetry.default_root() == root

    run_clean("--all", "--older-than", "30", "--yes")  # no --runs: logs untouched
    assert old.exists() and ws.exists()
    run_clean("--runs", "--older-than", "30", "--yes")
    assert not old.exists() and new.exists() and ws.exists()
    run_clean("--runs", "--yes")
    assert not new.exists()


def test_remove_refuses_paths_outside_the_state_directories(tmp_path):
    outside = tmp_path / "precious"
    outside.mkdir()
    with pytest.raises(ValueError, match="refusing"):
        workspace.remove_entry(workspace.Entry(path=outside, size=0, modified=0))
    assert outside.exists()


def test_nothing_to_delete_is_not_an_error(capsys):
    assert run_clean("--missing", "--yes") == 0
    assert "nothing to delete" in capsys.readouterr().out
