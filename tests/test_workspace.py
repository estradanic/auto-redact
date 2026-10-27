from auto_redact.workspace import resolve_run_paths, state_root, workspace_for


def test_state_root_honours_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTO_REDACT_STATE_DIR", str(tmp_path / "a"))
    assert state_root() == tmp_path / "a"
    monkeypatch.delenv("AUTO_REDACT_STATE_DIR")
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg"))
    assert state_root() == tmp_path / "xdg" / "auto-redact"


def test_workspace_is_keyed_by_the_videos_path(tmp_path):
    a, b = tmp_path / "x" / "Show S01E01.mkv", tmp_path / "y" / "Show S01E01.mkv"
    assert workspace_for(a) == workspace_for(a)
    assert workspace_for(a) != workspace_for(b)  # same name, different file
    assert workspace_for(a).name.startswith("Show_S01E01.mkv".rsplit(".", 1)[0])
    assert workspace_for(a).parent.name == "videos"


def test_workspace_name_is_filesystem_safe(tmp_path):
    name = workspace_for(tmp_path / "we/ird: name?*.mp4").name
    assert "/" not in name and ":" not in name and "?" not in name


def test_default_paths_live_in_the_workspace(tmp_path):
    meta, thumbs = resolve_run_paths(tmp_path / "v.mp4", tmp_path / "out.mp4")
    assert meta == workspace_for(tmp_path / "v.mp4") / "meta.json"
    assert thumbs == meta.parent / "thumbs"


def test_explicit_paths_override(tmp_path):
    meta, thumbs = resolve_run_paths(
        tmp_path / "v.mp4", tmp_path / "out.mp4", tmp_path / "m.json", tmp_path / "t"
    )
    assert (meta, thumbs) == (tmp_path / "m.json", tmp_path / "t")
    _, thumbs = resolve_run_paths(tmp_path / "v.mp4", tmp_path / "out.mp4", tmp_path / "m.json")
    assert thumbs == tmp_path / "m_thumbs"
