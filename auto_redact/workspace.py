"""Where a run keeps its working data.

Tracking results, thumbnails, the stage cache and (later) review labels are
working data, not output, so they do not belong next to the redacted video.
Each source video gets its own directory under the state directory::

    ~/.local/state/auto-redact/videos/<video-name>-<hash>/
        meta.json  meta.npz  meta.json.cache.json   tracklets + stage cache
        thumbs/                                     snapshots, contact sheet

The directory is keyed by the video's resolved path (not by ``--out``), so
redacting the same video to different outputs, or with different prompts,
shares its tracking. The root is ``$AUTO_REDACT_STATE_DIR`` if set, else
``$XDG_STATE_HOME/auto-redact``, else ``~/.local/state/auto-redact``.

``--meta`` and ``--thumbs`` still override the locations. Metadata written by
earlier versions next to the output (``<out stem>.meta.json``) keeps being
used in place, so existing work is neither orphaned nor moved.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


def state_root() -> Path:
    override = os.environ.get("AUTO_REDACT_STATE_DIR")
    if override:
        return Path(override)
    state = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(state) / "auto-redact"


def workspace_for(video: str | Path) -> Path:
    """The working directory for ``video`` (not created here)."""
    resolved = Path(video).resolve()
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", resolved.stem).strip("_")[:60] or "video"
    digest = hashlib.sha1(str(resolved).encode("utf-8")).hexdigest()[:8]
    return state_root() / "videos" / f"{slug}-{digest}"


LABELS_NAME = "labels.json"


def sibling_thumbs_dir(meta_file: Path) -> Path:
    return meta_file.with_name(meta_file.stem + "_thumbs")


def resolve_run_paths(
    video: str | Path,
    out: str | Path,
    meta_path: str | Path | None = None,
    thumbs_dir: str | Path | None = None,
    progress: Callable[[str], None] = lambda message: None,
) -> tuple[Path, Path]:
    """Decide the metadata file and thumbnail directory for a run."""
    if meta_path:
        meta_file = Path(meta_path)
        return meta_file, Path(thumbs_dir) if thumbs_dir else sibling_thumbs_dir(meta_file)

    out = Path(out)
    legacy = out.with_name(out.stem + ".meta.json")
    if legacy.is_file():
        progress(f"[workspace] using existing metadata next to the output: {legacy}")
        return legacy, Path(thumbs_dir) if thumbs_dir else sibling_thumbs_dir(legacy)

    workspace = workspace_for(video)
    progress(f"[workspace] {workspace}")
    return workspace / "meta.json", Path(thumbs_dir) if thumbs_dir else workspace / "thumbs"


def labels_path_for(meta_file: str | Path) -> Path:
    """Where review labels live for a metadata file (beside it)."""
    meta_file = Path(meta_file)
    if meta_file.name == "meta.json":  # a workspace
        return meta_file.with_name(LABELS_NAME)
    return meta_file.with_name(meta_file.stem + ".labels.json")


# --- inspecting and removing stored run data ---------------------------------

@dataclass
class Entry:
    """A directory of stored run data: a video workspace or a run log."""

    path: Path
    size: int
    modified: float
    video: str | None = None  # workspaces only
    video_exists: bool = True
    labeled: bool = False  # holds review labels (hand-made work)

    @property
    def name(self) -> str:
        return self.path.name


def _tree_size(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            pass
    return total


def _video_of(workspace: Path) -> str | None:
    try:
        data = json.loads((workspace / "meta.json").read_text(encoding="utf-8"))
        return str(data["video"]["path"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def list_workspaces() -> list[Entry]:
    """Every per-video workspace, oldest first."""
    root = state_root() / "videos"
    if not root.is_dir():
        return []
    entries = []
    for path in sorted(p for p in root.iterdir() if p.is_dir()):
        video = _video_of(path)
        entries.append(
            Entry(
                path=path,
                size=_tree_size(path),
                modified=path.stat().st_mtime,
                video=video,
                video_exists=video is not None and Path(video).exists(),
                labeled=(path / LABELS_NAME).is_file(),
            )
        )
    return sorted(entries, key=lambda e: e.modified)


def list_runs() -> list[Entry]:
    """Every run log directory (see :mod:`auto_redact.telemetry`), oldest first."""
    from . import telemetry

    root = telemetry.default_root()
    if not root.is_dir():
        return []
    entries = [
        Entry(path=p, size=_tree_size(p), modified=p.stat().st_mtime)
        for p in root.iterdir()
        if p.is_dir()
    ]
    return sorted(entries, key=lambda e: e.modified)


def remove_entry(entry: Entry) -> None:
    """Delete ``entry``, refusing anything outside the state directories."""
    from . import telemetry

    allowed = (state_root() / "videos").resolve(), telemetry.default_root().resolve()
    target = entry.path.resolve()
    if not any(root in target.parents for root in allowed):
        raise ValueError(f"refusing to remove {target}: not inside the auto-redact state directories")
    shutil.rmtree(target)


def human_size(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size} B"  # pragma: no cover


def select_for_clean(
    workspaces: list[Entry],
    *,
    everything: bool = False,
    missing: bool = False,
    video: str | Path | None = None,
    older_than_days: float | None = None,
    include_labeled: bool = False,
    now: float | None = None,
) -> tuple[list[Entry], list[Entry]]:
    """Pick workspaces to delete. Returns ``(selected, skipped_labeled)``.

    ``everything``, ``missing`` (the source video is gone or unknown) and
    ``video`` each select workspaces; ``older_than_days`` narrows whatever
    was selected, or selects by age alone when nothing else was asked for.
    Workspaces with review labels are hand-made work, so they are skipped
    unless ``include_labeled``.
    """
    import time

    now = time.time() if now is None else now
    wanted = None
    if video is not None:
        target = workspace_for(video)
        wanted = [w for w in workspaces if w.path == target]
    if missing:
        wanted = (wanted or []) + [
            w for w in workspaces if not w.video_exists and w not in (wanted or [])
        ]
    if everything:
        wanted = list(workspaces)
    if older_than_days is not None:
        cutoff = now - older_than_days * 86400
        wanted = [w for w in (workspaces if wanted is None else wanted) if w.modified <= cutoff]
    wanted = wanted or []
    skipped = [w for w in wanted if w.labeled and not include_labeled]
    return [w for w in wanted if w not in skipped], skipped


def select_runs(runs: list[Entry], older_than_days: float | None, now: float | None = None) -> list[Entry]:
    import time

    if older_than_days is None:
        return list(runs)
    cutoff = (time.time() if now is None else now) - older_than_days * 86400
    return [r for r in runs if r.modified <= cutoff]
