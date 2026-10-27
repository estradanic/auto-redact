"""Stage cache: which settings produced the data in a metadata file.

Each run (or standalone stage command) records, next to the metadata, a
fingerprint of everything that influenced that stage's output::

    <meta>.cache.json  ->  {"version": 1, "stages": {"track": "...", ...}}

A fingerprint is a hash chained through the stages: a stage's key covers
its own result-affecting settings *and* its upstream stage's key, so
changing any tracking setting also invalidates snapshots and decisions,
while changing only the prompt invalidates just the classification. Keys
use :func:`auto_redact.config.cache_fields`, so execution-only settings
(devices, worker counts, timeouts) never cause a re-run.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .config import ClassifyConfig, SnapshotConfig, TrackConfig, cache_fields

STAGES = ("track", "snapshot", "classify")
_VERSION = 1


def _digest(*parts: Any) -> str:
    text = json.dumps(parts, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def video_fingerprint(video: str | Path) -> list[Any]:
    """Identify a video by resolved path, size and modification time."""
    path = Path(video).resolve()
    try:
        stat = path.stat()
    except OSError:
        return [str(path)]
    return [str(path), stat.st_size, stat.st_mtime_ns]


def track_key(video: str | Path, config: TrackConfig) -> str:
    return _digest("track", video_fingerprint(video), cache_fields(config))


def snapshot_key(
    upstream: str, config: SnapshotConfig, thumbs_dir: str | Path
) -> str:
    return _digest(
        "snapshot", upstream, cache_fields(config), str(Path(thumbs_dir).resolve())
    )


def classify_key(upstream: str, prompt: str, config: ClassifyConfig) -> str:
    """Only the settings that matter for the chosen backend are included."""
    fields: dict[str, Any] = {
        "classifier": config.classifier,
        "model": config.resolved_model,
        "on_undecided": config.on_undecided,
    }
    if config.classifier == "siglip":
        fields["threshold"] = config.threshold
    else:
        fields["think"] = config.think
        fields["max_image_size"] = config.max_image_size
        fields["prefilter"] = config.prefilter
        if config.prefilter != "none":
            fields["prefilter_model"] = config.prefilter_model
            fields["prefilter_threshold"] = config.prefilter_threshold
    return _digest("classify", upstream, prompt.strip(), fields)


class StageCache:
    """Reads and writes the ``<meta>.cache.json`` manifest."""

    def __init__(self, meta_path: str | Path) -> None:
        meta_path = Path(meta_path)
        self.path = meta_path.with_name(meta_path.name + ".cache.json")
        self.stages: dict[str, str] = {}
        self.started: dict[str, str] = {}
        self._read()

    def _read(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(data, dict) or data.get("version") != _VERSION:
            return
        for name, target in (("stages", self.stages), ("started", self.started)):
            section = data.get(name)
            if isinstance(section, dict):
                target.update(
                    {k: v for k, v in section.items() if k in STAGES and isinstance(v, str)}
                )

    def matches(self, stage: str, key: str) -> bool:
        return self.stages.get(stage) == key

    def get(self, stage: str, default: str = "-") -> str:
        return self.stages.get(stage, default)

    def begin(self, stage: str, key: str) -> bool:
        """Mark ``stage`` as running with ``key``.

        Returns True when an earlier, interrupted attempt used the same key,
        i.e. its partial progress (checkpoints) may be resumed; otherwise
        any leftover checkpoints belong to other settings and must be
        ignored.
        """
        resuming = self.started.get(stage) == key
        self.invalidate(stage)
        self.started[stage] = key
        self._write()
        return resuming

    def record(self, stage: str, key: str) -> None:
        """Store ``key`` as the finished fingerprint of ``stage``."""
        self.started.pop(stage, None)
        self.stages[stage] = key
        self._write()

    def invalidate(self, stage: str) -> None:
        """Forget ``stage`` and everything after it."""
        for name in STAGES[STAGES.index(stage) :]:
            self.stages.pop(name, None)
        self._write()

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {"version": _VERSION, "stages": self.stages, "started": self.started},
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )


def adopt_legacy(
    stages: StageCache,
    meta_path: str | Path,
    meta: Any,
    keys: dict[str, str],
    prompt: str,
    config: ClassifyConfig,
) -> list[str]:
    """Adopt results written before manifests existed.

    Earlier versions trusted whatever was in the metadata file (and a
    ``<meta>.classify.json`` sidecar for the classification). Throwing that
    away would cost a lot of tracking time, so a metadata file with no
    manifest is taken as made with the current settings, once; every run
    after that is checked. Returns the stages adopted.
    """
    adopted = ["track"]
    stages.record("track", keys["track"])
    if meta.tracklets and all(
        t.best_snapshot_path is not None and Path(t.best_snapshot_path).is_file()
        for t in meta.tracklets
    ):
        stages.record("snapshot", keys["snapshot"])
        adopted.append("snapshot")
        sidecar = Path(meta_path)
        sidecar = sidecar.with_name(sidecar.name + ".classify.json")
        try:
            stored = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            stored = None
        if stored == _legacy_classify_key(prompt, config) and all(
            t.redact is not None for t in meta.tracklets
        ):
            stages.record("classify", keys["classify"])
            adopted.append("classify")
    return adopted


def _legacy_classify_key(prompt: str, config: ClassifyConfig) -> dict[str, Any]:
    key: dict[str, Any] = {"prompt": prompt.strip(), "model": config.resolved_model}
    if config.classifier == "siglip":
        key.update(classifier="siglip", threshold=config.threshold)
    elif config.think:
        key["think"] = True
    return key
