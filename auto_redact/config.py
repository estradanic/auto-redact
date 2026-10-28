"""Typed configuration: the single source of truth for every tunable.

Each pipeline stage has one frozen dataclass. Field defaults here are the
only defaults in the code base: stage functions take a config object, and
the command line is generated from these fields (see :mod:`.cli`).

Field metadata drives that generation and the stage cache:

``help``       argparse help text (a ``%(default)s`` placeholder is allowed)
``flag``       primary command-line flag (default: ``--<field-name>``)
``aliases``    extra flags accepted for the same option
``metavar``    argparse metavar
``choices``    allowed values
``cli``        ``False`` hides the field from the command line
``advanced``   internal tuning: still accepted as a flag and in config files,
               and settable with ``--set name=value``, but left out of
               ``--help``
``cache``      ``False`` marks execution-only settings (device, workers,
               timeouts, ...) that cannot change a stage's *result*, so they
               are left out of the stage cache key (see :mod:`.cache`)
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields
from typing import Any, Mapping, TypeVar

ENCODERS = ("libx264", "h264_nvenc")
CLASSIFIERS = ("vlm", "siglip")
PREFILTERS = ("none", "negatives", "positives")
UNDECIDED_POLICIES = ("redact", "keep", "error")

DEFAULT_VLM_MODEL = "gemma3:4b"
DEFAULT_SIGLIP_MODEL = "google/siglip2-base-patch16-224"


def _opt(default: Any, help: str = "", **meta: Any) -> Any:
    return field(default=default, metadata={"help": help, **meta})


@dataclass(frozen=True)
class TrackConfig:
    """Detection, association and scene-cut splitting."""

    stride: int = _opt(
        1,
        "Process every Nth frame (default: %(default)s, i.e. every frame)",
        flag="--stride",
        aliases=("--sample-rate",),
        metavar="N",
    )
    device: str = _opt(
        "auto",
        'Device for detection/tracking: "auto" (GPU if available, else CPU) '
        'or a CUDA index like "0"',
        cache=False,
    )
    half: bool = _opt(
        True, "FP16 tracking inference (CUDA only; default: on)", cache=False
    )
    conf: float = _opt(
        0.25,
        "Detection confidence threshold for tracking (default: %(default)s)",
        flag="--track-conf",
        metavar="F",
    )
    iou: float = _opt(
        0.7,
        "NMS IoU threshold for tracking; lower values split overlapping "
        "people (default: %(default)s)",
        flag="--track-iou",
        metavar="F",
    )
    imgsz: int | None = _opt(None, "Detector input size", cli=False)
    tracker: str = _opt(
        "bytetrack.yaml",
        "Ultralytics tracker config (botsort.yaml or bytetrack.yaml; "
        "default: %(default)s)",
        flag="--track-tracker",
        metavar="NAME",
    )
    orphans: bool = _opt(
        False,
        "After tracking, add tracklets for people detected but never "
        "tracked (extra pass)",
    )
    scene_cuts: bool = _opt(
        True,
        "Detect hard scene cuts and split tracklets that span them, so a "
        "person is not joined across an edit",
    )
    scene_cut_threshold: float = _opt(
        0.45,
        "Bhattacharyya distance threshold for a hard cut (default: %(default)s)",
        metavar="F",
    )


@dataclass(frozen=True)
class SnapshotConfig:
    """Best-view selection and bystander isolation."""

    padding: float = _opt(
        0.2,
        "Crop margin around each bbox, per side, as a fraction of its size "
        "(default: %(default)s)",
        metavar="F",
    )
    isolate: bool = _opt(
        True,
        "Black out other people who appear in a thumbnail so the classifier "
        "can only judge the tracklet's own subject",
    )
    sub_isolate: bool = _opt(
        False,
        "Also run a detector on each thumbnail and black out extra people "
        "the tracker merged into one tracklet (slower)",
    )
    closeup_area_fraction: float = _opt(
        0.35,
        "Subject bbox area fraction at/above which a frame is treated as a "
        "close-up and isolation is skipped (default: %(default)s)",
        advanced=True,
    )
    head_weight: float = _opt(
        0.5,
        "Weight for head-keypoint visibility in snapshot scoring "
        "(default: %(default)s)",
        advanced=True,
    )
    body_weight: float = _opt(
        0.5,
        "Weight for lower-body keypoint visibility in snapshot scoring "
        "(default: %(default)s)",
        advanced=True,
    )
    edge_penalty: float = _opt(
        0.85, "Per-touched-edge score multiplier (default: %(default)s)",
        advanced=True,
    )
    edge_tolerance: float = _opt(
        1.0, "Pixel tolerance for 'touches frame edge' (default: %(default)s)",
        advanced=True,
    )
    distractor_weight: float = _opt(
        0.5,
        "Weight for bystander intrusion in padded crop (default: %(default)s)",
        advanced=True,
    )
    same_person_iou: float = _opt(
        0.5,
        "IoU above which a bystander is treated as the subject "
        "(default: %(default)s)",
        advanced=True,
    )
    isolate_dilate: float = _opt(
        0.1,
        "Dilation (per side, fraction of box) for isolated bystanders "
        "(default: %(default)s)",
        advanced=True,
    )
    contact_sheet: str | None = _opt(
        "contact_sheet.jpg", "Contact sheet filename", cli=False, cache=False
    )


@dataclass(frozen=True)
class ClassifyConfig:
    """Which classifier judges the thumbnails, and how."""

    classifier: str = _opt(
        "vlm",
        "Classifier backend (default: %(default)s)",
        choices=CLASSIFIERS,
    )
    model: str | None = _opt(
        None,
        f"Model name: ollama model for vlm (default: {DEFAULT_VLM_MODEL}), "
        f"HF checkpoint for siglip (default: {DEFAULT_SIGLIP_MODEL})",
    )
    threshold: float = _opt(
        0.5,
        "siglip only: redact when match score >= F (default: %(default)s)",
        metavar="F",
    )
    device: str = _opt(
        "auto",
        'siglip only: "auto", "cpu", or a CUDA index (default: %(default)s)',
        flag="--classify-device",
        cache=False,
    )
    ollama_url: str = _opt(
        "http://localhost:11434",
        "ollama server URL (default: %(default)s)",
        metavar="URL",
        cache=False,
    )
    timeout: float = _opt(
        120.0,
        "Per-request ollama timeout in seconds (default: %(default)s)",
        metavar="SECONDS",
        cache=False,
    )
    workers: int = _opt(
        1,
        "vlm only: concurrent ollama requests (default: %(default)s). More "
        "only helps if the server allows it (set OLLAMA_NUM_PARALLEL on "
        "the ollama server)",
        flag="--classify-workers",
        metavar="N",
        cache=False,
    )
    think: bool = _opt(
        False,
        "vlm only: let the model reason before answering (slower; for "
        "models with a thinking mode)",
    )
    max_image_size: int = _opt(
        896,
        "vlm only: downscale thumbnails whose long side exceeds N px before "
        "sending to ollama (default: %(default)s; 0 disables)",
        metavar="N",
    )
    prefilter: str = _opt(
        "none",
        "vlm only: use SigLIP as a fast pre-filter. 'negatives' lets the "
        "VLM re-check SigLIP's negatives (catch false negatives); "
        "'positives' re-checks its positives (catch false positives); "
        "'none' classifies everything with the VLM (default)",
        choices=PREFILTERS,
    )
    prefilter_model: str | None = _opt(
        None,
        f"prefilter: SigLIP checkpoint (default: {DEFAULT_SIGLIP_MODEL})",
        metavar="MODEL",
    )
    prefilter_threshold: float | None = _opt(
        None, "prefilter: SigLIP threshold (default: 0.5)", metavar="F"
    )
    prefilter_device: str = _opt(
        "auto",
        "prefilter: device for SigLIP (default: %(default)s; 'auto' uses the "
        "GPU only if enough VRAM is free to also hold the VLM)",
        metavar="DEV",
        cache=False,
    )
    on_undecided: str = _opt(
        "redact",
        "What to do with a person the classifier could not decide on (a VLM "
        "reply with neither yes nor no, a classifier error): redact them "
        "(default, so a flaky classifier cannot leak), keep them visible, "
        "or fail the run (error)",
        choices=UNDECIDED_POLICIES,
    )
    keep_alive: str = _opt(
        "3h",
        "vlm only: how long ollama keeps the model loaded between "
        "classifications (default: %(default)s)",
        metavar="DUR",
        cache=False,
    )

    @property
    def resolved_model(self) -> str:
        """``model`` with the backend's default filled in."""
        if self.model is not None:
            return self.model
        return DEFAULT_SIGLIP_MODEL if self.classifier == "siglip" else DEFAULT_VLM_MODEL


@dataclass(frozen=True)
class RenderConfig:
    """Box stabilization and video encoding."""

    crf: int = _opt(
        18,
        "Encoder quality, lower is better (default: %(default)s)",
        metavar="N",
    )
    preset: str = _opt(
        "medium", "Encoder speed preset (default: %(default)s)"
    )
    encoder: str = _opt(
        "libx264",
        "Video encoder: libx264 (CPU) or h264_nvenc (NVIDIA GPU; --crf "
        "becomes its constant quality and --preset maps to p1-p7) "
        "(default: %(default)s)",
        choices=ENCODERS,
    )
    selective: bool = _opt(
        True,
        "Only re-encode frames containing redactions and stream-copy the "
        "rest (faster, visually identical; default). Use --no-selective "
        "for a full re-encode with --crf/--preset applied everywhere",
    )
    max_gap: int = _opt(
        30,
        "Interpolate redaction boxes across detection gaps of at most N "
        "frames inside a tracklet, and bridge short ID switches "
        "(default: %(default)s)",
        metavar="N",
    )
    smooth_window: int = _opt(
        9,
        "Odd moving-average window for redaction box coordinates; 1 "
        "disables (default: %(default)s)",
        metavar="N",
    )
    bridge_iou: float = _opt(
        0.5,
        "Minimum endpoint IoU to bridge a short gap between two redacted "
        "tracklets; 0 disables (default: %(default)s)",
        metavar="F",
    )
    min_track_frames: int = _opt(
        5,
        "Ignore redacted tracklets with fewer than N observations (drops "
        "brief false-merge flashes; default: %(default)s; use 1 to keep all)",
        metavar="N",
    )


@dataclass(frozen=True)
class Config:
    """Everything a run can be tuned with."""

    track: TrackConfig = field(default_factory=TrackConfig)
    snapshot: SnapshotConfig = field(default_factory=SnapshotConfig)
    classify: ClassifyConfig = field(default_factory=ClassifyConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    resume: bool = True


SECTIONS: dict[str, type] = {
    "track": TrackConfig,
    "snapshot": SnapshotConfig,
    "classify": ClassifyConfig,
    "render": RenderConfig,
}

T = TypeVar("T")


def resolve(cls: type[T], config: T | None, overrides: Mapping[str, Any]) -> T:
    """``config`` (or the defaults) with keyword ``overrides`` applied.

    Lets stage functions accept either a config object or plain keywords
    while keeping all defaults in this module. Unknown names raise
    ``TypeError`` instead of being silently ignored.
    """
    base = config if config is not None else cls()
    if not overrides:
        return base
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(overrides) - known)
    if unknown:
        raise TypeError(
            f"unexpected option(s) for {cls.__name__}: {', '.join(unknown)}"
        )
    return dataclasses.replace(base, **overrides)


def cache_fields(config: Any) -> dict[str, Any]:
    """The result-affecting fields of a config, for stage cache keys."""
    return {
        f.name: getattr(config, f.name)
        for f in fields(config)
        if f.metadata.get("cache", True)
    }


def dest_name(section: str, f: dataclasses.Field) -> str:
    """argparse ``dest`` for a field: derived from its primary flag."""
    return primary_flag(section, f).lstrip("-").replace("-", "_")


def primary_flag(section: str, f: dataclasses.Field) -> str:
    return f.metadata.get("flag") or "--" + f.name.replace("_", "-")


def from_namespace(args: Any) -> Config:
    """Build a :class:`Config` from a parsed command line.

    Options a subcommand does not define keep their defaults; ``--set``
    values (``args.config_overrides``, see :func:`parse_overrides`) win
    over everything.
    """
    overrides = getattr(args, "config_overrides", None) or {}
    sections: dict[str, Any] = {}
    for name, cls in SECTIONS.items():
        values = {}
        for f in fields(cls):
            if not f.metadata.get("cli", True):
                continue
            dest = dest_name(name, f)
            if hasattr(args, dest):
                values[f.name] = getattr(args, dest)
        values.update(overrides.get(name, {}))
        sections[name] = cls(**values)
    resume = getattr(args, "resume", True)
    return Config(resume=resume, **sections)


_TRUE, _FALSE = {"true", "yes", "on", "1"}, {"false", "no", "off", "0"}


def _coerce(f: dataclasses.Field, text: str) -> Any:
    kind = str(f.type)
    optional = "None" in kind
    base = kind.replace("| None", "").strip()
    if optional and text.strip().lower() in ("none", "null"):
        return None
    if base == "bool":
        lowered = text.strip().lower()
        if lowered in _TRUE:
            return True
        if lowered in _FALSE:
            return False
        raise ValueError(f"expected true or false, got {text!r}")
    try:
        value = {"int": int, "float": float, "str": str}[base](text)
    except (KeyError, ValueError):
        raise ValueError(f"expected {base}, got {text!r}") from None
    choices = f.metadata.get("choices")
    if choices and value not in choices:
        raise ValueError(f"must be one of {', '.join(map(str, choices))}, got {text!r}")
    return value


def option_names(sections: tuple[str, ...]) -> dict[str, tuple[str, dataclasses.Field]]:
    """Every settable name -> (section, field), for ``--set``.

    Accepts the command-line spelling (``head_weight``, ``track_conf``) and
    the config spelling (``snapshot.head_weight``, ``track.conf``).
    """
    names: dict[str, tuple[str, dataclasses.Field]] = {}
    for section in sections:
        for f in fields(SECTIONS[section]):
            for key in (f.name, f"{section}.{f.name}", dest_name(section, f)):
                names.setdefault(key, (section, f))
            names[f"{section}.{f.name}"] = (section, f)
    return names


def parse_overrides(
    assignments: list[str], sections: tuple[str, ...]
) -> dict[str, dict[str, Any]]:
    """Parse ``NAME=VALUE`` strings into ``{section: {field: value}}``.

    Raises ``ValueError`` with a usable message for unknown names or bad
    values.
    """
    names = option_names(sections)
    result: dict[str, dict[str, Any]] = {}
    for item in assignments:
        key, sep, text = item.partition("=")
        key = key.strip()
        if not sep:
            raise ValueError(f"--set expects NAME=VALUE, got {item!r}")
        if key not in names:
            raise ValueError(f"--set: unknown option {key!r} for this command")
        section, f = names[key]
        try:
            result.setdefault(section, {})[f.name] = _coerce(f, text)
        except ValueError as error:
            raise ValueError(f"--set {key}: {error}") from None
    return result
