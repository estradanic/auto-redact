"""Command-line interface for auto-redact.

One subcommand per pipeline stage (see README): ``track`` (issues #9/#10),
``snapshot`` (issue #12), ``classify`` (issue #13) and ``render``
(issue #15), plus ``run`` (issue #16) chaining them all.

Every tunable option is generated from the dataclasses in
:mod:`auto_redact.config`, so defaults and help text live in one place and
each subcommand simply picks which config sections it exposes.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import shutil
import sys
from pathlib import Path

from . import cache, telemetry, workspace
from .classify import ClassifyError, classify_snapshots
from .config import SECTIONS, dest_name, from_namespace, parse_overrides, primary_flag
from .meta import MetaError, load_meta, save_meta
from .pipeline import run_pipeline
from .render import RenderError, render_redacted
from .scene import SceneError
from .snapshot import SnapshotError, select_snapshots
from .streaming import run_pipeline_streaming
from .track import add_orphan_tracklets, build_tracklets
from .ui import progress_bar, setup_logging

DEFAULT_THUMBS_DIR = "thumbs"
COMMANDS = ("track", "snapshot", "classify", "render", "run", "review", "clean")

_SCALARS = {"int": int, "float": float, "str": str}


def add_config_options(
    parser: argparse.ArgumentParser,
    sections: tuple[str, ...],
    *,
    flags: dict[str, tuple[str, ...]] | None = None,
) -> None:
    """Add one option per config field of ``sections`` to ``parser``.

    ``flags`` maps ``"<section>.<field>"`` to flag names that replace the
    field's own flags on this command (its ``dest`` never changes, so the
    parsed namespace is the same everywhere).
    """
    for section in sections:
        for f in dataclasses.fields(SECTIONS[section]):
            if not f.metadata.get("cli", True):
                continue
            names = [primary_flag(section, f), *f.metadata.get("aliases", ())]
            names = list((flags or {}).get(f"{section}.{f.name}", ())) or names
            kwargs: dict = {
                "dest": dest_name(section, f),
                "default": f.default,
                "help": (
                    argparse.SUPPRESS if f.metadata.get("advanced") else f.metadata.get("help")
                ),
            }
            if isinstance(f.default, bool):
                kwargs["action"] = argparse.BooleanOptionalAction
            else:
                base = str(f.type).replace("| None", "").strip()
                kwargs["type"] = _SCALARS[base]
                if "choices" in f.metadata:
                    kwargs["choices"] = f.metadata["choices"]
                if "metavar" in f.metadata:
                    kwargs["metavar"] = f.metadata["metavar"]
            parser.add_argument(*names, **kwargs)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        dest="set_options",
        help=(
            "Set any option by name, including the advanced tuning weights that "
            "have no dedicated flag in --help, e.g. --set head_weight=0.1 "
            "(repeatable; also accepts section names like snapshot.head_weight)"
        ),
    )
    parser.set_defaults(config_sections=tuple(sections))


def _add_common(parser: argparse.ArgumentParser, *, resumable: bool = True) -> None:
    if resumable:
        parser.add_argument(
            "--resume",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Resume an interrupted stage from its checkpoint",
        )
    parser.add_argument(
        "--no-inhibit",
        action="store_true",
        default=False,
        help="Do not prevent the system from sleeping during processing",
    )
    parser.add_argument(
        "--config",
        metavar="FILE",
        default=None,
        help="YAML file with default values for any command-line option",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="auto-redact",
        description=(
            "Redact people from a video matching a prompt, "
            'e.g. "person in a red shirt". Each run writes a log and '
            "stats.json to ~/.local/state/auto-redact/runs/ (--no-stats "
            "or AUTO_REDACT_STATS=0 to disable)."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__import__('auto_redact').__version__}",
    )
    parser.add_argument(
        "--config",
        metavar="FILE",
        default=None,
        help="YAML file with default values for any command-line option",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    track = commands.add_parser(
        "track",
        help="Detect people and associate them into per-person tracklets",
        description=(
            "Run YOLO11n-pose detection plus tracker association over "
            "the video, writing tracklet metadata for later stages to consume."
        ),
    )
    track.add_argument("--video", required=True, help="Path to the input video file")
    track.add_argument(
        "--meta", required=True, help="Path to write the tracklet metadata JSON"
    )
    add_config_options(track, ("track",))
    _add_common(track, resumable=False)

    snapshot = commands.add_parser(
        "snapshot",
        help="Pick each tracklet's best frame and crop a thumbnail from it",
        description=(
            "Best-view selection: for each tracklet, crop the frame where "
            "it is best visible into <thumbs>/track_<id>.jpg, record "
            "best_frame + best_snapshot_path in the metadata, and dump a "
            "contact sheet of all thumbnails for eyeballing."
        ),
    )
    snapshot.add_argument(
        "--meta", required=True,
        help="Path to the tracklet metadata JSON, updated in place",
    )
    snapshot.add_argument(
        "--thumbs", default=DEFAULT_THUMBS_DIR, metavar="DIR",
        help=f"Directory for the thumbnails (default: {DEFAULT_THUMBS_DIR})",
    )
    add_config_options(snapshot, ("snapshot",))
    sheet = snapshot.add_mutually_exclusive_group()
    sheet.add_argument(
        "--contact-sheet", default=None, metavar="PATH",
        help=(
            "Contact sheet image path; a relative path resolves inside "
            "--thumbs (default: contact_sheet.jpg)"
        ),
    )
    sheet.add_argument(
        "--no-contact-sheet", action="store_true", help="Skip writing the contact sheet"
    )
    _add_common(snapshot)

    classify = commands.add_parser(
        "classify",
        help="Decide which tracklets match the prompt (VLM or SigLIP)",
        description=(
            "For each tracklet's snapshot, ask a vision model whether it "
            "shows a person matching the prompt, and set the tracklet's "
            "redact flag in the metadata."
        ),
    )
    classify.add_argument(
        "--meta", required=True,
        help="Path to the metadata JSON with snapshots, updated in place",
    )
    classify.add_argument(
        "--prompt", required=True, help='What to redact, e.g. "person in a red shirt"'
    )
    add_config_options(
        classify, ("classify",),
        flags={"classify.device": ("--device", "--classify-device")},
    )
    _add_common(classify)

    render = commands.add_parser(
        "render",
        help="Render a redacted video from a metadata JSON file",
        description=(
            "Black boxes over every tracklet flagged redact=true in the "
            "metadata; audio is copied unchanged."
        ),
    )
    render.add_argument(
        "--meta", required=True,
        help="Path to the tracklet metadata JSON (see --meta of earlier stages)",
    )
    render.add_argument("--out", required=True, help="Path for the redacted output video")
    add_config_options(render, ("render",))
    _add_common(render, resumable=False)

    run = commands.add_parser(
        "run",
        help="End to end: track, snapshot, classify and render in one command",
        description=(
            "Run every stage on a video and prompt, caching intermediate "
            "results next to the metadata file. A re-run reuses each stage "
            "whose settings (and upstream stages) are unchanged; use "
            "--force to redo everything."
        ),
    )
    run.add_argument("--video", required=True, help="Path to the input video")
    run.add_argument(
        "--prompt", required=True, help='What to redact, e.g. "person in a red shirt"'
    )
    run.add_argument("--out", required=True, help="Path for the redacted video")
    run.add_argument(
        "--meta", default=None,
        help=(
            "Metadata JSON path (default: a per-video directory under "
            "~/.local/state/auto-redact/videos/, see README)"
        ),
    )
    run.add_argument(
        "--thumbs", default=None, metavar="DIR",
        help="Thumbnail directory (default: thumbs/ beside the metadata)",
    )
    run.add_argument(
        "--force", action="store_true",
        help="Ignore cached metadata and redo every stage",
    )
    run.add_argument(
        "--streaming", action="store_true", default=False,
        help="Use the single-pass streaming pipeline (experimental)",
    )
    add_config_options(run, ("track", "snapshot", "classify", "render"))
    _add_common(run)

    review = commands.add_parser(
        "review",
        help="Open a browser page to check decisions, flag mistakes and tune settings",
        description=(
            "Serve a local review page over a video's stored tracking results: "
            "overrule the classifier per person, scrub the video with the "
            "redaction boxes drawn on, flag frames where someone is visible "
            "(leak) or a box is wrong (over), and get suggested settings that "
            "fit your flags. Your verdicts are saved with the video's data and "
            "applied by later runs."
        ),
    )
    where = review.add_mutually_exclusive_group(required=True)
    where.add_argument("--video", help="The source video (finds its stored data from a previous run)")
    where.add_argument("--meta", help="A metadata file to review directly")
    review.add_argument(
        "--out", default=None,
        help="Where the page's Render button writes the redacted video (omit to disable it)",
    )
    review.add_argument("--port", type=int, default=0, help="Port to serve on (default: any free port)")
    review.add_argument("--no-browser", action="store_true", help="Do not open a browser tab")
    add_config_options(review, ("render", "classify"))
    review.add_argument("--no-inhibit", action="store_true", default=False, help=argparse.SUPPRESS)
    review.add_argument("--config", metavar="FILE", default=None, help="YAML file with default values for any command-line option")

    clean = commands.add_parser(
        "clean",
        help="List or delete stored run data (per-video workspaces, run logs)",
        description=(
            "Without options, lists what is stored under the state directory "
            "(~/.local/state/auto-redact): per-video workspaces (tracking "
            "results, thumbnails, review labels) and run logs. Select what to "
            "delete with the options below; nothing is removed without "
            "confirmation (or --yes). Workspaces holding review labels are "
            "skipped unless --labeled."
        ),
    )
    clean.add_argument("--all", action="store_true", help="Select every workspace")
    clean.add_argument(
        "--missing", action="store_true",
        help="Select workspaces whose source video no longer exists",
    )
    clean.add_argument(
        "--video", metavar="PATH", default=None,
        help="Select the workspace of this source video",
    )
    clean.add_argument(
        "--older-than", type=float, metavar="DAYS", default=None,
        help="Only select entries not modified for DAYS (alone: select every such workspace)",
    )
    clean.add_argument(
        "--runs", action="store_true",
        help="Also select run logs (all, or those older than --older-than)",
    )
    clean.add_argument(
        "--labeled", action="store_true",
        help="Allow deleting workspaces that hold review labels",
    )
    clean.add_argument("--dry-run", action="store_true", help="Show what would be deleted")
    clean.add_argument("-y", "--yes", action="store_true", help="Do not ask for confirmation")
    clean.add_argument(
        "--no-inhibit", action="store_true", default=False, help=argparse.SUPPRESS
    )
    return parser


def _resolve_out(out: str | Path, input_path: str | Path) -> Path:
    """Resolve ``--out``: if it points at (or looks like) a directory, place
    the output inside it using the input file's name.
    """
    out = Path(out)
    input_path = Path(input_path)
    if out.is_dir() or not out.suffix:
        out.mkdir(parents=True, exist_ok=True)
        out = out / input_path.name
    return out


def _cmd_run(args: argparse.Namespace) -> int:
    setup_logging()
    out = _resolve_out(args.out, args.video)
    cfg = from_namespace(args)
    common = dict(
        meta_path=args.meta,
        thumbs_dir=args.thumbs,
        force=args.force,
        progress=lambda message: print(message, flush=True),
        bar=progress_bar,
    )
    if args.streaming:
        result = run_pipeline_streaming(args.video, args.prompt, out, cfg, **common)
        print(
            f"{result['tracklets']} tracklet(s), {result['redacted']} redacted; "
            f"wrote {result['out']}"
        )
        return 0
    summary = run_pipeline(args.video, args.prompt, out, cfg, **common)
    cached = f" (reused: {', '.join(summary.reused)})" if summary.reused else ""
    print(
        f"{summary.tracklets} tracklet(s), {summary.redacted} redacted; "
        f"wrote {summary.out}{cached}"
    )
    return 0


def _cmd_track(args: argparse.Namespace) -> int:
    cfg = from_namespace(args)
    stages = cache.StageCache(args.meta)
    key = cache.track_key(args.video, cfg.track)
    stages.begin("track", key)
    with progress_bar("track") as on_progress:
        meta = build_tracklets(
            args.video, cfg.track, on_progress=on_progress, meta_path=args.meta,
            resume=cfg.resume,
        )
    if cfg.track.orphans:
        with progress_bar("orphan") as orphan_progress:
            meta = add_orphan_tracklets(meta, args.video, on_progress=orphan_progress)
    if cfg.track.scene_cuts:
        print(
            f"scene cuts: {len(meta.scene_cuts)} cut(s), "
            f"{meta.scene_cut_splits} split(s)",
            file=sys.stderr,
        )
    save_meta(args.meta, meta)
    stages.record("track", key)
    print(f"wrote {len(meta.tracklets)} tracklet(s) to {args.meta}")
    return 0


def _cmd_snapshot(args: argparse.Namespace) -> int:
    cfg = from_namespace(args)
    contact_sheet = (
        None
        if args.no_contact_sheet
        else (args.contact_sheet or cfg.snapshot.contact_sheet)
    )
    snapshot_cfg = dataclasses.replace(cfg.snapshot, contact_sheet=contact_sheet)
    stages = cache.StageCache(args.meta)
    key = cache.snapshot_key(stages.get("track"), snapshot_cfg, args.thumbs)
    resume = cfg.resume and (
        stages.matches("snapshot", key) or stages.begin("snapshot", key)
    )
    meta = load_meta(args.meta)
    with progress_bar("snapshot") as on_progress:
        select_snapshots(
            meta, args.thumbs, snapshot_cfg,
            on_progress=on_progress, resume=resume, meta_path=args.meta,
        )
    save_meta(args.meta, meta)
    stages.record("snapshot", key)
    isolated = sum(len(t.isolated_track_ids) for t in meta.tracklets)
    note = f" ({isolated} bystander(s) isolated)" if isolated else ""
    print(f"wrote {len(meta.tracklets)} thumbnail(s) to {args.thumbs}{note}")
    return 0


def _cmd_classify(args: argparse.Namespace) -> int:
    setup_logging()
    cfg = from_namespace(args)
    classify = cfg.classify
    meta = load_meta(args.meta)
    print(
        "initializing classifier model"
        if classify.classifier == "siglip"
        else "waiting for ollama to load the model",
        file=sys.stderr,
    )

    stages = cache.StageCache(args.meta)
    key = cache.classify_key(stages.get("snapshot"), args.prompt, classify)
    resume = cfg.resume and (
        stages.matches("classify", key) or stages.begin("classify", key)
    )
    if not resume:
        for tracklet in meta.tracklets:
            tracklet.redact = None
            tracklet.score = None
            tracklet.detail = None

    with progress_bar("classify") as on_progress:
        classify_snapshots(
            meta, args.prompt, classify,
            on_progress=on_progress, resume=resume, meta_path=args.meta,
        )
    save_meta(args.meta, meta)
    stages.record("classify", key)
    flagged = sum(1 for t in meta.tracklets if t.effective_redact)
    print(f"classified {len(meta.tracklets)} tracklet(s), {flagged} to redact")
    return 0


def _cmd_render(args: argparse.Namespace) -> int:
    cfg = from_namespace(args)
    meta = load_meta(args.meta)
    out = _resolve_out(args.out, meta.video.path)
    with progress_bar("render") as on_progress:
        out = render_redacted(meta, out, cfg.render, on_progress=on_progress)
    print(f"wrote {out}")
    return 0


def _cmd_review(args: argparse.Namespace) -> int:
    from .review import Review, serve

    if args.meta:
        meta_path = Path(args.meta)
    else:
        meta_path = workspace.workspace_for(args.video) / "meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(
                f"no stored tracking for {args.video}: run `auto-redact run` on it first "
                "(or pass --meta for metadata kept elsewhere)"
            )
    cfg = from_namespace(args)
    session = Review(meta_path, cfg.render, cfg.classify, args.out)
    serve(session, port=args.port, open_browser=not args.no_browser)
    return 0


def _describe(entry: workspace.Entry) -> str:
    import datetime

    when = datetime.datetime.fromtimestamp(entry.modified).strftime("%Y-%m-%d %H:%M")
    size = workspace.human_size(entry.size).rjust(9)
    if entry.video is None and entry.path.parent.name != "videos":
        return f"{size}  {when}  {entry.name}"
    note = "" if entry.video_exists else "  [video missing]"
    note += "  [labels]" if entry.labeled else ""
    return f"{size}  {when}  {entry.name}  <- {entry.video or 'unknown video'}{note}"


def _cmd_clean(args: argparse.Namespace) -> int:
    workspaces = workspace.list_workspaces()
    runs = workspace.list_runs()
    chosen = (
        args.all or args.missing or args.video is not None
        or args.older_than is not None or args.runs
    )
    if not chosen:
        print(f"state directory: {workspace.state_root()}")
        print(f"\nworkspaces ({len(workspaces)}, "
              f"{workspace.human_size(sum(w.size for w in workspaces))}):")
        for entry in workspaces:
            print("  " + _describe(entry))
        print(f"\nrun logs ({len(runs)}, {workspace.human_size(sum(r.size for r in runs))}):")
        for entry in runs[-10:]:
            print("  " + _describe(entry))
        if len(runs) > 10:
            print(f"  ... and {len(runs) - 10} older")
        return 0

    selected, skipped = [], []
    if args.all or args.missing or args.video is not None or args.older_than is not None:
        selected, skipped = workspace.select_for_clean(
            workspaces, everything=args.all, missing=args.missing, video=args.video,
            older_than_days=args.older_than, include_labeled=args.labeled,
        )
    if args.runs:
        selected += workspace.select_runs(runs, args.older_than)
    for entry in skipped:
        print(f"skipping (has review labels; use --labeled): {entry.name}")
    if not selected:
        print("nothing to delete")
        return 0

    for entry in selected:
        print("  " + _describe(entry))
    total = workspace.human_size(sum(e.size for e in selected))
    if args.dry_run:
        print(f"would delete {len(selected)} entr{'y' if len(selected) == 1 else 'ies'} ({total})")
        return 0
    if not args.yes:
        if not sys.stdin.isatty():
            print("auto-redact: error: refusing to delete without --yes (not a terminal)", file=sys.stderr)
            return 1
        if input(f"Delete {len(selected)} entr{'y' if len(selected) == 1 else 'ies'} ({total})? [y/N] ").strip().lower() != "y":
            print("cancelled")
            return 1
    for entry in selected:
        workspace.remove_entry(entry)
    print(f"deleted {len(selected)} entr{'y' if len(selected) == 1 else 'ies'} ({total})")
    return 0


def _inhibit_sleep_if_available(argv: list[str]) -> None:
    """Re-exec under systemd-inhibit to prevent sleep, if available and not opted out."""
    if os.environ.get("AUTO_REDACT_INHIBITED"):
        return
    if "--no-inhibit" in argv:
        return
    # Only inhibit when invoked as the CLI entry point; programmatic callers
    # (tests, library use) must not be re-executed.
    prog = str(sys.argv[0]) if sys.argv else ""
    if not (prog.endswith("auto-redact") or prog.endswith("cli.py")):
        return
    inhibitor = shutil.which("systemd-inhibit")
    if inhibitor is None:
        return
    os.environ["AUTO_REDACT_INHIBITED"] = "1"
    # Re-exec the same invocation under the inhibitor. Prefer the argv we
    # were called with so programmatic callers (tests) keep their arguments;
    # fall back to sys.argv for the CLI entry point.
    base = list(argv) if argv else list(sys.argv[1:])
    if not base or not os.access(sys.argv[0], os.X_OK) or str(sys.argv[0]).endswith(".py"):
        cmd = [sys.executable, "-m", "auto_redact.cli"] + base
    else:
        cmd = [sys.argv[0]] + base
    os.execvp(inhibitor, [inhibitor, "--what=sleep:idle", "--who=auto-redact",
                          "--why=Processing video"] + cmd)


# Config files written for the old flat option names keep working.
_LEGACY_KEYS = {"sample_rate": "stride"}
_LEGACY_INVERTED = {"no_half": "half", "no_resume": "resume"}


def _load_config_file(path: str) -> dict:
    """Read a YAML defaults file into ``{argparse dest: value}``."""
    import yaml

    with open(path, encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"config {path} must be a YAML mapping")
    cfg = {}
    for key, value in loaded.items():
        if key in _LEGACY_INVERTED and isinstance(value, bool):
            cfg[_LEGACY_INVERTED[key]] = not value
        else:
            cfg[_LEGACY_KEYS.get(key, key)] = value
    return cfg


def _find_config_path(argv: list[str]) -> str | None:
    path = None
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok == "--config" and i + 1 < len(argv):
            path = argv[i + 1]
            i += 2
            continue
        if tok.startswith("--config="):
            path = tok.split("=", 1)[1]
        i += 1
    return path


def _apply_config_file(
    parser: argparse.ArgumentParser, argv: list[str], cfg: dict
) -> list[str]:
    """Make ``cfg`` the parser's defaults (CLI flags still win).

    Required options (``--video`` ...) are injected into ``argv`` so
    argparse's ``required=True`` is satisfied by the file too.
    """
    subcommand = next((tok for tok in argv if tok in COMMANDS), None)
    if subcommand is None and cfg.get("command") in COMMANDS:
        subcommand = cfg["command"]
        argv = [subcommand] + argv

    for key in ("video", "prompt", "out", "meta"):
        flag = "--" + key
        if key not in cfg or any(t == flag or t.startswith(flag + "=") for t in argv):
            continue
        if subcommand == "review" and key == "video" and "--meta" in argv:
            continue  # --video and --meta are alternatives there
        injection = [flag, str(cfg[key])]
        index = argv.index(subcommand) + 1 if subcommand in argv else 0
        argv = argv[:index] + injection + argv[index:]

    for action in parser._subparsers._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, sub in action.choices.items():
                if name == subcommand:
                    sub.set_defaults(**cfg)
    parser.set_defaults(**cfg)
    return argv


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    no_stats = "--no-stats" in argv
    _inhibit_sleep_if_available(argv)
    argv = [tok for tok in argv if tok != "--no-stats"]
    parser = build_parser()
    config_path = _find_config_path(argv)
    if config_path is not None:
        try:
            cfg = _load_config_file(config_path)
        except ImportError as error:  # pragma: no cover
            print(f"auto-redact: error: pyyaml is required for --config: {error}", file=sys.stderr)
            return 1
        except (OSError, ValueError) as error:
            print(f"auto-redact: error: could not read config {config_path}: {error}", file=sys.stderr)
            return 1
        argv = _apply_config_file(parser, argv, cfg)
    args = parser.parse_args(argv)
    if getattr(args, "set_options", None):
        try:
            args.config_overrides = parse_overrides(args.set_options, args.config_sections)
        except ValueError as error:
            parser.error(str(error))
    run = None if no_stats else telemetry.begin(
        args.command, {"argv": argv, "args": vars(args)}
    )
    exit_code, failure = 1, None
    try:
        handlers = {
            "run": _cmd_run, "track": _cmd_track, "snapshot": _cmd_snapshot,
            "classify": _cmd_classify, "render": _cmd_render, "clean": _cmd_clean,
            "review": _cmd_review,
        }
        exit_code = handlers[args.command](args)
        return exit_code
    except (
        ClassifyError,
        MetaError,
        RenderError,
        SnapshotError,
        SceneError,
        FileNotFoundError,
        ValueError,
    ) as error:
        failure = error
        print(f"auto-redact: error: {error}", file=sys.stderr)
        return 1
    except BaseException as error:
        failure = error
        raise
    finally:
        if run is not None:
            where = telemetry.end(exit_code, failure)
            if where is not None:
                print(f"auto-redact: stats and logs: {where}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
