"""Command-line interface for auto-redact.

One subcommand per pipeline stage (see README): ``track`` (issues #9/#10),
``snapshot`` (issue #12) and ``render`` (issue #15) exist now; classify
and the end-to-end command (#16) come later.
"""

from __future__ import annotations

import argparse
import sys

from .meta import MetaError, load_meta, save_meta
from .render import DEFAULT_CRF, DEFAULT_PRESET, RenderError, render_redacted
from .snapshot import (
    DEFAULT_CONTACT_SHEET,
    DEFAULT_PADDING,
    DEFAULT_THUMBS_DIR,
    SnapshotError,
    select_snapshots,
)
from .track import build_tracklets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="auto-redact",
        description=(
            "Redact people from a video matching a prompt, "
            'e.g. "person in a red shirt".'
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    track = commands.add_parser(
        "track",
        help="Detect people and associate them into per-person tracklets",
        description=(
            "Run YOLO11n-pose detection plus ByteTrack association over "
            "the video, writing tracklet metadata (schema v1) for later "
            "stages to consume."
        ),
    )
    track.add_argument(
        "--video",
        required=True,
        help="Path to the input video file",
    )
    track.add_argument(
        "--meta",
        required=True,
        help="Path to write the tracklet metadata JSON",
    )
    track.add_argument(
        "--stride",
        "--sample-rate",
        type=int,
        default=1,
        metavar="N",
        dest="sample_rate",
        help=(
            "Process every Nth frame, aka stride "
            "(default: 1, i.e. every frame)"
        ),
    )
    track.add_argument(
        "--device",
        default="auto",
        help=(
            'Device for detection/tracking inference: "auto" (GPU if '
            'available, else CPU), or a CUDA index like "0"'
        ),
    )

    snapshot = commands.add_parser(
        "snapshot",
        help="Pick each tracklet's best frame and crop a thumbnail from it",
        description=(
            "Best-view selection v1: for each tracklet, crop the frame "
            "where its bbox is largest (subject closest/largest) into "
            "<thumbs>/track_<id>.jpg, record best_frame + "
            "best_snapshot_path in the metadata, and dump a contact "
            "sheet of all thumbnails for eyeballing."
        ),
    )
    snapshot.add_argument(
        "--meta",
        required=True,
        help="Path to the tracklet metadata JSON, updated in place",
    )
    snapshot.add_argument(
        "--thumbs",
        default=DEFAULT_THUMBS_DIR,
        metavar="DIR",
        help=f"Directory for the thumbnails (default: {DEFAULT_THUMBS_DIR})",
    )
    snapshot.add_argument(
        "--padding",
        type=float,
        default=DEFAULT_PADDING,
        metavar="F",
        help=(
            "Crop margin around each bbox, per side, as a fraction of "
            f"its size (default: {DEFAULT_PADDING})"
        ),
    )
    sheet = snapshot.add_mutually_exclusive_group()
    sheet.add_argument(
        "--contact-sheet",
        default=None,
        metavar="PATH",
        help=(
            "Contact sheet image path; a relative path resolves inside "
            f"--thumbs (default: {DEFAULT_CONTACT_SHEET})"
        ),
    )
    sheet.add_argument(
        "--no-contact-sheet",
        action="store_true",
        help="Skip writing the contact sheet",
    )

    render = commands.add_parser(
        "render",
        help="Render a redacted video from a metadata JSON file",
        description=(
            "Full re-encode: black boxes over every tracklet flagged "
            "redact=true in the metadata; audio is copied unchanged."
        ),
    )
    render.add_argument(
        "--meta",
        required=True,
        help="Path to the tracklet metadata JSON (see --meta of earlier stages)",
    )
    render.add_argument(
        "--out",
        required=True,
        help="Path for the redacted output video",
    )
    render.add_argument(
        "--crf",
        type=int,
        default=DEFAULT_CRF,
        metavar="N",
        help=f"libx264 quality, lower is better (default: {DEFAULT_CRF})",
    )
    render.add_argument(
        "--preset",
        default=DEFAULT_PRESET,
        help=f"libx264 encoding speed preset (default: {DEFAULT_PRESET})",
    )
    return parser


def _cmd_track(args: argparse.Namespace) -> int:
    meta = build_tracklets(
        args.video,
        sample_rate=args.sample_rate,
        device=args.device,
    )
    save_meta(args.meta, meta)
    print(f"wrote {len(meta.tracklets)} tracklet(s) to {args.meta}")
    return 0


def _cmd_snapshot(args: argparse.Namespace) -> int:
    meta = load_meta(args.meta)
    contact_sheet = (
        None if args.no_contact_sheet else (args.contact_sheet or DEFAULT_CONTACT_SHEET)
    )
    select_snapshots(
        meta,
        thumbs_dir=args.thumbs,
        padding=args.padding,
        contact_sheet=contact_sheet,
    )
    save_meta(args.meta, meta)
    print(f"wrote {len(meta.tracklets)} thumbnail(s) to {args.thumbs}")
    return 0


def _cmd_render(args: argparse.Namespace) -> int:
    meta = load_meta(args.meta)
    out = render_redacted(
        meta, args.out, crf=args.crf, preset=args.preset
    )
    print(f"wrote {out}")
    return 0


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "track":
            return _cmd_track(args)
        if args.command == "snapshot":
            return _cmd_snapshot(args)
        if args.command == "render":
            return _cmd_render(args)
    except (
        MetaError,
        RenderError,
        SnapshotError,
        FileNotFoundError,
        ValueError,
    ) as error:
        print(f"auto-redact: error: {error}", file=sys.stderr)
        return 1
    raise AssertionError(f"unhandled command {args.command!r}")  # unreachable


if __name__ == "__main__":
    raise SystemExit(main())
