"""Command-line interface for auto-redact.

One subcommand per pipeline stage (see README): ``render`` exists now
(issue #15); track, classify and the end-to-end command (#16) come
later.
"""

from __future__ import annotations

import argparse
import sys

from .meta import MetaError, load_meta
from .render import DEFAULT_CRF, DEFAULT_PRESET, RenderError, render_redacted


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="auto-redact",
        description=(
            "Redact people from a video matching a prompt, "
            'e.g. "person in a red shirt".'
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

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
        if args.command == "render":
            return _cmd_render(args)
    except (MetaError, RenderError, FileNotFoundError, ValueError) as error:
        print(f"auto-redact: error: {error}", file=sys.stderr)
        return 1
    raise AssertionError(f"unhandled command {args.command!r}")  # unreachable


if __name__ == "__main__":
    raise SystemExit(main())
