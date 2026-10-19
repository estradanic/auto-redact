"""Command-line interface for auto-redact.

Pipeline (see README): track -> snapshot -> classify -> render. Only
argument parsing works so far; the pipeline itself is still stubs.
"""

import argparse


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="auto-redact",
        description=(
            "Redact people from a video matching a prompt, "
            'e.g. "person in a red shirt".'
        ),
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to the input video file",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help='Redaction prompt describing who to redact, e.g. "person in a red shirt"',
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Path for the redacted output video",
    )
    parser.add_argument(
        "--meta",
        help="Path to read/write tracklet metadata JSON (optional)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=1,
        metavar="N",
        help="Process every Nth frame (default: 1, i.e. every frame)",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    print(
        "auto-redact: not implemented yet "
        f"(would redact {args.prompt!r} from {args.video!r} -> {args.out!r})"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
