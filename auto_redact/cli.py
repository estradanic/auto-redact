"""Command-line interface for auto-redact.

One subcommand per pipeline stage (see README): ``track`` (issues #9/#10),
``snapshot`` (issue #12), ``classify`` (issue #13) and ``render``
(issue #15), plus ``run`` (issue #16) chaining them all.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .classify import (
    DEFAULT_MAX_IMAGE_SIZE,
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_URL,
    DEFAULT_TIMEOUT,
    DEFAULT_WORKERS,
    ClassifyError,
    classify_snapshots,
)
from .siglip import DEFAULT_SIGLIP_MODEL, DEFAULT_THRESHOLD, SiglipClassifier
from .meta import MetaError, load_meta, save_meta
from .render import (
    DEFAULT_BRIDGE_IOU,
    DEFAULT_CRF,
    DEFAULT_ENCODER,
    DEFAULT_MAX_GAP,
    DEFAULT_MIN_TRACK_FRAMES,
    DEFAULT_PRESET,
    DEFAULT_SMOOTH_WINDOW,
    ENCODERS,
    RenderError,
    render_redacted,
)
from .snapshot import (
    BODY_WEIGHT,
    CLOSEUP_AREA_FRACTION,
    DEFAULT_CONTACT_SHEET,
    DEFAULT_PADDING,
    DEFAULT_THUMBS_DIR,
    DISTRACTOR_WEIGHT,
    EDGE_PENALTY,
    EDGE_TOLERANCE,
    HEAD_WEIGHT,
    ISOLATE_DILATE,
    SAME_PERSON_IOU,
    SnapshotError,
    select_snapshots,
)
from .pipeline import run_pipeline, _classify_key
from .streaming import run_pipeline_streaming
from .track import build_tracklets
from .scene import SceneError
from .ui import progress_bar, setup_logging


def _add_stability_args(parser: argparse.ArgumentParser) -> None:
    """Flags that keep redaction boxes continuous across detection dropouts."""
    parser.add_argument(
        "--max-gap",
        type=int,
        default=DEFAULT_MAX_GAP,
        metavar="N",
        help=(
            "Interpolate redaction boxes across detection gaps of at most N "
            f"frames inside a tracklet, and bridge short ID switches "
            f"(default: {DEFAULT_MAX_GAP})"
        ),
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=DEFAULT_SMOOTH_WINDOW,
        metavar="N",
        help=(
            "Odd moving-average window for redaction box coordinates; 1 "
            f"disables (default: {DEFAULT_SMOOTH_WINDOW})"
        ),
    )
    parser.add_argument(
        "--bridge-iou",
        type=float,
        default=DEFAULT_BRIDGE_IOU,
        metavar="F",
        help=(
            "Minimum endpoint IoU to bridge a short gap between two "
            f"redacted tracklets; 0 disables (default: {DEFAULT_BRIDGE_IOU})"
        ),
    )
    parser.add_argument(
        "--min-track-frames",
        type=int,
        default=DEFAULT_MIN_TRACK_FRAMES,
        metavar="N",
        help=(
            "Ignore redacted tracklets with fewer than N observations "
            f"(drops brief false-merge flashes; default: {DEFAULT_MIN_TRACK_FRAMES}; "
            "use 1 to keep all)"
        ),
    )



def _add_scene_args(parser: argparse.ArgumentParser) -> None:
    """Flags for scene-cut detection and tracklet splitting."""
    parser.add_argument(
        "--scene-cuts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Detect hard scene cuts and split tracklets that span them, so "
            "a person is not joined across an edit (default: on)"
        ),
    )
    parser.add_argument(
        "--scene-cut-threshold",
        type=float,
        default=0.45,
        metavar="F",
        help="Bhattacharyya distance threshold for a hard cut (default: 0.45)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="auto-redact",
        description=(
            "Redact people from a video matching a prompt, "
            'e.g. "person in a red shirt".'
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
    track.add_argument(
        "--no-half",
        action="store_true",
        help="Disable FP16 tracking inference (on by default on CUDA)",
    )
    track.add_argument(
        "--track-workers",
        type=int,
        default=2,
        metavar="N",
        help=(
            "Track N chunks of the video concurrently (default: 2). Useful "
            "when the GPU is underused; tracklets crossing chunk seams are "
            "stitched by box overlap"
        ),
    )
    track.add_argument(
        "--no-resume",
        action="store_true",
        help=(
            "Disable resumable tracking: do not save chunk progress and do "
            "not reuse a previous interrupted run (default: resumable)"
        ),
    )

    _add_scene_args(track)
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
    snapshot.add_argument(
        "--isolate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Black out other people who appear in a thumbnail so the "
            "classifier can only judge the tracklet's own subject "
            "(default). Use --no-isolate for raw crops"
        ),
    )
    snapshot.add_argument(
        "--sub-isolate",
        action="store_true",
        default=False,
        help=(
            "Also run a detector on each thumbnail and black out extra "
            "people the tracker merged into one tracklet (slower)"
        ),
    )
    snapshot.add_argument(
        "--closeup-area-fraction",
        type=float,
        default=CLOSEUP_AREA_FRACTION,
        help=(
            "Subject bbox area fraction at/above which a frame is treated "
            "as a close-up and isolation is skipped (default: "
            f"{CLOSEUP_AREA_FRACTION})"
        ),
    )
    snapshot.add_argument(
        "--head-weight",
        type=float,
        default=HEAD_WEIGHT,
        help=f"Weight for head-keypoint visibility in snapshot scoring (default: {HEAD_WEIGHT})",
    )
    snapshot.add_argument(
        "--body-weight",
        type=float,
        default=BODY_WEIGHT,
        help=f"Weight for lower-body keypoint visibility in snapshot scoring (default: {BODY_WEIGHT})",
    )
    snapshot.add_argument(
        "--edge-penalty",
        type=float,
        default=EDGE_PENALTY,
        help=f"Per-touched-edge score multiplier (default: {EDGE_PENALTY})",
    )
    snapshot.add_argument(
        "--edge-tolerance",
        type=float,
        default=EDGE_TOLERANCE,
        help=f"Pixel tolerance for 'touches frame edge' (default: {EDGE_TOLERANCE})",
    )
    snapshot.add_argument(
        "--distractor-weight",
        type=float,
        default=DISTRACTOR_WEIGHT,
        help=f"Weight for bystander intrusion in padded crop (default: {DISTRACTOR_WEIGHT})",
    )
    snapshot.add_argument(
        "--same-person-iou",
        type=float,
        default=SAME_PERSON_IOU,
        help=f"IoU above which a bystander is treated as the subject (default: {SAME_PERSON_IOU})",
    )
    snapshot.add_argument(
        "--isolate-dilate",
        type=float,
        default=ISOLATE_DILATE,
        help=f"Dilation (per side, fraction of box) for isolated bystanders (default: {ISOLATE_DILATE})",
    )
    snapshot.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resumable snapshot selection (default: resumable).",
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

    classify = commands.add_parser(
        "classify",
        help="Decide which tracklets match the prompt (VLM or SigLIP)",
        description=(
            "Classifier v1: for each tracklet's snapshot, ask a vision "
            "model via ollama whether it shows a person matching the "
            "prompt, and set the tracklet's redact flag in the metadata."
        ),
    )
    classify.add_argument(
        "--meta",
        required=True,
        help="Path to the metadata JSON with snapshots, updated in place",
    )
    classify.add_argument(
        "--prompt",
        required=True,
        help='What to redact, e.g. "person in a red shirt"',
    )
    classify.add_argument(
        "--classifier",
        choices=("vlm", "siglip"),
        default="vlm",
        help="Classifier backend (default: vlm)",
    )
    classify.add_argument(
        "--model",
        default=None,
        help=(
            f"Model name: ollama model for vlm (default: {DEFAULT_MODEL}), "
            f"HF checkpoint for siglip (default: {DEFAULT_SIGLIP_MODEL})"
        ),
    )
    classify.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        metavar="F",
        help=(
            "siglip only: redact when match score >= F "
            f"(default: {DEFAULT_THRESHOLD})"
        ),
    )
    classify.add_argument(
        "--device",
        default="auto",
        help='siglip only: "auto", "cpu", or a CUDA index (default: auto)',
    )
    classify.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_URL,
        metavar="URL",
        help=f"ollama server URL (default: {DEFAULT_OLLAMA_URL})",
    )
    classify.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        metavar="SECONDS",
        help=f"Per-request timeout (default: {DEFAULT_TIMEOUT:g})",
    )
    classify.add_argument(
        "--think",
        action="store_true",
        help=(
            "vlm only: let the model reason before answering (slower; for "
            "models with a thinking mode)"
        ),
    )
    classify.add_argument(
        "--classify-workers",
        type=int,
        default=DEFAULT_WORKERS,
        metavar="N",
        help=(
            "vlm only: concurrent ollama requests (default: %d). More only "
            "helps if the server allows it (set the OLLAMA_NUM_PARALLEL env "
            "var on the ollama server)" % DEFAULT_WORKERS
        ),
    )
    classify.add_argument(
        "--max-image-size",
        type=int,
        default=DEFAULT_MAX_IMAGE_SIZE,
        metavar="N",
        help=(
            "vlm only: downscale thumbnails whose long side exceeds N px before "
            f"sending to ollama (default: {DEFAULT_MAX_IMAGE_SIZE}; 0 disables)"
        ),
    )
    classify.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resumable classification (default: resumable).",
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
    render.add_argument(
        "--encoder",
        choices=ENCODERS,
        default=DEFAULT_ENCODER,
        help=(
            "Video encoder: libx264 (CPU) or h264_nvenc (NVIDIA GPU; --crf "
            f"becomes its constant quality and --preset maps to p1-p7) "
            f"(default: {DEFAULT_ENCODER})"
        ),
    )
    render.add_argument(
        "--selective",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Only re-encode frames containing redactions and stream-copy the "
            "rest (faster, visually identical; default). Use --no-selective "
            "for a full re-encode with --crf/--preset applied everywhere"
        ),
    )
    _add_stability_args(render)

    run = commands.add_parser(
        "run",
        help="End to end: track, snapshot, classify and render in one command",
        description=(
            "Run every stage on a video and prompt, caching intermediate "
            "results in the metadata file. Re-running reuses cached stages "
            "(tracking, snapshots, and classification for the same "
            "prompt/model); use --force to redo everything."
        ),
    )
    run.add_argument("--video", required=True, help="Path to the input video")
    run.add_argument(
        "--prompt", required=True, help='What to redact, e.g. "person in a red shirt"'
    )
    run.add_argument("--out", required=True, help="Path for the redacted video")
    run.add_argument(
        "--meta",
        default=None,
        help="Metadata JSON cache path (default: <out stem>.meta.json next to --out)",
    )
    run.add_argument(
        "--thumbs",
        default=None,
        metavar="DIR",
        help="Thumbnail directory (default: <meta stem>_thumbs next to the metadata)",
    )
    run.add_argument(
        "--force",
        action="store_true",
        help="Ignore cached metadata and redo every stage",
    )
    run.add_argument(
        "--stride",
        type=int,
        default=1,
        metavar="N",
        dest="sample_rate",
        help="Track every Nth frame (default: 1, i.e. every frame)",
    )
    run.add_argument(
        "--device",
        default="auto",
        help='Tracking device: "auto" (GPU if available, else CPU) or a CUDA index',
    )
    run.add_argument(
        "--no-half",
        action="store_true",
        help="Disable FP16 tracking inference (on by default on CUDA)",
    )
    run.add_argument(
        "--track-workers",
        type=int,
        default=2,
        metavar="N",
        help=(
            "Track N chunks of the video concurrently (default: 2). Useful "
            "when the GPU is underused; tracklets crossing chunk seams are "
            "stitched by box overlap"
        ),
    )
    run.add_argument(
        "--no-resume",
        action="store_true",
        help=(
            "Disable resumable tracking (default: resumable)."
        ),
    )
    run.add_argument(
        "--padding",
        type=float,
        default=DEFAULT_PADDING,
        metavar="F",
        help=f"Snapshot crop margin per side as a fraction of bbox size (default: {DEFAULT_PADDING})",
    )
    run.add_argument(
        "--isolate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Black out other people who appear in a thumbnail so the "
            "classifier can only judge the tracklet's own subject "
            "(default). Use --no-isolate for raw crops"
        ),
    )
    run.add_argument(
        "--sub-isolate",
        action="store_true",
        default=False,
        help=(
            "Also run a detector on each thumbnail and black out extra "
            "people the tracker merged into one tracklet (slower)"
        ),
    )
    run.add_argument(
        "--closeup-area-fraction",
        type=float,
        default=CLOSEUP_AREA_FRACTION,
        help=(
            "Subject bbox area fraction at/above which a frame is treated "
            "as a close-up and isolation is skipped (default: "
            f"{CLOSEUP_AREA_FRACTION})"
        ),
    )
    run.add_argument(
        "--head-weight",
        type=float,
        default=HEAD_WEIGHT,
        help=f"Weight for head-keypoint visibility in snapshot scoring (default: {HEAD_WEIGHT})",
    )
    run.add_argument(
        "--body-weight",
        type=float,
        default=BODY_WEIGHT,
        help=f"Weight for lower-body keypoint visibility in snapshot scoring (default: {BODY_WEIGHT})",
    )
    run.add_argument(
        "--edge-penalty",
        type=float,
        default=EDGE_PENALTY,
        help=f"Per-touched-edge score multiplier (default: {EDGE_PENALTY})",
    )
    run.add_argument(
        "--edge-tolerance",
        type=float,
        default=EDGE_TOLERANCE,
        help=f"Pixel tolerance for 'touches frame edge' (default: {EDGE_TOLERANCE})",
    )
    run.add_argument(
        "--distractor-weight",
        type=float,
        default=DISTRACTOR_WEIGHT,
        help=f"Weight for bystander intrusion in padded crop (default: {DISTRACTOR_WEIGHT})",
    )
    run.add_argument(
        "--same-person-iou",
        type=float,
        default=SAME_PERSON_IOU,
        help=f"IoU above which a bystander is treated as the subject (default: {SAME_PERSON_IOU})",
    )
    run.add_argument(
        "--isolate-dilate",
        type=float,
        default=ISOLATE_DILATE,
        help=f"Dilation (per side, fraction of box) for isolated bystanders (default: {ISOLATE_DILATE})",
    )
    run.add_argument(
        "--classifier",
        choices=("vlm", "siglip"),
        default="vlm",
        help="Classifier backend (default: vlm)",
    )
    run.add_argument(
        "--model",
        default=None,
        help=(
            f"Model name: ollama model for vlm (default: {DEFAULT_MODEL}), "
            f"HF checkpoint for siglip (default: {DEFAULT_SIGLIP_MODEL})"
        ),
    )
    run.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        metavar="F",
        help=(
            "siglip only: redact when match score >= F "
            f"(default: {DEFAULT_THRESHOLD})"
        ),
    )
    run.add_argument(
        "--classify-device",
        default="auto",
        help='siglip only: "auto", "cpu", or a CUDA index (default: auto)',
    )
    run.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_URL,
        metavar="URL",
        help=f"ollama server URL (default: {DEFAULT_OLLAMA_URL})",
    )
    run.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        metavar="SECONDS",
        help=f"Per-request ollama timeout (default: {DEFAULT_TIMEOUT:g})",
    )
    run.add_argument(
        "--think",
        action="store_true",
        help=(
            "vlm only: let the model reason before answering (slower; for "
            "models with a thinking mode)"
        ),
    )
    run.add_argument(
        "--classify-workers",
        type=int,
        default=DEFAULT_WORKERS,
        metavar="N",
        help=(
            "vlm only: concurrent ollama requests (default: %d). More only "
            "helps if the server allows it (set the OLLAMA_NUM_PARALLEL env "
            "var on the ollama server)" % DEFAULT_WORKERS
        ),
    )
    run.add_argument(
        "--max-image-size",
        type=int,
        default=DEFAULT_MAX_IMAGE_SIZE,
        metavar="N",
        help=(
            "vlm only: downscale thumbnails whose long side exceeds N px before "
            f"sending to ollama (default: {DEFAULT_MAX_IMAGE_SIZE}; 0 disables)"
        ),
    )
    run.add_argument(
        "--keep-alive",
        default="3h",
        metavar="DUR",
        help="vlm only: how long ollama keeps the model loaded between classifications (default: 3h)",
    )
    run.add_argument(
        "--crf",
        type=int,
        default=DEFAULT_CRF,
        metavar="N",
        help=f"libx264 quality, lower is better (default: {DEFAULT_CRF})",
    )
    run.add_argument(
        "--preset",
        default=DEFAULT_PRESET,
        help=f"libx264 encoding speed preset (default: {DEFAULT_PRESET})",
    )
    run.add_argument(
        "--encoder",
        choices=ENCODERS,
        default=DEFAULT_ENCODER,
        help=(
            "Video encoder: libx264 (CPU) or h264_nvenc (NVIDIA GPU; --crf "
            f"becomes its constant quality and --preset maps to p1-p7) "
            f"(default: {DEFAULT_ENCODER})"
        ),
    )
    run.add_argument(
        "--selective",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Only re-encode frames containing redactions and stream-copy the "
            "rest (faster, visually identical; default). Use --no-selective "
            "for a full re-encode with --crf/--preset applied everywhere"
        ),
    )
    run.add_argument(
        "--streaming",
        action="store_true",
        default=False,
        help="Use the single-pass streaming pipeline (experimental)",
    )
    _add_stability_args(run)
    _add_scene_args(run)
    for _p in (track, run):
        _p.add_argument(
            "--track-conf",
            type=float,
            default=0.25,
            metavar="F",
            help="Detection confidence threshold for tracking (default: 0.25)",
        )
        _p.add_argument(
            "--track-iou",
            type=float,
            default=0.7,
            metavar="F",
            help="NMS IoU threshold for tracking; lower values split overlapping people (default: 0.7)",
        )
        _p.add_argument(
            "--track-tracker",
            default="bytetrack.yaml",
            metavar="NAME",
            help="Ultralytics tracker config (botsort.yaml or bytetrack.yaml; default: bytetrack.yaml)",
        )
        _p.add_argument(
            "--orphans",
            action="store_true",
            default=False,
            help="After tracking, add tracklets for people detected but never tracked (extra pass)",
        )
    # Allow --config on every subcommand too, so users can write
    # `auto-redact run --config x.yaml ...`.
    for _sub in getattr(commands, "choices", {}).values():
        _sub.add_argument(
            "--config",
            metavar="FILE",
            default=None,
            help="YAML file with default values for any command-line option",
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
    if getattr(args, "streaming", False):
        result = run_pipeline_streaming(
            args.video,
            args.prompt,
            out,
            meta_path=args.meta,
            thumbs_dir=args.thumbs,
            force=args.force,
            sample_rate=args.sample_rate,
            device=args.device,
            track_workers=args.track_workers,
            half=not args.no_half,
            resume=not args.no_resume,
            padding=args.padding,
            isolate=args.isolate,
            sub_isolate=args.sub_isolate,
            closeup_area_fraction=args.closeup_area_fraction,
            head_weight=args.head_weight,
            body_weight=args.body_weight,
            edge_penalty=args.edge_penalty,
            edge_tolerance=args.edge_tolerance,
            distractor_weight=args.distractor_weight,
            same_person_iou=args.same_person_iou,
            isolate_dilate=args.isolate_dilate,
            classifier=args.classifier,
            model=args.model,
            threshold=args.threshold,
            classify_device=args.classify_device,
            ollama_url=args.ollama_url,
            timeout=args.timeout,
            classify_workers=args.classify_workers,
            think=args.think,
            max_image_size=args.max_image_size,
            keep_alive=args.keep_alive,
            crf=args.crf,
            preset=args.preset,
            selective=args.selective,
            encoder=args.encoder,
            max_gap=args.max_gap,
            smooth_window=args.smooth_window,
            bridge_iou=args.bridge_iou,
            min_track_frames=args.min_track_frames,
            scene_cuts=args.scene_cuts,
            scene_cut_threshold=args.scene_cut_threshold,
            progress=lambda message: print(message, flush=True),
            bar=progress_bar,
        )
        print(
            f"{result['tracklets']} tracklet(s), {result['redacted']} redacted; "
            f"wrote {result['out']}"
        )
        return 0
    summary = run_pipeline(
        args.video,
        args.prompt,
        out,
        meta_path=args.meta,
        thumbs_dir=args.thumbs,
        force=args.force,
        sample_rate=args.sample_rate,
        device=args.device,
        track_workers=args.track_workers,
        half=not args.no_half,
        resume=not args.no_resume,
        padding=args.padding,
        isolate=args.isolate,
        sub_isolate=args.sub_isolate,
        closeup_area_fraction=args.closeup_area_fraction,
        head_weight=args.head_weight,
        body_weight=args.body_weight,
        edge_penalty=args.edge_penalty,
        edge_tolerance=args.edge_tolerance,
        distractor_weight=args.distractor_weight,
        same_person_iou=args.same_person_iou,
        isolate_dilate=args.isolate_dilate,
        classifier=args.classifier,
        model=args.model,
        threshold=args.threshold,
        classify_device=args.classify_device,
        ollama_url=args.ollama_url,
        timeout=args.timeout,
        classify_workers=args.classify_workers,
        think=args.think,
        max_image_size=args.max_image_size,
        crf=args.crf,
        preset=args.preset,
        selective=args.selective,
        encoder=args.encoder,
        max_gap=args.max_gap,
        smooth_window=args.smooth_window,
        bridge_iou=args.bridge_iou,
        min_track_frames=args.min_track_frames,
        scene_cuts=args.scene_cuts,
        scene_cut_threshold=args.scene_cut_threshold,
        progress=lambda message: print(message, flush=True),
        bar=progress_bar,
    )
    cached = f" (reused: {', '.join(summary.reused)})" if summary.reused else ""
    print(
        f"{summary.tracklets} tracklet(s), {summary.redacted} redacted; "
        f"wrote {summary.out}{cached}"
    )
    return 0


def _cmd_track(args: argparse.Namespace) -> int:
    with progress_bar("track") as on_progress:
        meta = build_tracklets(
            args.video,
            sample_rate=args.sample_rate,
            device=args.device,
            on_progress=on_progress,
            workers=args.track_workers,
            half=not args.no_half,
            conf=args.track_conf,
            iou=args.track_iou,
            tracker=args.track_tracker,
            meta_path=args.meta,
            force=False,
            resume=not args.no_resume,
        )
    if args.orphans:
        from auto_redact.track import add_orphan_tracklets
        with progress_bar("orphan") as orphan_progress:
            meta = add_orphan_tracklets(meta, args.video, on_progress=orphan_progress)
        save_meta(args.meta, meta)
    if args.scene_cuts:
        from auto_redact.scene import apply_scene_cuts
        with progress_bar("scene") as scene_progress:
            meta, cuts, n_splits = apply_scene_cuts(
                meta, threshold=args.scene_cut_threshold, on_progress=scene_progress
            )
        save_meta(args.meta, meta)
        print(f"scene cuts: {len(cuts)} cut(s), {n_splits} split(s)", file=sys.stderr)
    save_meta(args.meta, meta)
    print(f"wrote {len(meta.tracklets)} tracklet(s) to {args.meta}")
    return 0


def _cmd_snapshot(args: argparse.Namespace) -> int:
    meta = load_meta(args.meta)
    contact_sheet = (
        None if args.no_contact_sheet else (args.contact_sheet or DEFAULT_CONTACT_SHEET)
    )
    with progress_bar("snapshot") as on_progress:
        select_snapshots(
            meta,
            thumbs_dir=args.thumbs,
            padding=args.padding,
            isolate=args.isolate,
            sub_isolate=args.sub_isolate,
            closeup_area_fraction=args.closeup_area_fraction,
            head_weight=args.head_weight,
            body_weight=args.body_weight,
            edge_penalty=args.edge_penalty,
            edge_tolerance=args.edge_tolerance,
            distractor_weight=args.distractor_weight,
            same_person_iou=args.same_person_iou,
            isolate_dilate=args.isolate_dilate,
            contact_sheet=contact_sheet,
            on_progress=on_progress,
            resume=not args.no_resume,
            meta_path=args.meta,
        )
    save_meta(args.meta, meta)
    isolated = sum(len(t.isolated_track_ids) for t in meta.tracklets)
    note = f" ({isolated} bystander(s) isolated)" if isolated else ""
    print(f"wrote {len(meta.tracklets)} thumbnail(s) to {args.thumbs}{note}")
    return 0


def _cmd_classify(args: argparse.Namespace) -> int:
    setup_logging()
    meta = load_meta(args.meta)
    siglip = None
    if args.classifier == "siglip":
        print("initializing classifier model", file=sys.stderr)
        siglip = SiglipClassifier(
            args.model or DEFAULT_SIGLIP_MODEL, args.threshold, args.device
        )
    else:
        print("waiting for ollama to load the model", file=sys.stderr)

    model = args.model or (DEFAULT_SIGLIP_MODEL if siglip is not None else DEFAULT_MODEL)
    sidecar = Path(args.meta).with_name(Path(args.meta).name + ".classify.json")
    key = _classify_key(args.prompt, args.classifier, model, args.threshold, args.think)
    try:
        stored = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        stored = None

    resume = not args.no_resume
    if not resume:
        for tracklet in meta.tracklets:
            tracklet.redact = None
            tracklet.score = None
        sidecar.unlink(missing_ok=True)
    elif stored is not None and stored != key:
        for tracklet in meta.tracklets:
            tracklet.redact = None
            tracklet.score = None
        sidecar.unlink(missing_ok=True)

    with progress_bar("classify") as on_progress:
        if siglip is not None:
            classify_snapshots(
                meta,
                args.prompt,
                classifier=siglip,
                on_progress=on_progress,
                resume=resume,
                meta_path=args.meta,
            )
        else:
            classify_snapshots(
                meta,
                args.prompt,
                model=args.model or DEFAULT_MODEL,
                ollama_url=args.ollama_url,
                timeout=args.timeout,
                workers=args.classify_workers,
                think=args.think,
                max_image_size=args.max_image_size,
                on_progress=on_progress,
                resume=resume,
                meta_path=args.meta,
            )
    save_meta(args.meta, meta)
    sidecar.write_text(json.dumps(key) + "\n", encoding="utf-8")
    flagged = sum(1 for t in meta.tracklets if t.redact)
    print(f"classified {len(meta.tracklets)} tracklet(s), {flagged} to redact")
    return 0


def _cmd_render(args: argparse.Namespace) -> int:
    meta = load_meta(args.meta)
    out = _resolve_out(args.out, meta.video.path)
    with progress_bar("render") as on_progress:
        out = render_redacted(
            meta,
            out,
            crf=args.crf,
            preset=args.preset,
            selective=args.selective,
            encoder=args.encoder,
            max_gap=args.max_gap,
            smooth_window=args.smooth_window,
            bridge_iou=args.bridge_iou,
            min_track_frames=args.min_track_frames,
            on_progress=on_progress,
        )
    print(f"wrote {out}")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Locate a --config option (if any) and the subcommand up front so the
    # YAML values can be applied as defaults before the full parse.
    config_path = None
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok == "--config":
            if i + 1 < len(argv):
                config_path = argv[i + 1]
                i += 2
                continue
        elif tok.startswith("--config="):
            config_path = tok.split("=", 1)[1]
        i += 1
    parser = build_parser()
    cfg: dict = {}
    if config_path is not None:
        try:
            import yaml
        except Exception as error:  # pragma: no cover
            print(f"auto-redact: error: pyyaml is required for --config: {error}", file=sys.stderr)
            return 1
        try:
            with open(config_path, encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle)
        except (OSError, ValueError) as error:
            print(f"auto-redact: error: could not read config {config_path}: {error}", file=sys.stderr)
            return 1
        if not isinstance(loaded, dict):
            print(f"auto-redact: error: config {config_path} must be a YAML mapping", file=sys.stderr)
            return 1
        cfg = loaded

    subcommand = None
    _KNOWN_COMMANDS = {"track", "snapshot", "classify", "render", "run"}
    for tok in argv:
        if tok in _KNOWN_COMMANDS:
            subcommand = tok
            break
    if subcommand is None and isinstance(cfg, dict) and cfg.get("command") in _KNOWN_COMMANDS:
        subcommand = cfg["command"]
        argv = [subcommand] + argv

    if cfg:
        # Inject required arguments from the config into argv so argparse's
        # `required=True` is satisfied (defaults alone do not count). Only
        # inject when the user did not already pass the flag on the CLI.
        _REQUIRED_KEYS = {"video", "prompt", "out", "meta"}
        def _flag_present(name: str) -> bool:
            flag = "--" + name.replace("_", "-")
            alt = "--" + name.replace("_", "-")  # same
            for tok in argv:
                if tok == flag or tok.startswith(flag + "="):
                    return True
            return False
        for key, value in cfg.items():
            if key not in _REQUIRED_KEYS:
                continue
            if _flag_present(key):
                continue
            flag = "--" + key.replace("_", "-")
            if isinstance(value, bool):
                injection = [flag if value else ("--no-" + key.replace("_", "-"))]
            else:
                injection = [flag, str(value)]
            # Insert right after the subcommand token so argparse routes the
            # option to the correct subparser.
            if subcommand is not None and subcommand in argv:
                idx = argv.index(subcommand) + 1
                argv = argv[:idx] + injection + argv[idx:]
            else:
                argv = injection + argv
        # Apply the rest as defaults so CLI flags still win.
        if subcommand is not None and hasattr(parser, "_subparsers"):
            for action in parser._subparsers._actions:
                if isinstance(action, argparse._SubParsersAction):
                    choices = getattr(action, "choices", None) or {}
                    if subcommand in choices:
                        choices[subcommand].set_defaults(**cfg)
                        break
        parser.set_defaults(**cfg)
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            return _cmd_run(args)
        if args.command == "track":
            return _cmd_track(args)
        if args.command == "snapshot":
            return _cmd_snapshot(args)
        if args.command == "classify":
            return _cmd_classify(args)
        if args.command == "render":
            return _cmd_render(args)
    except (
        ClassifyError,
        MetaError,
        RenderError,
        SnapshotError,
        SceneError,
        FileNotFoundError,
        ValueError,
    ) as error:
        print(f"auto-redact: error: {error}", file=sys.stderr)
        return 1
    raise AssertionError(f"unhandled command {args.command!r}")  # unreachable


if __name__ == "__main__":
    raise SystemExit(main())
