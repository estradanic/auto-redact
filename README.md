# auto-redact

AI-powered video redaction: redact people from videos based on a runtime-specified prompt.

Give it a video and a prompt like `"person in a red shirt"`; it finds matching people and
blacks them out.

> **Scope:** proof-of-concept CLI. Prompts are limited to *static attributes* of a person,
> because classification runs on a single best-view snapshot per tracked person. Activity
> detection (temporal reasoning) is a future direction.

## Pipeline

1. **Track** — detect people and their pose keypoints per frame, associate detections into
   per-person tracklets (bounding boxes + keypoints over time).
2. **Snapshot** — pick the best view of each person, crop a thumbnail, and black out any
   *other* people who appear in it (`--no-isolate` to keep raw crops), so the classifier can
   only judge the person the snapshot was taken for.
3. **Classify** — ask a vision model whether each thumbnail matches the prompt.
4. **Render** — draw black boxes over matching people in the output video.

## Install

Requires Python 3.10+ and `ffmpeg`/`ffprobe` on `PATH`.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

`pip install -e .` only installs into the active virtualenv, so `auto-redact` is
only on `PATH` while that venv is activated. To make it available from any
shell without activating the venv, symlink the entry point into a directory on
your `PATH` (e.g. `~/.local/bin`, which is on `PATH` by default on most
distros):

```bash
ln -s "$(pwd)/.venv/bin/auto-redact" ~/.local/bin/auto-redact
auto-redact --version
```

This reuses the existing venv (no extra downloads) and persists across reboots.
If you'd rather have a fully isolated install, use `pipx`:

```bash
pipx install /path/to/auto-redact
```

Model weights download on first use (YOLO11n-pose, and SigLIP2 if you use it). The default
classifier also needs [ollama](https://ollama.com) running locally with a vision model:

```bash
ollama pull gemma3:4b
```

## Usage

One command runs everything:

```bash
auto-redact run --video in.mp4 --prompt "person in a red shirt" --out redacted.mp4
```

This writes `redacted.mp4` and nothing else next to it. Working data (tracking results, thumbnails,
the stage cache) goes to a per-video directory, `~/.local/state/auto-redact/videos/<name>-<hash>/`
(`$AUTO_REDACT_STATE_DIR` or `$XDG_STATE_HOME` to move it), keyed by the source video, so
redacting the same video to different outputs shares its tracking. `--meta`/`--thumbs` pick
other locations, and metadata that older versions left next to the output
(`redacted.meta.json`) keeps being used in place. Re-running reuses a
cached stage only when the settings that produced it, and everything upstream of it, are
unchanged (tracked in `meta.json.cache.json`): a new prompt re-classifies only, a new
`--padding` re-snapshots and re-classifies, a new `--stride` re-tracks. Settings that cannot
change the result (devices, worker counts, timeouts) never invalidate anything. `--force`
redoes everything. Only frames with redactions are re-encoded; the rest is
stream-copied (faster, visually identical). Pass `--no-selective` for a full re-encode, where
`--crf`/`--preset` apply to the whole video.

Re-encoding uses `libx264` on the CPU by default; pass `--encoder h264_nvenc` to use the NVIDIA
GPU instead (needs an ffmpeg build with NVENC). `--crf` then sets NVENC's constant quality and
`--preset` maps to its `p1`-`p7` presets, or you can pass those directly. Output at a given
quality is a bit larger than libx264's, and NVENC rejects very small frames (under ~145 px).

Each stage is also its own subcommand sharing a `--meta` JSON file, useful for inspecting
intermediate results:

```bash
auto-redact track    --video in.mp4 --meta m.json
auto-redact snapshot --meta m.json --thumbs thumbs   # also writes thumbs/contact_sheet.jpg
auto-redact classify --meta m.json --prompt "person in a red shirt"
auto-redact render   --meta m.json --out redacted.mp4
```

Each stage shows a progress bar when run in a terminal (hidden when output is piped).

Check `thumbs/contact_sheet.jpg` to see what the classifier sees, and the `redact` flags in
`m.json` to see its decisions. Run any command with `--help` for all options.

### Choosing a classifier

`run` and `classify` both take `--classifier {vlm,siglip}`:

```bash
auto-redact run --video in.mp4 --prompt "person in a suit" --out redacted.mp4 \
    --classifier siglip --threshold 0.3
```

| | `vlm` (default) | `siglip` |
|---|---|---|
| Model | `gemma3:4b` via ollama (`--model`, `--ollama-url`) | `google/siglip2-base-patch16-224` via transformers (`--model`, `--device`; on `run`: `--classify-device`, since `--device` is for tracking) |
| How | Asks "does this image show a person matching ...? yes/no" | Scores the thumbnail against the prompt and auto-derived negative labels |
| Output | `redact` | `redact` and `score` (0-1); redacts if `score >= --threshold` (default 0.5) |
| Benchmark | 23/24 correct, ~28 s/image | 19/24 correct at 0.5, ~1.7 s/image on CPU (faster on GPU) |

Use the VLM for accuracy and subtle attributes; use SigLIP for speed on simple attributes. Its
default threshold is conservative, and lower values (~0.15-0.3) looked better on the small
benchmark. See [benchmarks/README.md](benchmarks/README.md).

### Prompt tips

- Describe a *static, visible attribute*: `"person in a red shirt"`, `"person wearing
  sunglasses"`, `"person with a backpack"`. Be concrete (colours, clothing, accessories).
- Check the contact sheet if results are off: an attribute that isn't visible in the chosen
  thumbnail (e.g. a shirt hidden by a coat) can't be matched.

### Configuration file

Any option can be set in a YAML file and passed with `--config`:

```bash
auto-redact run --config my.yaml --video in.mp4 --out redacted.mp4
```

CLI flags always override config values, and `--out` may be a directory (the
output keeps the input's filename). Example `my.yaml`:

```yaml
prompt: "person in a red shirt"
model: qwen3-vl:2b-instruct
head_weight: 0.1
closeup_area_fraction: 0.3
sub_isolate: true
encoder: h264_nvenc
selective: true
max_gap: 30
smooth_window: 9
```
## Model choices

- **YOLO11n-pose** detects people with 17 keypoints per frame. The nano model is fast and
  small; keypoints drive best-snapshot scoring (visibility, head, occlusion).
- **ByteTrack** (via Ultralytics) links detections into per-person tracklets. It's simple,
  motion-based, and needs no extra model.
- **VLM via ollama** (`gemma3:4b`): a generative model answering yes/no per thumbnail.
  Moondream was tried first and ruled out: ollama's `moondream` returns empty replies to every
  yes/no question.
- **SigLIP2** (base): a zero-shot image-text model; much lighter and faster, less accurate on
  small or partially occluded attributes.

## Hardware notes

Targets modest hardware: 16 GB RAM and an NVIDIA RTX A500 with 4 GB VRAM. 4 GB is tight: while
ollama holds `gemma3:4b` (~2.5 GB) on the GPU, SigLIP may not fit next to it. With the default
`auto` device, SigLIP tries the GPU first and, on CUDA out-of-memory, logs a warning and falls
back to the CPU (~1.7 s/image). Stop ollama (or let it unload the model) to keep SigLIP on the
GPU. Everything runs on CPU too, just slower.

`--encoder h264_nvenc` needs free VRAM for its own CUDA context. On a 4 GB card shared with
ollama and the SigLIP/YOLO models it can fail with `CUDA_ERROR_OUT_OF_MEMORY`. The staged render
frees cached GPU memory first, and any NVENC failure falls back to `libx264` with a warning;
with `--streaming` the other models are still loaded, so the fallback is more likely there.

## Defaults

| Component | Default | Override |
|---|---|---|
| Detection/tracking (YOLO11n-pose + ByteTrack) | GPU if CUDA is available, else CPU; FP16 on CUDA | `--device`, `--no-half` |
| SigLIP classifier (`--classifier siglip`) | **GPU if available**, else CPU; CPU fallback on CUDA OOM | `--classify-device` on `run`, `--device` on `classify` |
| SigLIP pre-filter (`--prefilter`) | **GPU if available**, else CPU; CPU fallback on CUDA OOM | `--prefilter-device` |
| VLM classifier | `gemma3:4b` via ollama (device is ollama's choice) | `--model`, `--ollama-url` |
| Classifier | `vlm`; SigLIP threshold 0.5 | `--classifier`, `--threshold` |
| Pre-filter | `none` | `--prefilter {negatives,positives}` |
| Undecided people (VLM reply with no yes/no, classifier error) | **redacted** | `--on-undecided {redact,keep,error}` |
| Encoder | `libx264` on CPU, selective re-encode | `--encoder h264_nvenc`, `--no-selective` |
| Instrumentation | on | `--no-stats`, `AUTO_REDACT_STATS=0` |

The device each SigLIP model actually ended up on is logged (`siglip: ... loaded on cuda`) and
recorded in the run stats.

## Instrumentation

Every command writes a run directory to `~/.local/state/auto-redact/runs/<timestamp>-<command>/`
(override the root with `AUTO_REDACT_STATS_DIR`) and prints its path when done:

- `run.log`: debug log with timestamps and thread names, plus periodic RSS/GPU memory samples.
- `stats.json`: environment (versions, GPU, ffmpeg), arguments, stage timings, per-operation
  timing aggregates (count/total/mean/max, e.g. `siglip.score[cuda:0]`, `vlm.classify`), peak
  RSS and GPU memory, the devices used, and any error with its traceback.

When reporting a slowdown or odd behavior, attach those two files. Disable with `--no-stats`
or `AUTO_REDACT_STATS=0`.

## Limitations

- **Static attributes only.** Classification uses a single best-view snapshot per person, so
  activities ("person running") and anything that changes over time aren't supported.
- **Crowded scenes.** Tracking can swap or split identities when people overlap or leave and
  re-enter the frame; one wrong snapshot or track then affects the person's whole track.
  Redaction is per track, so a person split into several tracks is classified once per
  track, and results may differ between them.
- **Model accuracy.** Small local models miss subtle attributes. Spot-check the output
  before relying on it for anything sensitive; this is a proof of concept.

## Reviewing and tuning from your feedback

```bash
auto-redact review --video in.mp4 --out redacted.mp4
```

opens a page in your browser (served locally on 127.0.0.1) over the video's stored results:

- **People:** every tracked person with the classifier's decision, score and reason (the VLM's
  reply, or the SigLIP score). Click *Redact* / *Keep* to overrule it. Your verdict is saved as a
  human `override` on the tracklet and in `labels.json`, applied by every later `run`, and never
  discarded by re-classifying.
- **Frame viewer:** scrub the video with the redaction boxes drawn on top (red = painted, green =
  the tracker's person boxes). Where someone is visible but not covered, drag a box around them
  (*leak*); where a box shouldn't be, flag it (*over*).
- **Tune:** *Analyze* explains each flagged frame (never detected, tracked but not flagged,
  dropped by `min_track_frames`, in a gap wider than `max_gap`, smoothing lag, or off the end of
  a track) and which setting would fix it. It then finds the render settings that cover your
  leaks with the least extra black area and, if you labeled people, the SigLIP `--threshold`
  that matches them (misses are weighed before extra boxes). Sliders preview any setting on your
  flagged frames before you commit; *Try a classifier* scores a prompt or model on just the
  people you labeled. Suggestions are shown with their evidence and are never applied
  automatically; copy the flags, or use *Render now*.

Labels are tied to the tracking they were made on; after a re-track they are reported as stale
and ignored. Frames the detector never saw a person in can't be fixed by a render setting, and
the page says so and points at `--track-conf`, `imgsz` or `--orphans` instead.

## Managing stored data

`auto-redact clean` lists what is stored (per-video workspaces and run logs, with sizes and
whether the source video still exists). Select what to delete with `--all`, `--missing` (video
gone), `--video PATH`, `--older-than DAYS` (alone, or to narrow another selector) and `--runs`
(run logs); `--dry-run` previews, and it asks before deleting unless you pass `--yes`.
Workspaces holding review labels are hand-made work and are skipped unless you add `--labeled`.

## Tuning options

`--help` lists the options you are likely to change. The snapshot-scoring weights
(`head_weight`, `body_weight`, `edge_penalty`, `edge_tolerance`, `distractor_weight`,
`same_person_iou`, `isolate_dilate`, `closeup_area_fraction`) are internal tuning and are left
out of it, but every option can be set with `--set NAME=VALUE` (repeatable, also accepts
`snapshot.head_weight`-style names) or as a key in a `--config` file:

```bash
auto-redact run ... --set head_weight=0.1 --set closeup_area_fraction=0.3
```

## Code layout

Every tunable lives in one place, [`auto_redact/config.py`](auto_redact/config.py): a frozen
dataclass per stage (`TrackConfig`, `SnapshotConfig`, `ClassifyConfig`, `RenderConfig`). Stage
functions take a config object (keyword overrides are accepted too), and the command-line
options and their help text are generated from the same fields, so a new setting is one new
field. Fields marked `cache=False` are execution-only and stay out of the stage cache keys.

| Module | Role |
|---|---|
| `config`, `cache` | settings; stage fingerprints that decide what a re-run can reuse |
| `track`, `scene` | detection, tracking, scene-cut splitting |
| `snapshot` | best-view selection and bystander isolation |
| `classify`, `siglip` | classifier backends (`protocols.Classifier`), `make_classifier` |
| `boxes` | which boxes to paint on which frames (gap filling, smoothing, bridging) |
| `encode`, `segments`, `render` | encoder backends (`protocols.Encoder`), selective GOP copy, orchestration |
| `geometry` | pure box math shared by the above |
| `detectors` | the only module that imports ultralytics: `PersonTracker` / `PersonDetector` |
| `workspace`, `labels` | where run data lives and `clean`; your review verdicts |
| `tune`, `review` | pure tuning logic (diagnosis, setting/threshold search); the review server and page |
| `meta` | the metadata document: small JSON summary + compressed `.npz` arrays |
| `pipeline`, `streaming`, `cli` | staged and single-pass runners; the command line |

The metadata JSON holds the video info, scene cuts and each tracklet's snapshot and
classification fields (edit `redact` by hand if you like); the bulk per-frame boxes and
keypoints live in the `.npz` beside it. Older single-file (schema v1) metadata still loads, and
a metadata file from before the stage cache existed is trusted once and checked from then on.

## Development

Run the test suite from the repo root:

```bash
pip install pytest
pytest
```
