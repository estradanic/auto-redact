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

Which presets are worth it was measured on 5 minutes of real footage (identical redactions
every time): `--preset veryfast` cut the whole run by ~19% against `medium`, with SSIM 0.994
against the `medium` output and a smaller file; `faster` saved ~9% at SSIM 0.995. NVENC was
faster still but its output was 46% larger with SSIM 0.975, so it needs tuning before it is a
good default. Capping x264's threads makes the tracker faster but the *run* slower (the
re-encode becomes the long pole), so the encoder is left to use every core.

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
| Benchmark | 23/24 correct, ~28 s/image on the 4 GB laptop GPU (~0.1 s/image on an RX 7900 XT eGPU, accuracy not re-measured) | 19/24 correct at 0.5, ~1.7 s/image on CPU (faster on GPU) |

Use the VLM for accuracy and subtle attributes; use SigLIP for speed on simple attributes. Its
default threshold is conservative, and lower values (~0.15-0.3) looked better on the small
benchmark. See [benchmarks/README.md](benchmarks/README.md).

### Prompt expressions

When a rule is really several conditions, a prompt can be a boolean expression over yes/no
questions instead of one long description (small models often handle a pile of clauses worse
than a few short questions):

```yaml
prompt:
  all:
    - "a woman"
    - any:
        - "wearing underwear or lingerie"
        - "bare midriff"
        - "bare thighs above the knee"
    - not: "a child"
```

On the command line, the same rule as JSON (`--prompt '{"all": ["a woman", {"any": [...]}]}'`)
or as text: `--prompt '"a woman" and ("bare thighs" or "lingerie") and not "a child"'`. In the text
form every description is quoted; `not` binds tighter than `and`, which binds tighter than
`or`. A prompt that doesn't start with `{`, a quote, a parenthesis or `not "` is one
description exactly as before.

- Every leaf is asked as an ordinary positive question; `not` flips the answer in code, so the
  model is never given a negated question.
- Evaluation stops as soon as the result is settled, in the order written: `all` at the first
  "no", `any` at the first "yes". Put the cheapest or most selective leaf first (`"a woman"`
  settles most men in one call). A leaf the classifier can't decide makes the result undecided
  unless another leaf already settles it.
- Cost grows with the leaves actually asked. With the VLM that is one call each; SigLIP
  (including the `--prefilter`) scores each leaf on its own and compares it with the threshold
  first, so a `not` means "scores below the threshold". The tree's `score` is a fuzzy
  combination (min / max / 1 - s) for display and threshold tuning.
- The verdict's reason lists the leaves that were asked, e.g.
  `'a woman': yes; 'bare thighs': no (0.20) -> no`, and `stats.json` counts each leaf's answers
  (`counts["prompt.leaf_verdicts"]`) so you can see which leaves prune and reorder them.

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
### Pans and zooms inside a shot

A person gets one snapshot and one verdict. A camera move without a cut can change what is
visible inside a single track (hips and thighs, then a pan up to the face), and the snapshot
that wins is usually the face view, so the earlier part is judged by proxy and can leak.
`--view-split` (`view_split: true`) watches which body regions the pose model can see (head,
shoulders, hips, legs) and starts a new tracklet when two or more appear or vanish, so each
view gets its own snapshot and verdict. The split is placed where the change began, not where
it was noticed, and the few frames before it are held back from snapshot selection so they
cannot win on the wrong side. A view shorter than `--view-min-frames` (12 observations) never
splits; a tail that short (a person leaving as the camera moves) skips its own snapshot and
inherits the verdict before it, because the renderer drops redacted tracklets under
`min_track_frames`. `--view-min-regions` (2) sets how large a change counts. It is off by
default, and turning it on re-tracks (the settings are part of the tracking cache key only
once changed). `stats.json` counts `view_split` splits and inherited tails.

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

### Running the VLM on an eGPU

The VLM stage only talks to ollama over HTTP, so it uses whatever GPU ollama uses; no
`auto-redact` setting is involved. Putting ollama on a separate GPU (for example an AMD
RX 7900 XT in a Thunderbolt/USB4 enclosure) frees the laptop GPU's VRAM for YOLO, SigLIP and
NVENC, which run through PyTorch/CUDA and stay on the NVIDIA card. On the reference setup
(ThinkPad P16s Gen 2, ADT-Link UT4G enclosure, RX 7900 XT with 20 GB over a 40 Gb/s link),
`gemma3:4b` answers a yes/no question in about 0.1 s once loaded, and larger VLMs fit
(`--model`).

On Linux with an AMD card:

1. Install the Vulkan driver: `sudo pacman -S vulkan-radeon lib32-vulkan-radeon` (or your
   distro's `mesa-vulkan-drivers`). Check `vulkaninfo --summary` lists the card.
2. Authorize the enclosure if needed (`boltctl list`, `boltctl enroll --policy auto <uuid>`).
3. Enable ollama's Vulkan backend and pin it to the eGPU, for example with a systemd drop-in
   (`/etc/systemd/system/ollama.service.d/egpu.conf`):

   ```ini
   [Service]
   Environment="OLLAMA_VULKAN=1"
   Environment="CUDA_VISIBLE_DEVICES=-1"
   Environment="GGML_VK_VISIBLE_DEVICES=1"
   ```

   `GGML_VK_VISIBLE_DEVICES` is the Vulkan device index (check the `inference compute` lines in
   `journalctl -u ollama`); `CUDA_VISIBLE_DEVICES=-1` keeps ollama off the NVIDIA card. Run
   `sudo systemctl daemon-reload && sudo systemctl restart ollama`, then confirm
   `ollama ps` shows `100% GPU`. With the eGPU unplugged this configuration falls back to
   the CPU, so remove the drop-in to go back to the NVIDIA card.

Ollama's ROCm build is an alternative to Vulkan on supported AMD cards.

The SigLIP pre-filter (`--prefilter`) normally keeps ~2.5 GB of VRAM free for the VLM before it
takes the laptop GPU, which on a 4 GB card pushes it onto the CPU. This is automatic: before
loading SigLIP, `auto-redact` loads the VLM in ollama and checks whether an ollama process holds
memory on the NVIDIA GPU (`nvidia-smi`). If ollama runs on another GPU, on the CPU, or on a
remote host (`--ollama-url`), no VRAM is reserved and the pre-filter uses the NVIDIA GPU. If it
can't tell (no `nvidia-smi`, ollama unreachable) it reserves as before. Override with
`--vlm-vram-reserve-mb MB` (`0` never reserves), or force the device with `--prefilter-device 0`.
The decision is logged (`prefilter: ollama is not using this GPU ...`).

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
| VLM classifier | `gemma3:4b` via ollama (device is ollama's choice, see [Running the VLM on an eGPU](#running-the-vlm-on-an-egpu)) | `--model`, `--ollama-url` |
| Classifier | `vlm`; SigLIP threshold 0.5 | `--classifier`, `--threshold` |
| Pre-filter | `none` | `--prefilter {negatives,positives}` |
| Undecided people (VLM reply with no yes/no, classifier error) | **redacted** | `--on-undecided {redact,keep,error}` |
| Encoder | `libx264` on CPU, selective re-encode | `--encoder h264_nvenc`, `--no-selective` |
| Instrumentation | on | `--no-stats`, `AUTO_REDACT_STATS=0` |

The device each SigLIP model actually ended up on is logged (`siglip: ... loaded on cuda`) and
recorded in the run stats.

## Where the time goes

Tracking (detection + ByteTrack), classification and re-encoding run concurrently and the
slowest sets the pace, so speeding up one stage only helps until another becomes the limit. On a
5-minute segment the stages were roughly: tracking 138 s, classification 135 s (one worker),
re-encode the rest. Things that were measured and are *not* worth changing: FP32 instead of FP16
(slower), a lower `imgsz` (no gain), `--stride 2` (changes results), other trackers (ByteTrack
gave the fewest, longest tracks), and a different GPU for the tracker (the nano model is
CPU-bound; a 7900 XT was no faster). Things that were worth doing and are built in: SigLIP scores
every leaf of a prompt from one image embedding and cached text embeddings (9x faster, identical
scores), the scene-cut histogram is computed after shrinking the frame, and checkpoint crops are
written on a background thread.

### Classification workers and the VLM

On a long episode with many people, classification (not tracking) is usually the slow stage.
The ollama server answers one vision request at a time (the model runs in a single slot even with
`OLLAMA_NUM_PARALLEL` raised), and each follow-up question about the *same* image is much cheaper
than the first because the server keeps the image's encoding (about 0.3 s against 1.1 s per call).
So the VLM's questions about one image are kept together, and more than one worker helps only by
overlapping the other per-person work (SigLIP, thumbnails, isolation) with the VLM:

| Pool-scene slice, 7,200 frames | stage time |
|---|---|
| 1 worker | 451 s |
| `--classify-workers 2` | **366 s** (-19%; now bound by the VLM itself) |
| `--classify-workers 3` | 361 s (no further gain) |

Use 2 workers with the VLM; raising `OLLAMA_NUM_PARALLEL` does not help and, with workers that
interleave images, made a full episode 84% slower. Verdicts were identical in every setting.

Over a whole episode (62,281 frames, 4,510 tracklets, one long VLM-heavy pool scene) the gain was
smaller than on the slice: 1,581 s to 1,480 s, with peak memory down from 8.2 GB to 5.0 GB and all
4,510 verdicts unchanged. Two things limit it: the backlog cap stops tracking from running ahead
through the quieter minutes while the VLM works through the dense scene (tracking paused for 304 s
in total), and a second worker's CPU work slows the tracker (12 to 15 ms per frame). A larger
`--classify-backlog` should trade memory for time; that was not measured.
`--classify-backlog N` (default 256) pauses tracking while more than N closed tracklets wait for a
verdict, so a slow classifier cannot pile up snapshot crops in memory (an episode with a long
crowd scene reached 8-9 GB without it); `0` removes the limit. The run stats show where time is
spent: `decide.total` (summed over workers), `vlm.lock_wait` and `decide.backpressure_wait`.

## Instrumentation

Every command writes a run directory to `~/.local/state/auto-redact/runs/<timestamp>-<command>/`
(override the root with `AUTO_REDACT_STATS_DIR`) and prints its path when done:

- `run.log`: debug log with timestamps and thread names, plus periodic RSS/GPU memory samples.
- `stats.json`: environment (versions, GPU, ffmpeg), arguments, stage timings, per-operation
  timing aggregates (count/total/mean/max, e.g. `siglip.score[cuda:0]`, `vlm.classify`), peak
  RSS and GPU memory, the devices used, and any error with its traceback. With a VLM
  pre-filter it also records how people were decided: `dists` holds the pre-filter score
  distribution (10 bins over 0-1; `prefilter.score` for everyone, `prefilter.score[sent_to_vlm]`
  for those handed on), and `counts` holds who decided (`prefilter.routed`), the verdicts per
  decider (`decisions[prefilter]`, `decisions[vlm]`) and the VLM's verdict per pre-filter score
  band (`vlm.verdict_by_prefilter_band`, e.g. `0.3-0.4:keep`), which shows whether lowering the
  pre-filter threshold is finding anything. The `prefilter` note records the mode, model and
  threshold used.
- `decisions.jsonl`: one line per decided person (`track`, `prefilter_score`, `route`,
  `verdict`, the classifier's `detail`), for digging into individual calls.

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
