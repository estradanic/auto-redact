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
2. **Snapshot** — pick the best view of each person and crop a thumbnail.
3. **Classify** — ask a vision model whether each thumbnail matches the prompt.
4. **Render** — draw black boxes over matching people in the output video.

## Roadmap

Work is staged so each step is independently verifiable. See the
[issue tracker](https://github.com/estradanic/auto-redact/issues).

**Foundations**
- [x] #7 Scaffold project: package layout, pyproject, CLI stub
- [ ] #8 Video frame reader: ffmpeg decode + probe metadata
- [ ] #11 Metadata schema v1: save/load JSON (video info + tracklets)

**Detection**
- [ ] #9 Person detection: YOLO11n-pose per-frame boxes + keypoints
- [ ] #10 Tracking: ByteTrack association into per-person tracklets
- [ ] #12 Best snapshot selection v1: largest bbox per tracklet
- [ ] #17 Better snapshot scoring using keypoints (visibility, head, occlusion)

**Classification**
- [ ] #13 Classifier v1: VLM yes/no via ollama (Moondream)
- [ ] #14 Classifier v2: SigLIP2 zero-shot alternative

**Rendering / integration**
- [ ] #15 Render redacted video (full re-encode, draw black boxes)
- [ ] #16 End-to-end CLI: video + prompt -> redacted video
- [ ] #18 Selective rendering: only re-encode frames containing redactions
- [ ] #19 Docs: usage, model choices, hardware notes

## Hardware

Targets modest hardware (16 GB RAM, NVIDIA RTX A500 4 GB VRAM). Uses small models:
YOLO11n-pose + ByteTrack, and Moondream (~2B) or SigLIP2 for classification.
