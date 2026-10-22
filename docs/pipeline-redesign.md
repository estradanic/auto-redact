# Pipeline redesign: doing it in one pass

Plain-English plan for letting auto-redact chew through a video once instead of
four times, and maybe start spitting out output before the whole video is done.
Written for one person maintaining this. No code changes yet.

## What happens right now

When you run `auto-redact run`, the video gets read four separate times:

1. **Track** — find every person in every frame and link them into
   "this is person A from frame 12 to frame 400".
2. **Scene cuts** (optional) — find hard cuts and split those people so a
   person in shot 1 isn't confused with a person in shot 2.
3. **Snapshot** — for each person, pick their best-looking frame and save a
   thumbnail.
4. **Classify** — ask the model "does this thumbnail match the prompt?"
5. **Render** — read the video a last time, paint black boxes, write the file.

Steps 1, 2 and 5 each read the entire video. That's the waste. Classification
is the slow part (seconds per person), but reading the video three times is
dumb and it's what stops us from ever "streaming" anything.

## The one rule that makes streaming hard

This tool redacts people. If a frame *should* be redacted but we ship it
unredacted, that's a privacy failure. So:

> **We are not allowed to write a frame to the output until we know the
> answer for every person visible in that frame.**

That sounds obvious but it kills a lot of ideas. If someone is on screen for
the whole video, their best photo might be at minute 9, so we can't classify
them until minute 9, so frames from minute 1 can't be written until minute 9.
There is no clever trick that avoids this without either (a) risking a leak,
or (b) changing what "redact" means. Anyone who tells you otherwise is wrong.

So "streaming" here realistically means: **some frames come out early, the rest
come out as soon as it's safe, and the worst case is identical to today.** Not
magic, just not wasteful.

## What we *can* do in one forward pass

These parts don't need the future and can run while we read the video once:

- Tracking — the tracker already works frame by frame.
- Scene cuts — just comparing this frame's colours to the last frame's.
- Picking each person's best photo — we can keep "best so far" and update it
  as we go; the final answer is known when the person leaves or the video ends.
- Painting a frame — once we know who to redact, drawing boxes is instant.

Only "which photo is the best" and "does it match the prompt" need to wait.

## Options

### Option A — one pass, emit frames when it's safe (recommended)

Read the video once. Track, score photos, detect cuts as we go. When a person
disappears (or the video ends), crop their best photo, classify them, and
record the yes/no. Each frame goes into a waiting area. A frame is allowed out
only when *every* person visible in it has been classified. Write them out in
order as they become eligible.

Add a small optimisation: before the main pass, quickly track+classify the
first few seconds and the last few seconds. People who only appear in the
intro or outro (and long-lived people whose best photo happens to be near an
end) get decided early, which lets more of the middle stream out. This is the
"first and last chunks first" idea, and it's the only safe use of it — it only
adds early decisions, it never ships a frame before we know.

**Pros:** same results as today, one decode instead of three or four, partial
early output for short-lived people (cuts, bystanders, people who walk through).
**Cons:** if one person is on screen the whole time, nothing comes out until
the end. You need memory for the waiting frames.

### Option B — chop the video into chunks and do each chunk alone

Cut the video into ~3–5 second pieces. Run the whole pipeline on piece 1,
ship it, then piece 2, and so on. Process the first piece and the last piece
first so a player can start playing and learn the duration.

**Pros:** genuinely streamable; bounded memory; simple to build on what exists.
**Cons:** the same person in two different pieces may get different answers
(because their "best photo" differs per piece). That breaks "one decision per
person" and could redact them in one scene but not another. Annoying and
probably unacceptable for a redaction tool.

### Option C — classify as we go, fix mistakes later

Classify each person on their "best photo so far" the moment they leave, and
if a later better photo changes the answer, go back and re-render the already
shipped frames.

**Verdict:** no. Re-rendering shipped output is a mess and risks a window
where the wrong version is visible. Only useful for a live preview, not the
file.

### Option D — keep it two-pass but stop reading the video twice

Don't try to stream. Just merge the three "read the whole video" steps into
one pass that feeds all of them at once, then render from what we already
have in memory (or re-read once). Same results, less wasted work, no streaming
promise. This is the safe stepping stone.

### Option E — classify every frame

Classify each person in every frame and redact immediately. True streaming,
but 100–1000× the classifier cost. With a VLM at ~28 seconds per image that's
not a plan, it's a fantasy. Skip unless classifiers get ~100× faster.

## Recommended plan: Option A, in four shippable steps

Don't rewrite everything at once. Each step keeps the existing subcommands
(`track`, `snapshot`, `classify`, `render`) working and keeps tests passing.

**Step 1 — make tracking announce frames instead of returning a lump.**
Refactor so the tracker calls "I've seen frame N" with the detections, instead
of building the whole metadata at the end. Scene-cut detection moves into the
same loop. The existing `build_tracklets` becomes a thin wrapper. Chunked
parallel tracking stays, but its results get replayed in frame order. Nothing
visible changes; this just creates the single place where "a frame has been
seen" is announced.

**Step 2 — score best photos as we go, classify when a person leaves.**
Keep "best so far" per person, updated per frame. When a person disappears for
a while (or the video ends), crop their best frame, isolate them, save the
thumbnail, classify, store yes/no. Reuse the existing crop/isolate/classifier
code unchanged. Output should match the old pipeline exactly — test by
comparing metadata.

**Step 3 — the waiting area and the gate (this is the streaming part).**
Buffer frames that aren't ready yet. A frame is allowed out only when every
person in it has been classified. Add a small trailing buffer (a few frames)
so gap-filling and box smoothing still work before a frame ships — more delay,
but no leak risk. If the buffer gets too big, spill it to a temp file on disk
instead of RAM. Optionally do the bookend pass (first + last few seconds) first.

**Step 4 — chunked output ordering (optional, off by default).** Write the
first chunk, then the last chunk, then the middle, and stitch them with the
existing concat code. This is about how the file's bytes are ordered for a
player, not about redaction.

## Concerns and what to do about them

- **Leaks.** The gate handles it: never ship a frame whose people aren't all
  decided. No "best guess" shortcut, ever.
- **One person on screen the whole time.** Nothing ships until the end. That's
  the rule, not a bug. Spill the buffer to disk so RAM doesn't blow up.
- **Best photo might be in the future.** "Best so far" updates as we go; for a
  person present the whole video, the final pick happens at the end — same as
  today.
- **Isolation needs everyone in that frame.** Store which frame was best; decode
  just that one frame at the end (cheap).
- **Box smoothing needs future frames.** Keep a small trailing buffer before
  shipping, or accept a bit more flicker in streaming mode (flicker isn't a
  leak).
- **Scene cuts split a person mid-stream.** Close the old person and start a
  new one at the cut; carry over their best-photo state appropriately.
- **Tracking is parallel today.** Keep it parallel; just feed its results into
  the streaming layer in frame order.
- **Memory.** Spill to a temp rawvideo file past ~128 buffered frames (~1 GB
  at 720p), or fall back to two passes.
- **Caching/resume.** Keep `*.meta.json` as the final artifact. Add a small
  "last shipped frame + decided people" checkpoint so a crash can resume.
- **Subcommands.** Keep them; each becomes "run the pipeline up to this point
  and write the meta". Tests keep passing.
- **Tests.** Add a comparison: old pipeline vs new pipeline on the synthetic
  clips must produce identical metadata and identical output frames.

## The "first and last chunks first" idea specifically

Three ways to read it, in increasing cost:

1. **Delivery ordering only (Step 4).** Reorder a finished file's chunks for
   player startup. No pipeline change. Cheapest, zero risk.
2. **Decide boundary people early (Step 3 bookends).** Classify intro/outro
   people first to widen the streaming window. Moderate, safe.
3. **Per-chunk decisions (Option B).** Truly streamable, but inconsistent.
   Only if 1 and 2 aren't enough, and only behind an explicit flag with the
   accuracy trade-off documented.

Recommendation: do 1 and 2 first; revisit 3 only for a real live-preview use
case.

## Questions for you

1. Is partial early output enough, or do you need a guaranteed first frame by
   time T? (If the latter, Option B is the only route.)
2. Keep box smoothing in streaming mode (more delay) or drop it (more flicker)?
3. Spill threshold — 128 frames (~1 GB) okay?
4. Stream partial meta mid-run, or only write final meta?
5. Bookend pass on by default?

## Effort

Roughly 5–6 weeks solo, assuming the synthetic-clip tests get a golden
comparison added. Models and prompts don't change, so accuracy benchmarks stay
the same. Existing subcommands and the meta cache stay.

## What this deliberately doesn't do

- Doesn't change detection, tracking, or classifiers.
- Doesn't make classification cheaper — still one thumbnail per person.
- Doesn't promise real-time streaming on current hardware. It promises one
  decode pass and the earliest *safe* output, which is all the privacy rule
  allows.
- Doesn't remove the existing subcommands or the meta cache; they remain the
  debugging surface.