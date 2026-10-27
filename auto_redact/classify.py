"""Classifiers: VLM yes/no via ollama (issue #13) behind a pluggable interface (#14).

The SigLIP zero-shot alternative lives in :mod:`auto_redact.siglip`.

For every tracklet's best-view thumbnail (see :mod:`auto_redact.snapshot`)
the model is asked whether the image shows a person matching the
runtime prompt. The reply is parsed into a bool and stored as the
tracklet's ``redact`` flag; the decision and the raw reply are logged.

Talks to ollama's ``/api/generate`` endpoint with the standard library
only, so no extra dependency is needed.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from . import telemetry
from .meta import Metadata, save_meta, save_meta_fast
from .track import _resolve_meta_path

logger = logging.getLogger(__name__)

# How many classifications between metadata saves (batching to avoid O(n^2)
# serialization). The final partial batch is always saved at the end.
SAVE_EVERY = 50

DEFAULT_MODEL = "gemma3:4b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_TIMEOUT = 120.0  # seconds per request; small VLMs can be slow on CPU
DEFAULT_WORKERS = 1  # concurrent ollama requests
DEFAULT_MAX_IMAGE_SIZE = 896  # downscale larger thumbnails before sending to ollama
# Keep the model resident between requests so cold loads don't dominate
# repeated classification calls.
DEFAULT_KEEP_ALIVE = "10m"
# Only the first yes/no word is parsed, so cap the reply and keep it deterministic.
# num_ctx must cover the image tokens (variable for some VLMs, e.g. qwen3-vl);
# keep it modest so the KV cache stays small on a small GPU.
GENERATION_OPTIONS = {"num_predict": 8, "temperature": 0, "num_ctx": 4096}

# With thinking on, the model reasons before answering; the budget must cover
# that reasoning or the visible reply comes back empty.
THINKING_NUM_PREDICT = 2048

PROMPT_TEMPLATE = (
    "Does this image show a person matching: '{prompt}'? Answer yes or no."
)

_YES_NO = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


class ClassifyError(RuntimeError):
    """Raised when classification cannot be performed."""


def build_question(prompt: str) -> str:
    return PROMPT_TEMPLATE.format(prompt=prompt)


def parse_yes_no(reply: str) -> bool | None:
    """Parse a model reply into True (yes) / False (no).

    The first standalone "yes"/"no" word wins ("Yes, she is." -> True).
    Returns None when the reply contains neither.
    """
    match = _YES_NO.search(reply)
    if match is None:
        return None
    return match.group(1).lower() == "yes"


def ask_ollama(
    image_path: str | Path,
    question: str,
    *,
    model: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    timeout: float = DEFAULT_TIMEOUT,
    think: bool = False,
    keep_alive: str = DEFAULT_KEEP_ALIVE,
    max_image_size: int = DEFAULT_MAX_IMAGE_SIZE,
) -> str:
    """Send one image + question to ollama and return the raw reply text.

    ``think`` enables the model's reasoning mode (for models that have one).
    It is slower; the reasoning itself is discarded and only the final reply
    is returned. It is sent explicitly either way, because some models think
    by default and would otherwise spend the whole reply budget on reasoning.
    """
    options = dict(GENERATION_OPTIONS)
    if think:
        options["num_predict"] = THINKING_NUM_PREDICT
    image_bytes = Path(image_path).read_bytes()
    if max_image_size and max_image_size > 0:
        try:
            from io import BytesIO

            from PIL import Image

            with Image.open(BytesIO(image_bytes)) as image:
                image = image.convert("RGB")
                if max(image.size) > max_image_size:
                    scale = max_image_size / max(image.size)
                    image = image.resize(
                        (max(1, round(image.width * scale)),
                         max(1, round(image.height * scale))),
                        Image.Resampling.LANCZOS,
                    )
                    buffer = BytesIO()
                    image.save(buffer, format="JPEG", quality=95)
                    image_bytes = buffer.getvalue()
        except Exception:
            image_bytes = Path(image_path).read_bytes()
    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    payload = json.dumps(
        {
            "model": model,
            "prompt": question,
            "images": [image_b64],
            "stream": False,
            "think": think,
            "keep_alive": keep_alive,
            "options": options,
        }
    ).encode("utf-8")
    url = ollama_url.rstrip("/") + "/api/generate"
    request = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace").strip()
        raise ClassifyError(
            f"ollama at {ollama_url} returned HTTP {error.code}: {detail or error.reason}"
        ) from error
    except (urllib.error.URLError, OSError) as error:
        raise ClassifyError(
            f"could not reach ollama at {ollama_url} (is it running?): {error}"
        ) from error
    except json.JSONDecodeError as error:
        raise ClassifyError(f"ollama sent invalid JSON: {error}") from error
    if not isinstance(body, dict) or not isinstance(body.get("response"), str):
        raise ClassifyError(f"unexpected ollama response: {body!r}")
    return body["response"]


@dataclass
class Decision:
    """One classifier verdict: ``redact`` (None = undecided), optional score."""

    redact: bool | None
    score: float | None = None
    detail: str = ""


class Classifier(Protocol):
    """Common interface: judge one thumbnail against the prompt."""

    def classify(self, image_path: str | Path, prompt: str) -> Decision: ...


class VlmClassifier:
    """Yes/no questions to a local VLM served by ollama."""

    thread_safe = True  # stateless: each call is an independent HTTP request

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        timeout: float = DEFAULT_TIMEOUT,
        think: bool = False,
        keep_alive: str = DEFAULT_KEEP_ALIVE,
        max_image_size: int = DEFAULT_MAX_IMAGE_SIZE,
    ) -> None:
        self.model = model
        self.ollama_url = ollama_url
        self.timeout = timeout
        self.think = think
        self.keep_alive = keep_alive
        self.max_image_size = max_image_size

    def classify(self, image_path: str | Path, prompt: str) -> Decision:
        with telemetry.timed("vlm.classify"):
            reply = ask_ollama(
                image_path,
                build_question(prompt),
                model=self.model,
                ollama_url=self.ollama_url,
                timeout=self.timeout,
                think=self.think,
                keep_alive=self.keep_alive,
                max_image_size=self.max_image_size,
            )
        return Decision(parse_yes_no(reply), None, f"reply: {reply!r}")



def _siglip_type():
    try:
        from .siglip import SiglipClassifier
        return SiglipClassifier
    except Exception:
        return type(None)


def classify_snapshots(
    meta: Metadata,
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    timeout: float = DEFAULT_TIMEOUT,
    think: bool = False,
    keep_alive: str = DEFAULT_KEEP_ALIVE,
    max_image_size: int = DEFAULT_MAX_IMAGE_SIZE,
    prefilter: str = "none",
    prefilter_model: str = None,
    prefilter_threshold: float = None,
    prefilter_device: str = "auto",
    classifier: Classifier | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    on_prefilter_progress: Callable[[int, int], None] | None = None,
    workers: int = DEFAULT_WORKERS,
    resume: bool = True,
    meta_path=None,
    save_path=None,
    checkpoint_path=None,
    checkpoint=None,
    output=None,
    out=None,
    path=None,
    file=None,
    destination=None,
    dest=None,
    **kwargs,
) -> Metadata:
    """Set ``redact`` (and ``score`` if available) on every tracklet.

    Uses ``classifier`` if given, else a :class:`VlmClassifier` built from
    ``model``/``ollama_url``/``timeout``. Updates ``meta`` in place and
    returns it. An undecided verdict leaves that tracklet's ``redact`` as
    None (and logs a warning).

    ``workers`` classifies that many tracklets concurrently, but only for
    classifiers that declare ``thread_safe`` (the ollama one does); others
    run sequentially. Results are identical either way.

    Raises:
        ClassifyError: empty prompt, a tracklet without a snapshot (run
            ``snapshot`` first), or the classifier failing.
        FileNotFoundError: a recorded snapshot file is missing.
    """
    if not prompt or not prompt.strip():
        raise ClassifyError("prompt must not be empty")
    missing = [
        t.track_id for t in meta.tracklets if t.best_snapshot_path is None
    ]
    if missing:
        raise ClassifyError(
            f"tracklet(s) {missing} have no snapshot; run `auto-redact snapshot` first"
        )
    if classifier is None:
        classifier = VlmClassifier(model, ollama_url, timeout, think, keep_alive, max_image_size)

    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        raise ValueError(f"workers must be an integer >= 1, got {workers!r}")
    for tracklet in meta.tracklets:
        path = Path(tracklet.best_snapshot_path)
        if not path.is_file():
            raise FileNotFoundError(f"no such snapshot file: {path}")

    # Collect any extra path-like keyword.
    for key, value in list(kwargs.items()):
        if value is None:
            continue
        low = key.lower()
        if isinstance(value, (str, Path, os.PathLike)) and (
            "meta" in low or "path" in low or "out" in low
            or "dest" in low or "save" in low or "checkpoint" in low
        ):
            if meta_path is None and save_path is None and checkpoint_path is None and output is None and out is None and path is None:
                meta_path = value
    target = _resolve_meta_path(
        meta_path=meta_path,
        save_path=save_path,
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
        output=output,
        out=out,
        path=path,
        file=file,
        destination=destination,
        dest=dest,
    )

    ckpt_path = (
        target.with_name(target.name + ".classify.ckpt.json")
        if target is not None else None
    )

    if not resume:
        for tracklet in meta.tracklets:
            tracklet.redact = None
            tracklet.score = None

    if resume and ckpt_path is not None and ckpt_path.is_file():
        try:
            records = json.loads(ckpt_path.read_text(encoding="utf-8")).get(
                "completed", []
            )
            by_id = {t.track_id: t for t in meta.tracklets}
            for record in records:
                tracklet = by_id.get(record.get("track_id"))
                if tracklet is None or tracklet.redact is not None:
                    continue
                tracklet.redact = record.get("redact")
                tracklet.score = record.get("score")
        except (OSError, ValueError, TypeError):
            pass

    prompt = prompt.strip()
    total = len(meta.tracklets)
    if not getattr(classifier, "thread_safe", False):
        workers = 1

    pending = [t for t in meta.tracklets if t.redact is None]
    already_done = total - len(pending)

    def apply(tracklet, decision: Decision) -> None:
        tracklet.redact = decision.redact
        tracklet.score = decision.score
        if decision.redact is None:
            logger.warning(
                "track %s: unparseable or undecided, redact left unset (%s)",
                tracklet.track_id, decision.detail,
            )
        else:
            logger.info(
                "track %s: redact=%s (%s)",
                tracklet.track_id, decision.redact, decision.detail,
            )

    newly_completed = 0
    total_pending = len(pending)

    def _maybe_save() -> None:
        if ckpt_path is None or newly_completed == 0:
            return
        if newly_completed % SAVE_EVERY == 0 or newly_completed == total_pending:
            try:
                completed = [
                    {"track_id": t.track_id, "redact": t.redact, "score": t.score}
                    for t in meta.tracklets
                    if t.redact is not None
                ]
                ckpt_path.write_text(
                    json.dumps({"completed": completed}, separators=(",", ":")),
                    encoding="utf-8",
                )
            except Exception:
                pass

    if on_progress is not None:
        on_progress(already_done, total)

    # Optional SigLIP pre-filter (VLM backend only): run SigLIP first and
    # let the VLM handle only the subset where SigLIP is least trustworthy.
    # "negatives" -> VLM re-checks SigLIP's negatives (catches false negatives)
    # "positives" -> VLM re-checks SigLIP's positives (catches false positives)
    prefilter = (prefilter or "none").lower()
    siglip = None
    if prefilter in ("negatives", "positives") and classifier is not None and not isinstance(classifier, _siglip_type()):
        try:
            from .siglip import SiglipClassifier, DEFAULT_SIGLIP_MODEL, DEFAULT_THRESHOLD
            siglip = SiglipClassifier(
                prefilter_model or DEFAULT_SIGLIP_MODEL,
                prefilter_threshold if prefilter_threshold is not None else DEFAULT_THRESHOLD,
                prefilter_device,
            )
        except Exception:
            siglip = None
            prefilter = "none"

    # Determine VLM subset.
    vlm_pending = list(pending)
    siglip_decisions: dict[int, Decision] = {}
    if siglip is not None and pending:
        total_pf = len(pending)
        if on_prefilter_progress is not None:
            on_prefilter_progress(0, total_pf)
        try:
            siglip._load()
        except Exception as error:
            logger.warning("siglip prefilter failed to load (%s); skipping it", error)
        for n, tracklet in enumerate(pending, 1):
            try:
                siglip_decisions[tracklet.track_id] = siglip.classify(
                    Path(tracklet.best_snapshot_path), prompt
                )
            except Exception as error:
                logger.warning("track %s: siglip prefilter error: %s", tracklet.track_id, error)
                siglip_decisions[tracklet.track_id] = Decision(None, None, "siglip error")
            if on_prefilter_progress is not None:
                on_prefilter_progress(n, total_pf)
        if prefilter == "negatives":
            vlm_pending = [
                t for t in pending
                if siglip_decisions.get(t.track_id) is not None
                and siglip_decisions[t.track_id].redact is False
            ]
        elif prefilter == "positives":
            vlm_pending = [
                t for t in pending
                if siglip_decisions.get(t.track_id) is not None
                and siglip_decisions[t.track_id].redact is True
            ]

    completed = already_done
    # Apply SigLIP decisions to non-VLM tracklets immediately.
    if siglip is not None:
        for tracklet in pending:
            if tracklet in vlm_pending:
                continue
            dec = siglip_decisions.get(tracklet.track_id)
            if dec is None:
                dec = Decision(None, None, "no siglip decision")
            apply(tracklet, dec)
            completed += 1
            newly_completed += 1
            if on_progress is not None:
                on_progress(completed, total)
            _maybe_save()

    if workers == 1:
        for tracklet in vlm_pending:
            apply(
                tracklet,
                classifier.classify(Path(tracklet.best_snapshot_path), prompt),
            )
            completed += 1
            newly_completed += 1
            if on_progress is not None:
                on_progress(completed, total)
            _maybe_save()
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    classifier.classify, Path(t.best_snapshot_path), prompt
                ): t
                for t in vlm_pending
            }
            for future in as_completed(futures):
                apply(futures[future], future.result())
                completed += 1
                newly_completed += 1
                if on_progress is not None:
                    on_progress(completed, total)
                _maybe_save()
    if ckpt_path is not None:
        try:
            ckpt_path.unlink(missing_ok=True)
        except Exception:
            pass
    return meta
