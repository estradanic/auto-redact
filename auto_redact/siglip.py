"""Classifier v2: SigLIP2 zero-shot scoring (issue #14).

The thumbnail is scored against the prompt plus a small set of negative
labels; the softmax probability of the prompt label is the match score,
and ``score >= threshold`` sets ``redact``. Lighter and faster than a
generative VLM, best for simple static attributes (clothing colour, ...).
"""

from __future__ import annotations

import os
import time
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import logging
import threading
from pathlib import Path

from . import telemetry
from .classify import ClassifyError, Decision
from .config import DEFAULT_SIGLIP_MODEL, ClassifyConfig
from .prompts import Leaf, evaluate, parse_prompt

logger = logging.getLogger(__name__)

_COLORS = (
    "red", "orange", "yellow", "green", "blue", "purple",
    "pink", "black", "white", "gray", "brown",
)
_GENERIC_NEGATIVES = (
    "a photo of a person",
    "a photo of a person in plain clothes",
    "a photo of an empty scene",
)


IMAGE_CACHE_SIZE = 8  # recent image embeddings kept (one prompt expression scores one image at a time)


def positive_label(prompt: str) -> str:
    return f"a photo of {prompt.strip()}"


def derive_negatives(prompt: str) -> list[str]:
    """Auto-derive negative labels: other colours swapped in, plus generic ones."""
    words = prompt.strip().split()
    negatives: list[str] = []
    for i, word in enumerate(words):
        if word.lower() in _COLORS:
            for color in _COLORS:
                if color != word.lower():
                    swapped = words[:i] + [color] + words[i + 1:]
                    negatives.append(positive_label(" ".join(swapped)))
            break
    return negatives[:4] + list(_GENERIC_NEGATIVES)


class SiglipClassifier:
    """Zero-shot classifier using a SigLIP2 checkpoint via transformers."""

    thread_safe = True  # model/processor are stateless; safe to call from worker threads

    def __init__(
        self,
        model: str = DEFAULT_SIGLIP_MODEL,
        threshold: float = ClassifyConfig.threshold,
        device: str = "auto",
        vram_reserve_mb: int = 0,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ClassifyError(f"threshold must be in [0, 1], got {threshold}")
        self.model_name = model
        self.threshold = threshold
        self.device = device
        self.vram_reserve_mb = max(0, int(vram_reserve_mb))
        self._model = None
        self._processor = None
        self._device = None
        self._lock = threading.Lock()
        # Normalised image embeddings (a few recent images: every leaf of a
        # prompt expression scores the same one) and text embeddings (one per
        # leaf, constant across images). Both live on ``_device``.
        self._image_embeds: dict = {}
        self._text_embeds: dict = {}
        self._embeddings_unsupported = False
        # Set when "auto" placed us on the CPU because the GPU was too full
        # (or OOM'd on the way to it). Only such instances are eligible for
        # later promotion to the GPU.
        self._auto_cpu_pressure = False

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoProcessor

        device = self.device
        explicit = device != "auto"
        if device == "auto":
            if not torch.cuda.is_available():
                device = "cpu"
                logger.warning("siglip: CUDA not available, running on CPU (slow)")
            else:
                # Leave room for the VLM (and other GPU users such as the
                # tracker/NVENC) before taking the GPU for SigLIP: only go
                # to CUDA when enough VRAM is free to hold SigLIP (~1.5 GB)
                # plus the requested reserve.
                try:
                    free_bytes, _total = torch.cuda.mem_get_info()
                    free_mb = free_bytes / (1024 * 1024)
                    needed_mb = 1500 + self.vram_reserve_mb + 256
                    if free_mb >= needed_mb:
                        device = "cuda"
                    else:
                        device = "cpu"
                        self._auto_cpu_pressure = True
                        logger.info(
                            "siglip: only %.0f MB VRAM free (need ~%d MB); "
                            "running on CPU to keep the VLM on the GPU",
                            free_mb, needed_mb,
                        )
                except Exception:
                    device = "cuda"
        elif device.isdigit():
            device = f"cuda:{device}"
        try:
            self._processor = AutoProcessor.from_pretrained(self.model_name)
            model = AutoModel.from_pretrained(self.model_name).eval()
            start = time.perf_counter()
            try:
                model = model.to(device)
            except torch.cuda.OutOfMemoryError:
                if explicit:
                    raise
                logger.warning(
                    "siglip: CUDA out of memory loading %s (another process, e.g. "
                    "ollama, may hold the GPU); falling back to CPU", self.model_name,
                )
                torch.cuda.empty_cache()
                device = "cpu"
                self._auto_cpu_pressure = True
                model = model.to(device)
            self._model = model
        except OSError as error:
            raise ClassifyError(
                f"could not load SigLIP model {self.model_name!r}: {error}"
            ) from error
        self._device = device
        self._clear_embeddings()
        logger.info("siglip: %s loaded on %s (%.1fs to move)", self.model_name, device, time.perf_counter() - start)
        telemetry.note(f"siglip_device[{self.model_name}]", device)

    def promote_to_gpu(self) -> bool:
        """Try to move this instance onto the GPU if it fell back to CPU.

        Only instances that landed on the CPU because of VRAM pressure (auto
        mode) are eligible; explicitly CPU instances are left alone. Returns
        True if the model is on (or will load on) the GPU afterwards.

        Safe to call while other threads are scoring: it takes the instance
        lock, so an in-flight ``score`` finishes first.
        """
        import torch

        if not torch.cuda.is_available():
            return False
        with self._lock:
            if isinstance(self._device, str) and self._device.startswith("cuda"):
                return True
            if not getattr(self, "_auto_cpu_pressure", False):
                return False
            # Not loaded yet: just make future loads target the GPU.
            if self._model is None:
                self.device = "cuda"
                self._auto_cpu_pressure = False
                return True
            # Loaded on CPU: check there is room, then reload on the GPU so
            # an in-flight caller (none, thanks to the lock) can't observe a
            # half-moved module.
            try:
                free_bytes, _total = torch.cuda.mem_get_info()
                free_mb = free_bytes / (1024 * 1024)
                needed_mb = 1500 + 256
                if free_mb < needed_mb:
                    logger.info(
                        "siglip: only %.0f MB VRAM free (need ~%d MB); "
                        "keeping %s on CPU", free_mb, needed_mb, self.model_name,
                    )
                    return False
            except Exception:
                pass
            try:
                from transformers import AutoModel

                new_model = AutoModel.from_pretrained(self.model_name).eval()
                new_model = new_model.to("cuda")
            except Exception as error:
                logger.info(
                    "siglip: could not promote %s to GPU (%s); staying on CPU",
                    self.model_name, error,
                )
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                return False
            self._model = new_model
            self._device = "cuda"
            self.device = "cuda"
            self._clear_embeddings()
            self._auto_cpu_pressure = False
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            logger.info("siglip: promoted %s to cuda", self.model_name)
            telemetry.note(f"siglip_device[{self.model_name}]", "cuda")
            return True

    def score(self, image_path: str | Path, prompt: str) -> float:
        """Probability (0-1) that the image matches the single description ``prompt``."""
        return self._score_leaf(image_path, prompt)

    def score_prompt(self, image_path: str | Path, prompt: str) -> float | None:
        """Like :meth:`score`, but ``prompt`` may be an expression: the fuzzy
        combination (min / max / 1 - s) of its leaves' scores."""
        node = parse_prompt(prompt)
        if isinstance(node, Leaf):
            return self.score(image_path, node.text)
        return evaluate(node, lambda leaf: self._classify_leaf(image_path, leaf)).score

    def _score_leaf(self, image_path: str | Path, prompt: str) -> float:
        import torch
        from PIL import Image

        with self._lock:
            with telemetry.timed("siglip.load"):
                self._load()
            with telemetry.timed(f"siglip.score[{self._device}]"):
                return self._score(image_path, prompt, torch, Image)

    def _clear_embeddings(self) -> None:
        self._image_embeds.clear()
        self._text_embeds.clear()

    def _score(self, image_path, prompt, torch, Image) -> float:
        """Match probability for one description.

        The image is encoded once and its embedding reused by every leaf of a
        prompt expression, and each leaf's text embeddings are computed once
        for the whole run, so a leaf costs a dot product instead of a model
        forward pass. The numbers are identical to running the full model on
        (image, labels) each time; models without the embedding methods fall
        back to that.
        """
        if not self._embeddings_unsupported:
            try:
                return self._score_cached(image_path, prompt, torch, Image)
            except (AttributeError, TypeError) as error:
                logger.info("siglip: embedding cache unavailable (%s); scoring without it", error)
                self._embeddings_unsupported = True
                self._clear_embeddings()
        return self._score_direct(image_path, prompt, torch, Image)

    def _score_cached(self, image_path, prompt, torch, Image) -> float:
        def normalised(features):
            features = getattr(features, "pooler_output", features)  # newer transformers wrap the tensor
            return features / features.norm(p=2, dim=-1, keepdim=True)

        path = Path(image_path)
        stat = path.stat()
        key = (str(path), stat.st_mtime_ns, stat.st_size)
        image_embed = self._image_embeds.get(key)
        if image_embed is None:
            with Image.open(path) as image:
                image = image.convert("RGB")
            pixels = self._processor.image_processor(images=image, return_tensors="pt")
            pixels = {name: value.to(self._device) for name, value in pixels.items()}
            with torch.no_grad():
                image_embed = normalised(self._model.get_image_features(**pixels))
            if len(self._image_embeds) >= IMAGE_CACHE_SIZE:
                self._image_embeds.pop(next(iter(self._image_embeds)))
            self._image_embeds[key] = image_embed
        text_embed = self._text_embeds.get(prompt)
        if text_embed is None:
            labels = [positive_label(prompt)] + derive_negatives(prompt)
            tokens = self._processor(
                text=labels, padding="max_length", max_length=64, return_tensors="pt"
            ).to(self._device)
            text_inputs = {name: tokens[name] for name in ("input_ids", "attention_mask") if name in tokens}
            with torch.no_grad():
                text_embed = normalised(self._model.get_text_features(**text_inputs))
            self._text_embeds[prompt] = text_embed
        with torch.no_grad():
            logits = (image_embed @ text_embed.T) * self._model.logit_scale.exp() + self._model.logit_bias
            return float(torch.softmax(logits[0].float(), dim=-1)[0])

    def _score_direct(self, image_path, prompt, torch, Image) -> float:
        labels = [positive_label(prompt)] + derive_negatives(prompt)
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        inputs = self._processor(
            text=labels,
            images=image,
            padding="max_length",
            max_length=64,
            return_tensors="pt",
        ).to(self._device)
        with torch.no_grad():
            logits = self._model(**inputs).logits_per_image[0]
        return float(torch.softmax(logits.float(), dim=-1)[0])

    def classify(self, image_path: str | Path, prompt: str) -> Decision:
        """Score ``prompt``; for a prompt expression each leaf is scored and
        thresholded on its own, then combined (see :mod:`auto_redact.prompts`)."""
        return evaluate(
            parse_prompt(prompt), lambda leaf: self._classify_leaf(image_path, leaf), "siglip"
        )

    def _classify_leaf(self, image_path: str | Path, prompt: str) -> Decision:
        score = self.score(image_path, prompt)  # a leaf: overridable, as before
        return Decision(
            score >= self.threshold,
            score,
            f"score {score:.3f} vs threshold {self.threshold:g}",
        )
