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
from pathlib import Path

from . import telemetry
from .classify import ClassifyError, Decision
from .config import DEFAULT_SIGLIP_MODEL, ClassifyConfig

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
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ClassifyError(f"threshold must be in [0, 1], got {threshold}")
        self.model_name = model
        self.threshold = threshold
        self.device = device
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoProcessor

        device = self.device
        explicit = device != "auto"
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cpu":
                logger.warning("siglip: CUDA not available, running on CPU (slow)")
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
                model = model.to(device)
            self._model = model
        except OSError as error:
            raise ClassifyError(
                f"could not load SigLIP model {self.model_name!r}: {error}"
            ) from error
        self._device = device
        logger.info("siglip: %s loaded on %s (%.1fs to move)", self.model_name, device, time.perf_counter() - start)
        telemetry.note(f"siglip_device[{self.model_name}]", device)

    def score(self, image_path: str | Path, prompt: str) -> float:
        """Probability (0-1) that the image matches ``prompt`` over the negatives."""
        import torch
        from PIL import Image

        with telemetry.timed("siglip.load"):
            self._load()
        with telemetry.timed(f"siglip.score[{self._device}]"):
            return self._score(image_path, prompt, torch, Image)

    def _score(self, image_path, prompt, torch, Image) -> float:
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
        score = self.score(image_path, prompt)
        return Decision(
            score >= self.threshold,
            score,
            f"score {score:.3f} vs threshold {self.threshold:g}",
        )
