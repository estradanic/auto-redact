"""Classifier v2: SigLIP2 zero-shot scoring (issue #14).

The thumbnail is scored against the prompt plus a small set of negative
labels; the softmax probability of the prompt label is the match score,
and ``score >= threshold`` sets ``redact``. Lighter and faster than a
generative VLM, best for simple static attributes (clothing colour, ...).
"""

from __future__ import annotations

from pathlib import Path

from .classify import ClassifyError, Decision

DEFAULT_SIGLIP_MODEL = "google/siglip2-base-patch16-224"
DEFAULT_THRESHOLD = 0.5

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

    def __init__(
        self,
        model: str = DEFAULT_SIGLIP_MODEL,
        threshold: float = DEFAULT_THRESHOLD,
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
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device.isdigit():
            device = f"cuda:{device}"
        try:
            self._processor = AutoProcessor.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name).to(device).eval()
        except OSError as error:
            raise ClassifyError(
                f"could not load SigLIP model {self.model_name!r}: {error}"
            ) from error
        self._device = device

    def score(self, image_path: str | Path, prompt: str) -> float:
        """Probability (0-1) that the image matches ``prompt`` over the negatives."""
        import torch
        from PIL import Image

        self._load()
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
