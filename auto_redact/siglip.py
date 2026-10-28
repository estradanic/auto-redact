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
            self._auto_cpu_pressure = False
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            logger.info("siglip: promoted %s to cuda", self.model_name)
            telemetry.note(f"siglip_device[{self.model_name}]", "cuda")
            return True

    def score(self, image_path: str | Path, prompt: str) -> float:
        """Probability (0-1) that the image matches ``prompt`` over the negatives."""
        import torch
        from PIL import Image

        with self._lock:
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
