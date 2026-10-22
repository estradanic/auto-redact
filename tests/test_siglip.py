"""Tests for auto_redact.siglip (#14); model loading is not exercised here."""

import pytest

from auto_redact.classify import ClassifyError
from auto_redact.siglip import SiglipClassifier, derive_negatives, positive_label


def test_negatives_swap_colors_and_add_generic() -> None:
    negatives = derive_negatives("person in a red shirt")
    assert "a photo of person in a blue shirt" in negatives
    assert "a photo of person in a red shirt" not in negatives
    assert "a photo of a person" in negatives


def test_negatives_without_color_are_generic_only() -> None:
    assert derive_negatives("person with a backpack") == [
        "a photo of a person",
        "a photo of a person in plain clothes",
        "a photo of an empty scene",
    ]


def test_positive_label() -> None:
    assert positive_label(" person ") == "a photo of person"


def test_threshold_applied_to_score(tmp_path) -> None:
    classifier = SiglipClassifier(threshold=0.6)
    classifier.score = lambda path, prompt: 0.7
    decision = classifier.classify(tmp_path / "x.jpg", "p")
    assert decision.redact is True and decision.score == 0.7
    classifier.score = lambda path, prompt: 0.5
    assert classifier.classify(tmp_path / "x.jpg", "p").redact is False


def test_bad_threshold_rejected() -> None:
    with pytest.raises(ClassifyError, match="threshold"):
        SiglipClassifier(threshold=1.5)
