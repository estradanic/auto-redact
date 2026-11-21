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


def test_siglip_scores_each_leaf_then_combines_booleans(monkeypatch) -> None:
    from auto_redact.siglip import SiglipClassifier

    scores = {"a woman": 0.9, "bare thighs": 0.2, "lingerie": 0.7, "a child": 0.4}
    asked = []

    def fake(self, image_path, prompt):
        asked.append(prompt)
        return scores[prompt]

    monkeypatch.setattr(SiglipClassifier, "_score_leaf", fake)
    clf = SiglipClassifier(threshold=0.5, device="cpu")

    out = clf.classify("x.jpg", '"a woman" and ("bare thighs" or "lingerie")')
    assert out.redact is True and asked == ["a woman", "bare thighs", "lingerie"]
    assert out.score == 0.7  # fuzzy combination: min(0.9, max(0.2, 0.7))

    # `not` flips the thresholded leaf: 0.4 < 0.5 means "not a child" holds.
    assert clf.classify("x.jpg", 'not "a child"').redact is True
    assert clf.score_prompt("x.jpg", '"a woman" and "bare thighs"') == 0.2
    # A leaf that itself starts like an expression is still just text.
    scores["(left) person"] = 0.8
    assert clf.classify("x.jpg", '{"all": ["(left) person", "a woman"]}').redact is True
    # A plain prompt is still one leaf, untouched.
    asked.clear()
    assert clf.classify("x.jpg", "a woman").detail.startswith("score 0.900")
    assert asked == ["a woman"]


torch = pytest.importorskip("torch")


class _Batch(dict):
    def to(self, device):
        return _Batch({k: v.to(device) for k, v in self.items()})


class _FakeImageProcessor:
    def __init__(self, owner):
        self.owner = owner

    def __call__(self, images, return_tensors):
        # a deterministic "pixel" tensor derived from the image content
        import numpy as np

        arr = np.asarray(images, dtype="float32").reshape(-1)[:4]
        return _Batch(pixel_values=torch.tensor(arr).reshape(1, 4))


class _FakeProcessor:
    """Tokenises labels to a tensor of their lengths; counts what it was asked to do."""

    def __init__(self):
        self.image_processor = _FakeImageProcessor(self)
        self.text_calls = 0

    def __call__(self, text, images=None, padding=None, max_length=None, return_tensors=None):
        if images is not None:  # the direct (uncached) path asks for both at once
            raise AssertionError("cached path must not call the combined processor")
        self.text_calls += 1
        ids = torch.tensor([[float(len(label) % 7 + 1), float(len(label) % 5 + 1), 1.0, 2.0] for label in text])
        return _Batch(input_ids=ids)


class _Wrapped:
    def __init__(self, pooled):
        self.pooler_output = pooled


class _FakeModel:
    """Same maths as SigLIP: cosine similarity * scale + bias."""

    def __init__(self, wrap=False):
        self.logit_scale = torch.tensor(2.0)
        self.logit_bias = torch.tensor(-1.0)
        self.image_calls = self.text_calls = 0
        self.wrap = wrap

    def get_image_features(self, pixel_values):
        self.image_calls += 1
        out = pixel_values * 0.5 + 1.0
        return _Wrapped(out) if self.wrap else out

    def get_text_features(self, input_ids, attention_mask=None):
        self.text_calls += 1
        out = input_ids + 0.25
        return _Wrapped(out) if self.wrap else out

    def __call__(self, input_ids, pixel_values, **_):
        image = self.get_image_features(pixel_values)
        text = self.get_text_features(input_ids)
        image = getattr(image, "pooler_output", image)
        text = getattr(text, "pooler_output", text)
        image = image / image.norm(dim=-1, keepdim=True)
        text = text / text.norm(dim=-1, keepdim=True)
        logits = (image @ text.T) * self.logit_scale.exp() + self.logit_bias
        return type("Out", (), {"logits_per_image": logits})()


def _classifier(model):
    clf = SiglipClassifier(threshold=0.3, device="cpu")
    clf._model, clf._processor, clf._device = model, _FakeProcessor(), "cpu"
    return clf


def _image(tmp_path, name="a.png", color=(200, 30, 30)):
    from PIL import Image

    path = tmp_path / name
    Image.new("RGB", (16, 16), color).save(path)
    return path


@pytest.mark.parametrize("wrap", [False, True])
def test_cached_scores_equal_the_direct_model_run(tmp_path, wrap) -> None:
    from PIL import Image

    class Combined(_FakeProcessor):
        """What the direct path hands the model: tokens and pixels from one call."""

        def __call__(self, text, images=None, **kwargs):
            batch = _FakeProcessor.__call__(self, text, None, **kwargs)
            batch.update(self.image_processor(images, "pt"))
            return batch

    cached_clf = _classifier(_FakeModel(wrap=wrap))  # `wrap`: features come back inside a pooler_output object
    direct_clf = _classifier(_FakeModel())
    direct_clf._processor = Combined()
    path = _image(tmp_path)
    for leaf in ("a woman", "bare thighs", "lingerie"):
        cached = cached_clf._score_cached(path, leaf, torch, Image)
        direct = direct_clf._score_direct(path, leaf, torch, Image)
        assert cached == pytest.approx(direct, abs=1e-6)
        assert 0.0 < cached < 1.0


def test_each_image_and_each_leaf_is_encoded_once(tmp_path) -> None:
    model = _FakeModel()
    clf = _classifier(model)
    path = _image(tmp_path)
    for _ in range(3):
        for leaf in ("a woman", "bare thighs", "lingerie"):
            clf.score(path, leaf)
    assert model.image_calls == 1 and model.text_calls == 3
    clf.score(_image(tmp_path, "b.png", (10, 200, 10)), "a woman")
    assert model.image_calls == 2 and model.text_calls == 3  # new image; the leaf's text embedding is reused


def test_a_rewritten_image_file_is_not_served_from_the_cache(tmp_path) -> None:
    import os
    import time

    model = _FakeModel()
    clf = _classifier(model)
    path = _image(tmp_path, color=(200, 30, 30))
    before = clf.score(path, "a woman")
    _image(tmp_path, color=(5, 5, 250))  # same path, new content
    later = time.time_ns() + 10**9
    os.utime(path, ns=(later, later))
    after = clf.score(path, "a woman")
    assert model.image_calls == 2
    assert after != before


def test_the_image_cache_is_bounded(tmp_path) -> None:
    from auto_redact.siglip import IMAGE_CACHE_SIZE

    clf = _classifier(_FakeModel())
    for i in range(IMAGE_CACHE_SIZE + 4):
        clf.score(_image(tmp_path, f"{i}.png", (i * 9 % 255, 50, 50)), "a woman")
    assert len(clf._image_embeds) == IMAGE_CACHE_SIZE


def test_a_model_without_embedding_methods_falls_back_to_the_full_forward(tmp_path) -> None:
    class Plain:
        def __call__(self, **inputs):
            return type("Out", (), {"logits_per_image": torch.tensor([[2.0, 0.0, 0.0]])})()

    class Combined:
        def __call__(self, text, images, padding, max_length, return_tensors):
            return _Batch(input_ids=torch.zeros(3, 4), pixel_values=torch.zeros(1, 4))

    clf = SiglipClassifier(threshold=0.3, device="cpu")
    clf._model, clf._processor, clf._device = Plain(), Combined(), "cpu"
    path = _image(tmp_path)
    expected = float(torch.softmax(torch.tensor([2.0, 0.0, 0.0]), dim=-1)[0])
    assert clf.score(path, "a woman") == pytest.approx(expected)
    assert clf._embeddings_unsupported  # remembered, so it doesn't retry every call


def test_moving_the_model_clears_the_embeddings(tmp_path) -> None:
    model = _FakeModel()
    clf = _classifier(model)
    clf.score(_image(tmp_path), "a woman")
    assert clf._image_embeds and clf._text_embeds
    clf._clear_embeddings()
    assert not clf._image_embeds and not clf._text_embeds
