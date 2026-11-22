"""Prompt expressions: parsing, three-valued short-circuit evaluation, and the
classifiers' handling of them."""

import json

import pytest

from auto_redact.classify import Decision, VlmClassifier
from auto_redact.prompts import (
    Leaf,
    Op,
    evaluate,
    is_tree,
    leaves,
    normalize_prompt,
    parse_prompt,
)

A, B, C = Leaf("a"), Leaf("b"), Leaf("c")


def test_plain_text_is_one_leaf_verbatim() -> None:
    assert parse_prompt("person in a red shirt") == Leaf("person in a red shirt")
    # words that look like operators are just text when the prompt is plain
    assert parse_prompt("person in red and blue, or not wearing a hat") == Leaf(
        "person in red and blue, or not wearing a hat"
    )
    assert parse_prompt("and later, 'quoted' text") == Leaf("and later, 'quoted' text")
    assert parse_prompt("nothing to see") == Leaf("nothing to see")  # starts with "not", not an operator
    assert not is_tree("person in a red shirt")


def test_json_form() -> None:
    tree = parse_prompt('{"all": ["a woman", {"any": ["lingerie", "bare thighs"]}]}')
    assert tree == Op("all", (Leaf("a woman"), Op("any", (Leaf("lingerie"), Leaf("bare thighs")))))
    assert parse_prompt('{"not": "a child"}') == Op("not", (Leaf("a child"),))
    assert parse_prompt('{"all": ["only"]}') == Leaf("only")  # one child collapses


def test_expression_form_and_precedence() -> None:
    assert parse_prompt('"a" and "b" or "c"') == Op("any", (Op("all", (A, B)), C))
    assert parse_prompt('"a" and ("b" or "c")') == Op("all", (A, Op("any", (B, C))))
    assert parse_prompt("not 'a' and 'b'") == Op("all", (Op("not", (A,)), B))
    assert parse_prompt('"a" AND "b" And "c"') == Op("all", (A, B, C))  # flattened, case-insensitive
    assert parse_prompt(r'"say \"hi\""') == Leaf('say "hi"')


@pytest.mark.parametrize(
    "bad",
    [
        '"a" and',
        '("a" or "b"',
        '"a" "b"',
        '"a" xor "b"',
        '""',
        '{"all": []}',
        '{"all": "a"}',
        '{"maybe": ["a"]}',
        '{"all": ["a"], "any": ["b"]}',
        '["a", "b"]',
        "{not json",
        '{"all": [1]}',
    ],
)
def test_bad_prompts_raise_a_clear_error(bad) -> None:
    with pytest.raises(ValueError):
        parse_prompt(bad)


def test_yaml_structures_normalize_to_compact_json() -> None:
    structured = {"all": ["a woman", {"not": "a child"}]}
    text = normalize_prompt(structured)
    assert text == '{"all":["a woman",{"not":"a child"}]}'
    assert parse_prompt(text) == parse_prompt(json.dumps(structured))
    assert normalize_prompt("plain") == "plain"
    with pytest.raises(ValueError):
        normalize_prompt({"nope": []})


def test_leaves_lists_each_question_once() -> None:
    assert leaves(parse_prompt('("a" or "b") and not "a" and "c"')) == ["a", "b", "c"]


class Scripted:
    """Answers each leaf from a table and records what was asked."""

    def __init__(self, table):
        self.table, self.asked = table, []

    def __call__(self, leaf):
        self.asked.append(leaf)
        value = self.table[leaf]
        score = None if value is None else (0.9 if value else 0.1)
        return Decision(value, score, f"reply: {value}")


def run(prompt, table):
    ask = Scripted(table)
    return evaluate(parse_prompt(prompt), ask), ask.asked


def test_a_bare_leaf_passes_the_classifier_decision_through_untouched() -> None:
    decision, asked = run("a", {"a": True})
    assert decision == Decision(True, 0.9, "reply: True") and asked == ["a"]


def test_all_stops_at_the_first_no_and_any_at_the_first_yes() -> None:
    decision, asked = run('"a" and "b" and "c"', {"a": True, "b": False, "c": True})
    assert decision.redact is False and asked == ["a", "b"]
    decision, asked = run('"a" or "b" or "c"', {"a": False, "b": True, "c": True})
    assert decision.redact is True and asked == ["a", "b"]


def test_not_asks_the_positive_question_and_negates_the_answer() -> None:
    decision, asked = run('not "wearing a hat"', {"wearing a hat": True})
    assert decision.redact is False and asked == ["wearing a hat"]
    decision, _ = run('not "wearing a hat"', {"wearing a hat": False})
    assert decision.redact is True


def test_undecided_leaves_are_three_valued() -> None:
    table = {"a": None, "b": True, "c": False}
    assert run('"a" and "b"', table)[0].redact is None  # could be either
    assert run('"a" and "c"', table)[0].redact is False  # settled by the no
    assert run('"a" or "b"', table)[0].redact is True  # settled by the yes
    assert run('"a" or "c"', table)[0].redact is None
    assert run('not "a"', table)[0].redact is None


def test_scores_combine_fuzzily_and_the_detail_traces_what_was_asked() -> None:
    decision, _ = run('"a" and ("b" or "c")', {"a": True, "b": False, "c": True})
    assert decision.redact is True
    assert decision.score == pytest.approx(0.9)  # min(0.9, max(0.1, 0.9))
    assert decision.detail == "'a': yes (0.90); 'b': no (0.10); 'c': yes (0.90) -> yes"
    decision, _ = run('not "a"', {"a": True})
    assert decision.score == pytest.approx(0.1)
    decision, _ = run('"a" and "b"', {"a": True, "b": None})
    assert decision.score is None


def test_a_repeated_leaf_is_asked_once() -> None:
    decision, asked = run('"a" and ("b" or not "a")', {"a": True, "b": False})
    assert asked == ["a", "b"]
    assert decision.redact is False


class StubVlm(VlmClassifier):
    def __init__(self, answers):
        super().__init__()
        self.answers, self.questions = answers, []

    def _ask(self, image_path, prompt):
        self.questions.append(prompt)
        return Decision(self.answers[prompt], None, f"reply: {self.answers[prompt]}")


def test_vlm_classifier_asks_one_question_per_leaf_and_plain_prompts_stay_one() -> None:
    vlm = StubVlm({"a woman": True, "bare thighs": False, "lingerie": True})
    out = vlm.classify("x.jpg", '"a woman" and ("bare thighs" or "lingerie")')
    assert out.redact is True
    assert vlm.questions == ["a woman", "bare thighs", "lingerie"]
    vlm.questions.clear()
    out = vlm.classify("x.jpg", "a woman")
    assert vlm.questions == ["a woman"] and out.detail == "reply: True"


def test_one_images_questions_stay_together_when_workers_run_concurrently() -> None:
    import threading
    import time

    log = []

    class Recording(VlmClassifier):
        def __init__(self):
            super().__init__()

        def _ask(self, image_path, prompt):
            log.append((image_path, prompt))
            time.sleep(0.003)  # long enough for another thread to interleave if it could
            return Decision(False, None, "reply: no")

    clf = Recording()
    prompt = '"a" or "b" or "c" or "d"'  # every leaf answers no, so all four are asked
    threads = [threading.Thread(target=clf.classify, args=(f"img{i}", prompt)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(log) == 24
    for start in range(0, 24, 4):
        images = {image for image, _ in log[start:start + 4]}
        assert len(images) == 1, f"questions interleaved: {log[start:start + 4]}"
