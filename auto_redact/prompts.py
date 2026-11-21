"""Prompts as boolean expressions over yes/no questions.

A prompt is normally one description ("person in a red shirt"), which is a
single question per person. For rules that are really several conditions it
can instead be an expression::

    {"all": ["a woman", {"any": ["wearing lingerie", "bare thighs"]}]}
    "a woman" and ("wearing lingerie" or "bare thighs") and not "a child"

``all`` / ``any`` / ``not`` (or ``and`` / ``or`` / ``not`` in the text form,
``not`` binding tighter than ``and`` and ``and`` tighter than ``or``). Every
leaf is asked as an ordinary *positive* question, and ``not`` is applied to
the answer in code, so the model never sees a negated question.

Evaluation short-circuits in the written order (put the cheap or most
selective leaf first), and a leaf the classifier could not decide makes the
result undecided unless another leaf already settles it.

Plain prompts are unchanged: anything that is not JSON (``{...}``) or an
expression (starting with a quote, a parenthesis, or ``not`` followed by one
of those) is one leaf, verbatim.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

from . import telemetry


@dataclass(frozen=True)
class Leaf:
    text: str


@dataclass(frozen=True)
class Op:
    kind: str  # "all" | "any" | "not"
    children: tuple["Node", ...]


Node = Leaf | Op

_KINDS = ("all", "any", "not")


def from_object(obj) -> Node:
    """Build a tree from parsed JSON/YAML: strings are leaves, one-key mappings are operators."""
    if isinstance(obj, str):
        text = obj.strip()
        if not text:
            raise ValueError("prompt leaf is empty")
        return Leaf(text)
    if isinstance(obj, dict):
        if len(obj) != 1 or next(iter(obj)) not in _KINDS:
            raise ValueError(
                f"a prompt operator is a mapping with one key of {', '.join(_KINDS)}, got {obj!r}"
            )
        kind, value = next(iter(obj.items()))
        if kind == "not":
            return Op("not", (from_object(value),))
        if not isinstance(value, list) or not value:
            raise ValueError(f"'{kind}' needs a non-empty list of prompts, got {value!r}")
        children = tuple(from_object(v) for v in value)
        return children[0] if len(children) == 1 else Op(kind, children)
    raise ValueError(f"a prompt is text or an all/any/not mapping, got {obj!r}")


_TOKEN = re.compile(
    r"""\s*(?:(?P<str>"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')|(?P<paren>[()])|(?P<word>[A-Za-z]+))"""
)


def _tokenize(text: str) -> list[tuple[str, str]]:
    tokens, pos = [], 0
    while pos < len(text):
        if not text[pos:].strip():
            break
        match = _TOKEN.match(text, pos)
        if match is None:
            raise ValueError(f"unexpected text at {text[pos:].strip()[:20]!r} in prompt expression")
        pos = match.end()
        if match.group("str") is not None:
            raw = match.group("str")
            tokens.append(("str", raw[1:-1].replace("\\" + raw[0], raw[0]).replace("\\\\", "\\")))
        elif match.group("paren") is not None:
            tokens.append((match.group("paren"), match.group("paren")))
        else:
            word = match.group("word").lower()
            if word not in ("and", "or", "not"):
                raise ValueError(
                    f"unexpected word {match.group('word')!r} in prompt expression "
                    "(descriptions must be quoted; operators are and, or, not)"
                )
            tokens.append((word, word))
    return tokens


def parse_expression(text: str) -> Node:
    """Parse the text form: quoted leaves combined with and / or / not and parentheses."""
    tokens = _tokenize(text)
    pos = 0

    def peek() -> str | None:
        return tokens[pos][0] if pos < len(tokens) else None

    def take(kind: str | None = None) -> tuple[str, str]:
        nonlocal pos
        if pos >= len(tokens) or (kind is not None and tokens[pos][0] != kind):
            raise ValueError(f"prompt expression: expected {kind or 'more input'}")
        pos += 1
        return tokens[pos - 1]

    def join(kind: str, parts: list[Node]) -> Node:
        return parts[0] if len(parts) == 1 else Op(kind, tuple(parts))

    def parse_or() -> Node:
        parts = [parse_and()]
        while peek() == "or":
            take()
            parts.append(parse_and())
        return join("any", parts)

    def parse_and() -> Node:
        parts = [parse_not()]
        while peek() == "and":
            take()
            parts.append(parse_not())
        return join("all", parts)

    def parse_not() -> Node:
        if peek() == "not":
            take()
            return Op("not", (parse_not(),))
        return parse_atom()

    def parse_atom() -> Node:
        if peek() == "(":
            take()
            node = parse_or()
            take(")")
            return node
        value = take("str")[1].strip()
        if not value:
            raise ValueError("prompt leaf is empty")
        return Leaf(value)

    node = parse_or()
    if pos != len(tokens):
        raise ValueError(f"prompt expression: unexpected {tokens[pos][1]!r}")
    return node


_NOT_QUOTED = re.compile(r"""not\s*["'(]""", re.IGNORECASE)


@lru_cache(maxsize=256)
def parse_prompt(prompt: str) -> Node:
    """A prompt string as a tree; plain text is a single leaf, verbatim."""
    text = prompt.strip()
    if text[:1] in ("{", "["):
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError(f"prompt looks like JSON but is not valid: {error}") from error
        if isinstance(obj, list):
            raise ValueError('a prompt list is ambiguous; use {"all": [...]} or {"any": [...]}')
        return from_object(obj)
    if text[:1] in ('"', "'", "(") or _NOT_QUOTED.match(text):
        return parse_expression(text)
    return Leaf(prompt)


def normalize_prompt(prompt) -> str:
    """The prompt as a string for the pipeline: structured values (from a
    YAML config) become compact JSON, text passes through unchanged."""
    if isinstance(prompt, str):
        return prompt
    from_object(prompt)  # validates
    return json.dumps(prompt, separators=(",", ":"), ensure_ascii=False)


def leaves(node: Node) -> list[str]:
    """Every distinct leaf text, in order of first appearance."""
    if isinstance(node, Leaf):
        return [node.text]
    seen: list[str] = []
    for child in node.children:
        for text in leaves(child):
            if text not in seen:
                seen.append(text)
    return seen


def is_tree(prompt: str) -> bool:
    return isinstance(parse_prompt(prompt), Op)


def _word(verdict: bool | None) -> str:
    return "?" if verdict is None else ("yes" if verdict else "no")


def evaluate(node: Node, ask: Callable[[str], "Decision"], source: str = ""):
    """Evaluate ``node``; ``ask(leaf_text)`` returns a classifier ``Decision``.

    A bare leaf returns ``ask``'s decision untouched. For a tree the result is
    a ``Decision`` whose detail lists the leaves that were actually asked and
    whose score (when every asked leaf had one) combines them fuzzily
    (min for all, max for any, 1 - s for not). ``source`` labels the
    classifier ("vlm", "siglip") in the per-leaf statistics.
    """
    from .classify import Decision

    if isinstance(node, Leaf):
        return ask(node.text)

    trace: list[str] = []
    cache: dict[str, Decision] = {}

    def run(n: Node):
        if isinstance(n, Leaf):
            if n.text not in cache:
                decision = ask(n.text)
                cache[n.text] = decision
                score = f" ({decision.score:.2f})" if decision.score is not None else ""
                trace.append(f"{n.text!r}: {_word(decision.redact)}{score}")
                label = f"{source}: " if source else ""
                telemetry.count("prompt.leaf_verdicts", f"{label}{n.text}: {_word(decision.redact)}")
            d = cache[n.text]
            return d.redact, d.score
        if n.kind == "not":
            verdict, score = run(n.children[0])
            return (None if verdict is None else not verdict), (None if score is None else 1.0 - score)
        want_all = n.kind == "all"
        undecided, scores, missing = False, [], False
        for child in n.children:
            verdict, score = run(child)
            if score is None:
                missing = True
            else:
                scores.append(score)
            if verdict is None:
                undecided = True
            elif verdict is (not want_all):  # a "no" settles all, a "yes" settles any
                combine = min if want_all else max
                return (not want_all), (None if missing else combine(scores))
        combine = min if want_all else max
        score = None if missing else combine(scores)
        return (None if undecided else want_all), score

    verdict, score = run(node)
    return Decision(verdict, score, "; ".join(trace) + f" -> {_word(verdict)}")
