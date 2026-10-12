# -*- coding: utf-8 -*-
"""The OneJev request protocol, without transformers or pydantic.

Ported from the reference implementation (OmniJev/OneJev: ``qev/prompt.py``,
``qev/answers.py``, ``qev/media.py``). The checkpoint was trained against these
exact strings, so every constant is taken from there rather than re-derived.

A request is a state, optionally images the state points at with ``<image:N>``,
and typed questions. Each question becomes one prompt::

    <system> Apply the question to the state. Choose exactly one ... letter ...
    <user>   <state> ... </state>

             Question: <instructions>

             Options:
             A. yes: ...
             B. no: ...

             Answer with one letter: A, B.
    <assistant> <think></think>          <- the next token is the answer letter

The answer is the softmax, over the option letters' token ids, of the logits at
that last position. Nothing is generated.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence, Tuple

__all__ = [
    "SYSTEM_PROMPT", "LETTERS", "SLOT_LABELS", "MAX_SLOTS", "PLACEHOLDER",
    "Question", "choice", "score", "noul", "question_from_spec",
    "render_question", "state_block", "build_messages", "media_order",
    "make_answer", "RenderedQuestion",
]

PROMPT_VERSION = "qev-labels-v2"

SYSTEM_PROMPT = (
    "Apply the question to the state. Choose exactly one of the listed options. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

#: Two-letter labels the Qwen tokenizer splits into two tokens, so they cannot
#: serve as a single-token answer slot.
_NOT_ONE_TOKEN = {
    "BQ", "BZ", "CJ", "CQ", "CZ", "DQ", "DZ", "EJ", "EY", "FJ", "FQ", "FV", "FZ",
    "GJ", "GK", "GQ", "GZ", "HJ", "IY", "JF", "JG", "JH", "JL", "JN", "JQ",
}
SLOT_LABELS: Tuple[str, ...] = tuple(
    list(LETTERS)
    + [a + b for a in LETTERS for b in LETTERS if a + b not in _NOT_ONE_TOKEN][: 256 - len(LETTERS)]
)
MAX_SLOTS = len(SLOT_LABELS)

#: Media is referenced from the state as <image:N>, 1-based, once each, in order.
PLACEHOLDER = re.compile(r"<(image|video):(\d+)>")

NOUL_DEFAULT_INSTRUCTIONS = "Is the statement true, or is the answer to the question yes?"
NOUL_DEFAULT_TRUE = "the statement is true / the answer is yes"
NOUL_DEFAULT_FALSE = "the statement is false / the answer is no"

MAX_CHOICE_OPTIONS = 255
MAX_SCORE_LEVELS = 10


# --------------------------------------------------------------------- questions


@dataclass
class Question:
    type: str
    instructions: Any = None
    #: choice: {value: description}; score: [description, ...];
    #: noul: {"true": ..., "false": ...} or None
    criteria: Any = None


def choice(instructions: Any, criteria: Mapping[str, Any]) -> Question:
    """Pick one of the named options."""
    return Question("choice", instructions, dict(criteria))


def score(instructions: Any, levels: Sequence[Any]) -> Question:
    """Rate on an ordered scale; the first level is the lowest."""
    return Question("score", instructions, list(levels))


def noul(instructions: Any = None, true: Any = None, false: Any = None) -> Question:
    """A yes/no gate."""
    criteria = None
    if true is not None or false is not None:
        criteria = {"true": true, "false": false}
    return Question("noul", instructions, criteria)


def question_from_spec(spec: Mapping[str, Any]) -> Question:
    """A System One style dict: {type, instructions, criteria}."""
    kind = spec.get("type")
    if kind not in ("choice", "score", "noul"):
        raise ValueError("question type must be choice, score or noul, got %r" % (kind,))
    criteria = spec.get("criteria")
    if kind == "choice":
        if not isinstance(criteria, Mapping) or not criteria:
            raise ValueError("a choice question needs a non-empty criteria object")
        if len(criteria) > MAX_CHOICE_OPTIONS:
            raise ValueError("a choice question allows at most %d options" % MAX_CHOICE_OPTIONS)
    if kind == "score":
        if not isinstance(criteria, (list, tuple)) or not criteria:
            raise ValueError("a score question needs a non-empty criteria list")
        if len(criteria) > MAX_SCORE_LEVELS:
            raise ValueError("a score question allows at most %d levels" % MAX_SCORE_LEVELS)
    return Question(kind, spec.get("instructions"), criteria)


# ---------------------------------------------------------------------- rendering


def render_entry(entry: Any) -> str:
    """Render an instructions / criteria entry (string, object, array or null)."""
    if entry is None:
        return ""
    if isinstance(entry, str):
        return entry.strip()
    return json.dumps(entry, ensure_ascii=False, indent=2)


def render_state(state: Any) -> str:
    if state is None:
        return ""
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, indent=2)


@dataclass
class RenderedQuestion:
    kind: str
    labels: List[str]
    descriptions: List[Any]
    suffix: str
    n_slots: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_slots = len(self.labels)


def _options_block(names: Sequence[str], descriptions: Sequence[Any]) -> str:
    lines = []
    for letter, name, desc in zip(SLOT_LABELS, names, descriptions):
        text = render_entry(desc)
        if text:
            lines.append("%s. %s: %s" % (letter, name, text))
        else:
            lines.append("%s. %s" % (letter, name))
    return "\n".join(lines)


def render_question(question: Question) -> RenderedQuestion:
    """qev/prompt.py :: render_question, default style (tagged layout)."""
    if question.type == "noul":
        kind = "noul"
        instructions = render_entry(question.instructions) or NOUL_DEFAULT_INSTRUCTIONS
        crit = question.criteria or {}
        true = render_entry(crit.get("true")) if crit.get("true") is not None else NOUL_DEFAULT_TRUE
        false = render_entry(crit.get("false")) if crit.get("false") is not None else NOUL_DEFAULT_FALSE
        labels, names, descs = ["yes", "no"], ["yes", "no"], [true, false]
    elif question.type == "choice":
        kind = "choice"
        instructions = render_entry(question.instructions) or "Which option applies to the state?"
        labels = list(question.criteria.keys())
        names = list(labels)
        descs = [question.criteria[k] for k in labels]
    elif question.type == "score":
        kind = "score"
        instructions = render_entry(question.instructions) or "Which level describes the state?"
        labels = [str(i) for i in range(len(question.criteria))]
        names = ["level %d" % i for i in range(len(question.criteria))]
        descs = list(question.criteria)
    else:
        raise TypeError("unknown question type %r" % (question.type,))

    if len(labels) > MAX_SLOTS:
        raise ValueError(
            "%d options exceed the %d slots of prompt %s" % (len(labels), MAX_SLOTS, PROMPT_VERSION)
        )

    letters = list(SLOT_LABELS[: len(labels)])
    header = "Question:" if kind != "score" else "Rate the state:"
    answer_line = (
        "Answer with one letter: {letters}."
        if len(letters) <= len(LETTERS)
        else "Answer with one label: {letters}."
    ).format(letters=", ".join(letters))
    suffix = (
        "%s %s\n\nOptions:\n%s\n\n%s"
        % (header, instructions, _options_block(names, descs), answer_line)
    )
    return RenderedQuestion(kind, labels, descs, suffix)


def state_block(state: Any) -> str:
    return "<state>\n%s\n</state>\n\n" % render_state(state)


def build_messages(state: Any, suffix: str) -> List[Dict[str, str]]:
    """The user turn is state_block + suffix, so every question shares the same
    prefix up to the end of the state block."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": state_block(state) + suffix},
    ]


def media_order(text: str, count: int) -> List[int]:
    """Check that ``<image:N>`` placeholders cover 1..count once each, in order."""
    seen = []
    for match in PLACEHOLDER.finditer(text):
        if match.group(1) != "image":
            raise ValueError("this build handles images only; found %s" % match.group(0))
        number = int(match.group(2))
        if number < 1 or number > count:
            raise ValueError("placeholder %s does not match any of the %d images" % (match.group(0), count))
        seen.append(number)
    if seen != list(range(1, count + 1)):
        raise ValueError(
            "images must be referenced once each, in order, as <image:N>; "
            "saw %s for %d images" % (seen, count)
        )
    return seen


# ------------------------------------------------------------------------ answers


def _normalize(probs: Sequence[float]) -> List[float]:
    total = float(sum(probs))
    if total <= 0:
        return [1.0 / len(probs)] * len(probs)
    return [float(p) / total for p in probs]


def _round(x: float, digits: int = 6) -> float:
    return float(round(x, digits))


def _choice_confidence(probs: Sequence[float]) -> float:
    """Peak probability rescaled from uniform (0) to certainty (1)."""
    if len(probs) == 1:
        return 1.0
    p = _normalize(probs)
    uniform = 1.0 / len(p)
    return (max(p) - uniform) / (1.0 - uniform)


def _score_confidence(probs: Sequence[float]) -> float:
    """One minus the probability-weighted mean absolute deviation from the modal
    level, normalised by that of the uniform distribution."""
    if len(probs) == 1:
        return 1.0
    p = _normalize(probs)
    k = len(p)
    mode = max(range(k), key=p.__getitem__)
    distance = sum(pi * abs(i - mode) for i, pi in enumerate(p))
    centre = (k - 1) / 2
    uniform_mad = sum(abs(i - centre) for i in range(k)) / k
    return max(0.0, 1.0 - distance / uniform_mad)


def make_answer(rendered: RenderedQuestion, probabilities: Sequence[float]) -> Dict[str, Any]:
    """qev/answers.py: probabilities -> a typed answer."""
    p = _normalize(probabilities)
    if rendered.kind == "noul":
        # slot 0 is "yes"
        return {"type": "noul", "noul": _round(min(1.0, max(0.0, p[0])))}

    if rendered.kind == "choice":
        best = max(range(len(p)), key=p.__getitem__)
        return {
            "type": "choice",
            "choice": rendered.labels[best],
            "probabilities": {k: _round(v) for k, v in zip(rendered.labels, p)},
            "confidence": _round(_choice_confidence(p)),
        }

    expected = float(sum(i * pi for i, pi in enumerate(p)))
    return {
        "type": "score",
        "score": _round(expected),
        "legend": {str(i): level for i, level in enumerate(rendered.descriptions)},
        "probabilities": {str(i): _round(v) for i, v in enumerate(p)},
        "confidence": _round(_score_confidence(p)),
    }

