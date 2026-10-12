# -*- coding: utf-8 -*-
"""Prompt building and answer construction (pure Python; the golden comparison with the reference is verify/check_prompt.py)."""
import pytest

from onejev_decide import choice, noul, score
from onejev_decide.protocol import (LETTERS, SLOT_LABELS, build_messages, make_answer, media_order, question_from_spec,
                                    render_question)


def test_choice_prompt_lists_lettered_options_and_ends_with_the_instruction():
    r = render_question(choice("Which team?", {"billing": "charges", "tech": "bugs"}))
    assert r.n_slots == 2 and r.labels == ["billing", "tech"]
    assert "A. billing: charges" in r.suffix and "B. tech: bugs" in r.suffix
    assert r.suffix.rstrip().endswith("A, B.")


def test_noul_slot_zero_is_yes():
    r = render_question(noul("The customer is angry"))
    assert r.n_slots == 2
    assert make_answer(r, [0.75, 0.25]) == {"type": "noul", "noul": 0.75}


def test_choice_answer_picks_the_largest_probability():
    r = render_question(choice("q", {"a": "x", "b": "y", "c": "z"}))
    answer = make_answer(r, [0.1, 0.7, 0.2])
    assert answer["choice"] == "b" and answer["probabilities"]["b"] == 0.7
    assert 0 < answer["confidence"] <= 1


def test_score_answer_is_an_expected_level():
    r = render_question(score("urgency", ["low", "mid", "high"]))
    answer = make_answer(r, [0.0, 0.0, 1.0])
    assert answer["score"] == 2.0 and answer["legend"]["2"] == "high"


def test_more_than_26_options_use_two_letter_labels():
    r = render_question(choice("q", {"o%d" % i: "d" for i in range(30)}))
    assert r.n_slots == 30 and len(LETTERS) == 26 and len(SLOT_LABELS) > 26


def test_spec_validation():
    with pytest.raises(ValueError):
        question_from_spec({"type": "choice", "criteria": {}})
    with pytest.raises(ValueError):
        question_from_spec({"type": "rank"})
    with pytest.raises(ValueError):
        question_from_spec({"type": "score", "criteria": ["l"] * 11})
    assert question_from_spec({"type": "noul", "instructions": "x"}).type == "noul"


def test_images_must_be_referenced_once_each_in_order():
    assert media_order("see <image:1> and <image:2>", 2) == [1, 2]
    with pytest.raises(ValueError):
        media_order("see <image:2> and <image:1>", 2)
    with pytest.raises(ValueError):
        media_order("no placeholder", 1)


def test_messages_end_with_the_user_turn():
    r = render_question(noul("x"))
    messages = build_messages({"a": 1}, r.suffix)
    assert messages[0]["role"] == "system" and messages[-1]["role"] == "user"
    assert "<state>" in messages[-1]["content"]
