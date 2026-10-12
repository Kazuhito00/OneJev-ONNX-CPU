# -*- coding: utf-8 -*-
"""Where the model files are, and the command-line parsing of the demos (no model is loaded)."""
import argparse

import pytest

import demo_inference_text as demo
from onejev_decide import choice, noul, score
from onejev_decide.paths import default_model_dir
from onejev_decide.protocol import render_question


def test_model_dir_priority_argument_then_environment_then_checkout(monkeypatch, tmp_path):
    monkeypatch.delenv("ONEJEV_MODEL_DIR", raising=False)
    monkeypatch.delenv("ONEJEV4_MODEL_DIR", raising=False)
    assert default_model_dir("0.8b", tmp_path) == tmp_path / "models" / "onnx"
    assert default_model_dir("4b", tmp_path) == tmp_path / "models" / "onnx-4b"
    monkeypatch.setenv("ONEJEV_MODEL_DIR", str(tmp_path / "env08"))
    monkeypatch.setenv("ONEJEV4_MODEL_DIR", str(tmp_path / "env4"))
    assert default_model_dir("0.8b", tmp_path) == tmp_path / "env08"
    assert default_model_dir("4b", tmp_path) == tmp_path / "env4"
    assert default_model_dir("0.8b", tmp_path, "mine") == tmp_path / "mine"
    absolute = tmp_path / "abs"
    assert default_model_dir("4b", tmp_path, absolute) == absolute


def test_unknown_model_size_is_rejected():
    with pytest.raises(ValueError):
        default_model_dir("9b")


def _args(**kw):
    base = dict(choice=None, score=None, noul=None, instructions="")
    base.update(kw)
    return argparse.Namespace(**base)


def test_parse_questions_builds_each_kind():
    q = demo.parse_questions(_args(choice=["team=billing, tech"], score=["urgency=low,high"],
                                   noul=["angry=The customer is angry"]))
    assert q["team"] == choice("team", {"billing": "billing", "tech": "tech"})
    assert q["urgency"] == score("urgency", ["low", "high"])
    assert q["angry"] == noul("The customer is angry")


def test_builtin_example_is_valid():
    for question in demo.EXAMPLE_QUESTIONS.values():
        render_question(question)


def test_the_demo_has_the_device_options(capsys):
    with pytest.raises(SystemExit) as info:
        demo.main(["--help"])
    assert info.value.code == 0
    out = capsys.readouterr().out
    assert "--device" in out and "--require-gpu" in out
