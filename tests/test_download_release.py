# -*- coding: utf-8 -*-
"""download_model.py (repository choice, file selection, a full round trip from a local release) and release/make_release.py."""
import argparse
import functools
import hashlib
import http.server
import json
import pathlib
import sys
import threading

import pytest

import download_model as dl
from conftest import load_script


def _args(**kw):
    base = dict(repo=None, tag="latest", base_url=None, include=None, exclude=None, variant=None)
    base.update(kw)
    return argparse.Namespace(**base)


def test_repository_priority_argument_then_environment_then_default(monkeypatch):
    monkeypatch.delenv("ONEJEV_ONNX_REPO", raising=False)
    assert dl.base_url(_args()) == "https://github.com/Kazuhito00/OneJev-ONNX-CPU/releases/latest/download"
    monkeypatch.setenv("ONEJEV_ONNX_REPO", "someone/fork")
    assert dl.base_url(_args()) == "https://github.com/someone/fork/releases/latest/download"
    assert dl.base_url(_args(repo="other/repo", tag="v0.0.0")) == "https://github.com/other/repo/releases/download/v0.0.0"
    assert dl.base_url(_args(base_url="http://127.0.0.1:8000/")) == "http://127.0.0.1:8000"
    assert dl.DEFAULT_REPO == "Kazuhito00/OneJev-ONNX-CPU"


def _entries():
    def e(path, variant=None):
        return {"path": path, "asset": path.replace("/", "__"), "size": 1, "sha256": "0", "parts": None, "variant": variant}
    return [e("onnx/onnx/decoder_model_merged_fp16.onnx", "fp16"), e("onnx/onnx/decoder_model_merged_fp16.onnx_data", "fp16"),
            e("onnx/onnx/decoder_model_merged_q4f16.onnx", "q4f16"), e("onnx/onnx/vision_encoder_fp16.onnx_data", "fp16"),
            e("onnx/tokenizer.json"), e("onnx/config.json")]


def _paths(**kw):
    return [f["path"] for f in dl.select(_entries(), _args(**kw))]


def test_default_selection_is_everything():
    assert len(_paths()) == 6


def test_variant_keeps_shared_files():
    got = _paths(variant=["fp16"])
    assert "onnx/tokenizer.json" in got and "onnx/onnx/decoder_model_merged_q4f16.onnx" not in got and len(got) == 5


def test_text_only_drops_the_vision_encoder():
    assert not [p for p in _paths(exclude=["vision_encoder*"]) if "vision" in p]


def test_include_globs():
    assert sorted(_paths(include=["*.json"])) == ["onnx/config.json", "onnx/tokenizer.json"]


def _fake_models(root: pathlib.Path):
    onnx = root / "onnx"
    (onnx / "onnx").mkdir(parents=True)
    pool = bytes(range(256)) * 40                                  # 10240 bytes: split into parts in the test
    (onnx / "onnx" / "decoder_model_merged_fp16.onnx_data").write_bytes(pool)
    (onnx / "onnx" / "decoder_model_merged_fp16.onnx").write_bytes(b"graph")
    (onnx / "onnx" / "decoder_model_merged_q4f16.onnx").write_bytes(b"graph4")
    (onnx / "tokenizer.json").write_text("{}", encoding="utf-8")
    (root / "onnx-4b" / "onnx").mkdir(parents=True)
    (root / "onnx-4b" / "onnx" / "decoder_model_merged_fp16.onnx").write_bytes(b"graph4b")
    (root / "onnx-4b" / "tokenizer.json").write_text("{}", encoding="utf-8")
    return pool


def _serve(directory: pathlib.Path):
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
    handler.log_message = lambda *a, **k: None
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _stage(tmp_path, monkeypatch):
    mk = load_script("make_release", "release/make_release.py")
    pool = _fake_models(tmp_path / "models")
    monkeypatch.setattr(sys, "argv", ["make_release.py", "--models", str(tmp_path / "models"), "--out", str(tmp_path / "dist"),
                                      "--max-part-gib", str(4000 / (1 << 30))])
    assert mk.main() == 0
    return pool


def test_make_release_stages_parts_and_one_manifest_per_size(tmp_path, monkeypatch):
    pool = _stage(tmp_path, monkeypatch)
    manifest = json.loads((tmp_path / "dist" / "manifest-0.8b.json").read_text(encoding="utf-8"))
    by_path = {f["path"]: f for f in manifest["files"]}
    assert manifest["format"] == 1 and manifest["model_size"] == "0.8b"
    data = by_path["onnx/onnx/decoder_model_merged_fp16.onnx_data"]
    assert [p["size"] for p in data["parts"]] == [4000, 4000, 2240] and data["sha256"] == hashlib.sha256(pool).hexdigest()
    assert data["variant"] == "fp16" and by_path["onnx/tokenizer.json"]["variant"] is None
    assert "onnx/LICENSE-ONEJEV" in by_path and "onnx/LICENSE-QWEN" in by_path
    assert (tmp_path / "dist" / "manifest-4b.json").exists()
    assert (tmp_path / "dist" / "onnx__onnx__decoder_model_merged_fp16.onnx_data.part002").stat().st_size == 2240


def test_download_round_trip_from_a_local_release(tmp_path, monkeypatch):
    pool = _stage(tmp_path, monkeypatch)
    server = _serve(tmp_path / "dist")
    try:
        url = "http://127.0.0.1:%d" % server.server_address[1]
        dest = tmp_path / "out"
        monkeypatch.setattr(sys, "argv", ["download_model.py", "--base-url", url, "--dest", str(dest), "--model-size", "0.8b", "--variant", "fp16"])
        assert dl.main() == 0
        assert (dest / "onnx/onnx/decoder_model_merged_fp16.onnx_data").read_bytes() == pool       # parts joined, sha256 checked
        assert not (dest / "onnx/onnx/decoder_model_merged_q4f16.onnx").exists()
        assert not (dest / "onnx-4b").exists()                                                      # 4B is skipped with --model-size 0.8b
        monkeypatch.setattr(sys, "argv", ["download_model.py", "--base-url", url, "--dest", str(dest), "--model-size", "4b"])
        assert dl.main() == 0
        assert (dest / "onnx-4b/onnx/decoder_model_merged_fp16.onnx").read_bytes() == b"graph4b"
    finally:
        server.shutdown()


def test_a_corrupted_asset_is_deleted_and_stops(tmp_path, monkeypatch):
    _stage(tmp_path, monkeypatch)
    (tmp_path / "dist" / "onnx__tokenizer.json").write_text("{]", encoding="utf-8")                # same size, other bytes
    server = _serve(tmp_path / "dist")
    try:
        monkeypatch.setattr(sys, "argv", ["download_model.py", "--base-url", "http://127.0.0.1:%d" % server.server_address[1],
                                          "--dest", str(tmp_path / "out"), "--include", "tokenizer.json"])
        with pytest.raises(SystemExit, match="SHA-256 mismatch"):
            dl.main()
        assert not (tmp_path / "out/onnx/tokenizer.json").exists()
    finally:
        server.shutdown()
