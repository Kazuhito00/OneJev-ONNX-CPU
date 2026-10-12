# -*- coding: utf-8 -*-
"""The pure-Python byte-level BPE on a tiny tokenizer.json (the Rust-tokenizers comparison is verify/check_tokenizer_fuzz.py)."""
from onejev_decide.tokenizer import QwenBPE


def test_merges_apply(tiny_tokenizer_json):
    tok = QwenBPE(tiny_tokenizer_json)
    assert tok.encode("the") == [tok.vocab["the"]]
    assert tok.encode("and") == [tok.vocab["and"]]
    assert len(tok.encode("a b")) == 3            # a, space, b: no merge across the space


def test_special_tokens_are_one_id(tiny_tokenizer_json):
    tok = QwenBPE(tiny_tokenizer_json)
    assert len(tok.encode("<|image_pad|>")) == 1
    assert len(tok.encode("<|endoftext|>x")) == 2


def test_non_ascii_goes_through_bytes(tiny_tokenizer_json):
    tok = QwenBPE(tiny_tokenizer_json)
    assert len(tok.encode("あ")) == 3          # one Japanese character is three UTF-8 bytes, no merges for them
