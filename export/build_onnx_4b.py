# -*- coding: utf-8 -*-
"""Build an ONNX export of OneJev-4B by swapping weights into the Qwen3.5-4B ONNX graphs.

OneJev-4B is Qwen3.5-4B fine-tuned (same architecture, same tokenizer), and
onnx-community publishes an ONNX export of the base model, so the graphs can
be reused and only the weights replaced. How each ONNX tensor is derived from a
safetensors tensor (a transpose, a cast, -exp(A_log), ...) is NOT documented, and
a wrong guess does not fail: it produces a model that runs and answers wrongly.
So the rules below are first checked against the BASE checkpoint -- every swapped
tensor must reproduce the published base ONNX bytes -- and only then applied to OneJev.

    uv run --with onnx --with numpy export/build_onnx_4b.py verify   # rules vs base ONNX
    uv run --with onnx --with numpy export/build_onnx_4b.py build    # write OneJev-4B ONNX

Inputs are plain directories under work/onejev4b/ (override with --base/--onnx/--onejev); nothing is downloaded here:

    base/    Qwen/Qwen3.5-4B (safetensors, Apache-2.0)
    onnx/    the onnx/ folder (fp16 files) of onnx-community/Qwen3.5-4B-ONNX (not the -OPT repository: the shipped decoder has no
             num_logits_to_keep input and uses GroupQueryAttention, as that repository's graph does). That repo declares no
             licence (the base model Qwen/Qwen3.5-4B is Apache-2.0)
    onejev/  OmniJev/OneJev-4B (safetensors, Apache-2.0)
"""

from __future__ import annotations

import argparse
import json
import mmap
import pathlib
import re
import shutil
import sys

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORK = ROOT / "work" / "onejev4b"          # inputs (see the docstring); work/ is gitignored

GRAPHS = ("decoder_model_merged_fp16", "embed_tokens_fp16", "vision_encoder_fp16")

NP_OF = {TensorProto.FLOAT: np.float32, TensorProto.FLOAT16: np.float16, TensorProto.INT64: np.int64}


class SafeTensors:
    """Minimal safetensors reader (bf16 included) over one or more files."""

    def __init__(self, paths):
        self.index = {}
        self.maps = {}
        for p in paths:
            p = pathlib.Path(p)
            f = open(p, "rb")
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            n = int.from_bytes(mm[:8], "little")
            header = json.loads(mm[8:8 + n])
            header.pop("__metadata__", None)
            self.maps[p] = (f, mm, 8 + n)
            for k, v in header.items():
                self.index[k] = (p, v["dtype"], v["shape"], v["data_offsets"])

    def keys(self):
        return self.index.keys()

    def get(self, name) -> np.ndarray:
        p, dtype, shape, (a, b) = self.index[name]
        _, mm, base = self.maps[p]
        raw = np.frombuffer(mm, dtype=np.uint8, count=b - a, offset=base + a)
        if dtype == "BF16":
            out = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        elif dtype == "F16":
            out = raw.view(np.float16).astype(np.float32)
        elif dtype == "F32":
            out = raw.view(np.float32)
        else:
            raise ValueError(f"{name}: unsupported dtype {dtype}")
        return out.reshape(shape)


def read_external(tensor: TensorProto, onnx_dir: pathlib.Path) -> np.ndarray:
    info = {e.key: e.value for e in tensor.external_data}
    dt = NP_OF[tensor.data_type]
    count = int(np.prod(tensor.dims)) if tensor.dims else 1
    data = np.fromfile(onnx_dir / info["location"], dtype=dt, count=count, offset=int(info.get("offset", 0)))
    return data.reshape(tuple(tensor.dims))


def read_any(tensor: TensorProto, onnx_dir: pathlib.Path) -> np.ndarray:
    if tensor.data_location == TensorProto.EXTERNAL:
        return read_external(tensor, onnx_dir)
    return numpy_helper.to_array(tensor)


# ---- name rules: ONNX initializer -> (safetensors name, transform) ---------------------------------

LAYER = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
LM = "model.language_model."

#: ONNX suffix -> (safetensors suffix under layers.N., transform)
DECODER_RULES = {
    "input_layernorm.weight": ("input_layernorm.weight", "norm"),
    "post_attention_layernorm.weight": ("post_attention_layernorm.weight", "norm"),
    "gdn.conv1d.weight": ("linear_attn.conv1d.weight", "squeeze1"),
    "gdn.conv1d.weight_3d": ("linear_attn.conv1d.weight", "id"),
    "gdn.A_neg_exp": ("linear_attn.A_log", "negexp"),
    "gdn.dt_bias": ("linear_attn.dt_bias", "id"),
    # stored fp32 in the checkpoint; the published base ONNX holds it rounded to bf16 ("keep" = use the exact value,
    # verified against the rounded one)
    "gdn.norm.weight": ("linear_attn.norm.weight", "keep"),
    "gdn.in_proj_qkv.MatMul.weight": ("linear_attn.in_proj_qkv.weight", "T"),
    "gdn.in_proj_z.MatMul.weight": ("linear_attn.in_proj_z.weight", "T"),
    "gdn.in_proj_b.MatMul.weight": ("linear_attn.in_proj_b.weight", "T"),
    "gdn.in_proj_a.MatMul.weight": ("linear_attn.in_proj_a.weight", "T"),
    "gdn.out_proj.MatMul.weight": ("linear_attn.out_proj.weight", "T"),
    "attn.q_proj.MatMul.weight": ("self_attn.q_proj.weight", "T"),
    "attn.k_proj.MatMul.weight": ("self_attn.k_proj.weight", "T"),
    "attn.v_proj.MatMul.weight": ("self_attn.v_proj.weight", "T"),
    "attn.o_proj.MatMul.weight": ("self_attn.o_proj.weight", "T"),
    "attn.q_norm.layernorm.weight": ("self_attn.q_norm.weight", "norm"),
    "attn.k_norm.layernorm.weight": ("self_attn.k_norm.weight", "norm"),
    "mlp.gate_proj.MatMul.weight": ("mlp.gate_proj.weight", "T"),
    "mlp.up_proj.MatMul.weight": ("mlp.up_proj.weight", "T"),
    "mlp.down_proj.MatMul.weight": ("mlp.down_proj.weight", "T"),
}


def rule_for(graph: str, name: str):
    """(safetensors name, transform) or None when the tensor is a constant / not covered."""
    if graph.startswith("decoder"):
        m = LAYER.match(name)
        if m and m.group(2) in DECODER_RULES:
            suffix, tf = DECODER_RULES[m.group(2)]
            return f"{LM}layers.{m.group(1)}.{suffix}", tf
        if name == "lm_head.MatMul.weight":
            return "lm_head.weight", "T"
        if name.endswith(".final_norm_layernorm.weight"):
            return f"{LM}norm.weight", "norm"
    if graph.startswith("embed") and name == "model.embed_tokens.weight":
        return f"{LM}embed_tokens.weight", "id"
    return None


def get_source(st: SafeTensors, name: str) -> np.ndarray:
    """Tied checkpoints (the base 4B) have no lm_head.weight: it is the input embedding."""
    if name == "lm_head.weight" and name not in st.index:
        name = f"{LM}embed_tokens.weight"
    return st.get(name)


def to_bf16(w: np.ndarray) -> np.ndarray:
    u = w.astype(np.float32).view(np.uint32)
    u = (u + 0x7FFF + ((u >> 16) & 1)) >> 16 << 16  # round to nearest even
    return u.astype(np.uint32).view(np.float32)


def apply(tf: str, w: np.ndarray, norm_offset: float) -> np.ndarray:
    if tf in ("id", "keep"):
        return w
    if tf == "T":
        return np.ascontiguousarray(w.T)
    if tf == "squeeze1":
        return w.reshape(w.shape[0], w.shape[-1])
    if tf == "negexp":
        return -np.exp(w.astype(np.float32))
    if tf == "norm":
        return w + norm_offset
    raise ValueError(tf)


def load_graph(onnx_dir: pathlib.Path, graph: str):
    return onnx.load(str(onnx_dir / f"{graph}.onnx"), load_external_data=False)


def covered(model, graph):
    for t in model.graph.initializer:
        r = rule_for(graph, t.name)
        if r:
            yield t, r


def uncovered_big(model, graph):
    """Weights that are NOT covered by a rule (constants are tiny and ignored)."""
    out = []
    for t in model.graph.initializer:
        if rule_for(graph, t.name) is None and not t.name.startswith("/model/constants/"):
            out.append((t.name, list(t.dims), "EXT" if t.data_location == TensorProto.EXTERNAL else "inline"))
    return out


def detect_norm_offset(model, graph, onnx_dir, base: SafeTensors) -> float:
    """RMSNorm weights may be stored as w or 1+w in the ONNX; decide from the base checkpoint."""
    for t, (src, tf) in covered(model, graph):
        if tf == "norm":
            onnx_w = read_any(t, onnx_dir).astype(np.float32)
            w = get_source(base, src).astype(np.float32)
            for off in (0.0, 1.0):
                if np.allclose(onnx_w, w + off, atol=1e-2, rtol=1e-2):
                    return off
            raise SystemExit(f"cannot explain norm tensor {t.name} (max |onnx-w|={np.abs(onnx_w - w).max():.4f})")
    return 0.0


def cmd_verify(a):
    base = SafeTensors(sorted(pathlib.Path(a.base).glob("*.safetensors")))
    onnx_dir = pathlib.Path(a.onnx)
    bad = 0
    for graph in GRAPHS:
        model = load_graph(onnx_dir, graph)
        off = detect_norm_offset(model, graph, onnx_dir, base) if graph.startswith("decoder") else 0.0
        n = 0
        worst = 0.0
        for t, (src, tf) in covered(model, graph):
            want = read_any(t, onnx_dir)
            got = apply(tf, get_source(base, src), off)
            if tf == "keep":
                got = to_bf16(got)
            got = got.astype(want.dtype)
            if got.shape != want.shape:
                print(f"  SHAPE {t.name}: onnx {want.shape} vs rule {got.shape}")
                bad += 1
                continue
            d = float(np.abs(got.astype(np.float32) - want.astype(np.float32)).max())
            worst = max(worst, d)
            n += 1
            if d > 1e-3 * max(1.0, float(np.abs(want.astype(np.float32)).max())):
                print(f"  MISMATCH {t.name} <- {src} [{tf}]  max|d|={d:.5f}")
                bad += 1
        print(f"{graph}: {n} tensors checked, norm offset {off}, worst |d| {worst:.2e}")
        rest = uncovered_big(model, graph)
        print(f"  not covered by a rule: {len(rest)}")
        for r in rest[:40]:
            print("   ", r)
    print("RESULT:", "FAIL" if bad else "OK (rules reproduce the base ONNX)")
    return 1 if bad else 0


def cmd_build(a):
    base = SafeTensors(sorted(pathlib.Path(a.base).glob("*.safetensors")))
    one = SafeTensors(sorted(pathlib.Path(a.onejev).glob("*.safetensors")))
    onnx_dir = pathlib.Path(a.onnx)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for graph in GRAPHS:
        model = load_graph(onnx_dir, graph)
        off = detect_norm_offset(model, graph, onnx_dir, base) if graph.startswith("decoder") else 0.0
        files = sorted(p.name for p in onnx_dir.glob(f"{graph}.onnx_data*"))
        for f in files:
            if not (out / f).exists() or (out / f).stat().st_size != (onnx_dir / f).stat().st_size:
                shutil.copyfile(onnx_dir / f, out / f)
        maps = {f: np.memmap(out / f, dtype=np.uint8, mode="r+") for f in files}
        changed = same = 0
        for t, (src, tf) in covered(model, graph):
            new = apply(tf, get_source(one, src), off).astype(NP_OF[t.data_type])
            if tuple(new.shape) != tuple(t.dims):
                raise SystemExit(f"{t.name}: OneJev tensor shape {new.shape} != graph {tuple(t.dims)}")
            if t.data_location == TensorProto.EXTERNAL:
                info = {e.key: e.value for e in t.external_data}
                o = int(info.get("offset", 0))
                raw = np.frombuffer(new.tobytes(), dtype=np.uint8)
                if int(info.get("length", len(raw))) != len(raw):
                    raise SystemExit(f"{t.name}: byte length differs")
                cur = maps[info["location"]][o:o + len(raw)]
                if np.array_equal(cur, raw):
                    same += 1
                else:
                    maps[info["location"]][o:o + len(raw)] = raw
                    changed += 1
            else:
                t.raw_data = new.tobytes()
                changed += 1
        for m in maps.values():
            m.flush()
        onnx.save(model, str(out / f"{graph}.onnx"))
        print(f"{graph}: {changed} tensors replaced, {same} identical to base")
    for f in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
              "processor_config.json", "chat_template.jinja"):
        src = pathlib.Path(a.onejev) / f
        if src.exists():
            shutil.copyfile(src, out.parent / f)
    print("written to", out)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["verify", "build"])
    ap.add_argument("--base", default=str(WORK / "base"), help="Qwen3.5-4B safetensors dir")
    ap.add_argument("--onnx", default=str(WORK / "onnx"), help="Qwen3.5-4B-ONNX onnx/ dir (fp16 files)")
    ap.add_argument("--onejev", default=str(WORK / "onejev"), help="OneJev-4B safetensors dir")
    ap.add_argument("--out", default=str(ROOT / "models" / "onnx-4b" / "onnx"))
    a = ap.parse_args()
    sys.exit({"verify": cmd_verify, "build": cmd_build}[a.cmd](a))


if __name__ == "__main__":
    main()
