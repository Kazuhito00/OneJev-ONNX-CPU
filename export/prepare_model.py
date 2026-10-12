# -*- coding: utf-8 -*-
"""Put a runnable model under models/.

Every repository in this family exposes the same entry point; what it does
differs. OneJev already has a published ONNX export, so this downloads it.

    python export/prepare_model.py --list
    python export/prepare_model.py                  # fp16 (fastest on CPU)
    python export/prepare_model.py --variant q4f16  # smaller download
    python export/prepare_model.py --checkpoint     # + the PyTorch weights (oracle)

No git and no git-lfs: files come from Hugging Face's resolve endpoint using
only the standard library.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from export.download import fetch_files  # noqa: E402

ONNX_REPO = "onnx-community/OneJev-0.8B-ONNX"
CHECKPOINT_REPO = "OmniJev/OneJev-0.8B"
ONNX_DIR = ROOT / "models" / "onnx"
HF_DIR = ROOT / "models" / "hf"

VARIANTS = {
    "fp32": "reference precision. The decoder alone is 3.0 GB in ONE file.",
    "fp16": "default on CPU; ~2.1 GB total, fastest here",
    "q4f16": "smallest (~0.65 GB) but SLOWER on CPU (4-bit is a WebGPU optimisation)",
}

GRAPHS = ("embed_tokens", "vision_encoder", "decoder_model_merged")

#: Everything the runtime reads, beside the graphs.
SUPPORT_FILES = ["tokenizer.json", "tokenizer_config.json", "config.json",
                 "preprocessor_config.json", "processor_config.json",
                 "chat_template.jinja", "generation_config.json"]

#: What the PyTorch oracle needs. Only the oracle gates read these.
CHECKPOINT_FILES = SUPPORT_FILES[:2] + ["config.json", "processor_config.json",
                                        "chat_template.jinja", "generation_config.json",
                                        "model.safetensors"]


def suffix(variant: str) -> str:
    return "" if variant == "fp32" else "_" + variant


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="show the variants")
    parser.add_argument("--variant", default="fp16", choices=sorted(VARIANTS))
    parser.add_argument("--checkpoint", action="store_true",
                        help="also fetch the original PyTorch weights (2.2 GB) for the oracle gates")
    args = parser.parse_args()

    if args.list:
        print("variants:\n")
        for name, note in VARIANTS.items():
            print("  --variant %-7s %s" % (name, note))
        print("\nOn CPU prefer fp16. q4f16 is smaller but slower here.")
        print("\n  --checkpoint    also fetch %s (2.2 GB)" % CHECKPOINT_REPO)
        print("                  only the oracle gates in verify/ need it")
        return 0

    paths = list(SUPPORT_FILES)
    optional = []
    for graph in GRAPHS:
        stem = "onnx/%s%s.onnx" % (graph, suffix(args.variant))
        paths += [stem, stem + "_data"]
        # Graphs above 2 GB are split; the second shard only exists for those.
        paths.append(stem + "_data_1")
        optional.append(stem + "_data_1")

    print("%s (%s) -> %s" % (ONNX_REPO, args.variant, ONNX_DIR), flush=True)
    fetch_files(ONNX_REPO, paths, ONNX_DIR, optional=optional)

    if args.checkpoint:
        print("\n%s -> %s" % (CHECKPOINT_REPO, HF_DIR), flush=True)
        fetch_files(CHECKPOINT_REPO, sorted(set(CHECKPOINT_FILES)), HF_DIR)

    print("\nready: %s" % ONNX_DIR)
    print("run:    python demo_inference_text.py --variant %s" % args.variant)
    print("verify: python verify/run_all.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
