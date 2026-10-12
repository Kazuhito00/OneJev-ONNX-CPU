# -*- coding: utf-8 -*-
"""Stage a GitHub Release from models/: split what is too big, hash everything, write manifest-<size>.json.

    python release/make_release.py --plan                    # only list the assets (no hashing, no copying)
    python release/make_release.py                           # models/ -> dist/release/ (both sizes)
    python release/make_release.py --size 0.8b --version v0.0.0
    python release/make_release.py --models path/to/models   # stage from another checkout's models/

GitHub allows at most 2 GiB per release asset (and 1000 assets per release), so a file larger than ``--max-part-gib`` is cut into
byte parts (``<name>.part000``, ...) that download_model.py joins again and checks against the whole file's sha256. The default
1.99 GiB (2,136,746,229 bytes) leaves every file of both sizes whole (the largest, OneJev-4B's decoder weights, is 2,095,788,032
bytes), so at present nothing is split. Asset names are flat, so ``onnx/onnx/decoder_model_merged_fp16.onnx`` is published as
``onnx__onnx__decoder_model_merged_fp16.onnx``; the manifest keeps the real path.

What goes in, per size: everything under models/onnx (0.8B: fp16 and q4f16 graphs, their external data, tokenizer and settings)
or models/onnx-4b (4B: fp16), plus the two licence texts from release/licenses/ (LICENSE-ONEJEV, LICENSE-QWEN, unchanged).
models/hf (the PyTorch checkpoint for the oracle gates) is never published.

Nothing is uploaded here. Upload dist/release/* yourself, for example:

    gh release create v0.0.0 dist/release/* --title "v0.0.0" --notes-file release/notes.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
CHUNK = 1 << 20
FOLDERS = {"0.8b": "onnx", "4b": "onnx-4b"}
NAMES = {"0.8b": "onejev-0.8b-onnx", "4b": "onejev-4b-onnx"}
LICENSES = ["LICENSE-ONEJEV", "LICENSE-QWEN"]


def sha256_of(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def link_or_copy(src: pathlib.Path, dst: pathlib.Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)  # no second copy of a multi-GB file on the same drive
    except OSError:
        shutil.copy2(src, dst)


def split(src: pathlib.Path, out_dir: pathlib.Path, stem: str, part_bytes: int) -> list:
    parts = []
    with open(src, "rb") as handle:
        index = 0
        while True:
            name = "%s.part%03d" % (stem, index)
            digest, written = hashlib.sha256(), 0
            with open(out_dir / name, "wb") as out:
                while written < part_bytes:
                    block = handle.read(min(CHUNK, part_bytes - written))
                    if not block:
                        break
                    out.write(block)
                    digest.update(block)
                    written += len(block)
            if written == 0:
                (out_dir / name).unlink()
                break
            parts.append({"asset": name, "size": written, "sha256": digest.hexdigest()})
            index += 1
            if written < part_bytes:
                break
    return parts


def variant_of(name: str):
    """'fp16' / 'q4f16' for a graph or its external data, None for files shared by every variant."""
    match = re.search(r"_(fp16|q4f16)\.onnx", name)
    return match.group(1) if match else None


def collect(models: pathlib.Path, size: str) -> list:
    """(path relative to models/, source file) for one size: the model folder, then the licence texts."""
    folder = models / FOLDERS[size]
    chosen = sorted(p for p in folder.rglob("*") if p.is_file() and "__pycache__" not in p.parts and not p.name.endswith(".part"))
    items = [(p.relative_to(models).as_posix(), p) for p in chosen]
    for name in LICENSES:
        source = ROOT / "release" / "licenses" / name
        if source.exists():
            items.append(("%s/%s" % (FOLDERS[size], name), source))
    return items


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default=str(ROOT / "models"))
    parser.add_argument("--out", default=str(ROOT / "dist" / "release"))
    parser.add_argument("--size", default="all", choices=("0.8b", "4b", "all"))
    parser.add_argument("--version", default="v0.0.0")
    parser.add_argument("--max-part-gib", type=float, default=1.99,
                        help="assets above this are split (GitHub's limit is 2 GiB)")
    parser.add_argument("--plan", action="store_true", help="list the assets and their sizes only; no hashing, no files written")
    args = parser.parse_args()

    models, out = pathlib.Path(args.models), pathlib.Path(args.out)
    limit = int(args.max_part_gib * (1 << 30))
    sizes = list(FOLDERS) if args.size == "all" else [args.size]
    if not args.plan:
        out.mkdir(parents=True, exist_ok=True)

    written = 0
    for size in sizes:
        items = collect(models, size)
        if not (models / FOLDERS[size]).is_dir() or not items:
            raise SystemExit("nothing under %s (run download_model.py, export/prepare_model.py or export/build_onnx_4b.py first)"
                             % (models / FOLDERS[size]))
        print("== OneJev-%s" % size.upper())
        entries = []
        for rel, path in items:
            asset = rel.replace("/", "__")
            file_size = path.stat().st_size
            n_parts = -(-file_size // limit) if file_size > limit else 0
            if args.plan:
                print("  %13d B  %9.1f MB  %s%s" % (file_size, file_size / 1e6, asset, "  -> %d parts" % n_parts if n_parts else ""))
                entries.append(file_size)
                continue
            entry = {"path": rel, "asset": asset, "size": file_size, "sha256": sha256_of(path), "variant": variant_of(path.name),
                     "parts": None}
            if file_size > limit:
                entry["parts"] = split(path, out, asset, limit)
                print("split  %-52s %9.1f MB -> %d parts" % (rel, file_size / 1e6, len(entry["parts"])))
            else:
                link_or_copy(path, out / asset)
                print("asset  %-52s %9.1f MB" % (rel, file_size / 1e6))
            entries.append(entry)
        if args.plan:
            print("  total %d files, %.2f GB" % (len(entries), sum(entries) / 1e9))
            continue
        manifest = {"format": 1, "name": NAMES[size], "version": args.version, "model_size": size, "files": entries}
        (out / ("manifest-%s.json" % size)).write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        (out / ("SHA256SUMS-%s" % size)).write_text("".join("%s  %s\n" % (e["sha256"], e["path"]) for e in entries), encoding="utf-8")
        written += len(entries)

    if args.plan:
        return 0
    too_big = [a.name for a in out.iterdir() if a.stat().st_size >= (2 << 30)]
    if too_big:
        raise SystemExit("asset(s) of 2 GiB or more: %s" % ", ".join(too_big))
    print("wrote %s: %d files, %d assets" % (out, written, len(list(out.iterdir()))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
