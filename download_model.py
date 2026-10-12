# -*- coding: utf-8 -*-
"""Download the ONNX model for demo_inference_text.py / demo_inference_image.py from this repository's GitHub release.

    python download_model.py                                    # everything: OneJev-0.8B and 4B (about 13 GB), latest release, SHA-256 checked
    python download_model.py --model-size 0.8b                  # OneJev-0.8B only (all variants; about 2.9 GB)
    python download_model.py --model-size 4b                    # OneJev-4B only (fp16 only; about 10 GB)
    python download_model.py --variant fp16                     # skip the q4f16 graphs (0.8B)
    python download_model.py --exclude "vision_encoder*"        # text-only use: skip the vision encoder
    python download_model.py --tag v0.0.0                       # a given release
    python download_model.py --base-url http://127.0.0.1:8000   # a mirror, or a local release/ output
    python download_model.py --list
    python download_model.py --repo OWNER/NAME                  # a fork's release

The default repository is ``Kazuhito00/OneJev-ONNX-CPU``. Give another one with ``--repo OWNER/NAME`` or through the
environment: ``ONEJEV_ONNX_REPO=OWNER/NAME`` (the argument wins over the environment, the environment over the default).

Without arguments this fetches every file of both sizes (0.8B: fp16 and q4f16, the tokenizer and the settings; and 4B). The 4B files
are about 10 GB and need about 10 GB of RAM to run: use ``--model-size 0.8b`` to skip them.

The files land where the demos look for them: ``models/onnx`` (0.8B) and ``models/onnx-4b`` (4B).

The 0.8B ONNX files are the public onnx-community/OneJev-0.8B-ONNX, unchanged; ``python export/prepare_model.py`` fetches them
from Hugging Face instead. The 4B ONNX is built by export/build_onnx_4b.py and exists only in this release.

The release carries ``manifest-0.8b.json`` and ``manifest-4b.json`` (written by release/make_release.py): every file's path, size
and sha256 and, for files that were split to stay under GitHub's 2 GiB asset limit, the parts. This script fetches what the
manifest lists, joins the parts and checks every sha256 (a mismatch deletes the file and stops). A file already in place with the
right sha256 is skipped and an interrupted download continues where it stopped (HTTP Range), so the same command can simply be
run again. Plain HTTPS from the standard library: no git, no git-lfs, no huggingface_hub.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import pathlib
import sys
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent
DEFAULT_REPO = "Kazuhito00/OneJev-ONNX-CPU"
ENV_REPO = "ONEJEV_ONNX_REPO"
USER_AGENT = "onejev-onnx-cpu/0.1 (urllib)"
SIZES = ("0.8b", "4b")
CHUNK = 1 << 20  # 1 MiB


def _human(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return "%.1f %s" % (size, unit)
        size /= 1024
    return "%.1f GB" % size


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def base_url(args) -> str:
    if args.base_url:
        return args.base_url.rstrip("/")
    repo = args.repo or os.environ.get(ENV_REPO) or DEFAULT_REPO
    if args.tag == "latest":
        return "https://github.com/%s/releases/latest/download" % repo
    return "https://github.com/%s/releases/download/%s" % (repo, args.tag)


def fetch_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def fetch(url: str, dest: pathlib.Path, size: int, label: str = "", tries: int = 8) -> pathlib.Path:
    """Download ``url`` to ``dest`` (``size`` bytes), resuming the bytes already there; retries on dropped connections."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    for attempt in range(1, tries + 1):
        have = dest.stat().st_size if dest.exists() else 0
        if have == size:
            return dest
        if have > size:
            dest.unlink()
            have = 0
        headers = {"User-Agent": USER_AGENT}
        if have:
            headers["Range"] = "bytes=%d-" % have
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=60) as response:
                if have and response.status != 206:
                    have = 0  # the server ignored the Range header: start over
                with open(dest, "ab" if have else "wb") as handle:
                    done, last = have, 0.0
                    while True:
                        block = response.read(CHUNK)
                        if not block:
                            break
                        handle.write(block)
                        done += len(block)
                        now = time.time()
                        if now - last > 0.5:
                            last = now
                            rate = (done - have) / max(now - started, 1e-6)
                            sys.stdout.write("\r  %-44s %5.1f%%  %s/s   "
                                             % (label or dest.name, 100 * done / size, _human(rate)))
                            sys.stdout.flush()
            if dest.stat().st_size == size:
                return dest
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as exc:
            wait = min(60, 2 ** attempt)
            print("\n  %s: %s (try %d/%d, again in %d s)" % (dest.name, exc, attempt, tries, wait), flush=True)
            time.sleep(wait)
    raise SystemExit("could not download %s completely" % url)


def select(files: list, args) -> list:
    """The manifest entries wanted by --variant / --include / --exclude."""
    def wanted(f: dict) -> bool:
        path = f["path"]
        name = path.rsplit("/", 1)[-1]
        if args.variant and f.get("variant") and f["variant"] not in args.variant:
            return False                    # a graph of another variant (fp16 / q4f16); shared files have variant None
        if args.include and not any(fnmatch.fnmatch(path, g) or fnmatch.fnmatch(name, g) for g in args.include):
            return False
        return not (args.exclude and any(fnmatch.fnmatch(path, g) or fnmatch.fnmatch(name, g) for g in args.exclude))

    return [f for f in files if wanted(f)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", help="GitHub repository that holds the release (OWNER/NAME; default: $%s, else %s)"
                        % (ENV_REPO, DEFAULT_REPO))
    parser.add_argument("--tag", default="latest", help="release tag (default: latest)")
    parser.add_argument("--base-url", help="fetch the assets from this URL instead of a GitHub release (mirror, local server)")
    parser.add_argument("--dest", default=str(ROOT / "models"), help="where the files go (default: models/)")
    parser.add_argument("--model-size", default="all", choices=SIZES + ("all",),
                        help="all (default, both sizes, about 13 GB), 0.8b (about 2.9 GB with both variants) or 4b (about 10 GB)")
    parser.add_argument("--variant", action="append", choices=("fp16", "q4f16"),
                        help="only the graphs of this variant (repeatable); shared files (tokenizer, settings) always come along. "
                             "Default: every variant in the release")
    parser.add_argument("--include", action="append", help="only paths matching this glob (repeatable)")
    parser.add_argument("--exclude", action="append", help="skip paths matching this glob (repeatable)")
    parser.add_argument("--list", action="store_true", help="show the files of the release and exit")
    args = parser.parse_args()

    base = base_url(args)
    sizes = SIZES if args.model_size == "all" else (args.model_size,)
    files = []
    for size in sizes:
        url = "%s/manifest-%s.json" % (base, size)
        try:
            manifest = json.loads(fetch_bytes(url))
        except urllib.error.URLError as exc:
            raise SystemExit("could not read %s: %s" % (url, exc))
        if manifest.get("format") != 1:
            raise SystemExit("unknown manifest format: %r" % manifest.get("format"))
        print("%s %s" % (manifest.get("name"), manifest.get("version")))
        files += select(manifest["files"], args)
    total = sum(f["size"] for f in files)
    print("%d files, %s" % (len(files), _human(total)))
    if args.list:
        for f in files:
            print("  %10s  %s%s" % (_human(f["size"]), f["path"], "  (%d parts)" % len(f["parts"]) if f.get("parts") else ""))
        return 0

    dest = pathlib.Path(args.dest)
    for f in files:
        out = dest / f["path"]
        if out.exists() and out.stat().st_size == f["size"] and _sha256(out) == f["sha256"]:
            print("  have  %-44s %s" % (f["path"], _human(f["size"])))
            continue
        tmp = out.with_name(out.name + ".part")
        if not f.get("parts"):
            fetch("%s/%s" % (base, f["asset"]), tmp, f["size"], label=f["path"])
        else:  # join the parts into one file, one part at a time
            tmp.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "wb") as joined:
                for part in f["parts"]:
                    piece = tmp.with_name(part["asset"] + ".dl")
                    fetch("%s/%s" % (base, part["asset"]), piece, part["size"], label=part["asset"])
                    if _sha256(piece) != part["sha256"]:
                        piece.unlink()
                        raise SystemExit("%s: SHA-256 mismatch, file deleted" % part["asset"])
                    with open(piece, "rb") as src:
                        for block in iter(lambda: src.read(CHUNK), b""):
                            joined.write(block)
                    piece.unlink()
        if _sha256(tmp) != f["sha256"]:
            tmp.unlink()
            raise SystemExit("%s: SHA-256 mismatch, file deleted" % f["path"])
        os.replace(tmp, out)
        sys.stdout.write("\r  got   %-44s %s\n" % (f["path"], _human(f["size"])))
        sys.stdout.flush()

    print("\nready: %s" % dest)
    print("run:    uv run demo_inference_text.py%s" % (" --model-size 4b" if args.model_size == "4b" else ""))
    print("verify: uv run verify/run_all.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
