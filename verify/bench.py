# -*- coding: utf-8 -*-
"""One measurement per process.

Each decoder call allocates a [1, seq, 248320] logits tensor (50 MB at 102
tokens), so measuring several configurations inside one process grows ORT's
arena and makes later numbers meaningless -- text_2q coming out *faster* than
text_1q is the tell. One point per process keeps the numbers comparable.
"""
import argparse
import ctypes
import json
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from onejev_decide import Decider, GpuUnavailableError, add_device_args, choice, providers_from_args  # noqa: E402
from onejev_decide.devices import exit_on_gpu_error  # noqa: E402
from onejev_decide.paths import MODEL_SIZES, default_model_dir  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
FILLER = (
    "The customer opened a ticket about the export button. It crashes in Safari "
    "but works in Chrome. Support replied within an hour and suggested switching "
    "browsers while engineering investigates the rendering bug. "
)


def peak_rss_mb():
    """Peak working set of this process, in MB (Windows).

    argtypes/restype must be declared: without them ctypes truncates the 64-bit
    process handle and the call quietly fails.
    """
    try:
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        query = kernel32.K32GetProcessMemoryInfo
        query.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        query.restype = wintypes.BOOL

        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        if query(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return round(counters.PeakWorkingSetSize / (1024 * 1024), 1)
        return None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-size", default="0.8b", choices=MODEL_SIZES,
                        help="0.8b (models/onnx, $ONEJEV_MODEL_DIR) or 4b (models/onnx-4b, $ONEJEV4_MODEL_DIR)")
    ap.add_argument("--model", default=None, help="model directory (overrides --model-size and the environment)")
    ap.add_argument("--variant", default="fp16")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--runs", type=int, default=7)
    add_device_args(ap)
    ap.add_argument("--repeat", type=int, default=1, help="how many times to repeat the state")
    ap.add_argument("--options", type=int, default=3)
    ap.add_argument("--questions", type=int, default=1)
    ap.add_argument("--image", type=int, default=0, help="attach an NxN image of this size")
    args = ap.parse_args()

    providers, require_gpu = providers_from_args(args)
    t0 = time.perf_counter()
    try:
        d = Decider(default_model_dir(args.model_size, ROOT, args.model), variant=args.variant, intra_op_threads=args.threads,
                    load_vision=bool(args.image), providers=providers, require_gpu=require_gpu)
    except GpuUnavailableError as exc:
        exit_on_gpu_error(exc)
    load_s = time.perf_counter() - t0
    provider = d.provider

    text = (FILLER * args.repeat).strip()
    state = {"report": text, "picture": "<image:1>"} if args.image else text
    if args.options == 3:
        q = choice("How severe is this bug?",
                   {"cosmetic": "cosmetic only",
                    "degraded": "degraded but there is a workaround",
                    "blocking": "blocking, no workaround"})
    else:
        q = choice("Which option?", {"opt%02d" % i: "option number %d" % i
                                     for i in range(args.options)})
    questions = {"q%02d" % i: q for i in range(args.questions)}

    images = []
    if args.image:
        from PIL import Image
        images = [Image.new("RGB", (args.image, args.image), (200, 40, 40))]

    d.decide(state, questions, images=images)  # warm
    times = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        d.decide(state, questions, images=images)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()

    print(json.dumps({
        "model": args.model or "OneJev-" + args.model_size,
        "variant": args.variant,
        "provider": provider,
        "providers_used": d.providers_used,
        "threads": args.threads,
        "options": args.options,
        "questions": args.questions,
        "image_px": args.image,
        "load_s": round(load_s, 2),
        "median_ms": round(statistics.median(times), 1),
        "min_ms": round(times[0], 1),
        "max_ms": round(times[-1], 1),
        "peak_rss_mb": peak_rss_mb(),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
