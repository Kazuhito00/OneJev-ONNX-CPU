# -*- coding: utf-8 -*-
"""Which onnxruntime execution providers to use, and a check of which ones were really used.

The CPU is the default. ``cuda`` asks for CUDAExecutionProvider (then CPUExecutionProvider as the fallback for unsupported
nodes); ``auto`` takes CUDA when this onnxruntime build has it and the CPU otherwise.

onnxruntime does not raise when CUDA cannot be set up (missing cuBLAS / cuDNN DLLs, not enough VRAM): it writes a warning to
the log and quietly builds the session on the CPU. So what was *requested* proves nothing. ``check_used`` looks at what the
sessions report (``session.get_providers()``) and turns a silent CPU fallback into a warning, or into an error with
``require_gpu``.

Everything here is a plain function of its arguments (onnxruntime is imported lazily), so it is unit-tested without a GPU.
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, List, Optional, Sequence, Tuple

CPU = "CPUExecutionProvider"
CUDA = "CUDAExecutionProvider"
DEVICES = ("cpu", "cuda", "auto")


class GpuUnavailableError(RuntimeError):
    """``require_gpu`` was set but the model is not (fully) running on CUDA."""


def available_providers() -> List[str]:
    import onnxruntime as ort

    return list(ort.get_available_providers())


def select_providers(device: str = "cpu", available: Optional[Sequence[str]] = None,
                     require_gpu: bool = False) -> List[str]:
    """The ``providers`` list for ``ort.InferenceSession``.

    ``cpu`` -> [CPU]. ``cuda`` -> [CUDA, CPU]. ``auto`` -> [CUDA, CPU] if CUDA is available, else [CPU].
    ``cuda`` on a build without CUDA still returns [CPU] (``check_used`` then warns) unless ``require_gpu``, which raises.
    ``require_gpu`` with ``cpu`` is a contradiction and raises ValueError.
    """
    if device not in DEVICES:
        raise ValueError("device must be one of %s, not %r" % (", ".join(DEVICES), device))
    if require_gpu and device == "cpu":
        raise ValueError("require_gpu contradicts device 'cpu'")
    if device == "cpu":
        return [CPU]
    have = list(available_providers() if available is None else available)
    if device == "cuda" and CUDA not in have and not require_gpu:
        return [CPU]
    if CUDA in have:
        return [CUDA, CPU]
    if require_gpu:
        raise GpuUnavailableError(
            "CUDAExecutionProvider is not available in this onnxruntime build (available: %s). "
            "Install the gpu extra instead of onnxruntime (see the GPU section of the README)." % ", ".join(have))
    return [CPU]


def check_used(used: Sequence[str], requested: Optional[Sequence[str]], require_gpu: bool = False,
               name: str = "the model") -> Optional[str]:
    """Compare the provider a session really uses (``used[0]``) with what was requested.

    Returns a warning text when CUDA was requested but the session runs elsewhere (None when all is as asked).
    Raises GpuUnavailableError instead when ``require_gpu`` is set.
    """
    wants_cuda = bool(requested) and requested[0] == CUDA
    first = used[0] if used else None
    if require_gpu and first != CUDA:
        raise GpuUnavailableError(
            "%s runs on %s, not on CUDA (--require-gpu). Check that onnxruntime-gpu is installed instead of onnxruntime, "
            "and that the CUDA 13 / cuDNN 9 bin folders are on PATH." % (name, first))
    if wants_cuda and first != CUDA:
        return ("CUDA was requested but %s runs on %s (CUDA failed to load: DLLs missing on PATH, or not enough VRAM). "
                "This is NOT GPU inference." % (name, first))
    return None


def add_device_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", choices=DEVICES, default="cpu",
                        help="cpu (default) | cuda | auto (CUDA if this onnxruntime build has it). "
                             "cuda / auto need onnxruntime-gpu (the `gpu` extra) instead of onnxruntime")
    parser.add_argument("--require-gpu", action="store_true",
                        help="error out unless every graph really runs on CUDA (no silent fallback to the CPU); implies --device cuda")


def providers_from_args(args) -> Tuple[List[str], bool]:
    """``(providers, require_gpu)`` from the namespace ``add_device_args`` filled.

    Exits with an error message when ``--require-gpu`` cannot be met; warns when CUDA was asked for and is not in this build.
    """
    device = "cuda" if (args.require_gpu and args.device == "cpu") else args.device
    try:
        providers = select_providers(device, require_gpu=args.require_gpu)
    except GpuUnavailableError as exc:
        raise SystemExit("error: %s" % exc)
    if device == "cuda" and providers[0] != CUDA:
        print("warning: --device cuda, but this onnxruntime build has no CUDAExecutionProvider: running on the CPU "
              "(install the gpu extra instead of onnxruntime)", file=sys.stderr)
    return providers, bool(args.require_gpu)


def exit_on_gpu_error(exc: GpuUnavailableError):
    """For ``except GpuUnavailableError as exc:`` in the scripts: a one-line error instead of a traceback."""
    raise SystemExit("error: %s" % exc)


def report(label: str, providers_used: Dict[str, str], file=None) -> None:
    """One line: which provider each loaded graph runs on."""
    print("%s: %s" % (label, ", ".join("%s=%s" % (k, v.replace("ExecutionProvider", ""))
                                       for k, v in providers_used.items())), file=file or sys.stdout)
