# -*- coding: utf-8 -*-
"""Provider selection and the check of what the sessions really use (pure logic: no GPU, no model)."""
import argparse

import pytest

from onejev_decide.devices import (CPU, CUDA, GpuUnavailableError, add_device_args, check_used, providers_from_args,
                                   select_providers)

CPU_BUILD = ["AzureExecutionProvider", CPU]
GPU_BUILD = ["TensorrtExecutionProvider", CUDA, CPU]


def test_cpu_is_the_default_and_ignores_cuda():
    assert select_providers(available=GPU_BUILD) == [CPU]
    assert select_providers("cpu", GPU_BUILD) == [CPU]


def test_cuda_puts_cuda_first_then_cpu():
    assert select_providers("cuda", GPU_BUILD) == [CUDA, CPU]


def test_auto_takes_cuda_when_present_else_cpu():
    assert select_providers("auto", GPU_BUILD) == [CUDA, CPU]
    assert select_providers("auto", CPU_BUILD) == [CPU]


def test_cuda_on_a_cpu_build_falls_back_without_require_gpu():
    assert select_providers("cuda", CPU_BUILD) == [CPU]


def test_require_gpu_on_a_cpu_build_is_an_error():
    with pytest.raises(GpuUnavailableError, match="not available"):
        select_providers("cuda", CPU_BUILD, require_gpu=True)
    with pytest.raises(GpuUnavailableError):
        select_providers("auto", CPU_BUILD, require_gpu=True)
    assert select_providers("cuda", GPU_BUILD, require_gpu=True) == [CUDA, CPU]


def test_require_gpu_contradicts_cpu_and_unknown_devices_are_rejected():
    with pytest.raises(ValueError):
        select_providers("cpu", GPU_BUILD, require_gpu=True)
    with pytest.raises(ValueError):
        select_providers("tpu", GPU_BUILD)


def test_check_used_is_silent_when_everything_is_as_requested():
    assert check_used([CUDA, CPU], [CUDA, CPU]) is None
    assert check_used([CPU], [CPU]) is None
    assert check_used([CPU], None) is None


def test_check_used_warns_about_a_silent_cpu_fallback():
    # onnxruntime builds the session on the CPU without raising when the CUDA DLLs are missing
    warning = check_used([CPU], [CUDA, CPU], name="decoder.onnx")
    assert warning and "decoder.onnx" in warning and "NOT GPU" in warning


def test_check_used_raises_with_require_gpu():
    with pytest.raises(GpuUnavailableError, match="not on CUDA"):
        check_used([CPU], [CUDA, CPU], require_gpu=True)
    assert check_used([CUDA, CPU], [CUDA, CPU], require_gpu=True) is None
    with pytest.raises(GpuUnavailableError):
        check_used([], [CUDA, CPU], require_gpu=True)


def _parse(argv):
    parser = argparse.ArgumentParser()
    add_device_args(parser)
    return parser.parse_args(argv)


def test_cli_defaults_to_the_cpu(monkeypatch):
    monkeypatch.setattr("onejev_decide.devices.available_providers", lambda: GPU_BUILD)
    assert providers_from_args(_parse([])) == ([CPU], False)


def test_cli_require_gpu_implies_cuda(monkeypatch):
    monkeypatch.setattr("onejev_decide.devices.available_providers", lambda: GPU_BUILD)
    assert providers_from_args(_parse(["--require-gpu"])) == ([CUDA, CPU], True)


def test_cli_require_gpu_without_cuda_exits_with_a_message(monkeypatch):
    monkeypatch.setattr("onejev_decide.devices.available_providers", lambda: CPU_BUILD)
    with pytest.raises(SystemExit, match="CUDAExecutionProvider is not available"):
        providers_from_args(_parse(["--device", "cuda", "--require-gpu"]))


def test_cli_cuda_on_a_cpu_build_warns_and_runs_on_the_cpu(monkeypatch, capsys):
    monkeypatch.setattr("onejev_decide.devices.available_providers", lambda: CPU_BUILD)
    assert providers_from_args(_parse(["--device", "cuda"])) == ([CPU], False)
    assert "running on the CPU" in capsys.readouterr().err
