#!/usr/bin/env python3
"""Standalone GPU characterization and kernel timing report generator."""

import argparse
import dataclasses
import datetime as dt
import gc
import json
import logging
import math
import os
import platform
import random
import statistics
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

try:
    import torch
    import torch.nn.functional as F
    import torch.utils.benchmark as torch_benchmark
except ImportError as exc:
    raise SystemExit(
        "PyTorch is required. Install a CUDA or ROCm PyTorch build before running "
        "this script."
    ) from exc

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
    TRITON_IMPORT_ERROR: str | None = None
except ImportError as exc:
    triton = None
    tl = None
    HAS_TRITON = False
    TRITON_IMPORT_ERROR = str(exc)


SCHEMA_VERSION = "2.1.0"
SECONDS_TO_US = 1_000_000.0
MILLISECONDS_TO_US = 1000.0
VLLM_SOURCE = "https://github.com/vllm-project/vllm"
LOGGER = logging.getLogger("rocm_meter")
DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


class UnsupportedCase(RuntimeError):
    """Raised when a benchmark case is valid but unsupported by this system."""


@dataclasses.dataclass(slots=True)
class PreparedCase:
    function: Callable[[], Any]
    reference: Callable[[], Any]
    features: dict[str, Any]
    atol: float
    rtol: float


@dataclasses.dataclass(slots=True)
class BenchmarkCase:
    case_id: str
    category: str
    kernel: str
    implementation: str
    dtype: str
    shape: dict[str, int]
    prepare: Callable[[torch.device], PreparedCase]
    compile_function: bool = False
    jit_function: bool = False
    expected_error: type[BaseException] | None = None
    tags: tuple[str, ...] = ()


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def sorted_percentile(ordered: list[float], probability: float) -> float:
    """Linear-interpolated percentile of an already-sorted sample list."""
    if not ordered:
        raise ValueError("cannot compute a percentile of no values")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def sample_statistics(samples_us: list[float]) -> dict[str, Any]:
    if not samples_us:
        raise ValueError("at least one timing sample is required")
    count = len(samples_us)
    ordered = sorted(samples_us)
    mean = statistics.fmean(samples_us)
    stdev = statistics.stdev(samples_us) if count > 1 else 0.0
    standard_error = stdev / math.sqrt(count)
    p25 = sorted_percentile(ordered, 0.25)
    p50 = sorted_percentile(ordered, 0.50)
    p75 = sorted_percentile(ordered, 0.75)
    iqr = p75 - p25
    lower_fence = p25 - 1.5 * iqr
    upper_fence = p75 + 1.5 * iqr
    return {
        "samples_us": samples_us,
        "count": count,
        "mean_us": mean,
        "median_us": p50,
        "stdev_us": stdev,
        "standard_error_us": standard_error,
        "ci95_lower_us": max(0.0, mean - 1.96 * standard_error),
        "ci95_upper_us": mean + 1.96 * standard_error,
        "min_us": ordered[0],
        "p5_us": sorted_percentile(ordered, 0.05),
        "p25_us": p25,
        "p50_us": p50,
        "p75_us": p75,
        "p95_us": sorted_percentile(ordered, 0.95),
        "max_us": ordered[-1],
        "coefficient_of_variation_pct": 100.0 * stdev / mean if mean else None,
        "iqr_outlier_count": sum(
            value < lower_fence or value > upper_fence for value in samples_us
        ),
    }


def strict_json_value(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return strict_json_value(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): strict_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [strict_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_torch_utils(
    function: Callable[[], Any], warmups: int, samples: int
) -> tuple[list[float], dict[str, Any]]:
    for _ in range(warmups):
        function()
    num_threads = torch.get_num_threads()
    timer = torch_benchmark.Timer(
        stmt="function()",
        globals={"function": function},
        label="gpu-kernel",
        description="steady-state",
        num_threads=num_threads,
    )
    measurements = [timer.timeit(number=1) for _ in range(samples)]
    samples_us = [measurement.mean * SECONDS_TO_US for measurement in measurements]
    return samples_us, {
        "engine": "torch.utils.benchmark.Timer",
        "number_per_run": [measurement.number_per_run for measurement in measurements],
        "num_threads": num_threads,
        "label": "gpu-kernel",
        "description": "steady-state",
        "explicit_warmups": warmups,
        # Timer.timeit runs its own untimed max(number // 100, 2) warmup before
        # each measurement, so the callable executes more often than
        # `explicit_warmups + count` implies.
        "implicit_warmups_per_sample": 2,
        "total_invocations": warmups + 3 * samples,
    }


def benchmark_cuda_events(
    function: Callable[[], Any], device: torch.device, warmups: int, samples: int
) -> list[float]:
    """Device-side elapsed time per call, in microseconds.

    Interpretation: samples are recorded back-to-back with no host sync between
    them, so each interval covers the device timeline from one call to the next.
    When the device drains faster than the host can enqueue, the interval
    absorbs the host gap and reads *higher* than the wall-clock median. A
    negative `host_device_delta_median_us` is therefore the launch-bound signal,
    not a measurement error.
    """
    if device.type != "cuda":
        raise UnsupportedCase("CUDA-event timing requires a CUDA-like device")
    for _ in range(warmups):
        function()
    synchronize(device)
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(samples)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(samples)]
    for start, end in zip(starts, ends, strict=True):
        start.record()
        function()
        end.record()
    synchronize(device)
    # Event.elapsed_time is milliseconds.
    _record_device_window(starts[0], ends[-1])
    return [
        start.elapsed_time(end) * MILLISECONDS_TO_US
        for start, end in zip(starts, ends, strict=True)
    ]


# Absolute device-clock position of each case's timed region, so that an
# emulator's own per-dispatch trace can be split by case.  Elapsed time is only
# ever a difference, so an epoch event recorded once and never re-recorded is
# what turns it into a position.  Off unless asked for: on real hardware the
# epoch is the start of the process and the difference grows into the range
# where a float32 millisecond loses microsecond resolution.
_DEVICE_EPOCH: Any = None


def _device_window_enabled() -> bool:
    return os.environ.get("ROCM_METER_DEVICE_WINDOW", "") not in ("", "0")


def _open_device_epoch() -> None:
    """Fix the origin of the device timeline, before any work reaches it.

    Recorded as the first thing that touches the device, so that a position
    measured from it is a position on the *whole* device timeline.  An epoch
    taken later would still give consistent differences, but nothing outside
    this process -- an emulator's own dispatch trace, for one -- could line up
    with it, because the offset between the two origins would be unrecoverable.
    """
    global _DEVICE_EPOCH
    if not _device_window_enabled() or _DEVICE_EPOCH is not None:
        return
    try:
        _DEVICE_EPOCH = torch.cuda.Event(enable_timing=True)
        _DEVICE_EPOCH.record()
        torch.cuda.synchronize()
    except Exception:  # noqa: BLE001 - a diagnostic must never fail a run
        _DEVICE_EPOCH = None


def _record_device_window(first: Any, last: Any) -> None:
    if _DEVICE_EPOCH is None:
        return
    try:
        _LAST_DEVICE_WINDOW["start_us"] = (
            _DEVICE_EPOCH.elapsed_time(first) * MILLISECONDS_TO_US
        )
        _LAST_DEVICE_WINDOW["end_us"] = (
            _DEVICE_EPOCH.elapsed_time(last) * MILLISECONDS_TO_US
        )
    except Exception:  # noqa: BLE001 - a diagnostic must never fail a case
        _LAST_DEVICE_WINDOW.clear()


_LAST_DEVICE_WINDOW: dict[str, float] = {}


def flatten_tensors(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (list, tuple)):
        tensors: list[torch.Tensor] = []
        for item in value:
            tensors.extend(flatten_tensors(item))
        return tensors
    if isinstance(value, dict):
        tensors = []
        for item in value.values():
            tensors.extend(flatten_tensors(item))
        return tensors
    raise TypeError(f"kernel output contains unsupported value {type(value).__name__}")


def compare_outputs(
    actual: Any, expected: Any, atol: float, rtol: float
) -> dict[str, Any]:
    actual_tensors = flatten_tensors(actual)
    expected_tensors = flatten_tensors(expected)
    if len(actual_tensors) != len(expected_tensors):
        return {
            "passed": False,
            "reason": "output tensor count differs",
            "atol": atol,
            "rtol": rtol,
        }
    max_absolute_error = 0.0
    max_relative_error = 0.0
    all_close = True
    for actual_tensor, expected_tensor in zip(
        actual_tensors, expected_tensors, strict=True
    ):
        if actual_tensor.shape != expected_tensor.shape:
            return {
                "passed": False,
                "reason": (
                    f"shape differs: {tuple(actual_tensor.shape)} != "
                    f"{tuple(expected_tensor.shape)}"
                ),
                "atol": atol,
                "rtol": rtol,
            }
        actual_float = actual_tensor.detach().float()
        expected_float = expected_tensor.detach().float()
        difference = (actual_float - expected_float).abs()
        if difference.numel():
            max_absolute_error = max(max_absolute_error, difference.max().item())
            denominator = expected_float.abs().clamp_min(
                torch.finfo(torch.float32).tiny
            )
            max_relative_error = max(
                max_relative_error, (difference / denominator).max().item()
            )
        all_close = all_close and torch.allclose(
            actual_float, expected_float, atol=atol, rtol=rtol, equal_nan=True
        )
    return {
        "passed": all_close,
        "atol": atol,
        "rtol": rtol,
        "max_absolute_error": max_absolute_error,
        "max_relative_error": max_relative_error,
    }


def profile_function(
    function: Callable[[], Any], device: torch.device, iterations: int
) -> dict[str, Any]:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
    ) as profiler:
        for _ in range(iterations):
            function()
    synchronize(device)
    operators = []
    for event in profiler.key_averages(group_by_input_shape=True):
        self_device_us = getattr(
            event, "self_device_time_total", getattr(event, "self_cuda_time_total", 0.0)
        )
        total_device_us = getattr(
            event, "device_time_total", getattr(event, "cuda_time_total", 0.0)
        )
        operators.append(
            {
                "name": event.key,
                "calls": event.count,
                "input_shapes": event.input_shapes,
                "self_cpu_time_us": event.self_cpu_time_total,
                "total_cpu_time_us": event.cpu_time_total,
                "self_device_time_us": self_device_us,
                "total_device_time_us": total_device_us,
                "self_device_memory_bytes": getattr(
                    event, "self_device_memory_usage", 0
                ),
            }
        )
    operators.sort(
        key=lambda event: (event["self_device_time_us"], event["self_cpu_time_us"]),
        reverse=True,
    )
    total_self_cpu_time_us = sum(event["self_cpu_time_us"] for event in operators)
    total_self_device_time_us = sum(event["self_device_time_us"] for event in operators)
    return {
        "iterations": iterations,
        "operator_count": len(operators),
        "operators_truncated": len(operators) > 50,
        "operators": operators[:50],
        # Per-operator entries and these totals are cumulative over `iterations`
        # calls; the *_per_iteration fields are the ones comparable to the
        # per-call figures in `timing`.
        "totals_cumulative_over_iterations": True,
        "total_self_cpu_time_us": total_self_cpu_time_us,
        "total_self_device_time_us": total_self_device_time_us,
        "total_self_cpu_time_us_per_iteration": (
            total_self_cpu_time_us / iterations if iterations else None
        ),
        "total_self_device_time_us_per_iteration": (
            total_self_device_time_us / iterations if iterations else None
        ),
    }


def dtype_tolerance(dtype_name: str) -> tuple[float, float]:
    if dtype_name == "float32":
        return 1e-5, 1e-5
    if dtype_name == "float16":
        return 1e-2, 2e-3
    return 2e-2, 2e-2


# Adapted from vLLM's Apache-2.0 rms_norm implementation. Matches vLLM RMSNorm
# semantics, including float32 variance accumulation.
def rms_norm(
    x: torch.Tensor, weight: torch.Tensor | None, epsilon: float
) -> torch.Tensor:
    original_dtype = x.dtype
    x_float = x.float()
    variance = x_float.pow(2).mean(dim=-1, keepdim=True)
    normalized = x_float * torch.rsqrt(variance + epsilon)
    if weight is not None:
        normalized = normalized.to(weight.dtype) * weight
    return normalized.to(original_dtype)


# Adapted from vLLM's Apache-2.0 fused_add_rms_norm implementation. Matches vLLM
# fused-add RMSNorm semantics and returns the updated residual.
def fused_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor | None,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    original_dtype = x.dtype
    combined = x.float() + residual.float()
    residual_output = combined.to(original_dtype)
    variance = combined.pow(2).mean(dim=-1, keepdim=True)
    normalized = combined * torch.rsqrt(variance + epsilon)
    if weight is not None:
        normalized = normalized.to(weight.dtype) * weight
    return normalized.to(original_dtype), residual_output


def make_rms_norm_case(
    tokens: int,
    hidden_size: int,
    dtype_name: str,
    implementation: str,
    residual: bool = False,
) -> BenchmarkCase:
    dtype = DTYPES[dtype_name]
    element_size = torch.empty((), dtype=dtype).element_size()
    compile_function = implementation == "compiled"
    variant = "fused_add" if residual else "weighted"

    def prepare(device: torch.device) -> PreparedCase:
        generator = torch.Generator(device=device).manual_seed(
            1009 + tokens * 17 + hidden_size
        )
        x = torch.randn(
            (tokens, hidden_size), dtype=dtype, device=device, generator=generator
        )
        weight = torch.randn(
            (hidden_size,), dtype=dtype, device=device, generator=generator
        )
        epsilon = 1e-5
        atol, rtol = dtype_tolerance(dtype_name)
        if residual:
            residual_tensor = torch.randn(
                (tokens, hidden_size), dtype=dtype, device=device, generator=generator
            )

            def function() -> tuple[torch.Tensor, torch.Tensor]:
                return fused_add_rms_norm(x, residual_tensor, weight, epsilon)

            def reference() -> tuple[torch.Tensor, torch.Tensor]:
                combined = x.float() + residual_tensor.float()
                variance = combined.pow(2).mean(dim=-1, keepdim=True)
                output = combined * torch.rsqrt(variance + epsilon)
                return (output * weight.float()).to(dtype), combined.to(dtype)

            bytes_read = (2 * tokens * hidden_size + hidden_size) * element_size
            flops = 6 * tokens * hidden_size
        else:

            def function() -> torch.Tensor:
                return rms_norm(x, weight, epsilon)

            def reference() -> torch.Tensor:
                variance = x.float().pow(2).mean(dim=-1, keepdim=True)
                return (
                    x.float() * torch.rsqrt(variance + epsilon) * weight.float()
                ).to(dtype)

            bytes_read = (tokens * hidden_size + hidden_size) * element_size
            flops = 5 * tokens * hidden_size
        bytes_written = tokens * hidden_size * element_size * (2 if residual else 1)
        return PreparedCase(
            function=function,
            reference=reference,
            features={
                "input_shapes": [[tokens, hidden_size]],
                "input_strides": [[hidden_size, 1]],
                "layout": "contiguous",
                "estimated_flops": flops,
                "estimated_bytes_read": bytes_read,
                "estimated_bytes_written": bytes_written,
                "estimated_bytes_total": bytes_read + bytes_written,
                "estimate_quality": "analytical_approximation",
                "working_set_bytes": bytes_read + bytes_written,
                "epsilon": epsilon,
                "source": VLLM_SOURCE,
            },
            atol=atol,
            rtol=rtol,
        )

    return BenchmarkCase(
        case_id=f"rms_norm.{variant}.{implementation}.{dtype_name}.{tokens}x{hidden_size}",
        category="normalization",
        kernel="fused_add_rms_norm" if residual else "rms_norm",
        implementation=implementation,
        dtype=dtype_name,
        shape={"tokens": tokens, "hidden_size": hidden_size},
        prepare=prepare,
        compile_function=compile_function,
        tags=("vllm-derived", "edge" if tokens == 1 else "transformer"),
    )


def rms_norm_cases(tier: str, dtypes: set[str]) -> list[BenchmarkCase]:
    tier_shapes = {
        "quick": [(1, 768), (17, 1025)],
        "standard": [(1, 768), (128, 4096), (257, 5120)],
        "thorough": [
            (1, 768),
            (17, 1025),
            (128, 4096),
            (257, 5120),
            (2048, 8192),
        ],
    }
    cases = []
    for tokens, hidden_size in tier_shapes[tier]:
        for dtype_name in sorted(dtypes):
            for implementation in ("eager", "compiled"):
                cases.append(
                    make_rms_norm_case(
                        tokens, hidden_size, dtype_name, implementation, residual=False
                    )
                )
                cases.append(
                    make_rms_norm_case(
                        tokens, hidden_size, dtype_name, implementation, residual=True
                    )
                )
    return cases


def tensor_features(
    inputs: list[torch.Tensor],
    estimated_flops: int | None,
    estimated_bytes_written: int,
    **extra: Any,
) -> dict[str, Any]:
    bytes_read = sum(tensor.numel() * tensor.element_size() for tensor in inputs)
    total_bytes = bytes_read + estimated_bytes_written
    return {
        "input_shapes": [list(tensor.shape) for tensor in inputs],
        "input_strides": [list(tensor.stride()) for tensor in inputs],
        "layout": [
            "contiguous" if tensor.is_contiguous() else "strided" for tensor in inputs
        ],
        "estimated_flops": estimated_flops,
        "estimated_bytes_read": bytes_read,
        "estimated_bytes_written": estimated_bytes_written,
        "estimated_bytes_total": total_bytes,
        "arithmetic_intensity_flops_per_byte": (
            estimated_flops / total_bytes
            if estimated_flops is not None and total_bytes
            else None
        ),
        "working_set_bytes": total_bytes,
        "estimate_quality": "analytical_approximation",
        **extra,
    }


def make_elementwise_case(
    kernel: str,
    elements: int,
    dtype_name: str,
    implementation: str,
    layout: str = "contiguous",
) -> BenchmarkCase:
    dtype = DTYPES[dtype_name]
    compile_function = implementation == "compiled"

    def prepare(device: torch.device) -> PreparedCase:
        generator = torch.Generator(device=device).manual_seed(3109 + elements)
        physical_elements = (
            elements * 2
            if layout == "strided"
            else elements + (1 if layout == "misaligned" else 0)
        )
        storage = torch.randn(
            physical_elements, dtype=dtype, device=device, generator=generator
        )
        if layout == "strided":
            x = storage[::2]
        elif layout == "misaligned":
            x = storage[1:]
        else:
            x = storage
        y = torch.randn(elements, dtype=dtype, device=device, generator=generator)
        scalar = 0.375
        if kernel == "launch":
            function = lambda: x + 1.0
            reference = lambda: (x.float() + 1.0).to(dtype)
            flops = elements
            inputs = [x]
        elif kernel == "copy":
            function = lambda: x.clone()
            reference = lambda: x.float().to(dtype)
            flops = 0
            inputs = [x]
        elif kernel == "triad":
            function = lambda: x + scalar * y
            reference = lambda: (x.float() + scalar * y.float()).to(dtype)
            flops = 2 * elements
            inputs = [x, y]
        elif kernel == "atan":
            function = lambda: torch.atan(x)
            reference = lambda: torch.atan(x.float()).to(dtype)
            flops = None
            inputs = [x]
        elif kernel == "fused_elementwise":
            function = lambda: torch.tanh(torch.atan(x) * 0.5 + y).square()
            reference = (
                lambda: torch.tanh(torch.atan(x.float()) * 0.5 + y.float())
                .square()
                .to(dtype)
            )
            flops = None
            inputs = [x, y]
        else:
            raise AssertionError(f"unknown elementwise kernel {kernel}")
        atol, rtol = dtype_tolerance(dtype_name)
        return PreparedCase(
            function=function,
            reference=reference,
            features=tensor_features(
                inputs,
                flops,
                elements * x.element_size(),
                output_elements=elements,
                cache_regime=("launch-bound" if elements <= 4096 else "streaming"),
                access_pattern=layout,
            ),
            atol=max(atol, 2e-2 if kernel in {"atan", "fused_elementwise"} else atol),
            rtol=max(rtol, 2e-2 if kernel in {"atan", "fused_elementwise"} else rtol),
        )

    return BenchmarkCase(
        case_id=f"{kernel}.{implementation}.{dtype_name}.{layout}.{elements}",
        category="elementwise" if kernel not in {"copy", "triad"} else "memory",
        kernel=kernel,
        implementation=implementation,
        dtype=dtype_name,
        shape={"elements": elements},
        prepare=prepare,
        compile_function=compile_function,
        tags=("original", layout, "edge" if elements <= 4096 else "throughput"),
    )


def make_gemm_case(
    m: int, n: int, k: int, dtype_name: str, implementation: str
) -> BenchmarkCase:
    dtype = DTYPES[dtype_name]

    def prepare(device: torch.device) -> PreparedCase:
        generator = torch.Generator(device=device).manual_seed(4001 + m + n + k)
        a = torch.randn((m, k), dtype=dtype, device=device, generator=generator)
        b = torch.randn((k, n), dtype=dtype, device=device, generator=generator)
        function = lambda: torch.mm(a, b)
        reference = lambda: torch.mm(a.float(), b.float()).to(dtype)
        atol, rtol = dtype_tolerance(dtype_name)
        reduction_tolerance = max(atol, 2e-5 * math.sqrt(k))
        return PreparedCase(
            function=function,
            reference=reference,
            features=tensor_features(
                [a, b],
                2 * m * n * k,
                m * n * a.element_size(),
                output_elements=m * n,
                operation="C=A@B",
                shape_class=(
                    "decode"
                    if m == 1
                    else (
                        "irregular"
                        if any(value % 16 for value in (m, n, k))
                        else "dense"
                    )
                ),
            ),
            atol=reduction_tolerance,
            rtol=max(rtol, reduction_tolerance),
        )

    return BenchmarkCase(
        case_id=f"gemm.{implementation}.{dtype_name}.{m}x{n}x{k}",
        category="gemm",
        kernel="gemm",
        implementation=implementation,
        dtype=dtype_name,
        shape={"m": m, "n": n, "k": k},
        prepare=prepare,
        compile_function=implementation == "compiled",
        tags=("original", "decode" if m == 1 else "throughput"),
    )


def make_attention_case(
    operation: str,
    sequence: int,
    head_dim: int,
    dtype_name: str,
    implementation: str,
    causal: bool = False,
) -> BenchmarkCase:
    dtype = DTYPES[dtype_name]

    def prepare(device: torch.device) -> PreparedCase:
        generator = torch.Generator(device=device).manual_seed(
            5003 + sequence + head_dim
        )
        q = torch.randn(
            (sequence, head_dim), dtype=dtype, device=device, generator=generator
        )
        k = torch.randn(
            (sequence, head_dim), dtype=dtype, device=device, generator=generator
        )
        scores = torch.randn(
            (sequence, sequence), dtype=dtype, device=device, generator=generator
        )
        value = torch.randn(
            (sequence, head_dim), dtype=dtype, device=device, generator=generator
        )
        scale = head_dim**-0.5
        if operation == "qk":
            function = lambda: torch.mm(q, k.T) * scale
            reference = lambda: (torch.mm(q.float(), k.float().T) * scale).to(dtype)
            inputs = [q, k]
            flops = 2 * sequence * sequence * head_dim + sequence * sequence
            output_elements = sequence * sequence
        elif operation == "softmax":
            if causal:
                mask = torch.ones(
                    (sequence, sequence), device=device, dtype=torch.bool
                ).triu(1)
                function = lambda: torch.softmax(
                    scores.masked_fill(mask, float("-inf")), dim=-1
                )
                reference = lambda: torch.softmax(
                    scores.float().masked_fill(mask, float("-inf")), dim=-1
                ).to(dtype)
                inputs = [scores, mask]
            else:
                function = lambda: torch.softmax(scores, dim=-1)
                reference = lambda: torch.softmax(scores.float(), dim=-1).to(dtype)
                inputs = [scores]
            flops = 4 * sequence * sequence
            output_elements = sequence * sequence
        elif operation == "pv":
            probabilities = torch.softmax(scores.float(), dim=-1).to(dtype)
            function = lambda: torch.mm(probabilities, value)
            reference = lambda: torch.mm(probabilities.float(), value.float()).to(dtype)
            inputs = [probabilities, value]
            flops = 2 * sequence * sequence * head_dim
            output_elements = sequence * head_dim
        else:
            raise AssertionError(f"unknown attention operation {operation}")
        atol, rtol = dtype_tolerance(dtype_name)
        return PreparedCase(
            function=function,
            reference=reference,
            features=tensor_features(
                inputs,
                flops,
                output_elements * torch.empty((), dtype=dtype).element_size(),
                output_elements=output_elements,
                causal=causal,
                head_dim=head_dim,
            ),
            atol=max(atol, 2e-2),
            rtol=max(rtol, 2e-2),
        )

    suffix = ".causal" if causal else ""
    return BenchmarkCase(
        case_id=f"attention_{operation}{suffix}.{implementation}.{dtype_name}.{sequence}x{head_dim}",
        category="attention",
        kernel=f"attention_{operation}",
        implementation=implementation,
        dtype=dtype_name,
        shape={"sequence": sequence, "head_dim": head_dim},
        prepare=prepare,
        compile_function=implementation == "compiled",
        tags=("original", "causal" if causal else "non-causal"),
    )


def make_convolution_case(
    batch: int,
    channels: int,
    height: int,
    width: int,
    output_channels: int,
    kernel_size: int,
    groups: int,
    dtype_name: str,
    implementation: str,
    channels_last: bool,
) -> BenchmarkCase:
    dtype = DTYPES[dtype_name]

    def prepare(device: torch.device) -> PreparedCase:
        generator = torch.Generator(device=device).manual_seed(6007 + height + width)
        x = torch.randn(
            (batch, channels, height, width),
            dtype=dtype,
            device=device,
            generator=generator,
        )
        weight = torch.randn(
            (output_channels, channels // groups, kernel_size, kernel_size),
            dtype=dtype,
            device=device,
            generator=generator,
        )
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
            weight = weight.contiguous(memory_format=torch.channels_last)
        padding = kernel_size // 2
        function = lambda: F.conv2d(x, weight, padding=padding, groups=groups)
        reference = lambda: F.conv2d(
            x.float(), weight.float(), padding=padding, groups=groups
        ).to(dtype)
        output_elements = batch * output_channels * height * width
        flops = 2 * output_elements * (channels // groups) * kernel_size * kernel_size
        atol, rtol = dtype_tolerance(dtype_name)
        convolution_atol = 6.25e-2 if dtype_name == "float16" else max(atol, 3e-2)
        return PreparedCase(
            function=function,
            reference=reference,
            features=tensor_features(
                [x, weight],
                flops,
                output_elements * x.element_size(),
                output_elements=output_elements,
                groups=groups,
                memory_format="channels_last" if channels_last else "contiguous",
            ),
            atol=convolution_atol,
            rtol=max(rtol, 3e-2),
        )

    layout = "nhwc" if channels_last else "nchw"
    return BenchmarkCase(
        case_id=(
            f"conv2d.{implementation}.{dtype_name}.{layout}."
            f"{batch}x{channels}x{height}x{width}.{output_channels}x{kernel_size}.g{groups}"
        ),
        category="convolution",
        kernel="conv2d",
        implementation=implementation,
        dtype=dtype_name,
        shape={
            "batch": batch,
            "channels": channels,
            "height": height,
            "width": width,
            "output_channels": output_channels,
            "kernel_size": kernel_size,
            "groups": groups,
        },
        prepare=prepare,
        compile_function=implementation == "compiled",
        tags=("original", layout, "depthwise" if groups == channels else "dense"),
    )


# Adapted from vLLM ApplyRotaryEmb.forward_static (Apache-2.0). Implements the
# NeoX and GPT-J layouts supported by vLLM ApplyRotaryEmb.
def apply_rotary_embedding(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
    neox_style: bool,
    fp32_compute: bool,
) -> torch.Tensor:
    if rotary_dim <= 0 or rotary_dim > x.shape[-1] or rotary_dim % 2:
        raise ValueError(
            "rotary_dim must be a positive even value no larger than head_size"
        )
    original_dtype = x.dtype
    rotated = x[..., :rotary_dim].float() if fp32_compute else x[..., :rotary_dim]
    cos = cos.unsqueeze(-2).to(rotated.dtype)
    sin = sin.unsqueeze(-2).to(rotated.dtype)
    if neox_style:
        first, second = torch.chunk(rotated, 2, dim=-1)
    else:
        first, second = rotated[..., ::2], rotated[..., 1::2]
    first_out = first * cos - second * sin
    second_out = second * cos + first * sin
    if neox_style:
        rotary_output = torch.cat((first_out, second_out), dim=-1)
    else:
        rotary_output = torch.stack((first_out, second_out), dim=-1).flatten(-2)
    rotary_output = rotary_output.to(original_dtype)
    if rotary_dim == x.shape[-1]:
        return rotary_output
    return torch.cat((rotary_output, x[..., rotary_dim:]), dim=-1)


def make_rope_case(
    tokens: int,
    heads: int,
    head_size: int,
    rotary_dim: int,
    dtype_name: str,
    implementation: str,
    neox_style: bool,
    fp32_compute: bool,
) -> BenchmarkCase:
    dtype = DTYPES[dtype_name]

    def prepare(device: torch.device) -> PreparedCase:
        generator = torch.Generator(device=device).manual_seed(
            7001 + tokens + head_size
        )
        x = torch.randn(
            (tokens, heads, head_size), dtype=dtype, device=device, generator=generator
        )
        positions = torch.arange(tokens, dtype=torch.float32, device=device)
        frequencies = torch.arange(
            rotary_dim // 2, dtype=torch.float32, device=device
        ) / max(1, rotary_dim // 2)
        angles = positions[:, None] * torch.exp(
            -math.log(10000.0) * frequencies[None, :]
        )
        cos, sin = angles.cos(), angles.sin()
        function = lambda: apply_rotary_embedding(
            x, cos, sin, rotary_dim, neox_style, fp32_compute
        )

        def reference() -> torch.Tensor:
            rotated = x[..., :rotary_dim].float()
            if neox_style:
                first, second = torch.chunk(rotated, 2, dim=-1)
            else:
                first, second = rotated[..., ::2], rotated[..., 1::2]
            first_out = first * cos[:, None, :] - second * sin[:, None, :]
            second_out = second * cos[:, None, :] + first * sin[:, None, :]
            output = (
                torch.cat((first_out, second_out), dim=-1)
                if neox_style
                else torch.stack((first_out, second_out), dim=-1).flatten(-2)
            )
            if rotary_dim < head_size:
                output = torch.cat((output, x[..., rotary_dim:].float()), dim=-1)
            return output.to(dtype)

        elements = tokens * heads * rotary_dim
        atol, rtol = dtype_tolerance(dtype_name)
        return PreparedCase(
            function=function,
            reference=reference,
            features=tensor_features(
                [x, cos, sin],
                3 * elements,
                x.numel() * x.element_size(),
                output_elements=x.numel(),
                rotary_dim=rotary_dim,
                style="neox" if neox_style else "gptj",
                fp32_compute=fp32_compute,
                source=VLLM_SOURCE,
            ),
            atol=max(atol, 2e-2),
            rtol=max(rtol, 2e-2),
        )

    style = "neox" if neox_style else "gptj"
    precision = "fp32compute" if fp32_compute else "native"
    return BenchmarkCase(
        case_id=f"rope.{style}.{precision}.{implementation}.{dtype_name}.{tokens}x{heads}x{head_size}.r{rotary_dim}",
        category="position_embedding",
        kernel="rope",
        implementation=implementation,
        dtype=dtype_name,
        shape={
            "tokens": tokens,
            "heads": heads,
            "head_size": head_size,
            "rotary_dim": rotary_dim,
        },
        prepare=prepare,
        compile_function=implementation == "compiled",
        tags=("vllm-derived", style, "partial" if rotary_dim < head_size else "full"),
    )


# Adapted from vLLM SiluAndMul.forward_native (Apache-2.0). Implements vLLM
# SiluAndMul semantics: SiLU on the first half times the second.
def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    hidden = x.shape[-1] // 2
    if x.shape[-1] % 2:
        raise ValueError("SwiGLU input width must be even")
    return F.silu(x[..., :hidden]) * x[..., hidden:]


if HAS_TRITON:

    @triton.jit
    def _swiglu_kernel(
        input_pointer,
        output_pointer,
        hidden_size: tl.constexpr,
        block_size: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, block_size)
        mask = offsets < hidden_size
        row_input = input_pointer + row * 2 * hidden_size
        gate = tl.load(row_input + offsets, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(row_input + hidden_size + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        output = gate * tl.sigmoid(gate) * up
        tl.store(output_pointer + row * hidden_size + offsets, output, mask=mask)

    @triton.jit
    def _fused_qk_norm_rope_kernel(
        q_gate_pointer,
        k_pointer,
        q_output_pointer,
        k_output_pointer,
        gate_output_pointer,
        q_weight_pointer,
        k_weight_pointer,
        cache_pointer,
        position_pointer,
        q_gate_stride,
        k_stride,
        q_output_stride,
        k_output_stride,
        gate_output_stride,
        cache_stride,
        num_q_heads: tl.constexpr,
        num_kv_heads: tl.constexpr,
        head_dim: tl.constexpr,
        rotary_dim: tl.constexpr,
        half_rotary: tl.constexpr,
        epsilon: tl.constexpr,
        input_dtype: tl.constexpr,
        head_block: tl.constexpr,
        rotary_block: tl.constexpr,
        has_pass_through: tl.constexpr,
    ):
        token = tl.program_id(0)
        head = tl.program_id(1)
        is_key = head >= num_q_heads
        local_head = tl.where(is_key, head - num_q_heads, head)
        if is_key:
            input_base = k_pointer + token * k_stride + local_head * head_dim
            weight_pointer = k_weight_pointer
            output_base = (
                k_output_pointer + token * k_output_stride + local_head * head_dim
            )
        else:
            input_base = (
                q_gate_pointer + token * q_gate_stride + local_head * 2 * head_dim
            )
            weight_pointer = q_weight_pointer
            output_base = (
                q_output_pointer + token * q_output_stride + local_head * head_dim
            )

        head_offsets = tl.arange(0, head_block)
        head_mask = head_offsets < head_dim
        values = tl.load(input_base + head_offsets, mask=head_mask, other=0.0).to(
            tl.float32
        )
        variance = tl.sum(values * values, axis=0) / head_dim
        inverse_rms = tl.rsqrt(variance + epsilon)
        weights = tl.load(weight_pointer + head_offsets, mask=head_mask, other=0.0).to(
            tl.float32
        )
        normalized = (values * inverse_rms * weights).to(input_dtype).to(tl.float32)
        if has_pass_through:
            pass_mask = head_mask & (head_offsets >= rotary_dim)
            tl.store(output_base + head_offsets, normalized, mask=pass_mask)

        rotary_offsets = tl.arange(0, rotary_block)
        rotary_mask = rotary_offsets < half_rotary
        first = tl.load(input_base + rotary_offsets, mask=rotary_mask, other=0.0).to(
            tl.float32
        )
        second = tl.load(
            input_base + half_rotary + rotary_offsets,
            mask=rotary_mask,
            other=0.0,
        ).to(tl.float32)
        first_weight = tl.load(
            weight_pointer + rotary_offsets, mask=rotary_mask, other=0.0
        ).to(tl.float32)
        second_weight = tl.load(
            weight_pointer + half_rotary + rotary_offsets,
            mask=rotary_mask,
            other=0.0,
        ).to(tl.float32)
        first = (first * inverse_rms * first_weight).to(input_dtype).to(tl.float32)
        second = (second * inverse_rms * second_weight).to(input_dtype).to(tl.float32)

        position = tl.load(position_pointer + token).to(tl.int64)
        cache_offset = position * cache_stride
        cosine = tl.load(
            cache_pointer + cache_offset + rotary_offsets,
            mask=rotary_mask,
            other=0.0,
        ).to(tl.float32)
        sine = tl.load(
            cache_pointer + cache_offset + half_rotary + rotary_offsets,
            mask=rotary_mask,
            other=0.0,
        ).to(tl.float32)
        tl.store(
            output_base + rotary_offsets,
            first * cosine - second * sine,
            mask=rotary_mask,
        )
        tl.store(
            output_base + half_rotary + rotary_offsets,
            second * cosine + first * sine,
            mask=rotary_mask,
        )
        if not is_key:
            gate_input = input_base + head_dim
            gate_output = (
                gate_output_pointer + token * gate_output_stride + local_head * head_dim
            )
            gate = tl.load(gate_input + head_offsets, mask=head_mask, other=0.0)
            tl.store(gate_output + head_offsets, gate, mask=head_mask)

    @triton.jit
    def _scaled_int8_mm_kernel(
        a_pointer,
        b_pointer,
        scale_a_pointer,
        scale_b_pointer,
        bias_pointer,
        output_pointer,
        m_size: tl.constexpr,
        n_size: tl.constexpr,
        k_size: tl.constexpr,
        stride_am: tl.constexpr,
        stride_ak: tl.constexpr,
        stride_bk: tl.constexpr,
        stride_bn: tl.constexpr,
        stride_cm: tl.constexpr,
        stride_cn: tl.constexpr,
        per_token_a: tl.constexpr,
        per_channel_b: tl.constexpr,
        has_bias: tl.constexpr,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
        block_k: tl.constexpr,
    ):
        program_m = tl.program_id(0)
        program_n = tl.program_id(1)
        offsets_m = program_m * block_m + tl.arange(0, block_m)
        offsets_n = program_n * block_n + tl.arange(0, block_n)
        offsets_k = tl.arange(0, block_k)
        accumulator = tl.zeros((block_m, block_n), dtype=tl.int32)
        for block_start in range(0, k_size, block_k):
            active_k = block_start + offsets_k
            a_pointers = (
                a_pointer
                + offsets_m[:, None] * stride_am
                + active_k[None, :] * stride_ak
            )
            b_pointers = (
                b_pointer
                + active_k[:, None] * stride_bk
                + offsets_n[None, :] * stride_bn
            )
            a_values = tl.load(
                a_pointers,
                mask=(offsets_m[:, None] < m_size) & (active_k[None, :] < k_size),
                other=0,
            )
            b_values = tl.load(
                b_pointers,
                mask=(active_k[:, None] < k_size) & (offsets_n[None, :] < n_size),
                other=0,
            )
            accumulator += tl.dot(a_values, b_values, out_dtype=tl.int32)
        if per_token_a:
            scale_a = tl.load(
                scale_a_pointer + offsets_m, mask=offsets_m < m_size, other=0.0
            )
        else:
            scale_a = tl.load(scale_a_pointer)
        if per_channel_b:
            scale_b = tl.load(
                scale_b_pointer + offsets_n, mask=offsets_n < n_size, other=0.0
            )
        else:
            scale_b = tl.load(scale_b_pointer)
        output = accumulator.to(tl.float32) * scale_a[:, None] * scale_b[None, :]
        if has_bias:
            bias = tl.load(bias_pointer + offsets_n, mask=offsets_n < n_size, other=0.0)
            output += bias[None, :]
        output_pointers = (
            output_pointer
            + offsets_m[:, None] * stride_cm
            + offsets_n[None, :] * stride_cn
        )
        tl.store(
            output_pointers,
            output,
            mask=(offsets_m[:, None] < m_size) & (offsets_n[None, :] < n_size),
        )

    @triton.jit
    def _scaled_fp8_mm_kernel(
        a_pointer,
        b_pointer,
        scale_a_pointer,
        scale_b_pointer,
        output_pointer,
        m_size: tl.constexpr,
        n_size: tl.constexpr,
        k_size: tl.constexpr,
        stride_am: tl.constexpr,
        stride_ak: tl.constexpr,
        stride_bk: tl.constexpr,
        stride_bn: tl.constexpr,
        stride_cm: tl.constexpr,
        stride_cn: tl.constexpr,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
        block_k: tl.constexpr,
    ):
        program_m = tl.program_id(0)
        program_n = tl.program_id(1)
        offsets_m = program_m * block_m + tl.arange(0, block_m)
        offsets_n = program_n * block_n + tl.arange(0, block_n)
        offsets_k = tl.arange(0, block_k)
        accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)
        for block_start in range(0, k_size, block_k):
            active_k = block_start + offsets_k
            a_values = tl.load(
                a_pointer
                + offsets_m[:, None] * stride_am
                + active_k[None, :] * stride_ak,
                mask=(offsets_m[:, None] < m_size) & (active_k[None, :] < k_size),
                other=0.0,
            )
            b_values = tl.load(
                b_pointer
                + active_k[:, None] * stride_bk
                + offsets_n[None, :] * stride_bn,
                mask=(active_k[:, None] < k_size) & (offsets_n[None, :] < n_size),
                other=0.0,
            )
            accumulator += tl.dot(a_values, b_values, out_dtype=tl.float32)
        scale_a = tl.load(scale_a_pointer)
        scale_b = tl.load(scale_b_pointer)
        output = accumulator * scale_a * scale_b
        output_pointers = (
            output_pointer
            + offsets_m[:, None] * stride_cm
            + offsets_n[None, :] * stride_cn
        )
        tl.store(
            output_pointers,
            output,
            mask=(offsets_m[:, None] < m_size) & (offsets_n[None, :] < n_size),
        )


def triton_swiglu(x: torch.Tensor) -> torch.Tensor:
    if not HAS_TRITON:
        raise UnsupportedCase(f"Triton is unavailable: {TRITON_IMPORT_ERROR}")
    if x.ndim != 2 or x.shape[1] % 2:
        raise ValueError("Triton SwiGLU expects a 2D tensor with an even width")
    hidden_size = x.shape[1] // 2
    block_size = triton.next_power_of_2(hidden_size)
    if block_size > 65536:
        raise UnsupportedCase("SwiGLU hidden size exceeds the standalone Triton block")
    output = torch.empty((x.shape[0], hidden_size), dtype=x.dtype, device=x.device)
    _swiglu_kernel[(x.shape[0],)](
        x,
        output,
        hidden_size=hidden_size,
        block_size=block_size,
        num_warps=min(8, max(1, block_size // 256)),
    )
    return output


def triton_fused_qk_norm_rope(
    q_gate: torch.Tensor,
    key: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cache: torch.Tensor,
    positions: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not HAS_TRITON:
        raise UnsupportedCase(f"Triton is unavailable: {TRITON_IMPORT_ERROR}")
    if q_gate.dtype not in (torch.float16, torch.bfloat16):
        raise UnsupportedCase("fused QK norm/RoPE supports float16 and bfloat16")
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2:
        raise ValueError("rotary_dim must be positive, even, and <= head_dim")
    tokens = q_gate.shape[0]
    q_output = torch.empty(
        (tokens, num_q_heads * head_dim), dtype=q_gate.dtype, device=q_gate.device
    )
    k_output = torch.empty(
        (tokens, num_kv_heads * head_dim), dtype=key.dtype, device=key.device
    )
    gate_output = torch.empty_like(q_output)
    if tokens == 0:
        return q_output, k_output, gate_output
    half_rotary = rotary_dim // 2
    head_block = triton.next_power_of_2(head_dim)
    rotary_block = triton.next_power_of_2(half_rotary)
    _fused_qk_norm_rope_kernel[(tokens, num_q_heads + num_kv_heads)](
        q_gate,
        key,
        q_output,
        k_output,
        gate_output,
        q_weight,
        k_weight,
        cache,
        positions,
        q_gate.stride(0),
        key.stride(0),
        q_output.stride(0),
        k_output.stride(0),
        gate_output.stride(0),
        cache.stride(0),
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        half_rotary=half_rotary,
        epsilon=epsilon,
        input_dtype=tl.bfloat16 if q_gate.dtype == torch.bfloat16 else tl.float16,
        head_block=head_block,
        rotary_block=rotary_block,
        has_pass_through=rotary_dim < head_dim,
        num_warps=max(1, head_block // 64),
        num_stages=2,
    )
    return q_output, k_output, gate_output


def triton_scaled_int8_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    output_dtype: torch.dtype,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    if not HAS_TRITON:
        raise UnsupportedCase(f"Triton is unavailable: {TRITON_IMPORT_ERROR}")
    if a.dtype != torch.int8 or b.dtype != torch.int8:
        raise UnsupportedCase("standalone scaled-MM currently supports int8 inputs")
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("scaled-MM requires compatible 2D matrices")
    m_size, k_size = a.shape
    n_size = b.shape[1]
    if scale_a.numel() not in (1, m_size) or scale_b.numel() not in (1, n_size):
        raise ValueError("scale shapes must be scalar, per-token A, or per-channel B")
    output = torch.empty((m_size, n_size), dtype=output_dtype, device=a.device)
    block_m, block_n, block_k = 32, 32, 32
    bias_pointer = bias if bias is not None else output
    _scaled_int8_mm_kernel[
        (triton.cdiv(m_size, block_m), triton.cdiv(n_size, block_n))
    ](
        a,
        b,
        scale_a,
        scale_b,
        bias_pointer,
        output,
        m_size=m_size,
        n_size=n_size,
        k_size=k_size,
        stride_am=a.stride(0),
        stride_ak=a.stride(1),
        stride_bk=b.stride(0),
        stride_bn=b.stride(1),
        stride_cm=output.stride(0),
        stride_cn=output.stride(1),
        per_token_a=scale_a.numel() == m_size,
        per_channel_b=scale_b.numel() == n_size,
        has_bias=bias is not None,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
    )
    return output


def triton_scaled_fp8_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    if not HAS_TRITON:
        raise UnsupportedCase(f"Triton is unavailable: {TRITON_IMPORT_ERROR}")
    if "float8" not in str(a.dtype) or a.dtype != b.dtype:
        raise UnsupportedCase("scaled FP8 GEMM requires matching float8 inputs")
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("scaled FP8 GEMM requires compatible 2D matrices")
    m_size, k_size = a.shape
    n_size = b.shape[1]
    output = torch.empty((m_size, n_size), dtype=output_dtype, device=a.device)
    block_m, block_n, block_k = 32, 32, 32
    _scaled_fp8_mm_kernel[(triton.cdiv(m_size, block_m), triton.cdiv(n_size, block_n))](
        a,
        b,
        scale_a,
        scale_b,
        output,
        m_size=m_size,
        n_size=n_size,
        k_size=k_size,
        stride_am=a.stride(0),
        stride_ak=a.stride(1),
        stride_bk=b.stride(0),
        stride_bn=b.stride(1),
        stride_cm=output.stride(0),
        stride_cn=output.stride(1),
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
    )
    return output


def make_swiglu_case(
    tokens: int, hidden_size: int, dtype_name: str, implementation: str
) -> BenchmarkCase:
    dtype = DTYPES[dtype_name]

    def prepare(device: torch.device) -> PreparedCase:
        generator = torch.Generator(device=device).manual_seed(
            8009 + tokens + hidden_size
        )
        x = torch.randn(
            (tokens, 2 * hidden_size), dtype=dtype, device=device, generator=generator
        )
        function = lambda: silu_and_mul(x)
        reference = lambda: (
            F.silu(x[..., :hidden_size].float()) * x[..., hidden_size:].float()
        ).to(dtype)
        atol, rtol = dtype_tolerance(dtype_name)
        return PreparedCase(
            function=function,
            reference=reference,
            features=tensor_features(
                [x],
                None,
                tokens * hidden_size * x.element_size(),
                output_elements=tokens * hidden_size,
                source=VLLM_SOURCE,
                activation="silu",
            ),
            atol=max(atol, 2e-2),
            rtol=max(rtol, 2e-2),
        )

    return BenchmarkCase(
        case_id=f"swiglu.{implementation}.{dtype_name}.{tokens}x{hidden_size}",
        category="activation",
        kernel="swiglu",
        implementation=implementation,
        dtype=dtype_name,
        shape={"tokens": tokens, "hidden_size": hidden_size},
        prepare=prepare,
        compile_function=implementation == "compiled",
        tags=("vllm-derived", "edge" if tokens == 1 else "transformer"),
    )


def make_triton_swiglu_case(
    tokens: int, hidden_size: int, dtype_name: str, triton_enabled: bool
) -> BenchmarkCase:
    dtype = DTYPES[dtype_name]

    def prepare(device: torch.device) -> PreparedCase:
        if not triton_enabled:
            raise UnsupportedCase("Triton cases were disabled by --no-triton")
        if not HAS_TRITON:
            raise UnsupportedCase(f"Triton is unavailable: {TRITON_IMPORT_ERROR}")
        if dtype_name == "float32":
            raise UnsupportedCase("Triton SwiGLU cases target float16 and bfloat16")
        generator = torch.Generator(device=device).manual_seed(
            8101 + tokens + hidden_size
        )
        x = torch.randn(
            (tokens, 2 * hidden_size),
            dtype=dtype,
            device=device,
            generator=generator,
        )
        function = lambda: triton_swiglu(x)
        reference = lambda: silu_and_mul(x)
        atol, rtol = dtype_tolerance(dtype_name)
        return PreparedCase(
            function=function,
            reference=reference,
            features=tensor_features(
                [x],
                None,
                tokens * hidden_size * x.element_size(),
                output_elements=tokens * hidden_size,
                source=VLLM_SOURCE,
                block_size=(1 << (hidden_size - 1).bit_length()),
            ),
            atol=max(atol, 2e-2),
            rtol=max(rtol, 2e-2),
        )

    return BenchmarkCase(
        case_id=f"swiglu.triton.{dtype_name}.{tokens}x{hidden_size}",
        category="activation",
        kernel="swiglu",
        implementation="triton",
        dtype=dtype_name,
        shape={"tokens": tokens, "hidden_size": hidden_size},
        prepare=prepare,
        jit_function=True,
        tags=("vllm-derived", "triton", "edge" if tokens == 1 else "transformer"),
    )


def make_fused_qk_case(
    tokens: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    dtype_name: str,
    triton_enabled: bool,
) -> BenchmarkCase:
    dtype = DTYPES[dtype_name]

    def prepare(device: torch.device) -> PreparedCase:
        if not triton_enabled:
            raise UnsupportedCase("Triton cases were disabled by --no-triton")
        if not HAS_TRITON:
            raise UnsupportedCase(f"Triton is unavailable: {TRITON_IMPORT_ERROR}")
        generator = torch.Generator(device=device).manual_seed(9001 + tokens + head_dim)
        q_gate = torch.randn(
            (tokens, num_q_heads * 2 * head_dim),
            dtype=dtype,
            device=device,
            generator=generator,
        )
        key = torch.randn(
            (tokens, num_kv_heads * head_dim),
            dtype=dtype,
            device=device,
            generator=generator,
        )
        q_weight = torch.randn(
            (head_dim,), dtype=dtype, device=device, generator=generator
        )
        k_weight = torch.randn(
            (head_dim,), dtype=dtype, device=device, generator=generator
        )
        positions = torch.arange(tokens, dtype=torch.int64, device=device)
        half_rotary = rotary_dim // 2
        frequencies = torch.arange(
            half_rotary, dtype=torch.float32, device=device
        ) / max(1, half_rotary)
        angles = positions.float()[:, None] * torch.exp(
            -math.log(10000.0) * frequencies[None, :]
        )
        cache = torch.cat((angles.cos(), angles.sin()), dim=-1).to(dtype)
        epsilon = 1e-6
        function = lambda: triton_fused_qk_norm_rope(
            q_gate,
            key,
            q_weight,
            k_weight,
            cache,
            positions,
            num_q_heads,
            num_kv_heads,
            head_dim,
            rotary_dim,
            epsilon,
        )

        def reference() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            q_heads = q_gate.view(tokens, num_q_heads, 2, head_dim)
            q_values = q_heads[:, :, 0, :]
            gate = q_heads[:, :, 1, :].reshape(tokens, num_q_heads * head_dim)
            k_values = key.view(tokens, num_kv_heads, head_dim)
            q_variance = q_values.float().square().mean(dim=-1, keepdim=True)
            k_variance = k_values.float().square().mean(dim=-1, keepdim=True)
            q_normalized = (
                q_values.float() * torch.rsqrt(q_variance + epsilon) * q_weight.float()
            ).to(dtype)
            k_normalized = (
                k_values.float() * torch.rsqrt(k_variance + epsilon) * k_weight.float()
            ).to(dtype)
            cosine = cache[:, :half_rotary].float()
            sine = cache[:, half_rotary:].float()
            q_output = apply_rotary_embedding(
                q_normalized, cosine, sine, rotary_dim, True, True
            )
            k_output = apply_rotary_embedding(
                k_normalized, cosine, sine, rotary_dim, True, True
            )
            return (
                q_output.reshape(tokens, num_q_heads * head_dim),
                k_output.reshape(tokens, num_kv_heads * head_dim),
                gate,
            )

        output_elements = tokens * (2 * num_q_heads + num_kv_heads) * head_dim
        return PreparedCase(
            function=function,
            reference=reference,
            features=tensor_features(
                [q_gate, key, q_weight, k_weight, cache, positions],
                None,
                output_elements * torch.empty((), dtype=dtype).element_size(),
                output_elements=output_elements,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                rotary_dim=rotary_dim,
                source=VLLM_SOURCE,
            ),
            atol=2e-2 if dtype_name == "bfloat16" else 3e-3,
            rtol=2e-2 if dtype_name == "bfloat16" else 3e-3,
        )

    return BenchmarkCase(
        case_id=(
            f"fused_qk_norm_rope.triton.{dtype_name}.{tokens}t."
            f"{num_q_heads}q.{num_kv_heads}kv.{head_dim}d.r{rotary_dim}"
        ),
        category="fused_transformer",
        kernel="fused_qk_norm_rope",
        implementation="triton",
        dtype=dtype_name,
        shape={
            "tokens": tokens,
            "num_q_heads": num_q_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "rotary_dim": rotary_dim,
        },
        prepare=prepare,
        jit_function=True,
        tags=(
            "vllm-derived",
            "triton",
            "partial" if rotary_dim < head_dim else "full",
        ),
    )


def make_scaled_int8_case(
    m_size: int,
    n_size: int,
    k_size: int,
    output_dtype_name: str,
    per_token_a: bool,
    per_channel_b: bool,
    with_bias: bool,
    triton_enabled: bool,
) -> BenchmarkCase:
    output_dtype = DTYPES[output_dtype_name]

    def prepare(device: torch.device) -> PreparedCase:
        if not triton_enabled:
            raise UnsupportedCase("Triton cases were disabled by --no-triton")
        if not HAS_TRITON:
            raise UnsupportedCase(f"Triton is unavailable: {TRITON_IMPORT_ERROR}")
        generator = torch.Generator(device=device).manual_seed(
            10007 + m_size + n_size + k_size
        )
        a = torch.randint(
            -8,
            8,
            (m_size, k_size),
            dtype=torch.int8,
            device=device,
            generator=generator,
        )
        b = torch.randint(
            -8,
            8,
            (k_size, n_size),
            dtype=torch.int8,
            device=device,
            generator=generator,
        )
        scale_a = (
            torch.rand(
                (m_size if per_token_a else 1,),
                dtype=torch.float32,
                device=device,
                generator=generator,
            )
            * 0.05
        )
        scale_b = (
            torch.rand(
                (n_size if per_channel_b else 1,),
                dtype=torch.float32,
                device=device,
                generator=generator,
            )
            * 0.05
        )
        bias = (
            torch.randn(
                (n_size,),
                dtype=output_dtype,
                device=device,
                generator=generator,
            )
            if with_bias
            else None
        )
        function = lambda: triton_scaled_int8_mm(
            a, b, scale_a, scale_b, output_dtype, bias
        )

        def reference() -> torch.Tensor:
            output = torch.mm(a.float(), b.float())
            output = output * scale_a.reshape(-1, 1)
            output = output * scale_b.reshape(1, -1)
            output = output.to(output_dtype)
            if bias is not None:
                output = output + bias
            return output

        output_elements = m_size * n_size
        inputs = [a, b, scale_a, scale_b] + ([] if bias is None else [bias])
        return PreparedCase(
            function=function,
            reference=reference,
            features=tensor_features(
                inputs,
                2 * m_size * n_size * k_size,
                output_elements * torch.empty((), dtype=output_dtype).element_size(),
                output_elements=output_elements,
                quantization="int8",
                scale_a="per_token" if per_token_a else "per_tensor",
                scale_b="per_channel" if per_channel_b else "per_tensor",
                bias=with_bias,
                source=VLLM_SOURCE,
            ),
            atol=0.15,
            rtol=0.15,
        )

    scale_mode = (
        f"{'token' if per_token_a else 'tensor'}a."
        f"{'channel' if per_channel_b else 'tensor'}b"
    )
    bias_mode = "bias" if with_bias else "nobias"
    return BenchmarkCase(
        case_id=(
            f"scaled_int8_mm.triton.{output_dtype_name}."
            f"{m_size}x{n_size}x{k_size}.{scale_mode}.{bias_mode}"
        ),
        category="quantized_gemm",
        kernel="scaled_int8_mm",
        implementation="triton",
        dtype="int8",
        shape={"m": m_size, "n": n_size, "k": k_size},
        prepare=prepare,
        jit_function=True,
        tags=("vllm-derived", "triton", "quantized", scale_mode),
    )


def make_scaled_fp8_case(
    m_size: int,
    n_size: int,
    k_size: int,
    output_dtype_name: str,
    triton_enabled: bool,
) -> BenchmarkCase:
    output_dtype = DTYPES[output_dtype_name]

    def prepare(device: torch.device) -> PreparedCase:
        if not triton_enabled:
            raise UnsupportedCase("Triton cases were disabled by --no-triton")
        if not HAS_TRITON:
            raise UnsupportedCase(f"Triton is unavailable: {TRITON_IMPORT_ERROR}")
        fp8_name = "float8_e4m3fnuz" if torch.version.hip else "float8_e4m3fn"
        if not hasattr(torch, fp8_name):
            raise UnsupportedCase(f"PyTorch does not expose {fp8_name}")
        fp8_dtype = getattr(torch, fp8_name)
        generator = torch.Generator(device=device).manual_seed(
            11003 + m_size + n_size + k_size
        )
        a_source = torch.randn(
            (m_size, k_size),
            dtype=torch.float16,
            device=device,
            generator=generator,
        ).clamp(-4, 4)
        b_source = torch.randn(
            (k_size, n_size),
            dtype=torch.float16,
            device=device,
            generator=generator,
        ).clamp(-4, 4)
        a = a_source.to(fp8_dtype)
        b = b_source.to(fp8_dtype)
        scale_a = torch.tensor([0.25], dtype=torch.float32, device=device)
        scale_b = torch.tensor([0.5], dtype=torch.float32, device=device)
        function = lambda: triton_scaled_fp8_mm(a, b, scale_a, scale_b, output_dtype)
        reference = lambda: (torch.mm(a.float(), b.float()) * scale_a * scale_b).to(
            output_dtype
        )
        output_elements = m_size * n_size
        return PreparedCase(
            function=function,
            reference=reference,
            features=tensor_features(
                [a, b, scale_a, scale_b],
                2 * m_size * n_size * k_size,
                output_elements * torch.empty((), dtype=output_dtype).element_size(),
                output_elements=output_elements,
                quantization=fp8_name,
                scale_a="per_tensor",
                scale_b="per_tensor",
                source=VLLM_SOURCE,
            ),
            atol=0.2,
            rtol=0.2,
        )

    return BenchmarkCase(
        case_id=(
            f"scaled_fp8_mm.triton.{output_dtype_name}." f"{m_size}x{n_size}x{k_size}"
        ),
        category="quantized_gemm",
        kernel="scaled_fp8_mm",
        implementation="triton",
        dtype="float8",
        shape={"m": m_size, "n": n_size, "k": k_size},
        prepare=prepare,
        jit_function=True,
        tags=("vllm-derived", "triton", "quantized", "fp8"),
    )


def make_concurrent_gemm_case(size: int, dtype_name: str) -> BenchmarkCase:
    dtype = DTYPES[dtype_name]

    def prepare(device: torch.device) -> PreparedCase:
        generator = torch.Generator(device=device).manual_seed(12007 + size)
        a = torch.randn((size, size), dtype=dtype, device=device, generator=generator)
        b = torch.randn((size, size), dtype=dtype, device=device, generator=generator)
        c = torch.randn((size, size), dtype=dtype, device=device, generator=generator)
        d = torch.randn((size, size), dtype=dtype, device=device, generator=generator)
        stream_a = torch.cuda.Stream(device=device)
        stream_b = torch.cuda.Stream(device=device)

        def function() -> tuple[torch.Tensor, torch.Tensor]:
            current = torch.cuda.current_stream(device)
            stream_a.wait_stream(current)
            stream_b.wait_stream(current)
            with torch.cuda.stream(stream_a):
                first = torch.mm(a, b)
            with torch.cuda.stream(stream_b):
                second = torch.mm(c, d)
            current.wait_stream(stream_a)
            current.wait_stream(stream_b)
            return first, second

        reference = lambda: (
            torch.mm(a.float(), b.float()).to(dtype),
            torch.mm(c.float(), d.float()).to(dtype),
        )
        atol, rtol = dtype_tolerance(dtype_name)
        return PreparedCase(
            function=function,
            reference=reference,
            features=tensor_features(
                [a, b, c, d],
                4 * size * size * size,
                2 * size * size * a.element_size(),
                output_elements=2 * size * size,
                stream_count=2,
                concurrency="two_independent_gemms",
            ),
            atol=max(atol, 2e-5 * math.sqrt(size)),
            rtol=max(rtol, 2e-5 * math.sqrt(size)),
        )

    return BenchmarkCase(
        case_id=f"concurrent_gemm.streams2.{dtype_name}.{size}",
        category="concurrency",
        kernel="concurrent_gemm",
        implementation="eager",
        dtype=dtype_name,
        shape={"size": size, "stream_count": 2},
        prepare=prepare,
        tags=("original", "concurrent", "streams"),
    )


def make_expected_error_case(kernel: str, dtype_name: str) -> BenchmarkCase:
    dtype = DTYPES[dtype_name]

    def prepare(device: torch.device) -> PreparedCase:
        if kernel == "rope_invalid_rotary_dim":
            x = torch.ones((1, 1, 64), dtype=dtype, device=device)
            cos = torch.ones((1, 31), dtype=torch.float32, device=device)
            sin = torch.zeros_like(cos)
            function = lambda: apply_rotary_embedding(x, cos, sin, 63, True, False)
        elif kernel == "swiglu_odd_width":
            x = torch.ones((1, 65), dtype=dtype, device=device)
            function = lambda: silu_and_mul(x)
        else:
            raise AssertionError(f"unknown expected-error kernel {kernel}")
        return PreparedCase(
            function=function,
            reference=function,
            features={
                "expected_error": "ValueError",
                "edge_case": kernel,
                "estimated_flops": None,
                "estimated_bytes_total": 0,
            },
            atol=0.0,
            rtol=0.0,
        )

    return BenchmarkCase(
        case_id=f"edge.{kernel}.{dtype_name}",
        category="edge_case",
        kernel=kernel,
        implementation="eager",
        dtype=dtype_name,
        shape={},
        prepare=prepare,
        expected_error=ValueError,
        tags=("edge", "expected-error"),
    )


def portable_cases(
    tier: str, dtypes: set[str], triton_enabled: bool
) -> list[BenchmarkCase]:
    element_counts = {
        "quick": [4096, 262144],
        "standard": [4096, 1048576, 8388608],
        "thorough": [1, 4095, 4096, 4097, 1048576, 16777216],
    }[tier]
    gemm_shapes = {
        "quick": [(1, 256, 256), (127, 129, 257), (256, 256, 256)],
        "standard": [
            (1, 4096, 4096),
            (127, 129, 257),
            (256, 256, 256),
            (512, 512, 512),
        ],
        "thorough": [
            (1, 4096, 4096),
            (17, 4097, 1023),
            (127, 129, 257),
            (256, 256, 256),
            (512, 512, 512),
            (1024, 1024, 1024),
        ],
    }[tier]
    attention_shapes = {
        "quick": [(16, 64), (127, 80)],
        "standard": [(16, 64), (128, 128), (511, 128)],
        "thorough": [(1, 64), (16, 64), (127, 80), (512, 128), (1024, 128)],
    }[tier]
    cases: list[BenchmarkCase] = []
    for dtype_name in sorted(dtypes):
        for implementation in ("eager", "compiled"):
            for elements in element_counts:
                for kernel in ("launch", "copy", "triad", "atan", "fused_elementwise"):
                    cases.append(
                        make_elementwise_case(
                            kernel, elements, dtype_name, implementation
                        )
                    )
            if tier != "quick":
                cases.append(
                    make_elementwise_case(
                        "copy", 1048576, dtype_name, implementation, "strided"
                    )
                )
                cases.append(
                    make_elementwise_case(
                        "triad", 1048576, dtype_name, implementation, "misaligned"
                    )
                )
            for m, n, k in gemm_shapes:
                cases.append(make_gemm_case(m, n, k, dtype_name, implementation))
            for sequence, head_dim in attention_shapes:
                cases.append(
                    make_attention_case(
                        "qk", sequence, head_dim, dtype_name, implementation
                    )
                )
                cases.append(
                    make_attention_case(
                        "softmax", sequence, head_dim, dtype_name, implementation
                    )
                )
                cases.append(
                    make_attention_case(
                        "softmax",
                        sequence,
                        head_dim,
                        dtype_name,
                        implementation,
                        causal=True,
                    )
                )
                cases.append(
                    make_attention_case(
                        "pv", sequence, head_dim, dtype_name, implementation
                    )
                )
            for tokens, heads, head_size, rotary_dim in (
                (1, 1, 64, 64),
                (127, 8, 128, 64),
            ):
                for neox_style in (True, False):
                    cases.append(
                        make_rope_case(
                            tokens,
                            heads,
                            head_size,
                            rotary_dim,
                            dtype_name,
                            implementation,
                            neox_style,
                            False,
                        )
                    )
            for tokens, hidden_size in ((1, 768), (127, 4096)):
                cases.append(
                    make_swiglu_case(tokens, hidden_size, dtype_name, implementation)
                )
        if dtype_name in {"float16", "bfloat16"}:
            for tokens, hidden_size in ((1, 768), (127, 4096)):
                cases.append(
                    make_triton_swiglu_case(
                        tokens, hidden_size, dtype_name, triton_enabled
                    )
                )
            fused_shapes = [(1, 8, 2, 128, 128)]
            if tier != "quick":
                fused_shapes.append((17, 16, 4, 128, 64))
            for shape in fused_shapes:
                cases.append(make_fused_qk_case(*shape, dtype_name, triton_enabled))
        for convolution in (
            (1, 32, 31, 33, 64, 1, 1, False),
            (1, 32, 32, 32, 64, 3, 1, True),
            (1, 32, 35, 37, 32, 3, 32, False),
        ):
            *shape, channels_last = convolution
            cases.append(
                make_convolution_case(*shape, dtype_name, "eager", channels_last)
            )
            cases.append(
                make_convolution_case(*shape, dtype_name, "compiled", channels_last)
            )
    quantized_output_dtypes = sorted(dtypes & {"float16", "bfloat16"})
    for output_dtype_name in quantized_output_dtypes:
        scaled_shapes = [(1, 256, 128), (33, 257, 496)]
        if tier == "thorough":
            scaled_shapes.append((512, 1024, 1024))
        for index, shape in enumerate(scaled_shapes):
            cases.append(
                make_scaled_int8_case(
                    *shape,
                    output_dtype_name,
                    per_token_a=index > 0,
                    per_channel_b=index > 0,
                    with_bias=index % 2 == 1,
                    triton_enabled=triton_enabled,
                )
            )
        cases.append(
            make_scaled_fp8_case(32, 64, 64, output_dtype_name, triton_enabled)
        )
        cases.append(make_concurrent_gemm_case(256, output_dtype_name))
        cases.append(
            make_expected_error_case("rope_invalid_rotary_dim", output_dtype_name)
        )
        cases.append(make_expected_error_case("swiglu_odd_width", output_dtype_name))
    return cases


def device_metadata(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        raise UnsupportedCase("this suite currently requires a CUDA-like GPU")
    if not torch.cuda.is_available():
        raise UnsupportedCase(
            "PyTorch cannot access a CUDA-like GPU; install a CUDA/ROCm build and "
            "check device visibility"
        )
    properties = torch.cuda.get_device_properties(device)
    free_bytes, _ = torch.cuda.mem_get_info(device)
    return {
        "requested": str(device),
        "index": (
            device.index if device.index is not None else torch.cuda.current_device()
        ),
        "name": properties.name,
        "architecture": getattr(properties, "gcnArchName", None),
        "compute_capability": (
            f"{properties.major}.{properties.minor}"
            if hasattr(properties, "major")
            else None
        ),
        "total_memory_bytes": properties.total_memory,
        "free_memory_bytes_at_start": free_bytes,
        "multiprocessor_count": properties.multi_processor_count,
        "l2_cache_bytes": getattr(properties, "L2_cache_size", None),
        "pci_bus_id": getattr(properties, "pci_bus_id", None),
        "uuid": str(getattr(properties, "uuid", "")) or None,
    }


def timed_invocation_us(function: Callable[[], Any], device: torch.device) -> float:
    """Wall time of one synchronized invocation, in microseconds."""
    synchronize(device)
    start = time.perf_counter()
    function()
    synchronize(device)
    return (time.perf_counter() - start) * SECONDS_TO_US


def compile_callable(
    function: Callable[[], Any], device: torch.device
) -> tuple[Callable[[], Any], dict[str, Any]]:
    if not hasattr(torch, "compile"):
        raise UnsupportedCase("this PyTorch build does not provide torch.compile")
    # Every case built by the same make_*_case factory shares one code object,
    # so without a reset dynamo accumulates cache entries across cases and
    # eventually trips recompile_limit, after which later "compiled" cases run
    # eager while still being reported as compiled.
    torch._dynamo.reset()
    creation_start = time.perf_counter()
    compiled = torch.compile(function, fullgraph=False, dynamic=False)
    wrapper_creation_us = (time.perf_counter() - creation_start) * SECONDS_TO_US
    return compiled, {
        "kind": "torch_compile",
        "wrapper_creation_us": wrapper_creation_us,
        "first_invocation_us": timed_invocation_us(compiled, device),
    }


def merge_timing(
    metadata: dict[str, Any], statistics_fields: dict[str, Any]
) -> dict[str, Any]:
    """Flatten engine metadata and sample statistics into one timing dict.

    The two dicts share a namespace, so a silent overwrite would replace a
    measurement with metadata (or the reverse) and still emit valid JSON.
    """
    collisions = sorted(set(metadata) & set(statistics_fields))
    if collisions:
        raise AssertionError(f"timing key collision: {', '.join(collisions)}")
    return {**metadata, **statistics_fields}


def error_record(stage: str, exc: BaseException) -> dict[str, Any]:
    return {
        "stage": stage,
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": "".join(traceback.format_exception(exc))[-8000:],
    }


def run_case(
    case: BenchmarkCase,
    device: torch.device,
    warmups: int,
    samples: int,
    profile_iterations: int,
    profile_enabled: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "case_id": case.case_id,
        "category": case.category,
        "kernel": case.kernel,
        "implementation": case.implementation,
        "dtype": case.dtype,
        "shape": case.shape,
        "tags": list(case.tags),
        "status": "failed",
        "error": None,
    }
    stage = "prepare"
    try:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        allocated_before = torch.cuda.memory_allocated(device)
        reserved_before = torch.cuda.memory_reserved(device)
        prepared = case.prepare(device)
        result["features"] = prepared.features

        stage = "expected_error"
        if case.expected_error is not None:
            try:
                prepared.function()
            except case.expected_error as exc:
                result["status"] = "passed"
                result["expected_error"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
                return result
            raise AssertionError(
                f"expected {case.expected_error.__name__}, but no exception was raised"
            )

        function = prepared.function
        # Compilation must happen under the same grad mode the benchmark uses.
        # Dynamo guards on GLOBAL_STATE, so a graph traced outside
        # inference_mode is discarded and silently recompiled on the first
        # timed call, putting real compile cost into the untimed stages.
        if case.compile_function:
            stage = "compile"
            with torch.inference_mode():
                function, result["compilation"] = compile_callable(function, device)
        elif case.jit_function:
            stage = "compile"
            with torch.inference_mode():
                result["compilation"] = {
                    "kind": "triton_jit_first_invocation",
                    "first_invocation_us": timed_invocation_us(function, device),
                }

        stage = "correctness"
        with torch.inference_mode():
            actual = function()
            expected = prepared.reference()
        synchronize(device)
        result["correctness"] = compare_outputs(
            actual, expected, prepared.atol, prepared.rtol
        )
        if not result["correctness"]["passed"]:
            raise AssertionError(
                "kernel output did not match its independent reference"
            )

        stage = "benchmark"
        # Re-baseline here so the reported peak belongs to the benchmarked
        # callable. The correctness stage runs a float32 reference whose upcast
        # footprint otherwise dominates, and the profiler pass allocates too.
        torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode():
            benchmark_samples_us, benchmark_metadata = benchmark_torch_utils(
                function, warmups, samples
            )
            device_samples_us = (
                benchmark_cuda_events(function, device, warmups, samples)
                if device.type == "cuda"
                else None
            )
        result["timing"] = merge_timing(
            benchmark_metadata, sample_statistics(benchmark_samples_us)
        )
        if device_samples_us is not None:
            if _LAST_DEVICE_WINDOW:
                result["device_window"] = dict(_LAST_DEVICE_WINDOW)
                _LAST_DEVICE_WINDOW.clear()
            result["device_timing"] = {
                "engine": "torch.cuda.Event",
                **sample_statistics(device_samples_us),
            }
            # Host wall clock minus device elapsed time: the dispatch and
            # synchronization overhead folded into `timing`. Large relative to
            # median_us means the case is launch-bound, not kernel-bound.
            result["host_device_delta_median_us"] = (
                result["timing"]["median_us"] - result["device_timing"]["median_us"]
            )

        # Throughput stays on the host wall-clock median, which is an
        # end-to-end figure that includes dispatch. `latency_source` names it
        # explicitly, and `host_device_delta_median_us` is what tells a
        # consumer how much of it is overhead rather than kernel time.
        median_us = result["timing"]["median_us"]
        median_seconds = median_us / SECONDS_TO_US
        total_bytes = prepared.features.get("estimated_bytes_total")
        total_flops = prepared.features.get("estimated_flops")
        result["throughput"] = {
            "latency_source": "timing.median_us",
            "latency_us": median_us,
            "gb_per_second": (
                total_bytes / median_seconds / 1e9
                if total_bytes is not None and median_seconds > 0
                else None
            ),
            "tflops": (
                total_flops / median_seconds / 1e12
                if total_flops is not None and median_seconds > 0
                else None
            ),
        }

        result["memory"] = {
            "allocated_before_bytes": allocated_before,
            "reserved_before_bytes": reserved_before,
            "allocated_after_bytes": torch.cuda.memory_allocated(device),
            "reserved_after_bytes": torch.cuda.memory_reserved(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "peak_scope": "benchmark stage only, excludes reference and profiler",
        }

        if profile_enabled:
            stage = "profile"
            try:
                with torch.inference_mode():
                    result["profiler"] = profile_function(
                        function, device, profile_iterations
                    )
            except Exception as exc:
                result["profiler"] = {"error": error_record(stage, exc)}

        result["status"] = "passed"
    except UnsupportedCase as exc:
        result["status"] = "skipped"
        result["error"] = error_record(stage, exc)
    except torch.cuda.OutOfMemoryError as exc:
        result["status"] = "skipped"
        result["error"] = error_record(stage, exc)
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = error_record(stage, exc)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Characterize a CUDA/ROCm GPU and write a timing report."
    )
    parser.add_argument(
        "--device", default="cuda", help="PyTorch device (default: cuda)"
    )
    parser.add_argument(
        "--output", default="report.json", help="JSON path, or - for stdout"
    )
    parser.add_argument(
        "--tier",
        choices=("quick", "standard", "thorough"),
        default="standard",
        help="Case set and default sample counts (default: standard)",
    )
    parser.add_argument("--samples", type=int, help="Override timed samples per case")
    parser.add_argument("--warmups", type=int, help="Override warmup iterations")
    parser.add_argument("--profile-iterations", type=int, help="Profiler iterations")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument("--kernel", action="append", help="Kernel substring filter")
    parser.add_argument(
        "--dtype",
        action="append",
        choices=tuple(DTYPES),
        help="Dtype to include; repeat for multiple dtypes",
    )
    parser.add_argument(
        "--no-compile", action="store_true", help="Exclude torch.compile cases"
    )
    parser.add_argument(
        "--no-triton",
        action="store_true",
        help="Report Triton cases as skipped instead of running them",
    )
    parser.add_argument(
        "--no-profile", action="store_true", help="Disable torch.profiler passes"
    )
    parser.add_argument(
        "--fail-fast", action="store_true", help="Stop after the first failed case"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress informational progress logs"
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    for name in ("samples", "warmups", "profile_iterations"):
        value = getattr(args, name)
        if value is not None and value < (0 if name == "warmups" else 1):
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")


def configure_logging(quiet: bool) -> None:
    logging.basicConfig(
        level=logging.WARNING if quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stderr,
    )


def software_metadata() -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "hip_version": torch.version.hip,
        "triton_available": HAS_TRITON,
        "triton_version": getattr(triton, "__version__", None),
        "triton_import_error": TRITON_IMPORT_ERROR,
        "platform": platform.platform(),
        "hostname": platform.node(),
    }


def select_cases(args: argparse.Namespace) -> list[BenchmarkCase]:
    dtype_names = set(args.dtype or DTYPES)
    cases = rms_norm_cases(args.tier, dtype_names)
    cases.extend(
        portable_cases(args.tier, dtype_names, triton_enabled=not args.no_triton)
    )
    if args.no_compile:
        cases = [case for case in cases if not case.compile_function]
    if args.kernel:
        filters = tuple(args.kernel)
        cases = [
            case
            for case in cases
            if any(
                filter_text in case.kernel or filter_text in case.case_id
                for filter_text in filters
            )
        ]
    return cases


def write_report(report: dict[str, Any], output: str) -> None:
    report = strict_json_value(report)
    if output == "-":
        json.dump(report, sys.stdout, indent=2, sort_keys=True, allow_nan=False)
        sys.stdout.write("\n")
        return
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def overlap_latency_us(result: dict[str, Any]) -> float | None:
    timing = result.get("timing")
    if timing is None:
        return None
    return timing["median_us"]


def add_derived_metrics(results: list[dict[str, Any]]) -> None:
    isolated_gemms: dict[tuple[str, int], dict[str, Any]] = {}
    for result in results:
        shape = result.get("shape") or {}
        if (
            result["status"] == "passed"
            and result["kernel"] == "gemm"
            and result["implementation"] == "eager"
            and {"m", "n", "k"} <= shape.keys()
            and shape["m"] == shape["n"] == shape["k"]
        ):
            isolated_gemms[(result["dtype"], shape["m"])] = result
    for result in results:
        if result["status"] != "passed" or result["kernel"] != "concurrent_gemm":
            continue
        size = (result.get("shape") or {}).get("size")
        baseline = None if size is None else isolated_gemms.get((result["dtype"], size))
        if baseline is None:
            result["concurrency_metrics"] = {
                "available": False,
                "reason": "matching isolated GEMM baseline was not selected",
            }
            continue
        isolated_us = overlap_latency_us(baseline)
        concurrent_us = overlap_latency_us(result)
        if isolated_us is None or concurrent_us is None:
            result["concurrency_metrics"] = {
                "available": False,
                "reason": "a timing distribution is missing for this pair",
            }
            continue
        sequential_us = 2.0 * isolated_us
        overlap_savings_us = sequential_us - concurrent_us
        overlap_efficiency = (
            overlap_savings_us / isolated_us if isolated_us > 0 else None
        )
        result["concurrency_metrics"] = {
            "available": True,
            "baseline_case_id": baseline["case_id"],
            "isolated_single_workload_median_us": isolated_us,
            "estimated_sequential_median_us": sequential_us,
            "concurrent_median_us": concurrent_us,
            "speedup_vs_sequential": (
                sequential_us / concurrent_us if concurrent_us > 0 else None
            ),
            "overlap_savings_us": overlap_savings_us,
            "overlap_efficiency": overlap_efficiency,
            "latency_source": "timing.median_us",
            "interpretation": "0=no overlap, 1=ideal full overlap; values may exceed bounds under noise",
            "caveat": (
                "derived from host wall-clock medians: the sequential estimate "
                "doubles the per-measurement dispatch overhead that the "
                "concurrent case pays once, and the concurrent callable adds "
                "four stream waits. Small sizes can report negative overlap "
                "efficiency for this reason alone; compare "
                "host_device_delta_median_us on both cases before trusting it"
            ),
        }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    configure_logging(args.quiet)
    defaults = {
        "quick": {"samples": 5, "warmups": 2, "profile_iterations": 1},
        "standard": {"samples": 12, "warmups": 5, "profile_iterations": 3},
        "thorough": {"samples": 30, "warmups": 10, "profile_iterations": 5},
    }[args.tier]
    samples = args.samples or defaults["samples"]
    warmups = args.warmups if args.warmups is not None else defaults["warmups"]
    profile_iterations = args.profile_iterations or defaults["profile_iterations"]
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and torch.cuda.is_available():
        # torch.utils.benchmark's clock synchronizes the *current* accelerator,
        # not the device passed to the kernels. Without this, --device cuda:1
        # would be timed against an unsynchronized cuda:0 and every sample
        # would measure launch latency instead of execution.
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
        _open_device_epoch()
    wall_start = time.perf_counter()

    try:
        device_info = device_metadata(device)
    except Exception as exc:
        raise SystemExit(str(exc)) from exc

    cases = select_cases(args)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "started_at": utc_now(),
        "completed_at": None,
        "elapsed_seconds": None,
        "config": {
            "device": args.device,
            "tier": args.tier,
            "samples": samples,
            "warmups": warmups,
            "profile_iterations": profile_iterations,
            "profile_enabled": not args.no_profile,
            "compile_enabled": not args.no_compile,
            "triton_enabled": HAS_TRITON and not args.no_triton,
            "seed": args.seed,
            "kernel_filters": args.kernel or [],
            "dtypes": sorted(set(args.dtype or DTYPES)),
        },
        "methodology": {
            "benchmark_engine": "torch.utils.benchmark.Timer",
            "device_timing_cross_check": "torch.cuda.Event",
            # Scoped deliberately: every field whose name ends in `_us` is
            # microseconds. `elapsed_seconds` and the throughput rates are not.
            "timing_field_unit": "microseconds for every *_us field",
            "benchmark_clock": "host wall clock, includes dispatch and sync",
            "device_timing_clock": "device elapsed time, excludes host overhead",
            "warmups_excluded": True,
            "compilation_excluded_from_steady_state": True,
            "profiler_separate_from_benchmark": True,
            "error_bars": {
                "stdev": "sample standard deviation",
                "standard_error": "stdev / sqrt(sample_count)",
                "ci95": "mean +/- 1.96 * standard_error",
            },
            "throughput_statistic": (
                "median of device_timing where available, else timing; the "
                "source is named per case in throughput.latency_source"
            ),
        },
        "software": software_metadata(),
        "device": device_info,
        "results": [],
        "summary": {},
    }
    for index, case in enumerate(cases, start=1):
        LOGGER.info("[%d/%d] %s", index, len(cases), case.case_id)
        case_result = run_case(
            case,
            device,
            warmups,
            samples,
            profile_iterations,
            not args.no_profile,
        )
        report["results"].append(case_result)
        if args.fail_fast and case_result["status"] == "failed":
            break

    add_derived_metrics(report["results"])

    counts = {
        status: sum(result["status"] == status for result in report["results"])
        for status in ("passed", "failed", "skipped")
    }
    categories = sorted({result["category"] for result in report["results"]})
    by_category = {
        category: {
            status: sum(
                result["category"] == category and result["status"] == status
                for result in report["results"]
            )
            for status in ("passed", "failed", "skipped")
        }
        for category in categories
    }
    report["summary"] = {
        "selected_cases": len(cases),
        "attempted_cases": len(report["results"]),
        **counts,
        "by_category": by_category,
    }
    report["status"] = (
        "completed" if counts["failed"] == 0 else "completed_with_failures"
    )
    report["completed_at"] = utc_now()
    report["elapsed_seconds"] = time.perf_counter() - wall_start
    write_report(report, args.output)
    LOGGER.info(
        "Wrote %s: %d passed, %d failed, %d skipped",
        args.output,
        counts["passed"],
        counts["failed"],
        counts["skipped"],
    )
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
