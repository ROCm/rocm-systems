#!/usr/bin/env python3
"""Run one verified Gluon shared-memory kernel and emit benchmark measurements."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import sys
import time

RESULT_MARKER = "CONSAN_BENCHMARK_RESULT="


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aorta-dir", type=Path, required=True)
    parser.add_argument("--config-json", required=True)
    return parser.parse_args(argv)


def _instrumentation_control():
    if "RJ_CONSAN_MODE" not in os.environ:
        return (lambda: 0), (lambda: None), (lambda: None)
    hook_path = os.environ.get("HSA_TOOLS_LIB")
    if not hook_path:
        raise RuntimeError("RJ_CONSAN_MODE is set without HSA_TOOLS_LIB")
    hook = ctypes.CDLL(hook_path)
    query = hook.rj_dbi_consan_instrumentation_nanoseconds
    query.argtypes = ()
    query.restype = ctypes.c_uint64
    begin_window = hook.rj_dbi_consan_begin_epoch_analysis_window
    begin_window.argtypes = ()
    begin_window.restype = ctypes.c_uint32
    end_window = hook.rj_dbi_consan_end_epoch_analysis_window
    end_window.argtypes = ()
    end_window.restype = ctypes.c_uint32

    def set_analysis_window(open_window: bool) -> None:
        if os.environ.get("RJ_CONSAN_EPOCH_ANALYSIS") != "manual":
            return
        status = begin_window() if open_window else end_window()
        if status != 0:
            raise RuntimeError(
                f"could not {'open' if open_window else 'close'} the ConSan "
                f"epoch-analysis window: status {status}"
            )

    return query, lambda: set_analysis_window(True), lambda: set_analysis_window(False)


def _main(argv: list[str]) -> int:
    args = _parse_args(argv)
    if not (args.aorta_dir.resolve() / "src" / "aorta").is_dir():
        raise SystemExit(f"Aorta checkout not found under {args.aorta_dir.resolve()}")
    try:
        config = json.loads(args.config_json)
        size = int(config["size"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"invalid Gluon benchmark config: {error}") from error
    if size < 32 or size > 1024 or size & (size - 1):
        raise SystemExit(
            f"Gluon shared-memory size must be a power of two in [32, 1024], got {size}"
        )

    import torch
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl

    @gluon.jit
    def shared_roundtrip_kernel(
        source,
        destination,
        SIZE: gl.constexpr,
        layout: gl.constexpr,
        shared_layout: gl.constexpr,
    ):
        offsets = gl.arange(0, SIZE, layout=layout)
        values = gl.load(source + offsets)
        shared = gl.allocate_shared_memory(values.dtype, [SIZE], shared_layout)
        shared.store(values)
        values = shared.load(layout)
        gl.store(destination + offsets, values * 1.25 + 0.5)

    instrumentation_nanoseconds, begin_analysis, end_analysis = _instrumentation_control()
    instrumentation_begin = instrumentation_nanoseconds()
    setup_start = time.perf_counter()
    if not torch.cuda.is_available():
        raise SystemExit("the Gluon benchmark requires a visible ROCm GPU")
    torch.manual_seed(1234)
    host_source = torch.randn((size,), dtype=torch.float32)
    expected = host_source * 1.25 + 0.5
    source = host_source.cuda()
    destination = torch.empty_like(source)
    warp_size = torch.cuda.get_device_properties(0).warp_size
    layout = gl.BlockedLayout([1], [warp_size], [1], [0])
    shared_layout = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0])
    torch.cuda.synchronize()
    setup_ms = (time.perf_counter() - setup_start) * 1000.0
    instrumentation_before_run = instrumentation_nanoseconds()
    torch.cuda.reset_peak_memory_stats()

    runs = []
    for run_index in range(2):
        if run_index == 0:
            begin_analysis()
        instrumentation_before = instrumentation_nanoseconds()
        run_start = time.perf_counter()
        try:
            shared_roundtrip_kernel[(1,)](
                source,
                destination,
                SIZE=size,
                layout=layout,
                shared_layout=shared_layout,
                num_warps=1,
            )
            torch.cuda.synchronize()
        finally:
            if run_index == 0:
                end_analysis()
        run_ms = (time.perf_counter() - run_start) * 1000.0
        instrumentation_after = instrumentation_nanoseconds()
        actual = destination.cpu()
        max_abs_error = float((actual - expected).abs().max())
        passed = torch.allclose(actual, expected, rtol=2.0e-4, atol=2.0e-4)
        runs.append(
            {
                "index": run_index + 1,
                "result": {
                    "passed": bool(passed),
                    "metrics": {
                        "latency_ms": run_ms,
                        "element_count": size,
                        "warp_size": warp_size,
                        "max_abs_error": max_abs_error,
                    },
                },
                "phase_ms": run_ms,
                "instrumentation_ms": (instrumentation_after - instrumentation_before)
                / 1_000_000.0,
            }
        )
    instrumentation_after_run = instrumentation_nanoseconds()
    passed = all(run["result"]["passed"] for run in runs)
    instrumentation_during_run_ms = sum(run["instrumentation_ms"] for run in runs)
    payload = {
        "result": {"passed": passed, "metrics": runs[0]["result"]["metrics"]},
        "runs": runs,
        "phase_ms": {"setup": setup_ms, "run": runs[0]["phase_ms"]},
        "instrumentation_ms": {
            "before_run": (instrumentation_before_run - instrumentation_begin)
            / 1_000_000.0,
            "during_run": instrumentation_during_run_ms,
            "total": (instrumentation_after_run - instrumentation_begin) / 1_000_000.0,
        },
        "peak_device_memory": {
            "allocated_bytes": torch.cuda.max_memory_allocated(),
            "reserved_bytes": torch.cuda.max_memory_reserved(),
        },
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "triton": getattr(__import__("triton"), "__version__", "unknown"),
            "device_name": torch.cuda.get_device_name(),
            "architecture": torch.cuda.get_device_properties(0).gcnArchName,
        },
    }
    print(RESULT_MARKER + json.dumps(payload, sort_keys=True), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
