#!/usr/bin/env python3
"""Run one bounded Aorta inference workload and emit machine-readable results."""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

RESULT_MARKER = "CONSAN_BENCHMARK_RESULT="


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
        if os.environ.get("RJ_CONSAN_MOI_EPOCH_ANALYSIS") != "manual":
            return
        status = begin_window() if open_window else end_window()
        if status != 0:
            raise RuntimeError(
                f"could not {'open' if open_window else 'close'} the ConSan "
                f"epoch-analysis window: status {status}"
            )

    return query, lambda: set_analysis_window(True), lambda: set_analysis_window(False)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aorta-dir", type=Path, required=True)
    parser.add_argument("--config-json", required=True)
    return parser.parse_args(argv)


def _main(argv: list[str]) -> int:
    args = _parse_args(argv)
    source_dir = args.aorta_dir.resolve() / "src"
    if not (source_dir / "aorta" / "workloads" / "inference.py").is_file():
        raise SystemExit(
            f"Aorta inference workload not found under {args.aorta_dir.resolve()}"
        )
    try:
        config = json.loads(args.config_json)
    except json.JSONDecodeError as error:
        raise SystemExit(f"invalid --config-json: {error}") from error
    if not isinstance(config, dict):
        raise SystemExit("--config-json must contain a JSON object")

    sys.path.insert(0, str(source_dir))
    import torch
    from aorta.workloads.inference import InferenceWorkload

    instrumentation_nanoseconds, begin_analysis, end_analysis = _instrumentation_control()
    instrumentation_begin = instrumentation_nanoseconds()
    if not torch.cuda.is_available():
        raise SystemExit("the Aorta benchmark requires a visible ROCm GPU")
    torch.cuda.reset_peak_memory_stats()

    workload = InferenceWorkload(config)
    setup_start = time.perf_counter()
    workload.setup()
    torch.cuda.synchronize()
    setup_ms = (time.perf_counter() - setup_start) * 1000.0
    input_generator = getattr(workload, "_input_gen", None)
    if not isinstance(input_generator, torch.Generator):
        raise RuntimeError(
            "Aorta inference workload exposes no resettable input generator"
        )
    input_state = input_generator.get_state()
    instrumentation_before_run = instrumentation_nanoseconds()
    try:
        results = []
        runs = []
        for run_index in range(2):
            if run_index == 0:
                begin_analysis()
            input_generator.set_state(input_state)
            instrumentation_before = instrumentation_nanoseconds()
            run_start = time.perf_counter()
            try:
                result = workload.run()
                torch.cuda.synchronize()
            finally:
                if run_index == 0:
                    end_analysis()
            run_ms = (time.perf_counter() - run_start) * 1000.0
            instrumentation_after = instrumentation_nanoseconds()
            result_payload = asdict(result)
            results.append(result_payload)
            runs.append(
                {
                    "index": run_index + 1,
                    "result": result_payload,
                    "phase_ms": run_ms,
                    "instrumentation_ms": (
                        instrumentation_after - instrumentation_before
                    )
                    / 1_000_000.0,
                }
            )
        instrumentation_after_run = instrumentation_nanoseconds()
        combined_result = dict(results[0])
        checksums = tuple(
            result.get("metrics", {}).get("logits_checksum") for result in results
        )
        equivalent = checksums[0] is None or checksums[0] == checksums[1]
        combined_result["passed"] = (
            all(result["passed"] for result in results) and equivalent
        )
        combined_result["failure_count"] = sum(
            int(result["failure_count"]) for result in results
        ) + int(not equivalent)
        if not equivalent:
            combined_result.setdefault("failure_details", []).append(
                {"runs": [1, 2], "problems": ["logits-checksum-mismatch"]}
            )
        payload = {
            "result": combined_result,
            "runs": runs,
            "phase_ms": {"setup": setup_ms, "run": runs[0]["phase_ms"]},
            "instrumentation_ms": {
                "before_run": (instrumentation_before_run - instrumentation_begin)
                / 1_000_000.0,
                "during_run": sum(run["instrumentation_ms"] for run in runs),
                "total": (instrumentation_after_run - instrumentation_begin)
                / 1_000_000.0,
            },
            "peak_device_memory": {
                "allocated_bytes": torch.cuda.max_memory_allocated(),
                "reserved_bytes": torch.cuda.max_memory_reserved(),
            },
            "runtime": {
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "hip": torch.version.hip,
                "device_name": torch.cuda.get_device_name(),
                "architecture": torch.cuda.get_device_properties(0).gcnArchName,
            },
        }
        print(RESULT_MARKER + json.dumps(payload, sort_keys=True), flush=True)
        return 0 if combined_result["passed"] else 2
    finally:
        workload.cleanup()


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
