#!/usr/bin/env python3
"""Run one bounded Aorta inference workload and emit machine-readable results."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time


RESULT_MARKER = "CONSAN_BENCHMARK_RESULT="


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aorta-dir", type=Path, required=True)
    parser.add_argument("--config-json", required=True)
    parser.add_argument(
        "--profile-kernels",
        action="store_true",
        help="inventory dispatched GPU kernel names in a native discovery run",
    )
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

    if not torch.cuda.is_available():
        raise SystemExit("the Aorta benchmark requires a visible ROCm GPU")
    torch.cuda.reset_peak_memory_stats()

    workload = InferenceWorkload(config)
    setup_start = time.perf_counter()
    workload.setup()
    torch.cuda.synchronize()
    setup_ms = (time.perf_counter() - setup_start) * 1000.0
    try:
        run_start = time.perf_counter()
        profiler = None
        if args.profile_kernels:
            profiler = torch.profiler.profile(
                activities=(
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                )
            )
            profiler.__enter__()
        try:
            result = workload.run()
        finally:
            if profiler is not None:
                profiler.__exit__(None, None, None)
        torch.cuda.synchronize()
        run_ms = (time.perf_counter() - run_start) * 1000.0
        kernel_names = []
        kernel_stats = []
        if profiler is not None:
            device_events = [
                event
                for event in profiler.events()
                if event.device_type != torch.autograd.DeviceType.CPU
            ]
            kernel_names = sorted({event.name for event in device_events})
            kernel_stats = [
                {
                    "name": name,
                    "dispatches": sum(event.count for event in device_events if event.name == name),
                    "device_time_us": sum(
                        event.device_time_total for event in device_events if event.name == name
                    ),
                }
                for name in kernel_names
            ]
        payload = {
            "result": asdict(result),
            "phase_ms": {"setup": setup_ms, "run": run_ms},
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
            "kernel_names": kernel_names,
            "kernel_stats": kernel_stats,
        }
        print(RESULT_MARKER + json.dumps(payload, sort_keys=True), flush=True)
        return 0 if result.passed else 2
    finally:
        workload.cleanup()


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
