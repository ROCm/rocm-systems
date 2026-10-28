# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT


# -----------------------------------------------------------------------------
# run_benchmark.py
#
# Run empirical benchmarking on the device.
#
# Roofline benchmarking is called from within rocprofiler-compute profiling run.
# To run standalone roofline benchmarking without counter collection, use:
#   `rocprof-compute profile --bench-only -n test_bench`
#       -> runs on current device
#   `rocprof-compute profile --bench-only --device 2 -n test_bench`
#       -> runs on device 2 using the `--device` option
#
# Note: there is an expectation that if more than one device is requested
# for benchmarking that the devices are the same product.
# -----------------------------------------------------------------------------

import importlib
from pathlib import Path

from utils.mi_gpu_spec import mi_gpu_specs

BENCHMARKING_SUPPORTED = mi_gpu_specs.get_roofline_archs()


def run_roofline_benchmark(
    device_id: int, roofline_csv: Path, cache_sizes: dict
) -> None:
    """Load device benchmark, execute, and save results to CSV."""
    bench = load_bench(device_id, cache_sizes)
    benchmark_metrics = bench.run_benchmark(device_id)
    bench.dump_csv(benchmark_metrics, str(roofline_csv))


def bench_class(gpu_arch: str) -> type:
    """Return the benchmark class of gpu_arch, e.g. Bench_gfx942 for "gfx940"."""
    device = mi_gpu_specs.get_roofline_benchmark(gpu_arch)
    family = mi_gpu_specs.get_roofline_family(gpu_arch)
    module = importlib.import_module(f"roofline.benchmark.{family}.benchmark_{device}")
    return getattr(module, f"Bench_{device}")


def load_bench(device_id: int, cache_sizes: dict) -> object:
    try:
        from utils.hip_interface import hipGetDeviceProperties

        # The device's exact LLVM target, e.g. "gfx942" from "gfx942:sramecc+:xnack-"
        gpu_arch = hipGetDeviceProperties(device_id).gcnArchName.split(":", 1)[0]
        return bench_class(gpu_arch)(device_id, cache_sizes)
    except Exception as e:
        # Propagate error so users do not attempt to use a non-existent bench instance
        raise RuntimeError(
            f"Failed to load benchmark for devices {device_id}: {e}"
        ) from e
