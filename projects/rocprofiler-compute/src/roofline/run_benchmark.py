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


def run_roofline_benchmark(
    device_id: int, roofline_csv: Path, cache_sizes: dict
) -> None:
    """Load device benchmark, execute, and save results to CSV."""
    bench = load_bench(device_id, cache_sizes)
    benchmark_metrics = bench.run_benchmark(device_id)
    bench.dump_csv(benchmark_metrics, str(roofline_csv))


def load_bench(device_id: int, cache_sizes: dict) -> object:
    try:
        from utils.hip_interface import hipGetDeviceProperties

        # Get exact LLVM target name of the device
        gfx_device = (hipGetDeviceProperties(device_id).gcnArchName).split(":", 1)[0]

        # Force gfx940 MI300A_A0 and gfx941 MI300X_A0 products
        # to use same class as gfx942 MI300_A1
        if gfx_device == "gfx940" or gfx_device == "gfx941":
            gfx_device = "gfx942"

        # Get the gfx architecture of the device
        gfx_arch = gfx_device[:-2]

        # Dynamically import the bench class module
        bench_module = importlib.import_module(
            f"roofline.benchmark.{gfx_arch}.benchmark_{gfx_device}"
        )

        # Get the bench class from the module
        bench_class = getattr(bench_module, f"Bench_{gfx_device}")
        # Instantiate and return the bench class
        bench_instance = bench_class(device_id, cache_sizes)
        return bench_instance
    except Exception as e:
        # Propagate error so users do not attempt to use a non-existent bench instance
        raise RuntimeError(
            f"Failed to load benchmark for devices {device_id}: {e}"
        ) from e
