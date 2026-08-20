#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Phase A validation: run the evaluation pipeline with mock or real metrics."""

import argparse
from pathlib import Path
from typing import Optional

from membw.engine import evaluate_membw_tree
from membw.tree_spec import load_tree_spec
from membw.workload import load_membw_metrics

UTCL1_HBM_WORKLOAD: dict[str, Optional[float]] = {
    "L1 Cache - TA stalled by TCP (aggregated)": 22.4,
    "L1 Cache - TCP stalled by UTCL1": 18.7,
    "L1 Cache - TCP stalled by UTCL2": 1.2,
    "L1 Cache - TCP stalled by TD": 0.5,
    "L1 Cache - TCP stalled by L2": 2.1,
    "L1 Cache - VMEM stalled by L1 Cache": 3.0,
    "L2 Back Pressure Indicator": 5.0,
    "L2 Memory BW Bound - Combined Credit Pressure": 15.0,
    "L2 Memory BW Bound - Read Credit Pressure": 12.0,
    "L2 Memory BW Bound - Write Credit Pressure": 3.0,
    "L2 Internal Resource Pressure - Latency FIFO": 2.0,
    "L2 Internal Resource Pressure - Source FIFO": 1.0,
    "L2 Cache Efficiency": 80.0,
    "L2 Remote Access Pressure (GMI)": 0.5,
    "EA HBM BW Bound - Combined": 14.0,
    "EA HBM BW Bound - Read Credit Pressure": 11.0,
    "EA HBM BW Bound - Write Credit Pressure": 3.0,
    "EA GMI BW Bound - Combined": 0.1,
    "EA IO BW Bound - Combined": 0.0,
    "EA Write Backpressure": 1.0,
    "EA HBM Atomic Pressure": 0.5,
}


def main() -> None:
    args = _parse_args()
    arch = args.arch

    spec = load_tree_spec(arch)

    if args.workload_dir is not None:
        loaded = load_membw_metrics(args.workload_dir, arch)
        metric_values = loaded.values
        metric_units = loaded.units
        availability = loaded.availability
        availability_reason = loaded.availability_reason
    else:
        metric_values = UTCL1_HBM_WORKLOAD
        metric_units = None
        availability = "full"
        availability_reason = None

    result = evaluate_membw_tree(
        spec,
        metric_values,
        arch,
        availability,
        availability_reason,
        metric_units=metric_units,
    )
    print()
    for block in result.guidance_blocks:
        print(block)
        print()


# --- Private helpers ---


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run membw bottleneck tree evaluation.",
    )
    parser.add_argument(
        "--workload-dir",
        type=Path,
        default=None,
        help="Path to a profiled workload directory.",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="gfx950",
        help="GPU architecture (default: gfx950).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
