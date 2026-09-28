#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT
"""Evaluate experimental single-pass-packable allocate vs shipping heuristic.

Also reports SLOT_LIMIT fill-into-existing-passes impact (additional passes).

Example::

    PYTHONPATH=src:tools python3 tools/eval_single_pass_packable.py --arch gfx942
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT / "src", _ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from counter_grouping_inspector import (  # noqa: E402
    _build_inspector_soc,
    _rocprof_supported_superset,
    get_default_config_dir,
)
from rocprof_compute_soc.counter_grouping_refill import (  # noqa: E402
    count_multi_bucket_metrics,
)
from rocprof_compute_soc.counter_grouping_single_pass import (  # noqa: E402
    _count_packable_multi,
    collect_unique_packable_unions,
    collect_unique_slot_limit_unions,
    fill_slot_limit_into_existing_passes,
)

from utils.mi_gpu_spec import mi_gpu_specs  # noqa: E402


def _run_shipping(arch: str) -> dict[str, int | float]:
    config_dir = get_default_config_dir()
    perfmon_config = mi_gpu_specs.get_perfmon_config(arch)
    env_keys = (
        "ROCPROF_COMPUTE_PERFMON_SINGLE_PASS_PACKABLE",
        "ROCPROF_COMPUTE_PERFMON_CP_SAT",
        "ROCPROF_COMPUTE_COALESCE_REFILL",
    )
    backup = {k: os.environ.get(k) for k in env_keys}
    try:
        os.environ.pop("ROCPROF_COMPUTE_PERFMON_CP_SAT", None)
        os.environ.pop("ROCPROF_COMPUTE_PERFMON_SINGLE_PASS_PACKABLE", None)
        os.environ["ROCPROF_COMPUTE_COALESCE_REFILL"] = "1"
        with tempfile.TemporaryDirectory(prefix="eval_spp_") as tmp:
            soc = _build_inspector_soc(
                arch, config_dir, None, perfmon_config, Path(tmp)
            )
            counters, _ = soc.detect_counters()
            counters -= {"SQ_ACCUM_PREV_HIRES"}
            soc.get_rocprof_supported_counters = (  # type: ignore[method-assign]
                lambda c=counters: _rocprof_supported_superset(c)
            )
            t0 = time.perf_counter()
            files, _fc, _acc = soc._allocate_perfmon_counter_files(
                set(counters), apply_refill=True
            )
            elapsed = time.perf_counter() - t0
            unions, packable_n = collect_unique_packable_unions(
                soc, counters, perfmon_config
            )
            multi = count_multi_bucket_metrics(files, soc, counters)
            packable_multi = _count_packable_multi(files, unions)
    finally:
        for key, val in backup.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val
    return {
        "passes": len(files),
        "pmc_total": len(counters),
        "multi_metrics": multi,
        "packable_multi": packable_multi,
        "packable_metrics": packable_n,
        "unique_packable_unions": len(unions),
        "seconds": round(elapsed, 3),
    }


def _run_experimental_with_slot_fill(arch: str) -> dict[str, int | float]:
    config_dir = get_default_config_dir()
    perfmon_config = mi_gpu_specs.get_perfmon_config(arch)
    env_keys = (
        "ROCPROF_COMPUTE_PERFMON_SINGLE_PASS_PACKABLE",
        "ROCPROF_COMPUTE_PERFMON_CP_SAT",
        "ROCPROF_COMPUTE_COALESCE_REFILL",
    )
    backup = {k: os.environ.get(k) for k in env_keys}
    try:
        os.environ.pop("ROCPROF_COMPUTE_PERFMON_CP_SAT", None)
        os.environ["ROCPROF_COMPUTE_PERFMON_SINGLE_PASS_PACKABLE"] = "1"
        os.environ["ROCPROF_COMPUTE_COALESCE_REFILL"] = "0"
        with tempfile.TemporaryDirectory(prefix="eval_spp_") as tmp:
            soc = _build_inspector_soc(
                arch, config_dir, None, perfmon_config, Path(tmp)
            )
            counters, _ = soc.detect_counters()
            counters -= {"SQ_ACCUM_PREV_HIRES"}
            soc.get_rocprof_supported_counters = (  # type: ignore[method-assign]
                lambda c=counters: _rocprof_supported_superset(c)
            )
            t0 = time.perf_counter()
            files, _fc, _acc = soc._allocate_perfmon_counter_files(
                set(counters), apply_refill=False
            )
            elapsed = time.perf_counter() - t0
            unions, packable_n = collect_unique_packable_unions(
                soc, counters, perfmon_config
            )
            slot_unions, slot_n = collect_unique_slot_limit_unions(
                soc, counters, perfmon_config
            )
            packable_multi = _count_packable_multi(files, unions)
            passes_after_packable = len(files)

            _files2, _fc2, slot_stats = fill_slot_limit_into_existing_passes(
                files,
                slot_unions,
                perfmon_config,
                slot_limit_metric_count=slot_n,
            )
    finally:
        for key, val in backup.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val

    return {
        "passes_after_packable": passes_after_packable,
        "passes_after_slot_fill": slot_stats.passes_after,
        "additional_passes_for_slot_limit": slot_stats.additional_passes,
        "pmc_total": len(counters),
        "packable_multi": packable_multi,
        "packable_metrics": packable_n,
        "unique_packable_unions": len(unions),
        "slot_limit_metrics": slot_stats.slot_limit_metrics,
        "unique_slot_limit_unions": slot_stats.unique_slot_limit_unions,
        "slot_pmc_already_in_layout": slot_stats.pmc_already_covered,
        "slot_pmc_into_existing": slot_stats.pmc_placed_into_existing,
        "slot_pmc_into_new_buckets": slot_stats.pmc_placed_into_new,
        "seconds": round(elapsed, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="gfx942")
    args = parser.parse_args()

    baseline = _run_shipping(args.arch)
    experimental = _run_experimental_with_slot_fill(args.arch)

    print(f"arch={args.arch}")
    print()
    print("Shipping (heuristic + priority + refill)")
    for key, val in baseline.items():
        print(f"  {key}: {val}")
    print()
    print("Experimental (SINGLE_PASS_PACKABLE=1)")
    print(f"  passes after packable only: {experimental['passes_after_packable']}")
    print(
        f"  packable_multi (unions lacking full bucket): "
        f"{experimental['packable_multi']}"
    )
    print(f"  packable_metrics: {experimental['packable_metrics']}")
    print(f"  unique_packable_unions: {experimental['unique_packable_unions']}")
    print()
    print("SLOT_LIMIT fill into existing passes")
    print(f"  slot_limit_metrics: {experimental['slot_limit_metrics']}")
    print(f"  unique_slot_limit_unions: {experimental['unique_slot_limit_unions']}")
    print(
        f"  SLOT_LIMIT PMCs already in packable layout: "
        f"{experimental['slot_pmc_already_in_layout']}"
    )
    print(
        f"  SLOT_LIMIT PMCs placed into existing buckets: "
        f"{experimental['slot_pmc_into_existing']}"
    )
    print(
        f"  SLOT_LIMIT PMCs placed into new buckets: "
        f"{experimental['slot_pmc_into_new_buckets']}"
    )
    print()
    print(
        f"  total passes after SLOT_LIMIT fill: "
        f"{experimental['passes_after_slot_fill']}"
    )
    print(
        f"  additional passes for SLOT_LIMIT: "
        f"+{experimental['additional_passes_for_slot_limit']}"
    )
    print(
        f"  (packable {experimental['passes_after_packable']} → "
        f"{experimental['passes_after_slot_fill']} total)"
    )
    print()
    print(
        "vs shipping: "
        f"{experimental['passes_after_slot_fill'] - baseline['passes']:+d} "
        f"({baseline['passes']} → {experimental['passes_after_slot_fill']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
