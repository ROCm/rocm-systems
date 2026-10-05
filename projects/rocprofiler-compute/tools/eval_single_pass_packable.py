#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT
"""Compare default single-pass-packable allocate vs legacy heuristic.

Example::

    PYTHONPATH=src:tools python3 tools/eval_single_pass_packable.py --arch gfx942
"""

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional, Set, Tuple, Union

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT / "src", _ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from counter_grouping_inspector import (  # noqa: E402
    _build_inspector_soc,
    _rocprof_supported_superset,
    get_default_config_dir,
)

from rocprof_compute_soc.counter_grouping_buckets import (  # noqa: E402
    count_multi_bucket_metrics,
)
from rocprof_compute_soc.counter_grouping_single_pass import (  # noqa: E402
    _count_packable_multi,
    collect_unique_packable_unions,
    collect_unique_slot_limit_unions,
    try_allocate_single_pass_packable,
)
from rocprof_compute_soc.soc_base import OmniSoC_Base  # noqa: E402
from utils.mi_gpu_spec import mi_gpu_specs  # noqa: E402

_ENV_KEYS = (
    "ROCPROF_COMPUTE_PERFMON_SINGLE_PASS_PACKABLE",
    "ROCPROF_COMPUTE_PERFMON_LEGACY_HEURISTIC",
)


def _backup_env() -> Dict[str, Optional[str]]:
    return {k: os.environ.get(k) for k in _ENV_KEYS}


def _restore_env(backup: Dict[str, Optional[str]]) -> None:
    for key, val in backup.items():
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = val


def _build_soc(arch: str, tmp: Path) -> Tuple[OmniSoC_Base, Set[str], Dict[str, int]]:
    config_dir = get_default_config_dir()
    perfmon_config = mi_gpu_specs.get_perfmon_config(arch)
    soc = _build_inspector_soc(arch, config_dir, None, perfmon_config, tmp)
    counters, _ = soc.detect_counters()
    counters -= {"SQ_ACCUM_PREV_HIRES"}
    soc.get_rocprof_supported_counters = (  # type: ignore[method-assign]
        lambda c=counters: _rocprof_supported_superset(c)
    )
    return soc, counters, perfmon_config


def _run_legacy(arch: str) -> Dict[str, Union[int, float]]:
    backup = _backup_env()
    try:
        os.environ["ROCPROF_COMPUTE_PERFMON_LEGACY_HEURISTIC"] = "1"
        with tempfile.TemporaryDirectory(prefix="eval_spp_") as tmp:
            soc, counters, perfmon_config = _build_soc(arch, Path(tmp))
            t0 = time.perf_counter()
            files, _fc, _acc = soc._allocate_perfmon_counter_files(set(counters))
            elapsed = time.perf_counter() - t0
            unions, packable_n = collect_unique_packable_unions(
                soc, counters, perfmon_config
            )
            multi = count_multi_bucket_metrics(files, soc, counters)
            packable_multi = _count_packable_multi(files, unions)
    finally:
        _restore_env(backup)
    return {
        "passes": len(files),
        "pmc_total": len(counters),
        "multi_metrics": multi,
        "packable_multi": packable_multi,
        "packable_metrics": packable_n,
        "unique_packable_unions": len(unions),
        "seconds": round(elapsed, 3),
    }


def _run_spp_default(arch: str) -> Dict[str, Union[int, float]]:
    backup = _backup_env()
    try:
        os.environ.pop("ROCPROF_COMPUTE_PERFMON_LEGACY_HEURISTIC", None)
        os.environ.pop("ROCPROF_COMPUTE_PERFMON_SINGLE_PASS_PACKABLE", None)
        with tempfile.TemporaryDirectory(prefix="eval_spp_") as tmp:
            soc, counters, perfmon_config = _build_soc(arch, Path(tmp))
            t0 = time.perf_counter()
            result = try_allocate_single_pass_packable(
                soc, set(counters), perfmon_config
            )
            elapsed = time.perf_counter() - t0
            assert result is not None, "SPP allocate returned None (unexpected)"
            files, _fc, stats = result
            unions, packable_n = collect_unique_packable_unions(
                soc, counters, perfmon_config
            )
            _slot_unions, slot_n = collect_unique_slot_limit_unions(
                soc, counters, perfmon_config
            )
            packable_multi = _count_packable_multi(files, unions)
            # Product narrative: with_pmc minus SPU parents (inspector taxonomy).
            product_single_pass = packable_n  # allocator packable count
    finally:
        _restore_env(backup)

    return {
        "passes": stats.bucket_count,
        "pmc_total": len(counters),
        "packable_multi": packable_multi,
        "packable_metrics": packable_n,
        "product_single_pass_metrics": product_single_pass,
        "unique_packable_unions": len(unions),
        "slot_limit_metrics": stats.slot_limit_metrics,
        "unique_slot_limit_unions": stats.unique_slot_limit_unions,
        "slot_additional_passes": stats.slot_additional_passes,
        "merges_applied": stats.merges_applied,
        "seconds": round(elapsed, 3),
        "slot_n_check": slot_n,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="gfx942")
    args = parser.parse_args()

    baseline = _run_legacy(args.arch)
    spp = _run_spp_default(args.arch)

    print(f"arch={args.arch}")
    print()
    print("Legacy (LEGACY_HEURISTIC=1 + priority coalesce + first-fit)")
    for key, val in baseline.items():
        print(f"  {key}: {val}")
    print()
    print("Default single-pass-packable (+ SPU residual fill)")
    for key, val in spp.items():
        print(f"  {key}: {val}")
    print()
    print(
        "vs legacy: "
        f"{spp['passes'] - baseline['passes']:+d} "
        f"({baseline['passes']} → {spp['passes']})"
    )
    if args.arch == "gfx942":
        # Phase 1 SPP: 14 passes, packable_multi==0, SLOT fill +0.
        # Phase 2: SLOT_LIMIT parents are analyze composites (0 HW PMC parents).
        ok = (
            spp["packable_multi"] == 0
            and spp["passes"] == 14
            and spp["slot_additional_passes"] == 0
            and spp["slot_limit_metrics"] == 0
        )
        print(f"gfx942 gates: {'PASS' if ok else 'FAIL'}")
        if not ok:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
