#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT
"""Report single-pass metric allocation gates for one architecture.

Example::

    PYTHONPATH=src:tools python3 tools/eval_single_pass_metrics.py --arch gfx942
"""

import argparse
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Set, Tuple, Union

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT / "src", _ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from counter_grouping_inspector import (  # noqa: E402
    _build_inspector_soc,
    _rocprof_supported_superset,
    get_default_config_dir,
)

from rocprof_compute_soc.counter_grouping_single_pass import (  # noqa: E402
    _count_single_pass_metric_sets_without_pass,
    collect_unique_multi_pass_metric_sets,
    collect_unique_single_pass_metric_sets,
    try_allocate_single_pass_metrics,
)
from rocprof_compute_soc.soc_base import OmniSoC_Base  # noqa: E402
from utils.mi_gpu_spec import mi_gpu_specs  # noqa: E402


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


def _run_single_pass_metric_default(arch: str) -> Dict[str, Union[int, float]]:
    with tempfile.TemporaryDirectory(prefix="eval_single_pass_metric_") as tmp:
        soc, counters, perfmon_config = _build_soc(arch, Path(tmp))
        t0 = time.perf_counter()
        result = try_allocate_single_pass_metrics(soc, set(counters), perfmon_config)
        elapsed = time.perf_counter() - t0
        assert result is not None, (
            "single-pass metric allocation returned None (unexpected)"
        )
        files, _fc, stats = result
        metric_sets, single_pass_metric_count = collect_unique_single_pass_metric_sets(
            soc, counters, perfmon_config
        )
        _multi_pass_metric_sets, multi_pass_metric_count = (
            collect_unique_multi_pass_metric_sets(soc, counters, perfmon_config)
        )
        single_pass_metric_sets_without_pass = (
            _count_single_pass_metric_sets_without_pass(files, metric_sets)
        )
        product_single_pass = single_pass_metric_count

    return {
        "passes": stats.bucket_count,
        "pmc_total": len(counters),
        "single_pass_metric_sets_without_pass": single_pass_metric_sets_without_pass,
        "single_pass_metric_count": single_pass_metric_count,
        "product_single_pass_metrics": product_single_pass,
        "unique_single_pass_metric_sets": len(metric_sets),
        "multi_pass_metric_count": stats.multi_pass_metric_count,
        "unique_multi_pass_metric_sets": stats.unique_multi_pass_metric_sets,
        "multi_pass_additional_passes": stats.multi_pass_additional_passes,
        "merges_applied": stats.merges_applied,
        "seconds": round(elapsed, 3),
        "multi_pass_metric_count_check": multi_pass_metric_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="gfx942")
    args = parser.parse_args()

    single_pass_metrics = _run_single_pass_metric_default(args.arch)

    print(f"arch={args.arch}")
    print()
    print("Default single-pass metric (+ multi-pass metric fill)")
    for key, val in single_pass_metrics.items():
        print(f"  {key}: {val}")
    print()
    if args.arch == "gfx942":
        ok = (
            single_pass_metrics["single_pass_metric_sets_without_pass"] == 0
            and single_pass_metrics["passes"] == 14
            and single_pass_metrics["multi_pass_additional_passes"] == 0
            and single_pass_metrics["multi_pass_metric_count"] == 16
        )
        print(f"gfx942 gates: {'PASS' if ok else 'FAIL'}")
        if not ok:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
