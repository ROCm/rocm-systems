#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT
"""Three-way gfx942 allocator compare on a TCC-channel-free PMC subset."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _ROOT / "src"
_TOOLS_DIR = _ROOT / "tools"
for _p in (_SRC_DIR, _TOOLS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from counter_grouping_inspector import (  # noqa: E402
    _build_inspector_soc,
    _rocprof_supported_superset,
    get_default_config_dir,
)

from rocprof_compute_soc.counter_grouping_refill import (  # noqa: E402
    count_multi_bucket_metrics,
    count_packable_multi_bucket_metrics,
)
from rocprof_compute_soc.soc_base import is_tcc_channel_counter  # noqa: E402
from utils.mi_gpu_spec import mi_gpu_specs  # noqa: E402
from utils.utils_common import canonical_config_arch  # noqa: E402


def _tcc_free_counters(counters: set[str]) -> set[str]:
    return {
        c
        for c in counters
        if not is_tcc_channel_counter(c)
        and not (c.startswith("TCC") and c.endswith("["))
    }


def _run_mode(
    arch: str,
    config_dir: Path,
    perfmon_config: dict[str, int],
    counters: set[str],
    *,
    label: str,
    cp_sat: bool,
    metric_penalty: int | None,
) -> dict[str, float | int | str]:
    env_backup = {
        "ROCPROF_COMPUTE_PERFMON_CP_SAT": os.environ.get(
            "ROCPROF_COMPUTE_PERFMON_CP_SAT"
        ),
        "ROCPROF_COMPUTE_PERFMON_CP_SAT_METRIC_PENALTY": os.environ.get(
            "ROCPROF_COMPUTE_PERFMON_CP_SAT_METRIC_PENALTY"
        ),
    }
    try:
        if cp_sat:
            os.environ["ROCPROF_COMPUTE_PERFMON_CP_SAT"] = "1"
            if metric_penalty is not None:
                os.environ["ROCPROF_COMPUTE_PERFMON_CP_SAT_METRIC_PENALTY"] = str(
                    metric_penalty
                )
        else:
            os.environ.pop("ROCPROF_COMPUTE_PERFMON_CP_SAT", None)
            os.environ.pop("ROCPROF_COMPUTE_PERFMON_CP_SAT_METRIC_PENALTY", None)

        with tempfile.TemporaryDirectory(prefix="rocprof_three_way_") as tmpdir:
            workload_root = Path(tmpdir)
            soc = _build_inspector_soc(
                arch, config_dir, None, perfmon_config, workload_root
            )
            soc.get_rocprof_supported_counters = (  # type: ignore[method-assign]
                lambda c=counters: _rocprof_supported_superset(c)
            )
            t0 = time.perf_counter()
            output_files, _file_count, _acc = soc._allocate_perfmon_counter_files(
                set(counters),
                apply_refill=True,
            )
            elapsed = time.perf_counter() - t0
    finally:
        for key, val in env_backup.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val

    multi = count_multi_bucket_metrics(output_files, soc, counters)
    packable = count_packable_multi_bucket_metrics(
        output_files, soc, counters, perfmon_config
    )
    return {
        "mode": label,
        "buckets": len(output_files),
        "multi_bucket": multi,
        "packable_multi": packable,
        "seconds": round(elapsed, 2),
        "pmc_count": len(counters),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", default="gfx942")
    parser.add_argument("--config-dir", type=Path, default=None)
    args = parser.parse_args()

    config_dir = args.config_dir or get_default_config_dir()
    arch = args.arch
    perfmon_config = mi_gpu_specs.get_perfmon_config(arch)
    if not perfmon_config:
        raise SystemExit(f"No perfmon config for {arch}")

    with tempfile.TemporaryDirectory(prefix="rocprof_three_way_detect_") as tmpdir:
        soc = _build_inspector_soc(arch, config_dir, None, perfmon_config, Path(tmpdir))
        full, _ = soc.detect_counters()
        full -= {"SQ_ACCUM_PREV_HIRES"}

    tcc_free = _tcc_free_counters(full)
    tcc_dropped = len(full) - len(tcc_free)

    print(f"Architecture: {arch} ({canonical_config_arch(arch)})")
    print(f"Full PMC set: {len(full)} counters")
    print(
        f"TCC-free subset: {len(tcc_free)} counters "
        f"(dropped {tcc_dropped} TCC channel/template PMCs)"
    )
    print()

    modes = [
        ("heuristic+refill", False, None),
        ("cp-sat min-bins (penalty=0)", True, 0),
        ("cp-sat spread (penalty=100)", True, 100),
    ]
    rows: list[dict[str, float | int | str]] = []
    for label, cp_sat, penalty in modes:
        rows.append(
            _run_mode(
                arch,
                config_dir,
                perfmon_config,
                tcc_free,
                label=label,
                cp_sat=cp_sat,
                metric_penalty=penalty,
            )
        )

    header = f"{'Mode':<32} {'Buckets':>8} {'Multi':>8} {'Packable':>10} {'Sec':>8}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['mode']:<32} {row['buckets']:>8} {row['multi_bucket']:>8} "
            f"{row['packable_multi']:>10} {row['seconds']:>8}"
        )


if __name__ == "__main__":
    main()
