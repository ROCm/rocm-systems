#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT
"""Emit JSON for gfx942 problem-decompose HTML (stdout)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
_TOOLS = _ROOT / "tools"
for _p in (_SRC, _TOOLS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from counter_grouping_inspector import (  # noqa: E402
    _build_inspector_soc,
    _rocprof_supported_superset,
    get_default_config_dir,
    iter_yaml_metrics,
)

from rocprof_compute_soc.counter_grouping_refill import (  # noqa: E402
    apply_metric_coalesce_refill_pass,
    counters_fit_one_bucket,
)
from rocprof_compute_soc.counter_grouping_single_pass import (  # noqa: E402
    try_allocate_single_pass_packable,
)
from rocprof_compute_soc.soc_base import (  # noqa: E402
    CounterFile,
    flat_counters_in_perfmon_file,
)
from utils.mi_gpu_spec import mi_gpu_specs  # noqa: E402
from utils.utils_counter_defs import extract_counters_and_variables  # noqa: E402

_LEGACY_ENV = "ROCPROF_COMPUTE_PERFMON_LEGACY_HEURISTIC"
_SPP_ENV = "ROCPROF_COMPUTE_PERFMON_SINGLE_PASS_PACKABLE"


def _counter_to_bucket(output_files: list[CounterFile]) -> dict[str, str]:
    out: dict[str, str] = {}
    for cf in output_files:
        label = cf.name.replace(".txt", "")
        for ctr in flat_counters_in_perfmon_file(cf):
            out[ctr] = label
    return out


def _metric_rows(
    output_files: list[CounterFile],
    config_dir: Path,
    arch: str,
    profile_counters: set[str],
) -> list[dict]:
    ctr_to_bucket = _counter_to_bucket(output_files)
    gpu_series = mi_gpu_specs.get_gpu_series(arch)
    rows: list[dict] = []
    for file_id, panel_id, metric_idx, metric_name, metric_yaml in iter_yaml_metrics(
        config_dir, arch
    ):
        hw, _ = extract_counters_and_variables(
            metric_yaml, gpu_series, include_supported_denom=False
        )
        in_profile = {c for c in hw if c in profile_counters}
        if not in_profile:
            continue
        buckets = {ctr_to_bucket[c] for c in in_profile if c in ctr_to_bucket}
        bucket_count = len(buckets)
        packable = bucket_count > 1 and counters_fit_one_bucket(
            frozenset(in_profile), mi_gpu_specs.get_perfmon_config(arch)
        )
        slot_limit = (
            bucket_count > 1
            and not packable
            and not counters_fit_one_bucket(
                frozenset(in_profile), mi_gpu_specs.get_perfmon_config(arch)
            )
        )
        rows.append({
            "id": f"{file_id}.{panel_id}.{metric_idx}",
            "file_id": file_id,
            "panel_id": str(panel_id) if panel_id is not None else "-",
            "metric_idx": metric_idx,
            "name": metric_name,
            "bucket_count": bucket_count,
            "buckets": sorted(buckets),
            "pmc_count": len(in_profile),
            "pmcs": sorted(in_profile),
            "packable": packable,
            "slot_limit": slot_limit and bucket_count > 1,
            "single_pass": bucket_count <= 1,
        })
    return rows


def _pmc_duplicate_summary(metrics: list[dict]) -> dict:
    """Group metrics that share the exact same in-profile PMC set (panel mirrors)."""
    by_pmc: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for metric in metrics:
        key = tuple(metric["pmcs"])
        by_pmc[key].append({"id": metric["id"], "name": metric["name"]})

    groups = []
    for pmc_key, members in by_pmc.items():
        if len(members) < 2:
            continue
        names = sorted({m["name"] for m in members})
        groups.append({
            "count": len(members),
            "pmc_count": len(pmc_key),
            "names": names,
            "metrics": sorted(members, key=lambda m: m["id"]),
        })
    groups.sort(key=lambda g: (-g["count"], g["metrics"][0]["id"]))
    redundant = sum(g["count"] - 1 for g in groups)
    return {
        "metric_count": len(metrics),
        "unique_pmc_sets": len(by_pmc),
        "duplicate_groups": len(groups),
        "redundant_rows": redundant,
        "groups": groups,
    }


def _public_metric(metric: dict) -> dict:
    return {k: v for k, v in metric.items() if k != "pmcs"}


def _summarize(rows: list[dict]) -> dict:
    with_pmc = len(rows)
    single = [r for r in rows if r["single_pass"]]
    multi = [r for r in rows if r["bucket_count"] > 1]
    packable = [r for r in multi if r["packable"]]
    slot = [r for r in multi if r["slot_limit"]]
    return {
        "with_pmc": with_pmc,
        "single_count": len(single),
        "single_pct": round(100 * len(single) / with_pmc, 1) if with_pmc else 0,
        "multi": [_public_metric(r) for r in multi],
        "packable_multi": [_public_metric(r) for r in packable],
        "slot_limit": [_public_metric(r) for r in slot],
        "slot_limit_pmc_dups": _pmc_duplicate_summary(slot),
    }


def main() -> None:
    arch = "gfx942"
    config_dir = get_default_config_dir()
    perfmon_config = mi_gpu_specs.get_perfmon_config(arch)
    saved_env = {key: os.environ.get(key) for key in (_LEGACY_ENV, _SPP_ENV)}

    try:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            soc = _build_inspector_soc(arch, config_dir, None, perfmon_config, root)
            counters, _ = soc.detect_counters()
            counters -= {"SQ_ACCUM_PREV_HIRES"}
            soc.get_rocprof_supported_counters = (  # type: ignore[method-assign]
                lambda c=counters: _rocprof_supported_superset(c)
            )

            # Overview tables: legacy coalesce / first-fit / refill comparison.
            os.environ[_LEGACY_ENV] = "1"
            os.environ.pop(_SPP_ENV, None)
            before_files, fc, _ = soc._allocate_perfmon_counter_files(
                set(counters), apply_refill=False
            )
            after_files, _fc2, refill_stats = apply_metric_coalesce_refill_pass(
                soc, before_files, fc, set(counters), perfmon_config
            )

            # Default allocate path: single-pass packable + SLOT_LIMIT fill.
            os.environ.pop(_LEGACY_ENV, None)
            os.environ.pop(_SPP_ENV, None)
            spp = try_allocate_single_pass_packable(soc, set(counters), perfmon_config)
            if spp is None:
                raise RuntimeError("single-pass-packable allocate returned None")
            spp_files, _spp_fc, spp_stats = spp
    finally:
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    before_rows = _metric_rows(before_files, config_dir, arch, counters)
    after_rows = _metric_rows(after_files, config_dir, arch, counters)

    yaml_metric_total = sum(1 for _ in iter_yaml_metrics(config_dir, arch))

    payload = {
        "arch": arch,
        "yaml_metric_total": yaml_metric_total,
        "pmc_total": len(counters),
        "buckets_before": len(before_files),
        "buckets_after": len(after_files),
        "refill_consolidated": refill_stats.metrics_consolidated,
        "before": _summarize(before_rows),
        "after": _summarize(after_rows),
        "spp": {
            "buckets": spp_stats.bucket_count,
            "packable_metric_count": spp_stats.packable_metric_count,
            "unique_packable_unions": spp_stats.unique_packable_unions,
            "packable_multi_after": spp_stats.packable_multi_after,
            "merges_applied": spp_stats.merges_applied,
            "slot_limit_metrics": spp_stats.slot_limit_metrics,
            "unique_slot_limit_unions": spp_stats.unique_slot_limit_unions,
            "slot_additional_passes": spp_stats.slot_additional_passes,
            "pmc_files": len(spp_files),
        },
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
