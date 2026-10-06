# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Shared helpers for perfmon bucket packing (SPP, legacy, inspector tools)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from utils.utils_common import (
    METRIC_ID_RE,
    convert_metric_id_to_panel_info,
)
from utils.utils_counter_defs import extract_counters_and_variables

from .counter_file import CounterFile, flat_counters_in_perfmon_file

if TYPE_CHECKING:
    from .soc_base import OmniSoC_Base


def counters_fit_one_bucket(
    counters: frozenset[str],
    perfmon_config: dict[str, int],
) -> bool:
    if not counters:
        return True
    trial = CounterFile("trial", perfmon_config)
    for ctr in sorted(counters):
        if not trial.add(ctr):
            return False
    return bool(flat_counters_in_perfmon_file(trial))


def _bucket_counter_set(counter_file: CounterFile) -> set[str]:
    return set(flat_counters_in_perfmon_file(counter_file))


def _counter_to_bucket_index(
    output_files: list[CounterFile],
) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for idx, counter_file in enumerate(output_files):
        for ctr in flat_counters_in_perfmon_file(counter_file):
            mapping[ctr] = idx
    return mapping


def rebuild_counter_file(
    name: str,
    perfmon_config: dict[str, int],
    counters: set[str],
) -> CounterFile | None:
    """Rebuild a bucket from its PMC set (``*_ACCUM`` shares BASE when present)."""
    counter_file = CounterFile(name, perfmon_config)
    for ctr in sorted(counters):
        if not counter_file.add(ctr):
            return None
    return counter_file


def _iter_metric_groups(
    soc: OmniSoC_Base,
    profile_counters: set[str],
) -> list[tuple[tuple[Any, ...], frozenset[str], str]]:
    priority_keys: set[tuple[str, Any, int]] = set()
    for token in soc._same_bucket_priority_metric_ids():
        tid = token.strip()
        if not METRIC_ID_RE.match(tid):
            continue
        file_id, panel_id, metric_idx = convert_metric_id_to_panel_info(tid)
        if metric_idx is None:
            continue
        priority_keys.add((file_id, panel_id, metric_idx))

    rows: list[tuple[tuple[Any, ...], frozenset[str], str]] = []
    for (
        stem_id,
        panel_id,
        metric_idx,
        metric_name,
        metric_yaml,
    ) in soc._iter_arch_analysis_yaml_metrics():
        formula_hw, _ = extract_counters_and_variables(
            metric_yaml,
            soc._mspec.gpu_series,
            include_supported_denom=False,
        )
        hw = soc._expand_tcc_template_counters(formula_hw)
        counters = frozenset(hw & profile_counters)
        if not counters:
            continue
        tier = 0 if (stem_id, panel_id, metric_idx) in priority_keys else 1
        panel_s = str(panel_id) if panel_id is not None else ""
        sort_key = (tier, -len(counters), stem_id, panel_s, metric_idx)
        label = f"{stem_id}.{panel_s}.{metric_idx} ({metric_name})"
        rows.append((sort_key, counters, label))
    rows.sort(key=lambda row: row[0])
    return rows


def count_packable_multi_bucket_metrics(
    output_files: list[CounterFile],
    soc: OmniSoC_Base,
    profile_counters: set[str],
    perfmon_config: dict[str, int],
) -> int:
    """Metrics that span buckets but whose PMC set fits one hardware bucket."""
    metric_groups = _iter_metric_groups(soc, profile_counters)
    ctr_to_bucket = _counter_to_bucket_index(output_files)
    return _packable_multi_bucket_count(metric_groups, ctr_to_bucket, perfmon_config)


def count_multi_bucket_metrics(
    output_files: list[CounterFile],
    soc: OmniSoC_Base,
    profile_counters: set[str],
) -> int:
    """Metrics with in-profile PMCs spanning 2+ perfmon buckets."""
    metric_groups = _iter_metric_groups(soc, profile_counters)
    ctr_to_bucket = _counter_to_bucket_index(output_files)
    count = 0
    for _sort_key, group, _label in metric_groups:
        if _metric_bucket_count(group, ctr_to_bucket) > 1:
            count += 1
    return count


def _packable_multi_bucket_count(
    metric_groups: list[tuple[tuple[Any, ...], frozenset[str], str]],
    ctr_to_bucket: dict[str, int],
    perfmon_config: dict[str, int],
) -> int:
    count = 0
    for _sort_key, group, _label in metric_groups:
        buckets = {ctr_to_bucket[ctr] for ctr in group if ctr in ctr_to_bucket}
        if len(buckets) <= 1:
            continue
        if counters_fit_one_bucket(group, perfmon_config):
            count += 1
    return count


def _metric_bucket_count(
    group: frozenset[str],
    ctr_to_bucket: dict[str, int],
) -> int:
    return len({ctr_to_bucket[ctr] for ctr in group if ctr in ctr_to_bucket})
