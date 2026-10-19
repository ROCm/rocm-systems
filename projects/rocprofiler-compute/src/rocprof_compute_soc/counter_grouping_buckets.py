# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Shared helpers for single-pass, legacy, and inspector bucket packing."""

from typing import TYPE_CHECKING, Any, Dict, FrozenSet, List, Optional, Set, Tuple

from utils.utils_common import (
    METRIC_ID_RE,
    convert_metric_id_to_panel_info,
)
from utils.utils_counter_defs import extract_counters_and_variables

from .counter_file import CounterFile, flat_counters_in_perfmon_file

if TYPE_CHECKING:
    from .soc_base import OmniSoC_Base

_MetricGroup = Tuple[Tuple[Any, ...], FrozenSet[str], str]


def counters_fit_one_bucket(
    counters: FrozenSet[str],
    perfmon_config: Dict[str, int],
) -> bool:
    if not counters:
        return True
    trial = CounterFile("trial", perfmon_config)
    for ctr in sorted(counters):
        if not trial.add(ctr):
            return False
    return bool(flat_counters_in_perfmon_file(trial))


def bucket_counter_set(counter_file: CounterFile) -> Set[str]:
    """PMC names currently stored in one perfmon bucket."""
    return set(flat_counters_in_perfmon_file(counter_file))


def rebuild_counter_file(
    name: str,
    perfmon_config: Dict[str, int],
    counters: Set[str],
) -> Optional[CounterFile]:
    """Rebuild a bucket from its PMC set (*_ACCUM shares BASE when present)."""
    counter_file = CounterFile(name, perfmon_config)
    for ctr in sorted(counters):
        if not counter_file.add(ctr):
            return None
    return counter_file


def iter_metric_groups(
    soc: "OmniSoC_Base",
    profile_counters: Set[str],
) -> List[_MetricGroup]:
    """Metric PMC groups in priority-tier order for the profiled counters."""
    priority_keys: Set[Tuple[str, Any, int]] = set()
    for token in soc._same_bucket_priority_metric_ids():
        tid = token.strip()
        if not METRIC_ID_RE.match(tid):
            continue
        file_id, panel_id, metric_idx = convert_metric_id_to_panel_info(tid)
        if metric_idx is None:
            continue
        priority_keys.add((file_id, panel_id, metric_idx))

    rows: List[_MetricGroup] = []
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


def count_single_pass_metrics_split_across_buckets(
    output_files: List[CounterFile],
    soc: "OmniSoC_Base",
    profile_counters: Set[str],
    perfmon_config: Dict[str, int],
) -> int:
    """Metrics that span buckets but whose PMC set fits one hardware bucket."""
    metric_groups = iter_metric_groups(soc, profile_counters)
    bucket_sets = _bucket_counter_sets(output_files)
    return _single_pass_metrics_split_across_buckets_count(
        metric_groups, bucket_sets, perfmon_config
    )


def count_multi_bucket_metrics(
    output_files: List[CounterFile],
    soc: "OmniSoC_Base",
    profile_counters: Set[str],
) -> int:
    """Metrics with in-profile PMCs spanning 2+ perfmon buckets."""
    metric_groups = iter_metric_groups(soc, profile_counters)
    bucket_sets = _bucket_counter_sets(output_files)
    count = 0
    for _sort_key, group, _label in metric_groups:
        if _group_bucket_span(group, bucket_sets) > 1:
            count += 1
    return count


def _bucket_counter_sets(output_files: List[CounterFile]) -> List[Set[str]]:
    return [bucket_counter_set(counter_file) for counter_file in output_files]


def _group_bucket_span(
    group: FrozenSet[str],
    bucket_sets: List[Set[str]],
) -> int:
    """How many buckets the group occupies.

    Zero when a counter is missing. One when some bucket holds the whole
    group, even if a member is also copied into another bucket. Otherwise
    the number of buckets that contain at least one member.
    """
    if not group:
        return 0
    if not all(
        any(counter in counters for counters in bucket_sets) for counter in group
    ):
        return 0
    if any(group <= counters for counters in bucket_sets):
        return 1
    return sum(1 for counters in bucket_sets if group & counters)


def _single_pass_metrics_split_across_buckets_count(
    metric_groups: List[_MetricGroup],
    bucket_sets: List[Set[str]],
    perfmon_config: Dict[str, int],
) -> int:
    count = 0
    for _sort_key, group, _label in metric_groups:
        if _group_bucket_span(group, bucket_sets) <= 1:
            continue
        if counters_fit_one_bucket(group, perfmon_config):
            count += 1
    return count
