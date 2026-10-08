# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Shared helpers for perfmon bucket packing (SPP and inspector tools)."""

from typing import TYPE_CHECKING, Any, Dict, FrozenSet, List, Optional, Set, Tuple

from utils.metrics.expression import (
    is_composite_avg_formula,
    reject_counter_bearing_composite_extents,
)
from utils.utils_counter_defs import extract_counters_and_variables
from vendored import yaml

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


def _metric_body(metric_yaml: str) -> Optional[Dict[str, Any]]:
    try:
        body = yaml.safe_load(metric_yaml)
    except yaml.YAMLError:
        return None
    if isinstance(body, dict):
        return body
    return None


def _composite_formula(body: Dict[str, Any]) -> Optional[str]:
    formula = body.get("avg")
    if not isinstance(formula, str):
        formula = body.get("value")
    if isinstance(formula, str) and is_composite_avg_formula(formula):
        return formula
    return None


def _weight_counters_by_submetric(
    raw_metrics: List[Tuple[Any, ...]],
) -> Dict[Tuple[str, str, str], Set[str]]:
    """Map (stem, panel, sub-metric name) to WEIGHTED_AVG weight counters."""
    extras: Dict[Tuple[str, str, str], Set[str]] = {}
    for stem_id, panel_id, _metric_idx, _metric_name, metric_yaml in raw_metrics:
        body = _metric_body(metric_yaml)
        if body is None:
            continue
        meta = body.get("_weighted_avg")
        if not isinstance(meta, dict):
            continue
        panel_key = str(panel_id) if panel_id is not None else ""
        for sub_name, sub_meta in meta.items():
            if not isinstance(sub_meta, dict):
                continue
            counter = sub_meta.get("weight_counter")
            if not isinstance(counter, str) or not counter:
                continue
            key = (str(stem_id), panel_key, str(sub_name))
            extras.setdefault(key, set()).add(counter)
    return extras


def iter_metric_groups(
    soc: "OmniSoC_Base",
    profile_counters: Set[str],
) -> List[_MetricGroup]:
    """Metric PMC groups for the profiled counters, largest first.

    A WEIGHTED_AVG weight counter is added to the sub-ratio collectable so
    Phase 1 can place it in that collectable's bucket. Composite parents are
    not a second PMC pack.
    """
    raw_metrics = list(soc._iter_arch_analysis_yaml_metrics())
    weight_extras = _weight_counters_by_submetric(raw_metrics)

    rows: List[_MetricGroup] = []
    for (
        stem_id,
        panel_id,
        metric_idx,
        metric_name,
        metric_yaml,
    ) in raw_metrics:
        body = _metric_body(metric_yaml)
        if body is not None:
            reject_counter_bearing_composite_extents(metric_name, body)
            if _composite_formula(body) is not None:
                continue
        formula_hw, _ = extract_counters_and_variables(
            metric_yaml,
            soc._mspec.gpu_series,
            include_supported_denom=False,
        )
        panel_key = str(panel_id) if panel_id is not None else ""
        extra = weight_extras.get((str(stem_id), panel_key, str(metric_name)), set())
        hw = soc._expand_tcc_template_counters(set(formula_hw) | set(extra))
        counters = frozenset(hw & profile_counters)
        if not counters:
            continue
        sort_key = (-len(counters), stem_id, panel_key, metric_idx)
        label = f"{stem_id}.{panel_key}.{metric_idx} ({metric_name})"
        rows.append((sort_key, counters, label))
    rows.sort(key=lambda row: row[0])
    return rows


def count_packable_multi_bucket_metrics(
    output_files: List[CounterFile],
    soc: "OmniSoC_Base",
    profile_counters: Set[str],
    perfmon_config: Dict[str, int],
) -> int:
    """Metrics that span buckets but whose PMC set fits one hardware bucket."""
    metric_groups = iter_metric_groups(soc, profile_counters)
    ctr_to_bucket = _counter_to_bucket_index(output_files)
    return _packable_multi_bucket_count(metric_groups, ctr_to_bucket, perfmon_config)


def count_multi_bucket_metrics(
    output_files: List[CounterFile],
    soc: "OmniSoC_Base",
    profile_counters: Set[str],
) -> int:
    """Metrics with in-profile PMCs spanning 2+ perfmon buckets."""
    metric_groups = iter_metric_groups(soc, profile_counters)
    ctr_to_bucket = _counter_to_bucket_index(output_files)
    count = 0
    for _sort_key, group, _label in metric_groups:
        if _metric_bucket_count(group, ctr_to_bucket) > 1:
            count += 1
    return count


def _counter_to_bucket_index(
    output_files: List[CounterFile],
) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for idx, counter_file in enumerate(output_files):
        for ctr in flat_counters_in_perfmon_file(counter_file):
            mapping[ctr] = idx
    return mapping


def _packable_multi_bucket_count(
    metric_groups: List[_MetricGroup],
    ctr_to_bucket: Dict[str, int],
    perfmon_config: Dict[str, int],
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
    group: FrozenSet[str],
    ctr_to_bucket: Dict[str, int],
) -> int:
    return len({ctr_to_bucket[ctr] for ctr in group if ctr in ctr_to_bucket})
