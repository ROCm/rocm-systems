# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Second-pass coalesce repair: co-locate packable metrics without new replays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from utils.logger import console_debug
from utils.utils_common import (
    METRIC_ID_RE,
    convert_metric_id_to_panel_info,
    is_tcc_channel_counter,
)
from utils.utils_counter_defs import extract_counters_and_variables

from .soc_base import CounterFile, flat_counters_in_perfmon_file

if TYPE_CHECKING:
    from .soc_base import OmniSoC_Base


@dataclass(frozen=True)
class RefillStats:
    """Counters for packable multi-bucket metrics before/after refill."""

    bucket_count: int
    packable_multi_bucket_before: int
    packable_multi_bucket_after: int
    metrics_consolidated: int


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


def _needs_accum_reserve(name: str, counters: set[str]) -> bool:
    if name.endswith("_ACCUM"):
        return True
    return any(
        ctr.endswith("_ACCUM") and not is_tcc_channel_counter(ctr) for ctr in counters
    )


def rebuild_counter_file(
    name: str,
    perfmon_config: dict[str, int],
    counters: set[str],
) -> CounterFile | None:
    counter_file = CounterFile(name, perfmon_config)
    for ctr in sorted(counters):
        if not counter_file.add(ctr):
            return None
    if _needs_accum_reserve(name, counters):
        for ctr in flat_counters_in_perfmon_file(counter_file):
            if ctr.endswith("_ACCUM") and not is_tcc_channel_counter(ctr):
                if not counter_file.reserve(ctr, 1):
                    return None
    return counter_file


def _can_place_group_in_bucket(
    output_files: list[CounterFile],
    bucket_idx: int,
    group: frozenset[str],
    perfmon_config: dict[str, int],
) -> bool:
    union = _bucket_counter_set(output_files[bucket_idx]) | set(group)
    rebuilt = rebuild_counter_file(
        output_files[bucket_idx].name,
        perfmon_config,
        union,
    )
    return rebuilt is not None


def _consolidate_group_into_bucket(
    output_files: list[CounterFile],
    target_idx: int,
    group: frozenset[str],
    perfmon_config: dict[str, int],
) -> list[CounterFile]:
    updated = list(output_files)
    group_set = set(group)
    for idx, counter_file in enumerate(updated):
        present = _bucket_counter_set(counter_file)
        if idx == target_idx:
            new_counters = present | group_set
        else:
            new_counters = present - group_set
        if new_counters == present:
            continue
        rebuilt = rebuild_counter_file(
            counter_file.name,
            perfmon_config,
            new_counters,
        )
        if rebuilt is None:
            msg = f"refill rebuild failed for bucket {counter_file.name!r}"
            raise RuntimeError(msg)
        updated[idx] = rebuilt
    return updated


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


def _try_consolidate_metric(
    files: list[CounterFile],
    group: frozenset[str],
    perfmon_config: dict[str, int],
) -> list[CounterFile] | None:
    best_idx: int | None = None
    best_overlap = -1
    for idx in range(len(files)):
        if not _can_place_group_in_bucket(files, idx, group, perfmon_config):
            continue
        overlap = len(group & _bucket_counter_set(files[idx]))
        if overlap > best_overlap:
            best_overlap = overlap
            best_idx = idx
    if best_idx is None:
        return None
    return _consolidate_group_into_bucket(files, best_idx, group, perfmon_config)


def apply_metric_coalesce_refill_pass(
    soc: OmniSoC_Base,
    output_files: list[CounterFile],
    file_count: int,
    profile_counters: set[str],
    perfmon_config: dict[str, int],
) -> tuple[list[CounterFile], int, RefillStats]:
    """Co-locate packable multi-bucket metrics into one bucket each; no new replays."""
    files = list(output_files)
    metric_groups = _iter_metric_groups(soc, profile_counters)
    ctr_to_bucket = _counter_to_bucket_index(files)
    before = _packable_multi_bucket_count(metric_groups, ctr_to_bucket, perfmon_config)
    consolidated = 0

    improved = True
    while improved:
        improved = False
        packable_now = count_packable_multi_bucket_metrics(
            files,
            soc,
            profile_counters,
            perfmon_config,
        )
        multi_now = count_multi_bucket_metrics(files, soc, profile_counters)
        for _sort_key, group, label in metric_groups:
            ctr_to_bucket = _counter_to_bucket_index(files)
            if _metric_bucket_count(group, ctr_to_bucket) <= 1:
                continue
            if not counters_fit_one_bucket(group, perfmon_config):
                continue

            trial_files = _try_consolidate_metric(files, group, perfmon_config)
            if trial_files is None:
                console_debug(
                    "profiling",
                    f"refill: no target bucket for packable metric {label!r}",
                )
                continue

            trial_packable = count_packable_multi_bucket_metrics(
                trial_files,
                soc,
                profile_counters,
                perfmon_config,
            )
            trial_multi = count_multi_bucket_metrics(trial_files, soc, profile_counters)
            trial_ctr = _counter_to_bucket_index(trial_files)
            if _metric_bucket_count(group, trial_ctr) > 1:
                continue
            if trial_packable > packable_now:
                continue
            if trial_packable == packable_now and trial_multi >= multi_now:
                continue

            files = trial_files
            consolidated += 1
            improved = True
            packable_now = trial_packable
            console_debug(
                "profiling",
                f"refill: consolidated {label!r} "
                f"(packable multi-bucket now {packable_now})",
            )

    ctr_to_bucket = _counter_to_bucket_index(files)
    after = _packable_multi_bucket_count(metric_groups, ctr_to_bucket, perfmon_config)
    stats = RefillStats(
        bucket_count=len(files),
        packable_multi_bucket_before=before,
        packable_multi_bucket_after=after,
        metrics_consolidated=consolidated,
    )
    console_debug(
        "profiling",
        "refill pass: "
        f"packable multi-bucket {before} -> {after}, "
        f"consolidated {consolidated}, buckets {len(files)}",
    )
    return files, file_count, stats
