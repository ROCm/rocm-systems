# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Default allocator with a single-bucket guarantee for single-pass metrics.

A single-pass metric has some perfmon bucket containing its full PMC set.
The allocator minimizes the number of passes under that hard constraint.
Counters not required by any single-pass metric set use ordinary first-fit.
Remaining multi-pass metric PMCs are placed into existing buckets when
possible, and new passes are opened only if needed. Short-term TCC channel
rules live in counter_grouping_tcc: affinity pairs, the multi-column request
row, and which extra copies to keep.

Overlapping single-pass metric sets that cannot share one bucket are handled by
duplicating counters into an additional bucket (additive passes). That
may duplicate counters into an additional bucket.
"""

from dataclasses import dataclass
from itertools import combinations
from typing import (
    TYPE_CHECKING,
    Dict,
    FrozenSet,
    List,
    Optional,
    Set,
    Tuple,
)

from utils.logger import console_debug, console_warning
from utils.utils_common import is_tcc_channel_counter

from .counter_file import CounterFile, flat_counters_in_perfmon_file
from .counter_grouping_buckets import (
    bucket_counter_set,
    counters_fit_one_bucket,
    iter_metric_groups,
    rebuild_counter_file,
)
from .counter_grouping_tcc import (
    adjust_candidate_groups,
    keep_copies_groups_still_need,
)

if TYPE_CHECKING:
    from .soc_base import OmniSoC_Base


def try_allocate_single_pass_metrics(
    soc: "OmniSoC_Base",
    work_set: Set[str],
    perfmon_config: Dict[str, int],
    file_count_start: int = 0,
) -> Optional[Tuple[List[CounterFile], int, "SinglePassMetricStats"]]:
    """Allocate work_set with single-pass metrics forced into one bucket.

    On success, clears work_set.

    Returns:
        (files, file_count, stats) when every single-pass metric set has a
        full bucket. None when work_set is empty, when a single-pass metric
        set still lacks a full bucket after allocation, or when multi-pass
        metric fill drops that coverage.
    """
    if not work_set:
        return None

    (
        single_pass_metric_sets,
        single_pass_metric_count,
        multi_pass_metric_sets,
        multi_pass_metric_count,
    ) = _partition_unique_metric_sets(soc, work_set, perfmon_config)
    files: List[CounterFile] = []
    file_count = file_count_start
    for group in single_pass_metric_sets:
        files, file_count = _ensure_single_pass_metric_set(
            files, group, perfmon_config, file_count
        )

    files, file_count = _first_fit_unplaced(files, work_set, perfmon_config, file_count)
    files, merges = _reduce_passes(files, single_pass_metric_sets, perfmon_config)
    files = keep_copies_groups_still_need(
        files, perfmon_config, single_pass_metric_sets
    )

    single_pass_metric_sets_without_pass = _count_single_pass_metric_sets_without_pass(
        files, single_pass_metric_sets
    )
    if single_pass_metric_sets_without_pass > 0:
        console_warning(
            "profiling",
            "single-pass metric allocation: "
            f"{single_pass_metric_sets_without_pass} single-pass metric set(s) "
            "still lack a full bucket after allocation.",
        )
        return None

    files, file_count, multi_pass_fill_stats = (
        fill_multi_pass_metrics_into_existing_passes(
            files,
            multi_pass_metric_sets,
            perfmon_config,
            multi_pass_metric_count=multi_pass_metric_count,
            file_count_start=file_count,
        )
    )
    files = keep_copies_groups_still_need(
        files, perfmon_config, single_pass_metric_sets
    )
    single_pass_metric_sets_without_pass = _count_single_pass_metric_sets_without_pass(
        files, single_pass_metric_sets
    )
    if single_pass_metric_sets_without_pass > 0:
        console_warning(
            "profiling",
            "single-pass metric allocation: "
            f"{single_pass_metric_sets_without_pass} single-pass metric set(s) "
            "lost coverage after multi-pass metric fill.",
        )
        return None
    file_count = _next_bucket_number(files, file_count_start)

    stats = SinglePassMetricStats(
        bucket_count=len(files),
        single_pass_metric_count=single_pass_metric_count,
        unique_single_pass_metric_sets=len(single_pass_metric_sets),
        single_pass_metric_sets_without_pass=single_pass_metric_sets_without_pass,
        merges_applied=merges,
        multi_pass_metric_count=multi_pass_fill_stats.multi_pass_metric_count,
        unique_multi_pass_metric_sets=multi_pass_fill_stats.unique_multi_pass_metric_sets,
        multi_pass_additional_passes=multi_pass_fill_stats.additional_passes,
    )
    console_debug(
        "profiling",
        "single-pass metric allocation: "
        f"{stats.bucket_count} bucket(s), "
        f"{stats.single_pass_metric_count} single-pass metric(s), "
        f"{stats.unique_single_pass_metric_sets} unique metric set(s), "
        "single_pass_metric_sets_without_pass="
        f"{stats.single_pass_metric_sets_without_pass}, "
        f"merges={stats.merges_applied}, "
        f"multi_pass_metrics={stats.multi_pass_metric_count}, "
        f"multi_pass_+passes={stats.multi_pass_additional_passes}.",
    )

    work_set.clear()
    return files, file_count, stats


def collect_unique_single_pass_metric_sets(
    soc: "OmniSoC_Base",
    profile_counters: Set[str],
    perfmon_config: Dict[str, int],
) -> Tuple[List[FrozenSet[str]], int]:
    """Unique PMC sets for metrics that fit one hardware bucket.

    Each set is adjusted by the short-term TCC rules before it is stored.

    Returns:
        Groups sorted largest-first, and the single-pass metric count.
    """
    single_sets, single_count, _multi_sets, _multi_count = (
        _partition_unique_metric_sets(soc, profile_counters, perfmon_config)
    )
    return single_sets, single_count


def collect_unique_multi_pass_metric_sets(
    soc: "OmniSoC_Base",
    profile_counters: Set[str],
    perfmon_config: Dict[str, int],
) -> Tuple[List[FrozenSet[str]], int]:
    """Unique PMC sets for metrics that cannot fit one hardware bucket.

    Returns:
        Metric sets sorted largest-first, and the multi-pass metric count.
    """
    _single_sets, _single_count, multi_sets, multi_count = (
        _partition_unique_metric_sets(soc, profile_counters, perfmon_config)
    )
    return multi_sets, multi_count


def fill_multi_pass_metrics_into_existing_passes(
    files: List[CounterFile],
    multi_pass_metric_sets: List[FrozenSet[str]],
    perfmon_config: Dict[str, int],
    *,
    multi_pass_metric_count: int = 0,
    file_count_start: Optional[int] = None,
) -> Tuple[List[CounterFile], int, "MultiPassMetricFillStats"]:
    """Place multi-pass metric PMCs into existing buckets; open new ones only if needed.

    For each unique multi-pass metric PMC set (the full set cannot fit one bucket):
    repeatedly pack the largest remaining subset into the best existing
    bucket, else open one new bucket with the largest subset that fits an
    empty bucket. A counter that fits no bucket, including a fresh empty
    one, raises ValueError.

    Returns:
        Updated files, the next file count, and fill stats.
    """
    passes_before = len(files)
    files = list(files)
    floor = passes_before if file_count_start is None else file_count_start
    file_count = _next_bucket_number(files, floor)
    pmc_already = 0
    pmc_into_existing = 0
    pmc_into_new = 0

    for group in multi_pass_metric_sets:
        remaining = set(group)
        placed_anywhere = _counters_in_any_bucket(files)
        already = remaining & placed_anywhere
        pmc_already += len(already)
        need = remaining - placed_anywhere
        while need:
            placed = _extend_best_existing_bucket(files, need, perfmon_config)
            if placed is not None:
                pmc_into_existing += len(placed)
                need -= placed
                continue

            subset = _largest_subset_fitting_empty(need, perfmon_config)
            if not subset:
                _raise_unplaceable_multi_pass_metric_counters(need)
            new_bucket = rebuild_counter_file(
                str(file_count), perfmon_config, set(subset)
            )
            if new_bucket is None:
                _raise_unplaceable_multi_pass_metric_counters(set(subset))
            files.append(new_bucket)
            file_count += 1
            pmc_into_new += len(subset)
            need -= subset

    passes_after = len(files)
    stats = MultiPassMetricFillStats(
        passes_before=passes_before,
        passes_after=passes_after,
        additional_passes=passes_after - passes_before,
        multi_pass_metric_count=multi_pass_metric_count,
        unique_multi_pass_metric_sets=len(multi_pass_metric_sets),
        pmc_already_covered=pmc_already,
        pmc_placed_into_existing=pmc_into_existing,
        pmc_placed_into_new=pmc_into_new,
    )
    return files, file_count, stats


def _partition_unique_metric_sets(
    soc: "OmniSoC_Base",
    profile_counters: Set[str],
    perfmon_config: Dict[str, int],
) -> Tuple[List[FrozenSet[str]], int, List[FrozenSet[str]], int]:
    """Split metric PMC sets in one YAML walk.

    Returns single-pass sets, that count, multi-pass sets, and that count.
    """
    seen_single: Set[FrozenSet[str]] = set()
    seen_multi: Set[FrozenSet[str]] = set()
    single_sets: List[FrozenSet[str]] = []
    multi_sets: List[FrozenSet[str]] = []
    single_count = 0
    multi_count = 0
    for _sort_key, group, _label in iter_metric_groups(soc, profile_counters):
        if counters_fit_one_bucket(group, perfmon_config):
            single_count += 1
            for candidate in adjust_candidate_groups(group, profile_counters):
                if not counters_fit_one_bucket(candidate, perfmon_config):
                    candidate = group
                if candidate in seen_single:
                    continue
                seen_single.add(candidate)
                single_sets.append(candidate)
            continue
        multi_count += 1
        if group in seen_multi:
            continue
        seen_multi.add(group)
        multi_sets.append(group)
    single_sets.sort(key=lambda metric_set: (-len(metric_set), sorted(metric_set)))
    multi_sets.sort(key=lambda metric_set: (-len(metric_set), sorted(metric_set)))
    return single_sets, single_count, multi_sets, multi_count


def _channel_instance_base(counter: str) -> str:
    """Base name of a counter that carries an instance suffix."""
    return counter.split("[")[0]


def _counters_in_any_bucket(files: List[CounterFile]) -> Set[str]:
    return {ctr for bucket in files for ctr in flat_counters_in_perfmon_file(bucket)}


def _next_bucket_number(files: List[CounterFile], floor: int) -> int:
    """Smallest bucket index at or above floor and above every numeric name."""
    highest_next = max(
        (int(bucket.name) + 1 for bucket in files if bucket.name.isdigit()),
        default=floor,
    )
    return max(floor, highest_next)


def _largest_subset_fitting_bucket(
    bucket: CounterFile,
    remaining: Set[str],
    perfmon_config: Dict[str, int],
) -> Set[str]:
    """Greedy fit, treating all channels of one TCC series as one unit."""
    accepted: Set[str] = set()
    current = bucket_counter_set(bucket)
    units: Dict[str, Set[str]] = {}
    for counter in remaining:
        key = (
            _channel_instance_base(counter)
            if is_tcc_channel_counter(counter)
            else counter
        )
        units.setdefault(key, set()).add(counter)
    for key in sorted(units):
        unit = units[key]
        trial = rebuild_counter_file(
            bucket.name, perfmon_config, current | accepted | unit
        )
        if trial is not None:
            accepted.update(unit)
    return accepted


def _largest_subset_fitting_empty(
    remaining: Set[str],
    perfmon_config: Dict[str, int],
) -> Set[str]:
    """Largest subset of remaining that fits an empty hardware bucket."""
    empty = CounterFile("trial", perfmon_config)
    return _largest_subset_fitting_bucket(empty, remaining, perfmon_config)


def _raise_unplaceable_multi_pass_metric_counters(counters: Set[str]) -> None:
    """Stop residual fill when a counter exceeds a block budget alone."""
    names = ", ".join(sorted(counters))
    raise ValueError(
        "multi-pass metric counter(s) exceed a block budget "
        f"even in an empty bucket: {names}"
    )


def _bucket_has_full_group(bucket: CounterFile, group: FrozenSet[str]) -> bool:
    return set(group) <= bucket_counter_set(bucket)


def _any_bucket_has_full_group(
    files: List[CounterFile],
    group: FrozenSet[str],
) -> bool:
    return any(_bucket_has_full_group(bucket, group) for bucket in files)


def _try_extend_bucket_with_group(
    bucket: CounterFile,
    group: FrozenSet[str],
    perfmon_config: Dict[str, int],
) -> Optional[CounterFile]:
    union = bucket_counter_set(bucket) | set(group)
    return rebuild_counter_file(bucket.name, perfmon_config, union)


def _extend_best_existing_bucket(
    files: List[CounterFile],
    need: Set[str],
    perfmon_config: Dict[str, int],
) -> Optional[Set[str]]:
    """Place the largest fitting subset of need into an existing bucket."""
    best_idx: Optional[int] = None
    best_subset: Set[str] = set()
    for idx, bucket in enumerate(files):
        subset = _largest_subset_fitting_bucket(bucket, need, perfmon_config)
        if len(subset) > len(best_subset):
            best_subset = subset
            best_idx = idx
    if best_idx is None or not best_subset:
        return None
    trial = _try_extend_bucket_with_group(
        files[best_idx], frozenset(best_subset), perfmon_config
    )
    if trial is None:
        return None
    files[best_idx] = trial
    return best_subset


def _ensure_single_pass_metric_set(
    files: List[CounterFile],
    group: FrozenSet[str],
    perfmon_config: Dict[str, int],
    file_count: int,
) -> Tuple[List[CounterFile], int]:
    """Ensure some bucket contains the full group (may duplicate counters)."""
    if _any_bucket_has_full_group(files, group):
        return files, file_count

    best_idx: Optional[int] = None
    best_overlap = -1
    best_trial: Optional[CounterFile] = None
    for idx, bucket in enumerate(files):
        trial = _try_extend_bucket_with_group(bucket, group, perfmon_config)
        if trial is None:
            continue
        overlap = len(group & bucket_counter_set(bucket))
        if overlap > best_overlap:
            best_overlap = overlap
            best_idx = idx
            best_trial = trial
    if best_idx is not None and best_trial is not None:
        updated = list(files)
        updated[best_idx] = best_trial
        return updated, file_count

    new_bucket = rebuild_counter_file(str(file_count), perfmon_config, set(group))
    if new_bucket is None:
        console_warning(
            "profiling",
            "single-pass metric allocation: cannot open a bucket for the set.",
        )
        return files, file_count
    files = list(files)
    files.append(new_bucket)
    return files, file_count + 1


def _add_to_first_bucket_that_fits(
    files: List[CounterFile],
    counter: str,
    tcc_map: Dict[str, CounterFile],
) -> bool:
    for bucket in files:
        if not bucket.add(counter):
            continue
        if is_tcc_channel_counter(counter):
            tcc_map[_channel_instance_base(counter)] = bucket
        return True
    return False


def _first_fit_unplaced(
    files: List[CounterFile],
    work_set: Set[str],
    perfmon_config: Dict[str, int],
    file_count: int,
) -> Tuple[List[CounterFile], int]:
    """First-fit counters that are not present in any bucket yet.

    A counter that fits no bucket, including a fresh empty one, raises
    ValueError. The same situation in multi-pass fill is an error too.
    """
    placed = _counters_in_any_bucket(files)
    leftovers = sorted(work_set - placed)
    files = list(files)
    tcc_map: Dict[str, CounterFile] = {}
    for bucket in files:
        for ctr in flat_counters_in_perfmon_file(bucket):
            if is_tcc_channel_counter(ctr):
                tcc_map[_channel_instance_base(ctr)] = bucket

    for ctr in leftovers:
        if is_tcc_channel_counter(ctr):
            existing = tcc_map.get(_channel_instance_base(ctr))
            if existing is not None and existing.add(ctr):
                continue

        if _add_to_first_bucket_that_fits(files, ctr, tcc_map):
            continue

        bucket = CounterFile(str(file_count), perfmon_config)
        file_count += 1
        if not bucket.add(ctr):
            raise ValueError(
                "single-pass metric counter(s) exceed a block budget "
                f"even in an empty bucket: {ctr}"
            )
        files.append(bucket)
        if is_tcc_channel_counter(ctr):
            tcc_map[_channel_instance_base(ctr)] = bucket
    return files, file_count


def _count_single_pass_metric_sets_without_pass(
    files: List[CounterFile],
    metric_sets: List[FrozenSet[str]],
) -> int:
    """Count single-pass metric sets with no bucket containing the full PMC set."""
    return sum(
        1 for group in metric_sets if not _any_bucket_has_full_group(files, group)
    )


def _try_merge_bucket_indices(
    files: List[CounterFile],
    indices: Set[int],
    perfmon_config: Dict[str, int],
) -> Optional[List[CounterFile]]:
    if len(indices) < 2:
        return None
    ordered = sorted(indices)
    target = ordered[0]
    union: Set[str] = set()
    for idx in ordered:
        union |= bucket_counter_set(files[idx])
    merged = rebuild_counter_file(files[target].name, perfmon_config, union)
    if merged is None:
        return None
    updated: List[CounterFile] = []
    for idx, bucket in enumerate(files):
        if idx == target:
            updated.append(merged)
        elif idx in indices:
            continue
        else:
            updated.append(bucket)
    return updated


def _first_coverage_preserving_merge(
    files: List[CounterFile],
    unions: List[FrozenSet[str]],
    perfmon_config: Dict[str, int],
) -> Optional[List[CounterFile]]:
    """First pair whose merge keeps every single-pass metric set together."""
    for left, right in combinations(range(len(files)), 2):
        trial = _try_merge_bucket_indices(files, {left, right}, perfmon_config)
        if trial is None:
            continue
        if _count_single_pass_metric_sets_without_pass(trial, unions) > 0:
            continue
        return trial
    return None


def _reduce_passes(
    files: List[CounterFile],
    metric_sets: List[FrozenSet[str]],
    perfmon_config: Dict[str, int],
) -> Tuple[List[CounterFile], int]:
    """Merge bucket pairs while every single-pass metric set stays together."""
    merges = 0
    while True:
        trial = _first_coverage_preserving_merge(files, metric_sets, perfmon_config)
        if trial is None:
            return files, merges
        files = trial
        merges += 1


@dataclass(frozen=True)
class SinglePassMetricStats:
    """Outcome of the single-pass metric allocator."""

    bucket_count: int
    single_pass_metric_count: int
    unique_single_pass_metric_sets: int
    single_pass_metric_sets_without_pass: int
    merges_applied: int
    multi_pass_metric_count: int = 0
    unique_multi_pass_metric_sets: int = 0
    multi_pass_additional_passes: int = 0


@dataclass(frozen=True)
class MultiPassMetricFillStats:
    """Pass impact of filling multi-pass metric PMCs into an existing layout."""

    passes_before: int
    passes_after: int
    additional_passes: int
    multi_pass_metric_count: int
    unique_multi_pass_metric_sets: int
    pmc_already_covered: int
    pmc_placed_into_existing: int
    pmc_placed_into_new: int
