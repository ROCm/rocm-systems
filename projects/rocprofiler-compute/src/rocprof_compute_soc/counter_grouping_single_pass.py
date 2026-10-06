# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Default allocator: single-bucket guarantee for every packable metric.

Every metric whose PMC set fits one CounterFile (single-pass packable, not
SPU) has some perfmon bucket containing its full PMC set. The allocator
minimizes the number of passes under that hard constraint. Counters not
required by any packable union use ordinary first-fit. Remaining SPU PMCs
are placed into existing buckets when possible, and new passes are opened
only if needed. TCC series affinity keeps LEVEL with matching REQ/ATOMIC in
the same pass, covers every selected series, and never emits orphan REQ
duplicates across channel passes (see AIPROFCOMP-865 design).

Overlapping packable unions that cannot share one bucket are handled by
duplicating counters into an additional bucket (additive passes). That
differs from the legacy heuristic, which places each counter in at most one
bucket. Affinity forbids duplicating the same EA REQ series into a second
pass when a LEVEL+REQ home already exists.

Disable with ROCPROF_COMPUTE_PERFMON_LEGACY_HEURISTIC=1 (or
ROCPROF_COMPUTE_PERFMON_SINGLE_PASS_PACKABLE=0) to restore the priority
coalesce + first-fit path.
"""

import os
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

if TYPE_CHECKING:
    from .soc_base import OmniSoC_Base

# LEVEL base -> matching REQ/ATOMIC denominator base (hard co-residence).
_TCC_EA_LEVEL_TO_REQ: Dict[str, str] = {
    "TCC_EA0_RDREQ_LEVEL": "TCC_EA0_RDREQ",
    "TCC_EA0_WRREQ_LEVEL": "TCC_EA0_WRREQ",
    "TCC_EA0_ATOMIC_LEVEL": "TCC_EA0_ATOMIC",
}
_TCC_EA_REQ_TO_LEVEL: Dict[str, str] = {
    req: level for level, req in _TCC_EA_LEVEL_TO_REQ.items()
}
# Independent EA count columns (panel 1805); split so they do not force
# orphan REQ copies into the ATOMIC+LEVEL pass.
_TCC_EA_INDEPENDENT_REQ_BASES: FrozenSet[str] = frozenset(_TCC_EA_REQ_TO_LEVEL)


def try_allocate_single_pass_packable(
    soc: "OmniSoC_Base",
    work_set: Set[str],
    perfmon_config: Dict[str, int],
    file_count_start: int = 0,
) -> Optional[Tuple[List[CounterFile], int, "SinglePassPackableStats"]]:
    """Allocate work_set with packable metrics forced into one bucket.

    On success, clears work_set. The caller checks
    single_pass_packable_enabled_from_env and skips this function when the
    legacy heuristic is selected.

    Returns:
        (files, file_count, stats) when every packable union has a full bucket.
        None when work_set is empty, when packable unions still lack a full
        bucket after allocate, or when residual SPU fill drops that coverage.
    """
    if not work_set:
        return None

    unions, packable_metric_count = collect_unique_packable_unions(
        soc, work_set, perfmon_config
    )
    files: List[CounterFile] = []
    file_count = file_count_start
    for group in unions:
        files, file_count = _ensure_packable_union(
            files, group, perfmon_config, file_count
        )

    files, file_count = _first_fit_unplaced(files, work_set, perfmon_config, file_count)
    files, merges = _reduce_passes(files, unions, perfmon_config)
    files = _strip_orphan_tcc_ea_req_duplicates(files, perfmon_config, unions)

    packable_multi = _count_packable_multi(files, unions)
    if packable_multi > 0:
        console_warning(
            "profiling",
            "single-pass-packable: "
            f"{packable_multi} packable union(s) still lack a full bucket "
            "after allocate; falling back to legacy heuristic.",
        )
        return None

    spu_unions, spu_metric_count = collect_unique_slot_limit_unions(
        soc, work_set, perfmon_config
    )
    files, file_count, spu_fill_stats = fill_slot_limit_into_existing_passes(
        files,
        spu_unions,
        perfmon_config,
        slot_limit_metric_count=spu_metric_count,
        file_count_start=file_count,
    )
    files = _strip_orphan_tcc_ea_req_duplicates(files, perfmon_config, unions)
    packable_multi = _count_packable_multi(files, unions)
    if packable_multi > 0:
        console_warning(
            "profiling",
            "single-pass-packable: "
            f"{packable_multi} packable union(s) lost coverage after residual "
            "fill; falling back to legacy heuristic.",
        )
        return None
    file_count = _next_bucket_number(files, file_count_start)

    stats = SinglePassPackableStats(
        bucket_count=len(files),
        packable_metric_count=packable_metric_count,
        unique_packable_unions=len(unions),
        packable_multi_after=packable_multi,
        merges_applied=merges,
        slot_limit_metrics=spu_fill_stats.slot_limit_metrics,
        unique_slot_limit_unions=spu_fill_stats.unique_slot_limit_unions,
        slot_additional_passes=spu_fill_stats.additional_passes,
    )
    console_debug(
        "profiling",
        "single-pass-packable: "
        f"{stats.bucket_count} bucket(s), "
        f"{stats.packable_metric_count} packable metric(s), "
        f"{stats.unique_packable_unions} unique union(s), "
        f"packable_multi={stats.packable_multi_after}, "
        f"merges={stats.merges_applied}, "
        f"slot_limit={stats.slot_limit_metrics}, "
        f"slot_+passes={stats.slot_additional_passes}.",
    )

    work_set.clear()
    return files, file_count, stats


def single_pass_packable_enabled_from_env() -> bool:
    """Return True when single-pass-packable is the active allocate path.

    Default is on. Set ROCPROF_COMPUTE_PERFMON_LEGACY_HEURISTIC=1 to use the
    legacy heuristic. ROCPROF_COMPUTE_PERFMON_SINGLE_PASS_PACKABLE=0 also
    disables SPP for explicit A/B during migration.
    """
    if legacy_heuristic_enabled_from_env():
        return False
    raw = (
        os.environ
        .get("ROCPROF_COMPUTE_PERFMON_SINGLE_PASS_PACKABLE", "1")
        .strip()
        .lower()
    )
    return raw not in {"0", "false", "no", "off"}


def legacy_heuristic_enabled_from_env() -> bool:
    """Return True when the legacy coalesce / first-fit path is forced."""
    raw = os.environ.get("ROCPROF_COMPUTE_PERFMON_LEGACY_HEURISTIC", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def collect_unique_packable_unions(
    soc: "OmniSoC_Base",
    profile_counters: Set[str],
    perfmon_config: Dict[str, int],
) -> Tuple[List[FrozenSet[str]], int]:
    """Unique PMC sets for metrics that fit one hardware bucket.

    Multi-column EA REQ tables (for example panel 1805) are split into
    per-series unions so affinity can keep LEVEL+REQ co-resident without
    orphan REQ duplicates.

    Returns:
        Unions sorted largest-first, and the packable metric count.
    """
    seen: Set[FrozenSet[str]] = set()
    unions: List[FrozenSet[str]] = []
    packable_metric_count = 0
    for _sort_key, group, _label in iter_metric_groups(soc, profile_counters):
        if not counters_fit_one_bucket(group, perfmon_config):
            continue
        packable_metric_count += 1
        for subgroup in _split_independent_tcc_ea_req_union(group):
            affinity_group = _expand_tcc_ea_affinity_partners(
                subgroup, profile_counters
            )
            if not counters_fit_one_bucket(affinity_group, perfmon_config):
                affinity_group = subgroup
            if affinity_group in seen:
                continue
            seen.add(affinity_group)
            unions.append(affinity_group)
    unions.sort(key=lambda g: (-len(g), sorted(g)))
    return unions, packable_metric_count


def collect_unique_slot_limit_unions(
    soc: "OmniSoC_Base",
    profile_counters: Set[str],
    perfmon_config: Dict[str, int],
) -> Tuple[List[FrozenSet[str]], int]:
    """Unique PMC sets for metrics that cannot fit one hardware bucket.

    Returns:
        Unions sorted largest-first, and the SPU metric count.
    """
    seen: Set[FrozenSet[str]] = set()
    unions: List[FrozenSet[str]] = []
    slot_limit_metric_count = 0
    for _sort_key, group, _label in iter_metric_groups(soc, profile_counters):
        if counters_fit_one_bucket(group, perfmon_config):
            continue
        slot_limit_metric_count += 1
        if group in seen:
            continue
        seen.add(group)
        unions.append(group)
    unions.sort(key=lambda g: (-len(g), sorted(g)))
    return unions, slot_limit_metric_count


def fill_slot_limit_into_existing_passes(
    files: List[CounterFile],
    slot_limit_unions: List[FrozenSet[str]],
    perfmon_config: Dict[str, int],
    *,
    slot_limit_metric_count: int = 0,
    file_count_start: Optional[int] = None,
) -> Tuple[List[CounterFile], int, "SlotLimitFillStats"]:
    """Place SPU PMCs into existing buckets; open new ones only if needed.

    For each unique SPU PMC set (the full set cannot fit one bucket):
    repeatedly pack the largest remaining subset into the best existing
    bucket, else open a new bucket with the largest empty-bucket-fitting
    subset.

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

    for group in slot_limit_unions:
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
                console_warning(
                    "profiling",
                    "single-pass-packable: SPU PMC cannot fit any bucket.",
                )
                break
            new_bucket = rebuild_counter_file(
                str(file_count), perfmon_config, set(subset)
            )
            if new_bucket is None:
                break
            files.append(new_bucket)
            file_count += 1
            pmc_into_new += len(subset)
            need -= subset

    passes_after = len(files)
    stats = SlotLimitFillStats(
        passes_before=passes_before,
        passes_after=passes_after,
        additional_passes=passes_after - passes_before,
        slot_limit_metrics=slot_limit_metric_count,
        unique_slot_limit_unions=len(slot_limit_unions),
        pmc_already_covered=pmc_already,
        pmc_placed_into_existing=pmc_into_existing,
        pmc_placed_into_new=pmc_into_new,
    )
    return files, file_count, stats


def _tcc_channel_base(counter: str) -> str:
    return counter.split("[")[0]


def _bucket_tcc_channel_bases(bucket: CounterFile) -> Set[str]:
    return {
        _tcc_channel_base(ctr)
        for ctr in flat_counters_in_perfmon_file(bucket)
        if is_tcc_channel_counter(ctr)
    }


def _counters_in_any_bucket(files: List[CounterFile]) -> Set[str]:
    return {ctr for bucket in files for ctr in flat_counters_in_perfmon_file(bucket)}


def _next_bucket_number(files: List[CounterFile], floor: int) -> int:
    """Smallest bucket index at or above floor and above every numeric name."""
    highest_next = max(
        (int(bucket.name) + 1 for bucket in files if bucket.name.isdigit()),
        default=floor,
    )
    return max(floor, highest_next)


def _split_independent_tcc_ea_req_union(
    group: FrozenSet[str],
) -> List[FrozenSet[str]]:
    """Split multi-column EA REQ unions (e.g. panel 1805) into per-series groups.

    Independent count columns must not force REQ into a LEVEL-home pass for a
    different series (orphan REQ dups). Each series is covered alone.
    """
    bases_in_group = {
        _tcc_channel_base(ctr) for ctr in group if is_tcc_channel_counter(ctr)
    }
    if bases_in_group & set(_TCC_EA_LEVEL_TO_REQ):
        return [group]
    independent = bases_in_group & _TCC_EA_INDEPENDENT_REQ_BASES
    if len(independent) < 2:
        return [group]

    subgroups: List[FrozenSet[str]] = []
    remaining = set(group)
    for base in sorted(independent):
        series = frozenset(
            ctr
            for ctr in group
            if is_tcc_channel_counter(ctr) and _tcc_channel_base(ctr) == base
        )
        subgroups.append(series)
        remaining -= series
    if remaining:
        subgroups.append(frozenset(remaining))
    return subgroups


def _expand_tcc_ea_affinity_partners(
    group: FrozenSet[str],
    profile_counters: Set[str],
) -> FrozenSet[str]:
    """Ensure LEVEL bases keep matching REQ/ATOMIC channels from the profile."""
    bases = {_tcc_channel_base(ctr) for ctr in group if is_tcc_channel_counter(ctr)}
    expanded = set(group)
    for level_base, req_base in _TCC_EA_LEVEL_TO_REQ.items():
        if level_base not in bases:
            continue
        for ctr in profile_counters:
            if not is_tcc_channel_counter(ctr):
                continue
            if _tcc_channel_base(ctr) == req_base:
                expanded.add(ctr)
    return frozenset(expanded)


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
        key = _tcc_channel_base(counter) if is_tcc_channel_counter(counter) else counter
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


def _ensure_packable_union(
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
            "single-pass-packable: cannot open bucket for a packable union.",
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
            tcc_map[_tcc_channel_base(counter)] = bucket
        return True
    return False


def _first_fit_unplaced(
    files: List[CounterFile],
    work_set: Set[str],
    perfmon_config: Dict[str, int],
    file_count: int,
) -> Tuple[List[CounterFile], int]:
    """First-fit counters that are not present in any bucket yet."""
    placed = _counters_in_any_bucket(files)
    leftovers = sorted(work_set - placed)
    files = list(files)
    tcc_map: Dict[str, CounterFile] = {}
    for bucket in files:
        for ctr in flat_counters_in_perfmon_file(bucket):
            if is_tcc_channel_counter(ctr):
                tcc_map[_tcc_channel_base(ctr)] = bucket

    for ctr in leftovers:
        if is_tcc_channel_counter(ctr):
            existing = tcc_map.get(_tcc_channel_base(ctr))
            if existing is not None and existing.add(ctr):
                continue

        if _add_to_first_bucket_that_fits(files, ctr, tcc_map):
            continue

        bucket = CounterFile(str(file_count), perfmon_config)
        file_count += 1
        if not bucket.add(ctr):
            console_warning(
                "profiling",
                f"single-pass-packable: leftover {ctr!r} rejected by new bucket.",
            )
            continue
        files.append(bucket)
        if is_tcc_channel_counter(ctr):
            tcc_map[_tcc_channel_base(ctr)] = bucket
    return files, file_count


def _count_packable_multi(
    files: List[CounterFile],
    unions: List[FrozenSet[str]],
) -> int:
    """Count packable unions with no bucket containing the full PMC set."""
    return sum(1 for group in unions if not _any_bucket_has_full_group(files, group))


def _strip_orphan_tcc_ea_req_duplicates(
    files: List[CounterFile],
    perfmon_config: Dict[str, int],
    required_unions: Optional[List[FrozenSet[str]]] = None,
) -> List[CounterFile]:
    """Remove EA REQ series from non-home passes when a LEVEL+REQ home exists.

    Ban placing the same per-channel EA REQ family in two passes with
    divergent channel maps (AIPROFCOMP-865 section 1.4). *_sum aggregates
    are untouched (not channel series).
    """
    if not files:
        return files

    home_bases: Set[str] = set()
    for bucket in files:
        bases = _bucket_tcc_channel_bases(bucket)
        for req_base, level_base in _TCC_EA_REQ_TO_LEVEL.items():
            if req_base in bases and level_base in bases:
                home_bases.add(req_base)

    if not home_bases:
        return files

    updated = list(files)
    changed = False
    for index, bucket in enumerate(files):
        bases = _bucket_tcc_channel_bases(bucket)
        drop_bases = {
            req_base
            for req_base in home_bases
            if req_base in bases and _TCC_EA_REQ_TO_LEVEL[req_base] not in bases
        }
        if not drop_bases:
            continue
        kept = {
            ctr
            for ctr in flat_counters_in_perfmon_file(bucket)
            if not (
                is_tcc_channel_counter(ctr) and _tcc_channel_base(ctr) in drop_bases
            )
        }
        rebuilt = rebuild_counter_file(bucket.name, perfmon_config, kept)
        if rebuilt is None:
            console_warning(
                "profiling",
                "single-pass-packable: orphan TCC EA REQ strip rebuild failed "
                f"for bucket {bucket.name!r}; leaving bucket unchanged.",
            )
            continue
        trial = list(updated)
        if kept:
            trial[index] = rebuilt
        else:
            trial.pop(index)
        if required_unions and _count_packable_multi(trial, required_unions) > 0:
            console_debug(
                "profiling",
                "single-pass-packable: kept orphan TCC EA REQ series because "
                "stripping it would break SPP coverage.",
            )
            continue
        updated = trial
        changed = True
        if not kept:
            # Replacements keep updated aligned with files. A pop is followed
            # immediately by recursion so subsequent indices cannot refer to
            # the shortened list.
            return _strip_orphan_tcc_ea_req_duplicates(
                updated,
                perfmon_config,
                required_unions,
            )

    if changed:
        console_debug(
            "profiling",
            "single-pass-packable: stripped orphan TCC EA REQ series from "
            f"non-home passes ({sorted(home_bases)}).",
        )
    return updated


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
    """First pair, in combinations order, whose merge keeps packable coverage."""
    for left, right in combinations(range(len(files)), 2):
        trial = _try_merge_bucket_indices(files, {left, right}, perfmon_config)
        if trial is None:
            continue
        if _count_packable_multi(trial, unions) > 0:
            continue
        return trial
    return None


def _reduce_passes(
    files: List[CounterFile],
    unions: List[FrozenSet[str]],
    perfmon_config: Dict[str, int],
) -> Tuple[List[CounterFile], int]:
    """Merge bucket pairs when the union fits and packable coverage stays intact."""
    merges = 0
    while True:
        trial = _first_coverage_preserving_merge(files, unions, perfmon_config)
        if trial is None:
            return files, merges
        files = trial
        merges += 1


@dataclass(frozen=True)
class SinglePassPackableStats:
    """Outcome of the single-pass-packable allocator."""

    bucket_count: int
    packable_metric_count: int
    unique_packable_unions: int
    packable_multi_after: int
    merges_applied: int
    slot_limit_metrics: int = 0
    unique_slot_limit_unions: int = 0
    slot_additional_passes: int = 0


@dataclass(frozen=True)
class SlotLimitFillStats:
    """Pass impact of filling SPU PMCs into an existing layout."""

    passes_before: int
    passes_after: int
    additional_passes: int
    slot_limit_metrics: int
    unique_slot_limit_unions: int
    pmc_already_covered: int
    pmc_placed_into_existing: int
    pmc_placed_into_new: int
