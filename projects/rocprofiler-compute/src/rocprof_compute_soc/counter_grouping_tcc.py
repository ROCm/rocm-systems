# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Short-term TCC channel rules for single-pass packing.

This module is the only place that knows which TCC series are an affinity
pair, which multi-column request row stays one group, and which extra
request copies a later cleanup may drop. It is isolated so a later
hardware-instance design can replace it. See the LLD section
"Instance-aware metrics (out of scope)".

Single-die chips are a data set. Their L2 channels are the chip's own
banks. Every other arch is multi-die, and a template expands to one copy
of those banks per die the caller reports. This module does not invent a
die count and does not borrow one arch's count for another.

The TCC event-base budget stays the general hardware-block check in
CounterFile.add. This module does not own that budget.
"""

from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from utils.logger import console_debug, console_warning
from utils.utils_common import is_tcc_channel_counter

from .counter_file import CounterFile, flat_counters_in_perfmon_file
from .counter_grouping_buckets import bucket_counter_set, rebuild_counter_file

# LEVEL series -> matching request series (same-pass affinity).
# TCC_EA0_* is the MI100 / MI300 / MI350 name. TCC_EA_* is the MI200 name.
_LEVEL_TO_REQUEST: Dict[str, str] = {
    "TCC_EA0_RDREQ_LEVEL": "TCC_EA0_RDREQ",
    "TCC_EA0_WRREQ_LEVEL": "TCC_EA0_WRREQ",
    "TCC_EA0_ATOMIC_LEVEL": "TCC_EA0_ATOMIC",
    "TCC_EA_RDREQ_LEVEL": "TCC_EA_RDREQ",
    "TCC_EA_WRREQ_LEVEL": "TCC_EA_WRREQ",
    "TCC_EA_ATOMIC_LEVEL": "TCC_EA_ATOMIC",
}
_REQUEST_TO_LEVEL: Dict[str, str] = {
    request: level for level, request in _LEVEL_TO_REQUEST.items()
}

# Chips whose L2/TCC channels are the die itself. gfx908 is MI100.
# gfx1150-gfx1153 are the RDNA 3.5 parts that share analysis_configs/gfx115x.
# gfx115x is that shared config name. Multi-die archs are absent here on
# purpose: gfx90a, gfx940, gfx941, gfx942, gfx950, and gfx1250.
_SINGLE_DIE_ARCHS: FrozenSet[str] = frozenset({
    "gfx908",
    "gfx1150",
    "gfx1151",
    "gfx1152",
    "gfx1153",
    "gfx115x",
})


def collectable_tcc_channel_count(
    arch: str,
    banks_per_die: int,
    die_count: int,
) -> int:
    """Return how many channel indexes a TCC template expands to.

    Single-die archs use banks_per_die and ignore die_count. Multi-die
    archs use banks_per_die once per reported die. A non-positive die
    count is one die. banks_per_die comes from that arch's own L2 banks.
    """
    banks = banks_per_die if banks_per_die > 0 else 0
    if arch in _SINGLE_DIE_ARCHS:
        return banks
    dies = die_count if die_count > 0 else 1
    return banks * dies


def tcc_channel_definition(
    arch: str,
    counter_name: str,
    index: int,
    banks_per_die: int,
) -> Tuple[str, str]:
    """Return the description and select() expression for one channel.

    Single-die archs address the channel index directly. Multi-die archs
    split the index into a die and the channel on that die, using this
    arch's banks_per_die. The split is the per-die map, not another
    arch's channel count.

    Returns:
        Description text, then the select() expression.
    """
    if arch in _SINGLE_DIE_ARCHS:
        description = f"{counter_name} on channel {index}"
        expression = f"select({counter_name},[DIMENSION_INSTANCE=[{index}]])"
        return description, expression

    banks = banks_per_die if banks_per_die > 0 else 1
    die_index = index // banks
    channel_index = index % banks
    description = f"{counter_name} on {die_index}th XCC and {channel_index}th channel"
    expression = (
        f"select({counter_name},"
        f"[DIMENSION_XCC=[{die_index}], "
        f"DIMENSION_INSTANCE=[{channel_index}]])"
    )
    return description, expression


def adjust_candidate_groups(
    group: FrozenSet[str],
    profile_counters: Set[str],
) -> List[FrozenSet[str]]:
    """Return candidate groups after the short-term TCC rules.

    A multi-column request row with no LEVEL stays one group. That row is
    panel 1805: the read, write, and atomic request series. A LEVEL series
    stays in the group it arrived in and gains matching request channels
    from the profile. The caller keeps the original group when an expanded
    one does not fit one bucket.
    """
    return [_expand_affinity_partners(group, profile_counters)]


def keep_copies_groups_still_need(
    files: List[CounterFile],
    perfmon_config: Dict[str, int],
    required_groups: Optional[List[FrozenSet[str]]] = None,
) -> List[CounterFile]:
    """Drop a request series from a pass that lacks its LEVEL.

    Leave the series in the pass that also holds the matching LEVEL. Keep
    extra request copies when a multi-column request row needs them to stay
    complete in one pass. Same-pass bind still selects the pass that holds
    the whole expression, so a latency row does not read a copy that lacks
    its LEVEL. Per-channel series only; sum aggregates are left alone.
    """
    if not files:
        return files

    home_bases: Set[str] = set()
    for bucket in files:
        bases = _channel_bases_in_bucket(bucket)
        for request_base, level_base in _REQUEST_TO_LEVEL.items():
            if request_base in bases and level_base in bases:
                home_bases.add(request_base)

    if not home_bases:
        return files

    updated = list(files)
    changed = False
    for index, bucket in enumerate(files):
        bases = _channel_bases_in_bucket(bucket)
        drop_bases = {
            request_base
            for request_base in home_bases
            if request_base in bases and _REQUEST_TO_LEVEL[request_base] not in bases
        }
        if not drop_bases:
            continue
        kept = {
            counter
            for counter in flat_counters_in_perfmon_file(bucket)
            if not (
                is_tcc_channel_counter(counter) and _channel_base(counter) in drop_bases
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
        if required_groups and _required_group_missing(trial, required_groups):
            console_debug(
                "profiling",
                "single-pass-packable: kept EA request copies so a "
                "multi-column request row stays complete in one pass.",
            )
            continue
        updated = trial
        changed = True
        if not kept:
            # Replacements keep updated aligned with files. A pop is followed
            # immediately by recursion so subsequent indices cannot refer to
            # the shortened list.
            return keep_copies_groups_still_need(
                updated,
                perfmon_config,
                required_groups,
            )

    if changed:
        console_debug(
            "profiling",
            "single-pass-packable: stripped orphan TCC EA REQ series from "
            f"non-home passes ({sorted(home_bases)}).",
        )
    return updated


def _expand_affinity_partners(
    group: FrozenSet[str],
    profile_counters: Set[str],
) -> FrozenSet[str]:
    """Add matching request channels when a LEVEL series is in the group."""
    bases = {
        _channel_base(counter) for counter in group if is_tcc_channel_counter(counter)
    }
    expanded = set(group)
    for level_base, request_base in _LEVEL_TO_REQUEST.items():
        if level_base not in bases:
            continue
        for counter in profile_counters:
            if not is_tcc_channel_counter(counter):
                continue
            if _channel_base(counter) == request_base:
                expanded.add(counter)
    return frozenset(expanded)


def _channel_base(counter: str) -> str:
    return counter.split("[")[0]


def _channel_bases_in_bucket(bucket: CounterFile) -> Set[str]:
    return {
        _channel_base(counter)
        for counter in flat_counters_in_perfmon_file(bucket)
        if is_tcc_channel_counter(counter)
    }


def _required_group_missing(
    files: List[CounterFile],
    required_groups: List[FrozenSet[str]],
) -> bool:
    for group in required_groups:
        if any(set(group) <= bucket_counter_set(bucket) for bucket in files):
            continue
        return True
    return False
