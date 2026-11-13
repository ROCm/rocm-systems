# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Pack perfmon counters so that a metric whose counters fit one pass is
collected in one pass."""

from typing import Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

from utils.counters.counter_file import CounterFile, flat_counters_in_perfmon_file
from utils.logger import console_debug, console_error, console_warning
from utils.utils_counter_defs import extract_counters_and_variables


def metric_counter_sets(
    metric_expressions: Iterable[List[str]],
    gpu_series: str,
    profile_counters: Set[str],
) -> List[FrozenSet[str]]:
    """Return, for each metric, the profiled counters its expressions read.

    A TCC channel name such as TCC_HIT[ stands for every profiled channel of
    that counter. A metric that reads no profiled counter is left out.
    """
    counter_sets = []
    for expressions in metric_expressions:
        formula_counters, _ = extract_counters_and_variables(
            "\n".join(expressions), gpu_series, include_supported_denom=False
        )
        channel_prefixes = tuple(
            name for name in formula_counters if name.endswith("[")
        )
        counters = frozenset(
            counter
            for counter in profile_counters
            if counter in formula_counters or counter.startswith(channel_prefixes)
        )
        if counters:
            counter_sets.append(counters)
    return counter_sets


def allocate_single_pass_metrics(
    counters: Set[str],
    metric_counter_sets: List[FrozenSet[str]],
    perfmon_config: Dict[str, int],
) -> Optional[List[CounterFile]]:
    """Pack counters into perfmon passes.

    Each metric whose counters fit one empty pass gets a pass that holds all
    of them, and a counter is copied into more passes when needed. The other
    counters go into the first pass with room. Returns None if a metric that
    fits one pass ends up without such a pass.
    """
    single_pass_sets = _single_pass_sets(metric_counter_sets, perfmon_config)
    files: List[CounterFile] = []
    for metric_set in single_pass_sets:
        files = _place_together(files, metric_set, perfmon_config)
    for counter in sorted(counters - _counters_in(files)):
        files = _place_first_fit(files, counter, perfmon_config)

    if not all(_held(files, metric_set) for metric_set in single_pass_sets):
        console_warning(
            "profiling",
            "Single-pass packing left a metric without a pass that holds all of "
            "its counters.",
        )
        return None
    for index, counter_file in enumerate(files):
        counter_file.name = str(index)
    console_debug(
        "profiling",
        f"Single-pass packing: {len(files)} passes for {len(single_pass_sets)} "
        "single-pass metric counter sets",
    )
    return files


def _single_pass_sets(
    metric_counter_sets: List[FrozenSet[str]],
    perfmon_config: Dict[str, int],
) -> List[FrozenSet[str]]:
    """Return the unique metric counter sets that fit one empty pass, largest
    first."""
    empty = CounterFile("empty", perfmon_config)
    return sorted(
        {
            metric_set
            for metric_set in metric_counter_sets
            if _with_counters(empty, metric_set, perfmon_config) is not None
        },
        key=lambda metric_set: (-len(metric_set), sorted(metric_set)),
    )


def _with_counters(
    counter_file: CounterFile,
    counters: Iterable[str],
    perfmon_config: Dict[str, int],
) -> Optional[CounterFile]:
    """Return a copy of counter_file that also holds counters, or None when
    they do not fit."""
    extended = CounterFile(counter_file.name, perfmon_config)
    for counter in sorted(
        set(flat_counters_in_perfmon_file(counter_file)) | set(counters)
    ):
        if not extended.add(counter):
            return None
    return extended


def _counters_in(files: List[CounterFile]) -> Set[str]:
    return {
        counter
        for counter_file in files
        for counter in flat_counters_in_perfmon_file(counter_file)
    }


def _held(files: List[CounterFile], metric_set: FrozenSet[str]) -> bool:
    """Return whether one pass holds every counter of metric_set."""
    return any(
        metric_set <= set(flat_counters_in_perfmon_file(counter_file))
        for counter_file in files
    )


def _place_together(
    files: List[CounterFile],
    metric_set: FrozenSet[str],
    perfmon_config: Dict[str, int],
) -> List[CounterFile]:
    """Make one pass hold all of metric_set.

    Extend the pass that already holds the most of it, the earliest on a tie,
    or open a new pass.
    """
    if _held(files, metric_set):
        return files
    best: Optional[Tuple[int, int, CounterFile]] = None
    for index, counter_file in enumerate(files):
        extended = _with_counters(counter_file, metric_set, perfmon_config)
        overlap = len(metric_set & set(flat_counters_in_perfmon_file(counter_file)))
        if extended is not None and (best is None or overlap > best[0]):
            best = (overlap, index, extended)
    if best is None:
        new_file = CounterFile(str(len(files)), perfmon_config)
        return [*files, _with_counters(new_file, metric_set, perfmon_config)]
    _overlap, index, extended = best
    return [*files[:index], extended, *files[index + 1 :]]


def _place_first_fit(
    files: List[CounterFile],
    counter: str,
    perfmon_config: Dict[str, int],
) -> List[CounterFile]:
    """Add counter to the first pass with room for it, or to a new pass."""
    for index, counter_file in enumerate(files):
        extended = _with_counters(counter_file, [counter], perfmon_config)
        if extended is not None:
            return [*files[:index], extended, *files[index + 1 :]]
    new_file = _with_counters(
        CounterFile(str(len(files)), perfmon_config), [counter], perfmon_config
    )
    if new_file is None:
        console_error("profiling", f"{counter} does not fit an empty perfmon pass.")
    return [*files, new_file]
