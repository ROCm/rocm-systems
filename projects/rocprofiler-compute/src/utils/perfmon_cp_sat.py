# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
CP-SAT formulation for a restricted perfmon bucket problem.

Requires optional dependency ``ortools``. Used when environment variable
``ROCPROF_COMPUTE_PERFMON_CP_SAT=1`` is set (see ``OmniSoC_Base`` allocator).

**Model.** Items are PMC counter names. Each item uses one unit of capacity in
its IP block (same block mapping as ``CounterFile``). Bins have per-block caps
from ``perfmon_config``. Optional **same-bin groups** (e.g. priority metrics)
force all counters in a group into a single bin.

**Limitations.** TCC channel counters are not supported (conservative linear
capacity does not match ``LimitedSet`` channel sharing). Large instances are
rejected via ``_MAX_ITEMS``.

**Metric spread.** Optional ``metric_spread_index_groups`` + ``metric_spread_penalty``
add a term ``sum_{m,k} u_{m,k}`` (bins touched by each metric) so the solver
trades off fewer multi-bucket metrics vs raw bin count (see env in
``OmniSoC_Base._try_cp_sat_pmc_perf_buckets``).
"""

from __future__ import annotations

_MAX_ITEMS = 256
_MAX_BINS_CAP = 48
_MAX_METRIC_SPREAD_GROUPS = 240
_DEFAULT_TIME_LIMIT_S = 15.0
_DEFAULT_BIN_USED_WEIGHT = 1


def counter_ip_block_for_perfmon(counter: str) -> str:
    """IP bucket key matching ``CounterFile.add`` (SQ/SQC/SP merge)."""
    block = counter.split("_", 1)[0]
    if block == "SQC":
        return "SQ"
    if block == "SP":
        return "SQ"
    return block


def _items_fit_config(
    items: list[str],
    perfmon_config: dict[str, int],
) -> bool:
    for c in items:
        block = counter_ip_block_for_perfmon(c)
        if block not in perfmon_config:
            return False
    return True


def _same_bin_group_constraints_feasible(
    items: list[str],
    perfmon_config: dict[str, int],
    groups: list[frozenset[str]],
) -> bool:
    """Quick rejection: any group needs more of one block than a single bin allows."""
    item_set = set(items)
    for grp in groups:
        inter = [c for c in grp if c in item_set]
        if len(inter) < 2:
            continue
        per_block: dict[str, int] = {}
        for c in inter:
            b = counter_ip_block_for_perfmon(c)
            per_block[b] = per_block.get(b, 0) + 1
        for block_name, need in per_block.items():
            cap = perfmon_config.get(block_name, 0)
            if need > cap:
                return False
    return True


def cp_sat_partition_counters(
    items_sorted: list[str],
    perfmon_config: dict[str, int],
    same_bin_counter_groups: list[frozenset[str]],
    *,
    max_bins: int | None = None,
    time_limit_s: float = _DEFAULT_TIME_LIMIT_S,
    metric_spread_index_groups: list[list[int]] | None = None,
    metric_spread_penalty: int = 0,
    bin_used_weight: int = _DEFAULT_BIN_USED_WEIGHT,
) -> list[list[str]] | None:
    """
    Partition ``items_sorted`` under vector capacities.

    Each counter in ``same_bin_counter_groups`` that intersects ``items_sorted``
    must lie entirely in one bin (per group). Groups of size 0 or 1 are ignored.

    If ``metric_spread_penalty > 0`` and ``metric_spread_index_groups`` is set,
    minimize ``metric_spread_penalty * sum u_{m,k} + bin_used_weight * sum bin_used``
    where ``u_{m,k}=1`` iff metric ``m`` places a counter in bin ``k`` (metrics
    with only one item in the partition are omitted from ``groups``).

    Returns a list of non-empty bucket lists, or ``None`` if ortools is
    missing, inputs are invalid, time limit hit without feasibility, or solver
    reports infeasible.
    """
    try:
        from ortools.sat.python import cp_model
    except ImportError:
        return None

    if not items_sorted:
        return []
    if len(items_sorted) > _MAX_ITEMS:
        return None
    if not _items_fit_config(items_sorted, perfmon_config):
        return None
    if not _same_bin_group_constraints_feasible(
        items_sorted, perfmon_config, same_bin_counter_groups
    ):
        return None

    item_index = {name: idx for idx, name in enumerate(items_sorted)}
    n = len(items_sorted)
    block_of = [counter_ip_block_for_perfmon(name) for name in items_sorted]

    group_index_lists: list[list[int]] = []
    for grp in same_bin_counter_groups:
        idxs = sorted({item_index[c] for c in grp if c in item_index})
        if len(idxs) >= 2:
            group_index_lists.append(idxs)

    num_bins = min(n, max_bins if max_bins is not None else _MAX_BINS_CAP)

    model = cp_model.CpModel()
    assign: dict[tuple[int, int], object] = {}
    for item_idx in range(n):
        for bin_idx in range(num_bins):
            assign[(item_idx, bin_idx)] = model.NewBoolVar(f"x_{item_idx}_{bin_idx}")

    for item_idx in range(n):
        model.Add(sum(assign[(item_idx, bin_idx)] for bin_idx in range(num_bins)) == 1)

    for bin_idx in range(num_bins):
        for block_name, cap in perfmon_config.items():
            idxs = [i for i in range(n) if block_of[i] == block_name]
            if idxs:
                model.Add(sum(assign[(i, bin_idx)] for i in idxs) <= cap)

    for gix, grp_idxs in enumerate(group_index_lists):
        group_vars_per_bin: list[object] = []
        for bin_idx in range(num_bins):
            z_gk = model.NewBoolVar(f"z_g_{gix}_{bin_idx}")
            group_vars_per_bin.append(z_gk)
            for item_idx in grp_idxs:
                model.Add(z_gk <= assign[(item_idx, bin_idx)])
            model.Add(
                sum(assign[(item_idx, bin_idx)] for item_idx in grp_idxs)
                >= len(grp_idxs) * z_gk
            )
        model.Add(sum(group_vars_per_bin) == 1)

    bin_used = [model.NewBoolVar(f"bu_{bin_idx}") for bin_idx in range(num_bins)]
    for bin_idx in range(num_bins):
        for item_idx in range(n):
            model.Add(bin_used[bin_idx] >= assign[(item_idx, bin_idx)])

    spread_flat: list[object] = []
    if metric_spread_penalty > 0 and metric_spread_index_groups:
        capped = metric_spread_index_groups[:_MAX_METRIC_SPREAD_GROUPS]
        for mix, grp_idxs in enumerate(capped):
            if len(grp_idxs) < 2:
                continue
            for bin_idx in range(num_bins):
                u_mk = model.NewBoolVar(f"ms_{mix}_{bin_idx}")
                spread_flat.append(u_mk)
                for item_idx in grp_idxs:
                    model.Add(u_mk >= assign[(item_idx, bin_idx)])
                model.Add(
                    sum(assign[(item_idx, bin_idx)] for item_idx in grp_idxs)
                    <= len(grp_idxs) * u_mk
                )

    if spread_flat and metric_spread_penalty > 0:
        model.Minimize(
            metric_spread_penalty * sum(spread_flat) + bin_used_weight * sum(bin_used)
        )
    else:
        model.Minimize(bin_used_weight * sum(bin_used))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    buckets: list[list[str]] = [[] for _ in range(num_bins)]
    for item_idx in range(n):
        for bin_idx in range(num_bins):
            if solver.Value(assign[(item_idx, bin_idx)]) == 1:
                buckets[bin_idx].append(items_sorted[item_idx])
                break

    return [b for b in buckets if b]
