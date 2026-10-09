# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT
"""Same-pass analyze binding for SPP-duplicated PMC counters.

SPP may list the same counter in multiple perfmon passes so each packable
metric's PMC set is co-collected. Analyze must evaluate those metrics using
values from one qualifying pass, not a name-keyed mega-table last-write.
"""

import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Mapping, Match, Optional, Set, Tuple, Union

import pandas as pd

from utils.logger import console_debug, console_warning
from utils.metrics.expression import build_eval_string
from utils.utils_counter_defs import (
    SUPPORTED_DENOM,
    UNIT_COUNTER,
    extract_counters_and_variables,
    get_build_in_vars,
)

PASS_COLUMN_SEP = "@pass:"
PASS_VAR_SEP = "__pass"
PASS_KEY_COLUMN = "Pass_Key"

METRIC_ROW_REFS_ATTR = "metric_row_refs"
METRIC_ROW_PASS_ATTR = "metric_row_pass"

_COUNTER_COL_RE = re.compile(r"raw_pmc_df\[['\"]([^'\"]+)['\"]\]")
_AMMOLITE_VAR_RE = re.compile(r"ammolite__([A-Za-z0-9_]+)")
_DIGIT_SPLIT_RE = re.compile(r"(\d+)")

# System / unit variables that never take a pass suffix.
_SYSTEM_LIKE_VARS = frozenset({
    "num_xcd",
    "cu_per_gpu",
    "max_waves_per_cu",
    "max_sclk",
    "max_mclk",
    "num_memory_channels",
    "denom",
    "normUnit",
    "total_l2_chan",
    "simd_per_cu",
    "wave_size",
    "workgroup_max_size",
    "max_vgpr",
    "max_sgpr",
    "lds_banks_per_cu",
    "sqc_per_gpu",
    "pipes_per_gpu",
    "se_per_gpu",
})


def legacy_pass_merge_enabled() -> bool:
    """Escape hatch: restore pre-provenance name-keyed merge/eval."""
    raw = os.environ.get("ROCPROF_COMPUTE_ANALYZE_LEGACY_PASS_MERGE", "0")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def natural_pass_sort_key(key: str) -> Tuple[Union[int, str], ...]:
    """Sort ``pmc_perf_2`` before ``pmc_perf_10``."""
    parts = _DIGIT_SPLIT_RE.split(key)
    return tuple(int(part) if part.isdigit() else part for part in parts)


@dataclass(frozen=True)
class PassLayout:
    """Counters observed per profiling pass (from results, not perfmon YAML)."""

    pass_keys: Tuple[str, ...]
    counters_by_pass: Mapping[str, FrozenSet[str]]
    duplicated: FrozenSet[str]

    @property
    def has_duplicates(self) -> bool:
        return bool(self.duplicated)

    def ordinal(self, pass_key: str) -> int:
        return self.pass_keys.index(pass_key)

    def qualified_column(self, counter: str, pass_key: str) -> str:
        return f"{counter}{PASS_COLUMN_SEP}{pass_key}"

    def passes_containing(self, counters: FrozenSet[str]) -> List[str]:
        return [
            pass_key
            for pass_key in self.pass_keys
            if counters <= self.counters_by_pass[pass_key]
        ]

    @classmethod
    def empty(cls) -> "PassLayout":
        return cls(pass_keys=(), counters_by_pass={}, duplicated=frozenset())


@dataclass(frozen=True)
class MetricRowRefs:
    direct_counters: FrozenSet[str]
    builtin_vars: FrozenSet[str]


def build_pass_layout(long_df: pd.DataFrame) -> PassLayout:
    """Build layout from long-form counter rows that include ``Pass_Key``."""
    if long_df.empty or PASS_KEY_COLUMN not in long_df.columns:
        return PassLayout.empty()

    counters_by_pass: Dict[str, FrozenSet[str]] = {}
    for pass_key, group in long_df.groupby(PASS_KEY_COLUMN, sort=False):
        counters_by_pass[str(pass_key)] = frozenset(
            str(name) for name in group["Counter_Name"].dropna().unique()
        )

    pass_keys = tuple(sorted(counters_by_pass, key=natural_pass_sort_key))
    appearance = Counter(
        counter for counters in counters_by_pass.values() for counter in counters
    )
    duplicated = frozenset(
        counter for counter, count in appearance.items() if count > 1
    )
    return PassLayout(
        pass_keys=pass_keys,
        counters_by_pass=counters_by_pass,
        duplicated=duplicated,
    )


def extract_row_refs(built_exprs: List[str]) -> MetricRowRefs:
    """Collect ``raw_pmc_df`` counters and ``ammolite__`` vars from built strings."""
    counters: Set[str] = set()
    builtins: Set[str] = set()
    for expr in built_exprs:
        if not isinstance(expr, str) or not expr:
            continue
        counters.update(_COUNTER_COL_RE.findall(expr))
        for var_name in _AMMOLITE_VAR_RE.findall(expr):
            # Strip accidental pass suffixes if re-extracting.
            if PASS_VAR_SEP in var_name:
                var_name = var_name.split(PASS_VAR_SEP, 1)[0]
            if var_name in _SYSTEM_LIKE_VARS:
                continue
            builtins.add(var_name)
    return MetricRowRefs(
        direct_counters=frozenset(counters),
        builtin_vars=frozenset(builtins),
    )


def expand_required_counters(
    refs: MetricRowRefs,
    gpu_series: str,
) -> FrozenSet[str]:
    """Direct counters plus HW counters pulled in by referenced built-ins."""
    required = set(refs.direct_counters)
    build_ins = get_build_in_vars(gpu_series)
    for var_name in refs.builtin_vars:
        formula = build_ins.get(var_name)
        if not formula:
            continue
        hw, _ = extract_counters_and_variables(
            formula, gpu_series, include_supported_denom=False
        )
        required.update(hw)
    return frozenset(required)


def pass_scoped_builtins(layout: PassLayout, gpu_series: str) -> FrozenSet[str]:
    """Built-ins whose formulas touch a duplicated counter."""
    if not layout.has_duplicates:
        return frozenset()
    scoped: Set[str] = set()
    for var_name, formula in get_build_in_vars(gpu_series).items():
        hw, _ = extract_counters_and_variables(
            formula, gpu_series, include_supported_denom=False
        )
        if hw & layout.duplicated:
            scoped.add(var_name)
    return frozenset(scoped)


# Column names that are not perfmon counters. PassLayout records Counter_Name
# only, so timestamps and the per-kernel unit must not decide the pass.
_NON_PMC_BIND_NAMES = frozenset({
    "Start_Timestamp",
    "End_Timestamp",
    UNIT_COUNTER,
})


def hardware_counters_for_pass_select(
    required: FrozenSet[str],
    _gpu_series: str,
) -> FrozenSet[str]:
    """PMC names used to choose a pass.

    Timestamps and non-counter SUPPORTED_DENOM tokens are stripped. Hardware
    denominator counters such as SQ_WAVES stay, and the normalization fallback
    may still relax those.
    """
    drop = set(_NON_PMC_BIND_NAMES)
    for formula in SUPPORTED_DENOM.values():
        for token in ("Start_Timestamp", "End_Timestamp", UNIT_COUNTER):
            if token in formula:
                drop.add(token)
    return frozenset(name for name in required if name not in drop)


def select_pass(
    required: FrozenSet[str],
    layout: PassLayout,
) -> Optional[str]:
    """Choose the earliest pass (natural order) that contains every required counter."""
    if not required:
        return None
    candidates = layout.passes_containing(required)
    if not candidates:
        return None
    return candidates[0]


def select_pass_with_normalization_fallback(
    required: FrozenSet[str],
    layout: PassLayout,
    gpu_series: str,
) -> Tuple[Optional[str], bool]:
    """Select one pass, relaxing only runtime normalization counters if needed.

    Returns ``(pass_key, relaxed)``. ``pass_key`` is ``None`` when no pass was
    found even after relaxation. ``relaxed`` is true only when removing
    normalization counters produced a qualifying pass.
    """
    chosen = select_pass(required, layout)
    if chosen is not None:
        return chosen, False

    normalization_counters: Set[str] = set()
    for formula in SUPPORTED_DENOM.values():
        counters, _ = extract_counters_and_variables(
            formula,
            gpu_series,
            include_supported_denom=False,
        )
        normalization_counters.update(counters)
    formula_counters = required - normalization_counters
    if not formula_counters or formula_counters == required:
        return None, False
    fallback = select_pass(frozenset(formula_counters), layout)
    return fallback, fallback is not None


def choose_metric_pass(
    required: FrozenSet[str],
    layout: PassLayout,
    gpu_series: str,
    label: str,
) -> Optional[str]:
    """Pick a pass from the hardware PMC set.

    Timestamps alone do not make the row unset. If the remaining PMC set fits
    no pass, warn and return None so the caller leaves the metric unset.
    """
    pmc_required = hardware_counters_for_pass_select(required, gpu_series)
    chosen, normalization_relaxed = select_pass_with_normalization_fallback(
        pmc_required,
        layout,
        gpu_series,
    )
    if chosen is None:
        if pmc_required:
            console_warning(
                "pass_provenance",
                f"{label}: contract violation, PMC set {sorted(pmc_required)} "
                "does not fit one pass; leaving the metric unset",
            )
        else:
            console_debug(
                "pass_provenance",
                f"{label}: no hardware PMC set to bind",
            )
        return None
    if normalization_relaxed:
        missing = pmc_required - layout.counters_by_pass[chosen]
        console_warning(
            "pass_provenance",
            f"{label}: runtime normalization counters {sorted(missing)} "
            f"are not in {chosen}; binding formula counters to that pass",
        )
    return chosen


_ALREADY_BOUND_BUILTIN_RE = re.compile(rf"{re.escape(PASS_VAR_SEP)}\d+$")


def bind_expression(
    expr: str,
    pass_key: str,
    layout: PassLayout,
    scoped_builtins: FrozenSet[str],
) -> str:
    """Rewrite duplicated counters and pass-scoped built-ins onto ``pass_key``.

    Idempotent: a name that already ends in ``__passN``, or that merely shares
    a prefix with a scoped built-in, is left unchanged.
    """

    def _replace_counter(match: Match[str]) -> str:
        counter = match.group(1)
        if counter in layout.duplicated and counter in layout.counters_by_pass.get(
            pass_key, frozenset()
        ):
            qualified = layout.qualified_column(counter, pass_key)
            return f"raw_pmc_df['{qualified}']"
        return match.group(0)

    def _replace_builtin(match: Match[str]) -> str:
        var_name = match.group(1)
        if var_name not in scoped_builtins:
            return match.group(0)
        if _ALREADY_BOUND_BUILTIN_RE.search(var_name):
            return match.group(0)
        ordinal = layout.ordinal(pass_key)
        return f"ammolite__{var_name}{PASS_VAR_SEP}{ordinal}"

    bound = _COUNTER_COL_RE.sub(_replace_counter, expr)
    return _AMMOLITE_VAR_RE.sub(_replace_builtin, bound)


def ordered_scoped_builtin_bindings(
    build_in_vars: Mapping[str, str],
    pass_key: str,
    layout: PassLayout,
    scoped_builtins: FrozenSet[str],
) -> List[Tuple[str, str]]:
    """Bind scoped built-in formulas for one pass, ``PER_XCD`` names first."""
    per_xcd = [key for key in build_in_vars if "PER_XCD" in key]
    dependents = [key for key in build_in_vars if "PER_XCD" not in key]
    return [
        (
            variable_key,
            bind_expression(
                build_eval_string(build_in_vars[variable_key]),
                pass_key,
                layout,
                scoped_builtins,
            ),
        )
        for variable_key in per_xcd + dependents
    ]


def bind_metric_tables_to_passes(
    dfs: dict,
    dfs_type: dict,
    pass_layout: PassLayout,
    gpu_series: str,
    supported_fields: FrozenSet[str],
) -> Set[str]:
    """Rewrite metric-table expression cells onto a co-located pass.

    Mutates ``dfs`` in place. Evaluation then stores numeric results in those
    same frames, so callers pass the live workload tables. A second bind of a
    still-string formula is idempotent.

    Returns the set of pass keys that were selected for at least one row.
    """
    if not pass_layout.has_duplicates:
        return set()

    scoped = pass_scoped_builtins(pass_layout, gpu_series)
    used_passes: Set[str] = set()
    unbound = 0

    for table_id, df in dfs.items():
        if dfs_type.get(table_id) != "metric_table":
            continue
        row_pass: Dict[object, str] = {}
        for row_id, row in df.iterrows():
            exprs = [
                row[field]
                for field in df.columns
                if field in supported_fields
                and isinstance(row[field], str)
                and row[field]
            ]
            refs = extract_row_refs(exprs)
            required = expand_required_counters(refs, gpu_series)
            chosen = choose_metric_pass(
                required,
                pass_layout,
                gpu_series,
                f"row {row_id}",
            )
            if chosen is None:
                unbound += 1
                continue
            used_passes.add(chosen)
            row_pass[row_id] = chosen
            for field in df.columns:
                if field not in supported_fields:
                    continue
                value = df.at[row_id, field]
                if isinstance(value, str) and value:
                    df.at[row_id, field] = bind_expression(
                        value, chosen, pass_layout, scoped
                    )
        df.attrs[METRIC_ROW_PASS_ATTR] = row_pass

    if unbound:
        console_debug(
            "pass_provenance",
            f"same-pass bind: {unbound} metric row(s) fell back to base columns",
        )
    return used_passes


def bind_expression_dataframe(
    expression_df: pd.DataFrame,
    pass_layout: PassLayout,
    gpu_series: str,
) -> Set[str]:
    """Bind long-form DB expression rows onto co-located passes.

    Expects columns ``metric_id`` and ``value``. Mutates ``value`` in place and
    stores ``metric_id -> pass_key`` on ``expression_df.attrs[METRIC_ROW_PASS_ATTR]``.
    All value columns for one ``metric_id`` share a single pass.

    Returns the set of pass keys selected for at least one metric.
    """
    if (
        expression_df.empty
        or not pass_layout.has_duplicates
        or legacy_pass_merge_enabled()
    ):
        return set()
    if "metric_id" not in expression_df.columns or "value" not in expression_df.columns:
        return set()

    scoped = pass_scoped_builtins(pass_layout, gpu_series)
    used_passes: Set[str] = set()
    metric_pass: Dict[object, str] = {}
    unbound = 0

    for metric_id, group in expression_df.groupby("metric_id", sort=False):
        exprs = [
            value
            for value in group["value"].tolist()
            if isinstance(value, str) and value and value != "None"
        ]
        if not exprs:
            continue
        refs = extract_row_refs(exprs)
        required = expand_required_counters(refs, gpu_series)
        chosen = choose_metric_pass(
            required,
            pass_layout,
            gpu_series,
            f"metric {metric_id}",
        )
        if chosen is None:
            unbound += 1
            continue
        used_passes.add(chosen)
        metric_pass[metric_id] = chosen
        for row_index in group.index:
            value = expression_df.at[row_index, "value"]
            if isinstance(value, str) and value and value != "None":
                expression_df.at[row_index, "value"] = bind_expression(
                    value, chosen, pass_layout, scoped
                )

    expression_df.attrs[METRIC_ROW_PASS_ATTR] = metric_pass
    if unbound:
        console_debug(
            "pass_provenance",
            f"same-pass bind (DB): {unbound} metric(s) fell back to base columns",
        )
    return used_passes


def resolve_weight_counter_column(
    weight_counter: str,
    sub_metric_name: str,
    df: pd.DataFrame,
    pass_layout: Optional[PassLayout],
) -> Optional[str]:
    """Read a WEIGHTED_AVG weight from the sub-metric's pass.

    A weight that exists in only one pass is still that pass. If it is not
    the sub-ratio's pass, return the qualified name for the sub-ratio's pass
    so analysis does not read the other replay. None means the pass is
    unknown and the composite must stay unset.
    """
    if pass_layout is None or not pass_layout.pass_keys:
        return weight_counter
    pass_key = _bound_pass_for_sub_metric(df, sub_metric_name)
    if pass_key is None:
        console_warning(
            "pass_provenance",
            f"WEIGHTED_AVG weight {weight_counter!r} for {sub_metric_name!r} "
            "has no sub-metric pass; not reading another replay",
        )
        return None
    in_pass = weight_counter in pass_layout.counters_by_pass.get(pass_key, frozenset())
    if not in_pass:
        console_warning(
            "pass_provenance",
            f"WEIGHTED_AVG weight {weight_counter!r} for {sub_metric_name!r} "
            f"is not in {pass_key}; not using another pass",
        )
        return pass_layout.qualified_column(weight_counter, pass_key)
    if weight_counter in pass_layout.duplicated:
        return pass_layout.qualified_column(weight_counter, pass_key)
    return weight_counter


def _bound_pass_for_sub_metric(
    df: pd.DataFrame,
    sub_metric_name: str,
) -> Optional[str]:
    """Pass key same-pass bind stored for the sub-metric row."""
    row_pass = df.attrs.get(METRIC_ROW_PASS_ATTR, {})
    if not isinstance(row_pass, dict) or "Metric" not in df.columns:
        return None
    matches = df.index[df["Metric"] == sub_metric_name]
    if len(matches) == 0:
        return None
    pass_key = row_pass.get(matches[0])
    if not isinstance(pass_key, str) or not pass_key:
        return None
    return pass_key
