# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT
"""Same-pass analyze binding for SPP-duplicated PMC counters.

SPP may list the same counter in multiple perfmon passes so each packable
metric's PMC set is co-collected. Analyze must evaluate those metrics using
values from one qualifying pass, not a name-keyed mega-table last-write.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from utils.logger import console_debug
from utils.utils_counter_defs import extract_counters_and_variables, get_build_in_vars

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


def natural_pass_sort_key(key: str) -> tuple:
    """Sort ``pmc_perf_2`` before ``pmc_perf_10``."""
    parts = _DIGIT_SPLIT_RE.split(key)
    return tuple(int(part) if part.isdigit() else part for part in parts)


@dataclass(frozen=True)
class PassLayout:
    """Counters observed per profiling pass (from results, not perfmon YAML)."""

    pass_keys: tuple[str, ...]
    counters_by_pass: Mapping[str, frozenset[str]]
    duplicated: frozenset[str]

    @property
    def has_duplicates(self) -> bool:
        return bool(self.duplicated)

    def ordinal(self, pass_key: str) -> int:
        return self.pass_keys.index(pass_key)

    def qualified_column(self, counter: str, pass_key: str) -> str:
        return f"{counter}{PASS_COLUMN_SEP}{pass_key}"

    def passes_containing(self, counters: frozenset[str]) -> list[str]:
        return [
            pass_key
            for pass_key in self.pass_keys
            if counters <= self.counters_by_pass[pass_key]
        ]

    @classmethod
    def empty(cls) -> PassLayout:
        return cls(pass_keys=(), counters_by_pass={}, duplicated=frozenset())


@dataclass(frozen=True)
class MetricRowRefs:
    direct_counters: frozenset[str]
    builtin_vars: frozenset[str]


def build_pass_layout(long_df: pd.DataFrame) -> PassLayout:
    """Build layout from long-form counter rows that include ``Pass_Key``."""
    if long_df.empty or PASS_KEY_COLUMN not in long_df.columns:
        return PassLayout.empty()

    counters_by_pass: dict[str, frozenset[str]] = {}
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


def extract_row_refs(built_exprs: list[str]) -> MetricRowRefs:
    """Collect ``raw_pmc_df`` counters and ``ammolite__`` vars from built strings."""
    counters: set[str] = set()
    builtins: set[str] = set()
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
) -> frozenset[str]:
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


def pass_scoped_builtins(layout: PassLayout, gpu_series: str) -> frozenset[str]:
    """Built-ins whose formulas touch a duplicated counter."""
    if not layout.has_duplicates:
        return frozenset()
    scoped: set[str] = set()
    for var_name, formula in get_build_in_vars(gpu_series).items():
        hw, _ = extract_counters_and_variables(
            formula, gpu_series, include_supported_denom=False
        )
        if hw & layout.duplicated:
            scoped.add(var_name)
    return frozenset(scoped)


def select_pass(
    required: frozenset[str],
    layout: PassLayout,
) -> Optional[str]:
    """Choose the earliest pass (natural order) that contains every required counter."""
    if not required:
        return None
    candidates = layout.passes_containing(required)
    if not candidates:
        return None
    return candidates[0]


def bind_expression(
    expr: str,
    pass_key: str,
    layout: PassLayout,
    scoped_builtins: frozenset[str],
) -> str:
    """Rewrite duplicated counters and pass-scoped built-ins onto ``pass_key``."""

    def _replace_counter(match: re.Match[str]) -> str:
        counter = match.group(1)
        if counter in layout.duplicated and counter in layout.counters_by_pass.get(
            pass_key, frozenset()
        ):
            qualified = layout.qualified_column(counter, pass_key)
            return f"raw_pmc_df['{qualified}']"
        return match.group(0)

    bound = _COUNTER_COL_RE.sub(_replace_counter, expr)
    ordinal = layout.ordinal(pass_key)
    for var_name in scoped_builtins:
        bound = bound.replace(
            f"ammolite__{var_name}",
            f"ammolite__{var_name}{PASS_VAR_SEP}{ordinal}",
        )
    return bound


def bind_metric_tables_to_passes(
    dfs: dict,
    dfs_type: dict,
    pass_layout: PassLayout,
    gpu_series: str,
    supported_fields: frozenset[str],
) -> set[str]:
    """Rewrite metric table expressions onto a co-located pass.

    Returns the set of pass keys that were selected for at least one row.
    """
    if not pass_layout.has_duplicates:
        return set()

    scoped = pass_scoped_builtins(pass_layout, gpu_series)
    used_passes: set[str] = set()
    unbound = 0

    for table_id, df in dfs.items():
        if dfs_type.get(table_id) != "metric_table":
            continue
        row_pass: dict[object, str] = {}
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
            # Prefer a pass that also has any normalization counters named
            # directly (already in required if present in the formula).
            chosen = select_pass(required, pass_layout)
            if chosen is None:
                unbound += 1
                console_debug(
                    "pass_provenance",
                    f"row {row_id}: no single pass for {sorted(required)}; "
                    "using base columns",
                )
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
) -> set[str]:
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
    used_passes: set[str] = set()
    metric_pass: dict[object, str] = {}
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
        chosen = select_pass(required, pass_layout)
        if chosen is None:
            unbound += 1
            console_debug(
                "pass_provenance",
                f"metric {metric_id}: no single pass for {sorted(required)}; "
                "using base columns",
            )
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
) -> str:
    """Qualify a WEIGHTED_AVG weight counter to the sub-metric's bound pass."""
    if pass_layout is None or not pass_layout.has_duplicates:
        return weight_counter
    if weight_counter not in pass_layout.duplicated:
        return weight_counter
    row_pass = df.attrs.get(METRIC_ROW_PASS_ATTR, {})
    # Sub-metric rows are keyed by metric name in the Metric column.
    if "Metric" not in df.columns:
        return weight_counter
    matches = df.index[df["Metric"] == sub_metric_name]
    if len(matches) == 0:
        return weight_counter
    pass_key = row_pass.get(matches[0])
    if not pass_key:
        return weight_counter
    if weight_counter not in pass_layout.counters_by_pass.get(pass_key, frozenset()):
        return weight_counter
    return pass_layout.qualified_column(weight_counter, pass_key)
