# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT
"""Same-pass analyze binding for copied PMC counters.

Single-pass metric packing may list the same counter in multiple perfmon passes
so each metric's PMC set is co-collected. Analyze must evaluate those metrics
using values from one qualifying pass, not a name-keyed mega-table last-write.
"""

import re
from collections import Counter
from dataclasses import dataclass
from typing import (
    Callable,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Match,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Union,
)

import pandas as pd

from utils.logger import console_warning
from utils.metrics.expression import build_eval_string
from utils.metrics.pass_layout import (
    PASS_COLUMN_SEP,
    PassLayout,
    require_safe_pass_key,
)
from utils.utils_counter_defs import (
    SUPPORTED_DENOM,
    UNIT_COUNTER,
    extract_counters_and_variables,
    get_build_in_vars,
)

PASS_VAR_SEP = "__pass"
PASS_KEY_COLUMN = "Pass_Key"

METRIC_ROW_REFS_ATTR = "metric_row_refs"
METRIC_ROW_PASS_ATTR = "metric_row_pass"

_COUNTER_COL_RE = re.compile(r"raw_pmc_df\[['\"]([^'\"]+)['\"]\]")
_AMMOLITE_VAR_RE = re.compile(r"ammolite__([A-Za-z0-9_]+)")
_DIGIT_SPLIT_RE = re.compile(r"(\d+)")

# Copied onto every dispatch row from the kernel launch. Not a Counter_Name,
# so they never select or reject a perfmon pass.
KERNEL_LAUNCH_COLUMNS = frozenset({
    "Grid_Size",
    "Workgroup_Size",
    "LDS_Per_Workgroup",
    "Scratch_Per_Workitem",
    "Arch_VGPR",
    "Accum_VGPR",
    "SGPR",
})

# Wide-frame columns that are not hardware counters. $denom (per_kernel) and
# per-second formulas name these, and they are never a Counter_Name in a pass.
_NON_PMC_FRAME_COLUMNS = (
    frozenset({
        UNIT_COUNTER,
        "Start_Timestamp",
        "End_Timestamp",
    })
    | KERNEL_LAUNCH_COLUMNS
)

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


def natural_pass_sort_key(key: str) -> Tuple[Union[int, str], ...]:
    """Sort ``pmc_perf_2`` before ``pmc_perf_10``."""
    parts = _DIGIT_SPLIT_RE.split(key)
    return tuple(int(part) if part.isdigit() else part for part in parts)


def raw_pmc_column_names(expr: str) -> List[str]:
    """Return raw_pmc_df column names referenced by one expression string."""
    if not isinstance(expr, str) or not expr:
        return []
    return _COUNTER_COL_RE.findall(expr)


def _pmc_column_base(column: str) -> str:
    """Strip a pass qualifier so a second bind sees the hardware counter."""
    if PASS_COLUMN_SEP in column:
        return column.split(PASS_COLUMN_SEP, 1)[0]
    return column


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
        key = require_safe_pass_key(str(pass_key))
        counters_by_pass[key] = frozenset(
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
        for column in raw_pmc_column_names(expr):
            base = _pmc_column_base(column)
            if base in _NON_PMC_FRAME_COLUMNS:
                continue
            counters.add(base)
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
    """Select one pass. SQ_WAVES, GRBM_GUI_ACTIVE, and GRBM_GUI_ACTIVE_sum stay.

    Returns (pass_key, relaxed). pass_key is None when no pass holds the
    counters that must stay together. relaxed is true only when removing a
    runtime normalization counter other than those three produced a qualifying
    pass. GRBM_GUI_ACTIVE_sum is the gfx1250 per-cycle divisor.
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
    normalization_counters -= _SAME_PASS_COUNTERS
    formula_counters = required - normalization_counters
    if not formula_counters or formula_counters == required:
        return None, False
    fallback = select_pass(frozenset(formula_counters), layout)
    return fallback, fallback is not None


# Dividing by these counters across passes is not a same-pass result.
# Relaxing them would read another pass's copy.
_SAME_PASS_COUNTERS = frozenset({
    "SQ_WAVES",
    "GRBM_GUI_ACTIVE",
    "GRBM_GUI_ACTIVE_sum",
})

_UNSET_EXPRESSION = "None"

_ALREADY_BOUND_BUILTIN_RE = re.compile(rf"{re.escape(PASS_VAR_SEP)}\d+$")
_BOUND_BUILTIN_SUFFIX_RE = re.compile(
    rf"(ammolite__[A-Za-z0-9_]+?){re.escape(PASS_VAR_SEP)}\d+"
)


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


def strip_bound_builtin_suffixes(expressions: List[str]) -> List[str]:
    """Turn ``ammolite__var__passN`` back into ``ammolite__var``."""
    return [_BOUND_BUILTIN_SUFFIX_RE.sub(r"\1", expr) for expr in expressions]


def evaluate_pass_scoped_builtins(
    layout: PassLayout,
    used_passes: Set[str],
    gpu_series: str,
    expressions: List[str],
    evaluate: Callable[[str, str, Dict[str, object]], object],
    key_prefix: str,
) -> Dict[str, object]:
    """Evaluate each referenced pass-scoped built-in once per used pass.

    Passes run in natural order and PER_XCD built-ins run before their
    dependents. evaluate receives the built-in name, its pass-bound eval
    string and the values already computed for that pass. Result keys are
    key_prefix + name + "__passN".
    """
    if not used_passes or not layout.has_duplicates:
        return {}
    scoped = pass_scoped_builtins(layout, gpu_series)
    build_in_vars = _referenced_builtins(scoped, gpu_series, expressions)
    if not build_in_vars:
        return {}
    results: Dict[str, object] = {}
    for pass_key in sorted(used_passes, key=layout.ordinal):
        ordinal = layout.ordinal(pass_key)
        pass_locals: Dict[str, object] = {}
        for variable_key, eval_string in ordered_scoped_builtin_bindings(
            build_in_vars, pass_key, layout, scoped
        ):
            pass_locals[f"{key_prefix}{variable_key}{PASS_VAR_SEP}{ordinal}"] = (
                evaluate(variable_key, eval_string, pass_locals)
            )
        results.update(pass_locals)
    return results


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
        fields = [field for field in df.columns if field in supported_fields]
        row_pass: Dict[object, str] = {}
        for row_id in df.index:
            binding = _bind_metric_values(
                f"row {row_id}",
                [df.at[row_id, field] for field in fields],
                pass_layout,
                gpu_series,
                scoped,
            )
            for field, value in zip(fields, binding.values):
                df.at[row_id, field] = value
            unbound += binding.unset
            if binding.pass_key is not None:
                used_passes.add(binding.pass_key)
                row_pass[row_id] = binding.pass_key
        df.attrs[METRIC_ROW_PASS_ATTR] = row_pass

    if unbound:
        console_warning(
            "pass_provenance",
            f"same-pass bind: {unbound} metric row(s) left unset "
            "(counters do not share a pass)",
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
    if expression_df.empty or not pass_layout.has_duplicates:
        return set()
    if "metric_id" not in expression_df.columns or "value" not in expression_df.columns:
        return set()

    scoped = pass_scoped_builtins(pass_layout, gpu_series)
    used_passes: Set[str] = set()
    metric_pass: Dict[object, str] = {}
    unbound = 0

    for metric_id, group in expression_df.groupby("metric_id", sort=False):
        binding = _bind_metric_values(
            f"metric {metric_id}",
            group["value"].tolist(),
            pass_layout,
            gpu_series,
            scoped,
        )
        for row_index, value in zip(group.index, binding.values):
            expression_df.at[row_index, "value"] = value
        unbound += binding.unset
        if binding.pass_key is not None:
            used_passes.add(binding.pass_key)
            metric_pass[metric_id] = binding.pass_key

    expression_df.attrs[METRIC_ROW_PASS_ATTR] = metric_pass
    if unbound:
        console_warning(
            "pass_provenance",
            f"same-pass bind (DB): {unbound} metric(s) left unset "
            "(counters do not share a pass)",
        )
    return used_passes


def _is_metric_expression(value: object) -> bool:
    return isinstance(value, str) and bool(value) and value != _UNSET_EXPRESSION


def _bind_metric_values(
    label: str,
    values: List[object],
    layout: PassLayout,
    gpu_series: str,
    scoped_builtins: FrozenSet[str],
) -> "_MetricBinding":
    """Rewrite one metric's expression values onto the pass it reads from."""
    expressions = [value for value in values if _is_metric_expression(value)]
    pass_key, unset = _choose_metric_pass(label, expressions, layout, gpu_series)
    if unset:
        rewritten = [
            _UNSET_EXPRESSION if _is_metric_expression(value) else value
            for value in values
        ]
        return _MetricBinding(None, rewritten, True)
    if pass_key is None:
        return _MetricBinding(None, list(values), False)
    rewritten = [
        bind_expression(value, pass_key, layout, scoped_builtins)
        if _is_metric_expression(value)
        else value
        for value in values
    ]
    return _MetricBinding(pass_key, rewritten, False)


def _choose_metric_pass(
    label: str,
    expressions: List[str],
    layout: PassLayout,
    gpu_series: str,
) -> Tuple[Optional[str], bool]:
    """Pick the one pass a metric reads its counters from.

    Returns (pass_key, unset). A counter collected in one pass has one value,
    so a metric that reads no duplicated counter keeps its base columns even
    when its counters span passes. Only a metric that reads a duplicated
    counter and fits no single pass is left unset.
    """
    required = expand_required_counters(extract_row_refs(expressions), gpu_series)
    if not required:
        return None, False
    chosen, normalization_relaxed = select_pass_with_normalization_fallback(
        required, layout, gpu_series
    )
    if chosen is None:
        if not required & layout.duplicated:
            return None, False
        console_warning(
            "pass_provenance",
            f"{label}: no single pass for {sorted(required)}; leaving the metric unset",
        )
        return None, True
    if normalization_relaxed:
        missing = required - layout.counters_by_pass[chosen]
        console_warning(
            "pass_provenance",
            f"{label}: runtime normalization counters {sorted(missing)} are "
            f"not in {chosen}; binding formula counters to that pass",
        )
    return chosen, False


def _referenced_builtins(
    scoped_builtins: FrozenSet[str],
    gpu_series: str,
    expressions: List[str],
) -> Dict[str, str]:
    """Scoped built-in formulas that the expressions reach.

    Expressions may already be pass-bound. Built-ins resolve transitively,
    so a dependent such as GRBM_GUI_ACTIVE_PER_XCD is kept when only
    numActiveCUs is named.
    """
    _, builtin_names = extract_counters_and_variables(
        "\n".join(strip_bound_builtin_suffixes(expressions)), gpu_series
    )
    return {
        name: formula
        for name, formula in get_build_in_vars(gpu_series).items()
        if name in builtin_names and name in scoped_builtins
    }


class _MetricBinding(NamedTuple):
    pass_key: Optional[str]
    values: List[object]
    unset: bool
