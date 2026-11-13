#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Generate a metric health HTML report from one or two runs.

A run directory holds one subdirectory per workload, named after it (vcopy,
nbody, mega_kernel). Each workload subdirectory holds:

* `profile/`: the `rocprof-compute profile --output-directory` output.
* `analyze/`: the `rocprof-compute analyze --output-format csv` output of that
  profile, written by the same build that profiled it.

Pass the baseline run with `-p`. Pass the candidate run with a second `-p` to
compare the two; one run gives the report without comparison sections.

Values belong to the kernel with the longest total duration, the one
`analyze -k 0` selects. The primary statistic is Median:

* Use analyze's Median when the table has one.
* Else recompute it from per-dispatch data with this repo's analyze code.
  `SUM / SUM`, `AVG` and Min formulas are evaluated per dispatch.
* Else, if Min equals Max, use that value.
* Tables with only a Value column use Value.

Std Dev is the per-dispatch sample standard deviation, from analyze or
recomputed the same way. CV is Std Dev / |Median|.
"""

import argparse
import csv
import html
import math
import re
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import config
from rocprof_compute_soc.soc_base import iter_analysis_config_metrics
from utils import file_io, parser, schema
from utils.metrics.evaluation_pipeline import eval_metric
from utils.metrics.expression import build_metric_value_string
from utils.utils_common import canonical_config_arch

# Changes at least this large go in the delta table.
DELTA_REL_THRESHOLD = 0.05
# Changes at least this large are highlighted.
DELTA_HIGHLIGHT_THRESHOLD = 1.0
# A value closer to zero than this counts as zero.
DELTA_NEAR_ZERO = 1e-12
# Min and Max closer than this count as equal.
MIN_MAX_EQUAL_TOLERANCE = 1e-9

# The statistic every value column shows.
PRIMARY_STAT = "Median"

# Arches where Block 18 has one row per L2 channel ($total_l2_chan).
L2_CHANNEL_ROW_ARCHES = frozenset({"gfx942"})

DEFAULT_CANDIDATE_LABEL = "this branch"
DEFAULT_BASELINE_LABEL = "rocprofiler-compute-develop"

# Analyze value columns the report reads. A metric with none of them is skipped.
STAT_COLUMNS: Tuple[str, ...] = ("Median", "Value", "Min", "Max", "Avg")
# Columns that can hold a metric's formula; the first one that is set wins.
FORMULA_COLUMNS: Tuple[str, ...] = ("Value", "Avg", "Median", "Min", "Max")

LEVEL_COUNTER_RE = re.compile(r"_LEVEL(?:_sum)?|INFLIGHT_LEVEL", re.IGNORECASE)

COUNTER_SEMANTIC_OVERFLOW_NAMES = frozenset({
    "CP Utilization",
    "Pipeline Utilization - CMACC",
    "Pipeline Utilization - SMACC",
    "Pipeline Utilization - Double Precision",
    "Pipeline Utilization - XDL",
    "SPI Utilization",
})

ZERO_SUB_BUCKETS: Tuple[str, ...] = (
    "zero_healthy",
    "zero_optional",
    "zero_sdk_bug",
    "zero_other",
)

HEALTHY_ZERO_NAME_FRAGMENTS: Tuple[str, ...] = (
    "Alloc Failure",
    "FIFO Full",
    "Context Save",
    "Context Restore",
    "No-Allocation",
    "Throttle",
    "Backpressure",
    "Retry",
    "Timeout",
    "Error Rate",
    "Underflow",
    "Overflow Rate",
    "Conflict Rate",
    "Stall Rate",
)

OPTIONAL_ZERO_NAME_FRAGMENTS: Tuple[str, ...] = (
    "FP64",
    "FP8",
    "FP6",
    "FP4",
    "Int4",
    "Sparse",
    "VALU FLOPs (F64)",
    "Double Precision",
)

SDK_BUG_ZERO_NAME_FRAGMENTS: Tuple[str, ...] = (
    "VALU FLOPs",
    "VALU IOPs",
    "VALU Trans FLOPs",
    "WMMA FLOPs",
    "DRAM Bandwidth - Read",
    "GL2 to DRAM Read",
    "GL2 to DRAM Read Latency",
    "Buffer Atomics Instructions",
)

ZERO_BUCKET_LABELS: Dict[str, str] = {
    "zero_healthy": "Healthy zero (0% stall/failure)",
    "zero_optional": "Optional path (not exercised)",
    "zero_sdk_bug": "SDK / counter bug suspect",
    "zero_other": "Other all-workload zero",
}

ZERO_BUCKET_CARD_STYLES: Dict[str, Tuple[str, str]] = {
    "zero_healthy": ("#e8f5e9", "#2e7d32"),
    "zero_optional": ("#e3f2fd", "#1565c0"),
    "zero_sdk_bug": ("#ffcdd2", "#c62828"),
    "zero_other": ("#f5f5f5", "#616161"),
}

OVERFLOW_CLASSES = frozenset({"overflow", "formula_overflow"})

WORKLOAD_DESCRIPTIONS: Dict[str, str] = {
    "mega_kernel": "All-ops (RDNA35 mega kernel)",
    "vcopy": "Memory-bound",
    "nbody": "Compute-bound",
    "wmma_gemm": "WMMA",
}


@dataclass
class MetricValue:
    """Primary health value is Median (never Avg)."""

    value: Optional[float]
    source: str = ""  # median | recomputed | minmax_eq | value
    min_v: Optional[float] = None
    max_v: Optional[float] = None
    std_v: Optional[float] = None
    cls: str = ""


@dataclass
class WorkloadMetrics:
    """One workload's metrics for its longest-running kernel, keyed by metric."""

    kernel_name: str = ""
    dispatch_count: Optional[int] = None
    metrics: Dict[str, MetricValue] = field(default_factory=dict)
    units: Dict[str, str] = field(default_factory=dict)


@dataclass
class RunArtifacts:
    """Profile and analyze CSV directories of one run, keyed by workload."""

    label: str
    artifact_dir: Path
    profile_dirs: Dict[str, Path] = field(default_factory=dict)
    csv_dirs: Dict[str, Path] = field(default_factory=dict)


@dataclass
class MetricChange:
    """One workload's metric compared between the baseline and the candidate.

    `new` is None when the candidate has no row for a baseline metric.
    """

    workload: str
    key: str
    old: MetricValue
    new: Optional[MetricValue]
    rel: Optional[float] = None


@dataclass
class ReportData:
    """Everything the HTML report shows, computed before any rendering."""

    wl_names: List[str]
    metrics: Dict[str, Dict[str, MetricValue]]
    dispatch_counts: Dict[str, Optional[int]]
    all_keys: List[str]
    summary: Dict[str, Dict[str, int]] = field(default_factory=dict)
    improvements: List[MetricChange] = field(default_factory=list)
    deltas: List[MetricChange] = field(default_factory=list)
    missing: List[MetricChange] = field(default_factory=list)


def to_float(value: str) -> Optional[float]:
    cleaned = value.strip().replace(",", "")
    if not cleaned or cleaned.upper() in {"N/A", "NAN", "-", ""}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def relative_abs_delta(old: float, new: float) -> Optional[float]:
    """Return |new-old|/|old|, or None when the baseline is near zero."""
    if abs(old) < DELTA_NEAR_ZERO:
        return None
    return abs(new - old) / abs(old)


def uses_level_counter(formula: str) -> bool:
    return bool(LEVEL_COUNTER_RE.search(formula))


def classify_all_workload_zero(name: str, formula: str) -> str:
    if uses_level_counter(formula):
        return "zero_sdk_bug"
    if any(fragment in name for fragment in HEALTHY_ZERO_NAME_FRAGMENTS):
        return "zero_healthy"
    # Optional names such as "VALU FLOPs (F64)" also match the broader SDK
    # fragments, so the optional rule must run first.
    if any(fragment in name for fragment in OPTIONAL_ZERO_NAME_FRAGMENTS):
        return "zero_optional"
    if any(fragment in name for fragment in SDK_BUG_ZERO_NAME_FRAGMENTS):
        return "zero_sdk_bug"
    formula_upper = formula.upper()
    if any(token in formula_upper for token in ("FP64", "FP8", "FP6", "FP4", "SPARSE")):
        return "zero_optional"
    return "zero_other"


def metric_value_from_columns(columns: Dict[str, str]) -> MetricValue:
    """Pick a metric's primary value from its analyze value columns.

    Median wins, then Value when the table has no Median or Min column, then
    Min when it equals Max. Avg is never the primary value; an Avg-only metric
    waits for the Median recompute.
    """
    median = to_float(columns.get("Median", ""))
    value_only = to_float(columns.get("Value", ""))
    min_v = to_float(columns.get("Min", ""))
    max_v = to_float(columns.get("Max", ""))
    value: Optional[float] = None
    source = ""
    if median is not None:
        value, source = median, "median"
    elif value_only is not None and "Median" not in columns and "Min" not in columns:
        value, source = value_only, "value"
    elif (
        min_v is not None
        and max_v is not None
        and abs(min_v - max_v)
        <= max(MIN_MAX_EQUAL_TOLERANCE, MIN_MAX_EQUAL_TOLERANCE * abs(min_v))
    ):
        value, source = min_v, "minmax_eq"
    return MetricValue(
        value=value,
        source=source,
        min_v=min_v,
        max_v=max_v,
        std_v=to_float(columns.get("Std Dev", "")),
    )


def read_top_kernel_metrics(csv_dir: Path) -> WorkloadMetrics:
    """Read the longest-running kernel's metrics from an analyze CSV export.

    Analyze ranks kernels by total duration, so this is the kernel that
    analyze -k 0 selects.
    """
    kernels = _read_csv_rows(csv_dir / "kernel.csv")
    if not kernels:
        raise SystemExit(f"No kernels in {csv_dir / 'kernel.csv'}")
    top = max(kernels, key=lambda row: to_float(row["duration_ns_sum"]) or 0.0)
    columns_by_key: Dict[str, Dict[str, str]] = {}
    units: Dict[str, str] = {}
    for row in _read_csv_rows(csv_dir / "kernel_metric.csv"):
        if row["kernel_uuid"] != top["kernel_uuid"]:
            continue
        key = f"{row['metric_id']}|{row['metric_name']}"
        columns_by_key.setdefault(key, {})[row["value_name"]] = row["value"]
        units[key] = row["unit"]
    metrics = {
        key: metric_value_from_columns(columns)
        for key, columns in columns_by_key.items()
        if any(name in columns for name in STAT_COLUMNS)
    }
    return WorkloadMetrics(
        kernel_name=top["kernel_name"],
        dispatch_count=int(top["dispatch_count"]),
        metrics=metrics,
        units={key: units[key] for key in metrics},
    )


def min_formula_to_median_expr(min_formula: str) -> Optional[str]:
    """Rewrite a Min formula into a Median expression.

    Handles `MIN(x)` and `c * MIN(x)` / `MIN(x) * c` patterns that
    appear in analysis_configs.
    """
    formula = min_formula.strip()
    if not formula or formula == "None":
        return None

    if formula.startswith("MIN("):
        depth = 0
        for index, ch in enumerate(formula):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    if index != len(formula) - 1:
                        break
                    return f"MEDIAN{formula[3 : index + 1]}"
        return None

    factored = re.fullmatch(r"(.+?)\s*\*\s*(MIN\(.+\))", formula)
    if factored:
        left, min_call = factored.group(1).strip(), factored.group(2)
        median_call = min_formula_to_median_expr(min_call)
        if median_call:
            return f"{left} * {median_call}"
    factored = re.fullmatch(r"(MIN\(.+\))\s*\*\s*(.+)", formula)
    if factored:
        min_call, right = factored.group(1), factored.group(2).strip()
        median_call = min_formula_to_median_expr(min_call)
        if median_call:
            return f"{median_call} * {right}"
    return None


def aggregator_to_stat(expression: str, replacement: str) -> Optional[str]:
    """Replace exactly one Min or Median aggregation with a statistic."""
    hits = re.findall(r"\b(?:to_median|to_min|MEDIAN|MIN)\(", expression)
    if len(hits) != 1:
        return None
    return re.sub(
        r"\b(?:to_median|to_min|MEDIAN|MIN)\(",
        replacement + "(",
        expression,
        count=1,
    )


def per_dispatch_stat_expr(expression: object, statistic: str) -> Optional[str]:
    """Build a Median or Std Dev expression over per-dispatch values.

    to_sum(x) and to_avg(x) become x for each dispatch. A single Min or Median
    call becomes the statistic. Max and COLLECT_SUM give None.
    """
    if not isinstance(expression, str):
        return None
    text = expression.strip()
    if not text or text == "None":
        return None
    if statistic == "median":
        direct = aggregator_to_stat(text, "to_median")
        if direct is not None:
            return None if "to_sum(" in direct else direct
        reducer = "to_median"
    elif statistic == "stddev":
        direct = aggregator_to_stat(text, "to_std")
        if direct is not None:
            return None if "to_sum(" in direct else direct
        reducer = "to_std"
    else:
        raise ValueError(f"Unsupported statistic: {statistic}")
    if "to_sum(" not in text and "to_avg(" not in text:
        return None
    if re.search(r"\bto_(?:median|min|max|std|quantile)\(", text):
        return None
    return f"{reducer}({_replace_reducible_aggregates(text)})"


def load_workload_metric_tables(wl_dir: Path) -> Tuple[object, pd.DataFrame]:
    """Build the analyze metric tables for a profiled workload's own arch.

    Returns the arch config, with one table per panel, and the sysinfo frame.
    """
    sys_info, arch = read_sysinfo(wl_dir)
    config_dir = config.analysis_configs_dir / (canonical_config_arch(arch) or arch)
    arch_config = schema.ArchConfig()
    arch_config.panel_configs = file_io.load_panel_configs([str(config_dir)])
    parser.build_dfs(
        arch_configs=arch_config,
        filter_metrics=None,
        sys_info=sys_info.iloc[0],
        profiling_config=file_io.load_profiling_config(str(wl_dir)),
        arch=arch,
    )
    return arch_config, sys_info


def metric_formulas(arch_config: object) -> Dict[str, str]:
    """Map each metric key to its formula, before evaluation.

    Keys use the expanded names analyze writes (Block 18 Channel 0, ...).
    """
    formulas: Dict[str, str] = {}
    for df_id, frame in arch_config.dfs.items():
        if arch_config.dfs_type.get(df_id) != "metric_table":
            continue
        if "Metric" not in frame.columns:
            continue
        columns = [name for name in FORMULA_COLUMNS if name in frame.columns]
        for row_id, row in frame.iterrows():
            formulas[f"{row_id}|{row['Metric']}"] = next(
                (
                    row[name]
                    for name in columns
                    if isinstance(row[name], str) and row[name] not in ("", "None")
                ),
                "",
            )
    return formulas


def collect_formulas(
    run: RunArtifacts, results: Dict[str, WorkloadMetrics]
) -> Dict[str, Tuple[str, str, bool]]:
    """Map each reported metric key to (formula, unit, reads a LEVEL counter)."""
    formulas: Dict[str, Tuple[str, str, bool]] = {}
    for wl_name, wl_dir in run.profile_dirs.items():
        arch_config, _sys_info = load_workload_metric_tables(wl_dir)
        formula_by_key = metric_formulas(arch_config)
        for key, unit in results[wl_name].units.items():
            formula = formula_by_key.get(key, "")
            formulas[key] = (formula, unit, uses_level_counter(formula))
    return formulas


def compute_dispatch_stats_for_workload(
    wl_dir: Path,
    kernel_name: str,
) -> Tuple[Dict[str, Optional[float]], Dict[str, Optional[float]]]:
    """Recompute per-dispatch Median and sample Std Dev for one kernel."""
    arch_config, sys_info = load_workload_metric_tables(wl_dir)

    for df_id, frame in arch_config.dfs.items():
        if arch_config.dfs_type.get(df_id) != "metric_table":
            continue
        if "Min" not in frame.columns or "Median" in frame.columns:
            continue
        medians: List[Optional[str]] = []
        for _, row in frame.iterrows():
            min_expr = row.get("Min")
            if not isinstance(min_expr, str) or not min_expr or min_expr == "None":
                medians.append(None)
                continue
            medians.append(min_formula_to_median_expr(min_expr))
        frame["Median"] = medians

    build_metric_value_string(arch_config.dfs, arch_config.dfs_type, "per_kernel")

    for df_id, frame in arch_config.dfs.items():
        if arch_config.dfs_type.get(df_id) != "metric_table":
            continue
        if "Median" not in frame.columns or "Min" not in frame.columns:
            continue
        for row_id, row in frame.iterrows():
            median_expr = row.get("Median")
            min_expr = row.get("Min")
            if (not isinstance(median_expr, str) or not median_expr) and isinstance(
                min_expr, str
            ):
                if "to_min(" in min_expr:
                    frame.at[row_id, "Median"] = min_expr.replace(
                        "to_min(", "to_median(", 1
                    )

    _attach_stat_column(arch_config.dfs, arch_config.dfs_type, "median")
    _attach_stat_column(arch_config.dfs, arch_config.dfs_type, "stddev")

    workload = schema.Workload()
    workload.sys_info = sys_info
    workload.roofline_peaks = pd.DataFrame()
    workload.filter_gpu_ids = None
    workload.filter_kernel_ids = None
    workload.filter_dispatch_ids = None
    workload.raw_pmc, workload.pass_tables = file_io.create_df_pmc(
        str(wl_dir), verbose=0
    )
    filtered = parser.apply_filters(workload, debug=False)
    # Keep only the dispatches of the kernel analyze reported.
    filtered = filtered[filtered["Kernel_Name"] == kernel_name]
    if filtered.empty:
        raise SystemExit(f"Kernel {kernel_name} has no dispatches in {wl_dir}")
    eval_metric(
        arch_config.dfs,
        arch_config.dfs_type,
        arch_config.dfs_expressions,
        sys_info.iloc[0],
        workload.roofline_peaks,
        filtered,
        debug=False,
        pass_tables=[
            table.loc[filtered.index] for table in workload.pass_tables.values()
        ],
    )
    return (
        _read_stat_column(arch_config, "Median"),
        _read_stat_column(arch_config, "Std Dev"),
    )


def apply_recomputed_medians(
    results: Dict[str, WorkloadMetrics],
    profile_dirs: Dict[str, Path],
) -> None:
    """Fill missing Medians and Std Devs by re-evaluating each dispatch."""
    for wl_name, wl_dir in profile_dirs.items():
        workload_metrics = results[wl_name]
        medians, stddevs = compute_dispatch_stats_for_workload(
            wl_dir, workload_metrics.kernel_name
        )
        for key, med in medians.items():
            if med is None:
                continue
            existing = workload_metrics.metrics.get(key)
            if existing is None:
                workload_metrics.metrics[key] = MetricValue(
                    value=med, source="recomputed"
                )
            elif existing.source != "median":
                existing.value = med
                existing.source = "recomputed"
        for key, std in stddevs.items():
            if std is None:
                continue
            existing = workload_metrics.metrics.get(key)
            if existing is None or existing.std_v is not None:
                continue
            existing.std_v = std


def discover_run(run_dir: Path, label: str) -> RunArtifacts:
    """Find each workload's profile and analyze CSV directories in one run.

    Every subdirectory is a workload and must hold profile/ and analyze/.
    """
    if not run_dir.is_dir():
        raise SystemExit(f"{label} run directory not found: {run_dir}")
    run = RunArtifacts(label=label, artifact_dir=run_dir)
    for workload_dir in sorted(path for path in run_dir.iterdir() if path.is_dir()):
        sysinfo_paths = sorted((workload_dir / "profile").glob("**/sysinfo.csv"))
        csv_dir = workload_dir / "analyze"
        if not sysinfo_paths or not (csv_dir / "kernel_metric.csv").is_file():
            raise SystemExit(
                f"{label}: {workload_dir} needs profile/ with a profiled workload "
                "and analyze/ with analyze --output-format csv output"
            )
        run.profile_dirs[workload_dir.name] = sysinfo_paths[0].parent
        run.csv_dirs[workload_dir.name] = csv_dir
    if not run.csv_dirs:
        raise SystemExit(f"{label} run directory has no workloads: {run_dir}")
    return run


def load_run_metrics(run: RunArtifacts) -> Dict[str, WorkloadMetrics]:
    """Read every workload's analyze CSV export and fill Medians from its data."""
    results = {wl: read_top_kernel_metrics(path) for wl, path in run.csv_dirs.items()}
    apply_recomputed_medians(results, run.profile_dirs)
    return results


def read_sysinfo(profile_dir: Path) -> Tuple[pd.DataFrame, str]:
    """Return a profiled workload's sysinfo frame and its GPU arch."""
    sys_info = pd.read_csv(profile_dir / "sysinfo.csv")
    return sys_info, str(sys_info.iloc[0]["gpu_arch"])


def classify_metrics(
    workload_metrics: Dict[str, Dict[str, MetricValue]],
    metric_units: Dict[str, str],
    formula_by_key: Dict[str, str],
) -> None:
    mega = workload_metrics.get("mega_kernel", {})
    for wl_name, wl_metrics in workload_metrics.items():
        for key, mv in wl_metrics.items():
            unit = metric_units.get(key, "")
            name = key.split("|", 1)[1]
            formula = formula_by_key.get(key, "")
            if mv.value is None:
                mega_mv = mega.get(key)
                if (
                    wl_name != "mega_kernel"
                    and mega_mv is not None
                    and mega_mv.value is not None
                ):
                    mv.cls = "expected_na"
                else:
                    mv.cls = "na"
                continue
            if "percent" in unit.lower() and abs(mv.value) > 100:
                if name in COUNTER_SEMANTIC_OVERFLOW_NAMES:
                    mv.cls = "formula_overflow"
                else:
                    mv.cls = "overflow"
                continue
            mega_mv = mega.get(key)
            mega_val = mega_mv.value if mega_mv else None
            if abs(mv.value) < DELTA_NEAR_ZERO:
                if (
                    wl_name != "mega_kernel"
                    and mega_val is not None
                    and abs(mega_val) > DELTA_NEAR_ZERO
                ):
                    mv.cls = "workload_zero"
                elif all(
                    wl_metrics.get(key, MetricValue(None)).value is not None
                    and abs(wl_metrics.get(key, MetricValue(0.0)).value or 0)
                    < DELTA_NEAR_ZERO
                    for wl_metrics in workload_metrics.values()
                ):
                    mv.cls = classify_all_workload_zero(name, formula)
                else:
                    mv.cls = "good"
                continue
            mv.cls = "good"


def fmt_value(mv: Optional[MetricValue], unit: str = "") -> str:
    if mv is None or mv.value is None:
        return "N/A"
    return _fmt_number(mv.value, unit)


def fmt_std(mv: Optional[MetricValue], unit: str = "") -> str:
    """Format sample standard deviation, or a dash when it is unknown."""
    if mv is None or mv.std_v is None:
        return "—"
    return _fmt_number(mv.std_v, unit)


def fmt_cv(mv: Optional[MetricValue]) -> str:
    """Coefficient of variation, Std Dev / |Median|, as a percent."""
    if mv is None or mv.value is None or mv.std_v is None:
        return "—"
    if abs(mv.value) < DELTA_NEAR_ZERO:
        return "—"
    return f"{100.0 * mv.std_v / abs(mv.value):.1f}%"


def cell_style(cls: str) -> str:
    styles = {
        "good": "background:#c8e6c9;color:#1b5e20;font-weight:bold",
        "overflow": "background:#c62828;color:#fff;font-weight:bold",
        "formula_overflow": "background:#ff8f00;color:#fff;font-weight:bold",
        "workload_zero": "background:#b3e5fc;color:#01579b",
        "expected_na": "background:#d1c4e9;color:#4527a0",
        "na": "background:#e1bee7;color:#6a1b9a",
    }
    if cls in ZERO_BUCKET_CARD_STYLES:
        bg, fg = ZERO_BUCKET_CARD_STYLES[cls]
        weight = "font-weight:bold" if cls == "zero_sdk_bug" else ""
        return f"background:{bg};color:{fg};{weight}"
    return styles.get(cls, "")


def render_zero_sub_bucket_lines(summary: Dict[str, int]) -> str:
    zero_total = sum(summary.get(b, 0) for b in ZERO_SUB_BUCKETS)
    if zero_total == 0:
        return "<div style='color:#555'>⬜ Zero (all workloads): <b>0</b></div>"
    lines = [
        f"<div style='color:#555'>⬜ Zero (all workloads): <b>{zero_total}</b></div>",
    ]
    for bucket in ZERO_SUB_BUCKETS:
        count = summary.get(bucket, 0)
        if count == 0:
            continue
        bg, fg = ZERO_BUCKET_CARD_STYLES[bucket]
        label = ZERO_BUCKET_LABELS[bucket]
        lines.append(
            f"<div style='color:{fg};font-size:10px;padding-left:8px'>"
            f"▪ {html.escape(label)}: <b>{count}</b></div>"
        )
    return "".join(lines)


def count_yaml_metric_entries(config_arch: str) -> Tuple[int, int]:
    """Return (metric table entries, unique metric names) for analysis_configs."""
    metrics = list(
        iter_analysis_config_metrics(config.analysis_configs_dir / config_arch, set())
    )
    return len(metrics), len({metric.metric_name for metric in metrics})


def build_total_note(
    arch: str,
    config_arch: str,
    summary: Dict[str, Dict[str, int]],
    all_keys_count: int,
) -> str:
    """Arch-specific explanation of card Total vs YAML metric definitions."""
    yaml_entries, yaml_unique = count_yaml_metric_entries(config_arch)
    yaml_bit = (
        f"YAML metric definitions (~{yaml_entries} table entries / "
        f"~{yaml_unique} unique names in "
        f"<code>analysis_configs/{html.escape(config_arch)}</code>)"
    )
    totals = ", ".join(f"{html.escape(wl)}={s['total']}" for wl, s in summary.items())
    common_tail = (
        f"Each card Total counts every analyze-table row seen in any workload "
        f"(<b>{all_keys_count}</b> here; per-workload {totals}). "
        f"A metric with no value counts as N/A or Expected N/A and stays in Total. "
        f"Cell values are <b>{PRIMARY_STAT}</b> "
        f"(from analyze's Median, or recomputed from per-dispatch data). "
        f"Std Dev is the per-dispatch sample standard deviation. "
        f"CV is Std Dev / |Median|."
    )

    if arch in L2_CHANNEL_ROW_ARCHES:
        return (
            "<p style='font-size:11px;color:#555;margin:4px 0 10px;max-width:960px'>"
            "<b>Total</b> counts analyze-table rows (from the analyze CSV export), "
            f"not {yaml_bit}. "
            "Block 18 (L2 Cache per Channel) has one row per L2 channel: "
            "<code>$total_l2_chan</code>, which is "
            "<code>l2_banks × num_xcd</code> from the profile sysinfo. "
            "Total counts each of those rows. "
            f"{common_tail}"
            "</p>"
        )

    if config_arch == "gfx115x" or arch.startswith("gfx115"):
        return (
            "<p style='font-size:11px;color:#555;margin:4px 0 10px;max-width:960px'>"
            "<b>Total</b> counts analyze-table rows (from the analyze CSV export), "
            f"not {yaml_bit}. "
            "gfx115x has no per-channel Block 18 rows, so Total stays close to "
            "the YAML entry count. "
            f"{common_tail}"
            "</p>"
        )

    return (
        "<p style='font-size:11px;color:#555;margin:4px 0 10px;max-width:960px'>"
        "<b>Total</b> counts analyze-table rows (from the analyze CSV export), "
        f"not {yaml_bit}. "
        f"{common_tail}"
        "</p>"
    )


def build_report(
    primary_run: RunArtifacts,
    results: Dict[str, WorkloadMetrics],
    baseline: Optional[RunArtifacts],
    baseline_results: Optional[Dict[str, WorkloadMetrics]],
    formulas: Dict[str, Tuple[str, str, bool]],
    out_path: Path,
    report_date: str,
    host_label: str,
    arch: str,
    config_arch: str,
    iterations: Optional[int],
) -> None:
    """Compute the report data, render it, and write the HTML page."""
    data = compute_report(results, baseline_results, formulas)
    page = render_report(
        data,
        primary_run,
        baseline,
        formulas,
        report_date,
        host_label,
        arch,
        config_arch,
        iterations,
    )
    out_path.write_text(page, encoding="utf-8")


def compute_report(
    results: Dict[str, WorkloadMetrics],
    baseline_results: Optional[Dict[str, WorkloadMetrics]],
    formulas: Dict[str, Tuple[str, str, bool]],
) -> ReportData:
    """Classify metrics, count buckets, and compare against the baseline."""
    metric_units = {k: formulas.get(k, ("", "", False))[1] for k in formulas}
    formula_by_key = {k: formulas.get(k, ("", "", False))[0] for k in formulas}
    metrics: Dict[str, Dict[str, MetricValue]] = {
        wl: result.metrics for wl, result in results.items()
    }
    classify_metrics(metrics, metric_units, formula_by_key)
    data = ReportData(
        wl_names=list(results),
        metrics=metrics,
        dispatch_counts={wl: result.dispatch_count for wl, result in results.items()},
        all_keys=ordered_metric_keys(metrics),
    )
    data.summary = summarize_buckets(data.metrics, data.all_keys)
    if baseline_results is not None:
        base_metrics: Dict[str, Dict[str, MetricValue]] = {
            wl: result.metrics for wl, result in baseline_results.items()
        }
        classify_metrics(base_metrics, metric_units, formula_by_key)
        compare_runs(data, base_metrics)
    return data


def ordered_metric_keys(metrics: Dict[str, Dict[str, MetricValue]]) -> List[str]:
    """Return every metric key once, in Metric_ID order per workload."""
    keys: List[str] = []
    seen: Set[str] = set()
    for wl_metrics in metrics.values():
        for key in sorted(wl_metrics, key=_metric_id_sort_key):
            if key not in seen:
                seen.add(key)
                keys.append(key)
    return keys


def summarize_buckets(
    metrics: Dict[str, Dict[str, MetricValue]],
    all_keys: List[str],
) -> Dict[str, Dict[str, int]]:
    """Count each workload's metrics per bucket.

    A metric missing from one workload counts as N/A there, so every Total
    equals len(all_keys).
    """
    summary: Dict[str, Dict[str, int]] = {
        wl: {
            "total": 0,
            "good": 0,
            "workload_zero": 0,
            "expected_na": 0,
            "na": 0,
            "overflow": 0,
            "formula_overflow": 0,
            **{bucket: 0 for bucket in ZERO_SUB_BUCKETS},
        }
        for wl in metrics
    }
    for key in all_keys:
        for wl, wl_metrics in metrics.items():
            mv = wl_metrics.get(key)
            summary[wl]["total"] += 1
            if mv is None:
                summary[wl]["na"] += 1
                continue
            if mv.cls in summary[wl]:
                summary[wl][mv.cls] += 1
    return summary


def compare_runs(
    data: ReportData,
    base_metrics: Dict[str, Dict[str, MetricValue]],
) -> None:
    """Fill the overflow fixes, value deltas, and candidate-missing rows.

    Keys and workloads are the union of both runs, so a metric the baseline
    reports and the candidate lost shows up as missing instead of vanishing.
    """
    workloads = data.wl_names + [wl for wl in base_metrics if wl not in data.metrics]
    candidate_keys = set(data.all_keys)
    keys = data.all_keys + [
        key for key in ordered_metric_keys(base_metrics) if key not in candidate_keys
    ]
    for key in keys:
        for wl in workloads:
            old = base_metrics.get(wl, {}).get(key)
            if old is None:
                continue
            new = data.metrics.get(wl, {}).get(key)
            if new is None:
                data.missing.append(MetricChange(wl, key, old, None))
                continue
            if is_overflow_fix(old, new):
                data.improvements.append(MetricChange(wl, key, old, new))
            if old.value is None or new.value is None:
                continue
            # rel is None when the baseline is near zero.
            rel = relative_abs_delta(old.value, new.value)
            if rel is None or rel < DELTA_REL_THRESHOLD:
                continue
            data.deltas.append(MetricChange(wl, key, old, new, rel))


def is_overflow_fix(old: MetricValue, new: MetricValue) -> bool:
    """True when the baseline overflowed and the candidate has a real value."""
    return (
        old.cls in OVERFLOW_CLASSES
        and new.value is not None
        and new.cls not in OVERFLOW_CLASSES
    )


def render_report(
    data: ReportData,
    primary_run: RunArtifacts,
    baseline: Optional[RunArtifacts],
    formulas: Dict[str, Tuple[str, str, bool]],
    report_date: str,
    host_label: str,
    arch: str,
    config_arch: str,
    iterations: Optional[int],
) -> str:
    """Render computed report data as one self-contained HTML page."""
    wl_names = data.wl_names
    cards = [
        _render_card(wl, data.summary[wl], data.dispatch_counts.get(wl))
        for wl in wl_names
    ]
    total_note = build_total_note(arch, config_arch, data.summary, len(data.all_keys))

    body_rows = [_render_metric_row(data, key, formulas) for key in data.all_keys]

    wl_headers = "".join(
        f"<th>{html.escape(wl)}<br>"
        f"<small>{WORKLOAD_DESCRIPTIONS.get(wl, '')}</small><br>"
        f"<small>{PRIMARY_STAT}</small><br>"
        "<small>σ Std Dev</small><br><small>CV</small></th>"
        for wl in wl_names
    )

    iter_banner = _iteration_banner(iterations, data.dispatch_counts)
    improve_section = ""
    if baseline is not None:
        improve_section = _render_comparison_sections(
            data, formulas, baseline.label, primary_run.label
        )

    primary_label = "Candidate" if baseline is not None else "Run"
    runs_html = (
        f" | <b>{primary_label}:</b> {html.escape(primary_run.label)} "
        f"(<code>{html.escape(str(primary_run.artifact_dir))}</code>)"
    )
    if baseline is not None:
        runs_html += (
            f" | <b>Baseline:</b> {html.escape(baseline.label)} "
            f"(<code>{html.escape(str(baseline.artifact_dir))}</code>)"
        )

    doc = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>{html.escape(arch)} Metric Health {html.escape(report_date)}</title>
<style>
body{{font-family:'Segoe UI',Arial,sans-serif;background:#eceff1;margin:0;padding:16px;
font-size:13px}}
h1{{color:#0d47a1;margin-bottom:4px}}h2{{color:#263238;border-bottom:2px solid #1565c0;
padding-bottom:4px;margin-top:22px}}
.cards{{display:flex;gap:12px;flex-wrap:wrap;margin:8px 0}}
.card{{background:#fff;border-radius:8px;padding:10px 16px;box-shadow:0 2px 6px
rgba(0,0,0,.12);min-width:140px}}
.card h3{{margin:0 0 3px;font-size:13px;color:#555}}.desc{{font-size:10px;color:#888;
margin-bottom:4px}}
.improve-table{{border-collapse:collapse;width:100%;background:#fff;margin-top:6px;
font-size:12px}}
.improve-table th{{background:#1565c0;color:#fff;padding:5px 8px;text-align:left}}
.improve-table td{{padding:3px 8px;border-bottom:1px solid #eee}}
.sortable-table th{{cursor:pointer;user-select:none;white-space:nowrap}}
.sortable-table th:hover{{background:#0d47a1}}
.sortable-table th.sort-asc::after{{content:' \\25b2';font-size:10px}}
.sortable-table th.sort-desc::after{{content:' \\25bc';font-size:10px}}
tr.delta-ge-100 td{{background:#fff8e1}}
tr.delta-ge-100 td:nth-child(10){{background:#ffe082;font-weight:bold}}
tr.delta-ge-100:hover td{{filter:brightness(0.96)}}
.bucket-semantics{{border-collapse:collapse;width:100%;max-width:920px;background:#fff;
margin:10px 0 4px;font-size:12px;box-shadow:0 1px 4px rgba(0,0,0,.08)}}
.bucket-semantics th{{background:#eceff1;color:#37474f;padding:8px 10px;text-align:left;
font-weight:600;position:static}}
.bucket-semantics td{{padding:7px 10px;border-bottom:1px solid #e0e0e0;
vertical-align:top}}
.bucket-semantics tr:last-child td{{border-bottom:none}}
.bucket-semantics td:first-child{{font-weight:600;white-space:nowrap}}
.legend{{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0;user-select:none}}
.filter-note{{font-size:11px;color:#888;margin:-4px 0 4px}}
.tbl-wrap{{overflow-x:auto;max-height:65vh;overflow-y:auto}}
table{{border-collapse:collapse;width:100%;background:#fff;font-size:12px}}
th{{background:#37474f;color:#fff;padding:6px 8px;text-align:left;position:sticky;top:0;
z-index:2;white-space:nowrap}}
td{{padding:3px 5px;border-bottom:1px solid #e0e0e0;vertical-align:middle}}
tr:hover td{{filter:brightness(0.91)}}
.mid{{font-family:monospace;color:#546e7a;font-size:11px}}.mname{{max-width:180px}}
.unit{{color:#90a4ae;font-style:italic;font-size:10px}}
tr.row-overflow{{box-shadow:inset 0 0 0 2px #c62828}}
</style></head><body>
<h1>{html.escape(arch)} Metric Health Report</h1>
<p style="margin:2px 0;color:#607d8b;font-size:12px"><b>Date:</b>
{html.escape(report_date)}
 | <b>Host:</b> {html.escape(host_label)}
{runs_html}
 | <b>Arch:</b> {html.escape(arch)}
 | <b>Workloads:</b> {html.escape(", ".join(wl_names))}</p>
{iter_banner}
<p style="font-size:11px;color:#555">Analysis configs: <code>{html.escape(config_arch)}
</code>.
All-workload zeros are split into sub-buckets (healthy / optional path / SDK bug suspect
/ other).
Expected zero and Expected N/A compare the other workloads with mega_kernel when
present.</p>
<h2>Summary</h2>
{total_note}
<div class="cards">{"".join(cards)}</div>
<h2>Bucket semantics</h2>
<p style="font-size:11px;color:#555;margin:2px 0 6px">Each metric in each workload
column is assigned to <b>exactly one</b> bucket; counts on each card sum to Total.</p>
<table class="bucket-semantics">
<thead><tr><th>Bucket</th><th>Meaning</th><th>Healthy?</th></tr></thead>
<tbody>
<tr><td>Good</td><td>Value exists; not N/A; not &gt;100% (percent); not classified as
zero</td><td>Usually yes</td></tr>
<tr><td>Zero: healthy (<code>zero_healthy</code>)</td><td>Zero in every workload; metric
is a stall/failure/alloc-failure rate where 0% is good</td><td>Yes</td></tr>
<tr><td>Zero: optional path (<code>zero_optional</code>)</td><td>Zero in every workload;
precision or IP path mega_kernel does not exercise (FP64, etc.)</td><td>Usually yes</td>
</tr>
<tr><td>Zero: SDK bug suspect (<code>zero_sdk_bug</code>)</td><td>Zero in every workload
but mega_kernel exercises the path or counters should fire (VALU FLOPs, DRAM read)</td>
<td>Investigate; report it to the SDK team</td></tr>
<tr><td>Zero: other (<code>zero_other</code>)</td><td>Zero in every workload; no
category matched (often paths with little traffic)</td><td>Review case-by-case</td></tr>
<tr><td>Expected zero (<code>workload_zero</code>)</td><td>Zero here, but non-zero in
mega_kernel</td><td>Yes; the workload doesn't use that path</td></tr>
<tr><td>Expected N/A (<code>expected_na</code>)</td><td>N/A here (for example a zero
denominator), but mega_kernel has a value</td><td>Yes; the workload doesn't use that
path</td></tr>
<tr><td>N/A</td><td>Analyze couldn't produce a value and mega_kernel also has no value
</td><td>Investigate</td></tr>
<tr><td>&gt;100% formula (<code>formula_overflow</code>)</td><td>Percent &gt;100% that
the counters' known behavior explains (overlapping busy signals, pipeline slot counting)
</td><td>Usually benign; read the trend, not the absolute %</td></tr>
<tr><td>&gt;100% hardware (<code>overflow</code>)</td><td>Percent &gt;100% that known
behavior does not explain; likely a bad formula or counter bug</td><td>Investigate</td>
</tr>
</tbody></table>
{improve_section}
<h2>All metrics: Blocks 2-18</h2>
<p style="font-size:11px;color:#555;margin:4px 0">Column values are <b>{PRIMARY_STAT}
</b>.</p>
<div class="legend">
<div class="leg" data-cls="good" style="background:#c8e6c9;color:#1b5e20;
font-weight:bold;padding:5px 12px;border-radius:5px;cursor:pointer;font-size:12px">▲
Good (real data)</div>
<div class="leg" data-cls="overflow" style="background:#c62828;color:#fff;
font-weight:bold;padding:5px 12px;border-radius:5px;cursor:pointer;font-size:12px">🚨
&gt;100% hardware</div>
<div class="leg" data-cls="formula_overflow" style="background:#ff8f00;color:#fff;
font-weight:bold;padding:5px 12px;border-radius:5px;cursor:pointer;font-size:12px">📐
&gt;100% formula</div>
<div class="leg" data-cls="zero_healthy" style="background:#e8f5e9;color:#2e7d32;
padding:5px 12px;border-radius:5px;cursor:pointer;font-size:12px">✓ Zero: healthy (0%
stall)</div>
<div class="leg" data-cls="zero_optional" style="background:#e3f2fd;color:#1565c0;
padding:5px 12px;border-radius:5px;cursor:pointer;font-size:12px">○ Zero: optional path
</div>
<div class="leg" data-cls="zero_sdk_bug" style="background:#ffcdd2;color:#c62828;
font-weight:bold;padding:5px 12px;border-radius:5px;cursor:pointer;font-size:12px">❌
Zero: SDK bug suspect</div>
<div class="leg" data-cls="zero_other" style="background:#f5f5f5;color:#616161;
padding:5px 12px;border-radius:5px;cursor:pointer;font-size:12px">⬜ Zero: other</div>
<div class="leg" data-cls="workload_zero" style="background:#b3e5fc;color:#01579b;
padding:5px 12px;border-radius:5px;cursor:pointer;font-size:12px">⚪ Expected zero
(non-zero in mega)</div>
<div class="leg" data-cls="expected_na" style="background:#d1c4e9;color:#4527a0;
padding:5px 12px;border-radius:5px;cursor:pointer;font-size:12px">🔹 Expected N/A (valid
in mega)</div>
<div class="leg" data-cls="na" style="background:#e1bee7;color:#6a1b9a;padding:5px 12px;
border-radius:5px;cursor:pointer;font-size:12px">⚠️ N/A (investigate)</div>
</div>
<p class="filter-note">💡 Click legend to show/hide rows by category</p>
<div class="tbl-wrap"><table id="mt">
<thead><tr><th>ID</th><th>Metric Name</th><th>Unit</th><th>LEVEL<br><small>if
problematic</small></th><th>Formula (scroll →)</th>{wl_headers}</tr></thead>
<tbody>{"".join(body_rows)}</tbody></table></div>
<script>
const active=new
Set(['good','overflow','formula_overflow','zero_healthy','zero_optional','zero_sdk_bug','zero_other','workload_zero','expected_na','na']);
document.querySelectorAll('.leg').forEach(el=>{{
  el.addEventListener('click',()=>{{
    const c=el.dataset.cls;
    if(active.has(c)) active.delete(c); else active.add(c);
    el.style.opacity=active.has(c)?'1':'0.35';
    document.querySelectorAll('#mt tbody tr').forEach(row=>{{
      const cats=(row.dataset.cats||'').split(/\\s+/);
      row.style.display=cats.some(x=>active.has(x))?'':'none';
    }});
  }});
}});
document.querySelectorAll('table.sortable-table').forEach(table=>{{
  const headers=[...table.querySelectorAll('thead th')];
  headers.forEach((th, colIdx)=>{{
    th.addEventListener('click',()=>{{
      const tbody=table.tBodies[0];
      if(!tbody) return;
      const type=th.dataset.sort||'str';
      const rows=[...tbody.querySelectorAll('tr')];
      const asc=!th.classList.contains('sort-asc');
      headers.forEach(h=>h.classList.remove('sort-asc','sort-desc'));
      th.classList.add(asc?'sort-asc':'sort-desc');
      const parse=(text)=>{{
        const t=(text||'').trim().replace(/,/g,'');
        if(type==='num'){{
          const n=parseFloat(t.replace(/%$/,''));
          return Number.isFinite(n)?n:null;
        }}
        return t.toLowerCase();
      }};
      rows.sort((a,b)=>{{
        const av=parse(a.children[colIdx]?.textContent);
        const bv=parse(b.children[colIdx]?.textContent);
        if(av===null && bv===null) return 0;
        if(av===null) return 1;
        if(bv===null) return -1;
        if(av<bv) return asc?-1:1;
        if(av>bv) return asc?1:-1;
        return 0;
      }});
      rows.forEach(r=>tbody.appendChild(r));
    }});
  }});
}});
</script></body></html>"""

    return doc


def main() -> None:
    arg_parser = _build_arg_parser()
    args = arg_parser.parse_args()
    paths = [path for path_group in args.paths for path in path_group]
    if len(paths) > 2:
        arg_parser.error(
            "-p/--path accepts at most two paths: baseline first, candidate second"
        )

    baseline = discover_run(paths[0], args.baseline_label)
    candidate = (
        discover_run(paths[1], args.candidate_label) if len(paths) == 2 else None
    )
    primary_run = candidate or baseline
    _sys_info, arch = read_sysinfo(next(iter(primary_run.profile_dirs.values())))
    if candidate is not None:
        _sys_info, baseline_arch = read_sysinfo(
            next(iter(baseline.profile_dirs.values()))
        )
        if baseline_arch != arch:
            raise SystemExit(f"Runs profiled different archs: {baseline_arch}, {arch}")

    results = load_run_metrics(primary_run)
    formulas = collect_formulas(primary_run, results)
    baseline_results: Optional[Dict[str, WorkloadMetrics]] = None
    if candidate is not None:
        baseline_results = load_run_metrics(baseline)
        # Baseline-only metrics still need their unit in the missing section.
        formulas = {**collect_formulas(baseline, baseline_results), **formulas}
    build_report(
        primary_run,
        results,
        baseline if candidate is not None else None,
        baseline_results,
        formulas,
        args.out,
        args.date,
        args.host,
        arch,
        canonical_config_arch(arch) or arch,
        args.iterations,
    )
    print(f"Wrote {args.out}")


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise SystemExit(f"Analyze CSV file not found: {path}")
    with path.open(encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def _replace_reducible_aggregates(expression: str) -> str:
    """Replace each to_sum(x) or to_avg(x) with (x)."""
    pieces: List[str] = []
    index = 0
    while index < len(expression):
        matched = ""
        for name in ("to_sum", "to_avg"):
            prefix = f"{name}("
            if expression.startswith(prefix, index):
                matched = prefix
                break
        if not matched:
            pieces.append(expression[index])
            index += 1
            continue
        operand_start = index + len(matched)
        depth = 1
        operand_end = operand_start
        while operand_end < len(expression) and depth:
            if expression[operand_end] == "(":
                depth += 1
            elif expression[operand_end] == ")":
                depth -= 1
            operand_end += 1
        if depth:
            return expression
        operand = expression[operand_start : operand_end - 1]
        pieces.append(f"({_replace_reducible_aggregates(operand)})")
        index = operand_end
    return "".join(pieces)


def _attach_stat_column(dfs: dict, dfs_type: dict, statistic: str) -> None:
    """Add a per-dispatch statistic expression when that column is absent."""
    column = "Median" if statistic == "median" else "Std Dev"
    for _df_id, frame in dfs.items():
        if dfs_type.get(_df_id) != "metric_table" or column in frame.columns:
            continue
        source_col = next(
            (
                name
                for name in ("Median", "Min", "Avg", "Value")
                if name in frame.columns
            ),
            None,
        )
        if source_col is None:
            continue
        frame[column] = [
            per_dispatch_stat_expr(value, statistic) for value in frame[source_col]
        ]


def _read_stat_column(arch_config: object, column: str) -> Dict[str, Optional[float]]:
    """Read one evaluated statistic column as a dict from metric key to float."""
    out: Dict[str, Optional[float]] = {}
    for df_id, frame in arch_config.dfs.items():
        if arch_config.dfs_type.get(df_id) != "metric_table":
            continue
        if column not in frame.columns or "Metric" not in frame.columns:
            continue
        for row_id, row in frame.iterrows():
            key = f"{row_id}|{row['Metric']}"
            cell = row[column]
            if isinstance(cell, str):
                out[key] = to_float(cell)
            elif cell is None or (isinstance(cell, float) and math.isnan(cell)):
                out[key] = None
            else:
                try:
                    out[key] = float(cell)
                except (TypeError, ValueError):
                    out[key] = None
    return out


def _fmt_number(value: float, unit: str = "") -> str:
    if "percent" in unit.lower():
        return f"{value:.2f}".rstrip("0").rstrip(".")
    if abs(value) >= 1000:
        return f"{value:,.2f}".rstrip("0").rstrip(".")
    return f"{value:.2g}"


def _iteration_banner(
    iterations: Optional[int],
    dispatch_counts: Dict[str, Optional[int]],
) -> str:
    """Build the header line with the statistic, iterations and dispatch counts."""
    parts = [f"<b>Statistic:</b> {PRIMARY_STAT}"]
    if iterations is not None:
        parts.append(f"<b>Requested iterations:</b> {iterations}")
    count_bits = [
        f"{html.escape(wl)}={count}"
        for wl, count in dispatch_counts.items()
        if count is not None
    ]
    if count_bits:
        parts.append("<b>Top-kernel dispatches:</b> " + ", ".join(count_bits))
    return (
        '<p style="margin:2px 0;color:#37474f;font-size:12px">'
        + " | ".join(parts)
        + "</p>"
    )


def _metric_id_sort_key(key: str) -> List[int]:
    return [int(part) for part in key.split("|")[0].split(".")]


def _render_card(wl: str, s: Dict[str, int], n_disp: Optional[int]) -> str:
    disp_line = (
        f"<div style='font-size:10px;color:#546e7a'>dispatches: <b>{n_disp}</b></div>"
        if n_disp is not None
        else ""
    )
    return (
        f"<div class='card'><h3>{html.escape(wl)}</h3>"
        f"<div class='desc'>{WORKLOAD_DESCRIPTIONS.get(wl, '')}</div>"
        f"{disp_line}"
        f"<div>Total: <b>{s['total']}</b></div>"
        f"<div style='color:#1b5e20'>✅ Good: {s['good']}</div>"
        f"{render_zero_sub_bucket_lines(s)}"
        f"<div style='color:#01579b'>⚪ Expected zero: {s['workload_zero']}</div>"
        f"<div style='color:#4527a0'>🔹 Expected N/A: {s['expected_na']}</div>"
        f"<div style='color:#e65100'>⚠️ N/A: {s['na']}</div>"
        f"<div style='color:#ff8f00'>📐 &gt;100% formula: {s['formula_overflow']}</div>"
        f"<div style='color:#c62828'>🚨 &gt;100% hardware: {s['overflow']}</div></div>"
    )


def _render_metric_row(
    data: ReportData,
    key: str,
    formulas: Dict[str, Tuple[str, str, bool]],
) -> str:
    mid, name = key.split("|", 1)
    formula, unit, level = formulas.get(key, ("", "", False))
    row_cats: Set[str] = set()
    wl_cells = []
    for wl in data.wl_names:
        mv = data.metrics[wl].get(key)
        cls = mv.cls if mv else "na"
        row_cats.add(cls)
        val = fmt_value(mv, unit)
        std = fmt_std(mv, unit)
        cv = fmt_cv(mv)
        style = cell_style(cls)
        std_html = f"<div class='unit'>σ {html.escape(std)}</div>" if std != "—" else ""
        cv_html = f"<div class='unit'>CV {html.escape(cv)}</div>" if cv != "—" else ""
        wl_cells.append(
            f"<td style='{style}' data-cls='{cls}'>"
            f"{html.escape(val)}{std_html}{cv_html}</td>"
        )

    problematic = row_cats & {
        "na",
        "overflow",
        "formula_overflow",
        *ZERO_SUB_BUCKETS,
    }
    if problematic:
        level_cell = (
            "<b style='color:#6a1b9a'>Yes</b>"
            if level
            else "<span style='color:#546e7a'>No</span>"
        )
    else:
        level_cell = "—"

    formula_html = html.escape(f"value: {formula}") if formula else "—"
    return (
        f"<tr data-cats='{' '.join(sorted(row_cats))}'>"
        f"<td class='mid'>{html.escape(mid)}</td>"
        f"<td class='mname'>{html.escape(name)}</td>"
        f"<td class='unit'>{html.escape(unit)}</td>"
        f"<td style='text-align:center;font-size:11px'>{level_cell}</td>"
        "<td><div style='overflow-x:auto;white-space:nowrap;max-width:600px;"
        "font-size:10px;"
        f"font-family:monospace;background:#f0f4f8;color:#1a237e;padding:2px 5px;"
        f"border-radius:3px;border:1px solid #c5cae9'>{formula_html}</div></td>"
        + "".join(wl_cells)
        + "</tr>"
    )


def _render_comparison_sections(
    data: ReportData,
    formulas: Dict[str, Tuple[str, str, bool]],
    baseline_label: str,
    candidate_label: str,
) -> str:
    sections = ""
    if data.improvements:
        improve_rows = [
            _render_improvement_row(change, formulas) for change in data.improvements
        ]
        sections += (
            f"<h2>Overflow improvements vs {html.escape(baseline_label)} "
            f"({len(improve_rows)} metrics fixed)</h2>"
            "<table class='improve-table'><thead><tr><th>Workload</th><th>ID</th>"
            f"<th>Metric</th><th>Old ({PRIMARY_STAT})</th>"
            f"<th>New ({PRIMARY_STAT})</th><th>Unit</th></tr></thead><tbody>"
            + "".join(improve_rows)
            + "</tbody></table>"
        )
    if data.deltas:
        sections += _render_delta_section(data, formulas, baseline_label)
    if data.missing:
        missing_rows = [
            _render_missing_row(change, formulas) for change in data.missing
        ]
        n_missing_metrics = len({change.key for change in data.missing})
        sections += (
            f"<h2>Missing from {html.escape(candidate_label)} vs "
            f"{html.escape(baseline_label)} ({len(missing_rows)} rows, "
            f"{n_missing_metrics} metrics)</h2>"
            "<p class='filter-note'>The baseline reports these metrics for the "
            "workload, but the candidate analyze output has no row for them.</p>"
            "<table class='improve-table'><thead><tr><th>Workload</th><th>ID</th>"
            f"<th>Metric</th><th>Old ({PRIMARY_STAT})</th>"
            "<th>New</th><th>Unit</th></tr></thead><tbody>"
            + "".join(missing_rows)
            + "</tbody></table>"
        )
    return sections


def _render_delta_section(
    data: ReportData,
    formulas: Dict[str, Tuple[str, str, bool]],
    baseline_label: str,
) -> str:
    delta_rows = [_render_delta_row(change, formulas) for change in data.deltas]
    n_hi = sum(
        1 for change in data.deltas if (change.rel or 0) >= DELTA_HIGHLIGHT_THRESHOLD
    )
    n_delta = len(delta_rows)
    # Count each metric once, not once per workload.
    n_changed = len({change.key for change in data.deltas})
    n_metrics = len(data.all_keys)
    changed_pct = (100.0 * n_changed / n_metrics) if n_metrics else 0.0
    rel_pct = DELTA_REL_THRESHOLD * 100
    hi_pct = DELTA_HIGHLIGHT_THRESHOLD * 100
    near_zero = f"{DELTA_NEAR_ZERO:g}"
    counts = f"{n_delta} rows, {n_changed} of {n_metrics} metrics ({changed_pct:.1f}%)"
    return (
        f"<h2>Value deltas vs {html.escape(baseline_label)} "
        f"(|rel| ≥ {rel_pct:.0f}%, {counts})</h2>"
        "<p class='filter-note'>Click a column header to sort "
        "(Kernel name, ID, Metric, |rel|, and so on). "
        f"Compared values are <b>{PRIMARY_STAT}</b> from the analyze CSV export "
        "(or recomputed from per-dispatch "
        "data when the Median column is absent). "
        "|rel| = |new-old|/|old|. "
        "Std Dev is the per-dispatch sample standard deviation. "
        "CV = Std Dev / |Median|. "
        "A dash means Std Dev is missing or the median is near zero.</p>"
        "<p class='filter-note'>"
        f"Include when |rel| ≥ {rel_pct:.0f}% ({counts})<br>"
        f"Highlight when |rel| ≥ {hi_pct:.0f}% "
        f"({n_hi} rows)<br>"
        f"Skip when baseline |old| &lt; {near_zero} "
        "(|rel| undefined)"
        "</p>"
        "<table class='improve-table sortable-table' id='delta-table'>"
        "<thead><tr>"
        "<th data-sort='str'>Kernel</th>"
        "<th data-sort='str'>ID</th>"
        "<th data-sort='str'>Metric</th>"
        f"<th data-sort='num'>Old ({PRIMARY_STAT})</th>"
        "<th data-sort='num'>Old (Std Dev)</th>"
        "<th data-sort='num'>Old (CV)</th>"
        f"<th data-sort='num'>New ({PRIMARY_STAT})</th>"
        "<th data-sort='num'>New (Std Dev)</th>"
        "<th data-sort='num'>New (CV)</th>"
        "<th data-sort='num'>|rel|</th>"
        "<th data-sort='str'>Unit</th>"
        "</tr></thead><tbody>" + "".join(delta_rows) + "</tbody></table>"
    )


def _render_improvement_row(
    change: MetricChange, formulas: Dict[str, Tuple[str, str, bool]]
) -> str:
    mid, name = change.key.split("|", 1)
    unit = formulas.get(change.key, ("", "", False))[1]
    return (
        f"<tr><td>{html.escape(change.workload)}</td>"
        f"<td class='mid'>{html.escape(mid)}</td>"
        f"<td>{html.escape(name)}</td>"
        f"<td style='background:#ffcdd2;font-weight:bold'>"
        f"{html.escape(fmt_value(change.old, unit))}</td>"
        f"<td style='background:#c8e6c9;font-weight:bold'>"
        f"{html.escape(fmt_value(change.new, unit))}</td>"
        f"<td>{html.escape(unit)}</td></tr>"
    )


def _render_delta_row(
    change: MetricChange, formulas: Dict[str, Tuple[str, str, bool]]
) -> str:
    mid, name = change.key.split("|", 1)
    unit = formulas.get(change.key, ("", "", False))[1]
    rel = change.rel or 0.0
    rel_pct = rel * 100.0
    hi = rel >= DELTA_HIGHLIGHT_THRESHOLD
    row_cls = " class='delta-ge-100'" if hi else ""
    rel_cell = (
        f"<td style='background:#ffe082;font-weight:bold'>{rel_pct:.1f}%</td>"
        if hi
        else f"<td>{rel_pct:.1f}%</td>"
    )
    return (
        f"<tr{row_cls}><td>{html.escape(change.workload)}</td>"
        f"<td class='mid'>{html.escape(mid)}</td>"
        f"<td>{html.escape(name)}</td>"
        f"<td>{html.escape(fmt_value(change.old, unit))}</td>"
        f"<td>{html.escape(fmt_std(change.old, unit))}</td>"
        f"<td>{html.escape(fmt_cv(change.old))}</td>"
        f"<td>{html.escape(fmt_value(change.new, unit))}</td>"
        f"<td>{html.escape(fmt_std(change.new, unit))}</td>"
        f"<td>{html.escape(fmt_cv(change.new))}</td>"
        f"{rel_cell}"
        f"<td>{html.escape(unit)}</td></tr>"
    )


def _render_missing_row(
    change: MetricChange, formulas: Dict[str, Tuple[str, str, bool]]
) -> str:
    mid, name = change.key.split("|", 1)
    unit = formulas.get(change.key, ("", "", False))[1]
    return (
        f"<tr><td>{html.escape(change.workload)}</td>"
        f"<td class='mid'>{html.escape(mid)}</td>"
        f"<td>{html.escape(name)}</td>"
        f"<td>{html.escape(fmt_value(change.old, unit))}</td>"
        "<td style='background:#e1bee7;font-weight:bold'>missing</td>"
        f"<td>{html.escape(unit)}</td></tr>"
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    arg_parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    arg_parser.add_argument(
        "-p",
        "--path",
        dest="paths",
        type=Path,
        required=True,
        metavar="PATH",
        nargs="+",
        action="append",
        help=(
            "Run directories: baseline first; optional candidate second. "
            "Accepts at most two paths."
        ),
    )
    arg_parser.add_argument("--candidate-label", default=DEFAULT_CANDIDATE_LABEL)
    arg_parser.add_argument("--baseline-label", default=DEFAULT_BASELINE_LABEL)
    arg_parser.add_argument("--out", type=Path, required=True)
    arg_parser.add_argument("--date", default=date.today().isoformat())
    arg_parser.add_argument("--host", default="unknown")
    arg_parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Requested workload iterations (shown in report header)",
    )
    return arg_parser


if __name__ == "__main__":
    main()
