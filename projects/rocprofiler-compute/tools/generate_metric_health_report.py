#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT
# ruff: noqa: E501

"""Generate metric health HTML report from rocprof-compute analyze logs.

Analyze logs must be produced with ``--view table`` so chart-only panels (roofline,
memory charts, etc.) are emitted as plain metric tables. Default TTY output omits
metrics and yields incomplete coverage.

Primary statistic is **Median** (not Avg):

* Prefer the Median column when present in ``--view table`` output.
* Else recompute Median from per-dispatch data via ``--workload-dir`` (same
  ``MEDIAN(<min-inner>)`` path as ``tools/compare_spp_legacy_medians.py``).
* Else, if Min==Max, treat that value as the Median.
* Value-only panels (no Avg/Min/Max/Median) use the single Value column.

Workloads are passed as ``name:path`` pairs. Optional ``--baseline`` logs enable
SPP-vs-legacy overflow improvements and value-delta tables.

Arch → analysis_configs mapping (``--arch``):

* gfx942 / gfx950 / gfx90a / gfx908 → same-named config dir
* gfx1150 / gfx1151 / gfx1152 / gfx1153 → ``gfx115x``
* gfx1250 → ``gfx1250``
"""

from __future__ import annotations

import argparse
import html
import importlib.util
import re
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

# CLI arch → analysis_configs/<dir>
ARCH_CONFIG_DIR: dict[str, str] = {
    "gfx908": "gfx908",
    "gfx90a": "gfx90a",
    "gfx940": "gfx940",
    "gfx941": "gfx941",
    "gfx942": "gfx942",
    "gfx950": "gfx950",
    "gfx1150": "gfx115x",
    "gfx1151": "gfx115x",
    "gfx1152": "gfx115x",
    "gfx1153": "gfx115x",
    "gfx115x": "gfx115x",
    "gfx1250": "gfx1250",
}

ARCH_TITLE: dict[str, str] = {
    "gfx908": "MI100 / gfx908",
    "gfx90a": "MI250 / gfx90a",
    "gfx942": "MI300 / gfx942",
    "gfx950": "MI350 / gfx950",
    "gfx1150": "gfx1150",
    "gfx1151": "gfx1151 Strix Halo",
    "gfx1152": "gfx1152",
    "gfx1153": "gfx1153 Krackan2",
    "gfx115x": "gfx115x",
    "gfx1250": "MI450 / gfx1250",
}

# Relative change |new-old|/|old| above this → include in delta table.
DELTA_REL_THRESHOLD = 0.05
# |rel| at or above this → highlight the delta row / |rel| cell.
DELTA_HIGHLIGHT_THRESHOLD = 1.0
# Baselines below this make |rel| undefined; those pairs are skipped.
DELTA_NEAR_ZERO = 1e-12

ANSI = re.compile(r"\x1b\[[0-9;]*m")
ROW_RE = re.compile(
    r"^\s*[│|]\s*"
    r"([0-9]+(?:\.[0-9]+)*)\s*[│|]\s*"
    r"([^│|]+?)\s*[│|]\s*"
    r"(.+)$"
)
SUPPRESS_RE = re.compile(
    r"Not showing table with empty column\(s\):\s*(.+)$", re.IGNORECASE
)
LEVEL_COUNTER_RE = re.compile(r"_LEVEL(?:_sum)?|INFLIGHT_LEVEL", re.IGNORECASE)
COUNT_RE = re.compile(r"^\s*[│|]\s*0\s*[│|].*?[│|]\s*([0-9]+(?:\.[0-9]+)?)\s*[│|]")

COUNTER_SEMANTIC_OVERFLOW_NAMES = frozenset({
    "CP Utilization",
    "Pipeline Utilization - CMACC",
    "Pipeline Utilization - SMACC",
    "Pipeline Utilization - Double Precision",
    "Pipeline Utilization - XDL",
    "SPI Utilization",
})

ZERO_SUB_BUCKETS: tuple[str, ...] = (
    "zero_healthy",
    "zero_optional",
    "zero_sdk_bug",
    "zero_other",
)

HEALTHY_ZERO_NAME_FRAGMENTS: tuple[str, ...] = (
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

OPTIONAL_ZERO_NAME_FRAGMENTS: tuple[str, ...] = (
    "FP64",
    "FP8",
    "FP6",
    "FP4",
    "Int4",
    "Sparse",
    "VALU FLOPs (F64)",
    "Double Precision",
)

SDK_BUG_ZERO_NAME_FRAGMENTS: tuple[str, ...] = (
    "VALU FLOPs",
    "VALU IOPs",
    "VALU Trans FLOPs",
    "WMMA FLOPs",
    "DRAM Bandwidth - Read",
    "GL2 to DRAM Read",
    "GL2 to DRAM Read Latency",
    "Buffer Atomics Instructions",
)

ZERO_BUCKET_LABELS: dict[str, str] = {
    "zero_healthy": "Healthy zero (0% stall/failure)",
    "zero_optional": "Optional path (not exercised)",
    "zero_sdk_bug": "SDK / counter bug suspect",
    "zero_other": "Other all-workload zero",
}

ZERO_BUCKET_CARD_STYLES: dict[str, tuple[str, str]] = {
    "zero_healthy": ("#e8f5e9", "#2e7d32"),
    "zero_optional": ("#e3f2fd", "#1565c0"),
    "zero_sdk_bug": ("#ffcdd2", "#c62828"),
    "zero_other": ("#f5f5f5", "#616161"),
}


@dataclass
class MetricDef:
    metric_id: str
    name: str
    unit: str
    formula: str
    panel_block: int
    uses_level_counter: bool


@dataclass
class MetricValue:
    """Primary health value is Median (never Avg)."""

    value: float | None
    raw: str
    source: str = ""  # median | recomputed | minmax_eq | value
    min_v: float | None = None
    max_v: float | None = None
    cls: str = ""


@dataclass
class ParsedLog:
    metrics: dict[str, MetricValue] = field(default_factory=dict)
    suppressed: list[str] = field(default_factory=list)
    dispatch_count: int | None = None


def strip_ansi(text: str) -> str:
    return ANSI.sub("", text)


def to_float(value: str) -> float | None:
    cleaned = value.strip().replace(",", "")
    if not cleaned or cleaned.upper() in {"N/A", "NAN", "-", ""}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def relative_abs_delta(old: float, new: float) -> float | None:
    """Return |new-old|/|old|, or None when the baseline is ~0 (undefined).

    Health deltas compare Median values. A near-zero baseline must not be
    replaced with an epsilon — that yields absurd percentages such as 1e18%
    when old=0 and new is a small nonzero spike.
    """
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
    if any(fragment in name for fragment in SDK_BUG_ZERO_NAME_FRAGMENTS):
        return "zero_sdk_bug"
    if any(fragment in name for fragment in OPTIONAL_ZERO_NAME_FRAGMENTS):
        return "zero_optional"
    formula_upper = formula.upper()
    if any(token in formula_upper for token in ("FP64", "FP8", "FP6", "FP4", "SPARSE")):
        return "zero_optional"
    return "zero_other"


def _header_col_index(cols: list[str], name: str) -> int | None:
    for i, col in enumerate(cols):
        if col == name:
            return i
    return None


def _cell_at(cols: list[str], header_idx: int | None) -> tuple[str, float | None]:
    if header_idx is None:
        return "", None
    data_idx = max(0, header_idx - 2)
    if data_idx >= len(cols):
        return "", None
    raw = cols[data_idx]
    return raw, to_float(raw)


def parse_log(path: Path) -> ParsedLog:
    """Parse analyze --view table log; prefer Median over Value-only panels.

    Avg is never used as the primary value. Avg-only / Avg-Min-Max rows are
    kept with ``value=None`` until Median recompute or Min==Max fill.
    """
    text = strip_ansi(path.read_text(errors="replace"))
    result = ParsedLog()
    median_col: int | None = None
    min_col: int | None = None
    max_col: int | None = None
    value_col: int | None = None
    avg_col: int | None = None
    in_metric_table = False
    in_top_kernels = False

    for line in text.splitlines():
        if "0.1 Top Kernels" in line:
            in_top_kernels = True
            continue
        if in_top_kernels and line.startswith("0.2"):
            in_top_kernels = False
        if in_top_kernels:
            match_count = COUNT_RE.match(line)
            if match_count and result.dispatch_count is None:
                result.dispatch_count = int(float(match_count.group(1)))

        sup = SUPPRESS_RE.search(line)
        if sup:
            result.suppressed.append(sup.group(1).strip())
            continue

        if "Metric_ID" in line:
            cols = [c.strip() for c in re.split(r"[│|]", line) if c.strip()]
            # Reset so multi-bar panels without Median/Value cannot inherit.
            median_col = min_col = max_col = value_col = avg_col = None
            in_metric_table = False
            if not (
                "Median" in line or "Avg" in line or "Min" in line or "Value" in line
            ):
                continue
            median_col = _header_col_index(cols, "Median")
            min_col = _header_col_index(cols, "Min")
            max_col = _header_col_index(cols, "Max")
            value_col = _header_col_index(cols, "Value")
            avg_col = _header_col_index(cols, "Avg")
            in_metric_table = True
            continue

        match = ROW_RE.match(line)
        if not match or not in_metric_table:
            continue
        if (
            median_col is None
            and value_col is None
            and min_col is None
            and avg_col is None
        ):
            continue

        metric_id = match.group(1).strip()
        name = match.group(2).strip()
        cols = [c.strip() for c in re.split(r"[│|]", match.group(3)) if c.strip()]

        raw_med, med_v = _cell_at(cols, median_col)
        raw_min, min_v = _cell_at(cols, min_col)
        raw_max, max_v = _cell_at(cols, max_col)
        raw_val, val_v = _cell_at(cols, value_col)
        raw_avg, avg_v = _cell_at(cols, avg_col)

        value: float | None = None
        raw = ""
        source = ""
        if med_v is not None:
            value, raw, source = med_v, raw_med, "median"
        elif val_v is not None and median_col is None and min_col is None:
            # Value-only panels (scalar / peak tables) — not Avg.
            value, raw, source = val_v, raw_val, "value"
        elif (
            min_v is not None
            and max_v is not None
            and abs(min_v - max_v) <= max(1e-9, 1e-9 * abs(min_v))
        ):
            value, raw, source = min_v, raw_min, "minmax_eq"

        # Keep Avg/Min/Max rows even when Median is not yet known so recompute
        # / formula merge can see them. Never assign Avg to ``value``.
        # Also keep all-N/A rows (raw "N/A") so per-workload Totals and N/A
        # buckets stay aligned across workloads that omit different paths.
        has_present_cell = any(
            r != ""
            for r, idx in (
                (raw_med, median_col),
                (raw_min, min_col),
                (raw_max, max_col),
                (raw_val, value_col),
                (raw_avg, avg_col),
            )
            if idx is not None
        )
        if (
            value is None
            and med_v is None
            and min_v is None
            and val_v is None
            and avg_v is None
            and not has_present_cell
        ):
            continue

        if value is None and not raw:
            raw = next(
                (r for r in (raw_med, raw_val, raw_min, raw_max, raw_avg) if r),
                "N/A",
            )

        key = f"{metric_id}|{name}"
        result.metrics[key] = MetricValue(
            value=value,
            raw=raw,
            source=source,
            min_v=min_v,
            max_v=max_v,
        )

    return result


def _load_median_helpers():  # noqa: ANN202
    """Lazy-load median recompute helpers from compare_spp_legacy_medians."""
    helper_path = ROOT / "tools" / "compare_spp_legacy_medians.py"
    spec = importlib.util.spec_from_file_location(
        "compare_spp_legacy_medians", helper_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {helper_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def apply_recomputed_medians(
    parsed: dict[str, ParsedLog],
    workload_dirs: dict[str, Path],
) -> None:
    """Fill missing Medians from per-dispatch re-eval when dirs are provided.

    ``workload_dirs`` keys are ``{name}_spp`` / ``{name}_legacy`` or plain
    ``name`` (applied to the matching parsed SPP log only).
    """
    if not workload_dirs:
        return
    helpers = _load_median_helpers()
    compute = helpers.compute_medians_for_workload

    for dir_key, wl_dir in workload_dirs.items():
        if dir_key.endswith("_legacy"):
            continue  # baseline filled separately in build_report
        wl_name = dir_key.removesuffix("_spp")
        log = parsed.get(wl_name)
        if log is None:
            continue
        need = [k for k, mv in log.metrics.items() if mv.value is None]
        # Also recompute when only minmax_eq / absent median for Avg/Min/Max
        # panels — prefer true Median over Min==Max coincidence.
        need_all = [
            k
            for k, mv in log.metrics.items()
            if mv.source in {"", "minmax_eq"} or mv.value is None
        ]
        keys = need_all or need
        if not keys:
            # Still recompute all Min-backed rows to upgrade Avg/Min/Max panels
            # that had no Median column (source empty after Value-only skip).
            keys = list(log.metrics)
        try:
            medians = compute(wl_dir, keys)
        except Exception as exc:  # noqa: BLE001 — report soft-fail in HTML
            print(f"WARN: median recompute failed for {dir_key}: {exc}")
            continue
        for key, med in medians.items():
            if med is None:
                continue
            existing = log.metrics.get(key)
            if existing is None:
                log.metrics[key] = MetricValue(
                    value=med, raw=f"{med:g}", source="recomputed"
                )
            elif existing.source != "median":
                existing.value = med
                existing.raw = f"{med:g}"
                existing.source = "recomputed"


def apply_recomputed_medians_for_map(
    metrics_by_wl: dict[str, dict[str, MetricValue]],
    workload_dirs: dict[str, Path],
    mode_suffix: str,
) -> None:
    """Recompute medians into a flat wl→metrics map (for baseline)."""
    if not workload_dirs:
        return
    helpers = _load_median_helpers()
    compute = helpers.compute_medians_for_workload
    for wl_name, metrics in metrics_by_wl.items():
        dir_key = f"{wl_name}{mode_suffix}"
        wl_dir = workload_dirs.get(dir_key) or workload_dirs.get(wl_name)
        if wl_dir is None:
            continue
        try:
            medians = compute(wl_dir, list(metrics))
        except Exception as exc:  # noqa: BLE001
            print(f"WARN: median recompute failed for {dir_key}: {exc}")
            continue
        for key, med in medians.items():
            if med is None:
                continue
            existing = metrics.get(key)
            if existing is None:
                metrics[key] = MetricValue(
                    value=med, raw=f"{med:g}", source="recomputed"
                )
            elif existing.source != "median":
                existing.value = med
                existing.raw = f"{med:g}"
                existing.source = "recomputed"


def classify_metrics(
    workload_metrics: dict[str, dict[str, MetricValue]],
    metric_units: dict[str, str],
    metric_formulas: dict[str, str],
) -> None:
    mega = workload_metrics.get("mega_kernel", {})
    for wl_name, wl_metrics in workload_metrics.items():
        for key, mv in wl_metrics.items():
            unit = metric_units.get(key, "")
            name = key.split("|", 1)[1]
            formula = metric_formulas.get(key, "")
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
                _name = key.split("|", 1)[1]
                if _name in COUNTER_SEMANTIC_OVERFLOW_NAMES:
                    mv.cls = "formula_overflow"
                else:
                    mv.cls = "overflow"
                continue
            mega_mv = mega.get(key)
            mega_val = mega_mv.value if mega_mv else None
            if abs(mv.value) < 1e-12:
                if (
                    wl_name != "mega_kernel"
                    and mega_val is not None
                    and abs(mega_val) > 1e-12
                ):
                    mv.cls = "workload_zero"
                elif all(
                    wl_metrics.get(key, MetricValue(None, "")).value is not None
                    and abs(wl_metrics.get(key, MetricValue(0.0, "0")).value or 0)
                    < 1e-12
                    for wl_metrics in workload_metrics.values()
                ):
                    mv.cls = classify_all_workload_zero(name, formula)
                else:
                    mv.cls = "good"
                continue
            mv.cls = "good"


def fmt_value(mv: MetricValue | None, unit: str = "") -> str:
    if mv is None or mv.value is None:
        return "N/A"
    if "percent" in unit.lower():
        return f"{mv.value:.2f}".rstrip("0").rstrip(".")
    if abs(mv.value) >= 1000:
        return f"{mv.value:,.2f}".rstrip("0").rstrip(".")
    return f"{mv.value:.2g}"


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


def render_zero_sub_bucket_lines(summary: dict[str, int]) -> str:
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


# CDNA3 multi-XCD L2-per-channel expansion ($total_l2_chan) applies here.
_CDNA3_L2_CHAN_ARCHES = frozenset({"gfx940", "gfx941", "gfx942"})


def count_yaml_metric_entries(config_arch: str) -> tuple[int, int]:
    """Return (metric table entries, unique metric names) for analysis_configs."""
    config_dir = SRC / "rocprof_compute_soc" / "analysis_configs" / config_arch
    entries = 0
    names: set[str] = set()
    if not config_dir.is_dir():
        return 0, 0
    for yaml_path in sorted(config_dir.glob("*.yaml")):
        data = yaml.safe_load(yaml_path.read_text())
        panel = data.get("Panel Config", data)
        for entry in panel.get("data source", []):
            if not isinstance(entry, dict):
                continue
            table = entry.get("metric_table", entry)
            if not isinstance(table, dict) or "metric" not in table:
                continue
            metrics = table.get("metric") or {}
            entries += len(metrics)
            names.update(metrics)
    return entries, len(names)


def build_total_note(
    arch: str,
    config_arch: str,
    summary: dict[str, dict[str, int]],
    all_keys_count: int,
    stat_label: str,
) -> str:
    """Arch-specific explanation of card Total vs YAML metric definitions."""
    yaml_entries, yaml_unique = count_yaml_metric_entries(config_arch)
    yaml_bit = (
        f"YAML metric definitions (~{yaml_entries} table entries / "
        f"~{yaml_unique} unique names in <code>analysis_configs/{html.escape(config_arch)}</code>)"
    )
    totals = ", ".join(f"{html.escape(wl)}={s['total']}" for wl, s in summary.items())
    common_tail = (
        f"Each card Total equals the union of analyze-table rows across workloads "
        f"(<b>{all_keys_count}</b> here; per-workload "
        f"{totals}). All-N/A cells are kept and bucketed as N/A / Expected N/A — "
        f"they are not dropped from Total. Entire panels omitted by TTY "
        f"('<code>Not showing table with empty column(s)</code>') are not in any Total. "
        f"Cell values are <b>{html.escape(stat_label)}</b> "
        f"(from log Median column, or recomputed from per-dispatch data)."
    )

    if arch in _CDNA3_L2_CHAN_ARCHES:
        return (
            "<p style='font-size:11px;color:#555;margin:4px 0 10px;max-width:960px'>"
            "<b>Total</b> counts analyze-table rows (from <code>--view table</code> logs), "
            f"not {yaml_bit}. "
            "Block 18 (L2 Cache per Channel) expands each placeholder table across "
            "<code>$total_l2_chan</code> (= <code>l2_banks × num_xcd</code>). "
            "Example: this run reconciles to 16 channels (CPX / single-XCD) → ~471 rows; "
            "MI300X SPX would use "
            "<code>l2_banks(16)×num_xcd(8)=128</code> → "
            "<code>387 + 1 + 8×128 = 1412</code> rows. "
            "If sysinfo reports SPX/128 channels but the profiled ROCR device is a "
            "single-XCD die (<code>cu_per_gpu≈38</code>), channels 16–127 have no "
            "TCC results and appear as N/A (896 = 112×8 tables) — fix is reconciling "
            "<code>num_xcd</code>/<code>total_l2_chan</code> to the visible device "
            "(CPX → 16 channels). "
            f"{common_tail}"
            "</p>"
        )

    if config_arch == "gfx115x" or arch.startswith("gfx115"):
        return (
            "<p style='font-size:11px;color:#555;margin:4px 0 10px;max-width:960px'>"
            "<b>Total</b> counts analyze-table rows (from <code>--view table</code> logs), "
            f"not {yaml_bit}. "
            "RDNA3.5 / gfx115x has no CDNA Block-18 "
            "<code>$total_l2_chan</code> L2-per-channel expansion, so Totals stay near "
            "the YAML entry count (plus any non-channel table expansion). "
            f"{common_tail}"
            "</p>"
        )

    return (
        "<p style='font-size:11px;color:#555;margin:4px 0 10px;max-width:960px'>"
        "<b>Total</b> counts analyze-table rows (from <code>--view table</code> logs), "
        f"not {yaml_bit}. "
        f"{common_tail}"
        "</p>"
    )


def _iteration_banner(
    iterations: int | None,
    dispatch_counts: dict[str, int | None],
    stat_label: str,
) -> str:
    """Build the Statistic / iterations header strip.

    Collapses requested iterations and observed top-kernel dispatch counts into
    one "wl req/disp" pair list when both are available. CLI flag hints
    (-i / arg2 / -n) are omitted when every pair matches - they only repeat
    the same number.
    """
    parts = [f"<b>Statistic:</b> {html.escape(stat_label)}"]
    known_dispatches = {
        wl: count for wl, count in dispatch_counts.items() if count is not None
    }
    if iterations is not None and known_dispatches:
        pair_bits = [
            f"{html.escape(wl)} {iterations}/{count}"
            for wl, count in known_dispatches.items()
        ]
        parts.append(
            "<b>Iterations / top-kernel dispatches:</b> " + ", ".join(pair_bits)
        )
        if any(count != iterations for count in known_dispatches.values()):
            parts.append(
                f"CLI: vcopy <code>-i {iterations}</code>, "
                f"nbody arg2={iterations}, "
                f"mega_kernel <code>-n {iterations}</code>"
            )
    elif iterations is not None:
        parts.append(
            f"<b>Requested iterations:</b> n={iterations} "
            f"(vcopy <code>-i {iterations}</code>, "
            f"nbody arg2={iterations}, "
            f"mega_kernel <code>-n {iterations}</code>)"
        )
    elif known_dispatches:
        count_bits = [
            f"{html.escape(wl)}={count}" for wl, count in known_dispatches.items()
        ]
        parts.append("<b>Top-kernel Count (dispatches):</b> " + ", ".join(count_bits))
    return (
        '<p style="margin:2px 0;color:#37474f;font-size:12px">'
        + " | ".join(parts)
        + "</p>"
    )


def build_report(
    workloads: dict[str, Path],
    formulas: dict[str, tuple[str, str, bool]],
    baseline_logs: dict[str, Path] | None,
    out_path: Path,
    report_date: str,
    baseline_label: str,
    host_label: str,
    branch_label: str,
    arch: str,
    config_arch: str,
    iterations: int | None,
    stat_label: str,
    workload_dirs: dict[str, Path] | None,
) -> None:
    wl_names = list(workloads.keys())
    parsed_logs: dict[str, ParsedLog] = {}
    for wl, log_path in workloads.items():
        parsed_logs[wl] = parse_log(log_path)

    if workload_dirs:
        apply_recomputed_medians(parsed_logs, workload_dirs)

    parsed: dict[str, dict[str, MetricValue]] = {
        wl: log.metrics for wl, log in parsed_logs.items()
    }
    suppressed: dict[str, list[str]] = {
        wl: log.suppressed for wl, log in parsed_logs.items()
    }
    dispatch_counts: dict[str, int | None] = {
        wl: log.dispatch_count for wl, log in parsed_logs.items()
    }

    metric_units = {k: formulas.get(k, ("", "", False))[1] for k in formulas}
    metric_formulas = {k: formulas.get(k, ("", "", False))[0] for k in formulas}
    classify_metrics(parsed, metric_units, metric_formulas)

    all_keys: list[str] = []
    seen: set[str] = set()
    for wl in wl_names:
        for key in sorted(
            parsed[wl], key=lambda k: [int(p) for p in k.split("|")[0].split(".")]
        ):
            if key not in seen:
                seen.add(key)
                all_keys.append(key)

    summary: dict[str, dict[str, int]] = {
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
        for wl in wl_names
    }
    # Total counts analyze-table rows parsed from --view table logs (including
    # all-N/A cells), not YAML metric definitions. On CDNA3, Block 18 expands
    # across $total_l2_chan — see build_total_note(). Missing keys (present in
    # another workload's log only) count as N/A so each card Total matches the
    # main-table row universe (len(all_keys)).
    for key in all_keys:
        for wl in wl_names:
            mv = parsed[wl].get(key)
            summary[wl]["total"] += 1
            if mv is None:
                summary[wl]["na"] += 1
                continue
            if mv.cls in summary[wl]:
                summary[wl][mv.cls] += 1

    cards = []
    descs = {
        "mega_kernel": "All-ops (RDNA35 mega kernel)",
        "vcopy": "Memory-bound",
        "nbody": "Compute-bound",
        "wmma_gemm": "WMMA",
    }
    for wl in wl_names:
        s = summary[wl]
        n_disp = dispatch_counts.get(wl)
        disp_line = (
            f"<div style='font-size:10px;color:#546e7a'>"
            f"dispatches: <b>{n_disp}</b></div>"
            if n_disp is not None
            else ""
        )
        cards.append(
            f"<div class='card'><h3>{html.escape(wl)}</h3>"
            f"<div class='desc'>{descs.get(wl, '')}</div>"
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

    total_note = build_total_note(arch, config_arch, summary, len(all_keys), stat_label)

    improve_rows: list[str] = []
    delta_rows: list[str] = []
    if baseline_logs:
        base_parsed: dict[str, dict[str, MetricValue]] = {
            wl: parse_log(path).metrics for wl, path in baseline_logs.items()
        }
        if workload_dirs:
            apply_recomputed_medians_for_map(base_parsed, workload_dirs, "_legacy")
        classify_metrics(base_parsed, metric_units, metric_formulas)
        for key in all_keys:
            mid, name = key.split("|", 1)
            unit = formulas.get(key, ("", "", False))[1]
            for wl in wl_names:
                old = base_parsed.get(wl, {}).get(key)
                new = parsed.get(wl, {}).get(key)
                if not old or not new:
                    continue
                if old.cls in {"overflow", "formula_overflow"} and new.cls not in {
                    "overflow",
                    "formula_overflow",
                }:
                    improve_rows.append(
                        f"<tr><td>{html.escape(wl)}</td>"
                        f"<td class='mid'>{html.escape(mid)}</td>"
                        f"<td>{html.escape(name)}</td>"
                        f"<td style='background:#ffcdd2;font-weight:bold'>"
                        f"{html.escape(fmt_value(old, unit))}</td>"
                        f"<td style='background:#c8e6c9;font-weight:bold'>"
                        f"{html.escape(fmt_value(new, unit))}</td>"
                        f"<td>{html.escape(unit)}</td></tr>"
                    )
                if old.value is None or new.value is None:
                    continue
                # |rel| with old≈0 is undefined — skip (no huge % highlights).
                rel = relative_abs_delta(old.value, new.value)
                if rel is None or rel < DELTA_REL_THRESHOLD:
                    continue
                rel_pct = rel * 100.0
                hi = rel >= DELTA_HIGHLIGHT_THRESHOLD
                row_cls = " class='delta-ge-100'" if hi else ""
                rel_cell = (
                    f"<td style='background:#ffe082;font-weight:bold'>"
                    f"{rel_pct:.1f}%</td>"
                    if hi
                    else f"<td>{rel_pct:.1f}%</td>"
                )
                delta_rows.append(
                    f"<tr{row_cls}><td>{html.escape(wl)}</td>"
                    f"<td class='mid'>{html.escape(mid)}</td>"
                    f"<td>{html.escape(name)}</td>"
                    f"<td>{html.escape(fmt_value(old, unit))}</td>"
                    f"<td>{html.escape(fmt_value(new, unit))}</td>"
                    f"{rel_cell}"
                    f"<td>{html.escape(unit)}</td></tr>"
                )

    sup_rows = []
    for wl in wl_names:
        for table in suppressed.get(wl, []):
            sup_rows.append(
                f"<tr><td>{html.escape(wl)}</td><td>{html.escape(table)}</td></tr>"
            )

    body_rows = []
    for key in all_keys:
        mid, name = key.split("|", 1)
        formula, unit, level = formulas.get(key, ("", "", False))
        row_cats: set[str] = set()
        wl_cells = []
        for wl in wl_names:
            mv = parsed[wl].get(key)
            cls = mv.cls if mv else "na"
            row_cats.add(cls)
            val = fmt_value(mv, unit)
            style = cell_style(cls)
            wl_cells.append(
                f"<td style='{style}' data-cls='{cls}'>{html.escape(val)}</td>"
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
        body_rows.append(
            f"<tr data-cats='{' '.join(sorted(row_cats))}'>"
            f"<td class='mid'>{html.escape(mid)}</td>"
            f"<td class='mname'>{html.escape(name)}</td>"
            f"<td class='unit'>{html.escape(unit)}</td>"
            f"<td style='text-align:center;font-size:11px'>{level_cell}</td>"
            f"<td><div style='overflow-x:auto;white-space:nowrap;max-width:600px;font-size:10px;"
            f"font-family:monospace;background:#f0f4f8;color:#1a237e;padding:2px 5px;"
            f"border-radius:3px;border:1px solid #c5cae9'>{formula_html}</div></td>"
            + "".join(wl_cells)
            + "</tr>"
        )

    wl_headers = "".join(
        f"<th>{html.escape(wl)}<br><small>{descs.get(wl, '')}</small><br>"
        f"<small>{html.escape(stat_label)}</small></th>"
        for wl in wl_names
    )

    arch_title = ARCH_TITLE.get(arch, arch)
    iter_banner = _iteration_banner(iterations, dispatch_counts, stat_label)
    improve_section = ""
    if improve_rows:
        improve_section = (
            f"<h2>Overflow improvements vs {html.escape(baseline_label)} "
            f"({len(improve_rows)} metrics fixed)</h2>"
            "<table class='improve-table'><thead><tr><th>Workload</th><th>ID</th>"
            f"<th>Metric</th><th>Old ({html.escape(stat_label)})</th>"
            f"<th>New ({html.escape(stat_label)})</th><th>Unit</th></tr></thead><tbody>"
            + "".join(improve_rows)
            + "</tbody></table>"
        )
    if delta_rows:
        n_hi = sum(1 for r in delta_rows if "delta-ge-100" in r)
        n_delta = len(delta_rows)
        n_metrics = len(all_keys)
        delta_of_total_pct = (100.0 * n_delta / n_metrics) if n_metrics else 0.0
        rel_pct = DELTA_REL_THRESHOLD * 100
        hi_pct = DELTA_HIGHLIGHT_THRESHOLD * 100
        near_zero = f"{DELTA_NEAR_ZERO:g}"
        # Fraction uses unique analyze-table metrics (same universe as card
        # Total / main table rows), not metric×workload cells.
        improve_section += (
            f"<h2>Value deltas vs {html.escape(baseline_label)} "
            f"(|rel| ≥ {rel_pct:.0f}%, {n_delta} rows, "
            f"{delta_of_total_pct:.1f}% of {n_metrics} metrics)</h2>"
            "<p class='filter-note'>Click a column header to sort "
            "(Kernel name, ID, Metric, |rel|, …). "
            f"Compared values are <b>{html.escape(stat_label)}</b> from analyze "
            "<code>--view table</code> logs (or recomputed from per-dispatch "
            "data when the Median column is absent). "
            "|rel| = |new−old|/|old|.</p>"
            "<p class='filter-note'>"
            f"Include when |rel| ≥ {rel_pct:.0f}% "
            f"({n_delta} rows, {delta_of_total_pct:.1f}% of "
            f"{n_metrics} metrics)<br>"
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
            f"<th data-sort='num'>Old ({html.escape(stat_label)})</th>"
            f"<th data-sort='num'>New ({html.escape(stat_label)})</th>"
            "<th data-sort='num'>|rel|</th>"
            "<th data-sort='str'>Unit</th>"
            "</tr></thead><tbody>" + "".join(delta_rows) + "</tbody></table>"
        )

    doc = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>{html.escape(arch_title)} — Metric Health {html.escape(report_date)}</title>
<style>
body{{font-family:'Segoe UI',Arial,sans-serif;background:#eceff1;margin:0;padding:16px;font-size:13px}}
h1{{color:#0d47a1;margin-bottom:4px}}h2{{color:#263238;border-bottom:2px solid #1565c0;padding-bottom:4px;margin-top:22px}}
.cards{{display:flex;gap:12px;flex-wrap:wrap;margin:8px 0}}
.card{{background:#fff;border-radius:8px;padding:10px 16px;box-shadow:0 2px 6px rgba(0,0,0,.12);min-width:140px}}
.card h3{{margin:0 0 3px;font-size:13px;color:#555}}.desc{{font-size:10px;color:#888;margin-bottom:4px}}
.improve-table{{border-collapse:collapse;width:100%;background:#fff;margin-top:6px;font-size:12px}}
.improve-table th{{background:#1565c0;color:#fff;padding:5px 8px;text-align:left}}
.improve-table td{{padding:3px 8px;border-bottom:1px solid #eee}}
.sortable-table th{{cursor:pointer;user-select:none;white-space:nowrap}}
.sortable-table th:hover{{background:#0d47a1}}
.sortable-table th.sort-asc::after{{content:' \\25b2';font-size:10px}}
.sortable-table th.sort-desc::after{{content:' \\25bc';font-size:10px}}
tr.delta-ge-100 td{{background:#fff8e1}}
tr.delta-ge-100 td:nth-child(6){{background:#ffe082;font-weight:bold}}
tr.delta-ge-100:hover td{{filter:brightness(0.96)}}
.bucket-semantics{{border-collapse:collapse;width:100%;max-width:920px;background:#fff;margin:10px 0 4px;font-size:12px;box-shadow:0 1px 4px rgba(0,0,0,.08)}}
.bucket-semantics th{{background:#eceff1;color:#37474f;padding:8px 10px;text-align:left;font-weight:600;position:static}}
.bucket-semantics td{{padding:7px 10px;border-bottom:1px solid #e0e0e0;vertical-align:top}}
.bucket-semantics tr:last-child td{{border-bottom:none}}
.bucket-semantics td:first-child{{font-weight:600;white-space:nowrap}}
.legend{{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0;user-select:none}}
.filter-note{{font-size:11px;color:#888;margin:-4px 0 4px}}
.tbl-wrap{{overflow-x:auto;max-height:65vh;overflow-y:auto}}
table{{border-collapse:collapse;width:100%;background:#fff;font-size:12px}}
th{{background:#37474f;color:#fff;padding:6px 8px;text-align:left;position:sticky;top:0;z-index:2;white-space:nowrap}}
td{{padding:3px 5px;border-bottom:1px solid #e0e0e0;vertical-align:middle}}
tr:hover td{{filter:brightness(0.91)}}
.mid{{font-family:monospace;color:#546e7a;font-size:11px}}.mname{{max-width:180px}}
.unit{{color:#90a4ae;font-style:italic;font-size:10px}}
tr.row-overflow{{box-shadow:inset 0 0 0 2px #c62828}}
.sup-table{{font-size:12px}}.sup-table td{{padding:3px 8px}}.sup-table tr:nth-child(even){{background:#f9f9f9}}
</style></head><body>
<h1>{html.escape(arch_title)} — Metric Health Report</h1>
<p style="margin:2px 0;color:#607d8b;font-size:12px"><b>Date:</b> {html.escape(report_date)}
 | <b>Host:</b> {html.escape(host_label)}
 | <b>Branch:</b> {html.escape(branch_label)}
 | <b>Arch:</b> {html.escape(arch)}
 | <b>Workloads:</b> {html.escape(", ".join(wl_names))}</p>
{iter_banner}
<p style="font-size:11px;color:#555">Analysis configs: <code>{html.escape(config_arch)}</code>.
All-workload zeros are split into sub-buckets (healthy / optional path / SDK bug suspect / other).
Expected zero / Expected N/A compare non-mega workloads against mega_kernel when present.</p>
<h2>Summary</h2>
{total_note}
<div class="cards">{"".join(cards)}</div>
<h2>Bucket semantics</h2>
<p style="font-size:11px;color:#555;margin:2px 0 6px">Each metric in each workload column is assigned to <b>exactly one</b> bucket; counts on each card sum to Total.</p>
<table class="bucket-semantics">
<thead><tr><th>Bucket</th><th>Meaning</th><th>Healthy?</th></tr></thead>
<tbody>
<tr><td>Good</td><td>Parsed value exists; not N/A; not &gt;100% (percent); not classified as zero</td><td>Usually yes</td></tr>
<tr><td>Zero — healthy (<code>zero_healthy</code>)</td><td>Zero in every workload; metric is a stall/failure/alloc-failure rate where 0% is good</td><td>Yes</td></tr>
<tr><td>Zero — optional path (<code>zero_optional</code>)</td><td>Zero in every workload; precision or IP path mega_kernel does not exercise (FP64, etc.)</td><td>Usually yes</td></tr>
<tr><td>Zero — SDK bug suspect (<code>zero_sdk_bug</code>)</td><td>Zero in every workload but mega_kernel exercises the path or counters should fire (VALU FLOPs, DRAM read)</td><td>Investigate — file SDK JIRA</td></tr>
<tr><td>Zero — other (<code>zero_other</code>)</td><td>Zero in every workload; uncategorized (often low-traffic structural zeros)</td><td>Review case-by-case</td></tr>
<tr><td>Expected zero (<code>workload_zero</code>)</td><td>Zero here, but non-zero in mega_kernel</td><td>Yes — workload doesn't hit that path</td></tr>
<tr><td>Expected N/A (<code>expected_na</code>)</td><td>N/A here (formula guard / zero denominator), but mega_kernel has a value</td><td>Yes — workload doesn't hit that path</td></tr>
<tr><td>N/A</td><td>Analyze couldn't produce a value and mega_kernel also has no value</td><td>Investigate</td></tr>
<tr><td>&gt;100% formula (<code>formula_overflow</code>)</td><td>Percent &gt;100% from known counter semantics (OR-gate overlap, pipeline slot counting)</td><td>Usually benign — read trend not absolute %</td></tr>
<tr><td>&gt;100% hardware (<code>overflow</code>)</td><td>Percent &gt;100% not explained by known semantics — likely bad formula or counter bug</td><td>Investigate</td></tr>
</tbody></table>
{improve_section}
<h2>Suppressed Tables</h2>
<table class="sup-table"><tr style="background:#455a64;color:#fff"><th>Workload</th><th>Table</th></tr>
{"".join(sup_rows) if sup_rows else "<tr><td colspan=2>None</td></tr>"}
</table>
<h2>All Metrics — Blocks 2–18</h2>
<p style="font-size:11px;color:#555;margin:4px 0">Blocks 3 (Memory Chart) and 4 (Roofline) are chart/roofline panels — <b>no tabular metric rows in log output</b>.
Column values are <b>{html.escape(stat_label)}</b>.</p>
<div class="legend">
<div class="leg" data-cls="good" style="background:#c8e6c9;color:#1b5e20;font-weight:bold;padding:5px 12px;border-radius:5px;cursor:pointer;font-size:12px">▲ Good (real data)</div>
<div class="leg" data-cls="overflow" style="background:#c62828;color:#fff;font-weight:bold;padding:5px 12px;border-radius:5px;cursor:pointer;font-size:12px">🚨 &gt;100% hardware</div>
<div class="leg" data-cls="formula_overflow" style="background:#ff8f00;color:#fff;font-weight:bold;padding:5px 12px;border-radius:5px;cursor:pointer;font-size:12px">📐 &gt;100% formula</div>
<div class="leg" data-cls="zero_healthy" style="background:#e8f5e9;color:#2e7d32;padding:5px 12px;border-radius:5px;cursor:pointer;font-size:12px">✓ Zero — healthy (0% stall)</div>
<div class="leg" data-cls="zero_optional" style="background:#e3f2fd;color:#1565c0;padding:5px 12px;border-radius:5px;cursor:pointer;font-size:12px">○ Zero — optional path</div>
<div class="leg" data-cls="zero_sdk_bug" style="background:#ffcdd2;color:#c62828;font-weight:bold;padding:5px 12px;border-radius:5px;cursor:pointer;font-size:12px">❌ Zero — SDK bug suspect</div>
<div class="leg" data-cls="zero_other" style="background:#f5f5f5;color:#616161;padding:5px 12px;border-radius:5px;cursor:pointer;font-size:12px">⬜ Zero — other</div>
<div class="leg" data-cls="workload_zero" style="background:#b3e5fc;color:#01579b;padding:5px 12px;border-radius:5px;cursor:pointer;font-size:12px">⚪ Expected zero (non-zero in mega)</div>
<div class="leg" data-cls="expected_na" style="background:#d1c4e9;color:#4527a0;padding:5px 12px;border-radius:5px;cursor:pointer;font-size:12px">🔹 Expected N/A (valid in mega)</div>
<div class="leg" data-cls="na" style="background:#e1bee7;color:#6a1b9a;padding:5px 12px;border-radius:5px;cursor:pointer;font-size:12px">⚠️ N/A (investigate)</div>
</div>
<p class="filter-note">💡 Click legend to show/hide rows by category</p>
<div class="tbl-wrap"><table id="mt">
<thead><tr><th>ID</th><th>Metric Name</th><th>Unit</th><th>LEVEL<br><small>if problematic</small></th><th>Formula (scroll →)</th>{wl_headers}</tr></thead>
<tbody>{"".join(body_rows)}</tbody></table></div>
<script>
const active=new Set(['good','overflow','formula_overflow','zero_healthy','zero_optional','zero_sdk_bug','zero_other','workload_zero','expected_na','na']);
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

    out_path.write_text(doc, encoding="utf-8")


def merge_formulas_from_logs(
    workloads: dict[str, Path],
    config_arch: str,
) -> dict[str, tuple[str, str, bool]]:
    formulas: dict[str, tuple[str, str, bool]] = {}
    config_dir = SRC / "rocprof_compute_soc" / "analysis_configs" / config_arch
    if not config_dir.is_dir():
        raise SystemExit(f"analysis config dir missing: {config_dir}")
    yaml_formulas: dict[str, str] = {}
    yaml_units: dict[str, str] = {}

    for yaml_path in sorted(config_dir.glob("*.yaml")):
        data = yaml.safe_load(yaml_path.read_text())
        panel = data.get("Panel Config", data)
        for entry in panel.get("data source", []):
            if not isinstance(entry, dict):
                continue
            table = entry.get("metric_table", entry)
            if not isinstance(table, dict) or "metric" not in table:
                continue
            for name, spec in (table.get("metric") or {}).items():
                if not isinstance(spec, dict):
                    continue
                formula = str(spec.get("value", spec.get("avg", "")))
                unit = str(spec.get("unit", ""))
                yaml_formulas[name] = formula
                yaml_units[name] = unit

    for log_path in workloads.values():
        parsed = parse_log(log_path)
        for key in parsed.metrics:
            _mid, name = key.split("|", 1)
            formula = yaml_formulas.get(name, "")
            unit = yaml_units.get(name, "")
            formulas[key] = (formula, unit, uses_level_counter(formula))

    return formulas


def _parse_name_path_pairs(specs: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for spec in specs:
        name, path = spec.split(":", 1)
        result[name] = Path(path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arch",
        required=True,
        choices=sorted(ARCH_CONFIG_DIR),
        help="GPU arch (selects analysis_configs dir)",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--baseline-label", default="legacy heuristic")
    parser.add_argument("--host", default="unknown")
    parser.add_argument(
        "--branch",
        default="users/feizheng10/aiprofcomp-865-cp-sat-local",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Requested workload iterations (shown in report header)",
    )
    parser.add_argument(
        "--stat",
        default="Median",
        help="Primary statistic label for the report (default: Median)",
    )
    parser.add_argument(
        "--workload-dir",
        nargs="*",
        default=[],
        help=(
            "Optional name_mode:path dirs for Median recompute "
            "(e.g. vcopy_spp:/path/to/MI300X_A1)"
        ),
    )
    parser.add_argument("workload_log", nargs="+", help="name:path pairs (SPP)")
    parser.add_argument(
        "--baseline",
        nargs="*",
        default=[],
        help="baseline workload logs as name:path (e.g. legacy)",
    )
    args = parser.parse_args()

    config_arch = ARCH_CONFIG_DIR[args.arch]
    workloads = _parse_name_path_pairs(args.workload_log)
    baseline_logs = _parse_name_path_pairs(args.baseline) if args.baseline else None
    workload_dirs = (
        _parse_name_path_pairs(args.workload_dir) if args.workload_dir else None
    )

    formulas = merge_formulas_from_logs(workloads, config_arch)
    build_report(
        workloads,
        formulas,
        baseline_logs,
        args.out,
        args.date,
        args.baseline_label,
        args.host,
        args.branch,
        args.arch,
        config_arch,
        args.iterations,
        args.stat,
        workload_dirs,
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
