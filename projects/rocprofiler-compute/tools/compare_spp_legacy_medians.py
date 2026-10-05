#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Compare SPP vs legacy metric medians from 500-iter health workloads.

Analyze ``--view table`` logs expose Avg/Min/Max, not Median. This tool:

1. Parses Avg/Min/Max (+ dispatch Count) from the analyze logs.
2. When workload dirs are present, recomputes Median by evaluating
   ``MEDIAN(<min-formula-inner>)`` via the same MetricEvaluator path
   analyze uses (so Median matches the per-dispatch quantity behind Min/Max).
3. Emits CSV + markdown of |rel| > 100% on medians, with raw double-checks.

Usage:
  python3 tools/compare_spp_legacy_medians.py \\
    --artifacts validation-artifacts/spp-health-261001-gfx942-500med \\
    --tag 261001-gfx942-500med
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DELTA_NEAR_ZERO = 1e-12
REL_HIGH = 1.0
ANSI = re.compile(r"\x1b\[[0-9;]*m")
ROW_RE = re.compile(
    r"^\s*[│|]\s*"
    r"([0-9]+(?:\.[0-9]+)*)\s*[│|]\s*"
    r"([^│|]+?)\s*[│|]\s*"
    r"(.+)$"
)
COUNT_RE = re.compile(r"^\s*[│|]\s*0\s*[│|].*?[│|]\s*([0-9]+(?:\.[0-9]+)?)\s*[│|]")


@dataclass
class MetricStats:
    metric_id: str
    name: str
    avg: float | None
    min_v: float | None
    max_v: float | None
    median: float | None
    raw_avg: str
    raw_min: str
    raw_max: str
    raw_median: str


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
    if abs(old) < DELTA_NEAR_ZERO:
        return None
    return abs(new - old) / abs(old)


def parse_log_stats(path: Path) -> tuple[dict[str, MetricStats], int | None]:
    """Parse Avg/Min/Max/Median rows and Top-Kernels Count from analyze log."""
    text = strip_ansi(path.read_text(errors="replace"))
    metrics: dict[str, MetricStats] = {}
    avg_col: int | None = None
    min_col: int | None = None
    max_col: int | None = None
    median_col: int | None = None
    dispatch_count: int | None = None
    in_top_kernels = False

    for line in text.splitlines():
        if "0.1 Top Kernels" in line:
            in_top_kernels = True
            continue
        if in_top_kernels and line.startswith("0.2"):
            in_top_kernels = False
        if in_top_kernels:
            m = COUNT_RE.match(line)
            if m and dispatch_count is None:
                dispatch_count = int(float(m.group(1)))

        if "Metric_ID" in line:
            cols = [c.strip() for c in re.split(r"[│|]", line) if c.strip()]
            # Always reset on a new table header so multi-bar panels
            # (no Avg/Min/Median) cannot inherit stale column indices.
            avg_col = min_col = max_col = median_col = None
            if not ("Avg" in line or "Median" in line or "Min" in line):
                continue
            for i, col in enumerate(cols):
                if col == "Avg":
                    avg_col = i
                elif col == "Min":
                    min_col = i
                elif col == "Max":
                    max_col = i
                elif col == "Median":
                    median_col = i
            continue

        match = ROW_RE.match(line)
        if not match:
            continue
        if avg_col is None and median_col is None:
            continue

        metric_id = match.group(1).strip()
        name = match.group(2).strip()
        cols = [c.strip() for c in re.split(r"[│|]", match.group(3)) if c.strip()]

        def _cell(header_idx: int | None) -> tuple[str, float | None]:
            if header_idx is None:
                return "", None
            data_idx = max(0, header_idx - 2)
            if data_idx >= len(cols):
                return "", None
            raw = cols[data_idx]
            return raw, to_float(raw)

        raw_avg, avg_v = _cell(avg_col)
        raw_min, min_v = _cell(min_col)
        raw_max, max_v = _cell(max_col)
        raw_med, med_v = _cell(median_col)

        # Skip rows that only matched a stale header with no usable value.
        if avg_v is None and med_v is None and min_v is None:
            continue

        key = f"{metric_id}|{name}"
        metrics[key] = MetricStats(
            metric_id=metric_id,
            name=name,
            avg=avg_v,
            min_v=min_v,
            max_v=max_v,
            median=med_v,
            raw_avg=raw_avg,
            raw_min=raw_min,
            raw_max=raw_max,
            raw_median="" if med_v is None else raw_med,
        )
    return metrics, dispatch_count


def min_formula_to_median_expr(min_formula: str) -> str | None:
    """Rewrite a Min YAML/expr formula into a Median expression.

    Handles ``MIN(x)`` and ``c * MIN(x)`` / ``MIN(x) * c`` patterns that
    appear in analysis_configs.
    """
    s = min_formula.strip()
    if not s or s == "None":
        return None

    # Direct MIN(...)
    if s.startswith("MIN("):
        depth = 0
        for i, ch in enumerate(s):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    if i != len(s) - 1:
                        # trailing ops after MIN(...); fall through
                        break
                    return f"MEDIAN{s[3 : i + 1]}"
        return None

    # Factor * MIN(...)  or  MIN(...) * factor
    m = re.fullmatch(
        r"(.+?)\s*\*\s*(MIN\(.+\))",
        s,
    )
    if m:
        left, min_call = m.group(1).strip(), m.group(2)
        med = min_formula_to_median_expr(min_call)
        if med:
            return f"{left} * {med}"
    m = re.fullmatch(
        r"(MIN\(.+\))\s*\*\s*(.+)",
        s,
    )
    if m:
        min_call, right = m.group(1), m.group(2).strip()
        med = min_formula_to_median_expr(min_call)
        if med:
            return f"{med} * {right}"
    return None


def find_workload_dir(artifacts: Path, tag: str, name: str, mode: str) -> Path | None:
    base = artifacts / "gfx942" / f"{name}_{tag}_{mode}"
    if not base.exists():
        return None
    for cand in base.rglob("sysinfo.csv"):
        return cand.parent
    return None


def compute_medians_for_workload(
    wl_dir: Path,
    metric_keys: list[str],
) -> dict[str, float | None]:
    """Recompute Median for metrics that have a Min formula, via analyze path."""
    del metric_keys  # compute all Min-backed rows; keys unused
    import pandas as pd

    import config
    from utils import file_io, parser, schema
    from utils.metrics.evaluation_pipeline import eval_metric
    from utils.metrics.expression import build_metric_value_string
    from utils.utils_common import canonical_config_arch

    sys_info = parser.reconcile_sysinfo_l2_channels(pd.read_csv(wl_dir / "sysinfo.csv"))
    arch = str(sys_info.iloc[0]["gpu_arch"])
    config_arch = canonical_config_arch(arch) or arch
    config_dir = (
        config.rocprof_compute_home
        / "rocprof_compute_soc"
        / "analysis_configs"
        / config_arch
    )

    ac = schema.ArchConfig()
    ac.panel_configs = file_io.load_panel_configs([str(config_dir)])
    profiling_config = file_io.load_profiling_config(str(wl_dir))

    parser.build_dfs(
        arch_configs=ac,
        filter_metrics=None,
        sys_info=sys_info.iloc[0],
        profiling_config=profiling_config,
        arch=arch,
    )

    # Inject Median only where the panel has Min but no Median yet
    # (Avg/Min/Max tables). simple_box panels already define Median.
    for df_id, df in ac.dfs.items():
        if ac.dfs_type.get(df_id) != "metric_table":
            continue
        if "Min" not in df.columns:
            continue
        if "Median" in df.columns:
            continue
        medians: list[str | None] = []
        for _, row in df.iterrows():
            min_expr = row.get("Min")
            if not isinstance(min_expr, str) or not min_expr or min_expr == "None":
                medians.append(None)
                continue
            medians.append(min_formula_to_median_expr(min_expr))
        df["Median"] = medians

    build_metric_value_string(ac.dfs, ac.dfs_type, "per_kernel")

    for df_id, df in ac.dfs.items():
        if ac.dfs_type.get(df_id) != "metric_table":
            continue
        if "Median" not in df.columns or "Min" not in df.columns:
            continue
        for row_id, row in df.iterrows():
            med = row.get("Median")
            mn = row.get("Min")
            if (not isinstance(med, str) or not med) and isinstance(mn, str):
                if "to_min(" in mn:
                    df.at[row_id, "Median"] = mn.replace("to_min(", "to_median(", 1)

    workload = schema.Workload()
    workload.sys_info = sys_info
    workload.roofline_peaks = pd.DataFrame()
    workload.filter_gpu_ids = None
    workload.filter_kernel_ids = None
    workload.filter_dispatch_ids = None
    workload.raw_pmc = file_io.create_df_pmc(str(wl_dir), verbose=0)

    filtered = parser.apply_filters(
        workload,
        str(wl_dir),
        debug=False,
    )

    for df_id, df in ac.dfs.items():
        if ac.dfs_type.get(df_id) != "metric_table":
            continue
        if "Median" not in df.columns:
            continue
        exprs = ac.dfs_expressions.setdefault(df_id, [])
        for _, row in df.iterrows():
            med = row.get("Median")
            if isinstance(med, str) and med and med not in exprs:
                exprs.append(med)

    eval_metric(
        ac.dfs,
        ac.dfs_type,
        ac.dfs_expressions,
        sys_info.iloc[0],
        workload.roofline_peaks,
        filtered,
        debug=False,
    )

    skip_cols = {
        "Avg",
        "Min",
        "Max",
        "Median",
        "Unit",
        "Peak",
        "Percent of Peak",
        "Description",
        "Value",
    }
    out: dict[str, float | None] = {}
    for df_id, df in ac.dfs.items():
        if ac.dfs_type.get(df_id) != "metric_table":
            continue
        if "Median" not in df.columns:
            continue
        name_col = "Metric" if "Metric" in df.columns else None
        if name_col is None:
            for col in df.columns:
                if col not in skip_cols:
                    name_col = col
                    break
        if name_col is None:
            continue
        for row_id, row in df.iterrows():
            key = f"{row_id}|{row[name_col]}"
            med = row.get("Median")
            if med is None or (isinstance(med, float) and math.isnan(med)):
                out[key] = None
            elif isinstance(med, str):
                out[key] = to_float(med)
            else:
                try:
                    out[key] = float(med)
                except (TypeError, ValueError):
                    out[key] = None
    return out


def fallback_median_from_minmax(stats: MetricStats) -> float | None:
    """When Min==Max, median equals that value; else leave unknown."""
    if stats.min_v is None or stats.max_v is None:
        return None
    if abs(stats.min_v - stats.max_v) <= max(1e-9, 1e-9 * abs(stats.min_v)):
        return stats.min_v
    return None


def classify_offender(
    spp: MetricStats,
    leg: MetricStats,
    rel: float,
    n_spp: int | None,
    n_leg: int | None,
) -> str:
    """Heuristic: spurious vs real for a |rel|>100% median delta."""
    reasons: list[str] = []
    if n_spp is not None and n_spp < 10:
        reasons.append(f"few_spp_samples({n_spp})")
    if n_leg is not None and n_leg < 10:
        reasons.append(f"few_legacy_samples({n_leg})")
    if leg.median is not None and abs(leg.median) < DELTA_NEAR_ZERO:
        reasons.append("zero_baseline_median")
    if leg.avg is not None and abs(leg.avg) < DELTA_NEAR_ZERO:
        reasons.append("zero_baseline_avg")

    # Spike-skewed mean: Avg near Max while Min stays low.
    for label, st in (("spp", spp), ("legacy", leg)):
        if (
            st.avg is not None
            and st.min_v is not None
            and st.max_v is not None
            and st.max_v > st.min_v
        ):
            span = st.max_v - st.min_v
            if span > 0 and (st.avg - st.min_v) / span > 0.9:
                reasons.append(f"{label}_avg_near_max_spike")
            if st.median is not None and abs(st.avg - st.median) > 0.5 * max(
                abs(st.avg), abs(st.median), 1e-9
            ):
                reasons.append(f"{label}_avg_vs_median_skew")

    if not reasons and rel > REL_HIGH:
        return "likely_real"
    if reasons and all(
        r.startswith("few_") or "zero_baseline" in r or "spike" in r or "skew" in r
        for r in reasons
    ):
        if any("zero_baseline" in r or "spike" in r or "skew" in r for r in reasons):
            return "spurious:" + ",".join(reasons)
    if reasons:
        return "investigate:" + ",".join(reasons)
    return "likely_real"


def write_outputs(
    out_dir: Path,
    rows: list[dict[str, object]],
    summary_lines: list[str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "median_deltas_gt100.csv"
    md_path = out_dir / "median_recheck_500.md"
    fields = [
        "workload",
        "metric_id",
        "metric",
        "legacy_median",
        "spp_median",
        "rel_median",
        "legacy_avg",
        "spp_avg",
        "rel_avg",
        "legacy_min",
        "legacy_max",
        "spp_min",
        "spp_max",
        "n_legacy",
        "n_spp",
        "verdict",
    ]
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    md_path.write_text("\n".join(summary_lines) + "\n")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: <artifacts>/reports",
    )
    ap.add_argument(
        "--skip-reeval",
        action="store_true",
        help="Only use log Avg/Min/Max (Min==Max fallback for median)",
    )
    args = ap.parse_args()
    artifacts: Path = args.artifacts
    tag: str = args.tag
    out_dir = args.out_dir or (artifacts / "reports")
    logs = artifacts / "logs" / "gfx942"

    workloads = ["vcopy", "nbody", "mega_kernel"]
    all_high: list[dict[str, object]] = []
    summary: list[str] = [
        f"# SPP vs legacy median recheck ({tag})",
        "",
        "Iterations: 500 per workload. Compared **Median** of per-dispatch "
        "metric values (Min-formula inner expression) when workload dirs "
        "are available; otherwise Min==Max fallback.",
        "",
    ]

    total_compared = 0
    high_count = 0

    for wl in workloads:
        spp_log = logs / f"{wl}_spp.log"
        leg_log = logs / f"{wl}_legacy.log"
        if not spp_log.exists() or not leg_log.exists():
            summary.append(f"- **{wl}**: MISSING logs — skip")
            continue

        spp_m, n_spp = parse_log_stats(spp_log)
        leg_m, n_leg = parse_log_stats(leg_log)

        # Prefer dispatch counts from workload dirs when present.
        spp_dir = find_workload_dir(artifacts, tag, wl, "spp")
        leg_dir = find_workload_dir(artifacts, tag, wl, "legacy")
        if spp_dir and (spp_dir / "pmc_dispatch_info.csv").exists():
            n_spp = sum(1 for _ in (spp_dir / "pmc_dispatch_info.csv").open()) - 1
        if leg_dir and (leg_dir / "pmc_dispatch_info.csv").exists():
            n_leg = sum(1 for _ in (leg_dir / "pmc_dispatch_info.csv").open()) - 1

        if not args.skip_reeval and spp_dir and leg_dir:
            # Only re-eval metrics still missing Median (Avg/Min/Max panels).
            need = [
                k
                for k in set(spp_m) | set(leg_m)
                if (k in spp_m and spp_m[k].median is None)
                or (k in leg_m and leg_m[k].median is None)
            ]
            if need:
                try:
                    print(f"[median] re-evaluating {wl} spp @ {spp_dir}")
                    spp_med = compute_medians_for_workload(spp_dir, need)
                    print(f"[median] re-evaluating {wl} legacy @ {leg_dir}")
                    leg_med = compute_medians_for_workload(leg_dir, need)
                    for k, v in spp_med.items():
                        if k in spp_m and spp_m[k].median is None:
                            spp_m[k].median = v
                            spp_m[k].raw_median = "" if v is None else str(v)
                    for k, v in leg_med.items():
                        if k in leg_m and leg_m[k].median is None:
                            leg_m[k].median = v
                            leg_m[k].raw_median = "" if v is None else str(v)
                except Exception as exc:  # noqa: BLE001 — keep report going
                    summary.append(f"- **{wl}**: median re-eval failed: `{exc}`")
                    print(f"WARN: median re-eval failed for {wl}: {exc}")

        for st in list(spp_m.values()) + list(leg_m.values()):
            if st.median is None:
                st.median = fallback_median_from_minmax(st)

        keys = sorted(set(spp_m) & set(leg_m))
        wl_high = 0
        for key in keys:
            spp = spp_m[key]
            leg = leg_m[key]
            if spp.median is None or leg.median is None:
                continue
            total_compared += 1
            rel_m = relative_abs_delta(leg.median, spp.median)
            if rel_m is None or rel_m <= REL_HIGH:
                continue
            high_count += 1
            wl_high += 1
            rel_a = None
            if spp.avg is not None and leg.avg is not None:
                rel_a = relative_abs_delta(leg.avg, spp.avg)
            verdict = classify_offender(spp, leg, rel_m, n_spp, n_leg)
            all_high.append({
                "workload": wl,
                "metric_id": spp.metric_id,
                "metric": spp.name,
                "legacy_median": f"{leg.median:.6g}",
                "spp_median": f"{spp.median:.6g}",
                "rel_median": f"{100 * rel_m:.1f}%",
                "legacy_avg": "" if leg.avg is None else f"{leg.avg:.6g}",
                "spp_avg": "" if spp.avg is None else f"{spp.avg:.6g}",
                "rel_avg": "" if rel_a is None else f"{100 * rel_a:.1f}%",
                "legacy_min": leg.raw_min,
                "legacy_max": leg.raw_max,
                "spp_min": spp.raw_min,
                "spp_max": spp.raw_max,
                "n_legacy": n_leg if n_leg is not None else "",
                "n_spp": n_spp if n_spp is not None else "",
                "verdict": verdict,
            })

        summary.append(
            f"- **{wl}**: compared_medians ok; "
            f"n_dispatch spp={n_spp} legacy={n_leg}; "
            f"|rel_median|>100%: {wl_high}"
        )

    summary.extend([
        "",
        "## Overall",
        f"- Metrics with both-side medians compared: **{total_compared}**",
        f"- |rel| > 100% on medians: **{high_count}**",
        "",
        "## |rel| > 100% median deltas",
        "",
    ])
    if not all_high:
        summary.append("_None._")
        pass_fail = "PASS — no median |rel|>100% deltas"
    else:
        summary.append(
            "| workload | id | metric | leg med | spp med | |rel| | verdict |"
        )
        summary.append("|---|---|---|---|---|---|---|")
        for row in sorted(
            all_high,
            key=lambda r: -float(str(r["rel_median"]).rstrip("%")),
        ):
            summary.append(
                f"| {row['workload']} | {row['metric_id']} | "
                f"{row['metric']} | {row['legacy_median']} | "
                f"{row['spp_median']} | {row['rel_median']} | "
                f"{row['verdict']} |"
            )
        spurious = sum(1 for r in all_high if str(r["verdict"]).startswith("spurious"))
        realish = sum(1 for r in all_high if r["verdict"] == "likely_real")
        if realish == 0:
            pass_fail = (
                f"PASS-with-caveats — {high_count} |rel|>100% but "
                f"{spurious} classified spurious / none likely_real"
            )
        else:
            pass_fail = (
                f"FAIL/INVESTIGATE — {realish} likely_real of {high_count} "
                f"|rel|>100% median deltas"
            )

    summary.extend(["", "## Judgment", f"**{pass_fail}**", ""])
    write_outputs(out_dir, all_high, summary)
    print(pass_fail)
    print(f"high_count={high_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
