#!/usr/bin/env python3
###############################################################################
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Compare rocshmem functional test performance across build variants.

Parses log files produced by scripts/functional_tests/driver.sh, computes
three metrics per test (small-message latency, peak bandwidth, area under
the bandwidth curve), generates per-test latency-vs-size plots, and produces
a summary heatmap of performance differences relative to a baseline.

Usage:
    python3 scripts/perf_compare.py \\
        --baseline  build-develop/logs-develop \\
        --variants  sdma-off:build-sdma-off/logs-sdma-off \\
                    sdma-disabled:build-sdma-on/logs-sdma-disabled \\
                    sdma-enabled:build-sdma-on/logs-sdma-enabled \\
        --outdir    plots/
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

# Matches ANSI escape sequences
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# Extracts test name and PE/WG/thread counts from log filename
_FNAME_RE = re.compile(r"^(.+?)_n(\d+)_w(\d+)_z(\d+)")


def parse_log(filepath: Path) -> pd.DataFrame | None:
    """Parse a single driver log file into a DataFrame.

    Returns None if the file contains no usable data rows.
    """
    rows = []
    with open(filepath, "r", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            # Skip comment/header lines and ANSI warning lines
            if line.startswith("#") or line.startswith("###"):
                continue
            if _ANSI_RE.search(line):
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                row = {
                    "volume": int(parts[0]),
                    "msg_size": int(parts[1]),
                    "num_msgs": int(parts[2]),
                    "latency_us": float(parts[3]),
                    "bandwidth_gbs": float(parts[4]),
                    "msg_rate": float(parts[5]),
                }
                rows.append(row)
            except (ValueError, IndexError):
                continue
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df.sort_values("msg_size", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def collect_variant(log_dir: Path) -> dict[str, pd.DataFrame]:
    """Collect all test results from a single log directory.

    Returns a dict mapping test_key -> DataFrame.  The key encodes the test
    name and its PE/WG/thread configuration (e.g. "put_pe2w1z1").  When
    multiple log files map to the same key (different max message sizes),
    the one with the most data rows is kept.
    """
    results: dict[str, pd.DataFrame] = {}
    for logfile in sorted(log_dir.glob("*.log")):
        m = _FNAME_RE.match(logfile.stem)
        if not m:
            continue
        test_name, n, w, z = m.group(1), m.group(2), m.group(3), m.group(4)
        test_key = f"{test_name}_pe{n}w{w}z{z}"
        df = parse_log(logfile)
        if df is None or df.empty:
            continue
        if test_key not in results or len(df) > len(results[test_key]):
            results[test_key] = df
    return results


def _remove_outliers_iqr(values: np.ndarray) -> np.ndarray:
    """Remove outliers using the IQR method (1.5x IQR fence)."""
    if len(values) < 4:
        return values
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return values[(values >= lo) & (values <= hi)]


def collect_variant_multi(log_dirs: list[Path]) -> dict[str, pd.DataFrame]:
    """Collect test results from multiple iteration directories.

    For each test_key and message size, gathers latency/bandwidth values
    across all iterations, removes outliers via IQR, and returns the
    median.  This produces a single clean DataFrame per test_key.
    """
    # Step 1: gather all DataFrames per test_key across iterations
    raw: dict[str, list[pd.DataFrame]] = {}
    for log_dir in log_dirs:
        single = collect_variant(log_dir)
        for key, df in single.items():
            raw.setdefault(key, []).append(df)

    # Step 2: for each test_key, aggregate across iterations
    results: dict[str, pd.DataFrame] = {}
    for key, dfs in raw.items():
        # Stack all iterations, group by msg_size
        combined = pd.concat(dfs, ignore_index=True)
        grouped = combined.groupby("msg_size")

        rows = []
        for msg_size, grp in grouped:
            lat_clean = _remove_outliers_iqr(grp["latency_us"].values)
            bw_clean = _remove_outliers_iqr(grp["bandwidth_gbs"].values)
            rows.append({
                "msg_size": msg_size,
                "volume": grp["volume"].iloc[0],
                "num_msgs": grp["num_msgs"].iloc[0],
                "latency_us": float(np.median(lat_clean)) if len(lat_clean) else grp["latency_us"].median(),
                "bandwidth_gbs": float(np.median(bw_clean)) if len(bw_clean) else grp["bandwidth_gbs"].median(),
                "msg_rate": grp["msg_rate"].median(),
            })

        df_agg = pd.DataFrame(rows).sort_values("msg_size").reset_index(drop=True)
        results[key] = df_agg

    return results


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def small_msg_latency(df: pd.DataFrame) -> float:
    """Latency (us) at the smallest message size."""
    return df.iloc[0]["latency_us"]


def peak_bandwidth(df: pd.DataFrame) -> float:
    """Maximum bandwidth (GB/s) across all message sizes."""
    return df["bandwidth_gbs"].max()


def bandwidth_auc(df: pd.DataFrame) -> float:
    """Area under the bandwidth curve (trapezoidal, log2 msg_size axis)."""
    x = np.log2(df["msg_size"].values.astype(float))
    y = df["bandwidth_gbs"].values
    if len(x) < 2:
        return y[0] if len(y) else 0.0
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return float(_trapz(y, x))


METRICS = {
    "latency_us": small_msg_latency,
    "peak_bw_gbs": peak_bandwidth,
    "bw_auc": bandwidth_auc,
}

# For the heatmap: lower is better for latency, higher is better for BW/AUC
_LOWER_IS_BETTER = {"latency_us"}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_VARIANT_STYLES = {
    "baseline": {"color": "black", "linestyle": "-", "marker": "o"},
    "sdma-off": {"color": "tab:blue", "linestyle": "--", "marker": "s"},
    "sdma-disabled": {"color": "tab:orange", "linestyle": "-.", "marker": "^"},
    "sdma-enabled": {"color": "tab:red", "linestyle": "-", "marker": "D"},
}


def _fmt_size(x, _pos=None):
    """Format byte count as B / KiB / MiB."""
    if x < 1:
        return ""
    if x < 1024:
        return f"{int(x)} B"
    if x < 1024 * 1024:
        v = x / 1024
        return f"{int(v)} KiB" if v == int(v) else f"{v:.1f} KiB"
    v = x / (1024 * 1024)
    return f"{int(v)} MiB" if v == int(v) else f"{v:.1f} MiB"


def _fmt_latency(x, _pos=None):
    """Format latency in plain decimal (no scientific notation)."""
    if x < 1:
        return f"{x:.2f}"
    if x < 10:
        return f"{x:.1f}"
    return f"{int(x)}" if x == int(x) else f"{x:.1f}"


def plot_latency(
    test_key: str,
    baseline_df: pd.DataFrame,
    variant_dfs: dict[str, pd.DataFrame],
    outdir: Path,
) -> None:
    """Generate a latency-vs-message-size plot for one test."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for label, df in [("baseline", baseline_df), *variant_dfs.items()]:
        style = _VARIANT_STYLES.get(label, {})
        ax.plot(
            df["msg_size"],
            df["latency_us"],
            label=label,
            markersize=4,
            linewidth=1.5,
            **style,
        )

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(_fmt_size))

    # Only use log scale if all latency values are positive
    all_positive = all(
        (df["latency_us"] > 0).all()
        for df in [baseline_df, *variant_dfs.values()]
    )
    if all_positive:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(plt.LogLocator(base=10, subs=[1, 2, 5]))
        ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.yaxis.set_major_formatter(plt.FuncFormatter(_fmt_latency))

    ax.set_xlabel("Message Size")
    ax.set_ylabel("Latency (us)")
    ax.set_title(test_key)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.autofmt_xdate(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(outdir / f"{test_key}_latency.png", dpi=150)
    plt.close(fig)


def make_heatmap(
    baseline_data: dict[str, pd.DataFrame],
    variant_datasets: dict[str, dict[str, pd.DataFrame]],
    outdir: Path,
) -> None:
    """Generate a summary heatmap of % performance change vs baseline."""
    # Compute metrics for baseline
    baseline_metrics: dict[str, dict[str, float]] = {}
    for test_name, df in baseline_data.items():
        baseline_metrics[test_name] = {
            m: fn(df) for m, fn in METRICS.items()
        }

    # Build a table: rows = tests, columns = (variant, metric)
    records = []
    for vname, vdata in variant_datasets.items():
        for test_name in sorted(baseline_metrics.keys()):
            if test_name not in vdata:
                continue
            bm = baseline_metrics[test_name]
            vm = {m: fn(vdata[test_name]) for m, fn in METRICS.items()}
            for metric in METRICS:
                if bm[metric] == 0:
                    pct = 0.0
                else:
                    pct = (vm[metric] - bm[metric]) / abs(bm[metric]) * 100.0
                records.append({
                    "test": test_name,
                    "variant": vname,
                    "metric": metric,
                    "pct_change": pct,
                })

    if not records:
        print("No common tests found for heatmap.", file=sys.stderr)
        return

    df_all = pd.DataFrame(records)

    # Pivot: rows=test, columns=(variant, metric)
    pivot = df_all.pivot_table(
        index="test",
        columns=["variant", "metric"],
        values="pct_change",
    )
    pivot.sort_index(inplace=True)

    # For display: flip sign on latency so green = improvement everywhere
    display = pivot.copy()
    for col in display.columns:
        variant, metric = col
        if metric in _LOWER_IS_BETTER:
            display[col] = -display[col]

    # Rename columns for readability
    col_labels = []
    for variant, metric in display.columns:
        short_metric = {"latency_us": "lat", "peak_bw_gbs": "bw", "bw_auc": "auc"}[metric]
        col_labels.append(f"{variant}\n{short_metric}")
    display.columns = col_labels

    # Clamp display values for color mapping (annotations stay original)
    annot_df = display.copy()

    # Custom non-linear colormap:
    #   -100% deep red  →  -50% red  →  -20% orange  →  0 white
    #   0 white  →  +5% light green  →  +20% green  →  +100% dark green
    #   +100% → +1000%+ deep teal
    boundaries = [-100, -50, -20, -10, -5, 0, 5, 10, 20, 50, 100, 500, 5000]
    from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
    colors = [
        "#b2182b",  # -100 to -50: deep red
        "#d6604d",  # -50 to -20: red
        "#f4a582",  # -20 to -10: orange
        "#fddbc7",  # -10 to -5: light orange
        "#f7f7f7",  # -5 to 0: near white
        "#e5f5e0",  #  0 to +5: very light green
        "#a1d99b",  # +5 to +10: light green
        "#74c476",  # +10 to +20: medium green
        "#31a354",  # +20 to +50: green
        "#006d2c",  # +50 to +100: dark green
        "#00441b",  # +100 to +500: very dark green
        "#002a11",  # +500 to +5000: deepest green
    ]
    cmap = LinearSegmentedColormap.from_list("perf", colors, N=len(colors))
    norm = BoundaryNorm(boundaries, cmap.N, clip=True)

    # Clamp values into boundary range for coloring
    clamped = display.clip(lower=boundaries[0], upper=boundaries[-1])

    # Format annotations: use "k" suffix for values >= 1000
    def _fmt_annot(v):
        if not np.isfinite(v):
            return ""
        if abs(v) >= 1000:
            return f"{v/1000:.0f}k"
        return f"{v:.1f}"

    annot_text = annot_df.map(_fmt_annot)

    # Plot
    n_rows = len(display)
    fig_height = max(8, n_rows * 0.25 + 2)
    fig, ax = plt.subplots(figsize=(14, fig_height))

    sns.heatmap(
        clamped,
        cmap=cmap,
        norm=norm,
        annot=annot_text,
        fmt="",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "% change (green = better)", "ticks": boundaries},
    )
    ax.set_title("Performance vs develop baseline (% change)")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(outdir / "heatmap_summary.png", dpi=150)
    plt.close(fig)

    # Also save raw numbers as CSV
    pivot.to_csv(outdir / "heatmap_data.csv")
    print(f"Heatmap saved to {outdir / 'heatmap_summary.png'}")
    print(f"Raw data saved to {outdir / 'heatmap_data.csv'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _resolve_dirs(pattern: str) -> list[Path]:
    """Resolve a path or glob pattern to a list of directories."""
    p = Path(pattern)
    if p.is_dir():
        return [p]
    # Try glob from parent
    parent = p.parent if p.parent != p else Path(".")
    matches = sorted(parent.glob(p.name))
    dirs = [m for m in matches if m.is_dir()]
    if not dirs:
        print(f"Warning: no directories matched '{pattern}'", file=sys.stderr)
    return dirs


def main():
    parser = argparse.ArgumentParser(
        description="Compare rocshmem functional test performance across variants.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--baseline", required=True,
        help="Baseline log dir(s) — single path or glob (e.g. build-develop/logs-heatmap-*)",
    )
    parser.add_argument(
        "--variants", nargs="+", required=True,
        help="Variant specs as name:path-or-glob (e.g. sdma-off:build-sdma-off/logs-heatmap-*)",
    )
    parser.add_argument(
        "--outdir", type=Path, default=Path("plots"),
        help="Output directory for plots (default: plots/)",
    )
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    # Parse variant specs
    variant_dir_lists: dict[str, list[Path]] = {}
    for spec in args.variants:
        if ":" not in spec:
            parser.error(f"Invalid variant spec '{spec}', expected name:path")
        name, path = spec.split(":", 1)
        variant_dir_lists[name] = _resolve_dirs(path)

    # Collect data — use multi-iteration aggregation when multiple dirs
    baseline_dirs = _resolve_dirs(args.baseline)
    if len(baseline_dirs) > 1:
        print(f"Parsing baseline: {len(baseline_dirs)} iterations from {args.baseline}")
        baseline_data = collect_variant_multi(baseline_dirs)
    else:
        print(f"Parsing baseline: {baseline_dirs[0]}")
        baseline_data = collect_variant(baseline_dirs[0])
    print(f"  Found {len(baseline_data)} tests")

    variant_datasets: dict[str, dict[str, pd.DataFrame]] = {}
    for vname, vdirs in variant_dir_lists.items():
        if len(vdirs) > 1:
            print(f"Parsing variant '{vname}': {len(vdirs)} iterations")
            variant_datasets[vname] = collect_variant_multi(vdirs)
        elif len(vdirs) == 1:
            print(f"Parsing variant '{vname}': {vdirs[0]}")
            variant_datasets[vname] = collect_variant(vdirs[0])
        else:
            print(f"Skipping variant '{vname}': no directories found")
            continue
        print(f"  Found {len(variant_datasets[vname])} tests")

    # Find tests common to baseline and at least one variant
    all_test_names = set(baseline_data.keys())
    for vdata in variant_datasets.values():
        all_test_names &= set(vdata.keys())

    print(f"\n{len(all_test_names)} tests common across all variants")

    # Generate per-test plots
    plot_dir = args.outdir / "per_test"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for test_name in sorted(all_test_names):
        vdfs = {vn: vd[test_name] for vn, vd in variant_datasets.items()
                if test_name in vd}
        plot_latency(test_name, baseline_data[test_name], vdfs, plot_dir)
    print(f"Generated {len(all_test_names)} per-test plots in {plot_dir}/")

    # Generate summary heatmap
    make_heatmap(baseline_data, variant_datasets, args.outdir)

    # Print summary table to stdout
    print("\n" + "=" * 80)
    print("SUMMARY: metrics for tests common to all variants")
    print("=" * 80)
    header = f"{'Test':<30} {'Metric':<12}"
    for vname in variant_datasets:
        header += f" {vname:>14}"
    print(header)
    print("-" * len(header))

    for test_name in sorted(all_test_names):
        bdf = baseline_data[test_name]
        for metric_name, metric_fn in METRICS.items():
            bval = metric_fn(bdf)
            row = f"{test_name:<30} {metric_name:<12}"
            for vname in variant_datasets:
                if test_name in variant_datasets[vname]:
                    vval = metric_fn(variant_datasets[vname][test_name])
                    if bval != 0:
                        pct = (vval - bval) / abs(bval) * 100
                        row += f" {pct:>+13.1f}%"
                    else:
                        row += f" {'N/A':>14}"
                else:
                    row += f" {'missing':>14}"
            print(row)


if __name__ == "__main__":
    main()
