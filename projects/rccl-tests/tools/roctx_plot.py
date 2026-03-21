#!/usr/bin/env python3
"""Plot median kernel durations from multiple roctx_perf_run.py run directories.

Produces one subplot per dtype, with one line per (library, placement) combination.
Color encodes the library; line style (solid/dashed) and marker encode placement.

Usage:
    python3 tools/roctx_plot.py perf-runs/20260306-* --output comparison.png
    python3 tools/roctx_plot.py perf-runs/20260306-* --show
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from roctx_analyze import (
    BUS_BW_FACTOR,
    correlate,
    discover_multi_run_groups,
    discover_trace_files,
    generate_report,
    iqr_outliers,
    load_run_metadata,
    mad_outliers,
    parse_kernel_csv,
    parse_marker_csv,
)

# ---------------------------------------------------------------------------
# Library identification
# ---------------------------------------------------------------------------

def get_lib_id(run_dir):
    """Derive a short human-readable label for the librccl used in a run."""
    meta_path = os.path.join(run_dir, "metadata.json")
    if not os.path.isfile(meta_path):
        return os.path.basename(run_dir)
    with open(meta_path) as f:
        meta = json.load(f)
    librccl = meta.get("librccl", {})
    git_hash = librccl.get("git_hash", "")  # e.g. "rccl-parallel-build:a203c83+"
    if git_hash:
        branch, _, rest = git_hash.partition(":")
        short_hash = rest.rstrip("+")
        if branch in ("HEAD", ""):
            # System install: include version for clarity
            ver = librccl.get("rccl_version", "")
            return f"v{ver}:{short_hash}" if ver else short_hash
        # Local build: shorten long branch names to 20 chars
        if len(branch) > 20:
            branch = branch[:20]
        return f"{branch}:{short_hash}"
    ver = librccl.get("rccl_version")
    if ver:
        return f"v{ver}"
    return os.path.basename(run_dir)


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_data(run_dir, outlier_fn):
    """Return {dtype: {in_place: [(size_bytes, median_ns), ...]}} for a run dir."""
    result = {}
    multi_groups = discover_multi_run_groups(run_dir)
    if not multi_groups:
        print(f"WARNING: {run_dir} does not look like a multi-run directory, skipping.",
              file=sys.stderr)
        return result

    for (collective, dtype), subdirs in multi_groups.items():
        all_samples = defaultdict(list)
        for d in subdirs:
            for marker_path, kernel_path in discover_trace_files(d):
                markers = parse_marker_csv(marker_path)
                kernels = parse_kernel_csv(kernel_path)
                for key, durations in correlate(markers, kernels).items():
                    all_samples[key].extend(durations)

        rows = generate_report(all_samples, outlier_fn)

        by_place = defaultdict(list)
        for r in rows:
            if r["median"] is not None:
                by_place[r["in_place"]].append((r["size"], r["median"]))

        dtype_key = dtype  # e.g. "bfloat16", "float"
        if dtype_key not in result:
            result[dtype_key] = {}
        for in_place, points in by_place.items():
            result[dtype_key][in_place] = sorted(points)

    return result


# ---------------------------------------------------------------------------
# Axis formatting
# ---------------------------------------------------------------------------

def fmt_size_axis(nbytes, _pos=None):
    if nbytes >= 1024 ** 3:
        return f"{int(nbytes // 1024 ** 3)}G"
    if nbytes >= 1024 ** 2:
        return f"{int(nbytes // 1024 ** 2)}M"
    if nbytes >= 1024:
        return f"{int(nbytes // 1024)}K"
    return str(int(nbytes))


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

PLACE_STYLE = {
    0: dict(linestyle="--", marker="s", markersize=5, label_prefix="oop"),
    1: dict(linestyle="-",  marker="o", markersize=5, label_prefix="ip"),
}

# Prefer a colorblind-friendly palette; fall back to the default cycle.
_PALETTE = ["#377EB8", "#E41A1C", "#4DAF4A", "#FF7F00", "#984EA3", "#A65628"]


def plot_comparison(lib_data, all_dtypes, output_path, show=False):
    n = len(all_dtypes)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5.5), squeeze=False)
    fig.patch.set_facecolor("#f8f8f8")

    for col, dtype in enumerate(all_dtypes):
        ax = axes[0][col]
        ax.set_facecolor("#fdfdfd")

        for lib_idx, (lib_id, data) in enumerate(lib_data):
            color = _PALETTE[lib_idx % len(_PALETTE)]
            dtype_data = data.get(dtype, {})

            for in_place, style in PLACE_STYLE.items():
                points = dtype_data.get(in_place)
                if not points:
                    continue
                sizes     = [p[0] for p in points]
                medians_us = [p[1] / 1_000.0 for p in points]
                label = f"{style['label_prefix']} · {lib_id}"
                ax.plot(
                    sizes, medians_us,
                    color=color,
                    linestyle=style["linestyle"],
                    marker=style["marker"],
                    markersize=style["markersize"],
                    linewidth=1.6,
                    label=label,
                )

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_size_axis))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(
            lambda v, _: f"{v:.0f}" if v >= 1 else f"{v:.2g}"
        ))
        ax.tick_params(axis="x", labelrotation=45, labelsize=8)
        ax.tick_params(axis="y", labelsize=9)
        ax.set_xlabel("Message size", fontsize=10)
        ax.set_ylabel("Median kernel time (µs)", fontsize=10)
        ax.set_title(f"AllReduce — {dtype}", fontsize=12, fontweight="bold")
        ax.grid(True, which="major", linestyle=":", linewidth=0.6, alpha=0.6)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.3, alpha=0.35)
        ax.legend(fontsize=7.5, framealpha=0.9, loc="upper left")

    # Legend explanation footnote
    fig.text(0.5, -0.02,
             "Line style: solid = in-place, dashed = out-of-place   |   Color = library",
             ha="center", fontsize=8.5, color="#555555")

    fig.suptitle("AllReduce kernel time — library comparison", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    if show:
        matplotlib.use("TkAgg")
        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot median AllReduce kernel time across librccl builds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="example:\n  %(prog)s perf-runs/20260306-* --output comparison.png",
    )
    parser.add_argument("run_dirs", nargs="+", help="Run directories to compare")
    parser.add_argument("--output", default="comparison.png",
                        help="Output image file (default: comparison.png)")
    parser.add_argument("--labels", type=str, default=None,
                        help="Comma-separated library labels, one per run dir "
                             "(e.g. --labels 'rocm-7.0.0,fast-LDS,fast-flat')")
    parser.add_argument("--outlier", default="mad", choices=["mad", "iqr"],
                        help="Outlier detection method (default: mad)")
    parser.add_argument("--mad-threshold", type=float, default=3.5)
    parser.add_argument("--iqr-factor", type=float, default=1.5)
    parser.add_argument("--show", action="store_true",
                        help="Show interactive window after saving")
    args = parser.parse_args()

    if args.outlier == "mad":
        outlier_fn = lambda vals: mad_outliers(vals, threshold=args.mad_threshold)
    else:
        outlier_fn = lambda vals: iqr_outliers(vals, factor=args.iqr_factor)

    custom_labels = [l.strip() for l in args.labels.split(",")] if args.labels else []
    if custom_labels and len(custom_labels) != len(args.run_dirs):
        parser.error(f"--labels has {len(custom_labels)} entries but {len(args.run_dirs)} run dirs were given")

    lib_data = []
    for i, run_dir in enumerate(args.run_dirs):
        lib_id = custom_labels[i] if i < len(custom_labels) else get_lib_id(run_dir)
        data = collect_data(run_dir, outlier_fn)
        if data:
            lib_data.append((lib_id, data))
            print(f"  {os.path.basename(run_dir)} → {lib_id}  "
                  f"dtypes={list(data.keys())}")
        else:
            print(f"  WARNING: no data from {run_dir}", file=sys.stderr)

    if not lib_data:
        print("No data to plot.", file=sys.stderr)
        sys.exit(1)

    # Preserve dtype order as encountered
    all_dtypes = list(dict.fromkeys(
        dtype
        for _, data in lib_data
        for dtype in data
    ))

    plot_comparison(lib_data, all_dtypes, args.output, show=args.show)


if __name__ == "__main__":
    main()
