#!/usr/bin/env python3
"""
Plot box-and-whisker charts of RCCL kernel durations across ranks.

Reads rocprofv3 SQLite databases from timestamped run directories under
DATA_DIR.  When given multiple directories, produces grouped box plots
comparing runs side by side.

Usage:
    python plot_kernel_variance.py [RUN_DIR ...]

With no arguments, uses the most recent timestamped directory under DATA_DIR.
"""

import sqlite3
import json
import os
import sys
import re
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

DATA_DIR = "/work/lmeadows/data"
OUT_DIR = "/work/lmeadows/data/plots"
SKIP_BENCHMARKS = {"all_reduce_bias", "plots"}

RUN_COLORS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
]


def find_run_dirs(data_dir):
    """Return timestamped run directories sorted newest-first."""
    dirs = []
    if not os.path.isdir(data_dir):
        return dirs
    for name in os.listdir(data_dir):
        path = os.path.join(data_dir, name)
        if os.path.isdir(path) and re.match(r"\d{8}_\d{6}$", name):
            dirs.append(path)
    return sorted(dirs, reverse=True)


def _is_run_dir(path):
    return os.path.isdir(path) and re.match(r"\d{8}_\d{6}$", os.path.basename(path))


def resolve_run_dirs(args):
    """Return list of run-directory paths from CLI args, or the most recent."""
    if args:
        dirs = [os.path.abspath(d) for d in args if _is_run_dir(d)]
        if not dirs:
            print("Error: no valid timestamped run directories in arguments", file=sys.stderr)
            sys.exit(1)
        return dirs
    candidates = find_run_dirs(DATA_DIR)
    if not candidates:
        print(f"No timestamped run directories found in {DATA_DIR}", file=sys.stderr)
        sys.exit(1)
    return [candidates[0]]


def run_label(run_dir):
    return os.path.basename(run_dir)


# ── data extraction ──────────────────────────────────────────────────────


def find_results_db(rank_dir):
    dbs = glob.glob(os.path.join(rank_dir, "**", "*_results.db"), recursive=True)
    return dbs[0] if dbs else None


def extract_rank_data(db_path):
    """Return list of (label, kernel_duration_ns) from one rank's database."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute("""
        SELECT R.start, R.end, R.extdata
        FROM regions R
        WHERE R.category = 'MARKER_CORE_RANGE_API'
        ORDER BY R.start
    """).fetchall()

    regions = []
    for start, end, extdata in rows:
        try:
            msg = json.loads(extdata).get("message", "")
        except (json.JSONDecodeError, TypeError):
            continue
        if not msg or msg.startswith("RCCL-PROTO:"):
            continue
        regions.append((start, end, msg))

    kernels = conn.execute("""
        SELECT K.start, (K.end - K.start) as duration
        FROM kernels K
        WHERE K.name LIKE '%ncclDev%'
        ORDER BY K.start
    """).fetchall()
    conn.close()

    results = []
    ri = 0
    for k_start, k_dur in kernels:
        while ri < len(regions) - 1 and regions[ri + 1][0] <= k_start:
            ri += 1
        if ri < len(regions) and regions[ri][0] <= k_start <= regions[ri][1]:
            results.append((regions[ri][2], k_dur))
    return results


def human_bytes(n):
    for unit in ("B", "K", "M", "G"):
        if n < 1024:
            return f"{n}{unit}" if n == int(n) else f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}T"


def collect_run_benchmark_data(bench_dir):
    """Return {(size_bytes, placement): [dur_µs, …]} across all ranks."""
    rank_dirs = sorted(glob.glob(os.path.join(bench_dir, "rank_*")))
    data = defaultdict(list)
    for rank_dir in rank_dirs:
        db_path = find_results_db(rank_dir)
        if not db_path:
            continue
        for label, dur_ns in extract_rank_data(db_path):
            parts = label.split(":")
            if len(parts) != 3:
                continue
            size = int(parts[1])
            placement = parts[2]
            data[(size, placement)].append(dur_ns / 1e3)
    return data if data else None


def discover_benchmarks(run_dirs):
    """Return sorted benchmark names present in any run directory."""
    benchmarks = set()
    for run_dir in run_dirs:
        if not os.path.isdir(run_dir):
            continue
        for name in os.listdir(run_dir):
            path = os.path.join(run_dir, name)
            if os.path.isdir(path) and name not in SKIP_BENCHMARKS:
                benchmarks.add(name)
    return sorted(benchmarks)


# ── plotting ─────────────────────────────────────────────────────────────


def plot_benchmark(bench_name, runs, out_dir):
    """
    runs: [(label, data_or_None), …]
    data:  {(size_bytes, placement): [dur_µs, …]}
    """
    active_runs = [(lbl, data) for lbl, data in runs if data]
    if not active_runs:
        print(f"  SKIP {bench_name}: no data in any run")
        return

    n_runs = len(active_runs)
    multi = n_runs > 1

    for placement in ("outofplace", "inplace"):
        all_sizes = set()
        run_size_data = []
        for run_lbl, data in active_runs:
            size_data = {}
            for (size, plc), durs in data.items():
                if plc != placement:
                    continue
                size_data[size] = durs
                all_sizes.add(size)
            run_size_data.append((run_lbl, size_data))

        if not all_sizes:
            continue

        sizes = sorted(all_sizes)

        MAX_PER_PLOT = 7
        chunks = [
            list(range(i, min(i + MAX_PER_PLOT, len(sizes))))
            for i in range(0, len(sizes), MAX_PER_PLOT)
        ]

        for ci, chunk_indices in enumerate(chunks):
            chunk_sizes = [sizes[i] for i in chunk_indices]
            n_cs = len(chunk_sizes)

            fig_width = max(8, n_cs * max(1.2, 0.7 * n_runs))
            fig, ax = plt.subplots(figsize=(fig_width, 5))

            total_width = 0.8
            box_width = total_width / n_runs
            for ri, (rlbl, size_data) in enumerate(run_size_data):
                box_data = [size_data.get(s, []) for s in chunk_sizes]
                positions = [
                    j + 1 - total_width / 2 + box_width * (ri + 0.5)
                    for j in range(n_cs)
                ]
                color = RUN_COLORS[ri % len(RUN_COLORS)]
                bp = ax.boxplot(
                    box_data,
                    positions=positions,
                    widths=box_width * 0.85,
                    patch_artist=True,
                    showfliers=True,
                    showmeans=False,
                    medianprops=dict(color="black", linewidth=1.5),
                    flierprops=dict(marker="o", markersize=3, alpha=0.4,
                                    markerfacecolor=color, markeredgecolor=color),
                    whiskerprops=dict(color=color),
                    capprops=dict(color=color),
                    manage_ticks=False,
                )
                for patch in bp["boxes"]:
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                ax.plot([], [], color=color, linewidth=6, alpha=0.7, label=rlbl)

            ax.set_xticks(range(1, n_cs + 1))
            ax.set_xticklabels([human_bytes(s) for s in chunk_sizes])
            if n_runs > 1:
                ax.legend(fontsize=8, loc="upper left")

            ax.set_xlabel("Message Size")
            ax.set_ylabel("Kernel Duration (µs)")
            suffix = f" (part {ci+1})" if len(chunks) > 1 else ""
            title = f"{bench_name} — {placement}{suffix}"
            if not multi:
                title += f"\n[{run_size_data[0][0]}]"
            ax.set_title(title)
            ax.grid(axis="y", alpha=0.3)

            all_vals = []
            for _, sd in run_size_data:
                for s in chunk_sizes:
                    all_vals.extend(sd.get(s, []))
            if all_vals:
                vmin, vmax = min(all_vals), max(all_vals)
                if vmax / (vmin + 1e-9) > 20:
                    ax.set_yscale("log")

            fig.tight_layout()
            part = f"_part{ci+1}" if len(chunks) > 1 else ""
            fname = f"{bench_name}_{placement}{part}.png"
            fig.savefig(os.path.join(out_dir, fname), dpi=150)
            plt.close(fig)
            print(f"  wrote {fname}")


# ── main ─────────────────────────────────────────────────────────────────


def main():
    run_dirs = resolve_run_dirs(sys.argv[1:])
    labels = [run_label(d) for d in run_dirs]

    print("Run directories:")
    for lbl, d in zip(labels, run_dirs):
        print(f"  {lbl}: {d}")
    print()

    os.makedirs(OUT_DIR, exist_ok=True)
    benchmarks = discover_benchmarks(run_dirs)

    for bench in benchmarks:
        print(f"Processing {bench}...")
        runs = []
        for lbl, run_dir in zip(labels, run_dirs):
            bench_dir = os.path.join(run_dir, bench)
            if os.path.isdir(bench_dir):
                data = collect_run_benchmark_data(bench_dir)
                runs.append((lbl, data))
            else:
                runs.append((lbl, None))
        plot_benchmark(bench, runs, OUT_DIR)


if __name__ == "__main__":
    main()
