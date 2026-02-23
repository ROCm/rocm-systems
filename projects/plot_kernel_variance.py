#!/usr/bin/env python3
"""
Plot box-and-whisker charts of RCCL kernel durations across ranks.

Reads rocprofv3 SQLite databases from /work/lmeadows/data/<benchmark>/rank_*,
joins kernel dispatches with roctx marker regions to label each kernel by
(collective, byte_size, inplace/outofplace), then plots per-size kernel
duration distributions across all 8 ranks.
"""

import sqlite3
import json
import os
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

DATA_DIR = "/work/lmeadows/data"
OUT_DIR = "/work/lmeadows/data/plots"


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
        if not msg:
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


def plot_benchmark(bench_name, bench_dir):
    rank_dirs = sorted(glob.glob(os.path.join(bench_dir, "rank_*")))
    if not rank_dirs:
        print(f"  SKIP {bench_name}: no rank directories")
        return

    all_data = defaultdict(lambda: defaultdict(list))
    for rank_dir in rank_dirs:
        rank = os.path.basename(rank_dir)
        db_path = find_results_db(rank_dir)
        if not db_path:
            print(f"  SKIP {rank}: no results.db")
            continue
        for label, dur_ns in extract_rank_data(db_path):
            all_data[label][rank].append(dur_ns / 1e3)  # convert to µs

    if not all_data:
        print(f"  SKIP {bench_name}: no kernel data found")
        return

    oop_labels = sorted(
        [l for l in all_data if l.endswith(":outofplace")],
        key=lambda l: int(l.split(":")[1]),
    )
    ip_labels = sorted(
        [l for l in all_data if l.endswith(":inplace")],
        key=lambda l: int(l.split(":")[1]),
    )

    for placement, labels in [("outofplace", oop_labels), ("inplace", ip_labels)]:
        if not labels:
            continue

        sizes = [int(l.split(":")[1]) for l in labels]

        # Split into groups so each plot has at most MAX_PER_PLOT sizes
        MAX_PER_PLOT = 7
        chunks = [
            list(range(i, min(i + MAX_PER_PLOT, len(labels))))
            for i in range(0, len(labels), MAX_PER_PLOT)
        ]

        for ci, chunk_indices in enumerate(chunks):
            chunk_labels = [labels[i] for i in chunk_indices]
            chunk_sizes = [sizes[i] for i in chunk_indices]

            box_data = []
            tick_labels = []
            for label, sz in zip(chunk_labels, chunk_sizes):
                durations = []
                for rank_durs in all_data[label].values():
                    durations.extend(rank_durs)
                box_data.append(durations)
                tick_labels.append(human_bytes(sz))

            fig, ax = plt.subplots(figsize=(max(8, len(chunk_labels) * 1.2), 5))
            bp = ax.boxplot(
                box_data,
                labels=tick_labels,
                patch_artist=True,
                showfliers=True,
                flierprops=dict(marker="o", markersize=3, alpha=0.4),
            )
            for patch in bp["boxes"]:
                patch.set_facecolor("#4C72B0" if placement == "outofplace" else "#DD8452")
                patch.set_alpha(0.7)

            ax.set_xlabel("Message Size")
            ax.set_ylabel("Kernel Duration (µs)")
            suffix = f" (part {ci+1})" if len(chunks) > 1 else ""
            ax.set_title(f"{bench_name} — {placement}{suffix}")
            ax.grid(axis="y", alpha=0.3)

            if box_data and max(max(d) for d in box_data if d) / (min(min(d) for d in box_data if d) + 1e-9) > 20:
                ax.set_yscale("log")

            fig.tight_layout()
            part = f"_part{ci+1}" if len(chunks) > 1 else ""
            fname = f"{bench_name}_{placement}{part}.png"
            fig.savefig(os.path.join(OUT_DIR, fname), dpi=150)
            plt.close(fig)
            print(f"  wrote {fname}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    benchmarks = sorted(
        d
        for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d)) and d != "plots"
    )

    for bench in benchmarks:
        bench_dir = os.path.join(DATA_DIR, bench)
        print(f"Processing {bench}...")
        plot_benchmark(bench, bench_dir)


if __name__ == "__main__":
    main()
