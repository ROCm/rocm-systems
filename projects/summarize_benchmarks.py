#!/usr/bin/env python3
"""
Print min, max, and mean kernel durations per (benchmark, size, algo/proto),
with standard IQR outlier trimming.

Usage:
    python summarize_benchmarks.py [RUN_DIR ...]

With no arguments, uses the most recent timestamped directory under DATA_DIR.
"""

import sqlite3
import json
import os
import sys
import re
import glob
import numpy as np
from collections import defaultdict

DATA_DIR = "/work/lmeadows/data"
SKIP_BENCHMARKS = {"all_reduce_bias", "plots"}

NRANKS = 8
SCALED_FUNCS = {"ReduceScatter", "AllGather", "Broadcast", "Reduce"}


# ── run-directory helpers ────────────────────────────────────────────────


def find_run_dirs(data_dir):
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


def discover_benchmarks(run_dirs):
    benchmarks = set()
    for run_dir in run_dirs:
        if not os.path.isdir(run_dir):
            continue
        for name in os.listdir(run_dir):
            path = os.path.join(run_dir, name)
            if os.path.isdir(path) and name not in SKIP_BENCHMARKS:
                benchmarks.add(name)
    return sorted(benchmarks)


# ── data extraction ──────────────────────────────────────────────────────


def find_results_db(rank_dir):
    dbs = glob.glob(os.path.join(rank_dir, "**", "*_results.db"), recursive=True)
    return dbs[0] if dbs else None


def extract_rank_data(db_path):
    """Return list of (label, kernel_duration_µs)."""
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
        SELECT K.start, (K.end - K.start) AS duration
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
            results.append((regions[ri][2], k_dur / 1e3))
    return results


def collect_benchmark_data(bench_dir):
    """Return {(size_bytes, placement): [dur_µs, …]}."""
    rank_dirs = sorted(glob.glob(os.path.join(bench_dir, "rank_*")))
    if not rank_dirs:
        return None
    data = defaultdict(list)
    for rank_dir in rank_dirs:
        db_path = find_results_db(rank_dir)
        if not db_path:
            continue
        for label, dur_us in extract_rank_data(db_path):
            parts = label.split(":")
            if len(parts) != 3:
                continue
            size = int(parts[1])
            placement = parts[2]
            data[(size, placement)].append(dur_us)
    return data if data else None


def extract_protocol_map(bench_dir):
    """Return {size_bytes: {"proto": str, "algo": str}}."""
    rank_dirs = sorted(glob.glob(os.path.join(bench_dir, "rank_*")))
    proto_map = {}
    for rank_dir in rank_dirs[:1]:
        db_path = find_results_db(rank_dir)
        if not db_path:
            continue
        conn = sqlite3.connect(db_path)
        rows = conn.execute("""
            SELECT R.extdata
            FROM regions R
            WHERE R.category = 'MARKER_CORE_RANGE_API'
        """).fetchall()
        conn.close()
        for (extdata,) in rows:
            try:
                msg = json.loads(extdata).get("message", "")
            except (json.JSONDecodeError, TypeError):
                continue
            if not msg.startswith("RCCL-PROTO:"):
                continue
            parts = msg.split(":")
            if len(parts) < 6:
                continue
            _, func, nbytes_str, algo, proto, _ = parts[:6]
            nbytes = int(nbytes_str)
            chart_size = nbytes // NRANKS if func in SCALED_FUNCS else nbytes
            if chart_size not in proto_map:
                proto_map[chart_size] = {"proto": proto, "algo": algo}
        if proto_map:
            break
    return proto_map


# ── MAD outlier trimming ─────────────────────────────────────────────────

MAD_THRESHOLD = 3.5
MAD_SCALE = 0.6745  # normalises MAD to match σ for Gaussian data


def human_bytes(n):
    for unit in ("B", "K", "M", "G"):
        if n < 1024:
            return f"{int(n)}{unit}" if float(n) == int(n) else f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{int(n)}T" if float(n) == int(n) else f"{n:.1f}T"


def mad_trim(values):
    """Remove outliers using the modified Z-score (Iglewicz & Hoaglin).

    Modified Z-score = 0.6745 · (x − median) / MAD.
    Points with |score| > 3.5 are considered outliers.
    Falls back to keeping all values when MAD is zero (constant data).
    """
    arr = np.asarray(values)
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    if mad == 0:
        return arr
    scores = MAD_SCALE * np.abs(arr - med) / mad
    return arr[scores <= MAD_THRESHOLD]


# ── per-run stats collection ─────────────────────────────────────────────


def collect_run_stats(run_dir):
    """Return {bench: {(size, placement): {"min": …, "mean": …, "max": …, "n": …, "trimmed": …, "algo": …}}}."""
    benchmarks = discover_benchmarks([run_dir])
    result = {}
    for bench in benchmarks:
        bench_dir = os.path.join(run_dir, bench)
        if not os.path.isdir(bench_dir):
            continue
        data = collect_benchmark_data(bench_dir)
        if not data:
            continue
        proto_map = extract_protocol_map(bench_dir)
        stats = {}
        for (size, placement), durs in data.items():
            arr = np.asarray(durs)
            trimmed = mad_trim(arr)
            if len(trimmed) == 0:
                continue
            pm = proto_map.get(size)
            stats[(size, placement)] = {
                "min": float(np.min(trimmed)),
                "mean": float(np.mean(trimmed)),
                "max": float(np.max(trimmed)),
                "n": len(arr),
                "trimmed": len(arr) - len(trimmed),
                "algo": f"{pm['algo']}/{pm['proto']}" if pm else "—",
            }
        if stats:
            result[bench] = stats
    return result


# ── main ─────────────────────────────────────────────────────────────────


def fmt_val(v):
    return f"{v:>10.2f}" if v is not None else f"{'—':>10}"


def print_table(run_dirs):
    labels = [os.path.basename(d) for d in run_dirs]
    all_run_stats = [(lbl, collect_run_stats(d)) for lbl, d in zip(labels, run_dirs)]
    n = len(labels)

    all_benchmarks = sorted(set(
        bench for _, stats in all_run_stats for bench in stats
    ))

    # Column order: all oop columns first, then all ip columns.
    # Labels use [1], [2], … shorthand.
    col_hdrs = []
    for placement_tag in ("oop", "ip"):
        for i in range(n):
            col_hdrs.append(f"[{i+1}]-{placement_tag}")
    col_w = 10

    hdr_line = "".join(f" {h:>{col_w}}" for h in col_hdrs)
    row_prefix_w = 8 + 1 + 14  # size + space + algo
    sep = "─" * (row_prefix_w + len(col_hdrs) * (col_w + 1))

    print(f"  Values: trimmed min (µs)")
    print()

    for bench in all_benchmarks:
        all_keys = set()
        for _, run_stats in all_run_stats:
            if bench in run_stats:
                all_keys.update(run_stats[bench].keys())

        all_sizes = sorted(set(sz for sz, _ in all_keys))
        if not all_sizes:
            continue

        print(f"  {bench}")
        print(f"  {sep}")
        print(f"  {'Size':>8} {'Algo/Proto':<14}{hdr_line}")
        print(f"  {sep}")

        for size in all_sizes:
            algo_str = "—"
            cols = []
            for placement in ("outofplace", "inplace"):
                for _, run_stats in all_run_stats:
                    s = run_stats.get(bench, {}).get((size, placement))
                    if s:
                        cols.append(s["min"])
                        if s["algo"] != "—":
                            algo_str = s["algo"]
                    else:
                        cols.append(None)

            vals = "".join(f" {fmt_val(v):>{col_w}}" for v in cols)
            print(f"  {human_bytes(size):>8} {algo_str:<14}{vals}")

        print(f"  {sep}")
        print()

    # Legend
    print(f"  Legend:")
    for i, (lbl, d) in enumerate(zip(labels, run_dirs)):
        print(f"    [{i+1}] {lbl}  ({d})")
    print()


def main():
    run_dirs = resolve_run_dirs(sys.argv[1:])
    print_table(run_dirs)


if __name__ == "__main__":
    main()
