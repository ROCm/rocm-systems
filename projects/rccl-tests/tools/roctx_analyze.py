#!/usr/bin/env python3
"""Correlate rocTX markers with rocprofv3 kernel traces and report statistics.

Reads the rocprofv3 CSV output from a run directory, correlates kernel
dispatches to rocTX timed_loop markers via timestamp containment, groups
kernel durations by (size, in_place), performs outlier detection, and
prints a summary report.
"""

import argparse
import csv
import glob
import json
import math
import os
import re
import sys
import textwrap
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

COLLECTIVE_KERNEL_RE = re.compile(r"ncclDev")

BUS_BW_FACTOR = {
    "all_reduce":      lambda n: 2 * (n - 1) / n,
    "all_gather":      lambda n: (n - 1) / n,
    "reduce_scatter":  lambda n: (n - 1) / n,
    "alltoall":        lambda n: (n - 1) / n,
    "broadcast":       lambda n: 1,
    "reduce":          lambda n: 1,
    "scatter":         lambda n: 1,
    "gather":          lambda n: 1,
    "sendrecv":        lambda n: 1,
    "hypercube":       lambda n: 2 * (n - 1) / n,
}

MARKER_MSG_RE = re.compile(
    r"rccl-tests timed_loop"
    r" size=(?P<size>\d+)"
    r" count=(?P<count>\d+)"
    r" type=(?P<type>\d+)"
    r" op=(?P<op>\d+)"
    r" in_place=(?P<in_place>\d+)"
    r" proc=(?P<proc>\d+)"
)


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------

def parse_marker_csv(path):
    """Parse a rocprofv3 marker_api_trace CSV. Returns list of dicts."""
    markers = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fn = row["Function"]
            m = MARKER_MSG_RE.search(fn)
            if not m:
                continue
            markers.append({
                "pid": int(row["Process_Id"]),
                "start": int(row["Start_Timestamp"]),
                "end": int(row["End_Timestamp"]),
                "size": int(m.group("size")),
                "in_place": int(m.group("in_place")),
                "proc": int(m.group("proc")),
            })
    return markers


def parse_kernel_csv(path):
    """Parse a rocprofv3 kernel_trace CSV. Returns list of dicts."""
    kernels = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["Kernel_Name"]
            kernels.append({
                "pid": int(row["Thread_Id"]),
                "name": name,
                "start": int(row["Start_Timestamp"]),
                "end": int(row["End_Timestamp"]),
                "is_collective": bool(COLLECTIVE_KERNEL_RE.search(name)),
            })
    return kernels


def discover_trace_files(run_dir):
    """Find all (marker_csv, kernel_csv) pairs in a run directory."""
    pairs = []
    for marker_path in sorted(glob.glob(
        os.path.join(run_dir, "**", "*_marker_api_trace.csv"), recursive=True
    )):
        pid_prefix = os.path.basename(marker_path).split("_marker_api_trace.csv")[0]
        kernel_path = os.path.join(os.path.dirname(marker_path), f"{pid_prefix}_kernel_trace.csv")
        if os.path.isfile(kernel_path):
            pairs.append((marker_path, kernel_path))
    return pairs


def load_run_metadata(run_dir):
    """Load metadata.json and return (np, collective_name) or defaults."""
    meta_path = os.path.join(run_dir, "metadata.json")
    np_val = None
    collective = None
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        np_val = meta.get("np")
        tests = None
        if "matrix" in meta:
            tests = meta["matrix"].get("tests")
        if not tests and "args" in meta:
            tests = meta["args"].get("test")
        if tests and len(tests) == 1:
            collective = tests[0]
    return np_val, collective


def infer_collective(run_dir):
    """Infer collective name from profiler subdirectory names."""
    for entry in os.listdir(run_dir):
        if not os.path.isdir(os.path.join(run_dir, entry)):
            continue
        for name in BUS_BW_FACTOR:
            if entry.startswith(name + "_"):
                return name
    return None


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------

def correlate(markers, kernels):
    """For each marker, find contained collective kernels and return their durations.

    Returns dict: (size, in_place) -> list of kernel durations in ns.
    """
    kernels_sorted = sorted(kernels, key=lambda k: k["start"])
    samples = defaultdict(list)

    for mk in markers:
        key = (mk["size"], mk["in_place"])
        for k in kernels_sorted:
            if k["start"] < mk["start"]:
                continue
            if k["start"] > mk["end"]:
                break
            if k["end"] <= mk["end"] and k["is_collective"]:
                duration_ns = k["end"] - k["start"]
                samples[key].append(duration_ns)

    return samples


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------

def median(vals):
    s = sorted(vals)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def mad_outliers(vals, threshold=3.5):
    """Modified Z-score via MAD. Returns (inliers, outlier_count)."""
    if len(vals) < 3:
        return list(vals), 0
    med = median(vals)
    abs_devs = [abs(v - med) for v in vals]
    mad = median(abs_devs)
    if mad == 0:
        return list(vals), 0
    inliers = []
    outliers = 0
    for v in vals:
        z = 0.6745 * abs(v - med) / mad
        if z <= threshold:
            inliers.append(v)
        else:
            outliers += 1
    return inliers, outliers


def iqr_outliers(vals, factor=1.5):
    """IQR-based outlier detection. Returns (inliers, outlier_count)."""
    if len(vals) < 4:
        return list(vals), 0
    s = sorted(vals)
    n = len(s)
    q1 = s[n // 4]
    q3 = s[3 * n // 4]
    iqr = q3 - q1
    low = q1 - factor * iqr
    high = q3 + factor * iqr
    inliers = [v for v in vals if low <= v <= high]
    outliers = len(vals) - len(inliers)
    return inliers, outliers


OUTLIER_METHODS = {
    "mad": mad_outliers,
    "iqr": iqr_outliers,
}


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def fmt_size(nbytes):
    if nbytes >= 1024 * 1024 * 1024 and nbytes % (1024 * 1024 * 1024) == 0:
        return f"{nbytes // (1024 * 1024 * 1024)}G"
    if nbytes >= 1024 * 1024 and nbytes % (1024 * 1024) == 0:
        return f"{nbytes // (1024 * 1024)}M"
    if nbytes >= 1024 and nbytes % 1024 == 0:
        return f"{nbytes // 1024}K"
    return str(nbytes)


def fmt_ns(ns):
    if ns >= 1_000_000:
        return f"{ns / 1_000_000:.1f}ms"
    if ns >= 1_000:
        return f"{ns / 1_000:.1f}us"
    return f"{ns}ns"


def fmt_bw(gbps):
    """Format bandwidth in GB/s."""
    if gbps is None:
        return "--"
    if gbps >= 1.0:
        return f"{gbps:.2f}"
    if gbps >= 0.001:
        return f"{gbps:.4f}"
    return f"{gbps:.2e}"


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _compute_bw(size_bytes, duration_ns):
    """Return GB/s given message size in bytes and duration in nanoseconds."""
    if duration_ns <= 0:
        return None
    return size_bytes / duration_ns  # bytes/ns == GB/s


def generate_report(all_samples, outlier_fn, np_val=None, bus_factor=None):
    keys = sorted(all_samples.keys(), key=lambda k: (k[1], k[0]))

    rows = []
    for (size, in_place) in keys:
        vals = all_samples[(size, in_place)]
        total = len(vals)
        inliers, n_outliers = outlier_fn(vals)
        if not inliers:
            rows.append({
                "size": size, "in_place": in_place,
                "total": total, "retained": 0, "outliers": n_outliers,
                "min": None, "max": None, "median": None,
                "algbw": None, "busbw": None,
            })
            continue
        med = median(inliers)
        algbw = _compute_bw(size, med)
        busbw = algbw * bus_factor if algbw is not None and bus_factor is not None else None
        rows.append({
            "size": size, "in_place": in_place,
            "total": total, "retained": len(inliers), "outliers": n_outliers,
            "min": min(inliers), "max": max(inliers), "median": med,
            "algbw": algbw, "busbw": busbw,
        })

    return rows


def print_report(rows, method_name, show_bw=False):
    print(f"Outlier method: {method_name}")
    print()

    place_labels = {0: "oop", 1: "ip"}

    hdr = f"{'size':>10}  {'place':>5}  {'kept':>6}  {'out':>4}  {'min':>12}  {'median':>12}  {'max':>12}"
    if show_bw:
        hdr += f"  {'algbw':>10}  {'busbw':>10}"
    sep = "-" * len(hdr)

    current_place = None
    for r in rows:
        if r["in_place"] != current_place:
            if current_place is not None:
                print()
            current_place = r["in_place"]
            label = "out-of-place" if current_place == 0 else "in-place"
            print(f"  [{label}]")
            print(f"  {hdr}")
            print(f"  {sep}")

        line = f"  {fmt_size(r['size']):>10}  {place_labels[r['in_place']]:>5}  {r['retained']:>6}  {r['outliers']:>4}"
        if r["retained"] == 0:
            line += f"  {'--':>12}  {'--':>12}  {'--':>12}"
            if show_bw:
                line += f"  {'--':>10}  {'--':>10}"
        else:
            line += f"  {fmt_ns(r['min']):>12}  {fmt_ns(r['median']):>12}  {fmt_ns(r['max']):>12}"
            if show_bw:
                line += f"  {fmt_bw(r['algbw']):>10}  {fmt_bw(r['busbw']):>10}"
        print(line)

    if show_bw:
        print()
        print("  algbw/busbw in GB/s (computed from median kernel duration)")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Correlate rocTX markers with kernel traces and report statistics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              %(prog)s perf-runs/20260305-230534
              %(prog)s perf-runs/20260305-230534 --outlier iqr
              %(prog)s perf-runs/20260305-230534 --outlier mad --mad-threshold 3.0
        """),
    )
    parser.add_argument(
        "run_dir", type=str,
        help="Run directory containing rocprofv3 CSV output",
    )
    parser.add_argument(
        "--outlier", type=str, default="mad",
        choices=list(OUTLIER_METHODS.keys()),
        help="Outlier detection method (default: mad)",
    )
    parser.add_argument(
        "--mad-threshold", type=float, default=3.5,
        help="MAD modified Z-score threshold (default: 3.5)",
    )
    parser.add_argument(
        "--iqr-factor", type=float, default=1.5,
        help="IQR multiplier (default: 1.5)",
    )

    return parser.parse_args(argv)


def main():
    args = parse_args()

    trace_pairs = discover_trace_files(args.run_dir)
    if not trace_pairs:
        print(f"No marker/kernel trace CSV pairs found in {args.run_dir}", file=sys.stderr)
        sys.exit(1)

    np_val, meta_collective = load_run_metadata(args.run_dir)
    collective = meta_collective or infer_collective(args.run_dir)

    bus_factor = None
    show_bw = False
    if collective and np_val:
        factor_fn = BUS_BW_FACTOR.get(collective)
        if factor_fn:
            bus_factor = factor_fn(np_val)
            show_bw = True
            print(f"Collective: {collective}  np: {np_val}  bus_bw_factor: {bus_factor:.4f}")

    print(f"Found {len(trace_pairs)} trace file pairs across {args.run_dir}")

    all_samples = defaultdict(list)
    total_markers = 0
    total_kernels_matched = 0

    for marker_path, kernel_path in trace_pairs:
        markers = parse_marker_csv(marker_path)
        kernels = parse_kernel_csv(kernel_path)
        samples = correlate(markers, kernels)
        total_markers += len(markers)
        for key, durations in samples.items():
            total_kernels_matched += len(durations)
            all_samples[key].extend(durations)

    print(f"Markers: {total_markers}, collective kernel samples: {total_kernels_matched}")
    print(f"Unique (size, place) groups: {len(all_samples)}")
    print()

    if args.outlier == "mad":
        outlier_fn = lambda vals: mad_outliers(vals, threshold=args.mad_threshold)
        method_name = f"MAD (threshold={args.mad_threshold})"
    else:
        outlier_fn = lambda vals: iqr_outliers(vals, factor=args.iqr_factor)
        method_name = f"IQR (factor={args.iqr_factor})"

    rows = generate_report(all_samples, outlier_fn, np_val=np_val, bus_factor=bus_factor)
    print_report(rows, method_name, show_bw=show_bw)


if __name__ == "__main__":
    main()
