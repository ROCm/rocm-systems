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

# Kernel categorization is used ONLY for filtering/display -- never to drop data
# at parse/correlate time. Every kernel that falls inside a marker region is
# retained; callers decide which categories to show. This keeps otherwise
# invisible costs (e.g. rocclr copy/fill) accountable.
#
# Categories, matched in order:
#   collective : the actual reduction kernels -- RCCL generic (ncclDevKernel_*)
#                and Meta DDA direct-data-access/IPC kernels (meta::comms::dda*).
#   rocclr     : HIP runtime helper kernels (__amd_rocclr_copyBuffer / fill / ...).
#   harness    : rccl-tests data-init/verify kernels (prepareInput/Expected/verify).
#   other      : anything else, so nothing is silently ignored.
COLLECTIVE_KERNEL_RE = re.compile(r"ncclDevKernel|meta::comms::dda")
ROCCLR_KERNEL_RE = re.compile(r"^__amd_rocclr_")
HARNESS_KERNEL_RE = re.compile(r"prepareInput|prepareExpected|verifyPrepared")

# Display order for categories.
CATEGORY_ORDER = ["collective", "rocclr", "harness", "other"]


def categorize_kernel(name):
    """Classify a kernel by name for display purposes (never drops data)."""
    if COLLECTIVE_KERNEL_RE.search(name):
        return "collective"
    if ROCCLR_KERNEL_RE.search(name):
        return "rocclr"
    if HARNESS_KERNEL_RE.search(name):
        return "harness"
    return "other"

BUS_BW_FACTOR = {
    "all_gather":      lambda n: (n - 1) / n,
    "all_reduce_bias": lambda n: 2 * (n - 1) / n,
    "all_reduce":      lambda n: 2 * (n - 1) / n,
    "alltoallv":       lambda n: (n - 1) / n,
    "alltoall":        lambda n: (n - 1) / n,
    "broadcast":       lambda n: 1,
    "gather":          lambda n: 1,
    "hypercube":       lambda n: 2 * (n - 1) / n,
    "reduce":          lambda n: 1,
    "reduce_scatter":  lambda n: (n - 1) / n,
    "scatter":         lambda n: 1,
    "sendrecv":        lambda n: 1,
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
        # np may be at the top level or nested inside matrix/args
        np_val = (
            meta.get("np")
            or meta.get("matrix", {}).get("np")
            or meta.get("args", {}).get("np")
        )
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


# Matches subdirectory names like  all_reduce_bfloat16_rep3_4
_SUBDIR_RE = re.compile(
    r"^(?P<collective>" + "|".join(re.escape(k) for k in sorted(BUS_BW_FACTOR, key=len, reverse=True)) + r")"
    r"_(?P<dtype>.+)_rep(?P<rep>\d+)_(?P<rank>\d+)$"
)


def discover_multi_run_groups(run_dir):
    """Return an ordered dict of (collective, dtype) -> [subdir_path, ...] if run_dir
    looks like a top-level multi-test/multi-dtype run directory, else None.

    A directory qualifies when it contains at least two subdirectories whose names match
    ``{collective}_{dtype}_rep{N}`` (or any number of such dirs spanning more than one
    (collective, dtype) combination -- even a single combination with multiple reps).
    """
    groups = defaultdict(list)
    for entry in sorted(os.listdir(run_dir)):
        m = _SUBDIR_RE.match(entry)
        if m and os.path.isdir(os.path.join(run_dir, entry)):
            key = (m.group("collective"), m.group("dtype"))
            groups[key].append(os.path.join(run_dir, entry))
    return dict(groups) if groups else None


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------

def correlate(markers, kernels):
    """Assign every contained kernel to its marker region -- no filtering.

    For each marker (a timed_loop over one (size, in_place) point) collect ALL
    kernels whose [start, end] falls inside the marker window, keyed by kernel
    name. Categorization/filtering happens later at display time, so nothing is
    discarded here.

    Returns dict: (size, in_place) -> {kernel_name -> [durations_ns, ...]}.
    """
    kernels_sorted = sorted(kernels, key=lambda k: k["start"])
    regions = defaultdict(lambda: defaultdict(list))

    for mk in markers:
        key = (mk["size"], mk["in_place"])
        bucket = regions[key]
        for k in kernels_sorted:
            if k["start"] < mk["start"]:
                continue
            if k["start"] > mk["end"]:
                break
            if k["end"] <= mk["end"]:
                bucket[k["name"]].append(k["end"] - k["start"])

    return regions


def merge_regions(dst, src):
    """Merge a per-file regions dict (from correlate) into an accumulator."""
    for key, byname in src.items():
        d = dst[key]
        for name, durs in byname.items():
            d[name].extend(durs)
    return dst


def collective_samples(regions):
    """Collapse a regions dict to (size, in_place) -> [collective durations].

    This reproduces the historical view used by the default bandwidth report,
    but derived from the full retained data instead of a parse-time filter.
    """
    out = defaultdict(list)
    for key, byname in regions.items():
        for name, durs in byname.items():
            if categorize_kernel(name) == "collective":
                out[key].extend(durs)
    return out


def correlate_collective(markers, kernels):
    """Convenience: per-file (size, in_place) -> [collective durations].

    Equivalent to the historical ``correlate`` return shape, kept so callers that
    only want collective-kernel samples (plot/compare) don't need the full
    regions structure.
    """
    return collective_samples(correlate(markers, kernels))


def category_durations(regions):
    """Collapse a regions dict to (size, in_place) -> {category -> [durations]}."""
    out = defaultdict(lambda: defaultdict(list))
    for key, byname in regions.items():
        for name, durs in byname.items():
            out[key][categorize_kernel(name)].extend(durs)
    return out


def compute_overhead(regions):
    """Per (size, in_place) non-collective GPU time per iteration, in ns.

    For each region, sum the median duration of every *non-collective* category
    (e.g. the DDA tree's `cudaMemcpyAsync` staging copy, which shows up as a
    rocclr copyBuffer). One such helper runs per iteration alongside the
    collective on the same stream, so this approximates the extra per-call cost
    the collective kernel alone doesn't capture.

    Returns dict: (size, in_place) -> {"overhead_ns": int, "detail": {cat: median}}.
    """
    out = {}
    for key, bycat in category_durations(regions).items():
        overhead = 0
        detail = {}
        for cat, durs in bycat.items():
            if cat == "collective" or not durs:
                continue
            m = median(durs)
            detail[cat] = m
            overhead += m
        out[key] = {"overhead_ns": overhead, "detail": detail}
    return out


def collective_timing_samples(subdirs):
    """Per (size, in_place) GPU-timing samples for collective kernels.

    Unlike ``correlate``/``collective_samples`` (which keep only per-kernel
    durations), this preserves the launch *sequence* so we can measure the
    gaps between launches -- the thing rccl-tests' host_time/N cannot see.

    For every trace file, the collective kernels (``ncclDevKernel*`` /
    ``meta::comms::dda*``) inside each marker window are taken in start-time
    order, and we record:
      * ``dur`` = per-launch on-GPU duration (End - Start), in ns. This is the
        rocprofv3 hardware timestamp, accurate even when host/launch time is
        inflated by the profiler.
      * ``gap`` = inter-launch gap (next.Start - this.End), in ns, between
        consecutive collective launches within the *same* file's window. Gaps
        are per-rank/per-stream only and never span files.

    Returns dict: (size, in_place) -> {"dur": [ns, ...], "gap": [ns, ...]}.
    """
    out = defaultdict(lambda: {"dur": [], "gap": []})
    for d in subdirs:
        for marker_path, kernel_path in discover_trace_files(d):
            markers = parse_marker_csv(marker_path)
            kernels = parse_kernel_csv(kernel_path)
            coll = sorted(
                (k for k in kernels if categorize_kernel(k["name"]) == "collective"),
                key=lambda k: k["start"],
            )
            for mk in markers:
                key = (mk["size"], mk["in_place"])
                lo, hi = mk["start"], mk["end"]
                prev_end = None
                for k in coll:
                    if k["start"] < lo:
                        continue
                    if k["start"] > hi:
                        break
                    if k["end"] > hi:
                        continue
                    out[key]["dur"].append(k["end"] - k["start"])
                    if prev_end is not None:
                        out[key]["gap"].append(k["start"] - prev_end)
                    prev_end = k["end"]
    return out


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------

def median(vals):
    s = sorted(vals)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def percentile(vals, q):
    """Linear-interpolated percentile (numpy default method), pure Python.

    *q* in [0, 100]. Returns None for empty input.
    """
    if not vals:
        return None
    s = sorted(vals)
    n = len(s)
    if n == 1:
        return float(s[0])
    rank = (q / 100.0) * (n - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return float(s[lo])
    frac = rank - lo
    return s[lo] * (1.0 - frac) + s[hi] * frac


def summarize(vals):
    """Distribution summary for a sample list. Returns None for empty input.

    Keys: n, min, max, mean, std (population), p10, p25, p50, p75, p90.
    """
    if not vals:
        return None
    s = sorted(vals)
    n = len(s)
    mean = sum(s) / n
    var = sum((v - mean) ** 2 for v in s) / n if n > 1 else 0.0
    return {
        "n": n,
        "min": s[0],
        "max": s[-1],
        "mean": mean,
        "std": var ** 0.5,
        "p10": percentile(s, 10),
        "p25": percentile(s, 25),
        "p50": percentile(s, 50),
        "p75": percentile(s, 75),
        "p90": percentile(s, 90),
    }


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


def generate_report(all_samples, outlier_fn, np_val=None, bus_factor=None, overhead_by_key=None):
    keys = sorted(all_samples.keys(), key=lambda k: (k[1], k[0]))
    overhead_by_key = overhead_by_key or {}

    rows = []
    for (size, in_place) in keys:
        vals = all_samples[(size, in_place)]
        total = len(vals)
        inliers, n_outliers = outlier_fn(vals)
        overhead = overhead_by_key.get((size, in_place), {}).get("overhead_ns", 0)
        if not inliers:
            rows.append({
                "size": size, "in_place": in_place,
                "total": total, "retained": 0, "outliers": n_outliers,
                "min": None, "max": None, "median": None,
                "p10": None, "p25": None, "p75": None, "p90": None,
                "mean": None, "std": None, "samples": [],
                "algbw": None, "busbw": None,
                "overhead": overhead, "eff_median": None,
                "eff_algbw": None, "eff_busbw": None,
            })
            continue
        st = summarize(inliers)
        med = st["p50"]
        algbw = _compute_bw(size, med)
        busbw = algbw * bus_factor if algbw is not None and bus_factor is not None else None
        # Effective = collective + per-iteration non-collective overhead (e.g. DDA
        # staging copy), giving the true end-to-end per-call cost/bandwidth.
        eff_med = med + overhead
        eff_algbw = _compute_bw(size, eff_med)
        eff_busbw = eff_algbw * bus_factor if eff_algbw is not None and bus_factor is not None else None
        rows.append({
            "size": size, "in_place": in_place,
            "total": total, "retained": len(inliers), "outliers": n_outliers,
            "min": st["min"], "max": st["max"], "median": med,
            "p10": st["p10"], "p25": st["p25"], "p75": st["p75"], "p90": st["p90"],
            "mean": st["mean"], "std": st["std"], "samples": list(inliers),
            "algbw": algbw, "busbw": busbw,
            "overhead": overhead, "eff_median": eff_med,
            "eff_algbw": eff_algbw, "eff_busbw": eff_busbw,
        })

    return rows


def print_report(rows, method_name, show_bw=False):
    print(f"Outlier method: {method_name}")
    print()

    place_labels = {0: "oop", 1: "ip"}

    # Only surface the effective columns if some region actually has overhead.
    show_eff = show_bw and any(r.get("overhead") for r in rows)

    hdr = f"{'size':>10}  {'place':>5}  {'kept':>6}  {'out':>4}  {'min':>12}  {'median':>12}  {'max':>12}"
    if show_bw:
        hdr += f"  {'algbw':>10}  {'busbw':>10}"
    if show_eff:
        hdr += f"  {'ovhd':>10}  {'eff_busbw':>10}"
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
            if show_eff:
                line += f"  {'--':>10}  {'--':>10}"
        else:
            line += f"  {fmt_ns(r['min']):>12}  {fmt_ns(r['median']):>12}  {fmt_ns(r['max']):>12}"
            if show_bw:
                line += f"  {fmt_bw(r['algbw']):>10}  {fmt_bw(r['busbw']):>10}"
            if show_eff:
                ovhd = r.get("overhead") or 0
                ovhd_str = fmt_ns(ovhd) if ovhd else "--"
                line += f"  {ovhd_str:>10}  {fmt_bw(r['eff_busbw']):>10}"
        print(line)

    if show_bw:
        print()
        print("  algbw/busbw in GB/s (computed from median collective kernel duration)")
    if show_eff:
        print("  ovhd = per-call non-collective GPU time (e.g. DDA staging copy); "
              "eff_busbw = busbw incl. that overhead")
    print()


def print_breakdown(regions):
    """Per-(size, place) accounting of ALL kernels by category.

    Shows, for every marker region, how much GPU time each category consumed --
    including rocclr copy/fill and any 'other' kernels that the collective-only
    view hides. `share` is the category's fraction of the region's total kernel
    time (sum of durations), so expensive-but-ignored kernels become visible.
    """
    cat_durs = category_durations(regions)
    keys = sorted(cat_durs.keys(), key=lambda k: (k[1], k[0]))
    place_labels = {0: "oop", 1: "ip"}

    hdr = (f"{'size':>10}  {'place':>5}  {'category':>10}  {'count':>7}  "
           f"{'total':>12}  {'median':>12}  {'share':>7}")
    sep = "-" * len(hdr)

    current_place = None
    for key in keys:
        size, in_place = key
        if in_place != current_place:
            if current_place is not None:
                print()
            current_place = in_place
            label = "out-of-place" if current_place == 0 else "in-place"
            print(f"  [{label}]")
            print(f"  {hdr}")
            print(f"  {sep}")

        bycat = cat_durs[key]
        region_total = sum(sum(v) for v in bycat.values()) or 1
        first = True
        for cat in CATEGORY_ORDER:
            durs = bycat.get(cat)
            if not durs:
                continue
            total = sum(durs)
            share = 100.0 * total / region_total
            size_col = fmt_size(size) if first else ""
            place_col = place_labels[in_place] if first else ""
            first = False
            print(f"  {size_col:>10}  {place_col:>5}  {cat:>10}  {len(durs):>7}  "
                  f"{fmt_ns(total):>12}  {fmt_ns(median(durs)):>12}  {share:>6.1f}%")

    print()
    print("  total = summed GPU time in category within the region; "
          "share = % of region kernel time")
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
    parser.add_argument(
        "--breakdown", action="store_true",
        help="Also print a per-(size,place) breakdown of ALL kernels by category "
             "(collective/rocclr/harness/other), so hidden costs are visible.",
    )

    return parser.parse_args(argv)


def _build_outlier_fn(args):
    if args.outlier == "mad":
        return (
            lambda vals: mad_outliers(vals, threshold=args.mad_threshold),
            f"MAD (threshold={args.mad_threshold})",
        )
    return (
        lambda vals: iqr_outliers(vals, factor=args.iqr_factor),
        f"IQR (factor={args.iqr_factor})",
    )


def _collect_regions(pairs_iter):
    """Correlate every (marker, kernel) pair and merge into one regions dict.

    Returns (regions, total_pairs, total_markers, total_kernels_retained).
    """
    regions = defaultdict(lambda: defaultdict(list))
    total_pairs = total_markers = total_kernels = 0
    for marker_path, kernel_path in pairs_iter:
        total_pairs += 1
        markers = parse_marker_csv(marker_path)
        kernels = parse_kernel_csv(kernel_path)
        total_markers += len(markers)
        r = correlate(markers, kernels)
        for byname in r.values():
            for durs in byname.values():
                total_kernels += len(durs)
        merge_regions(regions, r)
    return regions, total_pairs, total_markers, total_kernels


def _analyze_dirs(dirs, np_val, collective, outlier_fn, method_name):
    """Collect trace pairs from *dirs*, correlate, and return (rows, show_bw, regions)."""
    bus_factor = None
    show_bw = False
    if collective and np_val:
        factor_fn = BUS_BW_FACTOR.get(collective)
        if factor_fn:
            bus_factor = factor_fn(np_val)
            show_bw = True

    def _pairs():
        for d in dirs:
            for p in discover_trace_files(d):
                yield p

    regions, total_pairs, total_markers, total_kernels = _collect_regions(_pairs())
    all_samples = collective_samples(regions)
    n_collective = sum(len(v) for v in all_samples.values())

    print(f"  Trace pairs: {total_pairs}  markers: {total_markers}  "
          f"kernels retained: {total_kernels}  collective samples: {n_collective}  "
          f"(size,place) groups: {len(all_samples)}")
    if show_bw:
        print(f"  bus_bw_factor: {bus_factor:.4f}  (np={np_val})")

    overhead_by_key = compute_overhead(regions)
    rows = generate_report(all_samples, outlier_fn, np_val=np_val, bus_factor=bus_factor,
                           overhead_by_key=overhead_by_key)
    return rows, show_bw, regions


def main():
    args = parse_args()
    outlier_fn, method_name = _build_outlier_fn(args)

    multi_groups = discover_multi_run_groups(args.run_dir)

    if multi_groups:
        np_val, _ = load_run_metadata(args.run_dir)
        print(f"Multi-run directory detected: {args.run_dir}")
        print(f"Groups: {', '.join(f'{c}/{d}' for c, d in multi_groups)}")
        print()

        for (collective, dtype), subdirs in multi_groups.items():
            print(f"{'=' * 60}")
            print(f"  {collective}  /  {dtype}  ({len(subdirs)} rep(s))")
            print(f"{'=' * 60}")
            rows, show_bw, regions = _analyze_dirs(subdirs, np_val, collective, outlier_fn, method_name)
            print_report(rows, method_name, show_bw=show_bw)
            if args.breakdown:
                print_breakdown(regions)
        return

    # Single-run directory (original behavior).
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

    regions, _, total_markers, total_kernels = _collect_regions(iter(trace_pairs))
    all_samples = collective_samples(regions)
    n_collective = sum(len(v) for v in all_samples.values())

    print(f"Markers: {total_markers}, kernels retained: {total_kernels}, "
          f"collective samples: {n_collective}")
    print(f"Unique (size, place) groups: {len(all_samples)}")
    print()

    overhead_by_key = compute_overhead(regions)
    rows = generate_report(all_samples, outlier_fn, np_val=np_val, bus_factor=bus_factor,
                           overhead_by_key=overhead_by_key)
    print_report(rows, method_name, show_bw=show_bw)
    if args.breakdown:
        print_breakdown(regions)


if __name__ == "__main__":
    main()
