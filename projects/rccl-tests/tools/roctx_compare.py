#!/usr/bin/env python3
"""Compare two rccl-tests perf-run directories side-by-side.

Reads baseline CSVs (default) or profiled rocTX kernel traces from two run
directories, joins on (size, in_place), and prints a comparison table with
bandwidth deltas.  Optionally writes the comparison to CSV.

Examples:
  %(prog)s perf-runs/20260319-201332 perf-runs/20260319-201434
  %(prog)s perf-runs/A perf-runs/B --source profiled --csv cmp.csv
  %(prog)s perf-runs/A perf-runs/B --source log
  %(prog)s perf-runs/A perf-runs/B --label-a "build-A" --label-b "build-B"
"""

import argparse
import csv
import json
import os
import re
import sys
import textwrap
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

import roctx_analyze as ra

_BASELINE_FILE_RE = re.compile(
    r"^(?P<collective>.+?)_(?P<dtype>[a-zA-Z0-9]+)_baseline_rep(?P<rep>\d+)\.csv$"
)
_LOG_FILE_RE = re.compile(
    r"^(?P<collective>.+?)_(?P<dtype>[a-zA-Z0-9]+)_(?:baseline_)?rep(?P<rep>\d+)\.log$"
)
_LOG_DATA_RE = re.compile(
    r"^\s*(?P<size>\d+)\s+\d+\s+\S+\s+\S+\s+[-\d]+\s+"
    r"(?P<time_oop>[\d.]+)\s+(?P<algbw_oop>[\d.]+)\s+(?P<busbw_oop>[\d.]+)\s+\d+\s+"
    r"(?P<time_ip>[\d.]+)\s+(?P<algbw_ip>[\d.]+)\s+(?P<busbw_ip>[\d.]+)\s+\d+"
)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def _run_label(run_dir, override=None):
    """Return a short human label for a run directory."""
    if override:
        return override
    meta_path = os.path.join(run_dir, "metadata.json")
    if os.path.isfile(meta_path):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            git_hash = (meta.get("librccl") or {}).get("git_hash")
            if git_hash:
                return git_hash
        except Exception:
            pass
    return os.path.basename(os.path.normpath(run_dir))


# ---------------------------------------------------------------------------
# Baseline loading
# ---------------------------------------------------------------------------

def _discover_baseline_files(run_dir):
    """Return dict: (collective, dtype) -> [csv_path, ...]."""
    groups = defaultdict(list)
    for entry in sorted(os.listdir(run_dir)):
        m = _BASELINE_FILE_RE.match(entry)
        if m:
            path = os.path.join(run_dir, entry)
            key = (m.group("collective"), m.group("dtype"))
            groups[key].append(path)
    return dict(groups)


_BASELINE_MPI_FIELDS = [
    "numCycle", "collective", "nodes", "ranks", "rankspernode", "gpusperrank",
    "size", "type", "redop", "inplace", "time", "algbw", "busbw", "wrong",
]

_BASELINE_NONMPI_FIELDS = [
    "numCycle", "collective", "gpus",
    "size", "type", "redop", "inplace", "time", "algbw", "busbw", "wrong",
]


def _parse_baseline_csv(path):
    """Parse one baseline CSV.  Returns list of row dicts with numeric fields.

    Handles both the fixed header (with ``nodes``) and the old header that was
    missing it (data had one more column than the header).
    """
    rows = []
    with open(path, newline="") as f:
        header_line = f.readline()
        header_cols = next(csv.reader([header_line]))
        first_data = f.readline()
        if not first_data.strip():
            return rows
        data_cols = next(csv.reader([first_data]))

        f.seek(0)
        if len(data_cols) > len(header_cols):
            next(f)  # skip broken header
            reader = csv.DictReader(f, fieldnames=_BASELINE_MPI_FIELDS)
        else:
            reader = csv.DictReader(f)  # header is correct, use as-is

        for row in reader:
            try:
                rows.append({
                    "size": int(row["size"]),
                    "inplace": int(row["inplace"]),
                    "time_us": float(row["time"]),
                    "algbw": float(row["algbw"]),
                    "busbw": float(row["busbw"]),
                })
            except (ValueError, KeyError):
                continue
    return rows


def load_baseline_data(run_dir, outlier_fn):
    """Load all baseline CSVs from *run_dir*.

    Returns dict: (collective, dtype) -> list of report-style row dicts
    (same schema as ``roctx_analyze.generate_report`` output, keyed on
    size/in_place with median/algbw/busbw computed from baseline timings).
    """
    file_groups = _discover_baseline_files(run_dir)
    np_val, _ = ra.load_run_metadata(run_dir)
    result = {}

    for (collective, dtype), csv_paths in file_groups.items():
        time_samples = defaultdict(list)
        algbw_samples = defaultdict(list)
        busbw_samples = defaultdict(list)

        for path in csv_paths:
            for row in _parse_baseline_csv(path):
                key = (row["size"], row["inplace"])
                time_samples[key].append(row["time_us"])
                algbw_samples[key].append(row["algbw"])
                busbw_samples[key].append(row["busbw"])

        rows = []
        for key in sorted(time_samples, key=lambda k: (k[1], k[0])):
            size, in_place = key
            times = time_samples[key]
            total = len(times)
            inlier_times, n_outliers = outlier_fn(times)
            if not inlier_times:
                rows.append({
                    "size": size, "in_place": in_place,
                    "total": total, "retained": 0, "outliers": n_outliers,
                    "min_us": None, "max_us": None, "median_us": None,
                    "p10_us": None, "p25_us": None, "p75_us": None, "p90_us": None,
                    "mean_us": None, "std_us": None, "samples_us": [],
                    "algbw": None, "busbw": None,
                })
                continue

            st = ra.summarize(inlier_times)
            inlier_algbw, _ = outlier_fn(algbw_samples[key])
            inlier_busbw, _ = outlier_fn(busbw_samples[key])
            med_algbw = ra.median(inlier_algbw) if inlier_algbw else None
            med_busbw = ra.median(inlier_busbw) if inlier_busbw else None

            rows.append({
                "size": size, "in_place": in_place,
                "total": total, "retained": len(inlier_times),
                "outliers": n_outliers,
                "min_us": st["min"], "max_us": st["max"],
                "median_us": st["p50"],
                "p10_us": st["p10"], "p25_us": st["p25"],
                "p75_us": st["p75"], "p90_us": st["p90"],
                "mean_us": st["mean"], "std_us": st["std"],
                "samples_us": list(inlier_times),
                "algbw": med_algbw, "busbw": med_busbw,
            })

        result[(collective, dtype)] = rows

    return result, np_val


# ---------------------------------------------------------------------------
# Log-file loading (parses rccl-tests stdout table format)
# ---------------------------------------------------------------------------

def _discover_log_files(run_dir, prefer_baseline=True):
    """Return dict: (collective, dtype) -> [log_path, ...].

    When *prefer_baseline* is True and ``*_baseline_rep*.log`` files exist for
    a (collective, dtype) group, only those are returned (they have no profiler
    overhead).  Otherwise the profiled ``*_rep*.log`` files are used.
    """
    baseline_groups = defaultdict(list)
    profiled_groups = defaultdict(list)
    for entry in sorted(os.listdir(run_dir)):
        m = _LOG_FILE_RE.match(entry)
        if not m:
            continue
        key = (m.group("collective"), m.group("dtype"))
        path = os.path.join(run_dir, entry)
        if "baseline" in entry:
            baseline_groups[key].append(path)
        else:
            profiled_groups[key].append(path)

    if prefer_baseline and baseline_groups:
        return dict(baseline_groups)
    return dict(profiled_groups) if profiled_groups else dict(baseline_groups)


def _parse_log_file(path):
    """Parse one rccl-tests stdout log.  Returns list of row dicts.

    Each data line contains both out-of-place and in-place results.
    """
    rows = []
    with open(path) as f:
        for line in f:
            m = _LOG_DATA_RE.match(line)
            if not m:
                continue
            size = int(m.group("size"))
            rows.append({
                "size": size, "inplace": 0,
                "time_us": float(m.group("time_oop")),
                "algbw": float(m.group("algbw_oop")),
                "busbw": float(m.group("busbw_oop")),
            })
            rows.append({
                "size": size, "inplace": 1,
                "time_us": float(m.group("time_ip")),
                "algbw": float(m.group("algbw_ip")),
                "busbw": float(m.group("busbw_ip")),
            })
    return rows


def load_log_data(run_dir, outlier_fn):
    """Load performance data by parsing rccl-tests stdout log files.

    Same return schema as ``load_baseline_data``.
    """
    file_groups = _discover_log_files(run_dir)
    np_val, _ = ra.load_run_metadata(run_dir)
    result = {}

    for (collective, dtype), log_paths in file_groups.items():
        time_samples = defaultdict(list)
        algbw_samples = defaultdict(list)
        busbw_samples = defaultdict(list)

        for path in log_paths:
            for row in _parse_log_file(path):
                key = (row["size"], row["inplace"])
                time_samples[key].append(row["time_us"])
                algbw_samples[key].append(row["algbw"])
                busbw_samples[key].append(row["busbw"])

        rows = []
        for key in sorted(time_samples, key=lambda k: (k[1], k[0])):
            size, in_place = key
            times = time_samples[key]
            total = len(times)
            inlier_times, n_outliers = outlier_fn(times)
            if not inlier_times:
                rows.append({
                    "size": size, "in_place": in_place,
                    "total": total, "retained": 0, "outliers": n_outliers,
                    "min_us": None, "max_us": None, "median_us": None,
                    "p10_us": None, "p25_us": None, "p75_us": None, "p90_us": None,
                    "mean_us": None, "std_us": None, "samples_us": [],
                    "algbw": None, "busbw": None,
                })
                continue

            st = ra.summarize(inlier_times)
            inlier_algbw, _ = outlier_fn(algbw_samples[key])
            inlier_busbw, _ = outlier_fn(busbw_samples[key])
            med_algbw = ra.median(inlier_algbw) if inlier_algbw else None
            med_busbw = ra.median(inlier_busbw) if inlier_busbw else None

            rows.append({
                "size": size, "in_place": in_place,
                "total": total, "retained": len(inlier_times),
                "outliers": n_outliers,
                "min_us": st["min"], "max_us": st["max"],
                "median_us": st["p50"],
                "p10_us": st["p10"], "p25_us": st["p25"],
                "p75_us": st["p75"], "p90_us": st["p90"],
                "mean_us": st["mean"], "std_us": st["std"],
                "samples_us": list(inlier_times),
                "algbw": med_algbw, "busbw": med_busbw,
            })

        result[(collective, dtype)] = rows

    return result, np_val


# ---------------------------------------------------------------------------
# Profiled loading
# ---------------------------------------------------------------------------

def load_profiled_data(run_dir, outlier_fn):
    """Load profiled rocTX trace data.

    Returns dict: (collective, dtype) -> list of report-style row dicts.
    Row dicts contain ``median`` (ns), ``algbw``, ``busbw``, plus a derived
    ``median_us`` field for uniform comparison.
    """
    np_val, _ = ra.load_run_metadata(run_dir)
    multi_groups = ra.discover_multi_run_groups(run_dir)
    result = {}

    if multi_groups:
        for (collective, dtype), subdirs in multi_groups.items():
            bus_factor = None
            factor_fn = ra.BUS_BW_FACTOR.get(collective)
            if factor_fn and np_val:
                bus_factor = factor_fn(np_val)

            all_samples = defaultdict(list)
            for d in subdirs:
                pairs = ra.discover_trace_files(d)
                for marker_path, kernel_path in pairs:
                    markers = ra.parse_marker_csv(marker_path)
                    kernels = ra.parse_kernel_csv(kernel_path)
                    samples = ra.correlate_collective(markers, kernels)
                    for key, durations in samples.items():
                        all_samples[key].extend(durations)

            rows = ra.generate_report(all_samples, outlier_fn,
                                      np_val=np_val, bus_factor=bus_factor)
            for r in rows:
                if r.get("median") is not None:
                    r["median_us"] = r["median"] / 1000.0
                else:
                    r["median_us"] = None
            result[(collective, dtype)] = rows
    else:
        np_val_m, collective = ra.load_run_metadata(run_dir)
        if np_val_m:
            np_val = np_val_m
        pairs = ra.discover_trace_files(run_dir)
        if not pairs:
            return result, np_val

        bus_factor = None
        if collective and np_val:
            factor_fn = ra.BUS_BW_FACTOR.get(collective)
            if factor_fn:
                bus_factor = factor_fn(np_val)

        all_samples = defaultdict(list)
        for marker_path, kernel_path in pairs:
            markers = ra.parse_marker_csv(marker_path)
            kernels = ra.parse_kernel_csv(kernel_path)
            samples = ra.correlate_collective(markers, kernels)
            for key, durations in samples.items():
                all_samples[key].extend(durations)

        rows = ra.generate_report(all_samples, outlier_fn,
                                  np_val=np_val, bus_factor=bus_factor)
        for r in rows:
            if r.get("median") is not None:
                r["median_us"] = r["median"] / 1000.0
            else:
                r["median_us"] = None
        dtype = "unknown"
        result[(collective or "unknown", dtype)] = rows

    return result, np_val


# ---------------------------------------------------------------------------
# Joining
# ---------------------------------------------------------------------------

def join_reports(rows_a, rows_b):
    """Join two row lists on (size, in_place).  Returns list of merged dicts."""
    index_b = {(r["size"], r["in_place"]): r for r in rows_b}

    all_keys = []
    seen = set()
    for r in rows_a:
        k = (r["size"], r["in_place"])
        all_keys.append(k)
        seen.add(k)
    for r in rows_b:
        k = (r["size"], r["in_place"])
        if k not in seen:
            all_keys.append(k)
            seen.add(k)

    all_keys.sort(key=lambda k: (k[1], k[0]))

    index_a = {(r["size"], r["in_place"]): r for r in rows_a}
    merged = []
    for key in all_keys:
        ra_row = index_a.get(key)
        rb_row = index_b.get(key)
        size, in_place = key

        med_a = (ra_row or {}).get("median_us")
        med_b = (rb_row or {}).get("median_us")
        algbw_a = (ra_row or {}).get("algbw")
        algbw_b = (rb_row or {}).get("algbw")
        busbw_a = (ra_row or {}).get("busbw")
        busbw_b = (rb_row or {}).get("busbw")

        delta_pct = None
        if med_a and med_b and med_a > 0:
            delta_pct = (med_a - med_b) / med_a * 100.0

        merged.append({
            "size": size,
            "in_place": in_place,
            "median_us_a": med_a,
            "median_us_b": med_b,
            "algbw_a": algbw_a,
            "algbw_b": algbw_b,
            "busbw_a": busbw_a,
            "busbw_b": busbw_b,
            "delta_pct": delta_pct,
        })

    return merged


# ---------------------------------------------------------------------------
# Terminal output
# ---------------------------------------------------------------------------

def _fmt_us(us):
    if us is None:
        return "--"
    if us >= 1000.0:
        return f"{us / 1000.0:.1f}ms"
    return f"{us:.1f}us"


def _fmt_bw(gbps):
    if gbps is None:
        return "--"
    if gbps >= 1.0:
        return f"{gbps:.2f}"
    if gbps >= 0.001:
        return f"{gbps:.4f}"
    return f"{gbps:.2e}"


def _fmt_delta(pct):
    if pct is None:
        return "--"
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}%"


def print_comparison(merged, collective, dtype, label_a, label_b, np_val=None):
    hdr_line = f"=== {collective} / {dtype}  (A: {label_a}  vs  B: {label_b})"
    if np_val:
        hdr_line += f"  np={np_val}"
    hdr_line += " ==="
    print(hdr_line)
    print()

    place_labels = {0: "oop", 1: "ip"}
    col = (
        f"{'size':>10}  {'place':>5}  "
        f"{'med_A':>10}  {'med_B':>10}  "
        f"{'algbw_A':>9}  {'algbw_B':>9}  "
        f"{'busbw_A':>9}  {'busbw_B':>9}  "
        f"{'delta%':>8}"
    )
    sep = "-" * len(col)

    current_place = None
    for r in merged:
        if r["in_place"] != current_place:
            if current_place is not None:
                print()
            current_place = r["in_place"]
            label = "out-of-place" if current_place == 0 else "in-place"
            print(f"  [{label}]")
            print(f"  {col}")
            print(f"  {sep}")

        line = (
            f"  {ra.fmt_size(r['size']):>10}  "
            f"{place_labels.get(r['in_place'], '?'):>5}  "
            f"{_fmt_us(r['median_us_a']):>10}  "
            f"{_fmt_us(r['median_us_b']):>10}  "
            f"{_fmt_bw(r['algbw_a']):>9}  "
            f"{_fmt_bw(r['algbw_b']):>9}  "
            f"{_fmt_bw(r['busbw_a']):>9}  "
            f"{_fmt_bw(r['busbw_b']):>9}  "
            f"{_fmt_delta(r['delta_pct']):>8}"
        )
        print(line)

    print()
    print("  algbw/busbw in GB/s; delta% = (time_A - time_B) / time_A  (positive = B faster)")
    print()


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def write_comparison_csv(path, all_merged, label_a, label_b):
    """Write all comparison data to a CSV file.

    *all_merged* is a list of (collective, dtype, merged_rows) tuples.
    """
    fieldnames = [
        "collective", "dtype", "size", "in_place",
        f"median_us_{label_a}", f"median_us_{label_b}",
        f"algbw_{label_a}", f"algbw_{label_b}",
        f"busbw_{label_a}", f"busbw_{label_b}",
        "delta_pct",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for collective, dtype, merged in all_merged:
            for r in merged:
                writer.writerow({
                    "collective": collective,
                    "dtype": dtype,
                    "size": r["size"],
                    "in_place": r["in_place"],
                    f"median_us_{label_a}": r["median_us_a"],
                    f"median_us_{label_b}": r["median_us_b"],
                    f"algbw_{label_a}": r["algbw_a"],
                    f"algbw_{label_b}": r["algbw_b"],
                    f"busbw_{label_a}": r["busbw_a"],
                    f"busbw_{label_b}": r["busbw_b"],
                    "delta_pct": f"{r['delta_pct']:.2f}" if r["delta_pct"] is not None else "",
                })
    print(f"Wrote comparison CSV: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _has_baseline_csvs(run_dir):
    """True if *run_dir* contains at least one ``*_baseline_rep*.csv``."""
    for entry in os.listdir(run_dir):
        if _BASELINE_FILE_RE.match(entry):
            return True
    return False


def _has_profiled_traces(run_dir):
    """True if *run_dir* contains profiled trace sub-directories."""
    return bool(ra.discover_multi_run_groups(run_dir))


def _has_log_files(run_dir):
    """True if *run_dir* contains at least one ``*_rep*.log``."""
    for entry in os.listdir(run_dir):
        if _LOG_FILE_RE.match(entry):
            return True
    return False


def _resolve_auto_source(dir_a, dir_b):
    """Pick the best common data source for two run directories.

    Returns (source_name, note_string).

    Priority:
      1. baseline CSVs if both dirs have them (cleanest host-side timing)
      2. profiled traces if both dirs have them (GPU kernel timing, comparable
         even when one dir was profiled and the other wasn't)
      3. log files as last resort (host-side timing from rccl-tests stdout;
         beware that profiled-run logs include profiler overhead)
    """
    a_base = _has_baseline_csvs(dir_a)
    b_base = _has_baseline_csvs(dir_b)
    if a_base and b_base:
        return "baseline", "both dirs have baseline CSVs"

    a_prof = _has_profiled_traces(dir_a)
    b_prof = _has_profiled_traces(dir_b)
    if a_prof and b_prof:
        note = "both dirs have profiled traces"
        if a_base != b_base:
            missing = "B" if a_base else "A"
            note += f" (dir {missing} has no baseline CSVs)"
        return "profiled", note

    a_log = _has_log_files(dir_a)
    b_log = _has_log_files(dir_b)
    if a_log and b_log:
        return "log", "using log files (no profiled traces in common)"

    return "baseline", "could not auto-detect; defaulting to baseline"


def _build_outlier_fn(args):
    if args.outlier == "mad":
        return (
            lambda vals: ra.mad_outliers(vals, threshold=args.mad_threshold),
            f"MAD (threshold={args.mad_threshold})",
        )
    return (
        lambda vals: ra.iqr_outliers(vals, factor=args.iqr_factor),
        f"IQR (factor={args.iqr_factor})",
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare two rccl-tests perf-run directories side-by-side",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              %(prog)s perf-runs/20260319-A perf-runs/20260319-B
              %(prog)s perf-runs/A perf-runs/B --source profiled
              %(prog)s perf-runs/A perf-runs/B --csv comparison.csv
              %(prog)s perf-runs/A perf-runs/B --label-a "old" --label-b "new"
        """),
    )
    parser.add_argument("run_dir_a", help="First run directory (A)")
    parser.add_argument("run_dir_b", help="Second run directory (B)")
    parser.add_argument(
        "--source", choices=["auto", "baseline", "profiled", "log"],
        default="auto",
        help="Data source to compare (default: auto).  'auto' picks the best"
             " common source; 'log' parses rccl-tests stdout log files.",
    )
    parser.add_argument("--label-a", default=None, help="Label for run A")
    parser.add_argument("--label-b", default=None, help="Label for run B")
    parser.add_argument(
        "--csv", default=None, metavar="FILE",
        help="Write comparison to CSV file",
    )
    parser.add_argument(
        "--outlier", choices=["mad", "iqr"], default="mad",
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
    outlier_fn, method_name = _build_outlier_fn(args)

    source = args.source
    if source == "auto":
        source, note = _resolve_auto_source(args.run_dir_a, args.run_dir_b)
        auto_msg = f"  Auto-selected source: {source}  ({note})"
    else:
        auto_msg = None

    label_a = _run_label(args.run_dir_a, args.label_a)
    label_b = _run_label(args.run_dir_b, args.label_b)

    print(f"Comparing:")
    print(f"  A: {args.run_dir_a}  ({label_a})")
    print(f"  B: {args.run_dir_b}  ({label_b})")
    print(f"  Source: {source}   Outlier: {method_name}")
    if auto_msg:
        print(auto_msg)
    print()

    _LOADERS = {
        "baseline": load_baseline_data,
        "profiled": load_profiled_data,
        "log":      load_log_data,
    }
    loader = _LOADERS[source]
    data_a, np_a = loader(args.run_dir_a, outlier_fn)
    data_b, np_b = loader(args.run_dir_b, outlier_fn)

    np_val = np_a or np_b

    all_keys = sorted(set(data_a.keys()) | set(data_b.keys()))
    if not all_keys:
        print("No matching data found in either run directory.", file=sys.stderr)
        sys.exit(1)

    all_merged = []
    for key in all_keys:
        collective, dtype = key
        rows_a = data_a.get(key, [])
        rows_b = data_b.get(key, [])
        merged = join_reports(rows_a, rows_b)
        all_merged.append((collective, dtype, merged))
        print_comparison(merged, collective, dtype, label_a, label_b, np_val)

    if args.csv:
        write_comparison_csv(args.csv, all_merged, label_a, label_b)


if __name__ == "__main__":
    main()
