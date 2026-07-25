#!/usr/bin/env python3
"""Tidy (long-form) records for rccl-tests perf runs, plus compact export.

This is the pivot layer between the source-specific loaders (baseline CSV,
profiled rocTX traces, log files) and any consumer (plots, comparisons,
notebooks).  Every source is normalized to a flat list of per-point records
with a uniform, microsecond-based schema and, crucially, the *retained sample
distribution* (`samples_us`) so downstream code can draw IQR bands, box/violin
plots, etc.

Because a record carries its own (library, machine, collective, dtype)
identity, records from many runs can simply be concatenated and then grouped by
whatever dimension you want to compare on.

Portability
-----------
The raw profiled kernel-trace CSVs are large; a run's tidy records are tiny by
comparison.  ``export_records`` writes them to a single JSON file so you can
copy just that to a well-provisioned box (e.g. a laptop) and plot there without
the GPU host or the bulky CSVs.

Record schema (all times in microseconds, bandwidth in GB/s)::

    run_dir, label, machine, source, collective, dtype, np,
    size, in_place, n, retained, outliers,
    min_us, p10_us, p25_us, median_us, p75_us, p90_us, max_us, mean_us, std_us,
    algbw, busbw, eff_busbw, overhead_us,
    samples_us: [float, ...]
"""

import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import roctx_analyze as ra
import roctx_compare as rc


# ---------------------------------------------------------------------------
# Run-level metadata
# ---------------------------------------------------------------------------

def _load_meta(run_dir):
    path = os.path.join(run_dir, "metadata.json")
    if os.path.isfile(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def run_label(run_dir, override=None):
    """Short human label for the librccl used in a run (the "library" axis)."""
    if override:
        return override
    meta = _load_meta(run_dir)
    git_hash = (meta.get("librccl") or {}).get("git_hash")
    if git_hash:
        branch, _, rest = git_hash.partition(":")
        short = rest.rstrip("+")
        if branch in ("HEAD", ""):
            ver = (meta.get("librccl") or {}).get("rccl_version", "")
            return f"v{ver}:{short}" if ver else (short or os.path.basename(run_dir))
        if len(branch) > 24:
            branch = branch[:24]
        return f"{branch}:{short}" if short else branch
    ver = (meta.get("librccl") or {}).get("rccl_version")
    if ver:
        return f"v{ver}"
    return os.path.basename(os.path.normpath(run_dir))


def run_machine(run_dir):
    return _load_meta(run_dir).get("hostname") or "?"


# ---------------------------------------------------------------------------
# Source resolution + loading
# ---------------------------------------------------------------------------

def resolve_source(run_dir, source="auto"):
    """Return an effective source name for *run_dir* ('baseline'|'profiled'|'log')."""
    if source != "auto":
        return source
    if rc._has_baseline_csvs(run_dir):
        return "baseline"
    if rc._has_profiled_traces(run_dir):
        return "profiled"
    if rc._has_log_files(run_dir):
        return "log"
    return None


_LOADERS = {
    "baseline": rc.load_baseline_data,
    "profiled": rc.load_profiled_data,
    "log": rc.load_log_data,
}


def _row_to_record_us(row, source):
    """Normalize one loader row (profiled ns-based or baseline/log us-based) to us."""
    if source == "profiled":
        def us(v):
            return (v / 1000.0) if v is not None else None
        samples_us = [s / 1000.0 for s in (row.get("samples") or [])]
        overhead = row.get("overhead")
        return {
            "min_us": us(row.get("min")),
            "p10_us": us(row.get("p10")),
            "p25_us": us(row.get("p25")),
            "median_us": us(row.get("median")),
            "p75_us": us(row.get("p75")),
            "p90_us": us(row.get("p90")),
            "max_us": us(row.get("max")),
            "mean_us": us(row.get("mean")),
            "std_us": us(row.get("std")),
            "algbw": row.get("algbw"),
            "busbw": row.get("busbw"),
            "eff_busbw": row.get("eff_busbw"),
            "overhead_us": us(overhead) if overhead else 0.0,
            "samples_us": samples_us,
        }
    # baseline / log rows are already microsecond-based
    return {
        "min_us": row.get("min_us"),
        "p10_us": row.get("p10_us"),
        "p25_us": row.get("p25_us"),
        "median_us": row.get("median_us"),
        "p75_us": row.get("p75_us"),
        "p90_us": row.get("p90_us"),
        "max_us": row.get("max_us"),
        "mean_us": row.get("mean_us"),
        "std_us": row.get("std_us"),
        "algbw": row.get("algbw"),
        "busbw": row.get("busbw"),
        "eff_busbw": None,
        "overhead_us": 0.0,
        "samples_us": list(row.get("samples_us") or []),
    }


def build_records(run_dir, source="auto", outlier_fn=None, label=None):
    """Load *run_dir* and return a flat list of tidy per-point records.

    *outlier_fn* defaults to MAD (threshold 3.5).  *label* overrides the
    library label (otherwise derived from metadata).
    """
    if outlier_fn is None:
        outlier_fn = lambda vals: ra.mad_outliers(vals, threshold=3.5)

    eff_source = resolve_source(run_dir, source)
    if eff_source is None:
        return []

    data, np_val = _LOADERS[eff_source](run_dir, outlier_fn)
    lbl = run_label(run_dir, label)
    machine = run_machine(run_dir)

    records = []
    for (collective, dtype), rows in data.items():
        for row in rows:
            rec = {
                "run_dir": os.path.basename(os.path.normpath(run_dir)),
                "label": lbl,
                "machine": machine,
                "source": eff_source,
                "collective": collective,
                "dtype": dtype,
                "np": np_val,
                "size": row["size"],
                "in_place": row["in_place"],
                "n": row.get("retained", 0),
                "retained": row.get("retained", 0),
                "outliers": row.get("outliers", 0),
            }
            rec.update(_row_to_record_us(row, eff_source))
            records.append(rec)
    return records


def build_records_multi(run_dirs, source="auto", outlier_fn=None, labels=None):
    """Concatenate tidy records from several run dirs.  *labels* is an optional
    parallel list of label overrides."""
    labels = labels or []
    out = []
    for i, rd in enumerate(run_dirs):
        lbl = labels[i] if i < len(labels) else None
        out.extend(build_records(rd, source=source, outlier_fn=outlier_fn, label=lbl))
    return out


# ---------------------------------------------------------------------------
# Compact export / import
# ---------------------------------------------------------------------------

def export_records(records, path, drop_samples=False):
    """Write *records* to a JSON file (portable, no GPU/CSV dependency).

    With *drop_samples* the per-iteration ``samples_us`` arrays are omitted for
    an even smaller file (percentiles/bands still work; box/violin won't)."""
    if drop_samples:
        records = [{k: v for k, v in r.items() if k != "samples_us"} for r in records]
    payload = {"schema": "roctx-records/1", "records": records}
    with open(path, "w") as f:
        json.dump(payload, f)
    return path


def load_records(path):
    """Load records previously written by ``export_records``."""
    with open(path) as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "records" in payload:
        return payload["records"]
    return payload  # tolerate a bare list


# ---------------------------------------------------------------------------
# CLI (export)
# ---------------------------------------------------------------------------

def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Export rccl-tests perf runs to a compact tidy-records JSON.",
    )
    parser.add_argument("run_dirs", nargs="+", help="Run directories")
    parser.add_argument("-o", "--output", default="records.json",
                        help="Output JSON file (default: records.json)")
    parser.add_argument("--source", choices=["auto", "baseline", "profiled", "log"],
                        default="auto")
    parser.add_argument("--drop-samples", action="store_true",
                        help="Omit per-iteration samples (smaller file, no box/violin)")
    parser.add_argument("--outlier", choices=["mad", "iqr"], default="mad")
    parser.add_argument("--mad-threshold", type=float, default=3.5)
    parser.add_argument("--iqr-factor", type=float, default=1.5)
    args = parser.parse_args(argv)

    if args.outlier == "mad":
        outlier_fn = lambda vals: ra.mad_outliers(vals, threshold=args.mad_threshold)
    else:
        outlier_fn = lambda vals: ra.iqr_outliers(vals, factor=args.iqr_factor)

    records = build_records_multi(args.run_dirs, source=args.source, outlier_fn=outlier_fn)
    if not records:
        print("No records produced.", file=sys.stderr)
        return 1
    export_records(records, args.output, drop_samples=args.drop_samples)
    n_runs = len({r["run_dir"] for r in records})
    n_groups = len({(r["collective"], r["dtype"]) for r in records})
    size_bytes = os.path.getsize(args.output)
    print(f"Wrote {len(records)} records from {n_runs} run(s), "
          f"{n_groups} (collective,dtype) group(s) -> {args.output} "
          f"({size_bytes / 1024:.1f} KiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
