#!/usr/bin/env python3
"""Unified benchmark analysis: report, compare, and plot rccl-tests perf runs.

Wraps the roctx_analyze, roctx_compare, and roctx_plot modules from the
rccl-tests tools directory.  Auto-discovers the tools path from the workspace
layout (rccl/*/projects/rccl-tests/tools/).

Subcommands:
    list      Show available runs with metadata labels
    report    Text report for a single run
    compare   Side-by-side comparison of two runs (text + optional plot)
    plot      Overlay plot of one or more runs

Examples:
    ./analyze.py list
    ./analyze.py report perf-runs/20260321-130508
    ./analyze.py compare perf-runs/20260321-125804 perf-runs/20260321-130508 --plot cmp.png
    ./analyze.py plot perf-runs/20260321-125804 perf-runs/20260321-130508 -o overlay.png
"""

import argparse
import glob
import json
import os
import sys
import textwrap
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Tool discovery
# ---------------------------------------------------------------------------

def find_tools_dir(workspace=None):
    return SCRIPT_DIR

def find_perf_runs_dir():
    runs = os.path.join(SCRIPT_DIR, "..", "perf-runs")
    if os.path.isdir(runs):
        return os.path.abspath(runs)
    return None

def setup_tools_path(tools_dir_override=None):
    """Add the roctx tools directory to sys.path and import modules."""
    tools_dir = tools_dir_override or find_tools_dir()
    if tools_dir is None:
        print("Cannot find rccl-tests tools directory.", file=sys.stderr)
        print("Use --tools-dir to specify it, or ensure rccl/*/projects/rccl-tests/tools/ exists.",
              file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(tools_dir):
        print(f"Tools directory does not exist: {tools_dir}", file=sys.stderr)
        sys.exit(1)
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    return tools_dir


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def load_meta(run_dir):
    """Load metadata.json from a run directory, or return empty dict."""
    path = os.path.join(run_dir, "metadata.json")
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return {}


def run_label(run_dir, override=None):
    """Derive a short label for a run.  Uses librccl.git_hash if available."""
    if override:
        return override
    meta = load_meta(run_dir)
    git_hash = (meta.get("librccl") or {}).get("git_hash")
    if git_hash:
        return git_hash
    return os.path.basename(os.path.normpath(run_dir))


def run_summary(run_dir):
    """One-line summary for a run directory."""
    meta = load_meta(run_dir)
    dirname = os.path.basename(os.path.normpath(run_dir))
    label = run_label(run_dir)
    status = meta.get("status", "?")
    host = meta.get("hostname", "?")
    np_val = (meta.get("matrix") or {}).get("np", "?")
    tests = (meta.get("matrix") or {}).get("tests", [])
    dtypes = (meta.get("matrix") or {}).get("dtypes", [])
    baseline = (meta.get("matrix") or {}).get("baseline", False)
    reps = (meta.get("matrix") or {}).get("repeats", "?")
    parts = [
        dirname,
        f"label={label}",
        f"host={host}",
        f"np={np_val}",
        f"tests={','.join(tests)}",
        f"dtypes={','.join(dtypes)}",
        f"reps={reps}",
    ]
    if baseline:
        parts.append("baseline=yes")
    parts.append(f"status={status}")
    return "  ".join(parts)


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------

def cmd_list(args):
    perf_dir = args.perf_dir or find_perf_runs_dir()
    if perf_dir is None or not os.path.isdir(perf_dir):
        print("No perf-runs directory found. Use --perf-dir.", file=sys.stderr)
        return 1

    entries = []
    for name in sorted(os.listdir(perf_dir)):
        if name.startswith("."):
            continue
        path = os.path.join(perf_dir, name)
        if os.path.isdir(path) and os.path.isfile(os.path.join(path, "metadata.json")):
            entries.append(path)

    if not entries:
        print(f"No runs found in {perf_dir}")
        return 0

    print(f"Runs in {perf_dir}:\n")
    for path in entries:
        print(f"  {run_summary(path)}")
    print()
    return 0


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def cmd_report(args):
    setup_tools_path(args.tools_dir)
    import roctx_analyze as ra
    import roctx_compare as rc

    run_dir = args.run_dir
    label = run_label(run_dir)
    meta = load_meta(run_dir)
    np_val = (meta.get("matrix") or meta.get("args") or {}).get("np")

    print(f"Run: {os.path.basename(run_dir)}  label={label}")
    if np_val:
        print(f"  np={np_val}")
    print()

    outlier_fn, method_name = _build_outlier_fn(args)

    source = args.source
    if source == "auto":
        has_base = rc._has_baseline_csvs(run_dir)
        has_prof = bool(ra.discover_multi_run_groups(run_dir))
        has_log = rc._has_log_files(run_dir)
        if has_base:
            source = "baseline"
        elif has_prof:
            source = "profiled"
        elif has_log:
            source = "log"
        else:
            print("No data found in run directory.", file=sys.stderr)
            return 1

    print(f"Source: {source}   Outlier: {method_name}")
    print()

    if source == "baseline":
        data, np_from_meta = rc.load_baseline_data(run_dir, outlier_fn)
        np_val = np_val or np_from_meta
        for (collective, dtype), rows in sorted(data.items()):
            _print_baseline_report(rows, collective, dtype, method_name, np_val)

    elif source == "profiled":
        multi_groups = ra.discover_multi_run_groups(run_dir)
        if multi_groups:
            for (collective, dtype), subdirs in multi_groups.items():
                print(f"{'=' * 60}")
                print(f"  {collective}  /  {dtype}  ({len(subdirs)} rep(s))")
                print(f"{'=' * 60}")
                rows, show_bw, _regions = ra._analyze_dirs(
                    subdirs, np_val, collective, outlier_fn, method_name)
                ra.print_report(rows, method_name, show_bw=show_bw)
        else:
            print("No profiled trace directories found.", file=sys.stderr)
            return 1

    elif source == "log":
        data, np_from_meta = rc.load_log_data(run_dir, outlier_fn)
        np_val = np_val or np_from_meta
        for (collective, dtype), rows in sorted(data.items()):
            _print_baseline_report(rows, collective, dtype, method_name, np_val)

    return 0


def _fmt_us_stat(ns):
    if ns is None:
        return "--"
    return f"{ns / 1000.0:.2f}"


def cmd_variance(args):
    """GPU-time jitter per (size, place) from profiled traces.

    Reports, per point: per-launch on-GPU duration (End-Start; the accurate
    rocprofv3 HW timestamp) and inter-launch gap (next.Start - this.End), with
    median/std/p90, plus the fraction of wall time spent in gaps. This exposes
    launch jitter/stalls that rccl-tests' host_time/N averaging hides.
    """
    setup_tools_path(args.tools_dir)
    import roctx_analyze as ra

    run_dir = args.run_dir
    groups = ra.discover_multi_run_groups(run_dir)
    if not groups:
        print("No profiled trace directories found (variance needs profiled "
              "runs).", file=sys.stderr)
        return 1

    print(f"Run: {os.path.basename(run_dir)}   (profiled GPU-time jitter)")
    print("  dur = per-launch on-GPU duration (End-Start); "
          "gap = inter-launch (next.Start - prev.End); us.")
    print()

    for (collective, dtype), subdirs in sorted(groups.items()):
        samples = ra.collective_timing_samples(subdirs)
        if not samples:
            continue
        print(f"{'=' * 78}")
        print(f"  {collective}  /  {dtype}  ({len(subdirs)} trace dir(s))")
        print(f"{'=' * 78}")
        hdr = (f"{'size':>10}  {'place':>5}  {'n':>6}  "
               f"{'dur_med':>8}  {'dur_std':>8}  {'dur_p90':>8}  "
               f"{'gap_med':>8}  {'gap_std':>8}  {'gap_p90':>8}  {'%gap':>6}")
        print(hdr)
        print("-" * len(hdr))
        place_labels = {0: "oop", 1: "ip"}
        for (size, in_place) in sorted(samples, key=lambda k: (k[1], k[0])):
            durs = samples[(size, in_place)]["dur"]
            gaps = samples[(size, in_place)]["gap"]
            ds = ra.summarize(durs)
            gs = ra.summarize(gaps)
            if ds is None:
                continue
            sum_dur = sum(durs)
            sum_gap = sum(gaps) if gaps else 0
            pct_gap = (100.0 * sum_gap / (sum_dur + sum_gap)) if (sum_dur + sum_gap) else 0.0
            print(
                f"{size:>10}  {place_labels.get(in_place, in_place):>5}  {ds['n']:>6}  "
                f"{_fmt_us_stat(ds['p50']):>8}  {_fmt_us_stat(ds['std']):>8}  "
                f"{_fmt_us_stat(ds['p90']):>8}  "
                f"{_fmt_us_stat(gs['p50']) if gs else '--':>8}  "
                f"{_fmt_us_stat(gs['std']) if gs else '--':>8}  "
                f"{_fmt_us_stat(gs['p90']) if gs else '--':>8}  "
                f"{pct_gap:>5.1f}%"
            )
        print()
    return 0


def _print_baseline_report(rows, collective, dtype, method_name, np_val=None):
    """Print a text report from baseline/log row data."""
    import roctx_analyze as ra

    print(f"{'=' * 60}")
    hdr = f"  {collective}  /  {dtype}"
    if np_val:
        factor_fn = ra.BUS_BW_FACTOR.get(collective)
        if factor_fn:
            bus = factor_fn(np_val)
            hdr += f"  (np={np_val}, bus_factor={bus:.4f})"
    print(hdr)
    print(f"{'=' * 60}")

    place_labels = {0: "oop", 1: "ip"}
    hdr_line = (f"{'size':>10}  {'place':>5}  {'kept':>6}  {'out':>4}  "
                f"{'min':>10}  {'median':>10}  {'max':>10}  "
                f"{'algbw':>9}  {'busbw':>9}")
    sep = "-" * len(hdr_line)

    current_place = None
    for r in rows:
        ip = r["in_place"]
        if ip != current_place:
            if current_place is not None:
                print()
            current_place = ip
            place_name = "out-of-place" if ip == 0 else "in-place"
            print(f"  [{place_name}]")
            print(f"  {hdr_line}")
            print(f"  {sep}")

        med_us = r.get("median_us")
        min_us = r.get("min_us")
        max_us = r.get("max_us")
        algbw = r.get("algbw")
        busbw = r.get("busbw")

        def _fus(v):
            if v is None:
                return "--"
            if v >= 1000:
                return f"{v/1000:.1f}ms"
            return f"{v:.1f}us"

        def _fbw(v):
            if v is None:
                return "--"
            if v >= 1.0:
                return f"{v:.2f}"
            if v >= 0.001:
                return f"{v:.4f}"
            return f"{v:.2e}"

        line = (f"  {ra.fmt_size(r['size']):>10}  "
                f"{place_labels.get(ip, '?'):>5}  "
                f"{r['retained']:>6}  {r['outliers']:>4}  "
                f"{_fus(min_us):>10}  {_fus(med_us):>10}  {_fus(max_us):>10}  "
                f"{_fbw(algbw):>9}  {_fbw(busbw):>9}")
        print(line)

    print()
    print(f"  Outlier method: {method_name}")
    print(f"  algbw/busbw in GB/s")
    print()


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------

def cmd_compare(args):
    setup_tools_path(args.tools_dir)
    import roctx_compare as rc

    outlier_fn, method_name = _build_outlier_fn(args)

    label_a = run_label(args.run_dir_a, args.label_a)
    label_b = run_label(args.run_dir_b, args.label_b)

    source = args.source
    if source == "auto":
        source, note = rc._resolve_auto_source(args.run_dir_a, args.run_dir_b)
        auto_msg = f"  Auto-selected source: {source}  ({note})"
    else:
        auto_msg = None

    print(f"Comparing:")
    print(f"  A: {args.run_dir_a}  ({label_a})")
    print(f"  B: {args.run_dir_b}  ({label_b})")
    print(f"  Source: {source}   Outlier: {method_name}")
    if auto_msg:
        print(auto_msg)
    print()

    loaders = {
        "baseline": rc.load_baseline_data,
        "profiled": rc.load_profiled_data,
        "log":      rc.load_log_data,
    }
    loader = loaders[source]
    data_a, np_a = loader(args.run_dir_a, outlier_fn)
    data_b, np_b = loader(args.run_dir_b, outlier_fn)
    np_val = np_a or np_b

    all_keys = sorted(set(data_a.keys()) | set(data_b.keys()))
    if not all_keys:
        print("No matching data found in either run directory.", file=sys.stderr)
        return 1

    all_merged = []
    for key in all_keys:
        collective, dtype = key
        rows_a = data_a.get(key, [])
        rows_b = data_b.get(key, [])
        merged = rc.join_reports(rows_a, rows_b)
        all_merged.append((collective, dtype, merged))
        rc.print_comparison(merged, collective, dtype, label_a, label_b, np_val)

    if args.csv:
        rc.write_comparison_csv(args.csv, all_merged, label_a, label_b)

    if args.plot:
        _plot_comparison(all_merged, label_a, label_b, args.plot, np_val)

    return 0


# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------

def cmd_plot(args):
    setup_tools_path(args.tools_dir)
    import roctx_analyze as ra
    import roctx_compare as rc

    outlier_fn, method_name = _build_outlier_fn(args)
    metric = args.metric

    if getattr(args, "backend", "matplotlib") == "plotly":
        return _cmd_plot_plotly(args, outlier_fn)

    source = args.source
    run_dirs = args.run_dirs

    lib_data = []
    for run_dir in run_dirs:
        label = run_label(run_dir)
        if source == "auto":
            has_base = rc._has_baseline_csvs(run_dir)
            has_prof = bool(ra.discover_multi_run_groups(run_dir))
            has_log = rc._has_log_files(run_dir)
            if has_base:
                eff_source = "baseline"
            elif has_prof:
                eff_source = "profiled"
            elif has_log:
                eff_source = "log"
            else:
                print(f"  WARNING: no data in {run_dir}", file=sys.stderr)
                continue
        else:
            eff_source = source

        loader = {
            "baseline": rc.load_baseline_data,
            "profiled": rc.load_profiled_data,
            "log":      rc.load_log_data,
        }[eff_source]
        data, np_val = loader(run_dir, outlier_fn)
        if data:
            lib_data.append((label, data, np_val, eff_source))
            print(f"  {os.path.basename(run_dir)} -> {label}  "
                  f"source={eff_source}  groups={list(data.keys())}")
        else:
            print(f"  WARNING: no data from {run_dir}", file=sys.stderr)

    if not lib_data:
        print("No data to plot.", file=sys.stderr)
        return 1

    all_keys = list(dict.fromkeys(
        key for _, data, _, _ in lib_data for key in data
    ))

    _plot_overlay(lib_data, all_keys, metric, args.output)
    return 0


def _records_from_inputs(inputs, source, outlier_fn):
    """Build tidy records from a mix of run directories and exported JSON files."""
    import roctx_records as rr
    records = []
    for inp in inputs:
        if inp.endswith(".json") and os.path.isfile(inp):
            records.extend(rr.load_records(inp))
        else:
            records.extend(rr.build_records(inp, source=source, outlier_fn=outlier_fn))
    return records


def _cmd_plot_plotly(args, outlier_fn):
    setup_tools_path(args.tools_dir)
    try:
        import viz
    except ImportError as e:
        print(f"plotly backend needs the 'plotly' package: {e}", file=sys.stderr)
        print("  install it (e.g. into your analysis venv):  pip install plotly", file=sys.stderr)
        return 1

    records = _records_from_inputs(args.run_dirs, args.source, outlier_fn)
    if not records:
        print("No records to plot.", file=sys.stderr)
        return 1

    out = args.output
    if out.endswith(".png"):  # plotly default output is html
        out = out[:-4] + ".html"

    kind = getattr(args, "kind", "line")
    if kind in ("box", "violin"):
        metric = args.metric if args.metric != "eff_busbw" else "time"
        fig = viz.build_box_figure(
            records, metric=metric, kind=kind, color_by=args.color_by,
        )
        detail = f"kind={kind}"
    else:
        fig = viz.build_line_figure(
            records, metric=args.metric, band=args.band, color_by=args.color_by,
            overhead=getattr(args, "overhead", False),
        )
        detail = f"band={args.band}" + (", +overhead" if args.overhead else "")
    viz.write_html(fig, out)
    n_runs = len({r["run_dir"] for r in records})
    print(f"Saved interactive plot: {out}  "
          f"({len(records)} records, {n_runs} run(s), {detail})")
    return 0


def cmd_export(args):
    setup_tools_path(args.tools_dir)
    import roctx_records as rr
    outlier_fn, _ = _build_outlier_fn(args)
    records = rr.build_records_multi(args.run_dirs, source=args.source,
                                     outlier_fn=outlier_fn)
    if not records:
        print("No records produced.", file=sys.stderr)
        return 1
    rr.export_records(records, args.output, drop_samples=args.drop_samples)
    n_runs = len({r["run_dir"] for r in records})
    n_groups = len({(r["collective"], r["dtype"]) for r in records})
    size_kib = os.path.getsize(args.output) / 1024
    print(f"Wrote {len(records)} records from {n_runs} run(s), "
          f"{n_groups} (collective,dtype) group(s) -> {args.output} ({size_kib:.1f} KiB)")
    return 0


# ---------------------------------------------------------------------------
# Comparison plot (for compare subcommand)
# ---------------------------------------------------------------------------

def _plot_comparison(all_merged, label_a, label_b, output_path, np_val=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    n = len(all_merged)
    if n == 0:
        return

    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), squeeze=False)

    palette_a = "#377EB8"
    palette_b = "#E41A1C"
    place_style = {0: ("--", "s", "oop"), 1: ("-", "o", "ip")}

    for idx, (collective, dtype, merged) in enumerate(all_merged):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        for in_place, (ls, marker, place_label) in place_style.items():
            subset = [r for r in merged if r["in_place"] == in_place]
            if not subset:
                continue
            sizes = [r["size"] for r in subset]
            bw_a = [r.get("busbw_a") for r in subset]
            bw_b = [r.get("busbw_b") for r in subset]

            valid_a = [(s, b) for s, b in zip(sizes, bw_a) if b is not None]
            valid_b = [(s, b) for s, b in zip(sizes, bw_b) if b is not None]

            if valid_a:
                ax.plot([p[0] for p in valid_a], [p[1] for p in valid_a],
                        color=palette_a, linestyle=ls, marker=marker, markersize=4,
                        linewidth=1.4, label=f"{place_label} A:{label_a}" if idx == 0 else "")
            if valid_b:
                ax.plot([p[0] for p in valid_b], [p[1] for p in valid_b],
                        color=palette_b, linestyle=ls, marker=marker, markersize=4,
                        linewidth=1.4, label=f"{place_label} B:{label_b}" if idx == 0 else "")

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(_fmt_size_axis))
        ax.set_xlabel("Message size")
        ax.set_ylabel("busbw (GB/s)")
        ax.set_title(f"{collective} / {dtype}", fontweight="bold")
        ax.grid(True, which="major", linestyle=":", linewidth=0.6, alpha=0.6)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.3, alpha=0.35)

    # Hide unused subplots
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    # Shared legend
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center",
                   ncol=min(len(handles), 4), fontsize=8, framealpha=0.9,
                   bbox_to_anchor=(0.5, 1.02))

    fig.text(0.5, -0.01,
             f"A: {label_a}  vs  B: {label_b}"
             + (f"  (np={np_val})" if np_val else ""),
             ha="center", fontsize=9, color="#555")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved comparison plot: {output_path}")


# ---------------------------------------------------------------------------
# Overlay plot (for plot subcommand)
# ---------------------------------------------------------------------------

_PALETTE = ["#377EB8", "#E41A1C", "#4DAF4A", "#FF7F00", "#984EA3", "#A65628"]


def _plot_overlay(lib_data, all_keys, metric, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    n = len(all_keys)
    if n == 0:
        return

    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), squeeze=False)

    place_style = {0: ("--", "s", "oop"), 1: ("-", "o", "ip")}

    for key_idx, (collective, dtype) in enumerate(all_keys):
        row, col = divmod(key_idx, ncols)
        ax = axes[row][col]

        for lib_idx, (label, data, np_val, eff_source) in enumerate(lib_data):
            color = _PALETTE[lib_idx % len(_PALETTE)]
            rows = data.get((collective, dtype), [])
            if not rows:
                continue

            for in_place, (ls, marker, place_label) in place_style.items():
                subset = [r for r in rows if r["in_place"] == in_place]
                if not subset:
                    continue

                sizes = [r["size"] for r in subset]
                if metric == "busbw":
                    vals = [r.get("busbw") for r in subset]
                    ylabel = "busbw (GB/s)"
                elif metric == "algbw":
                    vals = [r.get("algbw") for r in subset]
                    ylabel = "algbw (GB/s)"
                else:
                    vals = [r.get("median_us") or (r.get("median", 0) / 1000.0 if r.get("median") else None)
                            for r in subset]
                    ylabel = "time (us)"

                valid = [(s, v) for s, v in zip(sizes, vals) if v is not None and v > 0]
                if not valid:
                    continue
                ax.plot([p[0] for p in valid], [p[1] for p in valid],
                        color=color, linestyle=ls, marker=marker, markersize=4,
                        linewidth=1.4, label=f"{place_label} {label}")

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(_fmt_size_axis))
        ax.set_xlabel("Message size")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{collective} / {dtype}", fontweight="bold")
        ax.grid(True, which="major", linestyle=":", linewidth=0.6, alpha=0.6)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.3, alpha=0.35)
        ax.legend(fontsize=7, framealpha=0.9, loc="best")

    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(f"Performance comparison ({metric})", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved overlay plot: {output_path}")


def _fmt_size_axis(nbytes, _pos=None):
    if nbytes >= 1024 ** 3:
        return f"{int(nbytes // 1024 ** 3)}G"
    if nbytes >= 1024 ** 2:
        return f"{int(nbytes // 1024 ** 2)}M"
    if nbytes >= 1024:
        return f"{int(nbytes // 1024)}K"
    return str(int(nbytes))


# ---------------------------------------------------------------------------
# Shared CLI helpers
# ---------------------------------------------------------------------------

def _build_outlier_fn(args):
    import roctx_analyze as ra
    method = getattr(args, "outlier", "mad")
    if method == "mad":
        threshold = getattr(args, "mad_threshold", 3.5)
        return (
            lambda vals: ra.mad_outliers(vals, threshold=threshold),
            f"MAD (threshold={threshold})",
        )
    factor = getattr(args, "iqr_factor", 1.5)
    return (
        lambda vals: ra.iqr_outliers(vals, factor=factor),
        f"IQR (factor={factor})",
    )


def _add_common_args(parser):
    """Add arguments shared across subcommands."""
    parser.add_argument(
        "--tools-dir", default=None,
        help="Path to rccl-tests tools/ directory (auto-detected by default)",
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Unified rccl-tests benchmark analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              %(prog)s list
              %(prog)s report perf-runs/20260321-130508
              %(prog)s compare perf-runs/A perf-runs/B --plot cmp.png
              %(prog)s plot perf-runs/A perf-runs/B -o overlay.png --metric busbw
        """),
    )

    sub = parser.add_subparsers(dest="cmd", help="subcommand")

    # -- list --
    p_list = sub.add_parser("list", help="Show available runs")
    p_list.add_argument("--perf-dir", default=None,
                        help="perf-runs directory (auto-detected by default)")
    p_list.add_argument("--tools-dir", default=None)

    # -- report --
    p_report = sub.add_parser("report", help="Text report for a single run")
    p_report.add_argument("run_dir", help="Run directory")
    p_report.add_argument("--source", choices=["auto", "baseline", "profiled", "log"],
                          default="auto", help="Data source (default: auto)")
    _add_common_args(p_report)

    # -- compare --
    p_cmp = sub.add_parser("compare", help="Compare two runs")
    p_cmp.add_argument("run_dir_a", help="First run directory (A)")
    p_cmp.add_argument("run_dir_b", help="Second run directory (B)")
    p_cmp.add_argument("--source", choices=["auto", "baseline", "profiled", "log"],
                        default="auto")
    p_cmp.add_argument("--label-a", default=None, help="Label for run A")
    p_cmp.add_argument("--label-b", default=None, help="Label for run B")
    p_cmp.add_argument("--csv", default=None, metavar="FILE",
                        help="Write comparison to CSV")
    p_cmp.add_argument("--plot", default=None, metavar="FILE",
                        help="Write comparison plot (e.g. cmp.png)")
    _add_common_args(p_cmp)

    # -- plot --
    p_plot = sub.add_parser("plot", help="Overlay plot of runs")
    p_plot.add_argument("run_dirs", nargs="+",
                        help="Run directories or exported records .json files")
    p_plot.add_argument("--source", choices=["auto", "baseline", "profiled", "log"],
                        default="auto")
    p_plot.add_argument("-o", "--output", default="comparison.png",
                        help="Output file (.png for matplotlib, .html for plotly)")
    p_plot.add_argument("--metric", choices=["busbw", "algbw", "time", "eff_busbw"],
                        default="busbw",
                        help="Metric to plot (default: busbw)")
    p_plot.add_argument("--backend", choices=["matplotlib", "plotly"],
                        default="matplotlib",
                        help="Plot backend (default: matplotlib). plotly = "
                             "interactive HTML with variance bands.")
    p_plot.add_argument("--kind", choices=["line", "box", "violin"],
                        default="line",
                        help="plotly plot kind: line+band, or per-size box/violin "
                             "distributions (default: line)")
    p_plot.add_argument("--band", choices=["none", "iqr", "minmax", "std"],
                        default="iqr",
                        help="Variance band for plotly line kind (default: iqr)")
    p_plot.add_argument("--overhead", action="store_true",
                        help="Overlay effective (incl. per-call overhead, e.g. DDA "
                             "staging copy) as a dotted line + shaded gap (line kind)")
    p_plot.add_argument("--color-by", choices=["label", "dtype", "machine"],
                        default=None,
                        help="Comparison factor mapped to colour (plotly; "
                             "auto-detected by default)")
    _add_common_args(p_plot)

    # -- export --
    p_export = sub.add_parser("export",
                              help="Export runs to a compact tidy-records JSON")
    p_export.add_argument("run_dirs", nargs="+", help="Run directories")
    p_export.add_argument("-o", "--output", default="records.json",
                          help="Output JSON file (default: records.json)")
    p_export.add_argument("--source", choices=["auto", "baseline", "profiled", "log"],
                          default="auto")
    p_export.add_argument("--drop-samples", action="store_true",
                          help="Omit per-iteration samples (smaller; no box/violin)")
    _add_common_args(p_export)

    # -- variance --
    p_var = sub.add_parser(
        "variance",
        help="Per-size GPU-time jitter from profiled traces: per-launch "
             "duration spread and inter-launch gaps")
    p_var.add_argument("run_dir", help="Run directory (profiled traces)")
    p_var.add_argument("--tools-dir", default=None)

    args = parser.parse_args()

    if args.cmd is None:
        parser.print_help()
        return 1

    dispatch = {
        "list": cmd_list,
        "report": cmd_report,
        "compare": cmd_compare,
        "plot": cmd_plot,
        "export": cmd_export,
        "variance": cmd_variance,
    }
    return dispatch[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main() or 0)
