#!/usr/bin/env python3
"""
Interactive Plotly charts of RCCL kernel durations with Bayesian
changepoint segmentation.

Reads rocprofv3 data from timestamped run directories under DATA_DIR.
When given multiple directories, overlays each run on the same chart with
distinct colours and per-run segment analysis.

Usage:
    python plot_segments.py [RUN_DIR ...]

With no arguments, uses the most recent timestamped directory under DATA_DIR.

For each benchmark (excluding all_reduce_bias) and placement (inplace /
outofplace), produces one HTML file with:
  - all raw data points (semi-transparent scatter)      [single-run only]
  - min / max envelope lines                            [single-run only]
  - shaded IQR band (Q1–Q3)
  - median line
  - Bayesian-selected piecewise power-law segments with best-fit equations
  - vertical changepoint markers
  - protocol-band shading                               [single-run only]

Also generates an index.html linking every chart.
"""

import sqlite3
import json
import os
import sys
import re
import glob
import numpy as np
from scipy.special import gammaln
from collections import defaultdict
import plotly.graph_objects as go

DATA_DIR = "/work/lmeadows/data"
OUT_DIR = "/work/lmeadows/data/plots"
SKIP_BENCHMARKS = {"all_reduce_bias", "plots"}


# ── run-directory helpers ────────────────────────────────────────────────


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


# ── data extraction ──────────────────────────────────────────────────────


def find_results_db(rank_dir):
    dbs = glob.glob(os.path.join(rank_dir, "**", "*_results.db"), recursive=True)
    return dbs[0] if dbs else None


def extract_rank_data(db_path):
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


def human_bytes(n):
    for unit in ("B", "K", "M", "G"):
        if n < 1024:
            return f"{int(n)}{unit}" if float(n) == int(n) else f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{int(n)}T" if float(n) == int(n) else f"{n:.1f}T"


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


NRANKS = 8
SCALED_FUNCS = {"ReduceScatter", "AllGather", "Broadcast", "Reduce"}


def extract_protocol_map(bench_dir):
    """
    Return {rccl_tests_size: {"proto": str, "algo": str, "funcId": int}}
    from RCCL-PROTO markers.

    RCCL's nBytes is the total operation size.  For collectives that divide
    work across ranks (AllGather, ReduceScatter, Broadcast, Reduce), the
    rccl-tests marker reports the per-rank portion, so we divide nBytes by
    NRANKS for those to align the two coordinate systems.
    """
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
            _, func, nbytes_str, algo, proto, func_id = parts[:6]
            nbytes = int(nbytes_str)
            chart_size = nbytes // NRANKS if func in SCALED_FUNCS else nbytes
            if chart_size not in proto_map:
                proto_map[chart_size] = {"proto": proto, "algo": algo, "funcId": int(func_id)}
        if proto_map:
            break
    return proto_map


# ── Bayesian piecewise-linear changepoint detection ─────────────────────
#
# We work in log₁₀ space (log size vs log duration) and fit OLS lines per
# segment.  Model comparison uses the analytic log marginal likelihood
# under a conjugate Normal-Inverse-Gamma prior with weakly informative
# hyper-parameters.


def _segment_log_ml(y, X, prior_var=100.0, a0=1e-3, b0=1e-3):
    """
    Log marginal likelihood of y | X under a Normal-Inverse-Gamma
    conjugate prior:
        β | σ² ~ N(0, σ² · prior_var · I)
        σ²     ~ IG(a0, b0)
    Returns the (scalar) log marginal likelihood.
    """
    n, p = X.shape
    V0_inv = np.eye(p) / prior_var
    XtX = X.T @ X
    Vn_inv = XtX + V0_inv
    Vn = np.linalg.inv(Vn_inv)
    beta_n = Vn @ (X.T @ y)
    an = a0 + n / 2.0
    bn = b0 + 0.5 * (y @ y - beta_n @ Vn_inv @ beta_n)
    if bn <= 0:
        bn = 1e-15

    sign_V0, logdet_V0 = np.linalg.slogdet(V0_inv)
    sign_Vn, logdet_Vn = np.linalg.slogdet(Vn_inv)

    lml = (
        -n / 2.0 * np.log(2 * np.pi)
        + 0.5 * logdet_V0
        - 0.5 * logdet_Vn
        + a0 * np.log(b0)
        - an * np.log(bn)
        + gammaln(an)
        - gammaln(a0)
    )
    return lml


def _score_segmentation(log_s, log_t, breakpoints):
    """Total log marginal likelihood for a set of breakpoints."""
    n = len(log_s)
    bps = [0] + list(breakpoints) + [n]
    total = 0.0
    for i in range(len(bps) - 1):
        sl = slice(bps[i], bps[i + 1])
        seg_s = log_s[sl]
        seg_t = log_t[sl]
        if len(seg_s) < 3:
            return -np.inf
        X = np.column_stack([np.ones(len(seg_s)), seg_s])
        total += _segment_log_ml(seg_t, X)
    return total


MIN_SEG_PTS = 3


def find_changepoints(sizes, medians):
    """
    Exhaustive Bayesian model comparison over 0, 1, or 2 changepoints.
    Returns a list of indices into `sizes` where segments start (excluding 0).
    """
    log_s = np.log10(sizes.astype(np.float64))
    log_t = np.log10(medians)
    n = len(log_s)

    best_score = _score_segmentation(log_s, log_t, [])
    best_bps = []

    for cp1 in range(MIN_SEG_PTS, n - MIN_SEG_PTS + 1):
        score = _score_segmentation(log_s, log_t, [cp1])
        if score > best_score:
            best_score = score
            best_bps = [cp1]

    for cp1 in range(MIN_SEG_PTS, n - 2 * MIN_SEG_PTS + 1):
        for cp2 in range(cp1 + MIN_SEG_PTS, n - MIN_SEG_PTS + 1):
            score = _score_segmentation(log_s, log_t, [cp1, cp2])
            if score > best_score:
                best_score = score
                best_bps = [cp1, cp2]

    return best_bps


# ── segment fitting and labelling ───────────────────────────────────────


def fit_segment(sizes, medians):
    """
    OLS in log-log space.  Returns (slope, intercept, r², equation_string).
    Equation is in *physical* units (µs, bytes).
    """
    log_s = np.log10(sizes.astype(np.float64))
    log_t = np.log10(medians)
    X = np.column_stack([np.ones_like(log_s), log_s])
    beta, *_ = np.linalg.lstsq(X, log_t, rcond=None)
    intercept, slope = beta
    y_pred = X @ beta
    ss_res = np.sum((log_t - y_pred) ** 2)
    ss_tot = np.sum((log_t - np.mean(log_t)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    a = 10 ** intercept

    if abs(slope) < 0.10:
        mean_t = np.mean(medians)
        eq = f"T ≈ {mean_t:.1f} µs  (latency-bound)"
    elif abs(slope - 1.0) < 0.15:
        bw_bytes_per_us = 1.0 / a
        if bw_bytes_per_us > 1e3:
            eq = f"T ≈ size / {bw_bytes_per_us/1e3:.2f} MB/µs  (bandwidth-bound, slope={slope:.2f})"
        else:
            eq = f"T ≈ size / {bw_bytes_per_us:.2f} B/µs  (bandwidth-bound, slope={slope:.2f})"
    else:
        eq = f"T ≈ {a:.3g} · size^{slope:.2f}  (R²={r2:.3f})"

    return slope, intercept, r2, eq


# ── colour helpers ───────────────────────────────────────────────────────

# Per-segment colours used in single-run mode (original palette).
SEGMENT_COLORS = [
    "rgba(31,119,180,0.85)",   # blue
    "rgba(255,127,14,0.85)",   # orange
    "rgba(44,160,44,0.85)",    # green
]

# Base RGB tuples for colouring distinct runs.
_RUN_BASE_COLORS = [
    (31, 119, 180),    # blue
    (214, 39, 40),     # red
    (44, 160, 44),     # green
    (255, 127, 14),    # orange
    (148, 103, 189),   # purple
    (140, 86, 75),     # brown
    (227, 119, 194),   # pink
    (127, 127, 127),   # gray
]

PROTO_BAND_COLORS = {
    "LL":     "rgba(100,180,255,0.08)",
    "LL128":  "rgba(255,180,60,0.08)",
    "SIMPLE": "rgba(80,200,80,0.08)",
}
PROTO_LINE_COLORS = {
    "LL":     "rgba(100,180,255,0.5)",
    "LL128":  "rgba(255,180,60,0.5)",
    "SIMPLE": "rgba(80,200,80,0.5)",
}


def _rgba(rgb, alpha):
    return f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{alpha})"


# ── plotly chart builder ────────────────────────────────────────────────


def _add_protocol_bands(fig, proto_map):
    """Add vertical protocol-region shading to the figure."""
    proto_regions = []
    sorted_sizes = sorted(proto_map.keys())
    if not sorted_sizes:
        return
    cur_proto = proto_map[sorted_sizes[0]]["proto"]
    cur_algo = proto_map[sorted_sizes[0]]["algo"]
    region_start = sorted_sizes[0]
    for i in range(1, len(sorted_sizes)):
        s = sorted_sizes[i]
        p = proto_map[s]["proto"]
        a = proto_map[s]["algo"]
        if p != cur_proto or a != cur_algo:
            proto_regions.append((region_start, sorted_sizes[i - 1], cur_proto, cur_algo))
            cur_proto = p
            cur_algo = a
            region_start = s
    proto_regions.append((region_start, sorted_sizes[-1], cur_proto, cur_algo))

    for lo, hi, proto, algo in proto_regions:
        x0 = lo / 1.4
        x1 = hi * 1.4
        fill = PROTO_BAND_COLORS.get(proto, "rgba(200,200,200,0.08)")
        line_c = PROTO_LINE_COLORS.get(proto, "rgba(200,200,200,0.4)")
        fig.add_vrect(
            x0=x0, x1=x1,
            fillcolor=fill,
            line=dict(color=line_c, width=1, dash="dot"),
            layer="below",
            annotation_text=f"{proto} ({algo})",
            annotation_position="top left",
            annotation_font_size=10,
            annotation_font_color=line_c.replace("0.5)", "0.9)"),
        )


def _compute_run_stats(size_data):
    """Return (sizes, medians, q1, q3, mins, maxs, all_durations) arrays."""
    sizes = np.array(sorted(size_data.keys()))
    all_durations = {s: np.array(size_data[s]) for s in sizes}
    medians = np.array([np.median(all_durations[s]) for s in sizes])
    q1 = np.array([np.percentile(all_durations[s], 25) for s in sizes])
    q3 = np.array([np.percentile(all_durations[s], 75) for s in sizes])
    mins = np.array([np.min(all_durations[s]) for s in sizes])
    maxs = np.array([np.max(all_durations[s]) for s in sizes])
    return sizes, medians, q1, q3, mins, maxs, all_durations


def _compute_segments(sizes, medians):
    """Return list of segment dicts with fit info."""
    n_sizes = len(sizes)
    changepoints = find_changepoints(sizes, medians)
    seg_bounds = [0] + changepoints + [n_sizes]

    segments = []
    for i in range(len(seg_bounds) - 1):
        s_idx, e_idx = seg_bounds[i], seg_bounds[i + 1]
        seg_sizes = sizes[s_idx:e_idx]
        seg_medians = medians[s_idx:e_idx]
        slope, intercept, r2, eq = fit_segment(seg_sizes, seg_medians)
        fit_x = np.logspace(
            np.log10(float(seg_sizes[0])), np.log10(float(seg_sizes[-1])), 200
        )
        fit_y = (10 ** intercept) * fit_x ** slope
        segments.append(dict(
            s_idx=s_idx, e_idx=e_idx,
            slope=slope, intercept=intercept, r2=r2, eq=eq,
            fit_x=fit_x, fit_y=fit_y,
            lo=human_bytes(seg_sizes[0]), hi=human_bytes(seg_sizes[-1]),
        ))
    return segments


def build_chart(bench_name, runs_data, placement):
    """
    Build a Plotly Figure for one benchmark + placement across one or more runs.

    runs_data: [(label, {size: [durs]}, proto_map_or_None), …]
    Returns (fig, [(label, segments), …]) or (None, None).
    """
    active = [(lbl, sd, pm) for lbl, sd, pm in runs_data if sd]
    if not active:
        return None, None

    multi = len(active) > 1
    fig = go.Figure()
    all_segments_info = []

    for ri, (rlabel, size_data, proto_map) in enumerate(active):
        color_rgb = _RUN_BASE_COLORS[ri % len(_RUN_BASE_COLORS)]
        prefix = f"[{rlabel}] " if multi else ""

        sizes, medians, q1, q3, mins, maxs, all_durations = _compute_run_stats(size_data)
        segments = _compute_segments(sizes, medians)

        # ── protocol bands (single-run only, first run) ──
        if not multi and proto_map:
            _add_protocol_bands(fig, proto_map)

        # ── segment fits ──
        for i, seg in enumerate(segments):
            if multi:
                seg_color = _rgba(color_rgb, 0.85)
                dash = ["dash", "dot", "dashdot"][i % 3]
            else:
                seg_color = SEGMENT_COLORS[i % len(SEGMENT_COLORS)]
                dash = "dash"
            fig.add_trace(go.Scatter(
                x=seg["fit_x"], y=seg["fit_y"],
                mode="lines",
                line=dict(color=seg_color, width=3, dash=dash),
                name=f"{prefix}Seg {i+1} ({seg['lo']}–{seg['hi']}): {seg['eq']}",
            ))
            if seg["s_idx"] > 0:
                cp_size = float(sizes[seg["s_idx"]])
                cp_color = _rgba(color_rgb, 0.6) if multi else "red"
                fig.add_vline(
                    x=cp_size, line_dash="dash", line_color=cp_color, opacity=0.6,
                    annotation_text=f" {prefix}cp {human_bytes(int(cp_size))}",
                    annotation_position="top left",
                    annotation_font_size=11,
                    annotation_font_color=cp_color,
                )

        # ── median points ──
        median_color = _rgba(color_rgb, 1.0) if multi else "black"
        fig.add_trace(go.Scatter(
            x=sizes, y=medians,
            mode="markers",
            marker=dict(size=6, color=median_color, symbol="circle"),
            name=f"{prefix}Median",
        ))

        # ── IQR band ──
        iqr_color = _rgba(color_rgb, 0.12) if multi else "rgba(120,120,200,0.18)"
        fig.add_trace(go.Scatter(
            x=np.concatenate([sizes, sizes[::-1]]),
            y=np.concatenate([q3, q1[::-1]]),
            fill="toself",
            fillcolor=iqr_color,
            line=dict(width=0),
            name=f"{prefix}IQR (Q1–Q3)",
            hoverinfo="skip",
        ))

        # ── all raw data points ──
        scatter_color = _rgba(color_rgb, 0.25)
        raw_x, raw_y = [], []
        for s in sizes:
            for d in all_durations[s]:
                raw_x.append(s)
                raw_y.append(d)
        fig.add_trace(go.Scattergl(
            x=raw_x, y=raw_y,
            mode="markers",
            marker=dict(size=3, color=scatter_color),
            name=f"{prefix}All observations",
            hovertemplate="size=%{x}  dur=%{y:.2f} µs<extra></extra>",
        ))

        all_segments_info.append((rlabel, segments))

    # ── x-axis ticks from the union of all runs' sizes ──
    all_sizes_combined = set()
    for _, sd, _ in active:
        all_sizes_combined.update(sd.keys())
    tick_vals = sorted(all_sizes_combined)
    tick_text = [human_bytes(s) for s in tick_vals]
    if len(tick_vals) > 15:
        step = max(1, len(tick_vals) // 15)
        tick_vals = tick_vals[::step]
        tick_text = tick_text[::step]

    fig.update_layout(
        title=dict(
            text=f"{bench_name} — {placement}",
            font=dict(size=18),
        ),
        xaxis=dict(
            title="Message Size",
            type="log",
            tickvals=tick_vals,
            ticktext=tick_text,
            tickangle=-45,
            gridcolor="rgba(200,200,200,0.3)",
        ),
        yaxis=dict(
            title="Kernel Duration (µs)",
            type="log",
            gridcolor="rgba(200,200,200,0.3)",
        ),
        legend=dict(
            orientation="h", yanchor="top", y=-0.25,
            xanchor="left", x=0,
            font=dict(size=10),
        ),
        margin=dict(b=150),
        template="plotly_white",
        width=1100,
        height=650,
        hovermode="closest",
    )

    return fig, all_segments_info


# ── main ────────────────────────────────────────────────────────────────


def main():
    run_dirs = resolve_run_dirs(sys.argv[1:])
    labels = [run_label(d) for d in run_dirs]

    print("Run directories:")
    for lbl, d in zip(labels, run_dirs):
        print(f"  {lbl}: {d}")
    print()

    os.makedirs(OUT_DIR, exist_ok=True)
    benchmarks = discover_benchmarks(run_dirs)

    index_rows = []

    for bench in benchmarks:
        print(f"Processing {bench} ...")

        runs_all_data = []
        for lbl, run_dir in zip(labels, run_dirs):
            bench_dir = os.path.join(run_dir, bench)
            if not os.path.isdir(bench_dir):
                runs_all_data.append((lbl, None, None))
                continue

            data = collect_benchmark_data(bench_dir)
            if data is None:
                runs_all_data.append((lbl, None, None))
                continue

            proto_map = extract_protocol_map(bench_dir)
            if proto_map:
                print(f"  [{lbl}] protocol map: {len(proto_map)} sizes, "
                      f"protos={set(v['proto'] for v in proto_map.values())}")
            else:
                print(f"  [{lbl}] no RCCL-PROTO markers found")

            runs_all_data.append((lbl, data, proto_map))

        for placement in ("outofplace", "inplace"):
            runs_for_chart = []
            for lbl, data, proto_map in runs_all_data:
                if data is None:
                    runs_for_chart.append((lbl, None, None))
                    continue
                size_data = {
                    s: durs for (s, p), durs in data.items() if p == placement
                }
                runs_for_chart.append((lbl, size_data if size_data else None, proto_map))

            fig, all_seg_info = build_chart(bench, runs_for_chart, placement)
            if fig is None:
                continue

            fname = f"{bench}_{placement}.html"
            fpath = os.path.join(OUT_DIR, fname)
            fig.write_html(fpath, include_plotlyjs="cdn")

            n_seg_parts = []
            for rlbl, segments in all_seg_info:
                seg_str = " → ".join(
                    f"{s['lo']}–{s['hi']} slope={s['slope']:.2f}"
                    for s in segments
                )
                n_seg_parts.append(f"{rlbl}: {seg_str}" if len(all_seg_info) > 1 else seg_str)
                print(f"  wrote {fname}  [{rlbl}: {len(segments)} segments]")

            seg_summary = "<br>".join(n_seg_parts)
            run_labels_str = ", ".join(lbl for lbl, _, _ in runs_for_chart if _ is not None)
            index_rows.append((bench, placement, fname, seg_summary, run_labels_str))

    # ── index page ──
    rows_html = "\n".join(
        f'<tr><td>{b}</td><td>{p}</td>'
        f'<td><a href="{f}">{f}</a></td>'
        f'<td style="font-size:0.85em">{runs}</td>'
        f'<td style="font-size:0.85em">{s}</td></tr>'
        for b, p, f, s, runs in index_rows
    )
    index = f"""<!DOCTYPE html>
<html><head><title>RCCL Benchmark Segments</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2em; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: left; }}
th {{ background: #f4f4f4; }}
a {{ color: #1f77b4; }}
</style></head><body>
<h1>RCCL Kernel Duration – Bayesian Segmentation</h1>
<p>Runs: {', '.join(labels)}</p>
<table>
<tr><th>Benchmark</th><th>Placement</th><th>Chart</th><th>Runs</th><th>Segments</th></tr>
{rows_html}
</table>
</body></html>"""
    idx_path = os.path.join(OUT_DIR, "index.html")
    with open(idx_path, "w") as f:
        f.write(index)
    print(f"\nIndex page: {idx_path}")


if __name__ == "__main__":
    main()
