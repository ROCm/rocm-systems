#!/usr/bin/env python3
"""
Interactive Plotly charts of RCCL kernel durations with Bayesian
changepoint segmentation.

For each benchmark (excluding all_reduce_bias) and placement (inplace /
outofplace), produces one HTML file with:
  - all raw data points (semi-transparent scatter)
  - min / max envelope lines
  - shaded IQR band (Q1–Q3)
  - median line
  - Bayesian-selected piecewise power-law segments with best-fit equations
  - vertical changepoint markers

Also generates an index.html linking every chart.
"""

import sqlite3
import json
import os
import glob
import math
import numpy as np
from scipy.special import gammaln
from collections import defaultdict
import plotly.graph_objects as go

DATA_DIR = "/work/lmeadows/data"
OUT_DIR = "/work/lmeadows/data/plots"
SKIP_BENCHMARKS = {"all_reduce_bias", "plots"}

# ── data extraction (reused from earlier script) ────────────────────────


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


# ── plotly chart builder ────────────────────────────────────────────────

SEGMENT_COLORS = [
    "rgba(31,119,180,0.85)",   # blue
    "rgba(255,127,14,0.85)",   # orange
    "rgba(44,160,44,0.85)",    # green
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


def build_chart(bench_name, size_data, placement, proto_map=None):
    """
    Build a Plotly Figure for one benchmark + placement.
    size_data: {size_bytes: [dur_µs, …]}
    Returns (fig, segment_info_list) or (None, None).
    """
    if not size_data:
        return None, None

    sizes = np.array(sorted(size_data.keys()))
    n_sizes = len(sizes)

    all_durations = {s: np.array(size_data[s]) for s in sizes}
    medians = np.array([np.median(all_durations[s]) for s in sizes])
    q1 = np.array([np.percentile(all_durations[s], 25) for s in sizes])
    q3 = np.array([np.percentile(all_durations[s], 75) for s in sizes])
    mins = np.array([np.min(all_durations[s]) for s in sizes])
    maxs = np.array([np.max(all_durations[s]) for s in sizes])

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
        segments.append(
            dict(
                s_idx=s_idx, e_idx=e_idx,
                slope=slope, intercept=intercept, r2=r2, eq=eq,
                fit_x=fit_x, fit_y=fit_y,
                lo=human_bytes(seg_sizes[0]), hi=human_bytes(seg_sizes[-1]),
            )
        )

    fig = go.Figure()

    # ── protocol bands (background shading) ──
    if proto_map:
        proto_regions = []
        sorted_sizes = sorted(proto_map.keys())
        if sorted_sizes:
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

    # ── segment fits first (so they appear first in legend) ──
    for i, seg in enumerate(segments):
        color = SEGMENT_COLORS[i % len(SEGMENT_COLORS)]
        fig.add_trace(go.Scatter(
            x=seg["fit_x"], y=seg["fit_y"],
            mode="lines",
            line=dict(color=color, width=3, dash="dash"),
            name=f"Seg {i+1} ({seg['lo']}–{seg['hi']}): {seg['eq']}",
        ))
        if seg["s_idx"] > 0:
            cp_size = float(sizes[seg["s_idx"]])
            fig.add_vline(
                x=cp_size, line_dash="dash", line_color="red", opacity=0.6,
                annotation_text=f" changepoint {human_bytes(int(cp_size))}",
                annotation_position="top left",
                annotation_font_size=11,
                annotation_font_color="red",
            )

    # ── median line ──
    fig.add_trace(go.Scatter(
        x=sizes, y=medians,
        mode="lines+markers",
        line=dict(color="black", width=2),
        marker=dict(size=5, color="black"),
        name="Median",
    ))

    # ── shaded IQR band ──
    fig.add_trace(go.Scatter(
        x=np.concatenate([sizes, sizes[::-1]]),
        y=np.concatenate([q3, q1[::-1]]),
        fill="toself",
        fillcolor="rgba(120,120,200,0.18)",
        line=dict(width=0),
        name="IQR (Q1–Q3)",
        hoverinfo="skip",
    ))

    # ── min / max envelopes ──
    fig.add_trace(go.Scatter(
        x=sizes, y=maxs,
        mode="lines",
        line=dict(color="rgba(160,160,160,0.55)", width=1, dash="dot"),
        name="Max",
    ))
    fig.add_trace(go.Scatter(
        x=sizes, y=mins,
        mode="lines",
        line=dict(color="rgba(160,160,160,0.55)", width=1, dash="dot"),
        name="Min",
    ))

    # ── all raw data points ──
    raw_x, raw_y = [], []
    for s in sizes:
        for d in all_durations[s]:
            raw_x.append(s)
            raw_y.append(d)
    fig.add_trace(go.Scattergl(
        x=raw_x, y=raw_y,
        mode="markers",
        marker=dict(size=3, color="rgba(80,80,80,0.25)"),
        name="All observations",
        hovertemplate="size=%{x}  dur=%{y:.2f} µs<extra></extra>",
    ))

    tick_vals = sizes.tolist()
    tick_text = [human_bytes(s) for s in sizes]
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

    return fig, segments


# ── main ────────────────────────────────────────────────────────────────


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    benchmarks = sorted(
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d)) and d not in SKIP_BENCHMARKS
    )

    index_rows = []

    for bench in benchmarks:
        bench_dir = os.path.join(DATA_DIR, bench)
        print(f"Processing {bench} ...")
        data = collect_benchmark_data(bench_dir)
        if data is None:
            print(f"  SKIP: no data")
            continue

        proto_map = extract_protocol_map(bench_dir)
        if proto_map:
            print(f"  protocol map: {len(proto_map)} sizes, protos={set(v['proto'] for v in proto_map.values())}")
        else:
            print(f"  no RCCL-PROTO markers found")

        for placement in ("outofplace", "inplace"):
            size_data = {
                s: durs for (s, p), durs in data.items() if p == placement
            }
            if not size_data:
                continue

            fig, segments = build_chart(bench, size_data, placement, proto_map)
            if fig is None:
                continue

            fname = f"{bench}_{placement}.html"
            fpath = os.path.join(OUT_DIR, fname)
            fig.write_html(fpath, include_plotlyjs="cdn")
            print(f"  wrote {fname}  ({len(segments)} segments)")

            seg_summary = " → ".join(
                f"{s['lo']}–{s['hi']} slope={s['slope']:.2f}"
                for s in segments
            )
            index_rows.append((bench, placement, fname, seg_summary))

    # ── index page ──
    rows_html = "\n".join(
        f'<tr><td>{b}</td><td>{p}</td>'
        f'<td><a href="{f}">{f}</a></td>'
        f'<td style="font-size:0.85em">{s}</td></tr>'
        for b, p, f, s in index_rows
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
<table>
<tr><th>Benchmark</th><th>Placement</th><th>Chart</th><th>Segments</th></tr>
{rows_html}
</table>
</body></html>"""
    idx_path = os.path.join(OUT_DIR, "index.html")
    with open(idx_path, "w") as f:
        f.write(index)
    print(f"\nIndex page: {idx_path}")


if __name__ == "__main__":
    main()
