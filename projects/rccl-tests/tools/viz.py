#!/usr/bin/env python3
"""Interactive plotly visualisations for rccl-tests tidy records.

Consumes the tidy per-point records produced by :mod:`roctx_records` (either
freshly built from run directories or loaded from an exported JSON) and renders
self-contained interactive HTML.

Phase 1 provides the line/band view:
  * median line per (comparison factor, placement)
  * a shaded variance band: IQR (p25-p75), min/max, or mean +/- std
  * metric selectable: busbw / algbw / time / eff_busbw
  * facet grid (one subplot per collective, and per dtype when dtype is not the
    colour dimension); colour = the comparison factor (library / dtype / machine)

The HTML is written with plotly.js embedded so it opens offline on any machine
(handy after copying an exported records JSON to a laptop).
"""

import os
from collections import OrderedDict, defaultdict

import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Colourblind-friendly qualitative palette.
_PALETTE = [
    "#377EB8", "#E41A1C", "#4DAF4A", "#FF7F00",
    "#984EA3", "#A65628", "#F781BF", "#00CED1",
]

# Placement -> line dash + label.
_PLACE = OrderedDict([
    (1, dict(dash="solid", name="in-place")),
    (0, dict(dash="dash", name="out-of-place")),
])

_METRICS = {
    "busbw": dict(label="busbw (GB/s)", log=True),
    "algbw": dict(label="algbw (GB/s)", log=True),
    "time": dict(label="time (us)", log=True),
    "eff_busbw": dict(label="eff busbw (GB/s)", log=True),
}

_DIMS = ("label", "dtype", "machine", "collective")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _hex_to_rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _distinct(records, dim):
    """Ordered distinct values of a dimension across records."""
    return list(OrderedDict.fromkeys(r.get(dim) for r in records))


def _auto_color_by(records):
    """Pick the comparison factor: first dimension with >1 distinct value."""
    for dim in ("label", "machine", "dtype"):
        if len(_distinct(records, dim)) > 1:
            return dim
    return "dtype"


def _bus_factor(rec):
    """busbw/algbw ratio for this point, or None if unavailable."""
    algbw, busbw = rec.get("algbw"), rec.get("busbw")
    if algbw and busbw and algbw > 0:
        return busbw / algbw
    return None


def _value_at_time(rec, t_us, metric):
    """Map a per-iteration time (us) to the chosen metric for this record."""
    if t_us is None or t_us <= 0:
        return None
    if metric == "time":
        return t_us
    algbw = rec["size"] / (t_us * 1000.0)  # bytes/ns == GB/s
    if metric == "algbw":
        return algbw
    factor = _bus_factor(rec)
    if metric in ("busbw", "eff_busbw"):
        return algbw * factor if factor is not None else algbw
    return None


def _central(rec, metric):
    if metric == "time":
        return rec.get("median_us")
    if metric == "eff_busbw":
        return rec.get("eff_busbw") or rec.get("busbw")
    return rec.get(metric)


def _effective(rec, metric):
    """Metric value including per-call non-collective overhead (e.g. DDA copy).

    Returns None when there is no overhead for this record (nothing to show)."""
    ovhd = rec.get("overhead_us") or 0.0
    if ovhd <= 0:
        return None
    med = rec.get("median_us")
    if metric == "time":
        return (med + ovhd) if med is not None else None
    if med is None or (med + ovhd) <= 0:
        return None
    algbw = rec["size"] / ((med + ovhd) * 1000.0)
    if metric == "algbw":
        return algbw
    factor = _bus_factor(rec)
    return algbw * factor if factor is not None else algbw


def _band_bounds(rec, metric, band):
    """Return (lo, hi) for the requested band in *metric* space, or (None, None).

    Bands are computed in time space then mapped through the metric, so that a
    slower (higher-time) tail correctly becomes a lower-bandwidth bound.
    """
    if band == "none":
        return None, None
    if band == "iqr":
        t_lo, t_hi = rec.get("p75_us"), rec.get("p25_us")  # slow->low bw, fast->high bw
    elif band == "minmax":
        t_lo, t_hi = rec.get("max_us"), rec.get("min_us")
    elif band == "std":
        m, s = rec.get("mean_us"), rec.get("std_us")
        if m is None or s is None:
            return None, None
        t_lo, t_hi = m + s, m - s
    else:
        return None, None

    if metric == "time":
        # keep natural lo<=hi ordering in time space
        lo = rec.get("p25_us") if band == "iqr" else (
            rec.get("min_us") if band == "minmax" else (rec.get("mean_us") - rec.get("std_us")))
        hi = rec.get("p75_us") if band == "iqr" else (
            rec.get("max_us") if band == "minmax" else (rec.get("mean_us") + rec.get("std_us")))
        return lo, hi
    return _value_at_time(rec, t_lo, metric), _value_at_time(rec, t_hi, metric)


# ---------------------------------------------------------------------------
# Figure construction
# ---------------------------------------------------------------------------

def _facet_keys(records, color_by):
    """Subplot key = (collective[, dtype]) where dtype is faceted unless it is
    the colour dimension."""
    facet_dims = ["collective"]
    if color_by != "dtype" and len(_distinct(records, "dtype")) > 1:
        facet_dims.append("dtype")
    keys = list(OrderedDict.fromkeys(
        tuple(r.get(d) for d in facet_dims) for r in records
    ))
    return facet_dims, keys


def build_line_figure(records, metric="busbw", band="iqr",
                      color_by=None, placements=(1, 0), ncols=None,
                      title=None, overhead=False):
    """Build a plotly Figure of *metric* vs message size with variance bands.

    When *overhead* is set, series that carry per-call non-collective overhead
    (e.g. a DDA staging copy) also get a dashed "effective" line and a shaded
    gap between the collective-only and effective values, so the cost of that
    overhead is visible.
    """
    records = [r for r in records if r.get(metric if metric != "eff_busbw" else "busbw")
               is not None or _central(r, metric) is not None]
    if not records:
        raise ValueError("no plottable records for metric=%r" % metric)

    if color_by is None:
        color_by = _auto_color_by(records)
    minfo = _METRICS[metric]

    facet_dims, facet_keys = _facet_keys(records, color_by)
    color_vals = _distinct(records, color_by)
    color_map = {v: _PALETTE[i % len(_PALETTE)] for i, v in enumerate(color_vals)}

    n = len(facet_keys)
    ncols = ncols or min(n, 3)
    nrows = (n + ncols - 1) // ncols
    subplot_titles = [" / ".join(str(x) for x in k) for k in facet_keys]
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=subplot_titles,
                        horizontal_spacing=0.06, vertical_spacing=0.10)

    legend_seen = set()

    for f_idx, fkey in enumerate(facet_keys):
        row, col = divmod(f_idx, ncols)
        row += 1
        col += 1
        sub = [r for r in records
               if tuple(r.get(d) for d in facet_dims) == fkey]

        for cval in color_vals:
            color = color_map[cval]
            crecs = [r for r in sub if r.get(color_by) == cval]
            if not crecs:
                continue
            for place in placements:
                precs = sorted((r for r in crecs if r["in_place"] == place),
                               key=lambda r: r["size"])
                pts = [(r["size"], _central(r, metric)) for r in precs
                       if _central(r, metric) is not None]
                if not pts:
                    continue
                place_name = _PLACE[place]["name"]
                dash = _PLACE[place]["dash"]
                legname = f"{cval} · {place_name}"
                show_legend = legname not in legend_seen
                legend_seen.add(legname)

                # Variance band (added first so the line draws on top).
                if band != "none":
                    band_pts = [(r["size"], *_band_bounds(r, metric, band))
                                for r in precs]
                    band_pts = [(s, lo, hi) for (s, lo, hi) in band_pts
                                if lo is not None and hi is not None]
                    if band_pts:
                        xs = [p[0] for p in band_pts]
                        los = [p[1] for p in band_pts]
                        his = [p[2] for p in band_pts]
                        fig.add_trace(go.Scatter(
                            x=xs + xs[::-1], y=his + los[::-1],
                            fill="toself", fillcolor=_hex_to_rgba(color, 0.15),
                            line=dict(width=0), hoverinfo="skip",
                            showlegend=False, legendgroup=legname,
                        ), row=row, col=col)

                # Overhead: shade the gap between collective-only and effective,
                # and draw a dashed effective line.
                if overhead:
                    eff_pts = [(r["size"], _central(r, metric), _effective(r, metric))
                               for r in precs]
                    eff_pts = [(s, c, e) for (s, c, e) in eff_pts
                               if c is not None and e is not None]
                    if eff_pts:
                        xs = [p[0] for p in eff_pts]
                        cvals = [p[1] for p in eff_pts]
                        evals = [p[2] for p in eff_pts]
                        fig.add_trace(go.Scatter(
                            x=xs + xs[::-1], y=cvals + evals[::-1],
                            fill="toself", fillcolor="rgba(120,120,120,0.28)",
                            line=dict(width=0), hoverinfo="skip",
                            showlegend=False, legendgroup=legname,
                        ), row=row, col=col)
                        fig.add_trace(go.Scatter(
                            x=xs, y=evals, mode="lines",
                            line=dict(color=color, dash="dot", width=1.2),
                            name=legname, legendgroup=legname, showlegend=False,
                            hovertemplate=(
                                f"<b>{cval}</b> ({place_name}) effective<br>"
                                "size=%{x:,} B<br>"
                                f"{minfo['label']}=%{{y:.3g}} (incl. overhead)"
                                "<extra></extra>"
                            ),
                        ), row=row, col=col)

                # Median / central line.
                sizes = [p[0] for p in pts]
                vals = [p[1] for p in pts]
                fig.add_trace(go.Scatter(
                    x=sizes, y=vals, mode="lines+markers",
                    line=dict(color=color, dash=dash, width=1.8),
                    marker=dict(size=5),
                    name=legname, legendgroup=legname, showlegend=show_legend,
                    hovertemplate=(
                        f"<b>{cval}</b> ({place_name})<br>"
                        "size=%{x:,} B<br>"
                        f"{minfo['label']}=%{{y:.3g}}<extra></extra>"
                    ),
                ), row=row, col=col)

        fig.update_xaxes(type="log", title_text="message size (bytes)",
                         row=row, col=col)
        fig.update_yaxes(type="log" if minfo["log"] else "linear",
                         title_text=minfo["label"], row=row, col=col)

    band_desc = {"iqr": "IQR (p25–p75)", "minmax": "min–max",
                 "std": "mean ± std", "none": "no"}[band]
    ttl = title or (f"{metric} vs size  —  colour: {color_by}, "
                    f"dash: placement, band: {band_desc}"
                    + ("  (dotted = effective incl. overhead)" if overhead else ""))
    fig.update_layout(
        title=ttl,
        template="plotly_white",
        height=max(360 * nrows, 420),
        width=max(520 * ncols, 640),
        legend=dict(title=color_by, groupclick="toggleitem"),
        margin=dict(t=90, l=70, r=30, b=60),
    )
    return fig


def _fmt_size(nbytes):
    for unit, div in (("G", 1024 ** 3), ("M", 1024 ** 2), ("K", 1024)):
        if nbytes >= div and nbytes % div == 0:
            return f"{nbytes // div}{unit}"
    return str(nbytes)


def build_box_figure(records, metric="time", kind="box", color_by=None,
                     placements=(1, 0), ncols=None, title=None):
    """Box/violin distribution per message size (all sizes on one frame).

    One subplot per (collective[, dtype]); within each, one box (or violin) per
    message size, grouped by the colour dimension (default: placement).  Needs
    the retained per-iteration ``samples_us`` (i.e. the profiled source); rows
    with a single sample degenerate to a flat line.
    """
    records = [r for r in records if r.get("samples_us")]
    if not records:
        raise ValueError("no records with per-iteration samples (use --source profiled)")

    minfo = _METRICS[metric]

    # Colour dimension. 'placement' is synthetic; the rest are record fields.
    if color_by is None:
        color_by = "placement" if len({r["in_place"] for r in records}) > 1 else \
            _auto_color_by(records)

    facet_dims, facet_keys = _facet_keys(records, color_by if color_by != "placement" else "dtype")
    if color_by == "placement":
        group_vals = [p for p in placements
                      if any(r["in_place"] == p for r in records)]
        group_name = {p: _PLACE[p]["name"] for p in group_vals}
        def group_of(r):
            return r["in_place"]
    else:
        group_vals = _distinct(records, color_by)
        group_name = {v: str(v) for v in group_vals}
        def group_of(r):
            return r.get(color_by)
    color_map = {v: _PALETTE[i % len(_PALETTE)] for i, v in enumerate(group_vals)}

    n = len(facet_keys)
    ncols = ncols or min(n, 2)
    nrows = (n + ncols - 1) // ncols
    subplot_titles = [" / ".join(str(x) for x in k) for k in facet_keys]
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=subplot_titles,
                        horizontal_spacing=0.08, vertical_spacing=0.12)

    legend_seen = set()
    # Global size ordering so category axis is consistent across facets.
    all_sizes = sorted({r["size"] for r in records})
    size_labels = [_fmt_size(s) for s in all_sizes]

    for f_idx, fkey in enumerate(facet_keys):
        row, col = divmod(f_idx, ncols)
        row += 1
        col += 1
        sub = [r for r in records
               if tuple(r.get(d) for d in facet_dims) == fkey]

        for gval in group_vals:
            color = color_map[gval]
            grecs = [r for r in sub if group_of(r) == gval]
            if not grecs:
                continue
            xs, ys = [], []
            for r in grecs:
                lbl = _fmt_size(r["size"])
                for s in r["samples_us"]:
                    v = _value_at_time(r, s, metric)
                    if v is None:
                        continue
                    xs.append(lbl)
                    ys.append(v)
            if not ys:
                continue
            legname = group_name[gval]
            show_legend = legname not in legend_seen
            legend_seen.add(legname)
            common = dict(x=xs, y=ys, name=legname, legendgroup=legname,
                          showlegend=show_legend, marker_color=color,
                          offsetgroup=legname)
            if kind == "violin":
                fig.add_trace(go.Violin(points=False, spanmode="hard",
                                        line_width=1, **common),
                              row=row, col=col)
            else:
                fig.add_trace(go.Box(boxpoints="outliers", boxmean=True,
                                     line_width=1, **common),
                              row=row, col=col)

        fig.update_xaxes(categoryorder="array", categoryarray=size_labels,
                         title_text="message size", tickangle=45, row=row, col=col)
        fig.update_yaxes(type="log" if minfo["log"] else "linear",
                         title_text=minfo["label"], row=row, col=col)

    boxmode = "group"
    ttl = title or (f"{metric} distribution per size  —  "
                    f"{kind}, grouped by {color_by}")
    fig.update_layout(
        title=ttl, template="plotly_white",
        boxmode=boxmode, violinmode=boxmode,
        height=max(380 * nrows, 440), width=max(620 * ncols, 720),
        legend=dict(title=color_by, groupclick="toggleitem"),
        margin=dict(t=90, l=70, r=30, b=70),
    )
    return fig


def write_html(fig, path, embed_js=True):
    """Write *fig* to a self-contained (offline) HTML file by default."""
    fig.write_html(path, include_plotlyjs=(True if embed_js else "cdn"),
                   full_html=True)
    return path
