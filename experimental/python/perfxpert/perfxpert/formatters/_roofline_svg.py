###############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
###############################################################################

"""Pure-Python SVG renderer for the Live Roofline webview section.

Contract
--------
Input: the dict returned by ``perfxpert.tools.roofline.plot_points`` (see
its module docstring for the exact shape).
Output: a single ``<section class="scard">`` string containing:

- Inline ``<svg>`` element drawn in log-log space (AI on X, achieved
  FLOPs/s on Y).
- Horizontal peak-compute ceilings per dtype observed (dominant dtype
  full opacity, others dimmed).
- HBM-bandwidth diagonal (slope 1 in log-log space).
- One dot per top-K kernel carrying ``data-tip`` for the existing
  tooltip engine and ``data-k`` so the inline click handler can
  ``scrollIntoView`` the matching ``id="rec-<sanitized>"`` card.
- Legend with ``.leg + .dot`` entries (uses the existing webview CSS
  rules).
- <=30 LOC of inline JS for wheel-zoom (same ``<script>`` tag as the
  click handler).

No external libs. No template literals. No ``${}``. All ``<>`` / ``&`` /
``"`` characters in kernel names are escaped before they reach the
output.
"""

from __future__ import annotations

import html as _html
import math
import re
from typing import Any, Dict, List, Optional


__all__ = ["render_roofline_svg", "sanitize_rec_anchor"]


_W = 880
_H = 420
_PAD_L = 74
_PAD_R = 24
_PAD_T = 18
_PAD_B = 58

_X_MIN = 1e-2
_X_MAX = 1e3
_Y_MIN = 1e10
_Y_MAX = 5e14

_DTYPE_COLOR = {
    "fp64": "#9866cc",
    "fp32": "#4d8ef2",
    "fp16": "#3acc66",
    "bf16": "#28bca8",
    "fp8":  "#f08432",
    "int8": "#caa828",
}

_REGIME_COLOR = {
    "compute":  "#e84040",
    "memory":   "#f08432",
    "balanced": "#4d8ef2",
    "mixed":    "#9866cc",
    "unknown":  "#7f7f7f",
}


def _kernel_basename(name: str) -> str:
    """Extract demangled kernel basename — strip templates + args.

    ``heavy_valu_kernel(float const*, float*, int, int)`` ->
    ``heavy_valu_kernel``.
    """
    s = str(name or "")
    i = s.find("(")
    if i >= 0:
        s = s[:i]
    # Split namespace BEFORE stripping templates so ``ns::foo<int>::bar``
    # reduces to ``bar``.
    s = s.rsplit("::", 1)[-1]
    j = s.find("<")
    if j >= 0:
        s = s[:j]
    return s.strip() or str(name or "unknown")


def sanitize_rec_anchor(name: str) -> str:
    """Return the slug used by ``id="rec-<slug>"`` on a rec card."""
    s = _kernel_basename(name)
    s = re.sub(r"[^A-Za-z0-9_]+", "_", s).strip("_")
    return s or "unknown"


def _log_x(ai: float) -> float:
    if ai <= 0:
        ai = _X_MIN
    ai = max(_X_MIN, min(_X_MAX, ai))
    lx = (math.log10(ai) - math.log10(_X_MIN)) / (
        math.log10(_X_MAX) - math.log10(_X_MIN)
    )
    return _PAD_L + lx * (_W - _PAD_L - _PAD_R)


def _log_y(rate: float) -> float:
    if rate <= 0:
        rate = _Y_MIN
    rate = max(_Y_MIN, min(_Y_MAX, rate))
    ly = (math.log10(rate) - math.log10(_Y_MIN)) / (
        math.log10(_Y_MAX) - math.log10(_Y_MIN)
    )
    return _H - _PAD_B - ly * (_H - _PAD_T - _PAD_B)


def _fmt_tflops(rate: float) -> str:
    if rate >= 1e12:
        return f"{rate / 1e12:.1f} TF/s"
    if rate >= 1e9:
        return f"{rate / 1e9:.1f} GF/s"
    return f"{rate:.0f} F/s"


def _fmt_bw(bw: float) -> str:
    if bw >= 1e12:
        return f"{bw / 1e12:.1f} TB/s"
    if bw >= 1e9:
        return f"{bw / 1e9:.1f} GB/s"
    return f"{bw:.0f} B/s"


def render_roofline_svg(
    rf: Optional[Dict[str, Any]],
    theme: str = "dark",  # noqa: ARG001 — theme hook kept for parity with spec
) -> str:
    """Render a standalone roofline ``.scard`` section.

    Args:
        rf: The payload from ``roofline.plot_points``. When ``None`` or
            missing kernels, returns ``""`` so the webview slot stays
            empty.
        theme: Unused today — kept in the signature so downstream CSS
            theming hooks can be added later without another schema bump.

    Returns:
        HTML ``<section class="scard">...</section>`` string (may be
        ``""``).
    """
    if not rf:
        return ""
    kernels: List[Dict[str, Any]] = list(rf.get("kernels") or [])
    if not kernels:
        return ""

    arch = str(rf.get("arch", "gfx942"))
    peaks: Dict[str, float] = dict(rf.get("arch_peaks") or {})
    hbm_bps = float(rf.get("hbm_bandwidth_bytes_per_s") or 0.0)
    dominant_dtype = str(rf.get("dtype", "fp32")).lower()
    ridge = rf.get("ridge_point") or {}
    ridge_ai = float(ridge.get("ai", 0.0))
    ridge_rate = float(ridge.get("flops_per_s", 0.0))

    axes_parts: List[str] = []
    for exp in range(-2, 4):
        v = 10.0 ** exp
        gx = _log_x(v)
        axes_parts.append(
            f'<line x1="{gx:.1f}" y1="{_PAD_T}" x2="{gx:.1f}" y2="{_H - _PAD_B}" '
            f'stroke="#2a2a44" stroke-width="1" stroke-dasharray="2,3"/>'
        )
        label = f"{v:g}" if v >= 1 else f"{v}"
        axes_parts.append(
            f'<text x="{gx:.1f}" y="{_H - _PAD_B + 16}" '
            f'fill="#6868a0" font-size="11" text-anchor="middle">{label}</text>'
        )
    for exp in range(10, 15):
        v = 10.0 ** exp
        gy = _log_y(v)
        axes_parts.append(
            f'<line x1="{_PAD_L}" y1="{gy:.1f}" x2="{_W - _PAD_R}" y2="{gy:.1f}" '
            f'stroke="#2a2a44" stroke-width="1" stroke-dasharray="2,3"/>'
        )
        label = _fmt_tflops(v)
        axes_parts.append(
            f'<text x="{_PAD_L - 8}" y="{gy + 4:.1f}" '
            f'fill="#6868a0" font-size="11" text-anchor="end">{label}</text>'
        )
    axes_parts.append(
        f'<text x="{(_W + _PAD_L - _PAD_R) / 2:.1f}" y="{_H - 14}" '
        f'fill="#a8aace" font-size="12" text-anchor="middle" font-weight="600">'
        f'Arithmetic Intensity (FLOPs / Byte)</text>'
    )
    axes_parts.append(
        f'<text x="14" y="{(_H + _PAD_T - _PAD_B) / 2:.1f}" '
        f'fill="#a8aace" font-size="12" text-anchor="middle" font-weight="600" '
        f'transform="rotate(-90 14 {(_H + _PAD_T - _PAD_B) / 2:.1f})">'
        f'Achieved Performance (FLOPs/s)</text>'
    )

    diag_parts: List[str] = []
    if hbm_bps > 0:
        p1x = _log_x(_X_MIN)
        p1y = _log_y(max(_Y_MIN, _X_MIN * hbm_bps))
        if ridge_ai > 0 and ridge_rate > 0:
            p2x = _log_x(ridge_ai)
            p2y = _log_y(ridge_rate)
        else:
            p2x = _log_x(_X_MAX)
            p2y = _log_y(min(_Y_MAX, _X_MAX * hbm_bps))
        diag_parts.append(
            f'<line x1="{p1x:.1f}" y1="{p1y:.1f}" x2="{p2x:.1f}" y2="{p2y:.1f}" '
            f'stroke="#f08432" stroke-width="2" stroke-linecap="round"/>'
        )
        mid_x = (p1x + p2x) / 2
        mid_y = (p1y + p2y) / 2
        diag_parts.append(
            f'<text x="{mid_x:.1f}" y="{mid_y - 6:.1f}" fill="#f08432" '
            f'font-size="11" font-weight="600" transform="rotate(-22 {mid_x:.1f} {mid_y:.1f})">'
            f'HBM {_fmt_bw(hbm_bps)}</text>'
        )

    ceiling_parts: List[str] = []
    for dt in ("fp64", "fp32", "fp16", "bf16", "fp8", "int8"):
        peak = float(peaks.get(dt) or 0)
        if peak <= 0:
            continue
        gy = _log_y(peak)
        if gy < _PAD_T or gy > _H - _PAD_B:
            continue
        color = _DTYPE_COLOR.get(dt, "#4d8ef2")
        opacity = "1.0" if dt == dominant_dtype else "0.25"
        ceiling_parts.append(
            f'<line x1="{_PAD_L}" y1="{gy:.1f}" x2="{_W - _PAD_R}" y2="{gy:.1f}" '
            f'stroke="{color}" stroke-width="2" stroke-opacity="{opacity}"/>'
        )
        ceiling_parts.append(
            f'<text x="{_W - _PAD_R - 4:.1f}" y="{gy - 4:.1f}" '
            f'fill="{color}" font-size="10" font-weight="600" text-anchor="end" '
            f'fill-opacity="{opacity}">{dt.upper()} {_fmt_tflops(peak)}</text>'
        )

    dot_parts: List[str] = []
    for k in kernels:
        name = str(k.get("name", "unknown"))
        anchor = sanitize_rec_anchor(name)
        ai = float(k.get("ai", 0))
        rate = float(k.get("achieved_flops_per_s", 0))
        cx = _log_x(ai)
        cy = _log_y(rate)
        regime = str(k.get("bottleneck_class", "balanced"))
        color = _REGIME_COLOR.get(regime, "#4d8ef2")
        dur_pct = float(k.get("duration_pct", 0))
        conf = str(k.get("confidence", "high"))
        dtype = str(k.get("fp_type", "fp32"))
        tip = (
            f"<strong>{_html.escape(name)}</strong>"
            f"AI: {ai:.2f} FLOPs/Byte &middot; "
            f"Rate: {_fmt_tflops(rate)} &middot; "
            f"dtype: {dtype.upper()}<em>{regime} &middot; "
            f"{dur_pct:.1f}% of kernel time &middot; confidence: {conf}</em>"
        )
        tip_safe = tip.replace("'", "&#39;")
        dot_parts.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6" fill="{color}" '
            f'stroke="#0d0d14" stroke-width="1.5" data-k="{anchor}" '
            f"data-tip='{tip_safe}' "
            f"onclick=\"var e=document.getElementById('rec-'+this.dataset.k);"
            f"if(e){{e.scrollIntoView({{behavior:'smooth',block:'center'}});}}\" "
            f'style="cursor:pointer"/>'
        )

    ridge_annot = ""
    if ridge_ai > 0 and ridge_rate > 0:
        rx = _log_x(ridge_ai)
        ry = _log_y(ridge_rate)
        ridge_annot = (
            f'<circle cx="{rx:.1f}" cy="{ry:.1f}" r="5" fill="none" '
            f'stroke="#e0e3f2" stroke-width="2" stroke-dasharray="3,2"/>'
            f'<text x="{rx + 10:.1f}" y="{ry - 8:.1f}" fill="#e0e3f2" '
            f'font-size="11" font-weight="700">'
            f'{_html.escape(arch)} &middot; {_fmt_tflops(ridge_rate)} &middot; '
            f'{_fmt_bw(hbm_bps)} &middot; ridge @ {ridge_ai:.1f} FLOPs/B'
            f'</text>'
        )

    legend_parts: List[str] = []
    for regime in ("compute", "memory", "balanced"):
        c = _REGIME_COLOR[regime]
        legend_parts.append(
            f'<div class="leg"><div class="dot" style="background:{c}"></div>'
            f'{regime.title()}-bound</div>'
        )
    for dt in ("fp64", "fp32", "fp16", "bf16", "fp8", "int8"):
        if float(peaks.get(dt) or 0) <= 0:
            continue
        c = _DTYPE_COLOR[dt]
        opacity = "1" if dt == dominant_dtype else "0.35"
        legend_parts.append(
            f'<div class="leg" style="opacity:{opacity}">'
            f'<div class="dot" style="background:{c};border-radius:2px"></div>'
            f'{dt.upper()} peak</div>'
        )
    legend_parts.append(
        '<div class="leg"><div class="dot" style="background:#f08432;'
        'border-radius:2px"></div>HBM BW</div>'
    )
    legend_html = '<div class="legend" style="margin-top:.8rem">' + "".join(legend_parts) + "</div>"

    svg = (
        f'<svg id="rf-svg" viewBox="0 0 {_W} {_H}" width="100%" '
        f'preserveAspectRatio="xMidYMid meet" '
        f'style="display:block;background:var(--bg);border-radius:var(--r-sm)">'
        + "".join(axes_parts)
        + "".join(ceiling_parts)
        + "".join(diag_parts)
        + ridge_annot
        + '<g id="rf-dots">' + "".join(dot_parts) + "</g>"
        + "</svg>"
    )

    js = (
        "<script>(function(){"
        "var s=document.getElementById('rf-svg');if(!s){return;}"
        "var vb=s.viewBox.baseVal;var z=1.0;var cx=vb.x+vb.width/2;var cy=vb.y+vb.height/2;"
        "s.addEventListener('wheel',function(e){"
        "e.preventDefault();var f=e.deltaY<0?0.9:1.1;z*=f;if(z<0.25){z=0.25;}if(z>6){z=6;}"
        f"var nw={_W}*z;var nh={_H}*z;"
        "vb.x=cx-nw/2;vb.y=cy-nh/2;vb.width=nw;vb.height=nh;"
        "},{passive:false});"
        "s.addEventListener('dblclick',function(){"
        f"z=1.0;vb.x=0;vb.y=0;vb.width={_W};vb.height={_H};"
        "});"
        "})();</script>"
    )

    return (
        '<section class="scard">'
        '<div class="shdr">'
        '<span class="shdr-icon">&#128200;</span>'
        '<h2>Live Roofline</h2>'
        f'<span class="shdr-badge sbadge-info">{_html.escape(arch)} &middot; {dominant_dtype.upper()}</span>'
        '</div>'
        '<div class="sbody">'
        '<p class="hint" style="margin-bottom:.8rem">'
        'Log-log roofline. Click any dot to jump to its recommendation. '
        'Scroll-wheel to zoom, double-click to reset.'
        '</p>'
        + svg
        + legend_html
        + js
        + '</div></section>'
    )
