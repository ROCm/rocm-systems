###############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
###############################################################################

"""Inline-SVG ATT flame-graph renderer."""

from __future__ import annotations

import html as _html
import re
from typing import Any, Dict, List, Optional

__all__ = ["render_att_flamegraph"]

_CATEGORY_COLOR: Dict[str, str] = {
    "att_vmem_latency": "#ff8c00",
    "att_lds_conflict": "#e84040",
    "att_dependency_chain": "#cc44cc",
    "att_divergence": "#5599ee",
    "unknown": "#666677",
}

_CATEGORY_LABEL: Dict[str, str] = {
    "att_vmem_latency": "VMEM Latency",
    "att_lds_conflict": "LDS Conflict",
    "att_dependency_chain": "Dependency Chain",
    "att_divergence": "Branch Divergence",
    "unknown": "Other",
}

_CATEGORY_ORDER: List[str] = [
    "att_vmem_latency",
    "att_lds_conflict",
    "att_dependency_chain",
    "att_divergence",
    "unknown",
]

_WIDTH = 960
_LEFT_PAD = 168
_RIGHT_PAD = 12
_ROW_HEIGHT = 26
_ROW_GAP = 4
_HEADER_HEIGHT = 44

_SLUG_RE = re.compile(r"[^A-Za-z0-9_]+")


def _slug(name: str) -> str:
    s = _SLUG_RE.sub("_", (name or "").strip())
    s = s.strip("_")
    return s or "unknown"


def _esc(v: Any) -> str:
    return _html.escape(str(v), quote=True)


def render_att_flamegraph(
    att_data: Optional[Dict[str, Any]],
    theme: str = "dark",
) -> str:
    """Render an inline-SVG flame graph from the ``thread_trace`` payload."""
    if not att_data or not att_data.get("has_att_data"):
        return ""

    kernels = list(att_data.get("kernels") or [])
    if not kernels:
        return ""

    if theme == "light":
        bg = "#f7f7fb"
        fg = "#222233"
        grid = "#cccccc"
    else:
        bg = "#0f1020"
        fg = "#e6e6f0"
        grid = "#333344"

    rows: List[Dict[str, Any]] = []
    max_total = 0.0
    for k in kernels:
        name = str(k.get("name") or "unknown")
        category = str(k.get("stall_category") or "unknown")
        buckets: Dict[str, float] = {c: 0.0 for c in _CATEGORY_ORDER}
        total_w = float(k.get("total_weighted_stall") or 0)
        instrs = list(k.get("top_stalling_instructions") or [])
        if instrs:
            for instr in instrs:
                w = float(instr.get("weighted_stall") or 0)
                buckets[category if category in buckets else "unknown"] += w
        else:
            buckets[category if category in buckets else "unknown"] = total_w

        row_total = sum(buckets.values()) or 1.0
        max_total = max(max_total, row_total)
        rows.append(
            {
                "name": name,
                "slug": _slug(name),
                "buckets": buckets,
                "total": row_total,
                "avg_ratio": float(k.get("avg_stall_ratio") or 0) * 100.0,
            }
        )

    bar_width = _WIDTH - _LEFT_PAD - _RIGHT_PAD
    height = _HEADER_HEIGHT + len(rows) * (_ROW_HEIGHT + _ROW_GAP) + 20

    parts: List[str] = []
    parts.append(
        f'<div class="att-flame" data-tip="ATT stall flame graph - click a '
        f'rect to jump to its recommendation card.">'
    )
    parts.append(
        f'<svg class="att-flame-svg" xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {_WIDTH} {height}" '
        f'width="100%" height="{height}" '
        f'style="background:{bg};border-radius:8px" '
        f'role="img" aria-label="ATT stall flame graph">'
    )

    parts.append(
        f'<text x="{_LEFT_PAD}" y="24" fill="{fg}" '
        f'font-family="system-ui,sans-serif" font-size="13" '
        f'font-weight="600">Stall weight per kernel (wider = more stalled)</text>'
    )
    parts.append(
        f'<line x1="{_LEFT_PAD}" x2="{_WIDTH - _RIGHT_PAD}" '
        f'y1="{_HEADER_HEIGHT - 8}" y2="{_HEADER_HEIGHT - 8}" '
        f'stroke="{grid}" stroke-width="1"/>'
    )

    legend_x = _WIDTH - _RIGHT_PAD
    for cat in reversed(_CATEGORY_ORDER):
        label = _CATEGORY_LABEL[cat]
        color = _CATEGORY_COLOR[cat]
        legend_x -= 14
        parts.append(
            f'<rect x="{legend_x - 10}" y="10" width="10" height="10" '
            f'fill="{color}" rx="2"/>'
        )
        legend_x -= 6 + 6 * len(label)
        parts.append(
            f'<text x="{legend_x}" y="20" fill="{fg}" '
            f'font-family="system-ui,sans-serif" font-size="11">'
            f'{_esc(label)}</text>'
        )
        legend_x -= 10

    y = _HEADER_HEIGHT
    for row in rows:
        name = row["name"]
        slug = row["slug"]
        short = name if len(name) <= 22 else name[:19] + "\u2026"
        parts.append(
            f'<text x="8" y="{y + _ROW_HEIGHT - 8}" fill="{fg}" '
            f'font-family="ui-monospace,monospace" font-size="11" '
            f'data-tip="{_esc(name)}">{_esc(short)}</text>'
        )

        cx = _LEFT_PAD
        scale = bar_width / max_total if max_total > 0 else 0.0
        for cat in _CATEGORY_ORDER:
            w = row["buckets"].get(cat, 0.0)
            if w <= 0:
                continue
            rect_w = max(1.0, w * scale)
            color = _CATEGORY_COLOR[cat]
            label = _CATEGORY_LABEL[cat]
            pct_of_row = (w / row["total"]) * 100.0 if row["total"] else 0.0
            tip = _esc(
                f"<strong>{name}</strong>"
                f"{label}: {pct_of_row:.1f}% of this kernel's stalls"
                f"<em>avg stall: {row['avg_ratio']:.1f}%</em>"
            )
            inline = ""
            if rect_w >= 80:
                inline = (
                    f'<text x="{cx + 6}" y="{y + _ROW_HEIGHT - 9}" '
                    f'fill="#fff" font-family="system-ui,sans-serif" '
                    f'font-size="11" pointer-events="none">'
                    f'{_esc(label)} {pct_of_row:.0f}%</text>'
                )
            parts.append(
                f'<rect class="att-flame-rect" '
                f'x="{cx:.2f}" y="{y}" width="{rect_w:.2f}" '
                f'height="{_ROW_HEIGHT}" fill="{color}" '
                f'data-k="{_esc(slug)}" data-cat="{_esc(cat)}" '
                f'data-tip="{tip}" '
                f'onclick="var t=document.getElementById(\'rec-{_esc(slug)}\');'
                f'if(t){{t.scrollIntoView({{behavior:\'smooth\',block:\'center\'}});'
                f't.classList.add(\'rec-flash\');'
                f'setTimeout(function(){{t.classList.remove(\'rec-flash\');}},1400);}}" '
                f'style="cursor:pointer"/>'
            )
            parts.append(
                f'<title>{_esc(name)} - {_esc(label)}: {pct_of_row:.1f}% '
                f'(avg stall {row["avg_ratio"]:.1f}%)</title>'
            )
            if inline:
                parts.append(inline)
            cx += rect_w
        y += _ROW_HEIGHT + _ROW_GAP

    parts.append("</svg>")
    parts.append("</div>")
    return "".join(parts)
