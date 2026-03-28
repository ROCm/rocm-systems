###############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
###############################################################################

"""
WebView (HTML) formatting functions for ROCInsight analysis results.

HTML templates are loaded from the ``templates/`` subdirectory and filled
using :py:meth:`str.replace` with named placeholders.
"""

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from .json_fmt import _build_summary, _format_as_json, _tier0_to_dict

_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")


def _load_template(name: str) -> str:
    """Load an HTML template from the templates directory."""
    path = os.path.join(_TEMPLATE_DIR, name)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _format_as_webview(
    time_breakdown: Dict[str, Any],
    hotspots: List[Dict[str, Any]],
    memory_analysis: Dict[str, Dict[str, Any]],
    recommendations: List[Dict[str, Any]],
    hardware_counters: Optional[Dict[str, Any]] = None,
    database_path: str = "",
    interval_timeline=None,
    kernel_categories=None,
    short_kernels=None,
    att_analysis: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate a self-contained interactive HTML report.

    The file has no external dependencies -- all CSS and JS are inlined so it
    opens correctly from any local path or file-share without a web server.
    """
    import html as _html

    def _h(v: Any) -> str:
        """HTML-escape a value for safe text embedding."""
        return _html.escape(str(v), quote=True)

    def _fmt_ns(ns: Any) -> str:
        if ns is None:
            return "\u2014"
        ns = float(ns)
        if ns < 1_000:
            return f"{ns:.0f} ns"
        if ns < 1_000_000:
            return f"{ns / 1_000:.1f} \u00b5s"
        if ns < 1_000_000_000:
            return f"{ns / 1_000_000:.1f} ms"
        return f"{ns / 1_000_000_000:.2f} s"

    def _fmt_bytes(b: Any) -> str:
        if not b:
            return "\u2014"
        b = float(b)
        if b < 1_024:
            return f"{b:.0f} B"
        if b < 1_048_576:
            return f"{b / 1_024:.1f} KB"
        if b < 1_073_741_824:
            return f"{b / 1_048_576:.1f} MB"
        return f"{b / 1_073_741_824:.2f} GB"

    def _svg_gauge(pct: float, color: str, label: str, value_str: str) -> str:
        """SVG donut gauge -- semicircle (180 deg) style."""
        r = 36
        cx = cy = 44
        full = 3.14159265 * r  # half circumference (180 deg)
        dash = full * max(0.0, min(1.0, pct / 100.0))
        return (
            f'<div class="gauge-box">'
            f'<svg viewBox="0 0 88 50" width="130" height="74">'
            # track arc
            f'<path d="M {cx - r},{cy} A {r},{r} 0 0 1 {cx + r},{cy}"'
            f' fill="none" stroke="var(--bg3)" stroke-width="9" stroke-linecap="round"/>'
            # filled arc (clipped at cy so only top half shows)
            f'<path d="M {cx - r},{cy} A {r},{r} 0 0 1 {cx + r},{cy}"'
            f' fill="none" stroke="{_h(color)}" stroke-width="9" stroke-linecap="round"'
            f' stroke-dasharray="{dash:.2f} {full:.2f}"'
            f' stroke-dashoffset="0"/>'
            # value text
            f'<text x="{cx}" y="{cy - 4}" text-anchor="middle"'
            f' font-size="13" font-weight="700" fill="var(--text)">{_h(value_str)}</text>'
            f'<text x="{cx}" y="{cy + 10}" text-anchor="middle"'
            f' font-size="7.5" fill="var(--dim)">{_h(label.upper())}</text>'
            f"</svg>"
            f"</div>"
        )

    # -- derived values --
    breakdown = time_breakdown or {}
    hw = hardware_counters or {}
    has_counters = bool(hw.get("has_counters", False))
    total_ns = float(breakdown.get("total_runtime", 0))
    total_ms = total_ns / 1e6
    kernel_pct = float(breakdown.get("kernel_percent", 0))
    memcpy_pct = float(breakdown.get("memcpy_percent", 0))
    overhead_pct = float(breakdown.get("overhead_percent", 0))
    kernel_ms = breakdown.get("total_kernel_time", 0) / 1e6
    memcpy_ms = breakdown.get("total_memcpy_time", 0) / 1e6
    overhead_ms = max(0.0, total_ms * overhead_pct / 100.0)
    idle_pct = max(0.0, 100.0 - kernel_pct - memcpy_pct - overhead_pct)
    idle_ms = max(0.0, total_ms - kernel_ms - memcpy_ms - overhead_ms)
    tier = 2 if has_counters else 1
    analysis_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary = _build_summary(breakdown, hotspots or [], has_counters)
    bottleneck = summary.get("primary_bottleneck", "unknown")
    confidence = int(summary.get("confidence", 0) * 100)
    assessment = summary.get("overall_assessment", "")
    key_findings = summary.get("key_findings", [])
    metrics = hw.get("metrics", {}) or {}
    gpu_util = metrics.get("gpu_utilization_pct") or metrics.get(
        "gpu_utilization_percent"
    )
    avg_waves = metrics.get("avg_waves")
    max_waves = metrics.get("max_waves")

    BN_COLOR = {
        "compute": "#5599ee",
        "memory_transfer": "#ff8c00",
        "latency": "#cc44cc",
        "mixed": "#9999bb",
        "unknown": "#666677",
    }
    bn_color = BN_COLOR.get(bottleneck, "#888899")

    PRIORITY = {
        "HIGH": ("#e84040", "#2a0808"),
        "MEDIUM": ("#f08432", "#2a1600"),
        "LOW": ("#caa828", "#241e08"),
        "INFO": ("#4d8ef2", "#081428"),
    }
    PRIORITY_ICON = {
        "HIGH": "&#128308;",
        "MEDIUM": "&#128992;",
        "LOW": "&#128993;",
        "INFO": "&#8505;",
    }

    # -- recommendations HTML --
    recs_parts = []
    for ri, rec in enumerate(recommendations or []):
        p = rec.get("priority", "INFO")
        cat = rec.get("category", "")
        fg, bg_rec = PRIORITY.get(p, ("#888", "#1a1a2a"))
        picon = PRIORITY_ICON.get(p, "&#8505;")
        actions_li = "".join(f"<li>{_h(a)}</li>" for a in rec.get("actions", []))
        actions_html = f'<ol class="r-actions">{actions_li}</ol>' if actions_li else ""
        impact = rec.get("estimated_impact", "")
        impact_html = (
            f'<p class="r-impact">&#9889; Expected impact: {_h(impact)}</p>'
            if impact
            else ""
        )
        cmds_parts = []
        for ci, cmd in enumerate(rec.get("commands", [])):
            fc = cmd.get("full_command", "")
            tool = cmd.get("tool", "")
            desc = cmd.get("description", "")
            if not fc:
                continue
            cid = f"c{ri}_{ci}"
            cmds_parts.append(
                f'<div class="cmd-blk">'
                f'<span class="tool-tag">{_h(tool)}</span>'
                f'<span class="cmd-desc">{_h(desc)}</span>'
                f'<div class="cmd-row" id="{cid}">'
                f"<code>{_h(fc)}</code>"
                f'<button class="cp-btn" onclick="cpCmd(\'{cid}\')">Copy</button>'
                f"</div></div>"
            )
        cmds_html = "".join(cmds_parts)
        issue_txt = rec.get("issue", "")
        suggest = rec.get("suggestion", "")
        recs_parts.append(
            f'<div class="r-card" style="border-left-color:{fg}" data-p="{_h(p)}">'
            f'<div class="r-hdr" onclick="toggleR(this)">'
            f'<span class="r-priority-icon">{picon}</span>'
            f'<span class="r-badge" style="background:{fg};color:#fff">{_h(p)}</span>'
            f'<span class="r-cat">{_h(cat)}</span>'
            f'<span class="r-chev">&#9660;</span>'
            f"</div>"
            f'<div class="r-body">'
            f'<p class="r-issue"><strong>Issue:</strong> {_h(issue_txt)}</p>'
            f'<p class="r-suggest"><strong>What to do:</strong> {_h(suggest)}</p>'
            f"{actions_html}{impact_html}{cmds_html}"
            f"</div></div>"
        )
    recs_html = (
        "".join(recs_parts)
        or '<p class="dim">No recommendations \u2014 workload looks well-optimized.</p>'
    )

    # -- hotspots table --
    hotspot_rows = []
    for i, k in enumerate(hotspots or []):
        pct = float(k.get("percent_of_total", 0))
        bar = min(100.0, pct)
        name = k.get("name", "unknown")
        hot = ' class="hot-row"' if pct >= 20 else ""
        hotspot_rows.append(
            f"<tr{hot}>"
            f"<td>{i + 1}</td>"
            f'<td class="kname" title="{_h(name)}"><code>{_h(name)}</code></td>'
            f'<td data-v="{k.get("calls", 0)}">{int(k.get("calls", 0)):,}</td>'
            f'<td data-v="{k.get("total_duration", 0)}">{_fmt_ns(k.get("total_duration", 0))}</td>'
            f'<td data-v="{k.get("avg_duration", 0)}">{_fmt_ns(k.get("avg_duration", 0))}</td>'
            f'<td data-v="{k.get("min_duration", 0)}">{_fmt_ns(k.get("min_duration", 0))}</td>'
            f'<td data-v="{pct}">'
            f'<div class="pbar"><div class="pfill" style="width:{bar:.1f}%"></div>'
            f"<span>{pct:.1f}%</span></div>"
            f"</td></tr>"
        )
    hotspots_html = ""
    if hotspot_rows:
        hotspots_html = (
            '<section class="scard">'
            '<div class="shdr">'
            '<span class="shdr-icon">&#128293;</span>'
            "<h2>Top Kernel Hotspots</h2>"
            "</div>"
            '<div class="sbody"><div class="tbl-wrap">'
            '<table class="dtable sortable" id="hs-tbl">'
            "<thead><tr>"
            "<th data-tip='Rank by total execution time \u2014 1 is the hottest kernel.'>#</th>"
            "<th data-tip='Demangled GPU kernel function name dispatched to the GPU. Rows highlighted in red consume &gt;20% of total runtime.'>Kernel Name</th>"
            "<th data-tip='Number of times this kernel was dispatched. Very high call counts with low avg time suggest kernel launch overhead dominates useful work.'>Calls &#8645;</th>"
            "<th data-tip='Sum of all dispatch durations for this kernel \u2014 the primary metric for identifying hotspots. Longer total time = bigger optimization target.'>Total Time &#8645;</th>"
            "<th data-tip='Mean duration per single dispatch. Values below 10 &micro;s suggest kernel launch overhead may dominate the actual computation.'>Avg Time &#8645;</th>"
            "<th data-tip='Fastest observed single dispatch. Useful for spotting variance \u2014 a large gap between min and avg suggests irregular execution (cache effects, branch divergence).'>Min Time &#8645;</th>"
            "<th data-tip='Percentage of total profiling window time consumed by this kernel. Kernels above 20% are highlighted and are the highest-priority optimization targets.'>% Total &#8645;</th>"
            "</tr></thead>"
            "<tbody>" + "".join(hotspot_rows) + "</tbody>"
            "</table></div></div></section>"
        )

    # -- memory analysis table --
    _MEM_DIR_TIPS = {
        "Host-to-Device": (
            "<strong>Host-to-Device (H2D)</strong>"
            "CPU &rarr; GPU transfer over PCIe. Used to upload inputs, weights, or parameters before kernel execution. "
            "<em>PCIe 4.0 x16 peak: ~32 GB/s. Minimize by reusing GPU allocations across iterations.</em>"
        ),
        "Device-to-Host": (
            "<strong>Device-to-Host (D2H)</strong>"
            "GPU &rarr; CPU transfer over PCIe. Used to read results back after kernel execution. "
            "<em>Minimize these \u2014 prefer keeping results on GPU across multiple kernels. Use async memcpy to overlap with compute.</em>"
        ),
        "Device-to-Device": (
            "<strong>Device-to-Device (D2D)</strong>"
            "GPU &rarr; GPU on the same device, using HBM bandwidth directly (not PCIe). Very fast \u2014 can approach peak HBM bandwidth. "
            "<em>Use for in-GPU data reorganization. MI300X HBM peak: ~5.3 TB/s.</em>"
        ),
        "Peer-to-Peer": (
            "<strong>Peer-to-Peer (P2P)</strong>"
            "GPU &rarr; different GPU transfer. Speed depends on interconnect: Infinity Fabric is fast (&sim;900 GB/s on MI300X); PCIe is slower (~32 GB/s). "
            "<em>Enable peer access with hipDeviceEnablePeerAccess for direct transfers.</em>"
        ),
    }
    mem_rows = []
    for direction, s in (memory_analysis or {}).items():
        tb = s.get("total_bytes", 0)
        bw = s.get("bandwidth_bytes_per_sec", 0) / 1e9
        dir_tip = _MEM_DIR_TIPS.get(
            direction,
            f"<strong>{_h(direction)}</strong>Memory transfer direction between host and device.",
        )
        mem_rows.append(
            f"<tr>"
            f"<td data-tip='{dir_tip}'>{_h(direction)}</td>"
            f'<td>{int(s.get("count", 0)):,}</td>'
            f"<td>{_fmt_bytes(tb)}</td>"
            f'<td>{_fmt_ns(s.get("total_duration", 0))}</td>'
            f'<td>{_fmt_bytes(s.get("avg_bytes", 0))}</td>'
            f"<td>{bw:.2f} GB/s</td>"
            f"</tr>"
        )
    mem_html = ""
    if mem_rows:
        mem_html = (
            '<section class="scard">'
            '<div class="shdr">'
            '<span class="shdr-icon">&#128190;</span>'
            "<h2>Memory Transfer Analysis</h2>"
            "</div>"
            '<div class="sbody"><div class="tbl-wrap">'
            '<table class="dtable">'
            "<thead><tr>"
            "<th data-tip='Transfer direction. Hover each row to learn what each direction means.'>Direction</th>"
            "<th data-tip='Number of individual copy operations in this direction. Many small transfers are inefficient \u2014 batch them when possible.'>Count</th>"
            "<th data-tip='Total data volume transferred in this direction across all operations.'>Total Bytes</th>"
            "<th data-tip='Total wall-clock time spent on copies in this direction.'>Total Time</th>"
            "<th data-tip='Average bytes per copy operation. Transfers below 1 MB are typically inefficient due to PCIe transaction overhead \u2014 batch small transfers.'>Avg Size</th>"
            "<th data-tip='Achieved transfer bandwidth. PCIe 4.0 x16 theoretical peak is ~32 GB/s. Low bandwidth usually means many small transfers, not PCIe saturation.'>Bandwidth</th>"
            "</tr></thead>"
            "<tbody>" + "".join(mem_rows) + "</tbody>"
            "</table></div></div></section>"
        )

    # -- hardware counters --
    gauges_html = ""
    if gpu_util is not None:
        _gpu_u = float(gpu_util)
        gc = "#44dd66" if _gpu_u >= 70 else "#ff8800"
        hint = (
            '<p class="g-hint warn">&#9888; Low \u2014 increase parallelism</p>'
            if _gpu_u < 70
            else '<p class="g-hint ok">&#10003; Good utilization</p>'
        )
        _gpu_ok = _gpu_u >= 70
        _gpu_status = (
            '<span class="tok">Good \u2014 GPU is well-utilized.</span>'
            if _gpu_ok
            else '<span class="twarn">Low \u2014 reduce synchronization barriers, increase batch size, or launch larger kernels.</span>'
        )
        _gpu_tip = (
            f"<strong>GPU Utilization ({_gpu_u:.1f}%)</strong>"
            f"Percentage of GPU clock cycles where the hardware was actively processing work. "
            f"Derived from hardware counters: <code>GRBM_GUI_ACTIVE &divide; GRBM_COUNT</code>.<br>"
            f"<em>Target: &ge;70%. Below 70% means the GPU is frequently idle.</em><br>"
            f"{_gpu_status}"
        )
        gauges_html += (
            f"<div class=\"gauge-wrap\" data-tip='{_gpu_tip}'>"
            f'{_svg_gauge(_gpu_u, gc, "GPU Utilization", f"{_gpu_u:.1f}%")}'
            f"{hint}</div>"
        )
    if avg_waves is not None:
        _aw = float(avg_waves)
        wc = "#44dd66" if _aw >= 16 else "#ff8800"
        # Normalize waves to 0-100% assuming 64 waves/SIMD as 100%
        wpct = min(100.0, _aw / 64.0 * 100.0)
        whint = (
            '<p class="g-hint warn">&#9888; Low occupancy \u2014 check registers/LDS</p>'
            if _aw < 16
            else '<p class="g-hint ok">&#10003; Adequate occupancy</p>'
        )
        wave_str = f"{_aw:.0f}"
        if max_waves is not None:
            wave_str += f" / {float(max_waves):.0f}"
        _wave_ok = _aw >= 16
        _wave_status = (
            '<span class="tok">Good \u2014 adequate wavefront occupancy for latency hiding.</span>'
            if _wave_ok
            else '<span class="twarn">Low \u2014 reduce register usage or LDS allocation per wavefront to increase occupancy and hide memory latency.</span>'
        )
        _wave_tip = (
            f"<strong>Wave Occupancy (avg {_aw:.0f} waves)</strong>"
            f"Average number of wavefronts (64 threads each) simultaneously in-flight per compute unit. "
            f"Collected via the <code>SQ_WAVES</code> hardware counter. "
            f"Higher occupancy lets the GPU hide memory latency by switching to another wavefront while one waits for data.<br>"
            f"<em>Target: &ge;16 waves. Max practical: 64 waves/SIMD unit. "
            f"Low occupancy usually means each wavefront uses too many registers or too much LDS.</em><br>"
            f"{_wave_status}"
        )
        gauges_html += (
            f"<div class=\"gauge-wrap\" data-tip='{_wave_tip}'>"
            f'{_svg_gauge(wpct, wc, "Avg Waves", wave_str)}'
            f"{whint}</div>"
        )
    raw_counters = hw.get("counters", {}) or {}
    ctr_rows = "".join(
        f'<tr class="ctr-row" data-ctr="{_h(n)}"><td><code>{_h(n)}</code></td>'
        f'<td>{int(v.get("sample_count", 0)):,}</td>'
        f'<td>{float(v.get("avg_value", 0)):.2f}</td>'
        f'<td>{float(v.get("min_value", 0)):.2f}</td>'
        f'<td>{float(v.get("max_value", 0)):.2f}</td>'
        f'<td>{float(v.get("total_value", 0)):,.0f}</td></tr>'
        for n, v in raw_counters.items()
    )
    ctr_table = (
        (
            '<table class="dtable" style="margin-top:1rem">'
            "<thead><tr><th>Counter</th><th>Samples</th>"
            "<th>Avg</th><th>Min</th><th>Max</th><th>Total</th></tr></thead>"
            "<tbody>" + ctr_rows + "</tbody></table>"
        )
        if ctr_rows
        else ""
    )

    hw_inner = (
        f'<div class="gauges">{gauges_html}</div>{ctr_table}'
        if has_counters
        else (
            '<p class="dim">No hardware counter data \u2014 Tier 1 (trace-only) analysis.</p>'
            '<p class="hint" style="margin-top:.5rem">Collect counters with:</p>'
            '<div class="cmd-row" id="hw-hint">'
            "<code>rocprofv3 --pmc GRBM_COUNT GRBM_GUI_ACTIVE SQ_WAVES -- ./app</code>"
            '<button class="cp-btn" onclick="cpCmd(\'hw-hint\')">Copy</button>'
            "</div>"
        )
    )

    # -- key findings list --
    findings_li = "".join(f"<li>{_h(f)}</li>" for f in key_findings)
    findings_html = f'<ul class="findings">{findings_li}</ul>' if findings_li else ""

    # -- embed full JSON (sanitized for HTML context) --
    json_str = _format_as_json(
        time_breakdown,
        hotspots,
        memory_analysis,
        recommendations,
        hardware_counters,
        database_path,
    )
    json_embedded = json_str.replace("</script>", r"<\/script>").replace("<!--", r"<\!--")

    tier_label = "Hardware Counters (Tier 2)" if has_counters else "Trace Only (Tier 1)"
    bn_display = bottleneck.replace("_", " ").title()

    # -- Pre-computed tooltip strings (single-quote delimited in HTML attrs) --
    _TIP_KERNEL = (
        "<strong>Kernel Execution</strong>"
        "Time actively running GPU compute kernels. Higher is better \u2014 means more "
        "useful work is being done on the GPU silicon. "
        "<em>If this is low (&lt;40%), look for excessive GPU idle time or API launch overhead.</em>"
    )
    _TIP_MEMCPY = (
        "<strong>Memory Copies</strong>"
        "Time transferring data between CPU (host) and GPU (device) over the PCIe bus. "
        "High values (&gt;20%) indicate a PCIe bandwidth bottleneck. "
        "<em>Minimize by batching transfers, using pinned (page-locked) memory, "
        "or overlapping copies with kernel execution via async streams.</em>"
    )
    _TIP_OVERHEAD = (
        "<strong>API &amp; Launch Overhead</strong>"
        "Time in HIP/HSA runtime calls: kernel launch latency, "
        "synchronization barriers, and runtime bookkeeping. "
        "High values (&gt;15%) suggest too many small kernel dispatches or excessive "
        "CPU&ndash;GPU synchronization points. "
        "<em>Batch work into fewer larger kernels and minimize hipDeviceSynchronize calls.</em>"
    )
    _TIP_IDLE = (
        "<strong>GPU Idle</strong>"
        "Time when the GPU had no work to execute \u2014 pipeline bubbles between kernel launches. "
        "High idle time means the CPU is not submitting work fast enough, "
        "or there are long synchronization stalls waiting on host results. "
        "<em>Use asynchronous launches, CUDA/HIP streams, and reduce host processing "
        "between dispatches.</em>"
    )
    _BN_TIPS = {
        "compute": (
            "<strong>Compute Bottleneck</strong>"
            "GPU arithmetic units (VALU/MFMA) are the limiting factor. "
            "The workload is doing more FLOPs than the memory system can supply data for, "
            "meaning arithmetic throughput is the ceiling. "
            "<em>Optimize: use MFMA (matrix FMA) instructions, reduce register pressure, "
            "increase thread-level parallelism.</em>"
        ),
        "memory_transfer": (
            "<strong>Memory Transfer Bottleneck</strong>"
            "PCIe data transfers between CPU and GPU dominate execution time. "
            "The application is spending more time moving data than computing. "
            "<em>Optimize: keep data resident on GPU across multiple kernels, "
            "use pinned host memory, overlap transfers with computation via async streams.</em>"
        ),
        "memory_bandwidth": (
            "<strong>Memory Bandwidth Bottleneck</strong>"
            "HBM (High Bandwidth Memory) bandwidth is the limiting factor. "
            "Kernels are reading/writing more data than HBM can deliver per clock. "
            "<em>Optimize: improve data reuse via tiling, exploit L1/L2 cache locality, "
            "use LDS (shared memory) to reduce HBM traffic.</em>"
        ),
        "latency": (
            "<strong>Latency Bottleneck</strong>"
            "Many small, short-lived kernels where launch overhead dominates actual computation. "
            "GPU spends more time being launched than running. "
            "<em>Optimize: fuse multiple small kernels into one, increase work per dispatch, "
            "or use persistent kernel patterns.</em>"
        ),
        "mixed": (
            "<strong>Mixed Bottleneck</strong>"
            "Multiple performance limiters are present simultaneously. "
            "No single dominant bottleneck was identified. "
            "<em>Address the highest-priority recommendation first, re-profile, "
            "then iterate.</em>"
        ),
        "unknown": (
            "<strong>Bottleneck Unknown</strong>"
            "Analysis could not determine a clear primary bottleneck from available data. "
            "<em>Collect hardware counters for deeper analysis: "
            "rocprofv3 --pmc GRBM_COUNT GRBM_GUI_ACTIVE SQ_WAVES -- ./app</em>"
        ),
    }
    _tip_bn = _BN_TIPS.get(bottleneck, _BN_TIPS["unknown"])
    _tip_tier = (
        "<strong>Analysis Tier 2 \u2014 Hardware Counters</strong>"
        "Profiling data includes hardware performance counters collected via "
        "<code>rocprofv3 --pmc</code>. Enables GPU utilization, wave occupancy, "
        "and per-kernel counter breakdowns in addition to timing data."
        if has_counters
        else "<strong>Analysis Tier 1 \u2014 Trace Only</strong>"
        "Profiling data contains timing information only (no hardware counters). "
        "For deeper GPU-level insights, re-profile with: "
        "<em>rocprofv3 --pmc GRBM_COUNT GRBM_GUI_ACTIVE SQ_WAVES -- ./app</em>"
    )

    # -- Pre-compute badge / KPI status values --
    n_high = sum(1 for r in (recommendations or []) if r.get("priority") == "HIGH")
    n_medium = sum(1 for r in (recommendations or []) if r.get("priority") == "MEDIUM")
    n_low = sum(1 for r in (recommendations or []) if r.get("priority") == "LOW")
    n_info = sum(1 for r in (recommendations or []) if r.get("priority") == "INFO")

    # kernel utilization KPI health class
    if kernel_pct >= 60:
        _kpi_kernel_cls = "kpi-ok"
        _kpi_kernel_lbl = "Good"
    elif kernel_pct >= 30:
        _kpi_kernel_cls = "kpi-warn"
        _kpi_kernel_lbl = "Moderate"
    else:
        _kpi_kernel_cls = "kpi-crit"
        _kpi_kernel_lbl = "Low"

    _BN_ICON = {
        "compute": "&#128293;",
        "memory_transfer": "&#128230;",
        "memory_bandwidth": "&#128190;",
        "latency": "&#9889;",
        "mixed": "&#128256;",
        "unknown": "&#10067;",
    }
    _bn_icon = _BN_ICON.get(bottleneck, "&#10067;")

    _badge_parts = []
    if n_high:
        _badge_parts.append(
            f'<span class="hbadge hbadge-crit">&#9679; {n_high} Critical</span>'
        )
    if n_medium:
        _badge_parts.append(
            f'<span class="hbadge hbadge-warn">&#9679; {n_medium} Warning</span>'
        )
    if n_low:
        _badge_parts.append(f'<span class="hbadge hbadge-ok">&#9679; {n_low} Low</span>')
    if n_info:
        _badge_parts.append(
            f'<span class="hbadge hbadge-info">&#9679; {n_info} Info</span>'
        )
    header_badges_html = " ".join(_badge_parts)

    _recs_badge_html = ""
    if n_high:
        _recs_badge_html += (
            f'<span class="shdr-badge sbadge-crit">{n_high} Critical</span> '
        )
    if n_medium:
        _recs_badge_html += (
            f'<span class="shdr-badge sbadge-warn">{n_medium} Warning</span>'
        )

    _tier_icon = "&#128300;" if has_counters else "&#128225;"
    _tier_status_lbl = "HW Counters" if has_counters else "Trace Only"
    _hw_badge_html = (
        '<span class="shdr-badge sbadge-info">Tier 2</span>'
        if has_counters
        else '<span class="shdr-badge sbadge-info">Tier 1</span>'
    )

    _db_pill_html = ""
    if database_path:
        _db_label = database_path[-45:] if len(database_path) > 45 else database_path
        _db_pill_html = (
            f'<div class="hpill">'
            f'<span class="hpill-label">DB:</span>'
            f'<span class="hpill-value" title="{_h(database_path)}">{_h(_db_label)}</span>'
            f"</div>"
        )

    # Load template and fill placeholders
    template = _load_template("webview.html")

    replacements = {
        "%%TITLE%%": f"ROCInsight AI Analysis &#8212; {_h(database_path or 'GPU Performance Report')}",
        "%%HEADER_BADGES%%": header_badges_html,
        "%%TOTAL_MS%%": f"{total_ms:,.2f}",
        "%%NUM_HOTSPOTS%%": str(len(hotspots or [])),
        "%%TIER_LABEL%%": _h(tier_label),
        "%%ANALYSIS_DATE%%": analysis_date,
        "%%DB_PILL%%": _db_pill_html,
        "%%TIER%%": str(tier),
        "%%ASSESSMENT%%": _h(assessment),
        "%%TIP_BN%%": _tip_bn,
        "%%BN_ICON%%": _bn_icon,
        "%%BN_DISPLAY%%": _h(bn_display),
        "%%BN_COLOR%%": bn_color,
        "%%CONFIDENCE%%": str(confidence),
        "%%TOTAL_MS_VALUE%%": f"{total_ms:,.2f}",
        "%%KPI_KERNEL_CLS%%": _kpi_kernel_cls,
        "%%TIP_KERNEL%%": _TIP_KERNEL,
        "%%KPI_KERNEL_LBL%%": _kpi_kernel_lbl,
        "%%KERNEL_PCT%%": f"{kernel_pct:.1f}",
        "%%KERNEL_MS%%": f"{kernel_ms:,.2f}",
        "%%TIP_TIER%%": _tip_tier,
        "%%TIER_ICON%%": _tier_icon,
        "%%TIER_STATUS_LBL%%": _tier_status_lbl,
        "%%HAS_COUNTERS_SUB%%": "Hardware counters available" if has_counters else "Trace-level only",
        "%%FINDINGS_HTML%%": findings_html,
        "%%KERNEL_PCT_2F%%": f"{kernel_pct:.2f}",
        "%%MEMCPY_PCT_2F%%": f"{memcpy_pct:.2f}",
        "%%OVERHEAD_PCT_2F%%": f"{overhead_pct:.2f}",
        "%%IDLE_PCT_2F%%": f"{idle_pct:.2f}",
        "%%KERNEL_PCT_1F%%": f"{kernel_pct:.1f}",
        "%%MEMCPY_PCT_1F%%": f"{memcpy_pct:.1f}",
        "%%OVERHEAD_PCT_1F%%": f"{overhead_pct:.1f}",
        "%%IDLE_PCT_1F%%": f"{idle_pct:.1f}",
        "%%MEMCPY_MS%%": f"{memcpy_ms:,.2f}",
        "%%OVERHEAD_MS%%": f"{overhead_ms:,.2f}",
        "%%IDLE_MS%%": f"{idle_ms:,.2f}",
        "%%TIP_MEMCPY%%": _TIP_MEMCPY,
        "%%TIP_OVERHEAD%%": _TIP_OVERHEAD,
        "%%TIP_IDLE%%": _TIP_IDLE,
        "%%HOTSPOTS_HTML%%": hotspots_html,
        "%%MEM_HTML%%": mem_html,
        "%%HW_BADGE%%": _hw_badge_html,
        "%%HW_INNER%%": hw_inner,
        "%%RECS_BADGE%%": _recs_badge_html,
        "%%RECS_HTML%%": recs_html,
        "%%JSON_EMBEDDED%%": json_embedded,
    }

    html = template
    for placeholder, value in replacements.items():
        html = html.replace(placeholder, value)

    # --- ATT (Tier 3) section ---
    if att_analysis and att_analysis.get("has_att_data"):
        _CAT_LABELS = {
            "att_vmem_latency": "VMEM Latency",
            "att_lds_conflict": "LDS Conflict",
            "att_dependency_chain": "Dependency Chain",
            "att_divergence": "Branch Divergence",
        }
        _CAT_TIPS = {
            "att_vmem_latency": (
                "<strong>VMEM Latency</strong>Instructions stalling on global memory (HBM/L2/L1) round-trip. "
                "The GPU issued a load/store but had to wait many cycles for data to arrive. "
                "<em>Fix: improve data locality, coalesce access patterns, prefetch into LDS.</em>"
            ),
            "att_lds_conflict": (
                "<strong>LDS Conflict</strong>Instructions stalling on LDS (shared memory) bank conflicts. "
                "Multiple threads in a wavefront accessed the same LDS bank simultaneously. "
                "<em>Fix: pad LDS arrays by 1 element per row, or reorganize access patterns.</em>"
            ),
            "att_dependency_chain": (
                "<strong>Dependency Chain</strong>S_WAITCNT and similar instructions stalling while waiting for "
                "in-flight loads or LDS ops to complete before proceeding. "
                "<em>Fix: reorder instructions to hide latency, reduce synchronization barriers.</em>"
            ),
            "att_divergence": (
                "<strong>Branch Divergence</strong>Branch instructions causing threads in a wavefront to take "
                "different paths, serializing execution. "
                "<em>Fix: eliminate data-dependent branches, use predication, or restructure conditionals.</em>"
            ),
        }
        att_kernels = att_analysis.get("kernels", [])
        att_summary = att_analysis.get("summary", {})
        n_traced = att_summary.get("kernel_count", len(att_kernels))
        n_high_att = att_summary.get("high_stall_kernels", 0)
        high_color = "#e84040" if n_high_att > 0 else "#44dd66"
        kpi_class = "kpi-warn" if n_high_att > 0 else "kpi-ok"

        att_kpi = (
            f'<div class="kpi-grid" style="margin-bottom:1.4rem">'
            f"<div class=\"kpi kpi-info\" data-tip='<strong>Kernels Traced</strong>Number of unique kernels with ATT instruction-level data captured.'>"
            f'<div class="kpi-head"><span class="kpi-icon">&#129535;</span></div>'
            f'<div class="kpi-label">Kernels Traced</div>'
            f'<div class="kpi-value">{n_traced}</div></div>'
            f"<div class=\"kpi {kpi_class}\" data-tip='<strong>High-Stall Kernels</strong>Kernels where the top instruction stall ratio &ge; 60% and hitcount &ge; 6400 threads. These are the primary ATT optimization targets.'>"
            f'<div class="kpi-head"><span class="kpi-icon">&#9888;</span></div>'
            f'<div class="kpi-label">High-Stall Kernels</div>'
            f'<div class="kpi-value" style="color:{high_color}">{n_high_att}</div></div>'
            f"</div>"
        )

        att_rows = []
        for k in att_kernels:
            kname = k.get("name", "unknown")
            avg_ratio = float(k.get("avg_stall_ratio", 0)) * 100
            category = k.get("stall_category", "unknown")
            cat_label = _CAT_LABELS.get(category, category)
            cat_tip = _CAT_TIPS.get(category, f"<strong>{_h(cat_label)}</strong>")
            cat_tip_safe = cat_tip.replace("'", "&#39;")
            top = (k.get("top_stalling_instructions") or [{}])[0]
            _dash = "\u2014"
            top_instr = _h(top.get("pc_offset", _dash))
            top_ratio = float(top.get("stall_ratio", 0)) * 100
            weighted = int(k.get("total_weighted_stall", 0))
            # color by avg stall ratio
            ratio_color = (
                "#e84040"
                if avg_ratio >= 60
                else ("#ff8800" if avg_ratio >= 40 else "#44dd66")
            )
            top_color = (
                "#e84040"
                if top_ratio >= 60
                else ("#ff8800" if top_ratio >= 40 else "#44dd66")
            )
            bar_w = min(100, avg_ratio)

            # expand sub-instructions
            sub_rows = ""
            for instr in (k.get("top_stalling_instructions") or [])[:5]:
                i_ratio = float(instr.get("stall_ratio", 0)) * 100
                i_color = (
                    "#e84040"
                    if i_ratio >= 60
                    else ("#ff8800" if i_ratio >= 40 else "#44dd66")
                )
                sub_rows += (
                    f'<tr style="background:rgba(0,0,0,.18);font-size:.82rem">'
                    f'<td colspan="2" style="padding-left:2.5rem;font-family:monospace;color:#b0b8d8">'
                    f'{_h(instr.get("pc_offset", _dash))}</td>'
                    f'<td style="color:#888">hitcount: {int(instr.get("hitcount", 0)):,}</td>'
                    f'<td style="color:{i_color};text-align:center">{i_ratio:.0f}%</td>'
                    f'<td colspan="2" style="color:#888">wt: {int(instr.get("weighted_stall", 0)):,}</td>'
                    f"</tr>"
                )

            att_rows.append(
                f"<tr>"
                f'<td><code style="font-size:.88rem">{_h(kname)}</code></td>'
                f"<td>"
                f'<div style="display:flex;align-items:center;gap:.5rem">'
                f'<div style="width:80px;height:8px;background:#1a1a2e;border-radius:4px;overflow:hidden">'
                f'<div style="width:{bar_w:.1f}%;height:100%;background:{ratio_color};border-radius:4px"></div>'
                f"</div>"
                f'<span style="color:{ratio_color};font-weight:600">{avg_ratio:.1f}%</span>'
                f"</div></td>"
                f"<td><span data-tip='{cat_tip_safe}' style=\"cursor:help;border-bottom:1px dotted #555\">"
                f"{_h(cat_label)}</span></td>"
                f'<td><code style="font-size:.82rem;color:#b0b8d8">{top_instr}</code></td>'
                f'<td style="color:{top_color};font-weight:600;text-align:center">{top_ratio:.0f}%</td>'
                f'<td style="color:#888;font-size:.85rem">{weighted:,}</td>'
                f"</tr>" + sub_rows
            )

        att_table = (
            '<div class="tbl-wrap">'
            '<table class="dtable" style="margin-top:.5rem">'
            "<thead><tr>"
            "<th data-tip='Demangled GPU kernel name from the ATT CSV comment row.'>Kernel</th>"
            "<th data-tip='Average stall ratio across all instructions: stall cycles / total latency. &ge;60% = HIGH, &ge;40% = MEDIUM.'>Avg Stall %</th>"
            "<th data-tip='Primary stall category from the highest-stall instruction in this kernel.'>Category</th>"
            "<th data-tip='ISA mnemonic of the instruction with the highest weighted stall (stall &times; hitcount).'>Top Stalling Instr</th>"
            "<th data-tip='Stall ratio of the top stalling instruction.'>Top Stall %</th>"
            "<th data-tip='Total weighted stall = sum of (stall_cycles &times; hitcount) across all instructions. Primary sorting metric.'>Weighted Stall</th>"
            "</tr></thead>"
            "<tbody>" + "".join(att_rows) + "</tbody>"
            "</table></div>"
        )

        att_section = (
            '\n<section class="scard">'
            '\n<div class="shdr">'
            '\n<span class="shdr-icon">&#129535;</span>'
            "\n<h2>Thread Trace Analysis</h2>"
            '\n<span class="shdr-badge sbadge-info">Tier 3 &mdash; ATT</span>'
            "\n</div>"
            '\n<div class="sbody">'
            f"\n{att_kpi}"
            f"\n{att_table}"
            '\n<p class="dim" style="margin-top:1rem;font-size:.82rem">'
            "Sub-rows show the top 5 stalling instructions per kernel (indented). "
            "Weighted stall = stall_cycles &times; hitcount &mdash; the primary ranking metric."
            "</p>"
            "\n</div>\n</section>"
        )
        html = html.replace("<!-- ATT_SECTION_PLACEHOLDER -->", att_section)
    else:
        html = html.replace("<!-- ATT_SECTION_PLACEHOLDER -->", "")

    # --- Kernel category breakdown section (TraceLens) ---
    if kernel_categories:
        cat_rows_html = ""
        for cat in kernel_categories:
            avg_us = cat["avg_duration_ns"] / 1_000
            pct = cat["pct_of_kernel_time"]
            bar_w = max(2, int(pct * 2))  # scale to max 200px
            cat_rows_html += (
                f'<tr><td>{cat["category"]}</td>'
                f'<td>{cat["count"]}</td>'
                f'<td><div style="display:inline-block;height:12px;width:{bar_w}px;'
                f'background:#e01a22;border-radius:2px;vertical-align:middle"></div>'
                f" {pct:.1f}%</td>"
                f"<td>{avg_us:.1f}&#956;s</td></tr>"
            )
        cat_section = (
            '\n<section class="scard">'
            '\n<div class="shdr">'
            '\n<span class="shdr-icon">&#128202;</span>'
            "\n<h2>Kernel Category Breakdown</h2>"
            '\n<span class="shdr-badge sbadge-info">Tier 1 &mdash; TraceLens</span>'
            "\n</div>"
            '\n<div class="sbody">'
            '\n<table class="tbl">'
            "\n<thead><tr>"
            "<th>Category</th><th>Kernels</th>"
            "<th>% of Kernel Time</th><th>Avg Duration</th>"
            "</tr></thead>"
            "\n<tbody>" + cat_rows_html + "</tbody>"
            "\n</table>"
            "\n</div>\n</section>"
        )
        html = html.replace("<!-- CAT_SECTION_PLACEHOLDER -->", cat_section)
    else:
        html = html.replace("<!-- CAT_SECTION_PLACEHOLDER -->", "")

    return html


def _format_tier0_webview(tier0_result: Any) -> str:
    """Generate a self-contained AMD-themed HTML Tier 0 report (identical design system as Tier 1/2)."""
    import html as _html
    import json as _json

    def _h(v: Any) -> str:
        return _html.escape(str(v), quote=True)

    SEV_FG = {"high": "#e84040", "medium": "#f08432", "low": "#caa828", "info": "#4d8ef2"}
    SEV_BG = {
        "high": "rgba(232,64,64,.13)",
        "medium": "rgba(240,132,50,.13)",
        "low": "rgba(202,168,40,.13)",
        "info": "rgba(77,142,242,.13)",
    }
    PRIORITY = {
        "HIGH": ("#e84040", "#2a0808"),
        "MEDIUM": ("#f08432", "#2a1600"),
        "LOW": ("#caa828", "#241e08"),
        "INFO": ("#4d8ef2", "#081428"),
    }
    PRIORITY_ICON = {
        "HIGH": "&#128308;",
        "MEDIUM": "&#128992;",
        "LOW": "&#128993;",
        "INFO": "&#8505;",
    }

    analysis_date = tier0_result.analysis_timestamp
    src_dir = str(tier0_result.source_dir)
    src_display = src_dir[-45:] if len(src_dir) > 45 else src_dir

    # -- Counts --
    recs = tier0_result.recommendations or []
    n_high = sum(1 for r in recs if r.get("priority") == "HIGH")
    n_medium = sum(1 for r in recs if r.get("priority") == "MEDIUM")
    n_low = sum(1 for r in recs if r.get("priority") == "LOW")
    n_info = sum(1 for r in recs if r.get("priority") == "INFO")

    _badge_parts = []
    if n_high:
        _badge_parts.append(
            f'<span class="hbadge hbadge-crit">&#9679; {n_high} Critical</span>'
        )
    if n_medium:
        _badge_parts.append(
            f'<span class="hbadge hbadge-warn">&#9679; {n_medium} Warning</span>'
        )
    if n_low:
        _badge_parts.append(f'<span class="hbadge hbadge-ok">&#9679; {n_low} Low</span>')
    if n_info:
        _badge_parts.append(
            f'<span class="hbadge hbadge-info">&#9679; {n_info} Info</span>'
        )
    header_badges_html = " ".join(_badge_parts)

    _recs_badge_html = ""
    if n_high:
        _recs_badge_html += (
            f'<span class="shdr-badge sbadge-crit">{n_high} Critical</span> '
        )
    if n_medium:
        _recs_badge_html += (
            f'<span class="shdr-badge sbadge-warn">{n_medium} Warning</span>'
        )

    # -- Recommendations HTML (same .r-card format as Tier 1/2) --
    recs_parts = []
    for ri, rec in enumerate(recs):
        p = rec.get("priority", "INFO")
        cat = rec.get("category", "")
        fg, _ = PRIORITY.get(p, ("#888", "#1a1a2a"))
        picon = PRIORITY_ICON.get(p, "&#8505;")
        actions_li = "".join(f"<li>{_h(a)}</li>" for a in rec.get("actions", []))
        actions_html = f'<ol class="r-actions">{actions_li}</ol>' if actions_li else ""
        impact = rec.get("estimated_impact", "")
        impact_html = (
            f'<p class="r-impact">&#9889; Expected impact: {_h(impact)}</p>'
            if impact
            else ""
        )
        cmds_parts = []
        for ci, cmd in enumerate(rec.get("commands", [])):
            fc = cmd.get("full_command", "")
            tool = cmd.get("tool", "")
            desc = cmd.get("description", "")
            if not fc:
                continue
            cid = f"c{ri}_{ci}"
            cmds_parts.append(
                f'<div class="cmd-blk">'
                f'<span class="tool-tag">{_h(tool)}</span>'
                f'<span class="cmd-desc">{_h(desc)}</span>'
                f'<div class="cmd-row" id="{cid}">'
                f"<code>{_h(fc)}</code>"
                f'<button class="cp-btn" onclick="cpCmd(\'{cid}\')">Copy</button>'
                f"</div></div>"
            )
        cmds_html = "".join(cmds_parts)
        issue_txt = rec.get("issue", "")
        suggest = rec.get("suggestion", "")
        recs_parts.append(
            f'<div class="r-card" style="border-left-color:{fg}" data-p="{_h(p)}">'
            f'<div class="r-hdr" onclick="toggleR(this)">'
            f'<span class="r-priority-icon">{picon}</span>'
            f'<span class="r-badge" style="background:{fg};color:#fff">{_h(p)}</span>'
            f'<span class="r-cat">{_h(cat)}</span>'
            f'<span class="r-chev">&#9660;</span>'
            f"</div>"
            f'<div class="r-body">'
            f'<p class="r-issue"><strong>Issue:</strong> {_h(issue_txt)}</p>'
            f'<p class="r-suggest"><strong>What to do:</strong> {_h(suggest)}</p>'
            f"{actions_html}{impact_html}{cmds_html}"
            f"</div></div>"
        )
    recs_html = (
        "".join(recs_parts)
        or '<p class="dim">No recommendations \u2014 workload looks well-optimized.</p>'
    )

    # -- Kernels table --
    kernel_rows = []
    for i, k in enumerate(tier0_result.detected_kernels[:50]):
        fname = _h(k.get("file", "").split("/")[-1])
        kernel_rows.append(
            f"<tr>"
            f"<td>{i + 1}</td>"
            f'<td class="kname" title="{_h(k.get("name", ""))}"><code>{_h(k.get("name", ""))}</code></td>'
            f'<td>{_h(k.get("launch_type", ""))}</td>'
            f"<td>{fname}</td>"
            f'<td data-v="{k.get("line", 0)}">{_h(str(k.get("line", "")))}</td>'
            f"</tr>"
        )
    if kernel_rows:
        kernels_section = (
            '<section class="scard">'
            '<div class="shdr">'
            '<span class="shdr-icon">&#128187;</span>'
            "<h2>Detected GPU Kernels</h2>"
            f'<span class="shdr-badge sbadge-info">{tier0_result.kernel_count} found</span>'
            "</div>"
            '<div class="sbody"><div class="tbl-wrap">'
            '<table class="dtable sortable">'
            "<thead><tr>"
            "<th data-tip='Rank by order found in source.'>#</th>"
            "<th data-tip='GPU kernel function name detected in source code. For HIP/CUDA: __global__ functions.'>Kernel Name</th>"
            "<th data-tip='How the kernel is launched: __global__ for HIP/CUDA, kernel for OpenCL.'>Launch Type</th>"
            "<th data-tip='Source file where the kernel is defined (basename only).'>File</th>"
            "<th data-tip='Line number of the kernel definition in the source file.'>Line &#8645;</th>"
            "</tr></thead>"
            "<tbody>" + "".join(kernel_rows) + "</tbody>"
            "</table></div></div></section>"
        )
    else:
        kernels_section = (
            '<section class="scard">'
            '<div class="shdr"><span class="shdr-icon">&#128187;</span>'
            "<h2>Detected GPU Kernels</h2></div>"
            '<div class="sbody"><p class="dim">No GPU kernels detected in the source directory.</p></div>'
            "</section>"
        )

    # -- Patterns table --
    pattern_rows = []
    for pat in tier0_result.detected_patterns:
        sev = pat.get("severity", "info").lower()
        sfg = SEV_FG.get(sev, "#6b7280")
        sbg = SEV_BG.get(sev, "rgba(107,114,128,.13)")
        pattern_rows.append(
            f"<tr>"
            f'<td><span style="display:inline-block;padding:.14em .55em;border-radius:4px;'
            f"font-size:.69rem;font-weight:800;letter-spacing:.06em;"
            f'background:{sbg};color:{sfg}">{_h(sev.upper())}</span></td>'
            f'<td>{_h(pat.get("category", ""))}</td>'
            f'<td>{_h(pat.get("description", ""))}</td>'
            f'<td data-v="{pat.get("count", 0)}">{pat.get("count", 0)}</td>'
            f"</tr>"
        )
    if pattern_rows:
        patterns_section = (
            '<section class="scard">'
            '<div class="shdr">'
            '<span class="shdr-icon">&#128202;</span>'
            "<h2>Detected Performance Patterns</h2>"
            f'<span class="shdr-badge sbadge-warn">{len(tier0_result.detected_patterns)} found</span>'
            "</div>"
            '<div class="sbody"><div class="tbl-wrap">'
            '<table class="dtable sortable">'
            "<thead><tr>"
            "<th data-tip='Issue severity. HIGH = likely significant performance impact. MEDIUM = moderate. LOW = minor.'>Severity</th>"
            "<th data-tip='Category of the anti-pattern detected in source code (memory, compute, synchronization, etc.).'>Category</th>"
            "<th data-tip='Description of the specific pattern found and its likely performance impact.'>Description</th>"
            "<th data-tip='Number of occurrences of this pattern across all scanned source files.'>Count &#8645;</th>"
            "</tr></thead>"
            "<tbody>" + "".join(pattern_rows) + "</tbody>"
            "</table></div></div></section>"
        )
    else:
        patterns_section = ""

    # -- Risk areas --
    risk_li = "".join(f"<li>{_h(r)}</li>" for r in tier0_result.risk_areas)
    risk_section = ""
    if risk_li:
        risk_section = (
            '<section class="scard">'
            '<div class="shdr">'
            '<span class="shdr-icon">&#9888;</span>'
            "<h2>Risk Areas</h2>"
            f'<span class="shdr-badge sbadge-warn">{len(tier0_result.risk_areas)}</span>'
            "</div>"
            '<div class="sbody">'
            f'<ul class="findings">{risk_li}</ul>'
            "</div></section>"
        )

    # -- Suggested counters --
    ctr_badges = " ".join(
        f'<code style="background:rgba(77,142,242,.15);color:#4d8ef2;'
        f"padding:.14em .55em;border-radius:4px;font-size:.83rem;margin:.18rem .1rem;"
        f'display:inline-block">{_h(c)}</code>'
        for c in tier0_result.suggested_counters
    )
    counters_section = ""
    if tier0_result.suggested_counters:
        collect_cmd = (
            "rocprofv3 --sys-trace --pmc "
            + " ".join(tier0_result.suggested_counters)
            + " -- ./your_app"
        )
        counters_section = (
            '<section class="scard">'
            '<div class="shdr">'
            '<span class="shdr-icon">&#128300;</span>'
            "<h2>Suggested Hardware Counters</h2>"
            f'<span class="shdr-badge sbadge-info">{len(tier0_result.suggested_counters)} counters</span>'
            "</div>"
            '<div class="sbody">'
            '<p style="margin-bottom:.85rem;color:var(--sub);font-size:.9rem">'
            "Collect these counters to enable Tier 2 (hardware-level) analysis:</p>"
            f'<p style="margin-bottom:1rem;line-height:1.9">{ctr_badges}</p>'
            f'<div class="cmd-row" id="cmd-ctr">'
            f"<code>{_h(collect_cmd)}</code>"
            f'<button class="cp-btn" onclick="cpCmd(\'cmd-ctr\')">Copy</button>'
            "</div>"
            "</div></section>"
        )

    # -- Start Here --
    start_here_section = ""
    if tier0_result.suggested_first_command:
        fc = tier0_result.suggested_first_command
        start_here_section = (
            '<section class="scard">'
            '<div class="shdr">'
            '<span class="shdr-icon">&#9654;</span>'
            "<h2>Start Here</h2>"
            '<span class="shdr-badge sbadge-info">Recommended First Step</span>'
            "</div>"
            '<div class="sbody">'
            '<p style="margin-bottom:.85rem;color:var(--sub);font-size:.9rem">'
            "Run this command to collect profiling data for Tier 1/2 analysis:</p>"
            f'<div class="cmd-row" id="cmd-start">'
            f"<code>{_h(fc)}</code>"
            f'<button class="cp-btn" onclick="cpCmd(\'cmd-start\')">Copy</button>'
            "</div>"
            "</div></section>"
        )

    # -- LLM section --
    llm_section = ""
    if tier0_result.llm_explanation:
        llm_section = (
            '<section class="scard">'
            '<div class="shdr">'
            '<span class="shdr-icon">&#129302;</span>'
            "<h2>AI-Enhanced Insights</h2>"
            '<span class="shdr-badge sbadge-info">LLM</span>'
            "</div>"
            '<div class="sbody">'
            f'<pre style="white-space:pre-wrap;line-height:1.6;'
            f'color:var(--sub);font-size:.9rem">{_h(tier0_result.llm_explanation)}</pre>'
            "</div></section>"
        )

    # -- KPI grid --
    n_risks = len(tier0_result.risk_areas)
    risk_kpi_cls = "kpi-warn" if n_risks > 0 else "kpi-ok"
    risk_kpi_label = "Needs Attention" if n_risks > 0 else "None Found"
    model_upper = _h(tier0_result.programming_model.upper())
    assessment_txt = (
        f"Static source analysis of {tier0_result.files_scanned} file(s) found "
        f"{tier0_result.kernel_count} GPU kernel(s). "
        f"Programming model: {tier0_result.programming_model}. "
        "See recommendations below for the suggested profiling workflow."
    )
    n_patterns = len(tier0_result.detected_patterns)

    payload = _json.dumps(_tier0_to_dict(tier0_result))
    payload = payload.replace("</script>", r"<\/script>").replace("<!--", r"<\!--")

    # Load template and fill placeholders
    template = _load_template("tier0_webview.html")

    replacements = {
        "%%HEADER_BADGES%%": header_badges_html,
        "%%SRC_DIR%%": _h(src_dir),
        "%%SRC_DISPLAY%%": _h(src_display),
        "%%KERNEL_COUNT%%": str(tier0_result.kernel_count),
        "%%ANALYSIS_DATE%%": _h(analysis_date),
        "%%PROG_MODEL%%": _h(tier0_result.programming_model),
        "%%ASSESSMENT%%": _h(assessment_txt),
        "%%MODEL_UPPER%%": model_upper,
        "%%FILES_SCANNED%%": str(tier0_result.files_scanned),
        "%%FILES_SKIPPED%%": str(tier0_result.files_skipped),
        "%%N_PATTERNS%%": str(n_patterns),
        "%%N_RISKS%%": str(n_risks),
        "%%RISK_KPI_CLS%%": risk_kpi_cls,
        "%%RISK_KPI_LABEL%%": risk_kpi_label,
        "%%RISK_KPI_SUB%%": "requires profiling to confirm" if n_risks > 0 else "no obvious risk areas",
        "%%RECS_BADGE%%": _recs_badge_html,
        "%%RECS_HTML%%": recs_html,
        "%%KERNELS_SECTION%%": kernels_section,
        "%%PATTERNS_SECTION%%": patterns_section,
        "%%RISK_SECTION%%": risk_section,
        "%%COUNTERS_SECTION%%": counters_section,
        "%%START_HERE_SECTION%%": start_here_section,
        "%%LLM_SECTION%%": llm_section,
        "%%PAYLOAD%%": payload,
    }

    html = template
    for placeholder, value in replacements.items():
        html = html.replace(placeholder, value)

    return html
