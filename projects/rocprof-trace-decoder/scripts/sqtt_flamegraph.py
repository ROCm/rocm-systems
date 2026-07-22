#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Generate a flamegraph from SQTT auto-instrumentation traces.

Reads shaderdata JSON files (s_ttracedata markers) and .sqtt_funcmap sections
from code objects, then outputs folded stacks suitable for flamegraph.pl or
speedscope.

Usage:
    # Basic -- point at the base output directory, everything is auto-detected
    python3 sqtt_flamegraph.py build/test/ --demangle

    # Explicit code objects
    python3 sqtt_flamegraph.py build/test/ -c build/test/*/code_object_id_2.out

    # Filter to a single wave
    python3 sqtt_flamegraph.py build/test/ --demangle --cu 0 --simd 0 --wave 0

    # Save folded stacks to file (also printed to stdout)
    python3 sqtt_flamegraph.py build/test/ > stacks.folded

    # Output speedscope JSON
    python3 sqtt_flamegraph.py build/test/ --format speedscope -o trace.speedscope.json

    # Open interactive flamegraph in the browser
    python3 sqtt_flamegraph.py build/test/ --demangle --show

Auto-discovery:
    The base directory is searched recursively for:
      - ui_*/shaderdata_*.json  (trace data)
      - *code_object_id*.out    (code objects with .sqtt_funcmap sections)

    Each ui_* directory is processed as an independent time domain (SQTT
    time resets between collections). Folded stack counts are merged after
    per-directory processing.

Output:
    Folded stacks are always printed to stdout. An SVG flamegraph is
    always written to disk (default: sqtt_flamegraph.svg). Use --show
    to open the SVG in a browser. Use -o to specify an output path
    for SVG or speedscope JSON.
"""

import argparse
import html
import json
import os
import sys
import webbrowser
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from sqtt_data import (
    FuncMap,
    ShaderRecord,
    WaveSpan,
    decode_marker,
    demangle_name,
    discover_base_dir,
    find_wave_span_at,
    load_funcmaps,
    load_occupancy,
    load_shaderdata,
    build_kernel_to_co,
    make_funcmap_resolver,
    merge_folded,
    merge_funcmaps,
    preprocess_records,
)


# ---------------------------------------------------------------------------
# Flamegraph generation
# ---------------------------------------------------------------------------

# A unique wave identity
WaveKey = tuple[int, int, int, int, int]  # (se, cu, simd, wave_id, instance)


def build_stacks(
    records: list[ShaderRecord],
    per_co: dict[int, FuncMap],
    dispatches: dict[str, str],
    wave_spans: dict[tuple, list[WaveSpan]],
    kernel_to_co: dict[str, int],
    do_demangle: bool = False,
    filter_cu: Optional[int] = None,
    filter_simd: Optional[int] = None,
    filter_wave: Optional[int] = None,
) -> dict[str, int]:
    """
    Process shaderdata records into folded stacks weighted by time.

    For each wave, maintain a call stack. On ENTER markers, push.
    On EXIT markers, pop. Accumulate time spent in each unique stack.

    Uses occupancy data to prepend code_object_id and kernel name frames.

    Returns a dict mapping "frame1;frame2;...;frameN" -> (total_cycles, exec_count).
    """
    # Merged funcmap for fallback resolution.
    merged = merge_funcmaps(per_co.values())

    # Build sorted launch times per (SE, CU, SIMD, wave_slot) from occupancy.
    slot_launches: dict[tuple, list[int]] = defaultdict(list)
    for wk, spans in wave_spans.items():
        for span in spans:
            slot_launches[wk].append(span.launch_time)
    for wk in slot_launches:
        slot_launches[wk].sort()

    def _wave_instance(shader_engine: int, cu: int, simd: int, wave_id: int, time: int) -> int:
        """Find which wave instance (launch index) a record belongs to."""
        launches = slot_launches.get((shader_engine, cu, simd, wave_id), [])
        if not launches:
            return 0
        lo, hi = 0, len(launches) - 1
        result = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if launches[mid] <= time:
                result = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return result

    # Group records by (SE, CU, SIMD, wave_id, instance)
    wave_records: dict[tuple, list[ShaderRecord]] = defaultdict(list)
    for rec in records:
        if rec.flags & 2:
            continue
        if filter_cu is not None and rec.cu != filter_cu:
            continue
        if filter_simd is not None and rec.simd != filter_simd:
            continue
        if filter_wave is not None and rec.wave_id != filter_wave:
            continue
        inst = _wave_instance(rec.shader_engine, rec.cu, rec.simd, rec.wave_id, rec.time)
        key = (rec.shader_engine, rec.cu, rec.simd, rec.wave_id, inst)
        wave_records[key].append(rec)

    # (total_cycles, exec_count) per stack
    folded_cycles: dict[str, int] = defaultdict(int)
    folded_execs: dict[str, int] = defaultdict(int)

    def _dispatch_prefix(dispatch_id):
        """Build prefix frames and select funcmap for a dispatch."""
        raw_name = dispatches.get(str(dispatch_id), "")
        if not raw_name or raw_name.startswith("0 /"):
            return [], merged
        co_id = kernel_to_co.get(raw_name)
        display = demangle_name(raw_name) if do_demangle else raw_name
        if co_id is not None:
            return [f"code_object_{co_id}", display], per_co.get(co_id, merged)
        return [display], merged

    def _flush(prefix, stack, last_time, now):
        """Accumulate elapsed time for the current stack."""
        if last_time is not None and stack:
            dt = now - last_time
            if dt > 0:
                key = ";".join(prefix + stack)
                folded_cycles[key] += dt
                folded_execs[key] += 1

    for wave_key, recs in wave_records.items():
        recs.sort(key=lambda r: r.time)
        stack: list[str] = []
        last_time: Optional[int] = None
        cur_prefix: list[str] = []
        cur_dispatch_id: Optional[int] = None
        cur_funcmap: FuncMap = merged

        base_wk = wave_key[:4]
        spans = wave_spans.get(base_wk, [])

        for rec in recs:
            # Check dispatch context
            span = find_wave_span_at(spans, rec.time)
            dispatch_id = span.dispatch_id if span else None

            if dispatch_id != cur_dispatch_id:
                _flush(cur_prefix, stack, last_time, rec.time)
                cur_dispatch_id = dispatch_id
                stack = []
                cur_prefix, cur_funcmap = _dispatch_prefix(dispatch_id)

            marker_id, enter, exit_prev = decode_marker(rec.value, cur_funcmap)
            if marker_id == 0 and not exit_prev:
                continue

            name, mtype = cur_funcmap.resolve(marker_id)

            # Point markers (barriers, memory ops, user points) carry no
            # cycle attribution -- they're presence-only events. Skip them
            # entirely so they don't pollute the flamegraph with synthetic
            # 1-cycle samples. They're still in the raw shaderdata for
            # other tools that want to consume them.
            if mtype == "point":
                continue

            _flush(cur_prefix, stack, last_time, rec.time)
            last_time = rec.time

            # Handle exit_prev first (pop previous scope before entering new one)
            if exit_prev and stack:
                stack.pop()

            if enter:
                stack.append(name)

    return {k: (folded_cycles[k], folded_execs[k]) for k in folded_cycles}



# ---------------------------------------------------------------------------
# Output formats
# ---------------------------------------------------------------------------

def write_folded(folded: dict[str, tuple[int, int]], out):
    """Write folded stack format: 'frame1;frame2 total_cycles execs avg_cycles'."""
    for stack, (cycles, execs) in sorted(folded.items(), key=lambda x: -x[1][0]):
        avg = cycles / execs if execs else 0
        out.write(f"{stack} {cycles} {execs} {avg:.0f}\n")


def write_speedscope(folded: dict[str, tuple[int, int]], out):
    """Write speedscope JSON format."""
    all_frames: list[str] = []
    frame_index: dict[str, int] = {}
    for stack_str in folded:
        for frame in stack_str.split(";"):
            if frame not in frame_index:
                frame_index[frame] = len(all_frames)
                all_frames.append(frame)

    samples = []
    weights = []
    for stack_str, (cycles, execs) in sorted(folded.items(), key=lambda x: -x[1][0]):
        frames = stack_str.split(";")
        sample = [frame_index[f] for f in frames]
        samples.append(sample)
        weights.append(cycles)

    speedscope = {
        "$schema": "https://www.speedscope.app/file-format-schema.json",
        "shared": {
            "frames": [{"name": f} for f in all_frames],
        },
        "profiles": [{
            "type": "sampled",
            "name": "SQTT Trace",
            "unit": "cycles",
            "startValue": 0,
            "endValue": sum(weights),
            "samples": samples,
            "weights": weights,
        }],
        "name": "SQTT Flamegraph",
        "activeProfileIndex": 0,
        "exporter": "sqtt_flamegraph.py",
    }

    json.dump(speedscope, out, indent=2)
    out.write("\n")


# ---------------------------------------------------------------------------
# Interactive SVG flamegraph (no external dependencies)
# ---------------------------------------------------------------------------

@dataclass
class _FlameNode:
    name: str
    total: int = 0       # total cycles (self + children)
    execs: int = 0       # total execution count
    children: dict = field(default_factory=dict)  # name -> _FlameNode


def _build_flame_tree(folded: dict[str, tuple[int, int]]) -> _FlameNode:
    """Build a tree from folded stacks."""
    root = _FlameNode(name="all")
    for stack_str, (cycles, execs) in folded.items():
        frames = stack_str.split(";")
        node = root
        node.total += cycles
        node.execs += execs
        for frame in frames:
            if frame not in node.children:
                node.children[frame] = _FlameNode(name=frame)
            node = node.children[frame]
            node.total += cycles
            node.execs += execs
    return root


def _color_for(name: str) -> str:
    """Deterministic flame-themed color from name hash (stable across runs).

    Hue stays in the red-yellow range to keep the classic flamegraph look,
    but saturation and lightness vary widely so adjacent boxes contrast
    even when their hues are close.
    """
    import hashlib
    digest = hashlib.md5(name.encode("utf-8")).digest()
    h = int.from_bytes(digest[:4], "little")
    hue = h % 50              # 0-49: red → orange → yellow
    sat = 60 + ((h >> 8) % 40)   # 60-99%
    lit = 50 + ((h >> 16) % 30)  # 50-79%
    return f"hsl({hue},{sat}%,{lit}%)"


def render_flamegraph_svg(folded: dict[str, tuple[int, int]], title: str = "SQTT Flamegraph") -> str:
    """Render folded stacks as an interactive SVG flamegraph."""
    root = _build_flame_tree(folded)
    if root.total == 0:
        return "<svg></svg>"

    frame_h = 18
    font_size = 11
    min_width_px = 1
    pad_top = 76
    pad_bottom = 30
    canvas_w = 1200

    max_d = 0
    depth_stack = [(root, 0)]
    while depth_stack:
        node, d = depth_stack.pop()
        if not node.children:
            max_d = max(max_d, d)
        else:
            for c in node.children.values():
                depth_stack.append((c, d + 1))
    max_d += 1
    canvas_h = pad_top + max_d * frame_h + pad_bottom

    rects: list[str] = []
    total = root.total

    def _render_iterative(start_nodes: list[tuple[_FlameNode, int, float]]):
        work = list(start_nodes)
        while work:
            node, depth, x_offset = work.pop()
            w = (node.total / total) * canvas_w
            if w < min_width_px:
                continue
            y = canvas_h - pad_bottom - (depth + 1) * frame_h

            esc_name = html.escape(node.name, quote=True)
            pct = 100.0 * node.total / total
            avg = node.total / node.execs if node.execs else 0
            title_text = (f"{esc_name} ({node.total:,} cycles, {pct:.1f}%, "
                          f"{node.execs:,} execs, avg {avg:,.0f} cycles)")
            color = _color_for(node.name)

            max_chars = max(0, int(w / (font_size * 0.6)) - 1)
            label = node.name[:max_chars] if max_chars >= 2 else ""
            esc_label = html.escape(label, quote=True)

            rects.append(
                f'<g class="fr" data-name="{esc_name}">'
                f'<title>{title_text}</title>'
                f'<rect x="{x_offset:.1f}" y="{y}" width="{w:.1f}" '
                f'height="{frame_h - 1}" fill="{color}" rx="1"/>'
                f'<text x="{x_offset + 2:.1f}" y="{y + frame_h - 5}" '
                f'font-size="{font_size}" fill="#000">{esc_label}</text>'
                f'</g>'
            )

            cx = x_offset
            for child in sorted(node.children.values(), key=lambda c: -c.total):
                work.append((child, depth + 1, cx))
                cx += (child.total / total) * canvas_w

    cx = 0.0
    esc_title = html.escape(title)
    rects.append(
        f'<g class="fr" data-name="all">'
        f'<title>all ({root.total:,} cycles, {root.execs:,} execs, 100.0%)</title>'
        f'<rect x="0" y="{canvas_h - pad_bottom - frame_h}" '
        f'width="{canvas_w}" height="{frame_h - 1}" fill="#eee" rx="1"/>'
        f'<text x="2" y="{canvas_h - pad_bottom - 5}" '
        f'font-size="{font_size}" fill="#000">all ({root.total:,} cycles)</text>'
        f'</g>'
    )
    start_nodes = []
    for child in sorted(root.children.values(), key=lambda c: -c.total):
        start_nodes.append((child, 1, cx))
        cx += (child.total / total) * canvas_w
    _render_iterative(start_nodes)

    svg = f'''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_w}" height="{canvas_h}"
     viewBox="0 0 {canvas_w} {canvas_h}" font-family="monospace">
<style>
  .fr:hover rect {{ stroke: #000; stroke-width: 1; cursor: pointer; }}
  .fr.dim rect {{ opacity: 0.3; }}
  .fr.dim text {{ opacity: 0.3; }}
  .fr.hl rect {{ stroke: #f00; stroke-width: 1.5; }}
  #search {{ font-family: monospace; font-size: 13px; padding: 2px 6px;
             border: 1px solid #aaa; border-radius: 3px; width: 300px; }}
  #info {{ font-size: 12px; fill: #333; }}
</style>
<text x="10" y="22" font-size="16" font-weight="bold" fill="#000">{esc_title}</text>
<foreignObject x="10" y="30" width="400" height="26">
  <input xmlns="http://www.w3.org/1999/xhtml" id="search" type="text"
         placeholder="Search frames (regex)..." />
</foreignObject>
<text id="info" x="10" y="68"></text>
{"".join(rects)}
<script><![CDATA[
(function() {{
  var info = document.getElementById("info");
  var search = document.getElementById("search");
  var frames = document.querySelectorAll(".fr");
  frames.forEach(function(g) {{
    g.addEventListener("mouseenter", function() {{
      info.textContent = g.querySelector("title").textContent;
    }});
    g.addEventListener("mouseleave", function() {{
      info.textContent = "";
    }});
  }});
  search.addEventListener("input", function() {{
    var q = search.value;
    if (!q) {{
      frames.forEach(function(g) {{ g.classList.remove("dim","hl"); }});
      return;
    }}
    try {{ var re = new RegExp(q, "i"); }} catch(e) {{ return; }}
    frames.forEach(function(g) {{
      var name = g.getAttribute("data-name");
      if (re.test(name)) {{
        g.classList.add("hl");
        g.classList.remove("dim");
      }} else {{
        g.classList.add("dim");
        g.classList.remove("hl");
      }}
    }});
  }});
}})();
]]></script>
</svg>'''
    return svg


def save_flamegraph_svg(folded: dict[str, tuple[int, int]], title: str = "SQTT Flamegraph",
                       output_path: Optional[str] = None) -> str:
    """Render SVG flamegraph to disk. Returns the file path."""
    svg = render_flamegraph_svg(folded, title)

    path = output_path or os.path.join(os.getcwd(), "sqtt_flamegraph.svg")
    with open(path, "w") as f:
        f.write(svg)

    print(f"Wrote flamegraph to {path}", file=sys.stderr)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate flamegraph from SQTT auto-instrumentation traces")
    parser.add_argument("base_dir",
        help="Base directory containing trace output. Auto-discovers "
             "ui_*/shaderdata_*.json and *code_object_id*.out files "
             "recursively.")
    parser.add_argument("--code-objects", "-c", nargs="+",
        help="Code object files (.out) with .sqtt_funcmap sections. "
             "If omitted, auto-detected from base_dir.")
    parser.add_argument("--demangle", "-d", action="store_true",
        help="Demangle C++ function names")
    parser.add_argument("--cu", type=int, default=None,
        help="Filter to specific CU/WGP ID")
    parser.add_argument("--simd", type=int, default=None,
        help="Filter to specific SIMD ID")
    parser.add_argument("--wave", type=int, default=None,
        help="Filter to specific wave ID")
    parser.add_argument("--format", "-f", choices=["folded", "speedscope"],
        default="folded",
        help="Output format (default: folded)")
    parser.add_argument("--output", "-o", default=None,
        help="Output file (default: stdout)")
    parser.add_argument("--show", "-s", action="store_true",
        help="Render interactive SVG flamegraph and open in browser")
    args = parser.parse_args()

    base_dir = args.base_dir
    if not os.path.isdir(base_dir):
        print(f"Error: {base_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    trace_dirs, discovered_cos = discover_base_dir(base_dir)
    code_objects = args.code_objects or discovered_cos

    if not trace_dirs:
        print(f"Error: no shaderdata_*.json files found under {base_dir}",
              file=sys.stderr)
        sys.exit(1)

    if not code_objects:
        print("Error: no code objects found. Use --code-objects.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(trace_dirs)} trace dir(s), "
          f"{len(code_objects)} code object(s)", file=sys.stderr)
    for td in trace_dirs:
        print(f"  trace: {td}", file=sys.stderr)
    for co in code_objects:
        print(f"  code:  {co}", file=sys.stderr)

    # Load funcmaps (unmangled first for kernel->code_object lookup)
    per_co_raw = load_funcmaps(code_objects, do_demangle=False)
    kernel_to_co = build_kernel_to_co(per_co_raw)

    per_co = load_funcmaps(code_objects, args.demangle) if args.demangle else per_co_raw
    n_markers = sum(len(fm.markers) for fm in per_co.values())
    n_kernels = sum(len(fm.kernels) for fm in per_co.values())
    print(f"Loaded funcmaps from {len(per_co)} code object(s): "
          f"{n_markers} markers, {n_kernels} kernels", file=sys.stderr)

    if n_markers == 0:
        print("Warning: no instrumented functions found in code objects. "
              "Was the pass plugin loaded during compilation?", file=sys.stderr)

    # Build merged funcmap for preprocessing.
    merged = merge_funcmaps(per_co.values())

    # Process each trace directory independently
    all_folded: list[dict[str, tuple[int, int]]] = []
    total_records = 0
    all_waves: set[tuple] = set()

    for td in trace_dirs:
        records = load_shaderdata(td)
        if not records:
            continue
        dispatches, wave_spans = load_occupancy(td)
        n_dispatches = len([v for v in dispatches.values()
                            if not v.startswith("0 /")])

        # Filter out address trace records before building stacks
        resolver = make_funcmap_resolver(per_co, dispatches, kernel_to_co, wave_spans, merged)
        records, addr_traces = preprocess_records(records, merged, resolver)

        waves = set((record.shader_engine, record.cu, record.simd, record.wave_id) for record in records)
        total_records += len(records)
        all_waves.update(waves)

        n_addr = len(addr_traces)
        addr_info = f", {n_addr} addr traces" if n_addr else ""
        print(f"  {os.path.basename(td)}: {len(records)} records, "
              f"{len(waves)} waves, {n_dispatches} dispatch(es), "
              f"{len(wave_spans)} wave slots{addr_info}", file=sys.stderr)

        folded = build_stacks(
            records, per_co, dispatches, wave_spans, kernel_to_co,
            do_demangle=args.demangle,
            filter_cu=args.cu, filter_simd=args.simd, filter_wave=args.wave,
        )
        all_folded.append(folded)

    if total_records == 0:
        print("Error: no shaderdata records found", file=sys.stderr)
        sys.exit(1)

    print(f"Total: {total_records} records across {len(all_waves)} waves "
          f"in {len(trace_dirs)} directory(ies)", file=sys.stderr)

    folded = merge_folded(all_folded)

    if not folded:
        print("Warning: no stacks generated. Check filters or marker data.",
              file=sys.stderr)
        sys.exit(1)

    print(f"Generated {len(folded)} unique stacks", file=sys.stderr)

    write_folded(folded, sys.stdout)

    svg_path = args.output
    if svg_path and not svg_path.endswith(".svg"):
        svg_path = None
    svg_file = save_flamegraph_svg(folded, title="SQTT Flamegraph",
                                   output_path=svg_path)

    if args.output and args.output != svg_file:
        with open(args.output, "w") as out:
            if args.format == "speedscope":
                write_speedscope(folded, out)
            else:
                write_folded(folded, out)
        print(f"Wrote {args.output}", file=sys.stderr)

    if args.show:
        webbrowser.open(f"file://{os.path.abspath(svg_file)}")


if __name__ == "__main__":
    main()
