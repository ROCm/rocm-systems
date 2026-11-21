#!/usr/bin/env python3
###############################################################################
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
#
# Single entry point for rocSHMEM build-time analysis: parses .ninja_log once
# and from it produces everything the three older standalone scripts used to
# produce separately (ninja_build_timing.py, ftime_trace_summary.py,
# build_timeline_svg.py), plus the Graphviz dependency graphs that previously
# had to be hand-maintained:
#
#   - a per-component time budget + slowest-edges table (text)
#   - a -ftime-trace phase/template-instantiation summary (text), if the
#     build was configured with -DROCSHMEM_BUILD_TIME_TRACE=ON
#   - a Gantt-style build-timeline.svg
#   - build-dependency-graph.dot / build-dependency-graph-compact.dot,
#     rendered to .svg (and .png, if --png is passed)
#   - a single build-time-analysis.md report stitching all of the above
#     together
#
# Usage:
#
#   python3 scripts/analysis/build_time_report.py --build-dir build --output-dir docs/images/diagrams
#
# The HSACO-chain gate (which target `CMakeDeviceBitcodeTester.cmake`'s
# add_dependencies(${PROJECT_NAME}, device_bitcode_tester_hsacos) actually
# attaches to) and whether examples/unit-tests exist in this configuration
# are both detected from build.ninja/.ninja_log directly, not assumed, so
# this stays correct if the CMake wiring or MPI availability changes.

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap

# ============================================================================
# .ninja_log parsing (shared by every section below)
# ============================================================================


def parse_ninja_log(path):
    """Dedup by output, keeping the most recent run of each edge. Drops the
    absolute-path alias add_custom_command() targets leave behind alongside
    the build-dir-relative path, so group/edge totals aren't double-counted.
    """
    latest = {}
    with open(path) as f:
        first = f.readline()
        if not first.startswith("# ninja log"):
            f.seek(0)
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            try:
                start_ms, end_ms = int(parts[0]), int(parts[1])
            except ValueError:
                continue
            output = parts[3]
            if os.path.isabs(output):
                continue
            latest[output] = (start_ms, end_ms)
    return latest


# ============================================================================
# §1. Per-component budget + slowest edges (was ninja_build_timing.py)
# ============================================================================

COMPONENT_PATTERNS = [
    (re.compile(r"tests/unit_tests/"), "tests/unit"),
    (re.compile(r"tests/functional_tests/"), "tests/functional"),
    (re.compile(r"tests/rocprofiler_register/"), "tests/rocprofiler_register"),
    (re.compile(r"(^|/)examples/"), "examples"),
    (re.compile(r"(^|/)tools/"), "tools"),
    (re.compile(r"src/gda/"), "src/gda"),
    (re.compile(r"src/reverse_offload/"), "src/reverse_offload"),
    (re.compile(r"src/ipc/"), "src/ipc"),
    (re.compile(r"src/sdma/"), "src/sdma"),
    (re.compile(r"src/net/"), "src/net"),
    (re.compile(r"src/host/"), "src/host"),
    (re.compile(r"src/containers/"), "src/containers"),
    (re.compile(r"src/memory/"), "src/memory"),
    (re.compile(r"src/sync/"), "src/sync"),
    (re.compile(r"src/bootstrap/"), "src/bootstrap"),
    (re.compile(r"src/memfabric/"), "src/memfabric"),
    (re.compile(r"src/device/"), "src/device"),
    (re.compile(r"CMakeFiles/rocshmem\.dir/"), "src/(top-level)"),
]


def classify_component(output):
    for pattern, group in COMPONENT_PATTERNS:
        if pattern.search(output):
            return group
    if output.endswith((".o", ".obj")):
        return "other-compile"
    return "link/other"


def component_report(edges, top=30):
    import collections

    durations = sorted(((e - s, out) for out, (s, e) in edges.items()), reverse=True)
    wall_start = min(s for s, _ in edges.values())
    wall_end = max(e for _, e in edges.values())

    totals, counts = collections.Counter(), collections.Counter()
    for dur, out in durations:
        group = classify_component(out)
        totals[group] += dur
        counts[group] += 1

    lines = [
        f"ninja-tracked wall time: {(wall_end - wall_start) / 1000:.1f}s over {len(edges)} edges",
        f"sum of all (deduped) edge durations: {sum(totals.values()) / 1000:.1f}s",
        "",
        f"{'component':<24}{'#edges':>8}{'cpu-sum(s)':>12}{'avg(s)':>9}",
    ]
    component_rows = []
    for group, total in totals.most_common():
        count = counts[group]
        lines.append(f"{group:<24}{count:>8}{total / 1000:>12.1f}{(total / count) / 1000:>9.2f}")
        component_rows.append((group, count, total / 1000, (total / count) / 1000))

    lines += ["", f"slowest {top} edges:"]
    slow_rows = []
    for dur, out in durations[:top]:
        lines.append(f"{dur / 1000:8.1f}s  {out}")
        slow_rows.append((dur / 1000, out))

    return "\n".join(lines), component_rows, slow_rows, wall_start, wall_end


# ============================================================================
# §2. -ftime-trace phase + template-instantiation summary (was
#     ftime_trace_summary.py)
# ============================================================================


def iter_trace_files(build_dir):
    for root, _dirs, files in os.walk(build_dir):
        for name in files:
            if name.endswith(".json"):
                yield os.path.join(root, name)


def load_trace(path):
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or "traceEvents" not in data:
        return None
    return data["traceEvents"]


def ftime_trace_report(build_dir, top=30):
    import collections

    category_totals, category_counts = collections.Counter(), collections.Counter()
    detail_totals = collections.Counter()
    num_traces = 0

    for path in iter_trace_files(build_dir):
        events = load_trace(path)
        if events is None:
            continue
        num_traces += 1
        for ev in events:
            dur, name = ev.get("dur"), ev.get("name")
            if not dur or not name:
                continue
            category_totals[name] += dur
            category_counts[name] += 1
            if name in ("InstantiateFunction", "InstantiateClass", "ParseTemplate"):
                detail = ev.get("args", {}).get("detail")
                if detail:
                    detail_totals[(name, detail)] += dur

    if num_traces == 0:
        return None

    lines = [f"parsed {num_traces} time-trace file(s)", "",
             f"{'category':<30}{'#events':>10}{'sum(s)':>12}"]
    for name, total in category_totals.most_common(top):
        lines.append(f"{name:<30}{category_counts[name]:>10}{total / 1e6:>12.2f}")

    lines += ["", f"hottest instantiated symbols (summed across all TUs), top {top}:"]
    for (name, detail), total in detail_totals.most_common(top):
        lines.append(f"{total / 1e6:8.2f}s  [{name}] {detail}")

    phases = {
        "ExecuteCompiler": category_totals.get("Total ExecuteCompiler", 0) / 1e6,
        "Frontend": category_totals.get("Total Frontend", 0) / 1e6,
        "Source": category_totals.get("Total Source", 0) / 1e6,
        "Backend": category_totals.get("Total Backend", 0) / 1e6,
        "Optimizer": category_totals.get("Total Optimizer", 0) / 1e6,
    }
    top_instantiations = [(total / 1e6, detail) for (_name, detail), total in detail_totals.most_common(4)]

    return "\n".join(lines), num_traces, phases, top_instantiations


# ============================================================================
# §3. rocSHMEM-specific milestone detection (device bitcode / HSACO chain)
# ============================================================================

MILESTONE_PATTERNS = [
    ("bc_unopt_link", re.compile(r"^bitcode/[^/]+/librocshmem_device_.*_unopt\.bc$")),
    ("bc_opt", re.compile(r"^librocshmem_device_[^/]+\.bc$")),
    ("kernel_compile",
     re.compile(r"^tests/functional_tests/device_bitcode_tester_kernel_(?!.*_unopt)(?!.*_linked)[^/]+\.bc$")),
    ("kernel_unopt_link", re.compile(r"^tests/functional_tests/device_bitcode_tester_kernel_.*_unopt\.bc$")),
    ("kernel_opt", re.compile(r"^tests/functional_tests/device_bitcode_tester_kernel_.*_linked\.bc$")),
    ("kernel_codegen", re.compile(r"^tests/functional_tests/device_bitcode_tester_kernel_[^/]+\.o$")),
    ("hsaco_lld", re.compile(r"^tests/functional_tests/device_bitcode_tester_kernel_[^/]+\.hsaco$")),
    ("host_archive", re.compile(r"^librocshmem\.a$")),
    ("functest_link", re.compile(r"^tests/functional_tests/rocshmem_functional_tests$")),
]

TIMELINE_GROUP_PATTERNS = [
    ("bc_compiles", re.compile(r"^bitcode/[^/]+/(?!.*unopt).*\.bc$"), None),
    ("host_tus", re.compile(r"^CMakeFiles/rocshmem\.dir/"), None),
    # Not anchored to a single path segment: example targets can live in
    # subdirectories (e.g. examples/LL_MoE/ll_moe), and anchoring to
    # `examples/[^/]+$` silently dropped those from every count/total below.
    ("examples_links", re.compile(r"^examples/"), re.compile(r"\.o$")),
    ("other_links",
     re.compile(r"^(tools/|tests/unit_tests/rocshmem_unit_tests$|tests/rocprofiler_register/)"),
     re.compile(r"\.o$")),
    ("functest_sources",
     re.compile(r"^tests/functional_tests/CMakeFiles/rocshmem_functional_tests\.dir/"), None),
]


def find_milestones(edges):
    milestones = {}
    for name, pattern in MILESTONE_PATTERNS:
        matches = sorted(((s, e, out) for out, (s, e) in edges.items() if pattern.search(out)))
        if matches:
            s, e, out = matches[0]
            milestones[name] = (s, e, out)
    return milestones


def group_window(edges, include, exclude):
    starts, ends = [], []
    for out, (s, e) in edges.items():
        if not include.search(out):
            continue
        if exclude and exclude.search(out):
            continue
        starts.append(s)
        ends.append(e)
    if not starts:
        return None
    return min(starts), max(ends), len(starts), sum(e - s for s, e in zip(starts, ends))


def group_members(edges, include, exclude):
    """Like group_window, but returns the individual (output, duration_s)
    pairs sorted slowest-first -- for listing every target in a group inside
    one node label instead of just the aggregate.
    """
    members = [(out, (e - s) / 1000.0) for out, (s, e) in edges.items()
               if include.search(out) and not (exclude and exclude.search(out))]
    members.sort(key=lambda m: -m[1])
    return members


def short_target_name(out):
    """Strips the common group-path prefix off a ninja output so individual
    targets can be labeled by just their own name (keeping any subdirectory
    that distinguishes them, e.g. examples/LL_MoE/ll_moe -> LL_MoE/ll_moe).
    """
    for prefix in ("examples/", "tools/", "tests/unit_tests/", "tests/rocprofiler_register/"):
        if out.startswith(prefix):
            return out[len(prefix):]
    return os.path.basename(out)


def detect_hsaco_gate(build_ninja_path):
    """Figures out, from build.ninja itself, which target
    add_dependencies(${PROJECT_NAME}, device_bitcode_tester_hsacos) actually
    attached to -- rather than assuming it's still rocshmem_functional_tests,
    in case a future CMakeLists.txt change moves it.
    """
    if not os.path.exists(build_ninja_path):
        return None
    with open(build_ninja_path) as f:
        text = f.read()

    gated_targets = []
    for m in re.finditer(r"^build ([^:]+):.*$", text, re.MULTILINE):
        line = m.group(0)
        if "device_bitcode_tester_hsacos" in line and "device_bitcode_tester_hsacos:" not in line:
            target = m.group(1).strip()
            if "device_bitcode_tester" not in target:
                gated_targets.append(target)

    library_gated = any(
        t in ("rocshmem", "librocshmem.a") for t in gated_targets
    )
    return {"gated_targets": gated_targets, "library_gated": library_gated}


def detect_examples_built(edges):
    return group_window(edges, TIMELINE_GROUP_PATTERNS[2][1], TIMELINE_GROUP_PATTERNS[2][2]) is not None


def detect_unit_tests_built(edges):
    return any(re.search(r"^tests/unit_tests/rocshmem_unit_tests$", out) for out in edges)


# ============================================================================
# §4. Gantt-style timeline SVG (was build_timeline_svg.py)
# ============================================================================


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def read_stage_timings(timing_log):
    """Reads configure/install stage durations out of build_timing.log, if
    present. Supports two producers: a prior version of this tooling that
    hooked _common.sh directly and timed configure/build/install separately
    (`[timing] configure 4.97s` / `[timing] install 0.10s`), and the current
    scripts/analysis/timed_build.sh, which wraps an unmodified build script
    from the outside and can only measure build (read back exactly from
    .ninja_log, not from here) and a combined configure+install remainder
    (`[timing] configure+install (approx) 5.07s`, reported as configure_s
    with install_s left at 0 -- install is normally under a second anyway).
    """
    configure_s = install_s = 0.0
    if timing_log and os.path.exists(timing_log):
        with open(timing_log) as f:
            for line in f:
                m = re.match(r"\[timing\]\s+configure\s+([\d.]+)s", line)
                if m:
                    configure_s = float(m.group(1))
                m = re.match(r"\[timing\]\s+install\s+([\d.]+)s", line)
                if m:
                    install_s = float(m.group(1))
                m = re.match(r"\[timing\]\s+configure\+install \(approx\)\s+(-?[\d.]+)s", line)
                if m:
                    configure_s = max(float(m.group(1)), 0.0)
    return configure_s, install_s


def build_timeline_rows(edges, milestones, wall_start):
    def ms(name):
        s, e, _ = milestones[name]
        return (s - wall_start) / 1000.0, (e - wall_start) / 1000.0

    rows = []

    bc_group = group_window(edges, TIMELINE_GROUP_PATTERNS[0][1], TIMELINE_GROUP_PATTERNS[0][2])
    bc_segs = []
    if bc_group:
        s0, e0, n, _ = bc_group
        bc_segs.append(((s0 - wall_start) / 1000.0, (e0 - wall_start) / 1000.0,
                         f"{n} per-source .bc compiles (parallel)", "#b2f2bb"))
    if "bc_unopt_link" in milestones:
        s, e = ms("bc_unopt_link")
        bc_segs.append((s, e, f"llvm-link unopt.bc ({e - s:.1f}s)", "#ffd8a8"))
    if "bc_opt" in milestones:
        s, e = ms("bc_opt")
        bc_segs.append((s, e, f"opt -O3 librocshmem_device.bc -- SERIAL ({e - s:.1f}s)", "#ffd8a8"))
    if bc_segs:
        rows.append(("device bitcode library", bc_segs))

    host_group = group_window(edges, TIMELINE_GROUP_PATTERNS[1][1], TIMELINE_GROUP_PATTERNS[1][2])
    if host_group:
        s0, e0, n, cpu = host_group
        rows.append(("library host/device TUs", [
            ((s0 - wall_start) / 1000.0, (e0 - wall_start) / 1000.0,
             f"{n} TU compiles (parallel, {cpu/1000:.1f}s cpu-sum)", "#b2f2bb"),
        ]))

    if "host_archive" in milestones:
        s, e = ms("host_archive")
        rows.append(("librocshmem.a", [(s, max(e, s + 0.3), "archive ready", "#e9ecef")]))

    ex_group = group_window(edges, TIMELINE_GROUP_PATTERNS[2][1], TIMELINE_GROUP_PATTERNS[2][2])
    if ex_group:
        s0, e0, n, cpu = ex_group
        rows.append((f"examples x{n} (links)", [
            ((s0 - wall_start) / 1000.0, (e0 - wall_start) / 1000.0,
             f"-fgpu-rdc whole-program links ({cpu/1000:.1f}s cpu total)", "#ffa8a8"),
        ]))

    other_group = group_window(edges, TIMELINE_GROUP_PATTERNS[3][1], TIMELINE_GROUP_PATTERNS[3][2])
    if other_group:
        s0, e0, n, _ = other_group
        rows.append((f"other links (tools/unit-tests/\nrocprofiler-register, {n} targets)", [
            ((s0 - wall_start) / 1000.0, (e0 - wall_start) / 1000.0,
             f"{n} independent links", "#ffa8a8"),
        ]))

    hsaco_segs = []
    for name, color, label_fmt in [
        ("kernel_compile", "#b2f2bb", "kernel.bc compile"),
        ("kernel_unopt_link", "#ffd8a8", "llvm-link kernel+library"),
        ("kernel_opt", "#ffd8a8", "opt -O3 merged bc -- SERIAL"),
        ("kernel_codegen", "#e03131", "clang -x ir codegen -- SERIAL"),
        ("hsaco_lld", "#ffd8a8", "ld.lld HSACO link"),
    ]:
        if name in milestones:
            s, e = ms(name)
            hsaco_segs.append((s, e, f"{label_fmt} ({e - s:.1f}s)", color))
    if hsaco_segs:
        rows.append(("device_bitcode_tester\nHSACO chain (SERIAL)", hsaco_segs))

    ft_group = group_window(edges, TIMELINE_GROUP_PATTERNS[4][1], TIMELINE_GROUP_PATTERNS[4][2])
    ft_segs = []
    if ft_group:
        s0, e0, n, cpu = ft_group
        ft_segs.append(((s0 - wall_start) / 1000.0, (e0 - wall_start) / 1000.0,
                         f"{n} source compiles, parallel ({cpu/1000:.1f}s cpu)", "#b2f2bb"))
    if "functest_link" in milestones:
        s, e = ms("functest_link")
        ft_segs.append((s, e, f"final -fgpu-rdc whole-program link -- SERIAL ({e - s:.1f}s)", "#ffa8a8"))
    if ft_segs:
        rows.append(("rocshmem_functional_tests\n(gated by HSACO chain)", ft_segs))

    return rows


def build_critical_path(milestones, wall_start):
    order = ["bc_unopt_link", "bc_opt", "kernel_compile", "kernel_unopt_link",
             "kernel_opt", "kernel_codegen", "hsaco_lld", "functest_link"]
    path = []
    for name in order:
        if name not in milestones:
            continue
        s, e, _ = milestones[name]
        path.append(((s - wall_start) / 1000.0, (e - wall_start) / 1000.0))
    return path


def render_timeline_svg(rows, critical_path, wall_total, title, subtitle, configure_s, install_s, out_path):
    margin_l, margin_r, margin_t, margin_b = 300, 40, 70, 50
    row_h, row_gap, plot_w = 46, 14, 1500
    total = wall_total + max(configure_s, 0)

    def x(t):
        return margin_l + (t + configure_s) / total * plot_w

    height = margin_t + len(rows) * (row_h + row_gap) + margin_b + 90
    width = margin_l + plot_w + margin_r

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
           f'font-family="Helvetica,Arial,sans-serif">',
           f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>',
           f'<text x="{width/2}" y="28" font-size="18" font-weight="bold" '
           f'text-anchor="middle">{esc(title)}</text>',
           f'<text x="{width/2}" y="48" font-size="11" fill="#555" '
           f'text-anchor="middle">{esc(subtitle)}</text>']

    axis_top, axis_bot = margin_t, margin_t + len(rows) * (row_h + row_gap)
    t = 0
    step = max(60, round(wall_total / 12 / 60) * 60 or 60)
    while t <= wall_total + 1:
        gx = x(t)
        svg.append(f'<line x1="{gx:.1f}" y1="{axis_top-5}" x2="{gx:.1f}" y2="{axis_bot+5}" '
                    f'stroke="#e9ecef" stroke-width="1"/>')
        svg.append(f'<text x="{gx:.1f}" y="{axis_bot+20}" font-size="10" fill="#555" '
                    f'text-anchor="middle">{t:.0f}s</text>')
        t += step

    bx = x(0)
    svg.append(f'<line x1="{bx:.1f}" y1="{axis_top-5}" x2="{bx:.1f}" y2="{axis_bot+5}" '
                f'stroke="#868e96" stroke-width="1.5" stroke-dasharray="3,3"/>')
    svg.append(f'<text x="{bx:.1f}" y="{axis_top-10}" font-size="10" fill="#495057" '
                f'text-anchor="middle" font-weight="bold">build start (t=0)</text>')

    if configure_s > 0:
        svg.append(f'<rect x="{margin_l}" y="{axis_top-5}" width="{max(bx-margin_l,1):.1f}" '
                    f'height="{axis_bot-axis_top+10}" fill="#cfe2ff" opacity="0.25"/>')

    y = margin_t
    for label, segs in rows:
        lines = label.split("\n")
        ly = y + row_h/2 - (len(lines)-1)*6
        for i, line in enumerate(lines):
            svg.append(f'<text x="{margin_l-12}" y="{ly+i*13:.1f}" font-size="11" '
                        f'text-anchor="end" fill="#212529">{esc(line)}</text>')
        svg.append(f'<rect x="{margin_l}" y="{y}" width="{plot_w}" height="{row_h}" '
                    f'fill="#fafafa" stroke="#e9ecef"/>')
        for (s, e, seg_label, color) in segs:
            sx, ex = x(s), x(e)
            w = max(ex - sx, 1.0)
            svg.append(f'<rect x="{sx:.1f}" y="{y+6}" width="{w:.1f}" height="{row_h-12}" rx="3" '
                        f'fill="{color}" stroke="#495057" stroke-width="0.6"/>')
            if w > 40:
                svg.append(f'<text x="{sx+4:.1f}" y="{y+row_h/2+4:.1f}" font-size="9" '
                            f'fill="#1a1a1a">{esc(seg_label)}</text>')
            else:
                short = seg_label.split("(")[0].strip()
                svg.append(f'<text x="{sx:.1f}" y="{y-3}" font-size="8" fill="#495057">{esc(short)}</text>')
        y += row_h + row_gap

    if critical_path:
        cp_y = axis_bot + 34
        svg.append(f'<text x="{margin_l-12}" y="{cp_y+4}" font-size="11" text-anchor="end" '
                    f'fill="#e03131" font-weight="bold">critical path</text>')
        for (s, e) in critical_path:
            sx, ex = x(s), x(e)
            svg.append(f'<rect x="{sx:.1f}" y="{cp_y-8}" width="{max(ex-sx,1):.1f}" height="16" '
                        f'fill="#e03131" opacity="0.85"/>')
        svg.append(f'<rect x="{margin_l}" y="{cp_y-8}" width="{plot_w}" height="16" '
                    f'fill="none" stroke="#e9ecef"/>')

    svg.append("</svg>")
    with open(out_path, "w") as f:
        f.write("\n".join(svg))


# ============================================================================
# §5. Graphviz dependency graphs (full + compact), generated from the same
#     parsed edges/milestones instead of hand-maintained
# ============================================================================


def extract_ninja_commands(build_ninja_path, wanted_outputs):
    """Pulls the real `COMMAND = ...` line for each of `wanted_outputs` (an
    output path as it appears after `build `) straight out of build.ninja --
    the actual flags a build used, not a hand-maintained guess at them.
    """
    commands = {}
    remaining = set(wanted_outputs)
    if not remaining or not os.path.exists(build_ninja_path):
        return commands
    current_outputs = None
    with open(build_ninja_path) as f:
        for line in f:
            if line.startswith("build "):
                head = line[len("build "):].split(":", 1)[0].split("|", 1)[0]
                current_outputs = set(head.split())
            elif current_outputs and line.strip().startswith("COMMAND ="):
                hit = current_outputs & remaining
                if hit:
                    cmd = line.strip()[len("COMMAND = "):]
                    for out in hit:
                        commands[out] = cmd
                    remaining -= hit
                    if not remaining:
                        break
                current_outputs = None
    return commands


def extract_ninja_inputs(build_ninja_path, wanted_outputs):
    """Maps each of `wanted_outputs` to its explicit source input(s), read
    straight off the `build <out...>: RULE <in...> [| implicit] [|| order-only]`
    line -- the actual source file(s) ninja thinks that output is built from,
    so the dependency graph can show which sources feed each node (and where
    the same source feeds more than one node, e.g. a dual host+device compile).
    """
    inputs = {}
    remaining = set(wanted_outputs)
    if not remaining or not os.path.exists(build_ninja_path):
        return inputs
    with open(build_ninja_path) as f:
        for line in f:
            if not line.startswith("build "):
                continue
            content = line[len("build "):].rstrip("\n")
            outputs_part, sep, rest = content.partition(":")
            if not sep:
                continue
            outs = {tok for tok in outputs_part.split() if tok != "|"}
            hit = outs & remaining
            if not hit:
                continue
            rest = rest.strip().split(" || ", 1)[0].split(" | ", 1)[0]
            tokens = rest.split()[1:]  # drop the rule name, keep explicit inputs
            for out in hit:
                inputs[out] = tokens
            remaining -= hit
            if not remaining:
                break
    return inputs


def simplify_command(cmd):
    """Turns a raw ninja COMMAND (leading `cd <dir> &&`, absolute paths,
    possibly dozens of same-extension inputs) into a short, readable,
    still-real flag summary suitable for a graph node label.
    """
    cmd = cmd.strip()
    if " && " in cmd:
        cmd = cmd.rsplit(" && ", 1)[1]
    raw_tokens = cmd.split()
    tokens, skip_next = [], False
    for tok in raw_tokens:
        if skip_next:
            skip_next = False
            continue
        if tok in ("-isystem", "-I", "-L"):
            skip_next = True  # path is the next token
            continue
        if re.match(r"^-(I|isystem|L)/", tok):
            continue  # path baked into the flag (absolute, machine-local)
        tokens.append(os.path.basename(tok) if tok.startswith("/") else tok)
    out, i = [], 0
    while i < len(tokens):
        tok = tokens[i]
        if not tok.startswith("-") and "." in tok:
            ext = tok.rsplit(".", 1)[-1]
            j = i
            while j < len(tokens) and not tokens[j].startswith("-") and tokens[j].endswith(f".{ext}"):
                j += 1
            if j - i >= 4:
                out.append(f"<{j - i} x *.{ext} inputs>")
                i = j
                continue
        out.append(tok)
        i += 1
    return " ".join(out)


def read_build_config_summary(build_dir):
    """Short GPU_TARGETS/backend-flags summary read from CMakeCache.txt, for
    the cmake-configure node -- avoids hand-copying this into the graph.
    """
    cache_path = os.path.join(build_dir, "CMakeCache.txt")
    if not os.path.exists(cache_path):
        return None
    wanted = ("GPU_TARGETS", "USE_IPC", "USE_RO", "USE_GDA", "USE_THREADS",
              "BUILD_EXAMPLES", "BUILD_UNIT_TESTS", "USE_ROCPROFILER_REGISTER")
    values = {}
    with open(cache_path) as f:
        for line in f:
            if line.startswith("//") or ":" not in line or "=" not in line:
                continue
            name, rest = line.split(":", 1)
            if name not in wanted:
                continue
            _, _, value = rest.partition("=")
            values[name] = value.strip()
    if not values:
        return None
    parts = []
    if "GPU_TARGETS" in values:
        parts.append(f"GPU_TARGETS={values['GPU_TARGETS']}")
    backends_on = [name[len('USE_'):] for name in ("USE_IPC", "USE_RO", "USE_GDA", "USE_THREADS")
                   if values.get(name) == "ON"]
    if backends_on:
        parts.append(f"backends: {','.join(backends_on)}")
    extras_off = [name for name in ("BUILD_EXAMPLES", "BUILD_UNIT_TESTS", "USE_ROCPROFILER_REGISTER")
                  if values.get(name) == "OFF"]
    if extras_off:
        parts.append(f"off: {','.join(extras_off)}")
    return "\n".join(parts) if parts else None


def wrap_label(text, width=55):
    """Word-wraps each existing line of a node label independently, so a long
    single-line command (real flags pulled from build.ninja) doesn't blow up
    the box -- without disturbing labels that already chose their own line
    breaks.
    """
    out_lines = []
    for line in text.split("\n"):
        out_lines.extend(textwrap.wrap(line, width=width, break_long_words=False,
                                        break_on_hyphens=False) or [""])
    return "\n".join(out_lines)


def _circle(name, label, color="#f08c00", width=0.55, fontsize=8, penwidth=1.5):
    return (f'  t_{name} [label="{label}", shape=circle, fixedsize=true, width={width}, '
            f'height={width}, fontsize={fontsize}, fillcolor="#fff9db", color="{color}", '
            f'penwidth={penwidth}];\n'
            f'  {name} -> t_{name} [style=dashed, arrowhead=none, color="{color}", '
            f'constraint=false, weight=0];\n'
            f'  {{rank=same; {name}; t_{name};}}\n')


def _node(name, label, fillcolor, style="rounded,filled", color=None, penwidth=None, shape="box"):
    attrs = [f'label="{wrap_label(label)}"', f'style="{style}"', f'fillcolor="{fillcolor}"']
    if color:
        attrs.append(f'color={color}' if not color.startswith("#") else f'color="{color}"')
    if penwidth:
        attrs.append(f'penwidth={penwidth}')
    if shape != "box":
        attrs.append(f'shape={shape}')
    return f'  {name} [{", ".join(attrs)}];\n'


def _legend(fs):
    """A horizontal color-key strip (rank=same + invisible chaining edges, so
    it lays out as one row regardless of where the rest of the graph grows).
    """
    entries = [
        ("leg_config", "configure / source", "#cfe2ff", "rounded,filled", None, None, "box"),
        ("leg_parallel", "parallel compile (TUs)", "#b2f2bb", "rounded,filled", None, None, "box"),
        ("leg_serial", "serial custom step\n(clang/opt/lld)", "#ffd8a8", "rounded,filled", None, None, "box"),
        ("leg_bottleneck", "SERIAL bottleneck\n(red border = critical path)",
         "#ffd8a8", "rounded,filled", "red", 2, "box"),
        ("leg_link", "whole-program\n-fgpu-rdc link", "#ffa8a8", "rounded,filled", None, None, "box"),
        ("leg_artifact", "artifact / target / install\n(dashed, no duration of its own)",
         "#e9ecef", "rounded,filled,dashed", None, None, "box"),
        ("leg_gate", "cross-target\ndependency gate", "#fff3bf", "filled,dashed", "red", 2, "note"),
        ("leg_done", "final build outcome", "#d0bfff", "rounded,filled", None, None, "doubleoctagon"),
    ]
    lines = ["  subgraph cluster_legend {\n", f'    label="Legend"; style=dashed; fontsize={fs}; rank=same;\n']
    for name, label, fill, style, color, pw, shape in entries:
        lines.append("  " + _node(name, label, fill, style=style, color=color, penwidth=pw, shape=shape))
    lines.append('    leg_duration [label="measured\\nduration (s)", shape=circle, fixedsize=false, '
                  'width=0.7, height=0.7, fontsize=7, fillcolor="#fff9db", color="#f08c00", penwidth=1.5];\n')
    chain = " -> ".join(name for name, *_ in entries) + " -> leg_duration"
    lines.append(f"    {chain} [style=invis];\n")
    lines.append("  }\n")
    return "".join(lines)


def generate_dot_graphs(edges, milestones, gate_info, examples_built, unit_tests_built,
                         configure_s, install_s, wall_start, wall_end, out_full, out_compact,
                         build_ninja_path=None, build_dir=None):
    wall_total = (wall_end - wall_start) / 1000.0

    milestone_cmd_names = ("bc_unopt_link", "bc_opt", "kernel_compile", "kernel_unopt_link",
                            "kernel_opt", "kernel_codegen", "hsaco_lld")
    wanted_outputs = {milestones[n][2] for n in milestone_cmd_names if n in milestones}
    real_cmds = {out: simplify_command(cmd) for out, cmd in
                 extract_ninja_commands(build_ninja_path, wanted_outputs).items()} \
        if build_ninja_path else {}

    def real_cmd(name):
        return real_cmds.get(milestones[name][2]) if name in milestones else None

    config_summary = read_build_config_summary(build_dir) if build_dir else None

    host_component_breakdown = None
    if build_dir:
        host_outs = [out for out in edges if TIMELINE_GROUP_PATTERNS[1][1].search(out)]
        if host_outs:
            import collections
            counts = collections.Counter(classify_component(out) for out in host_outs)
            host_component_breakdown = ", ".join(f"{g}:{c}" for g, c in counts.most_common())

    # Source-file traceability: which .cpp/.hip actually feeds each parallel-compile
    # group, and where the *same* source feeds more than one group (e.g. a source
    # compiled once to device bitcode and once to a host object isn't a bug -- it's
    # how -fgpu-rdc dual host+device compilation works -- but it's exactly the kind
    # of thing that's invisible without seeing the source lists side by side).
    source_groups = {}
    if build_ninja_path:
        group_outs = {
            "bc_sources": {out for out in edges if TIMELINE_GROUP_PATTERNS[0][1].search(out)},
            "host_sources": {out for out in edges if TIMELINE_GROUP_PATTERNS[1][1].search(out)},
            "functest_sources": {out for out in edges if TIMELINE_GROUP_PATTERNS[4][1].search(out)},
        }
        all_outs = set().union(*group_outs.values())
        src_inputs = extract_ninja_inputs(build_ninja_path, all_outs)

        def sources_of(group):
            names = set()
            for out in group_outs[group]:
                for inp in src_inputs.get(out, []):
                    names.add(os.path.basename(inp))
            return names

        source_groups = {name: sources_of(name) for name in group_outs}

    def shared_sources_note(group_name):
        """"shares N files with <other group>" for every other group that overlaps --
        the signal that a source is intentionally dual-compiled (or, if that's
        unexpected, worth a second look).
        """
        mine = source_groups.get(group_name)
        if not mine:
            return None
        notes = []
        for other, files in source_groups.items():
            if other == group_name:
                continue
            shared = mine & files
            if shared:
                notes.append(f"shares {len(shared)} file(s) with {other}")
        return "; ".join(notes) if notes else None

    def dur(name):
        s, e, _ = milestones[name]
        return (e - s) / 1000.0

    def t(name):
        s, e, _ = milestones[name]
        return (s - wall_start) / 1000.0, (e - wall_start) / 1000.0

    host_group = group_window(edges, TIMELINE_GROUP_PATTERNS[1][1], TIMELINE_GROUP_PATTERNS[1][2])
    bc_group = group_window(edges, TIMELINE_GROUP_PATTERNS[0][1], TIMELINE_GROUP_PATTERNS[0][2])
    ex_group = group_window(edges, TIMELINE_GROUP_PATTERNS[2][1], TIMELINE_GROUP_PATTERNS[2][2])
    other_group = group_window(edges, TIMELINE_GROUP_PATTERNS[3][1], TIMELINE_GROUP_PATTERNS[3][2])
    ft_group = group_window(edges, TIMELINE_GROUP_PATTERNS[4][1], TIMELINE_GROUP_PATTERNS[4][2])
    ex_members = group_members(edges, TIMELINE_GROUP_PATTERNS[2][1], TIMELINE_GROUP_PATTERNS[2][2])
    other_members = group_members(edges, TIMELINE_GROUP_PATTERNS[3][1], TIMELINE_GROUP_PATTERNS[3][2])

    gated_note = (
        f"gates: {', '.join(gate_info['gated_targets'])}" if gate_info and gate_info["gated_targets"]
        else "no device_bitcode_tester_hsacos dependency found"
    )
    gates_library = bool(gate_info and gate_info["library_gated"])

    for compact, out_path in [(False, out_full), (True, out_compact)]:
        lines = []
        lines.append(f"// rocSHMEM build dependency graph{' (compact)' if compact else ''}")
        lines.append("// Auto-generated by scripts/analysis/build_time_report.py from build/.ninja_log")
        lines.append("// and build/build.ninja -- do not hand-edit, re-run the script instead.")
        lines.append(f"// HSACO-chain add_dependencies() {gated_note}"
                      f"{' (THIS GATES THE LIBRARY ITSELF)' if gates_library else ''}.")
        if not examples_built:
            lines.append("// examples/* not present in build.ninja this run (commonly: no MPI found).")
        if not unit_tests_built:
            lines.append("// rocshmem_unit_tests not present in build.ninja this run (commonly: no MPI found).")
        lines.append("")
        lines.append(f'digraph rocshmem_build{"_compact" if compact else ""} {{')
        lines.append("  rankdir=TB; fontname=\"Helvetica\"; labelloc=\"t\";")
        fs = 10 if compact else 11
        lines.append(f'  label=<<B>rocSHMEM build dependency graph{" (compact)" if compact else ""}</B>'
                      f'<BR/><FONT POINT-SIZE="{fs-1}">wall {wall_total:.1f}s -- generated from .ninja_log</FONT>>;')
        lines.append(f'  node [shape=box, style="rounded,filled", fontname="Helvetica", '
                      f'fontsize={fs}, margin="0.12,0.06"];')
        lines.append('  edge [fontname="Helvetica", fontsize=8, color="#555555"];')
        lines.append("")

        lines.append(_legend(fs - 1))
        lines.append("")

        # configure
        configure_label = "cmake configure"
        if not compact and config_summary:
            configure_label += f"\n{config_summary}"
        lines.append(_node("cmake_configure", configure_label, "#cfe2ff"))
        lines.append(_circle("cmake_configure", f"{configure_s:.2f}s"))

        # device bitcode
        if bc_group and "bc_opt" in milestones:
            bc_sources_label = f"BITCODE_SOURCES ({bc_group[2]} files)"
            if not compact and source_groups.get("bc_sources"):
                bc_sources_label += "\n" + ", ".join(sorted(source_groups["bc_sources"]))
            note = shared_sources_note("bc_sources")
            if not compact and note:
                bc_sources_label += f"\n({note})"
            lines.append(_node("bc_sources", bc_sources_label,
                                "#cfe2ff", style="rounded,filled,dashed"))
            lines.append(_node("bc_compile", "clang++ per-source bitcode compile"
                                if compact else
                                "clang++ -x hip --cuda-device-only -O3 -fgpu-rdc\n"
                                "-disable-llvm-passes (per-source .bc)",
                                "#b2f2bb"))
            longest_bc = max((e - s) for out, (s, e) in edges.items()
                              if TIMELINE_GROUP_PATTERNS[0][1].search(out)) / 1000.0
            lines.append(_circle("bc_compile", f"longest\n{longest_bc:.1f}s"))
            llvmlink_label = "llvm-link -> unopt.bc" if compact else (real_cmd("bc_unopt_link") or
                                                                        "llvm-link -> unopt.bc")
            lines.append(_node("bc_llvmlink", llvmlink_label, "#ffd8a8"))
            if "bc_unopt_link" in milestones:
                lines.append(_circle("bc_llvmlink", f"{dur('bc_unopt_link'):.1f}s", width=0.4, fontsize=7))
            s, e = t("bc_opt")
            bc_opt_label = "opt -O3 librocshmem_device.bc -- SERIAL" if compact else \
                (f"{real_cmd('bc_opt')} -- SERIAL" if real_cmd("bc_opt") else
                 "opt -O3 librocshmem_device.bc -- SERIAL")
            bc_opt_label += f"\nt={s:.1f}s -> {e:.1f}s"
            lines.append(_node("bc_opt", bc_opt_label, "#ffd8a8", color="red", penwidth=2))
            lines.append(_circle("bc_opt", f"{dur('bc_opt'):.1f}s", color="red", width=0.6, fontsize=9, penwidth=2))
            lines.append(_node("bc_target", "rocshmem_device_bitcode (ALL)", "#e9ecef",
                                style="rounded,filled,dashed"))
            lines += [
                "  cmake_configure -> bc_sources;\n", "  bc_sources -> bc_compile;\n",
                "  bc_compile -> bc_llvmlink;\n", "  bc_llvmlink -> bc_opt;\n",
                "  bc_opt -> bc_target;\n",
            ]

        # host library
        if host_group:
            s0, e0, n, cpu = host_group
            host_label = f"{n} host/device TU compiles\nparallel"
            if not compact and host_component_breakdown:
                host_label += f"\n{host_component_breakdown}"
            note = shared_sources_note("host_sources")
            if not compact and note:
                host_label += f"\n({note})"
            lines.append(_node("host_sources", host_label, "#b2f2bb"))
            lines.append(_circle("host_sources", f"{cpu/1000:.1f}s\ncpu-sum", width=0.6))
            lines.append(_node("host_archive", "librocshmem.a (-fgpu-rdc PUBLIC)", "#e9ecef",
                                style="rounded,filled,dashed"))
            lines += ["  cmake_configure -> host_sources;\n", "  host_sources -> host_archive;\n"]

        gate_color = "red" if not compact else "red"
        lines.append(_node("project_dep_hack",
                            f"add_dependencies(${{PROJECT_NAME}}, device_bitcode_tester_hsacos)\n{gated_note}",
                            "#fff3bf", style="filled,dashed", color=gate_color, penwidth=2, shape="note"))

        # HSACO chain
        hsaco_names = ["kernel_compile", "kernel_unopt_link", "kernel_opt", "kernel_codegen", "hsaco_lld"]
        if all(n in milestones for n in hsaco_names):
            chain_s, _, _ = milestones["kernel_compile"]
            _, chain_e, _ = milestones["hsaco_lld"]
            chain_dur = (chain_e - chain_s) / 1000.0
            lines.append(f"  subgraph cluster_hsaco {{\n"
                          f'    label="device_bitcode_tester_hsacos ({chain_dur:.1f}s serial chain)";\n'
                          f'    style="filled,dashed"; fillcolor="#fff5f5"; color=red; fontsize={fs};\n')
            lines.append(_node("kernel_hip", "device_bitcode_tester_kernel.hip", "#cfe2ff",
                                style="rounded,filled,dashed"))
            steps = [
                ("kernel_compile", "clang++ kernel compile", "#b2f2bb", 1, "#f08c00"),
                ("kernel_unopt_link", "llvm-link kernel+library", "#ffd8a8", 1, "#f08c00"),
                ("kernel_opt", "opt -O3 merged bc -- SERIAL", "#ffd8a8", 2, "red"),
                ("kernel_codegen", "clang -x ir codegen -- SERIAL", "#ffd8a8", 3, "red"),
                ("hsaco_lld", "ld.lld -> .hsaco", "#ffd8a8", 1, "#f08c00"),
            ]
            for name, label, fill, pw, ccolor in steps:
                s, e = t(name)
                extra = "\nSERIAL, single-threaded" if pw > 1 else ""
                full_label = real_cmd(name) or label
                lines.append(_node(name, f"{label}{extra}" if compact else
                                    f"{full_label}{extra}\nt={s:.1f}s -> {e:.1f}s", fill,
                                    color=ccolor, penwidth=pw if pw > 1 else None))
                lines.append(_circle(name, f"{dur(name):.1f}s", color=ccolor,
                                      width=0.4 + 0.1 * pw, fontsize=6 + pw, penwidth=pw))
            lines.append(_node("hsaco_target", "device_bitcode_tester_hsacos (ALL)", "#e9ecef",
                                style="rounded,filled,dashed"))
            lines.append("  }\n")
            lines += [
                "  kernel_hip -> kernel_compile;\n", "  kernel_compile -> kernel_unopt_link;\n",
                "  bc_opt -> kernel_unopt_link [color=red];\n" if "bc_opt" in milestones else "",
                "  kernel_unopt_link -> kernel_opt;\n", "  kernel_opt -> kernel_codegen;\n",
                "  kernel_codegen -> hsaco_lld;\n", "  hsaco_lld -> hsaco_target;\n",
                "  hsaco_target -> project_dep_hack [color=red, penwidth=2];\n",
            ]

        # examples (when present)
        if ex_group:
            s0, e0, n, cpu = ex_group
            lines.append(f"  subgraph cluster_examples {{\n"
                          f'    label="examples -- depend on librocshmem.a only'
                          f'{"\n(NOT gated by HSACO chain)" if not gates_library else ""}";\n'
                          f'    fillcolor="#f8f9fa"; style=filled; fontsize={fs};\n')
            ex_label = f"{n} targets (examples/*)"
            if not compact:
                ex_label += "\n" + "\n".join(f"{short_target_name(o)}: {d:.1f}s" for o, d in ex_members)
            lines.append(_node("examples_link", ex_label, "#ffa8a8"))
            lines.append(_circle("examples_link", f"{cpu/1000:.1f}s\ncpu total", color="red", width=0.7))
            lines.append("  }\n")
            if "host_archive" in [n for n in ["host_archive"] if host_group]:
                lines.append("  host_archive -> examples_link;\n")

        # other independent links (tools / rocprofiler-register / unit-tests, when present)
        if other_group:
            s0, e0, n, cpu = other_group
            lines.append(f"  subgraph cluster_consumers {{\n"
                          f'    label="independent links -- depend on librocshmem.a only'
                          f'{"\n(NOT gated by HSACO chain)" if not gates_library else ""}";\n'
                          f'    fillcolor="#f8f9fa"; style=filled; fontsize={fs};\n')
            other_label = "{} targets (tools/unit-tests/\nrocprofiler-register)".format(n)
            if not compact:
                other_label += "\n" + "\n".join(f"{short_target_name(o)}: {d:.1f}s" for o, d in other_members)
            lines.append(_node("other_links", other_label, "#ffa8a8"))
            lines.append(_circle("other_links", f"{cpu/1000:.1f}s\ncpu total", color="red", width=0.7))
            lines.append("  }\n")
            if "host_archive" in [n for n in ["host_archive"] if host_group]:
                lines.append("  host_archive -> other_links;\n")
        if not examples_built or not unit_tests_built:
            missing = []
            if not examples_built:
                missing.append("examples/*")
            if not unit_tests_built:
                missing.append("rocshmem_unit_tests")
            lines.append(_node("not_built", f"NOT BUILT this run:\n{', '.join(missing)}\n(commonly: no MPI found)",
                                "#e9ecef", style="rounded,filled,dashed", color="#adb5bd"))
            if "host_archive" in [n for n in ["host_archive"] if host_group]:
                lines.append('  host_archive -> not_built [style=dashed, color="#adb5bd"];\n')

        # functional tests
        if ft_group or "functest_link" in milestones:
            lines.append(f"  subgraph cluster_functest {{\n"
                          f'    label="rocshmem_functional_tests'
                          f'{" -- gated by device_bitcode_tester_hsacos" if not gates_library else ""}";\n'
                          f'    style="filled,dashed"; fillcolor="#fff5f5"; color=red; fontsize={fs};\n')
            if ft_group:
                s0, e0, n, cpu = ft_group
                functest_label = f"{n} functional-test source compiles\nparallel"
                note = shared_sources_note("functest_sources")
                if not compact and note:
                    functest_label += f"\n({note})"
                lines.append(_node("functest_sources", functest_label, "#b2f2bb"))
                longest = max((e - s) for out, (s, e) in edges.items()
                              if TIMELINE_GROUP_PATTERNS[4][1].search(out)) / 1000.0
                lines.append(_circle("functest_sources", f"{cpu/1000:.1f}s cpu\nlongest {longest:.1f}s", width=0.7))
            if "functest_link" in milestones:
                s, e = t("functest_link")
                lines.append(_node("functest_link",
                                    "rocshmem_functional_tests (link)" if compact else
                                    f"rocshmem_functional_tests (link)\nwhole-program -fgpu-rdc device LTO\n"
                                    f"t={s:.1f}s -> {e:.1f}s",
                                    "#ffa8a8", penwidth=2))
                lines.append(_circle("functest_link", f"{dur('functest_link'):.1f}s", color="red",
                                      width=0.65, fontsize=9, penwidth=2))
            lines.append("  }\n")
            lines.append("  project_dep_hack -> functest_sources [color=red, penwidth=2];\n"
                          if ft_group else "")
            lines.append("  functest_sources -> functest_link;\n" if ft_group else "")
            lines.append('  hsaco_target -> functest_link [style=dashed, label="  HSACO used at runtime"];\n'
                          if "functest_link" in milestones and all(n in milestones for n in hsaco_names) else "")

        # final outcome
        lines.append(_node("build_done", f"ninja build complete\nwall {wall_total:.1f}s", "#d0bfff",
                            shape="doubleoctagon"))
        lines.append(_circle("build_done", f"{wall_total:.1f}s\nwall", color="#7048e8", width=0.65, penwidth=2))
        for target in ["functest_link", "other_links", "examples_link", "bc_target"]:
            if any(f"{target} [" in l for l in lines):
                lines.append(f"  {target} -> build_done;\n")
        lines.append(_node("install_step", "cmake --install", "#e9ecef"))
        lines.append(_circle("install_step", f"{install_s:.2f}s", color="#868e96", width=0.45, fontsize=7))
        lines.append("  build_done -> install_step;\n")

        lines.append("}\n")
        with open(out_path, "w") as f:
            f.write("".join(l if l.endswith("\n") else l + "\n" for l in lines))


def run_dot(dot_path, svg_path, png_path=None):
    dot_bin = shutil.which("dot")
    if not dot_bin:
        print(f"warning: `dot` not found on PATH, skipping render of {dot_path}", file=sys.stderr)
        return False
    subprocess.run([dot_bin, "-Tsvg", dot_path, "-o", svg_path], check=True)
    if png_path:
        subprocess.run([dot_bin, "-Tpng", dot_path, "-o", png_path], check=True)
    return True


# ============================================================================
# §6. Markdown report
# ============================================================================


def write_markdown_report(path, component_rows, slow_rows, wall_start, wall_end, edges,
                           configure_s, build_s, install_s, ftime_result, gate_info,
                           examples_built, unit_tests_built, docs_rel_dir):
    wall_total = (wall_end - wall_start) / 1000.0
    total_cpu = sum(r[2] for r in component_rows)
    top10 = slow_rows[:10]
    top10_sum = sum(d for d, _ in top10)

    mpi_note = ""
    if not examples_built or not unit_tests_built:
        mpi_note = (
            "\n**This build has no `examples/*` and/or no `rocshmem_unit_tests`** "
            "(absent from `build.ninja` entirely) -- commonly because external MPI "
            "wasn't found (`BUILD_UNIT_TESTS`/`BUILD_EXAMPLES` are force-disabled by "
            "`CMakeLists.txt` when `MPI_FOUND` is false, regardless of their cache "
            "value). Check `build/CMakeCache.txt` for `MPI_CXX_COMPILER` if that's "
            "unexpected.\n"
        )

    gate_note = (
        f"`add_dependencies(${{PROJECT_NAME}}, device_bitcode_tester_hsacos)` "
        f"(in `tests/functional_tests/CMakeDeviceBitcodeTester.cmake`) currently gates: "
        f"**{', '.join(gate_info['gated_targets']) if gate_info and gate_info['gated_targets'] else 'nothing found'}**"
        + (" -- this includes the main library, so it blocks every downstream link."
           if gate_info and gate_info["library_gated"] else
           " -- not the main library, so other link targets are unaffected by this chain.")
    )

    lines = []
    lines.append("# rocSHMEM build-time analysis")
    lines.append("")
    lines.append("Generated by `scripts/analysis/build_time_report.py` from this build's "
                  "`build/.ninja_log`, `build/build_timing.log`, `build/build.ninja`, "
                  "and (if present) `-ftime-trace` JSON output. Re-run the script after "
                  "any new build to refresh this report and the dependency graphs/timeline "
                  f"under `{docs_rel_dir}/`.")
    lines.append(mpi_note)
    lines.append("## Summary")
    lines.append("")
    lines.append("```")
    lines.append(f"configure: {configure_s:>8.2f}s")
    lines.append(f"build:     {build_s:>8.2f}s")
    lines.append(f"install:   {install_s:>8.2f}s")
    lines.append("-" * 21)
    lines.append(f"total:     {configure_s + build_s + install_s:>8.2f}s")
    lines.append("")
    lines.append(f"ninja-tracked wall time: {wall_total:.1f}s over {len(edges)} edges")
    lines.append(f"sum of all (deduped) edge durations: {total_cpu:.1f}s "
                 f"-> realized parallelism ~{total_cpu/wall_total:.1f}x")
    lines.append("```")
    lines.append("")
    lines.append(f"{gate_note}")
    lines.append("")

    lines.append("## Per-component time budget")
    lines.append("")
    lines.append("| component | #edges | cpu-sum (s) | avg (s) |")
    lines.append("| --- | ---: | ---: | ---: |")
    for group, count, total, avg in component_rows:
        lines.append(f"| {group} | {count} | {total:.1f} | {avg:.2f} |")
    lines.append("")

    lines.append("## Slowest individual build edges")
    lines.append("")
    lines.append("| time | edge |")
    lines.append("| ---: | --- |")
    for dur, out in slow_rows[:16]:
        lines.append(f"| {dur:.1f}s | `{out}` |")
    lines.append("")
    lines.append(f"**The top 10 edges account for ~{top10_sum:.1f} of the {total_cpu:.1f} "
                 f"total cpu-seconds ({top10_sum/total_cpu*100:.0f}%).**")
    lines.append("")

    if ftime_result:
        _, num_traces, phases, top_instantiations = ftime_result
        lines.append("## `-ftime-trace` phase breakdown")
        lines.append("")
        lines.append(f"{num_traces} time-trace JSON file(s) captured.")
        lines.append("")
        lines.append("| phase | sum (s) |")
        lines.append("| --- | ---: |")
        for phase, val in phases.items():
            lines.append(f"| {phase} | {val:.1f}s |")
        lines.append("")
        if top_instantiations:
            lines.append("Top template instantiations (summed across every TU that instantiates "
                          "them -- note `-ftime-trace` durations are inclusive of nested spans, so "
                          "this table is not additive on top of the phase totals above):")
            lines.append("")
            for dur, detail in top_instantiations:
                short = detail if len(detail) < 160 else detail[:157] + "..."
                lines.append(f"- {dur:.1f}s  `{short}`")
            lines.append("")
    else:
        lines.append("## `-ftime-trace` phase breakdown")
        lines.append("")
        lines.append("No `-ftime-trace` JSON found under the build directory -- configure with "
                      "`-DROCSHMEM_BUILD_TIME_TRACE=ON` and rebuild to get this section.")
        lines.append("")

    lines.append("## Dependency graph and timeline")
    lines.append("")
    lines.append(f"`{docs_rel_dir}/` has the Graphviz dependency graph "
                  "(`build-dependency-graph.dot`, full compiler-flag detail; "
                  "`build-dependency-graph-compact.dot`, trimmed) and a Gantt-style "
                  "`build-timeline.svg`, all regenerated by this script from this run's data.")
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ============================================================================
# main
# ============================================================================


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-dir", default="build", help="build directory")
    ap.add_argument("--output-dir", default="docs/images/diagrams",
                     help="where to write the .dot/.svg/.png/timeline files")
    ap.add_argument("--report", default="build-time-analysis.md",
                     help="path to write the combined markdown report")
    ap.add_argument("--top", type=int, default=30, help="rows to show per text table")
    ap.add_argument("--png", action="store_true", help="also render .png alongside .svg")
    ap.add_argument("--no-report", action="store_true", help="skip writing the markdown report")
    ap.add_argument("--summary-only", action="store_true",
                     help="print only the per-component/slowest-edges text (what the old "
                          "ninja_build_timing.py did) and exit -- no ftime-trace, timeline, "
                          "dot graphs, or markdown report. Meant for a build script's "
                          "post-build hook, run every build, not just on-demand analysis.")
    args = ap.parse_args()

    ninja_log = os.path.join(args.build_dir, ".ninja_log")
    timing_log = os.path.join(args.build_dir, "build_timing.log")
    build_ninja = os.path.join(args.build_dir, "build.ninja")

    if not os.path.exists(ninja_log):
        print(f"no ninja log at {ninja_log} (did the build use -G Ninja?)", file=sys.stderr)
        return 1

    edges = parse_ninja_log(ninja_log)
    if not edges:
        print(f"{ninja_log} has no edges recorded yet", file=sys.stderr)
        return 1

    wall_start = min(s for s, _ in edges.values())
    wall_end = max(e for _, e in edges.values())

    text, component_rows, slow_rows, wall_start, wall_end = component_report(edges, top=args.top)
    print(text)

    if args.summary_only:
        return 0
    print()

    ftime_result = ftime_trace_report(args.build_dir, top=args.top)
    if ftime_result:
        print(ftime_result[0])
    else:
        print("no -ftime-trace JSON found; skip with -DROCSHMEM_BUILD_TIME_TRACE=ON off, "
              "or configure with it ON and rebuild for that section.", file=sys.stderr)

    configure_s, install_s = read_stage_timings(timing_log)
    build_s = (wall_end - wall_start) / 1000.0

    milestones = find_milestones(edges)
    gate_info = detect_hsaco_gate(build_ninja)
    examples_built = detect_examples_built(edges)
    unit_tests_built = detect_unit_tests_built(edges)

    os.makedirs(args.output_dir, exist_ok=True)

    rows = build_timeline_rows(edges, milestones, wall_start)
    critical_path = build_critical_path(milestones, wall_start)
    if install_s > 0:
        wall_total = (wall_end - wall_start) / 1000.0
        rows.append(("install", [(wall_total, wall_total + install_s,
                                   f"cmake --install ({install_s:.2f}s)", "#e9ecef")]))
    timeline_path = os.path.join(args.output_dir, "build-timeline.svg")
    subtitle = f"wall time {build_s:.1f}s over {len(edges)} edges -- {ninja_log}"
    render_timeline_svg(rows, critical_path, build_s + (install_s if install_s > 0 else 0),
                         "Build timeline", subtitle, configure_s, install_s, timeline_path)
    print(f"\nwrote {timeline_path}")

    dot_full = os.path.join(args.output_dir, "build-dependency-graph.dot")
    dot_compact = os.path.join(args.output_dir, "build-dependency-graph-compact.dot")
    generate_dot_graphs(edges, milestones, gate_info, examples_built, unit_tests_built,
                        configure_s, install_s, wall_start, wall_end, dot_full, dot_compact,
                        build_ninja_path=build_ninja, build_dir=args.build_dir)
    print(f"wrote {dot_full}")
    print(f"wrote {dot_compact}")

    for dot_path in (dot_full, dot_compact):
        svg_path = dot_path[:-4] + ".svg"
        png_path = dot_path[:-4] + ".png" if args.png else None
        if run_dot(dot_path, svg_path, png_path):
            print(f"wrote {svg_path}" + (f" and {png_path}" if png_path else ""))

    if not args.no_report:
        write_markdown_report(args.report, component_rows, slow_rows, wall_start, wall_end,
                               edges, configure_s, build_s, install_s, ftime_result, gate_info,
                               examples_built, unit_tests_built, args.output_dir)
        print(f"wrote {args.report}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
