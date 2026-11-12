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
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""Summarize a rocprofv3 rocpd trace database into a markdown triage report.

Reads one or more ``*_results.db`` files written by ``rocprofv3`` (default
``rocpd`` output format) and reports the GPU time breakdown, hot kernels,
memory copies, host API time, ROCTx ranges, GPU idle gaps, and rule-based
findings. Uses only the Python standard library (``sqlite3``) and the views
that rocprofv3 creates inside every database (``kernels``, ``memory_copies``,
``regions``, ``processes``, ``rocpd_info_agent``).
"""

from __future__ import annotations

import argparse
import heapq
import sqlite3
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

NS_PER_US = 1_000.0
NS_PER_MS = 1_000_000.0
BYTES_PER_MIB = 1024.0 * 1024.0

# Thresholds mirror the PerfXpert knowledge base
# (experimental/python/perfxpert/perfxpert/knowledge/top_down_analysis.yaml,
# bottleneck_types.yaml and tools/regression.py) so both tools agree.
IDLE_RED_FLAG = 0.10  # GPU idle share of the GPU activity window
MEMCPY_RED_FLAG = 0.20  # copy-engine busy share of the GPU activity window
HOT_COVERAGE = 0.80  # hot kernels = top-K covering 80% of kernel time ...
HOT_INDIVIDUAL = 0.03  # ... plus any kernel with at least 3% on its own
TINY_KERNEL_US = 10.0  # average duration below which launch overhead dominates
TINY_KERNEL_MIN_CALLS = 100  # ignore one-off tiny kernels

# Pinned host<->device copies on PCIe Gen4/Gen5 x16 or Infinity Fabric links
# sustain tens of GB/s; pageable copies are staged through a bounce buffer and
# typically land in the low single digits. Copies smaller than 1 MiB are
# latency-dominated, so they are excluded from this check.
HOST_LINK_SLOW_GBPS = 10.0
HOST_LINK_MIN_AVG_BYTES = 1024 * 1024

SYNC_APIS = {
    "hipDeviceSynchronize",
    "hipStreamSynchronize",
    "hipEventSynchronize",
    "hipMemcpy",
    "hipMemcpyDtoH",
    "hipMemcpyHtoD",
    "hipMemset",
}


@dataclass
class KernelStat:
    name: str
    calls: int = 0
    total_ns: float = 0.0
    min_ns: float = float("inf")
    max_ns: float = 0.0
    vgpr: int = 0
    agpr: int = 0
    sgpr: int = 0
    lds: int = 0
    scratch: int = 0
    workgroup: int = 0
    grid: int = 0

    def merge(self, row: tuple) -> None:
        _, calls, total, mn, mx, vgpr, agpr, sgpr, lds, scratch, wg, grid = row
        self.calls += int(calls)
        self.total_ns += float(total or 0)
        self.min_ns = min(self.min_ns, float(mn or 0))
        self.max_ns = max(self.max_ns, float(mx or 0))
        self.vgpr = max(self.vgpr, int(vgpr or 0))
        self.agpr = max(self.agpr, int(agpr or 0))
        self.sgpr = max(self.sgpr, int(sgpr or 0))
        self.lds = max(self.lds, int(lds or 0))
        self.scratch = max(self.scratch, int(scratch or 0))
        self.workgroup = max(self.workgroup, int(wg or 0))
        self.grid = max(self.grid, int(grid or 0))

    @property
    def avg_ns(self) -> float:
        return self.total_ns / self.calls if self.calls else 0.0


@dataclass
class CallStat:
    calls: int = 0
    total_ns: float = 0.0

    def add(self, calls: int, total: float) -> None:
        self.calls += int(calls)
        self.total_ns += float(total or 0)


@dataclass
class CopyStat:
    calls: int = 0
    total_bytes: float = 0.0
    total_ns: float = 0.0

    def add(self, calls: int, size: float, duration: float) -> None:
        self.calls += int(calls)
        self.total_bytes += float(size or 0)
        self.total_ns += float(duration or 0)


@dataclass
class Gap:
    length_ns: float
    at_ns: float
    before: str
    after: str


@dataclass
class AgentTimeline:
    label: str
    window_ns: float = 0.0
    kernel_busy_ns: float = 0.0
    copy_busy_ns: float = 0.0
    any_busy_ns: float = 0.0
    gaps: list[Gap] = field(default_factory=list)

    @property
    def idle_ns(self) -> float:
        return max(0.0, self.window_ns - self.any_busy_ns)


@dataclass
class Report:
    inputs: list[str] = field(default_factory=list)
    processes: list[tuple] = field(default_factory=list)
    kernels: dict[str, KernelStat] = field(default_factory=dict)
    copies: dict[str, CopyStat] = field(default_factory=lambda: defaultdict(CopyStat))
    apis: dict[tuple[str, str], CallStat] = field(
        default_factory=lambda: defaultdict(CallStat)
    )
    markers: dict[str, CallStat] = field(default_factory=lambda: defaultdict(CallStat))
    kfd: dict[str, CallStat] = field(default_factory=lambda: defaultdict(CallStat))
    kfd_overlap: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    timelines: list[AgentTimeline] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def kernel_total_ns(self) -> float:
        return sum(k.total_ns for k in self.kernels.values())


def discover_databases(paths: list[Path]) -> list[Path]:
    found: list[Path] = []
    for path in paths:
        if path.is_dir():
            found.extend(sorted(path.rglob("*.db")))
        elif path.is_file():
            found.append(path)
        else:
            raise ValueError(f"{path} does not exist")
    if not found:
        raise ValueError(
            "No .db files found. rocprofv3 writes <output-dir>/<output-file>_results.db "
            "(default %hostname%/%pid%_results.db); pass that file or its directory."
        )
    return found


def connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    names = {
        row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='view'")
    }
    missing = {"kernels", "memory_copies", "regions", "processes"} - names
    if missing:
        conn.close()
        raise ValueError(
            f"{path} is not a rocprofv3 rocpd database (missing views: {', '.join(sorted(missing))})"
        )
    return conn


def merge_intervals(intervals: list[tuple[float, float, str]]):
    """Yield merged (start, end, first_name, last_name) spans from start-sorted intervals."""
    current = None
    for start, end, name in intervals:
        if current is None:
            current = [start, end, name, name]
        elif start <= current[1]:
            if end > current[1]:
                current[1] = end
                current[3] = name
        else:
            yield tuple(current)
            current = [start, end, name, name]
    if current is not None:
        yield tuple(current)


def busy_time(intervals: list[tuple[float, float, str]]) -> float:
    return sum(end - start for start, end, _, _ in merge_intervals(intervals))


def agent_labels(conn: sqlite3.Connection) -> dict[int, str]:
    labels = {}
    for abs_index, name, product, type_index in conn.execute(
        "SELECT absolute_index, name, product_name, type_index FROM rocpd_info_agent WHERE type='GPU'"
    ):
        product = product or name
        labels[int(abs_index)] = f"GPU {type_index} ({product}, {name})"
    return labels


def collect_timelines(
    conn: sqlite3.Connection, top_gaps: int, source: str
) -> list[AgentTimeline]:
    labels = agent_labels(conn)
    keys = conn.execute(
        "SELECT DISTINCT nid, pid, agent_abs_index FROM kernels WHERE agent_type='GPU' "
        "UNION SELECT DISTINCT nid, pid, CASE WHEN dst_agent_type='GPU' THEN dst_agent_abs_index "
        "ELSE src_agent_abs_index END FROM memory_copies"
    ).fetchall()
    timelines = []
    for nid, pid, agent in keys:
        kernels = [
            (float(s), float(e), n)
            for s, e, n in conn.execute(
                "SELECT start, end, name FROM kernels WHERE nid=? AND pid=? AND agent_abs_index=? "
                "ORDER BY start",
                (nid, pid, agent),
            )
        ]
        copies = [
            (float(s), float(e), n)
            for s, e, n in conn.execute(
                "SELECT start, end, name FROM memory_copies WHERE nid=? AND pid=? AND "
                "((dst_agent_type='GPU' AND dst_agent_abs_index=?) OR "
                "(dst_agent_type!='GPU' AND src_agent_abs_index=?)) ORDER BY start",
                (nid, pid, agent, agent),
            )
        ]
        everything = sorted(kernels + copies)
        if not everything:
            continue
        window_start = everything[0][0]
        window_end = max(e for _, e, _ in everything)
        label = f"{labels.get(int(agent), f'agent {agent}')} pid {pid}"
        if source:
            label += f" [{source}]"
        timeline = AgentTimeline(
            label=label,
            window_ns=window_end - window_start,
            kernel_busy_ns=busy_time(kernels),
            copy_busy_ns=busy_time(copies),
            any_busy_ns=busy_time(everything),
        )
        previous = None
        gaps = []
        for span in merge_intervals(everything):
            if previous is not None and span[0] > previous[1]:
                gaps.append(
                    Gap(
                        length_ns=span[0] - previous[1],
                        at_ns=previous[1] - window_start,
                        before=previous[3],
                        after=span[2],
                    )
                )
            previous = span
        gaps.sort(key=lambda g: g.length_ns, reverse=True)
        timeline.gaps = gaps[:top_gaps]
        timelines.append(timeline)
    return timelines


def collect(conn: sqlite3.Connection, report: Report, top_gaps: int, source: str) -> None:
    report.processes.extend(
        conn.execute("SELECT pid, command, start, end FROM processes").fetchall()
    )

    for row in conn.execute(
        "SELECT name, COUNT(*), SUM(duration), MIN(duration), MAX(duration), "
        "MAX(vgpr_count), MAX(accum_vgpr_count), MAX(sgpr_count), MAX(lds_size), "
        "MAX(scratch_size), MAX(workgroup_x*workgroup_y*workgroup_z), "
        "MAX(grid_x*grid_y*grid_z) FROM kernels GROUP BY name"
    ):
        stat = report.kernels.setdefault(row[0], KernelStat(name=row[0]))
        stat.merge(row)

    for src_type, dst_type, same, calls, size, duration in conn.execute(
        "SELECT src_agent_type, dst_agent_type, src_agent_abs_index = dst_agent_abs_index, "
        "COUNT(*), SUM(size), SUM(duration) FROM memory_copies GROUP BY 1, 2, 3"
    ):
        report.copies[direction_label(src_type, dst_type, bool(same))].add(
            calls, size, duration
        )

    for category, name, calls, total in conn.execute(
        "SELECT category, name, COUNT(*), SUM(duration) FROM regions GROUP BY category, name"
    ):
        category = category or ""
        if category.startswith("MARKER"):
            report.markers[name].add(calls, total)
        elif category.startswith("KFD"):
            report.kfd[category].add(calls, total)
        else:
            report.apis[(category, name)].add(calls, total)

    if report.kfd:
        attribute_kfd_time(conn, report)

    report.timelines.extend(collect_timelines(conn, top_gaps, source))


def attribute_kfd_time(conn: sqlite3.Connection, report: Report) -> None:
    """Credit KFD event time to the host API calls it overlaps (sweep over start-sorted lists)."""
    events = defaultdict(list)
    for pid, start, end in conn.execute(
        "SELECT pid, start, end FROM regions WHERE category LIKE 'KFD%' AND end > start ORDER BY start"
    ):
        events[pid].append((float(start), float(end)))
    for pid, spans in events.items():
        apis = conn.execute(
            "SELECT start, end, name FROM regions WHERE pid = ? AND category NOT LIKE 'KFD%' "
            "AND category NOT LIKE 'MARKER%' ORDER BY start",
            (pid,),
        ).fetchall()
        active: list[tuple[float, float, str]] = []
        j = 0
        for start, end in spans:
            while j < len(apis) and apis[j][0] <= end:
                heapq.heappush(active, (float(apis[j][1]), float(apis[j][0]), apis[j][2]))
                j += 1
            while active and active[0][0] < start:
                heapq.heappop(active)
            for api_end, api_start, name in active:
                overlap = min(end, api_end) - max(start, api_start)
                if overlap > 0:
                    report.kfd_overlap[name] += overlap


def direction_label(src_type: str, dst_type: str, same_agent: bool) -> str:
    if src_type == "CPU" and dst_type == "GPU":
        return "Host to device (H2D)"
    if src_type == "GPU" and dst_type == "CPU":
        return "Device to host (D2H)"
    if src_type == "GPU" and dst_type == "GPU":
        return (
            "Device to device (same GPU)" if same_agent else "Peer to peer (GPU to GPU)"
        )
    return f"{src_type} to {dst_type}"


def hot_kernels(report: Report) -> list[KernelStat]:
    total = report.kernel_total_ns
    if total <= 0:
        return []
    ranked = sorted(report.kernels.values(), key=lambda k: k.total_ns, reverse=True)
    hot, covered = [], 0.0
    for kernel in ranked:
        hot.append(kernel)
        covered += kernel.total_ns / total
        if covered >= HOT_COVERAGE:
            break
    names = {k.name for k in hot}
    hot.extend(
        k for k in ranked if k.name not in names and k.total_ns / total >= HOT_INDIVIDUAL
    )
    return hot


def findings(report: Report) -> list[str]:
    out: list[str] = []
    total = report.kernel_total_ns

    hot = hot_kernels(report)
    if hot:
        names = ", ".join(f"`{short(k.name)}` ({pct(k.total_ns, total)})" for k in hot)
        out.append(
            f"**Hot kernels** (top kernels covering {HOT_COVERAGE:.0%} of kernel time, plus any "
            f"kernel at or above {HOT_INDIVIDUAL:.0%}): {names}. Optimize these first."
        )

    for tl in report.timelines:
        if tl.window_ns <= 0:
            continue
        idle = tl.idle_ns / tl.window_ns
        copy = tl.copy_busy_ns / tl.window_ns
        if idle >= IDLE_RED_FLAG:
            out.append(
                f"**GPU idle {idle:.0%}** of the activity window on {tl.label}. Look at the idle "
                "gaps table: gaps after synchronization or copies point to serialization, many "
                "small gaps point to launch overhead."
            )
        if copy >= MEMCPY_RED_FLAG:
            out.append(
                f"**Memory copies occupy {copy:.0%}** of the activity window on {tl.label}. "
                "Reduce transfers, keep data resident on the GPU, or overlap copies with kernels "
                "using pinned memory, hipMemcpyAsync and multiple streams."
            )

    for kernel in report.kernels.values():
        if (
            kernel.calls >= TINY_KERNEL_MIN_CALLS
            and kernel.avg_ns / NS_PER_US < TINY_KERNEL_US
        ):
            out.append(
                f"**Launch-bound candidate**: `{short(kernel.name)}` runs {kernel.calls} times "
                f"averaging {kernel.avg_ns / NS_PER_US:.1f} us. Kernels this short are dominated by "
                "launch overhead; fuse them, batch the work, or capture the sequence in a HIP graph."
            )
        if kernel.scratch > 0:
            out.append(
                f"**Scratch memory in use**: `{short(kernel.name)}` has a private segment of "
                f"{kernel.scratch} bytes per work-item, which usually means register spills or "
                "stack arrays. Check VGPR pressure (launch bounds, smaller tiles) and dynamic indexing."
            )

    for label, stat in report.copies.items():
        if "H2D" not in label and "D2H" not in label:
            continue
        if not stat.calls or not stat.total_ns:
            continue
        avg_bytes = stat.total_bytes / stat.calls
        gbps = stat.total_bytes / stat.total_ns
        if avg_bytes >= HOST_LINK_MIN_AVG_BYTES and gbps < HOST_LINK_SLOW_GBPS:
            out.append(
                f"**Slow {label} copies**: {gbps:.1f} GB/s effective for an average transfer of "
                f"{avg_bytes / BYTES_PER_MIB:.1f} MiB. This is typical of pageable host memory; "
                "allocate the host buffer with hipHostMalloc (pinned) and re-measure."
            )

    kfd_timed = {c: s for c, s in report.kfd.items() if s.total_ns > 0}
    if kfd_timed:
        parts = ", ".join(
            f"{c} x{s.calls} ({s.total_ns / NS_PER_MS:.2f} ms)"
            for c, s in sorted(kfd_timed.items())
        )
        kfd_ns = sum(s.total_ns for s in kfd_timed.values())
        if report.kfd_overlap:
            name, ns = max(report.kfd_overlap.items(), key=lambda kv: kv[1])
            where = f"mostly during `{name}` ({min(1.0, ns / kfd_ns):.0%} of KFD time)"
        else:
            where = "outside any traced host API call"
        out.append(
            f"**KFD driver events**: {parts}, {where}. These are driver page faults, migrations, "
            "or prefetches (SVM/HMM, managed memory, XNACK). Events inside a host call add to its "
            "latency; if that call is on the critical path, check whether the buffers are managed "
            "or system-allocated and whether HSA_XNACK is enabled."
        )

    sync_ns = sum(s.total_ns for (_, n), s in report.apis.items() if n in SYNC_APIS)
    api_ns = sum(s.total_ns for s in report.apis.values())
    if api_ns > 0 and sync_ns / api_ns >= 0.5:
        out.append(
            f"**Host mostly waits on the GPU**: synchronizing calls account for "
            f"{sync_ns / api_ns:.0%} of host API time. Time inside these calls is GPU work or "
            "transfers the host is blocked on, not API overhead."
        )

    if not report.kernels:
        out.append(
            "No kernel dispatches were recorded. Re-run with `--runtime-trace` or "
            "`--kernel-trace`, and confirm the application actually ran on the GPU."
        )
    return out


def short(name: str, width: int = 70) -> str:
    name = " ".join(name.split())
    return name if len(name) <= width else name[: width - 3] + "..."


def cell(text: str) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")


def pct(part: float, whole: float) -> str:
    return f"{100.0 * part / whole:.1f}%" if whole else "0.0%"


def render(report: Report, top: int) -> str:
    lines = ["# rocprofv3 Trace Report", ""]
    lines.append(f"- Inputs: {', '.join(report.inputs)}")
    for pid, command, start, end in report.processes:
        wall = (float(end) - float(start)) / NS_PER_MS if end and start else 0.0
        lines.append(
            f"- Process {pid}: `{cell(command)}` ({wall:.1f} ms profiled lifetime)"
        )
    for warning in report.warnings:
        lines.append(f"- Warning: {warning}")
    lines.append("")

    lines.append("## GPU time breakdown")
    lines.append("")
    if report.timelines:
        lines.append(
            "The window spans the first to the last GPU operation on each GPU; busy times are "
            "the union of intervals, so overlapping work is not double counted."
        )
        lines.append("")
        lines.append(
            "| GPU / process | Window (ms) | Kernels busy | Copies busy | Any busy | Idle |"
        )
        lines.append("|---|---|---|---|---|---|")
        timelines = sorted(report.timelines, key=lambda t: t.window_ns, reverse=True)
        for tl in timelines[:top]:
            lines.append(
                f"| {cell(tl.label)} | {tl.window_ns / NS_PER_MS:.2f} | "
                f"{pct(tl.kernel_busy_ns, tl.window_ns)} | {pct(tl.copy_busy_ns, tl.window_ns)} | "
                f"{pct(tl.any_busy_ns, tl.window_ns)} | {pct(tl.idle_ns, tl.window_ns)} |"
            )
        if len(timelines) > 1:
            busy = [t.any_busy_ns / t.window_ns for t in timelines if t.window_ns]
            lines.append("")
            lines.append(
                f"Busy share across {len(timelines)} GPU/process pairs: min {min(busy):.0%}, "
                f"median {statistics.median(busy):.0%}, max {max(busy):.0%}. A wide spread "
                "means load imbalance between ranks or devices."
            )
    else:
        lines.append("No GPU kernel or copy activity recorded.")
    lines.append("")

    total = report.kernel_total_ns
    lines.append(f"## Top kernels by total GPU time ({len(report.kernels)} unique)")
    lines.append("")
    if report.kernels:
        lines.append(
            "| # | Kernel | Calls | Total (ms) | % kernel time | Avg (us) | Min (us) | Max (us) "
            "| VGPR | AGPR | SGPR | LDS (B) | Scratch (B) | WG size | Grid (work-items) |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        ranked = sorted(report.kernels.values(), key=lambda k: k.total_ns, reverse=True)
        for rank, k in enumerate(ranked[:top], start=1):
            lines.append(
                f"| {rank} | {cell(short(k.name))} | {k.calls} | {k.total_ns / NS_PER_MS:.3f} | "
                f"{pct(k.total_ns, total)} | {k.avg_ns / NS_PER_US:.1f} | "
                f"{k.min_ns / NS_PER_US:.1f} | {k.max_ns / NS_PER_US:.1f} | {k.vgpr} | {k.agpr} | "
                f"{k.sgpr} | {k.lds} | {k.scratch} | {k.workgroup} | {k.grid} |"
            )
    else:
        lines.append("No kernel dispatches recorded.")
    lines.append("")

    if report.copies:
        lines.append("## Memory copies")
        lines.append("")
        lines.append(
            "| Direction | Calls | Total (MiB) | Total (ms) | Avg size (KiB) | Effective GB/s |"
        )
        lines.append("|---|---|---|---|---|---|")
        for label, s in sorted(
            report.copies.items(), key=lambda kv: kv[1].total_ns, reverse=True
        ):
            avg_kib = s.total_bytes / s.calls / 1024.0 if s.calls else 0.0
            gbps = s.total_bytes / s.total_ns if s.total_ns else 0.0
            lines.append(
                f"| {label} | {s.calls} | {s.total_bytes / BYTES_PER_MIB:.1f} | "
                f"{s.total_ns / NS_PER_MS:.3f} | {avg_kib:.1f} | {gbps:.1f} |"
            )
        lines.append("")

    if report.apis:
        lines.append("## Host API calls by total time")
        lines.append("")
        lines.append("| Domain | Function | Calls | Total (ms) | Avg (us) |")
        lines.append("|---|---|---|---|---|")
        ranked_api = sorted(
            report.apis.items(), key=lambda kv: kv[1].total_ns, reverse=True
        )
        for (category, name), s in ranked_api[:top]:
            avg = s.total_ns / s.calls / NS_PER_US if s.calls else 0.0
            lines.append(
                f"| {category} | {cell(name)} | {s.calls} | {s.total_ns / NS_PER_MS:.3f} | {avg:.1f} |"
            )
        lines.append("")

    if report.markers:
        lines.append("## ROCTx ranges and markers")
        lines.append("")
        lines.append("| Name | Calls | Total (ms) | Avg (us) |")
        lines.append("|---|---|---|---|")
        ranked_markers = sorted(
            report.markers.items(), key=lambda kv: kv[1].total_ns, reverse=True
        )
        for name, s in ranked_markers[:top]:
            avg = s.total_ns / s.calls / NS_PER_US if s.calls else 0.0
            lines.append(
                f"| {cell(name)} | {s.calls} | {s.total_ns / NS_PER_MS:.3f} | {avg:.1f} |"
            )
        lines.append("")

    gap_rows = [(tl, g) for tl in report.timelines for g in tl.gaps]
    if gap_rows:
        gap_rows.sort(key=lambda item: item[1].length_ns, reverse=True)
        lines.append("## Largest GPU idle gaps")
        lines.append("")
        lines.append(
            "| GPU / process | Gap (us) | At (ms into window) | After | Before |"
        )
        lines.append("|---|---|---|---|---|")
        for tl, g in gap_rows[:top]:
            lines.append(
                f"| {cell(tl.label)} | {g.length_ns / NS_PER_US:.1f} | {g.at_ns / NS_PER_MS:.3f} | "
                f"{cell(short(g.before, 50))} | {cell(short(g.after, 50))} |"
            )
        lines.append("")

    lines.append("## Findings")
    lines.append("")
    for item in findings(report) or ["No rule-based findings; inspect the tables above."]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append(
        "Timing note: the first call of each kind in a process (the first hipMalloc or "
        "hipHostMalloc, the first hipMemcpy, the first launch of each kernel) absorbs lazy runtime "
        "initialization; judge steady-state cost from repeated calls."
    )
    return "\n".join(lines)


def build_report(paths: list[Path], top: int) -> Report:
    report = Report()
    databases = discover_databases(paths)
    for db in databases:
        conn = connect(db)
        try:
            source = db.name if len(databases) > 1 else ""
            collect(conn, report, top_gaps=top, source=source)
        finally:
            conn.close()
        report.inputs.append(str(db))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize rocprofv3 rocpd trace databases for agent analysis.",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="rocpd database(s) (*_results.db) or directories to search recursively",
    )
    parser.add_argument(
        "--top", type=int, default=10, help="Rows to show in each table (default: 10)"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write the report here (default: stdout)",
    )
    args = parser.parse_args(argv)

    try:
        report = build_report(args.inputs, top=max(1, args.top))
    except (ValueError, OSError, sqlite3.Error) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    text = render(report, top=max(1, args.top))
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"Wrote report to {args.output}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
