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

"""Summarize rocprofv3 hardware counter output into per-kernel metrics.

Accepts rocpd databases (``*_results.db``, including the ``pass_N/``
directories that multi-pass collection creates) or counter CSV files
(``*counter_collection*.csv`` from ``--output-format csv`` or
``rocpd convert``). Merges every pass, averages each counter per dispatch,
derives L2 hit rate, DRAM bandwidth, instruction mix and DRAM bytes per
memory instruction, and flags common bottleneck signatures. Standard
library only.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# Thresholds mirror the PerfXpert knowledge base
# (experimental/python/perfxpert/perfxpert/knowledge/metric_thresholds.yaml).
L2_HIT_LOW = 0.50  # below this the working set exceeds L2 or locality is poor
HBM_BOUND = 0.70  # achieved/peak DRAM bandwidth at which a kernel is memory-bound
OCCUPANCY_LOW = 25.0  # OccupancyPercent below this cannot hide memory latency
VALU_HEAVY = 70.0  # VALUBusy at or above this is a compute-bound signature
MFMA_HEAVY = 50.0  # MfmaUtil at or above this is a matrix-core-bound signature

# The widest per-lane global access is 16 bytes (dwordx4), so one fully
# coalesced wave instruction with no cache reuse moves at most
# wave_size * 16 bytes from DRAM. More than that means over-fetch.
MAX_BYTES_PER_LANE = 16

# Public peak DRAM bandwidth (TB/s, decimal) from AMD Instinct spec sheets;
# also in perfxpert/knowledge/gpu_specs.yaml. Override with --peak-hbm-tbps.
PEAK_HBM_TBPS = {
    "MI100": 1.23,
    "MI210": 1.6,
    "MI250": 3.2,
    "MI250X": 3.2,
    "MI300A": 5.3,
    "MI300X": 5.3,
    "MI308X": 5.3,
    "MI325X": 6.0,
    "MI350X": 8.0,
    "MI355X": 8.0,
}

# SDK-derived metrics reported as percentages; values above 100 indicate a
# normalization artefact (seen on multi-XCD parts) and are only comparable
# between kernels, not against 100%.
PERCENT_METRICS = {
    "GPUBusy",
    "VALUBusy",
    "SALUBusy",
    "VALUUtilization",
    "MemUnitBusy",
    "MemUnitStalled",
    "WriteUnitStalled",
    "LDSBankConflict",
    "ALUStalledByLDS",
    "L2CacheHit",
    "OccupancyPercent",
    "MfmaUtil",
}

INSTRUCTION_MIX = [
    ("SQ_INSTS_VALU", "VALU"),
    ("SQ_INSTS_SALU", "SALU"),
    ("SQ_INSTS_VMEM_RD", "VMEM read"),
    ("SQ_INSTS_VMEM_WR", "VMEM write"),
    ("SQ_INSTS_LDS", "LDS"),
    ("SQ_INSTS_SMEM", "SMEM"),
]


@dataclass
class Sample:
    source: str
    pid: int
    dispatch_id: int
    kernel: str
    counter: str
    value: float
    duration_ns: float
    vgpr: int = 0
    agpr: int = 0
    sgpr: int = 0
    lds: int = 0
    scratch: int = 0
    workgroup: int = 0
    grid: int = 0


@dataclass
class CounterStat:
    values: list[float] = field(default_factory=list)
    durations: list[float] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return sum(self.values) / len(self.values) if self.values else 0.0

    @property
    def mean_duration_ns(self) -> float:
        return sum(self.durations) / len(self.durations) if self.durations else 0.0


@dataclass
class KernelStat:
    name: str
    counters: dict[str, CounterStat] = field(
        default_factory=lambda: defaultdict(CounterStat)
    )
    dispatch_durations: dict[tuple, float] = field(default_factory=dict)
    dispatches_per_source: dict[str, set] = field(
        default_factory=lambda: defaultdict(set)
    )
    vgpr: int = 0
    agpr: int = 0
    sgpr: int = 0
    lds: int = 0
    scratch: int = 0
    workgroup: int = 0
    grid: int = 0

    def add(self, s: Sample) -> None:
        stat = self.counters[s.counter]
        stat.values.append(s.value)
        stat.durations.append(s.duration_ns)
        self.dispatch_durations[(s.source, s.pid, s.dispatch_id)] = s.duration_ns
        self.dispatches_per_source[s.source].add((s.pid, s.dispatch_id))
        self.vgpr = max(self.vgpr, s.vgpr)
        self.agpr = max(self.agpr, s.agpr)
        self.sgpr = max(self.sgpr, s.sgpr)
        self.lds = max(self.lds, s.lds)
        self.scratch = max(self.scratch, s.scratch)
        self.workgroup = max(self.workgroup, s.workgroup)
        self.grid = max(self.grid, s.grid)

    @property
    def dispatches(self) -> int:
        return max((len(v) for v in self.dispatches_per_source.values()), default=0)

    @property
    def avg_duration_ns(self) -> float:
        durations = list(self.dispatch_durations.values())
        return sum(durations) / len(durations) if durations else 0.0

    @property
    def total_ns(self) -> float:
        return self.avg_duration_ns * self.dispatches

    def mean(self, *names: str) -> CounterStat | None:
        for name in names:
            if name in self.counters and self.counters[name].values:
                return self.counters[name]
        return None


@dataclass
class Report:
    inputs: list[str] = field(default_factory=list)
    gpu_product: str = ""
    gpu_arch: str = ""
    wave_size: int = 0
    peak_tbps: float | None = None
    kernels: dict[str, KernelStat] = field(default_factory=dict)

    def add(self, sample: Sample) -> None:
        self.kernels.setdefault(sample.kernel, KernelStat(sample.kernel)).add(sample)


def discover_inputs(paths: list[Path]) -> list[Path]:
    found: list[Path] = []
    for path in paths:
        if path.is_dir():
            found.extend(sorted(path.rglob("*.db")))
            found.extend(sorted(path.rglob("*counter_collection*.csv")))
        elif path.is_file():
            found.append(path)
        else:
            raise ValueError(f"{path} does not exist")
    if not found:
        raise ValueError(
            "No counter output found. Expected *_results.db (default rocpd output, one per "
            "pass_N/ directory for multi-pass runs) or *counter_collection*.csv files."
        )
    return found


def load_db(path: Path, report: Report) -> int:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        views = {
            r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='view'")
        }
        if "counters_collection" not in views:
            raise ValueError(f"{path} is not a rocprofv3 rocpd database")
        for name, product, extdata in conn.execute(
            "SELECT name, product_name, extdata FROM rocpd_info_agent WHERE type='GPU' "
            "ORDER BY absolute_index"
        ):
            if not report.gpu_arch:
                report.gpu_arch = name or ""
                report.gpu_product = product or ""
                try:
                    report.wave_size = int(
                        json.loads(extdata or "{}").get("wave_front_size", 0)
                    )
                except (ValueError, TypeError):
                    report.wave_size = 0
        rows = conn.execute(
            "SELECT pid, dispatch_id, kernel_name, counter_name, value, duration, vgpr_count, "
            "accum_vgpr_count, sgpr_count, lds_block_size, scratch_size, workgroup_size, grid_size "
            "FROM counters_collection"
        ).fetchall()
    finally:
        conn.close()
    for row in rows:
        report.add(
            Sample(
                source=str(path),
                pid=int(row[0] or 0),
                dispatch_id=int(row[1] or 0),
                kernel=row[2],
                counter=row[3],
                value=float(row[4] or 0.0),
                duration_ns=float(row[5] or 0.0),
                vgpr=int(row[6] or 0),
                agpr=int(row[7] or 0),
                sgpr=int(row[8] or 0),
                lds=int(row[9] or 0),
                scratch=int(row[10] or 0),
                workgroup=int(row[11] or 0),
                grid=int(row[12] or 0),
            )
        )
    return len(rows)


def lower_keys(row: dict) -> dict:
    return {k.lower(): v for k, v in row.items() if k is not None}


def load_agent_info_csv(csv_path: Path, report: Report) -> None:
    if report.gpu_arch:
        return
    for candidate in sorted(csv_path.parent.glob("*agent_info.csv")):
        with candidate.open(newline="", encoding="utf-8") as handle:
            for raw in csv.DictReader(handle):
                row = lower_keys(raw)
                if row.get("agent_type") == "GPU":
                    report.gpu_arch = row.get("name", "")
                    report.gpu_product = row.get("product_name", "")
                    try:
                        report.wave_size = int(row.get("wave_front_size") or 0)
                    except ValueError:
                        report.wave_size = 0
                    return


def load_csv(path: Path, report: Report) -> int:
    count = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row = lower_keys(raw)
            if "counter_name" not in row:
                raise ValueError(
                    f"{path} has no Counter_Name column; is it a counter collection CSV?"
                )
            start = float(row.get("start_timestamp") or 0)
            end = float(row.get("end_timestamp") or 0)
            report.add(
                Sample(
                    source=str(path),
                    pid=int(row.get("process_id") or 0),
                    dispatch_id=int(row.get("dispatch_id") or 0),
                    kernel=row.get("kernel_name", ""),
                    counter=row["counter_name"],
                    value=float(row.get("counter_value") or 0.0),
                    duration_ns=max(0.0, end - start),
                    vgpr=int(float(row.get("vgpr_count") or 0)),
                    agpr=int(float(row.get("accum_vgpr_count") or 0)),
                    sgpr=int(float(row.get("sgpr_count") or 0)),
                    lds=int(float(row.get("lds_block_size") or 0)),
                    scratch=int(float(row.get("scratch_size") or 0)),
                    workgroup=int(float(row.get("workgroup_size") or 0)),
                    grid=int(float(row.get("grid_size") or 0)),
                )
            )
            count += 1
    load_agent_info_csv(path, report)
    return count


def resolve_peak(product: str, override: float | None) -> float | None:
    if override:
        return override
    match = re.search(r"MI\d{3}[A-Z]?", product or "")
    if not match:
        return None
    token = match.group(0)
    return PEAK_HBM_TBPS.get(token) or PEAK_HBM_TBPS.get(token[:5])


def default_wave_size(arch: str) -> int:
    return 64 if arch.startswith("gfx9") else 32


@dataclass
class Derived:
    rows: list[tuple[str, str, str]] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    l2_hit: float | None = None
    dram_gbps: float | None = None
    peak_fraction: float | None = None


def derive(k: KernelStat, report: Report) -> Derived:
    d = Derived()
    wave_size = report.wave_size or default_wave_size(report.gpu_arch)

    # CDNA names the L2 block TCC; RDNA names it GL2C.
    hit = k.mean("TCC_HIT_sum", "TCC_HIT", "GL2C_HIT_sum", "GL2C_HIT")
    miss = k.mean("TCC_MISS_sum", "TCC_MISS", "GL2C_MISS_sum", "GL2C_MISS")
    sdk_hit = k.mean("L2CacheHit")
    if hit and miss and (hit.mean + miss.mean) > 0:
        d.l2_hit = hit.mean / (hit.mean + miss.mean)
        d.rows.append(("L2 hit rate", f"{d.l2_hit:.1%}", "L2 hits / (hits + misses)"))
    elif sdk_hit:
        d.l2_hit = sdk_hit.mean / 100.0
        d.rows.append(("L2 hit rate", f"{d.l2_hit:.1%}", "SDK-derived L2CacheHit"))
    if d.l2_hit is not None and d.l2_hit < L2_HIT_LOW:
        d.flags.append(
            f"L2 hit rate {d.l2_hit:.0%} is below {L2_HIT_LOW:.0%}: the working set exceeds L2 "
            "or accesses lack locality. Streaming kernels that touch each byte once miss by "
            "design; otherwise tile/block for reuse or fix the access order."
        )

    fetch = k.mean("FETCH_SIZE")
    write = k.mean("WRITE_SIZE")
    gbps_parts = []
    dram_bytes = 0.0
    for label, stat in (("read", fetch), ("write", write)):
        if stat is None:
            continue
        nbytes = stat.mean * 1024.0
        dram_bytes += nbytes
        d.rows.append(
            (
                f"DRAM {label} per dispatch",
                f"{nbytes / 2**20:.2f} MiB",
                f"{'FETCH' if label == 'read' else 'WRITE'}_SIZE (KiB) x 1024",
            )
        )
        if stat.mean_duration_ns > 0:
            gbps = nbytes / stat.mean_duration_ns
            gbps_parts.append(gbps)
            d.rows.append(
                (
                    f"DRAM {label} bandwidth",
                    f"{gbps:.1f} GB/s",
                    "bytes / dispatch duration of the same pass",
                )
            )
    if gbps_parts:
        d.dram_gbps = sum(gbps_parts)
        note = "read + write"
        if fetch is None or write is None:
            note += " (only one side collected)"
        value = f"{d.dram_gbps:.1f} GB/s"
        if report.peak_tbps:
            d.peak_fraction = d.dram_gbps / (report.peak_tbps * 1000.0)
            value += f" ({d.peak_fraction:.0%} of {report.peak_tbps:g} TB/s peak)"
        d.rows.append(("DRAM bandwidth", value, note))
        if d.peak_fraction is not None and d.peak_fraction >= HBM_BOUND:
            d.flags.append(
                f"DRAM bandwidth is {d.peak_fraction:.0%} of peak: memory-bandwidth-bound. Speedups "
                "come from moving fewer bytes (fusion, reuse, smaller data types), not more compute."
            )

    waves = k.mean("SQ_WAVES")
    if waves and waves.mean > 0:
        mix = []
        for counter, label in INSTRUCTION_MIX:
            stat = k.mean(counter)
            if stat:
                mix.append(f"{label} {stat.mean / waves.mean:.1f}")
        if mix:
            d.rows.append(
                ("Instructions per wave", ", ".join(mix), "SQ_INSTS_* / SQ_WAVES")
            )

    vmem_rd = k.mean("SQ_INSTS_VMEM_RD")
    vmem_wr = k.mean("SQ_INSTS_VMEM_WR")
    vmem = (vmem_rd.mean if vmem_rd else 0.0) + (vmem_wr.mean if vmem_wr else 0.0)
    if fetch and write and vmem > 0:
        per_inst = dram_bytes / vmem
        ceiling = wave_size * MAX_BYTES_PER_LANE
        d.rows.append(
            (
                "DRAM bytes per VMEM wave-instruction",
                f"{per_inst:.0f} B",
                f"coalesced ideal = wave size {wave_size} x bytes per lane (4 B -> {wave_size * 4} B)",
            )
        )
        if per_inst > ceiling:
            d.flags.append(
                f"{per_inst:.0f} DRAM bytes per memory instruction exceeds the {ceiling} B a fully "
                f"coalesced {MAX_BYTES_PER_LANE}-byte-per-lane access can move: accesses are strided "
                "or scattered, so each lane pulls its own cache line. Make consecutive lanes touch "
                "consecutive addresses (transpose via LDS, change the data layout, or swap loop order)."
            )

    for name in sorted(PERCENT_METRICS):
        stat = k.mean(name)
        if stat is None:
            continue
        if stat.mean > 100.0:
            d.flags.append(
                f"{name} reads {stat.mean:.0f}% (> 100%). This SDK-derived percentage is not "
                "normalized for this GPU; compare it between kernels rather than against 100%."
            )

    occupancy = k.mean("OccupancyPercent")
    if occupancy and occupancy.mean < OCCUPANCY_LOW:
        d.flags.append(
            f"OccupancyPercent {occupancy.mean:.0f}% is below {OCCUPANCY_LOW:.0f}%: too few waves to "
            "hide latency. Check VGPR/AGPR/LDS usage per workgroup and whether the grid fills every "
            "CU; very short kernels also read low because of launch ramp-up."
        )
    valu = k.mean("VALUBusy")
    if valu and valu.mean >= VALU_HEAVY:
        d.flags.append(
            f"VALUBusy {valu.mean:.0f}%: vector ALU dominates. Reduce instruction count (strength "
            "reduction, fast-math intrinsics, lower precision) or move matrix math to MFMA."
        )
    mfma = k.mean("MfmaUtil")
    if mfma and mfma.mean >= MFMA_HEAVY:
        d.flags.append(
            f"MfmaUtil {mfma.mean:.0f}%: matrix cores are the limiter; gains need lower precision "
            "formats or less total matrix work."
        )
    lds_conflict = k.mean("LDSBankConflict")
    if lds_conflict and lds_conflict.mean > 0:
        d.flags.append(
            f"LDSBankConflict {lds_conflict.mean:.1f}%: LDS accesses collide on banks; pad shared "
            "arrays (for example [N][N+1]) or swizzle indices."
        )
    if k.scratch > 0:
        d.flags.append(
            f"Scratch {k.scratch} B per work-item: register spills or stack arrays. Reduce VGPR "
            "pressure or avoid dynamically indexed local arrays."
        )
    return d


def cell(text: str) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")


def short(name: str, width: int = 70) -> str:
    name = " ".join(name.split())
    return name if len(name) <= width else name[: width - 3] + "..."


def fmt(value: float) -> str:
    if value == int(value) and abs(value) < 1e15:
        return f"{int(value):,}"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:.3g}"


def render(report: Report, top: int, kernel_filter: str | None) -> str:
    lines = ["# rocprofv3 Counter Report", ""]
    lines.append(f"- Inputs: {', '.join(report.inputs)}")
    gpu = report.gpu_product or "unknown GPU"
    if report.gpu_arch:
        gpu += f" ({report.gpu_arch})"
    lines.append(f"- GPU: {gpu}")
    if report.peak_tbps:
        lines.append(
            f"- Peak DRAM bandwidth used for % of peak: {report.peak_tbps:g} TB/s"
        )
    else:
        lines.append(
            "- Peak DRAM bandwidth unknown; pass --peak-hbm-tbps to get % of peak"
        )
    counters = sorted({c for k in report.kernels.values() for c in k.counters})
    lines.append(f"- Counters: {', '.join(counters)}")
    lines.append(
        "- Durations come from the counter-collection run, where kernels are serialized; use a "
        "trace run for timeline and overlap questions."
    )
    lines.append("")

    kernels = sorted(report.kernels.values(), key=lambda k: k.total_ns, reverse=True)
    if kernel_filter:
        kernels = [k for k in kernels if kernel_filter in k.name]
        if not kernels:
            lines.append(f"No kernels matched `{kernel_filter}`.")
            return "\n".join(lines)
    total_ns = sum(k.total_ns for k in kernels)
    derived = {k.name: derive(k, report) for k in kernels}

    lines.append("## Summary")
    lines.append("")
    lines.append(
        "| # | Kernel | Dispatches | Avg (us) | % time | L2 hit | DRAM GB/s | Occupancy % | Flags |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for rank, k in enumerate(kernels[:top], start=1):
        d = derived[k.name]
        occ = k.mean("OccupancyPercent")
        bw = "" if d.dram_gbps is None else f"{d.dram_gbps:.0f}"
        if d.peak_fraction is not None:
            bw += f" ({d.peak_fraction:.0%})"
        lines.append(
            f"| {rank} | {cell(short(k.name))} | {k.dispatches} | {k.avg_duration_ns / 1000:.1f} | "
            f"{100 * k.total_ns / total_ns if total_ns else 0:.1f}% | "
            f"{'' if d.l2_hit is None else f'{d.l2_hit:.0%}'} | {bw} | "
            f"{'' if occ is None else f'{occ.mean:.0f}'} | {len(d.flags)} |"
        )
    lines.append("")

    for k in kernels[:top]:
        d = derived[k.name]
        lines.append(f"## {cell(short(k.name, 120))}")
        lines.append("")
        lines.append(
            f"- Dispatches: {k.dispatches}; avg duration {k.avg_duration_ns / 1000:.1f} us; "
            f"VGPR {k.vgpr}, AGPR {k.agpr}, SGPR {k.sgpr}; LDS {k.lds} B; scratch {k.scratch} B; "
            f"workgroup {k.workgroup}; grid {k.grid} work-items"
        )
        lines.append("")
        lines.append("| Counter | Mean per dispatch | Samples |")
        lines.append("|---|---|---|")
        for name in sorted(k.counters):
            stat = k.counters[name]
            lines.append(f"| {name} | {fmt(stat.mean)} | {len(stat.values)} |")
        lines.append("")
        if d.rows:
            lines.append("| Derived metric | Value | How |")
            lines.append("|---|---|---|")
            for metric, value, how in d.rows:
                lines.append(f"| {metric} | {value} | {cell(how)} |")
            lines.append("")
        if d.flags:
            lines.append("Flags:")
            lines.append("")
            for flag in d.flags:
                lines.append(f"- {flag}")
            lines.append("")
    return "\n".join(lines)


def build_report(paths: list[Path], peak_override: float | None) -> Report:
    report = Report()
    for path in discover_inputs(paths):
        if path.suffix == ".db":
            count = load_db(path, report)
        elif path.suffix == ".csv":
            count = load_csv(path, report)
        else:
            raise ValueError(f"Unsupported input {path}; expected .db or .csv")
        if count:
            report.inputs.append(str(path))
    if not report.kernels:
        raise ValueError(
            "Inputs contain no counter records. Check that --pmc (or an input file with pmc) was "
            "given, that the kernel filters matched something, and that the run completed."
        )
    report.peak_tbps = resolve_peak(report.gpu_product, peak_override)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize rocprofv3 counter collection output."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="rocpd .db files, counter CSV files, or directories to search recursively",
    )
    parser.add_argument(
        "--top", type=int, default=10, help="Kernels to report (default: 10)"
    )
    parser.add_argument(
        "--kernel", default=None, help="Only kernels whose name contains this text"
    )
    parser.add_argument(
        "--peak-hbm-tbps",
        type=float,
        default=None,
        help="Peak DRAM bandwidth in TB/s (default: looked up from the GPU product name)",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None, help="Write report here"
    )
    args = parser.parse_args(argv)

    try:
        report = build_report(args.inputs, args.peak_hbm_tbps)
    except (ValueError, OSError, sqlite3.Error) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    text = render(report, top=max(1, args.top), kernel_filter=args.kernel)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"Wrote report to {args.output}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
