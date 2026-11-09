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

"""Rank instruction and source-line hotspots from rocprofv3 thread trace stats.

``rocprofv3 --att`` decodes each traced dispatch into a
``stats_ui_output_agent_<agent>_dispatch_<id>.csv`` file with one row per ISA
instruction (CodeObj, Vaddr, Instruction, Hitcount, Latency, Stall, Idle,
Source). This script ranks instructions and source lines by latency, breaks
latency down by instruction class (global memory, waits, LDS, VALU, MFMA, ...),
and flags the dominant bottleneck. Standard library only.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# A class that holds at least a fifth of all latency is called out as a
# finding; below that no single class dominates the kernel.
DOMINANT_SHARE = 0.20
# Idle cycles above this share of latency + idle mean the wave often had
# nothing ready to issue (dependencies, instruction cache misses, arbitration).
IDLE_HEAVY = 0.30
# A vmcnt wait that averages fewer cycles per execution than a vector L1 hit
# (roughly 120 cycles on CDNA3) almost never waited on memory; it typically
# sits in a hot loop and its stall is issue contention with other waves.
MEMORY_WAIT_MIN_CYCLES = 100

VMEM_WAITS = (
    "wait: vector memory (vmcnt)",
    "wait: vector memory + scalar/LDS",
    "wait: vector store (vscnt)",
)


@dataclass
class Instruction:
    kernel: str
    code_object: str
    vaddr: int
    text: str
    hitcount: int
    latency: int
    stall: int
    idle: int
    source: str

    @property
    def mnemonic(self) -> str:
        return self.text.split()[0] if self.text else ""

    @property
    def location(self) -> str:
        """Outermost frame of the inlining chain: the user's call site."""
        if not self.source:
            return ""
        return self.source.split(" -> ")[-1].strip()


@dataclass
class Bucket:
    latency: int = 0
    stall: int = 0
    idle: int = 0
    hits: int = 0
    instructions: int = 0

    def add(self, inst: Instruction) -> None:
        self.latency += inst.latency
        self.stall += inst.stall
        self.idle += inst.idle
        self.hits += inst.hitcount
        self.instructions += 1


@dataclass
class Dispatch:
    path: str
    kernels: list[str] = field(default_factory=list)
    instructions: list[Instruction] = field(default_factory=list)

    @property
    def executed(self) -> list[Instruction]:
        return [i for i in self.instructions if i.hitcount > 0]

    @property
    def total(self) -> Bucket:
        bucket = Bucket()
        for inst in self.executed:
            bucket.add(inst)
        return bucket


def classify(text: str) -> str:
    mnemonic = text.split()[0] if text else ""
    if mnemonic.startswith("s_waitcnt"):
        if "vscnt" in mnemonic or "vscnt" in text:
            return "wait: vector store (vscnt)"
        has_vm = "vmcnt" in text
        has_lgkm = "lgkmcnt" in text
        if has_vm and has_lgkm:
            return "wait: vector memory + scalar/LDS"
        if has_vm:
            return "wait: vector memory (vmcnt)"
        if has_lgkm:
            return "wait: scalar memory / LDS (lgkmcnt)"
        return "wait: other"
    if re.match(r"(global|buffer|flat|scratch)_atomic", mnemonic):
        return "VMEM atomic"
    if re.match(r"(global|buffer|flat|scratch)_load", mnemonic):
        return "VMEM load"
    if re.match(r"(global|buffer|flat|scratch)_store", mnemonic):
        return "VMEM store"
    if mnemonic.startswith("ds_"):
        return "LDS"
    if mnemonic.startswith(("s_load", "s_buffer_load", "s_scratch_load", "s_store")):
        return "SMEM (scalar load/store)"
    if mnemonic.startswith(("v_mfma", "v_smfmac", "v_wmma", "v_swmmac")):
        return "MFMA / WMMA"
    if mnemonic.startswith("v_"):
        return "VALU"
    if mnemonic.startswith("s_barrier"):
        return "barrier"
    if mnemonic.startswith(("s_cbranch", "s_branch", "s_setpc", "s_swappc", "s_getpc")):
        return "branch"
    if mnemonic.startswith(("s_nop", "s_sleep")):
        return "nop / sleep"
    if mnemonic.startswith("s_endpgm"):
        return "end of program"
    if mnemonic.startswith("s_"):
        return "SALU"
    return "other"


def discover(paths: list[Path]) -> list[Path]:
    found: list[Path] = []
    for path in paths:
        if path.is_dir():
            found.extend(sorted(path.rglob("stats_*.csv")))
        elif path.is_file():
            found.append(path)
        else:
            raise ValueError(f"{path} does not exist")
    if not found:
        raise ValueError(
            "No stats_*.csv files found. rocprofv3 --att writes them to the output directory "
            "after decoding; check that the decoder ran and that a kernel was actually traced."
        )
    return found


def as_int(value: str | None) -> int:
    try:
        return int(float(value or 0))
    except ValueError:
        return 0


def load(path: Path) -> Dispatch:
    dispatch = Dispatch(path=str(path))
    kernel = ""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"Instruction", "Hitcount", "Latency"}
        if not reader.fieldnames or not required <= set(reader.fieldnames):
            raise ValueError(
                f"{path} is not a thread trace stats CSV (columns: {reader.fieldnames})"
            )
        for row in reader:
            text = (row.get("Instruction") or "").strip()
            if text.startswith(";"):
                kernel = (row.get("Source") or text.lstrip("; ")).strip()
                dispatch.kernels.append(kernel)
                continue
            dispatch.instructions.append(
                Instruction(
                    kernel=kernel,
                    code_object=row.get("CodeObj", ""),
                    vaddr=as_int(row.get("Vaddr")),
                    text=text,
                    hitcount=as_int(row.get("Hitcount")),
                    latency=as_int(row.get("Latency")),
                    stall=as_int(row.get("Stall")),
                    idle=as_int(row.get("Idle")),
                    source=(row.get("Source") or "").strip(),
                )
            )
    return dispatch


def cell(text: str) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")


def share(part: int, whole: int) -> str:
    return f"{100.0 * part / whole:.1f}%" if whole else "0.0%"


def findings(dispatch: Dispatch, classes: dict[str, Bucket]) -> list[str]:
    total = dispatch.total
    out: list[str] = []
    if not dispatch.executed:
        return [
            "No traced instruction executed. The target CU (--att-target-cu) on the traced "
            "shader engines got no waves: launch more waves, widen --att-shader-engine-mask, "
            "pick another --att-target-cu, or restrict CUs with HSA_CU_MASK."
        ]

    memory = sum(b.latency for name, b in classes.items() if name.startswith("VMEM"))
    satisfied_waits = []
    for inst in dispatch.executed:
        if classify(inst.text) not in VMEM_WAITS:
            continue
        if inst.latency / inst.hitcount >= MEMORY_WAIT_MIN_CYCLES:
            memory += inst.latency
        else:
            satisfied_waits.append(inst)
    hot_waits = sum(i.latency for i in satisfied_waits)
    if total.latency and hot_waits / total.latency >= DOMINANT_SHARE:
        worst = max(satisfied_waits, key=lambda i: i.latency)
        out.append(
            f"`{worst.text}` at 0x{worst.vaddr:x} holds {share(hot_waits, total.latency)} of latency but "
            f"averages {worst.latency / worst.hitcount:.0f} cycles over {worst.hitcount:,} executions: "
            "it sits in a hot loop and is almost always already satisfied, so its stall is issue "
            "contention with other waves on the SIMD, not memory. Treat the loop body (VALU/SALU) "
            "as the limiter."
        )
    if total.latency and memory / total.latency >= DOMINANT_SHARE:
        out.append(
            f"Global memory accounts for {share(memory, total.latency)} of latency (VMEM issue "
            "stalls plus vmcnt waits). If counter collection shows DRAM bytes per memory instruction "
            "well above the coalesced ideal, fix the access pattern; otherwise raise memory-level "
            "parallelism (more waves, more independent loads in flight, prefetch into registers or LDS)."
        )
    for name, label, advice in (
        (
            "wait: scalar memory / LDS (lgkmcnt)",
            "Scalar-memory/LDS waits",
            "Kernel-argument and constant loads, or LDS reads, are not hidden; hoist scalar loads earlier and interleave LDS reads with math.",
        ),
        (
            "LDS",
            "LDS instructions",
            "Check bank conflicts (pad or swizzle shared arrays) and use wider ds_read_b128/ds_write_b128 where alignment allows.",
        ),
        (
            "barrier",
            "Barriers",
            "Waves wait on each other at __syncthreads(); balance work across the workgroup or reduce the number of barriers per tile.",
        ),
        (
            "MFMA / WMMA",
            "Matrix instructions",
            "Matrix cores are busy; check whether operands arrive in time (stalls before MFMA) and whether a lower precision format is acceptable.",
        ),
        (
            "VALU",
            "Vector ALU",
            "Reduce instruction count: strength-reduce index math, use fast-math intrinsics, or hoist invariant work out of loops.",
        ),
        (
            "SALU",
            "Scalar ALU",
            "Scalar index or loop-control math is heavy; simplify addressing or unroll.",
        ),
        (
            "branch",
            "Branches",
            "Divergent or frequent branches; flatten control flow or make the branch uniform per wave.",
        ),
    ):
        bucket = classes.get(name)
        if bucket and total.latency and bucket.latency / total.latency >= DOMINANT_SHARE:
            out.append(
                f"{label} account for {share(bucket.latency, total.latency)} of latency. {advice}"
            )
    wave_time = total.latency + total.idle
    if wave_time and total.idle / wave_time >= IDLE_HEAVY:
        worst = max(dispatch.executed, key=lambda i: i.idle)
        text = (
            f"Idle cycles are {share(total.idle, wave_time)} of latency + idle: waves often had no "
            "instruction ready (losing arbitration to another wave, a register dependency on an "
            "earlier result, or an instruction cache miss). The largest idle sits on "
            f"`{worst.text}` ({share(worst.idle, total.idle)} of idle)."
        )
        if worst.mnemonic == "s_endpgm":
            text += (
                " Idle before s_endpgm is waves waiting for outstanding memory operations "
                "(usually stores) to drain before they can exit, so count it as memory cost."
            )
        out.append(text)
    if all(not inst.source for inst in dispatch.executed):
        out.append(
            "No source attribution: rebuild the application with -g (keep -O3) to map ISA to source lines."
        )
    return out


def render(dispatches: list[Dispatch], top: int) -> str:
    lines = ["# rocprofv3 Thread Trace Report", ""]
    lines.append(
        "Latency = cycles from issue to completion summed over traced waves (stall + issue on "
        "gfx9, stall + execute on gfx10+). Stall = cycles the pipe could not issue. Idle = gap "
        "since the previous instruction finished. Only waves on the traced CU are counted, so "
        "treat values as a representative sample."
    )
    lines.append("")
    for dispatch in dispatches:
        total = dispatch.total
        executed = dispatch.executed
        lines.append(f"## {Path(dispatch.path).name}")
        lines.append("")
        if dispatch.kernels:
            lines.append(
                f"- Kernels: {', '.join(f'`{cell(k)}`' for k in dict.fromkeys(dispatch.kernels))}"
            )
        traced_waves = max((i.hitcount for i in executed[:1]), default=0)
        lines.append(f"- Traced waves (hitcount of first instruction): {traced_waves}")
        lines.append(
            f"- Instructions executed: {len(executed)} of {len(dispatch.instructions)}; total "
            f"latency {total.latency:,} cycles (stall {share(total.stall, total.latency)} of it); "
            f"idle {total.idle:,} cycles ({share(total.idle, total.latency + total.idle)} of latency + idle)"
        )
        lines.append("")

        classes: dict[str, Bucket] = defaultdict(Bucket)
        for inst in executed:
            classes[classify(inst.text)].add(inst)
        if classes:
            lines.append("### Latency by instruction class")
            lines.append("")
            lines.append(
                "| Class | Latency share | Stall share of class | Latency/hit | Instructions | Hits |"
            )
            lines.append("|---|---|---|---|---|---|")
            for name, b in sorted(
                classes.items(), key=lambda kv: kv[1].latency, reverse=True
            ):
                per_hit = b.latency / b.hits if b.hits else 0
                lines.append(
                    f"| {name} | {share(b.latency, total.latency)} | {share(b.stall, b.latency)} | "
                    f"{per_hit:.0f} | {b.instructions} | {b.hits:,} |"
                )
            lines.append("")

        lines.append(f"### Top {top} instructions by latency")
        lines.append("")
        lines.append(
            "| # | Vaddr | Instruction | Latency share | Latency/hit | Stall | Idle | Hits | Source |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for rank, inst in enumerate(
            sorted(executed, key=lambda i: i.latency, reverse=True)[:top], start=1
        ):
            per_hit = inst.latency / inst.hitcount if inst.hitcount else 0
            lines.append(
                f"| {rank} | 0x{inst.vaddr:x} | `{cell(inst.text)}` | {share(inst.latency, total.latency)} | "
                f"{per_hit:.0f} | {inst.stall:,} | {inst.idle:,} | {inst.hitcount:,} | {cell(inst.location)} |"
            )
        lines.append("")

        idle_ranked = [
            i for i in sorted(executed, key=lambda i: i.idle, reverse=True) if i.idle > 0
        ]
        if idle_ranked:
            idle_top = min(top, 5)
            lines.append(f"### Top {idle_top} instructions by idle")
            lines.append("")
            lines.append("| # | Vaddr | Instruction | Idle share | Idle | Source |")
            lines.append("|---|---|---|---|---|---|")
            for rank, inst in enumerate(idle_ranked[:idle_top], start=1):
                lines.append(
                    f"| {rank} | 0x{inst.vaddr:x} | `{cell(inst.text)}` | {share(inst.idle, total.idle)} | "
                    f"{inst.idle:,} | {cell(inst.location)} |"
                )
            lines.append("")

        by_line: dict[str, Bucket] = defaultdict(Bucket)
        for inst in executed:
            by_line[inst.location or "(no source)"].add(inst)
        lines.append(f"### Top {top} source lines by latency")
        lines.append("")
        lines.append(
            "| # | Source line | Latency share | Stall share of line | Instructions |"
        )
        lines.append("|---|---|---|---|---|")
        ranked_lines = sorted(
            by_line.items(), key=lambda kv: kv[1].latency, reverse=True
        )[:top]
        for rank, (loc, b) in enumerate(ranked_lines, start=1):
            lines.append(
                f"| {rank} | {cell(loc)} | {share(b.latency, total.latency)} | "
                f"{share(b.stall, b.latency)} | {b.instructions} |"
            )
        lines.append("")

        lines.append("### Findings")
        lines.append("")
        for item in findings(dispatch, classes) or [
            "No single instruction class dominates."
        ]:
            lines.append(f"- {item}")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize rocprofv3 thread trace stats CSVs."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="stats_*.csv files or the rocprofv3 --att output directory",
    )
    parser.add_argument(
        "--top", type=int, default=15, help="Rows per table (default: 15)"
    )
    parser.add_argument(
        "--kernel",
        default=None,
        help="Only dispatches whose kernel name contains this text",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None, help="Write report here"
    )
    args = parser.parse_args(argv)

    try:
        dispatches = [load(p) for p in discover(args.inputs)]
    except (ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    if args.kernel:
        dispatches = [d for d in dispatches if any(args.kernel in k for k in d.kernels)]
        if not dispatches:
            print(
                f"error: no traced dispatch matched kernel filter {args.kernel!r}",
                file=sys.stderr,
            )
            return 1

    text = render(dispatches, top=max(1, args.top))
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"Wrote report to {args.output}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
