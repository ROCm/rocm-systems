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

"""Parse rocprofv3 PC sampling output into a human-readable hotspot report.

Maps samples to kernels, ranks top instruction locations, and summarizes stall
reasons for stochastic sampling. Accepts JSON results, PC sampling CSV (with
optional kernel trace CSV), or a directory containing those files.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

STALL_PREFIX = "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_"
INST_TYPE_PREFIX = "ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_"


def shorten_enum(value: str | None, prefix: str) -> str:
    if not value:
        return ""
    if value.startswith(prefix):
        return value[len(prefix) :]
    return value


@dataclass
class Sample:
    dispatch_id: int
    instruction: str
    source: str
    code_object_id: int | None = None
    offset: int | None = None
    issued: bool | None = None
    stall_reason: str = ""
    inst_type: str = ""


@dataclass
class LocationStats:
    instruction: str
    source: str
    code_object_id: int | None
    offset: int | None
    count: int = 0
    issued: int = 0
    stalled: int = 0
    stall_reasons: Counter[str] = field(default_factory=Counter)

    def add(self, sample: Sample) -> None:
        self.count += 1
        if sample.issued is True:
            self.issued += 1
        elif sample.issued is False:
            self.stalled += 1
            if sample.stall_reason:
                self.stall_reasons[sample.stall_reason] += 1


@dataclass
class KernelStats:
    name: str
    dispatch_ids: set[int] = field(default_factory=set)
    locations: dict[tuple[Any, ...], LocationStats] = field(default_factory=dict)
    stall_reasons: Counter[str] = field(default_factory=Counter)

    @property
    def sample_count(self) -> int:
        return sum(loc.count for loc in self.locations.values())

    def add(self, sample: Sample) -> None:
        self.dispatch_ids.add(sample.dispatch_id)
        key = (
            sample.code_object_id,
            sample.offset,
            sample.instruction,
            sample.source,
        )
        if key not in self.locations:
            self.locations[key] = LocationStats(
                instruction=sample.instruction,
                source=sample.source,
                code_object_id=sample.code_object_id,
                offset=sample.offset,
            )
        self.locations[key].add(sample)
        if sample.issued is False and sample.stall_reason:
            self.stall_reasons[sample.stall_reason] += 1


@dataclass
class Report:
    method: str
    source_files: list[str]
    kernels: dict[str, KernelStats] = field(default_factory=dict)
    unmapped_dispatch_ids: set[int] = field(default_factory=set)
    undecodable_samples: int = 0

    @property
    def total_samples(self) -> int:
        return sum(k.sample_count for k in self.kernels.values())

    def add(self, sample: Sample, kernel_name: str | None) -> None:
        if is_undecodable_sample(sample):
            self.undecodable_samples += 1
        name = kernel_name or f"<unknown dispatch {sample.dispatch_id}>"
        if kernel_name is None:
            self.unmapped_dispatch_ids.add(sample.dispatch_id)
        if name not in self.kernels:
            self.kernels[name] = KernelStats(name=name)
        self.kernels[name].add(sample)


def is_undecodable_sample(sample: Sample) -> bool:
    return sample.source.startswith("unrecognized code object")


def parse_code_object_id(pc: dict[str, Any]) -> int | None:
    # code_object_id 0 is valid; do not use truthiness checks here.
    if "code_object_id" not in pc:
        return None
    return int(pc["code_object_id"])


def load_tool_record(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "rocprofiler-sdk-tool" in data:
        records = data["rocprofiler-sdk-tool"]
        if isinstance(records, list):
            if not records:
                raise ValueError(f"No tool records in {path}")
            return records[0]
        return records
    return data


def build_dispatch_map_from_json(tool: dict[str, Any]) -> dict[int, str]:
    kernel_symbols = tool.get("kernel_symbols", [])

    def kernel_name(kernel_id: int) -> str:
        if kernel_id < len(kernel_symbols):
            sym = kernel_symbols[kernel_id]
            return (
                sym.get("formatted_kernel_name")
                or sym.get("kernel_name")
                or f"kernel_{kernel_id}"
            )
        return f"kernel_{kernel_id}"

    dispatch_map: dict[int, str] = {}
    for dispatch in tool.get("buffer_records", {}).get("kernel_dispatch", []):
        info = dispatch["dispatch_info"]
        dispatch_map[int(info["dispatch_id"])] = kernel_name(int(info["kernel_id"]))
    return dispatch_map


def build_dispatch_map_from_csv(path: Path) -> dict[int, str]:
    dispatch_map: dict[int, str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            dispatch_id = int(row["Dispatch_Id"])
            dispatch_map[dispatch_id] = (
                row.get("Kernel_Name") or f"kernel_{row.get('Kernel_Id', '?')}"
            )
    return dispatch_map


def find_kernel_trace_csv(pc_csv: Path) -> Path | None:
    """Find the kernel trace CSV that pairs with a PC sampling CSV."""
    stem = pc_csv.stem
    for marker in ("pc_sampling_stochastic", "pc_sampling_host_trap"):
        if marker in stem:
            candidate = pc_csv.with_name(
                stem.replace(marker, "kernel_trace") + pc_csv.suffix
            )
            if candidate.exists():
                return candidate
    prefix, sep, _ = stem.partition("_pc_sampling_")
    if sep:
        candidate = pc_csv.with_name(f"{prefix}_kernel_trace{pc_csv.suffix}")
        if candidate.exists():
            return candidate
    return None


def dispatch_map_covers_samples(
    dispatch_map: dict[int, str], samples: list[Sample]
) -> bool:
    if not dispatch_map or not samples:
        return False
    sample_ids = {sample.dispatch_id for sample in samples}
    return bool(sample_ids & dispatch_map.keys())


def detect_pc_sampling_key(tool: dict[str, Any]) -> str:
    buffer_records = tool.get("buffer_records", {})
    stochastic = buffer_records.get("pc_sample_stochastic") or []
    host_trap = buffer_records.get("pc_sample_host_trap") or []
    if stochastic:
        return "stochastic"
    if host_trap:
        return "host_trap"
    raise ValueError("JSON file contains no PC sampling records")


def iter_json_samples(tool: dict[str, Any], method: str) -> Iterable[Sample]:
    strings = tool.get("strings", {})
    instructions = strings.get("pc_sample_instructions", [])
    comments = strings.get("pc_sample_comments", [])
    key = f"pc_sample_{method}"
    for entry in tool.get("buffer_records", {}).get(key, []):
        record = entry["record"]
        inst_index = entry.get("inst_index", -1)
        if inst_index is not None and inst_index >= 0 and inst_index < len(instructions):
            instruction = instructions[inst_index]
            source = comments[inst_index] if inst_index < len(comments) else ""
        else:
            instruction = ""
            source = f"unrecognized code object; raw offset={record['pc'].get('code_object_offset')}"

        issued: bool | None = None
        stall_reason = ""
        inst_type = ""
        if method == "stochastic":
            issued = bool(record.get("wave_issued"))
            snapshot = record.get("snapshot") or {}
            stall_reason = shorten_enum(snapshot.get("stall_reason"), STALL_PREFIX)
            inst_type = shorten_enum(record.get("inst_type"), INST_TYPE_PREFIX)

        yield Sample(
            dispatch_id=int(record["dispatch_id"]),
            instruction=instruction,
            source=source,
            code_object_id=parse_code_object_id(record["pc"]),
            offset=int(record["pc"].get("code_object_offset", 0)),
            issued=issued,
            stall_reason=stall_reason,
            inst_type=inst_type,
        )


def detect_csv_method(path: Path) -> str:
    name = path.name
    if "stochastic" in name:
        return "stochastic"
    if "host_trap" in name:
        return "host_trap"
    raise ValueError(f"Cannot infer PC sampling method from CSV filename: {path}")


def iter_csv_samples(path: Path, method: str) -> Iterable[Sample]:
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            issued: bool | None = None
            stall_reason = ""
            inst_type = ""
            if method == "stochastic":
                raw = row.get("Wave_Issued_Instruction", "")
                if raw != "":
                    issued = raw.strip() in {"1", "true", "True"}
                stall_reason = shorten_enum(row.get("Stall_Reason", ""), STALL_PREFIX)
                inst_type = shorten_enum(
                    row.get("Instruction_Type", ""), INST_TYPE_PREFIX
                )

            yield Sample(
                dispatch_id=int(row["Dispatch_Id"]),
                instruction=row.get("Instruction", "") or "",
                source=row.get("Instruction_Comment", "") or "",
                issued=issued,
                stall_reason=stall_reason,
                inst_type=inst_type,
            )


def discover_inputs(path: Path) -> tuple[str, Path, Path | None, list[str]]:
    """Return (kind, primary_path, kernel_trace_path, all_matched_files)."""
    matched: list[str] = []

    if path.is_file():
        if path.name.endswith(".json"):
            return "json", path, None, [str(path)]
        if "pc_sampling" in path.name and path.suffix == ".csv":
            matched.append(str(path))
            kernel_csv = find_kernel_trace_csv(path)
            if kernel_csv:
                matched.append(str(kernel_csv))
            return "csv", path, kernel_csv, matched
        raise ValueError(f"Unsupported file: {path}")

    json_files = sorted(path.glob("*results.json"))
    if json_files:
        matched.append(str(json_files[0]))
        return "json", json_files[0], None, matched

    csv_files = sorted(path.glob("*pc_sampling_*.csv"))
    if not csv_files:
        raise ValueError(f"No PC sampling output found under {path}")
    pc_csv = csv_files[0]
    matched.append(str(pc_csv))
    kernel_csv = find_kernel_trace_csv(pc_csv)
    if kernel_csv:
        matched.append(str(kernel_csv))
    return "csv", pc_csv, kernel_csv, matched


def parse_inputs(path: Path) -> Report:
    kind, primary, kernel_trace, files = discover_inputs(path)

    if kind == "json":
        tool = load_tool_record(primary)
        method = detect_pc_sampling_key(tool)
        dispatch_map = build_dispatch_map_from_json(tool)
        samples = list(iter_json_samples(tool, method))
    else:
        method = detect_csv_method(primary)
        samples = list(iter_csv_samples(primary, method))
        dispatch_map: dict[int, str] = {}
        if kernel_trace:
            candidate_map = build_dispatch_map_from_csv(kernel_trace)
            if dispatch_map_covers_samples(candidate_map, samples):
                dispatch_map = candidate_map
            else:
                files = [f for f in files if f != str(kernel_trace)]

    report = Report(method=method, source_files=files)
    for sample in samples:
        report.add(sample, dispatch_map.get(sample.dispatch_id))
    return report


def format_pct(count: int, total: int) -> str:
    if total == 0:
        return "0.0%"
    return f"{100.0 * count / total:.1f}%"


def format_stall_reasons(loc: LocationStats, max_reasons: int = 3) -> str:
    if not loc.stall_reasons:
        return ""
    return ", ".join(
        f"{reason} ({count})"
        for reason, count in loc.stall_reasons.most_common(max_reasons)
    )


def format_source_cell(loc: LocationStats) -> str:
    if loc.source:
        return format_table_cell(loc.source)
    if loc.offset is not None:
        return format_table_cell(f"(no source; offset={format_offset(loc.offset)})")
    return ""


def format_table_cell(text: str) -> str:
    """Normalize whitespace and escape characters that break markdown tables."""
    return " ".join(text.split()).replace("|", "\\|")


def format_offset(offset: int | None) -> str:
    if offset is None:
        return ""
    return f"0x{offset:x}"


def render_report(report: Report, top_n: int, kernel_filter: str | None) -> str:
    lines: list[str] = []
    lines.append("# PC Sampling Report")
    lines.append("")
    lines.append(f"- Method: **{report.method}**")
    lines.append(f"- Total samples: **{report.total_samples}**")
    lines.append(f"- Kernels: **{len(report.kernels)}**")
    lines.append(f"- Input files: {', '.join(report.source_files)}")
    if report.unmapped_dispatch_ids:
        ids = ", ".join(str(i) for i in sorted(report.unmapped_dispatch_ids))
        lines.append(
            f"- Warning: {len(report.unmapped_dispatch_ids)} dispatch id(s) had no kernel "
            f"trace mapping ({ids}). Re-run with `--kernel-trace` for kernel names."
        )
    if report.undecodable_samples:
        lines.append(
            f"- Note: **{report.undecodable_samples}** sample(s) could not be decoded to a "
            "known code object (e.g. runtime blit kernels or self-modifying code). These "
            "appear with an offset-only placeholder in the Source column."
        )
    lines.append("")

    kernels = sorted(report.kernels.values(), key=lambda k: k.sample_count, reverse=True)
    if kernel_filter:
        kernels = [k for k in kernels if kernel_filter in k.name]
        if not kernels:
            lines.append(f"No kernels matched filter `{kernel_filter}`.")
            return "\n".join(lines)

    for kernel in kernels:
        lines.append(f"## Kernel: {kernel.name}")
        lines.append("")
        dispatch_ids = ", ".join(str(i) for i in sorted(kernel.dispatch_ids))
        lines.append(f"- Dispatch IDs: {dispatch_ids}")
        lines.append(f"- Samples: **{kernel.sample_count}**")
        lines.append(f"- Unique sampled locations: **{len(kernel.locations)}**")
        lines.append("")

        ranked = sorted(
            kernel.locations.values(), key=lambda loc: loc.count, reverse=True
        )[:top_n]
        lines.append(f"### Top {len(ranked)} sample locations")
        lines.append("")

        if report.method == "stochastic":
            lines.append(
                "| Rank | Source | Instruction | Count | % Samples | Issued | % Issued | "
                "Stalled | % Stalled | Stall reasons |"
            )
            lines.append(
                "|------|--------|-------------|-------|-----------|--------|----------|"
                "---------|-----------|---------------|"
            )
            for rank, loc in enumerate(ranked, start=1):
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            str(rank),
                            format_source_cell(loc),
                            format_table_cell(loc.instruction),
                            str(loc.count),
                            format_pct(loc.count, kernel.sample_count),
                            str(loc.issued),
                            format_pct(loc.issued, loc.count),
                            str(loc.stalled),
                            format_pct(loc.stalled, loc.count),
                            format_table_cell(format_stall_reasons(loc)),
                        ]
                    )
                    + " |"
                )
        else:
            lines.append("| Rank | Source | Instruction | Count | % Samples |")
            lines.append("|------|--------|-------------|-------|-----------|")
            for rank, loc in enumerate(ranked, start=1):
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            str(rank),
                            format_source_cell(loc),
                            format_table_cell(loc.instruction),
                            str(loc.count),
                            format_pct(loc.count, kernel.sample_count),
                        ]
                    )
                    + " |"
                )

        lines.append("")

        if report.method == "stochastic" and kernel.stall_reasons:
            total_stalled = sum(kernel.stall_reasons.values())
            lines.append("### Stall reasons (stochastic, stalled samples only)")
            lines.append("")
            lines.append("| Stall reason | Count | % of stalled |")
            lines.append("|--------------|-------|--------------|")
            for reason, count in kernel.stall_reasons.most_common(top_n):
                pct = (100.0 * count / total_stalled) if total_stalled else 0.0
                lines.append(f"| {reason} | {count} | {pct:.1f}% |")
            lines.append("")

        lines.append("---")
        lines.append("")

    lines.append("## Notes for analysis")
    lines.append("")
    if report.method == "host_trap":
        lines.append(
            "- Host-trap sampling may skid up to ~2 instructions; inspect neighbors of top hotspots."
        )
    else:
        lines.append(
            "- High `WAITCNT` → memory latency; `ARBITER_NOT_WIN` → pipe contention; "
            "`ARBITER_WIN_EX_STALL` → execution backpressure."
        )
    lines.append(
        "- Empty source columns mean the app was built without `-g`; rebuild for source-level advice."
    )
    lines.append(
        "- Sample counts are relative within this run; compare before/after code changes."
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize rocprofv3 PC sampling output for agent analysis.",
    )
    parser.add_argument(
        "input",
        type=Path,
        help="JSON results file, PC sampling CSV, or directory containing output files",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=15,
        help="Number of top locations and stall reasons to show per kernel (default: 15)",
    )
    parser.add_argument(
        "--kernel",
        default=None,
        help="Show only kernels whose name contains this substring",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write report to this file (default: stdout)",
    )
    args = parser.parse_args(argv)

    try:
        report = parse_inputs(args.input)
    except (ValueError, OSError, json.JSONDecodeError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    text = render_report(report, top_n=max(1, args.top), kernel_filter=args.kernel)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote report to {args.output}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
