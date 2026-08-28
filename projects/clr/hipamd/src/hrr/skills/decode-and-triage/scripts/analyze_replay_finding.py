#!/usr/bin/env python3
"""Parse HRR replay/capture logs into a structured finding (read-only)."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# --- regex library (diverse workloads) ---

RE_PROGRESS = re.compile(
    r"\[HRR progress\].*seq=(\d+).*kernels=(\d+).*d2h_pass=(\d+).*"
    r"d2h_fail=(\d+).*d2h_attempted=(\d+).*last=\"([^\"]+)\""
)
RE_FATAL_EVENT = re.compile(
    r"\[HRR\] Fatal: T(\d+) Event (\d+) \(([^)]+)\) returned (\d+) \(([^)]+)\)"
)
RE_FATAL_GPU = re.compile(
    r"\[HRR\] Fatal: GPU error after T(\d+) Event (\d+) \(([^)]+)\): (\d+) \(([^)]+)\)"
)
RE_FATAL_GENERIC = re.compile(r"\[HRR\] Fatal: ([^\n]+)")
RE_MAF = re.compile(
    r"Memory access fault by GPU node-(\d+).*on address (0x[0-9a-fA-F]+)\.\s*"
    r"Reason:\s*([^.\n]+)"
)
RE_MEM_FAULT_ERR = re.compile(
    r"Memory Fault Error \[host: [^,]+, GPU index: \d+, faulting addr: (0x[0-9a-fA-F]+), "
    r"kernel: ([^\]]+)\]"
)
RE_HANG = re.compile(r"HSA_STATUS_ERROR_(MEMORY_FAULT|ABORTED|EXCEPTION)")
RE_PASS = re.compile(r"\[HRR\] PASS\b")
RE_FAIL = re.compile(r"\[HRR\] FAIL\b")
RE_ARCHIVE_RECOVERED = re.compile(
    r"recovered (\d+) events|Archive : (\d+) events, (\d+) kernels, (\d+) blobs, (\d+) code objects"
)
RE_ARCHIVE_COMPLETE = re.compile(r"Complete:\s+(YES|NO)")
RE_CAPTURE_MAF = RE_MAF
RE_D2H_SUMMARY = re.compile(r"D2H checks\s+: (\d+) pass.*?, (\d+) fail, (\d+) skipped")
RE_KERNARG = re.compile(r"kernarg_address=(0x[0-9a-fA-F]+)")
RE_GRID = re.compile(r"grid=\[([^\]]+)\], workgroup=\[([^\]]+)\]")
RE_CIJK = re.compile(r"(Cijk_[A-Za-z0-9_]+)")
RE_CAPTURE_HIP = re.compile(r"\[capture\] HIP_SO=(\S+)")
RE_VERSION_MISMATCH = re.compile(r"\[HRR\] Version mismatch: file=(\d+) reader=(\d+)")


@dataclass
class Finding:
    outcome: str
    fault_class: str
    fault_address: str | None = None
    fault_reason: str | None = None
    failing_event_seq: int | None = None
    failing_call_index: int | None = None
    failing_thread: int | None = None
    failing_api: str | None = None
    kernel_name: str | None = None
    kernel_family: str | None = None
    kernarg_address: str | None = None
    grid: str | None = None
    workgroup: str | None = None
    gpu_node: str | None = None
    last_progress_kernel: str | None = None
    kernels_launched: int | None = None
    d2h_pass: int | None = None
    d2h_fail: int | None = None
    d2h_attempted: int | None = None
    archive_events: int | None = None
    archive_kernels: int | None = None
    archive_complete: str | None = None
    capture_hip_so: str | None = None
    capture_hip_runtime_version: str | None = None
    capture_comgr_version: str | None = None
    capture_device_count: int | None = None
    capture_gcn_arch: str | None = None
    sources: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _classify(text: str, finding: Finding) -> str:
    if RE_VERSION_MISMATCH.search(text):
        return "version_mismatch"
    if RE_PASS.search(text):
        if finding.d2h_fail and finding.d2h_fail > 0:
            return "nan_inf_divergence"
        return "replay_pass"
    if "out of memory" in text.lower() or "hipErrorOutOfMemory" in text:
        return "replay_oom"
    if (
        RE_FATAL_EVENT.search(text)
        or RE_FATAL_GPU.search(text)
        or RE_FATAL_GENERIC.search(text)
    ):
        if "out of memory" in text.lower():
            return "replay_oom"
        return "replay_fatal_api"
    if RE_MAF.search(text) or RE_MEM_FAULT_ERR.search(text):
        reason = (finding.fault_reason or "").lower()
        if "read-only" in reason:
            return "read_only_page_fault"
        return "illegal_memory_access"
    if RE_HANG.search(text) and not RE_PASS.search(text):
        return "hang"
    if RE_FAIL.search(text) or (finding.d2h_fail and finding.d2h_fail > 0):
        return "nan_inf_divergence"
    if "Replay aborted" in text or "aborting replay" in text:
        return "replay_aborted"
    return "unknown"


def _kernel_family(name: str | None) -> str | None:
    if not name:
        return None
    if name.startswith("Cijk_"):
        m = re.search(r"_MT(\d+x\d+x\d+)", name)
        sk = "_SK3_" if "_SK3_" in name else ("_SK2_" if "_SK2_" in name else None)
        parts = ["hipblaslt_gemm"]
        if m:
            parts.append(f"MT{m.group(1)}")
        if sk:
            parts.append("streamk" if "SK3" in sk else "streamk_variant")
        return "/".join(parts)
    if name.startswith("_ZN"):
        return "pytorch_kernel"
    return "other"


def parse_text(text: str, source: str, finding: Finding) -> Finding:
    finding.sources.append(source)

    for m in RE_CAPTURE_HIP.finditer(text):
        finding.capture_hip_so = m.group(1)

    for m in RE_ARCHIVE_RECOVERED.finditer(text):
        g = m.groups()
        if g[0]:
            finding.archive_events = int(g[0])
        if len(g) >= 5 and g[1]:
            finding.archive_events = int(g[1])
            finding.archive_kernels = int(g[2])

    m = RE_ARCHIVE_COMPLETE.search(text)
    if m:
        finding.archive_complete = m.group(1)

    m = RE_VERSION_MISMATCH.search(text)
    if m:
        finding.notes.append(
            f"archive wire version {m.group(1)} does not match hrr-playback reader {m.group(2)}"
        )

    last_prog = None
    for m in RE_PROGRESS.finditer(text):
        finding.failing_event_seq = int(m.group(1))
        finding.kernels_launched = int(m.group(2))
        finding.d2h_pass = int(m.group(3))
        finding.d2h_fail = int(m.group(4))
        finding.d2h_attempted = int(m.group(5))
        last_prog = m.group(6)
    finding.last_progress_kernel = last_prog

    for m in (RE_FATAL_EVENT, RE_FATAL_GPU):
        hit = m.search(text)
        if hit:
            finding.failing_thread = int(hit.group(1))
            finding.failing_call_index = int(hit.group(2))
            finding.failing_api = hit.group(3)
            break

    m = RE_MAF.search(text)
    if m:
        finding.gpu_node = m.group(1)
        finding.fault_address = m.group(2)
        finding.fault_reason = m.group(3).strip()

    m = RE_MEM_FAULT_ERR.search(text)
    if m:
        finding.fault_address = finding.fault_address or m.group(1)
        finding.kernel_name = m.group(2).strip()

    if not finding.kernel_name:
        cijk = RE_CIJK.search(text)
        if cijk:
            finding.kernel_name = cijk.group(1)

    m = RE_KERNARG.search(text)
    if m:
        finding.kernarg_address = m.group(1)

    m = RE_GRID.search(text)
    if m:
        finding.grid = m.group(1)
        finding.workgroup = m.group(2)

    m = RE_D2H_SUMMARY.search(text)
    if m:
        finding.d2h_pass = int(m.group(1))
        finding.d2h_fail = int(m.group(2))

    new_class = _classify(text, finding)
    if new_class != "unknown" or finding.fault_class in (None, "unknown"):
        finding.fault_class = new_class

    new_outcome = finding.outcome
    if RE_PASS.search(text):
        new_outcome = "PASS"
    elif RE_MAF.search(text) or RE_MEM_FAULT_ERR.search(text):
        new_outcome = "MAF"
    elif RE_FAIL.search(text):
        new_outcome = "FAIL"
    elif (
        "aborting replay" in text
        or RE_FATAL_EVENT.search(text)
        or RE_VERSION_MISMATCH.search(text)
    ):
        new_outcome = "ABORT"
    elif finding.outcome == "UNKNOWN":
        new_outcome = "UNKNOWN"
    finding.outcome = new_outcome
    finding.kernel_family = _kernel_family(finding.kernel_name)
    return finding


def run_archive_info(archive: Path, hrr_playback: str | None) -> str:
    play = hrr_playback or "hrr-playback"
    try:
        proc = subprocess.run(
            [play, str(archive), "--info"],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        return proc.stdout + proc.stderr
    except FileNotFoundError:
        return ""
    except subprocess.TimeoutExpired:
        return "[timeout running hrr-playback --info]"


def render_markdown(f: Finding) -> str:
    lines = [
        "# HRR replay finding",
        "",
        "## Summary",
        f"- **Outcome**: {f.outcome}",
        f"- **Fault class**: `{f.fault_class}`",
        f"- **Kernel**: `{f.kernel_name or 'unknown'}`",
        f"- **Kernel family**: `{f.kernel_family or 'unknown'}`",
        "",
        "## Fault details",
        f"- **Fault address**: `{f.fault_address or 'n/a'}`",
        f"- **Fault reason**: {f.fault_reason or 'n/a'}",
        f"- **Failing event seq**: {f.failing_event_seq or 'n/a'}",
        f"- **Failing call index**: {f.failing_call_index or 'n/a'}",
        f"- **Failing API**: {f.failing_api or 'n/a'}",
        f"- **Kernarg address**: `{f.kernarg_address or 'n/a'}`",
        f"- **GPU node**: {f.gpu_node or 'n/a'}",
        f"- **Grid / workgroup**: {f.grid or 'n/a'} / {f.workgroup or 'n/a'}",
        "",
        "## Replay progress at fault",
        f"- **Kernels launched**: {f.kernels_launched or 'n/a'}",
        f"- **D2H**: pass={f.d2h_pass or 0} fail={f.d2h_fail or 0} attempted={f.d2h_attempted or 0}",
        f"- **Last progress kernel**: `{f.last_progress_kernel or 'n/a'}`",
        "",
        "## Archive / capture",
        f"- **Events**: {f.archive_events or 'n/a'}",
        f"- **Kernels (archive)**: {f.archive_kernels or 'n/a'}",
        f"- **Complete**: {f.archive_complete or 'n/a'}",
        f"- **Capture HIP**: `{f.capture_hip_so or 'n/a'}`",
        f"- **Capture HIP runtime**: `{f.capture_hip_runtime_version or 'n/a'}`",
        f"- **Capture comgr**: `{f.capture_comgr_version or 'n/a'}`",
        f"- **Capture device count**: {f.capture_device_count if f.capture_device_count is not None else 'n/a'}",
        f"- **Capture GPU arch**: `{f.capture_gcn_arch or 'n/a'}`",
        "",
        "## Sources",
    ]
    for s in f.sources:
        lines.append(f"- `{s}`")
    if f.notes:
        lines.extend(["", "## Notes"])
        lines.extend(f"- {n}" for n in f.notes)
    return "\n".join(lines) + "\n"


def apply_manifest_metadata(finding: Finding, archive: Path) -> None:
    try:
        from check_replay_compat import load_capture_metadata
    except ImportError:
        return
    meta = load_capture_metadata(archive)
    if meta is None:
        finding.notes.append("manifest metadata absent (legacy capture)")
        return
    finding.capture_hip_runtime_version = meta.hip_runtime_version
    finding.capture_comgr_version = meta.comgr_version
    finding.capture_device_count = meta.device_count
    if meta.devices:
        props = meta.devices[0].get("properties")
        if isinstance(props, dict):
            finding.capture_gcn_arch = props.get("gcn_arch_name")
        else:
            finding.capture_gcn_arch = meta.devices[0].get("gcn_arch_name")
    finding.sources.append(str(archive / "manifest.json"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--log", action="append", default=[], help="Replay or capture log (repeatable)"
    )
    ap.add_argument("--archive", help="HRR archive pid-* directory for --info")
    ap.add_argument("--hrr-playback", help="Path to hrr-playback binary")
    ap.add_argument("--format", choices=("json", "markdown"), default="markdown")
    ap.add_argument("-o", "--output", help="Write report to file")
    args = ap.parse_args()

    if not args.log and not args.archive:
        ap.error("provide --log and/or --archive")

    finding = Finding(outcome="UNKNOWN", fault_class="unknown")
    for log_path in args.log:
        p = Path(log_path)
        if not p.is_file():
            finding.notes.append(f"log not found: {p}")
            continue
        parse_text(p.read_text(encoding="utf-8", errors="replace"), str(p), finding)

    if args.archive:
        arch = Path(args.archive)
        apply_manifest_metadata(finding, arch)
        info = run_archive_info(arch, args.hrr_playback)
        if info:
            parse_text(info, f"{arch} (--info)", finding)
        else:
            finding.notes.append(
                "hrr-playback --info unavailable; archive path recorded only"
            )
            finding.sources.append(str(arch))

    out = (
        json.dumps(finding.to_dict(), indent=2)
        if args.format == "json"
        else render_markdown(finding)
    )
    if args.output:
        Path(args.output).write_text(out, encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
