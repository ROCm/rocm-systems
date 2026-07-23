#!/usr/bin/env python3
"""Parse HRR replay/capture logs into a structured finding (read-only).

Cross-platform: works on Linux and Windows.
On Windows hrr-playback.exe is resolved automatically when not provided.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Platform helpers
# ---------------------------------------------------------------------------
_IS_WIN = sys.platform == "win32"


def _playback_exe_name(name: str) -> str:
    """Append .exe on Windows when the name has no extension."""
    if _IS_WIN and not Path(name).suffix:
        return name + ".exe"
    return name


def _find_playback_in_sdk() -> str | None:
    """Search HIP SDK install for hrr-playback (Windows only)."""
    if not _IS_WIN:
        return None
    hip_root = os.environ.get("HIP_PATH") or os.environ.get("ROCM_PATH", "")
    candidates: list[Path] = []
    if hip_root:
        for sub in ("bin", r"hip\bin", r"tools\bin"):
            candidates.append(Path(hip_root) / sub / "hrr-playback.exe")
    # PATH walk
    for p in os.environ.get("PATH", "").split(os.pathsep):
        candidates.append(Path(p) / "hrr-playback.exe")
    for c in candidates:
        if c.is_file():
            return str(c)
    return None


# ---------------------------------------------------------------------------
# Regex library (unchanged from Linux version)
# ---------------------------------------------------------------------------

# Match [HRR progress] lines; last="..." is optional because compact_kernel_name
# can produce an empty string, and the greedy .* can sometimes fail to span the
# gap between d2h_attempted and last= on very long lines.
RE_PROGRESS = re.compile(
    r"\[HRR progress\]\s+elapsed_s=[\d.]+\s+"
    r"seq=(\d+)\s+kernels=(\d+)\s+"
    r"d2h_pass=(\d+)\s+d2h_fail=(\d+)\s+d2h_attempted=(\d+)"
    r"(?:\s+last=\"([^\"]*)\")?"
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
RE_ARCHIVE_COMPLETE = re.compile(r"Complete:\s+(yes|no|YES|NO)", re.IGNORECASE)
RE_CAPTURE_MAF = RE_MAF
RE_D2H_SUMMARY = re.compile(r"D2H checks\s+: (\d+) pass.*?, (\d+) fail, (\d+) skipped")
RE_KERNARG = re.compile(r"kernarg_address=(0x[0-9a-fA-F]+)")
RE_GRID = re.compile(r"grid=\[([^\]]+)\], workgroup=\[([^\]]+)\]")
RE_CIJK = re.compile(r"(Cijk_[A-Za-z0-9_]+)")
# Parses the "Kernel Call Counts:" table emitted by hrr-playback --info
# Format:  "  Kernel_name[...]   <count>"
RE_KERNEL_CALL_COUNT = re.compile(
    r"^\s+(Cijk_[A-Za-z0-9_.]+?)(?:\.\.\.)?[ \t]+(\d+)\s*$", re.MULTILINE
)
RE_CAPTURE_HIP = re.compile(r"\[capture\] HIP_SO=(\S+)")
RE_VERSION_MISMATCH = re.compile(r"\[HRR\] Version mismatch: file=(\d+) reader=(\d+)")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

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
    # All unique hipBLASLt kernel families found in the archive (populated on PASS).
    # On fault paths only kernel_name / kernel_family are set (single fault kernel).
    top_kernel_families: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

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
        # group(6) is optional (last="..." may be absent or empty)
        name = m.group(6)
        if name:
            last_prog = name
    finding.last_progress_kernel = last_prog

    for pattern in (RE_FATAL_EVENT, RE_FATAL_GPU):
        hit = pattern.search(text)
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

    # Collect ALL unique hipBLASLt kernel families from the archive.
    # Primary source: Kernel Call Counts table in --info output (has actual call counts).
    # The table may truncate names with "..." but enough of the name survives to extract
    # the MT tile-size family tag.
    cijk_fam_counts: dict[str, int] = {}
    for km in RE_KERNEL_CALL_COUNT.finditer(text):
        fam = _kernel_family(km.group(1))
        if fam:
            cijk_fam_counts[fam] = cijk_fam_counts.get(fam, 0) + int(km.group(2))
    # Fallback: scan full text for any Cijk_ token when the table is absent.
    if not cijk_fam_counts:
        for km in RE_CIJK.finditer(text):
            fam = _kernel_family(km.group(1))
            if fam:
                cijk_fam_counts[fam] = cijk_fam_counts.get(fam, 0) + 1
    if cijk_fam_counts:
        finding.top_kernel_families = [
            f"{fam} ({cnt} calls)"
            for fam, cnt in sorted(cijk_fam_counts.items(), key=lambda x: -x[1])
        ]

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


# ---------------------------------------------------------------------------
# Archive info via hrr-playback --info  (cross-platform)
# ---------------------------------------------------------------------------

def run_archive_info(archive: Path, hrr_playback: str | None) -> str:
    # Resolve playback binary: explicit arg > HRR_PLAYBACK env > SDK search > bare name
    play = (
        hrr_playback
        or os.environ.get("HRR_PLAYBACK")
        or (_find_playback_in_sdk() if _IS_WIN else None)
        or _playback_exe_name("hrr-playback")
    )
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


# ---------------------------------------------------------------------------
# Manifest metadata (imports check_replay_compat if available)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

def render_markdown(f: Finding) -> str:
    lines = [
        "# HRR replay finding",
        "",
        "## Summary",
        f"- **Outcome**: {f.outcome}",
        f"- **Fault class**: `{f.fault_class}`",
        # PASS: no single "fault" kernel; show all hipBLASLt families found.
        # FAULT: show the one kernel attributed to the fault.
        *(
            [f"- **hipBLASLt families**: {', '.join(f'`{k}`' for k in f.top_kernel_families)}"]
            if f.outcome == "PASS" and f.top_kernel_families
            else [
                f"- **Kernel**: `{f.kernel_name or 'unknown'}`",
                f"- **Kernel family**: `{f.kernel_family or 'unknown'}`",
            ]
        ),
        "",
        f"## {'Replay details' if f.outcome == 'PASS' else 'Fault details'}",
        f"- **Fault address**: `{f.fault_address or 'n/a'}`",
        f"- **Fault reason**: {f.fault_reason or 'n/a'}",
        # On a PASS the seq is the last heartbeat checkpoint, not a failure point.
        f"- **{'Last event seq' if f.outcome == 'PASS' else 'Failing event seq'}**: {f.failing_event_seq or 'n/a'}",
        f"- **Failing call index**: {f.failing_call_index or 'n/a'}",
        f"- **Failing API**: {f.failing_api or 'n/a'}",
        f"- **Kernarg address**: `{f.kernarg_address or 'n/a'}`",
        f"- **GPU node**: {f.gpu_node or 'n/a'}",
        f"- **Grid / workgroup**: {f.grid or 'n/a'} / {f.workgroup or 'n/a'}",
        "",
        f"## {'Replay progress' if f.outcome == 'PASS' else 'Replay progress at fault'}",
        # On PASS: the heartbeat may have fired before all kernels ran (early timer tick).
        # Fall back to archive_kernels (from --info) as the authoritative total.
        (
            f"- **Kernels launched**: {f.archive_kernels or f.kernels_launched or 'n/a'}"
            if f.outcome == "PASS"
            else f"- **Kernels launched**: {f.kernels_launched or 'n/a'}"
        ),
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
        f"- **Capture device count**: "
        f"{f.capture_device_count if f.capture_device_count is not None else 'n/a'}",
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_PS_NATIVE_PREFIX = re.compile(
    r"^[^\s:]+\.exe\s*:\s*",  # e.g. "hrr-playback.exe : "
    re.MULTILINE,
)
_PS_CONTINUATION = re.compile(
    # PowerShell splits ErrorRecord lines; the second line starts without a prefix.
    # Join them back: if a line ends mid-pattern (no closing quote) and the next
    # line starts with last="  or just a bare token, reconnect.
    r'(seq=\d+ kernels=\d+ d2h_pass=\d+ d2h_fail=\d+ d2h_attempted=\d+)\s*\n\s*(last=)',
)


def _strip_ps_formatting(text: str) -> str:
    """Remove PowerShell ErrorRecord prefix that Tee-Object adds when stderr is
    captured via PS 2>&1 instead of cmd /c 2>&1.  Also rejoin split progress lines."""
    text = _PS_NATIVE_PREFIX.sub("", text)
    text = _PS_CONTINUATION.sub(r"\1 \2", text)
    return text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--log", action="append", default=[], help="Replay or capture log (repeatable)"
    )
    ap.add_argument("--archive", help="HRR archive pid-* directory for --info")
    ap.add_argument("--hrr-playback", help="Path to hrr-playback(.exe) binary")
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
        raw = p.read_text(encoding="utf-8", errors="replace")
        # Strip PowerShell ErrorRecord prefix ("hrr-playback.exe : ...") that appears
        # when the log was captured via PS `2>&1 | Tee-Object` instead of `cmd /c 2>&1`.
        text = _strip_ps_formatting(raw)
        size_kb = p.stat().st_size // 1024
        print(f"[analyzer] log={p}  size={size_kb}KB  lines={text.count(chr(10))}",
              file=sys.stderr)
        parse_text(text, str(p), finding)

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
