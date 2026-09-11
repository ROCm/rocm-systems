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
# The leading fields of this bracket vary by ROCm build -- some emit `host:`,
# some start at `GPU index:` -- so anchor on the two fields actually consumed
# rather than on the whole prefix. Requiring `host:` dropped the kernel name on
# every build that omits it, which is the one field the report exists to give.
RE_MEM_FAULT_ERR = re.compile(
    r"Memory Fault Error \[[^\]]*?faulting addr: (0x[0-9a-fA-F]+), kernel: ([^\]]+)\]"
)
RE_HSA_STATUS = re.compile(r"HSA_STATUS_ERROR_(MEMORY_FAULT|ABORTED|EXCEPTION)")
# The runner bounds the replay and records the fact when it had to stop it.
# This is the only signal a hang leaves: it makes no progress and prints
# nothing, so without the marker the archive yields no finding at all.
RE_REPLAY_TIMEOUT = re.compile(r"replay timed out after (\d+)s")
# The replay process dying on a signal leaves the log truncated and nothing
# else, so the runner records it the same way. Read on its own it says the run
# failed, not that the workload did: no kernel was established as at fault.
RE_REPLAY_SIGNAL = re.compile(r"replay killed by signal (\d+)")
# A queue abort carries its own bracket, with the kernel but no faulting
# address, so the memory-fault regex above cannot see it. Observed on an
# out-of-bounds ATen gather, where this was the only line naming the culprit.
RE_QUEUE_ABORT_KERNEL = re.compile(
    r"aborting with error[^\[\n]*\[[^\]]*?kernel: ([^\]]+)\]"
)
RE_PASS = re.compile(r"\[HRR\] PASS\b")
RE_FAIL = re.compile(r"\[HRR\] FAIL\b")
RE_ARCHIVE_RECOVERED = re.compile(
    r"recovered (\d+) events|Archive : (\d+) events, (\d+) kernels, (\d+) blobs, (\d+) code objects"
)
# `--info` prints `Complete:     yes (clean shutdown)` or `Complete:     NO (no
# shutdown trailer; capture likely crashed)`, so the verdict is case-mixed and
# always carries a trailing explanation.
RE_ARCHIVE_COMPLETE = re.compile(r"Complete:\s+(yes|no)\b", re.IGNORECASE)
# `--info` reports the archive as labelled fields, one per line, while a replay
# run reports the same totals on a single `[HRR] Archive :` line.
RE_INFO_EVENTS = re.compile(r"^Events:\s+(\d+)\s*$", re.MULTILINE)
RE_INFO_KERNELS = re.compile(r"^Kernels:\s+(\d+)\s*$", re.MULTILINE)
RE_INFO_RECOVERED = re.compile(r"^Recovered:\s+(\d+)\s+events\s*$", re.MULTILINE)
# Short archives print no `Kernels:` total and report launches only in the API
# call-count block. Without this the total stays unknown and the single-kernel
# inference in finalize() can never fire.
RE_INFO_LAUNCH_COUNT = re.compile(r"\bhip\w*LaunchKernel\s+(\d+)\b")
# Rows of the `--info` kernel summary table: name, grid, block, with an optional
# leading id, since some builds omit the id column. A memory fault can kill the
# replay before any per-launch attribution reaches the log, and then this table
# is the only record of what the archive ran. The name must start with a letter
# or underscore, which is what keeps a bare id from being read as a symbol now
# that the id is optional.
RE_INFO_KERNEL_ROW = re.compile(
    r"^[ \t]*(?:\d+[ \t]+)?([A-Za-z_][^\s\[]*)[ \t]+\[[\d,\s]+\][ \t]+\[[\d,\s]+\]",
    re.MULTILINE,
)
# `--verbose` prints one line per replayed event. A GPU fault tears the process
# down before HRR writes its own Fatal line, so when a run was made verbose the
# last of these is the only record of the failing dispatch. The sync flags do
# not emit these lines: `hrr_playback.cpp` gates them on `ctx.verbose` alone.
RE_EVENT_PROGRESS = re.compile(
    r"^[ \t]*(?:\[HRR\][ \t]*)?Event (\d+):[ \t]*(\w+)"
    r"(?:[^\n]*?->[ \t]*Kernel '([^']+)')?",
    re.MULTILINE,
)
# PyTorch/ATen kernels reach the GPU through `<<<>>>` (hipLaunchByPtr) and pass
# device pointers inside by-value structs. Capture records those and replay
# translates them, so these kernels do replay faithfully on a current build. The
# detector is still a value-based heuristic, and an archive recorded before it
# landed carries no such offsets at all, so a fault on one of these symbols
# earns a caveat in the finding, not a different verdict.
RE_ATEN_CHEVRON = re.compile(r"_ZN2at6native|at::native::")
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
    replay_signal: int | None = None
    last_progress_kernel: str | None = None
    last_event_kernel: str | None = None
    kernels_launched: int | None = None
    d2h_pass: int | None = None
    d2h_fail: int | None = None
    d2h_attempted: int | None = None
    archive_events: int | None = None
    archive_kernels: int | None = None
    archive_kernel_names: list[str] = field(default_factory=list)
    archive_complete: str | None = None
    archive_format_version: int | None = None
    reader_format_version: int | None = None
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
    # A replay the runner had to stop made no progress against the clock, which
    # is what a hang is. This outranks the rest: whatever partial output the
    # replay printed before stalling does not describe why it stalled.
    if RE_REPLAY_TIMEOUT.search(text):
        return "hang"
    if RE_PASS.search(text):
        if finding.d2h_fail and finding.d2h_fail > 0:
            return "nan_inf_divergence"
        return "replay_pass"
    if "out of memory" in text.lower() or "hipErrorOutOfMemory" in text:
        return "replay_oom"
    # A GPU memory fault has to be classified before the generic abort branch.
    # Surfacing a fault through --sync-after-event makes the runtime print the
    # memory-fault line and hrr-playback print "Fatal: GPU error after ..." for
    # the same fault, so testing the abort first would report every fault as a
    # plain API error, and that is exactly the mode used to localize a fault.
    if RE_MAF.search(text) or RE_MEM_FAULT_ERR.search(text):
        reason = (finding.fault_reason or "").lower()
        if "read-only" in reason:
            return "read_only_page_fault"
        return "illegal_memory_access"
    if (
        RE_FATAL_EVENT.search(text)
        or RE_FATAL_GPU.search(text)
        or RE_FATAL_GENERIC.search(text)
    ):
        if "out of memory" in text.lower():
            return "replay_oom"
        return "replay_fatal_api"
    # An HSA queue abort is not a hang. MEMORY_FAULT says so in as many words,
    # and EXCEPTION is a hardware exception, which an out-of-bounds access
    # raises. Reporting either as a hang sends the reader after stalled work
    # that never existed and drops the kernel the abort just named. A real hang
    # shows up as no progress against the clock, which is what the replay
    # timeout above reports.
    hsa = RE_HSA_STATUS.search(text)
    if hsa and not RE_PASS.search(text):
        return (
            "illegal_memory_access"
            if hsa.group(1) == "MEMORY_FAULT"
            else "replay_aborted"
        )
    if RE_FAIL.search(text) or (finding.d2h_fail and finding.d2h_fail > 0):
        return "nan_inf_divergence"
    if "Replay aborted" in text or "aborting replay" in text:
        return "replay_aborted"
    # Last, so that a log which explains its own stop is believed first: a
    # memory fault also dies on a signal, and there the fault line is the
    # answer. On its own the signal says the replay process died without
    # reaching a verdict, which is a failure of the run and not of the workload.
    if RE_REPLAY_SIGNAL.search(text):
        return "replay_crashed"
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
    # Mangled and demangled forms of the same ATen symbols both occur: the
    # memory-fault line prints the mangled name, the queue-abort line prints
    # the demangled signature.
    if name.startswith("_ZN") or "at::native::" in name:
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
        finding.archive_complete = m.group(1).lower()

    for regex, attr in (
        (RE_INFO_RECOVERED, "archive_events"),
        (RE_INFO_EVENTS, "archive_events"),
        (RE_INFO_KERNELS, "archive_kernels"),
    ):
        m = regex.search(text)
        if m:
            setattr(finding, attr, int(m.group(1)))

    if finding.archive_kernels is None:
        m = RE_INFO_LAUNCH_COUNT.search(text)
        if m:
            finding.archive_kernels = int(m.group(1))

    for name in RE_INFO_KERNEL_ROW.findall(text):
        # The table truncates long names to its column width, and a truncated
        # symbol is worse than none: it cannot be looked up or handed over.
        if name.endswith("...") or name in finding.archive_kernel_names:
            continue
        finding.archive_kernel_names.append(name)

    m = RE_HSA_STATUS.search(text)
    if m and not RE_PASS.search(text):
        note = f"the queue aborted with HSA_STATUS_ERROR_{m.group(1)}"
        if note not in finding.notes:
            finding.notes.append(note)

    m = RE_REPLAY_SIGNAL.search(text)
    if m:
        finding.replay_signal = int(m.group(1))

    m = RE_VERSION_MISMATCH.search(text)
    if m:
        finding.archive_format_version = int(m.group(1))
        finding.reader_format_version = int(m.group(2))
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
    # Only when this input had progress lines. Assigning unconditionally let the
    # archive `--info` pass, which has none, wipe the kernel the replay log had
    # already recorded, and the default pipeline always parses both.
    if last_prog is not None:
        finding.last_progress_kernel = last_prog

    for m in (RE_FATAL_EVENT, RE_FATAL_GPU):
        hit = m.search(text)
        if hit:
            finding.failing_thread = int(hit.group(1))
            finding.failing_call_index = int(hit.group(2))
            finding.failing_api = hit.group(3)
            break

    last_event = None
    for m in RE_EVENT_PROGRESS.finditer(text):
        last_event = m
    if last_event is not None:
        # An HRR Fatal line names the failing event exactly; this is only the
        # last event that started, so it never overrides one.
        if finding.failing_call_index is None:
            finding.failing_call_index = int(last_event.group(1))
            finding.failing_api = last_event.group(2)
        finding.last_event_kernel = last_event.group(3)

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
        m = RE_QUEUE_ABORT_KERNEL.search(text)
        if m:
            finding.kernel_name = m.group(1).strip()

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
    if RE_REPLAY_TIMEOUT.search(text):
        new_outcome = "HANG"
    elif RE_PASS.search(text):
        new_outcome = "PASS"
    elif RE_MAF.search(text) or RE_MEM_FAULT_ERR.search(text):
        new_outcome = "MAF"
    elif RE_FAIL.search(text):
        new_outcome = "FAIL"
    elif RE_HSA_STATUS.search(text):
        # A queue abort is a GPU-side stop, so it is never UNKNOWN. Its own
        # status says which: MEMORY_FAULT is a fault, the rest are aborts.
        new_outcome = (
            "MAF"
            if RE_HSA_STATUS.search(text).group(1) == "MEMORY_FAULT"
            else "ABORT"
        )
    elif (
        "aborting replay" in text
        or RE_FATAL_EVENT.search(text)
        or RE_VERSION_MISMATCH.search(text)
        or RE_REPLAY_SIGNAL.search(text)
    ):
        new_outcome = "ABORT"
    elif finding.outcome == "UNKNOWN":
        new_outcome = "UNKNOWN"
    finding.outcome = new_outcome
    finding.kernel_family = _kernel_family(finding.kernel_name)
    return finding


def finalize(finding: Finding) -> Finding:
    """Settle kernel attribution once every input has been parsed.

    Attribution cannot be decided per input: the replay log and the archive
    `--info` dump each hold half of the evidence, and which half arrives first
    depends on how the caller was invoked.
    """
    # A clean replay implicates no kernel. The archive still lists the kernels
    # it ran, and a GEMM matched out of that listing would sit in the report
    # next to a pass as though it were a culprit.
    if finding.fault_class == "replay_pass":
        finding.kernel_name = None
        finding.kernel_family = None
        finding.last_event_kernel = None
        return finding

    # The replay process died without reaching a verdict, so nothing here is a
    # statement about the workload. Inferring a kernel from the archive would
    # hand over a culprit for a crash of the replay itself.
    if finding.fault_class == "replay_crashed":
        finding.kernel_name = None
        finding.kernel_family = None
        finding.notes.append(
            f"the replay process was killed by signal {finding.replay_signal} "
            f"without printing a verdict, so no kernel is implicated and the "
            f"workload is untested. Re-run the replay; if it dies the same way "
            f"again, this is a defect in hrr-playback or its environment rather "
            f"than in the captured workload."
        )
        return finding

    # Under --sync-after-launch the last launch to start is the one that
    # faulted, so it stands in when the runtime's fault line carried no kernel.
    if not finding.kernel_name and finding.last_event_kernel:
        finding.kernel_name = finding.last_event_kernel
        finding.notes.append(
            f"kernel name taken from the last launch to start before the fault "
            f"({finding.last_event_kernel}); the runtime fault line named no "
            f"kernel. Valid only for a replay run with --verbose, which is what "
            f"prints those per-event lines."
        )

    # A memory fault can tear the process down before the failing dispatch is
    # attributed, leaving a log with no kernel at all. When the archive holds
    # exactly one kernel, that kernel is the one that faulted. More than one
    # stays unknown: picking among several would be a guess.
    if (
        not finding.kernel_name
        and finding.archive_kernels == 1
        and len(finding.archive_kernel_names) == 1
    ):
        finding.kernel_name = finding.archive_kernel_names[0]
        finding.notes.append(
            f"kernel name inferred from the archive, which contains exactly one "
            f"kernel ({finding.kernel_name}); the replay log carried no "
            f"per-launch attribution. Re-run with --sync-after-launch to "
            f"confirm the faulting dispatch directly."
        )

    # A `<<<>>>`-launched ATen kernel passes device pointers inside by-value
    # structs. Current capture records those offsets and replay translates them,
    # so this is not automatically a recording artefact, but the detector is a
    # heuristic and an archive taken before it landed carries no offsets at all.
    # Both failure modes look exactly like a workload fault, so flag the
    # ambiguity rather than resolving it either way.
    if finding.fault_class in (
        "illegal_memory_access",
        "read_only_page_fault",
        "replay_aborted",
    ) and RE_ATEN_CHEVRON.search(finding.kernel_name or ""):
        finding.notes.append(
            "the faulting kernel is an ATen kernel launched through <<<>>> "
            "(hipLaunchByPtr), which passes device pointers inside by-value "
            "structs. Replay translates those via a value-based heuristic, and "
            "an archive recorded before that support landed has none recorded "
            "at all, so an untranslated pointer here would fault exactly like a "
            "workload defect. Confirm against the user's original failure "
            "signature before reporting this as their bug."
        )

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
        f"- **Last launch before fault**: `{f.last_event_kernel or 'n/a'}`",
        "",
        "## Archive / capture",
        f"- **Events**: {f.archive_events or 'n/a'}",
        f"- **Kernels (archive)**: {f.archive_kernels or 'n/a'}",
        f"- **Kernel names (archive)**: {', '.join(f.archive_kernel_names) or 'n/a'}",
        f"- **Complete**: {f.archive_complete or 'n/a'}",
        f"- **Archive / reader format**: "
        f"{f.archive_format_version if f.archive_format_version is not None else 'n/a'}"
        f" / {f.reader_format_version if f.reader_format_version is not None else 'n/a'}",
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

    finalize(finding)

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
