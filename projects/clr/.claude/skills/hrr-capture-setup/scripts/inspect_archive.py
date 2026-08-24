#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Report what an HRR capture archive actually contains.

Answers the only question that matters right after a capture run: did anything
get recorded, and is the archive worth sending on. It works without a GPU and
without `hrr-playback`, reading the manifests the capture writer leaves behind,
and uses `hrr-playback --info` as a cross-check when a compatible one is around.

Two properties of the on-disk layout drive the logic here:

* The **root manifest is written best-effort by whichever process last flushed
  it**, so its per-process counts can lag the truth. The per-process
  `pid-<pid>/manifest.json` is authoritative and is what this reports.
* A process killed hard may leave `events.bin` with **no manifest at all**.
  That is a capture with unknown counts, not an empty one, so it is reported as
  recorded-but-unfinalized rather than as a failure.

`complete: false` is likewise not an error. It means the clean-shutdown trailer
is missing, which is exactly what a workload that crashed is supposed to look
like.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

EXIT_OK = 0
EXIT_EMPTY = 1
EXIT_USAGE = 2

# Emitted by hrr_reader.cpp when the archive format and the reader disagree.
VERSION_MISMATCH = "Version mismatch"


@dataclass
class ProcessArchive:
    path: Path
    pid: int | None = None
    parent_pid: int | None = None
    complete: bool | None = None
    event_count: int | None = None
    blob_count: int | None = None
    events_bytes: int = 0
    blob_files: int = 0
    code_object_files: int = 0
    total_bytes: int = 0
    manifest_error: str | None = None

    @property
    def recorded(self) -> bool:
        """Something is in this process directory worth keeping."""
        if self.events_bytes > 0:
            return True
        return bool(self.event_count)

    @property
    def finalized(self) -> bool:
        return self.complete is not None


@dataclass
class ArchiveReport:
    root: Path
    processes: list[ProcessArchive] = field(default_factory=list)
    root_manifest: dict | None = None
    root_manifest_error: str | None = None
    total_bytes: int = 0
    playback_info: str | None = None
    playback_error: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def recorded_processes(self) -> list[ProcessArchive]:
        return [p for p in self.processes if p.recorded]


def _dir_stats(path: Path) -> tuple[int, int]:
    """(file count, total bytes) for a directory tree, cheaply and safely."""
    files = 0
    total = 0
    if not path.is_dir():
        return files, total
    for entry in path.rglob("*"):
        try:
            if entry.is_file():
                files += 1
                total += entry.stat().st_size
        except OSError:
            continue
    return files, total


def _read_manifest(path: Path) -> tuple[dict | None, str | None]:
    if not path.is_file():
        return None, None
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace")), None
    except (OSError, ValueError) as exc:
        return None, str(exc)


def _load_process(pid_dir: Path) -> ProcessArchive:
    proc = ProcessArchive(path=pid_dir)

    events = pid_dir / "events.bin"
    try:
        proc.events_bytes = events.stat().st_size if events.is_file() else 0
    except OSError:
        proc.events_bytes = 0

    proc.blob_files, blob_bytes = _dir_stats(pid_dir / "blobs")
    proc.code_object_files, code_bytes = _dir_stats(pid_dir / "code_objects")

    manifest, error = _read_manifest(pid_dir / "manifest.json")
    proc.manifest_error = error
    if manifest:
        proc.pid = manifest.get("pid")
        proc.parent_pid = manifest.get("parent_pid")
        proc.complete = manifest.get("complete")
        proc.event_count = manifest.get("event_count")
        proc.blob_count = manifest.get("blob_count")

    if proc.pid is None and pid_dir.name.startswith("pid-"):
        suffix = pid_dir.name[len("pid-") :]
        if suffix.isdigit():
            proc.pid = int(suffix)

    proc.total_bytes = proc.events_bytes + blob_bytes + code_bytes
    return proc


def _resolve(archive: Path) -> tuple[Path, list[Path]]:
    """Return (root, pid directories). Accepts a root or a single pid dir."""
    pid_dirs = sorted(d for d in archive.glob("pid-*") if d.is_dir())
    if pid_dirs:
        return archive, pid_dirs
    if (archive / "events.bin").is_file() or archive.name.startswith("pid-"):
        return archive.parent, [archive]
    return archive, []


def _find_playback(explicit: str | None) -> str | None:
    if explicit:
        return explicit if Path(explicit).is_file() else None
    found = shutil.which("hrr-playback")
    if found:
        return found
    rocm = Path(os.environ.get("ROCM_PATH", "/opt/rocm"))
    candidate = rocm / "bin" / "hrr-playback"
    return str(candidate) if candidate.is_file() else None


def _playback_env(playback: str) -> dict[str, str]:
    """Environment for running a playback binary out of a packaged build.

    Such a build ships `bin/`, `lib/` and `runtime-lib/` as siblings, and the
    binary needs the HIP and HSA libraries from its own `lib/` and
    `runtime-lib/`. Without them it picks up whichever libamdhip64 the host has
    and dies on a missing symbol version, which reads like a broken archive.
    """
    env = dict(os.environ)
    root = Path(playback).resolve().parent.parent
    dirs = [str(root / name) for name in ("lib", "runtime-lib") if (root / name).is_dir()]
    if dirs:
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = ":".join(dirs + ([existing] if existing else []))
    return env


def _run_playback_info(playback: str, target: Path, timeout: int) -> tuple[str | None, str | None]:
    try:
        completed = subprocess.run(
            [playback, str(target), "--info"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=_playback_env(playback),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, str(exc)
    output = (completed.stdout or "") + (completed.stderr or "")
    output = output.strip()
    if completed.returncode != 0 and not output:
        return None, f"hrr-playback --info exited {completed.returncode} with no output"
    return output, None


def inspect(
    archive: Path,
    playback: str | None = None,
    use_playback: bool = True,
    timeout: int = 120,
) -> ArchiveReport:
    root, pid_dirs = _resolve(archive)
    report = ArchiveReport(root=root)

    manifest, error = _read_manifest(root / "manifest.json")
    report.root_manifest = manifest
    report.root_manifest_error = error

    for pid_dir in pid_dirs:
        proc = _load_process(pid_dir)
        report.processes.append(proc)
        report.total_bytes += proc.total_bytes

    if manifest and isinstance(manifest.get("processes"), list):
        by_pid = {p.pid: p for p in report.processes if p.pid is not None}
        for entry in manifest["processes"]:
            pid = entry.get("pid")
            proc = by_pid.get(pid)
            if proc is None:
                report.warnings.append(
                    f"root manifest lists pid {pid} but no pid-{pid}/ directory exists"
                )
                continue
            if (
                proc.event_count is not None
                and entry.get("event_count") is not None
                and entry["event_count"] != proc.event_count
            ):
                report.warnings.append(
                    f"root manifest says pid {pid} has {entry['event_count']} events, "
                    f"its own manifest says {proc.event_count}; the per-process count is "
                    "authoritative because the root is rewritten best-effort"
                )

    for proc in report.processes:
        if proc.recorded and not proc.finalized:
            report.warnings.append(
                f"{proc.path.name}: events.bin holds {proc.events_bytes} bytes but there is "
                "no manifest, so the process died before finalizing; counts are unknown "
                "until hrr-playback --info reads it"
            )

    if use_playback and report.recorded_processes:
        binary = _find_playback(playback)
        if binary:
            # The substantive process, not the alphabetically first one: pid-406
            # sorts before pid-45, and in a server capture the small process is
            # usually the one that only registered fat binaries.
            target = max(
                report.recorded_processes,
                key=lambda p: (p.event_count or 0, p.events_bytes),
            ).path
            report.playback_info, report.playback_error = _run_playback_info(
                binary, target, timeout
            )
            if report.playback_info and VERSION_MISMATCH in report.playback_info:
                report.warnings.append(
                    "this hrr-playback cannot read this archive: the capture wrote one "
                    "format version and this reader expects another. The archive is fine; "
                    "the reader has to match the runtime that captured it"
                )
        else:
            report.playback_error = "no hrr-playback found on PATH or under $ROCM_PATH/bin"

    return report


def _human(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TiB"


def render(report: ArchiveReport) -> str:
    lines = [f"Archive: {report.root}"]

    if report.root_manifest_error:
        lines.append(f"Root manifest unreadable: {report.root_manifest_error}")

    if not report.processes:
        lines.append("")
        lines.append(
            "The usual cause is that the runtime the workload loaded has no capture "
            "support, or that HIP_HRR_CAPTURE_OUTPUT never reached the process that "
            "used the GPU. Run 'hrr_capture.sh preflight' in the same environment."
        )
        lines.append("")
        lines.append(
            "Verdict: nothing captured. The directory exists but holds no pid-* "
            "process directory, so no process ever opened a capture here."
        )
        return "\n".join(lines)

    lines.append(f"Processes: {len(report.processes)}   Size on disk: {_human(report.total_bytes)}")
    lines.append("")
    for proc in report.processes:
        events = "unknown" if proc.event_count is None else f"{proc.event_count:,}"
        blobs = "unknown" if proc.blob_count is None else f"{proc.blob_count:,}"
        if proc.complete is None:
            state = "not finalized (no manifest)"
        elif proc.complete:
            state = "complete"
        else:
            state = "incomplete (no clean-shutdown trailer)"
        lines.append(f"  {proc.path.name}")
        lines.append(f"    events: {events}   blobs: {blobs}   state: {state}")
        lines.append(
            f"    events.bin: {_human(proc.events_bytes)}   "
            f"blob files: {proc.blob_files}   code objects: {proc.code_object_files}"
        )
        if proc.parent_pid:
            lines.append(f"    parent pid: {proc.parent_pid}")

    incomplete = [p for p in report.processes if p.complete is False]
    if incomplete:
        lines.append("")
        lines.append(
            "At least one process has no clean-shutdown trailer. That is what a workload "
            "that crashed looks like, and such an archive is still readable and still "
            "worth sending: the events before the crash are all there."
        )

    if report.warnings:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"  - {w}" for w in report.warnings)

    if report.playback_info:
        lines.append("")
        lines.append("hrr-playback --info:")
        lines.extend(f"  {line}" for line in report.playback_info.splitlines())
    elif report.playback_error:
        lines.append("")
        lines.append(f"Cross-check with hrr-playback skipped: {report.playback_error}")

    lines.append("")
    if report.recorded_processes:
        lines.append(
            f"Verdict: recorded. {len(report.recorded_processes)} of {len(report.processes)} "
            "process directories hold events."
        )
        if len(report.processes) > 1:
            lines.append(
                "Send or replay the whole archive directory, not one pid-* directory out "
                "of it; the processes belong together."
            )
    else:
        lines.append("Verdict: empty. Process directories exist but no events were written.")

    return "\n".join(lines)


def to_dict(report: ArchiveReport) -> dict:
    return {
        "archive": str(report.root),
        "total_bytes": report.total_bytes,
        "recorded": bool(report.recorded_processes),
        "processes": [
            {
                "dir": p.path.name,
                "pid": p.pid,
                "parent_pid": p.parent_pid,
                "complete": p.complete,
                "event_count": p.event_count,
                "blob_count": p.blob_count,
                "events_bytes": p.events_bytes,
                "blob_files": p.blob_files,
                "code_object_files": p.code_object_files,
                "finalized": p.finalized,
            }
            for p in report.processes
        ],
        "warnings": report.warnings,
        "playback_info": report.playback_info,
        "playback_error": report.playback_error,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--archive", required=True, help="capture root or a pid-* directory")
    parser.add_argument("--playback", help="hrr-playback binary to cross-check with")
    parser.add_argument(
        "--no-playback", action="store_true", help="skip the hrr-playback cross-check"
    )
    # --info walks the whole event stream, so a multi-million-event archive on a
    # slow filesystem takes tens of seconds. Two minutes is generous for that and
    # still bounds a reader that has wedged.
    parser.add_argument("--timeout", type=int, default=120, help="seconds for hrr-playback --info")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = parser.parse_args(argv)

    archive = Path(args.archive).expanduser()
    if not archive.exists():
        # Distinct from an empty archive: nothing was created at all, so capture
        # never ran. Usual causes are a workload that died before its first HIP
        # call, or a runtime with no capture support.
        print(
            f"Archive: {archive}\n\n"
            "The runtime wrote nothing here. Either the workload failed before its "
            "first HIP call, or the runtime it loaded has no capture support. Run "
            "'hrr_capture.sh preflight' in the same environment to tell those apart."
            "\n\nVerdict: no archive. The directory was never created, so capture "
            "never started."
        )
        return EXIT_EMPTY
    if not archive.is_dir():
        print(f"error: not a directory: {archive}", file=sys.stderr)
        return EXIT_USAGE

    report = inspect(
        archive.resolve(),
        playback=args.playback,
        use_playback=not args.no_playback,
        timeout=args.timeout,
    )

    print(json.dumps(to_dict(report), indent=2) if args.json else render(report))
    return EXIT_OK if report.recorded_processes else EXIT_EMPTY


if __name__ == "__main__":
    sys.exit(main())
