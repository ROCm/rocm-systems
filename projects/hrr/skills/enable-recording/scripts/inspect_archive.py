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
import struct
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

EXIT_OK = 0
EXIT_EMPTY = 1
EXIT_USAGE = 2
# A cross-check the caller asked for and that did not happen. Distinct from a
# bad archive: the archive may be perfect and the reader unable to start. It
# gets its own code because exiting 0 here lets a pre-send gate pass on a check
# that never ran.
EXIT_NO_CROSS_CHECK = 3

# Emitted by hrr_reader.cpp when the archive format and the reader disagree.
VERSION_MISMATCH = "Version mismatch"

# `events.bin` is hrr_file_header(8) + events + optional hrr_eof_record(44).
# The header is `{ uint32 magic, uint16 version, uint16 reserved }`, little
# endian, and its version is the number the reader compares against its own
# when it says `Version mismatch: file=N reader=M`. The root manifest's
# `version` is a different quantity and comparing that one raises a false
# alarm on a perfectly matched reader.
# See projects/hrr/include/hrr/hrr_api_args.h and projects/hrr/DESIGN.md.
HEADER_BYTES = 8
HEADER_MAGIC = 0x52524845  # "HRRE"
# `hrr_event_header.event_type` sentinel for the clean-shutdown trailer. A file
# whose very first record is the trailer recorded nothing, however many bytes
# it holds: header plus trailer is 52 bytes of no events at all.
EOF_MARKER = 0xFFFF


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
    hip_runtime_version: str | None = None
    format_version: int | None = None
    holds_an_event: bool = False

    @property
    def recorded(self) -> bool:
        """Something is in this process directory worth keeping.

        Not "the file is non-empty": `events.bin` opens with an 8-byte header
        and a clean shutdown appends a 44-byte trailer, so a capture that
        recorded nothing at all still leaves bytes behind. The manifest's count
        is authoritative where it exists; a process killed before it could write
        one is decided by whether the first record in the file is a real event
        rather than the clean-shutdown trailer.
        """
        if self.event_count is not None:
            return self.event_count > 0
        return self.holds_an_event

    @property
    def finalized(self) -> bool:
        return self.complete is not None

    @property
    def blob_count_reconciles(self) -> bool | None:
        """Does the manifest's blob total agree with what is on disk?

        The manifest counts code objects among its blobs while `blobs/` and
        `hrr-playback --info` do not, so `blob_count` legitimately equals either
        the file count or the file count plus the code objects. Anything else is
        a real discrepancy. Unknown when there is no manifest.
        """
        if self.blob_count is None:
            return None
        return self.blob_count in (self.blob_files, self.blob_files + self.code_object_files)


@dataclass
class ArchiveReport:
    root: Path
    processes: list[ProcessArchive] = field(default_factory=list)
    root_manifest: dict | None = None
    root_manifest_error: str | None = None
    total_bytes: int = 0
    playback_info: str | None = None
    playback_error: str | None = None
    playback_binary: str | None = None
    playback_version: str | None = None
    playback_format_version: int | None = None
    capture_runtime: str | None = None
    # True only when a reader actually decoded the archive. A reader that
    # refused it explained itself, which is useful, but the archive still went
    # unread and a pre-send gate should not pass on that.
    cross_check_ran: bool = False
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


def _archive_format_version(events: Path) -> int | None:
    """The format version written in the first 8 bytes of `events.bin`."""
    try:
        with events.open("rb") as handle:
            head = handle.read(HEADER_BYTES)
    except OSError:
        return None
    if len(head) < HEADER_BYTES:
        return None
    magic, version, _ = struct.unpack("<IHH", head)
    return version if magic == HEADER_MAGIC else None


def _holds_an_event(events: Path) -> bool:
    """Is there a real event in this file, rather than framing alone?

    Only asked when a process died before writing its manifest, so the count
    has to come off the file. Bytes are not the answer: a clean trailer with no
    events, or a first record torn mid-write, both leave more than the header.
    """
    try:
        with events.open("rb") as handle:
            head = handle.read(HEADER_BYTES + 2)
    except OSError:
        return False
    if len(head) < HEADER_BYTES + 2:
        return False
    return struct.unpack_from("<H", head, HEADER_BYTES)[0] != EOF_MARKER


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

    proc.format_version = _archive_format_version(events)
    proc.holds_an_event = _holds_an_event(events)
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
        runtime = (manifest.get("metadata") or {}).get("runtime") or {}
        proc.hip_runtime_version = runtime.get("hip_runtime_version")

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
    """The reader to cross-check with, in the same order the sibling skill uses.

    `HRR_PLAYBACK`, then `$ROCM_PATH/bin`, then `PATH`, which is what
    `decode-and-triage/scripts/ensure_playback.sh` does. Searching in a
    different order meant the same host could verify an archive with one reader
    and replay it with another, on a skill whose whole point is that the reader
    has to match the runtime that captured.
    """
    if explicit:
        return explicit if Path(explicit).is_file() else None
    from_env = os.environ.get("HRR_PLAYBACK")
    if from_env and Path(from_env).is_file():
        return from_env
    rocm = Path(os.environ.get("ROCM_PATH", "/opt/rocm"))
    candidate = rocm / "bin" / "hrr-playback"
    if candidate.is_file():
        return str(candidate)
    return shutil.which("hrr-playback")


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


def _playback_version(playback: str, timeout: int = 20) -> tuple[str | None, int | None]:
    """What the reader says it is: (identity, archive format version it reads).

    `--version` prints a banner, then `archive format version`, the source
    revision and branch, and the HIP runtime version as an integer. The format
    version is the number worth comparing, because it is the same quantity the
    archive's own manifest carries; the HIP integer is not comparable with the
    dotted version capture records.

    Older builds have no `--version` at all, so `(None, None)` is common and is
    not an error. It does mean the pairing cannot be established, which
    matters: a reader built from a different API table accepts the archive and
    decodes the event names shifted, without complaining.
    """
    try:
        completed = subprocess.run(
            [playback, "--version"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=_playback_env(playback),
        )
    except (OSError, subprocess.SubprocessError):
        return None, None
    if completed.returncode != 0:
        return None, None
    output = ((completed.stdout or "") + (completed.stderr or "")).strip()
    if not output:
        return None, None

    fields = {}
    for line in output.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            fields[key.strip()] = value.strip()

    format_version = None
    raw = fields.get("archive format version")
    if raw and raw.isdigit():
        format_version = int(raw)

    identity = ", ".join(
        part
        for part in (
            f"reads archive format v{format_version}" if format_version is not None else None,
            f"rev {fields['source revision']}" if fields.get("source revision") else None,
        )
        if part
    )
    return (identity or output.splitlines()[0]), format_version


def _info_field(info: str, label: str) -> int | None:
    """An integer field out of `hrr-playback --info`, such as `Events:`."""
    for line in info.splitlines():
        stripped = line.strip()
        if stripped.startswith(f"{label}:"):
            value = stripped.split(":", 1)[1].strip().split()[0]
            if value.isdigit():
                return int(value)
    return None


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
    output = ((completed.stdout or "") + (completed.stderr or "")).strip()
    if completed.returncode != 0:
        # A reader that refused the archive explains itself, and that explanation
        # is worth keeping. A reader that never started prints a loader error,
        # and printing that under an `--info` heading passes a crash off as
        # archive information: it is a failed cross-check and has to read as one.
        if VERSION_MISMATCH in output:
            return output, None
        detail = output.splitlines()[0] if output else "no output"
        return None, f"hrr-playback --info exited {completed.returncode}: {detail}"
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
        if proc.blob_count_reconciles is False:
            report.warnings.append(
                f"{proc.path.name}: the manifest counts {proc.blob_count} blobs but the "
                f"directory holds {proc.blob_files} blob files and {proc.code_object_files} "
                "code objects, which is neither total; something is missing from disk"
            )
        if proc.hip_runtime_version and not report.capture_runtime:
            report.capture_runtime = proc.hip_runtime_version

    if use_playback and report.recorded_processes:
        binary = _find_playback(playback)
        if binary:
            report.playback_binary = binary
            report.playback_version, report.playback_format_version = _playback_version(binary)
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
            elif report.playback_info:
                report.cross_check_ran = True
                _cross_check(report, target)
        else:
            report.playback_error = (
                "no hrr-playback found on PATH or under $ROCM_PATH/bin, so this reports "
                "from the manifests alone. The sibling skill's "
                "decode-and-triage/scripts/ensure_playback.sh finds or builds one that "
                "matches this machine"
            )

    return report


def _cross_check(report: ArchiveReport, target: Path) -> None:
    """Compare what the reader says with what the archive says.

    A reader that refuses an archive says so. A reader from a neighbouring
    build does something worse: it accepts the archive and decodes the event
    names against its own API table, so the counts are right and the names are
    wrong, with no error anywhere. Nothing here can detect that from the
    output, so the reader is identified instead and an unestablished pairing is
    reported as exactly that.
    """
    info = report.playback_info or ""
    proc = next((p for p in report.processes if p.path == target), None)

    if proc is not None and proc.event_count is not None:
        reported = _info_field(info, "Events")
        if reported is not None and reported != proc.event_count:
            report.warnings.append(
                f"{proc.path.name}: the manifest records {proc.event_count:,} events and this "
                f"reader counts {reported:,}. Trust neither until the reader is the one that "
                "matches the capturing runtime"
            )

    kernels = _info_field(info, "Kernels")
    if kernels == 0:
        report.warnings.append(
            "this archive contains no kernel launches at all. The process made HIP calls "
            "and never reached the GPU, which is what a run that died during setup looks "
            "like: check it is the run that reproduces your failure before sending it"
        )

    if proc is not None:
        reported_blobs = _info_field(info, "Blobs")
        if reported_blobs is not None and reported_blobs != proc.blob_files:
            report.warnings.append(
                f"{proc.path.name}: this reader counts {reported_blobs:,} blobs against "
                f"{proc.blob_files:,} files in blobs/"
            )

    archive_format = proc.format_version if proc is not None else None
    if (
        isinstance(archive_format, int)
        and report.playback_format_version is not None
        and archive_format != report.playback_format_version
    ):
        report.warnings.append(
            f"the archive is format v{archive_format} and this reader reads "
            f"v{report.playback_format_version}. It has not refused the archive here, but "
            "the two do not agree and nothing it reports can be relied on"
        )
    elif report.playback_format_version is None:
        # Keyed on the format version rather than on the identity string: a
        # reader whose `--version` succeeds without printing that field still
        # establishes nothing, and testing the identity instead let it pass as
        # though it had.
        report.warnings.append(
            "this hrr-playback does not report the archive format it reads, so the pairing "
            "with the runtime that captured is unverified. A reader from a different build "
            "reads the archive without complaining and decodes the event names against its "
            "own API table: if "
            "the API mix below does not look like the workload, the reader is wrong rather "
            "than the archive"
        )


def _human(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TiB"


def verdict(report: ArchiveReport) -> str:
    """One of the four documented verdicts, for the text and JSON paths alike.

    `no archive` is decided before a report exists, in `missing_archive`.
    """
    if report.recorded_processes:
        return "recorded"
    if not report.processes:
        return "nothing captured"
    return "empty"


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
        # Files on disk rather than the manifest total, which counts code
        # objects among its blobs and so disagrees with both `blobs/` and
        # `hrr-playback --info` for the same archive.
        blobs = f"{proc.blob_files:,}"
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
        reader = report.playback_binary or "hrr-playback"
        version = report.playback_version or "version not reported"
        lines.append(f"Reader: {reader} ({version})")
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
        "playback_binary": report.playback_binary,
        "playback_version": report.playback_version,
        "playback_format_version": report.playback_format_version,
        "capture_runtime": report.capture_runtime,
        "verdict": verdict(report),
    }


def missing_archive(archive: Path) -> tuple[str, dict]:
    """The report for a path capture never created, in both output forms.

    Distinct from an empty archive: nothing was created at all, so capture
    never ran. Usual causes are a runtime with no capture compiled in, or a
    workload that died before its first HIP call.
    """
    text = (
        f"Archive: {archive}\n\n"
        "The runtime wrote nothing here. Either the runtime it loaded has no capture "
        "support, which is the common case, or the workload failed before its first HIP "
        "call. Run 'hrr_capture.sh preflight' with the workload command in the same "
        "environment to tell those apart."
        "\n\nVerdict: no archive. The directory was never created, so capture never started."
    )
    payload = {
        "archive": str(archive),
        "total_bytes": 0,
        "recorded": False,
        "processes": [],
        "warnings": [],
        "playback_info": None,
        "playback_error": None,
        "playback_binary": None,
        "playback_version": None,
        "playback_format_version": None,
        "capture_runtime": None,
        "verdict": "no archive",
    }
    return text, payload


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

    # A reader the caller named and that cannot run is a mistake to report, not
    # one to fall back from: a silently skipped cross-check exiting 0 passes a
    # scripted gate that was written to depend on it.
    if args.playback and not (Path(args.playback).is_file() and os.access(args.playback, os.X_OK)):
        message = f"--playback is not an executable file: {args.playback}"
        if args.json:
            print(json.dumps({"error": message, "verdict": "usage error"}, indent=2))
        else:
            print(f"error: {message}", file=sys.stderr)
        return EXIT_USAGE

    if not archive.exists():
        text, payload = missing_archive(archive)
        print(json.dumps(payload, indent=2) if args.json else text)
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
    if not report.recorded_processes:
        return EXIT_EMPTY
    # A reader that was found and then failed is a check that did not happen,
    # whoever asked for it. No reader on the machine at all is the documented
    # manifests-only path and not a failure, or every host without one would
    # report red.
    if not report.cross_check_ran and (args.playback or report.playback_binary):
        return EXIT_NO_CROSS_CHECK
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
