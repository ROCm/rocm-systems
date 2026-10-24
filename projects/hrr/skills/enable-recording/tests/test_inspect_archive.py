#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unit tests for inspect_archive.py.

Run: python3 -m pytest tests/test_inspect_archive.py

The archive layouts here are built to match what the capture writer produces:
a root manifest listing processes, and one manifest per pid directory carrying
pid, parent_pid, complete, event_count and blob_count.
"""

from __future__ import annotations

import json
import os
import re
import signal
import struct
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import inspect_archive  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def make_process(
    root: Path,
    pid: int,
    *,
    events: int | None = 100,
    blobs: int | None = 5,
    complete: bool | None = True,
    events_bytes: int = 4096,
    write_manifest: bool = True,
    format_version: int = 5,
) -> Path:
    pid_dir = root / f"pid-{pid}"
    (pid_dir / "blobs").mkdir(parents=True, exist_ok=True)
    # Real shape: hrr_file_header { magic, version, reserved }, then a whole
    # hrr_event_header whose payload_length counts itself, then the payload.
    header = struct.pack("<IHH", inspect_archive.HEADER_MAGIC, format_version, 0)
    body = b""
    if events_bytes:
        remaining = max(events_bytes - len(header), 0)
        if remaining >= inspect_archive.EVENT_HEADER_BYTES:
            payload_length = remaining
            body = struct.pack("<HQQQI2x", 1, 1, 0, 0, payload_length)
            body += b"\0" * (remaining - inspect_archive.EVENT_HEADER_BYTES)
        else:
            body = b"\0" * remaining
    (pid_dir / "events.bin").write_bytes((header + body) if events_bytes else b"")
    for i in range(blobs or 0):
        (pid_dir / "blobs" / f"blob-{i}").write_bytes(b"x")
    if write_manifest:
        manifest = {
            "pid": pid,
            "parent_pid": 1,
            "complete": complete,
            "event_count": events,
            "blob_count": blobs,
        }
        (pid_dir / "manifest.json").write_text(json.dumps(manifest))
    return pid_dir


def make_root(root: Path, processes: list[dict]) -> None:
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "version": 4,
                "capture_mode": "in-tree",
                "owner_pid": processes[0]["pid"] if processes else 0,
                "processes": processes,
            }
        )
    )


def test_single_process_archive_is_recorded(tmp_path):
    make_process(tmp_path, 42)
    report = inspect_archive.inspect(tmp_path, use_playback=False)

    assert len(report.processes) == 1
    assert report.recorded_processes
    proc = report.processes[0]
    assert proc.pid == 42
    assert proc.event_count == 100
    assert proc.complete is True
    assert "Verdict: recorded" in inspect_archive.render(report)


def test_a_pid_directory_can_be_passed_directly(tmp_path):
    pid_dir = make_process(tmp_path, 7)
    report = inspect_archive.inspect(pid_dir, use_playback=False)

    assert [p.pid for p in report.processes] == [7]


def test_a_pid_directory_passed_alone_is_its_own_archive(tmp_path):
    """The parent's manifest lists the other processes, which were not passed,
    so reading it warned that their real directories were missing.
    """
    pid_dir = make_process(tmp_path, 49)
    make_process(tmp_path, 50)
    make_root(tmp_path, [{"pid": 49, "event_count": 100}, {"pid": 50, "event_count": 100}])

    report = inspect_archive.inspect(pid_dir, use_playback=False)

    assert report.root == pid_dir
    assert report.root_manifest is None and report.root_manifest_error is None
    assert report.warnings == []
    assert inspect_archive.render(report).startswith(f"Archive: {pid_dir}")


def test_archive_with_no_process_directories_is_empty(tmp_path):
    report = inspect_archive.inspect(tmp_path, use_playback=False)

    assert not report.processes
    assert not report.recorded_processes
    rendered = inspect_archive.render(report)
    assert "Verdict: nothing captured" in rendered
    assert "preflight" in rendered


def test_process_directory_without_events_is_not_recorded(tmp_path):
    make_process(tmp_path, 9, events=0, blobs=0, events_bytes=0)
    report = inspect_archive.inspect(tmp_path, use_playback=False)

    assert report.processes
    assert not report.recorded_processes
    assert "Verdict: empty" in inspect_archive.render(report)


def test_incomplete_process_is_reported_but_still_recorded(tmp_path):
    """A crashed capture is the case worth keeping, not a failure."""
    make_process(tmp_path, 11, complete=False)
    report = inspect_archive.inspect(tmp_path, use_playback=False)

    assert report.recorded_processes
    rendered = inspect_archive.render(report)
    assert "incomplete" in rendered
    assert "still worth sending" in rendered
    # A lossy capture has no trailer either, and only the always-on line of
    # capture's stderr tells it apart: the louder one needs AMD_LOG_LEVEL.
    assert "Archive marked INCOMPLETE" in rendered


def test_events_without_a_manifest_are_reported_as_unfinalized(tmp_path):
    """Killed before finalizing: counts unknown, but the events are there."""
    make_process(tmp_path, 13, write_manifest=False)
    report = inspect_archive.inspect(tmp_path, use_playback=False)

    proc = report.processes[0]
    assert proc.recorded
    assert not proc.finalized
    assert proc.pid == 13, "pid must fall back to the directory name"
    assert any("died before finalizing" in w for w in report.warnings)
    assert "not finalized (no manifest)" in inspect_archive.render(report)


def test_stale_root_manifest_is_flagged_and_not_believed(tmp_path):
    """The root manifest is rewritten best-effort, so it can lag the truth."""
    make_process(tmp_path, 21, events=9_000_000)
    make_root(tmp_path, [{"pid": 21, "event_count": 4096, "blob_count": 5}])

    report = inspect_archive.inspect(tmp_path, use_playback=False)

    assert report.processes[0].event_count == 9_000_000
    assert any("authoritative" in w for w in report.warnings)


def test_root_manifest_listing_a_missing_process_is_flagged(tmp_path):
    make_process(tmp_path, 31)
    make_root(
        tmp_path,
        [
            {"pid": 31, "event_count": 100, "blob_count": 5},
            {"pid": 32, "event_count": 50, "blob_count": 1},
        ],
    )

    report = inspect_archive.inspect(tmp_path, use_playback=False)

    assert any("no pid-32/ directory" in w for w in report.warnings)


def test_size_on_disk_counts_the_whole_process_directory(tmp_path):
    """The manifests and the allocator's regions/ sidecars travel with the
    archive too, and leaving them out understated what is being sent.
    """
    pid_dir = make_process(tmp_path, 57)
    (pid_dir / "regions").mkdir()
    (pid_dir / "regions" / "torch-0.hrrr").write_bytes(b"r" * 1000)

    report = inspect_archive.inspect(tmp_path, use_playback=False)

    on_disk = sum(f.stat().st_size for f in pid_dir.rglob("*") if f.is_file())
    assert report.processes[0].total_bytes == on_disk
    assert report.total_bytes == on_disk


def test_multi_process_archive_says_to_keep_it_together(tmp_path):
    make_process(tmp_path, 100)
    make_process(tmp_path, 101)
    report = inspect_archive.inspect(tmp_path, use_playback=False)

    assert len(report.recorded_processes) == 2
    rendered = inspect_archive.render(report)
    assert "whole archive directory" in rendered
    # hrr-playback refuses a root holding several processes, and triage takes one.
    assert "one at a time" in rendered


def test_unreadable_root_manifest_does_not_hide_the_processes(tmp_path):
    make_process(tmp_path, 55)
    (tmp_path / "manifest.json").write_text("{not json")

    report = inspect_archive.inspect(tmp_path, use_playback=False)

    assert report.root_manifest_error
    assert report.recorded_processes


@pytest.mark.parametrize("content", ["[]", "1", '"text"', "[1]"])
def test_a_manifest_that_is_not_an_object_is_unreadable_not_fatal(tmp_path, content):
    """Valid JSON that is not an object crashed the inspector on its first
    `.get`, and this is the tool that reports on damaged captures.
    """
    pid_dir = make_process(tmp_path, 56)
    (pid_dir / "manifest.json").write_text(content)
    (tmp_path / "manifest.json").write_text(content)

    report = inspect_archive.inspect(tmp_path, use_playback=False)

    assert report.root_manifest_error
    assert report.processes[0].manifest_error
    assert report.recorded_processes, "the events are still there"
    # Said, rather than shown as an ordinary process that never finalized.
    assert any("pid-56: manifest.json is unreadable" in w for w in report.warnings)
    assert not any("died before finalizing" in w for w in report.warnings)
    assert "state: unknown (manifest unreadable)" in inspect_archive.render(report)


def test_manifest_fields_of_the_wrong_type_are_unknown_not_fatal(tmp_path):
    """`"event_count": "100"` reached `event_count > 0` and raised, so verify
    produced no report at all for a damaged archive.
    """
    pid_dir = make_process(tmp_path, 57)
    (pid_dir / "manifest.json").write_text(json.dumps({
        "pid": "57", "parent_pid": [1], "complete": "yes",
        "event_count": "100", "blob_count": True,
        "metadata": {"runtime": {"hip_runtime_version": 7}},
    }))
    make_root(tmp_path, [{"pid": [57], "event_count": 1}])

    report = inspect_archive.inspect(tmp_path, use_playback=False)
    rendered = inspect_archive.render(report)

    proc = report.processes[0]
    assert (proc.pid, proc.event_count, proc.blob_count, proc.complete) == (57, None, None, None)
    assert report.recorded_processes, "the events are still there"
    warning = next(w for w in report.warnings if "wrong type" in w)
    assert "event_count: str" in warning and "blob_count: bool" in warning
    assert not any("died before finalizing" in w for w in report.warnings)
    assert "state: unknown (manifest damaged)" in rendered


def test_a_negative_count_is_damaged_not_zero(tmp_path):
    """`"event_count": -1` passed the type check, `recorded` came out false and
    the verdict read empty although the events file holds a stream.
    """
    pid_dir = make_process(tmp_path, 60)
    (pid_dir / "manifest.json").write_text(json.dumps({
        "pid": 60, "complete": True, "event_count": -1, "blob_count": -3,
    }))

    report = inspect_archive.inspect(tmp_path, use_playback=False)

    proc = report.processes[0]
    assert (proc.event_count, proc.blob_count) == (None, None)
    assert report.recorded_processes, "the events are still there"
    warning = next(w for w in report.warnings if "out of range" in w)
    assert "event_count: -1" in warning and "blob_count: -3" in warning


@pytest.mark.parametrize("content", ["{}", '{"pid": 58}'])
def test_an_empty_or_partial_manifest_is_damaged_not_missing(tmp_path, content):
    """`{}` was skipped as if the file did not exist, and the process read as
    one killed before it could finalize.
    """
    pid_dir = make_process(tmp_path, 58)
    (pid_dir / "manifest.json").write_text(content)

    report = inspect_archive.inspect(tmp_path, use_playback=False)

    assert any("complete: missing" in w for w in report.warnings)
    assert not any("died before finalizing" in w for w in report.warnings)
    assert "state: unknown (manifest damaged)" in inspect_archive.render(report)


def test_a_null_manifest_field_is_damaged_not_missing(tmp_path):
    """`{"complete": null}` read as no manifest at all, and the process as one
    that died before finalizing, although the file was there.
    """
    pid_dir = make_process(tmp_path, 59)
    (pid_dir / "manifest.json").write_text(
        json.dumps({"pid": 59, "complete": None, "event_count": None})
    )

    report = inspect_archive.inspect(tmp_path, use_playback=False)

    warning = next(w for w in report.warnings if "wrong type" in w)
    assert "complete: NoneType" in warning and "event_count: NoneType" in warning
    assert not any("died before finalizing" in w for w in report.warnings)
    assert "state: unknown (manifest damaged)" in inspect_archive.render(report)


def test_a_pid_directory_without_events_is_a_leftover_not_a_capture(tmp_path):
    """The reader skips a pid-* directory with no events.bin, so the verdict
    has to as well: an output holding only one captured nothing.
    """
    stale = tmp_path / "pid-123"
    stale.mkdir()

    report = inspect_archive.inspect(tmp_path, use_playback=False)
    assert report.processes == []
    assert inspect_archive.verdict(report) == "nothing captured"

    direct = inspect_archive.inspect(stale, use_playback=False)
    assert direct.processes == []
    assert inspect_archive.verdict(direct) == "nothing captured"


def test_an_archive_moved_aside_is_not_a_process_of_the_capture(tmp_path):
    """The writer moves an earlier process's archive with the same pid to
    pid-<pid>.prev-<n>, and the reader skips it, so the verdict has to as well.
    """
    make_process(tmp_path, 61).rename(tmp_path / "pid-61.prev-1")
    make_process(tmp_path, 61)

    report = inspect_archive.inspect(tmp_path, use_playback=False)
    assert [p.pid for p in report.processes] == [61]


def test_the_json_report_carries_an_unreadable_root_manifest(tmp_path, capsys):
    make_process(tmp_path, 59)
    (tmp_path / "manifest.json").write_text("{not json")

    inspect_archive.main(["--archive", str(tmp_path), "--no-playback", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert payload["root_manifest_error"]
    assert any("root manifest.json is unreadable" in w for w in payload["warnings"])


def test_size_on_disk_counts_the_root_manifest_but_not_the_parent(tmp_path):
    """The whole archive directory is what gets sent; a single pid-* directory
    passed on its own is sent alone.
    """
    pid_dir = make_process(tmp_path, 58)
    make_root(tmp_path, [{"pid": 58, "event_count": 100, "blob_count": 5}])
    process_bytes = sum(f.stat().st_size for f in pid_dir.rglob("*") if f.is_file())
    root_bytes = (tmp_path / "manifest.json").stat().st_size

    assert inspect_archive.inspect(tmp_path, use_playback=False).total_bytes == (
        process_bytes + root_bytes
    )
    assert inspect_archive.inspect(pid_dir, use_playback=False).total_bytes == process_bytes


@pytest.mark.parametrize(
    "metadata", [[], None, "text", {"runtime": None}, {"runtime": []}, {"runtime": "7.16"}]
)
def test_metadata_of_the_wrong_type_is_not_fatal(tmp_path, metadata):
    pid_dir = make_process(tmp_path, 57)
    (pid_dir / "manifest.json").write_text(
        json.dumps({"pid": 57, "complete": True, "metadata": metadata})
    )

    report = inspect_archive.inspect(tmp_path, use_playback=False)

    assert report.processes[0].hip_runtime_version is None
    assert report.recorded_processes


def test_version_mismatch_from_playback_is_explained(tmp_path, monkeypatch):
    """A reader that cannot read the archive is a tooling problem, not a bad capture.

    The fixture is real output from a reader built on a different release line
    than the runtime that captured. Format versions do not track the tool, so
    this happens whenever the two come from different lines.
    """
    make_process(tmp_path, 77)
    fixture = (FIXTURES / "playback_info_version_mismatch.txt").read_text()

    monkeypatch.setattr(inspect_archive, "_find_playback", lambda explicit: "/usr/bin/hrr-playback")
    monkeypatch.setattr(
        inspect_archive, "_run_playback_info", lambda binary, target, timeout: (fixture, None)
    )

    report = inspect_archive.inspect(tmp_path, use_playback=True)

    assert any("cannot read this archive" in w for w in report.warnings)
    assert report.recorded_processes, "the archive itself is still fine"


def test_playback_info_is_included_verbatim(tmp_path, monkeypatch):
    make_process(tmp_path, 78)
    fixture = (FIXTURES / "playback_info_pass.txt").read_text()

    monkeypatch.setattr(inspect_archive, "_find_playback", lambda explicit: "/usr/bin/hrr-playback")
    monkeypatch.setattr(
        inspect_archive, "_run_playback_info", lambda binary, target, timeout: (fixture, None)
    )

    report = inspect_archive.inspect(tmp_path, use_playback=True)
    rendered = inspect_archive.render(report)

    assert "hrr-playback --info:" in rendered
    for line in fixture.splitlines():
        if line.strip():
            assert line.strip() in rendered


def test_playback_cross_check_uses_the_substantive_process(tmp_path, monkeypatch):
    """Not the alphabetically first one.

    Real capture: pid-406 held 10,512 registration events and pid-45 held
    258,042 including every kernel. Sorted as text, pid-406 comes first, so the
    cross-check reported on the process with nothing in it.
    """
    make_process(tmp_path, 406, events=10_512, blobs=4_082)
    make_process(tmp_path, 45, events=258_042, blobs=4_515)
    seen = {}

    monkeypatch.setattr(inspect_archive, "_find_playback", lambda explicit: "/usr/bin/hrr-playback")
    monkeypatch.setattr(
        inspect_archive,
        "_run_playback_info",
        lambda binary, target, timeout: (seen.update(target=target), ("info", None))[1],
    )

    inspect_archive.inspect(tmp_path, use_playback=True)

    assert seen["target"].name == "pid-45"


def test_packaged_playback_gets_its_own_libraries_on_the_path(tmp_path):
    """A packaged build ships bin/, lib/ and runtime-lib/ as siblings.

    Run without those on the library path, the binary loads whatever
    libamdhip64 the host has and dies on a missing symbol version, which looks
    like a broken archive rather than a broken invocation.
    """
    root = tmp_path / "playback-build"
    (root / "bin").mkdir(parents=True)
    (root / "lib").mkdir()
    (root / "runtime-lib").mkdir()
    binary = root / "bin" / "hrr-playback"
    binary.write_text("#!/bin/sh\n")

    env = inspect_archive._playback_env(str(binary))

    assert env["LD_LIBRARY_PATH"].split(":")[:2] == [str(root / "lib"), str(root / "runtime-lib")]


def test_build_tree_playback_gets_rocr_lib_on_the_path(tmp_path, monkeypatch):
    """As triage_archive.sh does: an in-tree reader may need the matching
    libhsa-runtime64, and ROCR_LIB is where it is."""
    binary = tmp_path / "build" / "playback" / "hrr-playback"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\n")
    rocr = tmp_path / "rocr" / "lib"
    rocr.mkdir(parents=True)
    (rocr / "libhsa-runtime64.so.1").write_text("")
    monkeypatch.setenv("ROCR_LIB", str(rocr))
    monkeypatch.setenv("LD_LIBRARY_PATH", "/mine")
    monkeypatch.setenv("ROCM_PATH", str(tmp_path / "no-rocm"))

    env = inspect_archive._playback_env(str(binary))

    assert env["LD_LIBRARY_PATH"] == f"{rocr}:/mine"

    # A ROCR_LIB without the library in it is not put on the path.
    (rocr / "libhsa-runtime64.so.1").unlink()
    assert inspect_archive._playback_env(str(binary))["LD_LIBRARY_PATH"] == "/mine"


def test_build_tree_playback_gets_rocm_lib_last(tmp_path, monkeypatch):
    """As ensure_playback.sh and triage_archive.sh do: the rest of ROCm comes
    from $ROCM_PATH/lib, after the caller's own path so the system release
    does not bind ahead of it."""
    binary = tmp_path / "build" / "playback" / "hrr-playback"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\n")
    rocm = tmp_path / "rocm"
    (rocm / "lib").mkdir(parents=True)
    monkeypatch.delenv("ROCR_LIB", raising=False)
    monkeypatch.setenv("ROCM_PATH", str(rocm))
    monkeypatch.setenv("LD_LIBRARY_PATH", "/mine")

    assert inspect_archive._playback_env(str(binary))["LD_LIBRARY_PATH"] == f"/mine:{rocm / 'lib'}"

    monkeypatch.delenv("LD_LIBRARY_PATH")
    assert inspect_archive._playback_env(str(binary))["LD_LIBRARY_PATH"] == str(rocm / "lib")


def test_missing_playback_is_reported_not_fatal(tmp_path, monkeypatch):
    make_process(tmp_path, 79)
    monkeypatch.setattr(inspect_archive, "_find_playback", lambda explicit: None)

    report = inspect_archive.inspect(tmp_path, use_playback=True)

    assert report.playback_error
    assert report.recorded_processes


def test_missing_archive_says_capture_never_started(tmp_path, capsys):
    """A workload that died before its first HIP call leaves no directory.

    Reporting that as a usage error hides the one useful fact: capture never
    ran, which is a different problem from an archive that came out empty.
    """
    rc = inspect_archive.main(["--archive", str(tmp_path / "never-created.hrr"), "--no-playback"])
    out = capsys.readouterr().out

    assert rc == inspect_archive.EXIT_EMPTY
    assert "Verdict: no archive" in out
    assert "capture never started" in out
    assert "preflight" in out


def test_exit_code_is_one_when_nothing_was_captured(tmp_path, capsys):
    rc = inspect_archive.main(["--archive", str(tmp_path), "--no-playback"])
    capsys.readouterr()
    assert rc == inspect_archive.EXIT_EMPTY


def test_exit_code_is_zero_when_something_was_captured(tmp_path, capsys):
    make_process(tmp_path, 88)
    rc = inspect_archive.main(["--archive", str(tmp_path), "--no-playback"])
    capsys.readouterr()
    assert rc == inspect_archive.EXIT_OK


def test_json_output_is_machine_readable(tmp_path, capsys):
    make_process(tmp_path, 99, complete=False)
    inspect_archive.main(["--archive", str(tmp_path), "--no-playback", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert payload["recorded"] is True
    assert payload["processes"][0]["complete"] is False
    assert payload["processes"][0]["pid"] == 99


def make_elf(sections: dict[str, bytes], allocated: tuple[str, ...] = ()) -> bytes:
    """A minimal ELF64 little-endian image holding `sections`, then .shstrtab.

    Sections named in `allocated` carry SHF_ALLOC.
    """
    names = b"\0"
    offsets = {}
    for name in [*sections, ".shstrtab"]:
        offsets[name] = len(names)
        names += name.encode() + b"\0"
    body = b"\0" * 64
    headers = [struct.pack("<IIQQQQIIQQ", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)]
    for name, data in [*sections.items(), (".shstrtab", names)]:
        headers.append(
            struct.pack(
                "<IIQQQQIIQQ",
                offsets[name], 1, 2 if name in allocated else 0, 0, len(body), len(data), 0, 0, 1, 0,
            )
        )
        body += data
    ident = b"\x7fELF" + bytes([2, 1, 1]) + b"\0" * 9
    header = ident + struct.pack(
        "<HHIQQQIHHHHHH", 3, 224, 1, 0, 0, len(body), 0, 64, 0, 0, 64, len(headers), len(headers) - 1
    )
    return header + body[64:] + b"".join(headers)


def make_bundle(elf: bytes) -> bytes:
    """An uncompressed offload bundle: an empty host entry, then `elf`."""
    host, target = b"host-x86_64-unknown-linux-gnu-", b"hipv4-amdgcn-amd-amdhsa--gfx950"
    table = inspect_archive.BUNDLE_MAGIC + struct.pack("<Q", 2)
    start = len(table) + 2 * 24 + len(host) + len(target)
    table += struct.pack("<QQQ", start, 0, len(host)) + host
    table += struct.pack("<QQQ", start, len(elf), len(target)) + target
    return table + elf


def register_fat_binary(
    pid_dir: Path,
    lo: int,
    hi: int,
    data: bytes,
    event_id: int = inspect_archive.REGISTER_FAT_BINARY[9],
    blob_size: int | None = None,
) -> None:
    """Store `data` as the capture writer stores a fat binary, and append the
    `__hipRegisterFatBinary` record that names it to `events.bin`, with
    `blob_size` in the record when given, else the size of `data`.
    """
    name = f"{lo:016x}{hi:016x}"
    (pid_dir / "blobs" / name[:2]).mkdir(parents=True, exist_ok=True)
    (pid_dir / "blobs" / name[:2] / f"{name}.blob").write_bytes(data)
    record = struct.pack("<HQQQI2x", event_id, 2, 0, 0, 64)
    record += struct.pack("<QQQQ", 0, lo, hi, len(data) if blob_size is None else blob_size)
    with (pid_dir / "events.bin").open("ab") as handle:
        handle.write(record)


def test_verify_reports_hardware_identifiers_and_build_info(tmp_path, capsys):
    """A capture that will be shared should name neither the GPU nor the build
    tree, and `verify` is where whoever approves the transfer looks.
    """
    leaky = make_process(tmp_path, 31, format_version=9)
    manifest = json.loads((leaky / "manifest.json").read_text())
    manifest["metadata"] = {
        "devices": [{"properties": {"name": "gfx950", "pci": "0:134:0", "uuid": "ab12", "luid": ""}}]
    }
    (leaky / "manifest.json").write_text(json.dumps(manifest))
    (leaky / "code_objects").mkdir()
    (leaky / "code_objects" / "a.hsaco").write_bytes(make_elf({".comment": b"AMD clang version 23\0"}))
    register_fat_binary(
        leaky, 0xFF01, 1, make_bundle(make_elf({".text": b"\1" * 8, ".debug_line_str": b"/app/build\0"}))
    )
    register_fat_binary(leaky, 0xFF02, 2, b"CCOB" + b"\1" * 20)

    clean = make_process(tmp_path, 32)
    manifest = json.loads((clean / "manifest.json").read_text())
    manifest["metadata"] = {"devices": [{"properties": {"pci": "", "uuid": "", "luid": ""}}]}
    (clean / "manifest.json").write_text(json.dumps(manifest))
    (clean / "code_objects").mkdir()
    (clean / "code_objects" / "a.hsaco").write_bytes(
        make_elf({".text": b"\1" * 8, ".comment": b"\0" * 21, ".debug_str": b"\0" * 11})
    )
    # Buffer contents are not code objects, whatever bytes they hold.
    (clean / "blobs" / "blob-0").write_bytes(make_elf({".comment": b"user data\0"})[4:])

    inspect_archive.main(["--archive", str(tmp_path), "--no-playback", "--json"])
    procs = {p["pid"]: p for p in json.loads(capsys.readouterr().out)["processes"]}
    assert procs[31]["hardware_ids"] == ["pci", "uuid"]
    assert procs[31]["debug_info_files"] == 2
    assert procs[31]["unchecked_fat_binaries"] == 1
    assert procs[32]["hardware_ids"] == []
    assert procs[32]["debug_info_files"] == 0
    assert procs[32]["unchecked_fat_binaries"] == 0

    inspect_archive.main(["--archive", str(tmp_path), "--no-playback"])
    text = capsys.readouterr().out
    assert (
        "hardware identifiers: pci, uuid   .comment/.debug* in 2 code objects, "
        "absolute paths in 0, 1 compressed fat binaries not checked" in text
    )
    assert "hardware identifiers: blank   .comment/.debug* in 0 code objects, absolute paths in 0" in text


def test_an_allocated_comment_still_counts_as_build_info(tmp_path, capsys):
    """Capture zeroes only non-allocated sections, so an allocated .comment
    reaches the archive as it was; `verify` must not call it clean.
    """
    pid_dir = make_process(tmp_path, 33)
    (pid_dir / "code_objects").mkdir()
    (pid_dir / "code_objects" / "a.hsaco").write_bytes(
        make_elf({".comment": b"/home/dev/build\0"}, allocated=(".comment",))
    )
    inspect_archive.main(["--archive", str(tmp_path), "--no-playback", "--json"])
    assert json.loads(capsys.readouterr().out)["processes"][0]["debug_info_files"] == 1


def test_an_absolute_path_in_loaded_data_is_reported(tmp_path, capsys):
    """A device assert() keeps its source file name in .rodata, which the GPU
    loads, so capture leaves it and `verify` has to say it is there. Machine
    code and sections the GPU never loads are not read as strings.
    """
    leaky = make_process(tmp_path, 35)
    (leaky / "code_objects").mkdir()
    (leaky / "code_objects" / "a.hsaco").write_bytes(
        make_elf({".rodata": b"p >= 0\0/home/dev/src/k.hip\0"}, allocated=(".rodata",))
    )
    clean = make_process(tmp_path, 36)
    (clean / "code_objects").mkdir()
    (clean / "code_objects" / "a.hsaco").write_bytes(
        make_elf({".rodata": b"p >= 0\0k.hip\0//\0C:\\k.hip\0", ".note": b"\0/a/b\0"}, allocated=(".rodata",))
    )
    text = bytearray(make_elf({".text": b"\0/a/b\0"}, allocated=(".text",)))
    struct.pack_into("<Q", text, len(text) - 2 * 64 + 8, 6)  # .text: SHF_ALLOC | SHF_EXECINSTR
    (clean / "code_objects" / "b.hsaco").write_bytes(text)

    inspect_archive.main(["--archive", str(tmp_path), "--no-playback", "--json"])
    procs = {p["pid"]: p for p in json.loads(capsys.readouterr().out)["processes"]}
    assert procs[35]["path_files"] == 1
    assert procs[36]["path_files"] == 0
    assert procs[35]["debug_info_files"] == procs[36]["debug_info_files"] == 0


@pytest.mark.parametrize(
    "string",
    [
        b"/home/dev/build tree/k.hip",
        "/home/dév/src/k.hip".encode(),
        b"C:\\Users\\dev\\k.hip",
        b"D:/src/k.hip",
    ],
)
def test_paths_with_spaces_non_ascii_or_a_drive_letter_are_reported(tmp_path, capsys, string):
    """A build tree may sit under a directory with a space or a non-ASCII name,
    or on a Windows drive; its assert() strings are build paths all the same.
    """
    pid_dir = make_process(tmp_path, 37)
    (pid_dir / "code_objects").mkdir()
    (pid_dir / "code_objects" / "a.hsaco").write_bytes(
        make_elf({".rodata": b"p >= 0\0" + string + b"\0"}, allocated=(".rodata",))
    )
    inspect_archive.main(["--archive", str(tmp_path), "--no-playback", "--json"])
    assert json.loads(capsys.readouterr().out)["processes"][0]["path_files"] == 1


def test_only_blobs_registered_as_fat_binaries_are_checked(tmp_path, capsys):
    """`blobs/` also holds host buffers, whose bytes can be anything: a 4-byte
    buffer "CCOB", or an offload bundle the application keeps in memory. Only
    the blobs a `__hipRegisterFatBinary` record names are fat binaries.
    """
    pid_dir = make_process(tmp_path, 34, format_version=9)
    leaky = make_bundle(make_elf({".text": b"\1" * 8, ".debug_str": b"/app/build\0"}))
    for name, data in (("aa" + "0" * 30, b"CCOB"), ("ab" + "0" * 30, leaky)):
        (pid_dir / "blobs" / name[:2]).mkdir()
        (pid_dir / "blobs" / name[:2] / f"{name}.blob").write_bytes(data)
    register_fat_binary(pid_dir, 0xAC, 3, leaky)

    inspect_archive.main(["--archive", str(tmp_path), "--no-playback", "--json"])
    proc = json.loads(capsys.readouterr().out)["processes"][0]
    assert proc["debug_info_files"] == 1
    assert proc["unchecked_fat_binaries"] == 0


def test_a_registration_naming_a_code_object_names_no_blob(tmp_path, capsys):
    """A registration record with a blob size of 0 names the code object HIP
    loaded, which is under `code_objects/` and scanned there. A blob that
    happens to have that hash is a host buffer, not a fat binary.
    """
    pid_dir = make_process(tmp_path, 38, format_version=10)
    leaky = make_bundle(make_elf({".text": b"\1" * 8, ".debug_str": b"/app/build\0"}))
    register_fat_binary(pid_dir, 0xAD, 4, leaky, inspect_archive.REGISTER_FAT_BINARY[10], blob_size=0)
    register_fat_binary(pid_dir, 0xAE, 5, b"CCOB", inspect_archive.REGISTER_FAT_BINARY[10], blob_size=0)

    inspect_archive.main(["--archive", str(tmp_path), "--no-playback", "--json"])
    proc = json.loads(capsys.readouterr().out)["processes"][0]
    assert proc["debug_info_files"] == 0
    assert proc["unchecked_fat_binaries"] == 0
    assert proc["fat_binaries_scanned"] is True


def test_fat_binaries_are_found_by_the_id_of_the_archive_version(tmp_path, capsys):
    """`__hipRegisterFatBinary` was id 549 in format v8 and is 550 in v9 and
    v10. A v7 archive may hold either, so `verify` says it did not look rather
    than report no build information.
    """
    leaky = make_bundle(make_elf({".text": b"\1" * 8, ".debug_str": b"/app/build\0"}))
    cases = {40: (8, 549), 41: (8, 550), 42: (9, 550), 43: (10, 550), 39: (7, 549)}
    for pid, (version, event_id) in cases.items():
        register_fat_binary(make_process(tmp_path, pid, format_version=version), pid, 1, leaky, event_id)

    inspect_archive.main(["--archive", str(tmp_path), "--no-playback", "--json"])
    procs = {p["pid"]: p for p in json.loads(capsys.readouterr().out)["processes"]}
    assert [procs[pid]["debug_info_files"] for pid in cases] == [1, 0, 1, 1, 0]
    assert [procs[pid]["fat_binaries_scanned"] for pid in cases] == [True, True, True, True, False]
    inspect_archive.main(["--archive", str(tmp_path), "--no-playback"])
    assert "fat binaries not checked (archive format v7)" in capsys.readouterr().out


def test_a_section_name_ends_inside_its_string_table(tmp_path, capsys):
    """Capture skips a section whose name has no terminator inside the string
    table; `verify` must not complete `.comm` into `.comment` from the bytes
    after the table either.
    """
    pid_dir = make_process(tmp_path, 44)
    (pid_dir / "code_objects").mkdir()
    elf = make_elf({".comment": b"AMD clang version 23\0"})
    (pid_dir / "code_objects" / "a.hsaco").write_bytes(elf)
    cut = bytearray(elf)
    struct.pack_into("<Q", cut, len(cut) - 64 + 32, 1 + len(".comm"))  # .shstrtab's sh_size
    (pid_dir / "code_objects" / "b.hsaco").write_bytes(cut)

    inspect_archive.main(["--archive", str(tmp_path), "--no-playback", "--json"])
    assert json.loads(capsys.readouterr().out)["processes"][0]["debug_info_files"] == 1


def test_the_json_verdict_is_one_of_the_four_documented_ones(tmp_path, capsys):
    """`--json` used to know only `recorded` and `empty`, so a consumer could
    not tell an archive with no events from a directory no process ever opened.
    """
    (tmp_path / "manifest.json").write_text("{}")

    inspect_archive.main(["--archive", str(tmp_path), "--no-playback", "--json"])
    assert json.loads(capsys.readouterr().out)["verdict"] == "nothing captured"

    make_process(tmp_path, 75, events=0, blobs=0, events_bytes=52)
    inspect_archive.main(["--archive", str(tmp_path), "--no-playback", "--json"])
    assert json.loads(capsys.readouterr().out)["verdict"] == "empty"

    make_process(tmp_path, 76)
    inspect_archive.main(["--archive", str(tmp_path), "--no-playback", "--json"])
    assert json.loads(capsys.readouterr().out)["verdict"] == "recorded"


def test_the_reader_comes_from_hrr_playback_first(tmp_path, monkeypatch):
    """Same order as the sibling skill's ensure_playback.sh, so one host does
    not verify with one reader and replay with another.
    """
    named = tmp_path / "from-env" / "hrr-playback"
    named.parent.mkdir()
    named.write_text("#!/bin/sh\n")
    named.chmod(0o755)
    rocm = tmp_path / "rocm"
    (rocm / "bin").mkdir(parents=True)
    (rocm / "bin" / "hrr-playback").write_text("#!/bin/sh\n")
    (rocm / "bin" / "hrr-playback").chmod(0o755)

    monkeypatch.setenv("HRR_PLAYBACK", str(named))
    monkeypatch.setenv("ROCM_PATH", str(rocm))
    monkeypatch.setattr(inspect_archive.shutil, "which", lambda name: "/usr/bin/hrr-playback")

    assert inspect_archive._find_playback(None) == str(named)

    # Only a file that can run, as its `-x` does: one that cannot is passed over.
    named.chmod(0o644)
    assert inspect_archive._find_playback(None) == str(rocm / "bin" / "hrr-playback")

    monkeypatch.delenv("HRR_PLAYBACK")
    assert inspect_archive._find_playback(None) == str(rocm / "bin" / "hrr-playback")

    # PATH last, and only when ROCm has none.
    (rocm / "bin" / "hrr-playback").unlink()
    assert inspect_archive._find_playback(None) == "/usr/bin/hrr-playback"


def test_preflight_sees_the_reader_hrr_playback_names(tmp_path):
    """Preflight said there was no reader while verify went on to use the one
    HRR_PLAYBACK named.
    """
    reader = tmp_path / "hrr-playback"
    reader.write_text("#!/bin/sh\n")
    reader.chmod(0o755)
    env = {**os.environ, "HRR_PLAYBACK": str(reader), "ROCM_PATH": str(tmp_path / "no-rocm")}

    result = subprocess.run(
        ["bash", str(SCRIPT), "preflight"], capture_output=True, text=True, env=env, timeout=120
    )

    assert "hrr-playback is available" in result.stderr


def test_preflight_refuses_a_directory_as_the_reader(tmp_path):
    """`-x` is true of a directory, which verify then refuses as a reader."""
    env = {**os.environ, "ROCM_PATH": str(tmp_path / "no-rocm")}

    result = subprocess.run(
        ["bash", str(SCRIPT), "preflight", "--playback", str(tmp_path)],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )

    assert result.returncode != 0
    assert "not an executable file" in result.stderr


def test_json_is_still_json_when_there_is_no_archive(tmp_path, capsys):
    """The case a script most needs to recognise, and the one that printed prose.

    A pre-send gate parses this output. Emitting a paragraph on the one path
    that says capture never ran defeats the gate exactly when it matters.
    """
    rc = inspect_archive.main(
        ["--archive", str(tmp_path / "never-created.hrr"), "--no-playback", "--json"]
    )
    payload = json.loads(capsys.readouterr().out)

    assert rc == inspect_archive.EXIT_EMPTY
    assert payload["verdict"] == "no archive"
    assert payload["recorded"] is False


def test_an_unusable_playback_path_is_an_error_not_a_skip(tmp_path, capsys):
    """A typo in --playback used to skip the cross-check and exit 0."""
    make_process(tmp_path, 55)
    rc = inspect_archive.main(
        ["--archive", str(tmp_path), "--playback", str(tmp_path / "no-such-binary")]
    )

    assert rc == inspect_archive.EXIT_USAGE
    assert "not an executable file" in capsys.readouterr().err

    # A gate parsing --json gets JSON for this too, not prose on stderr.
    rc = inspect_archive.main(
        ["--archive", str(tmp_path), "--playback", str(tmp_path / "no-such-binary"), "--json"]
    )
    assert rc == inspect_archive.EXIT_USAGE
    assert json.loads(capsys.readouterr().out)["verdict"] == "usage error"


def test_an_archive_path_that_is_a_file_is_a_usage_error(tmp_path, capsys):
    """Not an empty capture: exit 1 would read as "nothing recorded"."""
    events = tmp_path / "events.bin"
    events.write_bytes(b"")

    rc = inspect_archive.main(["--archive", str(events), "--no-playback"])

    assert rc == inspect_archive.EXIT_USAGE
    assert f"not a directory: {events}" in capsys.readouterr().err


def test_blob_count_including_code_objects_is_not_a_discrepancy(tmp_path):
    """The manifest counts code objects among its blobs; blobs/ and --info do not.

    Real archive: manifest 8, blob files 7, code objects 1. Three numbers, one
    truth, and the report used to show them side by side with no explanation.
    """
    pid_dir = make_process(tmp_path, 61, blobs=7)
    (pid_dir / "code_objects").mkdir()
    (pid_dir / "code_objects" / "co-0").write_bytes(b"x")
    (pid_dir / "manifest.json").write_text(
        json.dumps({"pid": 61, "complete": True, "event_count": 89, "blob_count": 8})
    )

    report = inspect_archive.inspect(tmp_path, use_playback=False)
    rendered = inspect_archive.render(report)

    assert report.warnings == []
    assert "blobs: 7" in rendered, "report the files on disk, which --info agrees with"


def test_a_blob_count_matching_nothing_on_disk_is_flagged(tmp_path):
    pid_dir = make_process(tmp_path, 62, blobs=3)
    (pid_dir / "manifest.json").write_text(
        json.dumps({"pid": 62, "complete": True, "event_count": 10, "blob_count": 99})
    )

    report = inspect_archive.inspect(tmp_path, use_playback=False)

    assert any("neither total" in w for w in report.warnings)


def test_a_reader_that_cannot_name_itself_is_reported_as_unverified(tmp_path, monkeypatch):
    """The silent failure: a reader from another build accepts the archive and
    decodes the event names against its own API table, with no error at all."""
    pid_dir = make_process(tmp_path, 63)
    (pid_dir / "manifest.json").write_text(
        json.dumps(
            {
                "pid": 63,
                "complete": True,
                "event_count": 100,
                "blob_count": 5,
                "metadata": {"runtime": {"hip_runtime_version": "7.16.26320"}},
            }
        )
    )

    monkeypatch.setattr(inspect_archive, "_find_playback", lambda explicit: "/usr/bin/hrr-playback")
    monkeypatch.setattr(
        inspect_archive, "_playback_version", lambda binary, timeout=20: (None, None)
    )
    monkeypatch.setattr(
        inspect_archive,
        "_run_playback_info",
        lambda binary, target, timeout: ("Events:       100\nBlobs:        5", None),
    )

    report = inspect_archive.inspect(tmp_path, use_playback=True)

    assert any("unverified" in w for w in report.warnings)


def test_a_reader_that_never_started_is_a_failed_cross_check(tmp_path, monkeypatch):
    """Observed on a real box: the reader could not load its own libraries, and
    the loader error was printed under the `hrr-playback --info:` heading with a
    `Verdict: recorded` under it and exit 0. A crash is not archive information.
    """
    make_process(tmp_path, 65)

    class Completed:
        returncode = 127
        stdout = ""
        stderr = (
            "/reader/bin/hrr-playback: error while loading shared libraries: "
            "librocprofiler-register.so.0: cannot open shared object file"
        )

    monkeypatch.setattr(inspect_archive.subprocess, "run", lambda *a, **k: Completed())

    info, error = inspect_archive._run_playback_info("/reader/bin/hrr-playback", tmp_path, 10)

    assert info is None
    assert "exited 127" in error
    assert "librocprofiler-register" in error


def test_a_reader_that_prints_nothing_is_a_failed_cross_check(tmp_path, monkeypatch):
    """Exit 0 and no output returned neither text nor an error, so the report
    failed its exit status with nothing to say why.
    """
    make_process(tmp_path, 66)

    class Completed:
        returncode = 0
        stdout = ""
        stderr = "  \n"

    monkeypatch.setattr(inspect_archive.subprocess, "run", lambda *a, **k: Completed())

    info, error = inspect_archive._run_playback_info("/reader/bin/hrr-playback", tmp_path, 10)

    assert info is None
    assert error == "hrr-playback --info exited 0 and printed nothing"


def test_a_reader_that_refuses_the_archive_still_explains_itself(tmp_path, monkeypatch):
    """Non-zero, but the output is the reader's own verdict rather than a crash."""

    class Completed:
        returncode = 1
        stdout = "[HRR] Version mismatch: file=4 reader=5"
        stderr = ""

    monkeypatch.setattr(inspect_archive.subprocess, "run", lambda *a, **k: Completed())

    info, error = inspect_archive._run_playback_info("/usr/bin/hrr-playback", tmp_path, 10)

    assert error is None
    assert inspect_archive.VERSION_MISMATCH in info


def test_a_header_only_events_file_is_not_a_recording(tmp_path):
    """`events.bin` opens with an 8-byte header and a clean shutdown appends a
    44-byte trailer, so bytes on disk are not evidence that anything was
    recorded. The manifest's count decides where there is one.
    """
    make_process(tmp_path, 67, events=0, blobs=0, events_bytes=52)

    report = inspect_archive.inspect(tmp_path, use_playback=False)

    assert report.recorded_processes == []
    assert "Verdict: empty" in inspect_archive.render(report)


def test_a_process_killed_before_its_manifest_is_still_a_recording(tmp_path):
    make_process(tmp_path, 68, events_bytes=4096, write_manifest=False)

    report = inspect_archive.inspect(tmp_path, use_playback=False)

    assert len(report.recorded_processes) == 1


def test_a_damaged_root_entry_does_not_hide_the_processes(tmp_path):
    """The root manifest is rewritten best-effort, so it can be valid JSON with
    a `processes` entry that is not an object. The pid directories still count.
    """
    make_process(tmp_path, 79)
    (tmp_path / "manifest.json").write_text(json.dumps({"version": 4, "processes": [None]}))

    report = inspect_archive.inspect(tmp_path, use_playback=False)

    assert len(report.recorded_processes) == 1
    assert any("not an object: NoneType" in w for w in report.warnings)


def test_a_torn_first_event_header_is_not_a_recording(tmp_path):
    """A process killed mid-write can leave a header cut in half, and half a
    header is not an event. Nor is one whose declared length cannot fit.
    """
    pid_dir = make_process(tmp_path, 77, write_manifest=False)
    header = struct.pack("<IHH", inspect_archive.HEADER_MAGIC, 5, 0)

    (pid_dir / "events.bin").write_bytes(header + b"\x01\x00" + b"\0" * 10)
    assert inspect_archive.inspect(tmp_path, use_playback=False).recorded_processes == []

    # A whole header claiming a record longer than the file holds.
    (pid_dir / "events.bin").write_bytes(header + struct.pack("<HQQQI2x", 1, 1, 0, 0, 1 << 20))
    assert inspect_archive.inspect(tmp_path, use_playback=False).recorded_processes == []

    # One longer than the reader takes, in a file that does hold it.
    too_long = inspect_archive.MAX_RECORD_BYTES + 1
    with (pid_dir / "events.bin").open("wb") as handle:
        handle.write(header + struct.pack("<HQQQI2x", 1, 1, 0, 0, too_long))
        handle.truncate(len(header) + too_long)
    assert inspect_archive.inspect(tmp_path, use_playback=False).recorded_processes == []

    # A plausible event behind a file header that is not HRR's.
    bad_header = struct.pack("<IHH", 0, 5, 0)
    (pid_dir / "events.bin").write_bytes(bad_header + struct.pack("<HQQQI2x", 1, 1, 0, 0, 32))
    assert inspect_archive.inspect(tmp_path, use_playback=False).recorded_processes == []


def test_framing_without_events_is_not_a_recording_either(tmp_path):
    """Header plus clean trailer is 52 bytes and no events at all, so a byte
    threshold cannot decide this. The first record says what it is.
    """
    pid_dir = make_process(tmp_path, 74, write_manifest=False)
    header = struct.pack("<IHH", inspect_archive.HEADER_MAGIC, 5, 0)
    trailer = struct.pack(
        "<HQQQI2xQI", inspect_archive.EOF_MARKER, 0, 0, 0,
        inspect_archive.EOF_RECORD_BYTES, 0, inspect_archive.EOF_MAGIC,
    )
    (pid_dir / "events.bin").write_bytes(header + trailer)

    report = inspect_archive.inspect(tmp_path, use_playback=False)

    assert report.recorded_processes == []


def test_the_trailer_sentinel_alone_does_not_end_the_recording(tmp_path):
    """hrr_reader.cpp keeps a 0xFFFF record that is not a whole trailer with
    its magic as an unknown event, so the inspector counts it as one too.
    """
    pid_dir = make_process(tmp_path, 78, write_manifest=False)
    header = struct.pack("<IHH", inspect_archive.HEADER_MAGIC, 5, 0)
    unknown = struct.pack("<HQQQI2x", inspect_archive.EOF_MARKER, 0, 0, 0, 32)
    (pid_dir / "events.bin").write_bytes(header + unknown)

    report = inspect_archive.inspect(tmp_path, use_playback=False)

    assert len(report.recorded_processes) == 1


def test_the_reader_version_banner_is_parsed_not_compared_as_a_string(tmp_path, monkeypatch):
    """`--version` prints a banner, the format version, the revision, and the
    HIP runtime as an integer. Comparing the banner against the dotted HIP
    version the capture records warns on every correctly matched reader.
    """

    class Completed:
        returncode = 0
        stdout = (
            "hrr-playback (HIP Record & Replay)\n"
            "  archive format version : 5\n"
            "  source revision        : 50048dee07\n"
            "  source branch          : develop\n"
            "  build type             : Release\n"
            "  hip runtime version    : 70160000\n"
        )
        stderr = ""

    monkeypatch.setattr(inspect_archive.subprocess, "run", lambda *a, **k: Completed())

    identity, format_version = inspect_archive._playback_version("/usr/bin/hrr-playback")

    assert format_version == 5
    assert "v5" in identity and "50048dee07" in identity


def test_a_reader_reading_another_archive_format_is_flagged(tmp_path, monkeypatch):
    make_process(tmp_path, 69, format_version=4)
    make_root(tmp_path, [{"pid": 69, "event_count": 100, "blob_count": 5}])

    monkeypatch.setattr(inspect_archive, "_find_playback", lambda explicit: "/usr/bin/hrr-playback")
    monkeypatch.setattr(
        inspect_archive,
        "_playback_version",
        lambda binary, timeout=20: ("reads archive format v9", 9),
    )
    monkeypatch.setattr(
        inspect_archive,
        "_run_playback_info",
        lambda binary, target, timeout: ("Events:       100\nBlobs:        5", None),
    )

    report = inspect_archive.inspect(tmp_path, use_playback=True)

    assert any("format v4" in w and "v9" in w for w in report.warnings)


def test_a_cross_check_that_did_not_run_does_not_exit_zero(tmp_path, capsys, monkeypatch):
    make_process(tmp_path, 66)
    reader = tmp_path / "hrr-playback"
    reader.write_text("#!/bin/sh\n")
    reader.chmod(0o755)

    monkeypatch.setattr(
        inspect_archive,
        "_run_playback_info",
        lambda binary, target, timeout: (None, "hrr-playback --info exited 127: loader error"),
    )
    monkeypatch.setattr(
        inspect_archive, "_playback_version", lambda binary, timeout=20: (None, None)
    )

    rc = inspect_archive.main(["--archive", str(tmp_path), "--playback", str(reader)])
    capsys.readouterr()

    assert rc == inspect_archive.EXIT_NO_CROSS_CHECK


def test_a_reader_that_names_itself_but_not_the_format_is_still_unverified(tmp_path, monkeypatch):
    """Keying the warning on the identity string let a reader suppress it by
    printing a banner and nothing else, which is the case the warning is for.
    """
    make_process(tmp_path, 71)

    monkeypatch.setattr(inspect_archive, "_find_playback", lambda explicit: "/usr/bin/hrr-playback")
    monkeypatch.setattr(
        inspect_archive,
        "_playback_version",
        lambda binary, timeout=20: ("hrr-playback (HIP Record & Replay)", None),
    )
    monkeypatch.setattr(
        inspect_archive,
        "_run_playback_info",
        lambda binary, target, timeout: ("Events:       100\nBlobs:        5\nKernels:      3", None),
    )

    report = inspect_archive.inspect(tmp_path, use_playback=True)

    assert any("unverified" in w for w in report.warnings)


def test_a_refused_archive_is_not_a_completed_cross_check(tmp_path, capsys, monkeypatch):
    """The reader explained itself, and the archive still went unread."""
    make_process(tmp_path, 72)
    fixture = (FIXTURES / "playback_info_version_mismatch.txt").read_text()

    monkeypatch.setattr(inspect_archive, "_find_playback", lambda explicit: "/usr/bin/hrr-playback")
    monkeypatch.setattr(inspect_archive, "_playback_version", lambda binary, timeout=20: (None, None))
    monkeypatch.setattr(
        inspect_archive, "_run_playback_info", lambda binary, target, timeout: (fixture, None)
    )

    rc = inspect_archive.main(["--archive", str(tmp_path)])
    capsys.readouterr()

    assert rc == inspect_archive.EXIT_NO_CROSS_CHECK


def test_an_archive_with_no_kernel_launches_is_flagged(tmp_path, monkeypatch):
    """Observed when a runtime mismatch killed the workload at hipSetDevice:
    eight events, `complete`, `recorded`, and nothing ever ran on the GPU.
    """
    make_process(tmp_path, 73, events=8)

    monkeypatch.setattr(inspect_archive, "_find_playback", lambda explicit: "/usr/bin/hrr-playback")
    monkeypatch.setattr(
        inspect_archive, "_playback_version", lambda binary, timeout=20: ("reads v5", 5)
    )
    monkeypatch.setattr(
        inspect_archive,
        "_run_playback_info",
        lambda binary, target, timeout: ("Events:       8\nKernels:      0\nBlobs:        5", None),
    )

    report = inspect_archive.inspect(tmp_path, use_playback=True)

    assert any("no kernel launches" in w for w in report.warnings)


def test_no_reader_on_the_machine_is_not_a_failed_cross_check(tmp_path, capsys, monkeypatch):
    """The manifests-only path is documented and normal. Exiting non-zero here
    would turn every host without a reader red for an archive that is fine.
    """
    make_process(tmp_path, 70)
    monkeypatch.setattr(inspect_archive, "_find_playback", lambda explicit: None)

    rc = inspect_archive.main(["--archive", str(tmp_path)])
    capsys.readouterr()

    assert rc == inspect_archive.EXIT_OK


def test_a_reader_counting_different_events_is_flagged(tmp_path, monkeypatch):
    make_process(tmp_path, 64, events=100, blobs=5)

    monkeypatch.setattr(inspect_archive, "_find_playback", lambda explicit: "/usr/bin/hrr-playback")
    monkeypatch.setattr(
        inspect_archive,
        "_playback_version",
        lambda binary, timeout=20: ("reads archive format v4", 4),
    )
    monkeypatch.setattr(
        inspect_archive,
        "_run_playback_info",
        lambda binary, target, timeout: ("Events:       7\nBlobs:        5", None),
    )

    report = inspect_archive.inspect(tmp_path, use_playback=True)

    assert any("reader counts 7" in w for w in report.warnings)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


# --- the wrapper script's launcher handling ---------------------------------
#
# Not the inspector, but it belongs with the suite the CI runs: preflight
# resolving the wrong executable is how a capture comes back empty.

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "hrr_capture.sh"


def _strip_launchers(command: str) -> str:
    function = re.search(
        r"^strip_launchers\(\).*?^\}", SCRIPT.read_text(encoding="utf-8"), re.M | re.S
    )
    assert function, "strip_launchers is gone from the script"
    result = subprocess.run(
        ["bash", "-c", f"{function.group(0)}\nstrip_launchers {command}"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


@pytest.mark.parametrize(
    "command, expected",
    [
        ("python train.py", "python"),
        # env's assignments come first, one a line, so preflight can apply them.
        ("env FOO=1 python train.py", "FOO=1\npython"),
        ("env A=1 nice -n 5 env B=2 app", "A=1\nB=2\napp"),
        # The operand cases: skipping only dashed arguments used to leave the
        # duration or the niceness as the workload.
        ("timeout 60s python train.py", "python"),
        ("timeout -s KILL 30 python train.py", "python"),
        ("nice -n 10 vllm serve", "vllm"),
        ("ionice -c 2 -n 4 python train.py", "python"),
        ("stdbuf -oL python train.py", "python"),
        # Launchers whose grammar we do not model stop the search at themselves
        # rather than guessing.
        ("srun -N2 python train.py", "srun"),
        # env's own options change the environment in ways an assignment
        # cannot carry, and used to come back as the workload themselves.
        ("env -u LD_PRELOAD app", "env"),
        ("env --ignore-environment app", "env"),
        ("/usr/bin/env -i A=1 app", "/usr/bin/env"),
        ("env -- A=1 app", "A=1\napp"),
    ],
)
def test_the_workload_is_found_behind_its_launcher(command, expected):
    assert _strip_launchers(command) == expected


@pytest.mark.parametrize(
    "shebang, expected",
    [
        ("#!/venv/bin/python3", "/venv/bin/python3"),
        ("#!/usr/bin/env python3", "python3"),
        # -S used to come back as the interpreter, and the package scan ran it.
        ("#!/usr/bin/env -S python3 -u", "python3"),
        ("#!/usr/bin/env -Spython3", "python3"),
        ("#!/usr/bin/env --split-string=python3 -u", "python3"),
        ("#!/usr/bin/env -S PYTHONUNBUFFERED=1 python3", "python3"),
        # An option we do not model is not a binary: fall back to the PATH's.
        ("#!/usr/bin/env -i python3", "FALLBACK"),
    ],
)
def test_the_interpreter_is_read_off_the_shebang(tmp_path, shebang, expected):
    function = re.search(
        r"^workload_interpreter\(\).*?^\}", SCRIPT.read_text(encoding="utf-8"), re.M | re.S
    )
    assert function, "workload_interpreter is gone from the script"
    script = tmp_path / "console-script"
    script.write_text(f"{shebang}\nimport sys\n")
    # The PATH's python3 is what the fallback answers, so make it recognisable.
    bindir = tmp_path / "bin"
    bindir.mkdir()
    fallback = bindir / "python3"
    fallback.write_text("#!/bin/sh\n")
    fallback.chmod(0o755)
    result = subprocess.run(
        ["/bin/bash", "-c", f"{function.group(0)}\nworkload_interpreter"],
        capture_output=True,
        text=True,
        env={**os.environ, "WORKLOAD_BIN": str(script), "PATH": str(bindir)},
        check=True,
    )
    assert result.stdout.strip() == (str(fallback) if expected == "FALLBACK" else expected)


def test_preflight_resolves_with_the_launchers_environment():
    """`env LD_LIBRARY_PATH=/capture/lib app` runs app with that path, and
    preflight resolved its runtime with the parent's path instead. The
    assignments apply to that check only, not to the rest of the script.
    """
    function = re.search(
        r"^check_runtime_as_launched\(\).*?^\}", SCRIPT.read_text(encoding="utf-8"), re.M | re.S
    )
    assert function, "check_runtime_as_launched is gone from the script"
    result = subprocess.run(
        ["bash", "-c",
         "check_runtime() { sh -c 'echo \"in=$LD_LIBRARY_PATH|$LD_PRELOAD\"'; }\n"
         f"{function.group(0)}\n"
         "LD_LIBRARY_PATH=/parent; export LD_LIBRARY_PATH\n"
         "LAUNCHER_ENV=(LD_LIBRARY_PATH=/capture/lib LD_PRELOAD=/capture/libamdhip64.so.7 'not a name=x')\n"
         "check_runtime_as_launched\n"
         'echo "after=$LD_LIBRARY_PATH|${LD_PRELOAD-unset}"'],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.split() == [
        "in=/capture/lib|/capture/libamdhip64.so.7",
        "after=/parent|unset",
    ]


def _resolved_runtime(ldd_output: str, binary: Path) -> str:
    """Run the script's resolver with a stubbed ldd."""
    function = re.search(
        r"^resolved_runtime\(\).*?^\}", SCRIPT.read_text(encoding="utf-8"), re.M | re.S
    )
    assert function, "resolved_runtime is gone from the script"
    stub = f'ldd() {{ cat <<"LDD"\n{ldd_output}\nLDD\n}}\n'
    result = subprocess.run(
        ["bash", "-c", f'{stub}{function.group(0)}\nresolved_runtime "{binary}" || true'],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def test_ldd_version_errors_do_not_defeat_resolution(tmp_path):
    """A binary built against a newer ROCm than the host's makes ldd print
    symbol-version errors first, and those lines mention libamdhip64 too. The
    third field of one of them is the word `version`, which resolved to
    nothing and made preflight blame an interpreter for a compiled binary.
    """
    runtime = tmp_path / "libamdhip64.so.7"
    runtime.write_bytes(b"\0")
    binary = tmp_path / "workload"
    binary.write_bytes(b"\0")

    noisy = (
        f"{binary}: {runtime}: version `hip_7.17' not found (required by {binary})\n"
        "\tlinux-vdso.so.1 (0x00007fff)\n"
        f"\tlibamdhip64.so.7 => {runtime} (0x00007f00)"
    )
    assert _resolved_runtime(noisy, binary) == str(runtime)

    # An interpreter links no HIP runtime of its own, and that has to stay
    # unresolved rather than resolving to something arbitrary.
    assert _resolved_runtime("\tlibc.so.6 => /usr/lib/libc.so.6 (0x00007f00)", binary) == ""


def test_a_runtime_preloaded_by_path_is_the_one_resolved(tmp_path):
    """ldd prints a library preloaded by absolute path as `/path (0x...)` and
    prints no `=>` line for the binary's own entry, which the preload
    satisfies. Reading `=>` lines only refused the LD_PRELOAD remedy the skill
    recommends for a compiled binary.
    """
    runtime = tmp_path / "libamdhip64.so.7"
    runtime.write_bytes(b"\0")
    binary = tmp_path / "workload"
    binary.write_bytes(b"\0")

    preloaded = (
        f"\t{runtime} (0x00007fffff75e000)\n"
        "\tlinux-vdso.so.1 (0x00007fff)\n"
        "\tlibc.so.6 => /usr/lib/libc.so.6 (0x00007f00)"
    )
    assert _resolved_runtime(preloaded, binary) == str(runtime)


def _bash_version() -> tuple[int, int]:
    result = subprocess.run(
        ["bash", "-c", "echo ${BASH_VERSINFO[0]} ${BASH_VERSINFO[1]}"],
        capture_output=True,
        text=True,
        check=True,
    )
    major, minor = result.stdout.split()
    return int(major), int(minor)


def _run(tmp_path: Path, *workload: str, **popen) -> subprocess.Popen:
    """Start `run` on a Python workload, with no preflight and no reader."""
    return subprocess.Popen(
        ["bash", str(SCRIPT), "run", "--skip-preflight", "--output", str(tmp_path / "capture"),
         "--no-playback", "--", sys.executable, *workload],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        **popen,
    )


def test_the_wrapper_waits_for_the_workload_it_signalled(tmp_path):
    """A trapped signal ends `wait` early. The wrapper took that for the
    workload's exit, verified an archive still being written, and returned
    while the workload was still shutting down.
    """
    ready, done = tmp_path / "ready", tmp_path / "done"
    workload = tmp_path / "workload.py"
    workload.write_text(
        "import signal, sys, time\n"
        "def stop(*_):\n"
        "    time.sleep(1)\n"
        f"    open({str(done)!r}, 'w').close()\n"
        "    sys.exit(0)\n"
        "signal.signal(signal.SIGTERM, stop)\n"
        f"open({str(ready)!r}, 'w').close()\n"
        "time.sleep(60)\n"
    )
    wrapper = _run(tmp_path, str(workload))
    try:
        deadline = time.monotonic() + 30
        while not ready.exists():
            assert wrapper.poll() is None and time.monotonic() < deadline, "workload never started"
            time.sleep(0.05)
        wrapper.send_signal(signal.SIGTERM)
        _, stderr = wrapper.communicate(timeout=30)
    finally:
        if wrapper.poll() is None:
            wrapper.kill()

    assert done.exists(), "the wrapper returned before the workload had finished"
    assert "workload exited 0" in stderr
    assert wrapper.returncode == 0


def test_a_signal_reaches_the_workers_a_launcher_started(tmp_path):
    """The signal went to the wrapper's direct child only. A launcher that
    exits on it left its workers capturing, and the wrapper verified the
    archive they were still writing.
    """
    ready, done = tmp_path / "ready", tmp_path / "done"
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import signal, sys, time\n"
        "def stop(*_):\n"
        "    time.sleep(1)\n"
        f"    open({str(done)!r}, 'w').close()\n"
        "    sys.exit(0)\n"
        "signal.signal(signal.SIGTERM, stop)\n"
        f"open({str(ready)!r}, 'w').close()\n"
        "time.sleep(60)\n"
    )
    launcher = tmp_path / "launcher.py"
    launcher.write_text(
        "import subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, {str(worker)!r}])\n"
        "time.sleep(60)\n"
    )
    wrapper = _run(tmp_path, str(launcher))
    try:
        deadline = time.monotonic() + 30
        while not ready.exists():
            assert wrapper.poll() is None and time.monotonic() < deadline, "worker never started"
            time.sleep(0.05)
        wrapper.send_signal(signal.SIGTERM)
        _, stderr = wrapper.communicate(timeout=30)
    finally:
        if wrapper.poll() is None:
            wrapper.kill()

    assert done.exists(), "the worker was not stopped, or the wrapper did not wait for it"
    assert "workload exited" in stderr


@pytest.mark.skipif(_bash_version() < (4, 3), reason="needs bash >= 4.3 (Linux target)")
def test_the_workload_can_be_interrupted(tmp_path):
    """A background job starts with SIGINT ignored and the workload inherited
    it, so a Ctrl-C, or the INT the wrapper forwards, left it running.
    """
    seen = tmp_path / "sigint"
    probe = (
        "import signal, sys\n"
        "open(sys.argv[1], 'w').write(str(signal.getsignal(signal.SIGINT) is signal.default_int_handler))"
    )
    wrapper = _run(
        tmp_path, "-c", probe, str(seen),
        preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL),
    )
    wrapper.communicate(timeout=60)

    assert seen.read_text() == "True"


def test_the_archive_directory_is_private_and_the_workload_umask_is_not(tmp_path):
    """The writer creates its files 0644, so under a 022 umask the archive was
    readable by every user on the host. The directory is created 0700 instead,
    and the workload keeps the umask it was started with.
    """
    seen = tmp_path / "umask"
    probe = "import os, sys\nopen(sys.argv[1], 'w').write(oct(os.umask(0)))"
    wrapper = _run(tmp_path, "-c", probe, str(seen), preexec_fn=lambda: os.umask(0o022))
    wrapper.communicate(timeout=60)

    assert (tmp_path / "capture").stat().st_mode & 0o777 == 0o700
    assert seen.read_text() == "0o22"


def _output_path_log(out: Path) -> str:
    """Run the script's output-path check with the filesystem lookups stubbed."""
    function = re.search(
        r"^check_output_path\(\).*?^\}", SCRIPT.read_text(encoding="utf-8"), re.M | re.S
    )
    assert function, "check_output_path is gone from the script"
    stubs = (
        "log() { printf '%s\\n' \"$*\" >&2; }\n"
        "fail() { printf 'error: %s\\n' \"$*\" >&2; exit 1; }\n"
        'path_fstype() { echo "fs-of:$1"; }\n'
        "df() { printf 'Filesystem 1024-blocks Used Available Capacity Mounted\\n"
        "stub 1 0 104857600 0%% /\\n'; }\n"
    )
    result = subprocess.run(
        ["bash", "-c", f'{stubs}{function.group(0)}\ncheck_output_path "{out}" 0'],
        capture_output=True,
        text=True,
    )
    return result.stderr


def test_an_output_directory_that_exists_is_checked_itself(tmp_path):
    """`-v /host/captures:/data/captures --output /data/captures`: the parent
    `/data` is root-owned and on the container's overlay, and checking it
    refused a writable bind mount, as not writable or as dying with the
    container.
    """
    out = tmp_path / "captures"
    out.mkdir()
    assert f"filesystem: fs-of:{out}," in _output_path_log(out)

    # Not there yet: capture creates it inside its parent, so the parent counts.
    assert f"filesystem: fs-of:{tmp_path}," in _output_path_log(tmp_path / "run.hrr")


@pytest.mark.skipif(os.geteuid() == 0, reason="root passes every permission check")
def test_an_output_directory_without_search_permission_is_refused(tmp_path):
    """Creating `pid-*/` takes search permission as well as write, so a
    directory with only `-w` passed preflight and the run recorded nothing.
    """
    out = tmp_path / "captures"
    out.mkdir()
    out.chmod(0o600)
    try:
        assert f"error: not writable (needs write and search permission): {out}" in (
            _output_path_log(out)
        )
    finally:
        out.chmod(0o700)


def test_an_output_path_that_is_not_a_directory_is_refused(tmp_path):
    """Capture cannot create its archive directory on top of a file, and a
    preflight that passed left the run to end with no archive at all.
    """
    regular = tmp_path / "run.hrr"
    regular.write_text("")
    assert f"error: {regular} exists and is not a directory" in _output_path_log(regular)

    dangling = tmp_path / "gone.hrr"
    dangling.symlink_to(tmp_path / "nowhere")
    assert f"error: {dangling} exists and is not a directory" in _output_path_log(dangling)
