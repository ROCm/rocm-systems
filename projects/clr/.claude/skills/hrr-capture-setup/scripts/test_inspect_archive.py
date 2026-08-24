#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unit tests for inspect_archive.py.

Run: python3 -m pytest scripts/test_inspect_archive.py

The archive layouts here are built to match what the capture writer produces:
a root manifest listing processes, and one manifest per pid directory carrying
pid, parent_pid, complete, event_count and blob_count.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import inspect_archive  # noqa: E402

FIXTURES = Path(__file__).resolve().parent.parent / "evals" / "fixtures"


def make_process(
    root: Path,
    pid: int,
    *,
    events: int | None = 100,
    blobs: int | None = 5,
    complete: bool | None = True,
    events_bytes: int = 4096,
    write_manifest: bool = True,
) -> Path:
    pid_dir = root / f"pid-{pid}"
    (pid_dir / "blobs").mkdir(parents=True, exist_ok=True)
    (pid_dir / "events.bin").write_bytes(b"\0" * events_bytes)
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


def test_events_without_a_manifest_are_reported_as_unfinalized(tmp_path):
    """Killed before finalizing: counts unknown, but the events are there."""
    make_process(tmp_path, 13, write_manifest=False)
    report = inspect_archive.inspect(tmp_path, use_playback=False)

    proc = report.processes[0]
    assert proc.recorded
    assert not proc.finalized
    assert proc.pid == 13, "pid must fall back to the directory name"
    assert any("died before finalizing" in w for w in report.warnings)


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


def test_multi_process_archive_says_to_keep_it_together(tmp_path):
    make_process(tmp_path, 100)
    make_process(tmp_path, 101)
    report = inspect_archive.inspect(tmp_path, use_playback=False)

    assert len(report.recorded_processes) == 2
    assert "whole archive directory" in inspect_archive.render(report)


def test_unreadable_root_manifest_does_not_hide_the_processes(tmp_path):
    make_process(tmp_path, 55)
    (tmp_path / "manifest.json").write_text("{not json")

    report = inspect_archive.inspect(tmp_path, use_playback=False)

    assert report.root_manifest_error
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
