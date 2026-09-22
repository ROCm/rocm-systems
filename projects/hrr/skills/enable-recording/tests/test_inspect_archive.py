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
import struct
import sys
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
    # Real shape: hrr_file_header { magic, version, reserved } then events.
    header = struct.pack("<IHH", inspect_archive.HEADER_MAGIC, format_version, 0)
    body = b"\0" * max(events_bytes - len(header), 0) if events_bytes else b""
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
    rocm = tmp_path / "rocm"
    (rocm / "bin").mkdir(parents=True)
    (rocm / "bin" / "hrr-playback").write_text("#!/bin/sh\n")

    monkeypatch.setenv("HRR_PLAYBACK", str(named))
    monkeypatch.setenv("ROCM_PATH", str(rocm))
    monkeypatch.setattr(inspect_archive.shutil, "which", lambda name: "/usr/bin/hrr-playback")

    assert inspect_archive._find_playback(None) == str(named)

    monkeypatch.delenv("HRR_PLAYBACK")
    assert inspect_archive._find_playback(None) == str(rocm / "bin" / "hrr-playback")


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


def test_framing_without_events_is_not_a_recording_either(tmp_path):
    """Header plus clean trailer is 52 bytes and no events at all, so a byte
    threshold cannot decide this. The first record says what it is.
    """
    pid_dir = make_process(tmp_path, 74, write_manifest=False)
    header = struct.pack("<IHH", inspect_archive.HEADER_MAGIC, 5, 0)
    trailer = struct.pack("<H", inspect_archive.EOF_MARKER) + b"\0" * 42
    (pid_dir / "events.bin").write_bytes(header + trailer)

    report = inspect_archive.inspect(tmp_path, use_playback=False)

    assert report.recorded_processes == []


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
