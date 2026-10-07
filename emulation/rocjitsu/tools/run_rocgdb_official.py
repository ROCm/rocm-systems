#!/usr/bin/env python3

# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run each official gdb.rocm test in a fresh verified Mirage daemon session."""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

# Every path below is resolved at startup from the repository this script lives
# in, an environment variable, or a command-line flag -- in that order of
# increasing precedence. Nothing is hard-coded to one developer's checkout, so
# the same run is reproducible in CI and in a second worktree.
SESSION_ROOT = Path(f"/run/user/{os.getuid()}/mirage/session")

# Resolved by configure() before main() does any work.
ROOT = Path()
SUITE_ROOT = Path()
TESTSUITE = Path()
TEST_DIR = Path()
MIRAGE = Path()
ROCJITSU = Path()
VENV = Path()
SDK = Path()
CORE = Path()
GDB = Path()


def default_root() -> Path:
    """Repository root, inferred from this file's location."""
    return Path(__file__).resolve().parents[3]


def env_path(name: str, fallback: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else fallback


def find_sdk_package(venv: Path, package: str) -> Path:
    """Locate a _rocm_sdk_* package without pinning the interpreter version."""
    matches = sorted(venv.glob(f"lib/python3.*/site-packages/{package}"))
    return matches[-1] if matches else venv / f"lib/python3/site-packages/{package}"


def configure(args: argparse.Namespace) -> None:
    global ROOT, SUITE_ROOT, TESTSUITE, TEST_DIR, MIRAGE, ROCJITSU, VENV, SDK, CORE, GDB
    ROOT = (args.root or env_path("ROCM_SYSTEMS_ROOT", default_root())).resolve()
    SUITE_ROOT = (
        args.rocgdb_suite or env_path("ROCGDB_SUITE", Path("/tmp/ROCgdb-tests"))
    ).resolve()
    TESTSUITE = SUITE_ROOT / "gdb/testsuite"
    TEST_DIR = TESTSUITE / "gdb.rocm"
    MIRAGE = args.mirage or ROOT / "emulation/mirage/target/debug/mirage"
    ROCJITSU = args.rocjitsu or ROOT / "emulation/rocjitsu/build/librocjitsu.so"
    VENV = (
        args.venv or env_path("ROCM_SDK_VENV", ROOT / "emulation/mirage/.venv-mi350")
    ).resolve()
    SDK = find_sdk_package(VENV, "_rocm_sdk_devel")
    CORE = find_sdk_package(VENV, "_rocm_sdk_core")
    GDB = (args.gdb or env_path("ROCGDB", Path("/tmp/ROCgdb-build/gdb/gdb"))).resolve()


STATUS_RE = re.compile(
    r"^(PASS|FAIL|XFAIL|XPASS|KFAIL|KPASS|UNRESOLVED|UNTESTED|UNSUPPORTED|ERROR|WARNING):",
    re.MULTILINE,
)
BAD_STATUSES = {
    "FAIL",
    "UNRESOLVED",
    "ERROR",
}


def utc_stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str, cwd: Path) -> str:
    return subprocess.check_output(["git", "-C", str(cwd), *args], text=True).strip()


def copy_if_present(source: Path, destination: Path) -> None:
    if source.exists() or source.is_symlink():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination, follow_symlinks=True)


def existing_sessions() -> set[Path]:
    """Session directories present before a test launches."""
    if not SESSION_ROOT.exists():
        return set()
    return {definition.parent for definition in SESSION_ROOT.glob("*/def.json")}


def fresh_session(before: set[Path]) -> Path | None:
    """The session this test created, identified by set difference.

    Picking "newest by mtime" instead looks correct until something else on the
    machine creates a session mid-test -- a second copy of this script, or a
    stray `mirage run`. The newest directory is then somebody else's, and the
    run both mis-attributes the health check and stops a session it does not
    own, which shows up as an unrelated test failing with an empty gdb.sum.
    Identify the session positively, and refuse to guess when ambiguous.
    """
    appeared = sorted(existing_sessions() - before)
    if not appeared:
        return None
    if len(appeared) > 1:
        print(
            "warning: several sessions appeared during one test "
            f"({', '.join(path.name for path in appeared)}); "
            "another mirage run is active and results are not trustworthy",
            file=sys.stderr,
        )
        return None
    return appeared[0]


def snapshot_session(session_dir: Path, output: Path) -> dict[str, object]:
    evidence = output / "session"
    evidence.mkdir(parents=True, exist_ok=True)
    paths = {
        "def.json": session_dir / "def.json",
        "health-before-stop.json": session_dir / "health.json",
        "rj_config.json": session_dir / "rj_config.json",
        "exec-def.json": session_dir / "exec/e-000000/def.json",
        "exec-status.json": session_dir / "exec/e-000000/status.json",
        "daemon.pid": session_dir / "node/0/pid",
        # The daemon hosts the emulated KFD, so anything the driver reports
        # about a failing test is here and nowhere else: its stderr does not
        # reach the test console. It has to be copied before the stop below,
        # which removes the whole session directory -- taking it afterwards
        # captures nothing at all, which is worse than missing the last few
        # lines of shutdown output.
        "host.log": session_dir / "node/0/host.log",
    }
    for destination, source in paths.items():
        copy_if_present(source, evidence / destination)

    session_definition: dict[str, object] = {}
    try:
        session_definition = json.loads((session_dir / "def.json").read_text())
    except (OSError, json.JSONDecodeError):
        pass
    session_id = str(session_definition.get("id", session_dir.name))
    daemon_mode = session_definition.get("daemon") is True

    daemon_pid: int | None = None
    daemon_alive = False
    try:
        daemon_pid = int((session_dir / "node/0/pid").read_text().strip())
        os.kill(daemon_pid, 0)
        daemon_alive = True
        process = subprocess.run(
            ["ps", "-p", str(daemon_pid), "-o", "pid,lstart,etime,stat,args"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        (evidence / "daemon-process.txt").write_text(process.stdout)
    except (OSError, ValueError):
        pass

    stop = subprocess.run(
        [str(MIRAGE), "session", "stop", session_id],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (evidence / "stop.log").write_text(stop.stdout)
    return {
        "id": session_id,
        "daemon_mode": daemon_mode,
        "daemon_pid": daemon_pid,
        "daemon_alive_before_stop": daemon_alive,
        "stop_rc": stop.returncode,
    }


def parse_summary(path: Path) -> tuple[collections.Counter[str], bool]:
    if not path.is_file():
        return collections.Counter(), False
    text = path.read_text(errors="replace")
    statuses = collections.Counter(STATUS_RE.findall(text))
    summary = text.rpartition("=== gdb Summary ===")[2]
    complete = bool(summary) and "gdb version" in summary
    return statuses, complete


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run each official gdb.rocm test in a fresh verified Mirage daemon session."
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--stop-after-failure", action="store_true")
    parser.add_argument("--root", type=Path, help="rocm-systems checkout under test")
    parser.add_argument("--rocgdb-suite", type=Path, help="ROCgdb source checkout")
    parser.add_argument("--gdb", type=Path, help="ROCgdb binary to drive")
    parser.add_argument("--venv", type=Path, help="venv holding the ROCm SDK wheels")
    parser.add_argument("--mirage", type=Path, help="mirage binary")
    parser.add_argument("--rocjitsu", type=Path, help="librocjitsu.so under test")
    parser.add_argument(
        "--tests",
        nargs="+",
        metavar="NAME.exp",
        help="run only these gdb.rocm files (default: all of them)",
    )
    parser.add_argument(
        "--rj-log",
        default="/dev/null",
        help="RJ_LOG_FILE for the traced process; %(default)s discards it. "
        "Pass a directory to keep one log per test.",
    )
    parser.add_argument(
        "--expect-tests",
        type=int,
        help="fail unless the suite holds exactly this many .exp files",
    )
    args = parser.parse_args()
    configure(args)

    required = [TEST_DIR, MIRAGE, ROCJITSU, SDK, CORE, GDB]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print("missing required paths:", *missing, sep="\n", file=sys.stderr)
        return 2

    available = sorted(path.name for path in TEST_DIR.glob("*.exp"))
    if args.expect_tests is not None and len(available) != args.expect_tests:
        print(
            f"expected {args.expect_tests} gdb.rocm files, found {len(available)}",
            file=sys.stderr,
        )
        return 2
    if args.tests:
        unknown = sorted(set(args.tests) - set(available))
        if unknown:
            print("no such gdb.rocm files:", *unknown, sep="\n", file=sys.stderr)
            return 2
        tests = sorted(args.tests)
    else:
        tests = available
    if not tests:
        print(f"no .exp files under {TEST_DIR}", file=sys.stderr)
        return 2
    total = len(tests)

    output = args.output or ROOT / "rocgdb-official-logs" / f"run-{utc_stamp()}"
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    manifest = {
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "workspace_commit": git("rev-parse", "HEAD", cwd=ROOT),
        "workspace_branch": git("branch", "--show-current", cwd=ROOT),
        "rocgdb_commit": git("rev-parse", "HEAD", cwd=SUITE_ROOT),
        "rocgdb_gdb": str(GDB),
        "mirage": str(MIRAGE),
        "mirage_sha256": sha256(MIRAGE),
        "rocjitsu": str(ROCJITSU),
        "rocjitsu_sha256": sha256(ROCJITSU),
        "sdk": str(SDK),
        "core": str(CORE),
        "profile": "mi350x",
        "daemon_required": True,
        "per_file_timeout_seconds": args.timeout,
        "tests": tests,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "test-manifest.txt").write_text("\n".join(tests) + "\n")

    aggregate: collections.Counter[str] = collections.Counter()
    records: list[dict[str, object]] = []
    failures = 0
    ld_path = f"{CORE / 'lib'}:{SDK / 'lib'}"
    tool_path = f"{SDK / 'lib/llvm/bin'}:{os.environ.get('PATH', '')}"
    runflags = (
        f"GDB={GDB} CC_FOR_TARGET=gcc CXX_FOR_TARGET=g++ "
        f"HIP_COMPILER_FOR_TARGET={VENV / 'bin/amdclang++'}"
    )

    for index, test in enumerate(tests, start=1):
        name = test.removesuffix(".exp")
        test_output = output / f"{index:02d}-{name}"
        test_output.mkdir()
        sessions_before = existing_sessions()
        start = time.monotonic()
        rj_log_root = Path(args.rj_log)
        rj_log = (
            rj_log_root / f"{name}.log" if rj_log_root.is_dir() else Path(args.rj_log)
        )
        command = [
            "setsid",
            "--wait",
            "timeout",
            "-k",
            "10",
            str(args.timeout),
            str(MIRAGE),
            "run",
            "--daemon",
            "--keep-session",
            "--profile",
            "mi350x",
            "--env",
            f"LD_LIBRARY_PATH={ld_path}",
            "--",
            "env",
            f"ROCM_PATH={SDK}",
            "HCC_AMDGPU_TARGET=gfx950",
            f"PATH={tool_path}",
            f"LD_LIBRARY_PATH={ld_path}",
            f"RJ_LOG_FILE={rj_log}",
            "make",
            "check",
            f"RUNTESTFLAGS={runflags}",
            f"TESTS=gdb.rocm/{test}",
        ]
        (test_output / "command.json").write_text(json.dumps(command, indent=2) + "\n")
        subprocess.run(
            ["make", "clean"],
            cwd=TESTSUITE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        for stale_output in (TESTSUITE / "gdb.log", TESTSUITE / "gdb.sum"):
            stale_output.unlink(missing_ok=True)
        with (test_output / "console.log").open("w") as console:
            process = subprocess.run(
                command,
                cwd=TESTSUITE,
                stdout=console,
                stderr=subprocess.STDOUT,
                check=False,
            )
        elapsed = round(time.monotonic() - start, 3)
        copy_if_present(TESTSUITE / "gdb.log", test_output / "gdb.log")
        copy_if_present(TESTSUITE / "gdb.sum", test_output / "gdb.sum")
        session_dir = fresh_session(sessions_before)
        session = (
            snapshot_session(session_dir, test_output)
            if session_dir is not None
            else {
                "id": None,
                "daemon_mode": False,
                "daemon_pid": None,
                "daemon_alive_before_stop": False,
                "stop_rc": None,
            }
        )
        statuses, summary_complete = parse_summary(test_output / "gdb.sum")
        aggregate.update(statuses)
        bad = {
            key: value
            for key, value in statuses.items()
            if key in BAD_STATUSES and value
        }
        # Count the assertions that actually ran. A file can legitimately report
        # nothing but UNSUPPORTED (an unmet `require`), so this is not a
        # per-file pass condition -- but a file that emitted no status at all
        # verified nothing, and a whole suite of them means the harness is
        # broken rather than the code being correct. Recorded per file and
        # enforced across the run below.
        executed = sum(statuses.values())
        passed = (
            process.returncode == 0
            and summary_complete
            and not bad
            and session["daemon_mode"] is True
            and session["daemon_alive_before_stop"] is True
        )
        record = {
            "index": index,
            "test": f"gdb.rocm/{test}",
            "rc": process.returncode,
            "elapsed_seconds": elapsed,
            "summary_complete": summary_complete,
            "executed_assertions": executed,
            "statuses": dict(sorted(statuses.items())),
            "bad_statuses": bad,
            "session": session,
            "passed": passed,
        }
        records.append(record)
        (test_output / "result.json").write_text(json.dumps(record, indent=2) + "\n")
        print(
            f"[{index:02d}/{total}] {test}: {'PASS' if passed else 'FAIL'} "
            f"rc={process.returncode} elapsed={elapsed:.1f}s statuses={dict(statuses)} "
            f"session={session['id']}",
            flush=True,
        )
        if not passed:
            failures += 1
            if args.stop_after_failure:
                break

    # A suite that ran to completion without executing a single assertion is a
    # harness failure wearing a success: every file would report rc=0, a
    # well-formed summary and no bad statuses. Refuse to call that a pass.
    total_executed = sum(aggregate.values())
    silent = [
        record["test"] for record in records if record["executed_assertions"] == 0
    ]
    if silent:
        print(
            f"warning: {len(silent)} file(s) produced no test status at all: "
            + ", ".join(silent),
            file=sys.stderr,
        )

    result = {
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "executed_assertions": total_executed,
        "files_with_no_status": silent,
        "planned_files": len(tests),
        "completed_files": len(records),
        "passed_files": sum(bool(record["passed"]) for record in records),
        "failed_files": failures,
        "aggregate_statuses": dict(sorted(aggregate.items())),
        "all_passed": (
            len(records) == len(tests) and failures == 0 and total_executed > 0
        ),
        "records": records,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    with (output / "summary.tsv").open("w") as summary:
        summary.write("index\ttest\tpassed\trc\telapsed_seconds\tPASS\tbad\tsession\n")
        for record in records:
            bad_count = sum(record["bad_statuses"].values())
            summary.write(
                f"{record['index']}\t{record['test']}\t{record['passed']}\t{record['rc']}\t"
                f"{record['elapsed_seconds']}\t{record['statuses'].get('PASS', 0)}\t{bad_count}\t"
                f"{record['session']['id']}\n"
            )
    print(
        f"output={output} completed={len(records)}/{total} passed={result['passed_files']} "
        f"failed={failures} aggregate={dict(aggregate)}",
        flush=True,
    )
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
