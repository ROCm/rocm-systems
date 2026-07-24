#!/usr/bin/env python3
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

ROOT = Path("/home/arosa/rocm-systems")
SUITE_ROOT = Path("/tmp/ROCgdb-tests")
TESTSUITE = SUITE_ROOT / "gdb/testsuite"
TEST_DIR = TESTSUITE / "gdb.rocm"
MIRAGE = ROOT / "emulation/mirage/target/debug/mirage"
ROCJITSU = ROOT / "emulation/rocjitsu/build/librocjitsu.so"
VENV = ROOT / "emulation/mirage/.venv-mi350"
SDK = VENV / "lib/python3.12/site-packages/_rocm_sdk_devel"
CORE = VENV / "lib/python3.12/site-packages/_rocm_sdk_core"
GDB = Path("/tmp/ROCgdb-build/gdb/gdb")
SESSION_ROOT = Path(f"/run/user/{os.getuid()}/mirage/session")
STATUS_RE = re.compile(
    r"^(PASS|FAIL|XFAIL|XPASS|KFAIL|KPASS|UNRESOLVED|UNTESTED|UNSUPPORTED|ERROR|WARNING):",
    re.MULTILINE,
)
BAD_STATUSES = {
    "FAIL",
    "UNRESOLVED",
    "ERROR",
    "WARNING",
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


def fresh_session(marker_ns: int) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    if not SESSION_ROOT.exists():
        return None
    for definition in SESSION_ROOT.glob("*/def.json"):
        try:
            stamp = definition.stat().st_mtime_ns
        except FileNotFoundError:
            continue
        if stamp >= marker_ns:
            candidates.append((stamp, definition.parent))
    return max(candidates, default=(0, None))[1]


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
    complete = "=== gdb Summary ===" in text and any(
        line.startswith("# of ") for line in text.splitlines()
    )
    return statuses, complete


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--stop-after-failure", action="store_true")
    args = parser.parse_args()

    required = [TEST_DIR, MIRAGE, ROCJITSU, SDK, CORE, GDB]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print("missing required paths:", *missing, sep="\n", file=sys.stderr)
        return 2

    tests = sorted(path.name for path in TEST_DIR.glob("*.exp"))
    if len(tests) != 89:
        print(f"expected 89 gdb.rocm files, found {len(tests)}", file=sys.stderr)
        return 2

    output = args.output or ROOT / "rocgdb-official-logs" / f"all-89-{utc_stamp()}"
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
    runflags = (
        f"GDB={GDB} CC_FOR_TARGET=gcc CXX_FOR_TARGET=g++ "
        f"HIP_COMPILER_FOR_TARGET={VENV / 'bin/amdclang++'}"
    )

    for index, test in enumerate(tests, start=1):
        name = test.removesuffix(".exp")
        test_output = output / f"{index:02d}-{name}"
        test_output.mkdir()
        marker_ns = time.time_ns()
        start = time.monotonic()
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
            f"LD_LIBRARY_PATH={ld_path}",
            "RJ_LOG_FILE=/dev/null",
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
        session_dir = fresh_session(marker_ns)
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
        passed = (
            process.returncode == 0
            and summary_complete
            and statuses["PASS"] > 0
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
            "statuses": dict(sorted(statuses.items())),
            "bad_statuses": bad,
            "session": session,
            "passed": passed,
        }
        records.append(record)
        (test_output / "result.json").write_text(json.dumps(record, indent=2) + "\n")
        print(
            f"[{index:02d}/89] {test}: {'PASS' if passed else 'FAIL'} "
            f"rc={process.returncode} elapsed={elapsed:.1f}s statuses={dict(statuses)} "
            f"session={session['id']}",
            flush=True,
        )
        if not passed:
            failures += 1
            if args.stop_after_failure:
                break

    result = {
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "planned_files": len(tests),
        "completed_files": len(records),
        "passed_files": sum(bool(record["passed"]) for record in records),
        "failed_files": failures,
        "aggregate_statuses": dict(sorted(aggregate.items())),
        "all_passed": len(records) == len(tests) and failures == 0,
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
        f"output={output} completed={len(records)}/89 passed={result['passed_files']} "
        f"failed={failures} aggregate={dict(aggregate)}",
        flush=True,
    )
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
