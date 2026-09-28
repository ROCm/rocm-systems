#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run gtest binaries under rocJITsu and write a parity JSON.

The hipRAND parity workflow in rocm-libraries vendors a copy of this file so a
labeled pull request can run before this script is on develop. Keep the two
copies in sync.
"""

import argparse
import json
import os
import queue
import re
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

GTEST_RESULT = re.compile(
    r"^\[\s+(OK|FAILED)\s+\]\s+(\S+)\s+\((\d+)\s+ms\)\s*$"
)
UNSUPPORTED_MARKERS = (
    "UnimplementedInst",
    "unsupported instruction",
    "rocjitsu: error",
)


def parse_gtest_output(text: str) -> list[dict]:
    """Return per-case verdicts from gtest OK/FAILED lines that include a time."""
    cases = []
    for line in text.splitlines():
        match = GTEST_RESULT.match(line.strip())
        if not match:
            continue
        status_word, case_id, millis = match.groups()
        cases.append(
            {
                "id": case_id,
                "status": "passed" if status_word == "OK" else "failed",
                "seconds": int(millis) / 1000.0,
            }
        )
    return cases


def looks_unsupported(text: str) -> bool:
    lowered = text.lower()
    return any(marker.lower() in lowered for marker in UNSUPPORTED_MARKERS)


def ctest_result_line(
    index: int, total: int, name: str, passed: bool, seconds: float
) -> str:
    """One CTest -V result line, matching the hardware baseline log."""
    status = "Passed" if passed else "***Failed"
    left = f"{index}/{total} Test #{index}: {name} "
    dots = "." * max(3, 54 - len(left))
    return f"{left}{dots}   {status}    {seconds:.2f} sec"


def stream_merged_output(proc: subprocess.Popen, index: int, timeout_seconds: int):
    """Print each output line as `{index}: ...`, the same prefix CTest -V uses."""
    lines = []
    timed_out = False
    pending = queue.Queue()

    def reader():
        for line in proc.stdout:
            pending.put(line)
        pending.put(None)

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    deadline = time.monotonic() + timeout_seconds
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            proc.kill()
            timed_out = True
            break
        try:
            line = pending.get(timeout=min(1.0, remaining))
        except queue.Empty:
            if proc.poll() is not None and not thread.is_alive():
                break
            continue
        if line is None:
            break
        text = line.rstrip("\n")
        lines.append(text)
        print(f"{index}: {text}", flush=True)
    returncode = proc.wait()
    return lines, returncode, timed_out


def run_binary(
    rocjitsu: str,
    config: str,
    binary: str,
    gtest_filter: str,
    timeout_seconds: int,
    env: dict,
    daemon: bool,
    index: int = 1,
    total: int = 1,
) -> dict:
    command = [rocjitsu]
    if daemon:
        command.append("--daemon")
    command.extend(["--config", config, "--", binary])
    if gtest_filter:
        command.append(f"--gtest_filter={gtest_filter}")
    name = Path(binary).name
    header = [
        f"    Start {index}: {name}",
        "",
        f"{index}: Test command: {shlex.join(command)}",
        f"{index}: Test timeout computed to be: {timeout_seconds}",
    ]
    for line in header:
        print(line, flush=True)
    started = time.monotonic()
    proc = subprocess.Popen(
        command,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    output_lines, returncode, timed_out = stream_merged_output(
        proc, index, timeout_seconds
    )
    elapsed = time.monotonic() - started
    output = "\n".join(output_lines)
    cases = parse_gtest_output(output)
    for case in cases:
        case["binary"] = name
    passed = returncode == 0 and not timed_out and bool(cases)
    summary = ctest_result_line(index, total, name, passed, elapsed)
    print("", flush=True)
    print(summary, flush=True)
    print("", flush=True)
    error = None
    if timed_out or (returncode != 0 and not cases) or (
        looks_unsupported(output) and not cases
    ):
        error = {
            "returncode": returncode,
            "timed_out": timed_out,
            "tail": "\n".join(output_lines[-40:]),
        }
        cases.append(
            {
                "id": gtest_filter or name,
                "status": "unsupported",
                "seconds": round(elapsed, 3),
                "binary": name,
            }
        )
    log_lines = header + [f"{index}: {line}" for line in output_lines] + ["", summary, ""]
    return {
        "returncode": returncode,
        "cases": cases,
        "error": error,
        "log_lines": log_lines,
    }


def build_env(rocm_root: str) -> dict:
    env = os.environ.copy()
    lib = str(Path(rocm_root) / "lib")
    bin_dir = str(Path(rocm_root) / "bin")
    env["ROCM_PATH"] = rocm_root
    env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
    previous = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = lib if not previous else lib + os.pathsep + previous
    return env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rocjitsu", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--rocm-root", required=True)
    parser.add_argument("--binary", action="append", required=True)
    parser.add_argument("--gtest-filter", default="")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Fork the rocJITsu daemon. Required for kmd configs when local mode cannot see the simulated GPU.",
    )
    parser.add_argument("--rocm-version", default="")
    parser.add_argument("--runner-label", default="")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    env = build_env(args.rocm_root)
    cases = []
    errors = []
    log_lines = [f"Test project {args.rocm_root}/bin/hipRAND", ""]
    print(log_lines[0], flush=True)
    print(flush=True)
    total = len(args.binary)
    for index, binary in enumerate(args.binary, start=1):
        result = run_binary(
            args.rocjitsu,
            args.config,
            binary,
            args.gtest_filter,
            args.timeout_seconds,
            env,
            args.daemon,
            index=index,
            total=total,
        )
        cases.extend(result["cases"])
        log_lines.extend(result["log_lines"])
        if result["error"]:
            errors.append({"binary": binary, **result["error"]})

    payload = {
        "component": "hiprand",
        "arch": "gfx942",
        "config": args.config,
        "rocm_root": args.rocm_root,
        "rocm_version": args.rocm_version,
        "gtest_filter": args.gtest_filter,
        "runner_label": args.runner_label,
        "cases": cases,
        "errors": errors,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    log_path = out.with_name("ctest.log")
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(f"wrote {out} cases={len(cases)} errors={len(errors)}")
    print(f"wrote {log_path}")
    if errors or not cases:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
