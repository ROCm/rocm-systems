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
import re
import subprocess
import sys
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


def run_binary(
    rocjitsu: str,
    config: str,
    binary: str,
    gtest_filter: str,
    timeout_seconds: int,
    env: dict,
) -> dict:
    command = [rocjitsu, "--daemon", "--config", config, "--", binary]
    if gtest_filter:
        command.append(f"--gtest_filter={gtest_filter}")
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
            check=False,
        )
        output = completed.stdout or ""
        returncode = completed.returncode
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        if isinstance(exc.stdout, bytes):
            output = exc.stdout.decode("utf-8", errors="replace")
        returncode = 124
        timed_out = True
    elapsed = time.monotonic() - started
    cases = parse_gtest_output(output)
    for case in cases:
        case["binary"] = Path(binary).name
    error = None
    if timed_out or (returncode != 0 and not cases) or (
        looks_unsupported(output) and not cases
    ):
        error = {
            "returncode": returncode,
            "timed_out": timed_out,
            "tail": "\n".join(output.splitlines()[-40:]),
        }
        cases.append(
            {
                "id": gtest_filter or Path(binary).name,
                "status": "unsupported",
                "seconds": round(elapsed, 3),
                "binary": Path(binary).name,
            }
        )
    return {"returncode": returncode, "cases": cases, "error": error}


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
    parser.add_argument("--rocm-version", default="")
    parser.add_argument("--runner-label", default="")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    env = build_env(args.rocm_root)
    cases = []
    errors = []
    for binary in args.binary:
        result = run_binary(
            args.rocjitsu,
            args.config,
            binary,
            args.gtest_filter,
            args.timeout_seconds,
            env,
        )
        cases.extend(result["cases"])
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
    print(f"wrote {out} cases={len(cases)} errors={len(errors)}")
    if errors or not cases:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
