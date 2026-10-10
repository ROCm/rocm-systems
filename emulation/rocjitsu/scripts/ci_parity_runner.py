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
    r"^(?:\d+:\s*)?\[\s+(OK|FAILED)\s+\]\s+(.+?)\s+\((\d+)\s+ms\)\s*$"
)
GTEST_TYPEPARAM = re.compile(r",\s*where\s+TypeParam\s+=.*$")
CTEST_SUMMARY = re.compile(
    r"^(?P<index>\d+)/(?P<total>\d+) Test #(?P<n>\d+): (?P<name>\S+)\s*\.*\s*"
    r"(?P<status>\*{3}Timeout|\*{3}Failed|\*{3}Not Run|Passed)\s+"
    r"(?P<seconds>[0-9.]+) sec\s*$"
)
ADD_TEST_RE = re.compile(
    r'^add_test\((?P<name>[A-Za-z0-9_\-]+)\s+"(?P<command>[^"]+)"\)\s*$'
)
LABELS_RE = re.compile(
    r'^set_tests_properties\("(?P<name>[^"]+)"\s+PROPERTIES\s+LABELS\s+"(?P<labels>[^"]+)"\)\s*$'
)
UNSUPPORTED_MARKERS = (
    "UnimplementedInst",
    "unsupported instruction",
    "rocjitsu: error",
)


def gtest_case_id(raw_id: str) -> str:
    """Drop gtest's ', where TypeParam = ...' suffix so the id matches the baseline."""
    return GTEST_TYPEPARAM.sub("", raw_id).strip()


def parse_gtest_output(text: str) -> list[dict]:
    """Return per-case verdicts from gtest OK/FAILED lines that include a time.

    CTest -V prints each line, and --output-on-failure prints a failed test's
    output again. Keep one row per id. A failure replaces an earlier pass.
    """
    cases = []
    index_by_id = {}
    for line in text.splitlines():
        match = GTEST_RESULT.match(line.strip())
        if not match:
            continue
        status_word, raw_id, millis = match.groups()
        case = {
            "id": gtest_case_id(raw_id),
            "status": "passed" if status_word == "OK" else "failed",
            "seconds": int(millis) / 1000.0,
        }
        previous = index_by_id.get(case["id"])
        if previous is None:
            index_by_id[case["id"]] = len(cases)
            cases.append(case)
            continue
        if cases[previous]["status"] != "failed":
            cases[previous] = case
    return cases


def parse_ctest_binaries(text: str) -> list[dict]:
    """Return one row per CTest summary line (Passed, ***Failed, ***Timeout)."""
    status_names = {
        "Passed": "passed",
        "***Failed": "failed",
        "***Timeout": "timeout",
        "***Not Run": "not_run",
    }
    binaries = []
    for line in text.splitlines():
        match = CTEST_SUMMARY.match(line.strip())
        if not match:
            continue
        binaries.append(
            {
                "name": match.group("name"),
                "status": status_names[match.group("status")],
                "seconds": float(match.group("seconds")),
            }
        )
    return binaries


def parse_installed_tests(text: str) -> list[dict]:
    """Read add_test() and LABELS lines from an install-tree CTestTestfile.cmake."""
    commands = {}
    labels = {}
    for raw in text.splitlines():
        line = raw.strip()
        added = ADD_TEST_RE.match(line)
        if added:
            commands[added.group("name")] = added.group("command")
            continue
        labeled = LABELS_RE.match(line)
        if labeled:
            labels[labeled.group("name")] = labeled.group("labels").split(";")
    return [
        {
            "name": name,
            "command": command,
            "labels": labels.get(name, []),
        }
        for name, command in commands.items()
    ]


def select_labeled_tests(tests: list[dict], label_regex: str) -> list[dict]:
    """Match labels the way `ctest -L` does: a search against each label."""
    pattern = re.compile(label_regex)
    return [
        test
        for test in tests
        if any(pattern.search(label) for label in test["labels"])
    ]


def apply_gtest_filter_env(env: dict, gtest_filter: str) -> dict:
    """Drop an empty GTEST_FILTER.

    The workflow sets GTEST_FILTER to "" when no filter was requested. gtest
    treats that empty value as a filter matching nothing, so every binary
    reports "Running 0 tests" and exits.
    """
    test_env = env.copy()
    if gtest_filter:
        test_env["GTEST_FILTER"] = gtest_filter
    else:
        test_env.pop("GTEST_FILTER", None)
    return test_env


def cmake_quote(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


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
    proc.stdout.close()
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
        env=apply_gtest_filter_env(env, gtest_filter),
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


def stream_raw_output(proc: subprocess.Popen) -> tuple[list[str], int]:
    """Print a process's merged output as it arrives, without an extra prefix."""
    lines = []
    pending = queue.Queue()

    def reader():
        for line in proc.stdout:
            pending.put(line)
        pending.put(None)

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    while True:
        line = pending.get()
        if line is None:
            break
        text = line.rstrip("\n")
        lines.append(text)
        print(text, flush=True)
    returncode = proc.wait()
    proc.stdout.close()
    return lines, returncode


def run_labeled_ctest(
    rocjitsu: str,
    config: str,
    rocm_root: str,
    label_regex: str,
    gtest_filter: str,
    timeout_seconds: int,
    env: dict,
    daemon: bool,
    out_dir: Path,
) -> dict:
    """Run `ctest -L <label> -V` with rocJITsu in front of each test command.

    hipRAND's quick and standard labels select the same five binaries the
    hardware baseline runs. Real ctest prints the Start / Test command /
    [ RUN ] / Passed lines, so the job log matches that baseline.
    """
    ctest_file = Path(rocm_root) / "bin" / "hipRAND" / "CTestTestfile.cmake"
    if not ctest_file.is_file():
        raise FileNotFoundError(f"missing install-tree ctest file: {ctest_file}")
    selected = select_labeled_tests(
        parse_installed_tests(ctest_file.read_text(encoding="utf-8")),
        label_regex,
    )
    if not selected:
        raise RuntimeError(f"no hipRAND tests matched ctest label {label_regex}")

    # ctest changes into --test-dir before it looks up the command. A relative
    # launcher path is then resolved from that directory and reported as
    # "Could not find executable".
    work = (out_dir / "ctest-quick").resolve()
    work.mkdir(parents=True, exist_ok=True)
    launcher_parts = [shlex.quote(str(Path(rocjitsu).resolve()))]
    if daemon:
        launcher_parts.append("--daemon")
    launcher_parts.extend(
        ["--config", shlex.quote(str(Path(config).resolve())), "--", '"$@"']
    )
    launcher = (work / "rocjitsu-launch").resolve()
    launcher.write_text(
        "#!/bin/bash\nset -euo pipefail\nexec " + " ".join(launcher_parts) + "\n",
        encoding="utf-8",
    )
    launcher.chmod(0o755)

    hiprand_dir = ctest_file.parent
    cmake_lines = [
        "# Generated so ctest -V runs each hipRAND binary under rocJITsu.",
    ]
    for test in selected:
        binary = str((hiprand_dir / test["command"]).resolve())
        args = [str(launcher), binary]
        if gtest_filter:
            args.append(f"--gtest_filter={gtest_filter}")
        quoted = " ".join(cmake_quote(arg) for arg in args)
        cmake_lines.append(f"add_test({test['name']} {quoted})")
        label_value = ";".join(test["labels"])
        cmake_lines.append(
            f'set_tests_properties("{test["name"]}" PROPERTIES '
            f'LABELS "{label_value}" '
            f"WORKING_DIRECTORY {cmake_quote(str(hiprand_dir.resolve()))})"
        )
    (work / "CTestTestfile.cmake").write_text(
        "\n".join(cmake_lines) + "\n", encoding="utf-8"
    )

    command = [
        "ctest",
        "--test-dir",
        str(work),
        "-L",
        label_regex,
        "-LE",
        "ex_gpu",
        "--output-on-failure",
        "--parallel",
        "1",
        "--timeout",
        str(timeout_seconds),
        "-V",
        "--test-output-size-passed",
        "0",
        "--test-output-size-failed",
        "0",
    ]
    print(f"Test project {work}", flush=True)
    print(flush=True)
    print(f"Running: {shlex.join(command)}", flush=True)
    proc = subprocess.Popen(
        command,
        env=apply_gtest_filter_env(env, gtest_filter),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    log_lines, returncode = stream_raw_output(proc)
    log_text = "\n".join(log_lines)
    cases = parse_gtest_output(log_text)
    binaries = parse_ctest_binaries(log_text)
    errors = []
    if returncode != 0 and not cases:
        errors.append(
            {
                "binary": "ctest",
                "returncode": returncode,
                "timed_out": False,
                "tail": "\n".join(log_lines[-40:]),
            }
        )
    return {
        "returncode": returncode,
        "cases": cases,
        "binaries": binaries,
        "errors": errors,
        "log_lines": [f"Running: {shlex.join(command)}", ""] + log_lines,
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
    parser.add_argument("--binary", action="append", default=[])
    parser.add_argument(
        "--ctest-label",
        default="",
        help="Run install-tree tests whose labels match this regex, via ctest -L -V.",
    )
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

    if not args.ctest_label and not args.binary:
        parser.error("pass --ctest-label or at least one --binary")

    env = build_env(args.rocm_root)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cases = []
    errors = []
    binaries = []
    if args.ctest_label:
        result = run_labeled_ctest(
            args.rocjitsu,
            args.config,
            args.rocm_root,
            args.ctest_label,
            args.gtest_filter,
            args.timeout_seconds,
            env,
            args.daemon,
            out.parent,
        )
        cases.extend(result["cases"])
        binaries.extend(result["binaries"])
        errors.extend(result["errors"])
        log_lines = result["log_lines"]
        ctest_status = result["returncode"]
    else:
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
        ctest_status = 0

    payload = {
        "component": "hiprand",
        "arch": "gfx942",
        "config": args.config,
        "rocm_root": args.rocm_root,
        "rocm_version": args.rocm_version,
        "gtest_filter": args.gtest_filter,
        "ctest_label": args.ctest_label,
        "runner_label": args.runner_label,
        "cases": cases,
        "binaries": binaries,
        "errors": errors,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    log_path = out.with_name("ctest.log")
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(f"wrote {out} cases={len(cases)} errors={len(errors)}")
    print(f"wrote {log_path}")
    if errors or not cases or ctest_status != 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
