#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run a suite under rocJITsu or on a native GPU and print a CTest -V log."""

import argparse
import os
import queue
import signal
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path


def ctest_result_line(
    index: int, total: int, name: str, passed: bool, seconds: float
) -> str:
    status = "Passed" if passed else "***Failed"
    left = f"{index}/{total} Test #{index}: {name} "
    dots = "." * max(3, 54 - len(left))
    return f"{left}{dots}   {status}    {seconds:.2f} sec"


def test_env(base: dict, gtest_filter: str) -> dict:
    """Drop an empty GTEST_FILTER. gtest treats that as a filter matching nothing."""
    env = base.copy()
    if gtest_filter:
        env["GTEST_FILTER"] = gtest_filter
    else:
        env.pop("GTEST_FILTER", None)
    return env


def kill_test_processes(proc: subprocess.Popen):
    if os.name == "posix":
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    else:
        proc.kill()


def stream_output(proc: subprocess.Popen, index: int, timeout_seconds: int):
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
            # The launcher can leave CTest and test children holding stdout open.
            # Kill the entire session so timeout also bounds draining the pipe.
            kill_test_processes(proc)
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
    try:
        returncode = proc.wait(timeout=max(0.01, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        kill_test_processes(proc)
        timed_out = True
        returncode = proc.wait()
    thread.join(timeout=5)
    while not pending.empty():
        line = pending.get_nowait()
        if line is not None:
            text = line.rstrip("\n")
            lines.append(text)
            print(f"{index}: {text}", flush=True)
    if timed_out:
        message = f"Timed out after {timeout_seconds} seconds"
        lines.append(message)
        print(f"{index}: {message}", flush=True)
    proc.stdout.close()
    return lines, returncode, timed_out


def run_under_rocjitsu(
    rocjitsu: str,
    config: str,
    command: list[str],
    name: str,
    timeout_seconds: int,
    env: dict,
    cwd: str,
    index: int = 1,
    total: int = 1,
    gtest_filter: str = "",
) -> dict:
    launch = [rocjitsu, "--config", config, "--", *command]
    return run_command(
        launch, name, timeout_seconds, env, cwd, index, total, gtest_filter
    )


def native_test_env(base: dict) -> dict:
    """Preload the artifact's ASAN runtime in the child, never in this driver."""
    runtime = base.get("ASAN_RUNTIME_PATH", "")
    if not runtime or not Path(runtime).is_file():
        raise FileNotFoundError(
            f"ASAN_RUNTIME_PATH must name an existing runtime file: {runtime!r}"
        )
    env = base.copy()
    preload = env.get("LD_PRELOAD", "")
    env["LD_PRELOAD"] = runtime + (os.pathsep + preload if preload else "")
    return env


def run_command(
    launch: list[str],
    name: str,
    timeout_seconds: int,
    env: dict,
    cwd: str,
    index: int = 1,
    total: int = 1,
    gtest_filter: str = "",
) -> dict:
    header = [
        f"    Start {index}: {name}",
        "",
        f"{index}: Test command: {shlex.join(launch)}",
        f"{index}: Working Directory: {cwd}",
        f"{index}: Test timeout computed to be: {timeout_seconds}",
    ]
    for line in header:
        print(line, flush=True)
    started = time.monotonic()
    proc = subprocess.Popen(
        launch,
        cwd=cwd,
        env=test_env(env, gtest_filter),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        start_new_session=os.name == "posix",
    )
    output_lines, returncode, timed_out = stream_output(proc, index, timeout_seconds)
    elapsed = time.monotonic() - started
    passed = returncode == 0 and not timed_out
    summary = ctest_result_line(index, total, name, passed, elapsed)
    print("", flush=True)
    print(summary, flush=True)
    print("", flush=True)
    log_lines = header + [f"{index}: {line}" for line in output_lines] + ["", summary, ""]
    return {
        "returncode": returncode,
        "timed_out": timed_out,
        "passed": passed,
        "log_lines": log_lines,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--rocjitsu")
    mode.add_argument("--native", action="store_true")
    parser.add_argument("--config")
    parser.add_argument("--name", required=True)
    parser.add_argument("--cwd", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--gtest-filter", default="")
    parser.add_argument("--log", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    if args.rocjitsu and not args.config:
        parser.error("--config is required with --rocjitsu")
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("pass the test command after --")

    env = os.environ.copy()
    root = env.get("ROCM_ROOT", "")
    if root:
        lib = str(Path(root) / "lib")
        bin_dir = str(Path(root) / "bin")
        env["ROCM_PATH"] = root
        env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
        previous = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = lib if not previous else lib + os.pathsep + previous

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if args.native:
            result = run_command(
                command, args.name, args.timeout_seconds, native_test_env(env),
                args.cwd, gtest_filter=args.gtest_filter,
            )
        else:
            result = run_under_rocjitsu(
                args.rocjitsu,
                args.config,
                command,
                args.name,
                args.timeout_seconds,
                env,
                args.cwd,
                gtest_filter=args.gtest_filter,
            )
    except OSError as error:
        # A missing executable or working directory is a failed suite, with a
        # durable diagnostic just like a nonzero test process exit.
        result = {"passed": False, "log_lines": [f"{args.name}: {error}"]}
    log_path.write_text("\n".join(result["log_lines"]) + "\n", encoding="utf-8")
    print(f"wrote {log_path}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
