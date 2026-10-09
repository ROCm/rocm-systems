# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
from pathlib import Path
import subprocess

import pytest

from debugger_console import Console


def command(request, launcher, output):
    return [
        launcher,
        "--rocprofv3",
        str(Path(request.config.getoption("--rocm")) / "bin/rocprofv3"),
        "--att-shader-engine-mask",
        "0x3",
        "--att-buffer-size",
        "16777216",
        "-d",
        str(output),
    ]


def check_trace(output, captures=1):
    traces = list(output.rglob("*.att"))
    assert len(traces) == 2 * captures
    assert all(path.stat().st_size > 0 for path in traces)
    stats = list(output.glob("stats_*.csv"))
    assert len(stats) == captures
    for path in stats:
        text = path.read_text()
        assert "inside_kernel(int*)" in text
        assert "before_kernel(int*)" not in text
        assert "after_kernel(int*)" not in text
        assert "warmup_kernel(int*)" not in text


@pytest.mark.parametrize("ending", ["breakpoint", "timeout", "both", "concurrent"])
def test_capture(request, launcher, hip_app, tmp_path, ending):
    output = tmp_path / "output with spaces"
    args = command(request, launcher, output) + ["--batch", "--start", "capture_begin"]
    if ending in ("breakpoint", "both", "concurrent"):
        args += ["--stop", "capture_end"]
    if ending in ("timeout", "both"):
        args += ["--timeout", "50ms"]
    marker = tmp_path / "must-not-be-created"
    literal = f"literal $(touch {marker}) `touch {marker}`"
    mode = ending if ending in ("timeout", "concurrent") else "normal"
    args += ["--", hip_app, mode, literal]
    result = subprocess.run(
        args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=45
    )
    (tmp_path / "debugger.log").write_text(result.stdout)
    assert result.returncode == 0, result.stdout
    assert result.stdout.count("[att] ACTIVE") == 1
    assert result.stdout.count("[att] DONE") == 1
    assert (
        "reason=" + ("timeout" if ending == "timeout" else "breakpoint") in result.stdout
    )
    assert "RESULT=7" in result.stdout
    assert "ARGUMENT=" + literal in result.stdout
    assert not marker.exists()
    check_trace(output)


def test_missing_breakpoint_is_incomplete(request, launcher, hip_app, tmp_path):
    result = subprocess.run(
        command(request, launcher, tmp_path / "output")
        + [
            "--batch",
            "--start",
            "not_a_function",
            "--stop",
            "capture_end",
            "--",
            hip_app,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=45,
    )
    assert result.returncode != 0
    assert "incomplete capture" in result.stdout
    assert not list(tmp_path.rglob("*.att"))


@pytest.mark.parametrize("skip", [100, 101])
def test_skip_warmup_hits(request, launcher, hip_app, tmp_path, skip):
    output = tmp_path / "output"
    result = subprocess.run(
        command(request, launcher, output)
        + [
            "--batch",
            "--start",
            "capture_begin",
            "--skip",
            str(skip),
            "--stop",
            "capture_end",
            "--",
            hip_app,
            "warmup",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=45,
    )
    (tmp_path / "debugger.log").write_text(result.stdout)
    assert "RESULT=707" in result.stdout
    if skip == 100:
        assert result.returncode == 0, result.stdout
        assert "start hit=101;" in result.stdout
        assert result.stdout.count("[att] ACTIVE") == 1
        check_trace(output)
    else:
        assert result.returncode != 0
        assert "incomplete capture" in result.stdout
        assert not list(output.rglob("*.att"))


def test_hip_kernel_host_stub_trigger(request, launcher, hip_app_noinline, tmp_path):
    output = tmp_path / "output"
    result = subprocess.run(
        command(request, launcher, output)
        + [
            "--batch",
            "--start",
            "__device_stub__inside_kernel(int*)",
            "--stop",
            "capture_end",
            "--",
            hip_app_noinline,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=45,
    )
    (tmp_path / "debugger.log").write_text(result.stdout)
    assert result.returncode == 0, result.stdout
    assert "RESULT=7" in result.stdout
    check_trace(output)


def test_interactive_skip(request, launcher, hip_app, tmp_path):
    output = tmp_path / "output"
    console = Console(
        command(request, launcher, output) + ["--", hip_app, "warmup"],
        tmp_path / "debugger.log",
    )
    try:
        console.expect("(gdb)")
        console.send("att arm --start capture_begin --skip 100 --stop capture_end")
        console.expect("start on hit 101")
        console.send("att status")
        console.expect("skip_remaining=100")
        console.send("run")
        console.expect("start hit=101;")
        console.expect("DONE capture=1")
        console.expect("exited normally")
    finally:
        console.close()
    assert console.process.returncode == 0
    check_trace(output)


def test_short_timeout_recovery_captures_next_iteration(
    request, launcher, hip_app, tmp_path
):
    output = tmp_path / "output"
    with Console(
        command(request, launcher, output)
        + ["--command", "break between_captures", "--", hip_app, "repeat"],
        tmp_path / "debugger.log",
    ) as console:
        console.expect("(gdb)")
        console.send("att arm --start capture_begin --timeout 0.001us\nrun")
        console.expect("timeout expired before application continuation")
        console.send("att cancel")
        console.expect("CANCELLED; failed capture is stopped")
        console.send("continue")
        console.expect("hit Breakpoint 1")
        console.send("att arm --start capture_begin --stop capture_end")
        console.expect("ARMED capture=2")
        console.send("disable 1\ncontinue")
        console.expect("DONE capture=2")
        console.expect("exited normally")
    assert console.process.returncode == 0
    assert b"RESULT=14" in console.output
    # Capture 1 expired before any new waves. Capture 2 must contain useful data.
    traces = list(output.rglob("*_2.att"))
    assert len(traces) == 2 and all(path.stat().st_size > 0 for path in traces)
    stats = [path.read_text() for path in output.glob("stats_*.csv")]
    assert sum("inside_kernel(int*)" in text for text in stats) == 1
    assert all(
        "before_kernel(int*)" not in text and "after_kernel(int*)" not in text
        for text in stats
    )


def test_interactive_cancel_arm_and_rearm(request, launcher, hip_app, tmp_path):
    output = tmp_path / "output"
    console = Console(
        command(request, launcher, output)
        + ["--command", "break between_captures", "--", hip_app, "repeat"],
        tmp_path / "debugger.log",
    )
    try:
        console.expect("(gdb)")
        console.send("att arm --start capture_begin --stop capture_end")
        console.expect("ARMED capture=1")
        console.send("att cancel")
        console.expect("CANCELLED; no capture started")
        console.send("att arm --start capture_begin --stop capture_end\nrun")
        console.expect("DONE capture=2")
        console.expect("hit Breakpoint 1")
        console.send("att status")
        console.expect("state=DONE")
        console.send("att arm --start capture_begin --stop capture_end")
        console.expect("ARMED capture=3")
        console.send("disable 1\ncontinue")
        console.expect("DONE capture=3")
        console.expect("exited normally")
    finally:
        console.close()
    assert console.process.returncode == 0
    assert b"RESULT=14" in console.output
    check_trace(output, captures=2)


def test_user_breakpoints_stay_stopped_and_active_cancel(
    request, launcher, hip_app, tmp_path
):
    output = tmp_path / "output"
    console = Console(
        command(request, launcher, output)
        + [
            "--command",
            "break capture_begin",
            "--command",
            "break capture_end",
            "--",
            hip_app,
        ],
        tmp_path / "debugger.log",
    )
    try:
        console.expect("(gdb)")
        console.send("att arm --start capture_begin --timeout 2s\nrun")
        console.expect("ACTIVE capture=1")
        console.send("python print('USER_STOP=', gdb.selected_thread().is_stopped())")
        console.expect("USER_STOP= True")
        console.send("continue")
        console.expect("hit Breakpoint 2")
        console.send("att cancel")
        console.expect("DONE capture=1 reason=cancel")
        console.send("python print('USER_STOP=', gdb.selected_thread().is_stopped())")
        console.expect("USER_STOP= True")
        console.send("continue")
        console.expect("exited normally")
    finally:
        console.close()
    assert console.process.returncode == 0
    check_trace(output)
