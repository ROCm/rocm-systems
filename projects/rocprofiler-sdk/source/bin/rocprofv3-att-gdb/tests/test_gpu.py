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
        request.config.getoption("--profiler")
        or str(Path(request.config.getoption("--rocm")) / "bin/rocprofv3"),
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
    stats = list(output.rglob("stats_*.csv"))
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


def test_same_source_line_captures_one_loop_round_trip(
    request, launcher, hip_app, tmp_path
):
    output = tmp_path / "output"
    location = "application.cpp:45"
    result = subprocess.run(
        command(request, launcher, output)
        + ["--batch", "--start", location, "--stop", location, "--", hip_app, "repeat"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=45,
    )
    (tmp_path / "debugger.log").write_text(result.stdout)
    assert result.returncode == 0, result.stdout
    assert result.stdout.count("[att] ACTIVE") == 1
    assert result.stdout.count("[att] DONE") == 1
    assert "RESULT=14" in result.stdout
    traces = list(output.rglob("*.att"))
    assert len(traces) == 2 and all(path.stat().st_size > 0 for path in traces)
    stats = list(output.rglob("stats_*.csv"))
    assert len(stats) == 1
    # From capture_begin in iteration 1 to capture_begin in iteration 2:
    # inside -> after -> next iteration's before kernel.
    for kernel in ("inside_kernel", "after_kernel", "before_kernel"):
        assert kernel + "(int*)" in stats[0].read_text()


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


def test_short_timeout_can_rearm_and_capture_next_iteration(
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
        console.expect("DONE capture=1 reason=timeout")
        # The user breakpoint may arrive before or after the timer reply.
        if b"hit Breakpoint 1" not in console.output:
            console.expect("hit Breakpoint 1")
        console.send("att arm --start capture_begin --stop capture_end")
        console.expect("ARMED capture=2")
        console.send("disable 1\ncontinue")
        console.expect("DONE capture=2")
        console.expect("exited normally")
    assert console.process.returncode == 0
    assert b"RESULT=14" in console.output
    # Capture 1 may stop before new waves launch. Capture 2 must contain useful data.
    traces = list(output.rglob("*_2.att"))
    assert len(traces) == 2 and all(path.stat().st_size > 0 for path in traces)
    stats = [path.read_text() for path in output.rglob("stats_*_dispatch_2.csv")]
    assert len(stats) == 1 and "inside_kernel(int*)" in stats[0]
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


@pytest.mark.parametrize(
    "mode",
    ["fork", "spawn", "fork-orphan", "spawn-orphan", "fork-timeout", "fork-markers"],
)
def test_nested_gpu_worker(request, launcher, gpu_process_tree, tmp_path, mode):
    import json
    import os

    root, trigger, worker = gpu_process_tree
    output = tmp_path / "output"
    ending = ["--timeout", "50ms"] if "timeout" in mode else ["--stop", "tree_end"]
    if "markers" in mode:
        ending += ["--marker-trace"]
    environment = dict(os.environ)
    if "markers" in mode:
        environment["ATT_TEST_MARKERS"] = "1"
    result = subprocess.run(
        command(request, launcher, output)
        + [
            "--batch",
            "--start",
            "tree_begin",
            *ending,
            "--",
            root,
            trigger,
            worker,
            mode,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=45,
        env=environment,
    )
    (tmp_path / "debugger.log").write_text(result.stdout)
    assert result.returncode == 0, result.stdout
    assert "TREE_RESULT=825600" in result.stdout
    traces = list(output.rglob("*.att"))
    assert len(traces) == 2 and all(path.stat().st_size for path in traces)
    stats = list(output.rglob("stats_*.csv"))
    assert len(stats) == 1 and "tree_kernel(int*)" in stats[0].read_text()
    manifest = json.loads(next(output.rglob("capture_1.json")).read_text())
    assert manifest["state"] == "DONE"
    participants = manifest["participants"]
    assert len(participants) == (2 if "orphan" in mode else 3)
    assert all(p["started"] and p["stopped"] for p in participants)
    # GPU output belongs to C, while B owns the breakpoint.
    gpu_pid = int(
        next(p for p in traces[0].parents if p.name.startswith("process_")).name.split(
            "_"
        )[1]
    )
    assert manifest["trigger_pid"] != gpu_pid


def test_attach_participant_to_existing_session(
    request, launcher, process_tree, gpu_process_tree, tmp_path
):
    import os
    import re
    import select
    import time

    profiler = request.config.getoption("--profiler") or str(
        Path(request.config.getoption("--rocm")) / "bin/rocprofv3"
    )
    commands_read, commands_write = os.pipe()
    replies_read, replies_write = os.pipe()
    gpu = subprocess.Popen(
        [gpu_process_tree[2], str(commands_read), str(replies_write)],
        pass_fds=(commands_read, replies_write),
        env=dict(os.environ, ROCP_TOOL_ATTACH="1"),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    os.close(commands_read)
    os.close(replies_write)
    attach = None
    attach_log = tmp_path / "attach.log"
    try:
        assert select.select([replies_read], [], [], 10)[0]
        assert os.read(replies_read, 1) == b"R"
        output = tmp_path / "output"
        with Console(
            command(request, launcher, output)
            + [
                "--command",
                "break tree_begin",
                "--command",
                "break tree_end",
                "--",
                process_tree[1],
                process_tree[2],
                "fork",
            ],
            tmp_path / "debugger.log",
        ) as console:
            console.expect("(gdb)")
            console.send("att session")
            console.expect("session file:")
            console.expect("session.json")
            session_file = re.search(rb"session file: ([^\r\n]+)", console.output)[
                1
            ].decode()
            with attach_log.open("wb") as log:
                attach = subprocess.Popen(
                    [
                        profiler,
                        "--att",
                        "--att-debugger-session",
                        session_file,
                        "--attach",
                        str(gpu.pid),
                        "--attach-children=false",
                        "--attach-sync-output",
                        "--att-shader-engine-mask",
                        "0x3",
                        "--att-buffer-size",
                        "16777216",
                        "-d",
                        str(output),
                    ],
                    stdin=subprocess.PIPE,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                deadline = time.monotonic() + 15
                while "Press Enter to detach" not in attach_log.read_text():
                    assert attach.poll() is None, attach_log.read_text()
                    assert time.monotonic() < deadline, attach_log.read_text()
                    time.sleep(0.02)
                console.expect(f"PID {gpu.pid},")
                console.send("att arm --start tree_begin --stop tree_end\nrun")
                console.expect("ACTIVE capture=1")
                os.write(commands_write, b"W")
                assert select.select([replies_read], [], [], 10)[0]
                assert os.read(replies_read, 1) == b"D"
                console.send("continue")
                console.expect("DONE capture=1")
                attach.communicate(b"\n", timeout=20)
                assert attach.returncode == 0, attach_log.read_text()
                attach = None
                console.send("continue")
            traces = list(output.rglob("*.att"))
            assert len(traces) == 2 and all(p.stat().st_size for p in traces)
            assert any(
                "tree_kernel(int*)" in p.read_text() for p in output.rglob("stats_*.csv")
            )
        os.write(commands_write, b"Q")
        data, _ = gpu.communicate(timeout=10)
        assert gpu.returncode == 0, data.decode()
    finally:
        if attach is not None:
            attach.kill()
            attach.wait()
        if gpu.poll() is None:
            gpu.kill()
            gpu.wait()
        os.close(commands_write)
        os.close(replies_read)


def test_process_exit_finishes_active_participant(request, launcher, hip_app, tmp_path):
    import json

    result = subprocess.run(
        command(request, launcher, "relative-output")
        + ["--batch", "--start", "capture_begin", "--timeout", "1s", "--", hip_app],
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=20,
    )
    (tmp_path / "debugger.log").write_text(result.stdout)
    assert result.returncode == 0, result.stdout
    assert "DONE capture=1 reason=exit" in result.stdout
    traces = list(tmp_path.rglob("*.att"))
    assert len(traces) == 2 and all(path.stat().st_size for path in traces)
    assert "inside_kernel(int*)" in next(tmp_path.rglob("stats_*.csv")).read_text()
    manifest = json.loads(next(tmp_path.rglob("capture_1.json")).read_text())
    for participant in manifest["participants"]:
        output = Path(participant["registration"]["output"])
        assert output.is_absolute() and output.is_dir()
        assert participant["reason"] == "exit"
