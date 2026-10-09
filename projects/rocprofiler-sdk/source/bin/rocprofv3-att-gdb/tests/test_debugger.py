# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Exercise debugger failure paths with real GDB and controllable ROCTx calls."""

from contextlib import ExitStack
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import pytest

from debugger_console import Console


@pytest.fixture(scope="session")
def debugger(request):
    rocgdb = Path(request.config.getoption("--rocm")) / "bin/rocgdb"
    command = (
        str(rocgdb) if rocgdb.is_file() else shutil.which("rocgdb") or shutil.which("gdb")
    )
    if not command:
        pytest.skip("debugger tests need ROCgdb or GDB with Python")
    return command


@pytest.fixture(scope="session")
def cpu_app(debugger, tmp_path_factory):
    binary = tmp_path_factory.mktemp("debugger-app") / "application"
    subprocess.run(
        [
            os.environ.get("CXX", "c++"),
            "-g",
            "-O0",
            str(Path(__file__).parent / "debugger_application.cpp"),
            "-o",
            str(binary),
        ],
        check=True,
    )
    return str(binary)


@pytest.fixture
def session(debugger, cpu_app, helper, fake_roctx, launcher, tmp_path):
    with ExitStack() as stack:

        def launch(
            start="capture_begin", stop="capture_end", timeout=0, mode="", **environment
        ):
            directory = Path(
                stack.enter_context(tempfile.TemporaryDirectory(prefix="att-gdb-test-"))
            )
            config = directory / "config.json"
            config.write_text(
                json.dumps(
                    dict(
                        socket=str(directory / "socket"),
                        batch=False,
                        start=start,
                        stop=stop,
                        timeout=timeout,
                        wrapper=["/usr/bin/env", f"LD_PRELOAD={helper}:{fake_roctx}"],
                    )
                )
            )
            extension = (
                Path(launcher).parent.parent
                / "share/rocprofiler-sdk/rocprofv3_att_gdb.py"
            )
            env = dict(os.environ, ATT_TEST_CALLS=str(tmp_path / "calls"), **environment)
            env.pop("ROCPROFV3_GDB_OWNER", None)
            console = stack.enter_context(
                Console(
                    [
                        debugger,
                        "-q",
                        "-nx",
                        "-ex",
                        "set confirm off",
                        "-ex",
                        "python import os; os.environ['ROCPROFV3_GDB_CONFIG'] = "
                        + repr(str(config)),
                        "-ex",
                        "source " + str(extension),
                        "--args",
                        cpu_app,
                        mode,
                    ],
                    tmp_path / "debugger.log",
                    env=env,
                )
            )
            console.expect("(gdb)")
            # Shorten failure detection while retaining the real asynchronous event loop.
            console.send("python capture.control_timeout = 0.5")
            return console

        yield launch


def test_timed_stop_has_completion_watchdog(session, tmp_path):
    console = session(
        stop=None, timeout=50000000, ATT_TEST_PAUSE_GATE=str(tmp_path / "pause")
    )
    console.send("break user_stop\nrun")
    console.expect("ACTIVE capture=1")
    console.expect("WAITING_stop")
    console.expect("timed capture did not stop within 0.5s after its deadline")
    console.send("att status")
    console.expect("state=ERROR")


def test_same_source_line_captures_one_loop_round_trip(session, tmp_path):
    location = "debugger_application.cpp:11"
    console = session(start=location, stop=location)
    console.send("run")
    console.expect("DONE capture=1 reason=breakpoint")
    console.expect("exited normally")
    # First iteration must run before stopping at the same line in iteration 2.
    assert console.output.index(b"PASSED_USER_STOP") < console.output.index(b"[att] DONE")
    assert console.output.count(b"[att] ACTIVE") == 1
    assert (tmp_path / "calls").read_text().splitlines() == ["start", "stop"]


def test_timed_stop_can_finish_during_completion_grace(session, tmp_path):
    gate = tmp_path / "pause"
    console = session(stop=None, timeout=50000000, ATT_TEST_PAUSE_GATE=str(gate))
    console.send("break user_stop\nrun")
    console.expect("ACTIVE capture=1")
    console.expect("WAITING_stop")
    gate.touch()
    console.expect("DONE capture=1")
    console.send("python print('USER_STOP=', gdb.selected_thread().is_stopped())")
    console.expect("USER_STOP= True")
    assert b"[att] ERROR" not in console.output


@pytest.mark.parametrize("stop_kind", ["breakpoint", "signal"])
def test_late_ack_preserves_new_user_stop(session, tmp_path, stop_kind):
    gate = tmp_path / "resume"
    console = session(mode=stop_kind, ATT_TEST_RESUME_GATE=str(gate))
    if stop_kind == "breakpoint":
        console.send("break user_stop")
        console.expect("Breakpoint 3")
    else:
        console.send("handle SIGUSR1 stop print nopass")
    console.send("run")
    console.expect("WAITING_start")
    console.send("continue")
    console.expect(
        "hit Breakpoint 3" if stop_kind == "breakpoint" else "received signal SIGUSR1"
    )
    gate.touch()
    console.expect("ACTIVE capture=1")
    console.send("python print('USER_STOP=', gdb.selected_thread().is_stopped())")
    console.expect("USER_STOP= True")
    assert b"PASSED_USER_STOP" not in console.output
    console.send("continue")
    console.expect("DONE capture=1")


def test_short_timeout_completes_and_can_rearm(session, tmp_path):
    console = session(stop=None, timeout=1)
    console.send("break between_captures\nrun")
    console.expect("DONE capture=1")
    console.send("att arm --start capture_begin --stop capture_end\ncontinue")
    console.expect("DONE capture=2")
    console.send("disable 2\ncontinue")
    console.expect("exited normally")
    assert (tmp_path / "calls").read_text().splitlines() == [
        "start",
        "stop",
        "start",
        "stop",
    ]


def test_timeout_waits_for_shared_user_breakpoint_continuation(session, tmp_path):
    console = session(stop=None, timeout=20000000)
    console.send("break capture_begin\nbreak user_stop\nrun")
    console.expect("ACTIVE capture=1")
    console.send("python import time; time.sleep(0.05)\natt status")
    console.expect("state=ACTIVE capture=1")
    assert (tmp_path / "calls").read_text().splitlines() == ["start"]
    console.send("continue")
    console.expect("DONE capture=1 reason=timeout")
    console.send("python print('USER_STOP=', gdb.selected_thread().is_stopped())")
    console.expect("USER_STOP= True")


def test_timeout_excludes_slow_resume_and_debugger_continuation(session, tmp_path):
    gate = tmp_path / "resume"
    console = session(stop=None, timeout=20000000, ATT_TEST_RESUME_GATE=str(gate))
    # Hold the continue command for longer than the capture interval, after the
    # native GDB resume but before returning to the extension. The timer must
    # stay disarmed throughout the command, not just until the continue event.
    script = tmp_path / "slow_continue.py"
    script.write_text(
        "import time\n"
        "original_execute = gdb.execute\n"
        "def slow_execute(command, *args, **kwargs):\n"
        "    result = original_execute(command, *args, **kwargs)\n"
        "    if command == 'continue &':\n"
        "        time.sleep(0.05)\n"
        f"        assert open({str(tmp_path / 'calls')!r}).read().splitlines() == ['start']\n"
        "        print('CONTINUATION_COMPLETED')\n"
        "    return result\n"
        "gdb.execute = slow_execute\n"
    )
    console.send(f"source {script}\nbreak user_stop\nrun")
    console.expect("WAITING_start")
    console.send("python import time; time.sleep(0.05)")
    console.send("python print('RESUME_HELD')")
    console.expect("RESUME_HELD\n")
    gate.touch()
    console.expect("CONTINUATION_COMPLETED\n")
    console.expect("DONE capture=1 reason=timeout")
    assert b"AssertionError" not in console.output
    assert b"[att] ERROR" not in console.output


@pytest.mark.parametrize("armed", [True, False])
def test_connection_retries_after_loader_pause(session, armed):
    console = session(start="capture_begin" if armed else None)
    console.send("python capture.connect_timeout = 0.1\nstarti")
    console.expect("stopped.")
    console.expect("control unavailable:")
    if not armed:
        console.send("att arm --start capture_begin --stop capture_end")
    console.send("continue")
    console.expect("control worker ready")
    console.expect("DONE capture=1")
    console.expect("exited normally")
    assert b"[att] ERROR" not in console.output


def test_completed_capture_watchdog_does_not_affect_rearm(session):
    console = session(timeout=50000000)
    console.send("break between_captures\nrun")
    console.expect("DONE capture=1")
    console.expect("hit Breakpoint 3")
    console.send("att arm --start capture_begin --stop capture_end")
    console.expect("ARMED capture=2")
    console.send("break user_stop\ncontinue")
    console.expect("ACTIVE capture=2")
    console.expect("hit Breakpoint 6")
    # Drain events after the first capture's deadline and completion grace.
    console.send("python import time; time.sleep(0.7)")
    console.send("att status")
    console.expect("state=ACTIVE capture=2")
    assert b"[att] ERROR" not in console.output
