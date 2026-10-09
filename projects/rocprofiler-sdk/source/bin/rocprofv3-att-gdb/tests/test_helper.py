# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
import os
import socket
import subprocess
import time

import pytest


@pytest.fixture
def session(helper, fake_roctx, tmp_path, request):
    # UNIX socket paths are limited to 108 bytes, including the final NUL.
    import tempfile

    with tempfile.TemporaryDirectory(prefix="att-test-") as directory:
        path = directory + "/socket"
        calls = tmp_path / "calls"
        env = dict(
            os.environ,
            LD_PRELOAD=f"{helper}:{fake_roctx}",
            ROCPROFV3_GDB_SOCKET=path,
            ATT_TEST_CALLS=str(calls),
        )
        env.pop("ROCPROFV3_GDB_OWNER", None)
        env.update(getattr(request, "param", {}))
        process = subprocess.Popen(["/bin/sleep", "30"], env=env)
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.settimeout(3)
        try:
            deadline = time.monotonic() + 3
            while True:
                try:
                    connection.connect(path)
                    break
                except (FileNotFoundError, ConnectionRefusedError):
                    assert time.monotonic() < deadline
                    time.sleep(0.01)
            stream = connection.makefile("r")
            assert stream.readline().startswith(f"HELLO 2 {process.pid} ")
            yield connection, stream, calls
            stream.close()
        finally:
            connection.close()
            process.terminate()
            process.wait(timeout=3)


def request(session, command):
    connection, stream, _ = session
    connection.sendall((command + "\n").encode())
    return stream.readline().strip()


def test_breakpoint_stop_is_idempotent_and_rearm_rejects_stale_stop(session):
    assert request(session, "START 1 0").startswith("STARTED 1 ")
    stopped = request(session, "STOP 1")
    assert stopped.startswith("STOPPED 1 request ")
    assert request(session, "STOP 1") == stopped
    assert request(session, "START 2 0").startswith("STARTED 2 ")
    assert request(session, "STOP 1").startswith("ERROR 1 ")
    assert request(session, "STOP 2").startswith("STOPPED 2 request ")
    assert session[2].read_text().splitlines() == ["start", "stop", "start", "stop"]


def test_timeout_excludes_wait_for_continuation_and_stops_without_a_stop_request(session):
    assert request(session, "START 1 20000000").startswith("STARTED 1 ")
    time.sleep(0.04)
    assert session[2].read_text().splitlines() == ["start"]
    before = time.monotonic_ns()
    timed = request(session, "CONTINUED 1")
    assert timed.startswith("TIMED 1 ")
    deadline = int(timed.split()[2])
    assert deadline >= before + 20000000
    stopped = session[1].readline().strip()
    assert stopped.startswith("STOPPED 1 timeout ")
    assert int(stopped.split()[3]) >= deadline
    assert request(session, "STOP 1") == stopped
    assert session[2].read_text().splitlines() == ["start", "stop"]


def test_stop_disarms_deadline_before_next_capture(session):
    assert request(session, "START 1 20000000").startswith("STARTED 1 ")
    assert request(session, "CONTINUED 1").startswith("TIMED 1 ")
    assert request(session, "STOP 1").startswith("STOPPED 1 ")
    assert request(session, "START 2 0").startswith("STARTED 2 ")
    time.sleep(0.04)
    assert session[2].read_text().splitlines() == ["start", "stop", "start"]
    assert request(session, "STOP 2").startswith("STOPPED 2 request ")


def test_duplicate_continuation_does_not_extend_deadline_and_stale_one_is_rejected(
    session,
):
    assert request(session, "START 1 200000000").startswith("STARTED 1 ")
    timed = request(session, "CONTINUED 1")
    assert timed.startswith("TIMED 1 ")
    assert request(session, "CONTINUED 1") == timed
    assert request(session, "STOP 1").startswith("STOPPED 1 ")
    assert request(session, "START 2 200000000").startswith("STARTED 2 ")
    assert request(session, "CONTINUED 1").startswith("ERROR 1 ")
    assert request(session, "STOP 2").startswith("STOPPED 2 ")
    assert request(session, "CONTINUED 2").startswith("ERROR 2 ")


def test_disconnect_stops_active_capture(session):
    assert request(session, "START 1 0").startswith("STARTED 1 ")
    session[0].shutdown(socket.SHUT_RDWR)
    deadline = time.monotonic() + 3
    while session[2].read_text().splitlines() != ["start", "stop"]:
        assert time.monotonic() < deadline
        time.sleep(0.01)


def test_duplicate_start_does_not_restart_trace(session):
    assert request(session, "START 1 0").startswith("STARTED 1 ")
    assert request(session, "START 1 0").startswith("ERROR 1 ")
    assert request(session, "START 2 0").startswith("ERROR 2 ")
    assert session[2].read_text().splitlines() == ["start"]
    assert request(session, "STOP 1").startswith("STOPPED 1 ")


@pytest.mark.parametrize("session", [{"ATT_TEST_NOT_READY": "1"}], indirect=True)
def test_start_before_sdk_initialization_is_rejected(session):
    assert "SDK not initialized" in request(session, "START 1 0")
    assert not session[2].exists()
