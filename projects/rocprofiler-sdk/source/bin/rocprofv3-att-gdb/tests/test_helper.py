# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
import json
import os
import socket
import subprocess
import tempfile
import time
import uuid

import pytest


class Participant:
    def __init__(self, listener):
        self.socket, _ = listener.accept()
        self.socket.settimeout(3)
        self.stream = self.socket.makefile("r")
        self.hello = self.read()
        assert self.hello["op"] == "REGISTER" and self.hello["version"] == 3

    def read(self):
        return json.loads(self.stream.readline())

    def send(self, op, capture=1, **fields):
        message = dict(
            op=op,
            capture=capture,
            session=self.hello["session"],
            instance=self.hello["instance"],
            **fields,
        )
        self.socket.sendall((json.dumps(message) + "\n").encode())

    def request(self, op, capture=1, **fields):
        self.send(op, capture, **fields)
        return self.read()

    def close(self):
        self.stream.close()
        self.socket.close()


@pytest.fixture
def session(helper, fake_roctx, tmp_path, request):
    with tempfile.TemporaryDirectory(prefix="att-test-") as directory:
        path = directory + "/socket"
        calls = tmp_path / "calls"
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(path)
            listener.listen()
            listener.settimeout(3)
            env = dict(
                os.environ,
                LD_PRELOAD=f"{helper}:{fake_roctx}",
                ROCPROFV3_GDB_SOCKET=path,
                ROCPROFV3_GDB_SESSION=uuid.uuid4().hex,
                ATT_TEST_CALLS=str(calls),
            )
            env.update(getattr(request, "param", {}))
            process = subprocess.Popen(["/bin/sleep", "30"], env=env)
            participant = None
            try:
                participant = Participant(listener)
                assert participant.hello["pid"] == process.pid
                yield participant, calls
            finally:
                if participant:
                    participant.close()
                process.terminate()
                process.wait(timeout=3)


def test_stop_is_idempotent_and_old_stop_cannot_stop_new_capture(session):
    peer, calls = session
    assert peer.request("START")["op"] == "STARTED"
    stopped = peer.request("STOP")
    assert stopped["op"] == "STOPPED"
    assert peer.request("STOP") == stopped
    assert peer.request("START", 2)["op"] == "STARTED"
    assert peer.request("STOP", 1)["op"] == "ERROR"
    assert peer.request("STOP", 2)["op"] == "STOPPED"
    assert calls.read_text().splitlines() == ["start", "stop", "start", "stop"]


def test_timer_waits_for_deadline_command_and_never_stops_early(session):
    peer, calls = session
    assert peer.request("START")["op"] == "STARTED"
    time.sleep(0.04)
    assert calls.read_text().splitlines() == ["start"]
    deadline = time.monotonic_ns() + 20_000_000
    timed = peer.request("DEADLINE", deadline=deadline)
    assert timed["op"] == "TIMED" and timed["deadline"] == deadline
    stopped = peer.read()
    assert stopped["op"] == "STOPPED" and stopped["reason"] == "timeout"
    assert stopped["timestamp"] >= deadline
    assert peer.request("STOP") == stopped


def test_stop_disarms_deadline_before_rearm(session):
    peer, calls = session
    assert peer.request("START")["op"] == "STARTED"
    assert (
        peer.request("DEADLINE", deadline=time.monotonic_ns() + 20_000_000)["op"]
        == "TIMED"
    )
    assert peer.request("STOP")["op"] == "STOPPED"
    assert peer.request("START", 2)["op"] == "STARTED"
    time.sleep(0.04)
    assert calls.read_text().splitlines() == ["start", "stop", "start"]
    assert peer.request("STOP", 2)["op"] == "STOPPED"


def test_duplicate_deadline_does_not_extend_capture(session):
    peer, _ = session
    assert peer.request("START")["op"] == "STARTED"
    deadline = time.monotonic_ns() + 200_000_000
    timed = peer.request("DEADLINE", deadline=deadline)
    assert peer.request("DEADLINE", deadline=deadline) == timed
    assert peer.request("DEADLINE", deadline=deadline + 1)["op"] == "ERROR"
    assert peer.request("STOP")["op"] == "STOPPED"
    assert peer.request("START", 2)["op"] == "STARTED"
    assert peer.request("DEADLINE", 1, deadline=deadline)["op"] == "ERROR"
    assert peer.request("STOP", 2)["op"] == "STOPPED"


def test_disconnect_stops_active_capture(session):
    peer, calls = session
    assert peer.request("START")["op"] == "STARTED"
    peer.socket.shutdown(socket.SHUT_RDWR)
    deadline = time.monotonic() + 3
    while calls.read_text().splitlines() != ["start", "stop"]:
        assert time.monotonic() < deadline
        time.sleep(0.01)


def test_duplicate_start_does_not_restart_trace(session):
    peer, calls = session
    assert peer.request("START")["op"] == "STARTED"
    assert peer.request("START")["op"] == "ERROR"
    assert peer.request("START", 2)["op"] == "ERROR"
    assert calls.read_text().splitlines() == ["start"]
    assert peer.request("STOP")["op"] == "STOPPED"


@pytest.mark.parametrize("session", [{"ATT_TEST_NOT_READY": "1"}], indirect=True)
def test_backend_rejection_is_reported(session):
    peer, calls = session
    reply = peer.request("START")
    assert reply["op"] == "ERROR" and "SDK not initialized" in reply["message"]
    assert not calls.exists()
    assert peer.request("STOP")["op"] == "STOPPED"


def test_late_deadline_delivery_stops_late(session):
    peer, _ = session
    assert peer.request("START")["op"] == "STARTED"
    deadline = time.monotonic_ns() - 10_000_000
    assert peer.request("DEADLINE", deadline=deadline)["op"] == "TIMED"
    assert peer.read()["timestamp"] >= deadline
