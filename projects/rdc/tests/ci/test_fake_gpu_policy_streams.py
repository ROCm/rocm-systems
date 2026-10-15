"""rdcd's policy streams (RegisterPolicy) on fake GPUs: a stream ends when its
client goes away or rdcd stops, every stream of a group gets every notification,
rdcd admits at most 64 streams, and a missing group is an error.

Runs like test_fake_gpu.py (same environment variables and helpers). Also runs
rdc_policy_client_tests, the gtest of the client's registration against a fake
rdcd, from the tested prefix.
"""

import ctypes
import os
import re
import signal
import subprocess
import threading
import time

import pytest

# The helpers and the `tools` fixture come from test_fake_gpu.py, not conftest.py:
# tests/ci/conftest.py is not installed with the suite (share/rdc/rdctst_tests/fake_gpu).
import test_fake_gpu
from test_fake_gpu import FACTOR, PREFIX, REQUIRE, Rdcd, _created_id, tools  # noqa: F401 (fixture)

NOTIFICATION = "A Temperature Limit exceeds the threshold 1 with the value"
RDC_ST_OK = 0
RDC_ST_CONFLICT = 7
RDC_ST_CLIENT_ERROR = 8
RDC_ST_MAX_LIMIT = 10
RDC_ST_GROUP_NOT_FOUND = 17
MAX_STREAMS = 64  # RdcAPIServiceImpl::kMaxPolicyStreams
RDC_GROUP_EMPTY = 1

CALLBACK = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)


def _lib():
    lib = ctypes.CDLL(os.path.join(PREFIX, "lib", "librdc_bootstrap.so"))
    lib.rdc_init.argtypes = [ctypes.c_uint64]
    lib.rdc_connect.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)] + [
        ctypes.c_char_p
    ] * 3
    lib.rdc_disconnect.argtypes = [ctypes.c_void_p]
    lib.rdc_group_gpu_create.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    lib.rdc_group_get_all_ids.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
    ]
    lib.rdc_policy_register.argtypes = [ctypes.c_void_p, ctypes.c_uint32, CALLBACK]
    lib.rdc_policy_unregister.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    lib.rdc_status_string.restype = ctypes.c_char_p
    lib.rdc_status_string.argtypes = [ctypes.c_int]
    assert lib.rdc_init(0) == RDC_ST_OK
    return lib


class Client:
    """One rdc_connect() handle to rdcd."""

    def __init__(self, lib, d):
        self.lib = lib
        self.handle = ctypes.c_void_p()
        host = f"127.0.0.1:{d.port}".encode()
        assert lib.rdc_connect(host, ctypes.byref(self.handle), None, None, None) == RDC_ST_OK

    def status(self, code):
        return f"{code} ({self.lib.rdc_status_string(code).decode()})"

    def create_group(self, name):
        group = ctypes.c_uint32()
        code = self.lib.rdc_group_gpu_create(
            self.handle, RDC_GROUP_EMPTY, name.encode(), ctypes.byref(group)
        )
        assert code == RDC_ST_OK, self.status(code)
        return group.value

    def group_count(self):
        ids = (ctypes.c_uint32 * 64)()
        count = ctypes.c_uint32()
        code = self.lib.rdc_group_get_all_ids(self.handle, ids, ctypes.byref(count))
        assert code == RDC_ST_OK, self.status(code)
        return count.value

    def register(self, group, callback):
        return self.lib.rdc_policy_register(self.handle, group, callback)

    def unregister(self, group):
        return self.lib.rdc_policy_unregister(self.handle, group)

    def disconnect(self):
        if self.handle:
            assert self.lib.rdc_disconnect(self.handle) == RDC_ST_OK
            self.handle = ctypes.c_void_p()


IGNORE = CALLBACK(lambda _: 0)


@pytest.fixture
def rdcd(tools, tmp_path):
    d = Rdcd(tools, tmp_path).start()
    yield d
    code = d.stop()
    assert code == 0, f"rdcd exited with {code}\n{d.output()}"


def _policy_group(d, threshold=1):
    """A group with GPU 0 and a temperature policy; 1 C is always exceeded."""
    gid = _created_id(d.ok("group", "-c", "policy_group"))
    d.ok("group", "-g", gid, "-a", "0")
    d.ok("policy", "-g", gid, "-T", str(threshold), "-A", "0")
    return int(gid)


class Watcher:
    """`rdci policy --reg` on a group, counting the notifications it prints."""

    def __init__(self, d, gid):
        cmd = [
            d.rdci_path,
            "policy",
            "--reg",
            "-g",
            str(gid),
            "-u",
            "--host",
            f"127.0.0.1:{d.port}",
        ]
        self.proc = subprocess.Popen(
            cmd, env=d.rdci_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        self.lines = []
        self.count = 0
        self.first = threading.Event()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            self.lines.append(line)
            if NOTIFICATION in line:
                self.count += 1
                self.first.set()

    def wait_first(self):
        if not self.first.wait(10 * FACTOR):
            self.kill()
            pytest.fail("no notification\n" + "".join(self.lines))

    def kill(self):
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait()


def _threads(pid):
    with open(f"/proc/{pid}/status") as f:
        return int(re.search(r"^Threads:\s+(\d+)", f.read(), re.M).group(1))


def _settled_threads(pid):
    # rdcd's thread count once it holds still for 0.5 s: threads of earlier calls wind down.
    count, same = _threads(pid), 0
    deadline = time.monotonic() + 5 * FACTOR
    while same < 2 and time.monotonic() < deadline:
        time.sleep(0.25)
        now = _threads(pid)
        same = same + 1 if now == count else 0
        count = now
    return count


def _wait_until(predicate, timeout):
    deadline = time.monotonic() + timeout * FACTOR
    while not predicate():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.1)
    return True


def test_policy_client_gtest():
    binary = os.path.join(PREFIX, "share", "rdc", "rdctst_tests", "rdc_policy_client_tests")
    if not os.path.isfile(binary):
        if REQUIRE:
            pytest.fail(f"{binary} is missing")
        pytest.skip(f"{binary} is missing")
    r = subprocess.run([binary], capture_output=True, text=True, timeout=120 * FACTOR)
    assert r.returncode == 0, (
        f"rdc_policy_client_tests exited with {r.returncode}\n{r.stdout[-3000:]}{r.stderr[-2000:]}"
    )


def test_streams_of_clients_that_went_away_end(rdcd):
    gid = _policy_group(rdcd)
    idle = _settled_threads(rdcd.proc.pid)
    watchers = [Watcher(rdcd, gid) for _ in range(4)]
    try:
        for w in watchers:
            w.wait_first()
        busy = _threads(rdcd.proc.pid)
    finally:
        for w in watchers:
            w.kill()
    # The streams hold threads of rdcd's (measured with 4 streams: 3 more, as gRPC's pool
    # already had a spare one); none is left waiting for a client (one spare may stay).
    assert busy > idle + 1, f"{busy} threads with the streams, {idle} before them"
    assert _wait_until(lambda: _threads(rdcd.proc.pid) <= idle + 1, 10), (
        f"{_threads(rdcd.proc.pid)} threads, {idle} before the streams, {busy} with them"
    )


def test_every_stream_of_a_group_gets_every_notification(rdcd):
    # Through the C API: rdci prints at most one notification per 5 s.
    gid = _policy_group(rdcd)
    lib = _lib()
    counts = [0, 0]

    def counter(i):
        def callback(_):
            counts[i] += 1
            return 0

        return CALLBACK(callback)

    callbacks = [counter(0), counter(1)]
    first, second = Client(lib, rdcd), Client(lib, rdcd)
    window = 5 * FACTOR  # about 10 checks of 500 ms
    try:
        assert first.register(gid, callbacks[0]) == RDC_ST_OK
        assert _wait_until(lambda: counts[0] > 0, 10), "no notification"
        before = counts[0]
        time.sleep(window)
        alone = counts[0] - before
        assert alone >= 5, f"{alone} notifications in {window} s"
        assert second.register(gid, callbacks[1]) == RDC_ST_OK
        assert _wait_until(lambda: counts[1] > 0, 10), "no notification for the second client"
        a, b = counts
        time.sleep(window)
        both = (counts[0] - a, counts[1] - b)
        # Shared, each client would get about half of them.
        assert min(both) >= 0.7 * alone, f"{both} notifications each, {alone} for one client"
    finally:
        second.disconnect()
        first.disconnect()


def test_registering_a_missing_group_fails(rdcd):
    r = rdcd.rdci("policy", "--reg", "-g", "42", timeout=15)
    assert r.returncode != 0, r.stdout + r.stderr
    lib = _lib()
    client = Client(lib, rdcd)
    try:
        code = client.register(42, IGNORE)
        assert code == RDC_ST_GROUP_NOT_FOUND, client.status(code)
    finally:
        client.disconnect()


def test_rdcd_serves_64_policy_streams(rdcd):
    lib = _lib()
    first, second = Client(lib, rdcd), Client(lib, rdcd)
    try:
        groups = [first.create_group(f"g{i}") for i in range(MAX_STREAMS)]
        for group in groups:
            code = first.register(group, IGNORE)
            assert code == RDC_ST_OK, f"group {group}: {first.status(code)}"
        code = second.register(groups[0], IGNORE)
        assert code == RDC_ST_MAX_LIMIT, first.status(code)
        # The open streams do not block other calls on the same connection.
        assert first.group_count() == MAX_STREAMS
        # A stream that ends frees its place.
        assert first.unregister(groups[1]) == RDC_ST_OK
        code = _register_eventually(second, groups[0])
        assert code == RDC_ST_OK, first.status(code)
    finally:
        second.disconnect()
        first.disconnect()


def _register_eventually(client, group):
    # rdcd frees a stream's place once its handler sees the end, within about a second.
    deadline = time.monotonic() + 10 * FACTOR
    while True:
        code = client.register(group, IGNORE)
        if code != RDC_ST_MAX_LIMIT or time.monotonic() > deadline:
            return code
        time.sleep(0.2)


def test_rdcd_stops_while_registrations_arrive(tools, tmp_path):
    d = Rdcd(tools, tmp_path).start()
    stop = threading.Event()
    errors = []
    threads = []
    try:
        lib = _lib()
        setup = Client(lib, d)
        groups = [setup.create_group(f"g{i}") for i in range(4)]
        setup.disconnect()

        def churn(group):
            try:
                client = Client(lib, d)
            except AssertionError as e:  # rdcd is already gone
                errors.append(e)
                return
            while not stop.is_set():
                if client.register(group, IGNORE) == RDC_ST_OK:
                    client.unregister(group)
            client.disconnect()

        threads = [threading.Thread(target=churn, args=(g,), daemon=True) for g in groups]
        for t in threads:
            t.start()
        time.sleep(1 * FACTOR)
    finally:
        try:
            code = d.stop()  # fails the test if rdcd does not exit in time
        finally:
            stop.set()
            for t in threads:
                t.join(30 * FACTOR)
    assert code == 0, f"rdcd exited with {code}\n{d.output()}"
    assert not any(t.is_alive() for t in threads), "a client hung after rdcd stopped"
    assert not errors, errors


def test_a_registration_survives_an_rdcd_restart(tools, tmp_path, monkeypatch):
    d = Rdcd(tools, tmp_path).start()
    lib = _lib()
    client = None
    calls = []
    callback = CALLBACK(lambda response: calls.append(1) or 0)
    try:
        gid = _policy_group(d)
        client = Client(lib, d)
        assert client.register(gid, callback) == RDC_ST_OK
        assert _wait_until(lambda: calls, 10), "no notification"
        assert d.stop() == 0
        # The same address again: the stream the old rdcd ended no longer blocks the group.
        monkeypatch.setattr(test_fake_gpu, "_free_port", lambda port=d.port: port)
        d.start()
        assert _policy_group(d) == gid
        calls.clear()
        # The channel may still be backing off from the old rdcd: retry until it reconnects.
        deadline = time.monotonic() + 15 * FACTOR
        code = client.register(gid, callback)
        while code == RDC_ST_CLIENT_ERROR and time.monotonic() < deadline:
            time.sleep(0.5)
            code = client.register(gid, callback)
        assert code == RDC_ST_OK, client.status(code)
        assert _wait_until(lambda: calls, 10), "no notification after the restart"
    finally:
        if client:
            client.disconnect()
        code = d.stop()
    assert code == 0, f"rdcd exited with {code}\n{d.output()}"


# ---------------------------------------------------------------------------
# Stress (RDC_TEST_STRESS=1)
# ---------------------------------------------------------------------------
@pytest.mark.stress
def test_callbacks_unregister_their_group_200_times(rdcd):
    gid = _policy_group(rdcd)
    lib = _lib()
    client = Client(lib, rdcd)
    done = threading.Event()
    results = []

    def unregister_own(_):
        results.append(client.unregister(gid))
        done.set()
        return 0

    callback = CALLBACK(unregister_own)
    try:
        for cycle in range(200):
            done.clear()
            deadline = time.monotonic() + 10 * FACTOR
            code = client.register(gid, callback)
            while code == RDC_ST_CONFLICT and time.monotonic() < deadline:  # still ending
                time.sleep(0.05)
                code = client.register(gid, callback)
            assert code == RDC_ST_OK, f"cycle {cycle}: {client.status(code)}"
            assert done.wait(10 * FACTOR), f"cycle {cycle}: no callback"
        assert set(results) == {RDC_ST_OK}, results
    finally:
        client.disconnect()


@pytest.mark.stress
def test_stream_of_a_stopped_client_ends(tools, tmp_path):
    # rdcd pings every 60 s and gives up 20 s after an unanswered ping.
    d = Rdcd(tools, tmp_path, RDC_LOG="DEBUG").start()
    w = None
    try:
        gid = _policy_group(d)
        w = Watcher(d, gid)
        w.wait_first()
        w.proc.send_signal(signal.SIGSTOP)
        ended = f"Policy stream for group {gid} ended"
        start = time.monotonic()
        while ended not in d.output():
            assert time.monotonic() - start < 240, "the stream did not end within 240 s"
            time.sleep(1)
    finally:
        if w:
            w.proc.send_signal(signal.SIGCONT)
            w.kill()
        code = d.stop()
    assert code == 0, f"rdcd exited with {code}\n{d.output()}"


@pytest.mark.stress
def test_idle_stream_still_notifies_after_pings(rdcd):
    gid = _policy_group(rdcd, threshold=1000)  # never exceeded
    w = Watcher(rdcd, gid)
    try:
        time.sleep(190)  # more than three ping intervals without a message
        assert w.proc.poll() is None, "".join(w.lines)
        rdcd.ok("policy", "-g", str(gid), "-T", "1", "-A", "0")
        w.wait_first()
    finally:
        w.kill()
