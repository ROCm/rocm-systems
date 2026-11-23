"""rdci policy --reg on fake GPUs: the client's policy notification stream.

The client reads the stream on a thread of its own. These tests check that
the thread stops when the stream ends and when rdci unregisters.
"""

import ctypes
import os
import signal
import subprocess
import threading
import time

import pytest

# The helpers and the `tools` fixture come from test_fake_gpu.py, not conftest.py:
from test_fake_gpu import FACTOR, PREFIX, Rdcd, _created_id, tools  # noqa: F401 (fixture)

NOTIFICATION = "A Temperature Limit exceeds the threshold 1 with the value"
# rdc_status_t in include/rdc/rdc.h
RDC_ST_CONFLICT = 7
RDC_ST_CLIENT_ERROR = 8


@pytest.fixture
def rdcd(tools, tmp_path):  # noqa: F811 (the fixture)
    d = Rdcd(tools, tmp_path).start()
    yield d
    d.stop()


def _policy_group(d):
    """A group with GPU 0 and a policy that is always exceeded."""
    gid = _created_id(d.ok("group", "-c", "policy_group"))
    d.ok("group", "-g", gid, "-a", "0")
    d.ok("policy", "-g", gid, "-T", "1", "-A", "0")  # 1 C: always exceeded
    return int(gid)


def _registered_rdci(d):
    """Start `rdci policy --reg` on a group whose policy is always exceeded and
    wait for its first notification."""
    gid = str(_policy_group(d))
    cmd = [d.rdci_path, "policy", "--reg", "-g", gid, "-u", "--host", f"127.0.0.1:{d.port}"]
    proc = subprocess.Popen(
        cmd, env=d.rdci_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    lines, seen = [], threading.Event()

    def read():
        for line in proc.stdout:
            lines.append(line)
            if NOTIFICATION in line:
                seen.set()

    threading.Thread(target=read, daemon=True).start()
    if not seen.wait(10 * FACTOR):
        proc.kill()
        proc.wait()
        pytest.fail("no notification\n" + "".join(lines))
    return proc, lines


def _cpu_seconds(pid):
    with open(f"/proc/{pid}/stat") as f:
        fields = f.read().rsplit(")", 1)[1].split()
    return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")  # utime + stime


def test_registered_rdci_idles_when_rdcd_goes_away(rdcd):
    proc, _ = _registered_rdci(rdcd)
    try:
        rdcd.kill()
        time.sleep(1)
        assert proc.poll() is None, f"rdci exited with {proc.returncode}"
        # CPU time over wall time: these sleeps need no FACTOR.
        before = _cpu_seconds(proc.pid)
        time.sleep(2)
        used = _cpu_seconds(proc.pid) - before
        assert used < 0.5, f"rdci used {used:.2f} s of CPU in the 2 s after rdcd went away"
    finally:
        proc.kill()
        proc.wait()


def test_registered_rdci_exits_cleanly_on_sigint(rdcd):
    for _ in range(3):
        proc, lines = _registered_rdci(rdcd)
        time.sleep(0.2 * FACTOR)  # rdci installs its SIGINT handler after registering
        proc.send_signal(signal.SIGINT)  # rdci unregisters and exits
        try:
            proc.wait(timeout=20 * FACTOR)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            pytest.fail("rdci did not exit after SIGINT\n" + "".join(lines))
        assert proc.returncode == 0, f"rdci exited with {proc.returncode}\n{''.join(lines)}"


CALLBACK = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)


def _connect(d):
    """librdc_bootstrap through ctypes, and a handle connected to `d`."""
    lib = ctypes.CDLL(os.path.join(PREFIX, "lib", "librdc_bootstrap.so"))
    lib.rdc_init.argtypes = [ctypes.c_uint64]
    lib.rdc_connect.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)] + [
        ctypes.c_char_p
    ] * 3
    lib.rdc_policy_register.argtypes = [ctypes.c_void_p, ctypes.c_uint32, CALLBACK]
    lib.rdc_policy_unregister.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    lib.rdc_disconnect.argtypes = [ctypes.c_void_p]
    handle = ctypes.c_void_p()
    assert lib.rdc_init(0) == 0
    host = f"127.0.0.1:{d.port}".encode()
    assert lib.rdc_connect(host, ctypes.byref(handle), None, None, None) == 0
    return lib, handle


def test_callbacks_can_unregister_groups(rdcd):
    """Through the C API: a callback unregisters its own group, then two
    callbacks unregister each other's group at the same time, then
    rdc_disconnect() runs while a callback is registered. None may hang."""
    a, b = _policy_group(rdcd), _policy_group(rdcd)
    lib, handle = _connect(rdcd)
    callbacks, results = [], {}  # callbacks must outlive their registration
    wait = 10 * FACTOR

    def register(group, on_first_call):
        called = threading.Event()

        def callback(_):
            if not called.is_set():
                called.set()
                on_first_call()
            return 0

        callbacks.append(CALLBACK(callback))
        assert lib.rdc_policy_register(handle, group, callbacks[-1]) == 0
        return called

    def unregister(name, group):
        results[name] = lib.rdc_policy_unregister(handle, group)

    disconnected = None
    try:
        # Each group delivers a notification and unregisters once before the
        # callbacks below unregister groups.
        for group in (a, b):
            assert register(group, lambda: None).wait(wait), f"no callback for group {group}"
            assert lib.rdc_policy_unregister(handle, group) == 0

        assert register(a, lambda: unregister("own", a)).wait(wait), "no callback"
        deadline = time.monotonic() + wait
        while "own" not in results and time.monotonic() < deadline:
            time.sleep(0.05)
        assert results.get("own") == 0, results

        both_in = threading.Barrier(2, timeout=wait)

        def cross(name, other):
            both_in.wait()  # both callbacks run at once
            unregister(name, other)

        register(a, lambda: cross("a", b))
        register(b, lambda: cross("b", a))
        deadline = time.monotonic() + 2 * wait
        while len(results) < 3 and time.monotonic() < deadline:
            time.sleep(0.05)
        assert results == {"own": 0, "a": 0, "b": 0}, results

        assert register(a, lambda: None).wait(wait), "no callback"
    finally:
        # Also after a failure: the readers stop before the callbacks go away.
        disconnected = lib.rdc_disconnect(handle)
    assert disconnected == 0


def test_registering_waits_for_an_unregister_to_return(rdcd):
    """Through the C API: while rdc_policy_unregister() waits for a running
    callback of the group, registering the group fails with RDC_ST_CONFLICT.
    rdcd unregisters by group, so the old unregister could otherwise end the
    new stream. Once it returns, registering works."""
    group = _policy_group(rdcd)
    lib, handle = _connect(rdcd)
    wait = 10 * FACTOR
    in_callback, release, called_again = threading.Event(), threading.Event(), threading.Event()

    def blocking(_):
        if not in_callback.is_set():
            in_callback.set()
            release.wait(wait)
        return 0

    callbacks = [
        CALLBACK(blocking),
        CALLBACK(lambda _: 0),
        CALLBACK(lambda _: called_again.set() or 0),
    ]
    results = {}
    unregistering = threading.Thread(
        target=lambda: results.update(unregister=lib.rdc_policy_unregister(handle, group))
    )
    disconnected = None
    try:
        assert lib.rdc_policy_register(handle, group, callbacks[0]) == 0
        assert in_callback.wait(wait), "no callback"
        unregistering.start()
        time.sleep(1 * FACTOR)  # rdcd ends the stream; the unregister waits for the callback
        results["register"] = lib.rdc_policy_register(handle, group, callbacks[1])
        release.set()
        unregistering.join(wait)
        assert results == {"register": RDC_ST_CONFLICT, "unregister": 0}, results
        assert lib.rdc_policy_register(handle, group, callbacks[2]) == 0
        assert called_again.wait(wait), "no callback after registering again"
    finally:
        release.set()
        disconnected = lib.rdc_disconnect(handle)
    assert disconnected == 0


def test_unregister_frees_the_group_when_rdcd_is_gone(rdcd):
    """Through the C API: once rdcd is gone, rdc_policy_unregister() fails but
    returns, and the group is free again: registering it no longer returns
    RDC_ST_CONFLICT. It waits for rdcd to admit the stream, so it reports that
    rdcd cannot be reached."""
    group = _policy_group(rdcd)
    lib, handle = _connect(rdcd)
    wait = 10 * FACTOR
    called = threading.Event()
    callbacks = [CALLBACK(lambda _: called.set() or 0), CALLBACK(lambda _: 0)]
    results = {}
    unregistering = threading.Thread(
        target=lambda: results.update(unregister=lib.rdc_policy_unregister(handle, group)),
        daemon=True,
    )
    disconnected = None
    try:
        assert lib.rdc_policy_register(handle, group, callbacks[0]) == 0
        assert called.wait(wait), "no callback"
        rdcd.kill()
        unregistering.start()
        unregistering.join(wait)
        assert not unregistering.is_alive(), "rdc_policy_unregister() did not return"
        assert results == {"unregister": RDC_ST_CLIENT_ERROR}, results
        assert lib.rdc_policy_register(handle, group, callbacks[1]) == RDC_ST_CLIENT_ERROR
    finally:
        # Not while the unregister still runs: that would free the handle under it.
        if not unregistering.is_alive():
            disconnected = lib.rdc_disconnect(handle)
    assert disconnected == 0


def test_readers_unregistered_from_callbacks_are_joined(rdcd):
    """Through the C API: a group registered again and again, whose callback
    unregisters it. rdc_policy_register() joins the readers of those streams
    once they are done, so their thread stacks do not pile up until
    rdc_disconnect()."""
    group = _policy_group(rdcd)
    lib, handle = _connect(rdcd)
    wait = 10 * FACTOR
    callbacks, results = [], []

    def cycle():
        unregistered = threading.Event()

        def callback(_):
            if not unregistered.is_set():
                results.append(lib.rdc_policy_unregister(handle, group))
                unregistered.set()
            return 0

        callbacks.append(CALLBACK(callback))
        assert lib.rdc_policy_register(handle, group, callbacks[-1]) == 0
        assert unregistered.wait(wait), "no callback"

    def guard_pages():
        # glibc puts a PROT_NONE guard page below each thread stack.
        with open("/proc/self/maps") as f:
            return sum(1 for line in f if line.split()[1] == "---p")

    disconnected = None
    try:
        for _ in range(4):
            cycle()
        before = guard_pages()
        for _ in range(16):
            cycle()
        grown = guard_pages() - before
    finally:
        disconnected = lib.rdc_disconnect(handle)
    assert disconnected == 0
    assert results == [0] * 20, results
    if "asan" not in os.environ.get("LD_PRELOAD", ""):  # ASan maps memory per thread too
        assert grown < 6, f"{grown} more guard pages after 16 more registrations"
