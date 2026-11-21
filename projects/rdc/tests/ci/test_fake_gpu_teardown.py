"""Embedded RDC's lifecycle, against the fake amd-smi: rdc_shutdown() while a
handle is still open, restarts after rdc_shutdown(), RDC libraries from another
build, background updates asked for from several threads, and how often the
update loop wakes up.

Runs like test_fake_gpu.py (same environment variables and helpers).
"""

import json
import os
import shutil
import subprocess
import sys
import time

import pytest

# The helpers and the fixtures come from test_fake_gpu.py, not conftest.py:
# tests/ci/conftest.py is not installed with the suite (share/rdc/rdctst_tests/fake_gpu).
from test_fake_gpu import (  # noqa: F401 (tools and rdcd are fixtures)
    FACTOR,
    NUM_GPUS,
    PREFIX,
    _first_file,
    _missing,
    golden,
    rdcd,
    tools,
)

RDC_ST_FAIL_LOAD_MODULE = 3
OTHER_BUILD_NAME = "librdc_other_build.so"
FIELDS = (100, 502)

# What every child script starts with: the bootstrap, the value structs and helpers.
_PRELUDE = r"""
import ctypes, json, sys, threading, time
lib = ctypes.CDLL(sys.argv[1])
NUM_GPUS = int(sys.argv[2])
AUTO, MANUAL = 0, 1  # rdc_operation_mode_t
class Data(ctypes.Union):
    _fields_ = [("l_int", ctypes.c_int64), ("dbl", ctypes.c_double), ("str", ctypes.c_char * 256)]
class Value(ctypes.Structure):
    _fields_ = [("field_id", ctypes.c_uint32), ("status", ctypes.c_int), ("ts", ctypes.c_uint64),
                ("type", ctypes.c_int), ("value", Data)]
def call(name, *args):
    st = getattr(lib, name)(*args)
    if st != 0:
        print(json.dumps({"failed": name, "status": st}), flush=True)
        sys.exit(1)
def start(mode):
    handle = ctypes.c_void_p()
    call("rdc_start_embedded", mode, ctypes.byref(handle))
    return handle
def watch(handle, name, fields, freq_us, gpus=None, keep=10):
    group, fgroup = ctypes.c_uint32(), ctypes.c_uint32()
    call("rdc_group_gpu_create", handle, 1, name, ctypes.byref(group))  # RDC_GROUP_EMPTY
    for gpu in (range(NUM_GPUS) if gpus is None else gpus):
        call("rdc_group_gpu_add", handle, group, gpu)
    ids = (ctypes.c_uint32 * len(fields))(*fields)
    call("rdc_group_field_create", handle, len(fields), ids, name, ctypes.byref(fgroup))
    # The cache keeps the last `keep` samples of each field, for at most 60 s.
    call("rdc_field_watch", handle, group, fgroup, ctypes.c_uint64(freq_us), ctypes.c_double(60), keep)
def latest(handle, fields):
    values = {}
    for gpu in range(NUM_GPUS):
        for f in fields:
            v = Value()
            st = lib.rdc_field_get_latest_value(handle, gpu, f, ctypes.byref(v))
            values["%d:%d" % (gpu, f)] = v.value.l_int if st == 0 and v.status == 0 else "status %d/%d" % (st, v.status)
    return values
def samples(handle, gpu, field):
    # Counts the cached samples of one field, oldest first.
    count, since, nxt, v = 0, 0, ctypes.c_uint64(), Value()
    while count < 100000:
        if lib.rdc_field_get_value_since(handle, gpu, field, ctypes.c_uint64(since), ctypes.byref(nxt), ctypes.byref(v)) != 0:
            break
        count += 1
        if nxt.value <= since or nxt.value == v.ts + 1:
            break
        since = nxt.value
    return count
"""


@pytest.fixture(scope="module")
def bootstrap():
    path = _first_file(os.path.join(PREFIX, "lib", "librdc_bootstrap.so"))
    if path is None:
        _missing("librdc_bootstrap.so")
    return path


def _want(fields):
    return {f"{gpu}:{f}": golden.expected(f, gpu) for gpu in range(NUM_GPUS) for f in fields}


def _embedded(tools, bootstrap, script, *args, timeout=120, env_update=None):
    """Run _PRELUDE + `script` with the fake preloaded; return the JSON it printed last."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("RDC_FAKE_")}
    env.update(LD_PRELOAD=tools[2], RDC_FAKE_GPUS=str(NUM_GPUS), RDC_FAKE_EVENT_WAIT_MS="100")
    env.update(env_update or {})
    try:
        r = subprocess.run(
            [sys.executable, "-c", _PRELUDE + script, bootstrap, str(NUM_GPUS), *args],
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout * FACTOR,
        )
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        pytest.fail(
            f"child did not finish within {timeout * FACTOR:.0f} s; it printed:\n{out[-3000:]}"
        )
    assert r.returncode == 0, f"child exited {r.returncode}\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}"
    return json.loads(r.stdout.strip().splitlines()[-1])


# ---------------------------------------------------------------------------
# rdc_shutdown() with a handle still open, and restarts after rdc_shutdown()
# ---------------------------------------------------------------------------
_LIVE_HANDLE = r"""
fields = [int(f) for f in sys.argv[3].split(",")]
call("rdc_init", ctypes.c_uint64(0))
handle = start(AUTO)  # RDC's update threads run
watch(handle, b"live", fields, 10000)  # every 10 ms
call("rdc_field_update_all", handle, 1)
before = latest(handle, fields)
call("rdc_shutdown")  # the handle is still open
time.sleep(0.5)  # its update thread keeps running
after = latest(handle, fields)
call("rdc_stop_embedded", handle)
call("rdc_init", ctypes.c_uint64(0))
handle = start(AUTO)
watch(handle, b"again", fields, 1000000)
call("rdc_field_update_all", handle, 1)
again = latest(handle, fields)
call("rdc_stop_embedded", handle)
call("rdc_shutdown")
print(json.dumps({"before": before, "after": after, "again": again}))
"""


def test_rdc_shutdown_leaves_an_open_handle_working(tools, bootstrap):
    # rdc_shutdown() used to unload librdc under the open handle's threads and
    # vtable: a crash in builds where nothing else keeps librdc loaded (clang).
    out = _embedded(tools, bootstrap, _LIVE_HANDLE, ",".join(map(str, FIELDS)))
    want = _want(FIELDS)
    for phase in ("before", "after", "again"):
        assert out[phase] == want, f"{phase}: {out[phase]}"


_RESTARTS = r"""
fields, rounds = [int(f) for f in sys.argv[3].split(",")], int(sys.argv[4])
out = []
for r in range(rounds):
    call("rdc_init", ctypes.c_uint64(0))
    handle = start(AUTO)
    watch(handle, b"r%d" % r, fields, 1000000)
    call("rdc_field_update_all", handle, 1)
    out.append(latest(handle, fields))
    call("rdc_stop_embedded", handle)
    call("rdc_shutdown")
    print("round %d done" % r, flush=True)
print(json.dumps(out))
"""


def test_embedded_rdc_restarts_after_rdc_shutdown(tools, bootstrap):
    # Stop, shut down, initialize and start again, 20 times in one process (an
    # embedded restart once hung in rdc_field_watch on one machine).
    rounds = _embedded(tools, bootstrap, _RESTARTS, ",".join(map(str, FIELDS)), "20")
    want = _want(FIELDS)
    assert len(rounds) == 20
    for n, got in enumerate(rounds):
        assert got == want, f"round {n}: {got}"


# ---------------------------------------------------------------------------
# librdc.so and librdc_client.so from another build
# ---------------------------------------------------------------------------
_OTHER_BUILD = r"""
handle = ctypes.c_void_p(0x1234)
st = lib.rdc_start_embedded(MANUAL, ctypes.byref(handle))
client = ctypes.c_void_p(0x1234)
connect = lib.rdc_connect(b"127.0.0.1:1", ctypes.byref(client), None, None, None)
print(json.dumps({"start": st, "handle": handle.value or 0, "connect": connect,
                  "client": client.value or 0}))
"""


def test_rdc_libraries_from_another_build_fail_to_load(tools, bootstrap, tmp_path):
    # A library that calls a function the bootstrap's build lacks must fail to load,
    # not stop the process at that call.
    other = _first_file(os.path.join(os.path.dirname(tools[2]), OTHER_BUILD_NAME))
    if other is None:
        _missing(OTHER_BUILD_NAME)
    for name in ("librdc.so", "librdc_client.so"):
        shutil.copyfile(other, tmp_path / name)
    path = os.environ.get("LD_LIBRARY_PATH")
    out = _embedded(
        tools,
        bootstrap,
        _OTHER_BUILD,
        env_update={"LD_LIBRARY_PATH": str(tmp_path) + (":" + path if path else "")},
    )
    assert out == {
        "start": RDC_ST_FAIL_LOAD_MODULE,
        "handle": 0,
        "connect": RDC_ST_FAIL_LOAD_MODULE,
        "client": 0,
    }, out


# ---------------------------------------------------------------------------
# Background updates asked for from several threads
# ---------------------------------------------------------------------------
_BACKGROUND_UPDATES = r"""
fields, threads, calls = [int(f) for f in sys.argv[3].split(",")], int(sys.argv[4]), int(sys.argv[5])
call("rdc_init", ctypes.c_uint64(0))
handle = start(MANUAL)
watch(handle, b"bg", fields, 1000)  # due at every update
failed = []
def ask():
    for _ in range(calls):
        st = lib.rdc_field_update_all(handle, 0)  # 0: in the background
        if st != 0:
            failed.append(st)
workers = [threading.Thread(target=ask) for _ in range(threads)]
for w in workers:
    w.start()
for w in workers:
    w.join()
call("rdc_field_update_all", handle, 1)
values = latest(handle, fields)
call("rdc_stop_embedded", handle)
call("rdc_shutdown")
print(json.dumps({"failed": failed, "values": values}))
"""


def test_background_updates_from_several_threads(tools, bootstrap):
    # rdcd passes every client's UpdateAllFields to rdc_field_update_all(), so
    # requests from several clients overlap like these threads' calls.
    out = _embedded(tools, bootstrap, _BACKGROUND_UPDATES, ",".join(map(str, FIELDS)), "8", "500")
    assert out["failed"] == [], out["failed"][:20]
    assert out["values"] == _want(FIELDS), out["values"]


# ---------------------------------------------------------------------------
# The update loop: on time for watched fields, asleep when nothing is due
# ---------------------------------------------------------------------------
_FREQUENCIES = r"""
window = float(sys.argv[3])
call("rdc_init", ctypes.c_uint64(0))
handle = start(AUTO)
watch(handle, b"fast", [100], 10000, gpus=[0], keep=100000)  # every 10 ms
watch(handle, b"slow", [502], 1000000, gpus=[0], keep=100000)  # every second
time.sleep(window)
out = {"fast": samples(handle, 0, 100), "slow": samples(handle, 0, 502)}
call("rdc_stop_embedded", handle)
call("rdc_shutdown")
print(json.dumps(out))
"""
FREQUENCY_WINDOW_S = 3


def test_watched_fields_keep_their_update_frequency(tools, bootstrap):
    out = _embedded(tools, bootstrap, _FREQUENCIES, str(FREQUENCY_WINDOW_S))
    # About 300 and 3; lower bounds only, for loaded hosts.
    assert out["fast"] >= 100, f"{out['fast']} samples of a 10 ms field in {FREQUENCY_WINDOW_S} s"
    assert out["slow"] >= 2, f"{out['slow']} samples of a 1 s field in {FREQUENCY_WINDOW_S} s"


IDLE_WINDOW_S = 3
# RDC's update loop runs in the thread it names rdc_update. With nothing due it wakes
# up 10 times a second; a loop that sleeps 1 ms wakes up about 1000 times.
UPDATE_THREAD = "rdc_update"
UPDATE_MAX_WAKEUPS_PER_S = 50


def _voluntary_switches(pid, name):
    """Voluntary context switches (each wait is one) of the process's threads named `name`."""
    counts = []
    for tid in os.listdir(f"/proc/{pid}/task"):
        try:
            with open(f"/proc/{pid}/task/{tid}/comm") as comm:
                if comm.read().strip() != name:
                    continue
            with open(f"/proc/{pid}/task/{tid}/status") as status:
                for line in status:
                    if line.startswith("voluntary_ctxt_switches:"):
                        counts.append(int(line.split()[1]))
        except OSError:
            pass  # the thread has ended
    return counts


def test_an_idle_rdcd_rarely_wakes_up(rdcd):
    rdcd.ok("discovery", "-l")
    time.sleep(1)
    before = _voluntary_switches(rdcd.proc.pid, UPDATE_THREAD)
    assert len(before) == 1, f"rdcd has {len(before)} threads named {UPDATE_THREAD}"
    time.sleep(IDLE_WINDOW_S)
    after = _voluntary_switches(rdcd.proc.pid, UPDATE_THREAD)
    rdcd.check_alive("the idle window")
    assert len(after) == 1, f"rdcd has {len(after)} threads named {UPDATE_THREAD}"
    rate = (after[0] - before[0]) / IDLE_WINDOW_S
    assert rate < UPDATE_MAX_WAKEUPS_PER_S, (
        f"an idle rdcd's update loop woke up {rate:.0f} times a second"
    )
