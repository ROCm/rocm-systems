"""Self-tests of the fake amd-smi's fault knobs, through ctypes on the library itself.

Each test runs a child Python process with the knob in its environment, because the fake reads
its environment once, at the first amdsmi_init(). Helpers come from test_fake_gpu.py.
"""

import json
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fake_gpu import FACTOR, FAKE_NAME, PREFIX, _first_file, _missing  # noqa: E402

AMDSMI_STATUS_SUCCESS = 0
AMDSMI_STATUS_INVAL = 1
AMDSMI_STATUS_NOT_FOUND = 31
AMDSMI_STATUS_NO_DATA = 40
AMDSMI_STATUS_INSUFFICIENT_SIZE = 41
AMDSMI_STATUS_UNKNOWN_ERROR = 0xFFFFFFFF

# Prepended to every child script: loads the fake and defines handle helpers.
_LOAD = r"""
import ctypes, json, os, resource, sys, time
# No core dump from the knobs that abort: with a limit of 1, not 0, the kernel also skips a dump
# piped to a handler such as apport. A hard limit of 0 cannot be raised to 1.
if resource.getrlimit(resource.RLIMIT_CORE)[1] != 0:
    resource.setrlimit(resource.RLIMIT_CORE, (1, 1))
lib = ctypes.CDLL(sys.argv[1])
lib.amdsmi_init.argtypes = [ctypes.c_uint64]
for name in ("amdsmi_init", "amdsmi_shut_down", "amdsmi_get_socket_handles",
             "amdsmi_get_processor_handles", "amdsmi_get_gpu_asic_info",
             "amdsmi_get_gpu_event_notification", "amdsmi_get_clk_freq",
             "amdsmi_get_gpu_compute_process_info", "amdsmi_get_energy_count",
             "amdsmi_get_violation_status", "amdsmi_get_lib_version",
             "amdsmi_get_gpu_cper_entries", "amdsmi_topo_get_link_weight"):
    getattr(lib, name).restype = ctypes.c_uint32

def sockets():
    n = ctypes.c_uint32(64)
    out = (ctypes.c_void_p * 64)()
    assert lib.amdsmi_get_socket_handles(ctypes.byref(n), out) == 0
    return [out[i] for i in range(n.value)]

def procs_of(socket):
    m = ctypes.c_uint32(8)
    out = (ctypes.c_void_p * 8)()
    st = lib.amdsmi_get_processor_handles(ctypes.c_void_p(socket), ctypes.byref(m), out)
    return st, [out[j] for j in range(m.value)] if st == 0 else []

def handles():
    procs = []
    for s in sockets():
        st, p = procs_of(s)
        assert st == 0
        procs += p
    return procs

def asic(h, buf=None):
    buf = buf or ctypes.create_string_buffer(4096)
    return lib.amdsmi_get_gpu_asic_info(ctypes.c_void_p(h), buf), buf
"""
# The usual start: the library initialized once.
_PRELUDE = _LOAD + "assert lib.amdsmi_init(2) == 0\n"


@pytest.fixture(scope="module")
def fake():
    path = _first_file(
        os.environ.get("RDC_FAKE_LIB"),
        os.path.join(PREFIX, "share", "rdc", "rdctst_tests", "fake_gpu", FAKE_NAME),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), FAKE_NAME),
    )
    if path is None:
        _missing(FAKE_NAME)
    return path


def _run(fake, script, prelude=_PRELUDE, **env):
    # Only the test's own knobs: one exported by the caller would change the result. LD_PRELOAD
    # stays, as sanitizer jobs preload their runtime for the instrumented fake.
    full = {k: v for k, v in os.environ.items() if not k.startswith("RDC_FAKE_")}
    full["RDC_FAKE_GPUS"] = "2"
    full.update({k: str(v) for k, v in env.items()})
    return subprocess.run(
        [sys.executable, "-c", prelude + script, fake],
        env=full,
        capture_output=True,
        text=True,
        timeout=60 * FACTOR,
    )


def _child(fake, script, prelude=_PRELUDE, **env):
    r = _run(fake, script, prelude, **env)
    assert r.returncode == 0, f"child exited {r.returncode}\n{r.stdout}\n{r.stderr[-3000:]}"
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_handles_are_never_null_and_go_stale_after_the_last_shutdown(fake):
    out = _child(
        fake,
        r"""
def weight(src, dst):  # a call with a second processor handle
    w = ctypes.c_uint64()
    return lib.amdsmi_topo_get_link_weight(ctypes.c_void_p(src), ctypes.c_void_p(dst), ctypes.byref(w))
old = handles()
old_socket = sockets()[0]
before = asic(old[0])[0]
lib.amdsmi_init(2)            # a second reference: the handles stay valid
lib.amdsmi_shut_down()
still = asic(old[0])[0]
lib.amdsmi_shut_down()        # the last reference: a new generation
lib.amdsmi_init(2)
new = handles()
print(json.dumps({"old": old, "new": new, "before": before, "still": still,
                  "stale": asic(old[0])[0], "fresh": asic(new[0])[0],
                  "null": asic(0)[0], "gen0": asic(5)[0],
                  "stale_socket": procs_of(old_socket)[0], "fresh_socket": procs_of(sockets()[0])[0],
                  "null_socket": procs_of(0)[0], "gen0_socket": procs_of(5)[0],
                  "dst": [weight(new[0], new[1]), weight(new[0], old[1]), weight(new[0], 0)]}))
""",
    )
    assert all(out["old"]) and all(out["new"]), "a handle is NULL"
    assert set(out["old"]).isdisjoint(out["new"]), "re-init handed out the old handle values"
    assert out["before"] == out["still"] == out["fresh"] == AMDSMI_STATUS_SUCCESS
    # As from amd-smi: an unknown processor gets NOT_FOUND, an unknown socket INVAL.
    assert out["stale"] == out["gen0"] == AMDSMI_STATUS_NOT_FOUND
    assert out["null"] == AMDSMI_STATUS_INVAL
    assert out["fresh_socket"] == AMDSMI_STATUS_SUCCESS
    assert out["stale_socket"] == out["null_socket"] == out["gen0_socket"] == AMDSMI_STATUS_INVAL
    # The same rules for the second processor handle of a topology call.
    assert out["dst"] == [AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_FOUND, AMDSMI_STATUS_INVAL]


def test_unfilled_call_succeeds_without_writing(fake):
    out = _child(
        fake,
        r"""
h = handles()
st0, b0 = asic(h[0], ctypes.create_string_buffer(b"\x5a" * 4096, 4096))
st1, b1 = asic(h[1], ctypes.create_string_buffer(b"\x5a" * 4096, 4096))
print(json.dumps({"st": [st0, st1], "untouched": [b0.raw == b"\x5a" * 4096,
                                                 b1.raw == b"\x5a" * 4096]}))
""",
        RDC_FAKE_UNFILLED="amdsmi_get_gpu_asic_info@1",
    )
    assert out["st"] == [AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_SUCCESS]
    assert out["untouched"] == [False, True]


def test_fail_once_fails_only_the_first_call(fake):
    out = _child(
        fake,
        r"""
h = handles()
print(json.dumps([asic(h[0])[0], asic(h[0])[0], asic(h[1])[0]]))
""",
        RDC_FAKE_FAIL_ONCE="amdsmi_get_gpu_asic_info@0",
    )
    assert out == [AMDSMI_STATUS_UNKNOWN_ERROR, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_SUCCESS]


def test_init_honors_fail_once_and_delay(fake):
    out = _child(
        fake,
        r"""
t = time.monotonic()
first = lib.amdsmi_init(2)
print(json.dumps([first, time.monotonic() - t, lib.amdsmi_init(2)]))
""",
        prelude=_LOAD,
        RDC_FAKE_FAIL_ONCE="amdsmi_init",
        RDC_FAKE_DELAY_MS="amdsmi_init=300",
    )
    assert [out[0], out[2]] == [AMDSMI_STATUS_UNKNOWN_ERROR, AMDSMI_STATUS_SUCCESS], out
    assert out[1] >= 0.3, out


def test_delay_slows_down_the_call(fake):
    out = _child(
        fake,
        r"""
h = handles()
t = time.monotonic(); st = asic(h[0])[0]
print(json.dumps({"st": st, "s": time.monotonic() - t}))
""",
        RDC_FAKE_DELAY_MS="amdsmi_get_gpu_asic_info@0=300",
    )
    assert out["st"] == AMDSMI_STATUS_SUCCESS
    assert out["s"] >= 0.3


def test_cper_count_can_exceed_the_capacity(fake):
    out = _child(
        fake,
        r"""
h = handles()
data = ctypes.create_string_buffer(65536)
size = ctypes.c_uint64(65536)
hdrs = (ctypes.c_void_p * 256)()
count = ctypes.c_uint64(256)
cursor = ctypes.c_uint64(0)
st = lib.amdsmi_get_gpu_cper_entries(ctypes.c_void_p(h[0]), ctypes.c_uint32(0xFF), data,
                                     ctypes.byref(size), hdrs, ctypes.byref(count),
                                     ctypes.byref(cursor))
print(json.dumps({"st": st, "count": count.value, "set": sum(1 for x in hdrs if x),
                  "sig": ctypes.string_at(hdrs[0], 4).decode()}))
""",
        RDC_FAKE_CPER_COUNT="300",
    )
    assert out == {"st": AMDSMI_STATUS_SUCCESS, "count": 300, "set": 256, "sig": "CPER"}


def test_events_are_delivered_once_with_the_reported_count(fake):
    out = _child(
        fake,
        r"""
class Event(ctypes.Structure):
    _fields_ = [("processor_handle", ctypes.c_void_p), ("event", ctypes.c_uint32),
                ("message", ctypes.c_char * 256)]
h = handles()
data = (Event * 64)()
n = ctypes.c_uint32(64)
st = lib.amdsmi_get_gpu_event_notification(0, ctypes.byref(n), data)
first = {"st": st, "n": n.value, "events": [data[0].event, data[1].event],
         "handles": [asic(data[0].processor_handle or 0)[0], asic(data[1].processor_handle or 0)[0]]}
n2 = ctypes.c_uint32(64)
st2 = lib.amdsmi_get_gpu_event_notification(0, ctypes.byref(n2), data)
print(json.dumps({"first": first, "second": [st2, n2.value]}))
""",
        RDC_FAKE_EVENTS="1@1,14@x,count=200",
    )
    assert out["first"] == {
        "st": AMDSMI_STATUS_SUCCESS,
        "n": 200,
        "events": [1, 14],
        "handles": [AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_FOUND],
    }
    assert out["second"] == [AMDSMI_STATUS_NO_DATA, 0]


def test_events_fill_at_most_the_capacity_per_call(fake):
    out = _child(
        fake,
        r"""
class Event(ctypes.Structure):
    _fields_ = [("processor_handle", ctypes.c_void_p), ("event", ctypes.c_uint32),
                ("message", ctypes.c_char * 256)]
handles()
data = (Event * 4)()
calls = []
for room in (0, 1, 1, 1):
    n = ctypes.c_uint32(room)
    st = lib.amdsmi_get_gpu_event_notification(0, ctypes.byref(n), data)
    calls.append([st, n.value, data[0].event if st == 0 else None])
print(json.dumps(calls))
""",
        RDC_FAKE_EVENTS="1@0,2@1",
    )
    assert out == [
        [AMDSMI_STATUS_INVAL, 0, None],  # no room: amd-smi rejects the call
        [AMDSMI_STATUS_SUCCESS, 1, 1],
        [AMDSMI_STATUS_SUCCESS, 1, 2],
        [AMDSMI_STATUS_NO_DATA, 0, None],
    ]


@pytest.mark.parametrize("n", [0, 2])
def test_num_freq_sets_the_levels_and_no_current_level(fake, n):
    out = _child(
        fake,
        r"""
class Freqs(ctypes.Structure):
    _fields_ = [("has_deep_sleep", ctypes.c_bool), ("num_supported", ctypes.c_uint32),
                ("current", ctypes.c_uint32), ("frequency", ctypes.c_uint64 * 33)]
h = handles()
f = Freqs()
st = lib.amdsmi_get_clk_freq(ctypes.c_void_p(h[0]), 0, ctypes.byref(f))
print(json.dumps([st, f.num_supported, f.current]))
""",
        RDC_FAKE_NUM_FREQ=str(n),
    )
    # amd-smi's "no current level"; only the empty table (n = 0) is something it never returns.
    assert out == [AMDSMI_STATUS_SUCCESS, n, 0xFFFFFFFF]


def test_process_list_grows_between_calls(fake):
    out = _child(
        fake,
        r"""
class Proc(ctypes.Structure):
    _fields_ = [("process_id", ctypes.c_uint32), ("vram_usage", ctypes.c_uint64),
                ("sdma_usage", ctypes.c_uint64), ("cu_occupancy", ctypes.c_uint32),
                ("evicted_time", ctypes.c_uint32)]
info = lib.amdsmi_get_gpu_compute_process_info
gpus = lib.amdsmi_get_gpu_compute_process_gpus
gpus.restype = ctypes.c_uint32
def third_known():  # pid 1002, the third process
    k = ctypes.c_uint32(0)
    return gpus(1002, None, ctypes.byref(k))
n = ctypes.c_uint32(0)
st1 = info(None, ctypes.byref(n))           # 2 processes
before = third_known()
count = n.value
small = (Proc * count)()
st2 = info(small, ctypes.byref(n))          # 3 processes: does not fit
after = third_known()
m = ctypes.c_uint32(8)
big = (Proc * 8)()
st3 = info(big, ctypes.byref(m))            # 4 processes
print(json.dumps({"count": [st1, count], "small": [st2, n.value], "third": [before, after],
                  "small_pids": [p.process_id for p in small],
                  "small_cus": [p.cu_occupancy for p in small],
                  "big": [st3, m.value], "big_pids": [p.process_id for p in big[:m.value]],
                  "big_cus": [p.cu_occupancy for p in big[:m.value]]}))
""",
        RDC_FAKE_PROCS="2",
        RDC_FAKE_PROCS_GROW="1",
    )
    assert out["count"] == [AMDSMI_STATUS_SUCCESS, 2]
    # The process-GPU query knows the processes of the last list.
    assert out["third"] == [AMDSMI_STATUS_NOT_FOUND, AMDSMI_STATUS_SUCCESS]
    # As from amd-smi: a list that does not fit keeps its count and holds process IDs only.
    assert out["small"] == [AMDSMI_STATUS_INSUFFICIENT_SIZE, 2]
    assert out["small_pids"] == [1000, 1001] and out["small_cus"] == [0, 0]
    assert out["big"] == [AMDSMI_STATUS_SUCCESS, 4]
    assert out["big_pids"] == [1000, 1001, 1002, 1003]
    assert min(out["big_cus"]) >= 1, "RDC skips a process that occupies no CU"


def test_process_gpus_count_and_fill_like_amdsmi(fake):
    out = _child(
        fake,
        r"""
gpus = lib.amdsmi_get_gpu_compute_process_gpus
gpus.restype = ctypes.c_uint32
def query(pid, room, with_list=True, with_count=True):
    n = ctypes.c_uint32(room)
    ids = (ctypes.c_uint32 * 4)(99, 99, 99, 99)
    st = gpus(pid, ids if with_list else None, ctypes.byref(n) if with_count else None)
    return [st, n.value, ids[0]]
info = lib.amdsmi_get_gpu_compute_process_info
m = ctypes.c_uint32(0)
assert info(None, ctypes.byref(m)) == 0 and m.value == 2
print(json.dumps({"count": query(1001, 0, with_list=False), "no_room": query(1001, 0),
                  "fill": query(1001, 4), "no_count": query(1001, 4, with_count=False),
                  "unknown": query(1002, 4)}))
""",
        RDC_FAKE_PROCS="2",
    )
    # As with amd-smi: a NULL list asks for the count; a list without room keeps the count.
    assert out["count"] == [AMDSMI_STATUS_SUCCESS, 1, 99]
    assert out["no_room"] == [AMDSMI_STATUS_INSUFFICIENT_SIZE, 0, 99]
    assert out["fill"] == [AMDSMI_STATUS_SUCCESS, 1, 1]  # pid 1001 runs on GPU 1
    assert out["no_count"][0] == AMDSMI_STATUS_INVAL
    assert out["unknown"][0] == AMDSMI_STATUS_NOT_FOUND


def test_counter_reset_steps_back_when_armed(fake, tmp_path):
    arm = tmp_path / "arm"
    out = _child(
        fake,
        r"""
h = ctypes.c_void_p(handles()[0])
e = ctypes.c_uint64(); r = ctypes.c_float(); t = ctypes.c_uint64()
class Violation(ctypes.Structure):  # the head of amdsmi_violation_status_t
    _fields_ = [("reference_timestamp", ctypes.c_uint64), ("violation_timestamp", ctypes.c_uint64),
                ("acc_counter", ctypes.c_uint64)]
v = ctypes.create_string_buffer(1 << 18)  # room for the whole struct
def read():
    assert lib.amdsmi_get_energy_count(h, ctypes.byref(e), ctypes.byref(r), ctypes.byref(t)) == 0
    assert lib.amdsmi_get_violation_status(h, v) == 0
    return [e.value, Violation.from_buffer(v).acc_counter]
time.sleep(0.3 * float(os.environ.get("RDC_TEST_TIME_FACTOR", "1")))
before = read()
open(os.environ["RDC_FAKE_ARM"], "w").close()
print(json.dumps([before, read()]))
""",
        RDC_FAKE_COUNTER_RESET="1",
        RDC_FAKE_ARM=str(arm),
    )
    before, after = out
    assert min(before) >= 300, out
    # Restarted from 0 at the reset, so behind the earlier reading.
    assert max(after) < min(before), out


def test_version_build_can_be_null(fake):
    out = _child(
        fake,
        r"""
class Version(ctypes.Structure):
    _fields_ = [("major", ctypes.c_uint32), ("minor", ctypes.c_uint32),
                ("release", ctypes.c_uint32), ("build", ctypes.c_void_p)]
v = Version()
st = lib.amdsmi_get_lib_version(ctypes.byref(v))
print(json.dumps([st, v.build]))
""",
        RDC_FAKE_VERSION_BUILD_NULL="1",
    )
    assert out == [AMDSMI_STATUS_SUCCESS, None]


@pytest.mark.parametrize(
    "knob,value",
    [
        ("RDC_FAKE_DELAY_MS", "amdsmi_init"),
        ("RDC_FAKE_DELAY_MS", "amdsmi_init=-1"),
        ("RDC_FAKE_EVENTS", "16@0"),
        ("RDC_FAKE_EVENTS", "1@2"),
        ("RDC_FAKE_EVENTS", "count=5"),
        ("RDC_FAKE_CPER_COUNT", "many"),
        ("RDC_FAKE_FAIL", "amdsmi_get_power_info@2"),  # GPUs 0 and 1 only
        ("RDC_FAKE_FAIL", "amdsmi_init=AMDSMI_STATUS_TYPO"),
        ("RDC_FAKE_COUNTER_RESET", "1"),  # without RDC_FAKE_ARM it would never reset
        ("RDC_FAKE_LOG", "/nonexistent/fake.log"),
    ],
)
def test_a_malformed_knob_aborts(fake, knob, value):
    r = _run(fake, "lib.amdsmi_init(2)\n", prelude=_LOAD, **{knob: value})
    assert r.returncode != 0 and f"fake amd-smi: {knob}" in r.stderr, r.stderr[-3000:]
