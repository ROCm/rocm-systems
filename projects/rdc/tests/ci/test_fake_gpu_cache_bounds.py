"""Retention limits of watched fields (rdc_field_watch()'s max_keep_samples and
max_keep_age), end to end in embedded mode against the fake amd-smi.

Each scenario watches its own (GPU, field) pair. The scenarios run in three child
Python processes, so a crash or a hang in RDC cannot take pytest down, and a crash
in one group of scenarios leaves the others' results. Helpers come from
test_fake_gpu.py.

RDC evicts cached samples at most once a second (RdcWatchTableImpl::clean_up()), so
a child watches for WATCH_SECONDS, stops every watch and job, waits SETTLE_SECONDS
for one more eviction, and only then counts the samples each pair kept.
"""

import json
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fake_gpu import FACTOR, NUM_GPUS, PREFIX, _first_file, _missing, tools  # noqa: E402,F401

UINT32_MAX = 0xFFFFFFFF
HOUR = 3600.0
UPDATE_US = 10000  # one sample every 10 ms
WATCH_SECONDS = 2.5
SETTLE_SECONDS = 1.5
# Samples a field without an effective limit keeps at least: about 250 arrive
# while it is watched.
MIN_SAMPLES = 120

# Field IDs (rdc.h): job statistics watch GPU_TEMP and GPU_UTIL, among others;
# JUNCTION_TEMP and MEMORY_TEMP are not job fields.
MEMORY_TEMP, GPU_TEMP, JUNCTION_TEMP, GPU_UTIL = 200, 201, 202, 500

_CHILD = r"""
import ctypes, json, sys, time
lib = ctypes.CDLL(sys.argv[1])
spec = json.loads(sys.argv[2])
U32, U64 = ctypes.c_uint32, ctypes.c_uint64

class Data(ctypes.Union):
    _fields_ = [("l_int", ctypes.c_int64), ("dbl", ctypes.c_double), ("str", ctypes.c_char * 256)]
class Value(ctypes.Structure):
    _fields_ = [("field_id", ctypes.c_int), ("status", ctypes.c_int), ("ts", U64),
                ("type", ctypes.c_int), ("value", Data)]

lib.rdc_field_watch.argtypes = [ctypes.c_void_p, U32, U32, U64, ctypes.c_double, U32]
lib.rdc_field_get_value_since.argtypes = [ctypes.c_void_p, U32, U32, U64,
                                          ctypes.POINTER(U64), ctypes.POINTER(Value)]

def call(name, *args):
    st = getattr(lib, name)(*args)
    if st != 0:
        print(json.dumps({"failed": name, "status": st}), flush=True)
        sys.exit(1)

handle = ctypes.c_void_p()
call("rdc_init", U64(0))
call("rdc_start_embedded", 0, ctypes.byref(handle))

groups = {}
def group(gpus):
    key = tuple(gpus)
    if key not in groups:
        gid = U32()
        call("rdc_group_gpu_create", handle, 1, b"cb_%d" % len(groups), ctypes.byref(gid))
        for g in gpus:
            call("rdc_group_gpu_add", handle, gid, g)
        groups[key] = gid
    return groups[key]

watches = []
for i, w in enumerate(spec["watches"]):
    gid = group([w["gpu"]])
    fids = (U32 * 1)(w["field"])
    fg = U32()
    call("rdc_group_field_create", handle, 1, fids, b"cbf_%d" % i, ctypes.byref(fg))
    call("rdc_field_watch", handle, gid, fg, U64(spec["update_us"]), w["age"], w["samples"])
    watches.append((gid, fg))
job = ctypes.create_string_buffer(b"cache_bounds_job", 64)
if spec.get("job_gpus"):
    call("rdc_job_start_stats", handle, group(spec["job_gpus"]), job, U64(spec["update_us"]))

time.sleep(spec["watch_s"])
if spec.get("job_gpus"):
    call("rdc_job_stop_stats", handle, job)
for gid, fg in watches:
    call("rdc_field_unwatch", handle, gid, fg)
time.sleep(spec["settle_s"])

result = {}
for gpu, field in spec["count"]:
    count, since, value = 0, 0, Value()
    while count < 100000:
        nxt = U64(0)
        st = lib.rdc_field_get_value_since(handle, gpu, field, U64(since), ctypes.byref(nxt),
                                           ctypes.byref(value))
        if st != 0 or nxt.value == since:
            break
        count, since = count + 1, nxt.value
    result["%d:%d" % (gpu, field)] = count
print(json.dumps({"counts": result}), flush=True)
call("rdc_stop_embedded", handle)
call("rdc_shutdown")
"""


def _run(tools, watches, count, job_gpus=()):
    """Runs one child; returns {(gpu, field): number of samples cached}."""
    bootstrap = _first_file(os.path.join(PREFIX, "lib", "librdc_bootstrap.so"))
    if bootstrap is None:
        _missing("librdc_bootstrap.so")
    spec = {
        "watches": [{"gpu": g, "field": f, "samples": s, "age": a} for g, f, s, a in watches],
        "job_gpus": list(job_gpus),
        "count": [list(pair) for pair in count],
        "update_us": UPDATE_US,
        "watch_s": WATCH_SECONDS,
        "settle_s": SETTLE_SECONDS,
    }
    env = dict(os.environ, LD_PRELOAD=tools[2], RDC_FAKE_GPUS=str(NUM_GPUS))
    cmd = [sys.executable, "-c", _CHILD, bootstrap, json.dumps(spec)]
    try:
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=60 * FACTOR)
    except subprocess.TimeoutExpired:
        pytest.fail(f"embedded child hung (killed): {spec}")
    assert r.returncode == 0, f"embedded child exited {r.returncode}\n{r.stdout}{r.stderr[-3000:]}"
    # RDC logs to stdout too; the child's result is its one JSON line.
    lines = [line for line in r.stdout.splitlines() if line.startswith("{")]
    assert len(lines) == 1, r.stdout
    counts = json.loads(lines[0])["counts"]
    return {tuple(int(x) for x in key.split(":")): n for key, n in counts.items()}


# Sample limits above INT32_MAX, UINT32_MAX among them. Before, the number of
# samples to drop wrapped around and exceeded the number held, and the eviction
# wrote outside the cache.
LARGE = [3 * 2**30, UINT32_MAX - 1, UINT32_MAX]


@pytest.fixture(scope="module")
def large_limits(tools):
    watches = [(gpu, MEMORY_TEMP, limit, HOUR) for gpu, limit in enumerate(LARGE)]
    return _run(tools, watches, [(gpu, MEMORY_TEMP) for gpu in range(len(LARGE))])


@pytest.mark.parametrize("gpu", range(len(LARGE)), ids=[str(v) for v in LARGE])
def test_a_sample_limit_above_int32_max_keeps_every_sample(large_limits, gpu):
    assert large_limits[(gpu, MEMORY_TEMP)] >= MIN_SAMPLES


# rdc.h: a max_keep_samples or max_keep_age of 0 keeps no samples, and when several
# watches cover a field, the largest limits apply.
@pytest.fixture(scope="module")
def documented_limits(tools):
    watches = [
        (0, JUNCTION_TEMP, 0, HOUR),
        (1, JUNCTION_TEMP, 10, 0.0),
        (2, JUNCTION_TEMP, 10, HOUR),
        (2, JUNCTION_TEMP, 0, HOUR),
        (3, JUNCTION_TEMP, 10, HOUR),
    ]
    return _run(tools, watches, [(gpu, JUNCTION_TEMP) for gpu in range(4)])


def test_a_sample_limit_of_zero_keeps_no_sample(documented_limits):
    assert documented_limits[(0, JUNCTION_TEMP)] == 0


def test_an_age_limit_of_zero_keeps_no_sample(documented_limits):
    assert documented_limits[(1, JUNCTION_TEMP)] == 0


def test_watches_of_one_field_keep_samples_by_the_largest_limits(documented_limits):
    assert documented_limits[(2, JUNCTION_TEMP)] == 10


def test_a_sample_limit_keeps_that_many_samples(documented_limits):
    assert documented_limits[(3, JUNCTION_TEMP)] == 10


# Job statistics watch their fields with limits of 0: they need no cached samples.
@pytest.fixture(scope="module")
def with_a_job(tools):
    watches = [(3, GPU_UTIL, 10, HOUR)]
    return _run(tools, watches, [(3, GPU_UTIL), (3, GPU_TEMP)], job_gpus=[3])


def test_a_job_alone_keeps_no_sample(with_a_job):
    assert with_a_job[(3, GPU_TEMP)] == 0


def test_a_job_does_not_lower_a_watchs_limits(with_a_job):
    assert with_a_job[(3, GPU_UTIL)] == 10
