"""Configuration settings (rdc_config_set(), rdc_config_get(), rdc_config_clear())
end to end in embedded mode against the fake amd-smi.

Each scenario runs in a child Python process, so a crash in RDC cannot take pytest
down. The child prints one JSON line: the status of each call and the amd-smi
configuration calls (RDC_FAKE_LOG) that call made. Helpers come from
test_fake_gpu.py.
"""

import json
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fake_gpu import FACTOR, NUM_GPUS, PREFIX, _first_file, _missing, tools  # noqa: E402,F401

# rdc_status_t and rdc_config_type_t (rdc.h).
OK, BAD_PARAMETER, NOT_FOUND, UNKNOWN_ERROR = 0, 5, 6, 0xFFFFFFFF
GFX, MEM, POWER = 0, 1, 2

# The amd-smi calls the configuration settings make.
CONFIG_APIS = {
    "amdsmi_get_power_cap_info",
    "amdsmi_set_power_cap",
    "amdsmi_get_clock_info",
    "amdsmi_set_gpu_clk_limit",
    "amdsmi_get_clk_freq",
    "amdsmi_set_gpu_perf_level",
}

_CHILD = r"""
import ctypes, json, os, sys
lib = ctypes.CDLL(sys.argv[1])
spec = json.loads(sys.argv[2])
U32, U64 = ctypes.c_uint32, ctypes.c_uint64

class Setting(ctypes.Structure):
    _fields_ = [("type", U32), ("target_value", U64)]
class Settings(ctypes.Structure):
    _fields_ = [("total_settings", U32), ("settings", Setting * 32)]

lib.rdc_config_set.argtypes = [ctypes.c_void_p, U32, Setting]
lib.rdc_config_get.argtypes = [ctypes.c_void_p, U32, ctypes.POINTER(Settings)]
lib.rdc_config_clear.argtypes = [ctypes.c_void_p, U32]
for name in ("rdc_config_set", "rdc_config_get", "rdc_config_clear"):
    getattr(lib, name).restype = U32

def call(name, *args):
    st = getattr(lib, name)(*args)
    if st != 0:
        print(json.dumps({"failed": name, "status": st}), flush=True)
        sys.exit(1)

seen = [0]
def new_calls():
    with open(os.environ["RDC_FAKE_LOG"]) as f:
        lines = f.read().splitlines()
    out, seen[0] = lines[seen[0]:], len(lines)
    return out

handle = ctypes.c_void_p()
call("rdc_init", U64(0))
call("rdc_start_embedded", 0, ctypes.byref(handle))
groups = {}
for name, gpus in spec["groups"].items():
    gid = U32()
    call("rdc_group_gpu_create", handle, 1, name.encode(), ctypes.byref(gid))
    for g in gpus:
        call("rdc_group_gpu_add", handle, gid, g)
    groups[name] = gid.value
new_calls()

results = []
for step in spec["steps"]:
    if step[0] == "arm":
        open(os.environ["RDC_FAKE_ARM"], "w").close()
        continue
    op, gid = step[0], groups[step[1]]
    result = {"op": op}
    if op == "set":
        result["status"] = lib.rdc_config_set(handle, gid, Setting(step[2], step[3]))
    elif op == "get":
        out = Settings()
        result["status"] = lib.rdc_config_get(handle, gid, ctypes.byref(out))
        if result["status"] == 0:
            n = min(out.total_settings, 32)
            result["settings"] = sorted([s.type, s.target_value] for s in out.settings[:n])
    elif op == "clear":
        result["status"] = lib.rdc_config_clear(handle, gid)
    result["calls"] = new_calls()
    results.append(result)
print(json.dumps({"results": results}), flush=True)
call("rdc_stop_embedded", handle)
call("rdc_shutdown")
"""


def _run(tools, tmp_path, groups, steps, **fake_env):
    """Runs one child; returns its results, one per step that is not "arm"."""
    bootstrap = _first_file(os.path.join(PREFIX, "lib", "librdc_bootstrap.so"))
    if bootstrap is None:
        _missing("librdc_bootstrap.so")
    env = dict(
        os.environ,
        LD_PRELOAD=tools[2],
        RDC_FAKE_GPUS=str(NUM_GPUS),
        RDC_FAKE_LOG=str(tmp_path / "fake_calls.log"),
        RDC_FAKE_ARM=str(tmp_path / "armed"),
    )
    env.update({k: str(v) for k, v in fake_env.items()})
    spec = {"groups": groups, "steps": steps}
    cmd = [sys.executable, "-c", _CHILD, bootstrap, json.dumps(spec)]
    try:
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=60 * FACTOR)
    except subprocess.TimeoutExpired as e:
        # The results come before the teardown, so show them if they came.
        out = e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else e.stdout or ""
        printed = [line for line in out.splitlines() if line.startswith('{"results"')]
        pytest.fail(f"embedded child hung (killed): {spec}\n{printed}")
    assert r.returncode == 0, f"embedded child exited {r.returncode}\n{r.stdout}{r.stderr[-3000:]}"
    # RDC logs to stdout too; the child's result is its one JSON line.
    lines = [line for line in r.stdout.splitlines() if line.startswith('{"results"')]
    assert len(lines) == 1, r.stdout
    results = json.loads(lines[0])["results"]
    for result in results:
        result["calls"] = [
            (parts[0], int(parts[1]), parts[2:] == ["!"])
            for parts in (line.split() for line in result["calls"])
            if parts[0] in CONFIG_APIS
        ]
    return results


def _gpus(calls, api):
    return [(gpu, failed) for name, gpu, failed in calls if name == api]


def test_an_unknown_setting_type_is_rejected(tools, tmp_path):
    """It used to be cached and reported as set, without reaching any GPU."""
    set_, get = _run(tools, tmp_path, {"one": [0]}, [["set", "one", 99, 300], ["get", "one"]])
    assert set_["status"] == BAD_PARAMETER
    assert set_["calls"] == []
    assert get["status"] == NOT_FOUND


def test_a_setting_for_an_empty_group_is_rejected(tools, tmp_path):
    """The status used to be read uninitialized: set, cached or failed by chance."""
    set_, get = _run(tools, tmp_path, {"none": []}, [["set", "none", POWER, 400], ["get", "none"]])
    assert set_["status"] == BAD_PARAMETER
    assert get["status"] == NOT_FOUND


def test_a_power_limit_too_large_for_microwatts_is_rejected(tools, tmp_path):
    """18446744074210 W used to wrap around to 500.448384 W, which the GPU took."""
    set_, get = _run(
        tools, tmp_path, {"one": [0]}, [["set", "one", POWER, 18446744074210], ["get", "one"]]
    )
    assert set_["status"] == BAD_PARAMETER
    assert _gpus(set_["calls"], "amdsmi_set_power_cap") == []
    assert get["status"] == NOT_FOUND


def test_clearing_a_clock_limit_with_an_empty_frequency_table(tools, tmp_path):
    """K5: amd-smi reports no frequency level. Clearing used to read the level
    before the first one, far outside the table, and crash."""
    set_, clear = _run(
        tools,
        tmp_path,
        {"one": [0]},
        [["set", "one", GFX, 1500], ["arm"], ["clear", "one"]],
        RDC_FAKE_NUM_FREQ=0,
    )
    assert set_["status"] == OK
    assert clear["status"] == OK
    assert _gpus(clear["calls"], "amdsmi_set_gpu_perf_level") == [(0, False)]


def test_a_set_that_a_gpu_rejects_puts_back_the_gpus_it_changed(tools, tmp_path):
    """GPU 2 rejects the limit: GPUs 0 and 1 go back to their default and GPU 3
    is left alone. GPUs 0 and 1 used to keep the new limit, which nothing
    cleared."""
    set_, get = _run(
        tools,
        tmp_path,
        {"all": list(range(NUM_GPUS))},
        [["arm"], ["set", "all", POWER, 400], ["get", "all"]],
        RDC_FAKE_FAIL="amdsmi_set_power_cap@2",
    )
    assert set_["status"] == UNKNOWN_ERROR
    sets = _gpus(set_["calls"], "amdsmi_set_power_cap")
    assert sets[:3] == [(0, False), (1, False), (2, True)], sets
    assert sorted(sets[3:]) == [(0, False), (1, False)], sets
    assert get["status"] == NOT_FOUND


def test_clear_resets_every_gpu_when_one_fails(tools, tmp_path):
    """GPU 1's reset fails once: the other GPUs are reset all the same, and the
    call reports the failure. Clearing used to stop at the failing GPU."""
    set_, clear, get = _run(
        tools,
        tmp_path,
        {"all": list(range(NUM_GPUS))},
        [["set", "all", POWER, 400], ["arm"], ["clear", "all"], ["get", "all"]],
        RDC_FAKE_FAIL_ONCE="amdsmi_set_power_cap@1",
    )
    assert set_["status"] == OK
    assert clear["status"] == UNKNOWN_ERROR
    assert _gpus(clear["calls"], "amdsmi_set_power_cap") == [
        (0, False),
        (1, True),
        (2, False),
        (3, False),
    ]
    assert get["status"] == NOT_FOUND
