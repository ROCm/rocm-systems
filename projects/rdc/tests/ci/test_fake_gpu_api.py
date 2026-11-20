"""RDC API calls against the fake amd-smi: a group of all GPUs when a GPU cannot be
added, the amd-smi version without a build string, and rdctst next to another
process named rdcd.

Runs like test_fake_gpu.py (same environment variables and helpers).
"""

import json
import os
import socket
import subprocess
import sys

import pytest

# The helpers and the `tools` fixture come from test_fake_gpu.py, not conftest.py:
# tests/ci/conftest.py is not installed with the suite (share/rdc/rdctst_tests/fake_gpu).
from test_fake_gpu import (  # noqa: F401 (tools is a fixture)
    FACTOR,
    NUM_GPUS,
    PREFIX,
    Rdcd,
    _first_file,
    _listening,
    _missing,
    tools,
)

# Creates a group of all GPUs (RDC_GROUP_DEFAULT) through embedded RDC once the
# fake's faults are armed (argv[2]), and prints the status and the group IDs
# before and after.
_ALL_GPU_GROUP = r"""
import ctypes, json, sys
lib = ctypes.CDLL(sys.argv[1])
def call(name, *args):
    st = getattr(lib, name)(*args)
    if st != 0:
        sys.exit(f"{name} returned {st}")
def group_ids():
    ids = (ctypes.c_uint32 * 64)()  # RDC_MAX_NUM_GROUPS
    count = ctypes.c_uint32(0)
    call("rdc_group_get_all_ids", handle, ids, ctypes.byref(count))
    return list(ids[: count.value])
handle, group = ctypes.c_void_p(), ctypes.c_uint32()
call("rdc_init", ctypes.c_uint64(0))
call("rdc_start_embedded", 0, ctypes.byref(handle))
before = group_ids()
open(sys.argv[2], "w").close()
status = lib.rdc_group_gpu_create(handle, 0, b"all_gpus", ctypes.byref(group))  # RDC_GROUP_DEFAULT
after = group_ids()
call("rdc_stop_embedded", handle)
call("rdc_shutdown")
print(json.dumps({"status": status, "before": before, "after": after}))
"""


@pytest.mark.parametrize("gpu", [1, NUM_GPUS - 1], ids=["a_middle_gpu", "the_last_gpu"])
def test_all_gpu_group_is_not_created_without_a_gpu(tools, tmp_path, gpu):
    bootstrap = _first_file(os.path.join(PREFIX, "lib", "librdc_bootstrap.so"))
    if bootstrap is None:
        _missing("librdc_bootstrap.so")
    armed = tmp_path / "armed"
    calls = tmp_path / "fake_calls.log"
    profile = "amdsmi_get_gpu_accelerator_partition_profile"
    env = dict(
        os.environ,
        LD_PRELOAD=tools[2],
        RDC_FAKE_GPUS=str(NUM_GPUS),
        RDC_FAKE_ARM=str(armed),
        RDC_FAKE_LOG=str(calls),
        # Adding a GPU to a group reads its partition profile.
        RDC_FAKE_FAIL_ONCE=f"{profile}@{gpu}",
    )
    try:
        r = subprocess.run(
            [sys.executable, "-c", _ALL_GPU_GROUP, bootstrap, str(armed)],
            env=env,
            capture_output=True,
            text=True,
            timeout=60 * FACTOR,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("group child hung (killed)")
    assert r.returncode == 0, f"group child exited {r.returncode}\n{r.stdout}{r.stderr[-3000:]}"
    failed = [line for line in calls.read_text().splitlines() if line.endswith(" !")]
    assert failed == [f"{profile} {gpu} !"], f"the fake failed {failed}"
    result = json.loads(r.stdout.strip().splitlines()[-1])
    assert result["status"] != 0, f"rdc_group_gpu_create() returned RDC_ST_OK without GPU {gpu}"
    assert result["after"] == result["before"], "the group was left behind"


def test_amdsmi_version_without_a_build_string(tools, tmp_path):
    # amdsmi_get_lib_version() reports 27.0.0 with a NULL build string, which
    # amd-smi never does.
    d = Rdcd(tools, tmp_path, RDC_FAKE_VERSION_BUILD_NULL=1).start()
    try:
        out = d.ok("discovery", "-v")
    finally:
        d.stop()
    assert "AMDSMI Library : 27.0.0" in out, out


# rdctst's standalone tests connect to localhost:50051 (tests/rdc_tests/functional).
RDCTST_PORT = 50051

# A process named rdcd that rdctst did not start.
_DECOY = r"""
import ctypes, time
ctypes.CDLL(None).prctl(15, b"rdcd", 0, 0, 0)  # PR_SET_NAME
print("ready", flush=True)
time.sleep(600)
"""


def _rdctst():
    rdctst = _first_file(
        os.path.join(PREFIX, "share", "rdc", "rdctst_tests", "rdctst"),
        os.path.join(PREFIX, "share", "rdc", "rdctst"),
    )
    if rdctst is None:
        _missing("rdctst")
    if _listening(RDCTST_PORT):
        pytest.skip(f"port {RDCTST_PORT}, which rdctst's own rdcd needs, is in use")
    return rdctst


def test_rdctst_leaves_another_rdcd_alone(tools):
    rdctst = _rdctst()
    rdcd, _, fake = tools
    # Started first, the decoy comes before rdctst's rdcd in /proc.
    decoy = subprocess.Popen([sys.executable, "-c", _DECOY], stdout=subprocess.PIPE, text=True)
    try:
        assert decoy.stdout.readline().strip() == "ready"
        with open(f"/proc/{decoy.pid}/comm") as comm:
            assert comm.read().strip() == "rdcd"
        env = dict(os.environ, LD_PRELOAD=fake, RDC_FAKE_GPUS=str(NUM_GPUS))
        try:
            r = subprocess.run(
                [rdctst, "-b", f"--start_rdcd={rdcd}"],
                env=env,
                capture_output=True,
                text=True,
                timeout=300 * FACTOR,
            )
        except subprocess.TimeoutExpired:
            pytest.fail("rdctst hung (killed)")
        assert r.returncode == 0, (
            f"rdctst exited {r.returncode}\n{r.stdout[-3000:]}{r.stderr[-2000:]}"
        )
        assert decoy.poll() is None, "rdctst stopped a process named rdcd that it did not start"
        assert not _listening(RDCTST_PORT), "rdctst left its rdcd running"
    finally:
        decoy.kill()
        decoy.wait()


def test_rdctst_does_not_start_rdcd_on_a_busy_port(tools):
    rdctst = _rdctst()
    rdcd, _, fake = tools
    env = dict(os.environ, LD_PRELOAD=fake, RDC_FAKE_GPUS=str(NUM_GPUS))
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as other:
        other.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        other.bind(("127.0.0.1", RDCTST_PORT))
        other.listen()
        try:
            r = subprocess.run(
                [rdctst, "-b", f"--start_rdcd={rdcd}"],
                env=env,
                capture_output=True,
                text=True,
                timeout=120 * FACTOR,
            )
        except subprocess.TimeoutExpired:
            pytest.fail("rdctst hung (killed)")
    assert r.returncode != 0, "rdctst passed without an rdcd of its own"
    assert f"localhost:{RDCTST_PORT} is in use" in r.stdout, r.stdout[-3000:]
