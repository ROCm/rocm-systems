"""Policies against the fake amd-smi: how often RDC checks them, and which GPUs.

Runs like test_fake_gpu.py (same environment variables and helpers).
"""

import re
import signal
import subprocess
import threading
import time

import pytest

# The helpers and the `tools` fixture come from test_fake_gpu.py, not conftest.py:
# tests/ci/conftest.py is not installed with the suite (share/rdc/rdctst_tests/fake_gpu).
from test_fake_gpu import FACTOR, Rdcd, _created_id, tools  # noqa: F401 (fixture)

RDC_ST_GROUP_NOT_FOUND = 17  # rdc_status_t in include/rdc/rdc.h; rdci exits with it
TEMP_READ = "amdsmi_get_temp_metric"  # one per GPU and check of a temperature policy
GPU_RESET = "amdsmi_reset_gpu"
WINDOW_S = 3


@pytest.fixture
def rdcd(tools, tmp_path):
    d = Rdcd(tools, tmp_path).start()
    yield d
    code = d.stop()
    assert code == 0, f"rdcd exited with {code} after SIGTERM\n{d.output()}"


def _group(d, gpus):
    gid = _created_id(d.ok("group", "-c", "policy_group"))
    d.ok("group", "-g", gid, "-a", ",".join(str(g) for g in gpus))
    return gid


def _gpus_called(d, api, since=0):
    return [gpu for name, gpu, _ in d.calls()[since:] if name == api]


def _wait_for_call(d, api, gpu):
    deadline = time.monotonic() + 5 * FACTOR
    while gpu not in _gpus_called(d, api):
        assert time.monotonic() < deadline, f"no {api} on GPU {gpu}"
        time.sleep(0.1)


def test_policy_checks_each_gpu_twice_a_second(rdcd):
    gid = _group(rdcd, [0, 1])
    rdcd.ok("policy", "-g", gid, "-T", "95", "-A", "0")  # the fake GPUs run at 30 to 33 C
    _wait_for_call(rdcd, TEMP_READ, 1)
    before, start = len(rdcd.calls()), time.monotonic()
    time.sleep(WINDOW_S)
    reads, elapsed = _gpus_called(rdcd, TEMP_READ, before), time.monotonic() - start
    # One check every 500 ms: about 2 reads per GPU and second. RDC used to check every 0.5 ms.
    for gpu in (0, 1):
        n = reads.count(gpu)
        assert elapsed <= n <= 4 * elapsed + 2, (
            f"GPU {gpu}: {n} temperature reads in {elapsed:.1f} s"
        )


def test_policy_stops_checking_a_deleted_group(rdcd):
    gid = _group(rdcd, [1])
    rdcd.ok("policy", "-g", gid, "-T", "1", "-A", "1")  # 1 C: always exceeded; reset the GPU
    _wait_for_call(rdcd, GPU_RESET, 1)
    rdcd.ok("group", "-d", gid)
    time.sleep(1)  # a check in progress may still finish
    before = len(rdcd.calls())
    time.sleep(2)
    for api in (TEMP_READ, GPU_RESET):
        assert _gpus_called(rdcd, api, before) == [], (
            f"the policy still calls {api} for a deleted group"
        )


def test_policy_set_on_a_missing_group_fails(rdcd):
    r = rdcd.rdci("policy", "-g", "4242", "-T", "95", "-A", "0")
    assert r.returncode == RDC_ST_GROUP_NOT_FOUND, (
        f"want {RDC_ST_GROUP_NOT_FOUND}, got {r.returncode}\n{r.stdout}{r.stderr}"
    )


def test_policy_get_lists_every_condition_of_a_group(rdcd):
    gid = _group(rdcd, [0])
    for option, value in (("-T", "95"), ("-P", "500"), ("-M", "10")):
        rdcd.ok("policy", "-g", gid, option, value, "-A", "0")
    out = rdcd.ok("policy", "-g", gid, "--get")
    for label, value in (("Temperature Limit", 95), ("Power Limit", 500), ("Page Retirement", 10)):
        assert re.search(rf"{label}\s*\|\s*{value}\s*\|", out), (label, out)


def test_policy_notification_reaches_a_registered_rdci(rdcd):
    gid = _group(rdcd, [0])
    rdcd.ok("policy", "-g", gid, "-T", "1", "-A", "0")  # 1 C: always exceeded
    expected = "A Temperature Limit exceeds the threshold 1 with the value"
    cmd = [rdcd.rdci_path, "policy", "--reg", "-g", gid, "-u", "--host", f"127.0.0.1:{rdcd.port}"]
    proc = subprocess.Popen(
        cmd, env=rdcd.rdci_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    lines, seen = [], threading.Event()

    def read():
        for line in proc.stdout:
            lines.append(line)
            if expected in line:
                seen.set()

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    try:
        seen.wait(10 * FACTOR)
        proc.send_signal(signal.SIGINT)  # rdci unregisters and exits
        proc.wait(timeout=20 * FACTOR)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        reader.join(5)
    out = "".join(lines)
    assert seen.is_set(), out
    assert proc.returncode == 0, f"rdci exited with {proc.returncode}\n{out}"
