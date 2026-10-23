"""How many GPUs and fields RDC handles at once, against the fake amd-smi.

Runs like test_fake_gpu.py (same environment variables and helpers). Also runs
rdc_telemetry_tests, the gtest of RdcTelemetryModule, from the tested prefix.
"""

import os
import re
import subprocess
import threading

import pytest

# The helpers and the `tools` fixture come from test_fake_gpu.py, not conftest.py:
# tests/ci/conftest.py is not installed with the suite (share/rdc/rdctst_tests/fake_gpu).
from test_fake_gpu import FACTOR, PREFIX, REQUIRE, Rdcd, _created_id, tools  # noqa: F401

RDC_MAX_NUM_DEVICES = 128  # include/rdc/rdc.h
# 128 fields that RDC reads from amd-smi (rdci dmon -l), so that with 64 GPUs each watch
# hands 8,192 GPU-field pairs to the same library. Fields of another library would split them.
FIELDS_128 = (
    list(range(1, 11))
    + list(range(100, 107))
    + [200, 201, 202, 300, 301]
    + list(range(400, 410))
    + list(range(500, 522))
    + [550]
    + list(range(600, 660))
    + [680]
    + list(range(700, 712))
)
assert len(FIELDS_128) == 128


def test_telemetry_module_gtest():
    binary = os.path.join(PREFIX, "share", "rdc", "rdctst_tests", "rdc_telemetry_tests")
    if not os.path.isfile(binary):
        if REQUIRE:
            pytest.fail(f"{binary} is missing")
        pytest.skip(f"{binary} is missing")
    r = subprocess.run([binary], capture_output=True, text=True, timeout=120 * FACTOR)
    assert r.returncode == 0, (
        f"rdc_telemetry_tests exited with {r.returncode}\n{r.stdout[-3000:]}{r.stderr[-2000:]}"
    )


def test_more_gpus_and_partitions_than_rdc_lists(tools, tmp_path):
    # 256 GPU partitions; RDC lists RDC_MAX_NUM_DEVICES of them. rdcd wrote all 256 into
    # an array of 128 for discovery and for a job's start.
    d = Rdcd(tools, tmp_path, gpus=128, RDC_FAKE_PARTITIONS=2).start()
    try:
        out = d.ok("discovery", "-l")
        assert f"{RDC_MAX_NUM_DEVICES} GPUs found" in out, out
        # Logged by the handler, or first by a GPU table that keeps only the first
        # RDC_MAX_NUM_DEVICES, so that the handler never sees more.
        logged = (
            f"RDC lists {RDC_MAX_NUM_DEVICES} of the 256 GPUs and GPU partitions",
            f"AMD SMI lists more than {RDC_MAX_NUM_DEVICES} GPUs; RDC uses the first of them",
        )
        assert any(line in d.output() for line in logged), d.output()
        listed = re.findall(r"^(\d+)\s+Fake ", out, re.M)
        assert listed == [str(i) for i in range(RDC_MAX_NUM_DEVICES)], out
        gid = _created_id(d.ok("group", "-c", "big_job"))
        d.ok("group", "-g", gid, "-a", "0,1")
        d.ok("stats", "-s", "big_job_1", "-g", gid)
        d.ok("stats", "-x", "big_job_1")
    finally:
        code = d.stop()
    assert code == 0, f"rdcd exited with {code}\n{d.output()}"


def test_two_watches_of_8192_fields(tools, tmp_path):
    # Both watches due in one update hand 16,384 fields to the amd-smi library at once.
    # rdcd crashed in some runs, when the two came due together.
    d = Rdcd(tools, tmp_path, gpus=128).start()
    try:
        groups = []
        for first in (0, 64):
            gid = _created_id(d.ok("group", "-c", f"half_{first}"))
            d.ok("group", "-g", gid, "-a", ",".join(str(g) for g in range(first, first + 64)))
            groups.append(gid)
        fields = ",".join(str(f) for f in FIELDS_128)
        fgid = _created_id(d.ok("fieldgroup", "-c", "fields_128", "-f", fields))

        results = {}
        start = threading.Barrier(len(groups), timeout=60 * FACTOR)

        def dmon(gid):
            try:
                start.wait()
                results[gid] = d.rdci("dmon", "-g", gid, "-f", fgid, "-c", "8", "-d", "100")
            except BaseException as e:  # noqa: BLE001 - reported below, in the test's thread
                results[gid] = e

        threads = [threading.Thread(target=dmon, args=(gid,)) for gid in groups]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for gid in groups:
            r = results[gid]
            if isinstance(r, BaseException):
                raise r
            assert r.returncode == 0, f"dmon on group {gid}: {r.returncode}\n{r.stdout}{r.stderr}"
        d.check_alive("two dmon watches")
    finally:
        code = d.stop()
    assert code == 0, f"rdcd exited with {code}\n{d.output()}"
