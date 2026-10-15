"""rdci and the RDC client library against rdcd with the fake amd-smi, at RDC's limits.

Runs like test_fake_gpu.py (same environment variables and helpers). Also runs
rdc_client_replies_tests, the gtest of the client against replies that state more
entries than they carry, from the tested prefix.
"""

import os
import re
import subprocess

import pytest

# The helpers and the `rdcd` fixture come from test_fake_gpu.py, not conftest.py:
# tests/ci/conftest.py is not installed with the suite (share/rdc/rdctst_tests/fake_gpu).
from test_fake_gpu import FACTOR, PREFIX, REQUIRE, _created_id, rdcd, tools  # noqa: F401

RDC_MAX_NUM_GROUPS = 64  # include/rdc/rdc.h


def test_client_replies_gtest():
    binary = os.path.join(PREFIX, "share", "rdc", "rdctst_tests", "rdc_client_replies_tests")
    if not os.path.isfile(binary):
        if REQUIRE:
            pytest.fail(f"{binary} is missing")
        pytest.skip(f"{binary} is missing")
    r = subprocess.run([binary], capture_output=True, text=True, timeout=120 * FACTOR)
    assert r.returncode == 0, (
        f"rdc_client_replies_tests exited with {r.returncode}\n{r.stdout[-3000:]}{r.stderr[-2000:]}"
    )


def test_rdci_lists_as_many_groups_as_rdcd_holds(rdcd):
    for i in range(RDC_MAX_NUM_GROUPS):
        _created_id(rdcd.ok("group", "-c", f"limit_{i}"))
    r = rdcd.rdci("group", "-c", "one_too_many")
    assert r.returncode != 0, f"rdcd took group {RDC_MAX_NUM_GROUPS + 1}\n{r.stdout}"
    out = rdcd.ok("group", "-l")
    listed = set(re.findall(r"\blimit_(\d+)\b", out))
    assert listed == {str(i) for i in range(RDC_MAX_NUM_GROUPS)}, out
