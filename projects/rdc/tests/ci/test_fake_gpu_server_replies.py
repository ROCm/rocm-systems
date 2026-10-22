"""rdcd's replies and the RDC C API's argument checks, from the tested prefix.

Runs the gtests rdc_server_replies_tests (rdcd's gRPC handlers against a library
that writes nothing, or every byte, of their outputs) and rdc_api_checks_tests
(the C API's job ID and name checks, and the status and data of cached values).
Runs like test_fake_gpu.py (same environment variables); needs no rdcd.
"""

import os
import subprocess

import pytest

# The helpers come from test_fake_gpu.py, not conftest.py: tests/ci/conftest.py
# is not installed with the suite (share/rdc/rdctst_tests/fake_gpu).
from test_fake_gpu import FACTOR, PREFIX, REQUIRE


@pytest.mark.parametrize("name", ["rdc_server_replies_tests", "rdc_api_checks_tests"])
def test_installed_gtest(name):
    binary = os.path.join(PREFIX, "share", "rdc", "rdctst_tests", name)
    if not os.path.isfile(binary):
        if REQUIRE:
            pytest.fail(f"{binary} is missing")
        pytest.skip(f"{binary} is missing")
    r = subprocess.run([binary], capture_output=True, text=True, timeout=120 * FACTOR)
    assert r.returncode == 0, (
        f"{name} exited with {r.returncode}\n{r.stdout[-3000:]}{r.stderr[-2000:]}"
    )
