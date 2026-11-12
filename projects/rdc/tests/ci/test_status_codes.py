"""
Statuses for a GPU group that does not exist
============================================
Runs the GPU-less rdc_status_tests binary, and checks that rdci health names a
GPU group that does not exist.
"""

import os
import subprocess

import pytest

from conftest import run_rdci

# Value of RDC_ST_GROUP_NOT_FOUND in rdc_status_t (include/rdc/rdc.h).
RDC_ST_GROUP_NOT_FOUND = 17
MISSING_GROUP = "4243"


def test_rdc_status_tests(rocm_dir, rdctst_path):
    # rdctst_path skips this test unless the RDC test package is installed.
    binary = os.path.join(rocm_dir, "share", "rdc", "rdctst_tests", "rdc_status_tests")
    assert os.path.isfile(binary), f"{binary} is missing from the RDC test package"

    result = subprocess.run([binary], capture_output=True, text=True, timeout=120)

    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    assert result.returncode == 0, (
        f"rdc_status_tests failed (rc={result.returncode}):\n"
        f"{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
    )


class TestHealthMissingGroup:
    """Verify rdci health -f and -c name a GPU group that does not exist (needs GPU)."""

    @pytest.mark.parametrize("flag", ["-f", "-c"])
    def test_health_missing_group(self, rdci_path, rdcd_server, flag):
        r = run_rdci(rdci_path, "health", "-g", MISSING_GROUP, flag)
        assert r.returncode == RDC_ST_GROUP_NOT_FOUND, r.stdout
        assert f"Cannot find the group {MISSING_GROUP}" in r.stdout
