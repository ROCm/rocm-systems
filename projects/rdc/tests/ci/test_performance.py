"""
4.5  Performance / Scalability Tests
=====================================
Validates high polling rates, multi-GPU handling, and long-running stats.
"""

import time
import pytest

from conftest import run_rdci


pytestmark = pytest.mark.performance


class TestHighPollingRate:
    """Verify dmon stability at aggressive polling intervals."""

    def test_dmon_fast_polling(self, rdci_path, rdcd_server):
        r = run_rdci(rdci_path, "dmon", "-f", "100,101,140", "-c", "10", "-d", "200", timeout=30)
        assert r.returncode == 0, f"Fast polling dmon failed: {r.stderr}"
        lines = [l for l in r.stdout.strip().splitlines() if l.strip()]
        assert len(lines) >= 1, "No output from fast polling dmon"


class TestMultiGPUStress:
    """Stress test covering all GPUs."""

    def test_dmon_all_gpus(self, rdci_path, rdcd_server):
        r = run_rdci(rdci_path, "dmon", "-f", "100,140,155", "-c", "5", "-d", "1000", timeout=30)
        assert r.returncode == 0, f"Multi-GPU dmon failed: {r.stderr}"


class TestLongJobStats:
    """Verify job stats remain consistent over a longer period."""

    def test_sustained_job_stats(self, rdci_path, rdcd_server):
        job_id = "ci_perf_job_001"

        r = run_rdci(rdci_path, "stats", "-s", job_id)
        assert r.returncode == 0, f"stats start failed: {r.stderr}"

        time.sleep(5)

        r = run_rdci(rdci_path, "stats", "-j", job_id)
        assert r.returncode == 0, f"stats get failed: {r.stderr}"
        assert len(r.stdout.strip()) > 0, "No stats output after 5 s"

        run_rdci(rdci_path, "stats", "-x", job_id)
        run_rdci(rdci_path, "stats", "-r", job_id)
