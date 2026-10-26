"""
4.5  Performance / Scalability Tests
=====================================
Validates high polling rates, multi-GPU handling, and long-running stats.
"""

import time
import pytest

from conftest import create_gpu_group, gpu_indexes, run_rdci


pytestmark = pytest.mark.performance


class TestHighPollingRate:
    """Verify dmon stability at aggressive polling intervals."""

    def test_dmon_fast_polling(self, rdci_path, rdcd_server):
        gpu = gpu_indexes(rdci_path)[0]
        r = run_rdci(
            rdci_path, "dmon", "-e", "100,101,201", "-i", gpu, "-c", "10", "-d", "200", timeout=30
        )
        assert r.returncode == 0, f"Fast polling dmon failed: {r.stdout}"
        samples = [line for line in r.stdout.splitlines() if line.split()[:1] == [gpu]]
        assert len(samples) == 10, f"expected 10 samples for GPU {gpu}:\n{r.stdout}"


class TestMultiGPUStress:
    """Stress test covering all GPUs."""

    def test_dmon_all_gpus(self, rdci_path, rdcd_server):
        gpus = gpu_indexes(rdci_path)
        r = run_rdci(
            rdci_path,
            "dmon",
            "-e",
            "100,201,300",
            "-i",
            ",".join(gpus),
            "-c",
            "5",
            "-d",
            "1000",
            timeout=30,
        )
        assert r.returncode == 0, f"Multi-GPU dmon failed: {r.stdout}"
        for gpu in gpus:
            samples = [line for line in r.stdout.splitlines() if line.split()[:1] == [gpu]]
            assert len(samples) == 5, f"expected 5 samples for GPU {gpu}:\n{r.stdout}"


class TestLongJobStats:
    """Verify job stats remain consistent over a longer period."""

    def test_sustained_job_stats(self, rdci_path, rdcd_server):
        job_id = "ci_perf_job_001"
        gid = create_gpu_group(rdci_path, "ci_perf_group", [gpu_indexes(rdci_path)[0]])
        try:
            r = run_rdci(rdci_path, "stats", "-s", job_id, "-g", gid)
            assert r.returncode == 0, f"stats start failed: {r.stdout}"

            time.sleep(5)

            r = run_rdci(rdci_path, "stats", "-j", job_id)
            assert r.returncode == 0, f"stats get failed: {r.stdout}"
            assert len(r.stdout.strip()) > 0, "No stats output after 5 s"
        finally:
            run_rdci(rdci_path, "stats", "-x", job_id)
            run_rdci(rdci_path, "stats", "-r", job_id)
            run_rdci(rdci_path, "group", "-d", gid)
