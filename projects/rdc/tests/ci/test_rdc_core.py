"""
4.1  RDC Core / CLI Tests
=========================
Validates rdcd daemon operation and rdci CLI commands including discovery,
group management, field groups, dmon, job statistics, and diagnostics.
"""

import os
import subprocess
import time
import pytest

from conftest import run_rdci


# ---------------------------------------------------------------------------
# Installation verification (runs without GPU)
# ---------------------------------------------------------------------------
class TestInstallation:
    """Verify RDC binaries and libraries are installed correctly."""

    def test_rdcd_exists(self, rdcd_path):
        assert os.path.isfile(rdcd_path)
        assert os.access(rdcd_path, os.X_OK)

    def test_rdci_exists(self, rdci_path):
        assert os.path.isfile(rdci_path)
        assert os.access(rdci_path, os.X_OK)

    def test_rdctst_exists(self, rdctst_path):
        assert os.path.isfile(rdctst_path)
        assert os.access(rdctst_path, os.X_OK)

    def test_core_libraries_present(self, rocm_dir):
        libs = ["librdc.so", "librdc_bootstrap.so", "librdc_client.so"]
        for lib in libs:
            path = os.path.join(rocm_dir, "lib", lib)
            assert os.path.exists(path), f"Missing library: {lib}"

    def test_module_libraries_present(self, rocm_dir):
        """Check optional module libraries (non-fatal if missing)."""
        optional = ["librdc_rocr.so", "librdc_rocp.so"]
        for lib in optional:
            path = os.path.join(rocm_dir, "lib", "rdc", lib)
            if not os.path.exists(path):
                pytest.skip(f"Optional library {lib} not present")


# ---------------------------------------------------------------------------
# CLI help / usage (runs without GPU or daemon)
# ---------------------------------------------------------------------------
class TestCLIUsage:
    """Verify rdci prints help and rejects invalid input gracefully."""

    @pytest.mark.parametrize("flag", ["--help", "-h"])
    def test_rdci_help(self, rdci_path, flag):
        result = subprocess.run([rdci_path, flag], capture_output=True, text=True, timeout=10)
        combined = (result.stdout + result.stderr).lower()
        assert result.returncode == 0
        assert "usage" in combined or "subsystem" in combined

    @pytest.mark.parametrize(
        "subcmd",
        [
            "discovery",
            "group",
            "fieldgroup",
            "dmon",
            "stats",
            "diag",
            "config",
            "policy",
            "health",
            "topo",
            "link",
        ],
    )
    def test_subcmd_help(self, rdci_path, subcmd):
        result = subprocess.run(
            [rdci_path, subcmd, "--help"], capture_output=True, text=True, timeout=10
        )
        combined = result.stdout + result.stderr
        assert len(combined) > 0, f"No output from rdci {subcmd} --help"

    def test_invalid_subcmd(self, rdci_path):
        result = subprocess.run(
            [rdci_path, "nonexistent_command"], capture_output=True, text=True, timeout=10
        )
        combined = (result.stdout + result.stderr).lower()
        assert result.returncode != 0
        assert "unknown subsystem" in combined or "usage" in combined

    def test_invalid_args_to_group(self, rdci_path):
        result = subprocess.run(
            [rdci_path, "group", "--invalid-flag-xyz"], capture_output=True, text=True, timeout=10
        )
        assert result.returncode != 0


# ---------------------------------------------------------------------------
# Daemon startup (requires GPU)
# ---------------------------------------------------------------------------
class TestDaemonStartup:
    """Verify rdcd daemon can be started and responds."""

    def test_daemon_running(self, rdcd_server):
        assert rdcd_server.poll() is None, "rdcd process exited unexpectedly"

    def test_daemon_port_open(self, rdcd_server):
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(5)
            result = s.connect_ex(("localhost", 50051))
            assert result == 0, "rdcd not listening on port 50051"


# ---------------------------------------------------------------------------
# Discovery (requires GPU + daemon)
# ---------------------------------------------------------------------------
class TestDiscovery:
    """Verify GPU discovery via rdci."""

    def test_discovery_lists_gpus(self, rdci_path, rdcd_server):
        r = run_rdci(rdci_path, "discovery", "-l")
        assert r.returncode == 0, f"rdci discovery failed: {r.stderr}"
        lines = r.stdout.strip().splitlines()
        assert len(lines) >= 1, "No GPUs discovered"

    def test_discovery_shows_attributes(self, rdci_path, rdcd_server):
        r = run_rdci(rdci_path, "discovery", "-i")
        assert r.returncode == 0, f"rdci discovery -i failed: {r.stderr}"


# ---------------------------------------------------------------------------
# GPU Group CRUD (requires GPU + daemon)
# ---------------------------------------------------------------------------
class TestGPUGroup:
    """Verify GPU group create / add / list / delete lifecycle."""

    def test_group_create_list_delete(self, rdci_path, rdcd_server):
        r = run_rdci(rdci_path, "group", "-c", "ci_test_group")
        assert r.returncode == 0, f"group create failed: {r.stderr}"

        r = run_rdci(rdci_path, "group", "-l")
        assert r.returncode == 0
        assert "ci_test_group" in r.stdout

        r = run_rdci(rdci_path, "group", "-d", "ci_test_group")
        assert r.returncode == 0 or "not found" in r.stderr.lower()

    def test_group_add_gpu(self, rdci_path, rdcd_server):
        run_rdci(rdci_path, "group", "-c", "ci_add_test")
        r = run_rdci(rdci_path, "group", "-a", "ci_add_test", "-g", "0")
        assert r.returncode == 0, f"group add GPU failed: {r.stderr}"
        run_rdci(rdci_path, "group", "-d", "ci_add_test")


# ---------------------------------------------------------------------------
# Field Group CRUD (requires GPU + daemon)
# ---------------------------------------------------------------------------
class TestFieldGroup:
    """Verify field group create / list / delete."""

    def test_fieldgroup_lifecycle(self, rdci_path, rdcd_server):
        r = run_rdci(rdci_path, "fieldgroup", "-c", "ci_fg", "-f", "100,101,140")
        assert r.returncode == 0, f"fieldgroup create failed: {r.stderr}"

        r = run_rdci(rdci_path, "fieldgroup", "-l")
        assert r.returncode == 0
        assert "ci_fg" in r.stdout

        r = run_rdci(rdci_path, "fieldgroup", "-d", "ci_fg")
        assert r.returncode == 0 or "not found" in r.stderr.lower()


# ---------------------------------------------------------------------------
# dmon (requires GPU + daemon)
# ---------------------------------------------------------------------------
class TestDmon:
    """Verify dmon metric streaming."""

    def test_dmon_short_run(self, rdci_path, rdcd_server):
        r = run_rdci(rdci_path, "dmon", "-f", "100,140", "-c", "3", "-d", "1000", timeout=30)
        assert r.returncode == 0, f"dmon failed: {r.stderr}"
        assert len(r.stdout.strip()) > 0, "dmon produced no output"


# ---------------------------------------------------------------------------
# Job Statistics (requires GPU + daemon)
# ---------------------------------------------------------------------------
class TestJobStats:
    """Verify job stats start / get / stop / remove."""

    def test_job_stats_lifecycle(self, rdci_path, rdcd_server):
        job_id = "ci_test_job_001"

        r = run_rdci(rdci_path, "stats", "-s", job_id)
        assert r.returncode == 0, f"stats start failed: {r.stderr}"

        time.sleep(2)

        r = run_rdci(rdci_path, "stats", "-j", job_id)
        assert r.returncode == 0, f"stats get failed: {r.stderr}"

        r = run_rdci(rdci_path, "stats", "-x", job_id)
        assert r.returncode == 0, f"stats stop failed: {r.stderr}"

        r = run_rdci(rdci_path, "stats", "-r", job_id)
        assert r.returncode == 0 or "not found" in r.stderr.lower()


# ---------------------------------------------------------------------------
# Diagnostics (requires GPU + daemon)
# ---------------------------------------------------------------------------
class TestDiagnostics:
    """Verify diagnostic runs produce results."""

    def test_diag_summary(self, rdci_path, rdcd_server):
        r = run_rdci(rdci_path, "diag", timeout=120)
        combined = r.stdout + r.stderr
        assert len(combined) > 0, "diag produced no output"


# ---------------------------------------------------------------------------
# Invalid commands / negative CLI tests (no daemon needed)
# ---------------------------------------------------------------------------
class TestNegativeCLI:
    """Verify graceful error handling for invalid CLI input."""

    def test_dmon_no_daemon(self, rdci_path):
        r = subprocess.run(
            [rdci_path, "dmon", "-u", "-f", "100", "-c", "1", "--host", "localhost:59999"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert r.returncode != -11, "rdci segfaulted (SIGSEGV)"
        assert r.returncode != 0, "Should fail when daemon is not running"

    def test_stats_no_daemon(self, rdci_path):
        r = subprocess.run(
            [rdci_path, "stats", "-u", "-j", "nonexistent", "--host", "localhost:59999"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert r.returncode != -11, "rdci segfaulted"
        assert r.returncode != 0, "Should fail when daemon is not running"
