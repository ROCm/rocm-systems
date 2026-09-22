"""
4.4  Negative / Error / Resilience Tests
=========================================
Validates error handling: missing certs, port blocked, daemon crash recovery,
unsupported fields, and unauthorized operations.
"""

import os
import subprocess
import time

import pytest

from conftest import run_rdci, terminate_process


# ---------------------------------------------------------------------------
# Authentication / TLS error paths
# ---------------------------------------------------------------------------
class TestAuthErrors:
    """Verify TLS/cert error handling."""

    def test_missing_certs_connection_fails(self, rdci_path):
        r = subprocess.run(
            [rdci_path, "discovery", "-l", "--host", "localhost:59999"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert r.returncode != -11, "rdci segfaulted (SIGSEGV)"
        assert r.returncode != 0, "Should fail without valid certs/daemon"

    def test_wrong_port_connection_refused(self, rdci_path):
        r = subprocess.run(
            [rdci_path, "discovery", "-u", "-l", "--host", "localhost:59999"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert r.returncode != 0
        assert r.returncode != -11, "rdci segfaulted"


# ---------------------------------------------------------------------------
# Client before daemon
# ---------------------------------------------------------------------------
class TestClientBeforeDaemon:
    """Verify client behavior when daemon is not running."""

    def test_rdci_dmon_no_daemon(self, rdci_path):
        r = subprocess.run(
            [rdci_path, "dmon", "-u", "-f", "100", "-c", "1", "--host", "localhost:59998"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert r.returncode != -11, "rdci segfaulted"
        assert r.returncode != 0, "Should fail without daemon"

    def test_rdci_stats_no_daemon(self, rdci_path):
        r = subprocess.run(
            [rdci_path, "stats", "-u", "-s", "no_daemon_job", "--host", "localhost:59998"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert r.returncode != -11, "rdci segfaulted"
        assert r.returncode != 0, "Should fail without daemon"

    def test_rdci_group_create_no_daemon(self, rdci_path):
        r = subprocess.run(
            [rdci_path, "group", "-u", "-c", "no_daemon_group", "--host", "localhost:59998"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert r.returncode != -11, "rdci segfaulted"
        assert r.returncode != 0, "Should fail without daemon"


# ---------------------------------------------------------------------------
# Unsupported / invalid metric fields (requires daemon + GPU)
# ---------------------------------------------------------------------------
class TestUnsupportedFields:
    """Verify graceful handling of unsupported or invalid fields."""

    def test_dmon_invalid_field_id(self, rdci_path, rdcd_server):
        r = run_rdci(rdci_path, "dmon", "-f", "99999", "-c", "1", "-d", "1000")
        assert r.returncode != -11, "rdci segfaulted on invalid field"

    def test_fieldgroup_empty_fields(self, rdci_path, rdcd_server):
        r = run_rdci(rdci_path, "fieldgroup", "-c", "ci_empty_fg", "-f", "")
        assert r.returncode != -11, "rdci segfaulted"


# ---------------------------------------------------------------------------
# Daemon restart resilience (requires GPU)
# ---------------------------------------------------------------------------
class TestDaemonResilience:
    """Verify behavior around daemon stop/restart scenarios."""

    def test_rdcd_responds_after_quick_restart(self, rdcd_path, has_gpu):
        if not has_gpu:
            pytest.skip("No GPU -- skipping resilience test")

        import socket

        def _port_open(port, timeout=2):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(timeout)
                return s.connect_ex(("localhost", port)) == 0

        port = 50099
        env = os.environ.copy()

        proc = None
        try:
            proc = subprocess.Popen(
                [rdcd_path, "-u", "-p", str(port)],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline and not _port_open(port):
                time.sleep(0.5)
            assert _port_open(port), "rdcd did not start on alternate port"

            terminate_process(proc)

            proc = subprocess.Popen(
                [rdcd_path, "-u", "-p", str(port)],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline and not _port_open(port):
                time.sleep(0.5)
            assert _port_open(port), "rdcd did not restart successfully"
        finally:
            terminate_process(proc)
