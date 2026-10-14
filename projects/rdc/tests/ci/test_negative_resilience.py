"""
4.4  Negative / Error / Resilience Tests
=========================================
Validates error handling: missing certs, port blocked, daemon crash recovery,
unsupported fields, unauthorized operations, and that rdc_shutdown() keeps the
client library loaded.
"""

import os
import struct
import subprocess
import sys
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
# GPU groups and field groups that do not exist (requires daemon + GPU)
# ---------------------------------------------------------------------------
# Values of rdc_status_t in include/rdc/rdc.h.
RDC_ST_GROUP_NOT_FOUND = 17
RDC_ST_FLDGROUP_NOT_FOUND = 18
MISSING_ID = "4242"


class TestMissingGroups:
    """Verify rdci names the GPU group or field group it cannot find."""

    def test_dmon_missing_group(self, rdci_path, rdcd_server):
        r = run_rdci(rdci_path, "dmon", "-g", MISSING_ID, "-f", MISSING_ID, "-c", "1")
        assert r.returncode == RDC_ST_GROUP_NOT_FOUND, r.stdout
        assert f"Cannot find the group {MISSING_ID}" in r.stdout

    def test_dmon_missing_field_group(self, rdci_path, rdcd_server):
        r = run_rdci(rdci_path, "dmon", "-i", "0", "-f", MISSING_ID, "-c", "1")
        assert r.returncode == RDC_ST_FLDGROUP_NOT_FOUND, r.stdout
        assert f"Cannot find the field group {MISSING_ID}" in r.stdout

    def test_health_set_missing_group(self, rdci_path, rdcd_server):
        r = run_rdci(rdci_path, "health", "-g", MISSING_ID, "-s", "pm")
        assert r.returncode == RDC_ST_GROUP_NOT_FOUND, r.stdout
        assert f"Cannot find the group {MISSING_ID}" in r.stdout


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


# ---------------------------------------------------------------------------
# Client library lifetime (no GPU or daemon needed)
# ---------------------------------------------------------------------------
_PT_DYNAMIC = 2
_DT_NULL = 0
_DT_FLAGS_1 = 0x6FFFFFFB
_DF_1_NODELETE = 0x8

# Runs two connect / failing RPC / disconnect / rdc_shutdown() cycles (the RPC
# starts gRPC's threads) and reports whether librdc_client.so is still mapped.
_SHUTDOWN_PROBE = """
import ctypes, os, sys
P = ctypes.c_void_p
lib_dir = sys.argv[1]
client = os.path.realpath(os.path.join(lib_dir, "librdc_client.so"))


def mapped():
    with open("/proc/self/maps") as maps:
        return any(line.rstrip().endswith(" " + client) for line in maps)


rdc = ctypes.CDLL(os.path.join(lib_dir, "librdc_bootstrap.so"))
rdc.rdc_init.argtypes = [ctypes.c_uint64]
rdc.rdc_connect.argtypes = [ctypes.c_char_p, ctypes.POINTER(P)] + [ctypes.c_char_p] * 3
rdc.rdc_device_get_all.argtypes = [P, P, P]
rdc.rdc_disconnect.argtypes = [P]
rdc.rdc_shutdown.argtypes = []
for _ in range(2):
    handle = P()
    assert rdc.rdc_init(0) == 0
    assert rdc.rdc_connect(b"localhost:1", ctypes.byref(handle), None, None, None) == 0
    assert mapped(), "rdc_connect() did not load " + client
    ids, count = (ctypes.c_uint32 * 128)(), ctypes.c_uint32()
    rdc.rdc_device_get_all(handle, ids, ctypes.byref(count))  # nothing listens on port 1
    assert rdc.rdc_disconnect(handle) == 0
    assert rdc.rdc_shutdown() == 0
print("resident" if mapped() else "unloaded")
"""


def _dt_flags_1(path):
    """Return DT_FLAGS_1 of a 64-bit little-endian ELF shared object, or 0 if absent."""
    with open(path, "rb") as f:
        hdr = f.read(64)
        assert hdr[:6] == b"\x7fELF\x02\x01", f"{path} is not a 64-bit little-endian ELF file"
        (phoff,) = struct.unpack_from("<Q", hdr, 0x20)  # e_phoff
        phentsize, phnum = struct.unpack_from("<HH", hdr, 0x36)  # e_phentsize, e_phnum
        for i in range(phnum):
            f.seek(phoff + i * phentsize)
            # Elf64_Phdr up to p_filesz: p_type, p_flags, p_offset, p_vaddr, p_paddr, p_filesz
            p_type, _, p_offset, _, _, p_filesz = struct.unpack("<IIQQQQ", f.read(40))
            if p_type != _PT_DYNAMIC:
                continue
            f.seek(p_offset)
            dyn = f.read(p_filesz)
            for off in range(0, len(dyn) - 15, 16):
                tag, val = struct.unpack_from("<qQ", dyn, off)
                if tag == _DT_NULL:
                    break
                if tag == _DT_FLAGS_1:
                    return val
    return 0


class TestClientLibraryLifetime:
    """rdc_shutdown() dlclose()s librdc_client.so while gRPC threads still run.

    Unloading the library under those threads crashes rdci and other clients on
    exit, so the library must stay resident.
    """

    @pytest.fixture
    def client_lib_dir(self, rocm_dir):
        lib_dir = os.path.join(rocm_dir, "lib")
        if not os.path.isfile(os.path.join(lib_dir, "librdc_client.so")):
            pytest.skip("librdc_client.so not installed")
        return lib_dir

    def test_client_library_is_nodelete(self, client_lib_dir):
        # The only check that fails without the flag in GCC builds with shared gRPC,
        # where the gRPC libraries keep librdc_client.so loaded anyway.
        path = os.path.realpath(os.path.join(client_lib_dir, "librdc_client.so"))
        assert _dt_flags_1(path) & _DF_1_NODELETE, "librdc_client.so is not linked -z nodelete"

    def test_client_library_stays_loaded_after_shutdown(self, client_lib_dir):
        # Make the bootstrap's dlopen("librdc_client.so") resolve to the library under test.
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            filter(None, [client_lib_dir, env.get("LD_LIBRARY_PATH")])
        )
        r = subprocess.run(
            [sys.executable, "-c", _SHUTDOWN_PROBE, client_lib_dir],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert r.returncode == 0, f"probe failed (rc={r.returncode}): {r.stderr[-2000:]}"
        assert "resident" in r.stdout.splitlines(), "rdc_shutdown() unloaded librdc_client.so"
