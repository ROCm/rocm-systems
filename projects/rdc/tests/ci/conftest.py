"""Shared pytest fixtures and helpers for RDC CI tests."""

import os
import re
import signal
import shutil
import subprocess
import time

import pytest


def _find_binary(name):
    """Locate an RDC binary, checking ROCM_DIR first, then PATH."""
    rocm = os.environ.get("ROCM_DIR", "/opt/rocm")
    candidate = os.path.join(rocm, "bin", name)
    if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
        return candidate
    found = shutil.which(name)
    if found:
        return found
    return None


@pytest.fixture(scope="session")
def rocm_dir():
    return os.environ.get("ROCM_DIR", "/opt/rocm")


@pytest.fixture(scope="session")
def rdcd_path(rocm_dir):
    path = _find_binary("rdcd")
    if path is None:
        pytest.skip("rdcd binary not found")
    return path


@pytest.fixture(scope="session")
def rdci_path(rocm_dir):
    path = _find_binary("rdci")
    if path is None:
        pytest.skip("rdci binary not found")
    return path


@pytest.fixture(scope="session")
def rdctst_path(rocm_dir):
    """Locate the rdctst GoogleTest binary."""
    candidates = [
        os.path.join(rocm_dir, "share", "rdc", "rdctst"),
        os.path.join(rocm_dir, "share", "rdc", "rdctst_tests", "rdctst"),
    ]
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    path = _find_binary("rdctst")
    if path is None:
        pytest.skip("rdctst binary not found")
    return path


@pytest.fixture(scope="session")
def has_gpu():
    """Return True if at least one AMD GPU is accessible."""
    try:
        if not os.access("/dev/kfd", os.R_OK | os.W_OK):
            return False

        drm_dir = "/sys/class/drm"
        device_dir = "/dev/dri"
        if not any(
            entry.startswith("renderD")  # codespell:ignore renderd
            and os.access(os.path.join(device_dir, entry), os.R_OK | os.W_OK)
            for entry in os.listdir(device_dir)
        ):
            return False

        for entry in os.listdir(drm_dir):
            suffix = entry.removeprefix("card")
            if not entry.startswith("card") or not suffix.isdigit():
                continue
            vendor_path = os.path.join(drm_dir, entry, "device", "vendor")
            if os.path.isfile(vendor_path):
                with open(vendor_path) as f:
                    if f.read().strip().lower() == "0x1002":
                        return True
    except Exception:
        pass
    return False


def _wait_for_port(port, timeout=10):
    """Block until something is listening on *port* or timeout expires."""
    import socket

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            if s.connect_ex(("localhost", port)) == 0:
                return True
        time.sleep(0.3)
    return False


def terminate_process(proc, timeout=10):
    """Terminate a child process without masking an earlier test failure."""
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.send_signal(signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        if proc.poll() is None:
            try:
                proc.kill()
            except ProcessLookupError:
                return
        proc.wait()


class RdcdServer:
    """The session rdcd. Only one rdcd can run per host (rdcd.lock)."""

    def __init__(self, rdcd_path):
        self.rdcd_path = rdcd_path
        self.proc = None

    def start(self):
        """Start rdcd on port 50051; return False if it is not listening within 15 s."""
        self.proc = subprocess.Popen(
            [self.rdcd_path, "-u"],  # unauthenticated mode
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return _wait_for_port(50051, timeout=15)

    def stop(self):
        terminate_process(self.proc)

    def poll(self):
        return self.proc.poll()


@pytest.fixture(scope="session")
def rdcd_server(rdcd_path, has_gpu):
    """Start rdcd in unauthenticated mode for the test session.

    Yields the RdcdServer.  Cleans up on teardown.
    Skips if no GPU is present (daemon won't be useful).
    """
    if not has_gpu:
        pytest.skip("No AMD GPU detected -- skipping daemon tests")

    server = RdcdServer(rdcd_path)
    if not server.start():
        server.stop()
        pytest.fail("rdcd did not start within 15 s")

    yield server

    server.stop()


def run_rdci(rdci_path, *args, timeout=30):
    """Run rdci with the given subsystem arguments and return CompletedProcess."""
    if not args:
        raise ValueError("an RDC subsystem is required")
    subsystem, *subsystem_args = args
    cmd = [rdci_path, subsystem]
    if "-u" not in subsystem_args and "--unauth" not in subsystem_args:
        cmd.append("-u")
    cmd.extend(subsystem_args)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def created_id(result):
    """ID from rdci's "Successfully created ... group ID <n>" message."""
    m = re.search(r"group ID (\d+)", result.stdout)
    assert result.returncode == 0 and m, f"create failed: {result.stdout}{result.stderr}"
    return m.group(1)


def gpu_indexes(rdci_path):
    """GPU indexes that `rdci discovery -l` reports."""
    r = run_rdci(rdci_path, "discovery", "-l")
    assert r.returncode == 0, f"rdci discovery failed: {r.stdout}"
    gpus = re.findall(r"^(\d+)\t", r.stdout, re.M)
    assert gpus, f"rdci discovery listed no GPUs:\n{r.stdout}"
    return gpus


def create_gpu_group(rdci_path, name, gpus):
    """Create a GPU group holding `gpus` and return its ID."""
    gid = created_id(run_rdci(rdci_path, "group", "-c", name))
    r = run_rdci(rdci_path, "group", "-g", gid, "-a", ",".join(gpus))
    if r.returncode != 0:
        run_rdci(rdci_path, "group", "-d", gid)
    assert r.returncode == 0, f"group add GPU failed: {r.stdout}"
    return gid


@pytest.fixture(scope="session")
def python_binding_dir():
    """Return path to python_binding/ directory."""
    base = os.environ.get(
        "RDC_SRC_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "python_binding")
    )
    base = os.path.realpath(base)
    if not os.path.isdir(base):
        pytest.skip("python_binding directory not found")
    return base
