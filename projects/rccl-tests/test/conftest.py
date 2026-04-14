# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import os
import re
import shlex
import socket
import subprocess
import pytest

from test_runner import BUILD_DIR


# ---------------------------------------------------------------------------
# GPU detection — called once at module import, results cached as globals
# ---------------------------------------------------------------------------

def _parse_visible_indices(spec: str) -> list:
    """Parse ROCR/HIP_VISIBLE_DEVICES into a sorted list of integer device indices.

    Supports comma-separated indices and range syntax:
      '0,1,2'  -> [0, 1, 2]
      '0-3'    -> [0, 1, 2, 3]
      '0-3,6'  -> [0, 1, 2, 3, 6]
      '0,0,2'  -> [0, 2]  (deduplicated)
    """
    indices = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            lo, hi = token.split("-", 1)
            indices.extend(range(int(lo), int(hi) + 1))
        else:
            indices.append(int(token))
    return sorted(set(indices))


def _detect_hardware_gpu_count() -> int:
    """Return total GPU count from hardware; 0 if both probes fail."""
    # Fast path: rocm_agent_enumerator -gpu (exits <50ms, one line per GPU agent)
    try:
        out = subprocess.check_output(
            ["rocm_agent_enumerator", "-gpu"],
            text=True, timeout=10, stderr=subprocess.DEVNULL,
        )
        lines = [l.strip() for l in out.splitlines() if l.strip()]
        if lines:
            return len(lines)
    except (subprocess.SubprocessError, OSError):
        pass

    # Slow path: rocminfo (1-5s, parse 'Device Type: GPU' lines)
    try:
        out = subprocess.check_output(
            ["rocminfo"], text=True, timeout=30, stderr=subprocess.DEVNULL,
        )
        count = sum(1 for line in out.splitlines()
                    if "Device Type" in line and "GPU" in line)
        if count > 0:
            return count
    except (subprocess.SubprocessError, OSError):
        pass

    return 0


def detect_gpu_count() -> int:
    """Detect usable GPU count, validating any visible-device env var against hardware.

    Logic:
      1. Probe hardware (rocm_agent_enumerator -> rocminfo) for total GPU count.
      2. If ROCR_VISIBLE_DEVICES or HIP_VISIBLE_DEVICES is set:
           - Parse its indices (range-aware, deduplicating).
           - Validate every index is in [0, hw_count). If not -> pytest.exit().
           - Return len(visible_indices).
      3. If no env var -> return hw_count.
      4. If hardware probe failed -> pytest.exit().
    """
    hw_count = _detect_hardware_gpu_count()

    for var in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
        raw = os.environ.get(var, "").strip()
        if not raw:
            continue

        try:
            visible = _parse_visible_indices(raw)
        except ValueError as e:
            pytest.exit(
                f"{var}={raw!r} could not be parsed as device indices: {e}",
                returncode=1,
            )

        if hw_count == 0:
            pytest.exit(
                f"{var}={raw!r} is set but hardware GPU detection failed — "
                f"cannot validate requested indices",
                returncode=1,
            )

        invalid = [i for i in visible if i >= hw_count]
        if invalid:
            pytest.exit(
                f"{var}={raw!r} requests device indices {invalid} but hardware "
                f"only has {hw_count} GPU(s) (valid indices: 0-{hw_count - 1})",
                returncode=1,
            )

        return len(visible)  # valid subset — use it

    # No env var set
    if hw_count == 0:
        pytest.exit(
            "Failed to detect GPU count (rocm_agent_enumerator and rocminfo both "
            "failed, no ROCR_VISIBLE_DEVICES/HIP_VISIBLE_DEVICES set)",
            returncode=1,
        )
    return hw_count


def detect_gpu_arch() -> str:
    """Return GPU arch string like 'gfx942', or None if undetectable."""
    try:
        out = subprocess.check_output(
            ["rocminfo"], text=True, timeout=30, stderr=subprocess.DEVNULL,
        )
        for line in out.splitlines():
            if "Name:" in line and "gfx" in line:
                name = line.split()[-1].strip()
                if name.startswith("gfx"):
                    return name
    except (subprocess.SubprocessError, OSError):
        pass
    return None


# Evaluated once at import time — used by fixtures and pytest_collection_modifyitems
NGPUS: int = detect_gpu_count()
GPU_ARCH = detect_gpu_arch()


def pytest_addoption(parser):
    group = parser.getgroup("rccl", "RCCL-Tests Options")
    group.addoption("--hostfile", action="store", default="",
                    help="MPI hostfile for multi-node tests")
    group.addoption("--test-timeout", action="store", default=300, type=int,
                    help="Per-test subprocess timeout in seconds (default: 300)")
    group.addoption("--mpi-args", action="store", default="",
                    help="Extra args inserted into every mpirun invocation, "
                         "shell-tokenized with shlex. "
                         "Example: --mpi-args='--bind-to none --mca pml ucx'")


# ---------------------------------------------------------------------------
# RCCL version detection for report header
# ---------------------------------------------------------------------------

_RCCL_VERSION_FIELDS = {
    "RCCL version": "rccl_version",
    "HIP version":  "hip_version",
    "ROCm version": "rocm_version",
    "Hostname":     "hostname",
    "Librccl path": "librccl_path",
}


def _detect_rccl_info():
    """Run a perf binary with NCCL_DEBUG=VERSION and parse metadata from stderr."""
    info = {}
    executable = os.path.join(BUILD_DIR, "all_reduce_perf")
    if not os.path.isfile(executable):
        info["_warning"] = f"binary not found: {executable}"
        return info

    env = os.environ.copy()
    env["NCCL_DEBUG"] = "VERSION"
    try:
        result = subprocess.run(
            [executable, "-b", "8", "-e", "8", "-t", "1", "-g", "1"],
            capture_output=True, text=True, timeout=30, env=env,
        )
    except subprocess.TimeoutExpired:
        info["_warning"] = f"{executable} timed out during version probe"
        return info
    except (subprocess.SubprocessError, OSError) as e:
        info["_warning"] = str(e)
        return info

    output = result.stderr + "\n" + result.stdout
    for line in output.splitlines():
        for label, key in _RCCL_VERSION_FIELDS.items():
            m = re.match(rf"{re.escape(label)}\s*:\s*(.+)", line)
            if m:
                info[key] = m.group(1).strip()
                break

    if not info and result.returncode != 0:
        hint = result.stderr.strip().splitlines()
        first_line = hint[0] if hint else f"exit code {result.returncode}"
        info["_warning"] = first_line

    return info


def pytest_report_header(config):
    timeout = config.getoption("--test-timeout")
    hostfile = config.getoption("--hostfile")
    mpi_args = config.getoption("--mpi-args")

    rccl = _detect_rccl_info()
    hostname = rccl.get("hostname", socket.gethostname())
    rocr_visible = os.environ.get("ROCR_VISIBLE_DEVICES", "all")
    hip_visible = os.environ.get("HIP_VISIBLE_DEVICES", "all")

    lines = [
        f"rccl-tests  timeout: {timeout}s  ngpus: {NGPUS}"
        + (f"  hostfile: {hostfile}" if hostfile else ""),
        f"  Hostname:             {hostname}",
        f"  ROCR_VISIBLE_DEVICES: {rocr_visible}",
        f"  HIP_VISIBLE_DEVICES:  {hip_visible}",
        f"  GPU arch:             {GPU_ARCH or 'unknown'}",
    ]
    if mpi_args:
        lines.append(f"  mpirun extra args:    {mpi_args}")
    if rccl.get("rccl_version"):
        lines.append(f"  RCCL version:         {rccl['rccl_version']}")
    if rccl.get("hip_version"):
        lines.append(f"  HIP version:          {rccl['hip_version']}")
    if rccl.get("rocm_version"):
        lines.append(f"  ROCm version:         {rccl['rocm_version']}")
    if rccl.get("librccl_path"):
        lines.append(f"  Librccl path:         {rccl['librccl_path']}")
    if rccl.get("_warning"):
        lines.append(f"  RCCL info:            (unavailable) {rccl['_warning']}")

    return lines


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def _configure_runner(request):
    """Push --test-timeout, --hostfile, and --mpi-args into test_runner globals.

    This wires the CLI options into every run_rccl_perf / run_rccl_mpi call
    without requiring each test function to accept and forward these fixtures.
    """
    import test_runner
    test_runner.DEFAULT_TIMEOUT = request.config.getoption("--test-timeout")
    hostfile = request.config.getoption("--hostfile")
    test_runner.DEFAULT_HOSTFILE = hostfile if hostfile else None
    mpi_args = request.config.getoption("--mpi-args")
    test_runner.DEFAULT_MPI_ARGS = shlex.split(mpi_args) if mpi_args else []


@pytest.fixture(scope="session")
def gpu_count() -> int:
    """Usable GPU count for this session (hardware-detected, env-var-filtered).

    Returns an int. Never 0 — detect_gpu_count() calls pytest.exit() first.
    """
    return NGPUS


# ---------------------------------------------------------------------------
# Arch gating
# ---------------------------------------------------------------------------

def pytest_collection_modifyitems(config, items):
    """Apply arch markers and auto-skip tests on non-matching hardware.

    Detects arch-gated tests by inspecting parametrize callspec values for
    Collective objects that carry an arch_gate tuple. This runs at collection
    time, before any test body executes, so it correctly skips items before
    they are scheduled to run.

    For each gated item:
      - Applies the corresponding registered marker (e.g. gfx942_gfx950) so
        -m gfx942_gfx950 can be used to select or exclude these tests.
      - Adds a skip marker if the detected GPU arch is not in the gate set.
    """
    for item in items:
        arch_gate = None
        if hasattr(item, "callspec"):
            for val in item.callspec.params.values():
                if hasattr(val, "arch_gate") and val.arch_gate:
                    arch_gate = val.arch_gate
                    break
        if not arch_gate:
            continue
        marker_name = "_".join(arch_gate)  # ("gfx942", "gfx950") -> "gfx942_gfx950"
        item.add_marker(getattr(pytest.mark, marker_name))
        if GPU_ARCH not in arch_gate:
            item.add_marker(pytest.mark.skip(
                reason=f"requires {'/'.join(arch_gate)}, got {GPU_ARCH}"))
