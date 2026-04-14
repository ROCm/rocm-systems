# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import os
import subprocess
import pytest

BUILD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build")

# Overridden at session start by conftest._configure_runner (autouse fixture)
DEFAULT_TIMEOUT = 300
DEFAULT_HOSTFILE = None
DEFAULT_MPI_ARGS = []  # extra args inserted between `-np N` and the executable

# Substrings that indicate a feature is not compiled into this RCCL build.
# When a binary exits non-zero with one of these messages the test is skipped
# rather than failed
_UNSUPPORTED_FEATURE_PATTERNS = [
    "This version of RCCL doesn't support ncclAllReduceWithBias",
]


def _check_unsupported(executable_name, stdout, stderr):
    combined = stdout + stderr
    for pattern in _UNSUPPORTED_FEATURE_PATTERNS:
        if pattern in combined:
            pytest.skip(f"{executable_name}: feature not available in this RCCL build "
                        f"({pattern!r})")


def _apply_env_overrides(env, env_overrides):
    """Merge env_overrides into env. Keys mapped to None are deleted."""
    if not env_overrides:
        return
    for key, value in env_overrides.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value


def run_rccl_perf(executable_name, args, env_overrides=None, timeout=None):
    """Run a rccl-tests perf binary and fail the test on error or timeout.

    env_overrides is merged on top of os.environ.copy() at call time.
    Keys whose value is None are removed from the inherited environment;
    use this to scrub variables such as NCCL_DEBUG that would pollute
    stdout in tests that match against the binary's output.
    Callers are responsible for preserving ROCR/HIP_VISIBLE_DEVICES in
    env_overrides if they need to override other keys without changing
    device visibility.
    """
    if timeout is None:
        timeout = DEFAULT_TIMEOUT
    if ".." in executable_name or os.sep in executable_name:
        pytest.fail(f"Invalid executable name (path traversal): {executable_name!r}")

    executable = os.path.join(BUILD_DIR, executable_name)
    if not os.path.isfile(executable):
        pytest.fail(f"Binary not found: {executable}")

    cmd = [executable] + args
    env = os.environ.copy()
    _apply_env_overrides(env, env_overrides)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=timeout, env=env)
    except FileNotFoundError as e:
        pytest.skip(f"required executable not on PATH: {e}")
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout.decode("utf-8", errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "(no output)")
        stderr = e.stderr.decode("utf-8", errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "(no output)")
        pytest.fail(f"{executable_name} timed out after {timeout}s\n"
                    f"stdout: {stdout}\nstderr: {stderr}")

    if result.returncode != 0:
        _check_unsupported(executable_name, result.stdout, result.stderr)
        pytest.fail(f"{executable_name} failed (rc={result.returncode})\n"
                    f"cmd: {' '.join(cmd)}\n"
                    f"stdout: {result.stdout}\nstderr: {result.stderr}")

    return result


def run_rccl_mpi(executable_name, nprocs, args, hostfile=None,
                 mpi_args=None, env_overrides=None, timeout=None):
    """Run a perf binary under mpirun.

    Extra mpirun flags (--bind-to, --mca, etc.) may be passed per-call via
    `mpi_args=[...]`; if omitted, the global DEFAULT_MPI_ARGS set by
    conftest._configure_runner from `--mpi-args=...` is used.
    """
    if nprocs <= 0:
        pytest.fail(f"nprocs must be positive, got {nprocs}")
    if timeout is None:
        timeout = DEFAULT_TIMEOUT
    if hostfile is None:
        hostfile = DEFAULT_HOSTFILE
    if mpi_args is None:
        mpi_args = DEFAULT_MPI_ARGS

    if ".." in executable_name or os.sep in executable_name:
        pytest.fail(f"Invalid executable name (path traversal): {executable_name!r}")

    executable = os.path.join(BUILD_DIR, executable_name)
    if not os.path.isfile(executable):
        pytest.fail(f"Binary not found: {executable}")

    cmd = ["mpirun", "-np", str(nprocs)]
    if hostfile:
        cmd += ["-hostfile", hostfile]
    cmd += list(mpi_args)
    cmd += [executable] + args

    env = os.environ.copy()
    _apply_env_overrides(env, env_overrides)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=timeout, env=env)
    except FileNotFoundError as e:
        pytest.skip(f"mpirun not on PATH (or required helper missing): {e}")
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout.decode("utf-8", errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "(no output)")
        stderr = e.stderr.decode("utf-8", errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "(no output)")
        pytest.fail(f"MPI {executable_name} timed out after {timeout}s\n"
                    f"stdout: {stdout}\nstderr: {stderr}")

    if result.returncode != 0:
        _check_unsupported(executable_name, result.stdout, result.stderr)
        pytest.fail(f"MPI {executable_name} failed (rc={result.returncode})\n"
                    f"cmd: {' '.join(cmd)}\n"
                    f"stdout: {result.stdout}\nstderr: {result.stderr}")

    return result
