# *************************************************************************
#  * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#  *
#  * See LICENSE.txt for license information
#  ************************************************************************

"""Stress/regression tests for NCCL inspector plugin teardown (NCCL issue #2000).

Exercises comm create → collectives → destroy under P2P + verbose kernel-channel
tracing and an aggressive dump thread. Uses subprocess timeouts to catch proxy
thread deadlocks and ncclCommDestroy hangs.
"""

import glob
import os
import shutil
import subprocess

import pytest

# Native lifecycle harness: <rccl>/plugins/profiler/inspector/test/
NATIVE_INSPECTOR_TEST_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..", "..", "..", "..",
        "plugins", "profiler", "inspector", "test",
    )
)

COMM_LIFECYCLE_ITERATIONS = 8
STRESS_TIMEOUT_SECONDS = 120
NATIVE_TEST_TIMEOUT_SECONDS = 60


def _inspector_stress_env(paths, dump_dir):
    env = os.environ.copy()
    env.update({
        "PATH": f"{paths.OMPI_INSTALL_DIR}/bin:{env.get('PATH', '')}",
        "LD_LIBRARY_PATH": (
            f"{paths.RCCL_INSTALL_DIR}:{paths.OMPI_INSTALL_DIR}/lib:"
            f"{paths.INSPECTOR_DIR}:{env.get('LD_LIBRARY_PATH', '')}"
        ),
        "HSA_NO_SCRATCH_RECLAIM": "1",
        "NCCL_PROFILER_PLUGIN": paths.INSPECTOR_SO,
        "NCCL_INSPECTOR_ENABLE": "1",
        "NCCL_INSPECTOR_ENABLE_P2P": "1",
        "NCCL_INSPECTOR_DUMP_VERBOSE": "1",
        "NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS": "100",
        "NCCL_INSPECTOR_DUMP_DIR": dump_dir,
        "NCCL_DEBUG": "INFO",
    })
    return env


def _mpirun_perf(paths, perf_binary, log_file, env):
    args = [
        f"{paths.OMPI_INSTALL_DIR}/bin/mpirun", "-np", "8",
        "--mca", "pml", "ucx",
        "--mca", "btl", "^vader,openib",
        perf_binary,
        "-b", "8",
        "-e", "4M",
        "-f", "2",
        "-g", "1",
    ]
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "w") as logfile:
        return subprocess.run(
            args,
            env=env,
            stdout=logfile,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            timeout=STRESS_TIMEOUT_SECONDS,
        )


@pytest.mark.ext_inspector
@pytest.mark.inspector_regression
def test_coll_info_lifecycle_native():
    """Unit regression for NCCL #2000 collInfo unlock-before-destroy ordering."""
    if shutil.which("make") is None or shutil.which("g++") is None:
        pytest.skip("g++/make not available to build the native inspector unit test")

    src = os.path.join(NATIVE_INSPECTOR_TEST_DIR, "test_coll_info_lifecycle.cpp")
    if not os.path.exists(src):
        pytest.skip(f"Native inspector unit test not found at {src}")

    try:
        build = subprocess.run(
            ["make", "-C", NATIVE_INSPECTOR_TEST_DIR],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert build.returncode == 0, (
            f"Failed to build native inspector unit test:\n"
            f"{build.stdout}\n{build.stderr}"
        )

        binary = os.path.join(NATIVE_INSPECTOR_TEST_DIR, "test_coll_info_lifecycle")
        run = subprocess.run(
            [binary],
            cwd=NATIVE_INSPECTOR_TEST_DIR,
            capture_output=True,
            text=True,
            timeout=NATIVE_TEST_TIMEOUT_SECONDS,
        )
        assert run.returncode == 0, (
            f"Native inspector lifecycle test failed:\n"
            f"{run.stdout}\n{run.stderr}"
        )
    finally:
        subprocess.run(
            ["make", "-C", NATIVE_INSPECTOR_TEST_DIR, "clean"],
            capture_output=True,
            text=True,
            timeout=30,
        )


@pytest.mark.ext_inspector
@pytest.mark.inspector_regression
@pytest.mark.parametrize(
    "collective,perf_binary_name",
    [
        pytest.param("allreduce", "all_reduce_perf", id="allreduce"),
        pytest.param("allgather", "all_gather_perf", id="allgather"),
    ],
)
def test_comm_lifecycle_stress(paths, collective, perf_binary_name):
    """Repeated full comm lifecycles (separate mpirun per iteration) with inspector tracing."""
    dump_dir = os.path.join(
        paths.INSPECTOR_DUMP_DIR, "lifecycle_stress", f"{collective}_lifecycle"
    )
    os.makedirs(dump_dir, exist_ok=True)

    for f in glob.glob(os.path.join(dump_dir, "*.log")):
        os.remove(f)

    env = _inspector_stress_env(paths, dump_dir)
    perf_binary = os.path.join(paths.RCCL_TESTS_DIR, "build", perf_binary_name)
    if not os.path.exists(perf_binary):
        pytest.skip(f"RCCL perf binary not found: {perf_binary}")

    log_dir = os.path.join(paths.LOGDIR, "inspector_lifecycle_stress_logs")
    os.makedirs(log_dir, exist_ok=True)

    for cycle in range(COMM_LIFECYCLE_ITERATIONS):
        log_file = os.path.join(log_dir, f"{collective}_lifecycle_cycle_{cycle}.log")
        result = _mpirun_perf(paths, perf_binary, log_file, env)
        assert result.returncode == 0, (
            f"Inspector {collective} comm lifecycle stress failed on cycle {cycle}, "
            f"see {log_file}"
        )
