# *************************************************************************
#  * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#  *
#  * See LICENSE.txt for license information
#  ************************************************************************
"""
Tests for the RCCL "mixed" plugin, which demonstrates the NCCL feature
"plugin system supports multiple plugin types from a single shared object".

A single libnccl-mixed.so exports both a net plugin (ncclNetPlugin_vXX) and a
tuner plugin (ncclTunerPlugin_vXX). When RCCL is pointed at this object via
NCCL_NET_PLUGIN and no dedicated tuner plugin is configured, RCCL must:
  1. load the net plugin from the object, and
  2. re-use the SAME object for the tuner plugin (via ncclGetNetPluginLib),
     instead of falling back to the built-in CSV tuner.

There are two tests:
  * test_mixed_symbols_exported - static, no GPU/MPI: builds the object and
    checks both plugin symbols are exported (pytest port of plugins/mixed/example/test.sh).
  * test_mixed_runtime_load - runtime: runs a real collective and asserts from
    the RCCL debug log that net AND tuner were served from the single object.
"""

import os
import re
import shutil
import subprocess

import pytest

# Single owner of the mixed-example paths for this suite. Repo-relative from this
# file at test/ext-plugins/tests/ext-mixed/ (four levels up to the repo root).
MIXED_EXAMPLE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "..", "plugins", "mixed", "example"
    )
)
MIXED_SO = os.path.join(MIXED_EXAMPLE_DIR, "libnccl-mixed.so")
INCLUDE_DIR = os.path.abspath(
    os.path.join(MIXED_EXAMPLE_DIR, "..", "..", "..", "src", "include", "plugin")
)

# Built once per test session; both tests share the same artifact.
_built_so = None


def _expected_symbol(header, macro):
    """Read the expected plugin symbol name from a plugin header (as test.sh does)."""
    path = os.path.join(INCLUDE_DIR, header)
    with open(path) as f:
        for line in f:
            if macro in line:
                # e.g. '#define NCCL_NET_PLUGIN_SYMBOL ncclNetPlugin_v12'
                token = line.split()[2]
                return token.strip().strip('"')
    return None


def _build_mixed_plugin():
    """Build libnccl-mixed.so from source once. Returns the .so path or skips."""
    global _built_so
    if _built_so is not None:
        return _built_so

    cc = os.environ.get("CC", "cc")
    if shutil.which("make") is None or shutil.which(cc) is None:
        pytest.skip(f"make/{cc} not available to build the mixed plugin")

    build = subprocess.run(
        ["make", "default"],
        cwd=MIXED_EXAMPLE_DIR,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert (
        build.returncode == 0
    ), f"Failed to build mixed plugin:\nstdout:\n{build.stdout}\nstderr:\n{build.stderr}"
    assert os.path.exists(MIXED_SO), f"Mixed plugin not produced at {MIXED_SO}"
    _built_so = MIXED_SO
    return MIXED_SO


@pytest.mark.ext_mixed
def test_mixed_symbols_exported():
    """The single shared object must export both a net and a tuner plugin symbol."""
    if shutil.which("nm") is None:
        pytest.skip("nm not available to inspect exported symbols")
    so_path = _build_mixed_plugin()

    net_sym = _expected_symbol("nccl_net.h", "NCCL_NET_PLUGIN_SYMBOL")
    tuner_sym = _expected_symbol("nccl_tuner.h", "NCCL_TUNER_PLUGIN_SYMBOL")
    assert (
        net_sym and tuner_sym
    ), "Could not resolve expected plugin symbols from headers"

    nm = subprocess.run(["nm", "-D", so_path], capture_output=True, text=True)
    assert nm.returncode == 0, f"nm failed on {so_path}: {nm.stderr}"
    exported = nm.stdout

    assert re.search(
        rf"\b{re.escape(net_sym)}\b", exported
    ), f"Net plugin symbol {net_sym} not exported by {so_path}\n{exported}"
    assert re.search(
        rf"\b{re.escape(tuner_sym)}\b", exported
    ), f"Tuner plugin symbol {tuner_sym} not exported by {so_path}\n{exported}"


def _gpu_count():
    """Best-effort count of ROCm GPUs; returns None if it cannot be determined."""
    rocminfo = shutil.which("rocminfo")
    if rocminfo is None:
        return None
    try:
        out = subprocess.run(
            [rocminfo], capture_output=True, text=True, timeout=30
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return None
    return sum(1 for line in out.splitlines() if re.search(r"Device Type:\s*GPU", line))


@pytest.mark.ext_mixed
@pytest.mark.allreduce
def test_mixed_runtime_load(paths):
    """RCCL must load both net and tuner from the single mixed shared object.

    Each process loads the plugins independently, so a 2-rank all_reduce exercises
    the reuse path. rccl-tests may be built with MPI (USE_MPI=ON): that binary
    calls MPI_Init and hangs if launched bare under OpenMPI 5.x (PMIx/PRTE), so
    when an OpenMPI is available it is launched via `mpirun -np 2` with one GPU per
    rank (-g 1), matching the device-api suite. Without OpenMPI (a non-MPI binary)
    it runs bare in one process with two GPUs (-g 2) for the same 2-rank collective.
    """
    so_path = _build_mixed_plugin()

    perf_bin = os.path.join(paths.RCCL_TESTS_DIR, "build", "all_reduce_perf")
    if not os.path.exists(perf_bin):
        pytest.skip(f"rccl-tests all_reduce_perf not found at: {perf_bin}")

    ngpu = _gpu_count()
    if ngpu is not None and ngpu < 2:
        pytest.skip(f"need >=2 GPUs for the mixed runtime test, found {ngpu}")

    env = os.environ.copy()
    # Ensure the reuse path is exercised: no dedicated tuner/profiler plugin.
    env.pop("NCCL_TUNER_PLUGIN", None)
    env.pop("NCCL_PROFILER_PLUGIN", None)
    env.update(
        {
            "LD_LIBRARY_PATH": f"{paths.RCCL_INSTALL_DIR}:{env.get('LD_LIBRARY_PATH', '')}",
            "HSA_NO_SCRATCH_RECLAIM": "1",
            "NCCL_NET_PLUGIN": so_path,
            "NCCL_DEBUG": "INFO",
            "NCCL_DEBUG_SUBSYS": "INIT,NET,TUNING",
        }
    )

    # A USE_MPI=ON rccl-tests binary calls MPI_Init and hangs if launched bare
    # under OpenMPI 5.x (PMIx/PRTE). When an OpenMPI is available, launch it via
    # `mpirun -np 2` with one GPU per rank (-g 1), matching the device-api suite;
    # mpirun forwards this env (NCCL_NET_PLUGIN etc.) to the ranks. Without an
    # OpenMPI (a non-MPI binary), run bare in one process with two GPUs (-g 2) for
    # the same 2-rank collective.
    mpirun = os.path.join(paths.OMPI_INSTALL_DIR, "bin", "mpirun")
    if os.path.exists(mpirun):
        env["PATH"] = f"{paths.OMPI_INSTALL_DIR}/bin:{env.get('PATH', '')}"
        env["LD_LIBRARY_PATH"] = f"{paths.OMPI_INSTALL_DIR}/lib:{env['LD_LIBRARY_PATH']}"
        args = [mpirun, "-np", "2", "-x", "LD_LIBRARY_PATH",
                perf_bin, "-b", "8", "-e", "1M", "-f", "4", "-g", "1", "-n", "5", "-w", "2"]
    else:
        args = [perf_bin, "-b", "8", "-e", "1M", "-f", "4", "-g", "2", "-n", "5", "-w", "2"]

    log_dir = os.path.join(paths.LOGDIR, "mixed_plugin_test_logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "test_mixed_runtime_load.log")
    with open(log_file, "w") as logfile:
        try:
            run = subprocess.run(
                args,
                env=env,
                stdout=logfile,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                timeout=300,
            )
        except subprocess.TimeoutExpired:
            pytest.fail(f"Mixed plugin collective run timed out after 300s, see {log_file}")

    with open(log_file) as logfile:
        log = logfile.read()

    assert run.returncode == 0, f"Mixed plugin collective run failed, see {log_file}"

    # 1) net plugin came from the mixed object specifically: require the object
    #    path on the "Successfully loaded external network plugin" line.
    assert re.search(
        r"Successfully loaded external network plugin\b.*libnccl-mixed\.so", log
    ), f"Net plugin was not loaded from the mixed object, see {log_file}"

    # 2) tuner was served from the SAME object (reuse via ncclGetNetPluginLib):
    #    a tuner is in use, but NOT via a dedicated tuner .so and NOT the built-in CSV.
    assert (
        "TUNER/Plugin: Using " in log
    ), f"No external tuner was loaded, see {log_file}"
    assert (
        "Successfully loaded external tuner plugin" not in log
    ), f"A dedicated tuner plugin was loaded instead of reusing the mixed object, see {log_file}"
    assert (
        "Using built-in CSV tuner" not in log
    ), f"RCCL fell back to the built-in CSV tuner instead of the mixed object, see {log_file}"

    # 3) collective correctness
    assert (
        "Out of bounds values" in log and "OK" in log
    ), f"Collective did not validate correctly, see {log_file}"
