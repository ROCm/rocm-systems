# *************************************************************************
#  * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#  *
#  * See LICENSE.txt for license information
#  ************************************************************************
"""Tests for one-sided RMA plugins."""

import ctypes
import os
import re
import shutil
import subprocess

import pytest

_GIN_EXAMPLE_NAME = "librccl-gin-example.so"


def _gin_example_so(rccl_install_dir):
    """Locate the GIN example .so, or return None when it was not built."""
    candidates = (
        os.path.join(rccl_install_dir, "plugins", "gin", "example", _GIN_EXAMPLE_NAME),
        os.path.join(rccl_install_dir, "build", "release", "test", "unit", "plugins", _GIN_EXAMPLE_NAME),
        os.path.join(rccl_install_dir, "build", "debug", "test", "unit", "plugins", _GIN_EXAMPLE_NAME),
        os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "..",
                         "plugins", "gin", "example", _GIN_EXAMPLE_NAME)
        ),
    )
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


class _RmaV15(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("init", ctypes.c_void_p),
        ("devices", ctypes.c_void_p),
        ("getProperties", ctypes.c_void_p),
        ("listen", ctypes.c_void_p),
        ("connect", ctypes.c_void_p),
        ("createContext", ctypes.c_void_p),
        ("regMrSym", ctypes.c_void_p),
        ("regMrSymDmaBuf", ctypes.c_void_p),
        ("deregMrSym", ctypes.c_void_p),
        ("destroyContext", ctypes.c_void_p),
        ("closeColl", ctypes.c_void_p),
        ("closeListen", ctypes.c_void_p),
        ("iput", ctypes.c_void_p),
        ("iputSignal", ctypes.c_void_p),
        ("iget", ctypes.c_void_p),
        ("iflush", ctypes.c_void_p),
        ("test", ctypes.c_void_p),
        ("rmaProgress", ctypes.c_void_p),
        ("queryLastError", ctypes.c_void_p),
        ("finalize", ctypes.c_void_p),
    ]


class _RmaRequest(ctypes.Structure):
    _fields_ = [
        ("done", ctypes.c_int),
        ("optFlags", ctypes.c_uint32),
    ]


def _run_all_reduce(paths, extra_env, log_name):
    """Run a short 8-rank all_reduce with extra_env applied; return (rc, log, log_file).

    A USE_MPI=ON rccl-tests binary calls MPI_Init and hangs if launched bare under
    OpenMPI 5.x, so use mpirun when one is available.
    """
    perf_bin = os.path.join(paths.RCCL_TESTS_DIR, "build", "all_reduce_perf")
    if not os.path.exists(perf_bin):
        pytest.skip(f"rccl-tests all_reduce_perf not found at: {perf_bin}")

    env = os.environ.copy()
    # A stray plugin selection in the caller's environment would decide adoption.
    for var in ("NCCL_RMA_PLUGIN", "NCCL_GIN_PLUGIN", "NCCL_NET_PLUGIN", "NCCL_TUNER_PLUGIN",
                "NCCL_PROFILER_PLUGIN", "NCCL_ENV_PLUGIN"):
        env.pop(var, None)
    env.update(
        {
            "LD_LIBRARY_PATH": f"{paths.RCCL_INSTALL_DIR}:{env.get('LD_LIBRARY_PATH', '')}",
            "HSA_NO_SCRATCH_RECLAIM": "1",
            "NCCL_DEBUG": "INFO",
            # rma.cc and gin.cc log adoption under NCCL_INIT | NCCL_NET.
            "NCCL_DEBUG_SUBSYS": "INIT,NET",
        }
    )
    env.update(extra_env)

    perf_args = ["-b", "8", "-e", "1M", "-f", "4", "-g", "1", "-n", "5", "-w", "2"]
    mpirun = os.path.join(paths.OMPI_INSTALL_DIR, "bin", "mpirun")
    if os.path.exists(mpirun):
        env["PATH"] = f"{paths.OMPI_INSTALL_DIR}/bin:{env.get('PATH', '')}"
        env["LD_LIBRARY_PATH"] = f"{paths.OMPI_INSTALL_DIR}/lib:{env['LD_LIBRARY_PATH']}"
        args = [mpirun, "-np", "8",
                "--mca", "pml", "ucx",
                "--mca", "btl", "^vader,openib",
                "-x", "LD_LIBRARY_PATH", perf_bin] + perf_args
    else:
        # Without MPI, one process still loads and adopts the plugin.
        args = [perf_bin] + perf_args

    log_dir = os.path.join(paths.LOGDIR, "rma_plugin_test_logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, log_name)
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
            pytest.fail(f"RMA plugin run timed out after 300s, see {log_file}")

    with open(log_file) as logfile:
        return run.returncode, logfile.read(), log_file


@pytest.mark.ext_rma
@pytest.mark.allreduce
def test_rma_v15_is_selected_and_preserves_opt_flags(paths):
    """RCCL selects v15, whose data-operation callbacks receive optFlags."""
    rc, log, log_file = _run_all_reduce(
        paths,
        {"NCCL_RMA_PLUGIN": paths.RMA_SO},
        "test_rma_v15_is_selected_and_preserves_opt_flags.log",
    )
    assert rc == 0, f"all_reduce with the RMA v15 example failed, see {log_file}"
    assert re.search(r"RMA/Plugin: Loaded rma plugin Example \(v15\)", log), \
        f"RCCL did not select ncclRmaPlugin_v15, see {log_file}"

    plugin = ctypes.CDLL(paths.RMA_SO)
    table = _RmaV15.in_dll(plugin, "ncclRmaPlugin_v15")
    assert table.name == b"Example"

    complete = ctypes.CFUNCTYPE(
        ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
    )(table.test)

    request = ctypes.c_void_p()
    iput = ctypes.CFUNCTYPE(
        ctypes.c_int,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p,
        ctypes.c_size_t, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint32,
        ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p),
    )(table.iput)
    assert iput(None, 0, 0, None, 0, 0, None, 0, 1, ctypes.byref(request)) == 0
    assert ctypes.cast(request, ctypes.POINTER(_RmaRequest)).contents.optFlags == 1
    done = ctypes.c_int()
    assert complete(None, request, ctypes.byref(done)) == 0

    request = ctypes.c_void_p()
    iput_signal = ctypes.CFUNCTYPE(
        ctypes.c_int,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p,
        ctypes.c_size_t, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint32,
        ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32,
        ctypes.c_bool, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p),
    )(table.iputSignal)
    assert iput_signal(
        None, 0, 0, None, 0, 0, None, 0, 0, None, 0, 0, False, 1,
        ctypes.byref(request),
    ) == 0
    assert ctypes.cast(request, ctypes.POINTER(_RmaRequest)).contents.optFlags == 1
    assert complete(None, request, ctypes.byref(done)) == 0

    request = ctypes.c_void_p()
    iget = ctypes.CFUNCTYPE(
        ctypes.c_int,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p,
        ctypes.c_size_t, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint32,
        ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p),
    )(table.iget)
    assert iget(None, 0, 0, None, 0, 0, None, 0, 1, ctypes.byref(request)) == 0
    assert ctypes.cast(request, ctypes.POINTER(_RmaRequest)).contents.optFlags == 1
    assert complete(None, request, ctypes.byref(done)) == 0


@pytest.mark.ext_rma
def test_rma_example_symbol_matches_negotiated_version(paths):
    """The example must export the current RMA v15 symbol."""
    if shutil.which("nm") is None:
        pytest.skip("nm not available to inspect exported symbols")

    nm = subprocess.run(["nm", "-D", paths.RMA_SO], capture_output=True, text=True)
    assert nm.returncode == 0, f"nm failed on {paths.RMA_SO}: {nm.stderr}"
    assert re.search(r"\bncclRmaPlugin_v15\b", nm.stdout), (
        f"{paths.RMA_SO} does not export ncclRmaPlugin_v15:\n{nm.stdout}"
    )


@pytest.mark.ext_rma
@pytest.mark.allreduce
def test_rma_example_is_adopted_over_builtin_proxy(paths):
    """The external RMA v15 example must be adopted over the built-in backend.

    Unlike gin.cc, rma.cc has no rule that skips external plugins.
    """
    rc, log, log_file = _run_all_reduce(
        paths,
        {"NCCL_RMA_PLUGIN": paths.RMA_SO},
        "test_rma_example_is_adopted_over_builtin_proxy.log",
    )

    assert rc == 0, f"all_reduce with the RMA example failed, see {log_file}"

    assert re.search(
        r"RMA/Plugin: Successfully loaded external plugin\b.*librccl-rma-example\.so", log
    ), f"RMA example was not loaded, see {log_file}"

    assert re.search(
        r"RMA/Plugin: Loaded rma plugin Example \(v15\)", log
    ), f"RMA example was not negotiated as v15, see {log_file}"

    assert re.search(
        r"RMA/Plugin: Assigned plugin Example to comm", log
    ), f"RMA example was not assigned to the comm, see {log_file}"

    # A stub data path must not break an ordinary collective.
    assert re.search(r"Out of bounds values\s*:\s*0\s+OK", log), \
        f"Collective did not validate correctly, see {log_file}"


@pytest.mark.ext_rma
@pytest.mark.allreduce
def test_rma_example_resolves_by_short_name(paths):
    """NCCL_RMA_PLUGIN=example must resolve to librccl-rma-example.so.

    plugin_open.cc builds the name from pluginPrefix[ncclPluginTypeRma], which is
    "librccl-rma"; the upstream libnccl-rma-example.so could never match.
    """
    plugin_dir = os.path.dirname(paths.RMA_SO)
    rc, log, log_file = _run_all_reduce(
        paths,
        {
            "NCCL_RMA_PLUGIN": "example",
            "LD_LIBRARY_PATH": f"{plugin_dir}:{paths.RCCL_INSTALL_DIR}:{os.environ.get('LD_LIBRARY_PATH', '')}",
        },
        "test_rma_example_resolves_by_short_name.log",
    )

    assert rc == 0, f"all_reduce with NCCL_RMA_PLUGIN=example failed, see {log_file}"
    assert re.search(
        r"RMA/Plugin: Successfully loaded external plugin\b.*librccl-rma-example\.so", log
    ), f"Short name 'example' did not resolve to librccl-rma-example.so, see {log_file}"
    assert re.search(
        r"RMA/Plugin: Assigned plugin Example to comm", log
    ), f"RMA example was not assigned to the comm, see {log_file}"


@pytest.mark.ext_rma
@pytest.mark.allreduce
def test_rma_example_backs_the_gin_proxy(paths):
    """The adopted RMA plugin must be the backend serving RCCL's GIN proxy."""
    gin_so = _gin_example_so(paths.RCCL_INSTALL_DIR)
    if gin_so is None:
        pytest.skip("librccl-gin-example.so not built; rebuild with -DBUILD_PLUGIN_EXAMPLES=ON")

    rc, log, log_file = _run_all_reduce(
        paths,
        {
            "NCCL_RMA_PLUGIN": paths.RMA_SO,
            "NCCL_GIN_PLUGIN": gin_so,
            "NCCL_GIN_ENABLE": "1",
        },
        "test_rma_example_backs_the_gin_proxy.log",
    )

    assert rc == 0, f"all_reduce with the RMA and GIN examples failed, see {log_file}"

    match = re.search(
        r"GIN/Plugin: Skipping external proxy plugin .*using NCCL GIN proxy over RMA backend (\S+)", log
    )
    assert match, f"GIN did not report skipping the external proxy plugin, see {log_file}"
    assert match.group(1) == "Example", (
        f"GIN proxy is backed by {match.group(1)!r}, expected the adopted RMA example, see {log_file}"
    )


_PUT_BURST_TEST = "GinMPIDeviceTests.PutBurst_LocalCounter"
_PUT_BURST_PUTS = 256
_PUT_BURST_QUEUE = 32
_RMA_EXAMPLE_COUNTS = re.compile(r"RMA/Example: (\d+) data ops, (\d+) aggregated")


def _unit_tests_mpi(rccl_install_dir):
    """Locate rccl-UnitTestsMPI, or return None when MPI tests were not built."""
    for build_type in ("release", "debug"):
        path = os.path.join(rccl_install_dir, "build", build_type, "test", "rccl-UnitTestsMPI")
        if os.path.exists(path):
            return path
    return None


def _run_put_burst(paths, unit_bin, extra_env, log_name):
    """Run the 2-rank GIN put burst over the RMA example; return (rc, log, log_file)."""
    env = os.environ.copy()
    for var in ("NCCL_RMA_PLUGIN", "NCCL_GIN_PLUGIN", "NCCL_NET_PLUGIN", "NCCL_TUNER_PLUGIN",
                "NCCL_PROFILER_PLUGIN", "NCCL_ENV_PLUGIN", "NCCL_GIN_PROXY_POLL_BATCH",
                "NCCL_GIN_PROXY_QUEUE_SIZE"):
        env.pop(var, None)
    run_env = {
        "LD_LIBRARY_PATH": f"{paths.OMPI_INSTALL_DIR}/lib:{paths.RCCL_INSTALL_DIR}:{env.get('LD_LIBRARY_PATH', '')}",
        "HSA_NO_SCRATCH_RECLAIM": "1",
        "NCCL_DEBUG": "INFO",
        "NCCL_DEBUG_SUBSYS": "INIT,NET",
        "NCCL_RMA_PLUGIN": paths.RMA_SO,
        "NCCL_GIN_ENABLE": "1",
        "NCCL_GIN_TYPE": "2",
        "NCCL_GIN_PROXY_QUEUE_SIZE": str(_PUT_BURST_QUEUE),
        "NCCL_CUMEM_ENABLE": "1",
        "NCCL_DMABUF_ENABLE": "1",
        "RCCL_ENABLE_INTRANET": "1",
    }
    run_env.update(extra_env)
    env.update(run_env)
    env["PATH"] = f"{paths.OMPI_INSTALL_DIR}/bin:{env.get('PATH', '')}"

    # Both ranks share a node, so let Open MPI pick its own transport.
    args = [os.path.join(paths.OMPI_INSTALL_DIR, "bin", "mpirun"), "-np", "2"]
    for var in run_env:
        args += ["-x", var]
    args += [unit_bin, f"--gtest_filter={_PUT_BURST_TEST}"]

    log_dir = os.path.join(paths.LOGDIR, "rma_plugin_test_logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, log_name)
    with open(log_file, "w") as logfile:
        try:
            run = subprocess.run(args, env=env, stdout=logfile, stderr=subprocess.STDOUT,
                                 universal_newlines=True, timeout=300)
        except subprocess.TimeoutExpired:
            pytest.fail(f"GIN put burst timed out after 300s, see {log_file}")

    with open(log_file) as logfile:
        log = logfile.read()
    if f"[  SKIPPED ] {_PUT_BURST_TEST}" in log:
        pytest.skip(f"{_PUT_BURST_TEST} skipped on this system, see {log_file}")
    return run.returncode, log, log_file


@pytest.mark.ext_rma
def test_gin_proxy_batches_and_hints_through_rma_example(paths):
    """The GIN proxy forwards device puts to the adopted RMA plugin with the aggregation hint."""
    unit_bin = _unit_tests_mpi(paths.RCCL_INSTALL_DIR)
    if unit_bin is None:
        pytest.skip("rccl-UnitTestsMPI not built; rebuild with --enable-mpi-tests")
    if not os.path.exists(os.path.join(paths.OMPI_INSTALL_DIR, "bin", "mpirun")):
        pytest.skip("mpirun not found under OMPI_INSTALL_DIR")

    # A run is capped by the batch and the queue, and its last op is never hinted, so
    # hinted <= puts - puts / min(batch, queue); a batch of one hints nothing.
    for batch, extra_env in ((1, {"NCCL_GIN_PROXY_POLL_BATCH": "1"}),
                             (4, {"NCCL_GIN_PROXY_POLL_BATCH": "4"}),
                             (32, {})):
        rc, log, log_file = _run_put_burst(paths, unit_bin, extra_env, f"gin_put_burst_poll_batch_{batch}.log")
        assert rc == 0, f"GIN put burst failed with poll batch {batch}, see {log_file}"
        counts = [(int(ops), int(agg)) for ops, agg in _RMA_EXAMPLE_COUNTS.findall(log)]
        hinted = [agg for ops, agg in counts if ops == _PUT_BURST_PUTS]
        assert hinted, f"RMA example did not report {_PUT_BURST_PUTS} puts: {counts}, see {log_file}"
        cap = _PUT_BURST_PUTS - _PUT_BURST_PUTS // min(batch, _PUT_BURST_QUEUE)
        low = 0 if batch == 1 else 1
        assert low <= hinted[0] <= cap, (
            f"Poll batch {batch}: {hinted[0]} of {_PUT_BURST_PUTS} puts hinted, expected {low}..{cap}, "
            f"see {log_file}"
        )

