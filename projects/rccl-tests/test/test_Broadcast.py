#################################################################################
# Copyright (C) 2019 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

import os
import shlex
import subprocess
import itertools
import math

import pytest

from .gin_sdma_harness import (
    GiB,
    MiB,
    assert_bcast_ok,
    bcast_data_failed,
    bcast_rows,
    bcast_tiers,
    detect_ngpus,
    env_int,
    gin_perf_argv,
    launch_bcast_gin_sdma,
    run_bcast_with_conn_gate_retry,
    run_with_conn_gate_retry,
)

try:
    _detected_ngpus = detect_ngpus()
except RuntimeError:
    _detected_ngpus = 0
ngpus = max(1, _detected_ngpus)
log_ngpus = int(math.log2(ngpus))

nthreads = ["1"]
nprocs = ["2"]
ngpus_single = [str(2**x) for x in range(log_ngpus + 1)]
ngpus_mpi = ["1", "2"]
byte_range = [("4", "128M")]
op = ["sum", "prod", "min", "max"]
step_factor = ["2"]
datatype = [
    "int8",
    "uint8",
    "int32",
    "uint32",
    "int64",
    "uint64",
    "half",
    "float",
    "double",
]
memory_type = ["coarse", "fine", "host"]

path = os.path.dirname(os.path.abspath(__file__))
executable = path + "/../build/broadcast_perf"


@pytest.mark.parametrize(
    "nthreads, ngpus_single, byte_range, op, step_factor, datatype, memory_type",
    itertools.product(
        nthreads, ngpus_single, byte_range, op, step_factor, datatype, memory_type
    ),
)
def test_BroadcastSingleProcess(
    nthreads, ngpus_single, byte_range, op, step_factor, datatype, memory_type
):
    try:
        args = [
            executable,
            "-t",
            nthreads,
            "-g",
            ngpus_single,
            "-b",
            byte_range[0],
            "-e",
            byte_range[1],
            "-o",
            op,
            "-f",
            step_factor,
            "-d",
            datatype,
            "-Y",
            memory_type,
        ]
        if memory_type == "fine":
            args.insert(0, "HSA_FORCE_FINE_GRAIN_PCIE=1")
        args_str = " ".join(args)
        rccl_test = subprocess.run(
            args_str, stdout=subprocess.PIPE, universal_newlines=True, shell=True
        )
    except subprocess.CalledProcessError as err:
        print(rccl_test.stdout)
        pytest.fail("Broadcast test error(s) detected.")

    assert rccl_test.returncode == 0


# ---------------------------------------------------------------------------
# GIN-SDMA Broadcast multi-segment regression tests (parity with AllGather /
# AllToAll in test_AllGather.py / test_AllToAll.py).
#
# These drive the real GinHybridBroadcastKernel (deviceImpl 3, NCCL_GIN_TYPE=6)
# at message sizes that cross the 128 MiB SDMA copy clamp in the Anvil-SDMA
# backend Put. For Broadcast the root issues one gin.put() per peer with the
# full message, so -b/-e is the broadcast payload size (not total/NP as in
# AllGather). The segmented suite disables the scatter+allgather and ring large
# tiers so the flat root-fanout GIN path is exercised; the hang guard uses
# default tier selection (ring at 2 GiB).
#
# Opt-in via RCCL_TESTS_GIN_SDMA_BCAST=1 on a GIN-SDMA-capable node. Config:
#   RCCL_TESTS_BCAST_NP, RCCL_TESTS_MPI_LAUNCHER, RCCL_TESTS_MPI_OPTS,
#   RCCL_TESTS_BCAST_XENV, RCCL_TESTS_BCAST_EXE, RCCL_TESTS_BCAST_CTAS,
#   RCCL_TESTS_BCAST_TIMEOUT_S, RCCL_TESTS_BCAST_CONN_RETRIES,
#   RCCL_TESTS_BCAST_GIN_TYPE   NCCL_GIN_TYPE (default: 6). The ANVIL_SDMA enum
#     is 6 on develop but 5 on the NCCL 2.30.7 line, so this must be settable
#     rather than baked in -- a wrong value silently loads no GIN plugin.

_bcast_enabled = os.environ.get("RCCL_TESTS_GIN_SDMA_BCAST", "") not in (
    "",
    "0",
    "false",
    "False",
)

BCAST_NP = env_int("RCCL_TESTS_BCAST_NP", 0) or (
    detect_ngpus() if _bcast_enabled else _detected_ngpus
)
BCAST_LAUNCHER = os.environ.get("RCCL_TESTS_MPI_LAUNCHER", "mpirun")
BCAST_CTAS = os.environ.get("RCCL_TESTS_BCAST_CTAS", "8")
BCAST_GIN_TYPE = os.environ.get("RCCL_TESTS_BCAST_GIN_TYPE", "6")
BCAST_TIMEOUT_S = env_int("RCCL_TESTS_BCAST_TIMEOUT_S", 900)
BCAST_CONN_RETRIES = env_int("RCCL_TESTS_BCAST_CONN_RETRIES", 5)
BCAST_MPI_OPTS = shlex.split(os.environ.get("RCCL_TESTS_MPI_OPTS", ""))
BCAST_XENV = shlex.split(os.environ.get("RCCL_TESTS_BCAST_XENV", ""))
BCAST_EXE = os.environ.get(
    "RCCL_TESTS_BCAST_EXE", os.path.join(path, "..", "build", "broadcast_perf")
)

_bcast_skip = pytest.mark.skipif(
    not _bcast_enabled,
    reason="GIN-SDMA Broadcast tests are opt-in; set RCCL_TESTS_GIN_SDMA_BCAST=1 on "
    "a GIN-SDMA-capable (e.g. 8x MI355X) node to enable.",
)


def _bcast_launch_kwargs():
    return {
        "exe": BCAST_EXE,
        "np": BCAST_NP,
        "launcher": BCAST_LAUNCHER,
        "mpi_opts": BCAST_MPI_OPTS,
        "xenv": BCAST_XENV,
        "ctas": BCAST_CTAS,
        "gin_type": BCAST_GIN_TYPE,
        "timeout_s": BCAST_TIMEOUT_S,
    }


def _run_bcast_gin_sdma(
    request, msg_bytes, dtype, *, force_flat_gin=False, force_sag_gin=False
):
    if BCAST_NP < 2:
        pytest.skip("need >= 2 ranks/GPUs for GIN-SDMA Broadcast")
    return run_bcast_with_conn_gate_retry(
        lambda: launch_bcast_gin_sdma(
            request,
            msg_bytes,
            dtype,
            force_flat_gin=force_flat_gin,
            force_sag_gin=force_sag_gin,
            **_bcast_launch_kwargs(),
        ),
        BCAST_CONN_RETRIES,
    )


@_bcast_skip
@pytest.mark.parametrize("msg_mib", [256, 2048])  # 256 MiB (2 seg), 2 GiB (16 seg)
@pytest.mark.parametrize("dtype", ["int32", "int64", "uint8"])
def test_BroadcastGinSdmaLargeSegmented(request, msg_mib, dtype):
    """Flat root-fanout gin.put() at sizes crossing the 128 MiB SDMA segment."""
    rc, out, debug = _run_bcast_gin_sdma(
        request, msg_mib * MiB, dtype, force_flat_gin=True
    )
    assert_bcast_ok(
        rc,
        out,
        debug,
        "flat-segmented {} MiB dtype={}".format(msg_mib, dtype),
        expect_tier="flat-gin",
    )


@_bcast_skip
def test_BroadcastGinSdmaScatterAllgather(request):
    """256 MiB scatter+allgather tier (ring disabled via RING_MIN_BYTES=0)."""
    rc, out, debug = _run_bcast_gin_sdma(
        request, 256 * MiB, "int32", force_sag_gin=True
    )
    assert_bcast_ok(
        rc, out, debug, "SAG 256 MiB", expect_tier="scatter-allgather"
    )


@_bcast_skip
def test_BroadcastGinSdma2GiBHangGuard(request):
    """2 GiB completion guard with default tier selection (ring path)."""
    rc, out, debug = _run_bcast_gin_sdma(
        request, 2 * GiB, "int32", force_flat_gin=False
    )
    assert_bcast_ok(
        rc,
        out,
        debug,
        "2 GiB default-tier",
        expect_tier=("ring-table", "ring-multi"),
    )


@_bcast_skip
def test_BroadcastGinSdma4GiBHangGuard(request):
    """4 GiB completion guard with default tier selection (ring path at 4 GiB)."""
    rc, out, debug = _run_bcast_gin_sdma(
        request, 4 * GiB, "int32", force_flat_gin=False
    )
    assert_bcast_ok(
        rc,
        out,
        debug,
        "4 GiB default-tier",
        expect_tier=("ring-table", "ring-multi"),
    )


# Offline parsing guards: no GIN hardware required. These pin the regression where
# _DATA_FAIL_RE matched the "#wrong" column header and made the connectivity retry
# unreachable, and where scientific-notation timings broke the row regex.
# Named into `-k GinSdma` (rccl-gin-bcast-pytest) because run-gin-ci.sh expands
# pytest args unquoted, so a filter with a space in it is not available.
def test_BroadcastGinSdmaRowRegexAcceptsScientificNotation():
    line = (
        "  134217728  134217728  int32  none  0"
        "  1.23e+02  1.00e+02  1.00e+02  0"
        "  1.23e+02  1.00e+02  1.00e+02  0"
    )
    rows = bcast_rows(line)
    assert rows == [(134217728, 0, "0", "0")]


def test_BroadcastGinSdmaDataFailedIgnoresColumnHeader():
    header = (
        "#       size         count      type   redop    root"
        "     time   algbw   busbw  #wrong"
    )
    assert not bcast_data_failed(header)


def test_BroadcastGinSdmaDataFailedDetectsNonzeroWrong():
    line = (
        "  1048576  1048576  int32  none  0"
        "  12.34  1.00  1.00  1"
        "  12.34  1.00  1.00  0"
    )
    assert bcast_data_failed(line)


def test_BroadcastGinSdmaDataFailedTreatsNaAsUncheckedNotFailed():
    line = (
        "  1048576  1048576  int32  none  0"
        "  12.34  1.00  1.00  N/A"
        "  12.34  1.00  1.00  0"
    )
    assert not bcast_data_failed(line)


def test_BroadcastGinSdmaConnRetryReturnsCleanPassWithoutRetry():
    calls = []

    def launch():
        calls.append(None)
        return 0, "clean"

    assert run_with_conn_gate_retry(launch, 5, settle_s=0) == (0, "clean")
    assert len(calls) == 1


def test_BroadcastGinSdmaConnRetryExhaustsConnectivityAborts():
    calls = []

    def launch():
        calls.append(None)
        return 1, "LSA signal connectivity gate failed"

    assert run_with_conn_gate_retry(launch, 3, settle_s=0) == (
        1,
        "LSA signal connectivity gate failed",
    )
    assert len(calls) == 3


def test_BroadcastGinSdmaConnRetryNeverRetriesDataFailure():
    calls = []

    def launch():
        calls.append(None)
        return 1, "#wrong = 1"

    assert run_with_conn_gate_retry(launch, 5, settle_s=0) == (1, "#wrong = 1")
    assert len(calls) == 1


def test_BroadcastGinSdmaConnRetryDoesNotRetryNonGateError():
    calls = []

    def launch():
        calls.append(None)
        return 1, "some other error"

    assert run_with_conn_gate_retry(launch, 5, settle_s=0) == (
        1,
        "some other error",
    )
    assert len(calls) == 1


def test_BroadcastGinSdmaDetectNgpusRequiresRocminfo(monkeypatch):
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    from . import gin_sdma_harness as harness

    monkeypatch.setattr(harness.shutil, "which", lambda _cmd: None)
    with pytest.raises(RuntimeError, match="rocminfo"):
        harness.detect_ngpus()


def test_BroadcastGinSdmaPerfArgvMakesAverageCountGapExplicit():
    ag = gin_perf_argv("all_gather_perf", 8, "int32", "8", extra=["-A", "1"])
    a2a = gin_perf_argv("alltoall_perf", 8, "int32", "16")
    assert ag[ag.index("-D") + 2 : ag.index("-V")] == ["-A", "1"]
    assert a2a[a2a.index("-D") + 2 : a2a.index("-V")] == []


def _clean_bcast_output(tier):
    """A minimal healthy -D 3 run reporting `tier`, for the guards below."""
    return "\n".join(
        [
            "#[bcast-tier] {}".format(tier),
            "  268435456  268435456  int32  none  0"
            "  12.34  1.00  1.00  0"
            "  12.34  1.00  1.00  0",
            "# Out of bounds values : 0 OK",
        ]
    )


# These four need no GPU, but they carry GinSdma in the name on purpose: the CI
# job runs `-k GinSdma` (rccl-gin-bcast-pytest in gin-tests.json), and the runner
# expands its args unquoted, so a filter with a space in it is not available.
def test_BroadcastGinSdmaTierGuardLineIsNotAResultsRow():
    """The tier line shares stdout with the table it must not corrupt."""
    assert bcast_rows("#[bcast-tier] scatter-allgather") == []
    assert bcast_tiers(_clean_bcast_output("ring-table")) == {"ring-table"}
    assert bcast_rows(_clean_bcast_output("ring-table"))


def test_BroadcastGinSdmaTierGuardAcceptsExpectedTier():
    out = _clean_bcast_output("scatter-allgather")
    assert_bcast_ok(0, out, "gin-anvil-sdma", "guard", expect_tier="scatter-allgather")
    # Ring passes against either decomposition outcome.
    assert_bcast_ok(
        0,
        _clean_bcast_output("ring-multi"),
        "gin-anvil-sdma",
        "guard",
        expect_tier=("ring-table", "ring-multi"),
    )


def test_BroadcastGinSdmaTierGuardRejectsFallThrough():
    """The regression this tier check exists for.

    Deleting the scatter+allgather arm sends its messages to the flat kernel,
    which still broadcasts correctly and still reports #wrong 0. Everything
    except the tier check passes on that output, so without this the SAG test
    would go green while covering nothing.
    """
    fell_through = _clean_bcast_output("flat-gin")
    assert_bcast_ok(0, fell_through, "gin-anvil-sdma", "guard", expect_tier="flat-gin")
    with pytest.raises(AssertionError, match="expected exactly one of"):
        assert_bcast_ok(
            0, fell_through, "gin-anvil-sdma", "guard", expect_tier="scatter-allgather"
        )


def test_BroadcastGinSdmaTierGuardRejectsMissingTierLine():
    """A binary predating the tier log must fail loudly, not pass vacuously."""
    no_tier = "\n".join(_clean_bcast_output("flat-gin").splitlines()[1:])
    with pytest.raises(AssertionError, match="no .*bcast-tier.* line"):
        assert_bcast_ok(
            0, no_tier, "gin-anvil-sdma", "guard", expect_tier="flat-gin"
        )


@pytest.mark.parametrize(
    "nthreads, nprocs, ngpus_mpi, byte_range, op, step_factor, datatype",
    itertools.product(
        nthreads, nprocs, ngpus_mpi, byte_range, op, step_factor, datatype
    ),
)
def test_BroadcastMPI(
    request, nthreads, nprocs, ngpus_mpi, byte_range, op, step_factor, datatype
):
    try:
        mpi_hostfile = request.config.getoption("--hostfile")
        if not mpi_hostfile:
            args = [
                "mpirun -np",
                nprocs,
                executable,
                "-p 1",
                "-t",
                nthreads,
                "-g",
                ngpus_mpi,
                "-b",
                byte_range[0],
                "-e",
                byte_range[1],
                "-o",
                op,
                "-f",
                step_factor,
                "-d",
                datatype,
            ]
        else:
            args = [
                "mpirun -np",
                nprocs,
                "-host",
                mpi_hostfile,
                executable,
                "-p 1",
                "-t",
                nthreads,
                "-g",
                ngpus_mpi,
                "-b",
                byte_range[0],
                "-e",
                byte_range[1],
                "-o",
                op,
                "-f",
                step_factor,
                "-d",
                datatype,
            ]
        args_str = " ".join(args)
        print(args_str)
        rccl_test = subprocess.run(args_str, universal_newlines=True, shell=True)
    except subprocess.CalledProcessError as err:
        print(rccl_test.stdout)
        pytest.fail("Broadcast test error(s) detected.")

    assert rccl_test.returncode == 0
