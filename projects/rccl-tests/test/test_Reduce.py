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
    MiB,
    detect_ngpus,
    env_int,
    gin_env_xflags,
    gin_hang_msg,
    gin_perf_argv,
    launch_mpi_shell,
    mpi_launch_prefix,
    run_with_conn_gate_retry,
)

ngpus = 0
if os.environ.get('ROCR_VISIBLE_DEVICES') is not None:
    ngpus = len(os.environ['ROCR_VISIBLE_DEVICES'].split(","))
elif os.environ.get('HIP_VISIBLE_DEVICES') is not None:
    ngpus = len(os.environ['HIP_VISIBLE_DEVICES'].split(","))
else:
    ngpus = int(subprocess.check_output("rocminfo | grep \"Device Type:.\s*.GPU\" | wc -l",shell=True))
log_ngpus = int(math.log2(ngpus))

nthreads = ["1"]
nprocs = ["2"]
ngpus_single = [str(2**x) for x in range(log_ngpus+1)]
ngpus_mpi = ["1","2"]
byte_range = [("4", "128M")]
op = ["sum", "prod", "min", "max"]
step_factor = ["2"]
datatype = ["int8", "uint8", "int32", "uint32", "int64", "uint64", "half", "float", "double"]
memory_type = ["coarse","fine", "host"]

path = os.path.dirname(os.path.abspath(__file__))
executable = path + "/../build/reduce_perf"

@pytest.mark.parametrize("nthreads, ngpus_single, byte_range, op, step_factor, datatype, memory_type",
    itertools.product(nthreads, ngpus_single, byte_range, op, step_factor, datatype, memory_type))
def test_ReduceSingleProcess(nthreads, ngpus_single, byte_range, op, step_factor, datatype, memory_type):
    try:
        args = [executable,
                "-t", nthreads,
                "-g", ngpus_single,
                "-b", byte_range[0],
                "-e", byte_range[1],
                "-o", op,
                "-f", step_factor,
                "-d", datatype,
                "-Y", memory_type]
        if memory_type == "fine":
            args.insert(0, "HSA_FORCE_FINE_GRAIN_PCIE=1")
        args_str = " ".join(args)
        rccl_test = subprocess.run(args_str, stdout=subprocess.PIPE, universal_newlines=True, shell=True)
    except subprocess.CalledProcessError as err:
        print(rccl_test.stdout)
        pytest.fail("Reduce test error(s) detected.")

    assert rccl_test.returncode == 0

@pytest.mark.parametrize("nthreads, nprocs, ngpus_mpi, byte_range, op, step_factor, datatype",
    itertools.product(nthreads, nprocs, ngpus_mpi, byte_range, op, step_factor, datatype))
def test_ReduceMPI(request, nthreads, nprocs, ngpus_mpi, byte_range, op, step_factor, datatype):
    try:
        mpi_hostfile = request.config.getoption('--hostfile')
        if not mpi_hostfile:
            args = ["mpirun -np", nprocs,
                    executable,
                    "-p 1",
                    "-t", nthreads,
                    "-g", ngpus_mpi,
                    "-b", byte_range[0],
                    "-e", byte_range[1],
                    "-o", op,
                    "-f", step_factor,
                    "-d", datatype]
        else:
            args = ["mpirun -np", nprocs,
                    "-host", mpi_hostfile,
                    executable,
                    "-p 1",
                    "-t", nthreads,
                    "-g", ngpus_mpi,
                    "-b", byte_range[0],
                    "-e", byte_range[1],
                    "-o", op,
                    "-f", step_factor,
                    "-d", datatype]
        args_str = " ".join(args)
        print(args_str)
        rccl_test = subprocess.run(args_str, universal_newlines=True, shell=True)
    except subprocess.CalledProcessError as err:
        print(rccl_test.stdout)
        pytest.fail("Reduce test error(s) detected.")

    assert rccl_test.returncode == 0


# ---------------------------------------------------------------------------
# GIN-SDMA Reduce (deviceImpl 3) correctness.
#
# GinReduceKernel folds every peer's slice in ascending rank order and stores
# the result into the root's recvbuff; the perf binary's data check (-c 1) exits
# non-zero on any wrong element. Sizes straddle the kernel's 48 MiB switch from
# the grid-stride schedule to the warp-unrolled one, and roots cover both ends
# of the rank range so the root-directed store is exercised off rank 0.
#
# Requires a single node with all ranks LSA-reachable, an MPI launcher, and a
# GIN-SDMA-capable RCCL build, so it is skipped unless RCCL_TESTS_GIN_SDMA_REDUCE
# is set. Configuration:
#   RCCL_TESTS_REDUCE_NP         MPI ranks (default: detected GPU count)
#   RCCL_TESTS_MPI_LAUNCHER      launcher binary (default: mpirun)
#   RCCL_TESTS_MPI_OPTS          extra launcher opts
#   RCCL_TESTS_REDUCE_XENV       extra "-x K=V" env the backend needs
#   RCCL_TESTS_REDUCE_EXE        path to reduce_perf (default: ../build/reduce_perf)
#   RCCL_TESTS_REDUCE_CTAS       device CTA count (-V) (default: 16)
#   RCCL_TESTS_REDUCE_TIMEOUT_S  per-run hang timeout in seconds (default: 600)
#   RCCL_TESTS_REDUCE_CONN_RETRIES  re-launches on the intermittent connectivity
#                                gate abort (default: 5); data failures never retry.

_reduce_gin_enabled = os.environ.get("RCCL_TESTS_GIN_SDMA_REDUCE", "") not in ("", "0", "false", "False")

REDUCE_NP = env_int("RCCL_TESTS_REDUCE_NP", 0) or (detect_ngpus() if _reduce_gin_enabled else ngpus)
REDUCE_LAUNCHER = os.environ.get("RCCL_TESTS_MPI_LAUNCHER", "mpirun")
REDUCE_CTAS = os.environ.get("RCCL_TESTS_REDUCE_CTAS", "16")
REDUCE_TIMEOUT_S = env_int("RCCL_TESTS_REDUCE_TIMEOUT_S", 600)
REDUCE_CONN_RETRIES = env_int("RCCL_TESTS_REDUCE_CONN_RETRIES", 5)
REDUCE_MPI_OPTS = shlex.split(os.environ.get("RCCL_TESTS_MPI_OPTS", ""))
REDUCE_XENV = shlex.split(os.environ.get("RCCL_TESTS_REDUCE_XENV", ""))
REDUCE_EXE = os.environ.get("RCCL_TESTS_REDUCE_EXE", executable)

_reduce_gin_skip = pytest.mark.skipif(
    not _reduce_gin_enabled,
    reason="GIN-SDMA Reduce tests are opt-in; set RCCL_TESTS_GIN_SDMA_REDUCE=1 on "
           "a single-node GIN-SDMA-capable machine to enable.")


def _run_reduce_gin_sdma(request, total_bytes, dtype, op, root):
    if REDUCE_NP < 2:
        pytest.skip("need >= 2 ranks/GPUs for GIN-SDMA Reduce")
    size = str(int(total_bytes))
    gin_kv = ["NCCL_GIN_ENABLE=1", "NCCL_GIN_TYPE=7"] + REDUCE_XENV
    args = (
        mpi_launch_prefix(request, REDUCE_LAUNCHER, REDUCE_NP, REDUCE_MPI_OPTS)
        + gin_env_xflags(gin_kv)
        + gin_perf_argv(REDUCE_EXE, size, dtype, REDUCE_CTAS,
                        extra=["-o", op, "-r", str(root)])
    )
    cmd = " ".join(shlex.quote(a) for a in args)
    hang_msg = gin_hang_msg(
        "Reduce", REDUCE_TIMEOUT_S, size, dtype, "{} MiB total".format(total_bytes // MiB))
    return run_with_conn_gate_retry(
        lambda: launch_mpi_shell(cmd, REDUCE_TIMEOUT_S, hang_msg),
        REDUCE_CONN_RETRIES,
    )


@_reduce_gin_skip
@pytest.mark.parametrize("total_mib", [1, 64])
@pytest.mark.parametrize("dtype, op", [
    ("int32", "sum"),
    ("float", "sum"),
    ("half", "max"),
    ("bfloat16", "avg"),
    ("double", "prod"),
])
@pytest.mark.parametrize("root_pos", ["first", "last"])
def test_ReduceGinSdma(request, total_mib, dtype, op, root_pos):
    root = 0 if root_pos == "first" else REDUCE_NP - 1
    rc, _ = _run_reduce_gin_sdma(request, total_mib * MiB, dtype, op, root)
    assert rc == 0, "Reduce -D 3 data check failed at {} MiB, dtype={}, op={}, root={}".format(
        total_mib, dtype, op, root)
