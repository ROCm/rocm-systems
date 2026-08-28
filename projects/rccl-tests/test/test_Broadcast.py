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
import re
import shlex
import signal
import subprocess
import itertools
import math
import time

import pytest

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
executable = path + "/../build/broadcast_perf"

@pytest.mark.parametrize("nthreads, ngpus_single, byte_range, op, step_factor, datatype, memory_type",
    itertools.product(nthreads, ngpus_single, byte_range, op, step_factor, datatype, memory_type))
def test_BroadcastSingleProcess(nthreads, ngpus_single, byte_range, op, step_factor, datatype, memory_type):
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
        pytest.fail("Broadcast test error(s) detected.")

    assert rccl_test.returncode == 0

# ---------------------------------------------------------------------------
# GIN-SDMA Broadcast multi-segment regression tests (parity with AllGather /
# AllToAll in test_AllGather.py / test_AllToAll.py).
#
# These drive the real GinHybridBroadcastKernel (deviceImpl 3, NCCL_GIN_TYPE=5)
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
#   RCCL_TESTS_BCAST_TIMEOUT_S, RCCL_TESTS_BCAST_CONN_RETRIES

MiB = 1024 * 1024
GiB = 1024 * MiB

_bcast_enabled = os.environ.get("RCCL_TESTS_GIN_SDMA_BCAST", "") not in ("", "0", "false", "False")

BCAST_NP = int(os.environ.get("RCCL_TESTS_BCAST_NP", "0")) or ngpus
BCAST_LAUNCHER = os.environ.get("RCCL_TESTS_MPI_LAUNCHER", "mpirun")
BCAST_CTAS = os.environ.get("RCCL_TESTS_BCAST_CTAS", "8")
BCAST_TIMEOUT_S = int(os.environ.get("RCCL_TESTS_BCAST_TIMEOUT_S", "900"))
BCAST_CONN_RETRIES = int(os.environ.get("RCCL_TESTS_BCAST_CONN_RETRIES", "5"))
BCAST_MPI_OPTS = shlex.split(os.environ.get("RCCL_TESTS_MPI_OPTS", ""))
BCAST_XENV = shlex.split(os.environ.get("RCCL_TESTS_BCAST_XENV", ""))
BCAST_EXE = os.environ.get(
    "RCCL_TESTS_BCAST_EXE", os.path.join(path, "..", "build", "broadcast_perf"))

_bcast_skip = pytest.mark.skipif(
    not _bcast_enabled,
    reason="GIN-SDMA Broadcast tests are opt-in; set RCCL_TESTS_GIN_SDMA_BCAST=1 on "
           "a GIN-SDMA-capable (e.g. 8x MI355X) node to enable.")

_CONN_GATE_RE = re.compile(r"LSA signal connectivity gate failed|unhandled system error", re.I)
_DATA_FAIL_RE = re.compile(r"Wrong|mismatch|check.*fail|Out of bounds values\s*:\s*[1-9]", re.I)


def _launch_bcast_gin_sdma(request, msg_bytes, dtype, *, force_flat_gin):
    """Launch broadcast_perf -D 3 at a fixed message size. When force_flat_gin,
    disable the SAG/ring large tiers so the root flat gin.put() path is used."""
    size = str(int(msg_bytes))
    gin_env = ["NCCL_GIN_ENABLE=1", "NCCL_GIN_TYPE=5",
               "NCCL_GIN_ANVIL_SDMA_THRESHOLD=0",
               "NCCL_GIN_ANVIL_SDMA_THRESHOLD_BROADCAST=0"]
    if force_flat_gin:
        gin_env += ["NCCL_GIN_ANVIL_BCAST_SCATTER_AG_MIN_BYTES=0",
                    "NCCL_GIN_ANVIL_BCAST_RING_MIN_BYTES=0"]
    gin_env += BCAST_XENV
    gin_env = sum([["-x", kv] for kv in gin_env], [])

    hostfile = request.config.getoption("--hostfile")
    launch = [BCAST_LAUNCHER, "-np", str(BCAST_NP)] + BCAST_MPI_OPTS
    if hostfile:
        launch += ["-host", hostfile]

    args = launch + gin_env + [
        BCAST_EXE,
        "-b", size, "-e", size,
        "-f", "2",
        "-g", "1",
        "-R", "2",
        "-D", "3",       # GinHybridBroadcastKernel
        "-A", "1",
        "-V", BCAST_CTAS,
        "-d", dtype,
        "-c", "1",
        "-w", "1",
        "-n", "3",
    ]
    cmd = " ".join(shlex.quote(a) for a in args)
    print(cmd)
    proc = subprocess.Popen(cmd, shell=True, universal_newlines=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            start_new_session=True)
    try:
        out, _ = proc.communicate(timeout=BCAST_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        out, _ = proc.communicate()
        pytest.fail(
            "Broadcast GIN-SDMA HANG: no completion within {}s at msg={} bytes "
            "({} MiB), dtype={}. Output tail:\n{}".format(
                BCAST_TIMEOUT_S, size, msg_bytes // MiB, dtype, (out or "")[-2000:]))
    print(out)
    return proc.returncode, out


def _run_bcast_gin_sdma(request, msg_bytes, dtype, *, force_flat_gin):
    if BCAST_NP < 2:
        pytest.skip("need >= 2 ranks/GPUs for GIN-SDMA Broadcast")

    rc, out = 1, ""
    for attempt in range(1, max(1, BCAST_CONN_RETRIES) + 1):
        rc, out = _launch_bcast_gin_sdma(request, msg_bytes, dtype,
                                         force_flat_gin=force_flat_gin)
        if rc == 0:
            return rc, out
        if _DATA_FAIL_RE.search(out or ""):
            return rc, out
        if _CONN_GATE_RE.search(out or "") and attempt < BCAST_CONN_RETRIES:
            print("=== connectivity-gate abort (attempt {}/{}); re-launching after settle ===".format(
                attempt, BCAST_CONN_RETRIES))
            time.sleep(3)
            continue
        return rc, out
    return rc, out


@_bcast_skip
@pytest.mark.parametrize("msg_mib", [256, 2048])  # 256 MiB (2 seg), 2 GiB (16 seg)
@pytest.mark.parametrize("dtype", ["int32", "int64", "uint8"])
def test_BroadcastGinSdmaLargeSegmented(request, msg_mib, dtype):
    """Flat root-fanout gin.put() at sizes crossing the 128 MiB SDMA segment."""
    rc, _ = _run_bcast_gin_sdma(request, msg_mib * MiB, dtype, force_flat_gin=True)
    assert rc == 0, "Broadcast data check failed (nonzero exit) at {} MiB, dtype={}".format(
        msg_mib, dtype)


@_bcast_skip
def test_BroadcastGinSdma2GiBHangGuard(request):
    """2 GiB completion guard with default tier selection (ring path)."""
    rc, _ = _run_bcast_gin_sdma(request, 2 * GiB, "int32", force_flat_gin=False)
    assert rc == 0, "Broadcast 2 GiB data check failed (nonzero exit)"


@_bcast_skip
def test_BroadcastGinSdma4GiBHangGuard(request):
    """4 GiB completion guard with default tier selection (ring / SAG path)."""
    rc, _ = _run_bcast_gin_sdma(request, 4 * GiB, "int32", force_flat_gin=False)
    assert rc == 0, "Broadcast 4 GiB data check failed (nonzero exit)"


@pytest.mark.parametrize("nthreads, nprocs, ngpus_mpi, byte_range, op, step_factor, datatype",
    itertools.product(nthreads, nprocs, ngpus_mpi, byte_range, op, step_factor, datatype))
def test_BroadcastMPI(request, nthreads, nprocs, ngpus_mpi, byte_range, op, step_factor, datatype):
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
        pytest.fail("Broadcast test error(s) detected.")

    assert rccl_test.returncode == 0
