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
import time
import itertools
import math

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
executable = path + "/../build/all_gather_perf"

@pytest.mark.parametrize("nthreads, ngpus_single, byte_range, op, step_factor, datatype, memory_type",
    itertools.product(nthreads, ngpus_single, byte_range, op, step_factor, datatype, memory_type))
def test_AllGatherSingleProcess(nthreads, ngpus_single, byte_range, op, step_factor, datatype, memory_type):
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
        pytest.fail("AllGather test error(s) detected.")

    assert rccl_test.returncode == 0


# ---------------------------------------------------------------------------
# GIN-SDMA AllGather multi-segment regression tests (parity with the >1 GiB
# SDMA hang fix validated for AllToAll in test_AllToAll.py / PR #9927).
#
# These drive the real GinHybridAllGatherKernel (deviceImpl 3, NCCL_GIN_TYPE=5)
# at per-rank chunk sizes that cross the two single-descriptor limits the
# 128 MiB SDMA copy clamp guards. That clamp lives in the Anvil-SDMA backend
# Put (ncclGinApi_Put<NCCL_NET_DEVICE_GIN_ANVIL_SDMA>), which segments every
# gin.put() into <=128 MiB SDMA copies with the completion signal carried only
# on the FINAL segment. Correctness of the single-final-signal scheme relies on
# every segment of a message serializing on one in-order SDMA queue; the backend
# forces a single SDMA channel (numChannels==1, NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS
# is ignored), so this holds structurally. The AllGather kernel just calls plain
# gin.put(), so it inherits the segmentation + single-channel guarantee.
#
# For AllGather, -e is the TOTAL gathered size and each rank contributes
# total/NP bytes, so the per-rank gin.put() to every peer is total/NP:
#
#   1. per-rank > 128 MiB -- exercises the multi-segment put loop.
#   2. per-rank >   1 GiB -- more than the 30-bit (1 GiB) single-copy count
#                            bound would allow unsegmented; each 128 MiB segment
#                            stays well under it, so a stale/truncated tail here
#                            would surface as a data-check failure.
#   3. 2 GiB total        -- multi-segment completion guard; a subprocess timeout
#                            turns a reintroduced SDMA hang into a test failure
#                            instead of stalling the runner forever.
#
# Exact-integer datatypes are used so a truncated or stale tail cannot be masked
# by floating-point tolerance; the perf binary's built-in data check (-c 1) sets
# a non-zero exit code on any wrong element. The SDMA (large) tier is forced for
# every size via NCCL_GIN_ANVIL_SDMA_THRESHOLD=0.
#
# These require an 8x MI355X (or similar) node, an MPI launcher, and a
# GIN-SDMA-capable RCCL build, so they are skipped unless RCCL_TESTS_GIN_SDMA_AG
# is set. Configuration mirrors test_AllToAll.py:
#   RCCL_TESTS_AG_NP         MPI ranks (default: detected GPU count)
#   RCCL_TESTS_MPI_LAUNCHER  launcher binary (default: mpirun)
#   RCCL_TESTS_MPI_OPTS      extra launcher opts (e.g. --allow-run-as-root -mca ...)
#   RCCL_TESTS_AG_XENV       extra "-x K=V" env the backend needs on this cluster
#   RCCL_TESTS_AG_EXE        path to all_gather_perf (default: ../build/all_gather_perf)
#   RCCL_TESTS_AG_CTAS       device CTA count (-V) (default: 8)
#   RCCL_TESTS_AG_TIMEOUT_S  per-run hang timeout in seconds (default: 900)
#   RCCL_TESTS_AG_CONN_RETRIES  re-launches on the intermittent gfx950 cuMem-VMM
#                            connectivity-gate abort (default: 5); a genuine data
#                            check mismatch is never retried.
#
# Verified on 8x MI355X (ROCm 7.13, NCCL_GIN_TYPE=5, force-single-channel):
# 256/512 MiB per-rank (2/4 segments) and 2 GiB total all pass with #wrong=0,
# no hang (busbw ~421-427 GB/s).

MiB = 1024 * 1024
GiB = 1024 * MiB

_ag_enabled = os.environ.get("RCCL_TESTS_GIN_SDMA_AG", "") not in ("", "0", "false", "False")

AG_NP = int(os.environ.get("RCCL_TESTS_AG_NP", "0")) or ngpus
AG_LAUNCHER = os.environ.get("RCCL_TESTS_MPI_LAUNCHER", "mpirun")
AG_CTAS = os.environ.get("RCCL_TESTS_AG_CTAS", "8")
AG_TIMEOUT_S = int(os.environ.get("RCCL_TESTS_AG_TIMEOUT_S", "900"))
AG_CONN_RETRIES = int(os.environ.get("RCCL_TESTS_AG_CONN_RETRIES", "5"))
AG_MPI_OPTS = shlex.split(os.environ.get("RCCL_TESTS_MPI_OPTS", ""))
AG_XENV = shlex.split(os.environ.get("RCCL_TESTS_AG_XENV", ""))
AG_EXE = os.environ.get(
    "RCCL_TESTS_AG_EXE", os.path.join(path, "..", "build", "all_gather_perf"))

_ag_skip = pytest.mark.skipif(
    not _ag_enabled,
    reason="GIN-SDMA AllGather tests are opt-in; set RCCL_TESTS_GIN_SDMA_AG=1 on "
           "a GIN-SDMA-capable (e.g. 8x MI355X) node to enable.")


# Intermittent gfx950 cuMem-VMM connectivity-gate abort (not a data error): the
# LSA signal connectivity self-test fails to map a peer aperture and the run
# aborts before any collective work. Re-launched after a short settle, exactly
# as the shell harness does. A genuine wrong-element data check is never retried.
_CONN_GATE_RE = re.compile(r"LSA signal connectivity gate failed|unhandled system error", re.I)
_DATA_FAIL_RE = re.compile(r"Wrong|mismatch|check.*fail|Out of bounds values\s*:\s*[1-9]", re.I)
_AG_DEVTIME_TIER_RE = re.compile(r"#\[ag-devtime\].*tier\s+(LSA|SDMA)", re.I)


def _launch_ag_gin_sdma(request, total_bytes, dtype, force_sdma_tier=True, device_timing=False):
    """Launch all_gather_perf -D 3 once at a fixed TOTAL gathered size (per-rank
    chunk = total/NP). When force_sdma_tier is True (default), threshold=0 pins the
    SDMA tier for every size. Returns (returncode, stdout). A timeout (hang) fails
    the test immediately."""
    size = str(int(total_bytes))
    gin_env = []
    for kv in ["NCCL_GIN_ENABLE=1", "NCCL_GIN_TYPE=5"] + AG_XENV:
        gin_env += ["-x", kv]
    if force_sdma_tier:
        for kv in ["NCCL_GIN_ANVIL_SDMA_THRESHOLD=0",
                   "NCCL_GIN_ANVIL_SDMA_THRESHOLD_ALLGATHER=0"]:
            gin_env += ["-x", kv]

    hostfile = request.config.getoption("--hostfile")
    launch = [AG_LAUNCHER, "-np", str(AG_NP)] + AG_MPI_OPTS
    if hostfile:
        launch += ["-host", hostfile]

    args = launch + gin_env + [
        AG_EXE,
        "-b", size, "-e", size,
        "-f", "2",
        "-g", "1",
        "-R", "2",
        "-D", "3",       # GinHybridAllGatherKernel
        "-A", "1",       # emit algo/proto/nchannels columns
        "-V", AG_CTAS,
        "-d", dtype,
        "-c", "1",       # data check: nonzero exit on any wrong element
        "-w", "1",
        "-n", "3",
    ]
    if device_timing:
        args += ["-B", "1"]  # augment stdout with #[ag-devtime] tier line
    cmd = " ".join(shlex.quote(a) for a in args)
    print(cmd)
    # New session so a hang can be killed as a whole process group -- otherwise a
    # wedged launcher leaves orphaned ranks pinning the GPUs (and exhausting SDMA
    # queues for later tests).
    proc = subprocess.Popen(cmd, shell=True, universal_newlines=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            start_new_session=True)
    try:
        out, _ = proc.communicate(timeout=AG_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        out, _ = proc.communicate()
        pytest.fail(
            "AllGather GIN-SDMA HANG: no completion within {}s at total={} bytes "
            "({} MiB/rank), dtype={}. Output tail:\n{}".format(
                AG_TIMEOUT_S, size, total_bytes // AG_NP // MiB, dtype, (out or "")[-2000:]))
    print(out)
    return proc.returncode, out


def _run_ag_gin_sdma(request, total_bytes, dtype, **launch_kw):
    """Launch with connectivity-gate retry. Returns (returncode, stdout) of the
    first run that either succeeds, hits a genuine data-check failure, or fails
    for a non-connectivity-gate reason; otherwise the last attempt."""
    if AG_NP < 2:
        pytest.skip("need >= 2 ranks/GPUs for GIN-SDMA AllGather")

    rc, out = 1, ""
    for attempt in range(1, max(1, AG_CONN_RETRIES) + 1):
        rc, out = _launch_ag_gin_sdma(request, total_bytes, dtype, **launch_kw)
        if rc == 0:
            return rc, out
        # A real wrong-element mismatch must fail now, never retried.
        if _DATA_FAIL_RE.search(out or ""):
            return rc, out
        # Intermittent gfx950 cuMem-VMM connectivity-gate abort: settle + retry.
        if _CONN_GATE_RE.search(out or "") and attempt < AG_CONN_RETRIES:
            print("=== connectivity-gate abort (attempt {}/{}); re-launching after settle ===".format(
                attempt, AG_CONN_RETRIES))
            time.sleep(3)
            continue
        return rc, out
    return rc, out


# (items 1 + 2) Multi-segment loop and the 1 GiB single-copy boundary. per_rank
# is each rank's contribution (the gin.put() to every peer); total = per_rank*NP.
@_ag_skip
@pytest.mark.parametrize("per_rank_mib", [256, 2048])  # 256 MiB (2 seg), 2 GiB (16 seg)
@pytest.mark.parametrize("dtype", ["int32", "int64", "uint8"])
def test_AllGatherGinSdmaLargeSegmented(request, per_rank_mib, dtype):
    total = per_rank_mib * MiB * AG_NP
    rc, _ = _run_ag_gin_sdma(request, total, dtype)
    assert rc == 0, "AllGather data check failed (nonzero exit) at {} MiB/rank, dtype={}".format(
        per_rank_mib, dtype)


# (item 3) 4 GiB-total completion guard. At 8 ranks this is 512 MiB/rank -- distinct
# from the 256 MiB/rank parametrized case above; a reintroduced hang trips the timeout.
@_ag_skip
def test_AllGatherGinSdma4GiBTotalHangGuard(request):
    rc, _ = _run_ag_gin_sdma(request, 4 * GiB, "int32")
    assert rc == 0, "AllGather 4 GiB-total data check failed (nonzero exit)"


# Default-threshold tier crossover (compiled 32 KiB/rank): 16 KiB/rank -> LSA,
# 64 KiB/rank -> SDMA. Does not force threshold=0 so the compiled default applies.
@_ag_skip
def test_AllGatherGinSdmaTierCrossover(request):
    cases = [
        (AG_NP * 16 * 1024, "LSA"),   # 16 KiB/rank <= 32 KiB default
        (AG_NP * 64 * 1024, "SDMA"),  # 64 KiB/rank > 32 KiB default
    ]
    for total, expect_tier in cases:
        rc, out = _run_ag_gin_sdma(
            request, total, "int8", force_sdma_tier=False, device_timing=True)
        assert rc == 0, "AllGather tier crossover failed at total={} bytes".format(total)
        m = _AG_DEVTIME_TIER_RE.search(out or "")
        assert m, "missing #[ag-devtime] tier line at total={} bytes".format(total)
        assert m.group(1).upper() == expect_tier, (
            "expected tier {} at total={} bytes, got {} in:\n{}".format(
                expect_tier, total, m.group(1), (out or "")[-1500:]))


@pytest.mark.parametrize("nthreads, nprocs, ngpus_mpi, byte_range, op, step_factor, datatype",
    itertools.product(nthreads, nprocs, ngpus_mpi, byte_range, op, step_factor, datatype))
def test_AllGatherMPI(request, nthreads, nprocs, ngpus_mpi, byte_range, op, step_factor, datatype):
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
        pytest.fail("AllGather test error(s) detected.")

    assert rccl_test.returncode == 0
