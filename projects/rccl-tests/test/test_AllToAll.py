#################################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

# GIN-SDMA AllToAll regression tests for the >1 GiB SDMA hang fix (PR #9927).
#
# These drive the real GinHybridAlltoAllKernel (deviceImpl 3, NCCL_GIN_TYPE=6)
# at per-peer transfer sizes that cross the two single-descriptor limits the
# 128 MiB SDMA copy clamp guards. The clamp now lives in the Anvil-SDMA backend
# Put (ncclGinApi_Put<NCCL_NET_DEVICE_GIN_ANVIL_SDMA>), which segments every
# gin.put() into <=128 MiB SDMA copies; the kernels just call plain gin.put():
#
#   1. >128 MiB/peer  -- exercises the new multi-segment put loop.
#   2. >1 GiB/peer    -- crosses the 30-bit (1 GiB) count-field boundary, where
#                        an unclamped single put would silently truncate.
#   3. 2 GiB total    -- the original hang repro (256 MiB/peer at 8 ranks); a
#                        subprocess timeout turns a reintroduced hang into a
#                        test failure instead of stalling the runner forever.
#
# Exact-integer datatypes are used so a truncated or stale tail cannot be masked
# by floating-point tolerance; the perf binary's built-in data check (-c 1) sets
# a non-zero exit code on any wrong element.
#
# These require an 8x MI355X (or similar) node, an MPI launcher, and a
# GIN-SDMA-capable RCCL build, so the module is skipped unless
# RCCL_TESTS_GIN_SDMA_A2A is set in the environment. Configuration is taken from
# environment variables (see below) with defaults matching the 8-GPU repro:
#   RCCL_TESTS_A2A_NP        MPI ranks (default: detected GPU count)
#   RCCL_TESTS_MPI_LAUNCHER  launcher binary (default: mpirun)
#   RCCL_TESTS_MPI_OPTS      extra launcher opts (e.g. --allow-run-as-root -mca ...)
#   RCCL_TESTS_A2A_XENV      extra "-x K=V" env the backend needs on this cluster
#   RCCL_TESTS_A2A_EXE       path to alltoall_perf (default: ../build/alltoall_perf)
#   RCCL_TESTS_A2A_CTAS      device CTA count (-V) (default: 16)
#   RCCL_TESTS_A2A_TIMEOUT_S per-run hang timeout in seconds (default: 900)
#   RCCL_TESTS_A2A_CONN_RETRIES  connectivity-gate retries (default: 5)
#
# Verified on 8x MI355X (ROCm 7.13, NCCL_GIN_TYPE=6, force1ch): 256 MiB/peer and
# 2 GiB/peer int32 and the 2 GiB-total guard all pass with #wrong=0, no hang
# (busbw ~424-428 GB/s).

import os
import shlex

import pytest

from .gin_sdma_harness import (
    GiB,
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

path = os.path.dirname(os.path.abspath(__file__))
executable = os.environ.get(
    "RCCL_TESTS_A2A_EXE", os.path.join(path, "..", "build", "alltoall_perf")
)

_enabled = os.environ.get("RCCL_TESTS_GIN_SDMA_A2A", "") not in (
    "",
    "0",
    "false",
    "False",
)

NP = env_int("RCCL_TESTS_A2A_NP", 0) or (detect_ngpus() if _enabled else 0)
LAUNCHER = os.environ.get("RCCL_TESTS_MPI_LAUNCHER", "mpirun")
CTAS = os.environ.get("RCCL_TESTS_A2A_CTAS", "16")
TIMEOUT_S = env_int("RCCL_TESTS_A2A_TIMEOUT_S", 900)
CONN_RETRIES = env_int("RCCL_TESTS_A2A_CONN_RETRIES", 5)
MPI_OPTS = shlex.split(os.environ.get("RCCL_TESTS_MPI_OPTS", ""))
XENV = shlex.split(os.environ.get("RCCL_TESTS_A2A_XENV", ""))

pytestmark = pytest.mark.skipif(
    not _enabled,
    reason="GIN-SDMA AllToAll tests are opt-in; set RCCL_TESTS_GIN_SDMA_A2A=1 on "
    "a GIN-SDMA-capable (e.g. 8x MI355X) node to enable.",
)


def _launch_a2a(request, total_bytes, dtype):
    """Launch alltoall_perf -D 3 once at a fixed total (per-rank) size."""
    size = str(int(total_bytes))
    gin_env = gin_env_xflags(
        [
            "NCCL_GIN_ENABLE=1",
            "NCCL_GIN_TYPE=6",
            "NCCL_GIN_ANVIL_SDMA_THRESHOLD=0",
            "NCCL_GIN_ANVIL_SDMA_THRESHOLD_ALLTOALL=0",
        ]
        + XENV
    )

    args = (
        mpi_launch_prefix(request, LAUNCHER, NP, MPI_OPTS)
        + gin_env
        + gin_perf_argv(executable, size, dtype, CTAS)
    )
    cmd = " ".join(shlex.quote(a) for a in args)
    hang_msg = gin_hang_msg(
        "AllToAll",
        TIMEOUT_S,
        size,
        dtype,
        "{} MiB/peer".format(total_bytes // NP // MiB),
    )
    return launch_mpi_shell(cmd, TIMEOUT_S, hang_msg)


def _run_a2a(request, total_bytes, dtype):
    """Run one all-to-all with connectivity-gate retry on gfx950 VMM aborts."""
    if NP < 2:
        pytest.skip("need >= 2 ranks/GPUs for AllToAll")
    return run_with_conn_gate_retry(
        lambda: _launch_a2a(request, total_bytes, dtype), CONN_RETRIES
    )


@pytest.mark.parametrize("per_peer_mib", [256, 2048])
@pytest.mark.parametrize("dtype", ["int32", "int64", "uint8"])
def test_AllToAllGinSdmaLargeSegmented(request, per_peer_mib, dtype):
    total = per_peer_mib * MiB * NP
    rc, _ = _run_a2a(request, total, dtype)
    assert rc == 0, (
        "AllToAll data check failed (nonzero exit) at {} MiB/peer, dtype={}".format(
            per_peer_mib, dtype
        )
    )


def test_AllToAllGinSdma2GiBTotalHangGuard(request):
    rc, _ = _run_a2a(request, 2 * GiB, "int32")
    assert rc == 0, "AllToAll 2 GiB-total data check failed (nonzero exit)"
