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
#                            (e.g. HSA_FORCE_FINE_GRAIN_PCIE=1 NCCL_CUMEM_ENABLE=1)
#   RCCL_TESTS_A2A_EXE       path to alltoall_perf (default: ../build/alltoall_perf)
#   RCCL_TESTS_A2A_TIMEOUT_S per-run hang timeout in seconds (default: 900)
#
# Verified on 8x MI355X (ROCm 7.13, NCCL_GIN_TYPE=6, force1ch): 256 MiB/peer and
# 2 GiB/peer int32 and the 2 GiB-total guard all pass with #wrong=0, no hang
# (busbw ~424-428 GB/s).

import os
import shlex
import signal
import subprocess

import pytest

MiB = 1024 * 1024
GiB = 1024 * MiB

path = os.path.dirname(os.path.abspath(__file__))
# Path to alltoall_perf. Override with RCCL_TESTS_A2A_EXE when the binary is not
# at the default build/ location (e.g. an installed/container path).
executable = os.environ.get(
    "RCCL_TESTS_A2A_EXE", os.path.join(path, "..", "build", "alltoall_perf"))

# Opt-in: only run where the GIN-SDMA backend + hardware are available.
_enabled = os.environ.get("RCCL_TESTS_GIN_SDMA_A2A", "") not in ("", "0", "false", "False")


def _detect_ngpus():
    if os.environ.get("ROCR_VISIBLE_DEVICES") is not None:
        return len(os.environ["ROCR_VISIBLE_DEVICES"].split(","))
    if os.environ.get("HIP_VISIBLE_DEVICES") is not None:
        return len(os.environ["HIP_VISIBLE_DEVICES"].split(","))
    try:
        out = subprocess.check_output(
            'rocminfo | grep "Device Type:.\\s*.GPU" | wc -l', shell=True)
        return int(out)
    except Exception:
        return 0


NP = int(os.environ.get("RCCL_TESTS_A2A_NP", "0")) or _detect_ngpus()
LAUNCHER = os.environ.get("RCCL_TESTS_MPI_LAUNCHER", "mpirun")
CTAS = os.environ.get("RCCL_TESTS_A2A_CTAS", "16")
# Generous per-call timeout; a real hang spins forever, so a finite cap is what
# converts item (3) into a detectable failure. Scaled up for the multi-GiB case.
TIMEOUT_S = int(os.environ.get("RCCL_TESTS_A2A_TIMEOUT_S", "900"))

# Extra launcher options (e.g. "--allow-run-as-root -mca pml ob1 ...") and any
# deployment-specific env the GIN-SDMA backend needs beyond the essentials below
# (e.g. HSA_FORCE_FINE_GRAIN_PCIE=1 NCCL_CUMEM_ENABLE=1 RCCL_ENABLE_INTRANET=1).
# GIN enablement itself is site-specific, so it is injected here rather than
# hard-coded, keeping the test portable across clusters.
MPI_OPTS = shlex.split(os.environ.get("RCCL_TESTS_MPI_OPTS", ""))
XENV = shlex.split(os.environ.get("RCCL_TESTS_A2A_XENV", ""))

pytestmark = pytest.mark.skipif(
    not _enabled,
    reason="GIN-SDMA AllToAll tests are opt-in; set RCCL_TESTS_GIN_SDMA_A2A=1 on "
           "a GIN-SDMA-capable (e.g. 8x MI355X) node to enable.")


def _run_a2a(request, total_bytes, dtype):
    """Run one all-to-all at a fixed total (per-rank) size, forcing the GIN-SDMA
    tier. Returns (returncode, stdout). A timeout (hang) fails the test."""
    if NP < 2:
        pytest.skip("need >= 2 ranks/GPUs for AllToAll")

    size = str(int(total_bytes))
    # Essentials to exercise the GIN-SDMA put path; force the SDMA (large) tier
    # for every size. Deployment-specific env comes from RCCL_TESTS_A2A_XENV.
    gin_env = []
    for kv in ["NCCL_GIN_ENABLE=1", "NCCL_GIN_TYPE=6",
               "NCCL_GIN_ANVIL_SDMA_THRESHOLD=0",
               "NCCL_GIN_ANVIL_SDMA_THRESHOLD_ALLTOALL=0"] + XENV:
        gin_env += ["-x", kv]

    hostfile = request.config.getoption("--hostfile")
    launch = [LAUNCHER, "-np", str(NP)] + MPI_OPTS
    if hostfile:
        launch += ["-host", hostfile]

    args = launch + gin_env + [
        executable,
        "-b", size, "-e", size,
        "-f", "2",
        "-g", "1",
        "-R", "2",
        "-D", "3",
        "-V", CTAS,
        "-d", dtype,
        "-c", "1",       # data check: nonzero exit on any wrong element
        "-w", "1",
        "-n", "3",
    ]
    cmd = " ".join(shlex.quote(a) for a in args)
    print(cmd)
    # Run in a new session so a hang can be killed as a whole process group --
    # otherwise a wedged launcher leaves orphaned ranks pinning the GPUs (and
    # exhausting SDMA queues for later tests).
    proc = subprocess.Popen(cmd, shell=True, universal_newlines=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            start_new_session=True)
    try:
        out, _ = proc.communicate(timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        out, _ = proc.communicate()
        pytest.fail(
            "AllToAll GIN-SDMA HANG: no completion within {}s at total={} bytes "
            "({} MiB/peer), dtype={}. Output tail:\n{}".format(
                TIMEOUT_S, size, total_bytes // NP // MiB, dtype, (out or "")[-2000:]))
    print(out)
    return proc.returncode, out


# (item 1 + 2) Multi-segment loop and the 1 GiB truncation boundary. per_peer is
# the transfer to each peer; total per-rank bytes = per_peer * NP.
@pytest.mark.parametrize("per_peer_mib", [256, 2048])  # 256 MiB (2 seg), 2 GiB (16 seg)
@pytest.mark.parametrize("dtype", ["int32", "int64", "uint8"])
def test_AllToAllGinSdmaLargeSegmented(request, per_peer_mib, dtype):
    total = per_peer_mib * MiB * NP
    rc, _ = _run_a2a(request, total, dtype)
    assert rc == 0, "AllToAll data check failed (nonzero exit) at {} MiB/peer, dtype={}".format(
        per_peer_mib, dtype)


# (item 3) The original 2 GiB-total hang repro, as a completion guard. At 8 ranks
# this is 256 MiB/peer -- the size that hung as a single unclamped put.
def test_AllToAllGinSdma2GiBTotalHangGuard(request):
    rc, _ = _run_a2a(request, 2 * GiB, "int32")
    assert rc == 0, "AllToAll 2 GiB-total data check failed (nonzero exit)"
