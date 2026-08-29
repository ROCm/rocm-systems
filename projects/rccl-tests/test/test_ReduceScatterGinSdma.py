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

# Opt-in GIN-SDMA ReduceScatter regression tests (reduce_scatter_perf -D 3,
# GinReduceScatterKernel). Unlike AllGather, the shipped RS kernel is a single-
# tier LSA read-reduce (no gin.put / SDMA put tier), so these focus on:
#   * bit-exact datacheck across the size-adaptive CTA ladder (32 vs 48 CTAs),
#   * multiple reduction ops and integer/floating dtypes, and
#   * a large (2 GiB total) completion guard so a hang fails fast.
#
# -b/-e are the per-rank output-slice byte sizes (recvbuff chunk); total moved
# bytes per iteration is slice * NP * sizeof(dtype) for the algorithm bandwidth.
#
# Requires an 8x MI355X (or similar) node, an MPI launcher, and a GIN-capable
# RCCL build with rocSHMEM linked into reduce_scatter_perf. Skipped unless
# RCCL_TESTS_GIN_SDMA_RS=1.
#
# Environment (mirrors test_AllGather.py / test_AllToAll.py):
#   RCCL_TESTS_GIN_SDMA_RS       enable this module (default: off)
#   RCCL_TESTS_RS_NP             MPI ranks (default: detected GPU count)
#   RCCL_TESTS_MPI_LAUNCHER      launcher binary (default: mpirun)
#   RCCL_TESTS_MPI_OPTS          extra launcher opts
#   RCCL_TESTS_RS_XENV           extra "-x K=V" env the backend needs
#   RCCL_TESTS_RS_EXE            path to reduce_scatter_perf
#   RCCL_TESTS_RS_CTAS           device CTA pool (-V) (default: 8)
#   RCCL_TESTS_RS_GIN_TYPE       NCCL_GIN_TYPE (default: 6, matches gin-tests.json)
#   RCCL_TESTS_RS_TIMEOUT_S      per-run hang timeout seconds (default: 900)
#   RCCL_TESTS_RS_CONN_RETRIES   connectivity-gate re-launches (default: 5)

import os
import re
import shlex
import signal
import subprocess
import time

import pytest

MiB = 1024 * 1024
GiB = 1024 * MiB

path = os.path.dirname(os.path.abspath(__file__))

_rs_enabled = os.environ.get("RCCL_TESTS_GIN_SDMA_RS", "") not in ("", "0", "false", "False")

RS_NP = int(os.environ.get("RCCL_TESTS_RS_NP", "0"))
if RS_NP <= 0:
    if os.environ.get("ROCR_VISIBLE_DEVICES") is not None:
        RS_NP = len(os.environ["ROCR_VISIBLE_DEVICES"].split(","))
    elif os.environ.get("HIP_VISIBLE_DEVICES") is not None:
        RS_NP = len(os.environ["HIP_VISIBLE_DEVICES"].split(","))
    else:
        try:
            RS_NP = int(subprocess.check_output(
                'rocminfo | grep "Device Type:.\\s*.GPU" | wc -l', shell=True))
        except Exception:
            RS_NP = 0

RS_LAUNCHER = os.environ.get("RCCL_TESTS_MPI_LAUNCHER", "mpirun")
RS_CTAS = os.environ.get("RCCL_TESTS_RS_CTAS", "8")
RS_GIN_TYPE = os.environ.get("RCCL_TESTS_RS_GIN_TYPE", "6")
RS_TIMEOUT_S = int(os.environ.get("RCCL_TESTS_RS_TIMEOUT_S", "900"))
RS_CONN_RETRIES = int(os.environ.get("RCCL_TESTS_RS_CONN_RETRIES", "5"))
RS_MPI_OPTS = shlex.split(os.environ.get("RCCL_TESTS_MPI_OPTS", ""))
RS_XENV = shlex.split(os.environ.get("RCCL_TESTS_RS_XENV", ""))
RS_EXE = os.environ.get(
    "RCCL_TESTS_RS_EXE", os.path.join(path, "..", "build", "reduce_scatter_perf"))

_rs_skip = pytest.mark.skipif(
    not _rs_enabled,
    reason="GIN-SDMA ReduceScatter tests are opt-in; set RCCL_TESTS_GIN_SDMA_RS=1 on "
           "a GIN-capable (e.g. 8x MI355X) node to enable.")

_CONN_GATE_RE = re.compile(r"LSA signal connectivity gate failed|unhandled system error", re.I)
_DATA_FAIL_RE = re.compile(r"Wrong|mismatch|check.*fail|Out of bounds values\s*:\s*[1-9]", re.I)


def _launch_rs_gin_sdma(request, per_rank_bytes, dtype, op):
    """Launch reduce_scatter_perf -D 3 at a fixed per-rank output-slice size."""
    size = str(int(per_rank_bytes))
    gin_env = []
    for kv in ["NCCL_GIN_ENABLE=1", "NCCL_GIN_TYPE={}".format(RS_GIN_TYPE),
               "NCCL_CUMEM_ENABLE=1", "RCCL_ENABLE_INTRANET=1",
               "NCCL_DMABUF_ENABLE=1", "HSA_NO_SCRATCH_RECLAIM=1"] + RS_XENV:
        gin_env += ["-x", kv]

    hostfile = request.config.getoption("--hostfile")
    launch = [RS_LAUNCHER, "-np", str(RS_NP)] + RS_MPI_OPTS
    if hostfile:
        launch += ["-host", hostfile]

    args = launch + gin_env + [
        RS_EXE,
        "-b", size, "-e", size,
        "-f", "2",
        "-g", "1",
        "-R", "2",
        "-D", "3",
        "-A", "1",
        "-V", RS_CTAS,
        "-d", dtype,
        "-o", op,
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
        out, _ = proc.communicate(timeout=RS_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        out, _ = proc.communicate()
        pytest.fail(
            "ReduceScatter GIN-SDMA HANG: no completion within {}s at {} B/rank, "
            "dtype={}, op={}. Output tail:\n{}".format(
                RS_TIMEOUT_S, size, dtype, op, (out or "")[-2000:]))
    print(out)
    return proc.returncode, out


def _run_rs_gin_sdma(request, per_rank_bytes, dtype, op):
    if RS_NP < 2:
        pytest.skip("need >= 2 ranks/GPUs for GIN-SDMA ReduceScatter")

    rc, out = 1, ""
    for attempt in range(1, max(1, RS_CONN_RETRIES) + 1):
        rc, out = _launch_rs_gin_sdma(request, per_rank_bytes, dtype, op)
        if rc == 0:
            return rc, out
        if _DATA_FAIL_RE.search(out or ""):
            return rc, out
        if _CONN_GATE_RE.search(out or "") and attempt < RS_CONN_RETRIES:
            print("=== connectivity-gate abort (attempt {}/{}); re-launching ===".format(
                attempt, RS_CONN_RETRIES))
            time.sleep(3)
            continue
        return rc, out
    return rc, out


# CTA ladder probes (int32, 8 ranks -> total = per_rank * 8):
#   256 KiB/rank ->  2 MiB total -> 32 CTAs (small)
#   1   MiB/rank ->  8 MiB total -> 48 CTAs (mid-band lo)
#   4   MiB/rank -> 32 MiB total -> 48 CTAs (mid-band)
#   8   MiB/rank -> 64 MiB total -> 32 CTAs (large)
@_rs_skip
@pytest.mark.parametrize("per_rank_mib,op", [
    (1, "sum"),
    (4, "sum"),
    (8, "min"),
    (8, "max"),
])
@pytest.mark.parametrize("dtype", ["int32", "float"])
def test_ReduceScatterGinSdmaCtaLadder(request, per_rank_mib, op, dtype):
    rc, _ = _run_rs_gin_sdma(request, per_rank_mib * MiB, dtype, op)
    assert rc == 0, "ReduceScatter datacheck failed at {} MiB/rank op={} dtype={}".format(
        per_rank_mib, op, dtype)


@_rs_skip
@pytest.mark.parametrize("dtype", ["int32", "int64", "half"])
def test_ReduceScatterGinSdma2GiBTotalHangGuard(request, dtype):
    # 256 MiB/rank * 8 ranks = 2 GiB total; exercises the grid-stride 8-way path.
    rc, _ = _run_rs_gin_sdma(request, 256 * MiB, dtype, "sum")
    assert rc == 0, "ReduceScatter 2 GiB-total datacheck failed (dtype={})".format(dtype)
