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
# GinReduceScatterKernel). The kernel takes the SDMA-scatter tier when the CTA
# pool is below kReduceScatterSdmaCtaCeil and scratch fits; otherwise it uses
# the LSA read-reduce. These focus on:
#   * bit-exact datacheck across the size-adaptive CTA ladder (32 vs 48 CTAs),
#   * the low-CTA SDMA pin (NCCL_GIN_ANVIL_RS_CTAS=4),
#   * 4-wide / 2-wide / odd-peer LSA fold arms at NP=4 and NP=7,
#   * multiple reduction ops and integer/floating dtypes, and
#   * a large (2 GiB total at NP=8) completion guard so a hang fails fast.
#
# -b/-e are the TOTAL send-buffer bytes. ReduceScatterGetCollByteCount divides
# that count by nranks to get the per-rank output slice, so a 256 MiB/rank
# case must pass -b = 256 MiB * NP (2 GiB at NP=8). The helpers below do that
# conversion; callers pass the per-rank slice.
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
#   RCCL_TESTS_RS_GIN_TYPE       NCCL_GIN_TYPE (default: 7 = NCCL_GIN_TYPE_ANVIL_SDMA)
#   RCCL_TESTS_RS_TIMEOUT_S      per-run hang timeout seconds (default: 900)
#   RCCL_TESTS_RS_CONN_RETRIES   connectivity-gate re-launches (default: 5)

import os
import shlex

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

path = os.path.dirname(os.path.abspath(__file__))

_rs_enabled = os.environ.get("RCCL_TESTS_GIN_SDMA_RS", "") not in (
    "",
    "0",
    "false",
    "False",
)

RS_NP = env_int("RCCL_TESTS_RS_NP", 0) or (detect_ngpus() if _rs_enabled else 0)
RS_LAUNCHER = os.environ.get("RCCL_TESTS_MPI_LAUNCHER", "mpirun")
RS_CTAS = os.environ.get("RCCL_TESTS_RS_CTAS", "8")
RS_GIN_TYPE = os.environ.get("RCCL_TESTS_RS_GIN_TYPE", "7")
# Manual / SUT defaults. GIN CI (run-gin-ci.sh) overrides both so
# HW_CASES * retries * TIMEOUT_S stays under GIN_PYTEST_RS_TIMEOUT; otherwise
# GNU timeout kills pytest and leaves the mpirun session (start_new_session)
# orphaned because killpg only runs in TimeoutExpired handler.
RS_TIMEOUT_S = env_int("RCCL_TESTS_RS_TIMEOUT_S", 900)
RS_CONN_RETRIES = env_int("RCCL_TESTS_RS_CONN_RETRIES", 5)
RS_MPI_OPTS = shlex.split(os.environ.get("RCCL_TESTS_MPI_OPTS", ""))
RS_XENV = shlex.split(os.environ.get("RCCL_TESTS_RS_XENV", ""))
RS_EXE = os.environ.get(
    "RCCL_TESTS_RS_EXE", os.path.join(path, "..", "build", "reduce_scatter_perf")
)

_rs_skip = pytest.mark.skipif(
    not _rs_enabled,
    reason="GIN-SDMA ReduceScatter tests are opt-in; set RCCL_TESTS_GIN_SDMA_RS=1 on "
    "a GIN-capable (e.g. 8x MI355X) node to enable.",
)

# Param lists live here so RS_HW_LAUNCH_COUNT stays the single source of truth
# for GIN CI's GIN_PYTEST_RS_HW_CASES (run-gin-ci.sh reads the constant).
_CTA_LADDER = [
    (1, "sum"),
    (4, "sum"),
    (4, "avg"),
    (4, "prod"),
    (8, "min"),
    (8, "max"),
]
_CTA_DTYPES = ["int32", "float"]
_NARROW_OPS = ["sum", "avg"]
_NARROW_DTYPES = ["bfloat16", "fp8_e4m3", "fp8_e5m2"]
_LOW_CTA_DTYPES = ["int32", "float"]
_LOW_CTA_SIZES_MIB = [1, 8]
_PEER_LOOP_NPS = [4, 7]
_HANG_GUARD_DTYPES = ["int32", "int64", "half"]

# Literal so run-gin-ci.sh can read it without importing this module (detect_ngpus
# / relative imports). The offline test below locks it to the param lists.
RS_HW_LAUNCH_COUNT = 27  # 12 CTA + 6 narrow + 4 low-CTA + 2 peer-loop + 3 hang


def _launch_rs_gin_sdma(request, per_rank_bytes, dtype, op, *, np=None, xenv=None):
    """Launch reduce_scatter_perf -D 3 at a fixed per-rank output-slice size.

    -b is the total send buffer, not the slice, so multiply by np.
    """
    ranks = RS_NP if np is None else np
    total_bytes = int(per_rank_bytes) * ranks
    size = str(total_bytes)
    extra_xenv = list(RS_XENV if xenv is None else xenv)
    gin_env = gin_env_xflags(
        [
            "NCCL_GIN_ENABLE=1",
            "NCCL_GIN_TYPE={}".format(RS_GIN_TYPE),
            "NCCL_CUMEM_ENABLE=1",
            "RCCL_ENABLE_INTRANET=1",
            "NCCL_DMABUF_ENABLE=1",
            "HSA_NO_SCRATCH_RECLAIM=1",
        ]
        + extra_xenv
    )
    args = (
        mpi_launch_prefix(request, RS_LAUNCHER, ranks, RS_MPI_OPTS)
        + gin_env
        + gin_perf_argv(
            RS_EXE, size, dtype, RS_CTAS, extra=["-A", "1", "-o", op]
        )
    )
    cmd = " ".join(shlex.quote(a) for a in args)
    hang_msg = gin_hang_msg(
        "ReduceScatter",
        RS_TIMEOUT_S,
        size,
        dtype,
        "{} B/rank op={}".format(per_rank_bytes, op),
    )
    return launch_mpi_shell(cmd, RS_TIMEOUT_S, hang_msg)


def _run_rs_gin_sdma(request, per_rank_bytes, dtype, op, *, np=None, xenv=None):
    ranks = RS_NP if np is None else np
    if ranks < 2:
        pytest.skip("need >= 2 ranks/GPUs for GIN-SDMA ReduceScatter")
    return run_with_conn_gate_retry(
        lambda: _launch_rs_gin_sdma(
            request, per_rank_bytes, dtype, op, np=ranks, xenv=xenv
        ),
        RS_CONN_RETRIES,
    )


# CTA ladder probes (int32, 8 ranks -> total = per_rank * 8):
#   1 MiB/rank ->  8 MiB total -> 48 CTAs (mid-band lo)
#   4 MiB/rank -> 32 MiB total -> 48 CTAs (mid-band)
#   8 MiB/rank -> 64 MiB total -> 32 CTAs (large)
@_rs_skip
@pytest.mark.parametrize("per_rank_mib,op", _CTA_LADDER)
@pytest.mark.parametrize("dtype", _CTA_DTYPES)
def test_ReduceScatterGinSdmaCtaLadder(request, per_rank_mib, op, dtype):
    rc, _ = _run_rs_gin_sdma(request, per_rank_mib * MiB, dtype, op)
    assert rc == 0, "ReduceScatter datacheck failed at {} MiB/rank op={} dtype={}".format(
        per_rank_mib, op, dtype
    )


# The narrowing types. gin_sdma_reduce.h carries a separate overload per op for
# each of these because the pairwise fold has to narrow back to T on every step
# rather than accumulate in float, so their result depends on the fold order the
# -D 3 kernel uses and is not covered by the int32/float ladder above. 4 MiB/rank
# is the mid-band 48-CTA probe, the same size the ladder runs sum and avg at.
# prod is excluded for fp8 in SPECIALIZE_REDUCE_KERNEL, and min/max do not narrow,
# so sum and avg are the ops that exercise the overloads.
@_rs_skip
@pytest.mark.parametrize("op", _NARROW_OPS)
@pytest.mark.parametrize("dtype", _NARROW_DTYPES)
def test_ReduceScatterGinSdmaNarrowingDtypes(request, dtype, op):
    rc, _ = _run_rs_gin_sdma(request, 4 * MiB, dtype, op)
    assert rc == 0, "ReduceScatter datacheck failed at 4 MiB/rank op={} dtype={}".format(
        op, dtype
    )


@_rs_skip
@pytest.mark.parametrize("dtype", _LOW_CTA_DTYPES)
def test_ReduceScatterGinSdmaLowCtaSdma(request, dtype):
    # A pin below 16 CTAs selects SDMA-scatter + local reduce. Two launches per
    # dtype: 1 MiB/rank and 8 MiB/rank.
    xenv = RS_XENV + ["NCCL_GIN_ANVIL_RS_CTAS=4"]
    for per_rank_mib in _LOW_CTA_SIZES_MIB:
        rc, _ = _run_rs_gin_sdma(
            request, per_rank_mib * MiB, dtype, "sum", xenv=xenv
        )
        assert rc == 0, (
            "ReduceScatter low-CTA SDMA {} MiB/rank datacheck failed dtype={}".format(
                per_rank_mib, dtype
            )
        )


# LSA peer-fold arms in ginReduceScatterBody: NP=4 takes the nRanks<8 seed path
# plus the 2-wide and odd-peer loops; NP=7 takes one 4-wide batch then a 2-wide.
# The default NP=8 ladder only hits the 8-wide batch.
@_rs_skip
@pytest.mark.parametrize("np", _PEER_LOOP_NPS)
def test_ReduceScatterGinSdmaPeerLoopArms(request, np):
    if RS_NP < np:
        pytest.skip("peer-loop arm needs RS_NP >= {} (got {})".format(np, RS_NP))
    rc, _ = _run_rs_gin_sdma(request, 4 * MiB, "int32", "sum", np=np)
    assert rc == 0, "ReduceScatter peer-loop datacheck failed at NP={}".format(np)


@_rs_skip
@pytest.mark.parametrize("dtype", _HANG_GUARD_DTYPES)
def test_ReduceScatterGinSdma2GiBTotalHangGuard(request, dtype):
    # 256 MiB/rank * NP. At NP=8 that is the 2 GiB total the single-tier
    # argument is measured against, and it takes the 8-way grid-stride path.
    if RS_NP < 8:
        pytest.skip("2 GiB total / 8-way path needs RS_NP >= 8 (got {})".format(RS_NP))
    rc, _ = _run_rs_gin_sdma(request, 256 * MiB, dtype, "sum")
    assert rc == 0, "ReduceScatter 2 GiB-total datacheck failed (dtype={})".format(dtype)


def test_ReduceScatterGinSdmaHwLaunchCountMatchesParamLists():
    """Keep RS_HW_LAUNCH_COUNT honest for GIN CI's outer timeout divisor."""
    assert RS_HW_LAUNCH_COUNT == (
        len(_CTA_LADDER) * len(_CTA_DTYPES)
        + len(_NARROW_OPS) * len(_NARROW_DTYPES)
        + len(_LOW_CTA_DTYPES) * len(_LOW_CTA_SIZES_MIB)
        + len(_PEER_LOOP_NPS)
        + len(_HANG_GUARD_DTYPES)
    )
    assert RS_HW_LAUNCH_COUNT == 27
