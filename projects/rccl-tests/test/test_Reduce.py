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
import subprocess
import itertools
import math

import pytest

from .gin_sdma_harness import (
    MiB,
    _NUM,
    _WRONG,
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
# -D 3 folds the ranks in ascending order into the root's recvbuff through LSA
# pointers; the perf binary's data check (-c 1)
# exits non-zero on any wrong element. The partial-tile size leaves each rank's slice
# a few packs past a whole number of warp tiles, so the tail path runs too, and
# roots cover both ends of the rank range so the root-directed store is
# exercised off rank 0.
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


GIN_KV = ["NCCL_GIN_ENABLE=1", "NCCL_GIN_TYPE=7"]


def _run_reduce_gin_sdma(request, total_bytes, dtype, op, root, ctas=REDUCE_CTAS, xenv=(),
                         np=None, gin_kv=GIN_KV, extra=()):
    """Launch reduce_perf -D 3 at one size on np ranks."""
    np = np or REDUCE_NP
    if np < 2:
        pytest.skip("need >= 2 ranks/GPUs for GIN-SDMA Reduce")
    if np > REDUCE_NP:
        pytest.skip("needs {} GPUs, have {}".format(np, REDUCE_NP))
    size = str(int(total_bytes))
    kv = list(gin_kv) + REDUCE_XENV + list(xenv)
    args = (
        mpi_launch_prefix(request, REDUCE_LAUNCHER, np, REDUCE_MPI_OPTS)
        + gin_env_xflags(kv)
        + gin_perf_argv(REDUCE_EXE, size, dtype, ctas,
                        extra=["-o", op, "-r", str(root)])
        + list(extra)
    )
    cmd = " ".join(shlex.quote(a) for a in args)
    hang_msg = gin_hang_msg(
        "Reduce", REDUCE_TIMEOUT_S, size, dtype, "{} MiB total".format(total_bytes // MiB))
    rc, out = run_with_conn_gate_retry(
        lambda: launch_mpi_shell(cmd, REDUCE_TIMEOUT_S, hang_msg),
        REDUCE_CONN_RETRIES,
    )
    # reduce_perf runs float instead of a type it was built without.
    if "invalid type {}".format(dtype) in (out or ""):
        pytest.skip("reduce_perf was built without {}".format(dtype))
    return rc, out


# size count type redop root, then out-of-place (time algbw busbw #wrong) and,
# unless -O 1, in-place. Not end-anchored, like gin_sdma_harness.BCAST_ROW_RE.
REDUCE_ROW_RE = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+\S+\s+\S+\s+(-?\d+)"
    r"\s+{n}\s+{n}\s+{n}\s+({w})"
    r"(?:\s+{n}\s+{n}\s+{n}\s+({w}))?".format(n=_NUM, w=_WRONG)
)
REDUCE_DEVTIME_RE = re.compile(r"^#\[reduce-devtime\] size\s+(\d+) B\s+ctas\s+(\d+)", re.M)


def _aligned_bytes(total_bytes, nparts):
    """Bytes reduce_perf runs: whole 16-byte packs in each of nparts parts."""
    return total_bytes // (nparts * 16) * nparts * 16


def _assert_reduce_ok(rc, out, ctx, total_bytes, nparts):
    """The run exited 0, measured exactly the aligned size, and checked every
    element of every row clean."""
    tail = (out or "")[-2000:]
    assert rc == 0, "Reduce {} failed (exit {}). Output tail:\n{}".format(ctx, rc, tail)
    rows = [m for m in (REDUCE_ROW_RE.match(l) for l in (out or "").splitlines()) if m]
    assert rows, "Reduce {} produced no result rows. Output tail:\n{}".format(ctx, tail)
    want = _aligned_bytes(total_bytes, nparts)
    assert want > 0, "Reduce {}: {} bytes round down to nothing".format(ctx, total_bytes)
    sizes = sorted({int(m.group(1)) for m in rows})
    assert sizes == [want], "Reduce {} measured sizes {}, expected [{}]".format(ctx, sizes, want)
    for m in rows:
        wrong = [m.group(4), m.group(5)]
        assert all(w not in (None, "N/A") and float(w) == 0.0 for w in wrong), (
            "Reduce {} row has #wrong {} (root {}); every element must be checked "
            "and correct.".format(ctx, wrong, m.group(3)))


@_reduce_gin_skip
@pytest.mark.parametrize("size", ["1MiB", "64MiB", "partial-tile"])
@pytest.mark.parametrize("dtype, op", [
    ("int32", "sum"),
    ("int32", "min"),
    ("int64", "avg"),
    ("float", "sum"),
    ("half", "max"),
    ("bfloat16", "avg"),
    ("double", "prod"),
    ("fp8_e4m3", "sum"),
    ("fp8_e5m2", "max"),
])
@pytest.mark.parametrize("root_pos", ["first", "last"])
def test_ReduceGinSdma(request, size, dtype, op, root_pos):
    root = 0 if root_pos == "first" else REDUCE_NP - 1
    # 5 packs of 16 B past 1 MiB per rank.
    total_bytes = {"1MiB": MiB, "64MiB": 64 * MiB, "partial-tile": REDUCE_NP * (MiB + 5 * 16)}[size]
    rc, out = _run_reduce_gin_sdma(request, total_bytes, dtype, op, root)
    _assert_reduce_ok(rc, out, "-D 3 at {} bytes, dtype={}, op={}, root={}".format(
        total_bytes, dtype, op, root), total_bytes, REDUCE_NP)


# -D 3 runs on LSA alone, so it needs no GIN.
@_reduce_gin_skip
def test_ReduceGinSdmaNoGin(request):
    total_bytes = 4 * MiB
    rc, out = _run_reduce_gin_sdma(request, total_bytes, "float", "sum", REDUCE_NP - 1,
                                   gin_kv=["NCCL_GIN_ENABLE=0"])
    _assert_reduce_ok(rc, out, "-D 3 without GIN at {} bytes".format(total_bytes),
                      total_bytes, REDUCE_NP)


# In-kernel timing for the -D 3 kernel. -B 1 adds a #[reduce-devtime] line for
# the timed kernel. -H 1 only checks the timed kernel's output with -B 2, which
# reports that kernel's latency as the metric.
@_reduce_gin_skip
@pytest.mark.parametrize("mode", ["augment", "check"])
def test_ReduceGinSdmaDevtime(request, mode):
    total_bytes = 4 * MiB
    timing = ["-B", "1"] if mode == "augment" else ["-B", "2", "-H", "1"]
    rc, out = _run_reduce_gin_sdma(request, total_bytes, "float", "sum", 0,
                                   extra=timing + ["-L", "5", "-P", "2"])
    _assert_reduce_ok(rc, out, "-D 3 {} at {} bytes".format(" ".join(timing), total_bytes),
                      total_bytes, REDUCE_NP)
    tail = out[-2000:]
    if mode == "check":
        assert "no in-kernel device-time measurement" not in out, (
            "the timed kernel did not run, so -H 1 checked nothing:\n{}".format(tail))
        assert "ERROR: --devtime_check:" not in out, "timed-kernel data check failed:\n{}".format(tail)
        return
    want = _aligned_bytes(total_bytes, REDUCE_NP)
    lines = REDUCE_DEVTIME_RE.findall(out or "")
    assert lines, "no #[reduce-devtime] line. Output tail:\n{}".format(tail)
    assert all(int(s) == want and int(c) == int(REDUCE_CTAS) for s, c in lines), (
        "devtime lines {} expected size {} and {} CTAs".format(lines, want, REDUCE_CTAS))


# Requests the device modes cannot serve fail on every rank instead of hanging
# or exiting 0 without results.
@_reduce_gin_skip
@pytest.mark.gin_ci
@pytest.mark.parametrize("case", ["mulsum", "root"])
def test_ReduceGinSdmaRejects(request, case):
    op, root, message = {
        "mulsum": ("mulsum", 0, "no mulsum"),
        "root": ("sum", REDUCE_NP, "is outside"),
    }[case]
    rc, out = _run_reduce_gin_sdma(request, MiB, "float", op, root)
    assert rc != 0, "Reduce {} should fail but exited 0".format(case)
    assert message in (out or ""), "Reduce {} failed without the expected message '{}':\n{}".format(
        case, message, (out or "")[-2000:])

