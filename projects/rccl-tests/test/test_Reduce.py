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
# -D 3 picks a tier by size: LL up to NCCL_GIN_ANVIL_REDUCE_LL_MAX_BYTES, the
# tiled LSA kernel above it, and the SDMA tier for out-of-place messages from
# NCCL_GIN_ANVIL_SDMA_THRESHOLD_REDUCE up. Every tier folds the ranks in
# ascending order into the root's recvbuff; the perf binary's data check (-c 1)
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
# An SDMA threshold above every size these tests run.
SDMA_OFF = 1 << 40


def _run_reduce_gin_sdma(request, total_bytes, dtype, op, root, ctas=REDUCE_CTAS, xenv=(),
                         np=None, threads=1, gin_kv=GIN_KV, oop_only=False, sdma_threshold=None,
                         extra=()):
    """Launch reduce_perf -D 3 at one size. np is the rank count; with threads > 1
    each process runs that many host threads (-t), so np // threads processes.
    sdma_threshold sets NCCL_GIN_ANVIL_SDMA_THRESHOLD_REDUCE; None keeps the default."""
    np = np or REDUCE_NP
    if np < 2:
        pytest.skip("need >= 2 ranks/GPUs for GIN-SDMA Reduce")
    if np > REDUCE_NP:
        pytest.skip("needs {} GPUs, have {}".format(np, REDUCE_NP))
    if np % threads:
        pytest.skip("{} ranks do not split into {} threads per process".format(np, threads))
    size = str(int(total_bytes))
    kv = list(gin_kv) + ["RCCL_TESTS_REDUCE_TIER_LOG=1"] + REDUCE_XENV + list(xenv)
    if sdma_threshold is not None:
        kv.append("NCCL_GIN_ANVIL_SDMA_THRESHOLD_REDUCE={}".format(int(sdma_threshold)))
    impl_args = ["-O", "1"] if oop_only else []
    if threads > 1:
        impl_args += ["-t", str(threads)]
    args = (
        mpi_launch_prefix(request, REDUCE_LAUNCHER, np // threads, REDUCE_MPI_OPTS)
        + gin_env_xflags(kv)
        + gin_perf_argv(REDUCE_EXE, size, dtype, ctas,
                        extra=impl_args + ["-o", op, "-r", str(root)])
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
REDUCE_TIER_RE = re.compile(r"^#\[reduce-tier\]\s+(\S+)", re.M)
REDUCE_LL_CAP_RE = re.compile(r"^# Reduce -D 3: LL tier for messages up to (\d+) bytes", re.M)
REDUCE_SDMA_THRESHOLD_RE = re.compile(r"^# Reduce -D 3: SDMA tier for out-of-place messages from (\d+) bytes", re.M)
REDUCE_DEVTIME_RE = re.compile(r"^#\[reduce-devtime\] size\s+(\d+) B\s+ctas\s+(\d+)", re.M)


def _aligned_bytes(total_bytes, nparts):
    """Bytes reduce_perf runs: whole 16-byte packs in each of nparts parts."""
    return total_bytes // (nparts * 16) * nparts * 16


def _default_sdma_threshold(ctas, nranks):
    """The SDMA threshold reduce_perf uses with no threshold set (None = off)."""
    if ctas < 2 * nranks:
        return None
    for limit, threshold in ((16, 24 * MiB), (32, 48 * MiB), (64, 768 * MiB)):
        if ctas <= limit:
            return threshold
    return 1536 * MiB


def _assert_reduce_ok(rc, out, ctx, total_bytes, nparts, expect_tier=None, in_place=True):
    """The run exited 0, measured exactly the aligned size, checked every
    element of every row clean, and (if asked) ran exactly the expected tier."""
    tail = (out or "")[-2000:]
    assert rc == 0, "Reduce {} failed (exit {}). Output tail:\n{}".format(ctx, rc, tail)
    rows = [m for m in (REDUCE_ROW_RE.match(l) for l in (out or "").splitlines()) if m]
    assert rows, "Reduce {} produced no result rows. Output tail:\n{}".format(ctx, tail)
    want = _aligned_bytes(total_bytes, nparts)
    assert want > 0, "Reduce {}: {} bytes round down to nothing".format(ctx, total_bytes)
    sizes = sorted({int(m.group(1)) for m in rows})
    assert sizes == [want], "Reduce {} measured sizes {}, expected [{}]".format(ctx, sizes, want)
    for m in rows:
        wrong = [m.group(4)] + ([m.group(5)] if in_place else [])
        assert all(w not in (None, "N/A") and float(w) == 0.0 for w in wrong), (
            "Reduce {} row has #wrong {} (root {}); every element must be checked "
            "and correct.".format(ctx, wrong, m.group(3)))
    if expect_tier is not None:
        want_tiers = {expect_tier} if isinstance(expect_tier, str) else set(expect_tier)
        tiers = set(REDUCE_TIER_RE.findall(out or ""))
        assert tiers == want_tiers, (
            "Reduce {} ran tier(s) {}, expected {}. Output tail:\n{}".format(
                ctx, sorted(tiers), sorted(want_tiers), tail))


def _default_ll_cap(nranks):
    """The -D 3 LL cap reduce_perf uses with NCCL_GIN_ANVIL_REDUCE_LL_MAX_BYTES unset."""
    return min(16 * MiB // (4 * nranks), MiB) // 8 * 8


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
    # LL and SDMA off, so every size runs the tiled kernel.
    rc, out = _run_reduce_gin_sdma(request, total_bytes, dtype, op, root, sdma_threshold=SDMA_OFF,
                                   xenv=["NCCL_GIN_ANVIL_REDUCE_LL_MAX_BYTES=0"])
    _assert_reduce_ok(rc, out, "-D 3 at {} bytes, dtype={}, op={}, root={}".format(
        total_bytes, dtype, op, root), total_bytes, REDUCE_NP, expect_tier="tiled")


# LL tier of deviceImpl 3 (GinReduceLLKernel): messages up to
# NCCL_GIN_ANVIL_REDUCE_LL_MAX_BYTES go to the root as LL packets with no
# barrier. "tiny" is the smallest aligned message (one CTA, a few units),
# "uneven" splits unevenly across several CTAs, and "max" forces the 1 MiB
# ceiling (256 CTAs). The perf binary runs in-place and out-of-place, and its
# warmup and timed iterations run back to back, which exercises the root's ack.
@_reduce_gin_skip
@pytest.mark.parametrize("size", ["tiny", "uneven", "max"])
@pytest.mark.parametrize("dtype, op", [
    ("float", "sum"),
    ("bfloat16", "avg"),
    ("int8", "max"),
    ("double", "prod"),
])
@pytest.mark.parametrize("root_pos", ["first", "last"])
def test_ReduceGinSdmaLL(request, size, dtype, op, root_pos):
    root = 0 if root_pos == "first" else REDUCE_NP - 1
    total_bytes = {"tiny": REDUCE_NP * 16, "uneven": REDUCE_NP * 16 * 1001, "max": MiB}[size]
    rc, out = _run_reduce_gin_sdma(request, total_bytes, dtype, op, root,
                                   xenv=["NCCL_GIN_ANVIL_REDUCE_LL_MAX_BYTES=1048576"])
    _assert_reduce_ok(rc, out, "-D 3 LL at {} bytes, dtype={}, op={}, root={}".format(
        total_bytes, dtype, op, root), total_bytes, REDUCE_NP, expect_tier="ll")


# 2000 back-to-back LL calls with no warmup gap. Without the root's ack, a
# sender gets two epochs ahead and overwrites slots the root has not read, which
# shows up as #wrong or a hang.
@_reduce_gin_skip
def test_ReduceGinSdmaLLBackToBack(request):
    total_bytes = REDUCE_NP * 16
    rc, out = _run_reduce_gin_sdma(request, total_bytes, "float", "sum", REDUCE_NP - 1,
                                   xenv=["NCCL_GIN_ANVIL_REDUCE_LL_MAX_BYTES=1048576"],
                                   extra=["-w", "0", "-n", "2000"])
    _assert_reduce_ok(rc, out, "-D 3 LL, 2000 back-to-back calls", total_bytes, REDUCE_NP,
                      expect_tier="ll")


# With NCCL_GIN_ANVIL_REDUCE_LL_MAX_BYTES unset: the printed cap is the
# documented default, a message at the cap takes the LL tier, and the next
# aligned size up takes the tiled kernel.
@_reduce_gin_skip
@pytest.mark.gin_ci
@pytest.mark.parametrize("where, tier", [("at-cap", "ll"), ("above-cap", "tiled")])
def test_ReduceGinSdmaLLDefaultCap(request, where, tier):
    cap = _default_ll_cap(REDUCE_NP)
    step = REDUCE_NP * 16
    total_bytes = cap if where == "at-cap" else _aligned_bytes(cap, REDUCE_NP) + step
    rc, out = _run_reduce_gin_sdma(request, total_bytes, "float", "sum", 0)
    _assert_reduce_ok(rc, out, "-D 3 default cap, {} ({} bytes)".format(where, total_bytes),
                      total_bytes, REDUCE_NP, expect_tier=tier)
    printed = REDUCE_LL_CAP_RE.search(out or "")
    assert printed and int(printed.group(1)) == cap, (
        "Reduce -D 3 printed LL cap {}, expected the default {}".format(
            printed.group(1) if printed else "nothing", cap))


# Without GIN the SDMA tier is off, so -D 3 runs on LSA alone even with the
# SDMA threshold at 0.
@_reduce_gin_skip
@pytest.mark.parametrize("tier, total_bytes", [("ll", 64 * 1024), ("tiled", 4 * MiB)])
def test_ReduceGinSdmaNoGin(request, tier, total_bytes):
    rc, out = _run_reduce_gin_sdma(request, total_bytes, "float", "sum", REDUCE_NP - 1,
                                   gin_kv=["NCCL_GIN_ENABLE=0"], sdma_threshold=0)
    _assert_reduce_ok(rc, out, "-D 3 without GIN at {} bytes".format(total_bytes),
                      total_bytes, REDUCE_NP, expect_tier=tier)


# Two host threads per process (-t 2) with the default -p 0: the main thread
# builds every devComm, so the worker thread must see the same LL cap and
# resource handles, or its ranks launch another kernel and the run hangs.
# The SDMA tier stays off even with the threshold at 0: the GIN Anvil SDMA
# backend takes a GPU memory fault when one process drives two GPUs, for
# broadcast's GIN tiers as well.
@_reduce_gin_skip
@pytest.mark.gin_ci
@pytest.mark.parametrize("tier, total_bytes", [("ll", 64 * 1024), ("tiled", 4 * MiB)])
def test_ReduceGinSdmaThreads(request, tier, total_bytes):
    rc, out = _run_reduce_gin_sdma(request, total_bytes, "float", "sum", REDUCE_NP - 1,
                                   threads=2, sdma_threshold=0)
    _assert_reduce_ok(rc, out, "-D 3 -t 2 at {} bytes".format(total_bytes),
                      total_bytes, REDUCE_NP, expect_tier=tier)


# In-kernel timing for the -D 3 kernels. -B 1 adds a #[reduce-devtime] line
# whose CTA count shows which kernel was timed: at 32 KiB the LL kernel runs 8
# CTAs, the tiled kernel -V. -H 1 only checks the timed kernel's output with
# -B 2, which reports that kernel's latency as the metric. Both time only the
# out-of-place pass, which the SDMA tier takes, and it has no timed kernel, so
# there nothing must be timed in its place.
@_reduce_gin_skip
@pytest.mark.parametrize("mode", ["augment", "check"])
@pytest.mark.parametrize("tier, total_bytes", [("ll", 32 * 1024), ("tiled", 4 * MiB), ("sdma", 4 * MiB)])
def test_ReduceGinSdmaDevtime(request, tier, total_bytes, mode):
    timing = ["-B", "1"] if mode == "augment" else ["-B", "2", "-H", "1"]
    threshold = 0 if tier == "sdma" else SDMA_OFF
    rc, out = _run_reduce_gin_sdma(request, total_bytes, "float", "sum", 0, sdma_threshold=threshold,
                                   extra=timing + ["-L", "5", "-P", "2"])
    sdma_tier = "sdma-staged" if int(REDUCE_CTAS) >= 2 * REDUCE_NP else "sdma-cu"
    # -B 2 leaves the out-of-place columns of a row it could not time blank, so
    # the SDMA row's only #wrong is the in-place one.
    _assert_reduce_ok(rc, out, "-D 3 {} at {} bytes".format(" ".join(timing), total_bytes),
                      total_bytes, REDUCE_NP,
                      expect_tier={sdma_tier, "tiled"} if tier == "sdma" else tier,
                      in_place=not (tier == "sdma" and mode == "check"))
    tail = out[-2000:]
    if tier == "sdma":
        if mode == "check":
            assert "no in-kernel device-time measurement" in out, (
                "-B 2 reported a device time for the SDMA tier:\n{}".format(tail))
        else:
            assert not REDUCE_DEVTIME_RE.search(out), (
                "-B 1 timed a kernel for the SDMA tier:\n{}".format(tail))
        return
    if mode == "check":
        assert "no in-kernel device-time measurement" not in out, (
            "the timed kernel did not run, so -H 1 checked nothing:\n{}".format(tail))
        assert "ERROR: --devtime_check:" not in out, "timed-kernel data check failed:\n{}".format(tail)
        return
    want = _aligned_bytes(total_bytes, REDUCE_NP)
    ll_ctas = max(1, min(-(-want // 8 // 512), 256))
    if ll_ctas == int(REDUCE_CTAS):
        pytest.skip("both -D 3 kernels launch {} CTAs at {} bytes".format(ll_ctas, want))
    lines = REDUCE_DEVTIME_RE.findall(out or "")
    assert lines, "no #[reduce-devtime] line. Output tail:\n{}".format(tail)
    expect_ctas = ll_ctas if tier == "ll" else int(REDUCE_CTAS)
    assert all(int(s) == want and int(c) == expect_ctas for s, c in lines), (
        "devtime lines {} expected size {} and {} CTAs".format(lines, want, expect_ctas))


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


# Tier choice around an SDMA threshold of 4 MiB. Below it, both passes run
# tiled; from it up, the out-of-place pass takes the SDMA tier and the in-place
# pass stays tiled, since the SDMA tier keeps its receive slots in recvbuff.
@_reduce_gin_skip
@pytest.mark.gin_ci
@pytest.mark.parametrize("case", ["below", "above"])
def test_ReduceGinSdmaHybrid(request, case):
    threshold = 4 * MiB
    total_bytes = 2 * MiB if case == "below" else 8 * MiB
    ctas = max(2 * REDUCE_NP, int(REDUCE_CTAS))
    if ctas > 128:
        pytest.skip("the staged SDMA path needs 2 * {} CTAs, over the 128-CTA limit".format(REDUCE_NP))
    rc, out = _run_reduce_gin_sdma(request, total_bytes, "float", "sum", REDUCE_NP - 1, ctas=ctas,
                                   sdma_threshold=threshold)
    tiers = {"sdma-staged", "tiled"} if case == "above" else "tiled"
    _assert_reduce_ok(rc, out, "-D 3 hybrid, {} ({} bytes)".format(case, total_bytes), total_bytes,
                      REDUCE_NP, expect_tier=tiers)
    printed = REDUCE_SDMA_THRESHOLD_RE.search(out or "")
    assert printed and int(printed.group(1)) == threshold, (
        "Reduce -D 3 printed SDMA threshold {}, expected {}".format(
            printed.group(1) if printed else "nothing", threshold))


# With no threshold set: the printed threshold is the default for -V, the
# largest aligned size below it runs tiled and the next one up takes the SDMA tier.
@_reduce_gin_skip
@pytest.mark.gin_ci
@pytest.mark.parametrize("where, tier", [("below", "tiled"), ("at", "sdma-staged")])
def test_ReduceGinSdmaDefaultThreshold(request, where, tier):
    ctas = int(REDUCE_CTAS)
    threshold = _default_sdma_threshold(ctas, REDUCE_NP)
    if threshold is None:
        pytest.skip("no default SDMA tier at {} CTAs with {} ranks".format(ctas, REDUCE_NP))
    if threshold > 64 * MiB:
        pytest.skip("default SDMA threshold {} at {} CTAs is too large for a quick test".format(threshold, ctas))
    total_bytes = threshold - REDUCE_NP * 16 if where == "below" else threshold
    rc, out = _run_reduce_gin_sdma(request, total_bytes, "float", "sum", 0, oop_only=True)
    _assert_reduce_ok(rc, out, "-D 3 default SDMA threshold, {} ({} bytes)".format(where, total_bytes),
                      total_bytes, REDUCE_NP, expect_tier=tier,
                      in_place=False)
    printed = REDUCE_SDMA_THRESHOLD_RE.search(out or "")
    assert printed and int(printed.group(1)) == threshold, (
        "Reduce -D 3 printed SDMA threshold {}, expected the default {}".format(
            printed.group(1) if printed else "nothing", threshold))


# The SDMA tier (GinSdmaReduceKernel), forced with a threshold of 0 and run
# out-of-place only. With fewer than 2N CTAs owners fold and store straight to
# the root ("cu"); with at least 2N, dedicated CTAs issue the SDMA puts and
# owners stage results for an SDMA copy to the root ("sdma").
def _check_sdma(request, total_bytes, dtype, op, root, path, np=None, xenv=()):
    """Run the SDMA tier on one path and check its rows and tier."""
    n = np or REDUCE_NP
    ctas = n if path == "cu" else max(2 * n, 16)
    if ctas > 128:
        pytest.skip("reduce_perf -V is capped at 128 CTAs")
    rc, out = _run_reduce_gin_sdma(request, total_bytes, dtype, op, root, ctas=ctas, np=np,
                                   xenv=xenv, oop_only=True, sdma_threshold=0)
    _assert_reduce_ok(rc, out, "-D 3 SDMA tier on {} ranks at {} bytes, dtype={}, op={}, path={}, root={}".format(
        n, total_bytes, dtype, op, path, root), total_bytes, n,
        expect_tier="sdma-" + ("cu" if path == "cu" else "staged"), in_place=False)


# 64 MiB splits each partition into several chunks.
@_reduce_gin_skip
@pytest.mark.parametrize("total_mib", [1, 64])
@pytest.mark.parametrize("dtype, op", [
    ("float", "sum"),
    ("bfloat16", "avg"),
])
@pytest.mark.parametrize("path", ["cu", "sdma"])
@pytest.mark.parametrize("root_pos", ["first", "last"])
def test_ReduceGinSdmaPut(request, total_mib, dtype, op, path, root_pos):
    root = 0 if root_pos == "first" else REDUCE_NP - 1
    _check_sdma(request, total_mib * MiB, dtype, op, root, path)


# SDMA-tier element types and ops beyond the float/bf16 pairs above, on both paths.
@_reduce_gin_skip
@pytest.mark.parametrize("dtype, op", [
    ("half", "max"),
    ("int32", "min"),
    ("int8", "sum"),
    ("double", "prod"),
    ("fp8_e4m3", "sum"),
    ("fp8_e5m2", "max"),
])
@pytest.mark.parametrize("path", ["cu", "sdma"])
def test_ReduceGinSdmaPutTypes(request, dtype, op, path):
    _check_sdma(request, MiB, dtype, op, REDUCE_NP - 1, path)


# NCCL_GIN_ANVIL_SDMA_REDUCE_CHUNKS=16 also splits the CU path's puts, up to one
# chunk per CTA. N*N packs make N-1 partitions of N+1 packs (4 with two ranks),
# so most chunks are empty, and issuers and reducers must skip the same ones;
# 1 MiB has no empty chunks. Above two ranks both sizes leave tail packs that
# the root reduces.
@_reduce_gin_skip
@pytest.mark.parametrize("size", ["n-squared-packs", "1MiB"])
@pytest.mark.parametrize("path", ["cu", "sdma"])
def test_ReduceGinSdmaPutChunks(request, size, path):
    total_bytes = {"n-squared-packs": REDUCE_NP * REDUCE_NP * 16, "1MiB": MiB}[size]
    _check_sdma(request, total_bytes, "float", "sum", REDUCE_NP - 1, path,
                xenv=["NCCL_GIN_ANVIL_SDMA_REDUCE_CHUNKS=16"])


# The SDMA tier with two ranks: one owner, which is also the only sender to the root.
@_reduce_gin_skip
@pytest.mark.gin_ci
@pytest.mark.parametrize("total_mib", [1, 64])
@pytest.mark.parametrize("path", ["cu", "sdma"])
@pytest.mark.parametrize("root", [0, 1])
def test_ReduceGinSdmaPutTwoRanks(request, total_mib, path, root):
    _check_sdma(request, total_mib * MiB, "float", "sum", root, path, np=2)
