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

# GPU smoke tests for GIN AllToAll in-kernel device timing (wall_clock64).
#
# Exercises the real GinAlltoAllTimedKernel path via alltoall_perf -D 3 with
# --device_timing modes 1/2 and optional --devtime_check. Requires an MPI
# launcher, >= 2 GPUs/ranks, and rccl-tests built with ENABLE_DEVICE_API=ON
# against RCCL >= 2.28.7.
#
# Opt-in locally via RCCL_TESTS_GIN_SDMA_DEVTIME=1. The device-api CI job sets
# that automatically (see projects/rccl/tools/ci/run-device-api-ci.sh).
#
# MPI launch, GPU detection, and the fixed-size -D 3 argv come from
# gin_sdma_harness. Test names stay as they are: run-device-api-ci.sh runs this
# file directly, and its pytest args are expanded unquoted.
#
# Environment (shared with test_AllToAll.py where noted):
#   RCCL_TESTS_GIN_SDMA_DEVTIME  enable this module (default: off)
#   RCCL_TESTS_A2A_NP            MPI ranks (default: detected GPU count)
#   RCCL_TESTS_MPI_LAUNCHER      launcher binary (default: mpirun)
#   RCCL_TESTS_MPI_OPTS          extra launcher opts
#   RCCL_TESTS_A2A_XENV          extra "-x K=V" env beyond the GIN essentials
#   RCCL_TESTS_A2A_EXE           path to alltoall_perf (default: ../build/...)
#   RCCL_TESTS_A2A_TIMEOUT_S     per-run timeout seconds (default: 300)
#   RCCL_TESTS_A2A_CONN_RETRIES  connectivity-gate retries (default: 5)
#   RCCL_TESTS_A2A_CTAS          -V grid CTAs (default: 8)
#   RCCL_TESTS_A2A_GIN_TYPE      NCCL_GIN_TYPE (default: 2, matches device-api CI)

import os
import re
import shlex

import pytest

from .gin_sdma_harness import (
    detect_ngpus,
    env_int,
    gin_env_xflags,
    gin_hang_msg,
    gin_perf_argv,
    launch_mpi_shell,
    mpi_launch_prefix,
    run_with_conn_gate_retry,
)

KiB = 1024
SMOKE_BYTES = 128 * KiB  # single-size smoke (128 KiB per rank)

path = os.path.dirname(os.path.abspath(__file__))
executable = os.environ.get(
    "RCCL_TESTS_A2A_EXE", os.path.join(path, "..", "build", "alltoall_perf"))

_enabled = os.environ.get("RCCL_TESTS_GIN_SDMA_DEVTIME", "") not in (
    "", "0", "false", "False")

DEVTIME_LINE_RE = re.compile(
    r"#\[a2a-devtime\].*?\bdevtime\s+([0-9]+(?:\.[0-9]+)?)\s+us")

# Release alltoall_perf prints a results row, not the DEBUG_PRINT `#wrong=`
# line in common.cu. Root is -1 (AlltoAllRunTest). The first #wrong column is
# out-of-place; in-place is N/A because alltoall.cu sets reportErrors = 0 there.
_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_WRONG = r"(?:{}|N/A)".format(_NUM)
_A2A_OOP_WRONG_RE = re.compile(
    r"^\s*\d+\s+\d+\s+\S+\s+\S+\s+-?\d+"
    r"\s+{n}\s+{n}\s+{n}\s+({w})".format(n=_NUM, w=_WRONG),
    re.M,
)
_OOB_RE = re.compile(r"Out of bounds values\s*:\s*(\d+)")


# detect_ngpus() raises when rocminfo is missing. Only call it when this module
# is enabled and RCCL_TESTS_A2A_NP is unset, so a GPU-free collection still works.
NP = env_int("RCCL_TESTS_A2A_NP", 0) or (detect_ngpus() if _enabled else 0)
LAUNCHER = os.environ.get("RCCL_TESTS_MPI_LAUNCHER", "mpirun")
CTAS = os.environ.get("RCCL_TESTS_A2A_CTAS", "8")
TIMEOUT_S = env_int("RCCL_TESTS_A2A_TIMEOUT_S", 300)
CONN_RETRIES = env_int("RCCL_TESTS_A2A_CONN_RETRIES", 5)
GIN_TYPE = os.environ.get("RCCL_TESTS_A2A_GIN_TYPE", "2")
MPI_OPTS = shlex.split(os.environ.get("RCCL_TESTS_MPI_OPTS", ""))
XENV = shlex.split(os.environ.get("RCCL_TESTS_A2A_XENV", ""))

_gpu = pytest.mark.skipif(
    not _enabled,
    reason="GIN AllToAll devtime smoke tests are opt-in; set "
           "RCCL_TESTS_GIN_SDMA_DEVTIME=1 on a GIN-capable node to enable.")


def _wrong_count(field):
    """Numeric #wrong, or None when the column is N/A because checks were off."""
    try:
        return float(field)
    except ValueError:
        return None


def _assert_datacheck_clean(out):
    """Fail unless the release-build datacheck columns are present and zero.

    `#wrong=` is printed only under DEBUG_PRINT (common.cu). A release binary
    reports the out-of-place `#wrong` table column and
    `Out of bounds values : N OK|FAILED`. A missing line is a failure: treating
    that silence as clean let `Out of bounds values : 3 FAILED` pass.
    """
    text = out or ""
    tail = text[-2000:]
    wrongs = _A2A_OOP_WRONG_RE.findall(text)
    if not wrongs:
        pytest.fail(
            "datacheck produced no out-of-place #wrong column. "
            "Release builds do not print #wrong=. tail:\n{}".format(tail))
    unchecked = [w for w in wrongs if _wrong_count(w) is None]
    if unchecked:
        pytest.fail(
            "out-of-place #wrong is N/A {}, so the data check never ran. "
            "tail:\n{}".format(unchecked, tail))
    bad = [w for w in wrongs if _wrong_count(w) != 0.0]
    if bad:
        pytest.fail(
            "out-of-place #wrong is nonzero {}. tail:\n{}".format(bad, tail))
    m = _OOB_RE.search(text)
    if not m or m.group(1) != "0":
        pytest.fail(
            "out-of-bounds count is {}. tail:\n{}".format(
                m.group(1) if m else "absent", tail))


def _launch_devtime(request, device_timing_mode, devtime_check=False):
    """Launch alltoall_perf once with GIN (-D 3) and device-timing CLI flags."""
    size = str(SMOKE_BYTES)
    # Match device-api CI gin-d3 essentials; deployment extras via RCCL_TESTS_A2A_XENV.
    gin_env = gin_env_xflags(
        [
            "NCCL_CUMEM_ENABLE=1",
            "HSA_FORCE_FINE_GRAIN_PCIE=1",
            "NCCL_DMABUF_ENABLE=1",
            "NCCL_GIN_TYPE={}".format(GIN_TYPE),
            "HSA_NO_SCRATCH_RECLAIM=1",
            "NCCL_ENV_PLUGIN=none",
            "RCCL_ENABLE_INTRANET=1",
        ]
        + XENV
    )
    timing = ["-B", str(device_timing_mode), "-L", "5", "-P", "2"]
    if devtime_check:
        timing += ["-H", "1"]
    args = (
        mpi_launch_prefix(request, LAUNCHER, NP, MPI_OPTS)
        + gin_env
        + gin_perf_argv(executable, size, "int32", CTAS)
        + timing
    )
    cmd = " ".join(shlex.quote(a) for a in args)
    hang_msg = gin_hang_msg(
        "AllToAll devtime smoke",
        TIMEOUT_S,
        size,
        "int32",
        "mode -B {}".format(device_timing_mode),
    )
    return launch_mpi_shell(cmd, TIMEOUT_S, hang_msg)


def _run_devtime(request, device_timing_mode, devtime_check=False):
    """Retry the devtime launch on gfx950 connectivity-gate aborts."""
    if NP < 2:
        pytest.skip("need >= 2 ranks/GPUs for AllToAll")
    return run_with_conn_gate_retry(
        lambda: _launch_devtime(request, device_timing_mode, devtime_check),
        CONN_RETRIES,
    )


@_gpu
def test_AllToAllDevtimeMode1Augment(request):
    """Mode 1: normal bench plus #[a2a-devtime] from wall_clock64 timed kernel."""
    rc, out = _run_devtime(request, device_timing_mode=1)
    assert rc == 0, "alltoall_perf exited {} (mode -B 1)".format(rc)
    _assert_datacheck_clean(out)
    match = DEVTIME_LINE_RE.search(out)
    assert match is not None, (
        "expected #[a2a-devtime] line in stdout (mode -B 1); tail:\n{}".format(out[-2000:]))
    devtime_us = float(match.group(1))
    assert devtime_us > 0.0, "devtime must be positive, got {} us".format(devtime_us)


@_gpu
def test_AllToAllDevtimeMode2DeviceOnly(request):
    """Mode 2: reported metric is in-kernel device latency (no host graph loop)."""
    rc, out = _run_devtime(request, device_timing_mode=2)
    assert rc == 0, "alltoall_perf exited {} (mode -B 2)".format(rc)
    _assert_datacheck_clean(out)
    assert "WARN --device_timing=2: no in-kernel device-time" not in out, (
        "device-time-only mode produced no valid measurement:\n{}".format(out[-2000:]))


@_gpu
def test_AllToAllDevtimeMode2WithTimedCheck(request):
    """Mode 2 + --devtime_check: validate timed-kernel output before datacheck."""
    rc, out = _run_devtime(request, device_timing_mode=2, devtime_check=True)
    assert rc == 0, "alltoall_perf exited {} (mode -B 2 -H 1)".format(rc)
    _assert_datacheck_clean(out)
    assert "ERROR: --devtime_check:" not in out, (
        "timed-kernel datacheck failed:\n{}".format(out[-2000:]))


def _sample_output(oop_wrong="0", oob="0", oob_tag="OK"):
    """One release-build alltoall_perf row. In-place #wrong stays N/A."""
    return "\n".join([
        "#       size         count      type   redop    root"
        "     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong",
        "      131072         32768     int32    none      -1"
        "     12.34   1.00   1.00  {oop}"
        "     12.34   1.00   1.00     N/A".format(oop=oop_wrong),
        "# Out of bounds values : {oob} {tag}".format(oob=oob, tag=oob_tag),
        "#[a2a-devtime] devtime 1.5 us",
    ])


def test_AllToAllDevtimeDatacheckAcceptsCleanReleaseOutput():
    """A release run has no `#wrong=` line; the table column and OOB count do."""
    _assert_datacheck_clean(_sample_output())


def test_AllToAllDevtimeDatacheckAcceptsScientificNotation():
    line = (
        "  131072  32768  int32  none  -1"
        "  1.23e+02  1.00e+02  1.00e+02  0"
        "  1.23e+02  1.00e+02  1.00e+02  N/A\n"
        "# Out of bounds values : 0 OK\n"
    )
    _assert_datacheck_clean(line)


def test_AllToAllDevtimeDatacheckRejectsNonzeroOutOfBounds():
    """`Out of bounds values : 3 FAILED` used to return clean."""
    out = "# Out of bounds values : 3 FAILED\n"
    with pytest.raises(pytest.fail.Exception, match="out-of-place #wrong|#wrong="):
        _assert_datacheck_clean(out)
    with pytest.raises(pytest.fail.Exception, match="out-of-bounds count is 3"):
        _assert_datacheck_clean(_sample_output(oob="3", oob_tag="FAILED"))


def test_AllToAllDevtimeDatacheckRejectsDebugPrintWrongEquals():
    """The DEBUG_PRINT `#wrong=0` line is not a release-build data guard."""
    with pytest.raises(pytest.fail.Exception, match="no out-of-place #wrong"):
        _assert_datacheck_clean("rank=0 #wrong=0\n")


def test_AllToAllDevtimeDatacheckRejectsNonzeroWrongColumn():
    with pytest.raises(pytest.fail.Exception, match="nonzero"):
        _assert_datacheck_clean(_sample_output(oop_wrong="3"))


def test_AllToAllDevtimeDatacheckRejectsUncheckedNa():
    with pytest.raises(pytest.fail.Exception, match="N/A"):
        _assert_datacheck_clean(_sample_output(oop_wrong="N/A"))


def test_AllToAllDevtimeDatacheckIgnoresColumnHeader():
    header = (
        "#       size         count      type   redop    root"
        "     time   algbw   busbw  #wrong\n"
    )
    assert _A2A_OOP_WRONG_RE.findall(header) == []
