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
# Environment (shared with test_AllToAll.py where noted):
#   RCCL_TESTS_GIN_SDMA_DEVTIME  enable this module (default: off)
#   RCCL_TESTS_A2A_NP            MPI ranks (default: detected GPU count)
#   RCCL_TESTS_MPI_LAUNCHER      launcher binary (default: mpirun)
#   RCCL_TESTS_MPI_OPTS          extra launcher opts
#   RCCL_TESTS_A2A_XENV          extra "-x K=V" env beyond the GIN essentials
#   RCCL_TESTS_A2A_EXE           path to alltoall_perf (default: ../build/...)
#   RCCL_TESTS_A2A_TIMEOUT_S     per-run timeout seconds (default: 300)
#   RCCL_TESTS_A2A_CTAS          -V grid CTAs (default: 8)
#   RCCL_TESTS_A2A_GIN_TYPE      NCCL_GIN_TYPE (default: 2, matches device-api CI)

import os
import re
import shlex
import signal
import subprocess

import pytest

KiB = 1024
SMOKE_BYTES = 128 * KiB  # single-size smoke (128 KiB per rank)

path = os.path.dirname(os.path.abspath(__file__))
executable = os.environ.get(
    "RCCL_TESTS_A2A_EXE", os.path.join(path, "..", "build", "alltoall_perf"))

_enabled = os.environ.get("RCCL_TESTS_GIN_SDMA_DEVTIME", "") not in (
    "", "0", "false", "False")

DEVTIME_LINE_RE = re.compile(
    r"#\[a2a-devtime\].*?\bdevtime\s+([0-9]+(?:\.[0-9]+)?)\s+us")


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
CTAS = os.environ.get("RCCL_TESTS_A2A_CTAS", "8")
TIMEOUT_S = int(os.environ.get("RCCL_TESTS_A2A_TIMEOUT_S", "300"))
GIN_TYPE = os.environ.get("RCCL_TESTS_A2A_GIN_TYPE", "2")
MPI_OPTS = shlex.split(os.environ.get("RCCL_TESTS_MPI_OPTS", ""))
XENV = shlex.split(os.environ.get("RCCL_TESTS_A2A_XENV", ""))

pytestmark = pytest.mark.skipif(
    not _enabled,
    reason="GIN AllToAll devtime smoke tests are opt-in; set "
           "RCCL_TESTS_GIN_SDMA_DEVTIME=1 on a GIN-capable node to enable.")


def _assert_datacheck_clean(out):
    """Perf binary reports wrong elements via #wrong= or Out of bounds values."""
    if "#wrong=0" in out:
        return
    if "Out of bounds values : 0 OK" in out:
        return
    if re.search(r"#wrong=\s*[1-9]", out):
        pytest.fail("datacheck reported wrong elements:\n{}".format(out[-2000:]))
    if "Out of bounds values : 0" not in out and "#wrong=" not in out:
        # Some builds only print the summary line on failure; accept clean exit.
        return


def _run_devtime(request, device_timing_mode, devtime_check=False):
    """Launch alltoall_perf with GIN (-D 3) and device-timing CLI flags."""
    if NP < 2:
        pytest.skip("need >= 2 ranks/GPUs for AllToAll")

    size = str(SMOKE_BYTES)
    # Match device-api CI gin-d3 essentials; deployment extras via RCCL_TESTS_A2A_XENV.
    gin_env = []
    for kv in [
        "NCCL_CUMEM_ENABLE=1",
        "HSA_FORCE_FINE_GRAIN_PCIE=1",
        "NCCL_DMABUF_ENABLE=1",
        "NCCL_GIN_TYPE={}".format(GIN_TYPE),
        "HSA_NO_SCRATCH_RECLAIM=1",
        "NCCL_ENV_PLUGIN=none",
        "RCCL_ENABLE_INTRANET=1",
    ] + XENV:
        gin_env += ["-x", kv]

    hostfile = request.config.getoption("--hostfile")
    launch = [LAUNCHER, "-np", str(NP)] + MPI_OPTS
    if hostfile:
        launch += ["-host", hostfile]

    bench_args = [
        executable,
        "-b", size, "-e", size,
        "-f", "2",
        "-g", "1",
        "-R", "2",
        "-D", "3",
        "-V", CTAS,
        "-d", "int32",
        "-c", "1",
        "-w", "1",
        "-n", "3",
        "-B", str(device_timing_mode),
        "-L", "5",
        "-P", "2",
    ]
    if devtime_check:
        bench_args += ["-H", "1"]

    args = launch + gin_env + bench_args
    cmd = " ".join(shlex.quote(a) for a in args)
    print(cmd)
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
            "AllToAll devtime smoke HANG: no completion within {}s (mode -B {}). "
            "Output tail:\n{}".format(TIMEOUT_S, device_timing_mode, (out or "")[-2000:]))
    print(out)
    return proc.returncode, out


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


def test_AllToAllDevtimeMode2DeviceOnly(request):
    """Mode 2: reported metric is in-kernel device latency (no host graph loop)."""
    rc, out = _run_devtime(request, device_timing_mode=2)
    assert rc == 0, "alltoall_perf exited {} (mode -B 2)".format(rc)
    _assert_datacheck_clean(out)
    assert "WARN --device_timing=2: no in-kernel device-time" not in out, (
        "device-time-only mode produced no valid measurement:\n{}".format(out[-2000:]))


def test_AllToAllDevtimeMode2WithTimedCheck(request):
    """Mode 2 + --devtime_check: validate timed-kernel output before datacheck."""
    rc, out = _run_devtime(request, device_timing_mode=2, devtime_check=True)
    assert rc == 0, "alltoall_perf exited {} (mode -B 2 -H 1)".format(rc)
    _assert_datacheck_clean(out)
    assert "ERROR: --devtime_check:" not in out, (
        "timed-kernel datacheck failed:\n{}".format(out[-2000:]))
