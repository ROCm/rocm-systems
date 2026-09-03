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
import glob
import shlex
import shutil
import signal
import subprocess
import itertools
import math
import tempfile
import time

import pytest

ngpus = 0
if os.environ.get("ROCR_VISIBLE_DEVICES") is not None:
    ngpus = len(os.environ["ROCR_VISIBLE_DEVICES"].split(","))
elif os.environ.get("HIP_VISIBLE_DEVICES") is not None:
    ngpus = len(os.environ["HIP_VISIBLE_DEVICES"].split(","))
else:
    ngpus = int(
        subprocess.check_output(
            'rocminfo | grep "Device Type:.\\s*.GPU" | wc -l', shell=True
        )
    )
log_ngpus = int(math.log2(ngpus))

nthreads = ["1"]
nprocs = ["2"]
ngpus_single = [str(2**x) for x in range(log_ngpus + 1)]
ngpus_mpi = ["1", "2"]
byte_range = [("4", "128M")]
op = ["sum", "prod", "min", "max"]
step_factor = ["2"]
datatype = [
    "int8",
    "uint8",
    "int32",
    "uint32",
    "int64",
    "uint64",
    "half",
    "float",
    "double",
]
memory_type = ["coarse", "fine", "host"]

path = os.path.dirname(os.path.abspath(__file__))
executable = path + "/../build/broadcast_perf"


@pytest.mark.parametrize(
    "nthreads, ngpus_single, byte_range, op, step_factor, datatype, memory_type",
    itertools.product(
        nthreads, ngpus_single, byte_range, op, step_factor, datatype, memory_type
    ),
)
def test_BroadcastSingleProcess(
    nthreads, ngpus_single, byte_range, op, step_factor, datatype, memory_type
):
    try:
        args = [
            executable,
            "-t",
            nthreads,
            "-g",
            ngpus_single,
            "-b",
            byte_range[0],
            "-e",
            byte_range[1],
            "-o",
            op,
            "-f",
            step_factor,
            "-d",
            datatype,
            "-Y",
            memory_type,
        ]
        if memory_type == "fine":
            args.insert(0, "HSA_FORCE_FINE_GRAIN_PCIE=1")
        args_str = " ".join(args)
        rccl_test = subprocess.run(
            args_str, stdout=subprocess.PIPE, universal_newlines=True, shell=True
        )
    except subprocess.CalledProcessError as err:
        print(rccl_test.stdout)
        pytest.fail("Broadcast test error(s) detected.")

    assert rccl_test.returncode == 0


# ---------------------------------------------------------------------------
# GIN-SDMA Broadcast multi-segment regression tests (parity with AllGather /
# AllToAll in test_AllGather.py / test_AllToAll.py).
#
# These drive the real GinHybridBroadcastKernel (deviceImpl 3, NCCL_GIN_TYPE=6)
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
#   RCCL_TESTS_BCAST_TIMEOUT_S, RCCL_TESTS_BCAST_CONN_RETRIES,
#   RCCL_TESTS_BCAST_GIN_TYPE   NCCL_GIN_TYPE (default: 6). The ANVIL_SDMA enum
#     is 6 on develop but 5 on the NCCL 2.30.7 line, so this must be settable
#     rather than baked in -- a wrong value silently loads no GIN plugin.

MiB = 1024 * 1024
GiB = 1024 * MiB

_bcast_enabled = os.environ.get("RCCL_TESTS_GIN_SDMA_BCAST", "") not in (
    "",
    "0",
    "false",
    "False",
)


def _env_int(name, default):
    """Parse an integer env var; bad values fall back so collection stays usable."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


BCAST_NP = _env_int("RCCL_TESTS_BCAST_NP", 0) or ngpus
BCAST_LAUNCHER = os.environ.get("RCCL_TESTS_MPI_LAUNCHER", "mpirun")
BCAST_CTAS = os.environ.get("RCCL_TESTS_BCAST_CTAS", "8")
BCAST_GIN_TYPE = os.environ.get("RCCL_TESTS_BCAST_GIN_TYPE", "6")
BCAST_TIMEOUT_S = _env_int("RCCL_TESTS_BCAST_TIMEOUT_S", 900)
BCAST_CONN_RETRIES = _env_int("RCCL_TESTS_BCAST_CONN_RETRIES", 5)
BCAST_MPI_OPTS = shlex.split(os.environ.get("RCCL_TESTS_MPI_OPTS", ""))
BCAST_XENV = shlex.split(os.environ.get("RCCL_TESTS_BCAST_XENV", ""))
BCAST_EXE = os.environ.get(
    "RCCL_TESTS_BCAST_EXE", os.path.join(path, "..", "build", "broadcast_perf")
)

_bcast_skip = pytest.mark.skipif(
    not _bcast_enabled,
    reason="GIN-SDMA Broadcast tests are opt-in; set RCCL_TESTS_GIN_SDMA_BCAST=1 on "
    "a GIN-SDMA-capable (e.g. 8x MI355X) node to enable.",
)

_CONN_GATE_RE = re.compile(
    r"LSA signal connectivity gate failed|unhandled system error", re.I
)

# The GIN Anvil-SDMA backend has to actually bind for any of this to mean
# something: a run that falls back to another transport still exits 0 and still
# prints a full results table. NCCL_DEBUG_SUBSYS=INIT,NET below is what puts the
# bind line in the captured output.
_PLUGIN_RE = re.compile(r"gin-anvil-sdma", re.I)

# One measured rccl-tests results row:
#   size count type redop root | time algbw busbw #wrong | time algbw busbw #wrong
# out-of-place columns first, then in-place. Timings come from getFloatStr, which
# falls back to scientific notation when a value will not fit its width, so the
# numeric fields have to admit exponents. The row is not end-anchored: an algo /
# proto / nchannels group and a timestamp may follow the in-place columns, and
# concurrent rank output can splice itself onto the end of the line.
_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_WRONG = r"(?:{}|N/A)".format(_NUM)
_ROW_RE = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+\S+\s+\S+\s+(\d+)"
    r"\s+{n}\s+{n}\s+{n}\s+({w})"
    r"\s+{n}\s+{n}\s+{n}\s+({w})".format(n=_NUM, w=_WRONG)
)
_OOB_RE = re.compile(r"Out of bounds values\s*:\s*(\d+)")


def _bcast_rows(out):
    """Measured rows as (size, root, wrong_out_of_place, wrong_in_place)."""
    rows = []
    for line in (out or "").splitlines():
        m = _ROW_RE.match(line)
        if m:
            rows.append((int(m.group(1)), int(m.group(3)), m.group(4), m.group(5)))
    return rows


def _wrong_count(field):
    """Numeric #wrong, or None when the column is N/A because checks were off."""
    try:
        return float(field)
    except ValueError:
        return None


def _data_failed(out):
    """True when the run produced demonstrably wrong results.

    Deliberately not a text match on "Wrong": rccl-tests prints a "#wrong" column
    header on every run that reaches the results table, so a bare pattern matched
    every healthy run and made the connectivity retry below unreachable. N/A is
    not a failure here -- it means checking was off, which _assert_bcast_ok
    rejects separately rather than burning retries on.
    """
    for _, _, oop, ip in _bcast_rows(out):
        if any(_wrong_count(f) not in (None, 0.0) for f in (oop, ip)):
            return True
    m = _OOB_RE.search(out or "")
    return bool(m and m.group(1) != "0")


def _read_debug_logs(debug_dir):
    """Merged per-rank NCCL debug output written under debug_dir."""
    chunks = []
    for path in sorted(glob.glob(os.path.join(debug_dir, "nccl-debug.*"))):
        try:
            with open(path, errors="replace") as fh:
                chunks.append(fh.read())
        except OSError:
            pass
    return "\n".join(chunks)


def _assert_bcast_ok(rc, out, debug, ctx):
    """Assert the run bound GIN-SDMA, measured something, and checked clean.

    Exit status alone is not enough: it stays 0 for a run that binds a different
    backend, and for one that produces no measured rows at all.
    """
    tail = (out or "")[-2000:]
    assert rc == 0, "Broadcast {} failed (exit {}). Output tail:\n{}".format(
        ctx, rc, tail
    )
    assert _PLUGIN_RE.search(debug or "") or _PLUGIN_RE.search(out or ""), (
        "Broadcast {} never bound the GIN Anvil-SDMA backend, so the run proves "
        "nothing about the GIN path. Output tail:\n{}".format(ctx, tail)
    )
    rows = _bcast_rows(out)
    assert rows, "Broadcast {} produced no measured rows. Output tail:\n{}".format(
        ctx, tail
    )
    unchecked = [
        (s, r)
        for s, r, oop, ip in rows
        if _wrong_count(oop) is None or _wrong_count(ip) is None
    ]
    assert not unchecked, (
        "Broadcast {} reported #wrong as N/A at (size, root) {}, so the data check "
        "never ran and the result is unverified.".format(ctx, unchecked)
    )
    bad = [
        (s, r, oop, ip)
        for s, r, oop, ip in rows
        if _wrong_count(oop) != 0.0 or _wrong_count(ip) != 0.0
    ]
    assert (
        not bad
    ), "Broadcast {} reported nonzero #wrong (size, root, oop, ip): {}".format(ctx, bad)
    m = _OOB_RE.search(out or "")
    assert m and m.group(1) == "0", "Broadcast {} out-of-bounds count is {}".format(
        ctx, m.group(1) if m else "absent"
    )


def _launch_bcast_gin_sdma(
    request, msg_bytes, dtype, *, force_flat_gin=False, force_sag_gin=False
):
    """Launch broadcast_perf -D 3 at a fixed message size. When force_flat_gin,
    disable the SAG/ring large tiers so the root flat gin.put() path is used.
    When force_sag_gin, disable only the ring tier so scatter+allgather runs.

    Returns (returncode, stdout, debug_text); debug_text is the merged per-rank
    NCCL debug log, kept off stdout so the results table stays parseable."""
    size = str(int(msg_bytes))
    debug_dir = tempfile.mkdtemp(prefix="bcast-gin-dbg-")
    gin_env = [
        "NCCL_GIN_ENABLE=1",
        "NCCL_GIN_TYPE={}".format(BCAST_GIN_TYPE),
        "NCCL_GIN_ANVIL_SDMA_THRESHOLD=0",
        "NCCL_GIN_ANVIL_SDMA_THRESHOLD_BROADCAST=0",
        # Debug output is what lets _PLUGIN_RE confirm which backend actually
        # bound. It goes to per-rank files rather than stdout because ranks
        # interleave: on stdout an INFO line splices itself into the middle of a
        # results row, which corrupts the very table we need to parse.
        "NCCL_DEBUG=INFO",
        "NCCL_DEBUG_SUBSYS=INIT,NET",
        "NCCL_DEBUG_FILE={}/nccl-debug.%h.%p.log".format(debug_dir),
    ]
    if force_flat_gin:
        gin_env += [
            "NCCL_GIN_ANVIL_BCAST_SCATTER_AG_MIN_BYTES=0",
            "NCCL_GIN_ANVIL_BCAST_RING_MIN_BYTES=0",
        ]
    elif force_sag_gin:
        # Ring is checked before SAG; disable ring only so SAG is exercised.
        gin_env += ["NCCL_GIN_ANVIL_BCAST_RING_MIN_BYTES=0"]
    gin_env += BCAST_XENV
    gin_env = sum([["-x", kv] for kv in gin_env], [])

    hostfile = request.config.getoption("--hostfile")
    launch = [BCAST_LAUNCHER, "-np", str(BCAST_NP)] + BCAST_MPI_OPTS
    if hostfile:
        launch += ["-host", hostfile]

    args = (
        launch
        + gin_env
        + [
            BCAST_EXE,
            "-b",
            size,
            "-e",
            size,
            "-f",
            "2",
            "-g",
            "1",
            "-R",
            "2",
            "-D",
            "3",  # GinHybridBroadcastKernel
            "-A",
            "1",
            "-V",
            BCAST_CTAS,
            "-d",
            dtype,
            "-c",
            "1",
            "-w",
            "1",
            "-n",
            "3",
        ]
    )
    cmd = " ".join(shlex.quote(a) for a in args)
    print(cmd)
    try:
        proc = subprocess.Popen(
            cmd,
            shell=True,
            universal_newlines=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
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
                    BCAST_TIMEOUT_S, size, msg_bytes // MiB, dtype, (out or "")[-2000:]
                )
            )
        debug = _read_debug_logs(debug_dir)
    finally:
        shutil.rmtree(debug_dir, ignore_errors=True)
    print(out)
    return proc.returncode, out, debug


def _run_bcast_gin_sdma(
    request, msg_bytes, dtype, *, force_flat_gin=False, force_sag_gin=False
):
    if BCAST_NP < 2:
        pytest.skip("need >= 2 ranks/GPUs for GIN-SDMA Broadcast")

    rc, out, debug = 1, "", ""
    for attempt in range(1, max(1, BCAST_CONN_RETRIES) + 1):
        rc, out, debug = _launch_bcast_gin_sdma(
            request,
            msg_bytes,
            dtype,
            force_flat_gin=force_flat_gin,
            force_sag_gin=force_sag_gin,
        )
        if rc == 0:
            return rc, out, debug
        if _data_failed(out):
            return rc, out, debug
        if _CONN_GATE_RE.search(out or "") and attempt < BCAST_CONN_RETRIES:
            print(
                "=== connectivity-gate abort (attempt {}/{}); re-launching after settle ===".format(
                    attempt, BCAST_CONN_RETRIES
                )
            )
            time.sleep(3)
            continue
        return rc, out, debug
    return rc, out, debug


@_bcast_skip
@pytest.mark.parametrize("msg_mib", [256, 2048])  # 256 MiB (2 seg), 2 GiB (16 seg)
@pytest.mark.parametrize("dtype", ["int32", "int64", "uint8"])
def test_BroadcastGinSdmaLargeSegmented(request, msg_mib, dtype):
    """Flat root-fanout gin.put() at sizes crossing the 128 MiB SDMA segment."""
    rc, out, debug = _run_bcast_gin_sdma(
        request, msg_mib * MiB, dtype, force_flat_gin=True
    )
    _assert_bcast_ok(
        rc, out, debug, "flat-segmented {} MiB dtype={}".format(msg_mib, dtype)
    )


@_bcast_skip
def test_BroadcastGinSdmaScatterAllgather(request):
    """256 MiB scatter+allgather tier (ring disabled via RING_MIN_BYTES=0)."""
    rc, out, debug = _run_bcast_gin_sdma(
        request, 256 * MiB, "int32", force_sag_gin=True
    )
    _assert_bcast_ok(rc, out, debug, "SAG 256 MiB")


@_bcast_skip
def test_BroadcastGinSdma2GiBHangGuard(request):
    """2 GiB completion guard with default tier selection (ring path)."""
    rc, out, debug = _run_bcast_gin_sdma(
        request, 2 * GiB, "int32", force_flat_gin=False
    )
    _assert_bcast_ok(rc, out, debug, "2 GiB default-tier")


@_bcast_skip
def test_BroadcastGinSdma4GiBHangGuard(request):
    """4 GiB completion guard with default tier selection (ring path at 4 GiB)."""
    rc, out, debug = _run_bcast_gin_sdma(
        request, 4 * GiB, "int32", force_flat_gin=False
    )
    _assert_bcast_ok(rc, out, debug, "4 GiB default-tier")


# Offline parsing guards: no GIN hardware required. These pin the regression where
# _DATA_FAIL_RE matched the "#wrong" column header and made the connectivity retry
# unreachable, and where scientific-notation timings broke the row regex.
def test_bcast_row_regex_accepts_scientific_notation():
    line = (
        "  134217728  134217728  int32  none  0"
        "  1.23e+02  1.00e+02  1.00e+02  0"
        "  1.23e+02  1.00e+02  1.00e+02  0"
    )
    rows = _bcast_rows(line)
    assert rows == [(134217728, 0, "0", "0")]


def test_bcast_data_failed_ignores_column_header():
    header = (
        "#       size         count      type   redop    root"
        "     time   algbw   busbw  #wrong"
    )
    assert not _data_failed(header)


def test_bcast_data_failed_detects_nonzero_wrong():
    line = (
        "  1048576  1048576  int32  none  0"
        "  12.34  1.00  1.00  1"
        "  12.34  1.00  1.00  0"
    )
    assert _data_failed(line)


def test_bcast_data_failed_treats_na_as_unchecked_not_failed():
    line = (
        "  1048576  1048576  int32  none  0"
        "  12.34  1.00  1.00  N/A"
        "  12.34  1.00  1.00  0"
    )
    assert not _data_failed(line)


@pytest.mark.parametrize(
    "nthreads, nprocs, ngpus_mpi, byte_range, op, step_factor, datatype",
    itertools.product(
        nthreads, nprocs, ngpus_mpi, byte_range, op, step_factor, datatype
    ),
)
def test_BroadcastMPI(
    request, nthreads, nprocs, ngpus_mpi, byte_range, op, step_factor, datatype
):
    try:
        mpi_hostfile = request.config.getoption("--hostfile")
        if not mpi_hostfile:
            args = [
                "mpirun -np",
                nprocs,
                executable,
                "-p 1",
                "-t",
                nthreads,
                "-g",
                ngpus_mpi,
                "-b",
                byte_range[0],
                "-e",
                byte_range[1],
                "-o",
                op,
                "-f",
                step_factor,
                "-d",
                datatype,
            ]
        else:
            args = [
                "mpirun -np",
                nprocs,
                "-host",
                mpi_hostfile,
                executable,
                "-p 1",
                "-t",
                nthreads,
                "-g",
                ngpus_mpi,
                "-b",
                byte_range[0],
                "-e",
                byte_range[1],
                "-o",
                op,
                "-f",
                step_factor,
                "-d",
                datatype,
            ]
        args_str = " ".join(args)
        print(args_str)
        rccl_test = subprocess.run(args_str, universal_newlines=True, shell=True)
    except subprocess.CalledProcessError as err:
        print(rccl_test.stdout)
        pytest.fail("Broadcast test error(s) detected.")

    assert rccl_test.returncode == 0
