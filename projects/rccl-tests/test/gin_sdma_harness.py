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

"""Shared helpers for GIN-SDMA rccl-tests pytest harnesses (AllGather, AllToAll,
Broadcast).  MPI launcher assembly, connectivity-gate retry, and (for Broadcast)
NCCL debug-file capture live here so the per-collective test modules stay thin."""

import glob
import os
import re
import shlex
import shutil
import signal
import subprocess
import tempfile
import time

import pytest

MiB = 1024 * 1024
GiB = 1024 * MiB

# Intermittent gfx950 cuMem-VMM connectivity-gate abort (not a data error).
CONN_GATE_RE = re.compile(
    r"LSA signal connectivity gate failed|unhandled system error", re.I
)
DATA_FAIL_RE = re.compile(
    r"#wrong\s*=\s*[1-9]|mismatch|check.*fail|Out of bounds values\s*:\s*[1-9]",
    re.I,
)


def detect_ngpus():
    """Return the visible GPU count.

    Prefer ROCR_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES. Otherwise require
    rocminfo on PATH: a missing binary used to look like zero GPUs because
    `rocminfo | grep | wc -l` still exits 0, and callers that clamp with
    max(1, ...) then silently collect a one-GPU matrix.
    """
    if os.environ.get("ROCR_VISIBLE_DEVICES") is not None:
        return len(os.environ["ROCR_VISIBLE_DEVICES"].split(","))
    if os.environ.get("HIP_VISIBLE_DEVICES") is not None:
        return len(os.environ["HIP_VISIBLE_DEVICES"].split(","))
    rocminfo = shutil.which("rocminfo")
    if rocminfo is None:
        raise RuntimeError(
            "rocminfo not on PATH; cannot detect GPU count. Set "
            "ROCR_VISIBLE_DEVICES, HIP_VISIBLE_DEVICES, or RCCL_TESTS_*_NP."
        )
    try:
        out = subprocess.check_output(
            [rocminfo], universal_newlines=True, stderr=subprocess.DEVNULL
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(
            "rocminfo failed; cannot detect GPU count. Set "
            "ROCR_VISIBLE_DEVICES, HIP_VISIBLE_DEVICES, or RCCL_TESTS_*_NP."
        ) from exc
    return sum(
        1 for line in out.splitlines() if re.search(r"Device Type:\s*GPU", line)
    )


def env_int(name, default):
    """Parse an integer env var; bad values fall back so collection stays usable."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def mpi_launch_prefix(request, launcher, np, mpi_opts):
    """Assemble [launcher, -np, np, ...mpi_opts, (-host hostfile)?]."""
    hostfile = request.config.getoption("--hostfile")
    launch = [launcher, "-np", str(np)] + list(mpi_opts)
    if hostfile:
        launch += ["-host", hostfile]
    return launch


def gin_env_xflags(kv_list):
    """Turn ["K=V", ...] into mpirun -x flags."""
    flags = []
    for kv in kv_list:
        flags += ["-x", kv]
    return flags


def gin_perf_argv(exe, size, dtype, ctas, extra=None):
    """Fixed-size GIN -D 3 perf argv.

    `extra` is inserted after `-D 3` and before `-V`, so the AllToAll vs
    AllGather/Broadcast `-A 1` gap is an explicit extra rather than a
    copy-paste drift. AllToAll omits `-A 1`; the others pass extra=["-A", "1"].
    """
    args = [
        exe,
        "-b",
        str(size),
        "-e",
        str(size),
        "-f",
        "2",
        "-g",
        "1",
        "-R",
        "2",
        "-D",
        "3",
    ]
    args += list(extra or [])
    args += [
        "-V",
        str(ctas),
        "-d",
        dtype,
        "-c",
        "1",
        "-w",
        "1",
        "-n",
        "3",
    ]
    return args


def gin_hang_msg(collective, timeout_s, size, dtype, size_desc):
    """Timeout diagnostic; size_desc is e.g. '256 MiB/rank' or '256 MiB'."""
    return (
        "{} GIN-SDMA HANG: no completion within {}s at {} bytes "
        "({}), dtype={}. Output tail:\n{{}}".format(
            collective, timeout_s, size, size_desc, dtype
        )
    )


def launch_mpi_shell(cmd, timeout_s, hang_msg):
    """Run a shell MPI command in a new session; kill the process group on timeout."""
    print(cmd)
    proc = subprocess.Popen(
        cmd,
        shell=True,
        universal_newlines=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        out, _ = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        out, _ = proc.communicate()
        pytest.fail(hang_msg.format((out or "")[-2000:]))
    print(out)
    return proc.returncode, out


def run_with_conn_gate_retry(
    launch_fn, conn_retries, settle_s=3, is_data_failure=None
):
    """Retry launch_fn on connectivity-gate aborts; never retry genuine data failures.

    launch_fn must return (returncode, stdout)."""
    if is_data_failure is None:
        is_data_failure = lambda out: bool(DATA_FAIL_RE.search(out or ""))

    rc, out = 1, ""
    for attempt in range(1, max(1, conn_retries) + 1):
        rc, out = launch_fn()
        if rc == 0:
            return rc, out
        if is_data_failure(out):
            return rc, out
        if CONN_GATE_RE.search(out or "") and attempt < conn_retries:
            print(
                "=== connectivity-gate abort (attempt {}/{}); "
                "re-launching after settle ===".format(attempt, conn_retries)
            )
            time.sleep(settle_s)
            continue
        return rc, out
    return rc, out


def read_nccl_debug_logs(debug_dir):
    """Merged per-rank NCCL debug output written under debug_dir."""
    chunks = []
    for log_path in sorted(glob.glob(os.path.join(debug_dir, "nccl-debug.*"))):
        try:
            with open(log_path, errors="replace") as fh:
                chunks.append(fh.read())
        except OSError:
            pass
    return "\n".join(chunks)


# --- Broadcast-specific parsing / assertions ---------------------------------

PLUGIN_RE = re.compile(r"gin-anvil-sdma", re.I)

_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_WRONG = r"(?:{}|N/A)".format(_NUM)
# Deliberately not end-anchored: rccl-tests may append an algo/proto/nchannels
# group and a timestamp, and concurrent rank output can splice onto the line.
BCAST_ROW_RE = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+\S+\s+\S+\s+(\d+)"
    r"\s+{n}\s+{n}\s+{n}\s+({w})"
    r"\s+{n}\s+{n}\s+{n}\s+({w})".format(n=_NUM, w=_WRONG)
)
BCAST_OOB_RE = re.compile(r"Out of bounds values\s*:\s*(\d+)")
# Matches the tier emitted by bcastReportTier() in src/broadcast.cu.
BCAST_TIER_RE = re.compile(r"^#\[bcast-tier\]\s+(\S+)", re.M)


def bcast_tiers(out):
    """Set of -D 3 tier names the run reported."""
    return set(BCAST_TIER_RE.findall(out or ""))


def bcast_rows(out):
    """Measured rows as (size, root, wrong_out_of_place, wrong_in_place)."""
    rows = []
    for line in (out or "").splitlines():
        m = BCAST_ROW_RE.match(line)
        if m:
            rows.append((int(m.group(1)), int(m.group(3)), m.group(4), m.group(5)))
    return rows


def bcast_wrong_count(field):
    """Numeric #wrong, or None when the column is N/A because checks were off."""
    try:
        return float(field)
    except ValueError:
        return None


def bcast_data_failed(out):
    """True when the run produced demonstrably wrong results.

    Deliberately not a text match on "Wrong": rccl-tests prints a "#wrong" column
    header on every run that reaches the results table."""
    for _, _, oop, ip in bcast_rows(out):
        if any(bcast_wrong_count(f) not in (None, 0.0) for f in (oop, ip)):
            return True
    m = BCAST_OOB_RE.search(out or "")
    return bool(m and m.group(1) != "0")


def assert_bcast_ok(rc, out, debug, ctx, expect_tier=None):
    """Assert the run bound GIN-SDMA, measured something, and checked clean."""
    tail = (out or "")[-2000:]
    assert rc == 0, "Broadcast {} failed (exit {}). Output tail:\n{}".format(
        ctx, rc, tail
    )
    assert PLUGIN_RE.search(debug or "") or PLUGIN_RE.search(out or ""), (
        "Broadcast {} never bound the GIN Anvil-SDMA backend, so the run proves "
        "nothing about the GIN path. Output tail:\n{}".format(ctx, tail)
    )
    rows = bcast_rows(out)
    assert rows, "Broadcast {} produced no measured rows. Output tail:\n{}".format(
        ctx, tail
    )
    unchecked = [
        (s, r)
        for s, r, oop, ip in rows
        if bcast_wrong_count(oop) is None or bcast_wrong_count(ip) is None
    ]
    assert not unchecked, (
        "Broadcast {} reported #wrong as N/A at (size, root) {}, so the data check "
        "never ran and the result is unverified.".format(ctx, unchecked)
    )
    bad = [
        (s, r, oop, ip)
        for s, r, oop, ip in rows
        if bcast_wrong_count(oop) != 0.0 or bcast_wrong_count(ip) != 0.0
    ]
    assert (
        not bad
    ), "Broadcast {} reported nonzero #wrong (size, root, oop, ip): {}".format(
        ctx, bad
    )
    m = BCAST_OOB_RE.search(out or "")
    assert m and m.group(1) == "0", "Broadcast {} out-of-bounds count is {}".format(
        ctx, m.group(1) if m else "absent"
    )
    if expect_tier is not None:
        allowed = (
            {expect_tier} if isinstance(expect_tier, str) else set(expect_tier)
        )
        tiers = bcast_tiers(out)
        assert tiers, (
            "Broadcast {} reported no #[bcast-tier] line, so which tier ran is "
            "unverified. Is RCCL_TESTS_BCAST_TIER_LOG set for this launch, and is "
            "the binary new enough to emit it? Output tail:\n{}".format(ctx, tail)
        )
        assert len(tiers) == 1 and tiers <= allowed, (
            "Broadcast {} ran tier(s) {} but expected exactly one of {}. The tier "
            "gate moved, or the tier under test fell through to another "
            "kernel.".format(ctx, sorted(tiers), sorted(allowed))
        )


def launch_bcast_gin_sdma(
    request,
    msg_bytes,
    dtype,
    *,
    exe,
    np,
    launcher,
    mpi_opts,
    xenv,
    ctas,
    gin_type,
    timeout_s,
    force_flat_gin=False,
    force_sag_gin=False,
):
    """Launch broadcast_perf -D 3; returns (returncode, stdout, debug_text)."""
    size = str(int(msg_bytes))
    debug_dir = tempfile.mkdtemp(prefix="bcast-gin-dbg-")
    gin_env = [
        "NCCL_GIN_ENABLE=1",
        "NCCL_GIN_TYPE={}".format(gin_type),
        "NCCL_GIN_ANVIL_SDMA_THRESHOLD=0",
        "NCCL_GIN_ANVIL_SDMA_THRESHOLD_BROADCAST=0",
        "NCCL_DEBUG=INFO",
        "NCCL_DEBUG_SUBSYS=INIT,NET",
        "NCCL_DEBUG_FILE={}/nccl-debug.%h.%p.log".format(debug_dir),
        "RCCL_TESTS_BCAST_TIER_LOG=1",
    ]
    if force_flat_gin:
        gin_env += [
            "NCCL_GIN_ANVIL_BCAST_SCATTER_AG_MIN_BYTES=0",
            "NCCL_GIN_ANVIL_BCAST_RING_MIN_BYTES=0",
        ]
    elif force_sag_gin:
        gin_env += ["NCCL_GIN_ANVIL_BCAST_RING_MIN_BYTES=0"]
    gin_env += list(xenv)

    args = (
        mpi_launch_prefix(request, launcher, np, mpi_opts)
        + gin_env_xflags(gin_env)
        + gin_perf_argv(exe, size, dtype, ctas, extra=["-A", "1"])
    )
    cmd = " ".join(shlex.quote(a) for a in args)
    hang_msg = gin_hang_msg(
        "Broadcast", timeout_s, size, dtype, "{} MiB".format(msg_bytes // MiB)
    )
    try:
        rc, out = launch_mpi_shell(cmd, timeout_s, hang_msg)
        debug = read_nccl_debug_logs(debug_dir)
    finally:
        shutil.rmtree(debug_dir, ignore_errors=True)
    return rc, out, debug


def run_bcast_with_conn_gate_retry(launch_fn, conn_retries, settle_s=3):
    """Connectivity-gate retry for Broadcast launches that return debug text too."""
    state = {"debug": ""}

    def attempt():
        rc, out, debug = launch_fn()
        state["debug"] = debug
        return rc, out

    rc, out = run_with_conn_gate_retry(
        attempt, conn_retries, settle_s=settle_s, is_data_failure=bcast_data_failed
    )
    return rc, out, state["debug"]
