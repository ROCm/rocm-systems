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

# Hierarchical AllToAll (alltoall_perf -D 5): an Anvil SDMA DevComm for local
# peers and a network DevComm (PROXY by default) for the rest, on one
# communicator. NCCL_GIN_TYPE stays unset so both backends can be active.
#
# On one node every peer is local, so the strict split only moves data over
# SDMA. RCCL_TESTS_A2A_HIER_GROUP_SIZE splits the node into groups and sends
# between groups over the network DevComm, so both backends carry data.
#
# Opt-in: set RCCL_TESTS_GIN_HIER_A2A=1 on a node with a GIN-capable RCCL built
# with ENABLE_ROCSHMEM_GIN. Uses the same RCCL_TESTS_A2A_* / RCCL_TESTS_MPI_*
# variables as test_AllToAll.py. A run that prints "# SKIP" (a backend is not
# active) fails, because opting in asserts both backends are available; set
# RCCL_TESTS_A2A_HIER_ALLOW_SKIP=1 to report it as a pytest skip instead.

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
executable = os.environ.get(
    "RCCL_TESTS_A2A_EXE", os.path.join(path, "..", "build", "alltoall_perf")
)

_FALSE_VALUES = ("", "0", "false", "False")
_enabled = os.environ.get("RCCL_TESTS_GIN_HIER_A2A", "") not in _FALSE_VALUES
ALLOW_SKIP = os.environ.get("RCCL_TESTS_A2A_HIER_ALLOW_SKIP", "") not in _FALSE_VALUES

NP = env_int("RCCL_TESTS_A2A_NP", 0) or (detect_ngpus() if _enabled else 0)
LAUNCHER = os.environ.get("RCCL_TESTS_MPI_LAUNCHER", "mpirun")
CTAS = os.environ.get("RCCL_TESTS_A2A_CTAS", "16")
TIMEOUT_S = env_int("RCCL_TESTS_A2A_TIMEOUT_S", 900)
CONN_RETRIES = env_int("RCCL_TESTS_A2A_CONN_RETRIES", 5)
MPI_OPTS = shlex.split(os.environ.get("RCCL_TESTS_MPI_OPTS", ""))
XENV = shlex.split(os.environ.get("RCCL_TESTS_A2A_XENV", ""))

pytestmark = pytest.mark.skipif(
    not _enabled,
    reason="Hierarchical AllToAll tests are opt-in; set RCCL_TESTS_GIN_HIER_A2A=1 "
    "on a node where Anvil SDMA and the GIN proxy are both available.",
)


def _run_hier(request, per_peer_bytes, dtype, group_size):
    if NP < 2:
        pytest.skip("need >= 2 ranks/GPUs for AllToAll")
    size = str(int(per_peer_bytes * NP))
    env = ["NCCL_GIN_ENABLE=1"]
    if group_size:
        env.append("RCCL_TESTS_A2A_HIER_GROUP_SIZE={}".format(group_size))
    args = (
        mpi_launch_prefix(request, LAUNCHER, NP, MPI_OPTS)
        + gin_env_xflags(env + XENV)
        + gin_perf_argv(executable, size, dtype, CTAS, impl=5)
    )
    cmd = " ".join(shlex.quote(a) for a in args)
    hang_msg = gin_hang_msg(
        "Hierarchical AllToAll",
        TIMEOUT_S,
        size,
        dtype,
        "{} KiB/peer, group size {}".format(per_peer_bytes // 1024, group_size),
    )
    rc, out = run_with_conn_gate_retry(
        lambda: launch_mpi_shell(cmd, TIMEOUT_S, hang_msg), CONN_RETRIES
    )
    if rc != 0:
        return rc
    skip = next((l for l in (out or "").splitlines() if l.startswith("# SKIP")), None)
    if skip:
        if not ALLOW_SKIP:
            pytest.fail(
                "{} (set RCCL_TESTS_A2A_HIER_ALLOW_SKIP=1 to report this as a skip)".format(skip)
            )
        pytest.skip(skip)
    return rc


@pytest.mark.parametrize("per_peer_bytes", [64 * 1024, 16 * MiB])
@pytest.mark.parametrize("dtype", ["int32", "uint8"])
def test_AllToAllHierLocalSdma(request, per_peer_bytes, dtype):
    rc = _run_hier(request, per_peer_bytes, dtype, group_size=0)
    assert rc == 0, "hierarchical AllToAll failed (exit {})".format(rc)


@pytest.mark.parametrize("group_size", [1, 2, 3, 4])
@pytest.mark.parametrize("per_peer_bytes", [64 * 1024, 16 * MiB])
def test_AllToAllHierSplitGroups(request, group_size, per_peer_bytes):
    if group_size >= NP:
        pytest.skip(
            "group size {} leaves no network peers at {} ranks".format(group_size, NP)
        )
    rc = _run_hier(request, per_peer_bytes, "int32", group_size)
    assert rc == 0, "hierarchical AllToAll failed (exit {}) with group size {}".format(
        rc, group_size
    )
