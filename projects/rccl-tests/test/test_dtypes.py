# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import pytest
from collectives import DATATYPES
from test_runner import run_rccl_perf, run_rccl_mpi


_MODES = [
    pytest.param("standalone", id="standalone"),
    pytest.param("mpi",        id="MPI", marks=pytest.mark.mpi),
]


def _run(mode, executable, gpu_count, args):
    """Dispatch to standalone or MPI runner with correct thread/rank layout."""
    if mode == "mpi":
        return run_rccl_mpi(executable, gpu_count, ["-t", "1", "-g", "1"] + args)
    return run_rccl_perf(executable, ["-t", str(gpu_count), "-g", "1"] + args)


# ---------------------------------------------------------------------------
# Group 16: Parametrized individual datatypes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("dtype", DATATYPES, ids=lambda d: d)
def test_dtype(dtype, mode, gpu_count):
    """Each datatype individually — granular CI reporting.

    Asserts the dtype name appears in the output data rows, confirming the
    binary ran the requested dtype rather than just exiting cleanly.
    """
    result = _run(mode, "all_reduce_perf", gpu_count,
                  ["-b", "1M", "-e", "1M", "-o", "sum", "-d", dtype])
    assert dtype in result.stdout, f"dtype '{dtype}' not found in output"
