# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import re
import pytest
from collectives import COLLECTIVES, COLLECTIVES_WITH_OPS, COLLECTIVES_WITH_ROOT, OPS, DATATYPES
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
# Group 14: -o all / -d all / -r all sweep modes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("collective", COLLECTIVES_WITH_OPS, ids=lambda c: c.name)
def test_ops_all(collective, mode, gpu_count):
    """All reduction ops in a single sweep (-o all).

    Asserts every op name appears in the output data rows, confirming the
    binary ran the full op sweep rather than just exiting cleanly.
    """
    result = _run(mode, collective.executable, gpu_count,
                  ["-o", "all", "-d", "float"])
    for op in OPS:
        assert op in result.stdout, f"op '{op}' not found in output"


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("collective", COLLECTIVES, ids=lambda c: c.name)
def test_dtypes_all(collective, mode, gpu_count):
    """All datatypes in a single sweep (-d all).

    Asserts every dtype name appears in the output data rows, confirming the
    binary ran the full dtype sweep rather than just exiting cleanly.
    """
    args = ["-d", "all"]
    if collective.has_ops:
        args += ["-o", "sum"]
    if collective.has_root:
        args += ["-r", "0"]
    result = _run(mode, collective.executable, gpu_count, args)
    for dtype in DATATYPES:
        assert dtype in result.stdout, f"dtype '{dtype}' not found in output"


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("collective", COLLECTIVES_WITH_ROOT, ids=lambda c: c.name)
def test_root_all(collective, mode, gpu_count):
    """Rotate through all root ranks (-r all).

    Parses the root column from output data rows and asserts every rank
    0..gpu_count-1 appears, confirming full root rotation occurred.
    """
    args = ["-r", "all", "-d", "float"]
    if collective.has_ops:
        args += ["-o", "sum"]
    result = _run(mode, collective.executable, gpu_count, args)

    # Data rows: leading whitespace, size, count, type, redop, root, ...
    # Extract the root column (5th whitespace-separated field) from each data row.
    roots_seen = set()
    for line in result.stdout.splitlines():
        m = re.match(r'^\s+\d+\s+\d+\s+\S+\s+\S+\s+(\d+)\s+', line)
        if m:
            roots_seen.add(int(m.group(1)))

    expected = set(range(gpu_count))
    assert expected == roots_seen, (
        f"expected roots {sorted(expected)}, got {sorted(roots_seen)}"
    )
