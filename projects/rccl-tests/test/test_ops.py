# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import pytest
from collectives import COLLECTIVES_WITH_OPS, OPS
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
# Group 15: Parametrized individual reduction ops
# ---------------------------------------------------------------------------

# (collective, op, mode) tuples that currently hang on MI355X (RCCL 2.28.3).
# Listed individually so it's obvious which combinations are quarantined and
# can be re-enabled once the upstream hang is fixed.
_KNOWN_HANG_OP_COMBOS = {
    ("allreduce", "prod", "standalone"),  # all_reduce_perf -t 8 -g 1 -o prod -b 1M -e 1M hangs >300s
}


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("op", OPS, ids=lambda o: o)
@pytest.mark.parametrize("collective", COLLECTIVES_WITH_OPS, ids=lambda c: c.name)
def test_op(collective, op, mode, gpu_count):
    """Each reduction op × each op-capable collective — granular CI reporting.

    Asserts the op name appears in the output data rows, confirming the
    binary ran the requested op rather than just exiting cleanly.
    """
    if (collective.name, op, mode) in _KNOWN_HANG_OP_COMBOS:
        pytest.skip(
            f"known RCCL bug (untracked): {collective.executable} -o {op} "
            f"({mode}) hangs at 1 MiB on MI355X (RCCL 2.28.3); see "
            "test_op[allreduce-prod-standalone] timeout in stdout.log"
        )
    args = ["-b", "1M", "-e", "1M", "-o", op, "-d", "float"]
    if collective.has_root:
        args += ["-r", "0"]
    result = _run(mode, collective.executable, gpu_count, args)
    assert op in result.stdout, f"op '{op}' not found in output"
