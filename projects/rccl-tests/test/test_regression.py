# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import pytest
from conftest import NGPUS
from collectives import COLLECTIVES, DATATYPES
from test_runner import run_rccl_perf, run_rccl_mpi


BYTE_RANGES = [("4", "1K"), ("1K", "1M"), ("1M", "4G")]


def _regression_gpu_counts(ngpus: int) -> list:
    """Build a regression-appropriate GPU count grid from detected count.

    Uses powers-of-2 starting at 2 up to ngpus, always including ngpus itself
    if it isn't already in the list. This keeps regression multi-GPU focused
    on machines with >=2 GPUs while still ensuring the full machine is
    exercised. On a single-GPU host the grid falls back to ``[1]`` so the
    regression marker is never empty (the corresponding 1-GPU coverage in the
    functional suites is the same matrix at smaller scale).

    Examples:
      ngpus=8  -> [2, 4, 8]
      ngpus=6  -> [2, 4, 6]
      ngpus=4  -> [2, 4]
      ngpus=2  -> [2]
      ngpus=1  -> [1]
    """
    powers = [2**i for i in range(ngpus.bit_length()) if 2**i <= ngpus and 2**i > 1]
    if ngpus not in powers:
        powers.append(ngpus)
    return sorted(powers)


# Computed once at module import — always valid for the current node
REGRESSION_GPU_COUNTS = _regression_gpu_counts(NGPUS)


@pytest.mark.regression
@pytest.mark.parametrize("n_gpus", REGRESSION_GPU_COUNTS, ids=lambda n: f"gpus{n}")
@pytest.mark.parametrize("min_b,max_b", BYTE_RANGES,
                         ids=lambda r: f"{r[0]}-{r[1]}" if isinstance(r, tuple) else r)
@pytest.mark.parametrize("dtype", DATATYPES, ids=lambda d: d)
@pytest.mark.parametrize("collective", COLLECTIVES, ids=lambda c: c.name)
def test_regression_single(collective, dtype, min_b, max_b, n_gpus):
    args = ["-t", str(n_gpus), "-g", "1",
            "-b", min_b, "-e", max_b, "-f", "2", "-d", dtype]
    if collective.has_ops:
        args += ["-o", "sum"]
    if collective.has_root:
        args += ["-r", "0"]
    run_rccl_perf(collective.executable, args)


@pytest.mark.regression
@pytest.mark.mpi
@pytest.mark.parametrize("n_gpus", REGRESSION_GPU_COUNTS, ids=lambda n: f"gpus{n}")
@pytest.mark.parametrize("min_b,max_b", BYTE_RANGES,
                         ids=lambda r: f"{r[0]}-{r[1]}" if isinstance(r, tuple) else r)
@pytest.mark.parametrize("dtype", DATATYPES, ids=lambda d: d)
@pytest.mark.parametrize("collective", COLLECTIVES, ids=lambda c: c.name)
def test_regression_mpi(collective, dtype, min_b, max_b, n_gpus):
    args = ["-t", "1", "-g", "1",
            "-b", min_b, "-e", max_b, "-f", "2", "-d", dtype]
    if collective.has_ops:
        args += ["-o", "sum"]
    if collective.has_root:
        args += ["-r", "0"]
    run_rccl_mpi(collective.executable, n_gpus, args)
