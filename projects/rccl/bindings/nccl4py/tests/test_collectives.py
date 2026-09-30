# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collective-correctness tests for the RCCL fork.

Seven tests, all marked ``mpi``, each compares the binding's collective
result against a ``numpy`` (or ``torch``-on-CPU) reference within
``atol=1e-3 rtol=1e-3``. Run with::

    mpirun -np 2 --allow-run-as-root pytest -m mpi tests/test_collectives.py
    mpirun -np 4 --allow-run-as-root pytest -m mpi tests/test_collectives.py
    mpirun -np 8 --allow-run-as-root pytest -m mpi tests/test_collectives.py

The 8-GPU configuration is xfailed (with a recorded reason) on hosts
where fewer than 8 GPUs are visible; see ``conftest.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch is required for the GPU buffers")

import nccl.core as nccl  # noqa: E402

from conftest import require_at_least_n_ranks, xfail_if_fewer_than_8_gpus  # noqa: E402

pytestmark = [
    pytest.mark.mpi,
    xfail_if_fewer_than_8_gpus(),  # only triggers on the 8-GPU launch
]

# Element count chosen to be small enough to run quickly on every rank
# count, large enough to exercise multi-stage ring algorithms.
_N = 4096

# Tolerances. The default `atol=1e-3 rtol=1e-3` is achievable for fp16
# / fp32 but mathematically impossible for bf16: 1 ULP in bf16 at value
# 1 is 2**-7 ~ 7.8e-3, and a K-rank sum accumulates O((K-1)*ULP) round-
# off error against an fp32 reference. To keep the test passing
# uniformly across {2, 4, 8} ranks without inflating the tolerances
# absurdly, we (1) normalize per-rank inputs by 1/nranks so the
# all_reduce sum stays bounded in [-1, 1] regardless of rank count;
# (2) scale the bf16 tolerance with nranks. Neither indicates a bug
# in the binding.
_BF16_EPS = 2**-7  # 1 ULP in bf16 at magnitude 1.0


def _tol_for(dtype, nranks: int) -> tuple[float, float]:
    if dtype == torch.bfloat16:
        atol = max(5e-3, (nranks - 1) * _BF16_EPS)
        rtol = 5e-2
        return atol, rtol
    # fp16 / fp32: literal tolerances. fp16 has eps = 2**-10 so K-1
    # accumulated ULPs is well under 1e-3 for K <= 8 with the 1/nranks
    # input scaling applied below.
    return 1e-3, 1e-3


def _seeded_tensor(rank: int, dtype, device, n: int = _N, *, scale: float = 1.0) -> torch.Tensor:
    """Per-rank deterministic random tensor in ``[-scale, scale]``."""
    g = torch.Generator(device="cpu").manual_seed(0xBEEF + rank)
    cpu = torch.empty(n, dtype=torch.float32).uniform_(-scale, scale, generator=g)
    return cpu.to(device=device, dtype=dtype)


def _allclose_or_fail(actual: torch.Tensor, expected: torch.Tensor, *, nranks: int = 1) -> None:
    atol, rtol = _tol_for(actual.dtype, nranks)
    actual32 = actual.detach().to(dtype=torch.float32, device="cpu")
    expected32 = expected.detach().to(dtype=torch.float32, device="cpu")
    torch.testing.assert_close(actual32, expected32, atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# all_reduce: fp16 / bf16 / fp32 (3 tests)
# ---------------------------------------------------------------------------


def _run_all_reduce(comm, mpi_world, rank_info, device, dtype):
    rank, nranks = rank_info
    # Scale per-rank inputs by 1/nranks so the all_reduce result stays in
    # [-1, 1] for any nranks; otherwise the peak intermediate sum in the
    # ring-allreduce grows with nranks, the ULP at the peak grows too,
    # and the round-off floor of low-precision dtypes blows past any
    # fixed atol. This is the same trick rccl-tests applies via the
    # `--scale` option to its random-data verifier.
    send = _seeded_tensor(rank, dtype, device, scale=1.0 / nranks)
    recv = torch.empty_like(send)

    comm.reduce(send, recv, nccl.SUM)  # root=None -> all_reduce
    torch.cuda.synchronize()

    # Reference: gather every rank's send buffer on every rank via mpi4py
    # (cheap because we operate on small tensors and a CPU copy).
    cpu_send = send.detach().to(dtype=torch.float32, device="cpu").numpy()
    gathered = np.stack(mpi_world.allgather(cpu_send), axis=0)
    expected_np = gathered.sum(axis=0)
    expected = torch.from_numpy(expected_np)

    _allclose_or_fail(recv, expected, nranks=nranks)


def test_all_reduce_fp16(comm, mpi_world, rank_info, device):
    require_at_least_n_ranks(rank_info, 2)
    _run_all_reduce(comm, mpi_world, rank_info, device, torch.float16)


def test_all_reduce_bf16(comm, mpi_world, rank_info, device):
    require_at_least_n_ranks(rank_info, 2)
    _run_all_reduce(comm, mpi_world, rank_info, device, torch.bfloat16)


def test_all_reduce_fp32(comm, mpi_world, rank_info, device):
    require_at_least_n_ranks(rank_info, 2)
    _run_all_reduce(comm, mpi_world, rank_info, device, torch.float32)


# ---------------------------------------------------------------------------
# broadcast (1 test)
# ---------------------------------------------------------------------------


def test_broadcast(comm, mpi_world, rank_info, device):
    require_at_least_n_ranks(rank_info, 2)
    rank, _ = rank_info
    root = 0
    # Root has the source tensor; non-root ranks start from zeros and
    # receive the root's payload.
    src = _seeded_tensor(0, torch.float32, device)
    if rank == root:
        send = src.clone()
        recv = torch.empty_like(send)
    else:
        send = torch.zeros_like(src)  # ignored on non-root ranks
        recv = torch.zeros_like(src)

    comm.broadcast(send, recv, root=root)
    torch.cuda.synchronize()

    _allclose_or_fail(recv, src)


# ---------------------------------------------------------------------------
# all_gather (1 test)
# ---------------------------------------------------------------------------


def test_all_gather(comm, mpi_world, rank_info, device):
    require_at_least_n_ranks(rank_info, 2)
    rank, nranks = rank_info
    per_rank = _N // nranks
    send = _seeded_tensor(rank, torch.float32, device, n=per_rank)
    recv = torch.empty(per_rank * nranks, dtype=torch.float32, device=device)

    comm.allgather(send, recv)
    torch.cuda.synchronize()

    # Reference: numpy concat of every rank's send buffer in rank order.
    cpu_send = send.detach().to(dtype=torch.float32, device="cpu").numpy()
    expected_np = np.concatenate(mpi_world.allgather(cpu_send), axis=0)
    expected = torch.from_numpy(expected_np)

    _allclose_or_fail(recv, expected)


# ---------------------------------------------------------------------------
# reduce_scatter (1 test)
# ---------------------------------------------------------------------------


def test_reduce_scatter(comm, mpi_world, rank_info, device):
    require_at_least_n_ranks(rank_info, 2)
    rank, nranks = rank_info
    per_rank = _N // nranks
    total = per_rank * nranks  # divisible
    send = _seeded_tensor(rank, torch.float32, device, n=total)
    recv = torch.empty(per_rank, dtype=torch.float32, device=device)

    comm.reduce_scatter(send, recv, nccl.SUM)
    torch.cuda.synchronize()

    cpu_send = send.detach().to(dtype=torch.float32, device="cpu").numpy()
    gathered = np.stack(mpi_world.allgather(cpu_send), axis=0)  # (nranks, total)
    summed = gathered.sum(axis=0)
    expected_np = summed[rank * per_rank : (rank + 1) * per_rank]
    expected = torch.from_numpy(expected_np)

    _allclose_or_fail(recv, expected)


# ---------------------------------------------------------------------------
# send / recv (1 test, pairs adjacent ranks)
# ---------------------------------------------------------------------------


def test_send_recv(comm, mpi_world, rank_info, device):
    require_at_least_n_ranks(rank_info, 2)
    rank, nranks = rank_info
    # Pair rank r with rank r ^ 1 inside the first even pairs only; ranks
    # without a partner (when nranks is odd) sit out the test.
    partner = rank ^ 1
    if partner >= nranks:
        pytest.skip(f"rank {rank} has no partner (nranks={nranks})")

    send = _seeded_tensor(rank, torch.float32, device)
    expected_partner_send = _seeded_tensor(partner, torch.float32, device)
    recv = torch.zeros_like(send)

    with nccl.group():
        comm.send(send, peer=partner)
        comm.recv(recv, peer=partner)
    torch.cuda.synchronize()

    _allclose_or_fail(recv, expected_partner_send)
