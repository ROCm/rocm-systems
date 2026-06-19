# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared pytest fixtures for the RCCL fork's mpi-marked test suite.

All fixtures here exist for ``tests/test_collectives.py`` and
``tests/test_interop.py``. The unit-style files (``test_loader_stubs.py``,
``test_shim_surface.py``, ``test_rocm_extensions.py``) do not import any
of these fixtures and remain runnable in a single-rank CPU environment.
"""

from __future__ import annotations

import os

import pytest

# ---------------------------------------------------------------------------
# MPI / hardware availability checks (module-level, evaluated at collect time
# so the unit-style files never get blocked on a missing MPI runtime).
# ---------------------------------------------------------------------------


def _maybe_mpi():
    """Return MPI module if available, else None (no exception)."""
    try:
        from mpi4py import MPI  # type: ignore
    except Exception:  # pragma: no cover - mpi4py optional on CPU hosts
        return None
    return MPI


def _maybe_torch():
    try:
        import torch  # type: ignore
    except Exception:  # pragma: no cover
        return None
    return torch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def mpi_world():
    """``MPI.COMM_WORLD`` or skip if mpi4py / a live MPI runtime is missing."""
    MPI = _maybe_mpi()
    if MPI is None:
        pytest.skip("mpi4py not installed; rerun under `mpirun -np N pytest -m mpi`")
    return MPI.COMM_WORLD


@pytest.fixture(scope="session")
def rank_info(mpi_world):
    """Tuple of ``(rank, nranks)`` for the current MPI run."""
    return mpi_world.Get_rank(), mpi_world.Get_size()


@pytest.fixture(scope="session")
def device(rank_info):
    """torch.device assigned to this rank as ``cuda:rank % device_count``.

    Skips the entire session if either torch or any visible GPU is missing.
    """
    torch = _maybe_torch()
    if torch is None:
        pytest.skip("torch not installed; required for GPU buffer ownership")
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        pytest.skip("no GPU visible to torch.cuda")
    rank, _ = rank_info
    dev_id = rank % torch.cuda.device_count()
    dev = torch.device(f"cuda:{dev_id}")
    torch.cuda.set_device(dev)
    return dev


@pytest.fixture(scope="session")
def comm(mpi_world, rank_info, device):
    """Session-scoped NCCL communicator, torn down collectively at exit.

    Yields a ``nccl.core.Communicator``; the caller can read
    ``rank``/``nranks`` from the ``rank_info`` fixture or via the
    communicator's own attributes.
    """
    import nccl.core as nccl

    rank, nranks = rank_info
    root = 0
    unique_id = nccl.get_unique_id() if rank == root else None
    unique_id = mpi_world.bcast(unique_id, root=root)

    nccl_comm = nccl.Communicator.init(nranks=nranks, rank=rank, unique_id=unique_id)
    try:
        yield nccl_comm
    finally:
        # destroy() is collective; all ranks must call it.
        nccl_comm.destroy()


# ---------------------------------------------------------------------------
# Helpers exposed as plain functions (imported by individual test modules).
# ---------------------------------------------------------------------------


def require_at_least_n_ranks(rank_info, n: int) -> None:
    """``pytest.skip`` when fewer than ``n`` ranks were spawned."""
    _, nranks = rank_info
    if nranks < n:
        pytest.skip(f"requires >= {n} MPI ranks (got {nranks})")


def xfail_if_fewer_than_8_gpus():
    """xfail marker for the 8-GPU run if the host does not have 8 GPUs visible.

    The 8-GPU run goes green if hardware is available, otherwise it is
    marked xfail with the reason recorded in the test file.

    The probe is intentionally cheap — it only inspects ``torch.cuda``
    and an optional ``RCCL4PY_FORCE_8GPU_XFAIL=1`` knob (used by CI to
    force the xfail path on hosts that do report 8 devices but cannot
    link them).
    """
    torch = _maybe_torch()
    visible = torch.cuda.device_count() if torch and torch.cuda.is_available() else 0
    forced = os.environ.get("RCCL4PY_FORCE_8GPU_XFAIL") == "1"
    if forced or visible < 8:
        return pytest.mark.xfail(
            reason=(
                f"8-GPU run not supported on this host: visible GPUs={visible}, "
                f"RCCL4PY_FORCE_8GPU_XFAIL={int(forced)}"
            ),
            strict=False,
        )
    return pytest.mark.usefixtures()  # no-op marker so callers can apply unconditionally
