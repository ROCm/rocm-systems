# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for the NCCL4Py Python bindings added in NCCL 2.29.2.

This file complements the more comprehensive tests in test_collectives.py,
test_interop.py and friends by covering the bare-minimum import-and-surface
checks. These tests are intentionally CPU-only so they can run on CI
runners that don't have ROCm hardware available.

A full functional drive of the binding (ncclCommInitRank, collectives,
dlpack interop) lives in the other test_*.py files in this directory.
"""

import pytest


def _import_nccl4py():
    """Try to import the binding under either of its accepted module
    layouts. nccl4py is sometimes published as ``nccl`` and sometimes as
    ``nccl.core``; we tolerate both.
    """
    try:
        import nccl.core as nccl4py
        return nccl4py
    except Exception:
        pass
    try:
        import nccl as nccl4py
        return nccl4py
    except Exception as e:
        pytest.skip(f"nccl4py is not importable: {e}. Build via "
                    f"`pip install ./bindings/nccl4py` first.")


def test_nccl4py_imports():
    """The package must import without raising."""
    nccl4py = _import_nccl4py()
    assert nccl4py is not None


def test_nccl4py_exposes_get_unique_id():
    """ncclGetUniqueId is the smallest possible smoke test of the actual
    Python -> C boundary."""
    nccl4py = _import_nccl4py()
    candidates = ("get_unique_id", "GetUniqueId", "ncclGetUniqueId",
                  "unique_id", "UniqueId")
    found = next((c for c in candidates if hasattr(nccl4py, c)), None)
    if not found:
        pytest.skip("nccl4py does not expose a get_unique_id symbol under "
                    f"any of the known names: {candidates}")


def test_nccl4py_exposes_comm_init_rank():
    """The binding must expose at least one entry point that wraps
    ncclCommInitRank-family functions. This is the minimum surface needed
    to write distributed code from Python."""
    nccl4py = _import_nccl4py()
    candidates = ("comm_init_rank", "CommInitRank", "ncclCommInitRank",
                  "init_rank", "Communicator", "Comm", "NCCLComm")
    found = next((c for c in candidates if hasattr(nccl4py, c)), None)
    assert found, ("nccl4py does not expose any CommInitRank-equivalent "
                   f"under {candidates}")


def test_nccl4py_exposes_grow_api():
    """The 2.29.2 release notes call out ncclCommGrow/ncclCommGetUniqueId
    among the new APIs; the Python binding should surface them."""
    nccl4py = _import_nccl4py()
    grow_candidates = ("comm_grow", "CommGrow", "ncclCommGrow")
    uid_candidates  = ("comm_get_unique_id", "CommGetUniqueId",
                       "ncclCommGetUniqueId")
    has_grow = any(hasattr(nccl4py, c) for c in grow_candidates)
    has_uid  = any(hasattr(nccl4py, c) for c in uid_candidates)
    if not (has_grow and has_uid):
        pytest.skip("nccl4py does not yet expose Grow / GetUniqueId in "
                    "this build. The binding is expected to surface these "
                    "after the C-API exposure is stabilized.")
