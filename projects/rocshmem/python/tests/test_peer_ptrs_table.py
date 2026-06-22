"""Direct peer pointer table tests.

Validates rocshmem4py.rocshmem_get_peer_ptr_table (raw pointer list),
rocshmem4py.interop.torch.get_peer_ptr_table (int64 CUDA tensor), and the
get_heap_bases aliases.
"""

import pytest

import rocshmem4py
from rocshmem4py import (
    rocshmem_my_pe,
    rocshmem_n_pes,
    rocshmem_get_heap_bases,
    rocshmem_get_peer_ptr_table,
)
from conftest import requires_torch


def test_get_peer_ptr_table_returns_n_pes_entries():
    """Bare-Python helper returns one entry per PE."""
    nbytes = 1024
    buf = rocshmem4py.SymmetricBuffer(nbytes)
    try:
        peer_ptr_table = rocshmem_get_peer_ptr_table(buf.ptr)
        assert isinstance(peer_ptr_table, list)
        assert len(peer_ptr_table) == rocshmem_n_pes()
    finally:
        buf.free()


def test_local_pe_entry_is_self_pointer():
    """peer_ptrs[my_pe] == the local allocation's pointer."""
    nbytes = 1024
    buf = rocshmem4py.SymmetricBuffer(nbytes)
    try:
        peer_ptr_table = rocshmem_get_peer_ptr_table(buf.ptr)
        me = rocshmem_my_pe()
        assert peer_ptr_table[me] == buf.ptr
    finally:
        buf.free()


@requires_torch
def test_get_peer_ptr_table_torch_helper_shape_and_dtype():
    """Torch helper returns (n_pes,) int64 CUDA tensor."""
    import torch
    from rocshmem4py.interop.torch import create_tensor, get_peer_ptr_table

    n = rocshmem_n_pes()
    t = create_tensor((1024,), torch.uint8)
    try:
        peer_ptr_table = get_peer_ptr_table(t)
        assert peer_ptr_table.shape == (n,)
        assert peer_ptr_table.dtype == torch.int64
        assert peer_ptr_table.is_cuda
        # Local entry matches the local data pointer.
        assert int(peer_ptr_table[rocshmem_my_pe()].item()) == t.data_ptr()
    finally:
        from rocshmem4py.interop.torch import free_tensor
        free_tensor(t)


@requires_torch
def test_get_peer_ptr_table_rejects_non_symmetric_tensor():
    """Helper raises on a non-symmetric tensor."""
    import torch
    from rocshmem4py.interop.torch import get_peer_ptr_table

    t = torch.zeros(10, device="cuda")
    with pytest.raises(ValueError, match="not a symmetric tensor"):
        get_peer_ptr_table(t)


def test_legacy_aliases_match_peer_ptr_table():
    """Legacy names remain compatibility aliases."""
    nbytes = 1024
    buf = rocshmem4py.SymmetricBuffer(nbytes)
    try:
        peer_ptr_table = rocshmem_get_peer_ptr_table(buf.ptr)
        assert rocshmem_get_heap_bases(buf.ptr) == peer_ptr_table
    finally:
        buf.free()
