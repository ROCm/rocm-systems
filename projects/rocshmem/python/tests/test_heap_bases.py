"""Heap-base introspection tests.

Validates rocshmem4py.rocshmem_get_heap_bases (raw pointer list) and
rocshmem4py.interop.torch.get_heap_bases (int64 CUDA tensor).
"""

import pytest

import rocshmem4py
from rocshmem4py import (
    rocshmem_my_pe,
    rocshmem_n_pes,
    rocshmem_get_heap_bases,
)
from conftest import requires_torch


def test_get_heap_bases_returns_n_pes_entries():
    """Bare-Python helper returns one entry per PE."""
    nbytes = 1024
    buf = rocshmem4py.SymmetricBuffer(nbytes)
    try:
        bases = rocshmem_get_heap_bases(buf.ptr)
        assert isinstance(bases, list)
        assert len(bases) == rocshmem_n_pes()
    finally:
        buf.free()


def test_local_pe_entry_is_self_pointer():
    """bases[my_pe] == the local allocation's pointer."""
    nbytes = 1024
    buf = rocshmem4py.SymmetricBuffer(nbytes)
    try:
        bases = rocshmem_get_heap_bases(buf.ptr)
        me = rocshmem_my_pe()
        assert bases[me] == buf.ptr
    finally:
        buf.free()


@requires_torch
def test_get_heap_bases_torch_helper_shape_and_dtype():
    """Torch helper returns (n_pes,) int64 CUDA tensor."""
    import torch
    from rocshmem4py.interop.torch import create_tensor, get_heap_bases

    n = rocshmem_n_pes()
    t = create_tensor((1024,), torch.uint8)
    try:
        bases = get_heap_bases(t)
        assert bases.shape == (n,)
        assert bases.dtype == torch.int64
        assert bases.is_cuda
        # Local entry matches the local data pointer.
        assert int(bases[rocshmem_my_pe()].item()) == t.data_ptr()
    finally:
        from rocshmem4py.interop.torch import free_tensor
        free_tensor(t)


@requires_torch
def test_get_heap_bases_rejects_non_symmetric_tensor():
    """Helper raises on a non-symmetric tensor."""
    import torch
    from rocshmem4py.interop.torch import get_heap_bases

    t = torch.zeros(10, device="cuda")
    with pytest.raises(ValueError, match="not a symmetric tensor"):
        get_heap_bases(t)
