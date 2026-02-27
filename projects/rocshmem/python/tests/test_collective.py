"""Multi-PE tests for rocshmem4py (requires >= 2 PEs)."""

import os
import pytest
import torch
import rocshmem4py

pytestmark = pytest.mark.skipif(
    "WORLD_SIZE" not in os.environ or int(os.environ.get("WORLD_SIZE", 1)) < 2,
    reason="Requires at least 2 PEs",
)


def _pe_info():
    my_pe = rocshmem4py.rocshmem_my_pe()
    n_pes = rocshmem4py.rocshmem_n_pes()
    peer = (my_pe + 1) % n_pes
    return my_pe, n_pes, peer


def test_atomic_fetch_add():
    my_pe, n_pes, _ = _pe_info()

    for width, fn in [(4, rocshmem4py.rocshmem_int_atomic_fetch_add),
                      (8, rocshmem4py.rocshmem_long_atomic_fetch_add)]:
        buf = rocshmem4py.SymmetricBuffer(width)
        rocshmem4py.rocshmem_barrier_all()
        if my_pe != 0:
            old = fn(buf.ptr, 1, 0)
            assert isinstance(old, int)
        rocshmem4py.rocshmem_barrier_all()
        buf.free()



def test_getmem():
    my_pe, n_pes, peer = _pe_info()
    nelems = 64

    src = rocshmem4py.rocshmem_create_tensor((nelems,), torch.int32)
    dst = rocshmem4py.rocshmem_create_tensor((nelems,), torch.int32)
    src.fill_(my_pe)
    dst.fill_(-1)
    torch.cuda.synchronize()

    rocshmem4py.rocshmem_barrier_all()
    rocshmem4py.rocshmem_getmem(
        dst.data_ptr(), src.data_ptr(), nelems * src.element_size(), peer)
    rocshmem4py.rocshmem_quiet()
    rocshmem4py.rocshmem_barrier_all()
    torch.cuda.synchronize()

    torch.testing.assert_close(
        dst, torch.full((nelems,), peer, dtype=torch.int32, device="cuda"))


def test_tensor_list_memcpy():
    my_pe, n_pes, peer = _pe_info()
    nelems_per_rank = 32

    buf = rocshmem4py.rocshmem_create_tensor((n_pes * nelems_per_rank,), torch.int32)
    buf.fill_(0)
    torch.cuda.synchronize()

    ref = torch.arange(n_pes * nelems_per_rank, dtype=torch.int32, device="cuda")
    start = nelems_per_rank * my_pe
    buf[start:start + nelems_per_rank].copy_(ref[start:start + nelems_per_rank])
    torch.cuda.synchronize()

    rocshmem4py.rocshmem_barrier_all()

    nbytes = nelems_per_rank * buf.element_size()
    offset = buf.data_ptr() + start * buf.element_size()
    rocshmem4py.rocshmem_putmem(offset, offset, nbytes, peer)
    rocshmem4py.rocshmem_quiet()
    rocshmem4py.rocshmem_barrier_all()
    torch.cuda.synchronize()

    sender = (my_pe - 1 + n_pes) % n_pes
    s = nelems_per_rank * sender
    torch.testing.assert_close(buf[s:s + nelems_per_rank], ref[s:s + nelems_per_rank])


def test_putmem_on_stream():
    my_pe, n_pes, peer = _pe_info()
    nelems = 128

    src = rocshmem4py.rocshmem_create_tensor((nelems,), torch.float32)
    dst = rocshmem4py.rocshmem_create_tensor((nelems,), torch.float32)
    src.fill_(float(my_pe))
    dst.fill_(-1.0)
    torch.cuda.synchronize()

    rocshmem4py.rocshmem_barrier_all()

    stream = torch.cuda.current_stream()
    rocshmem4py.rocshmem_putmem_on_stream(
        dst.data_ptr(), src.data_ptr(), nelems * src.element_size(),
        peer, stream.cuda_stream)
    rocshmem4py.rocshmem_barrier_all_on_stream(stream.cuda_stream)
    torch.cuda.synchronize()

    sender = (my_pe - 1 + n_pes) % n_pes
    torch.testing.assert_close(
        dst, torch.full((nelems,), float(sender), dtype=torch.float32,
                        device="cuda"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
