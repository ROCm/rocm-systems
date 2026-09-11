"""Multi-PE tests for rocshmem4py (requires >= 2 PEs).

These tests exercise the Python binding layer using the portable rocSHMEM
surface -- ``rocshmem_barrier_all``, ``*_on_stream`` data movement, and
``rocshmem_ptr`` -- so a single test pass is valid across every rocSHMEM
backend (RO / IPC / GDA).

Host-side blocking APIs (``rocshmem_fence`` / ``rocshmem_quiet`` /
``rocshmem_putmem`` / ``rocshmem_getmem`` / ``rocshmem_*_atomic_*``) are
not exercised here. Their coverage depends on the backend rocSHMEM was
built with and belongs in rocSHMEM's own integration tests, not in the
Python-binding test suite.
"""
import pytest

from conftest import requires_multi_pe, requires_torch  # noqa: E402

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

import rocshmem4py  # noqa: E402
from rocshmem4py.interop import torch as rshmem_torch  # noqa: E402

pytestmark = [requires_torch, requires_multi_pe]


def _pe_info():
    my_pe = rocshmem4py.rocshmem_my_pe()
    n_pes = rocshmem4py.rocshmem_n_pes()
    peer = (my_pe + 1) % n_pes
    return my_pe, n_pes, peer


def test_getmem_on_stream():
    my_pe, n_pes, peer = _pe_info()
    nelems = 64

    src = rshmem_torch.create_tensor((nelems,), torch.int32)
    dst = rshmem_torch.create_tensor((nelems,), torch.int32)
    src.fill_(my_pe)
    dst.fill_(-1)
    torch.cuda.synchronize()

    stream = torch.cuda.current_stream()
    rocshmem4py.rocshmem_barrier_all_on_stream(stream.cuda_stream)
    rocshmem4py.rocshmem_getmem_on_stream(
        dst.data_ptr(), src.data_ptr(), nelems * src.element_size(),
        peer, stream.cuda_stream)
    rocshmem4py.rocshmem_barrier_all_on_stream(stream.cuda_stream)
    torch.cuda.synchronize()

    torch.testing.assert_close(
        dst, torch.full((nelems,), peer, dtype=torch.int32, device="cuda"))


def test_putmem_on_stream():
    my_pe, n_pes, peer = _pe_info()
    nelems = 128

    src = rshmem_torch.create_tensor((nelems,), torch.float32)
    dst = rshmem_torch.create_tensor((nelems,), torch.float32)
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


def test_tensor_list_intra_node():
    my_pe, n_pes, peer = _pe_info()
    nelems_per_rank = 32

    buf = rshmem_torch.create_tensor((n_pes * nelems_per_rank,), torch.int32)
    buf.fill_(0)
    torch.cuda.synchronize()

    ref = torch.arange(n_pes * nelems_per_rank, dtype=torch.int32, device="cuda")
    start = nelems_per_rank * my_pe
    buf[start:start + nelems_per_rank].copy_(ref[start:start + nelems_per_rank])
    torch.cuda.synchronize()

    stream = torch.cuda.current_stream()
    rocshmem4py.rocshmem_barrier_all_on_stream(stream.cuda_stream)

    # Use the stream-based transfer to propagate my slice to the peer;
    # portable across all rocSHMEM backends.
    nbytes = nelems_per_rank * buf.element_size()
    offset = buf.data_ptr() + start * buf.element_size()
    rocshmem4py.rocshmem_putmem_on_stream(
        offset, offset, nbytes, peer, stream.cuda_stream)
    rocshmem4py.rocshmem_barrier_all_on_stream(stream.cuda_stream)
    torch.cuda.synchronize()

    sender = (my_pe - 1 + n_pes) % n_pes
    s = nelems_per_rank * sender
    torch.testing.assert_close(buf[s:s + nelems_per_rank], ref[s:s + nelems_per_rank])


def test_symm_rocshmem_tensor_peer_view():
    """``get_peer_tensor`` returns a zero-copy view of peer's symmetric tensor
    on IPC backends.  On backends without direct remote memory access,
    ``rocshmem_ptr`` returns NULL and the function raises -- skipped here;
    the explicit copy path is covered by ``test_getmem_on_stream`` above.
    """
    my_pe, n_pes, peer = _pe_info()
    t = rshmem_torch.create_tensor((16,), torch.float32)
    t.fill_(float(my_pe))
    torch.cuda.synchronize()
    rocshmem4py.rocshmem_barrier_all()

    try:
        peer_view = rshmem_torch.get_peer_tensor(t, peer)
    except RuntimeError:
        pytest.skip("rocshmem_ptr returned NULL -- backend lacks direct remote access")

    assert peer_view.is_cuda
    assert peer_view.shape == t.shape
    assert peer_view.dtype == t.dtype
    torch.cuda.synchronize()
    expected = torch.full((16,), float(peer), dtype=torch.float32, device="cuda")
    torch.testing.assert_close(peer_view, expected)


def test_interop_torch_put_get():
    """Tensor-aware ``interop.torch.put`` / ``get`` / ``barrier_all`` wrappers.

    These wrappers accept torch tensors directly and default the stream to
    ``torch.cuda.current_stream()``.  Portable across every rocSHMEM backend
    since they delegate to the ``*_on_stream`` primitives.
    """
    my_pe, n_pes, peer = _pe_info()
    nelems = 64

    src = rshmem_torch.create_tensor((nelems,), torch.int32)
    dst = rshmem_torch.create_tensor((nelems,), torch.int32)
    src.fill_(my_pe)
    dst.fill_(-1)
    torch.cuda.synchronize()
    rshmem_torch.barrier_all()  # default stream

    # get: pull peer's src into local dst
    rshmem_torch.get(dst, src, peer)  # default stream
    rshmem_torch.barrier_all()
    torch.cuda.synchronize()

    torch.testing.assert_close(
        dst, torch.full((nelems,), peer, dtype=torch.int32, device="cuda")
    )

    # put: push local src (still filled with my_pe) to peer's dst
    dst.fill_(-1)
    torch.cuda.synchronize()
    rshmem_torch.barrier_all()

    stream = torch.cuda.current_stream()
    rshmem_torch.put(dst, src, peer, stream=stream)
    rshmem_torch.barrier_all(stream=stream)
    torch.cuda.synchronize()

    sender = (my_pe - 1 + n_pes) % n_pes
    torch.testing.assert_close(
        dst, torch.full((nelems,), sender, dtype=torch.int32, device="cuda")
    )


def test_interop_torch_put_get_nbytes_mismatch():
    """``put`` / ``get`` wrappers must reject nbytes mismatches up-front."""
    my_pe, n_pes, peer = _pe_info()
    a = rshmem_torch.create_tensor((8,), torch.int32)
    b = rshmem_torch.create_tensor((16,), torch.int32)  # different nbytes

    with pytest.raises(ValueError, match="nbytes"):
        rshmem_torch.put(a, b, peer)
    with pytest.raises(ValueError, match="nbytes"):
        rshmem_torch.get(a, b, peer)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
