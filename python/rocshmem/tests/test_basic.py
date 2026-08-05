"""Single-PE tests for rocshmem4py."""

import pytest

from conftest import requires_torch  # noqa: E402

# `torch` is not a hard dependency of rocshmem4py.  Tests that need it are
# marked individually with ``@requires_torch``; torch-free tests always run.
try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

import rocshmem4py  # noqa: E402


def test_import_and_constants():
    assert rocshmem4py.__version__ is not None
    assert rocshmem4py.ROCSHMEM_SUCCESS == 0
    assert rocshmem4py.ROCSHMEM_TEAM_WORLD == 0
    assert rocshmem4py.ROCSHMEM_TEAM_INVALID == -1
    assert rocshmem4py.ROCSHMEM_TEAM_WORLD != rocshmem4py.ROCSHMEM_TEAM_INVALID

    assert rocshmem4py.ROCSHMEM_SIGNAL_SET == 0
    assert rocshmem4py.ROCSHMEM_SIGNAL_ADD == 1

    for name in ("CMP_EQ", "CMP_NE", "CMP_GT", "CMP_GE", "CMP_LT", "CMP_LE"):
        assert isinstance(getattr(rocshmem4py, f"ROCSHMEM_{name}"), int)


def test_pe_info():
    my_pe = rocshmem4py.rocshmem_my_pe()
    n_pes = rocshmem4py.rocshmem_n_pes()
    assert isinstance(my_pe, int) and my_pe >= 0
    assert isinstance(n_pes, int) and n_pes >= 1
    assert my_pe < n_pes


def test_malloc_free():
    ptr = rocshmem4py.rocshmem_malloc(1024)
    assert ptr > 0
    rocshmem4py.rocshmem_free(ptr)


def test_barrier_all():
    # barrier_all is implemented across every rocSHMEM backend and is the
    # minimum sync primitive the bindings need to pass-through correctly.
    rocshmem4py.rocshmem_barrier_all()


@requires_torch
def test_symmetric_buffer():
    """SymmetricBuffer with torch integration: __cuda_array_interface__, torch.as_tensor,
    non-owning view, _device attribute.

    The torch-free RAII lifecycle (allocation, pointer validity, remote ptr, free)
    is covered by ``test_smoke.py::test_symmetric_buffer_lifecycle``.
    """
    size = 1024
    buf = rocshmem4py.SymmetricBuffer(size)

    assert buf.size == size
    assert buf.nbytes == size
    assert buf.ptr > 0
    assert buf.own_data is True
    assert int(buf) == buf.ptr
    assert isinstance(buf._device, int) and buf._device >= 0

    iface = buf.__cuda_array_interface__
    assert iface == {
        "data": (buf.ptr, False),
        "shape": (size,),
        "typestr": "<i1",
        "strides": None,
        "version": 3,
    }

    t = torch.as_tensor(buf, device="cuda")
    assert t.is_cuda and t.numel() == size

    view = rocshmem4py.SymmetricBuffer(size, ptr=buf.ptr, own_data=False)
    assert view.ptr == buf.ptr and view.own_data is False
    view.free()
    assert view._freed is True and buf._freed is False

    my_pe = rocshmem4py.rocshmem_my_pe()
    assert rocshmem4py.rocshmem_ptr(buf.ptr, my_pe) >= 0
    assert buf.get_remote_ptr(my_pe) >= 0

    buf.free()
    assert buf._freed is True


@requires_torch
def test_barrier_all_on_stream():
    stream = torch.cuda.current_stream()
    rocshmem4py.rocshmem_barrier_all_on_stream(stream.cuda_stream)
    torch.cuda.synchronize()


@requires_torch
def test_create_tensor():
    from rocshmem4py.interop import torch as rshmem_torch

    for dtype in (torch.float32, torch.bfloat16, torch.int32):
        t = rshmem_torch.create_tensor((16,), dtype)
        assert t.shape == (16,) and t.dtype == dtype and t.is_cuda
        assert getattr(t, "__symm_tensor__", False) is True

    t = rshmem_torch.create_tensor((8,), torch.float32)
    view = rshmem_torch.get_peer_tensor(t, rocshmem4py.rocshmem_my_pe())
    assert view.data_ptr() == t.data_ptr()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
