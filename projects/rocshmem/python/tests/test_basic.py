"""Single-PE tests for rocshmem4py."""

import pytest
import torch
import rocshmem4py


def test_import_and_constants():
    assert rocshmem4py.__version__ is not None
    assert rocshmem4py.ROCSHMEM_SUCCESS == 0
    assert rocshmem4py.ROCSHMEM_TEAM_WORLD == 0
    assert rocshmem4py.ROCSHMEM_TEAM_INVALID == -1

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


def test_symmetric_buffer():
    size = 1024
    buf = rocshmem4py.SymmetricBuffer(size)

    assert buf.size == size
    assert buf.nbytes == size
    assert buf.ptr > 0
    assert buf.own_data is True
    assert int(buf) == buf.ptr
    assert isinstance(buf._device, int) and buf._device >= 0

    cai = buf.__cuda_array_interface__
    assert cai == {
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


def test_sync_primitives():
    rocshmem4py.rocshmem_barrier_all()
    rocshmem4py.rocshmem_fence()
    rocshmem4py.rocshmem_quiet()


def test_create_tensor():
    for dtype in (torch.float32, torch.bfloat16, torch.int32):
        t = rocshmem4py.rocshmem_create_tensor((16,), dtype)
        assert t.shape == (16,) and t.dtype == dtype and t.is_cuda
        assert getattr(t, "__symm_tensor__", False) is True

    t = rocshmem4py.rocshmem_create_tensor((8,), torch.float32)
    view = rocshmem4py.symm_rocshmem_tensor(t, rocshmem4py.rocshmem_my_pe())
    assert view.data_ptr() == t.data_ptr()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
