"""Memory-management tests for rocshmem4py.

Covers the symmetric-heap allocators and the user-buffer registration APIs
that previously had no Python coverage:

  * ``rocshmem_calloc``              -- zero-initialized collective allocation
  * ``rocshmem_align``               -- aligned collective allocation
  * ``rocshmem_buffer_register``     -- register a non-symmetric user buffer
  * ``rocshmem_buffer_unregister``   -- deregister a single user buffer
  * ``rocshmem_buffer_unregister_all`` -- deregister every user buffer

These tests are torch-free: they drive HIP directly through ``ctypes`` so they
run in any CI image.  ``calloc`` / ``align`` are collective (each wraps an
internal ``rocshmem_barrier_all``), so every PE must call them the same number
of times -- pytest runs each test on every rank, which preserves that.  The
buffer register/unregister APIs are local backend operations and need real
device memory (the GDA backend registers it with the NIC via ``ibv_reg_mr``),
hence the ``hipMalloc`` shim below.
"""

import ctypes
import pytest

import rocshmem4py  # noqa: E402


# ---------------------------------------------------------------------------
# Minimal HIP ctypes shim (alloc/free + a device->host copy for verification)
# ---------------------------------------------------------------------------

try:
    _hip = ctypes.CDLL("libamdhip64.so")
except OSError as e:
    pytest.skip(f"libamdhip64.so not loadable: {e}", allow_module_level=True)

HIP_MEMCPY_DEVICE_TO_HOST = 2

_hip.hipMalloc.restype = ctypes.c_int
_hip.hipMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
_hip.hipFree.restype = ctypes.c_int
_hip.hipFree.argtypes = [ctypes.c_void_p]
_hip.hipMemcpy.restype = ctypes.c_int
_hip.hipMemcpy.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
]
_hip.hipDeviceSynchronize.restype = ctypes.c_int


def _hip_check(rc: int, fn: str) -> None:
    if rc != 0:
        raise RuntimeError(f"{fn} failed with HIP error {rc}")


def hip_malloc(nbytes: int) -> int:
    p = ctypes.c_void_p()
    _hip_check(_hip.hipMalloc(ctypes.byref(p), nbytes), "hipMalloc")
    return p.value or 0


def hip_free(device_ptr: int) -> None:
    _hip_check(_hip.hipFree(ctypes.c_void_p(device_ptr)), "hipFree")


def d2h(host_buf, device_ptr: int) -> None:
    _hip_check(
        _hip.hipMemcpy(
            ctypes.cast(host_buf, ctypes.c_void_p),
            ctypes.c_void_p(device_ptr),
            ctypes.sizeof(host_buf),
            HIP_MEMCPY_DEVICE_TO_HOST,
        ),
        "hipMemcpy D2H",
    )
    _hip_check(_hip.hipDeviceSynchronize(), "hipDeviceSynchronize")


# ---------------------------------------------------------------------------
# rocshmem_calloc
# ---------------------------------------------------------------------------


def test_calloc_zero_initialized():
    """calloc returns a valid pointer to count*size zero-initialized bytes."""
    count, size = 64, ctypes.sizeof(ctypes.c_int32)
    ptr = rocshmem4py.rocshmem_calloc(count, size)
    try:
        assert ptr > 0
        rx = (ctypes.c_int32 * count)(*([0xABCD] * count))
        d2h(rx, ptr)
        assert all(v == 0 for v in rx), f"calloc memory not zeroed: {list(rx)[:8]}..."
    finally:
        rocshmem4py.rocshmem_free(ptr)


def test_calloc_zero_args_raises():
    """count==0 or size==0 yields NULL in the C API, which the binding maps
    to RuntimeError."""
    with pytest.raises(RuntimeError):
        rocshmem4py.rocshmem_calloc(0, 4)


# ---------------------------------------------------------------------------
# rocshmem_align
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alignment", [8, 64, 256, 4096])
def test_align_returns_aligned_pointer(alignment):
    ptr = rocshmem4py.rocshmem_align(alignment, 1024)
    try:
        assert ptr > 0
        assert ptr % alignment == 0, (
            f"pointer {hex(ptr)} not aligned to {alignment}"
        )
    finally:
        rocshmem4py.rocshmem_free(ptr)


@pytest.mark.parametrize("alignment", [0, 3, 24])
def test_align_invalid_alignment_raises(alignment):
    """alignment must be a non-zero power of two and a multiple of
    sizeof(void*); otherwise the C API returns NULL -> RuntimeError.

    Note: rocshmem_align is collective even on the invalid path (it still
    executes the internal barrier), so calling it on every rank is safe.
    """
    with pytest.raises(RuntimeError):
        rocshmem4py.rocshmem_align(alignment, 1024)


# ---------------------------------------------------------------------------
# rocshmem_buffer_register / rocshmem_buffer_unregister
# ---------------------------------------------------------------------------


def test_buffer_register_unregister():
    """A real (non-symmetric) HIP device buffer can be registered and then
    deregistered, both returning ROCSHMEM_SUCCESS."""
    nbytes = 4096
    addr = hip_malloc(nbytes)
    try:
        rc = rocshmem4py.rocshmem_buffer_register(addr, nbytes)
        assert rc == rocshmem4py.ROCSHMEM_SUCCESS, (
            f"buffer_register returned {rc}"
        )

        rc = rocshmem4py.rocshmem_buffer_unregister(addr)
        assert rc == rocshmem4py.ROCSHMEM_SUCCESS, (
            f"buffer_unregister returned {rc}"
        )
    finally:
        hip_free(addr)


def test_buffer_register_invalid_args():
    """Null address and zero length are rejected (non-zero status)."""
    assert rocshmem4py.rocshmem_buffer_register(0, 4096) != \
        rocshmem4py.ROCSHMEM_SUCCESS

    addr = hip_malloc(4096)
    try:
        assert rocshmem4py.rocshmem_buffer_register(addr, 0) != \
            rocshmem4py.ROCSHMEM_SUCCESS
    finally:
        hip_free(addr)


def test_buffer_register_overlap_rejected():
    """Registering an already-registered region must fail; the original
    registration is still valid and must be cleaned up."""
    nbytes = 4096
    addr = hip_malloc(nbytes)
    try:
        assert rocshmem4py.rocshmem_buffer_register(addr, nbytes) == \
            rocshmem4py.ROCSHMEM_SUCCESS
        # Second registration of the same range overlaps -> rejected.
        assert rocshmem4py.rocshmem_buffer_register(addr, nbytes) != \
            rocshmem4py.ROCSHMEM_SUCCESS
    finally:
        rocshmem4py.rocshmem_buffer_unregister(addr)
        hip_free(addr)


def test_buffer_unregister_unknown_fails():
    """Deregistering an address that was never registered returns non-success."""
    addr = hip_malloc(4096)
    try:
        assert rocshmem4py.rocshmem_buffer_unregister(addr) != \
            rocshmem4py.ROCSHMEM_SUCCESS
    finally:
        hip_free(addr)


# ---------------------------------------------------------------------------
# rocshmem_buffer_unregister_all
# ---------------------------------------------------------------------------


def test_buffer_unregister_all():
    """unregister_all clears every registration; a subsequent single
    unregister of those buffers then fails (already cleared)."""
    nbytes = 4096
    addrs = [hip_malloc(nbytes) for _ in range(3)]
    try:
        for a in addrs:
            assert rocshmem4py.rocshmem_buffer_register(a, nbytes) == \
                rocshmem4py.ROCSHMEM_SUCCESS

        # Returns None (void); must not raise.
        assert rocshmem4py.rocshmem_buffer_unregister_all() is None

        # Everything has been deregistered already.
        for a in addrs:
            assert rocshmem4py.rocshmem_buffer_unregister(a) != \
                rocshmem4py.ROCSHMEM_SUCCESS
    finally:
        for a in addrs:
            hip_free(a)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
