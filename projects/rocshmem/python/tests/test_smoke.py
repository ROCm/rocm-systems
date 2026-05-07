"""Torch-free smoke tests for rocshmem4py.

These tests exercise the native host API through ``ctypes`` + HIP, so they
run in any CI environment even when ``torch`` is not installed.

Scope:

  * ``test_create_buffer`` -- validates the framework-agnostic collective
    allocation API (:func:`rocshmem4py.rocshmem_create_buffer`).
  * ``test_symmetric_buffer_lifecycle`` -- SymmetricBuffer RAII lifecycle;
    torch integration path is covered by ``test_basic.py::test_symmetric_buffer``.
  * ``test_host_device_roundtrip`` -- validates the symmetric heap is
    accessible as ordinary HIP device memory via ctypes H2D/D2H transfers.
  * ``test_get_peer_buffer`` -- validates :func:`rocshmem4py.rocshmem_get_peer_buffer`
    (torch-free equivalent of ``symm_rocshmem_tensor``).
  * Multi-PE stream-based transfers using ``rocshmem_create_buffer`` for
    allocation (``*_on_stream`` portable APIs only -- see test_collective.py
    docstring for rationale).

"""

import ctypes
import pytest

from conftest import requires_multi_pe  # noqa: E402

import rocshmem4py  # noqa: E402


# ---------------------------------------------------------------------------
# Minimal HIP ctypes shim (just enough to move bytes and drive a stream)
# ---------------------------------------------------------------------------

try:
    _hip = ctypes.CDLL("libamdhip64.so")
except OSError as e:
    pytest.skip(f"libamdhip64.so not loadable: {e}", allow_module_level=True)

HIP_MEMCPY_HOST_TO_DEVICE = 1
HIP_MEMCPY_DEVICE_TO_HOST = 2

_hip.hipDeviceSynchronize.restype = ctypes.c_int
_hip.hipMemcpy.restype = ctypes.c_int
_hip.hipMemcpy.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
]
_hip.hipStreamCreate.restype = ctypes.c_int
_hip.hipStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
_hip.hipStreamDestroy.restype = ctypes.c_int
_hip.hipStreamDestroy.argtypes = [ctypes.c_void_p]
_hip.hipStreamSynchronize.restype = ctypes.c_int
_hip.hipStreamSynchronize.argtypes = [ctypes.c_void_p]


def _hip_check(rc: int, fn: str) -> None:
    if rc != 0:
        raise RuntimeError(f"{fn} failed with HIP error {rc}")


def h2d(device_ptr: int, host_buf) -> None:
    _hip_check(
        _hip.hipMemcpy(
            ctypes.c_void_p(device_ptr),
            ctypes.cast(host_buf, ctypes.c_void_p),
            ctypes.sizeof(host_buf),
            HIP_MEMCPY_HOST_TO_DEVICE,
        ),
        "hipMemcpy H2D",
    )


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


def hip_sync() -> None:
    _hip_check(_hip.hipDeviceSynchronize(), "hipDeviceSynchronize")


class HipStream:
    """Context-manager wrapping a raw hipStream_t.

    The integer handle matches the type expected by ``rocshmem_*_on_stream``
    bindings.  In torch environments the equivalent is
    ``torch.cuda.current_stream().cuda_stream``.
    """

    def __init__(self) -> None:
        s = ctypes.c_void_p()
        _hip_check(_hip.hipStreamCreate(ctypes.byref(s)), "hipStreamCreate")
        self._handle = s

    @property
    def handle(self) -> int:
        return self._handle.value or 0

    def synchronize(self) -> None:
        _hip_check(
            _hip.hipStreamSynchronize(self._handle), "hipStreamSynchronize"
        )

    def __enter__(self) -> "HipStream":
        return self

    def __exit__(self, *exc) -> None:
        if self._handle:
            _hip.hipStreamDestroy(self._handle)
            self._handle = ctypes.c_void_p()


# ---------------------------------------------------------------------------
# Single-PE tests unique to this file (torch-free but HIP-specific)
# ---------------------------------------------------------------------------


def test_create_buffer():
    """rocshmem_create_buffer is the framework-agnostic collective allocator.

    Validates that the returned SymmetricBuffer has the correct shape, pointer,
    __cuda_array_interface__ fields, and a valid symmetric address for self.
    """
    nbytes = 4096
    buf = rocshmem4py.rocshmem_create_buffer(nbytes)
    try:
        assert isinstance(buf, rocshmem4py.SymmetricBuffer)
        assert buf.ptr > 0
        assert buf.nbytes == nbytes
        assert buf.size == nbytes
        assert buf.own_data is True
        assert int(buf) == buf.ptr
        # __cuda_array_interface__ must be present for framework interop (e.g. torch)
        cai = buf.__cuda_array_interface__
        assert cai["data"] == (buf.ptr, False)
        assert cai["shape"] == (nbytes,)
        assert cai["version"] == 3
        # Symmetric pointer to self must be valid
        my_pe = rocshmem4py.rocshmem_my_pe()
        assert buf.get_remote_ptr(my_pe) >= 0
    finally:
        buf.free()
        assert buf._freed is True


def test_symmetric_buffer_lifecycle():
    """SymmetricBuffer RAII lifecycle: allocation, pointer validity, remote ptr, and free.

    Covers the torch-free API surface only.  The torch integration path
    (``__cuda_array_interface__``, ``torch.as_tensor``, non-owning view,
    ``_device``) is covered by ``test_basic.py::test_symmetric_buffer``.
    """
    size = 2048
    buf = rocshmem4py.SymmetricBuffer(size)
    try:
        assert buf.size == size and buf.nbytes == size and buf.ptr > 0
        assert int(buf) == buf.ptr
        my_pe = rocshmem4py.rocshmem_my_pe()
        assert buf.get_remote_ptr(my_pe) >= 0
    finally:
        buf.free()
        assert buf._freed is True


def test_host_device_roundtrip():
    """Symmetric heap memory is accessible as ordinary HIP device memory."""
    nelems = 16
    nbytes = nelems * ctypes.sizeof(ctypes.c_int32)
    buf = rocshmem4py.rocshmem_create_buffer(nbytes)
    try:
        tx = (ctypes.c_int32 * nelems)(*range(nelems))
        h2d(buf.ptr, tx)
        hip_sync()

        rx = (ctypes.c_int32 * nelems)()
        d2h(rx, buf.ptr)
        hip_sync()

        assert list(rx) == list(range(nelems))
    finally:
        buf.free()


# ---------------------------------------------------------------------------
# Multi-PE tests — stream-based portable APIs only.
# Host-side blocking APIs are not exercised here; see test_collective.py
# module docstring for the rationale.
# ---------------------------------------------------------------------------


@requires_multi_pe
def test_putmem_on_stream_peer():
    """Torch-free: push local buffer to a peer via rocshmem_putmem_on_stream."""
    nelems = 32
    nbytes = nelems * ctypes.sizeof(ctypes.c_int32)

    my_pe = rocshmem4py.rocshmem_my_pe()
    n_pes = rocshmem4py.rocshmem_n_pes()
    peer = (my_pe + 1) % n_pes

    src = rocshmem4py.rocshmem_create_buffer(nbytes)
    dst = rocshmem4py.rocshmem_create_buffer(nbytes)
    try:
        src_host = (ctypes.c_int32 * nelems)(*([my_pe] * nelems))
        dst_host = (ctypes.c_int32 * nelems)(*([-1] * nelems))
        h2d(src.ptr, src_host)
        h2d(dst.ptr, dst_host)
        hip_sync()
        rocshmem4py.rocshmem_barrier_all()

        with HipStream() as s:
            rocshmem4py.rocshmem_putmem_on_stream(
                dst.ptr, src.ptr, nbytes, peer, s.handle
            )
            rocshmem4py.rocshmem_barrier_all_on_stream(s.handle)
            s.synchronize()

        rx = (ctypes.c_int32 * nelems)()
        d2h(rx, dst.ptr)
        hip_sync()

        sender = (my_pe - 1 + n_pes) % n_pes
        assert all(v == sender for v in rx), (
            f"PE {my_pe}: expected all {sender}, got {list(rx)[:8]}..."
        )
    finally:
        src.free()
        dst.free()


@requires_multi_pe
def test_getmem_on_stream_peer():
    """Torch-free: pull peer buffer into local buffer via rocshmem_getmem_on_stream."""
    nelems = 32
    nbytes = nelems * ctypes.sizeof(ctypes.c_int32)

    my_pe = rocshmem4py.rocshmem_my_pe()
    n_pes = rocshmem4py.rocshmem_n_pes()
    peer = (my_pe + 1) % n_pes

    src = rocshmem4py.rocshmem_create_buffer(nbytes)
    dst = rocshmem4py.rocshmem_create_buffer(nbytes)
    try:
        src_host = (ctypes.c_int32 * nelems)(*([my_pe] * nelems))
        dst_host = (ctypes.c_int32 * nelems)(*([-1] * nelems))
        h2d(src.ptr, src_host)
        h2d(dst.ptr, dst_host)
        hip_sync()

        with HipStream() as s:
            rocshmem4py.rocshmem_barrier_all_on_stream(s.handle)
            rocshmem4py.rocshmem_getmem_on_stream(
                dst.ptr, src.ptr, nbytes, peer, s.handle
            )
            rocshmem4py.rocshmem_barrier_all_on_stream(s.handle)
            s.synchronize()

        rx = (ctypes.c_int32 * nelems)()
        d2h(rx, dst.ptr)
        hip_sync()

        assert all(v == peer for v in rx), (
            f"PE {my_pe}: expected all {peer}, got {list(rx)[:8]}..."
        )
    finally:
        src.free()
        dst.free()


@requires_multi_pe
def test_get_peer_buffer():
    """``rocshmem_get_peer_buffer`` returns a zero-copy view on IPC backends.

    On backends without direct remote memory access ``rocshmem_ptr`` returns
    NULL and the function raises ``RuntimeError`` (skipped here).  Use
    ``rocshmem_getmem_on_stream`` to copy peer data on those backends —
    that path is exercised by ``test_getmem_on_stream_peer`` above.
    """
    my_pe = rocshmem4py.rocshmem_my_pe()
    n_pes = rocshmem4py.rocshmem_n_pes()
    peer = (my_pe + 1) % n_pes

    nbytes = 256
    buf = rocshmem4py.rocshmem_create_buffer(nbytes)
    try:
        rocshmem4py.rocshmem_barrier_all()
        try:
            peer_buf = rocshmem4py.rocshmem_get_peer_buffer(buf, peer)
        except RuntimeError:
            pytest.skip(
                "rocshmem_ptr returned NULL -- backend lacks direct remote "
                "memory access; copy path covered by test_getmem_on_stream_peer"
            )
        assert isinstance(peer_buf, rocshmem4py.SymmetricBuffer)
        assert peer_buf.ptr > 0
        assert peer_buf.own_data is False
        assert peer_buf.nbytes == nbytes
    finally:
        buf.free()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
