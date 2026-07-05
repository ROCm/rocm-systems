# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Surface tests for the ``nccl._hip_compat.cuda_core_shim`` package.

Verifies that the HIP-backed ``cuda.core`` shim implements exactly the
subset that ``nccl/core/*.py`` imports: :class:`Device`, :class:`Stream`,
:class:`Buffer`, :class:`MemoryResource`, :func:`system.get_num_devices`,
:class:`StridedMemoryView`, :func:`args_viewable_as_strided_memory`,
and DLPack capsule export with ``kROCM=10``.

The whole module skips when hip-python is not installed or no HIP
devices are visible, so the same file is safe to collect on a
CPU-only / NVIDIA host.
"""

import numpy as np
import pytest

import nccl  # noqa: F401  (registers the cuda.core shim under sys.modules)

pytest.importorskip("hip", reason="hip-python is required for the HIP shim tests")


from cuda.core import Buffer, Device, MemoryResource, Stream, system  # noqa: E402
from cuda.core.utils import StridedMemoryView, args_viewable_as_strided_memory  # noqa: E402

if system.get_num_devices() == 0:  # pragma: no cover - host without GPUs
    pytest.skip("no HIP devices visible", allow_module_level=True)


# ---------------------------------------------------------------------------
# Device (4 tests)
# ---------------------------------------------------------------------------


class TestDevice:
    def test_device_with_id(self):
        assert Device(0).device_id == 0

    def test_device_default_returns_current(self):
        d = Device()
        assert 0 <= d.device_id < system.get_num_devices()

    def test_set_current_does_not_raise(self):
        Device(0).set_current()

    def test_default_stream_is_stream(self):
        d = Device(0)
        s = d.default_stream
        assert isinstance(s, Stream)
        assert int(s.handle) == 0


# ---------------------------------------------------------------------------
# Stream (3 tests)
# ---------------------------------------------------------------------------


class TestStream:
    def test_from_handle_zero(self):
        assert int(Stream.from_handle(0).handle) == 0

    def test_cuda_stream_protocol_returns_zero_version_and_ptr(self):
        s = Stream.from_handle(0xDEADBEEF)
        info = s.__cuda_stream__()
        assert int(info[0]) == 0
        assert int(info[1]) == 0xDEADBEEF

    def test_round_trip_from_handle(self):
        first = Stream.from_handle(0xCAFE)
        second = Stream.from_handle(first.handle)
        assert int(second.handle) == 0xCAFE


# ---------------------------------------------------------------------------
# Buffer (3 tests)
# ---------------------------------------------------------------------------


class TestBuffer:
    def test_from_handle_round_trip(self):
        b = Buffer.from_handle(ptr=0xBEEF, size=1024, mr=None)
        assert int(b.handle) == 0xBEEF
        assert b.size == 1024
        b.close()

    def test_view_via_strided_memory_view_with_sentinel_stream(self):
        Device(0).set_current()
        b = Buffer.allocate(1024)
        try:
            view = StridedMemoryView(b, stream_ptr=-1)
            assert int(view.ptr) == int(b.handle)
            assert view.shape == (1024,)
            assert view.is_device_accessible is True
        finally:
            b.close()

    def test_allocate_returns_nonzero_handle(self):
        Device(0).set_current()
        b = Buffer.allocate(4096)
        try:
            assert int(b.handle) != 0
            assert b.size == 4096
        finally:
            b.close()


# ---------------------------------------------------------------------------
# MemoryResource (3 tests)
# ---------------------------------------------------------------------------


class TestMemoryResource:
    def test_subclass_allocate_deallocate_cycle(self):
        from nccl._hip_compat.cuda_core_shim._memory import _HipMallocResource

        Device(0).set_current()
        mr = _HipMallocResource(0)
        assert isinstance(mr, MemoryResource)
        assert mr.is_device_accessible is True
        assert mr.is_host_accessible is False
        assert mr.device_id == 0
        b = mr.allocate(2048)
        try:
            assert int(b.handle) != 0
            assert b.size == 2048
        finally:
            b.close()

    def test_double_close_is_safe(self):
        Device(0).set_current()
        b = Buffer.allocate(256)
        b.close()
        b.close()  # must not raise

    def test_deallocate_zero_pointer_is_noop(self):
        from nccl._hip_compat.cuda_core_shim._memory import _HipMallocResource

        mr = _HipMallocResource(0)
        mr.deallocate(0, 0)


# ---------------------------------------------------------------------------
# StridedMemoryView (5 tests)
# ---------------------------------------------------------------------------


class TestStridedMemoryView:
    def test_from_torch_tensor_kROCM(self):
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("torch.cuda (PyTorch-ROCm) not available")
        Device(0).set_current()
        t = torch.empty(64, dtype=torch.float32, device="cuda:0")
        view = StridedMemoryView(t, stream_ptr=-1)
        assert int(view.ptr) == int(t.data_ptr())
        assert view.shape == (64,)
        assert view.dtype == np.float32
        assert int(view.device_id) == 0
        assert view.is_device_accessible is True

    def test_from_cupy_cai(self):
        cp = pytest.importorskip("cupy")
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("cupy reports no CUDA/ROCm devices")
        a = cp.zeros(32, dtype=cp.float32)
        view = StridedMemoryView(a, stream_ptr=-1)
        assert view.shape == (32,)
        assert view.dtype == np.float32
        assert view.is_device_accessible is True

    def test_from_numpy_host_array(self):
        a = np.arange(8, dtype=np.float32)
        view = StridedMemoryView(a, stream_ptr=-1)
        assert view.shape == (8,)
        assert view.dtype == np.dtype(np.float32)
        assert int(view.device_id) == -1
        assert view.is_device_accessible is False

    def test_attributes_populated(self):
        a = np.arange(16, dtype=np.float64)
        view = StridedMemoryView(a, stream_ptr=-1)
        for attr in (
            "ptr",
            "shape",
            "strides",
            "dtype",
            "device_id",
            "is_device_accessible",
            "readonly",
            "exporting_obj",
        ):
            assert hasattr(view, attr), f"missing {attr}"
        assert view.exporting_obj is a

    def test_sentinel_stream_minus_one_for_cpu_tensor(self):
        a = np.arange(4, dtype=np.float32)
        view = StridedMemoryView(a, stream_ptr=-1)
        assert int(view.device_id) == -1


# ---------------------------------------------------------------------------
# args_viewable_as_strided_memory decorator (2 tests)
# ---------------------------------------------------------------------------


class TestArgsViewableDecorator:
    def test_wraps_positional_arg_index_zero(self):
        from nccl._hip_compat.cuda_core_shim._memoryview import _StridedMemoryViewProxy

        captured: dict = {}

        @args_viewable_as_strided_memory((0,))
        def fn(buf, stream_ptr):
            captured["proxy_type"] = type(buf)
            return buf.view(stream_ptr)

        a = np.arange(4, dtype=np.int32)
        view = fn(a, -1)
        assert isinstance(view, StridedMemoryView)
        assert captured["proxy_type"] is _StridedMemoryViewProxy

    def test_preserves_kwargs(self):
        @args_viewable_as_strided_memory((0,))
        def fn(buf, *, stream_ptr=-1):
            return buf.view(stream_ptr)

        a = np.arange(4, dtype=np.int32)
        view = fn(a, stream_ptr=-1)
        assert view.shape == (4,)


# ---------------------------------------------------------------------------
# DLPack producer reports kROCM (1 test)
# ---------------------------------------------------------------------------


class TestDLPackKROCM:
    def test_buffer_dlpack_device_and_capsule_report_kROCM(self):
        Device(0).set_current()
        b = Buffer.allocate(256)
        try:
            dev_type, dev_id = b.__dlpack_device__()
            assert int(dev_type) == 10  # kROCM
            assert int(dev_id) == 0

            from nccl._hip_compat.cuda_core_shim._dlpack import parse_capsule

            capsule = b.__dlpack__(max_version=(1, 0))
            dl_tensor, _, _ = parse_capsule(capsule)
            assert int(dl_tensor.device.device_type) == 10
            assert int(dl_tensor.device.device_id) == 0
        finally:
            b.close()
