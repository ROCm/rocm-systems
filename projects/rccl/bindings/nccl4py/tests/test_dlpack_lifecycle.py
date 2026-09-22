# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lifecycle / leak regression tests for the cuda.core HIP shim DLPack
producer + consumer pair.

These tests exercise the upstream-aligned reference-counting contract
that ``nccl/_hip_compat/cuda_core_shim/_dlpack.py`` (producer) and
``nccl/_hip_compat/cuda_core_shim/_memoryview.py`` (consumer) inherit
from ``cuda-core 0.7.0`` ``_dlpack.pyx`` / ``_memoryview.pyx``:

* Producer: ``manager_ctx = Py_INCREF(buf)`` + a real C ``deleter``
  that ``Py_DECREF``s on cleanup.
* Consumer: rename to ``used_dltensor[_versioned]`` claims ownership;
  the StridedMemoryView's ``weakref.finalize`` invokes
  ``dl_tensor.deleter`` exactly once on view destruction.

Both tests run on a CPU-only host (no HIP / no GPU). They use a fake
producer object that satisfies the ``buf``-shape duck-typing required
by :func:`make_py_capsule` (``handle`` / ``size`` / ``is_*_accessible``
/ ``device_id``) and reports ``kCPU`` from ``__dlpack_device__`` so the
DLPack consumer path treats it as a host buffer.
"""

from __future__ import annotations

import gc
import weakref

import pytest

import nccl  # noqa: F401  (registers the cuda.core shim under sys.modules)

# Importing _dlpack pulls only ctypes; importing StridedMemoryView pulls
# _memoryview which imports numpy.
pytest.importorskip("numpy", reason="numpy is required for the DLPack consumer")

from cuda.core.utils import StridedMemoryView  # noqa: E402

from nccl._hip_compat.cuda_core_shim import _dlpack as _dl  # noqa: E402


class _FakeBuf:
    """Minimal duck-type satisfying :func:`_dl.make_py_capsule`'s ``buf``
    contract. ``handle == 0`` is fine because the test never reads the
    pointer; the lifecycle plumbing only cares about refcount.

    Reports ``host-accessible`` (``kCPU``) so the consumer path in
    ``view_as_dlpack`` does not require a HIP device or stream.
    """

    __slots__ = ("__weakref__", "handle", "size", "is_device_accessible",
                 "is_host_accessible", "device_id")

    def __init__(self, size: int = 32):
        self.handle = 0
        self.size = size
        self.is_device_accessible = False
        self.is_host_accessible = True
        self.device_id = 0


class _FakeProducer:
    """DLPack producer wrapper around a :class:`_FakeBuf`. Reports
    ``kCPU`` from ``__dlpack_device__`` so the consumer skips the
    HIP-stream / device-id machinery."""

    __slots__ = ("buf",)

    def __init__(self, buf: _FakeBuf):
        self.buf = buf

    def __dlpack__(self, stream=None, max_version=None):
        versioned = max_version is not None and tuple(max_version) >= (1, 0)
        return _dl.make_py_capsule(self.buf, versioned=versioned)

    def __dlpack_device__(self):
        return (int(_dl.DLDeviceType.kCPU), 0)


def _drop_and_collect():
    """Run a few GC passes to flush weakref.finalize callbacks. CPython
    refcount-based teardown usually runs the finalizer synchronously,
    but a couple of cycles defend against scheduling-dependent timing."""
    for _ in range(3):
        gc.collect()


# ---------------------------------------------------------------------------
# Producer side: PyCapsule destructor releases manager_ctx INCREF
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("versioned", [True, False])
def test_producer_capsule_destructor_releases_manager_ctx(versioned):
    """When a capsule is destroyed without being consumed (consumer never
    renamed to ``used_*``), the PyCapsule destructor must invoke the
    in-struct ``dlm.deleter`` which DECREFs ``manager_ctx``. Mirrors
    the upstream ``pycapsule_deleter`` behavior — failure here would
    leak the producer-pinned buffer for the lifetime of the process.
    """
    buf = _FakeBuf()
    weak = weakref.ref(buf)

    capsule = _dl.make_py_capsule(buf, versioned=versioned)

    # While the capsule lives, the manager_ctx INCREF pins ``buf``
    # even after the user drops their own reference.
    del buf
    gc.collect()
    assert weak() is not None, (
        "capsule should keep buf alive via Py_INCREF on manager_ctx"
    )

    # Drop the capsule: refcount -> 0 -> our pycapsule_deleter runs ->
    # name is unconsumed (dltensor[_versioned]) -> dlm.deleter(dlm) ->
    # Py_DecRef(manager_ctx) -> buf freed.
    del capsule
    _drop_and_collect()

    assert weak() is None, (
        "buf was leaked: PyCapsule destructor did not release the "
        "manager_ctx INCREF"
    )


# ---------------------------------------------------------------------------
# Consumer side: StridedMemoryView finalizer invokes dl_tensor.deleter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("versioned", [True, False])
def test_consumer_smv_finalizer_invokes_dl_tensor_deleter(versioned):
    """Per DLPack spec, a consumer that renamed the capsule to
    ``used_dltensor[_versioned]`` is responsible for calling
    ``dl_tensor.deleter(dl_tensor)`` when done with the data.
    StridedMemoryView's ``weakref.finalize`` does that on SMV
    destruction; failure here would leak the producer-pinned buffer
    on every external ``__dlpack__`` import (PyTorch / cupy / JAX /
    etc.) for the lifetime of the SMV's circular-GC chain.
    """
    buf = _FakeBuf()
    weak = weakref.ref(buf)

    producer = _FakeProducer(buf)
    max_version = (1, 0) if versioned else None

    # Materialize the view (consumer renames capsule to used_*; finalizer
    # registered to call dl_tensor.deleter on SMV destruction).
    view = StridedMemoryView(producer, stream_ptr=-1)

    # Drop user-side references; ``view.exporting_obj`` still pins
    # producer (and thus buf), and the finalizer's bound capsule still
    # pins manager_ctx -> buf.
    del buf, producer, max_version
    gc.collect()
    assert weak() is not None, (
        "buf must remain alive while the SMV holds it via exporting_obj "
        "and the capsule pins it via manager_ctx"
    )

    # Drop the SMV: weakref.finalize fires -> _release_dlpack ->
    # dl_tensor.deleter -> Py_DecRef(manager_ctx); also exporting_obj
    # is released. Both refs to buf go away -> buf freed.
    del view
    _drop_and_collect()

    assert weak() is None, (
        "consumer leaked the producer's manager_ctx: SMV finalizer did "
        "not invoke dl_tensor.deleter"
    )


# ---------------------------------------------------------------------------
# Stress: 1000-iteration loop must not accumulate live buffers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("versioned", [True, False])
def test_consumer_stress_no_accumulating_leak(versioned):
    """Replays the consumer cleanup test in a tight 1000-iteration loop
    that mirrors the production hot path (every ``comm.reduce`` /
    ``comm.allreduce`` call goes through ``view_as_dlpack`` ->
    StridedMemoryView). After ``gc.collect()`` no buffer should remain
    alive.
    """
    n = 1000
    weakrefs = []

    for _ in range(n):
        buf = _FakeBuf()
        producer = _FakeProducer(buf)
        view = StridedMemoryView(producer, stream_ptr=-1)
        weakrefs.append(weakref.ref(buf))
        del producer, view, buf

    _drop_and_collect()

    alive = sum(1 for w in weakrefs if w() is not None)
    assert alive == 0, f"{alive}/{n} buffers leaked across the consumer cycle"
