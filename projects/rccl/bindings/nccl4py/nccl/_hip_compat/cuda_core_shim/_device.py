# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``cuda.core.Device`` shim, ported from cuda-core 0.3.2 ``_device.py``.

Surface trimmed to exactly what ``nccl/core/*.py`` touches::

    Device()            -> current HIP device
    Device(int)         -> device with given ordinal
    .device_id          -> int
    .set_current()      -> hipSetDevice
    .default_stream     -> Stream wrapping the HIP NULL stream
    .create_stream(obj) -> Stream wrapping a foreign __cuda_stream__ object

``DeviceProperties`` / contexts / events / graphs / kernels / launch are
intentionally NOT ported. ``nccl/core`` imports none of them.
"""

from __future__ import annotations

import threading
from typing import Optional

from ._hip import check_hip, hip
from ._stream import Stream, StreamOptions, default_stream

_tls = threading.local()


def _hip_device_count() -> int:
    return int(check_hip(hip.hipGetDeviceCount(), "hipGetDeviceCount"))


def _hip_current_device() -> int:
    return int(check_hip(hip.hipGetDevice(), "hipGetDevice"))


def _hip_set_device(dev_id: int) -> None:
    check_hip(hip.hipSetDevice(int(dev_id)), f"hipSetDevice({dev_id})")


class Device:
    """HIP/ROCm device handle.

    A thin wrapper that intentionally does not inherit from cuda-core
    0.3.2's full ``Device`` class: only the methods nccl/core/*.py
    consumes are provided. ``Device(i)`` is deduplicated per-thread, so
    repeated ``Device(0)`` calls return the same instance from the same
    thread (matches upstream behavior).
    """

    __slots__ = ("_id",)

    def __new__(cls, device_id: Optional[int] = None):
        if device_id is None:
            device_id = _hip_current_device()
        else:
            device_id = int(device_id)
            n = _hip_device_count()
            if device_id < 0 or device_id >= n:
                raise ValueError(
                    f"device_id {device_id} is out of range; HIP reports {n} device(s)"
                )

        cache = getattr(_tls, "devices", None)
        if cache is None:
            cache = {}
            _tls.devices = cache
        inst = cache.get(device_id)
        if inst is not None:
            return inst
        inst = super().__new__(cls)
        inst._id = device_id
        cache[device_id] = inst
        return inst

    @property
    def device_id(self) -> int:
        return self._id

    def set_current(self) -> None:
        _hip_set_device(self._id)

    @property
    def default_stream(self) -> Stream:
        return default_stream()

    def create_stream(self, obj=None, options: Optional[StreamOptions] = None) -> Stream:
        """Wrap a foreign stream-like object (passes ``__cuda_stream__`` check)
        or create a new HIP stream with optional :class:`StreamOptions`.
        """
        return Stream._init(obj=obj, options=options, device_id=self._id)

    def __repr__(self) -> str:
        return f"<Device id={self._id} (HIP shim)>"

    def __eq__(self, other) -> bool:
        return isinstance(other, Device) and other._id == self._id

    def __hash__(self) -> int:
        return hash(("nccl._hip_compat.Device", self._id))
