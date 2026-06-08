"""Shared HIP ctypes helpers for torch-free Python tests."""

from __future__ import annotations

import ctypes

import pytest

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


def hip_check(rc: int, fn: str) -> None:
    if rc != 0:
        raise RuntimeError(f"{fn} failed with HIP error {rc}")


def _as_void_p(buf: object) -> ctypes.c_void_p:
    try:
        return ctypes.cast(buf, ctypes.c_void_p)
    except (TypeError, ctypes.ArgumentError):
        return ctypes.cast(ctypes.byref(buf), ctypes.c_void_p)


def h2d(device_ptr: int, host_buf: object) -> None:
    hip_check(
        _hip.hipMemcpy(
            ctypes.c_void_p(device_ptr),
            _as_void_p(host_buf),
            ctypes.sizeof(host_buf),
            HIP_MEMCPY_HOST_TO_DEVICE,
        ),
        "hipMemcpy H2D",
    )


def d2h(host_buf: object, device_ptr: int) -> None:
    hip_check(
        _hip.hipMemcpy(
            _as_void_p(host_buf),
            ctypes.c_void_p(device_ptr),
            ctypes.sizeof(host_buf),
            HIP_MEMCPY_DEVICE_TO_HOST,
        ),
        "hipMemcpy D2H",
    )


def hip_sync() -> None:
    hip_check(_hip.hipDeviceSynchronize(), "hipDeviceSynchronize")


def store_u64(device_ptr: int, value: int) -> None:
    host_value = ctypes.c_uint64(value)
    h2d(device_ptr, host_value)


def load_u64(device_ptr: int) -> int:
    host_value = ctypes.c_uint64()
    d2h(host_value, device_ptr)
    return int(host_value.value)


class HipStream:
    """Context-manager wrapping a raw hipStream_t."""

    def __init__(self) -> None:
        s = ctypes.c_void_p()
        hip_check(_hip.hipStreamCreate(ctypes.byref(s)), "hipStreamCreate")
        self._handle = s

    @property
    def handle(self) -> int:
        return self._handle.value or 0

    def synchronize(self) -> None:
        hip_check(
            _hip.hipStreamSynchronize(self._handle), "hipStreamSynchronize"
        )

    def __enter__(self) -> "HipStream":
        return self

    def __exit__(self, *exc) -> None:
        if self._handle:
            _hip.hipStreamDestroy(self._handle)
            self._handle = ctypes.c_void_p()
