# pylint: disable=all
"""
This is a hack to have some semblance of GPU memory management
without introducing a dependency at this early stage of
development. Do not rely upon anything in this module.
"""

import ctypes
import sys

# Load the HIP runtime library
if sys.platform.startswith("linux"):
    _hip_lib_name = "libamdhip64.so"
elif sys.platform == "win32":
    _hip_lib_name = "amdhip64.dll"
else:
    raise OSError("Unsupported platform for HIP runtime")

_hip = ctypes.CDLL(_hip_lib_name)

# hipError_t hipMalloc(void** ptr, size_t size);
_hip.hipMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
_hip.hipMalloc.restype = ctypes.c_int

# hipError_t hipFree(void* ptr);
_hip.hipFree.argtypes = [ctypes.c_void_p]
_hip.hipFree.restype = ctypes.c_int


def hipMalloc(size_bytes: int) -> ctypes.c_void_p:
    """Allocate *size_bytes* of GPU device memory.

    Returns
    -------
    ctypes.c_void_p
        Pointer to the allocated device memory.

    Raises
    ------
    RuntimeError
        If the HIP runtime reports an error.
    """
    d_ptr = ctypes.c_void_p()
    status = _hip.hipMalloc(ctypes.byref(d_ptr), ctypes.c_size_t(size_bytes))
    if status != 0:
        raise RuntimeError(f"hipMalloc failed ({status})")
    return d_ptr


def hipFree(ptr: ctypes.c_void_p) -> None:
    """Free GPU device memory previously allocated by ``hipMalloc``.

    Raises
    ------
    RuntimeError
        If the HIP runtime reports an error.
    """
    status = _hip.hipFree(ptr)
    if status != 0:
        raise RuntimeError(f"hipFree failed ({status})")
