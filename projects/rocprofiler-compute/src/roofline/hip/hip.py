##############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

##############################################################################

import ctypes
import os
from ctypes import (
    POINTER,
    Structure,
    byref,
    c_char,
    c_char_p,
    c_float,
    c_int,
    c_size_t,
    c_uint,
    c_uint8,
    c_void_p,
)

_lib = ctypes.CDLL(f"{os.getenv('ROCM_PATH', '/opt/rocm')}/lib/libamdhip64.so")


# Mirrors struct hipUUID_t
class HIPUUID(Structure):
    _fields_ = [
        ("uuid", c_uint8 * 16),
    ]


# Mirrors hipDeviceArch_t
class HIPDeviceArch(Structure):
    _fields_ = [
        # 32-bit Atomics
        ("hasGlobalInt32Atomics", c_uint, 1),
        ("hasGlobalFloatAtomicExch", c_uint, 1),
        ("hasSharedInt32Atomics", c_uint, 1),
        ("hasSharedFloatAtomicExch", c_uint, 1),
        ("hasFloatAtomicAdd", c_uint, 1),
        # 64-bit Atomics
        ("hasGlobalInt64Atomics", c_uint, 1),
        ("hasSharedInt64Atomics", c_uint, 1),
        # Doubles
        ("hasDoubles", c_uint, 1),
        # Warp cross-lane operations
        ("hasWarpVote", c_uint, 1),
        ("hasWarpBallot", c_uint, 1),
        ("hasWarpShuffle", c_uint, 1),
        ("hasFunnelShift", c_uint, 1),
        # Sync
        ("hasThreadFenceSystem", c_uint, 1),
        ("hasSyncThreadsExt", c_uint, 1),
        # Misc
        ("hasSurfaceFuncs", c_uint, 1),
        ("has3dGrid", c_uint, 1),
        ("hasDynamicParallelism", c_uint, 1),
    ]


# Mirrors hipDeviceProp_tR0600
class HIPDeviceProperties(Structure):
    _fields_ = [
        ("name_str", c_char * 256),
        ("uuid", HIPUUID),
        ("luid", c_char * 8),
        ("luidDeviceNodeMask", c_uint),
        ("totalGlobalMem", c_size_t),
        ("sharedMemPerBlock", c_size_t),
        ("regsPerBlock", c_int),
        ("warpSize", c_int),
        ("memPitch", c_size_t),
        ("maxThreadsPerBlock", c_int),
        ("maxThreadsDim", c_int * 3),
        ("maxGridSize", c_int * 3),
        ("clockRate", c_int),
        ("totalConstMem", c_size_t),
        ("major", c_int),
        ("minor", c_int),
        ("textureAlignment", c_size_t),
        ("texturePitchAlignment", c_size_t),
        ("deviceOverlap", c_int),
        ("multiProcessorCount", c_int),
        ("kernelExecTimeoutEnabled", c_int),
        ("integrated", c_int),
        ("canMapHostMemory", c_int),
        ("computeMode", c_int),
        ("maxTexture1D", c_int),
        ("maxTexture1DMipmap", c_int),
        ("maxTexture1DLinear", c_int),
        ("maxTexture2D", c_int * 2),
        ("maxTexture2DMipmap", c_int * 2),
        ("maxTexture2DLinear", c_int * 3),
        ("maxTexture2DGather", c_int * 2),
        ("maxTexture3D", c_int * 3),
        ("maxTexture3DAlt", c_int * 3),
        ("maxTextureCubemap", c_int),
        ("maxTexture1DLayered", c_int * 2),
        ("maxTexture2DLayered", c_int * 3),
        ("maxTextureCubemapLayered", c_int * 2),
        ("maxSurface1D", c_int),
        ("maxSurface2D", c_int * 2),
        ("maxSurface3D", c_int * 3),
        ("maxSurface1DLayered", c_int * 2),
        ("maxSurface2DLayered", c_int * 3),
        ("maxSurfaceCubemap", c_int),
        ("maxSurfaceCubemapLayered", c_int * 2),
        ("surfaceAlignment", c_size_t),
        ("concurrentKernels", c_int),
        ("ECCEnabled", c_int),
        ("pciBusID", c_int),
        ("pciDeviceID", c_int),
        ("pciDomainID", c_int),
        ("tccDriver", c_int),
        ("asyncEngineCount", c_int),
        ("unifiedAddressing", c_int),
        ("memoryClockRate", c_int),
        ("memoryBusWidth", c_int),
        ("l2CacheSize", c_int),
        ("persistingL2CacheMaxSize", c_int),
        ("maxThreadsPerMultiProcessor", c_int),
        ("streamPrioritiesSupported", c_int),
        ("globalL1CacheSupported", c_int),
        ("localL1CacheSupported", c_int),
        ("sharedMemPerMultiprocessor", c_size_t),
        ("regsPerMultiprocessor", c_int),
        ("managedMemory", c_int),
        ("isMultiGpuBoard", c_int),
        ("multiGpuBoardGroupID", c_int),
        ("hostNativeAtomicSupported", c_int),
        ("singleToDoublePrecisionPerfRatio", c_int),
        ("pageableMemoryAccess", c_int),
        ("concurrentManagedAccess", c_int),
        ("computePreemptionSupported", c_int),
        ("canUseHostPointerForRegisteredMem", c_int),
        ("cooperativeLaunch", c_int),
        ("cooperativeMultiDeviceLaunch", c_int),
        ("sharedMemPerBlockOptin", c_size_t),
        ("pageableMemoryAccessUsesHostPageTables", c_int),
        ("directManagedMemAccessFromHost", c_int),
        ("maxBlocksPerMultiProcessor", c_int),
        ("accessPolicyMaxWindowSize", c_int),
        ("reservedSharedMemPerBlock", c_size_t),
        ("hostRegisterSupported", c_int),
        ("sparseHipArraySupported", c_int),
        ("hostRegisterReadOnlySupported", c_int),
        ("timelineSemaphoreInteropSupported", c_int),
        ("memoryPoolsSupported", c_int),
        ("gpuDirectRDMASupported", c_int),
        ("gpuDirectRDMAFlushWritesOptions", c_uint),
        ("gpuDirectRDMAWritesOrdering", c_int),
        ("memoryPoolSupportedHandleTypes", c_uint),
        ("deferredMappingHipArraySupported", c_int),
        ("ipcEventSupported", c_int),
        ("clusterLaunch", c_int),
        ("unifiedFunctionPointers", c_int),
        ("reserved", c_int * 63),
        ("hipReserved", c_int * 32),
        # HIP-only
        ("gcnArchName_str", c_char * 256),
        ("maxSharedMemoryPerMultiProcessor", c_size_t),
        ("clockInstructionRate", c_int),
        ("arch", HIPDeviceArch),
        ("hdpMemFlushCntl", POINTER(c_uint)),
        ("hdpRegFlushCntl", POINTER(c_uint)),
        ("cooperativeMultiDeviceUnmatchedFunc", c_int),
        ("cooperativeMultiDeviceUnmatchedGridDim", c_int),
        ("cooperativeMultiDeviceUnmatchedBlockDim", c_int),
        ("cooperativeMultiDeviceUnmatchedSharedMem", c_int),
        ("isLargeBar", c_int),
        ("asicRevision", c_int),
    ]

    # Add properties as needed
    @property
    def name(self) -> str:
        return self.name_str.decode("utf-8")

    @property
    def gcnArchName(self) -> str:
        return self.gcnArchName_str.decode("utf-8")


# Declare HIP functions here
_lib.hipGetDeviceCount.restype = c_int
_lib.hipGetDeviceCount.argtypes = [POINTER(c_int)]

_lib.hipGetDevicePropertiesR0600.restype = c_int
_lib.hipGetDevicePropertiesR0600.argtypes = [POINTER(HIPDeviceProperties), c_int]

_lib.hipMalloc.restype = c_int
_lib.hipMalloc.argtypes = [POINTER(c_void_p), c_size_t]

_lib.hipFree.restype = c_int
_lib.hipFree.argtypes = [c_void_p]

_lib.hipMemcpyHtoD.restype = c_int
_lib.hipMemcpyHtoD.argtypes = [c_void_p, c_void_p, c_size_t]

_lib.hipMemcpyDtoH.restype = c_int
_lib.hipMemcpyDtoH.argtypes = [c_void_p, c_void_p, c_size_t]

_lib.hipSetDevice.restype = c_int
_lib.hipSetDevice.argtypes = [c_int]

_lib.hipModuleLoadData.restype = c_int
_lib.hipModuleLoadData.argtypes = [POINTER(c_void_p), c_char_p]

_lib.hipModuleUnload.restype = c_int
_lib.hipModuleUnload.argtypes = [c_void_p]

_lib.hipModuleGetFunction.restype = c_int
_lib.hipModuleGetFunction.argtypes = [POINTER(c_void_p), c_void_p, c_char_p]

_lib.hipDeviceSynchronize.restype = c_int
_lib.hipDeviceSynchronize.argtypes = []

_lib.hipModuleLaunchKernel.restype = c_int
_lib.hipModuleLaunchKernel.argtypes = [
    c_void_p,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_void_p,
    POINTER(c_void_p),
    POINTER(c_void_p),
]

_lib.hipEventCreate.restype = c_int
_lib.hipEventCreate.argtypes = [POINTER(c_void_p)]

_lib.hipEventDestroy.restype = c_int
_lib.hipEventDestroy.argtypes = [c_void_p]

_lib.hipEventRecord.restype = c_int
_lib.hipEventRecord.argtypes = [c_void_p, c_void_p]

_lib.hipEventElapsedTime.restype = c_int
_lib.hipEventElapsedTime.argtypes = [POINTER(c_float), c_void_p, c_void_p]


class HIPError(Exception):
    def __init__(self, code: int) -> None:
        self.code = code
        self.message = f"HIP Error {self.code}"

    def __str__(self) -> str:
        return self.message


class HIPDeviceMemory:
    def __init__(self, ptr: POINTER) -> None:
        self.ptr = ptr

    def __del__(self) -> None:
        _lib.hipFree(self.ptr)


class HIPEvent:
    def __init__(self, handle: POINTER) -> None:
        self.handle = handle

    def __del__(self) -> None:
        _lib.hipEventDestroy(self.handle)


class HIPModule:
    def __init__(self, handle: POINTER) -> None:
        self.handle = handle

    def __del__(self) -> None:
        _lib.hipModuleUnload(self.handle)


# Implement HIP functions here


def hipGetDeviceCount() -> int:
    device_count = c_int()
    status = _lib.hipGetDeviceCount(byref(device_count))

    if status != 0:
        raise HIPError(status)

    return device_count.value


def hipGetDeviceProperties(device_id: int) -> HIPDeviceProperties:
    props = HIPDeviceProperties()
    res = _lib.hipGetDevicePropertiesR0600(byref(props), device_id)

    if res != 0:
        raise HIPError(res)

    return props


def hipMalloc(size: int) -> HIPDeviceMemory:
    buf_size = c_size_t(size)
    ptr = c_void_p()

    status = _lib.hipMalloc(byref(ptr), buf_size)

    if status != 0:
        raise HIPError(status)

    return HIPDeviceMemory(ptr)


def hipMemcpyHtoD(dst: HIPDeviceMemory, src: POINTER, size: int) -> None:
    res = _lib.hipMemcpyHtoD(dst.ptr, src, size)

    if res != 0:
        raise HIPError(res)


def hipMemcpyDtoH(dst: POINTER, src: HIPDeviceMemory, size: int) -> None:
    res = _lib.hipMemcpyDtoH(dst, src.ptr, size)

    if res != 0:
        raise HIPError(res)


def hipSetDevice(id: int) -> None:
    status = _lib.hipSetDevice(id)

    if status != 0:
        raise HIPError(status)


def hipDeviceSynchronize() -> None:
    res = _lib.hipDeviceSynchronize()

    if res != 0:
        raise HIPError(res)


def hipModuleLoadData(code: POINTER) -> HIPModule:
    module = c_void_p()
    res = _lib.hipModuleLoadData(byref(module), code)

    if res != 0:
        raise HIPError(res)

    return HIPModule(module)


def hipModuleGetFunction(module: POINTER, name: str) -> POINTER:
    name_bytes = name.encode("utf-8")
    func = c_void_p()

    res = _lib.hipModuleGetFunction(byref(func), module.handle, name_bytes)

    if res != 0:
        raise HIPError(res)

    return func


def hipModuleLaunchKernel(
    func: POINTER,
    grid_dim_x: int,
    grid_dim_y: int,
    grid_dim_z: int,
    block_dim_x: int,
    block_dim_y: int,
    block_dim_z: int,
    shared_mem_size: int,
    stream: POINTER,
    kernel_params: POINTER,
    extra: POINTER = None,
) -> None:
    res = _lib.hipModuleLaunchKernel(
        func,
        grid_dim_x,
        grid_dim_y,
        grid_dim_z,
        block_dim_x,
        block_dim_y,
        block_dim_z,
        shared_mem_size,
        stream,
        kernel_params,
        extra,
    )

    if res != 0:
        raise HIPError(res)


def hipEventCreate() -> HIPEvent:
    handle = c_void_p()

    res = _lib.hipEventCreate(byref(handle))

    if res != 0:
        raise HIPError(res)

    return HIPEvent(handle)


def hipEventRecord(event: HIPEvent, stream: POINTER = None) -> None:
    res = _lib.hipEventRecord(event.handle, stream)

    if res != 0:
        raise HIPError(res)


def hipEventElapsedTime(start: HIPEvent, stop: HIPEvent) -> float:
    ms = c_float()

    res = _lib.hipEventElapsedTime(byref(ms), start.handle, stop.handle)

    if res != 0:
        raise HIPError(res)

    return ms.value
