##############################################################################
# MIT License
#
# Copyright (c) 2021 - 2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

# TODO: Proper way to find library
lib = ctypes.CDLL("/opt/rocm/lib/libamdhip64.so")

# Mirrors struct hipUUID_t
class HIPUUID(ctypes.Structure):
    _fields_ = [
        ("char", ctypes.c_uint8 * 16),
    ]

# Mirrors hipDeviceArch_t
class HIPDeviceArch(ctypes.Structure):
    _fields_ = [
        # 32-bit Atomics
        ("hasGlobalInt32Atomics", ctypes.c_uint, 1),
        ("hasGlobalFloatAtomicExch", ctypes.c_uint, 1),
        ("hasSharedInt32Atomics", ctypes.c_uint, 1),
        ("hasSharedFloatAtomicExch", ctypes.c_uint, 1),
        ("hasFloatAtomicAdd", ctypes.c_uint, 1),

        # 64-bit Atomics
        ("hasGlobalInt64Atomics", ctypes.c_uint, 1),
        ("hasSharedInt64Atomics", ctypes.c_uint, 1),

        # Doubles
        ("hasDoubles", ctypes.c_uint, 1),

        # Warp cross-lane operations
        ("hasWarpVote", ctypes.c_uint, 1),
        ("hasWarpBallot", ctypes.c_uint, 1),
        ("hasWarpShuffle", ctypes.c_uint, 1),
        ("hasFunnelShift", ctypes.c_uint, 1),

        # Sync
        ("hasThreadFenceSystem", ctypes.c_uint, 1),
        ("hasSyncThreadsExt", ctypes.c_uint, 1),

        # Misc
        ("hasSurfaceFuncs", ctypes.c_uint, 1),
        ("has3dGrid", ctypes.c_uint, 1),
        ("hasDynamicParallelism", ctypes.c_uint, 1),
    ]

# Mirrors hipDeviceProp_t
class HIPDeviceProperties(ctypes.Structure):
    _fields_ = [
        ("name_str", ctypes.c_char * 256),
        ("uuid", HIPUUID),
        ("luid", ctypes.c_char * 8),
        ("luidDeviceNodeMask", ctypes.c_uint),
        ("totalGlobalMem", ctypes.c_size_t),
        ("sharedMemPerBlock", ctypes.c_size_t),
        ("regsPerBlock", ctypes.c_int),
        ("warpSize", ctypes.c_int),
        ("memPitch", ctypes.c_size_t),
        ("maxThreadsPerBlock", ctypes.c_int),
        ("maxThreadsDim", ctypes.c_int * 3),
        ("maxGridSize", ctypes.c_int * 3),
        ("clockRate", ctypes.c_int),
        ("totalConstMem", ctypes.c_size_t),
        ("major", ctypes.c_int),
        ("minor", ctypes.c_int),
        ("textureAlignment", ctypes.c_size_t),
        ("texturePitchAlignment", ctypes.c_size_t),
        ("deviceOverlap", ctypes.c_int),
        ("multiProcessorCount", ctypes.c_int),
        ("kernelExecTimeoutEnabled", ctypes.c_int),
        ("integrated", ctypes.c_int),
        ("canMapHostMemory", ctypes.c_int),
        ("computeMode", ctypes.c_int),
        ("maxTexture1D", ctypes.c_int),
        ("maxTexture1DMipmap", ctypes.c_int),
        ("maxTexture1DLinear", ctypes.c_int),
        ("maxTexture2D", ctypes.c_int * 2),
        ("maxTexture2DMipmap", ctypes.c_int * 2),
        ("maxTexture2DLinear", ctypes.c_int * 3),
        ("maxTexture2DGather", ctypes.c_int * 2),
        ("maxTexture3D", ctypes.c_int * 3),
        ("maxTexture3DAlt", ctypes.c_int * 3),
        ("maxTextureCubemap", ctypes.c_int),
        ("maxTexture1DLayered", ctypes.c_int * 2),
        ("maxTexture2DLayered", ctypes.c_int * 3),
        ("maxTextureCubemapLayered", ctypes.c_int * 2),
        ("maxSurface1D", ctypes.c_int),
        ("maxSurface2D", ctypes.c_int * 2),
        ("maxSurface3D", ctypes.c_int * 3),
        ("maxSurface1DLayered", ctypes.c_int * 2),
        ("maxSurface2DLayered", ctypes.c_int * 3),
        ("maxSurfaceCubemap", ctypes.c_int),
        ("maxSurfaceCubemapLayered", ctypes.c_int * 2),
        ("surfaceAlignment", ctypes.c_size_t),
        ("concurrentKernels", ctypes.c_int),
        ("ECCEnabled", ctypes.c_int),
        ("pciBusID", ctypes.c_int),
        ("pciDeviceID", ctypes.c_int),
        ("pciDomainID", ctypes.c_int),
        ("tccDriver", ctypes.c_int),
        ("asyncEngineCount", ctypes.c_int),
        ("unifiedAddressing", ctypes.c_int),
        ("memoryClockRate", ctypes.c_int),
        ("memoryBusWidth", ctypes.c_int),
        ("l2CacheSize", ctypes.c_int),
        ("persistingL2CacheMaxSize", ctypes.c_int),
        ("maxThreadsPerMultiProcessor", ctypes.c_int),
        ("streamPrioritiesSupported", ctypes.c_int),
        ("globalL1CacheSupported", ctypes.c_int),
        ("localL1CacheSupported", ctypes.c_int),
        ("sharedMemPerMultiprocessor", ctypes.c_size_t),
        ("regsPerMultiprocessor", ctypes.c_int),
        ("managedMemory", ctypes.c_int),
        ("isMultiGpuBoard", ctypes.c_int),
        ("multiGpuBoardGroupID", ctypes.c_int),
        ("hostNativeAtomicSupported", ctypes.c_int),
        ("singleToDoublePrecisionPerfRatio", ctypes.c_int),
        ("pageableMemoryAccess", ctypes.c_int),
        ("concurrentManagedAccess", ctypes.c_int),
        ("computePreemptionSupported", ctypes.c_int),
        ("canUseHostPointerForRegisteredMem", ctypes.c_int),
        ("cooperativeLaunch", ctypes.c_int),
        ("cooperativeMultiDeviceLaunch", ctypes.c_int),
        ("sharedMemPerBlockOptin", ctypes.c_size_t),
        ("pageableMemoryAccessUsesHostPageTables", ctypes.c_int),
        ("directManagedMemAccessFromHost", ctypes.c_int),
        ("maxBlocksPerMultiProcessor", ctypes.c_int),
        ("accessPolicyMaxWindowSize", ctypes.c_int),
        ("reservedSharedMemPerBlock", ctypes.c_size_t),
        ("hostRegisterSupported", ctypes.c_int),
        ("sparseHipArraySupported", ctypes.c_int),
        ("hostRegisterReadOnlySupported", ctypes.c_int),
        ("timelineSemaphoreInteropSupported", ctypes.c_int),
        ("memoryPoolsSupported", ctypes.c_int),
        ("gpuDirectRDMASupported", ctypes.c_int),
        ("gpuDirectRDMAFlushWritesOptions", ctypes.c_uint),
        ("gpuDirectRDMAWritesOrdering", ctypes.c_int),
        ("memoryPoolSupportedHandleTypes", ctypes.c_uint),
        ("deferredMappingHipArraySupported", ctypes.c_int),
        ("ipcEventSupported", ctypes.c_int),
        ("clusterLaunch", ctypes.c_int),
        ("unifiedFunctionPointers", ctypes.c_int),
        ("reserved", ctypes.c_int * 63),
        ("hipReserved", ctypes.c_int * 32),

        # HIP-only
        ("gcnArchName_str", ctypes.c_char * 256),
        ("maxSharedMemoryPerMultiProcessor", ctypes.c_size_t),
        ("clockInstructionRate", ctypes.c_int),
        ("arch", HIPDeviceArch),
        ("hdpMemFlushCntl", ctypes.POINTER(ctypes.c_uint)),
        ("hdpRegFlushCntl", ctypes.POINTER(ctypes.c_uint)),
        ("cooperativeMultiDeviceUnmatchedFunc", ctypes.c_int),
        ("cooperativeMultiDeviceUnmatchedGridDim", ctypes.c_int),
        ("cooperativeMultiDeviceUnmatchedBlockDim", ctypes.c_int),
        ("cooperativeMultiDeviceUnmatchedSharedMem", ctypes.c_int),
        ("isLargeBar", ctypes.c_int),
        ("asicRevision", ctypes.c_int),
    ]

    @property
    def name(self):
        return self.name_str.decode('utf-8')

    @property
    def gcnArchName(self):
        return self.gcnArchName_str.decode('utf-8')

def getDeviceCount():

    lib.hipGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    lib.hipGetDeviceCount.restype = ctypes.c_int

    device_count = ctypes.c_int()
    status = lib.hipGetDeviceCount(ctypes.byref(device_count))

    if status != 0:
        raise "HIP error " + str(status)

    return device_count.value

def getDeviceProperties(device_id):
    lib.hipGetDevicePropertiesR0600.argtypes = [ctypes.POINTER(HIPDeviceProperties), ctypes.c_int]
    lib.hipGetDevicePropertiesR0600.restype = ctypes.c_int

    props = HIPDeviceProperties()

    status = lib.hipGetDevicePropertiesR0600(ctypes.byref(props), device_id)

    if status != 0:
        raise "HIP error " + str(status)
    
    return props
