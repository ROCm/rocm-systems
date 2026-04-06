/*
 * Copyright 2025 Advanced Micro Devices, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file hip_api_device.c
 * @brief Device management API implementations for remote HIP
 */

#include "hip_remote/hip_remote_internal.h"
#include "hip_remote/hip_remote_protocol.h"

#include <string.h>

#ifdef HIP_REMOTE_HAS_HIP_HEADERS
#include <hip/hip_runtime_api.h>
#endif

/* ============================================================================
 * Device Management APIs
 * ============================================================================ */

static int g_cached_device_count = -1;

hipError_t hipGetDeviceCount(int* count) {
    if (!count) {
        return hipErrorInvalidValue;
    }

    if (g_cached_device_count >= 0) {
        *count = g_cached_device_count;
        return hipSuccess;
    }

    HipRemoteDeviceCountResponse resp;
    hipError_t err = hip_remote_request(
        HIP_OP_GET_DEVICE_COUNT,
        NULL, 0,
        &resp, sizeof(resp)
    );

    if (err == hipSuccess) {
        *count = resp.count;
        g_cached_device_count = resp.count;
    }
    return err;
}

static int g_cached_current_device = -1;

hipError_t hipSetDevice(int deviceId) {
    /* Skip if already on the requested device */
    if (g_cached_current_device == deviceId) {
        return hipSuccess;
    }

    HipRemoteDeviceRequest req;
    memset(&req, 0, sizeof(req));
    req.device_id = deviceId;
    HipRemoteResponseHeader resp;

    hipError_t err = hip_remote_request(
        HIP_OP_SET_DEVICE,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );

    if (err == hipSuccess) {
        g_cached_current_device = deviceId;
    }
    return err;
}

hipError_t hipGetDevice(int* deviceId) {
    if (!deviceId) {
        return hipErrorInvalidValue;
    }

    if (g_cached_current_device >= 0) {
        *deviceId = g_cached_current_device;
        return hipSuccess;
    }

    HipRemoteGetDeviceResponse resp;
    hipError_t err = hip_remote_request(
        HIP_OP_GET_DEVICE,
        NULL, 0,
        &resp, sizeof(resp)
    );

    if (err == hipSuccess) {
        *deviceId = resp.device_id;
        g_cached_current_device = resp.device_id;
    }
    return err;
}

hipError_t hipDeviceSynchronize(void) {
    HipRemoteResponseHeader resp;
    return hip_remote_request(
        HIP_OP_DEVICE_SYNCHRONIZE,
        NULL, 0,
        &resp, sizeof(resp)
    );
}

hipError_t hipDeviceReset(void) {
    HipRemoteResponseHeader resp;
    return hip_remote_request(
        HIP_OP_DEVICE_RESET,
        NULL, 0,
        &resp, sizeof(resp)
    );
}

/* Attribute cache: key = (deviceId << 16) | attr, value = result */
#define ATTR_CACHE_SIZE 4096
static struct { uint32_t key; int value; int valid; } g_attr_cache[ATTR_CACHE_SIZE];
static hip_mutex_t g_attr_cache_lock = HIP_MUTEX_INIT;

hipError_t hipDeviceGetAttribute(int* value, hipDeviceAttribute_t attr, int deviceId) {
    if (!value) {
        return hipErrorInvalidValue;
    }

    /* Check cache first */
    uint32_t cache_key = ((uint32_t)deviceId << 16) | ((uint32_t)attr & 0xFFFF);
    uint32_t cache_idx = cache_key % ATTR_CACHE_SIZE;

    hip_mutex_lock(&g_attr_cache_lock);
    if (g_attr_cache[cache_idx].valid && g_attr_cache[cache_idx].key == cache_key) {
        *value = g_attr_cache[cache_idx].value;
        hip_mutex_unlock(&g_attr_cache_lock);
        return hipSuccess;
    }
    hip_mutex_unlock(&g_attr_cache_lock);

    HipRemoteDeviceAttributeRequest req;
    memset(&req, 0, sizeof(req));
    req.device_id = deviceId;
    req.attribute = attr;
    HipRemoteDeviceAttributeResponse resp;

    hipError_t err = hip_remote_request(
        HIP_OP_DEVICE_GET_ATTRIBUTE,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );

    if (err == hipSuccess) {
        *value = resp.value;
        hip_mutex_lock(&g_attr_cache_lock);
        g_attr_cache[cache_idx].key = cache_key;
        g_attr_cache[cache_idx].value = resp.value;
        g_attr_cache[cache_idx].valid = 1;
        hip_mutex_unlock(&g_attr_cache_lock);
    }
    return err;
}

/* ============================================================================
 * Device Properties
 *
 * Note: We define a minimal hipDeviceProp_t here for the remote client.
 * The full definition would come from HIP headers, but for the remote
 * client we only need to map the fields we care about.
 * ============================================================================ */

#define MAX_CACHED_DEVICES 16
static int g_props_cached[MAX_CACHED_DEVICES];

#ifdef hipGetDeviceProperties
#undef hipGetDeviceProperties
#endif
hipError_t hipGetDeviceProperties(hipDeviceProp_t* prop, int deviceId) {
    return hipGetDevicePropertiesR0600(prop, deviceId);
}

/* ============================================================================
 * Runtime/Driver Version
 * ============================================================================ */

static int g_cached_runtime_version = 0;

hipError_t hipRuntimeGetVersion(int* runtimeVersion) {
    if (!runtimeVersion) {
        return hipErrorInvalidValue;
    }

    if (g_cached_runtime_version > 0) {
        *runtimeVersion = g_cached_runtime_version;
        return hipSuccess;
    }

    HipRemoteVersionResponse resp;
    hipError_t err = hip_remote_request(
        HIP_OP_RUNTIME_GET_VERSION,
        NULL, 0,
        &resp, sizeof(resp)
    );

    if (err == hipSuccess) {
        *runtimeVersion = resp.version;
        g_cached_runtime_version = resp.version;
    }
    return err;
}

hipError_t hipDriverGetVersion(int* driverVersion) {
    static int g_cached_driver_version = 0;
    if (!driverVersion) {
        return hipErrorInvalidValue;
    }

    if (g_cached_driver_version != 0) {
        *driverVersion = g_cached_driver_version;
        return hipSuccess;
    }

    HipRemoteVersionResponse resp;
    hipError_t err = hip_remote_request(
        HIP_OP_DRIVER_GET_VERSION,
        NULL, 0,
        &resp, sizeof(resp)
    );

    if (err == hipSuccess) {
        *driverVersion = resp.version;
        g_cached_driver_version = resp.version;
    }
    return err;
}

/* ============================================================================
 * Device Limits
 * ============================================================================ */

hipError_t hipDeviceGetLimit(size_t* pValue, enum hipLimit_t limit) {
    if (!pValue) {
        return hipErrorInvalidValue;
    }

    HipRemoteDeviceLimitRequest req = {
        .limit = (int32_t)limit,
        .value = 0
    };
    HipRemoteDeviceLimitResponse resp;

    hipError_t err = hip_remote_request(
        HIP_OP_DEVICE_GET_LIMIT,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );

    if (err == hipSuccess) {
        *pValue = (size_t)resp.value;
    }
    return err;
}

hipError_t hipDeviceSetLimit(enum hipLimit_t limit, size_t value) {
    HipRemoteDeviceLimitRequest req = {
        .limit = (int32_t)limit,
        .value = (uint64_t)value
    };

    return hip_remote_request_fire_and_forget(
        HIP_OP_DEVICE_SET_LIMIT, &req, sizeof(req)
    );
}

/* ============================================================================
 * Peer Access
 * ============================================================================ */

hipError_t hipDeviceCanAccessPeer(int* canAccessPeer, int deviceId, int peerDeviceId) {
    if (!canAccessPeer) {
        return hipErrorInvalidValue;
    }

    HipRemoteDeviceCanAccessPeerRequest req = {
        .device_id = deviceId,
        .peer_device_id = peerDeviceId
    };
    HipRemoteDeviceCanAccessPeerResponse resp;

    hipError_t err = hip_remote_request(
        HIP_OP_DEVICE_CAN_ACCESS_PEER,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );

    if (err == hipSuccess) {
        *canAccessPeer = resp.can_access_peer;
    }
    return err;
}

hipError_t hipDeviceEnablePeerAccess(int peerDeviceId, unsigned int flags) {
    HipRemoteDevicePeerAccessRequest req = {
        .peer_device_id = peerDeviceId,
        .flags = flags
    };

    return hip_remote_request_fire_and_forget(
        HIP_OP_DEVICE_ENABLE_PEER_ACCESS, &req, sizeof(req)
    );
}

hipError_t hipDeviceDisablePeerAccess(int peerDeviceId) {
    HipRemoteDevicePeerAccessRequest req = {
        .peer_device_id = peerDeviceId,
        .flags = 0
    };

    return hip_remote_request_fire_and_forget(
        HIP_OP_DEVICE_DISABLE_PEER_ACCESS, &req, sizeof(req)
    );
}

/* ============================================================================
 * Error Handling
 * ============================================================================ */

hipError_t hipGetLastError(void) {
    HipRemoteClientState* state = hip_remote_get_client_state();
    if (!state) return hipSuccess;
    hipError_t err = state->last_error;
    state->last_error = hipSuccess;
    return err;
}

hipError_t hipPeekAtLastError(void) {
    HipRemoteClientState* state = hip_remote_get_client_state();
    if (!state) return hipSuccess;
    return state->last_error;
}

/*
 * Error string implementations.
 *
 * These provide the same functionality as the HIP runtime functions in
 * rocm-systems/projects/clr/hipamd/src/hip_error.cpp.
 *
 * We implement these locally because:
 * 1. The remote client runs on macOS where HIP libraries are not available
 * 2. The client needs to translate error codes returned from the worker
 *
 * Error codes match hip_runtime_api.h. Last synced: HIP 6.3 (ROCm 6.3)
 */
const char* hipGetErrorName(hipError_t error) {
    switch (error) {
        case hipSuccess: return "hipSuccess";
        case hipErrorInvalidValue: return "hipErrorInvalidValue";
        case hipErrorOutOfMemory: return "hipErrorOutOfMemory";
        case hipErrorNotInitialized: return "hipErrorNotInitialized";
        case hipErrorDeinitialized: return "hipErrorDeinitialized";
        case hipErrorProfilerDisabled: return "hipErrorProfilerDisabled";
        case hipErrorProfilerNotInitialized: return "hipErrorProfilerNotInitialized";
        case hipErrorProfilerAlreadyStarted: return "hipErrorProfilerAlreadyStarted";
        case hipErrorProfilerAlreadyStopped: return "hipErrorProfilerAlreadyStopped";
        case hipErrorInvalidConfiguration: return "hipErrorInvalidConfiguration";
        case hipErrorInvalidSymbol: return "hipErrorInvalidSymbol";
        case hipErrorInvalidDevicePointer: return "hipErrorInvalidDevicePointer";
        case hipErrorInvalidMemcpyDirection: return "hipErrorInvalidMemcpyDirection";
        case hipErrorInsufficientDriver: return "hipErrorInsufficientDriver";
        case hipErrorMissingConfiguration: return "hipErrorMissingConfiguration";
        case hipErrorPriorLaunchFailure: return "hipErrorPriorLaunchFailure";
        case hipErrorInvalidDeviceFunction: return "hipErrorInvalidDeviceFunction";
        case hipErrorNoDevice: return "hipErrorNoDevice";
        case hipErrorInvalidDevice: return "hipErrorInvalidDevice";
        case hipErrorInvalidPitchValue: return "hipErrorInvalidPitchValue";
        case hipErrorInvalidImage: return "hipErrorInvalidImage";
        case hipErrorInvalidContext: return "hipErrorInvalidContext";
        case hipErrorContextAlreadyCurrent: return "hipErrorContextAlreadyCurrent";
        case hipErrorMapFailed: return "hipErrorMapFailed";
        case hipErrorUnmapFailed: return "hipErrorUnmapFailed";
        case hipErrorArrayIsMapped: return "hipErrorArrayIsMapped";
        case hipErrorAlreadyMapped: return "hipErrorAlreadyMapped";
        case hipErrorNoBinaryForGpu: return "hipErrorNoBinaryForGpu";
        case hipErrorAlreadyAcquired: return "hipErrorAlreadyAcquired";
        case hipErrorNotMapped: return "hipErrorNotMapped";
        case hipErrorNotMappedAsArray: return "hipErrorNotMappedAsArray";
        case hipErrorNotMappedAsPointer: return "hipErrorNotMappedAsPointer";
        case hipErrorECCNotCorrectable: return "hipErrorECCNotCorrectable";
        case hipErrorUnsupportedLimit: return "hipErrorUnsupportedLimit";
        case hipErrorContextAlreadyInUse: return "hipErrorContextAlreadyInUse";
        case hipErrorPeerAccessUnsupported: return "hipErrorPeerAccessUnsupported";
        case hipErrorInvalidKernelFile: return "hipErrorInvalidKernelFile";
        case hipErrorInvalidGraphicsContext: return "hipErrorInvalidGraphicsContext";
        case hipErrorInvalidSource: return "hipErrorInvalidSource";
        case hipErrorFileNotFound: return "hipErrorFileNotFound";
        case hipErrorSharedObjectSymbolNotFound: return "hipErrorSharedObjectSymbolNotFound";
        case hipErrorSharedObjectInitFailed: return "hipErrorSharedObjectInitFailed";
        case hipErrorOperatingSystem: return "hipErrorOperatingSystem";
        case hipErrorInvalidHandle: return "hipErrorInvalidHandle";
        case hipErrorIllegalState: return "hipErrorIllegalState";
        case hipErrorNotFound: return "hipErrorNotFound";
        case hipErrorNotReady: return "hipErrorNotReady";
        case hipErrorIllegalAddress: return "hipErrorIllegalAddress";
        case hipErrorLaunchOutOfResources: return "hipErrorLaunchOutOfResources";
        case hipErrorLaunchTimeOut: return "hipErrorLaunchTimeOut";
        case hipErrorPeerAccessAlreadyEnabled: return "hipErrorPeerAccessAlreadyEnabled";
        case hipErrorPeerAccessNotEnabled: return "hipErrorPeerAccessNotEnabled";
        case hipErrorSetOnActiveProcess: return "hipErrorSetOnActiveProcess";
        case hipErrorContextIsDestroyed: return "hipErrorContextIsDestroyed";
        case hipErrorAssert: return "hipErrorAssert";
        case hipErrorHostMemoryAlreadyRegistered: return "hipErrorHostMemoryAlreadyRegistered";
        case hipErrorHostMemoryNotRegistered: return "hipErrorHostMemoryNotRegistered";
        case hipErrorLaunchFailure: return "hipErrorLaunchFailure";
        case hipErrorNotSupported: return "hipErrorNotSupported";
        case hipErrorUnknown: return "hipErrorUnknown";
        case hipErrorRuntimeMemory: return "hipErrorRuntimeMemory";
        case hipErrorRuntimeOther: return "hipErrorRuntimeOther";
        case hipErrorCooperativeLaunchTooLarge: return "hipErrorCooperativeLaunchTooLarge";
        case hipErrorStreamCaptureUnsupported: return "hipErrorStreamCaptureUnsupported";
        case hipErrorStreamCaptureInvalidated: return "hipErrorStreamCaptureInvalidated";
        case hipErrorStreamCaptureMerge: return "hipErrorStreamCaptureMerge";
        case hipErrorStreamCaptureUnmatched: return "hipErrorStreamCaptureUnmatched";
        case hipErrorStreamCaptureUnjoined: return "hipErrorStreamCaptureUnjoined";
        case hipErrorStreamCaptureIsolation: return "hipErrorStreamCaptureIsolation";
        case hipErrorStreamCaptureImplicit: return "hipErrorStreamCaptureImplicit";
        case hipErrorCapturedEvent: return "hipErrorCapturedEvent";
        case hipErrorStreamCaptureWrongThread: return "hipErrorStreamCaptureWrongThread";
        case hipErrorGraphExecUpdateFailure: return "hipErrorGraphExecUpdateFailure";
        case hipErrorInvalidChannelDescriptor: return "hipErrorInvalidChannelDescriptor";
        case hipErrorInvalidTexture: return "hipErrorInvalidTexture";
        case hipErrorTbd: return "hipErrorTbd";
        default: return "hipErrorUnknown";
    }
}

const char* hipGetErrorString(hipError_t error) {
    switch (error) {
        case hipSuccess: return "no error";
        case hipErrorInvalidValue: return "invalid argument";
        case hipErrorOutOfMemory: return "out of memory";
        case hipErrorNotInitialized: return "initialization error";
        case hipErrorDeinitialized: return "driver shutting down";
        case hipErrorProfilerDisabled: return "profiler disabled while using external profiling tool";
        case hipErrorProfilerNotInitialized: return "profiler is not initialized";
        case hipErrorProfilerAlreadyStarted: return "profiler already started";
        case hipErrorProfilerAlreadyStopped: return "profiler already stopped";
        case hipErrorInvalidConfiguration: return "invalid configuration argument";
        case hipErrorInvalidPitchValue: return "invalid pitch argument";
        case hipErrorInvalidSymbol: return "invalid device symbol";
        case hipErrorInvalidDevicePointer: return "invalid device pointer";
        case hipErrorInvalidMemcpyDirection: return "invalid copy direction for memcpy";
        case hipErrorInsufficientDriver: return "driver version is insufficient for runtime version";
        case hipErrorMissingConfiguration: return "__global__ function call is not configured";
        case hipErrorPriorLaunchFailure: return "unspecified launch failure in prior launch";
        case hipErrorInvalidDeviceFunction: return "invalid device function";
        case hipErrorNoDevice: return "no ROCm-capable device is detected";
        case hipErrorInvalidDevice: return "invalid device ordinal";
        case hipErrorInvalidImage: return "device kernel image is invalid";
        case hipErrorInvalidContext: return "invalid device context";
        case hipErrorContextAlreadyCurrent: return "context is already current context";
        case hipErrorMapFailed: return "mapping of buffer object failed";
        case hipErrorUnmapFailed: return "unmapping of buffer object failed";
        case hipErrorArrayIsMapped: return "array is mapped";
        case hipErrorAlreadyMapped: return "resource already mapped";
        case hipErrorNoBinaryForGpu: return "no kernel image is available for execution on the device";
        case hipErrorAlreadyAcquired: return "resource already acquired";
        case hipErrorNotMapped: return "resource not mapped";
        case hipErrorNotMappedAsArray: return "resource not mapped as array";
        case hipErrorNotMappedAsPointer: return "resource not mapped as pointer";
        case hipErrorECCNotCorrectable: return "uncorrectable ECC error encountered";
        case hipErrorUnsupportedLimit: return "limit is not supported on this architecture";
        case hipErrorContextAlreadyInUse: return "exclusive-thread device already in use by a different thread";
        case hipErrorPeerAccessUnsupported: return "peer access is not supported between these two devices";
        case hipErrorInvalidKernelFile: return "invalid kernel file";
        case hipErrorInvalidGraphicsContext: return "invalid OpenGL or DirectX context";
        case hipErrorInvalidSource: return "device kernel image is invalid";
        case hipErrorFileNotFound: return "file not found";
        case hipErrorSharedObjectSymbolNotFound: return "shared object symbol not found";
        case hipErrorSharedObjectInitFailed: return "shared object initialization failed";
        case hipErrorOperatingSystem: return "OS call failed or operation not supported on this OS";
        case hipErrorInvalidHandle: return "invalid resource handle";
        case hipErrorIllegalState: return "the operation cannot be performed in the present state";
        case hipErrorNotFound: return "named symbol not found";
        case hipErrorNotReady: return "device not ready";
        case hipErrorIllegalAddress: return "an illegal memory access was encountered";
        case hipErrorLaunchOutOfResources: return "too many resources requested for launch";
        case hipErrorLaunchTimeOut: return "the launch timed out and was terminated";
        case hipErrorPeerAccessAlreadyEnabled: return "peer access is already enabled";
        case hipErrorPeerAccessNotEnabled: return "peer access has not been enabled";
        case hipErrorSetOnActiveProcess: return "cannot set while device is active in this process";
        case hipErrorContextIsDestroyed: return "context is destroyed";
        case hipErrorAssert: return "device-side assert triggered";
        case hipErrorHostMemoryAlreadyRegistered: return "part or all of the requested memory range is already mapped";
        case hipErrorHostMemoryNotRegistered: return "pointer does not correspond to a registered memory region";
        case hipErrorLaunchFailure: return "unspecified launch failure";
        case hipErrorCooperativeLaunchTooLarge: return "too many blocks in cooperative launch";
        case hipErrorNotSupported: return "operation not supported";
        case hipErrorStreamCaptureUnsupported: return "operation not permitted when stream is capturing";
        case hipErrorStreamCaptureInvalidated: return "operation failed due to a previous error during capture";
        case hipErrorStreamCaptureMerge: return "operation would result in a merge of separate capture sequences";
        case hipErrorStreamCaptureUnmatched: return "capture was not ended in the same stream as it began";
        case hipErrorStreamCaptureUnjoined: return "capturing stream has unjoined work";
        case hipErrorStreamCaptureIsolation: return "dependency created on uncaptured work in another stream";
        case hipErrorStreamCaptureImplicit: return "operation would make the legacy stream depend on a capturing blocking stream";
        case hipErrorCapturedEvent: return "operation not permitted on an event last recorded in a capturing stream";
        case hipErrorStreamCaptureWrongThread: return "attempt to terminate a thread-local capture sequence from another thread";
        case hipErrorGraphExecUpdateFailure: return "the graph update was not performed because it included changes which violated constraints specific to instantiated graph update";
        case hipErrorInvalidChannelDescriptor: return "invalid channel descriptor";
        case hipErrorInvalidTexture: return "invalid texture";
        case hipErrorRuntimeMemory: return "runtime memory call returned error";
        case hipErrorRuntimeOther: return "runtime call other than memory returned error";
        case hipErrorUnknown:
        default: return "unknown error";
    }
}

/* ============================================================================
 * Additional Device/Host Memory Stubs
 * ============================================================================ */

hipError_t hipHostMalloc(void** ptr, size_t size, unsigned int flags) {
    if (flags & ~0u) {
        hip_remote_log_debug("hipHostMalloc: flags=0x%x (flags not forwarded to worker)", flags);
    }
    return hipMallocHost(ptr, size);
}

/* hipHostFree is the modern name for hipFreeHost */
hipError_t hipHostFree(void* ptr) {
    return hipFreeHost(ptr);
}

hipError_t hipHostAlloc(void** ptr, size_t size, unsigned int flags) {
    return hipHostMalloc(ptr, size, flags);
}

hipError_t hipHostGetDevicePointer(void** devPtr, void* hostPtr, unsigned int flags) {
    if (!devPtr) return hipErrorInvalidValue;
    *devPtr = hostPtr;
    return hipSuccess;
}

hipError_t hipHostRegister(void* hostPtr, size_t sizeBytes, unsigned int flags) {
    hip_remote_log_debug("hipHostRegister: ptr=%p size=%zu flags=0x%x (not supported remotely)",
                         hostPtr, sizeBytes, flags);
    return hipSuccess;
}

hipError_t hipHostUnregister(void* hostPtr) {
    hip_remote_log_debug("hipHostUnregister: ptr=%p (not supported remotely)", hostPtr);
    return hipSuccess;
}

hipError_t hipDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority) {
    if (leastPriority) *leastPriority = 0;
    if (greatestPriority) *greatestPriority = 0;
    return hipSuccess;
}

hipError_t hipDeviceGetDefaultMemPool(hipMemPool_t* memPool, int device) {
    hip_remote_log_debug("hipDeviceGetDefaultMemPool: device=%d (not supported remotely)", device);
    if (memPool) *memPool = NULL;
    return hipSuccess;
}

/* hipDeviceProp_t comes from <hip/hip_runtime_api.h> via hip_remote_client.h.
 * The following compat struct is kept for reference but not compiled. */
#if 0
typedef struct {
    char name[256];                   /* 0 */
    char uuid[16];                    /* 256 */
    char luid[8];                     /* 272 */
    unsigned int luidDeviceNodeMask;  /* 280 */
    size_t totalGlobalMem;            /* 288 */
    size_t sharedMemPerBlock;         /* 296 */
    int regsPerBlock;                 /* 304 */
    int warpSize;                     /* 308 */
    size_t memPitch;                  /* 312 */
    int maxThreadsPerBlock;           /* 320 */
    int maxThreadsDim[3];             /* 324 */
    int maxGridSize[3];               /* 336 */
    int clockRate;                    /* 348 */
    size_t totalConstMem;             /* 352 */
    int major;                        /* 360 */
    int minor;                        /* 364 */
    size_t textureAlignment;          /* 368 */
    size_t texturePitchAlignment;     /* 376 */
    int deviceOverlap;                /* 384 */
    int multiProcessorCount;          /* 388 */
    int kernelExecTimeoutEnabled;     /* 392 */
    int integrated;                   /* 396 */
    int canMapHostMemory;             /* 400 */
    int computeMode;                  /* 404 */
    int maxTexture1D;                 /* 408 */
    int maxTexture1DMipmap;           /* 412 */
    int maxTexture1DLinear;           /* 416 */
    int maxTexture2D[2];              /* 420 */
    int maxTexture2DMipmap[2];        /* 428 */
    int maxTexture2DLinear[3];        /* 436 */
    int maxTexture2DGather[2];        /* 448 */
    int maxTexture3D[3];              /* 456 */
    int maxTexture3DAlt[3];           /* 468 */
    int maxTextureCubemap;            /* 480 */
    int maxTexture1DLayered[2];       /* 484 */
    int maxTexture2DLayered[3];       /* 492 */
    int maxTextureCubemapLayered[2];  /* 504 */
    int maxSurface1D;                 /* 512 */
    int maxSurface2D[2];              /* 516 */
    int maxSurface3D[3];              /* 524 */
    int maxSurface1DLayered[2];       /* 536 */
    int maxSurface2DLayered[3];       /* 544 */
    int maxSurfaceCubemap;            /* 556 */
    int maxSurfaceCubemapLayered[2];  /* 560 */
    size_t surfaceAlignment;          /* 568 */
    int concurrentKernels;            /* 576 */
    int ECCEnabled;                   /* 580 */
    int pciBusID;                     /* 584 */
    int pciDeviceID;                  /* 588 */
    int pciDomainID;                  /* 592 */
    int tccDriver;                    /* 596 */
    int asyncEngineCount;             /* 600 */
    int unifiedAddressing;            /* 604 */
    int memoryClockRate;              /* 608 */
    int memoryBusWidth;               /* 612 */
    int l2CacheSize;                  /* 616 */
    int persistingL2CacheMaxSize;     /* 620 */
    int maxThreadsPerMultiProcessor;  /* 624 */
    int streamPrioritiesSupported;    /* 628 */
    int globalL1CacheSupported;       /* 632 */
    int localL1CacheSupported;        /* 636 */
    size_t sharedMemPerMultiprocessor;/* 640 */
    int regsPerMultiprocessor;        /* 648 */
    int managedMemory;                /* 652 */
    int isMultiGpuBoard;              /* 656 */
    int multiGpuBoardGroupID;         /* 660 */
    int hostNativeAtomicSupported;    /* 664 */
    int singleToDoublePrecisionPerfRatio; /* 668 */
    int pageableMemoryAccess;         /* 672 */
    int concurrentManagedAccess;      /* 676 */
    int computePreemptionSupported;   /* 680 */
    int canUseHostPointerForRegisteredMem; /* 684 */
    int cooperativeLaunch;            /* 688 */
    int cooperativeMultiDeviceLaunch; /* 692 */
    size_t sharedMemPerBlockOptin;    /* 696 */
    int pageableMemoryAccessUsesHostPageTables; /* 704 */
    int directManagedMemAccessFromHost; /* 708 */
    int maxBlocksPerMultiProcessor;   /* 712 */
    int accessPolicyMaxWindowSize;    /* 716 */
    size_t reservedSharedMemPerBlock;  /* 720 */
    int hostRegisterSupported;        /* 728 */
    int sparseHipArraySupported;      /* 732 */
    int hostRegisterReadOnlySupported;/* 736 */
    int timelineSemaphoreInteropSupported; /* 740 */
    int memoryPoolsSupported;         /* 744 */
    int gpuDirectRDMASupported;       /* 748 */
    unsigned int gpuDirectRDMAFlushWritesOptions; /* 752 */
    int gpuDirectRDMAWritesOrdering;  /* 756 */
    unsigned int memoryPoolSupportedHandleTypes; /* 760 */
    int deferredMappingHipArraySupported; /* 764 */
    int ipcEventSupported;            /* 768 */
    int clusterLaunch;                /* 772 */
    int unifiedFunctionPointers;      /* 776 */
    int reserved[63];                 /* 780 - padding */
    int hipReserved[32];              /* 1032 */
    char gcnArchName[256];            /* 1160 - arch name */
} hipDeviceProp_tR0600_Compat_;
#endif

/* The worker sends the raw hipDeviceProp_t struct preceded by a response
 * header.  Since both sides include the same hip_runtime_api.h, the struct
 * layout is identical -- no manual field-by-field copy needed. */

typedef struct {
    HipRemoteResponseHeader header;
    hipDeviceProp_t props;
} DevicePropsFullResponse;

static hipDeviceProp_t g_cached_full_props[MAX_CACHED_DEVICES];

hipError_t hipGetDevicePropertiesR0600(hipDeviceProp_t* prop, int deviceId) {
    if (!prop) return hipErrorInvalidValue;

    if (deviceId >= 0 && deviceId < MAX_CACHED_DEVICES && g_props_cached[deviceId]) {
        memcpy(prop, &g_cached_full_props[deviceId], sizeof(hipDeviceProp_t));
        return hipSuccess;
    }

    HipRemoteDeviceRequest req;
    memset(&req, 0, sizeof(req));
    req.device_id = deviceId;

    DevicePropsFullResponse resp;
    memset(&resp, 0, sizeof(resp));

    hipError_t err = hip_remote_request(
        HIP_OP_GET_DEVICE_PROPERTIES,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );
    if (err != hipSuccess) return err;

    memcpy(prop, &resp.props, sizeof(hipDeviceProp_t));

    if (deviceId >= 0 && deviceId < MAX_CACHED_DEVICES) {
        memcpy(&g_cached_full_props[deviceId], prop, sizeof(hipDeviceProp_t));
        g_props_cached[deviceId] = 1;
    }

    return hipSuccess;
}

hipError_t hipDeviceGetGcnArchName(char* buf, int deviceId) {
    if (!buf) return hipErrorInvalidValue;

    hipDeviceProp_t prop;
    hipError_t err = hipGetDevicePropertiesR0600(&prop, deviceId);
    if (err != hipSuccess) return err;
    strncpy(buf, prop.gcnArchName, 255);
    buf[255] = '\0';
    return hipSuccess;
}

hipError_t hipDeviceTotalMem(size_t* bytes, int device) {
    hip_remote_log_debug("hipDeviceTotalMem: device=%d (querying current device)", device);
    if (bytes) *bytes = 0;
    size_t free_bytes = 0, total_bytes = 0;
    hipError_t err = hipMemGetInfo(&free_bytes, &total_bytes);
    if (err == hipSuccess && bytes) *bytes = total_bytes;
    return err;
}

hipError_t hipInit(unsigned int flags) {
    hip_remote_log_debug("hipInit: flags=0x%x", flags);
    return hipSuccess;
}

hipError_t hipExtGetLastError(void) {
    return hipGetLastError();
}

hipError_t hipMemPtrGetInfo(void* ptr, size_t* size) {
    HipRemoteMemPtrGetInfoRequest req = {
        .ptr = (uint64_t)(uintptr_t)ptr
    };
    HipRemoteMemPtrGetInfoResponse resp;

    hipError_t err = hip_remote_request(
        HIP_OP_MEM_PTR_GET_INFO,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );
    if (err == hipSuccess && size) {
        *size = (size_t)resp.size;
    } else if (size) {
        *size = 0;
    }
    return err;
}

hipError_t hipStreamGetDevice(hipStream_t stream, hipDevice_t* device) {
    hip_remote_log_debug("hipStreamGetDevice: stream=%p (returning device 0)", stream);
    if (device) *device = 0;
    return hipSuccess;
}

hipError_t hipCtxGetCurrent(hipCtx_t* ctx) {
    if (ctx) *ctx = (void*)(uintptr_t)1;
    return hipSuccess;
}

hipError_t hipDevicePrimaryCtxGetState(int device, unsigned int* flags, int* active) {
    hip_remote_log_debug("hipDevicePrimaryCtxGetState: device=%d", device);
    if (flags) *flags = 0;
    if (active) *active = 1;
    return hipSuccess;
}

hipError_t hipDrvGetErrorString(hipError_t hipError, const char** errorString) {
    const char* s = hipGetErrorString(hipError);
    if (errorString) *errorString = s;
    return hipSuccess;
}

hipError_t hipFuncSetAttribute(const void* func, hipFuncAttribute attr, int value) {
    hip_remote_log_debug("hipFuncSetAttribute: func=%p attr=%d value=%d (not forwarded)", func, attr, value);
    return hipSuccess;
}

hipError_t hipLaunchHostFunc(hipStream_t stream, void (*fn)(void*), void* userData) {
    hip_remote_log_debug("hipLaunchHostFunc: stream=%p (synchronizing then calling locally)", (void*)stream);
    if (stream) {
        hipStreamSynchronize(stream);
    } else {
        hipDeviceSynchronize();
    }
    if (fn) fn(userData);
    return hipSuccess;
}

hipError_t hipMemcpyFromSymbol(void* dst, const void* symbol, size_t count, size_t offset, hipMemcpyKind kind) {
    (void)dst; (void)symbol; (void)count; (void)offset; (void)kind;
    return hipErrorNotSupported;
}

hipError_t hipMemAdvise(const void* devPtr, size_t count, hipMemoryAdvise advice, int device) {
    hip_remote_log_debug("hipMemAdvise: ptr=%p count=%zu advice=%d device=%d (hint, not forwarded)",
                         devPtr, count, advice, device);
    return hipSuccess;
}

hipError_t hipCtxSetCurrent(hipCtx_t ctx) {
    hip_remote_log_debug("hipCtxSetCurrent: ctx=%p (single-context remote mode)", ctx);
    return hipSuccess;
}

hipError_t hipDeviceGet(int* device, int ordinal) {
    if (device) *device = ordinal;
    return hipSuccess;
}

hipError_t hipDevicePrimaryCtxRetain(hipCtx_t* pctx, hipDevice_t device) {
    hip_remote_log_debug("hipDevicePrimaryCtxRetain: device=%d", device);
    if (pctx) *pctx = (void*)(uintptr_t)1;
    return hipSuccess;
}

hipError_t hipFuncGetAttribute(int* value, hipFunction_attribute attrib, hipFunction_t hfunc) {
    hip_remote_log_debug("hipFuncGetAttribute: attrib=%d func=%p", attrib, hfunc);
    if (!value) return hipErrorInvalidValue;

    /* Return sensible defaults for common attributes that runtimes
     * like Triton query to validate kernel launch parameters. */
    switch (attrib) {
        case 0:  *value = 1024; break; /* MAX_THREADS_PER_BLOCK */
        case 1:  *value = 65536; break; /* SHARED_SIZE_BYTES */
        case 2:  *value = 0; break;    /* CONST_SIZE_BYTES */
        case 3:  *value = 0; break;    /* LOCAL_SIZE_BYTES */
        case 4:  *value = 64; break;   /* NUM_REGS */
        case 5:  *value = 0; break;    /* PTX_VERSION */
        case 6:  *value = 0; break;    /* BINARY_VERSION */
        case 7:  *value = 0; break;    /* CACHE_MODE_CA */
        case 8:  *value = 65536; break; /* MAX_DYNAMIC_SHARED_SIZE_BYTES */
        case 9:  *value = 0; break;    /* PREFERRED_SHARED_MEMORY_CARVEOUT */
        default: *value = 0; break;
    }
    return hipSuccess;
}

hipError_t hipFuncSetCacheConfig(const void* func, hipFuncCache_t cacheConfig) {
    hip_remote_log_debug("hipFuncSetCacheConfig: func=%p config=%d (hint, not forwarded)", func, cacheConfig);
    return hipSuccess;
}

hipError_t hipPointerGetAttribute(void* data, hipPointer_attribute attribute, hipDeviceptr_t ptr) {
    if (!data || !ptr) return hipErrorInvalidValue;

    /* Only claim the pointer is a device pointer if it's in the vaddr range
     * (allocated by our hipMalloc). CPU pointers are outside this range and
     * should return hipErrorInvalidValue so callers (e.g. Triton) can detect
     * them and raise appropriate errors. */
    int is_device_ptr = ((uintptr_t)ptr >= 0x7F0000000000ULL);

    switch (attribute) {
        case 3: /* hipPointerAttributeDevicePointer */
            if (!is_device_ptr) return hipErrorInvalidValue;
            *(void**)data = ptr;
            return hipSuccess;
        case 2: /* hipPointerAttributeMemoryType */
            if (!is_device_ptr) return hipErrorInvalidValue;
            *(unsigned int*)data = 2; /* hipMemoryTypeDevice */
            return hipSuccess;
        case 4: /* hipPointerAttributeHostPointer */
            *(void**)data = is_device_ptr ? NULL : ptr;
            return hipSuccess;
    }

    HipRemotePointerGetAttributeRequest req = {
        .ptr = (uint64_t)(uintptr_t)ptr,
        .attribute = attribute
    };
    HipRemotePointerGetAttributeResponse resp;

    hipError_t err = hip_remote_request(
        HIP_OP_POINTER_GET_ATTRIBUTE,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );
    if (err == hipSuccess) {
        memcpy(data, &resp.data, sizeof(uint64_t));
    }
    return err;
}

hipError_t hipExtStreamGetCUMask(hipStream_t stream, uint32_t cuMaskSize, uint32_t* cuMask) {
    hip_remote_log_debug("hipExtStreamGetCUMask: stream=%p size=%u (returning all-ones mask)", (void*)stream, cuMaskSize);
    if (cuMask && cuMaskSize > 0) memset(cuMask, 0xFF, cuMaskSize * sizeof(uint32_t));
    return hipSuccess;
}

hipError_t hipSetValidDevices(int* device_arr, int len) {
    if (!device_arr || len <= 0) return hipErrorInvalidValue;
    return hipSuccess;
}

hipError_t hipChooseDevice(int* device, const hipDeviceProp_t* prop) {
    (void)prop;
    if (!device) return hipErrorInvalidValue;
    int device_count = 0;
    hipError_t err = hipGetDeviceCount(&device_count);
    if (err != hipSuccess) return err;
    if (device_count == 0) return hipErrorNoDevice;
    *device = 0;
    return hipSuccess;
}

hipError_t hipFuncGetAttributes(struct hipFuncAttributes* attr, const void* func) {
    if (attr) memset(attr, 0, 56);
    (void)func;
    return hipSuccess;
}
