/*
 * Copyright 2025 Advanced Micro Devices, Inc.
 *
 * HIP Context API implementations for remote client
 *
 * Note: Context APIs are deprecated on AMD platforms but maintained
 * for CUDA compatibility. Most operations are implemented as pass-through
 * to the remote worker which uses HIP's implicit context model.
 */

#include "hip_remote/hip_remote_internal.h"
#include "hip_remote/hip_remote_protocol.h"
#include <string.h>

/* Declare the request function */
extern hipError_t hip_remote_request(HipRemoteOpCode opcode,
                                      const void* request_data, size_t request_size,
                                      void* response_data, size_t response_size);

/* ============================================================================
 * Context Creation & Destruction
 * ============================================================================ */

hipError_t hipCtxCreate(hipCtx_t* ctx, unsigned int flags, hipDevice_t device) {
    if (!ctx) return hipErrorInvalidValue;

    HipRemoteCtxCreateRequest req = {
        .flags = flags,
        .device = (uint32_t)device
    };

    HipRemoteCtxCreateResponse resp;
    hipError_t err = hip_remote_request(HIP_OP_CTX_CREATE, &req, sizeof(req), &resp, sizeof(resp));
    if (err == hipSuccess) {
        *ctx = (hipCtx_t)(uintptr_t)resp.ctx;
    }
    return err;
}

hipError_t hipCtxDestroy(hipCtx_t ctx) {
    HipRemoteCtxDestroyRequest req = {
        .ctx = (uint64_t)(uintptr_t)ctx
    };

    HipRemoteResponseHeader resp;
    return hip_remote_request(HIP_OP_CTX_DESTROY, &req, sizeof(req), &resp, sizeof(resp));
}

/* ============================================================================
 * Context Management
 * ============================================================================ */

hipError_t hipCtxSetCurrent(hipCtx_t ctx) {
    HipRemoteCtxSetCurrentRequest req = {
        .ctx = (uint64_t)(uintptr_t)ctx
    };

    HipRemoteResponseHeader resp;
    return hip_remote_request(HIP_OP_CTX_SET_CURRENT, &req, sizeof(req), &resp, sizeof(resp));
}

hipError_t hipCtxGetCurrent(hipCtx_t* ctx) {
    if (!ctx) return hipErrorInvalidValue;

    HipRemoteCtxGetCurrentResponse resp;
    hipError_t err = hip_remote_request(HIP_OP_CTX_GET_CURRENT, NULL, 0, &resp, sizeof(resp));
    if (err == hipSuccess) {
        *ctx = (hipCtx_t)(uintptr_t)resp.ctx;
    }
    return err;
}

hipError_t hipCtxPushCurrent(hipCtx_t ctx) {
    HipRemoteCtxPushCurrentRequest req = {
        .ctx = (uint64_t)(uintptr_t)ctx
    };

    HipRemoteResponseHeader resp;
    return hip_remote_request(HIP_OP_CTX_PUSH_CURRENT, &req, sizeof(req), &resp, sizeof(resp));
}

hipError_t hipCtxPopCurrent(hipCtx_t* ctx) {
    if (!ctx) return hipErrorInvalidValue;

    HipRemoteCtxPopCurrentResponse resp;
    hipError_t err = hip_remote_request(HIP_OP_CTX_POP_CURRENT, NULL, 0, &resp, sizeof(resp));
    if (err == hipSuccess) {
        *ctx = (hipCtx_t)(uintptr_t)resp.ctx;
    }
    return err;
}

hipError_t hipCtxSynchronize(void) {
    HipRemoteResponseHeader resp;
    return hip_remote_request(HIP_OP_CTX_SYNCHRONIZE, NULL, 0, &resp, sizeof(resp));
}

/* ============================================================================
 * Device Association
 * ============================================================================ */

hipError_t hipCtxGetDevice(hipDevice_t* device) {
    if (!device) return hipErrorInvalidValue;

    HipRemoteCtxGetDeviceResponse resp;
    hipError_t err = hip_remote_request(HIP_OP_CTX_GET_DEVICE, NULL, 0, &resp, sizeof(resp));
    if (err == hipSuccess) {
        *device = (hipDevice_t)resp.device;
    }
    return err;
}

/* ============================================================================
 * Configuration & State
 * ============================================================================ */

hipError_t hipCtxGetFlags(unsigned int* flags) {
    if (!flags) return hipErrorInvalidValue;

    HipRemoteCtxGetFlagsResponse resp;
    hipError_t err = hip_remote_request(HIP_OP_CTX_GET_FLAGS, NULL, 0, &resp, sizeof(resp));
    if (err == hipSuccess) {
        *flags = resp.flags;
    }
    return err;
}

hipError_t hipCtxGetApiVersion(hipCtx_t ctx, int* apiVersion) {
    if (!apiVersion) return hipErrorInvalidValue;

    HipRemoteCtxGetApiVersionRequest req = {
        .ctx = (uint64_t)(uintptr_t)ctx
    };

    HipRemoteCtxGetApiVersionResponse resp;
    hipError_t err = hip_remote_request(HIP_OP_CTX_GET_API_VERSION, &req, sizeof(req), &resp, sizeof(resp));
    if (err == hipSuccess) {
        *apiVersion = (int)resp.version;
    }
    return err;
}

hipError_t hipCtxGetCacheConfig(hipFuncCache_t* cacheConfig) {
    if (!cacheConfig) return hipErrorInvalidValue;

    HipRemoteCtxGetCacheConfigResponse resp;
    hipError_t err = hip_remote_request(HIP_OP_CTX_GET_CACHE_CONFIG, NULL, 0, &resp, sizeof(resp));
    if (err == hipSuccess) {
        *cacheConfig = (hipFuncCache_t)resp.config;
    }
    return err;
}

hipError_t hipCtxSetCacheConfig(hipFuncCache_t cacheConfig) {
    HipRemoteCtxSetCacheConfigRequest req = {
        .config = (int32_t)cacheConfig,
        .reserved = 0
    };

    HipRemoteResponseHeader resp;
    return hip_remote_request(HIP_OP_CTX_SET_CACHE_CONFIG, &req, sizeof(req), &resp, sizeof(resp));
}

hipError_t hipCtxGetSharedMemConfig(hipSharedMemConfig* pConfig) {
    if (!pConfig) return hipErrorInvalidValue;

    HipRemoteCtxGetSharedMemConfigResponse resp;
    hipError_t err = hip_remote_request(HIP_OP_CTX_GET_SHARED_MEM_CONFIG, NULL, 0, &resp, sizeof(resp));
    if (err == hipSuccess) {
        *pConfig = (hipSharedMemConfig)resp.config;
    }
    return err;
}

hipError_t hipCtxSetSharedMemConfig(hipSharedMemConfig config) {
    HipRemoteCtxSetSharedMemConfigRequest req = {
        .config = (int32_t)config,
        .reserved = 0
    };

    HipRemoteResponseHeader resp;
    return hip_remote_request(HIP_OP_CTX_SET_SHARED_MEM_CONFIG, &req, sizeof(req), &resp, sizeof(resp));
}

/* ============================================================================
 * Peer Access
 * ============================================================================ */

hipError_t hipCtxEnablePeerAccess(hipCtx_t peerCtx, unsigned int flags) {
    HipRemoteCtxEnablePeerAccessRequest req = {
        .peer_ctx = (uint64_t)(uintptr_t)peerCtx,
        .flags = flags,
        .reserved = 0
    };

    HipRemoteResponseHeader resp;
    return hip_remote_request(HIP_OP_CTX_ENABLE_PEER_ACCESS, &req, sizeof(req), &resp, sizeof(resp));
}

hipError_t hipCtxDisablePeerAccess(hipCtx_t peerCtx) {
    HipRemoteCtxDisablePeerAccessRequest req = {
        .peer_ctx = (uint64_t)(uintptr_t)peerCtx
    };

    HipRemoteResponseHeader resp;
    return hip_remote_request(HIP_OP_CTX_DISABLE_PEER_ACCESS, &req, sizeof(req), &resp, sizeof(resp));
}

/* ============================================================================
 * Primary Context Functions
 * ============================================================================ */

hipError_t hipDevicePrimaryCtxGetState(hipDevice_t dev, unsigned int* flags, int* active) {
    if (!flags || !active) return hipErrorInvalidValue;

    HipRemoteDevicePrimaryCtxGetStateRequest req = {
        .device = (int32_t)dev,
        .reserved = 0
    };

    HipRemoteDevicePrimaryCtxGetStateResponse resp;
    hipError_t err = hip_remote_request(HIP_OP_DEVICE_PRIMARY_CTX_GET_STATE, &req, sizeof(req), &resp, sizeof(resp));
    if (err == hipSuccess) {
        *flags = resp.flags;
        *active = resp.active;
    }
    return err;
}

hipError_t hipDevicePrimaryCtxRetain(hipCtx_t* pctx, hipDevice_t dev) {
    if (!pctx) return hipErrorInvalidValue;

    HipRemoteDevicePrimaryCtxRetainRequest req = {
        .device = (int32_t)dev,
        .reserved = 0
    };

    HipRemoteDevicePrimaryCtxRetainResponse resp;
    hipError_t err = hip_remote_request(HIP_OP_DEVICE_PRIMARY_CTX_RETAIN, &req, sizeof(req), &resp, sizeof(resp));
    if (err == hipSuccess) {
        *pctx = (hipCtx_t)(uintptr_t)resp.ctx;
    }
    return err;
}

hipError_t hipDevicePrimaryCtxRelease(hipDevice_t dev) {
    HipRemoteDevicePrimaryCtxReleaseRequest req = {
        .device = (int32_t)dev,
        .reserved = 0
    };

    HipRemoteResponseHeader resp;
    return hip_remote_request(HIP_OP_DEVICE_PRIMARY_CTX_RELEASE, &req, sizeof(req), &resp, sizeof(resp));
}

hipError_t hipDevicePrimaryCtxReset(hipDevice_t dev) {
    HipRemoteDevicePrimaryCtxResetRequest req = {
        .device = (int32_t)dev,
        .reserved = 0
    };

    HipRemoteResponseHeader resp;
    return hip_remote_request(HIP_OP_DEVICE_PRIMARY_CTX_RESET, &req, sizeof(req), &resp, sizeof(resp));
}

hipError_t hipDevicePrimaryCtxSetFlags(hipDevice_t dev, unsigned int flags) {
    HipRemoteDevicePrimaryCtxSetFlagsRequest req = {
        .device = (int32_t)dev,
        .flags = flags
    };

    HipRemoteResponseHeader resp;
    return hip_remote_request(HIP_OP_DEVICE_PRIMARY_CTX_SET_FLAGS, &req, sizeof(req), &resp, sizeof(resp));
}
