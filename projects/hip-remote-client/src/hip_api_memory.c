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
 * @file hip_api_memory.c
 * @brief Memory management API implementations for remote HIP
 */

#include "hip_remote/hip_remote_client.h"
#include "hip_remote/hip_remote_protocol.h"

#include <stdlib.h>
#include <string.h>

/* ============================================================================
 * Memory Allocation
 * ============================================================================ */

hipError_t hipMalloc(void** ptr, size_t size) {
    if (!ptr) {
        return hipErrorInvalidValue;
    }
    if (size == 0) {
        *ptr = NULL;
        return hipSuccess;
    }

    HipRemoteMallocRequest req = {
        .size = size,
        .flags = 0
    };
    HipRemoteMallocResponse resp;

    hipError_t err = hip_remote_request(
        HIP_OP_MALLOC,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );

    if (err == hipSuccess) {
        /* Store the remote pointer as an opaque handle */
        *ptr = (void*)(uintptr_t)resp.device_ptr;
    } else {
        *ptr = NULL;
    }
    return err;
}

hipError_t hipFree(void* ptr) {
    if (!ptr) {
        return hipSuccess;  /* NULL free is a no-op */
    }

    HipRemoteFreeRequest req = {
        .device_ptr = (uint64_t)(uintptr_t)ptr
    };
    HipRemoteResponseHeader resp;

    return hip_remote_request(
        HIP_OP_FREE,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );
}

hipError_t hipMallocHost(void** ptr, size_t size) {
    if (!ptr) {
        return hipErrorInvalidValue;
    }
    if (size == 0) {
        *ptr = NULL;
        return hipSuccess;
    }

    /* For host memory, we allocate locally but register with remote */
    /* This allows zero-copy transfers when possible */
    *ptr = malloc(size);
    if (!*ptr) {
        return hipErrorOutOfMemory;
    }

    /* Optionally notify remote about pinned memory (for optimization) */
    HipRemoteMallocRequest req = {
        .size = size,
        .flags = 0
    };
    HipRemoteMallocResponse resp;

    hipError_t err = hip_remote_request(
        HIP_OP_MALLOC_HOST,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );

    /* If remote registration failed, still return the local allocation */
    if (err != hipSuccess) {
        hip_remote_log_debug("Host malloc registration failed, using local only");
    }

    return hipSuccess;
}

hipError_t hipFreeHost(void* ptr) {
    if (!ptr) {
        return hipSuccess;
    }

    /* Notify remote (best effort) */
    HipRemoteFreeRequest req = {
        .device_ptr = (uint64_t)(uintptr_t)ptr
    };
    HipRemoteResponseHeader resp;
    (void)hip_remote_request(
        HIP_OP_FREE_HOST,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );

    /* Free local memory */
    free(ptr);
    return hipSuccess;
}

hipError_t hipMallocManaged(void** ptr, size_t size, unsigned int flags) {
    if (!ptr) {
        return hipErrorInvalidValue;
    }
    if (size == 0) {
        *ptr = NULL;
        return hipSuccess;
    }

    HipRemoteMallocRequest req = {
        .size = size,
        .flags = flags
    };
    HipRemoteMallocResponse resp;

    hipError_t err = hip_remote_request(
        HIP_OP_MALLOC_MANAGED,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );

    if (err == hipSuccess) {
        *ptr = (void*)(uintptr_t)resp.device_ptr;
    } else {
        *ptr = NULL;
    }
    return err;
}

hipError_t hipMallocAsync(void** ptr, size_t size, void* stream) {
    if (!ptr) {
        return hipErrorInvalidValue;
    }
    if (size == 0) {
        *ptr = NULL;
        return hipSuccess;
    }

    HipRemoteMallocAsyncRequest req = {
        .size = size,
        .stream = (uint64_t)(uintptr_t)stream
    };
    HipRemoteMallocResponse resp;

    hipError_t err = hip_remote_request(
        HIP_OP_MALLOC_ASYNC,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );

    if (err == hipSuccess) {
        *ptr = (void*)(uintptr_t)resp.device_ptr;
    } else {
        *ptr = NULL;
    }
    return err;
}

hipError_t hipFreeAsync(void* ptr, void* stream) {
    if (!ptr) {
        return hipSuccess;  /* NULL free is a no-op */
    }

    HipRemoteFreeAsyncRequest req = {
        .device_ptr = (uint64_t)(uintptr_t)ptr,
        .stream = (uint64_t)(uintptr_t)stream
    };
    HipRemoteResponseHeader resp;

    return hip_remote_request(
        HIP_OP_FREE_ASYNC,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );
}

/* ============================================================================
 * Memory Copy
 * ============================================================================ */

hipError_t hipMemcpy(void* dst, const void* src, size_t size, hipMemcpyKind kind) {
    if (size == 0) {
        return hipSuccess;
    }
    if (!dst || !src) {
        return hipErrorInvalidValue;
    }

    HipRemoteMemcpyRequest req = {
        .dst = (uint64_t)(uintptr_t)dst,
        .src = (uint64_t)(uintptr_t)src,
        .size = size,
        .kind = (int32_t)kind,
        .stream = 0  /* Default stream */
    };

    switch (kind) {
        case hipMemcpyHostToDevice: {
            /* Send data to remote device */
            HipRemoteMemcpyResponse resp;
            return hip_remote_request_with_data(
                HIP_OP_MEMCPY,
                &req, sizeof(req),
                src, size,
                &resp, sizeof(resp)
            );
        }

        case hipMemcpyDeviceToHost: {
            /* Receive data from remote device */
            HipRemoteMemcpyResponse resp;
            return hip_remote_request_receive_data(
                HIP_OP_MEMCPY,
                &req, sizeof(req),
                &resp, sizeof(resp),
                dst, size
            );
        }

        case hipMemcpyDeviceToDevice: {
            /* Remote-to-remote copy, no data transfer */
            HipRemoteMemcpyResponse resp;
            return hip_remote_request(
                HIP_OP_MEMCPY,
                &req, sizeof(req),
                &resp, sizeof(resp)
            );
        }

        case hipMemcpyHostToHost: {
            /* Local copy */
            memmove(dst, src, size);
            return hipSuccess;
        }

        case hipMemcpyDefault: {
            /* Let remote figure out the direction */
            /* For now, assume D2D */
            HipRemoteMemcpyResponse resp;
            return hip_remote_request(
                HIP_OP_MEMCPY,
                &req, sizeof(req),
                &resp, sizeof(resp)
            );
        }

        default:
            return hipErrorInvalidValue;
    }
}

hipError_t hipMemcpyAsync(void* dst, const void* src, size_t size,
                          hipMemcpyKind kind, void* stream) {
    if (size == 0) {
        return hipSuccess;
    }
    if (!dst || !src) {
        return hipErrorInvalidValue;
    }

    HipRemoteMemcpyRequest req = {
        .dst = (uint64_t)(uintptr_t)dst,
        .src = (uint64_t)(uintptr_t)src,
        .size = size,
        .kind = (int32_t)kind,
        .stream = (uint64_t)(uintptr_t)stream
    };

    /* For async, we still block on the network but the GPU operation is async */
    switch (kind) {
        case hipMemcpyHostToDevice: {
            HipRemoteMemcpyResponse resp;
            return hip_remote_request_with_data(
                HIP_OP_MEMCPY_ASYNC,
                &req, sizeof(req),
                src, size,
                &resp, sizeof(resp)
            );
        }

        case hipMemcpyDeviceToHost: {
            HipRemoteMemcpyResponse resp;
            return hip_remote_request_receive_data(
                HIP_OP_MEMCPY_ASYNC,
                &req, sizeof(req),
                &resp, sizeof(resp),
                dst, size
            );
        }

        case hipMemcpyDeviceToDevice: {
            HipRemoteMemcpyResponse resp;
            return hip_remote_request(
                HIP_OP_MEMCPY_ASYNC,
                &req, sizeof(req),
                &resp, sizeof(resp)
            );
        }

        case hipMemcpyHostToHost: {
            memmove(dst, src, size);
            return hipSuccess;
        }

        default: {
            HipRemoteMemcpyResponse resp;
            return hip_remote_request(
                HIP_OP_MEMCPY_ASYNC,
                &req, sizeof(req),
                &resp, sizeof(resp)
            );
        }
    }
}

/* Convenience functions */
hipError_t hipMemcpyHtoD(void* dst, const void* src, size_t size) {
    return hipMemcpy(dst, src, size, hipMemcpyHostToDevice);
}

hipError_t hipMemcpyDtoH(void* dst, const void* src, size_t size) {
    return hipMemcpy(dst, src, size, hipMemcpyDeviceToHost);
}

hipError_t hipMemcpyDtoD(void* dst, const void* src, size_t size) {
    return hipMemcpy(dst, src, size, hipMemcpyDeviceToDevice);
}

/* ============================================================================
 * 2D Memory Copy
 * ============================================================================ */

hipError_t hipMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch,
                       size_t width, size_t height, hipMemcpyKind kind) {
    if (width == 0 || height == 0) {
        return hipSuccess;
    }
    if (!dst || !src) {
        return hipErrorInvalidValue;
    }

    HipRemoteMemcpy2DRequest req = {
        .dst = (uint64_t)(uintptr_t)dst,
        .dpitch = dpitch,
        .src = (uint64_t)(uintptr_t)src,
        .spitch = spitch,
        .width = width,
        .height = height,
        .kind = (int32_t)kind,
        .reserved = 0,
        .stream = 0
    };

    switch (kind) {
        case hipMemcpyHostToDevice: {
            /* For H2D 2D copy, we need to send pitched data */
            /* Allocate contiguous buffer and copy pitched data */
            size_t total_size = spitch * height;
            HipRemoteMemcpyResponse resp;
            return hip_remote_request_with_data(
                HIP_OP_MEMCPY_2D,
                &req, sizeof(req),
                src, total_size,
                &resp, sizeof(resp)
            );
        }

        case hipMemcpyDeviceToHost: {
            /* For D2H 2D copy, receive pitched data */
            size_t total_size = dpitch * height;
            HipRemoteMemcpyResponse resp;
            return hip_remote_request_receive_data(
                HIP_OP_MEMCPY_2D,
                &req, sizeof(req),
                &resp, sizeof(resp),
                dst, total_size
            );
        }

        case hipMemcpyDeviceToDevice: {
            /* Remote-to-remote copy, no data transfer over network */
            HipRemoteMemcpyResponse resp;
            return hip_remote_request(
                HIP_OP_MEMCPY_2D,
                &req, sizeof(req),
                &resp, sizeof(resp)
            );
        }

        case hipMemcpyHostToHost: {
            /* Local copy - do it row by row */
            const char* s = (const char*)src;
            char* d = (char*)dst;
            for (size_t row = 0; row < height; row++) {
                memcpy(d + row * dpitch, s + row * spitch, width);
            }
            return hipSuccess;
        }

        default:
            return hipErrorInvalidValue;
    }
}

hipError_t hipMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch,
                            size_t width, size_t height, hipMemcpyKind kind, void* stream) {
    if (width == 0 || height == 0) {
        return hipSuccess;
    }
    if (!dst || !src) {
        return hipErrorInvalidValue;
    }

    HipRemoteMemcpy2DRequest req = {
        .dst = (uint64_t)(uintptr_t)dst,
        .dpitch = dpitch,
        .src = (uint64_t)(uintptr_t)src,
        .spitch = spitch,
        .width = width,
        .height = height,
        .kind = (int32_t)kind,
        .reserved = 0,
        .stream = (uint64_t)(uintptr_t)stream
    };

    switch (kind) {
        case hipMemcpyHostToDevice: {
            size_t total_size = spitch * height;
            HipRemoteMemcpyResponse resp;
            return hip_remote_request_with_data(
                HIP_OP_MEMCPY_2D_ASYNC,
                &req, sizeof(req),
                src, total_size,
                &resp, sizeof(resp)
            );
        }

        case hipMemcpyDeviceToHost: {
            size_t total_size = dpitch * height;
            HipRemoteMemcpyResponse resp;
            return hip_remote_request_receive_data(
                HIP_OP_MEMCPY_2D_ASYNC,
                &req, sizeof(req),
                &resp, sizeof(resp),
                dst, total_size
            );
        }

        case hipMemcpyDeviceToDevice: {
            HipRemoteMemcpyResponse resp;
            return hip_remote_request(
                HIP_OP_MEMCPY_2D_ASYNC,
                &req, sizeof(req),
                &resp, sizeof(resp)
            );
        }

        case hipMemcpyHostToHost: {
            const char* s = (const char*)src;
            char* d = (char*)dst;
            for (size_t row = 0; row < height; row++) {
                memcpy(d + row * dpitch, s + row * spitch, width);
            }
            return hipSuccess;
        }

        default:
            return hipErrorInvalidValue;
    }
}

/* ============================================================================
 * Memory Set
 * ============================================================================ */

hipError_t hipMemset(void* dst, int value, size_t size) {
    if (size == 0) {
        return hipSuccess;
    }
    if (!dst) {
        return hipErrorInvalidValue;
    }

    HipRemoteMemsetRequest req = {
        .dst = (uint64_t)(uintptr_t)dst,
        .value = value,
        .size = size,
        .stream = 0
    };
    HipRemoteResponseHeader resp;

    return hip_remote_request(
        HIP_OP_MEMSET,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );
}

hipError_t hipMemsetAsync(void* dst, int value, size_t size, void* stream) {
    if (size == 0) {
        return hipSuccess;
    }
    if (!dst) {
        return hipErrorInvalidValue;
    }

    HipRemoteMemsetRequest req = {
        .dst = (uint64_t)(uintptr_t)dst,
        .value = value,
        .size = size,
        .stream = (uint64_t)(uintptr_t)stream
    };
    HipRemoteResponseHeader resp;

    return hip_remote_request(
        HIP_OP_MEMSET_ASYNC,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );
}

/* ============================================================================
 * Memory Info
 * ============================================================================ */

hipError_t hipMemGetInfo(size_t* free_bytes, size_t* total_bytes) {
    if (!free_bytes || !total_bytes) {
        return hipErrorInvalidValue;
    }

    HipRemoteMemGetInfoResponse resp;
    hipError_t err = hip_remote_request(
        HIP_OP_MEM_GET_INFO,
        NULL, 0,
        &resp, sizeof(resp)
    );

    if (err == hipSuccess) {
        *free_bytes = (size_t)resp.free_bytes;
        *total_bytes = (size_t)resp.total_bytes;
    }
    return err;
}

hipError_t hipPointerGetAttributes(hipPointerAttribute_t* attributes, const void* ptr) {
    if (!attributes) {
        return hipErrorInvalidValue;
    }

    HipRemotePointerGetAttributesRequest req = {
        .ptr = (uint64_t)(uintptr_t)ptr
    };
    HipRemotePointerGetAttributesResponse resp;

    hipError_t err = hip_remote_request(
        HIP_OP_POINTER_GET_ATTRIBUTES,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );

    if (err == hipSuccess) {
        attributes->type = (hipMemoryType)resp.memory_type;
        attributes->device = resp.device;
        attributes->devicePointer = (void*)(uintptr_t)resp.device_pointer;
        attributes->hostPointer = (void*)(uintptr_t)resp.host_pointer;
        attributes->isManaged = resp.is_managed;
        attributes->allocationFlags = resp.allocation_flags;
    }
    return err;
}

/* ============================================================================
 * 3D Memory Copy
 * ============================================================================ */

hipError_t hipMemcpy3D(const hipMemcpy3DParms* p) {
    if (!p) {
        return hipErrorInvalidValue;
    }

    /* Convert hipMemcpy3DParms to wire format */
    HipRemoteMemcpy3DRequest req = {
        .src_ptr = (uint64_t)(uintptr_t)p->srcPtr.ptr,
        .src_pitch = p->srcPtr.pitch,
        .src_height = p->srcPtr.ysize,
        .src_x_offset = p->srcPos.x,
        .src_y_offset = p->srcPos.y,
        .src_z_offset = p->srcPos.z,
        .dst_ptr = (uint64_t)(uintptr_t)p->dstPtr.ptr,
        .dst_pitch = p->dstPtr.pitch,
        .dst_height = p->dstPtr.ysize,
        .dst_x_offset = p->dstPos.x,
        .dst_y_offset = p->dstPos.y,
        .dst_z_offset = p->dstPos.z,
        .width = p->extent.width,
        .height = p->extent.height,
        .depth = p->extent.depth,
        .kind = (int32_t)p->kind,
        .reserved = 0,
        .stream = 0
    };

    HipRemoteResponseHeader resp;
    return hip_remote_request(
        HIP_OP_MEMCPY_3D,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );
}

hipError_t hipMemcpy3DAsync(const hipMemcpy3DParms* p, void* stream) {
    if (!p) {
        return hipErrorInvalidValue;
    }

    HipRemoteMemcpy3DRequest req = {
        .src_ptr = (uint64_t)(uintptr_t)p->srcPtr.ptr,
        .src_pitch = p->srcPtr.pitch,
        .src_height = p->srcPtr.ysize,
        .src_x_offset = p->srcPos.x,
        .src_y_offset = p->srcPos.y,
        .src_z_offset = p->srcPos.z,
        .dst_ptr = (uint64_t)(uintptr_t)p->dstPtr.ptr,
        .dst_pitch = p->dstPtr.pitch,
        .dst_height = p->dstPtr.ysize,
        .dst_x_offset = p->dstPos.x,
        .dst_y_offset = p->dstPos.y,
        .dst_z_offset = p->dstPos.z,
        .width = p->extent.width,
        .height = p->extent.height,
        .depth = p->extent.depth,
        .kind = (int32_t)p->kind,
        .reserved = 0,
        .stream = (uint64_t)(uintptr_t)stream
    };

    HipRemoteResponseHeader resp;
    return hip_remote_request(
        HIP_OP_MEMCPY_3D_ASYNC,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );
}

/* ============================================================================
 * Peer Memory Copy
 * ============================================================================ */

hipError_t hipMemcpyPeer(void* dst, int dstDeviceId, const void* src,
                          int srcDeviceId, size_t sizeBytes) {
    if (sizeBytes == 0) {
        return hipSuccess;
    }
    if (!dst || !src) {
        return hipErrorInvalidValue;
    }

    HipRemoteMemcpyPeerRequest req = {
        .dst = (uint64_t)(uintptr_t)dst,
        .dst_device = dstDeviceId,
        .src = (uint64_t)(uintptr_t)src,
        .src_device = srcDeviceId,
        .size = sizeBytes,
        .stream = 0
    };
    HipRemoteResponseHeader resp;

    return hip_remote_request(
        HIP_OP_MEMCPY_PEER,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );
}

hipError_t hipMemcpyPeerAsync(void* dst, int dstDeviceId, const void* src,
                               int srcDeviceId, size_t sizeBytes, void* stream) {
    if (sizeBytes == 0) {
        return hipSuccess;
    }
    if (!dst || !src) {
        return hipErrorInvalidValue;
    }

    HipRemoteMemcpyPeerRequest req = {
        .dst = (uint64_t)(uintptr_t)dst,
        .dst_device = dstDeviceId,
        .src = (uint64_t)(uintptr_t)src,
        .src_device = srcDeviceId,
        .size = sizeBytes,
        .stream = (uint64_t)(uintptr_t)stream
    };
    HipRemoteResponseHeader resp;

    return hip_remote_request(
        HIP_OP_MEMCPY_PEER_ASYNC,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );
}

/* ============================================================================
 * IPC (Inter-Process Communication) APIs
 * ============================================================================ */

hipError_t hipIpcGetMemHandle(hipIpcMemHandle_t* handle, void* devPtr) {
    if (!handle) {
        return hipErrorInvalidValue;
    }
    if (!devPtr) {
        return hipErrorInvalidDevicePointer;
    }

    HipRemoteIpcGetMemHandleRequest req = {
        .device_ptr = (uint64_t)(uintptr_t)devPtr
    };
    HipRemoteIpcGetMemHandleResponse resp;

    hipError_t err = hip_remote_request(
        HIP_OP_IPC_GET_MEM_HANDLE,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );

    if (err == hipSuccess) {
        memcpy(handle->reserved, resp.handle, HIP_REMOTE_IPC_HANDLE_SIZE);
    }
    return err;
}

hipError_t hipIpcOpenMemHandle(void** devPtr, hipIpcMemHandle_t handle, unsigned int flags) {
    if (!devPtr) {
        return hipErrorInvalidValue;
    }

    HipRemoteIpcOpenMemHandleRequest req;
    memcpy(req.handle, handle.reserved, HIP_REMOTE_IPC_HANDLE_SIZE);
    req.flags = flags;

    HipRemoteIpcOpenMemHandleResponse resp;

    hipError_t err = hip_remote_request(
        HIP_OP_IPC_OPEN_MEM_HANDLE,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );

    if (err == hipSuccess) {
        *devPtr = (void*)(uintptr_t)resp.device_ptr;
    } else {
        *devPtr = NULL;
    }
    return err;
}

hipError_t hipIpcCloseMemHandle(void* devPtr) {
    if (!devPtr) {
        return hipErrorInvalidValue;
    }

    HipRemoteIpcCloseMemHandleRequest req = {
        .device_ptr = (uint64_t)(uintptr_t)devPtr
    };
    HipRemoteResponseHeader resp;

    return hip_remote_request(
        HIP_OP_IPC_CLOSE_MEM_HANDLE,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );
}

hipError_t hipIpcGetEventHandle(hipIpcEventHandle_t* handle, hipEvent_t event) {
    if (!handle) {
        return hipErrorInvalidValue;
    }
    if (!event) {
        return hipErrorInvalidResourceHandle;
    }

    HipRemoteIpcGetEventHandleRequest req = {
        .event = (uint64_t)(uintptr_t)event
    };
    HipRemoteIpcGetEventHandleResponse resp;

    hipError_t err = hip_remote_request(
        HIP_OP_IPC_GET_EVENT_HANDLE,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );

    if (err == hipSuccess) {
        memcpy(handle->reserved, resp.handle, HIP_REMOTE_IPC_HANDLE_SIZE);
    }
    return err;
}

hipError_t hipIpcOpenEventHandle(hipEvent_t* event, hipIpcEventHandle_t handle) {
    if (!event) {
        return hipErrorInvalidValue;
    }

    HipRemoteIpcOpenEventHandleRequest req;
    memcpy(req.handle, handle.reserved, HIP_REMOTE_IPC_HANDLE_SIZE);

    HipRemoteIpcOpenEventHandleResponse resp;

    hipError_t err = hip_remote_request(
        HIP_OP_IPC_OPEN_EVENT_HANDLE,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );

    if (err == hipSuccess) {
        *event = (hipEvent_t)(uintptr_t)resp.event;
    } else {
        *event = NULL;
    }
    return err;
}

/* ============================================================================
 * Memory Pool APIs
 * ============================================================================ */

hipError_t hipMemPoolCreate(hipMemPool_t* memPool, const hipMemPoolProps* poolProps) {
    if (!memPool || !poolProps) {
        return hipErrorInvalidValue;
    }

    HipRemoteMemPoolCreateRequest req = {
        .alloc_type = (int32_t)poolProps->allocType,
        .handle_types = (int32_t)poolProps->handleTypes,
        .location_type = (int32_t)poolProps->location.type,
        .location_id = poolProps->location.id,
        .max_size = poolProps->maxSize,
        .reserved = {0}
    };

    HipRemoteMemPoolCreateResponse resp;

    hipError_t err = hip_remote_request(
        HIP_OP_MEM_POOL_CREATE,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );

    if (err == hipSuccess) {
        *memPool = (hipMemPool_t)(uintptr_t)resp.mem_pool;
    } else {
        *memPool = NULL;
    }
    return err;
}

hipError_t hipMemPoolDestroy(hipMemPool_t memPool) {
    if (!memPool) {
        return hipErrorInvalidValue;
    }

    HipRemoteMemPoolDestroyRequest req = {
        .mem_pool = (uint64_t)(uintptr_t)memPool
    };
    HipRemoteResponseHeader resp;

    return hip_remote_request(
        HIP_OP_MEM_POOL_DESTROY,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );
}

hipError_t hipMemPoolSetAttribute(hipMemPool_t memPool, hipMemPoolAttr attr, void* value) {
    if (!memPool || !value) {
        return hipErrorInvalidValue;
    }

    /* For most attributes, value is a pointer to uint64_t */
    uint64_t attrValue = 0;
    switch (attr) {
        case hipMemPoolReuseFollowEventDependencies:
        case hipMemPoolReuseAllowOpportunistic:
        case hipMemPoolReuseAllowInternalDependencies:
            attrValue = *(int*)value;
            break;
        case hipMemPoolAttrReleaseThreshold:
        case hipMemPoolAttrReservedMemHigh:
        case hipMemPoolAttrUsedMemHigh:
            attrValue = *(uint64_t*)value;
            break;
        default:
            return hipErrorInvalidValue;
    }

    HipRemoteMemPoolSetAttributeRequest req = {
        .mem_pool = (uint64_t)(uintptr_t)memPool,
        .attr = (int32_t)attr,
        .reserved = 0,
        .value = attrValue
    };
    HipRemoteResponseHeader resp;

    return hip_remote_request(
        HIP_OP_MEM_POOL_SET_ATTRIBUTE,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );
}

hipError_t hipMemPoolGetAttribute(hipMemPool_t memPool, hipMemPoolAttr attr, void* value) {
    if (!memPool || !value) {
        return hipErrorInvalidValue;
    }

    HipRemoteMemPoolGetAttributeRequest req = {
        .mem_pool = (uint64_t)(uintptr_t)memPool,
        .attr = (int32_t)attr
    };
    HipRemoteMemPoolGetAttributeResponse resp;

    hipError_t err = hip_remote_request(
        HIP_OP_MEM_POOL_GET_ATTRIBUTE,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );

    if (err == hipSuccess) {
        /* Copy value based on attribute type */
        switch (attr) {
            case hipMemPoolReuseFollowEventDependencies:
            case hipMemPoolReuseAllowOpportunistic:
            case hipMemPoolReuseAllowInternalDependencies:
                *(int*)value = (int)resp.value;
                break;
            case hipMemPoolAttrReleaseThreshold:
            case hipMemPoolAttrReservedMemCurrent:
            case hipMemPoolAttrReservedMemHigh:
            case hipMemPoolAttrUsedMemCurrent:
            case hipMemPoolAttrUsedMemHigh:
                *(uint64_t*)value = resp.value;
                break;
            default:
                return hipErrorInvalidValue;
        }
    }
    return err;
}

hipError_t hipMallocFromPoolAsync(void** devPtr, size_t size, hipMemPool_t memPool, hipStream_t stream) {
    if (!devPtr) {
        return hipErrorInvalidValue;
    }
    if (size == 0) {
        *devPtr = NULL;
        return hipSuccess;
    }
    if (!memPool) {
        return hipErrorInvalidValue;
    }

    HipRemoteMallocFromPoolAsyncRequest req = {
        .size = size,
        .mem_pool = (uint64_t)(uintptr_t)memPool,
        .stream = (uint64_t)(uintptr_t)stream
    };
    HipRemoteMallocResponse resp;

    hipError_t err = hip_remote_request(
        HIP_OP_MALLOC_FROM_POOL_ASYNC,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );

    if (err == hipSuccess) {
        *devPtr = (void*)(uintptr_t)resp.device_ptr;
    } else {
        *devPtr = NULL;
    }
    return err;
}

hipError_t hipMemPoolTrimTo(hipMemPool_t memPool, size_t minBytesToKeep) {
    if (!memPool) {
        return hipErrorInvalidValue;
    }

    HipRemoteMemPoolTrimToRequest req = {
        .mem_pool = (uint64_t)(uintptr_t)memPool,
        .min_bytes_to_hold = minBytesToKeep
    };
    HipRemoteResponseHeader resp;

    return hip_remote_request(
        HIP_OP_MEM_POOL_TRIM_TO,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );
}

hipError_t hipDeviceGetDefaultMemPool(hipMemPool_t* memPool, int device) {
    if (!memPool) {
        return hipErrorInvalidValue;
    }

    HipRemoteDeviceGetMemPoolRequest req = {
        .device = device
    };
    HipRemoteDeviceGetMemPoolResponse resp;

    hipError_t err = hip_remote_request(
        HIP_OP_DEVICE_GET_DEFAULT_MEM_POOL,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );

    if (err == hipSuccess) {
        *memPool = (hipMemPool_t)(uintptr_t)resp.mem_pool;
    } else {
        *memPool = NULL;
    }
    return err;
}

hipError_t hipDeviceSetMemPool(int device, hipMemPool_t memPool) {
    if (!memPool) {
        return hipErrorInvalidValue;
    }

    HipRemoteDeviceSetMemPoolRequest req = {
        .device = device,
        .reserved = 0,
        .mem_pool = (uint64_t)(uintptr_t)memPool
    };
    HipRemoteResponseHeader resp;

    return hip_remote_request(
        HIP_OP_DEVICE_SET_MEM_POOL,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );
}

hipError_t hipDeviceGetMemPool(hipMemPool_t* memPool, int device) {
    if (!memPool) {
        return hipErrorInvalidValue;
    }

    HipRemoteDeviceGetMemPoolRequest req = {
        .device = device
    };
    HipRemoteDeviceGetMemPoolResponse resp;

    hipError_t err = hip_remote_request(
        HIP_OP_DEVICE_GET_MEM_POOL,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );

    if (err == hipSuccess) {
        *memPool = (hipMemPool_t)(uintptr_t)resp.mem_pool;
    } else {
        *memPool = NULL;
    }
    return err;
}

/* ============================================================================
 * Host Memory Registration
 * ============================================================================ */

hipError_t hipHostRegister(void* hostPtr, size_t sizeBytes, unsigned int flags) {
    if (!hostPtr || sizeBytes == 0) {
        return hipErrorInvalidValue;
    }

    HipRemoteHostRegisterRequest req = {
        .host_ptr = (uint64_t)(uintptr_t)hostPtr,
        .size_bytes = sizeBytes,
        .flags = flags
    };
    HipRemoteResponseHeader resp;

    return hip_remote_request(
        HIP_OP_HOST_REGISTER,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );
}

hipError_t hipHostUnregister(void* hostPtr) {
    if (!hostPtr) {
        return hipErrorInvalidValue;
    }

    HipRemoteHostUnregisterRequest req = {
        .host_ptr = (uint64_t)(uintptr_t)hostPtr
    };
    HipRemoteResponseHeader resp;

    return hip_remote_request(
        HIP_OP_HOST_UNREGISTER,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );
}

hipError_t hipHostGetDevicePointer(void** devPtr, void* hstPtr, unsigned int flags) {
    if (!devPtr || !hstPtr) {
        return hipErrorInvalidValue;
    }

    HipRemoteHostGetDevicePointerRequest req = {
        .host_ptr = (uint64_t)(uintptr_t)hstPtr,
        .flags = flags
    };
    HipRemoteHostGetDevicePointerResponse resp;

    hipError_t err = hip_remote_request(
        HIP_OP_HOST_GET_DEVICE_POINTER,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );

    if (err == hipSuccess) {
        *devPtr = (void*)(uintptr_t)resp.device_ptr;
    }
    return err;
}

hipError_t hipHostGetFlags(unsigned int* flagsPtr, void* hostPtr) {
    if (!flagsPtr || !hostPtr) {
        return hipErrorInvalidValue;
    }

    HipRemoteHostGetFlagsRequest req = {
        .host_ptr = (uint64_t)(uintptr_t)hostPtr
    };
    HipRemoteHostGetFlagsResponse resp;

    hipError_t err = hip_remote_request(
        HIP_OP_HOST_GET_FLAGS,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );

    if (err == hipSuccess) {
        *flagsPtr = resp.flags;
    }
    return err;
}

hipError_t hipHostAlloc(void** ptr, size_t size, unsigned int flags) {
    if (!ptr) {
        return hipErrorInvalidValue;
    }
    if (size == 0) {
        *ptr = NULL;
        return hipSuccess;
    }

    HipRemoteHostAllocRequest req = {
        .size = size,
        .flags = flags
    };
    HipRemoteHostAllocResponse resp;

    hipError_t err = hip_remote_request(
        HIP_OP_HOST_ALLOC,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );

    if (err == hipSuccess) {
        *ptr = (void*)(uintptr_t)resp.ptr;
    }
    return err;
}

hipError_t hipHostFree(void* ptr) {
    if (!ptr) {
        return hipSuccess;
    }

    HipRemoteHostFreeRequest req = {
        .ptr = (uint64_t)(uintptr_t)ptr
    };
    HipRemoteResponseHeader resp;

    return hip_remote_request(
        HIP_OP_HOST_FREE,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );
}

/* ============================================================================
 * Pitched Memory Allocation
 * ============================================================================ */

hipError_t hipMemAllocPitch(void** dptr, size_t* pitch, size_t widthInBytes,
                            size_t height, unsigned int elementSizeBytes) {
    if (!dptr || !pitch) {
        return hipErrorInvalidValue;
    }

    HipRemoteMemAllocPitchRequest req = {
        .width_in_bytes = widthInBytes,
        .height = height,
        .element_size = elementSizeBytes
    };
    HipRemoteMemAllocPitchResponse resp;

    hipError_t err = hip_remote_request(
        HIP_OP_MEM_ALLOC_PITCH,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );

    if (err == hipSuccess) {
        *dptr = (void*)(uintptr_t)resp.dptr;
        *pitch = (size_t)resp.pitch;
    }
    return err;
}

/* ============================================================================
 * Unified Memory Management
 * ============================================================================ */

hipError_t hipMemAdvise(const void* dev_ptr, size_t count, hipMemoryAdvise advice, int device) {
    if (!dev_ptr || count == 0) {
        return hipErrorInvalidValue;
    }

    HipRemoteMemAdviseRequest req = {
        .dev_ptr = (uint64_t)(uintptr_t)dev_ptr,
        .count = count,
        .advice = (int32_t)advice,
        .device = device
    };
    HipRemoteResponseHeader resp;

    return hip_remote_request(
        HIP_OP_MEM_ADVISE,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );
}

hipError_t hipMemPrefetchAsync(const void* dev_ptr, size_t count, int device, void* stream) {
    if (!dev_ptr || count == 0) {
        return hipErrorInvalidValue;
    }

    HipRemoteMemPrefetchAsyncRequest req = {
        .dev_ptr = (uint64_t)(uintptr_t)dev_ptr,
        .count = count,
        .device = device,
        .reserved = 0,
        .stream = (uint64_t)(uintptr_t)stream
    };
    HipRemoteResponseHeader resp;

    return hip_remote_request(
        HIP_OP_MEM_PREFETCH_ASYNC,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );
}

hipError_t hipMemRangeGetAttribute(void* data, size_t data_size, hipMemRangeAttribute attribute,
                                   const void* dev_ptr, size_t count) {
    if (!data || !dev_ptr || data_size == 0 || count == 0) {
        return hipErrorInvalidValue;
    }

    HipRemoteMemRangeGetAttributeRequest req = {
        .data_size = data_size,
        .attribute = (int32_t)attribute,
        .reserved = 0,
        .dev_ptr = (uint64_t)(uintptr_t)dev_ptr,
        .count = count
    };

    /* Allocate buffer for response + data */
    size_t resp_size = sizeof(HipRemoteMemRangeGetAttributeResponse) + data_size;
    uint8_t* resp_buf = (uint8_t*)malloc(resp_size);
    if (!resp_buf) {
        return hipErrorOutOfMemory;
    }

    hipError_t err = hip_remote_request(
        HIP_OP_MEM_RANGE_GET_ATTRIBUTE,
        &req, sizeof(req),
        resp_buf, resp_size
    );

    if (err == hipSuccess) {
        /* Copy data from response */
        memcpy(data, resp_buf + sizeof(HipRemoteMemRangeGetAttributeResponse), data_size);
    }

    free(resp_buf);
    return err;
}

hipError_t hipMemRangeGetAttributes(void** data, size_t* data_sizes, hipMemRangeAttribute* attributes,
                                    size_t num_attributes, const void* dev_ptr, size_t count) {
    if (!data || !data_sizes || !attributes || num_attributes == 0 || !dev_ptr || count == 0) {
        return hipErrorInvalidValue;
    }

    if (num_attributes > HIP_REMOTE_MAX_MEM_RANGE_ATTRIBUTES) {
        return hipErrorInvalidValue;
    }

    /* Build request with variable-length arrays */
    size_t req_size = sizeof(HipRemoteMemRangeGetAttributesRequest) +
                      num_attributes * sizeof(int32_t) +  /* attributes */
                      num_attributes * sizeof(uint64_t);  /* data_sizes */

    uint8_t* req_buf = (uint8_t*)malloc(req_size);
    if (!req_buf) {
        return hipErrorOutOfMemory;
    }

    HipRemoteMemRangeGetAttributesRequest* req = (HipRemoteMemRangeGetAttributesRequest*)req_buf;
    req->num_attributes = (uint32_t)num_attributes;
    req->reserved = 0;
    req->dev_ptr = (uint64_t)(uintptr_t)dev_ptr;
    req->count = count;

    /* Copy attributes */
    int32_t* attrs_ptr = (int32_t*)(req_buf + sizeof(HipRemoteMemRangeGetAttributesRequest));
    for (size_t i = 0; i < num_attributes; i++) {
        attrs_ptr[i] = (int32_t)attributes[i];
    }

    /* Copy data sizes */
    uint64_t* sizes_ptr = (uint64_t*)(attrs_ptr + num_attributes);
    for (size_t i = 0; i < num_attributes; i++) {
        sizes_ptr[i] = (uint64_t)data_sizes[i];
    }

    /* Calculate total response data size */
    size_t total_data_size = 0;
    for (size_t i = 0; i < num_attributes; i++) {
        total_data_size += data_sizes[i];
    }

    size_t resp_size = sizeof(HipRemoteMemRangeGetAttributesResponse) + total_data_size;
    uint8_t* resp_buf = (uint8_t*)malloc(resp_size);
    if (!resp_buf) {
        free(req_buf);
        return hipErrorOutOfMemory;
    }

    hipError_t err = hip_remote_request(
        HIP_OP_MEM_RANGE_GET_ATTRIBUTES,
        req_buf, req_size,
        resp_buf, resp_size
    );

    if (err == hipSuccess) {
        /* Copy data for each attribute */
        uint8_t* data_ptr = resp_buf + sizeof(HipRemoteMemRangeGetAttributesResponse);
        for (size_t i = 0; i < num_attributes; i++) {
            memcpy(data[i], data_ptr, data_sizes[i]);
            data_ptr += data_sizes[i];
        }
    }

    free(req_buf);
    free(resp_buf);
    return err;
}

hipError_t hipPointerGetAttribute(void* data, hipPointer_attribute attribute, const void* ptr) {
    if (!data || !ptr) {
        return hipErrorInvalidValue;
    }

    HipRemotePointerGetAttributeRequest req = {
        .ptr = (uint64_t)(uintptr_t)ptr,
        .attribute = (int32_t)attribute,
        .reserved = 0
    };
    HipRemotePointerGetAttributeResponse resp;

    hipError_t err = hip_remote_request(
        HIP_OP_POINTER_GET_ATTRIBUTE,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );

    if (err == hipSuccess) {
        /* The response contains a uint64_t which can represent various types */
        *(uint64_t*)data = resp.data;
    }
    return err;
}

