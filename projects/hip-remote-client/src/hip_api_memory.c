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
