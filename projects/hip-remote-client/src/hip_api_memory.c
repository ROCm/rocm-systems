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

#include "hip_remote/hip_remote_internal.h"
#include "hip_remote/hip_remote_protocol.h"

#define XXH_INLINE_ALL
#include "hip_remote/xxhash.h"

#include <stdlib.h>
#include <string.h>

/* ============================================================================
 * Virtual Address Allocator
 *
 * The client assigns opaque virtual addresses locally and sends allocations
 * as fire-and-forget.  The worker maintains a vaddr->real_ptr hash map and
 * translates addresses in every subsequent operation.  This eliminates the
 * synchronous round-trip that hipMalloc previously required.
 *
 * The address range starts high (0x7F00_0000_0000) to avoid collisions with
 * real host pointers.  Each allocation is page-aligned so that pointer
 * arithmetic on the client produces valid offsets on the worker.
 * ============================================================================ */

#define VADDR_BASE  0x7F0000000000ULL
#define VADDR_ALIGN 4096ULL

static uint64_t g_next_vaddr = VADDR_BASE;

static uint64_t vaddr_alloc(size_t size) {
    uint64_t addr = g_next_vaddr;
    uint64_t aligned = (size + VADDR_ALIGN - 1) & ~(VADDR_ALIGN - 1);
    if (aligned == 0) aligned = VADDR_ALIGN;
    g_next_vaddr += aligned;
    return addr;
}

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

    /* During graph capture, use synchronous allocation to avoid executing
     * hipMalloc outside the capture context (which would invalidate it). */
    if (hip_remote_is_capturing()) {
        HipRemoteMallocRequest req = { .size = size, .flags = 0 };
        HipRemoteMallocResponse resp;
        hipError_t err = hip_remote_request(
            HIP_OP_MALLOC, &req, sizeof(req),
            &resp, sizeof(resp)
        );
        if (err == hipSuccess) {
            *ptr = (void*)(uintptr_t)resp.device_ptr;
        } else {
            *ptr = NULL;
        }
        return err;
    }

    uint64_t vaddr = vaddr_alloc(size);
    HipRemoteMallocVaddrRequest req = {
        .vaddr = vaddr,
        .size = size,
        .stream = 0,
        .flags = 0
    };

    hipError_t err = hip_remote_request_fire_and_forget(
        HIP_OP_MALLOC_VADDR, &req, sizeof(req)
    );

    if (err == hipSuccess) {
        *ptr = (void*)(uintptr_t)vaddr;
    } else {
        *ptr = NULL;
    }
    return err;
}

hipError_t hipMallocBatch(void** ptrs, const size_t* sizes, uint32_t count) {
    if (!ptrs || !sizes || count == 0) {
        return hipErrorInvalidValue;
    }
    if (count > HIP_REMOTE_MAX_BATCH_MALLOC) {
        return hipErrorInvalidValue;
    }

    HipRemoteMallocBatchRequest req;
    memset(&req, 0, sizeof(req));
    req.count = count;
    for (uint32_t i = 0; i < count; i++) {
        req.sizes[i] = sizes[i];
    }

    HipRemoteMallocBatchResponse resp;
    memset(&resp, 0, sizeof(resp));

    hipError_t err = hip_remote_request(
        HIP_OP_MALLOC_BATCH,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );

    if (err == hipSuccess) {
        for (uint32_t i = 0; i < count; i++) {
            ptrs[i] = (void*)(uintptr_t)resp.ptrs[i];
        }
    } else {
        for (uint32_t i = 0; i < count; i++) {
            ptrs[i] = NULL;
        }
    }
    return err;
}

hipError_t hipFree(void* ptr) {
    if (!ptr) {
        return hipSuccess;
    }

    HipRemoteFreeRequest req = {
        .device_ptr = (uint64_t)(uintptr_t)ptr
    };

    return hip_remote_request_fire_and_forget(
        HIP_OP_FREE, &req, sizeof(req)
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

    return hipSuccess;
}

hipError_t hipFreeHost(void* ptr) {
    if (!ptr) {
        return hipSuccess;
    }

    /* hipMallocHost allocates locally, so just free locally. */
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

    uint64_t vaddr = vaddr_alloc(size);
    HipRemoteMallocVaddrRequest req = {
        .vaddr = vaddr,
        .size = size,
        .stream = 0,
        .flags = flags
    };

    hipError_t err = hip_remote_request_fire_and_forget(
        HIP_OP_MALLOC_VADDR, &req, sizeof(req)
    );

    if (err == hipSuccess) {
        *ptr = (void*)(uintptr_t)vaddr;
    } else {
        *ptr = NULL;
    }
    return err;
}

hipError_t hipMallocAsync(void** ptr, size_t size, hipStream_t stream) {
    if (!ptr) {
        return hipErrorInvalidValue;
    }
    if (size == 0) {
        *ptr = NULL;
        return hipSuccess;
    }

    if (hip_remote_is_capturing()) {
        HipRemoteMallocAsyncRequest req = {
            .size = size,
            .stream = (uint64_t)(uintptr_t)stream
        };
        HipRemoteMallocResponse resp;
        hipError_t err = hip_remote_request(
            HIP_OP_MALLOC_ASYNC, &req, sizeof(req),
            &resp, sizeof(resp)
        );
        if (err == hipSuccess) {
            *ptr = (void*)(uintptr_t)resp.device_ptr;
        } else {
            *ptr = NULL;
        }
        return err;
    }

    uint64_t vaddr = vaddr_alloc(size);
    HipRemoteMallocVaddrRequest req = {
        .vaddr = vaddr,
        .size = size,
        .stream = (uint64_t)(uintptr_t)stream,
        .flags = 0
    };

    hipError_t err = hip_remote_request_fire_and_forget(
        HIP_OP_MALLOC_ASYNC_VADDR, &req, sizeof(req)
    );

    if (err == hipSuccess) {
        *ptr = (void*)(uintptr_t)vaddr;
    } else {
        *ptr = NULL;
    }
    return err;
}

hipError_t hipFreeAsync(void* ptr, hipStream_t stream) {
    if (!ptr) {
        return hipSuccess;  /* NULL free is a no-op */
    }

    HipRemoteFreeAsyncRequest req = {
        .device_ptr = (uint64_t)(uintptr_t)ptr,
        .stream = (uint64_t)(uintptr_t)stream
    };

    return hip_remote_request_fire_and_forget(
        HIP_OP_FREE_ASYNC, &req, sizeof(req)
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
            if (hip_remote_content_cache_enabled()) {
                uint64_t hash = XXH3_64bits(src, size);
                HipRemoteMemcpyCachedRequest creq = {
                    .hash = hash, .dst = req.dst,
                    .size = size, .stream = 0
                };
                if (hip_remote_content_cache_has(hash)) {
                    return hip_remote_request_fire_and_forget(
                        HIP_OP_MEMCPY_HTOD_CACHED, &creq, sizeof(creq));
                }
                hip_remote_content_cache_add(hash);
                return hip_remote_request_with_data_fire_and_forget(
                    HIP_OP_MEMCPY_HTOD_CACHED, &creq, sizeof(creq), src, size);
            }
            return hip_remote_request_with_data_fire_and_forget(
                HIP_OP_MEMCPY,
                &req, sizeof(req),
                src, size
            );
        }

        case hipMemcpyDeviceToHost: {
            HipRemoteMemcpyResponse resp;
            return hip_remote_request_receive_data(
                HIP_OP_MEMCPY,
                &req, sizeof(req),
                &resp, sizeof(resp),
                dst, size
            );
        }

        case hipMemcpyDeviceToDevice: {
            return hip_remote_request_fire_and_forget(
                HIP_OP_MEMCPY, &req, sizeof(req)
            );
        }

        case hipMemcpyHostToHost: {
            memmove(dst, src, size);
            return hipSuccess;
        }

        case hipMemcpyDefault: {
            return hip_remote_request_fire_and_forget(
                HIP_OP_MEMCPY, &req, sizeof(req)
            );
        }

        default:
            return hipErrorInvalidValue;
    }
}

hipError_t hipMemcpyAsync(void* dst, const void* src, size_t size,
                          hipMemcpyKind kind, hipStream_t stream) {
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

    switch (kind) {
        case hipMemcpyHostToDevice: {
            if (hip_remote_content_cache_enabled()) {
                uint64_t hash = XXH3_64bits(src, size);
                HipRemoteMemcpyCachedRequest creq = {
                    .hash = hash, .dst = req.dst,
                    .size = size, .stream = req.stream
                };
                if (hip_remote_content_cache_has(hash)) {
                    return hip_remote_request_fire_and_forget(
                        HIP_OP_MEMCPY_HTOD_CACHED, &creq, sizeof(creq));
                }
                hip_remote_content_cache_add(hash);
                return hip_remote_request_with_data_fire_and_forget(
                    HIP_OP_MEMCPY_HTOD_CACHED, &creq, sizeof(creq), src, size);
            }
            return hip_remote_request_with_data_fire_and_forget(
                HIP_OP_MEMCPY_ASYNC,
                &req, sizeof(req),
                src, size
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
            return hip_remote_request_fire_and_forget(
                HIP_OP_MEMCPY_ASYNC, &req, sizeof(req)
            );
        }

        case hipMemcpyHostToHost: {
            memmove(dst, src, size);
            return hipSuccess;
        }

        default: {
            return hip_remote_request_fire_and_forget(
                HIP_OP_MEMCPY_ASYNC, &req, sizeof(req)
            );
        }
    }
}

/* Convenience functions */
hipError_t hipMemcpyHtoD(void* dst, const void* src, size_t size) {
    return hipMemcpy(dst, src, size, hipMemcpyHostToDevice);
}

hipError_t hipMemcpyDtoH(void* dst, hipDeviceptr_t src, size_t size) {
    return hipMemcpy(dst, src, size, hipMemcpyDeviceToHost);
}

hipError_t hipMemcpyDtoD(hipDeviceptr_t dst, hipDeviceptr_t src, size_t size) {
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
            size_t total_size = spitch * height;
            return hip_remote_request_with_data_fire_and_forget(
                HIP_OP_MEMCPY_2D,
                &req, sizeof(req),
                src, total_size
            );
        }

        case hipMemcpyDeviceToHost: {
            size_t total_size = dpitch * height;
            HipRemoteMemcpyResponse resp;
            return hip_remote_request_receive_data(
                HIP_OP_MEMCPY_2D,
                &req, sizeof(req),
                &resp, sizeof(resp),
                dst, total_size
            );
        }

        case hipMemcpyDeviceToDevice:
            return hip_remote_request_fire_and_forget(
                HIP_OP_MEMCPY_2D, &req, sizeof(req)
            );

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

hipError_t hipMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch,
                            size_t width, size_t height, hipMemcpyKind kind, hipStream_t stream) {
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
            return hip_remote_request_with_data_fire_and_forget(
                HIP_OP_MEMCPY_2D_ASYNC,
                &req, sizeof(req),
                src, total_size
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

        case hipMemcpyDeviceToDevice:
            return hip_remote_request_fire_and_forget(
                HIP_OP_MEMCPY_2D_ASYNC, &req, sizeof(req)
            );

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

    return hip_remote_request_fire_and_forget(
        HIP_OP_MEMSET, &req, sizeof(req)
    );
}

hipError_t hipMemsetAsync(void* dst, int value, size_t size, hipStream_t stream) {
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

    return hip_remote_request_fire_and_forget(
        HIP_OP_MEMSET_ASYNC, &req, sizeof(req)
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

    return hip_remote_request_fire_and_forget(
        HIP_OP_MEMCPY_3D, &req, sizeof(req)
    );
}

hipError_t hipMemcpy3DAsync(const hipMemcpy3DParms* p, hipStream_t stream) {
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

    return hip_remote_request_fire_and_forget(
        HIP_OP_MEMCPY_3D_ASYNC, &req, sizeof(req)
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

    return hip_remote_request_fire_and_forget(
        HIP_OP_MEMCPY_PEER, &req, sizeof(req)
    );
}

hipError_t hipMemcpyPeerAsync(void* dst, int dstDeviceId, const void* src,
                               int srcDeviceId, size_t sizeBytes, hipStream_t stream) {
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

    return hip_remote_request_fire_and_forget(
        HIP_OP_MEMCPY_PEER_ASYNC, &req, sizeof(req)
    );
}

/* ============================================================================
 * Additional Memory Stubs
 * ============================================================================ */

hipError_t hipMemcpyWithStream(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind, hipStream_t stream) {
    return hipMemcpyAsync(dst, src, sizeBytes, (hipMemcpyKind)kind, stream);
}

hipError_t hipMemcpyToSymbol(const void* symbol, const void* src, size_t count, size_t offset, hipMemcpyKind kind) {
    (void)symbol; (void)src; (void)count; (void)offset; (void)kind;
    hip_remote_log_error("hipMemcpyToSymbol: not implemented in remote mode");
    return hipErrorNotSupported;
}

hipError_t hipGetSymbolAddress(void** devPtr, const void* symbol) {
    (void)devPtr; (void)symbol;
    hip_remote_log_error("hipGetSymbolAddress: not implemented in remote mode");
    return hipErrorNotSupported;
}

hipError_t hipMemPoolGetAttribute(hipMemPool_t memPool, hipMemPoolAttr attr, void* value) {
    HipRemoteMemPoolAttrRequest req = {
        .mem_pool = (uint64_t)(uintptr_t)memPool,
        .attr = attr
    };
    HipRemoteMemPoolAttrResponse resp;

    hipError_t err = hip_remote_request(
        HIP_OP_MEMPOOL_GET_ATTRIBUTE,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );
    if (err == hipSuccess && value) {
        memcpy(value, &resp.value, sizeof(uint64_t));
    }
    return err;
}

hipError_t hipMemPoolSetAttribute(hipMemPool_t memPool, hipMemPoolAttr attr, void* value) {
    HipRemoteMemPoolSetAttrRequest req;
    memset(&req, 0, sizeof(req));
    req.mem_pool = (uint64_t)(uintptr_t)memPool;
    req.attr = attr;
    if (value) memcpy(&req.value, value, sizeof(uint64_t));
    HipRemoteResponseHeader resp;

    return hip_remote_request(
        HIP_OP_MEMPOOL_SET_ATTRIBUTE,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );
}

hipError_t hipMemPoolSetAccess(hipMemPool_t memPool, const hipMemAccessDesc* descList, size_t count) {
    hip_remote_log_debug("hipMemPoolSetAccess: pool=%p count=%zu (forwarding to worker)", memPool, count);
    HipRemoteResponseHeader resp;
    return hip_remote_request(
        HIP_OP_MEMPOOL_SET_ACCESS,
        &memPool, sizeof(uint64_t),
        &resp, sizeof(resp)
    );
}

hipError_t hipMemPoolTrimTo(hipMemPool_t memPool, size_t minBytesToKeep) {
    HipRemoteMemPoolTrimRequest req = {
        .mem_pool = (uint64_t)(uintptr_t)memPool,
        .min_bytes_to_keep = minBytesToKeep
    };
    HipRemoteResponseHeader resp;

    return hip_remote_request(
        HIP_OP_MEMPOOL_TRIM_TO,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );
}

hipError_t hipIpcGetMemHandle(hipIpcMemHandle_t* handle, void* devPtr) {
    (void)handle; (void)devPtr;
    return hipErrorNotSupported;
}

hipError_t hipIpcOpenMemHandle(void** devPtr, hipIpcMemHandle_t handle, unsigned int flags) {
    (void)devPtr; (void)handle; (void)flags;
    return hipErrorNotSupported;
}

hipError_t hipIpcCloseMemHandle(void* devPtr) {
    (void)devPtr;
    return hipErrorNotSupported;
}

hipError_t hipIpcGetEventHandle(hipIpcEventHandle_t* handle, hipEvent_t event) {
    (void)handle; (void)event;
    return hipErrorNotSupported;
}

hipError_t hipIpcOpenEventHandle(hipEvent_t* event, hipIpcEventHandle_t handle) {
    (void)event; (void)handle;
    return hipErrorNotSupported;
}
