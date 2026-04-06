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
 * @file hip_api_module.c
 * @brief Module loading and kernel launch API implementation for remote HIP
 */

#include "hip_remote/hip_remote_internal.h"
#include "hip_remote/hip_remote_protocol.h"

#include "hip_remote/hip_remote_platform.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================================
 * Function Info Tracking
 *
 * The remote protocol returns the kernel argument count from the worker.
 * We store this info so that hipModuleLaunchKernel can use the correct
 * argument count instead of requiring NULL-terminated arrays.
 * ============================================================================ */

#define FUNC_INFO_INITIAL 1024

typedef struct {
    hipFunction_t function;
    uint32_t kernarg_size;
    uint32_t num_params;
    HipRemoteParamDesc params[HIP_REMOTE_MAX_PARAM_DESCS];
} FunctionInfo;

static FunctionInfo* g_function_info = NULL;
static uint32_t g_function_count = 0;
static uint32_t g_function_capacity = 0;
static hip_mutex_t g_function_lock = HIP_MUTEX_INIT;

static int func_info_ensure_capacity(void) {
    if (g_function_count < g_function_capacity) return 0;
    uint32_t new_cap = g_function_capacity == 0 ? FUNC_INFO_INITIAL : g_function_capacity * 2;
    FunctionInfo* new_arr = (FunctionInfo*)realloc(g_function_info, new_cap * sizeof(FunctionInfo));
    if (!new_arr) return -1;
    memset(new_arr + g_function_capacity, 0,
           (new_cap - g_function_capacity) * sizeof(FunctionInfo));
    g_function_info = new_arr;
    g_function_capacity = new_cap;
    return 0;
}

/* Thread-local ext-launch state set by hipExtModuleLaunchKernel and
 * consumed by hipModuleLaunchKernel so the events/flags are forwarded
 * through the protocol without duplicating request-building logic. */
typedef struct {
    uint64_t start_event;
    uint64_t stop_event;
    uint32_t ext_flags;
    int      active;
} ExtLaunchState;

#ifdef _WIN32
static __declspec(thread) ExtLaunchState tls_ext_launch = {0};
#else
static __thread ExtLaunchState tls_ext_launch = {0};
#endif

static void fill_ext_launch_fields(HipRemoteLaunchKernelRequest* req) {
    if (tls_ext_launch.active) {
        req->start_event = tls_ext_launch.start_event;
        req->stop_event  = tls_ext_launch.stop_event;
        req->ext_flags   = tls_ext_launch.ext_flags;
    } else {
        req->start_event = 0;
        req->stop_event  = 0;
        req->ext_flags   = 0;
    }
}

void store_function_info_full(hipFunction_t function,
                                     uint32_t kernarg_size,
                                     uint32_t num_params,
                                     const HipRemoteParamDesc* params) {
    hip_mutex_lock(&g_function_lock);

    /* Check if already exists */
    for (uint32_t i = 0; i < g_function_count; i++) {
        if (g_function_info[i].function == function) {
            g_function_info[i].kernarg_size = kernarg_size;
            g_function_info[i].num_params = num_params;
            if (num_params > HIP_REMOTE_MAX_PARAM_DESCS) num_params = HIP_REMOTE_MAX_PARAM_DESCS;
            memcpy(g_function_info[i].params, params, num_params * sizeof(HipRemoteParamDesc));
            hip_mutex_unlock(&g_function_lock);
            return;
        }
    }

    /* Add new entry */
    if (func_info_ensure_capacity() == 0) {
        uint32_t idx = g_function_count++;
        g_function_info[idx].function = function;
        g_function_info[idx].kernarg_size = kernarg_size;
        g_function_info[idx].num_params = num_params;
        if (num_params > HIP_REMOTE_MAX_PARAM_DESCS) num_params = HIP_REMOTE_MAX_PARAM_DESCS;
        memcpy(g_function_info[idx].params, params, num_params * sizeof(HipRemoteParamDesc));
    }

    hip_mutex_unlock(&g_function_lock);
}

static const FunctionInfo* get_function_info(hipFunction_t function) {
    hip_mutex_lock(&g_function_lock);

    for (uint32_t i = 0; i < g_function_count; i++) {
        if (g_function_info[i].function == function) {
            const FunctionInfo* fi = &g_function_info[i];
            hip_mutex_unlock(&g_function_lock);
            return fi;
        }
    }

    hip_mutex_unlock(&g_function_lock);
    return NULL;
}

/* ============================================================================
 * Module Management
 * ============================================================================ */

hipError_t hipModuleLoadData(hipModule_t* module, const void* image) {
    if (!module || !image) {
        return hipErrorInvalidValue;
    }

    /*
     * The image can be either:
     * 1. Raw ELF code object (starts with 0x7f 'E' 'L' 'F')
     * 2. Clang offload bundle (starts with '__CLANG_OFFLOAD_BUNDLE__')
     *
     * We need to determine the size. For bundles, we scan for a reasonable size.
     * The HIP runtime on the worker side will handle extraction.
     */
    const unsigned char* data = (const unsigned char*)image;
    size_t approx_size;

    /* Check for ELF magic */
    if (data[0] == 0x7f && data[1] == 'E' && data[2] == 'L' && data[3] == 'F') {
        /* Parse ELF header to get size (64-bit ELF) */
        uint64_t e_shoff = *(uint64_t*)(data + 40);
        uint16_t e_shentsize = *(uint16_t*)(data + 58);
        uint16_t e_shnum = *(uint16_t*)(data + 60);
        approx_size = (size_t)(e_shoff + e_shnum * e_shentsize);
    }
    /* Check for Clang offload bundle magic */
    else if (memcmp(data, "__CLANG_OFFLOAD_BUNDLE__", 24) == 0) {
        /*
         * Offload bundle format:
         * - Magic (24 bytes): "__CLANG_OFFLOAD_BUNDLE__"
         * - Number of bundles (8 bytes)
         * - For each bundle:
         *   - Offset (8 bytes)
         *   - Size (8 bytes)
         *   - Triple length (8 bytes)
         *   - Triple string
         *
         * Find the largest offset + size to get total bundle size.
         */
        uint64_t num_bundles = *(uint64_t*)(data + 24);
        const unsigned char* ptr = data + 32;
        uint64_t max_end = 0;

        for (uint64_t i = 0; i < num_bundles && i < 16; i++) {
            uint64_t offset = *(uint64_t*)ptr;
            uint64_t size = *(uint64_t*)(ptr + 8);
            uint64_t triple_len = *(uint64_t*)(ptr + 16);
            uint64_t end = offset + size;
            if (end > max_end) max_end = end;
            ptr += 24 + triple_len;
        }
        approx_size = (size_t)max_end;
    }
    /* Check for compressed offload bundle (CCOB) */
    else if (data[0] == 'C' && data[1] == 'C' && data[2] == 'O' && data[3] == 'B') {
        uint16_t version = *(uint16_t*)(data + 4);
        if (version >= 3) {
            /* V3 header: Magic(4) + Version(2) + Method(2) + FileSize(8) + UncompressedSize(8) + Hash(8) */
            uint64_t file_size = *(uint64_t*)(data + 8);
            approx_size = (size_t)file_size;
            hip_remote_log_debug("hipModuleLoadData: CCOB v%u, file_size=%zu", version, approx_size);
        } else if (version == 2) {
            /* V2 header: Magic(4) + Version(2) + Method(2) + FileSize(4) + UncompressedSize(4) + Hash(8) */
            uint32_t file_size = *(uint32_t*)(data + 8);
            approx_size = (size_t)file_size;
            hip_remote_log_debug("hipModuleLoadData: CCOB v%u, file_size=%zu", version, approx_size);
        } else {
            /* V1: Magic(4) + Version(2) + Method(2) + UncompressedSize(4) + Hash(8) = no total size */
            approx_size = 16 * 1024 * 1024;
            hip_remote_log_debug("hipModuleLoadData: CCOB v%u, using default size", version);
        }
    }
    else {
        hip_remote_log_debug("hipModuleLoadData: unknown format magic=0x%02x%02x%02x%02x, using default size",
                             data[0], data[1], data[2], data[3]);
        approx_size = 16 * 1024 * 1024;
    }

    if (approx_size < 64 || approx_size > HIP_REMOTE_MAX_PAYLOAD_SIZE) {
        approx_size = 16 * 1024 * 1024;
    }

    /* No minimum size check — trust the parser for known formats.
     * Unknown formats already default to 16MB. */

    HipRemoteModuleLoadRequest req;
    req.data_size = approx_size;

    HipRemoteModuleLoadResponse resp;
    memset(&resp, 0, sizeof(resp));

    hipError_t err = hip_remote_request_with_data(
        HIP_OP_MODULE_LOAD_DATA,
        &req, sizeof(req),
        image, approx_size,
        &resp, sizeof(resp)
    );

    if (err == hipSuccess) {
        *module = (hipModule_t)(uintptr_t)resp.module;
    }

    return err;
}

hipError_t hipModuleLoadDataEx(hipModule_t* module, const void* image,
                                unsigned int numOptions,
                                hipJitOption* options, void** optionValues) {
    if (numOptions > 0 && options) {
        hip_remote_log_debug("hipModuleLoadDataEx: %u JIT options provided but "
                             "not forwarded to worker (not yet supported)", numOptions);
    }
    return hipModuleLoadData(module, image);
}

hipError_t hipModuleUnload(hipModule_t module) {
    HipRemoteModuleUnloadRequest req;
    req.module = (uint64_t)(uintptr_t)module;

    return hip_remote_request_fire_and_forget(
        HIP_OP_MODULE_UNLOAD, &req, sizeof(req)
    );
}

hipError_t hipModuleGetFunction(hipFunction_t* function, hipModule_t module,
                                 const char* kname) {
    hip_remote_log_debug("hipModuleGetFunction ENTRY: func=%p module=%p kname=%p (%s)",
                         (void*)function, (void*)module, (void*)kname,
                         kname ? kname : "(null)");

    if (!function || !kname) {
        return hipErrorInvalidValue;
    }

    HipRemoteModuleGetFunctionRequest req;
    memset(&req, 0, sizeof(req));
    req.module = (uint64_t)(uintptr_t)module;
    strncpy(req.function_name, kname, sizeof(req.function_name) - 1);

    HipRemoteModuleGetFunctionResponse resp;
    memset(&resp, 0, sizeof(resp));

    hipError_t err = hip_remote_request(
        HIP_OP_MODULE_GET_FUNCTION,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );

    if (err == hipSuccess) {
        *function = (hipFunction_t)(uintptr_t)resp.function;
        store_function_info_full(*function, resp.num_args, resp.num_params, resp.params);
        hip_remote_log_debug("hipModuleGetFunction: function=%p, kernarg_size=%u, num_params=%u",
                             (void*)*function, resp.num_args, resp.num_params);
    }

    return err;
}

/* ============================================================================
 * Kernel Launch
 * ============================================================================ */

hipError_t hipModuleLaunchKernel(hipFunction_t f,
                                  unsigned int gridDimX,
                                  unsigned int gridDimY,
                                  unsigned int gridDimZ,
                                  unsigned int blockDimX,
                                  unsigned int blockDimY,
                                  unsigned int blockDimZ,
                                  unsigned int sharedMemBytes,
                                  hipStream_t stream,
                                  void** kernelParams,
                                  void** extra) {
    if (!f) {
        return hipErrorInvalidHandle;
    }

    /* Handle the 'extra' parameter (HIP_LAUNCH_PARAM_BUFFER_*).
     * Tensile/rocBLAS uses this to pass a flat kernarg buffer. */
    void* extra_buffer = NULL;
    size_t extra_buffer_size = 0;
    if (extra && extra[0]) {
        for (int ei = 0; extra[ei]; ei++) {
            if ((uintptr_t)extra[ei] == 0x01) { /* HIP_LAUNCH_PARAM_BUFFER_POINTER */
                extra_buffer = extra[ei + 1];
                ei++;
            } else if ((uintptr_t)extra[ei] == 0x02) { /* HIP_LAUNCH_PARAM_BUFFER_SIZE */
                extra_buffer_size = *(size_t*)extra[ei + 1];
                ei++;
            } else if ((uintptr_t)extra[ei] == 0x03) { /* HIP_LAUNCH_PARAM_END */
                break;
            }
        }
    }

    /* Get kernel param metadata from stored function info. */
    const FunctionInfo* fi = get_function_info(f);
    uint32_t kernarg_size = fi ? fi->kernarg_size : 0;
    uint32_t num_params = fi ? fi->num_params : 0;

    /* If we have a flat buffer from 'extra', use it directly */
    if (extra_buffer && extra_buffer_size > 0) {
        size_t total_arg_size = extra_buffer_size;

        size_t request_size = sizeof(HipRemoteLaunchKernelRequest) +
                              sizeof(HipRemoteKernelArg) + total_arg_size;

        uint8_t* buffer = (uint8_t*)malloc(request_size);
        if (!buffer) return hipErrorOutOfMemory;

        HipRemoteLaunchKernelRequest* req = (HipRemoteLaunchKernelRequest*)buffer;
        req->function = (uint64_t)(uintptr_t)f;
        req->grid_dim_x = gridDimX;
        req->grid_dim_y = gridDimY;
        req->grid_dim_z = gridDimZ;
        req->block_dim_x = blockDimX;
        req->block_dim_y = blockDimY;
        req->block_dim_z = blockDimZ;
        req->shared_mem_bytes = sharedMemBytes;
        req->stream = (uint64_t)(uintptr_t)stream;
        req->num_args = 1;
        req->launch_flags = 1; /* flat buffer via extra */
        fill_ext_launch_fields(req);

        HipRemoteKernelArg* args = (HipRemoteKernelArg*)(buffer + sizeof(HipRemoteLaunchKernelRequest));
        args[0].size = (uint32_t)extra_buffer_size;
        args[0].offset = 0;

        uint8_t* arg_data = (uint8_t*)(args + 1);
        memcpy(arg_data, extra_buffer, extra_buffer_size);

        hipError_t err = (req->start_event || req->stop_event)
            ? hip_remote_request(HIP_OP_LAUNCH_KERNEL, buffer, request_size,
                                 &(HipRemoteResponseHeader){0}, sizeof(HipRemoteResponseHeader))
            : hip_remote_request_fire_and_forget(HIP_OP_LAUNCH_KERNEL, buffer, request_size);

        free(buffer);
        return err;
    }

    /* Build a flat kernarg buffer using the kernel's actual parameter
     * metadata (offset, size per param) from hipModuleGetFunction.
     * This matches what the real HIP runtime does in captureAndSet:
     *   for each param i: memcpy(buf + desc.offset, kernelParams[i], desc.size) */
    if (num_params == 0 || kernelParams == NULL) {
        hip_remote_log_debug("hipModuleLaunchKernel: no params (num_params=%u, kp=%p)",
                             num_params, (void*)kernelParams);
        if (kernarg_size == 0) {
            /* Zero-arg kernel — send empty launch */
            HipRemoteLaunchKernelRequest req_hdr;
            memset(&req_hdr, 0, sizeof(req_hdr));
            req_hdr.function = (uint64_t)(uintptr_t)f;
            req_hdr.grid_dim_x = gridDimX; req_hdr.grid_dim_y = gridDimY; req_hdr.grid_dim_z = gridDimZ;
            req_hdr.block_dim_x = blockDimX; req_hdr.block_dim_y = blockDimY; req_hdr.block_dim_z = blockDimZ;
            req_hdr.shared_mem_bytes = sharedMemBytes;
            req_hdr.stream = (uint64_t)(uintptr_t)stream;
            req_hdr.num_args = 0;
            req_hdr.launch_flags = 1;
            fill_ext_launch_fields(&req_hdr);

            if (req_hdr.start_event || req_hdr.stop_event)
                return hip_remote_request(HIP_OP_LAUNCH_KERNEL, &req_hdr, sizeof(req_hdr),
                                          &(HipRemoteResponseHeader){0}, sizeof(HipRemoteResponseHeader));
            return hip_remote_request_fire_and_forget(HIP_OP_LAUNCH_KERNEL, &req_hdr, sizeof(req_hdr));
        }
    }

    size_t total_arg_size = kernarg_size > 0 ? kernarg_size : 256;

    /* Single flat buffer as "extra" — the worker always uses this path */
    size_t request_size = sizeof(HipRemoteLaunchKernelRequest) +
                          sizeof(HipRemoteKernelArg) + total_arg_size;

    uint8_t* buffer = (uint8_t*)malloc(request_size);
    if (!buffer) return hipErrorOutOfMemory;

    HipRemoteLaunchKernelRequest* req = (HipRemoteLaunchKernelRequest*)buffer;
    req->function = (uint64_t)(uintptr_t)f;
    req->grid_dim_x = gridDimX; req->grid_dim_y = gridDimY; req->grid_dim_z = gridDimZ;
    req->block_dim_x = blockDimX; req->block_dim_y = blockDimY; req->block_dim_z = blockDimZ;
    req->shared_mem_bytes = sharedMemBytes;
    req->stream = (uint64_t)(uintptr_t)stream;
    req->num_args = 1;
    req->launch_flags = 1; /* flat buffer via extra */
    fill_ext_launch_fields(req);

    HipRemoteKernelArg* args = (HipRemoteKernelArg*)(buffer + sizeof(HipRemoteLaunchKernelRequest));
    args[0].offset = 0;
    args[0].size = (uint32_t)total_arg_size;

    uint8_t* arg_data = (uint8_t*)(args + 1);
    memset(arg_data, 0, total_arg_size);

    if (num_params > 0 && fi && kernelParams) {
        /* Use exact parameter metadata from hipKernelGetParamInfo */
        for (uint32_t i = 0; i < num_params; i++) {
            uint32_t off = fi->params[i].offset;
            uint32_t sz = fi->params[i].size;
            if (off + sz > total_arg_size) break;
#ifdef _MSC_VER
            __try {
                memcpy(arg_data + off, kernelParams[i], sz);
            } __except(1) { break; }
#else
            memcpy(arg_data + off, kernelParams[i], sz);
#endif
        }
    } else if (kernelParams) {
        /* No COMGR metadata: copy kernelParams into the flat buffer
         * using the extra path.  Each kernelParams[i] points to the
         * value of the i-th argument.  We read 8 bytes from each and
         * pack them contiguously.  Stop at the first NULL entry or
         * when we've filled total_arg_size bytes. */
        uint32_t nargs = (uint32_t)(total_arg_size / 8);
        if (nargs > HIP_REMOTE_MAX_KERNEL_ARGS) nargs = HIP_REMOTE_MAX_KERNEL_ARGS;
        uint32_t actual_args = 0;
        for (uint32_t i = 0; i < nargs; i++) {
            if (kernelParams[i] == NULL) break;
            uint64_t ptr_val = (uint64_t)(uintptr_t)kernelParams[i];
            if (ptr_val < 0x10000 || ptr_val > 0x00007FFFFFFFFFFFULL) break;
            memcpy(arg_data + i * 8, kernelParams[i], 8);
            actual_args = i + 1;
        }

        hip_remote_log_debug("hipModuleLaunchKernel: no metadata, packed %u args into flat buffer", actual_args);
    }

    hip_remote_log_debug("hipModuleLaunchKernel: built flat kernarg (%u bytes, %u params)",
                         (uint32_t)total_arg_size, num_params);

    hipError_t err = (req->start_event || req->stop_event)
        ? hip_remote_request(HIP_OP_LAUNCH_KERNEL, buffer, request_size,
                             &(HipRemoteResponseHeader){0}, sizeof(HipRemoteResponseHeader))
        : hip_remote_request_fire_and_forget(HIP_OP_LAUNCH_KERNEL, buffer, request_size);

    free(buffer);
    return err;
}

/*
 * Fat binary registry lookup - defined in hip_api_fatbin.c.
 * Returns the remote hipFunction_t for a host function pointer that was
 * previously registered via __hipRegisterFunction, or NULL if not found.
 */
extern hipFunction_t hip_fatbin_lookup_function(const void* hostFunction);

extern uint32_t hip_fatbin_get_registered_count(void);

/*
 * Pop call configuration - defined in hip_api_fatbin.c.
 * Retrieves the launch configuration pushed by __hipPushCallConfiguration
 * (generated by the <<<>>> syntax).
 */
extern hipError_t __hipPopCallConfiguration(dim3* gridDim, dim3* blockDim,
                                            size_t* sharedMem,
                                            hipStream_t* stream);

hipError_t hipLaunchKernel(const void* function_address,
                            dim3 numBlocks,
                            dim3 dimBlocks,
                            void** args,
                            size_t sharedMemBytes,
                            hipStream_t stream) {
    /*
     * Look up the host function pointer in the fat binary registry.
     * This translates the local (meaningless) host stub pointer into the
     * remote hipFunction_t handle that was obtained when
     * __hipRegisterFunction called hipModuleGetFunction on the worker.
     */
    hipFunction_t func = hip_fatbin_lookup_function(function_address);
    if (!func) {
        hip_remote_log_debug(
            "hipLaunchKernel: host function %p not in fatbin registry (%u registered)",
            function_address, hip_fatbin_get_registered_count());
        return hipErrorInvalidDeviceFunction;
    }

    /* The kernel stub already popped the call configuration via
     * __hipPopCallConfiguration and passes the values as parameters.
     * Use them directly — do NOT pop again. */
    return hipModuleLaunchKernel(
        func,
        numBlocks.x, numBlocks.y, numBlocks.z,
        dimBlocks.x, dimBlocks.y, dimBlocks.z,
        (unsigned int)sharedMemBytes,
        stream,
        args,
        NULL
    );
}

/* ============================================================================
 * Cooperative Launch (stub)
 * ============================================================================ */

hipError_t hipLaunchCooperativeKernel(const void* f,
                                       dim3 gridDim,
                                       dim3 blockDim,
                                       void** kernelParams,
                                       unsigned int sharedMemBytes,
                                       hipStream_t stream) {
    hipFunction_t func = hip_fatbin_lookup_function(f);
    if (!func) {
        func = (hipFunction_t)f;
    }
    hip_remote_log_debug("hipLaunchCooperativeKernel: f=%p -> func=%p", f, (void*)func);

    return hipModuleLaunchKernel(
        func,
        gridDim.x, gridDim.y, gridDim.z,
        blockDim.x, blockDim.y, blockDim.z,
        sharedMemBytes,
        stream,
        kernelParams,
        NULL
    );
}

/* ============================================================================
 * Occupancy Cache
 *
 * Occupancy queries depend only on the kernel and its parameters, which are
 * fixed for the lifetime of a loaded module. Caching avoids repeated
 * round-trips for the same (function, sharedMem, blockSizeLimit) tuple.
 * ============================================================================ */

#define OCCUPANCY_CACHE_SIZE 64

typedef struct {
    uint64_t function;
    size_t   dyn_shared_mem;
    int      block_size_limit;
    int      min_grid_size;
    int      block_size;
} OccupancyPotentialEntry;

typedef struct {
    uint64_t function;
    int      block_size;
    size_t   dyn_shared_mem;
    int      num_blocks;
} OccupancyActiveEntry;

static OccupancyPotentialEntry g_occ_potential[OCCUPANCY_CACHE_SIZE];
static int g_occ_potential_count = 0;

static OccupancyActiveEntry g_occ_active[OCCUPANCY_CACHE_SIZE];
static int g_occ_active_count = 0;

/* ============================================================================
 * Occupancy APIs
 * ============================================================================ */

hipError_t hipOccupancyMaxPotentialBlockSize(int* minGridSize, int* blockSize,
                                              const void* f, size_t dynSharedMemPerBlk,
                                              int blockSizeLimit) {
    if (!minGridSize || !blockSize || !f) {
        return hipErrorInvalidValue;
    }

    uint64_t fh = (uint64_t)(uintptr_t)f;
    for (int i = 0; i < g_occ_potential_count; i++) {
        OccupancyPotentialEntry* e = &g_occ_potential[i];
        if (e->function == fh && e->dyn_shared_mem == dynSharedMemPerBlk
            && e->block_size_limit == blockSizeLimit) {
            *minGridSize = e->min_grid_size;
            *blockSize = e->block_size;
            return hipSuccess;
        }
    }

    HipRemoteOccupancyMaxPotentialBlockSizeRequest req = {
        .function = fh,
        .dyn_shared_mem = dynSharedMemPerBlk,
        .block_size_limit = blockSizeLimit,
        .flags = 0
    };
    HipRemoteOccupancyMaxPotentialBlockSizeResponse resp;

    hipError_t err = hip_remote_request(
        HIP_OP_OCCUPANCY_MAX_POTENTIAL_BLOCK_SIZE,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );

    if (err == hipSuccess) {
        *minGridSize = resp.min_grid_size;
        *blockSize = resp.block_size;
        if (g_occ_potential_count < OCCUPANCY_CACHE_SIZE) {
            OccupancyPotentialEntry* e = &g_occ_potential[g_occ_potential_count++];
            e->function = fh;
            e->dyn_shared_mem = dynSharedMemPerBlk;
            e->block_size_limit = blockSizeLimit;
            e->min_grid_size = resp.min_grid_size;
            e->block_size = resp.block_size;
        }
    }
    return err;
}

hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, const void* f,
                                                         int blockSize, size_t dynSharedMemPerBlk) {
    if (!numBlocks || !f) {
        return hipErrorInvalidValue;
    }

    uint64_t fh = (uint64_t)(uintptr_t)f;
    for (int i = 0; i < g_occ_active_count; i++) {
        OccupancyActiveEntry* e = &g_occ_active[i];
        if (e->function == fh && e->block_size == blockSize
            && e->dyn_shared_mem == dynSharedMemPerBlk) {
            *numBlocks = e->num_blocks;
            return hipSuccess;
        }
    }

    HipRemoteOccupancyMaxActiveBlocksPerSMRequest req = {
        .function = fh,
        .block_size = blockSize,
        .dyn_shared_mem = dynSharedMemPerBlk
    };
    HipRemoteOccupancyMaxActiveBlocksPerSMResponse resp;

    hipError_t err = hip_remote_request(
        HIP_OP_OCCUPANCY_MAX_ACTIVE_BLOCKS_PER_SM,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );

    if (err == hipSuccess) {
        *numBlocks = resp.num_blocks;
        if (g_occ_active_count < OCCUPANCY_CACHE_SIZE) {
            OccupancyActiveEntry* e = &g_occ_active[g_occ_active_count++];
            e->function = fh;
            e->block_size = blockSize;
            e->dyn_shared_mem = dynSharedMemPerBlk;
            e->num_blocks = resp.num_blocks;
        }
    }
    return err;
}

hipError_t hipExtModuleLaunchKernel(hipFunction_t f,
                                     unsigned int globalWorkSizeX,
                                     unsigned int globalWorkSizeY,
                                     unsigned int globalWorkSizeZ,
                                     unsigned int localWorkSizeX,
                                     unsigned int localWorkSizeY,
                                     unsigned int localWorkSizeZ,
                                     size_t sharedMemBytes,
                                     hipStream_t hStream,
                                     void** kernelParams,
                                     void** extra,
                                     hipEvent_t startEvent,
                                     hipEvent_t stopEvent,
                                     unsigned int flags) {
    hip_remote_log_debug("hipExtModuleLaunchKernel: f=%p grid=(%u,%u,%u) block=(%u,%u,%u) "
                         "shared=%zu stream=%p start=%p stop=%p flags=%u",
                         (void*)f, globalWorkSizeX, globalWorkSizeY, globalWorkSizeZ,
                         localWorkSizeX, localWorkSizeY, localWorkSizeZ,
                         sharedMemBytes, (void*)hStream,
                         (void*)startEvent, (void*)stopEvent, flags);

    tls_ext_launch.start_event = (uint64_t)(uintptr_t)startEvent;
    tls_ext_launch.stop_event  = (uint64_t)(uintptr_t)stopEvent;
    tls_ext_launch.ext_flags   = flags;
    tls_ext_launch.active      = 1;

    /* Convert globalWorkSize to gridDim (number of blocks) for the
     * remote protocol, which always uses gridDim semantics. */
    unsigned int gridX = localWorkSizeX > 0 ? globalWorkSizeX / localWorkSizeX : globalWorkSizeX;
    unsigned int gridY = localWorkSizeY > 0 ? globalWorkSizeY / localWorkSizeY : globalWorkSizeY;
    unsigned int gridZ = localWorkSizeZ > 0 ? globalWorkSizeZ / localWorkSizeZ : globalWorkSizeZ;

    hipError_t err = hipModuleLaunchKernel(f,
        gridX, gridY, gridZ,
        localWorkSizeX, localWorkSizeY, localWorkSizeZ,
        (unsigned int)sharedMemBytes, hStream, kernelParams, extra);

    tls_ext_launch.active = 0;

    hip_remote_log_debug("hipExtModuleLaunchKernel: err=%d", err);
    return err;
}

hipError_t hipModuleLoad(hipModule_t* module, const char* fname) {
    if (!module || !fname) return hipErrorInvalidValue;

    /* Read the file locally and send directly with the KNOWN file size.
     * Don't go through hipModuleLoadData's size guessing — it can get
     * the size wrong for offload bundles. */
    FILE* f = fopen(fname, "rb");
    if (!f) {
        hip_remote_log_error("hipModuleLoad: cannot open '%s'", fname);
        return hipErrorFileNotFound;
    }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (len <= 0) {
        fclose(f);
        return hipErrorInvalidImage;
    }
    void* buf = malloc((size_t)len);
    if (!buf) { fclose(f); return hipErrorOutOfMemory; }
    fread(buf, 1, (size_t)len, f);
    fclose(f);

    /* Send with the exact file size */
    HipRemoteModuleLoadRequest req;
    req.data_size = (uint64_t)len;

    HipRemoteModuleLoadResponse resp;
    memset(&resp, 0, sizeof(resp));

    hipError_t err = hip_remote_request_with_data(
        HIP_OP_MODULE_LOAD_DATA,
        &req, sizeof(req),
        buf, (size_t)len,
        &resp, sizeof(resp)
    );

    free(buf);

    if (err == hipSuccess) {
        *module = (hipModule_t)(uintptr_t)resp.module;
    } else {
        hip_remote_log_error("hipModuleLoad failed: %s\n error: %s", fname, hipGetErrorString(err));
    }
    return err;
}

hipError_t hipModuleLaunchCooperativeKernel(hipFunction_t f,
                                             unsigned int gridDimX,
                                             unsigned int gridDimY,
                                             unsigned int gridDimZ,
                                             unsigned int blockDimX,
                                             unsigned int blockDimY,
                                             unsigned int blockDimZ,
                                             unsigned int sharedMemBytes,
                                             hipStream_t stream,
                                             void** kernelParams) {
    return hipModuleLaunchKernel(f, gridDimX, gridDimY, gridDimZ,
                                blockDimX, blockDimY, blockDimZ,
                                sharedMemBytes, stream, kernelParams, NULL);
}

hipError_t hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks,
                                                               hipFunction_t f,
                                                               int blockSize,
                                                               size_t dynSharedMemPerBlk) {
    return hipOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, (const void*)f,
                                                        blockSize, dynSharedMemPerBlk);
}
