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

#include "hip_remote/hip_remote_client.h"
#include "hip_remote/hip_remote_protocol.h"

#include <pthread.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================================
 * Function Info Tracking
 *
 * The remote protocol returns the kernel argument count from the worker.
 * We store this info so that hipModuleLaunchKernel can use the correct
 * argument count instead of requiring NULL-terminated arrays.
 * ============================================================================ */

#define MAX_TRACKED_FUNCTIONS 1024

typedef struct {
    hipFunction_t function;
    uint32_t num_args;
} FunctionInfo;

static FunctionInfo g_function_info[MAX_TRACKED_FUNCTIONS];
static uint32_t g_function_count = 0;
static pthread_mutex_t g_function_lock = PTHREAD_MUTEX_INITIALIZER;

static void store_function_info(hipFunction_t function, uint32_t num_args) {
    pthread_mutex_lock(&g_function_lock);

    /* Check if already exists */
    for (uint32_t i = 0; i < g_function_count; i++) {
        if (g_function_info[i].function == function) {
            g_function_info[i].num_args = num_args;
            pthread_mutex_unlock(&g_function_lock);
            return;
        }
    }

    /* Add new entry */
    if (g_function_count < MAX_TRACKED_FUNCTIONS) {
        g_function_info[g_function_count].function = function;
        g_function_info[g_function_count].num_args = num_args;
        g_function_count++;
    }

    pthread_mutex_unlock(&g_function_lock);
}

static uint32_t get_function_num_args(hipFunction_t function) {
    pthread_mutex_lock(&g_function_lock);

    for (uint32_t i = 0; i < g_function_count; i++) {
        if (g_function_info[i].function == function) {
            uint32_t num_args = g_function_info[i].num_args;
            pthread_mutex_unlock(&g_function_lock);
            return num_args;
        }
    }

    pthread_mutex_unlock(&g_function_lock);
    return 0; /* Unknown function - return 0 args */
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
    else {
        /* Unknown format - use default max and let worker validate */
        hip_remote_log_debug("hipModuleLoadData: unknown format, using default size");
        approx_size = 16 * 1024 * 1024;
    }

    if (approx_size < 64 || approx_size > HIP_REMOTE_MAX_PAYLOAD_SIZE) {
        approx_size = 16 * 1024 * 1024;
    }

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
    /* For remote execution, JIT options are handled on the worker side */
    /* We just forward the request - options will be ignored for now */
    (void)numOptions;
    (void)options;
    (void)optionValues;

    return hipModuleLoadData(module, image);
}

hipError_t hipModuleUnload(hipModule_t module) {
    HipRemoteModuleUnloadRequest req;
    req.module = (uint64_t)(uintptr_t)module;

    HipRemoteResponseHeader resp;
    memset(&resp, 0, sizeof(resp));

    return hip_remote_request(
        HIP_OP_MODULE_UNLOAD,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );
}

hipError_t hipModuleGetFunction(hipFunction_t* function, hipModule_t module,
                                 const char* kname) {
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
        /* Store the argument count from the worker for use at launch time */
        store_function_info(*function, resp.num_args);
        hip_remote_log_debug("hipModuleGetFunction: function=%p, num_args=%u",
                             (void*)*function, resp.num_args);
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

    /* extra parameter is not supported in remote mode */
    if (extra && extra[0]) {
        hip_remote_log_error("hipModuleLaunchKernel: extra parameter not supported");
        return hipErrorNotSupported;
    }

    /* Get the number of arguments from stored function info.
     * This was populated when hipModuleGetFunction was called.
     * If num_args is 0, it means either:
     * 1. The kernel truly has 0 args, or
     * 2. The worker doesn't support hipKernelGetParamInfo (older ROCm)
     * In case 2, fall back to NULL-terminated counting. */
    uint32_t num_args = get_function_num_args(f);

    if (num_args == 0 && kernelParams != NULL) {
        /* Fall back to NULL-terminated counting for compatibility */
        while (kernelParams[num_args] != NULL && num_args < HIP_REMOTE_MAX_KERNEL_ARGS) {
            num_args++;
        }
        hip_remote_log_debug("hipModuleLaunchKernel: using NULL-terminated arg count: %u", num_args);
    } else if (num_args > 0 && kernelParams == NULL) {
        hip_remote_log_error("hipModuleLaunchKernel: kernel expects %u args but kernelParams is NULL", num_args);
        return hipErrorInvalidValue;
    }

    /* For simplicity, we assume each argument is a pointer (8 bytes) */
    /* In a real implementation, we'd need argument metadata from the kernel */
    /* For now, treat each kernelParams entry as a pointer-sized value */
    size_t arg_size = sizeof(void*);
    size_t total_arg_size = num_args * arg_size;

    /* Build the request */
    size_t request_size = sizeof(HipRemoteLaunchKernelRequest) +
                          num_args * sizeof(HipRemoteKernelArg) +
                          total_arg_size;

    uint8_t* buffer = (uint8_t*)malloc(request_size);
    if (!buffer) {
        return hipErrorOutOfMemory;
    }

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
    req->num_args = num_args;

    /* Fill in argument descriptors and data */
    HipRemoteKernelArg* args = (HipRemoteKernelArg*)(buffer + sizeof(HipRemoteLaunchKernelRequest));
    uint8_t* arg_data = (uint8_t*)(args + num_args);
    uint32_t offset = 0;

    for (uint32_t i = 0; i < num_args; i++) {
        args[i].size = (uint32_t)arg_size;
        args[i].offset = offset;
        memcpy(arg_data + offset, kernelParams[i], arg_size);
        offset += (uint32_t)arg_size;
    }

    HipRemoteResponseHeader resp;
    memset(&resp, 0, sizeof(resp));

    hipError_t err = hip_remote_request(
        HIP_OP_LAUNCH_KERNEL,
        buffer, request_size,
        &resp, sizeof(resp)
    );

    free(buffer);
    return err;
}

hipError_t hipLaunchKernel(const void* function_address,
                            dim3 numBlocks,
                            dim3 dimBlocks,
                            void** args,
                            size_t sharedMemBytes,
                            hipStream_t stream) {
    /*
     * hipLaunchKernel with a host function pointer is used when the kernel
     * is linked into the executable. For remote execution, we need the
     * module/function approach instead.
     *
     * This function cannot work directly in remote mode because the function
     * pointer is meaningless on the remote worker. Applications should use
     * hipModuleLaunchKernel instead.
     */
    hip_remote_log_error("hipLaunchKernel: host function pointers not supported in remote mode");
    hip_remote_log_error("Use hipModuleLoadData + hipModuleGetFunction + hipModuleLaunchKernel instead");
    (void)function_address;
    (void)numBlocks;
    (void)dimBlocks;
    (void)args;
    (void)sharedMemBytes;
    (void)stream;
    return hipErrorNotSupported;
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
    /* Cooperative kernels require additional synchronization that's complex
     * to implement over the network. For now, fall back to regular launch. */
    hip_remote_log_debug("hipLaunchCooperativeKernel: using regular launch");

    return hipModuleLaunchKernel(
        (hipFunction_t)f,
        gridDim.x, gridDim.y, gridDim.z,
        blockDim.x, blockDim.y, blockDim.z,
        sharedMemBytes,
        stream,
        kernelParams,
        NULL
    );
}

/* ============================================================================
 * Occupancy APIs
 * ============================================================================ */

hipError_t hipOccupancyMaxPotentialBlockSize(int* minGridSize, int* blockSize,
                                              hipFunction_t f, size_t dynSharedMemPerBlk,
                                              int blockSizeLimit) {
    if (!minGridSize || !blockSize || !f) {
        return hipErrorInvalidValue;
    }

    HipRemoteOccupancyMaxPotentialBlockSizeRequest req = {
        .function = (uint64_t)(uintptr_t)f,
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
    }
    return err;
}

hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, hipFunction_t f,
                                                         int blockSize, size_t dynSharedMemPerBlk) {
    if (!numBlocks || !f) {
        return hipErrorInvalidValue;
    }

    HipRemoteOccupancyMaxActiveBlocksPerSMRequest req = {
        .function = (uint64_t)(uintptr_t)f,
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
    }
    return err;
}

hipError_t hipLaunchCooperativeKernelMultiDevice(hipFunctionLaunchParams* launchParamsList,
                                                  int numDevices, unsigned int flags) {
    if (!launchParamsList || numDevices <= 0) {
        return hipErrorInvalidValue;
    }

    /* This is a complex API that requires coordination across multiple devices.
     * For remote execution, this would require:
     * 1. Serializing the launch params for each device
     * 2. Serializing all kernel arguments
     * 3. Coordinating launch across multiple remote devices
     *
     * For now, return hipErrorNotSupported as this is rarely used and complex
     * to implement in remote mode. */
    (void)launchParamsList;
    (void)numDevices;
    (void)flags;

    hip_remote_log_error("hipLaunchCooperativeKernelMultiDevice: not supported in remote mode");
    return hipErrorNotSupported;
}

/* ============================================================================
 * Function Attributes APIs
 * ============================================================================ */

hipError_t hipFuncGetAttributes(hipFuncAttributes* attr, const void* func) {
    if (!attr || !func) {
        return hipErrorInvalidValue;
    }

    HipRemoteFuncGetAttributesRequest req = {
        .function = (uint64_t)(uintptr_t)func
    };
    HipRemoteFuncGetAttributesResponse resp;

    hipError_t err = hip_remote_request(
        HIP_OP_FUNC_GET_ATTRIBUTES,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );

    if (err == hipSuccess) {
        attr->sharedSizeBytes = resp.shared_size_bytes;
        attr->constSizeBytes = resp.const_size_bytes;
        attr->localSizeBytes = resp.local_size_bytes;
        attr->numRegs = resp.num_regs;
        attr->maxThreadsPerBlock = resp.max_threads_per_block;
        attr->ptxVersion = resp.ptx_version;
        attr->binaryVersion = resp.binary_version;
        attr->cacheModeCA = resp.cache_mode_ca;
        attr->maxDynamicSharedSizeBytes = resp.max_dynamic_shared_size_bytes;
        attr->preferredShmemCarveout = resp.preferred_shared_memory_carveout;
    }
    return err;
}

hipError_t hipFuncSetAttribute(const void* func, hipFunction_attribute attr, int value) {
    if (!func) {
        return hipErrorInvalidValue;
    }

    HipRemoteFuncSetAttributeRequest req = {
        .function = (uint64_t)(uintptr_t)func,
        .attribute = (int32_t)attr,
        .value = value
    };
    HipRemoteResponseHeader resp;

    return hip_remote_request(
        HIP_OP_FUNC_SET_ATTRIBUTE,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );
}

hipError_t hipFuncSetCacheConfig(const void* func, hipFuncCache_t cacheConfig) {
    if (!func) {
        return hipErrorInvalidValue;
    }

    HipRemoteFuncSetCacheConfigRequest req = {
        .function = (uint64_t)(uintptr_t)func,
        .cache_config = (int32_t)cacheConfig
    };
    HipRemoteResponseHeader resp;

    return hip_remote_request(
        HIP_OP_FUNC_SET_CACHE_CONFIG,
        &req, sizeof(req),
        &resp, sizeof(resp)
    );
}

/* ============================================================================
 * PyTorch Compatibility Extensions
 * ============================================================================ */

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
    (void)startEvent; (void)stopEvent; (void)flags;
    hip_remote_log_debug("hipExtModuleLaunchKernel: f=%p grid=(%u,%u,%u) block=(%u,%u,%u) "
                         "shared=%zu stream=%p kp=%p extra=%p",
                         (void*)f, globalWorkSizeX, globalWorkSizeY, globalWorkSizeZ,
                         localWorkSizeX, localWorkSizeY, localWorkSizeZ,
                         sharedMemBytes, (void*)hStream, (void*)kernelParams, (void*)extra);
    hipError_t err = hipModuleLaunchKernel(f,
        globalWorkSizeX, globalWorkSizeY, globalWorkSizeZ,
        localWorkSizeX, localWorkSizeY, localWorkSizeZ,
        (unsigned int)sharedMemBytes, hStream, kernelParams, extra);
    hip_remote_log_debug("hipExtModuleLaunchKernel: err=%d", err);
    return err;
}

hipError_t hipModuleLoad(hipModule_t* module, const char* fname) {
    if (!module || !fname) return hipErrorInvalidValue;

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
