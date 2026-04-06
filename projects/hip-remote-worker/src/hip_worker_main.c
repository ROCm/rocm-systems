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
 * @file hip_worker_main.c
 * @brief HIP worker service for remote HIP execution
 *
 * This service runs on a Linux system with AMD GPUs and handles
 * HIP API requests from remote clients (e.g., macOS).
 */

#ifndef _WIN32
#define _POSIX_C_SOURCE 200809L
#endif

#include <errno.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#pragma comment(lib, "ws2_32.lib")
typedef SOCKET hip_worker_socket_t;
#define HIP_WORKER_INVALID_SOCKET INVALID_SOCKET
#define close_socket closesocket
static int worker_socket_init(void) {
    WSADATA wsa;
    return WSAStartup(MAKEWORD(2, 2), &wsa);
}
static void worker_socket_cleanup(void) { WSACleanup(); }
#else
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <unistd.h>
#include <pthread.h>
#include <dlfcn.h>
typedef int hip_worker_socket_t;
#define HIP_WORKER_INVALID_SOCKET (-1)
#define close_socket close
static int worker_socket_init(void) { return 0; }
static void worker_socket_cleanup(void) {}
#endif

#ifdef _WIN32
#define getpid _getpid
#include <process.h>
#include <signal.h>
#endif

#include <hip/hip_runtime.h>

#ifdef _WIN32
extern hipError_t hipExtModuleLaunchKernel(hipFunction_t, unsigned int, unsigned int,
    unsigned int, unsigned int, unsigned int, unsigned int, size_t, hipStream_t,
    void**, void**, hipEvent_t, hipEvent_t, unsigned int);
#endif

/* Include protocol from client */
#include "hip_remote/hip_remote_protocol.h"

/* SMI handlers (conditionally compiled) */
#ifdef HIP_WORKER_SMI_ENABLED
#include "smi_worker_handlers.h"
#endif

/* ============================================================================
 * Configuration
 * ============================================================================ */

static int g_listen_port = HIP_REMOTE_DEFAULT_PORT;
static int g_default_device = 0;
static bool g_debug_enabled = false;
static volatile bool g_running = true;
static int g_server_fd = -1;
static int g_hip_initialized = 0;

/* ============================================================================
 * Virtual Address Translation
 *
 * The client assigns opaque virtual addresses (vaddrs) for GPU pointers and
 * stream handles.  The worker maintains a hash map from vaddr to the real
 * pointer/handle returned by the HIP runtime.  Every handler translates
 * incoming vaddrs before calling the real HIP API.
 * ============================================================================ */

#define VADDR_MAP_SIZE (1 << 20)  /* 1M slots, ~16MB */
#define VADDR_MAP_MASK (VADDR_MAP_SIZE - 1)
#define VADDR_BASE     0x7F0000000000ULL
#define VSTREAM_BASE   0x5F0000000000ULL
#define VADDR_EMPTY    0ULL
#define VADDR_TOMBSTONE 1ULL  /* deleted slot, continue probing */
#define VADDR_ERROR    2ULL   /* sentinel: allocation failed */

typedef struct {
    uint64_t vaddr;
    uint64_t real_ptr;
    uint64_t size;         /* allocation size, for offset lookups */
} VaddrEntry;

static VaddrEntry g_vaddr_map[VADDR_MAP_SIZE];
static hipError_t g_deferred_alloc_error = hipSuccess;

#define VADDR_ALLOC_MAX 65536
typedef struct { uint64_t base; uint64_t real; uint64_t size; } VaddrAlloc;
static VaddrAlloc g_vaddr_allocs[VADDR_ALLOC_MAX];
static int g_vaddr_alloc_count = 0;
static int g_vaddr_allocs_sorted = 1;

/* Last-hit cache: most kernel launches re-use the same few allocations */
static uint64_t g_vaddr_cache_vbase = 0;
static uint64_t g_vaddr_cache_rbase = 0;
static uint64_t g_vaddr_cache_size  = 0;

static int vaddr_alloc_cmp(const void* a, const void* b) {
    uint64_t ba = ((const VaddrAlloc*)a)->base;
    uint64_t bb = ((const VaddrAlloc*)b)->base;
    return (ba > bb) - (ba < bb);
}

static void vaddr_allocs_ensure_sorted(void) {
    if (!g_vaddr_allocs_sorted && g_vaddr_alloc_count > 1) {
        qsort(g_vaddr_allocs, g_vaddr_alloc_count, sizeof(VaddrAlloc), vaddr_alloc_cmp);
        g_vaddr_allocs_sorted = 1;
    }
}

static const VaddrAlloc* vaddr_allocs_bsearch(uint64_t vaddr) {
    vaddr_allocs_ensure_sorted();
    int lo = 0, hi = g_vaddr_alloc_count - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (vaddr < g_vaddr_allocs[mid].base) {
            hi = mid - 1;
        } else if (vaddr >= g_vaddr_allocs[mid].base + g_vaddr_allocs[mid].size) {
            lo = mid + 1;
        } else {
            return &g_vaddr_allocs[mid];
        }
    }
    return NULL;
}

static void vaddr_map_put(uint64_t vaddr, uint64_t real_ptr, uint64_t size) {
    /* Insert into hash map for O(1) exact lookup */
    uint32_t idx = (uint32_t)(vaddr >> 12) & VADDR_MAP_MASK;
    uint32_t first_tombstone = UINT32_MAX;
    for (uint32_t i = 0; i < VADDR_MAP_SIZE; i++) {
        uint32_t slot = (idx + i) & VADDR_MAP_MASK;
        if (g_vaddr_map[slot].vaddr == vaddr) {
            g_vaddr_map[slot].real_ptr = real_ptr;
            g_vaddr_map[slot].size = size;
            goto update_alloc_list;
        }
        if (g_vaddr_map[slot].vaddr == VADDR_TOMBSTONE && first_tombstone == UINT32_MAX) {
            first_tombstone = slot;
        }
        if (g_vaddr_map[slot].vaddr == VADDR_EMPTY) {
            uint32_t target = (first_tombstone != UINT32_MAX) ? first_tombstone : slot;
            g_vaddr_map[target].vaddr = vaddr;
            g_vaddr_map[target].real_ptr = real_ptr;
            g_vaddr_map[target].size = size;
            goto update_alloc_list;
        }
    }
    if (first_tombstone != UINT32_MAX) {
        g_vaddr_map[first_tombstone].vaddr = vaddr;
        g_vaddr_map[first_tombstone].real_ptr = real_ptr;
        g_vaddr_map[first_tombstone].size = size;
    }

update_alloc_list:
    /* Also record in compact alloc list for fast range lookups */
    if (size > 0 && real_ptr != VADDR_ERROR) {
        for (int i = 0; i < g_vaddr_alloc_count; i++) {
            if (g_vaddr_allocs[i].base == vaddr) {
                g_vaddr_allocs[i].real = real_ptr;
                g_vaddr_allocs[i].size = size;
                return;
            }
        }
        if (g_vaddr_alloc_count < VADDR_ALLOC_MAX) {
            g_vaddr_allocs[g_vaddr_alloc_count].base = vaddr;
            g_vaddr_allocs[g_vaddr_alloc_count].real = real_ptr;
            g_vaddr_allocs[g_vaddr_alloc_count].size = size;
            g_vaddr_alloc_count++;
            g_vaddr_allocs_sorted = 0;
        }
    }
}

static uint64_t vaddr_map_get(uint64_t vaddr) {
    if (vaddr == 0) return 0;
    if (vaddr < VSTREAM_BASE) return vaddr;

    /* Hot path: last-hit cache (same allocation block as previous lookup) */
    if (g_vaddr_cache_size &&
        vaddr >= g_vaddr_cache_vbase &&
        vaddr <  g_vaddr_cache_vbase + g_vaddr_cache_size) {
        return g_vaddr_cache_rbase + (vaddr - g_vaddr_cache_vbase);
    }

    /* Fast path: exact match via hash */
    uint32_t idx = (uint32_t)(vaddr >> 12) & VADDR_MAP_MASK;
    for (uint32_t i = 0; i < VADDR_MAP_SIZE; i++) {
        uint32_t slot = (idx + i) & VADDR_MAP_MASK;
        if (g_vaddr_map[slot].vaddr == vaddr) {
            if (g_vaddr_map[slot].size > 0) {
                g_vaddr_cache_vbase = vaddr;
                g_vaddr_cache_rbase = g_vaddr_map[slot].real_ptr;
                g_vaddr_cache_size  = g_vaddr_map[slot].size;
            }
            return g_vaddr_map[slot].real_ptr;
        }
        if (g_vaddr_map[slot].vaddr == VADDR_EMPTY)
            break;
    }

    /* Range lookup via sorted array + binary search (O(log N)) */
    const VaddrAlloc* hit = vaddr_allocs_bsearch(vaddr);
    if (hit) {
        g_vaddr_cache_vbase = hit->base;
        g_vaddr_cache_rbase = hit->real;
        g_vaddr_cache_size  = hit->size;
        return hit->real + (vaddr - hit->base);
    }

    return vaddr;
}

static void vaddr_map_remove(uint64_t vaddr) {
    if (vaddr == 0) return;
    uint32_t idx = (uint32_t)(vaddr >> 12) & VADDR_MAP_MASK;
    for (uint32_t i = 0; i < VADDR_MAP_SIZE; i++) {
        uint32_t slot = (idx + i) & VADDR_MAP_MASK;
        if (g_vaddr_map[slot].vaddr == vaddr) {
            g_vaddr_map[slot].vaddr = VADDR_TOMBSTONE;
            g_vaddr_map[slot].real_ptr = 0;
            g_vaddr_map[slot].size = 0;
            break;
        }
        if (g_vaddr_map[slot].vaddr == VADDR_EMPTY) break;
    }
    /* Invalidate cache if removing the cached block */
    if (g_vaddr_cache_vbase == vaddr) g_vaddr_cache_size = 0;
    /* Remove from alloc list */
    for (int i = 0; i < g_vaddr_alloc_count; i++) {
        if (g_vaddr_allocs[i].base == vaddr) {
            g_vaddr_allocs[i] = g_vaddr_allocs[--g_vaddr_alloc_count];
            g_vaddr_allocs_sorted = 0;
            return;
        }
    }
}

static void* vaddr_translate(uint64_t v) {
    return (void*)(uintptr_t)vaddr_map_get(v);
}

/* ============================================================================
 * Code Object Storage (for COMGR metadata extraction)
 *
 * Dynamically growable so Triton workloads with thousands of JIT-compiled
 * kernels never lose module data needed for COMGR metadata extraction.
 * ============================================================================ */

#define LOADED_MODULES_INITIAL 256

typedef struct {
    hipModule_t module;
    void* data;
    size_t size;
} LoadedModuleEntry;

static LoadedModuleEntry* g_loaded_modules = NULL;
static int g_loaded_module_count = 0;
static int g_loaded_module_capacity = 0;

static int loaded_modules_ensure_capacity(void) {
    if (g_loaded_module_count < g_loaded_module_capacity) return 0;
    int new_cap = g_loaded_module_capacity == 0 ? LOADED_MODULES_INITIAL : g_loaded_module_capacity * 2;
    LoadedModuleEntry* new_arr = (LoadedModuleEntry*)realloc(
        g_loaded_modules, new_cap * sizeof(LoadedModuleEntry));
    if (!new_arr) return -1;
    memset(new_arr + g_loaded_module_capacity, 0,
           (new_cap - g_loaded_module_capacity) * sizeof(LoadedModuleEntry));
    g_loaded_modules = new_arr;
    g_loaded_module_capacity = new_cap;
    return 0;
}

/* Per-function COMGR metadata cache — used to reconstruct kernelParams
 * from a flat buffer during kernel launch.  Dynamically growable so that
 * Triton workloads with thousands of unique JIT-compiled kernels don't
 * overflow and fall back to the fragile 8-byte scan heuristic. */
#define FUNC_CACHE_INITIAL 1024

typedef struct {
    hipFunction_t function;
    uint32_t num_params;
    uint32_t kernarg_size;          /* User-args-only size (from param offsets) */
    uint32_t kernarg_segment_size;  /* Full segment size from code object metadata */
    HipRemoteParamDesc params[HIP_REMOTE_MAX_PARAM_DESCS];
} CachedFunctionInfo;

static CachedFunctionInfo* g_func_cache = NULL;
static int g_func_cache_count = 0;
static int g_func_cache_capacity = 0;

static int func_cache_ensure_capacity(void) {
    if (g_func_cache_count < g_func_cache_capacity) return 0;
    int new_cap = g_func_cache_capacity == 0 ? FUNC_CACHE_INITIAL : g_func_cache_capacity * 2;
    CachedFunctionInfo* new_arr = (CachedFunctionInfo*)realloc(
        g_func_cache, new_cap * sizeof(CachedFunctionInfo));
    if (!new_arr) return -1;
    memset(new_arr + g_func_cache_capacity, 0,
           (new_cap - g_func_cache_capacity) * sizeof(CachedFunctionInfo));
    g_func_cache = new_arr;
    g_func_cache_capacity = new_cap;
    return 0;
}

static void cache_function_info(hipFunction_t func, uint32_t num_params,
                                uint32_t kernarg_size,
                                uint32_t kernarg_segment_size,
                                const HipRemoteParamDesc* params) {
    for (int i = 0; i < g_func_cache_count; i++) {
        if (g_func_cache[i].function == func) {
            g_func_cache[i].num_params = num_params;
            g_func_cache[i].kernarg_size = kernarg_size;
            g_func_cache[i].kernarg_segment_size = kernarg_segment_size;
            memcpy(g_func_cache[i].params, params, num_params * sizeof(HipRemoteParamDesc));
            return;
        }
    }
    if (func_cache_ensure_capacity() != 0) {
        fprintf(stderr, "[HIP-Worker ERROR] func cache realloc failed (count=%d)\n", g_func_cache_count);
        return;
    }
    int idx = g_func_cache_count++;
    g_func_cache[idx].function = func;
    g_func_cache[idx].num_params = num_params;
    g_func_cache[idx].kernarg_size = kernarg_size;
    g_func_cache[idx].kernarg_segment_size = kernarg_segment_size;
    memcpy(g_func_cache[idx].params, params, num_params * sizeof(HipRemoteParamDesc));
}

static const CachedFunctionInfo* lookup_function_info(hipFunction_t func) {
    for (int i = 0; i < g_func_cache_count; i++) {
        if (g_func_cache[i].function == func)
            return &g_func_cache[i];
    }
    if (g_debug_enabled) {
        fprintf(stderr, "[HIP-Worker] lookup_function_info: func=%p NOT FOUND in %d entries\n",
                (void*)func, g_func_cache_count);
    }
    return NULL;
}

static void store_module_data(hipModule_t module, const void* data, size_t size) {
    /* Check if this module handle already has stored data (handle reuse after
     * hipModuleUnload + hipModuleLoadData). Update in place to ensure
     * find_module_data returns the latest code object. */
    for (int i = 0; i < g_loaded_module_count; i++) {
        if (g_loaded_modules[i].module == module) {
            free(g_loaded_modules[i].data);
            g_loaded_modules[i].data = malloc(size);
            if (g_loaded_modules[i].data) {
                memcpy(g_loaded_modules[i].data, data, size);
                g_loaded_modules[i].size = size;
            } else {
                g_loaded_modules[i].size = 0;
            }
            return;
        }
    }

    if (loaded_modules_ensure_capacity() != 0) {
        fprintf(stderr, "[HIP-Worker ERROR] loaded_modules realloc failed (count=%d)\n",
                g_loaded_module_count);
        return;
    }
    int idx = g_loaded_module_count++;
    g_loaded_modules[idx].module = module;
    g_loaded_modules[idx].data = malloc(size);
    if (g_loaded_modules[idx].data) {
        memcpy(g_loaded_modules[idx].data, data, size);
        g_loaded_modules[idx].size = size;
    } else {
        g_loaded_modules[idx].size = 0;
    }
}

static const LoadedModuleEntry* find_module_data(hipModule_t module) {
    for (int i = 0; i < g_loaded_module_count; i++) {
        if (g_loaded_modules[i].module == module)
            return &g_loaded_modules[i];
    }
    return NULL;
}

/* kernarg_segment_size cache per function handle -- dynamically growable */
#define KERNARG_SIZES_INITIAL 4096

typedef struct {
    hipFunction_t func;
    uint32_t kernarg_size;
} KernargSizeEntry;

static KernargSizeEntry* g_kernarg_sizes = NULL;
static int g_kernarg_size_count = 0;
static int g_kernarg_size_capacity = 0;

static void store_kernarg_size(hipFunction_t func, uint32_t size) {
    for (int i = 0; i < g_kernarg_size_count; i++) {
        if (g_kernarg_sizes[i].func == func) {
            g_kernarg_sizes[i].kernarg_size = size;
            return;
        }
    }
    if (g_kernarg_size_count >= g_kernarg_size_capacity) {
        int new_cap = g_kernarg_size_capacity == 0 ? KERNARG_SIZES_INITIAL : g_kernarg_size_capacity * 2;
        KernargSizeEntry* new_arr = (KernargSizeEntry*)realloc(
            g_kernarg_sizes, new_cap * sizeof(KernargSizeEntry));
        if (!new_arr) return;
        g_kernarg_sizes = new_arr;
        g_kernarg_size_capacity = new_cap;
    }
    g_kernarg_sizes[g_kernarg_size_count].func = func;
    g_kernarg_sizes[g_kernarg_size_count].kernarg_size = size;
    g_kernarg_size_count++;
}

static uint32_t get_kernarg_size(hipFunction_t func) {
    for (int i = 0; i < g_kernarg_size_count; i++) {
        if (g_kernarg_sizes[i].func == func) return g_kernarg_sizes[i].kernarg_size;
    }
    return 0;
}

/* Cache for COMGR-extracted kernel arg metadata -- dynamically growable */
#define KERNEL_ARG_CACHE_INITIAL 1024

typedef struct {
    hipModule_t module;
    char kernel_name[512];
    uint32_t num_params;
    uint32_t kernarg_segment_size;
    HipRemoteParamDesc params[HIP_REMOTE_MAX_PARAM_DESCS];
    int valid;
} CachedKernelArgs;

static CachedKernelArgs* g_kernel_arg_cache = NULL;
static int g_kernel_arg_cache_count = 0;
static int g_kernel_arg_cache_capacity = 0;

static const CachedKernelArgs* find_cached_kernel_args(hipModule_t module, const char* name) {
    for (int i = 0; i < g_kernel_arg_cache_count; i++) {
        if (g_kernel_arg_cache[i].valid &&
            g_kernel_arg_cache[i].module == module &&
            strcmp(g_kernel_arg_cache[i].kernel_name, name) == 0) {
            return &g_kernel_arg_cache[i];
        }
    }
    return NULL;
}

static void cache_kernel_args(hipModule_t module, const char* name,
                               uint32_t num_params, uint32_t kernarg_segment_size,
                               const HipRemoteParamDesc* params) {
    if (g_kernel_arg_cache_count >= g_kernel_arg_cache_capacity) {
        int new_cap = g_kernel_arg_cache_capacity == 0 ? KERNEL_ARG_CACHE_INITIAL : g_kernel_arg_cache_capacity * 2;
        CachedKernelArgs* new_arr = (CachedKernelArgs*)realloc(
            g_kernel_arg_cache, new_cap * sizeof(CachedKernelArgs));
        if (!new_arr) return;
        memset(new_arr + g_kernel_arg_cache_capacity, 0,
               (new_cap - g_kernel_arg_cache_capacity) * sizeof(CachedKernelArgs));
        g_kernel_arg_cache = new_arr;
        g_kernel_arg_cache_capacity = new_cap;
    }

    int idx = g_kernel_arg_cache_count++;
    g_kernel_arg_cache[idx].module = module;
    strncpy(g_kernel_arg_cache[idx].kernel_name, name, sizeof(g_kernel_arg_cache[idx].kernel_name) - 1);
    g_kernel_arg_cache[idx].kernel_name[sizeof(g_kernel_arg_cache[idx].kernel_name) - 1] = '\0';
    g_kernel_arg_cache[idx].num_params = num_params;
    g_kernel_arg_cache[idx].kernarg_segment_size = kernarg_segment_size;
    if (num_params > 0 && params)
        memcpy(g_kernel_arg_cache[idx].params, params, num_params * sizeof(HipRemoteParamDesc));
    g_kernel_arg_cache[idx].valid = 1;
}

/* ============================================================================
 * Content-Addressed GPU Cache
 *
 * Caches H2D memcpy payloads in GPU memory keyed by XXH64 hash.  When the
 * client re-sends the same weight data on a subsequent load it includes only
 * the hash; the worker performs a fast D2D copy from the cache instead of
 * receiving the data over the network again.
 * ============================================================================ */

#define XXH_INLINE_ALL
#include "xxhash.h"

#define GPU_CACHE_INITIAL 4096
#define GPU_CACHE_DEFAULT_VRAM_FRACTION 0.33

typedef struct {
    uint64_t hash;
    void*    gpu_ptr;     /* hipMalloc'd cache buffer */
    uint64_t size;
    uint64_t last_used;   /* monotonic counter for LRU */
} GpuCacheEntry;

static GpuCacheEntry* g_gpu_cache = NULL;
static int g_gpu_cache_count = 0;
static int g_gpu_cache_capacity = 0;
static uint64_t g_gpu_cache_total_bytes = 0;
static uint64_t g_gpu_cache_max_bytes = 0;  /* set at runtime from VRAM size */
static int g_gpu_cache_max_explicit = 0;    /* 1 if user set HIP_REMOTE_CACHE_SIZE_MB */
static uint64_t g_gpu_cache_lru_clock = 0;
static int g_gpu_cache_enabled = 0;

static void gpu_cache_init(void) {
    const char* env = getenv("HIP_REMOTE_CACHE_SIZE_MB");
    if (env && env[0] != '\0') {
        long mb = atol(env);
        if (mb > 0) {
            g_gpu_cache_max_bytes = (uint64_t)mb * 1024 * 1024;
            g_gpu_cache_max_explicit = 1;
        }
    }
    g_gpu_cache_enabled = 1;
}

static void gpu_cache_size_from_vram(void) {
    if (g_gpu_cache_max_explicit) return;

    hipDeviceProp_t props;
    if (hipGetDeviceProperties(&props, g_default_device) == hipSuccess &&
        props.totalGlobalMem > 0) {
        g_gpu_cache_max_bytes = (uint64_t)(props.totalGlobalMem *
                                           GPU_CACHE_DEFAULT_VRAM_FRACTION);
        fprintf(stderr, "[HIP-Worker] Content cache: %.0f%% of %.1f GB VRAM = %llu MB\n",
                GPU_CACHE_DEFAULT_VRAM_FRACTION * 100,
                props.totalGlobalMem / (1024.0 * 1024.0 * 1024.0),
                (unsigned long long)(g_gpu_cache_max_bytes / (1024 * 1024)));
    } else {
        g_gpu_cache_max_bytes = (uint64_t)16384 * 1024 * 1024;
        fprintf(stderr, "[HIP-Worker] Content cache: could not query VRAM, defaulting to 16 GB\n");
    }
}

static void* gpu_cache_lookup(uint64_t hash) {
    for (int i = 0; i < g_gpu_cache_count; i++) {
        if (g_gpu_cache[i].hash == hash) {
            g_gpu_cache[i].last_used = ++g_gpu_cache_lru_clock;
            return g_gpu_cache[i].gpu_ptr;
        }
    }
    return NULL;
}

static uint64_t gpu_cache_lookup_size(uint64_t hash) {
    for (int i = 0; i < g_gpu_cache_count; i++) {
        if (g_gpu_cache[i].hash == hash)
            return g_gpu_cache[i].size;
    }
    return 0;
}

static void gpu_cache_evict_lru(void) {
    if (g_gpu_cache_count == 0) return;
    int victim = 0;
    uint64_t oldest = g_gpu_cache[0].last_used;
    for (int i = 1; i < g_gpu_cache_count; i++) {
        if (g_gpu_cache[i].last_used < oldest) {
            oldest = g_gpu_cache[i].last_used;
            victim = i;
        }
    }
    g_gpu_cache_total_bytes -= g_gpu_cache[victim].size;
    hipFree(g_gpu_cache[victim].gpu_ptr);
    g_gpu_cache[victim] = g_gpu_cache[g_gpu_cache_count - 1];
    g_gpu_cache_count--;
}

static void gpu_cache_store(uint64_t hash, const void* src_gpu, uint64_t size) {
    if (gpu_cache_lookup(hash)) return;

    while (g_gpu_cache_total_bytes + size > g_gpu_cache_max_bytes &&
           g_gpu_cache_count > 0) {
        gpu_cache_evict_lru();
    }

    void* buf = NULL;
    hipError_t err = hipMalloc(&buf, size);
    if (err != hipSuccess || !buf) return;

    err = hipMemcpy(buf, src_gpu, size, hipMemcpyDeviceToDevice);
    if (err != hipSuccess) { hipFree(buf); return; }

    if (g_gpu_cache_count >= g_gpu_cache_capacity) {
        int new_cap = g_gpu_cache_capacity == 0 ? GPU_CACHE_INITIAL : g_gpu_cache_capacity * 2;
        GpuCacheEntry* new_arr = (GpuCacheEntry*)realloc(
            g_gpu_cache, new_cap * sizeof(GpuCacheEntry));
        if (!new_arr) { hipFree(buf); return; }
        g_gpu_cache = new_arr;
        g_gpu_cache_capacity = new_cap;
    }

    int idx = g_gpu_cache_count++;
    g_gpu_cache[idx].hash = hash;
    g_gpu_cache[idx].gpu_ptr = buf;
    g_gpu_cache[idx].size = size;
    g_gpu_cache[idx].last_used = ++g_gpu_cache_lru_clock;
    g_gpu_cache_total_bytes += size;
}

static int gpu_cache_get_hashes(uint64_t* out, int max_count) {
    int n = g_gpu_cache_count < max_count ? g_gpu_cache_count : max_count;
    for (int i = 0; i < n; i++)
        out[i] = g_gpu_cache[i].hash;
    return n;
}

/* ============================================================================
 * COMGR-based kernel argument metadata extraction
 *
 * Uses amd_comgr to parse the AMDGPU MSGPACK metadata from code objects
 * to get the exact offset and size of each kernel parameter. This is
 * critical for correctly serializing kernelParams[] on the client side.
 * ============================================================================ */

typedef int amd_comgr_status_t;
typedef struct { uint64_t handle; } amd_comgr_data_t;
typedef struct { uint64_t handle; } amd_comgr_metadata_node_t;

#define AMD_COMGR_DATA_KIND_EXECUTABLE 0x8
#define AMD_COMGR_DATA_KIND_FATBIN 0x10
#define AMD_COMGR_DATA_KIND_OBJ_BUNDLE 0x14
#define AMD_COMGR_STATUS_SUCCESS 0

typedef amd_comgr_status_t (*pfn_create_data)(int kind, amd_comgr_data_t*);
typedef amd_comgr_status_t (*pfn_set_data)(amd_comgr_data_t, size_t, const char*);
typedef amd_comgr_status_t (*pfn_get_data_metadata)(amd_comgr_data_t, amd_comgr_metadata_node_t*);
typedef amd_comgr_status_t (*pfn_metadata_lookup)(amd_comgr_metadata_node_t, const char*, amd_comgr_metadata_node_t*);
typedef amd_comgr_status_t (*pfn_get_metadata_list_size)(amd_comgr_metadata_node_t, size_t*);
typedef amd_comgr_status_t (*pfn_index_list_metadata)(amd_comgr_metadata_node_t, size_t, amd_comgr_metadata_node_t*);
typedef amd_comgr_status_t (*pfn_get_metadata_string)(amd_comgr_metadata_node_t, size_t*, char*);
typedef amd_comgr_status_t (*pfn_destroy_metadata)(amd_comgr_metadata_node_t);
typedef amd_comgr_status_t (*pfn_release_data)(amd_comgr_data_t);

static struct {
    void* lib;
    pfn_create_data create_data;
    pfn_set_data set_data;
    pfn_get_data_metadata get_data_metadata;
    pfn_metadata_lookup metadata_lookup;
    pfn_get_metadata_list_size get_metadata_list_size;
    pfn_index_list_metadata index_list_metadata;
    pfn_get_metadata_string get_metadata_string;
    pfn_destroy_metadata destroy_metadata;
    pfn_release_data release_data;
    int loaded;
} g_comgr = {0};

static int load_comgr(void) {
    if (g_comgr.loaded) return g_comgr.lib != NULL;
    g_comgr.loaded = 1;
#ifdef _WIN32
    g_comgr.lib = (void*)LoadLibraryA("amd_comgr0702.dll");
    if (!g_comgr.lib) g_comgr.lib = (void*)LoadLibraryA("amd_comgr.dll");
#else
    g_comgr.lib = dlopen("libamd_comgr.so", RTLD_LAZY);
#endif
    if (!g_comgr.lib) return 0;

#ifdef _WIN32
    #define LOAD_FN(name) g_comgr.name = (pfn_##name)(void*)GetProcAddress((HMODULE)g_comgr.lib, "amd_comgr_" #name)
#else
    #define LOAD_FN(name) g_comgr.name = (pfn_##name)dlsym(g_comgr.lib, "amd_comgr_" #name)
#endif
    LOAD_FN(create_data);
    LOAD_FN(set_data);
    LOAD_FN(get_data_metadata);
    LOAD_FN(metadata_lookup);
    LOAD_FN(get_metadata_list_size);
    LOAD_FN(index_list_metadata);
    LOAD_FN(get_metadata_string);
    LOAD_FN(destroy_metadata);
    LOAD_FN(release_data);
    #undef LOAD_FN

    return g_comgr.create_data && g_comgr.get_data_metadata && g_comgr.metadata_lookup;
}

/**
 * Look up a string value by key from a COMGR metadata map node.
 * Uses metadata_lookup + two-call get_metadata_string pattern,
 * matching the CLR runtime's getMetaBuf approach (devkernel.cpp:47).
 * Returns heap-allocated string or NULL. Caller must free().
 */
static char* comgr_lookup_string(amd_comgr_metadata_node_t map_node, const char* key) {
    amd_comgr_metadata_node_t value_node = {0};
    if (g_comgr.metadata_lookup(map_node, key, &value_node) != AMD_COMGR_STATUS_SUCCESS)
        return NULL;

    size_t size = 0;
    if (g_comgr.get_metadata_string(value_node, &size, NULL) != AMD_COMGR_STATUS_SUCCESS) {
        g_comgr.destroy_metadata(value_node);
        return NULL;
    }

    char* buf = (char*)malloc(size);
    if (buf) {
        g_comgr.get_metadata_string(value_node, &size, buf);
    }
    g_comgr.destroy_metadata(value_node);
    return buf;
}

/**
 * Extract kernel argument metadata from a code object using COMGR.
 * Returns the number of params found, or 0 on failure.
 */
static uint32_t comgr_extract_kernel_params(const void* code_data, size_t code_size,
                                             const char* kernel_name,
                                             HipRemoteParamDesc* params, uint32_t max_params,
                                             uint32_t* out_kernarg_segment_size) {
    if (!load_comgr()) {
        if (g_debug_enabled) fprintf(stderr, "[HIP-Worker] COMGR: failed to load library\n");
        return 0;
    }

    /* The code object may be:
     * 1. Raw ELF (starts with \x7fELF)
     * 2. Uncompressed offload bundle (__CLANG_OFFLOAD_BUNDLE__)
     * 3. Compressed offload bundle (CCOB v2/v3)
     * We need to extract the raw ELF for COMGR. */
    const void* elf_data = code_data;
    size_t elf_size = code_size;
    void* decompressed = NULL;

    const uint8_t* bytes = (const uint8_t*)code_data;

    if (code_size >= 4 && memcmp(bytes, "CCOB", 4) == 0) {
        /* CCOB v3: magic(4) + version(2) + file_size(8) + compression(2) + uncompressed_size(8) + hash(8) + data */
        if (code_size >= 32) {
            uint64_t uncomp_size = 0;
            memcpy(&uncomp_size, bytes + 16, 8); /* uncompressed_size at offset 16 */
            if (uncomp_size > 0 && uncomp_size < 256 * 1024 * 1024) {
                /* Load zstd dynamically */
                typedef size_t (*pfn_ZSTD_decompress)(void*, size_t, const void*, size_t);
                typedef unsigned (*pfn_ZSTD_isError)(size_t);
                static pfn_ZSTD_decompress fn_decompress = NULL;
                static pfn_ZSTD_isError fn_isError = NULL;
                static int zstd_loaded = 0;
                if (!zstd_loaded) {
#ifdef _WIN32
                    void* zlib = (void*)LoadLibraryA("zstd.dll");
                    if (!zlib) zlib = (void*)LoadLibraryA("libzstd.dll");
                    if (zlib) {
                        fn_decompress = (pfn_ZSTD_decompress)(void*)GetProcAddress((HMODULE)zlib, "ZSTD_decompress");
                        fn_isError = (pfn_ZSTD_isError)(void*)GetProcAddress((HMODULE)zlib, "ZSTD_isError");
#else
                    void* zlib = dlopen("libzstd.so", RTLD_LAZY);
                    if (zlib) {
                        fn_decompress = (pfn_ZSTD_decompress)dlsym(zlib, "ZSTD_decompress");
                        fn_isError = (pfn_ZSTD_isError)dlsym(zlib, "ZSTD_isError");
#endif
                    }
                    zstd_loaded = 1;
                }
                if (fn_decompress && fn_isError) {
                    decompressed = malloc(uncomp_size);
                    if (decompressed) {
                        size_t res = fn_decompress(decompressed, uncomp_size, bytes + 32, code_size - 32);
                        if (!fn_isError(res)) {
                            elf_data = decompressed;
                            elf_size = res;
                            if (g_debug_enabled)
                                fprintf(stderr, "[HIP-Worker] COMGR: decompressed CCOB %zu -> %zu bytes\n", code_size, elf_size);
                        } else {
                            free(decompressed);
                            decompressed = NULL;
                        }
                    }
                }
            }
        }
    }

    /* If it's an offload bundle, collect all ELF entries so we can search
     * each one for the kernel (different kernels may be in different entries) */
    #define MAX_BUNDLE_ELFS 16
    struct { const void* data; size_t size; } bundle_elfs[MAX_BUNDLE_ELFS];
    int num_bundle_elfs = 0;

    const uint8_t* eb = (const uint8_t*)elf_data;
    if (elf_size >= 24 && memcmp(eb, "__CLANG_OFFLOAD_BUNDLE__", 24) == 0) {
        uint64_t num_entries = 0;
        memcpy(&num_entries, eb + 24, 8);
        size_t boffset = 32;
        for (uint64_t i = 0; i < num_entries && boffset + 24 <= elf_size; i++) {
            uint64_t entry_offset = 0, entry_size = 0, triple_size = 0;
            memcpy(&entry_offset, eb + boffset, 8);
            memcpy(&entry_size, eb + boffset + 8, 8);
            memcpy(&triple_size, eb + boffset + 16, 8);
            boffset += 24 + triple_size;

            if (entry_size > 4 && entry_offset + entry_size <= elf_size) {
                const uint8_t* entry = eb + entry_offset;
                if (entry[0] == 0x7f && entry[1] == 'E' && entry[2] == 'L' && entry[3] == 'F') {
                    if (num_bundle_elfs < MAX_BUNDLE_ELFS) {
                        bundle_elfs[num_bundle_elfs].data = entry;
                        bundle_elfs[num_bundle_elfs].size = entry_size;
                        num_bundle_elfs++;
                    }
                }
            }
        }
        if (num_bundle_elfs > 0) {
            elf_data = bundle_elfs[0].data;
            elf_size = bundle_elfs[0].size;
            if (g_debug_enabled)
                fprintf(stderr, "[HIP-Worker] COMGR: bundle has %d ELF entries\n", num_bundle_elfs);
        }
    } else {
        bundle_elfs[0].data = elf_data;
        bundle_elfs[0].size = elf_size;
        num_bundle_elfs = 1;
    }

    /* Try each ELF entry until we find the kernel */
    for (int elf_idx = 0; elf_idx < num_bundle_elfs; elf_idx++) {
    elf_data = bundle_elfs[elf_idx].data;
    elf_size = bundle_elfs[elf_idx].size;

    /* Now try COMGR with the raw ELF */
    amd_comgr_data_t co_data;
    amd_comgr_status_t st;
    st = g_comgr.create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &co_data);
    if (st != AMD_COMGR_STATUS_SUCCESS) {
        continue;
    }
    st = g_comgr.set_data(co_data, elf_size, (const char*)elf_data);
    if (st != AMD_COMGR_STATUS_SUCCESS) {
        g_comgr.release_data(co_data);
        continue;
    }

    amd_comgr_metadata_node_t md = {0};
    st = g_comgr.get_data_metadata(co_data, &md);
    if (st != AMD_COMGR_STATUS_SUCCESS) {
        if (g_debug_enabled) fprintf(stderr, "[HIP-Worker] COMGR: get_data_metadata failed: %d (elf_size=%zu)\n", st, elf_size);
        g_comgr.release_data(co_data);
        continue;
    }

    /* Look up "amdhsa.kernels" */
    amd_comgr_metadata_node_t kernels_md = {0};
    st = g_comgr.metadata_lookup(md, "amdhsa.kernels", &kernels_md);
    if (st != AMD_COMGR_STATUS_SUCCESS) {
        if (g_debug_enabled) fprintf(stderr, "[HIP-Worker] COMGR: metadata_lookup 'amdhsa.kernels' failed: %d\n", st);
        g_comgr.destroy_metadata(md);
        g_comgr.release_data(co_data);
        continue;
    }

    size_t num_kernels = 0;
    g_comgr.get_metadata_list_size(kernels_md, &num_kernels);

    uint32_t result = 0;

    for (size_t ki = 0; ki < num_kernels; ki++) {
        amd_comgr_metadata_node_t kernel_node = {0};
        if (g_comgr.index_list_metadata(kernels_md, ki, &kernel_node) != AMD_COMGR_STATUS_SUCCESS)
            continue;

        /* Check if this kernel matches the requested name.
         * The metadata ".symbol" field contains the mangled name with ".kd" suffix.
         * CK kernel symbols can be 2000+ chars so we use dynamic allocation. */
        char* sym = comgr_lookup_string(kernel_node, ".symbol");
        char* name_str = comgr_lookup_string(kernel_node, ".name");

        /* Strip ".kd" suffix from symbol if present */
        if (sym) {
            size_t slen = strlen(sym);
            if (slen > 3 && strcmp(sym + slen - 3, ".kd") == 0)
                sym[slen - 3] = '\0';
        }

        if (g_debug_enabled && ki < 3) {
            fprintf(stderr, "[HIP-Worker] COMGR: match check ki=%zu sym=%s name=%s target=%s\n",
                    ki, sym ? sym : "(null)", name_str ? name_str : "(null)", kernel_name);
        }

        int matched = (sym && strcmp(sym, kernel_name) == 0)
                   || (name_str && strcmp(name_str, kernel_name) == 0);
        free(sym);
        free(name_str);

        if (!matched) {
            g_comgr.destroy_metadata(kernel_node);
            continue;
        }

        /* Read .kernarg_segment_size from code object metadata */
        if (out_kernarg_segment_size) {
            char* ksize_str = comgr_lookup_string(kernel_node, ".kernarg_segment_size");
            *out_kernarg_segment_size = ksize_str ? (uint32_t)atoi(ksize_str) : 0;
            free(ksize_str);
        }

        /* Extract .args */
        amd_comgr_metadata_node_t args_md = {0};
        if (g_comgr.metadata_lookup(kernel_node, ".args", &args_md) != AMD_COMGR_STATUS_SUCCESS) {
            g_comgr.destroy_metadata(kernel_node);
            break;
        }

        size_t num_args = 0;
        g_comgr.get_metadata_list_size(args_md, &num_args);

        for (size_t ai = 0; ai < num_args && result < max_params; ai++) {
            amd_comgr_metadata_node_t arg_node = {0};
            if (g_comgr.index_list_metadata(args_md, ai, &arg_node) != AMD_COMGR_STATUS_SUCCESS)
                continue;

            char* val_kind = comgr_lookup_string(arg_node, ".value_kind");

            /* Skip hidden args (hidden_global_offset_x/y/z, etc.)
             * -- they're not passed via kernelParams */
            if (val_kind && strncmp(val_kind, "hidden_", 7) == 0) {
                free(val_kind);
                g_comgr.destroy_metadata(arg_node);
                continue;
            }

            int is_ptr = (val_kind && strcmp(val_kind, "global_buffer") == 0);

            char* off_str = comgr_lookup_string(arg_node, ".offset");
            char* sz_str = comgr_lookup_string(arg_node, ".size");

            params[result].offset = off_str ? (uint32_t)atoi(off_str) : 0;
            params[result].size = sz_str ? (uint32_t)atoi(sz_str) : 0;
            params[result].is_pointer = (uint8_t)is_ptr;
            memset(params[result]._pad, 0, sizeof(params[result]._pad));

            if (g_debug_enabled) {
                fprintf(stderr, "[HIP-Worker] COMGR:   arg[%zu]: off=%u size=%u value_kind=%s is_ptr=%d\n",
                        ai, params[result].offset, params[result].size,
                        val_kind ? val_kind : "(null)", is_ptr);
            }
            free(val_kind);
            free(off_str);
            free(sz_str);
            result++;

            g_comgr.destroy_metadata(arg_node);
        }

        g_comgr.destroy_metadata(args_md);
        g_comgr.destroy_metadata(kernel_node);
        break;
    }

    if (result == 0 && g_debug_enabled) {
        fprintf(stderr, "[HIP-Worker] COMGR: no match for '%s' in %zu kernels\n",
                kernel_name, num_kernels);
        for (size_t ki = 0; ki < num_kernels && ki < 3; ki++) {
            amd_comgr_metadata_node_t kn = {0};
            if (g_comgr.index_list_metadata(kernels_md, ki, &kn) == AMD_COMGR_STATUS_SUCCESS) {
                char* s = comgr_lookup_string(kn, ".symbol");
                char* n = comgr_lookup_string(kn, ".name");
                fprintf(stderr, "[HIP-Worker] COMGR:   kernel[%zu] symbol='%s' name='%s'\n",
                        ki, s ? s : "(null)", n ? n : "(null)");
                free(s);
                free(n);
                g_comgr.destroy_metadata(kn);
            }
        }
    }

    g_comgr.destroy_metadata(kernels_md);
    g_comgr.destroy_metadata(md);
    g_comgr.release_data(co_data);

    if (result > 0) {
        free(decompressed);
        return result;
    }
    } /* end for elf_idx */

    free(decompressed);
    return 0;
}

/* ============================================================================
 * Logging
 * ============================================================================ */

#define LOG_DEBUG(fmt, ...) do { \
    if (g_debug_enabled) { \
        fprintf(stderr, "[HIP-Worker] " fmt "\n", ##__VA_ARGS__); \
    } \
} while (0)

#define LOG_INFO(fmt, ...) \
    fprintf(stderr, "[HIP-Worker] " fmt "\n", ##__VA_ARGS__)

#define LOG_ERROR(fmt, ...) \
    fprintf(stderr, "[HIP-Worker ERROR] " fmt "\n", ##__VA_ARGS__)

/* ============================================================================
 * Network Helpers
 * ============================================================================ */

static int send_all(int fd, const void* data, size_t len) {
    const uint8_t* p = (const uint8_t*)data;
    while (len > 0) {
#ifdef _WIN32
        int chunk = len > 0x7FFFFFFF ? 0x7FFFFFFF : (int)len;
        int n = send(fd, (const char*)p, chunk, 0);
#elif defined(MSG_NOSIGNAL)
        ssize_t n = send(fd, p, len, MSG_NOSIGNAL);
#else
        ssize_t n = send(fd, p, len, 0);
#endif
        if (n < 0) {
#ifdef _WIN32
            int err = WSAGetLastError();
            if (err == WSAEINTR) continue;
#else
            if (errno == EINTR) continue;
#endif
            return -1;
        }
        if (n == 0) return -1;
        p += (size_t)n;
        len -= (size_t)n;
    }
    return 0;
}

static int recv_all(int fd, void* data, size_t len) {
    uint8_t* p = (uint8_t*)data;
    while (len > 0) {
#ifdef _WIN32
        int chunk = len > 0x7FFFFFFF ? 0x7FFFFFFF : (int)len;
        int n = recv(fd, (char*)p, chunk, 0);
#else
        ssize_t n = recv(fd, p, len, 0);
#endif
        if (n < 0) {
#ifdef _WIN32
            int err = WSAGetLastError();
            if (err == WSAEINTR) continue;
#else
            if (errno == EINTR) continue;
#endif
            return -1;
        }
        if (n == 0) return -1;
        p += (size_t)n;
        len -= (size_t)n;
    }
    return 0;
}

/* ============================================================================
 * Response Helpers
 * ============================================================================ */

static int g_suppress_response = 0;

static int send_response(int fd, HipRemoteOpCode op_code, uint32_t request_id,
                         const void* payload, size_t payload_size) {
    if (g_suppress_response) return 0;

    HipRemoteHeader header;
    hip_remote_init_header(&header, op_code, request_id, (uint32_t)payload_size);
    header.flags |= HIP_REMOTE_FLAG_RESPONSE;

    if (send_all(fd, &header, sizeof(header)) != 0) return -1;
    if (payload && payload_size > 0) {
        if (send_all(fd, payload, payload_size) != 0) return -1;
    }
    return 0;
}

static int send_simple_response(int fd, HipRemoteOpCode op_code,
                                uint32_t request_id, hipError_t err) {
    HipRemoteResponseHeader resp = { .error_code = (int32_t)err };
    return send_response(fd, op_code, request_id, &resp, sizeof(resp));
}

/* ============================================================================
 * Request Handlers
 * ============================================================================ */

static void handle_init(int fd, uint32_t request_id) {
    hipError_t err = hipSetDevice(g_default_device);
    LOG_DEBUG("Init: device=%d, err=%d", g_default_device, err);

    if (g_gpu_cache_enabled && err == hipSuccess) {
        int count = g_gpu_cache_count;
        size_t resp_size = sizeof(HipRemoteInitResponse) + count * sizeof(uint64_t);
        uint8_t* buf = (uint8_t*)malloc(resp_size);
        if (buf) {
            HipRemoteInitResponse* resp = (HipRemoteInitResponse*)buf;
            resp->header.error_code = (int32_t)err;
            resp->cache_epoch = (uint32_t)getpid();
            resp->cache_count = (uint32_t)count;
            uint64_t* hashes = (uint64_t*)(buf + sizeof(HipRemoteInitResponse));
            gpu_cache_get_hashes(hashes, count);
            LOG_DEBUG("Init: sending cache manifest with %d entries", count);
            send_response(fd, HIP_OP_INIT, request_id, buf, resp_size);
            free(buf);
            return;
        }
    }
    send_simple_response(fd, HIP_OP_INIT, request_id, err);
}

static void handle_shutdown(int fd, uint32_t request_id) {
    LOG_DEBUG("Shutdown");
    send_simple_response(fd, HIP_OP_SHUTDOWN, request_id, hipSuccess);
}

static void handle_get_device_count(int fd, uint32_t request_id) {
    int count = 0;
    hipError_t err = hipGetDeviceCount(&count);
    LOG_DEBUG("GetDeviceCount: count=%d, err=%d", count, err);

    HipRemoteDeviceCountResponse resp = {
        .header = { .error_code = (int32_t)err },
        .count = count
    };
    send_response(fd, HIP_OP_GET_DEVICE_COUNT, request_id, &resp, sizeof(resp));
}

static void handle_set_device(int fd, uint32_t request_id,
                              const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteDeviceRequest)) {
        send_simple_response(fd, HIP_OP_SET_DEVICE, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteDeviceRequest* req = (const HipRemoteDeviceRequest*)payload;
    hipError_t err = hipSetDevice(req->device_id);
    LOG_DEBUG("SetDevice: device=%d, err=%d", req->device_id, err);
    send_simple_response(fd, HIP_OP_SET_DEVICE, request_id, err);
}

static void handle_get_device(int fd, uint32_t request_id) {
    int device = 0;
    hipError_t err = hipGetDevice(&device);
    LOG_DEBUG("GetDevice: device=%d, err=%d", device, err);

    HipRemoteGetDeviceResponse resp = {
        .header = { .error_code = (int32_t)err },
        .device_id = device
    };
    send_response(fd, HIP_OP_GET_DEVICE, request_id, &resp, sizeof(resp));
}

static void handle_device_synchronize(int fd, uint32_t request_id) {
    hipError_t err = hipDeviceSynchronize();
    if (err == hipSuccess && g_deferred_alloc_error != hipSuccess) {
        err = g_deferred_alloc_error;
        g_deferred_alloc_error = hipSuccess;
    }
    LOG_DEBUG("DeviceSynchronize: err=%d", err);
    send_simple_response(fd, HIP_OP_DEVICE_SYNCHRONIZE, request_id, err);
}

static void handle_get_device_properties(int fd, uint32_t request_id,
                                         const void* payload, size_t payload_size) {
    int device = 0;
    if (payload && payload_size >= sizeof(HipRemoteDeviceRequest)) {
        const HipRemoteDeviceRequest* req = (const HipRemoteDeviceRequest*)payload;
        device = req->device_id;
    }

    hipDeviceProp_t props;
    memset(&props, 0, sizeof(props));
    hipError_t err = hipGetDeviceProperties(&props, device);
    LOG_DEBUG("GetDeviceProperties: device=%d, name=%s, err=%d",
              device, props.name, err);

    /* Send the raw hipDeviceProp_t directly -- both sides include the same
     * hip_runtime_api.h, so the struct layout matches. No fields are lost. */
    struct {
        HipRemoteResponseHeader header;
        hipDeviceProp_t props;
    } resp;
    memset(&resp, 0, sizeof(resp));
    resp.header.error_code = (int32_t)err;
    if (err == hipSuccess) {
        memcpy(&resp.props, &props, sizeof(hipDeviceProp_t));
    }
    send_response(fd, HIP_OP_GET_DEVICE_PROPERTIES, request_id, &resp, sizeof(resp));
}

static void handle_device_get_attribute(int fd, uint32_t request_id,
                                        const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteDeviceAttributeRequest)) {
        send_simple_response(fd, HIP_OP_DEVICE_GET_ATTRIBUTE, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteDeviceAttributeRequest* req = (const HipRemoteDeviceAttributeRequest*)payload;
    int value = 0;
    hipError_t err = hipDeviceGetAttribute(&value, (hipDeviceAttribute_t)req->attribute, req->device_id);
    LOG_DEBUG("DeviceGetAttribute: device=%d, attr=%d, value=%d, err=%d",
              req->device_id, req->attribute, value, err);

    HipRemoteDeviceAttributeResponse resp = {
        .header = { .error_code = (int32_t)err },
        .value = value
    };
    send_response(fd, HIP_OP_DEVICE_GET_ATTRIBUTE, request_id, &resp, sizeof(resp));
}

static void handle_malloc(int fd, uint32_t request_id,
                          const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteMallocRequest)) {
        send_simple_response(fd, HIP_OP_MALLOC, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteMallocRequest* req = (const HipRemoteMallocRequest*)payload;
    void* ptr = NULL;
    hipError_t err = hipMalloc(&ptr, req->size);
    LOG_DEBUG("Malloc: size=%lu, ptr=%p, err=%d", (unsigned long)req->size, ptr, err);

    HipRemoteMallocResponse resp = {
        .header = { .error_code = (int32_t)err },
        .device_ptr = (uint64_t)(uintptr_t)ptr
    };
    send_response(fd, HIP_OP_MALLOC, request_id, &resp, sizeof(resp));
}

static void handle_malloc_batch(int fd, uint32_t request_id,
                                const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteMallocBatchRequest)) {
        send_simple_response(fd, HIP_OP_MALLOC_BATCH, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteMallocBatchRequest* req = (const HipRemoteMallocBatchRequest*)payload;
    uint32_t count = req->count;
    if (count > HIP_REMOTE_MAX_BATCH_MALLOC) {
        send_simple_response(fd, HIP_OP_MALLOC_BATCH, request_id, hipErrorInvalidValue);
        return;
    }

    HipRemoteMallocBatchResponse resp;
    memset(&resp, 0, sizeof(resp));
    resp.count = count;

    hipError_t first_err = hipSuccess;
    for (uint32_t i = 0; i < count; i++) {
        void* ptr = NULL;
        hipError_t err = hipMalloc(&ptr, (size_t)req->sizes[i]);
        resp.ptrs[i] = (uint64_t)(uintptr_t)ptr;
        if (err != hipSuccess && first_err == hipSuccess) {
            first_err = err;
        }
        LOG_DEBUG("MallocBatch[%u/%u]: size=%lu ptr=%p err=%d",
                  i, count, (unsigned long)req->sizes[i], ptr, err);
    }

    resp.header.error_code = (int32_t)first_err;
    send_response(fd, HIP_OP_MALLOC_BATCH, request_id, &resp, sizeof(resp));
}

static void handle_free(int fd, uint32_t request_id,
                        const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteFreeRequest)) {
        send_simple_response(fd, HIP_OP_FREE, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteFreeRequest* req = (const HipRemoteFreeRequest*)payload;
    uint64_t real = vaddr_map_get(req->device_ptr);
    if (real == VADDR_ERROR) {
        vaddr_map_remove(req->device_ptr);
        send_simple_response(fd, HIP_OP_FREE, request_id, hipSuccess);
        return;
    }
    void* ptr = (void*)(uintptr_t)real;
    hipError_t err = hipFree(ptr);
    vaddr_map_remove(req->device_ptr);
    LOG_DEBUG("Free: vaddr=0x%lx ptr=%p, err=%d", (unsigned long)req->device_ptr, ptr, err);
    send_simple_response(fd, HIP_OP_FREE, request_id, err);
}

static void handle_malloc_host(int fd, uint32_t request_id,
                               const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteMallocRequest)) {
        send_simple_response(fd, HIP_OP_MALLOC_HOST, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteMallocRequest* req = (const HipRemoteMallocRequest*)payload;
    void* ptr = NULL;
    hipError_t err = hipHostMalloc(&ptr, req->size, 0);
    LOG_DEBUG("MallocHost: size=%lu, ptr=%p, err=%d", (unsigned long)req->size, ptr, err);

    HipRemoteMallocResponse resp = {
        .header = { .error_code = (int32_t)err },
        .device_ptr = (uint64_t)(uintptr_t)ptr
    };
    send_response(fd, HIP_OP_MALLOC_HOST, request_id, &resp, sizeof(resp));
}

static void handle_free_host(int fd, uint32_t request_id,
                             const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteFreeRequest)) {
        send_simple_response(fd, HIP_OP_FREE_HOST, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteFreeRequest* req = (const HipRemoteFreeRequest*)payload;
    void* ptr = (void*)(uintptr_t)req->device_ptr;
    hipError_t err = hipHostFree(ptr);
    LOG_DEBUG("FreeHost: ptr=%p, err=%d", ptr, err);
    send_simple_response(fd, HIP_OP_FREE_HOST, request_id, err);
}

static void handle_malloc_async(int fd, uint32_t request_id,
                                const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteMallocAsyncRequest)) {
        send_simple_response(fd, HIP_OP_MALLOC_ASYNC, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteMallocAsyncRequest* req = (const HipRemoteMallocAsyncRequest*)payload;
    void* ptr = NULL;
    hipStream_t stream = vaddr_translate(req->stream);
    hipError_t err = hipMallocAsync(&ptr, req->size, stream);
    LOG_DEBUG("MallocAsync: size=%lu, stream=%p, ptr=%p, err=%d",
              (unsigned long)req->size, stream, ptr, err);

    HipRemoteMallocResponse resp = {
        .header = { .error_code = (int32_t)err },
        .device_ptr = (uint64_t)(uintptr_t)ptr
    };
    send_response(fd, HIP_OP_MALLOC_ASYNC, request_id, &resp, sizeof(resp));
}

static void handle_malloc_vaddr(int fd, uint32_t request_id,
                                const void* payload, size_t payload_size,
                                int is_async) {
    if (!payload || payload_size < sizeof(HipRemoteMallocVaddrRequest)) {
        send_simple_response(fd, is_async ? HIP_OP_MALLOC_ASYNC_VADDR : HIP_OP_MALLOC_VADDR,
                             request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteMallocVaddrRequest* req = (const HipRemoteMallocVaddrRequest*)payload;
    void* ptr = NULL;
    hipError_t err;

    /* Retry transient allocation failures (GPU contention under high load) */
    for (int attempt = 0; attempt < 3; attempt++) {
        if (is_async && req->stream) {
            hipStream_t stream = vaddr_translate(req->stream);
            err = hipMallocAsync(&ptr, req->size, stream);
        } else {
            err = hipMalloc(&ptr, req->size);
        }
        if (err == hipSuccess && ptr) break;
        if (attempt < 2) {
            hipDeviceSynchronize();
#ifdef _WIN32
            Sleep(attempt + 1);
#else
            usleep(1000 * (attempt + 1));
#endif
        }
    }

    if (err == hipSuccess && ptr) {
        vaddr_map_put(req->vaddr, (uint64_t)(uintptr_t)ptr, req->size);
        LOG_DEBUG("MallocVaddr: vaddr=0x%lx size=%lu -> real=%p",
                  (unsigned long)req->vaddr, (unsigned long)req->size, ptr);
    } else {
        vaddr_map_put(req->vaddr, VADDR_ERROR, 0);
        g_deferred_alloc_error = err;
        fprintf(stderr, "[HIP-Worker ERROR] MallocVaddr: vaddr=0x%lx size=%lu FAILED err=%d\n",
                (unsigned long)req->vaddr, (unsigned long)req->size, err);
    }
    send_simple_response(fd, is_async ? HIP_OP_MALLOC_ASYNC_VADDR : HIP_OP_MALLOC_VADDR,
                         request_id, err);
}

static void handle_free_async(int fd, uint32_t request_id,
                              const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteFreeAsyncRequest)) {
        send_simple_response(fd, HIP_OP_FREE_ASYNC, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteFreeAsyncRequest* req = (const HipRemoteFreeAsyncRequest*)payload;
    void* ptr = vaddr_translate(req->device_ptr);
    hipStream_t stream = vaddr_translate(req->stream);
    hipError_t err = hipFreeAsync(ptr, stream);
    vaddr_map_remove(req->device_ptr);
    LOG_DEBUG("FreeAsync: ptr=%p, stream=%p, err=%d", ptr, stream, err);
    send_simple_response(fd, HIP_OP_FREE_ASYNC, request_id, err);
}

static void handle_memcpy(int fd, uint32_t request_id,
                          const void* payload, size_t payload_size,
                          bool has_inline_data) {
    if (!payload || payload_size < sizeof(HipRemoteMemcpyRequest)) {
        send_simple_response(fd, HIP_OP_MEMCPY, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteMemcpyRequest* req = (const HipRemoteMemcpyRequest*)payload;
    hipError_t err = hipSuccess;

    void* dst_real = vaddr_translate(req->dst);
    void* src_real = vaddr_translate(req->src);
    hipStream_t stream_real = vaddr_translate(req->stream);

    LOG_DEBUG("Memcpy: dst=%p(v=0x%lx), src=%p(v=0x%lx), size=%lu, kind=%d",
              dst_real, (unsigned long)req->dst, src_real, (unsigned long)req->src,
              (unsigned long)req->size, req->kind);

    if (req->kind == hipMemcpyHostToDevice && has_inline_data) {
        const uint8_t* data = (const uint8_t*)payload + sizeof(HipRemoteMemcpyRequest);
        size_t data_available = payload_size - sizeof(HipRemoteMemcpyRequest);

        if (data_available >= req->size) {
            err = hipMemcpy(dst_real, data, req->size, hipMemcpyHostToDevice);
        } else {
            err = hipErrorInvalidValue;
        }
        send_simple_response(fd, HIP_OP_MEMCPY, request_id, err);

    } else if (req->kind == hipMemcpyDeviceToHost) {
        void* buffer = malloc(req->size);
        if (!buffer) {
            send_simple_response(fd, HIP_OP_MEMCPY, request_id, hipErrorOutOfMemory);
            return;
        }

        if (stream_real) {
            err = hipMemcpyAsync(buffer, src_real, req->size,
                                 hipMemcpyDeviceToHost, stream_real);
            if (err == hipSuccess)
                err = hipStreamSynchronize(stream_real);
        } else {
            err = hipMemcpy(buffer, src_real, req->size, hipMemcpyDeviceToHost);
        }

        if (err == hipSuccess) {
            HipRemoteHeader header;
            hip_remote_init_header(&header, HIP_OP_MEMCPY, request_id,
                                   sizeof(HipRemoteMemcpyResponse) + req->size);
            header.flags |= HIP_REMOTE_FLAG_RESPONSE | HIP_REMOTE_FLAG_HAS_INLINE_DATA;

            HipRemoteMemcpyResponse resp = {
                .header = { .error_code = (int32_t)err }
            };

            send_all(fd, &header, sizeof(header));
            send_all(fd, &resp, sizeof(resp));
            send_all(fd, buffer, req->size);
        } else {
            send_simple_response(fd, HIP_OP_MEMCPY, request_id, err);
        }

        free(buffer);

    } else if (req->kind == hipMemcpyDeviceToDevice) {
        err = hipMemcpy(dst_real, src_real, req->size, hipMemcpyDeviceToDevice);
        send_simple_response(fd, HIP_OP_MEMCPY, request_id, err);

    } else {
        send_simple_response(fd, HIP_OP_MEMCPY, request_id, hipErrorInvalidValue);
    }
}

static void handle_memcpy_htod_cached(int fd, uint32_t request_id,
                                       const void* payload, size_t payload_size,
                                       bool has_inline_data) {
    if (!payload || payload_size < sizeof(HipRemoteMemcpyCachedRequest)) {
        send_simple_response(fd, HIP_OP_MEMCPY_HTOD_CACHED, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteMemcpyCachedRequest* req = (const HipRemoteMemcpyCachedRequest*)payload;
    void* dst = vaddr_translate(req->dst);
    hipError_t err = hipSuccess;

    if (has_inline_data) {
        const uint8_t* data = (const uint8_t*)payload + sizeof(HipRemoteMemcpyCachedRequest);
        size_t data_available = payload_size - sizeof(HipRemoteMemcpyCachedRequest);

        if (data_available >= req->size) {
            err = hipMemcpy(dst, data, req->size, hipMemcpyHostToDevice);
            if (err == hipSuccess && g_gpu_cache_enabled) {
                gpu_cache_store(req->hash, dst, req->size);
            }
        } else {
            LOG_DEBUG("Cached memcpy: data truncated (%zu < %lu)", data_available, (unsigned long)req->size);
            err = hipErrorInvalidValue;
        }
    } else {
        void* cached = gpu_cache_lookup(req->hash);
        if (cached) {
            err = hipMemcpy(dst, cached, req->size, hipMemcpyDeviceToDevice);
        } else {
            LOG_DEBUG("Cached memcpy: cache miss for hash=0x%llx, no data",
                      (unsigned long long)req->hash);
            err = hipErrorNotFound;
        }
    }

    send_simple_response(fd, HIP_OP_MEMCPY_HTOD_CACHED, request_id, err);
}

static void handle_memcpy2d(int fd, uint32_t request_id,
                            const void* payload, size_t payload_size,
                            bool has_inline_data, bool is_async) {
    if (!payload || payload_size < sizeof(HipRemoteMemcpy2DRequest)) {
        send_simple_response(fd, HIP_OP_MEMCPY_2D, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteMemcpy2DRequest* req = (const HipRemoteMemcpy2DRequest*)payload;
    hipError_t err = hipSuccess;
    hipStream_t stream = vaddr_translate(req->stream);
    void* dst_real = vaddr_translate(req->dst);
    void* src_real = vaddr_translate(req->src);

    LOG_DEBUG("Memcpy2D: dst=%p, dpitch=%lu, src=%p, spitch=%lu, width=%lu, height=%lu, kind=%d, stream=%p",
              dst_real, (unsigned long)req->dpitch,
              src_real, (unsigned long)req->spitch,
              (unsigned long)req->width, (unsigned long)req->height,
              req->kind, stream);

    if (req->kind == hipMemcpyHostToDevice && has_inline_data) {
        /* Inline data follows request struct */
        const uint8_t* data = (const uint8_t*)payload + sizeof(HipRemoteMemcpy2DRequest);
        size_t data_available = payload_size - sizeof(HipRemoteMemcpy2DRequest);
        size_t expected_size = req->spitch * req->height;

        if (data_available >= expected_size) {
            if (is_async) {
                err = hipMemcpy2DAsync(dst_real, req->dpitch,
                                       data, req->spitch, req->width, req->height,
                                       hipMemcpyHostToDevice, stream);
            } else {
                err = hipMemcpy2D(dst_real, req->dpitch,
                                  data, req->spitch, req->width, req->height,
                                  hipMemcpyHostToDevice);
            }
        } else {
            err = hipErrorInvalidValue;
        }
        send_simple_response(fd, is_async ? HIP_OP_MEMCPY_2D_ASYNC : HIP_OP_MEMCPY_2D, request_id, err);

    } else if (req->kind == hipMemcpyDeviceToHost) {
        /* Need to send data back */
        size_t total_size = req->dpitch * req->height;
        void* buffer = malloc(total_size);
        if (!buffer) {
            send_simple_response(fd, is_async ? HIP_OP_MEMCPY_2D_ASYNC : HIP_OP_MEMCPY_2D,
                                 request_id, hipErrorOutOfMemory);
            return;
        }

        if (is_async) {
            err = hipMemcpy2DAsync(buffer, req->dpitch,
                                   src_real, req->spitch,
                                   req->width, req->height, hipMemcpyDeviceToHost, stream);
            if (err == hipSuccess) {
                hipStreamSynchronize(stream);
            }
        } else {
            err = hipMemcpy2D(buffer, req->dpitch,
                              src_real, req->spitch,
                              req->width, req->height, hipMemcpyDeviceToHost);
        }

        if (err == hipSuccess) {
            HipRemoteHeader header;
            hip_remote_init_header(&header, is_async ? HIP_OP_MEMCPY_2D_ASYNC : HIP_OP_MEMCPY_2D,
                                   request_id, sizeof(HipRemoteMemcpyResponse) + total_size);
            header.flags |= HIP_REMOTE_FLAG_RESPONSE | HIP_REMOTE_FLAG_HAS_INLINE_DATA;

            HipRemoteMemcpyResponse resp = {
                .header = { .error_code = (int32_t)err }
            };

            send_all(fd, &header, sizeof(header));
            send_all(fd, &resp, sizeof(resp));
            send_all(fd, buffer, total_size);
        } else {
            send_simple_response(fd, is_async ? HIP_OP_MEMCPY_2D_ASYNC : HIP_OP_MEMCPY_2D,
                                 request_id, err);
        }

        free(buffer);

    } else if (req->kind == hipMemcpyDeviceToDevice) {
        if (is_async) {
            err = hipMemcpy2DAsync(dst_real, req->dpitch,
                                   src_real, req->spitch,
                                   req->width, req->height, hipMemcpyDeviceToDevice, stream);
        } else {
            err = hipMemcpy2D(dst_real, req->dpitch,
                              src_real, req->spitch,
                              req->width, req->height, hipMemcpyDeviceToDevice);
        }
        send_simple_response(fd, is_async ? HIP_OP_MEMCPY_2D_ASYNC : HIP_OP_MEMCPY_2D, request_id, err);

    } else {
        send_simple_response(fd, is_async ? HIP_OP_MEMCPY_2D_ASYNC : HIP_OP_MEMCPY_2D,
                             request_id, hipErrorInvalidValue);
    }
}

static void handle_memset(int fd, uint32_t request_id,
                          const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteMemsetRequest)) {
        send_simple_response(fd, HIP_OP_MEMSET, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteMemsetRequest* req = (const HipRemoteMemsetRequest*)payload;
    void* dst = vaddr_translate(req->dst);
    hipError_t err = hipMemset(dst, req->value, req->size);
    LOG_DEBUG("Memset: dst=%p, value=%d, size=%lu, err=%d",
              dst, req->value, (unsigned long)req->size, err);
    send_simple_response(fd, HIP_OP_MEMSET, request_id, err);
}

static void handle_mem_get_info(int fd, uint32_t request_id) {
    size_t free_bytes = 0, total_bytes = 0;
    hipError_t err = hipMemGetInfo(&free_bytes, &total_bytes);
    LOG_DEBUG("MemGetInfo: free=%lu, total=%lu, err=%d",
              (unsigned long)free_bytes, (unsigned long)total_bytes, err);

    HipRemoteMemGetInfoResponse resp = {
        .header = { .error_code = (int32_t)err },
        .free_bytes = free_bytes,
        .total_bytes = total_bytes
    };
    send_response(fd, HIP_OP_MEM_GET_INFO, request_id, &resp, sizeof(resp));
}

static void handle_pointer_get_attributes(int fd, uint32_t request_id,
                                          const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemotePointerGetAttributesRequest)) {
        send_simple_response(fd, HIP_OP_POINTER_GET_ATTRIBUTES, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemotePointerGetAttributesRequest* req = (const HipRemotePointerGetAttributesRequest*)payload;
    void* ptr = vaddr_translate(req->ptr);
    hipPointerAttribute_t attrs = {0};
    hipError_t err = hipPointerGetAttributes(&attrs, ptr);
    LOG_DEBUG("PointerGetAttributes: ptr=%p, type=%d, device=%d, err=%d",
              ptr, (int)attrs.type, attrs.device, err);

    HipRemotePointerGetAttributesResponse resp = {
        .header = { .error_code = (int32_t)err },
        .memory_type = (int32_t)attrs.type,
        .device = attrs.device,
        .device_pointer = (uint64_t)(uintptr_t)attrs.devicePointer,
        .host_pointer = (uint64_t)(uintptr_t)attrs.hostPointer,
        .is_managed = attrs.isManaged,
        .allocation_flags = attrs.allocationFlags
    };
    send_response(fd, HIP_OP_POINTER_GET_ATTRIBUTES, request_id, &resp, sizeof(resp));
}

/* ============================================================================
 * IPC Operations
 * ============================================================================ */

static void handle_ipc_get_mem_handle(int fd, uint32_t request_id,
                                      const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteIpcGetMemHandleRequest)) {
        send_simple_response(fd, HIP_OP_IPC_GET_MEM_HANDLE, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteIpcGetMemHandleRequest* req = (const HipRemoteIpcGetMemHandleRequest*)payload;
    void* devPtr = (void*)(uintptr_t)req->device_ptr;

    hipIpcMemHandle_t handle;
    memset(&handle, 0, sizeof(handle));
    hipError_t err = hipIpcGetMemHandle(&handle, devPtr);
    LOG_DEBUG("IpcGetMemHandle: devPtr=%p, err=%d", devPtr, err);

    HipRemoteIpcGetMemHandleResponse resp;
    resp.header.error_code = (int32_t)err;
    memcpy(resp.handle, &handle, HIP_REMOTE_IPC_HANDLE_SIZE);
    send_response(fd, HIP_OP_IPC_GET_MEM_HANDLE, request_id, &resp, sizeof(resp));
}

static void handle_ipc_open_mem_handle(int fd, uint32_t request_id,
                                       const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteIpcOpenMemHandleRequest)) {
        send_simple_response(fd, HIP_OP_IPC_OPEN_MEM_HANDLE, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteIpcOpenMemHandleRequest* req = (const HipRemoteIpcOpenMemHandleRequest*)payload;

    hipIpcMemHandle_t handle;
    memcpy(&handle, req->handle, HIP_REMOTE_IPC_HANDLE_SIZE);

    void* devPtr = NULL;
    hipError_t err = hipIpcOpenMemHandle(&devPtr, handle, req->flags);
    LOG_DEBUG("IpcOpenMemHandle: flags=%u, devPtr=%p, err=%d", req->flags, devPtr, err);

    HipRemoteIpcOpenMemHandleResponse resp = {
        .header = { .error_code = (int32_t)err },
        .device_ptr = (uint64_t)(uintptr_t)devPtr
    };
    send_response(fd, HIP_OP_IPC_OPEN_MEM_HANDLE, request_id, &resp, sizeof(resp));
}

static void handle_ipc_close_mem_handle(int fd, uint32_t request_id,
                                        const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteIpcCloseMemHandleRequest)) {
        send_simple_response(fd, HIP_OP_IPC_CLOSE_MEM_HANDLE, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteIpcCloseMemHandleRequest* req = (const HipRemoteIpcCloseMemHandleRequest*)payload;
    void* devPtr = (void*)(uintptr_t)req->device_ptr;

    hipError_t err = hipIpcCloseMemHandle(devPtr);
    LOG_DEBUG("IpcCloseMemHandle: devPtr=%p, err=%d", devPtr, err);

    send_simple_response(fd, HIP_OP_IPC_CLOSE_MEM_HANDLE, request_id, err);
}

static void handle_ipc_get_event_handle(int fd, uint32_t request_id,
                                        const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteIpcGetEventHandleRequest)) {
        send_simple_response(fd, HIP_OP_IPC_GET_EVENT_HANDLE, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteIpcGetEventHandleRequest* req = (const HipRemoteIpcGetEventHandleRequest*)payload;
    hipEvent_t event = (hipEvent_t)(uintptr_t)req->event;

    hipIpcEventHandle_t handle;
    memset(&handle, 0, sizeof(handle));
    hipError_t err = hipIpcGetEventHandle(&handle, event);
    LOG_DEBUG("IpcGetEventHandle: event=%p, err=%d", event, err);

    HipRemoteIpcGetEventHandleResponse resp;
    resp.header.error_code = (int32_t)err;
    memcpy(resp.handle, &handle, HIP_REMOTE_IPC_HANDLE_SIZE);
    send_response(fd, HIP_OP_IPC_GET_EVENT_HANDLE, request_id, &resp, sizeof(resp));
}

static void handle_ipc_open_event_handle(int fd, uint32_t request_id,
                                         const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteIpcOpenEventHandleRequest)) {
        send_simple_response(fd, HIP_OP_IPC_OPEN_EVENT_HANDLE, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteIpcOpenEventHandleRequest* req = (const HipRemoteIpcOpenEventHandleRequest*)payload;

    hipIpcEventHandle_t handle;
    memcpy(&handle, req->handle, HIP_REMOTE_IPC_HANDLE_SIZE);

    hipEvent_t event = NULL;
    hipError_t err = hipIpcOpenEventHandle(&event, handle);
    LOG_DEBUG("IpcOpenEventHandle: event=%p, err=%d", event, err);

    HipRemoteIpcOpenEventHandleResponse resp = {
        .header = { .error_code = (int32_t)err },
        .event = (uint64_t)(uintptr_t)event
    };
    send_response(fd, HIP_OP_IPC_OPEN_EVENT_HANDLE, request_id, &resp, sizeof(resp));
}

/* ============================================================================
 * Memory Pool Operations
 * ============================================================================ */

static void handle_mem_pool_create(int fd, uint32_t request_id,
                                   const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteMemPoolCreateRequest)) {
        send_simple_response(fd, HIP_OP_MEM_POOL_CREATE, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteMemPoolCreateRequest* req = (const HipRemoteMemPoolCreateRequest*)payload;

    hipMemPoolProps props;
    memset(&props, 0, sizeof(props));
    props.allocType = (hipMemAllocationType)req->alloc_type;
    props.handleTypes = (hipMemHandleType)req->handle_types;
    props.location.type = (hipMemLocationType)req->location_type;
    props.location.id = req->location_id;
    props.maxSize = req->max_size;

    hipMemPool_t memPool = NULL;
    hipError_t err = hipMemPoolCreate(&memPool, &props);
    LOG_DEBUG("MemPoolCreate: allocType=%d, device=%d, pool=%p, err=%d",
              req->alloc_type, req->location_id, memPool, err);

    HipRemoteMemPoolCreateResponse resp = {
        .header = { .error_code = (int32_t)err },
        .mem_pool = (uint64_t)(uintptr_t)memPool
    };
    send_response(fd, HIP_OP_MEM_POOL_CREATE, request_id, &resp, sizeof(resp));
}

static void handle_mem_pool_destroy(int fd, uint32_t request_id,
                                    const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteMemPoolDestroyRequest)) {
        send_simple_response(fd, HIP_OP_MEM_POOL_DESTROY, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteMemPoolDestroyRequest* req = (const HipRemoteMemPoolDestroyRequest*)payload;
    hipMemPool_t memPool = (hipMemPool_t)(uintptr_t)req->mem_pool;

    hipError_t err = hipMemPoolDestroy(memPool);
    LOG_DEBUG("MemPoolDestroy: pool=%p, err=%d", memPool, err);

    send_simple_response(fd, HIP_OP_MEM_POOL_DESTROY, request_id, err);
}

static void handle_mem_pool_set_attribute(int fd, uint32_t request_id,
                                          const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteMemPoolSetAttributeRequest)) {
        send_simple_response(fd, HIP_OP_MEM_POOL_SET_ATTRIBUTE, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteMemPoolSetAttributeRequest* req = (const HipRemoteMemPoolSetAttributeRequest*)payload;
    hipMemPool_t memPool = (hipMemPool_t)(uintptr_t)req->mem_pool;
    hipMemPoolAttr attr = (hipMemPoolAttr)req->attr;
    uint64_t value = req->value;

    hipError_t err = hipMemPoolSetAttribute(memPool, attr, &value);
    LOG_DEBUG("MemPoolSetAttribute: pool=%p, attr=%d, value=%lu, err=%d",
              memPool, attr, (unsigned long)value, err);

    send_simple_response(fd, HIP_OP_MEM_POOL_SET_ATTRIBUTE, request_id, err);
}

static void handle_mem_pool_get_attribute(int fd, uint32_t request_id,
                                          const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteMemPoolGetAttributeRequest)) {
        send_simple_response(fd, HIP_OP_MEM_POOL_GET_ATTRIBUTE, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteMemPoolGetAttributeRequest* req = (const HipRemoteMemPoolGetAttributeRequest*)payload;
    hipMemPool_t memPool = (hipMemPool_t)(uintptr_t)req->mem_pool;
    hipMemPoolAttr attr = (hipMemPoolAttr)req->attr;
    uint64_t value = 0;

    hipError_t err = hipMemPoolGetAttribute(memPool, attr, &value);
    LOG_DEBUG("MemPoolGetAttribute: pool=%p, attr=%d, value=%lu, err=%d",
              memPool, attr, (unsigned long)value, err);

    HipRemoteMemPoolGetAttributeResponse resp = {
        .header = { .error_code = (int32_t)err },
        .value = value
    };
    send_response(fd, HIP_OP_MEM_POOL_GET_ATTRIBUTE, request_id, &resp, sizeof(resp));
}

static void handle_malloc_from_pool_async(int fd, uint32_t request_id,
                                          const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteMallocFromPoolAsyncRequest)) {
        send_simple_response(fd, HIP_OP_MALLOC_FROM_POOL_ASYNC, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteMallocFromPoolAsyncRequest* req = (const HipRemoteMallocFromPoolAsyncRequest*)payload;
    hipMemPool_t memPool = (hipMemPool_t)(uintptr_t)req->mem_pool;
    hipStream_t stream = vaddr_translate(req->stream);

    void* devPtr = NULL;
    hipError_t err = hipMallocFromPoolAsync(&devPtr, req->size, memPool, stream);
    LOG_DEBUG("MallocFromPoolAsync: size=%lu, pool=%p, stream=%p, ptr=%p, err=%d",
              (unsigned long)req->size, memPool, stream, devPtr, err);

    HipRemoteMallocResponse resp = {
        .header = { .error_code = (int32_t)err },
        .device_ptr = (uint64_t)(uintptr_t)devPtr
    };
    send_response(fd, HIP_OP_MALLOC_FROM_POOL_ASYNC, request_id, &resp, sizeof(resp));
}

static void handle_mem_pool_trim_to(int fd, uint32_t request_id,
                                    const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteMemPoolTrimToRequest)) {
        send_simple_response(fd, HIP_OP_MEM_POOL_TRIM_TO, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteMemPoolTrimToRequest* req = (const HipRemoteMemPoolTrimToRequest*)payload;
    hipMemPool_t memPool = (hipMemPool_t)(uintptr_t)req->mem_pool;

    hipError_t err = hipMemPoolTrimTo(memPool, req->min_bytes_to_hold);
    LOG_DEBUG("MemPoolTrimTo: pool=%p, minBytes=%lu, err=%d",
              memPool, (unsigned long)req->min_bytes_to_hold, err);

    send_simple_response(fd, HIP_OP_MEM_POOL_TRIM_TO, request_id, err);
}

static void handle_device_get_default_mem_pool(int fd, uint32_t request_id,
                                               const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteDeviceGetMemPoolRequest)) {
        send_simple_response(fd, HIP_OP_DEVICE_GET_DEFAULT_MEM_POOL, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteDeviceGetMemPoolRequest* req = (const HipRemoteDeviceGetMemPoolRequest*)payload;

    hipMemPool_t memPool = NULL;
    hipError_t err = hipDeviceGetDefaultMemPool(&memPool, req->device);
    LOG_DEBUG("DeviceGetDefaultMemPool: device=%d, pool=%p, err=%d",
              req->device, memPool, err);

    HipRemoteDeviceGetMemPoolResponse resp = {
        .header = { .error_code = (int32_t)err },
        .mem_pool = (uint64_t)(uintptr_t)memPool
    };
    send_response(fd, HIP_OP_DEVICE_GET_DEFAULT_MEM_POOL, request_id, &resp, sizeof(resp));
}

static void handle_device_set_mem_pool(int fd, uint32_t request_id,
                                       const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteDeviceSetMemPoolRequest)) {
        send_simple_response(fd, HIP_OP_DEVICE_SET_MEM_POOL, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteDeviceSetMemPoolRequest* req = (const HipRemoteDeviceSetMemPoolRequest*)payload;
    hipMemPool_t memPool = (hipMemPool_t)(uintptr_t)req->mem_pool;

    hipError_t err = hipDeviceSetMemPool(req->device, memPool);
    LOG_DEBUG("DeviceSetMemPool: device=%d, pool=%p, err=%d",
              req->device, memPool, err);

    send_simple_response(fd, HIP_OP_DEVICE_SET_MEM_POOL, request_id, err);
}

static void handle_device_get_mem_pool(int fd, uint32_t request_id,
                                       const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteDeviceGetMemPoolRequest)) {
        send_simple_response(fd, HIP_OP_DEVICE_GET_MEM_POOL, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteDeviceGetMemPoolRequest* req = (const HipRemoteDeviceGetMemPoolRequest*)payload;

    hipMemPool_t memPool = NULL;
    hipError_t err = hipDeviceGetMemPool(&memPool, req->device);
    LOG_DEBUG("DeviceGetMemPool: device=%d, pool=%p, err=%d",
              req->device, memPool, err);

    HipRemoteDeviceGetMemPoolResponse resp = {
        .header = { .error_code = (int32_t)err },
        .mem_pool = (uint64_t)(uintptr_t)memPool
    };
    send_response(fd, HIP_OP_DEVICE_GET_MEM_POOL, request_id, &resp, sizeof(resp));
}


/* ============================================================================
 * Host Memory Registration Handlers
 * ============================================================================ */

static void handle_host_register(int fd, uint32_t request_id,
                                 const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteHostRegisterRequest)) {
        send_simple_response(fd, HIP_OP_HOST_REGISTER, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteHostRegisterRequest* req = (const HipRemoteHostRegisterRequest*)payload;
    void* hostPtr = (void*)(uintptr_t)req->host_ptr;
    hipError_t err = hipHostRegister(hostPtr, req->size_bytes, req->flags);
    LOG_DEBUG("HostRegister: ptr=%p, size=%lu, flags=%u, err=%d",
              hostPtr, req->size_bytes, req->flags, err);
    send_simple_response(fd, HIP_OP_HOST_REGISTER, request_id, err);
}

static void handle_host_unregister(int fd, uint32_t request_id,
                                   const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteHostUnregisterRequest)) {
        send_simple_response(fd, HIP_OP_HOST_UNREGISTER, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteHostUnregisterRequest* req = (const HipRemoteHostUnregisterRequest*)payload;
    void* hostPtr = (void*)(uintptr_t)req->host_ptr;
    hipError_t err = hipHostUnregister(hostPtr);
    LOG_DEBUG("HostUnregister: ptr=%p, err=%d", hostPtr, err);
    send_simple_response(fd, HIP_OP_HOST_UNREGISTER, request_id, err);
}

static void handle_host_get_device_pointer(int fd, uint32_t request_id,
                                           const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteHostGetDevicePointerRequest)) {
        send_simple_response(fd, HIP_OP_HOST_GET_DEVICE_POINTER, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteHostGetDevicePointerRequest* req = (const HipRemoteHostGetDevicePointerRequest*)payload;
    void* hostPtr = (void*)(uintptr_t)req->host_ptr;
    void* devPtr = NULL;
    hipError_t err = hipHostGetDevicePointer(&devPtr, hostPtr, req->flags);
    LOG_DEBUG("HostGetDevicePointer: host=%p, dev=%p, err=%d", hostPtr, devPtr, err);

    HipRemoteHostGetDevicePointerResponse resp = {
        .header = { .error_code = (int32_t)err },
        .device_ptr = (uint64_t)(uintptr_t)devPtr
    };
    send_response(fd, HIP_OP_HOST_GET_DEVICE_POINTER, request_id, &resp, sizeof(resp));
}

static void handle_host_get_flags(int fd, uint32_t request_id,
                                  const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteHostGetFlagsRequest)) {
        send_simple_response(fd, HIP_OP_HOST_GET_FLAGS, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteHostGetFlagsRequest* req = (const HipRemoteHostGetFlagsRequest*)payload;
    void* hostPtr = (void*)(uintptr_t)req->host_ptr;
    unsigned int flags = 0;
    hipError_t err = hipHostGetFlags(&flags, hostPtr);
    LOG_DEBUG("HostGetFlags: ptr=%p, flags=%u, err=%d", hostPtr, flags, err);

    HipRemoteHostGetFlagsResponse resp = {
        .header = { .error_code = (int32_t)err },
        .flags = flags
    };
    send_response(fd, HIP_OP_HOST_GET_FLAGS, request_id, &resp, sizeof(resp));
}

static void handle_host_alloc(int fd, uint32_t request_id,
                              const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteHostAllocRequest)) {
        send_simple_response(fd, HIP_OP_HOST_ALLOC, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteHostAllocRequest* req = (const HipRemoteHostAllocRequest*)payload;
    void* ptr = NULL;
    hipError_t err = hipHostAlloc(&ptr, req->size, req->flags);
    LOG_DEBUG("HostAlloc: size=%lu, flags=%u, ptr=%p, err=%d",
              req->size, req->flags, ptr, err);

    HipRemoteHostAllocResponse resp = {
        .header = { .error_code = (int32_t)err },
        .ptr = (uint64_t)(uintptr_t)ptr
    };
    send_response(fd, HIP_OP_HOST_ALLOC, request_id, &resp, sizeof(resp));
}

static void handle_host_free(int fd, uint32_t request_id,
                             const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteHostFreeRequest)) {
        send_simple_response(fd, HIP_OP_HOST_FREE, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteHostFreeRequest* req = (const HipRemoteHostFreeRequest*)payload;
    void* ptr = (void*)(uintptr_t)req->ptr;
    hipError_t err = hipHostFree(ptr);
    LOG_DEBUG("HostFree: ptr=%p, err=%d", ptr, err);
    send_simple_response(fd, HIP_OP_HOST_FREE, request_id, err);
}

static void handle_mem_alloc_pitch(int fd, uint32_t request_id,
                                   const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteMemAllocPitchRequest)) {
        send_simple_response(fd, HIP_OP_MEM_ALLOC_PITCH, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteMemAllocPitchRequest* req = (const HipRemoteMemAllocPitchRequest*)payload;
    hipDeviceptr_t dptr;
    size_t pitch;
    hipError_t err = hipMemAllocPitch(&dptr, &pitch, req->width_in_bytes,
                                      req->height, req->element_size);
    LOG_DEBUG("MemAllocPitch: width=%lu, height=%lu, elem=%u, pitch=%lu, ptr=%p, err=%d",
              req->width_in_bytes, req->height, req->element_size, pitch, (void*)dptr, err);

    HipRemoteMemAllocPitchResponse resp = {
        .header = { .error_code = (int32_t)err },
        .dptr = (uint64_t)dptr,
        .pitch = pitch
    };
    send_response(fd, HIP_OP_MEM_ALLOC_PITCH, request_id, &resp, sizeof(resp));
}

/* ============================================================================
 * Unified Memory Management Handlers
 * ============================================================================ */

static void handle_mem_advise(int fd, uint32_t request_id,
                              const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteMemAdviseRequest)) {
        send_simple_response(fd, HIP_OP_MEM_ADVISE, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteMemAdviseRequest* req = (const HipRemoteMemAdviseRequest*)payload;
    const void* devPtr = (const void*)(uintptr_t)req->dev_ptr;
    hipError_t err = hipMemAdvise(devPtr, req->count, (hipMemoryAdvise)req->advice, req->device);
    LOG_DEBUG("MemAdvise: ptr=%p, count=%lu, advice=%d, device=%d, err=%d",
              devPtr, req->count, req->advice, req->device, err);
    send_simple_response(fd, HIP_OP_MEM_ADVISE, request_id, err);
}

static void handle_mem_prefetch_async(int fd, uint32_t request_id,
                                      const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteMemPrefetchAsyncRequest)) {
        send_simple_response(fd, HIP_OP_MEM_PREFETCH_ASYNC, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteMemPrefetchAsyncRequest* req = (const HipRemoteMemPrefetchAsyncRequest*)payload;
    const void* devPtr = (const void*)(uintptr_t)req->dev_ptr;
    hipStream_t stream = vaddr_translate(req->stream);
    hipError_t err = hipMemPrefetchAsync(devPtr, req->count, req->device, stream);
    LOG_DEBUG("MemPrefetchAsync: ptr=%p, count=%lu, device=%d, stream=%p, err=%d",
              devPtr, req->count, req->device, stream, err);
    send_simple_response(fd, HIP_OP_MEM_PREFETCH_ASYNC, request_id, err);
}

static void handle_mem_range_get_attribute(int fd, uint32_t request_id,
                                           const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteMemRangeGetAttributeRequest)) {
        send_simple_response(fd, HIP_OP_MEM_RANGE_GET_ATTRIBUTE, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteMemRangeGetAttributeRequest* req = (const HipRemoteMemRangeGetAttributeRequest*)payload;
    const void* devPtr = (const void*)(uintptr_t)req->dev_ptr;

    /* Allocate buffer for attribute data */
    void* data = malloc(req->data_size);
    if (!data) {
        send_simple_response(fd, HIP_OP_MEM_RANGE_GET_ATTRIBUTE, request_id, hipErrorOutOfMemory);
        return;
    }

    hipError_t err = hipMemRangeGetAttribute(data, req->data_size,
                                             (hipMemRangeAttribute)req->attribute,
                                             devPtr, req->count);
    LOG_DEBUG("MemRangeGetAttribute: ptr=%p, count=%lu, attr=%d, size=%lu, err=%d",
              devPtr, req->count, req->attribute, req->data_size, err);

    /* Send response with data */
    size_t resp_size = sizeof(HipRemoteMemRangeGetAttributeResponse) + req->data_size;
    uint8_t* resp_buf = (uint8_t*)malloc(resp_size);
    if (!resp_buf) {
        free(data);
        send_simple_response(fd, HIP_OP_MEM_RANGE_GET_ATTRIBUTE, request_id, hipErrorOutOfMemory);
        return;
    }

    HipRemoteMemRangeGetAttributeResponse* resp = (HipRemoteMemRangeGetAttributeResponse*)resp_buf;
    resp->header.error_code = (int32_t)err;
    if (err == hipSuccess) {
        memcpy(resp_buf + sizeof(HipRemoteMemRangeGetAttributeResponse), data, req->data_size);
    }

    send_response(fd, HIP_OP_MEM_RANGE_GET_ATTRIBUTE, request_id, resp_buf, resp_size);
    free(data);
    free(resp_buf);
}

static void handle_mem_range_get_attributes(int fd, uint32_t request_id,
                                            const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteMemRangeGetAttributesRequest)) {
        send_simple_response(fd, HIP_OP_MEM_RANGE_GET_ATTRIBUTES, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteMemRangeGetAttributesRequest* req = (const HipRemoteMemRangeGetAttributesRequest*)payload;

    if (req->num_attributes == 0 || req->num_attributes > HIP_REMOTE_MAX_MEM_RANGE_ATTRIBUTES) {
        send_simple_response(fd, HIP_OP_MEM_RANGE_GET_ATTRIBUTES, request_id, hipErrorInvalidValue);
        return;
    }

    const void* devPtr = (const void*)(uintptr_t)req->dev_ptr;
    const uint8_t* req_data = (const uint8_t*)payload + sizeof(HipRemoteMemRangeGetAttributesRequest);

    /* Parse attributes and data sizes from request */
    const int32_t* attributes = (const int32_t*)req_data;
    const uint64_t* data_sizes = (const uint64_t*)(attributes + req->num_attributes);

    /* Allocate arrays for hipMemRangeGetAttributes */
    void** data_ptrs = (void**)malloc(req->num_attributes * sizeof(void*));
    size_t* sizes = (size_t*)malloc(req->num_attributes * sizeof(size_t));
    hipMemRangeAttribute* attrs = (hipMemRangeAttribute*)malloc(req->num_attributes * sizeof(hipMemRangeAttribute));

    if (!data_ptrs || !sizes || !attrs) {
        free(data_ptrs);
        free(sizes);
        free(attrs);
        send_simple_response(fd, HIP_OP_MEM_RANGE_GET_ATTRIBUTES, request_id, hipErrorOutOfMemory);
        return;
    }

    /* Allocate data buffers */
    size_t total_size = 0;
    for (size_t i = 0; i < req->num_attributes; i++) {
        sizes[i] = (size_t)data_sizes[i];
        attrs[i] = (hipMemRangeAttribute)attributes[i];
        data_ptrs[i] = malloc(sizes[i]);
        if (!data_ptrs[i]) {
            for (size_t j = 0; j < i; j++) free(data_ptrs[j]);
            free(data_ptrs);
            free(sizes);
            free(attrs);
            send_simple_response(fd, HIP_OP_MEM_RANGE_GET_ATTRIBUTES, request_id, hipErrorOutOfMemory);
            return;
        }
        total_size += sizes[i];
    }

    hipError_t err = hipMemRangeGetAttributes(data_ptrs, sizes, attrs,
                                              req->num_attributes, devPtr, req->count);
    LOG_DEBUG("MemRangeGetAttributes: ptr=%p, count=%lu, num_attrs=%u, err=%d",
              devPtr, req->count, req->num_attributes, err);

    /* Build response with all attribute data */
    size_t resp_size = sizeof(HipRemoteMemRangeGetAttributesResponse) + total_size;
    uint8_t* resp_buf = (uint8_t*)malloc(resp_size);
    if (!resp_buf) {
        for (size_t i = 0; i < req->num_attributes; i++) free(data_ptrs[i]);
        free(data_ptrs);
        free(sizes);
        free(attrs);
        send_simple_response(fd, HIP_OP_MEM_RANGE_GET_ATTRIBUTES, request_id, hipErrorOutOfMemory);
        return;
    }

    HipRemoteMemRangeGetAttributesResponse* resp = (HipRemoteMemRangeGetAttributesResponse*)resp_buf;
    resp->header.error_code = (int32_t)err;

    if (err == hipSuccess) {
        uint8_t* data_ptr = resp_buf + sizeof(HipRemoteMemRangeGetAttributesResponse);
        for (size_t i = 0; i < req->num_attributes; i++) {
            memcpy(data_ptr, data_ptrs[i], sizes[i]);
            data_ptr += sizes[i];
        }
    }

    send_response(fd, HIP_OP_MEM_RANGE_GET_ATTRIBUTES, request_id, resp_buf, resp_size);

    for (size_t i = 0; i < req->num_attributes; i++) free(data_ptrs[i]);
    free(data_ptrs);
    free(sizes);
    free(attrs);
    free(resp_buf);
}

/* ============================================================================
 * Graph Node Operations
 * ============================================================================ */

static void handle_graph_add_memcpy_node_1d(int fd, uint32_t request_id,
                                            const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteGraphAddMemcpyNode1DRequest)) {
        send_simple_response(fd, HIP_OP_GRAPH_ADD_MEMCPY_NODE_1D, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteGraphAddMemcpyNode1DRequest* req = (const HipRemoteGraphAddMemcpyNode1DRequest*)payload;
    hipGraph_t graph = (hipGraph_t)(uintptr_t)req->graph;

    /* Extract dependencies */
    size_t deps_offset = sizeof(HipRemoteGraphAddMemcpyNode1DRequest);
    const uint64_t* dep_handles = (const uint64_t*)((const uint8_t*)payload + deps_offset);
    hipGraphNode_t deps[HIP_REMOTE_MAX_GRAPH_DEPENDENCIES];
    size_t num_deps = req->num_deps;
    if (num_deps > HIP_REMOTE_MAX_GRAPH_DEPENDENCIES) num_deps = HIP_REMOTE_MAX_GRAPH_DEPENDENCIES;
    for (size_t i = 0; i < num_deps; i++) {
        deps[i] = (hipGraphNode_t)(uintptr_t)dep_handles[i];
    }

    hipGraphNode_t node = NULL;
    hipError_t err = hipGraphAddMemcpyNode1D(&node, graph,
                                              num_deps > 0 ? deps : NULL, num_deps,
                                              (void*)(uintptr_t)req->dst,
                                              (const void*)(uintptr_t)req->src,
                                              req->count, (hipMemcpyKind)req->kind);
    LOG_DEBUG("GraphAddMemcpyNode1D: graph=%p, count=%lu, node=%p, err=%d",
              graph, (unsigned long)req->count, node, err);

    HipRemoteGraphAddNodeResponse resp = {
        .header = { .error_code = (int32_t)err },
        .node = (uint64_t)(uintptr_t)node
    };
    send_response(fd, HIP_OP_GRAPH_ADD_MEMCPY_NODE_1D, request_id, &resp, sizeof(resp));
}

static void handle_graph_add_memset_node(int fd, uint32_t request_id,
                                         const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteGraphAddMemsetNodeRequest)) {
        send_simple_response(fd, HIP_OP_GRAPH_ADD_MEMSET_NODE, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteGraphAddMemsetNodeRequest* req = (const HipRemoteGraphAddMemsetNodeRequest*)payload;
    hipGraph_t graph = (hipGraph_t)(uintptr_t)req->graph;

    /* Extract dependencies */
    size_t deps_offset = sizeof(HipRemoteGraphAddMemsetNodeRequest);
    const uint64_t* dep_handles = (const uint64_t*)((const uint8_t*)payload + deps_offset);
    hipGraphNode_t deps[HIP_REMOTE_MAX_GRAPH_DEPENDENCIES];
    size_t num_deps = req->num_deps;
    if (num_deps > HIP_REMOTE_MAX_GRAPH_DEPENDENCIES) num_deps = HIP_REMOTE_MAX_GRAPH_DEPENDENCIES;
    for (size_t i = 0; i < num_deps; i++) {
        deps[i] = (hipGraphNode_t)(uintptr_t)dep_handles[i];
    }

    hipMemsetParams params = {
        .dst = (void*)(uintptr_t)req->dst,
        .pitch = req->pitch,
        .value = (unsigned int)req->value,
        .elementSize = req->element_size,
        .width = req->width,
        .height = req->height
    };

    hipGraphNode_t node = NULL;
    hipError_t err = hipGraphAddMemsetNode(&node, graph,
                                            num_deps > 0 ? deps : NULL, num_deps,
                                            &params);
    LOG_DEBUG("GraphAddMemsetNode: graph=%p, dst=%p, value=%d, node=%p, err=%d",
              graph, params.dst, params.value, node, err);

    HipRemoteGraphAddNodeResponse resp = {
        .header = { .error_code = (int32_t)err },
        .node = (uint64_t)(uintptr_t)node
    };
    send_response(fd, HIP_OP_GRAPH_ADD_MEMSET_NODE, request_id, &resp, sizeof(resp));
}

static void handle_graph_add_empty_node(int fd, uint32_t request_id,
                                        const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteGraphAddEmptyNodeRequest)) {
        send_simple_response(fd, HIP_OP_GRAPH_ADD_EMPTY_NODE, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteGraphAddEmptyNodeRequest* req = (const HipRemoteGraphAddEmptyNodeRequest*)payload;
    hipGraph_t graph = (hipGraph_t)(uintptr_t)req->graph;

    /* Extract dependencies */
    size_t deps_offset = sizeof(HipRemoteGraphAddEmptyNodeRequest);
    const uint64_t* dep_handles = (const uint64_t*)((const uint8_t*)payload + deps_offset);
    hipGraphNode_t deps[HIP_REMOTE_MAX_GRAPH_DEPENDENCIES];
    size_t num_deps = req->num_deps;
    if (num_deps > HIP_REMOTE_MAX_GRAPH_DEPENDENCIES) num_deps = HIP_REMOTE_MAX_GRAPH_DEPENDENCIES;
    for (size_t i = 0; i < num_deps; i++) {
        deps[i] = (hipGraphNode_t)(uintptr_t)dep_handles[i];
    }

    hipGraphNode_t node = NULL;
    hipError_t err = hipGraphAddEmptyNode(&node, graph,
                                           num_deps > 0 ? deps : NULL, num_deps);
    LOG_DEBUG("GraphAddEmptyNode: graph=%p, node=%p, err=%d", graph, node, err);

    HipRemoteGraphAddNodeResponse resp = {
        .header = { .error_code = (int32_t)err },
        .node = (uint64_t)(uintptr_t)node
    };
    send_response(fd, HIP_OP_GRAPH_ADD_EMPTY_NODE, request_id, &resp, sizeof(resp));
}

static void handle_graph_add_dependencies(int fd, uint32_t request_id,
                                          const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteGraphAddDependenciesRequest)) {
        send_simple_response(fd, HIP_OP_GRAPH_ADD_DEPENDENCIES, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteGraphAddDependenciesRequest* req = (const HipRemoteGraphAddDependenciesRequest*)payload;
    hipGraph_t graph = (hipGraph_t)(uintptr_t)req->graph;

    /* Extract from/to pairs */
    size_t pairs_offset = sizeof(HipRemoteGraphAddDependenciesRequest);
    const uint64_t* pairs = (const uint64_t*)((const uint8_t*)payload + pairs_offset);

    hipGraphNode_t* from = (hipGraphNode_t*)malloc(req->num_deps * sizeof(hipGraphNode_t));
    hipGraphNode_t* to = (hipGraphNode_t*)malloc(req->num_deps * sizeof(hipGraphNode_t));
    if (!from || !to) {
        free(from);
        free(to);
        send_simple_response(fd, HIP_OP_GRAPH_ADD_DEPENDENCIES, request_id, hipErrorOutOfMemory);
        return;
    }

    for (uint32_t i = 0; i < req->num_deps; i++) {
        from[i] = (hipGraphNode_t)(uintptr_t)pairs[i * 2];
        to[i] = (hipGraphNode_t)(uintptr_t)pairs[i * 2 + 1];
    }

    hipError_t err = hipGraphAddDependencies(graph, from, to, req->num_deps);
    LOG_DEBUG("GraphAddDependencies: graph=%p, num=%u, err=%d", graph, req->num_deps, err);

    free(from);
    free(to);
    send_simple_response(fd, HIP_OP_GRAPH_ADD_DEPENDENCIES, request_id, err);
}

static void handle_graph_get_nodes(int fd, uint32_t request_id,
                                   const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteGraphGetNodesRequest)) {
        send_simple_response(fd, HIP_OP_GRAPH_GET_NODES, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteGraphGetNodesRequest* req = (const HipRemoteGraphGetNodesRequest*)payload;
    hipGraph_t graph = (hipGraph_t)(uintptr_t)req->graph;

    size_t numNodes = req->max_nodes;
    hipGraphNode_t* nodes = NULL;
    if (req->max_nodes > 0) {
        nodes = (hipGraphNode_t*)malloc(req->max_nodes * sizeof(hipGraphNode_t));
        if (!nodes) {
            send_simple_response(fd, HIP_OP_GRAPH_GET_NODES, request_id, hipErrorOutOfMemory);
            return;
        }
    }

    hipError_t err = hipGraphGetNodes(graph, nodes, &numNodes);
    LOG_DEBUG("GraphGetNodes: graph=%p, numNodes=%lu, err=%d", graph, (unsigned long)numNodes, err);

    /* Build variable-length response */
    size_t resp_size = sizeof(HipRemoteGraphGetNodesResponse) + numNodes * sizeof(uint64_t);
    uint8_t* buffer = (uint8_t*)malloc(resp_size);
    if (!buffer) {
        free(nodes);
        send_simple_response(fd, HIP_OP_GRAPH_GET_NODES, request_id, hipErrorOutOfMemory);
        return;
    }

    HipRemoteGraphGetNodesResponse* resp = (HipRemoteGraphGetNodesResponse*)buffer;
    resp->header.error_code = (int32_t)err;
    resp->num_nodes = (uint32_t)numNodes;
    resp->reserved = 0;

    uint64_t* node_handles = (uint64_t*)(buffer + sizeof(HipRemoteGraphGetNodesResponse));
    for (size_t i = 0; i < numNodes && i < req->max_nodes; i++) {
        node_handles[i] = (uint64_t)(uintptr_t)nodes[i];
    }

    send_response(fd, HIP_OP_GRAPH_GET_NODES, request_id, buffer, resp_size);
    free(buffer);
    free(nodes);
}

static void handle_graph_get_root_nodes(int fd, uint32_t request_id,
                                        const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteGraphGetNodesRequest)) {
        send_simple_response(fd, HIP_OP_GRAPH_GET_ROOT_NODES, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteGraphGetNodesRequest* req = (const HipRemoteGraphGetNodesRequest*)payload;
    hipGraph_t graph = (hipGraph_t)(uintptr_t)req->graph;

    size_t numNodes = req->max_nodes;
    hipGraphNode_t* nodes = NULL;
    if (req->max_nodes > 0) {
        nodes = (hipGraphNode_t*)malloc(req->max_nodes * sizeof(hipGraphNode_t));
        if (!nodes) {
            send_simple_response(fd, HIP_OP_GRAPH_GET_ROOT_NODES, request_id, hipErrorOutOfMemory);
            return;
        }
    }

    hipError_t err = hipGraphGetRootNodes(graph, nodes, &numNodes);
    LOG_DEBUG("GraphGetRootNodes: graph=%p, numNodes=%lu, err=%d", graph, (unsigned long)numNodes, err);

    size_t resp_size = sizeof(HipRemoteGraphGetNodesResponse) + numNodes * sizeof(uint64_t);
    uint8_t* buffer = (uint8_t*)malloc(resp_size);
    if (!buffer) {
        free(nodes);
        send_simple_response(fd, HIP_OP_GRAPH_GET_ROOT_NODES, request_id, hipErrorOutOfMemory);
        return;
    }

    HipRemoteGraphGetNodesResponse* resp = (HipRemoteGraphGetNodesResponse*)buffer;
    resp->header.error_code = (int32_t)err;
    resp->num_nodes = (uint32_t)numNodes;
    resp->reserved = 0;

    uint64_t* node_handles = (uint64_t*)(buffer + sizeof(HipRemoteGraphGetNodesResponse));
    for (size_t i = 0; i < numNodes && i < req->max_nodes; i++) {
        node_handles[i] = (uint64_t)(uintptr_t)nodes[i];
    }

    send_response(fd, HIP_OP_GRAPH_GET_ROOT_NODES, request_id, buffer, resp_size);
    free(buffer);
    free(nodes);
}

static void handle_graph_node_get_type(int fd, uint32_t request_id,
                                       const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteGraphNodeGetTypeRequest)) {
        send_simple_response(fd, HIP_OP_GRAPH_NODE_GET_TYPE, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteGraphNodeGetTypeRequest* req = (const HipRemoteGraphNodeGetTypeRequest*)payload;
    hipGraphNode_t node = (hipGraphNode_t)(uintptr_t)req->node;

    hipGraphNodeType type = hipGraphNodeTypeEmpty;
    hipError_t err = hipGraphNodeGetType(node, &type);
    LOG_DEBUG("GraphNodeGetType: node=%p, type=%d, err=%d", node, type, err);

    HipRemoteGraphNodeGetTypeResponse resp = {
        .header = { .error_code = (int32_t)err },
        .type = (int32_t)type
    };
    send_response(fd, HIP_OP_GRAPH_NODE_GET_TYPE, request_id, &resp, sizeof(resp));
}

static void handle_graph_destroy_node(int fd, uint32_t request_id,
                                      const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteGraphDestroyNodeRequest)) {
        send_simple_response(fd, HIP_OP_GRAPH_DESTROY_NODE, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteGraphDestroyNodeRequest* req = (const HipRemoteGraphDestroyNodeRequest*)payload;
    hipGraphNode_t node = (hipGraphNode_t)(uintptr_t)req->node;

    hipError_t err = hipGraphDestroyNode(node);
    LOG_DEBUG("GraphDestroyNode: node=%p, err=%d", node, err);

    send_simple_response(fd, HIP_OP_GRAPH_DESTROY_NODE, request_id, err);
}

static void handle_stream_create(int fd, uint32_t request_id,
                                 const void* payload, size_t payload_size) {
    unsigned int flags = 0;
    int32_t priority = 0;
    uint64_t vhandle = 0;
    if (payload && payload_size >= sizeof(HipRemoteStreamCreateRequest)) {
        const HipRemoteStreamCreateRequest* req = (const HipRemoteStreamCreateRequest*)payload;
        flags = req->flags;
        priority = req->priority;
        vhandle = req->vhandle;
    }

    hipStream_t stream = NULL;
    hipError_t err;
    if (priority != 0) {
        err = hipStreamCreateWithPriority(&stream, flags, priority);
    } else {
        err = hipStreamCreateWithFlags(&stream, flags);
    }

    if (vhandle && err == hipSuccess) {
        vaddr_map_put(vhandle, (uint64_t)(uintptr_t)stream, 0);
        LOG_DEBUG("StreamCreate: vhandle=0x%lx flags=%u -> real=%p",
                  (unsigned long)vhandle, flags, stream);
    } else {
        LOG_DEBUG("StreamCreate: flags=%u, stream=%p, err=%d", flags, stream, err);
    }

    HipRemoteStreamCreateResponse resp = {
        .header = { .error_code = (int32_t)err },
        .stream = (uint64_t)(uintptr_t)stream
    };
    send_response(fd, HIP_OP_STREAM_CREATE, request_id, &resp, sizeof(resp));
}

static void handle_stream_destroy(int fd, uint32_t request_id,
                                  const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteStreamRequest)) {
        send_simple_response(fd, HIP_OP_STREAM_DESTROY, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteStreamRequest* req = (const HipRemoteStreamRequest*)payload;
    hipStream_t stream = vaddr_translate(req->stream);
    hipError_t err = hipStreamDestroy(stream);
    vaddr_map_remove(req->stream);
    LOG_DEBUG("StreamDestroy: stream=%p, err=%d", stream, err);
    send_simple_response(fd, HIP_OP_STREAM_DESTROY, request_id, err);
}

static void handle_stream_synchronize(int fd, uint32_t request_id,
                                      const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteStreamRequest)) {
        send_simple_response(fd, HIP_OP_STREAM_SYNCHRONIZE, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteStreamRequest* req = (const HipRemoteStreamRequest*)payload;
    hipStream_t stream = vaddr_translate(req->stream);
    hipError_t err = hipStreamSynchronize(stream);
    LOG_DEBUG("StreamSynchronize: stream=%p, err=%d", stream, err);
    send_simple_response(fd, HIP_OP_STREAM_SYNCHRONIZE, request_id, err);
}

static void handle_stream_get_flags(int fd, uint32_t request_id,
                                    const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteStreamRequest)) {
        send_simple_response(fd, HIP_OP_STREAM_GET_FLAGS, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteStreamRequest* req = (const HipRemoteStreamRequest*)payload;
    hipStream_t stream = vaddr_translate(req->stream);
    unsigned int flags = 0;
    hipError_t err = hipStreamGetFlags(stream, &flags);
    LOG_DEBUG("StreamGetFlags: stream=%p, flags=%u, err=%d", stream, flags, err);

    HipRemoteStreamGetFlagsResponse resp = {
        .header = { .error_code = (int32_t)err },
        .flags = flags
    };
    send_response(fd, HIP_OP_STREAM_GET_FLAGS, request_id, &resp, sizeof(resp));
}

static void handle_stream_get_priority(int fd, uint32_t request_id,
                                       const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteStreamRequest)) {
        send_simple_response(fd, HIP_OP_STREAM_GET_PRIORITY, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteStreamRequest* req = (const HipRemoteStreamRequest*)payload;
    hipStream_t stream = vaddr_translate(req->stream);
    int priority = 0;
    hipError_t err = hipStreamGetPriority(stream, &priority);
    LOG_DEBUG("StreamGetPriority: stream=%p, priority=%d, err=%d", stream, priority, err);

    HipRemoteStreamGetPriorityResponse resp = {
        .header = { .error_code = (int32_t)err },
        .priority = priority
    };
    send_response(fd, HIP_OP_STREAM_GET_PRIORITY, request_id, &resp, sizeof(resp));
}

static void handle_stream_wait_event(int fd, uint32_t request_id,
                                     const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteStreamWaitEventRequest)) {
        send_simple_response(fd, HIP_OP_STREAM_WAIT_EVENT, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteStreamWaitEventRequest* req = (const HipRemoteStreamWaitEventRequest*)payload;
    hipStream_t stream = vaddr_translate(req->stream);
    hipEvent_t event = (hipEvent_t)(uintptr_t)req->event;
    hipError_t err = hipStreamWaitEvent(stream, event, req->flags);
    LOG_DEBUG("StreamWaitEvent: stream=%p, event=%p, flags=%u, err=%d",
              stream, event, req->flags, err);
    send_simple_response(fd, HIP_OP_STREAM_WAIT_EVENT, request_id, err);
}

static void handle_event_create(int fd, uint32_t request_id,
                                const void* payload, size_t payload_size) {
    unsigned int flags = 0;
    if (payload && payload_size >= sizeof(HipRemoteEventCreateRequest)) {
        const HipRemoteEventCreateRequest* req = (const HipRemoteEventCreateRequest*)payload;
        flags = req->flags;
    }

    hipEvent_t event = NULL;
    hipError_t err = hipEventCreateWithFlags(&event, flags);
    LOG_DEBUG("EventCreate: flags=%u, event=%p, err=%d", flags, event, err);

    HipRemoteEventCreateResponse resp = {
        .header = { .error_code = (int32_t)err },
        .event = (uint64_t)(uintptr_t)event
    };
    send_response(fd, HIP_OP_EVENT_CREATE, request_id, &resp, sizeof(resp));
}

static void handle_event_create_batch(int fd, uint32_t request_id,
                                      const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteHandleBatchRequest)) {
        send_simple_response(fd, HIP_OP_EVENT_CREATE_BATCH, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteHandleBatchRequest* req = (const HipRemoteHandleBatchRequest*)payload;
    uint32_t count = req->count;
    if (count > HIP_REMOTE_MAX_BATCH_HANDLES) count = HIP_REMOTE_MAX_BATCH_HANDLES;

    HipRemoteHandleBatchResponse resp;
    memset(&resp, 0, sizeof(resp));
    resp.count = 0;

    hipError_t first_err = hipSuccess;
    for (uint32_t i = 0; i < count; i++) {
        hipEvent_t event = NULL;
        hipError_t err = hipEventCreateWithFlags(&event, req->flags);
        if (err == hipSuccess && event) {
            resp.handles[resp.count++] = (uint64_t)(uintptr_t)event;
        } else if (first_err == hipSuccess) {
            first_err = err;
            break;
        }
    }

    resp.header.error_code = (int32_t)first_err;
    LOG_DEBUG("EventCreateBatch: requested=%u created=%u flags=%u err=%d",
              count, resp.count, req->flags, first_err);
    send_response(fd, HIP_OP_EVENT_CREATE_BATCH, request_id, &resp, sizeof(resp));
}

static void handle_stream_create_batch(int fd, uint32_t request_id,
                                        const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteHandleBatchRequest)) {
        send_simple_response(fd, HIP_OP_STREAM_CREATE_BATCH, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteHandleBatchRequest* req = (const HipRemoteHandleBatchRequest*)payload;
    uint32_t count = req->count;
    if (count > HIP_REMOTE_MAX_BATCH_HANDLES) count = HIP_REMOTE_MAX_BATCH_HANDLES;

    HipRemoteHandleBatchResponse resp;
    memset(&resp, 0, sizeof(resp));
    resp.count = 0;

    hipError_t first_err = hipSuccess;
    for (uint32_t i = 0; i < count; i++) {
        hipStream_t stream = NULL;
        hipError_t err = hipStreamCreateWithFlags(&stream, req->flags);
        if (err == hipSuccess && stream) {
            resp.handles[resp.count++] = (uint64_t)(uintptr_t)stream;
        } else if (first_err == hipSuccess) {
            first_err = err;
            break;
        }
    }

    resp.header.error_code = (int32_t)first_err;
    LOG_DEBUG("StreamCreateBatch: requested=%u created=%u flags=%u err=%d",
              count, resp.count, req->flags, first_err);
    send_response(fd, HIP_OP_STREAM_CREATE_BATCH, request_id, &resp, sizeof(resp));
}

static void handle_event_destroy(int fd, uint32_t request_id,
                                 const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteEventRequest)) {
        send_simple_response(fd, HIP_OP_EVENT_DESTROY, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteEventRequest* req = (const HipRemoteEventRequest*)payload;
    hipEvent_t event = (hipEvent_t)(uintptr_t)req->event;
    hipError_t err = hipEventDestroy(event);
    LOG_DEBUG("EventDestroy: event=%p, err=%d", event, err);
    send_simple_response(fd, HIP_OP_EVENT_DESTROY, request_id, err);
}

static void handle_event_record(int fd, uint32_t request_id,
                                const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteEventRecordRequest)) {
        send_simple_response(fd, HIP_OP_EVENT_RECORD, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteEventRecordRequest* req = (const HipRemoteEventRecordRequest*)payload;
    hipEvent_t event = (hipEvent_t)(uintptr_t)req->event;
    hipStream_t stream = vaddr_translate(req->stream);
    hipError_t err = hipEventRecord(event, stream);
    LOG_DEBUG("EventRecord: event=%p, stream=%p, err=%d", event, stream, err);
    send_simple_response(fd, HIP_OP_EVENT_RECORD, request_id, err);
}

static void handle_event_synchronize(int fd, uint32_t request_id,
                                     const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteEventRequest)) {
        send_simple_response(fd, HIP_OP_EVENT_SYNCHRONIZE, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteEventRequest* req = (const HipRemoteEventRequest*)payload;
    hipEvent_t event = (hipEvent_t)(uintptr_t)req->event;
    hipError_t err = hipEventSynchronize(event);
    LOG_DEBUG("EventSynchronize: event=%p, err=%d", event, err);
    send_simple_response(fd, HIP_OP_EVENT_SYNCHRONIZE, request_id, err);
}

static void handle_event_query(int fd, uint32_t request_id,
                               const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteEventRequest)) {
        send_simple_response(fd, HIP_OP_EVENT_QUERY, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteEventRequest* req = (const HipRemoteEventRequest*)payload;
    hipEvent_t event = (hipEvent_t)(uintptr_t)req->event;
    hipError_t err = hipEventQuery(event);
    LOG_DEBUG("EventQuery: event=%p, err=%d", event, err);
    send_simple_response(fd, HIP_OP_EVENT_QUERY, request_id, err);
}

static void handle_event_elapsed_time(int fd, uint32_t request_id,
                                      const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteEventElapsedTimeRequest)) {
        send_simple_response(fd, HIP_OP_EVENT_ELAPSED_TIME, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteEventElapsedTimeRequest* req = (const HipRemoteEventElapsedTimeRequest*)payload;
    hipEvent_t start = (hipEvent_t)(uintptr_t)req->start_event;
    hipEvent_t end = (hipEvent_t)(uintptr_t)req->end_event;
    float ms = 0.0f;
    hipError_t err = hipEventElapsedTime(&ms, start, end);
    LOG_DEBUG("EventElapsedTime: start=%p, end=%p, ms=%.3f, err=%d", start, end, ms, err);

    HipRemoteEventElapsedTimeResponse resp = {
        .header = { .error_code = (int32_t)err },
        .milliseconds = ms
    };
    send_response(fd, HIP_OP_EVENT_ELAPSED_TIME, request_id, &resp, sizeof(resp));
}

static void handle_runtime_get_version(int fd, uint32_t request_id) {
    int version = 0;
    hipError_t err = hipRuntimeGetVersion(&version);
    LOG_DEBUG("RuntimeGetVersion: version=%d, err=%d", version, err);

    HipRemoteVersionResponse resp = {
        .header = { .error_code = (int32_t)err },
        .version = version
    };
    send_response(fd, HIP_OP_RUNTIME_GET_VERSION, request_id, &resp, sizeof(resp));
}

static void handle_driver_get_version(int fd, uint32_t request_id) {
    int version = 0;
    hipError_t err = hipDriverGetVersion(&version);
    LOG_DEBUG("DriverGetVersion: version=%d, err=%d", version, err);

    HipRemoteVersionResponse resp = {
        .header = { .error_code = (int32_t)err },
        .version = version
    };
    send_response(fd, HIP_OP_DRIVER_GET_VERSION, request_id, &resp, sizeof(resp));
}

static void handle_get_last_error(int fd, uint32_t request_id) {
    hipError_t err = hipGetLastError();
    LOG_DEBUG("GetLastError: err=%d", err);
    send_simple_response(fd, HIP_OP_GET_LAST_ERROR, request_id, err);
}

static void handle_peek_at_last_error(int fd, uint32_t request_id) {
    hipError_t err = hipPeekAtLastError();
    LOG_DEBUG("PeekAtLastError: err=%d", err);
    send_simple_response(fd, HIP_OP_PEEK_AT_LAST_ERROR, request_id, err);
}

/* ============================================================================
 * Device Limit Handlers
 * ============================================================================ */

static void handle_device_get_limit(int fd, uint32_t request_id,
                                    const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteDeviceLimitRequest)) {
        send_simple_response(fd, HIP_OP_DEVICE_GET_LIMIT, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteDeviceLimitRequest* req = (const HipRemoteDeviceLimitRequest*)payload;
    size_t value = 0;
    hipError_t err = hipDeviceGetLimit(&value, (enum hipLimit_t)req->limit);
    LOG_DEBUG("DeviceGetLimit: limit=%d, value=%zu, err=%d", req->limit, value, err);

    HipRemoteDeviceLimitResponse resp = {
        .header = { .error_code = (int32_t)err },
        .value = (uint64_t)value
    };
    send_response(fd, HIP_OP_DEVICE_GET_LIMIT, request_id, &resp, sizeof(resp));
}

static void handle_device_set_limit(int fd, uint32_t request_id,
                                    const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteDeviceLimitRequest)) {
        send_simple_response(fd, HIP_OP_DEVICE_SET_LIMIT, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteDeviceLimitRequest* req = (const HipRemoteDeviceLimitRequest*)payload;
    hipError_t err = hipDeviceSetLimit((enum hipLimit_t)req->limit, (size_t)req->value);
    LOG_DEBUG("DeviceSetLimit: limit=%d, value=%lu, err=%d", req->limit, (unsigned long)req->value, err);
    send_simple_response(fd, HIP_OP_DEVICE_SET_LIMIT, request_id, err);
}

/* ============================================================================
 * Peer Access Handlers
 * ============================================================================ */

static void handle_device_can_access_peer(int fd, uint32_t request_id,
                                          const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteDeviceCanAccessPeerRequest)) {
        send_simple_response(fd, HIP_OP_DEVICE_CAN_ACCESS_PEER, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteDeviceCanAccessPeerRequest* req = (const HipRemoteDeviceCanAccessPeerRequest*)payload;
    int can_access = 0;
    hipError_t err = hipDeviceCanAccessPeer(&can_access, req->device_id, req->peer_device_id);
    LOG_DEBUG("DeviceCanAccessPeer: device=%d, peer=%d, can_access=%d, err=%d",
              req->device_id, req->peer_device_id, can_access, err);

    HipRemoteDeviceCanAccessPeerResponse resp = {
        .header = { .error_code = (int32_t)err },
        .can_access_peer = can_access
    };
    send_response(fd, HIP_OP_DEVICE_CAN_ACCESS_PEER, request_id, &resp, sizeof(resp));
}

static void handle_device_enable_peer_access(int fd, uint32_t request_id,
                                             const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteDevicePeerAccessRequest)) {
        send_simple_response(fd, HIP_OP_DEVICE_ENABLE_PEER_ACCESS, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteDevicePeerAccessRequest* req = (const HipRemoteDevicePeerAccessRequest*)payload;
    hipError_t err = hipDeviceEnablePeerAccess(req->peer_device_id, req->flags);
    LOG_DEBUG("DeviceEnablePeerAccess: peer=%d, flags=%u, err=%d",
              req->peer_device_id, req->flags, err);
    send_simple_response(fd, HIP_OP_DEVICE_ENABLE_PEER_ACCESS, request_id, err);
}

static void handle_device_disable_peer_access(int fd, uint32_t request_id,
                                              const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteDevicePeerAccessRequest)) {
        send_simple_response(fd, HIP_OP_DEVICE_DISABLE_PEER_ACCESS, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteDevicePeerAccessRequest* req = (const HipRemoteDevicePeerAccessRequest*)payload;
    hipError_t err = hipDeviceDisablePeerAccess(req->peer_device_id);
    LOG_DEBUG("DeviceDisablePeerAccess: peer=%d, err=%d", req->peer_device_id, err);
    send_simple_response(fd, HIP_OP_DEVICE_DISABLE_PEER_ACCESS, request_id, err);
}

/* ============================================================================
 * Device Driver APIs
 * ============================================================================ */

static void handle_device_get(int fd, uint32_t request_id,
                               const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteDeviceGetRequest)) {
        send_simple_response(fd, HIP_OP_DEVICE_GET, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteDeviceGetRequest* req = (const HipRemoteDeviceGetRequest*)payload;
    hipDevice_t device;
    hipError_t err = hipDeviceGet(&device, req->ordinal);
    LOG_DEBUG("DeviceGet: ordinal=%d, device=%d, err=%d", req->ordinal, device, err);

    HipRemoteDeviceGetResponse resp = {
        .header = { .error_code = (int32_t)err },
        .device = (uint64_t)device
    };
    send_response(fd, HIP_OP_DEVICE_GET, request_id, &resp, sizeof(resp));
}

static void handle_device_get_name(int fd, uint32_t request_id,
                                   const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteDeviceGetNameRequest)) {
        send_simple_response(fd, HIP_OP_DEVICE_GET_NAME, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteDeviceGetNameRequest* req = (const HipRemoteDeviceGetNameRequest*)payload;
    char name[256] = {0};
    hipDevice_t device = (hipDevice_t)req->device;
    hipError_t err = hipDeviceGetName(name, sizeof(name), device);
    LOG_DEBUG("DeviceGetName: device=%d, name=%s, err=%d", device, name, err);

    HipRemoteDeviceGetNameResponse resp;
    memset(&resp, 0, sizeof(resp));
    resp.header.error_code = (int32_t)err;
    if (err == hipSuccess) {
        strncpy(resp.name, name, sizeof(resp.name) - 1);
    }
    send_response(fd, HIP_OP_DEVICE_GET_NAME, request_id, &resp, sizeof(resp));
}

static void handle_device_total_mem(int fd, uint32_t request_id,
                                    const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteDeviceTotalMemRequest)) {
        send_simple_response(fd, HIP_OP_DEVICE_TOTAL_MEM, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteDeviceTotalMemRequest* req = (const HipRemoteDeviceTotalMemRequest*)payload;
    size_t bytes = 0;
    hipDevice_t device = (hipDevice_t)req->device;
    hipError_t err = hipDeviceTotalMem(&bytes, device);
    LOG_DEBUG("DeviceTotalMem: device=%d, bytes=%zu, err=%d", device, bytes, err);

    HipRemoteDeviceTotalMemResponse resp = {
        .header = { .error_code = (int32_t)err },
        .bytes = (uint64_t)bytes
    };
    send_response(fd, HIP_OP_DEVICE_TOTAL_MEM, request_id, &resp, sizeof(resp));
}

static void handle_device_get_pci_bus_id(int fd, uint32_t request_id,
                                         const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteDeviceGetPCIBusIdRequest)) {
        send_simple_response(fd, HIP_OP_DEVICE_GET_PCI_BUS_ID, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteDeviceGetPCIBusIdRequest* req = (const HipRemoteDeviceGetPCIBusIdRequest*)payload;
    char pci_bus_id[32] = {0};
    hipError_t err = hipDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), req->device);
    LOG_DEBUG("DeviceGetPCIBusId: device=%d, pci_bus_id=%s, err=%d", req->device, pci_bus_id, err);

    HipRemoteDeviceGetPCIBusIdResponse resp;
    memset(&resp, 0, sizeof(resp));
    resp.header.error_code = (int32_t)err;
    if (err == hipSuccess) {
        strncpy(resp.pci_bus_id, pci_bus_id, sizeof(resp.pci_bus_id) - 1);
    }
    send_response(fd, HIP_OP_DEVICE_GET_PCI_BUS_ID, request_id, &resp, sizeof(resp));
}

static void handle_device_get_by_pci_bus_id(int fd, uint32_t request_id,
                                            const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteDeviceGetByPCIBusIdRequest)) {
        send_simple_response(fd, HIP_OP_DEVICE_GET_BY_PCI_BUS_ID, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteDeviceGetByPCIBusIdRequest* req = (const HipRemoteDeviceGetByPCIBusIdRequest*)payload;
    int device = 0;
    hipError_t err = hipDeviceGetByPCIBusId(&device, req->pci_bus_id);
    LOG_DEBUG("DeviceGetByPCIBusId: pci_bus_id=%s, device=%d, err=%d", req->pci_bus_id, device, err);

    HipRemoteDeviceGetByPCIBusIdResponse resp = {
        .header = { .error_code = (int32_t)err },
        .device = device
    };
    send_response(fd, HIP_OP_DEVICE_GET_BY_PCI_BUS_ID, request_id, &resp, sizeof(resp));
}

static void handle_device_compute_capability(int fd, uint32_t request_id,
                                             const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteDeviceComputeCapabilityRequest)) {
        send_simple_response(fd, HIP_OP_DEVICE_COMPUTE_CAPABILITY, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteDeviceComputeCapabilityRequest* req = (const HipRemoteDeviceComputeCapabilityRequest*)payload;
    int major = 0, minor = 0;
    hipDevice_t device = (hipDevice_t)req->device;
    hipError_t err = hipDeviceComputeCapability(&major, &minor, device);
    LOG_DEBUG("DeviceComputeCapability: device=%d, major=%d, minor=%d, err=%d", device, major, minor, err);

    HipRemoteDeviceComputeCapabilityResponse resp = {
        .header = { .error_code = (int32_t)err },
        .major = major,
        .minor = minor
    };
    send_response(fd, HIP_OP_DEVICE_COMPUTE_CAPABILITY, request_id, &resp, sizeof(resp));
}

static void handle_device_get_uuid(int fd, uint32_t request_id,
                                   const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteDeviceGetUuidRequest)) {
        send_simple_response(fd, HIP_OP_DEVICE_GET_UUID, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteDeviceGetUuidRequest* req = (const HipRemoteDeviceGetUuidRequest*)payload;
    hipUUID uuid;
    memset(&uuid, 0, sizeof(uuid));
    hipDevice_t device = (hipDevice_t)req->device;
    hipError_t err = hipDeviceGetUuid(&uuid, device);
    LOG_DEBUG("DeviceGetUuid: device=%d, err=%d", device, err);

    HipRemoteDeviceGetUuidResponse resp;
    memset(&resp, 0, sizeof(resp));
    resp.header.error_code = (int32_t)err;
    if (err == hipSuccess) {
        memcpy(resp.uuid, uuid.bytes, HIP_UUID_SIZE);
    }
    send_response(fd, HIP_OP_DEVICE_GET_UUID, request_id, &resp, sizeof(resp));
}

/* ============================================================================
 * Device Cache/Config APIs
 * ============================================================================ */

static void handle_device_get_cache_config(int fd, uint32_t request_id) {
    hipFuncCache_t cache_config;
    hipError_t err = hipDeviceGetCacheConfig(&cache_config);
    LOG_DEBUG("DeviceGetCacheConfig: config=%d, err=%d", cache_config, err);

    HipRemoteDeviceCacheConfigResponse resp = {
        .header = { .error_code = (int32_t)err },
        .cache_config = (int32_t)cache_config
    };
    send_response(fd, HIP_OP_DEVICE_GET_CACHE_CONFIG, request_id, &resp, sizeof(resp));
}

static void handle_device_set_cache_config(int fd, uint32_t request_id,
                                           const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteDeviceCacheConfigRequest)) {
        send_simple_response(fd, HIP_OP_DEVICE_SET_CACHE_CONFIG, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteDeviceCacheConfigRequest* req = (const HipRemoteDeviceCacheConfigRequest*)payload;
    hipFuncCache_t cache_config = (hipFuncCache_t)req->cache_config;
    hipError_t err = hipDeviceSetCacheConfig(cache_config);
    LOG_DEBUG("DeviceSetCacheConfig: config=%d, err=%d", cache_config, err);
    send_simple_response(fd, HIP_OP_DEVICE_SET_CACHE_CONFIG, request_id, err);
}

static void handle_device_get_shared_mem_config(int fd, uint32_t request_id) {
    hipSharedMemConfig config;
    hipError_t err = hipDeviceGetSharedMemConfig(&config);
    LOG_DEBUG("DeviceGetSharedMemConfig: config=%d, err=%d", config, err);

    HipRemoteDeviceSharedMemConfigResponse resp = {
        .header = { .error_code = (int32_t)err },
        .shared_mem_config = (int32_t)config
    };
    send_response(fd, HIP_OP_DEVICE_GET_SHARED_MEM_CONFIG, request_id, &resp, sizeof(resp));
}

static void handle_device_set_shared_mem_config(int fd, uint32_t request_id,
                                                const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteDeviceSharedMemConfigRequest)) {
        send_simple_response(fd, HIP_OP_DEVICE_SET_SHARED_MEM_CONFIG, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteDeviceSharedMemConfigRequest* req = (const HipRemoteDeviceSharedMemConfigRequest*)payload;
    hipSharedMemConfig config = (hipSharedMemConfig)req->shared_mem_config;
    hipError_t err = hipDeviceSetSharedMemConfig(config);
    LOG_DEBUG("DeviceSetSharedMemConfig: config=%d, err=%d", config, err);
    send_simple_response(fd, HIP_OP_DEVICE_SET_SHARED_MEM_CONFIG, request_id, err);
}

static void handle_get_device_flags(int fd, uint32_t request_id) {
    unsigned int flags = 0;
    hipError_t err = hipGetDeviceFlags(&flags);
    LOG_DEBUG("GetDeviceFlags: flags=%u, err=%d", flags, err);

    HipRemoteDeviceFlagsResponse resp = {
        .header = { .error_code = (int32_t)err },
        .flags = flags
    };
    send_response(fd, HIP_OP_GET_DEVICE_FLAGS, request_id, &resp, sizeof(resp));
}

static void handle_set_device_flags(int fd, uint32_t request_id,
                                    const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteDeviceFlagsRequest)) {
        send_simple_response(fd, HIP_OP_SET_DEVICE_FLAGS, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteDeviceFlagsRequest* req = (const HipRemoteDeviceFlagsRequest*)payload;
    hipError_t err = hipSetDeviceFlags(req->flags);
    LOG_DEBUG("SetDeviceFlags: flags=%u, err=%d", req->flags, err);
    send_simple_response(fd, HIP_OP_SET_DEVICE_FLAGS, request_id, err);
}

static void handle_device_get_p2p_attribute(int fd, uint32_t request_id,
                                            const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteDeviceGetP2PAttributeRequest)) {
        send_simple_response(fd, HIP_OP_DEVICE_GET_P2P_ATTRIBUTE, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteDeviceGetP2PAttributeRequest* req = (const HipRemoteDeviceGetP2PAttributeRequest*)payload;
    int value = 0;
    hipDeviceP2PAttr attr = (hipDeviceP2PAttr)req->attr;
    hipError_t err = hipDeviceGetP2PAttribute(&value, attr, req->src_device, req->dst_device);
    LOG_DEBUG("DeviceGetP2PAttribute: attr=%d, src=%d, dst=%d, value=%d, err=%d",
              attr, req->src_device, req->dst_device, value, err);

    HipRemoteDeviceGetP2PAttributeResponse resp = {
        .header = { .error_code = (int32_t)err },
        .value = value
    };
    send_response(fd, HIP_OP_DEVICE_GET_P2P_ATTRIBUTE, request_id, &resp, sizeof(resp));
}

static void handle_device_get_stream_priority_range(int fd, uint32_t request_id,
                                                     const void* payload, size_t payload_size) {
    (void)payload;
    (void)payload_size;

    int least_priority = 0;
    int greatest_priority = 0;
    hipError_t err = hipDeviceGetStreamPriorityRange(&least_priority, &greatest_priority);
    LOG_DEBUG("DeviceGetStreamPriorityRange: least=%d, greatest=%d, err=%d",
              least_priority, greatest_priority, err);

    HipRemoteDeviceGetStreamPriorityRangeResponse resp = {
        .header = { .error_code = (int32_t)err },
        .least_priority = least_priority,
        .greatest_priority = greatest_priority
    };
    send_response(fd, HIP_OP_DEVICE_GET_STREAM_PRIORITY_RANGE, request_id, &resp, sizeof(resp));
}

static void handle_set_valid_devices(int fd, uint32_t request_id,
                                      const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteSetValidDevicesRequest)) {
        send_simple_response(fd, HIP_OP_SET_VALID_DEVICES, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteSetValidDevicesRequest* req = (const HipRemoteSetValidDevicesRequest*)payload;
    size_t expected_size = sizeof(HipRemoteSetValidDevicesRequest) + req->len * sizeof(int32_t);
    if (payload_size < expected_size) {
        send_simple_response(fd, HIP_OP_SET_VALID_DEVICES, request_id, hipErrorInvalidValue);
        return;
    }

    const int32_t* device_ids = (const int32_t*)((const uint8_t*)payload + sizeof(HipRemoteSetValidDevicesRequest));
    int* devices = (int*)malloc(req->len * sizeof(int));
    if (!devices) {
        send_simple_response(fd, HIP_OP_SET_VALID_DEVICES, request_id, hipErrorOutOfMemory);
        return;
    }

    for (int32_t i = 0; i < req->len; i++) {
        devices[i] = device_ids[i];
    }

    hipError_t err = hipSetValidDevices(devices, req->len);
    LOG_DEBUG("SetValidDevices: count=%d, err=%d", req->len, err);

    free(devices);
    send_simple_response(fd, HIP_OP_SET_VALID_DEVICES, request_id, err);
}

static void handle_choose_device(int fd, uint32_t request_id,
                                  const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteChooseDeviceRequest)) {
        send_simple_response(fd, HIP_OP_CHOOSE_DEVICE, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteChooseDeviceRequest* req = (const HipRemoteChooseDeviceRequest*)payload;

    /* Build simplified hipDeviceProp_t from request fields */
    hipDeviceProp_t prop = {0};
    prop.totalGlobalMem = req->total_global_mem;
    prop.major = req->major;
    prop.minor = req->minor;
    prop.multiProcessorCount = req->multi_processor_count;
    prop.warpSize = req->warp_size;
    prop.maxThreadsPerBlock = req->max_threads_per_block;

    int device = 0;
    hipError_t err = hipChooseDevice(&device, &prop);
    LOG_DEBUG("ChooseDevice: major=%d, minor=%d, device=%d, err=%d",
              req->major, req->minor, device, err);

    HipRemoteChooseDeviceResponse resp = {
        .header = { .error_code = (int32_t)err },
        .device = device
    };
    send_response(fd, HIP_OP_CHOOSE_DEVICE, request_id, &resp, sizeof(resp));
}

static void handle_stream_get_capture_info(int fd, uint32_t request_id,
                                            const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteStreamGetCaptureInfoRequest)) {
        send_simple_response(fd, HIP_OP_STREAM_GET_CAPTURE_INFO, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteStreamGetCaptureInfoRequest* req = (const HipRemoteStreamGetCaptureInfoRequest*)payload;
    hipStream_t stream = vaddr_translate(req->stream);

    hipStreamCaptureStatus capture_status;
    unsigned long long id = 0;
    hipError_t err = hipStreamGetCaptureInfo(stream, &capture_status, &id);
    LOG_DEBUG("StreamGetCaptureInfo: stream=%p, status=%d, id=%llu, err=%d",
              stream, capture_status, id, err);

    HipRemoteStreamGetCaptureInfoResponse resp = {
        .header = { .error_code = (int32_t)err },
        .capture_status = (int32_t)capture_status,
        .graph = id,
        .reserved = 0
    };
    send_response(fd, HIP_OP_STREAM_GET_CAPTURE_INFO, request_id, &resp, sizeof(resp));
}

static void handle_stream_update_capture_dependencies(int fd, uint32_t request_id,
                                                       const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteStreamUpdateCaptureDependenciesRequest)) {
        send_simple_response(fd, HIP_OP_STREAM_UPDATE_CAPTURE_DEPENDENCIES, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteStreamUpdateCaptureDependenciesRequest* req =
        (const HipRemoteStreamUpdateCaptureDependenciesRequest*)payload;
    size_t expected_size = sizeof(HipRemoteStreamUpdateCaptureDependenciesRequest) +
                          req->num_dependencies * sizeof(uint64_t);
    if (payload_size < expected_size) {
        send_simple_response(fd, HIP_OP_STREAM_UPDATE_CAPTURE_DEPENDENCIES, request_id, hipErrorInvalidValue);
        return;
    }

    hipStream_t stream = vaddr_translate(req->stream);
    const uint64_t* node_handles = (const uint64_t*)((const uint8_t*)payload +
                                   sizeof(HipRemoteStreamUpdateCaptureDependenciesRequest));

    hipGraphNode_t* dependencies = NULL;
    if (req->num_dependencies > 0) {
        dependencies = (hipGraphNode_t*)malloc(req->num_dependencies * sizeof(hipGraphNode_t));
        if (!dependencies) {
            send_simple_response(fd, HIP_OP_STREAM_UPDATE_CAPTURE_DEPENDENCIES, request_id, hipErrorOutOfMemory);
            return;
        }
        for (uint32_t i = 0; i < req->num_dependencies; i++) {
            dependencies[i] = (hipGraphNode_t)(uintptr_t)node_handles[i];
        }
    }

    hipError_t err = hipStreamUpdateCaptureDependencies(stream, dependencies,
                                                         req->num_dependencies, req->flags);
    LOG_DEBUG("StreamUpdateCaptureDependencies: stream=%p, count=%u, flags=%u, err=%d",
              stream, req->num_dependencies, req->flags, err);

    free(dependencies);
    send_simple_response(fd, HIP_OP_STREAM_UPDATE_CAPTURE_DEPENDENCIES, request_id, err);
}

static void handle_pointer_get_attribute(int fd, uint32_t request_id,
                                          const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemotePointerGetAttributeRequest)) {
        send_simple_response(fd, HIP_OP_POINTER_GET_ATTRIBUTE, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemotePointerGetAttributeRequest* req = (const HipRemotePointerGetAttributeRequest*)payload;
    void* ptr = (void*)(uintptr_t)req->ptr;
    hipPointer_attribute attribute = (hipPointer_attribute)req->attribute;

    uint64_t data = 0;
    hipError_t err = hipPointerGetAttribute(&data, attribute, ptr);
    LOG_DEBUG("PointerGetAttribute: ptr=%p, attr=%d, data=%llu, err=%d",
              ptr, attribute, (unsigned long long)data, err);

    HipRemotePointerGetAttributeResponse resp = {
        .header = { .error_code = (int32_t)err },
        .data = data
    };
    send_response(fd, HIP_OP_POINTER_GET_ATTRIBUTE, request_id, &resp, sizeof(resp));
}

/* ============================================================================
 * 3D Memory Copy Handler
 * ============================================================================ */

static void handle_memcpy3d(int fd, uint32_t request_id,
                            const void* payload, size_t payload_size, bool is_async) {
    if (!payload || payload_size < sizeof(HipRemoteMemcpy3DRequest)) {
        send_simple_response(fd, HIP_OP_MEMCPY_3D, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteMemcpy3DRequest* req = (const HipRemoteMemcpy3DRequest*)payload;
    hipStream_t stream = vaddr_translate(req->stream);

    LOG_DEBUG("Memcpy3D: extent=(%lu,%lu,%lu), kind=%d, stream=%p",
              (unsigned long)req->width, (unsigned long)req->height,
              (unsigned long)req->depth, req->kind, stream);

    /* Construct hipMemcpy3DParms */
    hipMemcpy3DParms parms = {0};
    parms.srcPtr = make_hipPitchedPtr((void*)(uintptr_t)req->src_ptr,
                                       req->src_pitch, req->src_pitch, req->src_height);
    parms.srcPos = make_hipPos(req->src_x_offset, req->src_y_offset, req->src_z_offset);
    parms.dstPtr = make_hipPitchedPtr((void*)(uintptr_t)req->dst_ptr,
                                       req->dst_pitch, req->dst_pitch, req->dst_height);
    parms.dstPos = make_hipPos(req->dst_x_offset, req->dst_y_offset, req->dst_z_offset);
    parms.extent = make_hipExtent(req->width, req->height, req->depth);
    parms.kind = (hipMemcpyKind)req->kind;

    hipError_t err;
    if (is_async) {
        err = hipMemcpy3DAsync(&parms, stream);
    } else {
        err = hipMemcpy3D(&parms);
    }
    send_simple_response(fd, is_async ? HIP_OP_MEMCPY_3D_ASYNC : HIP_OP_MEMCPY_3D, request_id, err);
}

/* ============================================================================
 * Peer Memory Copy Handler
 * ============================================================================ */

static void handle_memcpy_peer(int fd, uint32_t request_id,
                               const void* payload, size_t payload_size, bool is_async) {
    if (!payload || payload_size < sizeof(HipRemoteMemcpyPeerRequest)) {
        send_simple_response(fd, HIP_OP_MEMCPY_PEER, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteMemcpyPeerRequest* req = (const HipRemoteMemcpyPeerRequest*)payload;
    hipStream_t stream = vaddr_translate(req->stream);

    LOG_DEBUG("MemcpyPeer: dst=%p (dev %d), src=%p (dev %d), size=%lu, stream=%p",
              (void*)(uintptr_t)req->dst, req->dst_device,
              (void*)(uintptr_t)req->src, req->src_device,
              (unsigned long)req->size, stream);

    hipError_t err;
    if (is_async) {
        err = hipMemcpyPeerAsync((void*)(uintptr_t)req->dst, req->dst_device,
                                  (void*)(uintptr_t)req->src, req->src_device,
                                  req->size, stream);
    } else {
        err = hipMemcpyPeer((void*)(uintptr_t)req->dst, req->dst_device,
                             (void*)(uintptr_t)req->src, req->src_device, req->size);
    }
    send_simple_response(fd, is_async ? HIP_OP_MEMCPY_PEER_ASYNC : HIP_OP_MEMCPY_PEER, request_id, err);
}

/* ============================================================================
 * Function Attributes Handlers
 * ============================================================================ */

static void handle_func_get_attributes(int fd, uint32_t request_id,
                                        const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteFuncGetAttributesRequest)) {
        send_simple_response(fd, HIP_OP_FUNC_GET_ATTRIBUTES, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteFuncGetAttributesRequest* req = (const HipRemoteFuncGetAttributesRequest*)payload;
    hipFunction_t func = (hipFunction_t)(uintptr_t)req->function;

    hipFuncAttributes attr;
    hipError_t err = hipFuncGetAttributes(&attr, (const void*)func);
    LOG_DEBUG("FuncGetAttributes: func=%p, shared=%d, regs=%d, err=%d",
              (void*)func, attr.sharedSizeBytes, attr.numRegs, err);

    HipRemoteFuncGetAttributesResponse resp = {
        .header = { .error_code = (int32_t)err },
        .shared_size_bytes = attr.sharedSizeBytes,
        .const_size_bytes = attr.constSizeBytes,
        .local_size_bytes = attr.localSizeBytes,
        .num_regs = attr.numRegs,
        .max_threads_per_block = attr.maxThreadsPerBlock,
        .ptx_version = attr.ptxVersion,
        .binary_version = attr.binaryVersion,
        .cache_mode_ca = attr.cacheModeCA,
        .max_dynamic_shared_size_bytes = attr.maxDynamicSharedSizeBytes,
        .preferred_shared_memory_carveout = attr.preferredShmemCarveout
    };
    send_response(fd, HIP_OP_FUNC_GET_ATTRIBUTES, request_id, &resp, sizeof(resp));
}

static void handle_func_set_attribute(int fd, uint32_t request_id,
                                       const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteFuncSetAttributeRequest)) {
        send_simple_response(fd, HIP_OP_FUNC_SET_ATTRIBUTE, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteFuncSetAttributeRequest* req = (const HipRemoteFuncSetAttributeRequest*)payload;
    hipFunction_t func = (hipFunction_t)(uintptr_t)req->function;
    hipFunction_attribute attr = (hipFunction_attribute)req->attribute;

    hipError_t err = hipFuncSetAttribute((const void*)func, attr, req->value);
    LOG_DEBUG("FuncSetAttribute: func=%p, attr=%d, value=%d, err=%d",
              (void*)func, attr, req->value, err);

    send_simple_response(fd, HIP_OP_FUNC_SET_ATTRIBUTE, request_id, err);
}

static void handle_func_set_cache_config(int fd, uint32_t request_id,
                                          const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteFuncSetCacheConfigRequest)) {
        send_simple_response(fd, HIP_OP_FUNC_SET_CACHE_CONFIG, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteFuncSetCacheConfigRequest* req = (const HipRemoteFuncSetCacheConfigRequest*)payload;
    hipFunction_t func = (hipFunction_t)(uintptr_t)req->function;
    hipFuncCache_t cache_config = (hipFuncCache_t)req->cache_config;

    hipError_t err = hipFuncSetCacheConfig((const void*)func, cache_config);
    LOG_DEBUG("FuncSetCacheConfig: func=%p, config=%d, err=%d",
              (void*)func, cache_config, err);

    send_simple_response(fd, HIP_OP_FUNC_SET_CACHE_CONFIG, request_id, err);
}

/* ============================================================================
 * Occupancy Handlers
 * ============================================================================ */

static void handle_occupancy_max_potential_block_size(int fd, uint32_t request_id,
                                                       const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteOccupancyMaxPotentialBlockSizeRequest)) {
        send_simple_response(fd, HIP_OP_OCCUPANCY_MAX_POTENTIAL_BLOCK_SIZE, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteOccupancyMaxPotentialBlockSizeRequest* req =
        (const HipRemoteOccupancyMaxPotentialBlockSizeRequest*)payload;
    hipFunction_t f = (hipFunction_t)(uintptr_t)req->function;

    int min_grid_size = 0, block_size = 0;
    hipError_t err = hipOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                                                        f, req->dyn_shared_mem,
                                                        req->block_size_limit);
    LOG_DEBUG("OccupancyMaxPotentialBlockSize: func=%p, min_grid=%d, block=%d, err=%d",
              (void*)f, min_grid_size, block_size, err);

    HipRemoteOccupancyMaxPotentialBlockSizeResponse resp = {
        .header = { .error_code = (int32_t)err },
        .min_grid_size = min_grid_size,
        .block_size = block_size
    };
    send_response(fd, HIP_OP_OCCUPANCY_MAX_POTENTIAL_BLOCK_SIZE, request_id, &resp, sizeof(resp));
}

static void handle_occupancy_max_active_blocks_per_sm(int fd, uint32_t request_id,
                                                       const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteOccupancyMaxActiveBlocksPerSMRequest)) {
        send_simple_response(fd, HIP_OP_OCCUPANCY_MAX_ACTIVE_BLOCKS_PER_SM, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteOccupancyMaxActiveBlocksPerSMRequest* req =
        (const HipRemoteOccupancyMaxActiveBlocksPerSMRequest*)payload;
    hipFunction_t f = (hipFunction_t)(uintptr_t)req->function;

    int num_blocks = 0;
    hipError_t err = hipOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, f,
                                                                   req->block_size,
                                                                   req->dyn_shared_mem);
    LOG_DEBUG("OccupancyMaxActiveBlocksPerSM: func=%p, block_size=%d, num_blocks=%d, err=%d",
              (void*)f, req->block_size, num_blocks, err);

    HipRemoteOccupancyMaxActiveBlocksPerSMResponse resp = {
        .header = { .error_code = (int32_t)err },
        .num_blocks = num_blocks
    };
    send_response(fd, HIP_OP_OCCUPANCY_MAX_ACTIVE_BLOCKS_PER_SM, request_id, &resp, sizeof(resp));
}

/* ============================================================================
 * Graph Handlers
 * ============================================================================ */

static void handle_graph_create(int fd, uint32_t request_id,
                                const void* payload, size_t payload_size) {
    unsigned int flags = 0;
    if (payload && payload_size >= sizeof(HipRemoteGraphCreateRequest)) {
        const HipRemoteGraphCreateRequest* req = (const HipRemoteGraphCreateRequest*)payload;
        flags = req->flags;
    }

    hipGraph_t graph = NULL;
    hipError_t err = hipGraphCreate(&graph, flags);
    LOG_DEBUG("GraphCreate: flags=%u, graph=%p, err=%d", flags, (void*)graph, err);

    HipRemoteGraphCreateResponse resp = {
        .header = { .error_code = (int32_t)err },
        .graph = (uint64_t)(uintptr_t)graph
    };
    send_response(fd, HIP_OP_GRAPH_CREATE, request_id, &resp, sizeof(resp));
}

static void handle_graph_destroy(int fd, uint32_t request_id,
                                 const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteGraphDestroyRequest)) {
        send_simple_response(fd, HIP_OP_GRAPH_DESTROY, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteGraphDestroyRequest* req = (const HipRemoteGraphDestroyRequest*)payload;
    hipGraph_t graph = (hipGraph_t)(uintptr_t)req->graph;
    hipError_t err = hipGraphDestroy(graph);
    LOG_DEBUG("GraphDestroy: graph=%p, err=%d", (void*)graph, err);
    send_simple_response(fd, HIP_OP_GRAPH_DESTROY, request_id, err);
}

static void handle_graph_instantiate(int fd, uint32_t request_id,
                                     const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteGraphInstantiateRequest)) {
        send_simple_response(fd, HIP_OP_GRAPH_INSTANTIATE, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteGraphInstantiateRequest* req = (const HipRemoteGraphInstantiateRequest*)payload;
    hipGraph_t graph = (hipGraph_t)(uintptr_t)req->graph;
    hipGraphExec_t graphExec = NULL;
    hipError_t err = hipGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
    LOG_DEBUG("GraphInstantiate: graph=%p, graphExec=%p, err=%d",
              (void*)graph, (void*)graphExec, err);

    HipRemoteGraphInstantiateResponse resp = {
        .header = { .error_code = (int32_t)err },
        .graph_exec = (uint64_t)(uintptr_t)graphExec
    };
    send_response(fd, HIP_OP_GRAPH_INSTANTIATE, request_id, &resp, sizeof(resp));
}

static void handle_graph_launch(int fd, uint32_t request_id,
                                const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteGraphLaunchRequest)) {
        send_simple_response(fd, HIP_OP_GRAPH_LAUNCH, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteGraphLaunchRequest* req = (const HipRemoteGraphLaunchRequest*)payload;
    hipGraphExec_t graphExec = (hipGraphExec_t)(uintptr_t)req->graph_exec;
    hipStream_t stream = vaddr_translate(req->stream);
    hipError_t err = hipGraphLaunch(graphExec, stream);
    LOG_DEBUG("GraphLaunch: graphExec=%p, stream=%p, err=%d",
              (void*)graphExec, (void*)stream, err);
    send_simple_response(fd, HIP_OP_GRAPH_LAUNCH, request_id, err);
}

static void handle_graph_exec_destroy(int fd, uint32_t request_id,
                                      const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteGraphExecDestroyRequest)) {
        send_simple_response(fd, HIP_OP_GRAPH_EXEC_DESTROY, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteGraphExecDestroyRequest* req = (const HipRemoteGraphExecDestroyRequest*)payload;
    hipGraphExec_t graphExec = (hipGraphExec_t)(uintptr_t)req->graph_exec;
    hipError_t err = hipGraphExecDestroy(graphExec);
    LOG_DEBUG("GraphExecDestroy: graphExec=%p, err=%d", (void*)graphExec, err);
    send_simple_response(fd, HIP_OP_GRAPH_EXEC_DESTROY, request_id, err);
}

static void handle_graph_clone(int fd, uint32_t request_id,
                                const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteGraphCloneRequest)) {
        send_simple_response(fd, HIP_OP_GRAPH_CLONE, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteGraphCloneRequest* req = (const HipRemoteGraphCloneRequest*)payload;
    hipGraph_t original = (hipGraph_t)(uintptr_t)req->original_graph;
    hipGraph_t cloned = NULL;
    hipError_t err = hipGraphClone(&cloned, original);
    LOG_DEBUG("GraphClone: original=%p, cloned=%p, err=%d",
              (void*)original, (void*)cloned, err);

    HipRemoteGraphCloneResponse resp = {
        .header = { .error_code = (int32_t)err },
        .cloned_graph = (uint64_t)(uintptr_t)cloned
    };
    send_response(fd, HIP_OP_GRAPH_CLONE, request_id, &resp, sizeof(resp));
}

static void handle_graph_node_get_dependencies(int fd, uint32_t request_id,
                                                const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteGraphNodeGetDependenciesRequest)) {
        send_simple_response(fd, HIP_OP_GRAPH_NODE_GET_DEPENDENCIES, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteGraphNodeGetDependenciesRequest* req =
        (const HipRemoteGraphNodeGetDependenciesRequest*)payload;
    hipGraphNode_t node = (hipGraphNode_t)(uintptr_t)req->node;
    size_t max_nodes = req->max_nodes;

    /* Allocate buffer for dependencies */
    hipGraphNode_t* deps = (hipGraphNode_t*)malloc(max_nodes * sizeof(hipGraphNode_t));
    if (!deps && max_nodes > 0) {
        send_simple_response(fd, HIP_OP_GRAPH_NODE_GET_DEPENDENCIES, request_id, hipErrorOutOfMemory);
        return;
    }

    size_t num_deps = max_nodes;
    hipError_t err = hipGraphNodeGetDependencies(node, deps, &num_deps);
    LOG_DEBUG("GraphNodeGetDependencies: node=%p, num=%zu, err=%d",
              (void*)node, num_deps, err);

    /* Build response with node handles */
    size_t resp_size = sizeof(HipRemoteGraphNodeGetDependenciesResponse) +
                       num_deps * sizeof(uint64_t);
    uint8_t* resp_buffer = (uint8_t*)malloc(resp_size);
    if (!resp_buffer) {
        free(deps);
        send_simple_response(fd, HIP_OP_GRAPH_NODE_GET_DEPENDENCIES, request_id, hipErrorOutOfMemory);
        return;
    }

    HipRemoteGraphNodeGetDependenciesResponse* resp =
        (HipRemoteGraphNodeGetDependenciesResponse*)resp_buffer;
    resp->header.error_code = (int32_t)err;
    resp->num_nodes = (uint32_t)num_deps;
    resp->reserved = 0;

    uint64_t* node_handles = (uint64_t*)(resp_buffer + sizeof(*resp));
    for (size_t i = 0; i < num_deps; i++) {
        node_handles[i] = (uint64_t)(uintptr_t)deps[i];
    }

    send_response(fd, HIP_OP_GRAPH_NODE_GET_DEPENDENCIES, request_id, resp_buffer, resp_size);
    free(deps);
    free(resp_buffer);
}

static void handle_graph_node_get_dependent_nodes(int fd, uint32_t request_id,
                                                   const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteGraphNodeGetDependenciesRequest)) {
        send_simple_response(fd, HIP_OP_GRAPH_NODE_GET_DEPENDENT_NODES, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteGraphNodeGetDependenciesRequest* req =
        (const HipRemoteGraphNodeGetDependenciesRequest*)payload;
    hipGraphNode_t node = (hipGraphNode_t)(uintptr_t)req->node;
    size_t max_nodes = req->max_nodes;

    /* Allocate buffer for dependent nodes */
    hipGraphNode_t* deps = (hipGraphNode_t*)malloc(max_nodes * sizeof(hipGraphNode_t));
    if (!deps && max_nodes > 0) {
        send_simple_response(fd, HIP_OP_GRAPH_NODE_GET_DEPENDENT_NODES, request_id, hipErrorOutOfMemory);
        return;
    }

    size_t num_deps = max_nodes;
    hipError_t err = hipGraphNodeGetDependentNodes(node, deps, &num_deps);
    LOG_DEBUG("GraphNodeGetDependentNodes: node=%p, num=%zu, err=%d",
              (void*)node, num_deps, err);

    /* Build response with node handles */
    size_t resp_size = sizeof(HipRemoteGraphNodeGetDependenciesResponse) +
                       num_deps * sizeof(uint64_t);
    uint8_t* resp_buffer = (uint8_t*)malloc(resp_size);
    if (!resp_buffer) {
        free(deps);
        send_simple_response(fd, HIP_OP_GRAPH_NODE_GET_DEPENDENT_NODES, request_id, hipErrorOutOfMemory);
        return;
    }

    HipRemoteGraphNodeGetDependenciesResponse* resp =
        (HipRemoteGraphNodeGetDependenciesResponse*)resp_buffer;
    resp->header.error_code = (int32_t)err;
    resp->num_nodes = (uint32_t)num_deps;
    resp->reserved = 0;

    uint64_t* node_handles = (uint64_t*)(resp_buffer + sizeof(*resp));
    for (size_t i = 0; i < num_deps; i++) {
        node_handles[i] = (uint64_t)(uintptr_t)deps[i];
    }

    send_response(fd, HIP_OP_GRAPH_NODE_GET_DEPENDENT_NODES, request_id, resp_buffer, resp_size);
    free(deps);
    free(resp_buffer);
}

static void handle_graph_exec_update(int fd, uint32_t request_id,
                                      const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteGraphExecUpdateRequest)) {
        send_simple_response(fd, HIP_OP_GRAPH_EXEC_UPDATE, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteGraphExecUpdateRequest* req = (const HipRemoteGraphExecUpdateRequest*)payload;
    hipGraphExec_t graphExec = (hipGraphExec_t)(uintptr_t)req->graph_exec;
    hipGraph_t graph = (hipGraph_t)(uintptr_t)req->graph;

    hipGraphNode_t errorNode = NULL;
    hipGraphExecUpdateResult updateResult;
    hipError_t err = hipGraphExecUpdate(graphExec, graph, &errorNode, &updateResult);
    LOG_DEBUG("GraphExecUpdate: graphExec=%p, graph=%p, result=%d, err=%d",
              (void*)graphExec, (void*)graph, updateResult, err);

    HipRemoteGraphExecUpdateResponse resp = {
        .header = { .error_code = (int32_t)err },
        .update_result = (int32_t)updateResult
    };
    send_response(fd, HIP_OP_GRAPH_EXEC_UPDATE, request_id, &resp, sizeof(resp));
}

static void handle_graph_exec_kernel_node_set_params(int fd, uint32_t request_id,
                                                      const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteGraphExecKernelNodeSetParamsRequest)) {
        send_simple_response(fd, HIP_OP_GRAPH_EXEC_KERNEL_NODE_SET_PARAMS, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteGraphExecKernelNodeSetParamsRequest* req =
        (const HipRemoteGraphExecKernelNodeSetParamsRequest*)payload;
    hipGraphExec_t graphExec = (hipGraphExec_t)(uintptr_t)req->graph_exec;
    hipGraphNode_t node = (hipGraphNode_t)(uintptr_t)req->node;

    /* Build kernel node params structure */
    hipKernelNodeParams params;
    params.func = (void*)(uintptr_t)req->func;
    params.gridDim.x = req->grid_dim_x;
    params.gridDim.y = req->grid_dim_y;
    params.gridDim.z = req->grid_dim_z;
    params.blockDim.x = req->block_dim_x;
    params.blockDim.y = req->block_dim_y;
    params.blockDim.z = req->block_dim_z;
    params.sharedMemBytes = req->shared_mem;

    /* Extract kernel parameters from variable-length payload */
    uint32_t num_params = req->num_params;
    void** kernelParams = NULL;
    if (num_params > 0) {
        kernelParams = (void**)malloc((num_params + 1) * sizeof(void*));
        if (!kernelParams) {
            send_simple_response(fd, HIP_OP_GRAPH_EXEC_KERNEL_NODE_SET_PARAMS, request_id, hipErrorOutOfMemory);
            return;
        }

        const uint64_t* param_ptrs = (const uint64_t*)((const uint8_t*)payload +
                                                        sizeof(HipRemoteGraphExecKernelNodeSetParamsRequest));
        for (uint32_t i = 0; i < num_params; i++) {
            kernelParams[i] = (void*)(uintptr_t)param_ptrs[i];
        }
        kernelParams[num_params] = NULL;
    }
    params.kernelParams = kernelParams;
    params.extra = NULL;

    hipError_t err = hipGraphExecKernelNodeSetParams(graphExec, node, &params);
    LOG_DEBUG("GraphExecKernelNodeSetParams: graphExec=%p, node=%p, func=%p, err=%d",
              (void*)graphExec, (void*)node, params.func, err);

    free(kernelParams);
    send_simple_response(fd, HIP_OP_GRAPH_EXEC_KERNEL_NODE_SET_PARAMS, request_id, err);
}

static void handle_stream_begin_capture(int fd, uint32_t request_id,
                                        const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteStreamBeginCaptureRequest)) {
        send_simple_response(fd, HIP_OP_STREAM_BEGIN_CAPTURE, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteStreamBeginCaptureRequest* req = (const HipRemoteStreamBeginCaptureRequest*)payload;
    hipStream_t stream = vaddr_translate(req->stream);
    hipError_t err = hipStreamBeginCapture(stream, (hipStreamCaptureMode)req->mode);
    LOG_DEBUG("StreamBeginCapture: stream=%p, mode=%d, err=%d",
              (void*)stream, req->mode, err);
    send_simple_response(fd, HIP_OP_STREAM_BEGIN_CAPTURE, request_id, err);
}

static void handle_stream_end_capture(int fd, uint32_t request_id,
                                      const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteStreamEndCaptureRequest)) {
        send_simple_response(fd, HIP_OP_STREAM_END_CAPTURE, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteStreamEndCaptureRequest* req = (const HipRemoteStreamEndCaptureRequest*)payload;
    hipStream_t stream = vaddr_translate(req->stream);
    hipGraph_t graph = NULL;
    hipError_t err = hipStreamEndCapture(stream, &graph);
    LOG_DEBUG("StreamEndCapture: stream=%p, graph=%p, err=%d",
              (void*)stream, (void*)graph, err);

    HipRemoteStreamEndCaptureResponse resp = {
        .header = { .error_code = (int32_t)err },
        .graph = (uint64_t)(uintptr_t)graph
    };
    send_response(fd, HIP_OP_STREAM_END_CAPTURE, request_id, &resp, sizeof(resp));
}

static void handle_stream_is_capturing(int fd, uint32_t request_id,
                                       const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteStreamIsCapturingRequest)) {
        send_simple_response(fd, HIP_OP_STREAM_IS_CAPTURING, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteStreamIsCapturingRequest* req = (const HipRemoteStreamIsCapturingRequest*)payload;
    hipStream_t stream = vaddr_translate(req->stream);
    hipStreamCaptureStatus captureStatus = hipStreamCaptureStatusNone;
    hipError_t err = hipStreamIsCapturing(stream, &captureStatus);
    LOG_DEBUG("StreamIsCapturing: stream=%p, status=%d, err=%d",
              (void*)stream, captureStatus, err);

    HipRemoteStreamIsCapturingResponse resp = {
        .header = { .error_code = (int32_t)err },
        .capture_status = (int32_t)captureStatus
    };
    send_response(fd, HIP_OP_STREAM_IS_CAPTURING, request_id, &resp, sizeof(resp));
}

/* ============================================================================
 * Module and Kernel Handlers
 * ============================================================================ */

static void handle_module_load_data(int fd, uint32_t request_id,
                                     const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteModuleLoadRequest)) {
        send_simple_response(fd, HIP_OP_MODULE_LOAD_DATA, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteModuleLoadRequest* req = (const HipRemoteModuleLoadRequest*)payload;
    const void* code_data = (const uint8_t*)payload + sizeof(HipRemoteModuleLoadRequest);
    size_t code_size = payload_size - sizeof(HipRemoteModuleLoadRequest);

    if (code_size < req->data_size) {
        LOG_ERROR("ModuleLoadData: incomplete data (got %zu, expected %lu)", code_size, req->data_size);
        send_simple_response(fd, HIP_OP_MODULE_LOAD_DATA, request_id, hipErrorInvalidValue);
        return;
    }

    /* Drain pending async GPU errors so hipModuleLoadData doesn't hang.
     * Log the error but proceed -- MIOpen's solver search triggers many
     * module loads and can recover from transient GPU errors. */
    hipError_t sync_err = hipDeviceSynchronize();
    if (sync_err != hipSuccess) {
        LOG_ERROR("ModuleLoadData: GPU had pending error: %d (%s), clearing",
                  sync_err, hipGetErrorString(sync_err));
        hipGetLastError();
    }

    hipModule_t module = NULL;
    hipError_t err = hipModuleLoadData(&module, code_data);
    LOG_DEBUG("ModuleLoadData: size=%lu, module=%p, err=%d", req->data_size, (void*)module, err);

    if (err == hipSuccess && module != NULL) {
        store_module_data(module, code_data, code_size);
        if (g_debug_enabled && code_size >= 16) {
            const uint8_t* b = (const uint8_t*)code_data;
            fprintf(stderr, "[HIP-Worker] ModuleLoadData first 16 bytes: %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x\n",
                    b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]);
        }
    }

    HipRemoteModuleLoadResponse resp = {
        .header = { .error_code = (int32_t)err },
        .module = (uint64_t)(uintptr_t)module
    };
    send_response(fd, HIP_OP_MODULE_LOAD_DATA, request_id, &resp, sizeof(resp));
}

static void handle_module_unload(int fd, uint32_t request_id,
                                  const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteModuleUnloadRequest)) {
        send_simple_response(fd, HIP_OP_MODULE_UNLOAD, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteModuleUnloadRequest* req = (const HipRemoteModuleUnloadRequest*)payload;
    hipModule_t module = (hipModule_t)(uintptr_t)req->module;

    /* Invalidate cached kernel arg metadata for this module BEFORE unloading,
     * since HIP may reuse the module handle for a future hipModuleLoadData.
     * Without this, stale cached metadata (wrong is_pointer flags, wrong param
     * sizes) would be returned for the new module's kernels. */
    for (int i = 0; i < g_kernel_arg_cache_count; i++) {
        if (g_kernel_arg_cache[i].valid && g_kernel_arg_cache[i].module == module) {
            g_kernel_arg_cache[i].valid = 0;
        }
    }
    for (int i = 0; i < g_loaded_module_count; i++) {
        if (g_loaded_modules[i].module == module) {
            free(g_loaded_modules[i].data);
            g_loaded_modules[i].data = NULL;
            g_loaded_modules[i].size = 0;
            g_loaded_modules[i].module = NULL;
        }
    }

    hipError_t err = hipModuleUnload(module);
    LOG_DEBUG("ModuleUnload: module=%p, err=%d", (void*)module, err);
    send_simple_response(fd, HIP_OP_MODULE_UNLOAD, request_id, err);
}

static void handle_module_get_function(int fd, uint32_t request_id,
                                        const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteModuleGetFunctionRequest)) {
        send_simple_response(fd, HIP_OP_MODULE_GET_FUNCTION, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteModuleGetFunctionRequest* req = (const HipRemoteModuleGetFunctionRequest*)payload;
    hipModule_t module = (hipModule_t)(uintptr_t)req->module;

    hipFunction_t function = NULL;
    hipError_t err = hipModuleGetFunction(&function, module, req->function_name);

    HipRemoteModuleGetFunctionResponse resp;
    memset(&resp, 0, sizeof(resp));
    resp.header.error_code = (int32_t)err;
    resp.function = (uint64_t)(uintptr_t)function;

    if (err == hipSuccess && function != NULL) {
        /* Check cache first to avoid slow COMGR re-parsing */
        uint32_t np = 0;
        const CachedKernelArgs* cached = find_cached_kernel_args(module, req->function_name);
        uint32_t comgr_segment_size = 0;
        if (cached) {
            np = cached->num_params;
            comgr_segment_size = cached->kernarg_segment_size;
            memcpy(resp.params, cached->params, np * sizeof(HipRemoteParamDesc));
        } else {
            const LoadedModuleEntry* mod_entry = find_module_data(module);
            if (mod_entry && mod_entry->data) {
                np = comgr_extract_kernel_params(mod_entry->data, mod_entry->size,
                                                  req->function_name,
                                                  resp.params, HIP_REMOTE_MAX_PARAM_DESCS,
                                                  &comgr_segment_size);
            }
            /* Cross-module fallback: if COMGR didn't find the kernel in the
             * expected module, search all loaded modules. This handles cases
             * where hipModuleGetFunction finds the kernel but the client
             * registered it under a different fatbin module. */
            if (np == 0) {
                for (int mi = 0; mi < g_loaded_module_count && np == 0; mi++) {
                    if (g_loaded_modules[mi].data &&
                        g_loaded_modules[mi].module != module) {
                        np = comgr_extract_kernel_params(
                            g_loaded_modules[mi].data, g_loaded_modules[mi].size,
                            req->function_name,
                            resp.params, HIP_REMOTE_MAX_PARAM_DESCS,
                            &comgr_segment_size);
                    }
                }
            }
            cache_kernel_args(module, req->function_name, np, comgr_segment_size, resp.params);
        }
        resp.num_params = np;

        /* Use .kernarg_segment_size from code object metadata if available,
         * otherwise fall back to computing from user param offsets. */
        if (comgr_segment_size > 0) {
            resp.num_args = comgr_segment_size;
        } else {
            uint32_t user_args_end = 0;
            for (uint32_t i = 0; i < np; i++) {
                uint32_t end = resp.params[i].offset + resp.params[i].size;
                if (end > user_args_end) user_args_end = end;
            }
            resp.num_args = user_args_end > 0 ? user_args_end : 256;
        }

        LOG_DEBUG("ModuleGetFunction: module=%p, name=%s, function=%p, num_params=%u, kernarg_size=%u, segment_size=%u, err=%d",
                  (void*)module, req->function_name, (void*)function, np, resp.num_args, comgr_segment_size, err);

        store_kernarg_size(function, resp.num_args);
        if (resp.num_params > 0) {
            cache_function_info(function, resp.num_params,
                                resp.num_args, comgr_segment_size,
                                resp.params);
        }
    } else {
        LOG_DEBUG("ModuleGetFunction: module=%p, name=%s, err=%d", (void*)module, req->function_name, err);
    }

    send_response(fd, HIP_OP_MODULE_GET_FUNCTION, request_id, &resp, sizeof(resp));
}

static void handle_module_load_and_get_function(int fd, uint32_t request_id,
                                                 const void* payload, size_t payload_size) {
    size_t min_size = sizeof(HipRemoteModuleLoadRequest)
                    + sizeof(HipRemoteModuleLoadAndGetFunctionRequest);
    if (!payload || payload_size < min_size) {
        send_simple_response(fd, HIP_OP_MODULE_LOAD_AND_GET_FUNCTION, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteModuleLoadRequest* load_req = (const HipRemoteModuleLoadRequest*)payload;
    const HipRemoteModuleLoadAndGetFunctionRequest* func_req =
        (const HipRemoteModuleLoadAndGetFunctionRequest*)((const uint8_t*)payload + sizeof(HipRemoteModuleLoadRequest));

    uint32_t name_len = func_req->name_length;
    size_t name_offset = min_size;
    if (payload_size < name_offset + name_len) {
        LOG_ERROR("ModuleLoadAndGetFunction: name truncated");
        send_simple_response(fd, HIP_OP_MODULE_LOAD_AND_GET_FUNCTION, request_id, hipErrorInvalidValue);
        return;
    }

    char* kernel_name = (char*)malloc(name_len + 1);
    if (!kernel_name) {
        send_simple_response(fd, HIP_OP_MODULE_LOAD_AND_GET_FUNCTION, request_id, hipErrorOutOfMemory);
        return;
    }
    memcpy(kernel_name, (const uint8_t*)payload + name_offset, name_len);
    kernel_name[name_len] = '\0';

    const void* code_data = (const uint8_t*)payload + name_offset + name_len;
    size_t code_size = payload_size - name_offset - name_len;

    if (code_size < load_req->data_size) {
        LOG_ERROR("ModuleLoadAndGetFunction: incomplete data (got %zu, expected %lu)", code_size, load_req->data_size);
        free(kernel_name);
        send_simple_response(fd, HIP_OP_MODULE_LOAD_AND_GET_FUNCTION, request_id, hipErrorInvalidValue);
        return;
    }

    hipError_t sync_err = hipDeviceSynchronize();
    if (sync_err != hipSuccess) {
        LOG_ERROR("ModuleLoadAndGetFunction: GPU had pending error: %d (%s), clearing",
                  sync_err, hipGetErrorString(sync_err));
        hipGetLastError();
    }

    hipModule_t module = NULL;
    hipError_t err = hipModuleLoadData(&module, code_data);
    LOG_DEBUG("ModuleLoadAndGetFunction: load size=%lu module=%p err=%d", load_req->data_size, (void*)module, err);

    if (err != hipSuccess || !module) {
        HipRemoteModuleLoadAndGetFunctionResponse resp;
        memset(&resp, 0, sizeof(resp));
        resp.header.error_code = (int32_t)err;
        send_response(fd, HIP_OP_MODULE_LOAD_AND_GET_FUNCTION, request_id, &resp, sizeof(resp));
        free(kernel_name);
        return;
    }

    store_module_data(module, code_data, code_size);

    hipFunction_t function = NULL;
    err = hipModuleGetFunction(&function, module, kernel_name);

    HipRemoteModuleLoadAndGetFunctionResponse resp;
    memset(&resp, 0, sizeof(resp));
    resp.header.error_code = (int32_t)err;
    resp.module = (uint64_t)(uintptr_t)module;
    resp.function = (uint64_t)(uintptr_t)function;

    if (err == hipSuccess && function != NULL) {
        uint32_t np = 0;
        uint32_t comgr_segment_size = 0;
        const CachedKernelArgs* cached = find_cached_kernel_args(module, kernel_name);
        if (cached) {
            np = cached->num_params;
            comgr_segment_size = cached->kernarg_segment_size;
            memcpy(resp.params, cached->params, np * sizeof(HipRemoteParamDesc));
        } else {
            const LoadedModuleEntry* mod_entry = find_module_data(module);
            if (mod_entry && mod_entry->data) {
                np = comgr_extract_kernel_params(mod_entry->data, mod_entry->size,
                                                  kernel_name,
                                                  resp.params, HIP_REMOTE_MAX_PARAM_DESCS,
                                                  &comgr_segment_size);
            }
            if (np == 0) {
                for (int mi = 0; mi < g_loaded_module_count && np == 0; mi++) {
                    if (g_loaded_modules[mi].data &&
                        g_loaded_modules[mi].module != module) {
                        np = comgr_extract_kernel_params(
                            g_loaded_modules[mi].data, g_loaded_modules[mi].size,
                            kernel_name,
                            resp.params, HIP_REMOTE_MAX_PARAM_DESCS,
                            &comgr_segment_size);
                    }
                }
            }
            cache_kernel_args(module, kernel_name, np, comgr_segment_size, resp.params);
        }
        resp.num_params = np;

        if (comgr_segment_size > 0) {
            resp.kernarg_size = comgr_segment_size;
        } else {
            uint32_t user_args_end = 0;
            for (uint32_t i = 0; i < np; i++) {
                uint32_t end = resp.params[i].offset + resp.params[i].size;
                if (end > user_args_end) user_args_end = end;
            }
            resp.kernarg_size = user_args_end > 0 ? user_args_end : 256;
        }

        LOG_DEBUG("ModuleLoadAndGetFunction: name=%s function=%p num_params=%u kernarg_size=%u",
                  kernel_name, (void*)function, np, resp.kernarg_size);

        store_kernarg_size(function, resp.kernarg_size);
        if (np > 0) {
            cache_function_info(function, np, resp.kernarg_size, comgr_segment_size, resp.params);
        }
    } else {
        LOG_DEBUG("ModuleLoadAndGetFunction: name=%s err=%d", kernel_name, err);
    }

    free(kernel_name);
    send_response(fd, HIP_OP_MODULE_LOAD_AND_GET_FUNCTION, request_id, &resp, sizeof(resp));
}

static void handle_launch_kernel(int fd, uint32_t request_id,
                                  const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteLaunchKernelRequest)) {
        send_simple_response(fd, HIP_OP_LAUNCH_KERNEL, request_id, hipErrorInvalidValue);
        return;
    }

    const HipRemoteLaunchKernelRequest* req = (const HipRemoteLaunchKernelRequest*)payload;

    /* Validate sizes */
    size_t expected_min = sizeof(HipRemoteLaunchKernelRequest) +
                          req->num_args * sizeof(HipRemoteKernelArg);
    if (payload_size < expected_min) {
        LOG_ERROR("LaunchKernel: payload too small (got %zu, expected %zu)", payload_size, expected_min);
        send_simple_response(fd, HIP_OP_LAUNCH_KERNEL, request_id, hipErrorInvalidValue);
        return;
    }

    hipFunction_t function = (hipFunction_t)(uintptr_t)req->function;
    hipStream_t stream = vaddr_translate(req->stream);
    hipEvent_t start_event = (hipEvent_t)(uintptr_t)req->start_event;
    hipEvent_t stop_event  = (hipEvent_t)(uintptr_t)req->stop_event;
    unsigned int ext_flags = req->ext_flags;

    LOG_DEBUG("LaunchKernel: func=%p, grid=(%u,%u,%u), block=(%u,%u,%u), shared=%u, stream=%p, args=%u, startEvt=%p, stopEvt=%p",
              (void*)function, req->grid_dim_x, req->grid_dim_y, req->grid_dim_z,
              req->block_dim_x, req->block_dim_y, req->block_dim_z,
              req->shared_mem_bytes, (void*)stream, req->num_args,
              (void*)start_event, (void*)stop_event);

    /* Debug: dump first 64 bytes of arg data */
    if (g_debug_enabled) {
        const HipRemoteKernelArg* dbg_args = (const HipRemoteKernelArg*)((const uint8_t*)payload +
                                               sizeof(HipRemoteLaunchKernelRequest));
        const uint8_t* dbg_data = (const uint8_t*)(dbg_args + req->num_args);
        size_t dbg_total = 0;
        for (uint32_t i = 0; i < req->num_args && i < 8; i++) {
            size_t e = dbg_args[i].offset + dbg_args[i].size;
            if (e > dbg_total) dbg_total = e;
        }
        fprintf(stderr, "[HIP-Worker] ArgData (%zu bytes):", dbg_total);
        for (size_t i = 0; i < dbg_total && i < 64; i++) {
            if (i % 8 == 0) fprintf(stderr, " |");
            fprintf(stderr, " %02x", dbg_data[i]);
        }
        fprintf(stderr, "\n");
    }

    /* Extract kernel arguments as a flat buffer.
     * The client sends arg data packed sequentially. We pass it to the
     * HIP runtime via the 'extra' parameter (HIP_LAUNCH_PARAM_BUFFER_*),
     * which lets the runtime use kernel metadata to extract individual
     * args at their correct offsets from the flat buffer. */
    const HipRemoteKernelArg* arg_descs = (const HipRemoteKernelArg*)((const uint8_t*)payload +
                                           sizeof(HipRemoteLaunchKernelRequest));
    const uint8_t* arg_data = (const uint8_t*)(arg_descs + req->num_args);
    size_t total_arg_size = 0;
    for (uint32_t i = 0; i < req->num_args; i++) {
        size_t end = arg_descs[i].offset + arg_descs[i].size;
        if (end > total_arg_size) total_arg_size = end;
    }

    hipError_t err;

    if (req->launch_flags == 1) {
        /* Client sent a flat kernarg buffer via extra. Use hipExtModuleLaunchKernel
         * with the HIP_LAUNCH_PARAM_BUFFER path. The buffer size must be at least
         * kernarg_segment_size so the runtime can fill hidden args at dispatch time. */
        const CachedFunctionInfo* fi = lookup_function_info(function);
        size_t segment_size = fi ? fi->kernarg_segment_size : 0;

        size_t buf_size = total_arg_size > segment_size ? total_arg_size : segment_size;
        if (buf_size == 0) buf_size = total_arg_size;

        void* arg_copy = calloc(1, buf_size);
        if (!arg_copy) {
            send_simple_response(fd, HIP_OP_LAUNCH_KERNEL, request_id, hipErrorOutOfMemory);
            return;
        }
        memcpy(arg_copy, arg_data, total_arg_size);

        /* Translate virtual GPU addresses in the flat kernarg buffer.
         * Use COMGR metadata when available; fall back to scanning
         * 8-byte-aligned positions that are known vaddrs. */
        LOG_DEBUG("Vaddr translation: fi=%p num_params=%u total_arg_size=%zu",
                  (void*)fi, fi ? fi->num_params : 0, total_arg_size);
        if (fi && g_debug_enabled) {
            for (uint32_t pi = 0; pi < fi->num_params; pi++) {
                uint64_t val = 0;
                if (fi->params[pi].offset + fi->params[pi].size <= buf_size)
                    memcpy(&val, (uint8_t*)arg_copy + fi->params[pi].offset,
                           fi->params[pi].size < 8 ? fi->params[pi].size : 8);
                fprintf(stderr, "[HIP-Worker]   param[%u]: off=%u size=%u ptr=%d val=0x%lx\n",
                        pi, fi->params[pi].offset, fi->params[pi].size,
                        fi->params[pi].is_pointer, (unsigned long)val);
            }
        }
        /* Translate vaddrs using COMGR metadata:
         * - global_buffer (is_pointer=1): range lookup
         * - by_value, size == 8: translate if >= VADDR_BASE (Tensile marks
         *   pointer params as by_value; real scalars are always < VADDR_BASE)
         * - by_value, size > 8: scan ALL 8-byte sub-fields (PyTorch packs
         *   multiple tensor pointers into structs)
         * - by_value, size < 8: skip (definitely a scalar) */
        if (fi && fi->num_params > 0) {
            for (uint32_t pi = 0; pi < fi->num_params; pi++) {
                uint32_t poff = fi->params[pi].offset;
                uint32_t psize = fi->params[pi].size;

                if (psize >= 8) {
                    if (fi->params[pi].is_pointer || psize == 8) {
                        if (poff + 8 <= buf_size) {
                            uint64_t* slot = (uint64_t*)((uint8_t*)arg_copy + poff);
                            if (*slot >= VADDR_BASE) {
                                uint64_t translated = vaddr_map_get(*slot);
                                if (translated != *slot) *slot = translated;
                            }
                        }
                    } else {
                        for (uint32_t soff = 0; soff + 8 <= psize; soff += 8) {
                            if (poff + soff + 8 <= buf_size) {
                                uint64_t* slot = (uint64_t*)((uint8_t*)arg_copy + poff + soff);
                                if (*slot >= VADDR_BASE) {
                                    uint64_t translated = vaddr_map_get(*slot);
                                    if (translated != *slot) *slot = translated;
                                }
                            }
                        }
                    }
                }
            }
        } else {
            /* No COMGR metadata (e.g. Tensile/rocBLAS assembly kernels, or
             * flat buffer from hipExtModuleLaunchKernel). Scan all 8-byte-
             * aligned positions for vaddrs. Tensile kernel scalars (matrix
             * dimensions, strides, alpha/beta) are small values that won't
             * collide with the vaddr range (>= 0x7F0000000000). */
            for (uint32_t off = 0; off + 8 <= buf_size; off += 8) {
                uint64_t* slot = (uint64_t*)((uint8_t*)arg_copy + off);
                if (*slot >= VADDR_BASE) {
                    uint64_t translated = vaddr_map_get(*slot);
                    if (translated != *slot) *slot = translated;
                }
            }
        }

        void* config[] = {
            (void*)0x01, /* HIP_LAUNCH_PARAM_BUFFER_POINTER */
            arg_copy,
            (void*)0x02, /* HIP_LAUNCH_PARAM_BUFFER_SIZE */
            &buf_size,
            (void*)0x03  /* HIP_LAUNCH_PARAM_END */
        };
        /* hipExtModuleLaunchKernel takes globalWorkSize (total threads).
         * The protocol uses gridDim (number of blocks), so convert. */
        err = hipExtModuleLaunchKernel(
            function,
            req->grid_dim_x * req->block_dim_x,
            req->grid_dim_y * req->block_dim_y,
            req->grid_dim_z * req->block_dim_z,
            req->block_dim_x, req->block_dim_y, req->block_dim_z,
            req->shared_mem_bytes,
            stream,
            NULL,    /* kernelParams = NULL */
            config,  /* extra */
            start_event,
            stop_event,
            ext_flags
        );
        free(arg_copy);
    } else {
        /* Client sent individual kernelParams values. Reconstruct the
         * kernel_params array so the HIP runtime can use kernel metadata
         * to map each value to the correct kernarg offset. */
        uint8_t arg_data_copy[4096];
        size_t copy_size = total_arg_size < sizeof(arg_data_copy) ? total_arg_size : sizeof(arg_data_copy);
        memcpy(arg_data_copy, arg_data, copy_size);
        for (uint32_t i = 0; i < req->num_args; i++) {
            if (arg_descs[i].size == 8 && arg_descs[i].offset + 8 <= copy_size) {
                uint64_t* slot = (uint64_t*)(arg_data_copy + arg_descs[i].offset);
                if (*slot >= VADDR_BASE) {
                    uint64_t translated = vaddr_map_get(*slot);
                    if (translated != *slot) *slot = translated;
                }
            }
        }

        void* kernel_params[HIP_REMOTE_MAX_KERNEL_ARGS];
        for (uint32_t i = 0; i < req->num_args && i < HIP_REMOTE_MAX_KERNEL_ARGS; i++) {
            kernel_params[i] = (void*)(arg_data_copy + arg_descs[i].offset);
        }

        if (start_event || stop_event) {
            err = hipExtModuleLaunchKernel(
                function,
                req->grid_dim_x * req->block_dim_x,
                req->grid_dim_y * req->block_dim_y,
                req->grid_dim_z * req->block_dim_z,
                req->block_dim_x, req->block_dim_y, req->block_dim_z,
                req->shared_mem_bytes,
                stream,
                kernel_params,
                NULL,
                start_event,
                stop_event,
                ext_flags
            );
        } else {
            err = hipModuleLaunchKernel(
                function,
                req->grid_dim_x, req->grid_dim_y, req->grid_dim_z,
                req->block_dim_x, req->block_dim_y, req->block_dim_z,
                req->shared_mem_bytes,
                stream,
                kernel_params,
                NULL
            );
        }
    }

    LOG_DEBUG("LaunchKernel: err=%d", err);
    send_simple_response(fd, HIP_OP_LAUNCH_KERNEL, request_id, err);
}

/* ============================================================================
 * Memory Pool / Function Introspection Handlers
 * ============================================================================ */

static void handle_mempool_get_attribute(int fd, uint32_t request_id,
                                          const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteMemPoolAttrRequest)) {
        send_simple_response(fd, HIP_OP_MEMPOOL_GET_ATTRIBUTE, request_id, hipErrorInvalidValue);
        return;
    }
    const HipRemoteMemPoolAttrRequest* req = (const HipRemoteMemPoolAttrRequest*)payload;
    HipRemoteMemPoolAttrResponse resp;
    memset(&resp, 0, sizeof(resp));
    hipMemPool_t pool = (hipMemPool_t)(uintptr_t)req->mem_pool;
    hipError_t err = hipMemPoolGetAttribute(pool, (hipMemPoolAttr)req->attr, (void*)&resp.value);
    resp.header.error_code = (int32_t)err;
    send_response(fd, HIP_OP_MEMPOOL_GET_ATTRIBUTE, request_id, &resp, sizeof(resp));
}

static void handle_mempool_set_attribute(int fd, uint32_t request_id,
                                          const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteMemPoolSetAttrRequest)) {
        send_simple_response(fd, HIP_OP_MEMPOOL_SET_ATTRIBUTE, request_id, hipErrorInvalidValue);
        return;
    }
    const HipRemoteMemPoolSetAttrRequest* req = (const HipRemoteMemPoolSetAttrRequest*)payload;
    hipMemPool_t pool = (hipMemPool_t)(uintptr_t)req->mem_pool;
    hipError_t err = hipMemPoolSetAttribute(pool, (hipMemPoolAttr)req->attr, (void*)req->value);
    send_simple_response(fd, HIP_OP_MEMPOOL_SET_ATTRIBUTE, request_id, err);
}

static void handle_mempool_trim_to(int fd, uint32_t request_id,
                                    const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteMemPoolTrimRequest)) {
        send_simple_response(fd, HIP_OP_MEMPOOL_TRIM_TO, request_id, hipErrorInvalidValue);
        return;
    }
    const HipRemoteMemPoolTrimRequest* req = (const HipRemoteMemPoolTrimRequest*)payload;
    hipMemPool_t pool = (hipMemPool_t)(uintptr_t)req->mem_pool;
    hipError_t err = hipMemPoolTrimTo(pool, (size_t)req->min_bytes_to_keep);
    send_simple_response(fd, HIP_OP_MEMPOOL_TRIM_TO, request_id, err);
}

static void handle_mem_ptr_get_info(int fd, uint32_t request_id,
                                     const void* payload, size_t payload_size) {
    if (!payload || payload_size < sizeof(HipRemoteMemPtrGetInfoRequest)) {
        send_simple_response(fd, HIP_OP_MEM_PTR_GET_INFO, request_id, hipErrorInvalidValue);
        return;
    }
    const HipRemoteMemPtrGetInfoRequest* req = (const HipRemoteMemPtrGetInfoRequest*)payload;
    HipRemoteMemPtrGetInfoResponse resp;
    memset(&resp, 0, sizeof(resp));
    size_t sz = 0;
    void* ptr = vaddr_translate(req->ptr);
    hipError_t err = hipMemPtrGetInfo(ptr, &sz);
    resp.header.error_code = (int32_t)err;
    resp.size = (uint64_t)sz;
    send_response(fd, HIP_OP_MEM_PTR_GET_INFO, request_id, &resp, sizeof(resp));
}

/* ============================================================================
 * Client Handler
 * ============================================================================ */

static void reset_session_state(void) {
    memset(g_vaddr_map, 0, sizeof(g_vaddr_map));
    g_vaddr_alloc_count = 0;
    g_vaddr_allocs_sorted = 1;
    g_vaddr_cache_vbase = 0;
    g_vaddr_cache_rbase = 0;
    g_vaddr_cache_size = 0;
    g_deferred_alloc_error = hipSuccess;

    g_func_cache_count = 0;
    g_loaded_module_count = 0;
    g_kernarg_size_count = 0;
    g_kernel_arg_cache_count = 0;
}

static void handle_client(int client_fd) {
    LOG_INFO("Client connected (worker pid %d)", (int)getpid());

    while (g_running) {
        /* Read header */
        HipRemoteHeader header;
        if (recv_all(client_fd, &header, sizeof(header)) != 0) {
            LOG_DEBUG("Client disconnected");
            break;
        }

        if (hip_remote_validate_header(&header) != 0) {
            LOG_ERROR("Invalid header from client");
            break;
        }

        LOG_DEBUG("Request: %s (id=%u, payload=%lu)",
                  hip_remote_op_name((HipRemoteOpCode)header.op_code),
                  header.request_id, header.payload_length);

        /* Read payload if present */
        void* payload = NULL;
        if (header.payload_length > 0) {
            payload = malloc(header.payload_length);
            if (!payload) {
                LOG_ERROR("Failed to allocate payload buffer");
                break;
            }
            if (recv_all(client_fd, payload, header.payload_length) != 0) {
                LOG_ERROR("Failed to receive payload");
                free(payload);
                break;
            }
        }

        bool has_inline_data = (header.flags & HIP_REMOTE_FLAG_HAS_INLINE_DATA) != 0;

        g_suppress_response = (header.flags & HIP_REMOTE_FLAG_NO_REPLY) != 0;

        /* Dispatch */
        switch ((HipRemoteOpCode)header.op_code) {
            case HIP_OP_INIT:
                handle_init(client_fd, header.request_id);
                break;
            case HIP_OP_SHUTDOWN:
                handle_shutdown(client_fd, header.request_id);
                free(payload);
                goto client_done;

            case HIP_OP_GET_DEVICE_COUNT:
                handle_get_device_count(client_fd, header.request_id);
                break;
            case HIP_OP_SET_DEVICE:
                handle_set_device(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_GET_DEVICE:
                handle_get_device(client_fd, header.request_id);
                break;
            case HIP_OP_DEVICE_SYNCHRONIZE:
                handle_device_synchronize(client_fd, header.request_id);
                break;
            case HIP_OP_DEVICE_GET_ATTRIBUTE:
                handle_device_get_attribute(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_GET_DEVICE_PROPERTIES:
                handle_get_device_properties(client_fd, header.request_id, payload, header.payload_length);
                break;

            case HIP_OP_MALLOC:
                handle_malloc(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_MALLOC_BATCH:
                handle_malloc_batch(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_FREE:
                handle_free(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_MALLOC_HOST:
                handle_malloc_host(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_FREE_HOST:
                handle_free_host(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_MALLOC_ASYNC:
                handle_malloc_async(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_MALLOC_VADDR:
                handle_malloc_vaddr(client_fd, header.request_id, payload, header.payload_length, 0);
                break;
            case HIP_OP_MALLOC_ASYNC_VADDR:
                handle_malloc_vaddr(client_fd, header.request_id, payload, header.payload_length, 1);
                break;
            case HIP_OP_FREE_ASYNC:
                handle_free_async(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_MEMCPY:
            case HIP_OP_MEMCPY_ASYNC:
                handle_memcpy(client_fd, header.request_id, payload, header.payload_length, has_inline_data);
                break;
            case HIP_OP_MEMCPY_HTOD_CACHED:
                handle_memcpy_htod_cached(client_fd, header.request_id, payload, header.payload_length, has_inline_data);
                break;
            case HIP_OP_MEMCPY_2D:
                handle_memcpy2d(client_fd, header.request_id, payload, header.payload_length, has_inline_data, false);
                break;
            case HIP_OP_MEMCPY_2D_ASYNC:
                handle_memcpy2d(client_fd, header.request_id, payload, header.payload_length, has_inline_data, true);
                break;
            case HIP_OP_MEMSET:
            case HIP_OP_MEMSET_ASYNC:
                handle_memset(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_MEM_GET_INFO:
                handle_mem_get_info(client_fd, header.request_id);
                break;
            case HIP_OP_POINTER_GET_ATTRIBUTES:
                handle_pointer_get_attributes(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_POINTER_GET_ATTRIBUTE:
                handle_pointer_get_attribute(client_fd, header.request_id, payload, header.payload_length);
                break;

            /* IPC operations */
            case HIP_OP_IPC_GET_MEM_HANDLE:
                handle_ipc_get_mem_handle(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_IPC_OPEN_MEM_HANDLE:
                handle_ipc_open_mem_handle(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_IPC_CLOSE_MEM_HANDLE:
                handle_ipc_close_mem_handle(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_IPC_GET_EVENT_HANDLE:
                handle_ipc_get_event_handle(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_IPC_OPEN_EVENT_HANDLE:
                handle_ipc_open_event_handle(client_fd, header.request_id, payload, header.payload_length);
                break;

            /* Memory Pool operations */
            case HIP_OP_MEM_POOL_CREATE:
                handle_mem_pool_create(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_MEM_POOL_DESTROY:
                handle_mem_pool_destroy(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_MEM_POOL_SET_ATTRIBUTE:
                handle_mem_pool_set_attribute(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_MEM_POOL_GET_ATTRIBUTE:
                handle_mem_pool_get_attribute(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_MALLOC_FROM_POOL_ASYNC:
                handle_malloc_from_pool_async(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_MEM_POOL_TRIM_TO:
                handle_mem_pool_trim_to(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_DEVICE_GET_DEFAULT_MEM_POOL:
                handle_device_get_default_mem_pool(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_DEVICE_SET_MEM_POOL:
                handle_device_set_mem_pool(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_DEVICE_GET_MEM_POOL:
                handle_device_get_mem_pool(client_fd, header.request_id, payload, header.payload_length);
                break;

            /* Host memory registration */
            case HIP_OP_HOST_REGISTER:
                handle_host_register(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_HOST_UNREGISTER:
                handle_host_unregister(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_HOST_GET_DEVICE_POINTER:
                handle_host_get_device_pointer(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_HOST_GET_FLAGS:
                handle_host_get_flags(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_HOST_ALLOC:
                handle_host_alloc(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_HOST_FREE:
                handle_host_free(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_MEM_ALLOC_PITCH:
                handle_mem_alloc_pitch(client_fd, header.request_id, payload, header.payload_length);
                break;

            /* Unified memory management */
            case HIP_OP_MEM_ADVISE:
                handle_mem_advise(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_MEM_PREFETCH_ASYNC:
                handle_mem_prefetch_async(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_MEM_RANGE_GET_ATTRIBUTE:
                handle_mem_range_get_attribute(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_MEM_RANGE_GET_ATTRIBUTES:
                handle_mem_range_get_attributes(client_fd, header.request_id, payload, header.payload_length);
                break;

            /* Graph Node operations */
            case HIP_OP_GRAPH_ADD_MEMCPY_NODE_1D:
                handle_graph_add_memcpy_node_1d(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_GRAPH_ADD_MEMSET_NODE:
                handle_graph_add_memset_node(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_GRAPH_ADD_EMPTY_NODE:
                handle_graph_add_empty_node(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_GRAPH_ADD_DEPENDENCIES:
                handle_graph_add_dependencies(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_GRAPH_GET_NODES:
                handle_graph_get_nodes(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_GRAPH_GET_ROOT_NODES:
                handle_graph_get_root_nodes(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_GRAPH_NODE_GET_TYPE:
                handle_graph_node_get_type(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_GRAPH_DESTROY_NODE:
                handle_graph_destroy_node(client_fd, header.request_id, payload, header.payload_length);
                break;

            case HIP_OP_MEMCPY_3D:
                handle_memcpy3d(client_fd, header.request_id, payload, header.payload_length, false);
                break;
            case HIP_OP_MEMCPY_3D_ASYNC:
                handle_memcpy3d(client_fd, header.request_id, payload, header.payload_length, true);
                break;
            case HIP_OP_MEMCPY_PEER:
                handle_memcpy_peer(client_fd, header.request_id, payload, header.payload_length, false);
                break;
            case HIP_OP_MEMCPY_PEER_ASYNC:
                handle_memcpy_peer(client_fd, header.request_id, payload, header.payload_length, true);
                break;

            case HIP_OP_STREAM_CREATE:
            case HIP_OP_STREAM_CREATE_WITH_FLAGS:
            case HIP_OP_STREAM_CREATE_WITH_PRIORITY:
                handle_stream_create(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_STREAM_DESTROY:
                handle_stream_destroy(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_STREAM_SYNCHRONIZE:
                handle_stream_synchronize(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_STREAM_GET_FLAGS:
                handle_stream_get_flags(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_STREAM_GET_PRIORITY:
                handle_stream_get_priority(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_STREAM_WAIT_EVENT:
                handle_stream_wait_event(client_fd, header.request_id, payload, header.payload_length);
                break;

            case HIP_OP_EVENT_CREATE:
            case HIP_OP_EVENT_CREATE_WITH_FLAGS:
                handle_event_create(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_EVENT_CREATE_BATCH:
                handle_event_create_batch(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_STREAM_CREATE_BATCH:
                handle_stream_create_batch(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_EVENT_DESTROY:
                handle_event_destroy(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_EVENT_RECORD:
                handle_event_record(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_EVENT_SYNCHRONIZE:
                handle_event_synchronize(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_EVENT_QUERY:
                handle_event_query(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_EVENT_ELAPSED_TIME:
                handle_event_elapsed_time(client_fd, header.request_id, payload, header.payload_length);
                break;

            case HIP_OP_RUNTIME_GET_VERSION:
                handle_runtime_get_version(client_fd, header.request_id);
                break;
            case HIP_OP_DRIVER_GET_VERSION:
                handle_driver_get_version(client_fd, header.request_id);
                break;

            case HIP_OP_DEVICE_GET_LIMIT:
                handle_device_get_limit(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_DEVICE_SET_LIMIT:
                handle_device_set_limit(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_DEVICE_CAN_ACCESS_PEER:
                handle_device_can_access_peer(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_DEVICE_ENABLE_PEER_ACCESS:
                handle_device_enable_peer_access(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_DEVICE_DISABLE_PEER_ACCESS:
                handle_device_disable_peer_access(client_fd, header.request_id, payload, header.payload_length);
                break;

            /* Device driver APIs */
            case HIP_OP_DEVICE_GET:
                handle_device_get(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_DEVICE_GET_NAME:
                handle_device_get_name(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_DEVICE_TOTAL_MEM:
                handle_device_total_mem(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_DEVICE_GET_PCI_BUS_ID:
                handle_device_get_pci_bus_id(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_DEVICE_GET_BY_PCI_BUS_ID:
                handle_device_get_by_pci_bus_id(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_DEVICE_COMPUTE_CAPABILITY:
                handle_device_compute_capability(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_DEVICE_GET_UUID:
                handle_device_get_uuid(client_fd, header.request_id, payload, header.payload_length);
                break;

            /* Device cache/config APIs */
            case HIP_OP_DEVICE_GET_CACHE_CONFIG:
                handle_device_get_cache_config(client_fd, header.request_id);
                break;
            case HIP_OP_DEVICE_SET_CACHE_CONFIG:
                handle_device_set_cache_config(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_DEVICE_GET_SHARED_MEM_CONFIG:
                handle_device_get_shared_mem_config(client_fd, header.request_id);
                break;
            case HIP_OP_DEVICE_SET_SHARED_MEM_CONFIG:
                handle_device_set_shared_mem_config(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_GET_DEVICE_FLAGS:
                handle_get_device_flags(client_fd, header.request_id);
                break;
            case HIP_OP_SET_DEVICE_FLAGS:
                handle_set_device_flags(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_DEVICE_GET_P2P_ATTRIBUTE:
                handle_device_get_p2p_attribute(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_DEVICE_GET_STREAM_PRIORITY_RANGE:
                handle_device_get_stream_priority_range(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_SET_VALID_DEVICES:
                handle_set_valid_devices(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_CHOOSE_DEVICE:
                handle_choose_device(client_fd, header.request_id, payload, header.payload_length);
                break;

            case HIP_OP_OCCUPANCY_MAX_POTENTIAL_BLOCK_SIZE:
                handle_occupancy_max_potential_block_size(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_OCCUPANCY_MAX_ACTIVE_BLOCKS_PER_SM:
                handle_occupancy_max_active_blocks_per_sm(client_fd, header.request_id, payload, header.payload_length);
                break;

            case HIP_OP_GRAPH_CREATE:
                handle_graph_create(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_GRAPH_DESTROY:
                handle_graph_destroy(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_GRAPH_INSTANTIATE:
                handle_graph_instantiate(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_GRAPH_LAUNCH:
                handle_graph_launch(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_GRAPH_EXEC_DESTROY:
                handle_graph_exec_destroy(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_GRAPH_CLONE:
                handle_graph_clone(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_GRAPH_NODE_GET_DEPENDENCIES:
                handle_graph_node_get_dependencies(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_GRAPH_NODE_GET_DEPENDENT_NODES:
                handle_graph_node_get_dependent_nodes(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_GRAPH_EXEC_UPDATE:
                handle_graph_exec_update(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_GRAPH_EXEC_KERNEL_NODE_SET_PARAMS:
                handle_graph_exec_kernel_node_set_params(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_STREAM_BEGIN_CAPTURE:
                handle_stream_begin_capture(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_STREAM_END_CAPTURE:
                handle_stream_end_capture(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_STREAM_IS_CAPTURING:
                handle_stream_is_capturing(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_STREAM_GET_CAPTURE_INFO:
                handle_stream_get_capture_info(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_STREAM_UPDATE_CAPTURE_DEPENDENCIES:
                handle_stream_update_capture_dependencies(client_fd, header.request_id, payload, header.payload_length);
                break;

            case HIP_OP_GET_LAST_ERROR:
                handle_get_last_error(client_fd, header.request_id);
                break;
            case HIP_OP_PEEK_AT_LAST_ERROR:
                handle_peek_at_last_error(client_fd, header.request_id);
                break;

            case HIP_OP_MODULE_LOAD_DATA:
            case HIP_OP_MODULE_LOAD_DATA_EX:
                handle_module_load_data(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_MODULE_UNLOAD:
                handle_module_unload(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_MODULE_GET_FUNCTION:
                handle_module_get_function(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_MODULE_LOAD_AND_GET_FUNCTION:
                handle_module_load_and_get_function(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_LAUNCH_KERNEL:
            case HIP_OP_MODULE_LAUNCH_KERNEL:
                handle_launch_kernel(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_FUNC_GET_ATTRIBUTES:
                handle_func_get_attributes(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_FUNC_SET_ATTRIBUTE:
                handle_func_set_attribute(client_fd, header.request_id, payload, header.payload_length);
                break;
            case HIP_OP_FUNC_SET_CACHE_CONFIG:
                handle_func_set_cache_config(client_fd, header.request_id, payload, header.payload_length);
                break;

            case HIP_OP_MEMPOOL_SET_ACCESS:
                send_simple_response(client_fd, (HipRemoteOpCode)header.op_code, header.request_id, hipSuccess);
                break;
            case HIP_OP_MEM_PTR_GET_INFO:
                handle_mem_ptr_get_info(client_fd, header.request_id, payload, header.payload_length);
                break;

            default:
#ifdef HIP_WORKER_SMI_ENABLED
                /* Check if this is an SMI operation (0x08xx range) */
                if ((header.op_code & 0xFF00) == 0x0800) {
                    smi_worker_dispatch(client_fd, header.op_code, header.request_id,
                                       payload, header.payload_length);
                    break;
                }
#endif
                LOG_ERROR("Unknown opcode: 0x%04x", header.op_code);
                send_simple_response(client_fd, (HipRemoteOpCode)header.op_code,
                                     header.request_id, hipErrorNotSupported);
                break;
        }

        g_suppress_response = 0;
        free(payload);
    }

client_done:
    close_socket(client_fd);
    LOG_INFO("Client disconnected");
}

/* ============================================================================
 * Signal Handling
 * ============================================================================ */

#ifndef _WIN32
static void sigchld_handler(int sig) {
    (void)sig;
    while (waitpid(-1, NULL, WNOHANG) > 0);
}
#endif

static void signal_handler(int sig) {
    (void)sig;
    g_running = false;
    if (g_server_fd >= 0) {
#ifdef _WIN32
        shutdown(g_server_fd, SD_BOTH);
#else
        shutdown(g_server_fd, SHUT_RDWR);
#endif
        close_socket(g_server_fd);
        g_server_fd = -1;
    }
}

/* ============================================================================
 * Main
 * ============================================================================ */

static void print_usage(const char* prog) {
    fprintf(stderr, "Usage: %s [options]\n", prog);
    fprintf(stderr, "\nOptions:\n");
    fprintf(stderr, "  -p PORT    Listen port (default: %d)\n", HIP_REMOTE_DEFAULT_PORT);
    fprintf(stderr, "  -d DEVICE  Default GPU device (default: 0)\n");
    fprintf(stderr, "  -v         Enable verbose logging\n");
    fprintf(stderr, "  -h         Show this help\n");
    fprintf(stderr, "\nEnvironment:\n");
    fprintf(stderr, "  TF_WORKER_PORT     Listen port\n");
    fprintf(stderr, "  TF_DEVICE_ID       Default device\n");
    fprintf(stderr, "  TF_DEBUG           Enable debug (1/0)\n");
}

int main(int argc, char** argv) {
    /* Parse environment */
    const char* port_str = getenv("TF_WORKER_PORT");
    if (port_str) g_listen_port = atoi(port_str);

    const char* device_str = getenv("TF_DEVICE_ID");
    if (device_str) g_default_device = atoi(device_str);

    const char* debug = getenv("TF_DEBUG");
    if (debug && strcmp(debug, "1") == 0) g_debug_enabled = true;

    /* Parse arguments */
#ifndef _WIN32
    int opt;
    while ((opt = getopt(argc, argv, "p:d:vh")) != -1) {
        switch (opt) {
            case 'p':
                g_listen_port = atoi(optarg);
                break;
            case 'd':
                g_default_device = atoi(optarg);
                break;
            case 'v':
                g_debug_enabled = true;
                break;
            case 'h':
                print_usage(argv[0]);
                return 0;
            default:
                print_usage(argv[0]);
                return 1;
        }
    }
#else
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) g_listen_port = atoi(argv[++i]);
        else if (strcmp(argv[i], "-d") == 0 && i + 1 < argc) g_default_device = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) g_debug_enabled = true;
        else if (strcmp(argv[i], "-h") == 0) { print_usage(argv[0]); return 0; }
    }
#endif

#ifdef _WIN32
    worker_socket_init();
    signal(SIGINT, signal_handler);
#else
    signal(SIGPIPE, SIG_IGN);
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    signal(SIGCHLD, sigchld_handler);
#endif

    /* Check if content caching is requested */
    const char* cache_env = getenv("HIP_REMOTE_CACHE");
    if (cache_env && strcmp(cache_env, "1") == 0) {
        gpu_cache_init();
        if (g_gpu_cache_max_explicit) {
            LOG_INFO("Content cache enabled (max %llu MB, sequential mode)",
                     (unsigned long long)(g_gpu_cache_max_bytes / (1024 * 1024)));
        } else {
            LOG_INFO("Content cache enabled (%.0f%% of VRAM, sequential mode)",
                     GPU_CACHE_DEFAULT_VRAM_FRACTION * 100);
        }
    }

    /* Do NOT initialize HIP in the parent process (when forking).
     * HIP GPU contexts don't survive fork() -- the child process must
     * initialize HIP from scratch. We defer all HIP calls to handle_client().
     * In cache mode (sequential), HIP init is deferred to first client. */
    LOG_INFO("Default device: %d (HIP init deferred to client handlers)", g_default_device);

#ifdef HIP_WORKER_SMI_ENABLED
    /* Initialize AMD SMI */
    if (smi_worker_init() == 0) {
        LOG_INFO("AMD SMI: %u GPU(s) available", smi_worker_get_processor_count());
    } else {
        LOG_INFO("AMD SMI: Not available (continuing without SMI support)");
    }
#endif

    /* Create server socket */
    g_server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (g_server_fd < 0) {
        LOG_ERROR("Failed to create socket: %s", strerror(errno));
        return 1;
    }

    int opt_val = 1;
    setsockopt(g_server_fd, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt_val, sizeof(opt_val));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons((uint16_t)g_listen_port);

    if (bind(g_server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        LOG_ERROR("Failed to bind to port %d: %s", g_listen_port, strerror(errno));
        return 1;
    }

    if (listen(g_server_fd, 256) < 0) {
        LOG_ERROR("Failed to listen: %s", strerror(errno));
        return 1;
    }

    LOG_INFO("Listening on port %d", g_listen_port);

    /* Accept clients (fork-per-client for concurrent connections) */
    while (g_running) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(g_server_fd, (struct sockaddr*)&client_addr, &client_len);

        if (client_fd < 0) {
            if (errno == EINTR) continue;
            if (!g_running) break;
            LOG_ERROR("Accept failed: %s", strerror(errno));
            continue;
        }

        int nodelay = 1;
        setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, (const char*)&nodelay, sizeof(nodelay));
        int keepalive = 1;
        setsockopt(client_fd, SOL_SOCKET, SO_KEEPALIVE, (const char*)&keepalive, sizeof(keepalive));
#ifdef _WIN32
        if (g_gpu_cache_enabled) {
            DWORD tv_ms = 120000;
            setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv_ms, sizeof(tv_ms));
        }
#else
        int keepidle = g_gpu_cache_enabled ? 5 : 300;
        int keepintvl = g_gpu_cache_enabled ? 2 : 60;
        int keepcnt = g_gpu_cache_enabled ? 3 : 5;
        if (g_gpu_cache_enabled) {
            struct timeval tv = { .tv_sec = 120, .tv_usec = 0 };
            setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        }
        setsockopt(client_fd, IPPROTO_TCP, TCP_KEEPIDLE, &keepidle, sizeof(keepidle));
        setsockopt(client_fd, IPPROTO_TCP, TCP_KEEPINTVL, &keepintvl, sizeof(keepintvl));
        setsockopt(client_fd, IPPROTO_TCP, TCP_KEEPCNT, &keepcnt, sizeof(keepcnt));
#endif

#ifdef _WIN32
        /* Windows: always sequential mode (no fork) */
        if (!g_hip_initialized) {
            hipError_t herr = hipSetDevice(g_default_device);
            if (herr != hipSuccess) {
                LOG_ERROR("HIP init failed: %s", hipGetErrorString(herr));
                close_socket(client_fd);
                continue;
            }
            g_hip_initialized = 1;
            if (g_gpu_cache_enabled) gpu_cache_size_from_vram();
        }
        reset_session_state();
        handle_client(client_fd);
#else
        if (g_gpu_cache_enabled) {
            if (!g_hip_initialized) {
                hipError_t herr = hipSetDevice(g_default_device);
                if (herr != hipSuccess) {
                    LOG_ERROR("HIP init failed: %s", hipGetErrorString(herr));
                    close(client_fd);
                    continue;
                }
                g_hip_initialized = 1;
                gpu_cache_size_from_vram();
            }
            reset_session_state();
            handle_client(client_fd);
            continue;
        }

        pid_t pid = fork();
        if (pid < 0) {
            LOG_ERROR("fork failed: %s", strerror(errno));
            close(client_fd);
            continue;
        }

        if (pid == 0) {
            close(g_server_fd);
            hipError_t herr = hipSetDevice(g_default_device);
            if (herr != hipSuccess) {
                LOG_ERROR("Child HIP init failed: %s", hipGetErrorString(herr));
                _exit(1);
            }

            handle_client(client_fd);
            _exit(0);
        }

        close(client_fd);
#endif
    }

    if (g_server_fd >= 0) {
        close_socket(g_server_fd);
    }

#ifdef HIP_WORKER_SMI_ENABLED
    smi_worker_shutdown();
#endif

    LOG_INFO("Shutting down");
    return 0;
}
