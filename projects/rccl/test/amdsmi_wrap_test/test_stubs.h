/*************************************************************************
 * Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file test_stubs.h
 * @brief Stub definitions for standalone testing of amdsmi_wrap
 */

#ifndef TEST_STUBS_H_
#define TEST_STUBS_H_

#include <cstdio>
#include <cstdlib>
#include <cstdint>

// Include nccl.h for ncclResult_t
#include "nccl.h"

// Prevent including the real core.h
#define NCCL_CORE_H_

// Debug logging stubs
#define NCCL_INIT 1
#define NCCL_NET  2

#define INFO(subsys, fmt, ...) do { \
    fprintf(stderr, "[INFO] " fmt "\n", ##__VA_ARGS__); \
} while(0)

#define WARN(fmt, ...) do { \
    fprintf(stderr, "[WARN] " fmt "\n", ##__VA_ARGS__); \
} while(0)

#define ERROR(fmt, ...) do { \
    fprintf(stderr, "[ERROR] " fmt "\n", ##__VA_ARGS__); \
} while(0)

// NCCLCHECK macro
#define NCCLCHECK(cmd) do { \
    ncclResult_t __res = (cmd); \
    if (__res != ncclSuccess) return __res; \
} while(0)

// CUDACHECK macro (uses HIP for AMD)
// Stub out HIP types when hip_runtime.h is not available
#if __has_include(<hip/hip_runtime.h>)
#include <hip/hip_runtime.h>
#define CUDACHECK(cmd) do { \
    hipError_t __err = (cmd); \
    if (__err != hipSuccess) { \
        ERROR("HIP error %d: %s", __err, hipGetErrorString(__err)); \
        return ncclUnhandledCudaError; \
    } \
} while(0)
#else
// Stub HIP types for environments without ROCm
typedef int hipError_t;
#define hipSuccess 0
#define hipDeviceGetByPCIBusId(dev, bus) (0)
#define hipDeviceGetPCIBusId(bus, len, dev) (0)
#define hipGetDeviceCount(count) (*(count) = 0, 0)
#define hipGetErrorString(err) "HIP not available"
#define CUDACHECK(cmd) do { (void)(cmd); } while(0)
#endif

// RCCL_PARAM macro for runtime parameters
#define RCCL_PARAM(name, env, default_val) \
    static int64_t rcclParam##name() { \
        static int64_t value = -1; \
        if (value == -1) { \
            const char* str = getenv("RCCL_" env); \
            value = str ? atoll(str) : (default_val); \
        } \
        return value; \
    }

// busIdToInt64 declaration (implemented in test_utils.cc)
ncclResult_t busIdToInt64(const char* busId, int64_t* id);

#endif // TEST_STUBS_H_
