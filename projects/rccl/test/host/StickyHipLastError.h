/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_TEST_HOST_STICKY_HIP_LAST_ERROR_H_
#define RCCL_TEST_HOST_STICKY_HIP_LAST_ERROR_H_

#include <utility>

#include "ScopedHook.h"
#include "fakes/hip_fakes.h"

// HIP keeps a failed call's error as the thread's last error until hipGetLastError reads it.
// A hook reports its failure through Fail(); pending is what the application would read next.
struct StickyHipLastError {
    hipError_t pending = hipSuccess;
    HookOf<decltype(g_hipGetLastError)> read{g_hipGetLastError,
                                             [this] { return std::exchange(pending, hipSuccess); }};

    hipError_t Fail(hipError_t error) { return pending = error; }
};

#endif  // RCCL_TEST_HOST_STICKY_HIP_LAST_ERROR_H_
