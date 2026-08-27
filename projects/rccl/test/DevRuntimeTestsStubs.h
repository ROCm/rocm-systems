/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Controllable seams exposed by DevRuntimeTestsStubs.cc.
 *
 * Each hook defaults to the success behaviour the rest of the suite relies on.
 * A test that needs a HIP VMM call to fail installs its own via ScopedHook
 * (test/host/ScopedHook.h), which restores the previous one on scope exit.
 *************************************************************************/

#ifndef RCCL_TEST_DEVRUNTIMETESTSSTUBS_H_
#define RCCL_TEST_DEVRUNTIMETESTSSTUBS_H_

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <functional>

extern std::function<hipError_t(hipMemAllocationProp*, hipMemGenericAllocationHandle_t)>
    g_hipMemGetAllocationPropertiesFromHandle;

extern std::function<hipError_t(void*, hipMemGenericAllocationHandle_t, hipMemAllocationHandleType,
                                unsigned long long)>
    g_hipMemExportToShareableHandle;

extern std::function<hipError_t(void*, size_t, const hipMemAccessDesc*, size_t)> g_hipMemSetAccess;

// Restore every seam above to its default. Call from a fixture TearDown so a
// test cannot leak behaviour into the next one.
void ResetDevRuntimeFakes();

#endif  // RCCL_TEST_DEVRUNTIMETESTSSTUBS_H_
