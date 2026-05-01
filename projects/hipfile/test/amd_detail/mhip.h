/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "context.h"
#include "hip.h"

#include <gmock/gmock.h>

/* mhipxx (mock hip++)
 *
 * Mock implementations for Hip. Enables unit tests to mock HIP APIs.
 */
namespace hipFile {

struct MHip : Hip {
    ContextOverride<Hip> co;
    MHip() : co{this}
    {
    }
    MOCK_METHOD(hipPointerAttribute_t, hipPointerGetAttributes, (const void *ptr), (const override));
    MOCK_METHOD(void, hipMemcpy, (void *dst, const void *src, size_t sizeBytes, hipMemcpyKind kind),
                (const, override));
    MOCK_METHOD(void, hipStreamSynchronize, (hipStream_t stream), (const, override));
    MOCK_METHOD(void *, hipHostMalloc, (size_t size, unsigned int flags), (const, override));
    MOCK_METHOD(void, hipHostFree, (void *ptr), (const, override));
    MOCK_METHOD(void *, hipHostGetDevicePointer, (void *hstPtr, unsigned int flags), (const, override));
    MOCK_METHOD(int, hipRuntimeGetVersion, (), (const, override));
    MOCK_METHOD(void *, hipGetProcAddress,
                (const char *symbol, int hipVersion, uint64_t flags,
                 hipDriverProcAddressQueryResult *symbolStatus),
                (const, override));
    MOCK_METHOD(uint64_t, hipAmdFileRead,
                (hipAmdFileHandle_t handle, void *devicePtr, uint64_t size, int64_t file_offset),
                (const, override));
    MOCK_METHOD(uint64_t, hipAmdFileWrite,
                (hipAmdFileHandle_t handle, void *devicePtr, uint64_t size, int64_t file_offset),
                (const, override));
    MOCK_METHOD(HipMemAddressRange, hipMemGetAddressRange, (hipDeviceptr_t dptr), (const, override));
    MOCK_METHOD(void, hipLaunchHostFunc, (hipStream_t stream, hipHostFn_t fn, void *user_data),
                (const, override));
    MOCK_METHOD(void, hipLaunchKernel,
                (const void *function_address, dim3 numBlocks, dim3 dimBlocks, void **args,
                 size_t sharedMemBytes, hipStream_t stream),
                (const, override));
    MOCK_METHOD(int, hipDeviceGetAttribute, (hipDeviceAttribute_t attr, int device_id), (const, override));
    MOCK_METHOD(hipDevice_t, hipStreamGetDevice, (hipStream_t stream), (const, override));
    MOCK_METHOD(void, hipInit, (), (const, override));
    MOCK_METHOD(int, hipGetDevice, (), (const, override));
    MOCK_METHOD(int, hipGetDeviceCount, (), (const, override));
};
}
