/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "hipfile-warnings.h"
#include "hipfile.h"
#include "test-common.h"
#include "test-shared-fixtures.h"

#include <cstdint>
#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>
#include <memory>

HIPFILE_WARN_NO_GLOBAL_CTOR_OFF

class HipBuffer : public DriverInit {};

TEST_F(HipBuffer, hipFileBufRegisterWithDeviceMemoryIsSuccessful)
{
    void *device_ptr;
    ASSERT_EQ(hipMalloc(&device_ptr, 4096), hipSuccess);
    ASSERT_EQ(hipFileBufRegister(device_ptr, 4096, 0), HIPFILE_SUCCESS);
}

TEST_F(HipBuffer, hipFileBufRegisterWithOffsetDeviceMemoryIsSuccessful)
{
    void *device_ptr;
    ASSERT_EQ(hipMalloc(&device_ptr, 8192), hipSuccess);
    ASSERT_EQ(hipFileBufRegister(reinterpret_cast<void *>((reinterpret_cast<uintptr_t>(device_ptr) + 4096)),
                                 4096, 0),
              HIPFILE_SUCCESS);
}

TEST_F(HipBuffer, hipFileBufRegisterWithOversizeLengthErrors)
{
    void *device_ptr;
    ASSERT_EQ(hipMalloc(&device_ptr, 4096), hipSuccess);
    ASSERT_EQ(hipFileBufRegister(device_ptr, 8192, 0), HipFileOpError(hipFileHipPointerRangeError));
}

TEST_F(HipBuffer, hipFileBufRegisterWithOffsetPointerAndOversizeLengthErrors)
{
    void *device_ptr;
    ASSERT_EQ(hipMalloc(&device_ptr, 8192), hipSuccess);
    ASSERT_EQ(hipFileBufRegister(reinterpret_cast<void *>((reinterpret_cast<uintptr_t>(device_ptr) + 4096)),
                                 8192, 0),
              HipFileOpError(hipFileHipPointerRangeError));
}

TEST_F(HipBuffer, hipFileBufRegisterWithOverflowLengthErrors)
{
    void *device_ptr;
    ASSERT_EQ(hipMalloc(&device_ptr, 4096), hipSuccess);
    ASSERT_EQ(hipFileBufRegister(device_ptr, UINT64_MAX, 0), HipFileOpError(hipFileHipPointerRangeError));
}

TEST_F(HipBuffer, hipFileBufRegisterWithNullptrErrors)
{
    ASSERT_EQ(hipFileBufRegister(nullptr, 8192, 0), HipFileOpError(hipFileInvalidValue));
}

HIPFILE_WARN_NO_GLOBAL_CTOR_ON
