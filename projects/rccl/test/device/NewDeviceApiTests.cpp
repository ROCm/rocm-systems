/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Device-side smoke tests for the Copy / ReduceCopy / ReduceSum APIs added
// in NCCL 2.29.7 (release notes: "Introduced Copy, ReduceCopy, and
// ReduceSum with various data types and ops").
//
// These device APIs are defined as inline templates in
// <nccl_device/reduce_copy.h> (hipified to reduce_copy.h in the install
// tree). A full per-(dtype, op) sweep belongs in a HIP-compiled fixture
// alongside test/device/TestOp128.cpp.
//
// As of this RCCL sync the header-only template body relies on internal
// types (ncclDevComm, ncclTeam, ncclSymPtr) which are not part of the
// public-facing surface and require building inside the rccl-UnitTests
// device target. This file establishes the test stub and a compile-time
// assertion that the public headers can at least be included, so we catch
// header-include regressions early.

#include <gtest/gtest.h>
#include <rccl/rccl.h>
#include <cstdio>
#include <cstdlib>

// Validate that the host-side include path for the new device API headers
// works against the installed headers.
#define NCCL_HOSTLIB_ONLY
#include <nccl_device/core_tmp.h>

#include "ErrCode.hpp"

namespace RcclUnitTesting
{

// Test 1: The device-API public headers must include cleanly in host code.
// This is a compile-time guard against header regressions.
TEST(NewDeviceApi, HeadersIncludeCleanly)
{
    SUCCEED() << "<nccl_device/core_tmp.h> with NCCL_HOSTLIB_ONLY included "
                 "successfully";
}

// Test 2: ncclMemAlloc / ncclMemFree (used as the entry point for buffers
// suitable for the new device APIs) round-trip cleanly.
TEST(NewDeviceApi, MemAllocFreeRoundTrip)
{
    int devCount = 0;
    if (hipGetDeviceCount(&devCount) != hipSuccess || devCount < 1) {
        GTEST_SKIP() << "Need at least 1 GPU device";
        return;
    }
    (void)hipSetDevice(0);

    const size_t bytes = 1024 * 1024;
    void* p = nullptr;
    ncclResult_t r = ncclMemAlloc(&p, bytes);
    ASSERT_EQ(r, ncclSuccess)
        << "ncclMemAlloc failed: " << ncclGetErrorString(r);
    ASSERT_NE(p, nullptr);
    EXPECT_EQ(ncclMemFree(p), ncclSuccess);
}

// Test 3 (placeholder): A full per-(dtype, op) sweep of Copy / ReduceCopy /
// ReduceSum belongs here once the public signatures of the device-side
// entry points are stabilized and the header-only templates compile from a
// host-test translation unit. Tracked as a follow-up.
TEST(NewDeviceApi, ReduceCopySweep_TODO)
{
    GTEST_SKIP() << "TODO: HIP kernel-fixture sweep of "
                    "Copy/ReduceCopy/ReduceSum across (dtype, op) pairs - "
                    "blocked on finalized public device-API entry points "
                    "exported by this RCCL sync.";
}

} // namespace RcclUnitTesting
