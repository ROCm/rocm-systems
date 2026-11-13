/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// CUPROBE() must not leave a HIP last error behind. RCCL asks "is this a cuMem allocation?" by
// calling hipMemRetainAllocationHandle on user buffers (p2p IPC registration, multicast and
// collective registration); for hipMalloc buffers the call fails by design. Unlike the CUDA driver
// API, HIP stores that failure as the thread's last error, and the application's next
// hipGetLastError() check (PyTorch's, e.g.) then fails on an unrelated call.

#include <hip/hip_runtime.h>

#include "gtest/gtest.h"
#include "rocmwrap.h"

namespace RcclUnitTesting
{

namespace
{
class HipProbe : public ::testing::Test
{
protected:
    void SetUp() override
    {
        int count = 0;
        if(hipGetDeviceCount(&count) != hipSuccess || count == 0)
        {
            (void)hipGetLastError();
            GTEST_SKIP() << "no HIP device";
        }
        ASSERT_EQ(hipSetDevice(0), hipSuccess);
        ASSERT_EQ(hipMalloc(&buf, kBytes), hipSuccess);
        ASSERT_EQ(hipGetLastError(), hipSuccess);
    }

    void TearDown() override
    {
        if(buf) EXPECT_EQ(hipFree(buf), hipSuccess);
        (void)hipGetLastError();
    }

    static constexpr size_t kBytes = 1 << 20;
    void* buf = nullptr;
};
} // namespace

// The HIP behaviour CUPROBE exists for. If HIP stops recording the failure, the
// other tests no longer guard anything.
TEST_F(HipProbe, RawProbeOfHipMallocBufferLeavesLastError)
{
    hipMemGenericAllocationHandle_t handle;
    ASSERT_NE(CUPFN(hipMemRetainAllocationHandle(&handle, buf)), hipSuccess);
    void* other = nullptr;
    ASSERT_EQ(hipMalloc(&other, kBytes), hipSuccess);
    hipError_t last = hipGetLastError();
    EXPECT_EQ(hipFree(other), hipSuccess);
    if(last == hipSuccess) GTEST_SKIP() << "this HIP does not keep the probe failure as the last error";
    EXPECT_EQ(last, hipErrorInvalidValue);
}

TEST_F(HipProbe, FailedProbeClearsLastError)
{
    hipMemGenericAllocationHandle_t handle;
    EXPECT_NE(CUPROBE(hipMemRetainAllocationHandle(&handle, buf)), hipSuccess);
    void* other = nullptr;
    ASSERT_EQ(hipMalloc(&other, kBytes), hipSuccess);
    EXPECT_EQ(hipGetLastError(), hipSuccess);
    EXPECT_EQ(hipFree(other), hipSuccess);
}

TEST_F(HipProbe, RepeatedProbesLeaveNoLastError)
{
    // Several probes in a row (one per peer during registration) must leave nothing behind either.
    for(int i = 0; i < 4; i++)
    {
        hipMemGenericAllocationHandle_t handle;
        EXPECT_NE(CUPROBE(hipMemRetainAllocationHandle(&handle, buf)), hipSuccess);
        hipDeviceptr_t base = nullptr;
        size_t size = 0;
        EXPECT_EQ(CUPROBE(hipMemGetAddressRange(&base, &size, reinterpret_cast<hipDeviceptr_t>(buf))), hipSuccess);
    }
    EXPECT_EQ(hipGetLastError(), hipSuccess);
}

TEST_F(HipProbe, SuccessfulProbeOfCuMemAllocation)
{
    hipMemAllocationProp prop = {};
    prop.type = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id = 0;
    size_t gran = 0;
    if(hipMemGetAllocationGranularity(&gran, &prop, hipMemAllocationGranularityMinimum) != hipSuccess || gran == 0)
    {
        (void)hipGetLastError();
        GTEST_SKIP() << "HIP virtual memory management not supported";
    }
    hipMemGenericAllocationHandle_t created;
    ASSERT_EQ(hipMemCreate(&created, gran, &prop, 0), hipSuccess);
    void* va = nullptr;
    ASSERT_EQ(hipMemAddressReserve(&va, gran, 0, nullptr, 0), hipSuccess);
    ASSERT_EQ(hipMemMap(va, gran, 0, created, 0), hipSuccess);

    hipMemGenericAllocationHandle_t retained;
    EXPECT_EQ(CUPROBE(hipMemRetainAllocationHandle(&retained, va)), hipSuccess);
    EXPECT_EQ(hipMemRelease(retained), hipSuccess);
    EXPECT_EQ(hipGetLastError(), hipSuccess);

    EXPECT_EQ(hipMemUnmap(va, gran), hipSuccess);
    EXPECT_EQ(hipMemAddressFree(va, gran), hipSuccess);
    EXPECT_EQ(hipMemRelease(created), hipSuccess);
}

} // namespace RcclUnitTesting
