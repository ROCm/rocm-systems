/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// CUPROBE() must clear the HIP last error that an expected-failure probe leaves (see rocmwrap.h).

#include <hip/hip_runtime.h>

#include <string>

#include "common/ProcessIsolatedTestRunner.hpp"
#include "gtest/gtest.h"
#include "rocmwrap.h"

namespace RcclUnitTesting
{

namespace
{
constexpr size_t kBytes = 1 << 20;

// Returns the reason this host cannot run the test, or "" when it can.
// GTEST_SKIP() must be issued by the caller: it expands to a bare return and
// would otherwise only leave this helper, letting the test body run on.
std::string gpuSkipReason()
{
    int count = 0;
    if(hipGetDeviceCount(&count) != hipSuccess || count == 0)
    {
        (void)hipGetLastError();
        return "no HIP device";
    }
    return "";
}

std::string vmmSkipReason()
{
    int vmm = 0;
    if(hipDeviceGetAttribute(&vmm, hipDeviceAttributeVirtualMemoryManagementSupported, 0) != hipSuccess || !vmm)
    {
        (void)hipGetLastError();
        return "HIP virtual memory management not supported";
    }
    return "";
}

// Runs `check` on a hipMalloc buffer, the kind of user buffer RCCL probes. Callers are
// isolated, since later tests fork without exec and must not inherit an initialized HIP,
// and must skip on gpuSkipReason() first.
template <typename Check>
void withHipMallocBuffer(Check check)
{
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
    void* buf = nullptr;
    ASSERT_EQ(hipMalloc(&buf, kBytes), hipSuccess);
    ASSERT_EQ(hipGetLastError(), hipSuccess);
    check(buf);
    EXPECT_EQ(hipFree(buf), hipSuccess);
}
} // namespace

// The HIP behaviour CUPROBE exists for. If HIP stops recording the failure, the
// other tests no longer guard anything.
TEST(HipProbe, RawProbeOfHipMallocBufferLeavesLastError)
{
    RUN_ISOLATED_TEST("HipProbe.RawProbeOfHipMallocBufferLeavesLastError", []() {
        if(auto reason = gpuSkipReason(); !reason.empty()) GTEST_SKIP() << reason;
        hipError_t last = hipSuccess;
        withHipMallocBuffer([&](void* buf) {
            hipMemGenericAllocationHandle_t handle;
            ASSERT_NE(CUPFN(hipMemRetainAllocationHandle(&handle, buf)), hipSuccess);
            void* other = nullptr;
            ASSERT_EQ(hipMalloc(&other, kBytes), hipSuccess);
            last = hipGetLastError();
            EXPECT_EQ(hipFree(other), hipSuccess);
        });
        if(last == hipSuccess) GTEST_SKIP() << "this HIP does not keep the probe failure as the last error";
        EXPECT_EQ(last, hipErrorInvalidValue);
    });
}

TEST(HipProbe, FailedProbeClearsLastError)
{
    RUN_ISOLATED_TEST("HipProbe.FailedProbeClearsLastError", []() {
        if(auto reason = gpuSkipReason(); !reason.empty()) GTEST_SKIP() << reason;
        withHipMallocBuffer([](void* buf) {
            hipMemGenericAllocationHandle_t handle;
            EXPECT_NE(CUPROBE(hipMemRetainAllocationHandle(&handle, buf)), hipSuccess);
            void* other = nullptr;
            ASSERT_EQ(hipMalloc(&other, kBytes), hipSuccess);
            EXPECT_EQ(hipGetLastError(), hipSuccess);
            EXPECT_EQ(hipFree(other), hipSuccess);
        });
    });
}

TEST(HipProbe, RepeatedProbesLeaveNoLastError)
{
    RUN_ISOLATED_TEST("HipProbe.RepeatedProbesLeaveNoLastError", []() {
        if(auto reason = gpuSkipReason(); !reason.empty()) GTEST_SKIP() << reason;
        withHipMallocBuffer([](void* buf) {
            // Several probes in a row (one per peer during registration) must leave nothing behind either.
            for(int i = 0; i < 4; i++)
            {
                hipMemGenericAllocationHandle_t handle;
                EXPECT_NE(CUPROBE(hipMemRetainAllocationHandle(&handle, buf)), hipSuccess);
                hipDeviceptr_t base = nullptr;
                size_t size = 0;
                EXPECT_EQ(CUPROBE(hipMemGetAddressRange(&base, &size, reinterpret_cast<hipDeviceptr_t>(buf))),
                          hipSuccess);
            }
            EXPECT_EQ(hipGetLastError(), hipSuccess);
        });
    });
}

TEST(HipProbe, SuccessfulProbeOfCuMemAllocation)
{
    RUN_ISOLATED_TEST("HipProbe.SuccessfulProbeOfCuMemAllocation", []() {
        if(auto reason = gpuSkipReason(); !reason.empty()) GTEST_SKIP() << reason;
        ASSERT_EQ(hipSetDevice(0), hipSuccess);
        if(auto reason = vmmSkipReason(); !reason.empty()) GTEST_SKIP() << reason;
        hipMemAllocationProp prop = {};
        prop.type = hipMemAllocationTypePinned;
        prop.location.type = hipMemLocationTypeDevice;
        prop.location.id = 0;
        size_t gran = 0;
        ASSERT_EQ(hipMemGetAllocationGranularity(&gran, &prop, hipMemAllocationGranularityMinimum), hipSuccess);
        ASSERT_NE(gran, 0u);
        hipMemGenericAllocationHandle_t created;
        ASSERT_EQ(hipMemCreate(&created, gran, &prop, 0), hipSuccess);
        void* va = nullptr;
        ASSERT_EQ(hipMemAddressReserve(&va, gran, 0, nullptr, 0), hipSuccess);
        ASSERT_EQ(hipMemMap(va, gran, 0, created, 0), hipSuccess);

        hipMemGenericAllocationHandle_t retained = nullptr;
        const hipError_t probed = CUPROBE(hipMemRetainAllocationHandle(&retained, va));
        EXPECT_EQ(probed, hipSuccess);
        if(probed == hipSuccess) EXPECT_EQ(hipMemRelease(retained), hipSuccess);
        EXPECT_EQ(hipGetLastError(), hipSuccess);

        EXPECT_EQ(hipMemUnmap(va, gran), hipSuccess);
        EXPECT_EQ(hipMemAddressFree(va, gran), hipSuccess);
        EXPECT_EQ(hipMemRelease(created), hipSuccess);
    });
}

} // namespace RcclUnitTesting
