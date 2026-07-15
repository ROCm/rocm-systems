/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>
#include <rccl/rccl.h>

#include <cstring>

#include "common/ProcessIsolatedTestRunner.hpp"

namespace RcclUnitTesting
{
namespace {

// Mirror hipified p2p.h / allocator.cc: fabric uses int sentinel 0x8, attr probe uses 128.
constexpr int kFabricHandleTypeInt = 0x8;
constexpr int kFabricAttrInt       = 128;

int resolveMemAllocHandleType(
    int fabricHandleType, int requested, hipError_t attrErr, int fabricSupported)
{
    if(requested == fabricHandleType)
    {
        if(attrErr != hipSuccess || !fabricSupported)
        {
            return static_cast<int>(hipMemHandleTypePosixFileDescriptor);
        }
    }
    return requested;
}

} // namespace

#if ROCM_VERSION >= 70000
// Mirrors ncclMemAlloc_impl handle resolution: gfx1250 keeps FABRIC when supported;
// all other arches fall back to POSIX. Uses only public HIP/RCCL APIs so this runs
// in Release via rccl-UnitTestsFixtures (no librccl internal symbols).
TEST(Alloc, NcclMemAlloc_HandleTypeByArch)
{
    RUN_ISOLATED_TEST(
        "NcclMemAlloc_HandleTypeByArch",
        []()
        {
            ASSERT_EQ(setenv("NCCL_CUMEM_ENABLE", "1", 1), 0);
            ASSERT_EQ(hipSetDevice(0), hipSuccess);

            hipDeviceProp_t prop{};
            ASSERT_EQ(hipGetDeviceProperties(&prop, 0), hipSuccess);
            const bool gfx1250 = strncmp(prop.gcnArchName, "gfx1250", 7) == 0;

            int              fabricSupported = 0;
            const hipError_t attrErr         = hipDeviceGetAttribute(
                &fabricSupported,
                static_cast<hipDeviceAttribute_t>(kFabricAttrInt),
                0);

            const int fabricHandleType    = kFabricHandleTypeInt;
            const int requestedHandleType = resolveMemAllocHandleType(
                fabricHandleType, fabricHandleType, attrErr, fabricSupported);

            if(gfx1250)
            {
                EXPECT_EQ(attrErr, hipSuccess);
                EXPECT_NE(fabricSupported, 0);
                EXPECT_EQ(requestedHandleType, fabricHandleType);
            }
            else
            {
                EXPECT_EQ(requestedHandleType,
                          static_cast<int>(hipMemHandleTypePosixFileDescriptor))
                    << "Non-gfx1250 arch " << prop.gcnArchName
                    << " must use POSIX handles";
            }

            void*        ptr    = nullptr;
            const size_t kBytes = 4096;
            ASSERT_EQ(ncclMemAlloc(&ptr, kBytes), ncclSuccess);
            ASSERT_NE(ptr, nullptr);
            ASSERT_EQ(ncclMemFree(ptr), ncclSuccess);
        }
    );
}
#endif // ROCM_VERSION >= 70000
} // namespace RcclUnitTesting
