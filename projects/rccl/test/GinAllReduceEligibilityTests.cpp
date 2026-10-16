/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "common/GinAllReduceTestHelpers.hpp"

#include "algorithms/gin/gin_all_reduce.h"
#include "gtest/gtest.h"

namespace RcclUnitTesting
{

#if defined(ENABLE_ROCSHMEM_GIN)

class GinAllReduceEligibilityTest : public ::testing::Test
{
protected:
    GinAllReduceMockComm mockComm_;
    void*                sendbuff_{reinterpret_cast<void*>(0x1000)};
    void*                recvbuff_{reinterpret_cast<void*>(0x2000)};
    // 4 MiB of float32: below the default 256 MiB GIN floor, above zero.
    static constexpr size_t kSmallCount = (4ull * 1024 * 1024) / 4;
};

TEST_F(GinAllReduceEligibilityTest, NullComm)
{
    EXPECT_FALSE(ncclAllReduceGinSdmaEligible(nullptr, sendbuff_, recvbuff_, kSmallCount, ncclFloat32, ncclSum));
    EXPECT_FALSE(ncclAllReduceGinSdmaYieldToDda(nullptr, sendbuff_, recvbuff_, kSmallCount, ncclFloat32, ncclSum));
}

TEST_F(GinAllReduceEligibilityTest, NullSendbuff)
{
    EXPECT_FALSE(ncclAllReduceGinSdmaEligible(mockComm_.get(), nullptr, recvbuff_, kSmallCount, ncclFloat32, ncclSum));
}

TEST_F(GinAllReduceEligibilityTest, NullRecvbuff)
{
    EXPECT_FALSE(ncclAllReduceGinSdmaEligible(mockComm_.get(), sendbuff_, nullptr, kSmallCount, ncclFloat32, ncclSum));
}

TEST_F(GinAllReduceEligibilityTest, ZeroCount)
{
    EXPECT_FALSE(ncclAllReduceGinSdmaEligible(mockComm_.get(), sendbuff_, recvbuff_, 0, ncclFloat32, ncclSum));
}

TEST_F(GinAllReduceEligibilityTest, WrongArchRejected)
{
    mockComm_.setArch("gfx942");
    EXPECT_FALSE(ncclAllReduceGinSdmaEligible(mockComm_.get(), sendbuff_, recvbuff_, kSmallCount, ncclFloat32, ncclSum));
    EXPECT_FALSE(ncclAllReduceGinSdmaYieldToDda(mockComm_.get(), sendbuff_, recvbuff_, kSmallCount, ncclFloat32, ncclSum));
}

TEST_F(GinAllReduceEligibilityTest, WrongOpRejected)
{
    EXPECT_FALSE(ncclAllReduceGinSdmaEligible(mockComm_.get(), sendbuff_, recvbuff_, kSmallCount, ncclFloat32, ncclProd));
    EXPECT_FALSE(ncclAllReduceGinSdmaEligible(mockComm_.get(), sendbuff_, recvbuff_, kSmallCount, ncclFloat32, ncclMax));
}

TEST_F(GinAllReduceEligibilityTest, UnsupportedDatatypeRejected)
{
    EXPECT_FALSE(ncclAllReduceGinSdmaEligible(mockComm_.get(), sendbuff_, recvbuff_, kSmallCount, ncclInt32, ncclSum));
    EXPECT_FALSE(ncclAllReduceGinSdmaEligible(mockComm_.get(), sendbuff_, recvbuff_, kSmallCount, ncclFloat64, ncclSum));
}

TEST_F(GinAllReduceEligibilityTest, NoSymmetricSupportRejected)
{
    mockComm_.comm.symmetricSupport = false;
    EXPECT_FALSE(ncclAllReduceGinSdmaEligible(mockComm_.get(), sendbuff_, recvbuff_, kSmallCount, ncclFloat32, ncclSum));
    EXPECT_FALSE(ncclAllReduceGinSdmaYieldToDda(mockComm_.get(), sendbuff_, recvbuff_, kSmallCount, ncclFloat32, ncclSum));
}

#endif // ENABLE_ROCSHMEM_GIN

#if !defined(ENABLE_ROCSHMEM_GIN)
TEST(GinAllReduceEligibilityTest, SkippedWithoutGinBuild)
{
    GTEST_SKIP() << "GIN AllReduce eligibility tests require ENABLE_ROCSHMEM_GIN";
}
#endif

} // namespace RcclUnitTesting
