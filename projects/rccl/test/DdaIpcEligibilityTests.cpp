/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "common/DdaIpcTestHelpers.hpp"

#include "dda_all_gather_ipc.h"
#include "dda_alltoall_ipc.h"
#include "dda_reduce_scatter_ipc.h"
#include "gtest/gtest.h"
#include "ipc_init_detail.h"

namespace RcclUnitTesting
{

class DdaIpcEligibilityTest : public ::testing::Test
{
protected:
    DdaIpcMockComm mockComm_;
    void*          sendbuff_{reinterpret_cast<void*>(0x10)};
    void*          recvbuff_{reinterpret_cast<void*>(0x20)};
};

TEST_F(DdaIpcEligibilityTest, AllGather_NullComm)
{
    EXPECT_FALSE(ncclAllGatherDdaIpcEligible(
        nullptr, sendbuff_, recvbuff_, 4, ncclFloat32));
}

TEST_F(DdaIpcEligibilityTest, AllGather_MissingBootstrap)
{
    mockComm_.comm.bootstrap = nullptr;
    EXPECT_FALSE(ncclAllGatherDdaIpcEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32));
}

TEST_F(DdaIpcEligibilityTest, AllGather_MissingIpcResources)
{
    mockComm_.setIpcResourcesPresent(false);
    EXPECT_FALSE(ncclAllGatherDdaIpcEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32));
}

TEST_F(DdaIpcEligibilityTest, AllGather_ZeroCount)
{
    EXPECT_FALSE(ncclAllGatherDdaIpcEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 0, ncclFloat32));
}

TEST_F(DdaIpcEligibilityTest, AllGather_MultiNode)
{
    mockComm_.comm.nNodes = 2;
    EXPECT_FALSE(ncclAllGatherDdaIpcEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32));
}

TEST_F(DdaIpcEligibilityTest, AllGather_WrongRankCount)
{
    mockComm_.comm.nRanks = 4;
    EXPECT_FALSE(ncclAllGatherDdaIpcEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32));
}

TEST_F(DdaIpcEligibilityTest, AllGather_UnsupportedDatatype)
{
    EXPECT_FALSE(ncclAllGatherDdaIpcEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclInt32));
}

TEST_F(DdaIpcEligibilityTest, AllGather_ScratchTooSmall)
{
    mockComm_.comm.ddaIpcScratchBytes = 8;
    EXPECT_FALSE(ncclAllGatherDdaIpcEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32));
}

TEST_F(DdaIpcEligibilityTest, AllGather_UnalignedCount)
{
    EXPECT_FALSE(ncclAllGatherDdaIpcEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 3, ncclFloat32));
}

TEST_F(DdaIpcEligibilityTest, AllGather_EligibleFloat32)
{
    EXPECT_TRUE(ncclAllGatherDdaIpcEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32));
}

TEST_F(DdaIpcEligibilityTest, AllGather_EligibleFloat16)
{
    EXPECT_TRUE(ncclAllGatherDdaIpcEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 8, ncclFloat16));
}

TEST_F(DdaIpcEligibilityTest, AllGather_InvalidDatatypeDispatch)
{
    EXPECT_EQ(ncclAllGatherDdaIpc(
                  sendbuff_, recvbuff_, 4, ncclInt32, mockComm_.get(), nullptr),
              ncclInvalidArgument);
}

TEST_F(DdaIpcEligibilityTest, AllToAll_EligibleFloat32)
{
    EXPECT_TRUE(ncclAllToAllDdaIpcEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32));
}

TEST_F(DdaIpcEligibilityTest, AllToAll_ScratchTooSmallForTotal)
{
    mockComm_.comm.ddaIpcScratchBytes = 64;
    EXPECT_FALSE(ncclAllToAllDdaIpcEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32));
}

TEST_F(DdaIpcEligibilityTest, AllToAll_UnalignedCount)
{
    EXPECT_FALSE(ncclAllToAllDdaIpcEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 3, ncclFloat32));
}

TEST_F(DdaIpcEligibilityTest, AllToAll_InvalidDatatypeDispatch)
{
    EXPECT_EQ(ncclAllToAllDdaIpc(
                  sendbuff_, recvbuff_, 4, ncclInt32, mockComm_.get(), nullptr),
              ncclInvalidArgument);
}

TEST_F(DdaIpcEligibilityTest, ReduceScatter_EligibleFloat32)
{
    EXPECT_TRUE(ncclReduceScatterDdaIpcEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32, ncclSum));
}

TEST_F(DdaIpcEligibilityTest, ReduceScatter_UnsupportedOp)
{
    EXPECT_FALSE(ncclReduceScatterDdaIpcEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32, ncclMax));
}

TEST_F(DdaIpcEligibilityTest, ReduceScatter_NonAlignedRecvcount)
{
    EXPECT_FALSE(ncclReduceScatterDdaIpcEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 3, ncclFloat32, ncclSum));
}

TEST_F(DdaIpcEligibilityTest, ReduceScatter_PerRankUnaligned)
{
    EXPECT_FALSE(ncclReduceScatterDdaIpcEligible(
        mockComm_.get(), sendbuff_, recvbuff_, 2, ncclFloat32, ncclSum));
}

TEST_F(DdaIpcEligibilityTest, ReduceScatter_InvalidDatatypeDispatch)
{
    EXPECT_EQ(ncclReduceScatterDdaIpc(sendbuff_,
                                     recvbuff_,
                                     4,
                                     ncclInt32,
                                     ncclSum,
                                     mockComm_.get(),
                                     nullptr),
              ncclInvalidArgument);
}

} // namespace RcclUnitTesting
