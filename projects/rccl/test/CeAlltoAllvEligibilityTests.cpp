/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "common/CeAlltoAllvTestHelpers.hpp"

#include "ce_coll.h"
#include "collectives.h"
#include "gtest/gtest.h"
#include "nccl_common.h"
#include "sym_kernels.h"

#include <vector>

namespace RcclUnitTesting
{

class CeAlltoAllvEligibilityTest : public ::testing::Test
{
protected:
    CeAlltoAllvMockComm mockComm_;
};

TEST_F(CeAlltoAllvEligibilityTest, FuncToStringReturnsAlltoAllv)
{
    EXPECT_STREQ(ncclFuncToString(ncclFuncAlltoAllv), "AlltoAllv");
}

TEST_F(CeAlltoAllvEligibilityTest, FuncEnumValue)
{
    EXPECT_EQ(ncclFuncAlltoAllv, 18);
    EXPECT_EQ(ncclNumFuncs, 19);
}

TEST_F(CeAlltoAllvEligibilityTest, CeImplementedReturnsFalseForUnsupportedCollectives)
{
    EXPECT_FALSE(ncclCeImplemented(ncclFuncAllReduce, ncclDevSum, ncclFloat32));
    EXPECT_FALSE(ncclCeImplemented(ncclFuncBroadcast, ncclDevSum, ncclFloat32));
}

TEST_F(CeAlltoAllvEligibilityTest, CeImplementedReturnsTrueForAlltoAllvOnSupportedDriver)
{
    if (!isCeRuntimeDriverSupported())
        GTEST_SKIP() << "CE driver not in supported range "
                        "(need ROCm >= 7.12 or 7.0.2.x backport [70051831, 70060000))";

    EXPECT_TRUE(ncclCeImplemented(ncclFuncAlltoAllv, ncclDevSum, ncclFloat32));
}

TEST_F(CeAlltoAllvEligibilityTest, CeAvailable_EligibleWithSymmetricSingleNode)
{
    if (!isCeRuntimeDriverSupported())
        GTEST_SKIP() << "CE driver not in supported range";

    EXPECT_TRUE(ncclCeAvailable(mockComm_.get(),
                                ncclFuncAlltoAllv,
                                ncclDevSum,
                                ncclFloat32,
                                ncclSymSendRegRecvReg));
    EXPECT_TRUE(ncclCeAvailable(mockComm_.get(),
                                ncclFuncAlltoAllv,
                                ncclDevSum,
                                ncclFloat32,
                                ncclSymSendNonregRecvReg));
}

TEST_F(CeAlltoAllvEligibilityTest, CeAvailable_MultiNodeRejected)
{
    if (!isCeRuntimeDriverSupported())
        GTEST_SKIP() << "CE driver not in supported range";

    mockComm_.comm.nNodes = 2;
    EXPECT_FALSE(ncclCeAvailable(mockComm_.get(),
                                 ncclFuncAlltoAllv,
                                 ncclDevSum,
                                 ncclFloat32,
                                 ncclSymSendRegRecvReg));
}

TEST_F(CeAlltoAllvEligibilityTest, CeAvailable_NoSymmetricSupportRejected)
{
    if (!isCeRuntimeDriverSupported())
        GTEST_SKIP() << "CE driver not in supported range";

    mockComm_.comm.symmetricSupport = false;
    EXPECT_FALSE(ncclCeAvailable(mockComm_.get(),
                                 ncclFuncAlltoAllv,
                                 ncclDevSum,
                                 ncclFloat32,
                                 ncclSymSendRegRecvReg));
}

TEST_F(CeAlltoAllvEligibilityTest, CeAvailable_UnsupportedWindowRegistrationRejected)
{
    if (!isCeRuntimeDriverSupported())
        GTEST_SKIP() << "CE driver not in supported range";

    EXPECT_FALSE(ncclCeAvailable(mockComm_.get(),
                                 ncclFuncAlltoAllv,
                                 ncclDevSum,
                                 ncclFloat32,
                                 ncclSymSendNonregRecvNonreg));
    EXPECT_FALSE(ncclCeAvailable(mockComm_.get(),
                                 ncclFuncAlltoAllv,
                                 ncclDevSum,
                                 ncclFloat32,
                                 ncclSymSendRegRecvNonreg));
}

TEST_F(CeAlltoAllvEligibilityTest, LocalMetadataPackingMatchesGatheredLayout)
{
    constexpr int nRanks = 4;
    const size_t sendcounts[nRanks] = {16, 32, 0, 8};
    const size_t sdispls[nRanks]    = {0, 16, 48, 48};
    const size_t recvcounts[nRanks] = {8, 16, 32, 0};
    const size_t rdispls[nRanks]    = {0, 8, 24, 56};

    std::vector<size_t> local(4 * nRanks);
    packLocalAlltoAllvSizes(local.data(), nRanks, sendcounts, sdispls, recvcounts, rdispls);

    std::vector<size_t> gathered(4 * nRanks * nRanks, 0);
    const size_t blockBytes = static_cast<size_t>(4 * nRanks) * sizeof(size_t);
    std::memcpy(gathered.data() + ceAlltoAllvMetaBlockOffset(1, nRanks),
                local.data(),
                blockBytes);

    size_t* rank1SendSizes = ceAlltoAllvSendSizes(gathered.data(), 1, nRanks);
    size_t* rank1SendDispls = ceAlltoAllvSendDispls(gathered.data(), 1, nRanks);
    size_t* rank1RecvSizes = ceAlltoAllvRecvSizes(gathered.data(), 1, nRanks);
    size_t* rank1RecvDispls = ceAlltoAllvRecvDispls(gathered.data(), 1, nRanks);

    EXPECT_EQ(rank1SendSizes[2], 0u);
    EXPECT_EQ(rank1SendDispls[1], 16u);
    EXPECT_EQ(rank1RecvSizes[3], 0u);
    EXPECT_EQ(rank1RecvDispls[2], 24u);
    EXPECT_EQ(ceAlltoAllvTrafficBytes(gathered.data(), 1, nRanks), 56u);
}

TEST_F(CeAlltoAllvEligibilityTest, PeerMetadataIndexingMatchesCeCollLayout)
{
    constexpr int nRanks = 4;
    constexpr int myRank = 2;
    constexpr int dstRank = 3;

    std::vector<size_t> gathered(4 * nRanks * nRanks, 0);
    for (int r = 0; r < nRanks; ++r)
    {
        size_t* recvDispls = ceAlltoAllvRecvDispls(gathered.data(), r, nRanks);
        recvDispls[myRank] = static_cast<size_t>(100 * (r + 1));
    }

    size_t* peerRecvDispls = ceAlltoAllvRecvDispls(gathered.data(), dstRank, nRanks);
    EXPECT_EQ(peerRecvDispls[myRank], 400u);
}

} // namespace RcclUnitTesting
