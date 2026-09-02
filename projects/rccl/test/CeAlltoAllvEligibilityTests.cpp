/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "common/CeAlltoAllvTestHelpers.hpp"

#include <vector>

#include "gtest/gtest.h"

#include "ce_coll.h"
#include "collectives.h"
#include "group.h"
#include "nccl_common.h"
#include "sym_kernels.h"

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

TEST_F(CeAlltoAllvEligibilityTest, CeImplementedReturnsFalseForUnsupportedCollectives)
{
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

    EXPECT_TRUE(ncclCeAlltoAllvEligible(mockComm_.get(),
                                        ncclFloat32,
                                        ncclSymSendRegRecvReg,
                                        /*hasSysmemSegment=*/false,
                                        /*capturing=*/false));
    EXPECT_TRUE(ncclCeAlltoAllvEligible(mockComm_.get(),
                                        ncclFloat32,
                                        ncclSymSendNonregRecvReg,
                                        /*hasSysmemSegment=*/false,
                                        /*capturing=*/false));
}

TEST_F(CeAlltoAllvEligibilityTest, CeAlltoAllvEligible_RequiresZeroCtaPolicy)
{
    if (!isCeRuntimeDriverSupported())
        GTEST_SKIP() << "CE driver not in supported range";

    mockComm_.comm.config.CTAPolicy = NCCL_CTA_POLICY_DEFAULT;
    EXPECT_FALSE(ncclCeAlltoAllvEligible(mockComm_.get(),
                                         ncclFloat32,
                                         ncclSymSendRegRecvReg,
                                         /*hasSysmemSegment=*/false,
                                         /*capturing=*/false));

    mockComm_.comm.config.CTAPolicy = NCCL_CTA_POLICY_ZERO;
    EXPECT_TRUE(ncclCeAlltoAllvEligible(mockComm_.get(),
                                        ncclFloat32,
                                        ncclSymSendRegRecvReg,
                                        /*hasSysmemSegment=*/false,
                                        /*capturing=*/false));
}

TEST_F(CeAlltoAllvEligibilityTest, CeAlltoAllvEligible_RejectsSysmemSegmentOrCapture)
{
    if (!isCeRuntimeDriverSupported())
        GTEST_SKIP() << "CE driver not in supported range";

    EXPECT_FALSE(ncclCeAlltoAllvEligible(mockComm_.get(),
                                         ncclFloat32,
                                         ncclSymSendRegRecvReg,
                                         /*hasSysmemSegment=*/true,
                                         /*capturing=*/false));
    EXPECT_FALSE(ncclCeAlltoAllvEligible(mockComm_.get(),
                                         ncclFloat32,
                                         ncclSymSendRegRecvReg,
                                         /*hasSysmemSegment=*/false,
                                         /*capturing=*/true));
}

TEST_F(CeAlltoAllvEligibilityTest, CeAlltoAllvEligible_RejectsNestedGroup)
{
    if (!isCeRuntimeDriverSupported())
        GTEST_SKIP() << "CE driver not in supported range";

    const int savedGroupDepth = ncclGroupDepth;
    ncclGroupDepth = 1;
    EXPECT_FALSE(ncclCeAlltoAllvEligible(mockComm_.get(),
                                         ncclFloat32,
                                         ncclSymSendRegRecvReg,
                                         /*hasSysmemSegment=*/false,
                                         /*capturing=*/false));
    ncclGroupDepth = savedGroupDepth;
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
                                 ncclSymSendRegRecvReg, nullptr, nullptr));
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
                                 ncclSymSendRegRecvReg, nullptr, nullptr));
}

TEST_F(CeAlltoAllvEligibilityTest, CeAvailable_UnsupportedWindowRegistrationRejected)
{
    if (!isCeRuntimeDriverSupported())
        GTEST_SKIP() << "CE driver not in supported range";

    EXPECT_FALSE(ncclCeAvailable(mockComm_.get(),
                                 ncclFuncAlltoAllv,
                                 ncclDevSum,
                                 ncclFloat32,
                                 ncclSymSendNonregRecvNonreg, nullptr, nullptr));
    EXPECT_FALSE(ncclCeAvailable(mockComm_.get(),
                                 ncclFuncAlltoAllv,
                                 ncclDevSum,
                                 ncclFloat32,
                                 ncclSymSendRegRecvNonreg, nullptr, nullptr));
}

TEST_F(CeAlltoAllvEligibilityTest, LocalMetadataPackingMatchesGatheredLayout)
{
    constexpr int nRanks = 4;
    const size_t sendcounts[nRanks] = {16, 32, 0, 8};
    const size_t sdispls[nRanks]    = {0, 16, 48, 48};
    const size_t recvcounts[nRanks] = {8, 16, 32, 0};
    const size_t rdispls[nRanks]    = {0, 8, 24, 56};

    std::vector<size_t> local(4 * nRanks);
    ncclAlltoAllvPackLocalSizes(local.data(), nRanks, sendcounts, sdispls, recvcounts, rdispls);

    std::vector<size_t> gathered(4 * nRanks * nRanks, 0);
    const size_t blockBytes = static_cast<size_t>(4 * nRanks) * sizeof(size_t);
    std::memcpy(gathered.data() + ncclAlltoAllvMetaBlockOffset(1, nRanks),
                local.data(),
                blockBytes);

    size_t* rank1SendSizes = ncclAlltoAllvSendSizes(gathered.data(), 1, nRanks);
    size_t* rank1SendDispls = ncclAlltoAllvSendDispls(gathered.data(), 1, nRanks);
    size_t* rank1RecvSizes = ncclAlltoAllvRecvSizes(gathered.data(), 1, nRanks);
    size_t* rank1RecvDispls = ncclAlltoAllvRecvDispls(gathered.data(), 1, nRanks);

    EXPECT_EQ(rank1SendSizes[2], 0u);
    EXPECT_EQ(rank1SendDispls[1], 16u);
    EXPECT_EQ(rank1RecvSizes[3], 0u);
    EXPECT_EQ(rank1RecvDispls[2], 24u);
    EXPECT_EQ(ncclAlltoAllvTrafficBytes(gathered.data(), 1, nRanks), 56u);
}

TEST_F(CeAlltoAllvEligibilityTest, PeerMetadataIndexingMatchesCeCollLayout)
{
    constexpr int nRanks = 4;
    constexpr int myRank = 2;
    constexpr int dstRank = 3;

    std::vector<size_t> gathered(4 * nRanks * nRanks, 0);
    for (int r = 0; r < nRanks; ++r)
    {
        size_t* recvDispls = ncclAlltoAllvRecvDispls(gathered.data(), r, nRanks);
        recvDispls[myRank] = static_cast<size_t>(100 * (r + 1));
    }

    size_t* peerRecvDispls = ncclAlltoAllvRecvDispls(gathered.data(), dstRank, nRanks);
    EXPECT_EQ(peerRecvDispls[myRank], 400u);
}

TEST_F(CeAlltoAllvEligibilityTest, SizeMatrixVerdictIsRankIndependent)
{
    constexpr int nRanks = 4;
    std::vector<size_t> g(4 * nRanks * nRanks, 0);
    for (int src = 0; src < nRanks; ++src)
    {
        size_t* s = ncclAlltoAllvSendSizes(g.data(), src, nRanks);
        for (int dst = 0; dst < nRanks; ++dst)
        {
            s[dst] = static_cast<size_t>((src + 1) * (dst + 2));
        }
    }
    for (int dst = 0; dst < nRanks; ++dst)
    {
        size_t* r = ncclAlltoAllvRecvSizes(g.data(), dst, nRanks);
        for (int src = 0; src < nRanks; ++src)
        {
            r[src] = static_cast<size_t>((src + 1) * (dst + 2));
        }
    }
    EXPECT_EQ(ncclAlltoAllvValidateSizeMatrix(g.data(), nRanks), ncclSuccess);
    ncclAlltoAllvSendSizes(g.data(), 0, nRanks)[2] = 99;
    EXPECT_EQ(ncclAlltoAllvValidateSizeMatrix(g.data(), nRanks), ncclInvalidUsage);
}

TEST_F(CeAlltoAllvEligibilityTest, PeerSendSizeValidationRejectsMismatch)
{
    EXPECT_EQ(ncclAlltoAllvValidatePeerSendSize(128, 64, 0, 1), ncclInvalidUsage);
}

TEST_F(CeAlltoAllvEligibilityTest, PeerSendSizeValidationAcceptsMatch)
{
    EXPECT_EQ(ncclAlltoAllvValidatePeerSendSize(128, 128, 0, 1), ncclSuccess);
}

} // namespace RcclUnitTesting
