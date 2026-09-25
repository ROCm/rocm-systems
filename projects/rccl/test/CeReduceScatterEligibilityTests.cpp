/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "common/CeReduceScatterTestHelpers.hpp"
#include "common/ProcessIsolatedTestRunner.hpp"

#include "ce_coll.h"
#include "collectives.h"
#include "gtest/gtest.h"
#include "nccl.h"
#include "rccl_common.h"

#include <unordered_map>
#include <vector>

namespace RcclUnitTesting
{

class CeReduceScatterEligibilityTest : public ::testing::Test
{
protected:
    CeReduceScatterMockComm mockComm_;
};

TEST_F(CeReduceScatterEligibilityTest, FuncToStringReturnsReduceScatter)
{
    EXPECT_STREQ(ncclFuncToString(ncclFuncReduceScatter), "ReduceScatter");
}

TEST_F(CeReduceScatterEligibilityTest, CeImplementedReturnsFalseForUnsupportedCollectives)
{
    if(!isCeRuntimeDriverSupported())
        GTEST_SKIP() << "CE driver not in supported range";

    EXPECT_FALSE(ncclCeImplemented(ncclFuncBroadcast, ncclDevSum, ncclFloat32));
    EXPECT_FALSE(ncclCeImplemented(ncclFuncReduce, ncclDevSum, ncclFloat32));
}

TEST_F(CeReduceScatterEligibilityTest, CeImplementedReturnsTrueForReduceScatterOnSupportedDriver)
{
    if(!isCeRuntimeDriverSupported())
        GTEST_SKIP() << "CE driver not in supported range "
                        "(need ROCm >= 7.12 or 7.0.2.x backport [70051831, 70060000))";

    EXPECT_TRUE(ncclCeImplemented(ncclFuncReduceScatter, ncclDevSum, ncclFloat32));
}

TEST_F(CeReduceScatterEligibilityTest, CeAvailable_EligibleWithSymmetricSingleNode)
{
    if(!isCeRuntimeDriverSupported())
        GTEST_SKIP() << "CE driver not in supported range";

    EXPECT_TRUE(ncclCeAvailable(mockComm_.get(),
                                ncclFuncReduceScatter,
                                ncclDevSum,
                                ncclFloat32,
                                ncclSymSendRegRecvReg, nullptr, nullptr));
    EXPECT_TRUE(ncclCeAvailable(mockComm_.get(),
                                ncclFuncReduceScatter,
                                ncclDevSum,
                                ncclFloat32,
                                ncclSymSendNonregRecvReg, nullptr, nullptr));
}

TEST_F(CeReduceScatterEligibilityTest, CeAvailable_MultiNodeRejected)
{
    if(!isCeRuntimeDriverSupported())
        GTEST_SKIP() << "CE driver not in supported range";

    mockComm_.comm.nNodes = 2;
    EXPECT_FALSE(ncclCeAvailable(mockComm_.get(),
                                 ncclFuncReduceScatter,
                                 ncclDevSum,
                                 ncclFloat32,
                                 ncclSymSendRegRecvReg, nullptr, nullptr));
}

TEST_F(CeReduceScatterEligibilityTest, CeAvailable_NoSymmetricSupportRejected)
{
    if(!isCeRuntimeDriverSupported())
        GTEST_SKIP() << "CE driver not in supported range";

    mockComm_.comm.symmetricSupport = false;
    EXPECT_FALSE(ncclCeAvailable(mockComm_.get(),
                                 ncclFuncReduceScatter,
                                 ncclDevSum,
                                 ncclFloat32,
                                 ncclSymSendRegRecvReg, nullptr, nullptr));
}

TEST_F(CeReduceScatterEligibilityTest, CeAvailable_UnsupportedWindowRegistrationRejected)
{
    if(!isCeRuntimeDriverSupported())
        GTEST_SKIP() << "CE driver not in supported range";

    EXPECT_FALSE(ncclCeAvailable(mockComm_.get(),
                                 ncclFuncReduceScatter,
                                 ncclDevSum,
                                 ncclFloat32,
                                 ncclSymSendNonregRecvNonreg, nullptr, nullptr));
    EXPECT_FALSE(ncclCeAvailable(mockComm_.get(),
                                 ncclFuncReduceScatter,
                                 ncclDevSum,
                                 ncclFloat32,
                                 ncclSymSendRegRecvNonreg, nullptr, nullptr));
}

TEST_F(CeReduceScatterEligibilityTest, ChunkLayout_SmallMessageSingleChunk)
{
    constexpr int    nRanks    = 4;
    constexpr size_t recvcount = 1024;

    // ncclCeReduceScatter() only ever sees recvcounts the eligibility gate accepted:
    // nonzero, and a total message (recvcount * nRanks) no larger than the staging buffer.
    // Unlike AllReduce, recvcount is already the per-rank shard, so it does not have to
    // divide nRanks.
    ASSERT_GT(recvcount, 0u);
    ASSERT_LE(recvcount * sizeof(float) * static_cast<size_t>(nRanks),
              kCeRsMaxMsgBytesDefault);

    const size_t shardElems = recvcount;
    const size_t shardBytes = shardElems * sizeof(float);
    const size_t slotChunkBytes =
        ncclCeAllReduceSlotChunkBytes(ncclCeAllReduceMaxChunkBytes(nRanks));

    // A shard this small fits one slot, so ncclCeReduceScatter() sends it as a single
    // chunk and never enters the pipelined path.
    EXPECT_EQ(shardElems, 1024u);
    EXPECT_LE(shardBytes, slotChunkBytes);
}

// The host scatter addresses staging slots in bytes (rank * slotChunkBytes) while
// the reduce kernel addresses them in elements (rank * slotChunkElems). If those
// two strides disagree by even one byte, every rank but rank 0 reduces shifted
// data. CE ReduceScatter reuses the AllReduce staging layout, so the same
// alignment constraint applies. NCCL_CE_AR_STAGING_BYTES / nRanks only divides
// evenly for power-of-2 rank counts, so those were the only ones that used to work.
TEST_F(CeReduceScatterEligibilityTest, ChunkLayout_SlotStridesAgreeForAnyRankCount)
{
    const std::vector<int>    rankCounts   = {2, 3, 4, 5, 6, 7, 8, 12, 16, 24};
    const std::vector<size_t> elementSizes = {1, 2, 4, 8};

    for(int nRanks : rankCounts)
    {
        const size_t slotChunkBytes =
            ncclCeAllReduceSlotChunkBytes(ncclCeAllReduceMaxChunkBytes(nRanks));
        SCOPED_TRACE("nRanks=" + std::to_string(nRanks));

        // Rank boundaries stay aligned for the kernel's 16B vector loads, and the
        // slots stay inside the buffer ncclCeEnsureAllReduceStaging() sized from
        // the raw capacity.
        EXPECT_EQ(slotChunkBytes % 16, 0u);
        EXPECT_LE(slotChunkBytes, ncclCeAllReduceMaxChunkBytes(nRanks));

        for(size_t eltSize : elementSizes)
        {
            // A slot holds a whole number of elements, so the byte view and the
            // element view describe the same stride.
            EXPECT_EQ((slotChunkBytes / eltSize) * eltSize, slotChunkBytes)
                << "eltSize=" << eltSize;
        }
    }
}

TEST_F(CeReduceScatterEligibilityTest, ChunkLayout_LargeMessagePipelined)
{
    // A shard only spills past one slot when NCCL_CE_AR_STAGING_BYTES / nRanks is
    // not 16B-aligned, i.e. for a non-power-of-2 rank count at the per-rank cap.
    // For ReduceScatter the shard is recvcount itself.
    constexpr int nRanks     = 6;
    const size_t  shardElems = ncclCeAllReduceMaxChunkBytes(nRanks) / sizeof(float);
    const size_t  recvcount  = shardElems;

    // The gate checks the full send buffer (recvcount * nRanks), so this layout
    // is reachable: the truncated per-rank capacity times nRanks still fits.
    ASSERT_LE(recvcount * sizeof(float) * static_cast<size_t>(nRanks),
              kCeRsMaxMsgBytesDefault);

    const size_t shardBytes = shardElems * sizeof(float);
    const size_t slotChunkBytes =
        ncclCeAllReduceSlotChunkBytes(ncclCeAllReduceMaxChunkBytes(nRanks));
    ASSERT_GT(shardBytes, slotChunkBytes);

    // Same bookkeeping ncclCeReduceScatter() does once it has picked a chunk size.
    const size_t chunkBytes      = ncclCeAllReduceChooseChunkBytes(shardBytes, slotChunkBytes);
    const size_t baseChunkElems  = chunkBytes / sizeof(float);
    const size_t tailChunkElems  = shardElems % baseChunkElems;
    const size_t chunksPerShard  = shardElems / baseChunkElems + (tailChunkElems != 0 ? 1 : 0);
    const size_t lastChunkElems  = tailChunkElems != 0 ? tailChunkElems : baseChunkElems;

    ASSERT_GT(chunksPerShard, 1u);
    EXPECT_EQ(chunkBytes % 16, 0u);
    EXPECT_EQ(baseChunkElems * sizeof(float), chunkBytes);
    EXPECT_LE(chunkBytes, slotChunkBytes);

    // Chunks must cover the shard exactly: the host reads chunk ch at
    // ch * chunkBytes, so a chunk size that is not a whole number of elements
    // walks the last chunk past the end of the shard.
    EXPECT_EQ((chunksPerShard - 1) * baseChunkElems + lastChunkElems, shardElems);
    EXPECT_EQ((chunksPerShard - 1) * chunkBytes + lastChunkElems * sizeof(float), shardBytes);
}

TEST_F(CeReduceScatterEligibilityTest, MaxStagingBytesPerRank)
{
    // The whole message has to fit the per-rank staging capacity
    // ncclCeEnsureAllReduceStaging uses. ReduceScatter shares that buffer.
    for(int nRanks : {2, 3, 4, 5, 6, 7, 8, 12, 16, 24})
    {
        SCOPED_TRACE("nRanks=" + std::to_string(nRanks));
        EXPECT_LE(ncclCeAllReduceMaxChunkBytes(nRanks) * static_cast<size_t>(nRanks),
                  kCeRsMaxMsgBytesDefault);
    }
}

TEST(RcclCeReduceScatterEligibility, RcclUseCeReduceScatter_Isolated)
{
    struct UseCeRsCase
    {
        std::string                                  name;
        int                                          nRanks;
        int                                          nNodes;
        bool                                         symmetricSupport;
        int                                          ctaPolicy;
        size_t                                       recvcount;
        ncclRedOp_t                                  op;
        ncclDataType_t                               datatype;
        bool                                         expected;
        std::unordered_map<std::string, std::string> extraEnv;
    };

    const std::unordered_map<std::string, std::string> baseEnv = {
        {"RCCL_CE_REDUCESCATTER", "1"},
    };

    const std::vector<UseCeRsCase> cases = {
        {"DisabledByDefault_Isolated", 4, 1, true, NCCL_CTA_POLICY_ZERO, 1024, ncclSum, ncclFloat32, false, {}},
        {"EligibleFloat32Sum_Isolated", 4, 1, true, NCCL_CTA_POLICY_ZERO, 1024, ncclSum, ncclFloat32, true, baseEnv},
        {"MultiNodeRejected_Isolated", 4, 2, true, NCCL_CTA_POLICY_ZERO, 1024, ncclSum, ncclFloat32, false, baseEnv},
        {"NoSymmetricSupportRejected_Isolated", 4, 1, false, NCCL_CTA_POLICY_ZERO, 1024, ncclSum, ncclFloat32, false, baseEnv},
        {"WrongCtaPolicyRejected_Isolated", 4, 1, true, NCCL_CTA_POLICY_DEFAULT, 1024, ncclSum, ncclFloat32, false, baseEnv},
        // recvcount is already the per-rank shard, so it does not have to divide nRanks.
        {"RecvcountNotDivisibleByRanksStillEligible_Isolated", 4, 1, true, NCCL_CTA_POLICY_ZERO, 4097, ncclSum, ncclFloat32, true, baseEnv},
        // Non-power-of-2 rank counts are eligible too, and are the ones whose
        // staging layout the chunk-layout tests above cover.
        {"EligibleSixRanks_Isolated", 6, 1, true, NCCL_CTA_POLICY_ZERO, 683, ncclSum, ncclFloat32, true, baseEnv},
        {"ZeroCountRejected_Isolated", 4, 1, true, NCCL_CTA_POLICY_ZERO, 0, ncclSum, ncclFloat32, false, baseEnv},
        {"UnsupportedOpRejected_Isolated", 4, 1, true, NCCL_CTA_POLICY_ZERO, 1024, ncclAvg, ncclFloat32, false, baseEnv},
        {"Float8Rejected_Isolated", 4, 1, true, NCCL_CTA_POLICY_ZERO, 1024, ncclSum, ncclFloat8e4m3, false, baseEnv},
        // msgBytes is recvcount * sizeof(datatype) * nRanks, not recvcount alone.
        {"MessageTooLargeRejected_Isolated", 4, 1, true, NCCL_CTA_POLICY_ZERO,
         (kCeRsMaxMsgBytesDefault / (sizeof(float) * 4)) + 1, ncclSum, ncclFloat32, false, baseEnv},
        {"MessageAtCapAccepted_Isolated", 4, 1, true, NCCL_CTA_POLICY_ZERO,
         kCeRsMaxMsgBytesDefault / (sizeof(float) * 4), ncclSum, ncclFloat32, true, baseEnv},
    };

    for(const auto& tc : cases)
    {
        auto env = tc.extraEnv;
        ProcessIsolatedTestRunner::registerTest(
            ProcessIsolatedTestRunner::TestConfig(
                tc.name,
                [tc]()
                {
                    CeReduceScatterMockComm mock;
                    mock.comm.nRanks           = tc.nRanks;
                    mock.comm.nNodes           = tc.nNodes;
                    mock.comm.symmetricSupport = tc.symmetricSupport;
                    mock.comm.config.CTAPolicy = tc.ctaPolicy;

                    const bool result =
                        rcclUseCeReduceScatter(mock.get(), tc.recvcount, tc.datatype, tc.op);
                    EXPECT_EQ(result, tc.expected) << tc.name;
                })
                .withEnvironment(env)
                .withTimeout(std::chrono::seconds(30))
                .withNumGpus(0));
    }

    ProcessIsolatedTestRunner::ExecutionOptions options;
    options.stopOnFirstFailure = false;
    options.verboseLogging     = true;
    EXPECT_TRUE(ProcessIsolatedTestRunner::executeAllTests(options));
}

} // namespace RcclUnitTesting
