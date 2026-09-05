/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "common/CeAllReduceTestHelpers.hpp"
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

class CeAllReduceEligibilityTest : public ::testing::Test
{
protected:
    CeAllReduceMockComm mockComm_;
};

TEST_F(CeAllReduceEligibilityTest, FuncToStringReturnsAllReduce)
{
    EXPECT_STREQ(ncclFuncToString(ncclFuncAllReduce), "AllReduce");
}

TEST_F(CeAllReduceEligibilityTest, CeImplementedReturnsFalseForUnsupportedCollectives)
{
    if(!isCeRuntimeDriverSupported())
        GTEST_SKIP() << "CE driver not in supported range";

    EXPECT_FALSE(ncclCeImplemented(ncclFuncBroadcast, ncclDevSum, ncclFloat32));
    EXPECT_FALSE(ncclCeImplemented(ncclFuncReduce, ncclDevSum, ncclFloat32));
}

TEST_F(CeAllReduceEligibilityTest, CeImplementedReturnsTrueForAllReduceOnSupportedDriver)
{
    if(!isCeRuntimeDriverSupported())
        GTEST_SKIP() << "CE driver not in supported range "
                        "(need ROCm >= 7.12 or 7.0.2.x backport [70051831, 70060000))";

    EXPECT_TRUE(ncclCeImplemented(ncclFuncAllReduce, ncclDevSum, ncclFloat32));
}

TEST_F(CeAllReduceEligibilityTest, CeAvailable_EligibleWithSymmetricSingleNode)
{
    if(!isCeRuntimeDriverSupported())
        GTEST_SKIP() << "CE driver not in supported range";

    EXPECT_TRUE(ncclCeAvailable(mockComm_.get(),
                                ncclFuncAllReduce,
                                ncclDevSum,
                                ncclFloat32,
                                ncclSymSendRegRecvReg));
    EXPECT_TRUE(ncclCeAvailable(mockComm_.get(),
                                ncclFuncAllReduce,
                                ncclDevSum,
                                ncclFloat32,
                                ncclSymSendNonregRecvReg));
}

TEST_F(CeAllReduceEligibilityTest, CeAvailable_MultiNodeRejected)
{
    if(!isCeRuntimeDriverSupported())
        GTEST_SKIP() << "CE driver not in supported range";

    mockComm_.comm.nNodes = 2;
    EXPECT_FALSE(ncclCeAvailable(mockComm_.get(),
                                 ncclFuncAllReduce,
                                 ncclDevSum,
                                 ncclFloat32,
                                 ncclSymSendRegRecvReg));
}

TEST_F(CeAllReduceEligibilityTest, CeAvailable_NoSymmetricSupportRejected)
{
    if(!isCeRuntimeDriverSupported())
        GTEST_SKIP() << "CE driver not in supported range";

    mockComm_.comm.symmetricSupport = false;
    EXPECT_FALSE(ncclCeAvailable(mockComm_.get(),
                                 ncclFuncAllReduce,
                                 ncclDevSum,
                                 ncclFloat32,
                                 ncclSymSendRegRecvReg));
}

TEST_F(CeAllReduceEligibilityTest, CeAvailable_UnsupportedWindowRegistrationRejected)
{
    if(!isCeRuntimeDriverSupported())
        GTEST_SKIP() << "CE driver not in supported range";

    EXPECT_FALSE(ncclCeAvailable(mockComm_.get(),
                                 ncclFuncAllReduce,
                                 ncclDevSum,
                                 ncclFloat32,
                                 ncclSymSendNonregRecvNonreg));
    EXPECT_FALSE(ncclCeAvailable(mockComm_.get(),
                                 ncclFuncAllReduce,
                                 ncclDevSum,
                                 ncclFloat32,
                                 ncclSymSendRegRecvNonreg));
}

TEST_F(CeAllReduceEligibilityTest, ChunkLayout_SmallMessageSingleChunk)
{
    constexpr int    nRanks = 4;
    constexpr size_t count  = 4096;

    // ncclCeAllReduce() only ever sees counts the eligibility gate accepted: an
    // exact multiple of nRanks, and no larger than the staging buffer.
    ASSERT_EQ(count % static_cast<size_t>(nRanks), 0u);
    ASSERT_LE(count * sizeof(float), kCeArMaxMsgBytesDefault);

    const size_t shardElems = count / nRanks;
    const size_t shardBytes = shardElems * sizeof(float);
    const size_t slotChunkBytes =
        ncclCeAllReduceSlotChunkBytes(ceAllReduceMaxChunkBytes(nRanks));

    // A shard this small fits one slot, so ncclCeAllReduce() sends it as a single
    // chunk and never enters the pipelined path.
    EXPECT_EQ(shardElems, 1024u);
    EXPECT_LE(shardBytes, slotChunkBytes);
}

// The host scatter addresses staging slots in bytes (rank * slotChunkBytes) while
// the reduce kernel addresses them in elements (rank * slotChunkElems). If those
// two strides disagree by even one byte, every rank but rank 0 reduces shifted
// data. kCeArMaxMsgBytesDefault / nRanks only divides evenly for power-of-2 rank
// counts, so those were the only ones that used to work.
TEST_F(CeAllReduceEligibilityTest, ChunkLayout_SlotStridesAgreeForAnyRankCount)
{
    const std::vector<int>    rankCounts   = {2, 3, 4, 5, 6, 7, 8, 12, 16, 24};
    const std::vector<size_t> elementSizes = {1, 2, 4, 8};

    for(int nRanks : rankCounts)
    {
        const size_t slotChunkBytes =
            ncclCeAllReduceSlotChunkBytes(ceAllReduceMaxChunkBytes(nRanks));
        SCOPED_TRACE("nRanks=" + std::to_string(nRanks));

        // Rank boundaries stay aligned for the kernel's 16B vector loads, and the
        // slots stay inside the buffer ncclCeInit() sized from the raw capacity.
        EXPECT_EQ(slotChunkBytes % 16, 0u);
        EXPECT_LE(slotChunkBytes, ceAllReduceMaxChunkBytes(nRanks));

        for(size_t eltSize : elementSizes)
        {
            // A slot holds a whole number of elements, so the byte view and the
            // element view describe the same stride.
            EXPECT_EQ((slotChunkBytes / eltSize) * eltSize, slotChunkBytes)
                << "eltSize=" << eltSize;
        }
    }
}

TEST_F(CeAllReduceEligibilityTest, ChunkLayout_LargeMessagePipelined)
{
    // A shard only spills past one slot when kCeArMaxMsgBytesDefault / nRanks is
    // not 16B-aligned, i.e. for a non-power-of-2 rank count at the message cap.
    constexpr int nRanks     = 6;
    const size_t  shardElems = ceAllReduceMaxChunkBytes(nRanks) / sizeof(float);
    const size_t  count      = shardElems * nRanks;  // divisible by nRanks by construction

    // The gate would still accept this count, so the layout below is reachable.
    ASSERT_LE(count * sizeof(float), kCeArMaxMsgBytesDefault);

    const size_t shardBytes = shardElems * sizeof(float);
    const size_t slotChunkBytes =
        ncclCeAllReduceSlotChunkBytes(ceAllReduceMaxChunkBytes(nRanks));
    ASSERT_GT(shardBytes, slotChunkBytes);

    // Same bookkeeping ncclCeAllReduce() does once it has picked a chunk size.
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

TEST_F(CeAllReduceEligibilityTest, MaxStagingBytesPerRank)
{
    // The whole message has to fit the per-rank staging capacity ncclCeInit uses.
    for(int nRanks : {2, 3, 4, 5, 6, 7, 8, 12, 16, 24})
    {
        SCOPED_TRACE("nRanks=" + std::to_string(nRanks));
        EXPECT_LE(ceAllReduceMaxChunkBytes(nRanks) * static_cast<size_t>(nRanks),
                  kCeArMaxMsgBytesDefault);
    }
}

TEST(RcclCeAllReduceEligibility, RcclUseCeAllReduce_Isolated)
{
    struct UseCeArCase
    {
        std::string                                  name;
        int                                          nRanks;
        int                                          nNodes;
        bool                                         symmetricSupport;
        int                                          ctaPolicy;
        size_t                                       count;
        ncclRedOp_t                                  op;
        ncclDataType_t                               datatype;
        bool                                         expected;
        std::unordered_map<std::string, std::string> extraEnv;
    };

    const std::unordered_map<std::string, std::string> baseEnv = {
        {"RCCL_CE_ALLREDUCE", "1"},
    };

    const std::vector<UseCeArCase> cases = {
        {"DisabledByDefault_Isolated", 4, 1, true, NCCL_CTA_POLICY_ZERO, 4096, ncclSum, ncclFloat32, false, {}},
        {"EligibleFloat32Sum_Isolated", 4, 1, true, NCCL_CTA_POLICY_ZERO, 4096, ncclSum, ncclFloat32, true, baseEnv},
        {"MultiNodeRejected_Isolated", 4, 2, true, NCCL_CTA_POLICY_ZERO, 4096, ncclSum, ncclFloat32, false, baseEnv},
        {"NoSymmetricSupportRejected_Isolated", 4, 1, false, NCCL_CTA_POLICY_ZERO, 4096, ncclSum, ncclFloat32, false, baseEnv},
        {"WrongCtaPolicyRejected_Isolated", 4, 1, true, NCCL_CTA_POLICY_DEFAULT, 4096, ncclSum, ncclFloat32, false, baseEnv},
        {"CountNotDivisibleByRanksRejected_Isolated", 4, 1, true, NCCL_CTA_POLICY_ZERO, 4097, ncclSum, ncclFloat32, false, baseEnv},
        // Non-power-of-2 rank counts are eligible too, and are the ones whose
        // staging layout the chunk-layout tests above cover; 4098 = 6 * 683.
        {"EligibleSixRanks_Isolated", 6, 1, true, NCCL_CTA_POLICY_ZERO, 4098, ncclSum, ncclFloat32, true, baseEnv},
        {"CountNotDivisibleBySixRanksRejected_Isolated", 6, 1, true, NCCL_CTA_POLICY_ZERO, 4099, ncclSum, ncclFloat32, false, baseEnv},
        {"ZeroCountRejected_Isolated", 4, 1, true, NCCL_CTA_POLICY_ZERO, 0, ncclSum, ncclFloat32, false, baseEnv},
        {"UnsupportedOpRejected_Isolated", 4, 1, true, NCCL_CTA_POLICY_ZERO, 4096, ncclAvg, ncclFloat32, false, baseEnv},
        {"Float8Rejected_Isolated", 4, 1, true, NCCL_CTA_POLICY_ZERO, 4096, ncclSum, ncclFloat8e4m3, false, baseEnv},
        {"MessageTooLargeRejected_Isolated", 4, 1, true, NCCL_CTA_POLICY_ZERO,
         (kCeArMaxMsgBytesDefault / sizeof(float)) + 4, ncclSum, ncclFloat32, false, baseEnv},
    };

    for(const auto& tc : cases)
    {
        auto env = tc.extraEnv;
        ProcessIsolatedTestRunner::registerTest(
            ProcessIsolatedTestRunner::TestConfig(
                tc.name,
                [tc]()
                {
                    CeAllReduceMockComm mock;
                    mock.comm.nRanks           = tc.nRanks;
                    mock.comm.nNodes           = tc.nNodes;
                    mock.comm.symmetricSupport = tc.symmetricSupport;
                    mock.comm.config.CTAPolicy = tc.ctaPolicy;

                    const bool result =
                        rcclUseCeAllReduce(mock.get(), tc.count, tc.datatype, tc.op, /*acc=*/nullptr);
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

