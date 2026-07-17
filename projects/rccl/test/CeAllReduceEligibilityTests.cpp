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
     constexpr int nRanks = 4;
     size_t baseChunkElems{}, tailChunkElems{}, chunksPerShard{}, slotChunkElems{};
     ceAllReduceComputeChunking(4096, sizeof(float), nRanks,
                                baseChunkElems, tailChunkElems, chunksPerShard, slotChunkElems);
 
     EXPECT_EQ(chunksPerShard, 1u);
     EXPECT_EQ(tailChunkElems, 0u);
     EXPECT_EQ(baseChunkElems, 1024u);
     EXPECT_EQ(slotChunkElems, ceAllReduceMaxChunkBytes(nRanks) / sizeof(float));
 }
 
 TEST_F(CeAllReduceEligibilityTest, ChunkLayout_LargeMessagePipelined)
 {
     constexpr int nRanks = 8;
     const size_t count   = 32 * 1024 * 1024 / sizeof(float); // 32 MiB per rank
     size_t baseChunkElems{}, tailChunkElems{}, chunksPerShard{}, slotChunkElems{};
     ceAllReduceComputeChunking(count, sizeof(float), nRanks,
                                baseChunkElems, tailChunkElems, chunksPerShard, slotChunkElems);
 
     EXPECT_GT(chunksPerShard, 1u);
     EXPECT_EQ(slotChunkElems, ceAllReduceMaxChunkBytes(nRanks) / sizeof(float));
 }
 
 TEST_F(CeAllReduceEligibilityTest, MaxStagingBytesPerRank)
 {
     EXPECT_EQ(ceAllReduceMaxChunkBytes(4),
               (NCCL_CE_AR_MAX_MSG_BYTES / 2) / 4);
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
         {"ZeroCountRejected_Isolated", 4, 1, true, NCCL_CTA_POLICY_ZERO, 0, ncclSum, ncclFloat32, false, baseEnv},
         {"UnsupportedOpRejected_Isolated", 4, 1, true, NCCL_CTA_POLICY_ZERO, 4096, ncclAvg, ncclFloat32, false, baseEnv},
         {"Float8Rejected_Isolated", 4, 1, true, NCCL_CTA_POLICY_ZERO, 4096, ncclSum, ncclFloat8e4m3, false, baseEnv},
         {"MessageTooLargeRejected_Isolated", 4, 1, true, NCCL_CTA_POLICY_ZERO,
          (NCCL_CE_AR_MAX_MSG_BYTES / sizeof(float)) + 4, ncclSum, ncclFloat32, false, baseEnv},
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
                         rcclUseCeAllReduce(mock.get(), tc.count, tc.datatype, tc.op);
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
 
