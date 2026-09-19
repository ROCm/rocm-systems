/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Unit tests for the RCCL_DDA_NRANKS_RELAX low-rank gate applied to the
// AllGather / ReduceScatter / AllToAll DDA IPC paths (the non-AllReduce
// single-node DDA collectives). Mirrors the AllReduce DdaNranksRelax tests:
// with relax off only the full kDdaNranks clique is eligible; with relax on
// any 2..kDdaNranks participant count is eligible. No CI config sets
// RCCL_DDA_NRANKS_RELAX, so the low-rank runtime kernel these collectives
// dispatch to (NRANKS == 0) does not execute anywhere in CI;
// DdaCollectivesNranksRelaxIsolatedTest.RelaxedPathAdmitsTwoThroughEightRanks
// below is what actually covers the dispatch, in-process.

#include "common/DdaAlltoAllTestHelpers.hpp"
#include "common/DdaIpcTestHelpers.hpp"
#include "common/ProcessIsolatedTestRunner.hpp"

#include "algorithms/dda/all_gather/dda_all_gather.h"
#include "algorithms/dda/all_reduce/dda_all_reduce.h"
#include "algorithms/dda/alltoall/dda_alltoall.h"
#include "algorithms/dda/dda_init_detail.h"
#include "algorithms/dda/reduce_scatter/dda_reduce_scatter.h"
#include "gtest/gtest.h"

namespace RcclUnitTesting
{

class DdaCollectivesNranksRelaxTest : public ::testing::Test
{
protected:
    DdaIpcMockComm          mockComm_;
    void*                   sendbuff_{reinterpret_cast<void*>(0x10)};
    void*                   recvbuff_{reinterpret_cast<void*>(0x20)};
    static constexpr size_t kCount{256};  // 1 KiB fp32: flat path, 16-byte aligned
};

// With relax off (default), only the full kDdaNranks clique is eligible; a 4-rank
// comm is rejected for every collective.
TEST_F(DdaCollectivesNranksRelaxTest, FourRanksRejectedWhenRelaxOff)
{
    mockComm_.comm.nRanks = 4;
    EXPECT_FALSE(ncclAllGatherDdaIpcEligible(mockComm_.get(), sendbuff_, recvbuff_, kCount, ncclFloat32));
    EXPECT_FALSE(ncclReduceScatterDdaIpcEligible(mockComm_.get(), sendbuff_, recvbuff_, kCount, ncclFloat32, ncclSum));
    EXPECT_FALSE(ncclAllToAllDdaIpcEligible(mockComm_.get(), sendbuff_, recvbuff_, kCount, ncclFloat32));
}

// The full clique stays eligible by default for every collective.
TEST_F(DdaCollectivesNranksRelaxTest, FullCliqueEligibleByDefault)
{
    mockComm_.comm.nRanks = nccl_dda_detail::kDdaNranks;
    EXPECT_TRUE(ncclAllGatherDdaIpcEligible(mockComm_.get(), sendbuff_, recvbuff_, kCount, ncclFloat32));
    EXPECT_TRUE(ncclReduceScatterDdaIpcEligible(mockComm_.get(), sendbuff_, recvbuff_, kCount, ncclFloat32, ncclSum));
    EXPECT_TRUE(ncclAllToAllDdaIpcEligible(mockComm_.get(), sendbuff_, recvbuff_, kCount, ncclFloat32));
}

// Eligibility is only one of the two gates a low-rank comm has to clear: dispatch
// also calls rcclDdaEnabled() with a participant-count floor. Pin both floors
// directly, since the eligibility tests above never exercise the minRanks
// parameter. A 4-rank comm passes at the relaxed floor and fails at the default.
TEST_F(DdaCollectivesNranksRelaxTest, RcclDdaEnabledHonoursTheRelaxedRankFloor)
{
    DdaAlltoAllMockComm decisionComm;
    decisionComm.reset("gfx950:sramecc+:xnack-");
    decisionComm.comm.nRanks = 4;
    const size_t totalBytes = 8ull * 1024 * 1024;

    EXPECT_TRUE(rcclDdaEnabled(decisionComm.get(), totalBytes, 8388608, /*gfx950Default=*/0,
                               /*gfx1250Default=*/0, /*minRanks=*/2));
    EXPECT_FALSE(rcclDdaEnabled(decisionComm.get(), totalBytes, 8388608, /*gfx950Default=*/0,
                                /*gfx1250Default=*/0, /*minRanks=*/8));
}

// nNodes != 1 is what keeps this feature off multi-node comms, and it is checked
// before the rank gate in all three collectives. Relax must not weaken it.
TEST_F(DdaCollectivesNranksRelaxTest, MultiNodeRejectedRegardlessOfRankCount)
{
    mockComm_.comm.nNodes = 2;
    for (int nRanks : {2, 4, nccl_dda_detail::kDdaNranks})
    {
        mockComm_.comm.nRanks = nRanks;
        EXPECT_FALSE(ncclAllGatherDdaIpcEligible(mockComm_.get(), sendbuff_, recvbuff_, kCount, ncclFloat32))
            << "AllGather nRanks=" << nRanks;
        EXPECT_FALSE(
            ncclReduceScatterDdaIpcEligible(mockComm_.get(), sendbuff_, recvbuff_, kCount, ncclFloat32, ncclSum))
            << "ReduceScatter nRanks=" << nRanks;
        EXPECT_FALSE(ncclAllToAllDdaIpcEligible(mockComm_.get(), sendbuff_, recvbuff_, kCount, ncclFloat32))
            << "AllToAll nRanks=" << nRanks;
    }
}

// Relaxed path (RCCL_DDA_NRANKS_RELAX=1). RCCL_PARAM caches per-process and
// NCCL_NO_CACHE is parsed once, so the value has to be set before any param read:
// run in a fresh re-exec'd process with the env pre-set. Every collective's IPC
// eligibility gate must open for any count in [2, kDdaNranks] and reject outside.
TEST(DdaCollectivesNranksRelaxIsolatedTest, RelaxedPathAdmitsTwoThroughEightRanks)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "RelaxedPathAdmitsTwoThroughEightRanks",
        []()
        {
            void*                  sendbuff = reinterpret_cast<void*>(0x10);
            void*                  recvbuff = reinterpret_cast<void*>(0x20);
            constexpr size_t       count    = 256;  // 1 KiB fp32: flat path, 16-byte aligned
            DdaIpcMockComm         mockComm;

            for (int nRanks = 2; nRanks <= nccl_dda_detail::kDdaNranks; ++nRanks)
            {
                mockComm.comm.nRanks = nRanks;
                EXPECT_TRUE(ncclAllGatherDdaIpcEligible(mockComm.get(), sendbuff, recvbuff, count, ncclFloat32))
                    << "AllGather nRanks=" << nRanks;
                EXPECT_TRUE(ncclReduceScatterDdaIpcEligible(mockComm.get(), sendbuff, recvbuff, count, ncclFloat32,
                                                            ncclSum))
                    << "ReduceScatter nRanks=" << nRanks;
                EXPECT_TRUE(ncclAllToAllDdaIpcEligible(mockComm.get(), sendbuff, recvbuff, count, ncclFloat32))
                    << "AllToAll nRanks=" << nRanks;
            }

            for (int nRanks : {1, 9, 16})
            {
                mockComm.comm.nRanks = nRanks;
                EXPECT_FALSE(ncclAllGatherDdaIpcEligible(mockComm.get(), sendbuff, recvbuff, count, ncclFloat32))
                    << "AllGather nRanks=" << nRanks;
                EXPECT_FALSE(ncclReduceScatterDdaIpcEligible(mockComm.get(), sendbuff, recvbuff, count, ncclFloat32,
                                                             ncclSum))
                    << "ReduceScatter nRanks=" << nRanks;
                EXPECT_FALSE(ncclAllToAllDdaIpcEligible(mockComm.get(), sendbuff, recvbuff, count, ncclFloat32))
                    << "AllToAll nRanks=" << nRanks;
            }

            // Eligibility alone does not prove the dispatch routes the count: with
            // only the checks above, deleting a supported count from any of the
            // three *DdaIpcTyped() dispatchers would still pass. Drive the real
            // entry points and separate the two failure modes by rank count, the
            // same way the AllReduce isolated test does (DdaNranksRelax_test.cpp).
            //
            // ddaScratchBytes = 0 makes every supported count fail the scratch-size
            // check inside each collective's launch wrapper and return
            // ncclInvalidArgument, reached only after the dispatch has accepted the
            // count and before any barrier deref or kernel launch, so this needs no
            // GPU. An unsupported count is rejected earlier by the shared
            // ncclDdaIpcNranksSupported() gate and returns ncclInvalidUsage.
            mockComm.comm.ddaScratchBytes = 0;

            for (int nRanks = 2; nRanks <= nccl_dda_detail::kDdaNranks; ++nRanks)
            {
                mockComm.comm.nRanks = nRanks;
                EXPECT_EQ(ncclAllGatherDdaIpc(sendbuff, recvbuff, count, ncclFloat32, mockComm.get(), nullptr),
                          ncclInvalidArgument)
                    << "AllGather nRanks=" << nRanks << " should reach the launch path";
                EXPECT_EQ(
                    ncclReduceScatterDdaIpc(sendbuff, recvbuff, count, ncclFloat32, ncclSum, mockComm.get(), nullptr),
                    ncclInvalidArgument)
                    << "ReduceScatter nRanks=" << nRanks << " should reach the launch path";
                EXPECT_EQ(ncclAllToAllDdaIpc(sendbuff, recvbuff, count, ncclFloat32, mockComm.get(), nullptr),
                          ncclInvalidArgument)
                    << "AllToAll nRanks=" << nRanks << " should reach the launch path";
            }

            for (int nRanks : {1, 9, 16})
            {
                mockComm.comm.nRanks = nRanks;
                EXPECT_EQ(ncclAllGatherDdaIpc(sendbuff, recvbuff, count, ncclFloat32, mockComm.get(), nullptr),
                          ncclInvalidUsage)
                    << "AllGather nRanks=" << nRanks << " must be refused by the dispatch gate";
                EXPECT_EQ(
                    ncclReduceScatterDdaIpc(sendbuff, recvbuff, count, ncclFloat32, ncclSum, mockComm.get(), nullptr),
                    ncclInvalidUsage)
                    << "ReduceScatter nRanks=" << nRanks << " must be refused by the dispatch gate";
                EXPECT_EQ(ncclAllToAllDdaIpc(sendbuff, recvbuff, count, ncclFloat32, mockComm.get(), nullptr),
                          ncclInvalidUsage)
                    << "AllToAll nRanks=" << nRanks << " must be refused by the dispatch gate";
            }

            // Relax must not open the multi-node door: nNodes != 1 still rejects
            // every count, including ones the relaxed rank gate would admit.
            mockComm.comm.ddaScratchBytes = DDA_IPC_BUFFER_SIZE;
            mockComm.comm.nNodes = 2;
            for (int nRanks : {2, 4, nccl_dda_detail::kDdaNranks})
            {
                mockComm.comm.nRanks = nRanks;
                EXPECT_FALSE(ncclAllGatherDdaIpcEligible(mockComm.get(), sendbuff, recvbuff, count, ncclFloat32))
                    << "multi-node AllGather nRanks=" << nRanks;
                EXPECT_FALSE(ncclReduceScatterDdaIpcEligible(mockComm.get(), sendbuff, recvbuff, count, ncclFloat32,
                                                             ncclSum))
                    << "multi-node ReduceScatter nRanks=" << nRanks;
                EXPECT_FALSE(ncclAllToAllDdaIpcEligible(mockComm.get(), sendbuff, recvbuff, count, ncclFloat32))
                    << "multi-node AllToAll nRanks=" << nRanks;
            }
        },
        {{"RCCL_DDA_NRANKS_RELAX", "1"}});
}

}  // namespace RcclUnitTesting
