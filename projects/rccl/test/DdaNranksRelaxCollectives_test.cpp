/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Unit tests for the RCCL_DDA_NRANKS_RELAX low-rank gate applied to the
// AllGather / ReduceScatter / AllToAll DDA IPC paths (the non-AllReduce
// single-node DDA collectives). Mirrors the AllReduce DdaNranksRelax tests:
// with relax off only the full kDdaNranks clique is eligible; with relax on
// any 2..kDdaNranks participant count is eligible. End-to-end low-rank GPU
// speedups are covered by the rccl-tests collective sweeps.

#include "common/DdaIpcTestHelpers.hpp"
#include "common/ProcessIsolatedTestRunner.hpp"

#include "algorithms/dda/all_gather/dda_all_gather.h"
#include "algorithms/dda/reduce_scatter/dda_reduce_scatter.h"
#include "algorithms/dda/alltoall/dda_alltoall.h"
#include "algorithms/dda/dda_init_detail.h"
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
        },
        {{"RCCL_DDA_NRANKS_RELAX", "1"}});
}

}  // namespace RcclUnitTesting
