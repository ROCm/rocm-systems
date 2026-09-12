/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Unit tests for the RCCL_DDA_NRANKS_RELAX low-rank DDA IPC AllReduce gate.
//
// These exercise ncclAllReduceDdaIpcEligible() and ncclDdaNranksRelaxEnabled()
// with the mock ncclComm (no GPUs required). RCCL_PARAM values are cached
// per-process, so these tests cover the default (relax-off) semantics that the
// eligibility change must preserve: exactly kDdaNranks stays eligible and 2/4
// rank comms are rejected unless the operator explicitly opts in. The
// relax-enabled path (2/4-rank engagement + numerics) is covered end-to-end by
// the rccl-tests AllReduce sweep with RCCL_DDA_NRANKS_RELAX=1.

#include "common/DdaIpcTestHelpers.hpp"
#include "common/ProcessIsolatedTestRunner.hpp"

#include "algorithms/dda/all_reduce/dda_all_reduce.h"
#include "algorithms/dda/dda_init_detail.h"
#include "gtest/gtest.h"

namespace RcclUnitTesting
{

class DdaNranksRelaxTest : public ::testing::Test
{
protected:
    DdaIpcMockComm mockComm_;
    void*          sendbuff_{reinterpret_cast<void*>(0x10)};
    void*          recvbuff_{reinterpret_cast<void*>(0x20)};
    static constexpr size_t kCount{1024};  // 4 KiB fp32: flat path, 16B aligned
};

// With RCCL_DDA_NRANKS_RELAX unset the knob defaults to disabled.
TEST_F(DdaNranksRelaxTest, RelaxDisabledByDefault)
{
    EXPECT_FALSE(ncclDdaNranksRelaxEnabled());
}

// Default behaviour is preserved: a full kDdaNranks (8) clique stays eligible.
TEST_F(DdaNranksRelaxTest, FullCliqueEligibleByDefault)
{
    mockComm_.comm.nRanks = nccl_dda_detail::kDdaNranks;
    EXPECT_TRUE(ncclAllReduceDdaIpcEligible(
        mockComm_.get(), sendbuff_, recvbuff_, kCount, ncclFloat32, ncclSum));
}

// Without the relax knob, 4-rank comms are NOT eligible for the DDA IPC path.
TEST_F(DdaNranksRelaxTest, FourRanksRejectedWhenRelaxOff)
{
    mockComm_.comm.nRanks = 4;
    EXPECT_FALSE(ncclAllReduceDdaIpcEligible(
        mockComm_.get(), sendbuff_, recvbuff_, kCount, ncclFloat32, ncclSum));
}

// Likewise for 2-rank comms.
TEST_F(DdaNranksRelaxTest, TwoRanksRejectedWhenRelaxOff)
{
    mockComm_.comm.nRanks = 2;
    EXPECT_FALSE(ncclAllReduceDdaIpcEligible(
        mockComm_.get(), sendbuff_, recvbuff_, kCount, ncclFloat32, ncclSum));
}

// With relax off (default), any count other than the full kDdaNranks clique is
// rejected -- e.g. a 3-rank comm.
TEST_F(DdaNranksRelaxTest, ThreeRanksRejectedWhenRelaxOff)
{
    mockComm_.comm.nRanks = 3;
    EXPECT_FALSE(ncclAllReduceDdaIpcEligible(
        mockComm_.get(), sendbuff_, recvbuff_, kCount, ncclFloat32, ncclSum));
}

// Standard eligibility guards remain intact at full clique size.
TEST_F(DdaNranksRelaxTest, NullCommRejected)
{
    EXPECT_FALSE(ncclAllReduceDdaIpcEligible(
        nullptr, sendbuff_, recvbuff_, kCount, ncclFloat32, ncclSum));
}

TEST_F(DdaNranksRelaxTest, NonSumOpRejected)
{
    mockComm_.comm.nRanks = nccl_dda_detail::kDdaNranks;
    EXPECT_FALSE(ncclAllReduceDdaIpcEligible(
        mockComm_.get(), sendbuff_, recvbuff_, kCount, ncclFloat32, ncclMax));
}

TEST_F(DdaNranksRelaxTest, MissingIpcResourcesRejected)
{
    mockComm_.comm.nRanks = nccl_dda_detail::kDdaNranks;
    mockComm_.setIpcResourcesPresent(false);
    EXPECT_FALSE(ncclAllReduceDdaIpcEligible(
        mockComm_.get(), sendbuff_, recvbuff_, kCount, ncclFloat32, ncclSum));
}

// Relaxed path (RCCL_DDA_NRANKS_RELAX=1). RCCL_PARAM caches per-process and
// NCCL_NO_CACHE is parsed once, so the relaxed value has to be set before any
// param read: run in a fresh re-exec'd process with the env pre-set. This proves
// the eligibility gate opens for every single-node count in [2, kDdaNranks] (and
// still rejects counts outside that range) when the operator opts in; end-to-end
// low-rank GPU engagement is covered by the rccl-tests AllReduce sweep with
// RCCL_DDA_NRANKS_RELAX=1.
TEST(DdaNranksRelaxIsolatedTest, RelaxedPathAdmitsTwoThroughEightRanks)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "RelaxedPathAdmitsTwoThroughEightRanks",
        []()
        {
            void*                  sendbuff = reinterpret_cast<void*>(0x10);
            void*                  recvbuff = reinterpret_cast<void*>(0x20);
            constexpr size_t       count    = 1024;  // 4 KiB fp32: flat path, 16B aligned
            DdaIpcMockComm         mockComm;

            EXPECT_TRUE(ncclDdaNranksRelaxEnabled());

            // With relax on, every single-node count in [2, kDdaNranks] is eligible
            // (each has a template instantiation in the dispatch switch).
            for (int nRanks = 2; nRanks <= nccl_dda_detail::kDdaNranks; ++nRanks)
            {
                mockComm.comm.nRanks = nRanks;
                EXPECT_TRUE(ncclAllReduceDdaIpcEligible(
                    mockComm.get(), sendbuff, recvbuff, count, ncclFloat32, ncclSum))
                    << "nRanks=" << nRanks << " should be eligible with relax on";
            }

            // Counts outside [2, kDdaNranks] stay ineligible (no instantiation).
            for (int nRanks : {1, 9, 16})
            {
                mockComm.comm.nRanks = nRanks;
                EXPECT_FALSE(ncclAllReduceDdaIpcEligible(
                    mockComm.get(), sendbuff, recvbuff, count, ncclFloat32, ncclSum))
                    << "nRanks=" << nRanks << " must not be eligible";
            }
        },
        {{"RCCL_DDA_NRANKS_RELAX", "1"}});
}

}  // namespace RcclUnitTesting
