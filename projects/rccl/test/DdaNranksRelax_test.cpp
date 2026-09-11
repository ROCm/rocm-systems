/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Unit tests for the RCCL_DDA_NRANKS_RELAX low-rank DDA IPC AllReduce gate.
//
// These exercise ncclAllReduceDdaIpcEligible() and ncclDdaNranksRelaxEnabled()
// with the mock ncclComm (no GPUs required).
//
// RCCL_PARAM values are cached per-process, so the fixture tests below cover the
// default (relax-off) semantics that the eligibility change must preserve:
// exactly kDdaNranks stays eligible, and 2..7 rank comms are rejected unless the
// operator explicitly opts in.
//
// The relax-enabled path is covered by DdaNranksRelaxIsolatedTest, which re-execs
// this binary with RCCL_DDA_NRANKS_RELAX=1 pre-set (the value must be in the
// environment before any param read) and asserts every count in
// [2, kDdaNranks] becomes eligible while counts outside that range do not.
// End-to-end engagement and numerics are additionally covered by the rccl-tests
// AllReduce sweep with RCCL_DDA_NRANKS_RELAX=1.

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

// The dispatch gate agrees with eligibility when relax is off: the full clique
// reaches the launch path (and fails only the scratch-size check), while a
// low-rank comm is refused by ncclDdaIpcNranksSupported() before it. Distinct
// return codes keep the two apart. No GPU: both return before any kernel launch.
TEST_F(DdaNranksRelaxTest, DispatchRejectsLowRankWhenRelaxOff)
{
    mockComm_.comm.ddaScratchBytes = 0;

    mockComm_.comm.nRanks = nccl_dda_detail::kDdaNranks;
    EXPECT_EQ(ncclAllReduceDdaIpc(sendbuff_, recvbuff_, kCount, ncclFloat32, ncclSum,
                                  mockComm_.get(), nullptr),
              ncclInvalidArgument);

    for (int nRanks : {2, 3, 4, 5, 6, 7})
    {
        mockComm_.comm.nRanks = nRanks;
        EXPECT_EQ(ncclAllReduceDdaIpc(sendbuff_, recvbuff_, kCount, ncclFloat32, ncclSum,
                                      mockComm_.get(), nullptr),
                  ncclInvalidUsage)
            << "nRanks=" << nRanks << " must be refused with relax off";
    }
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

            // Eligibility alone does not prove the dispatch routes the count: with
            // only the checks above, deleting a supported count from
            // ncclAllReduceDdaIpcTyped() would still pass. Drive the real entry
            // point and separate the two failure modes by rank count.
            //
            // ddaScratchBytes = 0 makes every supported count fail the scratch-size
            // check inside ncclAllReduceDdaIpcLaunch() and return
            // ncclInvalidArgument, which is reached only after the dispatch has
            // accepted the count -- and before any barrier deref or kernel launch,
            // so this needs no GPU. An unsupported count is rejected earlier by the
            // ncclDdaIpcNranksSupported() gate and returns ncclInvalidUsage.
            mockComm.comm.ddaScratchBytes = 0;

            for (int nRanks = 2; nRanks <= nccl_dda_detail::kDdaNranks; ++nRanks)
            {
                mockComm.comm.nRanks = nRanks;
                EXPECT_EQ(ncclAllReduceDdaIpc(sendbuff, recvbuff, count, ncclFloat32,
                                              ncclSum, mockComm.get(), nullptr),
                          ncclInvalidArgument)
                    << "nRanks=" << nRanks
                    << " should reach the launch path and fail the scratch check";
            }

            for (int nRanks : {1, 9, 16})
            {
                mockComm.comm.nRanks = nRanks;
                EXPECT_EQ(ncclAllReduceDdaIpc(sendbuff, recvbuff, count, ncclFloat32,
                                              ncclSum, mockComm.get(), nullptr),
                          ncclInvalidUsage)
                    << "nRanks=" << nRanks << " must be refused by the dispatch gate";
            }
        },
        {{"RCCL_DDA_NRANKS_RELAX", "1"}});
}

}  // namespace RcclUnitTesting
