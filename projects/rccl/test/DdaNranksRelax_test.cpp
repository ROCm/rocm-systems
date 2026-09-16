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
// No test_runner config currently sets RCCL_DDA_NRANKS_RELAX as an environment
// variable, so end-to-end engagement via the rccl-tests AllReduce sweep is not
// yet exercised in CI; the isolated-process test above is what actually covers
// the relaxed dispatch today.

#include "common/DdaIpcTestHelpers.hpp"
#include "common/ProcessIsolatedTestRunner.hpp"

#include "algorithms/dda/all_reduce/dda_all_reduce.h"
#include "algorithms/dda/all_gather/dda_all_gather.h"
#include "algorithms/dda/reduce_scatter/dda_reduce_scatter.h"
#include "algorithms/dda/alltoall/dda_alltoall.h"
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
// still rejects counts outside that range) when the operator opts in. No
// test_runner config sets RCCL_DDA_NRANKS_RELAX as an environment variable yet,
// so this in-process test is what actually covers the relaxed dispatch today.
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
            // (kDdaNranks uses the specialised kernel, the rest the NRANKS == 0 one).
            for (int nRanks = 2; nRanks <= nccl_dda_detail::kDdaNranks; ++nRanks)
            {
                mockComm.comm.nRanks = nRanks;
                EXPECT_TRUE(ncclAllReduceDdaIpcEligible(
                    mockComm.get(), sendbuff, recvbuff, count, ncclFloat32, ncclSum))
                    << "nRanks=" << nRanks << " should be eligible with relax on";
            }

            // This PR relaxes only the AllReduce DDA IPC floor: the
            // AllGather/ReduceScatter/AllToAll DDA IPC *eligibility* functions
            // below must stay 8-rank-only even with the knob on, without
            // relying on inspecting rcclDdaEnabled()'s minRanks argument at
            // each call site. This does not mean those collectives are
            // unaffected by the knob at low rank counts: RCCL_FORCE_CE's
            // separate CE-scratch fast path (enqueue.cc) also keys off
            // comm->ddaScratch, which this knob now populates below 8 ranks
            // too -- see RCCL_DDA_NRANKS_RELAX in env-variables.rst.
            mockComm.comm.nRanks = 4;
            EXPECT_FALSE(ncclAllGatherDdaIpcEligible(mockComm.get(), sendbuff, recvbuff, count, ncclFloat32))
                << "AllGather must stay 8-rank-only even with relax on";
            EXPECT_FALSE(
                ncclReduceScatterDdaIpcEligible(mockComm.get(), sendbuff, recvbuff, count, ncclFloat32, ncclSum))
                << "ReduceScatter must stay 8-rank-only even with relax on";
            EXPECT_FALSE(ncclAllToAllDdaIpcEligible(mockComm.get(), sendbuff, recvbuff, count, ncclFloat32))
                << "AllToAll must stay 8-rank-only even with relax on";

            // Counts outside [2, kDdaNranks] stay ineligible.
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

// The isolated test above only ever uses count=1024 (4 KiB), always below
// kDdaFlatTreeThresholdBytes (256 KiB), so the tree branch's
// `count % comm->nRanks` divisibility check is never exercised at a relaxed
// (non-8) rank count -- that branch only runs above the threshold, and 8 was
// the only rank count reachable there before this PR. Pin it directly at
// nRanks=3: one count that divides evenly (eligible) and one immediately
// above it that does not (ineligible), both otherwise identical.
TEST(DdaNranksRelaxIsolatedTest, TreeThresholdDivisibilityAtRelaxedRankCount)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "TreeThresholdDivisibilityAtRelaxedRankCount",
        []()
        {
            void*          sendbuff = reinterpret_cast<void*>(0x10);
            void*          recvbuff = reinterpret_cast<void*>(0x20);
            DdaIpcMockComm mockComm;
            mockComm.comm.nRanks = 3;

            // 196608 floats = 786432 B (> 256 KiB, tree path); 196608 % 3 == 0
            // and the per-rank slice (65536 floats = 262144 B) is 16-byte aligned.
            EXPECT_TRUE(ncclAllReduceDdaIpcEligible(
                mockComm.get(), sendbuff, recvbuff, 196608, ncclFloat32, ncclSum))
                << "196608 % 3 == 0 should be eligible for the tree path";

            // One count higher: same size class, but 196612 % 3 == 1.
            EXPECT_FALSE(ncclAllReduceDdaIpcEligible(
                mockComm.get(), sendbuff, recvbuff, 196612, ncclFloat32, ncclSum))
                << "196612 % 3 == 1 must not be eligible for the tree path";

            // 196611 % 3 == 0 (divisible), but the per-rank slice (65537 floats =
            // 262148 B) is not 16-byte aligned: 262148 % 16 == 4. Divisibility
            // alone is not sufficient for the tree path.
            EXPECT_FALSE(ncclAllReduceDdaIpcEligible(
                mockComm.get(), sendbuff, recvbuff, 196611, ncclFloat32, ncclSum))
                << "196611 % 3 == 0 but the per-rank slice is not 16-byte aligned";
        },
        {{"RCCL_DDA_NRANKS_RELAX", "1"}});
}

// ncclAllReduceDdaIpcTreeEligible() directly: the pure predicate both
// ncclAllReduceDdaIpcEligible() and the IPC launch path call, so a mutation
// or drift between them shows up here as well as at the call sites above.
// No GPU, no comm, no relaxed-rank env var needed.
TEST(DdaAllReduceIpcTreeEligibleTest, RejectsNonPositiveRanks)
{
    EXPECT_FALSE(ncclAllReduceDdaIpcTreeEligible(/*count=*/1024, /*nRanks=*/0, /*typeSize=*/4));
    EXPECT_FALSE(ncclAllReduceDdaIpcTreeEligible(/*count=*/1024, /*nRanks=*/-1, /*typeSize=*/4));
}

TEST(DdaAllReduceIpcTreeEligibleTest, RejectsNonDivisibleCount)
{
    // 196612 % 3 == 1.
    EXPECT_FALSE(ncclAllReduceDdaIpcTreeEligible(/*count=*/196612, /*nRanks=*/3, /*typeSize=*/4));
}

TEST(DdaAllReduceIpcTreeEligibleTest, RejectsDivisibleButUnalignedSlice)
{
    // 196611 % 3 == 0, but (196611 / 3) * 4 == 262148, and 262148 % 16 == 4.
    EXPECT_FALSE(ncclAllReduceDdaIpcTreeEligible(/*count=*/196611, /*nRanks=*/3, /*typeSize=*/4));
}

TEST(DdaAllReduceIpcTreeEligibleTest, AcceptsDivisibleAndAlignedSlice)
{
    // 196608 % 3 == 0, and (196608 / 3) * 4 == 262144, which is 16-byte aligned.
    EXPECT_TRUE(ncclAllReduceDdaIpcTreeEligible(/*count=*/196608, /*nRanks=*/3, /*typeSize=*/4));
}

}  // namespace RcclUnitTesting
