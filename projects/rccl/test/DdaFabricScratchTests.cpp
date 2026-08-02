/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-side tests for the DDA fabric scratch-capacity contract: the single
// per-comm scratch buffer (comm->ddaScratchBytes) must be large enough for
// every DDA fabric tier that is nominally enabled, and each tier's eligibility
// predicate must fail closed (return false, i.e. fall back) when it is not.
//
// Whatever sizes the scratch buffer is the producer; these tests pin the
// consumer side, which lives in the eligibility predicates:
//   ncclAllReduceDdaFabricLL/LL128Eligible, and the AllGather/AllToAll/
//   ReduceScatter counterparts (declared in dda_*.h). The sibling
//   DdaFabricEligibilityTests.cpp exercises only the umbrella *DdaFabricEligible
//   predicates and does not call these LL/LL128 predicates at all. This file
//   fills that gap:
//   - every LL/LL128 predicate is invoked and shown to gate on scratch capacity;
//   - the scratch boundary is checked exactly (footprint vs footprint-one-line);
//   - the fixed LL/LL128 footprints are shown to scale with nRanks, so a scratch
//     that suffices at a low rank count is correctly rejected at a high one;
//   - LL128 AllReduce (the one message-dependent footprint) is shown to grow
//     with the message and to have a rank-dependent effective cap.
// The producer side of the same invariant (the shipped scratch buffer is large
// enough for the fixed tiers) is asserted in DdaFabricEpochTests.cpp, which also
// runs in Release CI.
//
// Target: rccl-UnitTestsFixturesDebug. The *DdaFabric{LL,LL128}Eligible
// predicates are out-of-line definitions in librccl (hidden under Release
// visibility), so this file must build in the Debug fixtures target, like its
// sibling DdaFabricEligibilityTests.cpp. No GPU is required: the predicates are
// pure host checks over comm fields.

#include "common/DdaFabricFootprints.hpp"
#include "common/DdaFabricTestHelpers.hpp"

#include "dda_all_gather.h"
#include "dda_all_reduce.h"
#include "dda_alltoall.h"
#include "dda_reduce_scatter.h"
#include "fabric_gpu_barrier.h"
#include "gtest/gtest.h"

namespace RcclUnitTesting {

namespace {

// Footprint constants/helpers (kLLPacketBytes, kLL128LineBytes, ddaLLFixedFootprint,
// ddaLL128FixedFootprint, ddaLL128ArFootprintForBytes) are shared via
// DdaFabricTestHelpers.hpp so this file and DdaFabricEpochTests.cpp mirror the
// launcher constants in one place.

// 4 * float32 = 16 B clears every non-scratch gate: %16 (AG/A2A), %8 (AR/RS),
// all MaxBytes caps, op == ncclSum. So the scratch clause alone decides.
constexpr int    kValidElemCount  = 4;
constexpr size_t kTooSmallScratch = 8; // below any tier footprint

} // namespace

// Fixture (mock comm + dummy buffers) is shared via DdaFabricTestHelpers.hpp.
class DdaFabricScratchTest : public DdaFabricFixture {
};

// ---------------------------------------------------------------------------
// Every LL / LL128 predicate is invoked and gates on scratch capacity
// (ample -> eligible, too small -> ineligible). Ample is the real shipped
// buffer, so the eligible arm asserts a reachable configuration.
// ---------------------------------------------------------------------------

TEST_F(DdaFabricScratchTest, AllReduce_LL_GatesOnScratchCapacity) {
    mockComm_.comm.ddaScratchBytes = DDA_FABRIC_BUFFER_SIZE;
    EXPECT_TRUE(
        ncclAllReduceDdaFabricLLEligible(mockComm_.get(), sendbuff_, recvbuff_, kValidElemCount, ncclFloat32, ncclSum));
    mockComm_.comm.ddaScratchBytes = kTooSmallScratch;
    EXPECT_FALSE(
        ncclAllReduceDdaFabricLLEligible(mockComm_.get(), sendbuff_, recvbuff_, kValidElemCount, ncclFloat32, ncclSum));
}

TEST_F(DdaFabricScratchTest, AllReduce_LL128_GatesOnScratchCapacity) {
    mockComm_.comm.ddaScratchBytes = DDA_FABRIC_BUFFER_SIZE;
    EXPECT_TRUE(ncclAllReduceDdaFabricLL128Eligible(
        mockComm_.get(), sendbuff_, recvbuff_, kValidElemCount, ncclFloat32, ncclSum));
    mockComm_.comm.ddaScratchBytes = kTooSmallScratch;
    EXPECT_FALSE(ncclAllReduceDdaFabricLL128Eligible(
        mockComm_.get(), sendbuff_, recvbuff_, kValidElemCount, ncclFloat32, ncclSum));
}

TEST_F(DdaFabricScratchTest, AllGather_LL_GatesOnScratchCapacity) {
    mockComm_.comm.ddaScratchBytes = DDA_FABRIC_BUFFER_SIZE;
    EXPECT_TRUE(ncclAllGatherDdaFabricLLEligible(mockComm_.get(), sendbuff_, recvbuff_, kValidElemCount, ncclFloat32));
    mockComm_.comm.ddaScratchBytes = kTooSmallScratch;
    EXPECT_FALSE(ncclAllGatherDdaFabricLLEligible(mockComm_.get(), sendbuff_, recvbuff_, kValidElemCount, ncclFloat32));
}

TEST_F(DdaFabricScratchTest, AllGather_LL128_GatesOnScratchCapacity) {
    mockComm_.comm.ddaScratchBytes = DDA_FABRIC_BUFFER_SIZE;
    EXPECT_TRUE(
        ncclAllGatherDdaFabricLL128Eligible(mockComm_.get(), sendbuff_, recvbuff_, kValidElemCount, ncclFloat32));
    mockComm_.comm.ddaScratchBytes = kTooSmallScratch;
    EXPECT_FALSE(
        ncclAllGatherDdaFabricLL128Eligible(mockComm_.get(), sendbuff_, recvbuff_, kValidElemCount, ncclFloat32));
}

TEST_F(DdaFabricScratchTest, AllToAll_LL_GatesOnScratchCapacity) {
    mockComm_.comm.ddaScratchBytes = DDA_FABRIC_BUFFER_SIZE;
    EXPECT_TRUE(ncclAllToAllDdaFabricLLEligible(mockComm_.get(), sendbuff_, recvbuff_, kValidElemCount, ncclFloat32));
    mockComm_.comm.ddaScratchBytes = kTooSmallScratch;
    EXPECT_FALSE(ncclAllToAllDdaFabricLLEligible(mockComm_.get(), sendbuff_, recvbuff_, kValidElemCount, ncclFloat32));
}

TEST_F(DdaFabricScratchTest, AllToAll_LL128_GatesOnScratchCapacity) {
    mockComm_.comm.ddaScratchBytes = DDA_FABRIC_BUFFER_SIZE;
    EXPECT_TRUE(
        ncclAllToAllDdaFabricLL128Eligible(mockComm_.get(), sendbuff_, recvbuff_, kValidElemCount, ncclFloat32));
    mockComm_.comm.ddaScratchBytes = kTooSmallScratch;
    EXPECT_FALSE(
        ncclAllToAllDdaFabricLL128Eligible(mockComm_.get(), sendbuff_, recvbuff_, kValidElemCount, ncclFloat32));
}

TEST_F(DdaFabricScratchTest, ReduceScatter_LL_GatesOnScratchCapacity) {
    mockComm_.comm.ddaScratchBytes = DDA_FABRIC_BUFFER_SIZE;
    EXPECT_TRUE(ncclReduceScatterDdaFabricLLEligible(
        mockComm_.get(), sendbuff_, recvbuff_, kValidElemCount, ncclFloat32, ncclSum));
    mockComm_.comm.ddaScratchBytes = kTooSmallScratch;
    EXPECT_FALSE(ncclReduceScatterDdaFabricLLEligible(
        mockComm_.get(), sendbuff_, recvbuff_, kValidElemCount, ncclFloat32, ncclSum));
}

TEST_F(DdaFabricScratchTest, ReduceScatter_LL128_GatesOnScratchCapacity) {
    mockComm_.comm.ddaScratchBytes = DDA_FABRIC_BUFFER_SIZE;
    EXPECT_TRUE(ncclReduceScatterDdaFabricLL128Eligible(
        mockComm_.get(), sendbuff_, recvbuff_, kValidElemCount, ncclFloat32, ncclSum));
    mockComm_.comm.ddaScratchBytes = kTooSmallScratch;
    EXPECT_FALSE(ncclReduceScatterDdaFabricLL128Eligible(
        mockComm_.get(), sendbuff_, recvbuff_, kValidElemCount, ncclFloat32, ncclSum));
}

// ---------------------------------------------------------------------------
// Exact scratch boundary: scratch equal to the tier footprint is eligible; one
// line/packet short is not. Covers all four LL tiers and the three fixed-stride
// LL128 tiers. If a mirrored constant drifts from the real launcher constant,
// these go red.
// ---------------------------------------------------------------------------

TEST_F(DdaFabricScratchTest, AllReduce_LL_ScratchBoundaryIsExact) {
    const int nRanks               = mockComm_.comm.nRanks;
    mockComm_.comm.ddaScratchBytes = ddaLLFixedFootprint(nRanks);
    EXPECT_TRUE(
        ncclAllReduceDdaFabricLLEligible(mockComm_.get(), sendbuff_, recvbuff_, kValidElemCount, ncclFloat32, ncclSum));
    mockComm_.comm.ddaScratchBytes = ddaLLFixedFootprint(nRanks) - kLLPacketBytes;
    EXPECT_FALSE(
        ncclAllReduceDdaFabricLLEligible(mockComm_.get(), sendbuff_, recvbuff_, kValidElemCount, ncclFloat32, ncclSum));
}

TEST_F(DdaFabricScratchTest, AllGather_LL_ScratchBoundaryIsExact) {
    const int nRanks               = mockComm_.comm.nRanks;
    mockComm_.comm.ddaScratchBytes = ddaLLFixedFootprint(nRanks);
    EXPECT_TRUE(ncclAllGatherDdaFabricLLEligible(mockComm_.get(), sendbuff_, recvbuff_, kValidElemCount, ncclFloat32));
    mockComm_.comm.ddaScratchBytes = ddaLLFixedFootprint(nRanks) - kLLPacketBytes;
    EXPECT_FALSE(ncclAllGatherDdaFabricLLEligible(mockComm_.get(), sendbuff_, recvbuff_, kValidElemCount, ncclFloat32));
}

TEST_F(DdaFabricScratchTest, AllToAll_LL_ScratchBoundaryIsExact) {
    const int nRanks               = mockComm_.comm.nRanks;
    mockComm_.comm.ddaScratchBytes = ddaLLFixedFootprint(nRanks);
    EXPECT_TRUE(ncclAllToAllDdaFabricLLEligible(mockComm_.get(), sendbuff_, recvbuff_, kValidElemCount, ncclFloat32));
    mockComm_.comm.ddaScratchBytes = ddaLLFixedFootprint(nRanks) - kLLPacketBytes;
    EXPECT_FALSE(ncclAllToAllDdaFabricLLEligible(mockComm_.get(), sendbuff_, recvbuff_, kValidElemCount, ncclFloat32));
}

TEST_F(DdaFabricScratchTest, ReduceScatter_LL_ScratchBoundaryIsExact) {
    const int nRanks               = mockComm_.comm.nRanks;
    mockComm_.comm.ddaScratchBytes = ddaLLFixedFootprint(nRanks);
    EXPECT_TRUE(ncclReduceScatterDdaFabricLLEligible(
        mockComm_.get(), sendbuff_, recvbuff_, kValidElemCount, ncclFloat32, ncclSum));
    mockComm_.comm.ddaScratchBytes = ddaLLFixedFootprint(nRanks) - kLLPacketBytes;
    EXPECT_FALSE(ncclReduceScatterDdaFabricLLEligible(
        mockComm_.get(), sendbuff_, recvbuff_, kValidElemCount, ncclFloat32, ncclSum));
}

TEST_F(DdaFabricScratchTest, AllGather_LL128_ScratchBoundaryIsExact) {
    const int nRanks               = mockComm_.comm.nRanks;
    mockComm_.comm.ddaScratchBytes = ddaLL128FixedFootprint(nRanks);
    EXPECT_TRUE(
        ncclAllGatherDdaFabricLL128Eligible(mockComm_.get(), sendbuff_, recvbuff_, kValidElemCount, ncclFloat32));
    mockComm_.comm.ddaScratchBytes = ddaLL128FixedFootprint(nRanks) - kLL128LineBytes;
    EXPECT_FALSE(
        ncclAllGatherDdaFabricLL128Eligible(mockComm_.get(), sendbuff_, recvbuff_, kValidElemCount, ncclFloat32));
}

TEST_F(DdaFabricScratchTest, AllToAll_LL128_ScratchBoundaryIsExact) {
    const int nRanks               = mockComm_.comm.nRanks;
    mockComm_.comm.ddaScratchBytes = ddaLL128FixedFootprint(nRanks);
    EXPECT_TRUE(
        ncclAllToAllDdaFabricLL128Eligible(mockComm_.get(), sendbuff_, recvbuff_, kValidElemCount, ncclFloat32));
    mockComm_.comm.ddaScratchBytes = ddaLL128FixedFootprint(nRanks) - kLL128LineBytes;
    EXPECT_FALSE(
        ncclAllToAllDdaFabricLL128Eligible(mockComm_.get(), sendbuff_, recvbuff_, kValidElemCount, ncclFloat32));
}

TEST_F(DdaFabricScratchTest, ReduceScatter_LL128_ScratchBoundaryIsExact) {
    const int nRanks               = mockComm_.comm.nRanks;
    mockComm_.comm.ddaScratchBytes = ddaLL128FixedFootprint(nRanks);
    EXPECT_TRUE(ncclReduceScatterDdaFabricLL128Eligible(
        mockComm_.get(), sendbuff_, recvbuff_, kValidElemCount, ncclFloat32, ncclSum));
    mockComm_.comm.ddaScratchBytes = ddaLL128FixedFootprint(nRanks) - kLL128LineBytes;
    EXPECT_FALSE(ncclReduceScatterDdaFabricLL128Eligible(
        mockComm_.get(), sendbuff_, recvbuff_, kValidElemCount, ncclFloat32, ncclSum));
}

// ---------------------------------------------------------------------------
// Fixed footprints scale with nRanks: a scratch that is exactly enough at a low
// rank count must be rejected at kDdaMaxNranks. This is the shape of the
// under-provisioning risk: a scratch not sized per rank count silently disables
// the tier at scale.
// ---------------------------------------------------------------------------

TEST_F(DdaFabricScratchTest, AllReduce_LL_FootprintScalesWithRanks) {
    mockComm_.comm.ddaScratchBytes = ddaLLFixedFootprint(2); // enough for 2 ranks only

    mockComm_.comm.nRanks = 2;
    EXPECT_TRUE(
        ncclAllReduceDdaFabricLLEligible(mockComm_.get(), sendbuff_, recvbuff_, kValidElemCount, ncclFloat32, ncclSum));

    mockComm_.comm.nRanks = meta::comms::kDdaMaxNranks;
    EXPECT_FALSE(
        ncclAllReduceDdaFabricLLEligible(mockComm_.get(), sendbuff_, recvbuff_, kValidElemCount, ncclFloat32, ncclSum))
        << "a scratch sized for 2 ranks must not satisfy " << meta::comms::kDdaMaxNranks << " ranks";
}

TEST_F(DdaFabricScratchTest, AllGather_LL128_FootprintScalesWithRanks) {
    mockComm_.comm.ddaScratchBytes = ddaLL128FixedFootprint(2); // enough for 2 ranks only

    mockComm_.comm.nRanks = 2;
    EXPECT_TRUE(
        ncclAllGatherDdaFabricLL128Eligible(mockComm_.get(), sendbuff_, recvbuff_, kValidElemCount, ncclFloat32));

    mockComm_.comm.nRanks = meta::comms::kDdaMaxNranks;
    EXPECT_FALSE(
        ncclAllGatherDdaFabricLL128Eligible(mockComm_.get(), sendbuff_, recvbuff_, kValidElemCount, ncclFloat32))
        << "a scratch sized for 2 ranks must not satisfy " << meta::comms::kDdaMaxNranks << " ranks";
}

// ---------------------------------------------------------------------------
// LL128 AllReduce is the one message-dependent footprint: its required scratch
// grows with the message, and (against a fixed buffer) its effective message cap
// therefore shrinks as nRanks grows.
// ---------------------------------------------------------------------------

TEST_F(DdaFabricScratchTest, AllReduce_LL128_FootprintGrowsWithMessage) {
    const int    nRanks     = mockComm_.comm.nRanks;
    const size_t smallBytes = (size_t)kValidElemCount * sizeof(float); // 16 B
    const size_t bigCount   = 65536 / sizeof(float);                   // 64 KiB message
    const size_t bigBytes   = bigCount * sizeof(float);

    // Scratch sized for the small message: small message eligible, 64 KiB not.
    mockComm_.comm.ddaScratchBytes = ddaLL128ArFootprintForBytes(nRanks, smallBytes);
    EXPECT_TRUE(ncclAllReduceDdaFabricLL128Eligible(
        mockComm_.get(), sendbuff_, recvbuff_, kValidElemCount, ncclFloat32, ncclSum));
    EXPECT_FALSE(
        ncclAllReduceDdaFabricLL128Eligible(mockComm_.get(), sendbuff_, recvbuff_, bigCount, ncclFloat32, ncclSum));

    // Grow scratch to the 64 KiB footprint (exact boundary): now eligible.
    mockComm_.comm.ddaScratchBytes = ddaLL128ArFootprintForBytes(nRanks, bigBytes);
    EXPECT_TRUE(
        ncclAllReduceDdaFabricLL128Eligible(mockComm_.get(), sendbuff_, recvbuff_, bigCount, ncclFloat32, ncclSum));
    mockComm_.comm.ddaScratchBytes = ddaLL128ArFootprintForBytes(nRanks, bigBytes) - kLL128LineBytes;
    EXPECT_FALSE(
        ncclAllReduceDdaFabricLL128Eligible(mockComm_.get(), sendbuff_, recvbuff_, bigCount, ncclFloat32, ncclSum));
}

TEST_F(DdaFabricScratchTest, AllReduce_LL128_EffectiveCapShrinksWithRanks) {
    // A message at the advertised cap (kDdaLL128ArMaxBytes = 1 GiB) is eligible
    // against the real buffer at a low rank count but not at kDdaMaxNranks,
    // because AR-LL128 scratch scales with nRanks * message.
    const size_t oneGiB = (size_t)1 << 30;
    const size_t count  = oneGiB / sizeof(float);
    mockComm_.comm.ddaScratchBytes = DDA_FABRIC_BUFFER_SIZE;

    mockComm_.comm.nRanks = 2;
    EXPECT_TRUE(
        ncclAllReduceDdaFabricLL128Eligible(mockComm_.get(), sendbuff_, recvbuff_, count, ncclFloat32, ncclSum));

    mockComm_.comm.nRanks = meta::comms::kDdaMaxNranks;
    EXPECT_FALSE(
        ncclAllReduceDdaFabricLL128Eligible(mockComm_.get(), sendbuff_, recvbuff_, count, ncclFloat32, ncclSum))
        << "the 1 GiB AR-LL128 cap is not reachable at " << meta::comms::kDdaMaxNranks
        << " ranks against a fixed scratch buffer";
}

// ---------------------------------------------------------------------------
// Count-side gates that precede the scratch check.
// ---------------------------------------------------------------------------

// The per-message alignment gate differs across siblings: AR/RS require %8, AG/A2A
// require %16. count=2 float32 (8 B) is aligned for AR-LL but not AG-LL.
TEST_F(DdaFabricScratchTest, AlignmentGateDiffersAcrossCollectives) {
    mockComm_.comm.ddaScratchBytes = DDA_FABRIC_BUFFER_SIZE;
    EXPECT_TRUE(ncclAllReduceDdaFabricLLEligible(mockComm_.get(), sendbuff_, recvbuff_, 2, ncclFloat32, ncclSum));
    EXPECT_FALSE(ncclAllGatherDdaFabricLLEligible(mockComm_.get(), sendbuff_, recvbuff_, 2, ncclFloat32));
}

// A zero count is rejected before the scratch check, even with ample scratch.
TEST_F(DdaFabricScratchTest, ZeroCountIneligibleRegardlessOfScratch) {
    mockComm_.comm.ddaScratchBytes = DDA_FABRIC_BUFFER_SIZE;
    EXPECT_FALSE(ncclAllReduceDdaFabricLLEligible(mockComm_.get(), sendbuff_, recvbuff_, 0, ncclFloat32, ncclSum));
    EXPECT_FALSE(ncclAllGatherDdaFabricLLEligible(mockComm_.get(), sendbuff_, recvbuff_, 0, ncclFloat32));
}

} // namespace RcclUnitTesting
