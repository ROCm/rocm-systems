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
// Whatever sizes the scratch buffer (a fixed value, or a derived one computed
// from the DDA thresholds and rank count) is the producer; these tests pin the
// consumer side, which lives in the eligibility predicates on this branch:
//   ncclAllReduceDdaFabricLL/LL128Eligible, and the AllGather/AllToAll/
//   ReduceScatter counterparts (declared in dda_*.h). The existing
//   DdaFabricEligibilityTests.cpp exercises only the umbrella *DdaFabricEligible
//   predicates and only a single fixed mock scratch; it never calls the LL/LL128
//   predicates nor varies scratch around a tier's footprint. This file fills
//   those gaps:
//   - every LL/LL128 predicate is invoked and shown to gate on scratch capacity;
//   - the scratch boundary is checked exactly (footprint vs footprint-16);
//   - the fixed LL footprint is shown to scale with nRanks, so a scratch that
//     suffices at a low rank count is correctly rejected at a high one.
//
// No GPU is required: the predicates are pure host checks over comm fields.

#include "common/DdaFabricTestHelpers.hpp"

#include "dda_all_gather.h"
#include "dda_all_reduce.h"
#include "dda_alltoall.h"
#include "dda_reduce_scatter.h"
#include "fabric_gpu_barrier.h"
#include "gtest/gtest.h"

namespace RcclUnitTesting {

namespace {

// Fixed LL fabric scratch footprint, mirroring ddaLL{Ar,Ag,A2A,Rs}ScratchSize in
// the launchers:  2 banks * nRanks slots * (kDdaLL*MaxBytes/8) pkts * 16B.
// kDdaLL*MaxBytes is 128 KiB for every LL tier, so the footprint is a fixed
// 512 KiB * nRanks, independent of the message size or configured LL threshold.
// If the source constant changes, the exact-boundary tests below go red (the
// real predicate uses the real constant), which is the intended drift alarm.
constexpr size_t kLLMaxBytesPerRank = 131072; // mirrors meta::comms::kDdaLLArMaxBytes etc.

size_t llFootprint(int nRanks) {
    return (size_t)2 * (size_t)nRanks * (kLLMaxBytesPerRank / 8) * 16;
}

// Larger than any tier's footprint at kDdaMaxNranks, so scratch is never the
// limiting factor. Set explicitly rather than relying on the mock default.
constexpr size_t kAmpleScratch = (size_t)16 << 30; // 16 GiB

} // namespace

class DdaFabricScratchTest : public ::testing::Test {
protected:
    DdaFabricMockComm mockComm_;
    void*             sendbuff_{reinterpret_cast<void*>(0x1000)};
    void*             recvbuff_{reinterpret_cast<void*>(0x2000)};
};

// ---------------------------------------------------------------------------
// Every LL / LL128 predicate is invoked and gates on scratch capacity
// (ample -> eligible, too small -> ineligible). Uses minimal valid messages so
// only the scratch check decides the outcome.
// ---------------------------------------------------------------------------

TEST_F(DdaFabricScratchTest, AllReduce_LL_GatesOnScratchCapacity) {
    mockComm_.comm.ddaScratchBytes = kAmpleScratch;
    EXPECT_TRUE(ncclAllReduceDdaFabricLLEligible(mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32, ncclSum));
    mockComm_.comm.ddaScratchBytes = 8;
    EXPECT_FALSE(ncclAllReduceDdaFabricLLEligible(mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32, ncclSum));
}

TEST_F(DdaFabricScratchTest, AllReduce_LL128_GatesOnScratchCapacity) {
    mockComm_.comm.ddaScratchBytes = kAmpleScratch;
    EXPECT_TRUE(ncclAllReduceDdaFabricLL128Eligible(mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32, ncclSum));
    mockComm_.comm.ddaScratchBytes = 8;
    EXPECT_FALSE(ncclAllReduceDdaFabricLL128Eligible(mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32, ncclSum));
}

TEST_F(DdaFabricScratchTest, AllGather_LL_GatesOnScratchCapacity) {
    mockComm_.comm.ddaScratchBytes = kAmpleScratch;
    EXPECT_TRUE(ncclAllGatherDdaFabricLLEligible(mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32));
    mockComm_.comm.ddaScratchBytes = 8;
    EXPECT_FALSE(ncclAllGatherDdaFabricLLEligible(mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32));
}

TEST_F(DdaFabricScratchTest, AllGather_LL128_GatesOnScratchCapacity) {
    mockComm_.comm.ddaScratchBytes = kAmpleScratch;
    EXPECT_TRUE(ncclAllGatherDdaFabricLL128Eligible(mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32));
    mockComm_.comm.ddaScratchBytes = 8;
    EXPECT_FALSE(ncclAllGatherDdaFabricLL128Eligible(mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32));
}

TEST_F(DdaFabricScratchTest, AllToAll_LL_GatesOnScratchCapacity) {
    mockComm_.comm.ddaScratchBytes = kAmpleScratch;
    EXPECT_TRUE(ncclAllToAllDdaFabricLLEligible(mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32));
    mockComm_.comm.ddaScratchBytes = 8;
    EXPECT_FALSE(ncclAllToAllDdaFabricLLEligible(mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32));
}

TEST_F(DdaFabricScratchTest, AllToAll_LL128_GatesOnScratchCapacity) {
    mockComm_.comm.ddaScratchBytes = kAmpleScratch;
    EXPECT_TRUE(ncclAllToAllDdaFabricLL128Eligible(mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32));
    mockComm_.comm.ddaScratchBytes = 8;
    EXPECT_FALSE(ncclAllToAllDdaFabricLL128Eligible(mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32));
}

TEST_F(DdaFabricScratchTest, ReduceScatter_LL_GatesOnScratchCapacity) {
    mockComm_.comm.ddaScratchBytes = kAmpleScratch;
    EXPECT_TRUE(ncclReduceScatterDdaFabricLLEligible(mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32, ncclSum));
    mockComm_.comm.ddaScratchBytes = 8;
    EXPECT_FALSE(ncclReduceScatterDdaFabricLLEligible(mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32, ncclSum));
}

TEST_F(DdaFabricScratchTest, ReduceScatter_LL128_GatesOnScratchCapacity) {
    mockComm_.comm.ddaScratchBytes = kAmpleScratch;
    EXPECT_TRUE(
        ncclReduceScatterDdaFabricLL128Eligible(mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32, ncclSum));
    mockComm_.comm.ddaScratchBytes = 8;
    EXPECT_FALSE(
        ncclReduceScatterDdaFabricLL128Eligible(mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32, ncclSum));
}

// ---------------------------------------------------------------------------
// Exact scratch boundary for the fixed LL footprint (512 KiB * nRanks): scratch
// exactly equal to the footprint is eligible; one packet short is not.
// ---------------------------------------------------------------------------

TEST_F(DdaFabricScratchTest, AllReduce_LL_ScratchBoundaryIsExact) {
    const int nRanks               = mockComm_.comm.nRanks; // mock default (8)
    mockComm_.comm.ddaScratchBytes = llFootprint(nRanks);
    EXPECT_TRUE(ncclAllReduceDdaFabricLLEligible(mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32, ncclSum));
    mockComm_.comm.ddaScratchBytes = llFootprint(nRanks) - 16;
    EXPECT_FALSE(ncclAllReduceDdaFabricLLEligible(mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32, ncclSum));
}

TEST_F(DdaFabricScratchTest, AllGather_LL_ScratchBoundaryIsExact) {
    const int nRanks               = mockComm_.comm.nRanks;
    mockComm_.comm.ddaScratchBytes = llFootprint(nRanks);
    EXPECT_TRUE(ncclAllGatherDdaFabricLLEligible(mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32));
    mockComm_.comm.ddaScratchBytes = llFootprint(nRanks) - 16;
    EXPECT_FALSE(ncclAllGatherDdaFabricLLEligible(mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32));
}

// ---------------------------------------------------------------------------
// The fixed LL footprint scales with nRanks, so a scratch that is exactly
// enough at a low rank count must be rejected at a high one. This is the shape
// of the under-provisioning risk: a scratch not sized per rank count silently
// disables the tier at scale.
// ---------------------------------------------------------------------------

TEST_F(DdaFabricScratchTest, AllReduce_LL_FootprintScalesWithRanks) {
    mockComm_.comm.ddaScratchBytes = llFootprint(2); // enough for 2 ranks only

    mockComm_.comm.nRanks = 2;
    EXPECT_TRUE(ncclAllReduceDdaFabricLLEligible(mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32, ncclSum));

    mockComm_.comm.nRanks = meta::comms::kDdaMaxNranks;
    EXPECT_FALSE(ncclAllReduceDdaFabricLLEligible(mockComm_.get(), sendbuff_, recvbuff_, 4, ncclFloat32, ncclSum))
        << "a scratch sized for 2 ranks must not satisfy " << meta::comms::kDdaMaxNranks << " ranks";
}

} // namespace RcclUnitTesting
