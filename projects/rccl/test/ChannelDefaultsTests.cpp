/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-side tests for the per-architecture P2P channel defaults resolved by
// ncclTopoComputeP2pChannels (src/graph/paths.cc):
//
//   gfx942 / gfx950 -> 4 * CHANNEL_LIMIT (64)
//   gfx1250         -> MAXCHANNELS
//
// and that setting NCCL_MAX_P2P_NCHANNELS is what moves the bound off that default.
//
// No GPU is required. comm->nChannels is set to MAXCHANNELS so the trailing
// initChannel() loop over [nChannels, p2pnChannels) never executes, which is the only
// part of the function that touches the device.

#include <gtest/gtest.h>
#include <rccl/rccl.h>

#include <memory>

#include "comm.h"
#include "common/MockComm.hpp"
#include "common/ProcessIsolatedTestRunner.hpp"
#include "device.h"
#include "graph.h"
#include "graph/topo.h"

namespace RcclUnitTesting
{
namespace
{

constexpr int kDefaultCollChannels = MAXCHANNELS;   // pool the P2P side inherits
constexpr int kInputChannelsPerPeer = 8;            // as set by ncclTopoComputeP2pChannelsPerPeer
constexpr int kNonGfx1250Default = 4 * CHANNEL_LIMIT;

struct ResolvedChannels
{
    int p2pnChannels;
    int p2pnChannelsPerPeer;
};

// Drive the production function with a mock comm and return what it resolved.
//
// topo->nRanks is what the arch-gated branches match on. Multi-node cases set it to the
// total rank count, which the gfx950 caps key off. Single-node cases leave it at 0 by
// default, which deliberately suppresses the per-peer doubling
// (`nodes[GPU].count == topo->nRanks`) so p2pnChannelsPerPeer passes through as
// kInputChannelsPerPeer and the saturate expectations stay easy to derive. Pass
// allRanksLocal to exercise the doubling instead.
ResolvedChannels ResolveP2pChannels(const char* arch, int nRanks, int collChannels = kDefaultCollChannels,
                                    int nNodes = 1, bool allRanksLocal = false)
{
    // Heap, not stack: ncclTopoSystem is ~13 MiB and the default stack is 8 MB.
    ncclComm_t comm      = nullptr;
    auto       topo      = std::make_unique<ncclTopoSystem>();
    auto       gpu       = std::make_unique<ncclTopoNode>();
    auto       sharedRes = std::make_unique<ncclSharedResources>();

    CreateMockComm(comm, *topo, *gpu, arch, nRanks);
    AttachMockSharedRes(comm, *sharedRes);
    if (nNodes > 1) SetMockNodes(comm, nNodes, nRanks);
    // Match nodes[GPU].count, which CreateMockComm sets to 1, so the doubling branch fires.
    else if (allRanksLocal) SetMockNodes(comm, 1, comm->topo->nodes[GPU].count);
    comm->nChannels           = collChannels;
    comm->p2pnChannelsPerPeer = kInputChannelsPerPeer;

    EXPECT_EQ(ncclTopoComputeP2pChannels(comm), ncclSuccess);
    ResolvedChannels resolved{comm->p2pnChannels, comm->p2pnChannelsPerPeer};

    CleanupMockComm(comm);
    return resolved;
}

// ncclP2pChannelToPart cannot recover part indices >= nP2pChannels, so a per-peer count
// above the pool silently aliases parts onto each other (see paths.cc). Every case must
// hold this.
void ExpectPoolInvariant(const ResolvedChannels& r)
{
    EXPECT_LE(r.p2pnChannelsPerPeer, r.p2pnChannels)
        << "p2pnChannelsPerPeer " << r.p2pnChannelsPerPeer << " exceeds pool " << r.p2pnChannels;
}

} // namespace

// ---------------------------------------------------------------------------
// Defaults with nothing set in the environment.
// ---------------------------------------------------------------------------

TEST(ChannelDefaults, Gfx942_DefaultsTo64)
{
    const ResolvedChannels r = ResolveP2pChannels("gfx942", /*nRanks=*/8);
    EXPECT_EQ(r.p2pnChannels, kNonGfx1250Default);
    ExpectPoolInvariant(r);
}

TEST(ChannelDefaults, Gfx950_DefaultsTo64)
{
    const ResolvedChannels r = ResolveP2pChannels("gfx950", /*nRanks=*/8);
    EXPECT_EQ(r.p2pnChannels, kNonGfx1250Default);
    ExpectPoolInvariant(r);
}

TEST(ChannelDefaults, Gfx1250_DefaultsToMaxChannels)
{
    const ResolvedChannels r = ResolveP2pChannels("gfx1250", /*nRanks=*/8);
    EXPECT_EQ(r.p2pnChannels, (int)MAXCHANNELS);
    ExpectPoolInvariant(r);
}

// The pool never exceeds the collective channel count it is drawn from, whatever the
// per-arch default allows.
TEST(ChannelDefaults, Gfx1250_PoolFollowsCollectiveChannels)
{
    const ResolvedChannels r = ResolveP2pChannels("gfx1250", /*nRanks=*/8, /*collChannels=*/32);
    EXPECT_EQ(r.p2pnChannels, 32);
    ExpectPoolInvariant(r);
}

// ---------------------------------------------------------------------------
// Saturate resolution. Unset means on for gfx1250 and off elsewhere.
// ---------------------------------------------------------------------------

TEST(ChannelDefaults, Gfx1250_SaturateOnByDefault)
{
    // pow2Down(pool / nRanks), tiling the pool without wrapping. Computed rather than
    // hardcoded: MAXCHANNELS is 512 in an ENABLE_WARP_SPEED build and 256 otherwise.
    const ResolvedChannels r = ResolveP2pChannels("gfx1250", /*nRanks=*/8);
    EXPECT_EQ(r.p2pnChannelsPerPeer, pow2Down(r.p2pnChannels / 8));
    EXPECT_GT(r.p2pnChannelsPerPeer, kInputChannelsPerPeer) << "saturate should raise the per-peer count";
    ExpectPoolInvariant(r);
}

// nRanks=4 so the two paths differ: saturate on would give pow2Down(64/4) = 16, saturate
// off leaves the input 8. At nRanks=8 both are 8 and the case cannot fail.
TEST(ChannelDefaults, Gfx950_SaturateOffByDefault)
{
    const ResolvedChannels r = ResolveP2pChannels("gfx950", /*nRanks=*/4);
    EXPECT_EQ(r.p2pnChannelsPerPeer, kInputChannelsPerPeer);
    ExpectPoolInvariant(r);
}

// Single-node per-peer doubling, gated on nodes[GPU].count == topo->nRanks and on the arch.
// gfx950 because saturate is off there by default and would otherwise overwrite the result.
TEST(ChannelDefaults, Gfx950_SingleNodeDoublesPerPeer)
{
    const ResolvedChannels r =
      ResolveP2pChannels("gfx950", /*nRanks=*/8, kDefaultCollChannels, /*nNodes=*/1, /*allRanksLocal=*/true);
    EXPECT_EQ(r.p2pnChannelsPerPeer, kInputChannelsPerPeer * 2) << "pow2Up(8) * 2";
    ExpectPoolInvariant(r);
}

// The doubling is arch-gated, so an otherwise identical gfx90a job does not get it.
TEST(ChannelDefaults, Gfx90a_SingleNodeDoesNotDoublePerPeer)
{
    const ResolvedChannels r =
      ResolveP2pChannels("gfx90a", /*nRanks=*/8, kDefaultCollChannels, /*nNodes=*/1, /*allRanksLocal=*/true);
    EXPECT_EQ(r.p2pnChannelsPerPeer, kInputChannelsPerPeer);
    ExpectPoolInvariant(r);
}

// Multi-node gfx950 caps. Dead code until the NCCL_MAX_P2P_NCHANNELS opt-in was fixed,
// since they are gated on upper == defaultMax.
TEST(ChannelDefaults, Gfx950_TwoNode16Ranks_CapsTo32)
{
    const ResolvedChannels r = ResolveP2pChannels("gfx950", /*nRanks=*/16, /*collChannels=*/64, /*nNodes=*/2);
    EXPECT_EQ(r.p2pnChannels, 32);
    ExpectPoolInvariant(r);
}

TEST(ChannelDefaults, Gfx950_TwoNodeHalfSubscribed_CapsTo16)
{
    const ResolvedChannels r = ResolveP2pChannels("gfx950", /*nRanks=*/8, /*collChannels=*/64, /*nNodes=*/2);
    EXPECT_EQ(r.p2pnChannels, 16);
    ExpectPoolInvariant(r);
}

// The caps are gfx950-only, so an otherwise identical gfx942 job keeps the full 64.
TEST(ChannelDefaults, Gfx942_TwoNode16Ranks_NotCapped)
{
    const ResolvedChannels r = ResolveP2pChannels("gfx942", /*nRanks=*/16, /*collChannels=*/64, /*nNodes=*/2);
    EXPECT_EQ(r.p2pnChannels, kNonGfx1250Default);
    ExpectPoolInvariant(r);
}

// ---------------------------------------------------------------------------
// Requested values. NCCL_PARAM caches per process, so each of these runs in its own
// process via the isolated runner.
// ---------------------------------------------------------------------------

TEST(ChannelDefaults, Gfx950_RequestedAbove64IsHonored)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "ChannelDefaults_Gfx950_RequestedAbove64",
        []()
        {
            const ResolvedChannels r = ResolveP2pChannels("gfx950", /*nRanks=*/8);
            EXPECT_EQ(r.p2pnChannels, 128);
            ExpectPoolInvariant(r);
        },
        {{"NCCL_MAX_P2P_NCHANNELS", "128"}});
}

TEST(ChannelDefaults, Gfx950_RequestedBelow64IsHonored)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "ChannelDefaults_Gfx950_RequestedBelow64",
        []()
        {
            const ResolvedChannels r = ResolveP2pChannels("gfx950", /*nRanks=*/8);
            EXPECT_EQ(r.p2pnChannels, 32);
            ExpectPoolInvariant(r);
        },
        {{"NCCL_MAX_P2P_NCHANNELS", "32"}});
}

// Single-node doubling raises p2pnChannelsPerPeer; a user max of 1 must still win.
TEST(ChannelDefaults, Gfx1250_Requested1CapsDoubling)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "ChannelDefaults_Gfx1250_Requested1",
        []()
        {
            const ResolvedChannels r =
              ResolveP2pChannels("gfx1250", /*nRanks=*/4, kDefaultCollChannels, /*nNodes=*/1,
                                 /*allRanksLocal=*/true);
            EXPECT_EQ(r.p2pnChannels, 1);
            EXPECT_EQ(r.p2pnChannelsPerPeer, 1);
            ExpectPoolInvariant(r);
        },
        {{"NCCL_MAX_P2P_NCHANNELS", "1"}});
}

TEST(ChannelDefaults, Gfx1250_RequestedOverridesArchDefault)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "ChannelDefaults_Gfx1250_Requested32",
        []()
        {
            const ResolvedChannels r = ResolveP2pChannels("gfx1250", /*nRanks=*/8);
            EXPECT_EQ(r.p2pnChannels, 32);
            // Saturate is still on, so the per-peer count tiles the smaller pool.
            EXPECT_EQ(r.p2pnChannelsPerPeer, 4);
            ExpectPoolInvariant(r);
        },
        {{"NCCL_MAX_P2P_NCHANNELS", "32"}});
}

// Below the arch default, a non-pow2 request takes the existing pow2Up rounding: 48 -> 64.
TEST(ChannelDefaults, Gfx1250_RequestedNonPow2RoundsUp)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "ChannelDefaults_Gfx1250_Requested48",
        []()
        {
            const ResolvedChannels r = ResolveP2pChannels("gfx1250", /*nRanks=*/8);
            EXPECT_EQ(r.p2pnChannels, 64);
            EXPECT_EQ(r.p2pnChannels & (r.p2pnChannels - 1), 0) << "pool must be a power of two";
            ExpectPoolInvariant(r);
        },
        {{"NCCL_MAX_P2P_NCHANNELS", "48"}});
}

TEST(ChannelDefaults, Gfx1250_SaturateCanBeDisabled)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "ChannelDefaults_Gfx1250_SaturateOff",
        []()
        {
            const ResolvedChannels r = ResolveP2pChannels("gfx1250", /*nRanks=*/8);
            EXPECT_EQ(r.p2pnChannels, (int)MAXCHANNELS);
            EXPECT_EQ(r.p2pnChannelsPerPeer, kInputChannelsPerPeer);
            ExpectPoolInvariant(r);
        },
        {{"RCCL_SATURATE_P2P_NCHANNELS", "0"}});
}

TEST(ChannelDefaults, Gfx950_SaturateCanBeEnabled)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "ChannelDefaults_Gfx950_SaturateOn",
        []()
        {
            // pow2Down(64 / 4 ranks) = 16, distinct from the unsaturated input of 8.
            const ResolvedChannels r = ResolveP2pChannels("gfx950", /*nRanks=*/4);
            EXPECT_EQ(r.p2pnChannels, kNonGfx1250Default);
            EXPECT_EQ(r.p2pnChannelsPerPeer, 16);
            ExpectPoolInvariant(r);
        },
        {{"RCCL_SATURATE_P2P_NCHANNELS", "1"}});
}

} // namespace RcclUnitTesting
