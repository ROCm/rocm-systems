/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-side tests for the per-architecture P2P channel defaults resolved by
// ncclTopoComputeP2pChannels (src/graph/paths.cc). These assert the same values the
// "P2P channel defaults" INFO line prints at init, so the log and the expectations
// cannot drift:
//
//   gfx942 / gfx950 -> 4 * CHANNEL_LIMIT (64)
//   gfx1250         -> MAXCHANNELS (256)
//
// and that setting NCCL_MAX_P2P_NCHANNELS is what moves the bound off that default.
//
// No GPU is required. comm->nChannels is set to MAXCHANNELS so the trailing
// initChannel() loop over [nChannels, p2pnChannels) never executes, which is the only
// part of the function that touches the device.

#include <gtest/gtest.h>
#include <rccl/rccl.h>

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
ResolvedChannels ResolveP2pChannels(const char* arch, int nRanks, int collChannels = kDefaultCollChannels)
{
    ncclComm_t                 comm = nullptr;
    struct ncclTopoSystem      topo;
    struct ncclTopoNode        gpu;
    struct ncclSharedResources sharedRes;

    CreateMockComm(comm, topo, gpu, arch, nRanks);
    AttachMockSharedRes(comm, sharedRes);
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
    // pow2Down(256 / 8 ranks) = 32, tiling the pool without wrapping.
    const ResolvedChannels r = ResolveP2pChannels("gfx1250", /*nRanks=*/8);
    EXPECT_EQ(r.p2pnChannelsPerPeer, 32);
    ExpectPoolInvariant(r);
}

TEST(ChannelDefaults, Gfx950_SaturateOffByDefault)
{
    const ResolvedChannels r = ResolveP2pChannels("gfx950", /*nRanks=*/8);
    EXPECT_EQ(r.p2pnChannelsPerPeer, kInputChannelsPerPeer);
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

// A non power of two request lands on the next power of two, because
// ncclP2pChannelForPart indexes the pool modulo its size.
TEST(ChannelDefaults, Gfx1250_RequestedNonPow2RoundsUp)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "ChannelDefaults_Gfx1250_Requested48",
        []()
        {
            const ResolvedChannels r = ResolveP2pChannels("gfx1250", /*nRanks=*/8);
            EXPECT_EQ(r.p2pnChannels, 64);
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
