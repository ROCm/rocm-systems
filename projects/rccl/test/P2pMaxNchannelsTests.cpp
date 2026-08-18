/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Unit tests for NCCL_MAX_P2P_NCHANNELS resolution and the p2pnChannels
// upper-bound logic in src/graph/paths.cc (ncclMaxP2pNchannels,
// ncclP2pChannelsUpperBound, ncclTopoComputeP2pChannels).
//
// Each case runs in an isolated process so ncclParamMaxP2pNChannels()'s cached
// parse is fresh per env value (same pattern as TopoEnvPolicyTests).
//
// Target: rccl-UnitTestsFixturesDebug (internal symbols are hidden in Release).

#include <gtest/gtest.h>

#include <cstring>
#include <string>

#include "checks.h"
#include "comm.h"
#include "common/ProcessIsolatedTestRunner.hpp"
#include "device.h"
#include "graph.h"
#include "graph/topo.h"

namespace RcclUnitTesting
{

namespace
{

constexpr int kDefaultP2pUpper = 4 * CHANNEL_LIMIT;  // 64

struct P2pChannelsComm
{
    ncclComm            comm{};
    ncclTopoSystem      topo{};
    ncclSharedResources sharedRes{};

    void initSingleNode(const char* gcn, int nRanks, int nChannels, int seedP2pPerPeer)
    {
        ncclTopoNode gpuNode{};
        memset(&comm, 0, sizeof(comm));
        memset(&topo, 0, sizeof(topo));
        memset(&sharedRes, 0, sizeof(sharedRes));
        memset(&gpuNode, 0, sizeof(gpuNode));

        strncpy(gpuNode.gpu.gcn, gcn, GCN_ARCH_NAME_LEN - 1);
        gpuNode.gpu.gcn[GCN_ARCH_NAME_LEN - 1] = '\0';

        topo.nodes[GPU].count = nRanks;
        topo.nRanks           = nRanks;
        topo.type             = RCCL_TOPO_XGMI_ALL;
        topo.nodes[GPU].nodes[0] = gpuNode;

        comm.topo                = &topo;
        comm.sharedRes           = &sharedRes;
        sharedRes.owner            = &comm;
        comm.nRanks                = nRanks;
        comm.nNodes                = 1;
        comm.nChannels             = nChannels;
        comm.p2pnChannelsPerPeer   = seedP2pPerPeer;
    }

    void initMultiNode(const char* gcn, int nNodes, int nRanks, int localGpus, int nChannels,
                       int seedP2pPerPeer)
    {
        ncclTopoNode gpuNode{};
        memset(&comm, 0, sizeof(comm));
        memset(&topo, 0, sizeof(topo));
        memset(&sharedRes, 0, sizeof(sharedRes));
        memset(&gpuNode, 0, sizeof(gpuNode));

        strncpy(gpuNode.gpu.gcn, gcn, GCN_ARCH_NAME_LEN - 1);
        gpuNode.gpu.gcn[GCN_ARCH_NAME_LEN - 1] = '\0';

        topo.nodes[GPU].count = localGpus;
        topo.nRanks           = nRanks;
        topo.type             = RCCL_TOPO_XGMI_ALL;
        topo.nodes[GPU].nodes[0] = gpuNode;

        comm.topo                = &topo;
        comm.sharedRes           = &sharedRes;
        sharedRes.owner            = &comm;
        comm.nRanks                = nRanks;
        comm.nNodes                = nNodes;
        comm.nChannels             = nChannels;
        comm.p2pnChannelsPerPeer   = seedP2pPerPeer;
    }

    ncclResult_t computeP2pChannels(int* outP2pChannels)
    {
        ncclResult_t res = ncclTopoComputeP2pChannels(&comm);
        if(res == ncclSuccess) *outP2pChannels = comm.p2pnChannels;
        return res;
    }
};

void checkMaxP2pResolver(int expected)
{
    EXPECT_EQ(ncclMaxP2pNchannels(), expected);
}

void checkUpperBound(int expectedUpper, bool expectedOptedHigher)
{
    bool optedHigher = !expectedOptedHigher;
    EXPECT_EQ(ncclP2pChannelsUpperBound(&optedHigher), expectedUpper);
    EXPECT_EQ(optedHigher, expectedOptedHigher);
    EXPECT_EQ(ncclP2pChannelsUpperBound(nullptr), expectedUpper);
}

void checkSingleNodeCompute(const char* gcn, int expectedP2pChannels)
{
    P2pChannelsComm fixture;
    // Seed values large enough that only the upper-bound policy (not topology)
    // determines the outcome: nChannels=256, per-peer seed=128 -> doubled to 256
    // before the cap is applied.
    fixture.initSingleNode(gcn, /*nRanks=*/8, /*nChannels=*/256, /*seedP2pPerPeer=*/128);
    int p2pnChannels = -1;
    ASSERT_EQ(fixture.computeP2pChannels(&p2pnChannels), ncclSuccess);
    EXPECT_EQ(p2pnChannels, expectedP2pChannels) << "arch=" << gcn;
}

}  // namespace

// --- ncclMaxP2pNchannels / ncclP2pChannelsUpperBound (param layer) ---------

TEST(P2pMaxNchannelsParamTests, UnsetEnv_ResolvesToMaxChannelsAndDefaultUpper)
{
    RUN_ISOLATED_TEST(
        "UnsetEnv_ResolvesToMaxChannelsAndDefaultUpper",
        []()
        {
            ::unsetenv("NCCL_MAX_P2P_NCHANNELS");
            checkMaxP2pResolver(MAXCHANNELS);
            checkUpperBound(kDefaultP2pUpper, /*expectedOptedHigher=*/false);
        });
}

TEST(P2pMaxNchannelsParamTests, Explicit64_Uses64_NoExtendedOptIn)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "Explicit64_Uses64_NoExtendedOptIn",
        []()
        {
            checkMaxP2pResolver(64);
            checkUpperBound(kDefaultP2pUpper, /*expectedOptedHigher=*/false);
        },
        {{"NCCL_MAX_P2P_NCHANNELS", "64"}});
}

TEST(P2pMaxNchannelsParamTests, Explicit32_Uses32_NoExtendedOptIn)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "Explicit32_Uses32_NoExtendedOptIn",
        []()
        {
            checkMaxP2pResolver(32);
            checkUpperBound(kDefaultP2pUpper, /*expectedOptedHigher=*/false);
        },
        {{"NCCL_MAX_P2P_NCHANNELS", "32"}});
}

TEST(P2pMaxNchannelsParamTests, Explicit128_OptsIntoExtendedUpper)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "Explicit128_OptsIntoExtendedUpper",
        []()
        {
            checkMaxP2pResolver(128);
            checkUpperBound(128, /*expectedOptedHigher=*/true);
        },
        {{"NCCL_MAX_P2P_NCHANNELS", "128"}});
}

TEST(P2pMaxNchannelsParamTests, Explicit256_OptsIntoExtendedUpper)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "Explicit256_OptsIntoExtendedUpper",
        []()
        {
            checkMaxP2pResolver(std::min(256, MAXCHANNELS));
            checkUpperBound(std::min(256, MAXCHANNELS), /*expectedOptedHigher=*/true);
        },
        {{"NCCL_MAX_P2P_NCHANNELS", "256"}});
}

TEST(P2pMaxNchannelsParamTests, ExplicitAboveMaxChannels_ClampsToMaxChannels)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "ExplicitAboveMaxChannels_ClampsToMaxChannels",
        []()
        {
            checkMaxP2pResolver(MAXCHANNELS);
            checkUpperBound(MAXCHANNELS, /*expectedOptedHigher=*/true);
        },
        {{"NCCL_MAX_P2P_NCHANNELS", "512"}});
}

// --- ncclTopoComputeP2pChannels integration (single-node MI3xx arches) ------

class P2pMaxNchannelsSingleNodeTest : public ::testing::TestWithParam<const char*>
{
};

TEST_P(P2pMaxNchannelsSingleNodeTest, UnsetEnv_CapsP2pPoolAt64)
{
    const char* gcn = GetParam();
    RUN_ISOLATED_TEST(
        std::string("UnsetEnv_CapsP2pPoolAt64_") + gcn,
        [gcn]() {
            ::unsetenv("NCCL_MAX_P2P_NCHANNELS");
            checkSingleNodeCompute(gcn, kDefaultP2pUpper);
        });
}

TEST_P(P2pMaxNchannelsSingleNodeTest, Explicit256_AllowsExtendedPool)
{
    const char* gcn = GetParam();
    RUN_ISOLATED_TEST_WITH_ENV(
        std::string("Explicit256_AllowsExtendedPool_") + gcn,
        [gcn]() { checkSingleNodeCompute(gcn, std::min(256, MAXCHANNELS)); },
        {{"NCCL_MAX_P2P_NCHANNELS", "256"}});
}

TEST_P(P2pMaxNchannelsSingleNodeTest, Explicit64_StaysAt64)
{
    const char* gcn = GetParam();
    RUN_ISOLATED_TEST_WITH_ENV(
        std::string("Explicit64_StaysAt64_") + gcn,
        [gcn]() { checkSingleNodeCompute(gcn, 64); },
        {{"NCCL_MAX_P2P_NCHANNELS", "64"}});
}

INSTANTIATE_TEST_SUITE_P(Mi3xxSingleNode, P2pMaxNchannelsSingleNodeTest,
                         ::testing::Values("gfx942", "gfx950", "gfx1250"));

// --- Multi-node MI350 caps apply only without extended opt-in ---------------

TEST(P2pMaxNchannelsMultiNodeTests, Gfx950_2Node16Rank_UnsetEnv_CapsAt32)
{
    RUN_ISOLATED_TEST(
        "Gfx950_2Node16Rank_UnsetEnv_CapsAt32",
        []()
        {
            ::unsetenv("NCCL_MAX_P2P_NCHANNELS");
            P2pChannelsComm fixture;
            fixture.initMultiNode("gfx950", /*nNodes=*/2, /*nRanks=*/16, /*localGpus=*/8,
                                  /*nChannels=*/128, /*seedP2pPerPeer=*/64);
            int p2pnChannels = -1;
            ASSERT_EQ(fixture.computeP2pChannels(&p2pnChannels), ncclSuccess);
            EXPECT_EQ(p2pnChannels, 32);
        });
}

TEST(P2pMaxNchannelsMultiNodeTests, Gfx950_2Node16Rank_Explicit256_Skips32Cap)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "Gfx950_2Node16Rank_Explicit256_Skips32Cap",
        []()
        {
            P2pChannelsComm fixture;
            fixture.initMultiNode("gfx950", /*nNodes=*/2, /*nRanks=*/16, /*localGpus=*/8,
                                  /*nChannels=*/256, /*seedP2pPerPeer=*/128);
            int p2pnChannels = -1;
            ASSERT_EQ(fixture.computeP2pChannels(&p2pnChannels), ncclSuccess);
            EXPECT_EQ(p2pnChannels, std::min(256, MAXCHANNELS));
        },
        {{"NCCL_MAX_P2P_NCHANNELS", "256"}});
}

TEST(P2pMaxNchannelsMultiNodeTests, Gfx950_2Node8Rank_HalfSub_UnsetEnv_CapsAt16)
{
    RUN_ISOLATED_TEST(
        "Gfx950_2Node8Rank_HalfSub_UnsetEnv_CapsAt16",
        []()
        {
            ::unsetenv("NCCL_MAX_P2P_NCHANNELS");
            P2pChannelsComm fixture;
            fixture.initMultiNode("gfx950", /*nNodes=*/2, /*nRanks=*/8, /*localGpus=*/4,
                                  /*nChannels=*/64, /*seedP2pPerPeer=*/32);
            int p2pnChannels = -1;
            ASSERT_EQ(fixture.computeP2pChannels(&p2pnChannels), ncclSuccess);
            EXPECT_EQ(p2pnChannels, 16);
        });
}

}  // namespace RcclUnitTesting
