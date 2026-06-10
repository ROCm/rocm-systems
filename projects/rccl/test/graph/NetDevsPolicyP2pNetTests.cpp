/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Adding topology unit test for getLocalNetCountByBw / ncclTopoGetLocalNet.
//
// NCCL 2.29.2 made all2all, send, and recv obey NCCL_NETDEVS_POLICY by routing
// P2P NET channel selection through ncclTopoGetLocalNet() (which applies the
// policy) instead of enumerating every local NIC via ncclTopoGetLocal().
// getLocalNetCountByBw() is the function ncclTopoGetNchannels() uses for remote
// peers, so these assertions exercise the same path as send/recv/alltoall.
//
// Uses topo_8p6l_5nic.xml (3 NICs on NUMA 0, GPU rank 0) with process-isolated
// env settings so std::call_once policy parsing is fresh per case.
//
// Target: rccl-UnitTestsFixturesDebug (internal symbols are hidden in Release).

#include <gtest/gtest.h>

#include <functional>
#include <string>
#include <unistd.h>

#include "graph.h"
#include "graph/topo.h"
#include "graph/xml.h"

#include "common/ProcessIsolatedTestRunner.hpp"

namespace RcclUnitTesting
{

namespace
{

static std::string multiNicTopoPath()
{
#ifdef RCCL_TEST_SOURCE_DIR
    return std::string(RCCL_TEST_SOURCE_DIR) + "/../tools/topo_expl/models/topo_8p6l_5nic.xml";
#else
    return "../tools/topo_expl/models/topo_8p6l_5nic.xml";
#endif
}

static ncclResult_t loadTopoSystem(const char* path, ncclTopoSystem** system)
{
    struct ncclXml* xml = nullptr;
    NCCLCHECK(xmlAlloc(&xml, NCCL_GRAPH_XML_MAX_NODES));
    NCCLCHECK(ncclTopoGetXmlFromFile(path, xml, /*warn=*/1));
    NCCLCHECK(ncclTopoGetSystemFromXml(xml, system, /*localHostHash=*/0));
    free(xml);
    return ncclSuccess;
}

static void runWithLoadedTopo(const std::function<void(ncclTopoSystem*)>& body)
{
    const std::string topoPath = multiNicTopoPath();
    if (access(topoPath.c_str(), R_OK) != 0)
    {
        GTEST_SKIP() << "Topology fixture not found: " << topoPath;
        return;
    }

    ncclTopoSystem* system = nullptr;
    ASSERT_EQ(loadTopoSystem(topoPath.c_str(), &system), ncclSuccess);
    ASSERT_NE(system, nullptr);
    ASSERT_EQ(ncclTopoComputePaths(system, /*comm=*/nullptr), ncclSuccess);

    body(system);

    ncclTopoFree(system);
}

static int gpuIndexForRank(ncclTopoSystem* system, int rank)
{
    for (int g = 0; g < system->nodes[GPU].count; g++)
    {
        if (system->nodes[GPU].nodes[g].gpu.rank == rank)
            return g;
    }
    return -1;
}

static void assertGetLocalNetCountByBw(ncclTopoSystem* system, int gpu, int expectedCount)
{
    int   count = -1;
    float bw    = 0.0f;
    ASSERT_EQ(getLocalNetCountByBw(system, gpu, &count, &bw), ncclSuccess);
    EXPECT_EQ(count, expectedCount);
}

static void assertDistinctNetsAcrossChannels(ncclTopoSystem* system,
                                             int              rank,
                                             bool             expectDistinct)
{
    int64_t netId0 = 0;
    int64_t netId1 = 0;
    ASSERT_EQ(ncclTopoGetLocalNet(system, rank, /*channelId=*/0, &netId0, nullptr), ncclSuccess);
    ASSERT_EQ(ncclTopoGetLocalNet(system, rank, /*channelId=*/1, &netId1, nullptr), ncclSuccess);

    if (expectDistinct)
        EXPECT_NE(netId0, netId1);
    else
        EXPECT_EQ(netId0, netId1);
}

} // namespace

TEST(NetDevsPolicyP2pNetTests, Max1_LimitsGetLocalNetCountByBw)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "Max1_LimitsGetLocalNetCountByBw",
        []()
        {
            runWithLoadedTopo([](ncclTopoSystem* system)
                              {
                                  const int gpu = gpuIndexForRank(system, /*rank=*/0);
                                  ASSERT_GE(gpu, 0);
                                  assertGetLocalNetCountByBw(system, gpu, /*expectedCount=*/1);
                                  assertDistinctNetsAcrossChannels(system, /*rank=*/0,
                                                                   /*expectDistinct=*/false);
                              });
        },
        {{"NCCL_NETDEVS_POLICY", "MAX:1"}});
}

TEST(NetDevsPolicyP2pNetTests, All_UsesMultipleNetsInGetLocalNetCountByBw)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "All_UsesMultipleNetsInGetLocalNetCountByBw",
        []()
        {
            runWithLoadedTopo([](ncclTopoSystem* system)
                              {
                                  const int gpu = gpuIndexForRank(system, /*rank=*/0);
                                  ASSERT_GE(gpu, 0);

                                  int   count = -1;
                                  float bw    = 0.0f;
                                  ASSERT_EQ(getLocalNetCountByBw(system, gpu, &count, &bw),
                                            ncclSuccess);
                                  EXPECT_GE(count, 2);

                                  assertDistinctNetsAcrossChannels(system, /*rank=*/0,
                                                                   /*expectDistinct=*/true);
                              });
        },
        {{"NCCL_NETDEVS_POLICY", "ALL"}});
}

} // namespace RcclUnitTesting
