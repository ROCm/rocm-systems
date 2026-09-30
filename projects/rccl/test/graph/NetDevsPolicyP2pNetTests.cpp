/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Topology unit tests proving send / recv / all-to-all honor
// NCCL_NETDEVS_POLICY on the P2P NET path.
//
// NCCL 2.29.2 made all2all, send, and recv obey NCCL_NETDEVS_POLICY when
// running over the network. The net transport (transport/net.cc) selects the
// NIC for each P2P send/recv channel through ncclTopoGetNetDev(), which for a
// remote peer resolves the device via the policy-aware ncclTopoGetLocalNet().
// These tests drive ncclTopoGetNetDev() directly -- the exact entry point the
// collectives use -- so the policy behavior is validated where send / recv /
// all-to-all actually consume it (rather than at the lower-level helpers).
//
// Setup is hardware-independent:
//   * Loads tools/topo_expl/models/topo_8p6l_5nic.xml (GPU rank 0 reaches 3
//     NICs) directly from XML -- no real GPUs, NICs, or MPI launch.
//   * Builds a minimal MockComm (topo + rank + nRanks + zeroed peerInfo) so
//     ncclTopoComputePaths()/ncclTopoGetNetDev() do not dereference a null comm.
//   * Sets NCCL_P2P_PXN_LEVEL=0 so (with the default NCCL_CROSS_NIC=2)
//     ncclTopoGetNetDev() skips the PXN/cross-NIC override and returns purely
//     the policy-selected local NIC for (rank, channelId).
//   * Runs each case in an isolated process (RUN_ISOLATED_TEST_WITH_ENV) so the
//     std::call_once policy parse is fresh per case and env values do not leak.
//
// Target: rccl-UnitTestsFixturesDebug (internal symbols are hidden in Release).

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <set>
#include <string>
#include <unistd.h>

#include "comm.h"        // struct ncclComm, struct ncclPeerInfo
#include "graph.h"       // ncclTopoGetNetDev(), ncclTopoComputePaths(), ncclTopoFree()
#include "graph/topo.h"  // struct ncclTopoSystem, ncclTopoGetSystemFromXml(), GPU/NET
#include "graph/xml.h"   // struct ncclXml, ncclTopoGetXmlFromFile()

#include "ProcessIsolatedTestRunner.hpp"

#ifndef RCCL_TEST_SOURCE_DIR
#define RCCL_TEST_SOURCE_DIR "."
#endif

namespace RcclUnitTesting
{

namespace
{

// GPU rank under test and the (remote) peer we resolve a NIC toward. With
// NCCL_P2P_PXN_LEVEL=0 and the default NCCL_CROSS_NIC=2, ncclTopoGetNetDev()
// returns the policy-selected local NIC of kRank for each channel, regardless
// of the peer, so kPeerRank only needs to differ from kRank.
constexpr int kRank     = 0;
constexpr int kPeerRank = 1;

// 8 GPUs / 6 leaf / multi-NIC model. GPU rank 0 lives on NUMA 0 and reaches
// 3 NICs (mlx5_0/1/2), shared with the other 3 GPUs on that NUMA node.
std::string multiNicTopoPath()
{
    return std::string(RCCL_TEST_SOURCE_DIR) +
           "/../tools/topo_expl/models/topo_8p6l_5nic.xml";
}

struct ncclXml* allocateXml(int maxNodes)
{
    size_t size =
        offsetof(struct ncclXml, nodes) + sizeof(struct ncclXmlNode) * maxNodes;
    struct ncclXml* xml = static_cast<struct ncclXml*>(malloc(size));
    if(xml)
    {
        memset(xml, 0, size);
        xml->maxNodes = maxNodes;
        xml->maxIndex = 0;
    }
    return xml;
}

// Build an ncclTopoSystem directly from a topology XML file (no real hardware).
ncclResult_t loadTopoSystem(const char* path, struct ncclTopoSystem** system)
{
    struct ncclXml* xml = allocateXml(NCCL_TOPO_XML_MAX_NODES);
    if(xml == nullptr) return ncclSystemError;

    ncclResult_t res = ncclTopoGetXmlFromFile(path, xml, /*warn=*/0);
    if(res != ncclSuccess)
    {
        free(xml);
        return res;
    }
    res = ncclTopoGetSystemFromXml(xml, system, /*localHostHash=*/0);
    free(xml);
    return res;
}

// Minimal stand-in communicator for topology-only path computation and NIC
// selection. ncclTopoComputePaths()/ncclTopoGetNetDev() dereference comm (PXN
// branch), so we supply just enough state and a zeroed peerInfo[] -- no real
// hardware. Mirrors the harness used by the NCCL_NETDEVS_POLICY parser tests.
struct MockComm
{
    struct ncclComm*     comm     = nullptr;
    struct ncclPeerInfo* peerInfo = nullptr;

    explicit MockComm(struct ncclTopoSystem* system)
    {
        comm = new ncclComm();
        memset(comm, 0, sizeof(*comm));
        comm->topo   = system;
        comm->rank   = kRank;
        comm->nRanks = system->nodes[GPU].count;
        comm->nNodes = 1;

        peerInfo = new ncclPeerInfo[NCCL_TOPO_MAX_NODES];
        memset(peerInfo, 0, NCCL_TOPO_MAX_NODES * sizeof(*peerInfo));
        for(int r = 0; r < NCCL_TOPO_MAX_NODES; r++)
            peerInfo[r].rank = r;
        comm->peerInfo = peerInfo;
    }

    ~MockComm()
    {
        delete[] peerInfo;
        delete comm;
    }

    MockComm(const MockComm&)            = delete;
    MockComm& operator=(const MockComm&) = delete;
};

// Load the multi-NIC fixture, compute paths, run body(comm), then free.
void withTopoComm(const std::function<void(struct ncclComm*)>& body)
{
    const std::string topoPath = multiNicTopoPath();
    if(access(topoPath.c_str(), R_OK) != 0)
    {
        GTEST_SKIP() << "Topology fixture not found: " << topoPath;
        return;
    }

    struct ncclTopoSystem* system = nullptr;
    ASSERT_EQ(loadTopoSystem(topoPath.c_str(), &system), ncclSuccess);
    ASSERT_NE(system, nullptr);

    MockComm mock(system);
    ASSERT_EQ(ncclTopoComputePaths(system, mock.comm), ncclSuccess);

    body(mock.comm);

    ncclTopoFree(system);
}

// NICs locally reachable from rank's GPU -- the pool the policy selects from.
int localNetCount(struct ncclComm* comm)
{
    int gpu = -1;
    EXPECT_EQ(ncclTopoRankToIndex(comm->topo, kRank, &gpu, /*showWarn=*/true), ncclSuccess);
    int locals[NCCL_TOPO_MAX_NODES];
    int count = 0;
    EXPECT_EQ(ncclTopoGetLocal(comm->topo, GPU, gpu, NET, locals, &count, nullptr), ncclSuccess);
    return count;
}

// Net devices that the send/recv NIC selector hands out as the channel rotates.
// This is the collective-facing path: transport/net.cc calls ncclTopoGetNetDev()
// per channel to pick the NIC for each P2P send/recv connection.
std::set<int> netDevsAcrossChannels(struct ncclComm* comm, int numChannels)
{
    std::set<int> devs;
    for(int c = 0; c < numChannels; c++)
    {
        int64_t netId    = -1;
        int     netDev   = -1;
        int     proxyRank = -1;
        EXPECT_EQ(ncclTopoGetNetDev(comm, kRank, /*graph=*/nullptr, c, kPeerRank,
                                    &netId, &netDev, &proxyRank),
                  ncclSuccess);
        devs.insert(netDev);
    }
    return devs;
}

}  // namespace

// NCCL_NETDEVS_POLICY=MAX:1 -- send/recv/all-to-all must funnel every channel
// through a single NIC: ncclTopoGetNetDev() returns the same device for all
// channels.
TEST(NetDevsPolicyP2pNetTests, Max1_SendRecvNetSelectionUsesSingleNic)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "Max1_SendRecvNetSelectionUsesSingleNic",
        []()
        {
            withTopoComm(
                [](struct ncclComm* comm)
                {
                    const int nets = localNetCount(comm);
                    ASSERT_GE(nets, 2) << "fixture must expose multiple NICs to rank 0";

                    std::set<int> devs = netDevsAcrossChannels(comm, /*numChannels=*/nets * 4);
                    EXPECT_EQ(devs.size(), 1u)
                        << "MAX:1 must restrict send/recv to a single network device";
                });
        },
        {{"NCCL_NETDEVS_POLICY", "MAX:1"}, {"NCCL_P2P_PXN_LEVEL", "0"}});
}

// NCCL_NETDEVS_POLICY=ALL -- send/recv/all-to-all must spread across every
// locally reachable NIC: ncclTopoGetNetDev() rotates over all of them.
TEST(NetDevsPolicyP2pNetTests, All_SendRecvNetSelectionRotatesAllNics)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "All_SendRecvNetSelectionRotatesAllNics",
        []()
        {
            withTopoComm(
                [](struct ncclComm* comm)
                {
                    const int nets = localNetCount(comm);
                    ASSERT_GE(nets, 2) << "fixture must expose multiple NICs to rank 0";

                    std::set<int> devs = netDevsAcrossChannels(comm, /*numChannels=*/nets * 4);
                    EXPECT_EQ(static_cast<int>(devs.size()), nets)
                        << "ALL must use every locally reachable network device";
                });
        },
        {{"NCCL_NETDEVS_POLICY", "ALL"}, {"NCCL_P2P_PXN_LEVEL", "0"}});
}

// NCCL_NETDEVS_POLICY=MAX:2 -- send/recv/all-to-all must use exactly two NICs
// (strictly between MAX:1 and the full 3-NIC pool).
TEST(NetDevsPolicyP2pNetTests, Max2_SendRecvNetSelectionUsesTwoNics)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "Max2_SendRecvNetSelectionUsesTwoNics",
        []()
        {
            withTopoComm(
                [](struct ncclComm* comm)
                {
                    const int nets = localNetCount(comm);
                    ASSERT_GE(nets, 3) << "fixture must expose at least 3 NICs to rank 0";

                    std::set<int> devs = netDevsAcrossChannels(comm, /*numChannels=*/nets * 4);
                    EXPECT_EQ(devs.size(), 2u)
                        << "MAX:2 must restrict send/recv to two network devices";
                });
        },
        {{"NCCL_NETDEVS_POLICY", "MAX:2"}, {"NCCL_P2P_PXN_LEVEL", "0"}});
}

}  // namespace RcclUnitTesting
