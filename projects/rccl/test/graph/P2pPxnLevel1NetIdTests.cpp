/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Regression tests for NCCL PR #2258 / AICOMRCCL-2038: communicator init
// failed on multi-system and MNNVL topologies when NCCL_P2P_PXN_LEVEL=1.
//
// ncclTopoGetNetDev()'s PXN level-1 branch called
//   ncclTopoGetIntermediateRank(topo, rank, *dev, proxyRank)
// passing the raw plugin device index. The callee's third argument is a
// packed topology id: netId = NCCL_TOPO_ID(systemId, dev) = (systemId << 56)
// | dev. On a fused multi-host / MNNVL XML, systemId != 0, so netId != dev.
// ncclTopoIdToIndex finds no NET node and returns ncclInternalError, which
// aborts init. The graph path and the PXN level-2 branch already passed
// netId. The same call also dereferenced *dev when the caller passed
// dev == NULL (ncclTopoGetPxnRanks, the init-time PXN proxy connect).
//
// These tests drive ncclTopoGetNetDev() / ncclTopoGetPxnRanks() /
// ncclTopoGetIntermediateRank() on a hand-built system whose only NET node
// has a packed id. That is the MNNVL shape: remote GPUs may be present, but
// the local rank's topology carries only local NICs, so no NET node exists
// whose id equals the bare device index. Pre-fix GetNetDev returns
// ncclInternalError (or SIGSEGV for the NULL-dev call); the fixed tree
// returns ncclSuccess.
//
// Hardware-independent. Each case runs in an isolated process because
// NCCL_PARAM caches P2P_PXN_LEVEL in a function-local static.
//
// Target: rccl-UnitTestsFixturesDebug (internal symbols are hidden in Release).

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>

#include "comm.h"
#include "graph.h"
#include "graph/topo.h"

#include "ProcessIsolatedTestRunner.hpp"

namespace RcclUnitTesting
{
namespace
{

constexpr int kRank     = 0;
constexpr int kPeerRank = 1;
constexpr int kNetDev   = 0;

void allocTypedPaths(struct ncclTopoNode* node, int type, int count, int pathType, float bw)
{
    node->paths[type] =
        static_cast<struct ncclTopoLinkList*>(calloc(count, sizeof(struct ncclTopoLinkList)));
    if (node->paths[type] == nullptr) return;
    for (int i = 0; i < count; i++)
    {
        node->paths[type][i].type = pathType;
        node->paths[type][i].bw   = bw;
    }
}

// One local GPU + one local NIC whose topology id is packed with systemId.
// When systemId != 0, netId != net.dev, which is the PR #2258 failure mode.
struct PackedNetComm
{
    struct ncclTopoSystem* system   = nullptr;
    struct ncclComm*       comm     = nullptr;
    struct ncclPeerInfo*   peerInfo = nullptr;

    PackedNetComm(int systemId, int nRanks, int pathType)
    {
        system = static_cast<struct ncclTopoSystem*>(calloc(1, sizeof(struct ncclTopoSystem)));
        if (system == nullptr) abort();
        system->systemId    = systemId;
        system->nHosts      = systemId + 1;
        system->nRanks      = nRanks;
        system->inter       = (nRanks > 1) ? 1 : 0;
        system->netGdrLevel = -2;

        system->nodes[GPU].count = 1;
        auto& gpu                = system->nodes[GPU].nodes[0];
        gpu.type                 = GPU;
        gpu.id                   = NCCL_TOPO_ID(systemId, 0x100);
        gpu.gpu.dev              = 0;
        gpu.gpu.rank             = kRank;
        gpu.gpu.gdrSupport       = 1;
        gpu.gpu.mloPart          = NCCL_TOPO_UNDEF;
        strncpy(gpu.gpu.gcn, "gfx942", GCN_ARCH_NAME_LEN - 1);

        system->nodes[NET].count = 1;
        auto& net                = system->nodes[NET].nodes[0];
        net.type                 = NET;
        net.id                   = NCCL_TOPO_ID(systemId, kNetDev);
        net.net.dev              = kNetDev;
        net.net.gdrSupport       = 1;
        net.net.bw               = 25.0f;

        allocTypedPaths(&gpu, NET, 1, pathType, 25.0f);
        allocTypedPaths(&net, GPU, 1, pathType, 25.0f);

        comm = new ncclComm();
        memset(static_cast<void*>(comm), 0, sizeof(*comm));
        comm->topo       = system;
        comm->rank       = kRank;
        comm->nRanks     = nRanks;
        comm->nNodes     = nRanks;
        comm->pxnDisable = 0; // PXN enabled; skip rcclSetPxn / NCCL_PXN_DISABLE default-off
        comm->ncclNetVer = 5;

        peerInfo = new ncclPeerInfo[nRanks];
        memset(peerInfo, 0, sizeof(ncclPeerInfo) * nRanks);
        for (int r = 0; r < nRanks; r++)
        {
            peerInfo[r].rank    = r;
            peerInfo[r].nvmlDev = 0; // same local GPU index: PXN "use peer's preferred NIC"
        }
        comm->peerInfo = peerInfo;
    }

    ~PackedNetComm()
    {
        if (system) ncclTopoFree(system);
        delete[] peerInfo;
        delete comm;
    }

    PackedNetComm(const PackedNetComm&)            = delete;
    PackedNetComm& operator=(const PackedNetComm&) = delete;

    int64_t packedNetId() const
    {
        return system->nodes[NET].nodes[0].id;
    }
};

const std::unordered_map<std::string, std::string> kPxnLevel1Env = {
    {"NCCL_P2P_PXN_LEVEL", "1"},
    {"NCCL_PXN_DISABLE", "0"},
};

const std::unordered_map<std::string, std::string> kPxnLevel0Env = {
    {"NCCL_P2P_PXN_LEVEL", "0"},
    {"NCCL_PXN_DISABLE", "0"},
};

} // namespace

// The lookup contract the PR repaired: a packed NET id is not interchangeable
// with the bare plugin index once systemId != 0. This is the exact failure
// GetNetDev used to propagate when it passed *dev.
TEST(P2pPxnLevel1NetId, IntermediateRank_PackedIdSucceedsBareDevFails)
{
    RUN_ISOLATED_TEST(
        "IntermediateRank_PackedIdSucceedsBareDevFails",
        []()
        {
            PackedNetComm fx(/*systemId=*/1, /*nRanks=*/2, PATH_PXB);
            ASSERT_NE(fx.packedNetId(), static_cast<int64_t>(kNetDev));

            int proxy = -1;
            EXPECT_EQ(ncclTopoGetIntermediateRank(fx.system, kRank, fx.packedNetId(), &proxy),
                      ncclSuccess);
            EXPECT_EQ(proxy, kRank);

            proxy = -1;
            EXPECT_EQ(ncclTopoGetIntermediateRank(fx.system, kRank, kNetDev, &proxy),
                      ncclInternalError)
                << "bare netDev must not match a packed NET id; this is the PR #2258 miss";
        });
}

// Single-system control: systemId == 0 makes netId == netDev, so the pre-fix
// call was accidentally correct. Keep that path green.
TEST(P2pPxnLevel1NetId, IntermediateRank_SystemId0BareDevMatches)
{
    RUN_ISOLATED_TEST(
        "IntermediateRank_SystemId0BareDevMatches",
        []()
        {
            PackedNetComm fx(/*systemId=*/0, /*nRanks=*/2, PATH_PXB);
            ASSERT_EQ(fx.packedNetId(), static_cast<int64_t>(kNetDev));

            int proxy = -1;
            EXPECT_EQ(ncclTopoGetIntermediateRank(fx.system, kRank, kNetDev, &proxy), ncclSuccess);
            EXPECT_EQ(proxy, kRank);
        });
}

// The production call: ncclTopoGetNetDev on a packed multi-system topology
// with NCCL_P2P_PXN_LEVEL=1. Pre-fix this returned ncclInternalError.
TEST(P2pPxnLevel1NetId, GetNetDev_Level1PackedIdSucceeds)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "GetNetDev_Level1PackedIdSucceeds",
        []()
        {
            PackedNetComm fx(/*systemId=*/1, /*nRanks=*/2, PATH_PXB);
            ASSERT_NE(fx.packedNetId(), static_cast<int64_t>(kNetDev));

            int64_t netId     = -1;
            int     netDev    = -1;
            int     proxyRank = -1;
            ASSERT_EQ(ncclTopoGetNetDev(fx.comm, kRank, /*graph=*/nullptr, /*channelId=*/0,
                                        kPeerRank, &netId, &netDev, &proxyRank),
                      ncclSuccess);
            EXPECT_EQ(netId, fx.packedNetId());
            EXPECT_EQ(netDev, kNetDev);
            EXPECT_NE(netId, static_cast<int64_t>(netDev));
            EXPECT_GE(proxyRank, 0);
        },
        kPxnLevel1Env);
}

// Init path: ncclTopoGetPxnRanks passes dev == NULL. Pre-fix that was an
// unguarded *dev dereference on top of the packed-id miss.
TEST(P2pPxnLevel1NetId, GetNetDev_Level1NullDevSucceeds)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "GetNetDev_Level1NullDevSucceeds",
        []()
        {
            PackedNetComm fx(/*systemId=*/1, /*nRanks=*/2, PATH_PXB);

            int64_t netId     = -1;
            int     proxyRank = -1;
            ASSERT_EQ(ncclTopoGetNetDev(fx.comm, kRank, /*graph=*/nullptr, /*channelId=*/0,
                                        kPeerRank, &netId, /*dev=*/nullptr, &proxyRank),
                      ncclSuccess);
            EXPECT_EQ(netId, fx.packedNetId());
            EXPECT_GE(proxyRank, 0);
        },
        kPxnLevel1Env);
}

// Communicator init (init.cc) connects remote PXN proxies via GetPxnRanks,
// which is the user-visible failure in the 2.31.2 release notes.
TEST(P2pPxnLevel1NetId, GetPxnRanks_Level1PackedIdSucceeds)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "GetPxnRanks_Level1PackedIdSucceeds",
        []()
        {
            PackedNetComm fx(/*systemId=*/1, /*nRanks=*/2, PATH_PXB);
            fx.system->inter = 1;

            int* ranks  = nullptr;
            int  nranks = 0;
            ASSERT_EQ(ncclTopoGetPxnRanks(fx.comm, &ranks, &nranks), ncclSuccess);
            free(ranks);
        },
        kPxnLevel1Env);
}

// LEVEL=0 skips the PXN override. Packed ids must still resolve through
// GetLocalNet without taking the broken branch.
TEST(P2pPxnLevel1NetId, GetNetDev_Level0PackedIdSucceeds)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "GetNetDev_Level0PackedIdSucceeds",
        []()
        {
            PackedNetComm fx(/*systemId=*/1, /*nRanks=*/2, PATH_PXB);

            int64_t netId     = -1;
            int     netDev    = -1;
            int     proxyRank = -1;
            ASSERT_EQ(ncclTopoGetNetDev(fx.comm, kRank, /*graph=*/nullptr, /*channelId=*/0,
                                        kPeerRank, &netId, &netDev, &proxyRank),
                      ncclSuccess);
            EXPECT_EQ(netId, fx.packedNetId());
            EXPECT_EQ(netDev, kNetDev);
            EXPECT_EQ(proxyRank, kRank);
        },
        kPxnLevel0Env);
}

// LEVEL=1 on systemId 0 (netId == netDev). Pre-fix and post-fix are identical.
TEST(P2pPxnLevel1NetId, GetNetDev_Level1SystemId0Succeeds)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "GetNetDev_Level1SystemId0Succeeds",
        []()
        {
            PackedNetComm fx(/*systemId=*/0, /*nRanks=*/2, PATH_PXB);
            ASSERT_EQ(fx.packedNetId(), static_cast<int64_t>(kNetDev));

            int64_t netId     = -1;
            int     netDev    = -1;
            int     proxyRank = -1;
            ASSERT_EQ(ncclTopoGetNetDev(fx.comm, kRank, /*graph=*/nullptr, /*channelId=*/0,
                                        kPeerRank, &netId, &netDev, &proxyRank),
                      ncclSuccess);
            EXPECT_EQ(netId, static_cast<int64_t>(kNetDev));
            EXPECT_EQ(netDev, kNetDev);
        },
        kPxnLevel1Env);
}

// A peer whose nvmlDev is absent from the local system (typical MNNVL remote
// GPU) must still return success: DevToRank fails closed and skips PXN, which
// is not the PR #2258 failure.
TEST(P2pPxnLevel1NetId, GetNetDev_Level1UnknownPeerNvmlDevSucceeds)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "GetNetDev_Level1UnknownPeerNvmlDevSucceeds",
        []()
        {
            PackedNetComm fx(/*systemId=*/1, /*nRanks=*/2, PATH_PXB);
            fx.peerInfo[kPeerRank].nvmlDev = 99;

            int64_t netId     = -1;
            int     netDev    = -1;
            int     proxyRank = -1;
            ASSERT_EQ(ncclTopoGetNetDev(fx.comm, kRank, /*graph=*/nullptr, /*channelId=*/0,
                                        kPeerRank, &netId, &netDev, &proxyRank),
                      ncclSuccess);
            EXPECT_EQ(proxyRank, kRank);
        },
        kPxnLevel1Env);
}

} // namespace RcclUnitTesting
