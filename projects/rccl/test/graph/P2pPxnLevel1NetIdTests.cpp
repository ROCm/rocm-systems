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
// | dev. The graph path and the PXN level-2 branch already passed netId.
// The same call also dereferenced *dev when the caller passed
// dev == NULL (ncclTopoGetPxnRanks, the init-time PXN proxy connect).
//
// On a fused MNNVL XML the local rank still sees host 0's NICs, whose ids
// are NCCL_TOPO_ID(0, dev) == dev. Pre-fix, *dev therefore resolves to host
// 0's NIC and returns success with no PXN redirect: the observable is a
// silently wrong proxyRank (the local rank), not ncclInternalError. A
// topology that omits those host-0 NICs still aborts, and the NULL-dev
// GetPxnRanks call still SIGSEGVs; those remain covered. The MNNVL cases
// below add host 0's NIC plus a real PATH_PXN hop and assert the
// intermediate proxy.
//
// Hardware-independent. Each case runs in an isolated process because
// NCCL_PARAM caches P2P_PXN_LEVEL (and CROSS_NIC) in a function-local static.
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

constexpr int     kRank       = 0;
constexpr int     kPeerRank   = 1;
constexpr int     kProxyRank  = 1; // PATH_PXN intermediate GPU
constexpr int     kNetDev     = 0;
constexpr int64_t kGpuLocalId = 0x100; // keep GPU local id distinct from NET's kNetDev
constexpr float   kNetBw      = 25.0f;
constexpr float   kRemoteBw   = 12.0f;

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

void fillGpu(struct ncclTopoNode* gpu, int systemId, int64_t localId, int dev, int rank)
{
    gpu->type           = GPU;
    gpu->id             = NCCL_TOPO_ID(systemId, localId);
    gpu->gpu.dev        = dev;
    gpu->gpu.rank       = rank;
    gpu->gpu.gdrSupport = 1;
    gpu->gpu.mloPart    = NCCL_TOPO_UNDEF;
    strncpy(gpu->gpu.gcn, "gfx942", GCN_ARCH_NAME_LEN - 1);
}

void fillNet(struct ncclTopoNode* net, int systemId, int netDev, float bw)
{
    net->type           = NET;
    net->id             = NCCL_TOPO_ID(systemId, netDev);
    net->net.dev        = netDev;
    net->net.gdrSupport = 1;
    net->net.bw         = bw;
}

// GPU -> NET PATH_PXN whose first non-NVS hop is proxyGpu. GetIntermediateRank
// starts at list[1], so count must be >= 2 and list[1]->remNode is the proxy.
void setPxnPath(struct ncclTopoNode* gpu, int netIndex, struct ncclTopoNode* proxyGpu)
{
    struct ncclTopoLinkList* path = gpu->paths[NET] + netIndex;
    path->type                    = PATH_PXN;
    path->bw                      = kNetBw;
    path->count                   = 2;
    path->list = static_cast<struct ncclTopoLink**>(calloc(2, sizeof(struct ncclTopoLink*)));
    if (path->list == nullptr) abort();
    gpu->nlinks            = 1;
    gpu->links[0].type     = LINK_NVL;
    gpu->links[0].remNode  = proxyGpu;
    path->list[1]          = &gpu->links[0];
}

// Local packed NIC (systemId may be 0 or != 0). Optional fused-MNNVL shape:
// host 0's NIC (id == bare netDev) plus a PATH_PXN hop through kProxyRank.
struct PackedNetComm
{
    struct ncclTopoSystem* system   = nullptr;
    struct ncclComm*       comm     = nullptr;
    struct ncclPeerInfo*   peerInfo = nullptr;

    struct MnnvlPxn
    {
    };

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
        fillGpu(&system->nodes[GPU].nodes[0], systemId, kGpuLocalId, /*dev=*/0, kRank);

        system->nodes[NET].count = 1;
        fillNet(&system->nodes[NET].nodes[0], systemId, kNetDev, kNetBw);

        allocTypedPaths(&system->nodes[GPU].nodes[0], NET, 1, pathType, kNetBw);
        allocTypedPaths(&system->nodes[NET].nodes[0], GPU, 1, pathType, kNetBw);

        bindComm(nRanks);
    }

    explicit PackedNetComm(MnnvlPxn)
    {
        constexpr int kSystemId = 1;

        system = static_cast<struct ncclTopoSystem*>(calloc(1, sizeof(struct ncclTopoSystem)));
        if (system == nullptr) abort();
        system->systemId    = kSystemId;
        system->nHosts      = 2;
        system->nRanks      = 2;
        system->inter       = 1;
        system->netGdrLevel = -2;

        system->nodes[GPU].count = 2;
        fillGpu(&system->nodes[GPU].nodes[0], kSystemId, kGpuLocalId, /*dev=*/0, kRank);
        fillGpu(&system->nodes[GPU].nodes[1], kSystemId, kGpuLocalId + 1, /*dev=*/1, kProxyRank);

        // Host 0's NIC first: id == kNetDev, the pre-fix *dev lookup target.
        system->nodes[NET].count = 2;
        fillNet(&system->nodes[NET].nodes[0], /*systemId=*/0, kNetDev, kRemoteBw);
        fillNet(&system->nodes[NET].nodes[1], kSystemId, kNetDev, kNetBw);

        allocTypedPaths(&system->nodes[GPU].nodes[0], NET, 2, PATH_SYS, kRemoteBw);
        allocTypedPaths(&system->nodes[GPU].nodes[1], NET, 2, PATH_SYS, kRemoteBw);
        system->nodes[GPU].nodes[1].paths[NET][1].type = PATH_PXB;
        system->nodes[GPU].nodes[1].paths[NET][1].bw   = kNetBw;
        setPxnPath(&system->nodes[GPU].nodes[0], /*netIndex=*/1, &system->nodes[GPU].nodes[1]);

        allocTypedPaths(&system->nodes[NET].nodes[0], GPU, 2, PATH_SYS, kRemoteBw);
        allocTypedPaths(&system->nodes[NET].nodes[1], GPU, 2, PATH_PXN, kNetBw);
        system->nodes[NET].nodes[1].paths[GPU][1].type = PATH_PXB;

        // CheckGdr (GetPxnRanks) hops to the proxy GPU then calls GetLocalCpu.
        system->nodes[CPU].count = 1;
        system->nodes[CPU].nodes[0].type = CPU;
        allocTypedPaths(&system->nodes[GPU].nodes[0], CPU, 1, PATH_PHB, kNetBw);
        allocTypedPaths(&system->nodes[GPU].nodes[1], CPU, 1, PATH_PHB, kNetBw);
        system->nodes[GPU].nodes[0].paths[CPU][0].count = 2;
        system->nodes[GPU].nodes[1].paths[CPU][0].count = 2;

        bindComm(/*nRanks=*/2);
    }

    ~PackedNetComm()
    {
        if (system) ncclTopoFree(system);
        delete[] peerInfo;
        delete comm;
    }

    PackedNetComm(const PackedNetComm&)            = delete;
    PackedNetComm& operator=(const PackedNetComm&) = delete;

    int packedNetIndex() const
    {
        for (int i = 0; i < system->nodes[NET].count; i++)
        {
            if (NCCL_TOPO_ID_SYSTEM_ID(system->nodes[NET].nodes[i].id) == system->systemId)
                return i;
        }
        return 0;
    }

    int64_t packedNetId() const
    {
        return system->nodes[NET].nodes[packedNetIndex()].id;
    }

private:
    void bindComm(int nRanks)
    {
        // Value-init zeroes POD and constructs rmaState's thread/mutex/cv.
        comm             = new ncclComm();
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
};

const std::unordered_map<std::string, std::string> kPxnLevel1Env = {
    {"NCCL_P2P_PXN_LEVEL", "1"},
    {"NCCL_PXN_DISABLE", "0"},
    {"NCCL_CROSS_NIC", "2"},
};

const std::unordered_map<std::string, std::string> kPxnLevel0Env = {
    {"NCCL_P2P_PXN_LEVEL", "0"},
    {"NCCL_PXN_DISABLE", "0"},
    {"NCCL_CROSS_NIC", "2"},
};

} // namespace

// The lookup contract the PR repaired: a packed NET id is not interchangeable
// with the bare plugin index once systemId != 0 and no host-0 NIC is present.
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
                << "bare netDev must not match a packed NET id when host 0's NIC is absent";
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

// Fused MNNVL: host 0's NIC makes the bare-dev lookup succeed with no
// redirect. Packed netId walks PATH_PXN to kProxyRank.
TEST(P2pPxnLevel1NetId, IntermediateRank_MnnvlPackedIdRedirectsBareHost0DoesNot)
{
    RUN_ISOLATED_TEST(
        "IntermediateRank_MnnvlPackedIdRedirectsBareHost0DoesNot",
        []()
        {
            PackedNetComm fx(PackedNetComm::MnnvlPxn{});
            ASSERT_NE(fx.packedNetId(), static_cast<int64_t>(kNetDev));

            int proxy = -1;
            EXPECT_EQ(ncclTopoGetIntermediateRank(fx.system, kRank, fx.packedNetId(), &proxy),
                      ncclSuccess);
            EXPECT_EQ(proxy, kProxyRank);

            proxy = -1;
            EXPECT_EQ(ncclTopoGetIntermediateRank(fx.system, kRank, kNetDev, &proxy), ncclSuccess)
                << "bare netDev matches host 0's NIC on a fused topology";
            EXPECT_EQ(proxy, kRank);
        });
}

// The production call: ncclTopoGetNetDev on a packed multi-system topology
// with NCCL_P2P_PXN_LEVEL=1 and no host-0 NIC. Pre-fix this returned
// ncclInternalError.
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
            EXPECT_EQ(proxyRank, kRank);
        },
        kPxnLevel1Env);
}

// Fused MNNVL production path. Pre-fix *dev looks up host 0's NIC and
// returns proxyRank == kRank; the fix walks the packed PATH_PXN hop.
TEST(P2pPxnLevel1NetId, GetNetDev_Level1MnnvlHost0NicRedirects)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "GetNetDev_Level1MnnvlHost0NicRedirects",
        []()
        {
            PackedNetComm fx(PackedNetComm::MnnvlPxn{});
            ASSERT_NE(fx.packedNetId(), static_cast<int64_t>(kNetDev));

            int64_t netId     = -1;
            int     netDev    = -1;
            int     proxyRank = -1;
            ASSERT_EQ(ncclTopoGetNetDev(fx.comm, kRank, /*graph=*/nullptr, /*channelId=*/0,
                                        kPeerRank, &netId, &netDev, &proxyRank),
                      ncclSuccess);
            EXPECT_EQ(netId, fx.packedNetId());
            EXPECT_EQ(netDev, kNetDev);
            EXPECT_EQ(proxyRank, kProxyRank);
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
            EXPECT_EQ(proxyRank, kRank);
        },
        kPxnLevel1Env);
}

// Communicator init (init.cc) connects remote PXN proxies via GetPxnRanks.
// PATH_PXB makes every proxyRank == rank, so the list is empty; pin that so
// a silently empty success cannot hide a crash-only assertion.
TEST(P2pPxnLevel1NetId, GetPxnRanks_Level1PackedIdSucceeds)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "GetPxnRanks_Level1PackedIdSucceeds",
        []()
        {
            PackedNetComm fx(/*systemId=*/1, /*nRanks=*/2, PATH_PXB);

            int* ranks  = nullptr;
            int  nranks = -1;
            ASSERT_EQ(ncclTopoGetPxnRanks(fx.comm, &ranks, &nranks), ncclSuccess);
            EXPECT_EQ(nranks, 0);
            EXPECT_EQ(ranks, nullptr);
            free(ranks);
        },
        kPxnLevel1Env);
}

// Same init path on the fused MNNVL shape: GetPxnRanks must report the
// PATH_PXN intermediate. Pre-fix *dev yields proxyRank == kRank and an empty list.
TEST(P2pPxnLevel1NetId, GetPxnRanks_Level1MnnvlHost0NicReturnsProxy)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "GetPxnRanks_Level1MnnvlHost0NicReturnsProxy",
        []()
        {
            PackedNetComm fx(PackedNetComm::MnnvlPxn{});

            int* ranks  = nullptr;
            int  nranks = -1;
            ASSERT_EQ(ncclTopoGetPxnRanks(fx.comm, &ranks, &nranks), ncclSuccess);
            ASSERT_EQ(nranks, 1);
            ASSERT_NE(ranks, nullptr);
            EXPECT_EQ(ranks[0], kProxyRank);
            free(ranks);
        },
        kPxnLevel1Env);
}

// LEVEL=0 skips the PXN override (CROSS_NIC pinned to the default 2). Packed
// ids must still resolve through GetLocalNet without taking the broken branch.
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
            EXPECT_EQ(proxyRank, kRank);
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
