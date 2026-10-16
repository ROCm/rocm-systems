/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// The NCCL_CROSS_NIC=0 rail-matching rule in ncclTopoSearchCheckNet()
// (src/graph/search.cc), which decides whether a candidate NIC is a valid
// "back-to-NIC" choice during ring search. Two NICs match only when they share a
// rail and a plane. When the net plugin reports none, NCCL generates them: each
// host numbers its distinct NIC ASICs in order of appearance, and the plane is
// the port. Generated rails line up across hosts only with
// NCCL_MNNVL_RAIL_PER_HOST; otherwise a NIC on another host never matches.
//
// ncclTopoSearchCheckNet() is internal: it has external linkage only in debug
// builds (this file is built into rccl-UnitTestsFixturesDebug), so we just
// forward-declare it. Each body runs via RUN_ISOLATED_TEST(_WITH_ENV) because
// NCCL_PARAM caches its env value in a function-local static that survives
// fork(); forking keeps the parent's ncclParamMnnvlRailPerHost() cache pristine.

#include "graph.h"
#include "graph/topo.h"
#include "gtest/gtest.h"

#include "../common/ProcessIsolatedTestRunner.hpp"

#include <cstdint>
#include <map>
#include <vector>

// Pull the hipified search.cc directly into this translation unit so we can
// exercise its internal (Release: hidden-visibility) symbols without relying on
// external linkage from librccl.so. SEARCH_CC_PATH is defined by test/CMakeLists.txt
// as the absolute path to the hipify-generated copy of src/graph/search.cc.
#include SEARCH_CC_PATH

namespace RcclUnitTesting
{
namespace
{

struct NetSpec
{
    int      systemId;
    int      dev;
    uint64_t asic;
    int      port;
    uint64_t pciId;
};

// Heap-allocate a zero-initialised ncclTopoSystem and populate only its NET
// nodes. ncclTopoSearchCheckNet() reads nothing else, so GPUs/paths/links are
// left empty.
ncclTopoSystem* makeSystemWithNets(const std::vector<NetSpec>& nics)
{
    auto* sys             = new ncclTopoSystem{};
    sys->nodes[NET].count = static_cast<int>(nics.size());
    // Rails/planes as NCCL generates them when the plugin reports none: each host numbers its
    // distinct NIC ASICs in order of appearance, and the plane comes from the port.
    std::map<int, std::map<uint64_t, int>> railOfAsic;
    for(size_t i = 0; i < nics.size(); ++i)
    {
        auto& node     = sys->nodes[NET].nodes[i];
        node.type      = NET;
        node.id        = NCCL_TOPO_ID(nics[i].systemId, nics[i].dev);
        node.net.dev   = nics[i].dev;
        node.net.asic  = nics[i].asic;
        node.net.port  = nics[i].port;
        node.net.pciId = nics[i].pciId;
        auto& rails    = railOfAsic[nics[i].systemId];
        node.net.railId  = NCCL_TOPO_UNDEF_BIT | rails.emplace(nics[i].asic, static_cast<int>(rails.size())).first->second;
        node.net.planeId = NCCL_TOPO_UNDEF_BIT | nics[i].port;
    }
    return sys;
}

ncclTopoGraph* makeRingCrossNic0()
{
    auto* g     = new ncclTopoGraph{};
    g->pattern  = NCCL_TOPO_PATTERN_RING;
    g->crossNic = 0;
    return g;
}

} // namespace

// ---------------------------------------------------------------------------
// Same host: the legacy (asic, port) match is unchanged by the fix.
// ---------------------------------------------------------------------------
TEST(SearchCheckNet, SameHost_AsicAndPortMatch_Accepts)
{
    RUN_ISOLATED_TEST(
        "SameHost_AsicAndPortMatch_Accepts",
        []()
        {
            auto* sys      = makeSystemWithNets({
                {0, 0, 0xAA, 1, 0xCC},
                {0, 1, 0xAA, 1, 0xDD}, // same asic + port (twin-port NIC), distinct dev
            });
            auto* g        = makeRingCrossNic0();
            auto* startNet = &sys->nodes[NET].nodes[0];
            EXPECT_TRUE(ncclTopoSearchCheckNet(sys, g, startNet, /*n=*/1, /*step=*/0));
            delete g;
            delete sys;
        });
}

TEST(SearchCheckNet, SameHost_DifferentAsic_Rejects)
{
    RUN_ISOLATED_TEST("SameHost_DifferentAsic_Rejects",
                      []()
                      {
                          auto* sys      = makeSystemWithNets({
                              {0, 0, 0xAA, 1, 0xCC},
                              {0, 1, 0xBB, 1, 0xCC},
                          });
                          auto* g        = makeRingCrossNic0();
                          auto* startNet = &sys->nodes[NET].nodes[0];
                          EXPECT_FALSE(
                              ncclTopoSearchCheckNet(sys, g, startNet, /*n=*/1, /*step=*/0));
                          delete g;
                          delete sys;
                      });
}

// ---------------------------------------------------------------------------
// Cross host, MNNVL_RAIL_PER_HOST off: generated rails never match across hosts.
// ---------------------------------------------------------------------------
TEST(SearchCheckNet, CrossHost_MnnvlRailOff_UniqueGuids_Rejects)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "CrossHost_MnnvlRailOff_UniqueGuids_Rejects",
        []()
        {
            auto* sys      = makeSystemWithNets({
                {0, 0, 0xAA01, 1, 0xCAFE},
                {1, 0, 0xAA02, 1, 0xCAFE}, // same generated rail index, different host
            });
            auto* g        = makeRingCrossNic0();
            auto* startNet = &sys->nodes[NET].nodes[0];
            EXPECT_FALSE(ncclTopoSearchCheckNet(sys, g, startNet, /*n=*/1, /*step=*/0));
            delete g;
            delete sys;
        },
        {{"NCCL_MNNVL_RAIL_PER_HOST", "0"}});
}

// ---------------------------------------------------------------------------
// Cross host, MNNVL_RAIL_PER_HOST on: generated rails match by index across
// hosts. NCCL_PARAM caches the env value, so each case forks.
// ---------------------------------------------------------------------------
TEST(SearchCheckNet, CrossHost_MnnvlRailOn_SameRailAndPort_Accepts)
{
    // Distinct GUIDs, but each is its host's first NIC, so both are rail 0.
    RUN_ISOLATED_TEST_WITH_ENV(
        "CrossHost_MnnvlRailOn_SameRailAndPort_Accepts",
        []()
        {
            auto* sys      = makeSystemWithNets({
                {0, 0, 0xAA01, 1, 0xCAFE},
                {1, 0, 0xAA02, 1, 0xCAFE},
            }
                    );
            auto* g        = makeRingCrossNic0();
            auto* startNet = &sys->nodes[NET].nodes[0];
            EXPECT_TRUE(ncclTopoSearchCheckNet(sys, g, startNet, /*n=*/1, /*step=*/0));
            delete g;
            delete sys;
    },
        {{"NCCL_MNNVL_RAIL_PER_HOST", "1"}});
}

TEST(SearchCheckNet, CrossHost_MnnvlRailOn_DifferentRailIndex_Rejects)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "CrossHost_MnnvlRailOn_DifferentRailIndex_Rejects",
        []()
        {
            auto* sys      = makeSystemWithNets({
                {0, 0, 0xAA01, 1, 0xCAFE},
                {1, 0, 0xBB01, 1, 0xCAFE},
                {1, 1, 0xBB02, 1, 0xBABE}, // this host's second NIC => rail 1, not rail 0
            }
                    );
            auto* g        = makeRingCrossNic0();
            auto* startNet = &sys->nodes[NET].nodes[0];
            EXPECT_FALSE(ncclTopoSearchCheckNet(sys, g, startNet, /*n=*/2, /*step=*/0));
            delete g;
            delete sys;
    },
        {{"NCCL_MNNVL_RAIL_PER_HOST", "1"}});
}

TEST(SearchCheckNet, CrossHost_MnnvlRailOn_PortMismatch_Rejects)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "CrossHost_MnnvlRailOn_PortMismatch_Rejects",
        []()
        {
            auto* sys      = makeSystemWithNets({
                {0, 0, 0xAA01, 1, 0xCAFE},
                {1, 0, 0xAA02, 2, 0xCAFE}, // matching pciId, different port
            }
                    );
            auto* g        = makeRingCrossNic0();
            auto* startNet = &sys->nodes[NET].nodes[0];
            EXPECT_FALSE(ncclTopoSearchCheckNet(sys, g, startNet, /*n=*/1, /*step=*/0));
            delete g;
            delete sys;
    },
        {{"NCCL_MNNVL_RAIL_PER_HOST", "1"}});
}

TEST(SearchCheckNet, SameHost_MnnvlRailOn_DifferentAsic_Rejects)
{
    // MNNVL_RAIL_PER_HOST only relaxes the cross-host check. Two ASICs on the same
    // host are two rails, so the pair is still rejected.
    RUN_ISOLATED_TEST_WITH_ENV(
        "SameHost_MnnvlRailOn_DifferentAsic_Rejects",
        []()
        {
            auto* sys      = makeSystemWithNets({
                {0, 0, 0xAA, 1, 0xCAFE},
                {0, 1, 0xBB, 1, 0xCAFE}, // same host + pciId, different asic
            }
                    );
            auto* g        = makeRingCrossNic0();
            auto* startNet = &sys->nodes[NET].nodes[0];
            EXPECT_FALSE(ncclTopoSearchCheckNet(sys, g, startNet, /*n=*/1, /*step=*/0));
            delete g;
            delete sys;
    },
        {{"NCCL_MNNVL_RAIL_PER_HOST", "1"}});
}

// ---------------------------------------------------------------------------
// RCCL_PCIE_ORDERED_RING gating (ringPcieOrdered() in src/graph/search.cc).
//
// The knob pins ring search to the PCI order of the topology array by
// suppressing the arbitrary-permutation attempts. It must be opt-in and must
// reach ring patterns only: every other pattern, and CollNet, keeps the stock
// search. These cases cover the gate itself; the ordering it produces is
// checked on hardware by reading back the ring from NCCL_DEBUG_SUBSYS=GRAPH.
//
// rcclParamPcieOrderedRing() caches its env lookup in a function-local static,
// so each case forks via RUN_ISOLATED_TEST(_WITH_ENV) to get a clean cache.
// ---------------------------------------------------------------------------
namespace
{

ncclTopoGraph* makeGraph(int pattern, int collNet = 0)
{
    auto* g    = new ncclTopoGraph{};
    g->pattern = pattern;
    g->collNet = collNet;
    return g;
}

} // namespace

// Default off: an unset knob must leave ring search exactly as it is today.
TEST(SearchPcieOrderedRing, Unset_RingNotForced)
{
    RUN_ISOLATED_TEST("Unset_RingNotForced",
                      []()
                      {
                          auto* g = makeGraph(NCCL_TOPO_PATTERN_RING);
                          EXPECT_FALSE(ringPcieOrdered(g));
                          delete g;
                      });
}

TEST(SearchPcieOrderedRing, RcclPrefix_RingForced)
{
    RUN_ISOLATED_TEST_WITH_ENV("RcclPrefix_RingForced",
                               []()
                               {
                                   auto* g = makeGraph(NCCL_TOPO_PATTERN_RING);
                                   EXPECT_TRUE(ringPcieOrdered(g));
                                   delete g;
                               },
                               {{"RCCL_PCIE_ORDERED_RING", "1"}});
}

// RCCL_PARAM_NCCL_ALIAS also accepts the NCCL_ spelling, which is what callers
// reach for out of habit when passing vars through `mpirun -x`.
TEST(SearchPcieOrderedRing, NcclAliasPrefix_RingForced)
{
    RUN_ISOLATED_TEST_WITH_ENV("NcclAliasPrefix_RingForced",
                               []()
                               {
                                   auto* g = makeGraph(NCCL_TOPO_PATTERN_RING);
                                   EXPECT_TRUE(ringPcieOrdered(g));
                                   delete g;
                               },
                               {{"NCCL_PCIE_ORDERED_RING", "1"}});
}

// Ring-only: trees keep the stock permutation search even when the knob is on.
TEST(SearchPcieOrderedRing, Enabled_TreePatternNotForced)
{
    RUN_ISOLATED_TEST_WITH_ENV("Enabled_TreePatternNotForced",
                               []()
                               {
                                   auto* tree     = makeGraph(NCCL_TOPO_PATTERN_TREE);
                                   auto* balanced = makeGraph(NCCL_TOPO_PATTERN_BALANCED_TREE);
                                   auto* nvls     = makeGraph(NCCL_TOPO_PATTERN_NVLS);
                                   EXPECT_FALSE(ringPcieOrdered(tree));
                                   EXPECT_FALSE(ringPcieOrdered(balanced));
                                   EXPECT_FALSE(ringPcieOrdered(nvls));
                                   delete nvls;
                                   delete balanced;
                                   delete tree;
                               },
                               {{"RCCL_PCIE_ORDERED_RING", "1"}});
}

// A CollNet ring keeps its own head/tail selection; pinning the order would
// fight it, so the gate excludes it.
TEST(SearchPcieOrderedRing, Enabled_CollNetRingNotForced)
{
    RUN_ISOLATED_TEST_WITH_ENV("Enabled_CollNetRingNotForced",
                               []()
                               {
                                   auto* g = makeGraph(NCCL_TOPO_PATTERN_RING, /*collNet=*/1);
                                   EXPECT_FALSE(ringPcieOrdered(g));
                                   delete g;
                               },
                               {{"RCCL_PCIE_ORDERED_RING", "1"}});
}

// Explicit 0 is honoured as off, so the knob can be force-disabled downstream.
TEST(SearchPcieOrderedRing, ExplicitZero_RingNotForced)
{
    RUN_ISOLATED_TEST_WITH_ENV("ExplicitZero_RingNotForced",
                               []()
                               {
                                   auto* g = makeGraph(NCCL_TOPO_PATTERN_RING);
                                   EXPECT_FALSE(ringPcieOrdered(g));
                                   delete g;
                               },
                               {{"RCCL_PCIE_ORDERED_RING", "0"}});
}

// ---------------------------------------------------------------------------
// Review fixes: NET entry-candidate selection, strict-mode failure, and the
// explicit-override conflict policy.
// ---------------------------------------------------------------------------
namespace
{

// One DPX partition. rank is independent of position so a test can express a
// permuted rank->device mapping.
struct DpxPart
{
    int     systemId;
    int64_t busId;
    int     mloPart;
    int     rank;
    int     dev;
};

// Minimum ncclTopoCompute() reads: GPU nodes carrying rank/mloPart/gcn, each
// pointing at a per-partition DEV node whose id encodes (systemId, busId,
// mloPart). Every path table is PATH_DIS, so the DFS search traverses nothing
// and finds no ring -- which is exactly the strict-mode failure case.
ncclTopoSystem* makeDpxSystem(const std::vector<DpxPart>& parts, const char* gcn = "gfx1250")
{
    auto* sys                   = new ncclTopoSystem{};
    sys->nRanks                 = static_cast<int>(parts.size());
    sys->skipPresetTopoMatching = true;
    sys->totalBw                = 100.0f;
    sys->nodes[GPU].count       = static_cast<int>(parts.size());

    for(size_t i = 0; i < parts.size(); ++i)
    {
        const auto&   p     = parts[i];
        const int64_t local = NCCL_TOPO_MLOPART_BUSID(p.busId, p.mloPart);
        const int64_t devId = NCCL_TOPO_ID(p.systemId, local);

        ncclTopoNode* dev = nullptr;
        for(int d = 0; d < sys->nodes[DEV].count; ++d)
            if(sys->nodes[DEV].nodes[d].id == devId) { dev = &sys->nodes[DEV].nodes[d]; break; }
        if(dev == nullptr)
        {
            dev                  = &sys->nodes[DEV].nodes[sys->nodes[DEV].count++];
            dev->type            = DEV;
            dev->id              = devId;
            dev->dev.cudaCompCap = 60;
            dev->dev.dev         = p.dev;
        }

        auto& g           = sys->nodes[GPU].nodes[i];
        g.type            = GPU;
        g.id              = NCCL_TOPO_ID(p.systemId, NCCL_TOPO_GPU_LOCAL_ID(local, 0));
        g.gpu.rank        = p.rank;
        g.gpu.dev         = p.dev;
        g.gpu.mloPart     = p.mloPart;
        g.gpu.cudaCompCap = 60;
        strncpy(g.gpu.gcn, gcn, GCN_ARCH_NAME_LEN - 1);
        g.gpu.parent = dev;
    }

    for(int t = 0; t < NCCL_TOPO_NODE_TYPES; ++t)
        for(int n = 0; n < sys->nodes[t].count; ++n)
            for(int pt = 0; pt < NCCL_TOPO_NODE_TYPES; ++pt)
            {
                const int cnt = sys->nodes[pt].count;
                if(cnt == 0) continue;
                auto* paths = static_cast<ncclTopoLinkList*>(calloc(cnt, sizeof(ncclTopoLinkList)));
                for(int i = 0; i < cnt; ++i) paths[i].type = PATH_DIS;
                sys->nodes[t].nodes[n].paths[pt] = paths;
            }
    return sys;
}

void freeDpxSystem(ncclTopoSystem* sys)
{
    for(int t = 0; t < NCCL_TOPO_NODE_TYPES; ++t)
        for(int n = 0; n < sys->nodes[t].count; ++n)
            for(int pt = 0; pt < NCCL_TOPO_NODE_TYPES; ++pt) free(sys->nodes[t].nodes[n].paths[pt]);
    delete sys;
}

ncclTopoGraph* makeRingGraph()
{
    auto* g        = new ncclTopoGraph{};
    g->id          = 0;
    g->pattern     = NCCL_TOPO_PATTERN_RING;
    g->collNet     = 0;
    g->minChannels = 1;
    g->maxChannels = MAXCHANNELS;
    return g;
}

std::vector<DpxPart> fourDies(int systemId = 0)
{
    std::vector<DpxPart> parts;
    for(int i = 0; i < 8; ++i) parts.push_back({systemId, 0x1000 * (i / 2 + 1), i % 2, i, i});
    return parts;
}

// One NIC and nGpus GPUs, every NET->GPU path disconnected by default. Tests
// then connect individual GPUs to express a candidate set.
ncclTopoSystem* makeNetSystem(int nGpus)
{
    auto* sys             = new ncclTopoSystem{};
    sys->nRanks           = nGpus;
    sys->nodes[GPU].count = nGpus;
    sys->nodes[NET].count = 1;

    for(int i = 0; i < nGpus; ++i)
    {
        auto& g           = sys->nodes[GPU].nodes[i];
        g.type            = GPU;
        g.id              = NCCL_TOPO_ID(0, i);
        g.gpu.rank        = i;
        g.gpu.dev         = i;
        g.gpu.gdrSupport  = 1;
        g.gpu.cudaCompCap = 60;
    }
    auto& net          = sys->nodes[NET].nodes[0];
    net.type           = NET;
    net.id             = NCCL_TOPO_ID(0, 0);
    net.net.dev        = 0;
    net.net.gdrSupport = 0; // keeps ncclTopoCheckGdr()'s answer uniformly false
    net.net.localGpu   = -1;

    // calloc leaves count at 0, which is the trap finding 2 is about.
    auto* paths = static_cast<ncclTopoLinkList*>(calloc(nGpus, sizeof(ncclTopoLinkList)));
    for(int i = 0; i < nGpus; ++i) paths[i].type = PATH_DIS;
    net.paths[GPU] = paths;
    return sys;
}

void connectNetToGpu(ncclTopoSystem* sys, int gpu, int hops, int type = PATH_PXB)
{
    auto* paths      = sys->nodes[NET].nodes[0].paths[GPU];
    paths[gpu].type  = type;
    paths[gpu].count = hops;
}

void freeNetSystem(ncclTopoSystem* sys)
{
    free(sys->nodes[NET].nodes[0].paths[GPU]);
    delete sys;
}

} // namespace

// Finding 2: an unreachable GPU is calloc'd to count 0, so a selection that
// scores before rejecting PATH_DIS picks it over every reachable candidate.
// Stock hid this by discarding the result; honouring NIC affinity makes it live.
TEST(SearchPcieOrderedRing, NetClosestGpu_RejectsDisconnectedCandidates)
{
    RUN_ISOLATED_TEST("NetClosestGpu_RejectsDisconnectedCandidates",
                      []()
                      {
                          auto* sys = makeNetSystem(4);
                          // GPU 0 stays PATH_DIS with count 0; GPU 2 is reachable in 3 hops.
                          connectNetToGpu(sys, 2, /*hops=*/3);
                          int f = -99;
                          EXPECT_EQ(ncclTopoNetClosestGpu(sys, &sys->nodes[NET].nodes[0], &f), ncclSuccess);
                          EXPECT_EQ(f, 2);
                          freeNetSystem(sys);
                      });
}

// Nearest reachable wins among several reachable candidates.
TEST(SearchPcieOrderedRing, NetClosestGpu_PicksFewestHops)
{
    RUN_ISOLATED_TEST("NetClosestGpu_PicksFewestHops",
                      []()
                      {
                          auto* sys = makeNetSystem(4);
                          connectNetToGpu(sys, 1, /*hops=*/5);
                          connectNetToGpu(sys, 3, /*hops=*/2);
                          int f = -99;
                          EXPECT_EQ(ncclTopoNetClosestGpu(sys, &sys->nodes[NET].nodes[0], &f), ncclSuccess);
                          EXPECT_EQ(f, 3);
                          freeNetSystem(sys);
                      });
}

// No GPU reaches this NIC: report -1 so the caller skips the forced attempt
// rather than entering the ring at a GPU the NIC cannot talk to.
TEST(SearchPcieOrderedRing, NetClosestGpu_AllDisconnectedYieldsNoCandidate)
{
    RUN_ISOLATED_TEST("NetClosestGpu_AllDisconnectedYieldsNoCandidate",
                      []()
                      {
                          auto* sys = makeNetSystem(4);
                          int   f   = -99;
                          EXPECT_EQ(ncclTopoNetClosestGpu(sys, &sys->nodes[NET].nodes[0], &f), ncclSuccess);
                          EXPECT_EQ(f, -1);
                          freeNetSystem(sys);
                      });
}

// Finding 3: with the knob on and no validated ring, report the failure instead
// of returning success on the unvalidated one-channel fallback.
TEST(SearchPcieOrderedRing, Enabled_NoValidatedRingReturnsError)
{
    RUN_ISOLATED_TEST_WITH_ENV("Enabled_NoValidatedRingReturnsError",
                               []()
                               {
                                   auto* sys = makeDpxSystem(fourDies());
                                   auto* g   = makeRingGraph();
                                   EXPECT_EQ(ncclTopoCompute(sys, g), ncclInvalidUsage);
                                   delete g;
                                   freeDpxSystem(sys);
                               },
                               {{"RCCL_PCIE_ORDERED_RING", "1"}});
}

// ...and with the knob off the legacy fallback is untouched: success, one
// channel, ranks in topology order.
TEST(SearchPcieOrderedRing, Disabled_NoValidatedRingKeepsLegacyFallback)
{
    RUN_ISOLATED_TEST("Disabled_NoValidatedRingKeepsLegacyFallback",
                      []()
                      {
                          auto* sys = makeDpxSystem(fourDies());
                          auto* g   = makeRingGraph();
                          EXPECT_EQ(ncclTopoCompute(sys, g), ncclSuccess);
                          EXPECT_EQ(g->nChannels, 1);
                          EXPECT_EQ(std::vector<int>(g->intra, g->intra + 8),
                                    (std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7}));
                          delete g;
                          freeDpxSystem(sys);
                      });
}

// Finding 4: an explicit NCCL_RINGS order outranks the knob -- the search must
// honour the user's ring and must not fail the way the strict path does.
TEST(SearchPcieOrderedRing, Enabled_ExplicitNcclRingsWins)
{
    RUN_ISOLATED_TEST_WITH_ENV("Enabled_ExplicitNcclRingsWins",
                               []()
                               {
                                   auto* sys = makeDpxSystem(fourDies());
                                   auto* g   = makeRingGraph();
                                   EXPECT_EQ(ncclTopoCompute(sys, g), ncclSuccess);
                                   EXPECT_GT(g->nChannels, 0);
                                   delete g;
                                   freeDpxSystem(sys);
                               },
                               {{"RCCL_PCIE_ORDERED_RING", "1"}, {"NCCL_RINGS", "0 1 2 3 4 5 6 7"}});
}

} // namespace RcclUnitTesting
