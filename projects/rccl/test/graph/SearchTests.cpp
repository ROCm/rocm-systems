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

// nGpus == nRanks on a single host is the case the knob is scoped to; passing a smaller
// nGpus models a comm whose ranks reach beyond the topology, where the search uses NET. The
// gate also reads nodes[0]'s arch, so every GPU carries one.
ncclTopoSystem* makeSystem(int nGpus, int nRanks, const char* gcn = "gfx1250")
{
    auto* s             = new ncclTopoSystem{};
    s->nodes[GPU].count = nGpus;
    s->nRanks           = nRanks;
    s->nHosts           = 1;
    for(int i = 0; i < nGpus; ++i)
    {
        auto& node = s->nodes[GPU].nodes[i];
        node.type  = GPU;
        snprintf(node.gpu.gcn, GCN_ARCH_NAME_LEN, "%s", gcn);
    }
    return s;
}

} // namespace

// Default on: an unset knob forces the order for an eligible ring. Guards the default value
// itself -- flipping it back to 0 must fail here.
TEST(SearchPcieOrderedRing, Unset_RingForcedByDefault)
{
    RUN_ISOLATED_TEST("Unset_RingForcedByDefault",
                      []()
                      {
                          auto* s = makeSystem(8, 8);
                          auto* g = makeGraph(NCCL_TOPO_PATTERN_RING);
                          EXPECT_TRUE(ringPcieOrdered(s, g));
                          delete s;
                          delete g;
                      });
}

TEST(SearchPcieOrderedRing, RcclPrefix_RingForced)
{
    RUN_ISOLATED_TEST_WITH_ENV("RcclPrefix_RingForced",
                               []()
                               {
                                   auto* s = makeSystem(8, 8);
                                   auto* g = makeGraph(NCCL_TOPO_PATTERN_RING);
                                   EXPECT_TRUE(ringPcieOrdered(s, g));
                                   delete s;
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
                                   auto* s = makeSystem(8, 8);
                                   auto* g = makeGraph(NCCL_TOPO_PATTERN_RING);
                                   EXPECT_TRUE(ringPcieOrdered(s, g));
                                   delete s;
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
                                   auto* s        = makeSystem(8, 8);
                                   EXPECT_FALSE(ringPcieOrdered(s, tree));
                                   EXPECT_FALSE(ringPcieOrdered(s, balanced));
                                   EXPECT_FALSE(ringPcieOrdered(s, nvls));
                                   delete s;
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
                                   auto* s = makeSystem(8, 8);
                                   auto* g = makeGraph(NCCL_TOPO_PATTERN_RING, /*collNet=*/1);
                                   EXPECT_FALSE(ringPcieOrdered(s, g));
                                   delete s;
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
                                   auto* s = makeSystem(8, 8);
                                   auto* g = makeGraph(NCCL_TOPO_PATTERN_RING);
                                   EXPECT_FALSE(ringPcieOrdered(s, g));
                                   delete s;
                                   delete g;
                               },
                               {{"RCCL_PCIE_ORDERED_RING", "0"}});
}

// Scope limit: the knob only applies inside one MNNVL domain (nNodes == 1), i.e. when every
// rank is a GPU of this topology. A comm reaching past the domain uses the NET search, whose
// entry-point selection this feature deliberately leaves alone.
TEST(SearchPcieOrderedRing, Enabled_MultiDomainNotForced)
{
    RUN_ISOLATED_TEST_WITH_ENV("Enabled_MultiDomainNotForced",
                               []()
                               {
                                   // 4 local GPUs, 8 ranks => ranks live outside this domain.
                                   auto* s = makeSystem(4, 8);
                                   auto* g = makeGraph(NCCL_TOPO_PATTERN_RING);
                                   EXPECT_FALSE(ringPcieOrdered(s, g));
                                   delete s;
                                   delete g;
                               },
                               {{"RCCL_PCIE_ORDERED_RING", "1"}});
}

// Arch limit: other architectures keep the stock permutation search entirely.
TEST(SearchPcieOrderedRing, Enabled_OtherArchNotForced)
{
    RUN_ISOLATED_TEST_WITH_ENV("Enabled_OtherArchNotForced",
                               []()
                               {
                                   auto* s = makeSystem(8, 8, "gfx942");
                                   auto* g = makeGraph(NCCL_TOPO_PATTERN_RING);
                                   EXPECT_FALSE(ringPcieOrdered(s, g));
                                   delete s;
                                   delete g;
                               },
                               {{"RCCL_PCIE_ORDERED_RING", "1"}});
}

// A topology fused from several hosts (an MNNVL clique) is out of scope even though all its
// ranks are GPUs: the clique peers carry PATH_DIS between hosts, so the search cannot route a
// ring in any order and the stock fallback -- which already emits topology order -- handles it.
// Forcing there would buy nothing and previously failed communicator init outright.
TEST(SearchPcieOrderedRing, Enabled_MultiHostNotForced)
{
    RUN_ISOLATED_TEST_WITH_ENV("Enabled_MultiHostNotForced",
                               []()
                               {
                                   auto* s   = makeSystem(8, 8);
                                   s->nHosts = 2;
                                   auto* g   = makeGraph(NCCL_TOPO_PATTERN_RING);
                                   EXPECT_FALSE(ringPcieOrdered(s, g));
                                   delete s;
                                   delete g;
                               },
                               {{"RCCL_PCIE_ORDERED_RING", "1"}});
}

// ---------------------------------------------------------------------------
// Behaviour of the generated graph, not just the gate. These drive
// ncclTopoCompute() end to end so a regression in the ring it produces (or in
// whether it produces one at all) fails here rather than on a 16-rank cluster.
// ---------------------------------------------------------------------------
namespace
{

// Every node the search touches must carry a path table for every node type, or the
// traversal dereferences a null table and segfaults. Default everything to unreachable;
// callers then open up only the edges they want.
void allocPaths(ncclTopoNode* node)
{
    for(int t = 0; t < NCCL_TOPO_NODE_TYPES; ++t)
    {
        node->paths[t] = static_cast<ncclTopoLinkList*>(calloc(NCCL_TOPO_MAX_NODES, sizeof(ncclTopoLinkList)));
        for(int j = 0; j < NCCL_TOPO_MAX_NODES; ++j) node->paths[t][j].type = PATH_DIS;
    }
}

// A system whose GPU-to-GPU paths are spelled out explicitly, so a case can model an
// MNNVL clique: fused into one topology, but with no topology edge between hosts.
// hostsSplit == 0 puts every GPU on one host; otherwise GPUs [0,hostsSplit) are host 0
// and the rest host 1, and every cross-host pair is PATH_DIS.
ncclTopoSystem* makePathSystem(int nGpus, int hostsSplit, const char* gcn = "gfx1250")
{
    auto* s                   = new ncclTopoSystem{};
    s->nodes[GPU].count       = nGpus;
    s->nRanks                 = nGpus;
    s->nHosts                 = hostsSplit ? 2 : 1;
    s->skipPresetTopoMatching = true; // gfx1250 bypasses preset matching
    // ncclTopoGetCompCap() errors out on a system with no DEV nodes (topo.cc).
    s->nodes[DEV].count = nGpus;
    for(int i = 0; i < nGpus; ++i)
    {
        auto& d           = s->nodes[DEV].nodes[i];
        d.type            = DEV;
        d.id              = i;
        d.dev.dev         = i;
        d.dev.cudaCompCap = 90;
        allocPaths(&d);
    }
    for(int i = 0; i < nGpus; ++i)
    {
        auto& node           = s->nodes[GPU].nodes[i];
        node.type            = GPU;
        node.id              = i;
        node.gpu.dev         = i;
        node.gpu.rank        = i;
        node.gpu.cudaCompCap = 90;
        snprintf(node.gpu.gcn, GCN_ARCH_NAME_LEN, "%s", gcn);
        node.gpu.parent = &s->nodes[DEV].nodes[i]; // gpuPciBw() dereferences this
        allocPaths(&node);
        // No NICs in this model. A null NET table is also what a remote MNNVL GPU carries,
        // and search.cc keys on exactly that ("Skip remote GPUs (MNNVL): paths[NET] is null").
        free(node.paths[NET]);
        node.paths[NET] = nullptr;
    }
    // GPU-to-GPU reachability.
    for(int i = 0; i < nGpus; ++i)
    {
        for(int j = 0; j < nGpus; ++j)
        {
            bool sameHost = (hostsSplit == 0) || ((i < hostsSplit) == (j < hostsSplit));
            auto& p       = s->nodes[GPU].nodes[i].paths[GPU][j];
            p.type        = (i == j) ? PATH_LOC : (sameHost ? PATH_NVL : PATH_DIS);
            p.bw          = (i == j) ? 0 : (sameHost ? 100.0f : 0);
            // count stays 0: followPath() then has no hops to walk, so a reachable edge
            // needs no link list. A nonzero count with a null list segfaults.
            p.count = 0;
        }
    }
    return s;
}

void freePathSystem(ncclTopoSystem* s)
{
    for(int nt = 0; nt < NCCL_TOPO_NODE_TYPES; ++nt)
        for(int i = 0; i < s->nodes[nt].count; ++i)
            for(int t = 0; t < NCCL_TOPO_NODE_TYPES; ++t) free(s->nodes[nt].nodes[i].paths[t]);
    delete s;
}

ncclTopoGraph* makeRingGraph()
{
    auto* g          = new ncclTopoGraph{};
    g->id            = 0;
    g->pattern       = NCCL_TOPO_PATTERN_RING;
    g->collNet       = 0;
    g->minChannels   = 1;
    g->maxChannels   = 8;
    return g;
}

} // namespace

// Single host, all GPUs mutually reachable: the forced ring must be built, and it must be
// the topology-array order 0..n-1 -- not merely "some ring".
TEST(SearchPcieOrderedRing, SingleHost_ProducesTopologyOrderedRing)
{
    RUN_ISOLATED_TEST_WITH_ENV("SingleHost_ProducesTopologyOrderedRing",
                               []()
                               {
                                   auto* s = makePathSystem(8, /*hostsSplit=*/0);
                                   auto* g = makeRingGraph();
                                   EXPECT_EQ(ncclTopoCompute(s, g), ncclSuccess);
                                   ASSERT_GT(g->nChannels, 0);
                                   for(int i = 0; i < 8; ++i) EXPECT_EQ(g->intra[i], i) << "ring position " << i;
                                   delete g;
                                   freePathSystem(s);
                               },
                               {{"RCCL_PCIE_ORDERED_RING", "1"}});
}

// The regression: an MNNVL clique fused from two hosts has PATH_DIS between them, so the
// graph search cannot route a ring. Stock falls back to topology order and succeeds; the
// forced path must not turn that into a communicator-init failure.
TEST(SearchPcieOrderedRing, MultiHostPathDis_StillInitialises)
{
    RUN_ISOLATED_TEST_WITH_ENV("MultiHostPathDis_StillInitialises",
                               []()
                               {
                                   auto* s = makePathSystem(16, /*hostsSplit=*/8);
                                   auto* g = makeRingGraph();
                                   EXPECT_EQ(ncclTopoCompute(s, g), ncclSuccess);
                                   EXPECT_GT(g->nChannels, 0);
                                   delete g;
                                   freePathSystem(s);
                               },
                               {{"RCCL_PCIE_ORDERED_RING", "1"}});
}

// The same topology with the knob off is the behaviour we must not change.
TEST(SearchPcieOrderedRing, MultiHostPathDis_DisabledUnchanged)
{
    RUN_ISOLATED_TEST("MultiHostPathDis_DisabledUnchanged",
                      []()
                      {
                          auto* s = makePathSystem(16, /*hostsSplit=*/8);
                          auto* g = makeRingGraph();
                          EXPECT_EQ(ncclTopoCompute(s, g), ncclSuccess);
                          EXPECT_GT(g->nChannels, 0);
                          delete g;
                          freePathSystem(s);
                      });
}

// In scope (single host) but genuinely unroutable: one GPU is cut off from the rest, so no
// ring exists in any order. Here the constraint really was unsatisfiable, so the forced path
// must say so rather than silently reporting an unvalidated fallback as success.
TEST(SearchPcieOrderedRing, SingleHost_UnroutableFailsPredictably)
{
    RUN_ISOLATED_TEST_WITH_ENV("SingleHost_UnroutableFailsPredictably",
                               []()
                               {
                                   auto* s = makePathSystem(8, /*hostsSplit=*/0);
                                   // Sever GPU 7 in both directions.
                                   for(int j = 0; j < 8; ++j)
                                   {
                                       if(j == 7) continue;
                                       s->nodes[GPU].nodes[7].paths[GPU][j].type = PATH_DIS;
                                       s->nodes[GPU].nodes[j].paths[GPU][7].type = PATH_DIS;
                                   }
                                   auto* g = makeRingGraph();
                                   EXPECT_EQ(ncclTopoCompute(s, g), ncclInvalidUsage);
                                   delete g;
                                   freePathSystem(s);
                               },
                               {{"RCCL_PCIE_ORDERED_RING", "2"}});
}

// Same unroutable topology, knob left at its default: must degrade to the stock fallback
// instead of failing init. Now that the default is on, hard-failing here would break
// topologies that worked before the default flipped.
TEST(SearchPcieOrderedRing, UnroutableUnsetDegradesInsteadOfFailing)
{
    RUN_ISOLATED_TEST("UnroutableUnsetDegradesInsteadOfFailing",
                      []()
                      {
                          auto* s = makePathSystem(8, /*hostsSplit=*/0);
                          // Sever GPU 7 in both directions, exactly as the explicit-request case.
                          for(int j = 0; j < 8; ++j)
                          {
                              if(j == 7) continue;
                              s->nodes[GPU].nodes[7].paths[GPU][j].type = PATH_DIS;
                              s->nodes[GPU].nodes[j].paths[GPU][7].type = PATH_DIS;
                          }
                          auto* g = makeRingGraph();
                          EXPECT_EQ(ncclTopoCompute(s, g), ncclSuccess);
                          delete g;
                          freePathSystem(s);
                      });
}

// Explicit user ordering outranks the knob: NCCL_RINGS must still decide the ring, and the
// run must succeed rather than trip the forced-path failure.
TEST(SearchPcieOrderedRing, ExplicitNcclRingsWinsOverForcedOrder)
{
    RUN_ISOLATED_TEST_WITH_ENV("ExplicitNcclRingsWinsOverForcedOrder",
                               []()
                               {
                                   auto* s = makePathSystem(4, /*hostsSplit=*/0);
                                   auto* g = makeRingGraph();
                                   EXPECT_EQ(ncclTopoCompute(s, g), ncclSuccess);
                                   ASSERT_GT(g->nChannels, 0);
                                   // The explicit ring, not the topology order 0 1 2 3.
                                   EXPECT_EQ(g->intra[0], 0);
                                   EXPECT_EQ(g->intra[1], 2);
                                   EXPECT_EQ(g->intra[2], 1);
                                   EXPECT_EQ(g->intra[3], 3);
                                   delete g;
                                   freePathSystem(s);
                               },
                               {{"RCCL_PCIE_ORDERED_RING", "1"}, {"NCCL_RINGS", "0 2 1 3"}});
}

} // namespace RcclUnitTesting
