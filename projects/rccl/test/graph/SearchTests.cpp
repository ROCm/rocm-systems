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

// nGpus == nRanks is the single-MNNVL-domain case the knob is scoped to; passing a smaller
// nGpus models a comm whose ranks reach beyond the domain, where the search uses NET. The
// gate also reads nodes[0]'s arch, so every GPU carries one.
ncclTopoSystem* makeSystem(int nGpus, int nRanks, const char* gcn = "gfx1250")
{
    auto* s             = new ncclTopoSystem{};
    s->nodes[GPU].count = nGpus;
    s->nRanks           = nRanks;
    for(int i = 0; i < nGpus; ++i)
    {
        auto& node = s->nodes[GPU].nodes[i];
        node.type  = GPU;
        snprintf(node.gpu.gcn, GCN_ARCH_NAME_LEN, "%s", gcn);
    }
    return s;
}

} // namespace

// Default off: an unset knob must leave ring search exactly as it is today.
TEST(SearchPcieOrderedRing, Unset_RingNotForced)
{
    RUN_ISOLATED_TEST("Unset_RingNotForced",
                      []()
                      {
                          auto* s = makeSystem(8, 8);
                          auto* g = makeGraph(NCCL_TOPO_PATTERN_RING);
                          EXPECT_FALSE(ringPcieOrdered(s, g));
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

// A domain spanning several hosts is still one domain: all 8 ranks are GPUs here, so the knob
// applies even though the topology was fused from two hosts.
TEST(SearchPcieOrderedRing, Enabled_MultiHostSingleDomainForced)
{
    RUN_ISOLATED_TEST_WITH_ENV("Enabled_MultiHostSingleDomainForced",
                               []()
                               {
                                   auto* s = makeSystem(8, 8);
                                   s->nHosts = 2;
                                   auto* g   = makeGraph(NCCL_TOPO_PATTERN_RING);
                                   EXPECT_TRUE(ringPcieOrdered(s, g));
                                   delete s;
                                   delete g;
                               },
                               {{"RCCL_PCIE_ORDERED_RING", "1"}});
}

} // namespace RcclUnitTesting
