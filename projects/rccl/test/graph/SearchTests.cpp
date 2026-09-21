/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Regression test for the NCCL_CROSS_NIC=0 rail-matching rule fixed in NCCL
// v2.28.7-1 ("large performance issue ... where NCCL cannot find a viable
// ring"), synced into RCCL by the v2.28.9-1 sync.
//
// The rule lives in ncclTopoSearchCheckNet() (src/graph/search.cc), which
// decides whether a candidate NIC is a valid "back-to-NIC" choice during ring
// search. Before the fix, crossNic=0 matched rails by (asic, port) only -- and
// `asic` is the NIC GUID, which is unique per host. So across hosts no NIC ever
// matched and the search could not close a ring. The fix adds a per-NIC `pciId`
// and NCCL_MNNVL_RAIL_PER_HOST: when set, cross-host NICs match by (pciId, port)
// instead, while same-host pairs keep using (asic, port).
//
// These cases pin that crossNic=0 truth table. The test requires net.pciId
// (added by the sync) so it cannot even compile on pre-2.28.7 trees, and the
// MNNVL_RAIL_PER_HOST accept case below fails on the pre-fix (asic-only) logic
// -- so it discriminates fixed-vs-unfixed code.
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
    for(size_t i = 0; i < nics.size(); ++i)
    {
        auto& node     = sys->nodes[NET].nodes[i];
        node.type      = NET;
        node.id        = NCCL_TOPO_ID(nics[i].systemId, nics[i].dev);
        node.net.dev   = nics[i].dev;
        node.net.asic  = nics[i].asic;
        node.net.port  = nics[i].port;
        node.net.pciId = nics[i].pciId;
        node.net.railId = NCCL_TOPO_UNDEF;
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
// Cross host, MNNVL_RAIL_PER_HOST off (default): only (asic, port) is tried.
// With realistic unique-per-NIC GUIDs the asic never matches across hosts, so
// the helper rejects -- this is the "cannot find a viable ring" condition the
// fix targets.
// ---------------------------------------------------------------------------
TEST(SearchCheckNet, CrossHost_MnnvlRailOff_UniqueGuids_Rejects)
{
    RUN_ISOLATED_TEST("CrossHost_MnnvlRailOff_UniqueGuids_Rejects",
                      []()
                      {
                          auto* sys      = makeSystemWithNets({
                              {0, 0, 0xAA01, 1, 0xCAFE},
                              {1, 0, 0xAA02, 1, 0xCAFE}, // unique GUID per host; pciId would match
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
// Cross host, MNNVL_RAIL_PER_HOST on (the fix): cross-host matching falls back
// to (pciId, port). NCCL_PARAM caches the env value, so each case forks.
// ---------------------------------------------------------------------------
TEST(SearchCheckNet, CrossHost_MnnvlRailOn_PciIdAndPortMatch_Accepts)
{
    // The discriminating case: same (pciId, port) but distinct GUIDs across
    // hosts. Accept-on-the-fixed-tree; the pre-fix asic-only logic rejects.
    RUN_ISOLATED_TEST_WITH_ENV(
        "CrossHost_MnnvlRailOn_PciIdAndPortMatch_Accepts",
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

TEST(SearchCheckNet, CrossHost_MnnvlRailOn_PciIdMismatch_Rejects)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "CrossHost_MnnvlRailOn_PciIdMismatch_Rejects",
        []()
        {
            auto* sys      = makeSystemWithNets({
                {0, 0, 0xAA01, 1, 0xCAFE},
                {1, 0, 0xAA02, 1, 0xBABE}, // different pciId => different rail
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

TEST(SearchCheckNet, SameHost_MnnvlRailOn_StillUsesAsicMatch_Rejects)
{
    // MNNVL_RAIL_PER_HOST only switches the key for NICs on *different* system
    // ids. Same-host pairs still match on (asic, port), so a same-host pair with
    // matching pciId but different asic is still rejected.
    RUN_ISOLATED_TEST_WITH_ENV(
        "SameHost_MnnvlRailOn_StillUsesAsicMatch_Rejects",
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

} // namespace RcclUnitTesting
