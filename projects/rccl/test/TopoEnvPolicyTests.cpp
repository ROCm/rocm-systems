/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Topology unit tests for environment-variable-driven behavior, built on
// hand-loaded synthetic topologies (no real GPUs, NICs, or MPI launch). Each
// case runs in an isolated process (RUN_ISOLATED_TEST*) so cached env parses
// and env values are fresh per case. Two groups live here today:
//
//   * NetDevsPolicyTests       -- NCCL_NETDEVS_POLICY on the P2P NET path.
//   * SymmetricMemP2pLevelTests -- symmetric-memory availability vs
//                                  NCCL_P2P_LEVEL
//
// --- NetDevsPolicyTests ---
//
// NCCL 2.29.2 changed send, recv, and all-to-all so they now obey
// NCCL_NETDEVS_POLICY when running over the network. Before this change, these
// operations counted every locally reachable NIC and ignored the policy, so
// settings like MAX:1 did not actually limit which network devices a GPU used.
//
// The policy is applied through ncclTopoGetLocalNet(), and the per-peer
// network channel count for remote ranks is derived from getLocalNetCountByBw()
// (called by ncclTopoGetNchannels() in src/graph/paths.cc). These are exactly
// the functions send/recv/all-to-all use to pick NICs and channels over the
// network. This test ties the policy to that P2P NET channel-selection path so
// the 2.29.2 behavior cannot silently regress.
//
// The test:
//   * Loads a synthetic multi-NIC topology
//     (tools/topo_expl/models/topo_8p6l_5nic.xml, 3 NICs reachable from GPU
//     rank 0) directly from XML, so no real GPUs, NICs, or MPI launch are
//     required.
//   * Runs each case in an isolated process (RUN_ISOLATED_TEST_WITH_ENV) so the
//     std::call_once policy parse is fresh per case and env values do not leak
//     between tests.
//
// The test target is rccl-UnitTestsFixturesDebug (Debug only): the topology
// symbols live inside librccl.so with default-hidden visibility in Release
// builds, so they are only linkable from the Debug fixtures binary -- exactly
// like ParamTests::ncclLoadParam, ArgCheckTests' argcheck helpers, etc.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <set>
#include <string>

#include "comm.h"        // struct ncclComm, struct ncclPeerInfo
#include "graph.h"       // ncclTopoGetLocalNet(), getLocalNetCountByBw(), ncclTopoComputePaths(), ncclTopoFree()
#include "graph/topo.h"  // struct ncclTopoSystem, ncclTopoGetSystemFromXml(), GPU/NET, ncclTopoRankToIndex()
#include "graph/xml.h"   // struct ncclXml, ncclTopoGetXmlFromFile()

#include "common/ProcessIsolatedTestRunner.hpp"

#ifndef RCCL_TEST_SOURCE_DIR
#define RCCL_TEST_SOURCE_DIR "."
#endif

namespace RcclUnitTesting
{

namespace
{

// 8 GPUs / 6 leaf / multi-NIC model. GPU rank 0 lives on NUMA 0 and reaches
// 3 NICs (mlx5_0/1/2), shared with the other 3 GPUs on that NUMA node.
const char* kTopoXmlPath =
    RCCL_TEST_SOURCE_DIR "/../tools/topo_expl/models/topo_8p6l_5nic.xml";

// 8 GPUs / 2 NICs sparse model. The interior GPUs (e.g. rank 1) have no
// directly-attached NIC, so their closest NIC set is the 2 NICs (mlx5_0,
// mlx5_2) reached at equal distance, while only 1 GPU sits closest to that
// first NIC. This isolates AUTO's divide-NICs-across-GPUs behavior with
// localNetCount(2) > localGpuCount(1): ceil(2 / 1) = 2, a non-degenerate value
// (unlike the 5nic model where AUTO collapses to ceil(3 / 4) = 1).
const char* kTopoNicsGtGpusXmlPath =
    RCCL_TEST_SOURCE_DIR "/../tools/topo_expl/models/topo_8p_ts1.xml";

// Rank inside kTopoNicsGtGpusXmlPath whose local NIC pool exceeds the GPUs
// sharing it (see scan: ranks 1..6 all yield nets=2, gpus=1).
constexpr int kAutoRank = 1;

// Rank under test.
constexpr int kRank = 0;

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
// warn=0 + localHostHash 0 matches how tools/topo_expl loads these models
// (the fixtures carry no host_hash attribute).
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

// Build a minimal stand-in communicator for topology-only path computation.
//
// ncclTopoComputePaths() cannot be called with comm == NULL: its PXN branch
// reaches rcclSetPxn(), which dereferences comm unconditionally. We therefore
// hand it a comm that carries just enough state, and nothing that requires real
// hardware:
//   * topo / rank / nRanks -> consumed by rcclSetPxn() (arch + rank threshold)
//     and the PXN walk.
//   * a zeroed peerInfo[]   -> p2pCanConnect() returns 0 early (hasFineGrain == 0)
//     and shmCanConnect() reports the peers as same-host / same-shmDev, so the
//     peer-trim loop marks no GPU pair inaccessible. This keeps the computed
//     paths identical to the comm-less intent while avoiding the NULL deref.
struct MockComm
{
    struct ncclComm*     comm     = nullptr;
    struct ncclPeerInfo* peerInfo = nullptr;

    explicit MockComm(struct ncclTopoSystem* system)
    {
        const int nRanks = system->nodes[GPU].count;
        comm             = new ncclComm();
        memset(comm, 0, sizeof(*comm));
        comm->topo   = system;
        comm->rank   = kRank;
        comm->nRanks = nRanks;
        comm->nNodes = 1;

        // Indexed by GPU rank; size to the topo node bound to stay safe even if
        // ranks are not contiguous.
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

// Load the given fixture (defaults to the 8p6l/5nic model), compute paths, run
// body(system), then free.
void withTopoSystem(const std::function<void(struct ncclTopoSystem*)>& body,
                    const char* topoPath = kTopoXmlPath)
{
    struct ncclTopoSystem* system = nullptr;
    ASSERT_EQ(loadTopoSystem(topoPath, &system), ncclSuccess)
        << "failed to load topo fixture: " << topoPath;
    ASSERT_NE(system, nullptr);

    MockComm mock(system);
    ASSERT_EQ(ncclTopoComputePaths(system, mock.comm), ncclSuccess);

    body(system);

    ncclTopoFree(system);
}

// NICs locally reachable from the GPU (the pool the policy selects from).
int localNetCount(struct ncclTopoSystem* system, int gpu)
{
    int locals[NCCL_TOPO_MAX_NODES];
    int count = 0;
    EXPECT_EQ(ncclTopoGetLocal(system, GPU, gpu, NET, locals, &count, nullptr), ncclSuccess);
    return count;
}

// GPUs sharing the GPU's first local NIC (the AUTO denominator).
int localGpuCount(struct ncclTopoSystem* system, int gpu)
{
    int nets[NCCL_TOPO_MAX_NODES];
    int netCount = 0;
    EXPECT_EQ(ncclTopoGetLocal(system, GPU, gpu, NET, nets, &netCount, nullptr), ncclSuccess);
    if(netCount == 0) return 0;

    int gpus[NCCL_TOPO_MAX_NODES];
    int gpuCount = 0;
    EXPECT_EQ(ncclTopoGetLocal(system, NET, nets[0], GPU, gpus, &gpuCount, nullptr), ncclSuccess);
    return gpuCount;
}

// Distinct NICs ncclTopoGetLocalNet() hands out as the channelId rotates --
// this equals netsPerGpu, the quantity NCCL_NETDEVS_POLICY drives.
int distinctNetsAcrossChannels(struct ncclTopoSystem* system, int nets, int rank = kRank)
{
    std::set<int> devs;
    int           channels = (nets > 0 ? nets : 1) * 4;
    for(int c = 0; c < channels; c++)
    {
        int     dev   = -1;
        int64_t netId = 0;
        EXPECT_EQ(ncclTopoGetLocalNet(system, rank, c, &netId, &dev), ncclSuccess);
        devs.insert(dev);
    }
    return static_cast<int>(devs.size());
}

inline int divUp(int a, int b) { return b > 0 ? (a + b - 1) / b : 0; }

// AUTO selects ceil(localNetCount / localGpuCount) NICs. Default-unset and all
// malformed/out-of-range values fall back to AUTO, so they share this check.
void expectAutoSelection(struct ncclTopoSystem* system)
{
    int gpu = -1;
    ASSERT_EQ(ncclTopoRankToIndex(system, kRank, &gpu, /*showWarn=*/true), ncclSuccess);
    int nets = localNetCount(system, gpu);
    int gpus = localGpuCount(system, gpu);
    ASSERT_GE(nets, 2);
    ASSERT_GE(gpus, 1);
    EXPECT_EQ(distinctNetsAcrossChannels(system, nets), divUp(nets, gpus));
}

constexpr int kXgmiPeerRank = 1;

void checkXgmiPairP2p(int* p2pOut, int* cudaP2pOut)
{
    *p2pOut     = -1;
    *cudaP2pOut = -1;

    struct ncclTopoSystem* system = nullptr;
    ASSERT_EQ(loadTopoSystem(kTopoXmlPath, &system), ncclSuccess)
        << "failed to load topo fixture: " << kTopoXmlPath;
    ASSERT_NE(system, nullptr);

    MockComm mock(system);
    ASSERT_EQ(ncclTopoComputePaths(system, mock.comm), ncclSuccess);

    int p2p = -1, cudaP2p = -1, intermediateRank = -1;
    ASSERT_EQ(ncclTopoCheckP2p(mock.comm, system, kRank, kXgmiPeerRank, &p2p,
                               /*read=*/nullptr, &intermediateRank, &cudaP2p),
              ncclSuccess);

    *p2pOut     = p2p;
    *cudaP2pOut = cudaP2p;

    ncclTopoFree(system);
}

}  // namespace

// NCCL_NETDEVS_POLICY=MAX:1 -- the policy must cap network device usage to a
// single NIC: getLocalNetCountByBw() returns 1, and ncclTopoGetLocalNet()
// returns the same NIC for channel 0 and channel 1.
TEST(NetDevsPolicyTests, Max1_LimitsGetLocalNetCountByBw)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "Max1_LimitsGetLocalNetCountByBw",
        []()
        {
            withTopoSystem(
                [](struct ncclTopoSystem* system)
                {
                    int gpu = -1;
                    ASSERT_EQ(ncclTopoRankToIndex(system, kRank, &gpu, /*showWarn=*/true),
                              ncclSuccess);

                    int   count = -1;
                    float bw    = 0.0f;
                    ASSERT_EQ(getLocalNetCountByBw(system, gpu, &count, &bw), ncclSuccess);
                    EXPECT_EQ(count, 1);

                    int     dev0 = -1, dev1 = -1;
                    int64_t id0 = 0, id1 = 0;
                    ASSERT_EQ(ncclTopoGetLocalNet(system, kRank, /*channelId=*/0, &id0, &dev0),
                              ncclSuccess);
                    ASSERT_EQ(ncclTopoGetLocalNet(system, kRank, /*channelId=*/1, &id1, &dev1),
                              ncclSuccess);
                    EXPECT_EQ(dev0, dev1);
                });
        },
        {{"NCCL_NETDEVS_POLICY", "MAX:1"}});
}

// NCCL_NETDEVS_POLICY=ALL -- the policy must use every locally reachable NIC:
// ncclTopoGetLocalNet() rotates across all reachable NICs (netsPerGpu ==
// localNetCount), so distinct devices seen across channels equals the pool and
// channels 0 and 1 map to different NICs.
//
// Note: this is asserted through ncclTopoGetLocalNet() rather than
// getLocalNetCountByBw(). getLocalNetCountByBw() is bandwidth-driven -- it stops
// once the accumulated NIC bandwidth meets the GPU's bandwidth -- so when a
// single NIC already saturates the GPU it returns 1 regardless of the policy.
// The policy itself only governs how ncclTopoGetLocalNet() distributes NICs.
TEST(NetDevsPolicyTests, All_UsesEveryLocalNic)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "All_UsesEveryLocalNic",
        []()
        {
            withTopoSystem(
                [](struct ncclTopoSystem* system)
                {
                    int gpu = -1;
                    ASSERT_EQ(ncclTopoRankToIndex(system, kRank, &gpu, /*showWarn=*/true),
                              ncclSuccess);
                    int nets = localNetCount(system, gpu);
                    ASSERT_GE(nets, 2);

                    EXPECT_EQ(distinctNetsAcrossChannels(system, nets), nets);

                    int     dev0 = -1, dev1 = -1;
                    int64_t id0 = 0, id1 = 0;
                    ASSERT_EQ(ncclTopoGetLocalNet(system, kRank, /*channelId=*/0, &id0, &dev0),
                              ncclSuccess);
                    ASSERT_EQ(ncclTopoGetLocalNet(system, kRank, /*channelId=*/1, &id1, &dev1),
                              ncclSuccess);
                    EXPECT_NE(dev0, dev1);
                });
        },
        {{"NCCL_NETDEVS_POLICY", "ALL"}});
}

// NCCL_NETDEVS_POLICY=MAX:2 -- caps usage to two NICs (strictly between 1 and
// the 3-NIC pool). The rotation has period 2: channels 0 and 1 differ, and
// channel 2 wraps back to channel 0's NIC.
TEST(NetDevsPolicyTests, Max2_SelectsTwoNics)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "Max2_SelectsTwoNics",
        []()
        {
            withTopoSystem(
                [](struct ncclTopoSystem* system)
                {
                    int gpu = -1;
                    ASSERT_EQ(ncclTopoRankToIndex(system, kRank, &gpu, /*showWarn=*/true),
                              ncclSuccess);
                    int nets = localNetCount(system, gpu);
                    ASSERT_GE(nets, 2);

                    EXPECT_EQ(distinctNetsAcrossChannels(system, nets), std::min(2, nets));

                    int     dev0 = -1, dev1 = -1, dev2 = -1;
                    int64_t id0 = 0, id1 = 0, id2 = 0;
                    ASSERT_EQ(ncclTopoGetLocalNet(system, kRank, 0, &id0, &dev0), ncclSuccess);
                    ASSERT_EQ(ncclTopoGetLocalNet(system, kRank, 1, &id1, &dev1), ncclSuccess);
                    ASSERT_EQ(ncclTopoGetLocalNet(system, kRank, 2, &id2, &dev2), ncclSuccess);
                    EXPECT_NE(dev0, dev1);
                    EXPECT_EQ(dev0, dev2);
                });
        },
        {{"NCCL_NETDEVS_POLICY", "MAX:2"}});
}

// NCCL_NETDEVS_POLICY=MAX:N with N == the number of reachable NICs -- uses the
// whole pool, equivalent to ALL.
TEST(NetDevsPolicyTests, MaxEqualsPool_UsesEveryLocalNic)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "MaxEqualsPool_UsesEveryLocalNic",
        []()
        {
            withTopoSystem(
                [](struct ncclTopoSystem* system)
                {
                    int gpu = -1;
                    ASSERT_EQ(ncclTopoRankToIndex(system, kRank, &gpu, /*showWarn=*/true),
                              ncclSuccess);
                    int nets = localNetCount(system, gpu);
                    ASSERT_EQ(nets, 3) << "fixture topo_8p6l_5nic exposes 3 NICs to rank 0";
                    EXPECT_EQ(distinctNetsAcrossChannels(system, nets), nets);
                });
        },
        {{"NCCL_NETDEVS_POLICY", "MAX:3"}});
}

// NCCL_NETDEVS_POLICY=MAX:N with N larger than the pool -- clamps to the number
// of reachable NICs (proves std::min(policyCount, localNetCount)).
TEST(NetDevsPolicyTests, MaxLarge_ClampsToLocalNetCount)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "MaxLarge_ClampsToLocalNetCount",
        []()
        {
            withTopoSystem(
                [](struct ncclTopoSystem* system)
                {
                    int gpu = -1;
                    ASSERT_EQ(ncclTopoRankToIndex(system, kRank, &gpu, /*showWarn=*/true),
                              ncclSuccess);
                    int nets = localNetCount(system, gpu);
                    ASSERT_GE(nets, 2);
                    EXPECT_EQ(distinctNetsAcrossChannels(system, nets), nets);
                });
        },
        {{"NCCL_NETDEVS_POLICY", "MAX:64"}});
}

// NCCL_NETDEVS_POLICY=AUTO -- selects ceil(localNetCount / localGpuCount) NICs.
TEST(NetDevsPolicyTests, Auto_DividesNicsAcrossGpus)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "Auto_DividesNicsAcrossGpus",
        []() { withTopoSystem(expectAutoSelection); },
        {{"NCCL_NETDEVS_POLICY", "AUTO"}});
}

// NCCL_NETDEVS_POLICY=AUTO with NICs > GPUs in a locality pocket -- in
// topo_8p_ts1, GPU rank 1 reaches 2 closest NICs that no other GPU shares
// (localNetCount=2, localGpuCount=1), so AUTO must hand out exactly
// ceil(2 / 1) = 2 distinct NICs. This pins AUTO's divide-across-GPUs branch to
// a non-degenerate value (the 5nic model collapses to ceil(3 / 4) = 1).
TEST(NetDevsPolicyTests, Auto_NicsExceedGpus_SelectsTwo)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "Auto_NicsExceedGpus_SelectsTwo",
        []()
        {
            withTopoSystem(
                [](struct ncclTopoSystem* system)
                {
                    int gpu = -1;
                    ASSERT_EQ(ncclTopoRankToIndex(system, kAutoRank, &gpu, /*showWarn=*/true),
                              ncclSuccess);

                    int nets = localNetCount(system, gpu);
                    int gpus = localGpuCount(system, gpu);
                    ASSERT_EQ(nets, 2) << "fixture topo_8p_ts1 exposes 2 closest NICs to rank "
                                       << kAutoRank;
                    ASSERT_EQ(gpus, 1) << "fixture topo_8p_ts1 has 1 GPU closest to that NIC";

                    EXPECT_EQ(distinctNetsAcrossChannels(system, nets, kAutoRank),
                              divUp(nets, gpus));
                    EXPECT_EQ(distinctNetsAcrossChannels(system, nets, kAutoRank), 2);
                },
                kTopoNicsGtGpusXmlPath);
        },
        {{"NCCL_NETDEVS_POLICY", "AUTO"}});
}

// Env unset -- defaults to AUTO.
TEST(NetDevsPolicyTests, DefaultUnset_MatchesAuto)
{
    RUN_ISOLATED_TEST(
        "DefaultUnset_MatchesAuto",
        []()
        {
            ::unsetenv("NCCL_NETDEVS_POLICY");
            withTopoSystem(expectAutoSelection);
        });
}

// MAX:0 -- non-positive count is rejected by the parser; falls back to AUTO.
TEST(NetDevsPolicyTests, MaxZero_FallsBackToAuto)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "MaxZero_FallsBackToAuto", []() { withTopoSystem(expectAutoSelection); },
        {{"NCCL_NETDEVS_POLICY", "MAX:0"}});
}

// MAX:-1 -- negative count fails the > 0 check; falls back to AUTO.
TEST(NetDevsPolicyTests, MaxNegative_FallsBackToAuto)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "MaxNegative_FallsBackToAuto", []() { withTopoSystem(expectAutoSelection); },
        {{"NCCL_NETDEVS_POLICY", "MAX:-1"}});
}

// "MAX:" with no number -- atoi("") == 0; falls back to AUTO.
TEST(NetDevsPolicyTests, MaxEmpty_FallsBackToAuto)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "MaxEmpty_FallsBackToAuto", []() { withTopoSystem(expectAutoSelection); },
        {{"NCCL_NETDEVS_POLICY", "MAX:"}});
}

// Unrecognized value -- parser warns and falls back to AUTO.
TEST(NetDevsPolicyTests, InvalidValue_FallsBackToAuto)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "InvalidValue_FallsBackToAuto", []() { withTopoSystem(expectAutoSelection); },
        {{"NCCL_NETDEVS_POLICY", "BOGUS"}});
}

// Empty string -- no branch matches; falls back to AUTO.
TEST(NetDevsPolicyTests, EmptyString_FallsBackToAuto)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "EmptyString_FallsBackToAuto", []() { withTopoSystem(expectAutoSelection); },
        {{"NCCL_NETDEVS_POLICY", ""}});
}

// ---------------------------------------------------------------------------
// Symmetric-memory availability vs NCCL_P2P_LEVEL
// ---------------------------------------------------------------------------
//
// Upstream NCCL 2.28.7 removed the NCCL_P2P_LEVEL distance check from the
// symmetric-memory availability decision: once CUDA P2P can be established, the
// symmetric path is eligible regardless of topology distance. RCCL absorbed
// this through the May 2026 upstream sync, which split the single P2P probe in
// ncclTransportCheckP2pType() into two flags surfaced by ncclTopoCheckP2p():
//
//   * p2p / isAllDirectP2p  -- distance-gated by NCCL_P2P_LEVEL (legacy "flat"
//     P2P schedule).
//   * cudaP2p / isAllCudaP2p -- raw CUDA P2P capability, independent of the
//     NCCL_P2P_LEVEL distance (on HIP: any same-host or MNNVL pair).
//
// The symmetric gate in src/init.cc reads isAllCudaP2p (NOT isAllDirectP2p):
//   comm->symmetricSupport = comm->isAllCudaP2p && ncclParamWinEnable()
//                            && ncclCuMemEnable() && (GIN || oneLsaTeams);
// so a restrictive NCCL_P2P_LEVEL must not, on its own, disable symmetric
// memory. These tests pin that divergence at the ncclTopoCheckP2p boundary.
//
// Each case runs in an isolated process so the cached ncclTopoUserP2pLevel
// static and the env value are fresh per case.

// Default P2P level: the XGMI pair is both within the distance gate
// and CUDA-P2P capable, so both flags are set.
TEST(SymmetricMemP2pLevelTests, DefaultLevel_BothP2pFlagsSet)
{
    RUN_ISOLATED_TEST(
        "DefaultLevel_BothP2pFlagsSet",
        []()
        {
            ::unsetenv("NCCL_P2P_LEVEL");
            ::unsetenv("NCCL_P2P_DISABLE");
            int p2p = -1, cudaP2p = -1;
            checkXgmiPairP2p(&p2p, &cudaP2p);
            EXPECT_EQ(p2p, 1) << "default level: distance-gated p2p should be set";
            EXPECT_EQ(cudaP2p, 1) << "default level: raw CUDA P2P should be set";
        });
}

// NCCL_P2P_LEVEL=LOC (PATH_LOC=0) is more restrictive than the XGMI distance
// (PATH_NVL=1), so the distance-gated flag (-> isAllDirectP2p) drops to 0. The
// raw CUDA-P2P flag (-> isAllCudaP2p, the symmetric-memory prerequisite) must
// stay 1.
TEST(SymmetricMemP2pLevelTests, RestrictiveLevelLOC_KeepsCudaP2p)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "RestrictiveLevelLOC_KeepsCudaP2p",
        []()
        {
            int p2p = -1, cudaP2p = -1;
            checkXgmiPairP2p(&p2p, &cudaP2p);
            EXPECT_EQ(p2p, 0) << "NCCL_P2P_LEVEL=LOC must engage the distance gate";
            EXPECT_EQ(cudaP2p, 1)
                << "raw CUDA P2P (symmetric-memory prereq) must be distance-independent";
        },
        {{"NCCL_P2P_LEVEL", "LOC"}});
}

// Old-style numeric form: NCCL_P2P_LEVEL=0 maps to PATH_LOC. Same contract as
// the string form -- distance gate engages, CUDA-P2P capability is unchanged.
TEST(SymmetricMemP2pLevelTests, RestrictiveLevelNumericZero_KeepsCudaP2p)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "RestrictiveLevelNumericZero_KeepsCudaP2p",
        []()
        {
            int p2p = -1, cudaP2p = -1;
            checkXgmiPairP2p(&p2p, &cudaP2p);
            EXPECT_EQ(p2p, 0) << "NCCL_P2P_LEVEL=0 must engage the distance gate";
            EXPECT_EQ(cudaP2p, 1)
                << "raw CUDA P2P (symmetric-memory prereq) must be distance-independent";
        },
        {{"NCCL_P2P_LEVEL", "0"}});
}

// NCCL_P2P_DISABLE=1 collapses the distance level to PATH_LOC. Even with
// P2P fully "disabled" by distance, raw CUDA-P2P capability stays 1 so the
// symmetric path remains eligible.
TEST(SymmetricMemP2pLevelTests, P2pDisable_KeepsCudaP2p)
{
    RUN_ISOLATED_TEST_WITH_ENV(
        "P2pDisable_KeepsCudaP2p",
        []()
        {
            int p2p = -1, cudaP2p = -1;
            checkXgmiPairP2p(&p2p, &cudaP2p);
            EXPECT_EQ(p2p, 0) << "NCCL_P2P_DISABLE=1 must engage the distance gate";
            EXPECT_EQ(cudaP2p, 1)
                << "raw CUDA P2P (symmetric-memory prereq) must be distance-independent";
        },
        {{"NCCL_P2P_DISABLE", "1"}});
}

}  // namespace RcclUnitTesting
