/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/graph/rome_models.cc: the ring/tree string parsers and the preset topology
// matchers, fed systems rebuilt from the model tables themselves.

#include <gtest/gtest.h>

#include <malloc.h>
#include <unistd.h>

#include <algorithm>
#include <climits>
#include <csignal>
#include <cstring>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "../../common/ScopedEnv.hpp"
#include "ScopedHook.h"
#include "fakes/graph_fakes.h"
#include "fakes/param_redirect.h"

#include "graph.h"

// rome_models.cc's per-channel NET lookup, observable by TreeGraph_ResolvesOutgoingNetFromTheSecondGpu.
static std::function<ncclResult_t(ncclTopoSystem*, int, int, int64_t*, int*)> g_romeGetLocalNet = ncclTopoGetLocalNet;
static ncclResult_t RomeGetLocalNet(ncclTopoSystem* s, int rank, int channelId, int64_t* id, int* dev) {
  return g_romeGetLocalNet(s, rank, channelId, id, dev);
}
#define ncclTopoGetLocalNet RomeGetLocalNet
#include ROME_MODELS_CC_PATH
#undef ncclTopoGetLocalNet

#include "TopoScene.h"

using namespace topo_scene;

namespace {

constexpr int kModels = sizeof(romeTopoModels) / sizeof(romeTopoModels[0]);
constexpr int kSixteenNicModel = 41;  // rome_model_84

#if defined(__GLIBC_PREREQ)
#if __GLIBC_PREREQ(2, 33)
#define ROME_TEST_HAS_MALLINFO2 1
#endif
#endif

// Bytes the heap holds; a leak shows up as growth across a call. mallinfo2 needs glibc 2.33.
size_t InUseBytes() {
#ifdef ROME_TEST_HAS_MALLINFO2
  struct mallinfo2 info = mallinfo2();
#else
  struct mallinfo info = mallinfo();
#endif
  return (size_t)info.uordblks + (size_t)info.hblkhd;
}

std::string BusId(int64_t id) {
  char busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
  int64ToBusId(id, busId);
  return busId;
}

// The topology a preset model describes: its GPUs and NICs straight under their NUMA node, XGMI from connMatrix.
// NUMA ids come from the devices; a "00" entry of the pattern is an empty node, numbered in the gap it fills.
std::vector<int> ModelNumas(const rcclRomeModel& m) {
  std::set<int> used(m.gpuNuma, m.gpuNuma + m.nGpus);
  used.insert(m.nicNuma, m.nicNuma + m.nNics);
  std::vector<int> ids;
  auto next = used.begin();
  for (int p = 0; p < m.nCpus; p++) {
    if (m.pattern[2 * p] == '0' && m.pattern[2 * p + 1] == '0') ids.push_back(ids.empty() ? 0 : ids.back() + 1);
    else ids.push_back(*next++);
  }
  return ids;
}

std::string ModelXml(const rcclRomeModel& m, const char* cpuIds = kRomeCpu) {
  std::string body;
  for (int c : ModelNumas(m)) {
    std::string cpu;
    for (int g = 0; g < m.nGpus; g++) {
      if (m.gpuNuma[g] != c) continue;
      std::string links;
      for (int p = 0; p < m.nGpus; p++) {
        if (m.connMatrix[g * m.nGpus + p]) links += Xgmi(BusId(m.gpuIds[p]), m.connMatrix[g * m.nGpus + p]);
      }
      cpu += GpuPci(BusId(m.gpuIds[g]), g, g, links);
    }
    for (int n = 0; n < m.nNics; n++) {
      if (m.nicNuma[n] == c) cpu += NicPci(BusId(m.nicIds[n]), RailNet("net", n, n));
    }
    body += Cpu(c, cpu, "", cpuIds);
  }
  return System(body);
}

class GraphRomeModelsMicrotest : public ::testing::Test {
 protected:
  SystemPtr system_;

  void SetUp() override { ResetGraphFakes(); }
  void TearDown() override {
    unsetenv("RCCL_DUMP_ROME_MODEL_FILE");
    ResetGraphFakes();
  }

  // Builds the model's node; the NIC distances are planted from its gdrLevel table.
  ncclTopoSystem* UseModel(const rcclRomeModel& m, int nRanks = -1, const char* cpuIds = kRomeCpu) {
    system_ = BuildWithPaths(ModelXml(m, cpuIds));
    for (int n = 0; n < system_->nodes[NET].count; n++) {
      int nic = system_->nodes[NET].nodes[n].net.dev;
      for (int g = 0; g < system_->nodes[GPU].count; g++) {
        ncclTopoNode* gpu = system_->nodes[GPU].nodes + g;
        gpu->paths[NET][n].type = m.gdrLevel[nic * m.nGpus + gpu->gpu.dev];
      }
    }
    system_->nRanks = nRanks == -1 ? m.nGpus : nRanks;
    system_->romeTopoModelIdx = RCCL_ROME_TOPO_PRESET_MODEL_IDX_NONE;
    return system_.get();
  }

  ncclTopoSystem* Use(const std::string& text, int nRanks = -1) {
    system_ = BuildWithPaths(text);
    if (nRanks != -1) system_->nRanks = nRanks;
    system_->romeTopoModelIdx = RCCL_ROME_TOPO_PRESET_MODEL_IDX_NONE;
    return system_.get();
  }

  std::vector<int> Channel(const ncclTopoGraph* graph, int c) {
    int ngpus = system_->nodes[GPU].count;
    return std::vector<int>(graph->intra + c * ngpus, graph->intra + (c + 1) * ngpus);
  }
};

// The matchers compare against these tables, so a model the builder cannot reproduce is never tested.
TEST_F(GraphRomeModelsMicrotest, ParseRomeSystem_RebuildsEveryModelsSignature) {
  for (int i = 0; i < kModels; i++) {
    const rcclRomeModel& m = romeTopoModels[i];
    UseModel(m);
    rcclRomeModel topo;
    char pattern[256];

    ASSERT_EQ(ncclSuccess, parseRomeSystem(system_.get(), &topo, pattern, nullptr)) << i;

    EXPECT_STREQ(m.pattern, pattern) << i;
    EXPECT_EQ(m.nLinks, topo.nLinks) << i;
    EXPECT_EQ(m.nNics, topo.nNics) << i;
    for (int g = 0; g < m.nGpus * m.nGpus; g++) ASSERT_EQ(m.connMatrix[g], topo.connMatrix[g]) << i << " at " << g;
  }
}

// --- Preset matchers

bool IsPermutation(const std::vector<int>& ranks) {
  return std::set<int>(ranks.begin(), ranks.end()).size() == ranks.size();
}

TEST_F(GraphRomeModelsMicrotest, Rome4P2H_EveryEightGpuModelMatchesItsOwnTopology) {
  for (int i = 0; i < kModels; i++) {
    const rcclRomeModel& m = romeTopoModels[i];
    if (m.nGpus > 8) continue;
    UseModel(m);
    GraphPtr ring = NewGraph(0, NCCL_TOPO_PATTERN_RING);
    g_tuningOverrideCalls = 0;

    ASSERT_EQ(ncclSuccess, parseRome4P2H(system_.get(), ring.get(), nullptr)) << i;

    ASSERT_GT(ring->nChannels, 0) << i;
    EXPECT_NE(RCCL_ROME_TOPO_PRESET_MODEL_IDX_NONE, system_->romeTopoModelIdx) << i;
    EXPECT_LE(system_->romeTopoModelIdx, i) << i;
    EXPECT_EQ(1, g_tuningOverrideCalls) << i;
    for (int c = 0; c < ring->nChannels; c++) EXPECT_TRUE(IsPermutation(Channel(ring.get(), c))) << i << "/" << c;
  }
}

TEST_F(GraphRomeModelsMicrotest, Rome4P2H_TreesUseTreeBaseOrFallBackToTheRings) {
  for (int i = 0; i < kModels; i++) {
    const rcclRomeModel& m = romeTopoModels[i];
    if (m.nGpus > 8) continue;
    UseModel(m);
    GraphPtr tree = NewGraph(1, NCCL_TOPO_PATTERN_BALANCED_TREE);

    ASSERT_EQ(ncclSuccess, parseRome4P2H(system_.get(), tree.get(), nullptr)) << i;

    const rcclRomeModel& found = romeTopoModels[system_->romeTopoModelIdx];
    if (found.treeBase && found.treeBase[0]) {
      EXPECT_NE(0, tree->treeBase[0][0]) << i;
    } else {
      EXPECT_GT(tree->nChannels, 0) << i;
    }
  }
}

TEST_F(GraphRomeModelsMicrotest, Rome4P2H_OtherPatternsAndBigSystemsAreSkipped) {
  UseModel(romeTopoModels[8]);
  GraphPtr nvls = NewGraph(3, NCCL_TOPO_PATTERN_NVLS);
  ASSERT_EQ(ncclSuccess, parseRome4P2H(system_.get(), nvls.get(), nullptr));
  EXPECT_EQ(0, nvls->nChannels);

  UseModel(romeTopoModels[24]);  // 16 GPUs
  GraphPtr ring = NewGraph(0, NCCL_TOPO_PATTERN_RING);
  ASSERT_EQ(ncclSuccess, parseRome4P2H(system_.get(), ring.get(), nullptr));
  EXPECT_EQ(0, ring->nChannels);
}

TEST_F(GraphRomeModelsMicrotest, Rome4P2H_NeedsRomeUnlessTheModelSkipsTheCpuCheck) {
  UseModel(romeTopoModels[8], -1, kAmdCpu);
  GraphPtr ring = NewGraph(0, NCCL_TOPO_PATTERN_RING);
  ASSERT_EQ(ncclSuccess, parseRome4P2H(system_.get(), ring.get(), nullptr));
  EXPECT_EQ(0, ring->nChannels);
  EXPECT_TRUE(system_->type & RCCL_TOPO_4P2H_ROME);  // XGMI-linked, so still flagged as Rome 4P2H

  UseModel(romeTopoModels[44], -1, kAmdCpu);  // noCpuCheck=1
  ASSERT_EQ(ncclSuccess, parseRome4P2H(system_.get(), ring.get(), nullptr));
  EXPECT_GT(ring->nChannels, 0);
}

TEST_F(GraphRomeModelsMicrotest, Rome4P2H_UserRingRemapReplacesTheModelRings) {
  UseModel(romeTopoModels[8]);
  GraphPtr ring = NewGraph(0, NCCL_TOPO_PATTERN_RING);

  ASSERT_EQ(ncclSuccess, parseRome4P2H(system_.get(), ring.get(), "0 1 2 3 4 5 6 7"));

  EXPECT_EQ(1, ring->nChannels);
}

TEST_F(GraphRomeModelsMicrotest, Rome4P2H_OddHostCountsUseTheTailRingsOnTheLastTwoHosts) {
  const rcclRomeModel& m = romeTopoModels[44];  // rome_model_88 has ringTail1/ringTail2
  for (int hostIdx : {0, 1, 2}) {
    UseModel(m);
    system_->nHosts = 3;
    system_->hostIdx = hostIdx;
    GraphPtr ring = NewGraph(0, NCCL_TOPO_PATTERN_RING);

    ASSERT_EQ(ncclSuccess, parseRome4P2H(system_.get(), ring.get(), nullptr)) << hostIdx;

    GraphPtr expected = NewGraph(0, NCCL_TOPO_PATTERN_RING);
    const char* rings = hostIdx == 2 ? m.ringTail1 : hostIdx == 1 ? m.ringTail2 : m.ringBase;
    int map[8] = {0, 1, 2, 3, 4, 5, 6, 7}, nets[8] = {0, 1, 2, 3, 0, 0, 0, 0};
    ASSERT_EQ(ncclSuccess, parseGraph(rings, system_.get(), expected.get(), map, nets, 0));
    ASSERT_EQ(expected->nChannels, ring->nChannels) << hostIdx;
    for (int c = 0; c < ring->nChannels; c++) EXPECT_EQ(Channel(expected.get(), c), Channel(ring.get(), c)) << hostIdx;
  }

  UseModel(romeTopoModels[8]);  // no tails
  system_->nHosts = 3;
  GraphPtr ring = NewGraph(0, NCCL_TOPO_PATTERN_RING);
  ASSERT_EQ(ncclSuccess, parseRome4P2H(system_.get(), ring.get(), nullptr));
  EXPECT_TRUE(Logged("Dropping back due to lack of support for odd-number of nodes"));
}

TEST_F(GraphRomeModelsMicrotest, Rome4P2H_EvenHostCountsAlternateTheRingDirection) {
  UseModel(romeTopoModels[8]);
  system_->nHosts = 2;
  GraphPtr forward = NewGraph(0, NCCL_TOPO_PATTERN_RING);
  ASSERT_EQ(ncclSuccess, parseRome4P2H(system_.get(), forward.get(), nullptr));

  system_->hostIdx = 1;
  system_->romeTopoModelIdx = RCCL_ROME_TOPO_PRESET_MODEL_IDX_NONE;
  GraphPtr reverse = NewGraph(0, NCCL_TOPO_PATTERN_RING);
  ASSERT_EQ(ncclSuccess, parseRome4P2H(system_.get(), reverse.get(), nullptr));

  std::vector<int> f = Channel(forward.get(), 0), r = Channel(reverse.get(), 0);
  std::reverse(r.begin(), r.end());
  EXPECT_EQ(f, r);
}

TEST_F(GraphRomeModelsMicrotest, Rome4P2H_NetOverridePairsGpusSharingASwitch) {
  const rcclRomeModel& m = romeTopoModels[43];  // rome_model_87: netOverride=1
  UseModel(m);
  GraphPtr tree = NewGraph(1, NCCL_TOPO_PATTERN_BALANCED_TREE);
  ASSERT_EQ(ncclSuccess, parseRome4P2H(system_.get(), tree.get(), nullptr));
  ASSERT_EQ(43, system_->romeTopoModelIdx);

  int pxb = 0;
  for (int g = 0; g < 8; g++) {
    for (int n = 0; n < 4; n++) pxb += system_->nodes[GPU].nodes[g].paths[NET][n].type == PATH_PXB;
  }
  EXPECT_EQ(8, pxb);  // the four planted PXB pairs, plus each partner GPU (dev / 2)

  UseModel(m);
  g_params["RCCL_DISABLE_NET_OVERRIDE"] = 1;
  tree = NewGraph(1, NCCL_TOPO_PATTERN_BALANCED_TREE);
  ASSERT_EQ(ncclSuccess, parseRome4P2H(system_.get(), tree.get(), nullptr));
  pxb = 0;
  for (int g = 0; g < 8; g++) {
    for (int n = 0; n < 4; n++) pxb += system_->nodes[GPU].nodes[g].paths[NET][n].type == PATH_PXB;
  }
  EXPECT_EQ(4, pxb);
}

TEST_F(GraphRomeModelsMicrotest, Rome4P2H_NicsMustMatchTheModelGdrLevels) {
  UseModel(romeTopoModels[43]);
  for (int g = 0; g < 8; g++) {
    for (int n = 0; n < 4; n++) system_->nodes[GPU].nodes[g].paths[NET][n].type = PATH_SYS;
  }
  GraphPtr ring = NewGraph(0, NCCL_TOPO_PATTERN_RING);

  ASSERT_EQ(ncclSuccess, parseRome4P2H(system_.get(), ring.get(), nullptr));

  EXPECT_EQ(0, ring->nChannels);
}

TEST_F(GraphRomeModelsMicrotest, Rome4P2H_TwoNicsPerGpuMatchGreedily) {
  const rcclRomeModel& m = romeTopoModels[41];  // rome_model_84: 16 NICs on 8 GPUs
  UseModel(m);
  GraphPtr ring = NewGraph(0, NCCL_TOPO_PATTERN_RING);
  ASSERT_EQ(ncclSuccess, parseRome4P2H(system_.get(), ring.get(), nullptr));
  EXPECT_EQ(41, system_->romeTopoModelIdx);

  UseModel(m);
  system_->nodes[GPU].nodes[0].paths[NET][0].type = PATH_DIS;
  ring = NewGraph(0, NCCL_TOPO_PATTERN_RING);
  ASSERT_EQ(ncclSuccess, parseRome4P2H(system_.get(), ring.get(), nullptr));
  EXPECT_EQ(0, ring->nChannels);
}

TEST_F(GraphRomeModelsMicrotest, A2a8P_MatchesEightFullyConnectedGpusOnTwoSockets) {
  int matched = 0;
  for (int i = 0; i < kModels; i++) {
    const rcclRomeModel& m = romeTopoModels[i];
    if (m.nGpus != 8 || m.nCpus != 2 || !checkAlltoallWidth(const_cast<rcclRomeModel*>(&m))) continue;
    if (i == kSixteenNicModel) continue;  // see A2a8P_KnownBug_SixteenNicsSearchEveryNicOrdering
    for (int pattern : {NCCL_TOPO_PATTERN_RING, NCCL_TOPO_PATTERN_BALANCED_TREE}) {
      UseModel(m, -1, kAmdCpu);
      GraphPtr graph = NewGraph(pattern == NCCL_TOPO_PATTERN_RING ? 0 : 1, pattern);

      ASSERT_EQ(ncclSuccess, parseA2a8P(system_.get(), graph.get(), nullptr)) << i;

      EXPECT_TRUE(graph->nChannels > 0 || graph->treeBase[0][0]) << i << "/" << pattern;
      EXPECT_TRUE(system_->type & RCCL_TOPO_4P2H_ROME) << i;
    }
    matched++;
  }
  EXPECT_GE(matched, 5);
}

// permuteNetIds starts at NIC 2 and checks only complete orderings, so 16 NICs walk ~15! of them.
// Pins AICOMRCCL-3028; a fix flips this pin.
TEST_F(GraphRomeModelsMicrotest, A2a8P_KnownBug_SixteenNicsSearchEveryNicOrdering) {
  UseModel(romeTopoModels[kSixteenNicModel]);
  GraphPtr ring = NewGraph(0, NCCL_TOPO_PATTERN_RING);
  const std::string style = ::testing::GTEST_FLAG(death_test_style);
  ::testing::GTEST_FLAG(death_test_style) = "threadsafe";

  EXPECT_EXIT(
      {
        alarm(2);
  ::testing::GTEST_FLAG(death_test_style) = style;
        (void)parseA2a8P(system_.get(), ring.get(), nullptr);
        _exit(0);
      },
      ::testing::KilledBySignal(SIGALRM), "");
}

TEST_F(GraphRomeModelsMicrotest, A2a8P_RailTreesUnlessDisabled) {
  const rcclRomeModel& m = romeTopoModels[40];  // rome_model_81 has treeRail
  UseModel(m, -1, kAmdCpu);
  GraphPtr tree = NewGraph(1, NCCL_TOPO_PATTERN_BALANCED_TREE);
  ASSERT_EQ(ncclSuccess, parseA2a8P(system_.get(), tree.get(), nullptr));
  EXPECT_TRUE(system_->useRailOptimizedTrees);

  UseModel(m, -1, kAmdCpu);
  g_params["RCCL_DISABLE_RAIL_TREES"] = 1;
  tree = NewGraph(1, NCCL_TOPO_PATTERN_BALANCED_TREE);
  ASSERT_EQ(ncclSuccess, parseA2a8P(system_.get(), tree.get(), nullptr));
  EXPECT_FALSE(system_->useRailOptimizedTrees);
  EXPECT_GT(tree->nChannels, 0);
}

TEST_F(GraphRomeModelsMicrotest, A2a8P_OddHostsUseTheTails) {
  const rcclRomeModel& m = romeTopoModels[40];
  for (int hostIdx : {0, 1, 2}) {
    UseModel(m, -1, kAmdCpu);
    system_->nHosts = 3;
    system_->hostIdx = hostIdx;
    GraphPtr ring = NewGraph(0, NCCL_TOPO_PATTERN_RING);

    ASSERT_EQ(ncclSuccess, parseA2a8P(system_.get(), ring.get(), nullptr)) << hostIdx;

    EXPECT_GT(ring->nChannels, 0) << hostIdx;
  }
  UseModel(romeTopoModels[38], -1, kAmdCpu);  // no tails
  system_->nHosts = 3;
  GraphPtr ring = NewGraph(0, NCCL_TOPO_PATTERN_RING);
  ASSERT_EQ(ncclSuccess, parseA2a8P(system_.get(), ring.get(), "0 1 2 3 4 5 6 7"));
  EXPECT_EQ(1, ring->nChannels);
  EXPECT_TRUE(Logged("Dropping back due to lack of support for odd-number of nodes"));
}

TEST_F(GraphRomeModelsMicrotest, A2a8P_SkipsOtherShapes) {
  Use(XgmiSystem(4));
  GraphPtr ring = NewGraph(0, NCCL_TOPO_PATTERN_RING);

  ASSERT_EQ(ncclSuccess, parseA2a8P(system_.get(), ring.get(), nullptr));

  EXPECT_EQ(0, ring->nChannels);
}

// The GPU permutation table is freed when no model matches (AICOMRCCL-3029). Under ASan mallinfo is not
// tracked, so LeakSanitizer is what checks this.
TEST_F(GraphRomeModelsMicrotest, A2a8P_NoMatchFreesThePermutationTable) {
  UseModel(romeTopoModels[44], -1, kAmdCpu);
  system_->nodes[DEV].nodes[0].links[0].bw /= 2;  // no model has a narrower link
  ASSERT_EQ(ncclSuccess, ncclTopoComputePaths(system_.get(), nullptr));
  GraphPtr ring = NewGraph(0, NCCL_TOPO_PATTERN_RING);

  size_t before = InUseBytes();
  ASSERT_EQ(ncclSuccess, parseA2a8P(system_.get(), ring.get(), nullptr));
  int64_t retained = (int64_t)InUseBytes() - (int64_t)before;

  EXPECT_EQ(0, ring->nChannels);
  EXPECT_LT(retained, (int64_t)(24 * 24 * 8 * sizeof(int)));
}

TEST_F(GraphRomeModelsMicrotest, A2a8P_EvenHostCountsAlternateTheRingDirection) {
  UseModel(romeTopoModels[40], -1, kAmdCpu);
  system_->nHosts = 2;
  GraphPtr forward = NewGraph(0, NCCL_TOPO_PATTERN_RING);
  ASSERT_EQ(ncclSuccess, parseA2a8P(system_.get(), forward.get(), nullptr));

  UseModel(romeTopoModels[40], -1, kAmdCpu);
  system_->nHosts = 2;
  system_->hostIdx = 1;
  GraphPtr reverse = NewGraph(0, NCCL_TOPO_PATTERN_RING);
  ASSERT_EQ(ncclSuccess, parseA2a8P(system_.get(), reverse.get(), nullptr));

  std::vector<int> f = Channel(forward.get(), 0), r = Channel(reverse.get(), 0);
  std::reverse(r.begin(), r.end());
  EXPECT_EQ(f, r);
}

TEST_F(GraphRomeModelsMicrotest, A2a8P_TreeBaseIsUsedWhenThereAreNoRailTrees) {
  rcclRomeModel& m = romeTopoModels[40];
  const char* saved = m.treeBase;
  m.treeBase = "(0(1))";
  UseModel(m, -1, kAmdCpu);
  g_params["RCCL_DISABLE_RAIL_TREES"] = 1;
  GraphPtr tree = NewGraph(1, NCCL_TOPO_PATTERN_BALANCED_TREE);

  ncclResult_t ret = parseA2a8P(system_.get(), tree.get(), nullptr);
  m.treeBase = saved;

  EXPECT_EQ(ncclSuccess, ret);
  EXPECT_STREQ("(0(1))", tree->treeBase[0]);
  EXPECT_GT(tree->nChannels, 0);  // channels still come from ringBase
}

TEST_F(GraphRomeModelsMicrotest, A2a8P_PixAndPxbNicsAreInterchangeable) {
  UseModel(romeTopoModels[40], -1, kAmdCpu);
  for (int g = 0; g < 8; g++) {
    for (int n = 0; n < 8; n++) {
      ncclTopoLinkList& path = system_->nodes[GPU].nodes[g].paths[NET][n];
      if (path.type == PATH_PXB) path.type = PATH_PIX;
    }
  }
  GraphPtr ring = NewGraph(0, NCCL_TOPO_PATTERN_RING);

  ASSERT_EQ(ncclSuccess, parseA2a8P(system_.get(), ring.get(), nullptr));

  EXPECT_EQ(40, system_->romeTopoModelIdx);
}

TEST_F(GraphRomeModelsMicrotest, P1H16P_MatchesSixteenGpuModels) {
  for (int i : {24, 26}) {
    UseModel(romeTopoModels[i]);
    GraphPtr ring = NewGraph(0, NCCL_TOPO_PATTERN_RING);

    ASSERT_EQ(ncclSuccess, parse1H16P(system_.get(), ring.get())) << i;

    EXPECT_GT(ring->nChannels, 0) << i;
    EXPECT_TRUE(system_->type & RCCL_TOPO_16P1H) << i;
    EXPECT_EQ(i, system_->romeTopoModelIdx);
  }
  EXPECT_TRUE(system_->ll128Enabled);  // rome_model_59's options
}

// As in parseA2a8P, the 21 MB permutation table is freed when no model matches (AICOMRCCL-3029).
TEST_F(GraphRomeModelsMicrotest, P1H16P_NoMatchFreesThePermutationTable) {
  rcclRomeModel rewired = romeTopoModels[24];
  auto link = [&](int a, int b, int v) { rewired.connMatrix[a * 16 + b] = rewired.connMatrix[b * 16 + a] = v; };
  link(0, 1, 0), link(8, 9, 0), link(0, 8, 1), link(1, 9, 1);  // same link counts, different shape
  UseModel(rewired);
  GraphPtr ring = NewGraph(0, NCCL_TOPO_PATTERN_RING);

  size_t before = InUseBytes();
  ASSERT_EQ(ncclSuccess, parse1H16P(system_.get(), ring.get()));
  int64_t retained = (int64_t)InUseBytes() - (int64_t)before;

  EXPECT_EQ(0, ring->nChannels);
  EXPECT_EQ(RCCL_ROME_TOPO_PRESET_MODEL_IDX_NONE, system_->romeTopoModelIdx);
  EXPECT_LT(retained, (int64_t)(24u * 24 * 24 * 24 * 16 * sizeof(int)));
}

TEST_F(GraphRomeModelsMicrotest, P1H16P_NeedsRomeAndSixteenGpus) {
  UseModel(romeTopoModels[24], -1, kAmdCpu);
  GraphPtr ring = NewGraph(0, NCCL_TOPO_PATTERN_RING);
  ASSERT_EQ(ncclSuccess, parse1H16P(system_.get(), ring.get()));
  EXPECT_EQ(0, ring->nChannels);

  UseModel(romeTopoModels[8]);
  ASSERT_EQ(ncclSuccess, parse1H16P(system_.get(), ring.get()));
  EXPECT_EQ(0, ring->nChannels);
}

TEST_F(GraphRomeModelsMicrotest, P4H4P_MatchesFourHivesWithANicPerGpu) {
  UseModel(romeTopoModels[32]);  // rome_model_68
  GraphPtr ring = NewGraph(0, NCCL_TOPO_PATTERN_RING);

  ASSERT_EQ(ncclSuccess, parse4H4P(system_.get(), ring.get()));

  EXPECT_GT(ring->nChannels, 0);
  EXPECT_EQ(RCCL_ROME_TOPO_PRESET_MODEL_IDX_4H4P, system_->romeTopoModelIdx);
  EXPECT_TRUE(system_->type & RCCL_TOPO_4P2H_ROME);
  EXPECT_TRUE(Logged("Found matching Rome model 4P4H"));
}

TEST_F(GraphRomeModelsMicrotest, P4H4P_RejectsOtherShapes) {
  GraphPtr ring = NewGraph(0, NCCL_TOPO_PATTERN_RING);
  UseModel(romeTopoModels[32], -1, kAmdCpu);
  ASSERT_EQ(ncclSuccess, parse4H4P(system_.get(), ring.get()));
  EXPECT_EQ(0, ring->nChannels);

  rcclRomeModel noNics = romeTopoModels[32];
  noNics.nNics = 0;
  UseModel(noNics);  // the same four hives, but no NICs
  ASSERT_EQ(ncclSuccess, parse4H4P(system_.get(), ring.get()));
  EXPECT_EQ(0, ring->nChannels);
  EXPECT_FALSE(Logged("Found matching Rome model 4P4H"));

  UseModel(romeTopoModels[32]);
  for (int g = 0; g < 16; g++) system_->nodes[GPU].nodes[g].paths[NET][0].type = PATH_PHB;  // NIC 0 has no PIX GPU
  ASSERT_EQ(ncclSuccess, parse4H4P(system_.get(), ring.get()));
  EXPECT_EQ(0, ring->nChannels);
}

TEST_F(GraphRomeModelsMicrotest, FindGpuHives_NeedsFullHives) {
  rcclRomeModel m = romeTopoModels[32];
  std::vector<int> hives(16, -1);
  int nHives[4] = {0, 0, 0, 0};
  ASSERT_EQ(ncclSuccess, find_gpu_hives(hives.data(), nHives, &m));
  EXPECT_EQ(4, nHives[3]);

  m.connMatrix[0 * 16 + 1] = m.connMatrix[1 * 16 + 0] = 0;  // GPU 1 no longer joins GPU 0's hive
  int again[4] = {0, 0, 0, 0};
  EXPECT_EQ(ncclInternalError, find_gpu_hives(hives.data(), again, &m));
}

TEST_F(GraphRomeModelsMicrotest, FindGpuHives_UnevenHivesAreAnError) {
  rcclRomeModel m = {};
  m.nGpus = 8;
  m.nLinks = 3;
  for (int a = 0; a < 8; a++) {
    for (int b = 0; b < 8; b++) m.connMatrix[a * 8 + b] = a != b && a / 4 == b / 4;
  }
  m.connMatrix[4 * 8 + 0] = m.connMatrix[0 * 8 + 4] = 1;  // GPU 4 joins GPU 0's hive, leaving 5 and 3
  std::vector<int> hives(8, -1);
  int nHives[2] = {0, 0};

  EXPECT_EQ(ncclInternalError, find_gpu_hives(hives.data(), nHives, &m));

  EXPECT_EQ(5, nHives[0]);
  EXPECT_EQ(3, nHives[1]);
}

TEST_F(GraphRomeModelsMicrotest, GioTopos_MatchesTwoHivesOfEight) {
  UseModel(gio16gColumbaModel, -1, kAmdCpu);
  GraphPtr ring = NewGraph(0, NCCL_TOPO_PATTERN_RING);

  ASSERT_EQ(ncclSuccess, parseGIOTopos(system_.get(), ring.get()));

  EXPECT_EQ(8, ring->nChannels);
  EXPECT_EQ(RCCL_ROME_TOPO_PRESET_MODEL_IDX_GIO_COLUMBA, system_->romeTopoModelIdx);
  EXPECT_TRUE(Logged("Found GigaIO rome topo"));
}

TEST_F(GraphRomeModelsMicrotest, GioTopos_SkipsOtherShapes) {
  GraphPtr ring = NewGraph(0, NCCL_TOPO_PATTERN_RING);
  UseModel(romeTopoModels[24]);  // 16 GPUs, but hives of 4
  ASSERT_EQ(ncclSuccess, parseGIOTopos(system_.get(), ring.get()));
  EXPECT_EQ(0, ring->nChannels);

  rcclRomeModel broken = gio16gColumbaModel;
  broken.connMatrix[0 * 16 + 1] = broken.connMatrix[1 * 16 + 0] = 0;
  broken.connMatrix[0 * 16 + 8] = broken.connMatrix[8 * 16 + 0] = 1;
  UseModel(broken, -1, kAmdCpu);
  ASSERT_EQ(ncclSuccess, parseGIOTopos(system_.get(), ring.get()));
  EXPECT_EQ(0, ring->nChannels);

  Use(XgmiSystem(4));
  ASSERT_EQ(ncclSuccess, parseGIOTopos(system_.get(), ring.get()));
  EXPECT_EQ(0, ring->nChannels);
}

// GPU i is XGMI-linked to every GPU but its partner; with nics, NIC n shares a switch with GPU 2n+1.
std::string ChordalRing(const int partner[8], bool nics) {
  std::string body;
  for (int g = 0; g < 8; g++) {
    std::string links;
    for (int p = 0; p < 8; p++) {
      if (p != g && p != partner[g]) links += Xgmi(GpuBus(p));
    }
    std::string gpu = GpuPci(GpuBus(g), g, g, links);
    body += nics && g % 2 ? Pci(SwitchBus(g), gpu + NicPci(NicBus(g), RailNet("net", g / 2, g / 2))) : gpu;
  }
  return System(Cpu(0, body));
}

TEST_F(GraphRomeModelsMicrotest, ChordalRing_LaysTheSixRingsOverRemappedGpus) {
  const int mirrored[8] = {7, 6, 5, 4, 3, 2, 1, 0};
  const int adjacent[8] = {1, 0, 3, 2, 5, 4, 7, 6};
  for (const int* partner : {mirrored, adjacent}) {
    Use(ChordalRing(partner, false));
    GraphPtr ring = NewGraph(0, NCCL_TOPO_PATTERN_RING);

    ASSERT_EQ(ncclSuccess, parseChordalRing(system_.get(), ring.get()));

    ASSERT_EQ(6, ring->nChannels);
    EXPECT_TRUE(system_->type & RCCL_TOPO_CR8G);
    for (int c = 0; c < 6; c++) {
      std::vector<int> ch = Channel(ring.get(), c);
      for (int i = 0; i < 8; i++) EXPECT_NE(partner[ch[i]], ch[(i + 1) % 8]) << c;  // every hop is a direct link
    }
  }
}

TEST_F(GraphRomeModelsMicrotest, ChordalRing_InterNodeStartsEachChannelAtItsNicsGpu) {
  const int mirrored[8] = {7, 6, 5, 4, 3, 2, 1, 0};
  Use(ChordalRing(mirrored, true), 16);
  GraphPtr ring = NewGraph(0, NCCL_TOPO_PATTERN_RING);

  ASSERT_EQ(ncclSuccess, parseChordalRing(system_.get(), ring.get()));

  ASSERT_EQ(4, ring->nChannels);
  for (int c = 0; c < 4; c++) {
    EXPECT_EQ(c, ring->inter[2 * c]);
    EXPECT_EQ(2 * c + 1, ring->intra[c * 8]) << c;  // the GPU on the NIC's switch
    std::vector<int> ch = Channel(ring.get(), c);
    EXPECT_EQ(8u, std::set<int>(ch.begin(), ch.end()).size());
  }
}

TEST_F(GraphRomeModelsMicrotest, ChordalRing_SkipsOtherShapes) {
  GraphPtr ring = NewGraph(0, NCCL_TOPO_PATTERN_RING);
  const int mirrored[8] = {7, 6, 5, 4, 3, 2, 1, 0};
  Use(ChordalRing(mirrored, false));
  system_->nodes[GPU].nodes[7].gpu.dev = 9;  // the partner sums no longer land on a dev
  ASSERT_EQ(ncclSuccess, parseChordalRing(system_.get(), ring.get()));
  EXPECT_EQ(0, ring->nChannels);

  Use(XgmiSystem(8));  // 7 links each
  ASSERT_EQ(ncclSuccess, parseChordalRing(system_.get(), ring.get()));
  EXPECT_EQ(0, ring->nChannels);
  Use(XgmiSystem(4));
  ASSERT_EQ(ncclSuccess, parseChordalRing(system_.get(), ring.get()));
  EXPECT_EQ(0, ring->nChannels);
}

TEST_F(GraphRomeModelsMicrotest, ParseRomeSystem_DumpsTheModelWhenAsked) {
  UseModel(romeTopoModels[43]);
  TempFile file;
  RcclUnitTesting::ScopedEnv dumpFile("RCCL_DUMP_ROME_MODEL_FILE", file.path());
  rcclRomeModel topo;
  char pattern[256];

  ASSERT_EQ(ncclSuccess, parseRomeSystem(system_.get(), &topo, pattern, nullptr));

  std::string text = file.Read();
  EXPECT_NE(std::string::npos, text.find(".nGpus = 8, .nCpus = 2, .nNics = 4, .nLinks = 7"));
  EXPECT_NE(std::string::npos, text.find(".pattern = \"4242\""));
  EXPECT_NE(std::string::npos, text.find("PATH_PXB"));
  RcclUnitTesting::ScopedEnv badDumpFile("RCCL_DUMP_ROME_MODEL_FILE", "/nonexistent/model.txt");
  ASSERT_EQ(ncclSuccess, parseRomeSystem(system_.get(), &topo, pattern, nullptr));
  EXPECT_TRUE(Logged("Unable to open /nonexistent/model.txt, not dumping Rome model"));
}

// --- Ring and tree strings

class GraphRomeModelsParseMicrotest : public GraphRomeModelsMicrotest {
 protected:
  GraphPtr ring_ = NewGraph(0, NCCL_TOPO_PATTERN_RING);

  void SetUp() override {
    GraphRomeModelsMicrotest::SetUp();
    Use(XgmiSystem(4, 1, true), 8);
    system_->totalBw = 120;
  }
};

TEST_F(GraphRomeModelsParseMicrotest, ParseGraph_KeepsOnlyChannelsThatFitTheNode) {
  ASSERT_EQ(ncclSuccess, parseGraph("0 1 2 3|0 1 2|0 1 2 9|0 1 1 3|3 2 1 0", system_.get(), ring_.get(), nullptr,
                                    nullptr, 0));

  ASSERT_EQ(2, ring_->nChannels);
  EXPECT_EQ(std::vector<int>({0, 1, 2, 3}), Channel(ring_.get(), 0));
  EXPECT_EQ(std::vector<int>({3, 2, 1, 0}), Channel(ring_.get(), 1));
  EXPECT_FLOAT_EQ(60, ring_->bwIntra);
  EXPECT_EQ(system_->nodes[NET].nodes[0].id, ring_->inter[0]);  // NICs dealt round robin
  EXPECT_EQ(system_->nodes[NET].nodes[1].id, ring_->inter[1]);
  EXPECT_EQ(system_->nodes[NET].nodes[2].id, ring_->inter[3]);
}

TEST_F(GraphRomeModelsParseMicrotest, ParseGraph_NicsMapsAndReversal) {
  int gpuMap[4] = {3, 2, 1, 0}, netMap[4] = {1, 0, 3, 2};

  ASSERT_EQ(ncclSuccess, parseGraph("N0 0 1 2 3 N1|N2 0 1 2 3|N12 0 1 2 3 N3", system_.get(), ring_.get(), gpuMap,
                                    netMap, 1));

  ASSERT_EQ(2, ring_->nIntraChannels);
  ASSERT_EQ(1, ring_->nChannels);  // N12 does not exist
  EXPECT_EQ(std::vector<int>({0, 1, 2, 3}), Channel(ring_.get(), 0));  // mapped 3 2 1 0, then reversed
  EXPECT_EQ(0, ring_->inter[0]);  // N1 mapped to 0, swapped in by the reversal
  EXPECT_EQ(1, ring_->inter[1]);
  EXPECT_EQ('N' + 0, ring_->intraNets[0]);
}

TEST_F(GraphRomeModelsParseMicrotest, ParseGraph_ReversalCanBeDisabled) {
  g_params["RCCL_MODEL_REVERSAL_DISABLE"] = 1;

  ASSERT_EQ(ncclSuccess, parseGraph("0 1 2 3", system_.get(), ring_.get(), nullptr, nullptr, 1));

  EXPECT_EQ(std::vector<int>({0, 1, 2, 3}), Channel(ring_.get(), 0));
}

TEST_F(GraphRomeModelsParseMicrotest, ParseGraph_TreeGraphTakesTheSecondGpusNic) {
  GraphPtr tree = NewGraph(1, NCCL_TOPO_PATTERN_BALANCED_TREE);

  ASSERT_EQ(ncclSuccess, parseGraph("0 2 1 3", system_.get(), tree.get(), nullptr, nullptr, 0));

  EXPECT_EQ(system_->nodes[NET].nodes[2].id, tree->inter[1]);
}

TEST_F(GraphRomeModelsParseMicrotest, ParseGraph_UnknownDevIsAnError) {
  int gpuMap[4] = {0, 1, 2, 7};

  EXPECT_EQ(ncclInternalError, parseGraph("0 1 2 3", system_.get(), ring_.get(), gpuMap, nullptr, 0));
}

TEST_F(GraphRomeModelsParseMicrotest, ParseGraph_StopsAtTooManyNics) {
  ASSERT_EQ(ncclSuccess, parseGraph("N0 N1 N2 N3 N0 N1 N2 N3 N0 0 1 2 3", system_.get(), ring_.get(), nullptr,
                                    nullptr, 0));

  EXPECT_EQ(0, ring_->nChannels);
}

TEST_F(GraphRomeModelsMicrotest, ParseOptions_SetsTheKnownKeys) {
  Use(XgmiSystem(1));

  parseOptions(system_.get(), "netGdrLevel=PHB,pivotA2AEnabled=1,pivotA2ANumBiRings=3,tuning=4,ll128Enabled=1");
  parseOptions(system_.get(), "baseBw=12.5, treeDefined=1,unknown=7");

  EXPECT_EQ(PATH_PHB, system_->netGdrLevel);
  EXPECT_TRUE(system_->pivotA2AEnabled);
  EXPECT_EQ(3, system_->pivotA2ANumBiRings);
  EXPECT_EQ(4, system_->tuning);
  EXPECT_TRUE(system_->ll128Enabled);
  EXPECT_FLOAT_EQ(12.5, system_->baseBw);
  EXPECT_TRUE(system_->treeDefined);
}

TEST_F(GraphRomeModelsMicrotest, ParseOptions_BadGdrLevelAndEmptyString) {
  Use(XgmiSystem(1));
  system_->tuning = 4;

  parseOptions(system_.get(), "netGdrLevel=SYS");
  EXPECT_EQ(PATH_SYS, system_->netGdrLevel);
  parseOptions(system_.get(), "netGdrLevel=FOO");
  parseOptions(system_.get(), "");

  EXPECT_EQ(-2, system_->netGdrLevel);
  EXPECT_TRUE(Logged("invalid netGdrLevel: FOO"));
  EXPECT_EQ(4, system_->tuning);
}

// Options are split into at most 10 tokens, so pairs past the fifth are dropped.
// Pins AICOMRCCL-3031; a fix flips this pin.
TEST_F(GraphRomeModelsMicrotest, ParseOptions_KnownBug_IgnoresPairsPastTheFifth) {
  Use(XgmiSystem(1));

  parseOptions(system_.get(), "a=1,b=1,c=1,d=1,tuning=4,treeDefined=1");

  EXPECT_EQ(4, system_->tuning);
  EXPECT_FALSE(system_->treeDefined);
}

TEST_F(GraphRomeModelsMicrotest, CheckOption_ReadsABooleanByName) {
  EXPECT_TRUE(checkOption("tuning=2,noCpuCheck=1", "noCpuCheck"));
  EXPECT_FALSE(checkOption("noCpuCheck=0", "noCpuCheck"));
  EXPECT_FALSE(checkOption("tuning=2", "noCpuCheck"));
  EXPECT_FALSE(checkOption("", "noCpuCheck"));
}

TEST_F(GraphRomeModelsMicrotest, SetPresetIdx_RejectsAConflictingModel) {
  Use(XgmiSystem(1));

  ASSERT_EQ(ncclSuccess, rcclTopoSetPresetRomeModelIdx(system_.get(), 5));
  ASSERT_EQ(ncclSuccess, rcclTopoSetPresetRomeModelIdx(system_.get(), 5));
  EXPECT_EQ(ncclInvalidUsage, rcclTopoSetPresetRomeModelIdx(system_.get(), 6));

  EXPECT_TRUE(Logged("conflicting preset Rome topology model index on this node (already 5, attempted 6)"));
}

TEST_F(GraphRomeModelsMicrotest, CheckAlltoallWidth_ZeroUnlessEveryPairHasTheSameWidth) {
  rcclRomeModel m = romeTopoModels[44];
  EXPECT_EQ(1, checkAlltoallWidth(&m));
  m.connMatrix[1] = 2;
  EXPECT_EQ(0, checkAlltoallWidth(&m));
  m.connMatrix[1] = 0;
  EXPECT_EQ(0, checkAlltoallWidth(&m));
}

// --- Ring and tree string bounds, on hand-built systems

class RomeModelsParseScene {
 public:
  explicit RomeModelsParseScene(int nGpus) : system_(new ncclTopoSystem{}) {
    system_->nodes[GPU].count = nGpus;
    for (int i = 0; i < nGpus; i++) {
      system_->nodes[GPU].nodes[i].gpu.dev = i;
      // Ranks are deliberately not equal to the dev numbers, so a test that reads
      // graph->intra cannot pass by accident if the translation step is skipped.
      system_->nodes[GPU].nodes[i].gpu.rank = kRankBase + i;
    }
  }

  static constexpr int kRankBase = 100;

  ncclTopoSystem* system() { return system_.get(); }

 private:
  std::unique_ptr<ncclTopoSystem> system_;
};

constexpr int kGpus = 16;

class RomeModelsParseGraphLightMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    scene_ = std::make_unique<RomeModelsParseScene>(kGpus);
    graph_ = std::make_unique<ncclTopoGraph>();
    std::memset(graph_.get(), 0, sizeof(*graph_));
  }

  void TearDown() override { ResetGraphFakes(); }

  ncclTopoSystem* system() { return scene_->system(); }
  ncclTopoGraph* graph() { return graph_.get(); }

 private:
  std::unique_ptr<RomeModelsParseScene> scene_;
  std::unique_ptr<ncclTopoGraph> graph_;
};

TEST_F(RomeModelsParseGraphLightMicrotest, NullString_IsRejected) {
  EXPECT_EQ(parseGraphLight(nullptr, system(), graph(), nullptr), ncclInvalidArgument);
}

TEST_F(RomeModelsParseGraphLightMicrotest, WellFormedTree_IsTranscribedToTreeBase) {
  const char* tree = "(4(2(14)(1))(6(5)))";
  graph()->treeBase[1][0] = 'x';  // the fixture zeroes the graph, so dirty the terminator

  ASSERT_EQ(parseGraphLight(tree, system(), graph(), nullptr), ncclSuccess);

  EXPECT_STREQ(graph()->treeBase[0], tree);
  EXPECT_EQ(graph()->treeBase[1][0], 0);
}

TEST_F(RomeModelsParseGraphLightMicrotest, MultipleSegments_ProduceOneTreeBaseRowEach) {
  graph()->treeBase[3][0] = 'Z';  // the fixture zeroes the graph, so dirty the terminator

  ASSERT_EQ(parseGraphLight("(1(2))|(3(4))|(5(6))", system(), graph(), nullptr), ncclSuccess);

  EXPECT_STREQ(graph()->treeBase[0], "(1(2))");
  EXPECT_STREQ(graph()->treeBase[1], "(3(4))");
  EXPECT_STREQ(graph()->treeBase[2], "(5(6))");
  EXPECT_EQ(graph()->treeBase[3][0], 0);
}

TEST_F(RomeModelsParseGraphLightMicrotest, EmptyString_ProducesEmptyTreeBase) {
  graph()->treeBase[0][0] = 'x';

  ASSERT_EQ(parseGraphLight("", system(), graph(), nullptr), ncclSuccess);

  EXPECT_EQ(graph()->treeBase[0][0], 0);
}

TEST_F(RomeModelsParseGraphLightMicrotest, MoreNumbersInSegmentThanMaxNodes_IsRejected) {
  std::string tooManyNumbers;
  for (int i = 0; i <= NCCL_TOPO_MAX_NODES; i++) tooManyNumbers += "(0)";

  EXPECT_EQ(parseGraphLight(tooManyNumbers.c_str(), system(), graph(), nullptr),
            ncclInvalidUsage);
}

TEST_F(RomeModelsParseGraphLightMicrotest, GpuNumberAsWideAsAnInt_IsTranscribed) {
  // The largest accepted number, so the guard's boundary itself is pinned. It is also ten
  // digits: one more than the buffer originally held (it has been expanded to accomodate ints)
  constexpr int kWideDev = INT_MAX;
  system()->nodes[GPU].nodes[kGpus - 1].gpu.dev = kWideDev;
  const std::string tree = "(" + std::to_string(kWideDev) + ")";

  ASSERT_EQ(parseGraphLight(tree.c_str(), system(), graph(), nullptr), ncclSuccess);

  EXPECT_STREQ(graph()->treeBase[0], tree.c_str());
}

TEST_F(RomeModelsParseGraphLightMicrotest, MostNegativeGpuNumberFromGpuMap_IsTranscribed) {
  system()->nodes[GPU].nodes[kGpus - 1].gpu.dev = INT_MIN;
  std::vector<int> mapToMostNegative(kGpus, 0);
  mapToMostNegative[0] = INT_MIN;

  ASSERT_EQ(parseGraphLight("(0)", system(), graph(), mapToMostNegative.data()), ncclSuccess);

  EXPECT_STREQ(graph()->treeBase[0], "(-2147483648)");
}

TEST_F(RomeModelsParseGraphLightMicrotest, GpuNumbersAreTranslatedThroughGpuMap) {
  std::vector<int> reversingMap(kGpus);
  for (int i = 0; i < kGpus; i++) reversingMap[i] = kGpus - 1 - i;

  const std::string highestAndLowest =
      "(" + std::to_string(kGpus - 1) + "(0))";  // "(15(0))" for 16 GPUs

  ASSERT_EQ(parseGraphLight(highestAndLowest.c_str(), system(), graph(), reversingMap.data()),
            ncclSuccess);

  EXPECT_STREQ(graph()->treeBase[0], "(0(15))");
}

TEST_F(RomeModelsParseGraphLightMicrotest, GpuNotInTheSystem_IsRejected) {
  const std::string absentGpu = "(" + std::to_string(kGpus + 4) + ")";

  EXPECT_EQ(parseGraphLight(absentGpu.c_str(), system(), graph(), nullptr), ncclInternalError);
}

TEST_F(RomeModelsParseGraphLightMicrotest, GpuNumberOutsideGpuMap_IsRejected) {
  // Exactly as long as the system has GPUs, matching the shortest table any caller
  // passes, and heap-allocated so a read past it is one ASan can see.
  std::vector<int> gpuMap(kGpus);
  for (int i = 0; i < kGpus; i++) gpuMap[i] = i;

  // The smallest number the table cannot hold, so the limit itself is pinned.
  const std::string justPastEnd = "(" + std::to_string(kGpus) + ")";

  EXPECT_EQ(parseGraphLight(justPastEnd.c_str(), system(), graph(), gpuMap.data()),
            ncclInvalidUsage);
}

TEST_F(RomeModelsParseGraphLightMicrotest, TrailingNumberWithNoClosingParen_IsRejected) {
  // Long enough that std::string puts it on the heap, so the overread is a heap
  // overflow ASan can see rather than an unremarkable read of adjacent stack.
  const std::string truncatedTree = "(1)(2)(3)(4)(5)(6";

  EXPECT_EQ(parseGraphLight(truncatedTree.c_str(), system(), graph(), nullptr),
            ncclInvalidUsage);
}

TEST_F(RomeModelsParseGraphLightMicrotest, StrayCharacterInsideATree_IsRejected) {
  // Each of these parses two GPU numbers but transcribes only the first: the scan for the
  // end of a number swallows the stray character and everything after it up to the next
  // parenthesis, so the second number is silently dropped from treeBase.
  for (const char* strayCharacter : {"(1x2)", "(0(1)(2,3))", "(-1)"}) {
    SCOPED_TRACE(strayCharacter);

    EXPECT_EQ(parseGraphLight(strayCharacter, system(), graph(), nullptr), ncclInvalidUsage);
  }
}

TEST_F(RomeModelsParseGraphLightMicrotest, MoreTokensThanNumbersParsed_IsRejected) {
  EXPECT_EQ(parseGraphLight("(5)(6)|(0)(x)", system(), graph(), nullptr), ncclInvalidUsage);
}

TEST_F(RomeModelsParseGraphLightMicrotest, MoreSegmentsThanTreeBaseRows_IsRejected) {
  // Exactly one segment too many: with one segment per row there is no row left to
  // terminate the list, so this is the smallest input that must be refused and it pins
  // the limit itself rather than merely being somewhere beyond it.
  const size_t rows = sizeof(graph()->treeBase) / sizeof(graph()->treeBase[0]);
  std::string tooManySegments = "(0)";
  for (size_t i = 1; i < rows; i++) tooManySegments += "|(0)";

  EXPECT_EQ(parseGraphLight(tooManySegments.c_str(), system(), graph(), nullptr),
            ncclInvalidUsage);
}

TEST_F(RomeModelsParseGraphLightMicrotest, NumberWiderThanIntRange_IsRejected) {
  // INT_MAX + 1: the smallest rejected number, so a guard that drifted by one would fail here.
  EXPECT_EQ(parseGraphLight("(2147483648)", system(), graph(), nullptr), ncclInvalidUsage);
}

TEST_F(RomeModelsParseGraphLightMicrotest, SegmentLongerThanTreeBaseRow_IsRejected) {
  const size_t rowLen = sizeof(graph()->treeBase[0]);

  // Overruns while copying a parenthesis.
  const std::string parens(rowLen, '(');
  // Overruns while copying a digit of a GPU number: the parentheses fill the row, so the
  // digits of "14" land beyond it. The trailing ')' keeps the number's end-of-token scan
  // inside the string.
  // Note the row can never run out with a digit as its last written character: a number
  // is only transcribed once its closing parenthesis has been found inside the segment,
  // and that parenthesis is then written too. So the digit check cannot be the one that
  // catches the final overrun, and only its presence is pinned here, not its exact
  // boundary.
  const std::string endsInNumber = std::string(rowLen - 1, '(') + "14)";

  for (const std::string& tooLong : {parens, endsInNumber}) {
    SCOPED_TRACE("segment length " + std::to_string(tooLong.size()));
    const char sentinel = 'Z';
    graph()->treeBase[1][0] = sentinel;

    EXPECT_EQ(parseGraphLight(tooLong.c_str(), system(), graph(), nullptr), ncclInvalidUsage);
    // Returning an error is not enough: the row after this one must be untouched, or the
    // overrun happened and was merely reported afterwards.
    EXPECT_EQ(graph()->treeBase[1][0], sentinel);
  }
}

// The shipping preset topologies go through the same guarded path as RCCL_TREES does
// (rome_models.cc passes romeTopoModels[i].treeBase with a non-NULL gpu_map), under
// NCCLCHECK, so a guard that rejected one of them would fail ncclCommInitRank outright.
TEST(RomeModelsPresetTopologiesMicrotest, EveryPresetTreeBase_IsAccepted) {
  constexpr size_t kModels = sizeof(romeTopoModels) / sizeof(romeTopoModels[0]);

  for (size_t i = 0; i < kModels; i++) {
    const rcclRomeModel& model = romeTopoModels[i];
    if (model.treeBase == nullptr) continue;
    SCOPED_TRACE("romeTopoModels[" + std::to_string(i) + "], treeBase " + model.treeBase);

    RomeModelsParseScene scene(model.nGpus);
    ncclTopoGraph graph;
    std::memset(&graph, 0, sizeof(graph));
    // The production map is a permutation of the GPU numbers; identity is the
    // representative one, and its length is what bounds the guarded lookup.
    std::vector<int> identityMap(model.nGpus);
    for (int g = 0; g < model.nGpus; g++) identityMap[g] = g;

    EXPECT_EQ(parseGraphLight(model.treeBase, scene.system(), &graph, identityMap.data()),
              ncclSuccess);
  }

  ResetGraphFakes();
}

// ---------------------------------------------------------------------------
// parseGraph: accepted-input contracts.
//
// These pin what a well-formed NCCL_RINGS must keep doing. Every bound added for
// ROCM-31973 is a chance to reject a valid topology by accident, and only a test
// that asserts on accepted input can catch that.
// ---------------------------------------------------------------------------

class RomeModelsParseGraphMicrotest : public ::testing::Test {
 protected:
  void SetUp() override { UseGpus(kRingGpus); }

  void TearDown() override { ResetGraphFakes(); }

  // Rebuild the scene and the graph for a GPU count other than the default, so a
  // test that needs a wider system still reads it through system() and graph().
  void UseGpus(int nGpus) {
    scene_ = std::make_unique<RomeModelsParseScene>(nGpus);
    scene_->system()->totalBw = kTotalBw;
    graph_ = std::make_unique<ncclTopoGraph>();
    std::memset(graph_.get(), 0, sizeof(*graph_));
  }

  ncclTopoSystem* system() { return scene_->system(); }
  ncclTopoGraph* graph() { return graph_.get(); }

  // The ranks the scene gives devs 0..3, in dev order.
  static std::vector<int> RanksOf(std::vector<int> devs) {
    for (int& d : devs) d += RomeModelsParseScene::kRankBase;
    return devs;
  }

  std::vector<int> IntraOfChannel(int channel) const {
    return std::vector<int>(graph_->intra + channel * kRingGpus,
                            graph_->intra + (channel + 1) * kRingGpus);
  }

  static constexpr int kRingGpus = 4;
  static constexpr float kTotalBw = 120.0f;

 private:
  std::unique_ptr<RomeModelsParseScene> scene_;
  std::unique_ptr<ncclTopoGraph> graph_;
};

TEST_F(RomeModelsParseGraphMicrotest, NullRingString_IsRejected) {
  // search.cc only reaches parseGraph when NCCL_RINGS is set, so this is
  // defensive rather than reachable from the environment -- but parseGraphLight
  // already rejects a null string and the two parsers should not disagree.
  EXPECT_EQ(parseGraph(nullptr, system(), graph(), nullptr, nullptr, 0), ncclInvalidArgument);
}

TEST_F(RomeModelsParseGraphMicrotest, WellFormedRings_AreTranscribedToIntra) {
  ASSERT_EQ(parseGraph("0 1 2 3|3 2 1 0", system(), graph(), nullptr, nullptr, 0), ncclSuccess);

  EXPECT_EQ(graph()->nChannels, 2);
  EXPECT_EQ(IntraOfChannel(0), RanksOf({0, 1, 2, 3}));
  EXPECT_EQ(IntraOfChannel(1), RanksOf({3, 2, 1, 0}));
  // Bandwidth is split across the channels that were accepted.
  EXPECT_FLOAT_EQ(graph()->bwIntra, kTotalBw / 2);
  EXPECT_FLOAT_EQ(graph()->bwInter, kTotalBw / 2);
}

TEST_F(RomeModelsParseGraphMicrotest, ReversedRing_IsTranscribedBackwards) {
  ASSERT_EQ(parseGraph("0 1 2 3", system(), graph(), nullptr, nullptr, 1), ncclSuccess);

  EXPECT_EQ(graph()->nChannels, 1);
  EXPECT_EQ(IntraOfChannel(0), RanksOf({3, 2, 1, 0}));
}

TEST_F(RomeModelsParseGraphMicrotest, GpuNumbersAreTranslatedThroughGpuMap) {
  // A reversing map, so a result equal to the input cannot pass by accident.
  std::vector<int> reversingMap(kRingGpus);
  for (int i = 0; i < kRingGpus; i++) reversingMap[i] = kRingGpus - 1 - i;

  ASSERT_EQ(parseGraph("0 1 2 3", system(), graph(), reversingMap.data(), nullptr, 0), ncclSuccess);

  EXPECT_EQ(IntraOfChannel(0), RanksOf({3, 2, 1, 0}));
}

TEST_F(RomeModelsParseGraphMicrotest, ChannelWithWrongGpuCount_IsIgnored) {
  // Rings for other GPU counts are skipped, not rejected. The short segment follows a
  // full one, so its stale gpus[3] keeps it otherwise valid and only the count check
  // can drop it.
  ASSERT_EQ(parseGraph("0 1 2 3|0 1 2", system(), graph(), nullptr, nullptr, 0), ncclSuccess);

  EXPECT_EQ(graph()->nChannels, 1);
  EXPECT_EQ(IntraOfChannel(0), RanksOf({0, 1, 2, 3}));
}

TEST_F(RomeModelsParseGraphMicrotest, DuplicateGpuInChannel_IsIgnored) {
  ASSERT_EQ(parseGraph("0 0 1 2|0 1 2 3", system(), graph(), nullptr, nullptr, 0), ncclSuccess);

  EXPECT_EQ(graph()->nChannels, 1);
  EXPECT_EQ(IntraOfChannel(0), RanksOf({0, 1, 2, 3}));
}

TEST_F(RomeModelsParseGraphMicrotest, MultiDigitGpuNumbers_AreParsed) {
  UseGpus(12);

  ASSERT_EQ(parseGraph("0 1 2 3 4 5 6 7 8 9 10 11", system(), graph(), nullptr, nullptr, 0),
            ncclSuccess);

  EXPECT_EQ(graph()->nChannels, 1);
  EXPECT_EQ(graph()->intra[10], RomeModelsParseScene::kRankBase + 10);
  EXPECT_EQ(graph()->intra[11], RomeModelsParseScene::kRankBase + 11);
}

// A ring list of `channels` identical well-formed segments for the 4-GPU scene.
std::string RingsWithChannels(int channels) {
  std::string rings = "0 1 2 3";
  for (int i = 1; i < channels; i++) rings += "|0 1 2 3";
  return rings;
}

// nIntraChannels sits immediately after inter[MAXCHANNELS * 2], so the first
// channel past the end writes exactly onto it. The overrun therefore stays inside
// the same ncclTopoGraph and no sanitizer can see it -- this sentinel is the only
// thing that can catch it.
constexpr int kIntraChannelsSentinel = 4242;

TEST_F(RomeModelsParseGraphMicrotest, MoreChannelsThanMaxChannels_IsRejected) {
  // One NET, so the inter[] writes that run off the end actually happen.
  system()->nodes[NET].count = 1;
  system()->nodes[NET].nodes[0].id = 77;
  graph()->nIntraChannels = kIntraChannelsSentinel;

  // Exactly one channel too many, so an off-by-one in the new bound fails here
  // rather than only something far past the limit.
  const std::string tooMany = RingsWithChannels(MAXCHANNELS + 1);

  EXPECT_EQ(parseGraph(tooMany.c_str(), system(), graph(), nullptr, nullptr, 0),
            ncclInvalidUsage);
  // Returning an error is not enough on its own: before the bound existed this
  // returned ncclSuccess and left nIntraChannels holding the NET id.
  EXPECT_EQ(graph()->nIntraChannels, kIntraChannelsSentinel);
  EXPECT_LE(graph()->nChannels, MAXCHANNELS);
}

// A single channel naming `count` GPU numbers, for a scene with that many GPUs.
std::string ChannelWithGpuCount(int count) {
  std::string rings;
  for (int i = 0; i < count; i++) {
    if (i) rings += " ";
    rings += std::to_string(i);
  }
  return rings;
}

TEST_F(RomeModelsParseGraphMicrotest, MoreGpusInOneChannelThanMaxNodes_IsRejected) {
  UseGpus(NCCL_TOPO_MAX_NODES);

  // Exactly one GPU number too many, so an off-by-one in the bound fails here.
  // Before the bound, the 145th number was written to gpus[144] -- one int past
  // the array -- which ASan reports as a stack-buffer-overflow at offset 608.
  const std::string tooManyGpus = ChannelWithGpuCount(NCCL_TOPO_MAX_NODES + 1);

  EXPECT_EQ(parseGraph(tooManyGpus.c_str(), system(), graph(), nullptr, nullptr, 0),
            ncclInvalidUsage);
}

TEST_F(RomeModelsParseGraphMicrotest, ExactlyMaxNodesGpusInOneChannel_IsAccepted) {
  UseGpus(NCCL_TOPO_MAX_NODES);

  // The widest channel that still fits, so a bound tightened by one breaks here.
  const std::string exactlyFull = ChannelWithGpuCount(NCCL_TOPO_MAX_NODES);

  ASSERT_EQ(parseGraph(exactlyFull.c_str(), system(), graph(), nullptr, nullptr, 0),
            ncclSuccess);

  EXPECT_EQ(graph()->nChannels, 1);
  EXPECT_EQ(graph()->intra[NCCL_TOPO_MAX_NODES - 1],
            RomeModelsParseScene::kRankBase + NCCL_TOPO_MAX_NODES - 1);
}

TEST_F(RomeModelsParseGraphMicrotest, GpuNumberWiderThanIntRange_IsRejected) {
  // INT_MAX + 1: the smallest number that cannot be accumulated, so a bound that
  // drifted by one would fail here. Accumulating it overflowed a signed int,
  // which is undefined, and the wrapped negative was then quietly swallowed by
  // the g < 0 test -- the channel vanished instead of being reported.
  EXPECT_EQ(parseGraph("2147483648", system(), graph(), nullptr, nullptr, 0), ncclInvalidUsage);
}

TEST_F(RomeModelsParseGraphMicrotest, GpuNumberAtIntMax_IsNotRejected) {
  // The largest number that still fits. It names no GPU this system has, so the
  // channel is ignored -- but ignoring is not rejecting, and a bound tightened by
  // one would turn this into an error.
  EXPECT_EQ(parseGraph("2147483647", system(), graph(), nullptr, nullptr, 0), ncclSuccess);
  EXPECT_EQ(graph()->nChannels, 0);
}

// A NET slot the ring did not name. nets[] holds a NET number biased by 'N', so an
// absent slot must translate to a negative NET id, which is what the translation
// and the inter[] checks both read as "no NET here".
bool IsAbsentNetSlot(int slot) { return slot - 'N' < 0; }

TEST_F(RomeModelsParseGraphMicrotest, NetSlotsTheRingDidNotName_AreNotCarriedOver) {
  system()->nodes[NET].count = 2;
  system()->nodes[NET].nodes[0].id = 50;
  system()->nodes[NET].nodes[1].id = 51;

  // The first segment names five NETs, which is odd, so it is ignored as a
  // channel -- but it still writes them into nets[]. The second segment names
  // only its two end NETs, leaving the slots between them holding the first
  // segment's numbers. Seeding the stale state from an earlier iteration of the
  // same call keeps this deterministic; the raw uninitialised values this used to
  // expose varied run to run.
  ASSERT_EQ(parseGraph("N5 0 N6 1 N7 2 N8 3 N9|N0 0 1 2 3 N1", system(), graph(), nullptr,
                       nullptr, 0),
            ncclSuccess);

  ASSERT_EQ(graph()->nChannels, 1);
  EXPECT_EQ(graph()->inter[0], 0);
  EXPECT_EQ(graph()->inter[1], 1);

  // Slots 1, 3 and 5 are where the ignored segment's N6, N7 and N8 landed.
  for (int slot : {1, 2, 3, 4, 5, 6}) {
    SCOPED_TRACE("intraNets slot " + std::to_string(slot));
    EXPECT_TRUE(IsAbsentNetSlot(graph()->intraNets[slot]))
        << "slot holds NET " << graph()->intraNets[slot] - 'N';
  }
}

TEST_F(RomeModelsParseGraphMicrotest, ChannelWithUnnamedEndNets_IsIgnored) {
  system()->nodes[NET].count = 2;
  system()->nodes[NET].nodes[0].id = 50;
  system()->nodes[NET].nodes[1].id = 51;

  // The second segment names its NETs between the GPUs rather than at the ends,
  // so the first end slot stays absent. An absent slot reads as a negative NET id,
  // and only a check for that can stop it reaching inter[] -- the limit on its own
  // sees a number below nnets and lets it through.
  ASSERT_EQ(parseGraph("N0 0 1 2 3 N1|0 N0 1 2 N1 3", system(), graph(), nullptr, nullptr, 0),
            ncclSuccess);

  ASSERT_EQ(graph()->nChannels, 1);
  EXPECT_EQ(graph()->inter[0], 0);
  EXPECT_EQ(graph()->inter[1], 1);
  // The ignored channel must not have left a NET id behind. ncclTopoIdToNetDev
  // fails multi-node init on one of these.
  EXPECT_GE(graph()->inter[2], 0);
  EXPECT_GE(graph()->inter[3], 0);
}

TEST_F(RomeModelsParseGraphMicrotest, ChannelWithAnUnnamedTrailingNet_IsIgnored) {
  system()->nodes[NET].count = 2;
  system()->nodes[NET].nodes[0].id = 50;
  system()->nodes[NET].nodes[1].id = 51;

  // The second segment names its leading NET but puts the other before the last
  // GPU instead of after it, so only the trailing end slot is absent. The leading
  // slot is valid, so the checks on it all pass and the one on the trailing slot
  // is the only thing that can drop the channel.
  ASSERT_EQ(parseGraph("N0 0 1 2 3 N1|N0 0 1 2 N1 3", system(), graph(), nullptr, nullptr, 0),
            ncclSuccess);

  ASSERT_EQ(graph()->nChannels, 1);
  EXPECT_EQ(graph()->inter[0], 0);
  EXPECT_EQ(graph()->inter[1], 1);
}

// A channel naming every GPU the array can hold, so the NET slot index derived
// from the GPU count (2 * gpu - 1) sits at the very end of nets[].
std::string WidestChannelThenNets(const char* trailingNets) {
  return ChannelWithGpuCount(NCCL_TOPO_MAX_NODES) + trailingNets;
}

TEST_F(RomeModelsParseGraphMicrotest, MoreNetsInOneChannelThanNetsArrayHolds_IsRejected) {
  UseGpus(NCCL_TOPO_MAX_NODES);
  system()->nodes[NET].count = 2;

  // The existing per-channel limit is the runtime ngpus * 2, not the size of
  // nets[], and it used > where the array needs >=. At the widest GPU count the
  // two coincide, so the second NET here wrote nets[288] -- one int past the end,
  // which ASan reports at offset 1888. This is the smallest input that reaches it.
  const std::string tooManyNets = WidestChannelThenNets(" N5 N6");

  EXPECT_EQ(parseGraph(tooManyNets.c_str(), system(), graph(), nullptr, nullptr, 0),
            ncclInvalidUsage);
}

TEST_F(RomeModelsParseGraphMicrotest, NetInTheLastSlotOfTheNetsArray_IsAccepted) {
  UseGpus(NCCL_TOPO_MAX_NODES);
  system()->nodes[NET].count = 2;

  // Writes nets[287], the last slot that exists, so a bound tightened by one
  // breaks here.
  const std::string exactlyFull = WidestChannelThenNets(" N5");

  EXPECT_EQ(parseGraph(exactlyFull.c_str(), system(), graph(), nullptr, nullptr, 0),
            ncclSuccess);
}

// NET numbers are held biased by 'N', so the widest one that still fits an int is
// INT_MAX - 'N' rather than INT_MAX.
constexpr int kWidestNetNumber = INT_MAX - 'N';

TEST_F(RomeModelsParseGraphMicrotest, NetNumberWiderThanIntRange_IsRejected) {
  system()->nodes[NET].count = 2;

  // One past the widest NET number that can be held. Accumulating it overflowed
  // a signed int, which is undefined; the result was then quietly discarded by
  // the "not a NET this system has" test and the ring silently lost a channel.
  const std::string tooWide = "N" + std::to_string(kWidestNetNumber + 1) + " 0 1 2 3 N1";

  EXPECT_EQ(parseGraph(tooWide.c_str(), system(), graph(), nullptr, nullptr, 0), ncclInvalidUsage);
}

TEST_F(RomeModelsParseGraphMicrotest, NetNumberAtTheWidestThatFits_IsNotRejected) {
  system()->nodes[NET].count = 2;

  // Exactly the widest NET number that fits. It names no NET this system has, so
  // the channel is ignored -- but a bound tightened by one would make it an error.
  const std::string widest = "N" + std::to_string(kWidestNetNumber) + " 0 1 2 3 N1";

  EXPECT_EQ(parseGraph(widest.c_str(), system(), graph(), nullptr, nullptr, 0), ncclSuccess);
}

TEST_F(RomeModelsParseGraphMicrotest, TreeGraph_ResolvesOutgoingNetFromTheSecondGpu) {
  // graph->id == 1 is the tree graph, whose outgoing NET is looked up per channel
  // from the rank in intra[i * ngpus + 1] rather than taken from the ring string.
  // That index is bounded only by nChannels, which is why it is safe now.
  graph()->id = 1;
  system()->nodes[NET].count = 1;

  std::vector<int> ranksAskedFor;
  ScopedHook getLocalNet(g_romeGetLocalNet, [&ranksAskedFor](struct ncclTopoSystem*, int rank, int channelId,
                                                             int64_t* id, int*) {
    ranksAskedFor.push_back(rank);
    *id = 900 + channelId;
    return ncclSuccess;
  });

  ASSERT_EQ(parseGraph("0 1 2 3|3 2 1 0", system(), graph(), nullptr, nullptr, 0), ncclSuccess);

  ASSERT_EQ(graph()->nChannels, 2);
  // The second GPU of each channel, which differs per channel, so a lookup that
  // ignored the channel would be visible here.
  EXPECT_EQ(ranksAskedFor, RanksOf({1, 2}));
  EXPECT_EQ(graph()->inter[1], 900);
  EXPECT_EQ(graph()->inter[3], 901);
}

TEST_F(RomeModelsParseGraphMicrotest, NoMatchingChannel_LeavesBandwidthAtZero) {
  // Callers test graph->nChannels after calling, so a ring list that matches
  // nothing is not an error -- but it must not leave a bandwidth behind either.
  // Seeded with a sentinel because the fixture zeroes the graph, and "bw is 0"
  // would otherwise be true without parseGraph doing anything at all.
  graph()->bwIntra = 99.0f;
  graph()->bwInter = 99.0f;

  // Two GPUs on a four-GPU system: no segment matches, so nChannels stays 0 and
  // totalBw / nChannels used to divide by zero.
  ASSERT_EQ(parseGraph("0 1", system(), graph(), nullptr, nullptr, 0), ncclSuccess);

  EXPECT_EQ(graph()->nChannels, 0);
  EXPECT_FLOAT_EQ(graph()->bwIntra, 0.0f);
  EXPECT_FLOAT_EQ(graph()->bwInter, 0.0f);
}

TEST_F(RomeModelsParseGraphMicrotest, ExactlyMaxChannels_IsAccepted) {
  system()->nodes[NET].count = 1;
  system()->nodes[NET].nodes[0].id = 77;
  graph()->nIntraChannels = kIntraChannelsSentinel;

  // The largest list that still fits, so a bound tightened by one breaks here.
  const std::string exactlyFull = RingsWithChannels(MAXCHANNELS);

  ASSERT_EQ(parseGraph(exactlyFull.c_str(), system(), graph(), nullptr, nullptr, 0), ncclSuccess);

  EXPECT_EQ(graph()->nChannels, MAXCHANNELS);
  EXPECT_EQ(graph()->nIntraChannels, kIntraChannelsSentinel);
}

// parseGIOTopos sizes devids, g_hives and rdm at its own NGPUS (16) but admits
// any system with at least that many GPUs, so a wider one overruns all three --
// and parseGraph's gpu_map bound, which trusts ngpus, then reads past rdm. The
// model describes exactly 16 GPUs, so admitting more was never intended.
//
// No GPU->CPU paths are set here, so a gate that admits this system does not just
// overrun -- it dereferences a null paths[CPU] in parseRomeSystem
// (rome_models.cc:4357) and crashes in the ordinary build, not only under ASan.
TEST(RomeModelsGioToposMicrotest, SystemWithMoreGpusThanTheModel_IsIgnored) {
  constexpr int kModelGpus = 16;
  constexpr int kCpus = 2;
  RomeModelsParseScene wider(kModelGpus + 1);
  wider.system()->nodes[CPU].count = kCpus;  // the model admits at most two
  wider.system()->nodes[NET].count = 0;      // and no NICs
  wider.system()->totalBw = 120.0f;

  for (int c = 0; c < kCpus; c++) wider.system()->nodes[CPU].nodes[c].id = c;

  ncclTopoGraph graph;
  std::memset(&graph, 0, sizeof(graph));

  EXPECT_EQ(parseGIOTopos(wider.system(), &graph), ncclSuccess);

  ResetGraphFakes();
}

// The shipping preset ring literals reach parseGraph under NCCLCHECK, so a bound
// that rejected one of them would fail ncclCommInitRank on hardware that matches
// that model. This is what stops the bounds added for ROCM-31973 from being
// tightened past what the presets themselves need.
TEST(RomeModelsPresetTopologiesMicrotest, EveryPresetRingBase_IsAccepted) {
  constexpr size_t kModels = sizeof(romeTopoModels) / sizeof(romeTopoModels[0]);

  for (size_t i = 0; i < kModels; i++) {
    const rcclRomeModel& model = romeTopoModels[i];

    for (const char* rings : {model.ringBase, model.ringTail1, model.ringTail2, model.treeRail}) {
      if (rings == nullptr) continue;
      SCOPED_TRACE("romeTopoModels[" + std::to_string(i) + "], rings " + rings);

      RomeModelsParseScene scene(model.nGpus);
      scene.system()->totalBw = 120.0f;
      scene.system()->nodes[NET].count = model.nNics;
      for (int n = 0; n < model.nNics; n++) scene.system()->nodes[NET].nodes[n].id = n;

      ncclTopoGraph graph;
      std::memset(&graph, 0, sizeof(graph));
      // The production map is a permutation of the GPU numbers; identity is the
      // representative one, and its length is what bounds the guarded lookup.
      std::vector<int> identityMap(model.nGpus);
      for (int g = 0; g < model.nGpus; g++) identityMap[g] = g;
      std::vector<int> identityNets(model.nNics > 0 ? model.nNics : 1);
      for (int n = 0; n < model.nNics; n++) identityNets[n] = n;

      EXPECT_EQ(parseGraph(rings, scene.system(), &graph, identityMap.data(), identityNets.data(),
                           /*reverse=*/0),
                ncclSuccess);
    }
  }

  ResetGraphFakes();
}
}  // namespace
