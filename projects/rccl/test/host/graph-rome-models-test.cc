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
#include <csignal>
#include <functional>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "../common/ScopedEnv.hpp"
#include "fakes/graph_fakes.h"
#include "fakes/param_redirect.h"

#include ROME_MODELS_CC_PATH

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
    if (i == kSixteenNicModel) continue;  // see A2a8P_SixteenNicsMatchWithoutWalkingEveryNicOrdering
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

// Runs body in a child with a 10 s alarm, so a slow NIC search fails instead of hanging the suite.
void ExpectFinishesQuickly(const std::function<bool()>& body) {
  const std::string style = ::testing::GTEST_FLAG(death_test_style);
  ::testing::GTEST_FLAG(death_test_style) = "threadsafe";

  EXPECT_EXIT(
      {
        alarm(10);
        _exit(body() ? 0 : 1);
      },
      ::testing::ExitedWithCode(0), "");
  ::testing::GTEST_FLAG(death_test_style) = style;
}

TEST_F(GraphRomeModelsMicrotest, A2a8P_SixteenNicsMatchWithoutWalkingEveryNicOrdering) {
  for (int pattern : {NCCL_TOPO_PATTERN_RING, NCCL_TOPO_PATTERN_BALANCED_TREE}) {
    UseModel(romeTopoModels[kSixteenNicModel], -1, kAmdCpu);
    ExpectFinishesQuickly([&]() {
      GraphPtr graph = NewGraph(pattern == NCCL_TOPO_PATTERN_RING ? 0 : 1, pattern);
      return parseA2a8P(system_.get(), graph.get(), nullptr) == ncclSuccess &&
             (graph->nChannels > 0 || graph->treeBase[0][0]) && (system_->type & RCCL_TOPO_4P2H_ROME);
    });
  }
}

TEST_F(GraphRomeModelsMicrotest, A2a8P_SixteenNicsThatFitNoModelGiveUpQuickly) {
  UseModel(romeTopoModels[kSixteenNicModel]);
  system_->nodes[GPU].nodes[0].paths[NET][0].type = PATH_DIS;

  ExpectFinishesQuickly([&]() {
    GraphPtr ring = NewGraph(0, NCCL_TOPO_PATTERN_RING);
    return parseA2a8P(system_.get(), ring.get(), nullptr) == ncclSuccess && ring->nChannels == 0;
  });
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

// Under ASan the heap counters do not move, and LeakSanitizer reports the table instead.
TEST_F(GraphRomeModelsMicrotest, A2a8P_NoMatchFreesThePermutationTable) {
  UseModel(romeTopoModels[44], -1, kAmdCpu);
  system_->nodes[DEV].nodes[0].links[0].bw /= 2;  // no model has a narrower link
  ASSERT_EQ(ncclSuccess, ncclTopoComputePaths(system_.get(), nullptr));
  GraphPtr ring = NewGraph(0, NCCL_TOPO_PATTERN_RING);

  size_t before = InUseBytes();
  ASSERT_EQ(ncclSuccess, parseA2a8P(system_.get(), ring.get(), nullptr));
  int64_t leaked = (int64_t)InUseBytes() - (int64_t)before;

  EXPECT_EQ(0, ring->nChannels);
  EXPECT_LT(leaked, (int64_t)(24 * 24 * 8 * sizeof(int)));
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

TEST_F(GraphRomeModelsMicrotest, P1H16P_NoMatchFreesThePermutationTable) {
  rcclRomeModel rewired = romeTopoModels[24];
  auto link = [&](int a, int b, int v) { rewired.connMatrix[a * 16 + b] = rewired.connMatrix[b * 16 + a] = v; };
  link(0, 1, 0), link(8, 9, 0), link(0, 8, 1), link(1, 9, 1);  // same link counts, different shape
  UseModel(rewired);
  GraphPtr ring = NewGraph(0, NCCL_TOPO_PATTERN_RING);

  size_t before = InUseBytes();
  ASSERT_EQ(ncclSuccess, parse1H16P(system_.get(), ring.get()));
  int64_t leaked = (int64_t)InUseBytes() - (int64_t)before;

  EXPECT_EQ(0, ring->nChannels);
  EXPECT_EQ(RCCL_ROME_TOPO_PRESET_MODEL_IDX_NONE, system_->romeTopoModelIdx);
  EXPECT_LT(leaked, (int64_t)(24u * 24 * 24 * 24 * 16 * sizeof(int)));
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

TEST_F(GraphRomeModelsMicrotest, ParseGraph_TreeGraphWithoutNicsLeavesTheInterNetsAlone) {
  Use(XgmiSystem(4), 4);
  system_->totalBw = 120;
  GraphPtr tree = NewGraph(1, NCCL_TOPO_PATTERN_BALANCED_TREE);
  const int64_t untouched = 0x5a5a5a5a5a5a5a5a;
  std::fill(tree->inter, tree->inter + 4, untouched);

  ASSERT_EQ(ncclSuccess, parseGraph("0 2 1 3|3 1 2 0", system_.get(), tree.get(), nullptr, nullptr, 0));

  ASSERT_EQ(2, tree->nChannels);
  for (int i = 0; i < 4; i++) EXPECT_EQ(untouched, tree->inter[i]) << i;
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

// Under ASan a write past gpus[] fails the test as a stack-buffer-overflow.
TEST_F(GraphRomeModelsParseMicrotest, ParseGraph_StopsBeforeOneGpuPastTheLimit) {
  std::string many;
  for (int g = 0; g <= NCCL_TOPO_MAX_NODES; g++) many += std::to_string(g % 4) + " ";

  ASSERT_EQ(ncclSuccess, parseGraph(many.c_str(), system_.get(), ring_.get(), nullptr, nullptr, 0));

  EXPECT_EQ(0, ring_->nChannels);
}

TEST_F(GraphRomeModelsParseMicrotest, ParseGraph_StopsAtMaxChannels) {
  std::string rings;
  for (int c = 0; c <= MAXCHANNELS; c++) rings += "0 1 2 3|";

  ASSERT_EQ(ncclSuccess, parseGraph(rings.c_str(), system_.get(), ring_.get(), nullptr, nullptr, 0));

  EXPECT_EQ(MAXCHANNELS, ring_->nChannels);
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

TEST_F(GraphRomeModelsMicrotest, ParseOptions_ReadsPairsPastTheFifth) {
  Use(XgmiSystem(1));

  parseOptions(system_.get(), "a=1,b=1,c=1,d=1,tuning=4,treeDefined=1");

  EXPECT_EQ(4, system_->tuning);
  EXPECT_TRUE(system_->treeDefined);
}

TEST_F(GraphRomeModelsMicrotest, ParseOptions_IgnoresAKeyWithoutAValue) {
  Use(XgmiSystem(1));

  parseOptions(system_.get(), "tuning=4,treeDefined");

  EXPECT_EQ(4, system_->tuning);
  EXPECT_FALSE(system_->treeDefined);
}

TEST_F(GraphRomeModelsMicrotest, CheckOption_ReadsABooleanByName) {
  EXPECT_TRUE(checkOption("tuning=2,noCpuCheck=1", "noCpuCheck"));
  EXPECT_FALSE(checkOption("noCpuCheck=0", "noCpuCheck"));
  EXPECT_FALSE(checkOption("tuning=2", "noCpuCheck"));
  EXPECT_FALSE(checkOption("", "noCpuCheck"));
  EXPECT_TRUE(checkOption("a=0,b=0,c=0,d=0,e=0,noCpuCheck=1", "noCpuCheck"));
  EXPECT_FALSE(checkOption("noCpuCheck=0,noCpuCheck=1", "noCpuCheck"));
  EXPECT_FALSE(checkOption("noCpuCheck", "noCpuCheck"));
}

// --- NIC ordering search

// The search permuteNetIds replaced: every ordering in swap order, each checked only once complete.
bool PlainPermuteNetIds(int* n, const int* g, int s, int last, rcclRomeModel* ref, rcclRomeModel* topo,
                        bool ignoreNuma) {
  if (s == last) {
    for (int i = 0; i <= last; i++) {
      if (!netIdFits(i, n[i], g, ref, topo, ignoreNuma)) return false;
    }
    return true;
  }
  for (int i = s; i <= last; i++) {
    std::swap(n[s], n[i]);
    if (PlainPermuteNetIds(n, g, s + 1, last, ref, topo, ignoreNuma)) return true;
    std::swap(n[s], n[i]);
  }
  return false;
}

std::unique_ptr<rcclRomeModel> NicModel(int nGpus, int nNics) {
  std::unique_ptr<rcclRomeModel> m(new rcclRomeModel());
  m->nGpus = nGpus;
  m->nNics = nNics;
  return m;
}

std::vector<int> CallerNetOrder(int nNics) {
  std::vector<int> n(nNics);
  for (int m = 0; m < nNics; m++) n[m] = (m + 2) % nNics;
  return n;
}

TEST_F(GraphRomeModelsMicrotest, PermuteNetIds_PicksTheOrderingAPlainPermutationPicks) {
  const uint8_t levels[] = {PATH_PIX, PATH_PXB, PATH_PHB, PATH_SYS, PATH_PXN};
  std::mt19937 rng(3028);
  int matched = 0, unmatched = 0;
  for (int trial = 0; trial < 300; trial++) {
    const int nGpus = 4, nNics = 4 + trial % 4;
    auto ref = NicModel(nGpus, nNics), topo = NicModel(nGpus, nNics);
    std::vector<int> nicOf(nNics), g(nGpus);
    for (int k = 0; k < nNics; k++) nicOf[k] = k;
    for (int j = 0; j < nGpus; j++) g[j] = j;
    std::shuffle(nicOf.begin(), nicOf.end(), rng);
    std::shuffle(g.begin(), g.end(), rng);
    for (int i = 0; i < nNics; i++) {
      ref->nicNuma[i] = rng() % 2;
      topo->nicNuma[nicOf[i]] = ref->nicNuma[i];
      for (int j = 0; j < nGpus; j++) {
        ref->gdrLevel[i * nGpus + j] = levels[rng() % 3];  // few levels, so many NICs fit several positions
        topo->gdrLevel[nicOf[i] * nGpus + g[j]] = ref->gdrLevel[i * nGpus + j];
      }
    }
    if (trial % 2) topo->gdrLevel[rng() % (nNics * nGpus)] = levels[rng() % 5];
    const bool ignoreNuma = trial % 3 == 0;
    if (ignoreNuma) topo->nicNuma[rng() % nNics] ^= 1;  // a NUMA mismatch only these trials may ignore

    std::vector<int> fast = CallerNetOrder(nNics), plain = fast;
    int time = 0;
    bool found = permuteNetIds(fast.data(), g.data(), 0, nNics - 1, ref.get(), topo.get(), &time, ignoreNuma);

    ASSERT_EQ(PlainPermuteNetIds(plain.data(), g.data(), 0, nNics - 1, ref.get(), topo.get(), ignoreNuma), found)
      << trial;
    EXPECT_EQ(plain, fast) << trial;
    (found ? matched : unmatched)++;
  }
  EXPECT_GT(matched, 50);
  EXPECT_GT(unmatched, 10);
}

// Fifteen NICs fit every position but the last: checking positions as they are placed still leaves 15! orderings.
TEST_F(GraphRomeModelsMicrotest, PermuteNetIds_SixteenNicsTakePolynomialSteps) {
  const int nGpus = 8, nNics = 16;
  auto ref = NicModel(nGpus, nNics), topo = NicModel(nGpus, nNics);
  for (int i = 0; i < nNics * nGpus; i++) ref->gdrLevel[i] = topo->gdrLevel[i] = PATH_PHB;
  ref->gdrLevel[(nNics - 1) * nGpus] = PATH_PIX;
  std::vector<int> g = {0, 1, 2, 3, 4, 5, 6, 7};

  ExpectFinishesQuickly([&]() {
    std::vector<int> n = CallerNetOrder(nNics);
    int time = 0;
    if (permuteNetIds(n.data(), g.data(), 0, nNics - 1, ref.get(), topo.get(), &time, false)) return false;
    if (n != CallerNetOrder(nNics) || time > nNics * nNics) return false;

    topo->gdrLevel[5 * nGpus] = PATH_PIX;  // only NIC 5 fits the last position
    time = 0;
    if (!permuteNetIds(n.data(), g.data(), 0, nNics - 1, ref.get(), topo.get(), &time, false)) return false;
    return n[nNics - 1] == 5 && std::set<int>(n.begin(), n.end()).size() == (size_t)nNics && time <= nNics * nNics;
  });
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

}  // namespace
