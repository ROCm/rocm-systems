/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/graph/search.cc: the ring/tree channel search over topologies built from XML,
// graph XML import/export, and the NIC and link-type queries the transports use.

#include <gtest/gtest.h>

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "ScopedHook.h"
#include "../common/ScopedEnv.hpp"
#include "fakes/graph_fakes.h"
#include "fakes/param_redirect.h"

#include SEARCH_CC_PATH

#include "TopoScene.h"

using namespace topo_scene;

namespace {

class SearchMicrotest : public ::testing::Test {
 protected:
  SystemPtr system_;

  void SetUp() override {
    ResetGraphFakes();
    g_params["RCCL_MODEL_MATCHING_DISABLE"] = 1;
  }
  void TearDown() override {
    unsetenv("NCCL_GRAPH_FILE");
    unsetenv("NCCL_GRAPH_DUMP_FILE");
    unsetenv("NCCL_RINGS");
    unsetenv("RCCL_TREES");
    ResetGraphFakes();
  }

  // Builds the system as init.cc does before searching: paths, trim-free, then the search bandwidths.
  ncclTopoSystem* Use(const std::string& text, int nRanks = -1) {
    system_ = BuildWithPaths(text);
    if (nRanks != -1) {
      system_->nRanks = nRanks;
      system_->inter = system_->nodes[GPU].count == nRanks ? 0 : 1;
      EXPECT_EQ(ncclSuccess, ncclTopoComputePaths(system_.get(), nullptr));
    }
    EXPECT_EQ(ncclSuccess, ncclTopoSearchInit(system_.get()));
    return system_.get();
  }

  GraphPtr Compute(int id, int pattern, int minChannels = 1, int maxChannels = MAXCHANNELS / 2) {
    GraphPtr graph = NewGraph(id, pattern, minChannels, maxChannels);
    EXPECT_EQ(ncclSuccess, ncclTopoCompute(system_.get(), graph.get()));
    return graph;
  }

  // Every channel lists each local rank once.
  void ExpectPermutations(const ncclTopoGraph* graph) {
    int ngpus = system_->nodes[GPU].count;
    for (int c = 0; c < graph->nChannels; c++) {
      std::set<int> ranks(graph->intra + c * ngpus, graph->intra + (c + 1) * ngpus);
      EXPECT_EQ((size_t)ngpus, ranks.size()) << "channel " << c;
    }
  }
};

// --- Search initialization

TEST_F(SearchMicrotest, SearchInit_SingleGpuUsesLocalBandwidth) {
  ncclTopoSystem* system = Use(XgmiSystem(1));

  EXPECT_FLOAT_EQ(LOC_BW, system->maxBw);
  EXPECT_FLOAT_EQ(LOC_BW, system->totalBw);
}

TEST_F(SearchMicrotest, SearchInit_IntraNodeTakesTheBestDevPathAndXgmiSum) {
  ncclTopoSystem* system = Use(XgmiSystem(4));

  EXPECT_FLOAT_EQ(GFX94X_XGMI_WIDTH, system->maxBw);
  EXPECT_FLOAT_EQ(3 * GFX94X_XGMI_WIDTH, system->totalBw);
}

TEST_F(SearchMicrotest, SearchInit_InterNodeTakesTheBestNicPath) {
  ncclTopoSystem* system = Use(XgmiSystem(2, 1, true), 4);

  EXPECT_FLOAT_EQ(48, system->maxBw);
  EXPECT_FLOAT_EQ(GFX94X_XGMI_WIDTH, system->totalBw);
}

TEST_F(SearchMicrotest, SearchInit_InterNodeMaxIsTheNicPathEvenWithFasterXgmi) {
  std::string body = GpuPci(GpuBus(0), 0, 0, Xgmi(GpuBus(1))) + GpuPci(GpuBus(1), 1, 1, Xgmi(GpuBus(0))) +
                     NicPci(NicBus(0), RailNet("net", 0, 0, 0, 100000));
  ncclTopoSystem* system = Use(System(Cpu(0, body)), 4);

  EXPECT_FLOAT_EQ(12.5, system->maxBw);
}

TEST_F(SearchMicrotest, SearchInit_PcieOnlyDevsTakeThePciLink) {
  ncclTopoSystem* system = Use(System(Cpu(0, GpuPci(GpuBus(0), 0, 0) + GpuPci(GpuBus(1), 1, 1))));

  EXPECT_FLOAT_EQ(48, system->totalBw);
}

TEST_F(SearchMicrotest, ComputeCommCPU_CopiesTheFirstCpu) {
  Use(System(Cpu(0, GpuPci(GpuBus(0), 0, 0), "", kIntelCpu)));
  auto comm = std::make_unique<ncclComm>();
  comm->topo = system_.get();

  ASSERT_EQ(ncclSuccess, ncclTopoComputeCommCPU(comm.get()));

  EXPECT_EQ(NCCL_TOPO_CPU_ARCH_X86, comm->cpuArch);
  EXPECT_EQ(NCCL_TOPO_CPU_VENDOR_INTEL, comm->cpuVendor);
}

// --- Intra-node search

TEST_F(SearchMicrotest, Compute_XgmiRingUsesEveryLinkAtFullWidth) {
  Use(XgmiSystem(4));

  GraphPtr ring = Compute(0, NCCL_TOPO_PATTERN_RING);

  EXPECT_EQ(PATH_NVL, ring->typeIntra);
  EXPECT_EQ(6, ring->nChannels);  // three edge-disjoint rings, doubled
  EXPECT_FLOAT_EQ(24, ring->bwIntra);
  EXPECT_EQ(0, ring->crossNic);
  ExpectPermutations(ring.get());
}

TEST_F(SearchMicrotest, Compute_TreeKeepsTheRingChannelCount) {
  Use(XgmiSystem(4));

  GraphPtr tree = Compute(1, NCCL_TOPO_PATTERN_BALANCED_TREE, 6, 6);

  EXPECT_EQ(6, tree->nChannels);
  EXPECT_EQ(PATH_NVL, tree->typeIntra);
  ExpectPermutations(tree.get());
}

TEST_F(SearchMicrotest, Compute_PcieRingFallsBackToThePciPathType) {
  Use(System(Cpu(0, Pci(SwitchBus(0), GpuPci(GpuBus(0), 0, 0) + GpuPci(GpuBus(1), 1, 1)))));

  GraphPtr ring = Compute(0, NCCL_TOPO_PATTERN_RING);

  EXPECT_EQ(PATH_PIX, ring->typeIntra);
  EXPECT_GE(ring->nChannels, 1);
  ExpectPermutations(ring.get());
}

TEST_F(SearchMicrotest, Compute_SingleGpuTreeBecomesAPlainTree) {
  Use(XgmiSystem(1));

  GraphPtr tree = Compute(1, NCCL_TOPO_PATTERN_BALANCED_TREE);

  EXPECT_EQ(NCCL_TOPO_PATTERN_TREE, tree->pattern);
  EXPECT_GE(tree->nChannels, 1);
  EXPECT_EQ(0, tree->intra[0]);
}

TEST_F(SearchMicrotest, Compute_NvlsNeedsAnNvSwitchAndHopper) {
  Use(XgmiSystem(2));

  GraphPtr nvls = Compute(3, NCCL_TOPO_PATTERN_NVLS);

  EXPECT_EQ(0, nvls->nChannels);
}

TEST_F(SearchMicrotest, Compute_UnreachableGpusFallBackToTheSimpleOrder) {
  Use(XgmiSystem(2, 1, true), 4);
  for (int g = 0; g < 2; g++) {
    for (int p = 0; p < 2; p++) {
      if (g != p) system_->nodes[GPU].nodes[g].paths[GPU][p].type = PATH_DIS;
    }
  }

  GraphPtr ring = Compute(0, NCCL_TOPO_PATTERN_RING);

  EXPECT_EQ(1, ring->nChannels);
  EXPECT_FLOAT_EQ(0.1, ring->bwIntra);
  EXPECT_EQ(PATH_SYS, ring->typeIntra);
  EXPECT_EQ(0, ring->intra[0]);
  EXPECT_EQ(1, ring->intra[1]);
  EXPECT_EQ(system_->nodes[NET].nodes[0].id, ring->inter[0]);
  EXPECT_EQ(system_->nodes[NET].nodes[1].id, ring->inter[1]);
  EXPECT_FLOAT_EQ(0.1, ring->bwInter);
  EXPECT_TRUE(Logged("Could not find a path for pattern 4, falling back to simple order"));
}

TEST_F(SearchMicrotest, Compute_FallbackWithoutNicsHasNoInterNode) {
  Use(XgmiSystem(2));
  for (int g = 0; g < 2; g++) system_->nodes[GPU].nodes[g].paths[GPU][1 - g].type = PATH_DIS;

  GraphPtr ring = Compute(0, NCCL_TOPO_PATTERN_RING);

  EXPECT_EQ(1, ring->nChannels);
  EXPECT_EQ(-1, ring->inter[0]);
  EXPECT_EQ(PATH_DIS, ring->typeInter);
  EXPECT_FLOAT_EQ(0, ring->bwInter);
}

// --- Inter-node search

TEST_F(SearchMicrotest, Compute_InterNodeRingStartsAndEndsOnLocalNics) {
  Use(XgmiSystem(2, 1, true), 4);

  GraphPtr ring = Compute(0, NCCL_TOPO_PATTERN_RING);

  ASSERT_GE(ring->nChannels, 1);
  ExpectPermutations(ring.get());
  for (int c = 0; c < ring->nChannels; c++) {
    int first = GpuIndexOfRank(system_.get(), ring->intra[2 * c]);
    int last = GpuIndexOfRank(system_.get(), ring->intra[2 * c + 1]);
    EXPECT_EQ(system_->nodes[NET].nodes[first].id, ring->inter[2 * c]) << c;
    EXPECT_EQ(system_->nodes[NET].nodes[last].id, ring->inter[2 * c + 1]) << c;
  }
  EXPECT_LE(ring->typeInter, PATH_PIX);
}

TEST_F(SearchMicrotest, Compute_InterNodeTreeStaysOnOneNicPerChannel) {
  Use(XgmiSystem(2, 1, true), 4);

  GraphPtr tree = Compute(1, NCCL_TOPO_PATTERN_BALANCED_TREE);

  ASSERT_GE(tree->nChannels, 1);
  ExpectPermutations(tree.get());
  for (int c = 0; c < tree->nChannels; c++) {
    EXPECT_EQ(system_->nodes[NET].nodes[0].id, tree->inter[2 * c]) << c;
    EXPECT_EQ(tree->inter[2 * c], tree->inter[2 * c + 1]) << c;
  }
}

TEST_F(SearchMicrotest, Compute_CollNetDirectUsesCollCapableNics) {
  std::string body;
  for (int g = 0; g < 2; g++) {
    body += Pci(SwitchBus(g), GpuPci(GpuBus(g), g, g, Xgmi(GpuBus(1 - g))) +
                                NicPci(NicBus(g), Net("n" + std::to_string(g), g, g == 0 ? "coll=\"1\"" : "")));
  }
  Use(System(Cpu(0, body)), 4);
  GraphPtr graph = NewGraph(4, NCCL_TOPO_PATTERN_COLLNET_DIRECT, 1, MAXCHANNELS);
  graph->collNet = 1;

  ASSERT_EQ(ncclSuccess, ncclTopoCompute(system_.get(), graph.get()));

  ASSERT_GE(graph->nChannels, 1);
  EXPECT_LE(graph->nChannels, 2);
  for (int c = 0; c < graph->nChannels; c++) {
    EXPECT_EQ(0, graph->inter[2 * c]) << c;  // n0, the only coll-capable NIC
    EXPECT_EQ(0, graph->inter[2 * c + 1]) << c;
  }
}

// --- Bandwidth accounting along a path

TEST_F(SearchMicrotest, FollowPath_IntelPciHopsCostTwentyPercentMore) {
  Use(System(Cpu(0, Pci(SwitchBus(0), GpuPci(GpuBus(0), 0, 0)) + Pci(SwitchBus(1), GpuPci(GpuBus(1), 1, 1)), "",
                 kIntelCpu)));
  GraphPtr graph = NewGraph(0, NCCL_TOPO_PATTERN_RING);
  graph->bwIntra = 10;
  graph->typeIntra = PATH_PHB;
  ncclTopoLink* pciLink = system_->nodes[GPU].nodes[0].paths[GPU][1].list[1];
  ncclTopoNode* node = nullptr;

  ASSERT_EQ(ncclSuccess, ncclTopoFollowPath(system_.get(), graph.get(), GPU, 0, GPU, 1, 1, &node));
  EXPECT_EQ(system_->nodes[GPU].nodes + 1, node);
  EXPECT_FLOAT_EQ(48 - 12, pciLink->bw);
  EXPECT_EQ(6, graph->nHops);
  ASSERT_EQ(ncclSuccess, ncclTopoFollowPath(system_.get(), graph.get(), GPU, 0, GPU, 1, -1, &node));
  EXPECT_FLOAT_EQ(48, pciLink->bw);
  EXPECT_EQ(0, graph->nHops);
}

TEST_F(SearchMicrotest, FollowPath_NotEnoughBandwidthRewinds) {
  Use(XgmiSystem(2));
  GraphPtr graph = NewGraph(0, NCCL_TOPO_PATTERN_RING);
  graph->bwIntra = 30;
  graph->typeIntra = PATH_NVL;
  ncclTopoNode* node = nullptr;

  ASSERT_EQ(ncclSuccess, ncclTopoFollowPath(system_.get(), graph.get(), GPU, 0, GPU, 1, 1, &node));
  ASSERT_NE(nullptr, node);
  ASSERT_EQ(ncclSuccess, ncclTopoFollowPath(system_.get(), graph.get(), GPU, 0, GPU, 1, 1, &node));

  EXPECT_EQ(nullptr, node);  // 18 GB/s left on the XGMI link
  ncclTopoLinkList& path = system_->nodes[GPU].nodes[0].paths[GPU][1];
  for (int i = 0; i < path.count; i++) {
    EXPECT_FLOAT_EQ(path.list[i]->type == LINK_NVL ? 18 : LOC_BW - 30, path.list[i]->bw);
  }
}

TEST_F(SearchMicrotest, FollowPath_AnyShortfallRewinds) {
  Use(XgmiSystem(2));
  GraphPtr graph = NewGraph(0, NCCL_TOPO_PATTERN_RING);
  graph->bwIntra = 24.5;
  graph->typeIntra = PATH_NVL;
  ncclTopoNode* node = nullptr;

  ASSERT_EQ(ncclSuccess, ncclTopoFollowPath(system_.get(), graph.get(), GPU, 0, GPU, 1, 1, &node));
  ASSERT_NE(nullptr, node);
  ASSERT_EQ(ncclSuccess, ncclTopoFollowPath(system_.get(), graph.get(), GPU, 0, GPU, 1, 1, &node));

  EXPECT_EQ(nullptr, node);  // 23.5 left
}

TEST_F(SearchMicrotest, FollowPath_PathTypeAndTreeReverseLimits) {
  Use(System(Cpu(0, GpuPci(GpuBus(0), 0, 0) + GpuPci(GpuBus(1), 1, 1))));
  GraphPtr graph = NewGraph(1, NCCL_TOPO_PATTERN_TREE);
  graph->bwIntra = 1;
  graph->typeIntra = PATH_PIX;
  ncclTopoNode* node = system_->nodes[CPU].nodes;

  ASSERT_EQ(ncclSuccess, ncclTopoFollowPath(system_.get(), graph.get(), GPU, 0, GPU, 1, 1, &node));
  EXPECT_EQ(nullptr, node);  // PHB is beyond PIX
  ASSERT_EQ(ncclSuccess, ncclTopoFollowPath(system_.get(), graph.get(), GPU, 0, CPU, 0, 1, &node));
  EXPECT_EQ(nullptr, node);  // the CPU path is fine, but a tree also needs the way back
  ASSERT_EQ(ncclSuccess, ncclTopoFollowPath(system_.get(), graph.get(), -1, 0, GPU, 1, 1, &node));
  EXPECT_EQ(system_->nodes[GPU].nodes + 1, node);
  GraphPtr ring = NewGraph(0, NCCL_TOPO_PATTERN_RING);
  ring->bwIntra = 1;
  ring->typeIntra = PATH_PHB - 1;
  ASSERT_EQ(ncclSuccess, ncclTopoFollowPath(system_.get(), ring.get(), GPU, 0, GPU, 1, 1, &node));
  EXPECT_EQ(nullptr, node);  // one level short is still too far
  system_->nodes[GPU].nodes[0].paths[GPU][1].type = PATH_DIS;
  graph->typeIntra = PATH_SYS;
  ASSERT_EQ(ncclSuccess, ncclTopoFollowPath(system_.get(), graph.get(), GPU, 0, GPU, 1, 1, &node));
  EXPECT_EQ(nullptr, node);
}

TEST_F(SearchMicrotest, FollowPath_MissingPathsAreAnError) {
  Use(XgmiSystem(1));
  GraphPtr graph = NewGraph(0, NCCL_TOPO_PATTERN_RING);
  ncclTopoNode* node = nullptr;

  EXPECT_EQ(ncclInternalError, ncclTopoFollowPath(system_.get(), graph.get(), GPU, 0, NVS, 0, 1, &node));

  EXPECT_TRUE(Logged("No path computed to go from GPU/0 to NVS/0"));
}

TEST_F(SearchMicrotest, FollowPath_PreAmpereDevsChargeTheReverseLinkFromANic) {
  Use(XgmiSystem(1, 1, true), 2);
  GraphPtr graph = NewGraph(0, NCCL_TOPO_PATTERN_RING);
  graph->bwInter = 8;
  graph->typeInter = PATH_PIX;
  ncclTopoNode* dev = system_->nodes[DEV].nodes;
  ncclTopoNode* node = nullptr;

  ASSERT_EQ(ncclSuccess, ncclTopoFollowPath(system_.get(), graph.get(), NET, 0, GPU, 0, 1, &node));

  ASSERT_NE(nullptr, node);
  ncclTopoLink* up = nullptr;
  for (int l = 0; l < dev->nlinks; l++) {
    if (dev->links[l].type == LINK_PCI) up = dev->links + l;
  }
  EXPECT_FLOAT_EQ(48 - 1, up->bw);  // an eighth of the forward bandwidth comes back as reads
}

TEST_F(SearchMicrotest, FollowPath_MissingReverseLinkIsAnError) {
  Use(XgmiSystem(1, 1, true), 2);
  GraphPtr graph = NewGraph(0, NCCL_TOPO_PATTERN_RING);
  graph->bwInter = 8;
  graph->typeInter = PATH_PIX;
  ncclTopoNode* dev = system_->nodes[DEV].nodes;
  for (int l = 0; l < dev->nlinks; l++) {
    if (dev->links[l].type == LINK_PCI) dev->links[l].type = LINK_SYS;
  }
  ncclTopoNode* node = nullptr;

  EXPECT_EQ(ncclInternalError, ncclTopoFollowPath(system_.get(), graph.get(), NET, 0, GPU, 0, 1, &node));

  EXPECT_TRUE(Logged("Could not find rev link for"));
}

TEST_F(SearchMicrotest, FollowPath_PowerNvlinkToTheCpuChargesBothWays) {
  std::string toCpu = Xgmi("0000:ff:00.0", 1, "0x068001");
  Use(System(Cpu(0, GpuPci(GpuBus(0), 0, 0, toCpu) + GpuPci(GpuBus(1), 1, 1, toCpu), "", "arch=\"ppc64\"")));
  GraphPtr graph = NewGraph(0, NCCL_TOPO_PATTERN_RING);
  graph->bwIntra = 10;
  graph->typeIntra = PATH_NVL;
  ncclTopoNode* cpu = system_->nodes[CPU].nodes;
  ncclTopoNode* node = nullptr;

  ASSERT_EQ(ncclSuccess, ncclTopoFollowPath(system_.get(), graph.get(), GPU, 0, GPU, 1, 1, &node));

  ASSERT_NE(nullptr, node);
  ncclTopoLink* toGpu0 = nullptr;
  for (int l = 0; l < cpu->nlinks; l++) {
    if (cpu->links[l].type == LINK_NVL && cpu->links[l].remNode == system_->nodes[DEV].nodes) toGpu0 = cpu->links + l;
  }
  ASSERT_NE(nullptr, toGpu0);
  EXPECT_FLOAT_EQ(GFX94X_XGMI_WIDTH - 10, toGpu0->bw);
}

// --- Search helpers

TEST_F(SearchMicrotest, GraphIndexes_UnknownRankOrNetFails) {
  Use(XgmiSystem(1));
  GraphPtr graph = NewGraph(0, NCCL_TOPO_PATTERN_RING);
  int index = -1;

  EXPECT_EQ(ncclInternalError, getGpuIndex(system_.get(), 3, &index));
  EXPECT_TRUE(Logged("Could not find gpu rank 3"));
  EXPECT_EQ(ncclInternalError, getNetIndex(system_.get(), 0x42, &index));
  EXPECT_TRUE(Logged("Could not find net id 42"));
  EXPECT_EQ(ncclInternalError, ncclTopoReplayGetGpu(system_.get(), graph.get(), 0, &index));
  graph->nChannels = 1;
  graph->intra[1] = 9;
  EXPECT_EQ(ncclInternalError, ncclTopoReplayGetGpu(system_.get(), graph.get(), 0, &index));
}

TEST_F(SearchMicrotest, NextGpuSort_SwitchPrefersTheRingNeighbors) {
  std::string toSwitch = Xgmi("fffffff:ffff:ff", 1, "0x068000");
  std::string body;
  for (int g = 0; g < 4; g++) body += GpuPci(GpuBus(g), g, g, toSwitch);
  Use(System(Cpu(0, body)));
  int next[NCCL_TOPO_MAX_NODES], count = 0;
  ncclTopoNode* gpu1 = system_->nodes[GPU].nodes + 1;

  for (int pattern : {NCCL_TOPO_PATTERN_RING, NCCL_TOPO_PATTERN_BALANCED_TREE, NCCL_TOPO_PATTERN_TREE}) {
    GraphPtr graph = NewGraph(0, pattern);
    ASSERT_EQ(ncclSuccess, ncclTopoSearchNextGpuSort(system_.get(), graph.get(), gpu1, next, &count, 0));
    if (pattern == NCCL_TOPO_PATTERN_RING) {
      ASSERT_EQ(2, count);
      EXPECT_EQ(0, next[0]);
      EXPECT_EQ(2, next[1]);
    } else if (pattern == NCCL_TOPO_PATTERN_TREE) {
      ASSERT_EQ(1, count);
      EXPECT_EQ(2, next[0]);
    } else {
      ASSERT_EQ(2, count);
      EXPECT_EQ(2, next[0]);
      EXPECT_EQ(0, next[1]);
    }
  }
}

TEST_F(SearchMicrotest, NextGpuSort_TwoGpusBehindASwitchHaveOneNeighbor) {
  std::string toSwitch = Xgmi("fffffff:ffff:ff", 1, "0x068000");
  Use(System(Cpu(0, GpuPci(GpuBus(0), 0, 0, toSwitch) + GpuPci(GpuBus(1), 1, 1, toSwitch))));
  GraphPtr graph = NewGraph(0, NCCL_TOPO_PATTERN_RING);
  int next[NCCL_TOPO_MAX_NODES], count = 0;

  ASSERT_EQ(ncclSuccess,
            ncclTopoSearchNextGpuSort(system_.get(), graph.get(), system_->nodes[GPU].nodes, next, &count, 0));

  EXPECT_EQ(1, count);
  EXPECT_EQ(1, next[0]);
}

TEST_F(SearchMicrotest, NextGpuSort_EqualIntraScoresGoInReverseWhenAwayFromTheNic) {
  Use(XgmiSystem(4, 1, true), 8);
  GraphPtr graph = NewGraph(0, NCCL_TOPO_PATTERN_RING);
  graph->inter[0] = system_->nodes[NET].nodes[0].id;
  int next[NCCL_TOPO_MAX_NODES], count = 0;

  ASSERT_EQ(ncclSuccess,
            ncclTopoSearchNextGpuSort(system_.get(), graph.get(), system_->nodes[GPU].nodes, next, &count, 0));
  EXPECT_EQ(std::vector<int>({1, 2, 3}), std::vector<int>(next, next + count));
  ASSERT_EQ(ncclSuccess,
            ncclTopoSearchNextGpuSort(system_.get(), graph.get(), system_->nodes[GPU].nodes, next, &count, -1));
  EXPECT_EQ(std::vector<int>({3, 2, 1}), std::vector<int>(next, next + count));
  ASSERT_EQ(ncclSuccess,
            ncclTopoSearchNextGpuSort(system_.get(), graph.get(), system_->nodes[GPU].nodes, next, &count, 1));
  EXPECT_EQ(3, count);
}

TEST_F(SearchMicrotest, NextGpuSort_NicBandwidthFirstThenFewerIntraHops) {
  Use(XgmiSystem(4, 1, true), 8);
  GraphPtr graph = NewGraph(0, NCCL_TOPO_PATTERN_RING);
  graph->inter[0] = system_->nodes[NET].nodes[0].id;
  int next[NCCL_TOPO_MAX_NODES], count = 0;
  ncclTopoNode* gpu0 = system_->nodes[GPU].nodes;

  system_->nodes[NET].nodes[0].paths[GPU][2].bw /= 2;
  ASSERT_EQ(ncclSuccess, ncclTopoSearchNextGpuSort(system_.get(), graph.get(), gpu0, next, &count, 1));
  EXPECT_EQ(std::vector<int>({1, 3, 2}), std::vector<int>(next, next + count));

  gpu0->paths[GPU][3].count = 1;
  ASSERT_EQ(ncclSuccess, ncclTopoSearchNextGpuSort(system_.get(), graph.get(), gpu0, next, &count, 0));
  EXPECT_EQ(std::vector<int>({3, 1, 2}), std::vector<int>(next, next + count));
}

TEST_F(SearchMicrotest, CompareGraphs_OddCountTiesGoToTheEvenReference) {
  Use(XgmiSystem(2, 1, true), 4);
  GraphPtr graph = NewGraph(0, NCCL_TOPO_PATTERN_RING);
  GraphPtr ref = NewGraph(0, NCCL_TOPO_PATTERN_RING);
  graph->nChannels = 1;
  graph->bwIntra = 40;
  graph->nHops = 1;
  ref->nChannels = 2;
  ref->bwIntra = 20;
  ref->nHops = 5;
  int copy = 0;

  ASSERT_EQ(ncclSuccess, ncclTopoCompareGraphs(system_.get(), graph.get(), ref.get(), &copy));

  EXPECT_EQ(0, copy);  // same aggregate bandwidth, so fewer hops do not matter
}

TEST_F(SearchMicrotest, CompareGraphs_ChannelsThenBandwidthThenHopsThenXgmi) {
  Use(XgmiSystem(2, 1, true), 4);
  GraphPtr graph = NewGraph(0, NCCL_TOPO_PATTERN_RING);
  GraphPtr ref = NewGraph(0, NCCL_TOPO_PATTERN_RING);
  auto compare = [&]() {
    int copy = 0;
    EXPECT_EQ(ncclSuccess, ncclTopoCompareGraphs(system_.get(), graph.get(), ref.get(), &copy));
    return copy;
  };
  graph->minChannels = 2;
  graph->nChannels = 1;
  EXPECT_EQ(0, compare());  // under the minimum

  graph->minChannels = 1;
  graph->nChannels = 1;
  graph->bwIntra = 30;
  ref->nChannels = 2;
  ref->bwIntra = 20;
  EXPECT_EQ(0, compare());  // an odd count loses to an even one with the bandwidth

  ref->bwIntra = 10;
  EXPECT_EQ(1, compare());  // 30 beats 2x10
  graph->nChannels = 2;
  ref->bwIntra = 30;
  graph->bwIntra = 20;
  EXPECT_EQ(0, compare());
  graph->bwIntra = 30;
  graph->nHops = 3;
  ref->nHops = 4;
  EXPECT_EQ(1, compare());  // fewer hops
  ref->crossNic = 1;
  EXPECT_EQ(0, compare());
  graph->intra[0] = ref->intra[0] = 0;
  graph->intra[1] = ref->intra[1] = 1;
  graph->intra[2] = 0;
  graph->intra[3] = 1;
  ref->intra[2] = ref->intra[3] = 0;
  EXPECT_EQ(1, compare());  // more XGMI hops in use
}

TEST_F(SearchMicrotest, CompareGraphs_NvlsWantsMoreHeads) {
  Use(XgmiSystem(4));
  GraphPtr graph = NewGraph(3, NCCL_TOPO_PATTERN_NVLS);
  GraphPtr ref = NewGraph(3, NCCL_TOPO_PATTERN_NVLS);
  int copy = 0;
  graph->nChannels = 3;
  ref->nChannels = 2;

  ASSERT_EQ(ncclSuccess, ncclTopoCompareGraphs(system_.get(), graph.get(), ref.get(), &copy));
  EXPECT_EQ(1, copy);

  copy = 0;
  graph->nChannels = 5;
  graph->bwInter = 1;
  ASSERT_EQ(ncclSuccess, ncclTopoCompareGraphs(system_.get(), graph.get(), ref.get(), &copy));
  EXPECT_EQ(1, copy);  // more heads than GPUs, but more aggregate bandwidth
  copy = 0;
  graph->bwInter = 0;
  ASSERT_EQ(ncclSuccess, ncclTopoCompareGraphs(system_.get(), graph.get(), ref.get(), &copy));
  EXPECT_EQ(0, copy);
}

class SearchNetsMicrotest : public SearchMicrotest {
 protected:
  // Two hosts: GPU0 and NIC0/NIC1 behind one switch here, GPU1 on the other host with no path to them.
  void SetUp() override {
    SearchMicrotest::SetUp();
    std::string here = Pci(SwitchBus(0), GpuPci(GpuBus(0), 0, 0) + NicPci(NicBus(0), RailNet("net", 0, 0)) +
                                         NicPci(NicBus(1), RailNet("net", 1, 1)));
    std::string far = Pci(SwitchBus(3), NicPci(NicBus(3), RailNet("net", 2, 2)));
    Use(System(Cpu(0, here + far) + Cpu(1, GpuPci(GpuBus(1), 1, 1), "host_hash=\"0x9\"")), 4);
  }

  std::vector<int> Select(int typeInter, int gpu) {
    int nets[NCCL_TOPO_MAX_NODES], count = 0;
    EXPECT_EQ(ncclSuccess, ncclTopoSelectNets(system_.get(), typeInter, gpu, nets, &count));
    return std::vector<int>(nets, nets + count);
  }
};

TEST_F(SearchNetsMicrotest, SelectNets_LocalNicsFirstThenOthersUpToTheType) {
  g_params["MNNVL_SCATTER_NETS_ENABLE"] = 0;

  EXPECT_EQ(std::vector<int>({0, 1}), Select(PATH_PIX, 0));
  EXPECT_EQ(std::vector<int>({0, 1, 2}), Select(PATH_PHB, 0));
  EXPECT_EQ(std::vector<int>({0, 1}), Select(PATH_PIX, -1));
}

TEST_F(SearchNetsMicrotest, SelectNets_MultiHostOrdersByGpuFirst) {
  ASSERT_EQ(2, system_->nHosts);

  EXPECT_EQ(std::vector<int>({0, 1}), Select(PATH_PIX, -1));
  EXPECT_EQ(std::vector<int>({0, 1, 2}), Select(PATH_SYS, 0));
}

TEST_F(SearchNetsMicrotest, SelectNets_MaxPolicyCapsTheList) {
  TopoSetNetDevsPolicy(NETDEVS_POLICY_MAX, 1);

  std::vector<int> one = Select(PATH_SYS, 0);
  std::vector<int> perGpu = Select(PATH_SYS, -1);

  TopoSetNetDevsPolicy(NETDEVS_POLICY_AUTO, -1);
  EXPECT_EQ(std::vector<int>({0}), one);
  EXPECT_EQ(std::vector<int>({0, 1}), perGpu);  // the cap is per GPU when selecting for all of them
}

TEST_F(SearchNetsMicrotest, CheckNet_PlaneRailAndPatternRules) {
  GraphPtr graph = NewGraph(0, NCCL_TOPO_PATTERN_RING);
  ncclTopoNode* start = system_->nodes[NET].nodes;
  graph->crossNic = 2;

  EXPECT_TRUE(ncclTopoSearchCheckNet(system_.get(), graph.get(), start, 1, 0));
  graph->nChannels = 1;
  graph->inter[0] = system_->nodes[NET].nodes[0].id;
  EXPECT_FALSE(ncclTopoSearchCheckNet(system_.get(), graph.get(), start, 1, 0));  // odd channel: same NIC back
  EXPECT_TRUE(ncclTopoSearchCheckNet(system_.get(), graph.get(), start, 0, 0));
  graph->crossNic = 0;
  EXPECT_FALSE(ncclTopoSearchCheckNet(system_.get(), graph.get(), start, 1, 0));  // other rail
  system_->nodes[NET].nodes[1].net.planeId = 3;
  graph->crossNic = 2;
  graph->nChannels = 0;
  EXPECT_FALSE(ncclTopoSearchCheckNet(system_.get(), graph.get(), start, 1, 0));  // other plane, even across rails
  graph->pattern = NCCL_TOPO_PATTERN_TREE;
  EXPECT_FALSE(ncclTopoSearchCheckNet(system_.get(), graph.get(), start, 2, 0));
  graph->pattern = NCCL_TOPO_PATTERN_BALANCED_TREE;
  graph->crossNic = 1;
  graph->nChannels = 0;
  graph->inter[1] = system_->nodes[NET].nodes[2].id;
  EXPECT_FALSE(ncclTopoSearchCheckNet(system_.get(), graph.get(), start, 0, 1));
  EXPECT_TRUE(ncclTopoSearchCheckNet(system_.get(), graph.get(), start, 2, 1));
}

TEST_F(SearchNetsMicrotest, CheckNet_UndefinedRailsMatchOnlyOnTheSameHostUnlessRailPerHost) {
  GraphPtr graph = NewGraph(0, NCCL_TOPO_PATTERN_RING);
  graph->crossNic = 0;
  ncclTopoNode start = system_->nodes[NET].nodes[0];
  start.net.railId = NCCL_TOPO_UNDEF_BIT | 1;
  start.id = NCCL_TOPO_ID(1, 0);
  system_->nodes[NET].nodes[1].net.railId = NCCL_TOPO_UNDEF_BIT | 1;

  EXPECT_FALSE(ncclTopoSearchCheckNet(system_.get(), graph.get(), &start, 1, 0));
  g_params["MNNVL_RAIL_PER_HOST"] = 1;
  EXPECT_TRUE(ncclTopoSearchCheckNet(system_.get(), graph.get(), &start, 1, 0));
  g_params["MNNVL_RAIL_PER_HOST"] = -1;
  for (int g = 0; g < system_->nodes[GPU].count; g++) system_->nodes[GPU].nodes[g].gpu.cudaCompCap = 107;
  EXPECT_TRUE(ncclTopoSearchCheckNet(system_.get(), graph.get(), &start, 1, 0));  // Rubin and later
  system_->nodes[GPU].nodes[1].gpu.cudaCompCap = 90;
  EXPECT_FALSE(ncclTopoSearchCheckNet(system_.get(), graph.get(), &start, 1, 0));
}

TEST_F(SearchMicrotest, PrefNets_SkipGpusWithoutNicPaths) {
  Use(System(Cpu(0, Pci(SwitchBus(0), GpuPci(GpuBus(0), 0, 0) + NicPci(NicBus(0), Net("a", 0)))) +
             Cpu(1, GpuPci(GpuBus(1), 1, 1), "host_hash=\"0x9\"")),
      4);
  int nets[NCCL_TOPO_MAX_NODES], count = 0;

  ASSERT_EQ(ncclSuccess, ncclTopoPrefNetsGpuFirst(system_.get(), -1, nets, &count));
  EXPECT_EQ(1, count);
  count = 0;
  ASSERT_EQ(ncclSuccess, ncclTopoPrefNetsChannelFirst(system_.get(), -1, nets, &count));
  EXPECT_EQ(1, count);
  count = 0;
  ASSERT_EQ(ncclSuccess, ncclTopoSelectNets(system_.get(), PATH_SYS, 1, nets, &count));
  EXPECT_EQ(0, count);
}

TEST_F(SearchMicrotest, PrefNets_ChannelFirstTakesEveryLocalNicOfTheGpu) {
  std::string nics;
  for (int n = 0; n < 3; n++) nics += NicPci(NicBus(n), RailNet("net", n, n));
  Use(System(Cpu(0, Pci(SwitchBus(0), GpuPci(GpuBus(0), 0, 0) + nics))), 4);
  int nets[NCCL_TOPO_MAX_NODES], count = 0;

  ASSERT_EQ(ncclSuccess, ncclTopoPrefNetsChannelFirst(system_.get(), 0, nets, &count));

  EXPECT_EQ(3, count);
}

// --- More of ncclTopoCompute

std::string XgmiRing4() {
  std::string body;
  for (int g = 0; g < 4; g++) body += GpuPci(GpuBus(g), g, g, Xgmi(GpuBus((g + 1) % 4)) + Xgmi(GpuBus((g + 3) % 4)));
  return System(Cpu(0, body));
}

TEST_F(SearchMicrotest, Compute_BalancedTreeSecondGpuAlsoReachesTheNic) {
  Use(System(Cpu(0, Pci(SwitchBus(0), GpuPci(GpuBus(0), 0, 0, Xgmi(GpuBus(1))) + NicPci(NicBus(0), Net("a", 0)))) +
             Cpu(1, GpuPci(GpuBus(1), 1, 1, Xgmi(GpuBus(0))))),
      4);

  GraphPtr tree = Compute(1, NCCL_TOPO_PATTERN_BALANCED_TREE);

  EXPECT_EQ(PATH_SYS, tree->typeInter);  // GPU 1 reaches the NIC across the sockets
  EXPECT_EQ(2, tree->nChannels);
  EXPECT_FLOAT_EQ(15, tree->bwInter);
}

TEST_F(SearchMicrotest, Compute_TreeStartsFromTheBandwidthItsReturnPathAdds) {
  Use(XgmiSystem(4));

  GraphPtr tree = Compute(1, NCCL_TOPO_PATTERN_BALANCED_TREE, 4, 4);

  EXPECT_EQ(4, tree->nChannels);  // four chains cover all twelve directed links
  EXPECT_FLOAT_EQ(48, tree->bwIntra);
  EXPECT_FLOAT_EQ(48, tree->bwInter);
}

TEST_F(SearchMicrotest, Compute_NicsOnOnePortShareItsBandwidth) {
  std::string body;
  for (int g = 0; g < 2; g++) {
    body += Pci(SwitchBus(g), GpuPci(GpuBus(g), g, g, Xgmi(GpuBus(1 - g))) +
                                  NicPci(NicBus(g), RailNet("net", g, g, 0, 200000)));
  }
  Use(System(Cpu(0, body)), 4);
  EXPECT_EQ(2, Compute(0, NCCL_TOPO_PATTERN_RING)->nChannels);

  Use(System(Cpu(0, body)), 4);
  system_->nodes[NET].nodes[1].net.asic = system_->nodes[NET].nodes[0].net.asic;
  system_->nodes[NET].nodes[1].net.port = system_->nodes[NET].nodes[0].net.port;
  EXPECT_EQ(1, Compute(0, NCCL_TOPO_PATTERN_RING)->nChannels);
}

TEST_F(SearchMicrotest, Compute_SplitXgmiDomainsNeedTwoRingChannels) {
  std::string body = GpuPci(GpuBus(0), 0, 0, Xgmi(GpuBus(1))) + GpuPci(GpuBus(1), 1, 1, Xgmi(GpuBus(0))) +
                     GpuPci(GpuBus(2), 2, 2, Xgmi(GpuBus(3))) + GpuPci(GpuBus(3), 3, 3, Xgmi(GpuBus(2)));
  Use(System(Cpu(0, body)));

  GraphPtr ring = Compute(0, NCCL_TOPO_PATTERN_RING);

  EXPECT_GE(ring->nChannels, 2);
  ExpectPermutations(ring.get());
}

TEST_F(SearchMicrotest, Compute_RomeSingleNodeRingIsCappedAtTwoChannels) {
  Use(XgmiSystem(4));
  system_->type |= RCCL_TOPO_4P2H_ROME;

  GraphPtr ring = Compute(0, NCCL_TOPO_PATTERN_RING);

  EXPECT_EQ(2, ring->maxChannels);
  EXPECT_LE(ring->nChannels, 2);
}

class SearchNvlsMicrotest : public SearchMicrotest {
 protected:
  void SetUp() override {
    SearchMicrotest::SetUp();
    std::string toSwitch = Xgmi("fffffff:ffff:ff", 4, "0x068000");
    std::string body;
    for (int g = 0; g < 4; g++) body += GpuPci(GpuBus(g), g, g, toSwitch);
    Use(System(Cpu(0, body)));
    for (int d = 0; d < system_->nodes[DEV].count; d++) system_->nodes[DEV].nodes[d].dev.cudaCompCap = 90;
  }
};

TEST_F(SearchNvlsMicrotest, Compute_NvlsHeadsEveryGpu) {
  GraphPtr nvls = Compute(3, NCCL_TOPO_PATTERN_NVLS, 1, MAXCHANNELS);

  EXPECT_EQ(4, nvls->maxChannels);
  EXPECT_EQ(4, nvls->nChannels);
  EXPECT_FLOAT_EQ(30, nvls->bwIntra);
  std::set<int> heads;
  for (int c = 0; c < nvls->nChannels; c++) heads.insert(nvls->intra[c * 4]);
  EXPECT_EQ(4u, heads.size());
}

TEST_F(SearchNvlsMicrotest, Compute_HopperBalancedTreeDuplicatesFourChannelsToEight) {
  GraphPtr tree = Compute(1, NCCL_TOPO_PATTERN_BALANCED_TREE);

  ASSERT_EQ(8, tree->nChannels);
  ExpectPermutations(tree.get());
  for (int i = 0; i < 4 * 4; i++) EXPECT_EQ(tree->intra[i], tree->intra[4 * 4 + i]) << i;
}

TEST_F(SearchMicrotest, Compute_InterNodeNvlsPicksOneNicPerHead) {
  std::string toSwitch = Xgmi("fffffff:ffff:ff", 4, "0x068000");
  std::string body;
  for (int g = 0; g < 2; g++) {
    body += Pci(SwitchBus(g), GpuPci(GpuBus(g), g, g, toSwitch) + NicPci(NicBus(g), Net("n" + std::to_string(g), g)));
  }
  body += Pci(SwitchBus(2), NicPci(NicBus(2), Net("n2", 2)));
  Use(System(Cpu(0, body)), 4);
  for (int d = 0; d < system_->nodes[DEV].count; d++) system_->nodes[DEV].nodes[d].dev.cudaCompCap = 90;

  GraphPtr nvls = Compute(3, NCCL_TOPO_PATTERN_NVLS, 1, MAXCHANNELS);

  ASSERT_GE(nvls->nChannels, 1);
  EXPECT_LE(nvls->nChannels, 2);
  if (nvls->nChannels == 2) EXPECT_NE(nvls->intra[0], nvls->intra[2]);
}

TEST_F(SearchMicrotest, Compute_GraphFileReplacesTheSearch) {
  Use(XgmiSystem(2));
  TempFile file("<graphs version=\"1\"><graph id=\"0\" pattern=\"4\" crossnic=\"0\" nchannels=\"2\" speedintra=\"20\" "
                "speedinter=\"20\" typeintra=\"NVL\" typeinter=\"PIX\" samechannels=\"1\">"
                "<channel><gpu dev=\"0\"/><gpu dev=\"1\"/></channel><channel><gpu dev=\"1\" rank=\"1\"/>"
                "<gpu dev=\"0\" rank=\"0\"/></channel></graph>"
                "<graph id=\"1\" pattern=\"1\" crossnic=\"0\" nchannels=\"1\" speedintra=\"1\" speedinter=\"1\" "
                "typeintra=\"SYS\" typeinter=\"SYS\" samechannels=\"0\"/></graphs>");
  RcclUnitTesting::ScopedEnv graphFile("NCCL_GRAPH_FILE", file.path());

  GraphPtr ring = Compute(0, NCCL_TOPO_PATTERN_RING);

  EXPECT_EQ(2, ring->nChannels);
  EXPECT_FLOAT_EQ(20, ring->bwIntra);
  EXPECT_EQ(PATH_NVL, ring->typeIntra);
  EXPECT_EQ(PATH_PIX, ring->typeInter);
  EXPECT_FLOAT_EQ(0, ring->latencyInter);
  EXPECT_EQ(std::vector<int>({0, 1, 1, 0}), std::vector<int>(ring->intra, ring->intra + 4));
  EXPECT_TRUE(Logged("Search 0 : 2 channels loaded from XML graph"));
  EXPECT_TRUE(Logged("latencyinter not found in graph, using 0.0"));
}

TEST_F(SearchMicrotest, Compute_EmptyGraphFileFallsThroughToTheSearch) {
  Use(XgmiSystem(2));
  TempFile file("<graphs version=\"1\"/>");
  RcclUnitTesting::ScopedEnv graphFile("NCCL_GRAPH_FILE", file.path());

  GraphPtr ring = Compute(0, NCCL_TOPO_PATTERN_RING);

  EXPECT_GE(ring->nChannels, 1);
}

TEST_F(SearchMicrotest, Compute_UserRingsAndTreesSkipTheSearch) {
  Use(
    System(Cpu(0, GpuPci(GpuBus(0), 0, 0, Xgmi(GpuBus(1))) + GpuPci(GpuBus(1), 1, 1, Xgmi(GpuBus(0))), "", kRomeCpu)));
  {
    RcclUnitTesting::ScopedEnv rings("NCCL_RINGS", "0 1|1 0");
    GraphPtr ring = Compute(0, NCCL_TOPO_PATTERN_RING);
    EXPECT_EQ(2, ring->nChannels);
    EXPECT_TRUE(system_->type & RCCL_TOPO_4P2H_ROME);
  }

  RcclUnitTesting::ScopedEnv trees("RCCL_TREES", "(0(1))");
  GraphPtr tree = Compute(1, NCCL_TOPO_PATTERN_BALANCED_TREE);
  EXPECT_TRUE(system_->treeDefined);
  EXPECT_STREQ("(0(1))", tree->treeBase[0]);
}

TEST_F(SearchMicrotest, Compute_UserRingsOnAZenHostDoNotMarkRome) {
  Use(XgmiSystem(2));
  RcclUnitTesting::ScopedEnv rings("NCCL_RINGS", "0 1");

  GraphPtr ring = Compute(0, NCCL_TOPO_PATTERN_RING);

  EXPECT_EQ(1, ring->nChannels);
  EXPECT_FALSE(system_->type & RCCL_TOPO_4P2H_ROME);
}

TEST_F(SearchMicrotest, Compute_ModelMatchingFallsThroughWhenNothingMatches) {
  g_params["RCCL_MODEL_MATCHING_DISABLE"] = 0;
  Use(XgmiSystem(2));

  GraphPtr ring = Compute(0, NCCL_TOPO_PATTERN_RING);

  EXPECT_GE(ring->nChannels, 1);
  EXPECT_EQ(RCCL_ROME_TOPO_PRESET_MODEL_IDX_NONE, system_->romeTopoModelIdx);
}

TEST_F(SearchMicrotest, GetSpeedArray_PicksTheTableForTheComputeCapability) {
  struct Case {
    int cc;
    int inter;
    int n;
    float first;
  };
  for (const Case& c : {Case{60, 0, (int)NSPEEDSINTRA, 48.0f}, Case{60, 1, (int)NSPEEDSINTER, 48.0f},
                        Case{90, 0, (int)NSPEEDSINTRA_SM90, 60.0f}, Case{90, 1, (int)NSPEEDSINTER_SM90, 48.0f},
                        Case{100, 0, (int)NSPEEDSINTRA_SM100, 90.0f}, Case{100, 1, (int)NSPEEDSINTER_SM100, 96.0f},
                        Case{107, 0, (int)NSPEEDSINTRA_RUBIN, 280.8f}, Case{107, 1, (int)NSPEEDSINTER_RUBIN, 280.8f}}) {
    int n = 0;
    float* speeds = nullptr;

    ncclTopoGetSpeedArray(c.inter, c.cc, &n, &speeds);

    EXPECT_EQ(c.n, n) << c.cc << "/" << c.inter;
    EXPECT_FLOAT_EQ(c.first, speeds[0]) << c.cc << "/" << c.inter;
  }
}

TEST_F(SearchMicrotest, DupChannels_DoublesWideGraphsOnly) {
  GraphPtr graph = NewGraph(0, NCCL_TOPO_PATTERN_RING, 1, 8);
  graph->nChannels = 3;
  graph->bwIntra = graph->bwInter = 40;
  graph->intra[0] = 7;
  graph->inter[1] = 9;

  ASSERT_EQ(ncclSuccess, ncclTopoDupChannels(graph.get(), 60, 1));
  EXPECT_EQ(6, graph->nChannels);
  EXPECT_FLOAT_EQ(20, graph->bwIntra);
  EXPECT_FLOAT_EQ(20, graph->bwInter);
  EXPECT_EQ(7, graph->intra[3]);
  EXPECT_EQ(9, graph->inter[7]);

  graph->bwIntra = graph->bwInter = 40;
  ASSERT_EQ(ncclSuccess, ncclTopoDupChannels(graph.get(), 60, 1));
  EXPECT_EQ(8, graph->nChannels);  // capped by maxChannels
  EXPECT_FLOAT_EQ(20, graph->bwIntra);

  graph->nChannels = 6;
  graph->bwIntra = 40;
  ASSERT_EQ(ncclSuccess, ncclTopoDupChannels(graph.get(), 90, 1));
  EXPECT_EQ(6, graph->nChannels);  // Hopper keeps many channels below 50 GB/s
  graph->bwIntra = 20;
  ASSERT_EQ(ncclSuccess, ncclTopoDupChannels(graph.get(), 60, 1));
  EXPECT_EQ(6, graph->nChannels);
  graph->pattern = NCCL_TOPO_PATTERN_NVLS;
  graph->bwIntra = 40;
  ASSERT_EQ(ncclSuccess, ncclTopoDupChannels(graph.get(), 60, 1));
  EXPECT_EQ(6, graph->nChannels);
  graph->nChannels = 0;
  ASSERT_EQ(ncclSuccess, ncclTopoDupChannels(graph.get(), 60, 1));
  EXPECT_EQ(0, graph->nChannels);
}

TEST_F(SearchMicrotest, CheckCrossNicSupport_FollowsTheParam) {
  bool supported = false;
  ASSERT_EQ(ncclSuccess, ncclTopoCheckCrossNicSupport(&supported));
  EXPECT_TRUE(supported);
  g_params["CROSS_NIC"] = 0;
  ASSERT_EQ(ncclSuccess, ncclTopoCheckCrossNicSupport(&supported));
  EXPECT_FALSE(supported);
}

int g_fusedDev = -1;
ncclResult_t FusedDevices(int* n) {
  *n = 3;
  return ncclSuccess;
}
ncclResult_t FusedProperties(int dev, ncclNetProperties_t* props) {
  *props = ncclNetProperties_t{};
  props->vProps.ndevs = dev == g_fusedDev ? 2 : 1;
  return ncclSuccess;
}

TEST_F(SearchMicrotest, CheckNicFused_AnyDeviceMadeOfSeveral) {
  auto comm = std::make_unique<ncclComm>();
  ncclNet_t net{};
  net.devices = FusedDevices;
  net.getProperties = FusedProperties;
  comm->ncclNet = &net;
  bool fused = true;

  g_fusedDev = -1;
  ASSERT_EQ(ncclSuccess, ncclTopoCheckNicFused(comm.get(), &fused));
  EXPECT_FALSE(fused);
  g_fusedDev = 2;
  ASSERT_EQ(ncclSuccess, ncclTopoCheckNicFused(comm.get(), &fused));
  EXPECT_TRUE(fused);
}

// --- Graph XML import and export

TEST_F(SearchMicrotest, XmlFromGraphs_ExportsChannelsAndReimportsThem) {
  Use(XgmiSystem(2, 1, true), 4);
  GraphPtr ring = Compute(0, NCCL_TOPO_PATTERN_RING);
  GraphPtr nvls = NewGraph(3, NCCL_TOPO_PATTERN_NVLS);
  nvls->nChannels = 1;
  nvls->intra[0] = 1;
  nvls->intra[1] = 0;
  ncclTopoGraph* graphs[] = {ring.get(), nvls.get()};
  XmlPtr xml = NewXml(1024);

  ASSERT_EQ(ncclSuccess, ncclTopoGetXmlFromGraphs(2, graphs, system_.get(), xml.get()));

  ncclXmlNode* nvlsGraph = xml->nodes[0].subs[1];
  ASSERT_EQ(1, nvlsGraph->nSubs);
  EXPECT_EQ(3, nvlsGraph->subs[0]->nSubs);  // net, the head GPU only, net
  GraphPtr back = NewGraph(0, NCCL_TOPO_PATTERN_RING);
  int nChannels = 0;
  ASSERT_EQ(ncclSuccess, ncclTopoGetGraphFromXml(xml->nodes, system_.get(), back.get(), &nChannels));
  EXPECT_EQ(ring->nChannels, nChannels);
  EXPECT_EQ(ring->typeIntra, back->typeIntra);
  EXPECT_EQ(ring->intra[1], back->intra[1]);
  EXPECT_EQ(ring->inter[1], back->inter[1]);
  ncclXmlNode* channel = xml->nodes[0].subs[0]->subs[0];
  const char *dev1 = nullptr, *dev2 = nullptr;
  ASSERT_EQ(ncclSuccess, xmlGetAttrStr(channel->subs[1], "dev", &dev1));
  ASSERT_EQ(ncclSuccess, xmlGetAttrStr(channel->subs[2], "dev", &dev2));
  EXPECT_STRNE(dev1, dev2);  // each GPU exports its own dev
}

TEST_F(SearchMicrotest, XmlFromGraphs_UnknownRankOrTypeFails) {
  Use(XgmiSystem(2));
  GraphPtr ring = NewGraph(0, NCCL_TOPO_PATTERN_RING);
  ring->nChannels = 1;
  ring->intra[0] = 5;
  ncclTopoGraph* graphs[] = {ring.get()};
  XmlPtr xml = NewXml();

  EXPECT_EQ(ncclInternalError, ncclTopoGetXmlFromGraphs(1, graphs, system_.get(), xml.get()));
  EXPECT_TRUE(Logged("XML Export Channel : rank 5 not found"));
  ring->typeIntra = 99;
  EXPECT_EQ(ncclInternalError, ncclTopoGetXmlFromGraphs(1, graphs, system_.get(), xml.get()));
}

TEST_F(SearchMicrotest, GraphFromXml_SkipsOtherIdsAndCrossNicWhenDisabled) {
  Use(XgmiSystem(2));
  TempFile file("<graphs version=\"1\"><graph id=\"0\" pattern=\"4\" crossnic=\"1\" nchannels=\"1\" speedintra=\"2\" "
                "speedinter=\"2\" latencyinter=\"1.5\" typeintra=\"PIX\" typeinter=\"PIX\" samechannels=\"0\">"
                "<channel><gpu dev=\"0\"/><gpu dev=\"9\"/></channel></graph></graphs>");
  XmlPtr xml = NewXml();
  ASSERT_EQ(ncclSuccess, ncclTopoGetXmlGraphFromFile(file.path(), xml.get()));
  GraphPtr graph = NewGraph(2, NCCL_TOPO_PATTERN_RING);
  int nChannels = 0;

  ASSERT_EQ(ncclSuccess, ncclTopoGetGraphFromXml(xml->nodes, system_.get(), graph.get(), &nChannels));
  EXPECT_EQ(0, nChannels);

  graph->id = 0;
  g_params["CROSS_NIC"] = 0;
  ASSERT_EQ(ncclSuccess, ncclTopoGetGraphFromXml(xml->nodes, system_.get(), graph.get(), &nChannels));
  EXPECT_EQ(0, nChannels);

  g_params.clear();
  EXPECT_EQ(ncclSystemError, ncclTopoGetGraphFromXml(xml->nodes, system_.get(), graph.get(), &nChannels));
  EXPECT_TRUE(Logged("XML Import Channel : dev 9 not found"));
  EXPECT_FLOAT_EQ(1.5, graph->latencyInter);
  EXPECT_EQ(1, graph->crossNic);
}

TEST_F(SearchMicrotest, DumpGraphs_WritesTheGraphFileWhenAsked) {
  Use(XgmiSystem(2));
  GraphPtr ring = Compute(0, NCCL_TOPO_PATTERN_RING);
  ncclTopoGraph* graphs[] = {ring.get()};
  ASSERT_EQ(ncclSuccess, ncclTopoDumpGraphs(system_.get(), 1, graphs));

  TempFile file;
  RcclUnitTesting::ScopedEnv dumpFile("NCCL_GRAPH_DUMP_FILE", file.path());
  ASSERT_EQ(ncclSuccess, ncclTopoDumpGraphs(system_.get(), 1, graphs));

  std::string text = file.Read();
  EXPECT_NE(std::string::npos, text.find("<graphs version=\"1\">"));
  EXPECT_NE(std::string::npos, text.find("typeintra=\"NVL\""));
  ring->intra[0] = 7;
  EXPECT_EQ(ncclInternalError, ncclTopoDumpGraphs(system_.get(), 1, graphs));
}

TEST_F(SearchMicrotest, PrintGraph_ShowsNicsAndGpus) {
  Use(XgmiSystem(2, 1, true), 4);
  GraphPtr ring = Compute(0, NCCL_TOPO_PATTERN_RING);

  ASSERT_EQ(ncclSuccess, ncclTopoPrintGraph(system_.get(), ring.get()));
  EXPECT_TRUE(Logged(" 0 : NET/0-0 GPU/0-"));

  GraphPtr intra = NewGraph(0, NCCL_TOPO_PATTERN_RING);
  intra->nChannels = 1;
  intra->nIntraChannels = 1;
  intra->intra[0] = 0;
  intra->intra[1] = 1;
  intra->intraNets[0] = 'N' + 1;
  intra->intraNets[1] = -1;
  intra->intraNets[3] = 'N' + 0;
  ASSERT_EQ(ncclSuccess, ncclTopoPrintGraph(system_.get(), intra.get()));
  EXPECT_TRUE(Logged(" 0 : NET/1 GPU/0-10000 GPU/0-20000 NET/0"));

  GraphPtr nvls = NewGraph(3, NCCL_TOPO_PATTERN_NVLS);
  nvls->nChannels = 1;
  nvls->intra[0] = 1;
  nvls->nIntraChannels = 1;
  ASSERT_EQ(ncclSuccess, ncclTopoPrintGraph(system_.get(), nvls.get()));
  EXPECT_TRUE(Logged(" 0 : GPU/0-20000"));
}

// --- NIC selection for transports

class SearchNetDevMicrotest : public SearchMicrotest {
 protected:
  std::unique_ptr<ncclComm> comm_ = std::make_unique<ncclComm>();
  std::vector<ncclPeerInfo> peers_;

  void Setup(const std::string& text, int nRanks) {
    AllowP2pBetweenEveryGpu();
    Use(text, nRanks);
    comm_->topo = system_.get();
    comm_->rank = 0;
    comm_->nRanks = nRanks;
    comm_->pxnDisable = RCCL_VALUE_UNSET;
    peers_.assign(nRanks, ncclPeerInfo{});
    for (int r = 0; r < nRanks; r++) peers_[r].nvmlDev = r % system_->nodes[GPU].count;
    comm_->peerInfo = peers_.data();
  }

  struct NetDev {
    int64_t id = -1;
    int dev = -1;
    int proxy = -1;
  };
  NetDev Get(ncclTopoGraph* graph, int channel, int peer, ncclResult_t expect = ncclSuccess) {
    NetDev d;
    EXPECT_EQ(expect, ncclTopoGetNetDev(comm_.get(), 0, graph, channel, peer, &d.id, &d.dev, &d.proxy));
    return d;
  }
};

TEST_F(SearchNetDevMicrotest, GetNetDev_HonorsTheGraphNics) {
  Setup(XgmiSystem(2, 1, true), 4);
  GraphPtr ring = NewGraph(0, NCCL_TOPO_PATTERN_RING);
  ring->nChannels = 2;
  ring->intra[0] = 0;
  ring->intra[1] = 1;
  ring->intra[2] = 1;
  ring->intra[3] = 0;
  for (int i = 0; i < 4; i++) ring->inter[i] = system_->nodes[NET].nodes[i % 2].id;

  EXPECT_EQ(0, Get(ring.get(), 0, -1).dev);  // rank 0 starts channel 0: its NIC in
  EXPECT_EQ(1, Get(ring.get(), 1, -1).dev);  // and ends channel 1: its NIC out
  EXPECT_EQ(0, Get(ring.get(), 2, -1).proxy);
}

TEST_F(SearchNetDevMicrotest, GetNetDev_NvlsUsesTheNicsOfTheRanksOwnHeads) {
  Setup(XgmiSystem(2, 1, true), 4);
  GraphPtr nvls = NewGraph(3, NCCL_TOPO_PATTERN_NVLS);
  nvls->nChannels = 3;
  nvls->intra[0] = 1;
  nvls->intra[2] = 0;
  nvls->intra[4] = 0;
  nvls->inter[2] = system_->nodes[NET].nodes[1].id;
  nvls->inter[4] = system_->nodes[NET].nodes[0].id;

  EXPECT_EQ(1, Get(nvls.get(), 0, -1).dev);
  EXPECT_EQ(0, Get(nvls.get(), 1, -1).dev);
  nvls->intra[2] = nvls->intra[4] = 1;
  Get(nvls.get(), 0, -1, ncclInternalError);
  EXPECT_TRUE(Logged("Could not find NIC for rank 0 in NVLS graph"));
}

TEST_F(SearchNetDevMicrotest, GetNetDev_WithoutGraphNeedsAPeer) {
  Setup(XgmiSystem(2, 1, true), 4);

  Get(nullptr, 0, -1, ncclInternalError);
  NetDev d = Get(nullptr, 0, 3);
  EXPECT_EQ(0, d.dev);  // PXN is off by default: our own NIC
  EXPECT_EQ(0, d.proxy);
}

TEST_F(SearchNetDevMicrotest, GetNetDev_NoCrossNicTakesThePeersRail) {
  Setup(XgmiSystem(2, 1, true), 4);
  g_params["CROSS_NIC"] = 0;

  NetDev d = Get(nullptr, 0, 3);

  EXPECT_EQ(1, d.dev);  // rank 3 sits on local dev 1, whose NIC is NIC 1
  EXPECT_EQ(0, d.proxy);
  peers_[3].nvmlDev = 7;
  EXPECT_EQ(0, Get(nullptr, 0, 3).dev);  // no local GPU with that dev
}

TEST_F(SearchNetDevMicrotest, GetNetDev_PxnLevelOneProxiesThroughTheNicsGpu) {
  Setup(XgmiSystem(2, 2, true), 4);
  comm_->pxnDisable = 0;
  ASSERT_EQ(ncclSuccess, ncclTopoComputePaths(system_.get(), comm_.get()));
  g_params["P2P_PXN_LEVEL"] = 1;

  NetDev d = Get(nullptr, 0, 3);

  EXPECT_EQ(1, d.dev);
  EXPECT_EQ(1, d.proxy);
}

TEST_F(SearchNetDevMicrotest, GetNetDev_PxnLevelTwoProxiesWheneverTheNicGpuHasGdr) {
  Setup(XgmiSystem(2, 2, true), 4);
  comm_->pxnDisable = 0;
  ASSERT_EQ(ncclSuccess, ncclTopoComputePaths(system_.get(), comm_.get()));

  NetDev d = Get(nullptr, 0, 3);
  EXPECT_EQ(1, d.dev);
  EXPECT_EQ(1, d.proxy);

  system_->nodes[NET].nodes[1].net.gdrSupport = 0;
  d = Get(nullptr, 0, 3);
  EXPECT_EQ(0, d.dev);
  EXPECT_EQ(0, d.proxy);
}

TEST_F(SearchNetDevMicrotest, GetNetDev_PxnLevelTwoNeedsXgmiToTheNicGpu) {
  std::string body;
  for (int g = 0; g < 2; g++) body += Pci(SwitchBus(g), GpuPci(GpuBus(g), g, g) + NicPci(NicBus(g), Net("a", g)));
  Setup(System(Cpu(0, body)), 4);
  comm_->pxnDisable = 0;
  ASSERT_EQ(ncclSuccess, ncclTopoComputePaths(system_.get(), comm_.get()));

  NetDev d = Get(nullptr, 0, 3);

  EXPECT_EQ(0, d.dev);
  EXPECT_EQ(0, d.proxy);
}

TEST_F(SearchMicrotest, GetIntraNetDev_ReadsTheIntraNetsOfTheRanksSlot) {
  Use(XgmiSystem(2, 1, true), 2);
  GraphPtr graph = NewGraph(0, NCCL_TOPO_PATTERN_RING);
  int64_t id = 0;
  int dev = 0;

  ASSERT_EQ(ncclSuccess, ncclTopoGetIntraNetDev(system_.get(), 0, nullptr, 0, 0, &id, &dev));
  EXPECT_EQ(-1, dev);
  EXPECT_EQ(-1, id);
  graph->nIntraChannels = 1;
  graph->intra[0] = 1;
  graph->intra[1] = 0;
  graph->intraNets[2] = 'N' + 1;
  graph->intraNets[3] = 'N' + 5;
  ASSERT_EQ(ncclSuccess, ncclTopoGetIntraNetDev(system_.get(), 0, graph.get(), 3, 0, &id, &dev));
  EXPECT_EQ(1, dev);
  EXPECT_EQ(NCCL_TOPO_ID(0, 1), id);
  ASSERT_EQ(ncclSuccess, ncclTopoGetIntraNetDev(system_.get(), 0, graph.get(), 3, 1, nullptr, &dev));
  EXPECT_EQ(-1, dev);  // NIC 5 does not exist
}

TEST_F(SearchMicrotest, GetLinkType_DirectOrThroughIntermediateXgmiGpus) {
  Use(XgmiRing4());
  bool xgmi = false;

  ASSERT_EQ(ncclSuccess, ncclTopoGetLinkType(system_.get(), 0, 1, &xgmi));
  EXPECT_TRUE(xgmi);
  ASSERT_EQ(ncclSuccess, ncclTopoGetLinkType(system_.get(), 0, 2, &xgmi, 0));
  EXPECT_FALSE(xgmi);
  ASSERT_EQ(ncclSuccess, ncclTopoGetLinkType(system_.get(), 0, 2, &xgmi));
  EXPECT_TRUE(xgmi);
  ASSERT_EQ(ncclSuccess, ncclTopoGetLinkType(system_.get(), 0, 9, &xgmi));
  EXPECT_FALSE(xgmi);

  Use(System(Cpu(0, GpuPci(GpuBus(0), 0, 0) + GpuPci(GpuBus(1), 1, 1) + GpuPci(GpuBus(2), 2, 2))));
  ASSERT_EQ(ncclSuccess, ncclTopoGetLinkType(system_.get(), 0, 1, &xgmi));
  EXPECT_FALSE(xgmi);
}

TEST_F(SearchMicrotest, FollowPath_TreeNeedsTheReversePathWithinTheTypeToo) {
  Use(XgmiSystem(2, 2, true), 4);
  auto comm = std::make_unique<ncclComm>();
  std::vector<ncclPeerInfo> peers(4);
  comm->peerInfo = peers.data();
  comm->pxnDisable = 0;
  AllowP2pBetweenEveryGpu();
  ASSERT_EQ(ncclSuccess, ncclTopoComputePaths(system_.get(), comm.get()));
  ASSERT_EQ(PATH_PXN, system_->nodes[GPU].nodes[1].paths[NET][0].type);
  GraphPtr graph = NewGraph(1, NCCL_TOPO_PATTERN_TREE);
  graph->bwInter = 1;
  graph->typeInter = PATH_PXN;
  ncclTopoNode* node = system_->nodes[CPU].nodes;

  ASSERT_EQ(ncclSuccess, ncclTopoFollowPath(system_.get(), graph.get(), GPU, 1, NET, 0, 1, &node));
  EXPECT_EQ(nullptr, node);
  graph->pattern = NCCL_TOPO_PATTERN_RING;
  ASSERT_EQ(ncclSuccess, ncclTopoFollowPath(system_.get(), graph.get(), GPU, 1, NET, 0, 1, &node));
  EXPECT_EQ(system_->nodes[NET].nodes, node);
}

TEST_F(SearchMicrotest, Compute_InterNodePlainTreeSearches) {
  Use(XgmiSystem(4, 1, true), 8);

  GraphPtr tree = Compute(1, NCCL_TOPO_PATTERN_TREE);

  ASSERT_GE(tree->nChannels, 1);
  ExpectPermutations(tree.get());
}

TEST_F(SearchNvlsMicrotest, Compute_NvlsBacksOffWhenTheUplinkIsShort) {
  for (int d = 0; d < system_->nodes[DEV].count; d++) {
    ncclTopoNode* dev = system_->nodes[DEV].nodes + d;
    for (int l = 0; l < dev->nlinks; l++) {
      if (dev->links[l].remNode->type == NVS) dev->links[l].bw = 60;  // from 192
    }
  }

  GraphPtr nvls = Compute(3, NCCL_TOPO_PATTERN_NVLS, 1, MAXCHANNELS);

  EXPECT_EQ(4, nvls->nChannels);
  EXPECT_FLOAT_EQ(12, nvls->bwIntra);
  EXPECT_FLOAT_EQ(24, nvls->bwInter);
}

TEST_F(SearchMicrotest, Compute_CollNetHeadsAreNotDuplicated) {
  std::string body =
    Pci(SwitchBus(0), GpuPci(GpuBus(0), 0, 0, Xgmi(GpuBus(1))) + NicPci(NicBus(0), Net("c0", 0, "coll=\"1\"")) +
                        NicPci(NicBus(2), Net("c2", 2, "coll=\"1\""))) +
    Pci(SwitchBus(1), GpuPci(GpuBus(1), 1, 1, Xgmi(GpuBus(0))));
  Use(System(Cpu(0, body)), 4);
  GraphPtr graph = NewGraph(4, NCCL_TOPO_PATTERN_COLLNET_DIRECT, 1, MAXCHANNELS);
  graph->collNet = 1;

  ASSERT_EQ(ncclSuccess, ncclTopoCompute(system_.get(), graph.get()));

  ASSERT_GE(graph->nChannels, 1);
  std::set<int> heads;
  for (int c = 0; c < graph->nChannels; c++) heads.insert(graph->intra[c * 2]);
  EXPECT_EQ((size_t)graph->nChannels, heads.size());
}

TEST_F(SearchMicrotest, Compute_SharedNicCarriesEveryRingChannel) {
  std::string body = Pci(SwitchBus(0), GpuPci(GpuBus(0), 0, 0, Xgmi(GpuBus(1))) +
                                         GpuPci(GpuBus(1), 1, 1, Xgmi(GpuBus(0))) + NicPci(NicBus(0), Net("n0", 0)));
  Use(System(Cpu(0, body)), 4);

  GraphPtr ring = Compute(0, NCCL_TOPO_PATTERN_RING);

  ASSERT_GE(ring->nChannels, 2);
  ExpectPermutations(ring.get());
  for (int c = 0; c < ring->nChannels; c++) {
    EXPECT_EQ(0, ring->inter[2 * c]) << c;
    EXPECT_EQ(0, ring->inter[2 * c + 1]) << c;
  }
}

TEST_F(SearchMicrotest, Compute_HopperInterNodeBalancedTree) {
  Use(XgmiSystem(2, 1, true), 4);
  for (int d = 0; d < system_->nodes[DEV].count; d++) system_->nodes[DEV].nodes[d].dev.cudaCompCap = 90;

  GraphPtr tree = Compute(1, NCCL_TOPO_PATTERN_BALANCED_TREE);

  ASSERT_GE(tree->nChannels, 1);
  ExpectPermutations(tree.get());
}

// Different rails force the crossNic retry; fast NICs leave room for the odd channels to look for their entry NIC.
TEST_F(SearchMicrotest, Compute_CrossNicRingEndsOnTheOtherRail) {
  std::string body;
  for (int g = 0; g < 2; g++) {
    body += Pci(SwitchBus(g), GpuPci(GpuBus(g), g, g, Xgmi(GpuBus(1 - g), 4)) +
                                  NicPci(NicBus(g), RailNet("net", g, g, 0, 1600000)));
  }
  Use(System(Cpu(0, body)), 4);

  GraphPtr ring = Compute(0, NCCL_TOPO_PATTERN_RING);

  ASSERT_GE(ring->nChannels, 1);
  EXPECT_EQ(2, ring->crossNic);
  EXPECT_NE(ring->inter[0], ring->inter[1]);
}



// ---------------------------------------------------------------------------
// ncclTopoGetChannelFromXml and its caller ncclTopoGetGraphFromXmlSub: the
// NCCL_GRAPH_FILE XML-graph import path.
//
// The input is attacker-controllable: NCCL_GRAPH_FILE names an XML file that
// ncclTopoGetXmlGraphFromFile parses into an ncclXml before these functions
// transcribe it into an ncclTopoGraph. These tests hand that ncclXml over
// directly rather than going through TopoScene, because what is under test is
// the transcription of arbitrary XML, not a topology RCCL itself discovered.
// ---------------------------------------------------------------------------

// A topology system with `nGpus` GPUs, each parented to a CPU node so that
// NCCL_TOPO_ID_SYSTEM_ID(gpu.parent->id) is well defined. Ranks are
// deliberately not equal to the dev numbers so a test reading graph->intra
// cannot pass by accident if the dev->rank translation is skipped.
class SearchXmlScene {
 public:
  explicit SearchXmlScene(int nGpus) : system_(new ncclTopoSystem{}) {
    system_->nodes[CPU].count = 1;
    system_->nodes[CPU].nodes[0].id = 0;
    system_->nodes[GPU].count = nGpus;
    for (int i = 0; i < nGpus; i++) {
      system_->nodes[GPU].nodes[i].gpu.dev = i;
      system_->nodes[GPU].nodes[i].gpu.rank = kRankBase + i;
      system_->nodes[GPU].nodes[i].gpu.parent = &system_->nodes[CPU].nodes[0];
    }
  }

  static constexpr int kRankBase = 100;

  ncclTopoSystem* system() { return system_.get(); }

 private:
  std::unique_ptr<ncclTopoSystem> system_;
};

constexpr int kGpus = 8;

class SearchXmlChannelMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    scene_ = std::make_unique<SearchXmlScene>(kGpus);
    graph_ = std::make_unique<ncclTopoGraph>();
    std::memset(graph_.get(), 0, sizeof(*graph_));
    // ncclXml's `nodes[1]` is a flexible array sized by xmlAlloc, so the nodes
    // are held in a deque instead: it never reallocates, keeping every
    // previously handed-out ncclXmlNode* stable as more nodes are added.
    nodes_.clear();
  }

  // Allocate a zeroed node from the fixture's arena.
  ncclXmlNode* NewNode(const char* name) {
    nodes_.emplace_back();
    ncclXmlNode* node = &nodes_.back();
    std::memset(node, 0, sizeof(*node));
    std::snprintf(node->name, sizeof(node->name), "%s", name);
    return node;
  }

  // Append a <name dev="..."> child to `parent`.
  ncclXmlNode* AddChild(ncclXmlNode* parent, const char* name, const char* dev) {
    ncclXmlNode* node = NewNode(name);
    node->parent = parent;
    parent->subs[parent->nSubs++] = node;
    SetAttr(node, "dev", dev);
    return node;
  }

  static void SetAttr(ncclXmlNode* node, const char* key, const char* value) {
    int index = node->nAttrs++;
    std::snprintf(node->attrs[index].key, sizeof(node->attrs[index].key), "%s", key);
    std::snprintf(node->attrs[index].value, sizeof(node->attrs[index].value), "%s", value);
  }

  ncclTopoSystem* system() { return scene_->system(); }
  ncclTopoGraph* graph() { return graph_.get(); }

 private:
  std::unique_ptr<SearchXmlScene> scene_;
  std::unique_ptr<ncclTopoGraph> graph_;
  std::deque<ncclXmlNode> nodes_;
};

// ---------------------------------------------------------------------------
// Test list (the agreed plan -- ROCM-31973).
//
// Accepted-input contracts (landed first; they are what stops a later guard
// being tightened into rejecting valid topology files):
//
//   [C1] WellFormedChannel_TranscribesGpuDevsToRanksInOrder
//   [C2] WellFormedChannel_TranscribesNetDevsToInter
//   [C3] ExplicitRankAttribute_IsUsedInsteadOfDevLookup
//   [C4] ChannelIndex_SelectsItsOwnSliceOfIntraAndInter
//   [C5] UnknownGpuDev_IsRejected
//   [C6] GraphGeneratedByNcclTopoGetXmlFromGraph_RoundTripsBackToTheSameGraph
//        (RCCL ships no graph XML of its own, so the format its own writer
//         emits is the "our own input" contract that licenses a reject fix)
//
// Rejection tests, one per row of the Phase 1 table:
//
//   [A] ChannelIndexBeyondMaxChannels_IsRejected
//       graph->inter + 2*c and graph->intra + ngpus*c both run past the end of
//       their arrays once c >= MAXCHANNELS; MAX_SUBS (512) admits c up to 511
//       against MAXCHANNELS 256.
//   [B] MoreNetSubsThanTheChannelHolds_IsRejected
//       inter[n++] has only 2 slots per channel; nothing bounds n.
//   [C] MoreGpuSubsThanGpus_IsRejected
//       intra[g++] has only ngpus slots per channel; nothing bounds g. The
//       overrun lands inside the same ncclTopoGraph, so it is invisible to
//       ASan -- the assertion has to be on the clobbered next channel.
//   [D] RankAttributeOutsideTheRankRange_IsRejected
//       strtol into an int: "4294967296" truncates to 0 silently.
//   [E] DevAttributeTooWideForTheIdSpace_IsRejected  -- expected to be
//       reclassified (probe first); likely subsumed by the dev-not-found arm.
//
// Caller (ncclTopoGetGraphFromXmlSub), its own commit:
//   [F] MoreChannelSubsThanMaxChannels_IsRejectedByTheCaller
//       the loop over xmlGraph->nSubs and *nChannels = xmlGraph->nSubs are
//       both unclamped, so the caller admits more than the buffers hold.
//
// TO BE WRITTEN -- this block is replaced at the end of the pass with what the
// pass actually found, including reclassified rows.
// ---------------------------------------------------------------------------

// --- Accepted-input contracts ---------------------------------------------
//
// A valid channel is still accepted and still produces the right graph. These
// are what stop any of the bounds below being tightened into rejecting a
// topology file RCCL itself would write.

TEST_F(SearchXmlChannelMicrotest, WellFormedChannel_TranscribesGpuDevsToRanksInOrder) {
  // Devs deliberately out of order, and ranks are kRankBase+dev, so neither a
  // skipped dev->rank translation nor a sorted write can pass by accident.
  const int devs[kGpus] = {3, 0, 5, 1, 7, 2, 6, 4};
  ncclXmlNode* channel = NewNode("channel");
  for (int dev : devs) AddChild(channel, "gpu", std::to_string(dev).c_str());

  ASSERT_EQ(ncclTopoGetChannelFromXml(channel, 0, system(), graph()), ncclSuccess);

  for (int i = 0; i < kGpus; i++) {
    EXPECT_EQ(graph()->intra[i], SearchXmlScene::kRankBase + devs[i]) << "slot " << i;
  }
}

TEST_F(SearchXmlChannelMicrotest, WellFormedChannel_TranscribesNetDevsToInter) {
  // The <dev> attribute of a net is parsed base 16 (see strtol(str, NULL, 16)).
  ncclXmlNode* channel = NewNode("channel");
  AddChild(channel, "net", "a");
  AddChild(channel, "gpu", "0");
  AddChild(channel, "net", "b");

  ASSERT_EQ(ncclTopoGetChannelFromXml(channel, 0, system(), graph()), ncclSuccess);

  EXPECT_EQ(graph()->inter[0], 0xa);
  EXPECT_EQ(graph()->inter[1], 0xb);
}

TEST_F(SearchXmlChannelMicrotest, ExplicitRankAttribute_IsUsedInsteadOfDevLookup) {
  // dev=0 would look up kRankBase+0; the explicit rank attribute must win.
  ncclXmlNode* channel = NewNode("channel");
  ncclXmlNode* gpu = AddChild(channel, "gpu", "0");
  SetAttr(gpu, "rank", "5");

  ASSERT_EQ(ncclTopoGetChannelFromXml(channel, 0, system(), graph()), ncclSuccess);

  EXPECT_EQ(graph()->intra[0], 5);
}

TEST_F(SearchXmlChannelMicrotest, ChannelIndex_SelectsItsOwnSliceOfIntraAndInter) {
  constexpr int kChannel = 1;
  constexpr int kSentinel = -42;
  // Dirty channel 0's slots: the fixture zeroes the graph, so "channel 0 is
  // untouched" would otherwise be true without the offset arithmetic running.
  for (int i = 0; i < kGpus; i++) graph()->intra[i] = kSentinel;
  graph()->inter[0] = kSentinel;
  graph()->inter[1] = kSentinel;

  ncclXmlNode* channel = NewNode("channel");
  AddChild(channel, "net", "1");
  AddChild(channel, "gpu", "6");
  AddChild(channel, "net", "2");

  ASSERT_EQ(ncclTopoGetChannelFromXml(channel, kChannel, system(), graph()), ncclSuccess);

  EXPECT_EQ(graph()->intra[kGpus * kChannel], SearchXmlScene::kRankBase + 6);
  EXPECT_EQ(graph()->inter[2 * kChannel], 1);
  EXPECT_EQ(graph()->inter[2 * kChannel + 1], 2);
  // Channel 0 is left alone.
  for (int i = 0; i < kGpus; i++) EXPECT_EQ(graph()->intra[i], kSentinel) << "slot " << i;
  EXPECT_EQ(graph()->inter[0], kSentinel);
  EXPECT_EQ(graph()->inter[1], kSentinel);
}

TEST_F(SearchXmlChannelMicrotest, UnknownGpuDev_IsRejected) {
  // kGpus devs exist (0..7), so dev 9 names no GPU in this system.
  ncclXmlNode* channel = NewNode("channel");
  AddChild(channel, "gpu", "9");

  EXPECT_EQ(ncclTopoGetChannelFromXml(channel, 0, system(), graph()), ncclSystemError);
}

TEST_F(SearchXmlChannelMicrotest,
       GraphGeneratedByNcclTopoGetXmlFromGraph_RoundTripsBackToTheSameGraph) {
  // RCCL ships no graph XML of its own, so the format its own writer emits is
  // the "our own input still parses" contract. Anything these guards reject
  // that ncclTopoGetXmlFromGraph can produce is a regression.
  const int ranks[kGpus] = {103, 100, 105, 101, 107, 102, 106, 104};
  system()->inter = 1;  // make the writer emit the leading and trailing <net>

  auto written = std::make_unique<ncclTopoGraph>();
  std::memset(written.get(), 0, sizeof(*written));
  written->id = 0;
  written->nChannels = 1;
  for (int i = 0; i < kGpus; i++) written->intra[i] = ranks[i];
  written->inter[0] = 0xa;
  written->inter[1] = 0xb;

  struct ncclXml* xml = nullptr;
  ASSERT_EQ(xmlAlloc(&xml, 64), ncclSuccess);
  std::unique_ptr<ncclXml, decltype(&std::free)> xmlOwner(xml, &std::free);
  struct ncclXmlNode* root = nullptr;
  ASSERT_EQ(xmlAddNode(xml, nullptr, "graph", &root), ncclSuccess);
  ASSERT_EQ(ncclTopoGetXmlFromChannel(written.get(), 0, system(), xml, root), ncclSuccess);
  ASSERT_EQ(root->nSubs, 1);

  ASSERT_EQ(ncclTopoGetChannelFromXml(root->subs[0], 0, system(), graph()), ncclSuccess);

  for (int i = 0; i < kGpus; i++) EXPECT_EQ(graph()->intra[i], ranks[i]) << "slot " << i;
  EXPECT_EQ(graph()->inter[0], 0xa);
  EXPECT_EQ(graph()->inter[1], 0xb);
}

// --- Rejection tests ------------------------------------------------------

TEST_F(SearchXmlChannelMicrotest, ChannelIndexBeyondMaxChannels_IsRejected) {
  // The caller indexes channels by XML sub position, bounded only by MAX_SUBS
  // (512), against graph->inter[MAXCHANNELS*2] (256 channels). At c ==
  // MAXCHANNELS the inter slice starts one element past the end of the array,
  // and the next fields of the same ncclTopoGraph are what get written --
  // so this is invisible to AddressSanitizer and must be asserted on the
  // clobbered fields.
  constexpr int kChannel = MAXCHANNELS;
  graph()->nIntraChannels = 0;
  graph()->intraNets[0] = 0;

  ncclXmlNode* channel = NewNode("channel");
  AddChild(channel, "net", "1122334455667788");

  EXPECT_EQ(ncclTopoGetChannelFromXml(channel, kChannel, system(), graph()), ncclInvalidUsage);
  EXPECT_EQ(graph()->nIntraChannels, 0);
  EXPECT_EQ(graph()->intraNets[0], 0);
}

TEST_F(SearchXmlChannelMicrotest, LastValidChannelIndex_IsStillAccepted) {
  // The tightest accepted input: one below the limit. Tightening the bound by
  // one has to break this, or the guard is free to creep inwards.
  constexpr int kChannel = MAXCHANNELS - 1;
  ncclXmlNode* channel = NewNode("channel");
  AddChild(channel, "net", "1");
  AddChild(channel, "gpu", "2");
  AddChild(channel, "net", "3");

  ASSERT_EQ(ncclTopoGetChannelFromXml(channel, kChannel, system(), graph()), ncclSuccess);

  EXPECT_EQ(graph()->inter[2 * kChannel], 1);
  EXPECT_EQ(graph()->inter[2 * kChannel + 1], 3);
  EXPECT_EQ(graph()->intra[kGpus * kChannel], SearchXmlScene::kRankBase + 2);
}

}  // namespace
