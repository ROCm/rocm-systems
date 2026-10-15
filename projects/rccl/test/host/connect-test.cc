/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/graph/connect.cc: per-rank presets from the search graphs, and the
// cross-node ring, tree, CollNet and NVLS wiring done in ncclTopoPostset.

#include <gtest/gtest.h>

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "fakes/graph_fakes.h"
#include "fakes/param_redirect.h"

#include CONNECT_CC_PATH

#include "TopoScene.h"

using namespace topo_scene;

namespace {

using GraphPtr = std::unique_ptr<ncclTopoGraph, FreeDeleter>;

class ConnectMicrotest : public ::testing::Test {
 protected:
  std::unique_ptr<ncclComm> comm_ = std::make_unique<ncclComm>();
  std::unique_ptr<ncclSharedResources> shared_ = std::make_unique<ncclSharedResources>();
  std::unique_ptr<ncclTopoGraph[]> graphStore_{new ncclTopoGraph[NCCL_NUM_ALGORITHMS]()};
  ncclTopoGraph* graphs_[NCCL_NUM_ALGORITHMS];
  SystemPtr system_;
  std::vector<int> rankToNode_, rankToLocalRank_;

  void SetUp() override {
    ResetGraphFakes();
    for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++) graphs_[a] = &graphStore_[a];
    comm_->sharedRes = shared_.get();
    shared_->owner = comm_.get();
    comm_->config.maxCTAs = MAXCHANNELS;
    comm_->config.minCTAs = 1;
  }
  void TearDown() override { ResetGraphFakes(); }

  // nNodes nodes of nGpus each, ranks node-major; the topology is one node's GPUs.
  void Use(int nGpus, int nNodes = 1, int rank = 0, const std::string& gcn = "gfx942") {
    system_ = BuildWithPaths(XgmiSystem(nGpus, 1, false, gcn));
    comm_->topo = system_.get();
    comm_->rank = rank;
    comm_->nRanks = nGpus * nNodes;
    comm_->nNodes = nNodes;
    comm_->node = rank / nGpus;
    comm_->localRanks = comm_->maxLocalRanks = nGpus;
    rankToNode_.resize(comm_->nRanks);
    rankToLocalRank_.resize(comm_->nRanks);
    for (int r = 0; r < comm_->nRanks; r++) {
      rankToNode_[r] = r / nGpus;
      rankToLocalRank_[r] = r % nGpus;
    }
    comm_->rankToNode = rankToNode_.data();
    comm_->rankToLocalRank = rankToLocalRank_.data();
  }

  void Intra(int algo, const std::vector<std::vector<int>>& channels) {
    ncclTopoGraph* graph = graphs_[algo];
    graph->nChannels = channels.size();
    for (size_t c = 0; c < channels.size(); c++)
      for (size_t i = 0; i < channels[c].size(); i++) graph->intra[c * channels[c].size() + i] = channels[c][i];
  }

  ncclTopoRanks Preset(ncclResult_t expect = ncclSuccess) {
    ncclTopoRanks ranks = {};
    EXPECT_EQ(expect, ncclTopoPreset(comm_.get(), graphs_, &ranks));
    return ranks;
  }

  // What ncclTopoPreset leaves in every channel before the rings and trees are connected.
  void ResetChannels() {
    for (int c = 0; c < MAXCHANNELS; c++) {
      ncclChannel* channel = comm_->channels + c;
      channel->ring.prev = channel->ring.next = channel->tree.up = -1;
      for (int i = 0; i < NCCL_MAX_TREE_ARITY; i++) channel->tree.down[i] = -1;
    }
  }

  // Presets of every rank with node n's ranks in order on even channels and reversed on odd ones.
  std::vector<ncclTopoRanks> AllRanks(int nGpus, int nNodes, int nChannels) {
    std::vector<ncclTopoRanks> all(nGpus * nNodes);
    for (int r = 0; r < nGpus * nNodes; r++) {
      ncclTopoRanks& t = all[r];
      for (int c = 0; c < MAXCHANNELS; c++) {
        t.ringNext[c] = t.ringPrev[c] = t.ringSend[c] = t.ringRecv[c] = -1;
        t.treeToParent[c] = t.treeToChild0[c] = t.treeToChild1[c] = -1;
      }
      int base = r / nGpus * nGpus;
      for (int c = 0; c < nChannels; c++) {
        std::vector<int> order;
        for (int i = 0; i < nGpus; i++) order.push_back(base + (c % 2 ? nGpus - 1 - i : i));
        int i = 0;
        while (order[i] != r) i++;
        t.ringRecv[c] = order[0];
        t.ringSend[c] = order[nGpus - 1];
        t.ringPrev[c] = i == 0 ? -1 : order[i - 1];
        t.ringNext[c] = i == nGpus - 1 ? -1 : order[i + 1];
        t.treeToParent[c] = order[0];
        t.treeToChild0[c] = order[nGpus > 1 ? 1 : 0];
        t.treeToChild1[c] = order[0];
      }
    }
    return all;
  }

  ncclResult_t Postset(std::vector<ncclTopoRanks>& all, int nc, std::vector<int>& rings, ncclComm* parent = nullptr) {
    std::vector<ncclTopoRanks*> ptrs;
    for (ncclTopoRanks& t : all) ptrs.push_back(&t);
    std::vector<int> firstRanks;
    for (int n = 0; n < comm_->nNodes; n++) firstRanks.push_back(n * comm_->nRanks / comm_->nNodes);
    rings.assign(MAXCHANNELS * comm_->nRanks, -1);
    return ncclTopoPostset(comm_.get(), firstRanks.data(), nullptr, ptrs.data(), rings.data(), graphs_, parent, nc);
  }

  std::vector<int> Ring(const std::vector<int>& rings, int c) {
    return std::vector<int>(rings.begin() + c * comm_->nRanks, rings.begin() + (c + 1) * comm_->nRanks);
  }
};

// --- ncclTopoPreset

TEST_F(ConnectMicrotest, Preset_RingTreeAndChainNeighboursFromTheIntraOrder) {
  Use(4, 1, 1);
  comm_->nChannels = 2;
  Intra(NCCL_ALGO_RING, {{0, 1, 2, 3}, {3, 2, 1, 0}});
  Intra(NCCL_ALGO_TREE, {{0, 1, 2, 3}, {2, 1, 0, 3}});
  graphs_[NCCL_ALGO_TREE]->pattern = NCCL_TOPO_PATTERN_BALANCED_TREE;
  Intra(NCCL_ALGO_COLLNET_CHAIN, {{1, 0, 2, 3}, {0, 1, 2, 3}});
  graphs_[NCCL_ALGO_RING]->crossNic = 2;

  ncclTopoRanks t = Preset();

  EXPECT_EQ(2, t.crossNicRing);
  EXPECT_EQ(std::vector<int>({0, 3, 0, 2}),
            std::vector<int>({t.ringRecv[0], t.ringSend[0], t.ringPrev[0], t.ringNext[0]}));
  EXPECT_EQ(std::vector<int>({3, 0, 2, 0}),
            std::vector<int>({t.ringRecv[1], t.ringSend[1], t.ringPrev[1], t.ringNext[1]}));
  EXPECT_EQ(-1, t.ringNext[2]);
  EXPECT_EQ(std::vector<int>({0, 1, 0}), std::vector<int>({t.treeToParent[0], t.treeToChild0[0], t.treeToChild1[0]}));
  EXPECT_EQ(std::vector<int>({2, 1, 2}), std::vector<int>({t.treeToParent[1], t.treeToChild0[1], t.treeToChild1[1]}));
  ncclChannel* ch = comm_->channels;
  EXPECT_EQ(0, ch[0].tree.up);
  EXPECT_EQ(2, ch[0].tree.down[0]);
  EXPECT_EQ(2, ch[1].tree.up);
  EXPECT_EQ(0, ch[1].tree.down[0]);
  EXPECT_EQ(4, ch[0].collnetChain.up);  // first in the chain: up is the network
  EXPECT_EQ(0, ch[0].collnetChain.down[0]);
  EXPECT_EQ(0, ch[1].collnetChain.up);
  EXPECT_EQ(2, ch[2].tree.down[0]);  // channels 2-3 duplicate 0-1
  EXPECT_EQ(0, ch[3].collnetChain.up);
  EXPECT_EQ(-1, ch[4].tree.up);
  EXPECT_EQ(-1, ch[4].collnetDirect.heads[0]);
}

TEST_F(ConnectMicrotest, Preset_EndsOfTheOrderHaveNoIntraNeighbour) {
  Use(3, 1, 0);
  comm_->nChannels = 1;
  Intra(NCCL_ALGO_RING, {{0, 1, 2}});
  Intra(NCCL_ALGO_TREE, {{0, 1, 2}});
  Intra(NCCL_ALGO_COLLNET_CHAIN, {{2, 1, 0}});

  ncclTopoRanks t = Preset();

  EXPECT_EQ(-1, t.ringPrev[0]);
  EXPECT_EQ(1, t.ringNext[0]);
  EXPECT_EQ(-1, comm_->channels[0].tree.up);
  EXPECT_EQ(-1, comm_->channels[0].collnetChain.down[0]);
  comm_->rank = 2;
  t = Preset();
  EXPECT_EQ(-1, t.ringNext[0]);
  EXPECT_EQ(-1, comm_->channels[0].tree.down[0]);
}

TEST_F(ConnectMicrotest, Preset_TreePatternPicksTheChildRanks) {
  Use(3, 1, 0);
  comm_->nChannels = 1;
  Intra(NCCL_ALGO_RING, {{0, 1, 2}});
  Intra(NCCL_ALGO_TREE, {{2, 0, 1}});
  struct Case {
    int pattern, child0, child1;
  } cases[] = {
    {NCCL_TOPO_PATTERN_TREE, 2, 2}, {NCCL_TOPO_PATTERN_SPLIT_TREE, 0, 0}, {NCCL_TOPO_PATTERN_BALANCED_TREE, 0, 2}
  };

  for (const Case& c : cases) {
    graphs_[NCCL_ALGO_TREE]->pattern = c.pattern;
    ncclTopoRanks t = Preset();
    EXPECT_EQ(2, t.treeToParent[0]) << c.pattern;
    EXPECT_EQ(c.child0, t.treeToChild0[0]) << c.pattern;
    EXPECT_EQ(c.child1, t.treeToChild1[0]) << c.pattern;
  }
}

TEST_F(ConnectMicrotest, Preset_NvlsHeadsAreTheDistinctFirstRanks) {
  Use(2, 1, 0);
  comm_->nChannels = 1;
  Intra(NCCL_ALGO_RING, {{0, 1}});
  Intra(NCCL_ALGO_TREE, {{0, 1}});
  Intra(NCCL_ALGO_NVLS, {{0, 1}, {1, 0}, {0, 1}});

  ncclTopoRanks t = Preset();

  ASSERT_EQ(2, t.nvlsHeadNum);
  EXPECT_EQ(0, t.nvlsHeads[0]);
  EXPECT_EQ(1, t.nvlsHeads[1]);
  EXPECT_EQ(-1, t.nvlsHeads[2]);
  EXPECT_EQ(1, comm_->nvlsHeads[1]);
}

TEST_F(ConnectMicrotest, Preset_RejectsIncompleteInputs) {
  Use(2, 1, 0);
  comm_->nChannels = 1;
  Intra(NCCL_ALGO_RING, {{0, 1}});
  Intra(NCCL_ALGO_TREE, {{0, 1}});
  ncclTopoRanks t;

  EXPECT_EQ(ncclInvalidArgument, ncclTopoPreset(nullptr, graphs_, &t));
  EXPECT_EQ(ncclInvalidArgument, ncclTopoPreset(comm_.get(), graphs_, nullptr));
  comm_->topo = nullptr;
  Preset(ncclInternalError);
  EXPECT_TRUE(Logged("TopoPreset: Communicator state is incomplete"));
  comm_->topo = system_.get();

  for (int n : {0, MAXCHANNELS + 1}) {
    comm_->nChannels = n;
    Preset(ncclInvalidUsage);
  }
  EXPECT_TRUE(Logged("TopoPreset: invalid nChannels=0 should be [1,"));
  comm_->nChannels = 2;
  Preset(ncclInternalError);
  EXPECT_TRUE(Logged("graph channel count smaller than comm->nChannels"));
  Intra(NCCL_ALGO_RING, {{0, 1}, {1, 0}});
  Preset(ncclInternalError);  // the tree graph is still short
  Intra(NCCL_ALGO_RING, {{0, 1}});
  comm_->nChannels = 1;

  graphs_[NCCL_ALGO_TREE] = nullptr;
  Preset(ncclInternalError);
  EXPECT_TRUE(Logged("Required topology graphs (Ring/Tree) are missing"));
  graphs_[NCCL_ALGO_TREE] = &graphStore_[NCCL_ALGO_TREE];
  graphs_[NCCL_ALGO_RING] = nullptr;
  Preset(ncclInternalError);
  graphs_[NCCL_ALGO_RING] = &graphStore_[NCCL_ALGO_RING];

  comm_->rank = 5;
  Preset(ncclInternalError);
  EXPECT_TRUE(Logged("Local rank 5 not found in intra-node graph"));
  comm_->rank = 0;

  system_->nodes[GPU].count = 0;
  Preset(ncclInternalError);
  system_->nodes[GPU].count = 2;
  EXPECT_TRUE(Logged("TopoPreset: invalid localRanks=0"));
}

TEST_F(ConnectMicrotest, Preset_IntraGraphGenPermutesChannelZerosRing) {
  Use(4, 1, 0);
  comm_->nChannels = 2;
  Intra(NCCL_ALGO_RING, {{0, 1, 2, 3}, {3, 3, 3, 3}});
  Intra(NCCL_ALGO_TREE, {{0, 1, 2, 3}, {0, 1, 2, 3}});
  g_params["RCCL_INTRA_GRAPH_GEN"] = 1;
  g_generateRings = [](int n, uint32_t nChannels, int* order) {
    const int orders[2][4] = {{0, 1, 2, 3}, {2, 0, 3, 1}};
    for (uint32_t c = 0; c < nChannels; c++)
      for (int i = 0; i < n; i++) order[c * n + i] = orders[c][i];
    return ncclSuccess;
  };

  ncclTopoRanks t = Preset();

  EXPECT_EQ(std::vector<int>({2, 0, 3, 1}),
            std::vector<int>(graphs_[NCCL_ALGO_RING]->intra + 4, graphs_[NCCL_ALGO_RING]->intra + 8));
  EXPECT_EQ(2, t.ringRecv[1]);
  EXPECT_EQ(2, t.ringPrev[1]);
  EXPECT_EQ(3, t.ringNext[1]);
  EXPECT_EQ(-1, t.ringPrev[0]);  // channel 0 stays open at the node ends
}

TEST_F(ConnectMicrotest, Preset_InterGraphGenClosesEachRingAtItsCut) {
  Use(4, 1, 0);
  comm_->nChannels = 1;
  Intra(NCCL_ALGO_RING, {{0, 1, 2, 3}});
  Intra(NCCL_ALGO_TREE, {{0, 1, 2, 3}});
  g_params["RCCL_INTRA_GRAPH_GEN"] = 1;
  g_params["RCCL_INTER_GRAPH_GEN"] = 1;
  g_findRingCutIndices = [](int nChannels, int, const int*, int* cut) {
    for (int c = 0; c < nChannels; c++) cut[c] = 1;
  };

  ncclTopoRanks t = Preset();

  EXPECT_EQ(2, t.ringRecv[0]);  // the ring leaves after index 1 and comes back in at index 2
  EXPECT_EQ(1, t.ringSend[0]);
  EXPECT_EQ(3, t.ringPrev[0]);  // wraps inside the node
  EXPECT_EQ(1, t.ringNext[0]);
}

TEST_F(ConnectMicrotest, Preset_Gfx11And12KeepOneRingOrderUnlessP2pIsOff) {
  for (const char* gcn : {"gfx1100", "gfx1201"}) {
    Use(4, 1, 0, gcn);
    comm_->nChannels = 2;
    Intra(NCCL_ALGO_RING, {{0, 1, 2, 3}, {3, 3, 3, 3}});
    Intra(NCCL_ALGO_TREE, {{0, 1, 2, 3}, {0, 1, 2, 3}});
    g_params["RCCL_INTRA_GRAPH_GEN"] = 1;
    g_generateRings = [](int n, uint32_t nChannels, int* order) {
      for (uint32_t c = 0; c < nChannels; c++)
        for (int i = 0; i < n; i++) order[c * n + i] = (i + c) % n;
      return ncclSuccess;
    };

    Preset();
    EXPECT_EQ(std::vector<int>({0, 1, 2, 3}),
              std::vector<int>(graphs_[NCCL_ALGO_RING]->intra + 4, graphs_[NCCL_ALGO_RING]->intra + 8))
      << gcn;

    g_params.erase("RCCL_INTRA_GRAPH_GEN");
    g_params["P2P_DISABLE"] = 1;  // P2P off turns graph generation on, with diverse rings
    Intra(NCCL_ALGO_RING, {{0, 1, 2, 3}, {3, 3, 3, 3}});
    Preset();
    EXPECT_EQ(std::vector<int>({1, 2, 3, 0}),
              std::vector<int>(graphs_[NCCL_ALGO_RING]->intra + 4, graphs_[NCCL_ALGO_RING]->intra + 8))
      << gcn;
    g_params.clear();
  }
}

TEST_F(ConnectMicrotest, Preset_GraphGenFailureIsReturned) {
  Use(2, 1, 0);
  comm_->nChannels = 1;
  Intra(NCCL_ALGO_RING, {{0, 1}});
  Intra(NCCL_ALGO_TREE, {{0, 1}});
  g_params["RCCL_INTRA_GRAPH_GEN"] = 1;
  g_generateRings = [](int, uint32_t, int*) { return ncclInternalError; };

  Preset(ncclInternalError);
}

TEST_F(ConnectMicrotest, Preset_OverHalfTheChannelsAreNotDuplicated) {
  Use(2, 1, 0);
  int n = MAXCHANNELS / 2 + 1;
  comm_->nChannels = n;
  std::vector<std::vector<int>> rings(n, std::vector<int>({1, 0}));
  Intra(NCCL_ALGO_RING, rings);
  Intra(NCCL_ALGO_TREE, rings);

  Preset();

  EXPECT_EQ(1, comm_->channels[0].tree.up);
  EXPECT_EQ(1, comm_->channels[n - 1].tree.up);
  EXPECT_EQ(-1, comm_->channels[n].tree.up);  // a copy would run past MAXCHANNELS
}

// --- Tree strings

TEST_F(ConnectMicrotest, IsRankHere_FindsWholeNumbersInTheRange) {
  const char* s = "(0(1)(23))";
  int len = strlen(s);

  EXPECT_TRUE(isRankHere(s, 0, len, 23));
  EXPECT_TRUE(isRankHere(s, 0, len, 1));
  EXPECT_FALSE(isRankHere(s, 0, len, 2));  // only part of 23
  EXPECT_FALSE(isRankHere(s, 0, len, 3));
  EXPECT_FALSE(isRankHere(s, 4, len, 1));
  EXPECT_FALSE(isRankHere(s, 3, 3, 1));
  EXPECT_FALSE(isRankHere(s, -1, len, 0));
  EXPECT_FALSE(isRankHere(s, 0, -1, 0));
}

struct TreeLinks {
  int up, down0, down1, down2;
  bool operator==(const TreeLinks& o) const {
    return up == o.up && down0 == o.down0 && down1 == o.down1 && down2 == o.down2;
  }
};

std::ostream& operator<<(std::ostream& os, const TreeLinks& t) {
  return os << t.up << "/" << t.down0 << "/" << t.down1 << "/" << t.down2;
}

class ConnectTreeBaseMicrotest : public ConnectMicrotest {
 protected:
  GraphPtr tree_{static_cast<ncclTopoGraph*>(calloc(1, sizeof(ncclTopoGraph)))};

  void SetTrees(const std::vector<const char*>& trees) {
    for (size_t i = 0; i < trees.size(); i++) strcpy(tree_->treeBase[i], trees[i]);
  }

  TreeLinks Links(int rank, int c = 0) {
    comm_->rank = rank;
    EXPECT_EQ(ncclSuccess, ncclTreeBasePostset(comm_.get(), tree_.get()));
    ncclTree& t = comm_->channels[c].tree;
    return {t.up, t.down[0], t.down[1], t.down[2]};
  }
};

TEST_F(ConnectTreeBaseMicrotest, TreeBasePostset_ReadsParentsAndChildren) {
  comm_->nChannels = 1;
  SetTrees({"(0(1(3))(2(4)(5)))"});

  EXPECT_EQ((TreeLinks{-1, 1, 2, -1}), Links(0));
  EXPECT_EQ((TreeLinks{0, 3, -1, -1}), Links(1));
  EXPECT_EQ((TreeLinks{0, 4, 5, -1}), Links(2));
  EXPECT_EQ((TreeLinks{1, -1, -1, -1}), Links(3));
  EXPECT_EQ((TreeLinks{2, -1, -1, -1}), Links(5));
}

TEST_F(ConnectTreeBaseMicrotest, TreeBasePostset_MultiDigitRanksAndThreeChildren) {
  comm_->nChannels = 1;
  SetTrees({"(10(11)(12(15))(13))"});

  EXPECT_EQ((TreeLinks{-1, 11, 12, 13}), Links(10));
  EXPECT_EQ((TreeLinks{10, 15, -1, -1}), Links(12));
  EXPECT_EQ((TreeLinks{12, -1, -1, -1}), Links(15));
}

TEST_F(ConnectTreeBaseMicrotest, TreeBasePostset_ChannelsCycleThroughTheTrees) {
  comm_->nChannels = 3;
  SetTrees({"(0(1))", "(1(0))"});

  Links(0);

  EXPECT_EQ(-1, comm_->channels[0].tree.up);
  EXPECT_EQ(1, comm_->channels[1].tree.up);
  EXPECT_EQ(-1, comm_->channels[2].tree.up);
  EXPECT_EQ(1, comm_->channels[2].tree.down[0]);
}

TEST_F(ConnectTreeBaseMicrotest, TreeBasePostset_NoTreesLeavesTheChannelsAlone) {
  comm_->nChannels = 1;
  comm_->channels[0].tree.up = 7;

  Links(0);

  EXPECT_EQ(7, comm_->channels[0].tree.up);
}

// --- Rings and trees across nodes

TEST_F(ConnectMicrotest, ConnectRings_EachNodesExitFeedsTheNextNodesEntry) {
  Use(2, 3, 1);
  comm_->nChannels = 1;
  std::vector<int> recv = {0, 2, 4}, send = {1, 3, 5}, prev(6, -1), next(6, -1);

  ASSERT_EQ(ncclSuccess, connectRings(comm_.get(), recv.data(), send.data(), prev.data(), next.data()));

  EXPECT_EQ(std::vector<int>({5, -1, 1, -1, 3, -1}), prev);
  EXPECT_EQ(std::vector<int>({-1, 2, -1, 4, -1, 0}), next);
  EXPECT_FALSE(Logged("[RINGS]"));
  comm_->rank = 0;
  ASSERT_EQ(ncclSuccess, connectRings(comm_.get(), recv.data(), send.data(), prev.data(), next.data()));
  EXPECT_TRUE(Logged("[RINGS]      00"));
  EXPECT_TRUE(Logged("[RINGS]  00->01"));
}

TEST_F(ConnectMicrotest, ConnectRingsLoadBalanced_FollowsTheGeneratedNodeOrder) {
  Use(2, 3, 1);
  comm_->nChannels = 2;
  g_generateRings = [](int n, uint32_t nChannels, int* order) {
    const int orders[2][3] = {{0, 1, 2}, {0, 2, 1}};
    for (uint32_t c = 0; c < nChannels; c++)
      for (int i = 0; i < n; i++) order[c * n + i] = orders[c][i];
    return ncclSuccess;
  };
  std::vector<int> recv = {0, 2, 4, 0, 2, 4}, send = {1, 3, 5, 1, 3, 5}, prev(12, -1), next(12, -1);

  ASSERT_EQ(ncclSuccess, connectRingsLoadBalanced(comm_.get(), recv.data(), send.data(), prev.data(), next.data()));

  EXPECT_EQ(2, next[1]);
  EXPECT_EQ(5, prev[0]);
  EXPECT_EQ(4, next[6 + 1]);  // channel 1 goes node 0 -> 2 -> 1
  EXPECT_EQ(5, prev[6 + 2]);
  EXPECT_EQ(0, next[6 + 3]);

  g_generateRings = [](int, uint32_t, int*) { return ncclInternalError; };
  EXPECT_EQ(ncclInternalError,
            connectRingsLoadBalanced(comm_.get(), recv.data(), send.data(), prev.data(), next.data()));
  comm_->nNodes = 0;
  EXPECT_EQ(ncclInvalidArgument,
            connectRingsLoadBalanced(comm_.get(), recv.data(), send.data(), prev.data(), next.data()));
}

// Node n's ranks are 2n (parent side, child1) and 2n+1 (child0) on every channel.
struct NodeTrees {
  std::vector<int> parent, child0, child1;
  NodeTrees(int nNodes, int nChannels) {
    for (int c = 0; c < nChannels; c++)
      for (int n = 0; n < nNodes; n++) {
        parent.push_back(2 * n);
        child0.push_back(2 * n + 1);
        child1.push_back(2 * n);
      }
  }
};

TEST_F(ConnectMicrotest, ConnectTrees_LinksNodesAlongBothBinaryTrees) {
  Use(2, 4, 4);  // node 2
  comm_->nChannels = 1;
  ResetChannels();
  NodeTrees t(4, 2);

  ASSERT_EQ(ncclSuccess, connectTrees(comm_.get(), t.parent.data(), t.child0.data(), t.child1.data(), nullptr));

  ncclTree& first = comm_->channels[0].tree;
  ncclTree& second = comm_->channels[1].tree;
  EXPECT_EQ(0, first.up);  // node 2 is node 0's second child
  EXPECT_EQ(6, first.down[0]);  // node 3 hangs off node 2
  EXPECT_EQ(-1, first.down[1]);
  EXPECT_EQ(3, second.up);  // mirrored: node 2 is node 1's first child
  EXPECT_EQ(3, first.depth);

  comm_->rank = 5;
  ResetChannels();
  ASSERT_EQ(ncclSuccess, connectTrees(comm_.get(), t.parent.data(), t.child0.data(), t.child1.data(), nullptr));
  EXPECT_EQ(-1, comm_->channels[0].tree.up);
  EXPECT_EQ(2, comm_->channels[0].tree.down[0]);  // node 1 is node 2's first child
  EXPECT_TRUE(Logged("Tree 0 : -1 -> 5 -> 2/-1/-1"));
}

TEST_F(ConnectMicrotest, ConnectTrees_OverTheChannelLimitTheSecondHalfHasItsOwnRanks) {
  Use(2, 4, 4, "gfx90a");
  comm_->nChannels = 2 * CHANNEL_LIMIT;
  ResetChannels();
  NodeTrees t(4, 2 * CHANNEL_LIMIT);
  for (int n = 0; n < 4; n++)
    t.child0[CHANNEL_LIMIT * 4 + n] = t.child1[CHANNEL_LIMIT * 4 + n] = t.parent[CHANNEL_LIMIT * 4 + n] = 2 * n;

  ASSERT_EQ(ncclSuccess, connectTrees(comm_.get(), t.parent.data(), t.child0.data(), t.child1.data(), nullptr));

  EXPECT_EQ(0, comm_->channels[0].tree.up);
  EXPECT_EQ(6, comm_->channels[0].tree.down[0]);
  EXPECT_EQ(2, comm_->channels[CHANNEL_LIMIT].tree.up);  // second tree, read from the second half's ranks
  EXPECT_EQ(-1, comm_->channels[CHANNEL_LIMIT].tree.down[0]);
  EXPECT_EQ(3, comm_->channels[CHANNEL_LIMIT].tree.depth);

  comm_->rank = 5;
  ResetChannels();
  ASSERT_EQ(ncclSuccess, connectTrees(comm_.get(), t.parent.data(), t.child0.data(), t.child1.data(), nullptr));
  EXPECT_EQ(2, comm_->channels[0].tree.down[0]);  // node 1 is node 2's first child
  EXPECT_EQ(-1, comm_->channels[CHANNEL_LIMIT].tree.up);
}

TEST_F(ConnectMicrotest, SetTreeDown_AFourthChildIsAnError) {
  ncclTree tree = {};
  int ranks[4] = {10, 11, 12, 13};
  tree.down[0] = tree.down[1] = tree.down[2] = -1;

  for (int d = 0; d < 3; d++) ASSERT_EQ(ncclSuccess, setTreeDown(&tree, ranks, d));
  EXPECT_EQ(ncclSuccess, setTreeDown(&tree, ranks, -1));
  EXPECT_EQ(ncclInternalError, setTreeDown(&tree, ranks, 3));

  EXPECT_EQ(12, tree.down[2]);
  EXPECT_TRUE(Logged("tree already has 3 children (10 11 12)"));
}

TEST_F(ConnectMicrotest, GetTreeNodeParity_FlipsOnFirstChildren) {
  EXPECT_EQ(-1, getTreeNodeParity(0, 4, -1));
  EXPECT_EQ(0, getTreeNodeParity(0, 4, 0));  // the first tree's root
  EXPECT_EQ(1, getTreeNodeParity(1, 4, 3));  // the second tree's root
  EXPECT_EQ(0, getTreeNodeParity(0, 4, 2));  // second child of the root
  EXPECT_EQ(1, getTreeNodeParity(0, 4, 1));  // first child of node 2
  EXPECT_EQ(0, getTreeNodeParity(0, 4, 3));
}

TEST_F(ConnectMicrotest, ConnectRailOptimizedTrees_SwapsChannelPairsOnOddParityNodes) {
  Use(2, 2, 2);  // node 1
  comm_->nChannels = 4;
  ResetChannels();
  // Complementary pairs: even channels lead with 2n, odd ones with 2n+1.
  std::vector<int> parent, child0, child1;
  for (int c = 0; c < 4; c++)
    for (int n = 0; n < 2; n++) {
      parent.push_back(c % 2 ? 2 * n + 1 : 2 * n);
      child0.push_back(c % 2 ? 2 * n : 2 * n + 1);
      child1.push_back(c % 2 ? 2 * n + 1 : 2 * n);
    }
  ncclTopoGraph& tree = comm_->graphs[NCCL_ALGO_TREE];
  for (int c = 0; c < 4; c++) {
    tree.inter[2 * c] = 100 + c;
    for (int g = 0; g < 2; g++) tree.intra[c * 2 + g] = 10 * c + g;
  }

  ASSERT_EQ(ncclSuccess, connectRailOptimizedTrees(comm_.get(), parent.data(), child0.data(), child1.data()));

  EXPECT_EQ(0, comm_->channels[0].tree.up);  // up-tree: node 1 under node 0, no swap
  EXPECT_EQ(-1, comm_->channels[1].tree.up);
  EXPECT_EQ(0, comm_->channels[3].tree.down[0]);  // down-tree: node 1 is the root and swaps channels 2 and 3
  EXPECT_EQ(103, tree.inter[4]);
  EXPECT_EQ(102, tree.inter[6]);
  EXPECT_EQ(30, tree.intra[4]);
  EXPECT_EQ(2, comm_->channels[0].tree.depth);
  EXPECT_TRUE(Logged("Building rail-optimized trees for 2 nodes"));
}

TEST_F(ConnectMicrotest, ConnectRailOptimizedTrees_ChildRanksFollowTheChildNodesParity) {
  Use(2, 4, 5);  // node 2's second rank
  comm_->nChannels = 4;
  ResetChannels();
  std::vector<int> parent, child0, child1;
  for (int c = 0; c < 4; c++)
    for (int n = 0; n < 4; n++) {
      parent.push_back(c % 2 ? 2 * n + 1 : 2 * n);
      child0.push_back(c % 2 ? 2 * n : 2 * n + 1);
      child1.push_back(c % 2 ? 2 * n + 1 : 2 * n);
    }

  ASSERT_EQ(ncclSuccess, connectRailOptimizedTrees(comm_.get(), parent.data(), child0.data(), child1.data()));

  EXPECT_EQ(3, comm_->channels[0].tree.down[0]);  // node 1 has odd parity: its parent rank on channel 1
  EXPECT_EQ(1, comm_->channels[1].tree.up);
  EXPECT_EQ(7, comm_->channels[1].tree.down[0]);
  EXPECT_EQ(3, comm_->channels[3].tree.up);  // down-tree: node 2 under node 1
  EXPECT_EQ(3, comm_->channels[0].tree.depth);
}

// --- CollNet and NVLS

TEST_F(ConnectMicrotest, ConnectCollNet_HeadsSendDownAndEveryoneSendsToTheOtherHeads) {
  Use(4, 2, 2);
  comm_->nChannels = 1;
  Intra(NCCL_ALGO_COLLNET_DIRECT, {{0, 1, 2, 3}, {2, 3, 0, 1}, {0, 2, 1, 3}});

  ASSERT_EQ(ncclSuccess, connectCollNet(comm_.get(), graphs_[NCCL_ALGO_COLLNET_DIRECT]));

  ncclDirect& head = comm_->channels[0].collnetDirect;
  EXPECT_EQ(1, head.headRank);
  EXPECT_EQ(8, head.out);
  EXPECT_EQ(std::vector<int>({3, 0, 1}), std::vector<int>(head.down, head.down + 3));
  EXPECT_EQ(0, head.up[0]);
  EXPECT_EQ(std::vector<int>({2, 0}), std::vector<int>(head.heads, head.heads + 2));
  EXPECT_EQ(2, head.nHeads);
  EXPECT_EQ(0, head.shift);
  EXPECT_EQ(2, head.depth);

  comm_->rank = 1;
  comm_->channels[0].collnetDirect.headRank = -1;
  ASSERT_EQ(ncclSuccess, connectCollNet(comm_.get(), graphs_[NCCL_ALGO_COLLNET_DIRECT]));
  ncclDirect& leaf = comm_->channels[0].collnetDirect;
  EXPECT_EQ(-1, leaf.headRank);
  EXPECT_EQ(std::vector<int>({0, 2}), std::vector<int>(leaf.up, leaf.up + 2));
  EXPECT_EQ(std::vector<int>({0, 2}), std::vector<int>(leaf.heads, leaf.heads + 2));
  EXPECT_EQ(1, leaf.shift);
}

TEST_F(ConnectMicrotest, ConnectCollNet_LoneHeadHasDepthOne) {
  Use(1, 1, 0);
  comm_->nChannels = 1;
  comm_->channels[0].collnetDirect.headRank = -1;
  Intra(NCCL_ALGO_COLLNET_DIRECT, {{0}});

  ASSERT_EQ(ncclSuccess, connectCollNet(comm_.get(), graphs_[NCCL_ALGO_COLLNET_DIRECT]));

  EXPECT_EQ(1, comm_->channels[0].collnetDirect.depth);
}

TEST_F(ConnectMicrotest, ConnectNvls_SingleNodeHeadsAndNoTree) {
  Use(2, 1, 1);
  comm_->nvlsChannels = 2;
  comm_->config.collnetEnable = 1;
  int heads[2] = {0, 1};

  ASSERT_EQ(ncclSuccess, connectNvls(comm_.get(), heads, 2));

  ncclNvls& nvls = comm_->channels[1].nvls;
  EXPECT_EQ(2, nvls.nHeads);
  EXPECT_EQ(3, nvls.up[0]);
  EXPECT_EQ(4, nvls.up[1]);
  EXPECT_EQ(-1, nvls.up[2]);
  EXPECT_EQ(4, nvls.down);  // head 1
  EXPECT_EQ(1, nvls.headRank);
  EXPECT_EQ(2, nvls.out);
  EXPECT_EQ(-1, nvls.treeUp);

  ASSERT_EQ(ncclSuccess, connectNvls(comm_.get(), heads, 0));
  EXPECT_EQ(0, comm_->nvlsChannels);
}

TEST_F(ConnectMicrotest, ConnectNvls_MultiNodeTreesFollowTheHeadsOnEachNode) {
  Use(2, 4, 4);  // node 2
  comm_->nvlsChannels = 2;
  int heads[2 * 4] = {0, 2, 4, 6, 1, 3, 5, 7};  // head h on node n

  ASSERT_EQ(ncclSuccess, connectNvls(comm_.get(), heads, 2));

  ncclNvls& c0 = comm_->channels[0].nvls;
  ncclNvls& c1 = comm_->channels[1].nvls;
  EXPECT_EQ(0, c0.headRank);
  EXPECT_EQ(-1, c0.out);
  EXPECT_EQ(0, c0.treeUp);  // first tree: node 2's parent is node 0
  EXPECT_EQ(c0.down, c0.treeDown[0]);
  EXPECT_EQ(2, c0.treeDown[1]);  // node 1
  EXPECT_EQ(6, c0.treeDown[2]);  // node 3
  EXPECT_EQ(2, c1.treeUp);  // second tree: node 2's parent is node 1
  EXPECT_EQ(-1, c1.treeDown[1]);

  comm_->rank = 0;
  comm_->node = 0;
  ASSERT_EQ(ncclSuccess, connectNvls(comm_.get(), heads, 2));
  EXPECT_TRUE(Logged("NVLS Head  0:  0  2  4  6"));
}

// --- Channel counts

TEST_F(ConnectMicrotest, MinNchannels_NewNameWinsAndTheRangeIsClamped) {
  EXPECT_EQ(2, ncclMinNchannels());
  g_params["MIN_NRINGS"] = 5;
  EXPECT_EQ(5, ncclMinNchannels());
  g_params["MIN_NCHANNELS"] = 7;
  EXPECT_EQ(7, ncclMinNchannels());
  g_params["MIN_NCHANNELS"] = MAXCHANNELS + 1;
  EXPECT_EQ(MAXCHANNELS, ncclMinNchannels());
  g_params["MIN_NCHANNELS"] = -3;
  EXPECT_EQ(0, ncclMinNchannels());
}

TEST_F(ConnectMicrotest, MaxNchannels_NewNameWinsAndTheWorkArgsBoundIt) {
  EXPECT_EQ(MAXCHANNELS, ncclMaxNchannels());
  g_params["MAX_NRINGS"] = 9;
  EXPECT_EQ(9, ncclMaxNchannels());
  g_params["MAX_NCHANNELS"] = 11;
  EXPECT_EQ(11, ncclMaxNchannels());
  g_params["MAX_NCHANNELS"] = MAXCHANNELS * 2;
  EXPECT_EQ(MAXCHANNELS, ncclMaxNchannels());
  g_params["MAX_NCHANNELS"] = 0;
  EXPECT_EQ(1, ncclMaxNchannels());
  g_params.erase("MAX_NCHANNELS");
  g_params.erase("MAX_NRINGS");
  g_params["WORK_ARGS_BYTES"] = sizeof(ncclDevKernelArgs) + 3 * sizeof(ncclDevWorkBatch);
  EXPECT_EQ(3, ncclMaxNchannels());
}

TEST_F(ConnectMicrotest, CopyChannels_RepeatsTheFirstChannels) {
  Use(2, 1, 0);
  std::vector<int> prev(2 * 6, -1), next(2 * 6, -1);
  for (int c = 0; c < 2; c++) {
    prev[2 * c] = 10 + c;
    next[2 * c] = 20 + c;
    comm_->channels[c].ring.prev = 30 + c;
  }

  EXPECT_EQ(5, copyChannels(comm_.get(), 2, 5, prev.data(), next.data()));

  EXPECT_EQ(10, prev[2 * 2]);
  EXPECT_EQ(21, next[2 * 3]);
  EXPECT_EQ(10, prev[2 * 4]);  // channel 4 copies channel 2, itself a copy of 0
  EXPECT_EQ(31, comm_->channels[3].ring.prev);
  EXPECT_EQ(-1, prev[2 * 5]);
}

TEST_F(ConnectMicrotest, RepairMissingChannels_FillsEmptyChannelsFromChannelZero) {
  std::vector<ncclTopoRanks> all(1);
  ncclTopoRanks& t = all[0];
  for (int c = 0; c < 3; c++) {
    t.ringNext[c] = t.ringPrev[c] = t.ringSend[c] = t.ringRecv[c] = -1;
    t.treeToParent[c] = t.treeToChild0[c] = t.treeToChild1[c] = -1;
  }
  t.ringNext[0] = 1, t.ringPrev[0] = 2, t.ringSend[0] = 3, t.ringRecv[0] = 4;
  t.treeToParent[0] = 5, t.treeToChild0[0] = 6, t.treeToChild1[0] = 7;
  t.ringNext[2] = 9;  // half set: left alone
  t.treeToChild0[2] = 8;
  ncclTopoRanks* ptr = &t;

  ASSERT_EQ(ncclSuccess, repairMissingChannels(&ptr, 1, 3));

  EXPECT_EQ(std::vector<int>({1, 2, 3, 4}),
            std::vector<int>({t.ringNext[1], t.ringPrev[1], t.ringSend[1], t.ringRecv[1]}));
  EXPECT_EQ(std::vector<int>({5, 6, 7}), std::vector<int>({t.treeToParent[1], t.treeToChild0[1], t.treeToChild1[1]}));
  EXPECT_EQ(-1, t.ringPrev[2]);
  EXPECT_EQ(-1, t.treeToParent[2]);
  EXPECT_TRUE(Logged("ring graph copied from  channel 0 to 1"));
}

TEST_F(ConnectMicrotest, ReconcileGrowChannels_TakesTheMinimumOverRanks) {
  Use(2, 2, 1);
  g_bootstrapAllGather = [](void*, void* data, int size) {
    int* v = static_cast<int*>(data);
    v[0] = 9, v[2] = 3, v[3] = 12;
    EXPECT_EQ(sizeof(int), (size_t)size);
    return ncclSuccess;
  };
  int value = 8;

  ASSERT_EQ(ncclSuccess, ncclTopoReconcileGrowChannels(comm_.get(), &value));
  EXPECT_EQ(3, value);

  g_bootstrapAllGather = [](void* , void* data, int) {
    static_cast<int*>(data)[0] = 1;
    return ncclSystemError;
  };
  value = 8;
  EXPECT_EQ(ncclSystemError, ncclTopoReconcileGrowChannels(comm_.get(), &value));
  EXPECT_EQ(8, value);  // left alone on failure
}

// --- ncclTopoPostset

class ConnectPostsetMicrotest : public ConnectMicrotest {
 protected:
  std::vector<int> rings_;

  void Use(int nGpus, int nNodes, int rank, const std::string& gcn = "gfx942") {
    ConnectMicrotest::Use(nGpus, nNodes, rank, gcn);
    ResetChannels();
  }
};

TEST_F(ConnectPostsetMicrotest, Postset_SingleNodeClosesTheRingsAndDoublesTheChannels) {
  Use(4, 1, 0);
  comm_->nChannels = 2;
  std::vector<ncclTopoRanks> all = AllRanks(4, 1, 2);

  ASSERT_EQ(ncclSuccess, Postset(all, 2, rings_));

  EXPECT_EQ(4, comm_->nChannels);
  EXPECT_EQ(4, comm_->collChannels);
  EXPECT_EQ(std::vector<int>({0, 1, 2, 3}), Ring(rings_, 0));
  EXPECT_EQ(std::vector<int>({0, 3, 2, 1}), Ring(rings_, 1));
  EXPECT_EQ(Ring(rings_, 1), Ring(rings_, 3));
  EXPECT_EQ(3, comm_->channels[0].ring.prev);
  EXPECT_EQ(1, comm_->channels[0].ring.next);
  EXPECT_EQ(3, comm_->channels[3].ring.next);
  EXPECT_EQ(3, comm_->channels[0].tree.depth);
}

TEST_F(ConnectPostsetMicrotest, Postset_BandwidthChannelCountRepeatsTheRings) {
  Use(4, 1, 0);
  comm_->nChannels = 2;
  std::vector<ncclTopoRanks> all = AllRanks(4, 1, 2);

  ASSERT_EQ(ncclSuccess, Postset(all, 8, rings_));

  EXPECT_EQ(16, comm_->nChannels);
  EXPECT_EQ(Ring(rings_, 1), Ring(rings_, 15));
  EXPECT_EQ(1, comm_->channels[13].ring.prev);
}

TEST_F(ConnectPostsetMicrotest, Postset_MaxCtasAndUserMaxCapTheCount) {
  Use(4, 1, 0);
  comm_->nChannels = 2;
  comm_->config.maxCTAs = 3;
  std::vector<ncclTopoRanks> all = AllRanks(4, 1, 2);
  ASSERT_EQ(ncclSuccess, Postset(all, 8, rings_));
  EXPECT_EQ(3, comm_->nChannels);

  Use(4, 1, 0);
  comm_->nChannels = 2;
  comm_->config.maxCTAs = MAXCHANNELS;
  g_params["MAX_NCHANNELS"] = 6;
  all = AllRanks(4, 1, 2);
  ASSERT_EQ(ncclSuccess, Postset(all, 8, rings_));
  EXPECT_EQ(6, comm_->nChannels);
}

TEST_F(ConnectPostsetMicrotest, Postset_UserMinIsIgnoredWhenOutOfRange) {
  Use(4, 1, 0);
  comm_->nChannels = 2;
  g_params["MIN_NCHANNELS"] = 100;
  std::vector<ncclTopoRanks> all = AllRanks(4, 1, 2);
  ASSERT_EQ(ncclSuccess, Postset(all, 1, rings_));
  EXPECT_EQ(4, comm_->nChannels);
  EXPECT_TRUE(Logged("NCCL_MIN_NCHANNELS set by environment is ignored due to less than 8 GPUs"));

  Use(4, 1, 0, "gfx90a");  // older GPUs stop at 2*CHANNEL_LIMIT
  comm_->nChannels = 2;
  g_params["MIN_NCHANNELS"] = 40;
  all = AllRanks(4, 1, 2);
  ASSERT_EQ(ncclSuccess, Postset(all, 1, rings_));
  EXPECT_EQ(4, comm_->nChannels);
  EXPECT_TRUE(Logged("ignored due to greater than max allowed 32 channels"));

  Use(4, 1, 0);
  comm_->nChannels = 2;
  g_params["MIN_NCHANNELS"] = 10;
  all = AllRanks(4, 1, 2);
  ASSERT_EQ(ncclSuccess, Postset(all, 1, rings_));
  EXPECT_EQ(10, comm_->nChannels);

  g_logs.clear();
  Use(4, 1, 0, "gfx1250");  // high counts work with fewer than 8 GPUs there
  comm_->nChannels = 2;
  g_params["MIN_NCHANNELS"] = 100;
  all = AllRanks(4, 1, 2);
  ASSERT_EQ(ncclSuccess, Postset(all, 1, rings_));
  EXPECT_EQ(100, comm_->nChannels);
  EXPECT_FALSE(Logged("ignored due to less than 8 GPUs"));
}

TEST_F(ConnectPostsetMicrotest, Postset_OverHalfTheMaxIsNotDoubled) {
  Use(4, 1, 0, "gfx90a");
  comm_->nChannels = 20;
  std::vector<ncclTopoRanks> all = AllRanks(4, 1, 20);

  ASSERT_EQ(ncclSuccess, Postset(all, 1, rings_));

  EXPECT_EQ(20, comm_->nChannels);
}

TEST_F(ConnectPostsetMicrotest, Postset_MultiNodeJoinsTheNodesAndCapsAt64) {
  Use(2, 2, 0);
  comm_->nChannels = 2;
  std::vector<ncclTopoRanks> all = AllRanks(2, 2, 2);

  ASSERT_EQ(ncclSuccess, Postset(all, 100, rings_));

  EXPECT_EQ(std::vector<int>({0, 1, 2, 3}), Ring(rings_, 0));
  EXPECT_EQ(std::vector<int>({0, 3, 2, 1}), Ring(rings_, 1));  // reversed inside each node
  EXPECT_EQ(64, comm_->nChannels);
  EXPECT_EQ(-1, comm_->channels[0].tree.up);  // node 0 roots the first tree
  EXPECT_EQ(2, comm_->channels[0].tree.down[0]);
  EXPECT_EQ(2, comm_->channels[2].tree.up);  // and hangs off node 1 in the second
}

TEST_F(ConnectPostsetMicrotest, Postset_CrossNicRingsSwapChannelPairsOnOddNodes) {
  Use(2, 2, 0);
  comm_->nChannels = 2;
  std::vector<ncclTopoRanks> all = AllRanks(2, 2, 2);
  for (ncclTopoRanks& t : all) t.crossNicRing = 2;

  ASSERT_EQ(ncclSuccess, Postset(all, 2, rings_));

  EXPECT_EQ(std::vector<int>({0, 1, 3, 2}), Ring(rings_, 0));  // node 1 runs its channel-1 order
  EXPECT_EQ(std::vector<int>({0, 2, 3, 1}), Ring(rings_, 1));
}

TEST_F(ConnectPostsetMicrotest, Postset_Gfx1151AndInterGraphGenUseLoadBalancedRings) {
  for (bool interGen : {false, true}) {
    Use(1, 3, 0, interGen ? "gfx942" : "gfx1151");
    comm_->nChannels = 2;
    if (interGen) g_params["RCCL_INTER_GRAPH_GEN"] = 1;
    g_generateRings = [](int n, uint32_t nChannels, int* order) {
      for (uint32_t c = 0; c < nChannels; c++)
        for (int i = 0; i < n; i++) order[c * n + i] = c % 2 ? n - 1 - i : i;
      return ncclSuccess;
    };
    std::vector<ncclTopoRanks> all = AllRanks(1, 3, 2);

    ASSERT_EQ(ncclSuccess, Postset(all, 2, rings_));

    EXPECT_EQ(std::vector<int>({0, 2, 1}), Ring(rings_, 1)) << interGen;
    g_params.clear();
  }
}

TEST_F(ConnectPostsetMicrotest, Postset_RailOptimizedTreesReplaceTheBinaryTrees) {
  Use(2, 2, 2);
  comm_->nChannels = 4;
  system_->useRailOptimizedTrees = true;
  std::vector<ncclTopoRanks> all = AllRanks(2, 2, 4);

  ASSERT_EQ(ncclSuccess, Postset(all, 4, rings_));

  EXPECT_TRUE(Logged("Building rail-optimized trees for 2 nodes"));
}

TEST_F(ConnectPostsetMicrotest, Postset_OutputTreesPrintsGraphvizEdges) {
  Use(2, 2, 0);
  comm_->nChannels = 2;
  g_params["RCCL_OUTPUT_TREES"] = 1;
  std::vector<ncclTopoRanks> all = AllRanks(2, 2, 2);
  for (int c = 0; c < 4; c++) graphs_[NCCL_ALGO_TREE]->inter[2 * c + 1] = 7;
  comm_->channels[0].tree.down[0] = 1;  // the intra-node chain from ncclTopoPreset

  ASSERT_EQ(ncclSuccess, Postset(all, 2, rings_));

  EXPECT_TRUE(Logged("[TREE] 0.0 [style=filled, fillcolor=red]"));
  EXPECT_TRUE(Logged("[TREE] 0.0->0.1 [style=solid,width=10,color=black,label=\"\"]"));
  EXPECT_TRUE(Logged("[TREE] 0.0->0.2 [style=dashed,width=10,color=red,label=\"N7\"]"));
}

TEST_F(ConnectPostsetMicrotest, Postset_GrowCommAgreesOnTheSmallestCount) {
  Use(2, 1, 0);
  comm_->nChannels = 2;
  comm_->isGrow = true;
  g_bootstrapAllGather = [](void*, void* data, int) {
    static_cast<int*>(data)[1] = 3;
    return ncclSuccess;
  };
  std::vector<ncclTopoRanks> all = AllRanks(2, 1, 2);

  ASSERT_EQ(ncclSuccess, Postset(all, 2, rings_));

  EXPECT_EQ(3, comm_->nChannels);
  EXPECT_EQ(3, comm_->collChannels);

  Use(2, 1, 0);
  comm_->nChannels = 2;
  g_bootstrapAllGather = [](void*, void*, int) { return ncclSystemError; };
  all = AllRanks(2, 1, 2);
  EXPECT_EQ(ncclSystemError, Postset(all, 2, rings_));
}

TEST_F(ConnectPostsetMicrotest, Postset_CollNetAddsHalfAgainAndConnectsTheHeads) {
  Use(2, 2, 0);
  comm_->nChannels = 2;
  comm_->config.collnetEnable = 1;
  graphs_[NCCL_ALGO_COLLNET_CHAIN]->bwIntra = 20;
  graphs_[NCCL_ALGO_COLLNET_CHAIN]->bwInter = 10;
  Intra(NCCL_ALGO_COLLNET_DIRECT, {{0, 1}});
  std::vector<ncclTopoRanks> all = AllRanks(2, 2, 2);

  ASSERT_EQ(ncclSuccess, Postset(all, 1, rings_));

  EXPECT_EQ(6, comm_->nChannels);  // 4, plus half
  EXPECT_EQ(2, comm_->channels[5].collnetChain.depth);
  EXPECT_EQ(0, comm_->channels[0].collnetDirect.headRank);

  Use(2, 2, 0);
  comm_->nChannels = 2;
  comm_->maxLocalRanks = NCCL_MAX_DIRECT_ARITY + 2;
  graphs_[NCCL_ALGO_COLLNET_CHAIN]->bwIntra = 5;
  comm_->channels[0].collnetDirect.headRank = -7;
  all = AllRanks(2, 2, 2);
  ASSERT_EQ(ncclSuccess, Postset(all, 1, rings_));
  EXPECT_EQ(4, comm_->nChannels);
  EXPECT_EQ(-7, comm_->channels[0].collnetDirect.headRank);  // too many local ranks for CollNet Direct
}

TEST_F(ConnectPostsetMicrotest, Postset_FastIntraNodeAndUnpackNetsDoubleTheChannels) {
  Use(2, 2, 0);
  comm_->nChannels = 2;
  comm_->minCompCap = 90;
  graphs_[NCCL_ALGO_RING]->bwIntra = 50;
  std::vector<ncclTopoRanks> all = AllRanks(2, 2, 2);
  ASSERT_EQ(ncclSuccess, Postset(all, 1, rings_));
  EXPECT_EQ(8, comm_->nChannels);

  Use(2, 2, 0);
  comm_->nChannels = 2;
  comm_->minCompCap = 0;
  comm_->netDeviceType = NCCL_NET_DEVICE_UNPACK;
  all = AllRanks(2, 2, 2);
  ASSERT_EQ(ncclSuccess, Postset(all, 1, rings_));
  EXPECT_EQ(8, comm_->nChannels);
  Use(2, 2, 0);
  comm_->nChannels = 2;
  g_params["UNPACK_DOUBLE_NCHANNELS"] = 0;
  all = AllRanks(2, 2, 2);
  ASSERT_EQ(ncclSuccess, Postset(all, 1, rings_));
  EXPECT_EQ(4, comm_->nChannels);
}

TEST_F(ConnectPostsetMicrotest, Postset_ChildCommStaysWithinItsParentsChannels) {
  Use(2, 1, 0);
  comm_->nChannels = 2;
  ncclComm parentOwner;
  shared_->owner = &parentOwner;
  shared_->tpNChannels = 3;
  std::vector<ncclTopoRanks> all = AllRanks(2, 1, 2);

  ASSERT_EQ(ncclSuccess, Postset(all, 8, rings_));

  EXPECT_EQ(3, comm_->nChannels);
}

TEST_F(ConnectPostsetMicrotest, Postset_SharedNvlsParentCapsTheChannels) {
  Use(2, 1, 0);
  comm_->nChannels = 2;
  auto parent = std::make_unique<ncclComm>();
  ncclSharedResources parentShared = {};
  parentShared.tpNChannels = 15;
  parent->sharedRes = &parentShared;
  parent->nvlsSupport = 1;
  parent->shareResources = true;
  std::vector<ncclTopoRanks> all = AllRanks(2, 1, 2);

  ASSERT_EQ(ncclSuccess, Postset(all, 8, rings_, parent.get()));

  EXPECT_EQ(15, comm_->nChannels);  // 16 before the cap
  EXPECT_EQ(15, comm_->collChannels);
}

TEST_F(ConnectPostsetMicrotest, Postset_Gfx950MultiNodeHonoursTheUserMax) {
  Use(2, 2, 0, "gfx950");
  comm_->nChannels = 2;
  g_params["MAX_NCHANNELS"] = 100;
  std::vector<ncclTopoRanks> all = AllRanks(2, 2, 2);

  ASSERT_EQ(ncclSuccess, Postset(all, 100, rings_));

  EXPECT_EQ(100, comm_->nChannels);  // past the usual 64 multi-node cap
  EXPECT_TRUE(Logged("RCCL MaxChannels is capped to: 100"));
}

TEST_F(ConnectPostsetMicrotest, Postset_The64CapSkipsGfx1250SingleNodeAndMnnvl) {
  Use(2, 1, 0, "gfx1250");
  comm_->nChannels = 2;
  graphs_[NCCL_ALGO_RING]->nIntraChannels = 1;
  std::vector<ncclTopoRanks> all = AllRanks(2, 1, 2);
  ASSERT_EQ(ncclSuccess, Postset(all, 100, rings_));
  EXPECT_EQ(200, comm_->nChannels);

  Use(2, 1, 0);
  comm_->nChannels = 2;
  all = AllRanks(2, 1, 2);
  ASSERT_EQ(ncclSuccess, Postset(all, 100, rings_));
  EXPECT_EQ(64, comm_->nChannels);  // intra-net rings keep the cap elsewhere

  Use(2, 2, 0);
  comm_->nChannels = 2;
  comm_->MNNVL = 1;
  all = AllRanks(2, 2, 2);
  ASSERT_EQ(ncclSuccess, Postset(all, 100, rings_));
  EXPECT_EQ(200, comm_->nChannels);
}

TEST_F(ConnectPostsetMicrotest, Postset_BrokenRingsAreReported) {
  Use(3, 1, 0);
  comm_->nChannels = 1;
  std::vector<ncclTopoRanks> all = AllRanks(3, 1, 1);
  all[1].ringNext[0] = 5;  // past the ranks

  EXPECT_EQ(ncclInternalError, Postset(all, 1, rings_));
}

}  // namespace
