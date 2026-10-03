/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for the REAL src/graph/connect.cc: ncclTopoPreset and ncclTopoPostset run unstubbed, with
// the real ring/tree helpers (rings.cc, trees.cc, rccl_graph_gen.cc) and IsArchMatch linked in.

#include <gtest/gtest.h>

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

#include CONNECT_CC_PATH

// ---------------------------------------------------------------------------
// Fakes for the symbols connect.cc takes from the rest of RCCL
// ---------------------------------------------------------------------------

int ncclDebugLevel = 0;
uint64_t ncclDebugMask = 0;
thread_local int ncclDebugNoWarn = 0;

void ncclDebugLog(ncclDebugLogLevel level, unsigned long, const char* filefunc, int line, const char* fmt, ...) {
  if (level != NCCL_LOG_WARN) return;
  std::fprintf(stderr, "[fake WARN] %s:%d ", filefunc ? filefunc : "?", line);
  va_list ap;
  va_start(ap, fmt);
  std::vfprintf(stderr, fmt, ap);
  va_end(ap);
  std::fputc('\n', stderr);
}

// Every param keeps its default; NCCL_PARAM/RCCL_PARAM read the result back from the cache.
int64_t ncclLoadParam(char const*, int64_t deftVal, int64_t, int64_t* cache, int8_t*) {
  *cache = deftVal;
  return deftVal;
}

static int64_t g_p2pDisable = 0;
int64_t ncclParamP2pDisable() { return g_p2pDisable; }
int64_t ncclParamWorkArgsBytes() { return INT64_MAX; }

ncclResult_t bootstrapAllGather(void*, void*, int) { return ncclInternalError; }

namespace {

constexpr int kRanks = 4;

// One single-node communicator plus the graphs Preset() and Postset() read.
class ConnectScene {
 public:
  ConnectScene(const char* arch, int ringChannels, int treeChannels)
      : comm_(new ncclComm{}), topo_(new ncclTopoSystem{}), shared_(new ncclSharedResources{}) {
    comm_->nRanks = kRanks;
    comm_->nNodes = 1;
    comm_->node = 0;
    comm_->localRanks = kRanks;
    comm_->rankToNode = rankToNode_;
    comm_->rankToLocalRank = rankToLocalRank_;
    comm_->topo = topo_.get();
    comm_->sharedRes = shared_.get();
    shared_->owner = comm_.get();
    comm_->config.minCTAs = 1;
    comm_->config.maxCTAs = MAXCHANNELS;
    topo_->nodes[GPU].count = kRanks;
    std::snprintf(topo_->nodes[GPU].nodes[0].gpu.gcn, sizeof(topo_->nodes[GPU].nodes[0].gpu.gcn), "%s", arch);

    for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++) {
      graphStore_.emplace_back(new ncclTopoGraph{});
      graphs_[a] = graphStore_.back().get();
      graphs_[a]->id = a;
      std::fill_n(graphs_[a]->intra, MAXCHANNELS * NCCL_TOPO_MAX_NODES, -1);
    }
    fillChain(graphs_[NCCL_ALGO_RING], ringChannels);
    fillChain(graphs_[NCCL_ALGO_TREE], treeChannels);
    graphs_[NCCL_ALGO_RING]->pattern = NCCL_TOPO_PATTERN_RING;
    graphs_[NCCL_ALGO_TREE]->pattern = NCCL_TOPO_PATTERN_BALANCED_TREE;
  }

  ncclComm* comm() { return comm_.get(); }
  ncclTopoRanks& topoRanks(int rank) { return topoRanks_[rank]; }

  // Preset() on every rank with comm->nChannels = presetChannels, ending with `rank` so the channel poisoning it
  // does matches the rank Postset() then runs as.
  ncclResult_t presetAllRanks(int rank, int presetChannels) {
    for (int i = 1; i <= kRanks; i++) {
      const int r = (rank + i) % kRanks;
      comm_->rank = r;
      comm_->nChannels = presetChannels;
      ncclResult_t res = ncclTopoPreset(comm_.get(), graphs_, &topoRanks_[r]);
      if (res != ncclSuccess) return res;
    }
    return ncclSuccess;
  }

  // What init.cc does between AllGather3 and Postset(): settle the final count, then call Postset().
  ncclResult_t postset(int nChannels) {
    comm_->nChannels = nChannels;
    struct ncclTopoRanks* all[kRanks];
    for (int r = 0; r < kRanks; r++) all[r] = &topoRanks_[r];
    int firstRanks[1] = {0};
    int treePatterns[1] = {graphs_[NCCL_ALGO_TREE]->pattern};
    std::vector<int> rings(kRanks * MAXCHANNELS);
    return ncclTopoPostset(comm_.get(), firstRanks, treePatterns, all, rings.data(), graphs_, nullptr, /*nc=*/1);
  }

 private:
  static void fillChain(ncclTopoGraph* g, int nChannels) {
    g->nChannels = nChannels;
    for (int c = 0; c < nChannels; c++) {
      for (int i = 0; i < kRanks; i++) g->intra[c * kRanks + i] = i;
    }
  }

  std::unique_ptr<ncclComm> comm_;
  std::unique_ptr<ncclTopoSystem> topo_;
  std::unique_ptr<ncclSharedResources> shared_;
  std::vector<std::unique_ptr<ncclTopoGraph>> graphStore_;
  ncclTopoGraph* graphs_[NCCL_NUM_ALGORITHMS];
  ncclTopoRanks topoRanks_[kRanks];
  int rankToNode_[kRanks] = {0, 0, 0, 0};
  int rankToLocalRank_[kRanks] = {0, 1, 2, 3};
};

class ConnectMicrotest : public ::testing::TestWithParam<int> {
 protected:
  void SetUp() override { g_p2pDisable = 0; }
  void TearDown() override { g_p2pDisable = 0; }
};

// The Navi SHM shape: graph generation forces many rings while the tree search falls back to one channel, so
// Preset() fills a single channel and the count grows after AllGather3. Postset() then doubles 6 to 12 on gfx1201,
// so the duplicate half is live as well. Every live channel must carry this rank's intra-node tree link.
TEST_P(ConnectMicrotest, PresetPostset_ChannelCountGrowsPastPreset_EveryChannelHasTheIntraNodeTree) {
  const int rank = GetParam();
  const int kRingChannels = 6;
  const int kTreeChannels = 1;
  g_p2pDisable = 1;
  ConnectScene scene("gfx1201", kRingChannels, kTreeChannels);
  ASSERT_EQ(ncclSuccess, scene.presetAllRanks(rank, std::min(kRingChannels, kTreeChannels)));
  ASSERT_EQ(ncclSuccess, scene.postset(kRingChannels));

  ncclComm* comm = scene.comm();
  ASSERT_EQ(2 * kRingChannels, comm->nChannels);
  const int expectedUp = rank == 0 ? -1 : rank - 1;
  const int expectedDown = rank == kRanks - 1 ? -1 : rank + 1;
  for (int c = 0; c < comm->nChannels; c++) {
    EXPECT_EQ(expectedUp, comm->channels[c].tree.up) << "rank " << rank << " channel " << c;
    EXPECT_EQ(expectedDown, comm->channels[c].tree.down[0]) << "rank " << rank << " channel " << c;
    EXPECT_EQ((rank + kRanks - 1) % kRanks, comm->channels[c].ring.prev) << "rank " << rank << " channel " << c;
    EXPECT_EQ((rank + 1) % kRanks, comm->channels[c].ring.next) << "rank " << rank << " channel " << c;
  }
}

// No growth: the count Preset() saw is the final one, and the duplicate half mirrors the first.
TEST_P(ConnectMicrotest, PresetPostset_ChannelCountUnchanged_DuplicatesMirrorTheFirstHalf) {
  const int rank = GetParam();
  const int kChannels = 2;
  ConnectScene scene("gfx1201", kChannels, kChannels);
  ASSERT_EQ(ncclSuccess, scene.presetAllRanks(rank, kChannels));
  ASSERT_EQ(ncclSuccess, scene.postset(kChannels));

  ncclComm* comm = scene.comm();
  ASSERT_EQ(2 * kChannels, comm->nChannels);
  for (int c = 0; c < comm->nChannels; c++) {
    EXPECT_EQ(rank == 0 ? -1 : rank - 1, comm->channels[c].tree.up) << "rank " << rank << " channel " << c;
    EXPECT_EQ(rank == kRanks - 1 ? -1 : rank + 1, comm->channels[c].tree.down[0])
      << "rank " << rank << " channel " << c;
  }
}

// A peer reporting fewer channels shrinks the count below what Preset() filled; the surviving channels and their
// duplicates still carry the tree.
TEST_P(ConnectMicrotest, PresetPostset_ChannelCountShrinksBelowPreset_EveryChannelHasTheIntraNodeTree) {
  const int rank = GetParam();
  const int kPresetChannels = 4;
  const int kFinalChannels = 2;
  ConnectScene scene("gfx1201", kPresetChannels, kPresetChannels);
  ASSERT_EQ(ncclSuccess, scene.presetAllRanks(rank, kPresetChannels));
  ASSERT_EQ(ncclSuccess, scene.postset(kFinalChannels));

  ncclComm* comm = scene.comm();
  ASSERT_EQ(2 * kFinalChannels, comm->nChannels);
  for (int c = 0; c < comm->nChannels; c++) {
    EXPECT_EQ(rank == 0 ? -1 : rank - 1, comm->channels[c].tree.up) << "rank " << rank << " channel " << c;
    EXPECT_EQ(rank == kRanks - 1 ? -1 : rank + 1, comm->channels[c].tree.down[0])
      << "rank " << rank << " channel " << c;
  }
}

INSTANTIATE_TEST_SUITE_P(EveryRank, ConnectMicrotest, ::testing::Range(0, kRanks));

// Postset() refuses a peer whose topoRanks Preset() never filled instead of building channels from poison.
TEST(ConnectMicrotestReject, PresetPostset_PeerWithoutPresetChannels_IsRejected) {
  ConnectScene scene("gfx1201", 2, 2);
  ASSERT_EQ(ncclSuccess, scene.presetAllRanks(0, 2));
  scene.topoRanks(2).nChannels = 0;
  EXPECT_EQ(ncclInternalError, scene.postset(2));
}

}  // namespace
