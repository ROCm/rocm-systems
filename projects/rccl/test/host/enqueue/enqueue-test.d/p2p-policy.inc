/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// P2p channel policy: rcclP2pTaskParts, rcclP2pPolicyChannels, calcP2pChannelCount,
// rcclEffectiveP2pBatchEnable and rcclP2pBatchEligible.
// Included into enqueue-test.cc's translation unit; see README.md.

// ===========================================================================
// rcclP2pTaskParts (enqueue.cc:1333) and rcclP2pPolicyChannels (enqueue.cc:1342)
//
// Multi-node AINIC Scatter takes the most power-of-two parts whose nRanks - 1
// flows fit p2pnChannels, at most p2pnChannelsPerPeer; everything else takes
// p2pnChannelsPerPeer. Both ends of a pair must get the same answer, so it may
// depend only on comm-wide fields.
// ===========================================================================

namespace {
std::unique_ptr<ncclComm> P2pPartsComm(int nNodes, int nRanks, int p2pnChannels, int perPeer) {
  std::unique_ptr<ncclComm> comm{new ncclComm{}};
  comm->nNodes = nNodes;
  comm->nRanks = nRanks;
  comm->p2pnChannels = p2pnChannels;
  comm->p2pnChannelsPerPeer = perPeer;
  comm->config.nChannelsPerNetPeer = NCCL_CONFIG_UNDEF_INT;
  return comm;
}
}  // namespace

TEST_F(EnqueueMicrotest, P2pTaskParts_AinicScatterTakesThePartsWhoseFlowsFitThePool) {
  g_rcclUseAinicValue = true;
  struct Case {
    int nNodes, nRanks, p2pnChannels, perPeer, scatterParts;
  };
  // 8 GPUs per node with the MI355X pools, the pool boundary (16 peers x 2 parts fill 32 channels,
  // 17 peers do not), power-of-two rounding (64 / 19 = 3) and the one-part floor (64 / 127 = 0).
  for (Case k : {Case{2, 16, 32, 2, 2}, Case{3, 24, 64, 4, 2}, Case{4, 32, 32, 2, 1}, Case{8, 64, 64, 2, 1},
                 Case{2, 17, 32, 2, 2}, Case{2, 18, 32, 2, 1}, Case{3, 20, 64, 8, 2}, Case{16, 128, 64, 2, 1}}) {
    auto comm = P2pPartsComm(k.nNodes, k.nRanks, k.p2pnChannels, k.perPeer);
    EXPECT_EQ(k.scatterParts, rcclP2pTaskParts(comm.get(), ncclFuncScatter, /*peer=*/0))
      << k.nRanks << " ranks, " << k.p2pnChannels << " channels";
    for (ncclFunc_t f : {ncclFuncSendRecv, ncclFuncSend, ncclFuncRecv, ncclFuncAlltoAll, ncclFuncGather,
                         ncclFuncAlltoAllv}) {
      EXPECT_EQ(k.perPeer, rcclP2pTaskParts(comm.get(), f, /*peer=*/0)) << "collAPI=" << int(f);
    }
  }
}

TEST_F(EnqueueMicrotest, P2pTaskParts_ScatterKeepsPerPeerOutsideMultiNodeAinicOrWithAUserCount) {
  auto comm = P2pPartsComm(/*nNodes=*/4, /*nRanks=*/32, /*p2pnChannels=*/32, /*perPeer=*/2);
  g_rcclUseAinicValue = false;
  EXPECT_EQ(2, rcclP2pTaskParts(comm.get(), ncclFuncScatter, /*peer=*/0)) << "not AINIC";
  g_rcclUseAinicValue = true;
  comm->config.nChannelsPerNetPeer = 2;
  EXPECT_EQ(2, rcclP2pTaskParts(comm.get(), ncclFuncScatter, /*peer=*/0))
    << "NCCL_NCHANNELS_PER_NET_PEER set";
  comm->config.nChannelsPerNetPeer = NCCL_CONFIG_UNDEF_INT;
  comm->nNodes = 1;
  EXPECT_EQ(2, rcclP2pTaskParts(comm.get(), ncclFuncScatter, /*peer=*/0)) << "single node";
}

TEST_F(EnqueueMicrotest, P2pPolicyChannels_ScatterBelowPerPeerReturnsItsParts) {
  // Returns before the arch policy table, so comm->topo is never read.
  g_rcclUseAinicValue = true;
  auto comm = P2pPartsComm(/*nNodes=*/4, /*nRanks=*/32, /*p2pnChannels=*/32, /*perPeer=*/2);
  ncclTaskP2p task{};
  task.collAPI = ncclFuncScatter;
  EXPECT_EQ(1, rcclP2pPolicyChannels(comm.get(), &task));
  EXPECT_EQ(-1, rcclP2pPolicyChannels(comm.get(), nullptr));
}

TEST_F(EnqueueMicrotest, P2pReservationChannels_P2pNetPreservesAinicScatterLimit) {
  g_rcclUseAinicValue = true;
  auto comm = P2pPartsComm(/*nNodes=*/4, /*nRanks=*/32, /*p2pnChannels=*/32, /*perPeer=*/2);
  comm->p2pNet = 1;
  ncclTaskP2p task{};
  task.collAPI = ncclFuncScatter;
  task.root = 1;
  task.bytes = 64 << 20;

  EXPECT_EQ(1, rcclP2pReservationChannels(comm.get(), /*recv=*/nullptr, &task));
}

// gfx942 remote alltoall cap (enqueue.cc). Returns 2 when the stored
// p2pnChannelsPerPeer is still above 2. paths.cc has already fitted that
// count to p2pMaxPeers, so this function does not fit again. The peer in
// this table is on the other node. A same-node peer, every other collective,
// arch, node count, a user-set nChannelsPerNetPeer, and RCCL_SATURATE_P2P_NCHANNELS
// fall through. gfx110 alltoall still reaches its own table entry.
TEST_F(EnqueueMicrotest, P2pPolicyChannels_Gfx942MultiNodeAlltoallCapsAboveTwoFittedChannels) {
  struct Case {
    const char* name;
    int nNodes, nRanks, channels, perPeer;
    const char* gcn;
    ncclFunc_t coll;
    bool userSet;
    int user;
    int expect;
  };
  const Case cases[] = {
    {"2x8 alltoall", 2, 16, 64, 4, "gfx942", ncclFuncAlltoAll, false, 0, 2},
    // A larger job whose stored count paths.cc left above 2 is capped too.
    // The old nRanks loop returned -1 here (4*24 > 64 halved to 2).
    {"3-node stored count still 4", 3, 24, 64, 4, "gfx942", ncclFuncAlltoAll, false, 0, 2},
    {"non-pow2 per peer", 2, 16, 64, 6, "gfx942", ncclFuncAlltoAll, false, 0, 2},
    // IsArchMatch is a prefix compare, so a stepping suffix still matches.
    {"gfx942 stepping", 2, 16, 64, 4, "gfx942:xnack-", ncclFuncAlltoAll, false, 0, 2},
    {"stored count 2", 4, 32, 64, 2, "gfx942", ncclFuncAlltoAll, false, 0, -1},
    {"stored count 1", 8, 64, 64, 1, "gfx942", ncclFuncAlltoAll, false, 0, -1},
    {"single node", 1, 8, 64, 4, "gfx942", ncclFuncAlltoAll, false, 0, -1},
    {"gfx950", 2, 16, 64, 4, "gfx950", ncclFuncAlltoAll, false, 0, -1},
    // Misses gfx942, then the gfx110 table returns 1. Single node must too:
    // the new gate is not allowed to swallow that entry.
    {"gfx110 multi-node", 2, 16, 64, 4, "gfx1100", ncclFuncAlltoAll, false, 0, 1},
    {"gfx110 single-node", 1, 8, 64, 4, "gfx1100", ncclFuncAlltoAll, false, 0, 1},
    {"send", 2, 16, 64, 4, "gfx942", ncclFuncSend, false, 0, -1},
    {"recv", 2, 16, 64, 4, "gfx942", ncclFuncRecv, false, 0, -1},
    {"sendrecv", 2, 16, 64, 4, "gfx942", ncclFuncSendRecv, false, 0, -1},
    {"gather", 2, 16, 64, 4, "gfx942", ncclFuncGather, false, 0, -1},
    {"alltoallv", 2, 16, 64, 4, "gfx942", ncclFuncAlltoAllv, false, 0, -1},
    // Either explicit count skips the cap, including one equal to the cap.
    {"user count 4", 2, 16, 64, 4, "gfx942", ncclFuncAlltoAll, true, 4, -1},
    {"user count 2", 2, 16, 64, 4, "gfx942", ncclFuncAlltoAll, true, 2, -1},
  };

  for (const Case& k : cases) {
    auto topo = std::unique_ptr<ncclTopoSystem>(new ncclTopoSystem{});
    auto comm = P2pPartsComm(k.nNodes, k.nRanks, k.channels, k.perPeer);
    comm->topo = topo.get();
    topo->nodes[GPU].count = 1;
    std::snprintf(topo->nodes[GPU].nodes[0].gpu.gcn,
                  sizeof(topo->nodes[GPU].nodes[0].gpu.gcn), "%s", k.gcn);
    if (k.userSet) comm->config.nChannelsPerNetPeer = k.user;
    int rankNode[2] = {0, 1};
    comm->node = 0;
    comm->rankToNode = rankNode;
    ncclTaskP2p task{};
    task.collAPI = k.coll;
    task.root = 1;
    EXPECT_EQ(k.expect, rcclP2pPolicyChannels(comm.get(), &task)) << k.name;
  }

  // task->root is the peer. Both ends see the other node, so both cap; a peer
  // on this node does not.
  auto topo = std::unique_ptr<ncclTopoSystem>(new ncclTopoSystem{});
  auto comm = P2pPartsComm(/*nNodes=*/2, /*nRanks=*/16, /*p2pnChannels=*/64, /*perPeer=*/4);
  comm->topo = topo.get();
  topo->nodes[GPU].count = 1;
  std::snprintf(topo->nodes[GPU].nodes[0].gpu.gcn, sizeof(topo->nodes[GPU].nodes[0].gpu.gcn), "gfx942");
  int rankNode[2] = {0, 1};
  comm->rankToNode = rankNode;
  ncclTaskP2p task{};
  task.collAPI = ncclFuncAlltoAll;
  comm->node = 0;
  task.root = 1;
  EXPECT_EQ(2, rcclP2pPolicyChannels(comm.get(), &task)) << "node 0 to node 1";
  comm->node = 1;
  task.root = 0;
  EXPECT_EQ(2, rcclP2pPolicyChannels(comm.get(), &task)) << "node 1 to node 0";
  comm->node = 0;
  task.root = 0;
  EXPECT_EQ(-1, rcclP2pPolicyChannels(comm.get(), &task)) << "same-node peer";

  // maxP2pPeers below nRanks must not grow a second fit. The stored count is the input.
  comm->p2pMaxPeers = 2;
  comm->nRanks = 16;
  comm->node = 0;
  task.root = 1;
  EXPECT_EQ(2, rcclP2pPolicyChannels(comm.get(), &task)) << "p2pMaxPeers below nRanks";
}

TEST_F(EnqueueMicrotest, P2pPolicyChannels_P2pDisableStillCapsRemoteGfx942Alltoall) {
  g_loadParam = [](const char* env, int64_t deft) {
    return std::strcmp(env, "P2P_DISABLE") == 0 ? int64_t(1) : deft;
  };
  auto topo = std::unique_ptr<ncclTopoSystem>(new ncclTopoSystem{});
  auto comm = P2pPartsComm(/*nNodes=*/2, /*nRanks=*/16, /*p2pnChannels=*/64, /*perPeer=*/4);
  comm->topo = topo.get();
  topo->nodes[GPU].count = 1;
  std::snprintf(topo->nodes[GPU].nodes[0].gpu.gcn, sizeof(topo->nodes[GPU].nodes[0].gpu.gcn), "gfx942");
  int rankNode[2] = {0, 1};
  comm->node = 0;
  comm->rankToNode = rankNode;
  ncclTaskP2p task{};
  task.collAPI = ncclFuncAlltoAll;
  task.root = 1;
  EXPECT_EQ(2, rcclP2pPolicyChannels(comm.get(), &task)) << "remote peer, P2P disabled";
  task.root = 0;
  EXPECT_EQ(-1, rcclP2pPolicyChannels(comm.get(), &task)) << "same-node peer, P2P disabled";
  std::snprintf(topo->nodes[GPU].nodes[0].gpu.gcn, sizeof(topo->nodes[GPU].nodes[0].gpu.gcn), "gfx1100");
  task.root = 1;
  EXPECT_EQ(-1, rcclP2pPolicyChannels(comm.get(), &task)) << "gfx110 stays below the disable return";
}

TEST_F(EnqueueMicrotest, P2pPolicyChannels_SaturateP2pSkipsTheGfx942Cap) {
  auto topo = std::unique_ptr<ncclTopoSystem>(new ncclTopoSystem{});
  auto comm = P2pPartsComm(/*nNodes=*/2, /*nRanks=*/16, /*p2pnChannels=*/64, /*perPeer=*/4);
  comm->topo = topo.get();
  topo->nodes[GPU].count = 1;
  std::snprintf(topo->nodes[GPU].nodes[0].gpu.gcn, sizeof(topo->nodes[GPU].nodes[0].gpu.gcn), "gfx942");
  int rankNode[2] = {0, 1};
  comm->node = 0;
  comm->rankToNode = rankNode;
  ncclTaskP2p task{};
  task.collAPI = ncclFuncAlltoAll;
  task.root = 1;

  // paths.cc turns saturation on for every value other than 0 and UNSET (-2),
  // including -1. A <= 0 gate caps -1 and still passes a test that only sets 1 and 0.
  for (int64_t on : {int64_t(1), int64_t(-1)}) {
    g_loadParam = [on](const char* env, int64_t deft) {
      return std::strcmp(env, "RCCL_SATURATE_P2P_NCHANNELS") == 0 ? on : deft;
    };
    EXPECT_EQ(-1, rcclP2pPolicyChannels(comm.get(), &task)) << "saturate " << on;
  }
  for (int64_t off : {int64_t(0), int64_t(RCCL_VALUE_UNSET)}) {
    g_loadParam = [off](const char* env, int64_t deft) {
      return std::strcmp(env, "RCCL_SATURATE_P2P_NCHANNELS") == 0 ? off : deft;
    };
    EXPECT_EQ(2, rcclP2pPolicyChannels(comm.get(), &task)) << "saturate " << off;
  }
}

// ===========================================================================
// calcP2pChannelCount (enqueue.cc:1550)
//
// DEAD CODE: repo-wide grep finds exactly one occurrence -- its own definition.
// Nothing calls it. Tested anyway to pin current behaviour, because the
// alternative is deleting it, and that is a maintainer's call not a test's.
// Reported in the PR body rather than silently covered.
// ===========================================================================

TEST_F(EnqueueMicrotest, CalcP2pChannelCount_FitsAtMinChannels_NoDoubling) {
  // size <= maxSize immediately, so the loop never runs.
  EXPECT_EQ(1, calcP2pChannelCount(/*total=*/1024, /*min=*/1, /*max=*/8,
                                   /*minSize=*/0, /*maxSize=*/4096));
}

TEST_F(EnqueueMicrotest, CalcP2pChannelCount_DoublesUntilUnderMaxSize) {
  // total/1 = 8192 > 2048 -> 2 (4096) -> 4 (2048) stops: 2048 > 2048 is false.
  EXPECT_EQ(4, calcP2pChannelCount(/*total=*/8192, /*min=*/1, /*max=*/8,
                                   /*minSize=*/0, /*maxSize=*/2048));
}

TEST_F(EnqueueMicrotest, CalcP2pChannelCount_StopsAtHalfMaxChannels) {
  // The guard is `nChannels <= maxChannels/2`, so with max=8 the last legal
  // doubling is 4->8. Even a still-too-large size cannot push past 8.
  EXPECT_EQ(8, calcP2pChannelCount(/*total=*/1 << 20, /*min=*/1, /*max=*/8,
                                   /*minSize=*/0, /*maxSize=*/1));
}

TEST_F(EnqueueMicrotest, CalcP2pChannelCount_MinSizeFloorsInitialSizeOnly) {
  // LATENT INCONSISTENCY (enqueue.cc:1551 vs :1555): the initial size is
  // max(minSize, divUp(total,min)) but the in-loop recompute drops the minSize
  // floor. Here minSize alone forces entry to the loop even though the true
  // per-channel size is already tiny; the first recompute then falls below
  // maxSize and stops. Pins that asymmetry.
  EXPECT_EQ(2, calcP2pChannelCount(/*total=*/16, /*min=*/1, /*max=*/8,
                                   /*minSize=*/4096, /*maxSize=*/2048));
}

// ===========================================================================
// rcclEffectiveP2pBatchEnable (enqueue.cc:1169)
// Four arms gating P2P batching for the whole file. The user override is
// checked FIRST, so it must win over both the node count and the arch.
// ===========================================================================

namespace {

struct BatchComm {
  std::unique_ptr<ncclTopoSystem> topo{new ncclTopoSystem{}};
  std::unique_ptr<ncclComm> comm{new ncclComm{}};
  BatchComm(int nNodes, const char* gcn) {
    comm->nNodes = nNodes;
    comm->topo = topo.get();
    SetSingleGpuArch(topo.get(), gcn);
  }
  ncclComm* get() { return comm.get(); }
};

void SetBatchParam(int64_t v) { SetParam("RCCL_P2P_BATCH_ENABLE", v); }

// Forces batching on; with thresholdSet false the threshold lookup falls back
// to its parameter default.
void SetBatchEnableAndThreshold(int64_t threshold, bool thresholdSet = true) {
  g_loadParam = [threshold, thresholdSet](const char* env, int64_t deft) {
    if (std::strcmp(env, "RCCL_P2P_BATCH_ENABLE") == 0) return int64_t(1);
    if (thresholdSet && std::strcmp(env, "RCCL_P2P_BATCH_THRESHOLD") == 0) return threshold;
    return deft;
  };
}
}  // namespace

TEST_F(EnqueueMicrotest, EffectiveP2pBatchEnable_UserOverrideWinsOverEverything) {
  // userInput >= 0 returns immediately -- single node and a non-gfx950 arch,
  // both of which would otherwise force 0.
  BatchComm bc(/*nNodes=*/1, "gfx942");
  SetBatchParam(1);
  EXPECT_EQ(1, rcclEffectiveP2pBatchEnable(bc.get()));
  SetBatchParam(0);
  EXPECT_EQ(0, rcclEffectiveP2pBatchEnable(bc.get()));
  SetBatchParam(7);
  EXPECT_EQ(7, rcclEffectiveP2pBatchEnable(bc.get())) << "returned verbatim, not clamped";
}

TEST_F(EnqueueMicrotest, EffectiveP2pBatchEnable_SingleNode_IsDisabled) {
  // With the override off (-1), nNodes <= 1 short-circuits to 0 before the arch
  // check -- so even gfx950 gives 0 here.
  BatchComm bc(/*nNodes=*/1, "gfx950");
  SetBatchParam(-1);
  EXPECT_EQ(0, rcclEffectiveP2pBatchEnable(bc.get()));
}

TEST_F(EnqueueMicrotest, EffectiveP2pBatchEnable_MultiNodeGfx950_IsEnabled) {
  // The real gfx950 arm. IsArchMatch comes from the archinfo.cc oracle, so this
  // is production's own matching, not a fake's.
  BatchComm bc(/*nNodes=*/2, "gfx950");
  SetBatchParam(-1);
  EXPECT_EQ(1, rcclEffectiveP2pBatchEnable(bc.get()));
}

TEST_F(EnqueueMicrotest, EffectiveP2pBatchEnable_Gfx950WithAinic_IsDisabled) {
  // The `!rcclUseAinic()` conjunct -- the reason g_rcclUseAinicValue is a seam in
  // fakes/transport_stubs.cc rather than the fail-loud stub it used to be.
  // Without it the flag stays false in every test and dropping the conjunct
  // survives. Differential with MultiNodeGfx950_IsEnabled, which is identical
  // but for the AINIC flag.
  BatchComm bc(/*nNodes=*/2, "gfx950");
  SetBatchParam(-1);
  g_rcclUseAinicValue = true;
  EXPECT_EQ(0, rcclEffectiveP2pBatchEnable(bc.get()))
      << "gfx950 with AINIC must not enable p2p batching";
}

TEST_F(EnqueueMicrotest, EffectiveP2pBatchEnable_MultiNodeOtherArch_IsDisabled) {
  // Differential against the test above: same node count, different arch.
  for (const char* gcn : {"gfx942", "gfx90a", "gfx1100"}) {
    BatchComm bc(/*nNodes=*/2, gcn);
    SetBatchParam(-1);
    EXPECT_EQ(0, rcclEffectiveP2pBatchEnable(bc.get())) << "gcn=" << gcn;
  }
}

TEST_F(EnqueueMicrotest, P2pBatchEligible_Gfx942FourNode_ThresholdDefaultVersusSet) {
  // Reaches the enqueue wrapper, which the channel.h tests do not. gfx942 leaves
  // batching off unless RCCL_P2P_BATCH_ENABLE is set, so both cases force it on.
  constexpr ssize_t perPeer = 128 * 1024;
  BatchComm bc(/*nNodes=*/4, "gfx942");

  SetBatchEnableAndThreshold(/*threshold=*/-1, /*thresholdSet=*/false);
  EXPECT_TRUE(rcclP2pBatchEligible(bc.get(), perPeer, perPeer));

  SetBatchEnableAndThreshold(/*threshold=*/-1);
  EXPECT_TRUE(rcclP2pBatchEligible(bc.get(), perPeer, perPeer));

  SetBatchEnableAndThreshold(/*threshold=*/64 * 1024);
  EXPECT_FALSE(rcclP2pBatchEligible(bc.get(), perPeer, perPeer));

  SetBatchEnableAndThreshold(/*threshold=*/perPeer);
  EXPECT_TRUE(rcclP2pBatchEligible(bc.get(), perPeer, perPeer));
}

TEST_F(EnqueueMicrotest, P2pBatchEligible_Gfx950_UnsetThresholdFollowsNodeCount) {
  // 128 KiB is above the 64 KiB arch default. Multi-node gfx950 turns batching
  // on with the threshold left unset. BatchComm leaves nRanks at 0, so passing
  // that instead of nNodes would fuse the 16-node case.
  constexpr ssize_t perPeer = 128 * 1024;
  BatchComm at16(/*nNodes=*/16, "gfx950");
  EXPECT_FALSE(rcclP2pBatchEligible(at16.get(), perPeer, perPeer));
  BatchComm at15(/*nNodes=*/15, "gfx950");
  EXPECT_TRUE(rcclP2pBatchEligible(at15.get(), perPeer, perPeer));
}

TEST_F(EnqueueMicrotest, P2pBatchEligible_Gfx942_UnsetThresholdStopsFusingMultiMiB) {
  // Unset threshold on gfx942: fuse up to 1 MiB per peer, not above.
  SetBatchParam(1);
  for (int nNodes : {1, 2, 3, 4, 8, 16}) {
    BatchComm bc(nNodes, "gfx942");
    for (ssize_t perPeer : {ssize_t(128) << 10, ssize_t(512) << 10, ssize_t(1) << 20}) {
      EXPECT_TRUE(rcclP2pBatchEligible(bc.get(), perPeer, perPeer))
          << "nNodes=" << nNodes << " perPeer=" << perPeer;
    }
    for (ssize_t perPeer : {(ssize_t(1) << 20) + 1, ssize_t(8) << 20, ssize_t(32) << 20}) {
      EXPECT_FALSE(rcclP2pBatchEligible(bc.get(), perPeer, perPeer))
          << "nNodes=" << nNodes << " perPeer=" << perPeer;
    }
  }
}

TEST_F(EnqueueMicrotest, P2pBatchEligible_Gfx942EightNode_ExplicitThresholdStillWins) {
  constexpr ssize_t perPeer = ssize_t(8) << 20;
  BatchComm bc(/*nNodes=*/8, "gfx942");
  SetBatchEnableAndThreshold(perPeer);
  EXPECT_TRUE(rcclP2pBatchEligible(bc.get(), perPeer, perPeer));
  SetBatchEnableAndThreshold(64 * 1024);
  EXPECT_FALSE(rcclP2pBatchEligible(bc.get(), 128 * 1024, 128 * 1024));
}

TEST_F(EnqueueMicrotest, P2pBatchEligible_UnsetThresholdOtherArchsKeepNodeScopedCutoff) {
  // gfx950 keeps 64 KiB at 16+ nodes and no cutoff below that.
  constexpr ssize_t perPeer = ssize_t(8) << 20;
  SetBatchParam(1);
  BatchComm gfx950At8(/*nNodes=*/8, "gfx950");
  EXPECT_TRUE(rcclP2pBatchEligible(gfx950At8.get(), perPeer, perPeer));
  BatchComm gfx950At16(/*nNodes=*/16, "gfx950");
  EXPECT_FALSE(rcclP2pBatchEligible(gfx950At16.get(), 128 * 1024, 128 * 1024));
  EXPECT_TRUE(rcclP2pBatchEligible(gfx950At16.get(), 64 * 1024, 64 * 1024));
  BatchComm gfx1250At8(/*nNodes=*/8, "gfx1250");
  EXPECT_TRUE(rcclP2pBatchEligible(gfx1250At8.get(), perPeer, perPeer));
}

TEST_F(EnqueueMicrotest, P2pBatchEligible_OnlyRankUniformTasksFuse) {
  // The round's tasks decide too: only AlltoAll / direct AllGather / ReduceScatter tasks fuse,
  // so ranks pairing explicit send/recv (alltoallv) never fuse rounds the peer does not.
  constexpr ssize_t perPeer = 64 * 1024;
  BatchComm bc(/*nNodes=*/4, "gfx950");
  SetBatchParam(1);
  ncclTaskP2p a2aSend{}, a2aRecv{}, agSend{}, rsRecv{}, send{}, recv{};
  a2aSend.collAPI = ncclFuncAlltoAll;
  a2aRecv.collAPI = ncclFuncAlltoAll;
  agSend.collAPI = ncclFuncAllGather;
  rsRecv.collAPI = ncclFuncReduceScatter;
  send.collAPI = ncclFuncSend;
  recv.collAPI = ncclFuncRecv;

  ncclTaskP2p* a2a[2] = {&a2aRecv, &a2aSend};
  ncclTaskP2p* recvOnly[2] = {&a2aRecv, nullptr};
  ncclTaskP2p* direct[2] = {&rsRecv, &agSend};
  EXPECT_TRUE(rcclP2pBatchEligible(bc.get(), a2a, perPeer, perPeer));
  EXPECT_TRUE(rcclP2pBatchEligible(bc.get(), recvOnly, perPeer, perPeer));
  EXPECT_TRUE(rcclP2pBatchEligible(bc.get(), direct, perPeer, perPeer));

  ncclTaskP2p* sendRecv[2] = {&recv, &send};
  ncclTaskP2p* mixed[2] = {&a2aRecv, &send};
  ncclTaskP2p* none[2] = {nullptr, nullptr};
  EXPECT_FALSE(rcclP2pBatchEligible(bc.get(), sendRecv, perPeer, perPeer));
  EXPECT_FALSE(rcclP2pBatchEligible(bc.get(), mixed, perPeer, perPeer));
  EXPECT_FALSE(rcclP2pBatchEligible(bc.get(), none, perPeer, perPeer));

  // The size policy still applies on top.
  EXPECT_FALSE(rcclP2pBatchEligible(bc.get(), a2a, perPeer, perPeer / 2));
  SetBatchParam(0);
  EXPECT_FALSE(rcclP2pBatchEligible(bc.get(), a2a, perPeer, perPeer));
}
