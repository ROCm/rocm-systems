/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// P2p channel policy: rcclP2pTaskParts, rcclP2pPolicyChannels, calcP2pChannelCount and batching enable.
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
  // SetParam covers one key; this case has to set the enable and the threshold.
  constexpr ssize_t perPeer = 128 * 1024;
  BatchComm bc(/*nNodes=*/4, "gfx942");
  auto setEnableAndThreshold = [](int64_t threshold, bool thresholdSet) {
    g_loadParam = [threshold, thresholdSet](const char* env, int64_t deft) {
      if (std::strcmp(env, "RCCL_P2P_BATCH_ENABLE") == 0) return int64_t(1);
      if (thresholdSet && std::strcmp(env, "RCCL_P2P_BATCH_THRESHOLD") == 0) return threshold;
      return deft;
    };
  };

  setEnableAndThreshold(/*threshold=*/-1, /*thresholdSet=*/false);
  EXPECT_TRUE(rcclP2pBatchEligible(bc.get(), perPeer, perPeer));

  setEnableAndThreshold(/*threshold=*/-1, /*thresholdSet=*/true);
  EXPECT_TRUE(rcclP2pBatchEligible(bc.get(), perPeer, perPeer));

  setEnableAndThreshold(/*threshold=*/64 * 1024, /*thresholdSet=*/true);
  EXPECT_FALSE(rcclP2pBatchEligible(bc.get(), perPeer, perPeer));

  setEnableAndThreshold(/*threshold=*/perPeer, /*thresholdSet=*/true);
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
