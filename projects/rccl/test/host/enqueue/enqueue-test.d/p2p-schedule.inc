/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// scheduleP2pTasksToPlan: self pairs, round budgets and per-round kernel-argument fitting.
// Included into enqueue-test.cc's translation unit; see README.md.

namespace {
// Rank 0 of two with one p2p task queued to itself; round 0 pairs rank 0 with itself.
struct SelfP2pScene {
  CollTaskComm cc;
  std::vector<ncclKernelPlanner::Peer> peers{2};
  std::vector<std::remove_pointer_t<decltype(ncclComm::p2pSchedule)>> schedule{2};
  ncclTaskP2p task{};
  ncclKernelPlan plan{};
  ncclKernelPlanBudget budget{};
  SelfP2pScene() {
    ncclComm* c = cc.get();
    c->nRanks = 2;
    c->p2pnChannels = 2;
    c->p2pnChannelsPerPeer = 1;
    c->planner.peers = peers.data();
    c->planner.nTasksP2p = 1;
    schedule[1] = {1, 1};
    c->p2pSchedule = schedule.data();
    plan.kernelSpecialized = true;
  }
  ncclResult_t Run(std::string* log) {
    int epoch = 0;
    int round = 0;
    ncclResult_t rc = ncclSuccess;
    *log = RcclUnitTesting::CaptureLog([&] { rc = scheduleP2pTasksToPlan(cc.get(), &epoch, &round, &plan, &budget); });
    return rc;
  }
};
}  // namespace

TEST_F(EnqueueMicrotest, ScheduleP2pTasksToPlan_UnpairedSelfSendOrRecv_IsInvalidUsage) {
  for (bool send : {true, false}) {
    SelfP2pScene sc;
    ncclIntruQueueEnqueue(send ? &sc.peers[0].sendQueue : &sc.peers[0].recvQueue, &sc.task);
    std::string log;
    EXPECT_EQ(ncclInvalidUsage, sc.Run(&log));
    const char* warn = send ? "Trying to send to self without a matching recv"
                            : "Trying to recv to self without a matching send";
    EXPECT_TRUE(RcclUnitTesting::LogHas(log, warn)) << log;
  }
}

namespace {
// Rank 0 of one with a grouped send+recv to itself, the world-size-1 all_to_all shape (ROCM-33800).
struct SelfPairBudgetScene {
  BatchPlanComm bp{/*nNodes=*/1, /*cudaArch=*/1250};
  std::unique_ptr<ncclTopoSystem> topo{new ncclTopoSystem{}};
  std::vector<ncclKernelPlanner::Peer> peers{1};
  std::vector<std::remove_pointer_t<decltype(ncclComm::p2pSchedule)>> schedule{1};
  char sendBuf[64]{};
  char recvBuf[64]{};
  ncclTaskP2p sendTask{};
  ncclTaskP2p recvTask{};
  ncclKernelPlanBudget budget{};

  explicit SelfPairBudgetScene(int channelsPerPeer, bool inPlace = false) {
    InstallSendRecvDevFuncIds();
    ncclComm* c = bp.c();
    c->rank = 0;
    c->nRanks = 1;
    c->p2pnChannels = channelsPerPeer;
    c->p2pnChannelsPerPeer = channelsPerPeer;
    c->p2pChunkSize = 128 << 10;
    c->buffSizes[NCCL_PROTO_SIMPLE] = 1 << 20;
    c->workArgsBytes = ncclMaxKernelArgsSize();
    c->workFifoBytes = 1 << 20;
    SetSingleGpuArch(topo.get(), "gfx1250");
    c->topo = topo.get();
    c->planner.peers = peers.data();
    schedule[0] = {0, 0};
    c->p2pSchedule = schedule.data();
    initTask(&sendTask, ncclFuncSend, sendBuf);
    initTask(&recvTask, ncclFuncRecv, inPlace ? sendBuf : recvBuf);
    ncclIntruQueueEnqueue(&peers[0].sendQueue, &sendTask);
    ncclIntruQueueEnqueue(&peers[0].recvQueue, &recvTask);
    c->planner.nTasksP2p = 2;
    c->planner.nTasksP2pSend = 1;
    c->planner.nTasksP2pRecv = 1;
    bp.p()->kernelSpecialized = true;
    budget.inArgsBytes = c->workArgsBytes - sizeof(struct ncclDevKernelArgs);
    budget.outArgsBytes = c->workFifoBytes / 2;
  }
  // Splits into one 16 KiB minimum part per channel at 256 channels; the host path never reads the buffers.
  static constexpr size_t kBytes = 4 << 20;
  static void initTask(ncclTaskP2p* task, ncclFunc_t func, char* buf) {
    task->func = func;
    task->collAPI = func;
    task->buff = buf;
    task->bytes = kBytes;
    task->maxCTAs = MAXCHANNELS;  // p2pTaskAppend's default for a plain send/recv
  }
  ncclResult_t Schedule(std::string* log) {
    int epoch = 0;
    int round = 0;
    ncclResult_t rc = ncclSuccess;
    *log = RcclUnitTesting::CaptureLog(
        [&] { rc = scheduleP2pTasksToPlan(bp.c(), &epoch, &round, bp.p(), &budget); });
    return rc;
  }
};
}  // namespace

TEST_F(EnqueueMicrotest, ScheduleP2pTasksToPlan_SelfPair256ChannelsDefaultArgs_SchedulesInOnePlan) {
  SelfPairBudgetScene sc(/*channelsPerPeer=*/256);
  std::string log;
  ASSERT_EQ(ncclSuccess, sc.Schedule(&log)) << log;
  EXPECT_EQ(0, sc.bp.c()->planner.nTasksP2p) << "the pair must leave the planner in this plan";
  EXPECT_EQ(ssize_t(sizeof(struct ncclDevWorkP2p)), ssize_t(sc.bp.p()->workBytes));
  EXPECT_EQ(256, sc.bp.p()->nWorkBatches) << "one batch per channel part";
}

TEST_F(EnqueueMicrotest, ScheduleP2pTasksToPlan_RoundDoesNotFitEmptyPlan_IsInvalidUsage) {
  SelfPairBudgetScene sc(/*channelsPerPeer=*/256);
  sc.budget.inArgsBytes = 255 * kBatch;  // one batch short of a full round
  std::string log;
  EXPECT_EQ(ncclInvalidUsage, sc.Schedule(&log));
  EXPECT_TRUE(RcclUnitTesting::LogHas(log, "does not fit an empty kernel plan")) << log;
  EXPECT_EQ(2, sc.bp.c()->planner.nTasksP2p);
}

TEST_F(EnqueueMicrotest, ScheduleP2pTasksToPlan_RoundExactlyFillsEmptyPlan_Schedules) {
  SelfPairBudgetScene sc(/*channelsPerPeer=*/256);
  sc.budget.inArgsBytes = 256 * kBatch;  // exactly one batch per channel
  std::string log;
  ASSERT_EQ(ncclSuccess, sc.Schedule(&log)) << log;
  EXPECT_EQ(0, sc.bp.c()->planner.nTasksP2p);
  EXPECT_EQ(256, sc.bp.p()->nWorkBatches);
}

TEST_F(EnqueueMicrotest, ScheduleP2pTasksToPlan_RoundDoesNotFitPartialPlan_DefersToNextPlan) {
  SelfPairBudgetScene sc(/*channelsPerPeer=*/256);
  sc.bp.p()->nWorkBatches = 1;  // earlier work already in this plan
  sc.budget.inArgsBytes = 256 * kBatch;
  std::string log;
  EXPECT_EQ(ncclSuccess, sc.Schedule(&log)) << log;
  EXPECT_EQ(2, sc.bp.c()->planner.nTasksP2p) << "the round waits for a fresh plan";
}

TEST_F(EnqueueMicrotest, LaunchPrepare_P2pRoundNeverFits_ReturnsErrorInsteadOfSpinning) {
  SelfPairBudgetScene sc(/*channelsPerPeer=*/256);
  sc.bp.c()->workArgsBytes = sizeof(struct ncclDevKernelArgs) + 255 * kBatch;
  ncclResult_t rc = ncclSuccess;
  const std::string log = RcclUnitTesting::CaptureLog([&] { rc = ncclLaunchPrepare(sc.bp.c()); });
  EXPECT_EQ(ncclInvalidUsage, rc);
  EXPECT_TRUE(RcclUnitTesting::LogHas(log, "does not fit an empty kernel plan")) << log;
}

TEST_F(EnqueueMicrotest, LaunchPrepare_InPlaceSelfPairOnly_ReturnsEmptyPlanToPool) {
  // An in-place self pair with P2P batching off is consumed without work, leaving an empty plan.
  SelfPairBudgetScene sc(/*channelsPerPeer=*/1, /*inPlace=*/true);
  ASSERT_EQ(ncclSuccess, ncclLaunchPrepare(sc.bp.c()));
  EXPECT_EQ(0, sc.bp.c()->planner.nTasksP2p);
  EXPECT_NE(nullptr, sc.bp.c()->memPool_ncclKernelPlan.head) << "the empty plan must go back to the pool";
}

// ===========================================================================
// scheduleP2pTasksToPlan -- per-round kernel-argument budget.
// A round emits one work item and at most one new batch per channel. Reserving
// more than that moved the last round of a 4-rank, 64-channel-per-peer AllToAll
// into a second kernel launch.
// ===========================================================================

namespace {
struct AllToAllPlanScene {
  static constexpr int kRanks = 4;
  static constexpr int kChannelsPerPeer = 64;
  static constexpr int kP2pChannels = kRanks * kChannelsPerPeer;

  BatchPlanComm bp{/*nNodes=*/1, /*cudaArch=*/942};
  std::unique_ptr<ncclTopoSystem> topo{new ncclTopoSystem{}};
  std::vector<ncclComm::P2pSchedulePair> rounds = std::vector<ncclComm::P2pSchedulePair>(kRanks);
  std::vector<ncclKernelPlanner::Peer> plannerPeers = std::vector<ncclKernelPlanner::Peer>(kRanks);
  std::vector<ncclChannelPeer> channelPeers = std::vector<ncclChannelPeer>(kRanks);
  std::vector<ncclChannelPeer*> peerSlots = std::vector<ncclChannelPeer*>(kRanks);
  std::vector<ncclTaskP2p> sends = std::vector<ncclTaskP2p>(kRanks);
  std::vector<ncclTaskP2p> recvs = std::vector<ncclTaskP2p>(kRanks);
  std::vector<char> sendBufs = std::vector<char>(kRanks);
  std::vector<char> recvBufs = std::vector<char>(kRanks);

  AllToAllPlanScene() {
    InstallSendRecvDevFuncIds();
    ncclComm* c = bp.c();
    c->rank = 0;
    c->nRanks = kRanks;
    c->p2pnChannels = kP2pChannels;
    c->p2pnChannelsPerPeer = kChannelsPerPeer;
    c->p2pChannelShiftSize = 0;
    c->p2pChunkSize = 512 << 10;
    c->buffSizes[NCCL_PROTO_SIMPLE] = 4 << 20;
    SetSingleGpuArch(topo.get(), "gfx942");
    c->topo = topo.get();

    for (int round = 0; round < kRanks; ++round) {
      rounds[round].sendRank = round;
      rounds[round].recvRank = (kRanks - round) % kRanks;
    }
    c->p2pSchedule = rounds.data();
    c->planner.peers = plannerPeers.data();
    for (int peer = 0; peer < kRanks; ++peer) peerSlots[peer] = &channelPeers[peer];
    for (int ch = 0; ch < kP2pChannels; ++ch) c->channels[ch].peers = peerSlots.data();

    for (int peer = 0; peer < kRanks; ++peer) {
      for (ncclTaskP2p* t : {&sends[peer], &recvs[peer]}) {
        t->collAPI = ncclFuncAlltoAll;
        t->bytes = size_t(1) << 30;
        t->maxCTAs = MAXCHANNELS;
      }
      sends[peer].func = ncclFuncSend;
      sends[peer].buff = &sendBufs[peer];
      recvs[peer].func = ncclFuncRecv;
      recvs[peer].buff = &recvBufs[peer];
      ncclIntruQueueEnqueue(&plannerPeers[peer].sendQueue, &sends[peer]);
      ncclIntruQueueEnqueue(&plannerPeers[peer].recvQueue, &recvs[peer]);
    }
    c->planner.nTasksP2p = 2 * kRanks;
    c->planner.nTasksP2pSend = kRanks;
    c->planner.nTasksP2pRecv = kRanks;
  }

  ncclResult_t schedule(ssize_t inArgsBytes, ssize_t outArgsBytes = ssize_t(1) << 30) {
    ncclKernelPlanBudget budget = MakeBudget(inArgsBytes, outArgsBytes);
    int epoch = 0, round = 0;
    return scheduleP2pTasksToPlan(bp.c(), &epoch, &round, bp.p(), &budget);
  }
};
}  // namespace

TEST_F(EnqueueMicrotest, ScheduleP2p_FourRoundsOf64Channels_FitOneKernelLaunch) {
  AllToAllPlanScene sc;
  const ssize_t inArgs = ssize_t(ncclMaxKernelArgsSize(sc.bp.c()->cudaArch)) - ssize_t(sizeof(struct ncclDevKernelArgs));
  ASSERT_GE(inArgs, ssize_t(AllToAllPlanScene::kP2pChannels) * kBatch)
      << "precondition: the kernel-argument budget holds one batch per channel";
  ASSERT_EQ(ncclSuccess, sc.schedule(inArgs));
  EXPECT_EQ(0, sc.bp.c()->planner.nTasksP2p) << "every AllToAll round must land in the first plan";
  EXPECT_EQ(AllToAllPlanScene::kP2pChannels, sc.bp.p()->nWorkBatches);
}

TEST_F(EnqueueMicrotest, ScheduleP2p_ReservesExactlyOneBatchPerChannelForTheNextRound) {
  // Three rounds use 192 batches. The fourth needs 64 more: one batch short must
  // defer it to the next plan, an exact fit must not.
  const int used = 3 * AllToAllPlanScene::kChannelsPerPeer;
  {
    AllToAllPlanScene sc;
    ASSERT_EQ(ncclSuccess, sc.schedule((used + AllToAllPlanScene::kChannelsPerPeer - 1) * kBatch));
    EXPECT_EQ(2, sc.bp.c()->planner.nTasksP2p) << "the fourth round's send and recv must wait";
    EXPECT_EQ(used, sc.bp.p()->nWorkBatches);
  }
  {
    AllToAllPlanScene sc;
    ASSERT_EQ(ncclSuccess, sc.schedule((used + AllToAllPlanScene::kChannelsPerPeer) * kBatch));
    EXPECT_EQ(0, sc.bp.c()->planner.nTasksP2p);
    EXPECT_EQ(used + AllToAllPlanScene::kChannelsPerPeer, sc.bp.p()->nWorkBatches);
  }
}

TEST_F(EnqueueMicrotest, ScheduleP2p_ReservesExactlyOneWorkItemForTheNextRound) {
  const ssize_t inArgs = AllToAllPlanScene::kP2pChannels * kBatch;
  const ssize_t work = sizeof(struct ncclDevWorkP2p);
  {
    AllToAllPlanScene sc;
    ASSERT_EQ(ncclSuccess, sc.schedule(inArgs, 4 * work - 1));
    EXPECT_EQ(2, sc.bp.c()->planner.nTasksP2p) << "the fourth round's send and recv must wait";
    EXPECT_EQ(3 * work, sc.bp.p()->workBytes);
  }
  {
    AllToAllPlanScene sc;
    ASSERT_EQ(ncclSuccess, sc.schedule(inArgs, 4 * work));
    EXPECT_EQ(0, sc.bp.c()->planner.nTasksP2p);
    EXPECT_EQ(4 * work, sc.bp.p()->workBytes);
  }
}


// ===========================================================================
// Direct P2P AllToAll peer layers (RCCL_A2A_P2P_PEERS_IN_FLIGHT): the plan cut before each layer and the
// pairing of a round's send and receive.
// ===========================================================================

namespace {
// A layered AllToAll plan on AllToAllPlanScene: 4 ranks, 64 channels per peer, 256 channels; K=2 gives 128 per peer.
struct LayeredAllToAllScene : AllToAllPlanScene {
  // Room for every round in one plan, so only the layer rule ends a plan.
  static constexpr ssize_t kRoomyArgs = ssize_t(1) << 20;
  ssize_t inArgs = ssize_t(ncclMaxKernelArgsSize(bp.c()->cudaArch)) - ssize_t(sizeof(struct ncclDevKernelArgs));
  LayeredAllToAllScene() {
    SetPeersInFlight(2);
    for (int peer = 0; peer < kRanks; ++peer) {
      sends[peer].root = peer;
      recvs[peer].root = peer;
    }
  }
  // Channels holding n batches in the current plan.
  int channelsWith(int n) {
    int count = 0;
    for (int ch = 0; ch < kP2pChannels; ++ch) {
      count += bp.queueLength(ch) == n ? 1 : 0;
    }
    return count;
  }
  ncclDevWorkP2p* work(int index) {
    ncclWorkList* node = bp.p()->workQueue.head;
    for (int i = 0; i < index && node != nullptr; ++i) {
      node = node->next;
    }
    return node == nullptr ? nullptr : reinterpret_cast<ncclDevWorkP2p*>(node + 1);
  }
  // A fresh plan, as ncclLaunchPrepare starts one.
  void newPlan() {
    bp.plan.reset(new ncclKernelPlan{});
    std::memset(&bp.c()->planner.wipPlan, 0, sizeof(bp.c()->planner.wipPlan));
  }
};
}  // namespace

TEST_F(EnqueueMicrotest, ScheduleP2p_LayeredAllToAllEndsThePlanBeforeEachLayer) {
  LayeredAllToAllScene sc;
  ASSERT_EQ(ncclSuccess, sc.schedule(LayeredAllToAllScene::kRoomyArgs));
  EXPECT_EQ(4, sc.bp.c()->planner.nTasksP2p) << "rounds 2 and 3 are the second layer: next plan";
  // Self copy (round 0) keeps its 64 channels 0-63; round 1 takes set 1 = channels 128-255.
  EXPECT_EQ(64 + 128, sc.bp.p()->nWorkBatches);
  ncclDevWorkP2p* peerWork = sc.work(1);
  ASSERT_NE(nullptr, peerWork);
  EXPECT_EQ(1, peerWork->sendRank);
  EXPECT_EQ(128, peerWork->nSendChannels);
  EXPECT_EQ(128, peerWork->nRecvChannels);
  EXPECT_EQ(2, peerWork->channelBase) << "set 1 starts at channel 2 * 64";
  for (int ch = 0; ch < AllToAllPlanScene::kP2pChannels; ++ch) {
    EXPECT_EQ(ch < 64 || ch >= 128 ? 1 : 0, sc.bp.queueLength(ch)) << "channel " << ch;
  }

  sc.newPlan();
  ASSERT_EQ(ncclSuccess, sc.schedule(LayeredAllToAllScene::kRoomyArgs));
  EXPECT_EQ(0, sc.bp.c()->planner.nTasksP2p);
  EXPECT_EQ(256, sc.bp.p()->nWorkBatches) << "rounds 2 and 3 fill sets 0 and 1";
  ASSERT_NE(nullptr, sc.work(1));
  EXPECT_EQ(0, sc.work(0)->channelBase);
  EXPECT_EQ(2, sc.work(1)->channelBase);
  EXPECT_EQ(256, sc.channelsWith(1)) << "one batch on every channel";
}

TEST_F(EnqueueMicrotest, ScheduleP2p_LayeredRoundReservesItsWholeChannelSet) {
  // The self copy takes 64 batches; round 1 needs its 128-channel set. One batch short defers it, an exact fit not.
  {
    LayeredAllToAllScene sc;
    ASSERT_EQ(ncclSuccess, sc.schedule((64 + 128 - 1) * kBatch));
    EXPECT_EQ(6, sc.bp.c()->planner.nTasksP2p) << "round 1 must wait for the next plan";
    EXPECT_EQ(64, sc.bp.p()->nWorkBatches);
  }
  {
    LayeredAllToAllScene sc;
    ASSERT_EQ(ncclSuccess, sc.schedule((64 + 128) * kBatch));
    EXPECT_EQ(4, sc.bp.c()->planner.nTasksP2p);
    EXPECT_EQ(64 + 128, sc.bp.p()->nWorkBatches);
  }
}

TEST_F(EnqueueMicrotest, ScheduleP2p_LayeringOffForSmallPeersOrCappedCtas) {
  {
    LayeredAllToAllScene sc;
    for (int peer = 0; peer < AllToAllPlanScene::kRanks; ++peer) {
      sc.sends[peer].bytes = (ssize_t(4) << 20) - 1;
      sc.recvs[peer].bytes = (ssize_t(4) << 20) - 1;
    }
    ASSERT_EQ(ncclSuccess, sc.schedule(sc.inArgs));
    EXPECT_EQ(0, sc.bp.c()->planner.nTasksP2p) << "below 4 MiB per peer: one plan as with the knob off";
  }
  {
    LayeredAllToAllScene sc;
    sc.sends[1].maxCTAs = MAXCHANNELS - 1;
    sc.recvs[3].maxCTAs = MAXCHANNELS - 1;
    EXPECT_EQ(0, rcclA2aP2pLayerPairParts(sc.bp.c(), &sc.recvs[3], &sc.sends[1]));
    EXPECT_EQ(0, rcclA2aP2pLayerPairParts(sc.bp.c(), &sc.recvs[2], &sc.sends[1]));
    EXPECT_EQ(0, rcclA2aP2pLayerPairParts(sc.bp.c(), &sc.recvs[3], &sc.sends[2])) << "only the recv capped";
    EXPECT_EQ(128, rcclA2aP2pLayerPairParts(sc.bp.c(), &sc.recvs[2], &sc.sends[2]));
    EXPECT_EQ(0, rcclA2aP2pLayerPairParts(sc.bp.c(), nullptr, &sc.sends[2]));
  }
}

// A round pairs one send and one recv. Both must be layered tasks of the same AllToAll, or the round keeps the default
// plan; otherwise one end could layer a link its peer does not.
TEST_F(EnqueueMicrotest, A2aP2pLayerPairParts_OffWhenTheRoundPairsAnotherOp) {
  LayeredAllToAllScene sc;
  ASSERT_EQ(128, rcclA2aP2pLayerPairParts(sc.bp.c(), &sc.recvs[2], &sc.sends[2]));
  sc.recvs[2].collAPI = ncclFuncRecv;
  EXPECT_EQ(0, rcclA2aP2pLayerPairParts(sc.bp.c(), &sc.recvs[2], &sc.sends[2])) << "explicit recv";
  sc.recvs[2].collAPI = ncclFuncAlltoAll;
  sc.sends[2].collAPI = ncclFuncSend;
  EXPECT_EQ(0, rcclA2aP2pLayerPairParts(sc.bp.c(), &sc.recvs[2], &sc.sends[2])) << "explicit send";
  sc.sends[2].collAPI = ncclFuncAlltoAll;
  sc.recvs[2].bytes = (ssize_t(4) << 20) - 1;
  EXPECT_EQ(0, rcclA2aP2pLayerPairParts(sc.bp.c(), &sc.recvs[2], &sc.sends[2])) << "small recv";
  sc.recvs[2].bytes = kLayerPeerBytes;
  sc.recvs[2].collOpCount = 1;
  sc.sends[2].collOpCount = 2;
  EXPECT_EQ(0, rcclA2aP2pLayerPairParts(sc.bp.c(), &sc.recvs[2], &sc.sends[2])) << "two AllToAll calls";
}

TEST_F(EnqueueMicrotest, ScheduleP2p_Gfx1250LayersALargeAllToAllByDefault) {
  struct Case {
    int cudaArch;
    int64_t knob;
    int perPeer;
  };
  // 8 channels per peer: unset on gfx1250 gives 16 in flight, 16 channels per peer at set round mod 16. The knob at 0
  // on gfx1250 and unset on gfx942 keep 8 per peer.
  for (Case k : {Case{1250, kPeersInFlightUnset, 16}, Case{1250, 0, 8}, Case{942, kPeersInFlightUnset, 8}}) {
    LayeredAllToAllScene sc;
    SetPeersInFlight(k.knob);
    sc.bp.c()->cudaArch = k.cudaArch;
    sc.bp.c()->p2pnChannelsPerPeer = 8;
    ASSERT_EQ(ncclSuccess, sc.schedule(LayeredAllToAllScene::kRoomyArgs));
    EXPECT_EQ(0, sc.bp.c()->planner.nTasksP2p) << "four rounds fit one plan";
    ncclDevWorkP2p* peerWork = sc.work(1);
    ASSERT_NE(nullptr, peerWork);
    EXPECT_EQ(1, peerWork->sendRank);
    EXPECT_EQ(k.perPeer, peerWork->nSendChannels) << "cudaArch=" << k.cudaArch << " knob=" << k.knob;
    EXPECT_EQ(k.perPeer, peerWork->nRecvChannels) << "cudaArch=" << k.cudaArch << " knob=" << k.knob;
    if (k.perPeer == 16) {
      EXPECT_EQ(2, peerWork->channelBase) << "set 1 starts at channel 16 = base 2 * 8";
      EXPECT_EQ(8 + 3 * 16, sc.bp.p()->nWorkBatches) << "the self copy keeps 8 channels";
    } else {
      EXPECT_EQ(4 * 8, sc.bp.p()->nWorkBatches);
    }
  }
}
