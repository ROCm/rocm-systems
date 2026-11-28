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
// isolateP2pCollTasks / scheduleP2pTasksToPlan -- with P2P batching on, a
// collective's tasks never share a round with explicit send/recv, and the
// collectives of one group still share a plan once the explicit ops drained.
// ===========================================================================

namespace {
struct TwoCollectivesScene {
  static constexpr int kRanks = 4;

  BatchPlanComm bp{/*nNodes=*/1, /*cudaArch=*/942};
  std::unique_ptr<ncclTopoSystem> topo{new ncclTopoSystem{}};
  std::vector<ncclComm::P2pSchedulePair> rounds = std::vector<ncclComm::P2pSchedulePair>(kRanks);
  std::vector<ncclKernelPlanner::Peer> plannerPeers = std::vector<ncclKernelPlanner::Peer>(kRanks);
  std::vector<ncclChannelPeer> channelPeers = std::vector<ncclChannelPeer>(kRanks);
  std::vector<ncclChannelPeer*> peerSlots = std::vector<ncclChannelPeer*>(kRanks);
  std::vector<ncclTaskP2p> tasks = std::vector<ncclTaskP2p>(4 * kRanks + 1);
  std::vector<char> bufs = std::vector<char>(4 * kRanks + 1);
  ncclTaskP2p* explicitSend = nullptr;

  // An AlltoAll (collOpCount 1) then a secondApi collective (collOpCount 2); withExplicitSend
  // adds an ncclSend to peer 1.
  explicit TwoCollectivesScene(bool withExplicitSend, ncclFunc_t secondApi = ncclFuncAlltoAll) {
    InstallSendRecvDevFuncIds();
    ncclComm* c = bp.c();
    c->rank = 0;
    c->nRanks = kRanks;
    c->p2pnChannels = kRanks;
    c->p2pnChannelsPerPeer = 1;
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
    for (int ch = 0; ch < kRanks; ++ch) c->channels[ch].peers = peerSlots.data();

    int next = 0;
    for (int coll = 1; coll <= 2; ++coll) {
      for (int peer = 0; peer < kRanks; ++peer) {
        for (bool isSend : {true, false}) {
          ncclTaskP2p* t = &tasks[next];
          t->func = isSend ? ncclFuncSend : ncclFuncRecv;
          t->collAPI = coll == 1 ? ncclFuncAlltoAll : secondApi;
          t->collOpCount = coll;
          t->bytes = 4096;
          t->maxCTAs = MAXCHANNELS;
          t->root = peer;
          t->buff = &bufs[next++];
          ncclIntruQueueEnqueue(isSend ? &plannerPeers[peer].sendQueue : &plannerPeers[peer].recvQueue, t);
        }
      }
    }
    c->planner.nTasksP2p = 4 * kRanks;
    c->planner.p2pCollOpCount = 2;
    if (withExplicitSend) {
      explicitSend = &tasks[next];
      explicitSend->func = ncclFuncSend;
      explicitSend->collAPI = ncclFuncSend;
      explicitSend->collOpCount = 0;
      explicitSend->bytes = 4096;
      explicitSend->maxCTAs = MAXCHANNELS;
      explicitSend->root = 1;
      explicitSend->buff = &bufs[next];
      ncclIntruQueueEnqueue(&plannerPeers[1].sendQueue, explicitSend);
      c->planner.nTasksP2p += 1;
    }
  }

  ncclResult_t schedule() {
    ncclKernelPlanBudget budget = MakeBudget(ssize_t(1) << 30, ssize_t(1) << 30);
    int epoch = 0, round = 0;
    return scheduleP2pTasksToPlan(bp.c(), &epoch, &round, bp.p(), &budget);
  }
};
}  // namespace

TEST_F(EnqueueMicrotest, IsolateP2pCollTasks_BatchingOn_KeepsExplicitSendOutOfCollectiveRounds) {
  TwoCollectivesScene sc(/*withExplicitSend=*/true);
  SetParam("RCCL_P2P_BATCH_ENABLE", 1);
  isolateP2pCollTasks(sc.bp.c());
  // Only the explicit send stays in the peer queues; every collective task waits in its own queue.
  EXPECT_EQ(sc.explicitSend, ncclIntruQueueHead(&sc.plannerPeers[1].sendQueue));
  EXPECT_EQ(nullptr, sc.explicitSend->next);
  for (int peer = 0; peer < TwoCollectivesScene::kRanks; ++peer) {
    EXPECT_TRUE(ncclIntruQueueEmpty(&sc.plannerPeers[peer].recvQueue)) << "peer " << peer;
    if (peer != 1) EXPECT_TRUE(ncclIntruQueueEmpty(&sc.plannerPeers[peer].sendQueue)) << "peer " << peer;
  }
  EXPECT_EQ(4 * TwoCollectivesScene::kRanks, sc.bp.c()->planner.p2pCollTaskQueue.nElems);
}

TEST_F(EnqueueMicrotest, IsolateP2pCollTasks_BatchingOff_LeavesTheQueuesAlone) {
  TwoCollectivesScene sc(/*withExplicitSend=*/true);
  SetParam("RCCL_P2P_BATCH_ENABLE", 0);
  isolateP2pCollTasks(sc.bp.c());
  EXPECT_EQ(0, sc.bp.c()->planner.p2pCollTaskQueue.nElems);
  EXPECT_NE(sc.explicitSend, ncclIntruQueueHead(&sc.plannerPeers[1].sendQueue)) << "the collective send stays first";
}

TEST_F(EnqueueMicrotest, ScheduleP2p_BatchIsolatedCollectives_ShareOnePlan) {
  // Isolation that comes only from batching keeps grouped collectives in one plan.
  TwoCollectivesScene sc(/*withExplicitSend=*/false);
  SetParam("RCCL_P2P_BATCH_ENABLE", 1);
  isolateP2pCollTasks(sc.bp.c());
  ASSERT_EQ(ncclSuccess, sc.schedule());
  EXPECT_EQ(0, sc.bp.c()->planner.nTasksP2p) << "both AlltoAlls must land in the first plan";
}

TEST_F(EnqueueMicrotest, ScheduleP2p_BatchIsolatedCollectivesOfOneLayout_ShareOnePlan) {
  TwoCollectivesScene sc(/*withExplicitSend=*/false, ncclFuncReduceScatter);
  SetParam("RCCL_P2P_BATCH_ENABLE", 1);
  isolateP2pCollTasks(sc.bp.c());
  ASSERT_EQ(ncclSuccess, sc.schedule());
  EXPECT_EQ(0, sc.bp.c()->planner.nTasksP2p) << "AlltoAll and direct ReduceScatter share the batched layout";
}

TEST_F(EnqueueMicrotest, ScheduleP2p_BatchIsolatedCollectivesOfMixedLayouts_SplitPlans) {
  // A Gather takes the unbatched layout; sharing a round with an AlltoAll would give the two ends
  // of one Gather transfer different channel bases.
  for (ncclFunc_t unbatched : {ncclFuncGather, ncclFuncScatter}) {
    TwoCollectivesScene sc(/*withExplicitSend=*/false, unbatched);
    SetParam("RCCL_P2P_BATCH_ENABLE", 1);
    isolateP2pCollTasks(sc.bp.c());
    ASSERT_EQ(ncclSuccess, sc.schedule());
    EXPECT_EQ(2 * TwoCollectivesScene::kRanks, sc.bp.c()->planner.nTasksP2p)
      << "collAPI " << int(unbatched) << " must wait for the next plan";
  }
}

TEST_F(EnqueueMicrotest, ScheduleP2p_MaxCtasIsolation_StillGivesEachCollectiveItsOwnPlan) {
  TwoCollectivesScene sc(/*withExplicitSend=*/false);
  SetParam("RCCL_P2P_BATCH_ENABLE", 1);
  sc.bp.c()->planner.p2pCollNeedsIsolation = true;
  isolateP2pCollTasks(sc.bp.c());
  ASSERT_EQ(ncclSuccess, sc.schedule());
  EXPECT_EQ(2 * TwoCollectivesScene::kRanks, sc.bp.c()->planner.nTasksP2p) << "the second AlltoAll waits";
}
