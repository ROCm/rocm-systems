/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// ncclTasksRegAndEnqueue: buffer registration and the warp-speed schedule.
// Included into enqueue-test.cc's translation unit; see README.md.

// ===========================================================================
// ncclTasksRegAndEnqueue (enqueue.cc:414)
// ===========================================================================

TEST_F(EnqueueMicrotest, TasksRegAndEnqueue_UnresolvableRegVariantId_IsInvalidUsage) {
  // nccl_stubs.cc leaves ncclDevFuncNameToId empty, so the reg-variant arm's
  // ncclDevFuncId returns -1 by construction. Unchecked it would dispatch as a funcId.
  BatchPlanComm bp;
  ScopedHook reg(g_ncclRegisterCollBuffers,
                 [](struct ncclComm*, struct ncclTaskColl*, void**, void**,
                    struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>*,
                    bool* needConnect) {
                   if (needConnect) *needConnect = false;
                   return ncclSuccess;
                 });

  ncclTaskColl task = {};
  task.func = ncclFuncAllReduce;      // in ll128_reg_variant_colls
  task.protocol = NCCL_PROTO_LL128;   // ...and LL128, so the reg-variant arm runs
  task.algorithm = NCCL_ALGO_RING;    // not NVLS: those skip the arm entirely
  ncclIntruQueueEnqueue(&bp.c()->planner.collTaskQueue, &task);

  EXPECT_EQ(ncclInvalidUsage, ncclTasksRegAndEnqueue(bp.c()));
  EXPECT_EQ(1, reg.calls);
}

namespace {

struct WarpSpeedScheduleScene {
  BatchPlanComm bp;
  std::unique_ptr<ncclTopoSystem> topo{new ncclTopoSystem{}};
  std::vector<ncclTaskColl> tasks;
  std::vector<std::unique_ptr<unsigned char[]>> workStorage;
  std::vector<ncclKernelPlanner::Peer> peers;
  std::vector<std::remove_pointer_t<decltype(ncclComm::p2pSchedule)>> p2pSchedule;
  std::vector<ncclChannelPeer> channelPeers;
  std::vector<ncclChannelPeer*> channelPeerSlots;
  ncclTaskP2p sendTask{};
  ncclTaskP2p recvTask{};
  unsigned char sendBuffer[16]{};
  unsigned char recvBuffer[16]{};
  ncclKernelPlanBudget budget{};
  char archName[24] = "gfx950:sramecc+:xnack-";
  int maxWorkBatches;
  bool planPersistent = true;

  WarpSpeedScheduleScene(std::vector<bool> const& useWarpSpeed, std::vector<int> const& algorithms,
                         int maxBatches, bool historicalMultistream = false)
      : tasks(useWarpSpeed.size()), workStorage(useWarpSpeed.size()),
        maxWorkBatches(maxBatches) {
    EXPECT_EQ(useWarpSpeed.size(), algorithms.size());
    ncclComm* comm = bp.c();
    comm->nRanks = historicalMultistream ? 8 : 16;
    comm->nNodes = historicalMultistream ? 1 : 2;
    comm->maxLocalRanks = 8;
    comm->archName = archName;
    comm->topo = topo.get();
    SetSingleGpuArch(topo.get(), "gfx950");
    comm->nChannels = 1;
    comm->nvlsChannels = 1;
    comm->WarpSize = 64;
    comm->buffSizes[NCCL_PROTO_SIMPLE] = 1 << 22;
    comm->workArgsBytes = 1 << 20;
    comm->workFifoBytes = 1 << 20;
    comm->planner.nTasksColl = static_cast<int>(tasks.size());
    ResetPlan();

    for (size_t i = 0; i < tasks.size(); ++i) {
      ncclTaskColl& task = tasks[i];
      task.func = ncclFuncAllReduce;
      task.datatype = ncclFloat;
      task.count = historicalMultistream ? 1048576 : 1024 + i;
      task.opHost = ncclSum;
      task.algorithm = algorithms[i];
      task.protocol = NCCL_PROTO_SIMPLE;
      task.nMaxChannels = 1;
      task.minCTAs = 1;
      task.maxCTAs = 1;
      task.nvlsCTAs = NCCL_CONFIG_UNDEF_INT;
      task.nWarps = 4;
      task.chunkSteps = 1;
      task.sliceSteps = 1;
      task.trafficBytes = 8192;
      task.devFuncId = 0;
#ifdef ENABLE_WARP_SPEED
      task.useWarpSpeed = useWarpSpeed[i];
#endif
      task.cgaClusterSize = NCCL_CONFIG_UNDEF_INT;
      ncclIntruQueueEnqueue(&comm->planner.collTaskQueue, &task);

      workStorage[i] = std::make_unique<unsigned char[]>(
          sizeof(ncclWorkList) + sizeof(ncclDevWorkColl));
      auto* workNode = reinterpret_cast<ncclWorkList*>(workStorage[i].get());
      memset(workNode, 0, sizeof(ncclWorkList) + sizeof(ncclDevWorkColl));
      workNode->workType = ncclDevWorkTypeColl;
      workNode->size = sizeof(ncclDevWorkColl);
      ncclIntruQueueEnqueue(&comm->planner.collWorkQueue, workNode);
    }

#ifdef ENABLE_WARP_SPEED
    g_rcclWarpSpeedSupportedPrefix =
        [](struct ncclComm*, struct ncclKernelPlan* candidatePlan,
           struct ncclTaskColl* head, int nCollTasks) {
          if (!ncclIntruQueueEmpty(&candidatePlan->p2pTaskQueue) ||
              !ncclIntruQueueEmpty(&candidatePlan->bcastTaskQueue)) {
            return false;
          }
          while (head != nullptr && nCollTasks != 0) {
            if (head->algorithm != NCCL_ALGO_RING || !head->useWarpSpeed) return false;
            head = head->next;
            if (nCollTasks > 0) --nCollTasks;
          }
          return nCollTasks <= 0;
        };
#endif
  }

  void AddP2pPair() {
    ncclComm* comm = bp.c();
    InstallSendRecvDevFuncIds();
    peers.resize(comm->nRanks);
    p2pSchedule.resize(comm->nRanks);
    channelPeers.resize(comm->nRanks);
    channelPeerSlots.resize(comm->nRanks);
    for (int rank = 0; rank < comm->nRanks; ++rank) {
      channelPeerSlots[rank] = &channelPeers[rank];
    }
    p2pSchedule[0] = {1, 1};
    comm->planner.peers = peers.data();
    comm->p2pSchedule = p2pSchedule.data();
    comm->p2pnChannels = 1;
    comm->p2pnChannelsPerPeer = 1;
    comm->p2pChunkSize = 128 << 10;
    comm->channels[0].peers = channelPeerSlots.data();

    sendTask.func = ncclFuncSend;
    sendTask.collAPI = ncclFuncSend;
    sendTask.buff = sendBuffer;
    sendTask.bytes = sizeof(sendBuffer);
    recvTask.func = ncclFuncRecv;
    recvTask.collAPI = ncclFuncRecv;
    recvTask.buff = recvBuffer;
    recvTask.bytes = sizeof(recvBuffer);
    // p2pTaskAppend's default for a plain send/recv (NCCL 2.32.6 bounds each P2P pair by its tasks' maxCTAs).
    sendTask.maxCTAs = recvTask.maxCTAs = MAXCHANNELS;
    ncclIntruQueueEnqueue(&peers[1].sendQueue, &sendTask);
    ncclIntruQueueEnqueue(&peers[1].recvQueue, &recvTask);
    comm->planner.nTasksP2p = 2;
    comm->planner.nTasksP2pSend = 1;
    comm->planner.nTasksP2pRecv = 1;
  }

  void ResetPlan() {
    bp.plan = std::make_unique<ncclKernelPlan>();
    memset(&bp.c()->planner.wipPlan, 0, sizeof(bp.c()->planner.wipPlan));
    bp.p()->persistent = planPersistent;
    bp.p()->workStorageType =
        planPersistent ? ncclDevWorkStorageTypePersistent : ncclDevWorkStorageTypeFifo;
    bp.p()->kernelSpecialized = true;
    budget.inArgsBytes = maxWorkBatches * sizeof(ncclDevWorkBatch);
    budget.outArgsBytes = 1 << 30;
  }

  void SetPersistent(bool persistent) {
    planPersistent = persistent;
    bp.p()->persistent = persistent;
    bp.p()->workStorageType =
        persistent ? ncclDevWorkStorageTypePersistent : ncclDevWorkStorageTypeFifo;
  }

  int ScheduleAndFinish() {
    EXPECT_EQ(ncclSuccess, scheduleCollTasksToPlan(bp.c(), bp.p(), &budget));
    finishPlan(bp.c(), bp.p());
    return PlanTaskCount();
  }

  int PlanTaskCount() {
    int count = 0;
    for (ncclTaskColl* task = ncclIntruQueueHead(&bp.p()->collTaskQueue);
         task != nullptr; task = task->next) {
      ++count;
    }
    return count;
  }

  int RemainingTaskCount() {
    int count = 0;
    for (ncclTaskColl* task = ncclIntruQueueHead(&bp.c()->planner.collTaskQueue);
         task != nullptr; task = task->next) {
      ++count;
    }
    return count;
  }

  int EmittedPlanCount() {
    int count = 0;
    for (ncclKernelPlan* plan = ncclIntruQueueHead(&bp.c()->planner.planQueue);
         plan != nullptr; plan = plan->next) {
      ++count;
    }
    return count;
  }

  ncclKernelPlan* EmittedPlan(int index) {
    ncclKernelPlan* plan = ncclIntruQueueHead(&bp.c()->planner.planQueue);
    while (plan != nullptr && index-- != 0) plan = plan->next;
    return plan;
  }

  void ExpectConservedInOrderFrom(int consumed, int accepted) {
    ASSERT_LE(consumed + accepted, static_cast<int>(tasks.size()));
    ncclTaskColl* task = ncclIntruQueueHead(&bp.p()->collTaskQueue);
    for (int i = consumed; i < consumed + accepted; ++i) {
      ASSERT_EQ(&tasks[i], task);
      task = task->next;
    }
    EXPECT_EQ(nullptr, task);

    task = ncclIntruQueueHead(&bp.c()->planner.collTaskQueue);
    for (size_t i = consumed + accepted; i < tasks.size(); ++i) {
      ASSERT_EQ(&tasks[i], task);
      task = task->next;
    }
    EXPECT_EQ(nullptr, task);
    EXPECT_EQ(static_cast<int>(tasks.size()) - consumed, PlanTaskCount() + RemainingTaskCount());
  }

  void ExpectConservedInOrder(int accepted) {
    ExpectConservedInOrderFrom(/*consumed=*/0, accepted);
  }

#ifdef ENABLE_WARP_SPEED
  void ExpectWarpSpeedMode(bool expected) {
    EXPECT_EQ(expected, bp.p()->warpSpeedPlan);
    EXPECT_EQ(expected, static_cast<bool>(bp.p()->kernelArgs->warpLevelComm));
  }
#endif
};
}  // namespace

TEST_F(EnqueueMicrotest, WarpSpeedPlanner_TopologyArchStorageIsBounded) {
  auto topo = std::make_unique<ncclTopoSystem>();
  auto& gpu = topo->nodes[GPU].nodes[0].gpu;
  std::memset(&gpu.arch, 0xa5, sizeof(gpu.arch));
  hipDeviceArch_t archBefore = gpu.arch;

  SetSingleGpuArch(topo.get(), "gfx950");
  EXPECT_STREQ("gfx950", gpu.gcn);
  EXPECT_EQ(0, std::memcmp(&archBefore, &gpu.arch, sizeof(gpu.arch)));

  WarpSpeedScheduleScene scene(
      {false, false, false, false},
      {NCCL_ALGO_RING, NCCL_ALGO_RING, NCCL_ALGO_RING, NCCL_ALGO_RING},
      /*maxWorkBatches=*/4, /*historicalMultistream=*/true);
  auto& sceneGpu = scene.topo->nodes[GPU].nodes[0].gpu;
  hipDeviceArch_t zeroArch{};
  EXPECT_STREQ("gfx950", sceneGpu.gcn);
  EXPECT_EQ(0, std::memcmp(&zeroArch, &sceneGpu.arch, sizeof(sceneGpu.arch)));
  EXPECT_STREQ("gfx950:sramecc+:xnack-", scene.bp.c()->archName);
  EXPECT_TRUE(IsArchMatch(scene.bp.c()->archName, "gfx950"));
}

#ifdef ENABLE_WARP_SPEED
TEST_F(EnqueueMicrotest, WarpSpeedPlanner_FourEligibleEmitThreeTaskWarpSpeedPlan) {
  WarpSpeedScheduleScene scene({true, true, true, true},
                               {NCCL_ALGO_RING, NCCL_ALGO_RING, NCCL_ALGO_RING, NCCL_ALGO_RING},
                               /*maxWorkBatches=*/4);
  EXPECT_EQ(3, scene.ScheduleAndFinish());
  scene.ExpectWarpSpeedMode(true);
  scene.ExpectConservedInOrder(/*accepted=*/3);
  EXPECT_FALSE(ncclPlanCanScheduleP2p(scene.bp.p()));
  ncclKernelPlan nextPlan{};
  EXPECT_TRUE(ncclPlanCanScheduleP2p(&nextPlan));
}

TEST_F(EnqueueMicrotest, WarpSpeedPlanner_LargeEligibleQueueEstimationIsLinear) {
  constexpr int kTasks = 768;
  WarpSpeedScheduleScene scene(std::vector<bool>(kTasks, true),
                               std::vector<int>(kTasks, NCCL_ALGO_RING),
                               /*maxWorkBatches=*/kTasks);
  g_rcclScheduleCollEstimateVisits = 0;
  int consumed = 0;
  int plans = 0;

  while (scene.RemainingTaskCount() != 0) {
    if (plans != 0) scene.ResetPlan();
    int const accepted = scene.ScheduleAndFinish();
    ASSERT_EQ(std::min(RCCL_WARP_SPEED_MAX_PLAN_COLLS, kTasks - consumed), accepted);
    EXPECT_EQ(&scene.tasks[consumed], ncclIntruQueueHead(&scene.bp.p()->collTaskQueue));
    scene.ExpectWarpSpeedMode(true);
    consumed += accepted;
    plans += 1;
  }

  EXPECT_EQ(kTasks, consumed);
  EXPECT_EQ(kTasks / RCCL_WARP_SPEED_MAX_PLAN_COLLS, plans);
  EXPECT_EQ(1023u, g_rcclScheduleCollEstimateVisits);
  EXPECT_LT(g_rcclScheduleCollEstimateVisits, 2u * kTasks);
}

TEST_F(EnqueueMicrotest, WarpSpeedPlanner_ThreeEligibleThenOrdinaryEmitSeparatePlans) {
  WarpSpeedScheduleScene scene({true, true, true, false},
                               {NCCL_ALGO_RING, NCCL_ALGO_RING, NCCL_ALGO_RING, NCCL_ALGO_RING},
                               /*maxWorkBatches=*/4);
  EXPECT_EQ(3, scene.ScheduleAndFinish());
  scene.ExpectWarpSpeedMode(true);
  scene.ExpectConservedInOrder(/*accepted=*/3);

  scene.ResetPlan();
  EXPECT_EQ(1, scene.ScheduleAndFinish());
  scene.ExpectWarpSpeedMode(false);
  EXPECT_EQ(&scene.tasks[3], ncclIntruQueueHead(&scene.bp.p()->collTaskQueue));
  EXPECT_EQ(nullptr, scene.tasks[3].next);
  EXPECT_EQ(0, scene.RemainingTaskCount());
}

TEST_F(EnqueueMicrotest, WarpSpeedPlanner_HistoricalOrdinaryMultistreamStillEmitsThreePlusOne) {
  WarpSpeedScheduleScene scene({false, false, false, false},
                               {NCCL_ALGO_RING, NCCL_ALGO_RING, NCCL_ALGO_RING, NCCL_ALGO_RING},
                               /*maxWorkBatches=*/4, /*historicalMultistream=*/true);
  EXPECT_EQ(3, scene.ScheduleAndFinish());
  scene.ExpectWarpSpeedMode(false);
  scene.ExpectConservedInOrder(/*accepted=*/3);
}

TEST_F(EnqueueMicrotest, WarpSpeedPlanner_SingleNodeGfx950CapsMixedPrefixes) {
  std::vector<std::vector<bool>> const historicalPatterns = {
      {false, true, true, true, true},
      {true, true, true, true, false},
      {true, false, true, false, true, true},
  };

  for (int eligibilityCase = 0; eligibilityCase < 3; ++eligibilityCase) {
    for (bool persistent : {false, true}) {
      for (int ranks : {2, 8}) {
        for (auto const& historical : historicalPatterns) {
          std::vector<bool> useWarpSpeed(historical.size(), eligibilityCase == 1);
          if (eligibilityCase == 2) {
            for (size_t i = 0; i < useWarpSpeed.size(); ++i) useWarpSpeed[i] = i % 2 == 0;
          }
          bool const expectWarpSpeedMode =
              std::all_of(useWarpSpeed.begin(), useWarpSpeed.end(), [](bool value) { return value; });
          WarpSpeedScheduleScene scene(
              useWarpSpeed, std::vector<int>(historical.size(), NCCL_ALGO_RING),
              /*maxWorkBatches=*/historical.size(), /*historicalMultistream=*/true);
          scene.bp.c()->nRanks = ranks;
          scene.SetPersistent(persistent);

          for (size_t i = 0; i < historical.size(); ++i) {
            if (historical[i]) continue;
            scene.tasks[i].func = ncclFuncReduce;
            scene.tasks[i].count = 1048575 + (i % 3);
            scene.tasks[i].datatype = i % 2 == 0 ? ncclFloat16 : ncclFloat;
            scene.tasks[i].opHost = ncclMax;
          }

          int consumed = 0;
          while (scene.RemainingTaskCount() != 0) {
            if (consumed != 0) scene.ResetPlan();
            int const accepted = scene.ScheduleAndFinish();
            EXPECT_EQ(std::min(RCCL_WARP_SPEED_MAX_PLAN_COLLS,
                               static_cast<int>(scene.tasks.size()) - consumed),
                      accepted);
            scene.ExpectWarpSpeedMode(expectWarpSpeedMode);
            scene.ExpectConservedInOrderFrom(consumed, accepted);
            consumed += accepted;
          }
          EXPECT_EQ(static_cast<int>(scene.tasks.size()), consumed);
        }
      }
    }
  }
}

TEST_F(EnqueueMicrotest, WarpSpeedPlanner_SingleNodeGfx950CapsSignatureNearMisses) {
  enum class Mutation { CountLow, ExactCount, CountHigh, OtherOp, OtherType };
  for (Mutation mutation : {Mutation::CountLow, Mutation::ExactCount, Mutation::CountHigh,
                            Mutation::OtherOp, Mutation::OtherType}) {
    WarpSpeedScheduleScene scene(
        {false, false, false, false},
        {NCCL_ALGO_RING, NCCL_ALGO_RING, NCCL_ALGO_RING, NCCL_ALGO_RING},
        /*maxWorkBatches=*/4, /*historicalMultistream=*/true);
    for (ncclTaskColl& task : scene.tasks) {
      if (mutation == Mutation::CountLow) task.count = 1048575;
      if (mutation == Mutation::CountHigh) task.count = 1048577;
      if (mutation == Mutation::OtherOp) task.opHost = ncclMax;
      if (mutation == Mutation::OtherType) task.datatype = ncclFloat16;
    }

    EXPECT_EQ(3, scene.ScheduleAndFinish());
    scene.ExpectWarpSpeedMode(false);
    scene.ExpectConservedInOrder(/*accepted=*/3);
  }
}

TEST_F(EnqueueMicrotest, WarpSpeedPlanner_BudgetTruncatedMixedTailKeepsOrdinaryMode) {
  WarpSpeedScheduleScene scene({true, true, false},
                               {NCCL_ALGO_RING, NCCL_ALGO_RING, NCCL_ALGO_RING},
                               /*maxWorkBatches=*/2);
  EXPECT_EQ(2, scene.ScheduleAndFinish());
  scene.ExpectWarpSpeedMode(false);
  scene.ExpectConservedInOrder(/*accepted=*/2);
}

TEST_F(EnqueueMicrotest, WarpSpeedPlanner_OppositeMixedOrderingStaysOrdinary) {
  WarpSpeedScheduleScene scene({false, true, true, true, true},
                               {NCCL_ALGO_RING, NCCL_ALGO_RING, NCCL_ALGO_RING,
                                NCCL_ALGO_RING, NCCL_ALGO_RING},
                               /*maxWorkBatches=*/5);
  EXPECT_EQ(5, scene.ScheduleAndFinish());
  scene.ExpectWarpSpeedMode(false);
  scene.ExpectConservedInOrder(/*accepted=*/5);
}

TEST_F(EnqueueMicrotest, WarpSpeedPlanner_TwoEligibleThenOrdinaryStayInOneOrdinaryPlan) {
  WarpSpeedScheduleScene scene({true, true, false},
                               {NCCL_ALGO_RING, NCCL_ALGO_RING, NCCL_ALGO_RING},
                               /*maxWorkBatches=*/3);
  EXPECT_EQ(3, scene.ScheduleAndFinish());
  scene.ExpectWarpSpeedMode(false);
  scene.ExpectConservedInOrder(/*accepted=*/3);
}

TEST_F(EnqueueMicrotest, WarpSpeedPlanner_TreeAndBelowThresholdTasksStayOrdinary) {
  WarpSpeedScheduleScene scene({false, false, false, false},
                               {NCCL_ALGO_TREE, NCCL_ALGO_RING, NCCL_ALGO_TREE, NCCL_ALGO_RING},
                               /*maxWorkBatches=*/4);
  EXPECT_EQ(4, scene.ScheduleAndFinish());
  scene.ExpectWarpSpeedMode(false);
  scene.ExpectConservedInOrder(/*accepted=*/4);
}

TEST_F(EnqueueMicrotest, WarpSpeedPlanner_MultiNodeOrdinary58TasksRemainOnePlan) {
  constexpr int kTasks = 58;
  WarpSpeedScheduleScene scene(
      std::vector<bool>(kTasks, false), std::vector<int>(kTasks, NCCL_ALGO_RING),
      /*maxWorkBatches=*/kTasks);
  ASSERT_EQ(2, scene.bp.c()->nNodes);
  g_rcclScheduleCollEstimateVisits = 0;

  EXPECT_EQ(kTasks, scene.ScheduleAndFinish());
  scene.ExpectWarpSpeedMode(false);
  scene.ExpectConservedInOrder(/*accepted=*/kTasks);
  EXPECT_EQ(kTasks, g_rcclScheduleCollEstimateVisits);
}

TEST_F(EnqueueMicrotest, WarpSpeedPlanner_DeferredP2pDoesNotChangeCollectiveMode) {
  WarpSpeedScheduleScene scene({true, true, true},
                               {NCCL_ALGO_RING, NCCL_ALGO_RING, NCCL_ALGO_RING},
                               /*maxWorkBatches=*/3);
  scene.bp.c()->planner.nTasksP2p = 1;
  EXPECT_EQ(3, scene.ScheduleAndFinish());
  scene.ExpectWarpSpeedMode(true);
  scene.ExpectConservedInOrder(/*accepted=*/3);
}

TEST_F(EnqueueMicrotest, WarpSpeedPlanner_AsymmetricP2pCountsEmitIdenticalCollectivePlans) {
  WarpSpeedScheduleScene rankWithoutP2p(
      {true, true, true, true},
      {NCCL_ALGO_RING, NCCL_ALGO_RING, NCCL_ALGO_RING, NCCL_ALGO_RING},
      /*maxWorkBatches=*/4);
  WarpSpeedScheduleScene rankWithP2p(
      {true, true, true, true},
      {NCCL_ALGO_RING, NCCL_ALGO_RING, NCCL_ALGO_RING, NCCL_ALGO_RING},
      /*maxWorkBatches=*/4);
  rankWithP2p.bp.c()->planner.nTasksP2p = 1;

  EXPECT_EQ(3, rankWithoutP2p.ScheduleAndFinish());
  EXPECT_EQ(3, rankWithP2p.ScheduleAndFinish());
  rankWithoutP2p.ExpectWarpSpeedMode(true);
  rankWithP2p.ExpectWarpSpeedMode(true);
  EXPECT_EQ(1, rankWithP2p.bp.c()->planner.nTasksP2p);
  EXPECT_EQ(rankWithoutP2p.bp.p()->channelMask.masks[0],
            rankWithP2p.bp.p()->channelMask.masks[0]);
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(rankWithoutP2p.tasks[i].channelLo, rankWithP2p.tasks[i].channelLo);
    EXPECT_EQ(rankWithoutP2p.tasks[i].channelHi, rankWithP2p.tasks[i].channelHi);
    EXPECT_EQ(rankWithoutP2p.tasks[i].nChannels, rankWithP2p.tasks[i].nChannels);
  }
  rankWithoutP2p.ExpectConservedInOrder(/*accepted=*/3);
  rankWithP2p.ExpectConservedInOrder(/*accepted=*/3);
}

TEST_F(EnqueueMicrotest, WarpSpeedPlanner_LaunchPrepareDefersAsymmetricP2pToNextPlan) {
  WarpSpeedScheduleScene rankWithoutP2p(
      {true, true, true},
      {NCCL_ALGO_RING, NCCL_ALGO_RING, NCCL_ALGO_RING},
      /*maxWorkBatches=*/3);
  WarpSpeedScheduleScene rankWithP2p(
      {true, true, true},
      {NCCL_ALGO_RING, NCCL_ALGO_RING, NCCL_ALGO_RING},
      /*maxWorkBatches=*/3);
  rankWithoutP2p.bp.c()->nNodes = 1;
  rankWithP2p.bp.c()->nNodes = 1;
  rankWithP2p.AddP2pPair();
  g_rcclLaunchPrepareStopAfterPlanning = true;

  ASSERT_EQ(ncclSuccess, ncclLaunchPrepare(rankWithoutP2p.bp.c()));
  ASSERT_EQ(ncclSuccess, ncclLaunchPrepare(rankWithP2p.bp.c()));
  ASSERT_EQ(1, rankWithoutP2p.EmittedPlanCount());
  ASSERT_EQ(2, rankWithP2p.EmittedPlanCount());

  ncclKernelPlan* noP2pCollPlan = rankWithoutP2p.EmittedPlan(0);
  ncclKernelPlan* withP2pCollPlan = rankWithP2p.EmittedPlan(0);
  ASSERT_NE(nullptr, noP2pCollPlan);
  ASSERT_NE(nullptr, withP2pCollPlan);
  EXPECT_TRUE(noP2pCollPlan->warpSpeedPlan);
  EXPECT_TRUE(withP2pCollPlan->warpSpeedPlan);
  EXPECT_TRUE(ncclIntruQueueEmpty(&noP2pCollPlan->p2pTaskQueue));
  EXPECT_TRUE(ncclIntruQueueEmpty(&withP2pCollPlan->p2pTaskQueue));
  EXPECT_EQ(noP2pCollPlan->channelMask.masks[0], withP2pCollPlan->channelMask.masks[0]);
  EXPECT_EQ(noP2pCollPlan->kernelArgs->warpLevelComm, withP2pCollPlan->kernelArgs->warpLevelComm);
  EXPECT_EQ(noP2pCollPlan->threadPerBlock, withP2pCollPlan->threadPerBlock);
  auto collCount = [](ncclKernelPlan* plan) {
    int count = 0;
    for (ncclTaskColl* task = ncclIntruQueueHead(&plan->collTaskQueue);
         task != nullptr; task = task->next) {
      ++count;
    }
    return count;
  };
  EXPECT_EQ(3, collCount(noP2pCollPlan));
  EXPECT_EQ(3, collCount(withP2pCollPlan));
  int noP2pChannels = 0;
  int withP2pChannels = 0;
  for (int i = 0; i < MAXCHANNELS / CHANNELS_PER_MASK_WORD; ++i) {
    noP2pChannels += countOneBits(noP2pCollPlan->channelMask.masks[i]);
    withP2pChannels += countOneBits(withP2pCollPlan->channelMask.masks[i]);
  }
  EXPECT_EQ(ncclPlanLaunchChannels(rankWithoutP2p.bp.c(), noP2pCollPlan, noP2pChannels),
            ncclPlanLaunchChannels(rankWithP2p.bp.c(), withP2pCollPlan, withP2pChannels));
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(rankWithoutP2p.tasks[i].channelLo, rankWithP2p.tasks[i].channelLo);
    EXPECT_EQ(rankWithoutP2p.tasks[i].channelHi, rankWithP2p.tasks[i].channelHi);
    EXPECT_EQ(rankWithoutP2p.tasks[i].nChannels, rankWithP2p.tasks[i].nChannels);
  }

  ncclKernelPlan* p2pPlan = rankWithP2p.EmittedPlan(1);
  ASSERT_NE(nullptr, p2pPlan);
  EXPECT_FALSE(p2pPlan->warpSpeedPlan);
  EXPECT_TRUE(ncclIntruQueueEmpty(&p2pPlan->collTaskQueue));
  EXPECT_EQ(&rankWithP2p.sendTask, ncclIntruQueueHead(&p2pPlan->p2pTaskQueue));
  EXPECT_EQ(&rankWithP2p.recvTask, rankWithP2p.sendTask.next);
  EXPECT_EQ(nullptr, rankWithP2p.recvTask.next);
  EXPECT_EQ(0, rankWithP2p.bp.c()->planner.nTasksP2p);
  EXPECT_EQ(0, rankWithP2p.bp.c()->planner.nTasksColl);
  EXPECT_TRUE(ncclIntruQueueEmpty(&rankWithP2p.peers[1].sendQueue));
  EXPECT_TRUE(ncclIntruQueueEmpty(&rankWithP2p.peers[1].recvQueue));
}

TEST_F(EnqueueMicrotest, WarpSpeedPlanner_LaunchGridUsesStoredPlanMode) {
  auto comm = std::make_unique<ncclComm>();
  auto plan = std::make_unique<ncclKernelPlan>();
  comm->WarpSize = 64;
  plan->threadPerBlock = 256;

  // The accepted task prefix may be WarpSpeed-eligible after a mixed tail was
  // budget-truncated, but launch geometry must keep the ordinary packing mode.
  g_rcclWarpSpeedSupported = true;
  plan->warpSpeedPlan = false;
  EXPECT_EQ(8, ncclPlanLaunchChannels(comm.get(), plan.get(), /*nChannels=*/8));

  plan->warpSpeedPlan = true;
  EXPECT_EQ(2, ncclPlanLaunchChannels(comm.get(), plan.get(), /*nChannels=*/8));
}
#endif

TEST_F(EnqueueMicrotest, WarpSpeedPlanner_MacroOffCoalescesOrdinary58Tasks) {
#ifdef ENABLE_WARP_SPEED
  GTEST_SKIP() << "Macro-off coverage runs in rccl-UnitTestsMicroEnqueue";
#else
  WarpSpeedScheduleScene scene(std::vector<bool>(58, false),
                               std::vector<int>(58, NCCL_ALGO_RING),
                               /*maxWorkBatches=*/58);
  EXPECT_EQ(58, scene.ScheduleAndFinish());
  scene.ExpectConservedInOrder(/*accepted=*/58);
#endif
}

TEST_F(EnqueueMicrotest, WarpSpeedPlanner_MacroOffDoesNotApplyHistoricalCap) {
#ifdef ENABLE_WARP_SPEED
  GTEST_SKIP() << "Macro-off coverage runs in rccl-UnitTestsMicroEnqueue";
#else
  WarpSpeedScheduleScene scene({false, false, false, false},
                               {NCCL_ALGO_RING, NCCL_ALGO_RING, NCCL_ALGO_RING, NCCL_ALGO_RING},
                               /*maxWorkBatches=*/4, /*historicalMultistream=*/true);
  EXPECT_EQ(4, scene.ScheduleAndFinish());
  scene.ExpectConservedInOrder(/*accepted=*/4);
#endif
}
