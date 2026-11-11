/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// ncclPrepareTasks, including the direct-AllGather costing it does for ncclGroupSimulateEnd.
// Included into enqueue-test.cc's translation unit; see README.md.

// ===========================================================================
// ncclGroupSimulateEnd estimatedTime: Direct AllGather is lowered to per-peer
// sends, so no collective task is costed. ncclPrepareTasks still costs it, and
// does not overwrite a non-negative estimate.
// ===========================================================================

namespace {
// Empty collective sorter, one send queue. ncclPrepareTasks reaches
// rcclSimulateDirectAllGather without scheduling a collective. The fast-path
// hook aborts unless a test installs it; ncclPrepareTasks calls it unconditionally.
struct DirectAllGatherScene {
  AlgoInfoComm cc;
  std::vector<ncclKernelPlanner::Peer> peers;
  ncclTaskP2p p2p{};
  bool algoNeedConnect[NCCL_NUM_ALGORITHMS] = {};
  bool needConnect = false;
  bool needGenkDevComm = false;

  DirectAllGatherScene() : peers(static_cast<size_t>(cc.get()->nRanks)) {
    ncclComm* comm = cc.get();
    comm->planner.peers = peers.data();
    comm->planner.capturingGraph = ncclCudaGraphNone(comm->config.graphUsageMode);
    comm->symmetricSupport = false;
    g_rcclSetCeAllReduceFastPath = [](struct ncclComm*) {};
  }

  void enqueue(ncclFunc_t collAPI, size_t count) {
    p2p = ncclTaskP2p{};
    p2p.collAPI = collAPI;
    p2p.count = count;
    p2p.datatype = ncclFloat32;
    ncclIntruQueueEnqueue(&peers[0].sendQueue, &p2p);
  }

  ncclResult_t run(ncclSimInfo_t* sim) {
    return ncclPrepareTasks(cc.get(), algoNeedConnect, &needConnect, &needGenkDevComm, sim);
  }
};
}  // namespace

TEST_F(EnqueueMicrotest, PrepareTasks_DirectAllGather_ReportsCostModelTime) {
  DirectAllGatherScene scene;
  scene.enqueue(ncclFuncAllGather, 1024);
  ScriptCellTime(NCCL_ALGO_TREE, NCCL_PROTO_LL, kChosenModelUs, kChosenModelUs);
  ncclSimInfo_t sim = NCCL_SIM_INFO_INITIALIZER;

  ASSERT_EQ(ncclSuccess, scene.run(&sim));
  EXPECT_FLOAT_EQ(kChosenModelUs, sim.estimatedTime)
      << "a direct AllGather never creates a collective task; the simulation still has to cost it";
}

TEST_F(EnqueueMicrotest, PrepareTasks_DirectAllGather_LeavesANonNegativeEstimate) {
  // A collective earlier in the group would already have stored a time, including
  // a literal 0. Direct AllGather must not overwrite either.
  for (float already : {0.0f, 7.0f}) {
    DirectAllGatherScene scene;
    scene.enqueue(ncclFuncAllGather, 1024);
    ScriptCellTime(NCCL_ALGO_TREE, NCCL_PROTO_LL, kChosenModelUs, kChosenModelUs);
    ncclSimInfo_t sim = NCCL_SIM_INFO_INITIALIZER;
    sim.estimatedTime = already;

    ASSERT_EQ(ncclSuccess, scene.run(&sim)) << already;
    EXPECT_FLOAT_EQ(already, sim.estimatedTime) << already;
  }
}

TEST_F(EnqueueMicrotest, PrepareTasks_NonAllGatherSend_LeavesEstimateUnset) {
  DirectAllGatherScene scene;
  scene.enqueue(ncclFuncReduceScatter, 1024);
  ScriptCellTime(NCCL_ALGO_TREE, NCCL_PROTO_LL, kChosenModelUs, kChosenModelUs);
  ncclSimInfo_t sim = NCCL_SIM_INFO_INITIALIZER;

  ASSERT_EQ(ncclSuccess, scene.run(&sim));
  EXPECT_FLOAT_EQ(NCCL_UNDEF_FLOAT, sim.estimatedTime);
}

TEST_F(EnqueueMicrotest, PrepareTasks_ZeroCountAllGather_LeavesEstimateUnset) {
  DirectAllGatherScene scene;
  scene.enqueue(ncclFuncAllGather, 0);
  ScriptCellTime(NCCL_ALGO_TREE, NCCL_PROTO_LL, kChosenModelUs, kChosenModelUs);
  ncclSimInfo_t sim = NCCL_SIM_INFO_INITIALIZER;

  ASSERT_EQ(ncclSuccess, scene.run(&sim));
  EXPECT_FLOAT_EQ(NCCL_UNDEF_FLOAT, sim.estimatedTime);
}

TEST_F(EnqueueMicrotest, PrepareTasks_NullSimInfo_Succeeds) {
  // A real launch passes a null simulation. The direct-AllGather costing must not run.
  DirectAllGatherScene scene;
  scene.enqueue(ncclFuncAllGather, 1024);
  ASSERT_EQ(ncclSuccess, scene.run(nullptr));
}
