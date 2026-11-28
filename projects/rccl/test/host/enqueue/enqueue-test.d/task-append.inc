/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Task append: collTaskAppend, ceCollTaskAppend, rawTaskAppend, ncclCeInitJob, rmaTaskAppend, capture checks.
// Included into enqueue-test.cc's translation unit; see README.md.

// ===========================================================================
// collTaskAppend -- symkExtract tag (enqueue.cc:3508-3511)
//
// collTaskAppend is static inside enqueue.cc and in scope here via the
// ENQUEUE_CC_PATH textual include. The two cases below pin that a
// non-symmetric decision writes RCCL_SYMK_EXTRACT_DENY and RCCL_SYMMETRIC
// writes RCCL_SYMK_EXTRACT_ALLOW. A third case pins decisionValid=false →
// RCCL_SYMK_EXTRACT_NONE. The consumer (ncclMakeSymmetricTaskList) is a
// fail-loud stub in test/host/fakes/sched_stubs.cc and is not reachable here;
// only the tag written onto the task is checked.
// ===========================================================================

static ncclInfo MakeCollInfo(ncclComm* comm, bool decisionValid, int algo) {
  ncclInfo info{};
  info.comm          = comm;
  info.coll          = ncclFuncAllReduce;
  info.op            = ncclSum;
  info.datatype      = ncclFloat32;
  info.count         = static_cast<size_t>(comm->nRanks);
  info.chunkSteps    = 1;
  info.sliceSteps    = 1;
  info.decisionValid = decisionValid;
  info.decision.algo = algo;
  return info;
}

TEST_F(EnqueueMicrotest, CollTaskAppend_NonSymmetricDecision_WritesDeny) {
  // decisionValid=true, algo != RCCL_SYMMETRIC → symkExtract must be DENY so
  // ncclMakeSymmetricTaskList does not extract the task even when buffers are
  // registered.
  CollTaskComm cc;
  ncclInfo info = MakeCollInfo(cc.get(), /*decisionValid=*/true, NCCL_ALGO_RING);
  ncclDevRedOpFull opDev{};
  ASSERT_EQ(ncclSuccess, collTaskAppend(cc.get(), &info, opDev));
  ASSERT_EQ(1, cc.get()->planner.nTasksColl);
  const ncclTaskColl* t = ncclTaskCollSorterDequeueAll(&cc.get()->planner.collSorter);
  ASSERT_NE(nullptr, t);
  EXPECT_EQ(RCCL_SYMK_EXTRACT_DENY, t->symkExtract);
}

TEST_F(EnqueueMicrotest, CollTaskAppend_SymmetricDecision_WritesAllow) {
  // decisionValid=true, algo == RCCL_SYMMETRIC → symkExtract must be ALLOW so
  // ncclMakeSymmetricTaskList can extract the task.
  CollTaskComm cc;
  ncclInfo info = MakeCollInfo(cc.get(), /*decisionValid=*/true, RCCL_SYMMETRIC);
  ncclDevRedOpFull opDev{};
  ASSERT_EQ(ncclSuccess, collTaskAppend(cc.get(), &info, opDev));
  ASSERT_EQ(1, cc.get()->planner.nTasksColl);
  const ncclTaskColl* t = ncclTaskCollSorterDequeueAll(&cc.get()->planner.collSorter);
  ASSERT_NE(nullptr, t);
  EXPECT_EQ(RCCL_SYMK_EXTRACT_ALLOW, t->symkExtract);
}

TEST_F(EnqueueMicrotest, CollTaskAppend_TdmWindowDecision_WritesAllowTdmOnly) {
  // A symmetric decision the AllGather TDM window admitted extracts only with a TDM kernel.
  CollTaskComm cc;
  ncclInfo info = MakeCollInfo(cc.get(), /*decisionValid=*/true, RCCL_SYMMETRIC);
  info.decision.symkTdmOnly = true;
  ncclDevRedOpFull opDev{};
  ASSERT_EQ(ncclSuccess, collTaskAppend(cc.get(), &info, opDev));
  const ncclTaskColl* t = ncclTaskCollSorterDequeueAll(&cc.get()->planner.collSorter);
  ASSERT_NE(nullptr, t);
  EXPECT_EQ(RCCL_SYMK_EXTRACT_ALLOW_TDM_ONLY, t->symkExtract);
}

TEST_F(EnqueueMicrotest, CollTaskAppend_NonSymmetricDecisionWithTdmOnly_WritesDeny) {
  // symkTdmOnly means nothing once another path won the decision.
  CollTaskComm cc;
  ncclInfo info = MakeCollInfo(cc.get(), /*decisionValid=*/true, NCCL_ALGO_RING);
  info.decision.symkTdmOnly = true;
  ncclDevRedOpFull opDev{};
  ASSERT_EQ(ncclSuccess, collTaskAppend(cc.get(), &info, opDev));
  const ncclTaskColl* t = ncclTaskCollSorterDequeueAll(&cc.get()->planner.collSorter);
  ASSERT_NE(nullptr, t);
  EXPECT_EQ(RCCL_SYMK_EXTRACT_DENY, t->symkExtract);
}

TEST_F(EnqueueMicrotest, CollTaskAppend_NoDecision_WritesNone) {
  // decisionValid=false → symkExtract must stay NONE so the extractor uses
  // window inspection rather than a stale/absent selector result.
  CollTaskComm cc;
  ncclInfo info = MakeCollInfo(cc.get(), /*decisionValid=*/false, NCCL_ALGO_RING);
  ncclDevRedOpFull opDev{};
  ASSERT_EQ(ncclSuccess, collTaskAppend(cc.get(), &info, opDev));
  ASSERT_EQ(1, cc.get()->planner.nTasksColl);
  const ncclTaskColl* t = ncclTaskCollSorterDequeueAll(&cc.get()->planner.collSorter);
  ASSERT_NE(nullptr, t);
  EXPECT_EQ(RCCL_SYMK_EXTRACT_NONE, t->symkExtract);
}

// ===========================================================================
// ceCollTaskAppend -- deferred CE staging setup (ROCM-32044)
// ===========================================================================

// Comm ready for ceCollTaskAppend: CE initialized, out of every group; leaves the thread's group heads clean.
struct CeAppendScene {
  CollTaskComm cc;
  CeAppendScene() {
    ClearGroupHeads();
    for (int type = 0; type < ncclGroupTaskTypeNum; ++type) {
      cc.get()->groupNext[type] = reinterpret_cast<ncclComm*>(NCCL_COMM_GROUP_INVALID);
    }
    cc.get()->ceColl.initialized = true;
  }
  ~CeAppendScene() { ClearGroupHeads(); }
  static void ClearGroupHeads() {
    for (int type = 0; type < ncclGroupTaskTypeNum; ++type) {
      ncclGroupCommHead[type] = nullptr;
    }
  }
  ncclResult_t Append(ncclFunc_t func) {
    ncclInfo info = MakeCollInfo(cc.get(), /*decisionValid=*/true, RCCL_CE_REGISTERED);
    info.coll = func;
    return ceCollTaskAppend(cc.get(), &info, nullptr, nullptr, nullptr, nullptr, ncclDevRedOpFull{});
  }
  bool JoinedSymRegister() { return ncclGroupCommHead[ncclGroupTaskTypeSymRegister] == cc.get(); }
};

TEST_F(EnqueueMicrotest, CeCollTaskAppend_AllReduceWithoutStaging_MarksPendingAndJoinsSymRegister) {
  CeAppendScene scene;
  ASSERT_EQ(ncclSuccess, scene.Append(ncclFuncAllReduce));
  EXPECT_TRUE(scene.cc.get()->ceColl.stagingPending);
  EXPECT_TRUE(scene.JoinedSymRegister());
  EXPECT_FALSE(ncclIntruQueueEmpty(&scene.cc.get()->planner.collCeTaskQueue));
}

TEST_F(EnqueueMicrotest, CeCollTaskAppend_AllReduceAlreadyPending_StillJoinsSymRegister) {
  CeAppendScene scene;
  scene.cc.get()->ceColl.stagingPending = true;  // left by a group that failed before its SymRegister job ran
  ASSERT_EQ(ncclSuccess, scene.Append(ncclFuncAllReduce));
  EXPECT_TRUE(scene.cc.get()->ceColl.stagingPending);
  EXPECT_TRUE(scene.JoinedSymRegister());
}

TEST_F(EnqueueMicrotest, CeCollTaskAppend_AllReduceWithStaging_QueuesNoSetup) {
  CeAppendScene scene;
  uint8_t staging = 0;
  scene.cc.get()->ceColl.ceARTmpBuf = &staging;
  ASSERT_EQ(ncclSuccess, scene.Append(ncclFuncAllReduce));
  EXPECT_FALSE(scene.cc.get()->ceColl.stagingPending);
  EXPECT_FALSE(scene.JoinedSymRegister());
}

TEST_F(EnqueueMicrotest, CeCollTaskAppend_AllGather_QueuesNoSetup) {
  CeAppendScene scene;
  ASSERT_EQ(ncclSuccess, scene.Append(ncclFuncAllGather));
  EXPECT_FALSE(scene.cc.get()->ceColl.stagingPending);
  EXPECT_FALSE(scene.JoinedSymRegister());
}

// ===========================================================================
// rawTaskAppend / ncclCeInitJob (the NCCL_ENQUEUE_REARCH_ENABLE=1 path), then argument validation arms.
// ===========================================================================

namespace {
const hipStream_t kRawStream = reinterpret_cast<hipStream_t>(0x5100);

struct RawTaskScene : CeAppendScene {
  RawTaskScene() {
    cc.get()->config.numRmaCtx = 2;
    cc.get()->config.numRmaSig = 3;
  }
  ncclInfo Info(ncclFunc_t func) {
    ncclInfo info = MakeCollInfo(cc.get(), /*decisionValid=*/false, NCCL_ALGO_RING);
    info.coll = func;
    info.stream = kRawStream;
    return info;
  }
  ncclResult_t Append(ncclInfo& info, std::string* log = nullptr) {
    ncclResult_t rc = ncclSuccess;
    std::string captured = RcclUnitTesting::CaptureLog([&] { rc = rawTaskAppend(cc.get(), &info); });
    if (log != nullptr) {
      *log = captured;
    }
    return rc;
  }
  ncclRawTask* Generic() { return ncclIntruQueueHead(&cc.get()->rawTaskQueue.genericQueue); }
  ncclRawTask* Bcast() { return ncclIntruQueueHead(&cc.get()->rawTaskQueue.bcastQueue); }
  bool JoinedRawTaskGroup() { return ncclGroupCommHead[ncclGroupTaskTypeRawTask] == cc.get(); }
};
}  // namespace

TEST_F(EnqueueMicrotest, RawTaskAppend_SendAndRecv_CopyTheP2pArgsAndJoinTheRawTaskGroup) {
  for (ncclFunc_t func : {ncclFuncSend, ncclFuncRecv}) {
    RawTaskScene sc;
    double buf = 0;
    ncclInfo info = sc.Info(func);
    info.recvbuff = &buf;
    info.count = 10;
    info.datatype = ncclFloat64;
    info.root = 3;
    ASSERT_EQ(ncclSuccess, sc.Append(info));
    ncclRawTask* t = sc.Generic();
    ASSERT_NE(nullptr, t);
    EXPECT_EQ(ncclTaskKindSendRecv, t->kind);
    EXPECT_EQ(func, t->sendRecv.func);
    EXPECT_EQ(func, t->sendRecv.collAPI);
    EXPECT_EQ(&buf, t->sendRecv.buff);
    EXPECT_EQ(10u, t->sendRecv.count);
    EXPECT_EQ(ncclFloat64, t->sendRecv.datatype);
    EXPECT_EQ(3, t->sendRecv.peer);
    EXPECT_EQ(80u, t->sendRecv.bytes);
    EXPECT_EQ(kRawStream, t->sendRecv.stream);
    EXPECT_TRUE(sc.JoinedRawTaskGroup());
  }
}

TEST_F(EnqueueMicrotest, RawTaskAppend_PutSignalAndSignal_CopyEveryArgument) {
  RawTaskScene sc;
  int src = 0;
  ncclInfo info = sc.Info(ncclFuncPutSignal);
  info.sendbuff = &src;
  info.count = 7;
  info.datatype = ncclInt32;
  info.root = 2;
  info.peerWin = reinterpret_cast<ncclWindow_t>(0x7700);
  info.peerWinOffset = 48;
  info.sigIdx = 2;
  info.ctx = 1;
  info.flags = 5;
  ASSERT_EQ(ncclSuccess, sc.Append(info));
  ncclRawTask* t = sc.Generic();
  ASSERT_NE(nullptr, t);
  EXPECT_EQ(ncclTaskKindRma, t->kind);
  EXPECT_EQ(ncclFuncPutSignal, t->rma.func);
  const ncclRawTaskPutSignal& put = t->rma.rmaOp.putSignal;
  EXPECT_EQ(&src, put.localbuff);
  EXPECT_EQ(7u, put.count);
  EXPECT_EQ(ncclInt32, put.datatype);
  EXPECT_EQ(2, put.peer);
  EXPECT_EQ(info.peerWin, put.peerWin);
  EXPECT_EQ(48u, put.peerWinOffset);
  EXPECT_EQ(2, put.sigIdx);
  EXPECT_EQ(1, put.ctx);
  EXPECT_EQ(5u, put.flags);
  EXPECT_EQ(kRawStream, put.stream);
  EXPECT_TRUE(sc.JoinedRawTaskGroup());
  RawTaskScene sigScene;
  ncclInfo sigInfo = sigScene.Info(ncclFuncSignal);
  sigInfo.count = 0;
  sigInfo.root = 3;
  sigInfo.sigIdx = 1;
  sigInfo.ctx = 1;
  sigInfo.flags = 6;
  ASSERT_EQ(ncclSuccess, sigScene.Append(sigInfo));
  ASSERT_NE(nullptr, sigScene.Generic());
  EXPECT_EQ(ncclFuncSignal, sigScene.Generic()->rma.func);
  const ncclRawTaskSignal& sig = sigScene.Generic()->rma.rmaOp.signal;
  EXPECT_EQ(3, sig.peer);
  EXPECT_EQ(1, sig.sigIdx);
  EXPECT_EQ(1, sig.ctx);
  EXPECT_EQ(6u, sig.flags);
  EXPECT_EQ(kRawStream, sig.stream);
}

TEST_F(EnqueueMicrotest, RawTaskAppend_BadRmaContextOrSignalCount_IsRejectedBeforeQueueing) {
  struct {
    ncclFunc_t func;
    int ctx;
    size_t count;
    std::string warn;
  } const cases[] = {{ncclFuncPutSignal, -1, 0, "invalid context -1 (must be in [0, 2))"},
                     {ncclFuncPutSignal, 2, 0, "invalid context 2 (must be in [0, 2))"},
                     {ncclFuncSignal, 2, 0, "invalid context 2 (must be in [0, 2))"},
                     {ncclFuncSignal, 0, 1, "ncclSignal: count must be 0"}};
  for (const auto& c : cases) {
    RawTaskScene sc;
    ncclInfo info = sc.Info(c.func);
    info.ctx = c.ctx;
    info.count = c.count;
    std::string log;
    EXPECT_EQ(ncclInvalidArgument, sc.Append(info, &log)) << c.warn;
    EXPECT_TRUE(RcclUnitTesting::LogHas(log, c.warn.c_str())) << log;
    EXPECT_EQ(nullptr, sc.Generic());
    EXPECT_FALSE(sc.JoinedRawTaskGroup());
  }
}

TEST_F(EnqueueMicrotest, RawTaskAppend_WaitSignal_DeepCopiesTheDescriptors) {
  RawTaskScene sc;
  ncclWaitSignalDesc_t descs[2] = {{/*opCnt=*/1, /*peer=*/3, /*sigIdx=*/2, /*ctx=*/1}, {4, 0, 0, 0}};
  ncclInfo info = sc.Info(ncclFuncWaitSignal);
  info.nDesc = 2;
  info.signalDescs = descs;
  ASSERT_EQ(ncclSuccess, sc.Append(info));
  ncclRawTask* t = sc.Generic();
  ASSERT_NE(nullptr, t);
  EXPECT_EQ(ncclTaskKindRma, t->kind);
  EXPECT_EQ(ncclFuncWaitSignal, t->rma.func);
  const ncclRawTaskWaitSignal& wait = t->rma.rmaOp.waitSignal;
  EXPECT_EQ(2, wait.nDesc);
  EXPECT_EQ(kRawStream, wait.stream);
  ASSERT_NE(descs, wait.signalDescs);
  EXPECT_EQ(0, std::memcmp(descs, wait.signalDescs, sizeof(descs)));
  descs[1].opCnt = 9;
  EXPECT_EQ(4, wait.signalDescs[1].opCnt);
  EXPECT_TRUE(sc.JoinedRawTaskGroup());
}

namespace {
struct BadWaitSignal {
  int nDesc;
  bool nullDescs;
  ncclWaitSignalDesc_t second;
  const char* warn;
};

// The bad descriptor sits second, behind a valid one, so the per-descriptor loop must reach it.
const BadWaitSignal kBadWaitSignals[] = {
    {0, false, {1, 0, 0, 0}, "ncclWaitSignal: invalid arguments"},
    {-1, false, {1, 0, 0, 0}, "ncclWaitSignal: invalid arguments"},
    {2, true, {1, 0, 0, 0}, "ncclWaitSignal: invalid arguments"},
    {2, false, {0, 0, 0, 0}, "descriptor 1 has invalid opCnt 0"},
    {2, false, {1, 0, 3, 0}, "descriptor 1 has invalid sigIdx 3 (must be in [0, 3))"},
    {2, false, {1, 0, -1, 0}, "descriptor 1 has invalid sigIdx -1"},
    {2, false, {1, 0, 0, 2}, "descriptor 1 has invalid context 2 (must be in [0, 2))"},
    {2, false, {1, 0, 0, -1}, "descriptor 1 has invalid context -1"},
    {2, false, {1, 4, 0, 0}, "descriptor 1 has invalid peer 4 (must be in [0, 4))"},
    {2, false, {1, -1, 0, 0}, "descriptor 1 has invalid peer -1"},
};

void PointAtBadWaitSignal(const BadWaitSignal& c, ncclWaitSignalDesc_t (&descs)[2], ncclInfo* info) {
  descs[0] = {1, 1, 1, 1};
  descs[1] = c.second;
  info->coll = ncclFuncWaitSignal;
  info->nDesc = c.nDesc;
  info->signalDescs = c.nullDescs ? nullptr : descs;
}
}  // namespace

TEST_F(EnqueueMicrotest, RawTaskAppend_WaitSignalWithABadDescriptor_IsRejected) {
  for (const BadWaitSignal& c : kBadWaitSignals) {
    RawTaskScene sc;
    ncclWaitSignalDesc_t descs[2];
    ncclInfo info = sc.Info(ncclFuncWaitSignal);
    PointAtBadWaitSignal(c, descs, &info);
    std::string log;
    EXPECT_EQ(ncclInvalidArgument, sc.Append(info, &log)) << c.warn;
    EXPECT_TRUE(RcclUnitTesting::LogHas(log, c.warn)) << log;
    EXPECT_EQ(nullptr, sc.Generic()) << c.warn;
  }
}

TEST_F(EnqueueMicrotest, RawTaskAppend_Collective_CopiesItsArgumentsIntoTheGenericQueue) {
  RawTaskScene sc;
  float send = 0;
  float recv = 0;
  float bias = 0;
  ncclInfo info = sc.Info(ncclFuncAllReduce);
  info.sendbuff = &send;
  info.recvbuff = &recv;
  info.acc = &bias;
  info.count = 12;
  info.root = 2;
  info.op = ncclProd;
  sc.cc.get()->opCount = 7;
  ASSERT_EQ(ncclSuccess, sc.Append(info));
  ncclRawTask* t = sc.Generic();
  ASSERT_NE(nullptr, t);
  EXPECT_EQ(ncclTaskKindColl, t->kind);
  EXPECT_EQ(ncclFuncAllReduce, t->coll.func);
  EXPECT_EQ(&send, t->coll.sendbuff);
  EXPECT_EQ(&recv, t->coll.recvbuff);
  EXPECT_EQ(12u, t->coll.count);
  EXPECT_EQ(2, t->coll.root);
  EXPECT_EQ(ncclFloat32, t->coll.datatype);
  EXPECT_EQ(ncclProd, t->coll.opHost);
  EXPECT_EQ(ncclDevProd, t->coll.opDev.op);
  EXPECT_EQ(kRawStream, t->coll.stream);
  EXPECT_EQ(&bias, t->coll.acc);
  EXPECT_EQ(7u, t->coll.opCount);
  EXPECT_TRUE(sc.JoinedRawTaskGroup());
}

TEST_F(EnqueueMicrotest, RawTaskAppend_BroadcastGoesToTheBcastQueueAndAnEmptyCollectiveIsDropped) {
  RawTaskScene sc;
  ncclInfo info = sc.Info(ncclFuncBroadcast);
  info.root = 1;
  ASSERT_EQ(ncclSuccess, sc.Append(info));
  EXPECT_EQ(nullptr, sc.Generic());
  ASSERT_NE(nullptr, sc.Bcast());
  EXPECT_EQ(1, sc.Bcast()->coll.root);

  RawTaskScene empty;
  ncclInfo none = empty.Info(ncclFuncAllReduce);
  none.count = 0;
  ASSERT_EQ(ncclSuccess, empty.Append(none));
  EXPECT_EQ(nullptr, empty.Generic());
  EXPECT_FALSE(empty.JoinedRawTaskGroup());
}

TEST_F(EnqueueMicrotest, RawTaskAppend_SingleRankCollective_LaunchesInPlaceOfQueueing) {
  RawTaskScene sc;
  sc.cc.get()->nRanks = 1;
  float send = 0;
  float recv = 0;
  float bias = 0;
  ncclInfo info = sc.Info(ncclFuncAllReduce);
  info.sendbuff = &send;
  info.recvbuff = &recv;
  info.acc = &bias;
  info.count = 5;
  ScopedHook launch(g_ncclLaunchOneRank, [&](void* dst, void const* src, size_t n, ncclDevRedOpFull op,
                                             ncclDataType_t, hipStream_t stream, hipEvent_t, void const* acc) {
    EXPECT_EQ(&recv, dst);
    EXPECT_EQ(&send, src);
    EXPECT_EQ(5u, n);
    EXPECT_EQ(ncclDevSum, op.op);
    EXPECT_EQ(kRawStream, stream);
    EXPECT_EQ(&bias, acc);
    return ncclSuccess;
  });
  ASSERT_EQ(ncclSuccess, sc.Append(info));
  EXPECT_EQ(1, launch.calls);
  EXPECT_EQ(nullptr, sc.Generic());
  EXPECT_FALSE(sc.JoinedRawTaskGroup());
}

TEST_F(EnqueueMicrotest, RawTaskAppend_Fp8ReductionBelowSm90_IsRejectedButFp8CopiesAreNot) {
  struct {
    ncclFunc_t func;
    int minCompCap;
    bool rejected;
  } const cases[] = {{ncclFuncAllReduce, 89, true}, {ncclFuncReduce, 89, true}, {ncclFuncReduceScatter, 89, true},
                     {ncclFuncAllGather, 89, false}, {ncclFuncAllReduce, 90, false}};
  for (ncclDataType_t fp8 : {ncclFloat8e4m3, ncclFloat8e5m2}) {
    for (const auto& c : cases) {
      RawTaskScene sc;
      sc.cc.get()->minCompCap = c.minCompCap;
      ncclInfo info = sc.Info(c.func);
      info.datatype = fp8;
      std::string log;
      EXPECT_EQ(c.rejected ? ncclInvalidArgument : ncclSuccess, sc.Append(info, &log)) << c.func << " " << fp8;
      EXPECT_EQ(c.rejected, RcclUnitTesting::LogHas(log, "FP8 reduction support begins with sm90")) << log;
      EXPECT_EQ(c.rejected, sc.Generic() == nullptr) << c.func << " " << fp8;
    }
  }
}

TEST_F(EnqueueMicrotest, TaskAppend_RearchEnabled_RoutesSendThroughTheRawTaskQueue) {
  SetParam("ENQUEUE_REARCH_ENABLE", 1);
  RawTaskScene sc;
  ncclInfo info = sc.Info(ncclFuncSend);
  info.count = 4;
  info.root = 1;
  ASSERT_EQ(ncclSuccess, taskAppend(sc.cc.get(), &info));
  ASSERT_NE(nullptr, sc.Generic());
  EXPECT_EQ(ncclTaskKindSendRecv, sc.Generic()->kind);
  EXPECT_EQ(1, sc.Generic()->sendRecv.peer);
  EXPECT_EQ(0, sc.cc.get()->planner.nTasksP2p);
}

namespace {
// CE not yet initialized and the rearch path on, so ceCollTaskAppend hands its init job to the management queue.
// The hook mirrors ncclMgmtTaskEnqueue's ownership: it keeps the job on success and frees it on failure.
struct RearchCeInitScene : CeAppendScene {
  ncclAsyncJob* job = nullptr;
  ncclResult_t (*jobFn)(ncclAsyncJob*) = nullptr;
  void (*jobFree)(void*) = nullptr;
  ncclComm* jobComm = nullptr;
  ncclResult_t enqueueResult = ncclSuccess;
  ScopedHook<ncclResult_t(ncclAsyncJob*, ncclResult_t (*)(ncclAsyncJob*), void (*)(void*), ncclComm*)> enqueue{
      g_ncclMgmtTaskEnqueue,
      [this](ncclAsyncJob* j, ncclResult_t (*fn)(ncclAsyncJob*), void (*d)(void*), ncclComm* comm) {
        job = j;
        jobFn = fn;
        jobFree = d;
        jobComm = comm;
        if (enqueueResult != ncclSuccess && d != nullptr) {
          d(j);
        }
        return enqueueResult;
      }};
  RearchCeInitScene() {
    SetParam("ENQUEUE_REARCH_ENABLE", 1);
    cc.get()->ceColl.initialized = false;
    cc.get()->cudaDev = 5;
  }
  ~RearchCeInitScene() {
    if (jobFree != nullptr && enqueueResult == ncclSuccess) {
      jobFree(job);
    }
  }
};
}  // namespace

TEST_F(EnqueueMicrotest, CeCollTaskAppend_RearchUninitialized_EnqueuesTheCeInitJobOnce) {
  RearchCeInitScene scene;
  ASSERT_EQ(ncclSuccess, scene.Append(ncclFuncAllGather));
  EXPECT_EQ(1, scene.enqueue.calls);
  EXPECT_EQ(scene.cc.get(), scene.jobComm);
  ASSERT_NE(nullptr, scene.job);
  EXPECT_TRUE(scene.cc.get()->ceColl.initialized);
  EXPECT_TRUE(ncclIntruQueueEmpty(&scene.cc.get()->ceInitTaskQueue));
  EXPECT_FALSE(ncclIntruQueueEmpty(&scene.cc.get()->planner.collCeTaskQueue));

  ASSERT_EQ(ncclSuccess, scene.Append(ncclFuncAllGather));
  EXPECT_EQ(1, scene.enqueue.calls) << "an initialized CE must not enqueue a second init job";
}

TEST_F(EnqueueMicrotest, CeInitJob_SetsTheDeviceThenInitializesCeAndClearsInitializedOnFailure) {
  struct {
    hipError_t setDevice;
    ncclResult_t ceInit;
    ncclResult_t rc;
    int ceInitCalls;
  } const cases[] = {{hipSuccess, ncclSuccess, ncclSuccess, 1},
                     {hipSuccess, ncclSystemError, ncclSystemError, 1},
                     {hipErrorInvalidDevice, ncclSuccess, ncclUnhandledCudaError, 0}};
  for (const auto& c : cases) {
    RearchCeInitScene scene;
    ASSERT_EQ(ncclSuccess, scene.Append(ncclFuncAllGather));
    int device = -1;
    ScopedHook setDevice(g_hipSetDevice, [&](int dev) {
      device = dev;
      return c.setDevice;
    });
    ScopedHook ceInit(g_ncclCeInit, [&](ncclComm* comm) {
      EXPECT_EQ(5, device) << "the device is set before CE init";
      EXPECT_EQ(scene.cc.get(), comm);
      return c.ceInit;
    });
    EXPECT_EQ(c.rc, scene.jobFn(scene.job));
    EXPECT_EQ(c.ceInitCalls, ceInit.calls);
    EXPECT_EQ(c.rc == ncclSuccess, scene.cc.get()->ceColl.initialized);
  }
}

TEST_F(EnqueueMicrotest, CeCollTaskAppend_RearchEnqueueFailure_LeavesCeUninitializedAndQueuesNoTask) {
  RearchCeInitScene scene;
  scene.enqueueResult = ncclInternalError;
  EXPECT_EQ(ncclInternalError, scene.Append(ncclFuncAllGather));
  EXPECT_EQ(1, scene.enqueue.calls);
  EXPECT_FALSE(scene.cc.get()->ceColl.initialized);
  EXPECT_TRUE(ncclIntruQueueEmpty(&scene.cc.get()->planner.collCeTaskQueue));
}

namespace {
constexpr int kRmaRegistered = 0x100;
ncclWindow_t const kPeerWinHandle = reinterpret_cast<ncclWindow_t>(0x9900);

// A PutSignal that passes every check; each case breaks exactly one input.
struct RmaArgsScene : RawTaskScene {
  ncclDevrWindow peerWin{};
  ncclDevrWindow srcWin{};
  ncclWindow_vidmem peerVidmem{};
  char srcBuf[64]{};
  ScopedHook<ncclResult_t(ncclShadowPool*, void*, void**)> toHost{
      g_shadowPoolToHost, [this](ncclShadowPool*, void* dev, void** host) {
        *host = dev == kPeerWinHandle ? &peerVidmem : nullptr;
        return ncclSuccess;
      }};
  ScopedHook<bool(int, ncclDevrRegisterCapability)> regEnabled{
      g_devrWinRegEnabled,
      [](int flags, ncclDevrRegisterCapability cap) { return cap == ncclDevrRegisterRma && (flags & kRmaRegistered); }};
  ScopedHook<ncclResult_t(ncclComm*, void const*, ncclDevrWindow**)> findWindow{
      g_devrFindWindow, [this](ncclComm*, void const* ptr, ncclDevrWindow** win) {
        *win = (ptr >= srcBuf && ptr < srcBuf + sizeof(srcBuf)) ? &srcWin : nullptr;
        return ncclSuccess;
      }};
  std::vector<ncclIntruQueue<ncclTaskRma, &ncclTaskRma::next>> rmaQueues{2};
  ncclInfo info;
  RmaArgsScene() : info(Info(ncclFuncPutSignal)) {
    cc.get()->hostRmaSupport = true;
    cc.get()->planner.rmaTaskQueues = rmaQueues.data();
    for (auto& queue : rmaQueues) {
      ncclIntruQueueConstruct(&queue);
    }
    peerVidmem.winHost = &peerWin;
    peerWin.size = 4096;
    peerWin.winFlags = kRmaRegistered;
    srcWin.userPtr = srcBuf;
    srcWin.winFlags = kRmaRegistered | NCCL_WIN_COLL_SYMMETRIC;
    info.peerWin = kPeerWinHandle;
    info.sendbuff = srcBuf + 8;
    info.count = 0;
  }
  ncclResult_t Run(std::string* log) {
    ncclResult_t rc = ncclSuccess;
    *log = RcclUnitTesting::CaptureLog([&] { rc = rmaTaskAppend(cc.get(), &info); });
    return rc;
  }
};

struct BadRmaCase {
  const char* warn;
  std::function<void(RmaArgsScene&)> breakIt;
  ncclResult_t rc = ncclInvalidArgument;
  int findWindowCalls = -1;
};
}  // namespace

TEST_F(EnqueueMicrotest, RmaTaskAppend_ValidPutSignal_QueuesOneTaskOnItsContext) {
  for (int symmetric : {0, 1}) {
    RmaArgsScene sc;
    sc.cc.get()->symmetricSupport = symmetric;
    sc.info.count = 4;
    sc.info.root = 2;
    sc.info.ctx = 1;
    sc.info.sigIdx = 2;
    sc.info.peerWinOffset = 64;
    std::string log;
    ASSERT_EQ(ncclSuccess, sc.Run(&log)) << log;
    EXPECT_TRUE(ncclIntruQueueEmpty(&sc.rmaQueues[0]));
    ncclTaskRma* t = ncclIntruQueueHead(&sc.rmaQueues[1]);
    ASSERT_NE(nullptr, t);
    EXPECT_EQ(nullptr, t->next);
    EXPECT_EQ(ncclFuncPutSignal, t->func);
    EXPECT_EQ(static_cast<const void*>(sc.srcBuf + 8), t->srcBuff);
    EXPECT_EQ(8u, t->srcWinOffset);
    EXPECT_EQ(&sc.srcWin, t->srcWinHost);
    EXPECT_EQ(4u, t->count);
    EXPECT_EQ(ncclFloat32, t->datatype);
    EXPECT_EQ(16u, t->bytes);
    EXPECT_EQ(1, t->ctx);
    EXPECT_EQ(2, t->signalIdx);
    EXPECT_EQ(2, t->peer);
    EXPECT_EQ(64u, t->peerWinOffset);
    EXPECT_EQ(&sc.peerWin, t->peerWinHost);
    EXPECT_EQ(NCCL_SIGNAL, t->signalMode);
    EXPECT_EQ(1, sc.cc.get()->planner.nTasksRma);
    EXPECT_EQ(sc.cc.get(), ncclGroupCommHead[ncclGroupTaskTypeCollective]);
  }
}

TEST_F(EnqueueMicrotest, RmaTaskAppend_ValidWaitSignal_QueuesOneTaskPerContext) {
  RmaArgsScene sc;
  ScopedHook rmaReady(g_rmaInitialized, [](ncclComm*) { return true; });
  ncclWaitSignalDesc_t descs[3] = {{/*opCnt=*/2, /*peer=*/1, /*sigIdx=*/0, /*ctx=*/1},
                                   {5, 3, 2, 0},
                                   {7, 2, 1, 1}};
  sc.info.coll = ncclFuncWaitSignal;
  sc.info.nDesc = 3;
  sc.info.signalDescs = descs;
  std::string log;
  ASSERT_EQ(ncclSuccess, sc.Run(&log)) << log;
  EXPECT_EQ(2, sc.cc.get()->planner.nTasksRma);
  struct {
    int ctx;
    std::vector<int> peers;
    std::vector<int> nsignals;
    std::vector<int> signalIdxs;
  } const want[] = {{0, {3}, {5}, {2}}, {1, {1, 2}, {2, 7}, {0, 1}}};
  for (const auto& w : want) {
    ncclTaskRma* t = ncclIntruQueueHead(&sc.rmaQueues[w.ctx]);
    ASSERT_NE(nullptr, t) << w.ctx;
    EXPECT_EQ(nullptr, t->next) << w.ctx;
    EXPECT_EQ(ncclFuncWaitSignal, t->func);
    EXPECT_EQ(w.ctx, t->ctx);
    EXPECT_EQ(NCCL_SIGNAL, t->signalMode);
    ASSERT_EQ(static_cast<int>(w.peers.size()), t->npeers) << w.ctx;
    EXPECT_EQ(w.peers, std::vector<int>(t->peers, t->peers + t->npeers)) << w.ctx;
    EXPECT_EQ(w.nsignals, std::vector<int>(t->nsignals, t->nsignals + t->npeers)) << w.ctx;
    EXPECT_EQ(w.signalIdxs, std::vector<int>(t->signalIdxs, t->signalIdxs + t->npeers)) << w.ctx;
  }
}

TEST_F(EnqueueMicrotest, RmaTaskAppend_EachInvalidArgument_IsRejectedWithItsOwnWarning) {
  const char* const kNeedsRma = "ncclPutSignal requires a window registered for RMA";
  const BadRmaCase cases[] = {
      {"host RMA is not supported", [](RmaArgsScene& s) { s.cc.get()->hostRmaSupport = false; }},
      {"Context 2 is invalid (must be in [0, 2))", [](RmaArgsScene& s) { s.info.ctx = 2; }},
      {"Context -1 is invalid", [](RmaArgsScene& s) { s.info.ctx = -1; }},
      {"Signal index 3 is invalid (must be in [0, 3))", [](RmaArgsScene& s) { s.info.sigIdx = 3; }},
      {"Signal index -1 is invalid", [](RmaArgsScene& s) { s.info.sigIdx = -1; }},
      {"Flags 1 is invalid (must be 0)", [](RmaArgsScene& s) { s.info.flags = 1; }},
      {"ncclPutSignal: peerWin is NULL", [](RmaArgsScene& s) { s.info.peerWin = nullptr; }},
      {kNeedsRma, [](RmaArgsScene& s) { s.peerWin.winFlags = 0; }, ncclInvalidArgument, 0},
      {"ncclPutSignal: srcBuff is NULL", [](RmaArgsScene& s) { s.info.sendbuff = nullptr; }},
      {"peerWinOffset 4096 is greater than peerWin size 4096", [](RmaArgsScene& s) { s.info.peerWinOffset = 4096; }},
      {"srcBuff is not inside a registered ncclWindow", [](RmaArgsScene& s) { s.info.sendbuff = &s; }},
      {"srcWinHost is not in a valid symmetric window",
       [](RmaArgsScene& s) {
         s.cc.get()->symmetricSupport = 1;
         s.srcWin.winFlags = kRmaRegistered;
       }},
      {kNeedsRma, [](RmaArgsScene& s) { s.srcWin.winFlags = 0; }, ncclInvalidArgument, 1},
      {"multiple physical cuMem segments",
       [](RmaArgsScene& s) {
         s.cc.get()->symmetricSupport = 1;
         g_devrWindowIsMultiSegmentValue = true;
       }},
      {"host-backed cuMem segments",
       [](RmaArgsScene& s) {
         s.cc.get()->symmetricSupport = 1;
         g_devrWindowHasSysmemSegmentValue = true;
       }},
      {"ncclSignal: count must be 0",
       [](RmaArgsScene& s) {
         s.info.coll = ncclFuncSignal;
         s.info.count = 1;
       }},
      {"called before RMA is initialized", [](RmaArgsScene& s) { s.info.coll = ncclFuncSignal; }, ncclInvalidUsage},
  };
  for (const BadRmaCase& c : cases) {
    RmaArgsScene sc;
    c.breakIt(sc);
    std::string log;
    EXPECT_EQ(c.rc, sc.Run(&log)) << c.warn;
    EXPECT_TRUE(RcclUnitTesting::LogHas(log, c.warn)) << c.warn << "\nactual log:\n" << log;
    EXPECT_EQ(nullptr, ncclGroupCommHead[ncclGroupTaskTypeCollective]) << c.warn;
    if (c.findWindowCalls >= 0) {
      EXPECT_EQ(c.findWindowCalls, sc.findWindow.calls) << c.warn;
    }
    g_devrWindowIsMultiSegmentValue = false;
    g_devrWindowHasSysmemSegmentValue = false;
  }
}

TEST_F(EnqueueMicrotest, RmaTaskAppend_WaitSignalWithABadDescriptor_IsRejected) {
  for (const BadWaitSignal& c : kBadWaitSignals) {
    RmaArgsScene sc;
    ncclWaitSignalDesc_t descs[2];
    PointAtBadWaitSignal(c, descs, &sc.info);
    std::string log;
    EXPECT_EQ(ncclInvalidArgument, sc.Run(&log)) << c.warn;
    EXPECT_TRUE(RcclUnitTesting::LogHas(log, c.warn)) << log;
  }
}

TEST_F(EnqueueMicrotest, PlannerSetCapturingGraph_StreamsFromDifferentGraphs_AreRejected) {
  const hipStream_t kCaptured = reinterpret_cast<hipStream_t>(0xA2);
  ScopedHook graphs(g_cudaGetCapturingGraph, [&](ncclCudaGraph* graph, hipStream_t stream, int mode) {
    *graph = ncclCudaGraphNone(mode);
    if (stream == kCaptured) {
      graph->graph = reinterpret_cast<hipGraph_t>(0xB0);
      graph->graphId = 7;
    }
    return ncclSuccess;
  });
  RawTaskScene sc;
  ncclGroupCommJoin(sc.cc.get(), ncclGroupTaskTypeCollective);
  ncclInfo info = sc.Info(ncclFuncAllReduce);
  for (uintptr_t uncaptured : {0xA1, 0xA3}) {
    info.stream = reinterpret_cast<hipStream_t>(uncaptured);
    ASSERT_EQ(ncclSuccess, ncclPlannerSetCapturingGraph(sc.cc.get(), &info));
  }
  EXPECT_EQ(2, sc.cc.get()->planner.numStreams);

  info.stream = kCaptured;
  ncclResult_t rc = ncclSuccess;
  const std::string log = RcclUnitTesting::CaptureLog([&] { rc = ncclPlannerSetCapturingGraph(sc.cc.get(), &info); });
  EXPECT_EQ(ncclInvalidUsage, rc);
  EXPECT_TRUE(RcclUnitTesting::LogHas(log, "must either be all uncaptured or all captured by the same graph")) << log;
  EXPECT_EQ(2, sc.cc.get()->planner.numStreams);
  EXPECT_FALSE(ncclCudaGraphValid(sc.cc.get()->planner.capturingGraph));
}

// ===========================================================================
// Direct P2P AllToAll peer layers (RCCL_A2A_P2P_PEERS_IN_FLIGHT): the group isolation and the pre-connect of a
// layer's channel set.
// ===========================================================================

namespace {
// 8 ranks, 32 channels, 2 per peer, one ncclChannelPeer per channel and rank, 4 in flight: 8 channels per peer and 4
// sets of 8 channels. Round r sends to r and receives from (8 - r) mod 8, as in AllToAllPlanScene.
struct LayerPreconnectScene {
  static constexpr int kRanks = 8;
  static constexpr int kPool = 32;
  std::unique_ptr<ncclComm> comm = P2pPartsComm(/*nNodes=*/1, kRanks, kPool, /*perPeer=*/2);
  std::vector<ncclComm::P2pSchedulePair> rounds = std::vector<ncclComm::P2pSchedulePair>(kRanks);
  std::vector<ncclChannelPeer> peers = std::vector<ncclChannelPeer>(kPool * kRanks);
  std::vector<std::vector<ncclChannelPeer*>> slots =
    std::vector<std::vector<ncclChannelPeer*>>(kPool, std::vector<ncclChannelPeer*>(kRanks));
  std::vector<channelMasks> send = std::vector<channelMasks>(3 * kRanks);
  std::vector<channelMasks> recv = std::vector<channelMasks>(3 * kRanks);

  LayerPreconnectScene() {
    SetPeersInFlight(4);
    comm->rank = 0;
    for (int round = 0; round < kRanks; ++round) {
      rounds[round].sendRank = round;
      rounds[round].recvRank = (kRanks - round) % kRanks;
    }
    comm->p2pSchedule = rounds.data();
    for (int ch = 0; ch < kPool; ++ch) {
      for (int r = 0; r < kRanks; ++r) {
        slots[ch][r] = &peers[ch * kRanks + r];
      }
      comm->channels[ch].peers = slots[ch].data();
    }
    comm->connectSend = send.data();
    comm->connectRecv = recv.data();
    // Not yet in this thread's preconnect list.
    comm->preconnectNext = reinterpret_cast<ncclComm*>(0x1);
    ncclGroupCommPreconnectHead = nullptr;
  }
  ~LayerPreconnectScene() {
    ncclGroupCommPreconnectHead = nullptr;
  }
};
}  // namespace

TEST_F(EnqueueMicrotest, A2aP2pLayerPreconnect_MarksTheRoundsSetOnceOnConnectorOne) {
  LayerPreconnectScene sc;
  ncclComm* comm = sc.comm.get();
  const int parts = rcclA2aP2pLayerParts(comm, ncclFuncAlltoAll, kLayerPeerBytes, /*peer=*/3);
  ASSERT_EQ(8, parts);

  // Send to 3 is round 3: set 3 = channels 24-31 (its receive round, 5, would be set 1).
  rcclA2aP2pLayerPreconnect(comm, /*isSendNotRecv=*/true, /*peer=*/3, parts);
  EXPECT_EQ(uint64_t(0xff000000), sc.send[3].masks[0]);
  EXPECT_EQ(uint64_t(0), sc.recv[3].masks[0]);
  for (int ch = 0; ch < LayerPreconnectScene::kPool; ++ch) {
    EXPECT_EQ(ch >= 24 ? 1 : 0, comm->channels[ch].peers[3]->send[1].hasSeen) << "channel " << ch;
    EXPECT_EQ(ch >= 24 ? 1 : 0, comm->channels[ch].peers[3]->send[1].p2pOnly) << "channel " << ch;
    EXPECT_EQ(0, comm->channels[ch].peers[3]->send[0].hasSeen);
  }
  EXPECT_EQ(comm, ncclGroupCommPreconnectHead) << "the comm joins the preconnect list";
  // Send to 2 is round 2: set 2 = channels 16-23. A channel already seen keeps its mask bit clear.
  comm->channels[19].peers[2]->send[1].hasSeen = 1;
  rcclA2aP2pLayerPreconnect(comm, /*isSendNotRecv=*/true, /*peer=*/2, parts);
  EXPECT_EQ(uint64_t(0x00f70000), sc.send[2].masks[0]);
  rcclA2aP2pLayerPreconnect(comm, /*isSendNotRecv=*/false, /*peer=*/1, parts);
  EXPECT_EQ(uint64_t(0xff000000), sc.recv[1].masks[0]) << "receive from 1 is round 7: set 3";
}

TEST_F(EnqueueMicrotest, A2aP2pLayerTaskAppend_IsolatesAndPreconnectsOnlyLayerableTasks) {
  struct Case {
    ncclFunc_t collAPI;
    ssize_t bytes;
    int maxCTAs;
    int connIndex;
    bool isolates;
    uint64_t sendMask;
  };
  // A large uncapped AllToAll send to 3 isolates the group and marks set 3 (channels 24-31). Capped CTAs, a small peer
  // or another call does neither; a connector other than 1 isolates without marking.
  const ssize_t small = (ssize_t(4) << 20) - 1;
  for (Case k : {Case{ncclFuncAlltoAll, kLayerPeerBytes, MAXCHANNELS, 1, true, 0xff000000},
                 Case{ncclFuncAlltoAll, kLayerPeerBytes, MAXCHANNELS - 1, 1, false, 0},
                 Case{ncclFuncAlltoAll, small, MAXCHANNELS, 1, false, 0},
                 Case{ncclFuncSend, kLayerPeerBytes, MAXCHANNELS, 1, false, 0},
                 Case{ncclFuncGather, kLayerPeerBytes, MAXCHANNELS, 1, false, 0},
                 Case{ncclFuncAlltoAll, kLayerPeerBytes, MAXCHANNELS, NCCL_CONN_IDX_P2P_CE, true, 0}}) {
    LayerPreconnectScene sc;
    ncclTaskP2p task{};
    task.func = ncclFuncSend;
    task.collAPI = k.collAPI;
    task.root = 3;
    task.bytes = k.bytes;
    task.maxCTAs = k.maxCTAs;
    rcclA2aP2pLayerTaskAppend(sc.comm.get(), &task, /*isSendNotRecv=*/true, k.connIndex);
    EXPECT_EQ(k.isolates, sc.comm->planner.p2pCollNeedsIsolation)
      << "collAPI=" << int(k.collAPI) << " bytes=" << k.bytes << " maxCTAs=" << k.maxCTAs;
    EXPECT_EQ(k.sendMask, sc.send[3].masks[0])
      << "collAPI=" << int(k.collAPI) << " bytes=" << k.bytes << " connIndex=" << k.connIndex;
  }
}
