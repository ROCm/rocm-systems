/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests; UUTs are #include'd. Line citations use src/scheduler/ numbers; hipify adds 1.

#include <gtest/gtest.h>

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#include "../common/LogCapture.hpp"
#include "ScopedHook.h"
#include "fakes/dev_runtime_micro_fakes.h"
#include "fakes/scheduler_fakes.h"
#include "fakes/sym_kernels_fakes.h"

// ENABLE_WARP_SPEED is a binary-wide compile definition (test/host/CMakeLists.txt), not defined here: this TU
// shares struct ncclComm with dev-runtime-test.cc's real dev_runtime.cc, and a per-TU #define would be an ODR
// violation the moment cross-TU code (like ncclDevrInitOnce) touches a WarpSpeed-conditional field.

// Hipify renames the allgatherv one to *_tmp.cc: src/enqueue/task_sched/allgatherv_sched.cc has the basename.
#include ALLGATHERV_SCHED_CC_PATH
#include SYMMETRIC_SCHED_CC_PATH

namespace {
constexpr int kBaselineChannels = 4;
constexpr uint64_t kPoison = 0xDEADBEEFDEADBEEFull;

// Recomputed independently of convertSymTaskDevOp's own union-pun, so a mutated divisor still gets caught.
uint64_t ConvertSymTaskDevOp_ExpectedReciprocalScalar(int nRanks) {
  union { float f32; uint64_t u64; } u;
  u.u64 = 0;
  u.f32 = float(1.0 / nRanks);
  return u.u64;
}

// Minimal ncclComm/ncclKernelPlan/planner.peers scaffold; ranks and bcast peers are the same set (numPeers).
class ScheduleBcastTasksToPlan_Scene {
 public:
  explicit ScheduleBcastTasksToPlan_Scene(int numPeers)
      : comm(new ncclComm{}),
        plan(new ncclKernelPlan{}),
        peers(new ncclKernelPlanner::Peer[numPeers]{}),
        ringTasks(new ncclTaskBcast*[numPeers]{}),
        rankToIndex(new int[numPeers]{}) {
    comm->nChannels = 1;
    comm->nRanks = numPeers;
    comm->rank = 0;
    comm->planner.nTasksBcast = 1;
    comm->planner.peers = peers.get();
    comm->planner.bcast_info.minBcastPeer = 0;
    comm->planner.bcast_info.maxBcastPeer = numPeers - 1;
    comm->ringTasks = ringTasks.get();
    comm->channels[0].ring.rankToIndex = rankToIndex.get();
    ncclMemoryStackConstruct(&comm->memScoped);
  }
  std::unique_ptr<ncclComm> comm;
  std::unique_ptr<ncclKernelPlan> plan;
  std::unique_ptr<ncclKernelPlanner::Peer[]> peers;
  std::unique_ptr<ncclTaskBcast*[]> ringTasks;
  std::unique_ptr<int[]> rankToIndex;
};

// Walks plan->workQueue to inspect the ncclDevWorkBcast built for each accepted ring-depth slice.
std::vector<ncclDevWorkBcast*> ScheduleBcastTasksToPlan_CollectWorkItems(struct ncclKernelPlan* plan) {
  std::vector<ncclDevWorkBcast*> items;
  for (ncclWorkList* node = ncclIntruQueueHead(&plan->workQueue); node != nullptr; node = node->next) {
    items.push_back(reinterpret_cast<ncclDevWorkBcast*>(node + 1));
  }
  return items;
}

// Walks plan->bcastTaskQueue (an intrusive queue keyed on ncclTaskBcast::next) to see what the tail loop drained.
std::vector<ncclTaskBcast*> ScheduleBcastTasksToPlan_CollectBcastTaskQueue(struct ncclKernelPlan* plan) {
  std::vector<ncclTaskBcast*> items;
  for (ncclTaskBcast* t = ncclIntruQueueHead(&plan->bcastTaskQueue); t != nullptr; t = t->next) items.push_back(t);
  return items;
}

// Minimal ncclComm scaffold for ncclMakeSymmetricTaskList; nRanks=1 skips the bootstrap-consensus block by default.
class MakeSymmetricTaskList_Scene {
 public:
  MakeSymmetricTaskList_Scene() : comm(new ncclComm{}) {
    comm->nRanks = 1;
    comm->rank = 0;
  }
  std::unique_ptr<ncclComm> comm;
};

// Mirrors ncclDevFuncId's general-collective key (device.h); AllGatherV never takes the special-cased branches.
uint64_t ScheduleBcastTasksToPlan_DevFuncKey(int proto) {
  return (uint64_t(ncclFuncAllGatherV & RCCL_FUNC_ID_MASK) << RCCL_COLL_SHIFT) |
         (uint64_t(NCCL_ALGO_RING & RCCL_FUNC_ID_MASK) << RCCL_ALGO_SHIFT) |
         (uint64_t(proto & RCCL_FUNC_ID_MASK) << RCCL_PROTO_SHIFT);
}
}  // namespace

class SchedulerMicrotest : public ::testing::Test {
 protected:
  void TearDown() override {
    ResetSchedulerFakes();
    ResetSymKernelsFakes();      // g_symRegType is a plain global, not a ScopedHook-restorable std::function
    ResetDevRuntimeMicroFakes();  // covers g_devrBootstrapAllGather, reused here for real bootstrapAllGather
  }
};

TEST_F(SchedulerMicrotest, AgvChannelCount_MultiplierAtMost1_ReturnsTunedChannelsUnchanged) {
  std::unique_ptr<ncclComm> comm(new ncclComm{});
  comm->warpSpeedChannelMultiplier = 1;
  RcclUnitTesting::ScopedDebugLogging debugLogging(NCCL_LOG_INFO, NCCL_COLL);
  int result = -1;
  const std::string log =
      RcclUnitTesting::CaptureLog([&]() { result = agvChannelCount(comm.get(), kBaselineChannels); });
  EXPECT_EQ(result, kBaselineChannels);
  EXPECT_FALSE(RcclUnitTesting::LogHas(log, "AllGatherV: WarpSpeed not supported"));
}

TEST_F(SchedulerMicrotest, AgvChannelCount_MultiplierAbove1_DividesTunedChannels) {
  std::unique_ptr<ncclComm> comm(new ncclComm{});
  comm->warpSpeedChannelMultiplier = 2;
  RcclUnitTesting::ScopedDebugLogging debugLogging(NCCL_LOG_INFO, NCCL_COLL);
  int result = -1;
  const std::string log =
      RcclUnitTesting::CaptureLog([&]() { result = agvChannelCount(comm.get(), kBaselineChannels); });
  EXPECT_EQ(result, kBaselineChannels / 2);
  EXPECT_TRUE(RcclUnitTesting::LogHas(log, "AllGatherV: WarpSpeed not supported"));
}

TEST_F(SchedulerMicrotest, AgvChannelCount_MultiplierAbove1_FloorsResultAtOne) {
  std::unique_ptr<ncclComm> comm(new ncclComm{});
  comm->warpSpeedChannelMultiplier = 8;
  EXPECT_EQ(agvChannelCount(comm.get(), /*tunedChannels=*/1), 1);
}

TEST_F(SchedulerMicrotest, SymkRedOp_Avg_ReturnsDevSumPostDivRegardlessOfInputDevOp) {
  EXPECT_EQ(symkRedOp(ncclAvg, ncclDevSum), ncclDevSumPostDiv);
  EXPECT_EQ(symkRedOp(ncclAvg, ncclDevMinMax), ncclDevSumPostDiv);
}

TEST_F(SchedulerMicrotest, SymkRedOp_NonAvg_ReturnsDevRedOpUnchanged) {
  EXPECT_EQ(symkRedOp(ncclSum, ncclDevSum), ncclDevSum);
  EXPECT_EQ(symkRedOp(ncclMax, ncclDevMinMax), ncclDevMinMax);
  EXPECT_EQ(symkRedOp(ncclProd, ncclDevProd), ncclDevProd);
}

TEST_F(SchedulerMicrotest, ConvertSymTaskDevOp_NonAvgPassthrough_LeavesScalarArgUntouched) {
  std::unique_ptr<ncclComm> comm(new ncclComm{});
  comm->nRanks = 4;
  ncclTaskColl task{};
  task.opHost = ncclSum;
  task.opDev.op = ncclDevSum;
  task.opDev.scalarArg = kPoison;
  task.devFuncId = ncclSymkKernelId_AllReduce_AGxLL_R;
  task.datatype = ncclFloat16;
  convertSymTaskDevOp(comm.get(), &task);
  EXPECT_EQ(task.opDev.op, ncclDevSum);
  EXPECT_EQ(task.opDev.scalarArg, kPoison);
}

TEST_F(SchedulerMicrotest, ConvertSymTaskDevOp_ReduceScatterLdmc_ReturnsEarlyWithoutPackingScalar) {
  std::unique_ptr<ncclComm> comm(new ncclComm{});
  comm->nRanks = 4;
  ncclTaskColl task{};
  task.opHost = ncclAvg;
  task.opDev.op = ncclDevSum;
  task.opDev.scalarArg = kPoison;
  task.devFuncId = ncclSymkKernelId_ReduceScatter_LDMC;
  task.datatype = ncclFloat16;
  convertSymTaskDevOp(comm.get(), &task);
  EXPECT_EQ(task.opDev.op, ncclDevSumPostDiv);
  EXPECT_EQ(task.opDev.scalarArg, kPoison);
}

TEST_F(SchedulerMicrotest, ConvertSymTaskDevOp_Float16_PacksReciprocalNRanksScalar) {
  std::unique_ptr<ncclComm> comm(new ncclComm{});
  comm->nRanks = 4;
  ncclTaskColl task{};
  task.opHost = ncclAvg;
  task.opDev.op = ncclDevSum;
  task.opDev.scalarArg = kPoison;
  task.devFuncId = ncclSymkKernelId_AllReduce_AGxLL_R;
  task.datatype = ncclFloat16;
  convertSymTaskDevOp(comm.get(), &task);
  EXPECT_EQ(task.opDev.op, ncclDevSumPostDiv);
  EXPECT_EQ(task.opDev.scalarArg, ConvertSymTaskDevOp_ExpectedReciprocalScalar(4));
}

TEST_F(SchedulerMicrotest, ConvertSymTaskDevOp_Bfloat16_PacksReciprocalNRanksScalar) {
  std::unique_ptr<ncclComm> comm(new ncclComm{});
  comm->nRanks = 4;
  ncclTaskColl task{};
  task.opHost = ncclAvg;
  task.opDev.op = ncclDevSum;
  task.opDev.scalarArg = kPoison;
  task.devFuncId = ncclSymkKernelId_AllReduce_AGxLL_R;
  task.datatype = ncclBfloat16;
  convertSymTaskDevOp(comm.get(), &task);
  EXPECT_EQ(task.opDev.scalarArg, ConvertSymTaskDevOp_ExpectedReciprocalScalar(4));
}

TEST_F(SchedulerMicrotest, ConvertSymTaskDevOp_Float8e4m3_PacksReciprocalNRanksScalar) {
  std::unique_ptr<ncclComm> comm(new ncclComm{});
  comm->nRanks = 4;
  ncclTaskColl task{};
  task.opHost = ncclAvg;
  task.opDev.op = ncclDevSum;
  task.opDev.scalarArg = kPoison;
  task.devFuncId = ncclSymkKernelId_AllReduce_AGxLL_R;
  task.datatype = ncclFloat8e4m3;
  convertSymTaskDevOp(comm.get(), &task);
  EXPECT_EQ(task.opDev.scalarArg, ConvertSymTaskDevOp_ExpectedReciprocalScalar(4));
}

TEST_F(SchedulerMicrotest, ConvertSymTaskDevOp_Float8e5m2_PacksReciprocalNRanksScalar) {
  std::unique_ptr<ncclComm> comm(new ncclComm{});
  comm->nRanks = 4;
  ncclTaskColl task{};
  task.opHost = ncclAvg;
  task.opDev.op = ncclDevSum;
  task.opDev.scalarArg = kPoison;
  task.devFuncId = ncclSymkKernelId_AllReduce_AGxLL_R;
  task.datatype = ncclFloat8e5m2;
  convertSymTaskDevOp(comm.get(), &task);
  EXPECT_EQ(task.opDev.scalarArg, ConvertSymTaskDevOp_ExpectedReciprocalScalar(4));
}

TEST_F(SchedulerMicrotest, ConvertSymTaskDevOp_DefaultDatatype_LeavesScalarArgUntouched) {
  std::unique_ptr<ncclComm> comm(new ncclComm{});
  comm->nRanks = 4;
  ncclTaskColl task{};
  task.opHost = ncclAvg;
  task.opDev.op = ncclDevSum;
  task.opDev.scalarArg = kPoison;
  task.devFuncId = ncclSymkKernelId_AllReduce_AGxLL_R;
  task.datatype = ncclInt32;
  convertSymTaskDevOp(comm.get(), &task);
  EXPECT_EQ(task.opDev.op, ncclDevSumPostDiv);
  EXPECT_EQ(task.opDev.scalarArg, kPoison);
}

TEST_F(SchedulerMicrotest, SymBatchAligned16B_SingleTaskNoWindowsAligned_ReturnsTrue) {
  ncclTaskColl t{};
  t.sendbuff = reinterpret_cast<void*>(0x1030);
  t.recvbuff = reinterpret_cast<void*>(0x1020);
  t.isSymLast = 1;
  EXPECT_TRUE(symBatchAligned16B(&t));
}

TEST_F(SchedulerMicrotest, SymBatchAligned16B_SingleTaskNoWindowsMisaligned_ReturnsFalse) {
  ncclTaskColl t{};
  t.sendbuff = reinterpret_cast<void*>(0x1028);  // offset 8: a multiple of 8 but not of 16
  t.recvbuff = reinterpret_cast<void*>(0x1020);
  t.isSymLast = 1;
  EXPECT_FALSE(symBatchAligned16B(&t));
}

TEST_F(SchedulerMicrotest, SymBatchAligned16B_FirstAlignedSecondMisaligned_TraversesAndReturnsFalse) {
  ncclTaskColl second{};
  second.sendbuff = reinterpret_cast<void*>(0x2031);
  second.recvbuff = reinterpret_cast<void*>(0x2020);
  second.isSymLast = 1;
  ncclTaskColl first{};
  first.sendbuff = reinterpret_cast<void*>(0x1030);
  first.recvbuff = reinterpret_cast<void*>(0x1020);
  first.isSymLast = 0;
  first.next = &second;
  EXPECT_FALSE(symBatchAligned16B(&first));
}

TEST_F(SchedulerMicrotest, SymBatchAligned16B_FirstMisalignedNotLast_ReturnsFalseWithoutTraversing) {
  ncclTaskColl first{};
  first.sendbuff = reinterpret_cast<void*>(0x1031);
  first.recvbuff = reinterpret_cast<void*>(0x1020);
  first.isSymLast = 0;
  first.next = reinterpret_cast<ncclTaskColl*>(0x1);  // must never be dereferenced
  EXPECT_FALSE(symBatchAligned16B(&first));
}

TEST_F(SchedulerMicrotest, SymBatchAligned16B_WindowOffsetsAligned_RawBuffersMisaligned_ReturnsTrue) {
  ncclDevrWindow sendWin{};
  sendWin.userPtr = reinterpret_cast<void*>(0x1003);
  ncclDevrWindow recvWin{};
  recvWin.userPtr = reinterpret_cast<void*>(0x2007);
  ncclTaskColl t{};
  t.sendWin = &sendWin;
  t.recvWin = &recvWin;
  t.sendbuff = reinterpret_cast<void*>(0x1013);  // inputOff (via window) = 16
  t.recvbuff = reinterpret_cast<void*>(0x2007);  // outputOff (via window) = 0
  t.isSymLast = 1;
  EXPECT_TRUE(symBatchAligned16B(&t));
}

TEST_F(SchedulerMicrotest, SymBatchAligned16B_SendWindowOnly_UsesWindowOffsetForSend_ReturnsTrue) {
  ncclDevrWindow sendWin{};
  sendWin.userPtr = reinterpret_cast<void*>(0x1003);
  ncclTaskColl t{};
  t.sendWin = &sendWin;
  t.recvWin = nullptr;
  t.sendbuff = reinterpret_cast<void*>(0x1013);  // inputOff (via window) = 16
  t.recvbuff = reinterpret_cast<void*>(0x2000);  // outputOff (raw, no window) = 0x2000
  t.isSymLast = 1;
  EXPECT_TRUE(symBatchAligned16B(&t));
}

TEST_F(SchedulerMicrotest, ScheduleBcastTasksToPlan_NoBcastTasks_ReturnsSuccessWithoutTouchingPeers) {
  ScheduleBcastTasksToPlan_Scene scene(/*numPeers=*/1);
  scene.comm->planner.nTasksBcast = 0;
  scene.comm->planner.peers = nullptr;  // would crash if the loop were ever reached
  EXPECT_EQ(ncclScheduleBcastTasksToPlan(scene.comm.get(), scene.plan.get(), nullptr), ncclSuccess);
}

TEST_F(SchedulerMicrotest, ScheduleBcastTasksToPlan_WorkBatchesAlreadyPresent_ReturnsSuccessImmediately) {
  ScheduleBcastTasksToPlan_Scene scene(/*numPeers=*/1);
  scene.plan->nWorkBatches = 1;
  scene.comm->planner.peers = nullptr;  // would crash if the loop were ever reached
  EXPECT_EQ(ncclScheduleBcastTasksToPlan(scene.comm.get(), scene.plan.get(), nullptr), ncclSuccess);
}

TEST_F(SchedulerMicrotest, ScheduleBcastTasksToPlan_AllPeersEmpty_SkipsEachAndReturnsSuccess) {
  ScheduleBcastTasksToPlan_Scene scene(/*numPeers=*/3);
  EXPECT_EQ(ncclScheduleBcastTasksToPlan(scene.comm.get(), scene.plan.get(), nullptr), ncclSuccess);
  EXPECT_EQ(g_testBudgetCalls, 0);  // never reached: every peer was skipped
}

TEST_F(SchedulerMicrotest, ScheduleBcastTasksToPlan_BudgetDeniesFirstPeer_StopsBeforeAccumulating) {
  ScheduleBcastTasksToPlan_Scene scene(/*numPeers=*/1);
  ncclTaskBcast task{};
  task.count = 100;
  scene.peers[0].bcastQueue.head = &task;

  int recordedNWorkBatches = -1;
  ssize_t recordedNWorkBytes = -1;
  ScopedHook budgetHook(g_testBudget, [&](struct ncclKernelPlanBudget*, int nWorkBatches, ssize_t nWorkBytes) {
    recordedNWorkBatches = nWorkBatches;
    recordedNWorkBytes = nWorkBytes;
    return false;
  });

  EXPECT_EQ(ncclScheduleBcastTasksToPlan(scene.comm.get(), scene.plan.get(), nullptr), ncclSuccess);
  EXPECT_EQ(budgetHook.calls, 1);
  EXPECT_EQ(recordedNWorkBatches, 1);
  EXPECT_EQ(recordedNWorkBytes, static_cast<ssize_t>(sizeof(ncclDevWorkBcast)));
}

TEST_F(SchedulerMicrotest, ScheduleBcastTasksToPlan_TwoPeersAccumulate_ProceedsPastBatchCheck) {
  ScheduleBcastTasksToPlan_Scene scene(/*numPeers=*/2);
  ncclTaskBcast task0{};
  task0.count = 111;
  ncclTaskBcast task1{};
  task1.count = 222;
  scene.peers[0].bcastQueue.head = &task0;
  scene.peers[1].bcastQueue.head = &task1;

  // ncclDevFuncNameToId is empty by default, so this proceeds past batchTasks==0, unlike Block 3.
  EXPECT_EQ(ncclScheduleBcastTasksToPlan(scene.comm.get(), scene.plan.get(), nullptr), ncclInvalidUsage);
}

TEST_F(SchedulerMicrotest, ScheduleBcastTasksToPlan_SkipThenAccumulate_ContinuesLoopPastNullPeer) {
  ScheduleBcastTasksToPlan_Scene scene(/*numPeers=*/3);
  ncclTaskBcast task1{};
  task1.count = 50;
  ncclTaskBcast task2{};
  task2.count = 60;
  scene.peers[1].bcastQueue.head = &task1;
  scene.peers[2].bcastQueue.head = &task2;

  EXPECT_EQ(ncclScheduleBcastTasksToPlan(scene.comm.get(), scene.plan.get(), nullptr), ncclInvalidUsage);
}

TEST_F(SchedulerMicrotest, ScheduleBcastTasksToPlan_AlgoInfoFails_PropagatesErrorAndWiresTcollFromMaxBcastBytes) {
  ScheduleBcastTasksToPlan_Scene scene(/*numPeers=*/3);
  ncclTaskBcast task0{};
  task0.count = 50;
  ncclTaskBcast task1{};
  task1.count = 200;
  ncclTaskBcast task2{};
  task2.count = 30;
  scene.peers[0].bcastQueue.head = &task0;
  scene.peers[1].bcastQueue.head = &task1;
  scene.peers[2].bcastQueue.head = &task2;

  ncclFunc_t recordedFunc = ncclFuncSend;
  size_t recordedCount = 0;
  ncclDataType_t recordedDatatype = ncclFloat32;
  int recordedAlgorithm = -1;
  int recordedProtocol = -1;
  ScopedHook algoInfoHook(g_getAlgoInfo,
                          [&](struct ncclComm*, struct ncclTaskColl* task, int, int, int, ncclSimInfo_t*) {
                            recordedFunc = task->func;
                            recordedCount = task->count;
                            recordedDatatype = task->datatype;
                            recordedAlgorithm = task->algorithm;
                            recordedProtocol = task->protocol;
                            return ncclInternalError;
                          });

  EXPECT_EQ(ncclScheduleBcastTasksToPlan(scene.comm.get(), scene.plan.get(), nullptr), ncclInternalError);
  EXPECT_EQ(algoInfoHook.calls, 1);
  EXPECT_EQ(recordedFunc, ncclFuncAllGather);
  EXPECT_EQ(recordedCount, 200u);  // maxBcastBytes: the middle peer's count, not the first or the last
  EXPECT_EQ(recordedDatatype, ncclInt8);
  EXPECT_EQ(recordedAlgorithm, NCCL_ALGO_RING);
  EXPECT_EQ(recordedProtocol, NCCL_PROTO_UNDEF);
}

TEST_F(SchedulerMicrotest, ScheduleBcastTasksToPlan_FuncIndexNotFound_ReturnsInvalidUsageAfterSettingThreadPerBlock) {
  ScheduleBcastTasksToPlan_Scene scene(/*numPeers=*/1);
  ncclTaskBcast task{};
  task.count = 100;
  scene.peers[0].bcastQueue.head = &task;
  scene.comm->WarpSize = 64;

  ScopedHook algoInfoHook(g_getAlgoInfo, [&](struct ncclComm*, struct ncclTaskColl* task, int, int, int,
                                             ncclSimInfo_t*) {
    task->protocol = NCCL_PROTO_SIMPLE;
    task->nMaxChannels = 0;
    task->nWarps = 4;
    return ncclSuccess;
  });
  // ncclDevFuncNameToId is left empty: funcIndex is always -1 regardless of the (proto, algo) key.

  EXPECT_EQ(ncclScheduleBcastTasksToPlan(scene.comm.get(), scene.plan.get(), nullptr), ncclInvalidUsage);
  EXPECT_EQ(scene.plan->threadPerBlock, 4 * 64);  // set before the funcIndex check, even on this failing path
}

TEST_F(SchedulerMicrotest, ScheduleBcastTasksToPlan_FuncIndexFound_NotSpecialized_CallsPlanSetDefaultKernel) {
  ScheduleBcastTasksToPlan_Scene scene(/*numPeers=*/1);
  ncclTaskBcast task{};
  task.count = 100;
  scene.peers[0].bcastQueue.head = &task;
  scene.comm->collOpCount = 5;
  scene.plan->kernelSpecialized = false;

  ScopedHook algoInfoHook(g_getAlgoInfo, [&](struct ncclComm*, struct ncclTaskColl* task, int, int, int,
                                             ncclSimInfo_t*) {
    task->protocol = NCCL_PROTO_SIMPLE;
    task->nMaxChannels = 0;  // makes nParts 0: safe to run this call to completion
    task->nWarps = 1;
    return ncclSuccess;
  });
  ncclDevFuncNameToId[ScheduleBcastTasksToPlan_DevFuncKey(NCCL_PROTO_SIMPLE)] = 0;
  ScopedHook kernelHook(g_planSetDefaultKernel, [&](struct ncclComm*, struct ncclKernelPlan*) {});

  EXPECT_EQ(ncclScheduleBcastTasksToPlan(scene.comm.get(), scene.plan.get(), nullptr), ncclSuccess);
  EXPECT_EQ(kernelHook.calls, 1);
  EXPECT_EQ(scene.comm->collOpCount, 6u);
}

TEST_F(SchedulerMicrotest, ScheduleBcastTasksToPlan_FuncIndexFound_AlreadySpecialized_SkipsPlanSetDefaultKernel) {
  ScheduleBcastTasksToPlan_Scene scene(/*numPeers=*/1);
  ncclTaskBcast task{};
  task.count = 100;
  scene.peers[0].bcastQueue.head = &task;
  scene.plan->kernelSpecialized = true;

  ScopedHook algoInfoHook(g_getAlgoInfo, [&](struct ncclComm*, struct ncclTaskColl* task, int, int, int,
                                             ncclSimInfo_t*) {
    task->protocol = NCCL_PROTO_SIMPLE;
    task->nMaxChannels = 0;
    task->nWarps = 1;
    return ncclSuccess;
  });
  ncclDevFuncNameToId[ScheduleBcastTasksToPlan_DevFuncKey(NCCL_PROTO_SIMPLE)] = 0;
  ScopedHook kernelHook(g_planSetDefaultKernel, [&](struct ncclComm*, struct ncclKernelPlan*) {});

  EXPECT_EQ(ncclScheduleBcastTasksToPlan(scene.comm.get(), scene.plan.get(), nullptr), ncclSuccess);
  EXPECT_EQ(kernelHook.calls, 0);
}

TEST_F(SchedulerMicrotest, ScheduleBcastTasksToPlan_ProtoLL_ReachesFuncIndexWithoutCrashing) {
  ScheduleBcastTasksToPlan_Scene scene(/*numPeers=*/1);
  ncclTaskBcast task{};
  task.count = 100;
  scene.peers[0].bcastQueue.head = &task;

  ScopedHook algoInfoHook(g_getAlgoInfo, [&](struct ncclComm*, struct ncclTaskColl* task, int, int, int,
                                             ncclSimInfo_t*) {
    task->protocol = NCCL_PROTO_LL;
    task->nMaxChannels = 0;
    task->nWarps = 1;
    return ncclSuccess;
  });
  ncclDevFuncNameToId[ScheduleBcastTasksToPlan_DevFuncKey(NCCL_PROTO_LL)] = 0;

  EXPECT_EQ(ncclScheduleBcastTasksToPlan(scene.comm.get(), scene.plan.get(), nullptr), ncclSuccess);
}

TEST_F(SchedulerMicrotest, ScheduleBcastTasksToPlan_ProtoLL128_ReachesFuncIndexWithoutCrashing) {
  ScheduleBcastTasksToPlan_Scene scene(/*numPeers=*/1);
  ncclTaskBcast task{};
  task.count = 100;
  scene.peers[0].bcastQueue.head = &task;
  scene.comm->WarpSize = 64;
  scene.comm->ll128DataElems = 1;
  scene.comm->ll128LineElems = 1;

  ScopedHook algoInfoHook(g_getAlgoInfo, [&](struct ncclComm*, struct ncclTaskColl* task, int, int, int,
                                             ncclSimInfo_t*) {
    task->protocol = NCCL_PROTO_LL128;
    task->nMaxChannels = 0;
    task->nWarps = 1;
    return ncclSuccess;
  });
  ncclDevFuncNameToId[ScheduleBcastTasksToPlan_DevFuncKey(NCCL_PROTO_LL128)] = 0;

  EXPECT_EQ(ncclScheduleBcastTasksToPlan(scene.comm.get(), scene.plan.get(), nullptr), ncclSuccess);
}

TEST_F(SchedulerMicrotest, ScheduleBcastTasksToPlan_MaxItemBoundary_StopsWithoutBudgetDenial) {
  const int maxitem = ncclMaxDevWorkBatchBytes(/*cudaArch=*/0) / static_cast<int>(sizeof(ncclDevWorkBcast));
  const int numPeers = maxitem + 3;
  ScheduleBcastTasksToPlan_Scene scene(numPeers);
  ncclTaskBcast sharedTask{};
  sharedTask.count = 1;
  for (int i = 0; i < numPeers; i++) scene.peers[i].bcastQueue.head = &sharedTask;

  int calls = 0;
  ScopedHook budgetHook(g_testBudget, [&](struct ncclKernelPlanBudget*, int, ssize_t) {
    ++calls;
    if (calls > maxitem) {
      std::fprintf(stderr, "budget queried more than maxitem times\n");
      std::fflush(stderr);
      std::abort();
    }
    return true;
  });

  EXPECT_EQ(ncclScheduleBcastTasksToPlan(scene.comm.get(), scene.plan.get(), nullptr), ncclInvalidUsage);
}

TEST_F(SchedulerMicrotest, ScheduleBcastTasksToPlan_FourRingDepths_BuildsWorkItemsAndProxyOp) {
  ScheduleBcastTasksToPlan_Scene scene(/*numPeers=*/4);  // numPeers doubles as nRanks
  // 4-cycle (non-involutive): peer0->depth3, peer1->depth2, peer2->depth1, peer3->depth0.
  scene.rankToIndex[0] = 1;
  scene.rankToIndex[1] = 2;
  scene.rankToIndex[2] = 3;
  scene.rankToIndex[3] = 0;
  scene.comm->collOpCount = 7;
  scene.comm->buffSizes[NCCL_PROTO_SIMPLE] = 4096;  // stepSize=512, chunkSize=512 (no halving/rounding)

  ncclTaskBcast task0{}, task1{}, task2{}, task3{};
  task0.count = 1000;
  task0.recvbuff = reinterpret_cast<void*>(0x1300);
  task1.count = 2000;
  task1.recvbuff = reinterpret_cast<void*>(0x1200);
  task2.count = 3000;
  task2.recvbuff = reinterpret_cast<void*>(0x1100);
  task3.count = 4000;
  task3.recvbuff = reinterpret_cast<void*>(0x1000);
  task3.sendbuff = reinterpret_cast<void*>(0x2000);  // only depth 0's task feeds work->sendbuff
  scene.peers[0].bcastQueue.head = &task0;
  scene.peers[1].bcastQueue.head = &task1;
  scene.peers[2].bcastQueue.head = &task2;
  scene.peers[3].bcastQueue.head = &task3;

  ScopedHook algoInfoHook(g_getAlgoInfo, [&](struct ncclComm*, struct ncclTaskColl* task, int, int, int,
                                             ncclSimInfo_t*) {
    task->protocol = NCCL_PROTO_SIMPLE;
    task->nMaxChannels = 1;
    task->nWarps = 1;
    return ncclSuccess;
  });
  ncclDevFuncNameToId[ScheduleBcastTasksToPlan_DevFuncKey(NCCL_PROTO_SIMPLE)] = 0;

  struct WorkBatchCall {
    int channelId;
    enum ncclDevWorkType workType;
    int devFuncId;
    uint32_t workOffset;
    bool newBatch;
  };
  std::vector<WorkBatchCall> workBatchCalls;
  ScopedHook workBatchHook(g_addWorkBatchToPlan,
                           [&](struct ncclComm*, struct ncclKernelPlan*, int channelId, enum ncclDevWorkType workType,
                               int devFuncId, uint32_t workOffset, int, int, bool newBatch) {
                             workBatchCalls.push_back({channelId, workType, devFuncId, workOffset, newBatch});
                           });
  struct ncclProxyOp recordedProxyOp {};
  ScopedHook proxyOpHook(g_addProxyOpIfNeeded,
                         [&](struct ncclComm*, struct ncclKernelPlan*, struct ncclProxyOp* op) {
                           recordedProxyOp = *op;
                           return ncclSuccess;
                         });

  EXPECT_EQ(ncclScheduleBcastTasksToPlan(scene.comm.get(), scene.plan.get(), nullptr), ncclSuccess);

  const std::vector<ncclDevWorkBcast*> items = ScheduleBcastTasksToPlan_CollectWorkItems(scene.plan.get());
  ASSERT_EQ(items.size(), 4u);
  EXPECT_EQ(items[0]->ringDepth, 0);
  EXPECT_EQ(items[0]->bytes, 4000u);
  EXPECT_EQ(items[0]->recvbuff, reinterpret_cast<char*>(0x1000));
  EXPECT_EQ(items[0]->sendbuff, reinterpret_cast<char*>(0x2000));  // only depth 0 gets a sendbuff
  EXPECT_EQ(items[0]->chunkSize, 512);
  EXPECT_EQ(items[1]->ringDepth, 1);
  EXPECT_EQ(items[1]->bytes, 3000u);
  EXPECT_EQ(items[1]->sendbuff, nullptr);
  EXPECT_EQ(items[2]->ringDepth, 2);
  EXPECT_EQ(items[2]->bytes, 2000u);
  EXPECT_EQ(items[3]->ringDepth, 3);
  EXPECT_EQ(items[3]->bytes, 1000u);
  EXPECT_EQ(items[3]->sendbuff, nullptr);

  EXPECT_EQ(scene.plan->channelMask.masks[0] & 1u, 1u);
  EXPECT_EQ(scene.plan->workBytes, 4 * sizeof(ncclDevWorkBcast));

  ASSERT_EQ(workBatchCalls.size(), 4u);
  for (int i = 0; i < 4; i++) {
    EXPECT_EQ(workBatchCalls[i].channelId, 0);
    EXPECT_EQ(workBatchCalls[i].devFuncId, 0);
    EXPECT_EQ(workBatchCalls[i].workOffset, i * sizeof(ncclDevWorkBcast));
    EXPECT_EQ(workBatchCalls[i].newBatch, i == 0);  // true only for the very first batch of the whole call
  }

  // divUp gives slices 2/4/6/8; depth 0 is sendSlices-only, depth 3(=nRanks-1) is recvSlices-only.
  EXPECT_EQ(recordedProxyOp.channelId, 0);
  EXPECT_EQ(recordedProxyOp.opCount, 14u);  // (collOpCount=7) << 1
  EXPECT_EQ(recordedProxyOp.rank, 0);
  EXPECT_EQ(recordedProxyOp.coll, ncclFuncAllGatherV);
  EXPECT_EQ(recordedProxyOp.pattern, ncclPatternRing);
  EXPECT_EQ(recordedProxyOp.specifics.bcast.sendSlices, 8 + 6 + 4);
  EXPECT_EQ(recordedProxyOp.specifics.bcast.recvSlices, 6 + 4 + 2);
  EXPECT_EQ(recordedProxyOp.specifics.bcast.stepSize, 512);
  EXPECT_EQ(recordedProxyOp.dtype, ncclInt8);
  EXPECT_EQ(recordedProxyOp.redOp, ncclSum);
  EXPECT_EQ(recordedProxyOp.protocol, NCCL_PROTO_SIMPLE);
  EXPECT_EQ(recordedProxyOp.chunkSize, 512u);
  EXPECT_EQ(recordedProxyOp.sliceSize, 512u);
  EXPECT_EQ(recordedProxyOp.nbytes, 512);
  EXPECT_EQ(recordedProxyOp.nChannels, 1);
}

TEST_F(SchedulerMicrotest, ScheduleBcastTasksToPlan_EmptySliceFromZeroCount_SkipsWorkItemButKeepsOtherRingDepth) {
  ScheduleBcastTasksToPlan_Scene scene(/*numPeers=*/2);
  scene.rankToIndex[0] = 0;  // ringDepth 0
  scene.rankToIndex[1] = 1;  // ringDepth = nRanks(2) - 1 = 1
  scene.comm->buffSizes[NCCL_PROTO_SIMPLE] = 4096;

  ncclTaskBcast task0{};
  task0.count = 0;  // partBytes=0 -> offset_hi==offset_lo==0 -> skipped
  ncclTaskBcast task1{};
  task1.count = 500;
  scene.peers[0].bcastQueue.head = &task0;
  scene.peers[1].bcastQueue.head = &task1;

  ScopedHook algoInfoHook(g_getAlgoInfo, [&](struct ncclComm*, struct ncclTaskColl* task, int, int, int,
                                             ncclSimInfo_t*) {
    task->protocol = NCCL_PROTO_SIMPLE;
    task->nMaxChannels = 1;
    task->nWarps = 1;
    return ncclSuccess;
  });
  ncclDevFuncNameToId[ScheduleBcastTasksToPlan_DevFuncKey(NCCL_PROTO_SIMPLE)] = 0;
  int workBatchCalls = 0;
  ScopedHook workBatchHook(g_addWorkBatchToPlan,
                           [&](struct ncclComm*, struct ncclKernelPlan*, int, enum ncclDevWorkType, int, uint32_t,
                               int, int, bool) { ++workBatchCalls; });

  EXPECT_EQ(ncclScheduleBcastTasksToPlan(scene.comm.get(), scene.plan.get(), nullptr), ncclSuccess);

  const std::vector<ncclDevWorkBcast*> items = ScheduleBcastTasksToPlan_CollectWorkItems(scene.plan.get());
  ASSERT_EQ(items.size(), 1u);  // only ringDepth 1's task produced a work item
  EXPECT_EQ(items[0]->ringDepth, 1);
  EXPECT_EQ(workBatchCalls, 1);
}

TEST_F(SchedulerMicrotest, ScheduleBcastTasksToPlan_AllSlicesEmpty_SkipsProxyOpEntirely) {
  ScheduleBcastTasksToPlan_Scene scene(/*numPeers=*/1);
  scene.rankToIndex[0] = 0;
  scene.comm->buffSizes[NCCL_PROTO_SIMPLE] = 4096;

  ncclTaskBcast task{};
  task.count = 0;
  scene.peers[0].bcastQueue.head = &task;

  ScopedHook algoInfoHook(g_getAlgoInfo, [&](struct ncclComm*, struct ncclTaskColl* task, int, int, int,
                                             ncclSimInfo_t*) {
    task->protocol = NCCL_PROTO_SIMPLE;
    task->nMaxChannels = 1;
    task->nWarps = 1;
    return ncclSuccess;
  });
  ncclDevFuncNameToId[ScheduleBcastTasksToPlan_DevFuncKey(NCCL_PROTO_SIMPLE)] = 0;
  ScopedHook proxyOpHook(g_addProxyOpIfNeeded, [&](struct ncclComm*, struct ncclKernelPlan*, struct ncclProxyOp*) {
    return ncclSuccess;
  });

  EXPECT_EQ(ncclScheduleBcastTasksToPlan(scene.comm.get(), scene.plan.get(), nullptr), ncclSuccess);
  EXPECT_EQ(proxyOpHook.calls, 0);
  EXPECT_TRUE(ScheduleBcastTasksToPlan_CollectWorkItems(scene.plan.get()).empty());
}

TEST_F(SchedulerMicrotest, ScheduleBcastTasksToPlan_ProtoLL_HalvesChunkSizeBeforeGrainAlignment) {
  ScheduleBcastTasksToPlan_Scene scene(/*numPeers=*/2);
  scene.rankToIndex[0] = 0;  // ringDepth 0: real task, drives sendSlices so the proxy op is built
  scene.rankToIndex[1] = 1;  // ringDepth = nRanks(2) - 1 = 1: empty (count 0)
  scene.comm->buffSizes[NCCL_PROTO_LL] = 8192;  // stepSize=1024, chunkSize=1024, halved=512, grain(16)-aligned=512

  ncclTaskBcast task0{};
  task0.count = 1000;
  ncclTaskBcast task1{};
  task1.count = 0;
  scene.peers[0].bcastQueue.head = &task0;
  scene.peers[1].bcastQueue.head = &task1;

  ScopedHook algoInfoHook(g_getAlgoInfo, [&](struct ncclComm*, struct ncclTaskColl* task, int, int, int,
                                             ncclSimInfo_t*) {
    task->protocol = NCCL_PROTO_LL;
    task->nMaxChannels = 1;
    task->nWarps = 1;
    return ncclSuccess;
  });
  ncclDevFuncNameToId[ScheduleBcastTasksToPlan_DevFuncKey(NCCL_PROTO_LL)] = 0;
  struct ncclProxyOp recordedProxyOp {};
  ScopedHook proxyOpHook(g_addProxyOpIfNeeded,
                         [&](struct ncclComm*, struct ncclKernelPlan*, struct ncclProxyOp* op) {
                           recordedProxyOp = *op;
                           return ncclSuccess;
                         });

  EXPECT_EQ(ncclScheduleBcastTasksToPlan(scene.comm.get(), scene.plan.get(), nullptr), ncclSuccess);

  const std::vector<ncclDevWorkBcast*> items = ScheduleBcastTasksToPlan_CollectWorkItems(scene.plan.get());
  ASSERT_EQ(items.size(), 1u);
  EXPECT_EQ(items[0]->chunkSize, 512);  // would be 1024 if the LL halving were dropped
  EXPECT_EQ(proxyOpHook.calls, 1);
  EXPECT_EQ(recordedProxyOp.chunkSize, 512u);
  EXPECT_EQ(recordedProxyOp.sliceSize, 512u);  // chunkSize/chunkSteps*sliceSteps: distinct from stepSize(1024) here
  EXPECT_EQ(recordedProxyOp.specifics.bcast.stepSize, 1024);
  EXPECT_EQ(recordedProxyOp.nbytes, 1024);  // stepSize*sliceSteps
}

TEST_F(SchedulerMicrotest, ScheduleBcastTasksToPlan_ProtoLL128_RoundsChunkSizeToLineElems) {
  ScheduleBcastTasksToPlan_Scene scene(/*numPeers=*/1);
  scene.rankToIndex[0] = 0;
  // stepSize=chunkSize=32768; LL128 round (32768/8)*7=28672; grain(4096)-aligned stays 28672 exactly.
  scene.comm->buffSizes[NCCL_PROTO_LL128] = 262144;
  scene.comm->WarpSize = 64;
  scene.comm->ll128DataElems = 1;
  scene.comm->ll128LineElems = 1;

  ncclTaskBcast task{};
  task.count = 50000;
  scene.peers[0].bcastQueue.head = &task;

  ScopedHook algoInfoHook(g_getAlgoInfo, [&](struct ncclComm*, struct ncclTaskColl* task, int, int, int,
                                             ncclSimInfo_t*) {
    task->protocol = NCCL_PROTO_LL128;
    task->nMaxChannels = 1;
    task->nWarps = 1;
    return ncclSuccess;
  });
  ncclDevFuncNameToId[ScheduleBcastTasksToPlan_DevFuncKey(NCCL_PROTO_LL128)] = 0;

  EXPECT_EQ(ncclScheduleBcastTasksToPlan(scene.comm.get(), scene.plan.get(), nullptr), ncclSuccess);

  const std::vector<ncclDevWorkBcast*> items = ScheduleBcastTasksToPlan_CollectWorkItems(scene.plan.get());
  ASSERT_EQ(items.size(), 1u);
  EXPECT_EQ(items[0]->chunkSize, 28672);  // would be 32768 if the LL128 rounding were dropped
}

TEST_F(SchedulerMicrotest, ScheduleBcastTasksToPlan_ThreeChannels_SplitsBytesPerPartAndResetsRingTasks) {
  ScheduleBcastTasksToPlan_Scene scene(/*numPeers=*/2);
  scene.rankToIndex[0] = 0;  // ringDepth 0
  scene.rankToIndex[1] = 1;  // ringDepth = nRanks(2) - 1 = 1, empty (count 0): keeps this test to one task
  scene.comm->channels[1].ring.rankToIndex = scene.rankToIndex.get();
  scene.comm->channels[2].ring.rankToIndex = scene.rankToIndex.get();
  scene.comm->buffSizes[NCCL_PROTO_SIMPLE] = 4096;

  ncclTaskBcast task0{};
  task0.count = 301;  // divUp(301,3)=101; alignUp(_,256) makes part 1's slice land empty
  task0.recvbuff = reinterpret_cast<void*>(0x4000);
  ncclTaskBcast task1{};
  task1.count = 0;
  scene.peers[0].bcastQueue.head = &task0;
  scene.peers[1].bcastQueue.head = &task1;

  ScopedHook algoInfoHook(g_getAlgoInfo, [&](struct ncclComm*, struct ncclTaskColl* task, int, int, int,
                                             ncclSimInfo_t*) {
    task->protocol = NCCL_PROTO_SIMPLE;
    task->nMaxChannels = 3;  // 3 parts/channels
    task->nWarps = 1;
    return ncclSuccess;
  });
  ncclDevFuncNameToId[ScheduleBcastTasksToPlan_DevFuncKey(NCCL_PROTO_SIMPLE)] = 0;
  std::vector<int> workBatchChannelIds;
  ScopedHook workBatchHook(g_addWorkBatchToPlan,
                           [&](struct ncclComm*, struct ncclKernelPlan*, int channelId, enum ncclDevWorkType, int,
                               uint32_t, int, int, bool) { workBatchChannelIds.push_back(channelId); });
  std::vector<int> proxyOpChannelIds;
  ScopedHook proxyOpHook(g_addProxyOpIfNeeded, [&](struct ncclComm*, struct ncclKernelPlan*, struct ncclProxyOp* op) {
    proxyOpChannelIds.push_back(op->channelId);
    return ncclSuccess;
  });

  EXPECT_EQ(ncclScheduleBcastTasksToPlan(scene.comm.get(), scene.plan.get(), nullptr), ncclSuccess);

  const std::vector<ncclDevWorkBcast*> items = ScheduleBcastTasksToPlan_CollectWorkItems(scene.plan.get());
  // part 0: bytes 256 (0..alignUp(101,256)=256). part 1: empty (256..alignUp(202,256)=256, skipped).
  // part 2: bytes 45 (256..alignUp(303,256)=301, clamped to count).
  ASSERT_EQ(items.size(), 2u);
  EXPECT_EQ(items[0]->bytes, 256u);
  EXPECT_EQ(items[0]->recvbuff, reinterpret_cast<char*>(0x4000));  // part 0's offset_lo is 0
  EXPECT_EQ(items[1]->bytes, 45u);
  EXPECT_EQ(items[1]->recvbuff, reinterpret_cast<char*>(0x4000) + 256);  // part 2's offset_lo is 256
  EXPECT_EQ(workBatchChannelIds, (std::vector<int>{0, 2}));  // channel 1 never called: its slice was empty
  EXPECT_EQ(proxyOpChannelIds, (std::vector<int>{0, 2}));
  EXPECT_EQ(scene.plan->channelMask.masks[0] & 0b101u, 0b101u);  // channels 0 and 2 set, not 1
}

TEST_F(SchedulerMicrotest, ScheduleBcastTasksToPlan_PartBytes_RoundsCountUpNotDown) {
  ScheduleBcastTasksToPlan_Scene scene(/*numPeers=*/1);
  scene.rankToIndex[0] = 0;
  scene.comm->channels[1].ring.rankToIndex = scene.rankToIndex.get();
  scene.comm->buffSizes[NCCL_PROTO_SIMPLE] = 4096;

  ncclTaskBcast task{};
  task.count = 513;  // divUp(513,2)=257, floor(513,2)=256: the two straddle the 256-byte alignUp boundary
  scene.peers[0].bcastQueue.head = &task;

  ScopedHook algoInfoHook(g_getAlgoInfo, [&](struct ncclComm*, struct ncclTaskColl* task, int, int, int,
                                             ncclSimInfo_t*) {
    task->protocol = NCCL_PROTO_SIMPLE;
    task->nMaxChannels = 2;
    task->nWarps = 1;
    return ncclSuccess;
  });
  ncclDevFuncNameToId[ScheduleBcastTasksToPlan_DevFuncKey(NCCL_PROTO_SIMPLE)] = 0;

  EXPECT_EQ(ncclScheduleBcastTasksToPlan(scene.comm.get(), scene.plan.get(), nullptr), ncclSuccess);

  const std::vector<ncclDevWorkBcast*> items = ScheduleBcastTasksToPlan_CollectWorkItems(scene.plan.get());
  ASSERT_EQ(items.size(), 2u);
  // With partBytes=257 (correct): part0 0..alignUp(257,256)=512 -> bytes 512; part1 512..513 -> bytes 1.
  // With partBytes=256 (floor, wrong): part0 bytes would be 256 and part1 256 instead.
  EXPECT_EQ(items[0]->bytes, 512u);
  EXPECT_EQ(items[1]->bytes, 1u);
}

TEST_F(SchedulerMicrotest, ScheduleBcastTasksToPlan_TailLoop_DrainsSkippedPeerAndFillsPlanQueueInOrder) {
  ScheduleBcastTasksToPlan_Scene scene(/*numPeers=*/3);
  scene.comm->planner.nTasksBcast = 10;  // distinct from batchTasks(2), so a wrong-subtrahend mutant is observable

  ncclTaskBcast task0{};
  task0.count = 0;  // count=0 keeps this test's focus on the tail loop, not Block 5's per-channel slicing
  ncclTaskBcast task2{};
  task2.count = 0;
  scene.peers[0].bcastQueue.head = &task0;
  // scene.peers[1] left empty: exercises the t==nullptr skip-peer arm inside the tail loop's walk.
  scene.peers[2].bcastQueue.head = &task2;

  ncclDevFuncNameToId[ScheduleBcastTasksToPlan_DevFuncKey(NCCL_PROTO_SIMPLE)] = 0;

  EXPECT_EQ(ncclScheduleBcastTasksToPlan(scene.comm.get(), scene.plan.get(), nullptr), ncclSuccess);

  const std::vector<ncclTaskBcast*> drained = ScheduleBcastTasksToPlan_CollectBcastTaskQueue(scene.plan.get());
  ASSERT_EQ(drained.size(), 2u);
  EXPECT_EQ(drained[0], &task0);  // order matters: peer0 walked before peer2, peer1 contributed nothing
  EXPECT_EQ(drained[1], &task2);
  EXPECT_EQ(scene.plan->nTasksBcast, 2);
  EXPECT_EQ(scene.comm->planner.nTasksBcast, 8);  // 10 - batchTasks(2), not 10 - 1
  EXPECT_EQ(scene.peers[0].bcastQueue.head, nullptr);  // fully drained, not left with a stale head
  EXPECT_EQ(scene.peers[1].bcastQueue.head, nullptr);  // was already empty
  EXPECT_EQ(scene.peers[2].bcastQueue.head, nullptr);
}

TEST_F(SchedulerMicrotest, ScheduleBcastTasksToPlan_TailLoop_PeerWithTwoQueuedTasks_DequeuesOnlyTheHeadThisCall) {
  ScheduleBcastTasksToPlan_Scene scene(/*numPeers=*/1);
  scene.comm->planner.nTasksBcast = 5;

  ncclTaskBcast task0{};
  task0.count = 0;
  ncclTaskBcast task1{};
  task1.count = 0;
  task0.next = &task1;  // 2 tasks queued on the same peer; only the head is peeked/batched this round
  scene.peers[0].bcastQueue.head = &task0;

  ncclDevFuncNameToId[ScheduleBcastTasksToPlan_DevFuncKey(NCCL_PROTO_SIMPLE)] = 0;

  EXPECT_EQ(ncclScheduleBcastTasksToPlan(scene.comm.get(), scene.plan.get(), nullptr), ncclSuccess);

  const std::vector<ncclTaskBcast*> drained = ScheduleBcastTasksToPlan_CollectBcastTaskQueue(scene.plan.get());
  ASSERT_EQ(drained.size(), 1u);  // TryDequeue removes one task per call, not the whole chain
  EXPECT_EQ(drained[0], &task0);
  EXPECT_EQ(scene.plan->nTasksBcast, 1);
  EXPECT_EQ(scene.comm->planner.nTasksBcast, 4);
  EXPECT_EQ(scene.peers[0].bcastQueue.head, &task1);  // task1 remains queued for a later call
}

TEST_F(SchedulerMicrotest, MakeSymmetricTaskList_TaskIsNull_SkipsDevrInitOnceAndReturnsSuccess) {
  MakeSymmetricTaskList_Scene scene;
  // Canary: if ncclDevrInitOnce were wrongly invoked despite task==nullptr, this config makes it fail loudly.
  scene.comm->symmetricSupport = true;
  scene.comm->peerInfo = nullptr;
  struct ncclTaskColl* remainTasksHead = reinterpret_cast<struct ncclTaskColl*>(0x1);

  EXPECT_EQ(ncclMakeSymmetricTaskList(scene.comm.get(), nullptr, nullptr, &remainTasksHead), ncclSuccess);
  EXPECT_EQ(remainTasksHead, nullptr);
}

// ncclDevrInitOnce is real dev_runtime.cc code (dev-runtime-test.cc compiles the real dev_runtime.cc into this
// same binary), not a stub -- promoting it would be wrong. Real cross-TU calls into it are safe now that
// ENABLE_WARP_SPEED is a binary-wide compile definition (test/host/CMakeLists.txt), not a per-TU #define.
TEST_F(SchedulerMicrotest, MakeSymmetricTaskList_TaskNotNull_CallsRealDevrInitOnceAndPropagatesItsError) {
  MakeSymmetricTaskList_Scene scene;
  scene.comm->symmetricSupport = true;
  scene.comm->peerInfo = nullptr;  // same canary as the task==nullptr test, but now task!=nullptr
  ncclTaskColl task{};
  task.func = ncclFuncBroadcast;
  struct ncclTaskColl* remainTasksHead = nullptr;

  EXPECT_EQ(ncclMakeSymmetricTaskList(scene.comm.get(), &task, nullptr, &remainTasksHead), ncclInternalError);
}

TEST_F(SchedulerMicrotest, MakeSymmetricTaskList_SymkAvailable_ReceivesTaskFieldsAndCorrectedSymkOp) {
  MakeSymmetricTaskList_Scene scene;
  ncclTaskColl task{};
  task.func = ncclFuncAllGather;
  task.datatype = ncclFloat16;
  task.count = 777;
  task.opHost = ncclAvg;
  task.opDev.op = ncclDevSum;  // symkRedOp(ncclAvg, ncclDevSum) == ncclDevSumPostDiv, not the raw ncclDevSum here

  ncclFunc_t capturedFunc = ncclFuncSend;
  int capturedRed = -1;
  ncclDataType_t capturedDtype = ncclInt32;
  size_t capturedCount = 0;
  ScopedHook symkHook(g_symkAvailable, [&](struct ncclComm*, ncclFunc_t coll, int red, ncclDataType_t ty, size_t c) {
    capturedFunc = coll;
    capturedRed = red;
    capturedDtype = ty;
    capturedCount = c;
    return false;
  });
  struct ncclTaskColl* remainTasksHead = nullptr;

  EXPECT_EQ(ncclMakeSymmetricTaskList(scene.comm.get(), &task, nullptr, &remainTasksHead), ncclSuccess);
  EXPECT_EQ(capturedFunc, ncclFuncAllGather);
  EXPECT_EQ(capturedRed, static_cast<int>(ncclDevSumPostDiv));
  EXPECT_EQ(capturedDtype, ncclFloat16);
  EXPECT_EQ(capturedCount, 777u);
  EXPECT_EQ(remainTasksHead, &task);
}

TEST_F(SchedulerMicrotest,
      MakeSymmetricTaskList_SymmetricTaskBetweenTwoRemainderTasks_AppendCorrectsStalePointer) {
  MakeSymmetricTaskList_Scene scene;
  scene.comm->planner.nTasksColl = 5;
  g_symRegType = ncclSymSendRegRecvReg;  // task2 (the middle task) accepts through the window check

  ncclTaskColl task3{};
  task3.func = ncclFuncBroadcast;
  task3.algMask = NCCL_TUNING_MASK_GENERAL_KERNELS;  // wantSym false: remainder
  ncclTaskColl task2{};
  task2.func = ncclFuncBroadcast;  // wantSym true (defaults): diverts into the symmetric bucket
  task2.next = &task3;
  ncclTaskColl task1{};
  task1.func = ncclFuncBroadcast;
  task1.algMask = NCCL_TUNING_MASK_GENERAL_KERNELS;  // wantSym false: remainder
  task1.next = &task2;  // input chain: task1(remainder) -> task2(symmetric, diverts) -> task3(remainder)
  struct ncclTaskColl* remainTasksHead = nullptr;

  // task1.next starts as &task2; only task3's append (remainTasksTail->next=task3) corrects it to &task3.
  EXPECT_EQ(ncclMakeSymmetricTaskList(scene.comm.get(), &task1, nullptr, &remainTasksHead), ncclInternalError);
  EXPECT_EQ(remainTasksHead, &task1);
  EXPECT_EQ(task1.next, &task3);  // would still be &task2 if the append line were dropped
  EXPECT_EQ(task3.next, nullptr);
  EXPECT_EQ(task2.next, nullptr);  // alone in its bucket
  EXPECT_EQ(scene.comm->planner.nTasksColl, 4);
}

TEST_F(SchedulerMicrotest, MakeSymmetricTaskList_RemainderTaskFollowedBySymmetricTask_NullsStaleNextPointer) {
  MakeSymmetricTaskList_Scene scene;
  scene.comm->planner.nTasksColl = 5;
  g_symRegType = ncclSymSendRegRecvReg;  // task2 (defaults otherwise) accepts through the window check
  ncclTaskColl task2{};
  task2.func = ncclFuncBroadcast;  // wantSym true: default symAvailable/algMask/windows all pass, func!=AllReduce
  ncclTaskColl task1{};
  task1.func = ncclFuncBroadcast;
  task1.algMask = NCCL_TUNING_MASK_GENERAL_KERNELS;  // wantSym false: rejected on algMask alone
  task1.next = &task2;  // input chain: task1 (remainder) is immediately followed by task2 (symmetric bucket)
  struct ncclTaskColl* remainTasksHead = nullptr;

  // task1.next starts as &task2; with no later remainder task, only the tail-nulling line clears it to nullptr.
  EXPECT_EQ(ncclMakeSymmetricTaskList(scene.comm.get(), &task1, nullptr, &remainTasksHead), ncclInternalError);
  EXPECT_EQ(remainTasksHead, &task1);
  EXPECT_EQ(task1.next, nullptr);  // would still be &task2 if the tail-nulling line were dropped
  EXPECT_EQ(task2.next, nullptr);  // alone in its bucket
  EXPECT_EQ(scene.comm->planner.nTasksColl, 4);
}

TEST_F(SchedulerMicrotest, MakeSymmetricTaskList_CfgAllowsSymkFalse_AlgMaskRestrictsAwayFromSym_GoesToRemainder) {
  MakeSymmetricTaskList_Scene scene;
  ncclTaskColl task{};
  task.func = ncclFuncBroadcast;
  task.algMask = NCCL_TUNING_MASK_GENERAL_KERNELS;  // nonzero, disjoint from NCCL_TUNING_MASK_SYM_KERNELS
  task.sendWin = reinterpret_cast<struct ncclDevrWindow*>(0xBADF00D);  // sentinel: window lookup must never run
  struct ncclTaskColl* remainTasksHead = nullptr;

  EXPECT_EQ(ncclMakeSymmetricTaskList(scene.comm.get(), &task, nullptr, &remainTasksHead), ncclSuccess);
  EXPECT_EQ(remainTasksHead, &task);
  EXPECT_EQ(task.next, nullptr);
  EXPECT_EQ(task.sendWin, reinterpret_cast<struct ncclDevrWindow*>(0xBADF00D));  // untouched: block was skipped
}

TEST_F(SchedulerMicrotest,
      MakeSymmetricTaskList_ForcedOverridesRestrictiveAlgMask_AcceptsThroughWindows_GoesToSymmetricBucket) {
  MakeSymmetricTaskList_Scene scene;
  scene.comm->planner.nTasksColl = 5;
  ncclTaskColl task{};
  task.func = ncclFuncBroadcast;
  task.algMask = NCCL_TUNING_MASK_GENERAL_KERNELS;  // would reject on its own, but forced below bypasses it
  scene.comm->tuningContext.forced[task.func] = 1;
  g_symRegType = ncclSymSendRegRecvReg;
  struct ncclTaskColl* remainTasksHead = nullptr;

  // foundSymm==true reaches Block 8's args-size guard next; workArgsBytes defaults to 0, so it always rejects.
  EXPECT_EQ(ncclMakeSymmetricTaskList(scene.comm.get(), &task, nullptr, &remainTasksHead), ncclInternalError);
  EXPECT_EQ(remainTasksHead, nullptr);
  EXPECT_EQ(task.next, nullptr);
  EXPECT_EQ(scene.comm->planner.nTasksColl, 4);
}

TEST_F(SchedulerMicrotest, MakeSymmetricTaskList_WindowRejection_DefaultRegType_GoesToRemainder) {
  MakeSymmetricTaskList_Scene scene;
  ncclTaskColl task{};
  task.func = ncclFuncBroadcast;
  // All defaults reach the window-rejection arm untouched: symAvailable/cfgAllowsSymk true, regType=reject.
  struct ncclTaskColl* remainTasksHead = nullptr;

  EXPECT_EQ(ncclMakeSymmetricTaskList(scene.comm.get(), &task, nullptr, &remainTasksHead), ncclSuccess);
  EXPECT_EQ(remainTasksHead, &task);
  EXPECT_EQ(task.next, nullptr);
}

TEST_F(SchedulerMicrotest, MakeSymmetricTaskList_AllReduceForcedOutDespiteGoodWindows_GoesToRemainder) {
  MakeSymmetricTaskList_Scene scene;
  ncclTaskColl task{};
  task.func = ncclFuncAllReduce;
  g_symRegType = ncclSymSendRegRecvReg;  // windows are GOOD here, isolating this from the window-rejection test
  struct ncclTaskColl* remainTasksHead = nullptr;

  EXPECT_EQ(ncclMakeSymmetricTaskList(scene.comm.get(), &task, nullptr, &remainTasksHead), ncclSuccess);
  EXPECT_EQ(remainTasksHead, &task);
  EXPECT_EQ(task.next, nullptr);
}

TEST_F(SchedulerMicrotest, MakeSymmetricTaskList_AllLocalChecksPass_NRanksOne_GoesToSymmetricBucket) {
  MakeSymmetricTaskList_Scene scene;  // nRanks=1: bootstrap-consensus block skipped by construction
  scene.comm->planner.nTasksColl = 5;
  ncclTaskColl task{};
  task.func = ncclFuncBroadcast;
  g_symRegType = ncclSymSendRegRecvReg;
  struct ncclTaskColl* remainTasksHead = nullptr;

  EXPECT_EQ(ncclMakeSymmetricTaskList(scene.comm.get(), &task, nullptr, &remainTasksHead), ncclInternalError);
  EXPECT_EQ(remainTasksHead, nullptr);
  EXPECT_EQ(task.next, nullptr);
  EXPECT_EQ(scene.comm->planner.nTasksColl, 4);
}

TEST_F(SchedulerMicrotest, MakeSymmetricTaskList_TwoTasksSameSymkOp_LifoChainUsesCorrectedOpNotRawOpDev) {
  MakeSymmetricTaskList_Scene scene;
  scene.comm->planner.nTasksColl = 5;
  g_symRegType = ncclSymSendRegRecvReg;

  ncclTaskColl task2{};
  task2.func = ncclFuncBroadcast;
  task2.datatype = ncclFloat32;
  task2.opHost = ncclSum;
  task2.opDev.op = ncclDevSumPostDiv;  // symkOp == ncclDevSumPostDiv directly
  ncclTaskColl task1{};
  task1.func = ncclFuncBroadcast;
  task1.datatype = ncclFloat32;
  task1.opHost = ncclAvg;
  task1.opDev.op = ncclDevSum;  // symkOp == ncclDevSumPostDiv too, via symkRedOp's averaging rule
  task1.next = &task2;          // input chain: task1 processed first, then task2

  struct ncclTaskColl* remainTasksHead = nullptr;

  EXPECT_EQ(ncclMakeSymmetricTaskList(scene.comm.get(), &task1, nullptr, &remainTasksHead), ncclInternalError);
  EXPECT_EQ(remainTasksHead, nullptr);
  EXPECT_EQ(task1.next, nullptr);   // task1 processed first: became its bucket's tail
  EXPECT_EQ(task2.next, &task1);    // task2 processed second: prepended, points at task1 -- same bucket as task1
  EXPECT_EQ(scene.comm->planner.nTasksColl, 3);  // both tasks counted as symmetric
}

TEST_F(SchedulerMicrotest, MakeSymmetricTaskList_TwoTasksDifferentSymkOp_LandInSeparateBuckets) {
  MakeSymmetricTaskList_Scene scene;
  scene.comm->planner.nTasksColl = 5;
  g_symRegType = ncclSymSendRegRecvReg;

  ncclTaskColl task2{};
  task2.func = ncclFuncBroadcast;
  task2.datatype = ncclFloat32;
  task2.opHost = ncclMax;
  task2.opDev.op = ncclDevMinMax;  // symkOp == ncclDevMinMax: distinct from task1's
  ncclTaskColl task1{};
  task1.func = ncclFuncBroadcast;
  task1.datatype = ncclFloat32;
  task1.opHost = ncclSum;
  task1.opDev.op = ncclDevSum;  // symkOp == ncclDevSum
  task1.next = &task2;

  struct ncclTaskColl* remainTasksHead = nullptr;

  EXPECT_EQ(ncclMakeSymmetricTaskList(scene.comm.get(), &task1, nullptr, &remainTasksHead), ncclInternalError);
  EXPECT_EQ(remainTasksHead, nullptr);
  EXPECT_EQ(task1.next, nullptr);  // singleton in its own bucket
  EXPECT_EQ(task2.next, nullptr);  // singleton in its own (different) bucket, NOT chained to task1
  EXPECT_EQ(scene.comm->planner.nTasksColl, 3);
}

TEST_F(SchedulerMicrotest, MakeSymmetricTaskList_TwoTasksSameFuncAndOpDifferentDatatype_LandInSeparateBuckets) {
  MakeSymmetricTaskList_Scene scene;
  scene.comm->planner.nTasksColl = 5;
  g_symRegType = ncclSymSendRegRecvReg;

  ncclTaskColl task2{};
  task2.func = ncclFuncBroadcast;
  task2.datatype = ncclFloat16;  // distinct from task1's
  task2.opHost = ncclSum;
  task2.opDev.op = ncclDevSum;
  ncclTaskColl task1{};
  task1.func = ncclFuncBroadcast;
  task1.datatype = ncclFloat32;
  task1.opHost = ncclSum;
  task1.opDev.op = ncclDevSum;
  task1.next = &task2;

  struct ncclTaskColl* remainTasksHead = nullptr;

  EXPECT_EQ(ncclMakeSymmetricTaskList(scene.comm.get(), &task1, nullptr, &remainTasksHead), ncclInternalError);
  EXPECT_EQ(remainTasksHead, nullptr);
  EXPECT_EQ(task1.next, nullptr);  // singleton in its own bucket
  EXPECT_EQ(task2.next, nullptr);  // singleton in its own (different) bucket, NOT chained to task1
  EXPECT_EQ(scene.comm->planner.nTasksColl, 3);
}

TEST_F(SchedulerMicrotest, MakeSymmetricTaskList_TwoTasksSameOpAndDatatypeDifferentFunc_LandInSeparateBuckets) {
  MakeSymmetricTaskList_Scene scene;
  scene.comm->planner.nTasksColl = 5;
  g_symRegType = ncclSymSendRegRecvReg;

  ncclTaskColl task2{};
  task2.func = ncclFuncAllGather;  // distinct from task1's
  task2.datatype = ncclFloat32;
  task2.opHost = ncclSum;
  task2.opDev.op = ncclDevSum;
  ncclTaskColl task1{};
  task1.func = ncclFuncBroadcast;
  task1.datatype = ncclFloat32;
  task1.opHost = ncclSum;
  task1.opDev.op = ncclDevSum;
  task1.next = &task2;

  struct ncclTaskColl* remainTasksHead2 = nullptr;

  EXPECT_EQ(ncclMakeSymmetricTaskList(scene.comm.get(), &task1, nullptr, &remainTasksHead2), ncclInternalError);
  EXPECT_EQ(remainTasksHead2, nullptr);
  EXPECT_EQ(task1.next, nullptr);  // singleton in its own bucket
  EXPECT_EQ(task2.next, nullptr);  // singleton in its own (different) bucket, NOT chained to task1
  EXPECT_EQ(scene.comm->planner.nTasksColl, 3);
}

TEST_F(SchedulerMicrotest, MakeSymmetricTaskList_NRanksTwoBootstrapNull_SkipsConsensusBlock) {
  MakeSymmetricTaskList_Scene scene;
  scene.comm->nRanks = 2;
  std::vector<int> rankToNode(2, 0);  // computeLsaSize (via ncclDevrInitOnce) reads this whenever nRanks>=2
  scene.comm->rankToNode = rankToNode.data();
  scene.comm->bootstrap = nullptr;  // half of the guard: consensus block must not run without this
  ncclTaskColl task{};
  task.func = ncclFuncBroadcast;
  ScopedHook symkHook(g_symkAvailable, [](struct ncclComm*, ncclFunc_t, int, ncclDataType_t, size_t) { return false; });
  ScopedHook bootstrapHook(g_devrBootstrapAllGather, [](void*, void*, int) { return ncclSuccess; });
  struct ncclTaskColl* remainTasksHead = nullptr;

  EXPECT_EQ(ncclMakeSymmetricTaskList(scene.comm.get(), &task, nullptr, &remainTasksHead), ncclSuccess);
  EXPECT_EQ(bootstrapHook.calls, 0);  // direct proof the consensus block's bootstrapAllGather never ran
  EXPECT_EQ(remainTasksHead, &task);
}

TEST_F(SchedulerMicrotest, MakeSymmetricTaskList_NRanksOneBootstrapNonNull_SkipsConsensusBlock) {
  MakeSymmetricTaskList_Scene scene;  // nRanks=1: the other half of the guard
  scene.comm->bootstrap = reinterpret_cast<void*>(0x1);
  ncclTaskColl task{};
  task.func = ncclFuncBroadcast;
  ScopedHook symkHook(g_symkAvailable, [](struct ncclComm*, ncclFunc_t, int, ncclDataType_t, size_t) { return false; });
  ScopedHook bootstrapHook(g_devrBootstrapAllGather, [](void*, void*, int) { return ncclSuccess; });
  struct ncclTaskColl* remainTasksHead = nullptr;

  EXPECT_EQ(ncclMakeSymmetricTaskList(scene.comm.get(), &task, nullptr, &remainTasksHead), ncclSuccess);
  EXPECT_EQ(bootstrapHook.calls, 0);
  EXPECT_EQ(remainTasksHead, &task);
}

TEST_F(SchedulerMicrotest, MakeSymmetricTaskList_ConsensusAllAgree_WantSymStaysTrue_GoesToSymmetricBucket) {
  MakeSymmetricTaskList_Scene scene;
  scene.comm->nRanks = 3;
  std::vector<int> rankToNode(3, 0);  // computeLsaSize (via ncclDevrInitOnce) reads this whenever nRanks>=2
  scene.comm->rankToNode = rankToNode.data();
  scene.comm->bootstrap = reinterpret_cast<void*>(0x1);
  scene.comm->planner.nTasksColl = 5;
  g_symRegType = ncclSymSendRegRecvReg;
  ncclTaskColl task{};
  task.func = ncclFuncBroadcast;

  uint8_t preGatherOwnFlag = 0xFF;
  ScopedHook bootstrapHook(g_devrBootstrapAllGather, [&](void*, void* allData, int) {
    uint8_t* buf = reinterpret_cast<uint8_t*>(allData);
    preGatherOwnFlag = buf[0];  // this rank's own wantSym flag, written before the call
    buf[0] = 1;
    buf[1] = 1;
    buf[2] = 1;  // every rank agrees
    return ncclSuccess;
  });
  struct ncclTaskColl* remainTasksHead = nullptr;

  EXPECT_EQ(ncclMakeSymmetricTaskList(scene.comm.get(), &task, nullptr, &remainTasksHead), ncclInternalError);
  EXPECT_EQ(bootstrapHook.calls, 1);
  EXPECT_EQ(preGatherOwnFlag, 1);  // proves flags[comm->rank] = wantSym?1:0 ran before bootstrapAllGather
  EXPECT_EQ(remainTasksHead, nullptr);
  EXPECT_EQ(task.next, nullptr);
  EXPECT_EQ(scene.comm->planner.nTasksColl, 4);
}

TEST_F(SchedulerMicrotest, MakeSymmetricTaskList_ConsensusOneDisagrees_WantSymFlipsFalse_GoesToRemainder) {
  MakeSymmetricTaskList_Scene scene;
  scene.comm->nRanks = 3;
  std::vector<int> rankToNode(3, 0);  // computeLsaSize (via ncclDevrInitOnce) reads this whenever nRanks>=2
  scene.comm->rankToNode = rankToNode.data();
  scene.comm->bootstrap = reinterpret_cast<void*>(0x1);
  g_symRegType = ncclSymSendRegRecvReg;
  ncclTaskColl task{};
  task.func = ncclFuncBroadcast;

  ScopedHook bootstrapHook(g_devrBootstrapAllGather, [&](void*, void* allData, int) {
    uint8_t* buf = reinterpret_cast<uint8_t*>(allData);
    buf[0] = 1;
    buf[1] = 0;  // rank 1 disagrees
    buf[2] = 1;
    return ncclSuccess;
  });
  struct ncclTaskColl* remainTasksHead = nullptr;

  EXPECT_EQ(ncclMakeSymmetricTaskList(scene.comm.get(), &task, nullptr, &remainTasksHead), ncclSuccess);
  EXPECT_EQ(bootstrapHook.calls, 1);
  EXPECT_EQ(remainTasksHead, &task);
  EXPECT_EQ(task.next, nullptr);
}

TEST_F(SchedulerMicrotest, MakeSymmetricTaskList_ArgsSizeGuard_TooSmall_ReturnsInternalErrorWithWarnLog) {
  MakeSymmetricTaskList_Scene scene;
  g_symRegType = ncclSymSendRegRecvReg;
  ncclTaskColl task{};
  task.func = ncclFuncBroadcast;
  task.datatype = ncclInt8;
  scene.comm->workArgsBytes = 0;  // deliberately below calcArgsSize(MAXCHANNELS, 1, false)'s minimum
  struct ncclTaskColl* remainTasksHead = nullptr;

  RcclUnitTesting::ScopedDebugLogging debugLogging(NCCL_LOG_WARN, NCCL_ALL);
  ncclResult_t result = ncclSuccess;
  const std::string log = RcclUnitTesting::CaptureLog(
      [&]() { result = ncclMakeSymmetricTaskList(scene.comm.get(), &task, nullptr, &remainTasksHead); });

  EXPECT_EQ(result, ncclInternalError);
  EXPECT_TRUE(RcclUnitTesting::LogHas(log, "Symmetric kernel args size"));
}

TEST_F(SchedulerMicrotest, MakeSymmetricTaskList_ArgsSizeGuard_SufficientlyLarge_ProceedsPastGuardAndCompletes) {
  MakeSymmetricTaskList_Scene scene;
  scene.comm->planner.nTasksColl = 5;
  g_symRegType = ncclSymSendRegRecvReg;
  ncclTaskColl task{};
  task.func = ncclFuncBroadcast;
  task.datatype = ncclInt8;
  scene.comm->workArgsBytes = static_cast<uint32_t>(ncclSymkDevWorkArgs::calcArgsSize(MAXCHANNELS, 1, false));
  struct ncclTaskColl* remainTasksHead = nullptr;

  // g_tuningCompute's default reports "no kernel found", so this batch safely falls back to the remainder.
  EXPECT_EQ(ncclMakeSymmetricTaskList(scene.comm.get(), &task, nullptr, &remainTasksHead), ncclSuccess);
  EXPECT_EQ(task.isSymLast, 1);  // single task in its bucket: task->next==nullptr disjunct
  EXPECT_EQ(remainTasksHead, &task);
  EXPECT_EQ(task.next, nullptr);
  EXPECT_EQ(scene.comm->planner.nTasksColl, 5);  // classification--'d then fallback++'d: round-trips to original
}

TEST_F(SchedulerMicrotest, MakeSymmetricTaskList_OuterCursorLoop_VisitsEveryDistinctBucket) {
  MakeSymmetricTaskList_Scene scene;
  scene.comm->planner.nTasksColl = 5;
  g_symRegType = ncclSymSendRegRecvReg;
  scene.comm->workArgsBytes = static_cast<uint32_t>(ncclSymkDevWorkArgs::calcArgsSize(MAXCHANNELS, 1, false));

  ncclTaskColl task2{};
  task2.func = ncclFuncAllGather;  // distinct bucket from task1 (different func)
  task2.datatype = ncclInt8;
  ncclTaskColl task1{};
  task1.func = ncclFuncBroadcast;
  task1.datatype = ncclInt8;
  task1.next = &task2;  // separate single-task buckets: no LIFO reordering to reason about

  struct ncclTaskColl* remainTasksHead = nullptr;
  EXPECT_EQ(ncclMakeSymmetricTaskList(scene.comm.get(), &task1, nullptr, &remainTasksHead), ncclSuccess);
  EXPECT_EQ(task1.isSymLast, 1);  // only true if the cursor loop actually visited task1's bucket
  EXPECT_EQ(task2.isSymLast, 1);  // ...and task2's bucket too, proving both cursor iterations ran
  EXPECT_EQ(remainTasksHead, &task1);  // cursor order follows first-time bucket creation order
  EXPECT_EQ(task1.next, &task2);
  EXPECT_EQ(task2.next, nullptr);
  EXPECT_EQ(scene.comm->planner.nTasksColl, 5);
}

TEST_F(SchedulerMicrotest, MakeSymmetricTaskList_BatchBoundary_ConfigBoundary_ForcesEarlyIsSymLastDespiteRoom) {
  MakeSymmetricTaskList_Scene scene;
  scene.comm->planner.nTasksColl = 5;
  g_symRegType = ncclSymSendRegRecvReg;
  scene.comm->workArgsBytes = static_cast<uint32_t>(ncclSymkDevWorkArgs::calcArgsSize(MAXCHANNELS, 5, false));

  ncclTaskColl task1{};  // classified first -> becomes the bucket's tail (processed second by Block 8)
  task1.func = ncclFuncBroadcast;
  task1.datatype = ncclInt8;
  ncclTaskColl task2{};  // classified second -> becomes the bucket's head (processed first by Block 8)
  task2.func = ncclFuncBroadcast;
  task2.datatype = ncclInt8;
  task2.aggIsolate = true;  // configBoundary fires via THIS (current, first-processed) task's own flag
  task1.next = &task2;      // classification input order: task1 then task2 (LIFO reverses processing order)

  struct ncclTaskColl* remainTasksHead = nullptr;
  EXPECT_EQ(ncclMakeSymmetricTaskList(scene.comm.get(), &task1, nullptr, &remainTasksHead), ncclSuccess);
  EXPECT_EQ(task2.isSymLast, 1);  // ends its own singleton batch despite task2.next(=task1)!=nullptr and no budget
  EXPECT_EQ(task1.isSymLast, 1);  // separate, later batch: ends naturally (task1.next==nullptr)
  EXPECT_EQ(remainTasksHead, &task2);  // task2's batch is processed (and falls back) before task1's
  EXPECT_EQ(task2.next, &task1);
  EXPECT_EQ(task1.next, nullptr);
  EXPECT_EQ(scene.comm->planner.nTasksColl, 5);
}

TEST_F(SchedulerMicrotest, MakeSymmetricTaskList_BatchBoundary_ConfigBoundary_NextTaskIsolated_AlsoEndsBatch) {
  MakeSymmetricTaskList_Scene scene;
  scene.comm->planner.nTasksColl = 5;
  g_symRegType = ncclSymSendRegRecvReg;
  scene.comm->workArgsBytes = static_cast<uint32_t>(ncclSymkDevWorkArgs::calcArgsSize(MAXCHANNELS, 5, false));

  ncclTaskColl task1{};  // classified first -> tail (processed second): its OWN aggIsolate isolates it too
  task1.func = ncclFuncBroadcast;
  task1.datatype = ncclInt8;
  task1.aggIsolate = true;
  ncclTaskColl task2{};  // classified second -> head (processed first): ends here only because task1 (its
  task2.func = ncclFuncBroadcast;  // Block-8-processing-order .next) is isolated -- task2 itself is not.
  task2.datatype = ncclInt8;
  task1.next = &task2;

  struct ncclTaskColl* remainTasksHead = nullptr;
  EXPECT_EQ(ncclMakeSymmetricTaskList(scene.comm.get(), &task1, nullptr, &remainTasksHead), ncclSuccess);
  EXPECT_EQ(task2.isSymLast, 1);  // configBoundary via task->next->aggIsolate, the OR's short-circuited half
  EXPECT_EQ(task1.isSymLast, 1);  // task1's own aggIsolate then isolates its own (separate) singleton batch
  EXPECT_EQ(remainTasksHead, &task2);
  EXPECT_EQ(task2.next, &task1);
  EXPECT_EQ(task1.next, nullptr);
  EXPECT_EQ(scene.comm->planner.nTasksColl, 5);
}

TEST_F(SchedulerMicrotest, MakeSymmetricTaskList_BatchBoundary_ArgsSizeBudgetExhausted_SplitsIntoTwoBatches) {
  MakeSymmetricTaskList_Scene scene;
  scene.comm->planner.nTasksColl = 5;
  g_symRegType = ncclSymSendRegRecvReg;
  // Room for exactly 2 works per batch, not 3: forces a break after the 2nd task of a 3-task bucket.
  scene.comm->workArgsBytes = static_cast<uint32_t>(ncclSymkDevWorkArgs::calcArgsSize(MAXCHANNELS, 2, false));

  // Bucket is LIFO (last classified = head = first processed), so classify taskThird, taskSecond, taskFirst.
  ncclTaskColl taskFirst{};
  taskFirst.func = ncclFuncBroadcast;
  taskFirst.datatype = ncclInt8;
  ncclTaskColl taskSecond{};
  taskSecond.func = ncclFuncBroadcast;
  taskSecond.datatype = ncclInt8;
  taskSecond.next = &taskFirst;
  ncclTaskColl taskThird{};
  taskThird.func = ncclFuncBroadcast;
  taskThird.datatype = ncclInt8;
  taskThird.next = &taskSecond;

  struct ncclTaskColl* remainTasksHead = nullptr;
  EXPECT_EQ(ncclMakeSymmetricTaskList(scene.comm.get(), &taskThird, nullptr, &remainTasksHead), ncclSuccess);
  EXPECT_EQ(taskFirst.isSymLast, 0);   // continues: room remains and neither next==nullptr nor configBoundary
  EXPECT_EQ(taskSecond.isSymLast, 1);  // budget for a 3rd work would exceed workArgsBytes: batch ends here
  EXPECT_EQ(taskThird.isSymLast, 1);   // new batch (outer while(task!=NULL) re-enters): ends naturally
  // Remainder order matches Block 8's own processing order: {taskFirst, taskSecond} first, {taskThird} second.
  EXPECT_EQ(remainTasksHead, &taskFirst);
  EXPECT_EQ(taskFirst.next, &taskSecond);
  EXPECT_EQ(taskSecond.next, &taskThird);
  EXPECT_EQ(taskThird.next, nullptr);
  EXPECT_EQ(scene.comm->planner.nTasksColl, 5);  // 3 classification-- + 3 fallback++ round-trips to original
}

TEST_F(SchedulerMicrotest, MakeSymmetricTaskList_TuningInput_BasicScalarFieldsWiring) {
  MakeSymmetricTaskList_Scene scene;
  g_symRegType = ncclSymSendRegRecvReg;
  ncclTaskColl task{};
  task.func = ncclFuncBroadcast;
  task.datatype = ncclInt8;
  task.opHost = ncclAvg;
  task.opDev.op = ncclDevSum;  // symkOp == ncclDevSumPostDiv, distinct from the raw opDev.op
  task.count = 777;
  task.winRegType = static_cast<ncclSymRegType_t>(0xBAD);  // sentinel, overwritten by the classification stage
  task.minCTAs = 2;
  task.maxCTAs = 8;
  task.CTAPolicy = 3;
  scene.comm->workArgsBytes = static_cast<uint32_t>(ncclSymkDevWorkArgs::calcArgsSize(MAXCHANNELS, 5, false));

  struct ncclTuningInput_t captured {};
  ScopedHook tuningHook(g_tuningCompute, [&](struct ncclTuningInput_t* input, struct ncclTuningResult_t*) {
    captured = *input;
    return ncclSuccess;
  });
  struct ncclTaskColl* remainTasksHead = nullptr;

  EXPECT_EQ(ncclMakeSymmetricTaskList(scene.comm.get(), &task, nullptr, &remainTasksHead), ncclSuccess);
  EXPECT_EQ(tuningHook.calls, 1);
  EXPECT_EQ(captured.comm, scene.comm.get());
  EXPECT_EQ(captured.func, ncclFuncBroadcast);
  EXPECT_EQ(captured.redOp, ncclAvg);
  EXPECT_EQ(captured.devRedOp, ncclDevSumPostDiv);  // wired from symkOp, not the raw task->opDev.op
  EXPECT_EQ(captured.datatype, ncclInt8);
  EXPECT_EQ(captured.numPipeOps, 0);
  EXPECT_EQ(captured.count, 777u);
  EXPECT_EQ(captured.winRegType, ncclSymSendRegRecvReg);  // the classification stage's value, not the sentinel
  EXPECT_EQ(captured.minCTAs, 2);
  EXPECT_EQ(captured.maxCTAs, 8);
  EXPECT_EQ(captured.CTAPolicy, 3);
}

TEST_F(SchedulerMicrotest, MakeSymmetricTaskList_TuningInput_AggregatesAcrossBatch_AndFallsBackWholeBatch) {
  MakeSymmetricTaskList_Scene scene;
  scene.comm->planner.nTasksColl = 5;
  g_symRegType = ncclSymSendRegRecvReg;
  scene.comm->workArgsBytes = static_cast<uint32_t>(ncclSymkDevWorkArgs::calcArgsSize(MAXCHANNELS, 5, false));

  // Bucket is LIFO (last classified = head = first processed), so classify secondTask before headTask.
  ncclTaskColl headTask{};
  headTask.func = ncclFuncBroadcast;
  headTask.datatype = ncclInt8;
  headTask.count = 500;   // alignUp(500, cellCount=1024) = 1024
  ncclTaskColl secondTask{};
  secondTask.func = ncclFuncBroadcast;
  secondTask.datatype = ncclInt8;
  secondTask.count = 2000;  // alignUp(2000, 1024) = 2048
  secondTask.next = &headTask;

  struct ncclTuningInput_t captured {};
  ScopedHook tuningHook(g_tuningCompute, [&](struct ncclTuningInput_t* input, struct ncclTuningResult_t*) {
    captured = *input;
    return ncclSuccess;
  });
  struct ncclTaskColl* remainTasksHead = nullptr;

  EXPECT_EQ(ncclMakeSymmetricTaskList(scene.comm.get(), &secondTask, nullptr, &remainTasksHead), ncclSuccess);
  EXPECT_EQ(captured.nWorks, 2);
  EXPECT_EQ(captured.nBytes, (1024u + 2048u) * 1u);  // countTotal * ncclTypeSize(ncclInt8)
  EXPECT_EQ(captured.countMax, 2000u);               // largest RAW count, not the aligned batch total
  EXPECT_EQ(captured.count, 500u);                   // headTask's own (unaligned) count
  // Neither task found a kernel (default g_tuningCompute-adjacent behavior): the whole batch falls back.
  EXPECT_EQ(remainTasksHead, &headTask);
  EXPECT_EQ(headTask.next, &secondTask);
  EXPECT_EQ(secondTask.next, nullptr);
  EXPECT_EQ(scene.comm->planner.nTasksColl, 5);
}

TEST_F(SchedulerMicrotest, MakeSymmetricTaskList_TuningInput_TuningMaskDefault_WhenEffAlgMaskZero) {
  MakeSymmetricTaskList_Scene scene;
  g_symRegType = ncclSymSendRegRecvReg;
  ncclTaskColl task{};
  task.func = ncclFuncBroadcast;
  task.datatype = ncclInt8;
  scene.comm->workArgsBytes = static_cast<uint32_t>(ncclSymkDevWorkArgs::calcArgsSize(MAXCHANNELS, 5, false));

  uint64_t capturedMask = 0;
  ScopedHook tuningHook(g_tuningCompute, [&](struct ncclTuningInput_t* input, struct ncclTuningResult_t*) {
    capturedMask = input->tuningMask;
    return ncclSuccess;
  });
  struct ncclTaskColl* remainTasksHead = nullptr;

  EXPECT_EQ(ncclMakeSymmetricTaskList(scene.comm.get(), &task, nullptr, &remainTasksHead), ncclSuccess);
  EXPECT_EQ(capturedMask, static_cast<uint64_t>(NCCL_TUNING_MASK_SYM_KERNELS));
}

TEST_F(SchedulerMicrotest, MakeSymmetricTaskList_TuningInput_TuningMaskOverridden_WhenSymkMaskBitsNonZero) {
  MakeSymmetricTaskList_Scene scene;
  g_symRegType = ncclSymSendRegRecvReg;
  ncclTaskColl task{};
  task.func = ncclFuncBroadcast;
  task.datatype = ncclInt8;
  // A single sym-range bit, not the whole mask, so the overridden value differs from the pre-override default.
  const uint64_t kOneSymBit = 1ull << NCCL_TUNING_SYM_KERNEL_ID_OFFSET;
  task.algMask = kOneSymBit;
  scene.comm->workArgsBytes = static_cast<uint32_t>(ncclSymkDevWorkArgs::calcArgsSize(MAXCHANNELS, 5, false));

  uint64_t capturedMask = 0;
  ScopedHook tuningHook(g_tuningCompute, [&](struct ncclTuningInput_t* input, struct ncclTuningResult_t*) {
    capturedMask = input->tuningMask;
    return ncclSuccess;
  });
  struct ncclTaskColl* remainTasksHead = nullptr;

  EXPECT_EQ(ncclMakeSymmetricTaskList(scene.comm.get(), &task, nullptr, &remainTasksHead), ncclSuccess);
  EXPECT_EQ(capturedMask, kOneSymBit);  // overridden to symkMask, distinct from the SYM_KERNELS default
}

// No "effAlgMask != 0 but symkMask == 0" test: Block 7's wantSym gate already proved that's unreachable here.

TEST_F(SchedulerMicrotest, MakeSymmetricTaskList_TuningInput_NvlsSupport_ViaNcclNvlsSupported) {
  MakeSymmetricTaskList_Scene scene;
  g_symRegType = ncclSymSendRegRecvReg;
  scene.comm->nvlsSupport = 1;
  ncclTaskColl task{};
  task.func = ncclFuncBroadcast;  // not AllGather: the bypass operand must not be what makes this true
  task.datatype = ncclInt32;
  task.opDev.op = ncclDevSum;  // ncclNvlsSupported(ncclDevSum, ncclInt32) == true
  scene.comm->workArgsBytes = static_cast<uint32_t>(ncclSymkDevWorkArgs::calcArgsSize(MAXCHANNELS, 5, false));

  int capturedNvlsSupport = -1;
  ScopedHook tuningHook(g_tuningCompute, [&](struct ncclTuningInput_t* input, struct ncclTuningResult_t*) {
    capturedNvlsSupport = input->nvlsSupport;
    return ncclSuccess;
  });
  struct ncclTaskColl* remainTasksHead = nullptr;

  EXPECT_EQ(ncclMakeSymmetricTaskList(scene.comm.get(), &task, nullptr, &remainTasksHead), ncclSuccess);
  EXPECT_TRUE(capturedNvlsSupport);
}

TEST_F(SchedulerMicrotest, MakeSymmetricTaskList_TuningInput_NvlsSupport_ViaAllGatherBypass) {
  MakeSymmetricTaskList_Scene scene;
  g_symRegType = ncclSymSendRegRecvReg;
  scene.comm->nvlsSupport = 1;
  ncclTaskColl task{};
  task.func = ncclFuncAllGather;  // bypasses ncclNvlsSupported entirely
  task.datatype = ncclInt8;       // ncclNvlsSupported(_, ncclInt8) == false: proves the bypass, not this
  scene.comm->workArgsBytes = static_cast<uint32_t>(ncclSymkDevWorkArgs::calcArgsSize(MAXCHANNELS, 5, false));

  int capturedNvlsSupport = -1;
  ScopedHook tuningHook(g_tuningCompute, [&](struct ncclTuningInput_t* input, struct ncclTuningResult_t*) {
    capturedNvlsSupport = input->nvlsSupport;
    return ncclSuccess;
  });
  struct ncclTaskColl* remainTasksHead = nullptr;

  EXPECT_EQ(ncclMakeSymmetricTaskList(scene.comm.get(), &task, nullptr, &remainTasksHead), ncclSuccess);
  EXPECT_TRUE(capturedNvlsSupport);
}

TEST_F(SchedulerMicrotest, MakeSymmetricTaskList_TuningInput_NvlsSupport_FalseWhenCommDoesNotSupportNvls) {
  MakeSymmetricTaskList_Scene scene;
  g_symRegType = ncclSymSendRegRecvReg;
  scene.comm->nvlsSupport = 0;  // short-circuits the whole && regardless of func/datatype
  ncclTaskColl task{};
  task.func = ncclFuncAllGather;
  task.datatype = ncclInt32;
  task.opDev.op = ncclDevSum;
  scene.comm->workArgsBytes = static_cast<uint32_t>(ncclSymkDevWorkArgs::calcArgsSize(MAXCHANNELS, 5, false));

  int capturedNvlsSupport = -1;
  ScopedHook tuningHook(g_tuningCompute, [&](struct ncclTuningInput_t* input, struct ncclTuningResult_t*) {
    capturedNvlsSupport = input->nvlsSupport;
    return ncclSuccess;
  });
  struct ncclTaskColl* remainTasksHead = nullptr;

  EXPECT_EQ(ncclMakeSymmetricTaskList(scene.comm.get(), &task, nullptr, &remainTasksHead), ncclSuccess);
  EXPECT_FALSE(capturedNvlsSupport);
}

TEST_F(SchedulerMicrotest, MakeSymmetricTaskList_TuningInput_CollNetSupportAndRegBuff_WiredFromTheirOwnFakes) {
  MakeSymmetricTaskList_Scene scene;
  g_symRegType = ncclSymSendRegRecvReg;
  ncclTaskColl task{};
  task.func = ncclFuncBroadcast;
  task.datatype = ncclInt8;
  scene.comm->workArgsBytes = static_cast<uint32_t>(ncclSymkDevWorkArgs::calcArgsSize(MAXCHANNELS, 5, false));

  ScopedHook collNetHook(g_getCollNetSupport, [](struct ncclComm*, struct ncclTaskColl*, int* out) {
    *out = 7;
    return ncclSuccess;
  });
  ScopedHook regBuffHook(g_getRegBuff, [](struct ncclComm*, struct ncclTaskColl*, int* out) {
    *out = 13;
    return ncclSuccess;
  });
  int capturedCollNetSupport = -1, capturedRegBuff = -1;
  ScopedHook tuningHook(g_tuningCompute, [&](struct ncclTuningInput_t* input, struct ncclTuningResult_t*) {
    capturedCollNetSupport = input->collNetSupport;
    capturedRegBuff = input->regBuff;
    return ncclSuccess;
  });
  struct ncclTaskColl* remainTasksHead = nullptr;

  EXPECT_EQ(ncclMakeSymmetricTaskList(scene.comm.get(), &task, nullptr, &remainTasksHead), ncclSuccess);
  EXPECT_EQ(collNetHook.calls, 1);
  EXPECT_EQ(regBuffHook.calls, 1);
  EXPECT_EQ(capturedCollNetSupport, 7);
  EXPECT_EQ(capturedRegBuff, 13);
}

TEST_F(SchedulerMicrotest, MakeSymmetricTaskList_TuningInput_SymAligned16B_TrueWhenBuffersAlign) {
  MakeSymmetricTaskList_Scene scene;
  g_symRegType = ncclSymSendRegRecvReg;
  ncclTaskColl task{};
  task.func = ncclFuncBroadcast;
  task.datatype = ncclInt8;
  task.sendbuff = reinterpret_cast<void*>(0x1030);  // matches SymBatchAligned16B_SingleTaskNoWindowsAligned
  task.recvbuff = reinterpret_cast<void*>(0x1020);  // offset diff 0x10: divisible by 16
  scene.comm->workArgsBytes = static_cast<uint32_t>(ncclSymkDevWorkArgs::calcArgsSize(MAXCHANNELS, 5, false));

  bool capturedAligned = false;
  ScopedHook tuningHook(g_tuningCompute, [&](struct ncclTuningInput_t* input, struct ncclTuningResult_t*) {
    capturedAligned = input->symAligned16B;
    return ncclSuccess;
  });
  struct ncclTaskColl* remainTasksHead = nullptr;

  EXPECT_EQ(ncclMakeSymmetricTaskList(scene.comm.get(), &task, nullptr, &remainTasksHead), ncclSuccess);
  EXPECT_TRUE(capturedAligned);
}

TEST_F(SchedulerMicrotest, MakeSymmetricTaskList_TuningInput_SymAligned16B_FalseWhenBuffersMisalign) {
  MakeSymmetricTaskList_Scene scene;
  g_symRegType = ncclSymSendRegRecvReg;
  ncclTaskColl task{};
  task.func = ncclFuncBroadcast;
  task.datatype = ncclInt8;
  task.sendbuff = reinterpret_cast<void*>(0x1028);  // matches SymBatchAligned16B_SingleTaskNoWindowsMisaligned
  task.recvbuff = reinterpret_cast<void*>(0x1020);  // offset diff 8: not divisible by 16
  scene.comm->workArgsBytes = static_cast<uint32_t>(ncclSymkDevWorkArgs::calcArgsSize(MAXCHANNELS, 5, false));

  bool capturedAligned = true;
  ScopedHook tuningHook(g_tuningCompute, [&](struct ncclTuningInput_t* input, struct ncclTuningResult_t*) {
    capturedAligned = input->symAligned16B;
    return ncclSuccess;
  });
  struct ncclTaskColl* remainTasksHead = nullptr;

  EXPECT_EQ(ncclMakeSymmetricTaskList(scene.comm.get(), &task, nullptr, &remainTasksHead), ncclSuccess);
  EXPECT_FALSE(capturedAligned);
}

TEST_F(SchedulerMicrotest, MakeSymmetricTaskList_HardErrorBranch_AllConditionsTrue_ReturnsInvalidArgumentWithWarn) {
  MakeSymmetricTaskList_Scene scene;
  g_symRegType = ncclSymSendRegRecvReg;
  ncclTaskColl task{};
  task.func = ncclFuncBroadcast;
  task.datatype = ncclInt8;
  task.algMask = NCCL_TUNING_MASK_SYM_KERNELS;  // only sym bits: satisfies both mask conditions at once
  task.forceAlgSelection = 1;
  scene.comm->workArgsBytes = static_cast<uint32_t>(ncclSymkDevWorkArgs::calcArgsSize(MAXCHANNELS, 5, false));
  // g_tuningCompute default: kernelId stays ncclSymkKernelId_Count -- the 5th condition this test needs true.
  struct ncclTaskColl* remainTasksHead = nullptr;

  RcclUnitTesting::ScopedDebugLogging debugLogging(NCCL_LOG_WARN, NCCL_ALL);
  ncclResult_t result = ncclSuccess;
  const std::string log = RcclUnitTesting::CaptureLog(
      [&]() { result = ncclMakeSymmetricTaskList(scene.comm.get(), &task, nullptr, &remainTasksHead); });

  EXPECT_EQ(result, ncclInvalidArgument);
  EXPECT_TRUE(RcclUnitTesting::LogHas(log, "algSelection names only symmetric kernel"));
}

TEST_F(SchedulerMicrotest, MakeSymmetricTaskList_HardErrorBranch_EffAlgMaskZero_SkipsAndFallsBack) {
  MakeSymmetricTaskList_Scene scene;
  scene.comm->planner.nTasksColl = 5;
  g_symRegType = ncclSymSendRegRecvReg;
  ncclTaskColl task{};
  task.func = ncclFuncBroadcast;
  task.datatype = ncclInt8;
  // task.algMask left at 0 (default): effAlgMask == 0, so the hard-error branch's 2nd condition is false.
  scene.comm->workArgsBytes = static_cast<uint32_t>(ncclSymkDevWorkArgs::calcArgsSize(MAXCHANNELS, 5, false));
  struct ncclTaskColl* remainTasksHead = nullptr;

  EXPECT_EQ(ncclMakeSymmetricTaskList(scene.comm.get(), &task, nullptr, &remainTasksHead), ncclSuccess);
  EXPECT_EQ(remainTasksHead, &task);  // kernelId==Count (default) still falls back safely on its own
  EXPECT_EQ(scene.comm->planner.nTasksColl, 5);
}

TEST_F(SchedulerMicrotest, MakeSymmetricTaskList_HardErrorBranch_NoSymkBitsInMask_SkipsAndFallsBack) {
  MakeSymmetricTaskList_Scene scene;
  scene.comm->planner.nTasksColl = 5;
  g_symRegType = ncclSymSendRegRecvReg;
  ncclTaskColl task{};
  task.func = ncclFuncBroadcast;
  task.datatype = ncclInt8;
  task.algMask = NCCL_TUNING_MASK_GENERAL_KERNELS;  // nonzero, but (effAlgMask & SYM_MASK) == 0
  task.forceAlgSelection = 1;
  scene.comm->workArgsBytes = static_cast<uint32_t>(ncclSymkDevWorkArgs::calcArgsSize(MAXCHANNELS, 5, false));
  struct ncclTaskColl* remainTasksHead = nullptr;

  EXPECT_EQ(ncclMakeSymmetricTaskList(scene.comm.get(), &task, nullptr, &remainTasksHead), ncclSuccess);
  EXPECT_EQ(remainTasksHead, &task);
  EXPECT_EQ(scene.comm->planner.nTasksColl, 5);
}

TEST_F(SchedulerMicrotest, MakeSymmetricTaskList_HardErrorBranch_HasGeneralBitsInMask_SkipsAndFallsBack) {
  MakeSymmetricTaskList_Scene scene;
  scene.comm->planner.nTasksColl = 5;
  g_symRegType = ncclSymSendRegRecvReg;
  ncclTaskColl task{};
  task.func = ncclFuncBroadcast;
  task.datatype = ncclInt8;
  // Has sym bits (would satisfy condition 3) but also general bits, so condition 4 ((mask&GENERAL)==0) is false.
  task.algMask = NCCL_TUNING_MASK_SYM_KERNELS | NCCL_TUNING_MASK_GENERAL_KERNELS;
  task.forceAlgSelection = 1;
  scene.comm->workArgsBytes = static_cast<uint32_t>(ncclSymkDevWorkArgs::calcArgsSize(MAXCHANNELS, 5, false));
  struct ncclTaskColl* remainTasksHead = nullptr;

  EXPECT_EQ(ncclMakeSymmetricTaskList(scene.comm.get(), &task, nullptr, &remainTasksHead), ncclSuccess);
  EXPECT_EQ(remainTasksHead, &task);
  EXPECT_EQ(scene.comm->planner.nTasksColl, 5);
}

TEST_F(SchedulerMicrotest, MakeSymmetricTaskList_HardErrorBranch_ForceAlgSelectionFalse_SkipsAndFallsBack) {
  MakeSymmetricTaskList_Scene scene;
  scene.comm->planner.nTasksColl = 5;
  g_symRegType = ncclSymSendRegRecvReg;
  ncclTaskColl task{};
  task.func = ncclFuncBroadcast;
  task.datatype = ncclInt8;
  task.algMask = NCCL_TUNING_MASK_SYM_KERNELS;
  // task.forceAlgSelection left at 0 (default): the 5th condition is false.
  scene.comm->workArgsBytes = static_cast<uint32_t>(ncclSymkDevWorkArgs::calcArgsSize(MAXCHANNELS, 5, false));
  struct ncclTaskColl* remainTasksHead = nullptr;

  EXPECT_EQ(ncclMakeSymmetricTaskList(scene.comm.get(), &task, nullptr, &remainTasksHead), ncclSuccess);
  EXPECT_EQ(remainTasksHead, &task);
  EXPECT_EQ(scene.comm->planner.nTasksColl, 5);
}

TEST_F(SchedulerMicrotest, MakeSymmetricTaskList_HardErrorBranch_KernelIdFound_SkipsAndProceedsSafely) {
  MakeSymmetricTaskList_Scene scene;
  g_symRegType = ncclSymSendRegRecvReg;
  ncclTaskColl task{};
  task.func = ncclFuncBroadcast;
  task.datatype = ncclInt8;
  task.algMask = NCCL_TUNING_MASK_SYM_KERNELS;  // every other condition true
  task.forceAlgSelection = 1;
  scene.comm->workArgsBytes = static_cast<uint32_t>(ncclSymkDevWorkArgs::calcArgsSize(MAXCHANNELS, 5, false));
  ScopedHook tuningHook(g_tuningCompute, [](struct ncclTuningInput_t*, struct ncclTuningResult_t* result) {
    result->symKernelId = ncclSymkKernelId_AllGather_LL;  // found: the 1st condition (kernelId==Count) is false
    result->nChannels = 1;
    result->nWarps = 1;
    return ncclSuccess;
  });
  struct ncclTaskColl* remainTasksHead = nullptr;

  EXPECT_EQ(ncclMakeSymmetricTaskList(scene.comm.get(), &task, nullptr, &remainTasksHead), ncclSuccess);
}

TEST_F(SchedulerMicrotest, MakeSymmetricTaskList_InfoLoggingBranch_EffAlgMaskNonZeroAndKernelFound_LogsInfo) {
  MakeSymmetricTaskList_Scene scene;
  g_symRegType = ncclSymSendRegRecvReg;
  ncclTaskColl task{};
  task.func = ncclFuncBroadcast;
  task.datatype = ncclInt8;
  task.algMask = NCCL_TUNING_MASK_SYM_KERNELS;
  scene.comm->workArgsBytes = static_cast<uint32_t>(ncclSymkDevWorkArgs::calcArgsSize(MAXCHANNELS, 5, false));
  ScopedHook tuningHook(g_tuningCompute, [](struct ncclTuningInput_t*, struct ncclTuningResult_t* result) {
    result->symKernelId = ncclSymkKernelId_AllGather_LL;
    result->nChannels = 1;
    result->nWarps = 1;
    return ncclSuccess;
  });
  struct ncclTaskColl* remainTasksHead = nullptr;

  RcclUnitTesting::ScopedDebugLogging debugLogging(NCCL_LOG_INFO, NCCL_TUNING);
  ncclResult_t result = ncclInvalidArgument;
  const std::string log = RcclUnitTesting::CaptureLog(
      [&]() { result = ncclMakeSymmetricTaskList(scene.comm.get(), &task, nullptr, &remainTasksHead); });

  EXPECT_EQ(result, ncclSuccess);
  EXPECT_TRUE(RcclUnitTesting::LogHas(log, "algSelection: sym-kernel picked within the selected set"));
}

TEST_F(SchedulerMicrotest, MakeSymmetricTaskList_InfoLoggingBranch_EffAlgMaskZero_DoesNotLogEvenWithKernelFound) {
  MakeSymmetricTaskList_Scene scene;
  g_symRegType = ncclSymSendRegRecvReg;
  ncclTaskColl task{};
  task.func = ncclFuncBroadcast;
  task.datatype = ncclInt8;
  // task.algMask left at 0: effAlgMask == 0, so the INFO condition's first operand is false.
  scene.comm->workArgsBytes = static_cast<uint32_t>(ncclSymkDevWorkArgs::calcArgsSize(MAXCHANNELS, 5, false));
  ScopedHook tuningHook(g_tuningCompute, [](struct ncclTuningInput_t*, struct ncclTuningResult_t* result) {
    result->symKernelId = ncclSymkKernelId_AllGather_LL;
    result->nChannels = 1;
    result->nWarps = 1;
    return ncclSuccess;
  });
  struct ncclTaskColl* remainTasksHead = nullptr;

  RcclUnitTesting::ScopedDebugLogging debugLogging(NCCL_LOG_INFO, NCCL_TUNING);
  ncclResult_t result = ncclInvalidArgument;
  const std::string log = RcclUnitTesting::CaptureLog(
      [&]() { result = ncclMakeSymmetricTaskList(scene.comm.get(), &task, nullptr, &remainTasksHead); });

  EXPECT_EQ(result, ncclSuccess);
  EXPECT_FALSE(RcclUnitTesting::LogHas(log, "algSelection:"));
}

TEST_F(SchedulerMicrotest, MakeSymmetricTaskList_InfoLoggingBranch_KernelIdCount_DoesNotLogEvenWithEffAlgMask) {
  MakeSymmetricTaskList_Scene scene;
  scene.comm->planner.nTasksColl = 5;
  g_symRegType = ncclSymSendRegRecvReg;
  ncclTaskColl task{};
  task.func = ncclFuncBroadcast;
  task.datatype = ncclInt8;
  task.algMask = NCCL_TUNING_MASK_SYM_KERNELS;
  scene.comm->workArgsBytes = static_cast<uint32_t>(ncclSymkDevWorkArgs::calcArgsSize(MAXCHANNELS, 5, false));
  // g_tuningCompute default: kernelId stays ncclSymkKernelId_Count, so the INFO condition's 2nd operand is false.
  struct ncclTaskColl* remainTasksHead = nullptr;

  RcclUnitTesting::ScopedDebugLogging debugLogging(NCCL_LOG_INFO, NCCL_TUNING);
  ncclResult_t result = ncclInvalidArgument;
  const std::string log = RcclUnitTesting::CaptureLog(
      [&]() { result = ncclMakeSymmetricTaskList(scene.comm.get(), &task, nullptr, &remainTasksHead); });

  EXPECT_EQ(result, ncclSuccess);
  EXPECT_FALSE(RcclUnitTesting::LogHas(log, "algSelection:"));
  EXPECT_EQ(remainTasksHead, &task);  // falls back via the kernelId==Count disjunct, same as always
  EXPECT_EQ(scene.comm->planner.nTasksColl, 5);
}

// Block 7's wantSym gate is the only writer of task->winRegType, so it's always ncclSymSendRegRecvReg here,
// never ncclSymSendNonregRecvNonreg: the LL-kernel-init check's whole && is provably always false.
TEST_F(SchedulerMicrotest, MakeSymmetricTaskList_LLKernelInit_NeverCalled_BecauseWinRegTypeIsAlwaysRegRecvReg) {
  MakeSymmetricTaskList_Scene scene;
  g_symRegType = ncclSymSendRegRecvReg;
  ncclTaskColl task{};
  task.func = ncclFuncBroadcast;
  task.datatype = ncclInt8;
  scene.comm->workArgsBytes = static_cast<uint32_t>(ncclSymkDevWorkArgs::calcArgsSize(MAXCHANNELS, 5, false));
  ScopedHook llMaskHook(g_symkLLKernelMask, []() { return ~0; });  // every bit set: isolates the regType operand
  ScopedHook tuningHook(g_tuningCompute, [](struct ncclTuningInput_t*, struct ncclTuningResult_t* result) {
    result->symKernelId = ncclSymkKernelId_AllGather_LL;
    result->nChannels = 1;
    result->nWarps = 1;
    return ncclSuccess;
  });
  ScopedHook initOnceHook(g_devrSymkInitOnce, [](struct ncclComm*) { return ncclSuccess; });
  struct ncclTaskColl* remainTasksHead = nullptr;

  EXPECT_EQ(ncclMakeSymmetricTaskList(scene.comm.get(), &task, nullptr, &remainTasksHead), ncclSuccess);
  EXPECT_EQ(initOnceHook.calls, 0);
}

TEST_F(SchedulerMicrotest, MakeSymmetricTaskList_FinalAssignmentLoop_SingleTask_SetsFieldsAndEnqueues) {
  MakeSymmetricTaskList_Scene scene;
  g_symRegType = ncclSymSendRegRecvReg;
  ncclTaskColl task{};
  task.func = ncclFuncBroadcast;
  task.datatype = ncclInt8;
  scene.comm->workArgsBytes = static_cast<uint32_t>(ncclSymkDevWorkArgs::calcArgsSize(MAXCHANNELS, 5, false));
  ScopedHook tuningHook(g_tuningCompute, [](struct ncclTuningInput_t*, struct ncclTuningResult_t* result) {
    result->symKernelId = ncclSymkKernelId_AllGather_LL;
    result->nChannels = 3;
    result->nWarps = 5;
    return ncclSuccess;
  });
  struct ncclTaskColl* remainTasksHead = nullptr;

  EXPECT_EQ(ncclMakeSymmetricTaskList(scene.comm.get(), &task, nullptr, &remainTasksHead), ncclSuccess);
  EXPECT_EQ(remainTasksHead, nullptr);  // kernel found, no sysmem segment: never touches the remainder list
  EXPECT_EQ(task.devFuncId, static_cast<uint32_t>(ncclSymkKernelId_AllGather_LL));
  EXPECT_EQ(task.nMaxChannels, 3);
  EXPECT_EQ(task.nWarps, 5);
  ncclTaskColl* queued = ncclIntruQueueHead(&scene.comm->planner.collSymTaskQueue);
  ASSERT_NE(queued, nullptr);
  EXPECT_EQ(queued, &task);
  EXPECT_EQ(queued->next, nullptr);
}

TEST_F(SchedulerMicrotest, MakeSymmetricTaskList_FinalAssignmentLoop_ConvertSymTaskDevOpWiring) {
  MakeSymmetricTaskList_Scene scene;
  scene.comm->nRanks = 4;
  std::vector<int> rankToNode(4, 0);  // computeLsaSize (via ncclDevrInitOnce) reads this whenever nRanks>=2
  scene.comm->rankToNode = rankToNode.data();
  g_symRegType = ncclSymSendRegRecvReg;
  ncclTaskColl task{};
  task.func = ncclFuncBroadcast;
  task.datatype = ncclFloat16;
  task.opHost = ncclAvg;
  task.opDev.op = ncclDevSum;
  task.opDev.scalarArg = kPoison;
  scene.comm->workArgsBytes = static_cast<uint32_t>(ncclSymkDevWorkArgs::calcArgsSize(MAXCHANNELS, 5, false));
  ScopedHook tuningHook(g_tuningCompute, [](struct ncclTuningInput_t*, struct ncclTuningResult_t* result) {
    result->symKernelId = ncclSymkKernelId_AllGather_LL;
    result->nChannels = 1;
    result->nWarps = 1;
    return ncclSuccess;
  });
  struct ncclTaskColl* remainTasksHead = nullptr;

  EXPECT_EQ(ncclMakeSymmetricTaskList(scene.comm.get(), &task, nullptr, &remainTasksHead), ncclSuccess);
  EXPECT_EQ(task.opDev.op, ncclDevSumPostDiv);
  EXPECT_EQ(task.opDev.scalarArg, ConvertSymTaskDevOp_ExpectedReciprocalScalar(4));
}

TEST_F(SchedulerMicrotest, MakeSymmetricTaskList_FinalAssignmentLoop_MultiTaskBatch_EnqueuesAllInOrder) {
  MakeSymmetricTaskList_Scene scene;
  g_symRegType = ncclSymSendRegRecvReg;
  scene.comm->workArgsBytes = static_cast<uint32_t>(ncclSymkDevWorkArgs::calcArgsSize(MAXCHANNELS, 5, false));

  // Bucket is LIFO (last classified = head = first processed), so classify secondTask before headTask.
  ncclTaskColl headTask{};
  headTask.func = ncclFuncBroadcast;
  headTask.datatype = ncclInt8;
  ncclTaskColl secondTask{};
  secondTask.func = ncclFuncBroadcast;
  secondTask.datatype = ncclInt8;
  secondTask.next = &headTask;

  ScopedHook tuningHook(g_tuningCompute, [](struct ncclTuningInput_t*, struct ncclTuningResult_t* result) {
    result->symKernelId = ncclSymkKernelId_AllGather_LL;
    result->nChannels = 2;
    result->nWarps = 4;
    return ncclSuccess;
  });
  struct ncclTaskColl* remainTasksHead = nullptr;

  EXPECT_EQ(ncclMakeSymmetricTaskList(scene.comm.get(), &secondTask, nullptr, &remainTasksHead), ncclSuccess);
  EXPECT_EQ(remainTasksHead, nullptr);
  EXPECT_EQ(headTask.devFuncId, static_cast<uint32_t>(ncclSymkKernelId_AllGather_LL));
  EXPECT_EQ(secondTask.devFuncId, static_cast<uint32_t>(ncclSymkKernelId_AllGather_LL));
  EXPECT_EQ(headTask.nMaxChannels, 2);
  EXPECT_EQ(secondTask.nMaxChannels, 2);
  EXPECT_EQ(headTask.nWarps, 4);
  EXPECT_EQ(secondTask.nWarps, 4);
  ncclTaskColl* first = ncclIntruQueueHead(&scene.comm->planner.collSymTaskQueue);
  ASSERT_NE(first, nullptr);
  EXPECT_EQ(first, &headTask);       // enqueued in Block 8/9's processing order, not classification order
  ASSERT_NE(first->next, nullptr);
  EXPECT_EQ(first->next, &secondTask);
  EXPECT_EQ(first->next->next, nullptr);  // loop stopped exactly at isSymLast, not beyond
}

TEST_F(SchedulerMicrotest, MakeSymmetricTaskList_OuterCursorLoop_MultipleBucketsAllKernelFound_EnqueuesBoth) {
  MakeSymmetricTaskList_Scene scene;
  g_symRegType = ncclSymSendRegRecvReg;
  scene.comm->workArgsBytes = static_cast<uint32_t>(ncclSymkDevWorkArgs::calcArgsSize(MAXCHANNELS, 1, false));

  ncclTaskColl task2{};
  task2.func = ncclFuncAllGather;  // distinct bucket from task1 (different func)
  task2.datatype = ncclInt8;
  ncclTaskColl task1{};
  task1.func = ncclFuncBroadcast;
  task1.datatype = ncclInt8;
  task1.next = &task2;

  ScopedHook tuningHook(g_tuningCompute, [](struct ncclTuningInput_t*, struct ncclTuningResult_t* result) {
    result->symKernelId = ncclSymkKernelId_AllGather_LL;
    result->nChannels = 1;
    result->nWarps = 1;
    return ncclSuccess;
  });
  struct ncclTaskColl* remainTasksHead = nullptr;

  EXPECT_EQ(ncclMakeSymmetricTaskList(scene.comm.get(), &task1, nullptr, &remainTasksHead), ncclSuccess);
  EXPECT_EQ(remainTasksHead, nullptr);
  ncclTaskColl* first = ncclIntruQueueHead(&scene.comm->planner.collSymTaskQueue);
  ASSERT_NE(first, nullptr);
  EXPECT_EQ(first, &task1);  // cursor order follows first-time bucket creation order
  ASSERT_NE(first->next, nullptr);
  EXPECT_EQ(first->next, &task2);
  EXPECT_EQ(first->next->next, nullptr);
}

TEST_F(SchedulerMicrotest, MakeSymmetricTaskList_MixedBuckets_OneKernelFoundOneNotFound_RoutesEachCorrectly) {
  MakeSymmetricTaskList_Scene scene;
  scene.comm->planner.nTasksColl = 5;
  g_symRegType = ncclSymSendRegRecvReg;
  scene.comm->workArgsBytes = static_cast<uint32_t>(ncclSymkDevWorkArgs::calcArgsSize(MAXCHANNELS, 1, false));

  ncclTaskColl foundTask{};
  foundTask.func = ncclFuncBroadcast;
  foundTask.datatype = ncclInt8;
  ncclTaskColl notFoundTask{};
  notFoundTask.func = ncclFuncAllGather;  // distinct bucket
  notFoundTask.datatype = ncclInt8;
  foundTask.next = &notFoundTask;

  ScopedHook tuningHook(g_tuningCompute, [](struct ncclTuningInput_t* input, struct ncclTuningResult_t* result) {
    if (input->func == ncclFuncBroadcast) {
      result->symKernelId = ncclSymkKernelId_AllGather_LL;
      result->nChannels = 1;
      result->nWarps = 1;
    }  // else: leave *result at its NCCL_TUNING_RESULT_INIT default (ncclSymkKernelId_Count -- "not found")
    return ncclSuccess;
  });
  struct ncclTaskColl* remainTasksHead = nullptr;

  EXPECT_EQ(ncclMakeSymmetricTaskList(scene.comm.get(), &foundTask, nullptr, &remainTasksHead), ncclSuccess);
  EXPECT_EQ(remainTasksHead, &notFoundTask);  // only the not-found task falls back to the remainder
  EXPECT_EQ(notFoundTask.next, nullptr);
  ncclTaskColl* queued = ncclIntruQueueHead(&scene.comm->planner.collSymTaskQueue);
  ASSERT_NE(queued, nullptr);
  EXPECT_EQ(queued, &foundTask);  // only the found task is enqueued for real kernel dispatch
  EXPECT_EQ(queued->next, nullptr);
  // 5 - 2 (both classified) + 1 (only notFoundTask's fallback++ cancels its own --): foundTask's stays decremented.
  EXPECT_EQ(scene.comm->planner.nTasksColl, 4);
}

TEST_F(SchedulerMicrotest, MakeSymmetricTaskList_FinalAssignmentLoop_StopsAtIsSymLast_NotIntoTheNextBatch) {
  MakeSymmetricTaskList_Scene scene;
  g_symRegType = ncclSymSendRegRecvReg;
  // Room for exactly 2 works per batch, not 3: same split as Block 8's ArgsSizeBudgetExhausted test.
  scene.comm->workArgsBytes = static_cast<uint32_t>(ncclSymkDevWorkArgs::calcArgsSize(MAXCHANNELS, 2, false));

  // Bucket is LIFO (last classified = head = first processed): classify taskThird, taskSecond, taskFirst.
  ncclTaskColl taskFirst{};
  taskFirst.func = ncclFuncBroadcast;
  taskFirst.datatype = ncclInt8;
  ncclTaskColl taskSecond{};
  taskSecond.func = ncclFuncBroadcast;
  taskSecond.datatype = ncclInt8;
  taskSecond.next = &taskFirst;
  ncclTaskColl taskThird{};
  taskThird.func = ncclFuncBroadcast;
  taskThird.datatype = ncclInt8;
  taskThird.next = &taskSecond;

  int tuningCalls = 0;
  ScopedHook tuningHook(g_tuningCompute, [&](struct ncclTuningInput_t*, struct ncclTuningResult_t* result) {
    ++tuningCalls;
    // One call per batch: distinct kernelIds make an isSymLast overrun into batch 2's task observable.
    result->symKernelId = (tuningCalls == 1) ? ncclSymkKernelId_AllGather_LL : ncclSymkKernelId_AllGather_LLMC;
    result->nChannels = 1;
    result->nWarps = 1;
    return ncclSuccess;
  });
  struct ncclTaskColl* remainTasksHead = nullptr;

  EXPECT_EQ(ncclMakeSymmetricTaskList(scene.comm.get(), &taskThird, nullptr, &remainTasksHead), ncclSuccess);
  EXPECT_EQ(tuningHook.calls, 2);
  EXPECT_EQ(taskFirst.devFuncId, static_cast<uint32_t>(ncclSymkKernelId_AllGather_LL));
  EXPECT_EQ(taskSecond.devFuncId, static_cast<uint32_t>(ncclSymkKernelId_AllGather_LL));
  // Would wrongly be AllGather_LL (batch 1's result) if the final loop overran isSymLast into batch 2's task.
  EXPECT_EQ(taskThird.devFuncId, static_cast<uint32_t>(ncclSymkKernelId_AllGather_LLMC));
  ncclTaskColl* first = ncclIntruQueueHead(&scene.comm->planner.collSymTaskQueue);
  ASSERT_NE(first, nullptr);
  EXPECT_EQ(first, &taskFirst);
  ASSERT_NE(first->next, nullptr);
  EXPECT_EQ(first->next, &taskSecond);
  ASSERT_NE(first->next->next, nullptr);
  EXPECT_EQ(first->next->next, &taskThird);
  EXPECT_EQ(first->next->next->next, nullptr);
}

