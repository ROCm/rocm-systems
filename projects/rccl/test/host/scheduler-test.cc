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
#include "fakes/scheduler_fakes.h"

// Local to this TU only: struct ncclComm is never shared across a TU boundary here, so no ABI mismatch.
#define ENABLE_WARP_SPEED

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

// Mirrors ncclDevFuncId's general-collective key (device.h); AllGatherV never takes the special-cased branches.
uint64_t ScheduleBcastTasksToPlan_DevFuncKey(int proto) {
  return (uint64_t(ncclFuncAllGatherV & RCCL_FUNC_ID_MASK) << RCCL_COLL_SHIFT) |
         (uint64_t(NCCL_ALGO_RING & RCCL_FUNC_ID_MASK) << RCCL_ALGO_SHIFT) |
         (uint64_t(proto & RCCL_FUNC_ID_MASK) << RCCL_PROTO_SHIFT);
}
}  // namespace

class SchedulerMicrotest : public ::testing::Test {
 protected:
  void TearDown() override { ResetSchedulerFakes(); }
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
