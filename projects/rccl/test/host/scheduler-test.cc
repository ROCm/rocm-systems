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

// Minimal ncclComm/ncclKernelPlan/planner.peers scaffold for ncclScheduleBcastTasksToPlan.
class ScheduleBcastTasksToPlan_Scene {
 public:
  explicit ScheduleBcastTasksToPlan_Scene(int numPeers)
      : comm(new ncclComm{}), plan(new ncclKernelPlan{}), peers(new ncclKernelPlanner::Peer[numPeers]{}) {
    comm->nChannels = 1;
    comm->planner.nTasksBcast = 1;
    comm->planner.peers = peers.get();
    comm->planner.bcast_info.minBcastPeer = 0;
    comm->planner.bcast_info.maxBcastPeer = numPeers - 1;
  }
  std::unique_ptr<ncclComm> comm;
  std::unique_ptr<ncclKernelPlan> plan;
  std::unique_ptr<ncclKernelPlanner::Peer[]> peers;
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
