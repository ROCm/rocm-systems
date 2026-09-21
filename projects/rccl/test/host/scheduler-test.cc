/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests; UUTs are #include'd. Line citations use src/scheduler/ numbers; hipify adds 1.

#include <gtest/gtest.h>

#include <algorithm>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "../common/LogCapture.hpp"
#include "ScopedHook.h"
#include "config/algorithm_registry.h"
#include "fakes/bootstrap_stubs.h"
#include "fakes/dev_runtime_fakes.h"
#include "fakes/enqueue_symbols_fakes.h"
#include "fakes/nccl_stubs.h"
#include "fakes/sym_kernels_fakes.h"
#include "fakes/sym_kernels_index_fakes.h"
#include "fakes/tuning_fakes.h"

// ENABLE_WARP_SPEED is this binary's own compile definition (CMakeLists.txt), not shared with any other TU.

// The only src/config/algorithm_registry.cc symbol symmetric_sched.cc calls, and only inside an INFO() log-guard.
const char* ncclAlgNameForSymk(int) { return "sym-kernel"; }

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
  explicit ScheduleBcastTasksToPlan_Scene(int numPeers, int nChannels = 1)
      : comm(new ncclComm{}),
        plan(new ncclKernelPlan{}),
        peers(new ncclKernelPlanner::Peer[numPeers]{}),
        ringTasks(new ncclTaskBcast*[numPeers]{}),
        rankToIndex(new int[numPeers]{}) {
    comm->nChannels = nChannels;
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
  ~ScheduleBcastTasksToPlan_Scene() { ncclMemoryStackDestruct(&comm->memScoped); }
  std::unique_ptr<ncclComm> comm;
  std::unique_ptr<ncclKernelPlan> plan;
  std::unique_ptr<ncclKernelPlanner::Peer[]> peers;
  std::unique_ptr<ncclTaskBcast*[]> ringTasks;
  std::unique_ptr<int[]> rankToIndex;
};

// Builds a g_ncclGetAlgoInfo hook reporting the given (protocol, nMaxChannels, nWarps) triple and succeeding.
std::function<ncclResult_t(struct ncclComm*, struct ncclTaskColl*, int, int, int, ncclSimInfo_t*)>
ScheduleBcastTasksToPlan_AlgoInfoHook(int protocol, int nMaxChannels, int nWarps) {
  return [protocol, nMaxChannels, nWarps](struct ncclComm*, struct ncclTaskColl* task, int, int, int,
                                          ncclSimInfo_t*) {
    task->protocol = protocol;
    task->nMaxChannels = nMaxChannels;
    task->nWarps = nWarps;
    return ncclSuccess;
  };
}

// Mirrors ncclDevFuncId's general-collective key (device.h); AllGatherV never takes the special-cased branches.
uint64_t ScheduleBcastTasksToPlan_DevFuncKey(int proto);

// Installs the AlgoInfoHook and stamps ncclDevFuncNameToId so funcIndex resolves: the pairing every test past
// the FuncIndexNotFound one needs. Capture with `auto` (ScopedHook is non-movable; C++17 elides the return).
auto ScheduleBcastTasksToPlan_StubAlgoInfoAndFuncId(int proto, int nMaxChannels = 1, int nWarps = 1) {
  ncclDevFuncNameToId[ScheduleBcastTasksToPlan_DevFuncKey(proto)] = 0;
  return ScopedHook(g_ncclGetAlgoInfo, ScheduleBcastTasksToPlan_AlgoInfoHook(proto, nMaxChannels, nWarps));
}

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

// Common setup for tests that need ncclMakeSymmetricTaskList to reach the tuning stage: good windows
// (g_symRegType) and args-buffer room for 5 tasks (workArgsBytes). Caller may still edit the result further.
ncclTaskColl MakeSymmetricTaskList_MakeTask(MakeSymmetricTaskList_Scene& scene, ncclFunc_t func = ncclFuncBroadcast,
                                            ncclDataType_t datatype = ncclInt8) {
  g_symRegType = ncclSymSendRegRecvReg;
  scene.comm->workArgsBytes = static_cast<uint32_t>(ncclSymkDevWorkArgs::calcArgsSize(MAXCHANNELS, 5, false));
  ncclTaskColl task{};
  task.func = func;
  task.datatype = datatype;
  return task;
}

// A g_tuningCompute hook that leaves *result untouched and reports success (no kernel found).
std::function<ncclResult_t(struct ncclTuningInput_t*, struct ncclTuningResult_t*)>
MakeSymmetricTaskList_TuningSucceeds() {
  return [](struct ncclTuningInput_t*, struct ncclTuningResult_t*) { return ncclSuccess; };
}

// A g_tuningCompute hook reporting a found AllGather_LL kernel with the given channel/warp counts.
std::function<ncclResult_t(struct ncclTuningInput_t*, struct ncclTuningResult_t*)>
MakeSymmetricTaskList_TuningFindsKernel(int nChannels, int nWarps) {
  return [nChannels, nWarps](struct ncclTuningInput_t*, struct ncclTuningResult_t* result) {
    result->symKernelId = ncclSymkKernelId_AllGather_LL;
    result->nChannels = nChannels;
    result->nWarps = nWarps;
    return ncclSuccess;
  };
}

// Unlike ncclMakeSymmetricTaskList (raw list + output param), this takes an already-populated queue directly.
class SymmetricTaskScheduler_Scene {
 public:
  SymmetricTaskScheduler_Scene() : comm(new ncclComm{}), plan(new ncclKernelPlan{}) {
    comm->nRanks = 1;
    comm->rank = 0;
  }
  ~SymmetricTaskScheduler_Scene() { free(plan->kernelSymArgs); }  // production's own calloc, plan doesn't own it
  std::unique_ptr<ncclComm> comm;
  std::unique_ptr<ncclKernelPlan> plan;
  struct ncclIntruQueue<struct ncclTaskColl, &ncclTaskColl::next> symTaskQueue{};
};

// Safe default: fields chosen so the whole function, including the packing loop past this scope, completes.
ncclTaskColl SymmetricTaskScheduler_MakeTask() {
  ncclTaskColl task{};
  task.func = ncclFuncAllGather;
  task.datatype = ncclInt8;
  task.devFuncId = ncclSymkKernelId_AllGather_LL;
  task.nMaxChannels = 2;
  task.nWarps = 4;
  task.opDev.op = ncclDevSum;
  task.count = 1024;  // exactly 1 cell for ncclInt8 (cellCount = 1024 / ncclTypeSize(ncclInt8) = 1024)
  task.cgaClusterSize = NCCL_CONFIG_UNDEF_INT;
  task.isSymLast = 1;
  return task;
}

// calcArgsSize() floors at sizeof(ncclSymkDevWorkArgs4K)=4096; kSymSchedBigBatchTasks alone clears it (64*64B).
constexpr int kSymSchedBigBatchTasks = 64;
// Headroom above kSymSchedBigBatchTasks so a 1-cell-per-task batch never exhausts the channel budget mid-loop.
constexpr int kSymSchedBigBatchChannels = 100;

// Builds n tasks (1 cell each, one channel apiece); only the last gets isSymLast. Caller must keep it alive.
std::vector<ncclTaskColl> SymmetricTaskScheduler_EnqueueBatch(
    struct ncclIntruQueue<struct ncclTaskColl, &ncclTaskColl::next>* queue, int n, uint32_t devFuncId,
    uint8_t lastIsSymLast) {
  std::vector<ncclTaskColl> tasks(n);
  for (int i = 0; i < n; ++i) {
    tasks[i] = SymmetricTaskScheduler_MakeTask();
    tasks[i].devFuncId = devFuncId;
    tasks[i].nMaxChannels = kSymSchedBigBatchChannels;
    tasks[i].isSymLast = (i == n - 1) ? lastIsSymLast : 0;
    ncclIntruQueueEnqueue(queue, &tasks[i]);
  }
  return tasks;
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
  // Suite-wide generous default: the canonical sym_kernels_fakes.cc default is false, silently halting classification.
  void SetUp() override {
    g_symkAvailable = [](struct ncclComm*, ncclFunc_t, int, ncclDataType_t, size_t) { return true; };
  }
  void TearDown() override {
    ResetEnqueueSymbolsFakes();   // this file's own enqueue_symbols_fakes.{h,cc} seams
    ResetNcclStubs();             // clears ncclDevFuncNameToId; other tests here use it as a plain global, not a hook
    ResetSymKernelsFakes();      // g_symRegType is a plain global, not a ScopedHook-restorable std::function
    ResetSymKernelsIndexFakes(); // g_symkGetKernelIndex's kernel-table arrays: plain globals, same reason
    ResetBootstrapStubs();       // covers g_bootstrapAllGather
    ResetDevRuntimeFakes();      // covers g_devrFindWindow/g_devrInitOnce/g_devrWindowHasSysmemSegment
    ResetTuningFakes();          // g_tuningCompute's canonical reset (its default is ncclSystemError, not success)
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

// nChannels=1/numPeers<=64 (every test above) cannot distinguish the real formula from either
// dropping the nChannels factor or replacing DIVUP with a raw count: all three collapse to the
// same value there. 65 peers past a 2-channel scene forces 3 distinct wrong answers apart from
// the correct one (see below).
TEST_F(SchedulerMicrotest, ScheduleBcastTasksToPlan_MultiChannelPast64Tasks_BudgetArgsScaleByChannelsAndCeilDiv) {
  constexpr int kNumPeers = 65;
  ScheduleBcastTasksToPlan_Scene scene(kNumPeers, /*nChannels=*/2);
  std::vector<ncclTaskBcast> tasks(kNumPeers);
  for (int peer = 0; peer < kNumPeers; ++peer) {
    tasks[peer].count = 1;
    scene.peers[peer].bcastQueue.head = &tasks[peer];
  }

  int recordedNWorkBatches = -1;
  ssize_t recordedNWorkBytes = -1;
  ScopedHook budgetHook(g_testBudget, [&](struct ncclKernelPlanBudget*, int nWorkBatches, ssize_t nWorkBytes) {
    recordedNWorkBatches = nWorkBatches;
    recordedNWorkBytes = nWorkBytes;
    return true;  // accept everything so the loop runs to completion over all 65 peers
  });

  ncclScheduleBcastTasksToPlan(scene.comm.get(), scene.plan.get(), nullptr);
  EXPECT_EQ(budgetHook.calls, kNumPeers);
  // Last call: batchTasks==64 pre-increment (65th peer). Correct: 2*DIVUP(65,64)==4 and 2*65*sizeof(..).
  // A dropped-nChannels mutant would give 2; a DIVUP-replaced-by-raw-count mutant would give 130.
  EXPECT_EQ(recordedNWorkBatches, 4);
  EXPECT_EQ(recordedNWorkBytes, static_cast<ssize_t>(2 * kNumPeers * sizeof(ncclDevWorkBcast)));
}

TEST_F(SchedulerMicrotest, ScheduleBcastTasksToPlan_TwoPeersAccumulate_ProceedsPastBatchCheck) {
  ScheduleBcastTasksToPlan_Scene scene(/*numPeers=*/2);
  ncclTaskBcast task0{};
  task0.count = 111;
  ncclTaskBcast task1{};
  task1.count = 222;
  scene.peers[0].bcastQueue.head = &task0;
  scene.peers[1].bcastQueue.head = &task1;

  // Unlike ScheduleBcastTasksToPlan_AllPeersEmpty_SkipsEachAndReturnsSuccess, batchTasks!=0 here reaches funcIndex.
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
  ScopedHook algoInfoHook(g_ncclGetAlgoInfo,
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

  ScopedHook algoInfoHook(g_ncclGetAlgoInfo, ScheduleBcastTasksToPlan_AlgoInfoHook(NCCL_PROTO_SIMPLE, 0, 4));
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

  // nMaxChannels 0 makes nParts 0: safe to run this call to completion.
  auto algoInfoHook = ScheduleBcastTasksToPlan_StubAlgoInfoAndFuncId(NCCL_PROTO_SIMPLE, /*nMaxChannels=*/0);
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

  auto algoInfoHook = ScheduleBcastTasksToPlan_StubAlgoInfoAndFuncId(NCCL_PROTO_SIMPLE, /*nMaxChannels=*/0);
  ScopedHook kernelHook(g_planSetDefaultKernel, [&](struct ncclComm*, struct ncclKernelPlan*) {});

  EXPECT_EQ(ncclScheduleBcastTasksToPlan(scene.comm.get(), scene.plan.get(), nullptr), ncclSuccess);
  EXPECT_EQ(kernelHook.calls, 0);
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
    EXPECT_LE(calls, maxitem) << "budget queried more than maxitem times";  // regression fails this test, not the binary
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

  auto algoInfoHook = ScheduleBcastTasksToPlan_StubAlgoInfoAndFuncId(NCCL_PROTO_SIMPLE);

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

  auto algoInfoHook = ScheduleBcastTasksToPlan_StubAlgoInfoAndFuncId(NCCL_PROTO_SIMPLE);
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

  auto algoInfoHook = ScheduleBcastTasksToPlan_StubAlgoInfoAndFuncId(NCCL_PROTO_SIMPLE);
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

  auto algoInfoHook = ScheduleBcastTasksToPlan_StubAlgoInfoAndFuncId(NCCL_PROTO_LL);
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

  auto algoInfoHook = ScheduleBcastTasksToPlan_StubAlgoInfoAndFuncId(NCCL_PROTO_LL128);

  EXPECT_EQ(ncclScheduleBcastTasksToPlan(scene.comm.get(), scene.plan.get(), nullptr), ncclSuccess);

  const std::vector<ncclDevWorkBcast*> items = ScheduleBcastTasksToPlan_CollectWorkItems(scene.plan.get());
  ASSERT_EQ(items.size(), 1u);
  EXPECT_EQ(items[0]->chunkSize, 28672);  // would be 32768 if the LL128 rounding were dropped
}

// Named for what it actually shows (the per-part byte split); it does not independently exercise the
// per-channel ringTasks reset, since all 3 channels here share one rankToIndex and see identical tasks,
// so a broken reset would just be overwritten with the same values and stay unobservable.
TEST_F(SchedulerMicrotest, ScheduleBcastTasksToPlan_ThreeChannels_SplitsBytesPerPart) {
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

  auto algoInfoHook = ScheduleBcastTasksToPlan_StubAlgoInfoAndFuncId(NCCL_PROTO_SIMPLE, /*nMaxChannels=*/3);
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

  auto algoInfoHook = ScheduleBcastTasksToPlan_StubAlgoInfoAndFuncId(NCCL_PROTO_SIMPLE, /*nMaxChannels=*/2);

  EXPECT_EQ(ncclScheduleBcastTasksToPlan(scene.comm.get(), scene.plan.get(), nullptr), ncclSuccess);

  const std::vector<ncclDevWorkBcast*> items = ScheduleBcastTasksToPlan_CollectWorkItems(scene.plan.get());
  ASSERT_EQ(items.size(), 2u);
  // With partBytes=257 (correct): part0 0..alignUp(257,256)=512 -> bytes 512; part1 512..513 -> bytes 1.
  // With partBytes=256 (floor, wrong): part0 bytes would be 256 and part1 256 instead.
  EXPECT_EQ(items[0]->bytes, 512u);
  EXPECT_EQ(items[1]->bytes, 1u);
}

TEST_F(SchedulerMicrotest, ScheduleBcastTasksToPlan_AddProxyOpIfNeededFails_PropagatesError) {
  // nRanks must be >= 2: with nRanks==1, ringDepth(0)==nRanks-1(0) zeroes both sendSlices and recvSlices,
  // so sendSlices+recvSlices is always 0 and the proxyOp call is skipped regardless of task bytes.
  ScheduleBcastTasksToPlan_Scene scene(/*numPeers=*/2);
  scene.comm->buffSizes[NCCL_PROTO_SIMPLE] = 4096;
  ncclTaskBcast task{};
  task.count = 100;  // ringDepth 0 (rankToIndex[0] defaults to 0): sendSlices > 0, recvSlices stays 0
  scene.peers[0].bcastQueue.head = &task;

  auto algoInfoHook = ScheduleBcastTasksToPlan_StubAlgoInfoAndFuncId(NCCL_PROTO_SIMPLE);
  ScopedHook proxyOpHook(g_addProxyOpIfNeeded,
                         [](struct ncclComm*, struct ncclKernelPlan*, struct ncclProxyOp*) {
                           return ncclInternalError;
                         });

  EXPECT_EQ(ncclScheduleBcastTasksToPlan(scene.comm.get(), scene.plan.get(), nullptr), ncclInternalError);
  EXPECT_EQ(proxyOpHook.calls, 1);
}

TEST_F(SchedulerMicrotest, ScheduleBcastTasksToPlan_WarpSpeedMultiplierAbove1_HalvesChannelsAtRealCallSite) {
  ScheduleBcastTasksToPlan_Scene scene(/*numPeers=*/1);
  scene.comm->channels[1].ring.rankToIndex = scene.rankToIndex.get();
  scene.comm->buffSizes[NCCL_PROTO_SIMPLE] = 4096;
  scene.comm->warpSpeedChannelMultiplier = 2;  // agvChannelCount must halve nMaxChannels(4) to 2 right here

  ncclTaskBcast task{};
  task.count = 512;
  task.recvbuff = reinterpret_cast<void*>(0x4000);
  scene.peers[0].bcastQueue.head = &task;

  auto algoInfoHook = ScheduleBcastTasksToPlan_StubAlgoInfoAndFuncId(NCCL_PROTO_SIMPLE, /*nMaxChannels=*/4);
  std::vector<int> workBatchChannelIds;
  ScopedHook workBatchHook(g_addWorkBatchToPlan,
                           [&](struct ncclComm*, struct ncclKernelPlan*, int channelId, enum ncclDevWorkType, int,
                               uint32_t, int, int, bool) { workBatchChannelIds.push_back(channelId); });

  RcclUnitTesting::ScopedDebugLogging debugLogging(NCCL_LOG_INFO, NCCL_COLL);
  ncclResult_t result = ncclInternalError;
  const std::string log = RcclUnitTesting::CaptureLog(
      [&]() { result = ncclScheduleBcastTasksToPlan(scene.comm.get(), scene.plan.get(), nullptr); });

  EXPECT_EQ(result, ncclSuccess);
  // Mutating the real call site to nChannels=tcoll.nMaxChannels (bypassing agvChannelCount) would still touch
  // channels 0-3 and would never print this message, so both lines together kill that mutant.
  EXPECT_TRUE(RcclUnitTesting::LogHas(log, "AllGatherV: WarpSpeed not supported; channels 4 -> 2"));
  EXPECT_EQ(workBatchChannelIds, (std::vector<int>{0, 1}));  // only 2 of the 4 tuned channels used, per the halving
  EXPECT_EQ(scene.plan->channelMask.masks[0] & 0b1111u, 0b0011u);  // channels 0,1 set; 2,3 never touched
}

TEST_F(SchedulerMicrotest, ScheduleBcastTasksToPlan_TailLoop_DrainsSkippedPeerAndFillsPlanQueueInOrder) {
  ScheduleBcastTasksToPlan_Scene scene(/*numPeers=*/3);
  scene.comm->planner.nTasksBcast = 10;  // distinct from batchTasks(2), so a wrong-subtrahend mutant is observable

  ncclTaskBcast task0{};
  task0.count = 0;  // count=0 keeps this test's focus on the tail loop, not the per-channel byte-splitting logic
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
  ScopedHook devrInitOnceHook(g_devrInitOnce, [](struct ncclComm*) { return ncclSuccess; });
  struct ncclTaskColl* remainTasksHead = reinterpret_cast<struct ncclTaskColl*>(0x1);

  EXPECT_EQ(ncclMakeSymmetricTaskList(scene.comm.get(), nullptr, nullptr, &remainTasksHead), ncclSuccess);
  EXPECT_EQ(devrInitOnceHook.calls, 0);  // task==nullptr: the guard must skip the call entirely
  EXPECT_EQ(remainTasksHead, nullptr);
}

TEST_F(SchedulerMicrotest, MakeSymmetricTaskList_TaskNotNull_CallsDevrInitOnceAndPropagatesItsError) {
  MakeSymmetricTaskList_Scene scene;
  ScopedHook devrInitOnceHook(g_devrInitOnce, [](struct ncclComm*) { return ncclInternalError; });
  ncclTaskColl task{};
  task.func = ncclFuncBroadcast;
  struct ncclTaskColl* remainTasksHead = nullptr;

  EXPECT_EQ(ncclMakeSymmetricTaskList(scene.comm.get(), &task, nullptr, &remainTasksHead), ncclInternalError);
  EXPECT_EQ(devrInitOnceHook.calls, 1);
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

  // foundSymm==true reaches the args-size guard next (symmetric_sched.cc:169); workArgsBytes defaults 0, so it rejects.
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

TEST_F(SchedulerMicrotest, MakeSymmetricTaskList_GetSymRegTypeFails_PropagatesError) {
  MakeSymmetricTaskList_Scene scene;
  g_getSymRegTypeResult = ncclInternalError;  // symAvailable/cfgAllowsSymk default true, so this seam is reached
  ncclTaskColl task{};
  task.func = ncclFuncBroadcast;
  struct ncclTaskColl* remainTasksHead = nullptr;

  EXPECT_EQ(ncclMakeSymmetricTaskList(scene.comm.get(), &task, nullptr, &remainTasksHead), ncclInternalError);
}

TEST_F(SchedulerMicrotest, MakeSymmetricTaskList_AllReduceNotForcedOutWhenSymKernelsGenerated_EntersSymmetricBucket) {
  MakeSymmetricTaskList_Scene scene;
  scene.comm->planner.nTasksColl = 5;
  ncclTaskColl task{};
  task.func = ncclFuncAllReduce;
  g_symRegType = ncclSymSendRegRecvReg;  // windows are GOOD here, isolating this from the window-rejection test
  struct ncclTaskColl* remainTasksHead = nullptr;

  // GENERATE_SYM_KERNELS is defined (the shipping config), so AllReduce is no longer special-cased out; it enters
  // the symmetric bucket like Broadcast below and hits the same zero-workArgsBytes guard.
  EXPECT_EQ(ncclMakeSymmetricTaskList(scene.comm.get(), &task, nullptr, &remainTasksHead), ncclInternalError);
  EXPECT_EQ(remainTasksHead, nullptr);
  EXPECT_EQ(task.next, nullptr);
  EXPECT_EQ(scene.comm->planner.nTasksColl, 4);
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
  scene.comm->bootstrap = nullptr;  // half of the guard: consensus block must not run without this
  ncclTaskColl task{};
  task.func = ncclFuncBroadcast;
  ScopedHook symkHook(g_symkAvailable, [](struct ncclComm*, ncclFunc_t, int, ncclDataType_t, size_t) { return false; });
  ScopedHook bootstrapHook(g_bootstrapAllGather, [](void*, void*, int) { return ncclSuccess; });
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
  ScopedHook bootstrapHook(g_bootstrapAllGather, [](void*, void*, int) { return ncclSuccess; });
  struct ncclTaskColl* remainTasksHead = nullptr;

  EXPECT_EQ(ncclMakeSymmetricTaskList(scene.comm.get(), &task, nullptr, &remainTasksHead), ncclSuccess);
  EXPECT_EQ(bootstrapHook.calls, 0);
  EXPECT_EQ(remainTasksHead, &task);
}

TEST_F(SchedulerMicrotest, MakeSymmetricTaskList_ConsensusBootstrapAllGatherFails_PropagatesError) {
  MakeSymmetricTaskList_Scene scene;
  scene.comm->nRanks = 2;
  scene.comm->bootstrap = reinterpret_cast<void*>(0x1);
  ncclTaskColl task{};
  task.func = ncclFuncBroadcast;
  ScopedHook bootstrapHook(g_bootstrapAllGather, [](void*, void*, int) { return ncclSystemError; });
  struct ncclTaskColl* remainTasksHead = nullptr;

  EXPECT_EQ(ncclMakeSymmetricTaskList(scene.comm.get(), &task, nullptr, &remainTasksHead), ncclSystemError);
  EXPECT_EQ(bootstrapHook.calls, 1);
}

TEST_F(SchedulerMicrotest, MakeSymmetricTaskList_ConsensusAllAgree_WantSymStaysTrue_GoesToSymmetricBucket) {
  MakeSymmetricTaskList_Scene scene;
  scene.comm->nRanks = 3;
  scene.comm->bootstrap = reinterpret_cast<void*>(0x1);
  scene.comm->planner.nTasksColl = 5;
  g_symRegType = ncclSymSendRegRecvReg;
  ncclTaskColl task{};
  task.func = ncclFuncBroadcast;

  uint8_t preGatherOwnFlag = 0xFF;
  ScopedHook bootstrapHook(g_bootstrapAllGather, [&](void*, void* allData, int) {
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
  scene.comm->bootstrap = reinterpret_cast<void*>(0x1);
  g_symRegType = ncclSymSendRegRecvReg;
  ncclTaskColl task{};
  task.func = ncclFuncBroadcast;

  ScopedHook bootstrapHook(g_bootstrapAllGather, [&](void*, void* allData, int) {
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

  // Explicit success: the fallback-to-remainder needs this to succeed, but the canonical default now fails loudly.
  ScopedHook tuningHook(g_tuningCompute, MakeSymmetricTaskList_TuningSucceeds());
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

  // Explicit success for both buckets' tuning calls: the canonical g_tuningCompute default now fails loudly.
  ScopedHook tuningHook(g_tuningCompute, MakeSymmetricTaskList_TuningSucceeds());
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

  ncclTaskColl task1{};  // classified first -> becomes the bucket's tail (processed 2nd by the cursor loop)
  task1.func = ncclFuncBroadcast;
  task1.datatype = ncclInt8;
  ncclTaskColl task2{};  // classified second -> becomes the bucket's head (processed 1st by the cursor loop)
  task2.func = ncclFuncBroadcast;
  task2.datatype = ncclInt8;
  task2.aggIsolate = true;  // configBoundary fires via THIS (current, first-processed) task's own flag
  task1.next = &task2;      // classification input order: task1 then task2 (LIFO reverses processing order)

  // Explicit success for both batches' tuning calls: the canonical g_tuningCompute default now fails loudly.
  ScopedHook tuningHook(g_tuningCompute, MakeSymmetricTaskList_TuningSucceeds());
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

  // Explicit success for both batches' tuning calls: the canonical g_tuningCompute default now fails loudly.
  ScopedHook tuningHook(g_tuningCompute, MakeSymmetricTaskList_TuningSucceeds());
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

  // Explicit success for both batches' tuning calls: the canonical g_tuningCompute default now fails loudly.
  ScopedHook tuningHook(g_tuningCompute, MakeSymmetricTaskList_TuningSucceeds());
  struct ncclTaskColl* remainTasksHead = nullptr;
  EXPECT_EQ(ncclMakeSymmetricTaskList(scene.comm.get(), &taskThird, nullptr, &remainTasksHead), ncclSuccess);
  EXPECT_EQ(taskFirst.isSymLast, 0);   // continues: room remains and neither next==nullptr nor configBoundary
  EXPECT_EQ(taskSecond.isSymLast, 1);  // budget for a 3rd work would exceed workArgsBytes: batch ends here
  EXPECT_EQ(taskThird.isSymLast, 1);   // new batch (outer while(task!=NULL) re-enters): ends naturally
  // Remainder order matches the cursor loop's own processing order: {taskFirst, taskSecond} first, {taskThird} second.
  EXPECT_EQ(remainTasksHead, &taskFirst);
  EXPECT_EQ(taskFirst.next, &taskSecond);
  EXPECT_EQ(taskSecond.next, &taskThird);
  EXPECT_EQ(taskThird.next, nullptr);
  EXPECT_EQ(scene.comm->planner.nTasksColl, 5);  // 3 classification-- + 3 fallback++ round-trips to original
}

TEST_F(SchedulerMicrotest, MakeSymmetricTaskList_TuningInput_BasicScalarFieldsWiring) {
  MakeSymmetricTaskList_Scene scene;
  ncclTaskColl task = MakeSymmetricTaskList_MakeTask(scene);
  task.opHost = ncclAvg;
  task.opDev.op = ncclDevSum;  // symkOp == ncclDevSumPostDiv, distinct from the raw opDev.op
  task.count = 777;
  task.winRegType = static_cast<ncclSymRegType_t>(0xBAD);  // sentinel, overwritten by the classification stage
  task.minCTAs = 2;
  task.maxCTAs = 8;
  task.CTAPolicy = 3;

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
  ncclTaskColl task = MakeSymmetricTaskList_MakeTask(scene);

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
  ncclTaskColl task = MakeSymmetricTaskList_MakeTask(scene);
  // A single sym-range bit, not the whole mask, so the overridden value differs from the pre-override default.
  const uint64_t kOneSymBit = 1ull << NCCL_TUNING_SYM_KERNEL_ID_OFFSET;
  task.algMask = kOneSymBit;

  uint64_t capturedMask = 0;
  ScopedHook tuningHook(g_tuningCompute, [&](struct ncclTuningInput_t* input, struct ncclTuningResult_t*) {
    capturedMask = input->tuningMask;
    return ncclSuccess;
  });
  struct ncclTaskColl* remainTasksHead = nullptr;

  EXPECT_EQ(ncclMakeSymmetricTaskList(scene.comm.get(), &task, nullptr, &remainTasksHead), ncclSuccess);
  EXPECT_EQ(capturedMask, kOneSymBit);  // overridden to symkMask, distinct from the SYM_KERNELS default
}

// No "effAlgMask != 0 but symkMask == 0" test: the classification loop's wantSym gate proves it unreachable here.

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
  ncclTaskColl task = MakeSymmetricTaskList_MakeTask(scene);

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

TEST_F(SchedulerMicrotest, MakeSymmetricTaskList_GetCollNetSupportFails_PropagatesErrorBeforeRegBuffOrTuning) {
  MakeSymmetricTaskList_Scene scene;
  ncclTaskColl task = MakeSymmetricTaskList_MakeTask(scene);

  ScopedHook collNetHook(g_getCollNetSupport,
                         [](struct ncclComm*, struct ncclTaskColl*, int*) { return ncclInternalError; });
  ScopedHook regBuffHook(g_getRegBuff, [](struct ncclComm*, struct ncclTaskColl*, int* out) {
    *out = 0;
    return ncclSuccess;
  });
  ScopedHook tuningHook(g_tuningCompute, MakeSymmetricTaskList_TuningSucceeds());
  struct ncclTaskColl* remainTasksHead = nullptr;

  EXPECT_EQ(ncclMakeSymmetricTaskList(scene.comm.get(), &task, nullptr, &remainTasksHead), ncclInternalError);
  EXPECT_EQ(collNetHook.calls, 1);
  EXPECT_EQ(regBuffHook.calls, 0);  // NCCLCHECK returns immediately: regBuff is never reached
  EXPECT_EQ(tuningHook.calls, 0);
}

TEST_F(SchedulerMicrotest, MakeSymmetricTaskList_GetRegBuffFails_PropagatesErrorBeforeTuning) {
  MakeSymmetricTaskList_Scene scene;
  ncclTaskColl task = MakeSymmetricTaskList_MakeTask(scene);

  ScopedHook regBuffHook(g_getRegBuff,
                         [](struct ncclComm*, struct ncclTaskColl*, int*) { return ncclInternalError; });
  ScopedHook tuningHook(g_tuningCompute, MakeSymmetricTaskList_TuningSucceeds());
  struct ncclTaskColl* remainTasksHead = nullptr;

  EXPECT_EQ(ncclMakeSymmetricTaskList(scene.comm.get(), &task, nullptr, &remainTasksHead), ncclInternalError);
  EXPECT_EQ(regBuffHook.calls, 1);
  EXPECT_EQ(tuningHook.calls, 0);  // NCCLCHECK returns immediately: tuning is never reached
}

TEST_F(SchedulerMicrotest, MakeSymmetricTaskList_TuningInput_SymAligned16B_TrueWhenBuffersAlign) {
  MakeSymmetricTaskList_Scene scene;
  ncclTaskColl task = MakeSymmetricTaskList_MakeTask(scene);
  task.sendbuff = reinterpret_cast<void*>(0x1030);  // matches SymBatchAligned16B_SingleTaskNoWindowsAligned
  task.recvbuff = reinterpret_cast<void*>(0x1020);  // offset diff 0x10: divisible by 16

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
  ncclTaskColl task = MakeSymmetricTaskList_MakeTask(scene);
  task.sendbuff = reinterpret_cast<void*>(0x1028);  // matches SymBatchAligned16B_SingleTaskNoWindowsMisaligned
  task.recvbuff = reinterpret_cast<void*>(0x1020);  // offset diff 8: not divisible by 16

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
  ncclTaskColl task = MakeSymmetricTaskList_MakeTask(scene);
  task.algMask = NCCL_TUNING_MASK_SYM_KERNELS;  // only sym bits: satisfies both mask conditions at once
  task.forceAlgSelection = 1;
  // Explicit success: the 5th condition needs kernelId==Count, but the canonical default now fails loudly.
  ScopedHook tuningHook(g_tuningCompute, MakeSymmetricTaskList_TuningSucceeds());
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
  ncclTaskColl task = MakeSymmetricTaskList_MakeTask(scene);
  // task.algMask left at 0 (default): effAlgMask == 0, so the hard-error branch's 2nd condition is false.
  // Explicit success: the canonical g_tuningCompute default now fails loudly instead of leaving kernelId==Count.
  ScopedHook tuningHook(g_tuningCompute, MakeSymmetricTaskList_TuningSucceeds());
  struct ncclTaskColl* remainTasksHead = nullptr;

  EXPECT_EQ(ncclMakeSymmetricTaskList(scene.comm.get(), &task, nullptr, &remainTasksHead), ncclSuccess);
  EXPECT_EQ(remainTasksHead, &task);  // kernelId==Count (default) still falls back safely on its own
  EXPECT_EQ(scene.comm->planner.nTasksColl, 5);
}

// No hard-error "no sym bits" test: cfgAllowsSymk (symmetric_sched.cc) keeps such a task from reaching headTask.

TEST_F(SchedulerMicrotest, MakeSymmetricTaskList_HardErrorBranch_HasGeneralBitsInMask_SkipsAndFallsBack) {
  MakeSymmetricTaskList_Scene scene;
  scene.comm->planner.nTasksColl = 5;
  ncclTaskColl task = MakeSymmetricTaskList_MakeTask(scene);
  // Has sym bits (would satisfy condition 3) but also general bits, so condition 4 ((mask&GENERAL)==0) is false.
  task.algMask = NCCL_TUNING_MASK_SYM_KERNELS | NCCL_TUNING_MASK_GENERAL_KERNELS;
  task.forceAlgSelection = 1;
  // Explicit success: the canonical g_tuningCompute default now fails loudly instead of leaving kernelId==Count.
  ScopedHook tuningHook(g_tuningCompute, MakeSymmetricTaskList_TuningSucceeds());
  struct ncclTaskColl* remainTasksHead = nullptr;

  EXPECT_EQ(ncclMakeSymmetricTaskList(scene.comm.get(), &task, nullptr, &remainTasksHead), ncclSuccess);
  EXPECT_EQ(remainTasksHead, &task);
  EXPECT_EQ(scene.comm->planner.nTasksColl, 5);
}

TEST_F(SchedulerMicrotest, MakeSymmetricTaskList_HardErrorBranch_ForceAlgSelectionFalse_SkipsAndFallsBack) {
  MakeSymmetricTaskList_Scene scene;
  scene.comm->planner.nTasksColl = 5;
  ncclTaskColl task = MakeSymmetricTaskList_MakeTask(scene);
  task.algMask = NCCL_TUNING_MASK_SYM_KERNELS;
  // task.forceAlgSelection left at 0 (default): the 5th condition is false.
  // Explicit success: the canonical g_tuningCompute default now fails loudly instead of leaving kernelId==Count.
  ScopedHook tuningHook(g_tuningCompute, MakeSymmetricTaskList_TuningSucceeds());
  struct ncclTaskColl* remainTasksHead = nullptr;

  EXPECT_EQ(ncclMakeSymmetricTaskList(scene.comm.get(), &task, nullptr, &remainTasksHead), ncclSuccess);
  EXPECT_EQ(remainTasksHead, &task);
  EXPECT_EQ(scene.comm->planner.nTasksColl, 5);
}

TEST_F(SchedulerMicrotest, MakeSymmetricTaskList_HardErrorBranch_KernelIdFound_SkipsAndProceedsSafely) {
  MakeSymmetricTaskList_Scene scene;
  ncclTaskColl task = MakeSymmetricTaskList_MakeTask(scene);
  task.algMask = NCCL_TUNING_MASK_SYM_KERNELS;  // every other condition true
  task.forceAlgSelection = 1;
  ScopedHook tuningHook(g_tuningCompute, MakeSymmetricTaskList_TuningFindsKernel(1, 1));
  struct ncclTaskColl* remainTasksHead = nullptr;

  EXPECT_EQ(ncclMakeSymmetricTaskList(scene.comm.get(), &task, nullptr, &remainTasksHead), ncclSuccess);
}

// No test drives the effAlgMask!=0 && kernelId!=Count INFO branch itself: that arm's only observable
// effect is the log line (symmetric_sched.cc:251), and asserting on log wording is fragile, so it is
// deliberately left uncovered here rather than pinned to a string.
TEST_F(SchedulerMicrotest, MakeSymmetricTaskList_InfoLoggingBranch_KernelIdCount_DoesNotLogEvenWithEffAlgMask) {
  MakeSymmetricTaskList_Scene scene;
  scene.comm->planner.nTasksColl = 5;
  ncclTaskColl task = MakeSymmetricTaskList_MakeTask(scene);
  task.algMask = NCCL_TUNING_MASK_SYM_KERNELS;
  // Explicit success: kernelId must stay Count, but the canonical default now fails loudly instead.
  ScopedHook tuningHook(g_tuningCompute, MakeSymmetricTaskList_TuningSucceeds());
  struct ncclTaskColl* remainTasksHead = nullptr;

  EXPECT_EQ(ncclMakeSymmetricTaskList(scene.comm.get(), &task, nullptr, &remainTasksHead), ncclSuccess);
  EXPECT_EQ(remainTasksHead, &task);  // falls back via the kernelId==Count disjunct, same as always
  EXPECT_EQ(scene.comm->planner.nTasksColl, 5);
}

// task->winRegType is always ncclSymSendRegRecvReg here (the classification loop's wantSym gate is its only writer).
TEST_F(SchedulerMicrotest, MakeSymmetricTaskList_LLKernelInit_NeverCalled_BecauseWinRegTypeIsAlwaysRegRecvReg) {
  MakeSymmetricTaskList_Scene scene;
  ncclTaskColl task = MakeSymmetricTaskList_MakeTask(scene);
  ScopedHook llMaskHook(g_symkLLKernelMask, []() { return ~0; });  // every bit set: isolates the regType operand
  ScopedHook tuningHook(g_tuningCompute, MakeSymmetricTaskList_TuningFindsKernel(1, 1));
  ScopedHook initOnceHook(g_symkInitOnce, [](struct ncclComm*) { return ncclSuccess; });
  struct ncclTaskColl* remainTasksHead = nullptr;

  EXPECT_EQ(ncclMakeSymmetricTaskList(scene.comm.get(), &task, nullptr, &remainTasksHead), ncclSuccess);
  EXPECT_EQ(initOnceHook.calls, 0);
}

TEST_F(SchedulerMicrotest, MakeSymmetricTaskList_FinalAssignmentLoop_SingleTask_SetsFieldsAndEnqueues) {
  MakeSymmetricTaskList_Scene scene;
  ncclTaskColl task = MakeSymmetricTaskList_MakeTask(scene);
  ScopedHook tuningHook(g_tuningCompute, MakeSymmetricTaskList_TuningFindsKernel(3, 5));
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
  ncclTaskColl task = MakeSymmetricTaskList_MakeTask(scene, ncclFuncBroadcast, ncclFloat16);
  task.opHost = ncclAvg;
  task.opDev.op = ncclDevSum;
  task.opDev.scalarArg = kPoison;
  ScopedHook tuningHook(g_tuningCompute, MakeSymmetricTaskList_TuningFindsKernel(1, 1));
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

  ScopedHook tuningHook(g_tuningCompute, MakeSymmetricTaskList_TuningFindsKernel(2, 4));
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
  EXPECT_EQ(first, &headTask);       // enqueued head-to-tail by final assignment, not LIFO classification order
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

  ScopedHook tuningHook(g_tuningCompute, MakeSymmetricTaskList_TuningFindsKernel(1, 1));
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

// task->sendWin/recvWin are overwritten by ncclDevrFindWindow(sendbuff/recvbuff) at symmetric_sched.cc:115-116
// before the fallback check runs, so the fake windows must come from that seam, keyed off the buffer identity.
// Real (not just distinguishable) objects: symBatchAligned16B dereferences ->userPtr before the fallback runs.
ncclDevrWindow MakeSymmetricTaskList_kSendWinStorage{};
ncclDevrWindow MakeSymmetricTaskList_kRecvWinStorage{};
auto* const MakeSymmetricTaskList_kSendWin = &MakeSymmetricTaskList_kSendWinStorage;
auto* const MakeSymmetricTaskList_kRecvWin = &MakeSymmetricTaskList_kRecvWinStorage;

std::function<ncclResult_t(struct ncclComm*, void const*, struct ncclDevrWindow**)>
MakeSymmetricTaskList_FindWindowHook(ncclTaskColl& task) {
  return [&task](struct ncclComm*, void const* ptr, struct ncclDevrWindow** window) {
    if (window) *window = (ptr == task.sendbuff) ? MakeSymmetricTaskList_kSendWin : MakeSymmetricTaskList_kRecvWin;
    return ncclSuccess;
  };
}

// kernelId IS found here (unlike the sibling KernelIdCount fallback tests): only the sendWin sysmem
// disjunct at symmetric_sched.cc:255-256 forces this into the remainder, isolating that operand.
TEST_F(SchedulerMicrotest, MakeSymmetricTaskList_SendWinHasSysmemSegment_OverridesFoundKernel_GoesToRemainder) {
  MakeSymmetricTaskList_Scene scene;
  ncclTaskColl task = MakeSymmetricTaskList_MakeTask(scene);
  task.sendbuff = reinterpret_cast<void*>(0x2001);
  task.recvbuff = reinterpret_cast<void*>(0x2002);
  ScopedHook tuningHook(g_tuningCompute, MakeSymmetricTaskList_TuningFindsKernel(1, 1));
  ScopedHook findWindowHook(g_devrFindWindow, MakeSymmetricTaskList_FindWindowHook(task));
  ScopedHook sysmemHook(g_devrWindowHasSysmemSegment,
                        [](struct ncclDevrWindow* win) { return win == MakeSymmetricTaskList_kSendWin; });
  struct ncclTaskColl* remainTasksHead = nullptr;

  EXPECT_EQ(ncclMakeSymmetricTaskList(scene.comm.get(), &task, nullptr, &remainTasksHead), ncclSuccess);
  EXPECT_EQ(remainTasksHead, &task);
  EXPECT_EQ(task.next, nullptr);
}

// Mirror of the sendWin test above with the sysmem hook's target swapped, isolating the recvWin disjunct instead.
TEST_F(SchedulerMicrotest, MakeSymmetricTaskList_RecvWinHasSysmemSegment_OverridesFoundKernel_GoesToRemainder) {
  MakeSymmetricTaskList_Scene scene;
  ncclTaskColl task = MakeSymmetricTaskList_MakeTask(scene);
  task.sendbuff = reinterpret_cast<void*>(0x2001);
  task.recvbuff = reinterpret_cast<void*>(0x2002);
  ScopedHook tuningHook(g_tuningCompute, MakeSymmetricTaskList_TuningFindsKernel(1, 1));
  ScopedHook findWindowHook(g_devrFindWindow, MakeSymmetricTaskList_FindWindowHook(task));
  ScopedHook sysmemHook(g_devrWindowHasSysmemSegment,
                        [](struct ncclDevrWindow* win) { return win == MakeSymmetricTaskList_kRecvWin; });
  struct ncclTaskColl* remainTasksHead = nullptr;

  EXPECT_EQ(ncclMakeSymmetricTaskList(scene.comm.get(), &task, nullptr, &remainTasksHead), ncclSuccess);
  EXPECT_EQ(remainTasksHead, &task);
  EXPECT_EQ(task.next, nullptr);
}

TEST_F(SchedulerMicrotest, MakeSymmetricTaskList_FinalAssignmentLoop_StopsAtIsSymLast_NotIntoTheNextBatch) {
  MakeSymmetricTaskList_Scene scene;
  g_symRegType = ncclSymSendRegRecvReg;
  // Room for exactly 2 works per batch, not 3: same split as ...ArgsSizeBudgetExhausted_SplitsIntoTwoBatches.
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

// Known bug reproduction, not correct-behavior coverage: the fallback loop at symmetric_sched.cc:259-270 never
// sets remainTasksTail->next = nullptr, so the remainder list's tail keeps its stale classification-time ->next.
TEST_F(SchedulerMicrotest, MakeSymmetricTaskList_KnownBug_FallbackTailNextNotNulled_AliasesNextBatchHead) {
  MakeSymmetricTaskList_Scene scene;
  g_symRegType = ncclSymSendRegRecvReg;
  // Room for exactly 2 works per batch, not 3: same split as ...ArgsSizeBudgetExhausted_SplitsIntoTwoBatches.
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
    // Batch 1 (taskFirst+taskSecond) leaves *result untouched (kernelId stays Count: falls back). Batch 2
    // (taskThird) reports a real kernel, so it takes the success path instead.
    if (tuningCalls == 2) {
      result->symKernelId = ncclSymkKernelId_AllGather_LL;
      result->nChannels = 1;
      result->nWarps = 1;
    }
    return ncclSuccess;
  });
  struct ncclTaskColl* remainTasksHead = nullptr;

  EXPECT_EQ(ncclMakeSymmetricTaskList(scene.comm.get(), &taskThird, nullptr, &remainTasksHead), ncclSuccess);
  EXPECT_EQ(tuningHook.calls, 2);

  // The remainder list looks right at a glance: taskFirst -> taskSecond.
  EXPECT_EQ(remainTasksHead, &taskFirst);
  ASSERT_NE(taskFirst.next, nullptr);
  EXPECT_EQ(taskFirst.next, &taskSecond);

  // The bug: taskSecond is the remainder list's tail, so taskSecond.next should be nullptr. It is instead
  // still the stale classification-time pointer to taskThird, aliasing a node also live in collSymTaskQueue.
  EXPECT_EQ(taskSecond.next, &taskThird);

  // taskThird is genuinely enqueued by batch 2's success path, not merely dangling off taskSecond.
  ncclTaskColl* queued = ncclIntruQueueHead(&scene.comm->planner.collSymTaskQueue);
  ASSERT_NE(queued, nullptr);
  EXPECT_EQ(queued, &taskThird);
  EXPECT_EQ(taskThird.next, nullptr);  // ncclIntruQueueEnqueue correctly nulled this end
}

// ncclSymmetricTaskScheduler setup + work-counting loop (symmetric_sched.cc:299-352); packing's own fields unasserted.

TEST_F(SchedulerMicrotest, SymmetricTaskScheduler_IsSymColl_SetTrue) {
  SymmetricTaskScheduler_Scene scene;
  ncclTaskColl task = SymmetricTaskScheduler_MakeTask();
  ncclIntruQueueEnqueue(&scene.symTaskQueue, &task);
  scene.plan->isSymColl = false;
  EXPECT_EQ(ncclSymmetricTaskScheduler(scene.comm.get(), &scene.symTaskQueue, scene.plan.get()), ncclSuccess);
  EXPECT_TRUE(scene.plan->isSymColl);
}

TEST_F(SchedulerMicrotest, SymmetricTaskScheduler_HasProxyOps_SetFalse) {
  SymmetricTaskScheduler_Scene scene;
  ncclTaskColl task = SymmetricTaskScheduler_MakeTask();
  ncclIntruQueueEnqueue(&scene.symTaskQueue, &task);
  scene.plan->hasProxyOps = true;
  EXPECT_EQ(ncclSymmetricTaskScheduler(scene.comm.get(), &scene.symTaskQueue, scene.plan.get()), ncclSuccess);
  EXPECT_FALSE(scene.plan->hasProxyOps);
}

// Targets the __HIPCC__ arm at symmetric_sched.cc:322-323 (comm->WarpSize), not the #else's WARP_SIZE.
TEST_F(SchedulerMicrotest, SymmetricTaskScheduler_ThreadPerBlock_UsesCommWarpSizeField_HipArmCompiles) {
  SymmetricTaskScheduler_Scene scene;
  scene.comm->WarpSize = 77;
  ncclTaskColl task = SymmetricTaskScheduler_MakeTask();
  task.nWarps = 3;
  ncclIntruQueueEnqueue(&scene.symTaskQueue, &task);
  EXPECT_EQ(ncclSymmetricTaskScheduler(scene.comm.get(), &scene.symTaskQueue, scene.plan.get()), ncclSuccess);
  EXPECT_EQ(scene.plan->threadPerBlock, 3 * 77);
}

TEST_F(SchedulerMicrotest, SymmetricTaskScheduler_KernelIdToString_CallSiteWiresHeadTaskDevFuncId) {
  SymmetricTaskScheduler_Scene scene;
  ncclTaskColl task = SymmetricTaskScheduler_MakeTask();
  ncclIntruQueueEnqueue(&scene.symTaskQueue, &task);
  int captured = -1;
  ScopedHook idToStringHook(g_symkKernelIdToString, [&](int kernelId) {
    captured = kernelId;
    return "captured-kernel";
  });
  EXPECT_EQ(ncclSymmetricTaskScheduler(scene.comm.get(), &scene.symTaskQueue, scene.plan.get()), ncclSuccess);
  EXPECT_EQ(idToStringHook.calls, 1);
  EXPECT_EQ(captured, static_cast<int>(task.devFuncId));
}

TEST_F(SchedulerMicrotest, SymmetricTaskScheduler_KernelIndex_CallSiteWiresHeadTaskFields) {
  SymmetricTaskScheduler_Scene scene;
  ncclTaskColl task = SymmetricTaskScheduler_MakeTask();
  // Non-zero, mutually distinct op/datatype: ncclDevSum==0 and ncclInt8==0 would let a dropped-arg mutant survive.
  task.opDev.op = ncclDevProd;
  task.datatype = ncclFloat32;
  ncclIntruQueueEnqueue(&scene.symTaskQueue, &task);
  ncclSymkKernelId capturedId = ncclSymkKernelId_Count;
  int capturedOp = -1;
  ncclDataType_t capturedTy = ncclNumTypes;
  ScopedHook kernelIndexHook(g_symkGetKernelIndex, [&](ncclSymkKernelId id, int op, ncclDataType_t ty) {
    capturedId = id;
    capturedOp = op;
    capturedTy = ty;
    return 0;
  });
  EXPECT_EQ(ncclSymmetricTaskScheduler(scene.comm.get(), &scene.symTaskQueue, scene.plan.get()), ncclSuccess);
  EXPECT_EQ(kernelIndexHook.calls, 1);
  EXPECT_EQ(capturedId, static_cast<ncclSymkKernelId>(task.devFuncId));
  EXPECT_EQ(capturedOp, task.opDev.op);
  EXPECT_EQ(capturedTy, task.datatype);
}

// kernelIndex is independent of kernelId: hooking it to a nonzero slot proves the three lookups at
// symmetric_sched.cc:338-341 use kernelIndex, not a hardcoded 0, for kernelFn/kernelDynSmem.
TEST_F(SchedulerMicrotest, SymmetricTaskScheduler_KernelIndex_NonZero_SelectsThatSlotsKernelTables) {
  SymmetricTaskScheduler_Scene scene;
  constexpr int kNonZeroKernelIndex = 3;
  static int slotZeroFn = 0, slotThreeFn = 0;
  ncclSymkKernelList[0] = &slotZeroFn;
  ncclSymkKernelList[kNonZeroKernelIndex] = &slotThreeFn;
  ncclSymkKernelMaxDynamicSmem[0] = 999;
  ncclSymkKernelMaxDynamicSmem[kNonZeroKernelIndex] = 555;
  ScopedHook kernelIndexHook(g_symkGetKernelIndex,
                             [](ncclSymkKernelId, int, ncclDataType_t) { return kNonZeroKernelIndex; });
  ncclTaskColl task = SymmetricTaskScheduler_MakeTask();  // devFuncId == ncclSymkKernelId_AllGather_LL
  ncclIntruQueueEnqueue(&scene.symTaskQueue, &task);
  ScopedHook smemMaskHook(g_symkDynamicSmemKernelMask, []() { return 1 << (int)ncclSymkKernelId_AllGather_LL; });
  EXPECT_EQ(ncclSymmetricTaskScheduler(scene.comm.get(), &scene.symTaskQueue, scene.plan.get()), ncclSuccess);
  EXPECT_EQ(scene.plan->kernelFn, static_cast<void*>(&slotThreeFn));
  EXPECT_EQ(scene.plan->kernelDynSmem, 555);
}

TEST_F(SchedulerMicrotest, SymmetricTaskScheduler_ProfilingRequested_FalseWhenPluginNotLoaded) {
  SymmetricTaskScheduler_Scene scene;
  ncclTaskColl task = SymmetricTaskScheduler_MakeTask();
  task.eActivationMask = ncclProfileKernelCh;  // bit set, but plugin-not-loaded (default) must still gate it off
  ncclIntruQueueEnqueue(&scene.symTaskQueue, &task);
  EXPECT_EQ(ncclSymmetricTaskScheduler(scene.comm.get(), &scene.symTaskQueue, scene.plan.get()), ncclSuccess);
  EXPECT_FALSE(scene.plan->hasProfilerOps);
}

TEST_F(SchedulerMicrotest, SymmetricTaskScheduler_ProfilingRequested_FalseWhenActivationMaskMissingBit) {
  SymmetricTaskScheduler_Scene scene;
  ScopedHook pluginHook(g_profilerPluginLoaded, []() { return true; });
  ncclTaskColl task = SymmetricTaskScheduler_MakeTask();
  task.eActivationMask = 0;  // plugin loaded, but the bit itself is missing
  ncclIntruQueueEnqueue(&scene.symTaskQueue, &task);
  EXPECT_EQ(ncclSymmetricTaskScheduler(scene.comm.get(), &scene.symTaskQueue, scene.plan.get()), ncclSuccess);
  EXPECT_FALSE(scene.plan->hasProfilerOps);
}

TEST_F(SchedulerMicrotest, SymmetricTaskScheduler_ProfilingRequested_TrueWhenPluginLoadedAndBitSet) {
  SymmetricTaskScheduler_Scene scene;
  ScopedHook pluginHook(g_profilerPluginLoaded, []() { return true; });
  ncclTaskColl task = SymmetricTaskScheduler_MakeTask();
  task.eActivationMask = ncclProfileKernelCh;
  ncclIntruQueueEnqueue(&scene.symTaskQueue, &task);
  EXPECT_EQ(ncclSymmetricTaskScheduler(scene.comm.get(), &scene.symTaskQueue, scene.plan.get()), ncclSuccess);
  EXPECT_TRUE(scene.plan->hasProfilerOps);
}

TEST_F(SchedulerMicrotest, SymmetricTaskScheduler_ProfilerEnabled_FalseWhenPersistentEvenThoughRequested) {
  SymmetricTaskScheduler_Scene scene;
  static int profileVariant = 0, baseVariant = 0;
  ncclSymkKernelListProfile[0] = &profileVariant;
  ncclSymkKernelList[0] = &baseVariant;
  ScopedHook pluginHook(g_profilerPluginLoaded, []() { return true; });
  ncclTaskColl task = SymmetricTaskScheduler_MakeTask();
  task.eActivationMask = ncclProfileKernelCh;
  ncclIntruQueueEnqueue(&scene.symTaskQueue, &task);
  scene.plan->persistent = true;  // requested but persistent: profilerEnabled must still be false
  EXPECT_EQ(ncclSymmetricTaskScheduler(scene.comm.get(), &scene.symTaskQueue, scene.plan.get()), ncclSuccess);
  EXPECT_TRUE(scene.plan->hasProfilerOps);  // hasProfilerOps mirrors profilingRequested, not profilerEnabled
  EXPECT_EQ(scene.plan->kernelFn, static_cast<void*>(&baseVariant));
}

TEST_F(SchedulerMicrotest, SymmetricTaskScheduler_KernelFn_SelectsProfileVariant_WhenEnabledAndNonNull) {
  SymmetricTaskScheduler_Scene scene;
  static int profileVariant = 0, baseVariant = 0;
  ncclSymkKernelListProfile[0] = &profileVariant;
  ncclSymkKernelList[0] = &baseVariant;
  ScopedHook pluginHook(g_profilerPluginLoaded, []() { return true; });
  ncclTaskColl task = SymmetricTaskScheduler_MakeTask();
  task.eActivationMask = ncclProfileKernelCh;
  ncclIntruQueueEnqueue(&scene.symTaskQueue, &task);
  scene.plan->persistent = false;
  EXPECT_EQ(ncclSymmetricTaskScheduler(scene.comm.get(), &scene.symTaskQueue, scene.plan.get()), ncclSuccess);
  EXPECT_EQ(scene.plan->kernelFn, static_cast<void*>(&profileVariant));
}

TEST_F(SchedulerMicrotest, SymmetricTaskScheduler_KernelFn_FallsBackToBase_WhenProfileVariantNull) {
  SymmetricTaskScheduler_Scene scene;
  static int baseVariant = 0;
  ncclSymkKernelListProfile[0] = nullptr;
  ncclSymkKernelList[0] = &baseVariant;
  ScopedHook pluginHook(g_profilerPluginLoaded, []() { return true; });
  ncclTaskColl task = SymmetricTaskScheduler_MakeTask();
  task.eActivationMask = ncclProfileKernelCh;
  ncclIntruQueueEnqueue(&scene.symTaskQueue, &task);
  scene.plan->persistent = false;  // profilerEnabled is true, but the profile variant itself is absent
  EXPECT_EQ(ncclSymmetricTaskScheduler(scene.comm.get(), &scene.symTaskQueue, scene.plan.get()), ncclSuccess);
  EXPECT_EQ(scene.plan->kernelFn, static_cast<void*>(&baseVariant));
}

TEST_F(SchedulerMicrotest, SymmetricTaskScheduler_KernelFn_FallsBackToBase_WhenProfilingNotRequested) {
  SymmetricTaskScheduler_Scene scene;
  static int profileVariant = 0, baseVariant = 0;
  ncclSymkKernelListProfile[0] = &profileVariant;
  ncclSymkKernelList[0] = &baseVariant;
  ncclTaskColl task = SymmetricTaskScheduler_MakeTask();  // eActivationMask left at 0: profiling never requested
  ncclIntruQueueEnqueue(&scene.symTaskQueue, &task);
  EXPECT_EQ(ncclSymmetricTaskScheduler(scene.comm.get(), &scene.symTaskQueue, scene.plan.get()), ncclSuccess);
  EXPECT_EQ(scene.plan->kernelFn, static_cast<void*>(&baseVariant));
}

TEST_F(SchedulerMicrotest, SymmetricTaskScheduler_KernelDynSmem_TrueWhenBitSetForKernelId) {
  SymmetricTaskScheduler_Scene scene;
  ncclSymkKernelMaxDynamicSmem[0] = 12345;
  ncclTaskColl task = SymmetricTaskScheduler_MakeTask();  // devFuncId == ncclSymkKernelId_AllGather_LL
  ncclIntruQueueEnqueue(&scene.symTaskQueue, &task);
  ScopedHook smemMaskHook(g_symkDynamicSmemKernelMask,
                          []() { return 1 << (int)ncclSymkKernelId_AllGather_LL; });
  EXPECT_EQ(ncclSymmetricTaskScheduler(scene.comm.get(), &scene.symTaskQueue, scene.plan.get()), ncclSuccess);
  EXPECT_EQ(scene.plan->kernelDynSmem, 12345);
}

TEST_F(SchedulerMicrotest, SymmetricTaskScheduler_KernelDynSmem_FalseWhenBitNotSetForKernelId) {
  SymmetricTaskScheduler_Scene scene;
  ncclSymkKernelMaxDynamicSmem[0] = 12345;
  ncclTaskColl task = SymmetricTaskScheduler_MakeTask();  // devFuncId == ncclSymkKernelId_AllGather_LL
  ncclIntruQueueEnqueue(&scene.symTaskQueue, &task);
  ScopedHook smemMaskHook(g_symkDynamicSmemKernelMask,
                          []() { return 1 << (int)ncclSymkKernelId_AllGather_LLMC; });  // a different kernel's bit
  EXPECT_EQ(ncclSymmetricTaskScheduler(scene.comm.get(), &scene.symTaskQueue, scene.plan.get()), ncclSuccess);
  EXPECT_EQ(scene.plan->kernelDynSmem, 0);
}

// Head task's nMaxChannels(60) differs from the rest of the batch(kSymSchedBigBatchChannels=100): proves the
// value comes from headTask specifically, not a std::max shared across the batch (both clear the 4096 floor).
TEST_F(SchedulerMicrotest, SymmetricTaskScheduler_NMaxChannels_DerivedFromHeadTask_FeedsArgsSize) {
  SymmetricTaskScheduler_Scene scene;
  auto tasks = SymmetricTaskScheduler_EnqueueBatch(&scene.symTaskQueue, kSymSchedBigBatchTasks,
                                                    ncclSymkKernelId_AllGather_LL, /*lastIsSymLast=*/1);
  constexpr int kHeadNMaxChannels = 60;
  tasks[0].nMaxChannels = kHeadNMaxChannels;
  EXPECT_EQ(ncclSymmetricTaskScheduler(scene.comm.get(), &scene.symTaskQueue, scene.plan.get()), ncclSuccess);
  EXPECT_EQ(scene.plan->kernelArgsSize,
            std::max(ncclSymkDevWorkArgs::calcArgsSize(kHeadNMaxChannels, kSymSchedBigBatchTasks, false),
                     sizeof(ncclSymkDevWorkArgs4K)));
}

// A hardcoded cellCount would give alignUp(1,1024)=1024 here, not the correct alignUp(1,256)=256 for float32.
TEST_F(SchedulerMicrotest, SymmetricTaskScheduler_CellCount_ScalesWithTypeSize_NotHardcoded) {
  SymmetricTaskScheduler_Scene scene;
  ncclTaskColl task = SymmetricTaskScheduler_MakeTask();
  task.datatype = ncclFloat32;  // cellCount = 1024 / ncclTypeSize(ncclFloat32) = 256
  task.count = 1;
  ncclIntruQueueEnqueue(&scene.symTaskQueue, &task);
  EXPECT_EQ(ncclSymmetricTaskScheduler(scene.comm.get(), &scene.symTaskQueue, scene.plan.get()), ncclSuccess);
  EXPECT_EQ(scene.plan->workBytes, 256u * 4u);  // totalCount(256) * ncclTypeSize(ncclFloat32)(4)
}

// A non-floored sanity signal; workCount's own mutants need the packing loop's separate checks (big-batch tests below).
TEST_F(SchedulerMicrotest, SymmetricTaskScheduler_WorkCountingLoop_SingleTask_ProcessesExactlyOnce) {
  SymmetricTaskScheduler_Scene scene;
  ncclTaskColl task = SymmetricTaskScheduler_MakeTask();
  ncclIntruQueueEnqueue(&scene.symTaskQueue, &task);
  int makeDevWorkCalls = 0;
  ScopedHook makeDevWorkHook(g_symkMakeDevWork,
                             [&](struct ncclComm*, struct ncclTaskColl*, struct ncclSymkDevWork*) {
                               ++makeDevWorkCalls;
                               return ncclSuccess;
                             });
  EXPECT_EQ(ncclSymmetricTaskScheduler(scene.comm.get(), &scene.symTaskQueue, scene.plan.get()), ncclSuccess);
  EXPECT_EQ(makeDevWorkCalls, 1);
}

TEST_F(SchedulerMicrotest, SymmetricTaskScheduler_WorkCountingLoop_ManySameDevFuncIdTasks_AggregateWorkCount) {
  SymmetricTaskScheduler_Scene scene;
  auto tasks = SymmetricTaskScheduler_EnqueueBatch(&scene.symTaskQueue, kSymSchedBigBatchTasks,
                                                    ncclSymkKernelId_AllGather_LL, /*lastIsSymLast=*/1);
  int makeDevWorkCalls = 0;
  ScopedHook makeDevWorkHook(g_symkMakeDevWork,
                             [&](struct ncclComm*, struct ncclTaskColl*, struct ncclSymkDevWork*) {
                               ++makeDevWorkCalls;
                               return ncclSuccess;
                             });
  EXPECT_EQ(ncclSymmetricTaskScheduler(scene.comm.get(), &scene.symTaskQueue, scene.plan.get()), ncclSuccess);
  // workCount-specific: one ncclSymkMakeDevWork call per task, so all kSymSchedBigBatchTasks were counted as one batch.
  EXPECT_EQ(makeDevWorkCalls, kSymSchedBigBatchTasks);
}

TEST_F(SchedulerMicrotest, SymmetricTaskScheduler_WorkCountingLoop_StopsAtDevFuncIdMismatch_ExcludesTrailingTask) {
  SymmetricTaskScheduler_Scene scene;
  auto tasks = SymmetricTaskScheduler_EnqueueBatch(&scene.symTaskQueue, kSymSchedBigBatchTasks,
                                                    ncclSymkKernelId_AllGather_LL, /*lastIsSymLast=*/0);
  ncclTaskColl mismatchTask = SymmetricTaskScheduler_MakeTask();
  mismatchTask.devFuncId = ncclSymkKernelId_AllGather_LLMC;  // different batch, appended after the big one
  ncclIntruQueueEnqueue(&scene.symTaskQueue, &mismatchTask);
  EXPECT_EQ(ncclSymmetricTaskScheduler(scene.comm.get(), &scene.symTaskQueue, scene.plan.get()), ncclSuccess);
  EXPECT_EQ(scene.plan->kernelArgsSize,
            std::max(ncclSymkDevWorkArgs::calcArgsSize(kSymSchedBigBatchChannels, kSymSchedBigBatchTasks, false),
                     sizeof(ncclSymkDevWorkArgs4K)));
}

TEST_F(SchedulerMicrotest, SymmetricTaskScheduler_WorkCountingLoop_IsSymLastBreak_ExcludesLaterSameBatchTask) {
  SymmetricTaskScheduler_Scene scene;
  auto tasks = SymmetricTaskScheduler_EnqueueBatch(&scene.symTaskQueue, kSymSchedBigBatchTasks,
                                                    ncclSymkKernelId_AllGather_LL, /*lastIsSymLast=*/1);
  ncclTaskColl trailingTask = SymmetricTaskScheduler_MakeTask();  // same devFuncId as the batch, appended after
  trailingTask.isSymLast = 0;
  ncclIntruQueueEnqueue(&scene.symTaskQueue, &trailingTask);
  EXPECT_EQ(ncclSymmetricTaskScheduler(scene.comm.get(), &scene.symTaskQueue, scene.plan.get()), ncclSuccess);
  EXPECT_EQ(scene.plan->kernelArgsSize,
            std::max(ncclSymkDevWorkArgs::calcArgsSize(kSymSchedBigBatchChannels, kSymSchedBigBatchTasks, false),
                     sizeof(ncclSymkDevWorkArgs4K)));
}

// Exercises task != nullptr alone: the chain ends before isSymLast ever fires, so only the null check stops it.
TEST_F(SchedulerMicrotest, SymmetricTaskScheduler_WorkCountingLoop_TerminatesAtChainEnd_WithoutIsSymLast) {
  SymmetricTaskScheduler_Scene scene;
  auto tasks = SymmetricTaskScheduler_EnqueueBatch(&scene.symTaskQueue, kSymSchedBigBatchTasks,
                                                    ncclSymkKernelId_AllGather_LL, /*lastIsSymLast=*/0);
  EXPECT_EQ(ncclSymmetricTaskScheduler(scene.comm.get(), &scene.symTaskQueue, scene.plan.get()), ncclSuccess);
  EXPECT_EQ(scene.plan->kernelArgsSize,
            std::max(ncclSymkDevWorkArgs::calcArgsSize(kSymSchedBigBatchChannels, kSymSchedBigBatchTasks, false),
                     sizeof(ncclSymkDevWorkArgs4K)));
}

TEST_F(SchedulerMicrotest, SymmetricTaskScheduler_WorkCountingLoop_TotalCount_AlignsEachTaskUpToCellCount) {
  SymmetricTaskScheduler_Scene scene;
  ncclTaskColl task1 = SymmetricTaskScheduler_MakeTask();
  task1.count = 1;  // alignUp(1, 1024) = 1024
  task1.isSymLast = 0;
  ncclTaskColl task2 = SymmetricTaskScheduler_MakeTask();
  task2.count = 1025;  // alignUp(1025, 1024) = 2048
  task2.isSymLast = 1;
  ncclIntruQueueEnqueue(&scene.symTaskQueue, &task1);
  ncclIntruQueueEnqueue(&scene.symTaskQueue, &task2);
  EXPECT_EQ(ncclSymmetricTaskScheduler(scene.comm.get(), &scene.symTaskQueue, scene.plan.get()), ncclSuccess);
  EXPECT_EQ(scene.plan->workBytes, (1024u + 2048u) * 1u);  // totalCount * ncclTypeSize(ncclInt8)
}

TEST_F(SchedulerMicrotest, SymmetricTaskScheduler_WorkCountingLoop_CgaClusterSize_PropagatedWhenNotUndef) {
  SymmetricTaskScheduler_Scene scene;
  ncclTaskColl task = SymmetricTaskScheduler_MakeTask();
  task.cgaClusterSize = 77;
  ncclIntruQueueEnqueue(&scene.symTaskQueue, &task);
  scene.plan->cgaClusterSize = -999;  // sentinel: must be overwritten, not already 77 by chance
  EXPECT_EQ(ncclSymmetricTaskScheduler(scene.comm.get(), &scene.symTaskQueue, scene.plan.get()), ncclSuccess);
  EXPECT_EQ(scene.plan->cgaClusterSize, 77);
}

TEST_F(SchedulerMicrotest, SymmetricTaskScheduler_WorkCountingLoop_CgaClusterSize_NotPropagatedWhenUndef) {
  SymmetricTaskScheduler_Scene scene;
  ncclTaskColl task = SymmetricTaskScheduler_MakeTask();  // cgaClusterSize == NCCL_CONFIG_UNDEF_INT
  ncclIntruQueueEnqueue(&scene.symTaskQueue, &task);
  scene.plan->cgaClusterSize = 555;
  EXPECT_EQ(ncclSymmetricTaskScheduler(scene.comm.get(), &scene.symTaskQueue, scene.plan.get()), ncclSuccess);
  EXPECT_EQ(scene.plan->cgaClusterSize, 555);  // guard correctly skipped the write
}

// logCount uses each task's raw count (1 + 1025 = 1026 Bytes), not the cell-aligned totalCount (3072) above.
// logCount (symmetric_sched.cc:348) is a local accumulator with no observable beyond the final INFO
// line, so its raw-vs-aligned distinction is deliberately left uncovered rather than pinned to log text.

// ncclSymmetricTaskScheduler args-buffer allocation (symmetric_sched.cc:354-364); packing's own fields unasserted.

// calcArgsSize(2,1,false) is well below the 4096-byte floor, so max() must pick the floor here.
TEST_F(SchedulerMicrotest, SymmetricTaskScheduler_ArgsSize_FloorWins_BelowCalcArgsSize) {
  SymmetricTaskScheduler_Scene scene;
  ncclTaskColl task = SymmetricTaskScheduler_MakeTask();
  ncclIntruQueueEnqueue(&scene.symTaskQueue, &task);
  EXPECT_EQ(ncclSymmetricTaskScheduler(scene.comm.get(), &scene.symTaskQueue, scene.plan.get()), ncclSuccess);
  EXPECT_EQ(scene.plan->kernelArgsSize, sizeof(ncclSymkDevWorkArgs4K));
}

// The big batch's raw calcArgsSize clears the 4096 floor; a crash here would mean calloc's size didn't match.
TEST_F(SchedulerMicrotest, SymmetricTaskScheduler_ArgsSize_CalcWins_AboveFloor) {
  SymmetricTaskScheduler_Scene scene;
  auto tasks = SymmetricTaskScheduler_EnqueueBatch(&scene.symTaskQueue, kSymSchedBigBatchTasks,
                                                    ncclSymkKernelId_AllGather_LL, /*lastIsSymLast=*/1);
  EXPECT_EQ(ncclSymmetricTaskScheduler(scene.comm.get(), &scene.symTaskQueue, scene.plan.get()), ncclSuccess);
  EXPECT_EQ(scene.plan->kernelArgsSize,
            ncclSymkDevWorkArgs::calcArgsSize(kSymSchedBigBatchChannels, kSymSchedBigBatchTasks, false));
  ASSERT_GT(scene.plan->kernelArgsSize, sizeof(ncclSymkDevWorkArgs4K));
}

// A big batch with profiling on: the profiler-counters region is large enough that dropping it would crash.
TEST_F(SchedulerMicrotest, SymmetricTaskScheduler_ArgsSize_IncludesProfilerCountersRegion_WhenEnabled) {
  SymmetricTaskScheduler_Scene scene;
  ScopedHook pluginHook(g_profilerPluginLoaded, []() { return true; });
  auto tasks = SymmetricTaskScheduler_EnqueueBatch(&scene.symTaskQueue, kSymSchedBigBatchTasks,
                                                    ncclSymkKernelId_AllGather_LL, /*lastIsSymLast=*/1);
  tasks[0].eActivationMask = ncclProfileKernelCh;
  scene.plan->persistent = false;
  EXPECT_EQ(ncclSymmetricTaskScheduler(scene.comm.get(), &scene.symTaskQueue, scene.plan.get()), ncclSuccess);
  EXPECT_EQ(scene.plan->kernelArgsSize,
            ncclSymkDevWorkArgs::calcArgsSize(kSymSchedBigBatchChannels, kSymSchedBigBatchTasks, true));
}

TEST_F(SchedulerMicrotest, SymmetricTaskScheduler_ArgsBufNMaxChannels_SetFromLocal) {
  SymmetricTaskScheduler_Scene scene;
  ncclTaskColl task = SymmetricTaskScheduler_MakeTask();
  task.nMaxChannels = 5;
  ncclIntruQueueEnqueue(&scene.symTaskQueue, &task);
  EXPECT_EQ(ncclSymmetricTaskScheduler(scene.comm.get(), &scene.symTaskQueue, scene.plan.get()), ncclSuccess);
  auto* argsBuf = static_cast<struct ncclSymkDevWorkArgs*>(scene.plan->kernelSymArgs);
  ASSERT_NE(argsBuf, nullptr);
  EXPECT_EQ(argsBuf->nMaxChannels, 5);
}

// ncclSymkKernelMaxDynamicSmem[0] flows into argsBuf->maxDynamicSmem unconditionally, unlike plan->kernelDynSmem.
TEST_F(SchedulerMicrotest, SymmetricTaskScheduler_ArgsBufMaxDynamicSmem_SetFromLocal) {
  SymmetricTaskScheduler_Scene scene;
  ncclSymkKernelMaxDynamicSmem[0] = 777;
  ncclTaskColl task = SymmetricTaskScheduler_MakeTask();
  ncclIntruQueueEnqueue(&scene.symTaskQueue, &task);
  EXPECT_EQ(ncclSymmetricTaskScheduler(scene.comm.get(), &scene.symTaskQueue, scene.plan.get()), ncclSuccess);
  auto* argsBuf = static_cast<struct ncclSymkDevWorkArgs*>(scene.plan->kernelSymArgs);
  ASSERT_NE(argsBuf, nullptr);
  EXPECT_EQ(argsBuf->maxDynamicSmem, 777);
}

TEST_F(SchedulerMicrotest, SymmetricTaskScheduler_ArgsBufProfilerEnabled_SetOneWhenTrue) {
  SymmetricTaskScheduler_Scene scene;
  ScopedHook pluginHook(g_profilerPluginLoaded, []() { return true; });
  ncclTaskColl task = SymmetricTaskScheduler_MakeTask();
  task.eActivationMask = ncclProfileKernelCh;
  ncclIntruQueueEnqueue(&scene.symTaskQueue, &task);
  scene.plan->persistent = false;
  EXPECT_EQ(ncclSymmetricTaskScheduler(scene.comm.get(), &scene.symTaskQueue, scene.plan.get()), ncclSuccess);
  auto* argsBuf = static_cast<struct ncclSymkDevWorkArgs*>(scene.plan->kernelSymArgs);
  ASSERT_NE(argsBuf, nullptr);
  EXPECT_EQ(argsBuf->profilerEnabled, 1);
}

TEST_F(SchedulerMicrotest, SymmetricTaskScheduler_ArgsBufProfilerEnabled_SetZeroWhenFalse) {
  SymmetricTaskScheduler_Scene scene;
  ncclTaskColl task = SymmetricTaskScheduler_MakeTask();  // eActivationMask left at 0: profiling never requested
  ncclIntruQueueEnqueue(&scene.symTaskQueue, &task);
  EXPECT_EQ(ncclSymmetricTaskScheduler(scene.comm.get(), &scene.symTaskQueue, scene.plan.get()), ncclSuccess);
  auto* argsBuf = static_cast<struct ncclSymkDevWorkArgs*>(scene.plan->kernelSymArgs);
  ASSERT_NE(argsBuf, nullptr);
  EXPECT_EQ(argsBuf->profilerEnabled, 0);
}

// profilerEnabled=true catches ordering bugs; nMaxChannels=5 (not 3) avoids the alignUp(n*4,16)==16 bucket tie.
TEST_F(SchedulerMicrotest, SymmetricTaskScheduler_WorkRangeAndWorks_PointAtIndependentlyComputedOffsets) {
  SymmetricTaskScheduler_Scene scene;
  ScopedHook pluginHook(g_profilerPluginLoaded, []() { return true; });
  ncclTaskColl task = SymmetricTaskScheduler_MakeTask();
  task.nMaxChannels = 5;
  task.eActivationMask = ncclProfileKernelCh;
  ncclIntruQueueEnqueue(&scene.symTaskQueue, &task);
  scene.plan->persistent = false;
  constexpr uint64_t kSentinel = 0x1234567890ABCDEFull;
  ScopedHook makeDevWorkHook(g_symkMakeDevWork,
                             [&](struct ncclComm*, struct ncclTaskColl*, struct ncclSymkDevWork* out) {
                               out->redOpArg = kSentinel;
                               return ncclSuccess;
                             });
  EXPECT_EQ(ncclSymmetricTaskScheduler(scene.comm.get(), &scene.symTaskQueue, scene.plan.get()), ncclSuccess);
  auto* argsBuf = static_cast<struct ncclSymkDevWorkArgs*>(scene.plan->kernelSymArgs);
  ASSERT_NE(argsBuf, nullptr);
  uint8_t* base = reinterpret_cast<uint8_t*>(argsBuf);
  size_t workRangeOff =
      alignUp(sizeof(struct ncclSymkDevWorkArgs), size_t(16)) + alignUp(size_t(5) * sizeof(uint64_t), size_t(16));
  size_t worksOff = workRangeOff + alignUp(size_t(5) * sizeof(struct ncclSymkChannelWorkRange), size_t(16));
  auto* expectedWorkRange = reinterpret_cast<struct ncclSymkChannelWorkRange*>(base + workRangeOff);
  auto* expectedWorks = reinterpret_cast<struct ncclSymkDevWork*>(base + worksOff);
  EXPECT_EQ(expectedWorkRange[0].fracHi, uint16_t(0xFFFF));
  EXPECT_EQ(expectedWorks[0].redOpArg, kSentinel);
}

// 3 one-cell tasks, not 1: a byte-scale (missing cellCount division) remainCell still sets fracHi on 1 task.
TEST_F(SchedulerMicrotest, SymmetricTaskScheduler_RemainCell_DerivedFromTotalCountNMaxChannelsCellCount) {
  SymmetricTaskScheduler_Scene scene;
  ncclTaskColl task1 = SymmetricTaskScheduler_MakeTask();
  task1.nMaxChannels = 3;
  task1.isSymLast = 0;
  ncclTaskColl task2 = SymmetricTaskScheduler_MakeTask();
  task2.nMaxChannels = 3;
  task2.isSymLast = 0;
  ncclTaskColl task3 = SymmetricTaskScheduler_MakeTask();
  task3.nMaxChannels = 3;
  task3.isSymLast = 1;
  ncclIntruQueueEnqueue(&scene.symTaskQueue, &task1);
  ncclIntruQueueEnqueue(&scene.symTaskQueue, &task2);
  ncclIntruQueueEnqueue(&scene.symTaskQueue, &task3);
  EXPECT_EQ(ncclSymmetricTaskScheduler(scene.comm.get(), &scene.symTaskQueue, scene.plan.get()), ncclSuccess);
  EXPECT_EQ(scene.plan->channelMask.masks[0], 0b111ull);
}

// A logCount-not-totalCount mutant overruns nMaxChannels=1; that goto fail skips ret, so kernelSymArgs signals it.
TEST_F(SchedulerMicrotest, SymmetricTaskScheduler_RemainCell_UsesAlignedTotalCountNotRawLogCount) {
  SymmetricTaskScheduler_Scene scene;
  std::vector<ncclTaskColl> tasks(5);
  for (int i = 0; i < 5; ++i) {
    tasks[i] = SymmetricTaskScheduler_MakeTask();
    tasks[i].nMaxChannels = 1;
    tasks[i].count = 1;  // raw count tiny; alignUp(1,1024) still contributes a full 1024-byte cell
    tasks[i].isSymLast = (i == 4) ? 1 : 0;
    ncclIntruQueueEnqueue(&scene.symTaskQueue, &tasks[i]);
  }
  EXPECT_EQ(ncclSymmetricTaskScheduler(scene.comm.get(), &scene.symTaskQueue, scene.plan.get()), ncclSuccess);
  EXPECT_NE(scene.plan->kernelSymArgs, nullptr);
}

// Packing loop dequeue/boundary + single-task traversal (symmetric_sched.cc:367-444); cross-task carry is covered
// by the SymmetricTaskScheduler_ContinuingTask_* tests further down.

TEST_F(SchedulerMicrotest, SymmetricTaskScheduler_PackingLoop_ProcessesAllTasks_UntilQueueDrains) {
  SymmetricTaskScheduler_Scene scene;
  std::vector<ncclTaskColl> tasks(3);
  for (int i = 0; i < 3; ++i) {
    tasks[i] = SymmetricTaskScheduler_MakeTask();
    tasks[i].nMaxChannels = 4;  // headroom above 3 tasks * 1 channel each: no exhaustion once the queue drains
    tasks[i].isSymLast = 0;     // none is last: only the queue running out should stop the loop
    ncclIntruQueueEnqueue(&scene.symTaskQueue, &tasks[i]);
  }
  int makeDevWorkCalls = 0;
  ScopedHook makeDevWorkHook(g_symkMakeDevWork,
                             [&](struct ncclComm*, struct ncclTaskColl*, struct ncclSymkDevWork*) {
                               ++makeDevWorkCalls;
                               return ncclSuccess;
                             });
  EXPECT_EQ(ncclSymmetricTaskScheduler(scene.comm.get(), &scene.symTaskQueue, scene.plan.get()), ncclSuccess);
  EXPECT_EQ(makeDevWorkCalls, 3);
  EXPECT_NE(scene.plan->kernelSymArgs, nullptr);
  // workHi is stamped with each task's own workIndex (0,1,2), distinct from calloc's zero default.
  auto* argsBuf = static_cast<struct ncclSymkDevWorkArgs*>(scene.plan->kernelSymArgs);
  auto* workRange = argsBuf->getWorkRange();
  EXPECT_EQ(workRange[0].workHi, 0u);
  EXPECT_EQ(workRange[1].workHi, 1u);
  EXPECT_EQ(workRange[2].workHi, 2u);
}

TEST_F(SchedulerMicrotest, SymmetricTaskScheduler_PackingLoop_StopsAtDevFuncIdMismatch_LeavesTrailingTaskQueued) {
  SymmetricTaskScheduler_Scene scene;
  ncclTaskColl task1 = SymmetricTaskScheduler_MakeTask();
  task1.isSymLast = 0;
  ncclTaskColl task2 = SymmetricTaskScheduler_MakeTask();
  task2.devFuncId = ncclSymkKernelId_AllGather_LLMC;  // different batch
  ncclIntruQueueEnqueue(&scene.symTaskQueue, &task1);
  ncclIntruQueueEnqueue(&scene.symTaskQueue, &task2);
  int makeDevWorkCalls = 0;
  ScopedHook makeDevWorkHook(g_symkMakeDevWork,
                             [&](struct ncclComm*, struct ncclTaskColl*, struct ncclSymkDevWork*) {
                               ++makeDevWorkCalls;
                               return ncclSuccess;
                             });
  EXPECT_EQ(ncclSymmetricTaskScheduler(scene.comm.get(), &scene.symTaskQueue, scene.plan.get()), ncclSuccess);
  EXPECT_EQ(makeDevWorkCalls, 1);  // task2's mismatched devFuncId stops the loop before it is ever dequeued
  EXPECT_EQ(ncclIntruQueueHead(&scene.symTaskQueue), &task2);
}

TEST_F(SchedulerMicrotest, SymmetricTaskScheduler_MakeDevWork_Fails_PropagatesErrorAndStopsProcessing) {
  SymmetricTaskScheduler_Scene scene;
  ncclTaskColl task1 = SymmetricTaskScheduler_MakeTask();
  task1.isSymLast = 0;
  ncclTaskColl task2 = SymmetricTaskScheduler_MakeTask();
  task2.isSymLast = 1;
  ncclIntruQueueEnqueue(&scene.symTaskQueue, &task1);
  ncclIntruQueueEnqueue(&scene.symTaskQueue, &task2);
  int makeDevWorkCalls = 0;
  ScopedHook makeDevWorkHook(g_symkMakeDevWork,
                             [&](struct ncclComm*, struct ncclTaskColl*, struct ncclSymkDevWork*) {
                               ++makeDevWorkCalls;
                               return ncclInternalError;
                             });
  EXPECT_EQ(ncclSymmetricTaskScheduler(scene.comm.get(), &scene.symTaskQueue, scene.plan.get()), ncclInternalError);
  EXPECT_EQ(makeDevWorkCalls, 1);  // task2 is never reached once task1's ncclSymkMakeDevWork call fails
}

// The first for-loop pass on a fresh channel always sets sChannelId/nChannels directly (not via fracHi).
TEST_F(SchedulerMicrotest, SymmetricTaskScheduler_FirstSegment_SetsSChannelIdAndNChannelsOnFreshChannel) {
  SymmetricTaskScheduler_Scene scene;
  ncclTaskColl task = SymmetricTaskScheduler_MakeTask();  // 1 cell, nMaxChannels=2: exact single-channel fill
  ncclIntruQueueEnqueue(&scene.symTaskQueue, &task);
  EXPECT_EQ(ncclSymmetricTaskScheduler(scene.comm.get(), &scene.symTaskQueue, scene.plan.get()), ncclSuccess);
  auto* argsBuf = static_cast<struct ncclSymkDevWorkArgs*>(scene.plan->kernelSymArgs);
  ASSERT_NE(argsBuf, nullptr);
  auto* works = argsBuf->getWorks(2);
  EXPECT_EQ(works[0].sChannelId, 0u);
  EXPECT_EQ(works[0].nChannels, 1u);
}

// task1 exactly fills channel 0; task2 (last) then starts fresh on channel 1 and ends mid-channel via "<".
TEST_F(SchedulerMicrotest, SymmetricTaskScheduler_CellLeftLessThanRemainCell_EndsLastTaskMidChannel) {
  SymmetricTaskScheduler_Scene scene;
  ncclTaskColl task1 = SymmetricTaskScheduler_MakeTask();
  task1.nMaxChannels = 2;
  task1.count = 2048;  // 2 cells: exactly fills channel 0's cellPerChannel==2 budget
  task1.isSymLast = 0;
  ncclTaskColl task2 = SymmetricTaskScheduler_MakeTask();
  task2.nMaxChannels = 2;
  task2.count = 1024;  // 1 cell: strictly less than channel 1's remainCell==2 budget
  task2.isSymLast = 1;
  ncclIntruQueueEnqueue(&scene.symTaskQueue, &task1);
  ncclIntruQueueEnqueue(&scene.symTaskQueue, &task2);
  EXPECT_EQ(ncclSymmetricTaskScheduler(scene.comm.get(), &scene.symTaskQueue, scene.plan.get()), ncclSuccess);
  auto* argsBuf = static_cast<struct ncclSymkDevWorkArgs*>(scene.plan->kernelSymArgs);
  ASSERT_NE(argsBuf, nullptr);
  auto* workRange = argsBuf->getWorkRange();
  EXPECT_EQ(workRange[0].fracHi, uint16_t(0xFFFF));  // task1's exact fill
  EXPECT_EQ(workRange[1].fracHi, uint16_t(0xFFFF));  // task2's partial fill (fracHi is 0xFFFF either way)
  auto* works = argsBuf->getWorks(2);
  EXPECT_EQ(works[1].sChannelId, 1u);
  EXPECT_EQ(works[1].nChannels, 1u);  // task2 never advanced past its own single (unfinished) channel
}

// Spans 2 channels with a 0-cell leftover (fuse true by size); a trailing queued task isolates that from queue-empty.
TEST_F(SchedulerMicrotest, SymmetricTaskScheduler_TaskSpansExactlyTwoChannels_FuseHeuristicTrue) {
  SymmetricTaskScheduler_Scene scene;
  ncclTaskColl task = SymmetricTaskScheduler_MakeTask();
  task.nMaxChannels = 2;
  task.count = 2048;  // 2 cells; cellPerChannel==1: channel 0 overflows, channel 1 exactly finishes (fuses)
  task.isSymLast = 1;  // stops the loop right after this task, so trailingTask below is never dequeued
  ncclTaskColl trailingTask = SymmetricTaskScheduler_MakeTask();
  ncclIntruQueueEnqueue(&scene.symTaskQueue, &task);
  ncclIntruQueueEnqueue(&scene.symTaskQueue, &trailingTask);
  EXPECT_EQ(ncclSymmetricTaskScheduler(scene.comm.get(), &scene.symTaskQueue, scene.plan.get()), ncclSuccess);
  auto* argsBuf = static_cast<struct ncclSymkDevWorkArgs*>(scene.plan->kernelSymArgs);
  ASSERT_NE(argsBuf, nullptr);
  auto* works = argsBuf->getWorks(2);
  EXPECT_EQ(works[0].sChannelId, 0u);
  EXPECT_EQ(works[0].nChannels, 2u);  // fused: both channels counted on the one devWork entry
}

// Spans 4 channels (2 middle segments) with a >1-cell leftover and a trailing queued task: both OR arms false.
TEST_F(SchedulerMicrotest, SymmetricTaskScheduler_TaskSpansFourChannels_MiddleSegments_FuseHeuristicFalse) {
  SymmetricTaskScheduler_Scene scene;
  ncclTaskColl task = SymmetricTaskScheduler_MakeTask();
  task.nMaxChannels = 4;
  task.count = 13312;  // 13 cells; cellPerChannel==4: channels 0,1,2 overflow, channel 3's leftover is 3 cells
  task.isSymLast = 1;  // stops the loop right after this task, so trailingTask below is never dequeued
  ncclTaskColl trailingTask = SymmetricTaskScheduler_MakeTask();
  ncclIntruQueueEnqueue(&scene.symTaskQueue, &task);
  ncclIntruQueueEnqueue(&scene.symTaskQueue, &trailingTask);
  EXPECT_EQ(ncclSymmetricTaskScheduler(scene.comm.get(), &scene.symTaskQueue, scene.plan.get()), ncclSuccess);
  auto* argsBuf = static_cast<struct ncclSymkDevWorkArgs*>(scene.plan->kernelSymArgs);
  ASSERT_NE(argsBuf, nullptr);
  auto* works = argsBuf->getWorks(4);
  auto* workRange = argsBuf->getWorkRange();
  EXPECT_EQ(works[0].sChannelId, 0u);
  EXPECT_EQ(works[0].nChannels, 3u);  // not fused: channel 3's leftover is left out of the count
  // Each of the 4 channels was actually visited (curChannel advanced), not just channel 0 written repeatedly.
  EXPECT_EQ(workRange[0].fracHi, 20164);
  EXPECT_EQ(workRange[1].fracHi, 40329);
  EXPECT_EQ(workRange[2].fracHi, 60494);
  EXPECT_EQ(workRange[3].fracHi, uint16_t(0xFFFF));
}

// Same leftover as FuseHeuristicFalse above, but no trailing task: queue-empty alone forces the fuse instead.
TEST_F(SchedulerMicrotest, SymmetricTaskScheduler_TaskSpansFourChannels_FusesViaQueueEmptyDespiteLargeLeftover) {
  SymmetricTaskScheduler_Scene scene;
  ncclTaskColl task = SymmetricTaskScheduler_MakeTask();
  task.nMaxChannels = 4;
  task.count = 13312;  // same 3-cell leftover on the last segment as the sibling test above
  ncclIntruQueueEnqueue(&scene.symTaskQueue, &task);  // no trailing task: queue is empty once this is dequeued
  EXPECT_EQ(ncclSymmetricTaskScheduler(scene.comm.get(), &scene.symTaskQueue, scene.plan.get()), ncclSuccess);
  auto* argsBuf = static_cast<struct ncclSymkDevWorkArgs*>(scene.plan->kernelSymArgs);
  ASSERT_NE(argsBuf, nullptr);
  auto* works = argsBuf->getWorks(4);
  EXPECT_EQ(works[0].nChannels, 4u);  // fused via queue-empty, despite the same large leftover as above
}

TEST_F(SchedulerMicrotest, SymmetricTaskScheduler_PostForLoop_WorkIndexAndCollTaskQueueAndGroupApiEventHandle) {
  SymmetricTaskScheduler_Scene scene;
  ncclTaskColl task = SymmetricTaskScheduler_MakeTask();
  static int sentinelHandle = 0;
  task.groupApiEventHandle = &sentinelHandle;
  ncclIntruQueueEnqueue(&scene.symTaskQueue, &task);
  EXPECT_EQ(ncclSymmetricTaskScheduler(scene.comm.get(), &scene.symTaskQueue, scene.plan.get()), ncclSuccess);
  EXPECT_EQ(scene.plan->groupApiEventHandle, &sentinelHandle);
  ncclTaskColl* queued = ncclIntruQueueHead(&scene.plan->collTaskQueue);
  ASSERT_NE(queued, nullptr);
  EXPECT_EQ(queued, &task);
}

TEST_F(SchedulerMicrotest, SymmetricTaskScheduler_IsSymLastBreak_StopsPackingLoopWithMoreTasksQueued) {
  SymmetricTaskScheduler_Scene scene;
  ncclTaskColl task1 = SymmetricTaskScheduler_MakeTask();
  task1.isSymLast = 1;  // breaks here even though task2 below shares its devFuncId
  ncclTaskColl task2 = SymmetricTaskScheduler_MakeTask();
  task2.isSymLast = 0;
  ncclIntruQueueEnqueue(&scene.symTaskQueue, &task1);
  ncclIntruQueueEnqueue(&scene.symTaskQueue, &task2);
  int makeDevWorkCalls = 0;
  ScopedHook makeDevWorkHook(g_symkMakeDevWork,
                             [&](struct ncclComm*, struct ncclTaskColl*, struct ncclSymkDevWork*) {
                               ++makeDevWorkCalls;
                               return ncclSuccess;
                             });
  EXPECT_EQ(ncclSymmetricTaskScheduler(scene.comm.get(), &scene.symTaskQueue, scene.plan.get()), ncclSuccess);
  EXPECT_EQ(makeDevWorkCalls, 1);
}

TEST_F(SchedulerMicrotest, SymmetricTaskScheduler_ChannelExhaustion_NotIsSymLast_SkipsTailCode) {
  SymmetricTaskScheduler_Scene scene;
  ncclTaskColl task = SymmetricTaskScheduler_MakeTask();
  task.nMaxChannels = 1;
  task.isSymLast = 0;  // not last: the exhaustion check below actually gets reached
  ncclIntruQueueEnqueue(&scene.symTaskQueue, &task);
  // ncclSuccess here is a production bug: the goto fail at symmetric_sched.cc:442 never sets ret first.
  EXPECT_EQ(ncclSymmetricTaskScheduler(scene.comm.get(), &scene.symTaskQueue, scene.plan.get()), ncclSuccess);
  EXPECT_EQ(scene.plan->kernelSymArgs, nullptr);
}

// Cross-task continuation (symmetric_sched.cc:400-410): task1 ends via "<", leaving task2 to start mid-channel.

// task2's own taskCell(1) fits within task1's leftover remainCell(1): continues filling the SAME channel.
TEST_F(SchedulerMicrotest, SymmetricTaskScheduler_ContinuingTask_FitsSharedChannel_JoinsSameChannel) {
  SymmetricTaskScheduler_Scene scene;
  ncclTaskColl task1 = SymmetricTaskScheduler_MakeTask();
  task1.nMaxChannels = 1;
  task1.count = 1024;  // 1 cell; leaves remainCell==1 of the aggregate cellPerChannel==2
  task1.isSymLast = 0;
  ncclTaskColl task2 = SymmetricTaskScheduler_MakeTask();
  task2.nMaxChannels = 1;
  task2.count = 1024;  // 1 cell; fits exactly into task1's leftover (taskCell <= remainCell)
  task2.isSymLast = 1;
  ncclIntruQueueEnqueue(&scene.symTaskQueue, &task1);
  ncclIntruQueueEnqueue(&scene.symTaskQueue, &task2);
  // Poison so line 407's assignment (not the calloc'd default) is what the assertion below actually pins.
  ScopedHook makeDevWorkHook(g_symkMakeDevWork,
                             [](struct ncclComm*, struct ncclTaskColl*, struct ncclSymkDevWork* outDevWork) {
                               outDevWork->sChannelId = 0xffff;
                               return ncclSuccess;
                             });
  EXPECT_EQ(ncclSymmetricTaskScheduler(scene.comm.get(), &scene.symTaskQueue, scene.plan.get()), ncclSuccess);
  auto* argsBuf = static_cast<struct ncclSymkDevWorkArgs*>(scene.plan->kernelSymArgs);
  ASSERT_NE(argsBuf, nullptr);
  auto* works = argsBuf->getWorks(1);
  EXPECT_EQ(works[1].sChannelId, 0u);  // task2's own devWork: same channel task1 left off on
  EXPECT_EQ(works[1].nChannels, 1u);
}

// task2's taskCell(3) exceeds task1's leftover(1): overflows immediately, landing on a NEW channel, not the shared one.
TEST_F(SchedulerMicrotest, SymmetricTaskScheduler_ContinuingTask_OverflowsSharedChannel_AttributesToNewChannel) {
  SymmetricTaskScheduler_Scene scene;
  ncclTaskColl task1 = SymmetricTaskScheduler_MakeTask();
  task1.nMaxChannels = 2;
  task1.count = 1024;  // 1 cell; leaves remainCell==1 of the aggregate cellPerChannel==2
  task1.isSymLast = 0;
  ncclTaskColl task2 = SymmetricTaskScheduler_MakeTask();
  task2.nMaxChannels = 2;
  task2.count = 3072;  // 3 cells; overflows task1's 1-cell leftover on its very first touch
  task2.isSymLast = 1;
  ncclIntruQueueEnqueue(&scene.symTaskQueue, &task1);
  ncclIntruQueueEnqueue(&scene.symTaskQueue, &task2);
  EXPECT_EQ(ncclSymmetricTaskScheduler(scene.comm.get(), &scene.symTaskQueue, scene.plan.get()), ncclSuccess);
  auto* argsBuf = static_cast<struct ncclSymkDevWorkArgs*>(scene.plan->kernelSymArgs);
  ASSERT_NE(argsBuf, nullptr);
  auto* works = argsBuf->getWorks(2);
  EXPECT_EQ(works[1].sChannelId, 1u);  // not channel 0 (task1's channel), but the new channel task2 lands on
  EXPECT_EQ(works[1].nChannels, 1u);
}

// Tail past the packing loop (symmetric_sched.cc:449,455): the last two previously-unasserted straight-line writes.

TEST_F(SchedulerMicrotest, SymmetricTaskScheduler_ArgsBufKcomm_CopiedFromCommSymkState) {
  SymmetricTaskScheduler_Scene scene;
  memset(&scene.comm->symkState.kcomm, 0xAB, sizeof(scene.comm->symkState.kcomm));
  ncclTaskColl task = SymmetricTaskScheduler_MakeTask();
  ncclIntruQueueEnqueue(&scene.symTaskQueue, &task);
  EXPECT_EQ(ncclSymmetricTaskScheduler(scene.comm.get(), &scene.symTaskQueue, scene.plan.get()), ncclSuccess);
  auto* argsBuf = static_cast<struct ncclSymkDevWorkArgs*>(scene.plan->kernelSymArgs);
  ASSERT_NE(argsBuf, nullptr);
  EXPECT_EQ(memcmp(&argsBuf->kcomm, &scene.comm->symkState.kcomm, sizeof(argsBuf->kcomm)), 0);
}

TEST_F(SchedulerMicrotest, SymmetricTaskScheduler_WorkStorageType_SetToArgs) {
  SymmetricTaskScheduler_Scene scene;
  scene.plan->workStorageType = ncclDevWorkStorageTypeFifo;  // sentinel: Args==0 is also the zero-init default
  ncclTaskColl task = SymmetricTaskScheduler_MakeTask();
  ncclIntruQueueEnqueue(&scene.symTaskQueue, &task);
  EXPECT_EQ(ncclSymmetricTaskScheduler(scene.comm.get(), &scene.symTaskQueue, scene.plan.get()), ncclSuccess);
  EXPECT_EQ(scene.plan->workStorageType, ncclDevWorkStorageTypeArgs);
}

