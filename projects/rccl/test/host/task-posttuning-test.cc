/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests; the UUT is #include'd. Hipify prepends one line, so hipified N is source N-1.

#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <vector>

#include "TaskPrepScene.h"
#include "fakes/bootstrap_stubs.h"

#include TASK_POSTTUNING_CC_PATH

namespace {

constexpr int kTrafficRanks = 4;
constexpr int kWideTrafficRanks = 7;
constexpr int kAllReduceTrafficPerByte = 2;
constexpr int kSingleTrafficPerByte = 1;
constexpr int kSingleStep = 1;
constexpr unsigned char kPoison = 0xA5;
constexpr int kQueuedArgs = 3;
constexpr int kFailingArgsIndex = 2;
constexpr uintptr_t kArgsSendBase = 0x71000;
constexpr uintptr_t kArgsRecvBase = 0x82000;
constexpr float kRunningTotalUs = 40.5f;
constexpr float kFirstTunedTimeUs = 3.25f;
constexpr float kSecondTunedTimeUs = 8.75f;
constexpr float kInvalidEntryTimeUs = 500.0f;
constexpr float kAllGatherVTimeUs = 250.0f;
constexpr float kNoTimeUs = 0.0f;
constexpr float kIgnoredEstimateUs = NCCL_TUNING_IGNORE;
constexpr int kInvalidTuning = 0;
constexpr int kNeverTunedEntries = 3;
constexpr int kCommMinCTAs = 2;
constexpr int kCommMaxCTAs = 16;
constexpr int kCommNvlsCTAs = 6;
constexpr int kCommCgaClusterSize = 3;
constexpr int kSourceEnv = 7;
constexpr int kSourcePerCall = 5;
constexpr int kSourceComm = 2;
constexpr int kResetMinCTAs = 1;
constexpr int kMaxCtasEnv = 9;
constexpr int kMaxCtasPerCallBelowComm = 5;
constexpr int kMaxCtasPerCallAboveComm = 20;
constexpr int kRejectedMaxCtas = 0;
constexpr int kRawRoot = 3;
constexpr uint64_t kRawScalarArg = 0x5A5A5A5A5A5A5A5Aull;
constexpr int kProfilerEventMask = 0x2A;
constexpr uint32_t kStampedDevFuncId = 37;
constexpr uint32_t kUnresolvedDevFuncId = 0;
constexpr int kTunedMaxChannels = 9;
constexpr int kTunedWarps = 6;
constexpr uint64_t kRingSimpleAlgBit = 1ull << 5;
constexpr uint64_t kAutomaticAlgMask = 0;
constexpr int kLlTrafficMultiplier = 4;
constexpr int kSingleNode = 1;
constexpr int kMultiNode = 2;
constexpr int kUnsetChannels = 0;
constexpr char kUnknownAlgSelection[] = "NOT_AN_ALGORITHM";
constexpr char kRingSimpleAlgSelection[] = "RING_SIMPLE";
constexpr char kAllReduceOnlyAlgSelection[] = "TREE_SIMPLE";

void* TaskPostTuning_Addr(uintptr_t address) {
  return reinterpret_cast<void*>(address);
}

struct ncclTaskColl TaskPostTuning_PoisonedColl(ncclFunc_t func) {
  struct ncclTaskColl task;
  std::memset(&task, kPoison, sizeof(task));
  task.func = func;
  return task;
}

void TaskPostTuning_ConstructQueues(struct ncclClassifiedTaskQueues* ctq) {
  for (TaskTuningInfoQueue* queue : {&ctq->symTaskQueue, &ctq->legacyTaskQueue, &ctq->allgathervTaskQueue,
                                     &ctq->p2pTaskQueue, &ctq->rmaTaskQueue, &ctq->ceTaskQueue}) {
    ncclIntruQueueConstruct(queue);
  }
}

struct ncclTaskTuningInfo* TaskPostTuning_Tuned(TaskPrepScene* scene, float timeUs) {
  struct ncclTaskTuningInfo* tInfo = scene->NewTuningInfo(scene->NewColl(ncclFuncAllReduce));
  tInfo->tuningOut.valid = kTunedValid;
  tInfo->tuningOut.timeUs = timeUs;
  return tInfo;
}

struct ncclTaskTuningInfo* TaskPostTuning_Untuned(TaskPrepScene* scene) {
  return scene->NewTuningInfo(scene->NewColl(ncclFuncAllReduce));
}

struct ncclArgsInfo* TaskPostTuning_NewArgsInfo(struct ncclComm* comm, const void* sendbuff, void* recvbuff) {
  struct ncclArgsInfo* argsInfo = static_cast<struct ncclArgsInfo*>(::calloc(1, sizeof(struct ncclArgsInfo)));
  argsInfo->info.coll = ncclFuncAllReduce;
  argsInfo->info.opName = "TaskPostTuningMicrotest";
  argsInfo->info.comm = comm;
  argsInfo->info.sendbuff = sendbuff;
  argsInfo->info.recvbuff = recvbuff;
  argsInfo->info.count = kCount;
  argsInfo->info.datatype = ncclFloat32;
  return argsInfo;
}

// Owns whatever production did not drain, because postTuneTasksDebug frees only the entries it dequeues.
class TaskPostTuning_ArgsInfoQueue {
 public:
  explicit TaskPostTuning_ArgsInfoQueue(TaskPrepScene* scene) : comm_(scene->comm()) {
    ncclIntruQueueConstruct(&comm_->argsInfoQueue);
  }

  ~TaskPostTuning_ArgsInfoQueue() {
    while (!ncclIntruQueueEmpty(&comm_->argsInfoQueue)) {
      ::free(ncclIntruQueueDequeue(&comm_->argsInfoQueue));
    }
  }

  TaskPostTuning_ArgsInfoQueue(const TaskPostTuning_ArgsInfoQueue&) = delete;
  TaskPostTuning_ArgsInfoQueue& operator=(const TaskPostTuning_ArgsInfoQueue&) = delete;

  void Enqueue(int index) {
    struct ncclArgsInfo* argsInfo = TaskPostTuning_NewArgsInfo(comm_, TaskPostTuning_Addr(kArgsSendBase + index),
                                                               TaskPostTuning_Addr(kArgsRecvBase + index));
    ncclIntruQueueEnqueue(&comm_->argsInfoQueue, argsInfo);
    buffers_.push_back(argsInfo->info.sendbuff);
    buffers_.push_back(argsInfo->info.recvbuff);
  }

  const std::vector<const void*>& buffers() const { return buffers_; }

 private:
  struct ncclComm* comm_;
  std::vector<const void*> buffers_;
};

using TaskPostTuning_FreeRawFn = ncclResult_t (*)(struct ncclComm*, struct ncclTaskTuningInfo*);

void TaskPostTuning_ExpectRawReturnedToPool(TaskPostTuning_FreeRawFn freeRaw) {
  TaskPrepScene scene;
  struct ncclRawTask* raw = scene.NewColl(ncclFuncAllReduce);
  struct ncclTaskTuningInfo* tInfo = scene.NewTuningInfo(raw);

  ASSERT_EQ(ncclSuccess, freeRaw(scene.comm(), tInfo));

  EXPECT_EQ(nullptr, tInfo->raw);
  EXPECT_EQ(raw, scene.NewRaw(ncclTaskKindColl));
}

void TaskPostTuning_ExpectClearedRawIsNoOp(TaskPostTuning_FreeRawFn freeRaw) {
  TaskPrepScene scene;
  struct ncclRawTask* untouched = scene.NewColl(ncclFuncAllReduce);
  struct ncclTaskTuningInfo* tInfo = scene.NewTuningInfo(untouched);
  tInfo->raw = nullptr;

  ASSERT_EQ(ncclSuccess, freeRaw(scene.comm(), tInfo));

  EXPECT_EQ(nullptr, tInfo->raw);
  EXPECT_NE(untouched, scene.NewRaw(ncclTaskKindColl));
}

using TaskPostTuning_TrafficFn = int (*)(ncclFunc_t, int);

void TaskPostTuning_ExpectAllReduceTraffic(TaskPostTuning_TrafficFn traffic) {
  EXPECT_EQ(kAllReduceTrafficPerByte, traffic(ncclFuncAllReduce, kTrafficRanks));
  EXPECT_EQ(kAllReduceTrafficPerByte, traffic(ncclFuncAllReduce, kWideTrafficRanks));
}

void TaskPostTuning_ExpectPerRankTraffic(TaskPostTuning_TrafficFn traffic) {
  for (ncclFunc_t func : {ncclFuncAllGather, ncclFuncReduceScatter}) {
    EXPECT_EQ(kTrafficRanks, traffic(func, kTrafficRanks)) << "func " << func;
    EXPECT_EQ(kWideTrafficRanks, traffic(func, kWideTrafficRanks)) << "func " << func;
  }
}

// ncclFuncAlltoAllvGda is excluded and pinned on its own because enqueue.cc:199 gives it nRanks, not 1.
void TaskPostTuning_ExpectSingleTraffic(TaskPostTuning_TrafficFn traffic) {
  for (int func = 0; func < ncclNumFuncs; func++) {
    if (func == ncclFuncAllReduce || func == ncclFuncAllGather || func == ncclFuncReduceScatter ||
        func == ncclFuncAlltoAllvGda) {
      continue;
    }
    EXPECT_EQ(kSingleTrafficPerByte, traffic(static_cast<ncclFunc_t>(func), kTrafficRanks)) << "func " << func;
  }
}

void TaskPostTuning_ExpectAlltoAllvGdaTraffic(TaskPostTuning_TrafficFn traffic) {
  EXPECT_EQ(kSingleTrafficPerByte, traffic(ncclFuncAlltoAllvGda, kTrafficRanks));
  EXPECT_EQ(kSingleTrafficPerByte, traffic(ncclFuncAlltoAllvGda, kWideTrafficRanks));
}

std::function<int64_t(const char*, int64_t)> TaskPostTuning_ParamOverride(const char* wanted, int64_t value) {
  return [wanted, value](const char* env, int64_t deft) -> int64_t {
    return std::strcmp(env, wanted) == 0 ? value : deft;
  };
}

// Mirrors ncclDevFuncId's general-collective key (device.h); AllReduce never takes the special-cased branches.
uint64_t TaskPostTuning_DevFuncKey(int coll, int devRedOp, int type, int algo, int proto) {
  return (static_cast<uint64_t>(coll & RCCL_FUNC_ID_MASK) << RCCL_COLL_SHIFT) |
         (static_cast<uint64_t>(algo & RCCL_FUNC_ID_MASK) << RCCL_ALGO_SHIFT) |
         (static_cast<uint64_t>(proto & RCCL_FUNC_ID_MASK) << RCCL_PROTO_SHIFT) |
         (static_cast<uint64_t>(devRedOp & RCCL_FUNC_ID_MASK) << RCCL_REDOP_SHIFT) |
         (static_cast<uint64_t>(type & RCCL_FUNC_ID_MASK) << RCCL_DTYPE_SHIFT);
}

void TaskPostTuning_StampDevFuncId(int algo, int proto) {
  ncclDevFuncNameToId[TaskPostTuning_DevFuncKey(ncclFuncAllReduce, ncclDevSum, ncclFloat32, algo, proto)] =
    kStampedDevFuncId;
}

void TaskPostTuning_SetTunerOutput(struct ncclTuningResult_t* out, int algo, int proto) {
  out->valid = kTunedValid;
  out->timeUs = kTunedTimeUs;
  out->algo = algo;
  out->proto = proto;
  out->maxChannels = kTunedMaxChannels;
  out->nWarps = kTunedWarps;
}

// One raw coll task, a poisoned destination, and a comm whose CTA config carries valid values throughout.
class TaskPostTuning_CollFill {
 public:
  explicit TaskPostTuning_CollFill(ncclFunc_t func = ncclFuncAllReduce, ncclDataType_t datatype = ncclFloat32)
    : tInfo_(scene_.NewTuningInfo(scene_.NewColl(func, datatype))) {
    const ncclCollConfig_t unsetConfig = NCCL_COLLCONFIG_INITIALIZER;
    std::memset(&task_, kPoison, sizeof(task_));
    tInfo_->raw->coll.collConfig = unsetConfig;
    scene_.comm()->config.CTAPolicy = NCCL_CTA_POLICY_DEFAULT;
    scene_.comm()->config.minCTAs = kCommMinCTAs;
    scene_.comm()->config.maxCTAs = kCommMaxCTAs;
    scene_.comm()->config.nvlsCTAs = kCommNvlsCTAs;
    scene_.comm()->config.cgaClusterSize = kCommCgaClusterSize;
  }

  struct ncclComm* comm() { return scene_.comm(); }
  struct ncclRawTaskColl* raw() { return &tInfo_->raw->coll; }
  ncclCollConfig_t* config() { return &tInfo_->raw->coll.collConfig; }
  struct ncclTuningResult_t* tuningOut() { return &tInfo_->tuningOut; }
  struct ncclTaskColl* task() { return &task_; }

  ncclResult_t Run() { return fillCollTaskFromRaw(comm(), tInfo_, &task_); }
  ncclResult_t RunApplyTuning() { return applyTuningToCollTask(comm(), tInfo_, &task_); }

 private:
  TaskPrepScene scene_;
  struct ncclTaskTuningInfo* tInfo_;
  struct ncclTaskColl task_;
};

struct TaskPostTuning_ConfigOption {
  const char* env;
  int ncclCollConfig_t::*perCall;
  int ncclConfig_t::*comm;
  int ncclTaskColl::*task;
  int lowerBound;
  int upperBound;
};

const TaskPostTuning_ConfigOption kTaskPostTuning_ConfigOptions[] = {
  {"MIN_CTAS", &ncclCollConfig_t::minCTAs, &ncclConfig_t::minCTAs, &ncclTaskColl::minCTAs, 1, MAXCHANNELS},
  {"NVLS_NCHANNELS", &ncclCollConfig_t::nvlsCTAs, &ncclConfig_t::nvlsCTAs, &ncclTaskColl::nvlsCTAs, 1, MAXCHANNELS},
  {"CGA_CLUSTER_SIZE", &ncclCollConfig_t::cgaClusterSize, &ncclConfig_t::cgaClusterSize,
   &ncclTaskColl::cgaClusterSize, 0, NCCL_MAX_CGA_CLUSTER_SIZE},
};

// The comm's CTA cap is maxed so that no probe value can trip the separate min > max reset.
int TaskPostTuning_ResolvedConfigOption(const TaskPostTuning_ConfigOption& option, int64_t envValue,
                                        int perCallValue, int commValue) {
  TaskPostTuning_CollFill fill;
  ScopedHook param(g_loadParam, TaskPostTuning_ParamOverride(option.env, envValue));
  fill.comm()->config.maxCTAs = MAXCHANNELS;
  (fill.comm()->config).*option.comm = commValue;
  fill.config()->*option.perCall = perCallValue;
  EXPECT_EQ(ncclSuccess, fill.Run()) << option.env;
  return fill.task()->*option.task;
}

class TaskPostTuningMicrotest : public TaskPrepFakesFixture {};

TEST_F(TaskPostTuningMicrotest, FuncTrafficPerByte_AllReduce_CountsEveryByteTwiceWhateverTheRankCount) {
  TaskPostTuning_ExpectAllReduceTraffic(ncclFuncTrafficPerByte);
}

TEST_F(TaskPostTuningMicrotest, FuncTrafficPerByte_AllGatherAndReduceScatter_CountEveryByteOncePerRank) {
  TaskPostTuning_ExpectPerRankTraffic(ncclFuncTrafficPerByte);
}

TEST_F(TaskPostTuningMicrotest, FuncTrafficPerByte_EveryOtherFunc_CountsEveryByteOnce) {
  TaskPostTuning_ExpectSingleTraffic(ncclFuncTrafficPerByte);
}

TEST_F(TaskPostTuningMicrotest, FuncTrafficPerByte_AlltoAllvGda_CountsEveryByteOnceUnlikeTheEnqueueCopy) {
  TaskPostTuning_ExpectAlltoAllvGdaTraffic(ncclFuncTrafficPerByte);
}

TEST_F(TaskPostTuningMicrotest, PostTuningFuncTrafficPerByte_AllReduce_CountsEveryByteTwiceWhateverTheRankCount) {
  TaskPostTuning_ExpectAllReduceTraffic(postTuningFuncTrafficPerByte);
}

TEST_F(TaskPostTuningMicrotest, PostTuningFuncTrafficPerByte_AllGatherAndReduceScatter_CountEveryByteOncePerRank) {
  TaskPostTuning_ExpectPerRankTraffic(postTuningFuncTrafficPerByte);
}

TEST_F(TaskPostTuningMicrotest, PostTuningFuncTrafficPerByte_EveryOtherFunc_CountsEveryByteOnce) {
  TaskPostTuning_ExpectSingleTraffic(postTuningFuncTrafficPerByte);
}

TEST_F(TaskPostTuningMicrotest, PostTuningFuncTrafficPerByte_AlltoAllvGda_CountsEveryByteOnceUnlikeTheEnqueueCopy) {
  TaskPostTuning_ExpectAlltoAllvGdaTraffic(postTuningFuncTrafficPerByte);
}

// ALLREDUCE_, ALLGATHER_ and REDUCESCATTER_CHUNKSTEPS are all equal, so these arms are not separable by value.
TEST_F(TaskPostTuningMicrotest, SetChunkSteps_PipelinedColls_TakeTheirOwnChunkAndSliceConstants) {
  struct ncclTaskColl allReduce = TaskPostTuning_PoisonedColl(ncclFuncAllReduce);
  struct ncclTaskColl allGather = TaskPostTuning_PoisonedColl(ncclFuncAllGather);
  struct ncclTaskColl reduceScatter = TaskPostTuning_PoisonedColl(ncclFuncReduceScatter);

  postTuningSetChunkSteps(&allReduce);
  postTuningSetChunkSteps(&allGather);
  postTuningSetChunkSteps(&reduceScatter);

  EXPECT_EQ(ALLREDUCE_CHUNKSTEPS, allReduce.chunkSteps);
  EXPECT_EQ(ALLREDUCE_SLICESTEPS, allReduce.sliceSteps);
  EXPECT_EQ(ALLGATHER_CHUNKSTEPS, allGather.chunkSteps);
  EXPECT_EQ(ALLGATHER_SLICESTEPS, allGather.sliceSteps);
  EXPECT_EQ(REDUCESCATTER_CHUNKSTEPS, reduceScatter.chunkSteps);
  EXPECT_EQ(REDUCESCATTER_SLICESTEPS, reduceScatter.sliceSteps);
  EXPECT_EQ(ncclFuncAllReduce, allReduce.func);
}

TEST_F(TaskPostTuningMicrotest, SetChunkSteps_BroadcastAndUnpipelinedFuncs_TakeASingleChunkAndSlice) {
  struct ncclTaskColl broadcast = TaskPostTuning_PoisonedColl(ncclFuncBroadcast);

  postTuningSetChunkSteps(&broadcast);

  EXPECT_EQ(BROADCAST_CHUNKSTEPS, broadcast.chunkSteps);
  EXPECT_EQ(BROADCAST_SLICESTEPS, broadcast.sliceSteps);
  for (ncclFunc_t func : {ncclFuncReduce, ncclFuncSend, ncclFuncAlltoAll, ncclFuncAllGatherV, ncclFuncPutSignal}) {
    struct ncclTaskColl task = TaskPostTuning_PoisonedColl(func);
    postTuningSetChunkSteps(&task);
    EXPECT_EQ(kSingleStep, task.chunkSteps) << "func " << func;
    EXPECT_EQ(kSingleStep, task.sliceSteps) << "func " << func;
  }
}

TEST_F(TaskPostTuningMicrotest, SymFreeTuningInfoRaw_RawStillHeld_ReturnsItToThePoolAndClearsThePointer) {
  TaskPostTuning_ExpectRawReturnedToPool(postTuneSymFreeTuningInfoRaw);
}

TEST_F(TaskPostTuningMicrotest, SymFreeTuningInfoRaw_RawAlreadyCleared_ReturnsNothingToThePool) {
  TaskPostTuning_ExpectClearedRawIsNoOp(postTuneSymFreeTuningInfoRaw);
}

TEST_F(TaskPostTuningMicrotest, RmaFreeTuningInfoRaw_RawStillHeld_ReturnsItToThePoolAndClearsThePointer) {
  TaskPostTuning_ExpectRawReturnedToPool(postTuneRmaFreeTuningInfoRaw);
}

TEST_F(TaskPostTuningMicrotest, RmaFreeTuningInfoRaw_RawAlreadyCleared_ReturnsNothingToThePool) {
  TaskPostTuning_ExpectClearedRawIsNoOp(postTuneRmaFreeTuningInfoRaw);
}

TEST_F(TaskPostTuningMicrotest, CeFreeTuningInfoRaw_RawStillHeld_ReturnsItToThePoolAndClearsThePointer) {
  TaskPostTuning_ExpectRawReturnedToPool(postTuneCeFreeTuningInfoRaw);
}

TEST_F(TaskPostTuningMicrotest, CeFreeTuningInfoRaw_RawAlreadyCleared_ReturnsNothingToThePool) {
  TaskPostTuning_ExpectClearedRawIsNoOp(postTuneCeFreeTuningInfoRaw);
}

TEST_F(TaskPostTuningMicrotest, SymTasksLazyInit_EmptyQueue_LeavesSymmetricKernelsUninitialized) {
  TaskPrepScene scene;
  TaskTuningInfoQueue queue;
  ncclIntruQueueConstruct(&queue);
  ScopedHook initOnce(g_symkInitOnce, [](struct ncclComm*) { return ncclSuccess; });

  EXPECT_EQ(ncclSuccess, postTuneSymTasksLazyInit(scene.comm(), &queue));

  EXPECT_EQ(0, initOnce.calls);
}

TEST_F(TaskPostTuningMicrotest, SymTasksLazyInit_QueuedTasks_InitializesSymmetricKernelsForThatComm) {
  TaskPrepScene scene;
  TaskTuningInfoQueue queue;
  ncclIntruQueueConstruct(&queue);
  ncclIntruQueueEnqueue(&queue, TaskPostTuning_Untuned(&scene));
  struct ncclComm* seen = nullptr;
  ScopedHook initOnce(g_symkInitOnce, [&seen](struct ncclComm* comm) {
    seen = comm;
    return ncclSuccess;
  });

  EXPECT_EQ(ncclSuccess, postTuneSymTasksLazyInit(scene.comm(), &queue));

  EXPECT_EQ(1, initOnce.calls);
  EXPECT_EQ(scene.comm(), seen);
}

TEST_F(TaskPostTuningMicrotest, SymTasksLazyInit_InitFails_PropagatesTheFailure) {
  TaskPrepScene scene;
  TaskTuningInfoQueue queue;
  ncclIntruQueueConstruct(&queue);
  ncclIntruQueueEnqueue(&queue, TaskPostTuning_Untuned(&scene));
  ScopedHook initOnce(g_symkInitOnce, [](struct ncclComm*) { return ncclSystemError; });

  EXPECT_EQ(ncclSystemError, postTuneSymTasksLazyInit(scene.comm(), &queue));

  EXPECT_EQ(1, initOnce.calls);
}

// Copy-engine tasks are dead even with the rearch gate on: the tuning mask omits NCCL_TUNING_MASK_CE.
TEST_F(TaskPostTuningMicrotest, CeTasksLazyInit_EmptyQueue_LeavesTheCopyEngineUninitialized) {
  TaskPrepScene scene;
  TaskTuningInfoQueue queue;
  ncclIntruQueueConstruct(&queue);
  ScopedHook ceInit(g_ncclCeInit, [](struct ncclComm*) { return ncclSuccess; });

  EXPECT_EQ(ncclSuccess, postTuneCeTasksLazyInit(scene.comm(), &queue));

  EXPECT_EQ(0, ceInit.calls);
}

TEST_F(TaskPostTuningMicrotest, CeTasksLazyInit_QueuedTasksAndUninitializedEngine_InitializesItForThatComm) {
  TaskPrepScene scene;
  TaskTuningInfoQueue queue;
  ncclIntruQueueConstruct(&queue);
  ncclIntruQueueEnqueue(&queue, TaskPostTuning_Untuned(&scene));
  scene.comm()->ceColl.initialized = false;
  struct ncclComm* seen = nullptr;
  ScopedHook ceInit(g_ncclCeInit, [&seen](struct ncclComm* comm) {
    seen = comm;
    return ncclSuccess;
  });

  EXPECT_EQ(ncclSuccess, postTuneCeTasksLazyInit(scene.comm(), &queue));

  EXPECT_EQ(1, ceInit.calls);
  EXPECT_EQ(scene.comm(), seen);
}

TEST_F(TaskPostTuningMicrotest, CeTasksLazyInit_QueuedTasksAndInitializedEngine_DoesNotInitializeAgain) {
  TaskPrepScene scene;
  TaskTuningInfoQueue queue;
  ncclIntruQueueConstruct(&queue);
  ncclIntruQueueEnqueue(&queue, TaskPostTuning_Untuned(&scene));
  scene.comm()->ceColl.initialized = true;
  ScopedHook ceInit(g_ncclCeInit, [](struct ncclComm*) { return ncclSuccess; });

  EXPECT_EQ(ncclSuccess, postTuneCeTasksLazyInit(scene.comm(), &queue));

  EXPECT_EQ(0, ceInit.calls);
}

TEST_F(TaskPostTuningMicrotest, CeTasksLazyInit_InitFails_PropagatesTheFailure) {
  TaskPrepScene scene;
  TaskTuningInfoQueue queue;
  ncclIntruQueueConstruct(&queue);
  ncclIntruQueueEnqueue(&queue, TaskPostTuning_Untuned(&scene));
  scene.comm()->ceColl.initialized = false;
  ScopedHook ceInit(g_ncclCeInit, [](struct ncclComm*) { return ncclSystemError; });

  EXPECT_EQ(ncclSystemError, postTuneCeTasksLazyInit(scene.comm(), &queue));

  EXPECT_EQ(1, ceInit.calls);
}

TEST_F(TaskPostTuningMicrotest, TasksDebug_QueuedArgs_CheckEachInQueueOrderAndDrainTheQueue) {
  TaskPrepScene scene;
  TaskPostTuning_ArgsInfoQueue queued(&scene);
  std::vector<const void*> checked;
  ScopedHook findWindow(g_devrFindWindow,
                        [&checked](struct ncclComm*, void const* ptr, struct ncclDevrWindow** out) {
                          checked.push_back(ptr);
                          *out = nullptr;
                          return ncclSuccess;
                        });
  ScopedHook allGather(g_bootstrapAllGather, [](void*, void*, int) { return ncclSuccess; });
  for (int index = 0; index < kQueuedArgs; index++) {
    queued.Enqueue(index);
  }

  EXPECT_EQ(ncclSuccess, postTuneTasksDebug(scene.comm()));

  EXPECT_EQ(queued.buffers(), checked);
  EXPECT_EQ(kQueuedArgs, allGather.calls);
  EXPECT_TRUE(ncclIntruQueueEmpty(&scene.comm()->argsInfoQueue));
}

TEST_F(TaskPostTuningMicrotest, TasksDebug_CheckFailsMidQueue_PropagatesAndLeavesTheRestQueued) {
  TaskPrepScene scene;
  TaskPostTuning_ArgsInfoQueue queued(&scene);
  int checks = 0;
  ScopedHook findWindow(g_devrFindWindow, [](struct ncclComm*, void const*, struct ncclDevrWindow** out) {
    *out = nullptr;
    return ncclSuccess;
  });
  ScopedHook allGather(g_bootstrapAllGather, [&checks](void*, void*, int) -> ncclResult_t {
    return ++checks == kFailingArgsIndex ? ncclInvalidUsage : ncclSuccess;
  });
  for (int index = 0; index < kQueuedArgs; index++) {
    queued.Enqueue(index);
  }
  struct ncclArgsInfo* survivor = ncclIntruQueueHead(&scene.comm()->argsInfoQueue)->next->next;

  EXPECT_EQ(ncclInvalidUsage, postTuneTasksDebug(scene.comm()));

  EXPECT_EQ(kFailingArgsIndex, checks);
  EXPECT_EQ(survivor, ncclIntruQueueHead(&scene.comm()->argsInfoQueue));
}

TEST_F(TaskPostTuningMicrotest, TasksDebug_EmptyQueue_ChecksNothing) {
  TaskPrepScene scene;
  TaskPostTuning_ArgsInfoQueue queued(&scene);
  ScopedHook allGather(g_bootstrapAllGather, [](void*, void*, int) { return ncclSuccess; });

  EXPECT_EQ(ncclSuccess, postTuneTasksDebug(scene.comm()));

  EXPECT_EQ(0, allGather.calls);
  EXPECT_TRUE(ncclIntruQueueEmpty(&scene.comm()->argsInfoQueue));
}

TEST_F(TaskPostTuningMicrotest, AccumulateTime_TunedEntries_AddEveryEstimateToTheRunningTotal) {
  TaskPrepScene scene;
  TaskTuningInfoQueue queue;
  ncclIntruQueueConstruct(&queue);
  ncclIntruQueueEnqueue(&queue, TaskPostTuning_Tuned(&scene, kFirstTunedTimeUs));
  ncclIntruQueueEnqueue(&queue, TaskPostTuning_Tuned(&scene, kSecondTunedTimeUs));

  float totalTimeUs = kRunningTotalUs;
  postTuningAccumulateTime(&queue, &totalTimeUs);

  EXPECT_FLOAT_EQ(kRunningTotalUs + kFirstTunedTimeUs + kSecondTunedTimeUs, totalTimeUs);
}

TEST_F(TaskPostTuningMicrotest, AccumulateTime_EmptyQueue_LeavesTheRunningTotalUnchanged) {
  TaskTuningInfoQueue queue;
  ncclIntruQueueConstruct(&queue);

  float totalTimeUs = kRunningTotalUs;
  postTuningAccumulateTime(&queue, &totalTimeUs);

  EXPECT_FLOAT_EQ(kRunningTotalUs, totalTimeUs);
}

TEST_F(TaskPostTuningMicrotest, AccumulateTime_EntriesMarkedInvalid_ContributeNothing) {
  TaskPrepScene scene;
  TaskTuningInfoQueue queue;
  ncclIntruQueueConstruct(&queue);
  struct ncclTaskTuningInfo* invalid = TaskPostTuning_Tuned(&scene, kInvalidEntryTimeUs);
  invalid->tuningOut.valid = kInvalidTuning;
  ncclIntruQueueEnqueue(&queue, invalid);
  ncclIntruQueueEnqueue(&queue, TaskPostTuning_Tuned(&scene, kFirstTunedTimeUs));

  float totalTimeUs = kRunningTotalUs;
  postTuningAccumulateTime(&queue, &totalTimeUs);

  EXPECT_FLOAT_EQ(kRunningTotalUs + kFirstTunedTimeUs, totalTimeUs);
}

// task_posttuning.cc:994 reads the untuned valid = -1 sentinel as true; fixing that guard turns this red.
TEST_F(TaskPostTuningMicrotest, AccumulateTime_NeverTunedEntries_CurrentlySubtractTheIgnoreSentinelPerEntry) {
  TaskPrepScene scene;
  TaskTuningInfoQueue queue;
  ncclIntruQueueConstruct(&queue);
  for (int i = 0; i < kNeverTunedEntries; i++) {
    ncclIntruQueueEnqueue(&queue, TaskPostTuning_Untuned(&scene));
  }

  float totalTimeUs = kRunningTotalUs;
  postTuningAccumulateTime(&queue, &totalTimeUs);

  EXPECT_FLOAT_EQ(kRunningTotalUs + kNeverTunedEntries * kIgnoredEstimateUs, totalTimeUs);
}

// task_posttuning.cc:994 reads the untuned valid = -1 sentinel as true; fixing that guard turns this red.
TEST_F(TaskPostTuningMicrotest, AccumulateTime_MixedTunedAndNeverTunedEntries_CurrentlyOffsetTheTunedEstimate) {
  TaskPrepScene scene;
  TaskTuningInfoQueue queue;
  ncclIntruQueueConstruct(&queue);
  ncclIntruQueueEnqueue(&queue, TaskPostTuning_Tuned(&scene, kFirstTunedTimeUs));
  ncclIntruQueueEnqueue(&queue, TaskPostTuning_Untuned(&scene));

  float totalTimeUs = kRunningTotalUs;
  postTuningAccumulateTime(&queue, &totalTimeUs);

  EXPECT_FLOAT_EQ(kRunningTotalUs + kFirstTunedTimeUs + kIgnoredEstimateUs, totalTimeUs);
}

TEST_F(TaskPostTuningMicrotest, Simulation_TunedEntries_SumEveryAccumulatedQueueIntoTheEstimate) {
  TaskPrepScene scene;
  struct ncclClassifiedTaskQueues ctq;
  TaskPostTuning_ConstructQueues(&ctq);
  TaskTuningInfoQueue* accumulated[] = {&ctq.symTaskQueue, &ctq.legacyTaskQueue, &ctq.p2pTaskQueue,
                                        &ctq.rmaTaskQueue, &ctq.ceTaskQueue};
  float expected = 0.0f;
  float timeUs = kFirstTunedTimeUs;
  for (TaskTuningInfoQueue* queue : accumulated) {
    ncclIntruQueueEnqueue(queue, TaskPostTuning_Tuned(&scene, timeUs));
    expected += timeUs;
    timeUs *= 2.0f;
  }

  ncclSimInfo_t sim = PoisonedSimInfo();
  EXPECT_EQ(ncclSuccess, postTuningSimulation(scene.comm(), &ctq, &sim));

  EXPECT_FLOAT_EQ(expected, sim.estimatedTime);
}

TEST_F(TaskPostTuningMicrotest, Simulation_AllQueuesEmpty_ReportsNoEstimatedTime) {
  TaskPrepScene scene;
  struct ncclClassifiedTaskQueues ctq;
  TaskPostTuning_ConstructQueues(&ctq);

  ncclSimInfo_t sim = PoisonedSimInfo();
  EXPECT_EQ(ncclSuccess, postTuningSimulation(scene.comm(), &ctq, &sim));

  EXPECT_FLOAT_EQ(kNoTimeUs, sim.estimatedTime);
}

TEST_F(TaskPostTuningMicrotest, Simulation_TunedAllGatherVEntry_DoesNotReachTheEstimate) {
  TaskPrepScene scene;
  struct ncclClassifiedTaskQueues ctq;
  TaskPostTuning_ConstructQueues(&ctq);
  ncclIntruQueueEnqueue(&ctq.allgathervTaskQueue, TaskPostTuning_Tuned(&scene, kAllGatherVTimeUs));
  ncclIntruQueueEnqueue(&ctq.legacyTaskQueue, TaskPostTuning_Tuned(&scene, kFirstTunedTimeUs));

  ncclSimInfo_t sim = PoisonedSimInfo();
  EXPECT_EQ(ncclSuccess, postTuningSimulation(scene.comm(), &ctq, &sim));

  EXPECT_FLOAT_EQ(kFirstTunedTimeUs, sim.estimatedTime);
}

// task_posttuning.cc:994 reads the untuned valid = -1 sentinel as true; fixing that guard turns this red.
TEST_F(TaskPostTuningMicrotest, Simulation_NeverTunedP2pAndRmaEntries_CurrentlyOffsetTheTunedTotal) {
  TaskPrepScene scene;
  struct ncclClassifiedTaskQueues ctq;
  TaskPostTuning_ConstructQueues(&ctq);
  ncclIntruQueueEnqueue(&ctq.p2pTaskQueue, TaskPostTuning_Untuned(&scene));
  ncclIntruQueueEnqueue(&ctq.p2pTaskQueue, TaskPostTuning_Untuned(&scene));
  ncclIntruQueueEnqueue(&ctq.rmaTaskQueue, TaskPostTuning_Untuned(&scene));
  ncclIntruQueueEnqueue(&ctq.legacyTaskQueue, TaskPostTuning_Tuned(&scene, kFirstTunedTimeUs));

  ncclSimInfo_t sim = PoisonedSimInfo();
  EXPECT_EQ(ncclSuccess, postTuningSimulation(scene.comm(), &ctq, &sim));

  EXPECT_FLOAT_EQ(kFirstTunedTimeUs + kNeverTunedEntries * kIgnoredEstimateUs, sim.estimatedTime);
}

TEST_F(TaskPostTuningMicrotest, FillCollTaskFromRaw_AllReduce_CopiesTheRawFieldsAndDerivesTheDependentOnes) {
  TaskPostTuning_CollFill fill;
  ncclProfilerEventMask = kProfilerEventMask;
  fill.raw()->root = kRawRoot;
  fill.raw()->opHost = ncclProd;
  fill.raw()->opDev.op = ncclDevProd;
  fill.raw()->opDev.scalarArg = kRawScalarArg;

  ASSERT_EQ(ncclSuccess, fill.Run());

  const struct ncclTaskColl* task = fill.task();
  EXPECT_EQ(ncclFuncAllReduce, task->func);
  EXPECT_EQ(fill.raw()->sendbuff, task->sendbuff);
  EXPECT_EQ(fill.raw()->recvbuff, task->recvbuff);
  EXPECT_EQ(kCount, task->count);
  EXPECT_EQ(kRawRoot, task->root);
  EXPECT_EQ(ncclFloat32, task->datatype);
  EXPECT_EQ(ncclProd, task->opHost);
  EXPECT_EQ(ncclDevProd, task->opDev.op);
  EXPECT_EQ(kRawScalarArg, task->opDev.scalarArg);
  EXPECT_EQ(kCount * sizeof(float) * kAllReduceTrafficPerByte, task->trafficBytes);
  EXPECT_EQ(ALLREDUCE_CHUNKSTEPS, task->chunkSteps);
  EXPECT_EQ(ALLREDUCE_SLICESTEPS, task->sliceSteps);
  EXPECT_EQ(kProfilerEventMask, task->eActivationMask);
  EXPECT_EQ(nullptr, task->groupApiEventHandle);
  EXPECT_EQ(nullptr, task->collApiEventHandle);
  EXPECT_EQ(1, task->forceAlgSelection);
  EXPECT_EQ(kAutomaticAlgMask, task->algMask);
  EXPECT_FALSE(task->aggIsolate);
}

TEST_F(TaskPostTuningMicrotest, FillCollTaskFromRaw_PoisonedDestination_ClearsTheFieldsTheRawDoesNotSupply) {
  TaskPostTuning_CollFill fill;

  ASSERT_EQ(ncclSuccess, fill.Run());

  const struct ncclTaskColl* task = fill.task();
  EXPECT_EQ(nullptr, task->next);
  EXPECT_EQ(nullptr, task->acc);
  EXPECT_EQ(nullptr, task->sizes);
  EXPECT_EQ(0u, task->opCount);
  EXPECT_EQ(0, task->regBufType);
  EXPECT_EQ(0u, task->nChannels);
  EXPECT_EQ(nullptr, task->sendWin);
  EXPECT_EQ(nullptr, task->recvWin);
}

TEST_F(TaskPostTuningMicrotest, FillCollTaskFromRaw_AllGather_RescalesTheCountToBytesAndSwitchesToInt8) {
  TaskPostTuning_CollFill fill(ncclFuncAllGather);

  ASSERT_EQ(ncclSuccess, fill.Run());

  EXPECT_EQ(kCount * sizeof(float), fill.task()->count);
  EXPECT_EQ(ncclInt8, fill.task()->datatype);
  EXPECT_EQ(kCount * sizeof(float) * kRanks, fill.task()->trafficBytes);
  EXPECT_EQ(kCount, fill.raw()->count);
  EXPECT_EQ(ncclFloat32, fill.raw()->datatype);
}

TEST_F(TaskPostTuningMicrotest, FillCollTaskFromRaw_Broadcast_RescalesTheCountToBytesAndCountsEachByteOnce) {
  TaskPostTuning_CollFill fill(ncclFuncBroadcast);

  ASSERT_EQ(ncclSuccess, fill.Run());

  EXPECT_EQ(kCount * sizeof(float), fill.task()->count);
  EXPECT_EQ(ncclInt8, fill.task()->datatype);
  EXPECT_EQ(kCount * sizeof(float) * kSingleTrafficPerByte, fill.task()->trafficBytes);
  EXPECT_EQ(BROADCAST_CHUNKSTEPS, fill.task()->chunkSteps);
}

TEST_F(TaskPostTuningMicrotest, FillCollTaskFromRaw_ReduceScatter_KeepsTheElementCountAndScalesTrafficByRank) {
  TaskPostTuning_CollFill fill(ncclFuncReduceScatter);

  ASSERT_EQ(ncclSuccess, fill.Run());

  EXPECT_EQ(kCount, fill.task()->count);
  EXPECT_EQ(ncclFloat32, fill.task()->datatype);
  EXPECT_EQ(kCount * sizeof(float) * kRanks, fill.task()->trafficBytes);
}

TEST_F(TaskPostTuningMicrotest, FillCollTaskFromRaw_PerCallCtaPolicy_OverridesTheCommPolicyAndIsolatesTheTask) {
  TaskPostTuning_CollFill fill;
  fill.config()->CTAPolicy = NCCL_CTA_POLICY_ZERO;

  ASSERT_EQ(ncclSuccess, fill.Run());

  EXPECT_EQ(NCCL_CTA_POLICY_ZERO, fill.task()->CTAPolicy);
  EXPECT_TRUE(fill.task()->aggIsolate);
}

TEST_F(TaskPostTuningMicrotest, FillCollTaskFromRaw_PerCallCtaPolicyAsksForZeroAndEfficiency_KeepsOnlyZero) {
  TaskPostTuning_CollFill fill;
  fill.config()->CTAPolicy = NCCL_CTA_POLICY_ZERO | NCCL_CTA_POLICY_EFFICIENCY;

  ASSERT_EQ(ncclSuccess, fill.Run());

  EXPECT_EQ(NCCL_CTA_POLICY_ZERO, fill.task()->CTAPolicy);
}

TEST_F(TaskPostTuningMicrotest, FillCollTaskFromRaw_EnvCtaPolicyOverride_IgnoresThePerCallPolicyAndDoesNotIsolate) {
  TaskPostTuning_CollFill fill;
  g_envCtaPolicy = NCCL_CTA_POLICY_EFFICIENCY;
  fill.comm()->config.CTAPolicy = NCCL_CTA_POLICY_EFFICIENCY;
  fill.config()->CTAPolicy = NCCL_CTA_POLICY_ZERO;

  ASSERT_EQ(ncclSuccess, fill.Run());

  EXPECT_EQ(NCCL_CTA_POLICY_EFFICIENCY, fill.task()->CTAPolicy);
  EXPECT_FALSE(fill.task()->aggIsolate);
}

TEST_F(TaskPostTuningMicrotest, FillCollTaskFromRaw_UnsetPerCallCtaPolicy_InheritsTheCommPolicyAndDoesNotIsolate) {
  TaskPostTuning_CollFill fill;
  fill.comm()->config.CTAPolicy = NCCL_CTA_POLICY_EFFICIENCY;

  ASSERT_EQ(ncclSuccess, fill.Run());

  EXPECT_EQ(NCCL_CTA_POLICY_EFFICIENCY, fill.task()->CTAPolicy);
  EXPECT_FALSE(fill.task()->aggIsolate);
}

TEST_F(TaskPostTuningMicrotest, FillCollTaskFromRaw_PerCallConfigBoundsTheCtaCount_IsolatesTheTaskFromAggregation) {
  TaskPostTuning_CollFill fill;
  fill.config()->minCTAs = kSourcePerCall;

  ASSERT_EQ(ncclSuccess, fill.Run());

  EXPECT_TRUE(fill.task()->aggIsolate);
  EXPECT_EQ(NCCL_CTA_POLICY_DEFAULT, fill.task()->CTAPolicy);
}

TEST_F(TaskPostTuningMicrotest, FillCollTaskFromRaw_NoPerCallConfigAtAll_LeavesTheTaskAggregatable) {
  TaskPostTuning_CollFill fill;
  fill.config()->size = 0;
  fill.config()->minCTAs = kSourcePerCall;

  ASSERT_EQ(ncclSuccess, fill.Run());

  EXPECT_FALSE(fill.task()->aggIsolate);
  EXPECT_EQ(kSourcePerCall, fill.task()->minCTAs);
}

TEST_F(TaskPostTuningMicrotest, FillCollTaskFromRaw_ConfigOptionEnvInBounds_WinsOverThePerCallAndCommValues) {
  for (const TaskPostTuning_ConfigOption& option : kTaskPostTuning_ConfigOptions) {
    EXPECT_EQ(kSourceEnv, TaskPostTuning_ResolvedConfigOption(option, kSourceEnv, kSourcePerCall, kSourceComm))
      << option.env;
  }
}

TEST_F(TaskPostTuningMicrotest, FillCollTaskFromRaw_ConfigOptionEnvAtEitherBound_IsAccepted) {
  for (const TaskPostTuning_ConfigOption& option : kTaskPostTuning_ConfigOptions) {
    EXPECT_EQ(option.lowerBound,
              TaskPostTuning_ResolvedConfigOption(option, option.lowerBound, kSourcePerCall, kSourceComm))
      << option.env;
    EXPECT_EQ(option.upperBound,
              TaskPostTuning_ResolvedConfigOption(option, option.upperBound, kSourcePerCall, kSourceComm))
      << option.env;
  }
}

TEST_F(TaskPostTuningMicrotest, FillCollTaskFromRaw_ConfigOptionEnvOutOfBounds_FallsBackToThePerCallValue) {
  for (const TaskPostTuning_ConfigOption& option : kTaskPostTuning_ConfigOptions) {
    EXPECT_EQ(kSourcePerCall,
              TaskPostTuning_ResolvedConfigOption(option, option.lowerBound - 1, kSourcePerCall, kSourceComm))
      << option.env;
    EXPECT_EQ(kSourcePerCall,
              TaskPostTuning_ResolvedConfigOption(option, option.upperBound + 1, kSourcePerCall, kSourceComm))
      << option.env;
  }
}

TEST_F(TaskPostTuningMicrotest, FillCollTaskFromRaw_ConfigOptionPerCallInBounds_WinsOverTheCommValue) {
  for (const TaskPostTuning_ConfigOption& option : kTaskPostTuning_ConfigOptions) {
    EXPECT_EQ(kSourcePerCall, TaskPostTuning_ResolvedConfigOption(option, NCCL_CONFIG_UNDEF_INT, kSourcePerCall,
                                                                  kSourceComm))
      << option.env;
  }
}

TEST_F(TaskPostTuningMicrotest, FillCollTaskFromRaw_ConfigOptionPerCallOutOfBounds_FallsBackToTheCommValue) {
  for (const TaskPostTuning_ConfigOption& option : kTaskPostTuning_ConfigOptions) {
    EXPECT_EQ(kSourceComm, TaskPostTuning_ResolvedConfigOption(option, NCCL_CONFIG_UNDEF_INT, option.lowerBound - 1,
                                                               kSourceComm))
      << option.env;
    EXPECT_EQ(kSourceComm, TaskPostTuning_ResolvedConfigOption(option, NCCL_CONFIG_UNDEF_INT, option.upperBound + 1,
                                                               kSourceComm))
      << option.env;
  }
}

TEST_F(TaskPostTuningMicrotest, FillCollTaskFromRaw_ConfigOptionUnsetEverywhere_TakesTheCommValue) {
  for (const TaskPostTuning_ConfigOption& option : kTaskPostTuning_ConfigOptions) {
    EXPECT_EQ(kSourceComm, TaskPostTuning_ResolvedConfigOption(option, NCCL_CONFIG_UNDEF_INT, NCCL_CONFIG_UNDEF_INT,
                                                               kSourceComm))
      << option.env;
  }
}

TEST_F(TaskPostTuningMicrotest, FillCollTaskFromRaw_MaxCtasEnvInBounds_WinsOverThePerCallAndCommValues) {
  TaskPostTuning_CollFill fill;
  ScopedHook param(g_loadParam, TaskPostTuning_ParamOverride("MAX_CTAS", kMaxCtasEnv));
  fill.config()->maxCTAs = kMaxCtasPerCallBelowComm;

  ASSERT_EQ(ncclSuccess, fill.Run());

  EXPECT_EQ(kMaxCtasEnv, fill.task()->maxCTAs);
}

TEST_F(TaskPostTuningMicrotest, FillCollTaskFromRaw_MaxCtasEnvOutOfBounds_FallsBackToThePerCallValue) {
  TaskPostTuning_CollFill fill;
  ScopedHook param(g_loadParam, TaskPostTuning_ParamOverride("MAX_CTAS", MAXCHANNELS + 1));
  fill.config()->maxCTAs = kMaxCtasPerCallBelowComm;

  ASSERT_EQ(ncclSuccess, fill.Run());

  EXPECT_EQ(kMaxCtasPerCallBelowComm, fill.task()->maxCTAs);
}

TEST_F(TaskPostTuningMicrotest, FillCollTaskFromRaw_MaxCtasPerCallBelowTheCommCap_TakesThePerCallValue) {
  TaskPostTuning_CollFill fill;
  fill.config()->maxCTAs = kMaxCtasPerCallBelowComm;

  ASSERT_EQ(ncclSuccess, fill.Run());

  EXPECT_EQ(kMaxCtasPerCallBelowComm, fill.task()->maxCTAs);
}

TEST_F(TaskPostTuningMicrotest, FillCollTaskFromRaw_MaxCtasPerCallAboveTheCommCap_ClampsToTheCommValue) {
  TaskPostTuning_CollFill fill;
  fill.config()->maxCTAs = kMaxCtasPerCallAboveComm;

  ASSERT_EQ(ncclSuccess, fill.Run());

  EXPECT_EQ(kCommMaxCTAs, fill.task()->maxCTAs);
}

TEST_F(TaskPostTuningMicrotest, FillCollTaskFromRaw_MaxCtasClampedBelowItsLowerBound_FallsBackToTheCommValue) {
  TaskPostTuning_CollFill fill;
  fill.config()->maxCTAs = kRejectedMaxCtas;

  ASSERT_EQ(ncclSuccess, fill.Run());

  EXPECT_EQ(kCommMaxCTAs, fill.task()->maxCTAs);
}

TEST_F(TaskPostTuningMicrotest, FillCollTaskFromRaw_MaxCtasUnsetEverywhere_TakesTheCommValue) {
  TaskPostTuning_CollFill fill;

  ASSERT_EQ(ncclSuccess, fill.Run());

  EXPECT_EQ(kCommMaxCTAs, fill.task()->maxCTAs);
  EXPECT_EQ(kCommMinCTAs, fill.task()->minCTAs);
}

TEST_F(TaskPostTuningMicrotest, FillCollTaskFromRaw_MinCtasAboveMaxCtas_ResetsMinCtasToOne) {
  TaskPostTuning_CollFill fill;
  fill.comm()->config.minCTAs = kCommMaxCTAs + 1;

  ASSERT_EQ(ncclSuccess, fill.Run());

  EXPECT_EQ(kResetMinCTAs, fill.task()->minCTAs);
  EXPECT_EQ(kCommMaxCTAs, fill.task()->maxCTAs);
}

TEST_F(TaskPostTuningMicrotest, FillCollTaskFromRaw_MinCtasEqualToMaxCtas_LeavesMinCtasAlone) {
  TaskPostTuning_CollFill fill;
  fill.comm()->config.minCTAs = kCommMaxCTAs;

  ASSERT_EQ(ncclSuccess, fill.Run());

  EXPECT_EQ(kCommMaxCTAs, fill.task()->minCTAs);
  EXPECT_EQ(kCommMaxCTAs, fill.task()->maxCTAs);
}

TEST_F(TaskPostTuningMicrotest, FillCollTaskFromRaw_RecognisedAlgSelection_NarrowsTheTaskAlgMaskAndIsolatesTheTask) {
  TaskPostTuning_CollFill fill;
  fill.config()->algSelection = kRingSimpleAlgSelection;

  ASSERT_EQ(ncclSuccess, fill.Run());

  EXPECT_EQ(kRingSimpleAlgBit, fill.task()->algMask);
  EXPECT_TRUE(fill.task()->aggIsolate);
  EXPECT_EQ(1, fill.task()->forceAlgSelection);
}

TEST_F(TaskPostTuningMicrotest, FillCollTaskFromRaw_UnparsableAlgSelectionForcedOn_FailsWithoutNarrowingTheMask) {
  TaskPostTuning_CollFill fill;
  fill.config()->algSelection = kUnknownAlgSelection;

  EXPECT_EQ(ncclInvalidArgument, fill.Run());

  EXPECT_EQ(kAutomaticAlgMask, fill.task()->algMask);
  EXPECT_EQ(ncclFuncAllReduce, fill.task()->func);
}

TEST_F(TaskPostTuningMicrotest, FillCollTaskFromRaw_UnparsableAlgSelectionForcedOff_FallsBackToAutomaticSelection) {
  TaskPostTuning_CollFill fill;
  fill.config()->algSelection = kUnknownAlgSelection;
  fill.config()->forceAlgSelection = 0;

  ASSERT_EQ(ncclSuccess, fill.Run());

  EXPECT_EQ(kAutomaticAlgMask, fill.task()->algMask);
  EXPECT_EQ(0, fill.task()->forceAlgSelection);
}

TEST_F(TaskPostTuningMicrotest, FillCollTaskFromRaw_AlgSelectionValidForAnotherCollective_FailsForThisCollective) {
  TaskPostTuning_CollFill fill(ncclFuncAllGather);
  fill.config()->algSelection = kAllReduceOnlyAlgSelection;

  EXPECT_EQ(ncclInvalidArgument, fill.Run());

  EXPECT_EQ(kAutomaticAlgMask, fill.task()->algMask);
  EXPECT_EQ(ncclFuncAllGather, fill.task()->func);
}

TEST_F(TaskPostTuningMicrotest, ApplyTuningToCollTask_ValidTunerOutput_AppliesItAndResolvesTheDeviceFunction) {
  TaskPostTuning_CollFill fill;
  fill.raw()->opDev.op = ncclDevProd;
  TaskPostTuning_SetTunerOutput(fill.tuningOut(), NCCL_ALGO_RING, NCCL_PROTO_SIMPLE);
  ncclDevFuncNameToId[TaskPostTuning_DevFuncKey(ncclFuncAllReduce, ncclDevProd, ncclFloat32, NCCL_ALGO_RING,
                                                NCCL_PROTO_SIMPLE)] = kStampedDevFuncId;

  ASSERT_EQ(ncclSuccess, fill.RunApplyTuning());

  const struct ncclTaskColl* task = fill.task();
  EXPECT_EQ(NCCL_ALGO_RING, task->algorithm);
  EXPECT_EQ(NCCL_PROTO_SIMPLE, task->protocol);
  EXPECT_EQ(kTunedMaxChannels, task->nMaxChannels);
  EXPECT_EQ(kTunedWarps, task->nWarps);
  EXPECT_EQ(kStampedDevFuncId, task->devFuncId);
  EXPECT_EQ(0u, task->isNvls);
  EXPECT_EQ(0u, task->isCollnet);
  EXPECT_EQ(kCount * sizeof(float) * kAllReduceTrafficPerByte, task->trafficBytes);
}

TEST_F(TaskPostTuningMicrotest, ApplyTuningToCollTask_EveryAlgorithm_SetsTheNvlsAndCollnetFlagsThatAlgorithmNeeds) {
  const struct {
    int algo;
    int nNodes;
    bool isOneRPN;
    unsigned isNvls;
    unsigned isCollnet;
  } kCases[] = {
    {NCCL_ALGO_NVLS, kSingleNode, true, 1u, 0u},
    {NCCL_ALGO_NVLS, kMultiNode, true, 1u, 1u},
    {NCCL_ALGO_NVLS_TREE, kSingleNode, true, 1u, 0u},
    {NCCL_ALGO_NVLS_TREE, kMultiNode, true, 1u, 0u},
    {NCCL_ALGO_PAT, kSingleNode, true, 0u, 0u},
    {NCCL_ALGO_PAT, kMultiNode, false, 1u, 0u},
    {NCCL_ALGO_COLLNET_CHAIN, kMultiNode, true, 0u, 1u},
    {NCCL_ALGO_COLLNET_DIRECT, kMultiNode, true, 0u, 1u},
    {NCCL_ALGO_TREE, kMultiNode, true, 0u, 0u},
    {NCCL_ALGO_RING, kSingleNode, false, 0u, 0u},
  };

  for (const auto& testCase : kCases) {
    TaskPostTuning_CollFill fill;
    fill.comm()->nNodes = testCase.nNodes;
    fill.comm()->isOneRPN = testCase.isOneRPN;
    TaskPostTuning_SetTunerOutput(fill.tuningOut(), testCase.algo, NCCL_PROTO_SIMPLE);
    TaskPostTuning_StampDevFuncId(testCase.algo, NCCL_PROTO_SIMPLE);

    ASSERT_EQ(ncclSuccess, fill.RunApplyTuning()) << "algo " << testCase.algo;

    EXPECT_EQ(testCase.isNvls, fill.task()->isNvls)
      << "algo " << testCase.algo << " nNodes " << testCase.nNodes << " isOneRPN " << testCase.isOneRPN;
    EXPECT_EQ(testCase.isCollnet, fill.task()->isCollnet)
      << "algo " << testCase.algo << " nNodes " << testCase.nNodes << " isOneRPN " << testCase.isOneRPN;
    EXPECT_EQ(kStampedDevFuncId, fill.task()->devFuncId) << "algo " << testCase.algo;
  }
}

TEST_F(TaskPostTuningMicrotest, ApplyTuningToCollTask_LatencyProtocol_QuadruplesTheTrafficEstimate) {
  TaskPostTuning_CollFill fill;
  TaskPostTuning_SetTunerOutput(fill.tuningOut(), NCCL_ALGO_RING, NCCL_PROTO_LL);
  TaskPostTuning_StampDevFuncId(NCCL_ALGO_RING, NCCL_PROTO_LL);

  ASSERT_EQ(ncclSuccess, fill.RunApplyTuning());

  EXPECT_EQ(kCount * sizeof(float) * kAllReduceTrafficPerByte * kLlTrafficMultiplier, fill.task()->trafficBytes);
  EXPECT_EQ(kStampedDevFuncId, fill.task()->devFuncId);
}

TEST_F(TaskPostTuningMicrotest, ApplyTuningToCollTask_EveryOtherProtocol_LeavesTheTrafficEstimateAlone) {
  for (int proto : {NCCL_PROTO_LL128, NCCL_PROTO_SIMPLE}) {
    TaskPostTuning_CollFill fill;
    TaskPostTuning_SetTunerOutput(fill.tuningOut(), NCCL_ALGO_RING, proto);
    TaskPostTuning_StampDevFuncId(NCCL_ALGO_RING, proto);

    ASSERT_EQ(ncclSuccess, fill.RunApplyTuning()) << "proto " << proto;

    EXPECT_EQ(kCount * sizeof(float) * kAllReduceTrafficPerByte, fill.task()->trafficBytes) << "proto " << proto;
    EXPECT_EQ(kStampedDevFuncId, fill.task()->devFuncId) << "proto " << proto;
  }
}

TEST_F(TaskPostTuningMicrotest, ApplyTuningToCollTask_TunerMarkedTheEntryInvalid_ReportsAnInternalError) {
  TaskPostTuning_CollFill fill;
  TaskPostTuning_SetTunerOutput(fill.tuningOut(), NCCL_ALGO_RING, NCCL_PROTO_SIMPLE);
  fill.tuningOut()->valid = kInvalidTuning;

  EXPECT_EQ(ncclInternalError, fill.RunApplyTuning());

  EXPECT_EQ(ncclFuncAllReduce, fill.task()->func);
  EXPECT_EQ(kUnsetChannels, fill.task()->nMaxChannels);
  EXPECT_EQ(kUnresolvedDevFuncId, fill.task()->devFuncId);
}

TEST_F(TaskPostTuningMicrotest, ApplyTuningToCollTask_FillRejectsTheConfig_PropagatesWithoutApplyingTheTunerOutput) {
  TaskPostTuning_CollFill fill;
  fill.config()->algSelection = kUnknownAlgSelection;
  TaskPostTuning_SetTunerOutput(fill.tuningOut(), NCCL_ALGO_RING, NCCL_PROTO_SIMPLE);

  EXPECT_EQ(ncclInvalidArgument, fill.RunApplyTuning());

  EXPECT_EQ(kUnsetChannels, fill.task()->nMaxChannels);
  EXPECT_EQ(kUnresolvedDevFuncId, fill.task()->devFuncId);
}

TEST_F(TaskPostTuningMicrotest, ApplyTuningToCollTask_TuningEntryLeftAtItsInitValue_CurrentlyPassesTheValidityGate) {
  TaskPostTuning_CollFill fill;

  EXPECT_EQ(ncclSuccess, fill.RunApplyTuning());

  EXPECT_EQ(NCCL_TUNING_ENTRY_INIT_VALUE, fill.tuningOut()->valid);
  EXPECT_EQ(NCCL_ALGO_UNDEF, fill.task()->algorithm);
  EXPECT_EQ(NCCL_PROTO_UNDEF, fill.task()->protocol);
}

TEST_F(TaskPostTuningMicrotest, DISABLED_ApplyTuningToCollTask_TuningEntryLeftAtItsInitValue_ReportsAnInternalError) {
  TaskPostTuning_CollFill fill;

  EXPECT_EQ(ncclInternalError, fill.RunApplyTuning());
}

}  // namespace
