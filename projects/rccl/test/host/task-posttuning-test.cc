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

constexpr bool kIsSend = true;
constexpr bool kIsRecv = false;
constexpr int kP2pRank = 1;
constexpr int kP2pPeer = 2;
constexpr int kP2pSendRound = 1;
constexpr int kP2pRecvRound = 3;
constexpr int kP2pChannels = 8;
constexpr int kP2pChannelsPerPeer = 2;
constexpr int kWideP2pChannels = 128;
constexpr int kWideP2pChannelsPerPeer = 64;
constexpr int kP2pConnIndex = 1;
constexpr int kUnusedConnIndex = 0;
constexpr int kNoChannelShift = 0;
constexpr int kScheduleSpare = 4;
constexpr int kNoScheduledRank = -1;
constexpr int kConnected = 1;
constexpr int kNotConnected = 0;
constexpr size_t kP2pBytes = kCount * sizeof(float);
constexpr int kRegistrationGranted = 1;
constexpr int kRegistrationDeclined = 0;
constexpr int kPxnEnabled = 0;
constexpr int kPxnDisabled = 1;
constexpr int kBatchDisabled = 0;
constexpr int kTwoLocalRanks = 2;
constexpr int kOneLocalRank = 1;
constexpr int kNoP2pAccess = 0;
constexpr int kProxyInThisProcess = 1;
constexpr int kProxyInAnotherProcess = 0;
constexpr int kP2pSetupConnIndex = 1;
constexpr uint8_t kPoisonedBase = 0xEE;

bool AllBytesAre(const unsigned char* p, std::size_t n, unsigned char v) {
  for (std::size_t i = 0; i < n; i++) {
    if (p[i] != v) {
      return false;
    }
  }
  return true;
}

std::vector<int> TaskPostTuning_SingleNodeChannels(int base, int nParts, int nChannels) {
  std::vector<int> channelIds;
  for (int part = 0; part < nParts; part++) {
    channelIds.push_back((base * nParts + part) & (nChannels - 1));
  }
  return channelIds;
}

::testing::AssertionResult TaskPostTuning_MaskHasExactly(const struct channelMasks& mask,
                                                         const std::vector<int>& channelIds) {
  struct channelMasks expected = {};
  for (int channelId : channelIds) {
    expected.masks[channelId / CHANNELS_PER_MASK_WORD] |= 1ULL << (channelId % CHANNELS_PER_MASK_WORD);
  }
  for (int word = 0; word < MAXCHANNELS / CHANNELS_PER_MASK_WORD; word++) {
    if (mask.masks[word] != expected.masks[word]) {
      return ::testing::AssertionFailure()
             << "masks[" << word << "] = " << mask.masks[word] << ", expected " << expected.masks[word];
    }
  }
  return ::testing::AssertionSuccess();
}

constexpr ncclFunc_t kLoweredCollectiveApis[] = {ncclFuncAlltoAll, ncclFuncScatter, ncclFuncGather};

bool TaskPostTuning_IsLoweredCollectiveApi(int collAPI) {
  for (ncclFunc_t lowered : kLoweredCollectiveApis) {
    if (collAPI == lowered) {
      return true;
    }
  }
  return false;
}

// The schedule is the full permutation init.cc:1498-1503 builds; the spare rounds hold no rank at all.
class TaskPostTuning_P2pScene {
 public:
  explicit TaskPostTuning_P2pScene(int nP2pChannels = kP2pChannels, int nChannelsPerPeer = kP2pChannelsPerPeer)
    : scene_(kRanks, kP2pRank),
      schedule_(kRanks + kScheduleSpare),
      planPeers_(kRanks),
      connectSend_(kRanks),
      connectRecv_(kRanks),
      channelPeers_(static_cast<size_t>(nP2pChannels) * kRanks),
      peerSlots_(static_cast<size_t>(nP2pChannels) * kRanks) {
    struct ncclComm* comm = scene_.comm();
    for (int round = 0; round < kRanks + kScheduleSpare; round++) {
      schedule_[round].sendRank = round < kRanks ? (kP2pRank + round) % kRanks : kNoScheduledRank;
      schedule_[round].recvRank = round < kRanks ? (kP2pRank - round + kRanks) % kRanks : kNoScheduledRank;
    }
    comm->p2pSchedule = schedule_.data();
    comm->planner.peers = planPeers_.data();
    comm->connectSend = connectSend_.data();
    comm->connectRecv = connectRecv_.data();
    comm->p2pnChannels = nP2pChannels;
    comm->p2pnChannelsPerPeer = nChannelsPerPeer;
    comm->p2pChannelShiftSize = kNoChannelShift;
    for (int channelId = 0; channelId < nP2pChannels; channelId++) {
      for (int peer = 0; peer < kRanks; peer++) {
        peerSlots_[Slot(channelId, peer)] = &channelPeers_[Slot(channelId, peer)];
      }
      comm->channels[channelId].peers = &peerSlots_[Slot(channelId, 0)];
    }
    ncclMemoryPoolConstruct(&comm->memPool_ncclTaskP2p);
  }

  struct ncclComm* comm() { return scene_.comm(); }
  TaskPrepScene* scene() { return &scene_; }
  ncclComm::P2pSchedulePair& ScheduleAt(int round) { return schedule_[round]; }
  struct ncclChannelPeer& ChannelPeer(int channelId, int peer) { return channelPeers_[Slot(channelId, peer)]; }
  struct ncclConnector* Send(int channelId, int peer) { return &ChannelPeer(channelId, peer).send[kP2pConnIndex]; }
  struct ncclConnector* Recv(int channelId, int peer) { return &ChannelPeer(channelId, peer).recv[kP2pConnIndex]; }

  struct ncclConnector* Conn(bool isSendNotRecv, int channelId, int peer) {
    return isSendNotRecv ? Send(channelId, peer) : Recv(channelId, peer);
  }

  const struct channelMasks& connectSend(int peer) const { return connectSend_[peer]; }
  const struct channelMasks& connectRecv(int peer) const { return connectRecv_[peer]; }

  const struct channelMasks& connect(bool isSendNotRecv, int peer) const {
    return isSendNotRecv ? connectSend_[peer] : connectRecv_[peer];
  }

  bool& Seen(bool isSendNotRecv, int peer) {
    return isSendNotRecv ? planPeers_[peer].sendSeen : planPeers_[peer].recvSeen;
  }

  std::vector<int> Channels(bool isSendNotRecv, int nParts, int nChannels) const {
    return TaskPostTuning_SingleNodeChannels(isSendNotRecv ? kP2pSendRound : kP2pRecvRound, nParts, nChannels);
  }

 private:
  size_t Slot(int channelId, int peer) const { return static_cast<size_t>(channelId) * kRanks + peer; }

  TaskPrepScene scene_;
  std::vector<ncclComm::P2pSchedulePair> schedule_;
  std::vector<ncclKernelPlanner::Peer> planPeers_;
  std::vector<struct channelMasks> connectSend_;
  std::vector<struct channelMasks> connectRecv_;
  std::vector<struct ncclChannelPeer> channelPeers_;
  std::vector<struct ncclChannelPeer*> peerSlots_;
};

TEST_F(TaskPostTuningMicrotest, P2pChannelBase_SendSide_TakesTheRoundThatSchedulesThePeerAsASendTarget) {
  TaskPostTuning_P2pScene p2p;
  uint8_t base = kPoisonedBase;

  ASSERT_EQ(ncclSuccess, postTuneP2pChannelBase(p2p.comm(), kP2pPeer, kIsSend, &base));

  EXPECT_EQ(kP2pSendRound, base);
}

TEST_F(TaskPostTuningMicrotest, P2pChannelBase_RecvSide_TakesTheRoundThatSchedulesThePeerAsARecvSource) {
  TaskPostTuning_P2pScene p2p;
  uint8_t base = kPoisonedBase;

  ASSERT_EQ(ncclSuccess, postTuneP2pChannelBase(p2p.comm(), kP2pPeer, kIsRecv, &base));

  EXPECT_EQ(kP2pRecvRound, base);
}

TEST_F(TaskPostTuningMicrotest, P2pChannelBase_EveryPeerInTheSchedule_ResolvesToItsOwnRoundOnEachSide) {
  TaskPostTuning_P2pScene p2p;

  for (int peer = 0; peer < kRanks; peer++) {
    uint8_t sendBase = kPoisonedBase;
    uint8_t recvBase = kPoisonedBase;

    ASSERT_EQ(ncclSuccess, postTuneP2pChannelBase(p2p.comm(), peer, kIsSend, &sendBase)) << "peer " << peer;
    ASSERT_EQ(ncclSuccess, postTuneP2pChannelBase(p2p.comm(), peer, kIsRecv, &recvBase)) << "peer " << peer;

    EXPECT_EQ((peer - kP2pRank + kRanks) % kRanks, sendBase) << "peer " << peer;
    EXPECT_EQ((kP2pRank - peer + kRanks) % kRanks, recvBase) << "peer " << peer;
  }
}

// Known single-node limit: nNodes is 1, so the returned value, nNodes and p2pChannelShiftSize stay unobservable.
TEST_F(TaskPostTuningMicrotest, P2pChannelBase_AnyPeer_AsksThisCommForItsEffectiveBatchSetting) {
  TaskPostTuning_P2pScene p2p;
  struct ncclComm* seen = nullptr;
  ScopedHook batchEnable(g_rcclEffectiveP2pBatchEnable, [&seen](struct ncclComm* comm) {
    seen = comm;
    return kBatchDisabled;
  });
  uint8_t base = kPoisonedBase;

  ASSERT_EQ(ncclSuccess, postTuneP2pChannelBase(p2p.comm(), kP2pPeer, kIsSend, &base));

  EXPECT_EQ(1, batchEnable.calls);
  EXPECT_EQ(p2p.comm(), seen);
  EXPECT_EQ(kP2pSendRound, base);
}

// Harness-only schedule: init.cc:1498-1503 builds p2pSchedule as a full permutation, so production always matches.
TEST_F(TaskPostTuningMicrotest, P2pChannelBase_HandBuiltScheduleOmittingThePeer_ScansPastTheRankCount) {
  TaskPostTuning_P2pScene p2p;
  const int kSpareRound = kRanks + kScheduleSpare - 1;
  p2p.ScheduleAt(kP2pSendRound).sendRank = kP2pRank;
  p2p.ScheduleAt(kSpareRound).sendRank = kP2pPeer;
  uint8_t base = kPoisonedBase;

  ASSERT_EQ(ncclSuccess, postTuneP2pChannelBase(p2p.comm(), kP2pPeer, kIsSend, &base));

  EXPECT_EQ(kSpareRound, base);
}

TEST_F(TaskPostTuningMicrotest, P2pRecordPreconnect_PeerOutsideTheCommunicator_RejectsItAndMarksNothing) {
  TaskPostTuning_P2pScene p2p;

  for (int peer : {-1, kRanks}) {
    bool needPreconnect = false;

    EXPECT_EQ(ncclInvalidArgument, postTuneP2pRecordPreconnect(p2p.comm(), peer, kIsSend, &needPreconnect))
      << "peer " << peer;

    EXPECT_FALSE(needPreconnect) << "peer " << peer;
  }
  EXPECT_FALSE(p2p.comm()->planner.peers[kP2pPeer].sendSeen);
  EXPECT_TRUE(TaskPostTuning_MaskHasExactly(p2p.connectSend(kP2pPeer), {}));
}

TEST_F(TaskPostTuningMicrotest, P2pRecordPreconnect_PeerIsThisRank_SucceedsWithoutRecordingTheSide) {
  TaskPostTuning_P2pScene p2p;
  bool needPreconnect = false;

  EXPECT_EQ(ncclSuccess, postTuneP2pRecordPreconnect(p2p.comm(), kP2pRank, kIsSend, &needPreconnect));

  EXPECT_FALSE(needPreconnect);
  EXPECT_FALSE(p2p.comm()->planner.peers[kP2pRank].sendSeen);
  EXPECT_TRUE(TaskPostTuning_MaskHasExactly(p2p.connectSend(kP2pRank), {}));
}

TEST_F(TaskPostTuningMicrotest, P2pRecordPreconnect_FreshSendPeer_ClaimsItsSendChannelsAndLeavesTheRecvSideAlone) {
  TaskPostTuning_P2pScene p2p;
  const std::vector<int> sendChannels =
    TaskPostTuning_SingleNodeChannels(kP2pSendRound, kP2pChannelsPerPeer, kP2pChannels);
  bool needPreconnect = false;

  ASSERT_EQ(ncclSuccess, postTuneP2pRecordPreconnect(p2p.comm(), kP2pPeer, kIsSend, &needPreconnect));

  EXPECT_TRUE(needPreconnect);
  EXPECT_TRUE(p2p.comm()->planner.peers[kP2pPeer].sendSeen);
  EXPECT_FALSE(p2p.comm()->planner.peers[kP2pPeer].recvSeen);
  for (int channelId : sendChannels) {
    EXPECT_EQ(kConnected, p2p.Send(channelId, kP2pPeer)->hasSeen) << "channel " << channelId;
    EXPECT_EQ(kConnected, p2p.Send(channelId, kP2pPeer)->p2pOnly) << "channel " << channelId;
    EXPECT_EQ(kNotConnected, p2p.Recv(channelId, kP2pPeer)->hasSeen) << "channel " << channelId;
    EXPECT_EQ(kNotConnected, p2p.ChannelPeer(channelId, kP2pPeer).send[kUnusedConnIndex].hasSeen)
      << "channel " << channelId;
  }
  EXPECT_TRUE(TaskPostTuning_MaskHasExactly(p2p.connectSend(kP2pPeer), sendChannels));
  EXPECT_TRUE(TaskPostTuning_MaskHasExactly(p2p.connectRecv(kP2pPeer), {}));
  EXPECT_TRUE(TaskPostTuning_MaskHasExactly(p2p.connectSend(kP2pRank), {}));
}

TEST_F(TaskPostTuningMicrotest, P2pRecordPreconnect_FreshRecvPeer_ClaimsItsRecvChannelsAndLeavesTheSendSideAlone) {
  TaskPostTuning_P2pScene p2p;
  const std::vector<int> recvChannels =
    TaskPostTuning_SingleNodeChannels(kP2pRecvRound, kP2pChannelsPerPeer, kP2pChannels);
  bool needPreconnect = false;

  ASSERT_EQ(ncclSuccess, postTuneP2pRecordPreconnect(p2p.comm(), kP2pPeer, kIsRecv, &needPreconnect));

  EXPECT_TRUE(needPreconnect);
  EXPECT_TRUE(p2p.comm()->planner.peers[kP2pPeer].recvSeen);
  EXPECT_FALSE(p2p.comm()->planner.peers[kP2pPeer].sendSeen);
  for (int channelId : recvChannels) {
    EXPECT_EQ(kConnected, p2p.Recv(channelId, kP2pPeer)->hasSeen) << "channel " << channelId;
    EXPECT_EQ(kConnected, p2p.Recv(channelId, kP2pPeer)->p2pOnly) << "channel " << channelId;
    EXPECT_EQ(kNotConnected, p2p.Send(channelId, kP2pPeer)->hasSeen) << "channel " << channelId;
    EXPECT_EQ(kNotConnected, p2p.ChannelPeer(channelId, kP2pPeer).recv[kUnusedConnIndex].hasSeen)
      << "channel " << channelId;
  }
  EXPECT_TRUE(TaskPostTuning_MaskHasExactly(p2p.connectRecv(kP2pPeer), recvChannels));
  EXPECT_TRUE(TaskPostTuning_MaskHasExactly(p2p.connectSend(kP2pPeer), {}));
}

// Production duplicates the loop body per side, so each of these runs on the recv copy as well as the send one.
TEST_F(TaskPostTuningMicrotest, P2pRecordPreconnect_PeerAlreadyRecordedOnThatSide_ClaimsNoChannel) {
  for (bool isSendNotRecv : {kIsSend, kIsRecv}) {
    TaskPostTuning_P2pScene p2p;
    p2p.Seen(isSendNotRecv, kP2pPeer) = true;
    bool needPreconnect = false;

    EXPECT_EQ(ncclSuccess, postTuneP2pRecordPreconnect(p2p.comm(), kP2pPeer, isSendNotRecv, &needPreconnect))
      << "isSendNotRecv " << isSendNotRecv;

    EXPECT_FALSE(needPreconnect) << "isSendNotRecv " << isSendNotRecv;
    EXPECT_TRUE(TaskPostTuning_MaskHasExactly(p2p.connect(isSendNotRecv, kP2pPeer), {}))
      << "isSendNotRecv " << isSendNotRecv;
    for (int channelId : p2p.Channels(isSendNotRecv, kP2pChannelsPerPeer, kP2pChannels)) {
      EXPECT_EQ(kNotConnected, p2p.Conn(isSendNotRecv, channelId, kP2pPeer)->hasSeen) << "channel " << channelId;
    }
  }
}

TEST_F(TaskPostTuningMicrotest, P2pRecordPreconnect_PeerRecordedOnTheOtherSideOnly_StillClaimsThisSide) {
  TaskPostTuning_P2pScene p2p;
  p2p.comm()->planner.peers[kP2pPeer].recvSeen = true;
  bool needPreconnect = false;

  ASSERT_EQ(ncclSuccess, postTuneP2pRecordPreconnect(p2p.comm(), kP2pPeer, kIsSend, &needPreconnect));

  EXPECT_TRUE(needPreconnect);
  EXPECT_TRUE(TaskPostTuning_MaskHasExactly(
    p2p.connectSend(kP2pPeer), TaskPostTuning_SingleNodeChannels(kP2pSendRound, kP2pChannelsPerPeer, kP2pChannels)));
}

TEST_F(TaskPostTuningMicrotest, P2pRecordPreconnect_SomeChannelsAlreadyConnected_RequestsOnlyTheRemainingOnes) {
  for (bool isSendNotRecv : {kIsSend, kIsRecv}) {
    TaskPostTuning_P2pScene p2p;
    const std::vector<int> channels = p2p.Channels(isSendNotRecv, kP2pChannelsPerPeer, kP2pChannels);
    p2p.Conn(isSendNotRecv, channels.front(), kP2pPeer)->hasSeen = kConnected;
    bool needPreconnect = false;

    ASSERT_EQ(ncclSuccess, postTuneP2pRecordPreconnect(p2p.comm(), kP2pPeer, isSendNotRecv, &needPreconnect))
      << "isSendNotRecv " << isSendNotRecv;

    EXPECT_TRUE(needPreconnect) << "isSendNotRecv " << isSendNotRecv;
    EXPECT_EQ(kNotConnected, p2p.Conn(isSendNotRecv, channels.front(), kP2pPeer)->p2pOnly);
    EXPECT_EQ(kConnected, p2p.Conn(isSendNotRecv, channels.back(), kP2pPeer)->p2pOnly);
    EXPECT_TRUE(TaskPostTuning_MaskHasExactly(p2p.connect(isSendNotRecv, kP2pPeer), {channels.back()}))
      << "isSendNotRecv " << isSendNotRecv;
  }
}

TEST_F(TaskPostTuningMicrotest, P2pRecordPreconnect_EveryChannelAlreadyConnected_RecordsTheSideWithoutAskingForSetup) {
  TaskPostTuning_P2pScene p2p;
  for (int channelId : TaskPostTuning_SingleNodeChannels(kP2pSendRound, kP2pChannelsPerPeer, kP2pChannels)) {
    p2p.Send(channelId, kP2pPeer)->hasSeen = kConnected;
  }
  bool needPreconnect = false;

  ASSERT_EQ(ncclSuccess, postTuneP2pRecordPreconnect(p2p.comm(), kP2pPeer, kIsSend, &needPreconnect));

  EXPECT_FALSE(needPreconnect);
  EXPECT_TRUE(p2p.comm()->planner.peers[kP2pPeer].sendSeen);
  EXPECT_TRUE(TaskPostTuning_MaskHasExactly(p2p.connectSend(kP2pPeer), {}));
}

TEST_F(TaskPostTuningMicrotest, P2pRecordPreconnect_CallerAlreadyNeedsSetup_KeepsTheFlagSetWhenNothingIsClaimed) {
  TaskPostTuning_P2pScene p2p;
  for (int channelId : TaskPostTuning_SingleNodeChannels(kP2pSendRound, kP2pChannelsPerPeer, kP2pChannels)) {
    p2p.Send(channelId, kP2pPeer)->hasSeen = kConnected;
  }
  bool needPreconnect = true;

  ASSERT_EQ(ncclSuccess, postTuneP2pRecordPreconnect(p2p.comm(), kP2pPeer, kIsSend, &needPreconnect));

  EXPECT_TRUE(needPreconnect);
}

TEST_F(TaskPostTuningMicrotest, P2pRecordPreconnect_ChannelsBeyondTheFirstMaskWord_SetBitsInTheirOwnWord) {
  for (bool isSendNotRecv : {kIsSend, kIsRecv}) {
    TaskPostTuning_P2pScene p2p(kWideP2pChannels, kWideP2pChannelsPerPeer);
    const std::vector<int> channels = p2p.Channels(isSendNotRecv, kWideP2pChannelsPerPeer, kWideP2pChannels);
    bool needPreconnect = false;

    ASSERT_EQ(ncclSuccess, postTuneP2pRecordPreconnect(p2p.comm(), kP2pPeer, isSendNotRecv, &needPreconnect))
      << "isSendNotRecv " << isSendNotRecv;

    EXPECT_LE(CHANNELS_PER_MASK_WORD, channels.front()) << "isSendNotRecv " << isSendNotRecv;
    EXPECT_TRUE(TaskPostTuning_MaskHasExactly(p2p.connect(isSendNotRecv, kP2pPeer), channels))
      << "isSendNotRecv " << isSendNotRecv;
  }
}

class TaskPostTuning_P2pFill {
 public:
  explicit TaskPostTuning_P2pFill(ncclFunc_t func = ncclFuncSend, int peer = kP2pPeer)
    : tInfo_(p2p_.scene()->NewTuningInfo(p2p_.scene()->NewSendRecv(func, peer))) {
    std::memset(&task_, kPoison, sizeof(task_));
  }

  struct ncclComm* comm() { return p2p_.comm(); }
  struct ncclRawTaskSendRecv* raw() { return &tInfo_->raw->sendRecv; }
  struct ncclTaskP2p* task() { return &task_; }
  ncclResult_t Run() { return fillP2pTaskFromRaw(comm(), tInfo_, &task_); }

  bool TaskUntouched() const {
    return AllBytesAre(reinterpret_cast<const unsigned char*>(&task_), sizeof(task_), kPoison);
  }

 private:
  TaskPostTuning_P2pScene p2p_;
  struct ncclTaskTuningInfo* tInfo_;
  struct ncclTaskP2p task_;
};

TEST_F(TaskPostTuningMicrotest, FillP2pTaskFromRaw_Send_CopiesTheRawFieldsAndAllowsUserBuffers) {
  TaskPostTuning_P2pFill fill;
  ncclProfilerEventMask = kProfilerEventMask;
  fill.raw()->collAPI = ncclFuncBroadcast;
  // The 0xA5 poison already reads true for a bool, so the write is only observable from a seeded false.
  fill.task()->allowUB = false;

  ASSERT_EQ(ncclSuccess, fill.Run());

  const struct ncclTaskP2p* task = fill.task();
  EXPECT_EQ(ncclFuncSend, task->func);
  EXPECT_EQ(ncclFuncBroadcast, task->collAPI);
  EXPECT_EQ(fill.raw()->buff, task->buff);
  EXPECT_EQ(kCount, task->count);
  EXPECT_EQ(ncclFloat32, task->datatype);
  EXPECT_EQ(kP2pPeer, task->root);
  EXPECT_EQ(kP2pBytes, task->bytes);
  EXPECT_TRUE(task->allowUB);
  EXPECT_EQ(kProfilerEventMask, task->eActivationMask);
  EXPECT_EQ(nullptr, task->groupApiEventHandle);
  EXPECT_EQ(nullptr, task->p2pApiEventHandle);
}

TEST_F(TaskPostTuningMicrotest, FillP2pTaskFromRaw_Recv_CopiesTheRawFieldsAndAllowsUserBuffers) {
  TaskPostTuning_P2pFill fill(ncclFuncRecv);
  fill.raw()->collAPI = ncclFuncBroadcast;
  fill.task()->allowUB = false;

  ASSERT_EQ(ncclSuccess, fill.Run());

  EXPECT_EQ(ncclFuncRecv, fill.task()->func);
  EXPECT_EQ(ncclFuncBroadcast, fill.task()->collAPI);
  EXPECT_EQ(kP2pPeer, fill.task()->root);
  EXPECT_TRUE(fill.task()->allowUB);
}

TEST_F(TaskPostTuningMicrotest, FillP2pTaskFromRaw_LoweredCollectiveApis_ForbidUserBuffers) {
  for (ncclFunc_t collAPI : kLoweredCollectiveApis) {
    TaskPostTuning_P2pFill fill;
    fill.raw()->collAPI = collAPI;

    ASSERT_EQ(ncclSuccess, fill.Run()) << "collAPI " << collAPI;

    EXPECT_FALSE(fill.task()->allowUB) << "collAPI " << collAPI;
    EXPECT_EQ(collAPI, fill.task()->collAPI) << "collAPI " << collAPI;
  }
}

TEST_F(TaskPostTuningMicrotest, FillP2pTaskFromRaw_EveryOtherCollectiveApi_AllowsUserBuffers) {
  for (int collAPI = 0; collAPI < ncclNumFuncs; collAPI++) {
    if (TaskPostTuning_IsLoweredCollectiveApi(collAPI)) {
      continue;
    }
    TaskPostTuning_P2pFill fill;
    fill.raw()->collAPI = static_cast<ncclFunc_t>(collAPI);
    fill.task()->allowUB = false;

    ASSERT_EQ(ncclSuccess, fill.Run()) << "collAPI " << collAPI;

    EXPECT_TRUE(fill.task()->allowUB) << "collAPI " << collAPI;
  }
}

TEST_F(TaskPostTuningMicrotest, FillP2pTaskFromRaw_PeerOutsideTheCommunicator_RejectsItWithoutWritingTheTask) {
  for (int peer : {-1, kRanks}) {
    TaskPostTuning_P2pFill fill(ncclFuncSend, peer);

    EXPECT_EQ(ncclInvalidArgument, fill.Run()) << "peer " << peer;

    EXPECT_TRUE(fill.TaskUntouched()) << "peer " << peer;
  }
}

TEST_F(TaskPostTuningMicrotest, FillP2pTaskFromRaw_PeerIsThisRank_IsAcceptedAndRootedAtThisRank) {
  TaskPostTuning_P2pFill fill(ncclFuncSend, kP2pRank);

  ASSERT_EQ(ncclSuccess, fill.Run());

  EXPECT_EQ(kP2pRank, fill.task()->root);
}

TEST_F(TaskPostTuningMicrotest, FillP2pTaskFromRaw_FuncIsNeitherSendNorRecv_ReportsAnInternalErrorAndWritesNothing) {
  TaskPostTuning_P2pFill fill(ncclFuncAllReduce);

  EXPECT_EQ(ncclInternalError, fill.Run());

  EXPECT_TRUE(fill.TaskUntouched());
}

struct TaskPostTuning_NetRegistrationLog {
  std::vector<struct ncclConnector*> conns;
  struct ncclComm* comm = nullptr;
  void* buff = nullptr;
  size_t bytes = 0;
  ncclCommCallbackQueue* cleanupQueue = nullptr;
};

struct TaskPostTuning_IpcRegistrationLog {
  struct ncclComm* comm = nullptr;
  void* buff = nullptr;
  size_t bytes = 0;
  int peer = kNoScheduledRank;
  ncclCommCallbackQueue* cleanupQueue = nullptr;
};

ncclRegisterP2pNetBufferFn TaskPostTuning_RecordNetRegistration(TaskPostTuning_NetRegistrationLog* log, int regFlag,
                                                                ncclResult_t result) {
  return [log, regFlag, result](struct ncclComm* comm, void* buff, size_t bytes, struct ncclConnector* conn,
                                int* regFlagOut, void** handle, ncclCommCallbackQueue* cleanupQueue) {
    log->conns.push_back(conn);
    log->comm = comm;
    log->buff = buff;
    log->bytes = bytes;
    log->cleanupQueue = cleanupQueue;
    *regFlagOut = regFlag;
    *handle = nullptr;
    return result;
  };
}

ncclRegisterP2pIpcBufferFn TaskPostTuning_RecordIpcRegistration(TaskPostTuning_IpcRegistrationLog* log,
                                                                ncclResult_t result) {
  return [log, result](struct ncclComm* comm, void* buff, size_t bytes, int peer, int* regFlagOut, void** regAddr,
                       ncclCommCallbackQueue* cleanupQueue) {
    log->comm = comm;
    log->buff = buff;
    log->bytes = bytes;
    log->peer = peer;
    log->cleanupQueue = cleanupQueue;
    *regFlagOut = kRegistrationGranted;
    *regAddr = nullptr;
    return result;
  };
}

class TaskPostTuning_P2pRegister {
 public:
  explicit TaskPostTuning_P2pRegister(bool isSendNotRecv = kIsSend)
    : isSendNotRecv_(isSendNotRecv),
      channels_(TaskPostTuning_SingleNodeChannels(isSendNotRecv ? kP2pSendRound : kP2pRecvRound, kP2pChannelsPerPeer,
                                                  kP2pChannels)) {
    task_ = {};
    task_.func = isSendNotRecv ? ncclFuncSend : ncclFuncRecv;
    task_.root = kP2pPeer;
    task_.buff = buffer_;
    task_.bytes = kP2pBytes;
    task_.allowUB = true;
  }

  TaskPostTuning_P2pScene* scene() { return &p2p_; }
  struct ncclComm* comm() { return p2p_.comm(); }
  struct ncclTaskP2p* task() { return &task_; }
  struct ncclConnector* conn(int part) { return p2p_.Conn(isSendNotRecv_, channels_[part], kP2pPeer); }
  int parts() const { return static_cast<int>(channels_.size()); }

  std::vector<struct ncclConnector*> AllConns() {
    std::vector<struct ncclConnector*> conns;
    for (int part = 0; part < parts(); part++) {
      conns.push_back(conn(part));
    }
    return conns;
  }

  void UseNetworkTransport() {
    for (int channelId = 0; channelId < kP2pChannels; channelId++) {
      for (int peer = 0; peer < kRanks; peer++) {
        struct ncclConnector* connector = p2p_.Conn(isSendNotRecv_, channelId, peer);
        connector->transportComm = isSendNotRecv_ ? &netTransport.send : &netTransport.recv;
        connector->proxyConn.sameProcess = kProxyInThisProcess;
        connector->conn.flags = NCCL_DIRECT_NIC;
      }
    }
  }

  void UseIpcTransport(int flags) { conn(0)->conn.flags = flags; }

  ncclResult_t Run(int protocol = NCCL_PROTO_SIMPLE) {
    return postTuneP2pRegisterBuffer(p2p_.comm(), &task_, isSendNotRecv_, protocol);
  }

 private:
  TaskPostTuning_P2pScene p2p_;
  bool isSendNotRecv_;
  std::vector<int> channels_;
  struct ncclTaskP2p task_;
  char buffer_[kP2pBytes] = {};
};

// Owns the log and the hook so each net site is one line; log_ is declared first so it outlives the hook.
class TaskPostTuning_P2pNetRegister : public TaskPostTuning_P2pRegister {
 public:
  explicit TaskPostTuning_P2pNetRegister(bool isSendNotRecv = kIsSend, int regFlag = kRegistrationGranted,
                                         ncclResult_t result = ncclSuccess)
    : TaskPostTuning_P2pRegister(isSendNotRecv),
      hook_(g_ncclRegisterP2pNetBuffer, TaskPostTuning_RecordNetRegistration(&log_, regFlag, result)) {
    UseNetworkTransport();
  }

  const TaskPostTuning_NetRegistrationLog& log() const { return log_; }
  int calls() const { return hook_.calls; }

 private:
  TaskPostTuning_NetRegistrationLog log_;
  ScopedHook<ncclResult_t(struct ncclComm*, void*, size_t, struct ncclConnector*, int*, void**,
                          ncclCommCallbackQueue*)>
    hook_;
};

// postTuneP2pTasks always passes an untuned NCCL_PROTO_UNDEF, so only a direct call reaches the body below.
TEST_F(TaskPostTuningMicrotest, P2pRegisterBuffer_ProtocolIsNotSimple_RegistersNothing) {
  for (int protocol : {NCCL_PROTO_UNDEF, NCCL_PROTO_LL, NCCL_PROTO_LL128}) {
    TaskPostTuning_P2pNetRegister reg;

    EXPECT_EQ(ncclSuccess, reg.Run(protocol)) << "protocol " << protocol;

    EXPECT_EQ(0, reg.calls()) << "protocol " << protocol;
  }
}

TEST_F(TaskPostTuningMicrotest, P2pRegisterBuffer_TaskIsNotEligible_RegistersNothing) {
  const struct {
    const char* name;
    void (*apply)(struct ncclTaskP2p*);
  } kIneligible[] = {
    {"allowUB", [](struct ncclTaskP2p* task) { task->allowUB = false; }},
    {"bytes", [](struct ncclTaskP2p* task) { task->bytes = 0; }},
    {"buff", [](struct ncclTaskP2p* task) { task->buff = nullptr; }},
    {"self", [](struct ncclTaskP2p* task) { task->root = kP2pRank; }},
  };

  for (const auto& ineligible : kIneligible) {
    TaskPostTuning_P2pNetRegister reg;
    ineligible.apply(reg.task());

    EXPECT_EQ(ncclSuccess, reg.Run()) << ineligible.name;

    EXPECT_EQ(0, reg.calls()) << ineligible.name;
  }
}

TEST_F(TaskPostTuningMicrotest, P2pRegisterBuffer_DirectNicSendConnection_RegistersTheBufferOnEveryPartConnection) {
  TaskPostTuning_P2pNetRegister reg;

  ASSERT_EQ(ncclSuccess, reg.Run());

  EXPECT_EQ(reg.parts(), reg.calls());
  EXPECT_EQ(reg.AllConns(), reg.log().conns);
  EXPECT_EQ(reg.comm(), reg.log().comm);
  EXPECT_EQ(reg.task()->buff, reg.log().buff);
  EXPECT_EQ(kP2pBytes, reg.log().bytes);
  EXPECT_EQ(&reg.comm()->planner.collCleanupQueue, reg.log().cleanupQueue);
}

TEST_F(TaskPostTuningMicrotest, P2pRegisterBuffer_DirectNicRecvConnection_RegistersOnTheRecvRoundsConnections) {
  TaskPostTuning_P2pNetRegister reg(kIsRecv);

  ASSERT_EQ(ncclSuccess, reg.Run());

  EXPECT_EQ(reg.parts(), reg.calls());
  EXPECT_EQ(reg.AllConns(), reg.log().conns);
}

TEST_F(TaskPostTuningMicrotest, P2pRegisterBuffer_FirstPartDeclinesRegistration_LeavesTheRemainingPartsUnregistered) {
  TaskPostTuning_P2pNetRegister reg(kIsSend, kRegistrationDeclined);

  ASSERT_EQ(ncclSuccess, reg.Run());

  EXPECT_EQ(1, reg.calls());
  EXPECT_EQ(std::vector<struct ncclConnector*>{reg.conn(0)}, reg.log().conns);
}

TEST_F(TaskPostTuningMicrotest, P2pRegisterBuffer_NetRegistrationFails_PropagatesTheFailure) {
  TaskPostTuning_P2pNetRegister reg(kIsSend, kRegistrationGranted, ncclSystemError);

  EXPECT_EQ(ncclSystemError, reg.Run());

  EXPECT_EQ(1, reg.calls());
}

TEST_F(TaskPostTuningMicrotest, P2pRegisterBuffer_NetworkConnectionWithoutDirectNic_RegistersNothing) {
  TaskPostTuning_P2pNetRegister reg;
  reg.conn(0)->conn.flags = NCCL_P2P_WRITE | NCCL_P2P_READ;

  EXPECT_EQ(ncclSuccess, reg.Run());

  EXPECT_EQ(0, reg.calls());
}

TEST_F(TaskPostTuningMicrotest, P2pRegisterBuffer_NetworkProxyInAnotherProcess_RegistersNothing) {
  TaskPostTuning_P2pNetRegister reg;
  reg.conn(0)->proxyConn.sameProcess = kProxyInAnotherProcess;

  EXPECT_EQ(ncclSuccess, reg.Run());

  EXPECT_EQ(0, reg.calls());
}

TEST_F(TaskPostTuningMicrotest, P2pRegisterBuffer_PxnCarriesTheTransfer_RegistersNothing) {
  TaskPostTuning_P2pNetRegister reg;
  reg.comm()->isAllNvlink = 1;
  reg.comm()->maxLocalRanks = kTwoLocalRanks;
  ScopedHook pxn(g_pxnDisable, [](struct ncclComm*) { return kPxnEnabled; });

  EXPECT_EQ(ncclSuccess, reg.Run());

  EXPECT_EQ(0, reg.calls());
}

TEST_F(TaskPostTuningMicrotest, P2pRegisterBuffer_PxnCannotCarryTheTransfer_RegistersEveryPartConnection) {
  const struct {
    const char* missing;
    int pxnDisabled;
    int isAllNvlink;
    int maxLocalRanks;
  } kNoPxn[] = {
    {"pxn disabled", kPxnDisabled, 1, kTwoLocalRanks},
    {"no nvlink", kPxnEnabled, 0, kTwoLocalRanks},
    {"single local rank", kPxnEnabled, 1, kOneLocalRank},
  };

  for (const auto& testCase : kNoPxn) {
    TaskPostTuning_P2pNetRegister reg;
    reg.comm()->isAllNvlink = testCase.isAllNvlink;
    reg.comm()->maxLocalRanks = testCase.maxLocalRanks;
    const int pxnDisabled = testCase.pxnDisabled;
    ScopedHook pxn(g_pxnDisable, [pxnDisabled](struct ncclComm*) { return pxnDisabled; });

    ASSERT_EQ(ncclSuccess, reg.Run()) << testCase.missing;

    EXPECT_EQ(reg.parts(), reg.calls()) << testCase.missing;
  }
}

TEST_F(TaskPostTuningMicrotest, P2pRegisterBuffer_PeerToPeerConnection_RegistersTheBufferOnceForThatPeer) {
  for (int flags : {NCCL_P2P_WRITE, NCCL_P2P_READ, NCCL_P2P_WRITE | NCCL_P2P_READ}) {
    TaskPostTuning_P2pRegister reg;
    TaskPostTuning_IpcRegistrationLog log;
    ScopedHook ipc(g_ncclRegisterP2pIpcBuffer, TaskPostTuning_RecordIpcRegistration(&log, ncclSuccess));
    reg.UseIpcTransport(flags);

    ASSERT_EQ(ncclSuccess, reg.Run()) << "flags " << flags;

    EXPECT_EQ(1, ipc.calls) << "flags " << flags;
    EXPECT_EQ(reg.comm(), log.comm) << "flags " << flags;
    EXPECT_EQ(reg.task()->buff, log.buff) << "flags " << flags;
    EXPECT_EQ(kP2pBytes, log.bytes) << "flags " << flags;
    EXPECT_EQ(kP2pPeer, log.peer) << "flags " << flags;
    EXPECT_EQ(&reg.comm()->planner.collCleanupQueue, log.cleanupQueue) << "flags " << flags;
  }
}

TEST_F(TaskPostTuningMicrotest, P2pRegisterBuffer_ConnectionWithoutPeerToPeerAccess_RegistersNothing) {
  TaskPostTuning_P2pRegister reg;
  TaskPostTuning_IpcRegistrationLog log;
  ScopedHook ipc(g_ncclRegisterP2pIpcBuffer, TaskPostTuning_RecordIpcRegistration(&log, ncclSuccess));
  reg.UseIpcTransport(kNoP2pAccess);

  EXPECT_EQ(ncclSuccess, reg.Run());

  EXPECT_EQ(0, ipc.calls);
}

TEST_F(TaskPostTuningMicrotest, P2pRegisterBuffer_PeerToPeerRegistrationFails_PropagatesTheFailure) {
  TaskPostTuning_P2pRegister reg;
  TaskPostTuning_IpcRegistrationLog log;
  ScopedHook ipc(g_ncclRegisterP2pIpcBuffer, TaskPostTuning_RecordIpcRegistration(&log, ncclInvalidUsage));
  reg.UseIpcTransport(NCCL_P2P_WRITE);

  EXPECT_EQ(ncclInvalidUsage, reg.Run());

  EXPECT_EQ(1, ipc.calls);
}

class TaskPostTuning_P2pDrive {
 public:
  TaskPostTuning_P2pDrive() { ncclIntruQueueConstruct(&queue_); }

  struct ncclTaskTuningInfo* Add(ncclFunc_t func, int peer) {
    struct ncclTaskTuningInfo* tInfo = p2p_.scene()->NewTuningInfo(p2p_.scene()->NewSendRecv(func, peer));
    ncclIntruQueueEnqueue(&queue_, tInfo);
    return tInfo;
  }

  void MarkEveryChannelConnected() {
    for (int channelId = 0; channelId < p2p_.comm()->p2pnChannels; channelId++) {
      for (int peer = 0; peer < kRanks; peer++) {
        p2p_.Send(channelId, peer)->hasSeen = kConnected;
        p2p_.Recv(channelId, peer)->hasSeen = kConnected;
      }
    }
  }

  std::vector<struct ncclTaskP2p*> Queued(int peer, bool isSendNotRecv) {
    struct ncclIntruQueue<struct ncclTaskP2p, &ncclTaskP2p::next>* queue =
      isSendNotRecv ? &comm()->planner.peers[peer].sendQueue : &comm()->planner.peers[peer].recvQueue;
    std::vector<struct ncclTaskP2p*> tasks;
    for (struct ncclTaskP2p* task = ncclIntruQueueHead(queue); task != nullptr; task = task->next) {
      tasks.push_back(task);
    }
    return tasks;
  }

  TaskPostTuning_P2pScene* scene() { return &p2p_; }
  struct ncclComm* comm() { return p2p_.comm(); }
  ncclResult_t Run() { return postTuneP2pTasks(p2p_.comm(), &queue_); }

 private:
  TaskPostTuning_P2pScene p2p_;
  TaskTuningInfoQueue queue_;
};

TEST_F(TaskPostTuningMicrotest, P2pTasks_EmptyQueue_ConnectsNothingAndEnqueuesNothing) {
  TaskPostTuning_P2pDrive drive;
  ScopedHook setup(g_ncclTransportP2pSetup,
                   [](struct ncclComm*, struct ncclTopoGraph*, int, bool*) { return ncclSuccess; });

  EXPECT_EQ(ncclSuccess, drive.Run());

  EXPECT_EQ(0, setup.calls);
  EXPECT_EQ(0, drive.comm()->planner.nTasksP2p);
}

TEST_F(TaskPostTuningMicrotest, P2pTasks_FreshPeers_ConnectsOnceThenEnqueuesEachTaskOnItsOwnPeerQueue) {
  TaskPostTuning_P2pDrive drive;
  struct ncclComm* seenComm = nullptr;
  int seenConnIndex = 0;
  ScopedHook setup(g_ncclTransportP2pSetup, [&](struct ncclComm* comm, struct ncclTopoGraph* graph, int connIndex,
                                                bool* needsProxy) {
    seenComm = comm;
    seenConnIndex = connIndex;
    EXPECT_EQ(nullptr, graph);
    EXPECT_EQ(nullptr, needsProxy);
    return ncclSuccess;
  });
  struct ncclTaskTuningInfo* firstSend = drive.Add(ncclFuncSend, kP2pPeer);
  struct ncclTaskTuningInfo* recv = drive.Add(ncclFuncRecv, kP2pPeer);
  struct ncclTaskTuningInfo* secondSend = drive.Add(ncclFuncSend, kP2pPeer);
  // The two sends are otherwise field-identical, so only a distinct size pins the tail-insertion order.
  firstSend->raw->sendRecv.bytes = kP2pBytes / 2;

  ASSERT_EQ(ncclSuccess, drive.Run());

  EXPECT_EQ(1, setup.calls);
  EXPECT_EQ(drive.comm(), seenComm);
  EXPECT_EQ(kP2pSetupConnIndex, seenConnIndex);
  const std::vector<struct ncclTaskP2p*> sent = drive.Queued(kP2pPeer, kIsSend);
  const std::vector<struct ncclTaskP2p*> received = drive.Queued(kP2pPeer, kIsRecv);
  ASSERT_EQ(2u, sent.size());
  ASSERT_EQ(1u, received.size());
  EXPECT_EQ(ncclFuncSend, sent[0]->func);
  EXPECT_EQ(ncclFuncSend, sent[1]->func);
  EXPECT_EQ(ncclFuncRecv, received[0]->func);
  EXPECT_EQ(kP2pBytes / 2, sent[0]->bytes);
  EXPECT_EQ(kP2pBytes, sent[1]->bytes);
  EXPECT_EQ(3, drive.comm()->planner.nTasksP2p);
  EXPECT_EQ(2, drive.comm()->planner.nTasksP2pSend);
  EXPECT_EQ(1, drive.comm()->planner.nTasksP2pRecv);
  EXPECT_TRUE(TaskPostTuning_MaskHasExactly(
    drive.scene()->connectSend(kP2pPeer),
    TaskPostTuning_SingleNodeChannels(kP2pSendRound, kP2pChannelsPerPeer, kP2pChannels)));
  EXPECT_TRUE(TaskPostTuning_MaskHasExactly(
    drive.scene()->connectRecv(kP2pPeer),
    TaskPostTuning_SingleNodeChannels(kP2pRecvRound, kP2pChannelsPerPeer, kP2pChannels)));
  EXPECT_EQ(nullptr, firstSend->raw);
  EXPECT_EQ(nullptr, recv->raw);
  EXPECT_EQ(nullptr, secondSend->raw);
}

TEST_F(TaskPostTuningMicrotest, P2pTasks_SeparatePeers_ClaimTheirOwnSendChannelsAndQueues) {
  TaskPostTuning_P2pDrive drive;
  const int kOtherPeer = 3;
  ScopedHook setup(g_ncclTransportP2pSetup,
                   [](struct ncclComm*, struct ncclTopoGraph*, int, bool*) { return ncclSuccess; });
  drive.Add(ncclFuncSend, kP2pPeer);
  drive.Add(ncclFuncSend, kOtherPeer);

  ASSERT_EQ(ncclSuccess, drive.Run());

  ASSERT_EQ(1u, drive.Queued(kP2pPeer, kIsSend).size());
  ASSERT_EQ(1u, drive.Queued(kOtherPeer, kIsSend).size());
  EXPECT_EQ(kP2pPeer, drive.Queued(kP2pPeer, kIsSend)[0]->root);
  EXPECT_EQ(kOtherPeer, drive.Queued(kOtherPeer, kIsSend)[0]->root);
  EXPECT_EQ(2, drive.comm()->planner.nTasksP2p);
  EXPECT_TRUE(TaskPostTuning_MaskHasExactly(
    drive.scene()->connectSend(kP2pPeer),
    TaskPostTuning_SingleNodeChannels(kP2pSendRound, kP2pChannelsPerPeer, kP2pChannels)));
  // Peer 3 rounds to its own disjoint channel pair, which one base derived for the whole queue would miss.
  EXPECT_TRUE(TaskPostTuning_MaskHasExactly(
    drive.scene()->connectSend(kOtherPeer),
    TaskPostTuning_SingleNodeChannels((kOtherPeer - kP2pRank + kRanks) % kRanks, kP2pChannelsPerPeer,
                                      kP2pChannels)));
  EXPECT_TRUE(TaskPostTuning_MaskHasExactly(drive.scene()->connectRecv(kP2pPeer), {}));
  EXPECT_TRUE(TaskPostTuning_MaskHasExactly(drive.scene()->connectRecv(kOtherPeer), {}));
}

TEST_F(TaskPostTuningMicrotest, P2pTasks_EveryChannelAlreadyConnected_SkipsTransportSetupAndStillEnqueues) {
  TaskPostTuning_P2pDrive drive;
  ScopedHook setup(g_ncclTransportP2pSetup,
                   [](struct ncclComm*, struct ncclTopoGraph*, int, bool*) { return ncclSuccess; });
  drive.MarkEveryChannelConnected();
  drive.Add(ncclFuncSend, kP2pPeer);

  ASSERT_EQ(ncclSuccess, drive.Run());

  EXPECT_EQ(0, setup.calls);
  EXPECT_EQ(1u, drive.Queued(kP2pPeer, kIsSend).size());
  EXPECT_EQ(1, drive.comm()->planner.nTasksP2p);
}

TEST_F(TaskPostTuningMicrotest, P2pTasks_TransportSetupFails_PropagatesWithoutEnqueueingAnyTask) {
  TaskPostTuning_P2pDrive drive;
  ScopedHook setup(g_ncclTransportP2pSetup,
                   [](struct ncclComm*, struct ncclTopoGraph*, int, bool*) { return ncclSystemError; });
  struct ncclTaskTuningInfo* tInfo = drive.Add(ncclFuncSend, kP2pPeer);

  EXPECT_EQ(ncclSystemError, drive.Run());

  EXPECT_EQ(1, setup.calls);
  EXPECT_TRUE(drive.Queued(kP2pPeer, kIsSend).empty());
  EXPECT_EQ(0, drive.comm()->planner.nTasksP2p);
  EXPECT_NE(nullptr, tInfo->raw);
}

TEST_F(TaskPostTuningMicrotest, P2pTasks_PeerOutsideTheCommunicator_RejectsTheGroupBeforeConnecting) {
  TaskPostTuning_P2pDrive drive;
  ScopedHook setup(g_ncclTransportP2pSetup,
                   [](struct ncclComm*, struct ncclTopoGraph*, int, bool*) { return ncclSuccess; });
  drive.Add(ncclFuncSend, kP2pPeer);
  drive.Add(ncclFuncSend, kRanks);

  EXPECT_EQ(ncclInvalidArgument, drive.Run());

  EXPECT_EQ(0, setup.calls);
  EXPECT_EQ(0, drive.comm()->planner.nTasksP2p);
  EXPECT_TRUE(drive.Queued(kP2pPeer, kIsSend).empty());
}

TEST_F(TaskPostTuningMicrotest, P2pTasks_LaterTaskIsMalformed_PropagatesAfterEnqueueingTheEarlierOne) {
  TaskPostTuning_P2pDrive drive;
  ScopedHook setup(g_ncclTransportP2pSetup,
                   [](struct ncclComm*, struct ncclTopoGraph*, int, bool*) { return ncclSuccess; });
  drive.Add(ncclFuncSend, kP2pPeer);
  drive.Add(ncclFuncAllReduce, kP2pPeer);

  EXPECT_EQ(ncclInternalError, drive.Run());

  EXPECT_EQ(1u, drive.Queued(kP2pPeer, kIsSend).size());
  EXPECT_EQ(1, drive.comm()->planner.nTasksP2p);
}

TEST_F(TaskPostTuningMicrotest, P2pTasks_UntunedTasks_CarryNoProtocolAndNeverReachBufferRegistration) {
  TaskPostTuning_P2pDrive drive;
  TaskPostTuning_NetRegistrationLog netLog;
  TaskPostTuning_IpcRegistrationLog ipcLog;
  ScopedHook setup(g_ncclTransportP2pSetup,
                   [](struct ncclComm*, struct ncclTopoGraph*, int, bool*) { return ncclSuccess; });
  ScopedHook net(g_ncclRegisterP2pNetBuffer,
                 TaskPostTuning_RecordNetRegistration(&netLog, kRegistrationGranted, ncclSuccess));
  ScopedHook ipc(g_ncclRegisterP2pIpcBuffer, TaskPostTuning_RecordIpcRegistration(&ipcLog, ncclSuccess));
  for (int channelId : TaskPostTuning_SingleNodeChannels(kP2pSendRound, kP2pChannelsPerPeer, kP2pChannels)) {
    drive.scene()->Send(channelId, kP2pPeer)->conn.flags = NCCL_P2P_WRITE;
  }
  struct ncclTaskTuningInfo* tInfo = drive.Add(ncclFuncSend, kP2pPeer);

  ASSERT_EQ(ncclSuccess, drive.Run());

  EXPECT_EQ(NCCL_PROTO_UNDEF, tInfo->tuningOut.proto);
  EXPECT_EQ(0, net.calls);
  EXPECT_EQ(0, ipc.calls);
  EXPECT_EQ(1u, drive.Queued(kP2pPeer, kIsSend).size());
}

// The only test that reaches postTuneP2pRegisterBuffer through the drain, so deleting that call turns it red.
TEST_F(TaskPostTuningMicrotest, P2pTasks_TunedToSimple_RegistersTheBufferForThePeer) {
  TaskPostTuning_P2pDrive drive;
  TaskPostTuning_IpcRegistrationLog ipcLog;
  ScopedHook setup(g_ncclTransportP2pSetup,
                   [](struct ncclComm*, struct ncclTopoGraph*, int, bool*) { return ncclSuccess; });
  ScopedHook ipc(g_ncclRegisterP2pIpcBuffer, TaskPostTuning_RecordIpcRegistration(&ipcLog, ncclSuccess));
  for (int channelId : TaskPostTuning_SingleNodeChannels(kP2pSendRound, kP2pChannelsPerPeer, kP2pChannels)) {
    drive.scene()->Send(channelId, kP2pPeer)->conn.flags = NCCL_P2P_WRITE;
  }
  struct ncclTaskTuningInfo* tInfo = drive.Add(ncclFuncSend, kP2pPeer);
  tInfo->tuningOut.proto = NCCL_PROTO_SIMPLE;

  ASSERT_EQ(ncclSuccess, drive.Run());

  EXPECT_EQ(1, ipc.calls);
  EXPECT_EQ(kP2pPeer, ipcLog.peer);
}

constexpr int kNumRmaCtx = 4;
constexpr int kNumRmaSig = 4;
constexpr int kRmaCtx = 2;
constexpr int kRmaSigIdx = 3;
constexpr int kRmaPeer = 2;
constexpr int kPreexistingRmaTasks = 5;
constexpr int kNoSignalsConfigured = 0;
constexpr int kSupportedDriverVersion = 12050;
constexpr int kUnsupportedDriverVersion = 12049;
constexpr unsigned int kRejectedRmaFlags = 1u;
constexpr uintptr_t kRmaSrcWindowBase = 0x90000000ull;
constexpr size_t kRmaSrcWindowSize = 1ull << 20;
constexpr size_t kRmaSrcOffset = 0x400;
constexpr size_t kRmaPeerWindowSize = 1ull << 21;
constexpr size_t kRmaPeerWinOffset = 0x800;
constexpr size_t kRmaCount = 64;
constexpr int kRmaOpCount = 3;
constexpr int kNoOperationsWaitedFor = 0;

ncclWaitSignalDesc_t TaskPostTuning_WaitDesc(int peer, int sigIdx, int ctx) {
  ncclWaitSignalDesc_t desc = {};
  desc.opCnt = kRmaOpCount;
  desc.peer = peer;
  desc.sigIdx = sigIdx;
  desc.ctx = ctx;
  return desc;
}

// Every gate ahead of the func-specific checks is satisfied here, so a test only has to break one.
class TaskPostTuning_RmaScene {
 public:
  TaskPostTuning_RmaScene() : rmaQueues_(kNumRmaCtx) {
    struct ncclComm* comm = scene_.comm();
    comm->hostRmaSupport = true;
    comm->config.numRmaCtx = kNumRmaCtx;
    comm->config.numRmaSig = kNumRmaSig;
    comm->planner.nTasksRma = kPreexistingRmaTasks;
    comm->planner.rmaTaskQueues = rmaQueues_.data();
    for (int ctx = 0; ctx < kNumRmaCtx; ctx++) {
      ncclIntruQueueConstruct(&rmaQueues_[ctx]);
    }
    ncclMemoryPoolConstruct(&comm->memPool_ncclTaskRma);
    ncclCudaDriverVersionCache = kSupportedDriverVersion;
    ncclProfilerEventMask = kProfilerEventMask;
    peerShadow_.winHost = &peerWindow_;
    peerWindow_.size = kRmaPeerWindowSize;
    srcWindow_.userPtr = TaskPostTuning_Addr(kRmaSrcWindowBase);
    srcWindow_.size = kRmaSrcWindowSize;
    srcWindow_.winFlags = NCCL_WIN_COLL_SYMMETRIC;
    g_shadowPoolToHost = [this](struct ncclShadowPool*, void* devObj, void** outHostObj) {
      shadowRequest_ = devObj;
      *outHostObj = &peerShadow_;
      return ncclSuccess;
    };
    g_devrFindWindow = [this](struct ncclComm*, void const* ptr, struct ncclDevrWindow** window) {
      findWindowRequest_ = ptr;
      *window = &srcWindow_;
      return ncclSuccess;
    };
    g_rmaInitialized = [](struct ncclComm*) { return true; };
  }

  struct ncclComm* comm() { return scene_.comm(); }
  struct ncclDevrWindow* srcWindow() { return &srcWindow_; }
  struct ncclDevrWindow* peerWindow() { return &peerWindow_; }
  const void* shadowRequest() const { return shadowRequest_; }
  const void* findWindowRequest() const { return findWindowRequest_; }

  struct ncclRawTaskRma PutSignal() {
    struct ncclRawTaskRma raw = {};
    raw.func = ncclFuncPutSignal;
    raw.rmaOp.putSignal.localbuff = TaskPostTuning_Addr(kRmaSrcWindowBase + kRmaSrcOffset);
    raw.rmaOp.putSignal.count = kRmaCount;
    raw.rmaOp.putSignal.datatype = ncclFloat32;
    raw.rmaOp.putSignal.peer = kRmaPeer;
    raw.rmaOp.putSignal.peerWin = &peerHandle_;
    raw.rmaOp.putSignal.peerWinOffset = kRmaPeerWinOffset;
    raw.rmaOp.putSignal.sigIdx = kRmaSigIdx;
    raw.rmaOp.putSignal.ctx = kRmaCtx;
    return raw;
  }

  struct ncclRawTaskRma Signal() {
    struct ncclRawTaskRma raw = {};
    raw.func = ncclFuncSignal;
    raw.rmaOp.signal.peer = kRmaPeer;
    raw.rmaOp.signal.sigIdx = kRmaSigIdx;
    raw.rmaOp.signal.ctx = kRmaCtx;
    return raw;
  }

  struct ncclRawTaskRma WaitSignal(std::vector<ncclWaitSignalDesc_t>* descs) {
    struct ncclRawTaskRma raw = {};
    raw.func = ncclFuncWaitSignal;
    raw.rmaOp.waitSignal.nDesc = static_cast<int>(descs->size());
    raw.rmaOp.waitSignal.signalDescs = descs->data();
    return raw;
  }

  ncclResult_t Run(const struct ncclRawTaskRma& raw) { return postTuneRmaTaskAppend(comm(), &raw); }

  std::vector<struct ncclTaskRma*> Tasks(int ctx) {
    std::vector<struct ncclTaskRma*> tasks;
    for (struct ncclTaskRma* task = ncclIntruQueueHead(&rmaQueues_[ctx]); task != nullptr; task = task->next) {
      tasks.push_back(task);
    }
    return tasks;
  }

  TaskTuningInfoQueue* TuningQueue() { return &tuningQueue_; }

  struct ncclTaskTuningInfo* EnqueueForDrain(const struct ncclRawTaskRma& raw) {
    struct ncclRawTask* rawTask = scene_.NewRaw(ncclTaskKindRma);
    rawTask->rma = raw;
    struct ncclTaskTuningInfo* tInfo = scene_.NewTuningInfo(rawTask);
    ncclIntruQueueEnqueue(&tuningQueue_, tInfo);
    return tInfo;
  }

  int AppendedTaskCount() { return comm()->planner.nTasksRma - kPreexistingRmaTasks; }

 private:
  TaskPrepScene scene_;
  std::vector<ncclIntruQueue<struct ncclTaskRma, &ncclTaskRma::next>> rmaQueues_;
  TaskTuningInfoQueue tuningQueue_ = {};
  struct ncclWindow_vidmem peerHandle_ = {};
  struct ncclWindow_vidmem peerShadow_ = {};
  struct ncclDevrWindow peerWindow_ = {};
  struct ncclDevrWindow srcWindow_ = {};
  const void* shadowRequest_ = nullptr;
  const void* findWindowRequest_ = nullptr;
};

::testing::AssertionResult TaskPostTuning_NoRmaTasksAppended(TaskPostTuning_RmaScene* rma) {
  if (rma->AppendedTaskCount() != 0) {
    return ::testing::AssertionFailure() << "planner.nTasksRma moved by " << rma->AppendedTaskCount();
  }
  for (int ctx = 0; ctx < kNumRmaCtx; ctx++) {
    if (!rma->Tasks(ctx).empty()) {
      return ::testing::AssertionFailure() << "rmaTaskQueues[" << ctx << "] holds " << rma->Tasks(ctx).size();
    }
  }
  return ::testing::AssertionSuccess();
}

TEST_F(TaskPostTuningMicrotest, RmaTaskAppend_FuncIsNotAnRmaOp_ReportsAnInternalErrorBeforeAnyCommCheck) {
  TaskPostTuning_RmaScene rma;
  rma.comm()->hostRmaSupport = false;
  struct ncclRawTaskRma raw = rma.PutSignal();
  raw.func = ncclFuncAllReduce;

  EXPECT_EQ(ncclInternalError, rma.Run(raw));
  EXPECT_TRUE(TaskPostTuning_NoRmaTasksAppended(&rma));
}

TEST_F(TaskPostTuningMicrotest, RmaTaskAppend_HostRmaUnsupported_RejectsEveryRmaFuncAndAppendsNothing) {
  std::vector<ncclWaitSignalDesc_t> descs = {TaskPostTuning_WaitDesc(kRmaPeer, kRmaSigIdx, kRmaCtx)};
  for (int func = 0; func < 3; func++) {
    TaskPostTuning_RmaScene rma;
    rma.comm()->hostRmaSupport = false;
    const struct ncclRawTaskRma raw =
      func == 0 ? rma.PutSignal() : (func == 1 ? rma.Signal() : rma.WaitSignal(&descs));

    EXPECT_EQ(ncclInvalidArgument, rma.Run(raw)) << "func index " << func;
    EXPECT_TRUE(TaskPostTuning_NoRmaTasksAppended(&rma)) << "func index " << func;
  }
}

TEST_F(TaskPostTuningMicrotest, RmaTaskAppend_DriverBelowTheRmaMinimum_RejectsTheTaskAndAppendsNothing) {
  TaskPostTuning_RmaScene rma;
  ncclCudaDriverVersionCache = kUnsupportedDriverVersion;

  EXPECT_EQ(ncclInvalidUsage, rma.Run(rma.PutSignal()));
  EXPECT_TRUE(TaskPostTuning_NoRmaTasksAppended(&rma));
}

TEST_F(TaskPostTuningMicrotest, RmaTaskAppend_DriverAtTheRmaMinimum_IsAccepted) {
  TaskPostTuning_RmaScene rma;
  ncclCudaDriverVersionCache = kSupportedDriverVersion;

  EXPECT_EQ(ncclSuccess, rma.Run(rma.PutSignal()));
  EXPECT_EQ(1, rma.AppendedTaskCount());
}

TEST_F(TaskPostTuningMicrotest, RmaTaskAppend_HostRmaUnsupportedAndDriverTooOld_ReportsTheUnsupportedCommFirst) {
  TaskPostTuning_RmaScene rma;
  rma.comm()->hostRmaSupport = false;
  ncclCudaDriverVersionCache = kUnsupportedDriverVersion;

  EXPECT_EQ(ncclInvalidArgument, rma.Run(rma.PutSignal()));
}

TEST_F(TaskPostTuningMicrotest, RmaTaskAppend_SignalIndexBelowZero_RejectsTheTaskAndAppendsNothing) {
  TaskPostTuning_RmaScene rma;
  struct ncclRawTaskRma raw = rma.Signal();
  raw.rmaOp.signal.sigIdx = -1;

  EXPECT_EQ(ncclInvalidArgument, rma.Run(raw));
  EXPECT_TRUE(TaskPostTuning_NoRmaTasksAppended(&rma));
}

TEST_F(TaskPostTuningMicrotest, RmaTaskAppend_SignalIndexAtTheConfiguredCount_RejectsTheTaskAndAppendsNothing) {
  TaskPostTuning_RmaScene rma;
  struct ncclRawTaskRma raw = rma.Signal();
  raw.rmaOp.signal.sigIdx = kNumRmaSig;

  EXPECT_EQ(ncclInvalidArgument, rma.Run(raw));
  EXPECT_TRUE(TaskPostTuning_NoRmaTasksAppended(&rma));
}

TEST_F(TaskPostTuningMicrotest, RmaTaskAppend_SignalIndexAtTheTopOfTheRange_IsAccepted) {
  TaskPostTuning_RmaScene rma;
  struct ncclRawTaskRma raw = rma.Signal();
  raw.rmaOp.signal.sigIdx = kNumRmaSig - 1;

  EXPECT_EQ(ncclSuccess, rma.Run(raw));
  EXPECT_EQ(1, rma.AppendedTaskCount());
}

TEST_F(TaskPostTuningMicrotest, RmaTaskAppend_NoSignalsConfigured_RejectsWaitSignalOnItsImplicitIndex) {
  TaskPostTuning_RmaScene rma;
  rma.comm()->config.numRmaSig = kNoSignalsConfigured;
  std::vector<ncclWaitSignalDesc_t> descs = {TaskPostTuning_WaitDesc(kRmaPeer, 0, kRmaCtx)};

  EXPECT_EQ(ncclInvalidArgument, rma.Run(rma.WaitSignal(&descs)));
  EXPECT_TRUE(TaskPostTuning_NoRmaTasksAppended(&rma));
}

TEST_F(TaskPostTuningMicrotest, RmaTaskAppend_PutSignalCarriesFlags_RejectsTheTaskAndAppendsNothing) {
  TaskPostTuning_RmaScene rma;
  struct ncclRawTaskRma raw = rma.PutSignal();
  raw.rmaOp.putSignal.flags = kRejectedRmaFlags;

  EXPECT_EQ(ncclInvalidArgument, rma.Run(raw));
  EXPECT_TRUE(TaskPostTuning_NoRmaTasksAppended(&rma));
}

TEST_F(TaskPostTuningMicrotest, RmaTaskAppend_SignalCarriesFlags_RejectsTheTaskAndAppendsNothing) {
  TaskPostTuning_RmaScene rma;
  struct ncclRawTaskRma raw = rma.Signal();
  raw.rmaOp.signal.flags = kRejectedRmaFlags;

  EXPECT_EQ(ncclInvalidArgument, rma.Run(raw));
  EXPECT_TRUE(TaskPostTuning_NoRmaTasksAppended(&rma));
}

TEST_F(TaskPostTuningMicrotest, RmaTaskAppend_WaitSignalOverlappingAFlagBearingUnion_IgnoresTheFlags) {
  TaskPostTuning_RmaScene rma;
  std::vector<ncclWaitSignalDesc_t> descs = {TaskPostTuning_WaitDesc(kRmaPeer, kRmaSigIdx, kRmaCtx)};
  struct ncclRawTaskRma raw = rma.PutSignal();
  raw.rmaOp.putSignal.flags = kRejectedRmaFlags;
  raw.func = ncclFuncWaitSignal;
  raw.rmaOp.waitSignal.nDesc = static_cast<int>(descs.size());
  raw.rmaOp.waitSignal.signalDescs = descs.data();

  EXPECT_EQ(ncclSuccess, rma.Run(raw));
  EXPECT_EQ(1, rma.AppendedTaskCount());
}

TEST_F(TaskPostTuningMicrotest, RmaTaskAppend_SignalBeforeRmaInit_RejectsTheTaskAndAppendsNothing) {
  TaskPostTuning_RmaScene rma;
  ScopedHook initialized(g_rmaInitialized, [](struct ncclComm*) { return false; });

  EXPECT_EQ(ncclInvalidUsage, rma.Run(rma.Signal()));
  EXPECT_TRUE(TaskPostTuning_NoRmaTasksAppended(&rma));
}

TEST_F(TaskPostTuningMicrotest, RmaTaskAppend_WaitSignalBeforeRmaInit_RejectsTheTaskAndAppendsNothing) {
  TaskPostTuning_RmaScene rma;
  std::vector<ncclWaitSignalDesc_t> descs = {TaskPostTuning_WaitDesc(kRmaPeer, kRmaSigIdx, kRmaCtx)};
  ScopedHook initialized(g_rmaInitialized, [](struct ncclComm*) { return false; });

  EXPECT_EQ(ncclInvalidUsage, rma.Run(rma.WaitSignal(&descs)));
  EXPECT_TRUE(TaskPostTuning_NoRmaTasksAppended(&rma));
}

TEST_F(TaskPostTuningMicrotest, RmaTaskAppend_PutSignalBeforeRmaInit_IsAcceptedBecauseItsWindowTriggersInit) {
  TaskPostTuning_RmaScene rma;
  ScopedHook initialized(g_rmaInitialized, [](struct ncclComm*) { return false; });

  EXPECT_EQ(ncclSuccess, rma.Run(rma.PutSignal()));
  EXPECT_EQ(1, rma.AppendedTaskCount());
}

TEST_F(TaskPostTuningMicrotest, RmaTaskAppend_SignalAfterRmaInit_AsksThisCommWhetherRmaIsInitialized) {
  TaskPostTuning_RmaScene rma;
  struct ncclComm* asked = nullptr;
  ScopedHook initialized(g_rmaInitialized, [&](struct ncclComm* comm) {
    asked = comm;
    return true;
  });

  ASSERT_EQ(ncclSuccess, rma.Run(rma.Signal()));

  EXPECT_EQ(1, initialized.calls);
  EXPECT_EQ(rma.comm(), asked);
}

TEST_F(TaskPostTuningMicrotest, RmaTaskAppend_PutSignalPeerWindowIsNull_RejectsItWithoutResolvingAShadow) {
  TaskPostTuning_RmaScene rma;
  ScopedHook shadow(g_shadowPoolToHost, [](struct ncclShadowPool*, void*, void**) { return ncclSuccess; });
  struct ncclRawTaskRma raw = rma.PutSignal();
  raw.rmaOp.putSignal.peerWin = nullptr;

  EXPECT_EQ(ncclInvalidArgument, rma.Run(raw));
  EXPECT_EQ(0, shadow.calls);
  EXPECT_TRUE(TaskPostTuning_NoRmaTasksAppended(&rma));
}

TEST_F(TaskPostTuningMicrotest, RmaTaskAppend_PutSignalShadowLookupFails_PropagatesTheFailure) {
  TaskPostTuning_RmaScene rma;
  ScopedHook shadow(g_shadowPoolToHost, [](struct ncclShadowPool*, void*, void**) { return ncclSystemError; });

  EXPECT_EQ(ncclSystemError, rma.Run(rma.PutSignal()));
  EXPECT_TRUE(TaskPostTuning_NoRmaTasksAppended(&rma));
}

TEST_F(TaskPostTuningMicrotest, RmaTaskAppend_PutSignalSourceBufferIsNull_RejectsItWithoutSearchingForAWindow) {
  TaskPostTuning_RmaScene rma;
  ScopedHook find(g_devrFindWindow, [](struct ncclComm*, void const*, struct ncclDevrWindow** window) {
    *window = nullptr;
    return ncclSuccess;
  });
  struct ncclRawTaskRma raw = rma.PutSignal();
  raw.rmaOp.putSignal.localbuff = nullptr;

  EXPECT_EQ(ncclInvalidArgument, rma.Run(raw));
  EXPECT_EQ(0, find.calls);
  EXPECT_TRUE(TaskPostTuning_NoRmaTasksAppended(&rma));
}

TEST_F(TaskPostTuningMicrotest, RmaTaskAppend_PutSignalWindowSearchFails_PropagatesTheFailure) {
  TaskPostTuning_RmaScene rma;
  ScopedHook find(g_devrFindWindow, [](struct ncclComm*, void const*, struct ncclDevrWindow** window) {
    *window = nullptr;
    return ncclSystemError;
  });

  EXPECT_EQ(ncclSystemError, rma.Run(rma.PutSignal()));
  EXPECT_TRUE(TaskPostTuning_NoRmaTasksAppended(&rma));
}

TEST_F(TaskPostTuningMicrotest, RmaTaskAppend_PutSignalSourceIsOutsideEveryWindow_RejectsTheTask) {
  TaskPostTuning_RmaScene rma;
  ScopedHook find(g_devrFindWindow, [](struct ncclComm*, void const*, struct ncclDevrWindow** window) {
    *window = nullptr;
    return ncclSuccess;
  });

  EXPECT_EQ(ncclInvalidArgument, rma.Run(rma.PutSignal()));
  EXPECT_TRUE(TaskPostTuning_NoRmaTasksAppended(&rma));
}

TEST_F(TaskPostTuningMicrotest, RmaTaskAppend_PutSignalAccepted_ResolvesThePeerHandleAndTheSourceBufferItWasGiven) {
  TaskPostTuning_RmaScene rma;
  const struct ncclRawTaskRma raw = rma.PutSignal();

  ASSERT_EQ(ncclSuccess, rma.Run(raw));

  EXPECT_EQ(raw.rmaOp.putSignal.peerWin, rma.shadowRequest());
  EXPECT_EQ(raw.rmaOp.putSignal.localbuff, rma.findWindowRequest());
}

TEST_F(TaskPostTuningMicrotest, RmaTaskAppend_PutSignalSourceWindowIsMultiSegment_RejectsTheTask) {
  TaskPostTuning_RmaScene rma;
  struct ncclDevrWindow* src = rma.srcWindow();
  ScopedHook multi(g_devrWindowIsMultiSegment,
                   [src](struct ncclDevrWindow* window) { return window == src; });

  EXPECT_EQ(ncclInvalidArgument, rma.Run(rma.PutSignal()));
  EXPECT_TRUE(TaskPostTuning_NoRmaTasksAppended(&rma));
}

TEST_F(TaskPostTuningMicrotest, RmaTaskAppend_PutSignalPeerWindowIsMultiSegment_RejectsTheTask) {
  TaskPostTuning_RmaScene rma;
  struct ncclDevrWindow* peer = rma.peerWindow();
  ScopedHook multi(g_devrWindowIsMultiSegment,
                   [peer](struct ncclDevrWindow* window) { return window == peer; });

  EXPECT_EQ(ncclInvalidArgument, rma.Run(rma.PutSignal()));
  EXPECT_TRUE(TaskPostTuning_NoRmaTasksAppended(&rma));
}

TEST_F(TaskPostTuningMicrotest, RmaTaskAppend_PutSignalSourceWindowHasAHostBackedSegment_RejectsTheTask) {
  TaskPostTuning_RmaScene rma;
  struct ncclDevrWindow* src = rma.srcWindow();
  ScopedHook sysmem(g_devrWindowHasSysmemSegment,
                    [src](struct ncclDevrWindow* window) { return window == src; });

  EXPECT_EQ(ncclInvalidArgument, rma.Run(rma.PutSignal()));
  EXPECT_TRUE(TaskPostTuning_NoRmaTasksAppended(&rma));
}

TEST_F(TaskPostTuningMicrotest, RmaTaskAppend_PutSignalPeerWindowHasAHostBackedSegment_RejectsTheTask) {
  TaskPostTuning_RmaScene rma;
  struct ncclDevrWindow* peer = rma.peerWindow();
  ScopedHook sysmem(g_devrWindowHasSysmemSegment,
                    [peer](struct ncclDevrWindow* window) { return window == peer; });

  EXPECT_EQ(ncclInvalidArgument, rma.Run(rma.PutSignal()));
  EXPECT_TRUE(TaskPostTuning_NoRmaTasksAppended(&rma));
}

TEST_F(TaskPostTuningMicrotest, RmaTaskAppend_PutSignalSourceWindowWithoutTheSymmetricFlag_CurrentlyRejectsIt) {
  TaskPostTuning_RmaScene rma;
  rma.comm()->symmetricSupport = 0;
  rma.srcWindow()->winFlags = 0;

  EXPECT_EQ(ncclInvalidArgument, rma.Run(rma.PutSignal()));
  EXPECT_TRUE(TaskPostTuning_NoRmaTasksAppended(&rma));
}

TEST_F(TaskPostTuningMicrotest, DISABLED_RmaTaskAppend_PutSignalSourceWindowOnANonSymmetricComm_IgnoresTheFlag) {
  TaskPostTuning_RmaScene rma;
  rma.comm()->symmetricSupport = 0;
  rma.srcWindow()->winFlags = 0;

  EXPECT_EQ(ncclSuccess, rma.Run(rma.PutSignal()));
  EXPECT_EQ(1, rma.AppendedTaskCount());
}

TEST_F(TaskPostTuningMicrotest, RmaTaskAppend_PutSignalPeerOffsetPastTheWindowEnd_CurrentlyAppendsTheTaskAnyway) {
  TaskPostTuning_RmaScene rma;
  struct ncclRawTaskRma raw = rma.PutSignal();
  raw.rmaOp.putSignal.peerWinOffset = kRmaPeerWindowSize;

  EXPECT_EQ(ncclSuccess, rma.Run(raw));
  ASSERT_EQ(1u, rma.Tasks(kRmaCtx).size());
  EXPECT_EQ(kRmaPeerWindowSize, rma.Tasks(kRmaCtx)[0]->peerWinOffset);
}

TEST_F(TaskPostTuningMicrotest, DISABLED_RmaTaskAppend_PutSignalPeerOffsetPastTheWindowEnd_RejectsTheTask) {
  TaskPostTuning_RmaScene rma;
  struct ncclRawTaskRma raw = rma.PutSignal();
  raw.rmaOp.putSignal.peerWinOffset = kRmaPeerWindowSize;

  EXPECT_EQ(ncclInvalidArgument, rma.Run(raw));
  EXPECT_TRUE(TaskPostTuning_NoRmaTasksAppended(&rma));
}

TEST_F(TaskPostTuningMicrotest, RmaTaskAppend_WaitSignalWithoutADescriptorArray_RejectsTheTaskAndAppendsNothing) {
  TaskPostTuning_RmaScene rma;
  std::vector<ncclWaitSignalDesc_t> descs = {TaskPostTuning_WaitDesc(kRmaPeer, kRmaSigIdx, kRmaCtx)};
  struct ncclRawTaskRma raw = rma.WaitSignal(&descs);
  raw.rmaOp.waitSignal.signalDescs = nullptr;

  EXPECT_EQ(ncclInvalidArgument, rma.Run(raw));
  EXPECT_TRUE(TaskPostTuning_NoRmaTasksAppended(&rma));
}

TEST_F(TaskPostTuningMicrotest, RmaTaskAppend_WaitSignalWithZeroDescriptors_RejectsTheTaskAndAppendsNothing) {
  TaskPostTuning_RmaScene rma;
  std::vector<ncclWaitSignalDesc_t> descs = {TaskPostTuning_WaitDesc(kRmaPeer, kRmaSigIdx, kRmaCtx)};
  struct ncclRawTaskRma raw = rma.WaitSignal(&descs);
  raw.rmaOp.waitSignal.nDesc = 0;

  EXPECT_EQ(ncclInvalidArgument, rma.Run(raw));
  EXPECT_TRUE(TaskPostTuning_NoRmaTasksAppended(&rma));
}

TEST_F(TaskPostTuningMicrotest, RmaTaskAppend_WaitSignalDescriptorWaitsForNoOperation_RejectsTheTaskAndAppendsNothing) {
  TaskPostTuning_RmaScene rma;
  std::vector<ncclWaitSignalDesc_t> descs = {TaskPostTuning_WaitDesc(kRmaPeer, kRmaSigIdx, kRmaCtx)};
  descs[0].opCnt = kNoOperationsWaitedFor;

  EXPECT_EQ(ncclInvalidArgument, rma.Run(rma.WaitSignal(&descs)));
  EXPECT_TRUE(TaskPostTuning_NoRmaTasksAppended(&rma));
}

TEST_F(TaskPostTuningMicrotest, RmaTaskAppend_WaitSignalDescriptorWaitsForOneOperation_IsAccepted) {
  TaskPostTuning_RmaScene rma;
  std::vector<ncclWaitSignalDesc_t> descs = {TaskPostTuning_WaitDesc(kRmaPeer, kRmaSigIdx, kRmaCtx)};
  descs[0].opCnt = 1;

  EXPECT_EQ(ncclSuccess, rma.Run(rma.WaitSignal(&descs)));
  EXPECT_EQ(1, rma.AppendedTaskCount());
}

TEST_F(TaskPostTuningMicrotest, RmaTaskAppend_WaitSignalDescriptorSignalIndexBelowZero_RejectsTheTaskAndAppendsNothing) {
  TaskPostTuning_RmaScene rma;
  std::vector<ncclWaitSignalDesc_t> descs = {TaskPostTuning_WaitDesc(kRmaPeer, -1, kRmaCtx)};

  EXPECT_EQ(ncclInvalidArgument, rma.Run(rma.WaitSignal(&descs)));
  EXPECT_TRUE(TaskPostTuning_NoRmaTasksAppended(&rma));
}

TEST_F(TaskPostTuningMicrotest, RmaTaskAppend_WaitSignalDescriptorSignalIndexAtTheCount_RejectsTheTaskAndAppendsNothing) {
  TaskPostTuning_RmaScene rma;
  std::vector<ncclWaitSignalDesc_t> descs = {TaskPostTuning_WaitDesc(kRmaPeer, kNumRmaSig, kRmaCtx)};

  EXPECT_EQ(ncclInvalidArgument, rma.Run(rma.WaitSignal(&descs)));
  EXPECT_TRUE(TaskPostTuning_NoRmaTasksAppended(&rma));
}

TEST_F(TaskPostTuningMicrotest, RmaTaskAppend_WaitSignalLaterDescriptorIsInvalid_RejectsTheWholeTask) {
  TaskPostTuning_RmaScene rma;
  std::vector<ncclWaitSignalDesc_t> descs = {TaskPostTuning_WaitDesc(kRmaPeer, kRmaSigIdx, kRmaCtx),
                                             TaskPostTuning_WaitDesc(kRmaPeer, kNumRmaSig, kRmaCtx)};

  EXPECT_EQ(ncclInvalidArgument, rma.Run(rma.WaitSignal(&descs)));
  EXPECT_TRUE(TaskPostTuning_NoRmaTasksAppended(&rma));
}

}  // namespace
