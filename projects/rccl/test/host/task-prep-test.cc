/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests; the UUT is #include'd. Hipify leaves task_prep.cc byte-identical.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <typeinfo>
#include <vector>

#include "ScopedHook.h"
#include "TaskPrepScene.h"

#include TASK_PREP_CC_PATH

// Declared by the task_prep TUs and defined in fakes/, exactly as those TUs declare them.
int64_t ncclParamMinCTAs();
int64_t ncclParamMaxCTAs();
int64_t ncclParamNvlsChannels();
int64_t ncclParamCGAClusterSize();

namespace {

// ncclRawTask is a union, so the entry count alone cannot tell which fill function ran; the mask can.
std::vector<uint64_t> TuningMasks(struct ncclTaskTuningInfoQueue* tiq) {
  std::vector<uint64_t> masks;
  for (struct ncclTaskTuningInfo* t = ncclIntruQueueHead(&tiq->queue); t != nullptr; t = t->next) {
    masks.push_back(t->tuningIn.tuningMask);
  }
  return masks;
}

// task_pretuning.cc:37 for the coll path; :89/:106/:130 use NCCL_TUNING_MASK_ALL for the other three.
constexpr uint64_t kCollTuningMask = NCCL_TUNING_MASK_SYM_KERNELS | NCCL_TUNING_MASK_GENERAL_KERNELS;

// At rank == root: AlltoAll lowers 2 per rank, Gather 1 send plus nRanks recvs, Scatter the mirror of that.
constexpr size_t kLoweredP2pTasks = 2 * kRanks + (1 + kRanks) + (kRanks + 1);

// Records the func of every tuning call and writes a result distinct from NCCL_TUNING_RESULT_INIT.
class TaskPrep_TuningSpy {
 public:
  explicit TaskPrep_TuningSpy(ncclResult_t result = ncclSuccess)
    : result_(result),
      hook_(g_tuningCompute,
            [this](struct ncclTuningInput_t* in, struct ncclTuningResult_t* out) -> ncclResult_t {
              funcs_.push_back(in->func);
              if (result_ != ncclSuccess) {
                return result_;
              }
              out->valid = kTunedValid;
              out->timeUs = kTunedTimeUs;
              out->algo = NCCL_ALGO_RING;
              return ncclSuccess;
            }) {}

  const std::vector<ncclFunc_t>& funcs() const { return funcs_; }

 private:
  ncclResult_t result_;
  std::vector<ncclFunc_t> funcs_;
  ScopedHook<ncclResult_t(struct ncclTuningInput_t*, struct ncclTuningResult_t*)> hook_;
};

class TaskPrepMicrotest : public TaskPrepFakesFixture {};

TEST_F(TaskPrepMicrotest, SceneBuildsAWellFormedComm) {
  TaskPrepScene scene;
  ASSERT_NE(nullptr, scene.comm());
  EXPECT_EQ(kRanks, scene.comm()->nRanks);
  EXPECT_EQ(0, scene.comm()->rank);
  EXPECT_TRUE(ncclIntruQueueEmpty(&scene.comm()->rawTaskQueue.genericQueue));
  EXPECT_TRUE(ncclIntruQueueEmpty(&scene.comm()->rawTaskQueue.bcastQueue));
}

TEST_F(TaskPrepMicrotest, SceneBuildersTagEachRawTaskKind) {
  TaskPrepScene scene;
  EXPECT_EQ(ncclTaskKindColl, scene.NewColl(ncclFuncAllReduce)->kind);
  EXPECT_EQ(ncclTaskKindSendRecv, scene.NewSendRecv(ncclFuncSend, 1)->kind);
  EXPECT_EQ(ncclTaskKindRma, scene.NewRma(ncclFuncPutSignal)->kind);

  struct ncclRawTask* agv = scene.NewAllGatherV();
  EXPECT_EQ(ncclTaskKindAllGatherV, agv->kind);
  EXPECT_EQ(kRanks, agv->allGatherV.nRanks);
  ASSERT_NE(nullptr, agv->allGatherV.recvbuff);
  ASSERT_NE(nullptr, agv->allGatherV.counts);
}

TEST_F(TaskPrepMicrotest, PreTuningDrainsEveryRawTaskKind) {
  TaskPrepScene scene;
  scene.EnqueueGeneric(scene.NewColl(ncclFuncAllReduce));
  scene.EnqueueGeneric(scene.NewSendRecv(ncclFuncSend, 1));
  scene.EnqueueGeneric(scene.NewRma(ncclFuncPutSignal));
  scene.EnqueueGeneric(scene.NewAllGatherV());
  scene.EnqueueBcast(scene.NewColl(ncclFuncBroadcast));

  struct ncclTaskTuningInfoQueue tiq;
  ncclIntruQueueConstruct(&tiq.queue);
  ASSERT_EQ(ncclSuccess, ncclTaskPreTuning(scene.comm(), &scene.comm()->rawTaskQueue, &tiq));

  EXPECT_EQ(5u, QueueTasks(&tiq.queue).size());
  // The bcast is last: preTuningBcastFallsBack always returns true, so it stays a coll.
  EXPECT_EQ((std::vector<uint64_t>{kCollTuningMask, NCCL_TUNING_MASK_ALL, NCCL_TUNING_MASK_ALL,
                                   NCCL_TUNING_MASK_ALL, kCollTuningMask}),
            TuningMasks(&tiq));
  EXPECT_TRUE(ncclIntruQueueEmpty(&scene.comm()->rawTaskQueue.genericQueue));
  EXPECT_TRUE(ncclIntruQueueEmpty(&scene.comm()->rawTaskQueue.bcastQueue));
}

// Scaffolding only: an empty group reaches every stage, so section GC cannot drop the closure.
TEST_F(TaskPrepMicrotest, TaskPrepareLinksAgainstItsRealSiblings) {
  TaskPrepScene scene;
  ASSERT_EQ(ncclSuccess, ncclTaskPrepare(scene.comm(), nullptr));
}

TEST_F(TaskPrepMicrotest, RegLocalIsValidSeamInstallsAndResets) {
  bool isValid = true;
  ASSERT_EQ(ncclSuccess, ncclRegLocalIsValid(nullptr, &isValid));
  EXPECT_FALSE(isValid) << "the default must keep every successful-registration arm opt-in";

  {
    ScopedHook hook(g_regLocalIsValid, [](struct ncclReg*, bool* out) {
      *out = true;
      return ncclSuccess;
    });
    ASSERT_EQ(ncclSuccess, ncclRegLocalIsValid(nullptr, &isValid));
    EXPECT_TRUE(isValid);
  }

  isValid = true;
  ASSERT_EQ(ncclSuccess, ncclRegLocalIsValid(nullptr, &isValid));
  EXPECT_FALSE(isValid) << "ScopedHook must restore the default";

  g_regLocalIsValid = [](struct ncclReg*, bool* out) {
    *out = true;
    return ncclSuccess;
  };
  ResetNcclFakes();
  isValid = true;
  ASSERT_EQ(ncclSuccess, ncclRegLocalIsValid(nullptr, &isValid));
  EXPECT_FALSE(isValid) << "ResetNcclFakes must restore the default";
}

TEST_F(TaskPrepMicrotest, RegisterSeamsInstallAndReset) {
  int calls = 0;
  bool needConnect = false;
  int regFlag = 0;
  void* regAddr = nullptr;
  // target_type() witnesses the restore without calling the fail-loud default back out.
  const std::type_info* nvlsDefault = &g_ncclRegisterCollNvlsBuffers.target_type();
  const std::type_info* ipcDefault = &g_ncclRegisterP2pIpcBuffer.target_type();
  const std::type_info* netDefault = &g_ncclRegisterP2pNetBuffer.target_type();

  // Assigned directly, not through ScopedHook, so the restore below can only come from the owner.
  g_ncclRegisterCollNvlsBuffers = [&calls](struct ncclComm*, struct ncclTaskColl*, void**, void**,
                                           ncclCommCallbackQueue*, bool*) {
    ++calls;
    return ncclSuccess;
  };
  g_ncclRegisterP2pIpcBuffer = [&calls](struct ncclComm*, void*, size_t, int, int*, void**,
                                        ncclCommCallbackQueue*) {
    ++calls;
    return ncclSuccess;
  };
  g_ncclRegisterP2pNetBuffer = [&calls](struct ncclComm*, void*, size_t, struct ncclConnector*,
                                        int*, void**, ncclCommCallbackQueue*) {
    ++calls;
    return ncclSuccess;
  };
  EXPECT_EQ(ncclSuccess,
            ncclRegisterCollNvlsBuffers(nullptr, nullptr, nullptr, nullptr, nullptr, &needConnect));
  EXPECT_EQ(ncclSuccess,
            ncclRegisterP2pIpcBuffer(nullptr, nullptr, 0, 0, &regFlag, &regAddr, nullptr));
  EXPECT_EQ(ncclSuccess,
            ncclRegisterP2pNetBuffer(nullptr, nullptr, 0, nullptr, &regFlag, &regAddr, nullptr));
  EXPECT_EQ(3, calls);

  ResetRegisterStubs();
  EXPECT_EQ(nvlsDefault, &g_ncclRegisterCollNvlsBuffers.target_type())
      << "ResetRegisterStubs must restore the default";
  EXPECT_EQ(ipcDefault, &g_ncclRegisterP2pIpcBuffer.target_type());
  EXPECT_EQ(netDefault, &g_ncclRegisterP2pNetBuffer.target_type());
}

TEST_F(TaskPrepMicrotest, TransportAndCeSeamsInstallAndReset) {
  int setupCalls = 0;
  int ceInitCalls = 0;
  // target_type() witnesses the restore without calling the fail-loud default back out.
  const std::type_info* setupDefault = &g_ncclTransportP2pSetup.target_type();
  const std::type_info* ceInitDefault = &g_ncclCeInit.target_type();
  // Assigned directly, not through ScopedHook, so the restore below can only come from the owner.
  g_ncclTransportP2pSetup = [&setupCalls](struct ncclComm*, struct ncclTopoGraph*, int, bool*) {
    ++setupCalls;
    return ncclSuccess;
  };
  g_ncclCeInit = [&ceInitCalls](struct ncclComm*) {
    ++ceInitCalls;
    return ncclSuccess;
  };
  EXPECT_EQ(ncclSuccess, ncclTransportP2pSetup(nullptr, nullptr, 0, nullptr));
  EXPECT_EQ(ncclSuccess, ncclCeInit(nullptr));
  EXPECT_EQ(1, setupCalls);
  EXPECT_EQ(1, ceInitCalls);

  ResetTransportStubs();
  ResetCeFakes();
  EXPECT_EQ(setupDefault, &g_ncclTransportP2pSetup.target_type())
      << "ResetTransportStubs must restore the default";
  EXPECT_EQ(ceInitDefault, &g_ncclCeInit.target_type()) << "ResetCeFakes must restore the default";
}

TEST_F(TaskPrepMicrotest, EffectiveP2pBatchEnableSeamInstallsAndResets) {
  EXPECT_EQ(0, rcclEffectiveP2pBatchEnable(nullptr));
  {
    ScopedHook hook(g_rcclEffectiveP2pBatchEnable, [](struct ncclComm*) { return 3; });
    EXPECT_EQ(3, rcclEffectiveP2pBatchEnable(nullptr));
  }
  EXPECT_EQ(0, rcclEffectiveP2pBatchEnable(nullptr));
}

TEST_F(TaskPrepMicrotest, HandWrittenParamsDefaultAndFollowTheEnvSeam) {
  EXPECT_EQ(NCCL_CONFIG_UNDEF_INT, ncclParamMinCTAs());
  EXPECT_EQ(NCCL_CONFIG_UNDEF_INT, ncclParamMaxCTAs());
  EXPECT_EQ(NCCL_CONFIG_UNDEF_INT, ncclParamNvlsChannels());
  EXPECT_EQ(NCCL_CONFIG_UNDEF_INT, ncclParamCGAClusterSize());
  EXPECT_EQ(ROCM_VERSION >= 71200 ? 1 : 0, ncclParamGraphRegister());

  ScopedHook hook(g_loadParam, [](const char* env, int64_t deft) -> int64_t {
    return std::strcmp(env, "MIN_CTAS") == 0         ? 7
         : std::strcmp(env, "MAX_CTAS") == 0         ? 8
         : std::strcmp(env, "NVLS_NCHANNELS") == 0   ? 9
         : std::strcmp(env, "CGA_CLUSTER_SIZE") == 0 ? 10
         : deft;
  });
  EXPECT_EQ(7, ncclParamMinCTAs());
  EXPECT_EQ(8, ncclParamMaxCTAs());
  EXPECT_EQ(9, ncclParamNvlsChannels());
  EXPECT_EQ(10, ncclParamCGAClusterSize());
  EXPECT_EQ(ROCM_VERSION >= 71200 ? 1 : 0, ncclParamGraphRegister());
}

TEST_F(TaskPrepMicrotest, TaskPrepare_EligibleColls_TunesEachInQueueOrderAndForwardsSimInfo) {
  // Production routes every Broadcast to bcastQueue, which ncclTaskPreTuning drains after the generic one.
  const std::vector<ncclFunc_t> kEligible = {ncclFuncAllReduce, ncclFuncReduceScatter,
                                             ncclFuncBroadcast};
  TaskPrepScene scene;
  TaskPrep_TuningSpy spy;
  scene.EnqueueGeneric(scene.NewColl(ncclFuncAllReduce));
  scene.EnqueueGeneric(scene.NewColl(ncclFuncReduceScatter));
  scene.EnqueueBcast(scene.NewColl(ncclFuncBroadcast));

  // A non-null sim returns out of ncclTaskPostTuning early, so the classified queues survive to be read.
  ncclSimInfo_t sim = PoisonedSimInfo();
  ASSERT_EQ(ncclSuccess, ncclTaskPrepare(scene.comm(), &sim));

  EXPECT_EQ(kEligible, spy.funcs());
  std::vector<struct ncclTaskTuningInfo*> legacy =
    QueueTasks(&scene.comm()->classifiedTaskQueues.legacyTaskQueue);
  ASSERT_EQ(kEligible.size(), legacy.size());
  for (size_t i = 0; i < legacy.size(); ++i) {
    EXPECT_EQ(kEligible[i], legacy[i]->tuningIn.func) << "legacy index " << i;
    EXPECT_TRUE(CarriesTuningEstimate(legacy[i]->tuningOut));
  }
  EXPECT_FLOAT_EQ(kEligible.size() * kTunedTimeUs, sim.estimatedTime);
}

TEST_F(TaskPrepMicrotest, TaskPrepare_NonCollKinds_SkipTuningAndCarryNoEstimate) {
  TaskPrepScene scene;
  TaskPrep_TuningSpy spy;
  scene.EnqueueGeneric(scene.NewColl(ncclFuncAllReduce));
  scene.EnqueueGeneric(scene.NewSendRecv(ncclFuncSend, 1));
  scene.EnqueueGeneric(scene.NewRma(ncclFuncPutSignal));
  scene.EnqueueGeneric(scene.NewAllGatherV());

  ncclSimInfo_t sim = PoisonedSimInfo();
  ASSERT_EQ(ncclSuccess, ncclTaskPrepare(scene.comm(), &sim));

  EXPECT_EQ(std::vector<ncclFunc_t>{ncclFuncAllReduce}, spy.funcs());
  struct ncclClassifiedTaskQueues* ctq = &scene.comm()->classifiedTaskQueues;
  const struct {
    TaskTuningInfoQueue* queue;
    ncclTaskKind kind;
  } kRouted[] = {{&ctq->p2pTaskQueue, ncclTaskKindSendRecv},
                 {&ctq->rmaTaskQueue, ncclTaskKindRma},
                 {&ctq->allgathervTaskQueue, ncclTaskKindAllGatherV}};
  for (const auto& routed : kRouted) {
    std::vector<struct ncclTaskTuningInfo*> tasks = QueueTasks(routed.queue);
    ASSERT_EQ(1u, tasks.size());
    EXPECT_EQ(routed.kind, tasks[0]->raw->kind);
    EXPECT_TRUE(CarriesNoTuningEstimate(tasks[0]->tuningOut));
  }
  ASSERT_EQ(1u, QueueTasks(&ctq->legacyTaskQueue).size());
  EXPECT_TRUE(CarriesTuningEstimate(QueueTasks(&ctq->legacyTaskQueue)[0]->tuningOut));
  // postTuningSimulation never accumulates allgathervTaskQueue, so only the p2p and rma entries land.
  EXPECT_FLOAT_EQ(kTunedTimeUs + 2 * NCCL_TUNING_IGNORE, sim.estimatedTime);
}

TEST_F(TaskPrepMicrotest, TaskPrepare_TuningExcludedCollFuncs_SkipTuningAndCarryNoEstimate) {
  TaskPrepScene scene;
  TaskPrep_TuningSpy spy;
  scene.EnqueueGeneric(scene.NewColl(ncclFuncAllReduce));
  scene.EnqueueGeneric(scene.NewColl(ncclFuncAlltoAll));
  scene.EnqueueGeneric(scene.NewColl(ncclFuncScatter));
  scene.EnqueueGeneric(scene.NewColl(ncclFuncGather));
  // Harness-only pair: no collectives.cc entry point builds a coll raw whose func is ncclFuncAllGatherV.
  scene.EnqueueGeneric(scene.NewColl(ncclFuncAllGatherV));

  ncclSimInfo_t sim = PoisonedSimInfo();
  ASSERT_EQ(ncclSuccess, ncclTaskPrepare(scene.comm(), &sim));

  EXPECT_EQ(std::vector<ncclFunc_t>{ncclFuncAllReduce}, spy.funcs());
  struct ncclClassifiedTaskQueues* ctq = &scene.comm()->classifiedTaskQueues;
  std::vector<struct ncclTaskTuningInfo*> legacy = QueueTasks(&ctq->legacyTaskQueue);
  ASSERT_EQ(2u, legacy.size());
  EXPECT_TRUE(CarriesTuningEstimate(legacy[0]->tuningOut));
  EXPECT_EQ(ncclFuncAllGatherV, legacy[1]->tuningIn.func);
  EXPECT_TRUE(CarriesNoTuningEstimate(legacy[1]->tuningOut));
  EXPECT_EQ(kLoweredP2pTasks, QueueTasks(&ctq->p2pTaskQueue).size());
  EXPECT_FLOAT_EQ(kTunedTimeUs + (1 + kLoweredP2pTasks) * NCCL_TUNING_IGNORE, sim.estimatedTime);
}

TEST_F(TaskPrepMicrotest, TaskPrepare_PreTuningFails_PropagatesWithoutTuningOrPostTuning) {
  TaskPrepScene scene;
  TaskPrep_TuningSpy spy;
  ScopedHook findWindow(g_devrFindWindow,
                        [](struct ncclComm*, void const*, struct ncclDevrWindow**) {
                          return ncclInternalError;
                        });
  scene.EnqueueGeneric(scene.NewColl(ncclFuncAllReduce));

  ncclSimInfo_t sim = PoisonedSimInfo();
  EXPECT_EQ(ncclInternalError, ncclTaskPrepare(scene.comm(), &sim));

  EXPECT_TRUE(spy.funcs().empty());
  EXPECT_FLOAT_EQ(kUnwrittenEstimate, sim.estimatedTime);
}

TEST_F(TaskPrepMicrotest, TaskPrepare_TuningComputeFails_PropagatesAtTheFirstFailure) {
  TaskPrepScene scene;
  TaskPrep_TuningSpy spy(ncclSystemError);
  scene.EnqueueGeneric(scene.NewColl(ncclFuncAllReduce));
  scene.EnqueueGeneric(scene.NewColl(ncclFuncReduceScatter));

  ncclSimInfo_t sim = PoisonedSimInfo();
  EXPECT_EQ(ncclSystemError, ncclTaskPrepare(scene.comm(), &sim));

  EXPECT_EQ(std::vector<ncclFunc_t>{ncclFuncAllReduce}, spy.funcs());
  EXPECT_TRUE(QueueTasks(&scene.comm()->classifiedTaskQueues.legacyTaskQueue).empty());
  EXPECT_FLOAT_EQ(kUnwrittenEstimate, sim.estimatedTime);
}

TEST_F(TaskPrepMicrotest, TaskPrepare_ClassificationFails_PropagatesWithoutPostTuning) {
  TaskPrepScene scene;
  bool tuningStageDone = false;
  ScopedHook tuning(g_tuningCompute,
                    [&tuningStageDone](struct ncclTuningInput_t*, struct ncclTuningResult_t*) {
                      tuningStageDone = true;
                      return ncclSuccess;
                    });
  ScopedHook regValid(g_regLocalIsValid,
                      [&tuningStageDone](struct ncclReg*, bool* isValid) -> ncclResult_t {
                        if (tuningStageDone) {
                          return ncclInternalError;
                        }
                        *isValid = false;
                        return ncclSuccess;
                      });
  scene.EnqueueGeneric(scene.NewColl(ncclFuncAllReduce));
  scene.EnqueueGeneric(scene.NewColl(ncclFuncAlltoAll));

  ncclSimInfo_t sim = PoisonedSimInfo();
  EXPECT_EQ(ncclInternalError, ncclTaskPrepare(scene.comm(), &sim));

  EXPECT_EQ(1, tuning.calls);
  EXPECT_FLOAT_EQ(kUnwrittenEstimate, sim.estimatedTime);
}

TEST_F(TaskPrepMicrotest, TaskPrepare_PostTuningFails_Propagates) {
  TaskPrepScene scene;
  TaskPrep_TuningSpy spy;
  scene.comm()->hostRmaSupport = false;
  scene.EnqueueGeneric(scene.NewRma(ncclFuncPutSignal));

  EXPECT_EQ(ncclInvalidArgument, ncclTaskPrepare(scene.comm(), nullptr));

  EXPECT_TRUE(spy.funcs().empty());
}

// A leaked value here silently clears the version gate at task_posttuning.cc:336-342 for the next test.
TEST_F(TaskPrepMicrotest, CudaDriverVersionCacheResets) {
  EXPECT_EQ(kDefaultCudaDriverVersion, ncclCudaDriverVersionCache);
  ncclCudaDriverVersionCache = 12050;
  ResetNcclStubs();
  EXPECT_EQ(kDefaultCudaDriverVersion, ncclCudaDriverVersionCache);
}

}  // namespace
