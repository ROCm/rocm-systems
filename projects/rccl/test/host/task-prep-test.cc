/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests; the UUT is #include'd. Hipify leaves task_prep.cc byte-identical.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <typeinfo>
#include <vector>

#include "ScopedHook.h"
#include "ce_coll.h"
#include "channel.h"
#include "fakes/ce_fakes.h"
#include "fakes/comm_fakes.h"
#include "fakes/dev_runtime_fakes.h"
#include "fakes/enqueue_fakes.h"
#include "fakes/env_fakes.h"
#include "fakes/group_fakes.h"
#include "fakes/hip_fakes.h"
#include "fakes/nccl_fakes.h"
#include "fakes/nccl_stubs.h"
#include "fakes/proxy_fakes.h"
#include "fakes/rccl_wrap_fakes.h"
#include "fakes/recorder_fakes.h"
#include "fakes/register_stubs.h"
#include "fakes/rma_fakes.h"
#include "fakes/sym_kernels_fakes.h"
#include "fakes/transport_stubs.h"
#include "fakes/tuning_fakes.h"
#include "transport.h"

#include TASK_PREP_CC_PATH

// Declared by the task_prep TUs and defined in fakes/, exactly as those TUs declare them.
int64_t ncclParamMinCTAs();
int64_t ncclParamMaxCTAs();
int64_t ncclParamNvlsChannels();
int64_t ncclParamCGAClusterSize();

namespace {

constexpr int kRanks = 4;
constexpr size_t kCount = 100;

// ncclComm carries channels[MAXCHANNELS] inline, so it lives on the heap; a stack instance overflows.
class TaskPrepScene {
 public:
  explicit TaskPrepScene(int nRanks = kRanks, int rank = 0) : comm_(new ncclComm{}) {
    comm_->nRanks = nRanks;
    comm_->rank = rank;
    comm_->nNodes = 1;
    comm_->nChannels = 1;
    comm_->config.minCTAs = NCCL_CONFIG_UNDEF_INT;
    comm_->config.maxCTAs = NCCL_CONFIG_UNDEF_INT;
    comm_->config.nvlsCTAs = NCCL_CONFIG_UNDEF_INT;
    comm_->config.cgaClusterSize = NCCL_CONFIG_UNDEF_INT;
    // new ncclComm{} zeroes graphId, which ncclCudaGraphValid reads as capturing; start not-capturing.
    comm_->planner.capturingGraph = ncclCudaGraphNone(kNoGraphUsageMode);
    ncclMemoryStackConstruct(&comm_->memScoped);
    ncclMemoryStackConstruct(&comm_->memPermanent);
    ncclMemoryPoolConstruct(&comm_->memPool_ncclRawTask);
    ncclIntruQueueConstruct(&comm_->rawTaskQueue.genericQueue);
    ncclIntruQueueConstruct(&comm_->rawTaskQueue.bcastQueue);
  }

  ~TaskPrepScene() {
    ncclMemoryStackDestruct(&comm_->memScoped);
    ncclMemoryStackDestruct(&comm_->memPermanent);
  }

  struct ncclComm* comm() { return comm_.get(); }

  // Raw tasks are pool-allocated out of the comm, matching how enqueue.cc:4304-4314 builds them.
  struct ncclRawTask* NewRaw(ncclTaskKind kind) {
    struct ncclRawTask* raw =
      ncclMemoryPoolAlloc<struct ncclRawTask>(&comm_->memPool_ncclRawTask, &comm_->memPermanent);
    raw->kind = kind;
    return raw;
  }

  struct ncclRawTask* NewColl(ncclFunc_t func, ncclDataType_t datatype = ncclFloat32) {
    struct ncclRawTask* raw = NewRaw(ncclTaskKindColl);
    raw->coll = {};
    raw->coll.func = func;
    raw->coll.sendbuff = sendBuf_.data();
    raw->coll.recvbuff = recvBuf_.data();
    raw->coll.count = kCount;
    raw->coll.datatype = datatype;
    raw->coll.collConfig.minCTAs = NCCL_CONFIG_UNDEF_INT;
    raw->coll.collConfig.maxCTAs = NCCL_CONFIG_UNDEF_INT;
    return raw;
  }

  struct ncclRawTask* NewSendRecv(ncclFunc_t func, int peer) {
    struct ncclRawTask* raw = NewRaw(ncclTaskKindSendRecv);
    raw->sendRecv = {};
    raw->sendRecv.func = func;
    raw->sendRecv.collAPI = func;
    raw->sendRecv.buff = sendBuf_.data();
    raw->sendRecv.count = kCount;
    raw->sendRecv.datatype = ncclFloat32;
    raw->sendRecv.peer = peer;
    raw->sendRecv.bytes = kCount * sizeof(float);
    return raw;
  }

  struct ncclRawTask* NewRma(ncclFunc_t func) {
    struct ncclRawTask* raw = NewRaw(ncclTaskKindRma);
    raw->rma = {};
    raw->rma.func = func;
    raw->rma.rmaOp.putSignal.localbuff = sendBuf_.data();
    raw->rma.rmaOp.putSignal.count = kCount;
    raw->rma.rmaOp.putSignal.datatype = ncclFloat32;
    return raw;
  }

  // recvbuff/counts are nRanks-long, as preTuningInitAllGatherVRaw (task_pretuning.cc:149-150) builds them.
  struct ncclRawTask* NewAllGatherV() {
    struct ncclRawTask* raw = NewRaw(ncclTaskKindAllGatherV);
    raw->allGatherV = {};
    raw->allGatherV.func = ncclFuncAllGatherV;
    raw->allGatherV.nRanks = comm_->nRanks;
    raw->allGatherV.recvbuff = ncclMemoryStackAlloc<void*>(&comm_->memScoped, comm_->nRanks);
    raw->allGatherV.counts = ncclMemoryStackAlloc<size_t>(&comm_->memScoped, comm_->nRanks);
    raw->allGatherV.maxCount = kCount;
    raw->allGatherV.datatype = ncclInt8;
    return raw;
  }

  void EnqueueGeneric(struct ncclRawTask* raw) {
    ncclIntruQueueEnqueue(&comm_->rawTaskQueue.genericQueue, raw);
  }

  void EnqueueBcast(struct ncclRawTask* raw) {
    ncclIntruQueueEnqueue(&comm_->rawTaskQueue.bcastQueue, raw);
  }

 private:
  static constexpr int kNoGraphUsageMode = 0;

  std::unique_ptr<ncclComm> comm_;
  std::vector<float> sendBuf_ = std::vector<float>(kCount * kRanks);
  std::vector<float> recvBuf_ = std::vector<float>(kCount * kRanks);
};

// Counts the tasks a tuning-info queue holds, the observable output of the pre-tuning stage.
size_t TuningQueueLength(struct ncclTaskTuningInfoQueue* tiq) {
  size_t n = 0;
  for (struct ncclTaskTuningInfo* t = ncclIntruQueueHead(&tiq->queue); t != nullptr; t = t->next) {
    ++n;
  }
  return n;
}

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

class TaskPrepMicrotest : public ::testing::Test {
 protected:
  void SetUp() override { ResetTaskPrepFakes(); }
  void TearDown() override { ResetTaskPrepFakes(); }

  static void ResetTaskPrepFakes() {
    ResetHipFakes();
    ResetNcclFakes();
    ResetNcclStubs();
    ResetCeFakes();
    ResetCommFakes();
    ResetDevRuntimeFakes();
    ResetEnqueueFakes();
    ResetGroupFakes();
    ResetProxyFakes();
    ResetRcclWrapFakes();
    ResetRecorderFakes();
    ResetRegisterStubs();
    ResetRmaFakes();
    ResetSymKernelsFakes();
    ResetTransportStubs();
    ResetTuningFakes();
    ResetEnvFakes();
  }
};

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

  EXPECT_EQ(5u, TuningQueueLength(&tiq));
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

// A leaked value here silently clears the version gate at task_posttuning.cc:336-342 for the next test.
TEST_F(TaskPrepMicrotest, CudaDriverVersionCacheResets) {
  EXPECT_EQ(kDefaultCudaDriverVersion, ncclCudaDriverVersionCache);
  ncclCudaDriverVersionCache = 12050;
  ResetNcclStubs();
  EXPECT_EQ(kDefaultCudaDriverVersion, ncclCudaDriverVersionCache);
}

}  // namespace
