/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_TEST_HOST_TASK_PREP_SCENE_H_
#define RCCL_TEST_HOST_TASK_PREP_SCENE_H_

#include <gtest/gtest.h>

#include <cstddef>
#include <memory>
#include <vector>

#include "ScopedHook.h"
#include "ce_coll.h"
#include "channel.h"
#include "comm.h"
#include "enqueue.h"
#include "enqueue/task_classify.h"
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

constexpr int kRanks = 4;
constexpr size_t kCount = 100;
constexpr float kTunedTimeUs = 12.5f;
constexpr float kUnwrittenEstimate = -777.0f;
constexpr int kTunedValid = 1;

using TaskTuningInfoQueue = struct ncclIntruQueue<struct ncclTaskTuningInfo, &ncclTaskTuningInfo::next>;

// ncclComm carries channels[MAXCHANNELS] inline, so it lives on the heap; a stack instance overflows.
class TaskPrepScene {
 public:
  explicit TaskPrepScene(int nRanks = kRanks, int rank = 0)
    : comm_(new ncclComm{}), sendBuf_(kCount * nRanks), recvBuf_(kCount * nRanks) {
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

  struct ncclTaskTuningInfo* NewTuningInfo(struct ncclRawTask* raw) {
    struct ncclTaskTuningInfo* tInfo =
      ncclMemoryStackAlloc<struct ncclTaskTuningInfo>(&comm_->memScoped);
    const ncclTuningResult_t init = NCCL_TUNING_RESULT_INIT;
    tInfo->raw = raw;
    tInfo->tuningOut = init;
    return tInfo;
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
  std::vector<float> sendBuf_;
  std::vector<float> recvBuf_;
};

inline std::vector<struct ncclTaskTuningInfo*> QueueTasks(TaskTuningInfoQueue* queue) {
  std::vector<struct ncclTaskTuningInfo*> tasks;
  for (struct ncclTaskTuningInfo* t = ncclIntruQueueHead(queue); t != nullptr; t = t->next) {
    tasks.push_back(t);
  }
  return tasks;
}

inline ::testing::AssertionResult CarriesNoTuningEstimate(const struct ncclTuningResult_t& out) {
  if (out.valid != NCCL_TUNING_ENTRY_INIT_VALUE) {
    return ::testing::AssertionFailure() << "tuningOut.valid = " << out.valid;
  }
  if (out.timeUs != NCCL_TUNING_IGNORE) {
    return ::testing::AssertionFailure() << "tuningOut.timeUs = " << out.timeUs;
  }
  return ::testing::AssertionSuccess();
}

inline ::testing::AssertionResult CarriesTuningEstimate(const struct ncclTuningResult_t& out) {
  if (out.valid != kTunedValid) {
    return ::testing::AssertionFailure() << "tuningOut.valid = " << out.valid;
  }
  if (out.timeUs != kTunedTimeUs) {
    return ::testing::AssertionFailure() << "tuningOut.timeUs = " << out.timeUs;
  }
  return ::testing::AssertionSuccess();
}

inline ncclSimInfo_t PoisonedSimInfo() {
  ncclSimInfo_t sim = NCCL_SIM_INFO_INITIALIZER;
  sim.estimatedTime = kUnwrittenEstimate;
  return sim;
}

class TaskPrepFakesFixture : public ::testing::Test {
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

#endif  // RCCL_TEST_HOST_TASK_PREP_SCENE_H_
