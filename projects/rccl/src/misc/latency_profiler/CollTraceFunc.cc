/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "latency_profiler/CollTraceFunc.h"

namespace latency_profiler {

namespace {
bool enableCollTrace() {
  const char* colltraceEnable = ncclGetEnv("RCCL_LATENCY_PROFILER");
  if (colltraceEnable != NULL) {
    INFO(
        NCCL_INIT,
        "RCCL_LATENCY_PROFILER set by environment to %s.",
        colltraceEnable);
    if (strcmp(colltraceEnable, "1") == 0) {
      return true;
    }
  }
  return false;
}
} // namespace

ncclResult_t collTraceInit(ncclComm* comm) {
  if (!enableCollTrace()) {
    return ncclSuccess;
  }
  comm->ctrace = std::make_unique<CollTrace>(comm);
  return ncclSuccess;
}

ncclResult_t collTraceDestroy(ncclComm* comm) {
  if (comm->ctrace == nullptr) {
    return ncclSuccess;
  }
  comm->ctrace.reset();
  return ncclSuccess;
}

ncclResult_t collTraceRecordStartEvent(
    ncclComm* comm,
    cudaStream_t launchStream,
    CollTraceEvent* event) {
  if (comm->ctrace && event) {
    CUDACHECK(
        cudaEventRecord(event->start.get()->getCudaEvent(), launchStream));
  }
  return ncclSuccess;
}

ncclResult_t collTraceRecordEndEvent(
    ncclComm* comm,
    ncclKernelPlan* plan,
    cudaStream_t launchStream,
    std::unique_ptr<CollTraceEvent> event) {
  if (comm->ctrace && event) {
    CUDACHECK(cudaEventRecord(event->stop.get()->getCudaEvent(), launchStream));
    comm->ctrace->enqueueEvent(std::move(event));
  }
  return ncclSuccess;
}

CollTraceInfo parseCollInfoFromCollTask(const ncclTaskColl& collTask) {
  return CollTraceInfo{
      .opName = std::string{ncclFuncToString(collTask.func)},
      .dataType = std::string{ncclDatatypeToString(collTask.datatype)},
      .count = (int64_t)collTask.count,
  };
}

std::shared_ptr<CollTraceInfo> parseCollInfoFromNcclKernelPlan(
    ncclKernelPlan& plan,
    cudaStream_t stream) {
  if (plan.comm == nullptr || plan.comm->ctrace == nullptr) {
    return nullptr;
  }
  auto collTaskHead = ncclIntruQueueHead(&plan.collTaskQueue);
  if (collTaskHead == nullptr) {
    WARN("CollTrace: no coll task in this plan, this plan is empty");
    return nullptr;
  }

  CollTraceInfo collInfo = parseCollInfoFromCollTask(*collTaskHead);
  return std::make_shared<CollTraceInfo>(collInfo);
}

std::unique_ptr<CollTraceEvent> collTraceAquireEventCommon(
    ncclComm* comm,
    CollTraceEvent::EventType type,
    cudaStream_t stream) {
  if (!comm->ctrace) {
    return nullptr;
  }
  struct ncclCudaGraph graph;
  auto res = ncclCudaGetCapturingGraph(&graph, stream);
  if (res != ncclSuccess) {
    WARN("Internal error: ncclCudaGetCapturingGraph failed by %d", res);
    return nullptr;
  }
  if (graph.graph != nullptr) {
    // We are in a cuda graph, this is currently unsupported
    WARN(
        "COLLTRACE: does not support cuda graph. Collectives from comm %lx will be skipped",
        comm->commHash);
    return nullptr;
  }
  auto event = comm->ctrace->createEvent(type);
  if (!event) {
    throw CollTraceError("Event init failed");
    return nullptr; /*Event init failed*/
  }
  return event;
}

std::unique_ptr<CollTraceEvent> collTraceAquireEventBaseline(
    ncclKernelPlan* plan,
    cudaStream_t stream) {
  auto collPtr = parseCollInfoFromNcclKernelPlan(*plan, stream);
  if (collPtr == nullptr) {
    return nullptr;
  }
  auto comm = plan->comm;
  if (!comm->ctrace) {
    return nullptr;
  }

  auto event =
      collTraceAquireEventCommon(comm, CollTraceEvent::EventType::COMM, stream);
  if (event == nullptr) {
    WARN("COLLTRACE: failed to aquire event");
    return nullptr;
  }
  event->coll = *collPtr;
  return event;
}

} // namespace latency_profiler
