/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include "CollTraceEvent.h"
#include "comm.h"
#include "CollTraceUtils.h"

namespace latency_profiler {
class CollTraceError : public std::runtime_error {
 public:
  explicit CollTraceError(const std::string& what) : std::runtime_error(what) {}
};

ncclResult_t collTraceInit(ncclComm* comm);

ncclResult_t collTraceDestroy(ncclComm* comm);

std::unique_ptr<CollTraceEvent> collTraceAquireEventBaseline(
    ncclKernelPlan* plan,
    cudaStream_t stream);

ncclResult_t collTraceRecordStartEvent(
    ncclComm* comm,
    cudaStream_t launchStream,
    CollTraceEvent* event);

ncclResult_t collTraceRecordEndEvent(
    ncclComm* comm,
    ncclKernelPlan* plan,
    cudaStream_t launchStream,
    std::unique_ptr<CollTraceEvent> event);

CollTraceInfo parseCollInfoFromCollTask(const ncclTaskColl& collTask);
} // namespace latency_profiler
