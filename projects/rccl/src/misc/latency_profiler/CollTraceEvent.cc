/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <thread>
#include "latency_profiler/CollTraceEvent.h"
#include "param.h"

NCCL_PARAM(ColltraceCheckIntervalMs, "COLLTRACE_CHECK_INTERVAL_MS", 10);

namespace latency_profiler {

ncclResult_t CudaWaitEvent::waitEventFinish() {
  // async polling case, query cuda whether event is ready every
  // NCCL_COLLTRACE_CHECK_INTERVAL_MS ms
  auto res = cudaEventQuery(event_.get());
  while (res != cudaSuccess) {
    if (res != cudaErrorNotReady) {
      CUDACHECK(res);
    }
    std::this_thread::sleep_for(
        std::chrono::milliseconds(ncclParamColltraceCheckIntervalMs()));
    res = cudaEventQuery(event_.get());
  }
  return ncclSuccess;
}

std::shared_ptr<float> CudaWaitEvent::getElapsedTimeSinceEvent(
    CudaWaitEvent* start) {
  float elapsedTime;
  auto res =
      cudaEventElapsedTime(&elapsedTime, start->event_.get(), event_.get());
  if (res != cudaSuccess) {
    WARN("get elapsed time failed error: %s", cudaGetErrorString(res));
    return nullptr;
  }
  return std::make_shared<float>(elapsedTime);
}

} // namespace latency_profiler
