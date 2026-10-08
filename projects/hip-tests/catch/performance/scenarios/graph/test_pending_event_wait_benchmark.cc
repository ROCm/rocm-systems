/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

#include <algorithm>
#include <array>
#include <vector>

namespace {

__global__ void PendingWaitIncrement(unsigned* value) { ++*value; }

enum class WaitMode { alone, pending_wait, ready_wait };

const char* ModeName(WaitMode mode) {
  switch (mode) {
    case WaitMode::alone:
      return "alone";
    case WaitMode::pending_wait:
      return "pending_wait";
    case WaitMode::ready_wait:
      return "ready_wait";
  }
  return "unknown";
}

double Median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const size_t middle = values.size() / 2;
  return values.size() % 2 == 0 ? (values[middle - 1] + values[middle]) / 2.0
                                : values[middle];
}

}  // namespace

// Measures how a pending cross-stream event wait changes the producer's GPU
// runtime. The waiter is submitted while the producer graph is still running;
// a ready wait provides a control case. The event is timed on the producer.
HIP_TEST_CASE(Performance_PendingEventWait) {
  constexpr int kKernelCount = 2048;
  hipStream_t producer = nullptr;
  hipStream_t waiter = nullptr;
  hipEvent_t begin = nullptr;
  hipEvent_t end = nullptr;
  unsigned* value = nullptr;
  HIP_CHECK(hipStreamCreateWithFlags(&producer, hipStreamNonBlocking));
  HIP_CHECK(hipStreamCreateWithFlags(&waiter, hipStreamNonBlocking));
  HIP_CHECK(hipEventCreate(&begin));
  HIP_CHECK(hipEventCreate(&end));
  HIP_CHECK(hipMalloc(&value, sizeof(unsigned)));

  hipGraph_t graph = nullptr;
  hipGraphExec_t exec = nullptr;
  HIP_CHECK(hipStreamBeginCapture(producer, hipStreamCaptureModeGlobal));
  for (int kernel = 0; kernel < kKernelCount; ++kernel) {
    PendingWaitIncrement<<<1, 1, 0, producer>>>(value);
  }
  HIP_CHECK(hipStreamEndCapture(producer, &graph));
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  std::array<std::vector<double>, 3> elapsed_us;
  std::array<int, 3> attempted{};
  const std::array<WaitMode, 3> modes{
      WaitMode::alone, WaitMode::pending_wait, WaitMode::ready_wait};
  // Rotate modes so an arm cannot win only because it ran later in a block.
  for (int block = -1; block < 5; ++block) {
    for (int position = 0; position < 3; ++position) {
      const int mode_index = (position + block + 3) % 3;
      const auto mode = modes[mode_index];
      for (int trial = 0; trial < 8; ++trial) {
        HIP_CHECK(hipMemsetAsync(value, 0, sizeof(unsigned), producer));
        HIP_CHECK(hipStreamSynchronize(producer));
        HIP_CHECK(hipEventRecord(begin, producer));
        HIP_CHECK(hipGraphLaunch(exec, producer));
        HIP_CHECK(hipEventRecord(end, producer));
        bool pending = false;
        if (mode == WaitMode::pending_wait) {
          const auto status = hipEventQuery(end);
          REQUIRE((status == hipSuccess || status == hipErrorNotReady));
          pending = status == hipErrorNotReady;
          HIP_CHECK(hipStreamWaitEvent(waiter, end, 0));
        }
        HIP_CHECK(hipEventSynchronize(end));
        if (mode == WaitMode::ready_wait) {
          HIP_CHECK(hipStreamWaitEvent(waiter, end, 0));
        }
        HIP_CHECK(hipStreamSynchronize(waiter));
        float milliseconds = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&milliseconds, begin, end));
        unsigned actual = 0;
        HIP_CHECK(hipMemcpy(&actual, value, sizeof(actual), hipMemcpyDeviceToHost));
        REQUIRE(actual == kKernelCount);
        if (block >= 0) {
          ++attempted[mode_index];
          if (mode != WaitMode::pending_wait || pending) {
            elapsed_us[mode_index].push_back(static_cast<double>(milliseconds) * 1000.0);
          }
        }
      }
    }
  }

  CONSOLE_PRINT("pending_event_wait_benchmark,mode,median_producer_gpu_us,accepted_trials,attempted_trials");
  for (size_t index = 0; index < modes.size(); ++index) {
    if (elapsed_us[index].empty()) {
      CONSOLE_PRINT("pending_event_wait_benchmark,%s,unavailable,0,%d",
                    ModeName(modes[index]), attempted[index]);
    } else {
      CONSOLE_PRINT("pending_event_wait_benchmark,%s,%.3f,%zu,%d", ModeName(modes[index]),
                    Median(elapsed_us[index]), elapsed_us[index].size(), attempted[index]);
    }
  }

  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(value));
  HIP_CHECK(hipEventDestroy(begin));
  HIP_CHECK(hipEventDestroy(end));
  HIP_CHECK(hipStreamDestroy(producer));
  HIP_CHECK(hipStreamDestroy(waiter));
}
