/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <chrono>

#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>

/**
 * @addtogroup hipGraphLaunch hipGraphLaunch
 * @{
 * @ingroup GraphTest
 */

namespace {
// Generous bound: a submitted graph of this size finishes in milliseconds.
constexpr auto kWaitLimit = std::chrono::seconds(10);

__global__ void IncrementKernel(int* counter) { atomicAdd(counter, 1); }

// The last node writes |value| to host-visible memory the host can poll.
__global__ void SignalHostKernel(volatile int* flag, int value) {
  *flag = value;
  __threadfence_system();
}

// Captures a linear graph of |nodes| small kernels followed by a host signal.
hipGraphExec_t CaptureGraph(hipStream_t stream, int nodes, int* counter, int* flag_dev,
                            int value) {
  hipGraph_t graph;
  hipGraphExec_t graph_exec;
  HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
  for (int i = 0; i < nodes; ++i) {
    IncrementKernel<<<1, 1, 0, stream>>>(counter);
  }
  SignalHostKernel<<<1, 1, 0, stream>>>(flag_dev, value);
  HIP_CHECK(hipStreamEndCapture(stream, &graph));
  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphDestroy(graph));
  return graph_exec;
}

// Launches the graph and waits for its last node by polling host memory only, with no
// synchronize, query or event call in between. Returns whether the graph finished.
bool FinishesWithoutSync(hipStream_t stream, hipGraphExec_t graph_exec,
                         volatile int* flag_host, int value) {
  HIP_CHECK(hipGraphLaunch(graph_exec, stream));
  const auto deadline = std::chrono::steady_clock::now() + kWaitLimit;
  while (*flag_host != value && std::chrono::steady_clock::now() < deadline) {
  }
  return *flag_host == value;
}
}  // namespace

/**
 * Test Description
 * ------------------------
 *    - A launched graph must run to completion without the application synchronizing,
 *      querying or recording events afterwards. Launches graphs of several sizes (a single
 *      node, fewer nodes than a submission batch, and several batches' worth), then polls
 *      a host-memory flag written by the graph's last node. Regression test for
 *      ROCm/TheRock#8786: without direct dispatch, the graph's last batch was not
 *      submitted until a later synchronize or query.
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphLaunchSubmission.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipGraphLaunch_CompletesWithoutSync) {
  const int nodes = GENERATE(1, 100, 1000);
  CAPTURE(nodes);

  hipStream_t stream;
  HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  int* counter;
  HIP_CHECK(hipMalloc(&counter, sizeof(int)));
  // Reset on the graph's stream: a non-blocking stream is not ordered with hipMemset's
  // null stream.
  HIP_CHECK(hipMemsetAsync(counter, 0, sizeof(int), stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  int* flag_host;
  int* flag_dev;
  HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&flag_host), sizeof(int),
                          hipHostMallocCoherent | hipHostMallocMapped));
  HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&flag_dev), flag_host, 0));
  *flag_host = 0;

  hipGraphExec_t graph_exec = CaptureGraph(stream, nodes, counter, flag_dev, 1);
  // Launch twice: the first launch also covers graph setup, the second the steady state.
  for (int launch = 1; launch <= 2; ++launch) {
    CAPTURE(launch);
    *flag_host = 0;
    REQUIRE(FinishesWithoutSync(stream, graph_exec, flag_host, 1));
    HIP_CHECK(hipStreamSynchronize(stream));
  }

  int count = 0;
  HIP_CHECK(hipMemcpy(&count, counter, sizeof(int), hipMemcpyDeviceToHost));
  CAPTURE(count);
  REQUIRE(count == 2 * nodes);

  HIP_CHECK(hipGraphExecDestroy(graph_exec));
  HIP_CHECK(hipHostFree(flag_host));
  HIP_CHECK(hipFree(counter));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * End doxygen group GraphTest.
 * @}
 */
