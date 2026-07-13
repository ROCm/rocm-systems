/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstddef>

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>

namespace {
constexpr int kExpectedValue = 0x1234;

__global__ void WriteValueKernel(int* output, int value) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *output = value;
  }
}
}  // namespace

HIP_TEST_CASE(Contract_CaptureToGraph_BeginCaptureIntoGraph_ProducesSameGraph) {
  hipStream_t stream = nullptr;
  hipGraph_t graph = nullptr;
  int* device_value = nullptr;

  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipMalloc(&device_value, sizeof(*device_value)));
  HIP_CHECK(hipMemset(device_value, 0, sizeof(*device_value)));

  // Capturing into a caller-provided graph must end capture producing that same
  // graph handle (not a freshly allocated one), and the captured work must be
  // present as nodes.
  HIP_CHECK(hipStreamBeginCaptureToGraph(stream, graph, nullptr, nullptr, 0,
                                         hipStreamCaptureModeGlobal));
  HIP_CHECK(hipMemsetAsync(device_value, 0, sizeof(*device_value), stream));
  hipLaunchKernelGGL(WriteValueKernel, dim3(1), dim3(1), 0, stream, device_value, kExpectedValue);

  hipGraph_t captured = nullptr;
  HIP_CHECK(hipStreamEndCapture(stream, &captured));
  REQUIRE(captured == graph);

  size_t node_count = 0;
  HIP_CHECK(hipGraphGetNodes(graph, nullptr, &node_count));
  REQUIRE(node_count >= 2);

  HIP_CHECK(hipFree(device_value));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
}

HIP_TEST_CASE(Contract_CaptureToGraph_BeginCaptureIntoGraph_LaunchWritesExpectedValue) {
  hipStream_t stream = nullptr;
  hipGraph_t graph = nullptr;
  hipGraphExec_t graph_exec = nullptr;
  int* device_value = nullptr;

  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipMalloc(&device_value, sizeof(*device_value)));
  HIP_CHECK(hipMemset(device_value, 0, sizeof(*device_value)));

  HIP_CHECK(hipStreamBeginCaptureToGraph(stream, graph, nullptr, nullptr, 0,
                                         hipStreamCaptureModeGlobal));
  hipLaunchKernelGGL(WriteValueKernel, dim3(1), dim3(1), 0, stream, device_value, kExpectedValue);
  hipGraph_t captured = nullptr;
  HIP_CHECK(hipStreamEndCapture(stream, &captured));
  REQUIRE(captured == graph);

  // The graph captured into the caller graph must instantiate, launch, and
  // produce the kernel's write.
  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graph_exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  int result = 0;
  HIP_CHECK(hipMemcpy(&result, device_value, sizeof(result), hipMemcpyDeviceToHost));
  REQUIRE(result == kExpectedValue);

  HIP_CHECK(hipGraphExecDestroy(graph_exec));
  HIP_CHECK(hipFree(device_value));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
}

HIP_TEST_CASE(Contract_CaptureToGraph_UpdateCaptureDependencies_AddsToDependencySet) {
  hipStream_t stream = nullptr;
  hipGraph_t graph = nullptr;
  int* device_value = nullptr;

  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipMalloc(&device_value, sizeof(*device_value)));
  HIP_CHECK(hipMemset(device_value, 0, sizeof(*device_value)));

  HIP_CHECK(hipStreamBeginCaptureToGraph(stream, graph, nullptr, nullptr, 0,
                                         hipStreamCaptureModeGlobal));
  HIP_CHECK(hipMemsetAsync(device_value, 0, sizeof(*device_value), stream));

  // Query the current capture dependency set mid-capture and feed it back
  // through the add-dependencies update. Adding the existing dependencies must
  // be accepted while capture stays active.
  hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
  unsigned long long capture_id = 0;
  hipGraph_t capture_graph = nullptr;
  const hipGraphNode_t* dependencies = nullptr;
  size_t num_dependencies = 0;
  HIP_CHECK(hipStreamGetCaptureInfo_v2(stream, &capture_status, &capture_id, &capture_graph,
                                       &dependencies, &num_dependencies));
  REQUIRE(capture_status == hipStreamCaptureStatusActive);
  REQUIRE(capture_graph == graph);

  HIP_CHECK(hipStreamUpdateCaptureDependencies(
      stream, const_cast<hipGraphNode_t*>(dependencies), num_dependencies,
      hipStreamAddCaptureDependencies));

  hipLaunchKernelGGL(WriteValueKernel, dim3(1), dim3(1), 0, stream, device_value, kExpectedValue);

  // End capture before asserting so the stream is never left in capture on an
  // assertion failure.
  hipGraph_t captured = nullptr;
  HIP_CHECK(hipStreamEndCapture(stream, &captured));
  REQUIRE(captured == graph);

  size_t node_count = 0;
  HIP_CHECK(hipGraphGetNodes(graph, nullptr, &node_count));
  REQUIRE(node_count >= 2);

  HIP_CHECK(hipFree(device_value));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
}
