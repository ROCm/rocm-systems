/*
 * Copyright Advanced Micro Devices, Inc.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>
#include <contract_cleanup.hh>

namespace {
constexpr int kOriginalValue = 0x1234;
constexpr int kRecapturedValue = 0x5678;

__global__ void WriteValueKernel(int* output, int value) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *output = value;
  }
}

void CaptureOneKernel(hipStream_t stream, int* output, int value, hipGraph_t* graph) {
  HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
  hipLaunchKernelGGL(WriteValueKernel, dim3(1), dim3(1), 0, stream, output, value);
  HIP_CHECK(hipStreamEndCapture(stream, graph));
}
}  // namespace

// @asserts: hipStreamBeginRecaptureToGraph - ending a recapture returns the input graph handle and keeps its node handles
HIP_TEST_CASE(Contract_CaptureToGraph_HipStreamBeginRecaptureToGraph_Recapture_ReturnsSameGraphAndNodes) {
  hip::contract::ContractCleanup cleanup;
  hipStream_t stream = nullptr;
  hipGraph_t graph = nullptr;
  int* device_value = nullptr;

  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });
  HIP_CHECK(hipMalloc(&device_value, sizeof(*device_value)));
  cleanup.Add([device_value] { (void)hipFree(device_value); });
  CaptureOneKernel(stream, device_value, kOriginalValue, &graph);
  cleanup.Add([graph] { (void)hipGraphDestroy(graph); });

  hipGraphNode_t node_before = nullptr;
  size_t node_count = 1;
  HIP_CHECK(hipGraphGetNodes(graph, &node_before, &node_count));
  REQUIRE(node_count == 1);

  HIP_CHECK(hipStreamBeginRecaptureToGraph(stream, hipStreamCaptureModeGlobal, graph, nullptr));
  hipLaunchKernelGGL(WriteValueKernel, dim3(1), dim3(1), 0, stream, device_value,
                     kRecapturedValue);
  hipGraph_t recaptured = nullptr;
  HIP_CHECK(hipStreamEndCapture(stream, &recaptured));
  REQUIRE(recaptured == graph);

  hipGraphNode_t node_after = nullptr;
  node_count = 1;
  HIP_CHECK(hipGraphGetNodes(graph, &node_after, &node_count));
  REQUIRE(node_count == 1);
  REQUIRE(node_after == node_before);
}

// @asserts: hipStreamBeginRecaptureToGraph - recaptured parameters are applied to the existing node and the launch writes the new value
HIP_TEST_CASE(Contract_CaptureToGraph_HipStreamBeginRecaptureToGraph_Recapture_LaunchWritesNewValue) {
  hip::contract::ContractCleanup cleanup;
  hipStream_t stream = nullptr;
  hipGraph_t graph = nullptr;
  hipGraphExec_t graph_exec = nullptr;
  int* device_value = nullptr;

  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });
  HIP_CHECK(hipMalloc(&device_value, sizeof(*device_value)));
  cleanup.Add([device_value] { (void)hipFree(device_value); });
  HIP_CHECK(hipMemset(device_value, 0, sizeof(*device_value)));
  CaptureOneKernel(stream, device_value, kOriginalValue, &graph);
  cleanup.Add([graph] { (void)hipGraphDestroy(graph); });

  HIP_CHECK(hipStreamBeginRecaptureToGraph(stream, hipStreamCaptureModeGlobal, graph, nullptr));
  hipLaunchKernelGGL(WriteValueKernel, dim3(1), dim3(1), 0, stream, device_value,
                     kRecapturedValue);
  hipGraph_t recaptured = nullptr;
  HIP_CHECK(hipStreamEndCapture(stream, &recaptured));
  REQUIRE(recaptured == graph);

  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
  cleanup.Add([graph_exec] { (void)hipGraphExecDestroy(graph_exec); });
  HIP_CHECK(hipGraphLaunch(graph_exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  int result = 0;
  HIP_CHECK(hipMemcpy(&result, device_value, sizeof(result), hipMemcpyDeviceToHost));
  REQUIRE(result == kRecapturedValue);
}

// @asserts: hipStreamBeginRecaptureToGraph - an incomplete recapture fails with hipErrorGraphRecaptureFailure, returns a null graph and destroys the input graph
HIP_TEST_CASE(Contract_CaptureToGraph_HipStreamBeginRecaptureToGraph_MissingNode_DestroysGraph) {
  hip::contract::ContractCleanup cleanup;
  hipStream_t stream = nullptr;
  hipGraph_t graph = nullptr;
  int* device_value = nullptr;

  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });
  HIP_CHECK(hipMalloc(&device_value, sizeof(*device_value)));
  cleanup.Add([device_value] { (void)hipFree(device_value); });
  CaptureOneKernel(stream, device_value, kOriginalValue, &graph);

  // Recapturing no work leaves the only original node unmatched.
  HIP_CHECK(hipStreamBeginRecaptureToGraph(stream, hipStreamCaptureModeGlobal, graph, nullptr));
  hipGraph_t recaptured = reinterpret_cast<hipGraph_t>(0x1);
  REQUIRE(hipStreamEndCapture(stream, &recaptured) == hipErrorGraphRecaptureFailure);
  REQUIRE(recaptured == nullptr);

  hipStreamCaptureStatus status = hipStreamCaptureStatusActive;
  HIP_CHECK(hipStreamIsCapturing(stream, &status));
  REQUIRE(status == hipStreamCaptureStatusNone);
  // The runtime already destroyed the graph, so the handle is no longer valid.
  REQUIRE(hipGraphDestroy(graph) == hipErrorInvalidValue);
}
