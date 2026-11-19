/*
 * Copyright Advanced Micro Devices, Inc.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <resource_guards.hh>

namespace {
constexpr int kOriginalValue = 0x1234;
constexpr int kRecapturedValue = 0x5678;

__global__ void WriteValue(int* output, int value) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *output = value;
  }
}

hipError_t LaunchWrite(hipStream_t stream, int* output, int value) {
  void* args[] = {&output, &value};
  return hipLaunchKernel(reinterpret_cast<const void*>(WriteValue), dim3(1), dim3(1), args, 0,
                         stream);
}

struct GraphOwner {
  hipGraph_t graph = nullptr;

  GraphOwner() = default;
  GraphOwner(const GraphOwner&) = delete;
  GraphOwner& operator=(const GraphOwner&) = delete;

  ~GraphOwner() {
    if (graph != nullptr) {
      (void)hipGraphDestroy(graph);
    }
  }
};

struct ExecOwner {
  hipGraphExec_t exec = nullptr;

  ExecOwner() = default;
  ExecOwner(const ExecOwner&) = delete;
  ExecOwner& operator=(const ExecOwner&) = delete;

  ~ExecOwner() {
    if (exec != nullptr) {
      (void)hipGraphExecDestroy(exec);
    }
  }
};

void CaptureOneKernel(hipStream_t stream, int* output, int value, hipGraph_t* graph) {
  HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
  HIP_CHECK(LaunchWrite(stream, output, value));
  HIP_CHECK(hipStreamEndCapture(stream, graph));
}
}  // namespace

// @asserts: hipStreamBeginRecaptureToGraph - ending a recapture returns the input graph handle and keeps its node handles
HIP_TEST_CASE(
    Contract_CaptureToGraph_HipStreamBeginRecaptureToGraph_Recapture_ReturnsSameGraphAndNodes) {
  StreamGuard stream{Streams::created};
  LinearAllocGuard<int> output(LinearAllocs::hipMalloc, sizeof(int));
  GraphOwner graph;
  CaptureOneKernel(stream.stream(), output.ptr(), kOriginalValue, &graph.graph);

  hipGraphNode_t node_before = nullptr;
  size_t node_count = 1;
  HIP_CHECK(hipGraphGetNodes(graph.graph, &node_before, &node_count));
  REQUIRE(node_count == 1);

  HIP_CHECK(hipStreamBeginRecaptureToGraph(stream.stream(), hipStreamCaptureModeGlobal, graph.graph,
                                           nullptr));
  HIP_CHECK(LaunchWrite(stream.stream(), output.ptr(), kRecapturedValue));
  hipGraph_t recaptured = nullptr;
  HIP_CHECK(hipStreamEndCapture(stream.stream(), &recaptured));
  REQUIRE(recaptured == graph.graph);

  hipGraphNode_t node_after = nullptr;
  node_count = 1;
  HIP_CHECK(hipGraphGetNodes(graph.graph, &node_after, &node_count));
  REQUIRE(node_count == 1);
  REQUIRE(node_after == node_before);
}

// @asserts: hipStreamBeginRecaptureToGraph - recaptured parameters are applied to the existing node and the launch writes the new value
HIP_TEST_CASE(
    Contract_CaptureToGraph_HipStreamBeginRecaptureToGraph_Recapture_LaunchWritesNewValue) {
  StreamGuard stream{Streams::created};
  LinearAllocGuard<int> output(LinearAllocs::hipMalloc, sizeof(int));
  HIP_CHECK(hipMemset(output.ptr(), 0, sizeof(int)));
  GraphOwner graph;
  CaptureOneKernel(stream.stream(), output.ptr(), kOriginalValue, &graph.graph);

  HIP_CHECK(hipStreamBeginRecaptureToGraph(stream.stream(), hipStreamCaptureModeGlobal, graph.graph,
                                           nullptr));
  HIP_CHECK(LaunchWrite(stream.stream(), output.ptr(), kRecapturedValue));
  hipGraph_t recaptured = nullptr;
  HIP_CHECK(hipStreamEndCapture(stream.stream(), &recaptured));
  REQUIRE(recaptured == graph.graph);

  ExecOwner exec;
  HIP_CHECK(hipGraphInstantiate(&exec.exec, graph.graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(exec.exec, stream.stream()));
  HIP_CHECK(hipStreamSynchronize(stream.stream()));

  int result = 0;
  HIP_CHECK(hipMemcpy(&result, output.ptr(), sizeof(result), hipMemcpyDeviceToHost));
  REQUIRE(result == kRecapturedValue);
}

// @asserts: hipStreamBeginRecaptureToGraph - an incomplete recapture fails with hipErrorGraphRecaptureFailure, returns a null graph, and leaves the stream not capturing
HIP_TEST_CASE(Contract_CaptureToGraph_HipStreamBeginRecaptureToGraph_MissingNode_EndsCapture) {
  StreamGuard stream{Streams::created};
  LinearAllocGuard<int> output(LinearAllocs::hipMalloc, sizeof(int));
  GraphOwner graph;
  CaptureOneKernel(stream.stream(), output.ptr(), kOriginalValue, &graph.graph);

  // Recapturing no work leaves the only original node unmatched. The runtime destroys that graph.
  HIP_CHECK(hipStreamBeginRecaptureToGraph(stream.stream(), hipStreamCaptureModeGlobal, graph.graph,
                                           nullptr));
  graph.graph = nullptr;
  hipGraph_t recaptured = reinterpret_cast<hipGraph_t>(0x1);
  REQUIRE(hipStreamEndCapture(stream.stream(), &recaptured) == hipErrorGraphRecaptureFailure);
  (void)hipGetLastError();
  REQUIRE(recaptured == nullptr);

  hipStreamCaptureStatus status = hipStreamCaptureStatusActive;
  HIP_CHECK(hipStreamIsCapturing(stream.stream(), &status));
  REQUIRE(status == hipStreamCaptureStatusNone);
}
