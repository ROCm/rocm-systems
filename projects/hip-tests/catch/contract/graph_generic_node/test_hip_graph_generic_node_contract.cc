/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <array>
#include <cstddef>
#include <cstdint>

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>

namespace {
constexpr size_t kByteCount = 64;

hipMemsetParams MakeByteMemsetParams(void* device_ptr, unsigned int value) {
  hipMemsetParams params{};
  params.dst = device_ptr;
  params.value = value;
  params.pitch = kByteCount;
  params.elementSize = sizeof(uint8_t);
  params.width = kByteCount;
  params.height = 1;
  return params;
}

// Wraps a byte-memset in the unified generic node parameter structure.
hipGraphNodeParams MakeMemsetNodeParams(void* device_ptr, unsigned int value) {
  hipGraphNodeParams params{};
  params.type = hipGraphNodeTypeMemset;
  params.memset = MakeByteMemsetParams(device_ptr, value);
  return params;
}

// Instantiates and launches the graph once, then reads the first byte of the
// device buffer back to the host so the memset effect can be asserted.
uint8_t LaunchAndReadFirstByte(hipGraph_t graph, void* device_ptr) {
  hipGraphExec_t exec = nullptr;
  hipStream_t stream = nullptr;
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipGraphLaunch(exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  uint8_t host = 0;
  HIP_CHECK(hipMemcpy(&host, device_ptr, sizeof(host), hipMemcpyDeviceToHost));

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipGraphExecDestroy(exec));
  return host;
}
}  // namespace

HIP_TEST_CASE(Contract_GraphGenericNode_AddMemsetNode_LaunchesExpectedValue) {
  void* device_ptr = nullptr;
  hipGraph_t graph = nullptr;
  hipGraphNode_t node = nullptr;

  HIP_CHECK(hipMalloc(&device_ptr, kByteCount));
  HIP_CHECK(hipGraphCreate(&graph, 0));

  // A memset node added through the generic hipGraphAddNode entry point must
  // behave exactly like one added through the typed hipGraphAddMemsetNode
  // entry point: launching the graph writes the requested byte value.
  hipGraphNodeParams params = MakeMemsetNodeParams(device_ptr, 0x5A);
  HIP_CHECK(hipGraphAddNode(&node, graph, nullptr, 0, &params));

  // The added node reports the generic memset type it was created with.
  hipGraphNodeType type{};
  HIP_CHECK(hipGraphNodeGetType(node, &type));
  REQUIRE(type == hipGraphNodeTypeMemset);

  REQUIRE(LaunchAndReadFirstByte(graph, device_ptr) == 0x5A);

  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(device_ptr));
}

HIP_TEST_CASE(Contract_GraphGenericNode_NodeSetParams_UpdatesValueBeforeInstantiate) {
  void* device_ptr = nullptr;
  hipGraph_t graph = nullptr;
  hipGraphNode_t node = nullptr;

  HIP_CHECK(hipMalloc(&device_ptr, kByteCount));
  HIP_CHECK(hipGraphCreate(&graph, 0));

  // Create the node with one value, then re-parameterize it through the generic
  // pre-instantiation setter. The launched graph must reflect the updated value.
  hipGraphNodeParams initial = MakeMemsetNodeParams(device_ptr, 0x11);
  HIP_CHECK(hipGraphAddNode(&node, graph, nullptr, 0, &initial));

  hipGraphNodeParams updated = MakeMemsetNodeParams(device_ptr, 0x22);
  HIP_CHECK(hipGraphNodeSetParams(node, &updated));

  REQUIRE(LaunchAndReadFirstByte(graph, device_ptr) == 0x22);

  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(device_ptr));
}

HIP_TEST_CASE(Contract_GraphGenericNode_ExecNodeSetParams_UpdatesValueAfterInstantiate) {
  void* device_ptr = nullptr;
  hipGraph_t graph = nullptr;
  hipGraphNode_t node = nullptr;

  HIP_CHECK(hipMalloc(&device_ptr, kByteCount));
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphNodeParams initial = MakeMemsetNodeParams(device_ptr, 0x33);
  HIP_CHECK(hipGraphAddNode(&node, graph, nullptr, 0, &initial));

  hipGraphExec_t exec = nullptr;
  hipStream_t stream = nullptr;
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  // Re-parameterize the already-instantiated node through the generic
  // executable-graph setter. The next launch must use the updated value, so the
  // update takes effect without re-instantiating the graph.
  hipGraphNodeParams updated = MakeMemsetNodeParams(device_ptr, 0x44);
  HIP_CHECK(hipGraphExecNodeSetParams(exec, node, &updated));

  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipGraphLaunch(exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  uint8_t host = 0;
  HIP_CHECK(hipMemcpy(&host, device_ptr, sizeof(host), hipMemcpyDeviceToHost));
  REQUIRE(host == 0x44);

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(device_ptr));
}
