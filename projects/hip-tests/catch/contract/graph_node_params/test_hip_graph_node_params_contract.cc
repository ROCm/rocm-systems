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
#include <contract_cleanup.hh>

namespace {
constexpr size_t kByteCount = 64;
constexpr size_t kGraphNodeCount = 2;

hipMemsetParams MakeMemsetParams(void* device_ptr, unsigned int value) {
  hipMemsetParams params{};
  params.dst = device_ptr;
  params.value = value;
  params.pitch = kByteCount;
  params.elementSize = sizeof(uint8_t);
  params.width = kByteCount;
  params.height = 1;
  return params;
}
}  // namespace

HIP_TEST_CASE(Contract_GraphNodeParams_MemsetNode_SetThenGet_RoundTripsParams) {
  hip::contract::ContractCleanup cleanup;
  void* initial_ptr = nullptr;
  void* updated_ptr = nullptr;
  hipGraph_t graph = nullptr;
  hipGraphNode_t node = nullptr;

  HIP_CHECK(hipMalloc(&initial_ptr, kByteCount));
  cleanup.Add([&] { (void)hipFree(initial_ptr); });
  HIP_CHECK(hipMalloc(&updated_ptr, kByteCount));
  cleanup.Add([&] { (void)hipFree(updated_ptr); });
  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([&] { (void)hipGraphDestroy(graph); });

  auto initial_params = MakeMemsetParams(initial_ptr, 0x11);
  auto updated_params = MakeMemsetParams(updated_ptr, 0x22);
  HIP_CHECK(hipGraphAddMemsetNode(&node, graph, nullptr, 0, &initial_params));
  HIP_CHECK(hipGraphMemsetNodeSetParams(node, &updated_params));

  hipMemsetParams returned_params{};
  HIP_CHECK(hipGraphMemsetNodeGetParams(node, &returned_params));

  REQUIRE(returned_params.dst == updated_params.dst);
  REQUIRE(returned_params.value == updated_params.value);
  REQUIRE(returned_params.pitch == updated_params.pitch);
  REQUIRE(returned_params.elementSize == updated_params.elementSize);
  REQUIRE(returned_params.width == updated_params.width);
  REQUIRE(returned_params.height == updated_params.height);
}

HIP_TEST_CASE(Contract_GraphNodeParams_MemcpyNode_GetParams_ReflectsAddedNode) {
  hip::contract::ContractCleanup cleanup;
  std::array<uint8_t, kByteCount> host{};
  void* device_ptr = nullptr;
  hipGraph_t graph = nullptr;
  hipGraphNode_t node = nullptr;

  HIP_CHECK(hipMalloc(&device_ptr, host.size()));
  cleanup.Add([&] { (void)hipFree(device_ptr); });
  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([&] { (void)hipGraphDestroy(graph); });
  HIP_CHECK(hipGraphAddMemcpyNode1D(&node, graph, nullptr, 0, device_ptr, host.data(), host.size(),
                                    hipMemcpyHostToDevice));

  hipMemcpy3DParms params{};
  HIP_CHECK(hipGraphMemcpyNodeGetParams(node, &params));

  REQUIRE(params.srcPtr.ptr == host.data());
  REQUIRE(params.dstPtr.ptr == device_ptr);
  REQUIRE(params.extent.width == host.size());
  REQUIRE(params.extent.height == 1);
  REQUIRE(params.extent.depth == 1);
  REQUIRE(params.kind == hipMemcpyHostToDevice);
}

HIP_TEST_CASE(Contract_GraphNodeParams_MemcpyNode_SetThenGet_RoundTripsParams) {
  hip::contract::ContractCleanup cleanup;
  std::array<uint8_t, kByteCount> initial_host{};
  std::array<uint8_t, kByteCount> updated_host{};
  void* initial_device_ptr = nullptr;
  void* updated_device_ptr = nullptr;
  hipGraph_t graph = nullptr;
  hipGraphNode_t node = nullptr;

  HIP_CHECK(hipMalloc(&initial_device_ptr, initial_host.size()));
  cleanup.Add([&] { (void)hipFree(initial_device_ptr); });
  HIP_CHECK(hipMalloc(&updated_device_ptr, updated_host.size()));
  cleanup.Add([&] { (void)hipFree(updated_device_ptr); });
  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([&] { (void)hipGraphDestroy(graph); });
  HIP_CHECK(hipGraphAddMemcpyNode1D(&node, graph, nullptr, 0, initial_device_ptr,
                                    initial_host.data(), initial_host.size(),
                                    hipMemcpyHostToDevice));

  hipMemcpy3DParms updated_params{};
  updated_params.srcPtr = make_hipPitchedPtr(updated_host.data(), updated_host.size(),
                                             updated_host.size(), 1);
  updated_params.dstPtr = make_hipPitchedPtr(updated_device_ptr, updated_host.size(),
                                             updated_host.size(), 1);
  updated_params.extent = make_hipExtent(updated_host.size(), 1, 1);
  updated_params.kind = hipMemcpyHostToDevice;
  HIP_CHECK(hipGraphMemcpyNodeSetParams(node, &updated_params));

  hipMemcpy3DParms returned_params{};
  HIP_CHECK(hipGraphMemcpyNodeGetParams(node, &returned_params));

  REQUIRE(returned_params.srcPtr.ptr == updated_host.data());
  REQUIRE(returned_params.dstPtr.ptr == updated_device_ptr);
  REQUIRE(returned_params.extent.width == updated_host.size());
  REQUIRE(returned_params.extent.height == 1);
  REQUIRE(returned_params.extent.depth == 1);
  REQUIRE(returned_params.kind == hipMemcpyHostToDevice);
}

HIP_TEST_CASE(Contract_GraphNodeParams_MemsetNodeGetParams_NullArgs_AreRejected) {
  hip::contract::ContractCleanup cleanup;
  void* device_ptr = nullptr;
  hipGraph_t graph = nullptr;
  hipGraphNode_t node = nullptr;
  hipMemsetParams returned_params{};

  HIP_CHECK(hipMalloc(&device_ptr, kByteCount));
  cleanup.Add([&] { (void)hipFree(device_ptr); });
  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([&] { (void)hipGraphDestroy(graph); });
  auto params = MakeMemsetParams(device_ptr, 0x33);
  HIP_CHECK(hipGraphAddMemsetNode(&node, graph, nullptr, 0, &params));

  REQUIRE(hipGraphMemsetNodeGetParams(nullptr, &returned_params) != hipSuccess);
  REQUIRE(hipGraphMemsetNodeGetParams(node, nullptr) != hipSuccess);
}

HIP_TEST_CASE(Contract_GraphNodeParams_MemcpyNodeGetParams_NullArgs_AreRejected) {
  hip::contract::ContractCleanup cleanup;
  std::array<uint8_t, kByteCount> host{};
  void* device_ptr = nullptr;
  hipGraph_t graph = nullptr;
  hipGraphNode_t node = nullptr;
  hipMemcpy3DParms returned_params{};

  HIP_CHECK(hipMalloc(&device_ptr, host.size()));
  cleanup.Add([&] { (void)hipFree(device_ptr); });
  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([&] { (void)hipGraphDestroy(graph); });
  HIP_CHECK(hipGraphAddMemcpyNode1D(&node, graph, nullptr, 0, device_ptr, host.data(), host.size(),
                                    hipMemcpyHostToDevice));

  REQUIRE(hipGraphMemcpyNodeGetParams(nullptr, &returned_params) != hipSuccess);
  REQUIRE(hipGraphMemcpyNodeGetParams(node, nullptr) != hipSuccess);
}

HIP_TEST_CASE(Contract_GraphNodeParams_EventRecordAndWaitNode_GetEvent_ReturnsBoundEvent) {
  hip::contract::ContractCleanup cleanup;
  hipGraph_t graph = nullptr;
  hipEvent_t event = nullptr;
  hipGraphNode_t record_node = nullptr;
  hipGraphNode_t wait_node = nullptr;

  HIP_CHECK(hipEventCreate(&event));
  cleanup.Add([&] { (void)hipEventDestroy(event); });
  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([&] { (void)hipGraphDestroy(graph); });
  HIP_CHECK(hipGraphAddEventRecordNode(&record_node, graph, nullptr, 0, event));
  HIP_CHECK(hipGraphAddEventWaitNode(&wait_node, graph, &record_node, 1, event));

  hipEvent_t returned_record_event = nullptr;
  hipEvent_t returned_wait_event = nullptr;
  HIP_CHECK(hipGraphEventRecordNodeGetEvent(record_node, &returned_record_event));
  HIP_CHECK(hipGraphEventWaitNodeGetEvent(wait_node, &returned_wait_event));

  REQUIRE(returned_record_event == event);
  REQUIRE(returned_wait_event == event);
}

HIP_TEST_CASE(Contract_GraphNodeParams_DestroyNode_RemovesNodeFromGraph) {
  hip::contract::ContractCleanup cleanup;
  hipGraph_t graph = nullptr;
  hipGraphNode_t first_node = nullptr;
  hipGraphNode_t second_node = nullptr;
  size_t node_count = 0;

  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([&] { (void)hipGraphDestroy(graph); });
  HIP_CHECK(hipGraphAddEmptyNode(&first_node, graph, nullptr, 0));
  HIP_CHECK(hipGraphAddEmptyNode(&second_node, graph, nullptr, 0));

  HIP_CHECK(hipGraphGetNodes(graph, nullptr, &node_count));
  REQUIRE(node_count == kGraphNodeCount);

  // Destroying a node is the behavior under test, not resource teardown, so it
  // stays inline rather than moving into the cleanup guard.
  HIP_CHECK(hipGraphDestroyNode(first_node));

  HIP_CHECK(hipGraphGetNodes(graph, nullptr, &node_count));
  REQUIRE(node_count == kGraphNodeCount - 1);
}
