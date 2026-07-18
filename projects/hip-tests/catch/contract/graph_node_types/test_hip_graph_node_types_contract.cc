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
}

HIP_TEST_CASE(Contract_GraphNodeTypes_GetType_EmptyNodeReportsEmpty) {
  hip::contract::ContractCleanup cleanup;
  hipGraph_t graph = nullptr;
  hipGraphNode_t node = nullptr;
  hipGraphNodeType node_type{};

  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([graph] { (void)hipGraphDestroy(graph); });
  HIP_CHECK(hipGraphAddEmptyNode(&node, graph, nullptr, 0));

  HIP_CHECK(hipGraphNodeGetType(node, &node_type));
  REQUIRE(node_type == hipGraphNodeTypeEmpty);
}

HIP_TEST_CASE(Contract_GraphNodeTypes_GetType_MemcpyNodeReportsMemcpy) {
  hip::contract::ContractCleanup cleanup;
  std::array<uint8_t, kByteCount> host{};
  void* device_ptr = nullptr;
  hipGraph_t graph = nullptr;
  hipGraphNode_t node = nullptr;
  hipGraphNodeType node_type{};

  HIP_CHECK(hipMalloc(&device_ptr, host.size()));
  cleanup.Add([device_ptr] { (void)hipFree(device_ptr); });
  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([graph] { (void)hipGraphDestroy(graph); });
  HIP_CHECK(hipGraphAddMemcpyNode1D(&node, graph, nullptr, 0, device_ptr, host.data(), host.size(),
                                    hipMemcpyHostToDevice));

  HIP_CHECK(hipGraphNodeGetType(node, &node_type));
  REQUIRE(node_type == hipGraphNodeTypeMemcpy);
}

HIP_TEST_CASE(Contract_GraphNodeTypes_GetType_MemsetNodeReportsMemset) {
  hip::contract::ContractCleanup cleanup;
  void* device_ptr = nullptr;
  hipMemsetParams params{};
  hipGraph_t graph = nullptr;
  hipGraphNode_t node = nullptr;
  hipGraphNodeType node_type{};

  HIP_CHECK(hipMalloc(&device_ptr, kByteCount));
  cleanup.Add([device_ptr] { (void)hipFree(device_ptr); });
  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([graph] { (void)hipGraphDestroy(graph); });

  params.dst = device_ptr;
  params.value = 0x1f;
  params.pitch = kByteCount;
  params.elementSize = sizeof(uint8_t);
  params.width = kByteCount;
  params.height = 1;

  HIP_CHECK(hipGraphAddMemsetNode(&node, graph, nullptr, 0, &params));

  HIP_CHECK(hipGraphNodeGetType(node, &node_type));
  REQUIRE(node_type == hipGraphNodeTypeMemset);
}

HIP_TEST_CASE(Contract_GraphNodeTypes_AddDependencies_CreatesEdge) {
  hip::contract::ContractCleanup cleanup;
  hipGraph_t graph = nullptr;
  hipGraphNode_t from = nullptr;
  hipGraphNode_t to = nullptr;
  size_t edge_count = 0;
  size_t dependency_count = 0;

  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([graph] { (void)hipGraphDestroy(graph); });
  HIP_CHECK(hipGraphAddEmptyNode(&from, graph, nullptr, 0));
  HIP_CHECK(hipGraphAddEmptyNode(&to, graph, nullptr, 0));

  HIP_CHECK(hipGraphGetEdges(graph, nullptr, nullptr, &edge_count));
  REQUIRE(edge_count == 0);

  HIP_CHECK(hipGraphAddDependencies(graph, &from, &to, 1));

  HIP_CHECK(hipGraphGetEdges(graph, nullptr, nullptr, &edge_count));
  REQUIRE(edge_count == 1);

  HIP_CHECK(hipGraphNodeGetDependencies(to, nullptr, &dependency_count));
  REQUIRE(dependency_count == 1);
}

HIP_TEST_CASE(Contract_GraphNodeTypes_RemoveDependencies_ClearsEdge) {
  hip::contract::ContractCleanup cleanup;
  hipGraph_t graph = nullptr;
  hipGraphNode_t from = nullptr;
  hipGraphNode_t to = nullptr;
  size_t edge_count = 0;

  HIP_CHECK(hipGraphCreate(&graph, 0));
  cleanup.Add([graph] { (void)hipGraphDestroy(graph); });
  HIP_CHECK(hipGraphAddEmptyNode(&from, graph, nullptr, 0));
  HIP_CHECK(hipGraphAddEmptyNode(&to, graph, nullptr, 0));

  HIP_CHECK(hipGraphAddDependencies(graph, &from, &to, 1));
  HIP_CHECK(hipGraphGetEdges(graph, nullptr, nullptr, &edge_count));
  REQUIRE(edge_count == 1);

  HIP_CHECK(hipGraphRemoveDependencies(graph, &from, &to, 1));

  HIP_CHECK(hipGraphGetEdges(graph, nullptr, nullptr, &edge_count));
  REQUIRE(edge_count == 0);
}
