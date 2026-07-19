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
constexpr size_t kElementCount = 128;

std::array<uint8_t, kElementCount> MakePattern(uint8_t seed) {
  std::array<uint8_t, kElementCount> pattern{};
  for (size_t i = 0; i < pattern.size(); ++i) {
    pattern[i] = static_cast<uint8_t>(seed + i);
  }
  return pattern;
}
}  // namespace

// @asserts: hipGraphAddChildGraphNode - launching a parent graph executes the embedded child graph's memcpy nodes end to end
HIP_TEST_CASE(Contract_GraphChild_AddChildGraphNode_ExecutesEmbeddedMemcpy) {
  hip::contract::ContractCleanup cleanup;
  const auto src = MakePattern(0x37);
  std::array<uint8_t, kElementCount> dst{};
  void* device_ptr = nullptr;
  hipGraph_t parent = nullptr;
  hipGraph_t child = nullptr;
  hipGraphExec_t graph_exec = nullptr;
  hipStream_t stream = nullptr;
  hipGraphNode_t h2d_node = nullptr;
  hipGraphNode_t d2h_node = nullptr;
  hipGraphNode_t child_node = nullptr;

  HIP_CHECK(hipMalloc(&device_ptr, src.size()));
  cleanup.Add([device_ptr] { (void)hipFree(device_ptr); });
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });

  HIP_CHECK(hipGraphCreate(&child, 0));
  cleanup.Add([child] { (void)hipGraphDestroy(child); });
  HIP_CHECK(hipGraphAddMemcpyNode1D(&h2d_node, child, nullptr, 0, device_ptr, src.data(), src.size(),
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&d2h_node, child, &h2d_node, 1, dst.data(), device_ptr,
                                    dst.size(), hipMemcpyDeviceToHost));

  HIP_CHECK(hipGraphCreate(&parent, 0));
  cleanup.Add([parent] { (void)hipGraphDestroy(parent); });
  HIP_CHECK(hipGraphAddChildGraphNode(&child_node, parent, nullptr, 0, child));

  HIP_CHECK(hipGraphInstantiate(&graph_exec, parent, nullptr, nullptr, 0));
  cleanup.Add([graph_exec] { (void)hipGraphExecDestroy(graph_exec); });
  HIP_CHECK(hipGraphLaunch(graph_exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  REQUIRE(dst == src);
}

// @asserts: hipGraphChildGraphNodeGetGraph - returns a non-null embedded graph handle whose node count matches the child graph
HIP_TEST_CASE(Contract_GraphChild_ChildGraphNodeGetGraph_ReturnsHandle) {
  hip::contract::ContractCleanup cleanup;
  hipGraph_t parent = nullptr;
  hipGraph_t child = nullptr;
  hipGraph_t embedded = nullptr;
  hipGraphNode_t empty_node = nullptr;
  hipGraphNode_t child_node = nullptr;
  size_t node_count = 0;

  HIP_CHECK(hipGraphCreate(&child, 0));
  cleanup.Add([child] { (void)hipGraphDestroy(child); });
  HIP_CHECK(hipGraphAddEmptyNode(&empty_node, child, nullptr, 0));

  HIP_CHECK(hipGraphCreate(&parent, 0));
  cleanup.Add([parent] { (void)hipGraphDestroy(parent); });
  HIP_CHECK(hipGraphAddChildGraphNode(&child_node, parent, nullptr, 0, child));

  HIP_CHECK(hipGraphChildGraphNodeGetGraph(child_node, &embedded));
  REQUIRE(embedded != nullptr);

  HIP_CHECK(hipGraphGetNodes(embedded, nullptr, &node_count));
  REQUIRE(node_count == 1);
}

// @asserts: hipGraphNodeGetType - a child-graph node reports node type hipGraphNodeTypeGraph
HIP_TEST_CASE(Contract_GraphChild_NodeType_ReportsGraph) {
  hip::contract::ContractCleanup cleanup;
  hipGraph_t parent = nullptr;
  hipGraph_t child = nullptr;
  hipGraphNode_t child_node = nullptr;
  hipGraphNodeType node_type{};

  HIP_CHECK(hipGraphCreate(&child, 0));
  cleanup.Add([child] { (void)hipGraphDestroy(child); });

  HIP_CHECK(hipGraphCreate(&parent, 0));
  cleanup.Add([parent] { (void)hipGraphDestroy(parent); });
  HIP_CHECK(hipGraphAddChildGraphNode(&child_node, parent, nullptr, 0, child));

  HIP_CHECK(hipGraphNodeGetType(child_node, &node_type));
  REQUIRE(node_type == hipGraphNodeTypeGraph);
}
