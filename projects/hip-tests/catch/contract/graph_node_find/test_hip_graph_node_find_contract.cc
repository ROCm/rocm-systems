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
}

HIP_TEST_CASE(Contract_GraphNodeFind_ClonedNode_IsFoundAndDistinct) {
  hipGraph_t graph = nullptr;
  hipGraph_t clone = nullptr;
  hipGraphNode_t original_node = nullptr;
  hipGraphNode_t found = nullptr;

  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipGraphAddEmptyNode(&original_node, graph, nullptr, 0));
  HIP_CHECK(hipGraphClone(&clone, graph));

  HIP_CHECK(hipGraphNodeFindInClone(&found, original_node, clone));

  REQUIRE(found != nullptr);
  REQUIRE(found != original_node);

  HIP_CHECK(hipGraphDestroy(clone));
  HIP_CHECK(hipGraphDestroy(graph));
}

HIP_TEST_CASE(Contract_GraphNodeFind_FoundNodeMatchesType) {
  std::array<uint8_t, kByteCount> host{};
  void* device_ptr = nullptr;
  hipGraph_t graph = nullptr;
  hipGraph_t clone = nullptr;
  hipGraphNode_t original_node = nullptr;
  hipGraphNode_t found = nullptr;
  hipGraphNodeType node_type{};

  HIP_CHECK(hipMalloc(&device_ptr, host.size()));
  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&original_node, graph, nullptr, 0, device_ptr, host.data(),
                                    host.size(), hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphClone(&clone, graph));

  HIP_CHECK(hipGraphNodeFindInClone(&found, original_node, clone));
  REQUIRE(found != nullptr);

  HIP_CHECK(hipGraphNodeGetType(found, &node_type));
  REQUIRE(node_type == hipGraphNodeTypeMemcpy);

  HIP_CHECK(hipGraphDestroy(clone));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(device_ptr));
}

HIP_TEST_CASE(Contract_GraphNodeFind_NodeOnlyInOriginal_ReturnsInvalidValue) {
  hipGraph_t graph = nullptr;
  hipGraph_t clone = nullptr;
  hipGraphNode_t first_node = nullptr;
  hipGraphNode_t second_node = nullptr;
  hipGraphNode_t found = nullptr;

  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipGraphAddEmptyNode(&first_node, graph, nullptr, 0));
  HIP_CHECK(hipGraphClone(&clone, graph));

  // The second node is added only to the original graph after cloning, so it
  // has no counterpart in the clone and the lookup must report a portable
  // invalid-value error.
  HIP_CHECK(hipGraphAddEmptyNode(&second_node, graph, nullptr, 0));

  HIP_CHECK_ERROR(hipGraphNodeFindInClone(&found, second_node, clone), hipErrorInvalidValue);

  HIP_CHECK(hipGraphDestroy(clone));
  HIP_CHECK(hipGraphDestroy(graph));
}
