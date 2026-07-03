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
constexpr size_t kElementCount = 128;

std::array<uint8_t, kElementCount> MakePattern(uint8_t seed) {
  std::array<uint8_t, kElementCount> pattern{};
  for (size_t i = 0; i < pattern.size(); ++i) {
    pattern[i] = static_cast<uint8_t>(seed + i);
  }
  return pattern;
}
}  // namespace

HIP_TEST_CASE(Contract_GraphChild_AddChildGraphNode_ExecutesEmbeddedMemcpy) {
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
  HIP_CHECK(hipStreamCreate(&stream));

  HIP_CHECK(hipGraphCreate(&child, 0));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&h2d_node, child, nullptr, 0, device_ptr, src.data(), src.size(),
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&d2h_node, child, &h2d_node, 1, dst.data(), device_ptr,
                                    dst.size(), hipMemcpyDeviceToHost));

  HIP_CHECK(hipGraphCreate(&parent, 0));
  HIP_CHECK(hipGraphAddChildGraphNode(&child_node, parent, nullptr, 0, child));

  HIP_CHECK(hipGraphInstantiate(&graph_exec, parent, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graph_exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  REQUIRE(dst == src);

  HIP_CHECK(hipGraphExecDestroy(graph_exec));
  HIP_CHECK(hipGraphDestroy(parent));
  HIP_CHECK(hipGraphDestroy(child));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(device_ptr));
}

HIP_TEST_CASE(Contract_GraphChild_ChildGraphNodeGetGraph_ReturnsHandle) {
  hipGraph_t parent = nullptr;
  hipGraph_t child = nullptr;
  hipGraph_t embedded = nullptr;
  hipGraphNode_t empty_node = nullptr;
  hipGraphNode_t child_node = nullptr;
  size_t node_count = 0;

  HIP_CHECK(hipGraphCreate(&child, 0));
  HIP_CHECK(hipGraphAddEmptyNode(&empty_node, child, nullptr, 0));

  HIP_CHECK(hipGraphCreate(&parent, 0));
  HIP_CHECK(hipGraphAddChildGraphNode(&child_node, parent, nullptr, 0, child));

  HIP_CHECK(hipGraphChildGraphNodeGetGraph(child_node, &embedded));
  REQUIRE(embedded != nullptr);

  HIP_CHECK(hipGraphGetNodes(embedded, nullptr, &node_count));
  REQUIRE(node_count == 1);

  HIP_CHECK(hipGraphDestroy(parent));
  HIP_CHECK(hipGraphDestroy(child));
}

HIP_TEST_CASE(Contract_GraphChild_NodeType_ReportsGraph) {
  hipGraph_t parent = nullptr;
  hipGraph_t child = nullptr;
  hipGraphNode_t child_node = nullptr;
  hipGraphNodeType node_type{};

  HIP_CHECK(hipGraphCreate(&child, 0));

  HIP_CHECK(hipGraphCreate(&parent, 0));
  HIP_CHECK(hipGraphAddChildGraphNode(&child_node, parent, nullptr, 0, child));

  HIP_CHECK(hipGraphNodeGetType(child_node, &node_type));
  REQUIRE(node_type == hipGraphNodeTypeGraph);

  HIP_CHECK(hipGraphDestroy(parent));
  HIP_CHECK(hipGraphDestroy(child));
}
