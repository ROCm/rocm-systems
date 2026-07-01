/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstdint>

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>

namespace {
constexpr int32_t kInitialValue = 0;
constexpr int32_t kExpectedValue = 1;

// Plain C callback: does not call HIP APIs, only mutates data through userData.
void IncrementCounter(void* userData) {
  auto* counter = static_cast<int32_t*>(userData);
  *counter += 1;
}
}  // namespace

HIP_TEST_CASE(Contract_GraphHost_AddHostNode_InvokesCallbackOnLaunch) {
  int32_t counter = kInitialValue;
  hipGraph_t graph = nullptr;
  hipGraphExec_t graph_exec = nullptr;
  hipStream_t stream = nullptr;
  hipGraphNode_t host_node = nullptr;

  hipHostNodeParams params{};
  params.fn = IncrementCounter;
  params.userData = &counter;

  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipGraphAddHostNode(&host_node, graph, nullptr, 0, &params));

  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graph_exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  REQUIRE(counter == kExpectedValue);

  HIP_CHECK(hipGraphExecDestroy(graph_exec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
}

HIP_TEST_CASE(Contract_GraphHost_HostNodeGetParams_RoundTripsFnAndUserData) {
  int32_t counter = kInitialValue;
  hipGraph_t graph = nullptr;
  hipGraphNode_t host_node = nullptr;

  hipHostNodeParams params{};
  params.fn = IncrementCounter;
  params.userData = &counter;

  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipGraphAddHostNode(&host_node, graph, nullptr, 0, &params));

  hipHostNodeParams retrieved{};
  HIP_CHECK(hipGraphHostNodeGetParams(host_node, &retrieved));

  REQUIRE(retrieved.fn == IncrementCounter);
  REQUIRE(retrieved.userData == &counter);

  HIP_CHECK(hipGraphDestroy(graph));
}

HIP_TEST_CASE(Contract_GraphHost_NodeType_ReportsHost) {
  int32_t counter = kInitialValue;
  hipGraph_t graph = nullptr;
  hipGraphNode_t host_node = nullptr;
  hipGraphNodeType node_type{};

  hipHostNodeParams params{};
  params.fn = IncrementCounter;
  params.userData = &counter;

  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipGraphAddHostNode(&host_node, graph, nullptr, 0, &params));

  HIP_CHECK(hipGraphNodeGetType(host_node, &node_type));
  REQUIRE(node_type == hipGraphNodeTypeHost);

  HIP_CHECK(hipGraphDestroy(graph));
}
