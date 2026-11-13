/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
Testcase Scenarios :
 Invalid-argument checks for the eight external-semaphore graph node APIs. Each
 case fails argument validation before any semaphore is used, so no external
 producer (Vulkan, DMA-buf) is needed, and each expects hipErrorInvalidValue.
 These APIs once dispatched through null runtime table entries and crashed on
 every call; these cases fail with a crash if that regresses.
 1) hipGraphAddExternalSemaphoresSignalNode / WaitNode: null node out-pointer,
    null graph, null node parameters, dependencies count without dependencies.
 2) hipGraphExternalSemaphores{Signal,Wait}Node{Get,Set}Params: null node.
 3) hipGraphExecExternalSemaphores{Signal,Wait}NodeSetParams: null executable
    graph, null node, node of another type.
*/
#include <hip_test_common.hh>

/**
 * Negative Scenarios hipGraphAddExternalSemaphoresSignalNode
 */
HIP_TEST_CASE(Unit_hipGraphAddExternalSemaphoresSignalNode_Negative_InvalidArgs) {
  auto graph = hipGraph_t{};
  HIP_CHECK(hipGraphCreate(&graph, 0));
  auto node = hipGraphNode_t{};
  auto params = hipExternalSemaphoreSignalNodeParams{};

  SECTION("Null node out-pointer") {
    HIP_CHECK_ERROR(hipGraphAddExternalSemaphoresSignalNode(nullptr, graph, nullptr, 0, &params),
                    hipErrorInvalidValue);
  }
  SECTION("Null graph") {
    HIP_CHECK_ERROR(hipGraphAddExternalSemaphoresSignalNode(&node, nullptr, nullptr, 0, &params),
                    hipErrorInvalidValue);
  }
  SECTION("Null node parameters") {
    HIP_CHECK_ERROR(hipGraphAddExternalSemaphoresSignalNode(&node, graph, nullptr, 0, nullptr),
                    hipErrorInvalidValue);
  }
  SECTION("Dependencies count without dependencies") {
    HIP_CHECK_ERROR(hipGraphAddExternalSemaphoresSignalNode(&node, graph, nullptr, 1, &params),
                    hipErrorInvalidValue);
  }

  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Negative Scenarios hipGraphAddExternalSemaphoresWaitNode
 */
HIP_TEST_CASE(Unit_hipGraphAddExternalSemaphoresWaitNode_Negative_InvalidArgs) {
  auto graph = hipGraph_t{};
  HIP_CHECK(hipGraphCreate(&graph, 0));
  auto node = hipGraphNode_t{};
  auto params = hipExternalSemaphoreWaitNodeParams{};

  SECTION("Null node out-pointer") {
    HIP_CHECK_ERROR(hipGraphAddExternalSemaphoresWaitNode(nullptr, graph, nullptr, 0, &params),
                    hipErrorInvalidValue);
  }
  SECTION("Null graph") {
    HIP_CHECK_ERROR(hipGraphAddExternalSemaphoresWaitNode(&node, nullptr, nullptr, 0, &params),
                    hipErrorInvalidValue);
  }
  SECTION("Null node parameters") {
    HIP_CHECK_ERROR(hipGraphAddExternalSemaphoresWaitNode(&node, graph, nullptr, 0, nullptr),
                    hipErrorInvalidValue);
  }
  SECTION("Dependencies count without dependencies") {
    HIP_CHECK_ERROR(hipGraphAddExternalSemaphoresWaitNode(&node, graph, nullptr, 1, &params),
                    hipErrorInvalidValue);
  }

  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Negative Scenarios hipGraphExternalSemaphores{Signal,Wait}Node{Get,Set}Params
 */
HIP_TEST_CASE(Unit_hipGraphExternalSemaphoresNodeParams_Negative_NullNode) {
  auto signal_params = hipExternalSemaphoreSignalNodeParams{};
  auto wait_params = hipExternalSemaphoreWaitNodeParams{};

  SECTION("hipGraphExternalSemaphoresSignalNodeGetParams") {
    HIP_CHECK_ERROR(hipGraphExternalSemaphoresSignalNodeGetParams(nullptr, &signal_params),
                    hipErrorInvalidValue);
  }
  SECTION("hipGraphExternalSemaphoresSignalNodeSetParams") {
    HIP_CHECK_ERROR(hipGraphExternalSemaphoresSignalNodeSetParams(nullptr, &signal_params),
                    hipErrorInvalidValue);
  }
  SECTION("hipGraphExternalSemaphoresWaitNodeGetParams") {
    HIP_CHECK_ERROR(hipGraphExternalSemaphoresWaitNodeGetParams(nullptr, &wait_params),
                    hipErrorInvalidValue);
  }
  SECTION("hipGraphExternalSemaphoresWaitNodeSetParams") {
    HIP_CHECK_ERROR(hipGraphExternalSemaphoresWaitNodeSetParams(nullptr, &wait_params),
                    hipErrorInvalidValue);
  }
}

/**
 * Negative Scenarios hipGraphExecExternalSemaphores{Signal,Wait}NodeSetParams
 */
HIP_TEST_CASE(Unit_hipGraphExecExternalSemaphoresNodeSetParams_Negative_InvalidArgs) {
  auto graph = hipGraph_t{};
  HIP_CHECK(hipGraphCreate(&graph, 0));
  auto empty_node = hipGraphNode_t{};
  HIP_CHECK(hipGraphAddEmptyNode(&empty_node, graph, nullptr, 0));
  auto graph_exec = hipGraphExec_t{};
  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
  auto signal_params = hipExternalSemaphoreSignalNodeParams{};
  auto wait_params = hipExternalSemaphoreWaitNodeParams{};

  SECTION("Null executable graph") {
    HIP_CHECK_ERROR(
        hipGraphExecExternalSemaphoresSignalNodeSetParams(nullptr, empty_node, &signal_params),
        hipErrorInvalidValue);
    HIP_CHECK_ERROR(
        hipGraphExecExternalSemaphoresWaitNodeSetParams(nullptr, empty_node, &wait_params),
        hipErrorInvalidValue);
  }
  SECTION("Null node") {
    HIP_CHECK_ERROR(
        hipGraphExecExternalSemaphoresSignalNodeSetParams(graph_exec, nullptr, &signal_params),
        hipErrorInvalidValue);
    HIP_CHECK_ERROR(
        hipGraphExecExternalSemaphoresWaitNodeSetParams(graph_exec, nullptr, &wait_params),
        hipErrorInvalidValue);
  }
  SECTION("Node of another type") {
    HIP_CHECK_ERROR(
        hipGraphExecExternalSemaphoresSignalNodeSetParams(graph_exec, empty_node, &signal_params),
        hipErrorInvalidValue);
    HIP_CHECK_ERROR(
        hipGraphExecExternalSemaphoresWaitNodeSetParams(graph_exec, empty_node, &wait_params),
        hipErrorInvalidValue);
  }

  HIP_CHECK(hipGraphExecDestroy(graph_exec));
  HIP_CHECK(hipGraphDestroy(graph));
}
