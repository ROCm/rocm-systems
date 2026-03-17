/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */


#include <hip_test_checkers.hh>
#include <hip_test_common.hh>

/**
 * @addtogroup hipGraphNodeSetParams hipGraphNodeSetParams
 * @{
 * @ingroup GraphTest
 * `hipGraphNodeSetParams(hipGraphNode_t  node,
 * hipGraphNodeParams *nodeParams)` -
 * Updates parameters of a graph’s node
 */
/**
 * Test Description
 * ------------------------
 *    - Verify API behavior with invalid arguments:
 *      -# node is nullptr
 *      -# nodeParams is nullptr
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphNodeSetParams.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.4
 */
TEST_CASE(Unit_hipGraphNodeSetParams_Negative_Parameters) {
  hipGraph_t graph;
  hipGraphExec_t graphExec;
  hipGraphNode_t node;
  hipGraphNodeParams node_params = {};
  char* A_d;
  char* A_h;
  size_t N = 10;
  size_t Nbytes = N * sizeof(char);

  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipMalloc(&A_d, Nbytes));
  node_params.type = hipGraphNodeTypeMemset;
  node_params.memset.dst = A_d;
  node_params.memset.elementSize = sizeof(char);
  node_params.memset.width = N;
  node_params.memset.height = 1;
  node_params.memset.pitch = N;
  node_params.memset.value = 99;

  HIP_CHECK(hipGraphAddNode(&node, graph, nullptr, 0, &node_params));
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, 0));
  A_h = reinterpret_cast<char*>(malloc(Nbytes));
  HIP_CHECK(hipMemcpy(A_h, A_d, Nbytes, hipMemcpyDeviceToHost));
  hipGraphNodeParams node_params2 = {};
  node_params2.type = hipGraphNodeTypeMemset;
  node_params2.memset.dst = A_d;
  node_params2.memset.elementSize = sizeof(char);
  node_params2.memset.width = N;
  node_params2.memset.height = 1;
  node_params2.memset.pitch = N;
  node_params2.memset.value = 110;

  SECTION("node == nullptr") {
    HIP_CHECK_ERROR(hipGraphNodeSetParams(nullptr, &node_params2), hipErrorInvalidValue);
  }

  SECTION("nodeParams == nullptr") {
    HIP_CHECK_ERROR(hipGraphNodeSetParams(node, nullptr), hipErrorInvalidValue);
  }

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(A_d));
  free(A_h);
}

/**
 * Test Description
 * ------------------------
 *    - This will verify the new node param values are successfully
 * copied to graph node
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphNodeSetParams.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.4
 */
TEST_CASE(Unit_hipGraphNodeSetParams_Positive) {
  hipGraph_t graph;
  hipGraphExec_t graphExec;
  hipGraphNode_t node;
  hipGraphNodeParams node_params = {};
  char *A_d, *A_h = nullptr;
  size_t N = 10;
  size_t Nbytes = N * sizeof(char);

  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipMalloc(&A_d, Nbytes));
  node_params.type = hipGraphNodeTypeMemset;
  node_params.memset.dst = A_d;
  node_params.memset.elementSize = sizeof(char);
  node_params.memset.width = N;
  node_params.memset.height = 1;
  node_params.memset.pitch = N;
  node_params.memset.value = 99;

  HIP_CHECK(hipGraphAddNode(&node, graph, nullptr, 0, &node_params));
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, 0));
  HIP_CHECK(hipStreamSynchronize(0));
  A_h = reinterpret_cast<char*>(malloc(Nbytes));
  HIP_CHECK(hipMemcpy(A_h, A_d, Nbytes, hipMemcpyDeviceToHost));
  for (int i = 0; i < N; i++) {
    REQUIRE(A_h[i] == 99);
  }
  HIP_CHECK(hipGraphExecDestroy(graphExec));

  hipGraphNodeParams node_params2 = {};
  node_params2.type = hipGraphNodeTypeMemset;
  node_params2.memset.dst = A_d;
  node_params2.memset.elementSize = sizeof(char);
  node_params2.memset.width = N;
  node_params2.memset.height = 1;
  node_params2.memset.pitch = N;
  node_params2.memset.value = 110;

  HIP_CHECK(hipGraphNodeSetParams(node, &node_params2));
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, 0));
  HIP_CHECK(hipStreamSynchronize(0));
  HIP_CHECK(hipMemcpy(A_h, A_d, Nbytes, hipMemcpyDeviceToHost));
  for (int i = 0; i < N; i++) {
    REQUIRE(A_h[i] == 110);
  }

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(A_d));
  free(A_h);
}

/**
 * End doxygen group GraphTest.
 * @}
 */
