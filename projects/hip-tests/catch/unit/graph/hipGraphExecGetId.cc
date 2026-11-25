/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip_test_kernels.hh>

#if HT_AMD || (defined(CUDA_VERSION) && CUDA_VERSION >= 13010)

/**
 * @addtogroup hipGraphExecGetId hipGraphExecGetId
 * @{
 * @ingroup GraphTest
 * `hipGraphExecGetId(hipGraphExec_t hGraphExec, unsigned int *graphID)` -
 * Returns the id of a given graph exec
 */

static __global__ void emptyKernel() {}

/**
 * Test Description
 * ------------------------
 *    - Verify graph and executable graph IDs share one creation-order counter:
 *        -# Create a graph, instantiate it, then repeat with a second graph
 *        -# Verify all four IDs are strictly increasing in creation order
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphExecGetId.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.18
 */
HIP_TEST_CASE(Unit_hipGraphExecGetId_Positive_SharesCounterWithGraphId) {
  hipGraph_t graph1;
  HIP_CHECK(hipGraphCreate(&graph1, 0));
  hipGraphNode_t node1;
  HIP_CHECK(hipGraphAddEmptyNode(&node1, graph1, nullptr, 0));
  unsigned int graphId1;
  HIP_CHECK(hipGraphGetId(graph1, &graphId1));

  hipGraphExec_t exec1;
  HIP_CHECK(hipGraphInstantiate(&exec1, graph1, nullptr, nullptr, 0));
  unsigned int execId1;
  HIP_CHECK(hipGraphExecGetId(exec1, &execId1));

  hipGraph_t graph2;
  HIP_CHECK(hipGraphCreate(&graph2, 0));
  hipGraphNode_t node2;
  HIP_CHECK(hipGraphAddEmptyNode(&node2, graph2, nullptr, 0));
  unsigned int graphId2;
  HIP_CHECK(hipGraphGetId(graph2, &graphId2));

  hipGraphExec_t exec2;
  HIP_CHECK(hipGraphInstantiate(&exec2, graph2, nullptr, nullptr, 0));
  unsigned int execId2;
  HIP_CHECK(hipGraphExecGetId(exec2, &execId2));

  REQUIRE(graphId1 < execId1);
  REQUIRE(execId1 < graphId2);
  REQUIRE(graphId2 < execId2);

  HIP_CHECK(hipGraphExecDestroy(exec1));
  HIP_CHECK(hipGraphExecDestroy(exec2));
  HIP_CHECK(hipGraphDestroy(graph1));
  HIP_CHECK(hipGraphDestroy(graph2));
}

/**
 * Test Description
 * ------------------------
 *    - Verify exec ID is stable after update and launch:
 *        -# Instantiate a graph, get exec ID
 *        -# Create a second compatible graph and update the exec with it
 *        -# Launch the updated exec
 *        -# Verify the exec ID is unchanged after both operations
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphExecGetId.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.18
 */
HIP_TEST_CASE(Unit_hipGraphExecGetId_Positive_StableAfterUpdateAndLaunch) {
  hipGraph_t graph1;
  HIP_CHECK(hipGraphCreate(&graph1, 0));
  hipGraphNode_t n1;
  hipKernelNodeParams kp = {};
  kp.func = reinterpret_cast<void*>(emptyKernel);
  kp.gridDim = dim3(1);
  kp.blockDim = dim3(1);
  kp.sharedMemBytes = 0;
  kp.kernelParams = nullptr;
  kp.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&n1, graph1, nullptr, 0, &kp));

  hipGraphExec_t exec;
  HIP_CHECK(hipGraphInstantiate(&exec, graph1, nullptr, nullptr, 0));

  unsigned int execIdBefore;
  HIP_CHECK(hipGraphExecGetId(exec, &execIdBefore));

  hipGraph_t graph2;
  HIP_CHECK(hipGraphCreate(&graph2, 0));
  hipGraphNode_t n2;
  hipKernelNodeParams kp2 = kp;
  kp2.gridDim = dim3(2);
  HIP_CHECK(hipGraphAddKernelNode(&n2, graph2, nullptr, 0, &kp2));

  hipGraphNode_t errorNode = nullptr;
  hipGraphExecUpdateResult updateResult;
  HIP_CHECK(hipGraphExecUpdate(exec, graph2, &errorNode, &updateResult));
  REQUIRE(updateResult == hipGraphExecUpdateSuccess);
  REQUIRE(errorNode == nullptr);

  unsigned int execIdAfterUpdate;
  HIP_CHECK(hipGraphExecGetId(exec, &execIdAfterUpdate));
  REQUIRE(execIdBefore == execIdAfterUpdate);

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipGraphLaunch(exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  unsigned int execIdAfterLaunch;
  HIP_CHECK(hipGraphExecGetId(exec, &execIdAfterLaunch));
  REQUIRE(execIdBefore == execIdAfterLaunch);

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(graph1));
  HIP_CHECK(hipGraphDestroy(graph2));
}

/**
 * Test Description
 * ------------------------
 *    - Verify exec IDs are not reused after destroy and reinstantiate:
 *        -# Instantiate a graph, get exec ID, destroy exec
 *        -# Instantiate the same graph again
 *        -# Verify the new exec has a different ID
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphExecGetId.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.18
 */
HIP_TEST_CASE(Unit_hipGraphExecGetId_Positive_NoReuseAfterDestroy) {
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipGraphNode_t node;
  HIP_CHECK(hipGraphAddEmptyNode(&node, graph, nullptr, 0));

  hipGraphExec_t exec1;
  HIP_CHECK(hipGraphInstantiate(&exec1, graph, nullptr, nullptr, 0));
  unsigned int id1;
  HIP_CHECK(hipGraphExecGetId(exec1, &id1));
  HIP_CHECK(hipGraphExecDestroy(exec1));

  hipGraphExec_t exec2;
  HIP_CHECK(hipGraphInstantiate(&exec2, graph, nullptr, nullptr, 0));
  unsigned int id2;
  HIP_CHECK(hipGraphExecGetId(exec2, &id2));
  HIP_CHECK(hipGraphExecDestroy(exec2));

  REQUIRE(id1 != id2);

  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * End doxygen group GraphTest.
 * @}
 */

#endif  // HT_AMD || CUDA_VERSION >= 13010
