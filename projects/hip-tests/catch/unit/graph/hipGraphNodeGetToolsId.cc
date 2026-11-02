/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

#if HT_AMD || (defined(CUDA_VERSION) && CUDA_VERSION >= 13010)

/**
 * @addtogroup hipGraphNodeGetToolsId hipGraphNodeGetToolsId
 * @{
 * @ingroup GraphTest
 * `hipGraphNodeGetToolsId(hipGraphNode_t hNode, unsigned long long *toolsNodeId)` -
 * Returns an id used by tools to identify a given node
 */

/**
 * Test Description
 * ------------------------
 *    - Verify that tools ID encodes graph ID and local node ID:
 *        -# Create a graph with multiple nodes
 *        -# Verify local IDs increment from zero and toolsId == (graphId << 32) | localId
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphNodeGetToolsId.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.18
 */
HIP_TEST_CASE(Unit_hipGraphNodeGetToolsId_Positive_EncodesGraphIdAndLocalId) {
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  unsigned int graphId;
  HIP_CHECK(hipGraphGetId(graph, &graphId));

  constexpr int numNodes = 3;
  for (int i = 0; i < numNodes; i++) {
    hipGraphNode_t node;
    HIP_CHECK(hipGraphAddEmptyNode(&node, graph, nullptr, 0));

    unsigned int localId;
    unsigned long long toolsId;
    HIP_CHECK(hipGraphNodeGetLocalId(node, &localId));
    HIP_CHECK(hipGraphNodeGetToolsId(node, &toolsId));

    REQUIRE(localId == static_cast<unsigned int>(i));
    unsigned long long expected =
        (static_cast<unsigned long long>(graphId) << 32) | localId;
    REQUIRE(toolsId == expected);
  }

  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *    - Verify tools ID encoding holds for nodes in a cloned graph:
 *        -# Create a graph with a node, clone it
 *        -# Verify cloned node's toolsId == (clonedGraphId << 32) | localId
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphNodeGetToolsId.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.18
 */
HIP_TEST_CASE(Unit_hipGraphNodeGetToolsId_Positive_EncodingHoldsForClonedGraph) {
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphNode_t node;
  HIP_CHECK(hipGraphAddEmptyNode(&node, graph, nullptr, 0));

  hipGraph_t clonedGraph;
  HIP_CHECK(hipGraphClone(&clonedGraph, graph));

  unsigned int clonedGraphId;
  HIP_CHECK(hipGraphGetId(clonedGraph, &clonedGraphId));

  size_t numNodes = 0;
  HIP_CHECK(hipGraphGetNodes(clonedGraph, nullptr, &numNodes));
  REQUIRE(numNodes == 1);

  hipGraphNode_t clonedNode;
  HIP_CHECK(hipGraphGetNodes(clonedGraph, &clonedNode, &numNodes));

  unsigned int clonedLocalId;
  unsigned long long clonedToolsId;
  HIP_CHECK(hipGraphNodeGetLocalId(clonedNode, &clonedLocalId));
  HIP_CHECK(hipGraphNodeGetToolsId(clonedNode, &clonedToolsId));

  unsigned long long expected =
      ((unsigned long long)clonedGraphId << 32) | clonedLocalId;
  REQUIRE(clonedToolsId == expected);

  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipGraphDestroy(clonedGraph));
}

/**
 * Test Description
 * ------------------------
 *    - Verify child graph node's tools ID uses the parent graph's ID:
 *        -# Create parent and child graphs
 *        -# Add child as child graph node in parent
 *        -# Verify child graph node's toolsId encodes the parent graph's ID
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphNodeGetToolsId.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.18
 */
HIP_TEST_CASE(Unit_hipGraphNodeGetToolsId_Positive_ChildGraphNodeUsesParentGraphId) {
  hipGraph_t parentGraph, childGraph;
  HIP_CHECK(hipGraphCreate(&parentGraph, 0));
  HIP_CHECK(hipGraphCreate(&childGraph, 0));

  hipGraphNode_t childEmptyNode;
  HIP_CHECK(hipGraphAddEmptyNode(&childEmptyNode, childGraph, nullptr, 0));

  hipGraphNode_t childGraphNode;
  HIP_CHECK(hipGraphAddChildGraphNode(&childGraphNode, parentGraph, nullptr, 0,
                                       childGraph));

  unsigned int parentGraphId;
  HIP_CHECK(hipGraphGetId(parentGraph, &parentGraphId));

  unsigned int localId;
  unsigned long long toolsId;
  HIP_CHECK(hipGraphNodeGetLocalId(childGraphNode, &localId));
  HIP_CHECK(hipGraphNodeGetToolsId(childGraphNode, &toolsId));

  unsigned long long expected =
      ((unsigned long long)parentGraphId << 32) | localId;
  REQUIRE(toolsId == expected);

  HIP_CHECK(hipGraphDestroy(parentGraph));
  HIP_CHECK(hipGraphDestroy(childGraph));
}

/**
 * End doxygen group GraphTest.
 * @}
 */

#endif  // HT_AMD || CUDA_VERSION >= 13010
