/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip_test_kernels.hh>
#include <set>
#include <vector>

#if HT_AMD || (defined(CUDA_VERSION) && CUDA_VERSION >= 13010)

/**
 * @addtogroup hipGraphNodeGetLocalId hipGraphNodeGetLocalId
 * @{
 * @ingroup GraphTest
 * `hipGraphNodeGetLocalId(hipGraphNode_t hNode, unsigned int *nodeId)` -
 * Returns the local node id of a given graph node
 */

static __global__ void emptyKernel() {}

/**
 * Test Description
 * ------------------------
 *    - Verify local node IDs are assigned consecutively from zero:
 *        -# Create a graph with multiple empty nodes
 *        -# Verify their local IDs are 0, 1, 2, 3, 4
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphNodeGetLocalId.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.18
 */
HIP_TEST_CASE(Unit_hipGraphNodeGetLocalId_Positive_StartsAtZeroAndIncrements) {
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  constexpr int numNodes = 5;
  std::vector<hipGraphNode_t> nodes(numNodes);
  std::vector<unsigned int> ids(numNodes);

  for (int i = 0; i < numNodes; i++) {
    HIP_CHECK(hipGraphAddEmptyNode(&nodes[i], graph, nullptr, 0));
    HIP_CHECK(hipGraphNodeGetLocalId(nodes[i], &ids[i]));
  }

  for (int i = 0; i < numNodes; i++) {
    REQUIRE(ids[i] == static_cast<unsigned int>(i));
  }

  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *    - Verify a child graph node uses the parent graph's local ID sequence:
 *        -# Create parent and child graphs
 *        -# Add an empty node and a child graph node to the parent
 *        -# Verify the child graph keeps a separate local ID sequence
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphNodeGetLocalId.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.18
 */
HIP_TEST_CASE(Unit_hipGraphNodeGetLocalId_Positive_ChildGraphNode) {
  hipGraph_t parentGraph, childGraph;
  HIP_CHECK(hipGraphCreate(&parentGraph, 0));
  HIP_CHECK(hipGraphCreate(&childGraph, 0));

  hipGraphNode_t emptyNodeInChildGraph;
  HIP_CHECK(hipGraphAddEmptyNode(&emptyNodeInChildGraph, childGraph, nullptr, 0));

  hipGraphNode_t emptyNodeInParentGraph;
  HIP_CHECK(hipGraphAddEmptyNode(&emptyNodeInParentGraph, parentGraph, nullptr, 0));

  hipGraphNode_t childGraphNode;
  HIP_CHECK(hipGraphAddChildGraphNode(&childGraphNode, parentGraph, nullptr, 0,
                                       childGraph));

  unsigned int emptyNodeInParentGraphId;
  HIP_CHECK(hipGraphNodeGetLocalId(emptyNodeInParentGraph, &emptyNodeInParentGraphId));

  unsigned int childGraphNodeId;
  HIP_CHECK(hipGraphNodeGetLocalId(childGraphNode, &childGraphNodeId));

  unsigned int emptyNodeInChildGraphId;
  HIP_CHECK(hipGraphNodeGetLocalId(emptyNodeInChildGraph, &emptyNodeInChildGraphId));

  REQUIRE(emptyNodeInParentGraphId == 0);
  REQUIRE(childGraphNodeId == 1);
  REQUIRE(emptyNodeInChildGraphId == 0);

  HIP_CHECK(hipGraphDestroy(parentGraph));
  HIP_CHECK(hipGraphDestroy(childGraph));
}

/**
 * Test Description
 * ------------------------
 *    - Verify node IDs are unique in a graph obtained from stream capture:
 *        -# Capture a stream with multiple kernel launches
 *        -# Extract nodes and verify all local IDs are unique
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphNodeGetLocalId.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.18
 */
HIP_TEST_CASE(Unit_hipGraphNodeGetLocalId_Positive_CapturedStream) {
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));

  emptyKernel<<<1, 1, 0, stream>>>();
  emptyKernel<<<1, 1, 0, stream>>>();

  hipGraph_t graph;
  HIP_CHECK(hipStreamEndCapture(stream, &graph));

  size_t numNodes = 0;
  HIP_CHECK(hipGraphGetNodes(graph, nullptr, &numNodes));
  REQUIRE(numNodes >= 2);

  std::vector<hipGraphNode_t> nodes(numNodes);
  HIP_CHECK(hipGraphGetNodes(graph, nodes.data(), &numNodes));

  std::set<unsigned int> idSet;
  for (size_t i = 0; i < numNodes; i++) {
    unsigned int nodeId;
    HIP_CHECK(hipGraphNodeGetLocalId(nodes[i], &nodeId));
    idSet.insert(nodeId);
  }

  REQUIRE(idSet.size() == numNodes);

  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 *    - Verify cloned graph preserves local node IDs from the original:
 *        -# Create a graph with three nodes, record their local IDs
 *        -# Clone the graph
 *        -# Verify cloned nodes have the same local IDs as originals
 *        -# Add a node to the clone and verify its ID follows the cloned sequence
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphNodeGetLocalId.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.18
 */
HIP_TEST_CASE(Unit_hipGraphNodeGetLocalId_Positive_ClonedGraphPreservesIds) {
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  constexpr int numNodes = 3;
  std::vector<hipGraphNode_t> origNodes(numNodes);
  std::vector<unsigned int> origIds(numNodes);
  for (int i = 0; i < numNodes; i++) {
    HIP_CHECK(hipGraphAddEmptyNode(&origNodes[i], graph, nullptr, 0));
    HIP_CHECK(hipGraphNodeGetLocalId(origNodes[i], &origIds[i]));
  }

  hipGraph_t clonedGraph;
  HIP_CHECK(hipGraphClone(&clonedGraph, graph));

  size_t clonedCount = 0;
  HIP_CHECK(hipGraphGetNodes(clonedGraph, nullptr, &clonedCount));
  REQUIRE(clonedCount == numNodes);

  std::vector<hipGraphNode_t> clonedNodes(clonedCount);
  HIP_CHECK(hipGraphGetNodes(clonedGraph, clonedNodes.data(), &clonedCount));

  std::set<unsigned int> origIdSet(origIds.begin(), origIds.end());
  std::set<unsigned int> clonedIdSet;
  for (size_t i = 0; i < clonedCount; i++) {
    unsigned int nodeId;
    HIP_CHECK(hipGraphNodeGetLocalId(clonedNodes[i], &nodeId));
    clonedIdSet.insert(nodeId);
  }

  REQUIRE(clonedIdSet == origIdSet);

  hipGraphNode_t appendedNode;
  HIP_CHECK(hipGraphAddEmptyNode(&appendedNode, clonedGraph, nullptr, 0));
  unsigned int appendedId;
  HIP_CHECK(hipGraphNodeGetLocalId(appendedNode, &appendedId));
  REQUIRE(appendedId == static_cast<unsigned int>(numNodes));

  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipGraphDestroy(clonedGraph));
}

/**
 * Test Description
 * ------------------------
 *    - Verify local node IDs restart for each graph:
 *        -# Create two graphs with two nodes each
 *        -# Verify both graphs assign local IDs 0 and 1
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphNodeGetLocalId.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.18
 */
HIP_TEST_CASE(Unit_hipGraphNodeGetLocalId_Positive_RestartsForEachGraph) {
  hipGraph_t graph1, graph2;
  HIP_CHECK(hipGraphCreate(&graph1, 0));
  HIP_CHECK(hipGraphCreate(&graph2, 0));

  hipGraphNode_t node1a, node1b, node2a, node2b;
  HIP_CHECK(hipGraphAddEmptyNode(&node1a, graph1, nullptr, 0));
  HIP_CHECK(hipGraphAddEmptyNode(&node1b, graph1, nullptr, 0));
  HIP_CHECK(hipGraphAddEmptyNode(&node2a, graph2, nullptr, 0));
  HIP_CHECK(hipGraphAddEmptyNode(&node2b, graph2, nullptr, 0));

  unsigned int id1a, id1b, id2a, id2b;
  HIP_CHECK(hipGraphNodeGetLocalId(node1a, &id1a));
  HIP_CHECK(hipGraphNodeGetLocalId(node1b, &id1b));
  HIP_CHECK(hipGraphNodeGetLocalId(node2a, &id2a));
  HIP_CHECK(hipGraphNodeGetLocalId(node2b, &id2b));

  REQUIRE(id1a == 0);
  REQUIRE(id1b == 1);
  REQUIRE(id2a == 0);
  REQUIRE(id2b == 1);

  HIP_CHECK(hipGraphDestroy(graph1));
  HIP_CHECK(hipGraphDestroy(graph2));
}

/**
 * Test Description
 * ------------------------
 *    - Verify node removal does not change or reuse local IDs:
 *        -# Create a graph with two nodes (A, B)
 *        -# Remove node A
 *        -# Verify node B keeps its ID and a new node C receives ID 2
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphNodeGetLocalId.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 7.18
 */
HIP_TEST_CASE(Unit_hipGraphNodeGetLocalId_Positive_RemovalPreservesAndAdvancesIds) {
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphNode_t nodeA, nodeB;
  HIP_CHECK(hipGraphAddEmptyNode(&nodeA, graph, nullptr, 0));
  HIP_CHECK(hipGraphAddEmptyNode(&nodeB, graph, nullptr, 0));

  unsigned int idBBefore;
  HIP_CHECK(hipGraphNodeGetLocalId(nodeB, &idBBefore));

  HIP_CHECK(hipGraphDestroyNode(nodeA));

  unsigned int idBAfter;
  HIP_CHECK(hipGraphNodeGetLocalId(nodeB, &idBAfter));

  hipGraphNode_t nodeC;
  HIP_CHECK(hipGraphAddEmptyNode(&nodeC, graph, nullptr, 0));

  unsigned int idC;
  HIP_CHECK(hipGraphNodeGetLocalId(nodeC, &idC));

  REQUIRE(idBBefore == 1);
  REQUIRE(idBAfter == idBBefore);
  REQUIRE(idC == 2);

  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * End doxygen group GraphTest.
 * @}
 */

#endif  // HT_AMD || CUDA_VERSION >= 13010
