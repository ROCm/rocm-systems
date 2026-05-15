/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <functional>

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>


#include "graph_dependency_common.hh"

/**
 * @addtogroup hipGraphGetRootNodes hipGraphGetRootNodes
 * @{
 * @ingroup GraphTest
 * `hipGraphGetRootNodes(hipGraph_t graph, hipGraphNode_t *nodes, size_t *numNodes)` -
 * returns graph's root nodes
 */

namespace {
inline constexpr size_t kNumOfRootNodes = 3;
}  // anonymous namespace

/**
 * Test Description
 * ------------------------
 *    - Functional test to validate API for different number of root nodes:
 *        -# Validate number of root nodes
 *        -# Validate root node list when numRootNodes = num of root nodes
 *        -# Validate root node list when numRootNodes < num of root nodes
 *        -# Validate root node list when numRootNodes > num of root nodes
 *        -# Validate numRootNodes is 0 when no nodes in graph
 * Test source
 * ------------------------
 *    - catch\unit\graph\hipGraphGetRootNodes.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGraphGetRootNodes_Positive_Functional") {
  using namespace std::placeholders;
  constexpr size_t N = 1024;
  hipGraph_t graph;
  hipStream_t streamForGraph;
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;
  hipGraphExec_t graphExec;

  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);

  HIP_CHECK(hipGraphCreate(&graph, 0));

  std::vector<hipGraphNode_t> from_nodes;
  std::vector<hipGraphNode_t> to_nodes;
  std::vector<hipGraphNode_t> nodelist;
  graphNodesCommon(graph, A_h, A_d, B_h, B_d, C_h, C_d, N, from_nodes, to_nodes, nodelist);

  // Create dependencies
  HIP_CHECK(hipGraphAddDependencies(graph, &from_nodes[0], &to_nodes[0], 6));

  std::vector<hipGraphNode_t> rootnodelist = {nodelist[0], nodelist[1], nodelist[2]};

  size_t numRootNodes{};
  // Get numRootNodes by passing rootnodes list as nullptr.
  // verify : numRootNodes is set to actual number of root nodes added
  // Scenario 1
  SECTION("Validate number of rootnodes") {
    HIP_CHECK(hipGraphGetRootNodes(graph, nullptr, &numRootNodes));
    INFO("Num of nodes returned by GetRootNodes : " << numRootNodes);
    REQUIRE(numRootNodes == rootnodelist.size());
  }

  // Scenario 2
  SECTION("Validate root node list when numRootNodes = num of root nodes") {
    validateGraphNodesCommon(std::bind(hipGraphGetRootNodes, graph, _1, _2), rootnodelist,
                             kNumOfRootNodes, GraphGetNodesTest::equalNumNodes);
  }

  // Scenario 3
  SECTION("Validate root node list when numRootNodes < num of root nodes") {
    validateGraphNodesCommon(std::bind(hipGraphGetRootNodes, graph, _1, _2), rootnodelist,
                             kNumOfRootNodes - 1, GraphGetNodesTest::lesserNumNodes);
  }

  // Scenario 4
  SECTION("Validate root node list when numRootNodes > num of root nodes") {
    validateGraphNodesCommon(std::bind(hipGraphGetRootNodes, graph, _1, _2), rootnodelist,
                             kNumOfRootNodes + 1, GraphGetNodesTest::greaterNumNodes);
  }

  // Scenario 5
  SECTION("Validate numRootNodes is 0 when no nodes in graph") {
    hipGraph_t emptyGraph{};
    HIP_CHECK(hipGraphCreate(&emptyGraph, 0));
    HIP_CHECK(hipGraphGetRootNodes(emptyGraph, nullptr, &numRootNodes));
    REQUIRE(numRootNodes == 0);
    HIP_CHECK(hipGraphDestroy(emptyGraph));
  }

  // Instantiate and launch the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  // Verify graph execution result
  HipTest::checkVectorADD(A_h, B_h, C_h, N);

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
}

/**
 * Test Description
 * ------------------------
 *    - - Test to verify root nodes of created graph are matching the captured operations
 * Test source
 * ------------------------
 *    - catch\unit\graph\hipGraphGetRootNodes.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGraphGetRootNodes_Positive_CapturedStream") {
  hipStream_t streamForGraph{nullptr};
  hipGraph_t graph{nullptr};
  hipGraphExec_t graphExec{nullptr};
  constexpr size_t N = 1000000;
  constexpr int expectedRootNodes = 3;
  size_t numRootNodes{};
  float *A_d, *B_d, *C_d;
  float *A_h, *B_h, *C_h;

  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
  HIP_CHECK(hipStreamCreate(&streamForGraph));

  // Initialize input buffer
  for (size_t i = 0; i < N; ++i) {
    A_h[i] = 3.146f + i;  // Pi
    B_h[i] = 3.146f + i;  // Pi
  }

  // Create streams and events
  StreamsGuard streams(3);
  EventsGuard events(3);

  // Capture stream
  captureNodesCommon(graph, A_h, A_d, B_h, B_d, C_h, C_d, N, streams.stream_list(),
                     events.event_list());
  REQUIRE(graph != nullptr);

  // Verify numof root nodes
  HIP_CHECK(hipGraphGetRootNodes(graph, nullptr, &numRootNodes));
  REQUIRE(numRootNodes == expectedRootNodes);
  INFO("Num of nodes returned by GetRootNodes : " << numRootNodes);

  int numBytes = sizeof(hipGraphNode_t) * numRootNodes;
  hipGraphNode_t* nodes = reinterpret_cast<hipGraphNode_t*>(malloc(numBytes));
  REQUIRE(nodes != nullptr);

  HIP_CHECK(hipGraphGetRootNodes(graph, nodes, &numRootNodes));
  REQUIRE(numRootNodes == expectedRootNodes);

#if HT_NVIDIA  // EXSWHTEC-225
  // Verify root nodes have correct type.
  hipGraphNodeType nodeType;
  HIP_CHECK(hipGraphNodeGetType(nodes[0], &nodeType));
  REQUIRE(nodeType == hipGraphNodeTypeMemset);
  HIP_CHECK(hipGraphNodeGetType(nodes[1], &nodeType));
  REQUIRE(nodeType == hipGraphNodeTypeMemset);
  HIP_CHECK(hipGraphNodeGetType(nodes[2], &nodeType));
  REQUIRE(nodeType == hipGraphNodeTypeKernel);
#endif

  // Instantiate and launch the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  // Validate the computation
  for (size_t i = 0; i < N; i++) {
    if (C_h[i] != A_h[i] + B_h[i]) {
      INFO("C not matching at " << i << " C_h[i] " << C_h[i] << " A_h[i] + B_h[i] "
                                << A_h[i] + B_h[i]);
      REQUIRE(false);
    }
  }

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  free(nodes);
}

/**
 * Test Description
 * ------------------------
 *    - Test to verify API behavior with invalid arguments:
 *        -# Null Graph
 *        -# Graph is uninitialized
 *        -# numRootNodes as nullptr
 * Test source
 * ------------------------
 *    - catch\unit\graph\hipGraphGetRootNodes.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGraphGetRootNodes_Negative_Parameters") {
  hipGraph_t graph{nullptr};
  size_t numRootNodes{0};

  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipEvent_t event_start, event_end;
  HIP_CHECK(hipEventCreateWithFlags(&event_start, hipEventDisableTiming));
  HIP_CHECK(hipEventCreateWithFlags(&event_end, hipEventDisableTiming));

  // create event record nodes
  hipGraphNode_t event_node_start, event_node_end;
  HIP_CHECK(hipGraphAddEventRecordNode(&event_node_start, graph, nullptr, 0, event_start));
  HIP_CHECK(hipGraphAddEventRecordNode(&event_node_end, graph, nullptr, 0, event_end));

  HIP_CHECK(hipGraphGetRootNodes(graph, nullptr, &numRootNodes));
  INFO("Num of nodes returned by GetRootNodes : " << numRootNodes);
  int numBytes = sizeof(hipGraphNode_t) * numRootNodes;
  hipGraphNode_t* nodes = reinterpret_cast<hipGraphNode_t*>(malloc(numBytes));
  REQUIRE(nodes != nullptr);

  SECTION("graph as nullptr") {
    HIP_CHECK_ERROR(hipGraphGetRootNodes(nullptr, nodes, &numRootNodes), hipErrorInvalidValue);
  }

  SECTION("graph is uninitialized") {
    hipGraph_t graph_uninit{};
    HIP_CHECK_ERROR(hipGraphGetRootNodes(graph_uninit, nodes, &numRootNodes), hipErrorInvalidValue);
  }

  SECTION("numRootNodes as nullptr") {
    HIP_CHECK_ERROR(hipGraphGetRootNodes(graph, nodes, nullptr), hipErrorInvalidValue);
  }

  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipEventDestroy(event_end));
  HIP_CHECK(hipEventDestroy(event_start));
  free(nodes);
}

/**
 * End doxygen group GraphTest.
 * @}
 */
