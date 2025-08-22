/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>


#include "graph_dependency_common.hh"

/**
 * @addtogroup hipGraphGetEdges hipGraphGetEdges
 * @{
 * @ingroup GraphTest
 * `hipGraphGetEdges(hipGraph_t graph, hipGraphNode_t *from, hipGraphNode_t *to, size_t *numEdges)`
 * - returns a graph's dependency edges
 */

namespace {
inline constexpr size_t kNumOfEdges = 6;
}  // anonymous namespace

/**
 * Local Function to validate number of edges.
 */
static void validate_hipGraphGetEdges_fromto(size_t testNumEdges, GraphGetNodesTest test_type,
                                             std::vector<hipGraphNode_t>& nodes_from,
                                             std::vector<hipGraphNode_t>& nodes_to,
                                             hipGraph_t graph) {
  size_t numEdges = testNumEdges;
  hipGraphNode_t* fromnode = new hipGraphNode_t[numEdges]{};
  hipGraphNode_t* tonode = new hipGraphNode_t[numEdges]{};
  HIP_CHECK(hipGraphGetEdges(graph, fromnode, tonode, &numEdges));
  bool nodeFound;
  int found_count = 0;
  for (int idx_from = 0; idx_from < nodes_from.size(); idx_from++) {
    nodeFound = false;
    int idx = 0;
    for (; idx < numEdges; idx++) {
      if (nodes_from[idx_from] == fromnode[idx]) {
        nodeFound = true;
        break;
      }
    }
    if (nodeFound && (tonode[idx] == nodes_to[idx_from])) {
      found_count++;
    }
  }

  // Verify that the found number of edges is expected
  switch (test_type) {
    case GraphGetNodesTest::equalNumNodes:
      REQUIRE(found_count == nodes_from.size());
      break;
    case GraphGetNodesTest::lesserNumNodes:
      // Verify numEdges is unchanged
      REQUIRE(numEdges == testNumEdges);
      REQUIRE(found_count == testNumEdges);
      break;
    case GraphGetNodesTest::greaterNumNodes:
      // Verify numEdges is reset to actual number of nodes
      REQUIRE(numEdges == nodes_from.size());
      REQUIRE(found_count == nodes_from.size());
      // Verify additional entries in edges are set to nullptr
      for (auto idx = numEdges; idx < testNumEdges; idx++) {
        REQUIRE(fromnode[idx] == nullptr);
        REQUIRE(tonode[idx] == nullptr);
      }
  }

  delete[] tonode;
  delete[] fromnode;
}

/**
 * Test Description
 * ------------------------
 *    - Functional test to validate API for different number of edges:
 *        -# Validate number of edges
 *        -# Validate from/to list when numEdges = num of edges
 *        -# Validate from/to list when numEdges = less than num of edges
 *        -# Validate from/to list when numEdges = more than num of edges
 *        -# Validate number of edges when zero or one node in graph
 * Test source
 * ------------------------
 *    - catch\unit\graph\hipGraphGetEdges.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGraphGetEdges_Positive_Functional") {
  constexpr size_t N = 1024;
  hipGraph_t graph;
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;

  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);

  HIP_CHECK(hipGraphCreate(&graph, 0));

  std::vector<hipGraphNode_t> from_nodes;
  std::vector<hipGraphNode_t> to_nodes;
  std::vector<hipGraphNode_t> nodelist;
  graphNodesCommon(graph, A_h, A_d, B_h, B_d, C_h, C_d, N, from_nodes, to_nodes, nodelist);

  // Create dependencies
  HIP_CHECK(hipGraphAddDependencies(graph, &from_nodes[0], &to_nodes[0], 6));

  // Validate hipGraphGetEdges() API
  // Scenario 1
  SECTION("Validate number of edges") {
    size_t numEdges = 0;
    HIP_CHECK(hipGraphGetEdges(graph, nullptr, nullptr, &numEdges));
    REQUIRE(numEdges == kNumOfEdges);
  }
  // Scenario 2
  SECTION("Validate from/to list when numEdges = num of edges") {
    validate_hipGraphGetEdges_fromto(kNumOfEdges, GraphGetNodesTest::equalNumNodes, from_nodes,
                                     to_nodes, graph);
  }
  // Scenario 3
  SECTION("Validate from/to list when numEdges = less than num of edges") {
    validate_hipGraphGetEdges_fromto(kNumOfEdges - 1, GraphGetNodesTest::lesserNumNodes, from_nodes,
                                     to_nodes, graph);
  }
  // Scenario 4
  SECTION("Validate from/to list when numEdges = more than num of edges") {
    validate_hipGraphGetEdges_fromto(kNumOfEdges + 1, GraphGetNodesTest::greaterNumNodes,
                                     from_nodes, to_nodes, graph);
  }
  // Scenario 5
  SECTION("Validate number of edges when zero or one node in graph") {
    size_t numEdges = 0;
    hipGraph_t graphempty;
    HIP_CHECK(hipGraphCreate(&graphempty, 0));
    HIP_CHECK(hipGraphGetEdges(graphempty, nullptr, nullptr, &numEdges));
    REQUIRE(numEdges == 0);
    // Add an empty node
    hipGraphNode_t emptyNode{};
    HIP_CHECK(hipGraphAddEmptyNode(&emptyNode, graphempty, nullptr, 0));
    HIP_CHECK(hipGraphGetEdges(graphempty, nullptr, nullptr, &numEdges));
    REQUIRE(numEdges == 0);
    HIP_CHECK(hipGraphDestroy(graphempty));
  }
  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *    - Test to verify edges of created graph are matching the captured operations
 * Test source
 * ------------------------
 *    - catch\unit\graph\hipGraphGetEdges.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGraphGetEdges_Positive_CapturedStream") {
  hipGraph_t graph{nullptr};
  constexpr size_t N = 1024;
  constexpr int numMemcpy[2]{2, 3}, numKernel[2]{2, 3}, numMemset[2]{2, 0};
  int cntMemcpy[2]{}, cntKernel[2]{}, cntMemset[2]{};
  hipGraphNodeType nodeType;
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;

  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);

  // Create streams and events
  StreamsGuard streams(3);
  EventsGuard events(3);

  // Capture stream
  captureNodesCommon(graph, A_h, A_d, B_h, B_d, C_h, C_d, N, streams.stream_list(),
                     events.event_list());
  REQUIRE(graph != nullptr);

  size_t numEdges = 0;
  HIP_CHECK(hipGraphGetEdges(graph, nullptr, nullptr, &numEdges));
  REQUIRE(numEdges == kNumOfEdges);

  int numBytes = sizeof(hipGraphNode_t) * numEdges;
  hipGraphNode_t* from_nodes = reinterpret_cast<hipGraphNode_t*>(malloc(numBytes));
  REQUIRE(from_nodes != nullptr);
  hipGraphNode_t* to_nodes = reinterpret_cast<hipGraphNode_t*>(malloc(numBytes));
  REQUIRE(to_nodes != nullptr);

  HIP_CHECK(hipGraphGetEdges(graph, from_nodes, to_nodes, &numEdges));
  for (size_t i = 0; i < 2; i++) {
    hipGraphNode_t* current_nodes = (i == 0) ? from_nodes : to_nodes;
    for (size_t j = 0; j < numEdges; j++) {
      HIP_CHECK(hipGraphNodeGetType(current_nodes[j], &nodeType));
      switch (nodeType) {
        case hipGraphNodeTypeMemcpy:
          cntMemcpy[i]++;
          break;

        case hipGraphNodeTypeKernel:
          cntKernel[i]++;
          break;

        case hipGraphNodeTypeMemset:
          cntMemset[i]++;
          break;

        default:
          INFO("Unexpected nodetype returned : " << nodeType);
          REQUIRE(false);
      }
    }
    REQUIRE(cntMemcpy[i] == numMemcpy[i]);
    REQUIRE(cntKernel[i] == numKernel[i]);
    REQUIRE(cntMemset[i] == numMemset[i]);
  }

  HIP_CHECK(hipGraphDestroy(graph));
  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
}

/**
 * Test Description
 * ------------------------
 *    - Test to verify API behavior with invalid arguments:
 *        -# Null Graph
 *        -# Graph is uninitialized
 *        -# From is nullptr
 *        -# To is nullptr
 *        -# numEdges is nullptr
 * Test source
 * ------------------------
 *    - catch\unit\graph\hipGraphGetEdges.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGraphGetEdges_Negative_Parameters") {
  hipGraph_t graph{}, graph_uninit{};
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipGraphNode_t nodes_from[kNumOfEdges]{}, nodes_to[kNumOfEdges]{};

  hipEvent_t event_start, event_end;
  HIP_CHECK(hipEventCreateWithFlags(&event_start, hipEventDisableTiming));
  HIP_CHECK(hipEventCreateWithFlags(&event_end, hipEventDisableTiming));

  // create event record nodes
  hipGraphNode_t event_node_start, event_node_end;
  HIP_CHECK(hipGraphAddEventRecordNode(&event_node_start, graph, nullptr, 0, event_start));
  HIP_CHECK(hipGraphAddEventRecordNode(&event_node_end, graph, nullptr, 0, event_end));

  // Add dependency between nodes
  HIP_CHECK(hipGraphAddDependencies(graph, &event_node_start, &event_node_end, 1));

  size_t numEdges = 0;
  SECTION("graph is nullptr") {
    HIP_CHECK_ERROR(hipGraphGetEdges(nullptr, nodes_from, nodes_to, &numEdges),
                    hipErrorInvalidValue);
  }

  SECTION("graph is uninitialized") {
    HIP_CHECK_ERROR(hipGraphGetEdges(graph_uninit, nodes_from, nodes_to, &numEdges),
                    hipErrorInvalidValue);
  }

  SECTION("From is nullptr") {
    HIP_CHECK_ERROR(hipGraphGetEdges(graph, nullptr, nodes_to, &numEdges), hipErrorInvalidValue);
  }

  SECTION("To is nullptr") {
    HIP_CHECK_ERROR(hipGraphGetEdges(graph, nodes_from, nullptr, &numEdges), hipErrorInvalidValue);
  }

  SECTION("numEdges is nullptr") {
    HIP_CHECK_ERROR(hipGraphGetEdges(graph, nodes_from, nodes_to, nullptr), hipErrorInvalidValue);
  }

  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipEventDestroy(event_end));
  HIP_CHECK(hipEventDestroy(event_start));
}

/**
 * End doxygen group GraphTest.
 * @}
 */
