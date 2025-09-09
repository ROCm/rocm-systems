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
 * @addtogroup hipGraphRemoveDependencies hipGraphRemoveDependencies
 * @{
 * @ingroup GraphTest
 * `hipGraphRemoveDependencies(hipGraph_t graph, const hipGraphNode_t *from, const hipGraphNode_t
 * *to, size_t numDependencies)` - removes dependency edges from a graph
 */

namespace {
inline constexpr size_t kNumOfEdges = 6;
}  // anonymous namespace

/**
 * Kernel Functions to perform square and return in the same
 * input memory location.
 */
static __global__ void vector_square(int* A_d, size_t N_ELMTS) {
  size_t gputhread = (blockIdx.x * blockDim.x + threadIdx.x);
  size_t stride = blockDim.x * gridDim.x;
  int temp = 0;
  for (size_t i = gputhread; i < N_ELMTS; i += stride) {
    temp = A_d[i] * A_d[i];
    A_d[i] = temp;
  }
}

/**
 * Test Description
 * ------------------------
 *    - Functional Test for removing dependencies in manually created graph and verifying number of
 * edges:
 *        -# Remove some dependencies
 *            -# Node by Node
 *            -# Node lists
 *        -# Remove all dependencies
 * Test source
 * ------------------------
 *    - catch\unit\graph\hipGraphRemoveDependencies.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGraphRemoveDependencies_Positive_Functional") {
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

  size_t numEdgesExpected = kNumOfEdges;
  SECTION("Remove some dependencies") {
    // Remove some dependencies
    constexpr size_t numEdgesRemoved = 3;
    hipGraphNode_t expected_from_nodes[numEdgesRemoved] = {from_nodes[2], from_nodes[3],
                                                           from_nodes[4]};
    hipGraphNode_t expected_to_nodes[numEdgesRemoved] = {to_nodes[2], to_nodes[3], to_nodes[4]};

    SECTION("Node by Node") {
      HIP_CHECK(hipGraphRemoveDependencies(graph, &from_nodes[2], &to_nodes[2], 1));
      HIP_CHECK(hipGraphRemoveDependencies(graph, &from_nodes[3], &to_nodes[3], 1));
      HIP_CHECK(hipGraphRemoveDependencies(graph, &from_nodes[4], &to_nodes[4], 1));
    }
    SECTION("Node lists") {
      HIP_CHECK(hipGraphRemoveDependencies(graph, expected_from_nodes, expected_to_nodes,
                                           numEdgesRemoved));
    }

    // Validate manually with hipGraphGetEdges() API
    hipGraphNode_t fromnode[kNumOfEdges]{};
    hipGraphNode_t tonode[kNumOfEdges]{};
    size_t numEdges = kNumOfEdges;
    HIP_CHECK(hipGraphGetEdges(graph, fromnode, tonode, &numEdges));

    bool nodeFound;
    int found_count = 0;
    for (size_t idx_from = 0; idx_from < numEdgesRemoved; idx_from++) {
      nodeFound = false;
      int idx = 0;
      for (; idx < kNumOfEdges; idx++) {
        if (expected_from_nodes[idx_from] == fromnode[idx]) {
          nodeFound = true;
          break;
        }
      }
      if (nodeFound && (tonode[idx] == expected_to_nodes[idx_from])) {
        found_count++;
      }
    }
    // Ensure none of the nodes are discovered
    REQUIRE(0 == found_count);
    numEdgesExpected = kNumOfEdges - numEdgesRemoved;
  }
  SECTION("Remove all dependencies") {
    size_t numEdges = kNumOfEdges;
    hipGraphNode_t fromnode[kNumOfEdges]{};
    hipGraphNode_t tonode[kNumOfEdges]{};
    HIP_CHECK(hipGraphGetEdges(graph, fromnode, tonode, &numEdges));

    HIP_CHECK(hipGraphRemoveDependencies(graph, fromnode, tonode, numEdges));
    numEdgesExpected = 0;
  }

  // Validate with returned number of edges from hipGraphGetEdges() API
  size_t numEdges = 0;
  HIP_CHECK(hipGraphGetEdges(graph, nullptr, nullptr, &numEdges));
  REQUIRE(numEdgesExpected == numEdges);
  // Destroy
  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *    - Functional Test for removing dependencies in stream captured graph and verifying number of
 * edges:
 *        -# Remove some dependencies
 *        -# Remove all dependencies
 * Test source
 * ------------------------
 *    - catch\unit\graph\hipGraphRemoveDependencies.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGraphRemoveDependenciesPositive_CapturedStream") {
  hipGraph_t graph;
  constexpr size_t N = 1024;
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;

  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
  // Create streams and events
  StreamsGuard streams(3);
  EventsGuard events(3);

  // Capture stream
  captureNodesCommon(graph, A_h, A_d, B_h, B_d, C_h, C_d, N, streams.stream_list(),
                     events.event_list());

  hipGraphNode_t* nodes{nullptr};
  size_t numNodes = 0, numEdges = 0;
  HIP_CHECK(hipGraphGetNodes(graph, nodes, &numNodes));
  HIP_CHECK(hipGraphGetEdges(graph, nullptr, nullptr, &numEdges));
  REQUIRE(7 == numNodes);
  REQUIRE(kNumOfEdges == numEdges);
  // Get the edges and remove one edge. Verify edge is removed.
  hipGraphNode_t fromnode[kNumOfEdges]{};
  hipGraphNode_t tonode[kNumOfEdges]{};
  HIP_CHECK(hipGraphGetEdges(graph, fromnode, tonode, &numEdges));
  size_t expected_num_edges = kNumOfEdges;

  SECTION("Remove some dependencies") {
    HIP_CHECK(hipGraphRemoveDependencies(graph, &fromnode[0], &tonode[0], 1));
    HIP_CHECK(hipGraphRemoveDependencies(graph, &fromnode[1], &tonode[1], 1));
    HIP_CHECK(hipGraphRemoveDependencies(graph, &fromnode[2], &tonode[2], 1));
    expected_num_edges = 3;
  }
  SECTION("Remove all dependencies") {
    HIP_CHECK(hipGraphRemoveDependencies(graph, fromnode, tonode, numEdges));
    expected_num_edges = 0;
  }
  // Verify
  HIP_CHECK(hipGraphGetEdges(graph, nullptr, nullptr, &numEdges));
  REQUIRE(expected_num_edges == numEdges);
  // Destroy
  HIP_CHECK(hipGraphDestroy(graph));
  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
}

/**
 * Test Description
 * ------------------------
 *    - Dynamically modify dependencies in a graph and verify the computation:
 * Test source
 * ------------------------
 *    - catch\unit\graph\hipGraphRemoveDependencies.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGraphRemoveDependencies_Positive_ChangeComputeFunc") {
  hipStream_t streamForGraph;
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  hipGraph_t graph;
  hipGraphNode_t memcpyH2D_A, memcpyH2D_B, memcpyD2H_C;
  hipGraphNode_t kernel_vecAdd, kernel_square;
  hipKernelNodeParams kernelNodeParams{};
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;
  size_t NElem{N};

  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);

  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_A, graph, nullptr, 0, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_B, graph, nullptr, 0, B_d, B_h, Nbytes,
                                    hipMemcpyHostToDevice));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_C, graph, nullptr, 0, C_h, C_d, Nbytes,
                                    hipMemcpyDeviceToHost));

  void* kernelArgs2[] = {&A_d, &B_d, &C_d, reinterpret_cast<void*>(&NElem)};
  kernelNodeParams.func = reinterpret_cast<void*>(HipTest::vectorADD<int>);
  kernelNodeParams.gridDim = dim3(blocks);
  kernelNodeParams.blockDim = dim3(threadsPerBlock);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs2);
  kernelNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kernel_vecAdd, graph, nullptr, 0, &kernelNodeParams));

  // Create dependencies
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D_A, &kernel_vecAdd, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D_B, &kernel_vecAdd, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &kernel_vecAdd, &memcpyD2H_C, 1));
  // Instantiate and execute Graph
  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));
  // Validate
  bool bMismatch = false;
  for (size_t idx = 0; idx < NElem; idx++) {
    if (C_h[idx] != (A_h[idx] + B_h[idx])) {
      bMismatch = true;
      break;
    }
  }
  REQUIRE(false == bMismatch);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  // Remove dependency memcpyH2D_B -> kernel_vecAdd and
  // add new dependencies memcpyH2D_B -> kernel_square -> kernel_vecAdd
  // Square kernel
  void* kernelArgs1[] = {&B_d, reinterpret_cast<void*>(&NElem)};
  kernelNodeParams.func = reinterpret_cast<void*>(vector_square);
  kernelNodeParams.gridDim = dim3(blocks);
  kernelNodeParams.blockDim = dim3(threadsPerBlock);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs1);
  kernelNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kernel_square, graph, nullptr, 0, &kernelNodeParams));
  HIP_CHECK(hipGraphRemoveDependencies(graph, &memcpyH2D_B, &kernel_vecAdd, 1));
  // Add new dependencies
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D_B, &kernel_square, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &kernel_square, &kernel_vecAdd, 1));
  size_t numEdges = 0, numNodes = 0;
  HIP_CHECK(hipGraphGetEdges(graph, nullptr, nullptr, &numEdges));
  REQUIRE(4 == numEdges);
  HIP_CHECK(hipGraphGetNodes(graph, nullptr, &numNodes));
  REQUIRE(5 == numNodes);
  // Instantiate and execute graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));
  // Validate
  bMismatch = false;
  for (size_t idx = 0; idx < NElem; idx++) {
    if (C_h[idx] != (A_h[idx] + B_h[idx] * B_h[idx])) {
      bMismatch = true;
      break;
    }
  }
  REQUIRE(false == bMismatch);
  // Destroy
  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
}

/**
 * Test Description
 * ------------------------
 *    - Test to verify API behavior with special cases of valid arguments:
 *        -# numDependencies is zero, To/From are nullptr
 *        -# numDependencies is zero, To or From are nullptr
 *        -# numDependencies is zero, To/From are valid
 *        -# numDependencies is zero, To/From are the same
 *        -# numDependencies < To/From length
 * Test source
 * ------------------------
 *    - catch\unit\graph\hipGraphRemoveDependencies.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGraphRemoveDependencies_Positive_Parameters") {
  constexpr size_t Nbytes = 1024;
  hipGraphNode_t memcpyH2D_A;
  hipGraphNode_t memcpyD2H_A;
  hipGraphNode_t memset_A;
  hipMemsetParams memsetParams{};
  char* A_d;
  char* A_h;
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  HIP_CHECK(hipMalloc(&A_d, Nbytes));
  A_h = reinterpret_cast<char*>(malloc(Nbytes));

  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = reinterpret_cast<void*>(A_d);
  memsetParams.value = 0;
  memsetParams.pitch = 0;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = Nbytes;
  memsetParams.height = 1;

  HIP_CHECK(hipGraphAddMemsetNode(&memset_A, graph, nullptr, 0, &memsetParams));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_A, graph, nullptr, 0, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_A, graph, nullptr, 0, A_h, A_d, Nbytes,
                                    hipMemcpyDeviceToHost));

  HIP_CHECK(hipGraphAddDependencies(graph, &memset_A, &memcpyH2D_A, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D_A, &memcpyD2H_A, 1));
  size_t totalEdges = 2;
#if HT_NVIDIA  // EXSWHTEC-218
  SECTION("numDependencies is zero, To/From are nullptr") {
    HIP_CHECK(hipGraphRemoveDependencies(graph, nullptr, nullptr, 0));
  }
  SECTION("numDependencies is zero, To or From are nullptr") {
    HIP_CHECK(hipGraphRemoveDependencies(graph, &memcpyH2D_A, nullptr, 0));
    HIP_CHECK(hipGraphRemoveDependencies(graph, nullptr, &memcpyH2D_A, 0));
  }
#endif
  SECTION("numDependencies is zero, To/From are valid") {
    HIP_CHECK(hipGraphRemoveDependencies(graph, &memcpyH2D_A, &memcpyD2H_A, 0));
  }
  SECTION("numDependencies is zero, To/From are the same") {
    HIP_CHECK(hipGraphRemoveDependencies(graph, &memcpyH2D_A, &memcpyH2D_A, 0));
  }

  size_t numEdges = 0;
  HIP_CHECK(hipGraphGetEdges(graph, nullptr, nullptr, &numEdges));
  REQUIRE(totalEdges == numEdges);

  SECTION("numDependencies < To/From length") {
    size_t numDependencies = 0;
    hipGraphNode_t from_list[] = {memset_A, memcpyH2D_A};
    hipGraphNode_t to_list[] = {memcpyH2D_A, memcpyD2H_A};
    HIP_CHECK(hipGraphRemoveDependencies(graph, from_list, to_list, 1));
    HIP_CHECK(hipGraphNodeGetDependencies(memcpyH2D_A, nullptr, &numDependencies));
    REQUIRE(numDependencies == 0);
    HIP_CHECK(hipGraphNodeGetDependencies(memcpyD2H_A, nullptr, &numDependencies));
    REQUIRE(numDependencies == 1);
  }
  // Destroy
  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipGraphDestroy(graph));
  free(A_h);
}

/**
 * Test Description
 * ------------------------
 *    - Test to verify API behavior with invalid arguments:
 *        -# Null Graph
 *        -# Graph is uninitialized
 *        -# To or From is nullptr
 *        -# To/From are nullptr
 *        -# From belongs to different graph
 *        -# To belongs to different graph
 *        -# Remove non existing dependency
 *        -# Remove same dependency twice
 *        -# numDependencies > To/From length
 * Test source
 * ------------------------
 *    - catch\unit\graph\hipGraphRemoveDependencies.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGraphRemoveDependencies_Negative_Parameters") {
  hipGraph_t graph{};
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipEvent_t event_start, event_end;
  HIP_CHECK(hipEventCreateWithFlags(&event_start, hipEventDisableTiming));
  HIP_CHECK(hipEventCreateWithFlags(&event_end, hipEventDisableTiming));
  // memset node
  constexpr size_t Nbytes = 1024;
  char* A_d;
  hipGraphNode_t memset_A;
  hipMemsetParams memsetParams{};
  HIP_CHECK(hipMalloc(&A_d, Nbytes));
  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = reinterpret_cast<void*>(A_d);
  memsetParams.value = 0;
  memsetParams.pitch = 0;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = Nbytes;
  memsetParams.height = 1;
  HIP_CHECK(hipGraphAddMemsetNode(&memset_A, graph, nullptr, 0, &memsetParams));
  // create event record node
  hipGraphNode_t event_node_start, event_node_end;
  HIP_CHECK(hipGraphAddEventRecordNode(&event_node_start, graph, nullptr, 0, event_start));
  HIP_CHECK(hipGraphAddEventRecordNode(&event_node_end, graph, nullptr, 0, event_end));
  // Add dependencies between nodes
  HIP_CHECK(hipGraphAddDependencies(graph, &event_node_start, &memset_A, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memset_A, &event_node_end, 1));

  SECTION("graph is nullptr") {
    HIP_CHECK_ERROR(hipGraphRemoveDependencies(nullptr, &event_node_start, &memset_A, 1),
                    hipErrorInvalidValue);
  }

  SECTION("graph is uninitialized") {
    hipGraph_t graph_uninit{};
    HIP_CHECK_ERROR(hipGraphRemoveDependencies(graph_uninit, &event_node_start, &memset_A, 1),
                    hipErrorInvalidValue);
  }

  SECTION("To or From is nullptr") {
    HIP_CHECK_ERROR(hipGraphRemoveDependencies(graph, nullptr, &memset_A, 1), hipErrorInvalidValue);
    HIP_CHECK_ERROR(hipGraphRemoveDependencies(graph, &event_node_start, nullptr, 1),
                    hipErrorInvalidValue);
  }

  SECTION("To/From are nullptr") {
    HIP_CHECK_ERROR(hipGraphRemoveDependencies(graph, nullptr, nullptr, 1), hipErrorInvalidValue);
  }
#if HT_NVIDIA  // EXSWHTEC-218
  SECTION("To/From belong to different graph") {
    hipGraph_t graph1;
    hipGraphNode_t emptyNode1{};
    hipGraphNode_t emptyNode2{};
    HIP_CHECK(hipGraphCreate(&graph1, 0));
    // create empty node
    HIP_CHECK(hipGraphAddEmptyNode(&emptyNode1, graph1, nullptr, 0));
    HIP_CHECK(hipGraphAddEmptyNode(&emptyNode2, graph1, nullptr, 0));
    HIP_CHECK(hipGraphAddDependencies(graph1, &emptyNode1, &emptyNode2, 1));
    HIP_CHECK_ERROR(hipGraphRemoveDependencies(graph, &emptyNode1, &emptyNode2, 1),
                    hipErrorInvalidValue);
    HIP_CHECK(hipGraphDestroy(graph1));
  }
#endif
  SECTION("Remove non existing dependency") {
    HIP_CHECK_ERROR(hipGraphRemoveDependencies(graph, &event_node_start, &event_node_end, 1),
                    hipErrorInvalidValue);
  }

  SECTION("Remove same dependency twice") {
    HIP_CHECK(hipGraphRemoveDependencies(graph, &event_node_start, &memset_A, 1));
    HIP_CHECK_ERROR(hipGraphRemoveDependencies(graph, &event_node_start, &memset_A, 1),
                    hipErrorInvalidValue);
  }

  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipEventDestroy(event_end));
  HIP_CHECK(hipEventDestroy(event_start));
}

/**
 * End doxygen group GraphTest.
 * @}
 */
