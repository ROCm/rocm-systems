/*
Copyright (c) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
 * @addtogroup hipGraphExecUpdate hipGraphExecUpdate
 * @{
 * @ingroup GraphTest
 * `hipGraphExecUpdate(hipGraphExec_t hGraphExec, hipGraph_t hGraph,
 *                     hipGraphExecUpdateResultInfo* resultInfo)` -
 * Check whether an executable graph can be updated with a graph
 * and perform the update if possible.
 */

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>

/**
 * Test Description
 * ------------------------
 *  - Test verifies hipGraphExecUpdate API Negative nullptr check scenarios.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphExecUpdate.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipGraphExecUpdate_Negative_Basic") {
  hipError_t ret;
  hipGraph_t graph{};
  hipGraphExec_t graphExec{};
  hipGraphNode_t hErrorNode_out{};
  hipGraphExecUpdateResult updateResult_out{};
  SECTION("Pass hGraphExec as nullptr") {
    ret = hipGraphExecUpdate(nullptr, graph, &hErrorNode_out, &updateResult_out);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass hGraph as nullptr") {
    ret = hipGraphExecUpdate(graphExec, nullptr, &hErrorNode_out, &updateResult_out);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass hErrorNode_out as nullptr") {
    ret = hipGraphExecUpdate(graphExec, graph, nullptr, &updateResult_out);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass updateResult_out as nullptr") {
    ret = hipGraphExecUpdate(graphExec, graph, &hErrorNode_out, nullptr);
    REQUIRE(hipErrorInvalidValue == ret);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Test verifies hipGraphExecUpdate API Negative scenarios.
 *    When the a graphExec was updated with with different type of node
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphExecUpdate.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */

TEST_CASE("Unit_hipGraphExecUpdate_Negative_TypeChange") {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(char);
  constexpr size_t val = 0;
  char* devData;
  int *A_d, *A_h;
  HipTest::initArrays<int>(&A_d, nullptr, nullptr, &A_h, nullptr, nullptr, N, false);
  HIP_CHECK(hipMalloc(&devData, Nbytes));
  hipGraph_t graph, graph2;
  hipGraphExec_t graphExec;
  hipStream_t streamForGraph;
  hipGraphNode_t memsetNode, memcpy_A, hErrorNode_out;
  hipError_t ret;
  hipGraphExecUpdateResult updateResult_out;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  hipMemsetParams memsetParams{};
  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = reinterpret_cast<void*>(devData);
  memsetParams.value = val;
  memsetParams.pitch = 0;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = Nbytes;
  memsetParams.height = 1;
  HIP_CHECK(hipGraphAddMemsetNode(&memsetNode, graph, nullptr, 0, &memsetParams));
  std::vector<hipGraphNode_t> dependencies;
  dependencies.push_back(memsetNode);
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphCreate(&graph2, 0));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_A, graph2, nullptr, 0, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));
  // graphExec was created before memcpyTemp was added to graph.
  ret = hipGraphExecUpdate(graphExec, graph2, &hErrorNode_out, &updateResult_out);
  REQUIRE(hipGraphExecUpdateErrorNodeTypeChanged == updateResult_out);
  REQUIRE(hipErrorGraphExecUpdateFailure == ret);
  HIP_CHECK(hipFree(devData));
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipGraphDestroy(graph2));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
}

/**
 * Test Description
 * ------------------------
 *  - Test verifies hipGraphExecUpdate API Negative scenarios.
 *    When the count of nodes differ in hGraphExec and hGraph
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphExecUpdate.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */

TEST_CASE("Unit_hipGraphExecUpdate_Negative_CountDiffer") {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;
  size_t NElem{N};
  int* hData = reinterpret_cast<int*>(malloc(Nbytes));
  REQUIRE(hData != nullptr);
  memset(hData, 0, Nbytes);
  hipGraphNode_t memcpy_A, memcpy_B, memcpy_C, memcpyTemp;
  hipGraphNode_t kernel_vecAdd;
  hipKernelNodeParams kernelNodeParams{};
  hipError_t ret;
  hipGraph_t graph1, graph2, graph3;
  hipGraphExec_t graphExec1, graphExec2;
  hipStream_t streamForGraph;
  hipGraphNode_t hErrorNode_out;
  hipGraphExecUpdateResult updateResult_out;
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  HIP_CHECK(hipGraphCreate(&graph1, 0));
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_A, graph1, nullptr, 0, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_B, graph1, nullptr, 0, B_d, B_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_C, graph1, nullptr, 0, C_h, C_d, Nbytes,
                                    hipMemcpyDeviceToHost));
  void* kernelArgs[] = {&A_d, &B_d, &C_d, reinterpret_cast<void*>(&NElem)};
  kernelNodeParams.func = reinterpret_cast<void*>(HipTest::vectorADD<int>);
  kernelNodeParams.gridDim = dim3(blocks);
  kernelNodeParams.blockDim = dim3(threadsPerBlock);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
  kernelNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kernel_vecAdd, graph1, nullptr, 0, &kernelNodeParams));
  // Create dependencies
  HIP_CHECK(hipGraphAddDependencies(graph1, &memcpy_A, &kernel_vecAdd, 1));
  HIP_CHECK(hipGraphAddDependencies(graph1, &memcpy_B, &kernel_vecAdd, 1));
  HIP_CHECK(hipGraphAddDependencies(graph1, &kernel_vecAdd, &memcpy_C, 1));
  // Create a cloned graph and added extra node to it
  HIP_CHECK(hipGraphClone(&graph2, graph1));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyTemp, graph2, nullptr, 0, C_h, C_d, Nbytes,
                                    hipMemcpyDeviceToHost));
  HIP_CHECK(hipGraphInstantiate(&graphExec1, graph1, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphInstantiate(&graphExec2, graph2, nullptr, nullptr, 0));
  SECTION("When a node deleted from Graph but not from its pair GraphExec") {
    ret = hipGraphExecUpdate(graphExec2, graph1, &hErrorNode_out, &updateResult_out);
    REQUIRE(hipErrorGraphExecUpdateFailure == ret);
  }
  SECTION("When a node deleted from GraphExec but not from its pair Graph") {
    ret = hipGraphExecUpdate(graphExec1, graph2, &hErrorNode_out, &updateResult_out);
    REQUIRE(hipErrorGraphExecUpdateFailure == ret);
  }
  SECTION("When the dependent nodes of a pair differ") {
    HIP_CHECK(hipGraphCreate(&graph3, 0));
    HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_A, graph3, nullptr, 0, A_d, A_h, Nbytes,
                                      hipMemcpyHostToDevice));
    HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_B, graph3, nullptr, 0, B_d, B_h, Nbytes,
                                      hipMemcpyHostToDevice));
    HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_C, graph3, nullptr, 0, C_h, C_d, Nbytes,
                                      hipMemcpyDeviceToHost));
    HIP_CHECK(hipGraphAddKernelNode(&kernel_vecAdd, graph3, nullptr, 0, &kernelNodeParams));
    // Create dependencies
    HIP_CHECK(hipGraphAddDependencies(graph3, &memcpy_A, &kernel_vecAdd, 1));
    HIP_CHECK(hipGraphAddDependencies(graph3, &memcpy_B, &kernel_vecAdd, 1));
    HIP_CHECK(hipGraphAddDependencies(graph3, &memcpy_C, &kernel_vecAdd, 1));
    ret = hipGraphExecUpdate(graphExec1, graph3, &hErrorNode_out, &updateResult_out);
    REQUIRE(hipErrorGraphExecUpdateFailure == ret);
    HIP_CHECK(hipGraphDestroy(graph3));
  }
  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec1));
  HIP_CHECK(hipGraphExecDestroy(graphExec2));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
  HIP_CHECK(hipGraphDestroy(graph1));
  HIP_CHECK(hipGraphDestroy(graph2));
  free(hData);
}

/**
 * Test Description
 * ------------------------
 *  - Functional Scenario -
    1) Make a clone of the created graph and update the executable-graph from a clone graph.
    2) Update the executable-graph from a graph and make sure they are taking effect.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphExecUpdate.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */

TEST_CASE("Unit_hipGraphExecUpdate_Functional") {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;
  size_t NElem{N};
  int* hData = reinterpret_cast<int*>(malloc(Nbytes));
  REQUIRE(hData != nullptr);
  memset(hData, 0, Nbytes);
  hipGraphNode_t memcpy_A, memcpy_B, memcpy_C;
  hipGraphNode_t kernel_vecAdd, kernel_vecSquare;
  hipKernelNodeParams kernelNodeParams{};
  hipGraph_t graph, graph2, clonedgraph{};
  hipGraphExec_t graphExec;
  hipStream_t streamForGraph;
  hipGraphNode_t hErrorNode_out;
  hipGraphExecUpdateResult updateResult_out;
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_A, graph, nullptr, 0, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_B, graph, nullptr, 0, B_d, B_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_C, graph, nullptr, 0, C_h, C_d, Nbytes,
                                    hipMemcpyDeviceToHost));
  void* kernelArgs[] = {&A_d, &B_d, &C_d, reinterpret_cast<void*>(&NElem)};
  kernelNodeParams.func = reinterpret_cast<void*>(HipTest::vector_square<int>);
  kernelNodeParams.gridDim = dim3(blocks);
  kernelNodeParams.blockDim = dim3(threadsPerBlock);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
  kernelNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kernel_vecSquare, graph, nullptr, 0, &kernelNodeParams));
  // Create dependencies
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpy_A, &kernel_vecSquare, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpy_B, &kernel_vecSquare, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &kernel_vecSquare, &memcpy_C, 1));
  // Instantiate and launch the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  SECTION("Update graphExec with clone graph") {
    HIP_CHECK(hipGraphClone(&clonedgraph, graph));
    HIP_CHECK(hipGraphExecUpdate(graphExec, clonedgraph, &hErrorNode_out, &updateResult_out));
  }
  // Code for new graph creation with samilar node setup
  HIP_CHECK(hipGraphCreate(&graph2, 0));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_A, graph2, nullptr, 0, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_B, graph2, nullptr, 0, B_d, B_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_C, graph2, nullptr, 0, C_h, C_d, Nbytes,
                                    hipMemcpyDeviceToHost));
  HIP_CHECK(hipGraphMemcpyNodeSetParams1D(memcpy_C, hData, C_d, Nbytes, hipMemcpyDeviceToHost));
  memset(&kernelNodeParams, 0, sizeof(hipKernelNodeParams));
  void* kernelArgs2[] = {&A_d, &B_d, &C_d, reinterpret_cast<void*>(&NElem)};
  kernelNodeParams.func = reinterpret_cast<void*>(HipTest::vectorADD<int>);
  kernelNodeParams.gridDim = dim3(blocks);
  kernelNodeParams.blockDim = dim3(threadsPerBlock);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs2);
  kernelNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kernel_vecAdd, graph2, nullptr, 0, &kernelNodeParams));
  // Create dependencies
  HIP_CHECK(hipGraphAddDependencies(graph2, &memcpy_A, &kernel_vecAdd, 1));
  HIP_CHECK(hipGraphAddDependencies(graph2, &memcpy_B, &kernel_vecAdd, 1));
  HIP_CHECK(hipGraphAddDependencies(graph2, &kernel_vecAdd, &memcpy_C, 1));
  // Update the graphExec graph from graph -> graph2
  HIP_CHECK(hipGraphExecUpdate(graphExec, graph2, &hErrorNode_out, &updateResult_out));
  REQUIRE(updateResult_out == hipGraphExecUpdateSuccess);
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));
  // Verify graph execution result
  HipTest::checkVectorADD(A_h, B_h, hData, N);
  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipGraphDestroy(graph2));
  HIP_CHECK(hipGraphDestroy(clonedgraph));
  free(hData);
}

/**
 * Test Description
 * ------------------------
 *  - Functional Basic Check Scenario - 1
      Create a graph1 with memcpy1D node with direction as hipMemcpyHostToDevice
      Create a graph2 with memcpy1D node with direction as hipMemcpyHostToDevice
      Update graphExec1 with graph2 and verify. It should not return error.
    - Negative Scenario - 2
      Create a graph1 with memcpy1D node with direction as hipMemcpyHostToDevice
      Instantiate graph1 in graphExec1
      Create a graph2 with memcpy1D node with direction as hipMemcpyDeviceToHost
      Update graphExec1 with graph2 and verify.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphExecUpdate.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */

TEST_CASE("Unit_hipGraphExecUpdate_Negative_Functional_ParametersChanged") {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  int *A_d, *B_d, *C_d, *A_h, *B_h, *C_h;
  hipGraphNode_t memcpy_A, memcpy_B;
  hipError_t ret;
  hipGraph_t graph1, graph2, graph3;
  hipGraphExec_t graphExec1;
  hipGraphNode_t hErrorNode_out;
  hipGraphExecUpdateResult updateResult_out;
  HipTest::initArrays<int>(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
  HIP_CHECK(hipGraphCreate(&graph1, 0));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_A, graph1, nullptr, 0, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphInstantiate(&graphExec1, graph1, nullptr, nullptr, 0));
  SECTION("Update graphExec with similar graph and verify") {
    HIP_CHECK(hipGraphCreate(&graph2, 0));
    HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_B, graph2, nullptr, 0, B_d, B_h, Nbytes,
                                      hipMemcpyHostToDevice));
    ret = hipGraphExecUpdate(graphExec1, graph2, &hErrorNode_out, &updateResult_out);
    REQUIRE(hipSuccess == ret);
    HIP_CHECK(hipGraphDestroy(graph2));
  }
  SECTION("Update graphExec with similar graph and verify") {
    HIP_CHECK(hipGraphCreate(&graph3, 0));
    HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_B, graph3, nullptr, 0, B_h, B_d, Nbytes,
                                      hipMemcpyDeviceToHost));
    ret = hipGraphExecUpdate(graphExec1, graph3, &hErrorNode_out, &updateResult_out);

    REQUIRE(hipErrorGraphExecUpdateFailure == ret);
    REQUIRE(hipGraphExecUpdateErrorParametersChanged == updateResult_out);
    REQUIRE(memcpy_B == hErrorNode_out);
    HIP_CHECK(hipGraphDestroy(graph3));
  }
  HipTest::freeArrays<int>(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec1));
  HIP_CHECK(hipGraphDestroy(graph1));
}

/**
 * Test Description
 * ------------------------
 *  - Negative Scenario - 3
      Create graph1 and graph2 with different number node in it.
      Instantiate graph1 in graphExec1
      Update graphExec1 with graph2 and verify.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphExecUpdate.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */

TEST_CASE("Unit_hipGraphExecUpdate_Negative_Functional_CountDiffer_1") {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  int *A_d, *B_d, *C_d, *A_h, *B_h, *C_h;
  hipGraphNode_t memcpy_A, memcpy_B, memcpy_C;
  hipError_t ret;
  hipGraph_t graph1, graph2;
  hipGraphExec_t graphExec1;
  hipGraphNode_t hErrorNode_out;
  hipGraphExecUpdateResult updateResult_out;
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
  HIP_CHECK(hipGraphCreate(&graph1, 0));
  HIP_CHECK(hipGraphCreate(&graph2, 0));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_A, graph1, nullptr, 0, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_B, graph1, nullptr, 0, B_d, B_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphInstantiate(&graphExec1, graph1, nullptr, nullptr, 0));
  // When count of nodes directly differ in graphExec1 and graph2
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_C, graph2, nullptr, 0, C_h, C_d, Nbytes,
                                    hipMemcpyDeviceToHost));
  ret = hipGraphExecUpdate(graphExec1, graph2, &hErrorNode_out, &updateResult_out);

  REQUIRE(hipErrorGraphExecUpdateFailure == ret);
  REQUIRE(hipGraphExecUpdateErrorTopologyChanged == updateResult_out);
  REQUIRE(NULL == hErrorNode_out);

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec1));
  HIP_CHECK(hipGraphDestroy(graph1));
  HIP_CHECK(hipGraphDestroy(graph2));
}

/**
 * Test Description
 * ------------------------
 *  - Negative Scenario -
   4) Create a graph1 with 2 node and hipGraphInstantiate to create graphExec1 from it.
      Delete a node from the Graph but not from its graphExec1
      Update graphExec1 with same graph (where node is deleted) and verify.
   5) Create a graph2 with 1 node and hipGraphInstantiate to create graphExec2 from it.
     (Now graph1 and Graph2 have 1 node each with similar topology)
     Update graphExec2 with graph1 (where node is deleted) and verify.
   6) Create a graph with 1 node & hipGraphInstantiate to create graphExec from it
      Add one more node to the Graph Update graphExec with same graph
   - (A node is deleted in hGraphExec but not its pair from hGraph) and verify
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphExecUpdate.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */

TEST_CASE("Unit_hipGraphExecUpdate_Negative_Functional_CountDiffer_2") {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  int *A_d, *B_d, *C_d, *A_h, *B_h, *C_h;
  hipGraphNode_t memcpy_A, memcpy_B, memcpy_C;
  hipError_t ret;
  hipGraph_t graph1, graph2, graph3;
  hipGraphExec_t graphExec1, graphExec2, graphExec3;
  hipGraphNode_t hErrorNode_out;
  hipGraphExecUpdateResult updateResult_out;
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
  HIP_CHECK(hipGraphCreate(&graph1, 0));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_A, graph1, nullptr, 0, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_B, graph1, nullptr, 0, B_d, B_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphInstantiate(&graphExec1, graph1, nullptr, nullptr, 0));
  // Delete a node from the graph
  HIP_CHECK(hipGraphDestroyNode(memcpy_B));
  SECTION("When a node deleted from Graph but not from its pair GraphExec") {
    ret = hipGraphExecUpdate(graphExec1, graph1, &hErrorNode_out, &updateResult_out);
    REQUIRE(hipErrorGraphExecUpdateFailure == ret);
    REQUIRE(hipGraphExecUpdateErrorTopologyChanged == updateResult_out);
#if HT_NVIDIA
    REQUIRE(NULL == hErrorNode_out);
#endif
  }
  SECTION("Update the GraphExec with similar graph where a node get deleted") {
    HIP_CHECK(hipGraphCreate(&graph2, 0));
    HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_C, graph2, nullptr, 0, C_d, C_h, Nbytes,
                                      hipMemcpyHostToDevice));
    HIP_CHECK(hipGraphInstantiate(&graphExec2, graph2, nullptr, nullptr, 0));
    ret = hipGraphExecUpdate(graphExec2, graph1, &hErrorNode_out, &updateResult_out);
    REQUIRE(hipSuccess == ret);
    HIP_CHECK(hipGraphExecDestroy(graphExec2));
    HIP_CHECK(hipGraphDestroy(graph2));
  }
  SECTION("When A node is deleted in GraphExec but not its pair from Graph") {
    HIP_CHECK(hipGraphCreate(&graph3, 0));
    HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_A, graph3, nullptr, 0, A_d, A_h, Nbytes,
                                      hipMemcpyHostToDevice));
    HIP_CHECK(hipGraphInstantiate(&graphExec3, graph3, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_B, graph3, nullptr, 0, B_d, B_h, Nbytes,
                                      hipMemcpyHostToDevice));
    ret = hipGraphExecUpdate(graphExec3, graph3, &hErrorNode_out, &updateResult_out);
    REQUIRE(hipErrorGraphExecUpdateFailure == ret);
    REQUIRE(hipGraphExecUpdateErrorTopologyChanged == updateResult_out);
    REQUIRE(NULL == hErrorNode_out);

    HIP_CHECK(hipGraphExecDestroy(graphExec3));
    HIP_CHECK(hipGraphDestroy(graph3));
  }
  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec1));
  HIP_CHECK(hipGraphDestroy(graph1));
}

/**
 * Test Description
 * ------------------------
 *  - Negative Scenario -
   7) Create a graph1 with memcpy_A, memcpy_B and memcpy_C,
      add dependency as memcpy_A->memcpy_B->memcpy_C
      and hipGraphInstantiate to create graphExec from it
      Create a graph2 with same nodes and
      dependency as memcpy_A->memcpy_C and memcpy_B->memcpy_C
      and Update graphExec with graph2 and verify
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphExecUpdate.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */

TEST_CASE("Unit_hipGraphExecUpdate_Negative_Dependent_NodesDiffer") {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  int *A_d, *B_d, *C_d, *A_h, *B_h, *C_h;
  hipGraphNode_t memcpy_A, memcpy_B, memcpy_C;
  hipError_t ret;
  hipGraph_t graph1, graph2;
  hipGraphExec_t graphExec;
  hipGraphNode_t hErrorNode_out;
  hipGraphExecUpdateResult updateResult_out;
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
  HIP_CHECK(hipGraphCreate(&graph1, 0));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_A, graph1, nullptr, 0, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_B, graph1, nullptr, 0, B_d, B_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_C, graph1, nullptr, 0, C_d, C_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddDependencies(graph1, &memcpy_A, &memcpy_B, 1));
  HIP_CHECK(hipGraphAddDependencies(graph1, &memcpy_B, &memcpy_C, 1));
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph1, nullptr, nullptr, 0));

  HIP_CHECK(hipGraphCreate(&graph2, 0));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_A, graph2, nullptr, 0, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_B, graph2, nullptr, 0, B_d, B_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_C, graph2, nullptr, 0, C_d, C_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddDependencies(graph2, &memcpy_A, &memcpy_C, 1));
  HIP_CHECK(hipGraphAddDependencies(graph2, &memcpy_B, &memcpy_C, 1));
  ret = hipGraphExecUpdate(graphExec, graph2, &hErrorNode_out, &updateResult_out);

  REQUIRE(hipErrorGraphExecUpdateFailure == ret);
  REQUIRE(hipGraphExecUpdateErrorTopologyChanged == updateResult_out);
  REQUIRE(NULL != hErrorNode_out);

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph1));
  HIP_CHECK(hipGraphDestroy(graph2));
}

/**
 * Test Description
 * ------------------------
 *  - Negative Scenario -
   8) Create a graph1 with memcpy_A, memcpy_B and dependency memcpy_A->memcpy_B
      and hipGraphInstantiate to create graphExec from it
      Create a graph2 with memcpy_A, memsetNode and dependency memcpy_A->memsetNode
      and Update graphExec with graph2 and verify
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphExecUpdate.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */

TEST_CASE("Unit_hipGraphExecUpdate_Negative_NodeType_Changed") {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  int *A_d, *B_d, *C_d, *A_h, *B_h, *C_h;
  hipGraphNode_t memcpy_A, memcpy_B, memsetNode;
  hipError_t ret;
  hipGraph_t graph1, graph2;
  hipGraphExec_t graphExec;
  hipGraphNode_t hErrorNode_out;
  hipGraphExecUpdateResult updateResult_out;
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);

  HIP_CHECK(hipGraphCreate(&graph1, 0));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_A, graph1, nullptr, 0, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_B, graph1, nullptr, 0, B_d, B_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddDependencies(graph1, &memcpy_A, &memcpy_B, 1));
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph1, nullptr, nullptr, 0));

  HIP_CHECK(hipGraphCreate(&graph2, 0));
  hipMemsetParams memsetParams{};
  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = reinterpret_cast<void*>(C_d);
  memsetParams.value = 3;
  memsetParams.pitch = 0;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = Nbytes;
  memsetParams.height = 1;
  HIP_CHECK(hipGraphAddMemsetNode(&memsetNode, graph2, nullptr, 0, &memsetParams));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_A, graph2, nullptr, 0, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddDependencies(graph2, &memcpy_A, &memsetNode, 1));
  ret = hipGraphExecUpdate(graphExec, graph2, &hErrorNode_out, &updateResult_out);
  REQUIRE(hipErrorGraphExecUpdateFailure == ret);
  REQUIRE(hipGraphExecUpdateErrorNodeTypeChanged == updateResult_out);
  REQUIRE(memsetNode == hErrorNode_out);

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph1));
  HIP_CHECK(hipGraphDestroy(graph2));
}

/**
 * Test Description
 * ------------------------
 *  - Negative Scenario -
   9) Multidevice case - set device 0 and
      Create a graph1 with ketnelNode as vector_ADD
      and hipGraphInstantiate to create graphExec from it
      set device 1 and Create a graph2 with ketnelNode as vector_SUB
      and Update graphExec with graph2 and verify.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphExecUpdate.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */

TEST_CASE("Unit_hipGraphExecUpdate_Negative_MultiDevice_Context_Changed") {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  size_t NElem{N};
  hipGraphNode_t memcpy_A, memcpy_B, memcpy_C;
  hipGraphNode_t kernel_vecADD, kernel_vecSUB;
  hipError_t ret;
  hipGraph_t graph1, graph2;
  hipGraphExec_t graphExec;
  hipGraphNode_t hErrorNode_out;
  hipGraphExecUpdateResult updateResult_out;

  int numDevices{}, peerAccess{};
  HIP_CHECK(hipGetDeviceCount(&numDevices));
  if (numDevices > 1) {
    HIP_CHECK(hipDeviceCanAccessPeer(&peerAccess, 1, 0));
  }
  if (!peerAccess) {
    WARN("Skipping test as peer device access is not found!");
    return;
  }
  HIP_CHECK(hipSetDevice(0));
  int *A_d, *B_d, *C_d, *A_h, *B_h, *C_h;
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipGraphCreate(&graph1, 0));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_A, graph1, nullptr, 0, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_B, graph1, nullptr, 0, B_d, B_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_C, graph1, nullptr, 0, C_h, C_d, Nbytes,
                                    hipMemcpyDeviceToHost));
  hipKernelNodeParams kernelNodeParams{};
  void* kernelArgs[] = {&A_d, &B_d, &C_d, reinterpret_cast<void*>(&NElem)};
  kernelNodeParams.func = reinterpret_cast<void*>(HipTest::vectorADD<int>);
  kernelNodeParams.gridDim = dim3(blocks);
  kernelNodeParams.blockDim = dim3(threadsPerBlock);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
  kernelNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kernel_vecADD, graph1, nullptr, 0, &kernelNodeParams));
  HIP_CHECK(hipGraphAddDependencies(graph1, &memcpy_A, &kernel_vecADD, 1));
  HIP_CHECK(hipGraphAddDependencies(graph1, &memcpy_B, &kernel_vecADD, 1));
  HIP_CHECK(hipGraphAddDependencies(graph1, &kernel_vecADD, &memcpy_C, 1));
  // Instantiate and launch the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph1, nullptr, nullptr, 0));

  HIP_CHECK(hipSetDevice(1));
  HIP_CHECK(hipGraphCreate(&graph2, 0));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_A, graph2, nullptr, 0, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_B, graph2, nullptr, 0, B_d, B_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_C, graph2, nullptr, 0, C_h, C_d, Nbytes,
                                    hipMemcpyDeviceToHost));
  memset(&kernelNodeParams, 0x00, sizeof(hipKernelNodeParams));
  void* kernelArgs1[] = {&A_d, &B_d, &C_d, reinterpret_cast<void*>(&NElem)};
  kernelNodeParams.func = reinterpret_cast<void*>(HipTest::vectorSUB<int>);
  kernelNodeParams.gridDim = dim3(blocks);
  kernelNodeParams.blockDim = dim3(threadsPerBlock);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs1);
  kernelNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kernel_vecSUB, graph2, nullptr, 0, &kernelNodeParams));
  HIP_CHECK(hipGraphAddDependencies(graph2, &memcpy_A, &kernel_vecSUB, 1));
  HIP_CHECK(hipGraphAddDependencies(graph2, &memcpy_B, &kernel_vecSUB, 1));
  HIP_CHECK(hipGraphAddDependencies(graph2, &kernel_vecSUB, &memcpy_C, 1));
  ret = hipGraphExecUpdate(graphExec, graph2, &hErrorNode_out, &updateResult_out);

  REQUIRE(hipErrorGraphExecUpdateFailure == ret);
  REQUIRE(hipGraphExecUpdateErrorUnsupportedFunctionChange == updateResult_out);
  REQUIRE(nullptr != hErrorNode_out);

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph1));
  HIP_CHECK(hipGraphDestroy(graph2));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 *  - Functional Scenario -
   1) Create a graph1 with ketnelNode as vector_ADD
      and hipGraphInstantiate to create graphExec from it
      Create a graph2 with ketnelNode as vector_SUB
      and Update graphExec with graph2 and verify update should work as expected.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphExecUpdate.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */

TEST_CASE("Unit_hipGraphExecUpdate_Functional_KernelFunction_Changed") {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  size_t NElem{N};
  hipGraphNode_t memcpy_A, memcpy_B, memcpy_C;
  hipGraphNode_t kernel_vecADD, kernel_vecSUB;
  hipError_t ret;
  hipGraph_t graph1, graph2;
  hipGraphExec_t graphExec;
  hipGraphNode_t hErrorNode_out;
  hipGraphExecUpdateResult updateResult_out;
  int *A_d, *B_d, *C_d, *A_h, *B_h, *C_h;
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipGraphCreate(&graph1, 0));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_A, graph1, nullptr, 0, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_B, graph1, nullptr, 0, B_d, B_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_C, graph1, nullptr, 0, C_h, C_d, Nbytes,
                                    hipMemcpyDeviceToHost));
  hipKernelNodeParams kernelNodeParams{};
  void* kernelArgs[] = {&A_d, &B_d, &C_d, reinterpret_cast<void*>(&NElem)};
  kernelNodeParams.func = reinterpret_cast<void*>(HipTest::vectorADD<int>);
  kernelNodeParams.gridDim = dim3(blocks);
  kernelNodeParams.blockDim = dim3(threadsPerBlock);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
  kernelNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kernel_vecADD, graph1, nullptr, 0, &kernelNodeParams));
  HIP_CHECK(hipGraphAddDependencies(graph1, &memcpy_A, &kernel_vecADD, 1));
  HIP_CHECK(hipGraphAddDependencies(graph1, &memcpy_B, &kernel_vecADD, 1));
  HIP_CHECK(hipGraphAddDependencies(graph1, &kernel_vecADD, &memcpy_C, 1));
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph1, nullptr, nullptr, 0));

  HIP_CHECK(hipGraphCreate(&graph2, 0));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_A, graph2, nullptr, 0, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_B, graph2, nullptr, 0, B_d, B_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_C, graph2, nullptr, 0, C_h, C_d, Nbytes,
                                    hipMemcpyDeviceToHost));
  memset(&kernelNodeParams, 0x00, sizeof(hipKernelNodeParams));
  void* kernelArgs1[] = {&A_d, &B_d, &C_d, reinterpret_cast<void*>(&NElem)};
  kernelNodeParams.func = reinterpret_cast<void*>(HipTest::vectorSUB<int>);
  kernelNodeParams.gridDim = dim3(blocks);
  kernelNodeParams.blockDim = dim3(threadsPerBlock);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs1);
  kernelNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kernel_vecSUB, graph2, nullptr, 0, &kernelNodeParams));
  HIP_CHECK(hipGraphAddDependencies(graph2, &memcpy_A, &kernel_vecSUB, 1));
  HIP_CHECK(hipGraphAddDependencies(graph2, &memcpy_B, &kernel_vecSUB, 1));
  HIP_CHECK(hipGraphAddDependencies(graph2, &kernel_vecSUB, &memcpy_C, 1));
  ret = hipGraphExecUpdate(graphExec, graph2, &hErrorNode_out, &updateResult_out);
  REQUIRE(hipSuccess == ret);
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // Verify graph execution result
  HipTest::checkVectorSUB(A_h, B_h, C_h, N);
  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph1));
  HIP_CHECK(hipGraphDestroy(graph2));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * End doxygen group GraphTest.
 * @}
 */
