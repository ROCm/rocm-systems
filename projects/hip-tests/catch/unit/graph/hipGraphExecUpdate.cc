/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
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
#include <chrono>
#include <thread>

#include <resource_guards.hh>
#include <utils.hh>

__global__ void GraphExecUpdateStore(int* output, int value) { output[0] = value; }

__global__ void GraphExecUpdateDelay(unsigned long long cycles) {
  const unsigned long long start = clock64();
  while (clock64() - start < cycles) {
  }
}

__global__ void GraphExecUpdateDelayStore(int* output, int value, unsigned long long cycles) {
  const unsigned long long start = clock64();
  while (clock64() - start < cycles) {
  }
  output[0] = value;
}

__global__ void GraphExecUpdateCopy(int* output, const int* input) { output[0] = input[0]; }

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
HIP_TEST_CASE(Unit_hipGraphExecUpdate_Negative_Basic) {
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

HIP_TEST_CASE(Unit_hipGraphExecUpdate_Negative_TypeChange) {
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
  HipTest::freeArrays<int>(A_d, nullptr, nullptr, A_h, nullptr, nullptr, false);
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

HIP_TEST_CASE(Unit_hipGraphExecUpdate_Negative_CountDiffer) {
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

HIP_TEST_CASE(Unit_hipGraphExecUpdate_Functional) {
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

HIP_TEST_CASE(Unit_hipGraphExecUpdate_Negative_Functional_ParametersChanged) {
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

HIP_TEST_CASE(Unit_hipGraphExecUpdate_Negative_Functional_CountDiffer_1) {
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

HIP_TEST_CASE(Unit_hipGraphExecUpdate_Negative_Functional_CountDiffer_2) {
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

HIP_TEST_CASE(Unit_hipGraphExecUpdate_Negative_Dependent_NodesDiffer) {
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

HIP_TEST_CASE(Unit_hipGraphExecUpdate_Negative_NodeType_Changed) {
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

HIP_TEST_CASE(Unit_hipGraphExecUpdate_Negative_MultiDevice_Context_Changed) {
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
    HIP_SKIP_TEST(HipTest::SkipReason::kPeerAccessUnavailable);
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

HIP_TEST_CASE(Unit_hipGraphExecUpdate_Functional_KernelFunction_Changed) {
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
 * Test Description
 * ------------------------
 *  - Verify updates while an earlier launch is in flight and updates using extra kernel params.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphExecUpdate.cc
 */
HIP_TEST_CASE(Unit_hipGraphExecUpdate_PacketRefresh) {
  SECTION("update while launch is in flight") {
    LinearAllocGuard<int> output_a(LinearAllocs::hipMalloc, sizeof(int));
    LinearAllocGuard<int> output_b(LinearAllocs::hipMalloc, sizeof(int));
    int zero = 0;
    HIP_CHECK(hipMemcpy(output_a.ptr(), &zero, sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(output_b.ptr(), &zero, sizeof(int), hipMemcpyHostToDevice));

    hipStream_t stream = nullptr;
    hipGraph_t graph = nullptr;
    hipGraphExec_t exec = nullptr;
    hipGraphNode_t delay_node = nullptr;
    hipGraphNode_t write_node = nullptr;
    HIP_CHECK(hipStreamCreate(&stream));
    HIP_CHECK(hipGraphCreate(&graph, 0));

    unsigned long long cycles = 50000000;
    void* delay_args[] = {&cycles};
    hipKernelNodeParams delay_params{};
    delay_params.func = reinterpret_cast<void*>(GraphExecUpdateDelay);
    delay_params.gridDim = dim3(1);
    delay_params.blockDim = dim3(1);
    delay_params.kernelParams = delay_args;
    HIP_CHECK(hipGraphAddKernelNode(&delay_node, graph, nullptr, 0, &delay_params));

    int* output = output_a.ptr();
    int value = 11;
    void* write_args[] = {&output, &value};
    hipKernelNodeParams write_params{};
    write_params.func = reinterpret_cast<void*>(GraphExecUpdateStore);
    write_params.gridDim = dim3(1);
    write_params.blockDim = dim3(1);
    write_params.kernelParams = write_args;
    HIP_CHECK(hipGraphAddKernelNode(&write_node, graph, &delay_node, 1, &write_params));
    HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(exec, stream));

    output = output_b.ptr();
    value = 22;
    HIP_CHECK(hipGraphKernelNodeSetParams(write_node, &write_params));
    hipGraphNode_t error_node = nullptr;
    hipGraphExecUpdateResult result = hipGraphExecUpdateError;
    HIP_CHECK(hipGraphExecUpdate(exec, graph, &error_node, &result));
    REQUIRE(result == hipGraphExecUpdateSuccess);
    REQUIRE(hipStreamQuery(stream) == hipErrorNotReady);
    HIP_CHECK(hipGraphLaunch(exec, stream));
    HIP_CHECK(hipStreamSynchronize(stream));

    int actual_a = 0;
    int actual_b = 0;
    HIP_CHECK(hipMemcpy(&actual_a, output_a.ptr(), sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&actual_b, output_b.ptr(), sizeof(int), hipMemcpyDeviceToHost));
    REQUIRE(actual_a == 11);
    REQUIRE(actual_b == 22);
    HIP_CHECK(hipGraphExecDestroy(exec));
    HIP_CHECK(hipGraphDestroy(graph));
    HIP_CHECK(hipStreamDestroy(stream));
  }

  SECTION("extra kernel params") {
    struct KernelArgs {
      int* output;
      int value;
    } args{};

    LinearAllocGuard<int> output(LinearAllocs::hipMalloc, sizeof(int));
    args.output = output.ptr();
    args.value = 31;
    size_t args_size = sizeof(args);
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE,
                      &args_size, HIP_LAUNCH_PARAM_END};
    hipKernelNodeParams params{};
    params.func = reinterpret_cast<void*>(GraphExecUpdateStore);
    params.gridDim = dim3(1);
    params.blockDim = dim3(1);
    params.extra = config;

    hipGraph_t graph = nullptr;
    hipGraphExec_t exec = nullptr;
    hipGraphNode_t node = nullptr;
    HIP_CHECK(hipGraphCreate(&graph, 0));
    HIP_CHECK(hipGraphAddKernelNode(&node, graph, nullptr, 0, &params));
    HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    args.value = 47;
    HIP_CHECK(hipGraphKernelNodeSetParams(node, &params));
    hipGraphNode_t error_node = nullptr;
    hipGraphExecUpdateResult result = hipGraphExecUpdateError;
    HIP_CHECK(hipGraphExecUpdate(exec, graph, &error_node, &result));
    REQUIRE(result == hipGraphExecUpdateSuccess);
    HIP_CHECK(hipGraphLaunch(exec, hipStreamPerThread));
    HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));

    int actual = 0;
    HIP_CHECK(hipMemcpy(&actual, output.ptr(), sizeof(int), hipMemcpyDeviceToHost));
    REQUIRE(actual == args.value);
    HIP_CHECK(hipGraphExecDestroy(exec));
    HIP_CHECK(hipGraphDestroy(graph));
  }
}

/**
 * Test Description
 * ------------------------
 *  - Verify that changing a kernel node between cooperative and normal launch is rejected.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphExecUpdate.cc
 */
HIP_TEST_CASE(Unit_hipGraphExecUpdate_CooperativeTransition) {
  if (!DeviceAttributesSupport(0, hipDeviceAttributeCooperativeLaunch)) {
    HIP_SKIP_TEST(HipTest::SkipReason::kCooperativeLaunchUnsupported);
  }
  LinearAllocGuard<int> output(LinearAllocs::hipMalloc, sizeof(int));
  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));

  auto capture_graph = [&](bool cooperative) {
    hipGraph_t graph = nullptr;
    int* output_ptr = output.ptr();
    int value = cooperative ? 2 : 1;
    void* args[] = {&output_ptr, &value};
    HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
    if (cooperative) {
      HIP_CHECK(hipLaunchCooperativeKernel(reinterpret_cast<const void*>(GraphExecUpdateStore),
                                           dim3(1), dim3(1), args, 0, stream));
    } else {
      HIP_CHECK(hipLaunchKernel(reinterpret_cast<const void*>(GraphExecUpdateStore), dim3(1),
                                dim3(1), args, 0, stream));
    }
    HIP_CHECK(hipStreamEndCapture(stream, &graph));
    return graph;
  };

  hipGraph_t normal = capture_graph(false);
  hipGraph_t cooperative = capture_graph(true);
  auto expect_rejected = [](hipGraph_t initial, hipGraph_t replacement) {
    hipGraphExec_t exec = nullptr;
    HIP_CHECK(hipGraphInstantiate(&exec, initial, nullptr, nullptr, 0));
    hipGraphNode_t error_node = nullptr;
    hipGraphExecUpdateResult result = hipGraphExecUpdateSuccess;
    HIP_CHECK_ERROR(hipGraphExecUpdate(exec, replacement, &error_node, &result),
                    hipErrorGraphExecUpdateFailure);
    REQUIRE(result == hipGraphExecUpdateErrorParametersChanged);
    HIP_CHECK(hipGraphExecDestroy(exec));
  };
  expect_rejected(normal, cooperative);
  expect_rejected(cooperative, normal);

  HIP_CHECK(hipGraphDestroy(cooperative));
  HIP_CHECK(hipGraphDestroy(normal));
  HIP_CHECK(hipStreamDestroy(stream));
}

__global__ void GraphExecUpdateSumTwo(int* output, const int* a, const int* b, int addend) {
  output[0] = a[0] + b[0] + addend;
}

/**
 * Test Description
 * ------------------------
 *  - Verify that an isomorphic graph whose dependencies were added in a different order
 *    updates successfully and applies the new parameters (dependencies compare as sets).
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphExecUpdate.cc
 */
HIP_TEST_CASE(Unit_hipGraphExecUpdate_ReorderedDependencies) {
  LinearAllocGuard<int> output(LinearAllocs::hipMalloc, 3 * sizeof(int));
  int* out = output.ptr();
  HIP_CHECK(hipMemset(out, 0, 3 * sizeof(int)));

  // A writes out[0], B writes out[1], C = out[0] + out[1] + addend and depends on both.
  auto build = [&](bool a_first, int addend) {
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipGraphCreate(&graph, 0));
    hipGraphNode_t a = nullptr, b = nullptr, c = nullptr;
    int* out0 = out;
    int* out1 = out + 1;
    int* out2 = out + 2;
    int va = 3, vb = 4;
    void* args_a[] = {&out0, &va};
    void* args_b[] = {&out1, &vb};
    void* args_c[] = {&out2, &out0, &out1, &addend};
    hipKernelNodeParams pa{}, pb{}, pc{};
    pa.func = reinterpret_cast<void*>(GraphExecUpdateStore);
    pa.gridDim = dim3(1); pa.blockDim = dim3(1); pa.kernelParams = args_a;
    pb = pa; pb.kernelParams = args_b;
    pc.func = reinterpret_cast<void*>(GraphExecUpdateSumTwo);
    pc.gridDim = dim3(1); pc.blockDim = dim3(1); pc.kernelParams = args_c;
    HIP_CHECK(hipGraphAddKernelNode(&a, graph, nullptr, 0, &pa));
    HIP_CHECK(hipGraphAddKernelNode(&b, graph, nullptr, 0, &pb));
    HIP_CHECK(hipGraphAddKernelNode(&c, graph, nullptr, 0, &pc));
    if (a_first) {
      HIP_CHECK(hipGraphAddDependencies(graph, &a, &c, 1));
      HIP_CHECK(hipGraphAddDependencies(graph, &b, &c, 1));
    } else {
      HIP_CHECK(hipGraphAddDependencies(graph, &b, &c, 1));
      HIP_CHECK(hipGraphAddDependencies(graph, &a, &c, 1));
    }
    return graph;
  };

  hipGraph_t graph1 = build(true, 10);
  hipGraph_t graph2 = build(false, 20);
  hipGraphExec_t exec = nullptr;
  HIP_CHECK(hipGraphInstantiate(&exec, graph1, nullptr, nullptr, 0));

  hipGraphNode_t error_node = nullptr;
  hipGraphExecUpdateResult result = hipGraphExecUpdateError;
  HIP_CHECK(hipGraphExecUpdate(exec, graph2, &error_node, &result));
  REQUIRE(result == hipGraphExecUpdateSuccess);
  HIP_CHECK(hipGraphLaunch(exec, hipStreamPerThread));
  HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));

  int actual[3] = {0, 0, 0};
  HIP_CHECK(hipMemcpy(actual, out, 3 * sizeof(int), hipMemcpyDeviceToHost));
  REQUIRE(actual[0] == 3);
  REQUIRE(actual[1] == 4);
  REQUIRE(actual[2] == 3 + 4 + 20);

  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(graph2));
  HIP_CHECK(hipGraphDestroy(graph1));
}

/**
 * Test Description
 * ------------------------
 *  - Verify that a graph with the same node and per-node dependency counts but a genuinely
 *    different edge set is rejected as a topology change (a dependency-count-only check would
 *    accept it). graph1 fans out A->{C,D}; graph2 wires A->C, B->D: same counts, no pairing
 *    of the identical topological orders preserves the edges.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphExecUpdate.cc
 */
HIP_TEST_CASE(Unit_hipGraphExecUpdate_RewiredDependencies_Negative) {
  LinearAllocGuard<int> output(LinearAllocs::hipMalloc, 4 * sizeof(int));
  int* out = output.ptr();

  // Four store nodes; graph1 wires A->C, A->D, graph2 wires A->C, B->D.
  auto build = [&](bool crossed) {
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipGraphCreate(&graph, 0));
    hipGraphNode_t nodes[4];
    int* ptrs[4] = {out, out + 1, out + 2, out + 3};
    int values[4] = {1, 2, 3, 4};
    for (int i = 0; i < 4; ++i) {
      void* args[] = {&ptrs[i], &values[i]};
      hipKernelNodeParams p{};
      p.func = reinterpret_cast<void*>(GraphExecUpdateStore);
      p.gridDim = dim3(1); p.blockDim = dim3(1); p.kernelParams = args;
      HIP_CHECK(hipGraphAddKernelNode(&nodes[i], graph, nullptr, 0, &p));
    }
    hipGraphNode_t& a = nodes[0]; hipGraphNode_t& b = nodes[1];
    hipGraphNode_t& c = nodes[2]; hipGraphNode_t& d = nodes[3];
    HIP_CHECK(hipGraphAddDependencies(graph, &a, &c, 1));
    if (!crossed) {
      HIP_CHECK(hipGraphAddDependencies(graph, &a, &d, 1));
    } else {
      HIP_CHECK(hipGraphAddDependencies(graph, &b, &d, 1));
    }
    return graph;
  };

  hipGraph_t graph1 = build(false);
  hipGraph_t graph2 = build(true);
  hipGraphExec_t exec = nullptr;
  HIP_CHECK(hipGraphInstantiate(&exec, graph1, nullptr, nullptr, 0));

  hipGraphNode_t error_node = nullptr;
  hipGraphExecUpdateResult result = hipGraphExecUpdateSuccess;
  HIP_CHECK_ERROR(hipGraphExecUpdate(exec, graph2, &error_node, &result),
                  hipErrorGraphExecUpdateFailure);
  REQUIRE(result == hipGraphExecUpdateErrorTopologyChanged);
  hipGraphExec_t exec2 = nullptr;
  HIP_CHECK(hipGraphInstantiate(&exec2, graph2, nullptr, nullptr, 0));
  result = hipGraphExecUpdateSuccess;
  HIP_CHECK_ERROR(hipGraphExecUpdate(exec2, graph1, &error_node, &result),
                  hipErrorGraphExecUpdateFailure);
  REQUIRE(result == hipGraphExecUpdateErrorTopologyChanged);
  HIP_CHECK(hipGraphExecDestroy(exec2));

  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(graph2));
  HIP_CHECK(hipGraphDestroy(graph1));
}

/**
 * Test Description
 * ------------------------
 *  - Verify that hipGraphExecUpdate propagates a kernel parameter change made inside a child
 *    graph (through hipGraphChildGraphNodeGetGraph) into the executable graph's packets.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphExecUpdate.cc
 */
HIP_TEST_CASE(Unit_hipGraphExecUpdate_ChildGraphKernelParams) {
  LinearAllocGuard<int> output(LinearAllocs::hipMalloc, 2 * sizeof(int));
  int* out = output.ptr();
  HIP_CHECK(hipMemset(out, 0, 2 * sizeof(int)));

  hipGraph_t parent = nullptr, child = nullptr;
  HIP_CHECK(hipGraphCreate(&parent, 0));
  HIP_CHECK(hipGraphCreate(&child, 0));

  int* out0 = out;
  int* out1 = out + 1;
  int parent_value = 1;
  int child_value = 10;
  void* parent_args[] = {&out0, &parent_value};
  void* child_args[] = {&out1, &child_value};
  hipKernelNodeParams pp{}, cp{};
  pp.func = reinterpret_cast<void*>(GraphExecUpdateStore);
  pp.gridDim = dim3(1); pp.blockDim = dim3(1); pp.kernelParams = parent_args;
  cp = pp; cp.kernelParams = child_args;

  hipGraphNode_t child_kernel = nullptr, parent_kernel = nullptr, child_node = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&child_kernel, child, nullptr, 0, &cp));
  HIP_CHECK(hipGraphAddKernelNode(&parent_kernel, parent, nullptr, 0, &pp));
  HIP_CHECK(hipGraphAddChildGraphNode(&child_node, parent, &parent_kernel, 1, child));

  hipGraphExec_t exec = nullptr;
  HIP_CHECK(hipGraphInstantiate(&exec, parent, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(exec, hipStreamPerThread));
  HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));
  int actual[2] = {0, 0};
  HIP_CHECK(hipMemcpy(actual, out, 2 * sizeof(int), hipMemcpyDeviceToHost));
  REQUIRE(actual[0] == 1);
  REQUIRE(actual[1] == 10);

  hipGraph_t embedded = nullptr;
  HIP_CHECK(hipGraphChildGraphNodeGetGraph(child_node, &embedded));
  size_t num_nodes = 0;
  HIP_CHECK(hipGraphGetNodes(embedded, nullptr, &num_nodes));
  REQUIRE(num_nodes == 1);
  hipGraphNode_t embedded_kernel = nullptr;
  HIP_CHECK(hipGraphGetNodes(embedded, &embedded_kernel, &num_nodes));
  child_value = 20;
  HIP_CHECK(hipGraphKernelNodeSetParams(embedded_kernel, &cp));

  hipGraphNode_t error_node = nullptr;
  hipGraphExecUpdateResult result = hipGraphExecUpdateError;
  HIP_CHECK(hipGraphExecUpdate(exec, parent, &error_node, &result));
  REQUIRE(result == hipGraphExecUpdateSuccess);
  HIP_CHECK(hipGraphLaunch(exec, hipStreamPerThread));
  HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));
  HIP_CHECK(hipMemcpy(actual, out, 2 * sizeof(int), hipMemcpyDeviceToHost));
  REQUIRE(actual[0] == 1);
  REQUIRE(actual[1] == 20);

  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(child));
  HIP_CHECK(hipGraphDestroy(parent));
}

/**
 * Test Description
 * ------------------------
 *  - Verify repeated updates from an unchanged graph are no-ops that keep the exec correct, and
 *    that a change followed by an identical update (fast path) still launches the new params.
 *    Uses kernel, memcpy and memset nodes so every node type's identity check is exercised.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphExecUpdate.cc
 */
HIP_TEST_CASE(Unit_hipGraphExecUpdate_IdenticalParamsIdempotent) {
  constexpr size_t kCount = 64;
  LinearAllocGuard<int> src(LinearAllocs::hipMalloc, kCount * sizeof(int));
  LinearAllocGuard<int> copy(LinearAllocs::hipMalloc, kCount * sizeof(int));
  LinearAllocGuard<int> set(LinearAllocs::hipMalloc, kCount * sizeof(int));
  LinearAllocGuard<int> output(LinearAllocs::hipMalloc, sizeof(int));
  std::vector<int> host(kCount);
  for (size_t i = 0; i < kCount; ++i) host[i] = static_cast<int>(i) + 100;
  HIP_CHECK(hipMemcpy(src.ptr(), host.data(), kCount * sizeof(int), hipMemcpyHostToDevice));

  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipGraphNode_t memcpy_node = nullptr, memset_node = nullptr, kernel_node = nullptr;
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_node, graph, nullptr, 0, copy.ptr(), src.ptr(),
                                    kCount * sizeof(int), hipMemcpyDeviceToDevice));
  hipMemsetParams memset_params{};
  memset_params.dst = set.ptr();
  memset_params.elementSize = sizeof(int);
  memset_params.width = kCount;
  memset_params.height = 1;
  memset_params.value = 5;
  HIP_CHECK(hipGraphAddMemsetNode(&memset_node, graph, nullptr, 0, &memset_params));
  int* out = output.ptr();
  int value = 7;
  void* args[] = {&out, &value};
  hipKernelNodeParams kp{};
  kp.func = reinterpret_cast<void*>(GraphExecUpdateStore);
  kp.gridDim = dim3(1); kp.blockDim = dim3(1); kp.kernelParams = args;
  HIP_CHECK(hipGraphAddKernelNode(&kernel_node, graph, nullptr, 0, &kp));

  hipGraphExec_t exec = nullptr;
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  auto update = [&]() {
    hipGraphNode_t error_node = nullptr;
    hipGraphExecUpdateResult result = hipGraphExecUpdateError;
    HIP_CHECK(hipGraphExecUpdate(exec, graph, &error_node, &result));
    REQUIRE(result == hipGraphExecUpdateSuccess);
  };
  auto verify = [&](int expected_value) {
    HIP_CHECK(hipGraphLaunch(exec, hipStreamPerThread));
    HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));
    std::vector<int> copied(kCount), memset_out(kCount);
    int actual = 0;
    HIP_CHECK(hipMemcpy(copied.data(), copy.ptr(), kCount * sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(memset_out.data(), set.ptr(), kCount * sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&actual, out, sizeof(int), hipMemcpyDeviceToHost));
    REQUIRE(copied == host);
    REQUIRE(memset_out == std::vector<int>(kCount, 5));
    REQUIRE(actual == expected_value);
  };

  update();
  update();
  update();
  verify(7);

  value = 9;
  HIP_CHECK(hipGraphKernelNodeSetParams(kernel_node, &kp));
  update();
  update();
  verify(9);

  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *  - Verify an exec update that changes the parameters of a disabled node in a fork/join graph
 *    keeps the node disabled, applies the change once the node is re-enabled, and leaves the
 *    other nodes intact.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphExecUpdate.cc
 */
HIP_TEST_CASE(Unit_hipGraphExecUpdate_DisabledNodeInBatch) {
  LinearAllocGuard<int> output(LinearAllocs::hipMalloc, 4 * sizeof(int));
  int* out = output.ptr();
  HIP_CHECK(hipMemset(out, 0, 4 * sizeof(int)));

  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipGraphNode_t nodes[4];
  int* ptrs[4] = {out, out + 1, out + 2, out + 3};
  int values[4] = {1, 2, 3, 4};
  hipKernelNodeParams params[4];
  void* args[4][2];
  for (int i = 0; i < 4; ++i) {
    args[i][0] = &ptrs[i];
    args[i][1] = &values[i];
    params[i] = hipKernelNodeParams{};
    params[i].func = reinterpret_cast<void*>(GraphExecUpdateStore);
    params[i].gridDim = dim3(1);
    params[i].blockDim = dim3(1);
    params[i].kernelParams = args[i];
    HIP_CHECK(hipGraphAddKernelNode(&nodes[i], graph, nullptr, 0, &params[i]));
  }
  HIP_CHECK(hipGraphAddDependencies(graph, &nodes[0], &nodes[1], 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &nodes[0], &nodes[2], 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &nodes[1], &nodes[3], 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &nodes[2], &nodes[3], 1));

  hipGraphExec_t exec = nullptr;
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphNodeSetEnabled(exec, nodes[3], 0));

  values[3] = 40;
  values[2] = 30;
  HIP_CHECK(hipGraphKernelNodeSetParams(nodes[3], &params[3]));
  HIP_CHECK(hipGraphKernelNodeSetParams(nodes[2], &params[2]));
  hipGraphNode_t error_node = nullptr;
  hipGraphExecUpdateResult result = hipGraphExecUpdateError;
  HIP_CHECK(hipGraphExecUpdate(exec, graph, &error_node, &result));
  REQUIRE(result == hipGraphExecUpdateSuccess);

  int actual[4] = {0, 0, 0, 0};
  for (int launch = 0; launch < 3; ++launch) {
    HIP_CHECK(hipGraphLaunch(exec, hipStreamPerThread));
    HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));
    HIP_CHECK(hipMemcpy(actual, out, 4 * sizeof(int), hipMemcpyDeviceToHost));
    REQUIRE(actual[0] == 1);
    REQUIRE(actual[1] == 2);
    REQUIRE(actual[2] == 30);
    REQUIRE(actual[3] == 0);
  }

  HIP_CHECK(hipGraphNodeSetEnabled(exec, nodes[3], 1));
  HIP_CHECK(hipGraphLaunch(exec, hipStreamPerThread));
  HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));
  HIP_CHECK(hipMemcpy(actual, out, 4 * sizeof(int), hipMemcpyDeviceToHost));
  REQUIRE(actual[2] == 30);
  REQUIRE(actual[3] == 40);

  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *  - Verify that a stream-captured graph (memcpy + kernel) re-captured with a different copy
 *    source updates the executable graph and the new data reaches the kernel.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphExecUpdate.cc
 */
HIP_TEST_CASE(Unit_hipGraphExecUpdate_CapturedMemcpyAndKernel) {
  LinearAllocGuard<int> src1(LinearAllocs::hipMalloc, sizeof(int));
  LinearAllocGuard<int> src2(LinearAllocs::hipMalloc, sizeof(int));
  LinearAllocGuard<int> staging(LinearAllocs::hipMalloc, sizeof(int));
  LinearAllocGuard<int> zero(LinearAllocs::hipMalloc, sizeof(int));
  LinearAllocGuard<int> output(LinearAllocs::hipMalloc, sizeof(int));
  int first = 111, second = 222, zero_host = 0;
  HIP_CHECK(hipMemcpy(src1.ptr(), &first, sizeof(int), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(src2.ptr(), &second, sizeof(int), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(zero.ptr(), &zero_host, sizeof(int), hipMemcpyHostToDevice));

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  auto capture = [&](int* source, int addend) {
    hipGraph_t graph = nullptr;
    int* out = output.ptr();
    int* stage = staging.ptr();
    int* z = zero.ptr();
    void* args[] = {&out, &stage, &z, &addend};
    HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
    HIP_CHECK(hipMemcpyAsync(stage, source, sizeof(int), hipMemcpyDeviceToDevice, stream));
    HIP_CHECK(hipLaunchKernel(reinterpret_cast<const void*>(GraphExecUpdateSumTwo), dim3(1),
                              dim3(1), args, 0, stream));
    HIP_CHECK(hipStreamEndCapture(stream, &graph));
    return graph;
  };

  hipGraph_t graph1 = capture(src1.ptr(), 1);
  hipGraph_t graph2 = capture(src2.ptr(), 2);
  hipGraphExec_t exec = nullptr;
  HIP_CHECK(hipGraphInstantiate(&exec, graph1, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  int actual = 0;
  HIP_CHECK(hipMemcpy(&actual, output.ptr(), sizeof(int), hipMemcpyDeviceToHost));
  REQUIRE(actual == first + 1);

  hipGraphNode_t error_node = nullptr;
  hipGraphExecUpdateResult result = hipGraphExecUpdateError;
  HIP_CHECK(hipGraphExecUpdate(exec, graph2, &error_node, &result));
  REQUIRE(result == hipGraphExecUpdateSuccess);
  HIP_CHECK(hipGraphLaunch(exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipMemcpy(&actual, output.ptr(), sizeof(int), hipMemcpyDeviceToHost));
  REQUIRE(actual == second + 2);

  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(graph2));
  HIP_CHECK(hipGraphDestroy(graph1));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 *  - Regression test for the full packet rebuild an exec update performs when a captured node
 *    stops being capturable (here: a memcpy node retargeted from a device source to a pageable
 *    host source). Batches that contain a disabled node dispatch their filtered packet buffer,
 *    so the rebuild must leave the cross-stream dependency and completion-signal patches
 *    pointing into that buffer. If it does not, the launch patches the unfiltered buffer and
 *    the dispatched packets carry zeroed signal slots: a consumer stops waiting for its
 *    producer (section 1) or a producer never signals its consumer (section 2).
 *  - The graph is a chain of three fork/join diamonds A -> {B1,C1} -> D1 -> ... -> D3 -> F -> E.
 *    Its dependency depth keeps the runtime from collapsing it onto a single stream, so the
 *    last join always carries a cross-stream dependency. The spinning producer is generated on
 *    either branch of the last diamond so the check does not depend on which branch the
 *    scheduler places on the second stream.
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphExecUpdate.cc
 */
HIP_TEST_CASE(Unit_hipGraphExecUpdate_RebuildWithDisabledNodeKeepsSync) {
  constexpr int kDiamonds = 3;
  constexpr int kProducerValue = 42;
  constexpr unsigned long long kSpinCycles = 100000000ULL;
  const int spin_branch = GENERATE(0, 1);
  LinearAllocGuard<int> output(LinearAllocs::hipMalloc, 8 * sizeof(int));
  LinearAllocGuard<int> copy_src(LinearAllocs::hipMalloc, sizeof(int));
  LinearAllocGuard<int> copy_dst(LinearAllocs::hipMalloc, sizeof(int));
  int* out = output.ptr();
  int* x = out + 1;
  int* y = out + 2;
  int* f_out = out + 3;
  int* branch_out = out + 4;
  HIP_CHECK(hipMemset(out, 0, 8 * sizeof(int)));
  HIP_CHECK(hipMemset(copy_src.ptr(), 0, sizeof(int)));
  int host_src = 99;

  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  auto add_kernel = [&](hipGraphNode_t* node, hipGraphNode_t* deps, size_t num_deps, void* func,
                        void** args) {
    hipKernelNodeParams params{};
    params.func = func;
    params.gridDim = dim3(1);
    params.blockDim = dim3(1);
    params.kernelParams = args;
    HIP_CHECK(hipGraphAddKernelNode(node, graph, deps, num_deps, &params));
  };
  int one = 1, two = 2, five = 5, producer_value = kProducerValue;
  unsigned long long spin_cycles = kSpinCycles;
  void* a_args[] = {&out, &one};
  void* branch_args[] = {&branch_out, &two};
  void* producer_args[] = {&x, &producer_value, &spin_cycles};
  void* join_args[] = {&y, &x};
  void* f_args[] = {&f_out, &five};
  hipGraphNode_t prev = nullptr, producer = nullptr, f = nullptr, e = nullptr;
  add_kernel(&prev, nullptr, 0, reinterpret_cast<void*>(GraphExecUpdateStore), a_args);
  for (int i = 0; i < kDiamonds; ++i) {
    hipGraphNode_t branches[2];
    for (int b = 0; b < 2; ++b) {
      if (i + 1 == kDiamonds && b == spin_branch) {
        add_kernel(&branches[b], &prev, 1, reinterpret_cast<void*>(GraphExecUpdateDelayStore),
                   producer_args);
        producer = branches[b];
      } else {
        add_kernel(&branches[b], &prev, 1, reinterpret_cast<void*>(GraphExecUpdateStore),
                   branch_args);
      }
    }
    hipGraphNode_t join = nullptr;
    add_kernel(&join, branches, 2, reinterpret_cast<void*>(GraphExecUpdateCopy), join_args);
    prev = join;
  }
  add_kernel(&f, &prev, 1, reinterpret_cast<void*>(GraphExecUpdateStore), f_args);
  HIP_CHECK(hipGraphAddMemcpyNode1D(&e, graph, &f, 1, copy_dst.ptr(), copy_src.ptr(),
                                    sizeof(int), hipMemcpyDefault));

  hipGraphExec_t exec = nullptr;
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  StreamGuard stream_guard(Streams::created);
  const hipStream_t stream = stream_guard.stream();

  // Bounded wait: a lost completion signal must fail the test, not hang the suite.
  auto launch_and_wait = [&]() {
    HIP_CHECK(hipGraphLaunch(exec, stream));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
    hipError_t query = hipErrorNotReady;
    while ((query = hipStreamQuery(stream)) == hipErrorNotReady &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    REQUIRE(query == hipSuccess);
  };
  auto read_outputs = [&](int* values) {
    HIP_CHECK(hipMemcpy(values, out, 8 * sizeof(int), hipMemcpyDeviceToHost));
  };
  // Retargets E to a pageable host source, forcing a full packet rebuild while a node in the last join's batch is disabled.
  auto force_full_rebuild = [&]() {
    HIP_CHECK(hipGraphExecMemcpyNodeSetParams1D(exec, e, copy_dst.ptr(), &host_src, sizeof(int),
                                                hipMemcpyDefault));
  };

  int values[8] = {};
  launch_and_wait();
  read_outputs(values);
  REQUIRE(values[1] == kProducerValue);
  REQUIRE(values[2] == kProducerValue);
  REQUIRE(values[3] == 5);

  SECTION("disabled node in the consumer batch keeps the dependency wait") {
    HIP_CHECK(hipGraphNodeSetEnabled(exec, f, 0));
    force_full_rebuild();
    for (int launch = 0; launch < 3; ++launch) {
      HIP_CHECK(hipMemset(out, 0, 8 * sizeof(int)));
      launch_and_wait();
      read_outputs(values);
      REQUIRE(values[1] == kProducerValue);
      REQUIRE(values[2] == kProducerValue);
      REQUIRE(values[3] == 0);
    }
    int copied = 0;
    HIP_CHECK(hipMemcpy(&copied, copy_dst.ptr(), sizeof(int), hipMemcpyDeviceToHost));
    REQUIRE(copied == host_src);
    HIP_CHECK(hipGraphNodeSetEnabled(exec, f, 1));
    HIP_CHECK(hipMemset(out, 0, 8 * sizeof(int)));
    launch_and_wait();
    read_outputs(values);
    REQUIRE(values[2] == kProducerValue);
    REQUIRE(values[3] == 5);
  }

  SECTION("disabled producer keeps the completion signal") {
    HIP_CHECK(hipGraphNodeSetEnabled(exec, producer, 0));
    force_full_rebuild();
    for (int launch = 0; launch < 3; ++launch) {
      HIP_CHECK(hipMemset(out, 0, 8 * sizeof(int)));
      launch_and_wait();
      read_outputs(values);
      REQUIRE(values[1] == 0);
      REQUIRE(values[2] == 0);
      REQUIRE(values[3] == 5);
      REQUIRE(values[4] == 2);
    }
    HIP_CHECK(hipGraphNodeSetEnabled(exec, producer, 1));
    HIP_CHECK(hipMemset(out, 0, 8 * sizeof(int)));
    launch_and_wait();
    read_outputs(values);
    REQUIRE(values[1] == kProducerValue);
    REQUIRE(values[2] == kProducerValue);
  }

  HIP_CHECK(hipGraphExecDestroy(exec));
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * End doxygen group GraphTest.
 * @}
 */
