/*
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include <hip_test_checkers.hh>
#include <hip_test_common.hh>
#include <hip_test_kernels.hh>
#include "graph_node_common.hh"

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
TEST_CASE("Unit_hipGraphNodeSetParams_Negative_Parameters") {
  hipGraph_t graph;
  hipGraphExec_t graphExec;
  hipGraphNode_t node;
  hipGraphNodeParams node_params = {};
  char *A_d;
  char *A_h;
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
  A_h = reinterpret_cast<char *>(malloc(Nbytes));
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
    HIP_CHECK_ERROR(hipGraphNodeSetParams(nullptr, &node_params2),
                    hipErrorInvalidValue);
  }

  SECTION("nodeParams == nullptr") {
    HIP_CHECK_ERROR(hipGraphNodeSetParams(node, nullptr), hipErrorInvalidValue);
  }

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
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
TEST_CASE("Unit_hipGraphNodeSetParams_nodeTypeMemset") {
  hipGraph_t graph;
  hipGraphExec_t graphExec;
  hipGraphNode_t node;
  hipGraphNodeParams node_params = {};
  int *A_d, *A_h = nullptr;
  // size_t N = 10;
  // size_t Nbytes = N * sizeof(int);

  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipMalloc(&A_d, Nbytes));

  node_params = getNodeTypeMemset(A_d, 99);
  HIP_CHECK(hipGraphAddNode(&node, graph, nullptr, 0, &node_params));

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, 0));
  HIP_CHECK(hipStreamSynchronize(0));
  A_h = reinterpret_cast<int *>(malloc(Nbytes));
  HIP_CHECK(hipMemcpy(A_h, A_d, Nbytes, hipMemcpyDeviceToHost));
  for (int i = 0; i < N; i++) {
    REQUIRE(A_h[i] == 99);
  }

  // Create new node params with value 100
  hipGraphNodeParams node_params2 = {};
  node_params2 = getNodeTypeMemset(A_d, 100);

  HIP_CHECK(hipGraphNodeSetParams(node, &node_params2));

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, 0));
  HIP_CHECK(hipStreamSynchronize(0));
  HIP_CHECK(hipMemcpy(A_h, A_d, Nbytes, hipMemcpyDeviceToHost));
  for (int i = 0; i < N; i++) {
    REQUIRE(A_h[i] == 100);
  }

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(A_d));
  free(A_h);
}

/**
 * Test Description
 * ------------------------
 *    - Create nod params of type kernel and add to graph
 *    - Verify the kernel resutls
 *    - Create another node params of type kernel with different kernal function
 *    - Update the node of the graph with new node params using
 * hipGraphNodeSetParams Api
 *    - Verify the kernel results
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphNodeSetParams.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.4
 */
TEST_CASE("Unit_hipGraphNodeSetParams_nodeTypekernel") {
  hipGraph_t graph;
  hipGraphExec_t graphExec;
  hipGraphNode_t node;
  hipGraphNodeParams node_params = {};
  HIP_CHECK(hipGraphCreate(&graph, 0));
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  size_t NElem{N};

  int *A_d, *A_h;
  int *B_d, *B_h;
  int *C_d, *C_h;

  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
  HIP_CHECK(hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(B_d, B_h, Nbytes, hipMemcpyHostToDevice));

  void *kernel_args[] = {&A_d, &B_d, &C_d, reinterpret_cast<void *>(&NElem)};

  // Create node params with kernel vectorADD
  node_params = getNodeTypeKernel(
      kernel_args, reinterpret_cast<void *>(HipTest::vectorADD<int>));

  HIP_CHECK(hipGraphAddNode(&node, graph, nullptr, 0, &node_params));
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, 0));
  HIP_CHECK(hipStreamSynchronize(0));
  HIP_CHECK(hipMemcpy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost));

  // verify Result
  HipTest::checkVectorADD<int>(A_h, B_h, C_h, N);

  // Create new node params with kernel vectorsub
  hipGraphNodeParams node_params2 = {};
  node_params2 = getNodeTypeKernel(
      kernel_args, reinterpret_cast<void *>(HipTest::vectorSUB<int>));

  HIP_CHECK(hipGraphNodeSetParams(node, &node_params2));

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, 0));
  HIP_CHECK(hipStreamSynchronize(0));

  HIP_CHECK(hipMemcpy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost));

  // verify Result
  HipTest::checkVectorSUB<int>(A_h, B_h, C_h, N);

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
}

/**
 * Test Description
 * ------------------------
 *    - Create nod params of type memcpy and add to graph
 *    - Verify the resutls
 *    - Create another node params of type memcpy with different value
 *    - Update the node of the graph with new node params using
 * hipGraphNodeSetParams Api
 *    - Verify the results
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphNodeSetParams.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.4
 */
TEST_CASE("Unit_hipGraphNodeSetParams_nodeTypeMemcopy") {
  hipGraph_t graph;
  hipGraphExec_t graphExec;
  hipGraphNode_t node;
  hipGraphNodeParams node_params = {};
  hipGraphNodeParams node_params2 = {};
  HIP_CHECK(hipGraphCreate(&graph, 0));

  int *src = nullptr;
  int *dst = nullptr;
  SECTION("memcpy kind is hipMemcpyHostToDevice") {
    src = reinterpret_cast<int *>(malloc(Nbytes));
    HIP_CHECK(hipMalloc(&dst, Nbytes));
    for (int i = 0; i < N; i++)
      src[i] = 10;
    node_params = getNodeTypememcpy(&src, &dst, hipMemcpyHostToDevice);
    HIP_CHECK(hipGraphAddNode(&node, graph, nullptr, 0, &node_params));
    HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(graphExec, 0));
    HIP_CHECK(hipStreamSynchronize(0));

    // verifyResult;
    verifyMemcpyResult(src, dst);
    // Create new Node params with new device memory and update graph
    int *devPtr;
    HIP_CHECK(hipMalloc(&devPtr, Nbytes));
    node_params2 = getNodeTypememcpy(&src, &devPtr, hipMemcpyHostToDevice);
    HIP_CHECK(hipGraphNodeSetParams(node, &node_params2));
    HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(graphExec, 0));
    HIP_CHECK(hipStreamSynchronize(0));

    // verifyResult;
    verifyMemcpyResult(src, devPtr);

    HIP_CHECK(hipFree(dst));
    HIP_CHECK(hipFree(devPtr));
    free(src);
  }
  SECTION("memcpy kind is hipMemcpyHostToHost") {
    src = reinterpret_cast<int *>(malloc(Nbytes));
    dst = reinterpret_cast<int *>(malloc(Nbytes));
    for (int i = 0; i < N; i++)
      src[i] = 10;
    node_params = getNodeTypememcpy(&src, &dst, hipMemcpyHostToHost);
    HIP_CHECK(hipGraphAddNode(&node, graph, nullptr, 0, &node_params));
    HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(graphExec, 0));
    HIP_CHECK(hipStreamSynchronize(0));
    // verifyResult;
    verifyMemcpyResult(src, dst);

    int *hostPtr = reinterpret_cast<int *>(malloc(Nbytes));
    node_params2 = getNodeTypememcpy(&src, &hostPtr, hipMemcpyHostToHost);
    HIP_CHECK(hipGraphNodeSetParams(node, &node_params2));
    HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(graphExec, 0));
    HIP_CHECK(hipStreamSynchronize(0));
    // verifyResult;
    verifyMemcpyResult(src, hostPtr);
    free(src);
    free(dst);
    free(hostPtr);
  }
  SECTION("memcpy kind is hipMemcpyDeviceToDevice") {
    int *A_h = reinterpret_cast<int *>(malloc(Nbytes));
    HIP_CHECK(hipMalloc(&src, Nbytes));
    HIP_CHECK(hipMalloc(&dst, Nbytes));
    for (int i = 0; i < N; i++)
      A_h[i] = 10;
    HIP_CHECK(hipMemcpy(src, A_h, Nbytes, hipMemcpyHostToDevice));
    node_params = getNodeTypememcpy(&src, &dst, hipMemcpyDeviceToDevice);
    HIP_CHECK(hipGraphAddNode(&node, graph, nullptr, 0, &node_params));
    HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(graphExec, 0));
    HIP_CHECK(hipStreamSynchronize(0));
    // verifyResult;
    verifyMemcpyResult(src, dst);

    int *devPtr;
    HIP_CHECK(hipMalloc(&devPtr, Nbytes));
    node_params2 = getNodeTypememcpy(&src, &devPtr, hipMemcpyDeviceToDevice);
    HIP_CHECK(hipGraphNodeSetParams(node, &node_params2));
    HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(graphExec, 0));
    HIP_CHECK(hipStreamSynchronize(0));

    // verifyResult;
    verifyMemcpyResult(src, devPtr);

    HIP_CHECK(hipFree(src));
    HIP_CHECK(hipFree(dst));
    HIP_CHECK(hipFree(devPtr));
    free(A_h);
  }
  SECTION("memcpy kind is hipMemcpyDeviceToHost ") {
    int *A_h = reinterpret_cast<int *>(malloc(Nbytes));
    HIP_CHECK(hipMalloc(&src, Nbytes));
    dst = reinterpret_cast<int *>(malloc(Nbytes));
    HIP_CHECK(hipMalloc(&src, Nbytes));
    for (int i = 0; i < N; i++)
      A_h[i] = 10;
    HIP_CHECK(hipMemcpy(src, A_h, Nbytes, hipMemcpyHostToDevice));
    node_params = getNodeTypememcpy(&src, &dst, hipMemcpyDeviceToHost);
    HIP_CHECK(hipGraphAddNode(&node, graph, nullptr, 0, &node_params));
    HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(graphExec, 0));
    HIP_CHECK(hipStreamSynchronize(0));

    int *hostPtr = reinterpret_cast<int *>(malloc(Nbytes));
    node_params2 = getNodeTypememcpy(&src, &hostPtr, hipMemcpyDeviceToHost);
    HIP_CHECK(hipGraphNodeSetParams(node, &node_params2));
    HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphLaunch(graphExec, 0));
    HIP_CHECK(hipStreamSynchronize(0));

    // verify the results
    verifyMemcpyResult(src, hostPtr);
    HIP_CHECK(hipFree(src));
    free(dst);
    free(A_h);
    free(hostPtr);
  }
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
}
/**
 * Test Description
 * ------------------------
 *    - Create nod params of type Host and add to graph
 *    - Verify the resutls
 *    - Create another node params of type host with different value
 *    - Update the node of the graph with new node params using
 * hipGraphNodeSetParams Api
 *    - Verify the results
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphNodeSetParams.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.4
 */

TEST_CASE("Unit_hipGraphNodeSetParams_nodeTypeHost") {
  // constexpr size_t allocation_size = N * sizeof(int);
  hipGraph_t graph;
  hipGraphExec_t graphExec;
  int *A_h = reinterpret_cast<int *>(malloc(Nbytes));
  int *B_h = reinterpret_cast<int *>(malloc(Nbytes));
  std::fill_n(A_h, N, 0);
  std::fill_n(B_h, N, 0);

  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipStream_t streamForGraph;
  HIP_CHECK(hipStreamCreate(&streamForGraph));

  hipGraphNode_t node;
  hipGraphNodeParams node_params = {};
  node_params = getNodeTypeHost(A_h, reinterpret_cast<void *>(callbackSum));
  HIP_CHECK(hipGraphAddNode(&node, graph, nullptr, 0, &node_params));

  // Instantiate and launch the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  // Verify execution result
  for (size_t i = 0; i < N; i++) {
    if (A_h[i] != i + i) {
      REQUIRE(false);
    }
  }
  // update node params
  hipGraphNodeParams node_params2 = {};
  node_params2 = getNodeTypeHost(B_h, reinterpret_cast<void *>(callbackSquare));
  HIP_CHECK(hipGraphNodeSetParams(node, &node_params2));
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  // Verify execution result
  for (size_t i = 0; i < N; i++) {
    if (B_h[i] != i * i) {
      REQUIRE(false);
    }
  }

  free(A_h);
  free(B_h);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
}
/**
 * Test Description
 * ------------------------
 *    - Create nod params of type graph with node type hipMemcpyHostToDevice and
 * add to graph
 *    - Verify the resutls
 *    - Create another node params of type graph with node type
 * hipMemcpyHostToHost node
 *    - Update the node of the graph with new node params using
 * hipGraphNodeSetParams Api
 *    - Verify the results
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphNodeSetParams.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.4
 */

TEST_CASE("Unit_hipGraphNodeSetParams_nodeTypeGraph") {
  hipGraph_t graph;
  hipGraphExec_t graphExec;
  hipGraphNode_t node;
  int *A_d{nullptr}, *B_d{nullptr}, *C_d{nullptr};
  int *A_h{nullptr}, *B_h{nullptr}, *C_h{nullptr};
  HipTest::initArrays<int>(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);

  HIP_CHECK(hipGraphCreate(&graph, 0));

  for (size_t i = 0; i < N; i++) {
    A_h[i] = i;
  }

  hipStream_t streamForGraph;
  HIP_CHECK(hipStreamCreate(&streamForGraph));

  hipGraphNode_t memcpyH2D_A;
  hipGraph_t childgraph;
  HIP_CHECK(hipGraphCreate(&childgraph, 0));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_A, childgraph, nullptr, 0, A_d,
                                    A_h, Nbytes, hipMemcpyHostToDevice));
  hipGraphNodeParams node_params = {};
  node_params = getNodeTypeGraph(childgraph);

  HIP_CHECK(hipGraphAddNode(&node, graph, nullptr, 0, &node_params));

  // Instantiate and launch the childgraph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  // Verify execution result
  HIP_CHECK(hipMemcpy(B_h, A_d, Nbytes, hipMemcpyDeviceToHost));
  for (size_t i = 0; i < N; i++) {
    if (B_h[i] != A_h[i]) {
      REQUIRE(false);
    }
  }

  hipGraphNodeParams node_params2 = {};
  hipGraphNode_t memcpyH2H_A;
  hipGraph_t childgraph2;
  HIP_CHECK(hipGraphCreate(&childgraph2, 0));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2H_A, childgraph2, nullptr, 0, C_h,
                                    A_h, Nbytes, hipMemcpyHostToHost));
  node_params2 = getNodeTypeGraph(childgraph2);

  HIP_CHECK(hipGraphNodeSetParams(node, &node_params2));

  // Instantiate and launch the childgraph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));
  for (size_t i = 0; i < N; i++) {
    if (C_h[i] != A_h[i]) {
      REQUIRE(false);
    }
  }

  HipTest::freeArrays<int>(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(childgraph));
  HIP_CHECK(hipGraphDestroy(childgraph2));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
}

/**
 * End doxygen group GraphTest.
 * @}
 */
