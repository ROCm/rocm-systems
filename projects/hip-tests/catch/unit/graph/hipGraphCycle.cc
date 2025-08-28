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

/**
Testcase Scenarios :
 1) Purpose is to check for topology of graph
 2) Basic tests with manually added nodes, make graph cyclic
 3) Basic tests with manually added nodes, remove edges from graph making sure
    node levels are correct if graph is not cyclic in a previously cyclic graph.
*/

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>

/**
 * Tests basic functionality of cycle detection in hipGraph APIs by
 * Adding manual empty nodes
 * Cyclic graph, cycle formation first, then adding more nodes
 */
TEST_CASE("Unit_hipGraph_BasicCyclic1") {
  hipGraph_t graph;
  hipStream_t streamForGraph;
  hipGraphExec_t graphExec;
  hipGraphNode_t emptyNode1, emptyNode2, emptyNode3, emptyNode4, emptyNode5, emptyNode6;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  // Create emptyNode and add it to graph with dependency
  HIP_CHECK(hipGraphAddEmptyNode(&emptyNode1, graph, nullptr, 0));
  HIP_CHECK(hipGraphAddEmptyNode(&emptyNode2, graph, nullptr, 0));
  HIP_CHECK(hipGraphAddEmptyNode(&emptyNode3, graph, nullptr, 0));
  HIP_CHECK(hipGraphAddEmptyNode(&emptyNode4, graph, nullptr, 0));
  HIP_CHECK(hipGraphAddEmptyNode(&emptyNode5, graph, nullptr, 0));
  HIP_CHECK(hipGraphAddEmptyNode(&emptyNode6, graph, nullptr, 0));
  HIP_CHECK(hipGraphAddDependencies(graph, &emptyNode1, &emptyNode2, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &emptyNode2, &emptyNode3, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &emptyNode3, &emptyNode1, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &emptyNode4, &emptyNode1, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &emptyNode5, &emptyNode4, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &emptyNode5, &emptyNode6, 1));
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  REQUIRE(hipErrorInvalidValue == hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Tests basic functionality of cycle detection in hipGraph APIs by
 * Adding manual empty nodes
 * Cyclic graph, cycle formation first, Remove edge to resolve cycle
 */
TEST_CASE("Unit_hipGraph_BasicCyclic2") {
  hipGraph_t graph;
  hipStream_t streamForGraph;
  hipGraphExec_t graphExec;
  hipGraphNode_t emptyNode1, emptyNode2, emptyNode3, emptyNode4, emptyNode5;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  // Create emptyNode and add it to graph with dependency
  HIP_CHECK(hipGraphAddEmptyNode(&emptyNode1, graph, nullptr, 0));
  HIP_CHECK(hipGraphAddEmptyNode(&emptyNode2, graph, nullptr, 0));
  HIP_CHECK(hipGraphAddEmptyNode(&emptyNode3, graph, nullptr, 0));
  HIP_CHECK(hipGraphAddEmptyNode(&emptyNode4, graph, nullptr, 0));
  HIP_CHECK(hipGraphAddEmptyNode(&emptyNode5, graph, nullptr, 0));
  HIP_CHECK(hipGraphAddDependencies(graph, &emptyNode1, &emptyNode2, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &emptyNode2, &emptyNode3, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &emptyNode3, &emptyNode1, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &emptyNode4, &emptyNode1, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &emptyNode5, &emptyNode4, 1));
  HIP_CHECK(hipGraphRemoveDependencies(graph, &emptyNode3, &emptyNode1, 1));
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
}

/**
 * Tests basic functionality of cycle detection in hipGraph APIs by
 * Adding manual empty nodes
 * Cyclic graph, cycle formation first, Remove edge causes disconnected graph which is still
 * cyclic
 */
TEST_CASE("Unit_hipGraph_BasicCyclic3") {
  hipGraph_t graph;
  hipStream_t streamForGraph;
  hipGraphExec_t graphExec;
  hipGraphNode_t emptyNode1, emptyNode2, emptyNode3, emptyNode4, emptyNode5, emptyNode6;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  // Create emptyNode and add it to graph with dependency
  HIP_CHECK(hipGraphAddEmptyNode(&emptyNode1, graph, nullptr, 0));
  HIP_CHECK(hipGraphAddEmptyNode(&emptyNode2, graph, nullptr, 0));
  HIP_CHECK(hipGraphAddEmptyNode(&emptyNode3, graph, nullptr, 0));
  HIP_CHECK(hipGraphAddEmptyNode(&emptyNode4, graph, nullptr, 0));
  HIP_CHECK(hipGraphAddEmptyNode(&emptyNode5, graph, nullptr, 0));
  HIP_CHECK(hipGraphAddEmptyNode(&emptyNode6, graph, nullptr, 0));
  HIP_CHECK(hipGraphAddDependencies(graph, &emptyNode1, &emptyNode2, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &emptyNode2, &emptyNode3, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &emptyNode3, &emptyNode1, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &emptyNode4, &emptyNode1, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &emptyNode5, &emptyNode4, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &emptyNode5, &emptyNode6, 1));
  HIP_CHECK(hipGraphRemoveDependencies(graph, &emptyNode5, &emptyNode4, 1));
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  REQUIRE(hipErrorInvalidValue == hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Tests basic functionality of cycle detection in hipGraph APIs by
 * Adding manual empty nodes
 * Uncyclic graph, removing edge from middle of linear graph
 */
TEST_CASE("Unit_hipGraph_BasicCyclic4") {
  int N = 1024 * 1024;
  int Nbytes = N * sizeof(int);
  hipGraph_t graph;
  hipGraphExec_t graphExec;
  hipStream_t stream;
  int *X_d, *Y_d, *X_h, *Y_h;

  HipTest::initArrays<int>(&X_d, &Y_d, nullptr, &X_h, &Y_h, nullptr, N, false);

  hipGraphNode_t kMemCpyH2D_X, memcpyD2D, memcpyD2H_RC, emptyNode1;

  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipGraphCreate(&graph, 0));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&kMemCpyH2D_X, graph, nullptr, 0, X_d, X_h, Nbytes,
                                    hipMemcpyHostToDevice));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2D, graph, nullptr, 0, Y_d, X_d, Nbytes,
                                    hipMemcpyDeviceToDevice));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_RC, graph, nullptr, 0, Y_h, Y_d, Nbytes,
                                    hipMemcpyDeviceToHost));
  HIP_CHECK(hipGraphAddEmptyNode(&emptyNode1, graph, nullptr, 0));

  HIP_CHECK(hipGraphAddDependencies(graph, &kMemCpyH2D_X, &emptyNode1, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &emptyNode1, &memcpyD2D, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyD2D, &memcpyD2H_RC, 1));

  HIP_CHECK(hipGraphRemoveDependencies(graph, &kMemCpyH2D_X, &emptyNode1, 1));
  HIP_CHECK(hipGraphRemoveDependencies(graph, &emptyNode1, &memcpyD2D, 1));
  HIP_CHECK(hipGraphDestroyNode(emptyNode1));

  HIP_CHECK(hipGraphAddDependencies(graph, &kMemCpyH2D_X, &memcpyD2D, 1));

  // Instantiate and launch the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // Verify graph result as X_h == Y_h
  for (size_t i = 0; i < N; i++) {
    if (Y_h[i] != X_h[i]) {
      INFO("Validation failed for graph at index " << i << " Y_h[i] " << Y_h[i] << " X_h[i] "
                                                   << X_h[i]);
      REQUIRE(false);
    }
  }
  HipTest::freeArrays<int>(X_d, Y_d, nullptr, X_h, Y_h, nullptr, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Tests basic functionality of cycle detection in hipGraph APIs by
 * Adding manual empty nodes
 * cyclic graph, removing edge to resolve cycle and remove edge from middle of graph
 */
TEST_CASE("Unit_hipGraph_BasicCyclic5") {
  int N = 1024 * 1024;
  int Nbytes = N * sizeof(int);
  hipGraph_t graph;
  hipGraphExec_t graphExec;
  hipStream_t stream;
  int *X_d, *Y_d, *X_h, *Y_h;

  HipTest::initArrays<int>(&X_d, &Y_d, nullptr, &X_h, &Y_h, nullptr, N, false);

  hipGraphNode_t kMemCpyH2D_X, memcpyD2D, memcpyD2H_RC, emptyNode1, emptyNode2, emptyNode3;

  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipGraphCreate(&graph, 0));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&kMemCpyH2D_X, graph, nullptr, 0, X_d, X_h, Nbytes,
                                    hipMemcpyHostToDevice));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2D, graph, nullptr, 0, Y_d, X_d, Nbytes,
                                    hipMemcpyDeviceToDevice));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_RC, graph, nullptr, 0, Y_h, Y_d, Nbytes,
                                    hipMemcpyDeviceToHost));
  HIP_CHECK(hipGraphAddEmptyNode(&emptyNode1, graph, nullptr, 0));
  HIP_CHECK(hipGraphAddEmptyNode(&emptyNode2, graph, nullptr, 0));
  HIP_CHECK(hipGraphAddEmptyNode(&emptyNode3, graph, nullptr, 0));

  HIP_CHECK(hipGraphAddDependencies(graph, &kMemCpyH2D_X, &emptyNode1, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &emptyNode1, &memcpyD2D, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyD2D, &memcpyD2H_RC, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyD2H_RC, &emptyNode2, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &emptyNode2, &emptyNode3, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &emptyNode3, &memcpyD2H_RC, 1));

  HIP_CHECK(hipGraphRemoveDependencies(graph, &kMemCpyH2D_X, &emptyNode1, 1));
  HIP_CHECK(hipGraphRemoveDependencies(graph, &emptyNode1, &memcpyD2D, 1));
  HIP_CHECK(hipGraphDestroyNode(emptyNode1));

  HIP_CHECK(hipGraphAddDependencies(graph, &kMemCpyH2D_X, &memcpyD2D, 1));
  HIP_CHECK(hipGraphRemoveDependencies(graph, &emptyNode3, &memcpyD2H_RC, 1));

  // Instantiate and launch the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // Verify graph result as X_h == Y_h
  for (size_t i = 0; i < N; i++) {
    if (Y_h[i] != X_h[i]) {
      INFO("Validation failed for graph at index " << i << " Y_h[i] " << Y_h[i] << " X_h[i] "
                                                   << X_h[i]);
      REQUIRE(false);
    }
  }
  HipTest::freeArrays<int>(X_d, Y_d, nullptr, X_h, Y_h, nullptr, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
}
