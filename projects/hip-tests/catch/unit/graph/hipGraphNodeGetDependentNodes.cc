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

#include <functional>

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>


#include "graph_dependency_common.hh"

/**
 * @addtogroup hipGraphNodeGetDependentNodes hipGraphNodeGetDependentNodes
 * @{
 * @ingroup GraphTest
 * `hipGraphNodeGetDependentNodes(hipGraphNode_t node, hipGraphNode_t *pDependentNodes, size_t
 * *pNumDependentNodes)` - returns a node's dependent nodes
 */

/**
 * Test Description
 * ------------------------
 *    - Functional test to validate API for different number of dependent nodes:
 *        -# Validate number of dependent nodes when numDeps = num of nodes
 *        -# Validate number of dependent nodes when numDeps < num of nodes
 *        -# Validate number of dependent nodes when numDeps > num of nodes
 *        -# Validate number of dependent nodes when passed node is the last in graph
 * Test source
 * ------------------------
 *    - catch\unit\graph\hipGraphNodeGetDependentNodes.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGraphNodeGetDependentNodes_Positive_Functional") {
  using namespace std::placeholders;
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  hipGraphNode_t kernel_vecSqr{}, kernel_vecAdd{};
  hipGraphNode_t kernelmod1{}, kernelmod2{}, kernelmod3{};
  hipGraphNode_t memcpyD2H{}, memcpyH2D_A{};
  hipKernelNodeParams kernelNodeParams{};
  hipGraph_t graph{};
  size_t numDeps{};
  hipStream_t streamForGraph;
  int *A_d, *C_d;
  int *A_h, *C_h;
  int *Res1_d, *Res2_d, *Res3_d;
  int *Sum_d, *Sum_h;
  hipGraphExec_t graphExec;
  size_t NElem{N};

  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipGraphCreate(&graph, 0));

  HipTest::initArrays<int>(&A_d, &C_d, &Sum_d, &A_h, &C_h, &Sum_h, N);
  HipTest::initArrays<int>(&Res1_d, &Res2_d, &Res3_d, nullptr, nullptr, nullptr, N);

  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);

  // Initialize input buffer and vecsqr result
  for (size_t i = 0; i < N; ++i) {
    A_h[i] = i + 1;
    C_h[i] = A_h[i] * A_h[i];
  }

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_A, graph, nullptr, 0, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));

  void* kernelArgsVS[] = {&A_d, &C_d, reinterpret_cast<void*>(&NElem)};
  memset(&kernelNodeParams, 0, sizeof(kernelNodeParams));
  kernelNodeParams.func = reinterpret_cast<void*>(HipTest::vector_square<int>);
  kernelNodeParams.gridDim = dim3(blocks);
  kernelNodeParams.blockDim = dim3(threadsPerBlock);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgsVS);
  kernelNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kernel_vecSqr, graph, &memcpyH2D_A, 1, &kernelNodeParams));

  // Create multiple nodes dependent on vecSqr node.
  // Dependent nodes takes vecSqr input and computes output independently.
  std::vector<hipGraphNode_t> nodelist;
  int incValue1{1};
  void* kernelArgs1[] = {&C_d, &Res1_d, &incValue1, reinterpret_cast<void*>(&NElem)};
  memset(&kernelNodeParams, 0, sizeof(kernelNodeParams));
  kernelNodeParams.func = reinterpret_cast<void*>(updateResult<int>);
  kernelNodeParams.gridDim = dim3(blocks);
  kernelNodeParams.blockDim = dim3(threadsPerBlock);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs1);
  kernelNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kernelmod1, graph, &kernel_vecSqr, 1, &kernelNodeParams));
  nodelist.push_back(kernelmod1);

  int incValue2{2};
  void* kernelArgs2[] = {&C_d, &Res2_d, &incValue2, reinterpret_cast<void*>(&NElem)};
  memset(&kernelNodeParams, 0, sizeof(kernelNodeParams));
  kernelNodeParams.func = reinterpret_cast<void*>(updateResult<int>);
  kernelNodeParams.gridDim = dim3(blocks);
  kernelNodeParams.blockDim = dim3(threadsPerBlock);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs2);
  kernelNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kernelmod2, graph, &kernel_vecSqr, 1, &kernelNodeParams));
  nodelist.push_back(kernelmod2);

  int incValue3{3};
  void* kernelArgs3[] = {&C_d, &Res3_d, &incValue3, reinterpret_cast<void*>(&NElem)};
  memset(&kernelNodeParams, 0, sizeof(kernelNodeParams));
  kernelNodeParams.func = reinterpret_cast<void*>(updateResult<int>);
  kernelNodeParams.gridDim = dim3(blocks);
  kernelNodeParams.blockDim = dim3(threadsPerBlock);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs3);
  kernelNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kernelmod3, graph, &kernel_vecSqr, 1, &kernelNodeParams));
  nodelist.push_back(kernelmod3);

  HIP_CHECK(hipGraphNodeGetDependentNodes(kernel_vecSqr, nullptr, &numDeps));
  REQUIRE(numDeps == nodelist.size());

  SECTION("Validate number of dependent nodes when numDeps = num of nodes") {
    validateGraphNodesCommon(std::bind(hipGraphNodeGetDependentNodes, kernel_vecSqr, _1, _2),
                             nodelist, numDeps, GraphGetNodesTest::equalNumNodes);
  }

  SECTION("Validate number of dependent nodes when numDeps < num of nodes") {
    validateGraphNodesCommon(std::bind(hipGraphNodeGetDependentNodes, kernel_vecSqr, _1, _2),
                             nodelist, numDeps - 1, GraphGetNodesTest::lesserNumNodes);
  }

  SECTION("Validate number of dependent nodes when numDeps > num of nodes") {
    validateGraphNodesCommon(std::bind(hipGraphNodeGetDependentNodes, kernel_vecSqr, _1, _2),
                             nodelist, numDeps + 1, GraphGetNodesTest::greaterNumNodes);
  }

  SECTION("Validate number of dependent nodes when passed node is the last in graph") {
    hipGraphNode_t depnodes;
    numDeps = 1;
    HIP_CHECK(hipGraphNodeGetDependentNodes(kernelmod3, &depnodes, &numDeps));

    // Api expected to return success and no dependent nodes.
    REQUIRE(numDeps == 0);
  }

  // Compute sum from all dependent nodes
  void* kernelArgsAdd[] = {&Res1_d, &Res2_d, &Res3_d, &Sum_d, reinterpret_cast<void*>(&NElem)};
  memset(&kernelNodeParams, 0, sizeof(kernelNodeParams));
  kernelNodeParams.func = reinterpret_cast<void*>(vectorSum<int>);
  kernelNodeParams.gridDim = dim3(blocks);
  kernelNodeParams.blockDim = dim3(threadsPerBlock);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgsAdd);
  kernelNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kernel_vecAdd, graph, nodelist.data(), nodelist.size(),
                                  &kernelNodeParams));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H, graph, &kernel_vecAdd, 1, Sum_h, Sum_d, Nbytes,
                                    hipMemcpyDeviceToHost));

  // Instantiate and launch the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  // Validate the computation
  for (size_t i = 0; i < N; i++) {
    if (Sum_h[i] != ((C_h[i] + incValue1) + (C_h[i] + incValue2) + (C_h[i] + incValue3))) {
      INFO("Sum not matching at " << i << " Sum_h[i] " << Sum_h[i] << " C_h[i] " << C_h[i]);
      REQUIRE(false);
    }
  }

  HipTest::freeArrays<int>(A_d, C_d, Sum_d, A_h, C_h, Sum_h, false);
  HipTest::freeArrays<int>(Res1_d, Res2_d, Res3_d, nullptr, nullptr, nullptr, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
}

/**
 * Test Description
 * ------------------------
 *    - Test to verify API behavior with invalid arguments:
 *        -# Node is nullptr
 *        -# NumDependentNodes is nullptr
 *        -# Node is un-initialized/invalid parameter
 * Test source
 * ------------------------
 *    - catch\unit\graph\hipGraphNodeGetDependentNodes.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGraphNodeGetDependentNodes_Negative_Parameters") {
  hipGraph_t graph{};
  const int numBytes = 100;
  size_t numDeps{1};
  hipGraphNode_t memsetNode{}, depnodes{};
  char* A_d;

  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipMalloc(&A_d, numBytes));
  hipMemsetParams memsetParams{};
  memsetParams.dst = reinterpret_cast<void*>(A_d);
  memsetParams.value = 1;
  memsetParams.pitch = 0;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = numBytes * sizeof(char);
  memsetParams.height = 1;
  HIP_CHECK(hipGraphAddMemsetNode(&memsetNode, graph, nullptr, 0, &memsetParams));

  SECTION("node as nullptr") {
    HIP_CHECK_ERROR(hipGraphNodeGetDependentNodes(nullptr, &depnodes, &numDeps),
                    hipErrorInvalidValue);
  }

  SECTION("NumDependentNodes as nullptr") {
    HIP_CHECK_ERROR(hipGraphNodeGetDependentNodes(memsetNode, &depnodes, nullptr),
                    hipErrorInvalidValue);
  }

  SECTION("node as un-initialized/invalid parameter") {
    hipGraphNode_t uninit_node{};
    HIP_CHECK_ERROR(hipGraphNodeGetDependentNodes(uninit_node, &depnodes, &numDeps),
                    hipErrorInvalidValue);
  }

  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(A_d));
}

/**
 * End doxygen group GraphTest.
 * @}
 */
