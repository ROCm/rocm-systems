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

/**
Testcase Scenarios :
 1) Create and add empty node to graph and verify addition is successful.
 2) Negative Scenarios
 3) Functional Test to use empty node as barrier to wait for multiple nodes.
    In a graph add 3 independent memcpy_h2d nodes, add an empty node with
    dependencies on the 3 memcpy_h2d nodes, add 3 independent kernel nodes,
    add another empty node with dependencies on the 3 kernel nodes and
    add 3 independent memcpy_d2h nodes with dependencies on previous empty
    node. Execute the graph and validate the results.
*/
#include <hip_test_checkers.hh>
#include <hip_test_common.hh>
#include <hip_test_kernels.hh>

#define TEST_LOOP_SIZE 50
/**
 * Functional Test to add empty node with dependencies
 */
TEST_CASE("Unit_hipGraphAddEmptyNode_Functional") {
  char *pOutBuff_d{};
  constexpr size_t size = 1024;
  hipGraph_t graph{};
  hipGraphNode_t memsetNode{}, emptyNode{};
  std::vector<hipGraphNode_t> dependencies;

  HIP_CHECK(hipMalloc(&pOutBuff_d, size));
  hipMemsetParams memsetParams{};
  memsetParams.dst = reinterpret_cast<void*>(pOutBuff_d);
  memsetParams.value = 0;
  memsetParams.pitch = 0;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = size * sizeof(char);
  memsetParams.height = 1;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipGraphAddMemsetNode(&memsetNode, graph, nullptr, 0,
                                                              &memsetParams));
  dependencies.push_back(memsetNode);

  // Create emptyNode and add it to graph with dependency
  HIP_CHECK(hipGraphAddEmptyNode(&emptyNode, graph, dependencies.data(),
                                                        dependencies.size()));

  REQUIRE(emptyNode != nullptr);
  HIP_CHECK(hipFree(pOutBuff_d));
  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Negative Scenarios hipGraphAddEmptyNode
 */
TEST_CASE("Unit_hipGraphAddEmptyNode_NegTest") {
  char *pOutBuff_d{};
  constexpr size_t size = 1024;
  hipGraph_t graph;
  hipGraphNode_t memsetNode{}, emptyNode{};
  std::vector<hipGraphNode_t> dependencies;

  HIP_CHECK(hipMalloc(&pOutBuff_d, size));
  hipMemsetParams memsetParams{};
  memsetParams.dst = reinterpret_cast<void*>(pOutBuff_d);
  memsetParams.value = 0;
  memsetParams.pitch = 0;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = size * sizeof(char);
  memsetParams.height = 1;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipGraphAddMemsetNode(&memsetNode, graph, nullptr, 0,
                                  &memsetParams));
  dependencies.push_back(memsetNode);
  // pGraphNode is nullptr
  SECTION("Null Empty Graph Node") {
    REQUIRE(hipErrorInvalidValue == hipGraphAddEmptyNode(nullptr, graph,
                          dependencies.data(), dependencies.size()));
  }
  // graph is nullptr
  SECTION("Null Graph") {
    REQUIRE(hipErrorInvalidValue == hipGraphAddEmptyNode(&emptyNode, nullptr,
                          dependencies.data(), dependencies.size()));
  }
  // pDependencies is nullptr
  SECTION("Dependencies is null") {
    REQUIRE(hipErrorInvalidValue == hipGraphAddEmptyNode(&emptyNode, graph,
                          nullptr, dependencies.size()));
  }

  HIP_CHECK(hipFree(pOutBuff_d));
  HIP_CHECK(hipGraphDestroy(graph));
}

// Function to fill input data
static void fillRandInpData(int *A1_h, int *A2_h, int *A3_h, size_t N) {
  unsigned int seed = time(nullptr);
  for (size_t i = 0; i < N; i++) {
    A1_h[i] = (HipTest::RAND_R(&seed) & 0xFF);
    A2_h[i] = (HipTest::RAND_R(&seed) & 0xFF);
    A3_h[i] = (HipTest::RAND_R(&seed) & 0xFF);
  }
}

// Function to validate result
static void validateOutData(int *A1_h, int *A2_h, size_t N) {
  for (size_t i = 0; i < N; i++) {
    int result = (A1_h[i]*A1_h[i]);
    REQUIRE(result == A2_h[i]);
  }
}
/**
 * Functional Test to use empty node as barrier to wait for multiple nodes.
 */
TEST_CASE("Unit_hipGraphAddEmptyNode_BarrierFunc") {
  size_t size = 1024;
  constexpr auto blocksPerCU = 6;
  constexpr auto threadsPerBlock = 256;
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU,
                            threadsPerBlock, size);
  hipGraph_t graph;
  std::vector<hipGraphNode_t> nodeDependencies;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  int *inputVec_d1{nullptr}, *inputVec_h1{nullptr}, *outputVec_h1{nullptr},
      *outputVec_d1{nullptr};
  int *inputVec_d2{nullptr}, *inputVec_h2{nullptr}, *outputVec_h2{nullptr},
      *outputVec_d2{nullptr};
  int *inputVec_d3{nullptr}, *inputVec_h3{nullptr}, *outputVec_h3{nullptr},
      *outputVec_d3{nullptr};
  // host and device allocation
  HipTest::initArrays<int>(&inputVec_d1, &outputVec_d1, nullptr,
               &inputVec_h1, &outputVec_h1, nullptr, size, false);
  HipTest::initArrays<int>(&inputVec_d2, &outputVec_d2, nullptr,
               &inputVec_h2, &outputVec_h2, nullptr, size, false);
  HipTest::initArrays<int>(&inputVec_d3, &outputVec_d3, nullptr,
               &inputVec_h3, &outputVec_h3, nullptr, size, false);
  // add nodes to graph
  hipGraphNode_t memcpyH2D_1, memcpyH2D_2, memcpyH2D_3;
  hipGraphNode_t vecSqr1, vecSqr2, vecSqr3;
  hipGraphNode_t memcpyD2H_1, memcpyD2H_2, memcpyD2H_3;
  hipGraphNode_t emptyNode1, emptyNode2;
  // Create memcpy h2d nodes
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_1, graph, nullptr,
  0, inputVec_d1, inputVec_h1, (sizeof(int)*size), hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_2, graph, nullptr,
  0, inputVec_d2, inputVec_h2, (sizeof(int)*size), hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_3, graph, nullptr,
  0, inputVec_d3, inputVec_h3, (sizeof(int)*size), hipMemcpyHostToDevice));
  // Create dependency list
  nodeDependencies.push_back(memcpyH2D_1);
  nodeDependencies.push_back(memcpyH2D_2);
  nodeDependencies.push_back(memcpyH2D_3);
  // Create emptyNode and add it to graph with dependency
  HIP_CHECK(hipGraphAddEmptyNode(&emptyNode1, graph, nodeDependencies.data(),
                                nodeDependencies.size()));
  nodeDependencies.clear();
  nodeDependencies.push_back(emptyNode1);
  // Creating kernel nodes
  hipKernelNodeParams kerNodeParams1{}, kerNodeParams2{}, kerNodeParams3{};
  void* kernelArgs1[] = {reinterpret_cast<void*>(&inputVec_d1),
                        reinterpret_cast<void*>(&outputVec_d1),
                        reinterpret_cast<void*>(&size)};
  kerNodeParams1.func = reinterpret_cast<void*>(HipTest::vector_square<int>);
  kerNodeParams1.gridDim = dim3(blocks);
  kerNodeParams1.blockDim = dim3(threadsPerBlock);
  kerNodeParams1.sharedMemBytes = 0;
  kerNodeParams1.kernelParams = reinterpret_cast<void**>(kernelArgs1);
  kerNodeParams1.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&vecSqr1, graph, nodeDependencies.data(),
            nodeDependencies.size(), &kerNodeParams1));
  void* kernelArgs2[] = {reinterpret_cast<void*>(&inputVec_d2),
                        reinterpret_cast<void*>(&outputVec_d2),
                        reinterpret_cast<void*>(&size)};
  kerNodeParams2.func = reinterpret_cast<void*>(HipTest::vector_square<int>);
  kerNodeParams2.gridDim = dim3(blocks);
  kerNodeParams2.blockDim = dim3(threadsPerBlock);
  kerNodeParams2.sharedMemBytes = 0;
  kerNodeParams2.kernelParams = reinterpret_cast<void**>(kernelArgs2);
  kerNodeParams2.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&vecSqr2, graph, nodeDependencies.data(),
            nodeDependencies.size(), &kerNodeParams2));
  void* kernelArgs3[] = {reinterpret_cast<void*>(&inputVec_d3),
                        reinterpret_cast<void*>(&outputVec_d3),
                        reinterpret_cast<void*>(&size)};
  kerNodeParams3.func = reinterpret_cast<void*>(HipTest::vector_square<int>);
  kerNodeParams3.gridDim = dim3(blocks);
  kerNodeParams3.blockDim = dim3(threadsPerBlock);
  kerNodeParams3.sharedMemBytes = 0;
  kerNodeParams3.kernelParams = reinterpret_cast<void**>(kernelArgs3);
  kerNodeParams3.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&vecSqr3, graph, nodeDependencies.data(),
            nodeDependencies.size(), &kerNodeParams3));
  nodeDependencies.clear();
  nodeDependencies.push_back(vecSqr1);
  nodeDependencies.push_back(vecSqr2);
  nodeDependencies.push_back(vecSqr3);
  // Create emptyNode and add it to graph with dependency
  HIP_CHECK(hipGraphAddEmptyNode(&emptyNode2, graph, nodeDependencies.data(),
                                nodeDependencies.size()));
  nodeDependencies.clear();
  nodeDependencies.push_back(emptyNode2);
  // Create memcpy d2h nodes
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_1, graph,
  nodeDependencies.data(), nodeDependencies.size(), outputVec_h1,
  outputVec_d1, (sizeof(int)*size), hipMemcpyDeviceToHost));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_2, graph,
  nodeDependencies.data(), nodeDependencies.size(), outputVec_h2,
  outputVec_d2, (sizeof(int)*size), hipMemcpyDeviceToHost));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_3, graph,
  nodeDependencies.data(), nodeDependencies.size(), outputVec_h3,
  outputVec_d3, (sizeof(int)*size), hipMemcpyDeviceToHost));
  nodeDependencies.clear();
  // Create executable graph
  hipStream_t streamForGraph;
  hipGraphExec_t graphExec{nullptr};
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr,
                                nullptr, 0));
  // Execute graph
  for (int iter = 0; iter < TEST_LOOP_SIZE; iter++) {
    fillRandInpData(inputVec_h1, inputVec_h2, inputVec_h3, size);
    HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
    HIP_CHECK(hipStreamSynchronize(streamForGraph));
    validateOutData(inputVec_h1, outputVec_h1, size);
    validateOutData(inputVec_h2, outputVec_h2, size);
    validateOutData(inputVec_h3, outputVec_h3, size);
  }
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
  // Free
  HipTest::freeArrays<int>(inputVec_d1, outputVec_d1, nullptr,
                   inputVec_h1, outputVec_h1, nullptr, false);
  HipTest::freeArrays<int>(inputVec_d2, outputVec_d2, nullptr,
                   inputVec_h2, outputVec_h2, nullptr, false);
  HipTest::freeArrays<int>(inputVec_d3, outputVec_d3, nullptr,
                   inputVec_h3, outputVec_h3, nullptr, false);
  HIP_CHECK(hipGraphDestroy(graph));
}
