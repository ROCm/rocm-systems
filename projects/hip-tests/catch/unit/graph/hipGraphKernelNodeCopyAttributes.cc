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

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>

/**
 * @addtogroup hipGraphKernelNodeCopyAttributes hipGraphKernelNodeCopyAttributes
 * @{
 * @ingroup GraphTest
 * `hipGraphKernelNodeCopyAttributes(hipGraphNode_t hSrc, hipGraphNode_t hDst)` -
 * Copies attributes from source node to destination node.
 */

/**
 * Test Description
 * ------------------------
 *  - Functional Test for API - hipGraphKernelNodeCopyAttributes
 *   Create graph with kernel node and add its- functionality.
 *   1) Copy kernelNodeAttribute to same graph kernel node.
 *   2) Copy kernelNodeAttribute to different graph kernel node.
 *   3) Copy kernelNodeAttribute to cloned graph kernel node.
 *   4) Copy kernelNodeAttribute to child graph kernel node.
 * Test source
 * ------------------------
 *  - /unit/graph/hipGraphKernelNodeCopyAttributes.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.6
 */

static bool validateKernelNodeAttrValue(hipKernelNodeAttrValue in, hipKernelNodeAttrValue out) {
  if ((in.accessPolicyWindow.base_ptr != out.accessPolicyWindow.base_ptr) ||
      (in.accessPolicyWindow.hitProp != out.accessPolicyWindow.hitProp) ||
      (in.accessPolicyWindow.hitRatio != out.accessPolicyWindow.hitRatio) ||
      (in.accessPolicyWindow.missProp != out.accessPolicyWindow.missProp) ||
      (in.accessPolicyWindow.num_bytes != out.accessPolicyWindow.num_bytes) ||
      (in.cooperative != out.cooperative)) {
    return false;
  }
  return true;
}

TEST_CASE("Unit_hipGraphKernelNodeCopyAttributes_Functional") {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  hipGraph_t graph;
  hipGraphExec_t graphExec;
  hipGraphNode_t memcpy_A, memcpy_B, memcpy_C, kernel_vecAdd;
  hipKernelNodeParams kNodeParams{};
  hipStream_t stream;
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;
  size_t NElem{N};

  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);

  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_A, graph, nullptr, 0, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_B, graph, nullptr, 0, B_d, B_h, Nbytes,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpy_C, graph, nullptr, 0, C_h, C_d, Nbytes,
                                    hipMemcpyDeviceToHost));

  void* kernelArgs[] = {&A_d, &B_d, &C_d, reinterpret_cast<void*>(&NElem)};
  kNodeParams.func = reinterpret_cast<void*>(HipTest::vectorADD<int>);
  kNodeParams.gridDim = dim3(blocks);
  kNodeParams.blockDim = dim3(threadsPerBlock);
  kNodeParams.sharedMemBytes = 0;
  kNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
  kNodeParams.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kernel_vecAdd, graph, nullptr, 0, &kNodeParams));

  // Create dependencies
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpy_A, &kernel_vecAdd, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &memcpy_B, &kernel_vecAdd, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &kernel_vecAdd, &memcpy_C, 1));

  hipKernelNodeAttrValue value_in, value_out;
  memset(&value_in, 0, sizeof(hipKernelNodeAttrValue));
  memset(&value_out, 0, sizeof(hipKernelNodeAttrValue));

  SECTION("Copy kernelNodeAttribute to same graph kernel node") {
    hipGraphNode_t kNode2;
    HIP_CHECK(hipGraphAddKernelNode(&kNode2, graph, nullptr, 0, &kNodeParams));

    memset(&value_in, 0, sizeof(hipKernelNodeAttrValue));
    memset(&value_out, 0, sizeof(hipKernelNodeAttrValue));

    HIP_CHECK(hipGraphKernelNodeGetAttribute(kernel_vecAdd,
                                             hipKernelNodeAttributeAccessPolicyWindow, &value_in));
    HIP_CHECK(hipGraphKernelNodeCopyAttributes(kernel_vecAdd, kNode2));
    HIP_CHECK(hipGraphKernelNodeGetAttribute(kNode2, hipKernelNodeAttributeAccessPolicyWindow,
                                             &value_out));
    REQUIRE(true == validateKernelNodeAttrValue(value_in, value_out));

    //  copy back the node attributes for functional verification only
    HIP_CHECK(hipGraphKernelNodeCopyAttributes(kNode2, kernel_vecAdd));
  }
  SECTION("Copy kernelNodeAttribute to different graph kernel node") {
    hipGraphNode_t kNode3;
    hipGraph_t graph3;
    HIP_CHECK(hipGraphCreate(&graph3, 0));
    HIP_CHECK(hipGraphAddKernelNode(&kNode3, graph3, nullptr, 0, &kNodeParams));

    memset(&value_in, 0, sizeof(hipKernelNodeAttrValue));
    memset(&value_out, 0, sizeof(hipKernelNodeAttrValue));

    HIP_CHECK(hipGraphKernelNodeGetAttribute(kernel_vecAdd,
                                             hipKernelNodeAttributeAccessPolicyWindow, &value_in));
    HIP_CHECK(hipGraphKernelNodeCopyAttributes(kernel_vecAdd, kNode3));
    HIP_CHECK(hipGraphKernelNodeGetAttribute(kNode3, hipKernelNodeAttributeAccessPolicyWindow,
                                             &value_out));
    REQUIRE(true == validateKernelNodeAttrValue(value_in, value_out));

    //  copy back the node attributes for functional verification only
    HIP_CHECK(hipGraphKernelNodeCopyAttributes(kNode3, kernel_vecAdd));
    HIP_CHECK(hipGraphDestroy(graph3));
  }
  SECTION("Copy kernelNodeAttribute to cloned graph kernel node") {
    hipGraphNode_t kNode4;
    hipGraph_t clonedGraph;
    HIP_CHECK(hipGraphClone(&clonedGraph, graph));
    HIP_CHECK(hipGraphAddKernelNode(&kNode4, clonedGraph, nullptr, 0, &kNodeParams));

    memset(&value_in, 0, sizeof(hipKernelNodeAttrValue));
    memset(&value_out, 0, sizeof(hipKernelNodeAttrValue));

    HIP_CHECK(hipGraphKernelNodeGetAttribute(kernel_vecAdd,
                                             hipKernelNodeAttributeAccessPolicyWindow, &value_in));
    HIP_CHECK(hipGraphKernelNodeCopyAttributes(kernel_vecAdd, kNode4));
    HIP_CHECK(hipGraphKernelNodeGetAttribute(kNode4, hipKernelNodeAttributeAccessPolicyWindow,
                                             &value_out));
    REQUIRE(true == validateKernelNodeAttrValue(value_in, value_out));

    //  copy back the node attributes for functional verification only
    HIP_CHECK(hipGraphKernelNodeCopyAttributes(kNode4, kernel_vecAdd));
    HIP_CHECK(hipGraphDestroy(clonedGraph));
  }
  SECTION("Copy kernelNodeAttribute to child graph kernel node") {
    hipGraphNode_t kNode5, childGraphNode;
    hipGraph_t childGraph;
    HIP_CHECK(hipGraphCreate(&childGraph, 0));
    HIP_CHECK(hipGraphAddKernelNode(&kNode5, childGraph, nullptr, 0, &kNodeParams));

    HIP_CHECK(hipGraphAddChildGraphNode(&childGraphNode, graph, nullptr, 0, childGraph));

    memset(&value_in, 0, sizeof(hipKernelNodeAttrValue));
    memset(&value_out, 0, sizeof(hipKernelNodeAttrValue));

    HIP_CHECK(hipGraphKernelNodeGetAttribute(kernel_vecAdd,
                                             hipKernelNodeAttributeAccessPolicyWindow, &value_in));
    HIP_CHECK(hipGraphKernelNodeCopyAttributes(kernel_vecAdd, kNode5));
    HIP_CHECK(hipGraphKernelNodeGetAttribute(kNode5, hipKernelNodeAttributeAccessPolicyWindow,
                                             &value_out));
    REQUIRE(true == validateKernelNodeAttrValue(value_in, value_out));

    //  copy back the node attributes for functional verification only
    HIP_CHECK(hipGraphKernelNodeCopyAttributes(kNode5, kernel_vecAdd));
    HIP_CHECK(hipGraphDestroy(childGraph));
  }

  // Instantiate and launch the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  // Verify graph execution result
  HipTest::checkVectorADD<int>(A_h, B_h, C_h, N);

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 *  - Negative Test for API - hipGraphKernelNodeCopyAttributes
 *   1) Pass source kernel node as nullptr for copy attribute api
 *   2) Pass destination kernel node as nullptr for copy attribute api
 *   3) Pass source kernel node as Uninitialize for copy attribute api
 *   4) Pass dest kernel node as Uninitialize for copy attribute api
 * Test source
 * ------------------------
 *  - unit/graph/hipGraphKernelNodeCopyAttributes.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.6
 */

TEST_CASE("Unit_hipGraphKernelNodeCopyAttributes_Attribute_Negative") {
  constexpr size_t N = 1024;
  constexpr size_t Nbytes = N * sizeof(int);
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  hipGraph_t graph;
  hipGraphNode_t memcpyNode, kNode, kNode_2;
  hipKernelNodeParams kNodeParams{};
  hipStream_t streamForGraph;
  int *A_d, *B_d, *C_d;
  int *A_h, *B_h, *C_h;
  std::vector<hipGraphNode_t> dependencies;
  size_t NElem{N};
  hipError_t ret;

  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HipTest::initArrays(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);

  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNode, graph, nullptr, 0, A_d, A_h, Nbytes,
                                    hipMemcpyHostToDevice));
  dependencies.push_back(memcpyNode);
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNode, graph, nullptr, 0, B_d, B_h, Nbytes,
                                    hipMemcpyHostToDevice));
  dependencies.push_back(memcpyNode);

  void* kernelArgs[] = {&A_d, &B_d, &C_d, reinterpret_cast<void*>(&NElem)};
  kNodeParams.func = reinterpret_cast<void*>(HipTest::vectorADD<int>);
  kNodeParams.gridDim = dim3(blocks);
  kNodeParams.blockDim = dim3(threadsPerBlock);
  kNodeParams.sharedMemBytes = 0;
  kNodeParams.kernelParams = reinterpret_cast<void**>(kernelArgs);
  kNodeParams.extra = nullptr;
  HIP_CHECK(
      hipGraphAddKernelNode(&kNode, graph, dependencies.data(), dependencies.size(), &kNodeParams));
  dependencies.clear();
  dependencies.push_back(kNode);
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyNode, graph, dependencies.data(), dependencies.size(),
                                    C_h, C_d, Nbytes, hipMemcpyDeviceToHost));
  HIP_CHECK(hipGraphAddKernelNode(&kNode_2, graph, nullptr, 0, &kNodeParams));

  SECTION("Pass source kernel node as nullptr for copy attribute api") {
    ret = hipGraphKernelNodeCopyAttributes(nullptr, kNode);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass destination kernel node as nullptr for copy attribute api") {
    ret = hipGraphKernelNodeCopyAttributes(kNode_2, nullptr);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass source kernel node as Uninitialize for copy attribute api") {
    hipGraphNode_t kNodeUninit{};
    ret = hipGraphKernelNodeCopyAttributes(kNodeUninit, kNode);
    REQUIRE(hipErrorInvalidValue == ret);
  }
  SECTION("Pass dest kernel node as Uninitialize for copy attribute api") {
    hipGraphNode_t kNodeUninit{};
    ret = hipGraphKernelNodeCopyAttributes(kNode_2, kNodeUninit);
    REQUIRE(hipErrorInvalidValue == ret);
  }

  HipTest::freeArrays(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
}

/**
 * End doxygen group GraphTest.
 * @}
 */
