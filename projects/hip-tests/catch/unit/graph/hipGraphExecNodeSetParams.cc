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

/**
 * @addtogroup hipGraphExecNodeSetParams hipGraphExecNodeSetParams
 * @{
 * @ingroup GraphTest
 * `hipGraphExecNodeSetParams(hipGraphExec_t  graphExec, hipGraphNode_t  node,
 * hipGraphNodeParams *nodeParams)` -
 * Updates parameters of a created node on executable graph
 */

/**
 * Test Description
 * ------------------------
 *    - Verify API behavior with invalid arguments:
 *      -# gGraphExec is nullptr
 *      -# node is nullptr
 *      -# nodeParams is nullptr
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphExecNodeSetParams.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.4
 */
TEST_CASE("Unit_hipGraphExecNodeSetParams_Negative_Parameters") {
  hipGraph_t graph;
  hipGraphExec_t graphExec;
  hipGraphNode_t node;
  hipGraphNodeParams node_params = {};
  char* A_d;
  size_t Nbytes = 10 * sizeof(char);

  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipMalloc(&A_d, Nbytes));
  node_params.type = hipGraphNodeTypeMemset;
  node_params.memset.dst = A_d;
  node_params.memset.elementSize = sizeof(char);
  node_params.memset.width = 10;
  node_params.memset.height = 1;
  node_params.memset.pitch = 10;
  node_params.memset.value = 99;

  HIP_CHECK(hipGraphAddNode(&node, graph, nullptr, 0, &node_params));
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, 0));

  SECTION("hGraphExec == nullptr") {
    HIP_CHECK_ERROR(hipGraphExecNodeSetParams(nullptr, node, &node_params), hipErrorInvalidValue);
  }

  SECTION("node == nullptr") {
    HIP_CHECK_ERROR(hipGraphExecNodeSetParams(graphExec, nullptr, &node_params),
                    hipErrorInvalidValue);
  }

  SECTION("node params == nullptr") {
    HIP_CHECK_ERROR(hipGraphExecNodeSetParams(graphExec, node, nullptr), hipErrorInvalidValue);
  }

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(A_d));
}
/**
 * Test Description
 * ------------------------
 *    - This will verify the new node param values are successfully
 * copied to graph node, after launching graphExec
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphExecNodeSetParams.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.4
 */
TEST_CASE("Unit_hipGraphExecNodeSetParams_Positive") {
  hipGraph_t graph;
  hipGraphExec_t graphExec;
  hipGraphNode_t node;
  hipGraphNodeParams node_params = {};
  char *A_d = nullptr, *A_h = nullptr;
  size_t Nbytes = 10 * sizeof(char);

  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipMalloc(&A_d, Nbytes));
  node_params.type = hipGraphNodeTypeMemset;
  node_params.memset.dst = A_d;
  node_params.memset.elementSize = sizeof(char);
  node_params.memset.width = 10;
  node_params.memset.height = 1;
  node_params.memset.pitch = 10;
  node_params.memset.value = 99;

  HIP_CHECK(hipGraphAddNode(&node, graph, nullptr, 0, &node_params));
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, 0));
  HIP_CHECK(hipStreamSynchronize(0));
  A_h = reinterpret_cast<char*>(malloc(Nbytes));
  HIP_CHECK(hipMemcpy(A_h, A_d, Nbytes, hipMemcpyDeviceToHost));
  for (int i = 0; i < 10; i++) {
    REQUIRE(A_h[i] == 99);
  }

  hipGraphNodeParams node_params2 = {};
  node_params2.type = hipGraphNodeTypeMemset;
  node_params2.memset.dst = A_d;
  node_params2.memset.elementSize = sizeof(char);
  node_params2.memset.width = 10;
  node_params2.memset.height = 1;
  node_params2.memset.pitch = 10;
  node_params2.memset.value = 110;

  HIP_CHECK(hipGraphExecNodeSetParams(graphExec, node, &node_params2));
  HIP_CHECK(hipGraphLaunch(graphExec, 0));
  HIP_CHECK(hipStreamSynchronize(0));
  HIP_CHECK(hipMemcpy(A_h, A_d, Nbytes, hipMemcpyDeviceToHost));
  for (int i = 0; i < 10; i++) {
    REQUIRE(A_h[i] == 110);
  }
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(A_d));
  free(A_h);
}
/**
 * End doxygen group GraphTest.
 * @}
 */
