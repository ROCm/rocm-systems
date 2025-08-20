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

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip_test_common.hh>
#include <resource_guards.hh>

/**
 * @addtogroup hipGraphMemsetNodeGetParams hipGraphMemsetNodeGetParams
 * @{
 * @ingroup GraphTest
 * `hipGraphMemsetNodeGetParams(hipGraphNode_t node, hipMemsetParams *pNodeParams)` -
 * 	Gets a memset node's parameters
 * ________________________
 * Test cases from other APIs:
 *  - @ref Unit_hipGraphMemsetNodeSetParams_Positive_Basic
 *  - @ref Unit_hipGraphExecMemsetNodeSetParams_Positive_Basic
 */

/**
 * Test Description
 * ------------------------
 *    - Verify API behaviour with invalid arguments:
 *      -# node is nullptr
 *      -# pNodeParams is nullptr
 *      -# node is destroyed
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphMemsetNodeGetParams.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGraphMemsetNodeGetParams_Negative_Parameters") {
  CHECK_IMAGE_SUPPORT

  LinearAllocGuard2D<int> alloc(1, 1);
  hipMemsetParams params = {};
  params.dst = alloc.ptr();
  params.elementSize = sizeof(int);
  params.width = 1;
  params.height = 1;

  hipGraph_t graph = nullptr;
  hipGraphNode_t node = nullptr;

  SECTION("node == nullptr") {
    HIP_CHECK_ERROR(hipGraphMemsetNodeGetParams(nullptr, &params), hipErrorInvalidValue);
  }

  SECTION("pNodeParams == nullptr") {
    HIP_CHECK(hipGraphCreate(&graph, 0));
    HIP_CHECK(hipGraphAddMemsetNode(&node, graph, nullptr, 0, &params));
    HIP_CHECK_ERROR(hipGraphMemsetNodeGetParams(node, nullptr), hipErrorInvalidValue);
    HIP_CHECK(hipGraphDestroy(graph));
  }

// Disabled on AMD due to defect - EXSWHTEC-208
#if 0
  SECTION("Node is destroyed") {
    HIP_CHECK(hipGraphCreate(&graph, 0));
    HIP_CHECK(hipGraphAddMemsetNode(&node, graph, nullptr, 0, &params));
    HIP_CHECK(hipGraphDestroy(graph));
    HIP_CHECK_ERROR(hipGraphMemsetNodeGetParams(node, &params), hipErrorInvalidValue);
  }
#endif
}

/**
 * End doxygen group GraphTest.
 * @}
 */
