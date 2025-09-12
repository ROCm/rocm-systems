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

#include <hip_test_defgroups.hh>
#include <hip_test_common.hh>
#include <resource_guards.hh>

/**
 * @addtogroup hipGraphMemcpyNodeGetParams hipGraphMemcpyNodeGetParams
 * @{
 * @ingroup GraphTest
 * `hipGraphMemcpyNodeGetParams(hipGraphNode_t node, hipMemcpy3DParms *pNodeParams)` -
 * 	Gets a memcpy node's parameters
 * ________________________
 * Test cases from other APIs:
 *  - @ref Unit_hipGraphMemcpyNodeSetParams_Positive_Basic
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
 *    - unit/graph/hipGraphMemcpyNodeGetParams.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE("Unit_hipGraphMemcpyNodeGetParams_Negative_Parameters") {
  CHECK_IMAGE_SUPPORT
  constexpr hipExtent extent{128 * sizeof(int), 128, 8};

  LinearAllocGuard3D<int> src_alloc(extent);
  LinearAllocGuard3D<int> dst_alloc(extent);

  hipMemcpy3DParms params = {};
  params.srcPtr = src_alloc.pitched_ptr();
  params.srcPos = make_hipPos(0, 0, 0);
  params.dstPtr = dst_alloc.pitched_ptr();
  params.dstPos = make_hipPos(0, 0, 0);
  params.extent = extent;
  params.kind = hipMemcpyDeviceToDevice;

  hipGraph_t graph = nullptr;
  hipGraphNode_t node = nullptr;

  SECTION("node == nullptr") {
    HIP_CHECK_ERROR(hipGraphMemcpyNodeGetParams(nullptr, &params), hipErrorInvalidValue);
  }

  SECTION("pNodeParams == nullptr") {
    HIP_CHECK(hipGraphCreate(&graph, 0));
    HIP_CHECK(hipGraphAddMemcpyNode(&node, graph, nullptr, 0, &params));
    HIP_CHECK_ERROR(hipGraphMemcpyNodeGetParams(node, nullptr), hipErrorInvalidValue);
    HIP_CHECK(hipGraphDestroy(graph));
  }

#if HT_NVIDIA  // Disabled on AMD due to defect - EXSWHTEC-208
  SECTION("Node is destroyed") {
    HIP_CHECK(hipGraphCreate(&graph, 0));
    HIP_CHECK(hipGraphAddMemcpyNode(&node, graph, nullptr, 0, &params));
    HIP_CHECK(hipGraphDestroy(graph));
    HIP_CHECK_ERROR(hipGraphMemcpyNodeGetParams(node, &params), hipErrorInvalidValue);
  }
#endif
}

/**
 * End doxygen group GraphTest.
 * @}
 */
