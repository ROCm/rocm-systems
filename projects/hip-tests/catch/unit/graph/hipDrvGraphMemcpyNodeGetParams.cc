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

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>
#include <memcpy3d_tests_common.hh>
#include <numeric>

/**
 * @addtogroup hipDrvGraphMemcpyNodeGetParams hipDrvGraphMemcpyNodeGetParams
 * @{
 * @ingroup GraphTest
 * `hipDrvGraphMemcpyNodeGetParams(hipGraphNode_t hNode, HIP_MEMCPY3D*
 * nodeParams)` - Gets a memcpy node's parameters
 * ________________________
 * Test cases from other APIs:
 *  - @ref Unit_hipDrvGraphMemcpyNodeSetParams_Positive_Basic
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
 *    - unit/graph/hipDrvGraphMemcpyNodeGetParams.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipDrvGraphMemcpyNodeGetParams_Negative_Parameters") {
  HIP_CHECK(hipInit(0));
  hipDevice_t device;
  hipCtx_t context;
  HIP_CHECK(hipDeviceGet(&device, 0));
  HIP_CHECK(hipCtxCreate(&context, 0, device));

  constexpr hipExtent extent{128 * sizeof(int), 128, 8};

  LinearAllocGuard3D<int> src_alloc(extent);
  LinearAllocGuard3D<int> dst_alloc(extent);

  auto params =
      GetDrvMemcpy3DParms(dst_alloc.pitched_ptr(), make_hipPos(0, 0, 0), src_alloc.pitched_ptr(),
                          make_hipPos(0, 0, 0), dst_alloc.extent(), hipMemcpyDeviceToDevice);

  hipGraph_t graph = nullptr;
  hipGraphNode_t node = nullptr;

  SECTION("node == nullptr") {
    HIP_CHECK_ERROR(hipDrvGraphMemcpyNodeGetParams(nullptr, &params), hipErrorInvalidValue);
  }

  SECTION("pNodeParams == nullptr") {
    HIP_CHECK(hipGraphCreate(&graph, 0));
    HIP_CHECK(hipDrvGraphAddMemcpyNode(&node, graph, nullptr, 0, &params, context));
    HIP_CHECK_ERROR(hipDrvGraphMemcpyNodeGetParams(node, nullptr), hipErrorInvalidValue);
    HIP_CHECK(hipGraphDestroy(graph));
  }
#if HT_AMD
  SECTION("Node is destroyed") {
    HIP_CHECK(hipGraphCreate(&graph, 0));
    HIP_CHECK(hipDrvGraphAddMemcpyNode(&node, graph, nullptr, 0, &params, context));
    HIP_CHECK(hipGraphDestroy(graph));
    HIP_CHECK_ERROR(hipDrvGraphMemcpyNodeGetParams(node, &params), hipErrorInvalidValue);
  }
#endif

  HIP_CHECK(hipCtxPopCurrent(&context));
  HIP_CHECK(hipCtxDestroy(context));
}
/**
 * Test Description
 * ------------------------
 *    - Create graph node with memcopy parameters and add to graph,
 *    get the parameters from the created graph node and compare
 *    the all the values.
 * Test source
 * ------------------------
 *    - unit/graph/hipDrvGraphMemcpyNodeGetParams.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.4
 */

TEST_CASE("Unit_hipDrvGraphMemcpyNodeGetParams_Positive") {
  size_t size = 10;
  size_t numW = size * sizeof(int);
  // Host Vectors
  std::vector<int> A_h(numW);
  std::vector<int> B_h(numW);
  hipCtx_t context;
  // Initialization
  std::iota(A_h.begin(), A_h.end(), 1);
  std::fill_n(B_h.begin(), size, 0);
  int deviceid = 0;
  hipGraph_t graph;
  hipStream_t streamForGraph;
  hipGraphExec_t graphExec;
  hipGraphNode_t node;

  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipSetDevice(deviceid));
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipCtxCreate(&context, 0, deviceid));

  HIP_MEMCPY3D memCpy_params{};
  memset(&memCpy_params, 0x0, sizeof(HIP_MEMCPY3D));
  memCpy_params.srcXInBytes = 0;
  memCpy_params.srcY = 0;
  memCpy_params.srcZ = 0;
  memCpy_params.dstXInBytes = 0;
  memCpy_params.dstY = 0;
  memCpy_params.dstZ = 0;
  memCpy_params.WidthInBytes = numW;
  memCpy_params.Height = 1;
  memCpy_params.Depth = 1;
  memCpy_params.srcMemoryType = hipMemoryTypeHost;
  memCpy_params.dstMemoryType = hipMemoryTypeHost;
  memCpy_params.srcHost = A_h.data();
  memCpy_params.srcPitch = numW;
  memCpy_params.srcHeight = 1;
  memCpy_params.dstHost = B_h.data();
  memCpy_params.dstPitch = numW;
  memCpy_params.dstHeight = 1;

  HIP_CHECK(hipDrvGraphAddMemcpyNode(&node, graph, nullptr, 0, &memCpy_params, context));
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));
  REQUIRE(memcmp(A_h.data(), B_h.data(), numW) == 0);

  HIP_MEMCPY3D memCpyGetParams{};
  HIP_CHECK(hipDrvGraphMemcpyNodeGetParams(node, &memCpyGetParams));

  REQUIRE(memCpy_params.srcXInBytes == memCpyGetParams.srcXInBytes);
  REQUIRE(memCpy_params.srcY == memCpyGetParams.srcY);
  REQUIRE(memCpy_params.srcZ == memCpyGetParams.srcZ);
  REQUIRE(memCpy_params.dstXInBytes == memCpyGetParams.dstXInBytes);
  REQUIRE(memCpy_params.dstY == memCpyGetParams.dstY);
  REQUIRE(memCpy_params.dstZ == memCpyGetParams.dstZ);
  REQUIRE(memCpy_params.WidthInBytes == memCpyGetParams.WidthInBytes);
  REQUIRE(memCpy_params.Height == memCpyGetParams.Height);
  REQUIRE(memCpy_params.Depth == memCpyGetParams.Depth);
  REQUIRE(memCpy_params.srcMemoryType == memCpyGetParams.srcMemoryType);
  REQUIRE(memCpy_params.dstMemoryType == memCpyGetParams.dstMemoryType);
  REQUIRE(memCpy_params.srcHost == memCpyGetParams.srcHost);
  REQUIRE(memCpy_params.srcPitch == memCpyGetParams.srcPitch);
  REQUIRE(memCpy_params.srcHeight == memCpyGetParams.srcHeight);
  REQUIRE(memCpy_params.dstHost == memCpyGetParams.dstHost);
  REQUIRE(memCpy_params.dstPitch == memCpyGetParams.dstPitch);
  REQUIRE(memCpy_params.dstHeight == memCpyGetParams.dstHeight);

  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
  HIP_CHECK(hipCtxPopCurrent(&context));
  HIP_CHECK(hipCtxDestroy(context));
}
/**
 * End doxygen group GraphTest.
 * @}
 */
