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

#include <hip_test_common.hh>
#include <numeric>

/**
 * @addtogroup hipDrvGraphExecMemcpyNodeSetParams hipDrvGraphExecMemcpyNodeSetParams
 * @{
 * @ingroup GraphTest
 * `hipDrvGraphExecMemcpyNodeSetParams((hipGraphExec_t hGraphExec,
 * hipGraphNode_t hNode, const HIP_MEMCPY3D *copyParams, hipCt_t ctx)` -
 * Sets the parameters for a memcpy node in the given graphExec
 */
/**
 * Test Description
 * ------------------------
 *  - Verify API behavior with invalid arguments:
 *      -# graphExec is nullptr
 *      -# node is nullptr
 *
 * Test source
 * ------------------------
 *  - /unit/graph/hipDrvGraphExecMemcpyNodeSetParams.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.4
 */
TEST_CASE("Unit_hipDrvGraphExecMemcpyNodeSetParams_Negative") {
  size_t size = 10;
  size_t numW = size * sizeof(int);
  // Host Vectors
  std::vector<int> A_h(numW);
  std::vector<int> B_h(numW);
  hipCtx_t context;
  // Initialization
  std::iota(A_h.begin(), A_h.end(), 1);
  std::fill_n(B_h.begin(), size, 2);
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
  SECTION("graphExec is nullptr") {
    HIP_CHECK_ERROR(hipDrvGraphExecMemcpyNodeSetParams(nullptr, node, &memCpy_params, context),
                    hipErrorInvalidValue);
  }
  SECTION("node is nullptr") {
    HIP_CHECK_ERROR(hipDrvGraphExecMemcpyNodeSetParams(graphExec, nullptr, &memCpy_params, context),
                    hipErrorInvalidValue);
  }
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
  HIP_CHECK(hipCtxDestroy(context));
}
/**
 * Test Description
 * ------------------------
 *  - It will verify soure data is copied destionation
 *  after adding new memcpy node.
 *  First will create and add mem copy node to graph.
 *  Create another mem copy param node with new source data and
 *  add to the graphExec using hipDrvGraphExecMemcpyNodeSetParams API and
 *  lauch the graph. Compare soure and destination data.
 *
 * Test source
 * ------------------------
 *  - /unit/graph/hipDrvGraphExecMemcpyNodeSetParams.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.4
 */

TEST_CASE("Unit_hipDrvGraphExecMemcpyNodeSetParams_Positive") {
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
  // Host Vectors for source
  std::vector<int> C_h(numW);
  std::iota(C_h.begin(), C_h.end(), 10);

  HIP_MEMCPY3D memCpy_params2{};
  memset(&memCpy_params2, 0x0, sizeof(HIP_MEMCPY3D));
  memCpy_params2.srcXInBytes = 0;
  memCpy_params2.srcY = 0;
  memCpy_params2.srcZ = 0;
  memCpy_params2.dstXInBytes = 0;
  memCpy_params2.dstY = 0;
  memCpy_params2.dstZ = 0;
  memCpy_params2.WidthInBytes = numW;
  memCpy_params2.Height = 1;
  memCpy_params2.Depth = 1;
  memCpy_params2.srcMemoryType = hipMemoryTypeHost;
  memCpy_params2.dstMemoryType = hipMemoryTypeHost;
  memCpy_params2.srcHost = C_h.data();
  memCpy_params2.srcPitch = numW;
  memCpy_params2.srcHeight = 1;
  memCpy_params2.dstHost = B_h.data();
  memCpy_params2.dstPitch = numW;
  memCpy_params2.dstHeight = 1;

  HIP_CHECK(hipDrvGraphExecMemcpyNodeSetParams(graphExec, node, &memCpy_params2, context));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  REQUIRE(memcmp(C_h.data(), B_h.data(), numW) == 0);

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
  HIP_CHECK(hipCtxDestroy(context));
}
/**
 * End doxygen group GraphTest.
 * @}
 */
