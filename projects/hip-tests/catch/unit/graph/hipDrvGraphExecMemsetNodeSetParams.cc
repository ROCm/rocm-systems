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
#include <vector>

const size_t width = 1024;
const size_t height = 1;
const int N = width * height;
const int value = 120;

/**
 * Test Description
 * ------------------------
 * - Test case to validate basic functionality of
 *   hipDrvGraphExecMemsetNodeSetParams with below scenario,
 *   1) Allocate device memory, host memory
 *   2) Create Graph
 *   3) Add Memset node to Graph to set the elements value to 120
 *   4) Add Memcpy node to Graph to copy from device to host memory
 *   4) Take GraphExec and Instantiate the graph
 *   5) Launch the Graph using GraphExec
 *   6) Validate the host array and all elements should contain value 120
 *   7) Prapare new Memset node parameters with value to 121
 *   8) Update the Node parameter in the memset node with new Memset node
 *      parameters using hipDrvGraphExecMemsetNodeSetParams
 *   9) Launch the Graph using GraphExec
 *   10) Again Validate the host array and all elements should contain
 *       value 121
 * Test source
 * ------------------------
 * - catch/unit/memory/hipDrvGraphExecMemsetNodeSetParams.cc
 */
TEST_CASE("Unit_hipDrvGraphExecMemsetNodeSetParams_BasicPositive") {
  CHECK_IMAGE_SUPPORT

  HIP_CHECK(hipInit(0));
  hipDevice_t device;
  hipCtx_t context;
  HIP_CHECK(hipDeviceGet(&device, 0));
  HIP_CHECK(hipCtxCreate(&context, 0, device));

  size_t pitch;
  hipDeviceptr_t devMemSrc;
  HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&devMemSrc), &pitch, width, height));

  char* hostMemDst = new char[N];
  REQUIRE(hostMemDst != nullptr);
  for (int i = 0; i < N; i++) {
    hostMemDst[i] = 0;
  }

  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphNode_t memsetNode, memcpyNode;

  // Prepare memset node
  hipMemsetParams initialMemsetParams{};
  initialMemsetParams.dst = reinterpret_cast<void*>(devMemSrc);
  initialMemsetParams.pitch = pitch;
  initialMemsetParams.elementSize = sizeof(char);
  initialMemsetParams.width = width;
  initialMemsetParams.height = height;
  initialMemsetParams.value = value;

  HIP_CHECK(
      hipDrvGraphAddMemsetNode(&memsetNode, graph, nullptr, 0, &initialMemsetParams, context));

  // Prepare memcpyNode
  ::std::vector<hipGraphNode_t> memcpyNodeDependencies;
  memcpyNodeDependencies.push_back(memsetNode);

  HIP_MEMCPY3D memcpyParams{};
  memcpyParams.srcMemoryType = hipMemoryTypeDevice;
  memcpyParams.srcDevice = devMemSrc;
  memcpyParams.srcPitch = pitch;
  memcpyParams.dstMemoryType = hipMemoryTypeHost;
  memcpyParams.dstHost = hostMemDst;
  memcpyParams.dstPitch = width;
  memcpyParams.srcXInBytes = 0;
  memcpyParams.srcY = 0;
  memcpyParams.srcZ = 0;
  memcpyParams.dstXInBytes = 0;
  memcpyParams.dstY = 0;
  memcpyParams.dstZ = 0;
  memcpyParams.WidthInBytes = width;
  memcpyParams.Height = height;
  memcpyParams.Depth = 1;

  HIP_CHECK(hipDrvGraphAddMemcpyNode(&memcpyNode, graph, memcpyNodeDependencies.data(),
                                     memcpyNodeDependencies.size(), &memcpyParams, context));

  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, nullptr));
  HIP_CHECK(hipStreamSynchronize(0));

  for (int i = 0; i < N; i++) {
    REQUIRE(hostMemDst[i] == value);
  }

  hipMemsetParams newMemsetParams{};
  newMemsetParams.dst = reinterpret_cast<void*>(devMemSrc);
  newMemsetParams.pitch = pitch;
  newMemsetParams.elementSize = sizeof(char);
  newMemsetParams.width = width;
  newMemsetParams.height = height;
  newMemsetParams.value = value + 1;

  // Update with new memset node params
  HIP_CHECK(hipDrvGraphExecMemsetNodeSetParams(graphExec, memsetNode, &newMemsetParams, context));

  for (int i = 0; i < N; i++) {
    hostMemDst[i] = 0;
  }

  HIP_CHECK(hipGraphLaunch(graphExec, nullptr));
  HIP_CHECK(hipStreamSynchronize(0));

  for (int i = 0; i < N; i++) {
    REQUIRE(hostMemDst[i] == (value + 1));
  }

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(reinterpret_cast<void*>(devMemSrc)));
  delete[] hostMemDst;
  HIP_CHECK(hipCtxDestroy(context));
}

/**
 * Test Description
 * ------------------------
 * - Test case to validate below scenarios,
 *    1) With Invalid Graph Exec
 *    2) With Invalid Node
 *    3) With Invalid Node params
 *    4) With Invalid destination address in Node params
 * ------------------------
 * - catch/unit/memory/hipDrvGraphExecMemsetNodeSetParams.cc
 */
TEST_CASE("Unit_hipDrvGraphExecMemsetNodeSetParams_Negative") {
  CHECK_IMAGE_SUPPORT

  HIP_CHECK(hipInit(0));
  hipDevice_t device;
  hipCtx_t context;
  HIP_CHECK(hipDeviceGet(&device, 0));
  HIP_CHECK(hipCtxCreate(&context, 0, device));

  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  size_t pitch;
  hipDeviceptr_t devMemSrc;
  HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&devMemSrc), &pitch, width, height));

  // Prepare memset node
  hipMemsetParams memsetParams{};
  memsetParams.dst = reinterpret_cast<void*>(devMemSrc);
  memsetParams.pitch = pitch;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = width;
  memsetParams.height = height;
  memsetParams.value = value;

  hipGraphNode_t memsetNode;
  HIP_CHECK(hipDrvGraphAddMemsetNode(&memsetNode, graph, nullptr, 0, &memsetParams, context));

  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  SECTION("Invalid Graph Exec") {
    HIP_CHECK_ERROR(hipDrvGraphExecMemsetNodeSetParams(nullptr, memsetNode, &memsetParams, context),
                    hipErrorInvalidValue);
  }
  SECTION("Invalid Node") {
    HIP_CHECK_ERROR(hipDrvGraphExecMemsetNodeSetParams(graphExec, nullptr, &memsetParams, context),
                    hipErrorInvalidValue);
  }
  SECTION("Invalid Node params") {
    HIP_CHECK_ERROR(hipDrvGraphExecMemsetNodeSetParams(graphExec, memsetNode, nullptr, context),
                    hipErrorInvalidValue);
  }
  SECTION("Invalid destination address in Node params") {
    hipMemsetParams memsetParams{};
    memsetParams.dst = 0;
    memsetParams.pitch = pitch;
    memsetParams.elementSize = sizeof(char);
    memsetParams.width = width;
    memsetParams.height = height;
    memsetParams.value = value;

    HIP_CHECK_ERROR(hipDrvGraphExecMemsetNodeSetParams(graphExec, memsetNode, nullptr, context),
                    hipErrorInvalidValue);
  }

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(reinterpret_cast<void*>(devMemSrc)));
  HIP_CHECK(hipCtxDestroy(context));
}
