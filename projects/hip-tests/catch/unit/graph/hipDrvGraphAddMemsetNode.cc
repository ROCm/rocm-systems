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

#include <functional>
#include <vector>

#include <hip_test_defgroups.hh>
#include <hip_test_common.hh>
#include <memcpy3d_tests_common.hh>
#include <resource_guards.hh>
#include <utils.hh>

#include "graph_memset_node_test_common.hh"
#include "graph_tests_common.hh"

#define SIZE 1024
static char memSetVal = 'a';

/**
 * @addtogroup hipDrvGraphAddMemsetNode hipDrvGraphAddMemsetNode
 * @{
 * @ingroup GraphTest
 * `hipDrvGraphAddMemsetNode(hipGraphNode_t* phGraphNode, hipGraph_t hGraph, const hipGraphNode_t*
 * dependencies, size_t numDependencies, const hipMemsetParams* memsetParams, hipCtx_t ctx)`
 * - Creates a memset node and adds it to a graph
 */

/**
 * Test Description
 * ------------------------
 *    - Verify that all elements of destination memory are set to the correct value.
 * The test is repeated for all valid element sizes(1, 2, 4), and several allocations of different
 * height and width, both on host and device.
 * Test source
 * ------------------------
 *    - unit/graph/hipDrvGraphAddMemsetNode.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.0
 */
TEMPLATE_TEST_CASE("Unit_hipDrvGraphAddMemsetNode_Positive_Basic", "", uint8_t, uint16_t,
                   uint32_t) {
  HIP_CHECK(hipInit(0));
  hipDevice_t device;
  hipCtx_t context;
  HIP_CHECK(hipDeviceGet(&device, 0));
  HIP_CHECK(hipCtxCreate(&context, 0, device));

  CHECK_IMAGE_SUPPORT

  const auto f = [&context](hipMemsetParams* params) {
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipGraphCreate(&graph, 0));

    hipGraphNode_t node = nullptr;
    HIP_CHECK(hipDrvGraphAddMemsetNode(&node, graph, nullptr, 0, params, context));

    hipGraphExec_t graph_exec = nullptr;
    HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

    HIP_CHECK(hipGraphLaunch(graph_exec, hipStreamPerThread));
    HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));

    HIP_CHECK(hipGraphExecDestroy(graph_exec));
    HIP_CHECK(hipGraphDestroy(graph));

    return hipSuccess;
  };

  GraphMemsetNodeCommonPositive<TestType, hipMemsetParams>(f);

  HIP_CHECK(hipCtxPopCurrent(&context));
  HIP_CHECK(hipCtxDestroy(context));
}

/**
 * Test Description
 * ------------------------
 *    - Verify API behaviour with invalid arguments:
 *        -# pGraphNode is nullptr
 *        -# graph is nullptr
 *        -# pDependencies is nullptr when numDependencies is not zero
 *        -# A node in pDependencies originates from a different graph
 *        -# numDependencies is invalid
 *        -# A node is duplicated in pDependencies
 *        -# pMemsetParams is nullptr
 *        -# pMemsetParams::dst is nullptr
 *        -# pMemsetParams::elementSize is different from 1, 2, and 4
 *        -# pMemsetParams::width is zero
 *        -# pMemsetParams::width is larger than the allocated memory region
 *        -# pMemsetParams::height is zero
 *        -# pMemsetParams::pitch is less than width when height is more than 1
 *        -# pMemsetParams::pitch * pMemsetParams::height is larger than the allocated memory region
 * Test source
 * ------------------------
 *    - unit/graph/hipDrvGraphAddMemsetNode.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipDrvGraphAddMemsetNode_Negative_Parameters") {
  using namespace std::placeholders;

  HIP_CHECK(hipInit(0));
  hipDevice_t device;
  hipCtx_t context;
  HIP_CHECK(hipDeviceGet(&device, 0));
  HIP_CHECK(hipCtxCreate(&context, 0, device));

  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  LinearAllocGuard<int> alloc(LinearAllocs::hipMalloc, 4 * sizeof(int));
  hipMemsetParams params = {};
  params.dst = alloc.ptr();
  params.elementSize = sizeof(*alloc.ptr());
  params.width = 1;
  params.height = 1;
  params.value = 42;

  GraphAddNodeCommonNegativeTests(
      std::bind(hipDrvGraphAddMemsetNode, _1, _2, _3, _4, &params, context), graph);

  hipGraphNode_t node = nullptr;
  MemsetCommonNegative(std::bind(hipDrvGraphAddMemsetNode, &node, graph, nullptr, 0, _1, context),
                       params);

  HIP_CHECK(hipGraphDestroy(graph));

  HIP_CHECK(hipCtxPopCurrent(&context));
  HIP_CHECK(hipCtxDestroy(context));
}

/**
 * Test Description
 * ------------------------
 *    - Allocate a 2D array using hipMallocPitch. Initialize the allocated memory using
 * hipDrvGraphAddMemsetNode. Copy the values in device memory to host using
 * hipDrvGraphAddMemcpyNode. Verify the results.
 * Test source
 * ------------------------
 *    - unit/graph/hipDrvGraphAddMemsetNode.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipDrvGraphAddMemsetNode_hipMallocPitch_2D") {
  HIP_CHECK(hipInit(0));
  hipDevice_t device;
  hipCtx_t context;
  HIP_CHECK(hipDeviceGet(&device, 0));
  HIP_CHECK(hipCtxCreate(&context, 0, device));

  CHECK_IMAGE_SUPPORT

  size_t width = SIZE * sizeof(char), numW{SIZE}, numH{SIZE}, pitch_A;
  char* A_d;

  hipGraph_t graph;
  std::vector<hipGraphNode_t> nodeDependencies;
  // Host memory.
  char* A_h = new char[numW * numH];
  for (size_t i = 0; i < numW; i++) {
    for (size_t j = 0; j < numH; j++) {
      *(A_h + i * numH + j) = ' ';
    }
  }
  // 2D Memory allocation hipMallocPitch
  HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&A_d), &pitch_A, width, numH));
  // Create Graph
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipGraphNode_t memsetNode, memcpyNode;
  // Add MemSet Node
  hipMemsetParams memsetParams{};
  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = reinterpret_cast<void*>(A_d);
  memsetParams.value = memSetVal;
  memsetParams.pitch = pitch_A;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = numW;
  memsetParams.height = numH;
  HIP_CHECK(hipDrvGraphAddMemsetNode(&memsetNode, graph, nullptr, 0, &memsetParams, context));
  nodeDependencies.push_back(memsetNode);

  // Add MemCpy Node
  auto srcPos = make_hipPos(0, 0, 0);
  auto dstPos = make_hipPos(0, 0, 0);
  auto srcPtr = make_hipPitchedPtr(A_d, pitch_A, numW, numH);
  auto dstPtr = make_hipPitchedPtr(A_h, width, numW, numH);
  auto extent = make_hipExtent(width, numH, 1);
  hipMemcpyKind kind = hipMemcpyDeviceToHost;

  HIP_MEMCPY3D myparms = GetDrvMemcpy3DParms(dstPtr, dstPos, srcPtr, srcPos, extent, kind);
  HIP_CHECK(hipDrvGraphAddMemcpyNode(&memcpyNode, graph, nodeDependencies.data(),
                                     nodeDependencies.size(), &myparms, context));
  nodeDependencies.clear();
  // Create executable graph
  hipStream_t streamForGraph;
  hipGraphExec_t graphExec;
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  // Verfication
  for (size_t i = 0; i < numW; i++) {
    for (size_t j = 0; j < numH; j++) {
      REQUIRE(*(A_h + i * numH + j) == memSetVal);
    }
  }
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
  delete[] A_h;
  HIP_CHECK(hipFree(A_d));

  HIP_CHECK(hipCtxPopCurrent(&context));
  HIP_CHECK(hipCtxDestroy(context));
}

/**
 * Test Description
 * ------------------------
 *    - Allocate a 1D array using hipMallocPitch. Initialize the allocated memory using
 * hipDrvGraphAddMemsetNode. Copy the values in device memory to host using
 * hipDrvGraphAddMemcpyNode. Verify the results.
 * Test source
 * ------------------------
 *    - unit/graph/hipDrvGraphAddMemsetNode.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipDrvGraphAddMemsetNode_hipMallocPitch_1D") {
  HIP_CHECK(hipInit(0));
  hipDevice_t device;
  hipCtx_t context;
  HIP_CHECK(hipDeviceGet(&device, 0));
  HIP_CHECK(hipCtxCreate(&context, 0, device));

  CHECK_IMAGE_SUPPORT

  size_t width = SIZE * sizeof(char), numW{SIZE}, pitch_A;
  char* A_d;

  // Initialize the host memory
  std::vector<char> A_h(numW, ' ');

  hipGraph_t graph;
  std::vector<hipGraphNode_t> nodeDependencies;
  // 1D Memory allocation hipMallocPitch
  HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&A_d), &pitch_A, width, 1));
  // Create Graph
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipGraphNode_t memsetNode, memcpyNode;
  // Add MemSet Node
  hipMemsetParams memsetParams{};
  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = reinterpret_cast<void*>(A_d);
  memsetParams.value = memSetVal;
  memsetParams.pitch = pitch_A;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = numW;
  memsetParams.height = 1;
  HIP_CHECK(hipDrvGraphAddMemsetNode(&memsetNode, graph, nullptr, 0, &memsetParams, context));
  nodeDependencies.push_back(memsetNode);

  // Add MemCpy Node
  auto srcPos = make_hipPos(0, 0, 0);
  auto dstPos = make_hipPos(0, 0, 0);
  auto srcPtr = make_hipPitchedPtr(A_d, pitch_A, numW, 1);
  auto dstPtr = make_hipPitchedPtr(A_h.data(), width, numW, 1);
  auto extent = make_hipExtent(width, 1, 1);
  hipMemcpyKind kind = hipMemcpyDeviceToHost;

  HIP_MEMCPY3D myparms = GetDrvMemcpy3DParms(dstPtr, dstPos, srcPtr, srcPos, extent, kind);
  HIP_CHECK(hipDrvGraphAddMemcpyNode(&memcpyNode, graph, nodeDependencies.data(),
                                     nodeDependencies.size(), &myparms, context));
  nodeDependencies.clear();

  // Create executable graph
  hipStream_t streamForGraph;
  hipGraphExec_t graphExec;
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  // Verfication
  for (size_t i = 0; i < numW; i++) {
    REQUIRE(A_h[i] == memSetVal);
  }
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
  HIP_CHECK(hipFree(A_d));

  HIP_CHECK(hipCtxPopCurrent(&context));
  HIP_CHECK(hipCtxDestroy(context));
}

/**
 * Test Description
 * ------------------------
 *    - Allocate a 2D array using hipMalloc3D. Initialize the allocated memory using
 * hipDrvGraphAddMemsetNode. Copy the values in device memory to host using
 * hipDrvGraphAddMemcpyNode. Verify the results.
 * Test source
 * ------------------------
 *    - unit/graph/hipDrvGraphAddMemsetNode.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipDrvGraphAddMemsetNode_hipMalloc3D_2D") {
  HIP_CHECK(hipInit(0));
  hipDevice_t device;
  hipCtx_t context;
  HIP_CHECK(hipDeviceGet(&device, 0));
  HIP_CHECK(hipCtxCreate(&context, 0, device));

  CHECK_IMAGE_SUPPORT

  size_t width = SIZE * sizeof(char);
  size_t numW = SIZE, numH = SIZE;

  // Host Memory
  char* A_h = new char[numW * numH];
  for (size_t i = 0; i < numW; i++) {
    for (size_t j = 0; j < numH; j++) {
      *(A_h + i * numH + j) = ' ';
    }
  }
  hipGraph_t graph;
  std::vector<hipGraphNode_t> nodeDependencies;

  hipPitchedPtr A_d;
  hipExtent extent3D = make_hipExtent(width, numH, 1);

  // Allocate 3D memory.
  HIPCHECK(hipMalloc3D(&A_d, extent3D));

  // Create Graph
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipGraphNode_t memsetNode, memcpyNode;

  // Add MemSet Node
  hipMemsetParams memsetParams{};
  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = reinterpret_cast<void*>(A_d.ptr);
  memsetParams.value = memSetVal;
  memsetParams.pitch = A_d.pitch;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = numW;
  memsetParams.height = numH;
  HIP_CHECK(hipDrvGraphAddMemsetNode(&memsetNode, graph, nullptr, 0, &memsetParams, context));
  nodeDependencies.push_back(memsetNode);

  // Add MemCpy Node
  auto srcPos = make_hipPos(0, 0, 0);
  auto dstPos = make_hipPos(0, 0, 0);
  auto srcPtr = A_d;
  auto dstPtr = make_hipPitchedPtr(A_h, width, numW, numH);
  auto extent = make_hipExtent(width, numH, 1);
  hipMemcpyKind kind = hipMemcpyDeviceToHost;

  HIP_MEMCPY3D myparms = GetDrvMemcpy3DParms(dstPtr, dstPos, srcPtr, srcPos, extent, kind);
  HIP_CHECK(hipDrvGraphAddMemcpyNode(&memcpyNode, graph, nodeDependencies.data(),
                                     nodeDependencies.size(), &myparms, context));
  nodeDependencies.clear();

  // Create executable graph
  hipStream_t streamForGraph;
  hipGraphExec_t graphExec;
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  // Verfication
  for (size_t i = 0; i < numW; i++) {
    for (size_t j = 0; j < numH; j++) {
      REQUIRE(*(A_h + i * numH + j) == memSetVal);
    }
  }
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
  delete[] A_h;
  HIP_CHECK(hipFree(A_d.ptr));

  HIP_CHECK(hipCtxPopCurrent(&context));
  HIP_CHECK(hipCtxDestroy(context));
}

/**
 * Test Description
 * ------------------------
 *    - Allocate a 1D array using hipMalloc3D. Initialize the allocated memory using
 * hipDrvGraphAddMemsetNode. Copy the values in device memory to host using
 * hipDrvGraphAddMemcpyNode. Verify the results.
 * Test source
 * ------------------------
 *    - unit/graph/hipDrvGraphAddMemsetNode.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipDrvGraphAddMemsetNode_hipMalloc3D_1D") {
  HIP_CHECK(hipInit(0));
  hipDevice_t device;
  hipCtx_t context;
  HIP_CHECK(hipDeviceGet(&device, 0));
  HIP_CHECK(hipCtxCreate(&context, 0, device));

  CHECK_IMAGE_SUPPORT

  size_t width = SIZE * sizeof(char);
  size_t numW = SIZE;

  // Initialize the host memory
  std::vector<char> A_h(numW, ' ');

  hipGraph_t graph;
  std::vector<hipGraphNode_t> nodeDependencies;

  hipPitchedPtr A_d;
  hipExtent extent1D = make_hipExtent(width, 1, 1);

  // Allocate 3D memory.
  HIPCHECK(hipMalloc3D(&A_d, extent1D));

  // Create Graph
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipGraphNode_t memsetNode, memcpyNode;

  // Add MemSet Node
  hipMemsetParams memsetParams{};
  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = reinterpret_cast<void*>(A_d.ptr);
  memsetParams.value = memSetVal;
  memsetParams.pitch = A_d.pitch;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = numW;
  memsetParams.height = 1;
  HIP_CHECK(hipDrvGraphAddMemsetNode(&memsetNode, graph, nullptr, 0, &memsetParams, context));
  nodeDependencies.push_back(memsetNode);

  // Add MemCpy Node
  auto srcPos = make_hipPos(0, 0, 0);
  auto dstPos = make_hipPos(0, 0, 0);
  auto srcPtr = A_d;
  auto dstPtr = make_hipPitchedPtr(A_h.data(), width, numW, 1);
  auto extent = make_hipExtent(width, 1, 1);
  hipMemcpyKind kind = hipMemcpyDeviceToHost;

  HIP_MEMCPY3D myparms = GetDrvMemcpy3DParms(dstPtr, dstPos, srcPtr, srcPos, extent, kind);
  HIP_CHECK(hipDrvGraphAddMemcpyNode(&memcpyNode, graph, nodeDependencies.data(),
                                     nodeDependencies.size(), &myparms, context));
  nodeDependencies.clear();

  // Create executable graph
  hipStream_t streamForGraph;
  hipGraphExec_t graphExec;
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  // Verfication
  for (size_t i = 0; i < numW; i++) {
    REQUIRE(A_h[i] == memSetVal);
  }
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph))
  HIP_CHECK(hipFree(A_d.ptr));

  HIP_CHECK(hipCtxPopCurrent(&context));
  HIP_CHECK(hipCtxDestroy(context));
}

/**
 * Test Description
 * ------------------------
 *    - Allocate a 1D array using hipMalloc. Initialize the allocated memory using
 * hipDrvGraphAddMemsetNode. Copy the values in device memory to host using
 * hipDrvGraphAddMemcpyNode. Verify the results.
 * Test source
 * ------------------------
 *    - unit/graph/hipDrvGraphAddMemsetNode.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipDrvGraphAddMemsetNode_hipMalloc_1D") {
  HIP_CHECK(hipInit(0));
  hipDevice_t device;
  hipCtx_t context;
  HIP_CHECK(hipDeviceGet(&device, 0));
  HIP_CHECK(hipCtxCreate(&context, 0, device));

  char* A_d;
  size_t NumW = SIZE;
  size_t Nbytes1D = SIZE * sizeof(char);

  // Initialize the host memory
  std::vector<char> A_h(NumW, ' ');

  // Allocate memory to Device pointer
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&A_d), Nbytes1D));

  // Create the graph
  hipGraph_t graph;
  std::vector<hipGraphNode_t> nodeDependencies;
  hipGraphNode_t memsetNode, memcpyNode;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  // Add Memset node
  hipMemsetParams memsetParams{};
  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = reinterpret_cast<void*>(A_d);
  memsetParams.value = memSetVal;
  memsetParams.pitch = Nbytes1D;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = NumW;
  memsetParams.height = 1;
  HIP_CHECK(hipDrvGraphAddMemsetNode(&memsetNode, graph, nullptr, 0, &memsetParams, context));
  nodeDependencies.push_back(memsetNode);

  // Add MemCpy Node
  hipPitchedPtr devPitchedPtr{A_d, Nbytes1D, NumW, 0};
  hipPitchedPtr hostPitchedPtr{A_h.data(), Nbytes1D, NumW, 0};
  auto srcPos = make_hipPos(0, 0, 0);
  auto dstPos = make_hipPos(0, 0, 0);
  auto srcPtr = devPitchedPtr;
  auto dstPtr = hostPitchedPtr;
  auto extent = make_hipExtent(Nbytes1D, 1, 1);
  hipMemcpyKind kind = hipMemcpyDeviceToHost;

  HIP_MEMCPY3D myparms = GetDrvMemcpy3DParms(dstPtr, dstPos, srcPtr, srcPos, extent, kind);
  HIP_CHECK(hipDrvGraphAddMemcpyNode(&memcpyNode, graph, nodeDependencies.data(),
                                     nodeDependencies.size(), &myparms, context));
  nodeDependencies.clear();
  // Create executable graph
  hipStream_t streamForGraph;
  hipGraphExec_t graphExec;
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  // Verfication
  for (size_t i = 0; i < NumW; i++) {
    REQUIRE(A_h[i] == memSetVal);
  }
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
  HIP_CHECK(hipFree(A_d));

  HIP_CHECK(hipCtxPopCurrent(&context));
  HIP_CHECK(hipCtxDestroy(context));
}

/**
 * Test Description
 * ------------------------
 *    - Allocate a 1D array using hipMallocManaged. Initialize the allocated memory using
 * hipDrvGraphAddMemsetNode. Copy the values in device memory to host using
 * hipDrvGraphAddMemcpyNode. Verify the results.
 * Test source
 * ------------------------
 *    - unit/graph/hipDrvGraphAddMemsetNode.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipDrvGraphAddMemsetNode_hipMallocManaged") {
  HIP_CHECK(hipInit(0));
  hipDevice_t device;
  hipCtx_t context;
  HIP_CHECK(hipDeviceGet(&device, 0));
  HIP_CHECK(hipCtxCreate(&context, 0, device));

  int managed = 0;
  HIP_CHECK(hipDeviceGetAttribute(&managed, hipDeviceAttributeManagedMemory, 0));
  INFO("hipDeviceAttributeManagedMemory: " << managed);
  if (managed != 1) {
    WARN(
        "GPU 0 doesn't support hipDeviceAttributeManagedMemory attribute"
        "so defaulting to system memory.");
  }
  size_t Nbytes1D = SIZE * sizeof(char);
  char* A_d;
  // Initialize the host memory
  std::vector<char> A_h(SIZE, ' ');
  // Device Memory
  HIP_CHECK(hipMallocManaged(&A_d, SIZE * sizeof(char)));
  // Create the graph
  hipGraph_t graph;
  std::vector<hipGraphNode_t> nodeDependencies;
  hipGraphNode_t memsetNode, memcpyNode;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  // Add Memset node
  hipMemsetParams memsetParams{};
  memset(&memsetParams, 0, sizeof(memsetParams));
  memsetParams.dst = reinterpret_cast<void*>(A_d);
  memsetParams.value = memSetVal;
  memsetParams.pitch = Nbytes1D;
  memsetParams.elementSize = sizeof(char);
  memsetParams.width = SIZE;
  memsetParams.height = 1;
  HIP_CHECK(hipDrvGraphAddMemsetNode(&memsetNode, graph, nullptr, 0, &memsetParams, context));
  nodeDependencies.push_back(memsetNode);

  // Add MemCpy Node
  hipPitchedPtr devPitchedPtr{A_d, Nbytes1D, SIZE, 1};
  hipPitchedPtr hostPitchedPtr{A_h.data(), Nbytes1D, SIZE, 1};

  auto srcPos = make_hipPos(0, 0, 0);
  auto dstPos = make_hipPos(0, 0, 0);
  auto srcPtr = devPitchedPtr;
  auto dstPtr = hostPitchedPtr;
  auto extent = make_hipExtent(Nbytes1D, 1, 1);
  hipMemcpyKind kind = hipMemcpyDeviceToHost;

  HIP_MEMCPY3D myparms = GetDrvMemcpy3DParms(dstPtr, dstPos, srcPtr, srcPos, extent, kind);
  HIP_CHECK(hipDrvGraphAddMemcpyNode(&memcpyNode, graph, nodeDependencies.data(),
                                     nodeDependencies.size(), &myparms, context));
  nodeDependencies.clear();

  // Create executable graph
  hipStream_t streamForGraph;
  hipGraphExec_t graphExec;
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  // Verfication
  for (size_t i = 0; i < SIZE; i++) {
    REQUIRE(A_h[i] == memSetVal);
  }

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
  HIP_CHECK(hipFree(A_d));

  HIP_CHECK(hipCtxPopCurrent(&context));
  HIP_CHECK(hipCtxDestroy(context));
}

/**
 * End doxygen group GraphTest.
 * @}
 */
