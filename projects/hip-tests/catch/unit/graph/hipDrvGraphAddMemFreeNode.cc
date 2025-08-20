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
#include <resource_guards.hh>

/**
 * @addtogroup hipDrvGraphAddMemFreeNode hipDrvGraphAddMemFreeNode
 * @{
 * @ingroup GraphTest
 * `hipDrvGraphAddMemFreeNode (	hipGraphNode_t * phGraphNode,
 *  hipGraph_t hGraph, const hipGraphNode_t * dependencies,
 *  size_t numDependencies, hipDeviceptr_t dptr)'-
 * Creates a memory free node and adds it to a graph.
 */
/**
 * Test Description
 * ------------------------
 *  - Test to verify hipDrvGraphAddMemFreeNode behavior with invalid arguments:
 *    -# Null graph node
 *    -# Null graph
 *    -# Invalid numDependencies for null list of dependencies
 *    -# Invalid numDependencies and valid list for dependencies
 *    -# Null dev_ptr
 * Test source
 * ------------------------
 *  - /unit/graph/hipDrvGraphAddMemFreeNode.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.4
 */
TEST_CASE("Unit_hipDrvGraphAddMemFreeNode_Negative_Params") {
  constexpr size_t N = 1024;
  hipGraph_t graph;
  hipGraphNode_t alloc_node, free_node;
  std::vector<hipGraphNode_t> dependencies;

  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipMemAllocNodeParams alloc_param;
  memset(&alloc_param, 0, sizeof(alloc_param));
  alloc_param.bytesize = N;
  alloc_param.poolProps.allocType = hipMemAllocationTypePinned;
  alloc_param.poolProps.location.id = 0;
  alloc_param.poolProps.location.type = hipMemLocationTypeDevice;

  HIP_CHECK(hipGraphAddMemAllocNode(&alloc_node, graph, nullptr, 0, &alloc_param));
  REQUIRE(alloc_param.dptr != nullptr);

  SECTION("Passing nullptr to graph node") {
    HIP_CHECK_ERROR(
        hipDrvGraphAddMemFreeNode(nullptr, graph, &alloc_node, 1, (hipDeviceptr_t)alloc_param.dptr),
        hipErrorInvalidValue);
  }

  SECTION("Passing nullptr to graph") {
    HIP_CHECK_ERROR(hipDrvGraphAddMemFreeNode(&free_node, nullptr, &alloc_node, 1,
                                              (hipDeviceptr_t)alloc_param.dptr),
                    hipErrorInvalidValue);
  }

  SECTION("Pass invalid numDependencies") {
    HIP_CHECK_ERROR(
        hipDrvGraphAddMemFreeNode(&free_node, graph, nullptr, 5, (hipDeviceptr_t)alloc_param.dptr),
        hipErrorInvalidValue);
  }

  SECTION("Pass invalid numDependencies and valid list for dependencies") {
    dependencies.push_back(alloc_node);
    HIP_CHECK_ERROR(
        hipDrvGraphAddMemFreeNode(&free_node, graph, dependencies.data(), dependencies.size() + 1,
                                  (hipDeviceptr_t)alloc_param.dptr),
        hipErrorInvalidValue);
  }

  SECTION("Passing nullptr to dev_ptr") {
    HIP_CHECK_ERROR(hipDrvGraphAddMemFreeNode(&alloc_node, graph, &alloc_node, 1, 0),
                    hipErrorInvalidValue);
  }

  HIP_CHECK(hipGraphDestroy(graph));
}
/**
 * Test Description
 * ------------------------
 *  - It will create memory alloation node and add to the graph then it
 *  will create memory free node to free allocated memory and add to the graph.
 * Test source
 * ------------------------
 *  - /unit/graph/hipDrvGraphAddMemFreeNode.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.4
 */
TEST_CASE("Unit_hipDrvGraphAddMemFreeNode_Positive") {
  constexpr size_t N = 1024;
  hipGraph_t graph;
  hipGraphExec_t graphExec;
  hipCtx_t context;
  hipStream_t streamForGraph;
  int deviceid = 0;
  hipGraphNode_t node = nullptr, memFreeNode = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipSetDevice(deviceid));
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipCtxCreate(&context, 0, deviceid));

  hipMemAllocNodeParams alloc_param;
  memset(&alloc_param, 0, sizeof(alloc_param));
  alloc_param.bytesize = N;
  alloc_param.poolProps.allocType = hipMemAllocationTypePinned;
  alloc_param.poolProps.location.id = 0;
  alloc_param.poolProps.location.type = hipMemLocationTypeDevice;
  HIP_CHECK(hipGraphAddMemAllocNode(&node, graph, nullptr, 0, &alloc_param));
  REQUIRE(alloc_param.dptr != nullptr);

  HIP_CHECK(
      hipDrvGraphAddMemFreeNode(&memFreeNode, graph, &node, 1, (hipDeviceptr_t)alloc_param.dptr));
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, 0));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
  HIP_CHECK(hipCtxDestroy(context));
}
/**
 * End doxygen group GraphTest.
 * @}
 */
