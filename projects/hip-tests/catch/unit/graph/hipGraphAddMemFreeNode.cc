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
 * @addtogroup hipGraphAddMemFreeNode hipGraphAddMemFreeNode
 * @{
 * @ingroup GraphTest
 * `hipGraphAddMemFreeNode (hipGraphNode_t *pGraphNode, hipGraph_t graph, const hipGraphNode_t
 * *pDependencies, size_t numDependencies, void *dev_ptr)` -
 * Creates a memory free node and adds it to a graph.
 */


/**
 * Test Description
 * ------------------------
 *  - Test to verify hipGraphAddMemFreeNode behavior with invalid arguments:
 *    -# Null graph node
 *    -# Null graph
 *    -# Invalid numDependencies for null list of dependencies
 *    -# Invalid numDependencies and valid list for dependencies
 *    -# Null dev_ptr
 *    -# Invalid dev_ptr address
 *    -# dev_ptr not allocated with alloc node
 *    -# Allocation is freed twice in the same graph
 * Test source
 * ------------------------
 *  - /unit/graph/hipGraphAddMemFreeNode.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipGraphAddMemFreeNode_Negative_Params") {
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
  int* A_d = reinterpret_cast<int*>(alloc_param.dptr);

  SECTION("Passing nullptr to graph node") {
    HIP_CHECK_ERROR(hipGraphAddMemFreeNode(nullptr, graph, &alloc_node, 1, (void*)A_d),
                    hipErrorInvalidValue);
  }

  SECTION("Passing nullptr to graph") {
    HIP_CHECK_ERROR(hipGraphAddMemFreeNode(&free_node, nullptr, &alloc_node, 1, (void*)A_d),
                    hipErrorInvalidValue);
  }

  SECTION("Pass invalid numDependencies") {
    HIP_CHECK_ERROR(hipGraphAddMemFreeNode(&free_node, graph, nullptr, 5, (void*)A_d),
                    hipErrorInvalidValue);
  }

  SECTION("Pass invalid numDependencies and valid list for dependencies") {
    dependencies.push_back(alloc_node);
    HIP_CHECK_ERROR(hipGraphAddMemFreeNode(&free_node, graph, dependencies.data(),
                                           dependencies.size() + 1, (void*)A_d),
                    hipErrorInvalidValue);
  }

  SECTION("Passing nullptr to dev_ptr") {
    HIP_CHECK_ERROR(hipGraphAddMemFreeNode(&alloc_node, graph, &alloc_node, 1, nullptr),
                    hipErrorInvalidValue);
  }

  SECTION("Passing invalid address to dev_ptr") {
    int value;
    HIP_CHECK_ERROR(hipGraphAddMemFreeNode(&alloc_node, graph, &alloc_node, 1, &value),
                    hipErrorInvalidValue);
  }

#if HT_NVIDIA  // EXSWHTEC-352
  SECTION("Passing address not allocated with alloc node to dev_ptr") {
    LinearAllocGuard<int> dev_alloc =
        LinearAllocGuard<int>(LinearAllocs::hipMalloc, N * sizeof(int));
    HIP_CHECK_ERROR(hipGraphAddMemFreeNode(&alloc_node, graph, &alloc_node, 1, dev_alloc.ptr()),
                    hipErrorInvalidValue);
  }

  SECTION("Free allocation twice in the same graph") {
    HIP_CHECK(hipGraphAddMemFreeNode(&alloc_node, graph, &alloc_node, 1, (void*)A_d));
    HIP_CHECK_ERROR(hipGraphAddMemFreeNode(&alloc_node, graph, &alloc_node, 1, (void*)A_d),
                    hipErrorInvalidValue);
  }
#endif

  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *  - Test to verify hipGraphAddMemFreeNode unsupported behavior:
 *    -# More than one instantiation of the graph exist at the same time
 *    -# Clone graph with mem free node
 *    -# Use graph with mem free node in a child node
 *    -# Delete edge of the graph with mem free node
 * Test source
 * ------------------------
 *  - /unit/graph/hipGraphAddMemFreeNode.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipGraphAddMemFreeNode_Negative_NotSupported") {
  constexpr size_t N = 1024;
  hipGraph_t graph1, graph2;
  hipGraphNode_t alloc_node, free_node;

  HIP_CHECK(hipGraphCreate(&graph1, 0));
  HIP_CHECK(hipGraphCreate(&graph2, 0));

  hipMemAllocNodeParams alloc_param;
  memset(&alloc_param, 0, sizeof(alloc_param));
  alloc_param.bytesize = N;
  alloc_param.poolProps.allocType = hipMemAllocationTypePinned;
  alloc_param.poolProps.location.id = 0;
  alloc_param.poolProps.location.type = hipMemLocationTypeDevice;

  HIP_CHECK(hipGraphAddMemAllocNode(&alloc_node, graph1, nullptr, 0, &alloc_param));
  REQUIRE(alloc_param.dptr != nullptr);
  int* A_d = reinterpret_cast<int*>(alloc_param.dptr);

  HIP_CHECK(hipGraphAddMemFreeNode(&free_node, graph2, nullptr, 0, (void*)A_d));

  SECTION("More than one instantation of the graph exists") {
    hipGraphExec_t graph_exec1, graph_exec2;
    HIP_CHECK(hipGraphInstantiate(&graph_exec1, graph2, nullptr, nullptr, 0));
    HIP_CHECK_ERROR(hipGraphInstantiate(&graph_exec2, graph2, nullptr, nullptr, 0),
                    hipErrorNotSupported);
    HIP_CHECK(hipGraphExecDestroy(graph_exec1));
  }

#if HT_NVIDIA  // EXSWHTEC-352
  SECTION("Clone graph with mem free node") {
    hipGraph_t cloned_graph;
    HIP_CHECK_ERROR(hipGraphClone(&cloned_graph, graph2), hipErrorNotSupported);
  }

  SECTION("Use graph in a child node") {
    hipGraph_t parent_graph;
    HIP_CHECK(hipGraphCreate(&parent_graph, 0));
    hipGraphNode_t child_graph_node;
    HIP_CHECK_ERROR(hipGraphAddChildGraphNode(&child_graph_node, parent_graph, nullptr, 0, graph2),
                    hipErrorNotSupported);
    HIP_CHECK(hipGraphDestroy(parent_graph));
  }

  SECTION("Delete edge of the graph") {
    hipGraphNode_t empty_node;
    HIP_CHECK(hipGraphAddEmptyNode(&empty_node, graph2, &free_node, 1));
    HIP_CHECK_ERROR(hipGraphRemoveDependencies(graph2, &free_node, &empty_node, 1),
                    hipErrorNotSupported);
  }
#endif

  HIP_CHECK(hipGraphDestroy(graph1));
  HIP_CHECK(hipGraphDestroy(graph2));
}


/**
 * Test Description
 * ------------------------
 * - Functional Test for API hipGraphAddMemFreeNode -
 * Measure memory footprint before creating graph.
 * Create a graph and add a node with hipGraphAddMemAllocNode and
 * hipGraphAddMemFreeNode and launch it.
 * Measure memory footprint after the launch and destroy of the graph.
 * Both before and after memory should be same after graph execution.
 * Test source
 * ------------------------
 * - /unit/graph/hipGraphAddMemFreeNode.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 6.1
 */
TEST_CASE("Unit_hipGraphAddMemFreeNode_Functional") {
  int mem_pool_support = 0;
  HIP_CHECK(hipDeviceGetAttribute(&mem_pool_support, hipDeviceAttributeMemoryPoolsSupported, 0));
  if (!mem_pool_support) {
    HipTest::HIP_SKIP_TEST(
        "Runtime doesn't support Memory Pool."
        " Skip the test case.");
    return;
  }

  constexpr size_t Nbytes = 512 * 1024 * 1024;
  hipGraph_t graph;
  hipGraphExec_t graphExec;
  hipStream_t stream;
  hipGraphNode_t allocNodeA, freeNodeA;
  hipMemAllocNodeParams allocParam;

  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipStreamCreate(&stream));

  memset(&allocParam, 0, sizeof(allocParam));
  allocParam.bytesize = Nbytes;
  allocParam.poolProps.allocType = hipMemAllocationTypePinned;
  allocParam.poolProps.location.id = 0;
  allocParam.poolProps.location.type = hipMemLocationTypeDevice;

  HIP_CHECK(hipGraphAddMemAllocNode(&allocNodeA, graph, NULL, 0, &allocParam));
  REQUIRE(allocParam.dptr != nullptr);
  HIP_CHECK(hipGraphAddMemFreeNode(&freeNodeA, graph, &allocNodeA, 1,
                                   reinterpret_cast<void*>(allocParam.dptr)));

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  size_t before = 0, after = 0;
  HIP_CHECK(hipDeviceGraphMemTrim(0));
  HIP_CHECK(hipDeviceGetGraphMemAttribute(0, hipGraphMemAttrUsedMemCurrent, &before));
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipDeviceGraphMemTrim(0));
  HIP_CHECK(hipDeviceGetGraphMemAttribute(0, hipGraphMemAttrUsedMemCurrent, &after));

  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipStreamDestroy(stream));

  REQUIRE(before == after);
}
/**
 * End doxygen group GraphTest.
 * @}
 */
