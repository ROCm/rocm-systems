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
#include <hip_test_checkers.hh>
#include <memcpy3d_tests_common.hh>
#include <resource_guards.hh>
#include <utils.hh>

#include "graph_memset_node_test_common.hh"
#include "graph_tests_common.hh"

#pragma clang diagnostic ignored "-Wunused-parameter"

/**
 * @addtogroup hipGraphAddNode hipGraphAddNode
 * @{
 * @ingroup GraphTest
 * `hipGraphAddNode(hipGraphNode_t *pGraphNode, hipGraph_t graph, const hipGraphNode_t
 * *pDependencies, size_t numDependencies, hipGraphNodeParams *nodeParams)` - Creates a node and
 * adds it to a graph
 */

static constexpr size_t N = 1024;

static void callbackfunc(void* A_h) {
  int* A = reinterpret_cast<int*>(A_h);
  for (int i = 0; i < N; i++) {
    A[i] = i;
  }
}

static void __global__ vector_square(int* A_d) {
  for (int i = 0; i < N; i++) {
    A_d[i] = A_d[i] * A_d[i];
  }
}

/**
 * Test Description
 * ------------------------
 *    - Verify that all elements of destination memory are set to the correct value.
 * The test is repeated for all valid element sizes(1, 2, 4), and several allocations of different
 * height and width, both on host and device.
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphAddNode.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.0
 */
TEMPLATE_TEST_CASE("Unit_hipGraphAddNodeTypeMemset_Positive_Basic", "", uint8_t, uint16_t,
                   uint32_t) {
  const auto f = [](hipMemsetParams* params) {
    hipGraph_t graph = nullptr;
    HIP_CHECK(hipGraphCreate(&graph, 0));

    hipGraphNode_t node = nullptr;
    hipGraphNodeParams node_params = {};
    node_params.type = hipGraphNodeTypeMemset;
    node_params.memset.dst = params->dst;
    node_params.memset.elementSize = params->elementSize;
    node_params.memset.width = params->width;
    node_params.memset.height = params->height;
    node_params.memset.pitch = params->pitch;
    node_params.memset.value = params->value;
    HIP_CHECK(hipGraphAddNode(&node, graph, nullptr, 0, &node_params));

    hipGraphExec_t graph_exec = nullptr;
    HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

    HIP_CHECK(hipGraphLaunch(graph_exec, hipStreamPerThread));
    HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));

    HIP_CHECK(hipGraphExecDestroy(graph_exec));
    HIP_CHECK(hipGraphDestroy(graph));

    return hipSuccess;
  };

  GraphMemsetNodeCommonPositive<TestType, hipMemsetParams>(f);
}

/**
 * Test Description
 * ------------------------
 *    - Verify that kernel node added with hipGraphAddNode executes correctly and does the square of
 * values in the device array. The result is copied to host and verified.
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphAddNode.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipGraphAddNodeTypeKernel_Positive_Basic") {
  constexpr size_t allocation_size = N * sizeof(int);
  hipGraph_t graph;
  hipGraphExec_t graphExec;

  int* A_d{nullptr};
  int *A_h{nullptr}, *B_h{nullptr};
  HipTest::initArrays<int>(&A_d, nullptr, nullptr, &A_h, &B_h, nullptr, N, false);

  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphNode_t memcpyH2D_A, memcpyD2H_B;
  hipStream_t streamForGraph;
  HIP_CHECK(hipStreamCreate(&streamForGraph));

  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_A, graph, nullptr, 0, A_d, A_h, allocation_size,
                                    hipMemcpyHostToDevice));

  hipGraphNode_t node;
  hipGraphNodeParams node_params = {};
  node_params.type = hipGraphNodeTypeKernel;
  void* kernel_args[] = {&A_d};
  node_params.kernel.func = reinterpret_cast<void*>(vector_square);
  node_params.kernel.gridDim = dim3(1);
  node_params.kernel.blockDim = dim3(1);
  node_params.kernel.sharedMemBytes = 0;
  node_params.kernel.kernelParams = reinterpret_cast<void**>(kernel_args);
  node_params.kernel.extra = nullptr;
  HIP_CHECK(hipGraphAddNode(&node, graph, nullptr, 0, &node_params));


  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_B, graph, nullptr, 0, B_h, A_d, allocation_size,
                                    hipMemcpyDeviceToHost));

  HIP_CHECK(hipGraphAddDependencies(graph, &memcpyH2D_A, &node, 1));
  HIP_CHECK(hipGraphAddDependencies(graph, &node, &memcpyD2H_B, 1));

  // Instantiate and launch the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  // Verify execution result
  for (size_t i = 0; i < N; i++) {
    if (B_h[i] != (A_h[i] * A_h[i])) {
      REQUIRE(false);
    }
  }

  HipTest::freeArrays<int>(A_d, nullptr, nullptr, A_h, B_h, nullptr, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
}

/**
 * Test Description
 * ------------------------
 *    - Verify that host node added with hipGraphAddNode executes correctly and sets values of host
 * array. The result is verified.
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphAddNode.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipGraphAddNodeTypeHost_Positive_Basic") {
  constexpr size_t allocation_size = N * sizeof(int);
  hipGraph_t graph;
  hipGraphExec_t graphExec;
  int* A_h = (int*)malloc(allocation_size);
  std::fill_n(A_h, N, 0);

  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipStream_t streamForGraph;
  HIP_CHECK(hipStreamCreate(&streamForGraph));

  hipGraphNode_t node;
  hipGraphNodeParams node_params = {};
  node_params.type = hipGraphNodeTypeHost;
  node_params.host.fn = callbackfunc;
  node_params.host.userData = A_h;
  HIP_CHECK(hipGraphAddNode(&node, graph, nullptr, 0, &node_params));

  // Instantiate and launch the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  // Verify execution result
  for (size_t i = 0; i < N; i++) {
    if (A_h[i] != static_cast<int>(i)) {
      REQUIRE(false);
    }
  }

  free(A_h);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
}

/**
 * Test Description
 * ------------------------
 *    - Verify that when graph is created and childgraph node is added with hipGraphAddNode, the
 * childgraph executes correctly.
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphAddNode.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipGraphAddNodeTypeChildGraph_Positive_Basic") {
  constexpr size_t allocation_size = N * sizeof(int);
  hipGraph_t graph, childgraph;
  hipGraphExec_t graphExec;

  int *A_d{nullptr}, *B_d{nullptr}, *C_d{nullptr};
  int *A_h{nullptr}, *B_h{nullptr}, *C_h{nullptr};
  HipTest::initArrays<int>(&A_d, &B_d, &C_d, &A_h, &B_h, &C_h, N, false);

  HIP_CHECK(hipGraphCreate(&graph, 0));

  for (size_t i = 0; i < N; i++) {
    B_h[i] = i;
  }

  hipGraphNode_t memcpyH2D_A, memcpyH2D_B, childGraphNode1, memcpyH2D_C;
  hipStream_t streamForGraph;
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipGraphCreate(&childgraph, 0));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_B, childgraph, nullptr, 0, B_d, B_h, allocation_size,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_A, childgraph, nullptr, 0, A_h, B_d, allocation_size,
                                    hipMemcpyDeviceToHost));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_C, graph, nullptr, 0, C_d, C_h, allocation_size,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D_C, graph, nullptr, 0, A_h, C_d, allocation_size,
                                    hipMemcpyDeviceToHost));

  hipGraphNodeParams node_params = {};
  node_params.type = hipGraphNodeTypeGraph;
  node_params.graph.graph = childgraph;
  HIP_CHECK(hipGraphAddNode(&childGraphNode1, graph, nullptr, 0, &node_params));

  HIP_CHECK(hipGraphAddDependencies(childgraph, &memcpyH2D_B, &memcpyH2D_A, 1));

  // Instantiate and launch the childgraph
  HIP_CHECK(hipGraphInstantiate(&graphExec, childgraph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  // Verify execution result
  for (size_t i = 0; i < N; i++) {
    if (B_h[i] != A_h[i]) {
      REQUIRE(false);
    }
  }

  HipTest::freeArrays<int>(A_d, B_d, C_d, A_h, B_h, C_h, false);
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(childgraph));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
}


static hipError_t MemcpyType3DWrapper(PtrVariant dst_ptr, hipPos dst_pos, PtrVariant src_ptr,
                                      hipPos src_pos, hipExtent extent, hipMemcpyKind kind,
                                      hipStream_t stream = nullptr) {
  auto parms = GetMemcpy3DParms(dst_ptr, dst_pos, src_ptr, src_pos, extent, kind);

  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipGraphNode_t node = nullptr;

  hipGraphNodeParams node_params = {};
  node_params.type = hipGraphNodeTypeMemcpy;
  memset(&node_params.memcpy, 0, sizeof(hipMemcpyNodeParams));
  node_params.memcpy.copyParams = parms;
  HIP_CHECK(hipGraphAddNode(&node, graph, nullptr, 0, &node_params));

  hipGraphExec_t graph_exec = nullptr;
  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graph_exec, hipStreamPerThread));
  HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));

  HIP_CHECK(hipGraphExecDestroy(graph_exec));
  HIP_CHECK(hipGraphDestroy(graph));

  return hipSuccess;
}

/**
 * Test Description
 * ------------------------
 *    - Verify basic API behavior. A Memcpy node is created using hipGraphAddNode with parameters
 * set according to the test run, after which the graph is run and the memcpy results are verified.
 * The test is run for all possible memcpy directions, with both the corresponding memcpy
 * kind and hipMemcpyDefault, as well as half page and full page allocation sizes.
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphAddNode.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipGraphAddNodeTypeMemcpy_Positive_Basic") {
  CHECK_IMAGE_SUPPORT

  constexpr bool async = false;

  SECTION("Device to host") { Memcpy3DDeviceToHostShell<async>(MemcpyType3DWrapper); }

  SECTION("Device to host with default kind") {
    Memcpy3DDeviceToHostShell<async>(MemcpyType3DWrapper);
  }

  SECTION("Host to device") { Memcpy3DHostToDeviceShell<async>(MemcpyType3DWrapper); }

  SECTION("Host to device with default kind") {
    Memcpy3DHostToDeviceShell<async>(MemcpyType3DWrapper);
  }

  SECTION("Host to host") { Memcpy3DHostToHostShell<async>(MemcpyType3DWrapper); }

  SECTION("Host to host with default kind") { Memcpy3DHostToHostShell<async>(MemcpyType3DWrapper); }

  SECTION("Device to device") {
    SECTION("Peer access enabled") {
      Memcpy3DDeviceToDeviceShell<async, true>(MemcpyType3DWrapper);
    }
    SECTION("Peer access disabled") {
      Memcpy3DDeviceToDeviceShell<async, false>(MemcpyType3DWrapper);
    }
  }

  SECTION("Device to device with default kind") {
    SECTION("Peer access enabled") {
      Memcpy3DDeviceToDeviceShell<async, true>(MemcpyType3DWrapper);
    }
    SECTION("Peer access disabled") {
      Memcpy3DDeviceToDeviceShell<async, false>(MemcpyType3DWrapper);
    }
  }

  SECTION("Array from/to Host") { Memcpy3DArrayHostShell<async>(MemcpyType3DWrapper); }

#if HT_NVIDIA  // Disabled on AMD due to defect - EXSWHTEC-220
  SECTION("Array from/to Device") { Memcpy3DArrayDeviceShell<async>(MemcpyType3DWrapper); }
#endif
}


/**
 * Test Description
 * ------------------------
 *    - Verify basic API functionality where one event record node is added to graph with
 * hipGraphAddNode and its correct behavior is verified.
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphAddNode.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipGraphAddNodeTypeEventRecord_Positive_Basic") {
  hipGraph_t graph;
  hipStream_t streamForGraph;
  hipGraphExec_t graphExec;
  hipGraphNode_t node;
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipEvent_t event;
  HIP_CHECK(hipEventCreate(&event));

  hipGraphNodeParams node_params = {};
  node_params.type = hipGraphNodeTypeEventRecord;
  node_params.eventRecord.event = event;
  HIP_CHECK(hipGraphAddNode(&node, graph, nullptr, 0, &node_params));

  // Instantiate and launch the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  // Wait for event
  HIP_CHECK(hipEventSynchronize(event));
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipEventDestroy(event));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
}

/**
 * Test Description
 * ------------------------
 *    - Verify basic API functionality where one event record and one event wait nodes are added to
 * graph with hipGraphAddNode and their correct behavior is verified.
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphAddNode.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipGraphAddNodeTypeEventWait_Positive_Basic") {
  hipGraph_t graph;
  hipStream_t streamForGraph;
  hipGraphExec_t graphExec;
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipEvent_t event;
  HIP_CHECK(hipEventCreate(&event));
  hipGraphNode_t event_rec_node, event_wait_node;

  // Create a event record node in graph
  hipGraphNodeParams rec_node_params = {};
  rec_node_params.type = hipGraphNodeTypeEventRecord;
  rec_node_params.eventRecord.event = event;
  HIP_CHECK(hipGraphAddNode(&event_rec_node, graph, nullptr, 0, &rec_node_params));

  // Create a event wait node in graph
  hipGraphNodeParams wait_node_params = {};
  wait_node_params.type = hipGraphNodeTypeWaitEvent;
  wait_node_params.eventWait.event = event;
  HIP_CHECK(hipGraphAddNode(&event_wait_node, graph, &event_rec_node, 1, &wait_node_params));

  // Instantiate and launch the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));

  HIP_CHECK(hipStreamSynchronize(streamForGraph));
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipEventDestroy(event));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
}

/**
 * Test Description
 * ------------------------
 *  - Test to verify basic API functionality when memalloc and memfree nodes are added with
 * hipGraphAddNode. Verify that memory is allocated correctly and graph behaves as expected when
 * free node is added to the same graph.
 * Test source
 * ------------------------
 *  - /unit/graph/hipGraphAddNode.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipGraphAddNodeTypeMemAlloc_Positive_Basic") {
  constexpr size_t allocation_size = N * sizeof(int);
  hipGraph_t graph;
  hipStream_t streamForGraph;
  hipGraphExec_t graphExec;
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphNode_t alloc_node;
  hipGraphNodeParams alloc_node_params = {};
  alloc_node_params.type = hipGraphNodeTypeMemAlloc;
  memset(&alloc_node_params.alloc, 0, sizeof(hipMemAllocNodeParams));
  alloc_node_params.alloc.bytesize = allocation_size;
  alloc_node_params.alloc.poolProps.allocType = hipMemAllocationTypePinned;
  alloc_node_params.alloc.poolProps.location.id = 0;
  alloc_node_params.alloc.poolProps.location.type = hipMemLocationTypeDevice;
  HIP_CHECK(hipGraphAddNode(&alloc_node, graph, nullptr, 0, &alloc_node_params));

  REQUIRE(alloc_node_params.alloc.dptr != nullptr);
  int* A_d = reinterpret_cast<int*>(alloc_node_params.alloc.dptr);

  hipGraphNode_t free_node;
  hipGraphNodeParams free_node_params = {};
  free_node_params.type = hipGraphNodeTypeMemFree;
  free_node_params.free.dptr = A_d;
  HIP_CHECK(hipGraphAddNode(&free_node, graph, &alloc_node, 1, &free_node_params));

  // Instantiate and launch the graph
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, streamForGraph));
  HIP_CHECK(hipStreamSynchronize(streamForGraph));

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(streamForGraph));
  HIP_CHECK(hipDeviceGraphMemTrim(0));
}

/**
 * Test Description
 * ------------------------
 *  - Test to verify hipGraphAddNode behavior with invalid arguments:
 *    -# Nullptr graph
 *    -# Nullptr graph node
 *    -# Invalid numDependencies for null list of dependencies
 *    -# Node in dependency is from different graph
 *    -# Invalid numNodes
 *    -# Duplicate node in dependencies
 *    -# Nullptr params
 *    -# params type is invalid
 * Test source
 * ------------------------
 *  - /unit/graph/hipGraphAddNode.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipGraphAddNode_Negative_Parameters") {
  using namespace std::placeholders;
  hipGraph_t graph = nullptr;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipEvent_t event;
  HIP_CHECK(hipEventCreate(&event));

  hipGraphNode_t node;
  hipGraphNodeParams node_params = {};
  node_params.type = hipGraphNodeTypeEventRecord;
  node_params.eventRecord.event = event;

  GraphAddNodeCommonNegativeTests(std::bind(hipGraphAddNode, _1, _2, _3, _4, &node_params), graph);

  SECTION("params == nullptr") {
    HIP_CHECK_ERROR(hipGraphAddNode(&node, graph, nullptr, 0, nullptr), hipErrorInvalidValue);
  }

  SECTION("params type is invalid") {
    node_params.type = static_cast<hipGraphNodeType>(0x20);
    HIP_CHECK_ERROR(hipGraphAddNode(&node, graph, nullptr, 0, &node_params), hipErrorInvalidValue);
  }

  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipEventDestroy(event));
}

/**
 * End doxygen group GraphTest.
 * @}
 */
