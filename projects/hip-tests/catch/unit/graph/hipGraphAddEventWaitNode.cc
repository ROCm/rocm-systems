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
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
Testcase Scenarios :
 1) Simple Scenario: Create an event record node and then create an event
wait node using the same event and add it to graph. Instantiate and Launch
the Graph. Wait for the graph to complete. The operation must succeed without
any failures.
 2) Create a graph 1 with memcpyh2d, event record node (event A), kernel1
and memcpyd2h nodes. Create a graph 2 with Event Wait (event A) , kernel2
and memcpyd2h nodes. Instantiate and launch graph1 on stream1 and graph2 on
stream2. Wait for both graph1 and graph2 to complete. Validate the result of
both graphs.
 3) Execute graph1 and graph2 in scenario 2 multiple times in a loop
(100 times).
 4) Execute scenario 2 with stream1 = stream2.
 5) Repeat scenario 2 for different event flags.
 6) Validate that no error is reported when numDeps <= dependencies length
 7) Negative Scenarios
    - Pass input node parameter as nullptr.
    - Pass input graph parameter as nullptr.
    - Pass input dependency parameter as nullptr.
    - Node in dependency is from different graph
    - Invalid numNodes
    - Duplicate node in dependencies
    - Pass input event parameter as nullptr.
    - Pass uninitialized input graph parameter.
    - Pass uninitialized input event parameter.
*/
#include <functional>

#include <hip_test_checkers.hh>
#include <hip_test_common.hh>
#include <hip_test_kernels.hh>

#include "graph_tests_common.hh"

/**
 * Scenario 1
 */
TEST_CASE("Unit_hipGraphAddEventWaitNode_Functional_Simple") {
  hipGraph_t graph;
  hipStream_t streamForGraph;
  hipGraphExec_t graphExec;
  HIP_CHECK(hipStreamCreate(&streamForGraph));
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipEvent_t event;
  HIP_CHECK(hipEventCreate(&event));
  hipGraphNode_t event_rec_node, event_wait_node;
  // Create a event record node in graph
  HIP_CHECK(hipGraphAddEventRecordNode(&event_rec_node, graph, nullptr, 0, event));
  // Create a event wait node in graph
  HIP_CHECK(hipGraphAddEventWaitNode(&event_wait_node, graph, nullptr, 0, event));
  HIP_CHECK(hipGraphAddDependencies(graph, &event_rec_node, &event_wait_node, 1));
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
 * Local Function
 */
static void validate_hipGraphAddEventWaitNode_internodedep(int test, int nstep,
                                                           unsigned flag = hipEventDefault) {
  constexpr size_t N = 1024;
  size_t memsize = N * sizeof(int);
  constexpr auto blocksPerCU = 6;  // to hide latency
  constexpr auto threadsPerBlock = 256;
  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, N);
  size_t NElem{N};
  hipGraph_t graph1, graph2;
  hipStream_t streamForGraph1, streamForGraph2;
  hipGraphExec_t graphExec1, graphExec2;
  HIP_CHECK(hipStreamCreate(&streamForGraph1));
  HIP_CHECK(hipGraphCreate(&graph1, 0));
  HIP_CHECK(hipGraphCreate(&graph2, 0));
  if (0 == test) {
    HIP_CHECK(hipStreamCreate(&streamForGraph2));
  } else if (1 == test) {
    streamForGraph2 = streamForGraph1;
  }
  hipEvent_t event1;
  HIP_CHECK(hipEventCreateWithFlags(&event1, flag));
  hipGraphNode_t event_rec_node, event_wait_node;
  int *inp_h, *inp_d, *out_h_g1, *out_d_g1, *out_h_g2, *out_d_g2;
  // Allocate host buffers
  inp_h = reinterpret_cast<int*>(malloc(memsize));
  REQUIRE(inp_h != nullptr);
  out_h_g1 = reinterpret_cast<int*>(malloc(memsize));
  REQUIRE(out_h_g1 != nullptr);
  out_h_g2 = reinterpret_cast<int*>(malloc(memsize));
  REQUIRE(out_h_g2 != nullptr);
  // Allocate device buffers
  HIP_CHECK(hipMalloc(&inp_d, memsize));
  HIP_CHECK(hipMalloc(&out_d_g1, memsize));
  HIP_CHECK(hipMalloc(&out_d_g2, memsize));
  // Initialize host buffer
  for (uint32_t i = 0; i < N; i++) {
    inp_h[i] = i;
    out_h_g1[i] = 0;
    out_h_g2[i] = 0;
  }
  // Graph1 creation ...........
  // Create event1 record node in graph1
  HIP_CHECK(hipGraphAddEventRecordNode(&event_rec_node, graph1, nullptr, 0, event1));

  // Create memcpy and kernel nodes for graph1
  hipGraphNode_t memcpyH2D, memcpyD2H_1, kernelnode_1;
  hipKernelNodeParams kernelNodeParams1{};
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyH2D, graph1, nullptr, 0, inp_d, inp_h, memsize,
                                    hipMemcpyHostToDevice));
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_1, graph1, nullptr, 0, out_h_g1, out_d_g1, memsize,
                                    hipMemcpyDeviceToHost));

  void* kernelArgs1[] = {&inp_d, &out_d_g1, reinterpret_cast<void*>(&NElem)};
  kernelNodeParams1.func = reinterpret_cast<void*>(HipTest::vector_square<int>);
  kernelNodeParams1.gridDim = dim3(blocks);
  kernelNodeParams1.blockDim = dim3(threadsPerBlock);
  kernelNodeParams1.sharedMemBytes = 0;
  kernelNodeParams1.kernelParams = reinterpret_cast<void**>(kernelArgs1);
  kernelNodeParams1.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kernelnode_1, graph1, nullptr, 0, &kernelNodeParams1));
  // Create dependencies for graph1
  HIP_CHECK(hipGraphAddDependencies(graph1, &memcpyH2D, &event_rec_node, 1));
  HIP_CHECK(hipGraphAddDependencies(graph1, &event_rec_node, &kernelnode_1, 1));
  HIP_CHECK(hipGraphAddDependencies(graph1, &kernelnode_1, &memcpyD2H_1, 1));

  // Graph2 creation ...........
  // Create event1 record node in graph2
  HIP_CHECK(hipGraphAddEventWaitNode(&event_wait_node, graph2, nullptr, 0, event1));

  // Create memcpy and kernel nodes for graph2
  hipGraphNode_t memcpyD2H_2, kernelnode_2;
  hipKernelNodeParams kernelNodeParams2{};
  HIP_CHECK(hipGraphAddMemcpyNode1D(&memcpyD2H_2, graph2, nullptr, 0, out_h_g2, out_d_g2, memsize,
                                    hipMemcpyDeviceToHost));

  void* kernelArgs2[] = {&inp_d, &out_d_g2, reinterpret_cast<void*>(&NElem)};
  kernelNodeParams2.func = reinterpret_cast<void*>(HipTest::vector_cubic<int>);
  kernelNodeParams2.gridDim = dim3(blocks);
  kernelNodeParams2.blockDim = dim3(threadsPerBlock);
  kernelNodeParams2.sharedMemBytes = 0;
  kernelNodeParams2.kernelParams = reinterpret_cast<void**>(kernelArgs2);
  kernelNodeParams2.extra = nullptr;
  HIP_CHECK(hipGraphAddKernelNode(&kernelnode_2, graph2, nullptr, 0, &kernelNodeParams2));
  // Create dependencies for graph2
  HIP_CHECK(hipGraphAddDependencies(graph2, &event_wait_node, &kernelnode_2, 1));
  HIP_CHECK(hipGraphAddDependencies(graph2, &kernelnode_2, &memcpyD2H_2, 1));

  // Instantiate and launch the graphs
  HIP_CHECK(hipGraphInstantiate(&graphExec1, graph1, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphInstantiate(&graphExec2, graph2, nullptr, nullptr, 0));
  for (int istep = 0; istep < nstep; istep++) {
    HIP_CHECK(hipGraphLaunch(graphExec1, streamForGraph1));
    HIP_CHECK(hipGraphLaunch(graphExec2, streamForGraph2));
    HIP_CHECK(hipStreamSynchronize(streamForGraph1));
    HIP_CHECK(hipStreamSynchronize(streamForGraph2));
    // Validate output
    bool btestPassed1 = true;
    for (uint32_t i = 0; i < N; i++) {
      if (out_h_g1[i] != (inp_h[i] * inp_h[i])) {
        btestPassed1 = false;
        break;
      }
    }
    REQUIRE(btestPassed1 == true);
    bool btestPassed2 = true;
    for (uint32_t i = 0; i < N; i++) {
      if (out_h_g2[i] != (inp_h[i] * inp_h[i] * inp_h[i])) {
        btestPassed2 = false;
        break;
      }
    }
    REQUIRE(btestPassed2 == true);
  }
  // Destroy all resources
  HIP_CHECK(hipFree(inp_d));
  HIP_CHECK(hipFree(out_d_g1));
  HIP_CHECK(hipFree(out_d_g2));
  free(inp_h);
  free(out_h_g1);
  free(out_h_g2);
  HIP_CHECK(hipGraphExecDestroy(graphExec1));
  HIP_CHECK(hipGraphExecDestroy(graphExec2));
  HIP_CHECK(hipGraphDestroy(graph1));
  HIP_CHECK(hipGraphDestroy(graph2));
  HIP_CHECK(hipEventDestroy(event1));
  HIP_CHECK(hipStreamDestroy(streamForGraph1));
  if (0 == test) {
    HIP_CHECK(hipStreamDestroy(streamForGraph2));
  }
}

/**
 * Scenario 2
 */
TEST_CASE("Unit_hipGraphAddEventWaitNode_MultGraphMultStrmDependency") {
  validate_hipGraphAddEventWaitNode_internodedep(0, 1);
}

/**
 * Scenario 3
 */
TEST_CASE("Unit_hipGraphAddEventWaitNode_MultipleRun") {
  validate_hipGraphAddEventWaitNode_internodedep(0, 100);
}

/**
 * Scenario 4
 */
TEST_CASE("Unit_hipGraphAddEventWaitNode_MultGraphOneStrmDependency") {
  validate_hipGraphAddEventWaitNode_internodedep(1, 1);
}

/**
 * Scenario 5
 */
TEST_CASE("Unit_hipGraphAddEventWaitNode_differentFlags") {
  SECTION("flag = hipEventBlockingSync") {
    validate_hipGraphAddEventWaitNode_internodedep(0, 1, hipEventBlockingSync);
  }
  SECTION("graph = hipEventDisableTiming") {
    validate_hipGraphAddEventWaitNode_internodedep(0, 1, hipEventDisableTiming);
  }
}

/**
 * Scenario 6: Positive parameter tests
 */
TEST_CASE("Unit_hipGraphAddEventWaitNode_Positive_Parameters") {
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipEvent_t event;
  HIP_CHECK(hipEventCreate(&event));
  hipGraphNode_t eventwait;

  hipGraphNode_t dep_node = nullptr;
  hipGraphNode_t dep_node2 = nullptr;
  HIP_CHECK(hipGraphAddEmptyNode(&dep_node, graph, nullptr, 0));
  HIP_CHECK(hipGraphAddEmptyNode(&dep_node2, graph, nullptr, 0));
  hipGraphNode_t dep_nodes[] = {dep_node, dep_node2};

  size_t numDeps = 0;
  SECTION("numDependencies is zero, dependencies is not nullptr") {
    HIP_CHECK(hipGraphAddEventWaitNode(&eventwait, graph, dep_nodes, 0, event));
    HIP_CHECK(hipGraphNodeGetDependencies(eventwait, nullptr, &numDeps));
    REQUIRE(numDeps == 0);
  }

  SECTION("numDependencies < dependencies length") {
    HIP_CHECK(hipGraphAddEventWaitNode(&eventwait, graph, dep_nodes, 1, event));
    HIP_CHECK(hipGraphNodeGetDependencies(eventwait, nullptr, &numDeps));
    REQUIRE(numDeps == 1);
  }

  SECTION("numDependencies == dependencies length") {
    HIP_CHECK(hipGraphAddEventWaitNode(&eventwait, graph, dep_nodes, 2, event));
    HIP_CHECK(hipGraphNodeGetDependencies(eventwait, nullptr, &numDeps));
    REQUIRE(numDeps == 2);
  }

  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipEventDestroy(event));
}

/**
 * Scenario 7
 */
TEST_CASE("Unit_hipGraphAddEventWaitNode_Negative") {
  using namespace std::placeholders;
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipEvent_t event;
  HIP_CHECK(hipEventCreate(&event));
  hipGraphNode_t eventwait;

  GraphAddNodeCommonNegativeTests(std::bind(hipGraphAddEventWaitNode, _1, _2, _3, _4, event),
                                  graph);

  SECTION("event = nullptr") {
    HIP_CHECK_ERROR(hipGraphAddEventWaitNode(&eventwait, graph, nullptr, 0, nullptr),
                    hipErrorInvalidValue);
  }

  SECTION("graph is uninitialized") {
    hipGraph_t graph_uninit{};
    HIP_CHECK_ERROR(hipGraphAddEventWaitNode(&eventwait, graph_uninit, nullptr, 0, event),
                    hipErrorInvalidValue);
  }

  SECTION("event is uninitialized") {
    hipEvent_t event_uninit{};
    HIP_CHECK_ERROR(hipGraphAddEventWaitNode(&eventwait, graph, nullptr, 0, event_uninit),
                    hipErrorInvalidValue);
  }

  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipEventDestroy(event));
}
