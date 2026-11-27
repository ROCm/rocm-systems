/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

#include <algorithm>
#include <vector>

/**
 * @addtogroup hipDeviceGraphMemTrim hipDeviceGraphMemTrim
 * @{
 * @ingroup GraphTest
 * `hipDeviceGraphMemTrim(int device)` - Free unused memory on specific device used for graph back
 * to OS.
 */

/**
 * Test Description
 * ------------------------
 *  - Basic test to verify that unused memory used for graph can be freed on each device.
 * Test source
 * ------------------------
 *  - /unit/graph/hipDeviceGraphMemTrim.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
HIP_TEST_CASE(Unit_hipDeviceGraphMemTrim_Positive_Default) {
  const auto device = GENERATE(range(0, HipTest::getDeviceCount()));

  // Check for each device
  HIP_CHECK(hipDeviceGraphMemTrim(device));
}

/**
 * Test Description
 * ------------------------
 *  - Test to verify hipDeviceGraphMemTrim behavior with invalid arguments:
 *    -# Device is not valid
 * Test source
 * ------------------------
 *  - /unit/graph/hipDeviceGraphMemTrim.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
HIP_TEST_CASE(Unit_hipDeviceGraphMemTrim_Negative_Parameters) {
  int device_id = 0;
  HIP_CHECK(hipSetDevice(device_id));

  int num_dev = 0;
  HIP_CHECK(hipGetDeviceCount(&num_dev));

  SECTION("Device is not valid") {
    HIP_CHECK_ERROR(hipDeviceGraphMemTrim(num_dev), hipErrorInvalidDevice);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Destroys graphs that contain child graph nodes, then trims graph memory and runs another
 *    graph with a child node. hipDeviceGraphMemTrim walks every live graph exec; child graph
 *    nodes are graph execs too, so a destroyed child node must not be read:
 *    -# Graphs destroyed without being instantiated (hipGraphDestroy deletes the child node).
 *    -# Graph execs launched and destroyed right after the stream is synchronized: the launch
 *       can still hold the exec, which deletes its child nodes when the launch completes,
 *       possibly while trim runs.
 *  - Run under a memory checker to see such a use: without one the stale read usually goes
 *    unnoticed.
 * Test source
 * ------------------------
 *  - /unit/graph/hipDeviceGraphMemTrim.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.14
 */
HIP_TEST_CASE(Unit_hipDeviceGraphMemTrim_Positive_AfterChildGraphDestroy) {
  constexpr size_t kSize = 1024;
  constexpr int kIterations = 16;
  int device = 0;
  HIP_CHECK(hipGetDevice(&device));
  int* dev_mem = nullptr;
  HIP_CHECK(hipMalloc(&dev_mem, kSize));
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  auto build_graph = [&](int value, hipGraph_t* graph) {
    hipGraph_t child_graph;
    HIP_CHECK(hipGraphCreate(&child_graph, 0));
    hipMemsetParams params{};
    params.dst = dev_mem;
    params.value = value;
    params.elementSize = sizeof(int);
    params.width = kSize / sizeof(int);
    params.height = 1;
    hipGraphNode_t memset_node;
    HIP_CHECK(hipGraphAddMemsetNode(&memset_node, child_graph, nullptr, 0, &params));
    HIP_CHECK(hipGraphCreate(graph, 0));
    hipGraphNode_t child_node;
    HIP_CHECK(hipGraphAddChildGraphNode(&child_node, *graph, nullptr, 0, child_graph));
    // The child node holds its own copy of the child graph
    HIP_CHECK(hipGraphDestroy(child_graph));
  };

  SECTION("Destroyed graphs") {
    for (int i = 0; i < kIterations; ++i) {
      hipGraph_t graph;
      build_graph(i, &graph);
      HIP_CHECK(hipGraphDestroy(graph));
    }
  }
  SECTION("Destroyed graph execs") {
    for (int i = 0; i < kIterations; ++i) {
      hipGraph_t graph;
      build_graph(i, &graph);
      hipGraphExec_t graph_exec;
      HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
      HIP_CHECK(hipGraphLaunch(graph_exec, stream));
      HIP_CHECK(hipStreamSynchronize(stream));
      HIP_CHECK(hipGraphExecDestroy(graph_exec));
      HIP_CHECK(hipGraphDestroy(graph));
    }
  }

  HIP_CHECK(hipDeviceGraphMemTrim(device));

  hipGraph_t graph;
  build_graph(kIterations, &graph);
  hipGraphExec_t graph_exec;
  HIP_CHECK(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graph_exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipDeviceGraphMemTrim(device));

  std::vector<int> host(kSize / sizeof(int));
  HIP_CHECK(hipMemcpy(host.data(), dev_mem, kSize, hipMemcpyDeviceToHost));
  REQUIRE(std::all_of(host.begin(), host.end(), [](int v) { return v == kIterations; }));

  HIP_CHECK(hipGraphExecDestroy(graph_exec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(dev_mem));
}

/**
 * End doxygen group GraphTest.
 * @}
 */
