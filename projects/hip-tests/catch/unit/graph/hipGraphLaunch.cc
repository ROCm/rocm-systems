/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>

/**
 * @addtogroup hipGraphLaunch hipGraphLaunch
 * @{
 * @ingroup GraphTest
 * `hipGraphLaunch(hipGraphExec_t graphExec, hipStream_t stream)` -
 * Launches an executable graph in a stream
 */

static void HostFunctionSetToZero(void* arg) {
  int* test_number = (int*)arg;
  (*test_number) = 0;
}

static void HostFunctionAddOne(void* arg) {
  int* test_number = (int*)arg;
  (*test_number) += 1;
}

/* create an executable graph that will set an integer pointed to by 'number' to one*/
static void CreateTestExecutableGraph(hipGraphExec_t* graph_exec, int* number) {
  hipGraph_t graph;
  hipGraphNode_t node_error;

  hipGraphNode_t node_set_zero;
  hipHostNodeParams params_set_to_zero = {HostFunctionSetToZero, number};

  hipGraphNode_t node_add_one;
  hipHostNodeParams params_set_add_one = {HostFunctionAddOne, number};

  HIP_CHECK(hipGraphCreate(&graph, 0));

  HIP_CHECK(hipGraphAddHostNode(&node_set_zero, graph, nullptr, 0, &params_set_to_zero));
  HIP_CHECK(hipGraphAddHostNode(&node_add_one, graph, &node_set_zero, 1, &params_set_add_one));

  HIP_CHECK(hipGraphInstantiate(graph_exec, graph, &node_error, nullptr, 0));
  HIP_CHECK(hipGraphDestroy(graph));
}

static void HipGraphLaunch_Positive_Simple(hipStream_t stream) {
  int number = 5;

  hipGraphExec_t graph_exec;
  CreateTestExecutableGraph(&graph_exec, &number);

  HIP_CHECK(hipGraphLaunch(graph_exec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));
  REQUIRE(number == 1);

  HIP_CHECK(hipGraphExecDestroy(graph_exec));
}


/**
 * Test Description
 * ------------------------
 *    - Basic positive test for hipGraphLaunch
 *        -# stream as a created stream
 *        -# with stream as hipStreamPerThread
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphLaunch.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE(Unit_hipGraphLaunch_Positive) {
  SECTION("stream as a created stream") {
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    HipGraphLaunch_Positive_Simple(stream);
    HIP_CHECK(hipStreamDestroy(stream));
  }

  SECTION("with stream as hipStreamPerThread") {
    HipGraphLaunch_Positive_Simple(hipStreamPerThread);
  }
}

/**
 * Test Description
 * ------------------------
 *    - Negative parameter test for hipGraphLaunch
 *        -# graphExec is nullptr and stream is a created stream
 *        -# graphExec is nullptr and stream is hipStreamPerThread
 *        -# graphExec is an empty object
 *        -# graphExec is destroyed before calling hipGraphLaunch
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphLaunch.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEST_CASE(Unit_hipGraphLaunch_Negative_Parameters) {
  SECTION("graphExec is nullptr and stream is a created stream") {
    hipStream_t stream;
    hipError_t ret;
    HIP_CHECK(hipStreamCreate(&stream));
    ret = hipGraphLaunch(nullptr, stream);
    HIP_CHECK(hipStreamDestroy(stream));
    REQUIRE(ret == hipErrorInvalidValue);
  }

  SECTION("graphExec is nullptr and stream is hipStreamPerThread") {
    HIP_CHECK_ERROR(hipGraphLaunch(nullptr, hipStreamPerThread), hipErrorInvalidValue);
  }

  SECTION("graphExec is an empty object") {
    hipGraphExec_t graph_exec{};
    HIP_CHECK_ERROR(hipGraphLaunch(graph_exec, hipStreamPerThread), hipErrorInvalidValue);
  }

  SECTION("graphExec is destroyed") {
    int number = 5;
    hipGraphExec_t graph_exec;
    CreateTestExecutableGraph(&graph_exec, &number);
    HIP_CHECK(hipGraphLaunch(graph_exec, hipStreamPerThread));
    HIP_CHECK(hipStreamSynchronize(hipStreamPerThread));
    REQUIRE(number == 1);
    HIP_CHECK(hipGraphExecDestroy(graph_exec));
    HIP_CHECK_ERROR(hipGraphLaunch(graph_exec, hipStreamPerThread), hipErrorInvalidValue);
  }
}

/**
 * End doxygen group GraphTest.
 * @}
 */
