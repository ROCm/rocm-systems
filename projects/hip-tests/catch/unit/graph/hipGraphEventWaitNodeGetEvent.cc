/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
Testcase Scenarios :
 1) Validate that the event returned by hipGraphEventWaitNodeGetEvent matches
with the event set in hipGraphAddEventWaitNode.
 2) Negative Scenarios
    - Input node parameter is passed as nullptr.
    - Output event parameter is passed as nullptr.
    - Input node parameter is an empty node.
    - Input node parameter is a memset node.
    - Input node parameter is a event record node.
    - Input node parameter is an uninitialized node.
*/

#include <hip_test_checkers.hh>
#include <hip_test_common.hh>
#include <hip_test_kernels.hh>


/**
 * Local Function
 */
static void validateEventWaitNodeGetEvent(unsigned flag) {
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipEvent_t event, event_out;
  HIP_CHECK(hipEventCreateWithFlags(&event, flag));
  hipGraphNode_t eventwait;
  HIP_CHECK(hipGraphAddEventWaitNode(&eventwait, graph, nullptr, 0, event));
  HIP_CHECK(hipGraphEventWaitNodeGetEvent(eventwait, &event_out));
  // validate set event and get event are same
  REQUIRE(event == event_out);
  // Instantiate and launch the graph
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipEventDestroy(event));
}

/**
 * Scenario 1
 */
HIP_TEST_CASE(Unit_hipGraphEventWaitNodeGetEvent_Functional) {
  // Create event nodes with different flags and validate with
  // hipGraphEventWaitNodeGetEvent.
  SECTION("Flag = hipEventDefault") { validateEventWaitNodeGetEvent(hipEventDefault); }

  SECTION("Flag = hipEventBlockingSync") { validateEventWaitNodeGetEvent(hipEventBlockingSync); }

  SECTION("Flag = hipEventDisableTiming") { validateEventWaitNodeGetEvent(hipEventDisableTiming); }
}

/**
 * Scenario 2
 */
HIP_TEST_CASE(Unit_hipGraphEventWaitNodeGetEvent_Negative) {
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipEvent_t event_out;
  hipEvent_t event1, event2;
  HIP_CHECK(hipEventCreate(&event1));
  HIP_CHECK(hipEventCreate(&event2));
  hipGraphNode_t eventrec, eventwait;
  HIP_CHECK(hipGraphAddEventRecordNode(&eventrec, graph, nullptr, 0, event1));
  HIP_CHECK(hipGraphAddEventWaitNode(&eventwait, graph, nullptr, 0, event2));

  SECTION("node = nullptr") {
    HIP_CHECK_ERROR(hipGraphEventWaitNodeGetEvent(nullptr, &event_out), hipErrorInvalidValue);
  }

  SECTION("event_out = nullptr") {
    HIP_CHECK_ERROR(hipGraphEventWaitNodeGetEvent(eventwait, nullptr), hipErrorInvalidValue);
  }

  SECTION("input node is empty node") {
    hipGraphNode_t EmptyGraphNode;
    HIP_CHECK(hipGraphAddEmptyNode(&EmptyGraphNode, graph, nullptr, 0));
    HIP_CHECK_ERROR(hipGraphEventWaitNodeGetEvent(EmptyGraphNode, &event_out),
                    hipErrorInvalidValue);
  }

  SECTION("input node is memset node") {
    constexpr size_t Nbytes = 1024;
    char* A_d;
    hipGraphNode_t memset_A;
    hipMemsetParams memsetParams{};
    HIP_CHECK(hipMalloc(&A_d, Nbytes));
    memset(&memsetParams, 0, sizeof(memsetParams));
    memsetParams.dst = reinterpret_cast<void*>(A_d);
    memsetParams.value = 0;
    memsetParams.pitch = 0;
    memsetParams.elementSize = sizeof(char);
    memsetParams.width = Nbytes;
    memsetParams.height = 1;
    HIP_CHECK(hipGraphAddMemsetNode(&memset_A, graph, nullptr, 0, &memsetParams));
    HIP_CHECK_ERROR(hipGraphEventWaitNodeGetEvent(memset_A, &event_out), hipErrorInvalidValue);
    HIP_CHECK(hipFree(A_d));
  }

  SECTION("input node is event record node") {
    HIP_CHECK_ERROR(hipGraphEventWaitNodeGetEvent(eventrec, &event_out), hipErrorInvalidValue);
  }

  SECTION("input node is uninitialized") {
    hipGraphNode_t node_uninit{};
    HIP_CHECK_ERROR(hipGraphEventWaitNodeGetEvent(node_uninit, &event_out), hipErrorInvalidValue);
  }

  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipEventDestroy(event1));
  HIP_CHECK(hipEventDestroy(event2));
}
