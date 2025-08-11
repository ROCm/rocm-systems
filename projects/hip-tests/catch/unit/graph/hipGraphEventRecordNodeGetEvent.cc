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
 1) Validate that the event returned by hipGraphEventRecordNodeGetEvent matches
with the event set in hipGraphAddEventRecordNode.
 2) Negative Scenarios
    - Input node is a nullptr.
    - Output event is a nullptr.
    - Input node is an empty node.
    - Input node is a memset node.
    - Input node is event wait node
    - Input node is an uninitialized node.
*/

#include <hip_test_checkers.hh>
#include <hip_test_common.hh>
#include <hip_test_kernels.hh>

/**
 * Local Function to set and get event record node property.
 */
static void validateEventRecordNodeGetEvent(unsigned flag) {
  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));
  hipEvent_t event, event_out;
  HIP_CHECK(hipEventCreateWithFlags(&event, flag));
  hipGraphNode_t eventrec;
  HIP_CHECK(hipGraphAddEventRecordNode(&eventrec, graph, nullptr, 0, event));
  HIP_CHECK(hipGraphEventRecordNodeGetEvent(eventrec, &event_out));
  // validate set event and get event are same
  REQUIRE(event == event_out);
  // Instantiate and launch the graph
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipEventDestroy(event));
}

/**
 * Scenario: Validate scenario 1 for different event flags.
 */
TEST_CASE("Unit_hipGraphEventRecordNodeGetEvent_Functional") {
  // Create event nodes with different flags and validate with
  // hipGraphEventRecordNodeGetEvent.
  SECTION("Flag = hipEventDefault") {
    validateEventRecordNodeGetEvent(hipEventDefault);
  }

  SECTION("Flag = hipEventBlockingSync") {
    validateEventRecordNodeGetEvent(hipEventBlockingSync);
  }

  SECTION("Flag = hipEventDisableTiming") {
    validateEventRecordNodeGetEvent(hipEventDisableTiming);
  }
}

/**
 * Scenario 2: Negative tests.
 */
TEST_CASE("Unit_hipGraphEventRecordNodeGetEvent_Negative") {
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
    HIP_CHECK_ERROR(hipGraphEventRecordNodeGetEvent(nullptr, &event_out), hipErrorInvalidValue);
  }

  SECTION("event_out = nullptr") {
    HIP_CHECK_ERROR(hipGraphEventRecordNodeGetEvent(eventrec, nullptr), hipErrorInvalidValue);
  }

  SECTION("input node is empty node") {
    hipGraphNode_t EmptyGraphNode;
    HIP_CHECK(hipGraphAddEmptyNode(&EmptyGraphNode, graph, nullptr, 0));
    HIP_CHECK_ERROR(hipGraphEventRecordNodeGetEvent(EmptyGraphNode, &event_out),
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
    HIP_CHECK_ERROR(hipGraphEventRecordNodeGetEvent(memset_A, &event_out), hipErrorInvalidValue);
    HIP_CHECK(hipFree(A_d));
  }

  SECTION("input node is event wait node") {
    HIP_CHECK_ERROR(hipGraphEventRecordNodeGetEvent(eventwait, &event_out), hipErrorInvalidValue);
  }

  SECTION("input node is uninitialized node") {
    hipGraphNode_t node_unit{};
    HIP_CHECK_ERROR(hipGraphEventRecordNodeGetEvent(node_unit, &event_out), hipErrorInvalidValue);
  }

  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipEventDestroy(event1));
  HIP_CHECK(hipEventDestroy(event2));
}
