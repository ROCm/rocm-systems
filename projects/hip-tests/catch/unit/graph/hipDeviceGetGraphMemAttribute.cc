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

#include <hip_test_common.hh>
#include <resource_guards.hh>
#include <utils.hh>

/**
 * @addtogroup hipDeviceGetGraphMemAttribute hipDeviceGetGraphMemAttribute
 * @{
 * @ingroup GraphTest
 * `hipDeviceGetGraphMemAttribute(int device, hipGraphMemAttributeType attr, void* value)` -
 * Get the mem attribute for graphs.
 */

static constexpr auto element_count{64 * 1024 * 1024};


/* Create graph with memory node */
static void createGraph(hipGraphExec_t* graph_exec, int** device_alloc = nullptr) {
  constexpr size_t num_bytes = element_count * sizeof(int);

  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  hipGraphNode_t alloc_node;
  hipMemAllocNodeParams alloc_param;
  memset(&alloc_param, 0, sizeof(alloc_param));
  alloc_param.bytesize = num_bytes;
  alloc_param.poolProps.allocType = hipMemAllocationTypePinned;
  alloc_param.poolProps.location.id = 0;
  alloc_param.poolProps.location.type = hipMemLocationTypeDevice;

  HIP_CHECK(hipGraphAddMemAllocNode(&alloc_node, graph, nullptr, 0, &alloc_param));
  REQUIRE(alloc_param.dptr != nullptr);
  int* A_d = reinterpret_cast<int*>(alloc_param.dptr);

  if (device_alloc == nullptr) {
    hipGraphNode_t free_node;
    HIP_CHECK(hipGraphAddMemFreeNode(&free_node, graph, &alloc_node, 1, (void*)A_d));
  } else {
    *device_alloc = A_d;
  }

  // Instantiate graph
  HIP_CHECK(hipGraphInstantiate(graph_exec, graph, nullptr, nullptr, 0));

  HIP_CHECK(hipGraphDestroy(graph));
}

/* check if memory attributes for graphs contain expected values */
static void checkGraphMemAttribute(size_t used_mem, size_t high_mem) {
  size_t read_mem;
  hipGraphMemAttributeType attr = hipGraphMemAttrUsedMemCurrent;
  HIP_CHECK(hipDeviceGetGraphMemAttribute(0, attr, reinterpret_cast<void*>(&read_mem)));
  REQUIRE(read_mem == used_mem);

  attr = hipGraphMemAttrReservedMemCurrent;
  HIP_CHECK(hipDeviceGetGraphMemAttribute(0, attr, reinterpret_cast<void*>(&read_mem)));
  REQUIRE(read_mem == used_mem);

  attr = hipGraphMemAttrUsedMemHigh;
  HIP_CHECK(hipDeviceGetGraphMemAttribute(0, attr, reinterpret_cast<void*>(&read_mem)));
  REQUIRE(read_mem == high_mem);

  attr = hipGraphMemAttrReservedMemHigh;
  HIP_CHECK(hipDeviceGetGraphMemAttribute(0, attr, reinterpret_cast<void*>(&read_mem)));
  REQUIRE(read_mem == high_mem);
}

// Reset memory graph attributes back to the original state
static void ResetGraphMemAttribute(unsigned deviceId = 0) {
  size_t mem_size = 0;
  hipGraphMemAttributeType attr = hipGraphMemAttrUsedMemHigh;
  HIP_CHECK(hipDeviceSetGraphMemAttribute(deviceId, attr, &mem_size));
  attr = hipGraphMemAttrReservedMemHigh;
  HIP_CHECK(hipDeviceSetGraphMemAttribute(deviceId, attr, &mem_size));
}

/**
 * Test Description
 * ------------------------
 *  - Basic test to verify that hipDeviceGetGraphMemAttribute return correct memory attribute values
 * when graphs with allocation nodes are launched, and after memory is freed to OS.
 * Test source
 * ------------------------
 *  - /unit/graph/hipDeviceGetGraphMemAttribute.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipDeviceGetGraphMemAttribute_Positive_DoubleMemory") {
  hipGraphExec_t graph_exec1, graph_exec2;
  int *dev_p1, *dev_p2;

  StreamGuard stream_guard(Streams::created);
  hipStream_t stream = stream_guard.stream();

  createGraph(&graph_exec1, &dev_p1);
  HIP_CHECK(hipGraphLaunch(graph_exec1, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  checkGraphMemAttribute(element_count * sizeof(int), element_count * sizeof(int));

  createGraph(&graph_exec2, &dev_p2);
  HIP_CHECK(hipGraphLaunch(graph_exec2, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  checkGraphMemAttribute(2 * element_count * sizeof(int), 2 * element_count * sizeof(int));

  HIP_CHECK(hipFree(dev_p1));
  HIP_CHECK(hipFree(dev_p2));

  HIP_CHECK(hipGraphExecDestroy(graph_exec1));
  HIP_CHECK(hipGraphExecDestroy(graph_exec2));

  HIP_CHECK(hipDeviceGraphMemTrim(0));
  HIP_CHECK(hipStreamSynchronize(0));

  checkGraphMemAttribute(0, 2 * element_count * sizeof(int));
  ResetGraphMemAttribute();
}

/**
 * Test Description
 * ------------------------
 *  - Test to verify hipDeviceGetGraphMemAttribute behavior with invalid arguments:
 *    -# Device is not valid
 *    -# Attribute value is not valid
 *    -# Get value is nullptr
 * Test source
 * ------------------------
 *  - /unit/graph/hipDeviceGetGraphMemAttribute.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.0
 */
TEST_CASE("Unit_hipDeviceGetGraphMemAttribute_Negative_Parameters") {
  int device_id = 0;
  HIP_CHECK(hipSetDevice(device_id));

  int num_dev = 0;
  HIP_CHECK(hipGetDeviceCount(&num_dev));

  hipGraphMemAttributeType attr = hipGraphMemAttrUsedMemHigh;
  size_t get_value = 0;

  SECTION("Device is not valid") {
    HIP_CHECK_ERROR(
        hipDeviceGetGraphMemAttribute(num_dev, attr, reinterpret_cast<void*>(&get_value)),
        hipErrorInvalidDevice);
  }

  SECTION("Attribute value is not valid") {
    HIP_CHECK_ERROR(hipDeviceGetGraphMemAttribute(0, static_cast<hipGraphMemAttributeType>(0x7),
                                                  reinterpret_cast<void*>(&get_value)),
                    hipErrorInvalidValue);
  }

  SECTION("Get value is nullptr") {
    HIP_CHECK_ERROR(hipDeviceGetGraphMemAttribute(0, attr, nullptr), hipErrorInvalidValue);
  }
  ResetGraphMemAttribute();
}

/**
 * Test Description
 * ------------------------
 *  - Functional Test for APIs
 *  - hipDeviceGetGraphMemAttribute
 *  - hipDeviceSetGraphMemAttribute
 *  - hipDeviceGraphMemTrim
 *   Scenarios - for api hipDeviceGetGraphMemAttribute
 *   Create a graph node using hipGraphAddMemAllocNode and corresponding
 *   hipGraphAddMemFreeNode to graph and launch and execute it.
 *   1) Check memory footprint check before launching graph.
 *   2) Check memory footprint check after launching and before destroying graph.
 *   3) Check memory footprint check after destroying graph.
 *   4) Test all those scenarios for all devices available.
 *
 *   Scenarios - for api hipDeviceSetGraphMemAttribute
 *   Create a graph node using hipGraphAddMemAllocNode and corresponding
 *   hipGraphAddMemFreeNode to graph and launch and execute it.
 *   1) Check memory footprint check after destroying graph and
 *      after Trim api call and reset to 0.
 *   2) Test all those scenarios for all devices available.
 *
 *   Scenarios - for api hipDeviceGraphMemTrim
 *   Create a graph node using hipGraphAddMemAllocNode and corresponding
 *   hipGraphAddMemFreeNode to graph and launch and execute it.
 *   1) Check memory footprint check after destroying graph and after Trim api call.
 *   2) Test all those scenarios for all devices available.
 *
 * Test source
 * ------------------------
 *  - /unit/graph/hipDeviceGetGraphMemAttribute.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.1
 */

static void Unit_hipDeviceGetGraphMemAttribute_Functional(unsigned deviceId = 0) {
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
  hipMemAllocNodeParams allocParams;

  HIP_CHECK(hipSetDevice(deviceId));

  HIP_CHECK(hipDeviceGraphMemTrim(deviceId));

  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipStreamCreate(&stream));

  memset(&allocParams, 0, sizeof(allocParams));
  allocParams.bytesize = Nbytes;
  allocParams.poolProps.allocType = hipMemAllocationTypePinned;
  allocParams.poolProps.location.id = deviceId;
  allocParams.poolProps.location.type = hipMemLocationTypeDevice;

  HIP_CHECK(hipGraphAddMemAllocNode(&allocNodeA, graph, nullptr, 0, &allocParams));
  REQUIRE(allocParams.dptr != nullptr);
  HIP_CHECK(hipGraphAddMemFreeNode(&freeNodeA, graph, &allocNodeA, 1,
                                   reinterpret_cast<void*>(allocParams.dptr)));

  size_t value = -1;
  SECTION("Memory footprint check before launching graph") {
    HIP_CHECK(hipDeviceGetGraphMemAttribute(deviceId, hipGraphMemAttrUsedMemCurrent, &value));
    REQUIRE(value == 0);
    HIP_CHECK(hipDeviceGetGraphMemAttribute(deviceId, hipGraphMemAttrUsedMemHigh, &value));
    REQUIRE(value == 0);
    HIP_CHECK(hipDeviceGetGraphMemAttribute(deviceId, hipGraphMemAttrReservedMemCurrent, &value));
    REQUIRE(value == 0);
    HIP_CHECK(hipDeviceGetGraphMemAttribute(deviceId, hipGraphMemAttrReservedMemHigh, &value));
    REQUIRE(value == 0);
  }

  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipStreamDestroy(stream));

  HIP_CHECK(hipDeviceGraphMemTrim(deviceId));

  value = -1;
  SECTION("Memory footprint check after destroying graph & Trim api call") {
    HIP_CHECK(hipDeviceGetGraphMemAttribute(deviceId, hipGraphMemAttrUsedMemCurrent, &value));
    REQUIRE(value == 0);
    HIP_CHECK(hipDeviceGetGraphMemAttribute(deviceId, hipGraphMemAttrUsedMemHigh, &value));
    REQUIRE(value == Nbytes);
    HIP_CHECK(hipDeviceGetGraphMemAttribute(deviceId, hipGraphMemAttrReservedMemCurrent, &value));
    REQUIRE(value == 0);
    HIP_CHECK(hipDeviceGetGraphMemAttribute(deviceId, hipGraphMemAttrReservedMemHigh, &value));
    REQUIRE(value == Nbytes);
  }

  value = 0;
  HIP_CHECK(hipDeviceSetGraphMemAttribute(deviceId, hipGraphMemAttrUsedMemHigh, &value));
  HIP_CHECK(hipDeviceSetGraphMemAttribute(deviceId, hipGraphMemAttrReservedMemHigh, &value));

  value = -1;
  SECTION(
      "Memory footprint check after destroying graph and"
      " after Trim api call and reset to 0") {
    HIP_CHECK(hipDeviceGetGraphMemAttribute(deviceId, hipGraphMemAttrUsedMemCurrent, &value));
    REQUIRE(value == 0);
    HIP_CHECK(hipDeviceGetGraphMemAttribute(deviceId, hipGraphMemAttrUsedMemHigh, &value));
    REQUIRE(value == 0);
    HIP_CHECK(hipDeviceGetGraphMemAttribute(deviceId, hipGraphMemAttrReservedMemCurrent, &value));
    REQUIRE(value == 0);
    HIP_CHECK(hipDeviceGetGraphMemAttribute(deviceId, hipGraphMemAttrReservedMemHigh, &value));
    REQUIRE(value == 0);
  }
  ResetGraphMemAttribute(deviceId);
}

TEST_CASE("Unit_hipDeviceGetGraphMemAttribute_Functional") {
  Unit_hipDeviceGetGraphMemAttribute_Functional();
}

TEST_CASE("Unit_hipDeviceGetGraphMemAttribute_Functional_Multi_Device") {
  int numDevices = 0;
  HIP_CHECK(hipGetDeviceCount(&numDevices));

  if (numDevices > 0) {
    for (int i = 0; i < numDevices; ++i) {
      Unit_hipDeviceGetGraphMemAttribute_Functional(i);
    }
  } else {
    HipTest::HIP_SKIP_TEST("Skipped test as there is no device to test.");
  }
}

/**
 * Test Description
 * ------------------------
 *  - Negative Test for API - hipDeviceGetGraphMemAttribute
 *  1) Pass device id as negative value.
 *  2) Pass device id which don't exist as INT_MAX.
 *  3) Pass hipGraphMemAttributeType other than existing (4 type) in this structure
 *  4) Pass value of footprint pointer as nullptr.
 * Test source
 * ------------------------
 *  - /unit/graph/hipDeviceGetGraphMemAttribute.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.1
 */

TEST_CASE("Unit_hipDeviceGetGraphMemAttribute_Negative") {
  size_t value = 0;
  hipError_t ret;
  SECTION("Pass device id as negative value") {
    ret = hipDeviceGetGraphMemAttribute(-1, hipGraphMemAttrUsedMemCurrent, &value);
    REQUIRE(ret == hipErrorInvalidDevice);
  }
  SECTION("Pass device id which don't exist as INT_MAX") {
    ret = hipDeviceGetGraphMemAttribute(INT_MAX, hipGraphMemAttrUsedMemCurrent, &value);
    REQUIRE(ret == hipErrorInvalidDevice);
  }
  SECTION("Pass hipGraphMemAttributeType other than existing 4 type") {
    ret = hipDeviceGetGraphMemAttribute(0, hipGraphMemAttributeType(99), &value);
    REQUIRE(ret == hipErrorInvalidValue);
  }
  SECTION("Pass value of footprint pointer as nullptr") {
    ret = hipDeviceGetGraphMemAttribute(0, hipGraphMemAttrUsedMemCurrent, nullptr);
    REQUIRE(ret == hipErrorInvalidValue);
  }
  ResetGraphMemAttribute();
}

/**
 * Test Description
 * ------------------------
 * - Negative Test for API - hipDeviceSetGraphMemAttribute
 *  1) Pass device id as negative value.
 *  2) Pass device id which don't exist as INT_MAX.
 *  3) Pass hipGraphMemAttributeType as hipGraphMemAttrUsedMemCurrent
 *  4) Pass hipGraphMemAttributeType as hipGraphMemAttrReservedMemCurrent
 *  5) Pass value of footprint pointer as nullptr
 *  6) Pass value of footprint pointer as negative value address
 * Test source
 * ------------------------
 *  - /unit/graph/hipDeviceGetGraphMemAttribute.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.1
 */

TEST_CASE("Unit_hipDeviceSetGraphMemAttribute_Negative") {
  size_t value = 0;
  hipError_t ret;
  SECTION("Pass device id as negative value") {
    ret = hipDeviceSetGraphMemAttribute(-3, hipGraphMemAttrUsedMemCurrent, &value);
    REQUIRE(ret == hipErrorInvalidDevice);
  }
  SECTION("Pass device id which don't exist as INT_MAX") {
    ret = hipDeviceSetGraphMemAttribute(INT_MAX, hipGraphMemAttrUsedMemCurrent, &value);
    REQUIRE(ret == hipErrorInvalidDevice);
  }
  SECTION("Pass hipGraphMemAttributeType as hipGraphMemAttrUsedMemCurrent") {
    ret = hipDeviceSetGraphMemAttribute(0, hipGraphMemAttrUsedMemCurrent, &value);
    REQUIRE(ret == hipErrorInvalidValue);
  }
  SECTION("Pass GraphMemAttributeType as hipGraphMemAttrReservedMemCurrent") {
    ret = hipDeviceSetGraphMemAttribute(0, hipGraphMemAttrReservedMemCurrent, &value);
    REQUIRE(ret == hipErrorInvalidValue);
  }
  SECTION("Pass value of footprint pointer as nullptr.") {
    ret = hipDeviceSetGraphMemAttribute(0, hipGraphMemAttrUsedMemCurrent, nullptr);
    REQUIRE(ret == hipErrorInvalidValue);
  }
  SECTION("Pass value of footprint pointer as negative value address.") {
    value = -1;
    ret = hipDeviceSetGraphMemAttribute(0, hipGraphMemAttrUsedMemCurrent, &value);
    REQUIRE(ret == hipErrorInvalidValue);
  }
  SECTION("Pass value of footprint pointer as -1.") {
    ret = hipDeviceSetGraphMemAttribute(0, hipGraphMemAttrUsedMemCurrent,
                                        reinterpret_cast<void*>(-1));
    REQUIRE(ret == hipErrorInvalidValue);
  }
  ResetGraphMemAttribute();
}

/**
 * End doxygen group GraphTest.
 * @}
 */
