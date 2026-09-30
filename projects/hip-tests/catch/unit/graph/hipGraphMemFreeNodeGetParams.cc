/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

/**
 * @addtogroup hipGraphMemFreeNodeGetParams hipGraphMemFreeNodeGetParams
 * @{
 * @ingroup GraphTest
 * `hipError_t hipGraphMemFreeNodeGetParams(hipGraphNode_t node, void *dev_ptr)`-
 * Returns parameters for memory free node.
 */

/**
 * Test Description
 * ------------------------
 *  - Test if the function returns expected values when valid arguments are provided.
 * Test source
 * ------------------------
 *  - catch/unit/graph/hipGraphMemFreeNodeGetParams.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.1
 */
HIP_TEST_CASE(Unit_hipGraphMemFreeNodeGetParams_ValidArgs) {
  hipGraphNode_t allocNode, freeNode;
  hipGraph_t graph;
  hipGraphExec_t graphExec;
  hipMemAllocNodeParams allocParam;
  hipStream_t stream;
  void* out_dev_ptr;

  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipStreamCreate(&stream));

  memset(&allocParam, 0, sizeof(allocParam));
  allocParam.bytesize = 256;
  allocParam.poolProps.allocType = hipMemAllocationTypePinned;
  allocParam.poolProps.location.id = 0;
  allocParam.poolProps.location.type = hipMemLocationTypeDevice;

  HIP_CHECK(hipGraphAddMemAllocNode(&allocNode, graph, nullptr, 0, &allocParam));

  HIP_CHECK(hipGraphAddMemFreeNode(&freeNode, graph, &allocNode, 1, allocParam.dptr));

  HIP_CHECK(hipGraphMemFreeNodeGetParams(freeNode, &out_dev_ptr));

  HIP_CHECK(
      hipGraphInstantiateWithFlags(&graphExec, graph, hipGraphInstantiateFlagAutoFreeOnLaunch));

  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipStreamDestroy(stream));

  REQUIRE(out_dev_ptr == allocParam.dptr);
}

/**
 * Test Description
 * ------------------------
 *  - Test if the function returns expected values when invalid arguments are provided.
 * Test source
 * ------------------------
 *  - catch/unit/graph/hipGraphMemFreeNodeGetParams.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 6.1
 */
HIP_TEST_CASE(Unit_hipGraphMemFreeNodeGetParams_InvalidArgs) {
  hipGraphNode_t allocNode, freeNode;
  hipMemAllocNodeParams allocParam;
  hipGraph_t graph;
  hipGraphExec_t graphExec;
  hipStream_t stream;

  HIP_CHECK(hipGraphCreate(&graph, 0));
  HIP_CHECK(hipStreamCreate(&stream));

  memset(&allocParam, 0, sizeof(allocParam));
  allocParam.bytesize = 256;
  allocParam.poolProps.allocType = hipMemAllocationTypePinned;
  allocParam.poolProps.location.id = 0;
  allocParam.poolProps.location.type = hipMemLocationTypeDevice;

  HIP_CHECK(hipGraphAddMemAllocNode(&allocNode, graph, nullptr, 0, &allocParam));

  HIP_CHECK(hipGraphAddMemFreeNode(&freeNode, graph, &allocNode, 1, allocParam.dptr));

  SECTION("Null graph node") {
    void* out_dev_ptr;
    HIP_CHECK_ERROR(hipGraphMemFreeNodeGetParams(nullptr, &out_dev_ptr), hipErrorInvalidValue);
  }

  SECTION("Null out pointer") {
    HIP_CHECK_ERROR(hipGraphMemFreeNodeGetParams(freeNode, nullptr), hipErrorInvalidValue);
  }

  SECTION("Mismatched node type") {
    void* out_dev_ptr;
    HIP_CHECK_ERROR(hipGraphMemFreeNodeGetParams(allocNode, &out_dev_ptr), hipErrorInvalidValue);
  }

  // Cleanup

  HIP_CHECK(
      hipGraphInstantiateWithFlags(&graphExec, graph, hipGraphInstantiateFlagAutoFreeOnLaunch));

  HIP_CHECK(hipGraphLaunch(graphExec, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipStreamDestroy(stream));
}
