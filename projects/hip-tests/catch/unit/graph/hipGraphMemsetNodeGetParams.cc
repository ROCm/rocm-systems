/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <resource_guards.hh>

/**
 * @addtogroup hipGraphMemsetNodeGetParams hipGraphMemsetNodeGetParams
 * @{
 * @ingroup GraphTest
 * `hipGraphMemsetNodeGetParams(hipGraphNode_t node, hipMemsetParams *pNodeParams)` -
 * 	Gets a memset node's parameters
 * ________________________
 * Test cases from other APIs:
 *  - @ref Unit_hipGraphMemsetNodeSetParams_Positive_Basic
 *  - @ref Unit_hipGraphExecMemsetNodeSetParams_Positive_Basic
 */

/**
 * Test Description
 * ------------------------
 *    - Verify API behaviour with invalid arguments:
 *      -# node is nullptr
 *      -# pNodeParams is nullptr
 *      -# node is destroyed
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphMemsetNodeGetParams.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_hipGraphMemsetNodeGetParams_Negative_Parameters) {
  CHECK_IMAGE_SUPPORT

  LinearAllocGuard2D<int> alloc(1, 1);
  hipMemsetParams params = {};
  params.dst = alloc.ptr();
  params.elementSize = sizeof(int);
  params.width = 1;
  params.height = 1;

  hipGraph_t graph = nullptr;
  hipGraphNode_t node = nullptr;

  SECTION("node == nullptr") {
    HIP_CHECK_ERROR(hipGraphMemsetNodeGetParams(nullptr, &params), hipErrorInvalidValue);
  }

  SECTION("pNodeParams == nullptr") {
    HIP_CHECK(hipGraphCreate(&graph, 0));
    HIP_CHECK(hipGraphAddMemsetNode(&node, graph, nullptr, 0, &params));
    HIP_CHECK_ERROR(hipGraphMemsetNodeGetParams(node, nullptr), hipErrorInvalidValue);
    HIP_CHECK(hipGraphDestroy(graph));
  }

// Disabled on AMD due to defect - EXSWHTEC-208
#if 0
  SECTION("Node is destroyed") {
    HIP_CHECK(hipGraphCreate(&graph, 0));
    HIP_CHECK(hipGraphAddMemsetNode(&node, graph, nullptr, 0, &params));
    HIP_CHECK(hipGraphDestroy(graph));
    HIP_CHECK_ERROR(hipGraphMemsetNodeGetParams(node, &params), hipErrorInvalidValue);
  }
#endif
}

/**
 * End doxygen group GraphTest.
 * @}
 */
