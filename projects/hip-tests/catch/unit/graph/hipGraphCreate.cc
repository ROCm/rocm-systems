/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

/**
 * @addtogroup hipGraphCreate hipGraphCreate
 * @{
 * @ingroup GraphTest
 * `hipGraphCreate(hipGraph_t *pGraph, unsigned int flags)` -
 * creates a graph
 */

/**
 * Test Description
 * ------------------------
 *    - Negative parameter test for hipGraphCreate:
 *        -# Expected hipErrorInvalidValue when pGraph is null
 *        -# Expected hipErrorInvalidValue when flags is not 0
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphCreate.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_hipGraphCreate_Negative_Parameters) {
  hipGraph_t graph = nullptr;

  SECTION("pGraph is nullptr") {
    HIP_CHECK_ERROR(hipGraphCreate(nullptr, 0), hipErrorInvalidValue);
  }

  SECTION("flags is not 0") { HIP_CHECK_ERROR(hipGraphCreate(&graph, 1), hipErrorInvalidValue); }
}

/**
 * Test Description
 * ------------------------
 *    - Basic positive test for hipGraphCreate
 *    - Create an emtpy graph
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphCreate.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_hipGraphCreate_Positive_Basic) {
  hipGraph_t graph = nullptr;

  HIP_CHECK(hipGraphCreate(&graph, 0));
  REQUIRE(nullptr != graph);

  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * End doxygen group GraphTest.
 * @}
 */
