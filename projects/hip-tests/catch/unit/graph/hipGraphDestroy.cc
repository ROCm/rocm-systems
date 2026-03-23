/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

/**
 * @addtogroup hipGraphDestroy hipGraphDestroy
 * @{
 * @ingroup GraphTest
 * `hipGraphDestroy(hipGraph_t graph)` -
 * Destroys a graph
 */

/**
 * Test Description
 * ------------------------
 *    - Basic positive test for hipGraphDestroy
 *    - Create an emtpy graph and then destroy it
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphDestroy.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_hipGraphDestroy_Positive_Basic) {
  hipGraph_t graph = nullptr;

  HIP_CHECK(hipGraphCreate(&graph, 0));
  REQUIRE(nullptr != graph);

  HIP_CHECK(hipGraphDestroy(graph));
}

/**
 * Test Description
 * ------------------------
 *    - Basic negative parameter test for hipGraphDestroy
 *        -# Expected hipErrorInvalidValue when graph is invalid
 * Test source
 * ------------------------
 *    - unit/graph/hipGraphDestroy.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_hipGraphDestroy_Negative_Parameters) {
  HIP_CHECK_ERROR(hipGraphDestroy(static_cast<hipGraph_t>(nullptr)), hipErrorInvalidValue);
}

/**
 * End doxygen group GraphTest.
 * @}
 */
