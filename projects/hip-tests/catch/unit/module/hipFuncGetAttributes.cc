/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>

/**
 * @addtogroup hipFuncGetAttributes
 * @{
 * @ingroup ModuleTest
 * `hipError_t hipFuncGetAttributes(struct hipFuncAttributes* attr, const void* func)` -
 * Find out attributes for a given function
 */

/**
 * Test Description
 * ------------------------
 * - Test case to Find out attributes for a given function.

 * Test source
 * ------------------------
 * - catch/unit/module/hipFuncGetAttributes.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 5.6
 */

__global__ void getAttrFn(float* px, float* py) {
  *px = *px + 1.0f;
  *py = *py + *px;
}

HIP_TEST_CASE(Unit_hipFuncGetAttributes_basic) {
  hipFuncAttributes attr{};

  auto r = hipFuncGetAttributes(&attr, reinterpret_cast<const void*>(&getAttrFn));
  REQUIRE(r == hipSuccess);
  REQUIRE(attr.maxThreadsPerBlock != 0);
}
