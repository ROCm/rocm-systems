/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip_test_defgroups.hh>

/**
 * @addtogroup hipFuncSetAttribute
 * @{
 * @ingroup ModuleTest
 * `hipError_t hipFuncSetAttribute(const void* func, hipFuncAttribute attr, int value)` -
 * Set attributes for a specific function
 */

/**
 * Test Description
 * ------------------------
 * - Test case to set attributes for a specific function

 * Test source
 * ------------------------
 * - catch/unit/module/hipFuncSetAttribute.cc
 * Test requirements
 * ------------------------
 * - HIP_VERSION >= 5.6
*/

__global__ void fn(float* px, float* py) {
  *px = *px + 1.0f;
  *py = *py + *px;
}

HIP_TEST_CASE(Unit_hipFuncSetAttribute_Basic) {
  HIP_CHECK(hipFuncSetAttribute(reinterpret_cast<const void*>(&fn),
                                hipFuncAttributeMaxDynamicSharedMemorySize, 0));
  HIP_CHECK(hipFuncSetAttribute(reinterpret_cast<const void*>(&fn),
                                hipFuncAttributePreferredSharedMemoryCarveout, 0));
}
