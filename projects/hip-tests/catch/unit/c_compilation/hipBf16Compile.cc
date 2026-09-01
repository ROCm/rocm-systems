/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
/**
 * Test Description
 * ------------------------
 *    - Include hip/hip_bf16.h from a plain C translation unit and use the
 *      bfloat16 POD types by name. Regression guard for the C-compatible
 *      fallback in amd_hip_bf16.h; if that fallback is removed the C object
 *      fails to compile and the build breaks.

 * Test source
 * ------------------------
 *    - catch/unit/c_compilation/hipBf16Compile.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.6
 */

extern "C" int hipBf16CCompile();

HIP_TEST_CASE(Unit_hipBf16Compile_ctest) {
  int result = hipBf16CCompile();
  REQUIRE(result == 1);
}
