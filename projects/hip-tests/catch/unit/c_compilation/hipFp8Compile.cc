/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
/**
 * Test Description
 * ------------------------
 *    - Include hip/hip_fp8.h from a plain C translation unit and use the fp8
 *      POD types by name. Regression guard for the C-compatible fallback in
 *      amd_hip_fp8.h; if that fallback is removed the C object fails to compile
 *      and the build breaks.

 * Test source
 * ------------------------
 *    - catch/unit/c_compilation/hipFp8Compile.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.6
 */

extern "C" int hipFp8CCompile();

HIP_TEST_CASE(Unit_hipFp8Compile_ctest) {
  int result = hipFp8CCompile();
  REQUIRE(result == 1);
}
