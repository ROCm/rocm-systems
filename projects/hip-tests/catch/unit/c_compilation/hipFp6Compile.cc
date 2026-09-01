/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
/**
 * Test Description
 * ------------------------
 *    - Include hip/hip_fp6.h from a plain C translation unit and use the fp6
 *      POD types by name. Regression guard for the C-compatible fallback in
 *      amd_hip_fp6.h; if that fallback is removed the C object fails to compile
 *      and the build breaks.

 * Test source
 * ------------------------
 *    - catch/unit/c_compilation/hipFp6Compile.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.6
 */

extern "C" int hipFp6CCompile();

HIP_TEST_CASE(Unit_hipFp6Compile_ctest) {
  int result = hipFp6CCompile();
  REQUIRE(result == 1);
}
