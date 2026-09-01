/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/*
 * Regression test for hip/hip_fp4.h being includable from a plain C
 * translation unit (compiled with `-x c`).
 *
 * hip/hip_fp4.h -> hip/amd_detail/amd_hip_fp4.h defines the fp4 types as
 * C++-only classes (constructors, `= default`, conversion operators) with no
 * C fallback, and pulls in amd_hip_fp8.h / amd_hip_ocp_types.h which use
 * C++-only syntax and file-scope `static_assert`. amd_hip_fp4.h provides
 * layout-compatible POD typedefs, the storage typedefs and the interpretation
 * enum under `#if !defined(__cplusplus)`; if that fallback is removed this TU
 * stops compiling and the c_compilation build target fails.
 */

#include <hip/hip_fp4.h>
#include <stdint.h>
#include <stdio.h>

/* Exercise the types by name in C, not just the include. */
int hipFp4CCompile() {
  enum __hip_fp4_interpretation_t interp = __HIP_E2M1;
  __hip_fp4_storage_t s1 = (__hip_fp4_storage_t)0x2U;
  __hip_fp4x2_storage_t s2 = (__hip_fp4x2_storage_t)0x22U;
  __hip_fp4x4_storage_t s4 = (__hip_fp4x4_storage_t)0x2222U;

  __hip_fp4_e2m1 a;
  __hip_fp4x2_e2m1 a2;
  __hip_fp4x4_e2m1 a4;

  a.__x = s1;
  a2.__x = s2;
  a4.__x = s4;

  /* POD layout must match the C++ definitions exactly. */
  if (sizeof(__hip_fp4_e2m1) == 1 && sizeof(__hip_fp4x2_e2m1) == 1 &&
      sizeof(__hip_fp4x4_e2m1) == 2 && a.__x == s1 && a2.__x == s2 &&
      a4.__x == s4 && interp == __HIP_E2M1) {
    printf("PASSED!\n");
    return 1;
  } else {
    printf("FAILED!\n");
    return 0;
  }
}
