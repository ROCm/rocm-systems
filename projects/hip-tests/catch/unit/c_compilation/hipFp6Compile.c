/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/*
 * Regression test for hip/hip_fp6.h being includable from a plain C
 * translation unit (compiled with `-x c`).
 *
 * hip/hip_fp6.h -> hip/amd_detail/amd_hip_fp6.h defines the fp6 types as
 * C++-only classes (constructors, `= default`, conversion operators) with no
 * C fallback, and pulls in amd_hip_fp8.h / amd_hip_ocp_types.h which use
 * C++-only syntax and file-scope `static_assert`. amd_hip_fp6.h provides
 * layout-compatible POD typedefs, the storage typedefs and the interpretation
 * enum under `#if !defined(__cplusplus)`; if that fallback is removed this TU
 * stops compiling and the c_compilation build target fails.
 */

#include <hip/hip_fp6.h>
#include <stdint.h>
#include <stdio.h>

/* Exercise the types by name in C, not just the include. */
int hipFp6CCompile() {
  enum __hip_fp6_interpretation_t interp = __HIP_E2M3;
  __hip_fp6_storage_t s1 = (__hip_fp6_storage_t)0x0CU;
  __hip_fp6x2_storage_t s2 = (__hip_fp6x2_storage_t)0x0C0CU;
  __hip_fp6x4_storage_t s4 = (__hip_fp6x4_storage_t)0x0C0C0C0CU;

  __hip_fp6_e2m3 a;
  __hip_fp6_e3m2 b;
  __hip_fp6x2_e2m3 a2;
  __hip_fp6x2_e3m2 b2;
  __hip_fp6x4_e2m3 a4;
  __hip_fp6x4_e3m2 b4;

  a.__x = s1;
  b.__x = s1;
  a2.__x = s2;
  b2.__x = s2;
  a4.__x = s4;
  b4.__x = s4;

  /* POD layout must match the C++ definitions exactly. */
  if (sizeof(__hip_fp6_e2m3) == 1 && sizeof(__hip_fp6_e3m2) == 1 &&
      sizeof(__hip_fp6x2_e2m3) == 2 && sizeof(__hip_fp6x2_e3m2) == 2 &&
      sizeof(__hip_fp6x4_e2m3) == 4 && sizeof(__hip_fp6x4_e3m2) == 4 &&
      a.__x == s1 && a2.__x == s2 && a4.__x == s4 && interp == __HIP_E2M3) {
    printf("PASSED!\n");
    return 1;
  } else {
    printf("FAILED!\n");
    return 0;
  }
}
