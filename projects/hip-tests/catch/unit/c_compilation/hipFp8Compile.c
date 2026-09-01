/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/*
 * Regression test for hip/hip_fp8.h being includable from a plain C
 * translation unit (compiled with `-x c`).
 *
 * hip/hip_fp8.h -> hip/amd_detail/amd_hip_fp8.h defines the fp8 types as
 * C++-only classes. Without a C fallback the header pulls in C++ stdlib
 * headers (<climits>) and C++-only syntax (constructors, `= default`,
 * conversion operators, namespaces/templates), and its transitive include
 * amd_hip_ocp_types.h uses file-scope `static_assert`, so this file fails to
 * compile as C. amd_hip_fp8.h provides layout-compatible POD typedefs, the
 * storage typedefs and the interpretation/saturation enums under
 * `#if !defined(__cplusplus)`; if that fallback is removed this TU stops
 * compiling and the c_compilation build target fails.
 */

#include <hip/hip_fp8.h>
#include <stdint.h>
#include <stdio.h>

/* Exercise the types by name in C, not just the include. */
int hipFp8CCompile() {
  /* enums and storage typedefs are part of the public C API surface. */
  enum __hip_fp8_interpretation_t interp = __HIP_E4M3;
  enum __hip_saturation_t sat = __HIP_NOSAT;
  __hip_fp8_storage_t s1 = (__hip_fp8_storage_t)0x38U;   /* ~1.0 in e4m3 */
  __hip_fp8x2_storage_t s2 = (__hip_fp8x2_storage_t)0x3838U;
  __hip_fp8x4_storage_t s4 = (__hip_fp8x4_storage_t)0x38383838U;

  /* fnuz family */
  __hip_fp8_e4m3_fnuz a;
  __hip_fp8x2_e4m3_fnuz a2;
  __hip_fp8x4_e4m3_fnuz a4;
  __hip_fp8_e5m2_fnuz b;
  /* ocp family */
  __hip_fp8_e4m3 c;
  __hip_fp8x2_e5m2 c2;
  __hip_fp8x4_e5m2 c4;
  /* scale type */
  __hip_fp8_e8m0 e;

  a.__x = s1;
  a2.__x = s2;
  a4.__x = s4;
  b.__x = s1;
  c.__x = s1;
  c2.__x = s2;
  c4.__x = s4;
  e.__x = s1;

  if (sizeof(__hip_fp8_e8m0) == 1 && sizeof(__hip_fp8_e4m3_fnuz) == 1 &&
      sizeof(__hip_fp8x2_e4m3_fnuz) == 2 && sizeof(__hip_fp8x4_e4m3_fnuz) == 4 &&
      sizeof(__hip_fp8_e5m2_fnuz) == 1 && sizeof(__hip_fp8_e4m3) == 1 &&
      sizeof(__hip_fp8x2_e5m2) == 2 && sizeof(__hip_fp8x4_e5m2) == 4 &&
      a.__x == s1 && a2.__x == s2 && a4.__x == s4 && e.__x == s1 &&
      interp == __HIP_E4M3 && sat == __HIP_NOSAT) {
    printf("PASSED!\n");
    return 1;
  } else {
    printf("FAILED!\n");
    return 0;
  }
}
