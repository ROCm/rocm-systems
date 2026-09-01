/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/*
 * Regression test for hip/hip_fp6.h being includable from a plain C
 * translation unit (compiled with `-x c`).
 *
 */

#include <hip/hip_fp6.h>
#include <stdint.h>
#include <stdio.h>

/* Exercise the C surface by name, not just the include. */
int hipFp6CCompile() {
  enum __hip_fp6_interpretation_t interp = __HIP_E2M3;
  __hip_fp6_storage_t s1 = (__hip_fp6_storage_t)0x0CU;
  __hip_fp6x2_storage_t s2 = (__hip_fp6x2_storage_t)0x0C0CU;
  __hip_fp6x4_storage_t s4 = (__hip_fp6x4_storage_t)0x0C0C0C0CU;

  if (sizeof(__hip_fp6_storage_t) == 1 && sizeof(__hip_fp6x2_storage_t) == 2 &&
      sizeof(__hip_fp6x4_storage_t) == 4 && s1 == 0x0CU && s2 == 0x0C0CU &&
      s4 == 0x0C0C0C0CU && interp == __HIP_E2M3) {
    printf("PASSED!\n");
    return 1;
  } else {
    printf("FAILED!\n");
    return 0;
  }
}
