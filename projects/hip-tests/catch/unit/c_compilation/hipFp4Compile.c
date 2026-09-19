/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/*
 * Regression test for hip/hip_fp4.h being includable from a plain C
 * translation unit (compiled with `-x c`).
 *
 */

#include <hip/hip_fp4.h>
#include <stdint.h>
#include <stdio.h>

/* Exercise the C surface by name, not just the include. */
int hipFp4CCompile() {
  enum __hip_fp4_interpretation_t interp = __HIP_E2M1;
  __hip_fp4_storage_t s1 = (__hip_fp4_storage_t)0x2U;
  __hip_fp4x2_storage_t s2 = (__hip_fp4x2_storage_t)0x22U;
  __hip_fp4x4_storage_t s4 = (__hip_fp4x4_storage_t)0x2222U;

  if (sizeof(__hip_fp4_storage_t) == 1 && sizeof(__hip_fp4x2_storage_t) == 1 &&
      sizeof(__hip_fp4x4_storage_t) == 2 && s1 == 0x2U && s2 == 0x22U &&
      s4 == 0x2222U && interp == __HIP_E2M1) {
    printf("PASSED!\n");
    return 1;
  } else {
    printf("FAILED!\n");
    return 0;
  }
}
