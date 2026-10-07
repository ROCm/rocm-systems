/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/*
 * Regression test for hip/hip_fp8.h being includable from a plain C
 * translation unit (compiled with `-x c`).
 *
 */

#include <hip/hip_fp8.h>
#include <stdint.h>
#include <stdio.h>

/* Exercise the C surface by name, not just the include. */
int hipFp8CCompile() {
  /* enums and storage typedefs are the public C API surface. */
  enum __hip_fp8_interpretation_t interp = __HIP_E4M3;
  enum __hip_saturation_t sat = __HIP_NOSAT;
  __hip_fp8_storage_t s1 = (__hip_fp8_storage_t)0x38U;   /* ~1.0 in e4m3 */
  __hip_fp8x2_storage_t s2 = (__hip_fp8x2_storage_t)0x3838U;
  __hip_fp8x4_storage_t s4 = (__hip_fp8x4_storage_t)0x38383838U;

  if (sizeof(__hip_fp8_storage_t) == 1 && sizeof(__hip_fp8x2_storage_t) == 2 &&
      sizeof(__hip_fp8x4_storage_t) == 4 && s1 == 0x38U && s2 == 0x3838U &&
      s4 == 0x38383838U && interp == __HIP_E4M3 && sat == __HIP_NOSAT) {
    printf("PASSED!\n");
    return 1;
  } else {
    printf("FAILED!\n");
    return 0;
  }
}
