/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/*
 * Regression test for hip/hip_bf16.h being includable from a plain C
 * translation unit (compiled with `-x c`).
 *
 */

#include <hip/hip_bf16.h>
#include <stdint.h>
#include <stdio.h>

/* Exercise the C surface by name, not just the include. */
int hipBf16CCompile() {
  __hip_bfloat16_raw raw16;
  __hip_bfloat162_raw raw162;

  raw16.x = (uint16_t)0x3F80U;   /* 1.0 in bfloat16 */
  raw162.x = raw16.x;
  raw162.y = raw16.x;

  /* POD layout must match the C++ definitions exactly. */
  if (sizeof(__hip_bfloat16_raw) == 2 && sizeof(__hip_bfloat162_raw) == 4 &&
      raw162.x == raw16.x && raw162.y == raw16.x) {
    printf("PASSED!\n");
    return 1;
  } else {
    printf("FAILED!\n");
    return 0;
  }
}
