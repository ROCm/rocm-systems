/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/*
 * Regression test for hip/hip_bf16.h being includable from a plain C
 * translation unit (compiled with `-x c`).
 *
 * hip/hip_bf16.h -> hip/amd_detail/amd_hip_bf16.h defines the bfloat16 types
 * as C++-only classes. Without a C fallback the header pulls in C++ stdlib
 * headers (<climits>/<cmath>) and C++-only syntax (constructors, `= default`,
 * conversion operators, static_cast/brace-init), so this file fails to compile
 * as C. amd_hip_bf16.h provides layout-compatible POD typedefs under
 * `#if !defined(__cplusplus)`; if that fallback is removed this TU stops
 * compiling and the c_compilation build target fails.
 */

#include <hip/hip_bf16.h>
#include <stdint.h>
#include <stdio.h>

/* Exercise the types by name in C, not just the include. */
int hipBf16CCompile() {
  __hip_bfloat16_raw raw16;
  __hip_bfloat162_raw raw162;
  __hip_bfloat16 bf16;
  __hip_bfloat162 bf162;

  raw16.x = (uint16_t)0x3F80U;   /* 1.0 in bfloat16 */
  raw162.x = raw16.x;
  raw162.y = raw16.x;
  bf16.__x = raw16.x;
  bf162.x = bf16;
  bf162.y = bf16;

  /* POD layout must match the C++ definitions exactly. */
  if (sizeof(__hip_bfloat16) == 2 && sizeof(__hip_bfloat16_raw) == 2 &&
      sizeof(__hip_bfloat162) == 4 && sizeof(__hip_bfloat162_raw) == 4 &&
      bf162.x.__x == raw16.x && raw162.y == raw16.x) {
    printf("PASSED!\n");
    return 1;
  } else {
    printf("FAILED!\n");
    return 0;
  }
}
