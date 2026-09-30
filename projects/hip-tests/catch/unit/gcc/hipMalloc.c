/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime_api.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif
int hipMallocfunc() {
  int* Ad;
  hipMalloc((void**)&Ad, 1024);
  printf("hipMalloc PASSED!\n");
  hipFree(Ad);
  return 1;
}
#ifdef __cplusplus
}
#endif
