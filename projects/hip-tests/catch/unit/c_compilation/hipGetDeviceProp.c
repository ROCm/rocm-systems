/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime_api.h>
#include <stdio.h>

int hipGetDeviceProp() {
  hipDeviceProp_t prop;
  hipError_t err = hipGetDeviceProperties(&prop, 0);

  if (err == hipSuccess) {
    printf("PASSED!\n");
    return 1;
  } else {
    printf("FAILED!\n");
    return 0;
  }
}
