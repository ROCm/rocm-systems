/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "hip/hip_runtime_api.h"
#include <iostream>
int main() {
  hipError_t err;
  err = hipPeekAtLastError();
  if (err == hipSuccess)
    return 1;
  else
    return 0;
}
