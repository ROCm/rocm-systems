/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime_api.h>
#include <iostream>
#include "hipMalloc.h"

int MallocFunc() {
  int* Ad;
  hipError_t err;
  err = hipMalloc(reinterpret_cast<void**>(&Ad), 1024);
  if (err == hipSuccess) {
    std::cout << "hipMalloc PASSED!" << std::endl;
    err = hipFree(Ad);
    if (err == hipSuccess) {
      std::cout << "hipFree PASSED!" << std::endl;
      return 1;
    } else {
      std::cout << "hipFree FAILED!" << std::endl;
      return 0;
    }
  } else {
    std::cout << "hipMalloc FAILED!" << std::endl;
    return 0;
  }
}
