// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdlib>
#include <hip/hip_runtime.h>
#include <iostream>

#define HIP_CHECK(expression)                                                                      \
  do {                                                                                             \
    const hipError_t status = (expression);                                                        \
    if (status != hipSuccess) {                                                                    \
      std::cerr << "HIP error " << status << ": " << hipGetErrorString(status) << " at "           \
                << __FILE__ << ":" << __LINE__ << std::endl;                                       \
      std::abort();                                                                                \
    }                                                                                              \
  } while (0)
