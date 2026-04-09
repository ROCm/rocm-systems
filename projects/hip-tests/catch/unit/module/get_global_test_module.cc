/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime.h>
#include "hip/hip_runtime_api.h"

#include "hipModuleGetGlobal.hh"

#define HIP_MODULE_GET_GLOBAL_TEST_DEFINE_DEVICE_GLOBALS(type)                                     \
  __device__ type type##_var = 0;                                                                  \
  __device__ type type##_arr[kArraySize] = {};                                                     \
  extern "C" {                                                                                     \
  __global__ void type##_var_address_validation_kernel(void* ptr, bool* out) {                     \
    *out = static_cast<void*>(&type##_var) == ptr;                                                 \
  }                                                                                                \
  __global__ void type##_arr_address_validation_kernel(void* ptr, bool* out) {                     \
    *out = static_cast<void*>(type##_arr) == ptr;                                                  \
  }                                                                                                \
  }

HIP_MODULE_GET_GLOBAL_TEST_DEFINE_DEVICE_GLOBALS(int)
HIP_MODULE_GET_GLOBAL_TEST_DEFINE_DEVICE_GLOBALS(float)
HIP_MODULE_GET_GLOBAL_TEST_DEFINE_DEVICE_GLOBALS(char)
HIP_MODULE_GET_GLOBAL_TEST_DEFINE_DEVICE_GLOBALS(double)