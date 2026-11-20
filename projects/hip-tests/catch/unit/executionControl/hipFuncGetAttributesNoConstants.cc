/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime.h>

#include "hipFuncGetAttributesConstantFixtures.hh"

namespace {
__device__ const unsigned int readonly_values[3] = {11, 13, 17};
__device__ unsigned int writable_values[5];

__global__ void no_constants_kernel(unsigned int* output, unsigned int index) {
  const char* literal = "ordinary read-only storage is not a registered constant allocation";
  output[0] = readonly_values[index % 3] + writable_values[index % 5];
  output[1] = literal[index % 63];
}
}  // namespace

const void* ConstantAttributeTest::NoConstantsKernel() {
  return reinterpret_cast<const void*>(no_constants_kernel);
}
