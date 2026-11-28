/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime.h>

#include "hipFuncGetAttributesConstantFixtures.hh"

namespace {
// The host symbol-size query keeps this extern-declared definition as real device storage.
extern __constant__ unsigned char first_constant[17];
__constant__ unsigned char first_constant[17];
// The host-side symbol-size query below makes this inline definition registered.
inline __constant__ unsigned char second_constant[29];

__device__ const unsigned int readonly_values[3] = {19, 23, 29};
__device__ unsigned int writable_values[5];

__global__ void multiple_constants_kernel() {}

__global__ void multiple_constants_reader_kernel(unsigned int* output, unsigned int index) {
  const char* literal = "read-only literals are excluded";
  output[0] = first_constant[index % 17] + second_constant[index % 29];
  output[1] = readonly_values[index % 3] + writable_values[index % 5];
  output[2] = literal[index % 29];
}
}  // namespace

const void* ConstantAttributeTest::MultipleConstantsKernel() {
  return reinterpret_cast<const void*>(multiple_constants_kernel);
}

const void* ConstantAttributeTest::MultipleConstantsReaderKernel() {
  return reinterpret_cast<const void*>(multiple_constants_reader_kernel);
}

hipError_t ConstantAttributeTest::GetFirstConstantSize(size_t* size) {
  return hipGetSymbolSize(size, HIP_SYMBOL(first_constant));
}

hipError_t ConstantAttributeTest::GetInlineConstantSize(size_t* size) {
  return hipGetSymbolSize(size, HIP_SYMBOL(second_constant));
}
