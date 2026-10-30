/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime.h>

namespace {

__managed__ int managedValue;

__global__ void ReadManagedValue(int* result) { *result = managedValue; }

}  // namespace

extern "C" hipError_t launchLateManagedVariable(int* deviceResult, hipStream_t stream, int value) {
  managedValue = value;
  ReadManagedValue<<<1, 1, 0, stream>>>(deviceResult);
  return hipGetLastError();
}

extern "C" int* lateManagedVariableAddress() { return &managedValue; }

extern "C" hipError_t readLateManagedVariable(int* deviceResult, hipStream_t stream) {
  ReadManagedValue<<<1, 1, 0, stream>>>(deviceResult);
  return hipGetLastError();
}
