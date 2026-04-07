/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

__global__ void StaticAssertErrorKernel1() {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  static_assert(tid % 2 == 1, "[StaticAssertErrorKernel1]");
}

__global__ void StaticAssertErrorKernel2() {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  static_assert(++tid > 2, "[StaticAssertErrorKernel2]");
}
