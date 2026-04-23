/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

__launch_bounds__(0) __global__ void MaxThreadsZero(int* sum) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  atomicAdd(sum, tid);
}

__launch_bounds__(1.5) __global__ void MaxThreadsNotInt(int* sum) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  atomicAdd(sum, tid);
}

__launch_bounds__(128, 1.5) __global__ void MinWarpsNotInt(int* sum) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  atomicAdd(sum, tid);
}
