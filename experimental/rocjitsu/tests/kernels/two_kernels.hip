// Copyright (c) 2026 Advanced Micro Devices, Inc.
// All rights reserved.

#include <hip/hip_runtime.h>

/// Two kernels in one compilation unit, for testing kernel name selection
/// in metadata parsing.

extern "C" __global__ void add_one(int *__restrict__ data, unsigned n) {
  unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < n) {
    data[gid] += 1;
  }
}

extern "C" __global__ void add_two(int *__restrict__ data, unsigned n, int value) {
  unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < n) {
    data[gid] += value;
  }
}
