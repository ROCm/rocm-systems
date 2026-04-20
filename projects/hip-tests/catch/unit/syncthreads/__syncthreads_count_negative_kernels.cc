/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

struct Dummy {
  __device__ Dummy() {}
  __device__ ~Dummy() {}
};

__global__ void __syncthreads_count_v1(int* predicate) {
  int result = __syncthreads_count(predicate);
}

__global__ void __syncthreads_count_v2(Dummy predicate) {
  int result = __syncthreads_count(predicate);
}