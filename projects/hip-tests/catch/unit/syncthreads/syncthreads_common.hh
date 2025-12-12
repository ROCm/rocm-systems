/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

enum class SyncthreadsKind { kDefault, kCount, kAnd, kOr };

template <SyncthreadsKind kind> __device__ int Syncthreads(int predicate) {
  if constexpr (kind == SyncthreadsKind::kDefault) {
    __syncthreads();
    return 0;
  } else if constexpr (kind == SyncthreadsKind::kCount) {
    return __syncthreads_count(predicate);
  } else if constexpr (kind == SyncthreadsKind::kAnd) {
    return __syncthreads_and(predicate);
  } else if constexpr (kind == SyncthreadsKind::kOr) {
    return __syncthreads_or(predicate);
  }
}

template <SyncthreadsKind kind> __global__ void SyncthreadsKernel(int* out) {
  extern __shared__ int shared_mem[];

  shared_mem[threadIdx.x] = threadIdx.x + 1;

  Syncthreads<kind>(0);

  if (threadIdx.x == 0) {
    int sum = 0;
    for (int i = 0; i < blockDim.x; ++i) {
      sum += shared_mem[i];
    }
    out[blockIdx.x] = sum;
  }
}

template <SyncthreadsKind kind> __global__ void SyncthreadsZeroKernel(int* out) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  out[tid] = Syncthreads<kind>(0);
}

template <SyncthreadsKind kind> __global__ void SyncthreadsOneKernel(int* out) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  out[tid] = Syncthreads<kind>(1);
}

template <SyncthreadsKind kind> __global__ void SyncthreadsOddEvenKernel(int* out) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  out[tid] = Syncthreads<kind>(threadIdx.x % 2);
}

template <SyncthreadsKind kind> __global__ void SyncthreadsNegativeKernel(int* out) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  out[tid] = Syncthreads<kind>(-1);
}

template <SyncthreadsKind kind> __global__ void SyncthreadsIdKernel(int* out) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  out[tid] = Syncthreads<kind>(threadIdx.x);
}