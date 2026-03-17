/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <hip/hip_runtime.h>

namespace HipTest {
template <typename T> __global__ void vectorADD(const T* A_d, const T* B_d, T* C_d, size_t NELEM) {
  size_t i = (blockIdx.x * blockDim.x + threadIdx.x);

  if (i < NELEM) {
    C_d[i] = A_d[i] + B_d[i];
  }
}

template <typename T> __global__ void vectorSUB(const T* A_d, const T* B_d, T* C_d, size_t NELEM) {
  size_t i = (blockIdx.x * blockDim.x + threadIdx.x);

  if (i < NELEM) {
    C_d[i] = A_d[i] - B_d[i];
  }
}

template <typename T>
__global__ void vectorADDReverse(const T* A_d, const T* B_d, T* C_d, size_t NELEM) {
  size_t i = (blockIdx.x * blockDim.x + threadIdx.x);

  if (i < NELEM) {
    C_d[i] = A_d[i] + B_d[i];
  }
}


template <typename T> __global__ void addCount(const T* A_d, T* C_d, size_t NELEM, int count) {
  size_t i = (blockIdx.x * blockDim.x + threadIdx.x);

  // Deliberately do this in an inefficient way to increase kernel runtime
  if (i < NELEM) {
    for (int i = 0; i < count; i++) {
      C_d[i] = A_d[i];
      atomicAdd(C_d + i, count);
    }
  }
}


template <typename T>
__global__ void addCountReverse(const T* A_d, T* C_d, int64_t NELEM, int count) {
  int64_t i = (blockIdx.x * blockDim.x + threadIdx.x);

  // Deliberately do this in an inefficient way to increase kernel runtime
  if (i < NELEM) {
    for (int i = 0; i < count; i++) {
      C_d[i] = A_d[i] + (T)count;
    }
  }
}

template <typename T> __global__ void memsetReverse(T* C_d, T val, int64_t NELEM) {
  int64_t i = (blockIdx.x * blockDim.x + threadIdx.x);

  if (i < NELEM) {
    C_d[i] = val;
  }
}

template <typename T> __global__ void vector_square(const T* A_d, T* C_d, size_t N_ELMTS) {
  size_t i = (blockIdx.x * blockDim.x + threadIdx.x);
  if (i < N_ELMTS) {
    C_d[i] = A_d[i] * A_d[i];
  }
}

template <typename T> __global__ void vector_cubic(const T* A_d, T* C_d, size_t N_ELMTS) {
  size_t i = (blockIdx.x * blockDim.x + threadIdx.x);

  if (i < N_ELMTS) {
    C_d[i] = A_d[i] * A_d[i] * A_d[i];
  }
}
}  // namespace HipTest
