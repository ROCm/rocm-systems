/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
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