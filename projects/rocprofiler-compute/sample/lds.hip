// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

/*
An example to explore LDS bandwidth and bank conflicts, written by Nicholas Curtis [AMD].
*/

#include "common.h"

constexpr unsigned max_threads = 256;
constexpr unsigned nbanks = 32;

__global__ void load(int* out, int flag) {
  __shared__ int array[max_threads];
  int index = threadIdx.x;
  // fake a store to the LDS array to avoid unwanted behavior
  if (flag)
    array[max_threads - index] = index;
  __syncthreads();
  int x = array[index];
  if (x == int(-1234567))
    out[threadIdx.x] = x;
}

__global__ void conflicts(int* out, int flag) {
  constexpr unsigned nelements = nbanks * max_threads;
  __shared__ int array[nelements];
  // each thread reads from the same bank
  int index = threadIdx.x * nbanks;
  // fake a store to the LDS array to avoid unwanted behavior
  if (flag)
    array[max_threads - index] = index;
  __syncthreads();
  int x = array[index];
  if (x == int(-1234567))
    out[threadIdx.x] = x;
}

void bandwidth_demo(int N) {
  for (int i = 1; i <= N; ++i)
    load<<<1,i>>>(nullptr, 0);
  hipCheck(hipDeviceSynchronize());
}

void conflicts_demo(int N) {
  for (int i = 1; i <= N; ++i)
    conflicts<<<1,i>>>(nullptr, 0);
  hipCheck(hipDeviceSynchronize());
}

int main() {
  bandwidth_demo(max_threads);
  conflicts_demo(max_threads);
}
