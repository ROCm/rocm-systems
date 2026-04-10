// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

/*
An example to explore spill/stack instructions.
Written by Nicholas Curtis [AMD].
*/

#include "common.h"

__global__ void knl(int* out, int filter) {
  int x[1024];
  x[filter] = 0;
  if (threadIdx.x < filter) out[threadIdx.x] = x[threadIdx.x];
}

int main() {
  knl<<<1, 1>>>(nullptr, 0);
  hipCheck(hipDeviceSynchronize());
}
