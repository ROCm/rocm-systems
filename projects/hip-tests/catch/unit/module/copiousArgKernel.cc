/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime.h>

extern "C" __global__ void kernelMultipleArgsSaxpy(int a1, int a2, int* x1, int b1, int b2, int* x2,
                                                   int c1, int c2, int* x3, int d1, int d2, int* x4,
                                                   int e1, int e2, int* x5, int f1, int f2,
                                                   int* x6) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  x1[id] = a1 * x1[id] + a2;
  x2[id] = b1 * x2[id] + b2;
  x3[id] = c1 * x3[id] + c2;
  x4[id] = d1 * x4[id] + d2;
  x5[id] = e1 * x5[id] + e2;
  x6[id] = f1 * x6[id] + f2;
}
