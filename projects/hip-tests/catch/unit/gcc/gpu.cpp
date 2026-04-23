/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <iostream>
#include <hip/hip_runtime.h>
#include "LaunchKernel.h"

extern "C" {

__global__ void kernel() {}

__global__ void kernel1(int* a) { *a = 333; }

__global__ void kernel2(int* a, int* b) { *a = *b; }

__global__ void kernel3(int* a, int* b, int* c) { *c = *a + *b; }

__global__ void kernel4(int* a, char c, short s, int i, struct things t) {
  *a = c + s + i + t.c + t.s + t.i;
}

const void* funcTable[] = {(const void*)kernel, (const void*)kernel1, (const void*)kernel2,
                           (const void*)kernel3, (const void*)kernel4};

const void* getKernelFunc(enum func f) { return funcTable[f]; }

}  // extern "C"
