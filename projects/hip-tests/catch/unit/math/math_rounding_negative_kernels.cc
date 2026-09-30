/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

class Dummy {
 public:
  __device__ Dummy() {}
  __device__ ~Dummy() {}
};

#define NEGATIVE_KERNELS_SHELL(func_name)                                                          \
  __global__ void func_name##_kernel_v1(double* x) { auto result = func_name(x); }                 \
  __global__ void func_name##_kernel_v2(Dummy x) { auto result = func_name(x); }                   \
  __global__ void func_name##f_kernel_v1(float* x) { auto result = func_name##f(x); }              \
  __global__ void func_name##f_kernel_v2(Dummy x) { auto result = func_name##f(x); }

NEGATIVE_KERNELS_SHELL(trunc)
NEGATIVE_KERNELS_SHELL(round)
NEGATIVE_KERNELS_SHELL(rint)
NEGATIVE_KERNELS_SHELL(nearbyint)
NEGATIVE_KERNELS_SHELL(ceil)
NEGATIVE_KERNELS_SHELL(floor)
NEGATIVE_KERNELS_SHELL(lrint)
NEGATIVE_KERNELS_SHELL(lround)
NEGATIVE_KERNELS_SHELL(llrint)
NEGATIVE_KERNELS_SHELL(llround)
