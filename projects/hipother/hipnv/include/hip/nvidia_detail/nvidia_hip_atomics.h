/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef HIP_INCLUDE_HIP_NVIDIA_DETAIL_HIP_ATOMICS_H
#define HIP_INCLUDE_HIP_NVIDIA_DETAIL_HIP_ATOMICS_H

__device__ inline float atomicMax(float* addr, float val) {
  int ret = __float_as_int(*addr);
  while (val > __int_as_float(ret)) {
    int old = ret;
    if ((ret = atomicCAS((int*)addr, old, __float_as_int(val))) == old) break;
  }
  return __int_as_float(ret);
}
__device__ inline double atomicMax(double* addr, double val) {
  unsigned long long ret = __double_as_longlong(*addr);
  while (val > __longlong_as_double(ret)) {
    unsigned long long old = ret;
    if ((ret = atomicCAS((unsigned long long*)addr, old, __double_as_longlong(val))) == old) break;
  }
  return __longlong_as_double(ret);
}

__device__ inline float atomicMin(float* addr, float val) {
  int ret = __float_as_int(*addr);
  while (val < __int_as_float(ret)) {
    int old = ret;
    if ((ret = atomicCAS((int*)addr, old, __float_as_int(val))) == old) break;
  }
  return __int_as_float(ret);
}

__device__ inline double atomicMin(double* addr, double val) {
  unsigned long long ret = __double_as_longlong(*addr);
  while (val < __longlong_as_double(ret)) {
    unsigned long long old = ret;
    if ((ret = atomicCAS((unsigned long long*)addr, old, __double_as_longlong(val))) == old) break;
  }
  return __longlong_as_double(ret);
}

#endif
