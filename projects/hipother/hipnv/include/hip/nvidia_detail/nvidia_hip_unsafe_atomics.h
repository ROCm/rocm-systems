/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef HIP_INCLUDE_HIP_NVIDIA_DETAIL_HIP_UNSAFE_ATOMICS_H
#define HIP_INCLUDE_HIP_NVIDIA_DETAIL_HIP_UNSAFE_ATOMICS_H

__device__ inline float unsafeAtomicAdd(float* addr, float value) { return atomicAdd(addr, value); }

__device__ inline double unsafeAtomicAdd(double* addr, double value) {
#if __CUDA_ARCH__ < 600
  unsigned long long* addr_cast = (unsigned long long*)addr;
  unsigned long long old_val = *addr_cast;
  unsigned long long expected;
  do {
    expected = old_val;
    old_val = atomicCAS(addr_cast, expected,
                        __double_as_longlong(value + __longlong_as_double(expected)));
  } while (__double_as_longlong(expected) != __double_as_longlong(old_val));
  return old_val;
#else
  return atomicAdd(addr, value);
#endif
}

__device__ inline float unsafeAtomicMax(float* addr, float value) { return atomicMax(addr, value); }

__device__ inline double unsafeAtomicMax(double* addr, double val) { return atomicMax(addr, val); }

__device__ inline float unsafeAtomicMin(float* addr, float value) { return atomicMin(addr, value); }

__device__ inline double unsafeAtomicMin(double* addr, double val) { return atomicMin(addr, val); }

__device__ inline float safeAtomicAdd(float* addr, float value) { return atomicAdd(addr, value); }

__device__ inline double safeAtomicAdd(double* addr, double value) {
#if __CUDA_ARCH__ < 600
  unsigned long long* addr_cast = (unsigned long long*)addr;
  unsigned long long old_val = *addr_cast;
  unsigned long long expected;
  do {
    expected = old_val;
    old_val = atomicCAS(addr_cast, expected,
                        __double_as_longlong(value + __longlong_as_double(expected)));
  } while (__double_as_longlong(expected) != __double_as_longlong(old_val));
  return old_val;
#else
  return atomicAdd(addr, value);
#endif
}

__device__ inline float safeAtomicMax(float* addr, float value) { return atomicMax(addr, value); }

__device__ inline double safeAtomicMax(double* addr, double val) { return atomicMax(addr, val); }

__device__ inline float safeAtomicMin(float* addr, float value) { return atomicMin(addr, value); }

__device__ inline double safeAtomicMin(double* addr, double val) { return atomicMin(addr, val); }

#endif
