/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#ifdef __cplusplus

// __scoped_atomic_* require integer/pointer objects. HIP's __hip_atomic_*
// accepted float/double; bitcast through the same-width unsigned type.
__device__ inline unsigned int __hip_fp_bits(float x) {
  union {
    float f;
    unsigned int u;
  } v{x};
  return v.u;
}
__device__ inline float __hip_bits_fp(unsigned int x) {
  union {
    unsigned int u;
    float f;
  } v{x};
  return v.f;
}
__device__ inline unsigned long long __hip_fp_bits(double x) {
  union {
    double f;
    unsigned long long u;
  } v{x};
  return v.u;
}
__device__ inline double __hip_bits_fp(unsigned long long x) {
  union {
    unsigned long long u;
    double f;
  } v{x};
  return v.f;
}

__device__ inline float __hip_scoped_fp_load(float* addr, int order, int clang_scope) {
  unsigned int bits =
      __scoped_atomic_load_n(reinterpret_cast<unsigned int*>(addr), order, clang_scope);
  return __hip_bits_fp(bits);
}
__device__ inline double __hip_scoped_fp_load(double* addr, int order, int clang_scope) {
  unsigned long long bits =
      __scoped_atomic_load_n(reinterpret_cast<unsigned long long*>(addr), order, clang_scope);
  return __hip_bits_fp(bits);
}
__device__ inline bool __hip_scoped_fp_cmpxchg(float* addr, float* expected, float desired,
                                               int success, int failure, int clang_scope) {
  unsigned int e = __hip_fp_bits(*expected);
  unsigned int d = __hip_fp_bits(desired);
  bool ok = __scoped_atomic_compare_exchange_n(reinterpret_cast<unsigned int*>(addr), &e, d, false,
                                               success, failure, clang_scope);
  *expected = __hip_bits_fp(e);
  return ok;
}
__device__ inline bool __hip_scoped_fp_cmpxchg(double* addr, double* expected, double desired,
                                               int success, int failure, int clang_scope) {
  unsigned long long e = __hip_fp_bits(*expected);
  unsigned long long d = __hip_fp_bits(desired);
  bool ok = __scoped_atomic_compare_exchange_n(reinterpret_cast<unsigned long long*>(addr), &e, d,
                                               false, success, failure, clang_scope);
  *expected = __hip_bits_fp(e);
  return ok;
}
__device__ inline float __hip_scoped_fp_fetch_add(float* addr, float value, int order,
                                                  int clang_scope) {
  unsigned int* bits_addr = reinterpret_cast<unsigned int*>(addr);
  unsigned int old = __scoped_atomic_load_n(bits_addr, order, clang_scope);
  unsigned int next;
  do {
    next = __hip_fp_bits(__hip_bits_fp(old) + value);
  } while (!__scoped_atomic_compare_exchange_n(bits_addr, &old, next, false, order, order,
                                               clang_scope));
  return __hip_bits_fp(old);
}
__device__ inline double __hip_scoped_fp_fetch_add(double* addr, double value, int order,
                                                   int clang_scope) {
  unsigned long long* bits_addr = reinterpret_cast<unsigned long long*>(addr);
  unsigned long long old = __scoped_atomic_load_n(bits_addr, order, clang_scope);
  unsigned long long next;
  do {
    next = __hip_fp_bits(__hip_bits_fp(old) + value);
  } while (!__scoped_atomic_compare_exchange_n(bits_addr, &old, next, false, order, order,
                                               clang_scope));
  return __hip_bits_fp(old);
}
__device__ inline float __hip_scoped_fp_exchange(float* addr, float value, int order,
                                                 int clang_scope) {
  unsigned int bits = __scoped_atomic_exchange_n(reinterpret_cast<unsigned int*>(addr),
                                                 __hip_fp_bits(value), order, clang_scope);
  return __hip_bits_fp(bits);
}
__device__ inline double __hip_scoped_fp_exchange(double* addr, double value, int order,
                                                  int clang_scope) {
  unsigned long long bits = __scoped_atomic_exchange_n(
      reinterpret_cast<unsigned long long*>(addr), __hip_fp_bits(value), order, clang_scope);
  return __hip_bits_fp(bits);
}
__device__ inline float __hip_scoped_fp_fetch_min(float* addr, float val, int order,
                                                  int clang_scope) {
  float expected = __hip_scoped_fp_load(addr, order, clang_scope);
  while (val < expected) {
    if (__hip_scoped_fp_cmpxchg(addr, &expected, val, order, order, clang_scope)) break;
  }
  return expected;
}
__device__ inline double __hip_scoped_fp_fetch_min(double* addr, double val, int order,
                                                   int clang_scope) {
  double expected = __hip_scoped_fp_load(addr, order, clang_scope);
  while (val < expected) {
    if (__hip_scoped_fp_cmpxchg(addr, &expected, val, order, order, clang_scope)) break;
  }
  return expected;
}
__device__ inline float __hip_scoped_fp_fetch_max(float* addr, float val, int order,
                                                  int clang_scope) {
  float expected = __hip_scoped_fp_load(addr, order, clang_scope);
  while (val > expected) {
    if (__hip_scoped_fp_cmpxchg(addr, &expected, val, order, order, clang_scope)) break;
  }
  return expected;
}
__device__ inline double __hip_scoped_fp_fetch_max(double* addr, double val, int order,
                                                   int clang_scope) {
  double expected = __hip_scoped_fp_load(addr, order, clang_scope);
  while (val > expected) {
    if (__hip_scoped_fp_cmpxchg(addr, &expected, val, order, order, clang_scope)) break;
  }
  return expected;
}

#endif  // __cplusplus
