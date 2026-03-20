/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

static constexpr int n = 32;

inline void getFactor(int& fact) { fact = 101; }
inline void getFactor(__half& fact) { fact = 2.5; }

template <typename T> inline T sum(T* a) {
  T cpuSum = 0;
  T factor;
  getFactor(factor);
  for (int i = 0; i < n; i++) {
    a[i] = i + factor;
    cpuSum += a[i];
  }
  return cpuSum;
}

template <typename T> inline bool compare(T gpuSum, T cpuSum) {
  if (gpuSum != cpuSum) {
    return true;
  }
  return false;
}

template <> inline __half sum(__half* a) {
  __half cpuSum = 0;
  __half factor;
  getFactor(factor);
  for (int i = 0; i < n; i++) {
    a[i] = i + __half2float(factor);
    cpuSum = __half2float(cpuSum) + __half2float(a[i]);
  }
  return cpuSum;
}

template <> inline bool compare(__half gpuSum, __half cpuSum) {
  if (__half2float(gpuSum) != __half2float(cpuSum)) {
    return true;
  }
  return false;
}
