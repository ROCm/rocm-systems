/*
Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.

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
