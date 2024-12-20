/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#define HIP_ENABLE_WARP_SYNC_BUILTINS

#include <hip_test_common.hh>
#include <hip/hip_cooperative_groups.h>
#include <hip/hip_fp16.h>

#define MASK_SHIFT(x, n) \
  (x & (static_cast<uint64_t>(1) << n)) >> n

const unsigned long long Every5thBit = 0x1084210842108421;
const unsigned long long Every9thBit = 0x8040201008040201;
const unsigned long long Every5thBut9th = Every5thBit & ~Every9thBit;
const unsigned long long AllThreads = ~0;

inline __device__ bool deactivate_thread(const uint64_t* const active_masks) {
  const auto warp =
      cooperative_groups::tiled_partition(cooperative_groups::this_thread_block(), warpSize);
  const auto block = cooperative_groups::this_thread_block();
  const auto warps_per_block = (block.size() + warpSize - 1) / warpSize;
  const auto block_rank = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
  const auto idx = block_rank * warps_per_block + block.thread_rank() / warpSize;
  return !(active_masks[idx] & (static_cast<uint64_t>(1) << warp.thread_rank()));
}

inline std::mt19937& GetRandomGenerator() {
  static std::mt19937 mt(std::random_device{}());
  return mt;
}

template <typename T> inline T GenerateRandomInteger(const T min, const T max) {
  std::uniform_int_distribution<T> dist(min, max);
  return dist(GetRandomGenerator());
}

template <typename T> inline T GenerateRandomReal(const T min, const T max) {
  std::uniform_real_distribution<T> dist(min, max);
  return dist(GetRandomGenerator());
}

inline int generate_width(int warp_size) {
  int exponent = 0;
  while (warp_size >>= 1) {
    ++exponent;
  }

  return GENERATE_COPY(map([](int e) { return 1 << e; }, range(1, exponent + 1)));
}

inline uint64_t get_active_mask(unsigned int warp_id, unsigned int warp_size) {
  uint64_t active_mask = 0;
  switch (warp_id % 5) {
    case 0:  // even threads in the warp
      active_mask = 0xAAAAAAAAAAAAAAAA;
      break;
    case 1:  // odd threads in the warp
      active_mask = 0x5555555555555555;
      break;
    case 2:  // first half of the warp
      for (int i = 0; i < warp_size / 2; i++) {
        active_mask = active_mask | (static_cast<uint64_t>(1) << i);
      }
      break;
    case 3:  // second half of the warp
      for (int i = warp_size / 2; i < warp_size; i++) {
        active_mask = active_mask | (static_cast<uint64_t>(1) << i);
      }
      break;
    case 4:  // all threads
      active_mask = 0xFFFFFFFFFFFFFFFF;
      break;
  }
  return active_mask;
}

template <typename T, std::enable_if_t<std::is_integral<T>::value, bool> = true>
inline T expandPrecision(int X) { return X; }

template <typename T, std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
inline T expandPrecision(int X) {
  return X * 3.141592653589793115997963468544185161590576171875;
}

template <typename T, std::enable_if_t<std::is_same<T, __half>::value, bool> = true>
inline __half expandPrecision(int X) {
  return (__half)expandPrecision<float>(X);
}

template <typename T, std::enable_if_t<std::is_same<T, __half2>::value, bool> = true>
inline __half2 expandPrecision(int X) {
  __half H = expandPrecision<float>(X);
  return {H, H};
}

template <typename T, std::enable_if_t<std::is_integral<T>::value, bool> = true>
inline void expandPrecision(T* Array, int size) {
  (void)Array;
  (void)size;
}

template <typename T, std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
inline void expandPrecision(T *Array, int size) {
  for (int i = 0; i != size; ++i) {
    Array[i] *= 3.141592653589793115997963468544185161590576171875;
  }
}

template <typename T>
inline void initializeInput(T *Input, int size) {
  int Values[] = {0, -1, 2, 3, 4, 5, -6, 7,
                  8, -9, 10, 11, 12, 13, -14, 15,
                  16, 17, -18, 19, 20, -21, 22, 23,
                  24, 25, 26, -27, 28, 29, 30, 31,
                  -32, 33, 34, 35, -36, 37, 38, -39,
                  40, 41, 42, 43, -44, -45, 46, 47,
                  48, 49, 50, -51, 52, 53, -54, 55,
                  56, 57, -58, 59, 60, 61, 62, -63};

  for (int i = 0; i != size; ++i) {
    Input[i] = expandPrecision<T>(Values[i]);
  }
}

template <typename T>
inline void initializeExpected(T *Expected, int *Values, int size) {
  for (int i = 0; i != size; ++i) {
    Expected[i] = expandPrecision<T>(Values[i]);
  }
}

template <typename T>
inline bool compareEqual(T X, T Y) { return X == Y; }

template <>
inline bool compareEqual(__half X, __half Y) {
  return __half2float(X) == __half2float(Y);
}

template <>
inline bool compareEqual(__half2 X, __half2 Y) {
  return compareEqual(X.x, Y.x) && compareEqual(X.y, Y.y);
}

inline bool compareMaskEqual(unsigned long long *Actual, unsigned long long *Expected,
                       int i, int warpSize) {
  if (warpSize == 32)
    return (unsigned)Actual[i] == (unsigned)Expected[i];
  return Actual[i] == Expected[i];
}

template <typename T>
inline T alignUp(T num, size_t n) {
  if (num % n == 0) {
    return num;
  }

  return ((num + n - 1) / n) * n;
}
