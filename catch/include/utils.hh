/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include <chrono>
#include <optional>

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>

namespace {
inline constexpr size_t kPageSize = 4096;
}  // anonymous namespace

template <typename T>
void ArrayMismatch(T* const expected, T* const actual, const size_t num_elements) {
  const auto ret = std::mismatch(expected, expected + num_elements, actual);
  if (ret.first != expected + num_elements) {
    const auto idx = std::distance(expected, ret.first);
    INFO("Value mismatch at index: " << idx);
    REQUIRE(expected[idx] == actual[idx]);
  }
}

template <typename It, typename T> void ArrayFindIfNot(It begin, It end, const T expected_value) {
  const auto it = std::find_if_not(
      begin, end, [expected_value](const T elem) { return expected_value == elem; });

  if (it != end) {
    const auto idx = std::distance(begin, it);
    INFO("Value mismatch at index " << idx);
    REQUIRE(expected_value == *it);
  }
}

template <typename T>
void ArrayFindIfNot(T* const array, const T expected_value, const size_t num_elements) {
  ArrayFindIfNot(array, array + num_elements, expected_value);
}

template <typename T, typename F>
static inline void ArrayAllOf(const T* arr, uint32_t count, F value_gen) {
  for (auto i = 0u; i < count; ++i) {
    const std::optional<T> expected_val = value_gen(i);
    if (!expected_val.has_value()) continue;
    // Using require on every iteration leads to a noticeable performance loss on large arrays,
    // even when the require passes.
    if (arr[i] != expected_val.value()) {
      INFO("Mismatch at index: " << i);
      REQUIRE(arr[i] == expected_val.value());
    }
  }
}

template <typename T>
static inline void ArrayInRange(const T* arr, uint32_t count,const T minval,const T maxval) {
  for (auto i = 0u; i < count; ++i) {
    if(arr[i] < minval)
    {
      INFO("Mismatch at index: " << i);
      REQUIRE(arr[i] > minval);
    }
    else if(arr[i] > maxval)
    {
      INFO("Mismatch at index: " << i);
      REQUIRE(arr[i] < maxval);
    }
  }
}


template <typename T, typename F>
void PitchedMemoryVerify(T* const ptr, const size_t pitch, const size_t width, const size_t height,
                         const size_t depth, F expected_value_generator) {
  for (size_t z = 0; z < depth; ++z) {
    for (size_t y = 0; y < height; ++y) {
      for (size_t x = 0; x < width; ++x) {
        const auto slice = reinterpret_cast<uint8_t*>(ptr) + pitch * height * z;
        const auto row = slice + pitch * y;
        if (reinterpret_cast<T*>(row)[x] != expected_value_generator(x, y, z)) {
          INFO("Mismatch at indices: " << x << ", " << y << ", " << z);
          REQUIRE(reinterpret_cast<T*>(row)[x] == expected_value_generator(x, y, z));
        }
      }
    }
  }
}

template <typename T, typename F>
void PitchedMemorySet(T* const ptr, const size_t pitch, const size_t width, const size_t height,
                      const size_t depth, F expected_value_generator) {
  for (size_t z = 0; z < depth; ++z) {
    for (size_t y = 0; y < height; ++y) {
      for (size_t x = 0; x < width; ++x) {
        const auto slice = reinterpret_cast<uint8_t*>(ptr) + pitch * height * z;
        const auto row = slice + pitch * y;
        reinterpret_cast<T*>(row)[x] = expected_value_generator(x, y, z);
      }
    }
  }
}

template <typename T>
__global__ void VectorIncrement(T* const vec, const T increment_value, size_t N) {
  size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = offset; i < N; i += stride) {
    vec[i] += increment_value;
  }
}

template <typename T> __global__ void VectorSet(T* const vec, const T value, size_t N) {
  size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = offset; i < N; i += stride) {
    vec[i] = value;
  }
}

// Will execute for atleast interval milliseconds
static __global__ void Delay(uint32_t interval, const uint32_t ticks_per_ms) {
  while (interval--) {
    #if HT_AMD
    uint64_t start = clock_function();
    while (clock_function() - start < ticks_per_ms) {
      __builtin_amdgcn_s_sleep(10);
    }
    #endif
    #if HT_NVIDIA
    uint64_t start = clock64();
    while (clock64() - start < ticks_per_ms) {
    }
    #endif
  }
}

template <typename T>
__global__ void Iota(T* const out, size_t pitch, size_t w, size_t h, size_t d) {
  const auto x = blockIdx.x * blockDim.x + threadIdx.x;
  const auto y = blockIdx.y * blockDim.y + threadIdx.y;
  const auto z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x < w && y < h && z < d) {
    char* const slice = reinterpret_cast<char*>(out) + pitch * h * z;
    char* const row = slice + pitch * y;
    reinterpret_cast<T*>(row)[x] = z * w * h + y * w + x;
  }
}

inline void LaunchDelayKernel(const std::chrono::milliseconds interval, const hipStream_t stream = nullptr) {
  int ticks_per_ms = 0;
  #if HT_AMD
  HIPCHECK(hipDeviceGetAttribute(&ticks_per_ms, hipDeviceAttributeWallClockRate, 0));
  #endif
  #if HT_NVIDIA
  HIPCHECK(hipDeviceGetAttribute(&ticks_per_ms, hipDeviceAttributeClockRate, 0));
  #endif
  Delay<<<1, 1, 0, stream>>>(interval.count(), ticks_per_ms);
}

template <typename... Attributes>
inline bool DeviceAttributesSupport(const int device, Attributes... attributes) {
  constexpr auto DeviceAttributeSupport = [](const int device,
                                             const hipDeviceAttribute_t attribute) {
    int value = 0;
    HIP_CHECK(hipDeviceGetAttribute(&value, attribute, device));
    return value;
  };
  return (... && DeviceAttributeSupport(device, attributes));
}

inline int GetDeviceAttribute(const hipDeviceAttribute_t attr, int device) {
  int value = 0;
  HIP_CHECK(hipDeviceGetAttribute(&value, attr, device));
  return value;
}
