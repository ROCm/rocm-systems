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

#include <stddef.h>
#include <numeric>

#include <hip/hip_runtime_api.h>
#include <resource_guards.hh>

namespace {
constexpr size_t kArraySize = 5;
}

#define HIP_GRAPH_MEMCPY_FROM_SYMBOL_NODE_DEFINE_GLOBALS(type)                                     \
  __device__ type type##_device_var = 1;                                                           \
  __constant__ __device__ type type##_const_device_var = 1;                                        \
  __device__ type type##_device_arr[kArraySize] = {1, 2, 3, 4, 5};                                 \
  __constant__ __device__ type type##_const_device_arr[kArraySize] = {1, 2, 3, 4, 5};

#define HIP_GRAPH_MEMCPY_FROM_SYMBOL_NODE_DEFINE_ALTERNATE_GLOBALS(type)                           \
  __device__ type type##_alt_device_var = 0;                                                       \
  __constant__ __device__ type type##_alt_const_device_var = 0;                                    \
  __device__ type type##_alt_device_arr[kArraySize] = {0, 0, 0, 0, 0};                             \
  __constant__ __device__ type type##_alt_const_device_arr[kArraySize] = {0, 0, 0, 0, 0};

template <typename T, typename F>
void MemcpyFromSymbolShell(F f, const void* symbol, size_t offset, const std::vector<T> expected) {
  const auto alloc_type = GENERATE(LinearAllocs::hipMalloc, LinearAllocs::hipHostMalloc);
  const auto size = expected.size() * sizeof(T);
  LinearAllocGuard<T> dst_alloc(alloc_type, size);

  hipMemcpyKind direction;
  if (alloc_type == LinearAllocs::hipMalloc) {
    direction = GENERATE(hipMemcpyDeviceToDevice, hipMemcpyDefault);
  } else {
    direction = GENERATE(hipMemcpyDeviceToHost, hipMemcpyDefault);
  }
  INFO("Memcpy direction: " << direction);
  HIP_CHECK(f(dst_alloc.ptr(), symbol, size, offset * sizeof(T), direction));

  std::vector<T> symbol_values(expected.size());
  HIP_CHECK(hipMemcpy(symbol_values.data(), dst_alloc.ptr(), size, hipMemcpyDefault));
  REQUIRE_THAT(expected, Catch::Equals(symbol_values));
}

template <typename T, typename F>
void MemcpyToSymbolShell(F f, const void* symbol, size_t offset, const std::vector<T> set_values) {
  const auto alloc_type = GENERATE(LinearAllocs::hipMalloc, LinearAllocs::hipHostMalloc);
  const auto size = set_values.size() * sizeof(T);
  LinearAllocGuard<T> src_alloc(alloc_type, size);
  HIP_CHECK(hipMemcpy(src_alloc.ptr(), set_values.data(), size, hipMemcpyDefault));

  hipMemcpyKind direction;
  if (alloc_type == LinearAllocs::hipMalloc) {
    direction = GENERATE(hipMemcpyDeviceToDevice, hipMemcpyDefault);
  } else {
    direction = GENERATE(hipMemcpyHostToDevice, hipMemcpyDefault);
  }
  INFO("Memcpy direction: " << direction);
  HIP_CHECK(f(symbol, src_alloc.ptr(), size, offset * sizeof(T), direction));

  std::vector<T> symbol_values(set_values.size());
  HIP_CHECK(hipMemcpyFromSymbol(symbol_values.data(), symbol, size, offset * sizeof(T)));
  REQUIRE_THAT(set_values, Catch::Equals(symbol_values));
}

template <typename F>
void MemcpyFromSymbolCommonNegative(F f, void* dst, const void* symbol, size_t count) {
  SECTION("dst == nullptr") {
    HIP_CHECK_ERROR(f(nullptr, symbol, count, 0, hipMemcpyDefault), hipErrorInvalidValue);
  }

  SECTION("symbol == nullptr") {
    HIP_CHECK_ERROR(f(dst, nullptr, count, 0, hipMemcpyDefault), hipErrorInvalidSymbol);
  }

// Disabled on AMD due to defect - EXSWHTEC-215
#if HT_NVIDIA
  SECTION("count == 0") {
    HIP_CHECK_ERROR(f(dst, symbol, 0, 0, hipMemcpyDefault), hipErrorInvalidValue);
  }
#endif

  SECTION("count > symbol size") {
    HIP_CHECK_ERROR(f(dst, symbol, count + 1, 0, hipMemcpyDefault), hipErrorInvalidValue);
  }

  SECTION("count + offset > symbol size") {
    HIP_CHECK_ERROR(f(dst, symbol, count, 1, hipMemcpyDefault), hipErrorInvalidValue);
  }

// Disabled on AMD due to defect
#if HT_NVIDIA
  SECTION("Illogical memcpy direction") {
    HIP_CHECK_ERROR(f(dst, symbol, count, 0, hipMemcpyHostToDevice),
                    hipErrorInvalidMemcpyDirection);
  }

  SECTION("Invalid memcpy direction") {
    HIP_CHECK_ERROR(f(dst, symbol, count, 0, static_cast<hipMemcpyKind>(-1)),
                    hipErrorInvalidMemcpyDirection);
  }
#endif
}

template <typename F>
void MemcpyToSymbolCommonNegative(F f, const void* symbol, void* src, size_t count) {
  SECTION("src == nullptr") {
    HIP_CHECK_ERROR(f(symbol, nullptr, count, 0, hipMemcpyDefault), hipErrorInvalidValue);
  }

  SECTION("symbol == nullptr") {
    HIP_CHECK_ERROR(f(nullptr, src, count, 0, hipMemcpyDefault), hipErrorInvalidSymbol);
  }

// Disabled on AMD due to defect - EXSWHTEC-215
#if HT_NVIDIA
  SECTION("count == 0") {
    HIP_CHECK_ERROR(f(symbol, src, 0, 0, hipMemcpyDefault), hipErrorInvalidValue);
  }
#endif

  SECTION("count > symbol size") {
    HIP_CHECK_ERROR(f(symbol, src, count + 1, 0, hipMemcpyDefault), hipErrorInvalidValue);
  }

  SECTION("count + offset > symbol size") {
    HIP_CHECK_ERROR(f(symbol, src, count, 1, hipMemcpyDefault), hipErrorInvalidValue);
  }

// Disabled on AMD due to defect
#if HT_NVIDIA
  SECTION("Illogical memcpy direction") {
    HIP_CHECK_ERROR(f(symbol, src, count, 0, hipMemcpyDeviceToHost),
                    hipErrorInvalidMemcpyDirection);
  }

  SECTION("Invalid memcpy direction") {
    HIP_CHECK_ERROR(f(symbol, src, count, 0, static_cast<hipMemcpyKind>(-1)),
                    hipErrorInvalidMemcpyDirection);
  }
#endif
}

#if HT_AMD
#define SYMBOL(expr) &HIP_SYMBOL(expr)
#else
#define SYMBOL(expr) HIP_SYMBOL(expr)
#endif

#define HIP_GRAPH_ADD_MEMCPY_NODE_TO_FROM_SYMBOL_TEST(f, init_val, type)                           \
  SECTION("Scalar variable") { f(SYMBOL(type##_device_var), 0, std::vector<type>{init_val}); }     \
                                                                                                   \
  SECTION("Constant scalar variable") {                                                            \
    f(SYMBOL(type##_const_device_var), 0, std::vector<type>{init_val});                            \
  }                                                                                                \
                                                                                                   \
  SECTION("Array") {                                                                               \
    const auto offset = GENERATE(0, kArraySize / 2);                                               \
    INFO("Array offset: " << offset);                                                              \
    std::vector<type> expected(kArraySize - offset);                                               \
    std::iota(expected.begin(), expected.end(), offset + init_val);                                \
    f(SYMBOL(type##_device_arr), offset, std::move(expected));                                     \
  }                                                                                                \
                                                                                                   \
  SECTION("Constant array") {                                                                      \
    const auto offset = GENERATE(0, kArraySize / 2);                                               \
    INFO("Array offset: " << offset);                                                              \
    std::vector<type> expected(kArraySize - offset);                                               \
    std::iota(expected.begin(), expected.end(), offset + init_val);                                \
    f(SYMBOL(type##_const_device_arr), offset, std::move(expected));                               \
  }

#define HIP_GRAPH_MEMCPY_NODE_SET_PARAMS_TO_FROM_SYMBOL_TEST(f, init_val, type)                    \
  SECTION("Scalar variable") {                                                                     \
    f(SYMBOL(type##_device_var), SYMBOL(type##_alt_device_var), 0, std::vector<type>{init_val});   \
  }                                                                                                \
                                                                                                   \
  SECTION("Constant scalar variable") {                                                            \
    f(SYMBOL(type##_const_device_var), SYMBOL(type##_alt_const_device_var), 0,                     \
      std::vector<type>{init_val});                                                                \
  }                                                                                                \
                                                                                                   \
  SECTION("Array") {                                                                               \
    const auto offset = GENERATE(0, kArraySize / 2);                                               \
    INFO("Array offset: " << offset);                                                              \
    std::vector<type> expected(kArraySize - offset);                                               \
    std::iota(expected.begin(), expected.end(), offset + init_val);                                \
    f(SYMBOL(type##_device_arr), SYMBOL(type##_alt_device_arr), offset, std::move(expected));      \
  }                                                                                                \
                                                                                                   \
  SECTION("Constant array") {                                                                      \
    const auto offset = GENERATE(0, kArraySize / 2);                                               \
    INFO("Array offset: " << offset);                                                              \
    std::vector<type> expected(kArraySize - offset);                                               \
    std::iota(expected.begin(), expected.end(), offset + init_val);                                \
    f(SYMBOL(type##_const_device_arr), SYMBOL(type##_alt_const_device_arr), offset,                \
      std::move(expected));                                                                        \
  }
