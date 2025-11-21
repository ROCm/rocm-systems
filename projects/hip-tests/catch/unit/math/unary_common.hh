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

#include "math_common.hh"
#include "math_special_values.hh"

#include <hip/hip_cooperative_groups.h>

namespace cg = cooperative_groups;

#define MATH_UNARY_KERNEL_DEF(func_name)                                                           \
  template <typename T, typename RT = T>                                                           \
  __global__ void func_name##_kernel(RT* const ys, const size_t num_xs, T* const xs) {             \
    const auto tid = cg::this_grid().thread_rank();                                                \
    const auto stride = cg::this_grid().size();                                                    \
                                                                                                   \
    for (auto i = tid; i < num_xs; i += stride) {                                                  \
      if constexpr (std::is_same_v<float, T>) {                                                    \
        ys[i] = func_name##f(xs[i]);                                                               \
      } else if constexpr (std::is_same_v<double, T>) {                                            \
        ys[i] = func_name(xs[i]);                                                                  \
      }                                                                                            \
    }                                                                                              \
  }

template <typename T, typename RT, typename RTArg, typename ValidatorBuilder>
void UnaryHalfPrecisionBruteForceTest(kernel_sig<T, Float16> kernel, ref_sig<RT, RTArg> ref_func,
                                      const ValidatorBuilder& validator_builder) {
  const auto [grid_size, block_size] = GetOccupancyMaxPotentialBlockSize(kernel);
  uint64_t stop = std::numeric_limits<uint16_t>::max() + 1ul;
  const auto max_batch_size =
      std::min(GetMaxAllowedDeviceMemoryUsage() / (sizeof(Float16) + sizeof(T)), stop);
  LinearAllocGuard<Float16> values{LinearAllocs::hipHostMalloc, max_batch_size * sizeof(Float16)};

  MathTest math_test(kernel, max_batch_size);

  auto batch_size = max_batch_size;
  const auto num_threads = thread_pool.thread_count();

  for (uint64_t v = 0u; v < stop;) {
    batch_size = std::min<uint64_t>(max_batch_size, stop - v);

    const auto min_sub_batch_size = batch_size / num_threads;
    const auto tail = batch_size % num_threads;

    auto base_idx = 0u;
    for (auto i = 0u; i < num_threads; ++i) {
      const auto sub_batch_size = min_sub_batch_size + (i < tail);

      thread_pool.Post([=, &values] {
        auto t = v;
        uint16_t val;
        for (auto j = 0u; j < sub_batch_size; ++j) {
          val = static_cast<uint16_t>(t++);
          values.ptr()[base_idx + j] = *reinterpret_cast<Float16*>(&val);
        }
      });

      v += sub_batch_size;
      base_idx += sub_batch_size;
    }

    thread_pool.Wait();

    math_test.Run(validator_builder, grid_size, block_size, ref_func, batch_size, values.ptr());
  }
}

template <typename T, typename RT, typename RTArg, typename ValidatorBuilder>
void UnarySinglePrecisionBruteForceTest(kernel_sig<T, float> kernel, ref_sig<RT, RTArg> ref_func,
                                        const ValidatorBuilder& validator_builder) {
  const auto [grid_size, block_size] = GetOccupancyMaxPotentialBlockSize(kernel);
  uint64_t stop = std::numeric_limits<uint32_t>::max() + 1ul;
  const auto max_batch_size =
      std::min(GetMaxAllowedDeviceMemoryUsage() / (sizeof(float) + sizeof(T)), stop);
  LinearAllocGuard<float> values{LinearAllocs::hipHostMalloc, max_batch_size * sizeof(float)};

  MathTest math_test(kernel, max_batch_size);

  auto batch_size = max_batch_size;
  const auto num_threads = thread_pool.thread_count();

  for (uint64_t v = 0u; v < stop;) {
    batch_size = std::min<uint64_t>(max_batch_size, stop - v);

    const auto min_sub_batch_size = batch_size / num_threads;
    const auto tail = batch_size % num_threads;

    auto base_idx = 0u;
    for (auto i = 0u; i < num_threads; ++i) {
      const auto sub_batch_size = min_sub_batch_size + (i < tail);

      thread_pool.Post([=, &values] {
        auto t = v;
        uint32_t val;
        for (auto j = 0u; j < sub_batch_size; ++j) {
          val = static_cast<uint32_t>(t++);
          values.ptr()[base_idx + j] = *reinterpret_cast<float*>(&val);
        }
      });

      v += sub_batch_size;
      base_idx += sub_batch_size;
    }

    thread_pool.Wait();

    math_test.Run(validator_builder, grid_size, block_size, ref_func, batch_size, values.ptr());
  }
}

template <typename T, typename RT, typename RTArg, typename ValidatorBuilder>
void UnarySinglePrecisionRangeTest(kernel_sig<T, float> kernel, ref_sig<RT, RTArg> ref_func,
                                   const ValidatorBuilder& validator_builder, const float a,
                                   const float b) {
  const auto [grid_size, block_size] = GetOccupancyMaxPotentialBlockSize(kernel);
  const auto max_batch_size = GetMaxAllowedDeviceMemoryUsage() / (sizeof(float) + sizeof(T));
  LinearAllocGuard<float> values{LinearAllocs::hipHostMalloc, max_batch_size * sizeof(float)};

  MathTest math_test(kernel, max_batch_size);

  size_t inserted = 0u;
  for (float v = a; v != b; v = std::nextafter(v, b)) {
    values.ptr()[inserted++] = v;
    if (inserted < max_batch_size) continue;

    math_test.Run(validator_builder, grid_size, block_size, ref_func, inserted, values.ptr());
    inserted = 0u;
  }
}

template <typename T, typename RT, typename RTArg, typename ValidatorBuilder>
void UnaryDoublePrecisionBruteForceTest(kernel_sig<T, double> kernel, ref_sig<RT, RTArg> ref_func,
                                        const ValidatorBuilder& validator_builder,
                                        const double a = std::numeric_limits<double>::lowest(),
                                        const double b = std::numeric_limits<double>::max()) {
  const auto [grid_size, block_size] = GetOccupancyMaxPotentialBlockSize(kernel);
  const uint64_t num_iterations = GetTestIterationCount();
  const auto max_batch_size =
      std::min(GetMaxAllowedDeviceMemoryUsage() / (sizeof(double) + sizeof(T)), num_iterations);
  LinearAllocGuard<double> values{LinearAllocs::hipHostMalloc, max_batch_size * sizeof(double)};

  MathTest math_test(kernel, max_batch_size);

  auto batch_size = max_batch_size;
  const auto num_threads = thread_pool.thread_count();
  for (uint64_t i = 0ul; i < num_iterations; i += batch_size) {
    batch_size = std::min<uint64_t>(max_batch_size, num_iterations - i);

    const auto min_sub_batch_size = batch_size / num_threads;
    const auto tail = batch_size % num_threads;

    auto base_idx = 0u;
    for (auto i = 0u; i < num_threads; ++i) {
      const auto sub_batch_size = min_sub_batch_size + (i < tail);
      thread_pool.Post([=, &values] {
        const auto generator = [=] {
          static thread_local std::mt19937 rng(std::random_device{}());
          std::uniform_real_distribution<long double> unif_dist(a, b);
          return static_cast<double>(unif_dist(rng));
        };
        std::generate(values.ptr() + base_idx, values.ptr() + base_idx + sub_batch_size, generator);
      });
      base_idx += sub_batch_size;
    }

    thread_pool.Wait();

    math_test.Run(validator_builder, grid_size, block_size, ref_func, batch_size, values.ptr());
  }
}

template <typename T, typename RT, typename RTArg, typename ValidatorBuilder>
void UnaryDoublePrecisionSpecialValuesTest(kernel_sig<T, double> kernel,
                                           ref_sig<RT, RTArg> ref_func,
                                           const ValidatorBuilder& validator_builder) {
  const auto [grid_size, block_size] = GetOccupancyMaxPotentialBlockSize(kernel);
  const auto values = std::get<SpecialVals<double>>(kSpecialValRegistry);

  MathTest math_test(kernel, values.size);
  math_test.template Run<false>(validator_builder, grid_size, block_size, ref_func, values.size,
                                values.data);
}

template <typename T, typename RT, typename RTArg, typename ValidatorBuilder>
void UnaryHalfPrecisionTest(kernel_sig<T, Float16> kernel, ref_sig<RT, RTArg> ref,
                            const ValidatorBuilder& validator_builder) {
  SECTION("Brute force") { UnaryHalfPrecisionBruteForceTest(kernel, ref, validator_builder); }
}

template <typename T, typename RT, typename RTArg, typename ValidatorBuilder>
void UnarySinglePrecisionTest(kernel_sig<T, float> kernel, ref_sig<RT, RTArg> ref,
                              const ValidatorBuilder& validator_builder) {
  SECTION("Brute force") { UnarySinglePrecisionBruteForceTest(kernel, ref, validator_builder); }
}

template <typename T, typename RT, typename RTArg, typename ValidatorBuilder>
void UnaryDoublePrecisionTest(kernel_sig<T, double> kernel, ref_sig<RT, RTArg> ref,
                              const ValidatorBuilder& validator_builder) {
  SECTION("Special values") {
    UnaryDoublePrecisionSpecialValuesTest(kernel, ref, validator_builder);
  }

  SECTION("Brute force") { UnaryDoublePrecisionBruteForceTest(kernel, ref, validator_builder); }
}

#define MATH_UNARY_WITHIN_ULP_TEST_DEF(kern_name, ref_func, sp_ulp, dp_ulp)                        \
  MATH_UNARY_KERNEL_DEF(kern_name)                                                                 \
                                                                                                   \
  TEST_CASE("Unit_Device_" #kern_name "_Accuracy_Positive - float") {                              \
    double (*ref)(double) = ref_func;                                                              \
    UnarySinglePrecisionTest(kern_name##_kernel<float>, ref,                                       \
                             ULPValidatorBuilderFactory<float>(sp_ulp));                           \
  }                                                                                                \
                                                                                                   \
  TEST_CASE("Unit_Device_" #kern_name "_Accuracy_Positive - double") {                             \
    long double (*ref)(long double) = ref_func;                                                    \
    UnaryDoublePrecisionTest(kern_name##_kernel<double>, ref,                                      \
                             ULPValidatorBuilderFactory<double>(dp_ulp));                          \
  }

#define MATH_UNARY_WITHIN_ULP_STL_REF_TEST_DEF(func_name, sp_ulp, dp_ulp)                          \
  MATH_UNARY_WITHIN_ULP_TEST_DEF(func_name, std::func_name, sp_ulp, dp_ulp)
