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

#include "unary_common.hh"
#include <fenv.h>

namespace cg = cooperative_groups;

#define CAST_KERNEL_DEF(func_name, T1, T2)                                                         \
  __global__ void func_name##_kernel(T1* const ys, const size_t num_xs, T2* const xs) {            \
    const auto tid = cg::this_grid().thread_rank();                                                \
    const auto stride = cg::this_grid().size();                                                    \
                                                                                                   \
    for (auto i = tid; i < num_xs; i += stride) {                                                  \
      ys[i] = func_name(xs[i]);                                                                    \
    }                                                                                              \
  }

#define CAST_BINARY_KERNEL_DEF(func_name, T1, T2)                                                  \
  __global__ void func_name##_kernel(T1* const ys, const size_t num_xs, T2* const x1s,             \
                                     T2* const x2s) {                                              \
    const auto tid = cg::this_grid().thread_rank();                                                \
    const auto stride = cg::this_grid().size();                                                    \
                                                                                                   \
    for (auto i = tid; i < num_xs; i += stride) {                                                  \
      ys[i] = func_name(x1s[i], x2s[i]);                                                           \
    }                                                                                              \
  }

#define CAST_F2I_REF_DEF(func_name, T1, T2, ref_func)                                              \
  T1 func_name##_ref(T2 arg) {                                                                     \
    if (arg >= static_cast<T2>(std::numeric_limits<T1>::max()))                                    \
      return std::numeric_limits<T1>::max();                                                       \
    else if (arg <= static_cast<T2>(std::numeric_limits<T1>::min()))                               \
      return std::numeric_limits<T1>::min();                                                       \
    T2 result = ref_func(arg);                                                                     \
    return result;                                                                                 \
  }

#define CAST_F2I_RZ_REF_DEF(func_name, T1, T2)                                                     \
  T1 func_name##_ref(T2 arg) {                                                                     \
    if (arg >= static_cast<double>(std::numeric_limits<T1>::max()))                                \
      return std::numeric_limits<T1>::max();                                                       \
    else if (arg <= static_cast<double>(std::numeric_limits<T1>::min()))                           \
      return std::numeric_limits<T1>::min();                                                       \
    T1 result = static_cast<T1>(arg);                                                              \
    return result;                                                                                 \
  }

#define CAST_RND_REF_DEF(func_name, T1, T2, round_dir)                                             \
  T1 func_name##_ref(T2 arg) {                                                                     \
    int curr_direction = fegetround();                                                             \
    fesetround(round_dir);                                                                         \
    T1 result = static_cast<T1>(arg);                                                              \
    fesetround(curr_direction);                                                                    \
    return result;                                                                                 \
  }

#define CAST_REF_DEF(func_name, T1, T2)                                                            \
  T1 func_name##_ref(T2 arg) {                                                                     \
    T1 result = static_cast<T1>(arg);                                                              \
    return result;                                                                                 \
  }

template <typename T1, typename T2> T1 type2_as_type1_ref(T2 arg) {
  T1 tmp;
  memcpy(&tmp, &arg, sizeof(tmp));
  return tmp;
}

template <typename T, typename RT, typename RTArg, typename ValidatorBuilder>
void CastUnaryHalfPrecisionBruteForceTest(kernel_sig<T, Float16> kernel,
                                          ref_sig<RT, RTArg> ref_func,
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
          if (std::isnan(values.ptr()[base_idx + j]) || std::isinf(values.ptr()[base_idx + j])) {
            values.ptr()[base_idx + j] = 0;
          }
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
void CastUnaryHalfPrecisionTest(kernel_sig<T, Float16> kernel, ref_sig<RT, RTArg> ref,
                                const ValidatorBuilder& validator_builder) {
  SECTION("Brute force") { CastUnaryHalfPrecisionBruteForceTest(kernel, ref, validator_builder); }
}


template <typename T, typename ValidatorBuilder>
void CastDoublePrecisionSpecialValuesTest(kernel_sig<T, double> kernel, ref_sig<T, double> ref_func,
                                          const ValidatorBuilder& validator_builder) {
  const auto [grid_size, block_size] = GetOccupancyMaxPotentialBlockSize(kernel);
  const auto values = std::get<SpecialVals<double>>(kSpecialValRegistry);
  std::vector<double> spec_values;

  if (!std::is_same_v<float, T> && !std::is_same_v<double, T> && !std::is_same_v<long double, T>) {
    for (int i = 0; i < values.size; i++) {
      if (!std::isnan(values.data[i]) && !std::isinf(values.data[i])) {
        spec_values.push_back(values.data[i]);
      }
    }
  }

  MathTest math_test(kernel, spec_values.size());
  math_test.template Run<false>(validator_builder, grid_size, block_size, ref_func,
                                spec_values.size(), spec_values.data());
}

template <typename T, typename ValidatorBuilder>
void CastDoublePrecisionTest(kernel_sig<T, double> kernel, ref_sig<T, double> ref,
                             const ValidatorBuilder& validator_builder) {
  SECTION("Special values") {
    CastDoublePrecisionSpecialValuesTest(kernel, ref, validator_builder);
  }

  SECTION("Brute force") { UnaryDoublePrecisionBruteForceTest(kernel, ref, validator_builder); }
}

template <typename T, typename TArg, typename RT, typename RTArg, typename ValidatorBuilder>
void CastIntRangeTest(kernel_sig<T, TArg> kernel, ref_sig<RT, RTArg> ref_func,
                      const ValidatorBuilder& validator_builder,
                      const TArg a = std::numeric_limits<TArg>::lowest(),
                      const TArg b = std::numeric_limits<TArg>::max()) {
  const auto [grid_size, block_size] = GetOccupancyMaxPotentialBlockSize(kernel);
  const auto max_batch_size = GetMaxAllowedDeviceMemoryUsage() / (sizeof(T) + sizeof(TArg));
  LinearAllocGuard<TArg> values{LinearAllocs::hipHostMalloc, max_batch_size * sizeof(TArg)};

  MathTest math_test(kernel, max_batch_size);

  size_t inserted = 0u;
  for (TArg v = a; v <= b; v++) {
    values.ptr()[inserted++] = v;
    if (inserted < max_batch_size) continue;

    math_test.Run(validator_builder, grid_size, block_size, ref_func, inserted, values.ptr());
    inserted = 0u;
  }
}

template <typename T, typename TArg, typename RT, typename RTArg, typename ValidatorBuilder>
void CastIntBruteForceTest(kernel_sig<T, TArg> kernel, ref_sig<RT, RTArg> ref_func,
                           const ValidatorBuilder& validator_builder,
                           const TArg a = std::numeric_limits<TArg>::lowest(),
                           const TArg b = std::numeric_limits<TArg>::max()) {
  const auto [grid_size, block_size] = GetOccupancyMaxPotentialBlockSize(kernel);
  const uint64_t num_iterations = GetTestIterationCount();
  const auto max_batch_size =
      std::min(GetMaxAllowedDeviceMemoryUsage() / (sizeof(T) + sizeof(TArg)), num_iterations);
  LinearAllocGuard<TArg> values{LinearAllocs::hipHostMalloc, max_batch_size * sizeof(TArg)};

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
          std::uniform_int_distribution<TArg> unif_dist(a, b);
          return static_cast<TArg>(unif_dist(rng));
        };
        std::generate(values.ptr() + base_idx, values.ptr() + base_idx + sub_batch_size, generator);
      });
      base_idx += sub_batch_size;
    }

    thread_pool.Wait();

    math_test.Run(validator_builder, grid_size, block_size, ref_func, batch_size, values.ptr());
  }
}

template <typename T1, typename T2, typename ValidatorBuilder>
void CastBinaryIntRangeTest(kernel_sig<T1, T2, T2> kernel, ref_sig<T1, T2, T2> ref_func,
                            const ValidatorBuilder& validator_builder,
                            const T2 a = std::numeric_limits<T2>::lowest(),
                            const T2 b = std::numeric_limits<T2>::max()) {
  const auto [grid_size, block_size] = GetOccupancyMaxPotentialBlockSize(kernel);
  const auto max_batch_size = GetMaxAllowedDeviceMemoryUsage() / (sizeof(T1) + 2 * sizeof(T2));
  LinearAllocGuard<T2> values1{LinearAllocs::hipHostMalloc, max_batch_size * sizeof(T2)};
  LinearAllocGuard<T2> values2{LinearAllocs::hipHostMalloc, max_batch_size * sizeof(T2)};

  MathTest math_test(kernel, max_batch_size);

  size_t inserted = 0u;
  for (T2 v = a; v <= b; v++) {
    values1.ptr()[inserted] = v;
    values2.ptr()[inserted++] = b - v;
    if (inserted < max_batch_size) continue;

    math_test.Run(validator_builder, grid_size, block_size, ref_func, inserted, values1.ptr(),
                  values2.ptr());
    inserted = 0u;
  }
}
