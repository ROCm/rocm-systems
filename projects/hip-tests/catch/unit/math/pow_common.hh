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

#define MATH_POW_INT_KERNEL_DEF(func_name)                                                         \
  template <typename T1, typename T2>                                                              \
  __global__ void func_name##_kernel(T1* const ys, const size_t num_xs, T1* const x1s,             \
                                     T2* const x2s) {                                              \
    const auto tid = cg::this_grid().thread_rank();                                                \
    const auto stride = cg::this_grid().size();                                                    \
                                                                                                   \
    for (auto i = tid; i < num_xs; i += stride) {                                                  \
      if constexpr (std::is_same_v<float, T1>) {                                                   \
        ys[i] = func_name##f(x1s[i], x2s[i]);                                                      \
      } else if constexpr (std::is_same_v<double, T1>) {                                           \
        ys[i] = func_name(x1s[i], x2s[i]);                                                         \
      }                                                                                            \
    }                                                                                              \
  }

template <typename T1, typename T2>
using kernel_pow_int_sig = void (*)(T1*, const size_t, T1*, T2*);

template <typename T1, typename T2> using ref_pow_int_sig = T1 (*)(T1, T2);

template <typename T1, typename T2, typename RT1, typename RT2, typename ValidatorBuilder>
void PowIntFloatingPointBruteForceTest(kernel_pow_int_sig<T1, T2> kernel,
                                       ref_pow_int_sig<RT1, RT2> ref_func,
                                       const ValidatorBuilder& validator_builder) {
  const auto [grid_size, block_size] = GetOccupancyMaxPotentialBlockSize(kernel);
  const uint64_t num_iterations = GetTestIterationCount();
  const auto max_batch_size =
      std::min(GetMaxAllowedDeviceMemoryUsage() / (sizeof(T1) * 2 + sizeof(T2)), num_iterations);
  LinearAllocGuard<T1> x1s{LinearAllocs::hipHostMalloc, max_batch_size * sizeof(T1)};
  LinearAllocGuard<T2> x2s{LinearAllocs::hipHostMalloc, max_batch_size * sizeof(T2)};

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
      thread_pool.Post([=, &x1s, &x2s] {
        const auto generator1 = [=] {
          static thread_local std::mt19937 rng(std::random_device{}());
          std::uniform_real_distribution<RefType_t<T1>> unif_dist(std::numeric_limits<T1>::lowest(),
                                                                  std::numeric_limits<T1>::max());
          return static_cast<T1>(unif_dist(rng));
        };
        const auto generator2 = [] {
          static thread_local std::mt19937 rng(std::random_device{}());
          std::uniform_int_distribution<T2> unif_dist(std::numeric_limits<T2>::lowest(),
                                                      std::numeric_limits<T2>::max());
          return unif_dist(rng);
        };
        std::generate(x1s.ptr() + base_idx, x1s.ptr() + base_idx + sub_batch_size, generator1);
        std::generate(x2s.ptr() + base_idx, x2s.ptr() + base_idx + sub_batch_size, generator2);
      });
      base_idx += sub_batch_size;
    }

    thread_pool.Wait();

    math_test.Run(validator_builder, grid_size, block_size, ref_func, batch_size, x1s.ptr(),
                  x2s.ptr());
  }
}

template <typename T1, typename T2, typename RT1, typename RT2, typename ValidatorBuilder>
void PowIntFloatingPointSpecialValuesTest(kernel_pow_int_sig<T1, T2> kernel,
                                          ref_pow_int_sig<RT1, RT2> ref_func,
                                          const ValidatorBuilder& validator_builder) {
  const auto [grid_size, block_size] = GetOccupancyMaxPotentialBlockSize(kernel);
  const auto values1 = std::get<SpecialVals<T1>>(kSpecialValRegistry);
  const auto values2 = std::get<SpecialVals<int>>(kSpecialValRegistry);

  const auto size = values1.size * values2.size;
  LinearAllocGuard<T1> x1s{LinearAllocs::hipHostMalloc, size * sizeof(T1)};
  LinearAllocGuard<T2> x2s{LinearAllocs::hipHostMalloc, size * sizeof(T2)};

  for (auto i = 0u; i < values1.size; ++i) {
    for (auto j = 0u; j < values2.size; ++j) {
      x1s.ptr()[i * values2.size + j] = values1.data[i];
      x2s.ptr()[i * values2.size + j] = static_cast<T2>(values2.data[j]);
    }
  }

  MathTest math_test(kernel, size);
  math_test.template Run<false>(validator_builder, grid_size, block_size, ref_func, size, x1s.ptr(),
                                x2s.ptr());
}

template <typename T1, typename T2, typename RT1, typename RT2, typename ValidatorBuilder>
void PowIntFloatingPointTest(kernel_pow_int_sig<T1, T2> kernel, ref_pow_int_sig<RT1, RT2> ref_func,
                             const ValidatorBuilder& validator_builder) {
  SECTION("Special values") {
    PowIntFloatingPointSpecialValuesTest(kernel, ref_func, validator_builder);
  }

  SECTION("Brute force") { PowIntFloatingPointBruteForceTest(kernel, ref_func, validator_builder); }
}