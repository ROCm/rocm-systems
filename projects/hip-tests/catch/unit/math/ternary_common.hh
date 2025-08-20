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

#define MATH_TERNARY_KERNEL_DEF(func_name)                                                         \
  template <typename T>                                                                            \
  __global__ void func_name##_kernel(T* const ys, const size_t num_xs, T* const x1s, T* const x2s, \
                                     T* const x3s) {                                               \
    const auto tid = cg::this_grid().thread_rank();                                                \
    const auto stride = cg::this_grid().size();                                                    \
                                                                                                   \
    for (auto i = tid; i < num_xs; i += stride) {                                                  \
      if constexpr (std::is_same_v<float, T>) {                                                    \
        ys[i] = func_name##f(x1s[i], x2s[i], x3s[i]);                                              \
      } else if constexpr (std::is_same_v<double, T>) {                                            \
        ys[i] = func_name(x1s[i], x2s[i], x3s[i]);                                                 \
      }                                                                                            \
    }                                                                                              \
  }

template <typename T, typename TArg, typename RT, typename RTArg, typename ValidatorBuilder>
void TernaryFloatingPointBruteForceTest(kernel_sig<T, TArg, TArg, TArg> kernel,
                                       ref_sig<RT, RTArg, RTArg, RTArg> ref_func,
                                       const ValidatorBuilder& validator_builder,
                                       const TArg a = std::numeric_limits<TArg>::lowest(),
                                       const TArg b = std::numeric_limits<TArg>::max()) {
  const auto [grid_size, block_size] = GetOccupancyMaxPotentialBlockSize(kernel);
  const uint64_t num_iterations = GetTestIterationCount();
  const auto max_batch_size =
      std::min(GetMaxAllowedDeviceMemoryUsage() / (sizeof(TArg) * 3 + sizeof(T)), num_iterations);
  LinearAllocGuard<TArg> x1s{LinearAllocs::hipHostMalloc, max_batch_size * sizeof(TArg)};
  LinearAllocGuard<TArg> x2s{LinearAllocs::hipHostMalloc, max_batch_size * sizeof(TArg)};
  LinearAllocGuard<TArg> x3s{LinearAllocs::hipHostMalloc, max_batch_size * sizeof(TArg)};

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
      thread_pool.Post([=, &x1s, &x2s, &x3s] {
        const auto generator = [=] {
          static thread_local std::mt19937 rng(std::random_device{}());
          if constexpr (std::is_same_v<TArg, Float16>) {
            std::uniform_real_distribution<RefType_t<Float16>> unif_dist(-FLOAT16_MAX, FLOAT16_MAX);
            return static_cast<Float16>(unif_dist(rng));
          } else {
            std::uniform_real_distribution<RefType_t<TArg>> unif_dist(a, b);
            return static_cast<TArg>(unif_dist(rng));
          }
        };
        std::generate(x1s.ptr() + base_idx, x1s.ptr() + base_idx + sub_batch_size, generator);
        std::generate(x2s.ptr() + base_idx, x2s.ptr() + base_idx + sub_batch_size, generator);
        std::generate(x3s.ptr() + base_idx, x3s.ptr() + base_idx + sub_batch_size, generator);
      });
      base_idx += sub_batch_size;
    }

    thread_pool.Wait();

    math_test.Run(validator_builder, grid_size, block_size, ref_func, batch_size, x1s.ptr(),
                  x2s.ptr(), x3s.ptr());
  }
}

template <typename T, typename TArg, typename RT, typename RTArg, typename ValidatorBuilder>
void TernaryFloatingPointSpecialValuesTest(kernel_sig<T, TArg, TArg, TArg> kernel,
                                           ref_sig<RT, RTArg, RTArg, RTArg> ref_func,
                                           const ValidatorBuilder& validator_builder) {
  const auto [grid_size, block_size] = GetOccupancyMaxPotentialBlockSize(kernel);
  using SpecialValsType = std::conditional_t<std::is_same_v<TArg, Float16>, float, TArg>;
  const auto values = std::get<SpecialVals<SpecialValsType>>(kSpecialValRegistry);

  const auto size = values.size * values.size * values.size;
  LinearAllocGuard<TArg> x1s{LinearAllocs::hipHostMalloc, size * sizeof(TArg)};
  LinearAllocGuard<TArg> x2s{LinearAllocs::hipHostMalloc, size * sizeof(TArg)};
  LinearAllocGuard<TArg> x3s{LinearAllocs::hipHostMalloc, size * sizeof(TArg)};

  for (auto i = 0u; i < values.size; ++i) {
    for (auto j = 0u; j < values.size; ++j) {
      for (auto k = 0u; k < values.size; ++k) {
        x1s.ptr()[(i * values.size + j) * values.size + k] = values.data[i];
        x2s.ptr()[(i * values.size + j) * values.size + k] = values.data[j];
        x3s.ptr()[(i * values.size + j) * values.size + k] = values.data[k];
      }
    }
  }

  MathTest math_test(kernel, size);
  math_test.template Run<false>(validator_builder, grid_size, block_size, ref_func, size, x1s.ptr(),
                                x2s.ptr(), x3s.ptr());
}

template <typename T, typename TArg, typename RT, typename RTArg, typename ValidatorBuilder>
void TernaryFloatingPointTest(kernel_sig<T, TArg, TArg, TArg> kernel,
                              ref_sig<RT, RTArg, RTArg, RTArg> ref_func,
                              const ValidatorBuilder& validator_builder) {
  SECTION("Special values") {
    TernaryFloatingPointSpecialValuesTest(kernel, ref_func, validator_builder);
  }

  SECTION("Brute force") {
    TernaryFloatingPointBruteForceTest(kernel, ref_func, validator_builder);
  }
}


#define MATH_TERNARY_WITHIN_ULP_TEST_DEF(kern_name, ref_func, sp_ulp, dp_ulp)                      \
  MATH_TERNARY_KERNEL_DEF(kern_name)                                                               \
                                                                                                   \
  TEMPLATE_TEST_CASE("Unit_Device_" #kern_name "_Accuracy_Positive", "", float, double) {          \
    using RT = RefType_t<TestType>;                                                                \
    RT (*ref)(RT, RT, RT) = ref_func;                                                              \
    const auto ulp = std::is_same_v<float, TestType> ? sp_ulp : dp_ulp;                            \
                                                                                                   \
    TernaryFloatingPointTest(kern_name##_kernel<TestType>, ref,                                    \
                             ULPValidatorBuilderFactory<TestType>(ulp));                           \
  }
