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

#define MATH_QUATERNARY_KERNEL_DEF(func_name)                                                      \
  template <typename T>                                                                            \
  __global__ void func_name##_kernel(T* const ys, const size_t num_xs, T* const x1s, T* const x2s, \
                                     T* const x3s, T* const x4s) {                                 \
    const auto tid = cg::this_grid().thread_rank();                                                \
    const auto stride = cg::this_grid().size();                                                    \
                                                                                                   \
    for (auto i = tid; i < num_xs; i += stride) {                                                  \
      if constexpr (std::is_same_v<float, T>) {                                                    \
        ys[i] = func_name##f(x1s[i], x2s[i], x3s[i], x4s[i]);                                      \
      } else if constexpr (std::is_same_v<double, T>) {                                            \
        ys[i] = func_name(x1s[i], x2s[i], x3s[i], x4s[i]);                                         \
      }                                                                                            \
    }                                                                                              \
  }

inline constexpr std::array kSpecialValuesReducedDouble{
    -std::numeric_limits<double>::quiet_NaN(),
    -std::numeric_limits<double>::infinity(),
    -std::numeric_limits<double>::max(),
    HEX_DBL(-, 1, 0000000000001, +, 64),
    HEX_DBL(-, 1, fffffffffffff, +, 63),
    HEX_DBL(-, 1, fffffffffffff, +, 62),
    HEX_DBL(-, 1, 0, +, 32),
    HEX_DBL(-, 1, 0000000000001, +, 31),
    HEX_DBL(-, 1, fffffffffffff, +, 30),
    -1000.0,
    -3.5,
    HEX_DBL(-, 1, 8000000000001, +, 1),
    -2.5,
    HEX_DBL(-, 1, 8000000000001, +, 0),
    -1.5,
    -0.5,
    -0.25,
    HEX_DBL(-, 1, fffffffffffff, -, 3),
    -std::numeric_limits<double>::min(),
    HEX_DBL(-, 0, fffffffffffff, -, 1022),
    HEX_DBL(-, 0, 0000000000001, -, 1022),
    -0.0,

    std::numeric_limits<double>::quiet_NaN(),
    std::numeric_limits<double>::infinity(),
    std::numeric_limits<double>::max(),
    HEX_DBL(+, 1, 0, +, 64),
    HEX_DBL(+, 1, 0000000000001, +, 63),
    HEX_DBL(+, 1, 000002, +, 32),
    HEX_DBL(+, 1, fffffffffffff, +, 31),
    HEX_DBL(+, 1, 0, +, 31),
    HEX_DBL(+, 1, fffffffffffff, +, 30),
    +100.0,
    +3.0,
    HEX_DBL(+, 1, 7ffffffffffff, +, 1),
    +2.0,
    HEX_DBL(+, 1, 7ffffffffffff, +, 0),
    +1.0,
    HEX_DBL(+, 1, fffffffffffff, -, 2),
    +std::numeric_limits<double>::min(),
    HEX_DBL(+, 0, 0000000000fff, -, 1022),
    HEX_DBL(+, 0, 0000000000007, -, 1022),
    +0.0,
};

inline constexpr std::array kSpecialValuesReducedFloat{
    -std::numeric_limits<float>::quiet_NaN(),
    -std::numeric_limits<float>::infinity(),
    -std::numeric_limits<float>::max(),
    HEX_FLT(-, 1, 000002, +, 64),
    HEX_FLT(-, 1, fffffe, +, 63),
    HEX_FLT(-, 1, fffffe, +, 62),
    HEX_FLT(-, 1, 0, +, 32),
    HEX_FLT(-, 1, fffffe, +, 31),
    HEX_FLT(-, 1, fffffe, +, 30),
    -1000.f,
    -3.5f,
    HEX_FLT(-, 1, 800002, +, 1),
    -2.5f,
    HEX_FLT(-, 1, 800002, +, 0),
    -1.5f,
    -0.5f,
    -0.25f,
    HEX_FLT(-, 1, fffffe, -, 3),
    -std::numeric_limits<float>::min(),
    HEX_FLT(-, 0, fffffe, -, 126),
    HEX_FLT(-, 0, 000002, -, 126),
    -0.0f,

    std::numeric_limits<float>::quiet_NaN(),
    std::numeric_limits<float>::infinity(),
    std::numeric_limits<float>::max(),
    HEX_FLT(+, 1, 0, +, 64),
    HEX_FLT(+, 1, 000002, +, 63),
    HEX_FLT(+, 1, 000002, +, 32),
    HEX_FLT(+, 1, 000002, +, 31),
    HEX_FLT(+, 1, fffffe, +, 30),
    +100.f,
    +4.0f,
    HEX_FLT(+, 1, 7ffffe, +, 1),
    +2.0f,
    HEX_FLT(+, 1, 7ffffe, +, 0),
    +1.0f,
    HEX_FLT(+, 1, fffffe, -, 2),
    +std::numeric_limits<float>::min(),
    HEX_FLT(+, 0, 000ffe, -, 126),
    HEX_FLT(+, 0, 000006, -, 126),
    +0.0f,
};

inline constexpr auto kSpecialValReducedRegistry = std::make_tuple(
    SpecialVals<float>{kSpecialValuesReducedFloat.data(), kSpecialValuesReducedFloat.size()},
    SpecialVals<double>{kSpecialValuesReducedDouble.data(), kSpecialValuesReducedDouble.size()});

template <typename T, typename TArg, typename RT, typename RTArg, typename ValidatorBuilder>
void QuaternaryFloatingPointBruteForceTest(kernel_sig<T, TArg, TArg, TArg, TArg> kernel,
                                           ref_sig<RT, RTArg, RTArg, RTArg, RTArg> ref_func,
                                           const ValidatorBuilder& validator_builder,
                                           const TArg a = std::numeric_limits<TArg>::lowest(),
                                           const TArg b = std::numeric_limits<TArg>::max()) {
  const auto [grid_size, block_size] = GetOccupancyMaxPotentialBlockSize(kernel);
  const uint64_t num_iterations = GetTestIterationCount();
  const auto max_batch_size =
      std::min(GetMaxAllowedDeviceMemoryUsage() / (sizeof(TArg) * 4 + sizeof(T)), num_iterations);
  LinearAllocGuard<TArg> x1s{LinearAllocs::hipHostMalloc, max_batch_size * sizeof(TArg)};
  LinearAllocGuard<TArg> x2s{LinearAllocs::hipHostMalloc, max_batch_size * sizeof(TArg)};
  LinearAllocGuard<TArg> x3s{LinearAllocs::hipHostMalloc, max_batch_size * sizeof(TArg)};
  LinearAllocGuard<TArg> x4s{LinearAllocs::hipHostMalloc, max_batch_size * sizeof(TArg)};

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
      thread_pool.Post([=, &x1s, &x2s, &x3s, &x4s] {
        const auto generator = [=] {
          static thread_local std::mt19937 rng(std::random_device{}());
          std::uniform_real_distribution<RefType_t<TArg>> unif_dist(a, b);
          return static_cast<TArg>(unif_dist(rng));
        };
        std::generate(x1s.ptr() + base_idx, x1s.ptr() + base_idx + sub_batch_size, generator);
        std::generate(x2s.ptr() + base_idx, x2s.ptr() + base_idx + sub_batch_size, generator);
        std::generate(x3s.ptr() + base_idx, x3s.ptr() + base_idx + sub_batch_size, generator);
        std::generate(x4s.ptr() + base_idx, x4s.ptr() + base_idx + sub_batch_size, generator);
      });
      base_idx += sub_batch_size;
    }

    thread_pool.Wait();

    math_test.Run(validator_builder, grid_size, block_size, ref_func, batch_size, x1s.ptr(),
                  x2s.ptr(), x3s.ptr(), x4s.ptr());
  }
}

template <typename T, typename TArg, typename RT, typename RTArg, typename ValidatorBuilder>
void QuaternaryFloatingPointSpecialValuesTest(kernel_sig<T, TArg, TArg, TArg, TArg> kernel,
                                              ref_sig<RT, RTArg, RTArg, RTArg, RTArg> ref_func,
                                              const ValidatorBuilder& validator_builder) {
  const auto [grid_size, block_size] = GetOccupancyMaxPotentialBlockSize(kernel);
  const auto values = std::get<SpecialVals<TArg>>(kSpecialValReducedRegistry);

  const auto size = values.size * values.size * values.size * values.size;
  LinearAllocGuard<TArg> x1s{LinearAllocs::hipHostMalloc, size * sizeof(TArg)};
  LinearAllocGuard<TArg> x2s{LinearAllocs::hipHostMalloc, size * sizeof(TArg)};
  LinearAllocGuard<TArg> x3s{LinearAllocs::hipHostMalloc, size * sizeof(TArg)};
  LinearAllocGuard<TArg> x4s{LinearAllocs::hipHostMalloc, size * sizeof(TArg)};

  for (auto i = 0u; i < values.size; ++i) {
    for (auto j = 0u; j < values.size; ++j) {
      for (auto k = 0u; k < values.size; ++k) {
        for (auto l = 0u; l < values.size; ++l) {
          x1s.ptr()[((i * values.size + j) * values.size + k) * values.size + l] = values.data[i];
          x2s.ptr()[((i * values.size + j) * values.size + k) * values.size + l] = values.data[j];
          x3s.ptr()[((i * values.size + j) * values.size + k) * values.size + l] = values.data[k];
          x4s.ptr()[((i * values.size + j) * values.size + k) * values.size + l] = values.data[l];
        }
      }
    }
  }

  MathTest math_test(kernel, size);
  math_test.template Run<false>(validator_builder, grid_size, block_size, ref_func, size, x1s.ptr(),
                                x2s.ptr(), x3s.ptr(), x4s.ptr());
}

template <typename T, typename TArg, typename RT, typename RTArg, typename ValidatorBuilder>
void QuaternaryFloatingPointTest(kernel_sig<T, TArg, TArg, TArg, TArg> kernel,
                                 ref_sig<RT, RTArg, RTArg, RTArg, RTArg> ref_func,
                                 const ValidatorBuilder& validator_builder) {
  SECTION("Special values") {
    QuaternaryFloatingPointSpecialValuesTest(kernel, ref_func, validator_builder);
  }

  SECTION("Brute force") {
    QuaternaryFloatingPointBruteForceTest(kernel, ref_func, validator_builder);
  }
}


#define MATH_QUATERNARY_WITHIN_ULP_TEST_DEF(kern_name, ref_func, sp_ulp, dp_ulp)                   \
  MATH_QUATERNARY_KERNEL_DEF(kern_name)                                                            \
                                                                                                   \
  TEMPLATE_TEST_CASE("Unit_Device_" #kern_name "_Accuracy_Positive", "", float, double) {          \
    using RT = RefType_t<TestType>;                                                                \
    RT (*ref)(RT, RT, RT, RT) = ref_func;                                                          \
    const auto ulp = std::is_same_v<float, TestType> ? sp_ulp : dp_ulp;                            \
                                                                                                   \
    QuaternaryFloatingPointTest(kern_name##_kernel<TestType>, ref,                                 \
                                ULPValidatorBuilderFactory<TestType>(ulp));                        \
  }
