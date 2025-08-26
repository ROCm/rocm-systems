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

#define MATH_BESSEL_N_KERNEL_DEF(func_name)                                                        \
  template <typename T>                                                                            \
  __global__ void func_name##_kernel(T* const ys, const size_t num_xs, int* n, T* const xs) {      \
    const auto tid = cg::this_grid().thread_rank();                                                \
    const auto stride = cg::this_grid().size();                                                    \
                                                                                                   \
    for (auto i = tid; i < num_xs; i += stride) {                                                  \
      if constexpr (std::is_same_v<float, T>) {                                                    \
        ys[i] = func_name##f(n[i], xs[i]);                                                         \
      } else if constexpr (std::is_same_v<double, T>) {                                            \
        ys[i] = func_name(n[i], xs[i]);                                                            \
      }                                                                                            \
    }                                                                                              \
  }

template <typename T> using kernel_bessel_n_sig = void (*)(T*, const size_t, int*, T*);

template <typename T> using ref_bessel_n_sig = T (*)(int, T);

template <typename ValidatorBuilder>
void BesselDoublePrecisionBruteForceTest(kernel_bessel_n_sig<double> kernel,
                                         ref_bessel_n_sig<long double> ref_func,
                                         const ValidatorBuilder& validator_builder, int n_input = 0,
                                         const double a = std::numeric_limits<double>::lowest(),
                                         const double b = std::numeric_limits<double>::max()) {
  const auto [grid_size, block_size] = GetOccupancyMaxPotentialBlockSize(kernel);
  const uint64_t num_iterations = GetTestIterationCount();
  const auto max_batch_size = std::min(
      GetMaxAllowedDeviceMemoryUsage() / (sizeof(double) * 2 + sizeof(int)), num_iterations);
  LinearAllocGuard<int> x1s{LinearAllocs::hipHostMalloc, max_batch_size * sizeof(int)};
  LinearAllocGuard<double> x2s{LinearAllocs::hipHostMalloc, max_batch_size * sizeof(double)};

  MathTest math_test(kernel, max_batch_size);
  std::fill_n(x1s.ptr(), max_batch_size, n_input);

  auto batch_size = max_batch_size;
  const auto num_threads = thread_pool.thread_count();
  for (uint64_t i = 0ul; i < num_iterations; i += batch_size) {
    batch_size = std::min<uint64_t>(max_batch_size, num_iterations - i);

    const auto min_sub_batch_size = batch_size / num_threads;
    const auto tail = batch_size % num_threads;

    auto base_idx = 0u;
    for (auto i = 0u; i < num_threads; ++i) {
      const auto sub_batch_size = min_sub_batch_size + (i < tail);
      thread_pool.Post([=, &x2s] {
        const auto generator = [=] {
          static thread_local std::mt19937 rng(std::random_device{}());
          std::uniform_real_distribution<RefType_t<double>> unif_dist(a, b);
          return static_cast<double>(unif_dist(rng));
        };
        std::generate(x2s.ptr() + base_idx, x2s.ptr() + base_idx + sub_batch_size, generator);
      });
      base_idx += sub_batch_size;
    }

    thread_pool.Wait();

    math_test.Run(validator_builder, grid_size, block_size, ref_func, batch_size, x1s.ptr(),
                  x2s.ptr());
  }
}

template <typename ValidatorBuilder>
void BesselSinglePrecisionRangeTest(kernel_bessel_n_sig<float> kernel,
                                    ref_bessel_n_sig<double> ref_func,
                                    const ValidatorBuilder& validator_builder, int n_input,
                                    const float a, const float b) {
  const auto [grid_size, block_size] = GetOccupancyMaxPotentialBlockSize(kernel);
  const auto max_batch_size = GetMaxAllowedDeviceMemoryUsage() / (sizeof(float) * 2 + sizeof(int));
  LinearAllocGuard<int> x1s{LinearAllocs::hipHostMalloc, max_batch_size * sizeof(int)};
  LinearAllocGuard<float> x2s{LinearAllocs::hipHostMalloc, max_batch_size * sizeof(float)};

  MathTest math_test(kernel, max_batch_size);
  std::fill_n(x1s.ptr(), max_batch_size, n_input);

  size_t inserted = 0u;
  for (float v = a; v != b; v = std::nextafter(v, b)) {
    x2s.ptr()[inserted++] = v;
    if (inserted < max_batch_size) continue;

    math_test.Run(validator_builder, grid_size, block_size, ref_func, inserted, x1s.ptr(),
                  x2s.ptr());
    inserted = 0u;
  }
}

template <typename T, typename F, typename ValidatorBuilder>
void SpecialSimpleTest(F kernel, const ValidatorBuilder& validator_builder, const T* x,
                       const T* ref, size_t num_args) {
  LinearAllocGuard<T> x_dev{LinearAllocs::hipMalloc, num_args * sizeof(T)};
  LinearAllocGuard<T> y{LinearAllocs::hipHostMalloc, num_args * sizeof(T)};
  LinearAllocGuard<T> y_dev{LinearAllocs::hipMalloc, num_args * sizeof(T)};

  HIP_CHECK(hipMemcpy(x_dev.ptr(), x, num_args * sizeof(T), hipMemcpyHostToDevice));

  kernel<<<1, num_args>>>(y_dev.ptr(), num_args, x_dev.ptr());
  HIP_CHECK(hipGetLastError());

  HIP_CHECK(hipMemcpy(y.ptr(), y_dev.ptr(), num_args * sizeof(T), hipMemcpyDeviceToHost));

  for (auto i = 0u; i < num_args; ++i) {
    const auto actual_val = y.ptr()[i];
    const auto ref_val = ref[i];
    const auto validator = validator_builder(ref_val);

    if (!validator->match(actual_val)) {
      std::stringstream ss;
      ss << "Input value(s): " << std::scientific
         << std::setprecision(std::numeric_limits<T>::max_digits10 - 1);
      ss << x[i] << " " << actual_val << " " << ref_val << "\n";
      INFO(ss.str());
      REQUIRE(false);
    }
  }
}
