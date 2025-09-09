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

#include <cmd_options.hh>
#include <hip_test_common.hh>
#include <resource_guards.hh>

#include <hip/hip_cooperative_groups.h>

#include "Float16.hh"
#include "thread_pool.hh"
#include "validators.hh"

namespace cg = cooperative_groups;

template <typename T, typename U>
std::enable_if_t<std::conjunction_v<std::is_arithmetic<T>, std::is_arithmetic<U>>, std::ostream&>
operator<<(std::ostream& os, const std::pair<T, U>& p) {
  const auto default_prec = os.precision();
  return os << "<" << std::setprecision(std::numeric_limits<T>::max_digits10 - 1) << p.first << ", "
            << std::setprecision(std::numeric_limits<U>::max_digits10 - 1) << p.second << ">"
            << std::setprecision(default_prec);
}

template <typename T>
std::enable_if_t<sizeof(T) / sizeof(decltype(T().x)) == 2 && !std::is_same_v<T, __half2>, std::ostream&>
operator<<(std::ostream& os, const T& p) {
  const auto default_prec = os.precision();
  return os << "<" << std::setprecision(std::numeric_limits<decltype(T().x)>::max_digits10 - 1) << p.x << ", "
            << std::setprecision(std::numeric_limits<decltype(T().x)>::max_digits10 - 1) << p.y << ">"
            << std::setprecision(default_prec);
}

// This class represents a generic numerical accuracy math test. Template parameter T is the output
// type of the function being tested, and template parameter pack Ts represents the input types. The
// constructor takes a kernel with the signature void(T*, const size_t, Ts*...). The first kernel
// parameter is the output array, the second parameter is the number of outputs, and the rest of the
// parameters are arrays containing input values. The number of input arrays depends on the arity of
// the function being tested e.g. one input array for unary functions, two input arrays for binary
// functions, etc. The kernel threads take one element from each input array at the index
// corresponding to that thread, feed the input elements to the testee function, and store the
// result in the output array at the corresponding index.
//
// E.g. for a binary function the kernel would have the following signature:
//   void kernel(float* y, const size_t n, float* x1, float* x2)
//
// The outputs would be calculated in parallel the following way:
//   y[0] = testee(x1[0], x2[0])
//   y[1] = testee(x1[1], x2[1])
//   y[2] = testee(x1[2], x2[2])
//   ...
//
// The constructor also takes max_num_args, which represents the maximum number of input values used
// for one kernel launch. The device memory for the input and output arrays is allocated based on
// that number.
template <typename T, typename... Ts> class MathTest {
 public:
  MathTest(void (*kernel)(T*, const size_t, Ts*...), const size_t max_num_args)
      : kernel_{kernel},
        xss_dev_(LinearAllocGuard<Ts>(LinearAllocs::hipMalloc, max_num_args * sizeof(Ts))...),
        y_dev_{LinearAllocs::hipMalloc, max_num_args * sizeof(T)},
        y_{LinearAllocs::hipHostMalloc, max_num_args * sizeof(T)} {}

  // This method runs the test with the following steps:
  // 1. Copy the values from the input arrays provided in the parameter pack xss to device memory
  // 2. Launch the kernel using the configuration provided in grid_dims and block_dims
  // 3. Copy the outputs back to host memory
  // 4. Generate the reference values using ref_func and compare against the outputs using the
  // validator provided by validator_builder
  // 5. If non-type template parameter parallel is true, then step 4 is broken up into chunks of
  // work that are done in parallel on the host.
  template <bool parallel = true, typename RT, typename ValidatorBuilder, typename... RTs>
  void Run(const ValidatorBuilder& validator_builder, const size_t grid_dims,
           const size_t block_dims, RT (*const ref_func)(RTs...), const size_t num_args,
           const Ts*... xss) {
    fail_flag_.store(false);
    error_info_.clear();
    RunImpl<parallel>(validator_builder, grid_dims, block_dims, ref_func, num_args,
                      std::index_sequence_for<Ts...>{}, xss...);
  }

 private:
  void (*kernel_)(T*, const size_t, Ts*...);
  std::tuple<LinearAllocGuard<Ts>...> xss_dev_;
  LinearAllocGuard<T> y_dev_;
  LinearAllocGuard<T> y_;
  std::atomic<bool> fail_flag_{false};
  std::mutex mtx_;
  std::string error_info_;

  template <bool parallel, typename RT, typename ValidatorBuilder, typename... RTs, size_t... I>
  void RunImpl(const ValidatorBuilder& validator_builder, const size_t grid_dim,
               const size_t block_dim, RT (*const ref_func)(RTs...), const size_t num_args,
               std::index_sequence<I...>, const Ts*... xss) {
    const auto xss_tup = std::make_tuple(xss...);

    constexpr auto f = [](auto dst, auto src, size_t size) {
      HIP_CHECK(hipMemcpy(dst, src, size, hipMemcpyHostToDevice))
    };

    ((f(std::get<I>(xss_dev_).ptr(), std::get<I>(xss_tup),
        num_args * sizeof(*std::get<I>(xss_tup)))),
     ...);

    kernel_<<<grid_dim, block_dim>>>(y_dev_.ptr(), num_args, std::get<I>(xss_dev_).ptr()...);
    HIP_CHECK(hipGetLastError());

    HIP_CHECK(hipMemcpy(y_.ptr(), y_dev_.ptr(), num_args * sizeof(T), hipMemcpyDeviceToHost));
    HIP_CHECK(hipStreamSynchronize(nullptr));

    if constexpr (!parallel) {
      for (auto i = 0u; i < num_args; ++i) {
        const auto actual_val = y_.ptr()[i];
        const auto ref_val = static_cast<T>(ref_func(xss[i]...));
        const auto validator = validator_builder(ref_val, xss[i]...);

        if (!validator->match(actual_val)) {
          const auto log = MakeLogMessage(actual_val, xss[i]...) + validator->describe() + "\n";
          INFO(log);
          REQUIRE(false);
        }
      }

      return;
    }

    const auto task = [&, this](size_t iters, size_t base_idx) {
      for (auto i = 0u; i < iters; ++i) {
        if (fail_flag_.load(std::memory_order_relaxed)) return;

        const auto actual_val = y_.ptr()[base_idx + i];
        const auto ref_val = static_cast<T>(ref_func(xss[base_idx + i]...));
        const auto validator = validator_builder(ref_val, xss[base_idx + i]...);

        if (!validator->match(actual_val)) {
          fail_flag_.store(true, std::memory_order_relaxed);
          // Several threads might have passed the first check, but failed validation. On the
          // chance of this happening, access to the string stream must be serialized.
          const auto log =
              MakeLogMessage(actual_val, xss[base_idx + i]...) + validator->describe() + "\n";
          {
            std::lock_guard lg{mtx_};
            error_info_ += log;
          }
          return;
        }
      }
    };

    const auto task_count = thread_pool.thread_count();
    const auto chunk_size = num_args / task_count;
    const auto tail = num_args % task_count;

    auto base_idx = 0u;
    for (auto i = 0u; i < task_count; ++i) {
      const auto iters = chunk_size + (i < tail);
      thread_pool.Post([=, &task] { task(iters, base_idx); });
      base_idx += iters;
    }

    thread_pool.Wait();

    INFO(error_info_);
    REQUIRE(!fail_flag_);
  }

  template <typename... Args> std::string MakeLogMessage(T actual_val, Args... args) {
    std::stringstream ss;
    ss << "Input value(s): " << std::scientific
       << std::setprecision(std::numeric_limits<T>::max_digits10 - 1);
    ((ss << " " << args), ...) << "\n" << actual_val << " ";

    return ss.str();
  }
};

template <typename T> struct RefType {};

template <> struct RefType<Float16> { using type = float; };

template <> struct RefType<float> { using type = double; };

template <> struct RefType<double> { using type = long double; };

template <typename T> using RefType_t = typename RefType<T>::type;

template <typename F> auto GetOccupancyMaxPotentialBlockSize(F kernel) {
  int grid_size = 0, block_size = 0;
  HIP_CHECK(hipOccupancyMaxPotentialBlockSize(&grid_size, &block_size, kernel, 0, 0));
  return std::make_tuple(grid_size, block_size);
}

inline size_t GetMaxAllowedDeviceMemoryUsage() {
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, 0));
  return props.totalGlobalMem * (cmd_options.accuracy_max_memory * 0.01f);
}

inline uint64_t GetTestIterationCount() { return cmd_options.accuracy_iterations; }

template <typename T, typename... Ts> using kernel_sig = void (*)(T*, const size_t, Ts*...);

template <typename T, typename... Ts> using ref_sig = T (*)(Ts...);

template <int error_num> void NegativeTestRTCWrapper(const char* program_source) {
  hiprtcProgram program{};

  HIPRTC_CHECK(
      hiprtcCreateProgram(&program, program_source, "math_test_rtc.cc", 0, nullptr, nullptr));
#if HT_AMD
  std::string args = std::string("-ferror-limit=200");
  const char* options[] = {args.c_str()};
  hiprtcResult result{hiprtcCompileProgram(program, 1, options)};
#else
  hiprtcResult result{hiprtcCompileProgram(program, 0, nullptr)};
#endif

  // Get the compile log and count compiler error messages
  size_t log_size{};
  HIPRTC_CHECK(hiprtcGetProgramLogSize(program, &log_size));
  std::string log(log_size, ' ');
  HIPRTC_CHECK(hiprtcGetProgramLog(program, log.data()));
  int error_count{0};

  int expected_error_count{error_num};
  std::string error_message{"error:"};

  size_t n_pos = log.find(error_message, 0);
  while (n_pos != std::string::npos) {
    ++error_count;
    n_pos = log.find(error_message, n_pos + 1);
  }

  HIPRTC_CHECK(hiprtcDestroyProgram(&program));
  HIPRTC_CHECK_ERROR(result, HIPRTC_ERROR_COMPILATION);
  REQUIRE(error_count == expected_error_count);
}
