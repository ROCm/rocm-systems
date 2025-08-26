/*
 Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

/**
* @addtogroup hipPerfMultiStreamKernelLaunch hipPerfMultiStreamKernelLaunch
* @{
* @ingroup perfStreamTest
*/

#include "hip_test_common.hh"
#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>

#if HT_AMD
#define device_clock64() wall_clock64()
#else
#define device_clock64() clock64()
#endif

__global__ void EmptyKernel() { }

__global__ void TimingKernel(uint64_t count) {
  uint64_t begin_time = device_clock64();
  uint64_t curr_time = begin_time;
  do {
    curr_time = device_clock64();
  } while (begin_time + count > curr_time);
}

class Experiment {
 public:
  struct Metrics {
    double walltime_us;
  };

  Experiment(uint32_t num_streams):
    num_streams_{num_streams}, streams_{num_streams, nullptr} { }

  Experiment(const Experiment& other):
    num_streams_{other.num_streams_}, streams_{other.num_streams_, nullptr} { }

  void init() {
    for (hipStream_t& s: streams_){
      HIP_CHECK(hipStreamCreate(&s));
    }
  }

  void cleanup() {
    for (hipStream_t& s: streams_) {
      if (s != nullptr) {
        HIP_CHECK(hipStreamDestroy(s));
        s = nullptr;
      }
    }
  }
  
  template<typename... Args, typename F = void (*)(Args...)>
  void do_warmup(const uint64_t iterations, const uint64_t dispatch_per_stream, F func, Args... args) const {
    for (uint64_t i = 0; i < iterations; i++) {
      for (uint32_t j = 0; j < dispatch_per_stream; j++) {
        for (const hipStream_t& s: streams_) {
          func<<<1,1,0,s>>>(args...);
        }
      }
      for (const hipStream_t& s: streams_) {
        HIP_CHECK(hipStreamSynchronize(s));
      }
    }
  }

  template<typename... Args, typename F = void (*)(Args...)>
  Metrics run(const uint64_t dispatch_per_stream, F func, Args... args) {
    auto start = std::chrono::steady_clock::now();
    for (uint32_t j = 0; j < dispatch_per_stream; j++) {
      for (const hipStream_t& s: streams_) {
        func<<<1,1,0,s>>>(args...);
      }
    }
    for (const hipStream_t& s: streams_) {
      HIP_CHECK(hipStreamSynchronize(s));
    }
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<microseconds>(end - start);
    return Metrics{duration.count()};
  }

 private:
  using microseconds = std::chrono::duration<double, std::chrono::microseconds::period>;
  uint64_t num_streams_;
  std::vector<hipStream_t> streams_;
};

TEST_CASE("Perf_hipPerfMultiStreamKernelLaunch") {
  constexpr uint64_t KERNEL_SLEEP_US = 100;
  constexpr uint64_t KERNEL_DISPATCHES_PER_STREAM = 10;
  constexpr uint64_t WARMUP_KERNEL_DISPATCHES_PER_STREAM = 10;
  constexpr uint64_t WARMUP_ITERATIONS = 10;
  constexpr uint64_t STREAMS_PER_EXPERIMENT[] = {
    2, 4, 8, 16, 32, 64, 128, 256, 512, 1024
  };
  int clock_rate = 0;  // in kHz
#if HT_AMD
  HIP_CHECK(hipDeviceGetAttribute(&clock_rate, hipDeviceAttributeWallClockRate, 0));
#else
  HIP_CHECK(hipDeviceGetAttribute(&clock_rate, hipDeviceAttributeClockRate, 0));
#endif
  uint64_t timer_freq_in_hz = clock_rate * 1000;

  // Log config
  std::cout << "Using " << (KERNEL_SLEEP_US == 0? "EmptyKernel": "TimingKernel") << ", duration (us): " << KERNEL_SLEEP_US << std::endl;
  std::cout << "Warmup Iterations: " << WARMUP_ITERATIONS << std::endl;
  std::cout << "Kernel dispatches per stream: " << KERNEL_DISPATCHES_PER_STREAM << std::endl;
  std::cout << std::setw(20) << "Num Streams " << "|" << std::setw(20) << "Walltime (us)" << std::endl;
  std::cout << std::string(20, '-') << "|" << std::string(20, '-') << std::endl;
  const uint64_t timing_count = timer_freq_in_hz * KERNEL_SLEEP_US / 1'000'000;

  for (const auto& num_streams : STREAMS_PER_EXPERIMENT) {
    Experiment exp(num_streams);
    Experiment::Metrics metrics;
    exp.init();
    exp.do_warmup(WARMUP_ITERATIONS, WARMUP_KERNEL_DISPATCHES_PER_STREAM, TimingKernel, timing_count);
    HIP_CHECK(hipDeviceSynchronize());
    if (KERNEL_SLEEP_US == 0) {
      metrics = exp.run(KERNEL_DISPATCHES_PER_STREAM, EmptyKernel);
    } else {
      metrics = exp.run(KERNEL_DISPATCHES_PER_STREAM, TimingKernel, timing_count);
    }
    exp.cleanup();
    std::cout << std::setw(20) << num_streams << "|" << std::setw(20) << std::setprecision(2) << std::fixed << metrics.walltime_us << std::endl;
  }
}

/**
* End doxygen group perfStreamTest.
* @}
*/