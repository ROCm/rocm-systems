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
* @addtogroup hipPerfMultiStreamKernelLaunch2 hipPerfMultiStreamKernelLaunch2
* @{
* @ingroup perfStreamTest
*
* @brief Performance test for selective multi-stream dispatch with inter-stream
*        event synchronization.
*
* This test exercises dynamic stream to hardware queue (HWq) assignment with
* selective stream usage patterns and cross-stream dependencies via events.
* It creates 8 streams but only dispatches work to a subset, testing how the
* dynamic queue heuristics handle sparse stream usage.
*
* Two run modes (toggle via USE_RUN2 environment variable):
*
* run (default): Two-phase dispatch with mid-point synchronization
*   - First half: dispatch kernels on streams {0, 3, 4, 7}
*   - Synchronize stream 0
*   - Second half: dispatch kernels on streams {0, 2, 3, 6}
*   - hipEventRecord on stream 3
*   - hipStreamWaitEvent on stream 5 (creates cross-stream dependency)
*   - Single kernel launch on stream 5
*
* run2 (USE_RUN2=1): Simple selective dispatch with event sync
*   - Dispatch kernels on streams {0, 3, 4, 7}
*   - hipEventRecord on stream 3
*   - hipStreamWaitEvent on stream 5
*   - Single kernel launch on stream 5
*
* This test validates:
* - Dynamic HWq assignment with non-uniform stream usage
* - Event-based cross-stream synchronization correctness
* - Queue selection when streams have inter-dependencies
*
* Related environment variables:
* - USE_RUN2: 0 or unset - use run() (default), 1 - use run2()
* - DEBUG_HIP_DYNAMIC_QUEUE: 0 - disabled, 1 - Queue Depth heuristics (default)
*/

#include "hip_test_common.hh"
#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>
#include <random>
#include <set>
#include <cstdlib>

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
  Metrics run2(const uint64_t dispatch_per_stream, F func, Args... args) {
    // Dispatch on streams 0, 3, 4, 7
    std::set<uint32_t> target_indices = {0, 3, 4, 7};

    // Additional event-based sync between stream 3 and 5
    constexpr uint32_t record_stream_idx = 3;
    constexpr uint32_t wait_stream_idx = 5;

    std::cout << "Dispatching on stream indices: ";
    for (uint32_t idx : target_indices) {
      std::cout << idx << " ";
    }
    std::cout << std::endl;
    std::cout << "Additionally: hipEventRecord on stream " << record_stream_idx
              << ", hipStreamWaitEvent on stream " << wait_stream_idx
              << ", kernel launch on stream " << wait_stream_idx << std::endl;

    hipEvent_t event;
    HIP_CHECK(hipEventCreate(&event));

    auto start = std::chrono::steady_clock::now();

    // Dispatch kernels on target streams
    for (uint32_t j = 0; j < dispatch_per_stream; j++) {
      uint32_t stream_idx = 0;
      for (const hipStream_t& s: streams_) {
        if (target_indices.find(stream_idx) != target_indices.end()) {
          func<<<1,1,0,s>>>(args...);
        }
        stream_idx++;
      }
    }

    // Record event on stream 3
    HIP_CHECK(hipEventRecord(event, streams_[record_stream_idx]));

    // Stream 5 waits for the event from stream 3
    HIP_CHECK(hipStreamWaitEvent(streams_[wait_stream_idx], event, 0));

    // Single kernel launch on stream 5
    func<<<1,1,0,streams_[wait_stream_idx]>>>(args...);

    // Synchronize all streams
    for (const hipStream_t& s: streams_) {
      HIP_CHECK(hipStreamSynchronize(s));
    }

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<microseconds>(end - start);

    HIP_CHECK(hipEventDestroy(event));

    return Metrics{duration.count()};
  }

  template<typename... Args, typename F = void (*)(Args...)>
  Metrics run(const uint64_t dispatch_per_stream, F func, Args... args) {
    // First half: dispatch on streams 0, 3, 4, 7
    std::set<uint32_t> first_half_indices = {0, 3, 4, 7};
    // Second half: dispatch on streams 0, 2, 3, 6
    std::set<uint32_t> second_half_indices = {0, 2, 3, 6};

    // Additional event-based sync between stream 3 and 5
    constexpr uint32_t record_stream_idx = 3;
    constexpr uint32_t wait_stream_idx = 5;

    std::cout << "First half dispatching on stream indices: ";
    for (uint32_t idx : first_half_indices) {
      std::cout << idx << " ";
    }
    std::cout << std::endl;
    std::cout << "Second half dispatching on stream indices: ";
    for (uint32_t idx : second_half_indices) {
      std::cout << idx << " ";
    }
    std::cout << std::endl;
    std::cout << "Additionally: hipEventRecord on stream " << record_stream_idx
              << ", hipStreamWaitEvent on stream " << wait_stream_idx
              << ", kernel launch on stream " << wait_stream_idx << std::endl;

    hipEvent_t event;
    HIP_CHECK(hipEventCreate(&event));

    auto start = std::chrono::steady_clock::now();

    const uint64_t half_dispatches = dispatch_per_stream / 2;

    // First half: dispatch kernels on streams 0, 3, 4, 7
    for (uint32_t j = 0; j < half_dispatches; j++) {
      uint32_t stream_idx = 0;
      for (const hipStream_t& s: streams_) {
        if (first_half_indices.find(stream_idx) != first_half_indices.end()) {
          func<<<1,1,0,s>>>(args...);
        }
        stream_idx++;
      }
    }

    // Synchronize stream 0 after first half
    HIP_CHECK(hipStreamSynchronize(streams_[0]));

    // Second half: dispatch kernels on streams 0, 2, 3, 6
    for (uint32_t j = half_dispatches; j < dispatch_per_stream * 2; j++) {
      uint32_t stream_idx = 0;
      for (const hipStream_t& s: streams_) {
        if (second_half_indices.find(stream_idx) != second_half_indices.end()) {
          func<<<1,1,0,s>>>(args...);
        }
        stream_idx++;
      }
    }

    // Record event on stream 3
    HIP_CHECK(hipEventRecord(event, streams_[record_stream_idx]));

    // Stream 5 waits for the event from stream 3
    HIP_CHECK(hipStreamWaitEvent(streams_[wait_stream_idx], event, 0));

    // Single kernel launch on stream 5
    func<<<1,1,0,streams_[wait_stream_idx]>>>(args...);

    // Synchronize all streams
    for (const hipStream_t& s: streams_) {
      HIP_CHECK(hipStreamSynchronize(s));
    }

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<microseconds>(end - start);

    HIP_CHECK(hipEventDestroy(event));

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
     8
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

  // Check for USE_RUN2 environment variable to select run2 instead of run (default)
  const char* use_run2_env = std::getenv("USE_RUN2");
  bool use_run2 = (use_run2_env != nullptr && std::string(use_run2_env) == "1");
  std::cout << "Using: " << (use_run2 ? "run2 (event sync)" : "run (default)") << std::endl;

  for (const auto& num_streams : STREAMS_PER_EXPERIMENT) {
    Experiment exp(num_streams);
    Experiment::Metrics metrics;
    exp.init();
    exp.do_warmup(WARMUP_ITERATIONS, WARMUP_KERNEL_DISPATCHES_PER_STREAM, TimingKernel, timing_count);
    HIP_CHECK(hipDeviceSynchronize());
    if (use_run2) {
      if (KERNEL_SLEEP_US == 0) {
        metrics = exp.run2(KERNEL_DISPATCHES_PER_STREAM, EmptyKernel);
      } else {
        metrics = exp.run2(KERNEL_DISPATCHES_PER_STREAM, TimingKernel, timing_count);
      }
    } else {
      if (KERNEL_SLEEP_US == 0) {
        metrics = exp.run(KERNEL_DISPATCHES_PER_STREAM, EmptyKernel);
      } else {
        metrics = exp.run(KERNEL_DISPATCHES_PER_STREAM, TimingKernel, timing_count);
      }
    }
    exp.cleanup();
    std::cout << std::setw(20) << num_streams << "|" << std::setw(20) << std::setprecision(2) << std::fixed << metrics.walltime_us << std::endl;
  }
}

/**
* End doxygen group perfStreamTest.
* @}
*/