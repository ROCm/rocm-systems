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

#include <algorithm>
#include <chrono>
#include <memory>
#include <numeric>
#include <type_traits>
#include <vector>

#include <cmd_options.hh>
#include <hip_test_common.hh>
#include <resource_guards.hh>

#pragma clang diagnostic ignored "-Wunused-but-set-variable"
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wunused-function"

#if defined(_WIN32)
#if defined(_WIN64)
typedef __int64 ssize_t;
#else   // !_WIN64
typedef __int32 ssize_t;
#endif  // !_WIN64
#endif  /*_WIN32*/

class Timer {
 public:
  Timer(const Timer&) = delete;
  Timer& operator=(const Timer&) = delete;

 protected:
  Timer(float& time, hipStream_t stream) : time_(time), stream_(stream) {}

  void Record(float time) { time_ += time; }

  hipStream_t GetStream() const { return stream_; }

 private:
  float& time_;
  hipStream_t stream_;
};

class EventTimer : public Timer {
 public:
  EventTimer(float& time, hipStream_t stream = nullptr) : Timer(time, stream) {
    HIP_CHECK(hipEventCreate(&start_));
    HIP_CHECK(hipEventCreate(&stop_));
    HIP_CHECK(hipEventRecord(start_, GetStream()));
  }

  ~EventTimer() {
    hipError_t error;  // to avoid compiler warnings

    error = hipEventRecord(stop_, GetStream());
    error = hipEventSynchronize(stop_);

    float ms;
    error = hipEventElapsedTime(&ms, start_, stop_);
    Record(ms);

    error = hipEventDestroy(start_);
    error = hipEventDestroy(stop_);
  }

 private:
  hipEvent_t start_;
  hipEvent_t stop_;
};

class CpuTimer : public Timer {
 public:
  CpuTimer(float& time, hipStream_t stream = nullptr) : Timer(time, stream) {
    start_ = std::chrono::steady_clock::now();
  }

  ~CpuTimer() {
    hipError_t error;  // to avoid compiler warnings
    error = hipStreamSynchronize(GetStream());

    stop_ = std::chrono::steady_clock::now();

    std::chrono::duration<float, std::milli> ms = stop_ - start_;
    Record(ms.count());
  }

 private:
  std::chrono::time_point<std::chrono::steady_clock> start_;
  std::chrono::time_point<std::chrono::steady_clock> stop_;
};

template <typename Derived> class Benchmark {
 public:
  Benchmark()
      : iterations_(cmd_options.iterations),
        warmups_(cmd_options.warmups),
        display_output_(!cmd_options.no_display),
        progress_bar_(cmd_options.progress) {
    benchmark_name_ = Catch::getResultCapture().getCurrentTestName();
  }

  Benchmark(const Benchmark&) = delete;
  Benchmark& operator=(const Benchmark&) = delete;

  static constexpr ssize_t kWarmup = -1;

  void Configure(size_t iterations, size_t warmups) {
    iterations_ = iterations;
    warmups_ = warmups;
  }

  void AddSectionName(const std::string& section_name) { benchmark_name_ += "/" + section_name; }

  using ModifierSignature = std::function<float(float)>;
  void RegisterModifier(const ModifierSignature& modifier) { modifier_ = modifier; }

  template <typename... Args> std::tuple<float, float, float, float> Run(Args&&... args) {
    AddSectionName(std::to_string(iterations_));
    AddSectionName(std::to_string(warmups_));

    auto& derived = static_cast<Derived&>(*this);

    current_ = kWarmup;
    for (size_t i = 0u; i < warmups_; ++i) {
      PrintProgress("warmup", static_cast<int>(100.f * (i + 1) / warmups_));
      derived(args...);
    }
    time_ = .0;

    std::vector<float> samples;
    samples.reserve(iterations_);

    for (current_ = 0; current_ < iterations_; ++current_) {
      PrintProgress("measurement", static_cast<int>(100.f * (current_ + 1) / iterations_));
      derived(args...);
      if (modifier_) time_ = modifier_(time_);
      samples.push_back(time_);
      time_ = .0;
    }

    float sum = std::accumulate(cbegin(samples), cend(samples), .0);
    float mean = sum / samples.size();

    float deviation =
        std::accumulate(cbegin(samples), cend(samples), .0,
                        [mean](float sum, float next) { return sum + std::pow(next - mean, 2); });
    deviation = sqrt(deviation / samples.size());

    float best = *std::min_element(cbegin(samples), cend(samples));
    float worst = *std::max_element(cbegin(samples), cend(samples));

    PrintStats(mean, deviation, best, worst);

    return {mean, deviation, best, worst};
  }

 protected:
  template <bool event_based>
  using TimerType = std::conditional_t<event_based, EventTimer, CpuTimer>;

  template <bool event_based = false>
  std::unique_ptr<TimerType<event_based>> GetTimer(hipStream_t stream = nullptr) {
    return std::make_unique<TimerType<event_based>>(time_, stream);
  }

  float time() const { return time_; }

  size_t iterations() const { return iterations_; }

  size_t warmups() const { return warmups_; }

  ssize_t current() const { return current_; }

 private:
  std::string benchmark_name_;
  float time_;
  size_t iterations_;
  size_t warmups_;
  ssize_t current_;
  bool display_output_;
  bool progress_bar_;

  ModifierSignature modifier_;

  void Print(const std::string& out = "") {
    if (!display_output_) return;
    std::cout << "\r" << std::setw(110) << std::left << benchmark_name_ << "\t|\t" << out
              << std::flush;
  }

  void PrintProgress(const std::string& name, int progress) {
    if (!(display_output_ && progress_bar_)) return;
    Print(name + ": [" + std::to_string(progress) + "%]");
  }

  void PrintStats(float mean, float deviation, float best, float worst) {
    if (!display_output_) return;
    Print("Average time: " + std::to_string(mean) + " ms, Standard deviation: " +
          std::to_string(deviation) + " ms, Fastest: " + std::to_string(best) +
          " ms, Slowest: " + std::to_string(worst) + " ms\n");
  }
};

constexpr bool kTimerTypeCpu = false;
constexpr bool kTimerTypeEvent = true;

#define TIMED_SECTION_STREAM(TIMER_TYPE, STREAM)                                                   \
  if (auto _ = this->template GetTimer<TIMER_TYPE>(STREAM); true)
#define TIMED_SECTION(TIMER_TYPE) TIMED_SECTION_STREAM(TIMER_TYPE, nullptr)

constexpr size_t operator"" _KB(unsigned long long int kb) { return kb << 10; }

constexpr size_t operator"" _MB(unsigned long long int mb) { return mb << 20; }

constexpr size_t operator"" _GB(unsigned long long int gb) { return gb << 30; }

static std::string GetAllocationSectionName(LinearAllocs allocation_type) {
  switch (allocation_type) {
    case LinearAllocs::malloc:
      return "host pageable";
    case LinearAllocs::hipHostMalloc:
      return "host pinned";
    case LinearAllocs::hipMalloc:
      return "device malloc";
    case LinearAllocs::hipMallocManaged:
      return "managed";
    default:
      return "unknown alloc type";
  }
}
