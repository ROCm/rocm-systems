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

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <mutex>
#include <numeric>
#include <vector>

#ifdef _WIN32
#define HIP_INTERNAL_TIMER_EXPORT __declspec(dllexport)
#else
#define HIP_INTERNAL_TIMER_EXPORT __attribute__((visibility("default")))
#endif

namespace hip {
/**
 * @brief Internal timing utility for runtime instrumentation.
 */
class HIP_INTERNAL_TIMER_EXPORT HipInternalTimer {
 public:
  HipInternalTimer() = default;
  ~HipInternalTimer() = default;

  void reserve(size_t count) {
    std::lock_guard<std::mutex> lock(mutex_);
    measurements_.reserve(count);
  }

  void clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    measurements_.clear();
  }

  double getAverage() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (measurements_.empty()) {
      return 0.0;
    }
    double sum = std::accumulate(measurements_.begin(), measurements_.end(), 0.0);
    return sum / static_cast<double>(measurements_.size());
  }

  double getStdDev() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (measurements_.empty()) {
      return 0.0;
    }
    double sum = std::accumulate(measurements_.begin(), measurements_.end(), 0.0);
    double mean = sum / static_cast<double>(measurements_.size());
    double sq_sum = 0.0;
    for (double value : measurements_) {
      double diff = value - mean;
      sq_sum += diff * diff;
    }
    double variance = sq_sum / static_cast<double>(measurements_.size());
    return std::sqrt(variance);
  }

  void record(double duration) {
    std::lock_guard<std::mutex> lock(mutex_);
    measurements_.push_back(duration);
  }

  void exportAsCSV(const char* filename) const {
    if (filename == nullptr) {
      return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    std::ofstream out(filename);
    if (!out.is_open()) {
      return;
    }
    for (size_t i = 0; i < measurements_.size(); ++i) {
      if (i > 0) {
        out << ",";
      }
      out << measurements_[i];
    }
    out << "\n";
  }

 private:
  mutable std::mutex mutex_;
  std::vector<double> measurements_;
};

HIP_INTERNAL_TIMER_EXPORT HipInternalTimer& getHipInternalTimer();
extern HIP_INTERNAL_TIMER_EXPORT HipInternalTimer* g_hipInternalTimer;
}  // namespace hip
