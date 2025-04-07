/* Copyright (c) 2022 Advanced Micro Devices, Inc.

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
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#ifndef SRC_CORE_COUNTERS_DERIVED_DERIVED_COUNTER_H_
#define SRC_CORE_COUNTERS_DERIVED_DERIVED_COUNTER_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>

#include "src/core/counters/basic/basic_counter.h"

namespace Counter {

class DerivedCounter : Counter {
 public:
  std::function<uint64_t()> evaluate_metric;
  DerivedCounter(std::string name, std::string description, std::string gpu_name);
  ~DerivedCounter();

  uint64_t getMetricId();
  uint64_t getValue();
  std::map<uint64_t, BasicCounter*>* getAllCounters();
  void addBasicCounter(uint64_t counter_id, BasicCounter* counter);
  BasicCounter* getBasicCounterFromDerived(uint64_t counter_id);

 private:
  uint64_t metric_id_;
  uint64_t value_;
  std::map<uint64_t, BasicCounter*> counters_;
};

static std::map<uint64_t, DerivedCounter> derived_counters;

uint64_t getDerivedCounter(const char* name, const char* gpu_name);

}  // namespace Counter

#endif  // SRC_CORE_COUNTERS_DERIVED_DERIVED_COUNTER_H_
