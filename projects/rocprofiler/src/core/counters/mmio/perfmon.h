/* Copyright (c) 2023 Advanced Micro Devices, Inc.

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

#ifndef SRC_CORE_COUNTERS_PERFMON_H
#define SRC_CORE_COUNTERS_PERFMON_H

#include "rocprofiler.h"
#include "mmio.h"
#include <vector>

namespace rocprofiler {

class PerfMon {
 public:
  virtual ~PerfMon(){};
  virtual void Start() = 0;
  virtual void Stop() = 0;
  virtual void Read(std::vector<rocprofiler_counters_sampler_counter_output_t>& values) = 0;
  virtual void SetCounterNames(std::vector<std::string>& counter_names) {
    counter_names_ = counter_names;
  };
  virtual mmio::mmap_type_t Type() = 0;

 protected:
  std::vector<std::string> counter_names_;
};

}  // namespace rocprofiler

#endif