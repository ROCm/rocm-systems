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

#ifndef SRC_CORE_SESSION_COUNTERS_SAMPLER_H_
#define SRC_CORE_SESSION_COUNTERS_SAMPLER_H_

#include "rocprofiler.h"
#include "src/core/counters/mmio/perfmon.h"
#include <atomic>
#include <thread>
#include <mutex>
#include <vector>

namespace rocprofiler {

class CountersSampler {
 public:
  CountersSampler(rocprofiler_buffer_id_t buffer_id, rocprofiler_filter_id_t filter_id,
                  rocprofiler_session_id_t session_id);
  ~CountersSampler();

  CountersSampler(const CountersSampler&) = delete;
  CountersSampler& operator=(const CountersSampler&) = delete;

  void Start();
  void Stop();
  void AddRecord(rocprofiler_record_counters_sampler_t& record);

 private:
  void SamplerLoop();

  rocprofiler_buffer_id_t buffer_id_;
  rocprofiler_filter_id_t filter_id_;
  rocprofiler_session_id_t session_id_;
  bool pci_system_initialized_{false};
  rocprofiler_counters_sampler_parameters_t params_;
  std::vector<PerfMon*> perfmon_instances_;

  std::atomic<bool> keep_running_{false};
  std::thread sampler_thread_;
};

}  // namespace rocprofiler

#endif