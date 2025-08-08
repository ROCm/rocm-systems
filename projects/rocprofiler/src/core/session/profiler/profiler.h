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

#ifndef SRC_TOOLS_PROFILER_PROFILER_H_
#define SRC_TOOLS_PROFILER_PROFILER_H_

#include <hsa/hsa_ven_amd_aqlprofile.h>

#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>
#include "src/core/hsa/packets/packets_generator.h"
#include "rocprofiler.h"
#include "src/core/counters/basic/basic_counter.h"
#include "src/core/counters/metrics/eval_metrics.h"

typedef void (*rocprofiler_add_profiler_record_t)(rocprofiler_record_profiler_t&& record,
                                                  rocprofiler_session_id_t session_id);

typedef rocprofiler_timestamp_t (*rocprofiler_get_timestamp_t)();

namespace rocprofiler {

typedef struct {
  uint64_t kernel_descriptor;
  hsa_signal_t original_signal;
  hsa_signal_t new_signal;
  rocprofiler_session_id_t session_id;
  rocprofiler_buffer_id_t buffer_id;
  uint64_t counters_count;
  std::unique_ptr<::Packet::AQLPacketProfile> profile;
  rocprofiler_kernel_properties_t kernel_properties;
  uint32_t thread_id;
  uint64_t queue_index;
  uint64_t correlation_id;
} pending_signal_t;

typedef std::unique_ptr<pending_signal_t> pending_signal_ptr_t;

namespace profiler {

uint64_t GetCounterID(std::string& counter_name);

class Profiler {
 public:
  Profiler(rocprofiler_buffer_id_t buffer_id, rocprofiler_filter_id_t filter_id,
           rocprofiler_session_id_t session_id);
  ~Profiler();

  void AddPendingSignals(uint32_t writer_id, uint64_t kernel_object,
                         const hsa_signal_t& original_completion_signal,
                         const hsa_signal_t& new_completion_signal,
                         rocprofiler_session_id_t session_id, rocprofiler_buffer_id_t buffer_id,
                         uint64_t session_data_count,
                         std::unique_ptr<Packet::AQLPacketProfile>&& profile,
                         rocprofiler_kernel_properties_t kernel_properties, uint32_t thread_id,
                         uint64_t queue_index, uint64_t correlation_id);

  std::vector<pending_signal_ptr_t> MovePendingSignals(uint32_t writer_id);
  void WaitForPendingAndDestroy();

  void AddCounterName(rocprofiler_counter_id_t handler, std::string counter_name);
  void AddCounterName(std::string& counter_name);
  std::string& GetCounterName(rocprofiler_counter_id_t handler);

  bool FindCounter(rocprofiler_counter_id_t counter_id);
  size_t GetCounterInfoSize(rocprofiler_counter_info_kind_t kind,
                            rocprofiler_counter_id_t counter_id);
  const char* GetCounterInfo(rocprofiler_counter_info_kind_t kind,
                             rocprofiler_counter_id_t counter_id);

  void StartReplayPass(rocprofiler_session_id_t session_id);
  void EndReplayPass();
  bool HasActivePass();
  std::atomic<bool> bIsSessionDestroying{false};

 private:
  std::mutex counter_names_lock_;
  std::map<uint64_t, std::string> counter_names_;
  rocprofiler_get_timestamp_t get_timestamp_fn_;
  rocprofiler_buffer_id_t buffer_id_;
  rocprofiler_filter_id_t filter_id_;
  rocprofiler_session_id_t session_id_;

  std::mutex sessions_pending_signals_lock_;
  std::map<uint32_t, std::vector<pending_signal_ptr_t>> sessions_pending_signals_{};
  std::condition_variable has_session_pending_cv;
};

}  // namespace profiler
}  // namespace rocprofiler

#endif  // SRC_TOOLS_PROFILER_PROFILER_H_
