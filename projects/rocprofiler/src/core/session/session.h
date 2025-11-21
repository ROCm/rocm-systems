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

#ifndef SRC_CORE_SESSION_SESSION_H_
#define SRC_CORE_SESSION_SESSION_H_

#include <hsa/hsa.h>
#include <hsa/hsa_ven_amd_aqlprofile.h>

#include <map>
#include <memory>
#include <mutex>
#include <stack>
#include <string>
#include <variant>
#include <vector>

#include "rocprofiler.h"
#include "src/core/memory/generic_buffer.h"
#include "src/core/session/filter.h"
#include "profiler/profiler.h"
#include "tracer/tracer.h"
#include "att/att.h"
#include "spm/spm.h"
#include "src/pcsampler/session/pc_sampler.h"
#include "counters_sampler.h"

#define ASSERTM(exp, msg) assert(((void)msg, exp))

namespace rocprofiler {

class Session {
 public:
  Session(rocprofiler_replay_mode_t replay_mode, rocprofiler_session_id_t session_id);
  ~Session();
  void DisableTools(rocprofiler_buffer_id_t buffer_id);
  void Start();
  void Terminate();
  rocprofiler_session_id_t GetId();
  bool IsActive();

  void DestroyTracer();

  profiler::Profiler* GetProfiler();
  tracer::Tracer* GetTracer();
  att::AttTracer* GetAttTracer();
  spm::SpmCounters* GetSpmCounter();
  pc_sampler::PCSampler* GetPCSampler();
  CountersSampler* GetCountersSampler();

  // Filter
  rocprofiler_filter_id_t CreateFilter(rocprofiler_filter_kind_t filter_kind,
                                       rocprofiler_filter_data_t filter_data, uint64_t data_count,
                                       rocprofiler_filter_property_t property);
  bool FindFilter(rocprofiler_filter_id_t filter_id);
  void DestroyFilter(rocprofiler_filter_id_t filter_id);
  Filter* GetFilter(rocprofiler_filter_id_t filter_id);
  bool HasFilter();

  bool FindFilterWithKind(rocprofiler_filter_kind_t kind);
  rocprofiler_filter_id_t GetFilterIdWithKind(rocprofiler_filter_kind_t kind);

  std::mutex& GetSessionLock();

  bool CheckFilterBufferSize(rocprofiler_filter_id_t filter_id, rocprofiler_buffer_id_t buffer_id);

  // Buffer
  rocprofiler_buffer_id_t CreateBuffer(rocprofiler_buffer_callback_t buffer_callback,
                                       size_t buffer_size);
  bool FindBuffer(rocprofiler_buffer_id_t buffer_id);
  void DestroyBuffer(rocprofiler_buffer_id_t buffer_id);
  Memory::GenericBuffer* GetBuffer(rocprofiler_buffer_id_t buffer_id);
  bool HasBuffer();

  rocprofiler_status_t startSpm();
  rocprofiler_status_t stopSpm();
  // Range Labels
  void PushRangeLabels(const std::string label);
  bool PopRangeLabels();
  std::string& GetCurrentRangeLabel();

 private:
  rocprofiler_session_id_t session_id_;
  std::atomic<bool> is_active_;
  rocprofiler_replay_mode_t replay_mode_;
  std::mutex session_lock_;

  std::atomic<uint64_t> filters_counter_{1};
  std::mutex filters_lock_;
  std::vector<Filter*> filters_;

  att::AttTracer* att_tracer_ = nullptr;
  bool spm_started_{false};

  profiler::Profiler* profiler_ = nullptr;
  tracer::Tracer* tracer_ = nullptr;
  spm::SpmCounters* spmcounter_ = nullptr;

  pc_sampler::PCSampler* pc_sampler_ = nullptr;
  CountersSampler* counters_sampler_ = nullptr;

  std::atomic<uint64_t> buffers_counter_{1};
  std::mutex buffers_lock_;
  std::map<uint64_t, Memory::GenericBuffer*>* buffers_ = nullptr;
  std::atomic<uint64_t> records_counter_{0};


  std::mutex range_labels_lock_;
  std::stack<std::string> range_labels_;
  std::string current_range_label_;
};

uint64_t GenerateUniqueSessionId();

}  // namespace rocprofiler

#endif  // SRC_CORE_SESSION_SESSION_H_
