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

#include "session.h"

#include <string.h>

#include <atomic>
#include <cassert>
#include <cstdint>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "rocprofiler.h"
#include "src/pcsampler/session/pc_sampler.h"
#include "src/utils/helper.h"
#include "src/core/hsa/queues/queue.h"

namespace rocprofiler {

Session::Session(rocprofiler_replay_mode_t replay_mode, rocprofiler_session_id_t session_id)
    : session_id_(session_id), is_active_(false), replay_mode_(replay_mode) {
  buffers_ = new std::map<uint64_t, Memory::GenericBuffer*>();
}

Session::~Session()
{
    std::lock_guard<std::mutex> lock(session_lock_);
    if (FindFilterWithKind(ROCPROFILER_SPM_COLLECTION) && spmcounter_)
      delete spmcounter_;

    if (FindFilterWithKind(ROCPROFILER_API_TRACE) && tracer_)
    {
      delete tracer_;
      tracer_ = nullptr;
    }
    if (FindFilterWithKind(ROCPROFILER_PC_SAMPLING_COLLECTION) && pc_sampler_)
    {
      delete pc_sampler_;
      pc_sampler_ = nullptr;
    }

    if (FindFilterWithKind(ROCPROFILER_COUNTERS_SAMPLER) && counters_sampler_)
    {
      delete counters_sampler_;
      counters_sampler_ = nullptr;
    }
    if ((FindFilterWithKind(ROCPROFILER_DISPATCH_TIMESTAMPS_COLLECTION) ||
         FindFilterWithKind(ROCPROFILER_COUNTERS_COLLECTION)) && profiler_)
    {
      rocprofiler::queue::Queue::ResetSessionID();
      delete profiler_;
      profiler_ = nullptr;
    }
    if (FindFilterWithKind(ROCPROFILER_ATT_TRACE_COLLECTION) && att_tracer_)
    {
      delete att_tracer_;
      att_tracer_ = nullptr;
    }
    for (auto& filter : filters_)
    if (filter)
      delete filter;

    filters_.clear();
    for (auto& buffer : *buffers_)
    {
      buffer.second->Flush();
      if (buffer.second) delete buffer.second;
    }
    buffers_->clear();
    if (buffers_) delete buffers_;
}

void Session::DisableTools(rocprofiler_buffer_id_t buffer_id) {
  if ((FindFilterWithKind(ROCPROFILER_COUNTERS_COLLECTION) &&
       GetFilter(GetFilterIdWithKind(ROCPROFILER_COUNTERS_COLLECTION))->GetBufferId().value ==
           buffer_id.value) ||
      (FindFilterWithKind(ROCPROFILER_DISPATCH_TIMESTAMPS_COLLECTION) &&
       GetFilter(GetFilterIdWithKind(ROCPROFILER_DISPATCH_TIMESTAMPS_COLLECTION))
               ->GetBufferId()
               .value == buffer_id.value)) {
  }
  if (FindFilterWithKind(ROCPROFILER_API_TRACE) &&
      GetFilter(GetFilterIdWithKind(ROCPROFILER_API_TRACE))->GetBufferId().value ==
          buffer_id.value)
  {
    if (tracer_)
      tracer_->DisableRoctracer();
  }
}

void Session::Start()
{
  std::lock_guard<std::mutex> lock(session_lock_);
  if (is_active_) return;

  if (!profiler_)
  {
    if (FindFilterWithKind(ROCPROFILER_DISPATCH_TIMESTAMPS_COLLECTION))
    {
      profiler_ = new profiler::Profiler(
          GetFilter(GetFilterIdWithKind(ROCPROFILER_DISPATCH_TIMESTAMPS_COLLECTION))
              ->GetBufferId(),
          GetFilter(GetFilterIdWithKind(ROCPROFILER_DISPATCH_TIMESTAMPS_COLLECTION))->GetId(),
          session_id_);
    }
    if (FindFilterWithKind(ROCPROFILER_COUNTERS_COLLECTION))
    {
      profiler_ = new profiler::Profiler(
          GetFilter(GetFilterIdWithKind(ROCPROFILER_COUNTERS_COLLECTION))->GetBufferId(),
          GetFilter(GetFilterIdWithKind(ROCPROFILER_COUNTERS_COLLECTION))->GetId(), session_id_);
    }
  }
  else
  {
    rocprofiler::queue::Queue::ResetSessionID(session_id_);
  }

  if (FindFilterWithKind(ROCPROFILER_ATT_TRACE_COLLECTION) && !att_tracer_)
  {
    att_tracer_ = new att::AttTracer(
          GetFilter(GetFilterIdWithKind(ROCPROFILER_ATT_TRACE_COLLECTION))->GetBufferId(),
          GetFilter(GetFilterIdWithKind(ROCPROFILER_ATT_TRACE_COLLECTION))->GetId(), session_id_);
  }

  if (FindFilterWithKind(ROCPROFILER_SPM_COLLECTION)) {
    if (!spm_started_)
    {
      rocprofiler_spm_parameter_t* spmparameter =
          GetFilter(GetFilterIdWithKind(ROCPROFILER_SPM_COLLECTION))->GetSpmParameterData();
      spmcounter_ = new spm::SpmCounters(
          GetFilter(GetFilterIdWithKind(ROCPROFILER_SPM_COLLECTION))->GetBufferId(),
          GetFilter(GetFilterIdWithKind(ROCPROFILER_SPM_COLLECTION))->GetId(), spmparameter,
          session_id_);
    }
    if (!profiler_)
    {
      profiler_ = new profiler::Profiler(
          GetFilter(GetFilterIdWithKind(ROCPROFILER_SPM_COLLECTION))->GetBufferId(),
          GetFilter(GetFilterIdWithKind(ROCPROFILER_SPM_COLLECTION))->GetId(), session_id_);
    }
    startSpm();
  }

  if (FindFilterWithKind(ROCPROFILER_API_TRACE))
  {
    std::vector<rocprofiler_tracer_activity_domain_t> domains =
        GetFilter(GetFilterIdWithKind(ROCPROFILER_API_TRACE))->GetTraceData();
    if (!tracer_)
    {
      tracer_ = new tracer::Tracer(
          session_id_,
          (GetFilter(GetFilterIdWithKind(ROCPROFILER_API_TRACE))->HasCallback()
                ? GetFilter(GetFilterIdWithKind(ROCPROFILER_API_TRACE))->GetCallback()
                : nullptr),
          GetFilter(GetFilterIdWithKind(ROCPROFILER_API_TRACE))->GetBufferId(), domains);
    }
    tracer_->StartRoctracer();
  }

  if (FindFilterWithKind(ROCPROFILER_PC_SAMPLING_COLLECTION))
  {
    if (!pc_sampler_)
    {
      pc_sampler_ = new pc_sampler::PCSampler(
          GetFilter(GetFilterIdWithKind(ROCPROFILER_PC_SAMPLING_COLLECTION))->GetBufferId(),
          GetFilter(GetFilterIdWithKind(ROCPROFILER_PC_SAMPLING_COLLECTION))->GetId(),
          session_id_);
    }
    pc_sampler_->Start();
  }

  if (FindFilterWithKind(ROCPROFILER_COUNTERS_SAMPLER))
  {
    if (!counters_sampler_)
    {
      counters_sampler_ = new CountersSampler(
          GetFilter(GetFilterIdWithKind(ROCPROFILER_COUNTERS_SAMPLER))->GetBufferId(),
          GetFilter(GetFilterIdWithKind(ROCPROFILER_COUNTERS_SAMPLER))->GetId(), session_id_);
    }
    counters_sampler_->Start();
  }

  if (profiler_) profiler_->bIsSessionDestroying.store(false);
  is_active_ = true;
}

void Session::Terminate()
{
  if (!is_active_.load(std::memory_order_acquire))
    return;

  rocprofiler::queue::Queue::ResetSessionID();
  if (GetProfiler())
    GetProfiler()->WaitForPendingAndDestroy();
  if (GetAttTracer())
    GetAttTracer()->WaitForPendingAndDestroy();
  Packet::AQLPacketProfile::WaitForProfileDeletion();

  std::lock_guard<std::mutex> lock(session_lock_);
  if (FindFilterWithKind(ROCPROFILER_SPM_COLLECTION)) {
    { stopSpm(); }
  }
  if (FindFilterWithKind(ROCPROFILER_API_TRACE))
  {
    std::vector<rocprofiler_tracer_activity_domain_t> domains =
        GetFilter(GetFilterIdWithKind(ROCPROFILER_API_TRACE))->GetTraceData();
    if (tracer_)
      tracer_->StopRoctracer();
  }
  if (FindFilterWithKind(ROCPROFILER_PC_SAMPLING_COLLECTION) && pc_sampler_)
    pc_sampler_->Stop();

  if (FindFilterWithKind(ROCPROFILER_COUNTERS_SAMPLER) && counters_sampler_)
    counters_sampler_->Stop();

  for (auto& buffer : *buffers_)
    buffer.second->Flush();

  is_active_.store(false, std::memory_order_relaxed);
}

rocprofiler_session_id_t Session::GetId() { return session_id_; }
bool Session::IsActive() { return is_active_; }

profiler::Profiler* Session::GetProfiler() { return profiler_; }
att::AttTracer* Session::GetAttTracer() { return att_tracer_; }
tracer::Tracer* Session::GetTracer() { return tracer_; }
spm::SpmCounters* Session::GetSpmCounter() { return spmcounter_; }
pc_sampler::PCSampler* Session::GetPCSampler() { return pc_sampler_; }
CountersSampler* Session::GetCountersSampler() { return counters_sampler_; }

rocprofiler_filter_id_t Session::CreateFilter(rocprofiler_filter_kind_t filter_kind,
                                              rocprofiler_filter_data_t filter_data,
                                              uint64_t data_count,
                                              rocprofiler_filter_property_t property) {
  rocprofiler_filter_id_t id =
      rocprofiler_filter_id_t{filters_counter_.fetch_add(1, std::memory_order_release)};
  {
    std::lock_guard<std::mutex> lock(filters_lock_);
    filters_.emplace_back(new Filter{id, filter_kind, filter_data, data_count});
    filters_.back()->SetProperty(property);
  }
  return id;
}

bool Session::FindFilter(rocprofiler_filter_id_t filter_id) {
  {
    std::lock_guard<std::mutex> lock(filters_lock_);
    for (auto& filter : filters_) {
      if (filter->GetId().value == filter_id.value) return true;
    }
  }
  return false;
}

void Session::DestroyFilter(rocprofiler_filter_id_t filter_id) {
  {
    std::vector<Filter*>::iterator filter;
    std::lock_guard<std::mutex> lock(filters_lock_);
    for (filter = filters_.begin(); filter != filters_.end(); ++filter) {
      if ((*filter) && (*filter)->GetId().value == filter_id.value) filters_.erase(filter);
    }
  }
}

Filter* Session::GetFilter(rocprofiler_filter_id_t filter_id) {
  {
    std::lock_guard<std::mutex> lock(filters_lock_);
    for (auto& filter : filters_) {
      if (filter->GetId().value == filter_id.value) return filter;
    }
  }
  fatal("Filter is not found!");
}

bool Session::CheckFilterBufferSize(rocprofiler_filter_id_t filter_id,
                                    rocprofiler_buffer_id_t buffer_id) {
  // TODO(aelwazir): To be implemented
  return true;
}

bool Session::HasFilter() { return filters_.size() > 0; }

bool Session::FindFilterWithKind(rocprofiler_filter_kind_t kind) {
  {
    std::lock_guard<std::mutex> lock(filters_lock_);
    for (auto& filter : filters_) {
      if (filter->GetKind() == kind) return true;
    }
  }
  return false;
}
rocprofiler_filter_id_t Session::GetFilterIdWithKind(rocprofiler_filter_kind_t kind) {
  {
    std::lock_guard<std::mutex> lock(filters_lock_);
    for (auto& filter : filters_) {
      if (filter->GetKind() == kind) return filter->GetId();
    }
  }
  return rocprofiler_filter_id_t{0};
}

bool Session::HasBuffer() { return buffers_->size() > 0; }

rocprofiler_buffer_id_t Session::CreateBuffer(rocprofiler_buffer_callback_t buffer_callback,
                                              size_t buffer_size) {
  rocprofiler_buffer_id_t id =
      rocprofiler_buffer_id_t{buffers_counter_.fetch_add(1, std::memory_order_release)};
  {
    std::lock_guard<std::mutex> lock(buffers_lock_);
    buffers_->emplace(id.value,
                      new Memory::GenericBuffer(session_id_, id, buffer_size, buffer_callback));
  }
  return id;
}

bool Session::FindBuffer(rocprofiler_buffer_id_t buffer_id) {
  {
    std::lock_guard<std::mutex> lock(buffers_lock_);
    return buffers_->find(buffer_id.value) != buffers_->end();
  }
}

void Session::DestroyTracer() { /* tracer_.reset(); */
}

void Session::DestroyBuffer(rocprofiler_buffer_id_t buffer_id) {
  {
    std::lock_guard<std::mutex> lock(filters_lock_);
    delete buffers_->at(buffer_id.value);
    buffers_->erase(buffer_id.value);
    // if (buffers_.find(buffer_id.value) != buffers_.end() &&
    // buffers_.at(buffer_id.value)->IsValid())
    //   buffers_.at(buffer_id.value).reset();
  }
}

rocprofiler_status_t Session::startSpm() {
  if (spmcounter_) {
    spm_started_ = true;
    return spmcounter_->startSpm();
  } else {
    std::cout << "Apply the SPM Filter" << std::endl;
    return ROCPROFILER_STATUS_ERROR;
  }
}

rocprofiler_status_t Session::stopSpm() {
  if (spmcounter_ && spm_started_) {
    spm_started_ = false;
    return spmcounter_->stopSpm();
  } else {
    std::cout << "SPM not started" << std::endl;
    return ROCPROFILER_STATUS_ERROR;
  }
}

Memory::GenericBuffer* Session::GetBuffer(rocprofiler_buffer_id_t buffer_id) {
  {
    std::lock_guard<std::mutex> lock(buffers_lock_);
    return buffers_->at(buffer_id.value);
  }
}

void Session::PushRangeLabels(const std::string label) {
  {
    std::lock_guard<std::mutex> lock(range_labels_lock_);
    range_labels_.push(label);
  }
  current_range_label_ = label;
}
bool Session::PopRangeLabels() {
  {
    std::lock_guard<std::mutex> lock(range_labels_lock_);
    if (range_labels_.empty()) {
      return false;
    }
    range_labels_.pop();
  }
  current_range_label_ = "";
  return true;
}
std::string& Session::GetCurrentRangeLabel() { return current_range_label_; }

std::mutex& Session::GetSessionLock() { return session_lock_; }

static std::atomic<uint64_t> SESSION_COUNTER{1};

// use some util function to generate a unique id
uint64_t GenerateUniqueSessionId() {
  return SESSION_COUNTER.fetch_add(1, std::memory_order_release);
}

}  // namespace rocprofiler
