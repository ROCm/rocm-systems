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

#include "profiler.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stack>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "src/core/counters/basic/basic_counter.h"
#include "src/utils/helper.h"
#include "src/utils/logger.h"

#define ASSERTM(exp, msg) assert(((void)msg, exp))

namespace rocprofiler {
namespace profiler {

uint64_t GetCounterID(std::string& counter_name) {
  static auto counter_hash_fn = std::hash<std::string>{};
  return counter_hash_fn(counter_name);
}

Profiler::Profiler(rocprofiler_buffer_id_t buffer_id, rocprofiler_filter_id_t filter_id,
                   rocprofiler_session_id_t session_id)
    : buffer_id_(buffer_id), filter_id_(filter_id), session_id_(session_id) {}

Profiler::~Profiler() {}

void Profiler::AddCounterName(rocprofiler_counter_id_t counter_id, std::string counter_name) {
  std::lock_guard<std::mutex> lock(counter_names_lock_);
  counter_names_.emplace(counter_id.handle, counter_name);
}

void Profiler::AddCounterName(std::string& counter_name) {
  std::lock_guard<std::mutex> lock(counter_names_lock_);
  counter_names_.emplace(GetCounterID(counter_name), counter_name);
}

std::string& Profiler::GetCounterName(rocprofiler_counter_id_t counter_id) {
  std::lock_guard<std::mutex> lock(counter_names_lock_);
  auto it = counter_names_.find(counter_id.handle);
  ASSERTM(it != counter_names_.end(), "Error: couldn't find kernel name with given descriptor!");
  return it->second;
}

bool Profiler::FindCounter(rocprofiler_counter_id_t counter_id) {
  std::lock_guard<std::mutex> lock(counter_names_lock_);
  return counter_names_.find(counter_id.handle) != counter_names_.end();
}
size_t Profiler::GetCounterInfoSize(rocprofiler_counter_info_kind_t kind,
                                    rocprofiler_counter_id_t counter_id) {
  switch (kind) {
    case ROCPROFILER_COUNTER_NAME: {
      std::lock_guard<std::mutex> lock(counter_names_lock_);
      return counter_names_.at(counter_id.handle).size();
      break;
    }

    default:
      warning("Not yet Supported!");
      break;
  }
  return 0;
}

const char* Profiler::GetCounterInfo(rocprofiler_counter_info_kind_t kind,
                                     rocprofiler_counter_id_t counter_id) {
  switch (kind) {
    case ROCPROFILER_COUNTER_NAME: {
      std::lock_guard<std::mutex> lock(counter_names_lock_);
      return counter_names_.at(counter_id.handle).c_str();
      break;
    }

    default:
      warning("Not yet Supported!");
      break;
  }
  return nullptr;
}

void Profiler::StartReplayPass(rocprofiler_session_id_t session_id) {
  warning("Not yet supported!");
}
void Profiler::EndReplayPass() { warning("Not yet supported!"); }
bool Profiler::HasActivePass() {
  warning("Not yet supported!");
  return true;
}

void Profiler::AddPendingSignals(
    uint32_t writer_id, uint64_t kernel_object, const hsa_signal_t& original_completion_signal,
    const hsa_signal_t& new_completion_signal, rocprofiler_session_id_t session_id,
    rocprofiler_buffer_id_t buffer_id,
    uint64_t session_data_count, std::unique_ptr<Packet::AQLPacketProfile>&& profile,
    rocprofiler_kernel_properties_t kernel_properties, uint32_t thread_id, uint64_t queue_index,
    uint64_t correlation_id)
{
  std::lock_guard<std::mutex> lock(sessions_pending_signals_lock_);

  if (sessions_pending_signals_.find(writer_id) == sessions_pending_signals_.end())
    sessions_pending_signals_.emplace(writer_id, std::vector<pending_signal_ptr_t>{});

  sessions_pending_signals_.at(writer_id).emplace_back(
    new pending_signal_t{
      kernel_object, original_completion_signal, new_completion_signal,
      session_id, buffer_id, session_data_count, std::move(profile),
      kernel_properties, thread_id, queue_index, correlation_id
    }
  );
}

std::vector<pending_signal_ptr_t> Profiler::MovePendingSignals(uint32_t writer_id)
{
  std::lock_guard<std::mutex> lock(sessions_pending_signals_lock_);

  auto it = sessions_pending_signals_.find(writer_id);
  if (it == sessions_pending_signals_.end())
    return {};

  auto move_pending = std::move(it->second);
  sessions_pending_signals_.erase(writer_id);
  if (bIsSessionDestroying.load() && sessions_pending_signals_.size() == 0)
    has_session_pending_cv.notify_all();

  return move_pending;
}

void Profiler::WaitForPendingAndDestroy()
{
  std::unique_lock<std::mutex> lk(sessions_pending_signals_lock_);
  bIsSessionDestroying.store(true);
  if (sessions_pending_signals_.size() == 0)
    return;

  has_session_pending_cv.wait_for(lk, std::chrono::seconds(2), [this] () {
    return this->sessions_pending_signals_.size() == 0;
  });
}

}  // namespace profiler
}  // namespace rocprofiler
