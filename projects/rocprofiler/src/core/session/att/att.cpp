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

#include "att.h"
#include <cassert>
#include <atomic>

namespace rocprofiler {

namespace att {

AttTracer::AttTracer(rocprofiler_buffer_id_t buffer_id, rocprofiler_filter_id_t filter_id,
                     rocprofiler_session_id_t session_id)
    : buffer_id_(buffer_id), filter_id_(filter_id), session_id_(session_id) {}

void AttTracer::AddPendingSignals(
    uint32_t writer_id, uint64_t kernel_object, const hsa_signal_t& original_completion_signal,
    const hsa_signal_t& new_completion_signal, rocprofiler_session_id_t session_id,
    rocprofiler_buffer_id_t buffer_id, hsa_ven_amd_aqlprofile_profile_t* profile,
    rocprofiler_kernel_properties_t kernel_properties, uint32_t thread_id, uint64_t queue_index) {
  std::lock_guard<std::mutex> lock(sessions_pending_signals_lock_);
  if (sessions_pending_signals_.find(writer_id) == sessions_pending_signals_.end())
    sessions_pending_signals_.emplace(writer_id, std::vector<att_pending_signal_t>());
  sessions_pending_signals_.at(writer_id).emplace_back(att_pending_signal_t{
      kernel_object, original_completion_signal, new_completion_signal, session_id_, buffer_id,
      profile, kernel_properties, thread_id, queue_index});
  std::atomic_thread_fence(std::memory_order_release);
}

const std::vector<att_pending_signal_t>& AttTracer::GetPendingSignals(uint32_t writer_id) {
  std::atomic_thread_fence(std::memory_order_acquire);
  std::lock_guard<std::mutex> lock(sessions_pending_signals_lock_);
  assert(sessions_pending_signals_.find(writer_id) != sessions_pending_signals_.end() &&
         "writer_id is not found in the pending_signals");
  return sessions_pending_signals_.at(writer_id);
}

}  // namespace att

}  // namespace rocprofiler
