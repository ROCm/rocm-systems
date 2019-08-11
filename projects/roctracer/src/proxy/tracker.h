/******************************************************************************
Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.

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
*******************************************************************************/

#ifndef SRC_PROXY_TRACKER_H_
#define SRC_PROXY_TRACKER_H_

#include <amd_hsa_signal.h>
#include <assert.h>
#include <hsa.h>
#include <hsa_ext_amd.h>

#include <atomic>

#include "util/hsa_rsrc_factory.h"
#include "util/exception.h"
#include "util/logger.h"
#include "core/trace_buffer.h"

namespace proxy {
class Tracker {
  public:
  typedef util::HsaRsrcFactory::timestamp_t timestamp_t;
  typedef roctracer::trace_entry_t entry_t;

  // Add tracker entry
  inline static void Enable(uint32_t type, const hsa_agent_t& agent, const hsa_signal_t& signal, entry_t* entry) {
    hsa_status_t status = HSA_STATUS_ERROR;
    util::HsaRsrcFactory* hsa_rsrc = &(util::HsaRsrcFactory::Instance());

    // Creating a new tracker entry
    entry->type = type;
    entry->agent = agent;
    entry->dev_index = 0; //hsa_rsrc->GetAgentInfo(agent)->dev_index;
    entry->orig = signal;
    entry->dispatch = hsa_rsrc->TimestampNs();
    entry->valid.store(roctracer::TRACE_ENTRY_INIT, std::memory_order_release);

    // Creating a proxy signal
    status = hsa_signal_create(1, 0, NULL, &(entry->signal));
    if (status != HSA_STATUS_SUCCESS) EXC_RAISING(status, "hsa_signal_create");
    status = hsa_amd_signal_async_handler(entry->signal, HSA_SIGNAL_CONDITION_LT, 1, Handler, entry);
    if (status != HSA_STATUS_SUCCESS) EXC_RAISING(status, "hsa_amd_signal_async_handler");
  }

  // Delete tracker entry
  inline static void Disable(entry_t* entry) {
    hsa_signal_destroy(entry->signal);
    entry->valid.store(roctracer::TRACE_ENTRY_INV, std::memory_order_release);
  }

  private:
  // Entry completion
  inline static void Complete(hsa_signal_value_t signal_value, entry_t* entry) {
    // Query begin/end and complete timestamps
    util::HsaRsrcFactory* hsa_rsrc = &(util::HsaRsrcFactory::Instance());
    if (entry->type == roctracer::COPY_ENTRY_TYPE) {
      hsa_amd_profiling_async_copy_time_t async_copy_time{};
      hsa_status_t status = hsa_amd_profiling_get_async_copy_time(entry->signal, &async_copy_time);
      if (status != HSA_STATUS_SUCCESS) EXC_RAISING(status, "hsa_amd_profiling_get_async_copy_time");
      entry->begin = hsa_rsrc->SysclockToNs(async_copy_time.start);
      entry->end = hsa_rsrc->SysclockToNs(async_copy_time.end);
    } else {
      hsa_amd_profiling_dispatch_time_t dispatch_time{};
      hsa_status_t status = hsa_amd_profiling_get_dispatch_time(entry->agent, entry->signal, &dispatch_time);
      if (status != HSA_STATUS_SUCCESS) EXC_RAISING(status, "hsa_amd_profiling_get_dispatch_time");
      entry->begin = hsa_rsrc->SysclockToNs(dispatch_time.start);
      entry->end = hsa_rsrc->SysclockToNs(dispatch_time.end);
      entry->dev_index = (hsa_rsrc->GetAgentInfo(entry->agent))->dev_index;
    }

    entry->complete = hsa_rsrc->TimestampNs();
    entry->valid.store(roctracer::TRACE_ENTRY_COMPL, std::memory_order_release);

    // Original intercepted signal completion
    hsa_signal_t orig = entry->orig;
    if (orig.handle) {
      amd_signal_t* orig_signal_ptr = reinterpret_cast<amd_signal_t*>(orig.handle);
      amd_signal_t* prof_signal_ptr = reinterpret_cast<amd_signal_t*>(entry->signal.handle);
      orig_signal_ptr->start_ts = prof_signal_ptr->start_ts;
      orig_signal_ptr->end_ts = prof_signal_ptr->end_ts;

      const hsa_signal_value_t new_value = hsa_signal_load_relaxed(orig) - 1;
      if (signal_value != new_value) EXC_ABORT(HSA_STATUS_ERROR, "Tracker::Complete bad signal value");
      hsa_signal_store_screlease(orig, signal_value);
    }
    hsa_signal_destroy(entry->signal);
  }

  // Handler for packet completion
  static bool Handler(hsa_signal_value_t signal_value, void* arg) {
    // Acquire entry
    entry_t* entry = reinterpret_cast<entry_t*>(arg);
    while (entry->valid.load(std::memory_order_acquire) != roctracer::TRACE_ENTRY_INIT) sched_yield();

    // Complete entry
    Tracker::Complete(signal_value, entry);

    return false;
  }
};

} // namespace rocprofiler

#endif // SRC_PROXY_TRACKER_H_
