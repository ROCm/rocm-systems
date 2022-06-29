/* Copyright (c) 2018-2022 Advanced Micro Devices, Inc.

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

#ifndef SRC_CORE_TRACKER_H_
#define SRC_CORE_TRACKER_H_

#include <assert.h>
#include <hsa/amd_hsa_signal.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <atomic>

#include "exception.h"
#include "util/logger.h"

namespace roctracer {
class Tracker {
 public:
  enum { ENTRY_INV = 0, ENTRY_INIT = 1, ENTRY_COMPL = 2 };

  enum entry_type_t {
    DFLT_ENTRY_TYPE = 0,
    API_ENTRY_TYPE = 1,
    COPY_ENTRY_TYPE = 2,
    KERNEL_ENTRY_TYPE = 3,
    NUM_ENTRY_TYPE = 4
  };

  struct entry_t {
    std::atomic<uint32_t> valid;
    entry_type_t type;
    uint64_t correlation_id;
    roctracer_timestamp_t begin;  // begin timestamp, ns
    roctracer_timestamp_t end;    // end timestamp, ns
    hsa_agent_t agent;
    uint32_t dev_index;
    hsa_signal_t orig;
    hsa_signal_t signal;
    void (*handler)(const entry_t*);
    MemoryPool* pool;
    union {
      struct {
      } copy;
      struct {
        const char* name;
        hsa_agent_t agent;
        uint32_t tid;
      } kernel;
    };
  };

  // Add tracker entry
  inline static void Enable(entry_type_t type, const hsa_agent_t& agent, const hsa_signal_t& signal,
                            entry_t* entry) {
    hsa_status_t status = HSA_STATUS_ERROR;

    // Creating a new tracker entry
    entry->type = type;
    entry->agent = agent;
    entry->dev_index = 0;  // hsa_rsrc->GetAgentInfo(agent)->dev_index;
    entry->orig = signal;
    entry->valid.store(ENTRY_INIT, std::memory_order_release);

    // Creating a proxy signal
    status = hsa_signal_create(1, 0, NULL, &(entry->signal));
    if (status != HSA_STATUS_SUCCESS) FATAL_LOGGING("hsa_signal_create failed");
    status =
        hsa_amd_signal_async_handler(entry->signal, HSA_SIGNAL_CONDITION_LT, 1, Handler, entry);
    if (status != HSA_STATUS_SUCCESS) FATAL_LOGGING("hsa_amd_signal_async_handler failed");
  }

  // Delete tracker entry
  inline static void Disable(entry_t* entry) {
    hsa_signal_destroy(entry->signal);
    entry->valid.store(ENTRY_INV, std::memory_order_release);
  }

 private:
  // Entry completion
  inline static void Complete(hsa_signal_value_t signal_value, entry_t* entry) {
    static roctracer_timestamp_t sysclock_period = []() {
      uint64_t sysclock_hz = 0;
      hsa_status_t status = hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &sysclock_hz);
      if (status != HSA_STATUS_SUCCESS) FATAL_LOGGING("hsa_system_get_info failed");
      return (uint64_t)1000000000 / sysclock_hz;
    }();

    if (entry->type == COPY_ENTRY_TYPE) {
      hsa_amd_profiling_async_copy_time_t async_copy_time{};
      hsa_status_t status = hsa_amd_profiling_get_async_copy_time(entry->signal, &async_copy_time);
      if (status != HSA_STATUS_SUCCESS)
        FATAL_LOGGING("hsa_amd_profiling_get_async_copy_time failed");
      entry->begin = async_copy_time.start * sysclock_period;
      entry->end = async_copy_time.end * sysclock_period;
    } else {
      assert(false && "should not reach here");
    }

    hsa_signal_t orig = entry->orig;
    hsa_signal_t signal = entry->signal;

    // Releasing completed entry
    entry->valid.store(ENTRY_COMPL, std::memory_order_release);

    assert(entry->handler != nullptr);
    entry->handler(entry);

    // Original intercepted signal completion
    if (orig.handle) {
      amd_signal_t* orig_signal_ptr = reinterpret_cast<amd_signal_t*>(orig.handle);
      amd_signal_t* prof_signal_ptr = reinterpret_cast<amd_signal_t*>(signal.handle);
      orig_signal_ptr->start_ts = prof_signal_ptr->start_ts;
      orig_signal_ptr->end_ts = prof_signal_ptr->end_ts;

      [[maybe_unused]] const hsa_signal_value_t new_value = hsa_signal_load_relaxed(orig) - 1;
      assert(signal_value == new_value && "Tracker::Complete bad signal value");
      hsa_signal_store_screlease(orig, signal_value);
    }
    hsa_signal_destroy(signal);
    delete entry;
  }

  // Handler for packet completion
  static bool Handler(hsa_signal_value_t signal_value, void* arg) {
    // Acquire entry
    entry_t* entry = reinterpret_cast<entry_t*>(arg);
    while (entry->valid.load(std::memory_order_acquire) != ENTRY_INIT) sched_yield();

    // Complete entry
    Tracker::Complete(signal_value, entry);
    return false;
  }
};

}  // namespace roctracer

#endif  // SRC_CORE_TRACKER_H_
