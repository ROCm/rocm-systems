/******************************************************************************
MIT License

Copyright (c) 2018 ROCm Core Technology

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*******************************************************************************/

#ifndef SRC_CORE_TRACKER_H_
#define SRC_CORE_TRACKER_H_

#include <amd_hsa_signal.h>
#include <assert.h>
#include <hsa.h>
#include <hsa_ext_amd.h>

#include <list>
#include <mutex>

#include "inc/rocprofiler.h"
#include "util/exception.h"
#include "util/logger.h"

namespace rocprofiler {

class Tracker {
  public:
  typedef uint64_t timestamp_t;
  typedef long double freq_t;
  typedef std::mutex mutex_t;
  typedef rocprofiler_dispatch_record_t record_t;
  struct entry_t;
  typedef std::list<entry_t*> sig_list_t;
  struct entry_t {
    Tracker* tracker;
    sig_list_t::iterator it;
    hsa_agent_t agent;
    hsa_signal_t orig;
    hsa_signal_t signal;
    record_t* record;
  };

  Tracker(uint64_t timeout = UINT64_MAX) : timeout_(timeout), outstanding(0) {
    timestamp_t timestamp_hz = 0;
    hsa_status_t status = hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &timestamp_hz);
    if (status != HSA_STATUS_SUCCESS) EXC_ABORT(status, "hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY)");
    timestamp_factor_ = (freq_t)1000000000 / (freq_t)timestamp_hz;
  }
  ~Tracker() {
    mutex_.lock();
    for (entry_t* entry : sig_list_) {
      assert(entry != NULL);
      while (1) {
        const hsa_signal_value_t signal_value = hsa_signal_wait_scacquire(
          entry->signal,
          HSA_SIGNAL_CONDITION_LT,
          1,
          timeout_,
          HSA_WAIT_STATE_BLOCKED);
        if (signal_value < 1) break;
        else WARN_LOGGING("tracker timeout");
      }
      Del(entry);
    }
    mutex_.unlock();
  }

  // Add tracker entry
  entry_t* Add(const hsa_agent_t& agent, const hsa_signal_t& orig) {
    hsa_status_t status = HSA_STATUS_ERROR;
    entry_t* entry = new entry_t{};
    assert(entry);
    entry->tracker = this;
    mutex_.lock();
    entry->it = sig_list_.insert(sig_list_.begin(), entry);
    mutex_.unlock();

    entry->agent = agent;
    entry->orig = orig;
    status = hsa_signal_create(1, 0, NULL, &(entry->signal));
    if (status != HSA_STATUS_SUCCESS) EXC_RAISING(status, "hsa_signal_create");

    record_t* record = new record_t{};
    assert(record);
    entry->record = record;

    timestamp_t dispatch_timestamp = 0;
    status = hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP, &dispatch_timestamp);
    if (status != HSA_STATUS_SUCCESS) EXC_RAISING(status, "hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP)");

    record->dispatch = timestamp2ns(dispatch_timestamp);

    status = hsa_amd_signal_async_handler(entry->signal, HSA_SIGNAL_CONDITION_LT, 1, Handler, entry);
    if (status != HSA_STATUS_SUCCESS) EXC_RAISING(status, "hsa_amd_signal_async_handler");

    if (trace_on_) {
      mutex_.lock();
      entry->tracker->outstanding++;
      fprintf(stdout, "Tracker::Add: entry %p, record %p, outst %lu\n", entry, entry->record, entry->tracker->outstanding);
      fflush(stdout);
      mutex_.unlock();
    }

    return entry;
  }

  private:
  // Delete tracker entry
  void Del(entry_t* entry) {
    hsa_signal_destroy(entry->signal);
    mutex_.lock();
    sig_list_.erase(entry->it);
    mutex_.unlock();
    delete entry;
  }

  // Handler for packet completion
  static bool Handler(hsa_signal_value_t value, void* arg) {
    entry_t* entry = reinterpret_cast<entry_t*>(arg);
    record_t* record = entry->record;

    if (trace_on_) {
      mutex_.lock();
      entry->tracker->outstanding--;
      fprintf(stdout, "Tracker::Handler: entry %p, record %p, outst %lu\n", entry, entry->record, entry->tracker->outstanding);
      fflush(stdout);
      mutex_.unlock();
    }

    timestamp_t complete_timestamp = 0;
    hsa_amd_profiling_dispatch_time_t dispatch_time{};

    hsa_status_t status = hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP, &complete_timestamp);
    if (status != HSA_STATUS_SUCCESS) EXC_RAISING(status, "hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP)");
    status = hsa_amd_profiling_get_dispatch_time(entry->agent, entry->signal, &dispatch_time);
    if (status != HSA_STATUS_SUCCESS) EXC_RAISING(status, "hsa_amd_profiling_get_dispatch_time");

    record->complete = entry->tracker->timestamp2ns(complete_timestamp);
    record->begin = entry->tracker->timestamp2ns(dispatch_time.start);
    record->end = entry->tracker->timestamp2ns(dispatch_time.end);

    hsa_signal_t orig = entry->orig;
    if (orig.handle) {
      amd_signal_t* orig_signal_ptr = reinterpret_cast<amd_signal_t*>(orig.handle);
      amd_signal_t* prof_signal_ptr = reinterpret_cast<amd_signal_t*>(entry->signal.handle);
      orig_signal_ptr->start_ts = prof_signal_ptr->start_ts;
      orig_signal_ptr->end_ts = prof_signal_ptr->end_ts;

      const hsa_signal_value_t value = hsa_signal_load_relaxed(orig);
      hsa_signal_store_screlease(orig, value - 1);
    }
    entry->tracker->Del(entry);

    return false;
  }

  inline timestamp_t timestamp2ns(const timestamp_t& timestamp) const {
    const freq_t timestamp_ns = (freq_t)timestamp * timestamp_factor_;
    return (timestamp_t)timestamp_ns;
  }

  // Timestamp frequency factor
  freq_t timestamp_factor_;
  // Timeout for wait on destruction
  timestamp_t timeout_;
  // Tracked signals list
  sig_list_t sig_list_;
  // Inter-thread synchronization
  static mutex_t mutex_;
  // Outstanding dispatches
  uint64_t outstanding;
  // Enable tracing
  static const bool trace_on_ = false;
};

} // namespace rocprofiler

#endif // SRC_CORE_TRACKER_H_
