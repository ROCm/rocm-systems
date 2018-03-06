#ifndef SRC_CORE_TRACKER_H_
#define SRC_CORE_TRACKER_H_

#include <assert.h>
#include <hsa.h>
#include <hsa_ext_amd.h>

#include <list>

#include "inc/rocprofiler.h"
#include "util/exception.h"
#include "util/logger.h"

namespace rocprofiler {

class Tracker {
  public:
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

  Tracker(uint64_t timeout = UINT64_MAX) : timeout_(timeout) {}
  ~Tracker() {
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
  }

  // Add tracker entry
  entry_t* Add(const hsa_agent_t& agent, const hsa_signal_t& orig) {
    entry_t* entry = new entry_t{};
    assert(entry);
    entry->tracker = this;
    entry->it = sig_list_.insert(sig_list_.begin(), entry);

    entry->agent = agent;
    entry->orig = orig;
    hsa_status_t status = hsa_signal_create(1, 0, NULL, &(entry->signal));
    if (status != HSA_STATUS_SUCCESS) EXC_RAISING(status, "hsa_signal_create");

    record_t* record = new record_t{};
    assert(record);
    entry->record = record;
    status = hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP, &record->dispatch);
    if (status != HSA_STATUS_SUCCESS) EXC_RAISING(status, "hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP)");

    hsa_amd_signal_async_handler(entry->signal, HSA_SIGNAL_CONDITION_LT, 1, Handler, entry);
    if (status != HSA_STATUS_SUCCESS) EXC_RAISING(status, "hsa_amd_signal_async_handler");

    return entry;
  }

  private:
  // Delete tracker entry
  void Del(entry_t* entry) {
    hsa_signal_destroy(entry->signal);
    sig_list_.erase(entry->it);
    delete entry;
  } 

  // Handler for packet completion
  static bool Handler(hsa_signal_value_t value, void* arg) {
    entry_t* entry = reinterpret_cast<entry_t*>(arg);
    record_t* record = entry->record;

    hsa_status_t status = hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP, &record->complete);
    if (status != HSA_STATUS_SUCCESS) EXC_RAISING(status, "hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP)");

    hsa_amd_profiling_dispatch_time_t dispatch_time{};
    status = hsa_amd_profiling_get_dispatch_time(entry->agent, entry->signal, &dispatch_time);
    if (status != HSA_STATUS_SUCCESS) EXC_RAISING(status, "hsa_amd_profiling_get_dispatch_time");

    record->begin = dispatch_time.start;
    record->end = dispatch_time.end;

    hsa_signal_t orig = entry->orig;
    if (orig.handle) {
      const hsa_signal_value_t value = hsa_signal_load_relaxed(orig);
      hsa_signal_store_relaxed(orig, value - 1);
    }
    entry->tracker->Del(entry);

    return false;
  }

  // Timeout for wait on destruction
  uint64_t timeout_;
  // Tracked signals list
  sig_list_t sig_list_;
};

} // namespace rocprofiler

#endif // SRC_CORE_TRACKER_H_
