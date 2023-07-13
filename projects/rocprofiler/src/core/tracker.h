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

#ifndef SRC_CORE_TRACKER_H_
#define SRC_CORE_TRACKER_H_

#include <hsa/amd_hsa_signal.h>
#include <assert.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <atomic>
#include <list>
#include <mutex>

#include "util/hsa_rsrc_factory.h"
#include "rocprofiler.h"
#include "util/exception.h"
#include "util/logger.h"

namespace rocprofiler {

class Tracker {
 public:
  typedef std::mutex mutex_t;
  typedef util::HsaRsrcFactory::timestamp_t timestamp_t;
  typedef rocprofiler_dispatch_record_t record_t;
  struct entry_t;
  typedef std::list<entry_t*> sig_list_t;
  typedef sig_list_t::iterator sig_list_it_t;
  typedef uint64_t counter_t;

  struct entry_t {
    counter_t index;
    std::atomic<bool> valid;
    Tracker* tracker;
    sig_list_t::iterator it;
    hsa_agent_t agent;
    hsa_signal_t orig;
    hsa_signal_t signal;
    record_t* record;
    std::atomic<void*> handler;
    void* arg;
    bool is_context;
    bool is_memcopy;
    bool is_proxy;
  };

  static Tracker* Create() {
    std::lock_guard<mutex_t> lck(glob_mutex_);
    Tracker* obj = instance_.load(std::memory_order_relaxed);
    if (obj == NULL) {
      obj = new Tracker;
      if (obj == NULL) EXC_ABORT(HSA_STATUS_ERROR, "Tracker creation failed");
      instance_.store(obj, std::memory_order_release);
    }
    return obj;
  }

  static Tracker& Instance() {
    Tracker* obj = instance_.load(std::memory_order_acquire);
    if (obj == NULL) obj = Create();
    return *obj;
  }

  static void Destroy() {
    std::lock_guard<mutex_t> lck(glob_mutex_);
    if (instance_ != NULL) delete instance_.load();
    instance_ = NULL;
  }

  // Add tracker entry
  entry_t* Alloc(const hsa_agent_t& agent, const hsa_signal_t& orig, bool proxy = true) {
    hsa_status_t status = HSA_STATUS_ERROR;

    // Creating a new tracker entry
    entry_t* entry = new entry_t{};
    assert(entry);
    entry->tracker = this;
    entry->agent = agent;
    entry->orig = orig;

    // Creating a record with the dispatch timestamps
    record_t* record = new record_t{};
    assert(record);
    record->dispatch = hsa_rsrc_->TimestampNs();
    entry->record = record;

    // Creating a proxy signal
    if (proxy) {
      entry->is_proxy = true;
      const hsa_signal_value_t signal_value =
          (orig.handle) ? hsa_api_.hsa_signal_load_relaxed(orig) : 1;
      status = hsa_api_.hsa_signal_create(signal_value, 0, NULL, &(entry->signal));
      if (status != HSA_STATUS_SUCCESS) EXC_RAISING(status, "hsa_signal_create");
      status = hsa_api_.hsa_amd_signal_async_handler(entry->signal, HSA_SIGNAL_CONDITION_LT,
                                                     signal_value, Handler, entry);
      if (status != HSA_STATUS_SUCCESS) EXC_RAISING(status, "hsa_amd_signal_async_handler");
    }

    // Adding antry to the list
    mutex_.lock();
    entry->it = sig_list_.insert(sig_list_.end(), entry);
    entry->index = counter_++;
    mutex_.unlock();

    return entry;
  }

  void SetHandler(entry_t* entry, Group* group) {
    hsa_signal_t& dispatch_signal = group->GetDispatchSignal();
    hsa_signal_t& handler_signal = group->GetBarrierSignal();
    entry->signal = dispatch_signal;
    hsa_status_t status = hsa_api_.hsa_amd_signal_async_handler(
        handler_signal, HSA_SIGNAL_CONDITION_LT, 1, Handler, entry);
    if (status != HSA_STATUS_SUCCESS) EXC_RAISING(status, "hsa_amd_signal_async_handler");
  }

  // Delete tracker entry
  void Delete(entry_t* entry) {
    if (entry->is_proxy && entry->signal.handle) hsa_api_.hsa_signal_destroy(entry->signal);
    mutex_.lock();
    sig_list_.erase(entry->it);
    mutex_.unlock();
    delete entry;
  }

  // Enable tracker entry
  void Enable(entry_t* entry, void* handler, void* arg) {
    // Set entry handler and release the entry
    entry->arg = arg;
    entry->handler.store(handler, std::memory_order_release);

    // Debug trace
    if (trace_on_) {
      auto outstanding = outstanding_.fetch_add(1);
      fprintf(stdout, "Tracker::Enable: entry %p, record %p, outst %lu\n", entry, entry->record,
              outstanding);
      fflush(stdout);
    }
  }

  void EnableContext(entry_t* entry, hsa_amd_signal_handler handler, void* arg) {
    entry->is_context = true;
    Enable(entry, reinterpret_cast<void*>(handler), arg);
  }
  void EnableDispatch(entry_t* entry, rocprofiler_handler_t handler, void* arg) {
    Enable(entry, reinterpret_cast<void*>(handler), arg);
  }
  void EnableMemcopy(entry_t* entry, hsa_amd_signal_handler handler, void* arg) {
    entry->is_memcopy = true;
    Enable(entry, reinterpret_cast<void*>(handler), arg);
  }

  // Enable tracking
  static void Enable_opt(Group* group, const hsa_signal_t& orig_signal) {
    group->SetOrigSignal(orig_signal);
    group->GetRecord()->dispatch = util::HsaRsrcFactory::Instance().TimestampNs();

    // Creating a proxy signal
    const hsa_signal_value_t signal_value = (orig_signal.handle)
        ? util::HsaRsrcFactory::Instance().HsaApi()->hsa_signal_load_relaxed(orig_signal)
        : 1;
    hsa_signal_t& dispatch_signal = group->GetDispatchSignal();
    util::HsaRsrcFactory::Instance().HsaApi()->hsa_signal_store_screlease(dispatch_signal,
                                                                          signal_value);
    hsa_status_t status = util::HsaRsrcFactory::Instance().HsaApi()->hsa_amd_signal_async_handler(
        dispatch_signal, HSA_SIGNAL_CONDITION_LT, signal_value, Handler_opt, group);
    if (status != HSA_STATUS_SUCCESS) EXC_RAISING(status, "hsa_amd_signal_async_handler");
  }

  // Tracker handler
  static bool Handler_opt(hsa_signal_value_t signal_value, void* arg) {
    Group* group = reinterpret_cast<Group*>(arg);
    Context* context = group->GetContext();
    hsa_signal_t dispatch_signal = group->GetDispatchSignal();
    record_t* record = group->GetRecord();
    hsa_amd_profiling_dispatch_time_t dispatch_time{};
    hsa_status_t status =
        util::HsaRsrcFactory::Instance().HsaApi()->hsa_amd_profiling_get_dispatch_time(
            context->GetAgent(), dispatch_signal, &dispatch_time);
    if (status != HSA_STATUS_SUCCESS) EXC_RAISING(status, "hsa_amd_profiling_get_dispatch_time");
    record->begin = util::HsaRsrcFactory::Instance().SysclockToNs(dispatch_time.start);
    record->end = util::HsaRsrcFactory::Instance().SysclockToNs(dispatch_time.end);
    record->complete = util::HsaRsrcFactory::Instance().TimestampNs();

    // Original intercepted signal completion
    const hsa_signal_t& orig_signal = group->GetOrigSignal();
    if (orig_signal.handle) {
      amd_signal_t* orig_signal_ptr = reinterpret_cast<amd_signal_t*>(orig_signal.handle);
      amd_signal_t* prof_signal_ptr = reinterpret_cast<amd_signal_t*>(dispatch_signal.handle);
      orig_signal_ptr->start_ts = prof_signal_ptr->start_ts;
      orig_signal_ptr->end_ts = prof_signal_ptr->end_ts;
      util::HsaRsrcFactory::Instance().HsaApi()->hsa_signal_store_screlease(orig_signal,
                                                                            signal_value);
    }

    return Context::Handler(signal_value, arg);
  }

 private:
  Tracker()
      : outstanding_(0),
        hsa_rsrc_(&(util::HsaRsrcFactory::Instance())),
        hsa_api_(*(hsa_rsrc_->HsaApi())) {}

  ~Tracker() {
    if (trace_on_) {
      fprintf(stdout, "Tracker::DESTR: sig list %d, outst %lu\n", (int)(sig_list_.size()),
              outstanding_.load());
      fflush(stdout);
    }

    auto it = sig_list_.begin();
    auto end = sig_list_.end();
    while (it != end) {
      auto cur = it++;
// The wait should be optiona as there possible some inter kernel dependencies and it possible to
// wait for the kernels will never be lunched as the application was finished by some reason.
#if 0
      // FIXME: currently the signal value for tracking signals are taken from original application signal
      hsa_rsrc_->SignalWait((*cur)->signal, 1);
#endif
      Erase(cur);
    }
  }

  // Delete an entry by iterator
  void Erase(const sig_list_it_t& it) { Delete(*it); }

  // Entry completion
  inline void Complete(hsa_signal_value_t signal_value, entry_t* entry) {
    record_t* record = entry->record;

    // Debug trace
    if (trace_on_) {
      auto outstanding = outstanding_.fetch_sub(1);
      fprintf(stdout, "Tracker::Complete: entry %p, record %p, outst %lu\n", entry, entry->record,
              outstanding);
      fflush(stdout);
    }

    // Query begin/end and complete timestamps
    if (entry->is_memcopy) {
      hsa_amd_profiling_async_copy_time_t async_copy_time{};
      hsa_status_t status =
          hsa_api_.hsa_amd_profiling_get_async_copy_time(entry->signal, &async_copy_time);
      if (status != HSA_STATUS_SUCCESS)
        EXC_RAISING(status, "hsa_amd_profiling_get_async_copy_time");
      record->begin = hsa_rsrc_->SysclockToNs(async_copy_time.start);
      record->end = hsa_rsrc_->SysclockToNs(async_copy_time.end);
    } else {
      hsa_amd_profiling_dispatch_time_t dispatch_time{};
      hsa_status_t status =
          hsa_api_.hsa_amd_profiling_get_dispatch_time(entry->agent, entry->signal, &dispatch_time);
      if (status != HSA_STATUS_SUCCESS) EXC_RAISING(status, "hsa_amd_profiling_get_dispatch_time");
      record->begin = hsa_rsrc_->SysclockToNs(dispatch_time.start);
      record->end = hsa_rsrc_->SysclockToNs(dispatch_time.end);
    }

    record->complete = hsa_rsrc_->TimestampNs();
    entry->valid.store(true, std::memory_order_release);

    // Original intercepted signal completion
    hsa_signal_t orig = entry->orig;
    if (orig.handle) {
      amd_signal_t* orig_signal_ptr = reinterpret_cast<amd_signal_t*>(orig.handle);
      amd_signal_t* prof_signal_ptr = reinterpret_cast<amd_signal_t*>(entry->signal.handle);
      orig_signal_ptr->start_ts = prof_signal_ptr->start_ts;
      orig_signal_ptr->end_ts = prof_signal_ptr->end_ts;
      hsa_api_.hsa_signal_store_screlease(orig, signal_value);
    }
  }

  inline static void HandleEntry(hsa_signal_value_t signal_value, entry_t* entry) {
    // Call entry handler
    void* handler = static_cast<void*>(entry->handler);
    if (entry->is_context || entry->is_memcopy) {
      reinterpret_cast<hsa_amd_signal_handler>(handler)(signal_value, entry->arg);
    } else {
      rocprofiler_group_t group{};
      reinterpret_cast<rocprofiler_handler_t>(handler)(group, entry->arg);
    }
    // Delete tracker entry
    entry->tracker->Delete(entry);
  }

  // Handler for packet completion
  static bool Handler(hsa_signal_value_t signal_value, void* arg) {
    // Acquire entry
    entry_t* entry = reinterpret_cast<entry_t*>(arg);
    volatile std::atomic<void*>* ptr = &entry->handler;
    while (ptr->load(std::memory_order_acquire) == NULL) sched_yield();

    // Complete entry
    Tracker* tracker = entry->tracker;
    tracker->Complete(signal_value, entry);

    if (ordering_enabled_ == false) {
      HandleEntry(signal_value, entry);
    } else {
      // Acquire last entry
      entry_t* back = tracker->sig_list_.back();
      volatile std::atomic<void*>* ptr = &back->handler;
      while (ptr->load(std::memory_order_acquire) == NULL) sched_yield();

      tracker->handler_mutex_.lock();
      sig_list_it_t it = tracker->sig_list_.begin();
      sig_list_it_t end = back->it;
      while (it != end) {
        entry = *(it++);
        if (entry->valid.load(std::memory_order_acquire)) {
          HandleEntry(signal_value, entry);
        } else {
          break;
        }
      }
      tracker->handler_mutex_.unlock();
    }

    return false;
  }

  // instance
  static std::atomic<Tracker*> instance_;
  static mutex_t glob_mutex_;
  static counter_t counter_;

  // Tracked signals list
  sig_list_t sig_list_;
  // Inter-thread synchronization
  mutex_t mutex_;
  mutex_t handler_mutex_;
  // Outstanding dispatches
  std::atomic<uint64_t> outstanding_;
  // HSA resources factory
  util::HsaRsrcFactory* hsa_rsrc_;
  const util::hsa_pfn_t& hsa_api_;
  // Handling ordering enabled
  static const bool ordering_enabled_ = false;
  // Enable tracing
  static const bool trace_on_ = false;
};

}  // namespace rocprofiler

#endif  // SRC_CORE_TRACKER_H_
