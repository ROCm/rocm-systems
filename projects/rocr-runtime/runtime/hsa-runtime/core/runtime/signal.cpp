////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2024, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and AMD HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef HSA_RUNTME_CORE_SIGNAL_CPP_
#define HSA_RUNTME_CORE_SIGNAL_CPP_

#include "core/inc/signal.h"

#include <algorithm>
#include <numeric>
#include <vector>

#include "core/util/timer.h"
#include "core/inc/runtime.h"

namespace rocr {
namespace core {

KernelMutex Signal::ipcLock_;
std::map<decltype(hsa_signal_t::handle), Signal*> Signal::ipcMap_;

void SharedSignalPool_t::clear() {
  ifdebug {
    size_t capacity = 0;
    for (auto& block : block_list_) capacity += block.second;
    if (capacity != free_list_.size())
      debug_print("Warning: Resource leak detected by SharedSignalPool, %ld Signals leaked.\n",
                  capacity - free_list_.size());
  }

  for (auto& block : block_list_) free_()(block.first);
  block_list_.clear();
  free_list_.clear();
}

SharedSignal* SharedSignalPool_t::alloc() {
  ScopedAcquire<HybridMutex> lock(&lock_);
  if (free_list_.empty()) {
    SharedSignal* block = reinterpret_cast<SharedSignal*>(
        allocate_()(block_size_ * sizeof(SharedSignal), __alignof(SharedSignal), 0, 0));
    if (block == nullptr) {
      block_size_ = minblock_;
      block = reinterpret_cast<SharedSignal*>(
          allocate_()(block_size_ * sizeof(SharedSignal), __alignof(SharedSignal), 0, 0));
      if (block == nullptr) throw std::bad_alloc();
    }

    MAKE_NAMED_SCOPE_GUARD(throwGuard, [&]() { free_()(block); });
    block_list_.push_back(std::make_pair(block, block_size_));
    throwGuard.Dismiss();


    for (int i = 0; i < block_size_; i++) {
      free_list_.push_back(&block[i]);
    }

    block_size_ *= 2;
  }

  SharedSignal* ret = free_list_.back();
  new (ret) SharedSignal();
  free_list_.pop_back();
  return ret;
}

void SharedSignalPool_t::free(SharedSignal* ptr) {
  if (ptr == nullptr) return;

  ptr->~SharedSignal();
  ScopedAcquire<HybridMutex> lock(&lock_);

  ifdebug {
    bool valid = false;
    for (auto& block : block_list_) {
      if ((block.first <= ptr) &&
          (uintptr_t(ptr) < uintptr_t(block.first) + block.second * sizeof(SharedSignal))) {
        valid = true;
        break;
      }
    }
    assert(valid && "Object does not belong to pool.");
  }

  free_list_.push_back(ptr);
}

LocalSignal::LocalSignal(hsa_signal_value_t initial_value, bool exportable)
    : local_signal_(exportable ? nullptr
                               : core::Runtime::runtime_singleton_->GetSharedSignalPool(),
                    exportable ? core::MemoryRegion::AllocateIPC : 0) {
  local_signal_.shared_object()->amd_signal.value = initial_value;
}

void Signal::registerIpc() {
  ScopedAcquire<KernelMutex> lock(&ipcLock_);
  auto handle = Convert(this);
  assert(ipcMap_.find(handle.handle) == ipcMap_.end() &&
         "Can't register the same IPC signal twice.");
  ipcMap_[handle.handle] = this;
}

bool Signal::deregisterIpc() {
  ScopedAcquire<KernelMutex> lock(&ipcLock_);
  if (refcount_ != 0) return false;
  auto handle = Convert(this);
  const auto& it = ipcMap_.find(handle.handle);
  assert(it != ipcMap_.end() && "Deregister on non-IPC signal.");
  ipcMap_.erase(it);
  return true;
}

Signal* Signal::lookupIpc(hsa_signal_t signal) {
  ScopedAcquire<KernelMutex> lock(&ipcLock_);
  const auto& it = ipcMap_.find(signal.handle);
  if (it == ipcMap_.end()) return nullptr;
  return it->second;
}

Signal* Signal::duplicateIpc(hsa_signal_t signal) {
  ScopedAcquire<KernelMutex> lock(&ipcLock_);
  const auto& it = ipcMap_.find(signal.handle);
  if (it == ipcMap_.end()) return nullptr;
  it->second->refcount_++;
  it->second->Retain();
  return it->second;
}

void Signal::Release() {
  if (--retained_ != 0) return;
  if (!isIPC())
    doDestroySignal();
  else if (deregisterIpc())
    doDestroySignal();
}

Signal::~Signal() {
  signal_.kind = AMD_SIGNAL_KIND_INVALID;
  if (refcount_ == 1 && isIPC()) {
    refcount_ = 0;
    deregisterIpc();
  }
}

uint32_t Signal::WaitMultiple(uint32_t signal_count, const hsa_signal_t* hsa_signals,
                              const hsa_signal_condition_t* conds, const hsa_signal_value_t* values,
                              uint64_t timeout, hsa_wait_state_t wait_hint,
                              std::vector<hsa_signal_value_t>& satisfying_values,
                              bool wait_on_all) {
  hsa_signal_handle* signals =
      reinterpret_cast<hsa_signal_handle*>(const_cast<hsa_signal_t*>(hsa_signals));

  for (uint32_t i = 0; i < signal_count; i++) signals[i]->Retain();

  MAKE_SCOPE_GUARD([&]() {
    for (uint32_t i = 0; i < signal_count; i++) signals[i]->Release();
  });

  uint32_t prior = 0;
  for (uint32_t i = 0; i < signal_count; i++) prior = Max(prior, signals[i]->waiting_++);

  MAKE_SCOPE_GUARD([&]() {
    for (uint32_t i = 0; i < signal_count; i++) signals[i]->waiting_--;
  });

  if (!core::Runtime::runtime_singleton_->KfdVersion().supports_event_age)
      // Allow only the first waiter to sleep. Without event age tracking,
      // race condition can cause some threads to sleep without wakeup since missing interrupt.
      if (prior != 0) wait_hint = HSA_WAIT_STATE_ACTIVE;

  // Ensure that all signals in the list can be slept on.
  if (wait_hint != HSA_WAIT_STATE_ACTIVE) {
    for (uint32_t i = 0; i < signal_count; i++) {
      if (signals[i]->EopEvent() == NULL) {
        wait_hint = HSA_WAIT_STATE_ACTIVE;
        break;
      }
    }
  }

  const uint32_t small_size = 10;
  HsaEvent* short_evts[small_size];
  HsaEvent** evts = NULL;
  uint32_t unique_evts = 0;
  if (wait_hint != HSA_WAIT_STATE_ACTIVE) {
    if (signal_count > small_size)
      evts = new HsaEvent* [signal_count];
    else
      evts = short_evts;
    for (uint32_t i = 0; i < signal_count; i++)
      evts[i] = signals[i]->EopEvent();
    std::sort(evts, evts + signal_count);
    HsaEvent** end = std::unique(evts, evts + signal_count);
    unique_evts = uint32_t(end - evts);
  }
  MAKE_SCOPE_GUARD([&]() {
    if (signal_count > small_size) delete[] evts;
  });

  uint64_t event_age[unique_evts];
  memset(event_age, 0, unique_evts * sizeof(uint64_t));
  if (core::Runtime::runtime_singleton_->KfdVersion().supports_event_age)
    for (uint32_t i = 0; i < unique_evts; i++)
      event_age[i] = 1;

  int64_t value;

  timer::fast_clock::time_point start_time = timer::fast_clock::now();

  // Set a polling timeout value
  const timer::fast_clock::duration kMaxElapsed = std::chrono::microseconds(200);

  // Convert timeout value into the fast_clock domain
  uint64_t hsa_freq = 0;
  HSA::hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &hsa_freq);
  const timer::fast_clock::duration fast_timeout =
      timer::duration_from_seconds<timer::fast_clock::duration>(
          double(timeout) / double(hsa_freq));

  std::vector<uint32_t> unmet_condition_ids(signal_count);
  std::iota(unmet_condition_ids.begin(), unmet_condition_ids.end(), 0);

  while (true) {
    // Cannot mwaitx - polling multiple signals
    for (auto it = unmet_condition_ids.begin(); it != unmet_condition_ids.end();) {
      auto i = *it;
      bool condition_met = false;
      if (!signals[i]->IsValid())
        return uint32_t(-1);

      value =
          atomic::Load(&signals[i]->signal_.value, std::memory_order_relaxed);

      switch (conds[i]) {
        case HSA_SIGNAL_CONDITION_EQ: {
          condition_met = (value == values[i]);
          break;
        }
        case HSA_SIGNAL_CONDITION_NE: {
          condition_met = (value != values[i]);
          break;
        }
        case HSA_SIGNAL_CONDITION_GTE: {
          condition_met = (value >= values[i]);
          break;
        }
        case HSA_SIGNAL_CONDITION_LT: {
          condition_met = (value < values[i]);
          break;
        }
        default:
          return uint32_t(-1);
      }
      if (condition_met) {
        it = unmet_condition_ids.erase(it);
        satisfying_values[i] = value;
        if (!wait_on_all)
          return i;
        else if (unmet_condition_ids.empty())
          return 0;
      } else {
        ++it;
      }
    }

    timer::fast_clock::time_point time = timer::fast_clock::now();
    if (time - start_time > fast_timeout) {
      return uint32_t(-1);
    }

    if (wait_hint == HSA_WAIT_STATE_ACTIVE) {
      continue;
    }

    if (time - start_time < kMaxElapsed) {
    //  os::uSleep(20);
      continue;
    }

    uint32_t wait_ms;
    auto time_remaining = fast_timeout - (time - start_time);
    uint64_t ct=timer::duration_cast<std::chrono::milliseconds>(
      time_remaining).count();
    wait_ms = (ct>0xFFFFFFFEu) ? 0xFFFFFFFEu : ct;
    HSAKMT_CALL(hsaKmtWaitOnMultipleEvents_Ext(evts, unique_evts, wait_on_all, wait_ms, event_age));
  }
}

/*
 * Special handler to wait listen for exceptions from underlying driver.
 */
uint32_t Signal::WaitAnyExceptions(uint32_t signal_count, const hsa_signal_t* hsa_signals,
                         const hsa_signal_condition_t* conds, const hsa_signal_value_t* values,
                         hsa_signal_value_t* satisfying_value) {

  uint32_t wait_ms = uint32_t(-1);
  hsa_signal_handle* signals =
      reinterpret_cast<hsa_signal_handle*>(const_cast<hsa_signal_t*>(hsa_signals));

  for (uint32_t i = 0; i < signal_count; i++) signals[i]->Retain();

  MAKE_SCOPE_GUARD([&]() {
    for (uint32_t i = 0; i < signal_count; i++) signals[i]->Release();
  });

  uint32_t prior = 0;
  for (uint32_t i = 0; i < signal_count; i++) prior = Max(prior, signals[i]->waiting_++);


  MAKE_SCOPE_GUARD([&]() {
    for (uint32_t i = 0; i < signal_count; i++) signals[i]->waiting_--;
  });

  if (!core::Runtime::runtime_singleton_->KfdVersion().supports_event_age)
      // Allow only the first waiter to sleep. Without event age tracking,
      // race condition can cause some threads to sleep without wakeup since missing interrupt.
      if (prior != 0) wait_ms = 0;

  HsaEvent** evts = new HsaEvent* [signal_count];
  MAKE_SCOPE_GUARD([&]() { delete[] evts; });

  uint32_t unique_evts = 0;

  for (uint32_t i = 0; i < signal_count; i++) {
    assert(signals[i]->EopEvent() != NULL);
    evts[i] = signals[i]->EopEvent();
  }

  std::sort(evts, evts + signal_count);
  HsaEvent** end = std::unique(evts, evts + signal_count);
  unique_evts = uint32_t(end - evts);

  uint64_t event_age[unique_evts];
  memset(event_age, 0, unique_evts * sizeof(uint64_t));
  if (core::Runtime::runtime_singleton_->KfdVersion().supports_event_age)
    for (uint32_t i = 0; i < unique_evts; i++)
      event_age[i] = 1;

  int64_t value;

  bool condition_met = false;
  while (true) {
    // Cannot mwaitx - polling multiple signals

    for (uint32_t i = 0; i < signal_count; i++) {
      if (!signals[i]->IsValid())
        return uint32_t(-1);

      const HSA_EVENTTYPE event_type = signals[i]->EopEvent()->EventData.EventType;
      if (event_type == HSA_EVENTTYPE_MEMORY) {
        const HsaMemoryAccessFault& fault =
            signals[i]->EopEvent()->EventData.EventData.MemoryAccessFault;
        if (fault.Flags == HSA_EVENTID_MEMORY_FATAL_PROCESS) return i;
      } else if (event_type == HSA_EVENTTYPE_HW_EXCEPTION) {
        const HsaHwException& exception =
            signals[i]->EopEvent()->EventData.EventData.HwException;
        if (exception.MemoryLost) return i;
      }

      value = atomic::Load(&signals[i]->signal_.value, std::memory_order_relaxed);

      switch (conds[i]) {
        case HSA_SIGNAL_CONDITION_EQ: {
          condition_met = (value == values[i]);
          break;
        }
        case HSA_SIGNAL_CONDITION_NE: {
          condition_met = (value != values[i]);
          break;
        }
        case HSA_SIGNAL_CONDITION_GTE: {
          condition_met = (value >= values[i]);
          break;
        }
        case HSA_SIGNAL_CONDITION_LT: {
          condition_met = (value < values[i]);
          break;
        }
        default: {
          return uint32_t(-1);
        }
      }
      if (condition_met) {
        if (satisfying_value != NULL) *satisfying_value = value;
        // Some other signal in the list satisfied condition
        return i;
      }
    }

    HSAKMT_CALL(hsaKmtWaitOnMultipleEvents_Ext(evts, unique_evts, false, wait_ms, event_age));
  } //while
}

SignalGroup::SignalGroup(uint32_t num_signals, const hsa_signal_t* hsa_signals)
    : count(num_signals) {
  if (count != 0) {
    signals = new hsa_signal_t[count];
  } else {
    signals = NULL;
  }
  if (signals == NULL) return;
  for (uint32_t i = 0; i < count; i++) signals[i] = hsa_signals[i];
}

}  // namespace core
}  // namespace rocr

#endif  // header guard
