////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2020, Advanced Micro Devices, Inc. All rights reserved.
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

#include "core/inc/interrupt_signal.h"
#include "core/inc/runtime.h"
#include "core/util/locks.h"

namespace rocr {
namespace core {

HsaEvent* InterruptSignal::EventPool::alloc() {
  ScopedAcquire<HybridMutex> lock(&lock_);
  if (events_.empty()) {
    if (!allEventsAllocated) {
      HsaEvent* evt = InterruptSignal::CreateEvent(HSA_EVENTTYPE_SIGNAL, false);
      if (evt == nullptr) allEventsAllocated = true;
      return evt;
    }
    return nullptr;
  }
  HsaEvent* ret = events_.back().release();
  events_.pop_back();
  return ret;
}

void InterruptSignal::EventPool::free(HsaEvent* evt) {
  if (evt == nullptr) return;
  ScopedAcquire<HybridMutex> lock(&lock_);
  events_.push_back(unique_event_ptr(evt));
}

HsaEvent* InterruptSignal::CreateEvent(HSA_EVENTTYPE type, bool manual_reset) {
  HsaEventDescriptor event_descriptor;
  event_descriptor.EventType = type;
  event_descriptor.SyncVar.SyncVar.UserData = NULL;
  event_descriptor.SyncVar.SyncVarSize = sizeof(hsa_signal_value_t);
  event_descriptor.NodeId = 0;

  HsaEvent* ret = NULL;
  if (HSAKMT_STATUS_SUCCESS ==
      HSAKMT_CALL(hsaKmtCreateEvent(&event_descriptor, manual_reset, false, &ret))) {
    if (type == HSA_EVENTTYPE_MEMORY) {
      memset(&ret->EventData.EventData.MemoryAccessFault.Failure, 0,
             sizeof(HsaAccessAttributeFailure));
    } else if (type == HSA_EVENTTYPE_HW_EXCEPTION) {
      memset(&ret->EventData.EventData.HwException, 0, sizeof(HsaHwException));
    }
  }

  return ret;
}

void InterruptSignal::DestroyEvent(HsaEvent* evt) { HSAKMT_CALL(hsaKmtDestroyEvent(evt)); }

InterruptSignal::InterruptSignal(hsa_signal_value_t initial_value, HsaEvent* use_event)
    : LocalSignal(initial_value, false), Signal(signal()) {
  if (use_event != nullptr) {
    event_ = use_event;
    free_event_ = false;
  } else {
    event_ = Runtime::runtime_singleton_->GetEventPool()->alloc();
    free_event_ = true;
  }

  if (event_ != nullptr) {
    signal_.event_id = event_->EventId;
    signal_.event_mailbox_ptr = event_->EventData.HWData2;
  } else {
    signal_.event_id = 0;
    signal_.event_mailbox_ptr = 0;
  }
  signal_.kind = AMD_SIGNAL_KIND_USER;
}

InterruptSignal::~InterruptSignal() {
  if (free_event_) Runtime::runtime_singleton_->GetEventPool()->free(event_);
}

hsa_signal_value_t InterruptSignal::LoadRelaxed() {
  return hsa_signal_value_t(
      atomic::Load(&signal_.value, std::memory_order_relaxed));
}

hsa_signal_value_t InterruptSignal::LoadAcquire() {
  return hsa_signal_value_t(
      atomic::Load(&signal_.value, std::memory_order_acquire));
}

void InterruptSignal::StoreRelaxed(hsa_signal_value_t value) {
  atomic::Store(&signal_.value, int64_t(value), std::memory_order_relaxed);
  SetEvent();
}

void InterruptSignal::StoreRelease(hsa_signal_value_t value) {
  atomic::Store(&signal_.value, int64_t(value), std::memory_order_release);
  SetEvent();
}

hsa_signal_value_t InterruptSignal::WaitRelaxed(hsa_signal_condition_t condition,
                                               hsa_signal_value_t compare_value,
                                               uint64_t timeout,
                                               hsa_wait_state_t wait_hint) {
  Retain();
  MAKE_SCOPE_GUARD([&]() { Release(); });

  uint32_t prior = waiting_++;
  MAKE_SCOPE_GUARD([&]() { waiting_--; });

  uint64_t event_age = core::Runtime::runtime_singleton_->KfdVersion().supports_event_age ? 1 : 0;
  if (!event_age && prior != 0) wait_hint = HSA_WAIT_STATE_ACTIVE;

  const timer::fast_clock::time_point start_time = timer::fast_clock::now();
  const timer::fast_clock::duration fast_timeout = timer::GetFastTimeout(timeout);
  const timer::fast_clock::duration kMaxElapsed = std::chrono::microseconds(200);
  const uint32_t &signal_abort_timeout =
    core::Runtime::runtime_singleton_->flag().signal_abort_timeout();

  while (true) {
    if (!IsValid()) return 0;

    int64_t value = atomic::Load(&signal_.value, std::memory_order_relaxed);

    if (CheckSignalCondition(value, condition, compare_value)) {
      return value;
    }

    auto now = timer::fast_clock::now();
    if (now - start_time > fast_timeout) {
      return value;
    }

    timer::CheckAbortTimeout(start_time, signal_abort_timeout);

    if (wait_hint == HSA_WAIT_STATE_ACTIVE) {
      if (g_use_mwaitx) {
        // Short timeout for active waiting
        timer::DoMwaitx(const_cast<int64_t*>(&signal_.value), 1000);
      }
      continue;
    }

    if (now - start_time < kMaxElapsed) {
      if (g_use_mwaitx) {
        // Longer timeout with timer for passive waiting
        timer::DoMwaitx(const_cast<int64_t*>(&signal_.value), 60000, true);
      }
      continue;
    }

    auto remaining_ms = timer::duration_cast<std::chrono::milliseconds>(
      fast_timeout - (now - start_time)).count();

    uint32_t wait_ms = std::min<uint32_t>(
      static_cast<uint32_t>(std::min<uint64_t>(remaining_ms, 0xFFFFFFFEUL)),
      static_cast<uint32_t>(signal_abort_timeout ? signal_abort_timeout * 1000 : 0xFFFFFFFFUL)
    );

    HSAKMT_CALL(hsaKmtWaitOnEvent_Ext(event_, wait_ms, &event_age));
  }
}

hsa_signal_value_t InterruptSignal::WaitAcquire(
    hsa_signal_condition_t condition, hsa_signal_value_t compare_value,
    uint64_t timeout, hsa_wait_state_t wait_hint) {
  hsa_signal_value_t ret =
      WaitRelaxed(condition, compare_value, timeout, wait_hint);
  std::atomic_thread_fence(std::memory_order_acquire);
  return ret;
}

void InterruptSignal::AndRelaxed(hsa_signal_value_t value) {
  atomic::And(&signal_.value, int64_t(value), std::memory_order_relaxed);
  SetEvent();
}

void InterruptSignal::AndAcquire(hsa_signal_value_t value) {
  atomic::And(&signal_.value, int64_t(value), std::memory_order_acquire);
  SetEvent();
}

void InterruptSignal::AndRelease(hsa_signal_value_t value) {
  atomic::And(&signal_.value, int64_t(value), std::memory_order_release);
  SetEvent();
}

void InterruptSignal::AndAcqRel(hsa_signal_value_t value) {
  atomic::And(&signal_.value, int64_t(value), std::memory_order_acq_rel);
  SetEvent();
}

void InterruptSignal::OrRelaxed(hsa_signal_value_t value) {
  atomic::Or(&signal_.value, int64_t(value), std::memory_order_relaxed);
  SetEvent();
}

void InterruptSignal::OrAcquire(hsa_signal_value_t value) {
  atomic::Or(&signal_.value, int64_t(value), std::memory_order_acquire);
  SetEvent();
}

void InterruptSignal::OrRelease(hsa_signal_value_t value) {
  atomic::Or(&signal_.value, int64_t(value), std::memory_order_release);
  SetEvent();
}

void InterruptSignal::OrAcqRel(hsa_signal_value_t value) {
  atomic::Or(&signal_.value, int64_t(value), std::memory_order_acq_rel);
  SetEvent();
}

void InterruptSignal::XorRelaxed(hsa_signal_value_t value) {
  atomic::Xor(&signal_.value, int64_t(value), std::memory_order_relaxed);
  SetEvent();
}

void InterruptSignal::XorAcquire(hsa_signal_value_t value) {
  atomic::Xor(&signal_.value, int64_t(value), std::memory_order_acquire);
  SetEvent();
}

void InterruptSignal::XorRelease(hsa_signal_value_t value) {
  atomic::Xor(&signal_.value, int64_t(value), std::memory_order_release);
  SetEvent();
}

void InterruptSignal::XorAcqRel(hsa_signal_value_t value) {
  atomic::Xor(&signal_.value, int64_t(value), std::memory_order_acq_rel);
  SetEvent();
}

void InterruptSignal::AddRelaxed(hsa_signal_value_t value) {
  atomic::Add(&signal_.value, int64_t(value), std::memory_order_relaxed);
  SetEvent();
}

void InterruptSignal::AddAcquire(hsa_signal_value_t value) {
  atomic::Add(&signal_.value, int64_t(value), std::memory_order_acquire);
  SetEvent();
}

void InterruptSignal::AddRelease(hsa_signal_value_t value) {
  atomic::Add(&signal_.value, int64_t(value), std::memory_order_release);
  SetEvent();
}

void InterruptSignal::AddAcqRel(hsa_signal_value_t value) {
  atomic::Add(&signal_.value, int64_t(value), std::memory_order_acq_rel);
  SetEvent();
}

void InterruptSignal::SubRelaxed(hsa_signal_value_t value) {
  atomic::Sub(&signal_.value, int64_t(value), std::memory_order_relaxed);
  SetEvent();
}

void InterruptSignal::SubAcquire(hsa_signal_value_t value) {
  atomic::Sub(&signal_.value, int64_t(value), std::memory_order_acquire);
  SetEvent();
}

void InterruptSignal::SubRelease(hsa_signal_value_t value) {
  atomic::Sub(&signal_.value, int64_t(value), std::memory_order_release);
  SetEvent();
}

void InterruptSignal::SubAcqRel(hsa_signal_value_t value) {
  atomic::Sub(&signal_.value, int64_t(value), std::memory_order_acq_rel);
  SetEvent();
}

hsa_signal_value_t InterruptSignal::ExchRelaxed(hsa_signal_value_t value) {
  hsa_signal_value_t ret = hsa_signal_value_t(atomic::Exchange(
      &signal_.value, int64_t(value), std::memory_order_relaxed));
  SetEvent();
  return ret;
}

hsa_signal_value_t InterruptSignal::ExchAcquire(hsa_signal_value_t value) {
  hsa_signal_value_t ret = hsa_signal_value_t(atomic::Exchange(
      &signal_.value, int64_t(value), std::memory_order_acquire));
  SetEvent();
  return ret;
}

hsa_signal_value_t InterruptSignal::ExchRelease(hsa_signal_value_t value) {
  hsa_signal_value_t ret = hsa_signal_value_t(atomic::Exchange(
      &signal_.value, int64_t(value), std::memory_order_release));
  SetEvent();
  return ret;
}

hsa_signal_value_t InterruptSignal::ExchAcqRel(hsa_signal_value_t value) {
  hsa_signal_value_t ret = hsa_signal_value_t(atomic::Exchange(
      &signal_.value, int64_t(value), std::memory_order_acq_rel));
  SetEvent();
  return ret;
}

hsa_signal_value_t InterruptSignal::CasRelaxed(hsa_signal_value_t expected,
                                               hsa_signal_value_t value) {
  hsa_signal_value_t ret = hsa_signal_value_t(
      atomic::Cas(&signal_.value, int64_t(value), int64_t(expected),
                  std::memory_order_relaxed));
  SetEvent();
  return ret;
}

hsa_signal_value_t InterruptSignal::CasAcquire(hsa_signal_value_t expected,
                                               hsa_signal_value_t value) {
  hsa_signal_value_t ret = hsa_signal_value_t(
      atomic::Cas(&signal_.value, int64_t(value), int64_t(expected),
                  std::memory_order_acquire));
  SetEvent();
  return ret;
}

hsa_signal_value_t InterruptSignal::CasRelease(hsa_signal_value_t expected,
                                               hsa_signal_value_t value) {
  hsa_signal_value_t ret = hsa_signal_value_t(
      atomic::Cas(&signal_.value, int64_t(value), int64_t(expected),
                  std::memory_order_release));
  SetEvent();
  return ret;
}

hsa_signal_value_t InterruptSignal::CasAcqRel(hsa_signal_value_t expected,
                                              hsa_signal_value_t value) {
  hsa_signal_value_t ret = hsa_signal_value_t(
      atomic::Cas(&signal_.value, int64_t(value), int64_t(expected),
                  std::memory_order_acq_rel));
  SetEvent();
  return ret;
}
  /// @brief Notify driver of signal value change if necessary.
  void InterruptSignal::SetEvent() {
    std::atomic_signal_fence(std::memory_order_seq_cst);
    if (InWaiting()) HSAKMT_CALL(hsaKmtSetEvent(event_));
  }

}  // namespace core
}  // namespace rocr
