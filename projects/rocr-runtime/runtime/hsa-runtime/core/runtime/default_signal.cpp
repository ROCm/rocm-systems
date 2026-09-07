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

#include "core/inc/default_signal.h"

#include <cstdio>
#include <cstdlib>

#if defined(__i386__) || defined(__x86_64__)
#include <mwaitxintrin.h>
#include <xmmintrin.h>
#define MWAITX_ECX_TIMER_ENABLE 0x2  // BIT(1)
// Drain the write combining buffers.  See StoreRelaxed().
#define ROCR_WC_DRAIN() _mm_sfence()
#else
#define ROCR_WC_DRAIN() std::atomic_thread_fence(std::memory_order_release)
#endif

namespace rocr {
namespace core {

BusyWaitSignal::BusyWaitSignal(SharedSignal* abi_block, bool enableIPC, bool device_resident_value)
    : Signal(abi_block, enableIPC, device_resident_value) {
  signal_.kind = AMD_SIGNAL_KIND_USER;
  signal_.event_mailbox_ptr = uint64_t(NULL);
}

void BusyWaitSignal::RejectHostAtomicRmw() const {
  if (!IsDeviceResidentValue()) return;

  // Reached only if a caller performed a read-modify-write on a signal it
  // asked to have placed in device memory.  The API layer rejects these with
  // HSA_STATUS_ERROR_INVALID_SIGNAL; these entry points have no status return,
  // and completing the operation would corrupt the value word, so stop.
  fprintf(stderr,
          "HSA: read-modify-write on a device resident signal value word is "
          "not supported.\n");
  abort();
}

hsa_signal_value_t BusyWaitSignal::LoadRelaxed() {
  return hsa_signal_value_t(
      atomic::Load(&signal_.value, std::memory_order_relaxed));
}

hsa_signal_value_t BusyWaitSignal::LoadAcquire() {
  return hsa_signal_value_t(
      atomic::Load(&signal_.value, std::memory_order_acquire));
}

void BusyWaitSignal::StoreRelaxed(hsa_signal_value_t value) {
  atomic::Store(&signal_.value, int64_t(value), std::memory_order_relaxed);

  if (IsDeviceResidentValue()) {
    // The value word is in device memory, written through a write combining
    // aperture.  A relaxed store can still sit in a write combining buffer when
    // a later store to a different aperture - a queue doorbell, typically - has
    // become visible, so the command processor can see the doorbell before the
    // value.  Same hazard, and same fence, as the device memory ring buffer
    // stores; unconditional here because this store is not per-dispatch, where
    // those sites test needsPcieOrdering().
    //
    // Not PcieWcFlush(): its body ends in a readback, which on a device resident
    // word is the host bus read this placement exists to remove, and it would be
    // paid on every store.
    //
    // This is the ONLY host write site drained.  StoreRelease() fences BEFORE
    // its store and emits nothing after it, and SharedSignal::CopyPrep() writes
    // a destination block's header with plain stores; neither is reachable with
    // a device resident target today, and a caller that makes one reachable must
    // close it.
    ROCR_WC_DRAIN();
  }
}

void BusyWaitSignal::StoreRelease(hsa_signal_value_t value) {
  atomic::Store(&signal_.value, int64_t(value), std::memory_order_release);
}

hsa_signal_value_t BusyWaitSignal::WaitRelaxed(hsa_signal_condition_t condition,
                                               hsa_signal_value_t compare_value, uint64_t timeout,
                                               hsa_wait_state_t wait_hint) {
  Retain();
  MAKE_SCOPE_GUARD([&]() { Release(); });

  waiting_++;
  MAKE_SCOPE_GUARD([&]() { waiting_--; });

  const uint32_t &signal_abort_timeout =
    core::Runtime::runtime_singleton_->flag().signal_abort_timeout();

  const timer::fast_clock::time_point start_time = timer::fast_clock::now();
  const timer::fast_clock::duration fast_timeout = timer::GetFastTimeout(timeout);

  while (true) {
    if (!IsValid()) return 0;

    int64_t value = atomic::Load(&signal_.value, std::memory_order_relaxed);

    if (CheckSignalCondition(value, condition, compare_value)) {
      return value;
    }

    if (timer::fast_clock::now() - start_time > fast_timeout) {
      return value;
    }

    timer::CheckAbortTimeout(start_time, signal_abort_timeout);

    // MONITORX arms a hardware monitor on a cache line.  A device resident value
    // word is mapped write combining, not write-back cacheable, and the
    // architecture does not guarantee a monitor can be established on such a
    // line - the park would degenerate into a timed sleep that the awaited write
    // does not end.  Spin instead; the loop already re-reads the word and
    // re-tests the condition every iteration.
    if (g_use_mwaitx && !IsDeviceResidentValue()) {
      // Use timer-enabled mwaitx for busy waiting
      timer::DoMwaitx(const_cast<int64_t*>(&signal_.value), value, 60000, true);
    }
  }
}

hsa_signal_value_t BusyWaitSignal::WaitAcquire(hsa_signal_condition_t condition,
                                               hsa_signal_value_t compare_value, uint64_t timeout,
                                               hsa_wait_state_t wait_hint) {
  hsa_signal_value_t ret =
      WaitRelaxed(condition, compare_value, timeout, wait_hint);
  std::atomic_thread_fence(std::memory_order_acquire);
  return ret;
}

void BusyWaitSignal::AndRelaxed(hsa_signal_value_t value) {
  RejectHostAtomicRmw();
  atomic::And(&signal_.value, int64_t(value), std::memory_order_relaxed);
}

void BusyWaitSignal::AndAcquire(hsa_signal_value_t value) {
  RejectHostAtomicRmw();
  atomic::And(&signal_.value, int64_t(value), std::memory_order_acquire);
}

void BusyWaitSignal::AndRelease(hsa_signal_value_t value) {
  RejectHostAtomicRmw();
  atomic::And(&signal_.value, int64_t(value), std::memory_order_release);
}

void BusyWaitSignal::AndAcqRel(hsa_signal_value_t value) {
  RejectHostAtomicRmw();
  atomic::And(&signal_.value, int64_t(value), std::memory_order_acq_rel);
}

void BusyWaitSignal::OrRelaxed(hsa_signal_value_t value) {
  RejectHostAtomicRmw();
  atomic::Or(&signal_.value, int64_t(value), std::memory_order_relaxed);
}

void BusyWaitSignal::OrAcquire(hsa_signal_value_t value) {
  RejectHostAtomicRmw();
  atomic::Or(&signal_.value, int64_t(value), std::memory_order_acquire);
}

void BusyWaitSignal::OrRelease(hsa_signal_value_t value) {
  RejectHostAtomicRmw();
  atomic::Or(&signal_.value, int64_t(value), std::memory_order_release);
}

void BusyWaitSignal::OrAcqRel(hsa_signal_value_t value) {
  RejectHostAtomicRmw();
  atomic::Or(&signal_.value, int64_t(value), std::memory_order_acq_rel);
}

void BusyWaitSignal::XorRelaxed(hsa_signal_value_t value) {
  RejectHostAtomicRmw();
  atomic::Xor(&signal_.value, int64_t(value), std::memory_order_relaxed);
}

void BusyWaitSignal::XorAcquire(hsa_signal_value_t value) {
  RejectHostAtomicRmw();
  atomic::Xor(&signal_.value, int64_t(value), std::memory_order_acquire);
}

void BusyWaitSignal::XorRelease(hsa_signal_value_t value) {
  RejectHostAtomicRmw();
  atomic::Xor(&signal_.value, int64_t(value), std::memory_order_release);
}

void BusyWaitSignal::XorAcqRel(hsa_signal_value_t value) {
  RejectHostAtomicRmw();
  atomic::Xor(&signal_.value, int64_t(value), std::memory_order_acq_rel);
}

void BusyWaitSignal::AddRelaxed(hsa_signal_value_t value) {
  RejectHostAtomicRmw();
  atomic::Add(&signal_.value, int64_t(value), std::memory_order_relaxed);
}

void BusyWaitSignal::AddAcquire(hsa_signal_value_t value) {
  RejectHostAtomicRmw();
  atomic::Add(&signal_.value, int64_t(value), std::memory_order_acquire);
}

void BusyWaitSignal::AddRelease(hsa_signal_value_t value) {
  RejectHostAtomicRmw();
  atomic::Add(&signal_.value, int64_t(value), std::memory_order_release);
}

void BusyWaitSignal::AddAcqRel(hsa_signal_value_t value) {
  RejectHostAtomicRmw();
  atomic::Add(&signal_.value, int64_t(value), std::memory_order_acq_rel);
}

void BusyWaitSignal::SubRelaxed(hsa_signal_value_t value) {
  RejectHostAtomicRmw();
  atomic::Sub(&signal_.value, int64_t(value), std::memory_order_relaxed);
}

void BusyWaitSignal::SubAcquire(hsa_signal_value_t value) {
  RejectHostAtomicRmw();
  atomic::Sub(&signal_.value, int64_t(value), std::memory_order_acquire);
}

void BusyWaitSignal::SubRelease(hsa_signal_value_t value) {
  RejectHostAtomicRmw();
  atomic::Sub(&signal_.value, int64_t(value), std::memory_order_release);
}

void BusyWaitSignal::SubAcqRel(hsa_signal_value_t value) {
  RejectHostAtomicRmw();
  atomic::Sub(&signal_.value, int64_t(value), std::memory_order_acq_rel);
}

hsa_signal_value_t BusyWaitSignal::ExchRelaxed(hsa_signal_value_t value) {
  RejectHostAtomicRmw();
  return hsa_signal_value_t(atomic::Exchange(&signal_.value, int64_t(value),
                                             std::memory_order_relaxed));
}

hsa_signal_value_t BusyWaitSignal::ExchAcquire(hsa_signal_value_t value) {
  RejectHostAtomicRmw();
  return hsa_signal_value_t(atomic::Exchange(&signal_.value, int64_t(value),
                                             std::memory_order_acquire));
}

hsa_signal_value_t BusyWaitSignal::ExchRelease(hsa_signal_value_t value) {
  RejectHostAtomicRmw();
  return hsa_signal_value_t(atomic::Exchange(&signal_.value, int64_t(value),
                                             std::memory_order_release));
}

hsa_signal_value_t BusyWaitSignal::ExchAcqRel(hsa_signal_value_t value) {
  RejectHostAtomicRmw();
  return hsa_signal_value_t(atomic::Exchange(&signal_.value, int64_t(value),
                                             std::memory_order_acq_rel));
}

hsa_signal_value_t BusyWaitSignal::CasRelaxed(hsa_signal_value_t expected,
                                              hsa_signal_value_t value) {
  RejectHostAtomicRmw();
  return hsa_signal_value_t(atomic::Cas(&signal_.value, int64_t(value),
                                        int64_t(expected),
                                        std::memory_order_relaxed));
}

hsa_signal_value_t BusyWaitSignal::CasAcquire(hsa_signal_value_t expected,
                                              hsa_signal_value_t value) {
  RejectHostAtomicRmw();
  return hsa_signal_value_t(atomic::Cas(&signal_.value, int64_t(value),
                                        int64_t(expected),
                                        std::memory_order_acquire));
}

hsa_signal_value_t BusyWaitSignal::CasRelease(hsa_signal_value_t expected,
                                              hsa_signal_value_t value) {
  RejectHostAtomicRmw();
  return hsa_signal_value_t(atomic::Cas(&signal_.value, int64_t(value),
                                        int64_t(expected),
                                        std::memory_order_release));
}

hsa_signal_value_t BusyWaitSignal::CasAcqRel(hsa_signal_value_t expected,
                                             hsa_signal_value_t value) {
  RejectHostAtomicRmw();
  return hsa_signal_value_t(atomic::Cas(&signal_.value, int64_t(value),
                                        int64_t(expected),
                                        std::memory_order_acq_rel));
}

}  // namespace core
}  // namespace rocr
