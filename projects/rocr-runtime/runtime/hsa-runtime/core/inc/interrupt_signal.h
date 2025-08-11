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

// HSA runtime C++ interface file.

#ifndef HSA_RUNTME_CORE_INC_INTERRUPT_SIGNAL_H_
#define HSA_RUNTME_CORE_INC_INTERRUPT_SIGNAL_H_

#include <memory>
#include <vector>

#include "hsakmt/hsakmt.h"

#include "core/inc/signal.h"
#include "core/util/utils.h"

namespace rocr {
namespace core {

/// @brief A Signal implementation using interrupts versus plain memory based.
/// Also see base class Signal.
///
/// Breaks common/vendor separation - signals in general needs to be re-worked
/// at the foundation level to make sense in a multi-device system.
/// Supports only one waiter for now.
/// KFD changes are needed to support multiple waiters and have device
/// signaling.
class InterruptSignal : private LocalSignal, public Signal {
 public:
  class EventPool {
   public:
    struct Deleter {
      void operator()(HsaEvent* evt) { InterruptSignal::DestroyEvent(evt); }
    };
    using unique_event_ptr = ::std::unique_ptr<HsaEvent, Deleter>;

    EventPool() : allEventsAllocated(false) {}

    HsaEvent* alloc();
    void free(HsaEvent* evt);
    void clear() {
      events_.clear();
      allEventsAllocated = false;
    }

   private:
    HybridMutex lock_;
    std::vector<unique_event_ptr> events_;
    bool allEventsAllocated;
  };

  static HsaEvent* CreateEvent(HSA_EVENTTYPE type, bool manual_reset);
  static void DestroyEvent(HsaEvent* evt);

  /// @brief Determines if a Signal* can be safely converted to an
  /// InterruptSignal* via static_cast.
  static __forceinline bool IsType(Signal* ptr) {
    return ptr->IsType(&rtti_id());
  }

  explicit InterruptSignal(hsa_signal_value_t initial_value,
                           HsaEvent* use_event = NULL);

  ~InterruptSignal();

  // Below are various methods corresponding to the APIs, which load/store the
  // signal value or modify the existing signal value automically and with
  // specified memory ordering semantics.

  hsa_signal_value_t LoadRelaxed();

  hsa_signal_value_t LoadAcquire();

  void StoreRelaxed(hsa_signal_value_t value);

  void StoreRelease(hsa_signal_value_t value);

  hsa_signal_value_t WaitRelaxed(hsa_signal_condition_t condition,
                                 hsa_signal_value_t compare_value,
                                 uint64_t timeout, hsa_wait_state_t wait_hint);

  hsa_signal_value_t WaitAcquire(hsa_signal_condition_t condition,
                                 hsa_signal_value_t compare_value,
                                 uint64_t timeout, hsa_wait_state_t wait_hint);

  void AndRelaxed(hsa_signal_value_t value);

  void AndAcquire(hsa_signal_value_t value);

  void AndRelease(hsa_signal_value_t value);

  void AndAcqRel(hsa_signal_value_t value);

  void OrRelaxed(hsa_signal_value_t value);

  void OrAcquire(hsa_signal_value_t value);

  void OrRelease(hsa_signal_value_t value);

  void OrAcqRel(hsa_signal_value_t value);

  void XorRelaxed(hsa_signal_value_t value);

  void XorAcquire(hsa_signal_value_t value);

  void XorRelease(hsa_signal_value_t value);

  void XorAcqRel(hsa_signal_value_t value);

  void AddRelaxed(hsa_signal_value_t value);

  void AddAcquire(hsa_signal_value_t value);

  void AddRelease(hsa_signal_value_t value);

  void AddAcqRel(hsa_signal_value_t value);

  void SubRelaxed(hsa_signal_value_t value);

  void SubAcquire(hsa_signal_value_t value);

  void SubRelease(hsa_signal_value_t value);

  void SubAcqRel(hsa_signal_value_t value);

  hsa_signal_value_t ExchRelaxed(hsa_signal_value_t value);

  hsa_signal_value_t ExchAcquire(hsa_signal_value_t value);

  hsa_signal_value_t ExchRelease(hsa_signal_value_t value);

  hsa_signal_value_t ExchAcqRel(hsa_signal_value_t value);

  hsa_signal_value_t CasRelaxed(hsa_signal_value_t expected,
                                hsa_signal_value_t value);

  hsa_signal_value_t CasAcquire(hsa_signal_value_t expected,
                                hsa_signal_value_t value);

  hsa_signal_value_t CasRelease(hsa_signal_value_t expected,
                                hsa_signal_value_t value);

  hsa_signal_value_t CasAcqRel(hsa_signal_value_t expected,
                               hsa_signal_value_t value);

  /// @brief See base class Signal.
  __forceinline hsa_signal_value_t* ValueLocation() const {
    return (hsa_signal_value_t*)&signal_.value;
  }

  /// @brief See base class Signal.
  __forceinline HsaEvent* EopEvent() { return event_; }

 protected:
  bool _IsA(rtti_t id) const { return id == &rtti_id(); }

 private:
  /// @variable KFD event on which the interrupt signal is based on.
  HsaEvent* event_;

  /// @variable Indicates whether the signal should release the event when it
  /// closes or not.
  bool free_event_;

  /// Used to obtain a globally unique value (address) for rtti.
  static __forceinline int& rtti_id() {
    static int rtti_id_ = 0;
    return rtti_id_;
  }

  void SetEvent();

  DISALLOW_COPY_AND_ASSIGN(InterruptSignal);
};

}  // namespace core
}  // namespace rocr
#endif  // header guard
