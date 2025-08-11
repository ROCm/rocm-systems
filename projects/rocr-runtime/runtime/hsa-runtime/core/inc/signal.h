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

// HSA runtime C++ interface file.

#ifndef HSA_RUNTME_CORE_INC_SIGNAL_H_
#define HSA_RUNTME_CORE_INC_SIGNAL_H_

#include <map>
#include <functional>
#include <memory>
#include <vector>
#include <utility>

#include "hsakmt/hsakmt.h"

#include "core/common/shared.h"

#include "core/inc/checked.h"
#include "core/inc/exceptions.h"

#include "core/util/utils.h"
#include "core/util/locks.h"
#include "core/util/timer.h"

#include "inc/amd_hsa_signal.h"

#if defined(__i386__) || defined(__x86_64__)
#include <mwaitxintrin.h>
#ifndef MWAITX_ECX_TIMER_ENABLE
#define MWAITX_ECX_TIMER_ENABLE 0x2  // BIT(1)
#endif
#endif

// Allow hsa_signal_t to be keys in STL structures.
namespace std {
template <> struct less<hsa_signal_t> {
  __forceinline bool operator()(const hsa_signal_t& x, const hsa_signal_t& y) const {
    return x.handle < y.handle;
  }
  typedef hsa_signal_t first_argument_type;
  typedef hsa_signal_t second_argument_type;
  typedef bool result_type;
};
}

namespace rocr {
namespace timer {
inline timer::fast_clock::duration GetFastTimeout(uint64_t timeout) {
  uint64_t hsa_freq = 0;
  HSA::hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &hsa_freq);
  return timer::duration_from_seconds<timer::fast_clock::duration>(
      double(timeout) / double(hsa_freq));
}

inline void CheckAbortTimeout(const timer::fast_clock::time_point& start_time,
                            uint32_t signal_abort_timeout) {
  if (signal_abort_timeout) {
    const timer::fast_clock::duration abort_timeout =
        std::chrono::seconds(signal_abort_timeout);
    if (timer::fast_clock::now() - start_time > abort_timeout) {
      throw AMD::hsa_exception(HSA_STATUS_ERROR_FATAL,
                             "Signal wait abort timeout.\n");
    }
  }
}

inline void DoMwaitx(int64_t* addr, uint32_t timeout, bool timer_enable = false) {
#if defined(__i386__) || defined(__x86_64__)
  _mm_monitorx(addr, 0, 0);
  _mm_mwaitx(0, timeout, timer_enable ? MWAITX_ECX_TIMER_ENABLE : 0);
#endif
}
} // namespace timer

inline bool CheckSignalCondition(int64_t value, hsa_signal_condition_t condition,
                               hsa_signal_value_t compare_value) {
  switch (condition) {
    case HSA_SIGNAL_CONDITION_EQ:
      return value == compare_value;
    case HSA_SIGNAL_CONDITION_NE:
      return value != compare_value;
    case HSA_SIGNAL_CONDITION_GTE:
      return value >= compare_value;
    case HSA_SIGNAL_CONDITION_LT:
      return value < compare_value;
    default:
      return false;
  }
}

namespace core {
class Agent;
class Signal;

/// @brief ABI and object conversion struct for signals.  May be shared between processes.
struct SharedSignal {
  amd_signal_t amd_signal;
  uint64_t sdma_start_ts;
  Signal* core_signal;
  Check<0x71FCCA6A3D5D5276, true> id;
  uint8_t reserved[8];
  uint64_t sdma_end_ts;
  uint8_t reserved2[24];

  SharedSignal() :
    sdma_start_ts(0),
    reserved{},
    sdma_end_ts(0),
    reserved2{} {
    memset(&amd_signal, 0, sizeof(amd_signal));
    amd_signal.kind = AMD_SIGNAL_KIND_INVALID;
    core_signal = nullptr;
  }

  bool IsValid() const { return (Convert(this).handle != 0) && id.IsValid(); }

  bool IsIPC() const { return core_signal == nullptr; }

  void GetSdmaTsAddresses(uint64_t*& start, uint64_t*& end) {
    /*
    SDMA timestamps on gfx7xx/8xxx require 32 byte alignment (gfx9xx relaxes
    alignment to 8 bytes).  This conflicts with the frozen format for amd_signal_t
    so we place the time stamps in sdma_start/end_ts instead (amd_signal.start_ts
    is also properly aligned).  Reading of the timestamps occurs in GetRawTs().
    */
    start = &sdma_start_ts;
    end = &sdma_end_ts;
  }

  void CopyPrep() {
    // Clear sdma_end_ts before a copy so we can detect if the copy was done via
    // SDMA or blit kernel.
    sdma_start_ts = 0;
    sdma_end_ts = 0;
  }

  void GetRawTs(bool FetchCopyTs, uint64_t& start, uint64_t& end) {
    /*
    If the read is for a copy we need to check if it was done by blit kernel or SDMA.
    Since we clear sdma_start/end_ts during CopyPrep we know it was a SDMA copy if one
    of those is non-zero.  Otherwise return compute kernel stamps from amd_signal.
    */
    if (FetchCopyTs && sdma_end_ts != 0) {
      start = sdma_start_ts;
      end = sdma_end_ts;
      return;
    }
    start = amd_signal.start_ts;
    end = amd_signal.end_ts;
  }

  static __forceinline SharedSignal* Convert(hsa_signal_t signal) {
    SharedSignal* ret = reinterpret_cast<SharedSignal*>(static_cast<uintptr_t>(signal.handle) -
                                                        offsetof(SharedSignal, amd_signal));
    return ret;
  }

  static __forceinline hsa_signal_t Convert(const SharedSignal* signal) {
    assert(signal != nullptr && "Conversion on null Signal object.");
    const uint64_t handle = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&signal->amd_signal));
    const hsa_signal_t signal_handle = {handle};
    return signal_handle;
  }
};
static_assert(std::is_standard_layout<SharedSignal>::value,
              "SharedSignal must remain standard layout for IPC use.");
static_assert(std::is_trivially_destructible<SharedSignal>::value,
              "SharedSignal must not be modified on delete for IPC use.");
static_assert((offsetof(SharedSignal, sdma_start_ts) % 32) == 0,
              "Bad SDMA time stamp alignment.");
static_assert((offsetof(SharedSignal, sdma_end_ts) % 32) == 0,
              "Bad SDMA time stamp alignment.");
static_assert(sizeof(SharedSignal) == 128,
              "Bad SharedSignal size.");


#define SIGNAL_PREALLOC_BLOCKS 512 //16K Signals

/// @brief Pool class for SharedSignal suitable for use with Shared.
class SharedSignalPool_t : private BaseShared {
 public:
  SharedSignalPool_t() : block_size_(SIGNAL_PREALLOC_BLOCKS * minblock_) {}
  ~SharedSignalPool_t() { clear(); }

  SharedSignal* alloc();
  void free(SharedSignal* ptr);
  void clear();

 private:
  static const size_t minblock_ = 4096 / sizeof(SharedSignal);
  HybridMutex lock_;
  std::vector<SharedSignal*> free_list_;
  std::vector<std::pair<void*, size_t>> block_list_;
  size_t block_size_;
};

class LocalSignal {
 public:
  // Temporary, for legacy tools lib support.
  explicit LocalSignal(hsa_signal_value_t initial_value) {
    local_signal_.shared_object()->amd_signal.value = initial_value;
  }
  LocalSignal(hsa_signal_value_t initial_value, bool exportable);

  SharedSignal* signal() const { return local_signal_.shared_object(); }

 private:
  Shared<SharedSignal, SharedSignalPool_t> local_signal_;
};

/// @brief An abstract base class which helps implement the public hsa_signal_t
/// type (an opaque handle) and its associated APIs. At its core, signal uses
/// a 32 or 64 bit value. This value can be waitied on or signaled atomically
/// using specified memory ordering semantics.
class Signal {
 public:
  /// @brief Constructor Links and publishes the signal interface object.
  explicit Signal(SharedSignal* abi_block, bool enableIPC = false)
      : signal_(abi_block->amd_signal), async_copy_agent_(NULL), refcount_(1) {
    assert(abi_block != nullptr && "Signal abi_block must not be NULL");

    waiting_ = 0;
    retained_ = 1;

    if (enableIPC) {
      abi_block->core_signal = nullptr;
      registerIpc();
    } else {
      abi_block->core_signal = this;
    }
  }

  /// @brief Interface to discard a signal handle (hsa_signal_t)
  /// Decrements signal ref count and invokes doDestroySignal() when
  /// Signal is no longer in use.
  void DestroySignal() {
    // If handle is now invalid wake any retained sleepers.
    if (--refcount_ == 0) CasRelaxed(0, 0);
    // Release signal, last release will destroy the object.
    Release();
  }

  /// @brief Converts from this interface class to the public
  /// hsa_signal_t type - an opaque handle.
  static __forceinline hsa_signal_t Convert(Signal* signal) {
    assert(signal != nullptr && "Conversion on null Signal object.");
    const uint64_t handle = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&signal->signal_));
    const hsa_signal_t signal_handle = {handle};
    return signal_handle;
  }

  /// @brief Converts from this interface class to the public
  /// hsa_signal_t type - an opaque handle.
  static __forceinline const hsa_signal_t Convert(const Signal* signal) {
    assert(signal != nullptr && "Conversion on null Signal object.");
    const uint64_t handle = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&signal->signal_));
    const hsa_signal_t signal_handle = {handle};
    return signal_handle;
  }

  /// @brief Converts from public hsa_signal_t type (an opaque handle) to
  /// this interface class object.
  static __forceinline Signal* Convert(hsa_signal_t signal) {
    if (signal.handle == 0) throw AMD::hsa_exception(HSA_STATUS_ERROR_INVALID_ARGUMENT, "");
    SharedSignal* shared = SharedSignal::Convert(signal);
    if (!shared->IsValid())
      throw AMD::hsa_exception(HSA_STATUS_ERROR_INVALID_SIGNAL, "Signal handle is invalid.");
    if (shared->IsIPC()) {
      Signal* ret = lookupIpc(signal);
      if (ret == nullptr)
        throw AMD::hsa_exception(HSA_STATUS_ERROR_INVALID_SIGNAL, "Signal handle is invalid.");
      return ret;
    } else {
      return shared->core_signal;
    }
  }

  static Signal* DuplicateHandle(hsa_signal_t signal) {
    if (signal.handle == 0) return nullptr;
    SharedSignal* shared = SharedSignal::Convert(signal);

    if (!shared->IsIPC()) {
      if (!shared->IsValid()) return nullptr;
      shared->core_signal->refcount_++;
      shared->core_signal->Retain();
      return shared->core_signal;
    }

    // IPC signals may only be duplicated while holding the ipcMap lock.
    return duplicateIpc(signal);
  }

  bool IsValid() const { return refcount_ != 0; }

  bool __forceinline isIPC() const { return SharedSignal::Convert(Convert(this))->IsIPC(); }

  // Below are various methods corresponding to the APIs, which load/store the
  // signal value or modify the existing signal value automically and with
  // specified memory ordering semantics.
  virtual hsa_signal_value_t LoadRelaxed() = 0;
  virtual hsa_signal_value_t LoadAcquire() = 0;

  virtual void StoreRelaxed(hsa_signal_value_t value) = 0;
  virtual void StoreRelease(hsa_signal_value_t value) = 0;

  virtual hsa_signal_value_t WaitRelaxed(hsa_signal_condition_t condition,
                                         hsa_signal_value_t compare_value,
                                         uint64_t timeout,
                                         hsa_wait_state_t wait_hint) = 0;
  virtual hsa_signal_value_t WaitAcquire(hsa_signal_condition_t condition,
                                         hsa_signal_value_t compare_value,
                                         uint64_t timeout,
                                         hsa_wait_state_t wait_hint) = 0;

  virtual void AndRelaxed(hsa_signal_value_t value) = 0;
  virtual void AndAcquire(hsa_signal_value_t value) = 0;
  virtual void AndRelease(hsa_signal_value_t value) = 0;
  virtual void AndAcqRel(hsa_signal_value_t value) = 0;

  virtual void OrRelaxed(hsa_signal_value_t value) = 0;
  virtual void OrAcquire(hsa_signal_value_t value) = 0;
  virtual void OrRelease(hsa_signal_value_t value) = 0;
  virtual void OrAcqRel(hsa_signal_value_t value) = 0;

  virtual void XorRelaxed(hsa_signal_value_t value) = 0;
  virtual void XorAcquire(hsa_signal_value_t value) = 0;
  virtual void XorRelease(hsa_signal_value_t value) = 0;
  virtual void XorAcqRel(hsa_signal_value_t value) = 0;

  virtual void AddRelaxed(hsa_signal_value_t value) = 0;
  virtual void AddAcquire(hsa_signal_value_t value) = 0;
  virtual void AddRelease(hsa_signal_value_t value) = 0;
  virtual void AddAcqRel(hsa_signal_value_t value) = 0;

  virtual void SubRelaxed(hsa_signal_value_t value) = 0;
  virtual void SubAcquire(hsa_signal_value_t value) = 0;
  virtual void SubRelease(hsa_signal_value_t value) = 0;
  virtual void SubAcqRel(hsa_signal_value_t value) = 0;

  virtual hsa_signal_value_t ExchRelaxed(hsa_signal_value_t value) = 0;
  virtual hsa_signal_value_t ExchAcquire(hsa_signal_value_t value) = 0;
  virtual hsa_signal_value_t ExchRelease(hsa_signal_value_t value) = 0;
  virtual hsa_signal_value_t ExchAcqRel(hsa_signal_value_t value) = 0;

  virtual hsa_signal_value_t CasRelaxed(hsa_signal_value_t expected,
                                        hsa_signal_value_t value) = 0;
  virtual hsa_signal_value_t CasAcquire(hsa_signal_value_t expected,
                                        hsa_signal_value_t value) = 0;
  virtual hsa_signal_value_t CasRelease(hsa_signal_value_t expected,
                                        hsa_signal_value_t value) = 0;
  virtual hsa_signal_value_t CasAcqRel(hsa_signal_value_t expected,
                                       hsa_signal_value_t value) = 0;

  //-------------------------
  // implementation specific
  //-------------------------
  typedef void* rtti_t;

  /// @brief Returns the address of the value.
  virtual hsa_signal_value_t* ValueLocation() const = 0;

  /// @brief Applies only to InterrupEvent type, returns the event used to.
  /// Returns NULL for DefaultEvent Type.
  virtual HsaEvent* EopEvent() = 0;

  /// @brief Waits until multiple signals in the list satisfy their conditions
  /// or a timeout is reached.
  /// @param signal_count Number of hsa_signals in the list.
  /// @param hsa_signals Pointer to array of HSA signals.
  /// @param conds Pointer to array of signal conditions.
  /// @param values Pointer to array of signal values.
  /// @param timeout Timeout hint value.
  /// @param wait_hint Hint about wait state.
  /// @param satisfying_values Vector of satisfying values. If \p wait_on_all
  /// is false (then we are waiting on any signal in the list) this will contain
  /// only the first satisfying value.
  /// @param wait_on_all Wait on all signals in the list to satisfy their
  /// conditions if true, else wait on any signal in the list to satisfy its
  /// condition.
  /// @return Return the index of the first signal in the list that satisfies
  /// its condition or -1 on a timeout. Note that if \p wait_on_all is true,
  /// then all signals in the list satisfy their conditions, thus the index will
  /// always be 0.
  static uint32_t WaitMultiple(uint32_t signal_count, const hsa_signal_t* hsa_signals,
                               const hsa_signal_condition_t* conds,
                               const hsa_signal_value_t* values, uint64_t timeout,
                               hsa_wait_state_t wait_hint,
                               std::vector<hsa_signal_value_t>& satisfying_values,
                               bool wait_on_all);

  /// @brief Dedicated funtion to wait on signals that are not of type HSA_EVENTTYPE_SIGNAL
  /// these events can only be received by calling the underlying driver (i.e via the hsaKmtWaitOnMultipleEvents_Ext
  /// function call). We still need to have 1 signal of type HSA_EVENT_TYPE_SIGNAL attached to the list of signals
  /// to be able to force hsaKmtWaitOnMultipleEvents_Ext to return.
  /// @param signal_count Number of hsa_signals
  /// @param hsa_signals Pointer to array of signals. All signals should have a valid EopEvent()
  /// @param conds list of conditions
  /// @param values list of values
  /// @param satisfying_value value to be satisfied
  /// @return index of signal that satisfies condition
  static uint32_t WaitAnyExceptions(uint32_t signal_count, const hsa_signal_t* hsa_signals,
                         const hsa_signal_condition_t* conds, const hsa_signal_value_t* values,
                         hsa_signal_value_t* satisfying_value);

  __forceinline bool IsType(rtti_t id) { return _IsA(id); }

  /// @brief Prevents the signal from being destroyed until the matching Release().
  void Retain() { retained_++; }
  void Release();

  /// @brief Checks if signal is currently in use by a wait API.
  bool InWaiting() const { return waiting_ != 0; }

  /// @brief Increments the waiting indicator.
  void WaitingInc() { waiting_++; }

  /// @brief Decrements the waiting indicator.
  void WaitingDec() { waiting_--; }

  // Prep for copy profiling.  Store copy agent and ready API block.
  __forceinline void async_copy_agent(core::Agent* agent) {
    async_copy_agent_ = agent;
    core::SharedSignal::Convert(Convert(this))->CopyPrep();
  }

  __forceinline core::Agent* async_copy_agent() { return async_copy_agent_; }

  void GetSdmaTsAddresses(uint64_t*& start, uint64_t*& end) {
    core::SharedSignal::Convert(Convert(this))->GetSdmaTsAddresses(start, end);
  }

  // Set FetchCopyTs = true when reading time stamps from a copy operation.
  void GetRawTs(bool FetchCopyTs, uint64_t& start, uint64_t& end) {
    core::SharedSignal::Convert(Convert(this))->GetRawTs(FetchCopyTs, start, end);
  }

  /// @brief Structure which defines key signal elements like type and value.
  /// Address of this struct is used as a value for the opaque handle of type
  /// hsa_signal_t provided to the public API.
  amd_signal_t& signal_;

 protected:
  virtual ~Signal();

  /// @brief Overrideable deletion function
  virtual void doDestroySignal() { delete this; }

  /// @brief Simple RTTI type checking helper
  /// Returns true if the object can be converted to the query type via
  /// static_cast.
  /// Do not use directly.  Use IsType in the desired derived type instead.
  virtual bool _IsA(rtti_t id) const = 0;

  /// @variable Indicates number of runtime threads waiting on this signal.
  /// Value of zero means no waits.
  std::atomic<uint32_t> waiting_;

  /// @variable Pointer to agent used to perform an async copy.
  core::Agent* async_copy_agent_;

 private:
  static KernelMutex ipcLock_;
  static std::map<decltype(hsa_signal_t::handle), Signal*> ipcMap_;

  static Signal* lookupIpc(hsa_signal_t signal);
  static Signal* duplicateIpc(hsa_signal_t signal);

  /// @variable Ref count of this signal's handle (see IPC APIs)
  std::atomic<uint32_t> refcount_;

  /// @variable Count of handle references and Retain() calls for this handle (see IPC APIs)
  std::atomic<uint32_t> retained_;

  void registerIpc();
  bool deregisterIpc();

  DISALLOW_COPY_AND_ASSIGN(Signal);
};

/// @brief Handle signal operations which are not for use on doorbells.
class DoorbellSignal : public Signal {
 public:
  using Signal::Signal;

  /// @brief This operation is illegal
  hsa_signal_value_t LoadRelaxed() final override {
    assert(false);
    return 0;
  }

  /// @brief This operation is illegal
  hsa_signal_value_t LoadAcquire() final override {
    assert(false);
    return 0;
  }

  /// @brief This operation is illegal
  hsa_signal_value_t WaitRelaxed(hsa_signal_condition_t condition, hsa_signal_value_t compare_value,
                                 uint64_t timeout, hsa_wait_state_t wait_hint) final override {
    assert(false);
    return 0;
  }

  /// @brief This operation is illegal
  hsa_signal_value_t WaitAcquire(hsa_signal_condition_t condition, hsa_signal_value_t compare_value,
                                 uint64_t timeout, hsa_wait_state_t wait_hint) final override {
    assert(false);
    return 0;
  }

  /// @brief This operation is illegal
  void AndRelaxed(hsa_signal_value_t value) final override { assert(false); }

  /// @brief This operation is illegal
  void AndAcquire(hsa_signal_value_t value) final override { assert(false); }

  /// @brief This operation is illegal
  void AndRelease(hsa_signal_value_t value) final override { assert(false); }

  /// @brief This operation is illegal
  void AndAcqRel(hsa_signal_value_t value) final override { assert(false); }

  /// @brief This operation is illegal
  void OrRelaxed(hsa_signal_value_t value) final override { assert(false); }

  /// @brief This operation is illegal
  void OrAcquire(hsa_signal_value_t value) final override { assert(false); }

  /// @brief This operation is illegal
  void OrRelease(hsa_signal_value_t value) final override { assert(false); }

  /// @brief This operation is illegal
  void OrAcqRel(hsa_signal_value_t value) final override { assert(false); }

  /// @brief This operation is illegal
  void XorRelaxed(hsa_signal_value_t value) final override { assert(false); }

  /// @brief This operation is illegal
  void XorAcquire(hsa_signal_value_t value) final override { assert(false); }

  /// @brief This operation is illegal
  void XorRelease(hsa_signal_value_t value) final override { assert(false); }

  /// @brief This operation is illegal
  void XorAcqRel(hsa_signal_value_t value) final override { assert(false); }

  /// @brief This operation is illegal
  void AddRelaxed(hsa_signal_value_t value) final override { assert(false); }

  /// @brief This operation is illegal
  void AddAcquire(hsa_signal_value_t value) final override { assert(false); }

  /// @brief This operation is illegal
  void AddRelease(hsa_signal_value_t value) final override { assert(false); }

  /// @brief This operation is illegal
  void AddAcqRel(hsa_signal_value_t value) final override { assert(false); }

  /// @brief This operation is illegal
  void SubRelaxed(hsa_signal_value_t value) final override { assert(false); }

  /// @brief This operation is illegal
  void SubAcquire(hsa_signal_value_t value) final override { assert(false); }

  /// @brief This operation is illegal
  void SubRelease(hsa_signal_value_t value) final override { assert(false); }

  /// @brief This operation is illegal
  void SubAcqRel(hsa_signal_value_t value) final override { assert(false); }

  /// @brief This operation is illegal
  hsa_signal_value_t ExchRelaxed(hsa_signal_value_t value) final override {
    assert(false);
    return 0;
  }

  /// @brief This operation is illegal
  hsa_signal_value_t ExchAcquire(hsa_signal_value_t value) final override {
    assert(false);
    return 0;
  }

  /// @brief This operation is illegal
  hsa_signal_value_t ExchRelease(hsa_signal_value_t value) final override {
    assert(false);
    return 0;
  }

  /// @brief This operation is illegal
  hsa_signal_value_t ExchAcqRel(hsa_signal_value_t value) final override {
    assert(false);
    return 0;
  }

  /// @brief This operation is illegal
  hsa_signal_value_t CasRelaxed(hsa_signal_value_t expected,
                                hsa_signal_value_t value) final override {
    assert(false);
    return 0;
  }

  /// @brief This operation is illegal
  hsa_signal_value_t CasAcquire(hsa_signal_value_t expected,
                                hsa_signal_value_t value) final override {
    assert(false);
    return 0;
  }

  /// @brief This operation is illegal
  hsa_signal_value_t CasRelease(hsa_signal_value_t expected,
                                hsa_signal_value_t value) final override {
    assert(false);
    return 0;
  }

  /// @brief This operation is illegal
  hsa_signal_value_t CasAcqRel(hsa_signal_value_t expected,
                               hsa_signal_value_t value) final override {
    assert(false);
    return 0;
  }

  /// @brief This operation is illegal
  hsa_signal_value_t* ValueLocation() const final override {
    assert(false);
    return NULL;
  }

  /// @brief This operation is illegal
  HsaEvent* EopEvent() final override {
    assert(false);
    return NULL;
  }

 protected:
  /// @brief Disallow destroying doorbell apart from its queue.
  void doDestroySignal() final override { assert(false); }
};

struct hsa_signal_handle {
  hsa_signal_t signal;

  hsa_signal_handle() {}
  hsa_signal_handle(hsa_signal_t Signal) { signal = Signal; }
  operator hsa_signal_t() { return signal; }
  Signal* operator->() { return core::Signal::Convert(signal); }
};
static_assert(
    sizeof(hsa_signal_handle) == sizeof(hsa_signal_t),
    "hsa_signal_handle and hsa_signal_t must have identical binary layout.");
static_assert(
    sizeof(hsa_signal_handle[2]) == sizeof(hsa_signal_t[2]),
    "hsa_signal_handle and hsa_signal_t must have identical binary layout.");

class SignalGroup : public Checked<0xBD35DDDD578F091> {
 public:
  static __forceinline hsa_signal_group_t Convert(SignalGroup* group) {
    const hsa_signal_group_t handle = {static_cast<uint64_t>(reinterpret_cast<uintptr_t>(group))};
    return handle;
  }
  static __forceinline SignalGroup* Convert(hsa_signal_group_t group) {
    return reinterpret_cast<SignalGroup*>(static_cast<uintptr_t>(group.handle));
  }

  SignalGroup(uint32_t num_signals, const hsa_signal_t* signals);
  ~SignalGroup() { delete[] signals; }

  bool IsValid() const {
    if (CheckedType::IsValid() && signals != NULL) return true;
    return false;
  }

  const hsa_signal_t* List() const { return signals; }
  uint32_t Count() const { return count; }

 private:
  hsa_signal_t* signals;
  const uint32_t count;
  DISALLOW_COPY_AND_ASSIGN(SignalGroup);
};

class SignalDeleter {
 public:
  void operator()(Signal* ptr) { ptr->DestroySignal(); }
};
using unique_signal_ptr = ::std::unique_ptr<core::Signal, SignalDeleter>;

}  // namespace core
}  // namespace rocr
#endif  // header guard
