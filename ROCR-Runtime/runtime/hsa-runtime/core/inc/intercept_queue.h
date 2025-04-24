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

#ifndef HSA_RUNTIME_CORE_INC_INTERCEPT_QUEUE_H_
#define HSA_RUNTIME_CORE_INC_INTERCEPT_QUEUE_H_

#include <vector>
#include <memory>
#include <utility>

#include "core/inc/runtime.h"
#include "core/inc/queue.h"
#include "core/inc/signal.h"
#include "core/inc/interrupt_signal.h"
#include "core/inc/exceptions.h"
#include "core/util/locks.h"

namespace rocr {
namespace core {

// @brief Generic container to forward Queue interfaces into Queue* member.
// Class only has utility as a base type customized Queue wrappers.
class QueueWrapper : public Queue {
 public:
  std::unique_ptr<Queue> wrapped;

  explicit QueueWrapper(std::unique_ptr<Queue> queue)
      : Queue(static_cast<core::SharedQueue*>(core::Runtime::runtime_singleton_->system_allocator()(
                  sizeof(core::SharedQueue), 4096, 0, 0)),
              0),
        wrapped(std::move(queue)) {
    memcpy(&amd_queue_, &wrapped->amd_queue_, sizeof(amd_queue_));
    wrapped->set_public_handle(wrapped.get(), public_handle_);
  }

  ~QueueWrapper() {
    if (shared_queue_) core::Runtime::runtime_singleton_->system_deallocator()(shared_queue_);
  }

  hsa_status_t Inactivate() override { return wrapped->Inactivate(); }
  hsa_status_t SetPriority(HSA_QUEUE_PRIORITY priority) override {
    return wrapped->SetPriority(priority);
  }
  uint64_t LoadReadIndexAcquire() override { return wrapped->LoadReadIndexAcquire(); }
  uint64_t LoadReadIndexRelaxed() override { return wrapped->LoadReadIndexRelaxed(); }
  uint64_t LoadWriteIndexRelaxed() override { return wrapped->LoadWriteIndexRelaxed(); }
  uint64_t LoadWriteIndexAcquire() override { return wrapped->LoadWriteIndexAcquire(); }
  void StoreReadIndexRelaxed(uint64_t value) override {
    return wrapped->StoreReadIndexRelaxed(value);
  }
  void StoreReadIndexRelease(uint64_t value) override {
    return wrapped->StoreReadIndexRelease(value);
  }
  void StoreWriteIndexRelaxed(uint64_t value) override {
    return wrapped->StoreWriteIndexRelaxed(value);
  }
  void StoreWriteIndexRelease(uint64_t value) override {
    return wrapped->StoreWriteIndexRelease(value);
  }
  uint64_t CasWriteIndexAcqRel(uint64_t expected, uint64_t value) override {
    return wrapped->CasWriteIndexAcqRel(expected, value);
  }
  uint64_t CasWriteIndexAcquire(uint64_t expected, uint64_t value) override {
    return wrapped->CasWriteIndexAcquire(expected, value);
  }
  uint64_t CasWriteIndexRelaxed(uint64_t expected, uint64_t value) override {
    return wrapped->CasWriteIndexRelaxed(expected, value);
  }
  uint64_t CasWriteIndexRelease(uint64_t expected, uint64_t value) override {
    return wrapped->CasWriteIndexRelease(expected, value);
  }
  uint64_t AddWriteIndexAcqRel(uint64_t value) override {
    return wrapped->AddWriteIndexAcqRel(value);
  }
  uint64_t AddWriteIndexAcquire(uint64_t value) override {
    return wrapped->AddWriteIndexAcquire(value);
  }
  uint64_t AddWriteIndexRelaxed(uint64_t value) override {
    return wrapped->AddWriteIndexRelaxed(value);
  }
  uint64_t AddWriteIndexRelease(uint64_t value) override {
    return wrapped->AddWriteIndexRelease(value);
  }
  hsa_status_t SetCUMasking(uint32_t num_cu_mask_count, const uint32_t* cu_mask) override {
    return wrapped->SetCUMasking(num_cu_mask_count, cu_mask);
  }
  hsa_status_t GetCUMasking(uint32_t num_cu_mask_count, uint32_t* cu_mask) override {
    return wrapped->GetCUMasking(num_cu_mask_count, cu_mask);
  }
  void ExecutePM4(uint32_t* cmd_data, size_t cmd_size_b,
                  hsa_fence_scope_t acquireFence = HSA_FENCE_SCOPE_NONE,
                  hsa_fence_scope_t releaseFence = HSA_FENCE_SCOPE_NONE,
                  hsa_signal_t* signal = NULL) override {
    wrapped->ExecutePM4(cmd_data, cmd_size_b, acquireFence, releaseFence, signal);
  }
  void SetProfiling(bool enabled) override { wrapped->SetProfiling(enabled); }

 protected:
  void do_set_public_handle(hsa_queue_t* handle) override {
    public_handle_ = handle;
    wrapped->set_public_handle(wrapped.get(), handle);
  }
};

// @brief Generic container for a proxy queue.
// Presents an proxy packet buffer and doorbell signal for an underlying Queue.  Write index
// operations act on the proxy buffer while all other operations pass through to the underlying
// queue.
class QueueProxy : public QueueWrapper {
 public:
  explicit QueueProxy(std::unique_ptr<Queue> queue) : QueueWrapper(std::move(queue)) {}

  uint64_t LoadReadIndexAcquire() override {
    return atomic::Load(&amd_queue_.read_dispatch_id, std::memory_order_acquire);
  }
  uint64_t LoadReadIndexRelaxed() override {
    return atomic::Load(&amd_queue_.read_dispatch_id, std::memory_order_relaxed);
  }
  void StoreReadIndexRelaxed(uint64_t value) override { assert(false); }
  void StoreReadIndexRelease(uint64_t value) override { assert(false); }

  uint64_t LoadWriteIndexRelaxed() override {
    return atomic::Load(&amd_queue_.write_dispatch_id, std::memory_order_relaxed);
  }
  uint64_t LoadWriteIndexAcquire() override {
    return atomic::Load(&amd_queue_.write_dispatch_id, std::memory_order_acquire);
  }
  void StoreWriteIndexRelaxed(uint64_t value) override {
    atomic::Store(&amd_queue_.write_dispatch_id, value, std::memory_order_relaxed);
  }
  void StoreWriteIndexRelease(uint64_t value) override {
    atomic::Store(&amd_queue_.write_dispatch_id, value, std::memory_order_release);
  }
  uint64_t CasWriteIndexAcqRel(uint64_t expected, uint64_t value) override {
    return atomic::Cas(&amd_queue_.write_dispatch_id, value, expected, std::memory_order_acq_rel);
  }
  uint64_t CasWriteIndexAcquire(uint64_t expected, uint64_t value) override {
    return atomic::Cas(&amd_queue_.write_dispatch_id, value, expected, std::memory_order_acquire);
  }
  uint64_t CasWriteIndexRelaxed(uint64_t expected, uint64_t value) override {
    return atomic::Cas(&amd_queue_.write_dispatch_id, value, expected, std::memory_order_relaxed);
  }
  uint64_t CasWriteIndexRelease(uint64_t expected, uint64_t value) override {
    return atomic::Cas(&amd_queue_.write_dispatch_id, value, expected, std::memory_order_release);
  }
  uint64_t AddWriteIndexAcqRel(uint64_t value) override {
    return atomic::Add(&amd_queue_.write_dispatch_id, value, std::memory_order_acq_rel);
  }
  uint64_t AddWriteIndexAcquire(uint64_t value) override {
    return atomic::Add(&amd_queue_.write_dispatch_id, value, std::memory_order_acquire);
  }
  uint64_t AddWriteIndexRelaxed(uint64_t value) override {
    return atomic::Add(&amd_queue_.write_dispatch_id, value, std::memory_order_relaxed);
  }
  uint64_t AddWriteIndexRelease(uint64_t value) override {
    return atomic::Add(&amd_queue_.write_dispatch_id, value, std::memory_order_release);
  }
};

// @brief Provides packet intercept and rewrite capability for a queue.
// Host-side dispatches are processed during doorbell ring.
// Device-side dispatches are processed as an asynchronous signal event.
class InterceptQueue : public QueueProxy, private LocalSignal, public DoorbellSignal {
 public:
  explicit InterceptQueue(std::unique_ptr<Queue> queue);
  ~InterceptQueue();

  void AddInterceptor(hsa_amd_queue_intercept_handler interceptor, void* data) {
    assert(interceptor != nullptr && "Packet intercept callback was nullptr.");
    interceptors.push_back(std::make_pair(interceptor, data));
  }

  hsa_status_t Inactivate() override {
    active_ = false;
    return wrapped->Inactivate();
  }

 private:
  // Serialize packet interception processing.
  KernelMutex lock_;

  // Largest processed packet index.
  uint64_t next_packet_;

  // Post interception packet overflow buffer
  std::vector<AqlPacket> overflow_;

  // Index at which async intercept processing was scheduled.
  uint64_t retry_index_;

  // Given the current value of the wrapped queue read index, determine if
  // there is a retry barrier packet already in the wrapped queue.
  bool IsPendingRetryPoint(uint64_t wrapped_current_read_index) const;

  // Event signal to use for async packet processing and control flag.
  InterruptSignal* async_doorbell_;
  std::atomic<bool> quit_;

  // Indicates queue active/inactive state.
  std::atomic<bool> active_;

  // Proxy packet buffer
  SharedArray<AqlPacket, 4096> buffer_;

  // Packet transform callbacks
  std::vector<std::pair<AMD::callback_t<hsa_amd_queue_intercept_handler>, void*>> interceptors;

  static const hsa_signal_value_t DOORBELL_MAX = 0xFFFFFFFFFFFFFFFFull;

  static bool HandleAsyncDoorbell(hsa_signal_value_t value, void* arg);
  static void PacketWriter(const void* pkts, uint64_t pkt_count);

  // Submit packets to the wrapped queue and return number of packets that were
  // submitted.
  uint64_t Submit(const AqlPacket* packets, uint64_t count);

  // Used as the final packet rewriter that submits the packets to the wrapped
  // queue.
  static void Submit(const void* pkts, uint64_t pkt_count, uint64_t user_pkt_index, void* data,
                     hsa_amd_queue_intercept_packet_writer writer);

  /*
   * Remaining Queue and Signal interface definitions.
   */
 public:
  /// @brief Update signal value using Relaxed semantics
  ///
  /// @param value Value of signal to update with
  void StoreRelaxed(hsa_signal_value_t value) override;

  /// @brief Update signal value using Release semantics
  ///
  /// @param value Value of signal to update with
  void StoreRelease(hsa_signal_value_t value) override {
    std::atomic_thread_fence(std::memory_order_release);
    StoreRelaxed(value);
  }

  /// @brief Provide information about the queue
  hsa_status_t GetInfo(hsa_queue_info_attribute_t attribute, void* value) override;

  static __forceinline bool IsType(core::Signal* signal) { return signal->IsType(&rtti_id()); }
  static __forceinline bool IsType(core::Queue* queue) { return queue->IsType(&rtti_id()); }

 protected:
  bool _IsA(Queue::rtti_t id) const override { return id == &rtti_id(); }

 private:
  static __forceinline int& rtti_id() {
    static int rtti_id_ = 0;
    return rtti_id_;
  }
};

}  // namespace core
}  // namespace rocr

#endif  // HSA_RUNTIME_CORE_INC_INTERCEPT_QUEUE_H_
