////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2025, Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef HSA_RUNTIME_CORE_INC_HOST_QUEUE_H_
#define HSA_RUNTIME_CORE_INC_HOST_QUEUE_H_

#include "core/inc/exceptions.h"
#include "core/inc/memory_region.h"
#include "core/inc/queue.h"
#include "core/inc/runtime.h"
#include "core/inc/signal.h"
#include "inc/amd_hsa_queue.h"

namespace rocr {
namespace core {
class HostQueue : public Queue {
 public:
  using QueueDescriptorT = amd_queue_v2_t;
  static __forceinline bool IsType(core::Queue* queue) { return queue->IsType(&rtti_id()); }

  HostQueue(core::SharedQueue* shared_queue, hsa_region_t region, uint32_t ring_size,
            hsa_queue_type32_t type, uint32_t features, hsa_signal_t doorbell_signal);

  ~HostQueue();

  hsa_status_t Inactivate() override { return HSA_STATUS_SUCCESS; }
  hsa_status_t SetPriority(HSA_QUEUE_PRIORITY priority) override {
    return HSA_STATUS_ERROR_INVALID_QUEUE;
  }

  uint64_t LoadReadIndexAcquire() override {
    return atomic::Load(&QueueDescriptor().read_dispatch_id, std::memory_order_acquire);
  }

  uint64_t LoadReadIndexRelaxed() override {
    return atomic::Load(&QueueDescriptor().read_dispatch_id, std::memory_order_relaxed);
  }

  uint64_t LoadWriteIndexAcquire() override {
    return atomic::Load(&QueueDescriptor().write_dispatch_id, std::memory_order_acquire);
  }

  uint64_t LoadWriteIndexRelaxed() override {
    return atomic::Load(&QueueDescriptor().write_dispatch_id, std::memory_order_relaxed);
  }

  void StoreReadIndexRelaxed(uint64_t value) override {
    atomic::Store(&QueueDescriptor().read_dispatch_id, value, std::memory_order_relaxed);
  }

  void StoreReadIndexRelease(uint64_t value) override {
    atomic::Store(&QueueDescriptor().read_dispatch_id, value, std::memory_order_release);
  }

  void StoreWriteIndexRelaxed(uint64_t value) override {
    atomic::Store(&QueueDescriptor().write_dispatch_id, value, std::memory_order_relaxed);
  }

  void StoreWriteIndexRelease(uint64_t value) override {
    atomic::Store(&QueueDescriptor().write_dispatch_id, value, std::memory_order_release);
  }

  uint64_t CasWriteIndexAcqRel(uint64_t expected, uint64_t value) override {
    return atomic::Cas(&QueueDescriptor().write_dispatch_id, value, expected,
                       std::memory_order_acq_rel);
  }

  uint64_t CasWriteIndexAcquire(uint64_t expected, uint64_t value) override {
    return atomic::Cas(&QueueDescriptor().write_dispatch_id, value, expected,
                       std::memory_order_acquire);
  }

  uint64_t CasWriteIndexRelaxed(uint64_t expected, uint64_t value) override {
    return atomic::Cas(&QueueDescriptor().write_dispatch_id, value, expected,
                       std::memory_order_relaxed);
  }

  uint64_t CasWriteIndexRelease(uint64_t expected, uint64_t value) override {
    return atomic::Cas(&QueueDescriptor().write_dispatch_id, value, expected,
                       std::memory_order_release);
  }

  uint64_t AddWriteIndexAcqRel(uint64_t value) override {
    return atomic::Add(&QueueDescriptor().write_dispatch_id, value, std::memory_order_acq_rel);
  }

  uint64_t AddWriteIndexAcquire(uint64_t value) override {
    return atomic::Add(&QueueDescriptor().write_dispatch_id, value, std::memory_order_acquire);
  }

  uint64_t AddWriteIndexRelaxed(uint64_t value) override {
    return atomic::Add(&QueueDescriptor().write_dispatch_id, value, std::memory_order_relaxed);
  }

  uint64_t AddWriteIndexRelease(uint64_t value) override {
    return atomic::Add(&QueueDescriptor().write_dispatch_id, value, std::memory_order_release);
  }

  hsa_status_t SetCUMasking(uint32_t num_cu_mask_count, const uint32_t* cu_mask) override {
    return HSA_STATUS_ERROR_INVALID_QUEUE;
  }

  hsa_status_t GetCUMasking(uint32_t num_cu_mask_count, uint32_t* cu_mask) override {
    return HSA_STATUS_ERROR_INVALID_QUEUE;
  }

  void ExecutePM4(uint32_t* cmd_data, size_t cmd_size_b,
                  hsa_fence_scope_t acquireFence = HSA_FENCE_SCOPE_NONE,
                  hsa_fence_scope_t releaseFence = HSA_FENCE_SCOPE_NONE,
                  hsa_signal_t* signal = NULL) override {
    assert(false && "HostQueue::ExecutePM4 is unimplemented");
  }

  void SetProfiling(bool enabled) override {
    throw AMD::hsa_exception(static_cast<hsa_status_t>(HSA_STATUS_ERROR_NOT_SUPPORTED),
                             "HostQueue::SetProfiling is not supported.");
  }

  hsa_status_t GetInfo(hsa_queue_info_attribute_t attribute, void* value) override {
    assert(false && "HostQueue::GetInfo is unimplemented");
    return HSA_STATUS_ERROR_INVALID_QUEUE;
  }

  __forceinline std::size_t SharedQueueSize() override { return shared_queue_size_; }

  __forceinline void SetInterceptQueueReadIndex(
      SharedQueue& shared_queue, volatile uint64_t*& read_dispatch_id) const override {
    read_dispatch_id =
        &(reinterpret_cast<QueueDescriptorT&>(shared_queue.hsa_interface_queue).read_dispatch_id);
  }

  void* operator new(size_t size) {
    return _aligned_malloc(AlignUp(size, HSA_QUEUE_ALIGN_BYTES), HSA_QUEUE_ALIGN_BYTES);
  }

  void* operator new(size_t size, void* ptr) { return ptr; }

  void operator delete(void* ptr) { _aligned_free(ptr); }

  void operator delete(void*, void*) {}

 protected:
  bool _IsA(Queue::rtti_t id) const override { return id == &rtti_id(); }

 private:
  static __forceinline int& rtti_id() {
    static int rtti_id_ = 0;
    return rtti_id_;
  }

  /// @brief Get a reference to this Queue implementation's queue descriptor.
  __forceinline QueueDescriptorT& QueueDescriptor() {
    return *reinterpret_cast<QueueDescriptorT*>(&shared_queue_->hsa_interface_queue);
  }

  static const size_t kRingAlignment = 256;
  const uint32_t size_;
  void* ring_;

  /// @brief This is the total size of the SharedQueue for HostQueue.
  /// @details The SharedQueue allows for conversion between a pointer to the
  /// public API queue handle (i.e., hsa_queue_t*) and a core::Queue* instance.
  /// Each concrete queue type may have its own queue descriptor struct with an
  /// hsa_queue_t struct embedded at the top. On creation, a queue implementation
  /// allocates its concrete queue descriptor type, casts it to an hsa_queue_t*,
  /// then stores it in the SharedQueue. The remaining bytes of its concrete
  /// queue descriptor struct are implicitly stored at the end of the SharedQueue.
  /// We could make this more explicit with variable-lenght arrays, but C++ does
  /// not support them. Thus the total size of the AqlQueue's queue descriptor
  /// is the following formula. We subtract the sizeof(hsa_queue_t) to avoid double
  /// counting it.
  static constexpr std::size_t shared_queue_size_ =
      sizeof(core::SharedQueue) + sizeof(QueueDescriptorT) - sizeof(hsa_queue_t);

  // Host queue id counter, starting from 0x80000000 to avoid overlaping
  // with aql queue id.
  static __forceinline std::atomic<uint32_t>& queue_count() {
    // This allocation is meant to last until the last thread has exited.
    // It is intentionally not freed.
    static std::atomic<uint32_t>* queue_count_ = new std::atomic<uint32_t>();
    return *queue_count_;
  }

  DISALLOW_COPY_AND_ASSIGN(HostQueue);
};
}  // namespace core
}  // namespace rocr
#endif  // header guard
