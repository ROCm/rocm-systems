/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef HSA_RUNTIME_CORE_INC_AMD_AIE_UMQ_QUEUE_H_
#define HSA_RUNTIME_CORE_INC_AMD_AIE_UMQ_QUEUE_H_

#include <atomic>

#include "core/inc/amd_aie_agent.h"
#include "core/inc/exceptions.h"
#include "core/inc/queue.h"
#include "core/inc/runtime.h"
#include "core/inc/signal.h"

namespace rocr {
namespace AMD {

/// @brief Encapsulates HW AIE4 user-mode queue (UMQ) functionality: a ring buffer BO shared with
/// the device and a doorbell, as opposed to @ref AieKmqQueue's kernel-mode (per-batch ioctl)
/// submission used on aie2/aie2p.
class AieUmqQueue : public core::Queue,
                    private core::LocalSignal,
                    core::DoorbellSignal {
 public:
  static __forceinline bool IsType(core::Signal *signal) {
    return signal->IsType(&rtti_id());
  }

  static __forceinline bool IsType(core::Queue *queue) {
    return queue->IsType(&rtti_id());
  }

  AieUmqQueue(core::SharedQueue* shared_queue, AieAgent* agent, size_t req_size_pkts,
              uint32_t node_id, core::HsaEventCallback callback, void* err_data, uint64_t flags);
  ~AieUmqQueue();

  hsa_status_t Inactivate() override;
  hsa_status_t SetPriority(HSA::hsa_amd_queue_priority_internal_t priority) override;
  void Destroy() override;
  uint64_t LoadReadIndexRelaxed() override;
  uint64_t LoadReadIndexAcquire() override;
  uint64_t LoadWriteIndexRelaxed() override;
  uint64_t LoadWriteIndexAcquire() override;
  void StoreReadIndexRelaxed(uint64_t value) override { assert(false); }
  void StoreReadIndexRelease(uint64_t value) override { assert(false); }
  void StoreWriteIndexRelaxed(uint64_t value) override;
  void StoreWriteIndexRelease(uint64_t value) override;
  uint64_t CasWriteIndexRelaxed(uint64_t expected, uint64_t value) override;
  uint64_t CasWriteIndexAcquire(uint64_t expected, uint64_t value) override;
  uint64_t CasWriteIndexRelease(uint64_t expected, uint64_t value) override;
  uint64_t CasWriteIndexAcqRel(uint64_t expected, uint64_t value) override;
  uint64_t AddWriteIndexRelaxed(uint64_t value) override;
  uint64_t AddWriteIndexAcquire(uint64_t value) override;
  uint64_t AddWriteIndexRelease(uint64_t value) override;
  uint64_t AddWriteIndexAcqRel(uint64_t value) override;
  void StoreRelaxed(hsa_signal_value_t value) override;
  void StoreRelease(hsa_signal_value_t value) override;
  hsa_status_t GetInfo(hsa_queue_info_attribute_t attribute,
                       void *value) override;
  hsa_status_t GetCUMasking(uint32_t num_cu_mask_count, uint32_t* cu_mask) override;
  hsa_status_t SetCUMasking(uint32_t num_cu_mask_count, const uint32_t* cu_mask) override;
  void ExecutePM4(uint32_t* cmd_data, size_t cmd_size_b, hsa_fence_scope_t acquireFence,
                  hsa_fence_scope_t releaseFence, hsa_signal_t* signal) override;

  /// @brief Invoke the per-queue error callback if a non-default one is registered.
  void InvokeErrorCallback(hsa_status_t error) {
    if (errors_callback_ != nullptr && errors_callback_ != core::Queue::DefaultErrorHandler) {
      errors_callback_(error, public_handle(), errors_data_);
    }
  }

 protected:
  bool _IsA(Queue::rtti_t id) const override { return id == &rtti_id(); }

 private:
  static __forceinline int& rtti_id() {
    static int rtti_id_ = 0;
    return rtti_id_;
  }

  /// @brief Hands the ready range of packets to XdnaDriver::SubmitUmqCmdChain and settles the read
  /// index against however many it completed, mirroring AieKmqQueue::SubmitPackets: submission is
  /// synchronous to the caller either way, the driver call owns everything device-specific, and a
  /// failure suspends the queue and reports it through the per-queue error callback.
  void SubmitPackets();

  /// @brief Queue size in bytes.
  uint32_t queue_size_bytes_ = 0;
  /// @brief Base of the queue's ring buffer storage. Handed to the device directly as the
  /// hardware context's umq_bo. Opens with a HostQueueHeader (CERT's own read/write indices),
  /// followed immediately by the packet ring the application writes into and CERT reads out of in
  /// place (see amd_aie_umq_queue.cpp).
  void* ring_buf_ = nullptr;
  /// @brief User Mode Queue (UMQ) metadata associated with this queue.
  void* umq_metadata_ = nullptr;
  /// @brief Mapped hardware doorbell. Ringing it is the entire submission path: there is no
  /// per-batch driver call, the same as a GPU hardware queue.
  volatile uint32_t* doorbell_ = nullptr;
  /// @brief Indicates if queue is active.
  std::atomic<bool> active_ = false;
  /// @brief Set when the queue is suspended after an error; no further packets are submitted.
  /// Only accessed on the packet-submission path, so it does not need to be atomic.
  bool suspended_ = false;
  /// @brief Per-queue error callback registered at queue creation.
  AMD::callback_t<core::HsaEventCallback> errors_callback_;
  /// @brief User data passed to @ref errors_callback_.
  void* errors_data_;
};

} // namespace AMD
} // namespace rocr

#endif  // header guard
