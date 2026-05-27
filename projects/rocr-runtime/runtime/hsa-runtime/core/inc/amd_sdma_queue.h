/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef HSA_RUNTIME_CORE_INC_AMD_SDMA_QUEUE_H_
#define HSA_RUNTIME_CORE_INC_AMD_SDMA_QUEUE_H_

#include "core/inc/queue.h"
#include "core/inc/agent.h"
#include "core/inc/signal.h"
#include "hsakmt/hsakmt.h"

#include <cstdint>

namespace rocr {
namespace AMD {

/// @brief User-mode SDMA queue implementation.
///
/// @details Represents an SDMA queue exposed to the user
/// for direct packet submission.
class SdmaQueue : public core::Queue {
 public:
  static __forceinline bool IsType(core::Queue* queue) { return queue->IsType(&rtti_id()); }

  /// @brief Constructor for SdmaQueue.
  ///
  /// @param[in] agent GPU agent that owns this queue.
  /// @param[in] size Size of the queue ring buffer.
  /// @param[in] use_xgmi Whether to use XGMI SDMA engine.
  /// @param[in] sdma_engine_id SDMA engine ID (-1 for auto-select).
  SdmaQueue(core::Agent* agent, size_t size, bool use_xgmi, int sdma_engine_id);

  /// @brief Destructor for SdmaQueue.
  virtual ~SdmaQueue();

  /// @brief Initialize the SDMA queue.
  hsa_status_t Initialize();

  hsa_status_t Inactivate() override { return HSA_STATUS_ERROR; }

  hsa_status_t SetPriority(HSA::hsa_amd_queue_priority_internal_t priority) override {
    return HSA_STATUS_ERROR_INVALID_QUEUE;
  }

  uint64_t LoadReadIndexAcquire() override;

  uint64_t LoadReadIndexRelaxed() override;

  uint64_t LoadWriteIndexAcquire() override;

  uint64_t LoadWriteIndexRelaxed() override;

  void StoreReadIndexRelaxed(uint64_t value) override;

  void StoreReadIndexRelease(uint64_t value) override;

  void StoreWriteIndexRelaxed(uint64_t value) override;

  void StoreWriteIndexRelease(uint64_t value) override;

  uint64_t CasWriteIndexAcqRel(uint64_t expected, uint64_t value) override;

  uint64_t CasWriteIndexAcquire(uint64_t expected, uint64_t value) override;

  uint64_t CasWriteIndexRelaxed(uint64_t expected, uint64_t value) override;

  uint64_t CasWriteIndexRelease(uint64_t expected, uint64_t value) override;

  uint64_t AddWriteIndexAcqRel(uint64_t value) override;

  uint64_t AddWriteIndexAcquire(uint64_t value) override;

  uint64_t AddWriteIndexRelaxed(uint64_t value) override;

  uint64_t AddWriteIndexRelease(uint64_t value) override;

  hsa_status_t SetCUMasking(uint32_t num_cu_mask_count, const uint32_t* cu_mask) override {
    assert(false && "SdmaQueue::SetCUMasking is unimplemented");
    return HSA_STATUS_ERROR_INVALID_QUEUE;
  }

  hsa_status_t GetCUMasking(uint32_t num_cu_mask_count, uint32_t* cu_mask) override {
    assert(false && "SdmaQueue::GetCUMasking is unimplemented");
    return HSA_STATUS_ERROR_INVALID_QUEUE;
  }

  void ExecutePM4(uint32_t* cmd_data, size_t cmd_size_b,
                  hsa_fence_scope_t acquireFence = HSA_FENCE_SCOPE_NONE,
                  hsa_fence_scope_t releaseFence = HSA_FENCE_SCOPE_NONE,
                  hsa_signal_t* signal = NULL) override {
    assert(false && "SdmaQueue::ExecutePM4 is unimplemented");
  }

  hsa_status_t GetInfo(hsa_queue_info_attribute_t attribute, void* value) override {
    assert(false && "SdmaQueue::GetInfo(hsa_queue_info_attribute_t, void*) is unimplemented");
    return HSA_STATUS_ERROR_INVALID_QUEUE;
  }

  /// @brief Query information for this SDMA queue.
  ///
  /// @param[in] attribute Requested SDMA queue attribute.
  /// @param[out] value Destination for the requested attribute.
  ///
  /// @return HSA_STATUS_SUCCESS on successful query.
  /// @return HSA_STATUS_ERROR_INVALID_ARGUMENT for invalid attribute.
  hsa_status_t GetInfo(hsa_amd_sdma_queue_info_attribute_t attribute, void* value) const;

  /// @brief Update write pointer and ring SDMA queue doorbell.
  ///
  /// @param[in] write_index New write index to publish.
  ///
  /// @return HSA_STATUS_SUCCESS on success, error code otherwise.
  hsa_status_t RingDoorbell(uint64_t write_index);

  /// @brief Get queue resource information.
  ///
  /// @return Reference to the HsaQueueResource structure.
  const HsaQueueResource& queue_resource() const { return queue_resource_; }

  /// @brief Get queue ring base address.
  ///
  /// @return Pointer to queue ring buffer base address.
  char* queue_start_addr() const { return queue_start_addr_; }

  /// @brief Get queue write pointer.
  ///
  /// @return Pointer to write index register.
  volatile uint64_t* queue_wptr() const { return queue_wptr_; }

  /// @brief Get queue read pointer.
  ///
  /// @return Pointer to read index register.
  volatile uint64_t* queue_rptr() const { return queue_rptr_; }

  /// @brief Get queue doorbell.
  ///
  /// @return Pointer to doorbell register.
  volatile uint64_t* queue_doorbell() const { return queue_doorbell_; }

  /// @brief Get SDMA engine ID.
  int32_t sdma_engine_id() const { return sdma_engine_id_; }

  /// @brief True if xGMI SDMA queue, false if PCIe.
  bool is_xgmi() const { return is_xgmi_; }

 private:
  static __forceinline int& rtti_id() {
    static int rtti_id_ = 0;
    return rtti_id_;
  }

 protected:
  bool _IsA(core::Queue::rtti_t id) const override { return id == &rtti_id(); }

 private:
  /// @brief Allocate queue ring buffer.
  hsa_status_t AllocateQueueBuffer();

  /// @brief Free queue ring buffer.
  void FreeQueueBuffer();

  /// GPU agent that owns this queue.
  core::Agent* agent_;

  /// Queue ring buffer base address.
  char* queue_start_addr_;

  /// Queue size in bytes.
  size_t queue_size_;

  /// Cached MMIO pointer to write pointer.
  volatile uint64_t* queue_wptr_;

  /// Cached MMIO pointer to read pointer.
  volatile uint64_t* queue_rptr_;

  /// Cached MMIO pointer to doorbell.
  volatile uint64_t* queue_doorbell_;

  /// Queue resource information.
  HsaQueueResource queue_resource_;

  /// SDMA engine ID.
  int32_t sdma_engine_id_;

  /// Whether this is an XGMI SDMA queue.
  bool is_xgmi_;
};

}  // namespace AMD
}  // namespace rocr

#endif  // HSA_RUNTIME_CORE_INC_AMD_SDMA_QUEUE_H_