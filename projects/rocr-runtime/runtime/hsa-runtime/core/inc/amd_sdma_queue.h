/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef HSA_RUNTIME_CORE_INC_AMD_SDMA_QUEUE_H_
#define HSA_RUNTIME_CORE_INC_AMD_SDMA_QUEUE_H_

#include "core/inc/agent.h"
#include "core/inc/queue.h"
#include "core/inc/signal.h"
#include "hsakmt/hsakmt.h"

#include <cassert>
#include <cstddef>
#include <cstdint>

namespace rocr {
namespace AMD {

class GpuAgent;

/// @brief User-mode SDMA queue implementation.
///
/// @details Represents an SDMA queue exposed through the unified public
/// hsa_queue_t handle for direct SDMA packet submission. This queue is not a
/// true multi-producer queue: callers must serialize packet production and
/// perform the equivalent of BlitSdma's AcquireWriteAddress/ReleaseWriteAddress
/// protocol, including ring-space checks, wrap/no-op handling, packet writes,
/// and ordered doorbell publication.
///
/// Write-pointer model (important): an SDMA queue tracks the write position in
/// two distinct places, unlike an AQL queue.
///   1. @c amd_queue_.write_dispatch_id is the *software* write index. It is a
///      monotonically increasing byte count (SDMA pointers are byte offsets)
///      and backs the public hsa_queue_load/store/add/cas_write_index helpers.
///      The SDMA engine never reads it.
///   2. @c queue_wptr_ (the KFD-provided @c HsaQueueResource::Queue_write_ptr)
///      is the *hardware* write pointer the SDMA engine actually consumes.
/// For an AQL queue these two are the same memory, because KFD is told to use
/// @c write_dispatch_id directly as the hardware write pointer. For an SDMA
/// queue KFD allocates its own write-pointer location, so the software index
/// must be explicitly copied into @c queue_wptr_ at publish time.
///
/// Submission protocol (single producer, performed by the caller per
/// submission, matching BlitSdma's AcquireWriteAddress/ReleaseWriteAddress):
///   a. Read the current write index (LoadWriteIndex* -> write_dispatch_id).
///   b. Ensure ring space against the read index (queue_rptr_) and pad to the
///      ring end with SDMA no-ops if the request would wrap.
///   c. Write complete packets into base_address + (index % size).
///   d. Advance the software write index (StoreWriteIndex*/AddWriteIndex*).
///   e. Publish via the doorbell signal store, which routes to RingDoorbell():
///      that is the *only* step that updates the hardware write pointer
///      (queue_wptr_) and the doorbell, in that order, after a release fence.
/// Steps (a)-(d) only touch the software index; the engine observes nothing
/// until (e). Callers must serialize a-e (e.g. a lock) because the index
/// accessors here do not implement a reservation/commit handshake.
class SdmaQueue : public core::Queue, private core::LocalSignal, public core::DoorbellSignal {
 public:
  static __forceinline bool IsType(core::Queue* queue) { return queue->IsType(&rtti_id()); }

  /// @param[in] agent GPU agent that owns this queue.
  /// @param[in] size_bytes Size of the queue ring buffer in bytes.
  /// @param[in] flags Queue memory placement flags.
  /// @param[in] sdma_engine_id SDMA engine index, or -1 for runtime selection.
  SdmaQueue(core::Agent* agent, size_t size_bytes, uint64_t flags, int32_t sdma_engine_id);
  ~SdmaQueue() override;

  /// @brief Create the backing KFD SDMA queue and populate the public handle.
  hsa_status_t Initialize();

  hsa_status_t Inactivate() override;

  hsa_status_t SetPriority(HSA::hsa_amd_queue_priority_internal_t priority) override {
    return HSA_STATUS_SUCCESS;
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

  // Implemented to satisfy the core::Queue interface and public hsa_queue_t
  // helpers. These update only the descriptor's software write index; they do
  // not reserve SDMA ring space or make SDMA submission multi-producer safe.
  uint64_t AddWriteIndexAcqRel(uint64_t value) override;
  uint64_t AddWriteIndexAcquire(uint64_t value) override;
  uint64_t AddWriteIndexRelaxed(uint64_t value) override;
  uint64_t AddWriteIndexRelease(uint64_t value) override;

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
    assert(false && "SdmaQueue::ExecutePM4 is unimplemented");
  }

  hsa_status_t GetInfo(hsa_queue_info_attribute_t attribute, void* value) override {
    return HSA_STATUS_ERROR_INVALID_QUEUE;
  }

  /// @brief Update write pointer and ring the SDMA queue doorbell.
  hsa_status_t RingDoorbell(uint64_t write_index);

  /// @brief Doorbell signal store entry points used by hsa_signal_store_*.
  void StoreRelaxed(hsa_signal_value_t value) override;
  void StoreRelease(hsa_signal_value_t value) override;

 private:
  static __forceinline int& rtti_id() {
    static int rtti_id_ = 0;
    return rtti_id_;
  }

 protected:
  bool _IsA(core::Queue::rtti_t id) const override { return id == &rtti_id(); }

 private:
  hsa_status_t AllocateQueueBuffer();
  void FreeQueueBuffer();

  AMD::GpuAgent* gpu_agent() const;

  core::Agent* agent_;
  char* queue_start_addr_;
  size_t queue_size_;
  volatile uint64_t* queue_wptr_;
  volatile uint64_t* queue_rptr_;
  volatile uint64_t* queue_doorbell_;
  HsaQueueResource queue_resource_;
  int32_t sdma_engine_id_;
  bool active_;
};

}  // namespace AMD
}  // namespace rocr

#endif  // HSA_RUNTIME_CORE_INC_AMD_SDMA_QUEUE_H_
