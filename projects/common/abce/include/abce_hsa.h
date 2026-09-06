/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Optional public-HSA adapter for the Accelerated Blit Copy Engine (ABCE).
// Include this header only in clients that submit to ROCr-created SDMA queues.

#ifndef ABCE_HSA_H_
#define ABCE_HSA_H_

#if __has_include(<hsa/hsa.h>)
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#else
#include <hsa.h>
#include <hsa_ext_amd.h>
#endif

#if __has_include(<hsa/amd_hsa_signal.h>)
#include <hsa/amd_hsa_signal.h>
#else
#include <amd_hsa_signal.h>
#endif

#include <cstddef>
#include <cstdint>

#include "abce_host.h"

namespace abce {

/// Completion target for an HSA signal. @p completion_value is the value the
/// signal holds once hardware finishes; see SignalRef for how each completion
/// packet reaches it. Lives here rather than on SignalRef so the frame composer
/// stays free of the amd_signal_t layout.
///
/// The fan-out coordination scratch is placed in the signal's own trailing
/// reserved words: they are a naturally aligned 64-bit slot inside the same
/// 64-byte signal object, so coordination shares a cache line with the value it
/// guards and needs no separate allocation.
inline SignalRef HsaSignalRef(amd_signal_t* signal, uint64_t completion_value = 0) {
  static_assert(sizeof(amd_signal_t::reserved3) == sizeof(uint64_t),
                "amd_signal_t::reserved3 is no longer a 64-bit slot");
  static_assert(offsetof(amd_signal_t, reserved3) % alignof(uint64_t) == 0,
                "amd_signal_t::reserved3 is not 64-bit aligned");
  return SignalRef(const_cast<int64_t*>(&signal->value), completion_value,
                   reinterpret_cast<void*>(signal->event_mailbox_ptr), signal->event_id,
                   &signal->reserved3[0]);
}

/// Suggested placement for CopyMetadata::execution_descriptor: the signal's
/// spare 32-bit reserved word, which shares the signal's cache line and so costs
/// no extra allocation. Opt in by assigning this into the metadata; ABCE never
/// writes it on its own, because which bytes of a signal may be repurposed is
/// the caller's ABI decision, not ABCE's.
///
/// amd_signal_t is a frozen, potentially cross-process format, so a client that
/// cannot spend reserved1 should point the descriptor at its own wrapper struct
/// instead -- exactly what ROCr does for SDMA timestamps, which live in
/// SharedSignal rather than in amd_signal_t. Nothing about the descriptor
/// depends on being inside the signal; it only has to be somewhere its reader
/// agrees to look.
inline uint32_t* HsaExecutionDescriptorSlot(amd_signal_t* signal) {
  return &signal->reserved1;
}

/// Dependency on an HSA signal. @p observed is the value read at record time,
/// used to elide an already-satisfied wait.
inline DepSignal HsaDepSignal(amd_signal_t* signal, uint64_t observed,
                              uint64_t reference = 0) {
  return DepSignal(const_cast<int64_t*>(&signal->value), observed, reference);
}

using HsaQueueGetInfoFn =
    hsa_status_t (*)(hsa_queue_t* queue, hsa_queue_info_attribute_t attribute, void* value);
using HsaQueueLoadReadIndexFn = uint64_t (*)(const hsa_queue_t* queue);
using HsaQueueLoadWriteIndexFn = uint64_t (*)(const hsa_queue_t* queue);
using HsaSignalStoreReleaseFn = void (*)(hsa_signal_t signal, hsa_signal_value_t value);

/// HSA entry points used by HsaQueueRing. Keeping them injectable lets clients
/// with a dynamically loaded ROCr runtime use the adapter without adding a
/// static runtime dependency.
struct HsaQueueApi {
  HsaQueueGetInfoFn queue_get_info = nullptr;
  HsaQueueLoadReadIndexFn load_read_index = nullptr;
  HsaQueueLoadWriteIndexFn load_write_index = nullptr;
  HsaSignalStoreReleaseFn signal_store_release = nullptr;

  bool complete() const {
    return queue_get_info != nullptr && load_read_index != nullptr &&
           load_write_index != nullptr && signal_store_release != nullptr;
  }
};

namespace detail {

inline hsa_status_t DirectQueueGetInfo(hsa_queue_t* queue,
                                       hsa_queue_info_attribute_t attribute, void* value) {
  return hsa_amd_queue_get_info(queue, attribute, value);
}

inline uint64_t DirectLoadReadIndex(const hsa_queue_t* queue) {
  return hsa_queue_load_read_index_relaxed(queue);
}

inline uint64_t DirectLoadWriteIndex(const hsa_queue_t* queue) {
  return hsa_queue_load_write_index_relaxed(queue);
}

inline void DirectSignalStoreRelease(hsa_signal_t signal, hsa_signal_value_t value) {
  hsa_signal_store_screlease(signal, value);
}

}  // namespace detail

/// Entry-point table for clients linked directly to the HSA runtime.
inline HsaQueueApi DirectHsaQueueApi() {
  return HsaQueueApi{&detail::DirectQueueGetInfo, &detail::DirectLoadReadIndex,
                     &detail::DirectLoadWriteIndex, &detail::DirectSignalStoreRelease};
}

struct HsaRingOptions {
  size_t min_submission_size = 0;
  bool align64 = false;
  SharedRingControl* shared_control = nullptr;
  CommitWaitFn commit_wait = nullptr;
  void* commit_wait_context = nullptr;
};

/// Non-owning adapter from a public SDMA hsa_queue_t to an ABCE RingBuffer.
///
/// The queue must remain alive until all submissions are complete and the
/// adapter is no longer registered with an orchestrator. Host/device sharing
/// is supported only when both views use the same SharedRingControl. The
/// adapter itself must remain at a stable address because RingBuffer callbacks
/// use it as their context.
class HsaQueueRing {
 public:
  HsaQueueRing() = default;
  HsaQueueRing(const HsaQueueRing&) = delete;
  HsaQueueRing& operator=(const HsaQueueRing&) = delete;
  HsaQueueRing(HsaQueueRing&&) = delete;
  HsaQueueRing& operator=(HsaQueueRing&&) = delete;

  hsa_status_t Init(hsa_queue_t* queue, const HsaQueueApi& api,
                    const HsaRingOptions& options = {}) {
    if (queue == nullptr || !api.complete() || queue_ != nullptr)
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;

    hsa_amd_queue_engine_t engine_type = HSA_AMD_QUEUE_ENGINE_COMPUTE;
    hsa_status_t status =
        api.queue_get_info(queue, HSA_AMD_QUEUE_INFO_ENGINE_TYPE, &engine_type);
    if (status != HSA_STATUS_SUCCESS) return status;
    if (engine_type != HSA_AMD_QUEUE_ENGINE_SDMA) return HSA_STATUS_ERROR_INVALID_QUEUE;

    uint32_t resolved_engine_id = HSA_AMD_SDMA_ENGINE_ID_ANY;
    status = api.queue_get_info(queue, HSA_AMD_QUEUE_INFO_SDMA_ENGINE_ID,
                                &resolved_engine_id);
    if (status != HSA_STATUS_SUCCESS) return status;
    if (resolved_engine_id == HSA_AMD_SDMA_ENGINE_ID_ANY)
      return HSA_STATUS_ERROR_INVALID_QUEUE;

    const size_t ring_size = static_cast<size_t>(queue->size);
    if (queue->base_address == nullptr || ring_size == 0 ||
        (ring_size & (ring_size - 1)) != 0)
      return HSA_STATUS_ERROR_INVALID_QUEUE;

    queue_ = queue;
    api_ = api;
    engine_id_ = resolved_engine_id;

    RingConfig config{};
    config.base = static_cast<char*>(queue->base_address);
    config.size = ring_size;
    config.queue_ops = RingQueueOps{&LoadReadIndex, &LoadWriteIndex, &PublishWriteIndex, this};
    config.shared_control = options.shared_control;
    config.min_submission_size = options.min_submission_size;
    config.align64 = options.align64;
    config.commit_wait = options.commit_wait;
    config.commit_wait_context = options.commit_wait_context;
    ring_.Init(config);
    return HSA_STATUS_SUCCESS;
  }

  hsa_status_t Init(hsa_queue_t* queue, const HsaRingOptions& options = {}) {
    return Init(queue, DirectHsaQueueApi(), options);
  }

  bool initialized() const { return queue_ != nullptr; }
  hsa_queue_t* queue() const { return queue_; }
  uint32_t engine_id() const { return engine_id_; }
  RingBuffer* ring() { return initialized() ? &ring_ : nullptr; }
  const RingBuffer* ring() const { return initialized() ? &ring_ : nullptr; }
  RingStatus Drain() const { return initialized() ? ring_.Drain() : RingStatus::kInvalidArgument; }

 private:
  static uint64_t LoadReadIndex(void* context) {
    HsaQueueRing* adapter = static_cast<HsaQueueRing*>(context);
    return adapter->api_.load_read_index(adapter->queue_);
  }

  static uint64_t LoadWriteIndex(void* context) {
    HsaQueueRing* adapter = static_cast<HsaQueueRing*>(context);
    return adapter->api_.load_write_index(adapter->queue_);
  }

  static void PublishWriteIndex(void* context, uint64_t new_write_index) {
    HsaQueueRing* adapter = static_cast<HsaQueueRing*>(context);
    // SdmaQueue::StoreRelease orders the canonical write pointer and packets
    // before ringing the KFD doorbell.
    adapter->api_.signal_store_release(adapter->queue_->doorbell_signal,
                                       static_cast<hsa_signal_value_t>(new_write_index));
  }

  hsa_queue_t* queue_ = nullptr;
  HsaQueueApi api_{};
  uint32_t engine_id_ = HSA_AMD_SDMA_ENGINE_ID_ANY;
  RingBuffer ring_{};
};

inline bool RegisterHsaEngine(CopyOrchestrator& orchestrator, uint32_t orchestrator_index,
                              HsaQueueRing& queue_ring) {
  if (!queue_ring.initialized()) return false;
  EngineAffinity affinity{};
  affinity.hw_engine_id = queue_ring.engine_id();
  return orchestrator.RegisterEngine(orchestrator_index, queue_ring.ring(), affinity);
}

}  // namespace abce

#endif  // ABCE_HSA_H_
