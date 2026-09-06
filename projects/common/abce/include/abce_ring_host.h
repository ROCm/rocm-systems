/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Accelerated Blit Copy Engine (ABCE) — host-side SDMA ring.
//
// The host wrapper around the shared reserve/commit protocol in
// abce_ring_core.h (the same algorithm the device DeviceRing in abce_device.h
// runs).  See abce_ring_core.h for the core; this file adds only the host-side
// concerns (zeroing, wptr/doorbell writes, the doorbell-notify callback).

#ifndef ABCE_RING_HOST_H_
#define ABCE_RING_HOST_H_

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "abce_ring_core.h"

namespace abce {

using DoorbellNotifyFn = void (*)(void* context, uint64_t new_write_index);
using QueuePublishFn = void (*)(void* context, volatile uint64_t* write_ptr,
                                volatile uint64_t* doorbell, uint64_t new_write_index);
using QueueLoadIndexFn = uint64_t (*)(void* context);
using QueuePublishIndexFn = void (*)(void* context, uint64_t new_write_index);
using CommitWaitFn = void (*)(void* context, const uint64_t* commit_cursor,
                              uint64_t observed_commit_index);

/// Queue-control operations for APIs that intentionally keep hardware control
/// words opaque. All three callbacks must be supplied together.
struct RingQueueOps {
  QueueLoadIndexFn load_read_index = nullptr;
  QueueLoadIndexFn load_write_index = nullptr;
  QueuePublishIndexFn publish_write_index = nullptr;
  void* context = nullptr;

  bool empty() const {
    return load_read_index == nullptr && load_write_index == nullptr &&
           publish_write_index == nullptr && context == nullptr;
  }

  bool complete() const {
    return load_read_index != nullptr && load_write_index != nullptr &&
           publish_write_index != nullptr;
  }
};

/// @brief Static configuration for one SDMA ring buffer.
///
/// The caller supplies exactly one queue-control backend: either queue_ops, or
/// the directly mapped write_ptr/read_ptr/doorbell fields. RingBuffer never
/// allocates or frees the ring or its queue-control state.
struct RingConfig {
  /// Base address of the ring buffer (queue command memory).
  char* base = nullptr;

  /// Ring size in bytes.  MUST be a power of two.
  size_t size = 0;

  /// Monotonic write index (in bytes) consumed by the engine.  Written by
  /// Release/pad in reservation order.  Device-visible.
  volatile uint64_t* write_ptr = nullptr;

  /// Monotonic read index (in bytes) advanced by the engine as it drains
  /// packets.  Read-only from the host.
  volatile uint64_t* read_ptr = nullptr;

  /// Doorbell word the engine polls; written with the new write index.
  volatile uint64_t* doorbell = nullptr;

  /// Opaque queue-control backend. Use this instead of the three directly
  /// mapped control pointers when queue indices and publication are APIs.
  RingQueueOps queue_ops{};

  /// Optional CPU/GPU-visible producer state. When null, RingBuffer uses
  /// private host-only state. Shared host/device submission requires this.
  SharedRingControl* shared_control = nullptr;

  /// Pad every submission up to at least this many bytes (0 = no minimum).
  /// The pad tail is left as zero DWORDs, which the engine interprets as NOPs.
  size_t min_submission_size = 0;

  /// When true, additionally pad the submission size up to a 64-byte multiple
  /// (required by some virtualized/DXG paths).
  bool align64 = false;

  /// Optional efficient wait used while an out-of-order producer waits for the
  /// commit cursor. ROCr integrations can use this hook for mwaitx. The default
  /// uses RingPause (sched_yield on the host).
  CommitWaitFn commit_wait = nullptr;
  void* commit_wait_context = nullptr;

  /// Optional platform-specific write-pointer/doorbell publication routine.
  /// When null, RingBuffer uses volatile writes with a release fence.
  QueuePublishFn publish = nullptr;
  void* publish_context = nullptr;
};

/// @brief Single-consumer (SDMA engine) ring with multi-producer host writers.
///
/// A thin host wrapper over the shared reserve/commit protocol in
/// abce_ring_core.h (the same code the device DeviceRing runs), adding only the
/// host-side concerns: zeroing acquired regions for the builders' zeroed-buffer
/// contract, the hardware wptr/doorbell writes, and the optional doorbell notify
/// callback.
///
/// The SDMA packet processor consumes the ring strictly in order and does not
/// tolerate the write index moving out of reservation order, so:
///   1. Acquire(n): reserve a contiguous n-byte region. If it would cross the
///      ring end, reserve and publish the wrap tail as NOPs first, then retry
///      the payload at offset zero. Zero the payload and return its token.
///   2. The caller writes packets into the region.  Zeroing on Acquire both
///      satisfies the builders' zeroed-buffer contract and turns any unused pad
///      tail into NOPs.
///   3. Release(token, n): publish the region.  Blocks until every earlier
///      reservation has been published, then advances wptr + rings the doorbell.
///
/// Concurrency model (see abce_ring_core.h):
///   - The reserve cursor is advanced lock-free via a single CAS, so producers
///     write disjoint regions concurrently — there is no reservation lock.
///   - The commit cursor enforces in-order publication: Release for region R
///     spins until it is R's turn, so the doorbell is monotone even when
///     producers finish writing out of order.
class RingBuffer {
 public:
  RingBuffer() = default;
  RingBuffer(const RingBuffer&) = delete;
  RingBuffer& operator=(const RingBuffer&) = delete;
  RingBuffer(RingBuffer&&) = delete;
  RingBuffer& operator=(RingBuffer&&) = delete;

  explicit RingBuffer(const RingConfig& cfg, DoorbellNotifyFn notify = nullptr,
                      void* notify_ctx = nullptr) {
    Init(cfg, notify, notify_ctx);
  }

  void Init(const RingConfig& cfg, DoorbellNotifyFn notify = nullptr, void* notify_ctx = nullptr) {
    assert(cfg.base != nullptr);
    const bool has_raw_backend =
        cfg.write_ptr != nullptr || cfg.read_ptr != nullptr || cfg.doorbell != nullptr;
    const bool raw_backend_complete =
        cfg.write_ptr != nullptr && cfg.read_ptr != nullptr && cfg.doorbell != nullptr;
    const bool has_ops_backend = !cfg.queue_ops.empty();
    assert(((raw_backend_complete && !has_ops_backend) ||
            (cfg.queue_ops.complete() && !has_raw_backend)) &&
           "configure exactly one complete queue-control backend");
    assert(!has_ops_backend || cfg.publish == nullptr);
    assert(cfg.size != 0 && (cfg.size & (cfg.size - 1)) == 0 && "ring size must be a power of two");
    cfg_ = cfg;
    notify_ = notify;
    notify_ctx_ = notify_ctx;
    // Seed the reserve/commit cursors from the queue's current monotonic write
    // pointer.  Attaching to an already-in-use queue and starting from 0 would
    // let the first Acquire hand out bytes the engine still owns, corrupting the
    // ring, so start the cursors at the queue's write pointer.
    const uint64_t initial_write_index = LoadWriteIndex();
    control_ = cfg_.shared_control ? cfg_.shared_control : &local_control_;
    RingAtomicStore(&control_->reserve_cursor, initial_write_index);
    RingAtomicStore(&control_->commit_cursor, initial_write_index);
    RingAtomicStore(&control_->max_write_index, initial_write_index);
  }

  /// @brief Wrap a monotonic index into a ring offset.
  uint32_t Wrap(uint64_t index) const { return static_cast<uint32_t>(RingWrap(index, cfg_.size)); }

  /// @brief Total reservation size for a payload of @p payload_bytes, after
  /// applying the configured minimum submission size and optional 64B rounding.
  /// Pass the result to both Acquire and Release.
  RingStatus PaddedSize(uint32_t payload_bytes, uint32_t& reserve_bytes) const {
    if (payload_bytes == 0) return RingStatus::kInvalidArgument;
    uint64_t padded = payload_bytes;
    if (padded < cfg_.min_submission_size) padded = cfg_.min_submission_size;
    if (cfg_.align64) padded = AlignUp(padded, 64);
    if (padded > UINT32_MAX || padded >= cfg_.size) return RingStatus::kTooLarge;
    reserve_bytes = static_cast<uint32_t>(padded);
    return RingStatus::kSuccess;
  }

  /// @brief Reserve a contiguous, zeroed @p reserve_bytes region of the ring.
  ///
  /// Blocks until the engine has freed enough space.  On success returns a
  /// pointer into the ring (zeroed) and sets @p token, which must be passed
  /// verbatim to Release along with the same @p reserve_bytes.  Fails only if
  /// @p reserve_bytes >= ring size (the request can never fit).
  RingStatus Acquire(uint32_t reserve_bytes, RingReservation& reservation, char*& buffer) {
    buffer = nullptr;
    const RingStatus status = RingReserveWithReadIndex(
        &control_->reserve_cursor, cfg_.size, reserve_bytes, reservation,
        [this]() { return LoadReadIndex(); },
        [this](const RingReservation& padding) {
          std::memset(cfg_.base + Wrap(padding.start), 0, static_cast<size_t>(padding.bytes()));
          return Release(padding);
        });
    if (status != RingStatus::kSuccess) return status;

    // Wrap padding, when needed, was already published as a separate NOP
    // reservation. This reservation contains only the contiguous payload.
    buffer = cfg_.base + Wrap(reservation.start);
    std::memset(buffer, 0, reserve_bytes);
    return RingStatus::kSuccess;
  }

  /// @brief Publish a previously acquired region and ring the doorbell.
  /// Blocks until all earlier reservations have been published so the write
  /// index advances strictly in reservation order.
  RingStatus Release(const RingReservation& reservation) {
    return RingPublish(
        &control_->commit_cursor, reservation,
        [this](uint64_t new_write_index) {
          PublishWriteIndex(new_write_index);
          RingAtomicStoreRelease(&control_->max_write_index, new_write_index);
          if (notify_) notify_(notify_ctx_, new_write_index);
        },
        [this](const uint64_t* commit_cursor, uint64_t observed_commit_index) {
          if (cfg_.commit_wait) {
            cfg_.commit_wait(cfg_.commit_wait_context, commit_cursor, observed_commit_index);
          } else {
            RingPause();
          }
        });
  }

  /// Publish an already-zeroed reservation as NOPs.
  RingStatus Cancel(const RingReservation& reservation) { return Release(reservation); }

  /// Wait until the engine has consumed everything published when this call
  /// snapshots the high-water mark.
  RingStatus Drain() const {
    if (!control_) return RingStatus::kInvalidArgument;
    const uint64_t target = RingAtomicLoadAcquire(&control_->max_write_index);
    while (LoadReadIndex() < target) {
      RingPause();
    }
    return RingStatus::kSuccess;
  }

 private:
  static uint64_t AlignUp(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
  }

  uint64_t LoadReadIndex() const {
    if (cfg_.queue_ops.complete())
      return cfg_.queue_ops.load_read_index(cfg_.queue_ops.context);
    return RingAtomicLoadHw(cfg_.read_ptr);
  }

  uint64_t LoadWriteIndex() const {
    if (cfg_.queue_ops.complete())
      return cfg_.queue_ops.load_write_index(cfg_.queue_ops.context);
    return RingAtomicLoadHw(cfg_.write_ptr);
  }

  void PublishWriteIndex(uint64_t new_write_index) const {
    if (cfg_.queue_ops.complete()) {
      cfg_.queue_ops.publish_write_index(cfg_.queue_ops.context, new_write_index);
    } else if (cfg_.publish) {
      cfg_.publish(cfg_.publish_context, cfg_.write_ptr, cfg_.doorbell, new_write_index);
    } else {
      *cfg_.write_ptr = new_write_index;
      std::atomic_thread_fence(std::memory_order_release);
      *cfg_.doorbell = new_write_index;
    }
  }

  RingConfig cfg_{};
  DoorbellNotifyFn notify_ = nullptr;
  void* notify_ctx_ = nullptr;

  SharedRingControl local_control_{};
  SharedRingControl* control_ = nullptr;
};

}  // namespace abce

#endif  // ABCE_RING_HOST_H_
