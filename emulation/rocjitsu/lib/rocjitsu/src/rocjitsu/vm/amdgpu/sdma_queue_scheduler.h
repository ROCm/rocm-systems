// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file sdma_queue_scheduler.h
/// @brief SoC-owned asynchronous SDMA queue scheduler.

#pragma once

#include "rocjitsu/vm/amdgpu/gpu_handles.h"
#include "rocjitsu/vm/amdgpu/gpu_queue_registry.h"
#include "rocjitsu/vm/amdgpu/sdma_packet_processor.h"
#include "rocjitsu/vm/amdgpu/sdma_ring_consumer.h"

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

namespace rocjitsu::amdgpu {

class GpuVm;
class DeviceCacheCoherence;

/// @brief Minimal identity exposed to SDMA environment callback factories.
struct SdmaQueueContext {
  uint32_t process_id = 0;
  uint32_t queue_id = 0;
  uint32_t engine_id = 0;

  friend bool operator==(const SdmaQueueContext &, const SdmaQueueContext &) = default;
};

/// @brief Frontend-independent configuration consumed by the SDMA scheduler.
/// @details Generic queue-kind and packet-format validation belongs to the
/// binding factory. The scheduler accepts only already translated SDMA state.
struct SdmaQueueConfig {
  SdmaRingConfig ring;
  SdmaQueueContext context;
  uint32_t doorbell_offset = 0;
  void *doorbell_base = nullptr;
  uint64_t last_doorbell = 0;
  bool host_accessible = false;
};

/// @brief Queue-consumer state published after an asynchronous service turn.
struct SdmaQueueProgress {
  uint64_t consumer_cursor = 0;
  bool terminal = false;
};

/// @brief Frontend observer for queue progress, separate from packet semantics.
/// @details Invoked without the scheduler mutex while the queue remains in an
/// active service turn. Detach therefore waits for an admitted observer to
/// return, and observers must not synchronously detach their own queue.
using SdmaQueueProgressObserver = std::function<void(const SdmaQueueProgress &)>;

/// @brief Independently schedules and executes all SDMA queues for one SoC.
/// @details Frontends submit typed queue state and producer notifications. The
/// dedicated worker owns every SdmaRingConsumer and is the only thread that
/// executes SDMA packets. Queue removal is a synchronous quiescence barrier.
class SdmaQueueScheduler {
public:
  /// @brief Generation-safe identity for a scheduler-owned queue.
  class Handle {
  public:
    /// @brief Construct an invalid queue handle.
    Handle() = default;

    /// @brief Return whether this handle names a potentially valid generation.
    explicit operator bool() const { return generation_ != 0; }
    friend bool operator==(const Handle &, const Handle &) = default;

  private:
    friend class SdmaQueueScheduler;

    Handle(uint32_t slot, uint64_t generation) : slot_(slot), generation_(generation) {}

    uint32_t slot_ = 0;
    uint64_t generation_ = 0;
  };

  explicit SdmaQueueScheduler(GpuVm &gpu_vm, SdmaPacketDialect dialect = SdmaPacketDialect::Legacy);
  SdmaQueueScheduler(GpuVm &gpu_vm, std::shared_ptr<DeviceCacheCoherence> coherence,
                     SdmaPacketDialect dialect = SdmaPacketDialect::Legacy);
  ~SdmaQueueScheduler();

  SdmaQueueScheduler(const SdmaQueueScheduler &) = delete;
  SdmaQueueScheduler &operator=(const SdmaQueueScheduler &) = delete;
  SdmaQueueScheduler(SdmaQueueScheduler &&) = delete;
  SdmaQueueScheduler &operator=(SdmaQueueScheduler &&) = delete;

  [[nodiscard]] Handle attach(SdmaQueueConfig config, SdmaPacketCallbacks callbacks,
                              SdmaQueueProgressObserver progress_observer = {});
  [[nodiscard]] QueueReconfigureStatus update(Handle handle, uint64_t ring_base, uint32_t ring_size,
                                              uint32_t scheduling_percentage);
  /// @brief Retain a monotonic producer target for asynchronous service.
  /// @returns Accepted once retained, or Faulted for a stale/terminal queue;
  /// packet blocking is retried solely by the scheduler worker.
  [[nodiscard]] QueueSubmissionStatus notify(Handle handle, uint64_t producer_cursor);
  /// @brief Gracefully detach only after retry-critical packet state is drained.
  [[nodiscard]] QueuePrepareCloseStatus prepare_detach(Handle handle) noexcept;
  /// @brief Force-cancel and detach during reset or frontend teardown.
  [[nodiscard]] bool detach(Handle handle) noexcept;

  /// @brief Update host doorbell mappings and exclude concurrent scans.
  void set_process_doorbell_base(uint32_t process_id, void *base);
  /// @brief Change packet decoding only when no queues are attached.
  [[nodiscard]] bool set_packet_dialect(SdmaPacketDialect dialect);
  /// @brief Stop and join the worker. Idempotent and safe with no queues.
  void shutdown() noexcept;

  /// @brief Return the number of queues currently owned by this scheduler.
  [[nodiscard]] std::size_t active_queues() const;
  /// @brief Return whether @p handle still names an attached queue.
  [[nodiscard]] bool contains(Handle handle) const;
  /// @brief Return the packet dialect used for newly attached queues.
  [[nodiscard]] SdmaPacketDialect packet_dialect() const;
  /// @brief Return this scheduler's device-scoped cache-coherence domain.
  const std::shared_ptr<DeviceCacheCoherence> &cache_coherence() const { return coherence_; }

private:
  class QueueRecord;

  class Slot {
  public:
    uint64_t generation = 1;
    std::shared_ptr<QueueRecord> queue;
  };

  [[nodiscard]] std::shared_ptr<QueueRecord> find_locked(Handle handle) const;
  [[nodiscard]] Handle allocate_locked(std::shared_ptr<QueueRecord> queue);
  [[nodiscard]] QueuePrepareCloseStatus detach(Handle handle, bool force) noexcept;
  void ensure_worker_locked();
  void worker_loop(std::stop_token stop_token);
  void scan_host_doorbells_locked();
  [[nodiscard]] std::shared_ptr<QueueRecord> select_queue_locked();
  [[nodiscard]] std::optional<std::chrono::steady_clock::time_point>
  next_wake_deadline_locked() const;

  static constexpr std::size_t kPacketsPerTurn = 64;

  GpuVm &gpu_vm_;
  std::shared_ptr<DeviceCacheCoherence> coherence_;
  mutable std::mutex mutex_;
  std::condition_variable condition_;
  std::vector<Slot> slots_;
  std::vector<uint32_t> free_slots_;
  std::jthread worker_;
  SdmaPacketDialect dialect_;
  std::size_t next_slot_ = 0;
  bool shutting_down_ = false;
};

} // namespace rocjitsu::amdgpu
