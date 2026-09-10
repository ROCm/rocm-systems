// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file pm4_queue_controller.h
/// @brief Command-processor-owned PM4 queue state and service.

#pragma once

#include "rocjitsu/vm/amdgpu/gpu_queue_registry.h"
#include "rocjitsu/vm/amdgpu/pm4_ring_consumer.h"

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>

namespace rocjitsu::amdgpu {

class GpuVm;

/// @brief PM4 queue configuration admitted by one CommandProcessor.
struct Pm4QueueConfig {
  AddressSpaceHandle address_space;
  uint64_t ring_base = 0;
  uint32_t ring_size_bytes = 0;
  uint64_t consumer_pointer_address = 0;
  std::optional<uint64_t> initial_consumer_cursor;
  Pm4PacketCallbacks packet_callbacks;
};

/// @brief Per-CP owner of PM4 queue state below GpuQueueRegistry.
/// @details Submission only records producer progress. The owning CP invokes
/// service() on its event context, keeping PM4 ring traversal and packet effects
/// out of MES and PCI/VFIO callbacks. A service lease keeps the queue alive while
/// VM accesses and packet callbacks run without the controller mutex; detach and
/// reconfiguration synchronously wait for that lease to finish.
class Pm4QueueController {
public:
  using RegistrationId = uint64_t;

  explicit Pm4QueueController(GpuVm &gpu_vm) : gpu_vm_(gpu_vm) {}

  [[nodiscard]] RegistrationId attach(Pm4QueueConfig config);
  /// @brief Gracefully remove a queue without discarding committed progress.
  [[nodiscard]] QueuePrepareCloseStatus prepare_detach(RegistrationId registration_id) noexcept;
  /// @brief Unconditionally cancel and remove a queue during rollback or reset.
  [[nodiscard]] bool detach(RegistrationId registration_id) noexcept;
  [[nodiscard]] QueueReconfigureStatus update(RegistrationId registration_id,
                                              const QueueReconfigureRequest &request);
  [[nodiscard]] QueueSubmissionStatus notify(RegistrationId registration_id,
                                             uint64_t producer_cursor);

  /// @brief Service every runnable PM4 queue once.
  /// @returns true when blocked or newly arrived work needs another service pass.
  [[nodiscard]] bool service();

  /// @brief Force-cancel every queue and release all retained VM bindings.
  void close_all() noexcept;

  [[nodiscard]] std::size_t active_queues() const;

private:
  static constexpr std::size_t kPacketsPerTurn = 256;

  class QueueRecord {
  public:
    QueueRecord(GpuVm &gpu_vm, Pm4QueueConfig config, GpuVmBindingLease address_space_lease);

    GpuVmBindingLease address_space_lease;
    Pm4PacketProcessor packet_processor;
    Pm4RingConsumer ring_consumer;
    uint64_t producer_cursor = 0;
    uint32_t management_operations = 0;
    bool enabled = true;
    bool ready = false;
    bool service_active = false;
    bool closing = false;
    bool terminal = false;
  };

  [[nodiscard]] QueueSubmissionStatus service_queue(QueueRecord &queue, uint64_t producer_cursor);
  [[nodiscard]] QueuePrepareCloseStatus detach(RegistrationId registration_id, bool force) noexcept;

  mutable std::mutex mutex_;
  std::condition_variable condition_;
  GpuVm &gpu_vm_;
  std::unordered_map<RegistrationId, std::shared_ptr<QueueRecord>> queues_;
  RegistrationId next_registration_id_ = 1;
};

} // namespace rocjitsu::amdgpu
