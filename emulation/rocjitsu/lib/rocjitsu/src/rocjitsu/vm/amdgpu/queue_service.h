// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file queue_service.h
/// @brief Frontend-neutral GPU queue lifetime and routing.

#pragma once

#include "rocjitsu/vm/amdgpu/gpu_handles.h"
#include "rocjitsu/vm/amdgpu/interrupt_sink.h"

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

namespace rocjitsu::amdgpu {

class GpuVm;

enum class QueueKind : uint8_t { Compute, Sdma };

/// @brief Backend result for one queue-doorbell notification.
enum class QueueDoorbellDisposition : uint8_t {
  Complete,
  Retry,
  Faulted,
};

struct QueueDoorbellResult {
  bool found = false;
  QueueDoorbellDisposition disposition = QueueDoorbellDisposition::Faulted;

  /// Preserve the historical service contract for callers that only need to
  /// know whether the handle was admitted.
  explicit operator bool() const { return found; }
};

struct QueueCreateInfo;

/// @brief Frontend-neutral execution backend for one registered GPU queue.
///
/// @details QueueService owns admission and lifetime. Implementations translate
/// these typed operations into an existing command processor or another queue
/// execution model. The service never invokes a backend while holding its lock.
class QueueBackend {
public:
  virtual ~QueueBackend() = default;

  /// @brief Attach a queue or throw after leaving enough state for rollback_attach().
  virtual void attach(const QueueCreateInfo &info) = 0;
  /// @brief Undo every side effect of an attach() attempt that threw.
  virtual void rollback_attach(const QueueCreateInfo &info) noexcept = 0;
  virtual void update(uint32_t queue_id, uint32_t process_id, uint64_t ring_base_va,
                      uint32_t ring_size, uint32_t queue_percentage) = 0;
  [[nodiscard]] virtual QueueDoorbellDisposition
  notify_doorbell(uint32_t queue_id, uint32_t process_id, uint64_t value) = 0;
  virtual void detach(uint32_t queue_id, uint32_t process_id) noexcept = 0;
};

/// @brief Frontend-independent description of a queue to attach to the backend.
struct QueueCreateInfo {
  AddressSpaceHandle address_space;
  InterruptSink interrupt_sink{};
  std::shared_ptr<QueueBackend> backend;
  uint32_t process_id = 0;
  uint32_t queue_id = 0;
  uint64_t ring_base_va = 0;
  uint32_t ring_size = 0;
  uint64_t read_ptr_va = 0;
  uint64_t write_ptr_va = 0;
  /// Optional initial hardware read cursor. MES supplies the cursor captured
  /// from the SDMA MQD; legacy/KFD queues leave it unset so the backend loads
  /// the already-published cursor from read_ptr_va on first service.
  std::optional<uint64_t> initial_read_pointer = std::nullopt;
  uint32_t doorbell_offset = 0;
  void *doorbell_base = nullptr;
  uint64_t doorbell_va = 0;
  uint64_t last_doorbell = 0;
  uint64_t queue_desc_va = 0;
  uint64_t exception_status_va = 0;
  uint32_t exception_event_id = 0;
  bool host_accessible = false;
  bool xcd_fanout = false;
  QueueKind kind = QueueKind::Compute;
};

/// @brief Shared queue registry used by legacy and PCI/VFIO front ends.
class QueueService {
public:
  explicit QueueService(GpuVm &gpu_vm) : gpu_vm_(gpu_vm) {}

  [[nodiscard]] QueueHandle create(const QueueCreateInfo &info);
  [[nodiscard]] bool update(QueueHandle handle, uint64_t ring_base_va, uint32_t ring_size,
                            uint32_t queue_percentage);
  [[nodiscard]] bool notify_doorbell(QueueHandle handle, uint64_t value);
  /// @brief Notify a backend while preserving its retry versus terminal result.
  [[nodiscard]] QueueDoorbellResult notify_doorbell_result(QueueHandle handle, uint64_t value);
  [[nodiscard]] bool destroy(QueueHandle handle);

  /// @brief Stop admission, detach every queue, revoke VM identities, and reopen.
  /// @details Queue and VM reset are one admission transaction so no create can
  /// retain an address space between the two teardown phases.
  [[nodiscard]] bool reset();

  [[nodiscard]] bool contains(QueueHandle handle) const;
  [[nodiscard]] std::size_t active_queues() const;
  [[nodiscard]] uint64_t reset_epoch() const;
  [[nodiscard]] bool accepting_creates_for_test() const;

private:
  enum class QueueState : uint8_t { Open, Closing, Dead };

  struct QueueRecord {
    AddressSpaceHandle address_space;
    std::shared_ptr<QueueBackend> backend;
    uint32_t process_id = 0;
    uint32_t queue_id = 0;
    uint32_t active_operations = 0;
    QueueState state = QueueState::Open;
  };

  struct Slot {
    uint64_t generation = 1;
    std::shared_ptr<QueueRecord> queue;
  };

  class OperationLease {
  public:
    OperationLease() = default;
    OperationLease(QueueService &service, std::shared_ptr<QueueRecord> queue);
    OperationLease(const OperationLease &) = delete;
    OperationLease &operator=(const OperationLease &) = delete;
    OperationLease(OperationLease &&other) noexcept;
    OperationLease &operator=(OperationLease &&other) noexcept;
    ~OperationLease();

    explicit operator bool() const { return queue_ != nullptr; }
    QueueRecord *operator->() const { return queue_.get(); }

  private:
    void release();

    QueueService *service_ = nullptr;
    std::shared_ptr<QueueRecord> queue_;
  };

  [[nodiscard]] QueueHandle allocate_locked(std::shared_ptr<QueueRecord> queue);
  [[nodiscard]] std::shared_ptr<QueueRecord> find_locked(QueueHandle handle) const;
  /// @brief Admit one callback only while the record is Open.
  /// @details Closing revokes the public handle first, then waits for every
  /// previously admitted lease to leave its backend callback.
  [[nodiscard]] OperationLease acquire_operation(QueueHandle handle);
  void release_operation(const std::shared_ptr<QueueRecord> &queue);
  [[nodiscard]] std::shared_ptr<QueueRecord> begin_close_locked(QueueHandle handle);
  [[nodiscard]] bool close(std::shared_ptr<QueueRecord> queue);
  void finish_create();

  GpuVm &gpu_vm_;
  mutable std::mutex mutex_;
  std::condition_variable reset_cv_;
  std::vector<Slot> slots_;
  std::vector<uint32_t> free_slots_;
  uint64_t reset_epoch_ = 1;
  uint64_t admission_epoch_ = 1;
  uint32_t active_creates_ = 0;
  uint32_t active_closes_ = 0;
  bool admission_open_ = true;
  bool reset_in_progress_ = false;
};

} // namespace rocjitsu::amdgpu
