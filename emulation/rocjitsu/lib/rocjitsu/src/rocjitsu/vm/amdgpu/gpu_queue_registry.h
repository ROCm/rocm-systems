// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file gpu_queue_registry.h
/// @brief Frontend-neutral GPU queue registration and lifetime ownership.

#pragma once

#include "rocjitsu/vm/amdgpu/gpu_handles.h"
#include "rocjitsu/vm/amdgpu/interrupt_sink.h"

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

namespace rocjitsu::amdgpu {

class GpuVm;
class GpuQueueRegistryTestAccess;

enum class QueueType : uint8_t { Compute, Sdma };

/// @brief Packet format consumed by one registered GPU queue.
enum class QueuePacketFormat : uint8_t { Aql, Pm4, Sdma };

/// @brief Result of asking a binding factory to create one queue binding.
enum class QueueBindingCreateStatus : uint8_t { Bound, Rejected };

/// @brief Result of submitting producer progress to a bound queue.
enum class QueueSubmissionStatus : uint8_t {
  Accepted,
  Retry,
  Faulted,
};

/// @brief Result of reconfiguring one bound queue.
enum class QueueReconfigureStatus : uint8_t {
  Applied,
  Disabled,
  Busy,
  Invalid,
  Stale,
};

/// @brief Policy used when removing a registered queue.
enum class QueueCloseMode : uint8_t {
  /// Preserve committed state that still needs externally visible publication.
  Graceful,
  /// Cancel all retained execution state during reset or frontend teardown.
  ForceCancel,
};

/// @brief Result of asking a binding whether graceful destruction is safe.
enum class QueuePrepareCloseStatus : uint8_t {
  Ready,
  Busy,
  Faulted,
};

/// @brief Result of removing one queue from the registry.
enum class QueueCloseStatus : uint8_t {
  Closed,
  Busy,
  Faulted,
  Stale,
};

class QueueCloseResult {
public:
  bool found = false;
  QueueCloseStatus status = QueueCloseStatus::Stale;

  /// @brief Whether the queue was removed and its binding was destroyed.
  explicit operator bool() const { return found && status == QueueCloseStatus::Closed; }
};

/// @brief Registry result for one queue reconfiguration.
class QueueReconfigureResult {
public:
  bool found = false;
  QueueReconfigureStatus status = QueueReconfigureStatus::Stale;

  /// @brief Whether the handle existed and the requested state was accepted.
  explicit operator bool() const {
    return found && (status == QueueReconfigureStatus::Applied ||
                     status == QueueReconfigureStatus::Disabled);
  }
};

/// @brief Registry result for one producer submission.
class QueueSubmissionResult {
public:
  bool found = false;
  QueueSubmissionStatus status = QueueSubmissionStatus::Faulted;

  /// @brief Return whether the handle identified a live queue binding.
  explicit operator bool() const { return found; }
};

struct QueueRegistrationRequest;

/// @brief Stable identity and fault-routing state shared by queue implementations.
struct QueueIdentity {
  AddressSpaceHandle address_space;
  InterruptSink interrupt_sink{};
  uint32_t process_id = 0;
  uint32_t queue_id = 0;
};

/// @brief Guest-visible ring storage and producer/consumer pointer locations.
struct QueueRingLayout {
  uint64_t base_address = 0;
  uint32_t size_bytes = 0;
  uint64_t consumer_pointer_address = 0;
  uint64_t producer_pointer_address = 0;
};

/// @brief Doorbell mapping used by host-polled and explicitly notified queues.
struct QueueDoorbellBinding {
  uint32_t offset = 0;
  void *host_base = nullptr;
  uint64_t address = 0;
  uint64_t last_value = 0;
  bool host_accessible = false;
};

/// @brief Queue state accepted by a reconfiguration operation.
struct QueueReconfigureRequest {
  uint64_t ring_base_address = 0;
  uint32_t ring_size_bytes = 0;
  uint32_t scheduling_percentage = 0;
};

/// @brief Unique execution binding owned by one registered GPU queue.
///
/// @details The registry owns admission and lifetime. A binding contains the
/// processor-specific registration token for exactly one queue. The registry
/// never invokes or destroys a binding while holding its mutex. Destruction is
/// the synchronous quiescence boundary and must release the processor-specific
/// registration exactly once.
class QueueBinding {
public:
  virtual ~QueueBinding() = default;

  /// @brief Atomically prepare this binding for ordinary queue removal.
  /// @details Return false when destruction would discard retry-critical state.
  /// The registry keeps the handle live so the caller may retry removal.
  [[nodiscard]] virtual QueuePrepareCloseStatus prepare_close() noexcept {
    return QueuePrepareCloseStatus::Ready;
  }

  /// @brief Apply a queue configuration update and report its status.
  [[nodiscard]] virtual QueueReconfigureStatus
  reconfigure(const QueueReconfigureRequest &request) = 0;
  /// @brief Submit a protocol-native producer notification to the queue owner.
  /// @details The selected binding defines whether this value is a producer
  /// cursor, a doorbell hint, or another protocol-specific progress marker.
  [[nodiscard]] virtual QueueSubmissionStatus submit_producer(uint64_t producer_value) = 0;
};

/// @brief Result of binding one queue to an execution owner.
class QueueBindingCreateResult {
public:
  QueueBindingCreateStatus status = QueueBindingCreateStatus::Rejected;
  std::unique_ptr<QueueBinding> binding;

  explicit operator bool() const {
    return status == QueueBindingCreateStatus::Bound && binding != nullptr;
  }
};

/// @brief Reusable creator for processor-specific per-queue bindings.
class QueueBindingFactory {
public:
  virtual ~QueueBindingFactory() = default;

  /// @brief Create a fully attached binding or leave no processor state behind.
  [[nodiscard]] virtual QueueBindingCreateResult
  create_binding(const QueueRegistrationRequest &request) = 0;
};

/// @brief Frontend-independent request to register one GPU queue.
struct QueueRegistrationRequest {
  QueueIdentity identity;
  QueueRingLayout ring;
  QueueDoorbellBinding doorbell;
  std::shared_ptr<QueueBindingFactory> binding_factory;
  /// @brief Optional initial hardware read cursor.
  /// @details MES supplies the cursor captured from the SDMA MQD; legacy/KFD
  /// queues leave it unset so the binding loads the already-published cursor
  /// from the consumer pointer address on first service.
  std::optional<uint64_t> initial_consumer_cursor = std::nullopt;
  uint64_t queue_descriptor_address = 0;
  uint64_t exception_status_address = 0;
  uint32_t exception_event_id = 0;
  uint32_t engine_id = 0;
  bool xcd_fanout = false;
  QueueType type = QueueType::Compute;
  QueuePacketFormat packet_format = QueuePacketFormat::Aql;
};

/// @brief Shared queue registry used by legacy and PCI/VFIO frontends.
class GpuQueueRegistry {
public:
  explicit GpuQueueRegistry(GpuVm &gpu_vm) : gpu_vm_(gpu_vm) {}
  ~GpuQueueRegistry();

  [[nodiscard]] QueueHandle register_queue(const QueueRegistrationRequest &request);
  [[nodiscard]] QueueReconfigureResult reconfigure_queue(QueueHandle handle,
                                                         QueueReconfigureRequest request);
  [[nodiscard]] QueueSubmissionResult submit_producer(QueueHandle handle, uint64_t producer_value);
  [[nodiscard]] QueueCloseResult unregister_queue(QueueHandle handle,
                                                  QueueCloseMode mode = QueueCloseMode::Graceful);

  /// @brief Stop admission, close every queue binding, and reopen registration.
  /// @details Address-space reset is owned by the SoC reset coordinator rather
  /// than this queue-lifetime registry.
  void close_all();

  [[nodiscard]] bool contains(QueueHandle handle) const;
  [[nodiscard]] std::size_t active_queues() const;
  [[nodiscard]] uint64_t lifecycle_epoch() const;
  [[nodiscard]] bool accepting_registrations_for_test() const;

private:
  friend class GpuQueueRegistryTestAccess;

  enum class QueueState : uint8_t { Open, Closing, Dead };

  class QueueRecord {
  public:
    explicit QueueRecord(QueueIdentity identity_value) : identity(std::move(identity_value)) {}
    ~QueueRecord();

    void close_binding() noexcept;

    QueueIdentity identity;
    std::unique_ptr<QueueBinding> binding;
    uint32_t active_operations = 0;
    QueueState state = QueueState::Open;
  };

  class Slot {
  public:
    uint64_t generation = 1;
    std::shared_ptr<QueueRecord> queue;
  };

  class OperationLease {
  public:
    OperationLease() = default;
    OperationLease(GpuQueueRegistry &registry, std::shared_ptr<QueueRecord> queue);
    OperationLease(const OperationLease &) = delete;
    OperationLease &operator=(const OperationLease &) = delete;
    OperationLease(OperationLease &&other) noexcept;
    OperationLease &operator=(OperationLease &&other) noexcept;
    ~OperationLease();

    explicit operator bool() const { return queue_ != nullptr; }
    QueueRecord *operator->() const { return queue_.get(); }

  private:
    void release();

    GpuQueueRegistry *registry_ = nullptr;
    std::shared_ptr<QueueRecord> queue_;
  };

  /// @brief Keep close-all exclusion for the complete registration transaction.
  class RegistrationTransaction {
  public:
    explicit RegistrationTransaction(GpuQueueRegistry &registry) : registry_(registry) {}
    RegistrationTransaction(const RegistrationTransaction &) = delete;
    RegistrationTransaction &operator=(const RegistrationTransaction &) = delete;
    ~RegistrationTransaction();

  private:
    GpuQueueRegistry &registry_;
  };

  /// @brief Roll back VM retention until registration commits.
  class RegistrationRollback {
  public:
    RegistrationRollback(GpuVm &gpu_vm, AddressSpaceHandle address_space)
        : gpu_vm_(gpu_vm), address_space_(address_space) {}
    RegistrationRollback(const RegistrationRollback &) = delete;
    RegistrationRollback &operator=(const RegistrationRollback &) = delete;
    ~RegistrationRollback();

    void commit() { committed_ = true; }

  private:
    GpuVm &gpu_vm_;
    AddressSpaceHandle address_space_;
    bool committed_ = false;
  };

  [[nodiscard]] QueueHandle allocate_locked(std::shared_ptr<QueueRecord> queue);
  [[nodiscard]] std::shared_ptr<QueueRecord> find_locked(QueueHandle handle) const;
  /// @brief Admit one callback only while the record is Open.
  /// @details Closing revokes the public handle first, then waits for every
  /// previously admitted lease to leave its binding callback.
  [[nodiscard]] OperationLease acquire_operation(QueueHandle handle);
  void release_operation(const std::shared_ptr<QueueRecord> &queue);
  void close(std::shared_ptr<QueueRecord> queue);
  void finish_registration();
  void close_all(bool reopen_registration);

  GpuVm &gpu_vm_;
  mutable std::mutex mutex_;
  std::condition_variable condition_;
  std::vector<Slot> slots_;
  std::vector<uint32_t> free_slots_;
  uint64_t lifecycle_epoch_ = 1;
  uint64_t admission_epoch_ = 1;
  uint32_t active_registrations_ = 0;
  uint32_t active_closes_ = 0;
  bool admission_open_ = true;
  bool close_all_in_progress_ = false;
  bool fail_next_registry_allocation_ = false;
};

} // namespace rocjitsu::amdgpu
