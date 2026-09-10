// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file mes_engine.h
/// @brief Transport-neutral MES queue management and command-processor
/// integration.

#pragma once

#include "rocjitsu/vm/amdgpu/gpu_queue_registry.h"
#include "rocjitsu/vm/amdgpu/gpu_vm.h"
#include "rocjitsu/vm/amdgpu/interrupt_sink.h"
#include "rocjitsu/vm/amdgpu/pm4_packet_processor.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rocjitsu::amdgpu {
class SdmaQueueBindingFactory;

/// @brief Processor binding and topology policy selected for one compute queue.
struct ComputeQueueBindingPlan {
  std::shared_ptr<QueueBindingFactory> factory;
  bool xcd_fanout = false;
};

/// @brief Device-level compute placement used by transport-neutral MES semantics.
class ComputeQueueBindingProvider {
public:
  virtual ~ComputeQueueBindingProvider() = default;

  [[nodiscard]] virtual std::optional<ComputeQueueBindingPlan> make_aql(uint32_t queue_ordinal) = 0;
  [[nodiscard]] virtual std::optional<ComputeQueueBindingPlan>
  make_pm4(uint32_t queue_ordinal, Pm4PacketCallbacks callbacks) = 0;
};

/// @brief Result of servicing one MES or MES-created queue doorbell.
enum class MesDoorbellDisposition : uint8_t { Ignored, Complete, Retry, Faulted };

/// @brief Transport-supplied state for one MES servicing attempt.
class MesDoorbellContext {
public:
  using PhysicalMemoryFactory = std::function<std::shared_ptr<PhysicalMemoryAccess>()>;

  /// @brief Capture the frontend memory factory and the transport generation it belongs to.
  MesDoorbellContext(PhysicalMemoryFactory physical_memory_factory, uintptr_t source_identity,
                     uint64_t source_generation)
      : physical_memory_factory_(std::move(physical_memory_factory)),
        source_identity_(source_identity), source_generation_(source_generation) {}

  /// @brief Construct a transport-backed physical-memory accessor for this attempt.
  [[nodiscard]] std::shared_ptr<PhysicalMemoryAccess> make_physical_memory() const {
    return physical_memory_factory_ ? physical_memory_factory_() : nullptr;
  }
  /// @brief Return the stable identity of the frontend that originated the notification.
  [[nodiscard]] uintptr_t source_identity() const { return source_identity_; }
  /// @brief Return the frontend generation used to reject stale retry state.
  [[nodiscard]] uint64_t source_generation() const { return source_generation_; }

private:
  PhysicalMemoryFactory physical_memory_factory_;
  uintptr_t source_identity_ = 0;
  uint64_t source_generation_ = 0;
};

/// @brief Register-programmed kernel MES queue presented by an MMIO adapter.
struct MesKernelQueue {
  /// @brief GPU virtual address of the firmware-free MES ring.
  uint64_t ring_base = 0;
  /// @brief GPU virtual address where MES publishes its consumer position.
  uint64_t read_pointer_address = 0;
  /// @brief GPU virtual address from which MES reads the producer position.
  uint64_t write_pointer_address = 0;
  /// @brief Ring capacity in 32-bit words.
  uint64_t ring_dwords = 0;
  /// @brief Byte offset of the doorbell assigned to this queue.
  uint64_t doorbell_offset = 0;
  /// @brief Whether the register-programmed queue is enabled.
  bool active = false;
};

/// @brief Frontend operations whose meaning remains outside the MES core.
class MesFrontendCallbacks {
public:
  using Pm4UconfigWriter = std::function<bool(uint64_t register_dword, uint32_t value)>;
  using QueuePointerPublisher = std::function<void(uint64_t read_pointer, uint64_t write_pointer)>;
  using DoorbellResolver =
      std::function<QueueDoorbellBinding(uint32_t offset, QueuePacketFormat packet_format)>;

  /// @brief Construct an empty callback set for a detached MES engine.
  MesFrontendCallbacks() = default;
  /// @brief Construct the complete set of frontend-specific MES operations.
  MesFrontendCallbacks(Pm4UconfigWriter pm4_uconfig_writer, QueuePointerPublisher pointer_publisher,
                       DoorbellResolver doorbell_resolver)
      : pm4_uconfig_writer_(std::move(pm4_uconfig_writer)),
        pointer_publisher_(std::move(pointer_publisher)),
        doorbell_resolver_(std::move(doorbell_resolver)) {}

  /// @brief Return the register-write callback used by the PM4 packet processor.
  [[nodiscard]] const Pm4UconfigWriter &pm4_uconfig_writer() const { return pm4_uconfig_writer_; }
  /// @brief Publish MES ring positions when the frontend exposes matching registers.
  void publish_queue_pointers(uint64_t read_pointer, uint64_t write_pointer) const {
    if (pointer_publisher_)
      pointer_publisher_(read_pointer, write_pointer);
  }
  /// @brief Resolve transport-specific notification state for a created queue.
  [[nodiscard]] QueueDoorbellBinding queue_doorbell(uint32_t offset,
                                                    QueuePacketFormat packet_format) const {
    return doorbell_resolver_ ? doorbell_resolver_(offset, packet_format)
                              : QueueDoorbellBinding{.offset = offset};
  }

private:
  Pm4UconfigWriter pm4_uconfig_writer_;
  QueuePointerPublisher pointer_publisher_;
  DoorbellResolver doorbell_resolver_;
};

/// @brief MES semantic engine shared by transport adapters.
///
/// @details This class owns queue/address-space lifetime, MES frame retirement,
/// retry journals, and firmware-free MES control execution. A PCI or legacy
/// frontend supplies only physical-memory access, register callbacks, and
/// interrupt/SDMA adapters. No MMIO register layout or PCI transport type is
/// visible here.
class MesEngine {
public:
  /// @brief Construct the semantic engine over shared VM and queue services.
  MesEngine(GpuVm &gpu_vm, GpuQueueRegistry &queue_registry,
            ComputeQueueBindingProvider &compute_bindings)
      : gpu_vm_(gpu_vm), queue_registry_(queue_registry), compute_bindings_(compute_bindings) {}

  MesEngine(const MesEngine &) = delete;
  MesEngine &operator=(const MesEngine &) = delete;
  MesEngine(MesEngine &&) = delete;
  MesEngine &operator=(MesEngine &&) = delete;

  /// @brief Attach one frontend adapter after all state from any predecessor is quiescent.
  /// @returns false if queues or retry journals still depend on the prior frontend.
  [[nodiscard]] bool
  attach_frontend(std::string diagnostic_name,
                  std::shared_ptr<SdmaQueueBindingFactory> sdma_queue_binding_factory,
                  InterruptSink interrupt_sink, MesFrontendCallbacks callbacks);
  /// @brief Detach frontend callbacks while retaining any core-owned orphan bindings.
  /// @returns false while a queue or retry journal still requires the frontend.
  [[nodiscard]] bool detach_frontend();
  /// @brief Destroy every MES-created queue and address-space binding that can be released.
  /// @returns true only when all frontend-created state was released.
  [[nodiscard]] bool teardown_queues();
  /// @brief Return the engine to its initial queue-allocation state.
  /// @returns false when a retained core reference prevents complete teardown.
  [[nodiscard]] bool reset();

  /// @brief Service one frontend doorbell through the shared MES semantics.
  /// @details The context identifies the transport generation and supplies the
  /// physical-memory route captured by any retry journal created by this call.
  [[nodiscard]] MesDoorbellDisposition
  notify_doorbell(uint64_t byte_offset, uint64_t write_pointer,
                  const std::optional<MesKernelQueue> &kernel_queue,
                  const MesDoorbellContext &context);

  /// @brief Return the number of queues currently mapped by MES.
  [[nodiscard]] std::size_t active_queues() const { return mapped_queues_.size(); }
  /// @brief Return the number of process address spaces currently tracked by MES.
  [[nodiscard]] std::size_t active_address_spaces() const { return address_spaces_.size(); }

private:
  enum class QueueKind : uint8_t { Mes, Compute, Sdma };
  enum class MesFramePhase : uint8_t { Semantic, Completion, ReadPointer, Terminal };

  class Queue {
  public:
    uint64_t ring_base = 0;
    uint64_t read_pointer_address = 0;
    uint64_t write_pointer_address = 0;
    uint64_t initial_read_pointer = 0;
    uint64_t page_table_base = 0;
    uint64_t process_context_address = 0;
    uint64_t ring_dwords = 0;
    uint64_t doorbell_offset = 0;
    uint32_t process_id = 0;
    uint32_t queue_id = 0;
    uint32_t engine_id = 0;
    bool active = false;
    bool aql = false;
    AddressSpaceHandle address_space;
    QueueHandle queue_handle;
    QueueKind kind = QueueKind::Mes;
  };

  class AddressSpace {
  public:
    AddressSpaceHandle handle;
    uint64_t page_table_base = 0;
    uint64_t process_context_address = 0;
  };

  class QueueLookup {
  public:
    QueueLookup(VmAccessOutcome outcome, std::optional<Queue> queue)
        : outcome_(outcome), queue_(std::move(queue)) {}

  private:
    friend class MesEngine;
    VmAccessOutcome outcome_ = VmAccessOutcome::Malformed;
    std::optional<Queue> queue_;
  };

  class PendingMesFrame {
  public:
    PendingMesFrame(Queue queue, GpuVmAccess access, uint64_t frame_read_pointer,
                    uint64_t next_read_pointer, uint64_t completion_address,
                    uint64_t completion_value, uintptr_t source_identity,
                    uint64_t source_generation)
        : queue_(std::move(queue)), access_(std::move(access)),
          frame_read_pointer_(frame_read_pointer), next_read_pointer_(next_read_pointer),
          completion_address_(completion_address), completion_value_(completion_value),
          source_identity_(source_identity), source_generation_(source_generation) {}

  private:
    friend class MesEngine;
    Queue queue_;
    GpuVmAccess access_;
    uint64_t frame_read_pointer_ = 0;
    uint64_t next_read_pointer_ = 0;
    uint64_t completion_address_ = 0;
    uint64_t completion_value_ = 0;
    MesFramePhase phase_ = MesFramePhase::Semantic;
    VmAccessOutcome terminal_outcome_ = VmAccessOutcome::Malformed;
    uintptr_t source_identity_ = 0;
    uint64_t source_generation_ = 0;
  };

  [[nodiscard]] static Queue kernel_queue(const MesKernelQueue &queue);
  [[nodiscard]] QueueLookup queue_from_mqd(uint64_t mqd_address, QueueKind kind,
                                           const GpuVmAccess &gart_access) const;
  [[nodiscard]] MesDoorbellDisposition process_queue(Queue queue, uint64_t write_pointer,
                                                     const MesDoorbellContext &context);
  [[nodiscard]] MesDoorbellDisposition process_mes_queue(Queue queue, uint64_t write_pointer,
                                                         const MesDoorbellContext &context);
  [[nodiscard]] VmAccessOutcome execute_mes_semantic(std::span<const std::byte> frame,
                                                     uint32_t opcode,
                                                     const MesDoorbellContext &context,
                                                     const GpuVmAccess &gart_access);
  [[nodiscard]] VmAccessOutcome publish_mes_frame(PendingMesFrame &pending);
  [[nodiscard]] VmAccessOutcome remove_queue(uint64_t doorbell_offset);
  [[nodiscard]] VmAccessOutcome commit_queue(Queue queue, bool created_address_space);
  void release_unused_address_space(uint32_t process_id);
  void reap_orphaned_address_spaces();
  void rollback_created_address_space(const Queue &queue, bool created_address_space);
  [[nodiscard]] bool bind_process_address_space(Queue &queue,
                                                std::shared_ptr<PhysicalMemoryAccess> memory,
                                                bool &created_address_space);
  [[nodiscard]] bool update_process_address_space(uint64_t process_context_address,
                                                  uint64_t page_table_base,
                                                  std::shared_ptr<PhysicalMemoryAccess> memory);
  [[nodiscard]] bool register_aql_queue(Queue &queue);
  [[nodiscard]] bool register_pm4_queue(Queue &queue);
  [[nodiscard]] bool register_sdma_queue(Queue &queue);
  [[nodiscard]] bool same_source(const PendingMesFrame &pending,
                                 const MesDoorbellContext &context) const;

  GpuVm &gpu_vm_;
  GpuQueueRegistry &queue_registry_;
  ComputeQueueBindingProvider &compute_bindings_;
  std::vector<Queue> mapped_queues_;
  std::unordered_map<uint32_t, AddressSpace> address_spaces_;
  std::vector<AddressSpaceHandle> orphaned_address_spaces_;
  std::unordered_map<uint64_t, PendingMesFrame> pending_mes_frames_;
  std::shared_ptr<SdmaQueueBindingFactory> sdma_queue_binding_factory_;
  InterruptSink interrupt_sink_;
  MesFrontendCallbacks callbacks_;
  std::string diagnostic_name_ = "MES";
  uint32_t next_queue_ordinal_ = 0;
  bool frontend_attached_ = false;
};

} // namespace rocjitsu::amdgpu
