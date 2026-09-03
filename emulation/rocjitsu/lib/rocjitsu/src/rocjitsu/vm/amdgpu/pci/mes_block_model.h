// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file mes_block_model.h
/// @brief Firmware-free execution of the kernel MES startup queue.

#pragma once

#include "rocjitsu/vm/amdgpu/gpu_vm.h"
#include "rocjitsu/vm/amdgpu/interrupt_sink.h"
#include "rocjitsu/vm/amdgpu/pci/ip_block_model.h"
#include "rocjitsu/vm/amdgpu/pci/ip_discovery.h"
#include "rocjitsu/vm/amdgpu/queue_service.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rocjitsu {

class SoC;

namespace amdgpu {
class CommandProcessor;
}

class SdmaBlockModel;

/// @brief The narrow MES KIQ contract used while the guest driver starts GFX.
class MesBlockModel final : public IpBlockModel {
public:
  [[nodiscard]] static std::unique_ptr<MesBlockModel> create(const IpBlock &block,
                                                             IpRegisterWindow registers);

  [[nodiscard]] std::vector<RegisterClaim> claims() const override;
  bool reset() override;
  /// @brief Drain and destroy every queue and process VM binding owned here.
  [[nodiscard]] bool teardown_queues();
  void attach_soc(SoC *soc) { soc_ = soc; }
  void attach_sdma(SdmaBlockModel *sdma) { sdma_ = sdma; }
  void attach_interrupt_sink(amdgpu::InterruptSink interrupt_sink) {
    interrupt_sink_ = std::move(interrupt_sink);
  }
  [[nodiscard]] DoorbellDisposition observe_doorbell_write(uint64_t byte_offset, uint64_t value,
                                                           std::size_t width,
                                                           PciMemoryAccess &memory) override;

private:
  enum class QueueKind : uint8_t { Mes, Compute, Sdma };
  enum class MesFramePhase : uint8_t { Semantic, Completion, ReadPointer, Terminal };
  enum class ComputeRunPhase : uint8_t { Execute, ReadPointer, Terminal };

  struct Queue {
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
    amdgpu::AddressSpaceHandle address_space;
    amdgpu::QueueHandle queue_handle;
    QueueKind kind = QueueKind::Mes;
  };

  struct AddressSpace {
    amdgpu::AddressSpaceHandle handle;
    uint64_t page_table_base = 0;
    uint64_t process_context_address = 0;
  };

  struct QueueLookup {
    amdgpu::VmAccessOutcome outcome = amdgpu::VmAccessOutcome::Malformed;
    std::optional<Queue> queue;
  };

  struct PendingMesFrame {
    Queue queue;
    amdgpu::GpuVmAccess access;
    uint64_t frame_read_pointer = 0;
    uint64_t next_read_pointer = 0;
    uint64_t completion_address = 0;
    uint64_t completion_value = 0;
    uint32_t opcode = 0;
    MesFramePhase phase = MesFramePhase::Semantic;
    amdgpu::VmAccessOutcome terminal_outcome = amdgpu::VmAccessOutcome::Malformed;
    std::weak_ptr<simdojo::PciTransportSession> transport_session;
    uint64_t transport_generation = 0;
  };

  struct PendingComputeRun {
    Queue queue;
    amdgpu::GpuVmAccess access;
    uint64_t read_pointer = 0;
    ComputeRunPhase phase = ComputeRunPhase::Execute;
    amdgpu::VmAccessOutcome terminal_outcome = amdgpu::VmAccessOutcome::Malformed;
    std::weak_ptr<simdojo::PciTransportSession> transport_session;
    uint64_t transport_generation = 0;
  };

  explicit MesBlockModel(IpRegisterWindow registers);

  [[nodiscard]] Queue direct_queue() const;
  [[nodiscard]] QueueLookup queue_from_mqd(uint64_t mqd_address, QueueKind kind,
                                           const amdgpu::GpuVmAccess &gart_access) const;
  [[nodiscard]] DoorbellDisposition process_queue(Queue queue, uint64_t write_pointer,
                                                  PciMemoryAccess &memory);
  [[nodiscard]] DoorbellDisposition process_mes_queue(Queue queue, uint64_t write_pointer,
                                                      PciMemoryAccess &memory);
  [[nodiscard]] amdgpu::VmAccessOutcome
  execute_mes_semantic(std::span<const std::byte> frame, uint32_t opcode, PciMemoryAccess &memory,
                       const amdgpu::GpuVmAccess &gart_access);
  [[nodiscard]] amdgpu::VmAccessOutcome publish_mes_frame(PendingMesFrame &pending);
  [[nodiscard]] DoorbellDisposition process_compute_queue(Queue queue, uint64_t write_pointer,
                                                          PciMemoryAccess &memory);
  [[nodiscard]] bool remove_queue(uint64_t doorbell_offset);
  [[nodiscard]] bool commit_queue(Queue queue, bool created_address_space);
  void release_unused_address_space(uint32_t process_id);
  void rollback_created_address_space(const Queue &queue, bool created_address_space);
  [[nodiscard]] bool bind_process_address_space(Queue &queue, PciMemoryAccess &memory,
                                                bool &created_address_space);
  [[nodiscard]] bool update_process_address_space(uint64_t process_context_address,
                                                  uint64_t page_table_base,
                                                  PciMemoryAccess &memory);
  [[nodiscard]] bool register_aql_queue(Queue &queue);
  [[nodiscard]] bool register_sdma_queue(Queue &queue, PciMemoryAccess &memory);
  [[nodiscard]] bool read_gpu(PciMemoryAccess &memory, uint64_t address,
                              std::span<std::byte> bytes) const;
  [[nodiscard]] bool write_gpu(PciMemoryAccess &memory, uint64_t address,
                               std::span<const std::byte> bytes) const;
  [[nodiscard]] bool read_queue_gpu(PciMemoryAccess &memory, const Queue &queue, uint64_t address,
                                    std::span<std::byte> bytes) const;
  [[nodiscard]] bool write_queue_gpu(PciMemoryAccess &memory, const Queue &queue, uint64_t address,
                                     std::span<const std::byte> bytes) const;

  std::vector<Queue> mapped_queues_;
  std::unordered_map<uint32_t, AddressSpace> address_spaces_;
  std::unordered_map<uint64_t, PendingMesFrame> pending_mes_frames_;
  std::unordered_map<uint64_t, PendingComputeRun> pending_compute_runs_;
  SoC *soc_ = nullptr;
  SdmaBlockModel *sdma_ = nullptr;
  amdgpu::InterruptSink interrupt_sink_;
  uint32_t next_queue_ordinal_ = 0;
};

} // namespace rocjitsu
