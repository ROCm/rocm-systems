// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file mes_block_model.h
/// @brief MMIO and doorbell adapter for the SoC-owned MES engine.

#pragma once

#include "rocjitsu/vm/amdgpu/interrupt_sink.h"
#include "rocjitsu/vm/amdgpu/pci/ip_block_model.h"
#include "rocjitsu/vm/amdgpu/pci/ip_discovery.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace rocjitsu::amdgpu {
class MesEngine;
struct MesKernelQueue;
class SdmaQueueBindingFactory;
} // namespace rocjitsu::amdgpu

namespace rocjitsu {

/// @brief Adapts the GFX12 MES register surface to the shared core engine.
class MesBlockModel final : public IpBlockModel {
public:
  /// @brief Construct an adapter for a supported MES discovery record.
  /// @returns A model when the record and register window are usable, otherwise nullptr.
  [[nodiscard]] static std::unique_ptr<MesBlockModel> create(const IpBlock &block,
                                                             IpRegisterWindow registers);

  /// @brief Return the register ranges owned by this adapter.
  [[nodiscard]] std::vector<RegisterClaim> claims() const override;
  /// @brief Restore the adapter's register-visible state to reset defaults.
  bool reset() override;

  /// @brief Connect this MMIO adapter to the SoC-owned MES engine and shared services.
  /// @returns false when the core engine is already attached to another frontend.
  [[nodiscard]] bool
  attach_engine(amdgpu::MesEngine &engine,
                std::shared_ptr<amdgpu::SdmaQueueBindingFactory> sdma_queue_binding_factory,
                amdgpu::InterruptSink interrupt_sink);
  /// @brief Disconnect the adapter after its queues and retry journals have drained.
  [[nodiscard]] bool detach_engine();
  /// @brief Request destruction of all queues created through this adapter.
  [[nodiscard]] bool teardown_queues();

  /// @brief Decode a trapped doorbell and delegate its semantics to the core engine.
  [[nodiscard]] DoorbellDisposition observe_doorbell_write(uint64_t byte_offset, uint64_t value,
                                                           std::size_t width,
                                                           PciMemoryAccess &memory) override;

private:
  explicit MesBlockModel(IpRegisterWindow registers);

  [[nodiscard]] amdgpu::MesKernelQueue kernel_queue() const;
  [[nodiscard]] bool write_pm4_uconfig_register(uint64_t register_dword, uint32_t value);
  void update_kernel_queue_pointers(uint64_t read_pointer, uint64_t write_pointer);

  amdgpu::MesEngine *engine_ = nullptr;
};

} // namespace rocjitsu
