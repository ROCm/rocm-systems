// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file physical_memory_access.h
/// @brief Adapter from PCI transport storage to the core GPU VM backing API.

#pragma once

#include "rocjitsu/vm/amdgpu/gpu_vm.h"

#include <memory>

namespace simdojo {
class PciTransportSession;
}

namespace rocjitsu {

class PciMemoryAccess;

/// Optional VFIO peer path discovered through a retained PCI transport lease.
/// Preparation owns the result and performs no memory effect or fabric send.
/// Nullopt means the IOVA is ordinary guest RAM; a known rejected peer must
/// return a terminal token so a caller cannot retry it through guest RAM.
/// The executing snapshot and selected translation pointers are borrowed only
/// for the prepare call. A provider must copy them if effect admission needs
/// either after preparation. A known peer route also needs the copied PT epoch
/// from the counted preparation walk; missing provenance is a terminal denial.
class PciPeerDataProvider {
public:
  virtual ~PciPeerDataProvider() = default;
  [[nodiscard]] virtual std::optional<amdgpu::VmTransferToken>
  prepare_peer_read(uint64_t requester_generation, uint64_t iova, std::size_t length,
                    const std::shared_ptr<amdgpu::VmTransferEffectGate> &gate,
                    const amdgpu::GpuVmAccess *executing_access, uint64_t gpu_va,
                    const amdgpu::VmTranslation *selected_translation,
                    const std::optional<amdgpu::MemoryPtEpoch> &pt_epoch) = 0;
  [[nodiscard]] virtual std::optional<amdgpu::VmTransferToken>
  prepare_peer_write(uint64_t requester_generation, uint64_t iova, std::span<const std::byte> bytes,
                     const std::shared_ptr<amdgpu::VmTransferEffectGate> &gate,
                     const amdgpu::GpuVmAccess *executing_access, uint64_t gpu_va,
                     const amdgpu::VmTranslation *selected_translation,
                     const std::optional<amdgpu::MemoryPtEpoch> &pt_epoch) = 0;
};

/// @brief Expose guest RAM and VRAM as physical domains to core VM translators.
///
/// @details The PCI layer owns transport and storage access only. Address-space
/// policy remains in the core translator, and the legacy compatibility domain
/// is deliberately unavailable through a PCI device.
class PciPhysicalMemoryAccess final : public amdgpu::PhysicalMemoryAccess {
public:
  explicit PciPhysicalMemoryAccess(PciMemoryAccess &memory);

  [[nodiscard]] bool supports_translated_local_cache() const override {
    return local_cache_coherent_;
  }

  [[nodiscard]] amdgpu::MemoryIngressAdmission admit_pt_rewalk() const override;
  [[nodiscard]] amdgpu::MemoryIngressAdmission
  admit_pt_rewalk(const amdgpu::MemoryPtEpoch &expected) const override;

  [[nodiscard]] std::optional<amdgpu::VmTransferToken>
  prepare_deferred_system_read(uint64_t address, std::size_t length, amdgpu::VmAccessKind access,
                               const std::shared_ptr<amdgpu::VmTransferEffectGate> &gate) override;
  [[nodiscard]] std::optional<amdgpu::VmTransferToken>
  prepare_deferred_system_read(uint64_t address, std::size_t length, amdgpu::VmAccessKind access,
                               const std::shared_ptr<amdgpu::VmTransferEffectGate> &gate,
                               const amdgpu::GpuVmAccess &executing_access, uint64_t gpu_va,
                               const amdgpu::VmTranslation &selected_translation) override;
  [[nodiscard]] std::optional<amdgpu::VmTransferToken>
  prepare_deferred_system_read(uint64_t address, std::size_t length, amdgpu::VmAccessKind access,
                               const std::shared_ptr<amdgpu::VmTransferEffectGate> &gate,
                               const amdgpu::GpuVmAccess &executing_access, uint64_t gpu_va,
                               const amdgpu::VmTranslation &selected_translation,
                               const std::optional<amdgpu::MemoryPtEpoch> &pt_epoch) override;
  [[nodiscard]] std::optional<amdgpu::VmTransferToken>
  prepare_deferred_system_write(uint64_t address, std::span<const std::byte> bytes,
                                const std::shared_ptr<amdgpu::VmTransferEffectGate> &gate) override;
  [[nodiscard]] std::optional<amdgpu::VmTransferToken>
  prepare_deferred_system_write(uint64_t address, std::span<const std::byte> bytes,
                                const std::shared_ptr<amdgpu::VmTransferEffectGate> &gate,
                                const amdgpu::GpuVmAccess &executing_access, uint64_t gpu_va,
                                const amdgpu::VmTranslation &selected_translation) override;
  [[nodiscard]] std::optional<amdgpu::VmTransferToken>
  prepare_deferred_system_write(uint64_t address, std::span<const std::byte> bytes,
                                const std::shared_ptr<amdgpu::VmTransferEffectGate> &gate,
                                const amdgpu::GpuVmAccess &executing_access, uint64_t gpu_va,
                                const amdgpu::VmTranslation &selected_translation,
                                const std::optional<amdgpu::MemoryPtEpoch> &pt_epoch) override;

  amdgpu::VmAccessOutcome read(amdgpu::VmMemoryDomain domain, uint64_t address,
                               std::span<std::byte> bytes) override;
  amdgpu::VmAccessOutcome write(amdgpu::VmMemoryDomain domain, uint64_t address,
                                std::span<const std::byte> bytes) override;
  amdgpu::AtomicLoadResult atomic_load(amdgpu::VmMemoryDomain domain, uint64_t address,
                                       uint32_t width) override;
  amdgpu::VmAccessOutcome atomic_store(amdgpu::VmMemoryDomain domain, uint64_t address,
                                       uint32_t width, uint64_t value) override;
  amdgpu::AtomicCompareExchangeResult compare_exchange(amdgpu::VmMemoryDomain domain,
                                                       uint64_t address, uint32_t width,
                                                       uint64_t expected,
                                                       uint64_t desired) override;

private:
  [[nodiscard]] std::optional<amdgpu::VmTransferToken>
  prepare_peer_read(uint64_t address, std::size_t length, amdgpu::VmAccessKind access,
                    const std::shared_ptr<amdgpu::VmTransferEffectGate> &gate,
                    const amdgpu::GpuVmAccess *executing_access, uint64_t gpu_va,
                    const amdgpu::VmTranslation *selected_translation,
                    const std::optional<amdgpu::MemoryPtEpoch> &pt_epoch);
  [[nodiscard]] std::optional<amdgpu::VmTransferToken>
  prepare_peer_write(uint64_t address, std::span<const std::byte> bytes,
                     const std::shared_ptr<amdgpu::VmTransferEffectGate> &gate,
                     const amdgpu::GpuVmAccess *executing_access, uint64_t gpu_va,
                     const amdgpu::VmTranslation *selected_translation,
                     const std::optional<amdgpu::MemoryPtEpoch> &pt_epoch);

  PciMemoryAccess *memory_;
  std::shared_ptr<simdojo::PciTransportSession> session_;
  bool peer_enabled_ = false;
  bool local_cache_coherent_ = false;
};

} // namespace rocjitsu
