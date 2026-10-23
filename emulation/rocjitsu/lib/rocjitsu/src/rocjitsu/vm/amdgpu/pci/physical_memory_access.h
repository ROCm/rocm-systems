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

/// @brief Expose guest RAM and VRAM as physical domains to core VM translators.
///
/// @details The PCI layer owns transport and storage access only. Address-space
/// policy remains in the core translator, and the legacy compatibility domain
/// is deliberately unavailable through a PCI device.
class PciPhysicalMemoryAccess final : public amdgpu::PhysicalMemoryAccess {
public:
  explicit PciPhysicalMemoryAccess(PciMemoryAccess &memory);

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
  PciMemoryAccess *memory_;
  std::shared_ptr<simdojo::PciTransportSession> session_;
};

} // namespace rocjitsu
