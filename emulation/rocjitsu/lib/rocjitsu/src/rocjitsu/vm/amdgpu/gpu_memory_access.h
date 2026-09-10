// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file gpu_memory_access.h
/// @brief Physical-memory adapter for the sparse GPU backing store.

#pragma once

#include "rocjitsu/vm/amdgpu/gpu_vm.h"

namespace rocjitsu::amdgpu {

class GpuMemory;

/// @brief Exposes GpuMemory as local physical storage to GpuVm.
///
/// @details Translation, permissions, and address-space lifetime remain owned
/// by GpuVm. This adapter only converts physical operations to the sparse
/// backing-store API. The referenced GpuMemory must outlive the adapter and any
/// GpuVmAccess snapshot retaining it.
class GpuMemoryPhysicalAccess final : public PhysicalMemoryAccess {
public:
  explicit GpuMemoryPhysicalAccess(GpuMemory &memory) : memory_(&memory) {}

  [[nodiscard]] VmAccessOutcome read(VmMemoryDomain domain, uint64_t address,
                                     std::span<std::byte> bytes) override;
  [[nodiscard]] VmAccessOutcome write(VmMemoryDomain domain, uint64_t address,
                                      std::span<const std::byte> bytes) override;
  [[nodiscard]] AtomicLoadResult atomic_load(VmMemoryDomain domain, uint64_t address,
                                             uint32_t width) override;
  [[nodiscard]] VmAccessOutcome atomic_store(VmMemoryDomain domain, uint64_t address,
                                             uint32_t width, uint64_t value) override;
  [[nodiscard]] AtomicCompareExchangeResult compare_exchange(VmMemoryDomain domain,
                                                             uint64_t address, uint32_t width,
                                                             uint64_t expected,
                                                             uint64_t desired) override;
  [[nodiscard]] VmAccessOutcome atomic_modify(VmMemoryDomain domain, uint64_t address,
                                              uint32_t width,
                                              const AtomicMutation &mutation) override;

private:
  GpuMemory *memory_;
};

} // namespace rocjitsu::amdgpu
