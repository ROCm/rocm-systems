// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/vm/amdgpu/gpu_memory_access.h"

namespace rocjitsu::test {
class CountingGpuMemory final : public amdgpu::PhysicalMemoryAccess {
public:
  explicit CountingGpuMemory(amdgpu::GpuMemory &memory) : backing_(memory) {}

  amdgpu::VmAccessOutcome read(amdgpu::VmMemoryDomain domain, uint64_t address,
                               std::span<std::byte> bytes) override {
    ++reads;
    return fault_reads ? amdgpu::VmAccessOutcome::Faulted : backing_.read(domain, address, bytes);
  }
  amdgpu::VmAccessOutcome write(amdgpu::VmMemoryDomain domain, uint64_t address,
                                std::span<const std::byte> bytes) override {
    return backing_.write(domain, address, bytes);
  }

  size_t reads = 0;
  bool fault_reads = false;

private:
  amdgpu::GpuMemoryPhysicalAccess backing_;
};
} // namespace rocjitsu::test
