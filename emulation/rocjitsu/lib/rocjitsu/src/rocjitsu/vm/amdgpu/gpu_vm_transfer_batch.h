// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/vm/amdgpu/device_cache_coherence.h"
#include "rocjitsu/vm/amdgpu/gpu_vm_access.h"

namespace rocjitsu::amdgpu {

/// Consecutive synchronous accesses share a PT permit and one exclusive cache
/// boundary. A deferred owner takes its own locks after the batch releases both.
/// Only the byte count and token outlive this issue attempt.
class GpuVmTransferBatch {
public:
  GpuVmTransferBatch(const GpuVmAccess &access, DeviceCacheCoherence &coherence,
                     size_t &completed_bytes, VmTransferToken &token)
      : access_(access), coherence_(coherence), completed_bytes_(completed_bytes), token_(token) {}
  VmAccessOutcome read(uint64_t address, std::span<std::byte> bytes);
  VmAccessOutcome write(uint64_t address, std::span<const std::byte> bytes);

private:
  friend class GpuVmAccess;
  template <typename Byte> VmAccessOutcome transfer(uint64_t address, std::span<Byte> bytes);
  VmAccessOutcome acquire();
  void release();

  const GpuVmAccess &access_;
  DeviceCacheCoherence &coherence_;
  size_t &completed_bytes_;
  VmTransferToken &token_;
  MemoryIngressPermit pt_permit_;
  std::optional<DeviceCacheCoherence::AtomicBoundary> boundary_;
};

} // namespace rocjitsu::amdgpu
