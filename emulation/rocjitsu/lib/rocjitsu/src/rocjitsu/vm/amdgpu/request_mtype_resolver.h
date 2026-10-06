// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/vm/amdgpu/gpu_vm.h"
#include "rocjitsu/vm/amdgpu/mtype.h"

#include <cstdint>
#include <optional>

namespace rocjitsu {
namespace amdgpu {

/// Resolves effective MTYPE through one lifetime-safe VM binding snapshot.
/// The caller's cache retains copied policy across requests, but no VM backing.
class RequestMtypeResolver {
public:
  RequestMtypeResolver(GpuVm *gpu_vm, uint32_t vmid, VmMtypeCache &cache,
                       std::optional<Mtype> instruction_mtype = std::nullopt)
      : access_(vmid != 0 && gpu_vm != nullptr ? gpu_vm->snapshot_vmid(vmid) : std::nullopt),
        mtype_cache_(cache), fallback_(instruction_mtype.value_or(Mtype::RW)),
        combine_(instruction_mtype.has_value()) {}

  Mtype fallback() const { return fallback_; }

  Mtype at(uint64_t addr) {
    if (!access_)
      return fallback_;
    const std::optional<Mtype> mtype = access_->query_mtype(addr, mtype_cache_);
    if (!mtype)
      return fallback_;
    return combine_ ? effective_mtype(fallback_, *mtype) : *mtype;
  }

private:
  std::optional<GpuVmAccess> access_;
  VmMtypeCache &mtype_cache_;
  Mtype fallback_;
  bool combine_;
};

} // namespace amdgpu
} // namespace rocjitsu
