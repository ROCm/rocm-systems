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
class RequestMtypeResolver {
public:
  RequestMtypeResolver(GpuVm *gpu_vm, uint32_t vmid)
      : access_(vmid != 0 && gpu_vm != nullptr ? gpu_vm->snapshot_vmid(vmid) : std::nullopt),
        fallback_(Mtype::RW), combine_(false) {}

  RequestMtypeResolver(GpuVm *gpu_vm, uint32_t vmid, Mtype instruction_mtype)
      : access_(vmid != 0 && gpu_vm != nullptr ? gpu_vm->snapshot_vmid(vmid) : std::nullopt),
        fallback_(instruction_mtype), combine_(true) {}

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
  VmMtypeCache mtype_cache_;
  Mtype fallback_;
  bool combine_;
};

} // namespace amdgpu
} // namespace rocjitsu
