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
  RequestMtypeResolver(GpuVm *gpu_vm, uint32_t vmid) : fallback_(Mtype::RW), combine_(false) {
    bind_access(gpu_vm, vmid);
  }

  RequestMtypeResolver(GpuVm *gpu_vm, uint32_t vmid, Mtype instruction_mtype)
      : fallback_(instruction_mtype), combine_(true) {
    bind_access(gpu_vm, vmid);
  }

  RequestMtypeResolver(const RequestMtypeResolver &) = delete;
  RequestMtypeResolver &operator=(const RequestMtypeResolver &) = delete;
  RequestMtypeResolver(RequestMtypeResolver &&) = delete;
  RequestMtypeResolver &operator=(RequestMtypeResolver &&) = delete;

  Mtype fallback() const { return fallback_; }

  Mtype at(uint64_t addr) {
    if (access_ == nullptr)
      return fallback_;
    const std::optional<Mtype> mtype = access_->query_mtype(addr, mtype_cache_);
    if (!mtype)
      return fallback_;
    return combine_ ? effective_mtype(fallback_, *mtype) : *mtype;
  }

private:
  void bind_access(GpuVm *gpu_vm, uint32_t vmid) {
    if (vmid == 0 || gpu_vm == nullptr)
      return;
    if (GpuVmAccessBatchGuard::active()) {
      access_ = gpu_vm->borrow_snapshot_vmid(vmid);
      return;
    }
    owned_access_ = gpu_vm->snapshot_vmid(vmid);
    access_ = owned_access_ ? &*owned_access_ : nullptr;
  }

  std::optional<GpuVmAccess> owned_access_;
  const GpuVmAccess *access_ = nullptr;
  VmMtypeCache mtype_cache_;
  Mtype fallback_;
  bool combine_;
};

} // namespace amdgpu
} // namespace rocjitsu
