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
      : mtype_cache_(cache), fallback_(instruction_mtype.value_or(Mtype::RW)),
        combine_(instruction_mtype.has_value()) {
    bind_access(gpu_vm, vmid);
  }

  RequestMtypeResolver(const RequestMtypeResolver &) = delete;
  RequestMtypeResolver &operator=(const RequestMtypeResolver &) = delete;
  RequestMtypeResolver(RequestMtypeResolver &&) = delete;
  RequestMtypeResolver &operator=(RequestMtypeResolver &&) = delete;

  Mtype fallback() const { return fallback_; }

  bool cached_private_uc_hint(uint64_t addr) const {
    return access_ && access_->cached_private_uc_hint(addr, mtype_cache_);
  }

  std::optional<Mtype> cached_private_ram_mtype(uint64_t addr) const {
    if (!access_)
      return std::nullopt;
    const auto policy = access_->cached_private_ram_mtype(addr, mtype_cache_);
    return policy && combine_ ? std::optional{effective_mtype(fallback_, *policy)} : policy;
  }

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
  VmMtypeCache &mtype_cache_;
  Mtype fallback_;
  bool combine_;
};

} // namespace amdgpu
} // namespace rocjitsu
