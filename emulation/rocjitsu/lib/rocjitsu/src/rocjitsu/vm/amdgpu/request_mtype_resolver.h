// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_VM_AMDGPU_REQUEST_MTYPE_RESOLVER_H_
#define ROCJITSU_VM_AMDGPU_REQUEST_MTYPE_RESOLVER_H_

#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/mtype.h"

#include <cstdint>
#include <limits>

namespace rocjitsu {
namespace amdgpu {

/// Caches the effective MTYPE for the current page within one memory request.
/// Each chunk briefly reacquires the page-table request lease and reuses the
/// cached value only while the VMID binding and page-table generation match.
class RequestMtypeResolver {
public:
  RequestMtypeResolver(GpuMemory *memory, uint32_t vmid)
      : memory_(memory), vmid_(vmid), fallback_(Mtype::RW), combine_(false) {}

  RequestMtypeResolver(GpuMemory *memory, uint32_t vmid, Mtype instruction_mtype)
      : memory_(memory), vmid_(vmid), fallback_(instruction_mtype), combine_(true) {}

  Mtype fallback() const { return fallback_; }

  Mtype at(uint64_t addr) {
    if (!memory_)
      return fallback_;

    const uint64_t page = addr >> GpuMemory::PAGE_SHIFT;
    if (vmid_ == 0) {
      if (page != page_) {
        page_ = page;
        page_mtype_ = fallback_;
      }
      return page_mtype_;
    }

    if (!memory_->reacquire_page_table_request(vmid_, request_guard_)) {
      // Registrations without a request lease perform a full MTYPE-only walk
      // under the page-table lock for every chunk.
      const Mtype pte_mtype = memory_->pte_mtype(addr, vmid_);
      return combine_ ? effective_mtype(fallback_, pte_mtype) : pte_mtype;
    }

    const bool cacheable = request_guard_.cacheable();
    const uint64_t registry_generation = request_guard_.registry_generation();
    const uint64_t page_table_generation = cacheable ? request_guard_.page_table_generation() : 0;
    if (!cacheable || page != page_ || registry_generation != registry_generation_ ||
        page_table_generation != page_table_generation_) {
      page_ = page;
      registry_generation_ = registry_generation;
      page_table_generation_ = page_table_generation;
      const Mtype pte_mtype = memory_->pte_mtype(addr, request_guard_);
      page_mtype_ = combine_ ? effective_mtype(fallback_, pte_mtype) : pte_mtype;
    }
    // Backing-memory translation may query allocator metadata without the VMID
    // and page-table locks. Release this lease first so that query can reenter
    // KFD page-table mutation; the next chunk revalidates both generations.
    request_guard_.unlock();
    return page_mtype_;
  }

private:
  GpuMemory *memory_;
  uint32_t vmid_;
  Mtype fallback_;
  bool combine_;
  GpuMemory::PageTableRequestGuard request_guard_;
  uint64_t page_ = std::numeric_limits<uint64_t>::max();
  uint64_t registry_generation_ = 0;
  uint64_t page_table_generation_ = 0;
  Mtype page_mtype_ = Mtype::RW;
};

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_REQUEST_MTYPE_RESOLVER_H_
