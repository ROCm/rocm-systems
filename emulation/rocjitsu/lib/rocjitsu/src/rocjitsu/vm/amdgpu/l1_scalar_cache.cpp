// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/l1_scalar_cache.h"

#include "rocjitsu/vm/amdgpu/device_cache_coherence.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/request_mtype_resolver.h"

#include <algorithm>
#include <cassert>
#include <cstring>

namespace rocjitsu {
namespace amdgpu {

L1ScalarCache::L1ScalarCache(L2Cache *l2)
    : l2_(l2), registered_domain_(l2_ ? l2_->coherence_domain() : nullptr),
      coherence_epoch_(registered_domain_ ? registered_domain_->current_epoch() : 0) {
  if (registered_domain_)
    registered_domain_->register_l1_scalar_cache(this);
}

L1ScalarCache::~L1ScalarCache() {
  if (registered_domain_)
    registered_domain_->unregister_l1_scalar_cache(this);
}

void L1ScalarCache::set_l2(L2Cache *l2) {
  std::shared_ptr<DeviceCacheCoherence> next_domain = l2 ? l2->coherence_domain() : nullptr;
  if (next_domain != registered_domain_) {
    // Registration takes the coordinator mutex. Never hold the L1 mutex here:
    // scoped maintenance takes the coordinator mutex before this L1 mutex.
    if (next_domain)
      next_domain->register_l1_scalar_cache(this);
    if (registered_domain_)
      registered_domain_->unregister_l1_scalar_cache(this);
  }
  std::lock_guard lock(maintenance_mutex_);
  invalidate_all_lines();
  l2_ = l2;
  registered_domain_ = std::move(next_domain);
  coherence_epoch_ = registered_domain_ ? registered_domain_->current_epoch() : 0;
}

void L1ScalarCache::set_gpu_vm(GpuVm *gpu_vm) {
  std::lock_guard lock(maintenance_mutex_);
  invalidate_all_lines();
  gpu_vm_ = gpu_vm;
  mtype_cache_.clear();
}

bool L1ScalarCache::try_read_peer_cached(uint64_t addr, std::span<uint8_t> dst, uint32_t vmid,
                                         uint64_t admission_generation, uint64_t cache_identity) {
  const uint32_t offset = CacheStore::line_offset(addr);
  if (dst.empty() || dst.size() > CacheStore::LINE_SIZE - offset ||
      (vmid == 0 ? (admission_generation != 0 || cache_identity != 0)
                 : (admission_generation == 0 || cache_identity == 0)))
    return false;

  std::lock_guard lock(maintenance_mutex_);
  if (l2_ == nullptr)
    return false;
  synchronize_epoch();
  const simdojo::CacheTag *tag = cache_.peek(addr, vmid);
  if (tag == nullptr || tag->admission_generation != admission_generation ||
      tag->cache_identity != cache_identity ||
      !CacheStore::bytes_valid(*tag, offset, static_cast<uint32_t>(dst.size())))
    return false;
  cache_.read_line(addr, dst.data(), offset, static_cast<uint32_t>(dst.size()), vmid);
  return true;
}

bool L1ScalarCache::try_fill_peer_cached(uint64_t addr, std::span<const uint8_t> src, uint32_t vmid,
                                         uint64_t admission_generation, uint64_t cache_identity) {
  const uint32_t offset = CacheStore::line_offset(addr);
  if (src.empty() || src.size() > CacheStore::LINE_SIZE - offset ||
      (vmid == 0 ? (admission_generation != 0 || cache_identity != 0)
                 : (admission_generation == 0 || cache_identity == 0)))
    return false;

  std::lock_guard lock(maintenance_mutex_);
  if (l2_ == nullptr)
    return false;
  synchronize_epoch();
  const simdojo::CacheTag *resident = cache_.peek(addr, vmid);
  if (resident != nullptr && resident->dirty)
    return false;
  if (resident != nullptr && (resident->admission_generation != admission_generation ||
                              resident->cache_identity != cache_identity)) {
    cache_.invalidate(addr, vmid);
    resident = nullptr;
  }
  if (resident == nullptr) {
    const CacheStore::Allocation allocated = cache_.allocate_clean_victim_with_data(addr, vmid);
    if (allocated.tag == nullptr)
      return false;
    allocated.tag->admission_generation = admission_generation;
    allocated.tag->cache_identity = cache_identity;
    CacheStore::clear_valid_bytes(*allocated.tag);
  }
  cache_.write_line(addr, src.data(), offset, static_cast<uint32_t>(src.size()), vmid);
  return true;
}

uint64_t L1ScalarCache::current_cache_identity(uint32_t vmid) const {
  if (vmid == 0 || gpu_vm_ == nullptr)
    return 0;
  return gpu_vm_->with_vmid_snapshot(vmid, [](const GpuVmAccess *access) {
    return access != nullptr && access->is_current() ? access->cache_identity() : uint64_t{0};
  });
}

bool L1ScalarCache::has_unidentified_overlap(std::span<const DeviceCacheAliasScope> aliases) {
  bool ambiguous = false;
  cache_.for_each_valid_line([&](const simdojo::CacheTag &tag, uint64_t address, const uint8_t *) {
    if (tag.vmid == 0 || (tag.cache_identity != 0 && tag.admission_generation != 0))
      return;
    ambiguous |= std::ranges::any_of(aliases, [&](const DeviceCacheAliasScope &alias) {
      return alias.domain.coherence.get() == registered_domain_.get() && alias.vmid == tag.vmid &&
             address >= alias.address && address - alias.address < alias.size;
    });
  });
  return ambiguous;
}

void L1ScalarCache::invalidate_selected_lines(std::span<const DeviceCacheAliasScope> aliases) {
  cache_.for_each_valid_line([&](simdojo::CacheTag &tag, uint64_t address, uint8_t *) {
    if (!std::ranges::any_of(aliases, [&](const DeviceCacheAliasScope &alias) {
          return alias.domain.coherence.get() == registered_domain_.get() &&
                 alias.vmid == tag.vmid && alias.admission_generation == tag.admission_generation &&
                 alias.cache_identity == tag.cache_identity && address >= alias.address &&
                 address - alias.address < alias.size;
        }))
      return;
    assert(!tag.dirty && "L1 K$ is write-through; lines should never be dirty");
    cache_.invalidate(address, tag.vmid);
  });
}

VmAccessAdmission L1ScalarCache::admit_access(uint64_t addr, uint32_t size, uint32_t vmid,
                                              VmAccessKind kind, bool report_fault) const {
  if (vmid == 0 || gpu_vm_ == nullptr)
    return {};
  return gpu_vm_->with_vmid_snapshot(vmid, [&](const GpuVmAccess *access) {
    return access ? access->admit_access(addr, size, kind, report_fault)
                  : VmAccessAdmission{.outcome = VmAccessOutcome::Faulted};
  });
}

VmAccessOutcome L1ScalarCache::ensure_line(uint64_t addr, uint32_t vmid,
                                           uint64_t admission_generation, bool fetch_on_miss) {
  const uint64_t cache_identity = current_cache_identity(vmid);
  simdojo::CacheTag *resident = nullptr;
  if (cache_.lookup(addr, &resident, vmid)) {
    if (resident->admission_generation != admission_generation ||
        resident->cache_identity != cache_identity) {
      assert(!resident->dirty && "L1 K$ is write-through; lines should never be dirty");
      cache_.invalidate(addr, vmid);
      resident = nullptr;
    }
  }
  if (resident != nullptr) {
    if (!fetch_on_miss || CacheStore::bytes_valid(*resident, 0, CacheStore::LINE_SIZE))
      return VmAccessOutcome::Complete;

    uint8_t line_buf[CacheStore::LINE_SIZE];
    const VmAccessOutcome outcome =
        l2_->read(CacheStore::line_address(addr), line_buf, CacheStore::LINE_SIZE, Mtype::RW, vmid);
    if (outcome != VmAccessOutcome::Complete)
      return outcome;
    const VmAccessAdmission current =
        admit_access(CacheStore::line_address(addr), CacheStore::LINE_SIZE, vmid,
                     VmAccessKind::Read, /*report_fault=*/false);
    if (!current || !current.cacheable || current.generation != admission_generation ||
        current_cache_identity(vmid) != cache_identity) {
      cache_.invalidate(addr, vmid);
      return VmAccessOutcome::Unavailable;
    }
    cache_.fill_missing_bytes(addr, line_buf, vmid);
    return VmAccessOutcome::Complete;
  }

  uint64_t line_addr = CacheStore::line_address(addr);
  simdojo::CacheTag evicted;
  simdojo::CacheTag *allocated = cache_.allocate(addr, vmid, &evicted);
  allocated->admission_generation = admission_generation;
  allocated->cache_identity = cache_identity;

  assert(!evicted.dirty && "L1 K$ is write-through; lines should never be dirty");

  if (!fetch_on_miss) {
    CacheStore::clear_valid_bytes(*allocated);
    return VmAccessOutcome::Complete;
  }

  uint8_t line_buf[CacheStore::LINE_SIZE];
  const VmAccessOutcome outcome =
      l2_->read(line_addr, line_buf, CacheStore::LINE_SIZE, Mtype::RW, vmid);
  if (outcome != VmAccessOutcome::Complete) {
    cache_.invalidate(addr, vmid);
    return outcome;
  }
  const VmAccessAdmission current = admit_access(line_addr, CacheStore::LINE_SIZE, vmid,
                                                 VmAccessKind::Read, /*report_fault=*/false);
  if (!current || !current.cacheable || current.generation != admission_generation ||
      current_cache_identity(vmid) != cache_identity) {
    cache_.invalidate(addr, vmid);
    return VmAccessOutcome::Unavailable;
  }
  cache_.fill_line(addr, line_buf, vmid);
  return VmAccessOutcome::Complete;
}

void L1ScalarCache::cache_partial_bytes(uint64_t addr, const uint8_t *src, uint32_t size,
                                        uint32_t vmid, uint64_t admission_generation,
                                        uint64_t cache_identity) {
  const VmAccessAdmission current =
      admit_access(CacheStore::line_address(addr), CacheStore::LINE_SIZE, vmid, VmAccessKind::Read,
                   /*report_fault=*/false);
  if (!current || !current.cacheable || current.generation != admission_generation ||
      current_cache_identity(vmid) != cache_identity)
    return;
  simdojo::CacheTag *tag = nullptr;
  if (cache_.lookup(addr, &tag, vmid) && (tag->admission_generation != admission_generation ||
                                          tag->cache_identity != cache_identity)) {
    assert(!tag->dirty && "L1 K$ is write-through; lines should never be dirty");
    cache_.invalidate(addr, vmid);
    tag = nullptr;
  }
  if (tag == nullptr) {
    simdojo::CacheTag evicted;
    tag = cache_.allocate(addr, vmid, &evicted);
    assert(!evicted.dirty && "L1 K$ is write-through; lines should never be dirty");
    tag->admission_generation = admission_generation;
    tag->cache_identity = cache_identity;
    CacheStore::clear_valid_bytes(*tag);
  }
  cache_.write_line(addr, src, CacheStore::line_offset(addr), size, vmid);
}

VmAccessOutcome L1ScalarCache::store(uint64_t addr, uint32_t num_dwords, const uint32_t *src,
                                     uint32_t vmid, Mtype instruction_mtype) {
  std::lock_guard lock(maintenance_mutex_);
  if (num_dwords == 0)
    return VmAccessOutcome::Complete;
  if (num_dwords > UINT32_MAX / sizeof(uint32_t))
    return VmAccessOutcome::Malformed;
  const VmAccessAdmission admission =
      admit_access(addr, num_dwords * sizeof(uint32_t), vmid, VmAccessKind::Write);
  if (!admission)
    return admission.outcome;
  synchronize_epoch();
  RequestMtypeResolver mtypes(gpu_vm_, vmid, mtype_cache_, instruction_mtype);
  for (uint32_t i = 0; i < num_dwords; ++i) {
    uint64_t ea = addr + i * 4;
    uint8_t buf[4];
    std::memcpy(buf, &src[i], 4);
    uint32_t copied = 0;
    while (copied < sizeof(buf)) {
      const uint64_t chunk_addr = ea + copied;
      const uint32_t line_offset = CacheStore::line_offset(chunk_addr);
      const uint32_t chunk =
          std::min<uint32_t>(sizeof(buf) - copied, CacheStore::LINE_SIZE - line_offset);

      const Mtype mtype = mtypes.at(chunk_addr);

      if (mtype == Mtype::UC) {
        flush_line(chunk_addr, vmid);
        const VmAccessOutcome outcome =
            l2_->write(chunk_addr, buf + copied, chunk, Mtype::UC, vmid);
        if (outcome != VmAccessOutcome::Complete)
          return outcome;
        copied += chunk;
        continue;
      }

      if (mtype == Mtype::CC) {
        flush_line(chunk_addr, vmid);
        const VmAccessOutcome outcome =
            l2_->write(chunk_addr, buf + copied, chunk, Mtype::CC, vmid);
        if (outcome != VmAccessOutcome::Complete)
          return outcome;
        copied += chunk;
        continue;
      }

      const VmAccessAdmission chunk_admission =
          admit_access(chunk_addr, chunk, vmid, VmAccessKind::Write);
      if (!chunk_admission)
        return chunk_admission.outcome;
      const VmAccessAdmission line_admission =
          admit_access(CacheStore::line_address(chunk_addr), CacheStore::LINE_SIZE, vmid,
                       VmAccessKind::Write, /*report_fault=*/false);
      if (!line_admission || !line_admission.cacheable ||
          line_admission.generation != chunk_admission.generation) {
        flush_line(chunk_addr, vmid);
        const VmAccessOutcome outcome = l2_->write(chunk_addr, buf + copied, chunk, mtype, vmid);
        if (outcome != VmAccessOutcome::Complete)
          return outcome;
        copied += chunk;
        continue;
      }

      const VmAccessOutcome fill_outcome =
          ensure_line(chunk_addr, vmid, chunk_admission.generation, /*fetch_on_miss=*/false);
      if (fill_outcome != VmAccessOutcome::Complete)
        return fill_outcome;

      const VmAccessOutcome write_outcome =
          l2_->write(chunk_addr, buf + copied, chunk, mtype, vmid);
      if (write_outcome != VmAccessOutcome::Complete)
        return write_outcome;

      simdojo::CacheTag *tag = nullptr;
      cache_.lookup(chunk_addr, &tag, vmid);
      assert(tag != nullptr && "ensure_line must guarantee hit");

      const VmAccessAdmission current =
          admit_access(CacheStore::line_address(chunk_addr), CacheStore::LINE_SIZE, vmid,
                       VmAccessKind::Write, /*report_fault=*/false);
      if (!current || !current.cacheable || current.generation != chunk_admission.generation ||
          tag->cache_identity != current_cache_identity(vmid)) {
        cache_.invalidate(chunk_addr, vmid);
        copied += chunk;
        continue;
      }

      cache_.write_line(chunk_addr, buf + copied, line_offset, chunk, vmid);
      tag->dirty = false;
      copied += chunk;
    }
  }
  return VmAccessOutcome::Complete;
}

void L1ScalarCache::writeback_all(uint32_t vmid) {
  // K$ is write-through, so all stored bytes have already reached L2.
  (void)vmid;
}

void L1ScalarCache::invalidate_all() {
  std::lock_guard lock(maintenance_mutex_);
  synchronize_epoch();
  invalidate_all_lines();
}

void L1ScalarCache::discard(uint64_t addr, uint32_t num_lines, uint32_t vmid) {
  std::lock_guard lock(maintenance_mutex_);
  synchronize_epoch();
  addr = CacheStore::line_address(addr);
  for (uint32_t i = 0; i < num_lines; ++i)
    cache_.invalidate(addr + i * CacheStore::LINE_SIZE, vmid);
}

void L1ScalarCache::invalidate_for_completion(uint32_t vmid, uint64_t cache_identity) {
  assert(vmid != 0 && cache_identity != 0);
  std::lock_guard lock(maintenance_mutex_);
  synchronize_epoch();
  cache_.for_each_valid_line([&](simdojo::CacheTag &tag, uint64_t address, uint8_t *) {
    if (tag.vmid != vmid || tag.cache_identity != cache_identity)
      return;
    assert(!tag.dirty && "L1 K$ is write-through; lines should never be dirty");
    cache_.invalidate(address, vmid);
  });
}

void L1ScalarCache::invalidate_all_lines() { cache_.invalidate_all(); }

void L1ScalarCache::synchronize_epoch() {
  assert(l2_ != nullptr && "L1 scalar cache requires an L2 cache");
  const uint64_t current_epoch = l2_->coherence_domain()->current_epoch();
  if (coherence_epoch_ == current_epoch)
    return;
  invalidate_all_lines();
  coherence_epoch_ = current_epoch;
}

void L1ScalarCache::flush_line(uint64_t addr, uint32_t vmid) {
  simdojo::CacheTag *tag = nullptr;
  if (!cache_.lookup(addr, &tag, vmid))
    return;

  assert(!tag->dirty && "L1 K$ is write-through; lines should never be dirty");
  cache_.invalidate(addr, vmid);
}

VmAccessOutcome L1ScalarCache::load(uint64_t addr, uint32_t num_dwords, uint32_t *dst,
                                    uint32_t vmid, bool allow_private_batch,
                                    Mtype instruction_mtype) {
  std::lock_guard lock(maintenance_mutex_);
  if (num_dwords == 0)
    return VmAccessOutcome::Complete;
  if (num_dwords > UINT32_MAX / sizeof(uint32_t))
    return VmAccessOutcome::Malformed;
  const VmAccessAdmission admission =
      admit_access(addr, num_dwords * sizeof(uint32_t), vmid, VmAccessKind::Read);
  if (!admission)
    return admission.outcome;
  synchronize_epoch();
  RequestMtypeResolver mtypes(gpu_vm_, vmid, mtype_cache_, instruction_mtype);
  for (uint32_t i = 0; i < num_dwords; ++i) {
    uint64_t ea = addr + i * 4;
    uint8_t buf[4]{};
    uint32_t copied = 0;
    while (copied < sizeof(buf)) {
      const uint64_t chunk_addr = ea + copied;
      const uint32_t line_offset = CacheStore::line_offset(chunk_addr);
      const uint32_t chunk =
          std::min<uint32_t>(sizeof(buf) - copied, CacheStore::LINE_SIZE - line_offset);

      const Mtype mtype = mtypes.at(chunk_addr);

      if (mtype == Mtype::UC) {
        flush_line(chunk_addr, vmid);
        if (allow_private_batch && i == 0 && copied == 0 && num_dwords >= 2 && num_dwords <= 16 &&
            (addr & 3) == 0 && num_dwords * 4 <= CacheStore::LINE_SIZE - line_offset &&
            mtypes.cached_private_uc_hint(addr) &&
            l2_->try_read_scalar_ram(addr, dst, num_dwords, vmid))
          return VmAccessOutcome::Complete;
        const VmAccessOutcome outcome = l2_->read(chunk_addr, buf + copied, chunk, Mtype::UC, vmid);
        if (outcome != VmAccessOutcome::Complete)
          return outcome;
      } else if (mtype == Mtype::CC) {
        flush_line(chunk_addr, vmid);
        const VmAccessOutcome outcome = l2_->read(chunk_addr, buf + copied, chunk, Mtype::CC, vmid);
        if (outcome != VmAccessOutcome::Complete)
          return outcome;
      } else {
        const VmAccessAdmission chunk_admission =
            admit_access(chunk_addr, chunk, vmid, VmAccessKind::Read);
        if (!chunk_admission)
          return chunk_admission.outcome;
        const VmAccessAdmission line_admission =
            admit_access(CacheStore::line_address(chunk_addr), CacheStore::LINE_SIZE, vmid,
                         VmAccessKind::Read, /*report_fault=*/false);
        if (!line_admission || !line_admission.cacheable ||
            line_admission.generation != chunk_admission.generation) {
          flush_line(chunk_addr, vmid);
          const VmAccessOutcome outcome = l2_->read(chunk_addr, buf + copied, chunk, mtype, vmid);
          if (outcome != VmAccessOutcome::Complete)
            return outcome;
          copied += chunk;
          continue;
        }
        simdojo::CacheTag *resident = nullptr;
        if (cache_.lookup(chunk_addr, &resident, vmid) &&
            resident->admission_generation == chunk_admission.generation &&
            resident->cache_identity == current_cache_identity(vmid) &&
            CacheStore::bytes_valid(*resident, line_offset, chunk)) {
          cache_.read_line(chunk_addr, buf + copied, line_offset, chunk, vmid);
        } else if (!l2_->can_fetch_range(CacheStore::line_address(chunk_addr),
                                         CacheStore::LINE_SIZE, vmid)) {
          const uint64_t cache_identity = current_cache_identity(vmid);
          const VmAccessOutcome outcome = l2_->read(chunk_addr, buf + copied, chunk, mtype, vmid);
          if (outcome != VmAccessOutcome::Complete)
            return outcome;
          cache_partial_bytes(chunk_addr, buf + copied, chunk, vmid, chunk_admission.generation,
                              cache_identity);
          copied += chunk;
          continue;
        } else {
          const VmAccessOutcome outcome = ensure_line(chunk_addr, vmid, chunk_admission.generation);
          if (outcome != VmAccessOutcome::Complete)
            return outcome;
          cache_.read_line(chunk_addr, buf + copied, line_offset, chunk, vmid);
        }
      }
      copied += chunk;
    }
    std::memcpy(&dst[i], buf, 4);
  }
  return VmAccessOutcome::Complete;
}

VmAccessOutcome L1ScalarCache::load_bytes(uint64_t addr, uint32_t num_bytes, uint8_t *dst,
                                          uint32_t vmid, Mtype instruction_mtype) {
  std::lock_guard lock(maintenance_mutex_);
  if (num_bytes == 0)
    return VmAccessOutcome::Complete;
  const VmAccessAdmission admission = admit_access(addr, num_bytes, vmid, VmAccessKind::Read);
  if (!admission)
    return admission.outcome;
  synchronize_epoch();
  RequestMtypeResolver mtypes(gpu_vm_, vmid, mtype_cache_, instruction_mtype);
  uint32_t copied = 0;
  while (copied < num_bytes) {
    uint64_t ea = addr + copied;
    uint32_t line_offset = CacheStore::line_offset(ea);
    uint32_t chunk = std::min(num_bytes - copied, CacheStore::LINE_SIZE - line_offset);

    const Mtype mtype = mtypes.at(ea);

    if (mtype == Mtype::UC) {
      flush_line(ea, vmid);
      const VmAccessOutcome outcome = l2_->read(ea, dst + copied, chunk, Mtype::UC, vmid);
      if (outcome != VmAccessOutcome::Complete)
        return outcome;
    } else if (mtype == Mtype::CC) {
      flush_line(ea, vmid);
      const VmAccessOutcome outcome = l2_->read(ea, dst + copied, chunk, Mtype::CC, vmid);
      if (outcome != VmAccessOutcome::Complete)
        return outcome;
    } else {
      const VmAccessAdmission chunk_admission = admit_access(ea, chunk, vmid, VmAccessKind::Read);
      if (!chunk_admission)
        return chunk_admission.outcome;
      const VmAccessAdmission line_admission =
          admit_access(CacheStore::line_address(ea), CacheStore::LINE_SIZE, vmid,
                       VmAccessKind::Read, /*report_fault=*/false);
      if (!line_admission || !line_admission.cacheable ||
          line_admission.generation != chunk_admission.generation) {
        flush_line(ea, vmid);
        const VmAccessOutcome outcome = l2_->read(ea, dst + copied, chunk, mtype, vmid);
        if (outcome != VmAccessOutcome::Complete)
          return outcome;
        copied += chunk;
        continue;
      }
      simdojo::CacheTag *resident = nullptr;
      if (cache_.lookup(ea, &resident, vmid) &&
          resident->admission_generation == chunk_admission.generation &&
          resident->cache_identity == current_cache_identity(vmid) &&
          CacheStore::bytes_valid(*resident, line_offset, chunk)) {
        cache_.read_line(ea, dst + copied, line_offset, chunk, vmid);
      } else if (!l2_->can_fetch_range(CacheStore::line_address(ea), CacheStore::LINE_SIZE, vmid)) {
        const uint64_t cache_identity = current_cache_identity(vmid);
        const VmAccessOutcome outcome = l2_->read(ea, dst + copied, chunk, mtype, vmid);
        if (outcome != VmAccessOutcome::Complete)
          return outcome;
        cache_partial_bytes(ea, dst + copied, chunk, vmid, chunk_admission.generation,
                            cache_identity);
        copied += chunk;
        continue;
      } else {
        const VmAccessOutcome outcome = ensure_line(ea, vmid, chunk_admission.generation);
        if (outcome != VmAccessOutcome::Complete)
          return outcome;
        cache_.read_line(ea, dst + copied, line_offset, chunk, vmid);
      }
    }
    copied += chunk;
  }
  return VmAccessOutcome::Complete;
}

} // namespace amdgpu
} // namespace rocjitsu
