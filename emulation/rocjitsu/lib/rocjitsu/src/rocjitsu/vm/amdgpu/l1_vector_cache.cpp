// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/l1_vector_cache.h"

#include "rocjitsu/base/rj_compiler.h"
#include "rocjitsu/vm/amdgpu/device_cache_coherence.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/request_mtype_resolver.h"
#include "rocjitsu/vm/amdgpu/translated_local_cache_policy.h"
#include "util/log.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cstring>
#include <format>

namespace rocjitsu {
namespace amdgpu {
namespace {

template <typename F>
uint32_t for_each_coalesced_lane_run(const uint64_t *addrs, uint64_t lane_mask, uint32_t wf_size,
                                     uint32_t stride, F &&fn) {
  uint32_t run_count = 0;
  uint64_t remaining = lane_mask;
  while (remaining) {
    const uint32_t first_lane = std::countr_zero(remaining);
    remaining &= ~(uint64_t{1} << first_lane);

    uint32_t last_lane = first_lane;
    while (last_lane + 1 < wf_size) {
      const uint32_t next_lane = last_lane + 1;
      const uint64_t next_bit = uint64_t{1} << next_lane;
      if (!(remaining & next_bit) || addrs[next_lane] != addrs[last_lane] + stride)
        break;
      remaining &= ~next_bit;
      last_lane = next_lane;
    }

    fn(first_lane, last_lane - first_lane + 1);
    ++run_count;
  }
  return run_count;
}

bool all_elements_use_lane_mask(std::span<const uint64_t> element_lane_masks, uint64_t lane_mask,
                                uint32_t num_elems) {
  if (element_lane_masks.empty())
    return true;
  assert(element_lane_masks.size() == num_elems);
  (void)num_elems;
  return std::ranges::all_of(element_lane_masks,
                             [lane_mask](uint64_t mask) { return mask == lane_mask; });
}

uint64_t fully_valid_lane_mask(std::span<const uint64_t> element_lane_masks, uint64_t lane_mask) {
  uint64_t full_lane_mask = lane_mask;
  for (uint64_t element_mask : element_lane_masks)
    full_lane_mask &= element_mask;
  return full_lane_mask;
}

} // namespace

L1VectorCache::L1VectorCache(L2Cache *l2)
    : l2_(l2), registered_domain_(l2_ ? l2_->coherence_domain() : nullptr),
      coherence_epoch_(registered_domain_ ? registered_domain_->current_epoch() : 0) {
  if (registered_domain_)
    registered_domain_->register_l1_vector_cache(this);
}

L1VectorCache::~L1VectorCache() {
  if (registered_domain_)
    registered_domain_->unregister_l1_vector_cache(this);
}

void L1VectorCache::set_l2(L2Cache *l2) {
  std::shared_ptr<DeviceCacheCoherence> next_domain = l2 ? l2->coherence_domain() : nullptr;
  if (next_domain != registered_domain_) {
    if (next_domain)
      next_domain->register_l1_vector_cache(this);
    if (registered_domain_)
      registered_domain_->unregister_l1_vector_cache(this);
  }
  std::lock_guard lock(maintenance_mutex_);
  invalidate_all_lines();
  l2_ = l2;
  registered_domain_ = std::move(next_domain);
  coherence_epoch_ = registered_domain_ ? registered_domain_->current_epoch() : 0;
}

void L1VectorCache::set_gpu_vm(GpuVm *gpu_vm) {
  std::lock_guard lock(maintenance_mutex_);
  invalidate_all_lines();
  gpu_vm_ = gpu_vm;
  mtype_cache_.clear();
}

VmAccessOutcome L1VectorCache::load_translated_local(const GpuVmAccess &access, uint64_t address,
                                                     std::span<uint8_t> bytes,
                                                     Mtype instruction_mtype, bool non_temporal,
                                                     bool request_l1_bypass) {
  const uint32_t offset = CacheStore::line_offset(address);
  if (l2_ == nullptr || registered_domain_ == nullptr || gpu_vm_ == nullptr || bytes.empty() ||
      bytes.size() > LINE_SIZE - offset || access.info().legacy_cache_compatible ||
      !access.supports_translated_local_cache() || access.cache_identity() == 0)
    return VmAccessOutcome::Malformed;

  MemoryIngressAdmission transport = access.admit_pt_rewalk();
  if (transport.outcome != VmAccessOutcome::Complete)
    return transport.outcome;

  // Fabric maintenance takes this domain exclusively through its backing
  // effect. The access lease also orders an HDP flush or local VRAM write with
  // a clean hit, without keeping a VM registry lock across a backing read.
  auto domain_access = registered_domain_->acquire_local_cache_access();
  std::lock_guard lock(maintenance_mutex_);
  synchronize_epoch();
  const uint64_t line_address = CacheStore::line_address(address);
  const auto selected =
      query_translated_local_line(access, line_address, LINE_SIZE, instruction_mtype);
  if (!selected)
    return VmAccessOutcome::Unavailable;
  const VmAccessAdmission admission =
      access.admit_access(line_address, LINE_SIZE, VmAccessKind::Read, false);
  if (!admission)
    return admission.outcome;
  const uint64_t identity = access.cache_identity();
  const uint64_t backing_epoch = registered_domain_->current_local_backing_epoch();
  const uint64_t line_epoch =
      registered_domain_->current_local_line_epoch(selected->backing_address);
  const bool use_l1 = admission.cacheable && !non_temporal && !request_l1_bypass &&
                      selected->effective_type != Mtype::NT;
  const simdojo::CacheTag *resident = cache_.peek(address, 0, identity);
  if (resident != nullptr &&
      (!use_l1 ||
       !translated_local_tag_matches(*resident, *selected, identity, backing_epoch, line_epoch) ||
       resident->admission_generation != admission.generation)) {
    cache_.invalidate(address, 0, identity);
    resident = nullptr;
  }
  if (resident != nullptr && CacheStore::bytes_valid(*resident, offset, bytes.size())) {
    auto current = gpu_vm_->try_acquire_current_namespace(access.cache_namespace().address_space,
                                                          access.cache_namespace());
    if (!current)
      return VmAccessOutcome::Unavailable;
    cache_.lookup(address, nullptr, 0, identity);
    cache_.read_line(address, bytes.data(), offset, bytes.size(), 0, identity);
    return VmAccessOutcome::Complete;
  }

  if (!use_l1)
    return l2_->read_translated_local(access, address, bytes, instruction_mtype, domain_access);

  std::array<uint8_t, LINE_SIZE> line_bytes{};
  const VmAccessOutcome outcome = l2_->read_translated_local(
      access, line_address, std::span<uint8_t>(line_bytes), instruction_mtype, domain_access);
  if (outcome != VmAccessOutcome::Complete)
    return outcome;
  const auto rewalked =
      query_translated_local_line(access, line_address, LINE_SIZE, instruction_mtype);
  if (!rewalked || rewalked->backing_address != selected->backing_address ||
      rewalked->mapping_attributes != selected->mapping_attributes ||
      registered_domain_->current_local_backing_epoch() != backing_epoch ||
      registered_domain_->current_local_line_epoch(selected->backing_address) != line_epoch)
    return VmAccessOutcome::Unavailable;
  auto current = gpu_vm_->try_acquire_current_namespace(access.cache_namespace().address_space,
                                                        access.cache_namespace());
  if (!current)
    return VmAccessOutcome::Unavailable;
  const CacheStore::Allocation allocated =
      cache_.allocate_clean_victim_with_data(line_address, 0, nullptr, nullptr, identity);
  if (allocated.tag != nullptr) {
    allocated.tag->cache_identity = identity;
    allocated.tag->admission_generation = admission.generation;
    allocated.tag->backing_address = selected->backing_address;
    allocated.tag->mapping_attributes = selected->mapping_attributes;
    allocated.tag->local_backing_epoch = backing_epoch;
    allocated.tag->local_line_epoch = line_epoch;
    allocated.tag->coherence = simdojo::CoherenceState::EXCLUSIVE;
    cache_.fill_line(line_address, line_bytes.data(), 0, identity);
    registered_domain_->mark_translated_local_cache_used();
  }
  std::memcpy(bytes.data(), line_bytes.data() + offset, bytes.size());
  return VmAccessOutcome::Complete;
}

bool L1VectorCache::try_read_peer_cached(uint64_t addr, std::span<uint8_t> dst, uint32_t vmid,
                                         uint64_t admission_generation, uint64_t cache_identity) {
  const uint32_t offset = CacheStore::line_offset(addr);
  if (dst.empty() || dst.size() > LINE_SIZE - offset ||
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

bool L1VectorCache::try_fill_peer_cached(uint64_t addr, std::span<const uint8_t> src, uint32_t vmid,
                                         uint64_t admission_generation, uint64_t cache_identity) {
  const uint32_t offset = CacheStore::line_offset(addr);
  if (src.empty() || src.size() > LINE_SIZE - offset ||
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

uint64_t L1VectorCache::current_cache_identity(uint32_t vmid) const {
  if (vmid == 0 || gpu_vm_ == nullptr)
    return 0;
  return gpu_vm_->with_vmid_snapshot(vmid, [](const GpuVmAccess *access) {
    return access != nullptr && access->is_current() ? access->cache_identity() : uint64_t{0};
  });
}

bool L1VectorCache::has_unidentified_overlap(std::span<const DeviceCacheAliasScope> aliases) {
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

void L1VectorCache::invalidate_selected_lines(std::span<const DeviceCacheAliasScope> aliases) {
  cache_.for_each_valid_line([&](simdojo::CacheTag &tag, uint64_t address, uint8_t *) {
    if (!std::ranges::any_of(aliases, [&](const DeviceCacheAliasScope &alias) {
          return alias.domain.coherence.get() == registered_domain_.get() &&
                 alias.vmid == tag.vmid && alias.admission_generation == tag.admission_generation &&
                 alias.cache_identity == tag.cache_identity && address >= alias.address &&
                 address - alias.address < alias.size;
        }))
      return;
    assert(!tag.dirty && "L1 V$ is write-through; lines should never be dirty");
    cache_.invalidate(address, tag.vmid, tag.address_space_identity);
  });
}

VmAccessAdmission L1VectorCache::admit_access(uint64_t addr, uint32_t size, uint32_t vmid,
                                              VmAccessKind kind, bool report_fault) const {
  if (vmid == 0 || gpu_vm_ == nullptr)
    return {};
  return gpu_vm_->with_vmid_snapshot(vmid, [&](const GpuVmAccess *access) {
    return access ? access->admit_access(addr, size, kind, report_fault)
                  : VmAccessAdmission{.outcome = VmAccessOutcome::Faulted};
  });
}

VmAccessOutcome L1VectorCache::ensure_line(uint64_t addr, uint32_t vmid,
                                           uint64_t admission_generation, bool fetch_on_miss) {
  const uint64_t cache_identity = current_cache_identity(vmid);
  simdojo::CacheTag *resident = nullptr;
  if (cache_.lookup(addr, &resident, vmid)) {
    if (resident->admission_generation != admission_generation ||
        resident->cache_identity != cache_identity) {
      assert(!resident->dirty && "L1 V$ is write-through; lines should never be dirty");
      cache_.invalidate(addr, vmid);
      resident = nullptr;
    }
  }
  if (resident != nullptr) {
    if (!fetch_on_miss || CacheStore::bytes_valid(*resident, 0, LINE_SIZE))
      return VmAccessOutcome::Complete;

    uint8_t line_buf[LINE_SIZE];
    const VmAccessOutcome outcome = l2_->fetch_line(CacheStore::line_address(addr), line_buf, vmid);
    if (outcome != VmAccessOutcome::Complete)
      return outcome;
    const VmAccessAdmission current =
        admit_access(CacheStore::line_address(addr), LINE_SIZE, vmid, VmAccessKind::Read,
                     /*report_fault=*/false);
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
  simdojo::CacheTag *allocated = cache_.allocate(addr, vmid, &evicted, nullptr);
  allocated->admission_generation = admission_generation;
  allocated->cache_identity = cache_identity;

  assert(!evicted.dirty && "L1 V$ is write-through; lines should never be dirty");

  // A store supplies only its architectural byte range. Keep all other bytes
  // absent so the store never speculatively reads backing memory.
  if (!fetch_on_miss) {
    CacheStore::clear_valid_bytes(*allocated);
    return VmAccessOutcome::Complete;
  }

  uint8_t line_buf[LINE_SIZE];
  const VmAccessOutcome outcome = l2_->fetch_line(line_addr, line_buf, vmid);
  if (outcome != VmAccessOutcome::Complete) {
    cache_.invalidate(addr, vmid);
    return outcome;
  }
  const VmAccessAdmission current =
      admit_access(line_addr, LINE_SIZE, vmid, VmAccessKind::Read, /*report_fault=*/false);
  if (!current || !current.cacheable || current.generation != admission_generation ||
      current_cache_identity(vmid) != cache_identity) {
    cache_.invalidate(addr, vmid);
    return VmAccessOutcome::Unavailable;
  }
  cache_.fill_line(addr, line_buf, vmid);
  return VmAccessOutcome::Complete;
}

void L1VectorCache::cache_partial_bytes(uint64_t addr, const uint8_t *src, uint32_t size,
                                        uint32_t vmid, uint64_t admission_generation,
                                        uint64_t cache_identity) {
  const VmAccessAdmission current =
      admit_access(CacheStore::line_address(addr), LINE_SIZE, vmid, VmAccessKind::Read,
                   /*report_fault=*/false);
  if (!current || !current.cacheable || current.generation != admission_generation ||
      current_cache_identity(vmid) != cache_identity)
    return;
  simdojo::CacheTag *tag = nullptr;
  if (cache_.lookup(addr, &tag, vmid) && (tag->admission_generation != admission_generation ||
                                          tag->cache_identity != cache_identity)) {
    assert(!tag->dirty && "L1 V$ is write-through; lines should never be dirty");
    cache_.invalidate(addr, vmid);
    tag = nullptr;
  }
  if (tag == nullptr) {
    simdojo::CacheTag evicted;
    tag = cache_.allocate(addr, vmid, &evicted, nullptr);
    assert(!evicted.dirty && "L1 V$ is write-through; lines should never be dirty");
    tag->admission_generation = admission_generation;
    tag->cache_identity = cache_identity;
    CacheStore::clear_valid_bytes(*tag);
  }
  cache_.write_line(addr, src, CacheStore::line_offset(addr), size, vmid);
}

// Per-line CC invalidation is sufficient: the CP serializes dispatch N's cache
// management before dispatch N+1 begins execution, so no blanket invalidation
// at dispatch boundaries is needed.
VmAccessOutcome L1VectorCache::read_bytes(uint64_t addr, uint8_t *dst, uint32_t size,
                                          bool non_temporal, bool request_l1_bypass, uint32_t vmid,
                                          RequestMtypeResolver &mtypes) {
  const VmAccessAdmission admission = admit_access(addr, size, vmid, VmAccessKind::Read);
  if (!admission)
    return admission.outcome;
  util::Logger::cp([&](auto &os) {
    const Mtype effective = mtypes.at(addr);
    static thread_local uint64_t mtype_counts[5] = {};
    static thread_local uint64_t total = 0;
    ++mtype_counts[static_cast<int>(effective)];
    ++total;
    if ((total & (total - 1)) == 0 && total >= 1024) {
      os << std::format("L1V_READ_MTYPE_STATS total={} UC={} CC={} RW={} WB={} NT={} "
                        "last: addr={:#x} inst={} eff={} vmid={}",
                        total, mtype_counts[0], mtype_counts[1], mtype_counts[2], mtype_counts[3],
                        mtype_counts[4], addr, static_cast<int>(mtypes.fallback()),
                        static_cast<int>(effective), vmid);
    }
  });

  uint32_t copied = 0;
  while (copied < size) {
    const uint64_t ea = addr + copied;
    const uint32_t line_offset = CacheStore::line_offset(ea);
    const uint32_t chunk = std::min(size - copied, LINE_SIZE - line_offset);
    const Mtype chunk_mtype = mtypes.at(ea);
    const VmAccessAdmission chunk_admission = admit_access(ea, chunk, vmid, VmAccessKind::Read);
    if (!chunk_admission)
      return chunk_admission.outcome;
    const VmAccessAdmission line_admission =
        admit_access(CacheStore::line_address(ea), LINE_SIZE, vmid, VmAccessKind::Read,
                     /*report_fault=*/false);
    const bool cache_line = line_admission && line_admission.cacheable &&
                            line_admission.generation == chunk_admission.generation;

    if (!cache_line) {
      cache_.invalidate(ea, vmid);
      const VmAccessOutcome outcome = l2_->read(ea, dst + copied, chunk, chunk_mtype, vmid);
      if (outcome != VmAccessOutcome::Complete)
        return outcome;
      copied += chunk;
      continue;
    }

    if (chunk_mtype == Mtype::UC || non_temporal || request_l1_bypass) {
      cache_.invalidate(ea, vmid);
      const VmAccessOutcome outcome = l2_->read(ea, dst + copied, chunk, chunk_mtype, vmid);
      if (outcome != VmAccessOutcome::Complete)
        return outcome;
      copied += chunk;
      continue;
    }

    if (chunk_mtype == Mtype::CC) {
      cache_.invalidate(ea, vmid);
      const VmAccessOutcome outcome = l2_->read(ea, dst + copied, chunk, chunk_mtype, vmid);
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
      copied += chunk;
      continue;
    }

    if (!l2_->can_fetch_range(CacheStore::line_address(ea), LINE_SIZE, vmid)) {
      const uint64_t cache_identity = current_cache_identity(vmid);
      const VmAccessOutcome outcome = l2_->read(ea, dst + copied, chunk, chunk_mtype, vmid);
      if (outcome != VmAccessOutcome::Complete)
        return outcome;
      cache_partial_bytes(ea, dst + copied, chunk, vmid, chunk_admission.generation,
                          cache_identity);
      copied += chunk;
      continue;
    }

    const VmAccessOutcome outcome = ensure_line(ea, vmid, chunk_admission.generation);
    if (outcome != VmAccessOutcome::Complete)
      return outcome;
    cache_.read_line(ea, dst + copied, line_offset, chunk, vmid);
    copied += chunk;
  }
  return VmAccessOutcome::Complete;
}

VmAccessOutcome L1VectorCache::write_bytes(uint64_t addr, const uint8_t *src, uint32_t size,
                                           bool non_temporal, uint32_t vmid,
                                           RequestMtypeResolver &mtypes) {
  const VmAccessAdmission admission = admit_access(addr, size, vmid, VmAccessKind::Write);
  if (!admission)
    return admission.outcome;
  util::Logger::vm([&](auto &os) {
    if (addr >= 0x4d00c00000ULL && addr < 0x4d00c00100ULL) {
      const Mtype effective = mtypes.at(addr);
      uint32_t val = 0;
      if (size >= 4)
        std::memcpy(&val, src, 4);
      else if (size >= 2)
        std::memcpy(&val, src, size);
      else
        val = src[0];
      static thread_local uint32_t tw = 0;
      if (++tw <= 20)
        os << std::format("L1_WRITE @{:#x} size={} val={:#x} mtype={}", addr, size, val,
                          static_cast<int>(effective));
    }
  });

  uint32_t copied = 0;
  while (copied < size) {
    const uint64_t ea = addr + copied;
    const uint32_t line_offset = CacheStore::line_offset(ea);
    const uint32_t chunk = std::min(size - copied, LINE_SIZE - line_offset);
    const Mtype chunk_mtype = mtypes.at(ea);
    const VmAccessAdmission chunk_admission = admit_access(ea, chunk, vmid, VmAccessKind::Write);
    if (!chunk_admission)
      return chunk_admission.outcome;
    const VmAccessAdmission line_admission =
        admit_access(CacheStore::line_address(ea), LINE_SIZE, vmid, VmAccessKind::Write,
                     /*report_fault=*/false);
    const bool cache_line = line_admission && line_admission.cacheable &&
                            line_admission.generation == chunk_admission.generation;

    if (!cache_line) {
      cache_.invalidate(ea, vmid);
      const VmAccessOutcome outcome = l2_->write(ea, src + copied, chunk, chunk_mtype, vmid);
      if (outcome != VmAccessOutcome::Complete)
        return outcome;
      copied += chunk;
      continue;
    }

    if (chunk_mtype == Mtype::UC || non_temporal) {
      cache_.invalidate(ea, vmid);
      const VmAccessOutcome outcome = l2_->write(ea, src + copied, chunk, chunk_mtype, vmid);
      if (outcome != VmAccessOutcome::Complete)
        return outcome;
      copied += chunk;
      continue;
    }

    const VmAccessOutcome fill_outcome =
        ensure_line(ea, vmid, chunk_admission.generation, /*fetch_on_miss=*/false);
    if (fill_outcome != VmAccessOutcome::Complete)
      return fill_outcome;

    // Write through to L2 for all cacheable stores. This ensures partial writes
    // from different CUs sharing the same L2 are properly merged at byte
    // granularity via L2::write(), rather than full-line replacement via
    // writeback_line() during L1 eviction/flush.
    const VmAccessOutcome write_outcome = l2_->write(ea, src + copied, chunk, chunk_mtype, vmid);
    if (write_outcome != VmAccessOutcome::Complete)
      return write_outcome;

    simdojo::CacheTag *tag = nullptr;
    cache_.lookup(ea, &tag, vmid);
    assert(tag != nullptr && "ensure_line must guarantee hit");
    const VmAccessAdmission current =
        admit_access(CacheStore::line_address(ea), LINE_SIZE, vmid, VmAccessKind::Write,
                     /*report_fault=*/false);
    if (!current || !current.cacheable || current.generation != chunk_admission.generation ||
        tag->cache_identity != current_cache_identity(vmid)) {
      cache_.invalidate(ea, vmid);
      copied += chunk;
      continue;
    }

    cache_.write_line(ea, src + copied, line_offset, chunk, vmid);

    // L1 line stays clean since L2 has the authoritative copy.
    tag->coherence = (chunk_mtype == Mtype::CC) ? simdojo::CoherenceState::SHARED
                                                : simdojo::CoherenceState::EXCLUSIVE;
    tag->dirty = false;
    copied += chunk;
  }
  return VmAccessOutcome::Complete;
}

VmAccessOutcome L1VectorCache::load(const uint64_t *addrs, uint64_t lane_mask, uint32_t elem_size,
                                    uint32_t num_elems, uint8_t *dst, Mtype mtype,
                                    bool non_temporal, bool request_l1_bypass, uint32_t wf_size,
                                    uint32_t vmid, uint32_t addr_stride, uint32_t addr_base_offset,
                                    std::span<const uint64_t> element_lane_masks,
                                    uint32_t swizzle_unit) {
  std::lock_guard lock(maintenance_mutex_);
  synchronize_epoch();
  RequestMtypeResolver mtypes(gpu_vm_, vmid, mtype_cache_, mtype);
  uint32_t stride = num_elems * elem_size;
  // Scratch and buffer swizzling: consecutive units of a lane sit
  // addr_stride bytes apart. Addresses are materialized per element, so this honours
  // per-element lane validity: an element a lane is not valid for is skipped
  // rather than strided over. With no element masks this walks exactly the
  // lanes and bytes the uniform path would. addr_base_offset identifies the
  // low bits added after swizzling so they do not move the logical swizzle-unit boundary.
  if (addr_stride != 0) {
    assert((swizzle_unit == 4 || swizzle_unit == 16) && addr_base_offset < swizzle_unit);
    const uint32_t astride = addr_stride;
    for (uint32_t elem = 0; elem < num_elems; ++elem) {
      uint64_t mask =
          element_lane_masks.empty() ? lane_mask : (element_lane_masks[elem] & lane_mask);
      while (mask) {
        const uint32_t lane = std::countr_zero(mask);
        mask &= mask - 1;
        const uint64_t base = addrs[lane];
        const uint32_t first_byte_in_unit =
            static_cast<uint32_t>((base - addr_base_offset) % swizzle_unit);
        uint32_t copied = elem * elem_size;
        const uint32_t elem_end = copied + elem_size;
        while (copied < elem_end) {
          const uint32_t logical_byte = first_byte_in_unit + copied;
          const uint32_t byte_in_unit = logical_byte % swizzle_unit;
          const uint32_t chunk = std::min(elem_end - copied, swizzle_unit - byte_in_unit);
          const uint64_t ea =
              base - first_byte_in_unit + logical_byte / swizzle_unit * astride + byte_in_unit;
          const VmAccessOutcome outcome = read_bytes(ea, dst + lane * stride + copied, chunk,
                                                     non_temporal, request_l1_bypass, vmid, mtypes);
          if (outcome != VmAccessOutcome::Complete)
            return outcome;
          copied += chunk;
        }
      }
    }
    return VmAccessOutcome::Complete;
  }
  if (!all_elements_use_lane_mask(element_lane_masks, lane_mask, num_elems)) {
    uint64_t full_lane_mask = fully_valid_lane_mask(element_lane_masks, lane_mask);
    VmAccessOutcome outcome = VmAccessOutcome::Complete;
    for_each_coalesced_lane_run(
        addrs, full_lane_mask, wf_size, stride, [&](uint32_t first_lane, uint32_t run_lanes) {
          if (outcome == VmAccessOutcome::Complete)
            outcome = read_bytes(addrs[first_lane], dst + first_lane * stride, run_lanes * stride,
                                 non_temporal, request_l1_bypass, vmid, mtypes);
        });
    if (outcome != VmAccessOutcome::Complete)
      return outcome;
    for (uint32_t elem = 0; elem < num_elems; ++elem) {
      uint64_t mask = element_lane_masks[elem] & lane_mask & ~full_lane_mask;
      while (mask) {
        const uint32_t lane = std::countr_zero(mask);
        mask &= ~(uint64_t{1} << lane);
        outcome = read_bytes(addrs[lane] + static_cast<uint64_t>(elem) * elem_size,
                             dst + lane * stride + elem * elem_size, elem_size, non_temporal,
                             request_l1_bypass, vmid, mtypes);
        if (outcome != VmAccessOutcome::Complete)
          return outcome;
      }
    }
    return VmAccessOutcome::Complete;
  }
  VmAccessOutcome outcome = VmAccessOutcome::Complete;
  for_each_coalesced_lane_run(
      addrs, lane_mask, wf_size, stride, [&](uint32_t first_lane, uint32_t run_lanes) {
        if (outcome == VmAccessOutcome::Complete)
          outcome = read_bytes(addrs[first_lane], dst + first_lane * stride, run_lanes * stride,
                               non_temporal, request_l1_bypass, vmid, mtypes);
      });
  return outcome;
}

// Keep bounded descriptor storage and swizzled-loop live ranges out of plain stores.
RJ_NOINLINE VmAccessOutcome L1VectorCache::store_swizzled(
    const uint64_t *addrs, uint64_t lane_mask, uint32_t elem_size, uint32_t num_elems,
    const uint8_t *src, Mtype mtype, bool non_temporal, uint32_t vmid, uint32_t addr_stride,
    uint32_t addr_base_offset, std::span<const uint64_t> element_lane_masks, uint32_t swizzle_unit,
    bool allow_private_batch, RequestMtypeResolver &mtypes) {
  const uint32_t stride = num_elems * elem_size;
  assert((swizzle_unit == 4 || swizzle_unit == 16) && addr_base_offset < swizzle_unit);
  const uint32_t astride = addr_stride;
  const bool batch_dwords =
      allow_private_batch && elem_size == sizeof(uint32_t) && (addr_base_offset & 3) == 0;
  uint64_t refused_page = UINT64_MAX;
  for (uint32_t elem = 0; elem < num_elems; ++elem) {
    uint64_t mask = element_lane_masks.empty() ? lane_mask : (element_lane_masks[elem] & lane_mask);
    store_l2_writes_ += std::popcount(mask);
    uint64_t attempted_line = UINT64_MAX;
    while (mask) {
      const uint32_t lane = std::countr_zero(mask);
      mask &= mask - 1;
      const uint64_t base = addrs[lane];
      const uint32_t first_byte_in_unit =
          static_cast<uint32_t>((base - addr_base_offset) % swizzle_unit);
      uint32_t copied = elem * elem_size;
      const uint32_t elem_end = copied + elem_size;
      while (copied < elem_end) {
        const uint32_t logical_byte = first_byte_in_unit + copied;
        const uint32_t byte_in_unit = logical_byte % swizzle_unit;
        const uint32_t chunk = std::min(elem_end - copied, swizzle_unit - byte_in_unit);
        const uint64_t ea =
            base - first_byte_in_unit + logical_byte / swizzle_unit * astride + byte_in_unit;
        const VmAccessOutcome outcome =
            write_bytes(ea, src + lane * stride + copied, chunk, non_temporal, vmid, mtypes);
        if (outcome != VmAccessOutcome::Complete)
          return outcome;
        copied += chunk;
        // Keep the first request ordinary. It owns all existing allocation,
        // dirty-prefix and fault behavior and warms the copied policy hint.
        const uint64_t line = ea >> LINE_SIZE_BITS;
        if (!batch_dwords || chunk != sizeof(uint32_t) || (ea & 3) || !mask ||
            attempted_line == line || refused_page == (ea >> 12))
          continue;
        attempted_line = line;
        const auto policy = mtypes.cached_private_ram_mtype(ea);
        if (!policy) {
          // False is always safe to retain for this instruction; never refresh
          // a policy or call an unknown translator just to attempt batching.
          refused_page = ea >> 12;
          continue;
        }
        const auto *resident = cache_.peek(ea, vmid);
        const VmAccessAdmission admission =
            admit_access(ea, sizeof(uint32_t), vmid, VmAccessKind::Write,
                         /*report_fault=*/false);
        if (!admission || !admission.cacheable ||
            (resident && (resident->admission_generation != admission.generation ||
                          resident->cache_identity != current_cache_identity(vmid))))
          continue;
        if (*policy == Mtype::UC || non_temporal) {
          if (resident)
            continue;
        } else if (!resident || resident->dirty) {
          continue;
        }
        std::array<VmRamDwordStore, VmRamDwordStore::kMaxBatch> stores;
        size_t count = 0;
        uint64_t remaining = mask;
        while (remaining && count < stores.size()) {
          const uint32_t next_lane = std::countr_zero(remaining);
          const uint64_t next_base = addrs[next_lane];
          const uint32_t first_byte =
              static_cast<uint32_t>((next_base - addr_base_offset) % swizzle_unit);
          const uint32_t logical = first_byte + elem * elem_size;
          const uint64_t address =
              next_base - first_byte + logical / swizzle_unit * astride + logical % swizzle_unit;
          if ((next_base & 3) || (address >> LINE_SIZE_BITS) != line)
            break;
          stores[count++] = {address, src + next_lane * stride + elem * elem_size};
          remaining &= remaining - 1;
        }
        const std::span<const VmRamDwordStore> group(stores.data(), count);
        if (count < 2)
          continue;
        if (!l2_->try_write_private_dwords(group, mtype, *policy, vmid)) {
          // A refusal grants no access and changes no guest state. Retaining it
          // only selects ordinary requests for later elements on this page.
          refused_page = ea >> 12;
          continue;
        }
        if (*policy != Mtype::UC && !non_temporal) {
          for (const auto &store : group) {
            const VmAccessAdmission store_admission =
                admit_access(store.address, sizeof(uint32_t), vmid, VmAccessKind::Write,
                             /*report_fault=*/false);
            simdojo::CacheTag *tag = nullptr;
            if (!store_admission || !store_admission.cacheable ||
                !cache_.lookup(store.address, &tag, vmid) ||
                tag->admission_generation != store_admission.generation ||
                tag->cache_identity != current_cache_identity(vmid)) {
              cache_.invalidate(store.address, vmid);
              continue;
            }
            cache_.write_line(store.address, store.source, CacheStore::line_offset(store.address),
                              sizeof(uint32_t), vmid);
            tag->coherence = *policy == Mtype::CC ? simdojo::CoherenceState::SHARED
                                                  : simdojo::CoherenceState::EXCLUSIVE;
          }
        }
        mask = remaining;
      }
    }
  }
  return VmAccessOutcome::Complete;
}

VmAccessOutcome L1VectorCache::store(const uint64_t *addrs, uint64_t lane_mask, uint32_t elem_size,
                                     uint32_t num_elems, const uint8_t *src, Mtype mtype,
                                     bool non_temporal, uint32_t wf_size, uint32_t vmid,
                                     uint32_t addr_stride, uint32_t addr_base_offset,
                                     std::span<const uint64_t> element_lane_masks,
                                     uint32_t swizzle_unit, bool allow_private_batch) {
  std::lock_guard lock(maintenance_mutex_);
  synchronize_epoch();
  RequestMtypeResolver mtypes(gpu_vm_, vmid, mtype_cache_, mtype);
  uint32_t stride = num_elems * elem_size;
  const uint32_t active_lanes = std::popcount(lane_mask);
  ++store_count_;
  if (active_lanes > 0)
    ++store_active_count_;
  // Scratch and buffer swizzling: consecutive units of a lane sit
  // addr_stride bytes apart. Addresses are materialized per element, so this honours
  // per-element lane validity: an element a lane is not valid for is skipped
  // rather than strided over. With no element masks this walks exactly the
  // lanes and bytes the uniform path would. addr_base_offset identifies the
  // low bits added after swizzling so they do not move the logical swizzle-unit boundary.
  if (addr_stride != 0)
    return store_swizzled(addrs, lane_mask, elem_size, num_elems, src, mtype, non_temporal, vmid,
                          addr_stride, addr_base_offset, element_lane_masks, swizzle_unit,
                          allow_private_batch, mtypes);
  if (!all_elements_use_lane_mask(element_lane_masks, lane_mask, num_elems)) {
    uint64_t full_lane_mask = fully_valid_lane_mask(element_lane_masks, lane_mask);
    VmAccessOutcome outcome = VmAccessOutcome::Complete;
    store_l2_writes_ += for_each_coalesced_lane_run(
        addrs, full_lane_mask, wf_size, stride, [&](uint32_t first_lane, uint32_t run_lanes) {
          if (outcome == VmAccessOutcome::Complete)
            outcome = write_bytes(addrs[first_lane], src + first_lane * stride, run_lanes * stride,
                                  non_temporal, vmid, mtypes);
        });
    if (outcome != VmAccessOutcome::Complete)
      return outcome;
    for (uint32_t elem = 0; elem < num_elems; ++elem) {
      uint64_t mask = element_lane_masks[elem] & lane_mask & ~full_lane_mask;
      store_l2_writes_ += std::popcount(mask);
      while (mask) {
        const uint32_t lane = std::countr_zero(mask);
        mask &= ~(uint64_t{1} << lane);
        outcome = write_bytes(addrs[lane] + static_cast<uint64_t>(elem) * elem_size,
                              src + lane * stride + elem * elem_size, elem_size, non_temporal, vmid,
                              mtypes);
        if (outcome != VmAccessOutcome::Complete)
          return outcome;
      }
    }
    return VmAccessOutcome::Complete;
  }
  VmAccessOutcome outcome = VmAccessOutcome::Complete;
  store_l2_writes_ += for_each_coalesced_lane_run(
      addrs, lane_mask, wf_size, stride, [&](uint32_t first_lane, uint32_t run_lanes) {
        if (outcome == VmAccessOutcome::Complete)
          outcome = write_bytes(addrs[first_lane], src + first_lane * stride, run_lanes * stride,
                                non_temporal, vmid, mtypes);
      });
  return outcome;
}

void L1VectorCache::invalidate(uint64_t addr, uint32_t vmid) {
  std::lock_guard lock(maintenance_mutex_);
  synchronize_epoch();
  cache_.invalidate(addr, vmid);
}

void L1VectorCache::invalidate_all() {
  std::lock_guard lock(maintenance_mutex_);
  synchronize_epoch();
  invalidate_all_lines();
}

void L1VectorCache::flush_all() {
  std::lock_guard lock(maintenance_mutex_);
  synchronize_epoch();
  invalidate_all_lines();
}

void L1VectorCache::invalidate_for_completion(uint32_t vmid, uint64_t cache_identity) {
  assert(vmid != 0 && cache_identity != 0);
  std::lock_guard lock(maintenance_mutex_);
  synchronize_epoch();
  cache_.for_each_valid_line([&](simdojo::CacheTag &tag, uint64_t address, uint8_t *) {
    if (!cache_tag_matches_completion_access(tag, vmid, cache_identity))
      return;
    assert(!tag.dirty && "L1 V$ is write-through; lines should never be dirty");
    cache_.invalidate(address, tag.vmid, tag.address_space_identity);
  });
}

void L1VectorCache::invalidate_all_lines() { cache_.invalidate_all(); }

void L1VectorCache::synchronize_epoch() {
  assert(l2_ != nullptr && "L1 vector cache requires an L2 cache");
  const uint64_t current_epoch = l2_->coherence_domain()->current_epoch();
  if (coherence_epoch_ == current_epoch)
    return;
  invalidate_all_lines();
  coherence_epoch_ = current_epoch;
}

} // namespace amdgpu
} // namespace rocjitsu
