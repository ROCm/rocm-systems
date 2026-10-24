// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/translated_local_cache_policy.h"
#include "rocjitsu/vm/prepared_canonical_writeback.h"
#include "util/except.h"
#include "util/log.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <string_view>

namespace rocjitsu {
namespace amdgpu {

L2Cache::L2Cache(std::string name, std::shared_ptr<DeviceCacheCoherence> coherence)
    : simdojo::Component(std::move(name)), coherence_(std::move(coherence)) {
  if (!coherence_)
    throw util::ConfigError("L2 cache requires a coherence domain");
  req_port_ = add_port(std::make_unique<simdojo::Port>("req", 0, this, simdojo::PortDirection::OUT,
                                                       simdojo::PortProtocol::MEMORY));
  coherence_epoch_ = coherence_->current_epoch();
  coherence_->register_l2_cache(this);
}

L2Cache::~L2Cache() { coherence_->unregister_l2_cache(this); }

void L2Cache::set_coherence_domain(std::shared_ptr<DeviceCacheCoherence> coherence) {
  if (!coherence)
    throw util::ConfigError("L2 cache requires a coherence domain");
  if (coherence == coherence_)
    return;
  coherence_->unregister_l2_cache(this);
  {
    std::unique_lock maintenance_lock(maintenance_mutex_);
    cache_.invalidate_all();
    clear_all_dirty_bytes();
  }
  coherence_ = std::move(coherence);
  coherence_epoch_ = coherence_->current_epoch();
  coherence_->register_l2_cache(this);
}

std::shared_lock<L2Cache::MaintenanceMutex> L2Cache::acquire_cache_access() {
  std::shared_lock access_lock(maintenance_mutex_);
  if (coherence_epoch_ == coherence_->current_epoch())
    return access_lock;

  access_lock.unlock();
  // Only stale readers serialize here. Recheck under shared admission so peers
  // reuse completed reconciliation instead of each closing the reader gate.
  std::lock_guard reconcile_lock(epoch_reconcile_mutex_);
  for (;;) {
    access_lock.lock();
    if (coherence_epoch_ == coherence_->current_epoch())
      return access_lock;

    access_lock.unlock();
    std::unique_lock maintenance_lock(maintenance_mutex_);
    synchronize_epoch_locked();
  }
}

std::unique_lock<L2Cache::MaintenanceMutex> L2Cache::acquire_cache_maintenance() {
  std::unique_lock maintenance_lock(maintenance_mutex_);
  synchronize_epoch_locked();
  return maintenance_lock;
}

void L2Cache::synchronize_epoch_locked() {
  const uint64_t current_epoch = coherence_->current_epoch();
  if (coherence_epoch_ == current_epoch)
    return;
  cache_.invalidate_all();
  clear_all_dirty_bytes();
  coherence_epoch_ = current_epoch;
}

void L2Cache::mark_dirty_bytes(uint64_t line_addr, uint32_t offset, uint32_t size, uint32_t vmid) {
  assert(offset < LINE_SIZE && size != 0 && size <= LINE_SIZE - offset);
  std::lock_guard lock(dirty_bytes_mutex_);
  DirtyMask &mask = dirty_bytes_[{vmid, line_addr}];
  for (uint32_t byte = offset; byte < offset + size; ++byte)
    mask[byte / 64] |= uint64_t{1} << (byte % 64);
  has_dirty_lines_.store(true, std::memory_order_relaxed);
}

bool L2Cache::clear_dirty_bytes(uint64_t line_addr, uint32_t offset, uint32_t size, uint32_t vmid) {
  assert(offset < LINE_SIZE && size != 0 && size <= LINE_SIZE - offset);
  // The caller holds the set lock, which orders any earlier dirty mask for
  // this line. Concurrent changes in other sets cannot make this line dirty.
  if (!has_dirty_lines_.load(std::memory_order_relaxed))
    return false;
  std::lock_guard lock(dirty_bytes_mutex_);
  const std::pair<uint32_t, uint64_t> key{vmid, line_addr};
  const std::map<std::pair<uint32_t, uint64_t>, DirtyMask>::iterator position =
      dirty_bytes_.find(key);
  if (position == dirty_bytes_.end())
    return false;

  for (uint32_t byte = offset; byte < offset + size; ++byte)
    position->second[byte / 64] &= ~(uint64_t{1} << (byte % 64));

  const bool line_remains_dirty =
      std::ranges::any_of(position->second, [](uint64_t word) { return word != 0; });
  if (!line_remains_dirty)
    dirty_bytes_.erase(position);
  has_dirty_lines_.store(!dirty_bytes_.empty(), std::memory_order_relaxed);
  return line_remains_dirty;
}

VmAccessOutcome L2Cache::publish_dirty_bytes(uint64_t line_addr, const uint8_t *data, uint32_t vmid,
                                             uint32_t discard_offset, uint32_t discard_size) {
  assert(discard_offset <= LINE_SIZE && discard_size <= LINE_SIZE - discard_offset);
  // An invalidation may discard part of this dirty line. Refuse publication
  // before changing even those discarded mask bits, so a later selected
  // canonical maintenance can still see the original authoritative bytes.
  if (direct_controlled_writeback(line_addr, LINE_SIZE, vmid)) {
    std::lock_guard lock(dirty_bytes_mutex_);
    const auto original = dirty_bytes_.find({vmid, line_addr});
    if (original != dirty_bytes_.end() &&
        dirty_mask_overlaps_controlled(line_addr, vmid, original->second))
      return VmAccessOutcome::Unavailable;
  }
  if (discard_size != 0)
    (void)clear_dirty_bytes(line_addr, discard_offset, discard_size, vmid);

  DirtyMask mask{};
  {
    std::lock_guard lock(dirty_bytes_mutex_);
    const std::map<std::pair<uint32_t, uint64_t>, DirtyMask>::iterator position =
        dirty_bytes_.find({vmid, line_addr});
    if (position == dirty_bytes_.end())
      return VmAccessOutcome::Complete;
    mask = position->second;
  }
  if (dirty_mask_overlaps_controlled(line_addr, vmid, mask))
    return VmAccessOutcome::Unavailable;

  uint32_t offset = 0;
  while (offset < LINE_SIZE) {
    while (offset < LINE_SIZE && (mask[offset / 64] & (uint64_t{1} << (offset % 64))) == 0)
      ++offset;
    const uint32_t start = offset;
    while (offset < LINE_SIZE && (mask[offset / 64] & (uint64_t{1} << (offset % 64))) != 0)
      ++offset;
    if (start != offset) {
      const VmAccessOutcome outcome =
          send_backing(line_addr + start, const_cast<uint8_t *>(data + start), offset - start,
                       simdojo::MessageOp::WRITE, vmid);
      if (outcome != VmAccessOutcome::Complete)
        return outcome;
      (void)clear_dirty_bytes(line_addr, start, offset - start, vmid);
    }
  }
  return VmAccessOutcome::Complete;
}

VmAccessOutcome L2Cache::publish_dirty_bytes_to_legacy_backing(uint64_t line_addr,
                                                               const uint8_t *data, uint32_t vmid) {
  if (legacy_maintenance_memory_ == nullptr || legacy_maintenance_vm_ == nullptr)
    throw util::ConfigError("L2 cache has no legacy maintenance backing");

  std::optional<GpuVmAccess> vm_access;
  if (vmid != 0) {
    vm_access = legacy_maintenance_vm_->snapshot_vmid(vmid);
    if (!vm_access)
      return VmAccessOutcome::Faulted;
  }

  DirtyMask mask{};
  {
    std::lock_guard lock(dirty_bytes_mutex_);
    const std::map<std::pair<uint32_t, uint64_t>, DirtyMask>::iterator position =
        dirty_bytes_.find({vmid, line_addr});
    assert(position != dirty_bytes_.end() && "dirty L2 line must have a byte mask");
    if (position == dirty_bytes_.end())
      return VmAccessOutcome::Complete;
    mask = position->second;
  }
  if (dirty_mask_overlaps_controlled(line_addr, vmid, mask))
    return VmAccessOutcome::Unavailable;

  uint32_t offset = 0;
  while (offset < LINE_SIZE) {
    while (offset < LINE_SIZE && (mask[offset / 64] & (uint64_t{1} << (offset % 64))) == 0)
      ++offset;
    const uint32_t start = offset;
    while (offset < LINE_SIZE && (mask[offset / 64] & (uint64_t{1} << (offset % 64))) != 0)
      ++offset;
    if (start != offset) {
      const std::span<const uint8_t> bytes(data + start, offset - start);
      std::size_t completed_bytes = 0;
      VmAccessOutcome outcome = VmAccessOutcome::Complete;
      if (vmid == 0) {
        legacy_maintenance_memory_->write_block(line_addr + start, bytes);
        completed_bytes = bytes.size();
      } else {
        outcome = vm_access->write(line_addr + start, std::as_bytes(bytes), completed_bytes);
      }
      if (completed_bytes > bytes.size() ||
          (outcome == VmAccessOutcome::Complete && completed_bytes != bytes.size()))
        return VmAccessOutcome::Malformed;
      // Retire the exact prefix known to have reached backing. A later retry
      // starts at the first unpublished byte even when one dirty run crosses
      // multiple translation extents.
      if (completed_bytes != 0)
        clear_dirty_bytes(line_addr, start, static_cast<uint32_t>(completed_bytes), vmid);
      if (outcome != VmAccessOutcome::Complete)
        return outcome;
      diagnostics(line_addr + start).backing_writes.fetch_add(1, std::memory_order_relaxed);
    }
  }
  return VmAccessOutcome::Complete;
}

void L2Cache::clear_all_dirty_bytes() {
  std::lock_guard lock(dirty_bytes_mutex_);
  dirty_bytes_.clear();
  has_dirty_lines_.store(false, std::memory_order_relaxed);
}

VmAccessOutcome L2Cache::publish_dirty_bytes_before_external_write(uint64_t line_addr,
                                                                   const uint8_t *data,
                                                                   const simdojo::CacheTag &tag,
                                                                   const GpuVmAccess *access) {
  if (tag.vmid != 0) {
    // A reused VMID cannot authorize publication of an old resident line.
    if (access == nullptr || tag.cache_identity == 0 ||
        tag.cache_identity != access->cache_identity()) {
      (void)clear_dirty_bytes(line_addr, 0, LINE_SIZE, tag.vmid);
      return VmAccessOutcome::Complete;
    }
  }

  DirtyMask mask{};
  {
    std::lock_guard lock(dirty_bytes_mutex_);
    const auto position = dirty_bytes_.find({tag.vmid, line_addr});
    if (position == dirty_bytes_.end())
      return VmAccessOutcome::Complete;
    mask = position->second;
  }
  if (dirty_mask_overlaps_controlled(line_addr, tag.vmid, mask))
    return VmAccessOutcome::Unavailable;

  // Admission can split a formerly dirty run across grants. Check each byte
  // under the same retained binding and coalesce eligible neighbors for IO.
  uint32_t run_start = LINE_SIZE;
  for (uint32_t byte = 0; byte <= LINE_SIZE; ++byte) {
    bool eligible = false;
    if (byte != LINE_SIZE && (mask[byte / 64] & (uint64_t{1} << (byte % 64))) != 0) {
      if (access) {
        // A peer alias needs a tokenized canonical-backing sink. A live peer
        // with a missing route must also retain dirty bytes for a later retry.
        const VmDataRoute route =
            access->classify_data_route(line_addr + byte, 1, VmAccessKind::Write,
                                        /*report_fault=*/false);
        if (route != VmDataRoute::Local) {
          if (access->revoked()) {
            (void)clear_dirty_bytes(line_addr, 0, LINE_SIZE, tag.vmid);
            return VmAccessOutcome::Complete;
          }
          return route == VmDataRoute::Deferred ? VmAccessOutcome::Unavailable
                                                : VmAccessOutcome::Faulted;
        }
        const VmAccessAdmission admission =
            access->admit_access(line_addr + byte, 1, VmAccessKind::Write,
                                 /*report_fault=*/false);
        if (admission.outcome == VmAccessOutcome::Revoked) {
          (void)clear_dirty_bytes(line_addr, 0, LINE_SIZE, tag.vmid);
          return VmAccessOutcome::Complete;
        }
        if (!admission)
          return admission.outcome;
        eligible = admission.cacheable && admission.generation == tag.admission_generation;
      } else {
        eligible = true;
      }
    }
    if (eligible) {
      if (run_start == LINE_SIZE)
        run_start = byte;
      continue;
    }
    if (run_start != LINE_SIZE) {
      const uint32_t run_size = byte - run_start;
      VmAccessOutcome outcome = VmAccessOutcome::Complete;
      if (access) {
        const std::span<const uint8_t> bytes(data + run_start, run_size);
        std::size_t completed_bytes = 0;
        outcome = access->write(line_addr + run_start, std::as_bytes(bytes), completed_bytes);
        if (completed_bytes > run_size ||
            (outcome == VmAccessOutcome::Complete && completed_bytes != run_size))
          return VmAccessOutcome::Malformed;
        if (completed_bytes != 0)
          (void)clear_dirty_bytes(line_addr, run_start, static_cast<uint32_t>(completed_bytes),
                                  tag.vmid);
        diagnostics(line_addr + run_start).backing_writes.fetch_add(1, std::memory_order_relaxed);
      } else {
        outcome = send_backing(line_addr + run_start, const_cast<uint8_t *>(data + run_start),
                               run_size, simdojo::MessageOp::WRITE, 0);
        if (outcome == VmAccessOutcome::Complete)
          (void)clear_dirty_bytes(line_addr, run_start, run_size, tag.vmid);
      }
      if (outcome == VmAccessOutcome::Revoked) {
        (void)clear_dirty_bytes(line_addr, 0, LINE_SIZE, tag.vmid);
        return VmAccessOutcome::Complete;
      }
      if (outcome != VmAccessOutcome::Complete)
        return outcome;
      run_start = LINE_SIZE;
    }
    if (byte != LINE_SIZE && (mask[byte / 64] & (uint64_t{1} << (byte % 64))) != 0)
      (void)clear_dirty_bytes(line_addr, byte, 1, tag.vmid);
  }
  return VmAccessOutcome::Complete;
}

VmAccessOutcome L2Cache::publish_dirty_bytes_for_completion(uint64_t line_addr, const uint8_t *data,
                                                            const simdojo::CacheTag &tag,
                                                            const GpuVmAccess &access) {
  DirtyMask mask{};
  {
    std::lock_guard lock(dirty_bytes_mutex_);
    const auto position = dirty_bytes_.find({tag.vmid, line_addr});
    if (position == dirty_bytes_.end())
      return VmAccessOutcome::Unavailable;
    mask = position->second;
  }
  if (dirty_mask_overlaps_controlled(line_addr, tag.vmid, mask))
    return VmAccessOutcome::Unavailable;

  for (uint32_t byte = 0; byte < LINE_SIZE;) {
    while (byte < LINE_SIZE && (mask[byte / 64] & (uint64_t{1} << (byte % 64))) == 0)
      ++byte;
    const uint32_t begin = byte;
    while (byte < LINE_SIZE && (mask[byte / 64] & (uint64_t{1} << (byte % 64))) != 0)
      ++byte;
    if (begin == byte)
      continue;
    if (!CacheStore::bytes_valid(tag, begin, byte - begin))
      return VmAccessOutcome::Unavailable;

    // Validate every dirty byte against the retained grant. A line can span
    // distinct admissions after it became dirty; none may be silently dropped
    // at a completion boundary.
    for (uint32_t offset = begin; offset < byte; ++offset) {
      const VmDataRoute route =
          access.classify_data_route(line_addr + offset, 1, VmAccessKind::Write,
                                     /*report_fault=*/false);
      if (route != VmDataRoute::Local)
        return access.revoked()                 ? VmAccessOutcome::Revoked
               : route == VmDataRoute::Deferred ? VmAccessOutcome::Unavailable
                                                : VmAccessOutcome::Faulted;
      const VmAccessAdmission admission =
          access.admit_access(line_addr + offset, 1, VmAccessKind::Write,
                              /*report_fault=*/false);
      if (!admission)
        return admission.outcome;
      if (!admission.direct_backing_write_allowed)
        return VmAccessOutcome::Unavailable;
      if (!admission.cacheable || admission.generation != tag.admission_generation)
        return VmAccessOutcome::Faulted;
    }

    const std::span<const uint8_t> run(data + begin, byte - begin);
    std::size_t completed_bytes = 0;
    const VmAccessOutcome outcome =
        access.write(line_addr + begin, std::as_bytes(run), completed_bytes);
    if (completed_bytes > run.size() ||
        (outcome == VmAccessOutcome::Complete && completed_bytes != run.size()))
      return VmAccessOutcome::Malformed;
    if (completed_bytes != 0)
      (void)clear_dirty_bytes(line_addr, begin, static_cast<uint32_t>(completed_bytes), tag.vmid);
    diagnostics(line_addr + begin).backing_writes.fetch_add(1, std::memory_order_relaxed);
    if (outcome != VmAccessOutcome::Complete)
      return outcome;
  }
  return VmAccessOutcome::Complete;
}

VmAccessOutcome L2Cache::access_outcome(simdojo::MessageStatus status) {
  switch (status) {
  case simdojo::MessageStatus::Complete:
    return VmAccessOutcome::Complete;
  case simdojo::MessageStatus::Unavailable:
    return VmAccessOutcome::Unavailable;
  case simdojo::MessageStatus::Faulted:
    return VmAccessOutcome::Faulted;
  case simdojo::MessageStatus::Malformed:
    return VmAccessOutcome::Malformed;
  }
  return VmAccessOutcome::Malformed;
}

VmAccessOutcome
L2Cache::read_translated_local(const GpuVmAccess &access, uint64_t addr, std::span<uint8_t> dst,
                               Mtype instruction_mtype,
                               const DeviceCacheCoherence::LocalCacheAccess &domain_access) {
  const uint32_t offset = CacheStore::line_offset(addr);
  if (!domain_access.belongs_to(coherence_.get()) || gpu_vm_ == nullptr || dst.empty() ||
      dst.size() > LINE_SIZE - offset || access.info().legacy_cache_compatible ||
      !access.supports_translated_local_cache() || access.cache_identity() == 0)
    return VmAccessOutcome::Malformed;

  MemoryIngressAdmission transport = access.admit_pt_rewalk();
  if (transport.outcome != VmAccessOutcome::Complete)
    return transport.outcome;

  const uint64_t line_address = CacheStore::line_address(addr);
  const auto selected =
      query_translated_local_line(access, line_address, LINE_SIZE, instruction_mtype);
  if (!selected)
    return VmAccessOutcome::Unavailable;
  const VmAccessAdmission admission =
      access.admit_access(line_address, LINE_SIZE, VmAccessKind::Read, false);
  if (!admission)
    return admission.outcome;
  if (!admission.cacheable)
    return access.read(addr, std::as_writable_bytes(dst));

  const uint64_t identity = access.cache_identity();
  const uint64_t backing_epoch = coherence_->current_local_backing_epoch();
  const uint64_t line_epoch = coherence_->current_local_line_epoch(selected->backing_address);
  auto cache_access = acquire_cache_access();
  std::lock_guard set_lock(set_mutex(addr));
  const simdojo::CacheTag *resident = cache_.peek(addr, 0, identity);
  if (resident != nullptr &&
      (!translated_local_tag_matches(*resident, *selected, identity, backing_epoch, line_epoch) ||
       resident->admission_generation != admission.generation)) {
    cache_.invalidate(addr, 0, identity);
    resident = nullptr;
  }
  if (resident != nullptr && CacheStore::bytes_valid(*resident, offset, dst.size())) {
    auto current = gpu_vm_->try_acquire_current_namespace(access.cache_namespace().address_space,
                                                          access.cache_namespace());
    if (!current)
      return VmAccessOutcome::Unavailable;
    cache_.lookup(addr, nullptr, 0, identity);
    cache_.read_line(addr, dst.data(), offset, dst.size(), 0, identity);
    return VmAccessOutcome::Complete;
  }

  std::array<uint8_t, LINE_SIZE> line_bytes{};
  const VmAccessOutcome fetched =
      access.read(line_address, std::as_writable_bytes(std::span<uint8_t>(line_bytes)));
  if (fetched != VmAccessOutcome::Complete)
    return fetched;
  diagnostics(line_address).backing_reads.fetch_add(1, std::memory_order_relaxed);
  const auto rewalked =
      query_translated_local_line(access, line_address, LINE_SIZE, instruction_mtype);
  if (!rewalked || rewalked->backing_address != selected->backing_address ||
      rewalked->mapping_attributes != selected->mapping_attributes ||
      coherence_->current_local_backing_epoch() != backing_epoch ||
      coherence_->current_local_line_epoch(selected->backing_address) != line_epoch)
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
    coherence_->mark_translated_local_cache_used();
  }
  std::memcpy(dst.data(), line_bytes.data() + offset, dst.size());
  return VmAccessOutcome::Complete;
}

bool L2Cache::can_fetch_range(uint64_t addr, uint32_t size, uint32_t vmid) const {
  if (vmid == 0 || gpu_vm_ == nullptr)
    return true;
  return gpu_vm_->with_vmid_snapshot(vmid, [&](const GpuVmAccess *vm_access) {
    return vm_access &&
           vm_access->query_access(addr, size, VmAccessKind::Read) == VmAccessOutcome::Complete;
  });
}

VmAccessAdmission L2Cache::admit_access(uint64_t addr, uint32_t size, uint32_t vmid,
                                        VmAccessKind access, bool report_fault) const {
  if (vmid == 0 || gpu_vm_ == nullptr)
    return {};
  return gpu_vm_->with_vmid_snapshot(vmid, [&](const GpuVmAccess *vm_access) {
    return vm_access ? vm_access->admit_access(addr, size, access, report_fault)
                     : VmAccessAdmission{.outcome = VmAccessOutcome::Faulted};
  });
}

VmAccessAdmission L2Cache::query_cache_write_grant(uint64_t addr, uint32_t size,
                                                   uint32_t vmid) const {
  if (vmid == 0 || gpu_vm_ == nullptr)
    return {};
  return gpu_vm_->with_vmid_snapshot(vmid, [&](const GpuVmAccess *vm_access) {
    return vm_access ? vm_access->query_cache_write_grant(addr, size)
                     : VmAccessAdmission{.outcome = VmAccessOutcome::Faulted};
  });
}

bool L2Cache::try_read_peer_cached(uint64_t addr, std::span<uint8_t> dst, uint32_t vmid,
                                   uint64_t admission_generation, uint64_t cache_identity) {
  const uint32_t offset = CacheStore::line_offset(addr);
  if (dst.empty() || dst.size() > LINE_SIZE - offset ||
      (vmid == 0 ? (admission_generation != 0 || cache_identity != 0)
                 : (admission_generation == 0 || cache_identity == 0)))
    return false;

  auto access_lock = acquire_cache_access();
  std::lock_guard set_lock(set_mutex(addr));
  const simdojo::CacheTag *tag = cache_.peek(addr, vmid);
  if (tag == nullptr || tag->admission_generation != admission_generation ||
      tag->cache_identity != cache_identity ||
      !CacheStore::bytes_valid(*tag, offset, static_cast<uint32_t>(dst.size())))
    return false;
  cache_.read_line(addr, dst.data(), offset, static_cast<uint32_t>(dst.size()), vmid);
  return true;
}

bool L2Cache::try_fill_peer_cached(uint64_t addr, std::span<const uint8_t> src, uint32_t vmid,
                                   uint64_t admission_generation, uint64_t cache_identity) {
  const uint32_t offset = CacheStore::line_offset(addr);
  if (src.empty() || src.size() > LINE_SIZE - offset ||
      (vmid == 0 ? (admission_generation != 0 || cache_identity != 0)
                 : (admission_generation == 0 || cache_identity == 0)))
    return false;

  auto access_lock = acquire_cache_access();
  std::lock_guard set_lock(set_mutex(addr));
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
    allocated.tag->coherence = simdojo::CoherenceState::EXCLUSIVE;
    CacheStore::clear_valid_bytes(*allocated.tag);
  }
  cache_.write_line(addr, src.data(), offset, static_cast<uint32_t>(src.size()), vmid);
  return true;
}

VmAccessOutcome L2Cache::preflight_atomic_access(uint64_t addr, uint32_t size,
                                                 uint32_t vmid) const {
  return admit_access(addr, size, vmid, VmAccessKind::Atomic).outcome;
}

VmAccessOutcome L2Cache::discard_stale_line_locked(uint64_t addr, uint32_t vmid,
                                                   uint64_t admission_generation) {
  simdojo::CacheTag *tag = nullptr;
  if (!cache_.lookup(addr, &tag, vmid))
    return VmAccessOutcome::Complete;
  if (tag->cache_identity != current_cache_identity(vmid)) {
    (void)clear_dirty_bytes(CacheStore::line_address(addr), 0, LINE_SIZE, vmid);
    cache_.invalidate(addr, vmid);
    return VmAccessOutcome::Complete;
  }
  if (tag->admission_generation == admission_generation)
    return VmAccessOutcome::Complete;
  // A different grant can claim another part of this line while the old
  // grant remains live. Publish its admitted dirty bytes before replacement.
  return flush_line_locked(addr, vmid);
}

uint64_t L2Cache::current_cache_identity(uint32_t vmid) const {
  if (vmid == 0 || gpu_vm_ == nullptr)
    return 0;
  return gpu_vm_->with_vmid_snapshot(vmid, [](const GpuVmAccess *access) {
    return access != nullptr && access->is_current() ? access->cache_identity() : uint64_t{0};
  });
}

bool L2Cache::retain_admitted_dirty_bytes(uint64_t line_addr, const simdojo::CacheTag &tag) {
  if (!tag.dirty)
    return true;
  if (tag.cache_identity != current_cache_identity(tag.vmid)) {
    (void)clear_dirty_bytes(line_addr, 0, LINE_SIZE, tag.vmid);
    return false;
  }
  DirtyMask mask{};
  {
    std::lock_guard lock(dirty_bytes_mutex_);
    const auto it = dirty_bytes_.find({tag.vmid, line_addr});
    if (it == dirty_bytes_.end())
      return false;
    mask = it->second;
  }
  bool retained = false;
  for (uint32_t offset = 0; offset < LINE_SIZE;) {
    while (offset < LINE_SIZE && (mask[offset / 64] & (uint64_t{1} << (offset % 64))) == 0)
      ++offset;
    const uint32_t start = offset;
    while (offset < LINE_SIZE && (mask[offset / 64] & (uint64_t{1} << (offset % 64))) != 0)
      ++offset;
    if (start == offset)
      continue;
    const VmAccessAdmission admission =
        query_cache_write_grant(line_addr + start, offset - start, tag.vmid);
    if (admission && admission.cacheable && admission.generation == tag.admission_generation) {
      retained = true;
      continue;
    }
    // A line can be split between grants after it became dirty. Retire only
    // bytes whose original grant is gone; a surviving prefix still needs
    // writeback even when the old dirty run crosses the new boundary.
    for (uint32_t byte = start; byte < offset; ++byte) {
      const VmAccessAdmission byte_admission =
          query_cache_write_grant(line_addr + byte, 1, tag.vmid);
      if (byte_admission && byte_admission.cacheable &&
          byte_admission.generation == tag.admission_generation)
        retained = true;
      else
        (void)clear_dirty_bytes(line_addr, byte, 1, tag.vmid);
    }
  }
  return retained;
}

bool L2Cache::direct_controlled_writeback(uint64_t address, uint32_t size,
                                          uint32_t vmid) const noexcept {
  GpuVm *const vm = gpu_vm_ != nullptr ? gpu_vm_ : legacy_maintenance_vm_;
  return vm != nullptr && vm->controlled_cache_backing_overlaps(vmid, address, size);
}

bool L2Cache::dirty_mask_overlaps_controlled(uint64_t line_addr, uint32_t vmid,
                                             const DirtyMask &mask) const noexcept {
  if (!direct_controlled_writeback(line_addr, LINE_SIZE, vmid))
    return false;
  for (uint32_t byte = 0; byte < LINE_SIZE;) {
    while (byte < LINE_SIZE && (mask[byte / 64] & (uint64_t{1} << (byte % 64))) == 0)
      ++byte;
    const uint32_t begin = byte;
    while (byte < LINE_SIZE && (mask[byte / 64] & (uint64_t{1} << (byte % 64))) != 0)
      ++byte;
    if (begin != byte && direct_controlled_writeback(line_addr + begin, byte - begin, vmid))
      return true;
  }
  return false;
}

bool L2Cache::guarded_dirty_allocation_victim(uint64_t replacement_addr) {
  const simdojo::CacheTag *victim = cache_.full_set_victim(replacement_addr);
  if (victim == nullptr || !victim->dirty)
    return false;
  static constexpr uint32_t SET_INDEX_BITS = std::bit_width(NUM_SETS - 1);
  const uint64_t victim_addr =
      (victim->tag << (LINE_SIZE_BITS + SET_INDEX_BITS)) |
      (static_cast<uint64_t>(CacheStore::set_index(replacement_addr)) << LINE_SIZE_BITS);
  if (!direct_controlled_writeback(victim_addr, LINE_SIZE, victim->vmid) ||
      victim->cache_identity != current_cache_identity(victim->vmid))
    return false;
  DirtyMask mask{};
  {
    std::lock_guard lock(dirty_bytes_mutex_);
    const auto original = dirty_bytes_.find({victim->vmid, victim_addr});
    if (original == dirty_bytes_.end())
      return false;
    mask = original->second;
  }
  // A retired grant's bytes are discarded during ordinary eviction. Only an
  // admitted dirty byte in the controlled range requires stable LRU refusal.
  for (uint32_t byte = 0; byte < LINE_SIZE;) {
    while (byte < LINE_SIZE && (mask[byte / 64] & (uint64_t{1} << (byte % 64))) == 0)
      ++byte;
    const uint32_t begin = byte;
    while (byte < LINE_SIZE && (mask[byte / 64] & (uint64_t{1} << (byte % 64))) != 0)
      ++byte;
    if (begin == byte ||
        !direct_controlled_writeback(victim_addr + begin, byte - begin, victim->vmid))
      continue;
    const VmAccessAdmission admission =
        query_cache_write_grant(victim_addr + begin, byte - begin, victim->vmid);
    if (admission && admission.cacheable && admission.generation == victim->admission_generation)
      return true;
    for (uint32_t offset = begin; offset < byte; ++offset) {
      if (!direct_controlled_writeback(victim_addr + offset, 1, victim->vmid))
        continue;
      const VmAccessAdmission part = query_cache_write_grant(victim_addr + offset, 1, victim->vmid);
      if (part && part.cacheable && part.generation == victim->admission_generation)
        return true;
    }
  }
  return false;
}

VmAccessOutcome L2Cache::send_backing(uint64_t addr, uint8_t *data, uint32_t size,
                                      simdojo::MessageOp op, uint32_t vmid) {
  if (op == simdojo::MessageOp::WRITE && direct_controlled_writeback(addr, size, vmid))
    return VmAccessOutcome::Unavailable;
  if (op == simdojo::MessageOp::WRITE)
    diagnostics(addr).backing_writes.fetch_add(1, std::memory_order_relaxed);
  else
    diagnostics(addr).backing_reads.fetch_add(1, std::memory_order_relaxed);
  if (backing_memory_) {
    if (vmid == 0 && op == simdojo::MessageOp::WRITE) {
      thread_local uint64_t wb_count = 0;
      if (++wb_count <= 3)
        util::Logger::vm("L2 writeback(backing) #", wb_count, " addr=0x", std::hex, addr,
                         " size=", std::dec, size);
      backing_memory_->write_block(addr, std::span<const uint8_t>(data, size));
      return VmAccessOutcome::Complete;
    }
    if (vmid == 0) {
      backing_memory_->read_block(addr, std::span<uint8_t>(data, size));
      return VmAccessOutcome::Complete;
    }
    if (gpu_vm_ == nullptr)
      return VmAccessOutcome::Unavailable;
    return gpu_vm_->with_vmid_snapshot(vmid, [&](const GpuVmAccess *vm_access) {
      if (!vm_access)
        return VmAccessOutcome::Faulted;
      return op == simdojo::MessageOp::WRITE
                 ? vm_access->write(addr, std::as_bytes(std::span<const uint8_t>(data, size)))
                 : vm_access->read(addr, std::as_writable_bytes(std::span<uint8_t>(data, size)));
    });
  }
  assert(req_port_ != nullptr && "L2Cache: req_port_ not set");
  if (req_port_->link() == nullptr ||
      req_port_->link()->exec_mode() != simdojo::ExecMode::FUNCTIONAL)
    return VmAccessOutcome::Malformed;
  auto message = std::make_unique<simdojo::Message>();
  simdojo::MessageHeader &header = message->header();
  simdojo::MessageStatus completion_status = simdojo::MessageStatus::Complete;
  header.addr = addr;
  header.size_bytes = size;
  header.op = op;
  header.vmid = vmid;
  header.completion_status = &completion_status;
  message->set_payload(reinterpret_cast<uintptr_t>(data));
  req_port_->send(std::move(message));
  return access_outcome(completion_status);
}

VmAccessOutcome L2Cache::send_atomic_backing(uint64_t addr, uint32_t size,
                                             const simdojo::MemoryAtomicMutation &mutation,
                                             uint32_t vmid) {
  diagnostics(addr).backing_reads.fetch_add(1, std::memory_order_relaxed);
  diagnostics(addr).backing_writes.fetch_add(1, std::memory_order_relaxed);
  assert(req_port_ != nullptr && "L2Cache: req_port_ not set");
  if (req_port_->link() == nullptr ||
      req_port_->link()->exec_mode() != simdojo::ExecMode::FUNCTIONAL)
    return VmAccessOutcome::Malformed;
  auto message = std::make_unique<simdojo::Message>();
  simdojo::MessageHeader &header = message->header();
  simdojo::MessageStatus completion_status = simdojo::MessageStatus::Complete;
  header.addr = addr;
  header.size_bytes = size;
  header.op = simdojo::MessageOp::ATOMIC;
  header.vmid = vmid;
  header.completion_status = &completion_status;
  message->set_payload(
      reinterpret_cast<uintptr_t>(const_cast<simdojo::MemoryAtomicMutation *>(&mutation)));
  req_port_->send(std::move(message));
  return access_outcome(completion_status);
}

VmAccessOutcome L2Cache::ensure_line(uint64_t addr, uint32_t vmid, uint64_t admission_generation,
                                     bool fetch_on_miss) {
  const VmAccessOutcome stale_outcome = discard_stale_line_locked(addr, vmid, admission_generation);
  if (stale_outcome != VmAccessOutcome::Complete)
    return stale_outcome;
  simdojo::CacheTag *resident = nullptr;
  if (cache_.lookup(addr, &resident, vmid)) {
    if (!fetch_on_miss || CacheStore::bytes_valid(*resident, 0, LINE_SIZE))
      return VmAccessOutcome::Complete;

    uint8_t line_buf[LINE_SIZE];
    const VmAccessOutcome outcome = send_backing(CacheStore::line_address(addr), line_buf,
                                                 LINE_SIZE, simdojo::MessageOp::READ, vmid);
    if (outcome != VmAccessOutcome::Complete)
      return outcome;
    cache_.fill_missing_bytes(addr, line_buf, vmid);
    return VmAccessOutcome::Complete;
  }

  if (guarded_dirty_allocation_victim(addr))
    return VmAccessOutcome::Unavailable;
  uint64_t line_addr = CacheStore::line_address(addr);
  simdojo::CacheTag evicted;
  uint8_t evicted_data[LINE_SIZE];
  simdojo::CacheTag *allocated = cache_.allocate(addr, vmid, &evicted, evicted_data);
  allocated->admission_generation = admission_generation;
  allocated->cache_identity = current_cache_identity(vmid);

  if (evicted.valid && evicted.dirty) {
    static constexpr uint32_t SET_INDEX_BITS = std::bit_width(NUM_SETS - 1);
    uint64_t evicted_addr = (evicted.tag << (LINE_SIZE_BITS + SET_INDEX_BITS)) |
                            (static_cast<uint64_t>(CacheStore::set_index(addr)) << LINE_SIZE_BITS);
    util::Logger::vm([&](auto &os) {
      static thread_local uint64_t evict_count = 0;
      if (++evict_count <= 5 || (evict_count % 10000) == 0)
        os << std::format("L2 evict #{} addr={:#x} new={:#x}", evict_count, evicted_addr, addr);
    });
    // The evicted line is written back under its own owning vmid, which may
    // differ from the current request's vmid when two processes alias the
    // same GPU VA in different ways of the same set.
    const bool admitted = retain_admitted_dirty_bytes(evicted_addr, evicted);
    if (!admitted)
      (void)clear_dirty_bytes(evicted_addr, 0, LINE_SIZE, evicted.vmid);
    const VmAccessOutcome outcome =
        admitted ? publish_dirty_bytes(evicted_addr, evicted_data, evicted.vmid)
                 : VmAccessOutcome::Complete;
    if (outcome != VmAccessOutcome::Complete) {
      cache_.invalidate(addr, vmid);
      CacheStore::Allocation restored = cache_.allocate_with_data(evicted_addr, evicted.vmid);
      *restored.tag = evicted;
      std::memcpy(restored.data, evicted_data, LINE_SIZE);
      return outcome;
    }
  }

  // A store supplies only its architectural byte range. Keep all other bytes
  // absent so the store never speculatively reads backing memory. Dirty victims
  // are still published before the caller modifies the new line.
  if (!fetch_on_miss) {
    simdojo::CacheTag *allocated = nullptr;
    cache_.lookup(addr, &allocated, vmid);
    assert(allocated != nullptr);
    CacheStore::clear_valid_bytes(*allocated);
    return VmAccessOutcome::Complete;
  }

  uint8_t line_buf[LINE_SIZE];
  const VmAccessOutcome outcome =
      send_backing(line_addr, line_buf, LINE_SIZE, simdojo::MessageOp::READ, vmid);
  if (outcome != VmAccessOutcome::Complete) {
    cache_.invalidate(addr, vmid);
    return outcome;
  }
  cache_.fill_line(addr, line_buf, vmid);
  return VmAccessOutcome::Complete;
}

VmAccessOutcome L2Cache::cache_partial_bytes(uint64_t addr, const uint8_t *src, uint32_t size,
                                             uint32_t vmid, uint64_t admission_generation) {
  const VmAccessOutcome stale_outcome = discard_stale_line_locked(addr, vmid, admission_generation);
  if (stale_outcome != VmAccessOutcome::Complete)
    return stale_outcome;
  simdojo::CacheTag *tag = nullptr;
  bool allocated_new = false;
  if (!cache_.lookup(addr, &tag, vmid)) {
    if (guarded_dirty_allocation_victim(addr))
      return VmAccessOutcome::Unavailable;
    simdojo::CacheTag evicted;
    uint8_t evicted_data[LINE_SIZE];
    CacheStore::Allocation allocated =
        cache_.allocate_with_data(addr, vmid, &evicted, evicted_data);

    if (evicted.valid && evicted.dirty) {
      static constexpr uint32_t SET_INDEX_BITS = std::bit_width(NUM_SETS - 1);
      const uint64_t evicted_addr =
          (evicted.tag << (LINE_SIZE_BITS + SET_INDEX_BITS)) |
          (static_cast<uint64_t>(CacheStore::set_index(addr)) << LINE_SIZE_BITS);
      const bool admitted = retain_admitted_dirty_bytes(evicted_addr, evicted);
      if (!admitted)
        (void)clear_dirty_bytes(evicted_addr, 0, LINE_SIZE, evicted.vmid);
      const VmAccessOutcome outcome =
          admitted ? publish_dirty_bytes(evicted_addr, evicted_data, evicted.vmid)
                   : VmAccessOutcome::Complete;
      if (outcome != VmAccessOutcome::Complete) {
        cache_.invalidate(addr, vmid);
        CacheStore::Allocation restored = cache_.allocate_with_data(evicted_addr, evicted.vmid);
        *restored.tag = evicted;
        std::memcpy(restored.data, evicted_data, LINE_SIZE);
        return outcome;
      }
    }
    tag = allocated.tag;
    tag->admission_generation = admission_generation;
    tag->cache_identity = current_cache_identity(vmid);
    CacheStore::clear_valid_bytes(*tag);
    allocated_new = true;
  }

  cache_.write_line(addr, src, CacheStore::line_offset(addr), size, vmid);
  if (allocated_new) {
    tag->dirty = false;
    tag->coherence = simdojo::CoherenceState::EXCLUSIVE;
  }
  return VmAccessOutcome::Complete;
}

bool L2Cache::try_read_scalar_ram(uint64_t addr, uint32_t *dst, uint32_t num_dwords,
                                  uint32_t vmid) {
  if (!backing_memory_ || !gpu_vm_ || vmid == 0 || num_dwords < 2 || num_dwords > 16 ||
      (addr & 3) || num_dwords * 4 > 64 - (addr & 63))
    return false;
  // Discover the snapshot before cache admission: a cold quantum snapshot may
  // allocate its retained entry. The optional copy itself cannot allocate.
  return gpu_vm_->with_vmid_snapshot(vmid, [&](const GpuVmAccess *access) {
    if (!access)
      return false;
    if (!access->admit_access(addr, num_dwords * sizeof(uint32_t), VmAccessKind::Read,
                              /*report_fault=*/false))
      return false;
    auto maintenance_lock = acquire_cache_access();
    std::lock_guard set_lock(set_mutex(addr));
    // Resident lines retain the existing flush/fault path. A miss neither
    // updates replacement state nor publishes or invalidates cached bytes.
    if (cache_.peek(addr, vmid))
      return false;
    const bool copied =
        access->try_read_uncached_ram(addr, std::as_writable_bytes(std::span(dst, num_dwords)));
    if (copied)
      diagnostics(addr).backing_reads.fetch_add(1, std::memory_order_relaxed);
    return copied;
  });
}

VmAccessOutcome L2Cache::read(uint64_t addr, uint8_t *dst, uint32_t size, Mtype mtype,
                              uint32_t vmid) {
  if (size == 0)
    return VmAccessOutcome::Complete;
  const VmAccessAdmission admission = admit_access(addr, size, vmid, VmAccessKind::Read);
  if (!admission)
    return admission.outcome;
  auto maintenance_lock = acquire_cache_access();
  uint32_t copied = 0;
  if (mtype == Mtype::UC) {
    while (copied < size) {
      const uint64_t ea = addr + copied;
      const uint32_t line_offset = CacheStore::line_offset(ea);
      const uint32_t chunk = std::min(size - copied, LINE_SIZE - line_offset);
      const VmAccessAdmission chunk_admission = admit_access(ea, chunk, vmid, VmAccessKind::Read);
      if (!chunk_admission)
        return chunk_admission.outcome;

      std::lock_guard set_lock(set_mutex(ea));
      const VmAccessOutcome flush_outcome = flush_line_locked(ea, vmid);
      if (flush_outcome != VmAccessOutcome::Complete)
        return flush_outcome;
      const VmAccessOutcome read_outcome =
          send_backing(ea, dst + copied, chunk, simdojo::MessageOp::READ, vmid);
      if (read_outcome != VmAccessOutcome::Complete)
        return read_outcome;
      copied += chunk;
    }
    return VmAccessOutcome::Complete;
  }

  copied = 0;
  while (copied < size) {
    const uint64_t ea = addr + copied;
    const uint32_t line_offset = CacheStore::line_offset(ea);
    const uint32_t chunk = std::min(size - copied, LINE_SIZE - line_offset);
    const VmAccessAdmission chunk_admission = admit_access(ea, chunk, vmid, VmAccessKind::Read);
    if (!chunk_admission)
      return chunk_admission.outcome;
    const VmAccessAdmission line_admission =
        admit_access(CacheStore::line_address(ea), LINE_SIZE, vmid, VmAccessKind::Read,
                     /*report_fault=*/false);
    const bool cache_line = line_admission && line_admission.cacheable &&
                            line_admission.generation == chunk_admission.generation;
    std::lock_guard set_lock(set_mutex(ea));

    if (!cache_line) {
      const VmAccessOutcome flush_outcome = flush_line_locked(ea, vmid);
      if (flush_outcome != VmAccessOutcome::Complete)
        return flush_outcome;
      const VmAccessOutcome outcome =
          send_backing(ea, dst + copied, chunk, simdojo::MessageOp::READ, vmid);
      if (outcome != VmAccessOutcome::Complete)
        return outcome;
      copied += chunk;
      continue;
    }
    const VmAccessOutcome stale_outcome =
        discard_stale_line_locked(ea, vmid, chunk_admission.generation);
    if (stale_outcome != VmAccessOutcome::Complete)
      return stale_outcome;

    if (mtype == Mtype::CC) {
      // Coherently Cacheable: invalidate L2 line before refetch, mirroring
      // the L1 CC behavior. This ensures cross-XCD store visibility when
      // different L2s share a backing store. Dirty data is flushed first.
      const VmAccessOutcome outcome = flush_line_locked(ea, vmid);
      if (outcome != VmAccessOutcome::Complete)
        return outcome;
    }

    simdojo::CacheTag *resident = nullptr;
    if (cache_.lookup(ea, &resident, vmid) &&
        CacheStore::bytes_valid(*resident, line_offset, chunk)) {
      cache_.read_line(ea, dst + copied, line_offset, chunk, vmid);
      copied += chunk;
      continue;
    }

    if (!can_fetch_range(CacheStore::line_address(ea), LINE_SIZE, vmid)) {
      const VmAccessOutcome outcome =
          send_backing(ea, dst + copied, chunk, simdojo::MessageOp::READ, vmid);
      if (outcome != VmAccessOutcome::Complete)
        return outcome;
      const VmAccessOutcome cache_outcome =
          cache_partial_bytes(ea, dst + copied, chunk, vmid, chunk_admission.generation);
      if (cache_outcome != VmAccessOutcome::Complete)
        return cache_outcome;
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

bool L2Cache::try_write_private_dwords(std::span<const VmRamDwordStore> stores,
                                       Mtype instruction_mtype, Mtype mtype, uint32_t vmid) {
  if (!backing_memory_ || !gpu_vm_ || !vmid || stores.size() < 2 ||
      stores.size() > VmRamDwordStore::kMaxBatch)
    return false;
  const uint64_t address = stores.front().address;
  for (const auto &store : stores)
    if ((store.address & 3) ||
        CacheStore::line_address(store.address) != CacheStore::line_address(address))
      return false;
  // A cold snapshot can allocate retained storage; prepare it before cache admission.
  return gpu_vm_->with_vmid_snapshot(vmid, [&](const GpuVmAccess *access) {
    if (!access)
      return false;
    uint64_t generation = 0;
    bool have_generation = false;
    for (const auto &store : stores) {
      const VmAccessAdmission admission =
          access->admit_access(store.address, sizeof(uint32_t), VmAccessKind::Write,
                               /*report_fault=*/false);
      if (!admission || !admission.cacheable ||
          (have_generation && generation != admission.generation))
        return false;
      generation = admission.generation;
      have_generation = true;
    }
    const VmAccessAdmission line_admission =
        access->admit_access(CacheStore::line_address(address), LINE_SIZE, VmAccessKind::Write,
                             /*report_fault=*/false);
    if (!line_admission || !line_admission.cacheable || line_admission.generation != generation)
      return false;
    auto maintenance_lock = acquire_cache_access();
    std::lock_guard set_lock(set_mutex(address));
    const auto *resident = cache_.peek(address, vmid);
    if (mtype == Mtype::UC ? resident != nullptr : !resident || resident->dirty)
      return false;
    if (resident && resident->admission_generation != generation)
      return false;
    if (!access->try_write_private_dwords(stores, instruction_mtype, mtype))
      return false;
    diagnostics(address).backing_writes.fetch_add(stores.size(), std::memory_order_relaxed);
    if (mtype != Mtype::UC) {
      for (const auto &store : stores) {
        simdojo::CacheTag *tag = nullptr;
        cache_.lookup(store.address, &tag, vmid);
        cache_.write_line(store.address, store.source, CacheStore::line_offset(store.address),
                          sizeof(uint32_t), vmid);
        tag->coherence = mtype == Mtype::CC ? simdojo::CoherenceState::SHARED
                                            : simdojo::CoherenceState::EXCLUSIVE;
      }
      diagnostics(address).writes.fetch_add(stores.size(), std::memory_order_relaxed);
    }
    return true;
  });
}

VmAccessOutcome L2Cache::write(uint64_t addr, const uint8_t *src, uint32_t size, Mtype mtype,
                               uint32_t vmid) {
  if (size == 0)
    return VmAccessOutcome::Complete;
  const VmAccessAdmission admission = admit_access(addr, size, vmid, VmAccessKind::Write);
  if (!admission)
    return admission.outcome;
  if (direct_controlled_writeback(addr, size, vmid))
    return VmAccessOutcome::Unavailable;
  auto maintenance_lock = acquire_cache_access();
  uint32_t copied = 0;
  if (mtype == Mtype::UC) {
    while (copied < size) {
      const uint64_t ea = addr + copied;
      const uint32_t line_offset = CacheStore::line_offset(ea);
      const uint32_t chunk = std::min(size - copied, LINE_SIZE - line_offset);
      const VmAccessAdmission chunk_admission = admit_access(ea, chunk, vmid, VmAccessKind::Write);
      if (!chunk_admission)
        return chunk_admission.outcome;

      std::lock_guard set_lock(set_mutex(ea));
      const VmAccessOutcome flush_outcome = flush_line_locked(ea, vmid);
      if (flush_outcome != VmAccessOutcome::Complete)
        return flush_outcome;
      const VmAccessOutcome write_outcome = send_backing(ea, const_cast<uint8_t *>(src + copied),
                                                         chunk, simdojo::MessageOp::WRITE, vmid);
      if (write_outcome != VmAccessOutcome::Complete)
        return write_outcome;
      copied += chunk;
    }
    return VmAccessOutcome::Complete;
  }

  copied = 0;
  while (copied < size) {
    const uint64_t ea = addr + copied;
    const uint32_t line_offset = CacheStore::line_offset(ea);
    const uint32_t chunk = std::min(size - copied, LINE_SIZE - line_offset);
    const VmAccessAdmission chunk_admission = admit_access(ea, chunk, vmid, VmAccessKind::Write);
    if (!chunk_admission)
      return chunk_admission.outcome;
    const VmAccessAdmission line_admission =
        admit_access(CacheStore::line_address(ea), LINE_SIZE, vmid, VmAccessKind::Write,
                     /*report_fault=*/false);
    const bool cache_line = line_admission && line_admission.cacheable &&
                            line_admission.generation == chunk_admission.generation;
    std::lock_guard set_lock(set_mutex(ea));

    if (!cache_line) {
      const VmAccessOutcome flush_outcome = flush_line_locked(ea, vmid);
      if (flush_outcome != VmAccessOutcome::Complete)
        return flush_outcome;
      const VmAccessOutcome outcome = send_backing(ea, const_cast<uint8_t *>(src + copied), chunk,
                                                   simdojo::MessageOp::WRITE, vmid);
      if (outcome != VmAccessOutcome::Complete)
        return outcome;
      copied += chunk;
      continue;
    }
    const VmAccessOutcome stale_outcome =
        discard_stale_line_locked(ea, vmid, chunk_admission.generation);
    if (stale_outcome != VmAccessOutcome::Complete)
      return stale_outcome;

    const VmAccessOutcome fill_outcome =
        ensure_line(ea, vmid, chunk_admission.generation, /*fetch_on_miss=*/false);
    if (fill_outcome != VmAccessOutcome::Complete)
      return fill_outcome;

    const VmAccessOutcome write_outcome = send_backing(ea, const_cast<uint8_t *>(src + copied),
                                                       chunk, simdojo::MessageOp::WRITE, vmid);
    if (write_outcome != VmAccessOutcome::Complete)
      return write_outcome;
    cache_.write_line(ea, src + copied, line_offset, chunk, vmid);

    simdojo::CacheTag *tag = nullptr;
    cache_.lookup(ea, &tag, vmid);
    assert(tag != nullptr && "ensure_line must guarantee hit");

    // Write through to backing store for all mtypes. In the simulator, GPU
    // virtual addresses are host-mapped (MAP_FIXED), so hipMemcpy reads from
    // backing store directly. Without write-through, RW-mtype stores remain
    // in L2 as dirty lines and never reach the host-visible pages, causing
    // stale reads on D2H copy.
    tag->dirty = clear_dirty_bytes(CacheStore::line_address(ea), line_offset, chunk, vmid);
    tag->coherence = tag->dirty ? simdojo::CoherenceState::MODIFIED
                                : (mtype == Mtype::CC ? simdojo::CoherenceState::SHARED
                                                      : simdojo::CoherenceState::EXCLUSIVE);

    diagnostics(ea).writes.fetch_add(1, std::memory_order_relaxed);
    copied += chunk;
  }
  return VmAccessOutcome::Complete;
}

VmAccessOutcome L2Cache::fetch_line(uint64_t addr, uint8_t *line_buf, uint32_t vmid) {
  const uint64_t line_addr = CacheStore::line_address(addr);
  const VmAccessAdmission admission = admit_access(line_addr, LINE_SIZE, vmid, VmAccessKind::Read);
  if (!admission)
    return admission.outcome;
  auto maintenance_lock = acquire_cache_access();
  std::lock_guard set_lock(set_mutex(addr));
  if (!admission.cacheable) {
    const VmAccessOutcome flush_outcome = flush_line_locked(line_addr, vmid);
    if (flush_outcome != VmAccessOutcome::Complete)
      return flush_outcome;
    return send_backing(line_addr, line_buf, LINE_SIZE, simdojo::MessageOp::READ, vmid);
  }
  const VmAccessOutcome outcome = ensure_line(line_addr, vmid, admission.generation);
  if (outcome != VmAccessOutcome::Complete)
    return outcome;
  cache_.read_line(line_addr, line_buf, 0, LINE_SIZE, vmid);
  return VmAccessOutcome::Complete;
}

VmAccessOutcome L2Cache::writeback_line(uint64_t line_addr, const uint8_t *data, Mtype mtype,
                                        uint32_t vmid) {
  return writeback_line(line_addr, data, 0, LINE_SIZE, mtype, vmid);
}

VmAccessOutcome L2Cache::writeback_line(uint64_t line_addr, const uint8_t *data,
                                        uint32_t dirty_offset, uint32_t dirty_size, Mtype mtype,
                                        uint32_t vmid) {
  assert(CacheStore::line_address(line_addr) == line_addr && "writeback address must be aligned");
  assert(dirty_offset < LINE_SIZE && dirty_size != 0 && dirty_size <= LINE_SIZE - dirty_offset &&
         "dirty range must fit in one L2 line");
  const VmAccessAdmission admission =
      admit_access(line_addr + dirty_offset, dirty_size, vmid, VmAccessKind::Write);
  if (!admission)
    return admission.outcome;
  if (mtype == Mtype::CC && direct_controlled_writeback(line_addr + dirty_offset, dirty_size, vmid))
    return VmAccessOutcome::Unavailable;
  auto maintenance_lock = acquire_cache_access();
  std::lock_guard set_lock(set_mutex(line_addr));
  const VmAccessAdmission line_admission =
      admit_access(line_addr, LINE_SIZE, vmid, VmAccessKind::Write, /*report_fault=*/false);
  if (!line_admission || !line_admission.cacheable ||
      line_admission.generation != admission.generation) {
    if (direct_controlled_writeback(line_addr + dirty_offset, dirty_size, vmid))
      return VmAccessOutcome::Unavailable;
    const VmAccessOutcome flush_outcome = flush_line_locked(line_addr, vmid);
    if (flush_outcome != VmAccessOutcome::Complete)
      return flush_outcome;
    return send_backing(line_addr + dirty_offset, const_cast<uint8_t *>(data + dirty_offset),
                        dirty_size, simdojo::MessageOp::WRITE, vmid);
  }
  const VmAccessOutcome fill_outcome =
      ensure_line(line_addr, vmid, admission.generation, /*fetch_on_miss=*/false);
  if (fill_outcome != VmAccessOutcome::Complete)
    return fill_outcome;
  simdojo::CacheTag *tag = nullptr;
  cache_.lookup(line_addr, &tag, vmid);
  assert(tag != nullptr && "ensure_line must guarantee hit");

  if (mtype == Mtype::CC) {
    const VmAccessOutcome write_outcome =
        send_backing(line_addr + dirty_offset, const_cast<uint8_t *>(data + dirty_offset),
                     dirty_size, simdojo::MessageOp::WRITE, vmid);
    if (write_outcome != VmAccessOutcome::Complete)
      return write_outcome;
    cache_.write_line(line_addr, data + dirty_offset, dirty_offset, dirty_size, vmid);
    tag->dirty = clear_dirty_bytes(line_addr, dirty_offset, dirty_size, vmid);
    tag->coherence =
        tag->dirty ? simdojo::CoherenceState::MODIFIED : simdojo::CoherenceState::SHARED;
  } else {
    cache_.write_line(line_addr, data + dirty_offset, dirty_offset, dirty_size, vmid);
    mark_dirty_bytes(line_addr, dirty_offset, dirty_size, vmid);
    tag->dirty = true;
    tag->coherence = simdojo::CoherenceState::MODIFIED;
  }
  return VmAccessOutcome::Complete;
}

VmAccessOutcome L2Cache::flush_line_locked(uint64_t addr, uint32_t vmid) {
  simdojo::CacheTag *tag = nullptr;
  if (!cache_.lookup(addr, &tag, vmid))
    return VmAccessOutcome::Complete;

  if (tag->dirty) {
    uint64_t line_addr = CacheStore::line_address(addr);
    if (!retain_admitted_dirty_bytes(line_addr, *tag)) {
      (void)clear_dirty_bytes(line_addr, 0, LINE_SIZE, vmid);
    } else {
      const uint8_t *line = cache_.line_data_for_read(addr, vmid);
      assert(line != nullptr);
      const VmAccessOutcome outcome = publish_dirty_bytes(line_addr, line, vmid);
      if (outcome != VmAccessOutcome::Complete)
        return outcome;
    }
  }
  cache_.invalidate(addr, vmid);
  return VmAccessOutcome::Complete;
}

VmAccessOutcome L2Cache::flush_line(uint64_t addr, uint32_t vmid) {
  auto maintenance_lock = acquire_cache_access();
  std::lock_guard set_lock(set_mutex(addr));
  return flush_line_locked(addr, vmid);
}

VmAccessOutcome L2Cache::flush_all(uint32_t vmid) {
  (void)vmid;
  auto maintenance_lock = acquire_cache_maintenance();
  const VmAccessOutcome outcome = flush_dirty_locked();
  if (outcome != VmAccessOutcome::Complete)
    return outcome;
  cache_.invalidate_all();
  clear_all_dirty_bytes();
  return VmAccessOutcome::Complete;
}

VmAccessOutcome L2Cache::flush_for_completion(const GpuVmAccess &access) {
  const uint32_t vmid = access.info().vmid;
  const uint64_t cache_identity = access.cache_identity();
  if (vmid == 0 || cache_identity == 0)
    return VmAccessOutcome::Malformed;
  if (!access.is_current())
    return VmAccessOutcome::Revoked;

  auto maintenance_lock = acquire_cache_maintenance();
  if (!access.is_current())
    return VmAccessOutcome::Revoked;
  VmAccessOutcome outcome = VmAccessOutcome::Complete;
  cache_.for_each_valid_line([&](simdojo::CacheTag &tag, uint64_t line_addr, uint8_t *data) {
    if (outcome != VmAccessOutcome::Complete ||
        !cache_tag_matches_completion_access(tag, vmid, cache_identity))
      return;
    if (!access.is_current()) {
      outcome = VmAccessOutcome::Revoked;
      return;
    }
    if (tag.dirty) {
      if (backing_memory_ == nullptr || gpu_vm_ == nullptr) {
        outcome = VmAccessOutcome::Malformed;
        return;
      }
      outcome = publish_dirty_bytes_for_completion(line_addr, data, tag, access);
      if (outcome != VmAccessOutcome::Complete)
        return;
    }
    cache_.invalidate(line_addr, tag.vmid, tag.address_space_identity);
  });
  return outcome == VmAccessOutcome::Complete && !access.is_current() ? VmAccessOutcome::Revoked
                                                                      : outcome;
}

VmAccessOutcome L2Cache::flush_dirty_locked() {
  if (!has_dirty_lines_.load(std::memory_order_relaxed))
    return VmAccessOutcome::Complete;

  uint32_t dirty_count = 0;
  uint64_t min_addr = std::numeric_limits<uint64_t>::max(), max_addr = 0;
  VmAccessOutcome outcome = VmAccessOutcome::Complete;
  cache_.for_each_dirty([this, &dirty_count, &min_addr, &max_addr,
                         &outcome](simdojo::CacheTag &tag, uint64_t line_addr, uint8_t *data) {
    if (outcome != VmAccessOutcome::Complete)
      return;
    if (!retain_admitted_dirty_bytes(line_addr, tag)) {
      (void)clear_dirty_bytes(line_addr, 0, LINE_SIZE, tag.vmid);
      tag.dirty = false;
      return;
    }
    // Each dirty line is written back under its own owning vmid.
    outcome = publish_dirty_bytes(line_addr, data, tag.vmid);
    if (outcome != VmAccessOutcome::Complete)
      return;
    tag.dirty = false;
    ++dirty_count;
    if (line_addr < min_addr)
      min_addr = line_addr;
    if (line_addr > max_addr)
      max_addr = line_addr;
  });
  if (dirty_count != 0)
    util::Logger::vm("L2 flush: ", dirty_count, " dirty lines [0x", std::hex, min_addr, "-0x",
                     max_addr, "]", std::dec, " total_writes=", write_count());
  return outcome;
}

VmAccessOutcome L2Cache::flush_dirty_to_legacy_backing_locked() {
  if (!has_dirty_lines_.load(std::memory_order_relaxed))
    return VmAccessOutcome::Complete;
  if (legacy_maintenance_memory_ == nullptr || legacy_maintenance_vm_ == nullptr)
    throw util::ConfigError("L2 cache has no legacy maintenance backing");

  VmAccessOutcome outcome = VmAccessOutcome::Complete;
  cache_.for_each_dirty(
      [this, &outcome](simdojo::CacheTag &tag, uint64_t line_addr, uint8_t *data) {
        if (outcome != VmAccessOutcome::Complete)
          return;
        if (!retain_admitted_dirty_bytes(line_addr, tag)) {
          (void)clear_dirty_bytes(line_addr, 0, LINE_SIZE, tag.vmid);
          tag.dirty = false;
          return;
        }
        outcome = publish_dirty_bytes_to_legacy_backing(line_addr, data, tag.vmid);
        if (outcome == VmAccessOutcome::Complete)
          tag.dirty = false;
      });
  return outcome;
}

VmAccessOutcome L2Cache::invalidate_range(uint64_t addr, uint32_t size, uint32_t vmid) {
  if (size == 0)
    return VmAccessOutcome::Complete;
  const uint64_t line_start = CacheStore::line_address(addr);
  const uint64_t requested_lines =
      (static_cast<uint64_t>(CacheStore::line_offset(addr)) + size + LINE_SIZE - 1) / LINE_SIZE;
  const uint64_t addressable_lines =
      (std::numeric_limits<uint64_t>::max() - line_start) / LINE_SIZE + 1;
  const uint64_t line_count = std::min(requested_lines, addressable_lines);

  if (line_count <= MAX_INVALIDATE_RANGE_SET_LOCKS) {
    auto maintenance_lock = acquire_cache_access();
    SetRangeLocks locks = lock_sets_for_range(line_start, line_count);
    return invalidate_range_locked(addr, size, vmid, line_start, line_count);
  }

  auto maintenance_lock = acquire_cache_maintenance();
  return invalidate_range_locked(addr, size, vmid, line_start, line_count);
}

VmAccessOutcome L2Cache::invalidate_range_locked(uint64_t addr, uint32_t size, uint32_t vmid,
                                                 uint64_t line_start, uint64_t line_count) {
  uint64_t remaining = size;
  for (uint64_t i = 0; i < line_count; ++i) {
    const uint64_t line_addr = line_start + i * LINE_SIZE;
    const uint32_t line_offset = i == 0 ? CacheStore::line_offset(addr) : 0;
    const uint32_t chunk =
        static_cast<uint32_t>(std::min<uint64_t>(remaining, LINE_SIZE - line_offset));

    simdojo::CacheTag *tag = nullptr;
    if (cache_.lookup(line_addr, &tag, vmid)) {
      if (tag->dirty) {
        if (!retain_admitted_dirty_bytes(line_addr, *tag)) {
          (void)clear_dirty_bytes(line_addr, 0, LINE_SIZE, vmid);
        } else {
          const uint8_t *dirty_data = cache_.line_data_for_read(line_addr, vmid);
          assert(dirty_data != nullptr);
          // The invalidated bytes are authoritative in backing memory. Publish
          // only dirty bytes outside that range before discarding the line.
          const VmAccessOutcome outcome =
              publish_dirty_bytes(line_addr, dirty_data, vmid, line_offset, chunk);
          if (outcome != VmAccessOutcome::Complete)
            return outcome;
        }
      }
      cache_.invalidate(line_addr, vmid);
    }
    remaining -= chunk;
  }
  return VmAccessOutcome::Complete;
}

VmAccessOutcome L2Cache::flush_external_line_locked(uint64_t line_addr, uint32_t vmid,
                                                    const GpuVmAccess *access) {
  simdojo::CacheTag *tag = nullptr;
  if (!cache_.lookup(line_addr, &tag, vmid))
    return VmAccessOutcome::Complete;
  if (tag->dirty) {
    const uint8_t *data = cache_.line_data_for_read(line_addr, vmid);
    assert(data != nullptr);
    const VmAccessOutcome outcome =
        publish_dirty_bytes_before_external_write(line_addr, data, *tag, access);
    if (outcome != VmAccessOutcome::Complete)
      return outcome;
  }
  (void)clear_dirty_bytes(line_addr, 0, LINE_SIZE, vmid);
  cache_.invalidate(line_addr, vmid);
  return VmAccessOutcome::Complete;
}

VmAccessOutcome
L2Cache::flush_prepared_external_line_locked(uint64_t line_addr, uint64_t requested_begin,
                                             uint64_t requested_last, uint32_t vmid,
                                             const PreparedCanonicalWriteback &prepared) {
  simdojo::CacheTag *tag = nullptr;
  if (!cache_.lookup(line_addr, &tag, vmid))
    return VmAccessOutcome::Complete;

  const uint64_t covered_begin = std::max(line_addr, requested_begin);
  const uint64_t covered_last = std::min(line_addr + LINE_SIZE - 1, requested_last);
  if (!prepared.covers(covered_begin, covered_last - covered_begin + 1, tag->vmid,
                       tag->cache_identity, tag->admission_generation))
    return VmAccessOutcome::Unavailable;

  if (tag->dirty) {
    DirtyMask mask{};
    {
      std::lock_guard lock(dirty_bytes_mutex_);
      const auto position = dirty_bytes_.find({vmid, line_addr});
      if (position == dirty_bytes_.end())
        return VmAccessOutcome::Unavailable;
      mask = position->second;
    }
    const uint8_t *data = cache_.line_data_for_read(line_addr, vmid);
    assert(data != nullptr);
    for (uint32_t byte = 0; byte < LINE_SIZE;) {
      while (byte < LINE_SIZE && (mask[byte / 64] & (uint64_t{1} << (byte % 64))) == 0)
        ++byte;
      const uint32_t run_begin = byte;
      while (byte < LINE_SIZE && (mask[byte / 64] & (uint64_t{1} << (byte % 64))) != 0)
        ++byte;
      if (run_begin == byte)
        continue;
      if (!CacheStore::bytes_valid(*tag, run_begin, byte - run_begin))
        return VmAccessOutcome::Unavailable;
      const std::span<const uint8_t> run(data + run_begin, byte - run_begin);
      const VmAccessOutcome outcome =
          prepared.write(line_addr + run_begin, std::as_bytes(run), tag->vmid, tag->cache_identity,
                         tag->admission_generation);
      if (outcome != VmAccessOutcome::Complete)
        return outcome;
      // Gate entry has ended before the dirty-mask mutex can be acquired.
      (void)clear_dirty_bytes(line_addr, run_begin, byte - run_begin, vmid);
      diagnostics(line_addr + run_begin).backing_writes.fetch_add(1, std::memory_order_relaxed);
    }
  }
  cache_.invalidate(line_addr, vmid);
  return VmAccessOutcome::Complete;
}

void L2Cache::invalidate_external_line_locked(uint64_t line_addr, uint32_t vmid) {
  (void)clear_dirty_bytes(line_addr, 0, LINE_SIZE, vmid);
  cache_.invalidate(line_addr, vmid);
}

VmAccessOutcome L2Cache::flush_range_before_external_write(uint64_t addr, uint32_t size,
                                                           uint32_t vmid) {
  if (size == 0)
    return VmAccessOutcome::Complete;
  if (vmid != 0 && gpu_vm_ == nullptr)
    return VmAccessOutcome::Unavailable;
  // Capture before L2 admission. VM replacement can wait on the retained
  // access state's effect lock while holding the VM registry lock; taking a
  // fresh registry snapshot inside the cache gate would invert that order.
  const std::optional<GpuVmAccess> access = vmid != 0 ? gpu_vm_->snapshot_vmid(vmid) : std::nullopt;
  const uint64_t line_start = CacheStore::line_address(addr);
  const uint64_t remaining_address_space = std::numeric_limits<uint64_t>::max() - addr;
  const uint64_t last_byte = addr + std::min<uint64_t>(size - 1, remaining_address_space);
  const uint64_t line_end = CacheStore::line_address(last_byte);
  const uint64_t line_count = (line_end - line_start) / LINE_SIZE + 1;

  if (line_count <= MAX_INVALIDATE_RANGE_SET_LOCKS) {
    auto maintenance_lock = acquire_cache_access();
    SetRangeLocks locks = lock_sets_for_range(line_start, line_count);
    for (uint64_t line_addr = line_start;; line_addr += LINE_SIZE) {
      const VmAccessOutcome outcome =
          flush_external_line_locked(line_addr, vmid, access ? &*access : nullptr);
      if (outcome != VmAccessOutcome::Complete)
        return outcome;
      if (line_addr == line_end)
        break;
    }
    return VmAccessOutcome::Complete;
  }

  auto maintenance_lock = acquire_cache_maintenance();
  VmAccessOutcome outcome = VmAccessOutcome::Complete;
  cache_.for_each_valid_line([&](simdojo::CacheTag &tag, uint64_t line_addr, uint8_t *) {
    if (outcome == VmAccessOutcome::Complete && tag.vmid == vmid && line_addr >= line_start &&
        line_addr <= line_end)
      outcome = flush_external_line_locked(line_addr, vmid, access ? &*access : nullptr);
  });
  return outcome;
}

VmAccessOutcome
L2Cache::flush_range_before_external_write(uint64_t addr, uint32_t size, uint32_t vmid,
                                           const PreparedCanonicalWriteback &prepared) {
  if (size == 0)
    return VmAccessOutcome::Complete;
  if (prepared.outcome() != VmAccessOutcome::Complete)
    return prepared.outcome();
  const uint64_t line_start = CacheStore::line_address(addr);
  const uint64_t remaining_address_space = std::numeric_limits<uint64_t>::max() - addr;
  const uint64_t last_byte = addr + std::min<uint64_t>(size - 1, remaining_address_space);
  const uint64_t line_end = CacheStore::line_address(last_byte);
  const uint64_t line_count = (line_end - line_start) / LINE_SIZE + 1;

  if (line_count <= MAX_INVALIDATE_RANGE_SET_LOCKS) {
    auto maintenance_lock = acquire_cache_access();
    SetRangeLocks locks = lock_sets_for_range(line_start, line_count);
    for (uint64_t line_addr = line_start;; line_addr += LINE_SIZE) {
      const VmAccessOutcome outcome =
          flush_prepared_external_line_locked(line_addr, addr, last_byte, vmid, prepared);
      if (outcome != VmAccessOutcome::Complete)
        return outcome;
      if (line_addr == line_end)
        break;
    }
    return VmAccessOutcome::Complete;
  }

  auto maintenance_lock = acquire_cache_maintenance();
  VmAccessOutcome outcome = VmAccessOutcome::Complete;
  cache_.for_each_valid_line([&](simdojo::CacheTag &tag, uint64_t line_addr, uint8_t *) {
    if (outcome == VmAccessOutcome::Complete && tag.vmid == vmid && line_addr >= line_start &&
        line_addr <= line_end)
      outcome = flush_prepared_external_line_locked(line_addr, addr, last_byte, vmid, prepared);
  });
  return outcome;
}

void L2Cache::invalidate_range_after_external_write(uint64_t addr, uint32_t size, uint32_t vmid) {
  if (size == 0)
    return;
  const uint64_t line_start = CacheStore::line_address(addr);
  const uint64_t remaining_address_space = std::numeric_limits<uint64_t>::max() - addr;
  const uint64_t last_byte = addr + std::min<uint64_t>(size - 1, remaining_address_space);
  const uint64_t line_end = CacheStore::line_address(last_byte);
  const uint64_t line_count = (line_end - line_start) / LINE_SIZE + 1;

  if (line_count <= MAX_INVALIDATE_RANGE_SET_LOCKS) {
    auto maintenance_lock = acquire_cache_access();
    SetRangeLocks locks = lock_sets_for_range(line_start, line_count);
    for (uint64_t line_addr = line_start;; line_addr += LINE_SIZE) {
      invalidate_external_line_locked(line_addr, vmid);
      if (line_addr == line_end)
        break;
    }
    return;
  }

  auto maintenance_lock = acquire_cache_maintenance();
  cache_.for_each_valid_line([&](simdojo::CacheTag &tag, uint64_t line_addr, uint8_t *) {
    if (tag.vmid == vmid && line_addr >= line_start && line_addr <= line_end)
      invalidate_external_line_locked(line_addr, vmid);
  });
}

} // namespace amdgpu
} // namespace rocjitsu
