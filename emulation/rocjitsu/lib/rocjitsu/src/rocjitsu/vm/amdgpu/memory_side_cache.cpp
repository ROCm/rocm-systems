// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/memory_side_cache.h"
#include "rocjitsu/vm/amdgpu/prepared_canonical_writeback.h"

#include "rocjitsu/vm/amdgpu/device_cache_coherence.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/gpu_vm.h"
#include "util/except.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstring>
#include <limits>
#include <optional>
#include <shared_mutex>
#include <span>

namespace rocjitsu {
namespace amdgpu {

MemorySideCache::MemorySideCache(std::string name, std::shared_ptr<DeviceCacheCoherence> coherence,
                                 GpuMemory *legacy_maintenance_memory)
    : simdojo::Component(std::move(name)), coherence_(std::move(coherence)),
      legacy_maintenance_memory_(legacy_maintenance_memory) {
  if (!coherence_)
    throw util::ConfigError("memory-side cache requires a coherence domain");
  req_ = add_port(std::make_unique<simdojo::Port>("req", 0, this, simdojo::PortDirection::OUT,
                                                  simdojo::PortProtocol::MEMORY));
  coherence_->register_memory_side_cache(this);
}

MemorySideCache::~MemorySideCache() { coherence_->unregister_memory_side_cache(this); }

void MemorySideCache::set_coherence_domain(std::shared_ptr<DeviceCacheCoherence> coherence) {
  if (!coherence)
    throw util::ConfigError("memory-side cache requires a coherence domain");
  if (coherence == coherence_)
    return;
  coherence_->unregister_memory_side_cache(this);
  {
    std::unique_lock access_lock(access_gate_);
    cache_.invalidate_all();
    may_have_dirty_lines_.store(false, std::memory_order_relaxed);
  }
  coherence_ = std::move(coherence);
  coherence_->register_memory_side_cache(this);
}

void WriterPreferredAccessGate::lock_shared() {
  std::unique_lock lock(mutex_);
  cv_.wait(lock, [this] { return !writer_active_ && waiting_writers_ == 0; });
  ++active_readers_;
}

bool WriterPreferredAccessGate::try_lock_shared() {
  std::lock_guard lock(mutex_);
  if (writer_active_ || waiting_writers_ != 0)
    return false;
  ++active_readers_;
  return true;
}

void WriterPreferredAccessGate::unlock_shared() {
  bool notify_writer = false;
  {
    std::lock_guard lock(mutex_);
    assert(active_readers_ != 0 && "unlock_shared without an active reader");
    --active_readers_;
    notify_writer = active_readers_ == 0 && waiting_writers_ != 0;
  }
  if (notify_writer)
    cv_.notify_all();
}

void WriterPreferredAccessGate::lock() {
  std::unique_lock lock(mutex_);
  ++waiting_writers_;
  cv_.wait(lock, [this] { return !writer_active_ && active_readers_ == 0; });
  --waiting_writers_;
  writer_active_ = true;
}

bool WriterPreferredAccessGate::try_lock() {
  std::lock_guard lock(mutex_);
  if (writer_active_ || active_readers_ != 0 || waiting_writers_ != 0)
    return false;
  writer_active_ = true;
  return true;
}

void WriterPreferredAccessGate::unlock() {
  {
    std::lock_guard lock(mutex_);
    assert(writer_active_ && "unlock without an active writer");
    writer_active_ = false;
  }
  cv_.notify_all();
}

simdojo::MessageStatus MemorySideCache::message_status(VmAccessOutcome outcome) {
  switch (outcome) {
  case VmAccessOutcome::Complete:
    return simdojo::MessageStatus::Complete;
  case VmAccessOutcome::Unavailable:
    return simdojo::MessageStatus::Unavailable;
  case VmAccessOutcome::Faulted:
    return simdojo::MessageStatus::Faulted;
  case VmAccessOutcome::Malformed:
    return simdojo::MessageStatus::Malformed;
  case VmAccessOutcome::Revoked:
    return simdojo::MessageStatus::Faulted;
  }
  return simdojo::MessageStatus::Malformed;
}

VmAccessOutcome MemorySideCache::access_outcome(simdojo::MessageStatus status) {
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

VmAccessOutcome MemorySideCache::send_backing(uint64_t addr, uint8_t *data, uint32_t size,
                                              simdojo::MessageOp op, uint32_t vmid) {
  GpuVm *const vm = cache_vm();
  if (op == simdojo::MessageOp::WRITE && vm != nullptr &&
      vm->controlled_cache_backing_overlaps(vmid, addr, size))
    return VmAccessOutcome::Unavailable;
  assert(req_ != nullptr && "MemorySideCache: req_ not set");
  if (req_->link() == nullptr || req_->link()->exec_mode() != simdojo::ExecMode::FUNCTIONAL)
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
  req_->send(std::move(message));
  return access_outcome(completion_status);
}

VmAccessOutcome MemorySideCache::send_atomic_backing(uint64_t addr, uint32_t size,
                                                     const simdojo::MemoryAtomicMutation &mutation,
                                                     uint32_t vmid) {
  assert(req_ != nullptr && "MemorySideCache: req_ not set");
  if (req_->link() == nullptr || req_->link()->exec_mode() != simdojo::ExecMode::FUNCTIONAL)
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
  req_->send(std::move(message));
  return access_outcome(completion_status);
}

VmAccessAdmission MemorySideCache::admit_access(uint64_t addr, uint32_t size, uint32_t vmid,
                                                VmAccessKind access, bool report_fault) const {
  GpuVm *vm = cache_vm();
  if (vmid == 0 || vm == nullptr)
    return {};
  const std::optional<GpuVmAccess> vm_access = vm->snapshot_vmid(vmid);
  return vm_access ? vm_access->admit_access(addr, size, access, report_fault)
                   : VmAccessAdmission{.outcome = VmAccessOutcome::Faulted};
}

bool MemorySideCache::try_read_peer_cached(uint64_t addr, std::span<uint8_t> dst, uint32_t vmid,
                                           uint64_t admission_generation, uint64_t cache_identity) {
  const uint32_t offset = CacheStore::line_offset(addr);
  if (dst.empty() || dst.size() > LINE_SIZE - offset ||
      (vmid == 0 ? (admission_generation != 0 || cache_identity != 0)
                 : (admission_generation == 0 || cache_identity == 0)))
    return false;

  std::shared_lock access_lock(access_gate_, std::try_to_lock);
  if (!access_lock.owns_lock())
    return false;
  std::unique_lock stripe_lock(stripes_[stripe_index(addr)], std::try_to_lock);
  if (!stripe_lock.owns_lock())
    return false;
  const simdojo::CacheTag *tag = cache_.peek(addr, vmid);
  if (tag == nullptr || tag->admission_generation != admission_generation ||
      tag->cache_identity != cache_identity ||
      !CacheStore::bytes_valid(*tag, offset, static_cast<uint32_t>(dst.size())))
    return false;
  cache_.read_line(addr, dst.data(), offset, static_cast<uint32_t>(dst.size()), vmid);
  return true;
}

bool MemorySideCache::try_fill_peer_cached(uint64_t addr, std::span<const uint8_t> src,
                                           uint32_t vmid, uint64_t admission_generation,
                                           uint64_t cache_identity) {
  const uint32_t offset = CacheStore::line_offset(addr);
  if (src.empty() || src.size() > LINE_SIZE - offset ||
      (vmid == 0 ? (admission_generation != 0 || cache_identity != 0)
                 : (admission_generation == 0 || cache_identity == 0)))
    return false;

  std::shared_lock access_lock(access_gate_, std::try_to_lock);
  if (!access_lock.owns_lock())
    return false;
  std::unique_lock stripe_lock(stripes_[stripe_index(addr)], std::try_to_lock);
  if (!stripe_lock.owns_lock())
    return false;
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

bool MemorySideCache::guarded_dirty_allocation_victim(uint64_t replacement_addr) const {
  const simdojo::CacheTag *victim = cache_.full_set_victim(replacement_addr);
  GpuVm *const vm = cache_vm();
  if (victim == nullptr || !victim->dirty || vm == nullptr)
    return false;
  static constexpr uint32_t SET_INDEX_BITS = std::bit_width(NUM_SETS - 1);
  const uint64_t victim_addr =
      (victim->tag << (LINE_SIZE_BITS + SET_INDEX_BITS)) |
      (static_cast<uint64_t>(CacheStore::set_index(replacement_addr)) << LINE_SIZE_BITS);
  const std::optional<GpuVmAccess> access = vm->snapshot_vmid(victim->vmid);
  if (!access || victim->cache_identity == 0 || victim->cache_identity != access->cache_identity())
    return false;
  const VmAccessAdmission admission = access->query_cache_write_grant(victim_addr, LINE_SIZE);
  if (!admission || !admission.cacheable || admission.generation != victim->admission_generation)
    return false;
  return vm->controlled_cache_backing_overlaps(victim->vmid, victim_addr, LINE_SIZE);
}

VmAccessOutcome MemorySideCache::ensure_line(uint64_t addr, uint32_t vmid,
                                             uint64_t admission_generation,
                                             const GpuVmAccess *access, bool fetch_on_miss) {
  if (vmid != 0 && access == nullptr)
    return VmAccessOutcome::Unavailable;
  if (access != nullptr && !access->is_current())
    return VmAccessOutcome::Revoked;
  const uint64_t cache_identity = access != nullptr ? access->cache_identity() : 0;
  const uint64_t current_epoch = coherence_->current_epoch();
  simdojo::CacheTag *resident = nullptr;
  if (cache_.lookup(addr, &resident, vmid)) {
    if (resident->coherence_epoch == current_epoch &&
        resident->admission_generation == admission_generation &&
        resident->cache_identity == cache_identity &&
        (!fetch_on_miss || CacheStore::bytes_valid(*resident, 0, LINE_SIZE)))
      return VmAccessOutcome::Complete;
    if (resident->coherence_epoch != current_epoch ||
        resident->admission_generation != admission_generation ||
        resident->cache_identity != cache_identity) {
      assert((!resident->dirty || resident->cache_identity != cache_identity) &&
             "stale write-through MSC line must be clean");
      cache_.invalidate(addr, vmid);
      resident = nullptr;
    } else {
      uint8_t line_buf[LINE_SIZE];
      const VmAccessOutcome outcome = send_backing(CacheStore::line_address(addr), line_buf,
                                                   LINE_SIZE, simdojo::MessageOp::READ, vmid);
      if (outcome != VmAccessOutcome::Complete)
        return outcome;
      if (access && !access->is_current()) {
        cache_.invalidate(addr, vmid);
        return VmAccessOutcome::Revoked;
      }
      cache_.fill_missing_bytes(addr, line_buf, vmid);
      return VmAccessOutcome::Complete;
    }
  }

  if (guarded_dirty_allocation_victim(addr))
    return VmAccessOutcome::Unavailable;
  uint64_t line_addr = CacheStore::line_address(addr);
  simdojo::CacheTag evicted;
  uint8_t evicted_data[LINE_SIZE];
  simdojo::CacheTag *allocated = cache_.allocate(addr, vmid, &evicted, evicted_data);
  allocated->admission_generation = admission_generation;
  allocated->cache_identity = cache_identity;

  if (evicted.valid && evicted.dirty) {
    static constexpr uint32_t SET_INDEX_BITS = std::bit_width(NUM_SETS - 1);
    uint64_t evicted_addr = (evicted.tag << (LINE_SIZE_BITS + SET_INDEX_BITS)) |
                            (static_cast<uint64_t>(CacheStore::set_index(addr)) << LINE_SIZE_BITS);
    const VmAccessOutcome outcome = publish_dirty_line(evicted_addr, evicted, evicted_data);
    if (outcome != VmAccessOutcome::Complete) {
      cache_.invalidate(addr, vmid);
      CacheStore::Allocation restored = cache_.allocate_with_data(evicted_addr, evicted.vmid);
      may_have_dirty_lines_.store(true, std::memory_order_relaxed);
      *restored.tag = evicted;
      std::memcpy(restored.data, evicted_data, LINE_SIZE);
      return outcome;
    }
  }

  if (!fetch_on_miss) {
    if (access && !access->is_current()) {
      cache_.invalidate(addr, vmid);
      return VmAccessOutcome::Revoked;
    }
    CacheStore::clear_valid_bytes(*allocated);
    allocated->coherence_epoch = current_epoch;
    return VmAccessOutcome::Complete;
  }

  uint8_t line_buf[LINE_SIZE];
  const VmAccessOutcome outcome =
      send_backing(line_addr, line_buf, LINE_SIZE, simdojo::MessageOp::READ, vmid);
  if (outcome != VmAccessOutcome::Complete) {
    cache_.invalidate(addr, vmid);
    return outcome;
  }
  if (access != nullptr && !access->is_current()) {
    cache_.invalidate(addr, vmid);
    return VmAccessOutcome::Revoked;
  }
  cache_.fill_line(addr, line_buf, vmid);
  allocated->coherence_epoch = current_epoch;
  return VmAccessOutcome::Complete;
}

VmAccessOutcome MemorySideCache::cache_partial_bytes(uint64_t addr, const uint8_t *src,
                                                     uint32_t size, uint32_t vmid,
                                                     uint64_t admission_generation,
                                                     const GpuVmAccess *access) {
  if (vmid != 0 && access == nullptr)
    return VmAccessOutcome::Unavailable;
  if (access != nullptr && !access->is_current())
    return VmAccessOutcome::Revoked;
  const uint64_t cache_identity = access != nullptr ? access->cache_identity() : 0;
  const uint64_t current_epoch = coherence_->current_epoch();
  simdojo::CacheTag *tag = nullptr;
  if (cache_.lookup(addr, &tag, vmid) &&
      (tag->coherence_epoch != current_epoch || tag->admission_generation != admission_generation ||
       tag->cache_identity != cache_identity)) {
    assert((!tag->dirty || tag->cache_identity != cache_identity) &&
           "stale write-through MSC line must be clean");
    cache_.invalidate(addr, vmid);
    tag = nullptr;
  }
  if (tag == nullptr) {
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
      const VmAccessOutcome outcome = publish_dirty_line(evicted_addr, evicted, evicted_data);
      if (outcome != VmAccessOutcome::Complete) {
        cache_.invalidate(addr, vmid);
        CacheStore::Allocation restored = cache_.allocate_with_data(evicted_addr, evicted.vmid);
        may_have_dirty_lines_.store(true, std::memory_order_relaxed);
        *restored.tag = evicted;
        std::memcpy(restored.data, evicted_data, LINE_SIZE);
        return outcome;
      }
    }
    tag = allocated.tag;
    tag->admission_generation = admission_generation;
    tag->cache_identity = cache_identity;
    CacheStore::clear_valid_bytes(*tag);
  }

  if (access != nullptr && !access->is_current()) {
    cache_.invalidate(addr, vmid);
    return VmAccessOutcome::Revoked;
  }

  cache_.write_line(addr, src, CacheStore::line_offset(addr), size, vmid);
  tag->coherence_epoch = current_epoch;
  tag->dirty = false;
  tag->coherence = simdojo::CoherenceState::EXCLUSIVE;
  return VmAccessOutcome::Complete;
}

VmAccessOutcome MemorySideCache::read(uint64_t addr, uint8_t *dst, uint32_t size, uint32_t vmid) {
  if (size == 0)
    return VmAccessOutcome::Complete;
  const VmAccessAdmission admission = admit_access(addr, size, vmid, VmAccessKind::Read);
  if (!admission)
    return admission.outcome;
  std::shared_lock access_lock(access_gate_);
  GpuVm *const vm = cache_vm();
  uint32_t copied = 0;
  while (copied < size) {
    const uint64_t ea = addr + copied;
    const uint32_t line_offset = CacheStore::line_offset(ea);
    const uint32_t chunk = std::min(size - copied, LINE_SIZE - line_offset);
    const std::optional<GpuVmAccess> access =
        vmid != 0 && vm != nullptr ? vm->snapshot_vmid(vmid) : std::nullopt;
    if (vmid != 0 && vm != nullptr && !access)
      return VmAccessOutcome::Faulted;
    const VmAccessAdmission chunk_admission =
        access ? access->admit_access(ea, chunk, VmAccessKind::Read) : VmAccessAdmission{};
    if (!chunk_admission)
      return chunk_admission.outcome;
    const VmAccessAdmission line_admission =
        access ? access->admit_access(CacheStore::line_address(ea), LINE_SIZE, VmAccessKind::Read,
                                      /*report_fault=*/false)
               : VmAccessAdmission{};
    const bool cache_line = (vmid == 0 || access.has_value()) && line_admission &&
                            line_admission.cacheable &&
                            line_admission.generation == chunk_admission.generation;
    const uint64_t cache_identity = access ? access->cache_identity() : 0;

    std::lock_guard<std::mutex> lock(stripes_[stripe_index(ea)]);
    if (!cache_line) {
      simdojo::CacheTag *tag = nullptr;
      if (cache_.lookup(ea, &tag, vmid)) {
        if (tag->dirty)
          return VmAccessOutcome::Faulted;
        cache_.invalidate(ea, vmid);
      }
      const VmAccessOutcome outcome =
          send_backing(ea, dst + copied, chunk, simdojo::MessageOp::READ, vmid);
      if (outcome != VmAccessOutcome::Complete)
        return outcome;
      copied += chunk;
      continue;
    }
    simdojo::CacheTag *resident = nullptr;
    if (cache_.lookup(ea, &resident, vmid) &&
        resident->coherence_epoch == coherence_->current_epoch() &&
        resident->admission_generation == chunk_admission.generation &&
        resident->cache_identity == cache_identity &&
        CacheStore::bytes_valid(*resident, line_offset, chunk)) {
      cache_.read_line(ea, dst + copied, line_offset, chunk, vmid);
      copied += chunk;
      continue;
    }
    if (access && access->query_access(CacheStore::line_address(ea), LINE_SIZE,
                                       VmAccessKind::Read) != VmAccessOutcome::Complete) {
      const VmAccessOutcome outcome =
          send_backing(ea, dst + copied, chunk, simdojo::MessageOp::READ, vmid);
      if (outcome != VmAccessOutcome::Complete)
        return outcome;
      const VmAccessOutcome cache_outcome = cache_partial_bytes(
          ea, dst + copied, chunk, vmid, chunk_admission.generation, access ? &*access : nullptr);
      if (cache_outcome != VmAccessOutcome::Complete)
        return cache_outcome;
      copied += chunk;
      continue;
    }
    const VmAccessOutcome outcome =
        ensure_line(ea, vmid, chunk_admission.generation, access ? &*access : nullptr);
    if (outcome != VmAccessOutcome::Complete)
      return outcome;
    cache_.read_line(ea, dst + copied, line_offset, chunk, vmid);
    copied += chunk;
  }
  return VmAccessOutcome::Complete;
}

VmAccessOutcome MemorySideCache::write(uint64_t addr, const uint8_t *src, uint32_t size,
                                       uint32_t vmid) {
  if (size == 0)
    return VmAccessOutcome::Complete;
  const VmAccessAdmission admission = admit_access(addr, size, vmid, VmAccessKind::Write);
  if (!admission)
    return admission.outcome;
  GpuVm *const vm = cache_vm();
  if (vm != nullptr && vm->controlled_cache_backing_overlaps(vmid, addr, size))
    return VmAccessOutcome::Unavailable;
  std::shared_lock access_lock(access_gate_);
  uint32_t copied = 0;
  while (copied < size) {
    const uint64_t ea = addr + copied;
    const uint32_t line_offset = CacheStore::line_offset(ea);
    const uint32_t chunk = std::min(size - copied, LINE_SIZE - line_offset);
    const std::optional<GpuVmAccess> access =
        vmid != 0 && vm != nullptr ? vm->snapshot_vmid(vmid) : std::nullopt;
    if (vmid != 0 && vm != nullptr && !access)
      return VmAccessOutcome::Faulted;
    const VmAccessAdmission chunk_admission =
        access ? access->admit_access(ea, chunk, VmAccessKind::Write) : VmAccessAdmission{};
    if (!chunk_admission)
      return chunk_admission.outcome;
    const VmAccessAdmission line_admission =
        access ? access->admit_access(CacheStore::line_address(ea), LINE_SIZE, VmAccessKind::Write,
                                      /*report_fault=*/false)
               : VmAccessAdmission{};
    const bool cache_line = (vmid == 0 || access.has_value()) && line_admission &&
                            line_admission.cacheable &&
                            line_admission.generation == chunk_admission.generation;

    std::lock_guard<std::mutex> lock(stripes_[stripe_index(ea)]);
    if (!cache_line) {
      simdojo::CacheTag *tag = nullptr;
      if (cache_.lookup(ea, &tag, vmid)) {
        if (tag->dirty)
          return VmAccessOutcome::Faulted;
        cache_.invalidate(ea, vmid);
      }
      const VmAccessOutcome outcome = send_backing(ea, const_cast<uint8_t *>(src + copied), chunk,
                                                   simdojo::MessageOp::WRITE, vmid);
      if (outcome != VmAccessOutcome::Complete)
        return outcome;
      copied += chunk;
      continue;
    }
    const VmAccessOutcome fill_outcome =
        ensure_line(ea, vmid, chunk_admission.generation, access ? &*access : nullptr,
                    /*fetch_on_miss=*/false);
    if (fill_outcome != VmAccessOutcome::Complete)
      return fill_outcome;

    const VmAccessOutcome write_outcome = send_backing(ea, const_cast<uint8_t *>(src + copied),
                                                       chunk, simdojo::MessageOp::WRITE, vmid);
    if (write_outcome != VmAccessOutcome::Complete)
      return write_outcome;
    if (access && !access->is_current()) {
      cache_.invalidate(ea, vmid);
      return VmAccessOutcome::Revoked;
    }

    cache_.write_line(ea, src + copied, line_offset, chunk, vmid);

    simdojo::CacheTag *tag = nullptr;
    cache_.lookup(ea, &tag, vmid);
    assert(tag != nullptr && "ensure_line must guarantee hit");
    tag->dirty = false;
    tag->coherence = simdojo::CoherenceState::EXCLUSIVE;
    copied += chunk;
  }
  return VmAccessOutcome::Complete;
}

VmAccessOutcome MemorySideCache::atomic_modify(uint64_t addr, uint32_t size,
                                               const simdojo::MemoryAtomicMutation &mutation,
                                               uint32_t vmid) {
  if (size == 0 || CacheStore::line_offset(addr) + size > LINE_SIZE)
    return VmAccessOutcome::Malformed;
  const VmAccessAdmission admission = admit_access(addr, size, vmid, VmAccessKind::Atomic);
  if (!admission)
    return admission.outcome;
  std::unique_lock access_lock(access_gate_);
  return send_atomic_backing(addr, size, mutation, vmid);
}

VmAccessOutcome MemorySideCache::flush_all() {
  // Exclude reads and writes while iterating every set. Ordinary accesses hold
  // this gate in shared mode and retain their per-stripe concurrency. Once a
  // flush waits, the gate blocks new shared entrants so the flush makes progress.
  std::unique_lock access_lock(access_gate_);
  const VmAccessOutcome outcome = flush_dirty_locked();
  if (outcome != VmAccessOutcome::Complete)
    return outcome;
  cache_.invalidate_all();
  may_have_dirty_lines_.store(false, std::memory_order_relaxed);
  return VmAccessOutcome::Complete;
}

void MemorySideCache::mark_line_dirty(simdojo::CacheTag &tag) {
  may_have_dirty_lines_.store(true, std::memory_order_relaxed);
  tag.dirty = true;
}

VmAccessOutcome MemorySideCache::publish_dirty_line(uint64_t line_addr,
                                                    const simdojo::CacheTag &tag, uint8_t *data,
                                                    bool legacy_maintenance) {
  if (tag.vmid == 0) {
    if (legacy_maintenance) {
      legacy_maintenance_memory_->write_block(line_addr, std::span<const uint8_t>(data, LINE_SIZE));
      return VmAccessOutcome::Complete;
    }
    return send_backing(line_addr, data, LINE_SIZE, simdojo::MessageOp::WRITE, 0);
  }

  GpuVm *vm = legacy_maintenance ? legacy_maintenance_vm_ : cache_vm();
  if (vm == nullptr)
    return VmAccessOutcome::Unavailable;
  const std::optional<GpuVmAccess> access = vm->snapshot_vmid(tag.vmid);
  // Numeric VMIDs can be reused with the same translator admission generation.
  // A dirty line may only publish through the exact backing generation that
  // filled it; an old line belongs to a retired address space and is dropped.
  if (!access || tag.cache_identity == 0 || tag.cache_identity != access->cache_identity())
    return VmAccessOutcome::Complete;

  const VmAccessAdmission admission = access->query_cache_write_grant(line_addr, LINE_SIZE);
  if (admission.outcome == VmAccessOutcome::Unavailable)
    return admission.outcome;
  if (!admission || !admission.cacheable || admission.generation != tag.admission_generation)
    return VmAccessOutcome::Complete;

  if (vm->controlled_cache_backing_overlaps(tag.vmid, line_addr, LINE_SIZE))
    return VmAccessOutcome::Unavailable;

  // The same snapshot checks admission and performs the effect. Its access
  // state holds the revocation lock through write, so a concurrent rebind
  // cannot redirect these bytes into the replacement VMID's backing.
  const std::span<const uint8_t> bytes(data, LINE_SIZE);
  const VmAccessOutcome outcome = access->write(line_addr, std::as_bytes(bytes));
  return outcome == VmAccessOutcome::Revoked ? VmAccessOutcome::Complete : outcome;
}

template <typename Visitor>
void MemorySideCache::visit_resident_lines_in_range(uint64_t addr, uint32_t size, uint32_t vmid,
                                                    Visitor &&visitor) {
  assert(size != 0 && "range walk requires a nonempty range");
  const uint64_t line_start = CacheStore::line_address(addr);
  const uint64_t remaining_address_space = std::numeric_limits<uint64_t>::max() - addr;
  const uint64_t last_byte = addr + std::min<uint64_t>(size - 1, remaining_address_space);
  const uint64_t line_end = CacheStore::line_address(last_byte);
  const uint64_t line_count = (line_end - line_start) / LINE_SIZE + 1;

  // Small ranges visit their exact sets. A large range visits at most the
  // cache's occupied tag storage rather than every address in the range.
  if (line_count <= NUM_SETS) {
    for (uint64_t i = 0; i < line_count; ++i) {
      const uint64_t line_addr = line_start + i * LINE_SIZE;
      simdojo::CacheTag *tag = nullptr;
      if (!cache_.lookup(line_addr, &tag, vmid))
        continue;
      const uint8_t *data = cache_.line_data_for_read(line_addr, vmid);
      if (!visitor(line_addr, *tag, data))
        return;
    }
    return;
  }

  bool visit_more = true;
  cache_.for_each_valid_line([&](simdojo::CacheTag &tag, uint64_t line_addr, uint8_t *data) {
    if (visit_more && tag.vmid == vmid && line_addr >= line_start && line_addr <= line_end)
      visit_more = visitor(line_addr, tag, data);
  });
}

VmAccessOutcome MemorySideCache::flush_range(uint64_t addr, uint32_t size, uint32_t vmid) {
  if (size == 0)
    return VmAccessOutcome::Complete;

  // Range maintenance runs before an external writer changes backing bytes.
  // The whole dirty line is still authoritative at this point, including the
  // sibling bytes outside the pending write. The caller must serialize its
  // later write against alias refills after this gate is released.
  std::unique_lock access_lock(access_gate_);
  VmAccessOutcome outcome = VmAccessOutcome::Complete;
  visit_resident_lines_in_range(
      addr, size, vmid,
      [this, vmid, &outcome](uint64_t line_addr, simdojo::CacheTag &tag, const uint8_t *data) {
        if (tag.dirty) {
          outcome = publish_dirty_line(line_addr, tag, const_cast<uint8_t *>(data));
          if (outcome != VmAccessOutcome::Complete)
            return false;
        }
        cache_.invalidate(line_addr, vmid);
        return true;
      });
  return outcome;
}

VmAccessOutcome MemorySideCache::flush_range(uint64_t addr, uint32_t size, uint32_t vmid,
                                             const PreparedCanonicalWriteback &prepared) {
  if (size == 0)
    return VmAccessOutcome::Complete;
  if (prepared.outcome() != VmAccessOutcome::Complete)
    return prepared.outcome();

  std::unique_lock access_lock(access_gate_);
  VmAccessOutcome outcome = VmAccessOutcome::Complete;
  visit_resident_lines_in_range(
      addr, size, vmid,
      [this, vmid, &prepared, &outcome](uint64_t line_addr, simdojo::CacheTag &tag,
                                        const uint8_t *data) {
        // MSC dirtiness covers the whole line. One grant and one backing view
        // must authorize it before any backing byte can be changed.
        if (!prepared.covers(line_addr, LINE_SIZE, tag.vmid, tag.cache_identity,
                             tag.admission_generation)) {
          outcome = VmAccessOutcome::Unavailable;
          return false;
        }
        if (tag.dirty) {
          if (!CacheStore::bytes_valid(tag, 0, LINE_SIZE)) {
            outcome = VmAccessOutcome::Unavailable;
            return false;
          }
          const std::span<const uint8_t> line(data, LINE_SIZE);
          outcome = prepared.write(line_addr, std::as_bytes(line), tag.vmid, tag.cache_identity,
                                   tag.admission_generation);
          if (outcome != VmAccessOutcome::Complete)
            return false;
          // Gate entry has ended before the cache metadata is changed.
          tag.dirty = false;
        }
        cache_.invalidate(line_addr, vmid);
        return true;
      });
  return outcome;
}

void MemorySideCache::invalidate_range_after_external_write(uint64_t addr, uint32_t size,
                                                            uint32_t vmid) {
  if (size == 0)
    return;

  std::unique_lock access_lock(access_gate_);
  visit_resident_lines_in_range(
      addr, size, vmid, [this, vmid](uint64_t line_addr, simdojo::CacheTag &, const uint8_t *) {
        cache_.invalidate(line_addr, vmid);
        return true;
      });
}

VmAccessOutcome MemorySideCache::flush_dirty_locked() {
  if (!may_have_dirty_lines_.load(std::memory_order_relaxed)) {
    ++skipped_clean_scans_;
    return VmAccessOutcome::Complete;
  }
  ++full_dirty_scans_;
  VmAccessOutcome outcome = VmAccessOutcome::Complete;
  cache_.for_each_dirty(
      [this, &outcome](simdojo::CacheTag &tag, uint64_t line_addr, uint8_t *data) {
        if (outcome != VmAccessOutcome::Complete)
          return;
        outcome = publish_dirty_line(line_addr, tag, data);
        if (outcome == VmAccessOutcome::Complete)
          tag.dirty = false;
      });
  if (outcome == VmAccessOutcome::Complete)
    may_have_dirty_lines_.store(false, std::memory_order_relaxed);
  return outcome;
}

VmAccessOutcome MemorySideCache::flush_dirty_to_legacy_backing_locked() {
  if (legacy_maintenance_memory_ == nullptr || legacy_maintenance_vm_ == nullptr)
    throw util::ConfigError("memory-side cache has no legacy maintenance backing");
  if (!may_have_dirty_lines_.load(std::memory_order_relaxed)) {
    ++skipped_clean_scans_;
    return VmAccessOutcome::Complete;
  }
  ++full_dirty_scans_;
  VmAccessOutcome outcome = VmAccessOutcome::Complete;
  cache_.for_each_dirty(
      [this, &outcome](simdojo::CacheTag &tag, uint64_t line_addr, uint8_t *data) {
        if (outcome != VmAccessOutcome::Complete)
          return;
        outcome = publish_dirty_line(line_addr, tag, data, /*legacy_maintenance=*/true);
        // The MSC tracks dirtiness per line rather than per byte. Retaining the
        // whole line makes a later retry idempotently republish authoritative bytes.
        if (outcome == VmAccessOutcome::Complete)
          tag.dirty = false;
      });
  if (outcome == VmAccessOutcome::Complete)
    may_have_dirty_lines_.store(false, std::memory_order_relaxed);
  return outcome;
}

} // namespace amdgpu
} // namespace rocjitsu
