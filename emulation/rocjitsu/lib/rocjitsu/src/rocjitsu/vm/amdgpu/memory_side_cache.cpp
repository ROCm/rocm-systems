// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/memory_side_cache.h"

#include "rocjitsu/vm/amdgpu/device_cache_coherence.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/gpu_vm.h"
#include "util/except.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstring>
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

VmAccessOutcome MemorySideCache::ensure_line(uint64_t addr, uint32_t vmid) {
  const uint64_t current_epoch = coherence_->current_epoch();
  simdojo::CacheTag *resident = nullptr;
  if (cache_.lookup(addr, &resident, vmid)) {
    if (resident->coherence_epoch == current_epoch)
      return VmAccessOutcome::Complete;
    assert(!resident->dirty && "stale write-through MSC line must be clean");
    cache_.invalidate(addr, vmid);
  }

  uint64_t line_addr = CacheStore::line_address(addr);
  simdojo::CacheTag evicted;
  uint8_t evicted_data[LINE_SIZE];
  simdojo::CacheTag *allocated = cache_.allocate(addr, vmid, &evicted, evicted_data);

  if (evicted.valid && evicted.dirty) {
    static constexpr uint32_t SET_INDEX_BITS = std::bit_width(NUM_SETS - 1);
    uint64_t evicted_addr = (evicted.tag << (LINE_SIZE_BITS + SET_INDEX_BITS)) |
                            (static_cast<uint64_t>(CacheStore::set_index(addr)) << LINE_SIZE_BITS);
    const VmAccessOutcome outcome = send_backing(evicted_addr, evicted_data, LINE_SIZE,
                                                 simdojo::MessageOp::WRITE, evicted.vmid);
    if (outcome != VmAccessOutcome::Complete) {
      cache_.invalidate(addr, vmid);
      CacheStore::Allocation restored = cache_.allocate_with_data(evicted_addr, evicted.vmid);
      *restored.tag = evicted;
      std::memcpy(restored.data, evicted_data, LINE_SIZE);
      return outcome;
    }
  }

  uint8_t line_buf[LINE_SIZE];
  const VmAccessOutcome outcome =
      send_backing(line_addr, line_buf, LINE_SIZE, simdojo::MessageOp::READ, vmid);
  if (outcome != VmAccessOutcome::Complete) {
    cache_.invalidate(addr, vmid);
    return outcome;
  }
  cache_.fill_line(addr, line_buf, vmid);
  allocated->coherence_epoch = current_epoch;
  return VmAccessOutcome::Complete;
}

VmAccessOutcome MemorySideCache::read(uint64_t addr, uint8_t *dst, uint32_t size, uint32_t vmid) {
  std::shared_lock access_lock(access_gate_);
  uint32_t copied = 0;
  while (copied < size) {
    const uint64_t ea = addr + copied;
    const uint32_t line_offset = CacheStore::line_offset(ea);
    const uint32_t chunk = std::min(size - copied, LINE_SIZE - line_offset);

    std::lock_guard<std::mutex> lock(stripes_[stripe_index(ea)]);
    const VmAccessOutcome outcome = ensure_line(ea, vmid);
    if (outcome != VmAccessOutcome::Complete)
      return outcome;
    cache_.read_line(ea, dst + copied, line_offset, chunk, vmid);
    copied += chunk;
  }
  return VmAccessOutcome::Complete;
}

VmAccessOutcome MemorySideCache::write(uint64_t addr, const uint8_t *src, uint32_t size,
                                       uint32_t vmid) {
  std::shared_lock access_lock(access_gate_);
  uint32_t copied = 0;
  while (copied < size) {
    const uint64_t ea = addr + copied;
    const uint32_t line_offset = CacheStore::line_offset(ea);
    const uint32_t chunk = std::min(size - copied, LINE_SIZE - line_offset);

    std::lock_guard<std::mutex> lock(stripes_[stripe_index(ea)]);
    const VmAccessOutcome fill_outcome = ensure_line(ea, vmid);
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
  return VmAccessOutcome::Complete;
}

VmAccessOutcome MemorySideCache::flush_dirty_locked() {
  VmAccessOutcome outcome = VmAccessOutcome::Complete;
  cache_.for_each_dirty(
      [this, &outcome](simdojo::CacheTag &tag, uint64_t line_addr, uint8_t *data) {
        if (outcome != VmAccessOutcome::Complete)
          return;
        outcome = send_backing(line_addr, data, LINE_SIZE, simdojo::MessageOp::WRITE, tag.vmid);
        if (outcome == VmAccessOutcome::Complete)
          tag.dirty = false;
      });
  return outcome;
}

VmAccessOutcome MemorySideCache::flush_dirty_to_legacy_backing_locked() {
  if (legacy_maintenance_memory_ == nullptr || legacy_maintenance_vm_ == nullptr)
    throw util::ConfigError("memory-side cache has no legacy maintenance backing");
  VmAccessOutcome outcome = VmAccessOutcome::Complete;
  cache_.for_each_dirty(
      [this, &outcome](simdojo::CacheTag &tag, uint64_t line_addr, uint8_t *data) {
        if (outcome != VmAccessOutcome::Complete)
          return;
        const std::span<const uint8_t> bytes(data, LINE_SIZE);
        if (tag.vmid == 0) {
          legacy_maintenance_memory_->write_block(line_addr, bytes);
          outcome = VmAccessOutcome::Complete;
        } else {
          std::optional<GpuVmAccess> vm_access = legacy_maintenance_vm_->snapshot_vmid(tag.vmid);
          outcome = vm_access ? vm_access->write(line_addr, std::as_bytes(bytes))
                              : VmAccessOutcome::Faulted;
        }
        // The MSC tracks dirtiness per line rather than per byte. Retaining the
        // whole line makes a later retry idempotently republish authoritative bytes.
        if (outcome == VmAccessOutcome::Complete)
          tag.dirty = false;
      });
  return outcome;
}

} // namespace amdgpu
} // namespace rocjitsu
