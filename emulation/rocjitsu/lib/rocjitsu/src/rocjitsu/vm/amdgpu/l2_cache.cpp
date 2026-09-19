// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
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

std::shared_lock<std::shared_mutex> L2Cache::acquire_cache_access() {
  for (;;) {
    std::shared_lock access_lock(maintenance_mutex_);
    if (coherence_epoch_ == coherence_->current_epoch())
      return access_lock;

    access_lock.unlock();
    std::unique_lock maintenance_lock(maintenance_mutex_);
    synchronize_epoch_locked();
  }
}

std::unique_lock<std::shared_mutex> L2Cache::acquire_cache_maintenance() {
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
      backing_write_transactions_.fetch_add(1, std::memory_order_relaxed);
    }
  }
  return VmAccessOutcome::Complete;
}

void L2Cache::clear_all_dirty_bytes() {
  std::lock_guard lock(dirty_bytes_mutex_);
  dirty_bytes_.clear();
  has_dirty_lines_.store(false, std::memory_order_relaxed);
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

VmAccessOutcome L2Cache::send_backing(uint64_t addr, uint8_t *data, uint32_t size,
                                      simdojo::MessageOp op, uint32_t vmid) {
  if (op == simdojo::MessageOp::WRITE)
    backing_write_transactions_.fetch_add(1, std::memory_order_relaxed);
  else
    backing_read_transactions_.fetch_add(1, std::memory_order_relaxed);
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
    std::optional<GpuVmAccess> vm_access = gpu_vm_->snapshot_vmid(vmid);
    if (!vm_access)
      return VmAccessOutcome::Faulted;
    const VmAccessOutcome outcome =
        op == simdojo::MessageOp::WRITE
            ? vm_access->write(addr, std::as_bytes(std::span<const uint8_t>(data, size)))
            : vm_access->read(addr, std::as_writable_bytes(std::span<uint8_t>(data, size)));
    return outcome;
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
  backing_read_transactions_.fetch_add(1, std::memory_order_relaxed);
  backing_write_transactions_.fetch_add(1, std::memory_order_relaxed);
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

VmAccessOutcome L2Cache::ensure_line(uint64_t addr, uint32_t vmid) {
  if (cache_.lookup(addr, nullptr, vmid))
    return VmAccessOutcome::Complete;

  uint64_t line_addr = CacheStore::line_address(addr);
  simdojo::CacheTag evicted;
  uint8_t evicted_data[LINE_SIZE];
  cache_.allocate(addr, vmid, &evicted, evicted_data);

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
    const VmAccessOutcome outcome = publish_dirty_bytes(evicted_addr, evicted_data, evicted.vmid);
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
  return VmAccessOutcome::Complete;
}

VmAccessOutcome L2Cache::read(uint64_t addr, uint8_t *dst, uint32_t size, Mtype mtype,
                              uint32_t vmid) {
  std::shared_lock<std::shared_mutex> maintenance_lock = acquire_cache_access();
  uint32_t copied = 0;
  if (mtype == Mtype::UC) {
    while (copied < size) {
      const uint64_t ea = addr + copied;
      const uint32_t line_offset = CacheStore::line_offset(ea);
      const uint32_t chunk = std::min(size - copied, LINE_SIZE - line_offset);

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
    std::lock_guard set_lock(set_mutex(ea));

    if (mtype == Mtype::CC) {
      // Coherently Cacheable: invalidate L2 line before refetch, mirroring
      // the L1 CC behavior. This ensures cross-XCD store visibility when
      // different L2s share a backing store. Dirty data is flushed first.
      const VmAccessOutcome outcome = flush_line_locked(ea, vmid);
      if (outcome != VmAccessOutcome::Complete)
        return outcome;
    }

    const VmAccessOutcome outcome = ensure_line(ea, vmid);
    if (outcome != VmAccessOutcome::Complete)
      return outcome;
    cache_.read_line(ea, dst + copied, line_offset, chunk, vmid);
    copied += chunk;
  }
  return VmAccessOutcome::Complete;
}

VmAccessOutcome L2Cache::write(uint64_t addr, const uint8_t *src, uint32_t size, Mtype mtype,
                               uint32_t vmid) {
  std::shared_lock<std::shared_mutex> maintenance_lock = acquire_cache_access();
  uint32_t copied = 0;
  if (mtype == Mtype::UC) {
    while (copied < size) {
      const uint64_t ea = addr + copied;
      const uint32_t line_offset = CacheStore::line_offset(ea);
      const uint32_t chunk = std::min(size - copied, LINE_SIZE - line_offset);

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
    std::lock_guard set_lock(set_mutex(ea));

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

    // Write through to backing store for all mtypes. In the simulator, GPU
    // virtual addresses are host-mapped (MAP_FIXED), so hipMemcpy reads from
    // backing store directly. Without write-through, RW-mtype stores remain
    // in L2 as dirty lines and never reach the host-visible pages, causing
    // stale reads on D2H copy.
    tag->dirty = clear_dirty_bytes(CacheStore::line_address(ea), line_offset, chunk, vmid);
    tag->coherence = tag->dirty ? simdojo::CoherenceState::MODIFIED
                                : (mtype == Mtype::CC ? simdojo::CoherenceState::SHARED
                                                      : simdojo::CoherenceState::EXCLUSIVE);

    write_count_.fetch_add(1, std::memory_order_relaxed);
    copied += chunk;
  }
  return VmAccessOutcome::Complete;
}

VmAccessOutcome L2Cache::fetch_line(uint64_t addr, uint8_t *line_buf, uint32_t vmid) {
  std::shared_lock<std::shared_mutex> maintenance_lock = acquire_cache_access();
  std::lock_guard set_lock(set_mutex(addr));
  uint64_t line_addr = CacheStore::line_address(addr);
  const uint8_t *line = cache_.line_data_for_read(line_addr, vmid);
  if (!line) {
    simdojo::CacheTag evicted;
    uint8_t evicted_data[LINE_SIZE];
    CacheStore::Allocation allocated =
        cache_.allocate_with_data(line_addr, vmid, &evicted, evicted_data);

    if (evicted.valid && evicted.dirty) {
      static constexpr uint32_t SET_INDEX_BITS = std::bit_width(NUM_SETS - 1);
      uint64_t evicted_addr =
          (evicted.tag << (LINE_SIZE_BITS + SET_INDEX_BITS)) |
          (static_cast<uint64_t>(CacheStore::set_index(line_addr)) << LINE_SIZE_BITS);
      const VmAccessOutcome outcome = publish_dirty_bytes(evicted_addr, evicted_data, evicted.vmid);
      if (outcome != VmAccessOutcome::Complete) {
        cache_.invalidate(line_addr, vmid);
        CacheStore::Allocation restored = cache_.allocate_with_data(evicted_addr, evicted.vmid);
        *restored.tag = evicted;
        std::memcpy(restored.data, evicted_data, LINE_SIZE);
        return outcome;
      }
    }

    const VmAccessOutcome outcome =
        send_backing(line_addr, allocated.data, LINE_SIZE, simdojo::MessageOp::READ, vmid);
    if (outcome != VmAccessOutcome::Complete) {
      cache_.invalidate(line_addr, vmid);
      return outcome;
    }
    line = allocated.data;
  }
  std::memcpy(line_buf, line, LINE_SIZE);
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
  std::shared_lock<std::shared_mutex> maintenance_lock = acquire_cache_access();
  std::lock_guard set_lock(set_mutex(line_addr));
  const VmAccessOutcome fill_outcome = ensure_line(line_addr, vmid);
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
    uint8_t line_buf[LINE_SIZE];
    cache_.read_line(addr, line_buf, 0, LINE_SIZE, vmid);
    uint64_t line_addr = CacheStore::line_address(addr);
    const VmAccessOutcome outcome = publish_dirty_bytes(line_addr, line_buf, vmid);
    if (outcome != VmAccessOutcome::Complete)
      return outcome;
  }
  cache_.invalidate(addr, vmid);
  return VmAccessOutcome::Complete;
}

VmAccessOutcome L2Cache::flush_line(uint64_t addr, uint32_t vmid) {
  std::shared_lock<std::shared_mutex> maintenance_lock = acquire_cache_access();
  std::lock_guard set_lock(set_mutex(addr));
  return flush_line_locked(addr, vmid);
}

VmAccessOutcome L2Cache::flush_all(uint32_t vmid) {
  (void)vmid;
  std::unique_lock<std::shared_mutex> maintenance_lock = acquire_cache_maintenance();
  const VmAccessOutcome outcome = flush_dirty_locked();
  if (outcome != VmAccessOutcome::Complete)
    return outcome;
  cache_.invalidate_all();
  clear_all_dirty_bytes();
  return VmAccessOutcome::Complete;
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
                     max_addr, "]", std::dec,
                     " total_writes=", write_count_.load(std::memory_order_relaxed));
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
    std::shared_lock<std::shared_mutex> maintenance_lock = acquire_cache_access();
    SetRangeLocks locks = lock_sets_for_range(line_start, line_count);
    return invalidate_range_locked(addr, size, vmid, line_start, line_count);
  }

  std::unique_lock<std::shared_mutex> maintenance_lock = acquire_cache_maintenance();
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
        std::array<uint8_t, LINE_SIZE> dirty_data{};
        cache_.read_line(line_addr, dirty_data.data(), 0, LINE_SIZE, vmid);
        // The invalidated bytes are authoritative in backing memory. Publish
        // only dirty bytes outside that range before discarding the line.
        const VmAccessOutcome outcome =
            publish_dirty_bytes(line_addr, dirty_data.data(), vmid, line_offset, chunk);
        if (outcome != VmAccessOutcome::Complete)
          return outcome;
      }
      cache_.invalidate(line_addr, vmid);
    }
    remaining -= chunk;
  }
  return VmAccessOutcome::Complete;
}

} // namespace amdgpu
} // namespace rocjitsu
