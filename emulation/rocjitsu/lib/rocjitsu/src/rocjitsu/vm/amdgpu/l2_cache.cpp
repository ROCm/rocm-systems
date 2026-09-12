// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "util/log.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstring>
#include <limits>
#include <span>

namespace rocjitsu {
namespace amdgpu {

L2Cache::L2Cache(std::string name) : simdojo::Component(std::move(name)) {
  req_port_ = add_port(std::make_unique<simdojo::Port>("req", 0, this, simdojo::PortDirection::OUT,
                                                       simdojo::PortProtocol::MEMORY));
  coherence_epoch_ = DeviceCacheCoherence::instance().current_epoch();
  DeviceCacheCoherence::instance().register_l2_cache(this);
}

L2Cache::~L2Cache() { DeviceCacheCoherence::instance().unregister_l2_cache(this); }

std::shared_lock<std::shared_mutex> L2Cache::acquire_cache_access() {
  for (;;) {
    std::shared_lock access_lock(maintenance_mutex_);
    if (coherence_epoch_ == DeviceCacheCoherence::instance().current_epoch())
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
  const uint64_t current_epoch = DeviceCacheCoherence::instance().current_epoch();
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
  const auto key = std::pair{vmid, line_addr};
  const auto it = dirty_bytes_.find(key);
  if (it == dirty_bytes_.end())
    return false;

  for (uint32_t byte = offset; byte < offset + size; ++byte)
    it->second[byte / 64] &= ~(uint64_t{1} << (byte % 64));

  const bool line_remains_dirty =
      std::ranges::any_of(it->second, [](uint64_t word) { return word != 0; });
  if (!line_remains_dirty)
    dirty_bytes_.erase(it);
  has_dirty_lines_.store(!dirty_bytes_.empty(), std::memory_order_relaxed);
  return line_remains_dirty;
}

void L2Cache::publish_dirty_bytes(uint64_t line_addr, const uint8_t *data, uint32_t vmid,
                                  uint32_t discard_offset, uint32_t discard_size) {
  assert(discard_offset <= LINE_SIZE && discard_size <= LINE_SIZE - discard_offset);
  DirtyMask mask{};
  {
    std::lock_guard lock(dirty_bytes_mutex_);
    const auto it = dirty_bytes_.find({vmid, line_addr});
    assert(it != dirty_bytes_.end() && "dirty L2 line must have a byte mask");
    if (it == dirty_bytes_.end())
      return;
    mask = it->second;
    dirty_bytes_.erase(it);
    has_dirty_lines_.store(!dirty_bytes_.empty(), std::memory_order_relaxed);
  }

  for (uint32_t byte = discard_offset; byte < discard_offset + discard_size; ++byte)
    mask[byte / 64] &= ~(uint64_t{1} << (byte % 64));

  uint32_t offset = 0;
  while (offset < LINE_SIZE) {
    while (offset < LINE_SIZE && (mask[offset / 64] & (uint64_t{1} << (offset % 64))) == 0)
      ++offset;
    const uint32_t start = offset;
    while (offset < LINE_SIZE && (mask[offset / 64] & (uint64_t{1} << (offset % 64))) != 0)
      ++offset;
    if (start != offset)
      send_backing(line_addr + start, const_cast<uint8_t *>(data + start), offset - start,
                   simdojo::MessageOp::WRITE, vmid);
  }
}

void L2Cache::clear_all_dirty_bytes() {
  std::lock_guard lock(dirty_bytes_mutex_);
  dirty_bytes_.clear();
  has_dirty_lines_.store(false, std::memory_order_relaxed);
}

void L2Cache::send_backing(uint64_t addr, uint8_t *data, uint32_t size, simdojo::MessageOp op,
                           uint32_t vmid) {
  if (op == simdojo::MessageOp::WRITE)
    backing_write_transactions_.fetch_add(1, std::memory_order_relaxed);
  else
    backing_read_transactions_.fetch_add(1, std::memory_order_relaxed);
  if (backing_memory_) {
    if (op == simdojo::MessageOp::WRITE) {
      thread_local uint64_t wb_count = 0;
      if (++wb_count <= 3)
        util::Logger::vm("L2 writeback(backing) #", wb_count, " addr=0x", std::hex, addr,
                         " size=", std::dec, size);
      backing_memory_->write_block(addr, std::span<const uint8_t>(data, size), vmid);
    } else {
      backing_memory_->read_block(addr, std::span<uint8_t>(data, size), vmid);
    }
    return;
  }
  assert(req_port_ != nullptr && "L2Cache: req_port_ not set");
  assert(req_port_->link() != nullptr && "L2Cache: req port has no link");
  auto msg = std::make_unique<simdojo::Message>();
  auto &hdr = msg->header();
  hdr.addr = addr;
  hdr.size_bytes = size;
  hdr.op = op;
  hdr.vmid = vmid;
  msg->set_payload(reinterpret_cast<uintptr_t>(data));
  req_port_->send(std::move(msg));
}

void L2Cache::ensure_line(uint64_t addr, uint32_t vmid) {
  if (cache_.lookup(addr, nullptr, vmid))
    return;

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
    publish_dirty_bytes(evicted_addr, evicted_data, evicted.vmid);
  }

  uint8_t line_buf[LINE_SIZE];
  send_backing(line_addr, line_buf, LINE_SIZE, simdojo::MessageOp::READ, vmid);
  cache_.fill_line(addr, line_buf, vmid);
}

void L2Cache::read(uint64_t addr, uint8_t *dst, uint32_t size, Mtype mtype, uint32_t vmid) {
  auto maintenance_lock = acquire_cache_access();
  uint32_t copied = 0;
  if (mtype == Mtype::UC) {
    while (copied < size) {
      const uint64_t ea = addr + copied;
      const uint32_t line_offset = CacheStore::line_offset(ea);
      const uint32_t chunk = std::min(size - copied, LINE_SIZE - line_offset);

      std::lock_guard set_lock(set_mutex(ea));
      flush_line_locked(ea, vmid);
      send_backing(ea, dst + copied, chunk, simdojo::MessageOp::READ, vmid);
      copied += chunk;
    }
    return;
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
      flush_line_locked(ea, vmid);
    }

    ensure_line(ea, vmid);
    cache_.read_line(ea, dst + copied, line_offset, chunk, vmid);
    copied += chunk;
  }
}

void L2Cache::write(uint64_t addr, const uint8_t *src, uint32_t size, Mtype mtype, uint32_t vmid) {
  auto maintenance_lock = acquire_cache_access();
  uint32_t copied = 0;
  if (mtype == Mtype::UC) {
    while (copied < size) {
      const uint64_t ea = addr + copied;
      const uint32_t line_offset = CacheStore::line_offset(ea);
      const uint32_t chunk = std::min(size - copied, LINE_SIZE - line_offset);

      std::lock_guard set_lock(set_mutex(ea));
      flush_line_locked(ea, vmid);
      send_backing(ea, const_cast<uint8_t *>(src + copied), chunk, simdojo::MessageOp::WRITE, vmid);
      copied += chunk;
    }
    return;
  }

  copied = 0;
  while (copied < size) {
    const uint64_t ea = addr + copied;
    const uint32_t line_offset = CacheStore::line_offset(ea);
    const uint32_t chunk = std::min(size - copied, LINE_SIZE - line_offset);
    std::lock_guard set_lock(set_mutex(ea));

    simdojo::CacheTag *tag = nullptr;
    // A full-line miss has no old bytes to preserve, so allocate it without a
    // read-for-ownership from backing memory.
    if (line_offset == 0 && chunk == LINE_SIZE) {
      if (!cache_.lookup(ea, &tag, vmid)) {
        simdojo::CacheTag evicted;
        uint8_t evicted_data[LINE_SIZE];
        auto allocated = cache_.allocate_with_data(ea, vmid, &evicted, evicted_data);
        if (evicted.valid && evicted.dirty) {
          static constexpr uint32_t SET_INDEX_BITS = std::bit_width(NUM_SETS - 1);
          const uint64_t evicted_addr =
              (evicted.tag << (LINE_SIZE_BITS + SET_INDEX_BITS)) |
              (static_cast<uint64_t>(CacheStore::set_index(ea)) << LINE_SIZE_BITS);
          publish_dirty_bytes(evicted_addr, evicted_data, evicted.vmid);
        }
        std::memcpy(allocated.data, src + copied, LINE_SIZE);
        tag = allocated.tag;
      } else {
        cache_.write_line(ea, src + copied, line_offset, chunk, vmid);
      }
    } else {
      ensure_line(ea, vmid);
      cache_.write_line(ea, src + copied, line_offset, chunk, vmid);
      cache_.lookup(ea, &tag, vmid);
    }
    assert(tag != nullptr && "ensure_line must guarantee hit");

    // Write through to backing store for all mtypes. In the simulator, GPU
    // virtual addresses are host-mapped (MAP_FIXED), so hipMemcpy reads from
    // backing store directly. Without write-through, RW-mtype stores remain
    // in L2 as dirty lines and never reach the host-visible pages, causing
    // stale reads on D2H copy.
    send_backing(ea, const_cast<uint8_t *>(src + copied), chunk, simdojo::MessageOp::WRITE, vmid);
    tag->dirty = clear_dirty_bytes(CacheStore::line_address(ea), line_offset, chunk, vmid);
    tag->coherence = tag->dirty ? simdojo::CoherenceState::MODIFIED
                                : (mtype == Mtype::CC ? simdojo::CoherenceState::SHARED
                                                      : simdojo::CoherenceState::EXCLUSIVE);

    write_count_.fetch_add(1, std::memory_order_relaxed);
    copied += chunk;
  }
}

void L2Cache::fetch_line(uint64_t addr, uint8_t *line_buf, uint32_t vmid) {
  auto maintenance_lock = acquire_cache_access();
  std::lock_guard set_lock(set_mutex(addr));
  uint64_t line_addr = CacheStore::line_address(addr);
  const uint8_t *line = cache_.line_data_for_read(line_addr, vmid);
  if (!line) {
    simdojo::CacheTag evicted;
    uint8_t evicted_data[LINE_SIZE];
    auto allocated = cache_.allocate_with_data(line_addr, vmid, &evicted, evicted_data);

    if (evicted.valid && evicted.dirty) {
      static constexpr uint32_t SET_INDEX_BITS = std::bit_width(NUM_SETS - 1);
      uint64_t evicted_addr =
          (evicted.tag << (LINE_SIZE_BITS + SET_INDEX_BITS)) |
          (static_cast<uint64_t>(CacheStore::set_index(line_addr)) << LINE_SIZE_BITS);
      publish_dirty_bytes(evicted_addr, evicted_data, evicted.vmid);
    }

    send_backing(line_addr, allocated.data, LINE_SIZE, simdojo::MessageOp::READ, vmid);
    line = allocated.data;
  }
  std::memcpy(line_buf, line, LINE_SIZE);
}

void L2Cache::writeback_line(uint64_t line_addr, const uint8_t *data, Mtype mtype, uint32_t vmid) {
  writeback_line(line_addr, data, 0, LINE_SIZE, mtype, vmid);
}

void L2Cache::writeback_line(uint64_t line_addr, const uint8_t *data, uint32_t dirty_offset,
                             uint32_t dirty_size, Mtype mtype, uint32_t vmid) {
  assert(CacheStore::line_address(line_addr) == line_addr && "writeback address must be aligned");
  assert(dirty_offset < LINE_SIZE && dirty_size != 0 && dirty_size <= LINE_SIZE - dirty_offset &&
         "dirty range must fit in one L2 line");
  auto maintenance_lock = acquire_cache_access();
  std::lock_guard set_lock(set_mutex(line_addr));
  ensure_line(line_addr, vmid);
  simdojo::CacheTag *tag = nullptr;
  cache_.lookup(line_addr, &tag, vmid);
  assert(tag != nullptr && "ensure_line must guarantee hit");
  cache_.write_line(line_addr, data + dirty_offset, dirty_offset, dirty_size, vmid);

  if (mtype == Mtype::CC) {
    send_backing(line_addr + dirty_offset, const_cast<uint8_t *>(data + dirty_offset), dirty_size,
                 simdojo::MessageOp::WRITE, vmid);
    tag->dirty = clear_dirty_bytes(line_addr, dirty_offset, dirty_size, vmid);
    tag->coherence =
        tag->dirty ? simdojo::CoherenceState::MODIFIED : simdojo::CoherenceState::SHARED;
  } else {
    mark_dirty_bytes(line_addr, dirty_offset, dirty_size, vmid);
    tag->dirty = true;
    tag->coherence = simdojo::CoherenceState::MODIFIED;
  }
}

void L2Cache::flush_line_locked(uint64_t addr, uint32_t vmid) {
  simdojo::CacheTag *tag = nullptr;
  if (!cache_.lookup(addr, &tag, vmid))
    return;

  if (tag->dirty) {
    uint8_t line_buf[LINE_SIZE];
    cache_.read_line(addr, line_buf, 0, LINE_SIZE, vmid);
    uint64_t line_addr = CacheStore::line_address(addr);
    publish_dirty_bytes(line_addr, line_buf, vmid);
  }
  cache_.invalidate(addr, vmid);
}

void L2Cache::flush_line(uint64_t addr, uint32_t vmid) {
  auto maintenance_lock = acquire_cache_access();
  std::lock_guard set_lock(set_mutex(addr));
  flush_line_locked(addr, vmid);
}

void L2Cache::flush_all(uint32_t vmid) {
  (void)vmid;
  auto maintenance_lock = acquire_cache_maintenance();
  flush_dirty_locked();
  cache_.invalidate_all();
  clear_all_dirty_bytes();
}

void L2Cache::flush_dirty_locked() {
  if (!has_dirty_lines_.load(std::memory_order_relaxed))
    return;

  uint32_t dirty_count = 0;
  uint64_t min_addr = std::numeric_limits<uint64_t>::max(), max_addr = 0;
  cache_.for_each_dirty([this, &dirty_count, &min_addr,
                         &max_addr](simdojo::CacheTag &tag, uint64_t line_addr, uint8_t *data) {
    // Each dirty line is written back under its own owning vmid.
    publish_dirty_bytes(line_addr, data, tag.vmid);
    tag.dirty = false;
    ++dirty_count;
    if (line_addr < min_addr)
      min_addr = line_addr;
    if (line_addr > max_addr)
      max_addr = line_addr;
  });
  clear_all_dirty_bytes();
  if (dirty_count != 0)
    util::Logger::vm("L2 flush: ", dirty_count, " dirty lines [0x", std::hex, min_addr, "-0x",
                     max_addr, "]", std::dec,
                     " total_writes=", write_count_.load(std::memory_order_relaxed));
}

void L2Cache::invalidate_range(uint64_t addr, uint32_t size, uint32_t vmid) {
  if (size == 0)
    return;
  const uint64_t line_start = CacheStore::line_address(addr);
  const uint64_t requested_lines =
      (static_cast<uint64_t>(CacheStore::line_offset(addr)) + size + LINE_SIZE - 1) / LINE_SIZE;
  const uint64_t addressable_lines =
      (std::numeric_limits<uint64_t>::max() - line_start) / LINE_SIZE + 1;
  const uint64_t line_count = std::min(requested_lines, addressable_lines);

  if (line_count <= MAX_INVALIDATE_RANGE_SET_LOCKS) {
    auto maintenance_lock = acquire_cache_access();
    auto locks = lock_sets_for_range(line_start, line_count);
    invalidate_range_locked(addr, size, vmid, line_start, line_count);
    return;
  }

  auto maintenance_lock = acquire_cache_maintenance();
  invalidate_range_locked(addr, size, vmid, line_start, line_count);
}

void L2Cache::invalidate_range_locked(uint64_t addr, uint32_t size, uint32_t vmid,
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
        publish_dirty_bytes(line_addr, dirty_data.data(), vmid, line_offset, chunk);
      }
      cache_.invalidate(line_addr, vmid);
    }
    remaining -= chunk;
  }
}

} // namespace amdgpu
} // namespace rocjitsu
