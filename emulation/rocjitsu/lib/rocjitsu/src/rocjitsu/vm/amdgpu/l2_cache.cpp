// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "util/log.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cassert>
#include <cstring>

namespace rocjitsu {
namespace amdgpu {

void L2Cache::send_backing(uint64_t addr, uint8_t *data, uint32_t size, simdojo::MessageOp op,
                           uint32_t vmid) {
  if (backing_memory_) {
    if (op == simdojo::MessageOp::WRITE) {
      static std::atomic<uint64_t> wb_count{0};
      const uint64_t count = wb_count.fetch_add(1, std::memory_order_relaxed) + 1;
      if (count <= 3)
        util::Logger::vm("L2 writeback(backing) #", count, " addr=0x", std::hex, addr,
                         " size=", std::dec, size);
      for (uint32_t i = 0; i < size; ++i)
        backing_memory_->write8(addr + i, data[i], vmid);
    } else {
      for (uint32_t i = 0; i < size; ++i)
        data[i] = backing_memory_->read8(addr + i, vmid);
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
    send_backing(evicted_addr, evicted_data, LINE_SIZE, simdojo::MessageOp::WRITE, evicted.vmid);
  }

  uint8_t line_buf[LINE_SIZE];
  send_backing(line_addr, line_buf, LINE_SIZE, simdojo::MessageOp::READ, vmid);
  cache_.fill_line(addr, line_buf, vmid);
}

void L2Cache::read(uint64_t addr, uint8_t *dst, uint32_t size, Mtype mtype, uint32_t vmid) {
  uint32_t copied = 0;
  if (mtype == Mtype::UC) {
    while (copied < size) {
      const uint64_t ea = addr + copied;
      const uint32_t line_offset = CacheStore::line_offset(ea);
      const uint32_t chunk = std::min(size - copied, LINE_SIZE - line_offset);

      flush_line(ea, vmid);
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

    if (mtype == Mtype::CC) {
      // Coherently Cacheable: invalidate L2 line before refetch, mirroring
      // the L1 CC behavior. This ensures cross-XCD store visibility when
      // different L2s share a backing store. Dirty data is flushed first.
      flush_line(ea, vmid);
    }

    ensure_line(ea, vmid);
    cache_.read_line(ea, dst + copied, line_offset, chunk, vmid);
    copied += chunk;
  }
}

void L2Cache::write(uint64_t addr, const uint8_t *src, uint32_t size, Mtype mtype, uint32_t vmid) {
  uint32_t copied = 0;
  if (mtype == Mtype::UC) {
    while (copied < size) {
      const uint64_t ea = addr + copied;
      const uint32_t line_offset = CacheStore::line_offset(ea);
      const uint32_t chunk = std::min(size - copied, LINE_SIZE - line_offset);

      flush_line(ea, vmid);
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

    ensure_line(ea, vmid);
    cache_.write_line(ea, src + copied, line_offset, chunk, vmid);

    simdojo::CacheTag *tag = nullptr;
    cache_.lookup(ea, &tag, vmid);
    assert(tag != nullptr && "ensure_line must guarantee hit");

    // Write through to backing store for all mtypes. In the simulator, GPU
    // virtual addresses are host-mapped (MAP_FIXED), so hipMemcpy reads from
    // backing store directly. Without write-through, RW-mtype stores remain
    // in L2 as dirty lines and never reach the host-visible pages, causing
    // stale reads on D2H copy.
    send_backing(ea, const_cast<uint8_t *>(src + copied), chunk, simdojo::MessageOp::WRITE, vmid);
    tag->coherence =
        (mtype == Mtype::CC) ? simdojo::CoherenceState::SHARED : simdojo::CoherenceState::EXCLUSIVE;
    tag->dirty = false;

    ++write_count_;
    copied += chunk;
  }
}

void L2Cache::fetch_line(uint64_t addr, uint8_t *line_buf, uint32_t vmid) {
  uint64_t line_addr = CacheStore::line_address(addr);
  ensure_line(line_addr, vmid);
  cache_.read_line(line_addr, line_buf, 0, LINE_SIZE, vmid);
}

void L2Cache::writeback_line(uint64_t line_addr, const uint8_t *data, Mtype mtype, uint32_t vmid) {
  simdojo::CacheTag *tag = nullptr;
  if (cache_.lookup(line_addr, &tag, vmid)) {
    cache_.write_line(line_addr, data, 0, LINE_SIZE, vmid);
  } else {
    simdojo::CacheTag evicted;
    uint8_t evicted_data[LINE_SIZE];
    cache_.allocate(line_addr, vmid, &evicted, evicted_data);

    if (evicted.valid && evicted.dirty) {
      static constexpr uint32_t SET_INDEX_BITS = std::bit_width(NUM_SETS - 1);
      uint64_t evicted_addr =
          (evicted.tag << (LINE_SIZE_BITS + SET_INDEX_BITS)) |
          (static_cast<uint64_t>(CacheStore::set_index(line_addr)) << LINE_SIZE_BITS);
      send_backing(evicted_addr, evicted_data, LINE_SIZE, simdojo::MessageOp::WRITE, evicted.vmid);
    }

    cache_.fill_line(line_addr, data, vmid);
    cache_.lookup(line_addr, &tag, vmid);
  }

  if (mtype == Mtype::CC) {
    send_backing(line_addr, const_cast<uint8_t *>(data), LINE_SIZE, simdojo::MessageOp::WRITE,
                 vmid);
    tag->coherence = simdojo::CoherenceState::SHARED;
  } else {
    tag->dirty = true;
    tag->coherence = simdojo::CoherenceState::MODIFIED;
  }
}

void L2Cache::flush_line(uint64_t addr, uint32_t vmid) {
  simdojo::CacheTag *tag = nullptr;
  if (!cache_.lookup(addr, &tag, vmid))
    return;

  if (tag->dirty) {
    uint8_t line_buf[LINE_SIZE];
    cache_.read_line(addr, line_buf, 0, LINE_SIZE, vmid);
    uint64_t line_addr = CacheStore::line_address(addr);
    send_backing(line_addr, line_buf, LINE_SIZE, simdojo::MessageOp::WRITE, vmid);
  }
  cache_.invalidate(addr, vmid);
}

void L2Cache::flush_all(uint32_t vmid) {
  (void)vmid;
  uint32_t dirty_count = 0;
  uint64_t min_addr = UINT64_MAX, max_addr = 0;
  cache_.for_each_dirty([this, &dirty_count, &min_addr,
                         &max_addr](simdojo::CacheTag &tag, uint64_t line_addr, uint8_t *data) {
    // Each dirty line is written back under its own owning vmid.
    send_backing(line_addr, data, LINE_SIZE, simdojo::MessageOp::WRITE, tag.vmid);
    tag.dirty = false;
    ++dirty_count;
    if (line_addr < min_addr)
      min_addr = line_addr;
    if (line_addr > max_addr)
      max_addr = line_addr;
  });
  util::Logger::vm("L2 flush: ", dirty_count, " dirty lines [0x", std::hex, min_addr, "-0x",
                   max_addr, "]", std::dec, " total_writes=", write_count_);
  cache_.invalidate_all();
}

} // namespace amdgpu
} // namespace rocjitsu
