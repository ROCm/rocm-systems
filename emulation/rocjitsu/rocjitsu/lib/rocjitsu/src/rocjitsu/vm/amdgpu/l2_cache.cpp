// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "util/log.h"

#include <bit>
#include <cassert>
#include <cstring>

namespace rocjitsu {
namespace amdgpu {

void L2Cache::send_backing(uint64_t addr, uint8_t *data, uint32_t size, simdojo::MessageOp op) {
  // In functional mode, always use direct GpuMemory writeback when available.
  // The req_port_ link may exist but point to a component (MSC) that doesn't
  // handle messages in functional mode. Direct write ensures data reaches
  // host-mapped memory.
  if (backing_memory_) {
    if (true) {
      if (op == simdojo::MessageOp::WRITE) {
        static uint64_t wb_count = 0;
        if (++wb_count <= 3)
          util::Logger::vm("L2 writeback(backing) #", wb_count, " addr=0x", std::hex, addr,
                           " size=", std::dec, size);
        for (uint32_t i = 0; i < size; ++i)
          backing_memory_->write8(addr + i, data[i]);
      } else {
        for (uint32_t i = 0; i < size; ++i)
          data[i] = backing_memory_->read8(addr + i);
      }
    } else {
      // No backing store — data is lost. This should not happen.
      static bool warned = false;
      if (!warned) {
        util::Logger::vm("L2 WARNING: send_backing with no backing_memory_ — data lost at 0x",
                         std::hex, addr, std::dec);
        warned = true;
      }
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
  msg->set_payload(reinterpret_cast<uintptr_t>(data));
  req_port_->send(std::move(msg));
}

void L2Cache::ensure_line(uint64_t addr) {
  if (cache_.lookup(addr))
    return;

  uint64_t line_addr = CacheStore::line_address(addr);
  simdojo::CacheTag evicted;
  uint8_t evicted_data[LINE_SIZE];
  cache_.allocate(addr, &evicted, evicted_data);

  if (evicted.valid && evicted.dirty) {
    static constexpr uint32_t SET_INDEX_BITS = std::bit_width(NUM_SETS - 1);
    uint64_t evicted_addr = (evicted.tag << (LINE_SIZE_BITS + SET_INDEX_BITS)) |
                            (static_cast<uint64_t>(CacheStore::set_index(addr)) << LINE_SIZE_BITS);
    util::Logger::vm([&](auto &os) {
      static thread_local uint64_t evict_count = 0;
      if (++evict_count <= 5 || (evict_count % 10000) == 0)
        os << std::format("L2 evict #{} addr={:#x} new={:#x}", evict_count, evicted_addr, addr);
    });
    send_backing(evicted_addr, evicted_data, LINE_SIZE, simdojo::MessageOp::WRITE);
  }

  uint8_t line_buf[LINE_SIZE];
  send_backing(line_addr, line_buf, LINE_SIZE, simdojo::MessageOp::READ);
  cache_.fill_line(addr, line_buf);
}

void L2Cache::read(uint64_t addr, uint8_t *dst, uint32_t size, Mtype mtype) {
  if (mtype == Mtype::UC) {
    // Uncacheable: bypass L2, read directly from HBM.
    send_backing(addr, dst, size, simdojo::MessageOp::READ);
    return;
  }

  if (mtype == Mtype::CC) {
    // Coherently Cacheable: invalidate L2 line before refetch, mirroring
    // the L1 CC behavior. This ensures cross-XCD store visibility when
    // different L2s share a backing store. Dirty data is flushed first.
    flush_line(addr);
  }

  ensure_line(addr);
  cache_.read_line(addr, dst, CacheStore::line_offset(addr), size);
}

void L2Cache::write(uint64_t addr, const uint8_t *src, uint32_t size, Mtype mtype) {
  if (mtype == Mtype::UC) {
    // Uncacheable: bypass L2, write directly to HBM.
    send_backing(addr, const_cast<uint8_t *>(src), size, simdojo::MessageOp::WRITE);
    return;
  }

  ensure_line(addr);
  cache_.write_line(addr, src, CacheStore::line_offset(addr), size);

  simdojo::CacheTag *tag = nullptr;
  cache_.lookup(addr, &tag);
  assert(tag != nullptr && "ensure_line must guarantee hit");

  // Write through to backing store for all mtypes. In the simulator, GPU
  // virtual addresses are host-mapped (MAP_FIXED), so hipMemcpy reads from
  // backing store directly. Without write-through, RW-mtype stores remain
  // in L2 as dirty lines and never reach the host-visible pages — causing
  // stale reads on D2H copy.
  send_backing(addr, const_cast<uint8_t *>(src), size, simdojo::MessageOp::WRITE);
  tag->coherence =
      (mtype == Mtype::CC) ? simdojo::CoherenceState::SHARED : simdojo::CoherenceState::EXCLUSIVE;
  tag->dirty = false;

  ++write_count_;
}

void L2Cache::fetch_line(uint64_t addr, uint8_t *line_buf) {
  uint64_t line_addr = CacheStore::line_address(addr);
  ensure_line(line_addr);
  cache_.read_line(line_addr, line_buf, 0, LINE_SIZE);
}

void L2Cache::writeback_line(uint64_t line_addr, const uint8_t *data, Mtype mtype) {
  // Write the line data into L2.
  simdojo::CacheTag *tag = nullptr;
  if (cache_.lookup(line_addr, &tag)) {
    cache_.write_line(line_addr, data, 0, LINE_SIZE);
  } else {
    simdojo::CacheTag evicted;
    uint8_t evicted_data[LINE_SIZE];
    cache_.allocate(line_addr, &evicted, evicted_data);

    if (evicted.valid && evicted.dirty) {
      static constexpr uint32_t SET_INDEX_BITS = std::bit_width(NUM_SETS - 1);
      uint64_t evicted_addr =
          (evicted.tag << (LINE_SIZE_BITS + SET_INDEX_BITS)) |
          (static_cast<uint64_t>(CacheStore::set_index(line_addr)) << LINE_SIZE_BITS);
      send_backing(evicted_addr, evicted_data, LINE_SIZE, simdojo::MessageOp::WRITE);
    }

    cache_.fill_line(line_addr, data);
    cache_.lookup(line_addr, &tag);
  }

  if (mtype == Mtype::CC) {
    // Write-through to HBM for coherent accesses.
    send_backing(line_addr, const_cast<uint8_t *>(data), LINE_SIZE, simdojo::MessageOp::WRITE);
    tag->coherence = simdojo::CoherenceState::SHARED;
  } else {
    tag->dirty = true;
    tag->coherence = simdojo::CoherenceState::MODIFIED;
  }
}

void L2Cache::flush_line(uint64_t addr) {
  simdojo::CacheTag *tag = nullptr;
  if (!cache_.lookup(addr, &tag))
    return; // Not cached -- nothing to flush.

  if (tag->dirty) {
    // Writeback dirty data before invalidating.
    uint8_t line_buf[LINE_SIZE];
    cache_.read_line(addr, line_buf, 0, LINE_SIZE);
    uint64_t line_addr = CacheStore::line_address(addr);
    send_backing(line_addr, line_buf, LINE_SIZE, simdojo::MessageOp::WRITE);
  }
  cache_.invalidate(addr);
}

void L2Cache::flush_all() {
  uint32_t dirty_count = 0;
  uint64_t min_addr = UINT64_MAX, max_addr = 0;
  cache_.for_each_dirty([this, &dirty_count, &min_addr,
                         &max_addr](simdojo::CacheTag &tag, uint64_t line_addr, uint8_t *data) {
    send_backing(line_addr, data, LINE_SIZE, simdojo::MessageOp::WRITE);
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
