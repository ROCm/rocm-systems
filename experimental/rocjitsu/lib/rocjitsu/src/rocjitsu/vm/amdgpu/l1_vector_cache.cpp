// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/l1_vector_cache.h"

#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "util/log.h"

#include <cassert>

namespace rocjitsu {
namespace amdgpu {

void L1VectorCache::ensure_line(uint64_t addr) {
  if (cache_.lookup(addr))
    return;

  uint64_t line_addr = CacheStore::line_address(addr);
  simdojo::CacheTag evicted;
  uint8_t evicted_data[LINE_SIZE];
  cache_.allocate(addr, &evicted, evicted_data);

  assert(!evicted.dirty && "L1 V$ is write-through; lines should never be dirty");

  uint8_t line_buf[LINE_SIZE];
  l2_->fetch_line(line_addr, line_buf);
  cache_.fill_line(addr, line_buf);
}

void L1VectorCache::read_bytes(uint64_t addr, uint8_t *dst, uint32_t size, Mtype mtype,
                               bool non_temporal) {
  if (mtype == Mtype::UC || non_temporal) {
    l2_->read(addr, dst, size, mtype);
    return;
  }

  if (mtype == Mtype::CC) {
    // SC0/GLC: invalidate stale L1 line to force refetch from L2.
    // On real hardware, coherently-cacheable loads re-validate against L2
    // even on L1 hit, ensuring inter-CU stores are visible.
    cache_.invalidate(addr);
  }

  ensure_line(addr);
  cache_.read_line(addr, dst, CacheStore::line_offset(addr), size);
}

void L1VectorCache::write_bytes(uint64_t addr, const uint8_t *src, uint32_t size, Mtype mtype,
                                bool non_temporal) {
  util::Logger::vm([&](auto &os) {
    if (addr >= 0x4d00c00000ULL && addr < 0x4d00c00100ULL) {
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
                          static_cast<int>(mtype));
    }
  });
  if (mtype == Mtype::UC || non_temporal) {
    l2_->write(addr, src, size, mtype);
    return;
  }

  ensure_line(addr);
  cache_.write_line(addr, src, CacheStore::line_offset(addr), size);

  // Write through to L2 for all cacheable stores. This ensures partial writes
  // from different CUs sharing the same L2 are properly merged at byte
  // granularity via L2::write(), rather than full-line replacement via
  // writeback_line() during L1 eviction/flush.
  l2_->write(addr, src, size, mtype);

  simdojo::CacheTag *tag = nullptr;
  cache_.lookup(addr, &tag);
  assert(tag != nullptr && "ensure_line must guarantee hit");

  // L1 line stays clean since L2 has the authoritative copy.
  tag->coherence =
      (mtype == Mtype::CC) ? simdojo::CoherenceState::SHARED : simdojo::CoherenceState::EXCLUSIVE;
  tag->dirty = false;
}

void L1VectorCache::load(const uint64_t *addrs, uint64_t lane_mask, uint32_t elem_size,
                         uint32_t num_elems, uint8_t *dst, Mtype mtype, bool non_temporal) {
  uint32_t stride = num_elems * elem_size;
  // Iterate only over lanes that are active according to lane_mask.
  uint64_t remaining = lane_mask;
  while (remaining) {
    uint32_t lane = __builtin_ctzll(remaining);
    remaining &= remaining - 1; // clear lowest set bit
    uint64_t base = addrs[lane];
    for (uint32_t e = 0; e < num_elems; ++e) {
      uint64_t ea = base + e * elem_size;
      read_bytes(ea, dst + lane * stride + e * elem_size, elem_size, mtype, non_temporal);
    }
  }
}

void L1VectorCache::store(const uint64_t *addrs, uint64_t lane_mask, uint32_t elem_size,
                          uint32_t num_elems, const uint8_t *src, Mtype mtype, bool non_temporal) {
  uint32_t stride = num_elems * elem_size;
  uint32_t active_lanes = __builtin_popcountll(lane_mask);
  ++store_count_;
  if (active_lanes > 0)
    ++store_active_count_;
  store_l2_writes_ += active_lanes * num_elems;
  // Iterate only over lanes that are active according to lane_mask.
  uint64_t remaining = lane_mask;
  while (remaining) {
    uint32_t lane = __builtin_ctzll(remaining);
    remaining &= remaining - 1; // clear lowest set bit
    uint64_t base = addrs[lane];
    for (uint32_t e = 0; e < num_elems; ++e) {
      uint64_t ea = base + e * elem_size;
      write_bytes(ea, src + lane * stride + e * elem_size, elem_size, mtype, non_temporal);
    }
  }
}

void L1VectorCache::flush_all() {
  // All stores are written through to L2 during execution, so no dirty lines
  // to write back. Just invalidate.
  cache_.invalidate_all();
}

} // namespace amdgpu
} // namespace rocjitsu
