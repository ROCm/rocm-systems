// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/l1_vector_cache.h"

#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "util/log.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstring>
#include <format>

namespace rocjitsu {
namespace amdgpu {

void L1VectorCache::ensure_line(uint64_t addr, uint32_t vmid) {
  if (cache_.lookup(addr))
    return;

  fetch_line(addr, vmid);
}

const uint8_t *L1VectorCache::fetch_line(uint64_t addr, uint32_t vmid) {
  uint64_t line_addr = CacheStore::line_address(addr);
  simdojo::CacheTag evicted;
  auto allocated = cache_.allocate_with_data(addr, &evicted);

  assert(!evicted.dirty && "L1 V$ is write-through; lines should never be dirty");

  l2_->fetch_line(line_addr, allocated.data, vmid);
  cached_read_line_addr_ = line_addr;
  cached_read_line_data_ = allocated.data;
  return allocated.data;
}

// Per-line CC invalidation is sufficient: the CP serializes dispatch N's cache
// management before dispatch N+1 begins execution, so no blanket invalidation
// at dispatch boundaries is needed.
void L1VectorCache::read_bytes(uint64_t addr, uint8_t *dst, uint32_t size, Mtype mtype,
                               bool non_temporal, uint32_t vmid) {
  Mtype inst_mtype = mtype;
  ++read_count_;

  util::Logger::cp([&](auto &os) {
    Mtype effective = inst_mtype;
    if (memory_)
      effective = effective_mtype(inst_mtype, memory_->pte_mtype(addr, vmid));
    static thread_local uint64_t mtype_counts[5] = {};
    static thread_local uint64_t total = 0;
    ++mtype_counts[static_cast<int>(effective)];
    ++total;
    if ((total & (total - 1)) == 0 && total >= 1024) {
      os << std::format("L1V_READ_MTYPE_STATS total={} UC={} CC={} RW={} WB={} NT={} "
                        "last: addr={:#x} inst={} eff={} vmid={}",
                        total, mtype_counts[0], mtype_counts[1], mtype_counts[2], mtype_counts[3],
                        mtype_counts[4], addr, static_cast<int>(inst_mtype),
                        static_cast<int>(effective), vmid);
    }
  });

  const bool has_page_mtypes = memory_ && vmid != 0;
  uint64_t cached_page = UINT64_MAX;
  Mtype cached_mtype = inst_mtype;

  if (!has_page_mtypes && !non_temporal && (inst_mtype == Mtype::RW || inst_mtype == Mtype::WB)) {
    uint32_t copied = 0;
    while (copied < size) {
      const uint64_t ea = addr + copied;
      const uint32_t line_offset = CacheStore::line_offset(ea);
      const uint32_t chunk = std::min(size - copied, LINE_SIZE - line_offset);
      const uint8_t *line = line_data_for_read(ea, vmid);
      std::memcpy(dst + copied, line + line_offset, chunk);
      copied += chunk;
    }
    return;
  }

  uint32_t copied = 0;
  while (copied < size) {
    const uint64_t ea = addr + copied;
    const uint32_t line_offset = CacheStore::line_offset(ea);
    const uint32_t chunk = std::min(size - copied, LINE_SIZE - line_offset);
    Mtype chunk_mtype = inst_mtype;
    if (has_page_mtypes) {
      const uint64_t page = ea >> GpuMemory::PAGE_SHIFT;
      if (page != cached_page) {
        cached_page = page;
        cached_mtype = effective_mtype(inst_mtype, memory_->pte_mtype(ea, vmid));
      }
      chunk_mtype = cached_mtype;
    }

    if (chunk_mtype == Mtype::UC || non_temporal) {
      l2_->read(ea, dst + copied, chunk, chunk_mtype, vmid);
      copied += chunk;
      continue;
    }

    if (chunk_mtype == Mtype::CC) {
      invalidate(ea);
      l2_->read(ea, dst + copied, chunk, chunk_mtype, vmid);
      copied += chunk;
      continue;
    }

    const uint8_t *line = line_data_for_read(ea, vmid);
    std::memcpy(dst + copied, line + line_offset, chunk);
    copied += chunk;
  }
}

void L1VectorCache::write_bytes(uint64_t addr, const uint8_t *src, uint32_t size, Mtype mtype,
                                bool non_temporal, uint32_t vmid) {
  Mtype inst_mtype = mtype;

  util::Logger::vm([&](auto &os) {
    Mtype effective = inst_mtype;
    if (memory_)
      effective = effective_mtype(inst_mtype, memory_->pte_mtype(addr, vmid));
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
                          static_cast<int>(effective));
    }
  });

  const bool has_page_mtypes = memory_ && vmid != 0;
  uint64_t cached_page = UINT64_MAX;
  Mtype cached_mtype = inst_mtype;

  uint32_t copied = 0;
  while (copied < size) {
    const uint64_t ea = addr + copied;
    const uint32_t line_offset = CacheStore::line_offset(ea);
    const uint32_t chunk = std::min(size - copied, LINE_SIZE - line_offset);
    Mtype chunk_mtype = inst_mtype;
    if (has_page_mtypes) {
      const uint64_t page = ea >> GpuMemory::PAGE_SHIFT;
      if (page != cached_page) {
        cached_page = page;
        cached_mtype = effective_mtype(inst_mtype, memory_->pte_mtype(ea, vmid));
      }
      chunk_mtype = cached_mtype;
    }

    if (chunk_mtype == Mtype::UC || non_temporal) {
      l2_->write(ea, src + copied, chunk, chunk_mtype, vmid);
      copied += chunk;
      continue;
    }

    ensure_line(ea, vmid);
    cache_.write_line(ea, src + copied, line_offset, chunk);

    // Write through to L2 for all cacheable stores. This ensures partial writes
    // from different CUs sharing the same L2 are properly merged at byte
    // granularity via L2::write(), rather than full-line replacement via
    // writeback_line() during L1 eviction/flush.
    l2_->write(ea, src + copied, chunk, chunk_mtype, vmid);

    simdojo::CacheTag *tag = nullptr;
    cache_.lookup(ea, &tag);
    assert(tag != nullptr && "ensure_line must guarantee hit");

    // L1 line stays clean since L2 has the authoritative copy.
    tag->coherence = (chunk_mtype == Mtype::CC) ? simdojo::CoherenceState::SHARED
                                                : simdojo::CoherenceState::EXCLUSIVE;
    tag->dirty = false;
    copied += chunk;
  }
}

void L1VectorCache::load(const uint64_t *addrs, uint64_t lane_mask, uint32_t elem_size,
                         uint32_t num_elems, uint8_t *dst, Mtype mtype, bool non_temporal,
                         uint32_t vmid) {
  uint32_t stride = num_elems * elem_size;
  if (stride == 0)
    return;

  uint64_t remaining = lane_mask;
  while (remaining) {
    uint32_t first_lane = std::countr_zero(remaining);
    remaining &= remaining - 1;

    uint32_t last_lane = first_lane;
    uint32_t run_lanes = 1;
    uint64_t next_addr = addrs[first_lane] + stride;

    while (remaining) {
      uint32_t lane = std::countr_zero(remaining);
      if (lane != last_lane + 1 || addrs[lane] != next_addr)
        break;

      remaining &= remaining - 1;
      last_lane = lane;
      ++run_lanes;
      next_addr += stride;
    }

    read_bytes(addrs[first_lane], dst + first_lane * stride, run_lanes * stride, mtype,
               non_temporal, vmid);
  }
}

void L1VectorCache::store(const uint64_t *addrs, uint64_t lane_mask, uint32_t elem_size,
                          uint32_t num_elems, const uint8_t *src, Mtype mtype, bool non_temporal,
                          uint32_t vmid) {
  uint32_t stride = num_elems * elem_size;
  uint32_t active_lanes = std::popcount(lane_mask);
  ++store_count_;
  if (active_lanes > 0)
    ++store_active_count_;
  store_l2_writes_ += active_lanes * num_elems;
  if (stride == 0)
    return;

  uint64_t remaining = lane_mask;
  while (remaining) {
    uint32_t first_lane = std::countr_zero(remaining);
    remaining &= remaining - 1;

    uint32_t last_lane = first_lane;
    uint32_t run_lanes = 1;
    uint64_t next_addr = addrs[first_lane] + stride;

    while (remaining) {
      uint32_t lane = std::countr_zero(remaining);
      if (lane != last_lane + 1 || addrs[lane] != next_addr)
        break;

      remaining &= remaining - 1;
      last_lane = lane;
      ++run_lanes;
      next_addr += stride;
    }

    write_bytes(addrs[first_lane], src + first_lane * stride, run_lanes * stride, mtype,
                non_temporal, vmid);
  }
}

void L1VectorCache::flush_all() { invalidate_all(); }

} // namespace amdgpu
} // namespace rocjitsu
