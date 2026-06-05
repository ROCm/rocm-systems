// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/l1_vector_cache.h"

#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "util/log.h"

#include <bit>
#include <cassert>
#include <cstring>
#include <format>

namespace rocjitsu {
namespace amdgpu {

void L1VectorCache::ensure_line(uint64_t addr, uint32_t vmid) {
  if (cache_.lookup(addr))
    return;

  uint64_t line_addr = CacheStore::line_address(addr);
  simdojo::CacheTag evicted;
  uint8_t evicted_data[LINE_SIZE];
  cache_.allocate(addr, &evicted, evicted_data);

  assert(!evicted.dirty && "L1 V$ is write-through; lines should never be dirty");

  uint8_t line_buf[LINE_SIZE];
  l2_->fetch_line(line_addr, line_buf, vmid);
  cache_.fill_line(addr, line_buf);
}

void L1VectorCache::read_bytes(uint64_t addr, uint8_t *dst, uint32_t size, Mtype mtype,
                               bool non_temporal, uint32_t vmid) {
  Mtype inst_mtype = mtype;
  if (memory_)
    mtype = effective_mtype(mtype, memory_->pte_mtype(addr, vmid));

  util::Logger::cp([&](auto &os) {
    static thread_local uint64_t mtype_counts[5] = {};
    static thread_local uint64_t total = 0;
    ++mtype_counts[static_cast<int>(mtype)];
    ++total;
    if ((total & (total - 1)) == 0 && total >= 1024) {
      os << std::format("L1V_READ_MTYPE_STATS total={} UC={} CC={} RW={} WB={} NT={} "
                        "last: addr={:#x} inst={} eff={} vmid={}",
                        total, mtype_counts[0], mtype_counts[1], mtype_counts[2], mtype_counts[3],
                        mtype_counts[4], addr, static_cast<int>(inst_mtype),
                        static_cast<int>(mtype), vmid);
    }
  });

  if (mtype == Mtype::UC || non_temporal) {
    l2_->read(addr, dst, size, mtype, vmid);
    return;
  }

  if (mtype == Mtype::CC) {
    cache_.invalidate(addr);
    l2_->read(addr, dst, size, mtype, vmid);
    return;
  }

  ensure_line(addr, vmid);
  cache_.read_line(addr, dst, CacheStore::line_offset(addr), size);
}

void L1VectorCache::write_bytes(uint64_t addr, const uint8_t *src, uint32_t size, Mtype mtype,
                                bool non_temporal, uint32_t vmid) {
  if (memory_)
    mtype = effective_mtype(mtype, memory_->pte_mtype(addr, vmid));

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
    l2_->write(addr, src, size, mtype, vmid);
    return;
  }

  ensure_line(addr, vmid);
  cache_.write_line(addr, src, CacheStore::line_offset(addr), size);

  l2_->write(addr, src, size, mtype, vmid);

  simdojo::CacheTag *tag = nullptr;
  cache_.lookup(addr, &tag);
  assert(tag != nullptr && "ensure_line must guarantee hit");

  tag->coherence =
      (mtype == Mtype::CC) ? simdojo::CoherenceState::SHARED : simdojo::CoherenceState::EXCLUSIVE;
  tag->dirty = false;
}

void L1VectorCache::load(const uint64_t *addrs, uint64_t lane_mask, uint32_t elem_size,
                         uint32_t num_elems, uint8_t *dst, Mtype mtype, bool non_temporal,
                         uint32_t vmid) {
  uint32_t stride = num_elems * elem_size;
  uint64_t remaining = lane_mask;
  while (remaining) {
    uint32_t lane = std::countr_zero(remaining);
    remaining &= remaining - 1;
    uint64_t base = addrs[lane];
    for (uint32_t e = 0; e < num_elems; ++e) {
      uint64_t ea = base + e * elem_size;
      read_bytes(ea, dst + lane * stride + e * elem_size, elem_size, mtype, non_temporal, vmid);
    }
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
  uint64_t remaining = lane_mask;
  while (remaining) {
    uint32_t lane = std::countr_zero(remaining);
    remaining &= remaining - 1;
    uint64_t base = addrs[lane];
    for (uint32_t e = 0; e < num_elems; ++e) {
      uint64_t ea = base + e * elem_size;
      write_bytes(ea, src + lane * stride + e * elem_size, elem_size, mtype, non_temporal, vmid);
    }
  }
}

void L1VectorCache::flush_all() { cache_.invalidate_all(); }

} // namespace amdgpu
} // namespace rocjitsu
