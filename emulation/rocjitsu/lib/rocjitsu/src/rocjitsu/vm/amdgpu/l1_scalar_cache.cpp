// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/l1_scalar_cache.h"

#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"

#include <cstring>

namespace rocjitsu {
namespace amdgpu {

void L1ScalarCache::ensure_line(uint64_t addr, uint32_t vmid) {
  if (cache_.lookup(addr))
    return;

  uint64_t line_addr = CacheStore::line_address(addr);
  simdojo::CacheTag evicted;
  uint8_t evicted_data[CacheStore::LINE_SIZE];
  cache_.allocate(addr, &evicted, evicted_data);

  if (evicted.valid && evicted.dirty) {
    constexpr uint32_t set_bits = 6; // log2(NUM_SETS=64)
    uint64_t evicted_line_addr =
        (evicted.tag << (LINE_SIZE_BITS + set_bits)) |
        (static_cast<uint64_t>(CacheStore::set_index(addr)) << LINE_SIZE_BITS);
    l2_->writeback_line(evicted_line_addr, evicted_data, Mtype::RW, vmid);
  }

  uint8_t line_buf[CacheStore::LINE_SIZE];
  l2_->read(line_addr, line_buf, CacheStore::LINE_SIZE, Mtype::RW, vmid);
  cache_.fill_line(addr, line_buf);
}

void L1ScalarCache::store(uint64_t addr, uint32_t num_dwords, const uint32_t *src, uint32_t vmid) {
  for (uint32_t i = 0; i < num_dwords; ++i) {
    uint64_t ea = addr + i * 4;
    Mtype mtype = Mtype::RW;
    if (memory_)
      mtype = memory_->pte_mtype(ea, vmid);

    uint8_t buf[4];
    std::memcpy(buf, &src[i], 4);

    if (mtype == Mtype::UC) {
      l2_->write(ea, buf, 4, Mtype::UC, vmid);
      continue;
    }

    if (mtype == Mtype::CC) {
      cache_.invalidate(ea);
      l2_->write(ea, buf, 4, Mtype::CC, vmid);
      continue;
    }

    ensure_line(ea, vmid);
    simdojo::CacheTag *tag = nullptr;
    cache_.lookup(ea, &tag);
    assert(tag != nullptr && "ensure_line must guarantee hit");
    cache_.write_line(ea, buf, CacheStore::line_offset(ea), 4);
    tag->dirty = true;
  }
}

void L1ScalarCache::writeback_all(uint32_t vmid) {
  cache_.for_each_dirty([this, vmid](simdojo::CacheTag &tag, uint64_t line_addr, uint8_t *data) {
    l2_->writeback_line(line_addr, data, Mtype::RW, vmid);
    tag.dirty = false;
  });
}

void L1ScalarCache::load(uint64_t addr, uint32_t num_dwords, uint32_t *dst, uint32_t vmid) {
  for (uint32_t i = 0; i < num_dwords; ++i) {
    uint64_t ea = addr + i * 4;
    Mtype mtype = Mtype::RW;
    if (memory_)
      mtype = memory_->pte_mtype(ea, vmid);

    if (mtype == Mtype::UC) {
      uint8_t buf[4]{};
      l2_->read(ea, buf, 4, Mtype::UC, vmid);
      std::memcpy(&dst[i], buf, 4);
      continue;
    }

    if (mtype == Mtype::CC) {
      cache_.invalidate(ea);
      uint8_t buf[4]{};
      l2_->read(ea, buf, 4, Mtype::CC, vmid);
      std::memcpy(&dst[i], buf, 4);
      continue;
    }

    ensure_line(ea, vmid);
    uint8_t buf[4]{};
    cache_.read_line(ea, buf, CacheStore::line_offset(ea), 4);
    std::memcpy(&dst[i], buf, 4);
  }
}

} // namespace amdgpu
} // namespace rocjitsu
