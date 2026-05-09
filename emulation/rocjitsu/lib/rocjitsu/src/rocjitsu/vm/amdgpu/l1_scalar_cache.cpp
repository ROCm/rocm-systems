// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/l1_scalar_cache.h"

#include "rocjitsu/vm/amdgpu/l2_cache.h"

#include <cstring>

namespace rocjitsu {
namespace amdgpu {

void L1ScalarCache::ensure_line(uint64_t addr) {
  if (cache_.lookup(addr))
    return;

  uint64_t line_addr = CacheStore::line_address(addr);
  simdojo::CacheTag evicted;
  uint8_t evicted_data[CacheStore::LINE_SIZE];
  cache_.allocate(addr, &evicted, evicted_data);

  if (evicted.valid && evicted.dirty) {
    // Reconstruct the evicted line's address from its tag and the set index
    // (eviction always happens within the same set as addr).
    // Bits: [tag | set_index | 0...0] where set occupies log2(NUM_SETS)=6 bits
    // and the line offset occupies LINE_SIZE_BITS=6 bits.
    constexpr uint32_t set_bits = 6; // log2(NUM_SETS=64)
    uint64_t evicted_line_addr =
        (evicted.tag << (LINE_SIZE_BITS + set_bits)) |
        (static_cast<uint64_t>(CacheStore::set_index(addr)) << LINE_SIZE_BITS);
    l2_->writeback_line(evicted_line_addr, evicted_data);
  }

  uint8_t line_buf[CacheStore::LINE_SIZE];
  l2_->read(line_addr, line_buf, CacheStore::LINE_SIZE);
  cache_.fill_line(addr, line_buf);
}

void L1ScalarCache::store(uint64_t addr, uint32_t num_dwords, const uint32_t *src) {
  for (uint32_t i = 0; i < num_dwords; ++i) {
    uint64_t ea = addr + i * 4;
    ensure_line(ea); // read-allocate on miss

    simdojo::CacheTag *tag = nullptr;
    cache_.lookup(ea, &tag);
    assert(tag != nullptr && "ensure_line must guarantee hit");

    uint8_t buf[4];
    std::memcpy(buf, &src[i], 4);
    cache_.write_line(ea, buf, CacheStore::line_offset(ea), 4);
    tag->dirty = true;
  }
}

void L1ScalarCache::writeback_all() {
  cache_.for_each_dirty([this](simdojo::CacheTag &tag, uint64_t line_addr, uint8_t *data) {
    l2_->writeback_line(line_addr, data);
    tag.dirty = false;
  });
}

void L1ScalarCache::load(uint64_t addr, uint32_t num_dwords, uint32_t *dst) {
  for (uint32_t i = 0; i < num_dwords; ++i) {
    uint64_t ea = addr + i * 4;
    ensure_line(ea);

    uint8_t buf[4]{};
    cache_.read_line(ea, buf, CacheStore::line_offset(ea), 4);
    std::memcpy(&dst[i], buf, 4);
  }
}

} // namespace amdgpu
} // namespace rocjitsu
