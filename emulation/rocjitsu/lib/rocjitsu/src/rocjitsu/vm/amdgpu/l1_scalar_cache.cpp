// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/l1_scalar_cache.h"

#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"

#include <algorithm>
#include <cstring>

namespace rocjitsu {
namespace amdgpu {

void L1ScalarCache::ensure_line(uint64_t addr, uint32_t vmid) {
  if (cache_.lookup(addr, nullptr, vmid))
    return;

  uint64_t line_addr = CacheStore::line_address(addr);
  simdojo::CacheTag evicted;
  uint8_t evicted_data[CacheStore::LINE_SIZE];
  cache_.allocate(addr, vmid, &evicted, evicted_data);

  if (evicted.valid && evicted.dirty) {
    constexpr uint32_t set_bits = 6; // log2(NUM_SETS=64)
    uint64_t evicted_line_addr =
        (evicted.tag << (LINE_SIZE_BITS + set_bits)) |
        (static_cast<uint64_t>(CacheStore::set_index(addr)) << LINE_SIZE_BITS);
    l2_->write(evicted_line_addr, evicted_data, CacheStore::LINE_SIZE, Mtype::RW, evicted.vmid);
  }

  uint8_t line_buf[CacheStore::LINE_SIZE];
  l2_->read(line_addr, line_buf, CacheStore::LINE_SIZE, Mtype::RW, vmid);
  cache_.fill_line(addr, line_buf, vmid);
}

void L1ScalarCache::store(uint64_t addr, uint32_t num_dwords, const uint32_t *src, uint32_t vmid) {
  for (uint32_t i = 0; i < num_dwords; ++i) {
    uint64_t ea = addr + i * 4;
    uint8_t buf[4];
    std::memcpy(buf, &src[i], 4);
    uint32_t copied = 0;
    while (copied < sizeof(buf)) {
      const uint64_t chunk_addr = ea + copied;
      const uint32_t line_offset = CacheStore::line_offset(chunk_addr);
      const uint32_t chunk =
          std::min<uint32_t>(sizeof(buf) - copied, CacheStore::LINE_SIZE - line_offset);

      Mtype mtype = Mtype::RW;
      if (memory_)
        mtype = memory_->pte_mtype(chunk_addr, vmid);

      if (mtype == Mtype::UC) {
        flush_line(chunk_addr, vmid);
        l2_->write(chunk_addr, buf + copied, chunk, Mtype::UC, vmid);
        copied += chunk;
        continue;
      }

      if (mtype == Mtype::CC) {
        flush_line(chunk_addr, vmid);
        l2_->write(chunk_addr, buf + copied, chunk, Mtype::CC, vmid);
        copied += chunk;
        continue;
      }

      ensure_line(chunk_addr, vmid); // read-allocate on miss

      simdojo::CacheTag *tag = nullptr;
      cache_.lookup(chunk_addr, &tag, vmid);
      assert(tag != nullptr && "ensure_line must guarantee hit");

      cache_.write_line(chunk_addr, buf + copied, line_offset, chunk, vmid);
      tag->dirty = true;
      copied += chunk;
    }
  }
}

void L1ScalarCache::writeback_all(uint32_t vmid) {
  // Each dirty line is written back under its own owning vmid (recorded in the
  // tag), not the caller's vmid. A CU can retain dirty K$ lines from process A
  // and then be flushed while processing process B; using the caller vmid would
  // install A's line into L2 under B's page table and corrupt another process's
  // VA. The eviction path in ensure_line() uses evicted.vmid for the same reason.
  (void)vmid;
  cache_.for_each_dirty([this](simdojo::CacheTag &tag, uint64_t line_addr, uint8_t *data) {
    l2_->write(line_addr, data, CacheStore::LINE_SIZE, Mtype::RW, tag.vmid);
    tag.dirty = false;
  });
}

void L1ScalarCache::flush_line(uint64_t addr, uint32_t vmid) {
  simdojo::CacheTag *tag = nullptr;
  if (!cache_.lookup(addr, &tag, vmid))
    return;

  if (tag->dirty) {
    uint8_t line_buf[CacheStore::LINE_SIZE];
    cache_.read_line(addr, line_buf, 0, CacheStore::LINE_SIZE, vmid);
    l2_->write(CacheStore::line_address(addr), line_buf, CacheStore::LINE_SIZE, Mtype::RW,
               tag->vmid);
  }
  cache_.invalidate(addr, vmid);
}

void L1ScalarCache::load(uint64_t addr, uint32_t num_dwords, uint32_t *dst, uint32_t vmid) {
  for (uint32_t i = 0; i < num_dwords; ++i) {
    uint64_t ea = addr + i * 4;
    uint8_t buf[4]{};
    uint32_t copied = 0;
    while (copied < sizeof(buf)) {
      const uint64_t chunk_addr = ea + copied;
      const uint32_t line_offset = CacheStore::line_offset(chunk_addr);
      const uint32_t chunk =
          std::min<uint32_t>(sizeof(buf) - copied, CacheStore::LINE_SIZE - line_offset);

      Mtype mtype = Mtype::RW;
      if (memory_)
        mtype = memory_->pte_mtype(chunk_addr, vmid);

      if (mtype == Mtype::UC) {
        flush_line(chunk_addr, vmid);
        l2_->read(chunk_addr, buf + copied, chunk, Mtype::UC, vmid);
      } else if (mtype == Mtype::CC) {
        flush_line(chunk_addr, vmid);
        l2_->read(chunk_addr, buf + copied, chunk, Mtype::CC, vmid);
      } else {
        ensure_line(chunk_addr, vmid);
        cache_.read_line(chunk_addr, buf + copied, line_offset, chunk, vmid);
      }
      copied += chunk;
    }
    std::memcpy(&dst[i], buf, 4);
  }
}

void L1ScalarCache::load_bytes(uint64_t addr, uint32_t num_bytes, uint8_t *dst, uint32_t vmid) {
  uint32_t copied = 0;
  while (copied < num_bytes) {
    uint64_t ea = addr + copied;
    uint32_t line_offset = CacheStore::line_offset(ea);
    uint32_t chunk = std::min(num_bytes - copied, CacheStore::LINE_SIZE - line_offset);

    Mtype mtype = Mtype::RW;
    if (memory_)
      mtype = memory_->pte_mtype(ea, vmid);

    if (mtype == Mtype::UC) {
      flush_line(ea, vmid);
      l2_->read(ea, dst + copied, chunk, Mtype::UC, vmid);
    } else if (mtype == Mtype::CC) {
      flush_line(ea, vmid);
      l2_->read(ea, dst + copied, chunk, Mtype::CC, vmid);
    } else {
      ensure_line(ea, vmid);
      cache_.read_line(ea, dst + copied, line_offset, chunk, vmid);
    }
    copied += chunk;
  }
}

} // namespace amdgpu
} // namespace rocjitsu
