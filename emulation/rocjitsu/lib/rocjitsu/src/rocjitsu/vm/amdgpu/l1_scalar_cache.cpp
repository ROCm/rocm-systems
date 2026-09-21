// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/l1_scalar_cache.h"

#include "rocjitsu/vm/amdgpu/device_cache_coherence.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/request_mtype_resolver.h"

#include <algorithm>
#include <cassert>
#include <cstring>

namespace rocjitsu {
namespace amdgpu {

L1ScalarCache::L1ScalarCache(L2Cache *l2)
    : l2_(l2), coherence_epoch_(DeviceCacheCoherence::instance().current_epoch()) {}

L1ScalarCache::~L1ScalarCache() = default;

void L1ScalarCache::set_l2(L2Cache *l2) {
  synchronize_epoch();
  l2_ = l2;
}

void L1ScalarCache::set_memory(GpuMemory *mem) {
  synchronize_epoch();
  memory_ = mem;
}

void L1ScalarCache::ensure_line(uint64_t addr, uint32_t vmid) {
  if (cache_.lookup(addr, nullptr, vmid))
    return;

  uint64_t line_addr = CacheStore::line_address(addr);
  simdojo::CacheTag evicted;
  cache_.allocate(addr, vmid, &evicted);

  assert(!evicted.dirty && "L1 K$ is write-through; lines should never be dirty");

  uint8_t line_buf[CacheStore::LINE_SIZE];
  l2_->read(line_addr, line_buf, CacheStore::LINE_SIZE, Mtype::RW, vmid);
  cache_.fill_line(addr, line_buf, vmid);
}

void L1ScalarCache::store(uint64_t addr, uint32_t num_dwords, const uint32_t *src, uint32_t vmid) {
  synchronize_epoch();
  RequestMtypeResolver mtypes(memory_, vmid);
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

      const Mtype mtype = mtypes.at(chunk_addr);

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
      l2_->write(chunk_addr, buf + copied, chunk, mtype, vmid);
      tag->dirty = false;
      copied += chunk;
    }
  }
}

void L1ScalarCache::writeback_all(uint32_t vmid) {
  // K$ is write-through, so all stored bytes have already reached L2.
  (void)vmid;
}

void L1ScalarCache::invalidate_all() {
  synchronize_epoch();
  invalidate_all_lines();
}

void L1ScalarCache::invalidate_all_lines() { cache_.invalidate_all(); }

void L1ScalarCache::synchronize_epoch() {
  const uint64_t current_epoch = DeviceCacheCoherence::instance().current_epoch();
  if (coherence_epoch_ == current_epoch)
    return;
  invalidate_all_lines();
  coherence_epoch_ = current_epoch;
}

void L1ScalarCache::flush_line(uint64_t addr, uint32_t vmid) {
  simdojo::CacheTag *tag = nullptr;
  if (!cache_.lookup(addr, &tag, vmid))
    return;

  assert(!tag->dirty && "L1 K$ is write-through; lines should never be dirty");
  cache_.invalidate(addr, vmid);
}

void L1ScalarCache::load(uint64_t addr, uint32_t num_dwords, uint32_t *dst, uint32_t vmid) {
  synchronize_epoch();
  RequestMtypeResolver mtypes(memory_, vmid);
  for (uint32_t i = 0; i < num_dwords; ++i) {
    uint64_t ea = addr + i * 4;
    uint8_t buf[4]{};
    uint32_t copied = 0;
    while (copied < sizeof(buf)) {
      const uint64_t chunk_addr = ea + copied;
      const uint32_t line_offset = CacheStore::line_offset(chunk_addr);
      const uint32_t chunk =
          std::min<uint32_t>(sizeof(buf) - copied, CacheStore::LINE_SIZE - line_offset);

      const Mtype mtype = mtypes.at(chunk_addr);

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
  synchronize_epoch();
  RequestMtypeResolver mtypes(memory_, vmid);
  uint32_t copied = 0;
  while (copied < num_bytes) {
    uint64_t ea = addr + copied;
    uint32_t line_offset = CacheStore::line_offset(ea);
    uint32_t chunk = std::min(num_bytes - copied, CacheStore::LINE_SIZE - line_offset);

    const Mtype mtype = mtypes.at(ea);

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
