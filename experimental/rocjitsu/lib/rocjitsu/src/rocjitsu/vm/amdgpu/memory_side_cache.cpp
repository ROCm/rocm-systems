// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/memory_side_cache.h"

#include <bit>
#include <cassert>
#include <cstring>

namespace rocjitsu {
namespace amdgpu {

void MemorySideCache::send_backing(uint64_t addr, uint8_t *data, uint32_t size,
                                   simdojo::MessageOp op) {
  assert(req_ != nullptr && "MemorySideCache: req_ not set");
  auto msg = std::make_unique<simdojo::Message>();
  auto &hdr = msg->header();
  hdr.addr = addr;
  hdr.size_bytes = size;
  hdr.op = op;
  msg->set_payload(reinterpret_cast<uintptr_t>(data));
  req_->send(std::move(msg));
}

void MemorySideCache::ensure_line(uint64_t addr) {
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
    send_backing(evicted_addr, evicted_data, LINE_SIZE, simdojo::MessageOp::WRITE);
  }

  uint8_t line_buf[LINE_SIZE];
  send_backing(line_addr, line_buf, LINE_SIZE, simdojo::MessageOp::READ);
  cache_.fill_line(addr, line_buf);
}

void MemorySideCache::read(uint64_t addr, uint8_t *dst, uint32_t size) {
  std::lock_guard<std::mutex> lock(stripes_[stripe_index(addr)]);
  ensure_line(addr);
  cache_.read_line(addr, dst, CacheStore::line_offset(addr), size);
}

void MemorySideCache::write(uint64_t addr, const uint8_t *src, uint32_t size) {
  std::lock_guard<std::mutex> lock(stripes_[stripe_index(addr)]);
  ensure_line(addr);
  cache_.write_line(addr, src, CacheStore::line_offset(addr), size);

  simdojo::CacheTag *tag = nullptr;
  cache_.lookup(addr, &tag);
  assert(tag != nullptr && "ensure_line must guarantee hit");
  tag->dirty = true;
  tag->coherence = simdojo::CoherenceState::MODIFIED;
}

void MemorySideCache::flush_all() {
  // flush_all iterates all sets, so acquire all stripes to prevent concurrent access.
  // This is only called after engine->run() completes (single-threaded), so no contention.
  std::lock_guard<std::mutex> flush_lock(flush_mutex_);
  for (auto &s : stripes_)
    s.lock();
  cache_.for_each_dirty([this](simdojo::CacheTag &tag, uint64_t line_addr, uint8_t *data) {
    send_backing(line_addr, data, LINE_SIZE, simdojo::MessageOp::WRITE);
    tag.dirty = false;
  });
  cache_.invalidate_all();
  for (auto &s : stripes_)
    s.unlock();
}

} // namespace amdgpu
} // namespace rocjitsu
