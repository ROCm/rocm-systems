// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/memory_side_cache.h"

#include "rocjitsu/vm/amdgpu/device_cache_coherence.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstring>
#include <shared_mutex>

namespace rocjitsu {
namespace amdgpu {

void WriterPreferredAccessGate::lock_shared() {
  std::unique_lock lock(mutex_);
  cv_.wait(lock, [this] { return !writer_active_ && waiting_writers_ == 0; });
  ++active_readers_;
}

bool WriterPreferredAccessGate::try_lock_shared() {
  std::lock_guard lock(mutex_);
  if (writer_active_ || waiting_writers_ != 0)
    return false;
  ++active_readers_;
  return true;
}

void WriterPreferredAccessGate::unlock_shared() {
  bool notify_writer = false;
  {
    std::lock_guard lock(mutex_);
    assert(active_readers_ != 0 && "unlock_shared without an active reader");
    --active_readers_;
    notify_writer = active_readers_ == 0 && waiting_writers_ != 0;
  }
  if (notify_writer)
    cv_.notify_all();
}

void WriterPreferredAccessGate::lock() {
  std::unique_lock lock(mutex_);
  ++waiting_writers_;
  cv_.wait(lock, [this] { return !writer_active_ && active_readers_ == 0; });
  --waiting_writers_;
  writer_active_ = true;
}

void WriterPreferredAccessGate::unlock() {
  {
    std::lock_guard lock(mutex_);
    assert(writer_active_ && "unlock without an active writer");
    writer_active_ = false;
  }
  cv_.notify_all();
}

void MemorySideCache::send_backing(uint64_t addr, uint8_t *data, uint32_t size,
                                   simdojo::MessageOp op, uint32_t vmid) {
  assert(req_ != nullptr && "MemorySideCache: req_ not set");
  auto msg = std::make_unique<simdojo::Message>();
  auto &hdr = msg->header();
  hdr.addr = addr;
  hdr.size_bytes = size;
  hdr.op = op;
  hdr.vmid = vmid;
  msg->set_payload(reinterpret_cast<uintptr_t>(data));
  req_->send(std::move(msg));
}

void MemorySideCache::ensure_line(uint64_t addr, uint32_t vmid) {
  const uint64_t current_epoch = DeviceCacheCoherence::instance().current_epoch();
  simdojo::CacheTag *resident = nullptr;
  if (cache_.lookup(addr, &resident, vmid)) {
    if (resident->coherence_epoch == current_epoch)
      return;
    assert(!resident->dirty && "stale write-through MSC line must be clean");
    cache_.invalidate(addr, vmid);
  }

  uint64_t line_addr = CacheStore::line_address(addr);
  simdojo::CacheTag evicted;
  uint8_t evicted_data[LINE_SIZE];
  simdojo::CacheTag *allocated = cache_.allocate(addr, vmid, &evicted, evicted_data);

  if (evicted.valid && evicted.dirty) {
    static constexpr uint32_t SET_INDEX_BITS = std::bit_width(NUM_SETS - 1);
    uint64_t evicted_addr = (evicted.tag << (LINE_SIZE_BITS + SET_INDEX_BITS)) |
                            (static_cast<uint64_t>(CacheStore::set_index(addr)) << LINE_SIZE_BITS);
    send_backing(evicted_addr, evicted_data, LINE_SIZE, simdojo::MessageOp::WRITE, evicted.vmid);
  }

  uint8_t line_buf[LINE_SIZE];
  send_backing(line_addr, line_buf, LINE_SIZE, simdojo::MessageOp::READ, vmid);
  cache_.fill_line(addr, line_buf, vmid);
  allocated->coherence_epoch = current_epoch;
}

void MemorySideCache::read(uint64_t addr, uint8_t *dst, uint32_t size, uint32_t vmid) {
  std::shared_lock access_lock(access_gate_);
  uint32_t copied = 0;
  while (copied < size) {
    const uint64_t ea = addr + copied;
    const uint32_t line_offset = CacheStore::line_offset(ea);
    const uint32_t chunk = std::min(size - copied, LINE_SIZE - line_offset);

    std::lock_guard<std::mutex> lock(stripes_[stripe_index(ea)]);
    ensure_line(ea, vmid);
    cache_.read_line(ea, dst + copied, line_offset, chunk, vmid);
    copied += chunk;
  }
}

void MemorySideCache::write(uint64_t addr, const uint8_t *src, uint32_t size, uint32_t vmid) {
  std::shared_lock access_lock(access_gate_);
  uint32_t copied = 0;
  while (copied < size) {
    const uint64_t ea = addr + copied;
    const uint32_t line_offset = CacheStore::line_offset(ea);
    const uint32_t chunk = std::min(size - copied, LINE_SIZE - line_offset);

    std::lock_guard<std::mutex> lock(stripes_[stripe_index(ea)]);
    ensure_line(ea, vmid);
    cache_.write_line(ea, src + copied, line_offset, chunk, vmid);

    simdojo::CacheTag *tag = nullptr;
    cache_.lookup(ea, &tag, vmid);
    assert(tag != nullptr && "ensure_line must guarantee hit");
    send_backing(ea, const_cast<uint8_t *>(src + copied), chunk, simdojo::MessageOp::WRITE, vmid);
    tag->dirty = false;
    tag->coherence = simdojo::CoherenceState::EXCLUSIVE;
    copied += chunk;
  }
}

void MemorySideCache::flush_all() {
  // Exclude reads and writes while iterating every set. Ordinary accesses hold
  // this gate in shared mode and retain their per-stripe concurrency. Once a
  // flush waits, the gate blocks new shared entrants so the flush makes progress.
  std::unique_lock access_lock(access_gate_);
  cache_.for_each_dirty([this](simdojo::CacheTag &tag, uint64_t line_addr, uint8_t *data) {
    send_backing(line_addr, data, LINE_SIZE, simdojo::MessageOp::WRITE, tag.vmid);
    tag.dirty = false;
  });
  cache_.invalidate_all();
}

} // namespace amdgpu
} // namespace rocjitsu
