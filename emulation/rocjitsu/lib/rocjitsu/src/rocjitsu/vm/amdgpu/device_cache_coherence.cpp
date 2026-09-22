// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/device_cache_coherence.h"

#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/memory_side_cache.h"
#include "util/except.h"

#include <algorithm>
#include <cassert>
#include <functional>
#include <string_view>

namespace rocjitsu {
namespace amdgpu {
namespace {

template <typename T> void unregister_cache(std::vector<T *> &caches, T *cache) {
  typename std::vector<T *>::iterator position = std::find(caches.begin(), caches.end(), cache);
  assert(position != caches.end());
  if (position != caches.end())
    caches.erase(position);
}

} // namespace

DeviceCacheMaintenanceLease::DeviceCacheMaintenanceLease(
    DeviceCacheCoherence *owner, size_t locked_l2_count, size_t locked_memory_side_count,
    std::unique_lock<std::mutex> atomic_lock, std::unique_lock<std::shared_mutex> coherence_lock)
    : owner_(owner), locked_l2_count_(locked_l2_count),
      locked_memory_side_count_(locked_memory_side_count), atomic_lock_(std::move(atomic_lock)),
      coherence_lock_(std::move(coherence_lock)) {}

DeviceCacheMaintenanceLease::DeviceCacheMaintenanceLease(std::function<void()> release)
    : release_(std::move(release)) {}

DeviceCacheMaintenanceLease::DeviceCacheMaintenanceLease(
    DeviceCacheMaintenanceLease &&other) noexcept
    : owner_(std::exchange(other.owner_, nullptr)),
      locked_l2_count_(std::exchange(other.locked_l2_count_, 0)),
      locked_memory_side_count_(std::exchange(other.locked_memory_side_count_, 0)),
      atomic_lock_(std::move(other.atomic_lock_)),
      coherence_lock_(std::move(other.coherence_lock_)),
      release_(std::exchange(other.release_, std::function<void()>{})) {}

DeviceCacheMaintenanceLease::~DeviceCacheMaintenanceLease() noexcept {
  if (owner_ != nullptr)
    owner_->complete_cache_maintenance(locked_l2_count_, locked_memory_side_count_);
  if (release_)
    release_();
}

DeviceCacheCoherence::AtomicBoundary::AtomicBoundary(AtomicBoundary &&other) noexcept
    : owner_(std::exchange(other.owner_, nullptr)),
      locked_l2_count_(std::exchange(other.locked_l2_count_, 0)),
      atomic_lock_(std::move(other.atomic_lock_)),
      coherence_lock_(std::move(other.coherence_lock_)), outcome_(other.outcome_) {}

DeviceCacheCoherence::AtomicBoundary::AtomicBoundary(VmAccessOutcome outcome) : outcome_(outcome) {}

DeviceCacheCoherence::AtomicBoundary::AtomicBoundary(
    DeviceCacheCoherence *owner, size_t locked_l2_count, std::unique_lock<std::mutex> atomic_lock,
    std::unique_lock<std::shared_mutex> coherence_lock)
    : owner_(owner), locked_l2_count_(locked_l2_count), atomic_lock_(std::move(atomic_lock)),
      coherence_lock_(std::move(coherence_lock)), outcome_(VmAccessOutcome::Complete) {}

VmAccessOutcome DeviceCacheCoherence::AtomicBoundary::outcome() const { return outcome_; }

DeviceCacheCoherence::AtomicBoundary::~AtomicBoundary() {
  if (owner_)
    owner_->release_cache_locks(locked_l2_count_, 0);
}

DeviceCacheCoherence::AtomicBoundary DeviceCacheCoherence::acquire_atomic_boundary() {
  std::unique_lock atomic_lock(atomic_mutex_);
  std::unique_lock coherence_lock(mutex_);

  size_t locked_l2_count = 0;
  try {
    // l2_caches_ is maintained in pointer order while mutex_ is held, so the
    // boundary needs neither a registry copy nor a per-atomic allocation.
    for (L2Cache *cache : l2_caches_) {
      cache->maintenance_mutex_.lock();
      ++locked_l2_count;
    }
    for (L2Cache *cache : l2_caches_) {
      const VmAccessOutcome outcome = cache->flush_dirty_locked();
      if (outcome != VmAccessOutcome::Complete) {
        release_cache_locks(locked_l2_count, 0);
        return AtomicBoundary(outcome);
      }
    }

    // Eagerly walking every cache tag makes lane-level atomics prohibitively
    // expensive. Advancing the epoch makes all extant clean lines logically
    // stale; each cache discards them once, on its next ordinary access.
    [[maybe_unused]] const uint64_t next_epoch =
        data_epoch_.fetch_add(1, std::memory_order_release) + 1;
    assert(next_epoch != 0 && "device cache coherence epoch wrapped");
  } catch (...) {
    release_cache_locks(locked_l2_count, 0);
    throw;
  }

  return AtomicBoundary(this, locked_l2_count, std::move(atomic_lock), std::move(coherence_lock));
}

DeviceCacheMaintenanceLease
DeviceCacheCoherence::acquire_cache_maintenance(DeviceCacheOperation operation) {
  std::unique_lock atomic_lock(atomic_mutex_);
  std::unique_lock coherence_lock(mutex_);

  if (operation == DeviceCacheOperation::WritebackInvalidate) {
    const bool missing_l2_backing = std::ranges::any_of(
        l2_caches_, [](const L2Cache *cache) { return !cache->has_legacy_maintenance_backing(); });
    const bool missing_memory_side_backing =
        std::ranges::any_of(memory_side_caches_, [](const MemorySideCache *cache) {
          return !cache->has_legacy_maintenance_backing();
        });
    if (missing_l2_backing || missing_memory_side_backing)
      throw util::ConfigError("cache hierarchy lacks legacy maintenance backing");
  }

  size_t locked_l2_count = 0;
  size_t locked_memory_side_count = 0;
  try {
    for (L2Cache *cache : l2_caches_) {
      cache->maintenance_mutex_.lock();
      ++locked_l2_count;
    }

    // This path can run on the standalone SDMA worker. It must never enter the
    // simulation topology from that thread, so legacy dirty state is published
    // directly to the SoC's GpuMemory backing while cache access is excluded.
    if (operation == DeviceCacheOperation::WritebackInvalidate) {
      for (L2Cache *cache : l2_caches_) {
        if (cache->flush_dirty_to_legacy_backing_locked() != VmAccessOutcome::Complete)
          throw util::Exception(std::string_view("L2 cache maintenance backing write failed"));
      }
    }

    for (MemorySideCache *cache : memory_side_caches_) {
      cache->access_gate_.lock();
      ++locked_memory_side_count;
    }

    if (operation == DeviceCacheOperation::WritebackInvalidate) {
      for (MemorySideCache *cache : memory_side_caches_) {
        if (cache->flush_dirty_to_legacy_backing_locked() != VmAccessOutcome::Complete)
          throw util::Exception(
              std::string_view("memory-side cache maintenance backing write failed"));
      }
    }

  } catch (...) {
    release_cache_locks(locked_l2_count, locked_memory_side_count);
    throw;
  }

  return DeviceCacheMaintenanceLease(this, locked_l2_count, locked_memory_side_count,
                                     std::move(atomic_lock), std::move(coherence_lock));
}

void DeviceCacheCoherence::complete_cache_maintenance(size_t locked_l2_count,
                                                      size_t locked_memory_side_count) noexcept {
  // Publish invalidation only after the protected backing-memory effect. Data
  // caches remain excluded until the epochs are visible. Private instruction
  // caches are not touched cross-thread and reconcile this epoch on their next
  // owner-thread fetch.
  [[maybe_unused]] const uint64_t next_data_epoch =
      data_epoch_.fetch_add(1, std::memory_order_release) + 1;
  [[maybe_unused]] const uint64_t next_instruction_epoch =
      instruction_epoch_.fetch_add(1, std::memory_order_release) + 1;
  assert(next_data_epoch != 0 && "device data-cache coherence epoch wrapped");
  assert(next_instruction_epoch != 0 && "device instruction-cache coherence epoch wrapped");
  release_cache_locks(locked_l2_count, locked_memory_side_count);
}

void DeviceCacheCoherence::release_cache_locks(size_t locked_l2_count,
                                               size_t locked_memory_side_count) noexcept {
  while (locked_memory_side_count != 0) {
    --locked_memory_side_count;
    memory_side_caches_[locked_memory_side_count]->access_gate_.unlock();
  }
  while (locked_l2_count != 0) {
    --locked_l2_count;
    l2_caches_[locked_l2_count]->maintenance_mutex_.unlock();
  }
}

void DeviceCacheCoherence::register_l2_cache(L2Cache *cache) {
  std::unique_lock lock(mutex_);
  assert(cache != nullptr);
  const std::vector<L2Cache *>::iterator position =
      std::lower_bound(l2_caches_.begin(), l2_caches_.end(), cache, std::less<L2Cache *>());
  assert(position == l2_caches_.end() || *position != cache);
  if (position == l2_caches_.end() || *position != cache)
    l2_caches_.insert(position, cache);
}

void DeviceCacheCoherence::unregister_l2_cache(L2Cache *cache) {
  std::unique_lock lock(mutex_);
  unregister_cache(l2_caches_, cache);
}

void DeviceCacheCoherence::register_memory_side_cache(MemorySideCache *cache) {
  std::unique_lock lock(mutex_);
  assert(cache != nullptr);
  const std::vector<MemorySideCache *>::iterator position =
      std::lower_bound(memory_side_caches_.begin(), memory_side_caches_.end(), cache,
                       std::less<MemorySideCache *>());
  assert(position == memory_side_caches_.end() || *position != cache);
  if (position == memory_side_caches_.end() || *position != cache)
    memory_side_caches_.insert(position, cache);
}

void DeviceCacheCoherence::unregister_memory_side_cache(MemorySideCache *cache) {
  std::unique_lock lock(mutex_);
  unregister_cache(memory_side_caches_, cache);
}

bool DeviceCacheCoherence::has_legacy_maintenance_backing() const {
  std::shared_lock lock(mutex_);
  return std::ranges::all_of(
             l2_caches_,
             [](const L2Cache *cache) { return cache->has_legacy_maintenance_backing(); }) &&
         std::ranges::all_of(memory_side_caches_, [](const MemorySideCache *cache) {
           return cache->has_legacy_maintenance_backing();
         });
}

} // namespace amdgpu
} // namespace rocjitsu
