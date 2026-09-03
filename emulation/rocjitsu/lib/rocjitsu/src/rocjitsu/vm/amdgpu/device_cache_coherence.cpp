// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/device_cache_coherence.h"

#include "rocjitsu/vm/amdgpu/l2_cache.h"

#include <algorithm>
#include <cassert>
#include <functional>

namespace rocjitsu {
namespace amdgpu {
namespace {

template <typename T> void unregister_cache(std::vector<T *> &caches, T *cache) {
  auto it = std::find(caches.begin(), caches.end(), cache);
  assert(it != caches.end());
  if (it != caches.end())
    caches.erase(it);
}

} // namespace

DeviceCacheCoherence &DeviceCacheCoherence::instance() {
  // Cache components can be destroyed during static teardown. A process-lifetime
  // coordinator avoids singleton destruction-order dependencies.
  static auto *coherence = new DeviceCacheCoherence;
  return *coherence;
}

DeviceCacheCoherence::AtomicBoundary::AtomicBoundary(AtomicBoundary &&other) noexcept
    : owner_(std::exchange(other.owner_, nullptr)),
      locked_l2_count_(std::exchange(other.locked_l2_count_, 0)),
      atomic_lock_(std::move(other.atomic_lock_)),
      coherence_lock_(std::move(other.coherence_lock_)) {}

DeviceCacheCoherence::AtomicBoundary::~AtomicBoundary() {
  if (owner_)
    owner_->release_l2_locks(locked_l2_count_);
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

    for (L2Cache *cache : l2_caches_)
      cache->flush_dirty_locked();

    // Eagerly walking every cache tag makes lane-level atomics prohibitively
    // expensive. Advancing the epoch makes all extant clean lines logically
    // stale; each cache discards them once, on its next ordinary access.
    [[maybe_unused]] const uint64_t next_epoch = epoch_.fetch_add(1, std::memory_order_release) + 1;
    assert(next_epoch != 0 && "device cache coherence epoch wrapped");
  } catch (...) {
    release_l2_locks(locked_l2_count);
    throw;
  }

  return AtomicBoundary(this, locked_l2_count, std::move(atomic_lock), std::move(coherence_lock));
}

void DeviceCacheCoherence::release_l2_locks(size_t locked_l2_count) noexcept {
  while (locked_l2_count != 0) {
    --locked_l2_count;
    l2_caches_[locked_l2_count]->maintenance_mutex_.unlock();
  }
}

void DeviceCacheCoherence::register_l2_cache(L2Cache *cache) {
  std::unique_lock lock(mutex_);
  assert(cache != nullptr);
  const auto it =
      std::lower_bound(l2_caches_.begin(), l2_caches_.end(), cache, std::less<L2Cache *>());
  assert(it == l2_caches_.end() || *it != cache);
  if (it == l2_caches_.end() || *it != cache)
    l2_caches_.insert(it, cache);
}

void DeviceCacheCoherence::unregister_l2_cache(L2Cache *cache) {
  std::unique_lock lock(mutex_);
  unregister_cache(l2_caches_, cache);
}

} // namespace amdgpu
} // namespace rocjitsu
