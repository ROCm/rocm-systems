// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_VM_AMDGPU_DEVICE_CACHE_COHERENCE_H_
#define ROCJITSU_VM_AMDGPU_DEVICE_CACHE_COHERENCE_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <shared_mutex>
#include <utility>
#include <vector>

namespace rocjitsu {
namespace amdgpu {

class L2Cache;

/// @brief Coordinates functional cache state at device-wide atomic boundaries.
///
/// An atomic RMW holds the exclusive guard while dirty L2 state is published
/// and the coherence epoch is advanced, preventing stale writeback or stale
/// post-atomic hits.
///
/// Ordinary cache accesses take no lock here. Each cache owns its epoch
/// bookkeeping and reconciles lazily: an L2 compares its epoch under its own
/// maintenance_mutex_, and an L1 -- which is private to one compute unit and
/// reaches L2 only through that same maintenance_mutex_ -- compares its epoch
/// with no lock at all. current_epoch() is the single point of cross-thread
/// contact, and it is a read-mostly atomic load.
class DeviceCacheCoherence {
public:
  /// @brief RAII guard for a fully prepared device-wide atomic boundary.
  ///
  /// Destruction releases all L2 maintenance locks before the device guard.
  class AtomicBoundary {
  public:
    AtomicBoundary(AtomicBoundary &&other) noexcept;
    AtomicBoundary &operator=(AtomicBoundary &&) = delete;
    AtomicBoundary(const AtomicBoundary &) = delete;
    AtomicBoundary &operator=(const AtomicBoundary &) = delete;
    ~AtomicBoundary();

  private:
    friend class DeviceCacheCoherence;

    AtomicBoundary(DeviceCacheCoherence *owner, size_t locked_l2_count,
                   std::unique_lock<std::mutex> atomic_lock,
                   std::unique_lock<std::shared_mutex> coherence_lock)
        : owner_(owner), locked_l2_count_(locked_l2_count), atomic_lock_(std::move(atomic_lock)),
          coherence_lock_(std::move(coherence_lock)) {}

    DeviceCacheCoherence *owner_ = nullptr;
    size_t locked_l2_count_ = 0;
    std::unique_lock<std::mutex> atomic_lock_;
    std::unique_lock<std::shared_mutex> coherence_lock_;
  };

  static DeviceCacheCoherence &instance();

  [[nodiscard]] AtomicBoundary acquire_atomic_boundary();
  uint64_t current_epoch() const { return epoch_.load(std::memory_order_acquire); }

  void register_l2_cache(L2Cache *cache);
  void unregister_l2_cache(L2Cache *cache);

private:
  DeviceCacheCoherence() = default;
  void release_l2_locks(size_t locked_l2_count) noexcept;

  std::mutex atomic_mutex_;
  std::shared_mutex mutex_;
  std::atomic<uint64_t> epoch_{1};
  std::vector<L2Cache *> l2_caches_;
};

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_DEVICE_CACHE_COHERENCE_H_
