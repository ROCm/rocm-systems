// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <shared_mutex>
#include <utility>
#include <vector>

namespace rocjitsu {
namespace amdgpu {

class L2Cache;
class MemorySideCache;
class DeviceCacheCoherence;
enum class VmAccessOutcome : uint8_t;

/// @brief Cache disposition requested at a device-wide maintenance boundary.
enum class DeviceCacheOperation {
  WritebackInvalidate,
  Invalidate,
};

/// @brief Move-only ownership of a quiesced device cache hierarchy.
///
/// The lease keeps device caches excluded until the direct backing-memory
/// operation protected by it has completed or yielded. Destroying the lease
/// releases every cache before releasing the device-wide coordinator locks.
class DeviceCacheMaintenanceLease {
public:
  DeviceCacheMaintenanceLease() = default;
  /// @brief Build a lease around an adapter-specific release operation.
  /// @note The release operation is invoked by the destructor and must not throw.
  explicit DeviceCacheMaintenanceLease(std::function<void()> release);
  DeviceCacheMaintenanceLease(DeviceCacheMaintenanceLease &&other) noexcept;
  DeviceCacheMaintenanceLease &operator=(DeviceCacheMaintenanceLease &&) = delete;
  DeviceCacheMaintenanceLease(const DeviceCacheMaintenanceLease &) = delete;
  DeviceCacheMaintenanceLease &operator=(const DeviceCacheMaintenanceLease &) = delete;
  ~DeviceCacheMaintenanceLease() noexcept;

private:
  friend class DeviceCacheCoherence;

  DeviceCacheMaintenanceLease(DeviceCacheCoherence *owner, size_t locked_l2_count,
                              size_t locked_memory_side_count,
                              std::unique_lock<std::mutex> atomic_lock,
                              std::unique_lock<std::shared_mutex> coherence_lock);
  DeviceCacheCoherence *owner_ = nullptr;
  size_t locked_l2_count_ = 0;
  size_t locked_memory_side_count_ = 0;
  std::unique_lock<std::mutex> atomic_lock_;
  std::unique_lock<std::shared_mutex> coherence_lock_;
  std::function<void()> release_;
};

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
  DeviceCacheCoherence() = default;

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

    /// @brief Result of publishing dirty cache state for this boundary.
    [[nodiscard]] VmAccessOutcome outcome() const;

  private:
    friend class DeviceCacheCoherence;

    explicit AtomicBoundary(VmAccessOutcome outcome);
    AtomicBoundary(DeviceCacheCoherence *owner, size_t locked_l2_count,
                   std::unique_lock<std::mutex> atomic_lock,
                   std::unique_lock<std::shared_mutex> coherence_lock);

    DeviceCacheCoherence *owner_ = nullptr;
    size_t locked_l2_count_ = 0;
    std::unique_lock<std::mutex> atomic_lock_;
    std::unique_lock<std::shared_mutex> coherence_lock_;
    VmAccessOutcome outcome_;
  };

  /// @brief Quiesce every registered L2 while preparing a device-wide atomic.
  /// @details Dirty L2 bytes are published before the data epoch advances. The
  /// returned boundary keeps all L2 maintenance locks held through the backing
  /// atomic operation.
  [[nodiscard]] AtomicBoundary acquire_atomic_boundary();
  /// @brief Quiesce the registered data-cache hierarchy for an external access.
  /// @details WritebackInvalidate publishes dirty state before returning. The
  /// returned lease advances the data and instruction epochs only after the
  /// protected access has completed and the lease is destroyed.
  [[nodiscard]] DeviceCacheMaintenanceLease
  acquire_cache_maintenance(DeviceCacheOperation operation);
  /// @brief Return the epoch observed by data caches during ordinary accesses.
  uint64_t current_epoch() const { return data_epoch_.load(std::memory_order_acquire); }
  /// @brief Return the epoch observed by private instruction caches.
  uint64_t current_instruction_epoch() const {
    return instruction_epoch_.load(std::memory_order_acquire);
  }

  /// @brief Register an L2 cache in this device-local coherence domain.
  void register_l2_cache(L2Cache *cache);
  /// @brief Remove an L2 cache before its storage is destroyed.
  void unregister_l2_cache(L2Cache *cache);
  /// @brief Register a memory-side cache in this device-local domain.
  void register_memory_side_cache(MemorySideCache *cache);
  /// @brief Remove a memory-side cache before its storage is destroyed.
  void unregister_memory_side_cache(MemorySideCache *cache);

  /// @brief Whether every registered cache can publish legacy dirty state
  /// directly to the SoC backing store during out-of-band maintenance.
  [[nodiscard]] bool has_legacy_maintenance_backing() const;

private:
  friend class DeviceCacheMaintenanceLease;

  void complete_cache_maintenance(size_t locked_l2_count, size_t locked_memory_side_count) noexcept;
  void release_cache_locks(size_t locked_l2_count, size_t locked_memory_side_count) noexcept;

  std::mutex atomic_mutex_;
  mutable std::shared_mutex mutex_;
  std::atomic<uint64_t> data_epoch_{1};
  std::atomic<uint64_t> instruction_epoch_{1};
  std::vector<L2Cache *> l2_caches_;
  std::vector<MemorySideCache *> memory_side_caches_;
};

} // namespace amdgpu
} // namespace rocjitsu
