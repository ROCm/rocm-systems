// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "util/except.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace rocjitsu {
namespace amdgpu {

class L2Cache;
class L1ScalarCache;
class L1VectorCache;
class MemorySideCache;
class DeviceCacheCoherence;
class PreparedCanonicalWriteback;
enum class VmAccessOutcome : uint8_t;

/// @brief Cache disposition requested at a device-wide maintenance boundary.
enum class DeviceCacheOperation {
  WritebackInvalidate,
  Invalidate,
};

/// @brief A cache-maintenance refusal before the protected external effect.
/// @details The caller may retry the same packet and continuation. Only throw
/// this after proving that no packet effect or partial prepared write occurred.
class CacheMaintenanceUnavailable final : public util::Exception {
public:
  explicit CacheMaintenanceUnavailable(std::string_view message) : util::Exception(message) {}
};

/// @brief One cache domain and the physical GPU that owns its dirty tags.
struct DeviceCacheDomain {
  std::shared_ptr<DeviceCacheCoherence> coherence;
  uint32_t gpu_id = 0;
};

/// @brief One exact resident alias span selected for scoped cache maintenance.
/// @details The address and size describe complete cache lines. The raw
/// admission generation and VM cache identity prevent a reused VMID or VA
/// from selecting a different backing generation.
struct DeviceCacheAliasScope {
  DeviceCacheDomain domain;
  uint32_t vmid = 0;
  uint64_t address = 0;
  uint64_t size = 0;
  uint64_t admission_generation = 0;
  uint64_t cache_identity = 0;
  uint64_t backing_id = 0;
  uint64_t backing_offset = 0;
};

/// @brief Prepare one complete dirty cache line for its own GPU and VMID.
/// @details The callback runs with no cache or domain locks held.
using PreparedWritebackResolver = std::function<PreparedCanonicalWriteback(
    uint32_t gpu_id, uint32_t vmid, uint64_t line_addr, size_t line_size)>;

/// @brief Read-only canonical identity of one unselected dirty resident line.
/// @details The type-erased retention pins the live alias grant and backing
/// without exposing a write or effect permit to the cache coordinator.
struct CanonicalLineWitness {
  uint32_t gpu_id = 0;
  uint32_t vmid = 0;
  uint64_t line_address = 0;
  size_t line_size = 0;
  uint64_t cache_identity = 0;
  uint64_t admission_generation = 0;
  uint64_t backing_id = 0;
  uint64_t backing_offset = 0;
  std::shared_ptr<const void> retention;
};

using CanonicalLineWitnessResolver = std::function<std::optional<CanonicalLineWitness>(
    uint32_t gpu_id, uint32_t vmid, uint64_t line_addr, size_t line_size, uint64_t cache_identity,
    uint64_t admission_generation)>;

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

  /// @brief Whether this lease published the atomic data epoch before lane execution.
  [[nodiscard]] bool prepared_atomic_boundary_for(const DeviceCacheCoherence *owner) const {
    return owner_ == owner && atomic_data_epoch_published_;
  }

private:
  friend class DeviceCacheCoherence;

  DeviceCacheMaintenanceLease(DeviceCacheCoherence *owner, size_t locked_l2_count,
                              size_t locked_memory_side_count,
                              std::unique_lock<std::mutex> atomic_lock,
                              std::unique_lock<std::shared_mutex> coherence_lock,
                              bool publish_epochs = true, size_t locked_scalar_count = 0,
                              size_t locked_vector_count = 0) noexcept;
  DeviceCacheCoherence *owner_ = nullptr;
  std::shared_ptr<DeviceCacheCoherence> owner_handle_;
  // Every group lease retains every prepared grant. The final copy is
  // destroyed after its destructor has released the last cache lock.
  std::shared_ptr<std::vector<PreparedCanonicalWriteback>> prepared_writebacks_;
  // Read-only alias permits obey the same release order.
  std::shared_ptr<std::vector<CanonicalLineWitness>> read_witnesses_;
  size_t locked_l2_count_ = 0;
  size_t locked_memory_side_count_ = 0;
  size_t locked_scalar_count_ = 0;
  size_t locked_vector_count_ = 0;
  bool publish_epochs_ = true;
  bool atomic_data_epoch_published_ = false;
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
/// Ordinary cache accesses take no coordinator lock. L2 compares its epoch
/// under its maintenance_mutex_; a private L1 compares under its own mutex.
/// Full-domain boundaries still use lazy L1 epoch reconciliation. Scoped
/// maintenance instead locks registered L1s before L2 and invalidates only
/// selected resident tags while the returned lease excludes new L1 accesses.
class DeviceCacheCoherence {
public:
  DeviceCacheCoherence() = default;

  class LocalCacheAccess {
  public:
    LocalCacheAccess(LocalCacheAccess &&) noexcept = default;
    LocalCacheAccess &operator=(LocalCacheAccess &&) noexcept = default;
    LocalCacheAccess(const LocalCacheAccess &) = delete;
    LocalCacheAccess &operator=(const LocalCacheAccess &) = delete;

    [[nodiscard]] bool belongs_to(const DeviceCacheCoherence *domain) const {
      return domain == owner_ && lock_.owns_lock();
    }

  private:
    friend class DeviceCacheCoherence;
    explicit LocalCacheAccess(const DeviceCacheCoherence *owner)
        : owner_(owner), lock_(owner->mutex_) {}

    const DeviceCacheCoherence *owner_;
    std::shared_lock<std::shared_mutex> lock_;
  };

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

    /// @brief Whether this boundary holds the specified coherence domain.
    [[nodiscard]] bool belongs_to(const DeviceCacheCoherence *owner) const {
      return owner_ != nullptr && owner_ == owner;
    }

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
  /// @details WritebackInvalidate acquires all L2 and memory-side locks, then
  /// publishes memory-side whole lines before L2 dirty runs. The returned
  /// lease advances the data and instruction epochs only after the protected
  /// access has completed and the lease is destroyed.
  [[nodiscard]] DeviceCacheMaintenanceLease
  acquire_cache_maintenance(DeviceCacheOperation operation);
  /// @brief Quiesce a group of device domains for one external access.
  /// @details Acquires every domain and cache lock before publishing dirty
  /// state. All memory-side lines are written before any L2 dirty runs, even
  /// when the caches belong to different domains. This legacy overload rejects
  /// dirty nonzero-VMID lines before any backing write.
  /// Distinct domains claiming the same VMID-zero backing line also fail
  /// before writeback because their dirty-writer order is unknown.
  /// Domains must be non-null and distinct. Release the returned leases in
  /// reverse order after the access; each lease retains its domain until release.
  [[nodiscard]] static std::vector<DeviceCacheMaintenanceLease>
  acquire_cache_maintenance_group(std::span<const std::shared_ptr<DeviceCacheCoherence>> domains,
                                  DeviceCacheOperation operation);
  /// @brief Quiesce GPU-labelled domains with prepared nonzero-VMID writeback.
  /// @details Each dirty line is resolved without cache locks. The complete
  /// snapshot and canonical backing ownership are checked before all required
  /// VM gates are entered. Multiple writes require retained direct byte views;
  /// a generic fallible backing can supply only one write. Mixed VMID-zero and
  /// nonzero groups are refused; zero-VMID-only groups use the legacy path.
  [[nodiscard]] static std::vector<DeviceCacheMaintenanceLease>
  acquire_cache_maintenance_group(std::span<const DeviceCacheDomain> domains,
                                  DeviceCacheOperation operation,
                                  const PreparedWritebackResolver &resolve_writeback);
  /// @brief Publish dirty state and the data epoch before a CU atomic's first lane.
  /// @details The returned leases retain all cache locks and prepared grants
  /// through every lane. Release publishes only the instruction epoch, so one
  /// vector attempt has one data-cache invalidation boundary.
  [[nodiscard]] static std::vector<DeviceCacheMaintenanceLease>
  acquire_prepared_atomic_boundary(std::span<const DeviceCacheDomain> domains,
                                   const PreparedWritebackResolver &resolve_writeback);
  /// @brief Publish and invalidate only resident lines matching exact aliases.
  /// @details All dirty resident lines are resolved to prove that no
  /// unselected writer overlaps a selected canonical backing line. Selected
  /// clean and dirty L2 and memory-side lines are revalidated before commit.
  /// Domains must include every GPU cache that could hold a dirty alias of the
  /// selected canonical backing. The returned leases retain every prepared
  /// grant and release cache locks without advancing device-wide data or
  /// instruction epochs. This requires the caller to exclude selected alias
  /// ingress through grant retirement and to prove that no live unselected
  /// alias can observe stale private-cache bytes for the same backing lines.
  /// @note L1's synchronous L2 fill currently rejects deferred peer access.
  /// A future cacheable deferred L1 path must release the L1 mutex around the
  /// transfer and revalidate its fill before scoped maintenance can use it.
  [[nodiscard]] static std::vector<DeviceCacheMaintenanceLease>
  acquire_scoped_alias_maintenance_group(std::span<const DeviceCacheDomain> domains,
                                         std::span<const DeviceCacheAliasScope> aliases,
                                         const PreparedWritebackResolver &resolve_writeback,
                                         const CanonicalLineWitnessResolver &resolve_witness = {});
  /// @brief Return the epoch observed by data caches during ordinary accesses.
  uint64_t current_epoch() const { return data_epoch_.load(std::memory_order_acquire); }
  /// @brief Return the epoch observed by private instruction caches.
  uint64_t current_instruction_epoch() const {
    return instruction_epoch_.load(std::memory_order_acquire);
  }

  /// @brief Serialize translated local cache hits and fills with fabric effects.
  /// @details The caller takes this before a private cache lock and releases it
  /// after copying the architectural bytes. It never treats a PASID as a VMID.
  [[nodiscard]] LocalCacheAccess acquire_local_cache_access() const {
    return LocalCacheAccess(this);
  }
  [[nodiscard]] uint64_t current_local_backing_epoch() const {
    return local_backing_epoch_.load(std::memory_order_acquire);
  }
  /// @brief Version the physical line selected by a translated local cache tag.
  /// @details The fixed table can collide, causing conservative extra misses.
  [[nodiscard]] uint64_t current_local_line_epoch(uint64_t backing_address) const;
  /// @brief Publish a bounded local VRAM write after its backing effect.
  /// @details Small writes retire only matching physical-line slots. Large or
  /// invalid ranges conservatively retire every translated local line. This
  /// may be called while an SDMA cache-maintenance lease is held.
  void publish_local_backing_write(uint64_t backing_address, size_t size);
  /// @brief Retire every translated local line after an HDP or fabric effect.
  /// @details Legacy dirty lines are never discarded here.
  void publish_local_backing_change();
  void mark_translated_local_cache_used() noexcept {
    translated_local_cache_used_.store(true, std::memory_order_release);
  }

  /// @brief Register an L2 cache in this device-local coherence domain.
  void register_l2_cache(L2Cache *cache);
  /// @brief Remove an L2 cache before its storage is destroyed.
  void unregister_l2_cache(L2Cache *cache);
  /// @brief Register a private scalar cache for exact scoped invalidation.
  void register_l1_scalar_cache(L1ScalarCache *cache);
  void unregister_l1_scalar_cache(L1ScalarCache *cache);
  /// @brief Register a private vector cache for exact scoped invalidation.
  void register_l1_vector_cache(L1VectorCache *cache);
  void unregister_l1_vector_cache(L1VectorCache *cache);
  /// @brief Register a memory-side cache in this device-local domain.
  void register_memory_side_cache(MemorySideCache *cache);
  /// @brief Remove a memory-side cache before its storage is destroyed.
  void unregister_memory_side_cache(MemorySideCache *cache);

  /// @brief Whether every registered cache can publish legacy dirty state
  /// directly to the SoC backing store during out-of-band maintenance.
  [[nodiscard]] bool has_legacy_maintenance_backing() const;

private:
  friend class DeviceCacheMaintenanceLease;

  static constexpr size_t kLocalLineEpochSlots = 8192;
  [[nodiscard]] static size_t local_line_slot(uint64_t backing_address);

  [[nodiscard]] static std::vector<DeviceCacheMaintenanceLease>
  acquire_prepared_cache_maintenance_group(
      std::span<const DeviceCacheDomain> domains, DeviceCacheOperation operation,
      const PreparedWritebackResolver &resolve_writeback,
      std::span<const DeviceCacheAliasScope> aliases, bool scoped,
      const CanonicalLineWitnessResolver &resolve_witness = {});

  void complete_cache_maintenance(size_t locked_l2_count, size_t locked_memory_side_count,
                                  bool publish_data_epoch) noexcept;
  void release_cache_locks(size_t locked_l2_count, size_t locked_memory_side_count,
                           size_t locked_scalar_count = 0, size_t locked_vector_count = 0) noexcept;

  std::mutex atomic_mutex_;
  mutable std::shared_mutex mutex_;
  std::atomic<uint64_t> data_epoch_{1};
  std::atomic<uint64_t> instruction_epoch_{1};
  std::atomic<uint64_t> local_backing_epoch_{1};
  std::array<std::atomic<uint64_t>, kLocalLineEpochSlots> local_line_epochs_{};
  std::atomic<bool> translated_local_cache_used_{false};
  // Protected by mutex_; detects unregister/register ABA across preparation.
  uint64_t cache_registry_generation_ = 1;
  std::vector<L1ScalarCache *> l1_scalar_caches_;
  std::vector<L1VectorCache *> l1_vector_caches_;
  std::vector<L2Cache *> l2_caches_;
  std::vector<MemorySideCache *> memory_side_caches_;
};

} // namespace amdgpu
} // namespace rocjitsu
