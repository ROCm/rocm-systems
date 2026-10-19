// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/device_cache_coherence.h"

#include "rocjitsu/vm/amdgpu/l1_scalar_cache.h"
#include "rocjitsu/vm/amdgpu/l1_vector_cache.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/memory_side_cache.h"
#include "rocjitsu/vm/prepared_canonical_writeback.h"
#include "util/except.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <functional>
#include <limits>
#include <string_view>

namespace rocjitsu {
namespace amdgpu {
namespace {

template <typename T> void unregister_cache(std::vector<T *> &caches, T *cache) {
  typename std::vector<T *>::iterator position = std::ranges::find(caches, cache);
  assert(position != caches.end());
  if (position != caches.end())
    caches.erase(position);
}

[[noreturn]] void throw_legacy_flush_failure(VmAccessOutcome outcome, std::string_view message) {
  // An unavailable legacy backing has not applied the protected external
  // packet effect. Completed cache-line publications remain authoritative and
  // need not be replayed; the still-dirty line can be retried.
  if (outcome == VmAccessOutcome::Unavailable)
    throw CacheMaintenanceUnavailable(message);
  throw util::Exception(message);
}

} // namespace

DeviceCacheMaintenanceLease::DeviceCacheMaintenanceLease(
    DeviceCacheCoherence *owner, size_t locked_l2_count, size_t locked_memory_side_count,
    std::unique_lock<std::mutex> atomic_lock, std::unique_lock<std::shared_mutex> coherence_lock,
    bool publish_epochs, size_t locked_scalar_count, size_t locked_vector_count) noexcept
    : owner_(owner), locked_l2_count_(locked_l2_count),
      locked_memory_side_count_(locked_memory_side_count),
      locked_scalar_count_(locked_scalar_count), locked_vector_count_(locked_vector_count),
      publish_epochs_(publish_epochs), atomic_lock_(std::move(atomic_lock)),
      coherence_lock_(std::move(coherence_lock)) {}

DeviceCacheMaintenanceLease::DeviceCacheMaintenanceLease(std::function<void()> release)
    : release_(std::move(release)) {}

DeviceCacheMaintenanceLease::DeviceCacheMaintenanceLease(
    DeviceCacheMaintenanceLease &&other) noexcept
    : owner_(std::exchange(other.owner_, nullptr)), owner_handle_(std::move(other.owner_handle_)),
      prepared_writebacks_(std::move(other.prepared_writebacks_)),
      read_witnesses_(std::move(other.read_witnesses_)),
      locked_l2_count_(std::exchange(other.locked_l2_count_, 0)),
      locked_memory_side_count_(std::exchange(other.locked_memory_side_count_, 0)),
      locked_scalar_count_(std::exchange(other.locked_scalar_count_, 0)),
      locked_vector_count_(std::exchange(other.locked_vector_count_, 0)),
      publish_epochs_(other.publish_epochs_),
      atomic_data_epoch_published_(other.atomic_data_epoch_published_),
      atomic_lock_(std::move(other.atomic_lock_)),
      coherence_lock_(std::move(other.coherence_lock_)),
      release_(std::exchange(other.release_, std::function<void()>{})) {}

DeviceCacheMaintenanceLease::~DeviceCacheMaintenanceLease() noexcept {
  if (owner_ != nullptr) {
    if (publish_epochs_)
      owner_->complete_cache_maintenance(locked_l2_count_, locked_memory_side_count_,
                                         !atomic_data_epoch_published_);
    else
      owner_->release_cache_locks(locked_l2_count_, locked_memory_side_count_, locked_scalar_count_,
                                  locked_vector_count_);
  }
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

void DeviceCacheCoherence::publish_local_backing_change() {
  // An HDP flush or fabric effect may hold an exclusive maintenance lease.
  // Publication cannot reacquire that lease. Readers compare the epoch before
  // and after a fill, and every later hit compares it again.
  [[maybe_unused]] const uint64_t next_epoch =
      local_backing_epoch_.fetch_add(1, std::memory_order_release) + 1;
  assert(next_epoch != 0 && "translated local backing epoch wrapped");
}

size_t DeviceCacheCoherence::local_line_slot(uint64_t backing_address) {
  static_assert(L1VectorCache::LINE_SIZE == L2Cache::LINE_SIZE);
  static_assert((kLocalLineEpochSlots & (kLocalLineEpochSlots - 1)) == 0);
  return (backing_address / L2Cache::LINE_SIZE) & (kLocalLineEpochSlots - 1);
}

uint64_t DeviceCacheCoherence::current_local_line_epoch(uint64_t backing_address) const {
  return local_line_epochs_[local_line_slot(backing_address)].load(std::memory_order_acquire);
}

void DeviceCacheCoherence::publish_local_backing_write(uint64_t backing_address, size_t size) {
  constexpr size_t kMaximumScopedWrite = 4096;
  if (size == 0)
    return;
  if (size > kMaximumScopedWrite || size - 1 > UINT64_MAX - backing_address) {
    publish_local_backing_change();
    return;
  }
  const uint64_t first_line = backing_address / L2Cache::LINE_SIZE;
  const uint64_t last_line = (backing_address + size - 1) / L2Cache::LINE_SIZE;
  for (uint64_t line = first_line; line <= last_line; ++line) {
    [[maybe_unused]] const uint64_t next_epoch =
        local_line_epochs_[line & (kLocalLineEpochSlots - 1)].fetch_add(1,
                                                                        std::memory_order_release) +
        1;
    assert(next_epoch != 0 && "translated local line epoch wrapped");
  }
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
    for (MemorySideCache *cache : memory_side_caches_) {
      cache->access_gate_.lock();
      ++locked_memory_side_count;
    }

    // This path can run on the standalone SDMA worker. It must never enter the
    // simulation topology from that thread, so legacy dirty state is published
    // directly to the SoC's GpuMemory backing while all cache access is excluded.
    // An MSC line can contain an older whole-line image than L2's dirty bytes.
    // Publish every MSC first, then overlay the newer L2 dirty runs.
    if (operation == DeviceCacheOperation::WritebackInvalidate) {
      for (MemorySideCache *cache : memory_side_caches_) {
        const VmAccessOutcome outcome = cache->flush_dirty_to_legacy_backing_locked();
        if (outcome != VmAccessOutcome::Complete)
          throw_legacy_flush_failure(outcome, "memory-side cache maintenance backing write failed");
      }
      for (L2Cache *cache : l2_caches_) {
        const VmAccessOutcome outcome = cache->flush_dirty_to_legacy_backing_locked();
        if (outcome != VmAccessOutcome::Complete)
          throw_legacy_flush_failure(outcome, "L2 cache maintenance backing write failed");
      }
    }

  } catch (...) {
    release_cache_locks(locked_l2_count, locked_memory_side_count);
    throw;
  }

  return DeviceCacheMaintenanceLease(this, locked_l2_count, locked_memory_side_count,
                                     std::move(atomic_lock), std::move(coherence_lock));
}

std::vector<DeviceCacheMaintenanceLease> DeviceCacheCoherence::acquire_cache_maintenance_group(
    std::span<const std::shared_ptr<DeviceCacheCoherence>> domains,
    DeviceCacheOperation operation) {
  std::vector<std::shared_ptr<DeviceCacheCoherence>> ordered_domains(domains.begin(),
                                                                     domains.end());
  if (std::ranges::any_of(ordered_domains, [](const auto &domain) { return !domain; }))
    throw util::ConfigError("cache maintenance domains must be non-null");
  std::ranges::sort(ordered_domains, [](const auto &lhs, const auto &rhs) {
    return std::less<DeviceCacheCoherence *>{}(lhs.get(), rhs.get());
  });
  if (std::ranges::adjacent_find(ordered_domains, [](const auto &lhs, const auto &rhs) {
        return lhs.get() == rhs.get();
      }) != ordered_domains.end())
    throw util::ConfigError("cache maintenance domains must be distinct");

  struct DomainLocks {
    std::shared_ptr<DeviceCacheCoherence> owner;
    std::unique_lock<std::mutex> atomic_lock;
    std::unique_lock<std::shared_mutex> coherence_lock;
    size_t locked_l2_count = 0;
    size_t locked_memory_side_count = 0;
  };
  std::vector<DomainLocks> locked;
  std::vector<DeviceCacheMaintenanceLease> leases;
  locked.reserve(ordered_domains.size());
  leases.reserve(ordered_domains.size());

  try {
    // Use one total domain order for every group so concurrent fabric effects
    // cannot acquire the same coordinator locks in opposite orders.
    for (const auto &domain : ordered_domains) {
      DomainLocks &entry = locked.emplace_back();
      entry.owner = domain;
      entry.atomic_lock = std::unique_lock(domain->atomic_mutex_);
      entry.coherence_lock = std::unique_lock(domain->mutex_);
    }

    // Check the entire hierarchy before writing any dirty cache state.
    if (operation == DeviceCacheOperation::WritebackInvalidate) {
      for (const DomainLocks &entry : locked) {
        const bool missing_l2_backing =
            std::ranges::any_of(entry.owner->l2_caches_, [](const L2Cache *cache) {
              return !cache->has_legacy_maintenance_backing();
            });
        const bool missing_memory_side_backing =
            std::ranges::any_of(entry.owner->memory_side_caches_, [](const MemorySideCache *cache) {
              return !cache->has_legacy_maintenance_backing();
            });
        if (missing_l2_backing || missing_memory_side_backing)
          throw util::ConfigError("cache hierarchy lacks legacy maintenance backing");
      }
    }

    for (DomainLocks &entry : locked) {
      for (L2Cache *cache : entry.owner->l2_caches_) {
        cache->maintenance_mutex_.lock();
        ++entry.locked_l2_count;
      }
      for (MemorySideCache *cache : entry.owner->memory_side_caches_) {
        cache->access_gate_.lock();
        ++entry.locked_memory_side_count;
      }
    }

    if (operation == DeviceCacheOperation::WritebackInvalidate) {
      // The current group sink is direct SparseMemory only. A nonzero VMID
      // needs a prepared canonical backing and exact VM gate; attempting its
      // legacy write under these locks can fail after an earlier domain wrote.
      // Every dirty tag must be checked before any backing byte is changed.
      struct DirtyBackingLine {
        GpuMemory *memory;
        uint64_t address;
        DeviceCacheCoherence *domain;
      };
      static_assert(L2Cache::LINE_SIZE == MemorySideCache::LINE_SIZE);
      std::vector<DirtyBackingLine> dirty_lines;
      const auto collect_line = [&dirty_lines](const DomainLocks &entry, GpuMemory *memory,
                                               const simdojo::CacheTag &tag, uint64_t line_addr) {
        if (tag.vmid != 0)
          throw util::Exception(
              std::string_view("group maintenance requires prepared nonzero-VMID writeback"));
        // A resident dirty line is one bounded candidate writer. Keep its
        // backing identity rather than comparing virtual addresses alone.
        dirty_lines.push_back({memory, line_addr, entry.owner.get()});
      };
      for (const DomainLocks &entry : locked) {
        for (MemorySideCache *cache : entry.owner->memory_side_caches_)
          if (cache->may_have_dirty_lines_.load(std::memory_order_relaxed))
            cache->cache_.for_each_dirty(
                [&](const simdojo::CacheTag &tag, uint64_t line_addr, uint8_t *) {
                  collect_line(entry, cache->legacy_maintenance_memory_, tag, line_addr);
                });
        for (L2Cache *cache : entry.owner->l2_caches_)
          if (cache->has_dirty_lines_.load(std::memory_order_relaxed))
            cache->cache_.for_each_dirty(
                [&](const simdojo::CacheTag &tag, uint64_t line_addr, uint8_t *) {
                  collect_line(entry, cache->legacy_maintenance_memory_, tag, line_addr);
                });
      }

      // MSC-before-L2 establishes the order within one domain. Distinct
      // domains have no dirty-writer ordering, even if their line addresses
      // match. Refuse an ambiguous shared backing line before either writes.
      std::ranges::sort(dirty_lines, [](const DirtyBackingLine &lhs, const DirtyBackingLine &rhs) {
        if (lhs.memory != rhs.memory)
          return std::less<GpuMemory *>{}(lhs.memory, rhs.memory);
        if (lhs.address != rhs.address)
          return lhs.address < rhs.address;
        return std::less<DeviceCacheCoherence *>{}(lhs.domain, rhs.domain);
      });
      for (size_t i = 1; i < dirty_lines.size(); ++i) {
        const DirtyBackingLine &previous = dirty_lines[i - 1];
        const DirtyBackingLine &current = dirty_lines[i];
        if (previous.memory == current.memory && previous.address == current.address &&
            previous.domain != current.domain)
          throw util::Exception(
              std::string_view("group maintenance has overlapping dirty backing lines"));
      }

      // SparseMemory allocates on first write. Reserve all target pages while
      // every cache is quiesced, before the first dirty-byte publication. A
      // failed allocation leaves at most newly allocated zero pages.
      const auto reserve_line = [](GpuMemory *memory, uint64_t line_addr, size_t size) {
        if (!memory->reserve_write_pages(line_addr, size))
          throw util::Exception(std::string_view("cache maintenance backing span overflows"));
      };
      for (const DomainLocks &entry : locked) {
        for (MemorySideCache *cache : entry.owner->memory_side_caches_)
          if (cache->may_have_dirty_lines_.load(std::memory_order_relaxed))
            cache->cache_.for_each_dirty(
                [&](const simdojo::CacheTag &, uint64_t line_addr, uint8_t *) {
                  reserve_line(cache->legacy_maintenance_memory_, line_addr,
                               MemorySideCache::LINE_SIZE);
                });
        for (L2Cache *cache : entry.owner->l2_caches_)
          if (cache->has_dirty_lines_.load(std::memory_order_relaxed))
            cache->cache_.for_each_dirty(
                [&](const simdojo::CacheTag &, uint64_t line_addr, uint8_t *) {
                  reserve_line(cache->legacy_maintenance_memory_, line_addr, L2Cache::LINE_SIZE);
                });
      }
    }

    // A later domain's whole MSC line may be older than another domain's L2
    // bytes. Publish every MSC first, then overlay all L2 dirty runs.
    if (operation == DeviceCacheOperation::WritebackInvalidate) {
      for (const DomainLocks &entry : locked) {
        for (MemorySideCache *cache : entry.owner->memory_side_caches_) {
          const VmAccessOutcome outcome = cache->flush_dirty_to_legacy_backing_locked();
          if (outcome != VmAccessOutcome::Complete)
            throw_legacy_flush_failure(outcome,
                                       "memory-side cache maintenance backing write failed");
        }
      }
      for (const DomainLocks &entry : locked) {
        for (L2Cache *cache : entry.owner->l2_caches_) {
          const VmAccessOutcome outcome = cache->flush_dirty_to_legacy_backing_locked();
          if (outcome != VmAccessOutcome::Complete)
            throw_legacy_flush_failure(outcome, "L2 cache maintenance backing write failed");
        }
      }
    }
  } catch (...) {
    // A failed preflush creates no lease and publishes no coherence epoch.
    for (auto entry = locked.rbegin(); entry != locked.rend(); ++entry) {
      entry->owner->release_cache_locks(entry->locked_l2_count, entry->locked_memory_side_count);
      if (entry->coherence_lock.owns_lock())
        entry->coherence_lock.unlock();
      if (entry->atomic_lock.owns_lock())
        entry->atomic_lock.unlock();
    }
    throw;
  }

  // Storage was reserved before acquiring locks. Constructing and moving a
  // lease cannot fail after the preflush has changed backing memory.
  for (DomainLocks &entry : locked) {
    DeviceCacheMaintenanceLease lease(entry.owner.get(), entry.locked_l2_count,
                                      entry.locked_memory_side_count, std::move(entry.atomic_lock),
                                      std::move(entry.coherence_lock));
    lease.owner_handle_ = entry.owner;
    leases.push_back(std::move(lease));
  }
  return leases;
}

std::vector<DeviceCacheMaintenanceLease> DeviceCacheCoherence::acquire_cache_maintenance_group(
    std::span<const DeviceCacheDomain> domains, DeviceCacheOperation operation,
    const PreparedWritebackResolver &resolve_writeback) {
  return acquire_prepared_cache_maintenance_group(domains, operation, resolve_writeback, {}, false);
}

std::vector<DeviceCacheMaintenanceLease> DeviceCacheCoherence::acquire_prepared_atomic_boundary(
    std::span<const DeviceCacheDomain> domains,
    const PreparedWritebackResolver &resolve_writeback) {
  std::vector<DeviceCacheMaintenanceLease> leases = acquire_cache_maintenance_group(
      domains, DeviceCacheOperation::WritebackInvalidate, resolve_writeback);
  // Group acquisition has published every dirty MSC line and L2 run and still
  // holds all cache locks. Any L1 access that observes this epoch must refetch
  // through those locks, so no stale line can race an atomic lane.
  for (DeviceCacheMaintenanceLease &lease : leases) {
    assert(lease.owner_ != nullptr && lease.atomic_lock_.owns_lock() &&
           lease.coherence_lock_.owns_lock());
    [[maybe_unused]] const uint64_t next_epoch =
        lease.owner_->data_epoch_.fetch_add(1, std::memory_order_release) + 1;
    assert(next_epoch != 0 && "device cache coherence epoch wrapped");
    lease.atomic_data_epoch_published_ = true;
  }
  return leases;
}

std::vector<DeviceCacheMaintenanceLease>
DeviceCacheCoherence::acquire_scoped_alias_maintenance_group(
    std::span<const DeviceCacheDomain> domains, std::span<const DeviceCacheAliasScope> aliases,
    const PreparedWritebackResolver &resolve_writeback,
    const CanonicalLineWitnessResolver &resolve_witness) {
  static_assert(L2Cache::LINE_SIZE == MemorySideCache::LINE_SIZE);
  if (domains.empty() || aliases.empty())
    throw util::ConfigError("scoped cache maintenance needs domains and aliases");
  for (const DeviceCacheAliasScope &alias : aliases) {
    if (!alias.domain.coherence || alias.vmid == 0 || alias.admission_generation == 0 ||
        alias.cache_identity == 0 || alias.size == 0 ||
        (alias.address & L2Cache::CacheStore::LINE_MASK) != 0 ||
        (alias.size & L2Cache::CacheStore::LINE_MASK) != 0 ||
        alias.size - 1 > std::numeric_limits<uint64_t>::max() - alias.address)
      throw util::ConfigError("scoped cache aliases need complete nonzero-VMID lines");
    const auto existing = std::ranges::find_if(domains, [&](const DeviceCacheDomain &domain) {
      return domain.coherence.get() == alias.domain.coherence.get();
    });
    if (existing == domains.end() || existing->gpu_id != alias.domain.gpu_id) {
      throw util::ConfigError("scoped cache aliases disagree on domain GPU ID");
    }
    if (resolve_witness &&
        (alias.backing_id == 0 ||
         alias.size > std::numeric_limits<uint64_t>::max() - alias.address ||
         (alias.backing_offset & L2Cache::CacheStore::LINE_MASK) != 0 ||
         alias.size > std::numeric_limits<uint64_t>::max() - alias.backing_offset))
      throw util::ConfigError("scoped cache witness needs canonical alias lines");
  }
  // Duplicate selected scopes are harmless only when every shared VA byte
  // names the same resident identity and canonical backing byte.
  if (resolve_witness)
    for (size_t first = 0; first < aliases.size(); ++first)
      for (size_t second = first + 1; second < aliases.size(); ++second) {
        const DeviceCacheAliasScope &left = aliases[first];
        const DeviceCacheAliasScope &right = aliases[second];
        if (left.domain.coherence.get() != right.domain.coherence.get() ||
            left.domain.gpu_id != right.domain.gpu_id || left.vmid != right.vmid)
          continue;
        const uint64_t overlap_address = std::max(left.address, right.address);
        if ((left.address <= right.address ? right.address - left.address >= left.size
                                           : left.address - right.address >= right.size))
          continue;
        if (left.cache_identity == right.cache_identity &&
            left.admission_generation == right.admission_generation &&
            left.backing_id == right.backing_id &&
            left.backing_offset + (overlap_address - left.address) ==
                right.backing_offset + (overlap_address - right.address))
          continue;
        throw util::ConfigError("scoped cache witness has conflicting selected aliases");
      }
  return acquire_prepared_cache_maintenance_group(
      domains, DeviceCacheOperation::WritebackInvalidate, resolve_writeback, aliases, true,
      resolve_witness);
}

std::vector<DeviceCacheMaintenanceLease>
DeviceCacheCoherence::acquire_prepared_cache_maintenance_group(
    std::span<const DeviceCacheDomain> domains, DeviceCacheOperation operation,
    const PreparedWritebackResolver &resolve_writeback,
    std::span<const DeviceCacheAliasScope> aliases, bool scoped,
    const CanonicalLineWitnessResolver &resolve_witness) {
  std::vector<DeviceCacheDomain> ordered_domains(domains.begin(), domains.end());
  if (std::ranges::any_of(ordered_domains,
                          [](const DeviceCacheDomain &domain) { return !domain.coherence; }))
    throw util::ConfigError("cache maintenance domains must be non-null");
  std::ranges::sort(
      ordered_domains, [](const DeviceCacheDomain &lhs, const DeviceCacheDomain &rhs) {
        return std::less<DeviceCacheCoherence *>{}(lhs.coherence.get(), rhs.coherence.get());
      });
  if (std::ranges::adjacent_find(ordered_domains,
                                 [](const DeviceCacheDomain &lhs, const DeviceCacheDomain &rhs) {
                                   return lhs.coherence.get() == rhs.coherence.get();
                                 }) != ordered_domains.end())
    throw util::ConfigError("cache maintenance domains must be distinct");

  std::vector<std::shared_ptr<DeviceCacheCoherence>> legacy_domains;
  legacy_domains.reserve(ordered_domains.size());
  for (const DeviceCacheDomain &domain : ordered_domains)
    legacy_domains.push_back(domain.coherence);
  if (operation != DeviceCacheOperation::WritebackInvalidate)
    return acquire_cache_maintenance_group(legacy_domains, operation);
  if (!resolve_writeback)
    throw util::ConfigError("prepared cache writeback resolver is required");

  struct DomainLocks {
    DeviceCacheDomain domain;
    std::unique_lock<std::mutex> atomic_lock;
    std::unique_lock<std::shared_mutex> coherence_lock;
    size_t locked_scalar_count = 0;
    size_t locked_vector_count = 0;
    size_t locked_l2_count = 0;
    size_t locked_memory_side_count = 0;
  };
  struct CacheRegistry {
    DeviceCacheCoherence *domain;
    uint64_t generation;
    uint64_t data_epoch;
    uint64_t instruction_epoch;
    std::vector<L1ScalarCache *> scalar;
    std::vector<L1VectorCache *> vector;
    std::vector<L2Cache *> l2;
    std::vector<MemorySideCache *> memory_side;
  };
  struct DirtyLine {
    DeviceCacheCoherence *domain;
    uint32_t gpu_id;
    L2Cache *l2 = nullptr;
    MemorySideCache *memory_side = nullptr;
    uint64_t address;
    simdojo::CacheTag tag;
    bool selected = true;
    std::array<uint64_t, 2> dirty_mask{};
    std::array<uint8_t, L2Cache::LINE_SIZE> bytes{};
  };
  struct Snapshot {
    std::vector<CacheRegistry> registries;
    std::vector<DirtyLine> dirty;
    std::vector<DirtyLine> selected_clean;
  };

  std::vector<DomainLocks> locked;
  std::vector<DeviceCacheMaintenanceLease> leases;
  locked.reserve(ordered_domains.size());
  leases.reserve(ordered_domains.size());
  for (const DeviceCacheDomain &domain : ordered_domains)
    locked.emplace_back().domain = domain;

  // Dropping the first lock set for KFD/VM preparation must not publish an
  // epoch. The second lock set becomes the returned maintenance leases.
  const auto release_unpublished = [&locked]() noexcept {
    for (auto entry = locked.rbegin(); entry != locked.rend(); ++entry) {
      if (entry->locked_l2_count != 0 || entry->locked_memory_side_count != 0 ||
          entry->locked_scalar_count != 0 || entry->locked_vector_count != 0) {
        entry->domain.coherence->release_cache_locks(
            entry->locked_l2_count, entry->locked_memory_side_count, entry->locked_scalar_count,
            entry->locked_vector_count);
        entry->locked_scalar_count = 0;
        entry->locked_vector_count = 0;
        entry->locked_l2_count = 0;
        entry->locked_memory_side_count = 0;
      }
      if (entry->coherence_lock.owns_lock())
        entry->coherence_lock.unlock();
      if (entry->atomic_lock.owns_lock())
        entry->atomic_lock.unlock();
    }
  };
  const auto acquire_all = [&locked, scoped]() {
    // All domains and their cache registries use one total lock order.
    for (DomainLocks &entry : locked) {
      DeviceCacheCoherence *const owner = entry.domain.coherence.get();
      entry.atomic_lock = std::unique_lock(owner->atomic_mutex_);
      entry.coherence_lock = std::unique_lock(owner->mutex_);
    }
    for (DomainLocks &entry : locked) {
      if (scoped) {
        // Ordinary L1 requests take L1 before L2. A synchronous L1 miss may
        // enter only the nondeferred L2 path; tokenized peer fills must split
        // the L1 lock and revalidate before they can use this protocol.
        for (L1ScalarCache *cache : entry.domain.coherence->l1_scalar_caches_) {
          cache->maintenance_mutex_.lock();
          ++entry.locked_scalar_count;
        }
        for (L1VectorCache *cache : entry.domain.coherence->l1_vector_caches_) {
          cache->maintenance_mutex_.lock();
          ++entry.locked_vector_count;
        }
      }
      for (L2Cache *cache : entry.domain.coherence->l2_caches_) {
        cache->maintenance_mutex_.lock();
        ++entry.locked_l2_count;
      }
      for (MemorySideCache *cache : entry.domain.coherence->memory_side_caches_) {
        cache->access_gate_.lock();
        ++entry.locked_memory_side_count;
      }
    }
  };
  const auto selected = [aliases, scoped](const DomainLocks &entry, const simdojo::CacheTag &tag,
                                          uint64_t address) {
    if (!scoped)
      return true;
    return std::ranges::any_of(aliases, [&](const DeviceCacheAliasScope &alias) {
      return alias.domain.coherence.get() == entry.domain.coherence.get() &&
             alias.domain.gpu_id == entry.domain.gpu_id && alias.vmid == tag.vmid &&
             alias.admission_generation == tag.admission_generation &&
             alias.cache_identity == tag.cache_identity && address >= alias.address &&
             address - alias.address < alias.size;
    });
  };
  const auto capture = [&locked, &selected, scoped]() {
    Snapshot snapshot;
    snapshot.registries.reserve(locked.size());
    for (const DomainLocks &entry : locked) {
      DeviceCacheCoherence *const owner = entry.domain.coherence.get();
      snapshot.registries.push_back({owner, owner->cache_registry_generation_,
                                     owner->data_epoch_.load(std::memory_order_acquire),
                                     owner->instruction_epoch_.load(std::memory_order_acquire),
                                     owner->l1_scalar_caches_, owner->l1_vector_caches_,
                                     owner->l2_caches_, owner->memory_side_caches_});
      // A scoped operation records every dirty writer for canonical-overlap
      // proof, but only matching residents are eligible for mutation.
      for (MemorySideCache *cache : owner->memory_side_caches_) {
        const auto collect = [&](const simdojo::CacheTag &tag, uint64_t address,
                                 const uint8_t *data) {
          if (data == nullptr)
            throw util::Exception(std::string_view("memory-side line has no bytes"));
          DirtyLine line{.domain = owner,
                         .gpu_id = entry.domain.gpu_id,
                         .memory_side = cache,
                         .address = address,
                         .tag = tag,
                         .selected = selected(entry, tag, address)};
          if (line.selected)
            std::copy_n(data, MemorySideCache::LINE_SIZE, line.bytes.begin());
          if (tag.dirty)
            snapshot.dirty.push_back(std::move(line));
          else if (line.selected)
            snapshot.selected_clean.push_back(std::move(line));
        };
        if (scoped)
          cache->cache_.for_each_valid_line(collect);
        else if (cache->may_have_dirty_lines_.load(std::memory_order_relaxed))
          cache->cache_.for_each_dirty(collect);
      }
      for (L2Cache *cache : owner->l2_caches_) {
        const auto collect = [&](const simdojo::CacheTag &tag, uint64_t address,
                                 const uint8_t *data) {
          if (data == nullptr)
            throw util::Exception(std::string_view("L2 line has no bytes"));
          DirtyLine line{.domain = owner,
                         .gpu_id = entry.domain.gpu_id,
                         .l2 = cache,
                         .address = address,
                         .tag = tag,
                         .selected = selected(entry, tag, address)};
          if (tag.dirty && line.selected) {
            std::lock_guard mask_lock(cache->dirty_bytes_mutex_);
            const auto position = cache->dirty_bytes_.find({tag.vmid, address});
            if (position == cache->dirty_bytes_.end())
              throw util::Exception(std::string_view("dirty L2 line has no byte mask"));
            line.dirty_mask = position->second;
          }
          if (line.selected)
            std::copy_n(data, L2Cache::LINE_SIZE, line.bytes.begin());
          if (tag.dirty)
            snapshot.dirty.push_back(std::move(line));
          else if (line.selected)
            snapshot.selected_clean.push_back(std::move(line));
        };
        if (scoped)
          cache->cache_.for_each_valid_line(collect);
        else if (cache->has_dirty_lines_.load(std::memory_order_relaxed))
          cache->cache_.for_each_dirty(collect);
      }
    }
    return snapshot;
  };
  const auto same_line = [](const DirtyLine &a, const DirtyLine &b) {
    const bool same_identity =
        a.domain == b.domain && a.gpu_id == b.gpu_id && a.l2 == b.l2 &&
        a.memory_side == b.memory_side && a.address == b.address && a.selected == b.selected &&
        a.tag.tag == b.tag.tag && a.tag.admission_generation == b.tag.admission_generation &&
        a.tag.cache_identity == b.tag.cache_identity &&
        a.tag.address_space_identity == b.tag.address_space_identity &&
        a.tag.backing_address == b.tag.backing_address &&
        a.tag.mapping_attributes == b.tag.mapping_attributes &&
        a.tag.local_backing_epoch == b.tag.local_backing_epoch &&
        a.tag.local_line_epoch == b.tag.local_line_epoch && a.tag.vmid == b.tag.vmid &&
        a.tag.valid == b.tag.valid && a.tag.dirty == b.tag.dirty;
    if (!same_identity)
      return false;
    if (!a.selected)
      return true;
    return a.dirty_mask == b.dirty_mask && a.bytes == b.bytes &&
           a.tag.coherence_epoch == b.tag.coherence_epoch &&
           a.tag.valid_bytes == b.tag.valid_bytes && a.tag.coherence == b.tag.coherence;
  };
  const auto same_snapshot = [&same_line](const Snapshot &lhs, const Snapshot &rhs) {
    if (lhs.registries.size() != rhs.registries.size() || lhs.dirty.size() != rhs.dirty.size() ||
        lhs.selected_clean.size() != rhs.selected_clean.size())
      return false;
    for (size_t i = 0; i < lhs.registries.size(); ++i) {
      const CacheRegistry &a = lhs.registries[i];
      const CacheRegistry &b = rhs.registries[i];
      if (a.domain != b.domain || a.generation != b.generation || a.data_epoch != b.data_epoch ||
          a.instruction_epoch != b.instruction_epoch || a.scalar != b.scalar ||
          a.vector != b.vector || a.l2 != b.l2 || a.memory_side != b.memory_side)
        return false;
    }
    for (size_t i = 0; i < lhs.dirty.size(); ++i)
      if (!same_line(lhs.dirty[i], rhs.dirty[i]))
        return false;
    for (size_t i = 0; i < lhs.selected_clean.size(); ++i)
      if (!same_line(lhs.selected_clean[i], rhs.selected_clean[i]))
        return false;
    return true;
  };

  struct PreparedLine {
    size_t snapshot_index;
    PreparedCanonicalWriteback::ResolvedLine backing;
  };
  struct CommitLine {
    const DirtyLine *snapshot;
    simdojo::CacheTag *tag;
    const uint8_t *source;
  };
  struct WriteOp {
    const uint8_t *source;
    std::byte *direct_destination;
    FabricBacking fallible_destination;
    uint32_t size;
    L2Cache *l2;
    uint64_t address;
  };
  std::shared_ptr<std::vector<PreparedCanonicalWriteback>> prepared;
  std::shared_ptr<std::vector<CanonicalLineWitness>> witnesses;
  try {
    acquire_all();
    const Snapshot before = capture();
    bool has_zero_writer = false;
    size_t prepared_write_count = 0;
    size_t prepared_line_count = 0;
    static_assert(L2Cache::LINE_SIZE == MemorySideCache::LINE_SIZE);
    for (const DirtyLine &line : before.dirty) {
      if (line.tag.vmid == 0) {
        has_zero_writer = true;
        continue;
      }
      ++prepared_line_count;
      if (!line.selected)
        continue;
      if (line.memory_side != nullptr) {
        if (!MemorySideCache::CacheStore::bytes_valid(line.tag, 0, MemorySideCache::LINE_SIZE))
          throw util::Exception(std::string_view("dirty memory-side line is incomplete"));
        ++prepared_write_count;
      } else {
        bool found_run = false;
        for (uint32_t byte = 0; byte < L2Cache::LINE_SIZE;) {
          while (byte < L2Cache::LINE_SIZE &&
                 (line.dirty_mask[byte / 64] & (uint64_t{1} << (byte % 64))) == 0)
            ++byte;
          const uint32_t begin = byte;
          while (byte < L2Cache::LINE_SIZE &&
                 (line.dirty_mask[byte / 64] & (uint64_t{1} << (byte % 64))) != 0)
            ++byte;
          if (begin == byte)
            continue;
          if (!L2Cache::CacheStore::bytes_valid(line.tag, begin, byte - begin))
            throw util::Exception(std::string_view("dirty L2 run is incomplete"));
          ++prepared_write_count;
          found_run = true;
        }
        if (!found_run)
          throw util::Exception(std::string_view("dirty L2 line has an empty byte mask"));
      }
    }
    if (scoped && has_zero_writer)
      throw CacheMaintenanceUnavailable("scoped maintenance cannot prove VMID-zero backing");
    if (!scoped && prepared_write_count == 0) {
      release_unpublished();
      return acquire_cache_maintenance_group(legacy_domains, operation);
    }
    if (has_zero_writer)
      throw CacheMaintenanceUnavailable("group maintenance has mixed dirty VMID writers");

    release_unpublished();
    prepared = std::make_shared<std::vector<PreparedCanonicalWriteback>>();
    prepared->reserve(prepared_line_count);
    if (resolve_witness)
      witnesses = std::make_shared<std::vector<CanonicalLineWitness>>();
    std::vector<PreparedLine> prepared_lines;
    prepared_lines.reserve(prepared_line_count);
    for (size_t index = 0; index < before.dirty.size(); ++index) {
      const DirtyLine &line = before.dirty[index];
      if (line.tag.vmid == 0)
        continue;
      if (scoped && !line.selected && resolve_witness) {
        std::optional<CanonicalLineWitness> witness =
            resolve_witness(line.gpu_id, line.tag.vmid, line.address, L2Cache::LINE_SIZE,
                            line.tag.cache_identity, line.tag.admission_generation);
        if (!witness || !witness->retention || witness->gpu_id != line.gpu_id ||
            witness->vmid != line.tag.vmid || witness->line_address != line.address ||
            witness->line_size != L2Cache::LINE_SIZE ||
            witness->cache_identity != line.tag.cache_identity ||
            witness->admission_generation != line.tag.admission_generation ||
            witness->backing_id == 0 ||
            (witness->backing_offset & L2Cache::CacheStore::LINE_MASK) != 0 ||
            witness->backing_offset > std::numeric_limits<uint64_t>::max() - L2Cache::LINE_SIZE)
          throw CacheMaintenanceUnavailable("unselected dirty line has no live canonical witness");
        witnesses->push_back(std::move(*witness));
        continue;
      }
      // Each callback resolves the resident tag's own GPU, VMID, and complete
      // line. Its counted grant remains live across every later callback.
      PreparedCanonicalWriteback candidate =
          resolve_writeback(line.gpu_id, line.tag.vmid, line.address, L2Cache::LINE_SIZE);
      std::optional<PreparedCanonicalWriteback::ResolvedLine> backing =
          candidate.resolve_line(line.address, L2Cache::LINE_SIZE, line.tag.vmid,
                                 line.tag.cache_identity, line.tag.admission_generation);
      // Only an unavailable grant can become ready without changing the
      // resolver contract. A Complete grant that omits this resident identity
      // or generation is a stable mismatch and must fault the queue.
      const VmAccessOutcome candidate_outcome = candidate.outcome();
      if (candidate_outcome == VmAccessOutcome::Unavailable)
        throw CacheMaintenanceUnavailable("prepared cache writeback is unavailable");
      if (candidate_outcome != VmAccessOutcome::Complete)
        throw util::Exception(std::string_view("prepared cache writeback was rejected"));
      if (!backing)
        throw util::Exception(
            std::string_view("prepared cache writeback does not cover dirty line"));
      if (resolve_witness && line.selected) {
        const auto matching =
            std::ranges::find_if(aliases, [&](const DeviceCacheAliasScope &scope) {
              return scope.domain.coherence.get() == line.domain &&
                     scope.domain.gpu_id == line.gpu_id && scope.vmid == line.tag.vmid &&
                     scope.admission_generation == line.tag.admission_generation &&
                     scope.cache_identity == line.tag.cache_identity &&
                     line.address >= scope.address && line.address - scope.address < scope.size;
            });
        if (matching == aliases.end() || backing->backing_id != matching->backing_id ||
            backing->backing_offset !=
                matching->backing_offset + (line.address - matching->address))
          throw util::Exception(std::string_view("selected dirty line changed canonical backing"));
      }
      prepared_lines.push_back({index, std::move(*backing)});
      prepared->push_back(std::move(candidate));
    }
    if (witnesses)
      for (const CanonicalLineWitness &witness : *witnesses)
        for (const DeviceCacheAliasScope &scope : aliases)
          if (witness.backing_id == scope.backing_id &&
              witness.backing_offset < scope.backing_offset + scope.size &&
              scope.backing_offset < witness.backing_offset + L2Cache::LINE_SIZE)
            throw CacheMaintenanceUnavailable(
                "scoped maintenance overlaps an unselected dirty alias");
    if (prepared_write_count > 1 &&
        std::ranges::any_of(prepared_lines, [&before](const PreparedLine &line) {
          return before.dirty[line.snapshot_index].selected && line.backing.direct_bytes.empty();
        }))
      throw util::Exception(
          std::string_view("multiple dirty writes require direct canonical backing"));

    // Compare full canonical lines, including aliases at different GPU VAs.
    // A single domain may overlay its L2 bytes on its own MSC line only when
    // both levels name the same resident VM line and generation.
    std::vector<size_t> backing_order(prepared_lines.size());
    for (size_t index = 0; index < backing_order.size(); ++index)
      backing_order[index] = index;
    std::ranges::sort(backing_order, [&prepared_lines](size_t first, size_t second) {
      const PreparedLine &left = prepared_lines[first];
      const PreparedLine &right = prepared_lines[second];
      if (left.backing.backing_id != right.backing.backing_id)
        return left.backing.backing_id < right.backing.backing_id;
      if (left.backing.backing_offset != right.backing.backing_offset)
        return left.backing.backing_offset < right.backing.backing_offset;
      return first < second;
    });
    for (size_t first = 0; first < backing_order.size(); ++first) {
      const PreparedLine &left = prepared_lines[backing_order[first]];
      for (size_t second = first + 1; second < backing_order.size(); ++second) {
        const PreparedLine &right = prepared_lines[backing_order[second]];
        if (left.backing.backing_id != right.backing.backing_id ||
            right.backing.backing_offset - left.backing.backing_offset >= L2Cache::LINE_SIZE)
          break;
        const DirtyLine &left_line = before.dirty[left.snapshot_index];
        const DirtyLine &right_line = before.dirty[right.snapshot_index];
        if (!left_line.selected && !right_line.selected)
          continue;
        if (left_line.selected != right_line.selected)
          throw CacheMaintenanceUnavailable(
              "scoped maintenance overlaps an unselected dirty alias");
        const bool ordered_overlay =
            left_line.domain == right_line.domain &&
            (left_line.memory_side != nullptr) != (right_line.memory_side != nullptr) &&
            left_line.address == right_line.address && left_line.tag.vmid == right_line.tag.vmid &&
            left_line.tag.cache_identity == right_line.tag.cache_identity &&
            left_line.tag.admission_generation == right_line.tag.admission_generation &&
            left.backing.backing_offset == right.backing.backing_offset;
        if (!ordered_overlay)
          throw util::Exception(
              std::string_view("group maintenance has overlapping canonical backing lines"));
      }
    }

    acquire_all();
    const Snapshot after = capture();
    if (!same_snapshot(before, after))
      throw CacheMaintenanceUnavailable("dirty cache state changed during preparation");
    if (scoped) {
      // A private line created without VM admission metadata could name the
      // selected address but cannot be proved to belong to its grant. Refuse
      // before any prepared write rather than silently leave stale bytes.
      for (const DomainLocks &entry : locked) {
        for (L1ScalarCache *cache : entry.domain.coherence->l1_scalar_caches_)
          if (cache->has_unidentified_overlap(aliases))
            throw CacheMaintenanceUnavailable("scoped maintenance found an unidentified L1 line");
        for (L1VectorCache *cache : entry.domain.coherence->l1_vector_caches_)
          if (cache->has_unidentified_overlap(aliases))
            throw CacheMaintenanceUnavailable("scoped maintenance found an unidentified L1 line");
      }
    }

    // Cache maintenance locks exclude ordinary mutations. Take all mask
    // mutexes in pointer order as well, then finish every lookup and backing
    // subview before entering a gate or copying the first byte.
    std::vector<L2Cache *> dirty_l2_caches;
    dirty_l2_caches.reserve(prepared_line_count);
    for (const PreparedLine &line : prepared_lines)
      if (before.dirty[line.snapshot_index].selected)
        if (L2Cache *const cache = before.dirty[line.snapshot_index].l2)
          dirty_l2_caches.push_back(cache);
    std::ranges::sort(dirty_l2_caches, std::less<L2Cache *>{});
    dirty_l2_caches.erase(std::ranges::unique(dirty_l2_caches).begin(), dirty_l2_caches.end());
    std::vector<std::unique_lock<std::mutex>> dirty_mask_locks;
    dirty_mask_locks.reserve(dirty_l2_caches.size());
    for (L2Cache *cache : dirty_l2_caches)
      dirty_mask_locks.emplace_back(cache->dirty_bytes_mutex_);

    std::vector<CommitLine> commit_lines;
    commit_lines.reserve(prepared_line_count);
    for (const PreparedLine &line : prepared_lines) {
      const DirtyLine &snapshot = before.dirty[line.snapshot_index];
      if (!snapshot.selected) {
        commit_lines.push_back({nullptr, nullptr, nullptr});
        continue;
      }
      simdojo::CacheTag *tag = nullptr;
      const uint8_t *source = nullptr;
      if (snapshot.l2 != nullptr) {
        L2Cache *const cache = snapshot.l2;
        if (!cache->cache_.lookup(snapshot.address, &tag, snapshot.tag.vmid))
          throw util::Exception(std::string_view("dirty L2 writer disappeared"));
        source = cache->cache_.line_data_for_read(snapshot.address, snapshot.tag.vmid);
        const auto mask = cache->dirty_bytes_.find({snapshot.tag.vmid, snapshot.address});
        if (mask == cache->dirty_bytes_.end() || mask->second != snapshot.dirty_mask)
          throw util::Exception(std::string_view("dirty L2 mask changed during preparation"));
      } else {
        MemorySideCache *const cache = snapshot.memory_side;
        if (!cache->cache_.lookup(snapshot.address, &tag, snapshot.tag.vmid))
          throw util::Exception(std::string_view("dirty memory-side writer disappeared"));
        source = cache->cache_.line_data_for_read(snapshot.address, snapshot.tag.vmid);
      }
      if (source == nullptr || tag == nullptr ||
          !std::equal(source, source + L2Cache::LINE_SIZE, snapshot.bytes.begin()))
        throw util::Exception(std::string_view("dirty cache writer changed during preparation"));
      commit_lines.push_back({&snapshot, tag, source});
    }

    std::vector<CommitLine> clean_lines;
    clean_lines.reserve(before.selected_clean.size());
    for (const DirtyLine &snapshot : before.selected_clean) {
      simdojo::CacheTag *tag = nullptr;
      const uint8_t *source = nullptr;
      if (snapshot.l2 != nullptr) {
        if (!snapshot.l2->cache_.lookup(snapshot.address, &tag, snapshot.tag.vmid))
          throw util::Exception(std::string_view("selected clean L2 line disappeared"));
        source = snapshot.l2->cache_.line_data_for_read(snapshot.address, snapshot.tag.vmid);
      } else {
        if (!snapshot.memory_side->cache_.lookup(snapshot.address, &tag, snapshot.tag.vmid))
          throw util::Exception(std::string_view("selected clean memory-side line disappeared"));
        source =
            snapshot.memory_side->cache_.line_data_for_read(snapshot.address, snapshot.tag.vmid);
      }
      if (source == nullptr || tag == nullptr ||
          !std::equal(source, source + L2Cache::LINE_SIZE, snapshot.bytes.begin()))
        throw util::Exception(std::string_view("selected clean cache line changed"));
      clean_lines.push_back({&snapshot, tag, source});
    }

    std::vector<WriteOp> writes;
    writes.reserve(prepared_write_count);
    const auto add_write = [&writes](const CommitLine &line,
                                     const PreparedCanonicalWriteback::ResolvedLine &backing,
                                     uint32_t begin, uint32_t size) {
      const uint64_t address = line.snapshot->address + begin;
      std::byte *destination = nullptr;
      FabricBacking fallible;
      if (!backing.direct_bytes.empty()) {
        destination = backing.direct_bytes.data() + begin;
      } else {
        fallible = backing.backing.subview(backing.backing_offset + begin, size);
        if (!fallible)
          throw util::Exception(std::string_view("prepared cache writeback run is unavailable"));
      }
      writes.push_back({line.source + begin, destination, std::move(fallible), size,
                        line.snapshot->l2, address});
    };
    // The MSC image can be older than the L2 dirty bytes. Apply all whole MSC
    // lines first, then overlay every L2 run, regardless of domain order.
    for (bool memory_side_pass : {true, false}) {
      for (size_t index = 0; index < prepared_lines.size(); ++index) {
        const PreparedLine &line = prepared_lines[index];
        const CommitLine &commit_line = commit_lines[index];
        if (commit_line.snapshot == nullptr)
          continue;
        if ((commit_line.snapshot->memory_side != nullptr) != memory_side_pass)
          continue;
        if (memory_side_pass) {
          add_write(commit_line, line.backing, 0, MemorySideCache::LINE_SIZE);
          continue;
        }
        for (uint32_t byte = 0; byte < L2Cache::LINE_SIZE;) {
          while (byte < L2Cache::LINE_SIZE &&
                 (commit_line.snapshot->dirty_mask[byte / 64] & (uint64_t{1} << (byte % 64))) == 0)
            ++byte;
          const uint32_t begin = byte;
          while (byte < L2Cache::LINE_SIZE &&
                 (commit_line.snapshot->dirty_mask[byte / 64] & (uint64_t{1} << (byte % 64))) != 0)
            ++byte;
          if (begin != byte)
            add_write(commit_line, line.backing, begin, byte - begin);
        }
      }
    }
    if (writes.size() != prepared_write_count)
      throw util::Exception(std::string_view("dirty cache write plan changed"));

    std::vector<std::shared_ptr<VmTransferEffectGate>> effect_gates;
    effect_gates.reserve(prepared->size());
    for (const PreparedCanonicalWriteback &entry : *prepared)
      effect_gates.push_back(entry.effect_gate_);
    std::ranges::sort(effect_gates, std::less<std::shared_ptr<VmTransferEffectGate>>{});
    effect_gates.erase(std::ranges::unique(effect_gates).begin(), effect_gates.end());
    std::vector<VmTransferEffectGate::Permit> gate_permits;
    gate_permits.reserve(effect_gates.size());
    for (const std::shared_ptr<VmTransferEffectGate> &gate : effect_gates) {
      std::optional<VmTransferEffectGate::Permit> permit = gate->try_acquire();
      if (!permit)
        throw util::Exception(std::string_view("prepared cache writeback VM gate was revoked"));
      gate_permits.push_back(std::move(*permit));
    }

    // A batch with several effects uses only retained direct KFD byte views.
    // The sole generic-owner path remains one fallible write with its existing
    // no-effect-on-failure contract. Nothing below takes a lock or allocates.
    for (const WriteOp &write : writes) {
      if (write.direct_destination != nullptr) {
        std::memcpy(write.direct_destination, write.source, write.size);
      } else if (!write.fallible_destination.write(
                     std::as_bytes(std::span<const uint8_t>(write.source, write.size)))) {
        throw util::Exception(std::string_view("prepared cache writeback failed"));
      }
    }
    for (const WriteOp &write : writes)
      if (write.l2 != nullptr)
        write.l2->diagnostics(write.address).backing_writes.fetch_add(1, std::memory_order_relaxed);
    for (const CommitLine &line : commit_lines) {
      if (line.snapshot == nullptr)
        continue;
      if (line.snapshot->l2 != nullptr) {
        L2Cache *const cache = line.snapshot->l2;
        cache->dirty_bytes_.erase({line.snapshot->tag.vmid, line.snapshot->address});
        if (!scoped)
          cache->has_dirty_lines_.store(!cache->dirty_bytes_.empty(), std::memory_order_relaxed);
      } else if (!scoped) {
        line.snapshot->memory_side->may_have_dirty_lines_.store(false, std::memory_order_relaxed);
      }
      if (scoped) {
        if (line.snapshot->l2 != nullptr)
          line.snapshot->l2->cache_.invalidate(line.snapshot->address, line.snapshot->tag.vmid,
                                               line.snapshot->tag.address_space_identity);
        else
          line.snapshot->memory_side->cache_.invalidate(line.snapshot->address,
                                                        line.snapshot->tag.vmid,
                                                        line.snapshot->tag.address_space_identity);
      } else {
        line.tag->dirty = false;
        line.tag->coherence = simdojo::CoherenceState::SHARED;
      }
    }
    for (const CommitLine &line : clean_lines) {
      if (line.snapshot->l2 != nullptr)
        line.snapshot->l2->cache_.invalidate(line.snapshot->address, line.snapshot->tag.vmid,
                                             line.snapshot->tag.address_space_identity);
      else
        line.snapshot->memory_side->cache_.invalidate(line.snapshot->address,
                                                      line.snapshot->tag.vmid,
                                                      line.snapshot->tag.address_space_identity);
    }
    if (scoped) {
      for (const DomainLocks &entry : locked) {
        for (L1ScalarCache *cache : entry.domain.coherence->l1_scalar_caches_)
          cache->invalidate_selected_lines(aliases);
        for (L1VectorCache *cache : entry.domain.coherence->l1_vector_caches_)
          cache->invalidate_selected_lines(aliases);
      }
    }
  } catch (...) {
    release_unpublished();
    throw;
  }

  for (DomainLocks &entry : locked) {
    DeviceCacheMaintenanceLease lease(entry.domain.coherence.get(), entry.locked_l2_count,
                                      entry.locked_memory_side_count, std::move(entry.atomic_lock),
                                      std::move(entry.coherence_lock), !scoped,
                                      entry.locked_scalar_count, entry.locked_vector_count);
    lease.owner_handle_ = entry.domain.coherence;
    lease.prepared_writebacks_ = prepared;
    lease.read_witnesses_ = witnesses;
    leases.push_back(std::move(lease));
  }
  return leases;
}

void DeviceCacheCoherence::complete_cache_maintenance(size_t locked_l2_count,
                                                      size_t locked_memory_side_count,
                                                      bool publish_data_epoch) noexcept {
  // Publish invalidation only after the protected backing-memory effect. Data
  // caches remain excluded until the epochs are visible. Private instruction
  // caches are not touched cross-thread and reconcile this epoch on their next
  // owner-thread fetch.
  if (publish_data_epoch) {
    [[maybe_unused]] const uint64_t next_data_epoch =
        data_epoch_.fetch_add(1, std::memory_order_release) + 1;
    assert(next_data_epoch != 0 && "device data-cache coherence epoch wrapped");
  }
  // A scoped maintenance does not advance the device-wide data epoch. Its
  // canonical aliases may still overlap clean, handle-bound local VRAM tags;
  // those have no dirty bytes, so conservatively retire them after the effect.
  if (translated_local_cache_used_.load(std::memory_order_acquire)) {
    [[maybe_unused]] const uint64_t next_local_epoch =
        local_backing_epoch_.fetch_add(1, std::memory_order_release) + 1;
    assert(next_local_epoch != 0 && "translated local backing epoch wrapped");
  }
  [[maybe_unused]] const uint64_t next_instruction_epoch =
      instruction_epoch_.fetch_add(1, std::memory_order_release) + 1;
  assert(next_instruction_epoch != 0 && "device instruction-cache coherence epoch wrapped");
  release_cache_locks(locked_l2_count, locked_memory_side_count);
}

void DeviceCacheCoherence::release_cache_locks(size_t locked_l2_count,
                                               size_t locked_memory_side_count,
                                               size_t locked_scalar_count,
                                               size_t locked_vector_count) noexcept {
  while (locked_memory_side_count != 0) {
    --locked_memory_side_count;
    memory_side_caches_[locked_memory_side_count]->access_gate_.unlock();
  }
  while (locked_l2_count != 0) {
    --locked_l2_count;
    l2_caches_[locked_l2_count]->maintenance_mutex_.unlock();
  }
  while (locked_vector_count != 0) {
    --locked_vector_count;
    l1_vector_caches_[locked_vector_count]->maintenance_mutex_.unlock();
  }
  while (locked_scalar_count != 0) {
    --locked_scalar_count;
    l1_scalar_caches_[locked_scalar_count]->maintenance_mutex_.unlock();
  }
}

void DeviceCacheCoherence::register_l2_cache(L2Cache *cache) {
  std::unique_lock lock(mutex_);
  assert(cache != nullptr);
  const std::vector<L2Cache *>::iterator position =
      std::ranges::lower_bound(l2_caches_, cache, std::less<L2Cache *>{});
  assert(position == l2_caches_.end() || *position != cache);
  if (position == l2_caches_.end() || *position != cache) {
    l2_caches_.insert(position, cache);
    ++cache_registry_generation_;
  }
}

void DeviceCacheCoherence::register_l1_scalar_cache(L1ScalarCache *cache) {
  std::unique_lock lock(mutex_);
  assert(cache != nullptr);
  const auto position =
      std::ranges::lower_bound(l1_scalar_caches_, cache, std::less<L1ScalarCache *>{});
  assert(position == l1_scalar_caches_.end() || *position != cache);
  if (position == l1_scalar_caches_.end() || *position != cache) {
    l1_scalar_caches_.insert(position, cache);
    ++cache_registry_generation_;
  }
}

void DeviceCacheCoherence::unregister_l1_scalar_cache(L1ScalarCache *cache) {
  std::unique_lock lock(mutex_);
  unregister_cache(l1_scalar_caches_, cache);
  ++cache_registry_generation_;
}

void DeviceCacheCoherence::register_l1_vector_cache(L1VectorCache *cache) {
  std::unique_lock lock(mutex_);
  assert(cache != nullptr);
  const auto position =
      std::ranges::lower_bound(l1_vector_caches_, cache, std::less<L1VectorCache *>{});
  assert(position == l1_vector_caches_.end() || *position != cache);
  if (position == l1_vector_caches_.end() || *position != cache) {
    l1_vector_caches_.insert(position, cache);
    ++cache_registry_generation_;
  }
}

void DeviceCacheCoherence::unregister_l1_vector_cache(L1VectorCache *cache) {
  std::unique_lock lock(mutex_);
  unregister_cache(l1_vector_caches_, cache);
  ++cache_registry_generation_;
}

void DeviceCacheCoherence::unregister_l2_cache(L2Cache *cache) {
  std::unique_lock lock(mutex_);
  unregister_cache(l2_caches_, cache);
  ++cache_registry_generation_;
}

void DeviceCacheCoherence::register_memory_side_cache(MemorySideCache *cache) {
  std::unique_lock lock(mutex_);
  assert(cache != nullptr);
  const std::vector<MemorySideCache *>::iterator position =
      std::ranges::lower_bound(memory_side_caches_, cache, std::less<MemorySideCache *>{});
  assert(position == memory_side_caches_.end() || *position != cache);
  if (position == memory_side_caches_.end() || *position != cache) {
    memory_side_caches_.insert(position, cache);
    ++cache_registry_generation_;
  }
}

void DeviceCacheCoherence::unregister_memory_side_cache(MemorySideCache *cache) {
  std::unique_lock lock(mutex_);
  unregister_cache(memory_side_caches_, cache);
  ++cache_registry_generation_;
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
