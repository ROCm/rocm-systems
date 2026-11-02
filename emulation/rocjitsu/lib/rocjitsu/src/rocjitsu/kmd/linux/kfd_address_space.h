// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/vm/amdgpu/legacy_page_table.h"
#include "rocjitsu/vm/amdgpu/mtype.h"
#include "rocjitsu/vm/amdgpu/vm_mutation_epoch.h"
#include "util/distributed_shared_mutex.h"
#include "util/intrusive_shared_ptr.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <memory_resource>
#include <mutex>
#include <shared_mutex>
#include <span>
#include <utility>
#include <vector>

namespace rocjitsu {

/// Translation state shared by a DRM file and any KFD binding that acquires it.
/// Registration snapshots retain this state independently of descriptor lifetime.
class KfdAddressSpace {
public:
  static constexpr uint64_t kPageShift = amdgpu::kLegacyPageShift;
  static constexpr uint64_t kPageSize = amdgpu::kLegacyPageSize;
  using HostExtentOwner = amdgpu::LegacyHostExtentOwner;
  using HostExtent = amdgpu::LegacyHostExtent;
  using PageTableEntry = amdgpu::LegacyPageTableEntry;
  using PageTable = amdgpu::LegacyPageTable;

  struct MappingRange {
    uint64_t gpu_va;
    size_t size;
  };

  /// @brief Import existing mapped extents, retaining their backing and page policy.
  /// @details Overlap with destination extents, incompatible policies within one page,
  /// and wrapping ranges fail without changing either table. Allocation failures
  /// propagate with the destination unchanged. Unmapped source bytes stay unmapped.
  bool import_mappings(const KfdAddressSpace &source, std::span<const MappingRange> ranges) {
    if (&source == this)
      return true;

    std::vector<MappingRange> selected;
    selected.reserve(ranges.size());
    for (const auto &range : ranges) {
      if (range.size > std::numeric_limits<uint64_t>::max() - range.gpu_va)
        return false;
      if (range.size != 0)
        selected.push_back(range);
    }
    if (selected.empty())
      return true;
    std::ranges::sort(selected, {}, &MappingRange::gpu_va);
    size_t count = 0;
    for (const auto &range : selected) {
      if (count != 0 && range.gpu_va <= selected[count - 1].gpu_va + selected[count - 1].size) {
        auto &previous = selected[count - 1];
        previous.size = std::max(previous.size, range.gpu_va - previous.gpu_va + range.size);
      } else {
        selected[count++] = range;
      }
    }
    selected.resize(count);

    // Acquire both request leases before either table lock, as map/unmap do.
    // std::scoped_lock also orders concurrent imports in opposite directions.
    std::scoped_lock requests(*page_table_request_mutex_, *source.page_table_request_mutex_);
    std::unique_lock target_lock(page_table_mutex_);
    std::shared_lock source_lock(source.page_table_mutex_);
    PageTable staged(page_table_, page_table_.get_allocator());
    bool changed = false;
    for (const auto &[page_index, source_page] : source.page_table_) {
      const uint64_t page_base = page_index << kPageShift;
      const uint64_t page_last = page_base | (kPageSize - 1);
      auto range = std::lower_bound(selected.begin(), selected.end(), page_base,
                                    [](const MappingRange &entry, uint64_t address) {
                                      return entry.gpu_va + entry.size <= address;
                                    });
      for (; range != selected.end() && range->gpu_va <= page_last; ++range) {
        const size_t begin = range->gpu_va > page_base ? range->gpu_va - page_base : 0;
        const size_t end = std::min<uint64_t>(kPageSize, range->gpu_va + range->size - page_base);
        for (const auto &extent : source_page.host_extents) {
          const size_t clipped_begin = std::max(begin, extent.gpu_page_offset);
          const size_t clipped_end =
              std::min(end, extent.gpu_page_offset + extent.host_backed_bytes);
          if (!extent.host_ptr || clipped_begin >= clipped_end)
            continue;
          if (const auto existing = page_table_.find(page_index); existing != page_table_.end()) {
            if (!existing->second.host_extents.empty() &&
                existing->second.mtype != source_page.mtype)
              return false;
            for (const auto &occupied : existing->second.host_extents)
              if (clipped_begin < occupied.gpu_page_offset + occupied.host_backed_bytes &&
                  occupied.gpu_page_offset < clipped_end)
                return false;
          }
          auto &page = staged[page_index];
          page.mtype = source_page.mtype;
          replace_host_extent(page, {extent.host_ptr + (clipped_begin - extent.gpu_page_offset),
                                     clipped_end - clipped_begin, clipped_begin, extent.owner});
          changed = true;
        }
      }
    }
    if (changed) {
      invalidate_page_policies_locked();
      page_table_.swap(staged);
      publish_page_table_mutation_locked();
    }
    return true;
  }

  /// @brief Map host pages into this address space's GPU page table.
  /// @param mtype PTE MTYPE for these pages (derived from allocation flags).
  /// @param owner Who may revoke the backing; see HostExtentOwner. Defaults to
  ///        Application so an unannotated caller is validated rather than
  ///        trusted.
  void map_pages(uint64_t gpu_va, void *host_ptr, size_t size,
                 amdgpu::Mtype mtype = amdgpu::Mtype::RW,
                 HostExtentOwner owner = HostExtentOwner::Application) {
    std::unique_lock request_lock(*page_table_request_mutex_);
    std::unique_lock lock(page_table_mutex_);
    invalidate_page_policies_locked();
    auto *base = static_cast<uint8_t *>(host_ptr);
    uint64_t mapped_va = gpu_va;
    size_t host_offset = 0;
    while (host_offset < size) {
      const size_t gpu_page_offset = mapped_va & (kPageSize - 1);
      const size_t host_backed_bytes =
          std::min<size_t>(kPageSize - gpu_page_offset, size - host_offset);
      auto [page, inserted] =
          page_table_.try_emplace(mapped_va >> kPageShift, base + host_offset, mtype,
                                  host_backed_bytes, gpu_page_offset, owner);
      if (!inserted) {
        page->second.mtype = mtype;
        replace_host_extent(page->second,
                            {base + host_offset, host_backed_bytes, gpu_page_offset, owner});
      }
      mapped_va += host_backed_bytes;
      host_offset += host_backed_bytes;
    }
    // Keep publication in the page-table critical section. Cached readers
    // validate this generation while holding the shared side of the same lock;
    // publishing after unlock would permit a stale-cache hit in between.
    publish_page_table_mutation_locked();
  }

  /// @brief Unmap pages from this address space's GPU page table.
  void unmap_pages(uint64_t gpu_va, size_t size) {
    std::unique_lock request_lock(*page_table_request_mutex_);
    std::unique_lock lock(page_table_mutex_);
    invalidate_page_policies_locked();
    uint64_t mapped_va = gpu_va;
    size_t unmapped_bytes = 0;
    while (unmapped_bytes < size) {
      const size_t chunk =
          std::min<size_t>(kPageSize - (mapped_va & (kPageSize - 1)), size - unmapped_bytes);
      auto page = page_table_.find(mapped_va >> kPageShift);
      if (page != page_table_.end()) {
        // Avoid building a temporary extent vector for a page being discarded.
        if (chunk != kPageSize)
          erase_host_extent(page->second, mapped_va & (kPageSize - 1), chunk);
        if (chunk == kPageSize || page->second.host_extents.empty())
          page_table_.erase(page);
      }
      mapped_va += chunk;
      unmapped_bytes += chunk;
    }
    // See map_pages(): the mutation and generation publication are one
    // page-table critical section by design.
    publish_page_table_mutation_locked();
  }

  /// @brief Replace mapped host pages while preserving their other PTE fields.
  /// @details The mutation and cache-generation publication occur under one
  /// page-table critical section. Only entries still pointing at the expected
  /// old page are changed.
  void remap_page_host_ptrs(uint64_t gpu_va, void *old_host_ptr, void *new_host_ptr, size_t size) {
    std::unique_lock request_lock(*page_table_request_mutex_);
    std::unique_lock lock(page_table_mutex_);
    invalidate_page_policies_locked();
    auto *old_base = static_cast<uint8_t *>(old_host_ptr);
    auto *new_base = static_cast<uint8_t *>(new_host_ptr);
    bool changed = false;
    uint64_t mapped_va = gpu_va;
    size_t host_offset = 0;
    while (host_offset < size) {
      auto page = page_table_.find(mapped_va >> kPageShift);
      if (page != page_table_.end()) {
        const uint64_t page_base = mapped_va & ~(kPageSize - 1);
        for (auto &extent : page->second.host_extents) {
          const uint64_t extent_va = page_base + extent.gpu_page_offset;
          if (extent_va < gpu_va || extent_va - gpu_va >= size)
            continue;
          const size_t extent_host_offset = extent_va - gpu_va;
          if (extent.host_ptr == old_base + extent_host_offset &&
              extent.host_ptr != new_base + extent_host_offset) {
            extent.host_ptr = new_base + extent_host_offset;
            changed = true;
          }
        }
      }
      const size_t chunk =
          std::min<size_t>(kPageSize - (mapped_va & (kPageSize - 1)), size - host_offset);
      mapped_va += chunk;
      host_offset += chunk;
    }
    if (changed)
      publish_page_table_mutation_locked();
  }

  /// @brief Update the MTYPE of mapped pages and invalidate cached PTE copies.
  void set_page_mtype(uint64_t gpu_va, size_t size, amdgpu::Mtype mtype) {
    std::unique_lock request_lock(*page_table_request_mutex_);
    std::unique_lock lock(page_table_mutex_);
    invalidate_page_policies_locked();
    bool changed = false;
    uint64_t mapped_va = gpu_va;
    size_t updated_bytes = 0;
    while (updated_bytes < size) {
      auto it = page_table_.find(mapped_va >> kPageShift);
      if (it != page_table_.end() && it->second.mtype != mtype) {
        it->second.mtype = mtype;
        changed = true;
      }
      const size_t chunk =
          std::min<size_t>(kPageSize - (mapped_va & (kPageSize - 1)), size - updated_bytes);
      mapped_va += chunk;
      updated_bytes += chunk;
    }
    if (changed)
      publish_page_table_mutation_locked();
  }

  /// @brief Return the mutation counter used by GpuMemory translation caches.
  const uint64_t *page_table_generation() const { return &page_table_generation_; }

  /// @brief Return the retained token invalidated before each page-policy mutation.
  util::IntrusiveSharedPtr<const amdgpu::VmMutationEpoch> page_table_mutation_epoch() const {
    return page_table_mutation_epoch_;
  }

  /// @brief Retained admission state for immutable copied PTE snapshots.
  std::shared_ptr<amdgpu::LegacyPageTableCacheState> page_table_cache_state() const {
    return page_table_cache_state_;
  }

  /// @brief Return the lease shared by page-table readers and mutations.
  std::shared_ptr<util::DistributedSharedMutex> page_table_request_mutex() const {
    return page_table_request_mutex_;
  }

  mutable util::DistributedSharedMutex page_table_mutex_;
  /// @brief Pool for page-table nodes and buckets.
  /// @details Writers serialize allocation under page_table_mutex_. Declaration
  /// order keeps the pool alive until the table is destroyed; capacity is
  /// retained for reuse until this address space is destroyed.
  std::pmr::unsynchronized_pool_resource page_table_pool_;
  PageTable page_table_{&page_table_pool_};

private:
  static void normalize_host_extents(PageTableEntry &page) {
    auto &extents = page.host_extents;
    if (extents.size() > 1)
      std::ranges::sort(extents, {}, &HostExtent::gpu_page_offset);
    size_t out = 0;
    for (const auto &extent : extents) {
      if (extent.host_ptr == nullptr || extent.host_backed_bytes == 0)
        continue;
      if (out > 0) {
        auto &previous = extents[out - 1];
        if (previous.owner == extent.owner &&
            previous.gpu_page_offset + previous.host_backed_bytes == extent.gpu_page_offset &&
            previous.host_ptr + previous.host_backed_bytes == extent.host_ptr) {
          previous.host_backed_bytes += extent.host_backed_bytes;
          continue;
        }
      }
      extents[out++] = extent;
    }
    extents.resize(out);
  }

  static void replace_host_extent(PageTableEntry &page, HostExtent replacement) {
    const size_t replacement_begin = replacement.gpu_page_offset;
    const size_t replacement_end = replacement_begin + replacement.host_backed_bytes;
    if (replacement_begin == 0 && replacement_end == kPageSize) {
      page.host_extents = std::vector<HostExtent>{replacement};
      return;
    }
    std::vector<HostExtent> updated;
    updated.reserve(page.host_extents.size() + 1);
    for (const auto &extent : page.host_extents) {
      const size_t extent_begin = extent.gpu_page_offset;
      const size_t extent_end = extent_begin + extent.host_backed_bytes;
      if (extent_end <= replacement_begin || replacement_end <= extent_begin) {
        updated.push_back(extent);
        continue;
      }
      if (extent_begin < replacement_begin)
        updated.push_back(
            {extent.host_ptr, replacement_begin - extent_begin, extent_begin, extent.owner});
      if (replacement_end < extent_end)
        updated.push_back({extent.host_ptr + (replacement_end - extent_begin),
                           extent_end - replacement_end, replacement_end, extent.owner});
    }
    updated.push_back(replacement);
    page.host_extents = std::move(updated);
    normalize_host_extents(page);
  }

  static void erase_host_extent(PageTableEntry &page, size_t erased_begin, size_t erased_bytes) {
    const size_t erased_end = erased_begin + erased_bytes;
    std::vector<HostExtent> updated;
    updated.reserve(page.host_extents.size() + 1);
    for (const auto &extent : page.host_extents) {
      const size_t extent_begin = extent.gpu_page_offset;
      const size_t extent_end = extent_begin + extent.host_backed_bytes;
      if (extent_end <= erased_begin || erased_end <= extent_begin) {
        updated.push_back(extent);
        continue;
      }
      if (extent_begin < erased_begin)
        updated.push_back(
            {extent.host_ptr, erased_begin - extent_begin, extent_begin, extent.owner});
      if (erased_end < extent_end)
        updated.push_back({extent.host_ptr + (erased_end - extent_begin), extent_end - erased_end,
                           erased_end, extent.owner});
    }
    page.host_extents = std::move(updated);
    normalize_host_extents(page);
  }

  // Invalidate copied policies before mutation, including a partial update
  // that throws before publishing the ordinary translation generation.
  void invalidate_page_policies_locked() {
    page_table_mutation_epoch_->value.fetch_add(1, std::memory_order_release);
    page_table_cache_state_->invalidate_and_wait();
  }

  void publish_page_table_mutation_locked() { ++page_table_generation_; }

  std::shared_ptr<util::DistributedSharedMutex> page_table_request_mutex_ =
      std::make_shared<util::DistributedSharedMutex>();

  /// @brief Page table version counter, bumped on every PTE mutation.
  /// @details GpuMemory keeps per-thread TLB-like translation caches keyed by
  ///          this generation. Mutations hold both page_table_request_mutex_
  ///          and page_table_mutex_; readers hold at least one of those locks,
  ///          so the counter itself does not need atomics.
  uint64_t page_table_generation_{1};
  /// @brief Retained atomic token for lockless copied-policy checks.
  /// @details Invalidate before mutation, including partially throwing updates.
  util::IntrusiveSharedPtr<amdgpu::VmMutationEpoch> page_table_mutation_epoch_ =
      util::IntrusiveSharedPtr<amdgpu::VmMutationEpoch>::make(1);
  std::shared_ptr<amdgpu::LegacyPageTableCacheState> page_table_cache_state_ =
      std::make_shared<amdgpu::LegacyPageTableCacheState>();
};

} // namespace rocjitsu
