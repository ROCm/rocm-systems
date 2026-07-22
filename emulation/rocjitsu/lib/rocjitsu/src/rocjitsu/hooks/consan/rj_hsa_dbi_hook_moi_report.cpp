// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "hsa/hsa_api_trace_minimal.h"

#include "rocjitsu/code/patch/consan/consan_moi.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_hook_internal.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_replay_provenance.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_sampled_sync.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>

namespace rocjitsu::consan_hook {

class AutoMoiReportBufferRegistry {
public:
  static AutoMoiReportBufferRegistry &instance() {
    static AutoMoiReportBufferRegistry registry;
    return registry;
  }

  void reject_plan(uint64_t reader, uint64_t required_size, uint64_t configured_cap,
                   std::string_view reason) {
    record_allocation_attempt(required_size);
    record_allocation_failure(required_size, /*capacity_failure=*/true);
    log_message(kLogInfo,
                "ConSan MOI auto report allocation reader=%llu outcome="
                "insufficient_report_capacity reason=%.*s required_bytes=%llu cap_bytes=%llu",
                static_cast<unsigned long long>(reader), static_cast<int>(reason.size()),
                reason.data(), static_cast<unsigned long long>(required_size),
                static_cast<unsigned long long>(configured_cap));
  }

  [[nodiscard]] bool
  allocate(CoreApiTable *core, hsa_agent_t agent, uint64_t reader, uint64_t required_size,
           uint64_t requested_size, uint64_t configured_cap,
           const rocjitsu::ConSanMoiReportBufferLayout &layout, rocjitsu::ConSanMoiEngine engine,
           bool track_barriers, bool track_atomics, bool test_seed_inline_exact_odd,
           uint64_t *address, uint64_t *registered_size, uint64_t *registered_generation) {
    record_allocation_attempt(required_size);
    const bool direct_sampled = engine == rocjitsu::ConSanMoiEngine::Sampled;
    const bool inline_shadow = engine == rocjitsu::ConSanMoiEngine::InlineShadow;
    if (required_size > configured_cap || requested_size > configured_cap ||
        requested_size > rocjitsu::kConSanMoiAutoReportBufferCeilingBytes) {
      record_allocation_failure(required_size, /*capacity_failure=*/true);
      log_message(
          kLogInfo,
          "ConSan MOI auto report allocation reader=%llu outcome="
          "insufficient_report_capacity required_bytes=%llu cap_bytes=%llu "
          "per_buffer_ceiling=%llu",
          static_cast<unsigned long long>(reader), static_cast<unsigned long long>(required_size),
          static_cast<unsigned long long>(configured_cap),
          static_cast<unsigned long long>(rocjitsu::kConSanMoiAutoReportBufferCeilingBytes));
      return false;
    }
    if (core == nullptr || core->hsa_agent_iterate_regions_fn == nullptr ||
        core->hsa_region_get_info_fn == nullptr || core->hsa_memory_allocate_fn == nullptr ||
        core->hsa_memory_free_fn == nullptr) {
      record_allocation_failure(required_size, /*capacity_failure=*/false);
      log_message(
          kLogInfo,
          "ConSan MOI auto report buffer requested but HSA allocation APIs are unavailable");
      return false;
    }
    if (requested_size > std::numeric_limits<size_t>::max()) {
      record_allocation_failure(required_size, /*capacity_failure=*/true);
      log_message(kLogInfo, "ConSan MOI auto report buffer size is too large: %llu",
                  static_cast<unsigned long long>(requested_size));
      return false;
    }

    const size_t requested = static_cast<size_t>(requested_size);
    if (requested < sizeof(rocjitsu::ConSanMoiReportHeader) || !layout.valid ||
        layout.required_bytes > requested) {
      record_allocation_failure(required_size, /*capacity_failure=*/true);
      log_message(kLogInfo,
                  "ConSan MOI auto report buffer is too small reader=%llu bytes=%zu "
                  "direct_sampled=%s inline_shadow=%s track_barriers=%s track_atomics=%s",
                  static_cast<unsigned long long>(reader), requested,
                  direct_sampled ? "true" : "false", inline_shadow ? "true" : "false",
                  track_barriers ? "true" : "false", track_atomics ? "true" : "false");
      return false;
    }

    if (!reserve_live_bytes(required_size, requested_size)) {
      log_message(
          kLogInfo,
          "ConSan MOI auto report allocation reader=%llu outcome="
          "insufficient_report_capacity required_bytes=%llu requested_bytes=%llu "
          "process_ceiling=%llu",
          static_cast<unsigned long long>(reader), static_cast<unsigned long long>(required_size),
          static_cast<unsigned long long>(requested_size),
          static_cast<unsigned long long>(rocjitsu::kConSanMoiAutoReportProcessCeilingBytes));
      return false;
    }
    const auto release_reservation = [&] { release_live_bytes(requested_size); };

    RegionSearch search;
    search.core = core;
    search.requested_size = requested;
    const hsa_status_t iterate_status =
        core->hsa_agent_iterate_regions_fn(agent, select_region, &search);
    if (iterate_status != HSA_STATUS_SUCCESS && iterate_status != HSA_STATUS_INFO_BREAK) {
      release_reservation();
      record_allocation_failure(required_size, /*capacity_failure=*/false);
      log_message(kLogInfo,
                  "ConSan MOI auto report buffer region iteration failed reader=%llu status=%d",
                  static_cast<unsigned long long>(reader), static_cast<int>(iterate_status));
      return false;
    }
    if (!search.found) {
      release_reservation();
      record_allocation_failure(required_size, /*capacity_failure=*/false);
      log_message(kLogInfo,
                  "ConSan MOI auto report buffer found no allocatable global HSA region "
                  "reader=%llu bytes=%zu",
                  static_cast<unsigned long long>(reader), requested);
      return false;
    }

    void *ptr = nullptr;
    const hsa_status_t status = core->hsa_memory_allocate_fn(search.region, requested, &ptr);
    if (status != HSA_STATUS_SUCCESS) {
      release_reservation();
      record_allocation_failure(required_size, /*capacity_failure=*/false);
      log_message(kLogInfo,
                  "ConSan MOI auto report buffer hsa_memory_allocate failed reader=%llu "
                  "status=%d bytes=%zu",
                  static_cast<unsigned long long>(reader), static_cast<int>(status), requested);
      return false;
    }
    std::memset(ptr, 0, requested);

    if (core->hsa_memory_assign_agent_fn != nullptr) {
      const hsa_status_t assign_status =
          core->hsa_memory_assign_agent_fn(ptr, agent, HSA_ACCESS_PERMISSION_RW);
      if (assign_status != HSA_STATUS_SUCCESS) {
        log_message(kLogInfo,
                    "ConSan MOI auto report buffer hsa_memory_assign_agent failed reader=%llu "
                    "status=%d",
                    static_cast<unsigned long long>(reader), static_cast<int>(assign_status));
        (void)core->hsa_memory_free_fn(ptr);
        release_reservation();
        record_allocation_failure(required_size, /*capacity_failure=*/false);
        return false;
      }
    }

    const uint64_t generation = next_generation_.fetch_add(1, std::memory_order_relaxed) + 1u;
    auto *header = static_cast<rocjitsu::ConSanMoiReportHeader *>(ptr);
    *header = rocjitsu::make_consan_moi_report_header_for_layout(generation, /*dispatch_id=*/reader,
                                                                 layout, engine);
    if (test_seed_inline_exact_odd && layout.exact_shadow_entry_capacity != 0) {
      auto *slot = reinterpret_cast<rocjitsu::ConSanMoiInlineExactShadowSlot *>(
          static_cast<uint8_t *>(ptr) + layout.exact_shadow_entries_offset);
      slot[0].version = 1u;
      log_message(kLogInfo,
                  "ConSan MOI test seeded reader=%llu exact_slot=0 version=1 state=publishing",
                  static_cast<unsigned long long>(reader));
    }

    {
      std::lock_guard lock(mutex_);
      if (entry_count_ >= entries_.size()) {
        log_message(kLogInfo, "ConSan MOI auto report buffer registry is full");
        (void)core->hsa_memory_free_fn(ptr);
        if (reserved_entry_count_ != 0)
          --reserved_entry_count_;
        (void)rocjitsu::release_consan_moi_auto_report_bytes(process_budget_, requested_size);
        ++allocation_failure_count_;
        return false;
      }
      if (reserved_entry_count_ != 0)
        --reserved_entry_count_;
      successful_allocated_bytes_ += requested_size;
      entries_[entry_count_++] = Entry{reader,
                                       ptr,
                                       requested,
                                       static_cast<size_t>(required_size),
                                       generation,
                                       layout,
                                       layout.access_record_capacity,
                                       layout.barrier_record_capacity,
                                       layout.atomic_record_capacity,
                                       layout.fence_record_capacity,
                                       layout.diagnostic_capacity,
                                       layout.exact_shadow_entry_capacity,
                                       layout.inline_atomic_release_capacity,
                                       layout.inline_acquired_epoch_token_capacity,
                                       layout.inline_causal_snapshot_capacity,
                                       layout.sampled_watchpoint_capacity,
                                       direct_sampled,
                                       inline_shadow,
                                       search.fine_grained};
    }
    *address = reinterpret_cast<uint64_t>(ptr);
    *registered_size = requested;
    if (registered_generation != nullptr)
      *registered_generation = generation;
    log_message(
        kLogInfo,
        "ConSan MOI auto report buffer reader=%llu addr=0x%llx bytes=%zu "
        "required_bytes=%llu cap_bytes=%llu process_current_bytes=%llu "
        "process_peak_bytes=%llu process_ceiling_bytes=%llu allocation_outcome=allocated "
        "access_record_capacity=%u barrier_record_capacity=%u atomic_record_capacity=%u "
        "fence_record_capacity=%u "
        "diagnostic_capacity=%u exact_shadow_entry_capacity=%u "
        "inline_atomic_release_capacity=%u "
        "inline_acquired_epoch_token_capacity=%u "
        "inline_causal_snapshot_capacity=%u "
        "sampled_watchpoint_capacity=%u sampled_causal_window_capacity=%u "
        "sampled_sync_metadata_capacity=%u sampled_pending_acquire_capacity=%u "
        "generation=%llu fine_grained=%s",
        static_cast<unsigned long long>(reader), static_cast<unsigned long long>(*address),
        requested, static_cast<unsigned long long>(required_size),
        static_cast<unsigned long long>(configured_cap),
        static_cast<unsigned long long>(current_live_bytes()),
        static_cast<unsigned long long>(peak_live_bytes()),
        static_cast<unsigned long long>(rocjitsu::kConSanMoiAutoReportProcessCeilingBytes),
        layout.access_record_capacity, layout.barrier_record_capacity,
        layout.atomic_record_capacity, layout.fence_record_capacity, layout.diagnostic_capacity,
        layout.exact_shadow_entry_capacity, layout.inline_atomic_release_capacity,
        layout.inline_acquired_epoch_token_capacity, layout.inline_causal_snapshot_capacity,
        layout.sampled_watchpoint_capacity, layout.sampled_causal_window_capacity,
        layout.sampled_sync_metadata_capacity, layout.sampled_pending_acquire_capacity,
        static_cast<unsigned long long>(generation), search.fine_grained ? "true" : "false");
    return true;
  }

  using Summary = AutoMoiReportSummary;

  void register_compact_tokens(uint64_t reader, const rocjitsu::ConSanResult &result) {
    std::lock_guard lock(mutex_);
    auto entry = std::find_if(entries_.begin(), entries_.begin() + entry_count_,
                              [reader](const Entry &item) { return item.reader == reader; });
    if (entry == entries_.begin() + entry_count_)
      return;

    auto *mappings = reinterpret_cast<rocjitsu::ConSanMoiCompactDiagnosticTokenMapping *>(
        static_cast<uint8_t *>(entry->ptr) + entry->layout.inline_compact_token_mappings_offset);
    size_t compact_patch_count = 0;
    for (const rocjitsu::ConSanPatchInfo &patch : result.patches) {
      if (!patch.workgroup_shadow_compact || patch.workgroup_shadow_compact_token == 0u)
        continue;
      ++compact_patch_count;
      if (patch.anchor_offset > rocjitsu::consan_moi_exact_shadow::max_instruction_offset ||
          patch.owner_descriptor_file_offsets.size() != 1u ||
          entry->compact_token_mapping_count >=
              entry->layout.inline_compact_token_mapping_capacity) {
        entry->compact_token_mapping_malformed = true;
        continue;
      }
      mappings[entry->compact_token_mapping_count++] =
          rocjitsu::ConSanMoiCompactDiagnosticTokenMapping{
              .owner_descriptor_file_offset = patch.owner_descriptor_file_offsets.front(),
              .instruction_offset = static_cast<uint32_t>(patch.anchor_offset),
              .token = patch.workgroup_shadow_compact_token,
          };
    }
    if (compact_patch_count != 0u) {
      log_message(kLogInfo,
                  "ConSan MOI compact diagnostic map reader=%llu patches=%zu mappings=%u "
                  "capacity=%u malformed=%s",
                  static_cast<unsigned long long>(reader), compact_patch_count,
                  entry->compact_token_mapping_count,
                  entry->layout.inline_compact_token_mapping_capacity,
                  entry->compact_token_mapping_malformed ? "true" : "false");
    }
  }

  Summary summarize_and_clear(CoreApiTable *core) {
    Summary total;
    std::lock_guard lock(mutex_);
    total.required_report_bytes = required_report_bytes_;
    total.allocated_report_bytes = successful_allocated_bytes_;
    total.current_live_report_bytes = process_budget_.current_live_bytes;
    total.peak_live_report_bytes = process_budget_.peak_live_bytes;
    total.allocation_failure_count = allocation_failure_count_;
    total.capacity_failure_count = capacity_failure_count_;
    for (size_t i = 0; i < entry_count_; ++i) {
      const Summary entry_summary = summarize(core, entries_[i]);
      total.buffer_count += entry_summary.buffer_count;
      total.visible_access_record_count += entry_summary.visible_access_record_count;
      total.visible_barrier_record_count += entry_summary.visible_barrier_record_count;
      total.visible_atomic_record_count += entry_summary.visible_atomic_record_count;
      total.visible_fence_record_count += entry_summary.visible_fence_record_count;
      total.visible_diagnostic_record_count += entry_summary.visible_diagnostic_record_count;
      total.visible_inline_publication_count += entry_summary.visible_inline_publication_count;
      total.visible_exact_shadow_entry_count += entry_summary.visible_exact_shadow_entry_count;
      total.exact_incomplete_snapshot_count += entry_summary.exact_incomplete_snapshot_count;
      total.exact_changed_snapshot_count += entry_summary.exact_changed_snapshot_count;
      total.exact_malformed_snapshot_count += entry_summary.exact_malformed_snapshot_count;
      total.visible_inline_atomic_release_count +=
          entry_summary.visible_inline_atomic_release_count;
      total.release_incomplete_snapshot_count += entry_summary.release_incomplete_snapshot_count;
      total.release_changed_snapshot_count += entry_summary.release_changed_snapshot_count;
      total.release_overflow_snapshot_count += entry_summary.release_overflow_snapshot_count;
      total.release_source_incomplete_snapshot_count +=
          entry_summary.release_source_incomplete_snapshot_count;
      total.release_malformed_snapshot_count += entry_summary.release_malformed_snapshot_count;
      total.visible_inline_acquired_token_count +=
          entry_summary.visible_inline_acquired_token_count;
      total.token_incomplete_snapshot_count += entry_summary.token_incomplete_snapshot_count;
      total.token_changed_snapshot_count += entry_summary.token_changed_snapshot_count;
      total.token_malformed_snapshot_count += entry_summary.token_malformed_snapshot_count;
      total.inline_undercoverage_count += entry_summary.inline_undercoverage_count;
      total.inline_overflow_count += entry_summary.inline_overflow_count;
      total.inline_unsupported_count += entry_summary.inline_unsupported_count;
      total.inline_malformed_count += entry_summary.inline_malformed_count;
      total.visible_sampled_watchpoint_count += entry_summary.visible_sampled_watchpoint_count;
      total.visible_sampled_sync_metadata_count +=
          entry_summary.visible_sampled_sync_metadata_count;
      total.dropped_access_record_count += entry_summary.dropped_access_record_count;
      total.dropped_barrier_record_count += entry_summary.dropped_barrier_record_count;
      total.dropped_atomic_record_count += entry_summary.dropped_atomic_record_count;
      total.dropped_fence_record_count += entry_summary.dropped_fence_record_count;
      total.dropped_diagnostic_record_count += entry_summary.dropped_diagnostic_record_count;
      total.replay_conflict_count += entry_summary.replay_conflict_count;
      total.replay_diagnostic_count += entry_summary.replay_diagnostic_count;
      total.replay_dropped_access_count += entry_summary.replay_dropped_access_count;
      total.replay_dropped_barrier_count += entry_summary.replay_dropped_barrier_count;
      total.replay_unsupported_access_count += entry_summary.replay_unsupported_access_count;
      total.replay_unsupported_atomic_count += entry_summary.replay_unsupported_atomic_count;
      total.replay_unsupported_fence_count += entry_summary.replay_unsupported_fence_count;
      total.replay_metadata_full_count += entry_summary.replay_metadata_full_count;
      total.replay_diagnostic_capacity_exhausted_count +=
          entry_summary.replay_diagnostic_capacity_exhausted_count;
      total.sampled_conflict_count += entry_summary.sampled_conflict_count;
      total.sampled_immediate_conflict_count += entry_summary.sampled_immediate_conflict_count;
      total.sampled_claimed_window_count += entry_summary.sampled_claimed_window_count;
      total.sampled_dropped_window_count += entry_summary.sampled_dropped_window_count;
      total.sampled_saturated_window_count += entry_summary.sampled_saturated_window_count;
      total.sampled_stale_snapshot_count += entry_summary.sampled_stale_snapshot_count;
      total.sampled_incomplete_snapshot_count += entry_summary.sampled_incomplete_snapshot_count;
      total.sampled_changed_snapshot_count += entry_summary.sampled_changed_snapshot_count;
      total.sampled_malformed_snapshot_count += entry_summary.sampled_malformed_snapshot_count;
      total.sampled_unsupported_sync_count += entry_summary.sampled_unsupported_sync_count;
      total.sampled_malformed_sync_count += entry_summary.sampled_malformed_sync_count;
      bool freed = entries_[i].ptr == nullptr;
      hsa_status_t free_status = HSA_STATUS_SUCCESS;
      if (!freed && (core == nullptr || core->hsa_memory_free_fn == nullptr)) {
        // ROCR clears its callable API table before invoking a tool's late
        // OnUnload callback. Allocations from that runtime are no longer live
        // at that point, so close the logical budget without pretending that
        // the unavailable API was called.
        freed = true;
        log_message(kLogInfo,
                    "ConSan MOI auto report cleanup reader=%llu bytes=%zu "
                    "outcome=runtime-reclaimed api=unavailable",
                    static_cast<unsigned long long>(entries_[i].reader), entries_[i].size);
      } else if (!freed) {
        free_status = core->hsa_memory_free_fn(entries_[i].ptr);
        // ROCR may invoke the tool's OnUnload callback after runtime shutdown
        // has already reclaimed its allocation registry. The free then
        // returns NOT_INITIALIZED (observed live) or INVALID_ALLOCATION even
        // though no live HSA allocation remains. Treat only those statuses as
        // successful runtime-owned cleanup; every other failure remains real.
        freed = free_status == HSA_STATUS_SUCCESS ||
                free_status == HSA_STATUS_ERROR_INVALID_ALLOCATION ||
                free_status == HSA_STATUS_ERROR_NOT_INITIALIZED;
        if (free_status == HSA_STATUS_ERROR_INVALID_ALLOCATION ||
            free_status == HSA_STATUS_ERROR_NOT_INITIALIZED) {
          log_message(kLogInfo,
                      "ConSan MOI auto report cleanup reader=%llu bytes=%zu "
                      "outcome=runtime-reclaimed status=%d",
                      static_cast<unsigned long long>(entries_[i].reader), entries_[i].size,
                      static_cast<int>(free_status));
        }
      }
      if (freed) {
        (void)rocjitsu::release_consan_moi_auto_report_bytes(process_budget_, entries_[i].size);
      } else {
        log_message(kLogInfo,
                    "ConSan MOI auto report cleanup reader=%llu bytes=%zu outcome=failed "
                    "status=%d",
                    static_cast<unsigned long long>(entries_[i].reader), entries_[i].size,
                    static_cast<int>(free_status));
        ++cleanup_failure_count_;
      }
      entries_[i] = Entry{};
    }
    entry_count_ = 0;
    total.current_live_report_bytes_after_cleanup = process_budget_.current_live_bytes;
    total.cleanup_failure_count = cleanup_failure_count_;
    required_report_bytes_ = 0;
    successful_allocated_bytes_ = 0;
    allocation_failure_count_ = 0;
    capacity_failure_count_ = 0;
    cleanup_failure_count_ = 0;
    process_budget_.peak_live_bytes = process_budget_.current_live_bytes;
    return total;
  }

private:
  struct RegionSearch {
    CoreApiTable *core = nullptr;
    size_t requested_size = 0;
    hsa_region_t region{};
    bool found = false;
    bool fine_grained = false;
  };

  struct Entry {
    uint64_t reader = 0;
    void *ptr = nullptr;
    size_t size = 0;
    size_t required_size = 0;
    uint64_t generation = 0;
    rocjitsu::ConSanMoiReportBufferLayout layout;
    uint32_t access_record_capacity = 0;
    uint32_t barrier_record_capacity = 0;
    uint32_t atomic_record_capacity = 0;
    uint32_t fence_record_capacity = 0;
    uint32_t diagnostic_capacity = 0;
    uint32_t exact_shadow_entry_capacity = 0;
    uint32_t inline_atomic_release_capacity = 0;
    uint32_t inline_acquired_epoch_token_capacity = 0;
    uint32_t inline_causal_snapshot_capacity = 0;
    uint32_t sampled_watchpoint_capacity = 0;
    bool direct_sampled = false;
    bool inline_shadow = false;
    bool fine_grained = false;
    uint32_t compact_token_mapping_count = 0;
    bool compact_token_mapping_malformed = false;
  };

  void record_allocation_attempt(uint64_t required_size) {
    std::lock_guard lock(mutex_);
    if (required_size > std::numeric_limits<uint64_t>::max() - required_report_bytes_)
      required_report_bytes_ = std::numeric_limits<uint64_t>::max();
    else
      required_report_bytes_ += required_size;
  }

  void record_allocation_failure(uint64_t, bool capacity_failure) {
    std::lock_guard lock(mutex_);
    ++allocation_failure_count_;
    if (capacity_failure)
      ++capacity_failure_count_;
  }

  [[nodiscard]] bool reserve_live_bytes(uint64_t, uint64_t requested_size) {
    std::lock_guard lock(mutex_);
    if (entry_count_ + reserved_entry_count_ >= entries_.size() ||
        !rocjitsu::reserve_consan_moi_auto_report_bytes(process_budget_, requested_size)) {
      ++allocation_failure_count_;
      ++capacity_failure_count_;
      return false;
    }
    ++reserved_entry_count_;
    return true;
  }

  void release_live_bytes(uint64_t requested_size) {
    std::lock_guard lock(mutex_);
    if (reserved_entry_count_ != 0)
      --reserved_entry_count_;
    (void)rocjitsu::release_consan_moi_auto_report_bytes(process_budget_, requested_size);
  }

  [[nodiscard]] uint64_t current_live_bytes() const {
    std::lock_guard lock(mutex_);
    return process_budget_.current_live_bytes;
  }

  [[nodiscard]] uint64_t peak_live_bytes() const {
    std::lock_guard lock(mutex_);
    return process_budget_.peak_live_bytes;
  }

  static hsa_status_t HSA_API select_region(hsa_region_t region, void *data) {
    auto *search = static_cast<RegionSearch *>(data);
    hsa_region_segment_t segment{};
    hsa_status_t status =
        search->core->hsa_region_get_info_fn(region, HSA_REGION_INFO_SEGMENT, &segment);
    if (status != HSA_STATUS_SUCCESS)
      return status;
    if (segment != HSA_REGION_SEGMENT_GLOBAL)
      return HSA_STATUS_SUCCESS;

    bool alloc_allowed = false;
    status = search->core->hsa_region_get_info_fn(region, HSA_REGION_INFO_RUNTIME_ALLOC_ALLOWED,
                                                  &alloc_allowed);
    if (status != HSA_STATUS_SUCCESS)
      return status;
    if (!alloc_allowed)
      return HSA_STATUS_SUCCESS;

    size_t max_size = 0;
    status =
        search->core->hsa_region_get_info_fn(region, HSA_REGION_INFO_ALLOC_MAX_SIZE, &max_size);
    if (status != HSA_STATUS_SUCCESS)
      return status;
    if (max_size < search->requested_size)
      return HSA_STATUS_SUCCESS;

    uint32_t flags = 0;
    status = search->core->hsa_region_get_info_fn(region, HSA_REGION_INFO_GLOBAL_FLAGS, &flags);
    if (status != HSA_STATUS_SUCCESS)
      return status;
    const bool fine_grained = (flags & HSA_REGION_GLOBAL_FLAG_FINE_GRAINED) != 0;
    const bool coarse_grained = (flags & HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED) != 0;
    if (fine_grained) {
      search->region = region;
      search->found = true;
      search->fine_grained = true;
      return HSA_STATUS_INFO_BREAK;
    }
    if (!search->found && coarse_grained) {
      search->region = region;
      search->found = true;
      search->fine_grained = false;
    }
    return HSA_STATUS_SUCCESS;
  }

  Summary summarize(CoreApiTable *core, const Entry &entry) {
    Summary summary;
    summary.buffer_count = 1;
    std::vector<uint8_t> snapshot;
    const void *report_ptr = entry.ptr;
    if (!entry.fine_grained) {
      if (core == nullptr || core->hsa_memory_copy_fn == nullptr) {
        log_message(kLogInfo,
                    "ConSan MOI auto report reader=%llu needs hsa_memory_copy for "
                    "coarse-grained summary",
                    static_cast<unsigned long long>(entry.reader));
        return summary;
      }
      snapshot.resize(entry.size);
      const hsa_status_t copy_status =
          core->hsa_memory_copy_fn(snapshot.data(), entry.ptr, entry.size);
      if (copy_status != HSA_STATUS_SUCCESS) {
        log_message(kLogInfo, "ConSan MOI auto report reader=%llu hsa_memory_copy failed status=%d",
                    static_cast<unsigned long long>(entry.reader), static_cast<int>(copy_status));
        return summary;
      }
      report_ptr = snapshot.data();
    }

    const auto *header = static_cast<const rocjitsu::ConSanMoiReportHeader *>(report_ptr);
    if (!rocjitsu::consan_moi_report_header_is_current(*header)) {
      log_message(kLogInfo,
                  "ConSan MOI auto report reader=%llu has invalid header magic=0x%08x "
                  "abi=%u header_size=%u",
                  static_cast<unsigned long long>(entry.reader), header->magic, header->abi_version,
                  header->header_size);
      return summary;
    }
    const rocjitsu::ConSanMoiReportBufferLayout &expected_layout = entry.layout;
    const rocjitsu::ConSanMoiEngine expected_engine =
        entry.inline_shadow    ? rocjitsu::ConSanMoiEngine::InlineShadow
        : entry.direct_sampled ? rocjitsu::ConSanMoiEngine::Sampled
                               : rocjitsu::ConSanMoiEngine::RecordReplay;
    if (!rocjitsu::consan_moi_report_layout_matches_header(*header, expected_layout,
                                                           expected_engine, entry.size)) {
      log_message(kLogInfo, "ConSan MOI auto report reader=%llu has inconsistent ABI-v%u layout",
                  static_cast<unsigned long long>(entry.reader),
                  rocjitsu::kConSanMoiReportAbiVersion);
      return summary;
    }
    const bool partition_mask_debug = [] {
      const char *debug = std::getenv("RJ_CONSAN_MOI_PARTITION_MASK_DEBUG");
      return debug != nullptr && std::string_view(debug) == "1";
    }();
    // The bounded partition debugger reuses record-count words as an explicit
    // non-acceptance side channel. Keep those words out of the ordinary
    // overflow/accounting summary while preserving the registered capacities
    // that define the report layout.
    const uint32_t access_record_count = partition_mask_debug ? 0 : header->access_record_count;
    const uint32_t barrier_record_count = partition_mask_debug ? 0 : header->barrier_record_count;
    const uint32_t atomic_record_count = partition_mask_debug ? 0 : header->atomic_record_count;
    const uint32_t visible_records = std::min(access_record_count, header->access_record_capacity);
    const uint32_t visible_barriers =
        std::min(barrier_record_count, header->barrier_record_capacity);
    const uint32_t visible_atomics = std::min(atomic_record_count, header->atomic_record_capacity);
    const uint32_t visible_fences =
        std::min(header->fence_record_count, entry.fence_record_capacity);
    const uint32_t visible_diagnostics =
        std::min(header->diagnostic_count, header->diagnostic_capacity);
    const uint32_t dropped_records =
        access_record_count > visible_records ? access_record_count - visible_records : 0;
    const uint32_t dropped_barriers =
        barrier_record_count > visible_barriers ? barrier_record_count - visible_barriers : 0;
    const uint32_t dropped_atomics =
        atomic_record_count > visible_atomics ? atomic_record_count - visible_atomics : 0;
    const uint32_t dropped_fences =
        entry.fence_record_capacity != 0 && header->fence_record_count > visible_fences
            ? header->fence_record_count - visible_fences
            : 0;
    const uint32_t dropped_diagnostics = header->diagnostic_count > visible_diagnostics
                                             ? header->diagnostic_count - visible_diagnostics
                                             : 0;
    const auto *bytes = static_cast<const uint8_t *>(report_ptr);
    const auto *exact_shadow = reinterpret_cast<const rocjitsu::ConSanMoiInlineExactShadowSlot *>(
        bytes + expected_layout.exact_shadow_entries_offset);
    const auto *inline_atomic_releases =
        reinterpret_cast<const rocjitsu::ConSanMoiInlineAtomicReleaseSlot *>(
            bytes + expected_layout.inline_atomic_release_slots_offset);
    const uint32_t inline_atomic_release_capacity =
        entry.inline_shadow ? header->inline_atomic_release_capacity : 0;
    const auto *inline_acquired_tokens =
        reinterpret_cast<const volatile rocjitsu::ConSanMoiInlineAcquiredEpochTokenSlot *>(
            bytes + expected_layout.inline_acquired_epoch_token_slots_offset);
    const uint32_t inline_acquired_token_capacity =
        entry.inline_shadow ? header->inline_acquired_epoch_token_capacity : 0;
    const auto *inline_causal_snapshots =
        reinterpret_cast<const rocjitsu::ConSanMoiInlineCausalSnapshot *>(
            bytes + expected_layout.inline_causal_snapshots_offset);
    const auto *sampled_causal_windows =
        reinterpret_cast<const rocjitsu::ConSanMoiSampledCausalWindow *>(
            bytes + expected_layout.sampled_causal_windows_offset);
    const uint32_t sampled_watchpoint_capacity =
        entry.direct_sampled ? header->sampled_watchpoint_capacity : 0;
    const uint32_t sampled_causal_window_capacity =
        entry.direct_sampled ? header->sampled_causal_window_capacity : 0;
    const auto *sampled =
        reinterpret_cast<const uint64_t *>(bytes + expected_layout.sampled_watchpoints_offset);
    const auto *sampled_sync_words = reinterpret_cast<const volatile uint32_t *>(
        bytes + expected_layout.sampled_sync_metadata_offset);
    const uint32_t sampled_sync_metadata_capacity =
        entry.direct_sampled ? header->sampled_sync_metadata_capacity : 0;
    const auto *sampled_pending_acquires =
        reinterpret_cast<const volatile rocjitsu::ConSanMoiSampledPendingAcquireSlot *>(
            bytes + expected_layout.sampled_pending_acquires_offset);
    const uint32_t sampled_pending_acquire_capacity =
        entry.direct_sampled ? header->sampled_pending_acquire_capacity : 0;
    struct ExactShadowEntry {
      uint32_t index = 0;
      rocjitsu::ConSanMoiExactShadowEntry entry;
      uint64_t dispatch_id = 0;
      uint32_t version = 0;
    };
    std::vector<ExactShadowEntry> visible_exact_shadow;
    for (uint32_t i = 0; i < header->exact_shadow_entry_capacity; ++i) {
      const volatile auto &slot = exact_shadow[i];
      const uint32_t version_before = slot.version;
      const uint64_t packed_access = slot.packed_access;
      const uint64_t dispatch_id = slot.dispatch_id;
      const uint32_t reserved = slot.reserved;
      const uint32_t version_after = slot.version;
      const auto snapshot = rocjitsu::classify_consan_moi_inline_exact_snapshot(
          {version_before, packed_access, dispatch_id, reserved, version_after});
      switch (snapshot.state) {
      case rocjitsu::ConSanMoiInlineExactSnapshotState::Empty:
        break;
      case rocjitsu::ConSanMoiInlineExactSnapshotState::Stable:
        visible_exact_shadow.push_back({i, snapshot.entry, snapshot.dispatch_id, snapshot.version});
        break;
      case rocjitsu::ConSanMoiInlineExactSnapshotState::Publishing:
        ++summary.exact_incomplete_snapshot_count;
        break;
      case rocjitsu::ConSanMoiInlineExactSnapshotState::ChangedDuringRead:
        ++summary.exact_changed_snapshot_count;
        break;
      case rocjitsu::ConSanMoiInlineExactSnapshotState::Malformed:
        if (summary.exact_malformed_snapshot_count == 0) {
          log_message(kLogInfo,
                      "ConSan MOI first malformed exact snapshot reader=%llu index=%u "
                      "version_before=%u packed_access=0x%016llx dispatch_id=0x%016llx "
                      "reserved=%u version_after=%u",
                      static_cast<unsigned long long>(entry.reader), i, version_before,
                      static_cast<unsigned long long>(packed_access),
                      static_cast<unsigned long long>(dispatch_id), reserved, version_after);
        }
        ++summary.exact_malformed_snapshot_count;
        break;
      }
    }
    struct InlineAtomicReleaseEntry {
      uint32_t index = 0;
      rocjitsu::ConSanMoiInlineAtomicReleaseSlot slot;
      rocjitsu::ConSanMoiInlineCausalSnapshot snapshot;
    };
    std::vector<InlineAtomicReleaseEntry> visible_inline_atomic_releases;
    for (uint32_t i = 0; i < inline_atomic_release_capacity; ++i) {
      const volatile auto &slot = inline_atomic_releases[i];
      const volatile auto &source_snapshot = inline_causal_snapshots[i];
      rocjitsu::ConSanMoiInlineReleaseSnapshotWords words;
      words.version_before = slot.version;
      words.slot.version = words.version_before;
      words.slot.owner_id = slot.owner_id;
      words.slot.epoch_plus_one = slot.epoch_plus_one;
      words.slot.workgroup_key = slot.workgroup_key;
      words.slot.atomic_address = slot.atomic_address;
      words.slot.dispatch_id = slot.dispatch_id;
      words.snapshot.entry_count = source_snapshot.entry_count;
      words.snapshot.flags = source_snapshot.flags;
      const auto *source_entries =
          reinterpret_cast<const volatile rocjitsu::ConSanMoiInlineCausalSnapshotEntry *>(
              &source_snapshot.entries);
      for (uint32_t entry_index = 0;
           entry_index < rocjitsu::kConSanMoiInlineCausalSnapshotEntryCapacity; ++entry_index) {
        words.snapshot.entries[entry_index].ancestor_owner_id =
            source_entries[entry_index].ancestor_owner_id;
        words.snapshot.entries[entry_index].ancestor_epoch_plus_one =
            source_entries[entry_index].ancestor_epoch_plus_one;
      }
      words.version_after = slot.version;
      const auto classified = rocjitsu::classify_consan_moi_inline_release_snapshot(words);
      switch (classified.state) {
      case rocjitsu::ConSanMoiInlineReleaseSnapshotState::Empty:
        break;
      case rocjitsu::ConSanMoiInlineReleaseSnapshotState::Stable:
        visible_inline_atomic_releases.push_back({i, words.slot, words.snapshot});
        break;
      case rocjitsu::ConSanMoiInlineReleaseSnapshotState::Publishing:
        ++summary.release_incomplete_snapshot_count;
        log_message(kLogInfo,
                    "ConSan MOI auto incomplete-inline-atomic-release reader=%llu index=%u "
                    "version=%u owner=%u epoch_plus_one=%u workgroup=%u address=0x%llx "
                    "dispatch=0x%llx",
                    static_cast<unsigned long long>(entry.reader), i, words.slot.version,
                    words.slot.owner_id, words.slot.epoch_plus_one, words.slot.workgroup_key,
                    static_cast<unsigned long long>(words.slot.atomic_address),
                    static_cast<unsigned long long>(words.slot.dispatch_id));
        break;
      case rocjitsu::ConSanMoiInlineReleaseSnapshotState::ChangedDuringRead:
        ++summary.release_changed_snapshot_count;
        break;
      case rocjitsu::ConSanMoiInlineReleaseSnapshotState::CapacityOverflow:
        ++summary.release_overflow_snapshot_count;
        break;
      case rocjitsu::ConSanMoiInlineReleaseSnapshotState::SourceIncomplete:
        ++summary.release_source_incomplete_snapshot_count;
        break;
      case rocjitsu::ConSanMoiInlineReleaseSnapshotState::Malformed:
        ++summary.release_malformed_snapshot_count;
        break;
      }
    }
    struct InlineAcquiredTokenEntry {
      uint32_t index = 0;
      rocjitsu::ConSanMoiInlineAcquiredEpochTokenSlot token;
    };
    std::vector<InlineAcquiredTokenEntry> visible_inline_acquired_tokens;
    for (uint32_t i = 0; i < inline_acquired_token_capacity; ++i) {
      const volatile auto &slot = inline_acquired_tokens[i];
      rocjitsu::ConSanMoiInlineAcquiredTokenSnapshot snapshot;
      snapshot.version_before = slot.version;
      snapshot.payload.version = snapshot.version_before;
      snapshot.payload.consumer_owner_id = slot.consumer_owner_id;
      snapshot.payload.producer_owner_id = slot.producer_owner_id;
      snapshot.payload.producer_epoch_plus_one = slot.producer_epoch_plus_one;
      snapshot.payload.workgroup_key = slot.workgroup_key;
      snapshot.payload.kind = slot.kind;
      snapshot.payload.dispatch_id = slot.dispatch_id;
      snapshot.payload.source_release_address = slot.source_release_address;
      snapshot.payload.source_release_version = slot.source_release_version;
      snapshot.payload.reserved = slot.reserved;
      snapshot.version_after = slot.version;
      const auto classified = rocjitsu::consan_moi_inline_classify_acquired_token(snapshot);
      switch (classified.state) {
      case rocjitsu::ConSanMoiInlineAcquiredTokenState::Empty:
        break;
      case rocjitsu::ConSanMoiInlineAcquiredTokenState::Stable:
        visible_inline_acquired_tokens.push_back({i, classified.token});
        break;
      case rocjitsu::ConSanMoiInlineAcquiredTokenState::Publishing:
        ++summary.token_incomplete_snapshot_count;
        break;
      case rocjitsu::ConSanMoiInlineAcquiredTokenState::Changed:
        ++summary.token_changed_snapshot_count;
        break;
      case rocjitsu::ConSanMoiInlineAcquiredTokenState::Malformed:
        ++summary.token_malformed_snapshot_count;
        break;
      }
    }
    struct SampledEntry {
      uint32_t index = 0;
      rocjitsu::ConSanMoiSampledWatchpointEntry entry;
      rocjitsu::ConSanMoiSampledSyncDecodeResult sync;
      uint32_t workgroup_x = 0;
      uint32_t workgroup_y = 0;
      uint32_t workgroup_z = 0;
      uint32_t epoch = 0;
    };
    std::vector<SampledEntry> visible_sampled;
    uint32_t visible_sampled_sync_metadata = 0;
    const auto *sampled_words = reinterpret_cast<const volatile uint32_t *>(sampled);
    const uint32_t active_sampled_generation =
        static_cast<uint32_t>(header->generation) &
        rocjitsu::consan_moi_sampled_watchpoint::max_generation;
    for (uint32_t i = 0; i < sampled_watchpoint_capacity; ++i) {
      if (i >= sampled_causal_window_capacity) {
        ++summary.sampled_malformed_snapshot_count;
        continue;
      }
      const volatile auto &window = sampled_causal_windows[i];
      const uint32_t state_before = window.publication_state;
      const uint32_t low_before = sampled_words[2u * i];
      const uint32_t high = sampled_words[2u * i + 1u];
      const uint32_t low_after = sampled_words[2u * i];
      const uint64_t window_generation = window.generation;
      const uint64_t window_dispatch_id = window.dispatch_id;
      const uint32_t window_x = window.workgroup_x;
      const uint32_t window_y = window.workgroup_y;
      const uint32_t window_z = window.workgroup_z;
      const uint32_t window_epoch = window.epoch;
      const uint32_t window_first_entry = window.first_entry;
      const uint32_t window_entry_count = window.entry_count;
      const uint32_t window_cluster_workgroup_id = window.cluster_workgroup_id;
      rocjitsu::ConSanMoiSampledSyncMetadataPacked sync_packed{};
      uint32_t sync_descriptor_before = 0;
      uint32_t sync_descriptor_after = 0;
      rocjitsu::ConSanMoiSampledPendingAcquireView pending_view{};
      if (i < sampled_sync_metadata_capacity) {
        const size_t sync_word = static_cast<size_t>(i) * (sizeof(sync_packed) / sizeof(uint32_t));
        sync_descriptor_before = sampled_sync_words[sync_word + 3u];
        sync_packed.address = static_cast<uint64_t>(sampled_sync_words[sync_word]) |
                              (static_cast<uint64_t>(sampled_sync_words[sync_word + 1u]) << 32u);
        sync_packed.byte_count = sampled_sync_words[sync_word + 2u];
        sync_packed.epoch_before = sampled_sync_words[sync_word + 4u];
        sync_packed.epoch_after = sampled_sync_words[sync_word + 5u];
        sync_descriptor_after = sampled_sync_words[sync_word + 3u];
      }
      if (i < sampled_pending_acquire_capacity) {
        const volatile auto &pending = sampled_pending_acquires[i];
        pending_view.version_before = pending.version;
        pending_view.slot.version = pending_view.version_before;
        pending_view.slot.selected_slot = pending.selected_slot;
        pending_view.slot.generation = pending.generation;
        pending_view.slot.dispatch_id = pending.dispatch_id;
        pending_view.slot.workgroup_x = pending.workgroup_x;
        pending_view.slot.workgroup_y = pending.workgroup_y;
        pending_view.slot.workgroup_z = pending.workgroup_z;
        pending_view.slot.owner_id = pending.owner_id;
        pending_view.slot.source_epoch = pending.source_epoch;
        pending_view.slot.reserved = pending.reserved;
        pending_view.slot.metadata.address = pending.metadata.address;
        pending_view.slot.metadata.byte_count = pending.metadata.byte_count;
        pending_view.slot.metadata.descriptor = pending.metadata.descriptor;
        pending_view.slot.metadata.epoch_before = pending.metadata.epoch_before;
        pending_view.slot.metadata.epoch_after = pending.metadata.epoch_after;
        pending_view.version_after = pending.version;
      }
      rocjitsu::ConSanMoiSampledSyncDecodeResult sync =
          i < sampled_sync_metadata_capacity
              ? rocjitsu::classify_consan_moi_sampled_sync_snapshot(
                    {sync_descriptor_before, sync_packed, sync_descriptor_after}, window_epoch)
              : rocjitsu::ConSanMoiSampledSyncDecodeResult{};
      const uint32_t state_after = window.publication_state;
      const rocjitsu::ConSanMoiSampledSnapshot snapshot =
          rocjitsu::classify_consan_moi_sampled_snapshot({low_before, high, low_after},
                                                         active_sampled_generation);
      if (state_before != state_after) {
        ++summary.sampled_changed_snapshot_count;
        continue;
      }
      const auto publication_state =
          static_cast<rocjitsu::ConSanMoiSampledCausalPublicationState>(state_after);
      if (publication_state == rocjitsu::ConSanMoiSampledCausalPublicationState::Empty) {
        if (snapshot.state != rocjitsu::ConSanMoiSampledSnapshotState::Empty ||
            sync.classification != rocjitsu::ConSanMoiSampledSyncClassification::Empty)
          ++summary.sampled_malformed_snapshot_count;
        continue;
      }
      if (publication_state == rocjitsu::ConSanMoiSampledCausalPublicationState::Publishing) {
        ++summary.sampled_incomplete_snapshot_count;
        continue;
      }
      if (publication_state != rocjitsu::ConSanMoiSampledCausalPublicationState::Ready ||
          window_generation != header->generation ||
          window_epoch > rocjitsu::consan_moi_sampled_watchpoint::max_epoch ||
          window_first_entry != i || window_entry_count != 1) {
        log_message(kLogInfo,
                    "ConSan MOI sampled malformed window index=%u state=%u generation=%llu/%llu "
                    "dispatch=%llu/%llu epoch=%u first=%u entries=%u "
                    "cluster_workgroup_id=%u snapshot=%u",
                    i, state_after, static_cast<unsigned long long>(window_generation),
                    static_cast<unsigned long long>(header->generation),
                    static_cast<unsigned long long>(window_dispatch_id),
                    static_cast<unsigned long long>(header->dispatch_id), window_epoch,
                    window_first_entry, window_entry_count, window_cluster_workgroup_id,
                    static_cast<uint32_t>(snapshot.state));
        ++summary.sampled_malformed_snapshot_count;
        continue;
      }
      bool sync_snapshot_usable = true;
      if (sync.classification == rocjitsu::ConSanMoiSampledSyncClassification::ChangedDuringRead) {
        ++summary.sampled_changed_snapshot_count;
        sync_snapshot_usable = false;
      }
      if (sync.classification == rocjitsu::ConSanMoiSampledSyncClassification::Publishing) {
        ++summary.sampled_incomplete_snapshot_count;
        sync_snapshot_usable = false;
      }
      const bool has_watchpoint = snapshot.state != rocjitsu::ConSanMoiSampledSnapshotState::Empty;
      if (!has_watchpoint) {
        log_message(kLogInfo, "ConSan MOI sampled malformed empty watchpoint index=%u state=%u", i,
                    static_cast<uint32_t>(snapshot.state));
        ++summary.sampled_malformed_snapshot_count;
        continue;
      }
      if (snapshot.state == rocjitsu::ConSanMoiSampledSnapshotState::Stable &&
          i < sampled_pending_acquire_capacity) {
        const auto pending_join = rocjitsu::consan_moi_sampled_join_pending_acquire(
            pending_view,
            {window_generation, window_dispatch_id, window_x, window_y, window_z, window_epoch,
             window_first_entry, window_entry_count, state_after, window_cluster_workgroup_id},
            static_cast<uint64_t>(low_after) | (static_cast<uint64_t>(high) << 32u), i);
        switch (pending_join.state) {
        case rocjitsu::ConSanMoiSampledPendingAcquireState::Empty:
          break;
        case rocjitsu::ConSanMoiSampledPendingAcquireState::Ready:
          if (sync.classification == rocjitsu::ConSanMoiSampledSyncClassification::Empty)
            sync = pending_join.sync;
          else {
            ++summary.sampled_malformed_sync_count;
            sync_snapshot_usable = false;
          }
          break;
        case rocjitsu::ConSanMoiSampledPendingAcquireState::Publishing:
          ++summary.sampled_incomplete_snapshot_count;
          ++summary.sampled_malformed_sync_count;
          sync_snapshot_usable = false;
          break;
        case rocjitsu::ConSanMoiSampledPendingAcquireState::ChangedDuringRead:
          ++summary.sampled_changed_snapshot_count;
          ++summary.sampled_malformed_sync_count;
          sync_snapshot_usable = false;
          break;
        case rocjitsu::ConSanMoiSampledPendingAcquireState::Malformed:
        case rocjitsu::ConSanMoiSampledPendingAcquireState::IdentityMismatch:
        case rocjitsu::ConSanMoiSampledPendingAcquireState::FutureEpoch:
          ++summary.sampled_malformed_sync_count;
          sync_snapshot_usable = false;
          break;
        }
      }
      const bool has_sync =
          sync.classification != rocjitsu::ConSanMoiSampledSyncClassification::Empty;
      if (has_sync && sync_snapshot_usable) {
        if (sync.classification != rocjitsu::ConSanMoiSampledSyncClassification::Valid) {
          ++summary.sampled_malformed_sync_count;
          sync_snapshot_usable = false;
        }
      }
      switch (snapshot.state) {
      case rocjitsu::ConSanMoiSampledSnapshotState::Empty:
        break;
      case rocjitsu::ConSanMoiSampledSnapshotState::Stable:
        if (snapshot.entry.epoch != window_epoch) {
          ++summary.sampled_malformed_snapshot_count;
          break;
        }
        if (sync_snapshot_usable && has_sync)
          ++visible_sampled_sync_metadata;
        visible_sampled.push_back(
            {i, snapshot.entry,
             sync_snapshot_usable ? sync : rocjitsu::ConSanMoiSampledSyncDecodeResult{}, window_x,
             window_y, window_z, window_epoch});
        break;
      case rocjitsu::ConSanMoiSampledSnapshotState::StaleGeneration:
        ++summary.sampled_stale_snapshot_count;
        break;
      case rocjitsu::ConSanMoiSampledSnapshotState::IncompletePublication:
        ++summary.sampled_incomplete_snapshot_count;
        break;
      case rocjitsu::ConSanMoiSampledSnapshotState::ChangedDuringRead:
        ++summary.sampled_changed_snapshot_count;
        break;
      case rocjitsu::ConSanMoiSampledSnapshotState::Malformed:
        log_message(kLogInfo,
                    "ConSan MOI sampled malformed packed watchpoint index=%u low=0x%08x "
                    "high=0x%08x epoch=%u",
                    i, low_after, high, window_epoch);
        ++summary.sampled_malformed_snapshot_count;
        break;
      }
    }
    uint32_t sampled_conflicts = 0;
    // A collision or capacity drop can leave a valid-looking first half in a
    // slot while discarding a different second publisher. Without per-slot
    // collision identity, disable all ordering suppression for that report.
    const bool sampled_sync_evidence_complete =
        rocjitsu::consan_moi_sampled_sync_report_is_complete(
            header->sampled_dropped_window_count, header->sampled_unsupported_sync_count,
            header->sampled_malformed_sync_count, header->sampled_pending_acquire_collision_count,
            header->sampled_pending_acquire_malformed_count) &&
        summary.sampled_malformed_sync_count == 0;
    std::optional<std::pair<SampledEntry, SampledEntry>> first_sampled_conflict;
    for (size_t i = 0; i < visible_sampled.size(); ++i) {
      const SampledEntry &current = visible_sampled[i];
      for (size_t prior_index = 0; prior_index < i; ++prior_index) {
        const SampledEntry &prior = visible_sampled[prior_index];
        if (current.workgroup_x != prior.workgroup_x || current.workgroup_y != prior.workgroup_y ||
            current.workgroup_z != prior.workgroup_z || current.epoch != prior.epoch)
          continue;
        if (!rocjitsu::consan_moi_sampled_watchpoints_conflict(current.entry, prior.entry))
          continue;
        if (sampled_sync_evidence_complete &&
            rocjitsu::consan_moi_sampled_atomic_pair_orders_same_workgroup(prior.sync,
                                                                           current.sync))
          continue;
        if (sampled_conflicts != std::numeric_limits<uint32_t>::max())
          ++sampled_conflicts;
        if (!first_sampled_conflict)
          first_sampled_conflict = std::make_pair(prior, current);
      }
    }
    summary.visible_access_record_count = visible_records;
    summary.visible_barrier_record_count = visible_barriers;
    summary.visible_atomic_record_count = visible_atomics;
    summary.visible_fence_record_count = visible_fences;
    summary.visible_diagnostic_record_count = visible_diagnostics;
    summary.visible_inline_publication_count =
        entry.inline_shadow && !partition_mask_debug ? header->event_counter : 0;
    summary.visible_exact_shadow_entry_count = visible_exact_shadow.size();
    summary.visible_inline_atomic_release_count = visible_inline_atomic_releases.size();
    summary.visible_inline_acquired_token_count = visible_inline_acquired_tokens.size();
    summary.visible_sampled_watchpoint_count = visible_sampled.size();
    summary.visible_sampled_sync_metadata_count = visible_sampled_sync_metadata;
    summary.dropped_access_record_count = dropped_records;
    summary.dropped_barrier_record_count = dropped_barriers;
    summary.dropped_atomic_record_count = dropped_atomics;
    summary.dropped_fence_record_count = dropped_fences;
    summary.dropped_diagnostic_record_count = dropped_diagnostics;
    summary.sampled_conflict_count = sampled_conflicts;
    summary.sampled_immediate_conflict_count =
        sampled_watchpoint_capacity != 0 ? header->event_counter : 0;
    summary.sampled_claimed_window_count =
        sampled_watchpoint_capacity != 0 ? header->sampled_causal_window_count : 0;
    summary.sampled_dropped_window_count =
        sampled_watchpoint_capacity != 0 ? header->sampled_dropped_window_count : 0;
    summary.sampled_saturated_window_count =
        sampled_watchpoint_capacity != 0 ? header->sampled_saturated_window_count : 0;
    summary.sampled_unsupported_sync_count =
        sampled_watchpoint_capacity != 0 ? header->sampled_unsupported_sync_count : 0;
    summary.sampled_malformed_sync_count +=
        sampled_watchpoint_capacity != 0 ? header->sampled_malformed_sync_count : 0;
    summary.inline_undercoverage_count =
        entry.inline_shadow && !partition_mask_debug ? header->inline_undercoverage_count : 0;
    summary.inline_overflow_count = entry.inline_shadow ? header->inline_overflow_count : 0;
    summary.inline_unsupported_count = entry.inline_shadow ? header->inline_unsupported_count : 0;
    summary.inline_malformed_count = entry.inline_shadow ? header->inline_malformed_count : 0;
    if (partition_mask_debug) {
      const uint64_t group_mask = header->dispatch_id;
      const uint64_t first_exchange_address =
          static_cast<uint64_t>(header->access_record_count) |
          (static_cast<uint64_t>(header->barrier_record_count) << 32u);
      const uint64_t final_exchange_address = static_cast<uint64_t>(header->atomic_record_count) |
                                              (static_cast<uint64_t>(header->flags) << 32u);
      log_message(kLogInfo,
                  "ConSan MOI partition-mask debug reader=%llu acceptance=false "
                  "final_group=0x%016llx exchange_count=%u "
                  "first_exchange_address=0x%016llx final_exchange_address=0x%016llx",
                  static_cast<unsigned long long>(entry.reader),
                  static_cast<unsigned long long>(group_mask), header->event_counter,
                  static_cast<unsigned long long>(first_exchange_address),
                  static_cast<unsigned long long>(final_exchange_address));
    }
    log_message(kLogInfo,
                "ConSan MOI auto report reader=%llu addr=0x%llx bytes=%zu generation=%llu "
                "event_counter=%u access_records=%u visible_records=%u dropped_records=%u "
                "capacity=%u "
                "barrier_records=%u visible_barriers=%u dropped_barriers=%u barrier_capacity=%u "
                "atomic_records=%u visible_atomics=%u dropped_atomics=%u atomic_capacity=%u "
                "fence_records=%u visible_fences=%u dropped_fences=%u fence_capacity=%u "
                "diagnostics=%u visible_diagnostics=%u dropped_diagnostics=%u "
                "diagnostic_capacity=%u "
                "exact_shadow_capacity=%u visible_exact_shadow=%zu "
                "exact_incomplete_snapshots=%llu exact_changed_snapshots=%llu "
                "exact_malformed_snapshots=%llu "
                "inline_atomic_release_capacity=%u visible_inline_atomic_releases=%zu "
                "release_incomplete_snapshots=%llu release_changed_snapshots=%llu "
                "release_overflow_snapshots=%llu release_source_incomplete_snapshots=%llu "
                "release_malformed_snapshots=%llu "
                "inline_acquired_token_capacity=%u visible_inline_acquired_tokens=%zu "
                "token_incomplete_snapshots=%llu token_changed_snapshots=%llu "
                "token_malformed_snapshots=%llu "
                "inline_undercoverage=%llu inline_overflow=%llu "
                "inline_unsupported=%llu inline_malformed=%llu "
                "sampled_watchpoints=%u visible_sampled=%zu sampled_sync_capacity=%u "
                "visible_sampled_sync=%u sampled_unsupported_sync=%u sampled_malformed_sync=%llu "
                "sampled_pending_acquire_capacity=%u sampled_pending_acquires=%u "
                "sampled_pending_acquire_contention=%u "
                "sampled_pending_acquire_collisions=%u sampled_pending_acquire_malformed=%u "
                "sampled_conflicts=%u "
                "sampled_immediate_conflicts=%u sampled_claimed_windows=%u "
                "sampled_dropped_windows=%u sampled_saturated_windows=%u "
                "sampled_stale_snapshots=%llu sampled_incomplete_snapshots=%llu "
                "sampled_changed_snapshots=%llu sampled_malformed_snapshots=%llu "
                "fine_grained=%s",
                static_cast<unsigned long long>(entry.reader),
                static_cast<unsigned long long>(reinterpret_cast<uint64_t>(entry.ptr)), entry.size,
                static_cast<unsigned long long>(header->generation), header->event_counter,
                access_record_count, visible_records, dropped_records,
                header->access_record_capacity, barrier_record_count, visible_barriers,
                dropped_barriers, header->barrier_record_capacity, atomic_record_count,
                visible_atomics, dropped_atomics, header->atomic_record_capacity,
                header->fence_record_count, visible_fences, dropped_fences,
                entry.fence_record_capacity, header->diagnostic_count, visible_diagnostics,
                dropped_diagnostics, header->diagnostic_capacity,
                header->exact_shadow_entry_capacity, visible_exact_shadow.size(),
                static_cast<unsigned long long>(summary.exact_incomplete_snapshot_count),
                static_cast<unsigned long long>(summary.exact_changed_snapshot_count),
                static_cast<unsigned long long>(summary.exact_malformed_snapshot_count),
                header->inline_atomic_release_capacity, visible_inline_atomic_releases.size(),
                static_cast<unsigned long long>(summary.release_incomplete_snapshot_count),
                static_cast<unsigned long long>(summary.release_changed_snapshot_count),
                static_cast<unsigned long long>(summary.release_overflow_snapshot_count),
                static_cast<unsigned long long>(summary.release_source_incomplete_snapshot_count),
                static_cast<unsigned long long>(summary.release_malformed_snapshot_count),
                header->inline_acquired_epoch_token_capacity, visible_inline_acquired_tokens.size(),
                static_cast<unsigned long long>(summary.token_incomplete_snapshot_count),
                static_cast<unsigned long long>(summary.token_changed_snapshot_count),
                static_cast<unsigned long long>(summary.token_malformed_snapshot_count),
                static_cast<unsigned long long>(summary.inline_undercoverage_count),
                static_cast<unsigned long long>(summary.inline_overflow_count),
                static_cast<unsigned long long>(summary.inline_unsupported_count),
                static_cast<unsigned long long>(summary.inline_malformed_count),
                sampled_watchpoint_capacity, visible_sampled.size(), sampled_sync_metadata_capacity,
                visible_sampled_sync_metadata, header->sampled_unsupported_sync_count,
                static_cast<unsigned long long>(summary.sampled_malformed_sync_count),
                header->sampled_pending_acquire_capacity, header->sampled_pending_acquire_count,
                header->sampled_pending_acquire_contention_count,
                header->sampled_pending_acquire_collision_count,
                header->sampled_pending_acquire_malformed_count, sampled_conflicts,
                static_cast<uint32_t>(summary.sampled_immediate_conflict_count),
                header->sampled_causal_window_count, header->sampled_dropped_window_count,
                header->sampled_saturated_window_count,
                static_cast<unsigned long long>(summary.sampled_stale_snapshot_count),
                static_cast<unsigned long long>(summary.sampled_incomplete_snapshot_count),
                static_cast<unsigned long long>(summary.sampled_changed_snapshot_count),
                static_cast<unsigned long long>(summary.sampled_malformed_snapshot_count),
                entry.fine_grained ? "true" : "false");

    for (size_t i = 0; i < visible_inline_atomic_releases.size(); ++i) {
      const auto &release = visible_inline_atomic_releases[i];
      const auto &slot = release.slot;
      const auto &snapshot = release.snapshot;
      log_message(
          kLogInfo,
          "ConSan MOI auto inline-atomic-release reader=%llu index=%u version=%u "
          "owner=%u epoch_plus_one=%u workgroup=%u address=0x%llx dispatch=0x%llx "
          "snapshot_flags=%u snapshot_count=%u "
          "snapshot0_owner=%u snapshot0_epoch_plus_one=%u "
          "snapshot1_owner=%u snapshot1_epoch_plus_one=%u "
          "snapshot2_owner=%u snapshot2_epoch_plus_one=%u "
          "snapshot3_owner=%u snapshot3_epoch_plus_one=%u",
          static_cast<unsigned long long>(entry.reader), release.index, slot.version, slot.owner_id,
          slot.epoch_plus_one, slot.workgroup_key,
          static_cast<unsigned long long>(slot.atomic_address),
          static_cast<unsigned long long>(slot.dispatch_id), snapshot.flags, snapshot.entry_count,
          snapshot.entries[0].ancestor_owner_id, snapshot.entries[0].ancestor_epoch_plus_one,
          snapshot.entries[1].ancestor_owner_id, snapshot.entries[1].ancestor_epoch_plus_one,
          snapshot.entries[2].ancestor_owner_id, snapshot.entries[2].ancestor_epoch_plus_one,
          snapshot.entries[3].ancestor_owner_id, snapshot.entries[3].ancestor_epoch_plus_one);
    }

    for (const auto &entry_token : visible_inline_acquired_tokens) {
      const auto &token = entry_token.token;
      log_message(
          kLogInfo,
          "ConSan MOI auto inline-acquired-token reader=%llu index=%u version=%u "
          "kind=%s consumer=%u producer=%u epoch_plus_one=%u workgroup=%u "
          "dispatch=0x%llx source_address=0x%llx source_version=%u",
          static_cast<unsigned long long>(entry.reader), entry_token.index, token.version,
          token.kind == static_cast<uint32_t>(rocjitsu::ConSanMoiInlineTokenEvidenceKind::Direct)
              ? "direct"
          : token.kind ==
                  static_cast<uint32_t>(rocjitsu::ConSanMoiInlineTokenEvidenceKind::Inherited)
              ? "inherited"
              : "release-sequence",
          token.consumer_owner_id, token.producer_owner_id, token.producer_epoch_plus_one,
          token.workgroup_key, static_cast<unsigned long long>(token.dispatch_id),
          static_cast<unsigned long long>(token.source_release_address),
          token.source_release_version);
    }

    const auto *records = reinterpret_cast<const rocjitsu::ConSanMoiAccessRecord *>(
        static_cast<const uint8_t *>(report_ptr) + sizeof(rocjitsu::ConSanMoiReportHeader));
    const auto *barriers = reinterpret_cast<const rocjitsu::ConSanMoiBarrierRecord *>(
        static_cast<const uint8_t *>(report_ptr) + sizeof(rocjitsu::ConSanMoiReportHeader) +
        static_cast<size_t>(header->access_record_capacity) *
            sizeof(rocjitsu::ConSanMoiAccessRecord));
    const auto *atomics = reinterpret_cast<const rocjitsu::ConSanMoiAtomicRecord *>(
        static_cast<const uint8_t *>(report_ptr) + sizeof(rocjitsu::ConSanMoiReportHeader) +
        static_cast<size_t>(header->access_record_capacity) *
            sizeof(rocjitsu::ConSanMoiAccessRecord) +
        static_cast<size_t>(header->barrier_record_capacity) *
            sizeof(rocjitsu::ConSanMoiBarrierRecord));
    const auto *fences = reinterpret_cast<const rocjitsu::ConSanMoiFenceRecord *>(
        static_cast<const uint8_t *>(report_ptr) + sizeof(rocjitsu::ConSanMoiReportHeader) +
        static_cast<size_t>(header->access_record_capacity) *
            sizeof(rocjitsu::ConSanMoiAccessRecord) +
        static_cast<size_t>(header->barrier_record_capacity) *
            sizeof(rocjitsu::ConSanMoiBarrierRecord) +
        static_cast<size_t>(header->atomic_record_capacity) *
            sizeof(rocjitsu::ConSanMoiAtomicRecord));
    const auto *diagnostics = reinterpret_cast<const rocjitsu::ConSanMoiDiagnosticRecord *>(
        static_cast<const uint8_t *>(report_ptr) + sizeof(rocjitsu::ConSanMoiReportHeader) +
        static_cast<size_t>(header->access_record_capacity) *
            sizeof(rocjitsu::ConSanMoiAccessRecord) +
        static_cast<size_t>(header->barrier_record_capacity) *
            sizeof(rocjitsu::ConSanMoiBarrierRecord) +
        static_cast<size_t>(header->atomic_record_capacity) *
            sizeof(rocjitsu::ConSanMoiAtomicRecord) +
        static_cast<size_t>(entry.fence_record_capacity) * sizeof(rocjitsu::ConSanMoiFenceRecord));
    std::vector<rocjitsu::ConSanMoiDiagnosticRecord> resolved_diagnostics;
    if (entry.compact_token_mapping_malformed)
      ++summary.inline_malformed_count;
    if (visible_diagnostics != 0u && entry.compact_token_mapping_count != 0u) {
      resolved_diagnostics.assign(diagnostics, diagnostics + visible_diagnostics);
      const auto *mappings =
          reinterpret_cast<const rocjitsu::ConSanMoiCompactDiagnosticTokenMapping *>(
              static_cast<const uint8_t *>(report_ptr) +
              entry.layout.inline_compact_token_mappings_offset);
      for (rocjitsu::ConSanMoiDiagnosticRecord &diagnostic : resolved_diagnostics) {
        const uint32_t tagged = diagnostic.first_instruction_offset;
        constexpr uint32_t kTokenPayloadMask = rocjitsu::consan_moi_exact_shadow::max_compact_token;
        constexpr uint32_t kAllowedBits =
            rocjitsu::consan_moi_exact_shadow::compact_diagnostic_token_tag | kTokenPayloadMask;
        if ((tagged & rocjitsu::consan_moi_exact_shadow::compact_diagnostic_token_tag) == 0u)
          continue;
        const uint16_t token = static_cast<uint16_t>(tagged & kTokenPayloadMask);
        const bool well_formed = token != 0u && (tagged & ~kAllowedBits) == 0u;
        const rocjitsu::ConSanMoiCompactDiagnosticTokenMapping *current = nullptr;
        const rocjitsu::ConSanMoiCompactDiagnosticTokenMapping *resolved = nullptr;
        bool current_ambiguous = false;
        bool prior_ambiguous = false;
        if (well_formed) {
          for (uint32_t i = 0; i < entry.compact_token_mapping_count; ++i) {
            const auto &mapping = mappings[i];
            if (mapping.instruction_offset != diagnostic.second_instruction_offset)
              continue;
            if (current != nullptr &&
                current->owner_descriptor_file_offset != mapping.owner_descriptor_file_offset) {
              current_ambiguous = true;
              break;
            }
            current = &mapping;
          }
          if (current != nullptr && !current_ambiguous) {
            for (uint32_t i = 0; i < entry.compact_token_mapping_count; ++i) {
              const auto &mapping = mappings[i];
              if (mapping.owner_descriptor_file_offset != current->owner_descriptor_file_offset ||
                  mapping.token != token) {
                continue;
              }
              if (resolved != nullptr &&
                  resolved->instruction_offset != mapping.instruction_offset) {
                prior_ambiguous = true;
                break;
              }
              resolved = &mapping;
            }
          }
        }
        if (!well_formed || current == nullptr || current_ambiguous || resolved == nullptr ||
            prior_ambiguous) {
          ++summary.inline_malformed_count;
          log_message(kLogInfo,
                      "ConSan MOI compact diagnostic token unresolved reader=%llu current=0x%x "
                      "tagged=0x%x well_formed=%s current_ambiguous=%s "
                      "prior_ambiguous=%s",
                      static_cast<unsigned long long>(entry.reader),
                      diagnostic.second_instruction_offset, tagged, well_formed ? "true" : "false",
                      current_ambiguous ? "true" : "false", prior_ambiguous ? "true" : "false");
          continue;
        }
        diagnostic.first_instruction_offset = resolved->instruction_offset;
      }
      diagnostics = resolved_diagnostics.data();
    }
    for (uint32_t i = 0; i < visible_fences; ++i) {
      const rocjitsu::ConSanMoiFenceRecord &fence = fences[i];
      log_message(kLogInfo,
                  "ConSan MOI auto fence reader=%llu index=%u event_index=%u owner=%u "
                  "generation=%llu epoch=%u workgroup=(%u,%u,%u) inst=0x%x kind=%u scope=%u "
                  "semantics=%u token=0x%016llx",
                  static_cast<unsigned long long>(entry.reader), i, fence.event_index,
                  fence.owner_id, static_cast<unsigned long long>(fence.generation), fence.epoch,
                  fence.workgroup_x, fence.workgroup_y, fence.workgroup_z, fence.instruction_offset,
                  fence.kind, fence.scope, fence.semantics,
                  static_cast<unsigned long long>(fence.communication_token));
    }
    if (visible_records != 0 || visible_barriers != 0 || visible_atomics != 0 ||
        visible_fences != 0) {
      uint64_t required_shadow_entries = 0;
      for (uint32_t i = 0; i < visible_records; ++i) {
        const rocjitsu::ConSanMoiAccessRecord &record = records[i];
        uint64_t record_end = static_cast<uint64_t>(record.start_cell) + record.cell_count;
        if (record_end == 0 && record.lds_byte_count != 0) {
          const rocjitsu::ConSanMoiLdsCellRange range =
              rocjitsu::consan_moi_lds_cell_range_for_bytes(record.lds_byte_offset,
                                                            record.lds_byte_count);
          record_end = static_cast<uint64_t>(range.start_cell) + range.cell_count;
        }
        required_shadow_entries = std::max(required_shadow_entries, record_end);
      }
      const uint64_t kMaxAutoReplayShadowEntries = 1u << 20u;
      if (required_shadow_entries > kMaxAutoReplayShadowEntries) {
        log_message(kLogInfo,
                    "ConSan MOI auto replay reader=%llu skipped required_shadow_entries=%llu "
                    "limit=%llu",
                    static_cast<unsigned long long>(entry.reader),
                    static_cast<unsigned long long>(required_shadow_entries),
                    static_cast<unsigned long long>(kMaxAutoReplayShadowEntries));
      } else {
        rocjitsu::ConSanMoiReportHeader replay_header = *header;
        replay_header.diagnostic_count = 0;
        replay_header.diagnostic_capacity = 4;
        std::vector<rocjitsu::ConSanMoiDiagnosticRecord> diagnostics(
            replay_header.diagnostic_capacity);
        std::vector<uint64_t> exact_shadow_entries(
            static_cast<size_t>(std::max<uint64_t>(required_shadow_entries, 1u)));
        const rocjitsu::ConSanMoiRecordReplayResult replay =
            rocjitsu::consan_moi_record_replay_access_records(
                replay_header,
                std::span<const rocjitsu::ConSanMoiAccessRecord>(records, visible_records),
                std::span<const rocjitsu::ConSanMoiBarrierRecord>(barriers, visible_barriers),
                std::span<const rocjitsu::ConSanMoiRecordReplayAtomicEvent>(atomics,
                                                                            visible_atomics),
                std::span<const rocjitsu::ConSanMoiRecordReplayFenceEvent>(fences, visible_fences),
                diagnostics, exact_shadow_entries);
        const uint32_t replay_visible_diagnostics =
            std::min<uint32_t>(replay.emitted_diagnostic_count, diagnostics.size());
        const rocjitsu::ConSanMoiReplayProvenanceRepair provenance =
            rocjitsu::repair_consan_moi_record_replay_provenance(
                std::span<const rocjitsu::ConSanMoiAccessRecord>(records, visible_records),
                std::span<rocjitsu::ConSanMoiDiagnosticRecord>(diagnostics.data(),
                                                               replay_visible_diagnostics));
        summary.replay_conflict_count = replay.conflict ? 1u : 0u;
        summary.replay_diagnostic_count = replay.emitted_diagnostic_count;
        summary.replay_dropped_access_count = replay.dropped_access_count;
        summary.replay_dropped_barrier_count = replay.dropped_barrier_count;
        summary.replay_unsupported_access_count = replay.unsupported_access_count;
        summary.replay_unsupported_atomic_count = replay.unsupported_atomic_count;
        summary.replay_unsupported_fence_count = replay.unsupported_fence_count;
        summary.replay_metadata_full_count = replay.metadata_full ? 1u : 0u;
        summary.replay_diagnostic_capacity_exhausted_count =
            replay.diagnostic_capacity_exhausted ? 1u : 0u;
        log_message(kLogInfo,
                    "ConSan MOI auto replay reader=%llu processed_access=%u processed_barriers=%u "
                    "processed_atomics=%u processed_fences=%u dropped_access=%u "
                    "dropped_barriers=%u unsupported_access=%u unsupported_atomics=%u "
                    "unsupported_fences=%u diagnostics=%u "
                    "conflict=%s metadata_full=%s diagnostic_capacity_exhausted=%s "
                    "provenance_repaired=%u provenance_unresolved=%u "
                    "shadow_entries=%zu",
                    static_cast<unsigned long long>(entry.reader), replay.processed_access_count,
                    replay.processed_barrier_count, replay.processed_atomic_count,
                    replay.processed_fence_count, replay.dropped_access_count,
                    replay.dropped_barrier_count, replay.unsupported_access_count,
                    replay.unsupported_atomic_count, replay.unsupported_fence_count,
                    replay.emitted_diagnostic_count, replay.conflict ? "true" : "false",
                    replay.metadata_full ? "true" : "false",
                    replay.diagnostic_capacity_exhausted ? "true" : "false",
                    provenance.repaired_diagnostic_count, provenance.unresolved_diagnostic_count,
                    exact_shadow_entries.size());
        for (uint32_t i = 0; i < replay_visible_diagnostics; ++i) {
          const rocjitsu::ConSanMoiDiagnosticRecord &diagnostic = diagnostics[i];
          log_message(kLogInfo,
                      "ConSan MOI auto replay diagnostic reader=%llu index=%u kind=%u "
                      "generation=%llu "
                      "epoch=%u first_owner=%u second_owner=%u first_inst=0x%x "
                      "second_inst=0x%x first_lds_known=%s first_lds=[%u,%u) "
                      "second_lds=[%u,%u) first_kind=%u second_kind=%u "
                      "first_lane_mask=0x%llx second_lane_mask=0x%llx",
                      static_cast<unsigned long long>(entry.reader), i, diagnostic.kind,
                      static_cast<unsigned long long>(diagnostic.generation), diagnostic.epoch,
                      diagnostic.first_owner_id, diagnostic.second_owner_id,
                      diagnostic.first_instruction_offset, diagnostic.second_instruction_offset,
                      diagnostic.first_lds_byte_count != 0 ? "true" : "false",
                      diagnostic.first_lds_byte_offset,
                      diagnostic.first_lds_byte_offset + diagnostic.first_lds_byte_count,
                      diagnostic.second_lds_byte_offset,
                      diagnostic.second_lds_byte_offset + diagnostic.second_lds_byte_count,
                      diagnostic.first_access_kind, diagnostic.second_access_kind,
                      static_cast<unsigned long long>(diagnostic.first_lane_mask),
                      static_cast<unsigned long long>(diagnostic.second_lane_mask));
        }
      }
    }

    const uint32_t sample_count = std::min<uint32_t>(visible_records, 4u);
    for (uint32_t i = 0; i < sample_count; ++i) {
      const rocjitsu::ConSanMoiAccessRecord &record = records[i];
      log_message(
          kLogInfo,
          "ConSan MOI auto record reader=%llu index=%u event_index=%u kind=%u wave=%u "
          "generation=%llu epoch=%u workgroup=(%u,%u,%u) inst=0x%x lds_offset=%u "
          "lds_bytes=%u "
          "cells=[%u,%u) "
          "lane_mask=0x%llx",
          static_cast<unsigned long long>(entry.reader), i, record.event_index, record.access_kind,
          record.wave_id, static_cast<unsigned long long>(record.generation), record.epoch,
          record.workgroup_x, record.workgroup_y, record.workgroup_z, record.instruction_offset,
          record.lds_byte_offset, record.lds_byte_count, record.start_cell,
          record.start_cell + record.cell_count, static_cast<unsigned long long>(record.lane_mask));
    }

    const uint32_t barrier_sample_count = std::min<uint32_t>(visible_barriers, 4u);
    for (uint32_t i = 0; i < barrier_sample_count; ++i) {
      const rocjitsu::ConSanMoiBarrierRecord &record = barriers[i];
      log_message(kLogInfo,
                  "ConSan MOI auto barrier reader=%llu index=%u event_index=%u wave=%u "
                  "inst=0x%x lane_mask=0x%llx",
                  static_cast<unsigned long long>(entry.reader), i, record.event_index,
                  record.wave_id, record.instruction_offset,
                  static_cast<unsigned long long>(record.lane_mask));
    }

    const uint32_t atomic_sample_count = std::min<uint32_t>(visible_atomics, 4u);
    for (uint32_t i = 0; i < atomic_sample_count; ++i) {
      const rocjitsu::ConSanMoiAtomicRecord &record = atomics[i];
      log_message(kLogInfo,
                  "ConSan MOI auto atomic reader=%llu index=%u event_index=%u kind=%u owner=%u "
                  "generation=%llu epoch=%u workgroup=(%u,%u,%u) inst=0x%x address=0x%llx "
                  "scope=%u semantics=%u",
                  static_cast<unsigned long long>(entry.reader), i, record.event_index,
                  static_cast<uint32_t>(record.kind), record.owner_id,
                  static_cast<unsigned long long>(record.generation), record.epoch,
                  record.workgroup_x, record.workgroup_y, record.workgroup_z,
                  record.instruction_offset, static_cast<unsigned long long>(record.atomic_address),
                  record.scope, record.semantics);
    }
    const uint32_t diagnostic_sample_count = std::min<uint32_t>(visible_diagnostics, 4u);
    for (uint32_t i = 0; i < diagnostic_sample_count; ++i) {
      const rocjitsu::ConSanMoiDiagnosticRecord &record = diagnostics[i];
      log_message(
          kLogInfo,
          "ConSan MOI auto diagnostic reader=%llu index=%u backend=%u kind=%u "
          "generation=%llu epoch=%u first_owner=%u second_owner=%u first_inst=0x%x "
          "second_inst=0x%x first_kind=%u second_kind=%u first_lanes=0x%llx "
          "second_lanes=0x%llx first_lds=[%u,%u) second_lds=[%u,%u)",
          static_cast<unsigned long long>(entry.reader), i, record.backend, record.kind,
          static_cast<unsigned long long>(record.generation), record.epoch, record.first_owner_id,
          record.second_owner_id, record.first_instruction_offset, record.second_instruction_offset,
          record.first_access_kind, record.second_access_kind,
          static_cast<unsigned long long>(record.first_lane_mask),
          static_cast<unsigned long long>(record.second_lane_mask), record.first_lds_byte_offset,
          record.first_lds_byte_offset + record.first_lds_byte_count, record.second_lds_byte_offset,
          record.second_lds_byte_offset + record.second_lds_byte_count);
    }
    for (uint32_t i = 0; i < std::min<size_t>(visible_exact_shadow.size(), 4u); ++i) {
      const ExactShadowEntry &shadow_entry = visible_exact_shadow[i];
      log_message(kLogInfo,
                  "ConSan MOI auto exact-shadow reader=%llu index=%u kind=%u owner=%u epoch=%u "
                  "generation=%u inst=0x%x dispatch=0x%llx version=%u",
                  static_cast<unsigned long long>(entry.reader), shadow_entry.index,
                  static_cast<uint32_t>(shadow_entry.entry.kind), shadow_entry.entry.owner_id,
                  shadow_entry.entry.epoch, shadow_entry.entry.generation,
                  shadow_entry.entry.instruction_offset,
                  static_cast<unsigned long long>(shadow_entry.dispatch_id), shadow_entry.version);
    }
    for (uint32_t i = 0; i < std::min<size_t>(visible_sampled.size(), 4u); ++i) {
      const SampledEntry &sampled_entry = visible_sampled[i];
      log_message(kLogInfo,
                  "ConSan MOI auto sampled reader=%llu index=%u kind=%u owner=%u epoch=%u "
                  "generation=%u cells=[%u,%u) consumed=%s",
                  static_cast<unsigned long long>(entry.reader), sampled_entry.index,
                  static_cast<uint32_t>(sampled_entry.entry.kind), sampled_entry.entry.owner_id,
                  sampled_entry.entry.epoch, sampled_entry.entry.generation,
                  sampled_entry.entry.start_cell,
                  sampled_entry.entry.start_cell + sampled_entry.entry.cell_count,
                  sampled_entry.entry.consumed ? "true" : "false");
    }
    if (first_sampled_conflict) {
      const SampledEntry &first = first_sampled_conflict->first;
      const SampledEntry &second = first_sampled_conflict->second;
      const uint32_t first_end = first.entry.start_cell + first.entry.cell_count;
      const uint32_t second_end = second.entry.start_cell + second.entry.cell_count;
      log_message(kLogInfo,
                  "ConSan MOI auto sampled conflict reader=%llu first_index=%u second_index=%u "
                  "first_kind=%u second_kind=%u first_owner=%u second_owner=%u epoch=%u "
                  "generation=%u first_cells=[%u,%u) second_cells=[%u,%u)",
                  static_cast<unsigned long long>(entry.reader), first.index, second.index,
                  static_cast<uint32_t>(first.entry.kind), static_cast<uint32_t>(second.entry.kind),
                  first.entry.owner_id, second.entry.owner_id, second.entry.epoch,
                  second.entry.generation, first.entry.start_cell, first_end,
                  second.entry.start_cell, second_end);
    }
    return summary;
  }

  mutable std::mutex mutex_;
  std::array<Entry, 256> entries_{};
  size_t entry_count_ = 0;
  size_t reserved_entry_count_ = 0;
  uint64_t required_report_bytes_ = 0;
  uint64_t successful_allocated_bytes_ = 0;
  rocjitsu::ConSanMoiAutoReportProcessBudget process_budget_;
  uint64_t allocation_failure_count_ = 0;
  uint64_t capacity_failure_count_ = 0;
  uint64_t cleanup_failure_count_ = 0;
  std::atomic<uint64_t> next_generation_{0};
};

void reject_auto_moi_report_plan(uint64_t reader, uint64_t required_size, uint64_t configured_cap,
                                 std::string_view reason) {
  AutoMoiReportBufferRegistry::instance().reject_plan(reader, required_size, configured_cap,
                                                      reason);
}

bool allocate_auto_moi_report_buffer(CoreApiTable *core, hsa_agent_t agent, uint64_t reader,
                                     uint64_t required_size, uint64_t requested_size,
                                     uint64_t configured_cap,
                                     const ConSanMoiReportBufferLayout &layout,
                                     ConSanMoiEngine engine, bool track_barriers,
                                     bool track_atomics, bool test_seed_inline_exact_odd,
                                     uint64_t *address, uint64_t *registered_size,
                                     uint64_t *registered_generation) {
  return AutoMoiReportBufferRegistry::instance().allocate(
      core, agent, reader, required_size, requested_size, configured_cap, layout, engine,
      track_barriers, track_atomics, test_seed_inline_exact_odd, address, registered_size,
      registered_generation);
}

void register_auto_moi_report_compact_tokens(uint64_t reader, const ConSanResult &result) {
  AutoMoiReportBufferRegistry::instance().register_compact_tokens(reader, result);
}

AutoMoiReportSummary summarize_and_clear_auto_moi_report_buffers(CoreApiTable *core) {
  return AutoMoiReportBufferRegistry::instance().summarize_and_clear(core);
}

} // namespace rocjitsu::consan_hook
