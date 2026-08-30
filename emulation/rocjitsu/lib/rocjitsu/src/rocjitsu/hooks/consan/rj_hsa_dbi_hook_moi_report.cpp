// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "hsa/hsa_api_trace_minimal.h"

#include "rocjitsu/checked_byte_budget.h"
#include "rocjitsu/code/patch/consan/consan_moi_report_contract.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_hook_internal.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_moi_report_pipeline.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_moi_report_snapshot.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_report_registry_lifecycle.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <vector>

namespace rocjitsu::consan_hook {

namespace {

bool copy_coarse_report_snapshot(void *context, void *destination, const void *source, size_t size,
                                 int32_t *status) {
  auto *core = static_cast<CoreApiTable *>(context);
  if (core == nullptr || core->hsa_memory_copy_fn == nullptr)
    return false;
  const hsa_status_t copy_status = core->hsa_memory_copy_fn(destination, source, size);
  if (status != nullptr)
    *status = static_cast<int32_t>(copy_status);
  return copy_status == HSA_STATUS_SUCCESS;
}

} // namespace

class AutoMoiReportBufferRegistry {
public:
  static AutoMoiReportBufferRegistry &instance() {
    // The HSA runtime may call OnUnload from a shared-library finalizer after
    // ordinary function-local statics have already been destroyed. Keep the
    // registry alive for the process lifetime and clear its contents explicitly
    // when the hook layer is uninstalled.
    static auto *registry = new AutoMoiReportBufferRegistry;
    return *registry;
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

  [[nodiscard]] bool allocate(CoreApiTable *core, hsa_agent_t agent, uint64_t reader,
                              uint64_t required_size, uint64_t requested_size,
                              uint64_t configured_cap,
                              const rocjitsu::ConSanMoiReportBufferLayout &layout,
                              bool track_barriers, bool track_atomics,
                              bool test_seed_inline_exact_odd, uint64_t *address,
                              uint64_t *registered_size, uint64_t *registered_generation) {
    record_allocation_attempt(required_size);
    const bool direct_sampled = layout.engine == rocjitsu::ConSanMoiEngine::Sampled;
    const bool inline_shadow = layout.engine == rocjitsu::ConSanMoiEngine::InlineShadow;
    const uint64_t engine_ceiling =
        rocjitsu::consan_moi_auto_report_buffer_ceiling_bytes(layout.engine);
    if (required_size > configured_cap || requested_size > configured_cap ||
        requested_size > engine_ceiling) {
      record_allocation_failure(required_size, /*capacity_failure=*/true);
      log_message(kLogInfo,
                  "ConSan MOI auto report allocation reader=%llu outcome="
                  "insufficient_report_capacity required_bytes=%llu cap_bytes=%llu "
                  "per_buffer_ceiling=%llu",
                  static_cast<unsigned long long>(reader),
                  static_cast<unsigned long long>(required_size),
                  static_cast<unsigned long long>(configured_cap),
                  static_cast<unsigned long long>(engine_ceiling));
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

    detail::AutoReportRegionSearch search{.core = core, .requested_size = requested};
    const hsa_status_t iterate_status =
        core->hsa_agent_iterate_regions_fn(agent, detail::select_auto_report_region, &search);
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
                                                                 layout);
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
                                       search.fine_grained,
                                       {},
                                       {},
                                       {}};
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
        "dispatch_token_capacity=%u dispatch_banks=%u owner_banks=%u "
        "address_group_headroom=%u "
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
        layout.record_replay_dispatch_token_capacity,
        layout.record_replay_access_dispatch_bank_count,
        layout.record_replay_access_owner_bank_count, layout.record_replay_address_group_headroom,
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

  void register_metadata(uint64_t reader, uint64_t generation, std::string_view input_fingerprint,
                         const rocjitsu::ConSanRuntimeStaticMapping &static_mapping) {
    std::lock_guard lock(mutex_);
    auto entry = std::find_if(entries_.begin(), entries_.begin() + entry_count_,
                              [reader, generation](const Entry &item) {
                                return item.reader == reader && item.generation == generation;
                              });
    if (entry == entries_.begin() + entry_count_)
      return;

    entry->input_fingerprint = input_fingerprint;
    entry->record_replay_static_mappings.clear();
    entry->record_replay_static_mapping_malformed = false;
    entry->sampled_static_mappings.clear();
    entry->sampled_static_mapping_malformed = false;
    entry->compact_token_mapping_count = 0;
    entry->compact_token_mapping_malformed = false;
    auto *mappings = reinterpret_cast<rocjitsu::ConSanMoiCompactDiagnosticTokenMapping *>(
        static_cast<uint8_t *>(entry->ptr) + entry->layout.inline_compact_token_mappings_offset);
    for (const rocjitsu::ConSanRecordReplayStaticAccessMapping &static_access :
         static_mapping.record_replay_accesses) {
      if (static_access.access.owner_provenance_complete &&
          static_access.access.execution_owner_descriptor_file_offsets.empty()) {
        entry->record_replay_static_mapping_malformed = true;
      }
      if (static_access.access.original_site.original_text_offset >
          std::numeric_limits<uint32_t>::max()) {
        entry->record_replay_static_mapping_malformed = true;
        continue;
      }
      const uint32_t instruction_offset =
          static_cast<uint32_t>(static_access.access.original_site.original_text_offset);
      auto mapping = std::ranges::find_if(
          entry->record_replay_static_mappings,
          [instruction_offset](const Entry::RecordReplayStaticMapping &candidate) {
            return candidate.instruction_offset == instruction_offset;
          });
      if (mapping == entry->record_replay_static_mappings.end()) {
        entry->record_replay_static_mappings.push_back({
            .instruction_offset = instruction_offset,
            .owner_descriptor_file_offsets =
                static_access.access.execution_owner_descriptor_file_offsets,
            .owner_provenance_complete =
                static_access.access.owner_provenance_complete &&
                !static_access.access.execution_owner_descriptor_file_offsets.empty(),
        });
      } else {
        mapping->owner_provenance_complete &= static_access.access.owner_provenance_complete;
        for (uint64_t owner : static_access.access.execution_owner_descriptor_file_offsets) {
          if (std::ranges::find(mapping->owner_descriptor_file_offsets, owner) ==
              mapping->owner_descriptor_file_offsets.end()) {
            mapping->owner_descriptor_file_offsets.push_back(owner);
          }
        }
      }
    }
    for (const rocjitsu::ConSanSampledStaticAccessMapping &static_access :
         static_mapping.sampled_accesses) {
      if (static_access.access.owner_provenance_complete &&
          static_access.access.execution_owner_descriptor_file_offsets.empty()) {
        entry->sampled_static_mapping_malformed = true;
      }
      const uint64_t slot_count =
          static_cast<uint64_t>(static_access.range_count) * static_access.bank_count;
      if (static_access.range_count != 0u && static_access.bank_count != 0u &&
          static_access.first_slot <= entry->layout.sampled_watchpoint_capacity &&
          slot_count <= entry->layout.sampled_watchpoint_capacity - static_access.first_slot) {
        entry->sampled_static_mappings.push_back({
            .first_slot = static_access.first_slot,
            .range_count = static_access.range_count,
            .bank_count = static_access.bank_count,
            .instruction_offset = static_access.access.original_site.original_text_offset,
            .emitted_probe_offset = static_access.emitted_probe_text_offset,
            .relocated_guest_offset = static_access.relocated_guest_text_offset.value_or(0u),
            .scratch_vgpr = static_access.scratch_vgpr,
            .owner_descriptor_file_offsets =
                static_access.access.execution_owner_descriptor_file_offsets,
            .owner_provenance_complete =
                static_access.access.owner_provenance_complete &&
                !static_access.access.execution_owner_descriptor_file_offsets.empty(),
        });
      } else {
        entry->sampled_static_mapping_malformed = true;
      }
    }
    for (const rocjitsu::ConSanInlineCompactStaticAccessMapping &static_access :
         static_mapping.inline_compact_accesses) {
      if (static_access.token == 0u ||
          static_access.access.original_site.original_text_offset >
              rocjitsu::consan_moi_exact_shadow::max_instruction_offset ||
          !static_access.access.owner_provenance_complete ||
          static_access.access.execution_owner_descriptor_file_offsets.size() != 1u ||
          entry->compact_token_mapping_count >=
              entry->layout.inline_compact_token_mapping_capacity) {
        entry->compact_token_mapping_malformed = true;
        continue;
      }
      mappings[entry->compact_token_mapping_count++] =
          rocjitsu::ConSanMoiCompactDiagnosticTokenMapping{
              .owner_descriptor_file_offset =
                  static_access.access.execution_owner_descriptor_file_offsets.front(),
              .instruction_offset =
                  static_cast<uint32_t>(static_access.access.original_site.original_text_offset),
              .token = static_access.token,
          };
    }
    if (!static_mapping.record_replay_accesses.empty()) {
      log_message(kLogInfo,
                  "ConSan MOI record-replay diagnostic map reader=%llu entries=%zu mappings=%zu "
                  "malformed=%s",
                  static_cast<unsigned long long>(reader),
                  static_mapping.record_replay_accesses.size(),
                  entry->record_replay_static_mappings.size(),
                  entry->record_replay_static_mapping_malformed ? "true" : "false");
    }
    if (!static_mapping.sampled_accesses.empty()) {
      log_message(kLogInfo,
                  "ConSan MOI sampled diagnostic map reader=%llu entries=%zu mappings=%zu "
                  "capacity=%u malformed=%s",
                  static_cast<unsigned long long>(reader), static_mapping.sampled_accesses.size(),
                  entry->sampled_static_mappings.size(), entry->layout.sampled_watchpoint_capacity,
                  entry->sampled_static_mapping_malformed ? "true" : "false");
    }
    if (!static_mapping.inline_compact_accesses.empty()) {
      log_message(kLogInfo,
                  "ConSan MOI compact diagnostic map reader=%llu entries=%zu mappings=%u "
                  "capacity=%u malformed=%s",
                  static_cast<unsigned long long>(reader),
                  static_mapping.inline_compact_accesses.size(), entry->compact_token_mapping_count,
                  entry->layout.inline_compact_token_mapping_capacity,
                  entry->compact_token_mapping_malformed ? "true" : "false");
    }
  }

  void bind_to_executable(uint64_t reader, uint64_t generation, hsa_executable_t executable) {
    std::lock_guard lock(mutex_);
    detail::bind_auto_report_entry(entries_, entry_count_, reader, generation, executable.handle);
  }

  void discard(CoreApiTable *core, uint64_t reader, uint64_t generation) {
    std::lock_guard lock(mutex_);
    detail::discard_auto_report_entry(
        entries_, entry_count_, reader, generation, [&](Entry &entry) {
          return release_entry(core, entry, /*allow_runtime_reclaimed=*/false);
        });
  }

  void retire(CoreApiTable *core, hsa_executable_t executable) {
    std::lock_guard lock(mutex_);
    detail::retire_auto_report_entries(
        entries_, entry_count_, executable.handle, retired_summary_,
        [&](const Entry &entry) { return summarize(core, entry); },
        [&](Entry &entry) { return release_entry(core, entry, /*allow_runtime_reclaimed=*/false); },
        accumulate_summary);
  }

  Summary summarize_and_clear(CoreApiTable *core) {
    std::lock_guard lock(mutex_);
    Summary total = retired_summary_;
    total.required_report_bytes = required_report_bytes_;
    total.allocated_report_bytes = successful_allocated_bytes_;
    total.current_live_report_bytes = process_budget_.current_live_bytes;
    total.peak_live_report_bytes = process_budget_.peak_live_bytes;
    total.allocation_failure_count = allocation_failure_count_;
    total.capacity_failure_count = capacity_failure_count_;
    for (size_t i = 0; i < entry_count_; ++i) {
      const Summary entry_summary = summarize(core, entries_[i]);
      accumulate_summary(total, entry_summary);
      if (!release_entry(core, entries_[i], /*allow_runtime_reclaimed=*/true))
        ++cleanup_failure_count_;
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
    retired_summary_ = {};
    process_budget_.peak_live_bytes = process_budget_.current_live_bytes;
    return total;
  }

private:
  struct Entry {
    using RecordReplayStaticMapping = AutoMoiRecordReplayStaticMapping;
    using SampledStaticMapping = AutoMoiSampledStaticMapping;

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
    std::string input_fingerprint;
    std::vector<RecordReplayStaticMapping> record_replay_static_mappings;
    std::vector<SampledStaticMapping> sampled_static_mappings;
    uint32_t compact_token_mapping_count = 0;
    bool compact_token_mapping_malformed = false;
    bool record_replay_static_mapping_malformed = false;
    bool sampled_static_mapping_malformed = false;
    uint64_t executable = 0;
    bool executable_bound = false;
  };

  [[nodiscard]] bool release_entry(CoreApiTable *core, Entry &entry, bool allow_runtime_reclaimed) {
    bool freed = entry.ptr == nullptr;
    hsa_status_t free_status = HSA_STATUS_SUCCESS;
    if (!freed && allow_runtime_reclaimed) {
      // OnUnload runs from inside ROCR shutdown. Even when the API-table entry
      // remains non-null, re-entering hsa_memory_free can deadlock against the
      // runtime's shutdown locks. ROCR owns and reclaims these allocations as
      // shutdown continues after OnUnload returns.
      freed = true;
      log_message(kLogInfo,
                  "ConSan MOI auto report cleanup reader=%llu bytes=%zu "
                  "outcome=runtime-reclaimed",
                  static_cast<unsigned long long>(entry.reader), entry.size);
    } else if (!freed && (core == nullptr || core->hsa_memory_free_fn == nullptr)) {
      // Non-shutdown cleanup requires a callable free API; keep the entry live
      // so a later executable-destroy or shutdown path can reclaim it.
    } else if (!freed) {
      free_status = core->hsa_memory_free_fn(entry.ptr);
      freed = free_status == HSA_STATUS_SUCCESS ||
              free_status == HSA_STATUS_ERROR_INVALID_ALLOCATION ||
              free_status == HSA_STATUS_ERROR_NOT_INITIALIZED;
      if (free_status == HSA_STATUS_ERROR_INVALID_ALLOCATION ||
          free_status == HSA_STATUS_ERROR_NOT_INITIALIZED) {
        log_message(kLogInfo,
                    "ConSan MOI auto report cleanup reader=%llu bytes=%zu "
                    "outcome=runtime-reclaimed status=%d",
                    static_cast<unsigned long long>(entry.reader), entry.size,
                    static_cast<int>(free_status));
      }
    }
    if (freed) {
      (void)rocjitsu::release_consan_moi_auto_report_bytes(process_budget_, entry.size);
      return true;
    }
    log_message(
        kLogInfo, "ConSan MOI auto report cleanup reader=%llu bytes=%zu outcome=failed status=%d",
        static_cast<unsigned long long>(entry.reader), entry.size, static_cast<int>(free_status));
    return false;
  }

  static void accumulate_summary(Summary &total, const Summary &entry) {
    total.buffer_count += entry.buffer_count;
    total.fine_grained_snapshot_bytes += entry.fine_grained_snapshot_bytes;
    total.coarse_grained_snapshot_bytes += entry.coarse_grained_snapshot_bytes;
    total.visible_access_record_count += entry.visible_access_record_count;
    total.visible_barrier_record_count += entry.visible_barrier_record_count;
    total.visible_atomic_record_count += entry.visible_atomic_record_count;
    total.visible_fence_record_count += entry.visible_fence_record_count;
    total.visible_diagnostic_record_count += entry.visible_diagnostic_record_count;
    total.visible_inline_publication_count += entry.visible_inline_publication_count;
    total.visible_exact_shadow_entry_count += entry.visible_exact_shadow_entry_count;
    total.exact_incomplete_snapshot_count += entry.exact_incomplete_snapshot_count;
    total.exact_changed_snapshot_count += entry.exact_changed_snapshot_count;
    total.exact_malformed_snapshot_count += entry.exact_malformed_snapshot_count;
    total.visible_inline_atomic_release_count += entry.visible_inline_atomic_release_count;
    total.release_incomplete_snapshot_count += entry.release_incomplete_snapshot_count;
    total.release_changed_snapshot_count += entry.release_changed_snapshot_count;
    total.release_overflow_snapshot_count += entry.release_overflow_snapshot_count;
    total.release_source_incomplete_snapshot_count +=
        entry.release_source_incomplete_snapshot_count;
    total.release_malformed_snapshot_count += entry.release_malformed_snapshot_count;
    total.visible_inline_acquired_token_count += entry.visible_inline_acquired_token_count;
    total.token_incomplete_snapshot_count += entry.token_incomplete_snapshot_count;
    total.token_changed_snapshot_count += entry.token_changed_snapshot_count;
    total.token_malformed_snapshot_count += entry.token_malformed_snapshot_count;
    total.inline_undercoverage_count += entry.inline_undercoverage_count;
    total.inline_overflow_count += entry.inline_overflow_count;
    total.inline_unsupported_count += entry.inline_unsupported_count;
    total.inline_malformed_count += entry.inline_malformed_count;
    total.visible_sampled_watchpoint_count += entry.visible_sampled_watchpoint_count;
    total.visible_sampled_sync_metadata_count += entry.visible_sampled_sync_metadata_count;
    total.dropped_access_record_count += entry.dropped_access_record_count;
    total.dropped_barrier_record_count += entry.dropped_barrier_record_count;
    total.dropped_atomic_record_count += entry.dropped_atomic_record_count;
    total.dropped_fence_record_count += entry.dropped_fence_record_count;
    total.dropped_diagnostic_record_count += entry.dropped_diagnostic_record_count;
    total.record_replay_bank_saturation_count += entry.record_replay_bank_saturation_count;
    total.record_replay_invalid_site_token_count += entry.record_replay_invalid_site_token_count;
    total.replay_conflict_count += entry.replay_conflict_count;
    total.replay_diagnostic_count += entry.replay_diagnostic_count;
    total.replay_dropped_access_count += entry.replay_dropped_access_count;
    total.replay_dropped_barrier_count += entry.replay_dropped_barrier_count;
    total.replay_unsupported_access_count += entry.replay_unsupported_access_count;
    total.replay_unsupported_atomic_count += entry.replay_unsupported_atomic_count;
    total.replay_unsupported_fence_count += entry.replay_unsupported_fence_count;
    total.replay_metadata_full_count += entry.replay_metadata_full_count;
    total.replay_diagnostic_capacity_exhausted_count +=
        entry.replay_diagnostic_capacity_exhausted_count;
    total.sampled_conflict_count += entry.sampled_conflict_count;
    total.sampled_immediate_conflict_count += entry.sampled_immediate_conflict_count;
    total.sampled_claimed_window_count += entry.sampled_claimed_window_count;
    total.sampled_dropped_window_count += entry.sampled_dropped_window_count;
    total.sampled_saturated_window_count += entry.sampled_saturated_window_count;
    total.sampled_stale_snapshot_count += entry.sampled_stale_snapshot_count;
    total.sampled_incomplete_snapshot_count += entry.sampled_incomplete_snapshot_count;
    total.sampled_changed_snapshot_count += entry.sampled_changed_snapshot_count;
    total.sampled_malformed_snapshot_count += entry.sampled_malformed_snapshot_count;
    total.sampled_static_mapping_malformed_count += entry.sampled_static_mapping_malformed_count;
    total.sampled_unsupported_sync_count += entry.sampled_unsupported_sync_count;
    total.sampled_malformed_sync_count += entry.sampled_malformed_sync_count;
  }

  void record_allocation_attempt(uint64_t required_size) {
    std::lock_guard lock(mutex_);
    required_report_bytes_ = byte_accounting::saturating_add(required_report_bytes_, required_size);
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

  Summary summarize(CoreApiTable *core, const Entry &entry) {
    Summary summary;
    summary.buffer_count = 1;
    if (entry.compact_token_mapping_malformed)
      ++summary.inline_malformed_count;
    if (entry.sampled_static_mapping_malformed)
      ++summary.sampled_static_mapping_malformed_count;

    const rocjitsu::ConSanMoiEngine expected_engine =
        entry.inline_shadow    ? rocjitsu::ConSanMoiEngine::InlineShadow
        : entry.direct_sampled ? rocjitsu::ConSanMoiEngine::Sampled
                               : rocjitsu::ConSanMoiEngine::RecordReplay;
    const AutoMoiReportSnapshot snapshot = capture_auto_moi_report_snapshot(
        {.source = entry.ptr,
         .size = entry.size,
         .expected_layout = entry.layout,
         .expected_engine = expected_engine,
         .fine_grained = entry.fine_grained},
        core != nullptr && core->hsa_memory_copy_fn != nullptr ? copy_coarse_report_snapshot
                                                               : nullptr,
        core);
    if (!snapshot.complete()) {
      if (snapshot.failure == AutoMoiReportSnapshotFailure::CopyUnavailable) {
        log_message(kLogInfo,
                    "ConSan MOI auto report reader=%llu needs hsa_memory_copy for "
                    "coarse-grained summary",
                    static_cast<unsigned long long>(entry.reader));
      } else if (snapshot.failure == AutoMoiReportSnapshotFailure::CopyFailed) {
        log_message(kLogInfo, "ConSan MOI auto report reader=%llu hsa_memory_copy failed status=%d",
                    static_cast<unsigned long long>(entry.reader), snapshot.copy_status);
      } else {
        log_message(kLogInfo, "ConSan MOI auto report reader=%llu has invalid snapshot source",
                    static_cast<unsigned long long>(entry.reader));
      }
      return summary;
    }
    if (entry.fine_grained)
      summary.fine_grained_snapshot_bytes = snapshot.copied_bytes;
    else
      summary.coarse_grained_snapshot_bytes = snapshot.copied_bytes;

    return summarize_auto_moi_report(
        {.reader = entry.reader,
         .source_address = reinterpret_cast<uint64_t>(entry.ptr),
         .size = entry.size,
         .layout = entry.layout,
         .fence_record_capacity = entry.fence_record_capacity,
         .direct_sampled = entry.direct_sampled,
         .inline_shadow = entry.inline_shadow,
         .fine_grained = entry.fine_grained,
         .input_fingerprint = entry.input_fingerprint,
         .record_replay_static_mappings = entry.record_replay_static_mappings,
         .sampled_static_mappings = entry.sampled_static_mappings,
         .compact_token_mapping_count = entry.compact_token_mapping_count,
         .compact_token_mapping_malformed = entry.compact_token_mapping_malformed,
         .record_replay_static_mapping_malformed = entry.record_replay_static_mapping_malformed,
         .sampled_static_mapping_malformed = entry.sampled_static_mapping_malformed},
        snapshot, summary);
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
  Summary retired_summary_;
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
                                     const ConSanMoiReportBufferLayout &layout, bool track_barriers,
                                     bool track_atomics, bool test_seed_inline_exact_odd,
                                     uint64_t *address, uint64_t *registered_size,
                                     uint64_t *registered_generation) {
  return AutoMoiReportBufferRegistry::instance().allocate(
      core, agent, reader, required_size, requested_size, configured_cap, layout, track_barriers,
      track_atomics, test_seed_inline_exact_odd, address, registered_size, registered_generation);
}

void register_auto_moi_report_metadata(uint64_t reader, uint64_t generation,
                                       std::string_view input_fingerprint,
                                       const ConSanRuntimeStaticMapping &static_mapping) {
  AutoMoiReportBufferRegistry::instance().register_metadata(reader, generation, input_fingerprint,
                                                            static_mapping);
}

void bind_auto_moi_report_buffer_to_executable(uint64_t reader, uint64_t generation,
                                               hsa_executable_t executable) {
  AutoMoiReportBufferRegistry::instance().bind_to_executable(reader, generation, executable);
}

void discard_auto_moi_report_buffer(CoreApiTable *core, uint64_t reader, uint64_t generation) {
  AutoMoiReportBufferRegistry::instance().discard(core, reader, generation);
}

void retire_auto_moi_report_buffers(CoreApiTable *core, hsa_executable_t executable) {
  AutoMoiReportBufferRegistry::instance().retire(core, executable);
}

AutoMoiReportSummary summarize_and_clear_auto_moi_report_buffers(CoreApiTable *core) {
  return AutoMoiReportBufferRegistry::instance().summarize_and_clear(core);
}

} // namespace rocjitsu::consan_hook
