// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "hsa/hsa_api_trace_minimal.h"

#include "rocjitsu/checked_byte_budget.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_hook_internal.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_report_pipeline.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_report_registry_lifecycle.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_report_snapshot.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <vector>

namespace rocjitsu::consan::hook {

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

class AutoReportBufferRegistry {
public:
  static AutoReportBufferRegistry &instance() {
    // The HSA runtime may call OnUnload from a shared-library finalizer after
    // ordinary function-local statics have already been destroyed. Keep the
    // registry alive for the process lifetime and clear its contents explicitly
    // when the hook layer is uninstalled.
    static auto *registry = new AutoReportBufferRegistry;
    return *registry;
  }

  void configure_epoch_analysis(HookConfig::EpochAnalysisPolicy policy, uint32_t conflict_limit,
                                uint32_t total_conflict_limit, bool allow_uniform_lds_stores) {
    std::lock_guard lock(mutex_);
    epoch_analysis_policy_ = policy;
    allow_uniform_lds_stores_ = allow_uniform_lds_stores;
    conflict_limit_ = conflict_limit;
    conflict_examples_remaining_ = total_conflict_limit;
    automatic_epoch_ = 0;
    manual_analysis_window_open_ = false;
  }

  bool advance_generation_for_test(uint64_t generation) {
    std::lock_guard lock(mutex_);
    // Device tests can reach rollover without thousands of compilations.
    // Never rewind identities or change a live/in-flight allocation.
    if (entry_count_ != 0 || reserved_entry_count_ != 0 ||
        generation < next_generation_.load(std::memory_order_relaxed) ||
        generation == std::numeric_limits<uint64_t>::max())
      return false;
    next_generation_.store(generation, std::memory_order_relaxed);
    return true;
  }

  [[nodiscard]] bool begin_epoch_analysis_window() {
    std::lock_guard lock(mutex_);
    if (manual_analysis_window_open_)
      return false;
    manual_analysis_window_open_ = true;
    return true;
  }

  [[nodiscard]] bool end_epoch_analysis_window() {
    std::lock_guard lock(mutex_);
    if (!manual_analysis_window_open_)
      return false;
    manual_analysis_window_open_ = false;
    return true;
  }

  void reject_plan(uint64_t reader, uint64_t required_size, uint64_t configured_cap,
                   std::string_view reason) {
    record_allocation_attempt(required_size);
    record_allocation_failure(required_size, /*capacity_failure=*/true);
    log_message(kLogInfo,
                "ConSan auto report allocation reader=%llu outcome="
                "insufficient_report_capacity reason=%.*s required_bytes=%llu cap_bytes=%llu",
                static_cast<unsigned long long>(reader), static_cast<int>(reason.size()),
                reason.data(), static_cast<unsigned long long>(required_size),
                static_cast<unsigned long long>(configured_cap));
  }

  [[nodiscard]] bool allocate(CoreApiTable *core, hsa_agent_t agent, uint64_t reader,
                              uint64_t required_size, uint64_t requested_size,
                              uint64_t configured_cap, const ReportBufferLayout &layout,
                              bool track_barriers, bool track_atomics, uint64_t *address,
                              uint64_t *registered_size, uint64_t *registered_generation) {
    record_allocation_attempt(required_size);
    const uint64_t mode_ceiling = kOrdinaryAutoReportBufferCeilingBytes;
    if (required_size > configured_cap || requested_size > configured_cap ||
        requested_size > mode_ceiling) {
      record_allocation_failure(required_size, /*capacity_failure=*/true);
      log_message(kLogInfo,
                  "ConSan auto report allocation reader=%llu outcome="
                  "insufficient_report_capacity required_bytes=%llu cap_bytes=%llu "
                  "per_buffer_ceiling=%llu",
                  static_cast<unsigned long long>(reader),
                  static_cast<unsigned long long>(required_size),
                  static_cast<unsigned long long>(configured_cap),
                  static_cast<unsigned long long>(mode_ceiling));
      return false;
    }
    if (!detail::auto_report_allocation_apis_available(core)) {
      record_allocation_failure(required_size, /*capacity_failure=*/false);
      log_message(kLogInfo,
                  "ConSan auto report buffer requested but HSA allocation APIs are unavailable");
      return false;
    }
    if (requested_size > std::numeric_limits<size_t>::max()) {
      record_allocation_failure(required_size, /*capacity_failure=*/true);
      log_message(kLogInfo, "ConSan auto report buffer size is too large: %llu",
                  static_cast<unsigned long long>(requested_size));
      return false;
    }

    const size_t requested = static_cast<size_t>(requested_size);
    if (requested < sizeof(ReportHeader) || !layout.valid || layout.required_bytes > requested) {
      record_allocation_failure(required_size, /*capacity_failure=*/true);
      log_message(kLogInfo,
                  "ConSan auto report buffer is too small reader=%llu bytes=%zu "
                  "mode=%s track_barriers=%s track_atomics=%s",
                  static_cast<unsigned long long>(reader), requested, "default",
                  track_barriers ? "true" : "false", track_atomics ? "true" : "false");
      return false;
    }

    if (!reserve_live_bytes(required_size, requested_size)) {
      log_message(kLogInfo,
                  "ConSan auto report allocation reader=%llu outcome="
                  "insufficient_report_capacity required_bytes=%llu requested_bytes=%llu "
                  "process_ceiling=%llu",
                  static_cast<unsigned long long>(reader),
                  static_cast<unsigned long long>(required_size),
                  static_cast<unsigned long long>(requested_size),
                  static_cast<unsigned long long>(kAutoReportProcessCeilingBytes));
      return false;
    }
    const auto release_reservation = [&] { release_live_bytes(requested_size); };

    const detail::AutoReportAllocation allocation =
        detail::allocate_auto_report_memory(core, agent, requested);
    if (!allocation) {
      release_reservation();
      record_allocation_failure(required_size, /*capacity_failure=*/false);
      log_message(kLogInfo,
                  "ConSan auto report allocation reader=%llu outcome=failed reason=%s "
                  "status=%d bytes=%zu",
                  static_cast<unsigned long long>(reader), allocation.failure_reason,
                  static_cast<int>(allocation.status), requested);
      return false;
    }
    void *ptr = allocation.data;

    const uint64_t generation = next_generation_.fetch_add(1, std::memory_order_relaxed) + 1u;
    auto *header = static_cast<ReportHeader *>(ptr);
    *header = make_report_header_for_layout(generation, /*dispatch_id=*/reader, layout);

    {
      std::lock_guard lock(mutex_);
      if (entry_count_ >= entries_.size()) {
        log_message(kLogInfo, "ConSan auto report buffer registry is full");
        (void)core->hsa_memory_free_fn(ptr);
        if (reserved_entry_count_ != 0)
          --reserved_entry_count_;
        (void)release_auto_report_bytes(process_budget_, requested_size);
        ++allocation_failure_count_;
        return false;
      }
      if (reserved_entry_count_ != 0)
        --reserved_entry_count_;
      successful_allocated_bytes_ += requested_size;
      entries_[entry_count_++] = Entry{
          .reader = reader,
          .ptr = ptr,
          .size = requested,
          .required_size = static_cast<size_t>(required_size),
          .generation = generation,
          .layout = layout,
          .fine_grained = allocation.fine_grained,
          .input_fingerprint = {},
          .static_metadata = {},
          .static_metadata_counted = false,
          .executable = 0,
          .executable_bound = false,
      };
    }
    *address = reinterpret_cast<uint64_t>(ptr);
    *registered_size = requested;
    if (registered_generation != nullptr)
      *registered_generation = generation;
    log_message(kLogInfo,
                "ConSan auto report buffer reader=%llu addr=0x%llx bytes=%zu "
                "required_bytes=%llu cap_bytes=%llu process_current_bytes=%llu "
                "process_peak_bytes=%llu process_ceiling_bytes=%llu allocation_outcome=allocated "
                ""
                ""
                ""
                ""
                ""
                ""
                ""
                ""
                "watchpoint_capacity=%u causal_window_capacity=%u "
                "sync_metadata_capacity=%u pending_acquire_capacity=%u "
                "generation=%llu fine_grained=%s",
                static_cast<unsigned long long>(reader), static_cast<unsigned long long>(*address),
                requested, static_cast<unsigned long long>(required_size),
                static_cast<unsigned long long>(configured_cap),
                static_cast<unsigned long long>(current_live_bytes()),
                static_cast<unsigned long long>(peak_live_bytes()),
                static_cast<unsigned long long>(kAutoReportProcessCeilingBytes),
                layout.watchpoint_capacity, layout.causal_window_capacity,
                layout.sync_metadata_capacity, layout.pending_acquire_capacity,
                static_cast<unsigned long long>(generation),
                allocation.fine_grained ? "true" : "false");
    return true;
  }

  using Summary = ReportSummary;

  void register_metadata(uint64_t reader, uint64_t generation, std::string_view input_fingerprint,
                         const StaticAccessMappings &static_mapping) {
    std::lock_guard lock(mutex_);
    auto entry = std::find_if(entries_.begin(), entries_.begin() + entry_count_,
                              [reader, generation](const Entry &item) {
                                return item.reader == reader && item.generation == generation;
                              });
    if (entry == entries_.begin() + entry_count_)
      return;

    entry->input_fingerprint = input_fingerprint;
    entry->static_metadata.reset();

    AccessStaticMetadata metadata;
    uint32_t minimum_banks = std::numeric_limits<uint32_t>::max();
    uint32_t maximum_banks = 0;
    for (const StaticAccessMapping &static_access : static_mapping) {
      if (static_access.access.owner_provenance_complete &&
          static_access.access.execution_owner_kernel_ids.empty()) {
        metadata.malformed = true;
      }
      const uint64_t slot_count =
          static_cast<uint64_t>(static_access.range_count) * static_access.bank_count;
      if (static_access.range_count != 0u && static_access.bank_count != 0u &&
          static_access.first_slot <= entry->layout.watchpoint_capacity &&
          slot_count <= entry->layout.watchpoint_capacity - static_access.first_slot) {
        minimum_banks = std::min(minimum_banks, static_access.bank_count);
        maximum_banks = std::max(maximum_banks, static_access.bank_count);
        metadata.mappings.push_back({
            .first_slot = static_access.first_slot,
            .range_count = static_access.range_count,
            .bank_count = static_access.bank_count,
            .instruction_offset = static_access.access.original_site.original_text_offset,
            .emitted_probe_offset = static_access.emitted_probe_text_offset,
            .relocated_guest_offset = static_access.relocated_guest_text_offset.value_or(0u),
            .scratch_vgpr = static_access.scratch_vgpr,
            .owner_kernel_ids = {},
            .owner_provenance_complete = static_access.access.owner_provenance_complete &&
                                         !static_access.access.execution_owner_kernel_ids.empty(),
            .uniform_lds_store = static_access.access.uniform_lds_store,
        });
        for (ProgramContainerId owner : static_access.access.execution_owner_kernel_ids)
          metadata.mappings.back().owner_kernel_ids.push_back(owner.ordinal);
      } else {
        metadata.malformed = true;
      }
    }
    if (!static_mapping.empty()) {
      log_message(kLogInfo,
                  "ConSan diagnostic map reader=%llu entries=%zu mappings=%zu "
                  "capacity=%u malformed=%s effective_banks_min=%u effective_banks_max=%u",
                  static_cast<unsigned long long>(reader), static_mapping.size(),
                  metadata.mappings.size(), entry->layout.watchpoint_capacity,
                  metadata.malformed ? "true" : "false",
                  metadata.mappings.empty() ? 0u : minimum_banks, maximum_banks);
    }
    entry->static_metadata = std::move(metadata);
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
        entries_, entry_count_, executable.handle, completed_summary_,
        [&](const Entry &entry) { return summarize(core, entry); },
        [&](Entry &entry) { return release_entry(core, entry, /*allow_runtime_reclaimed=*/false); },
        accumulate_summary);
  }

  ReportCheckpointResult checkpoint_after_device_synchronize(CoreApiTable *core, bool automatic) {
    std::lock_guard lock(mutex_);
    uint64_t automatic_epoch = 0;
    bool analyze = true;
    if (automatic) {
      if (automatic_epoch_ == std::numeric_limits<uint64_t>::max()) {
        log_message(kLogInfo, "ConSan epoch checkpoint outcome=selection-failed "
                              "reason=automatic-epoch-counter-overflow");
        return {.status = EpochCheckpointStatus::ReportSnapshotFailed,
                .report_count = entry_count_};
      }
      automatic_epoch = automatic_epoch_ + 1u;
      analyze = epoch_analysis_policy_.selects(automatic_epoch, manual_analysis_window_open_);
    }

    if (!analyze) {
      // A discarded epoch still needs a transactional reset: validate every
      // live header before changing any buffer, then clear all buffers. It
      // deliberately avoids the full snapshot/decode/analyze/render pipeline.
      for (const Entry &entry : std::span(entries_).first(entry_count_)) {
        const auto *header = static_cast<const ReportHeader *>(entry.ptr);
        if (header == nullptr || !report_header_is_current(*header) ||
            header->generation != entry.generation ||
            !report_layout_matches_header(*header, entry.layout, entry.size)) {
          log_message(kLogInfo,
                      "ConSan epoch checkpoint outcome=discard-failed reader=%llu "
                      "automatic_epoch=%llu reason=invalid-header-or-layout",
                      static_cast<unsigned long long>(entry.reader),
                      static_cast<unsigned long long>(automatic_epoch));
          return {.status = EpochCheckpointStatus::ReportSnapshotFailed,
                  .report_count = entry_count_};
        }
      }
      for (Entry &entry : std::span(entries_).first(entry_count_))
        reset_entry(entry);
      completed_summary_.discarded_epoch_count += entry_count_;
      automatic_epoch_ = automatic_epoch;
      log_message(kLogInfo,
                  "ConSan epoch checkpoint outcome=discarded automatic_epoch=%llu "
                  "reports=%zu",
                  static_cast<unsigned long long>(automatic_epoch), entry_count_);
      return {.status = EpochCheckpointStatus::Complete, .report_count = entry_count_};
    }

    struct PreparedEpoch {
      size_t entry_index = 0;
      ReportSnapshot snapshot;
      ReportSummary summary;
    };
    std::vector<PreparedEpoch> prepared;
    prepared.reserve(entry_count_);

    // Acquire every snapshot before changing any live report. A failed coarse
    // copy or malformed header therefore leaves the complete current epoch in
    // place and makes the checkpoint safely retryable.
    for (size_t index = 0; index < entry_count_; ++index) {
      const Entry &entry = entries_[index];
      ReportSnapshot snapshot = capture_snapshot(core, entry);
      if (!snapshot.complete() || snapshot.bytes.size() < sizeof(ReportHeader)) {
        log_message(kLogInfo, "ConSan epoch checkpoint outcome=snapshot-failed reader=%llu",
                    static_cast<unsigned long long>(entry.reader));
        return {.status = EpochCheckpointStatus::ReportSnapshotFailed,
                .report_count = entry_count_};
      }
      const auto *header = reinterpret_cast<const ReportHeader *>(snapshot.bytes.data());
      if (!report_header_is_current(*header) || header->generation != entry.generation ||
          !report_layout_matches_header(*header, entry.layout, entry.size)) {
        log_message(kLogInfo,
                    "ConSan epoch checkpoint outcome=snapshot-failed reader=%llu "
                    "reason=invalid-header-or-layout",
                    static_cast<unsigned long long>(entry.reader));
        return {.status = EpochCheckpointStatus::ReportSnapshotFailed,
                .report_count = entry_count_};
      }
      prepared.push_back({.entry_index = index, .snapshot = std::move(snapshot), .summary = {}});
    }

    for (PreparedEpoch &epoch : prepared) {
      const Entry &entry = entries_[epoch.entry_index];
      const ReportPipelineResult result = process_report(entry, epoch.snapshot);
      if (!result.complete) {
        log_message(kLogInfo,
                    "ConSan epoch checkpoint outcome=snapshot-failed reader=%llu "
                    "reason=decode-failed",
                    static_cast<unsigned long long>(entry.reader));
        return {.status = EpochCheckpointStatus::ReportSnapshotFailed,
                .report_count = entry_count_};
      }
      epoch.summary = result.summary;
      if (entry.fine_grained)
        epoch.summary.fine_grained_snapshot_bytes = epoch.snapshot.copied_bytes;
      else
        epoch.summary.coarse_grained_snapshot_bytes = epoch.snapshot.copied_bytes;
    }

    for (PreparedEpoch &epoch : prepared) {
      Entry &entry = entries_[epoch.entry_index];
      // Static metadata belongs to the allocation/code object, not to each
      // dynamic epoch. Preserve it for analysis while accounting it once.
      if (entry.static_metadata_counted)
        epoch.summary.static_mapping_malformed_count = 0;
      entry.static_metadata_counted = true;
      // Instrumented code embeds the allocation generation for ConSan and
      // Preserve the report identity while clearing all epoch-local state.
      reset_entry(entry);

      // `buffer_count` describes physical allocations, not logical epochs.
      // The same allocation remains live and will be counted exactly once at
      // retirement or process unload.
      epoch.summary.buffer_count = 0;
      ++epoch.summary.completed_epoch_count;
      accumulate_summary(completed_summary_, epoch.summary);
    }
    if (automatic)
      automatic_epoch_ = automatic_epoch;
    log_message(kLogInfo,
                "ConSan epoch checkpoint outcome=complete automatic_epoch=%llu reports=%zu",
                static_cast<unsigned long long>(automatic_epoch), prepared.size());
    return {.status = EpochCheckpointStatus::Complete, .report_count = prepared.size()};
  }

  Summary summarize_and_clear(CoreApiTable *core) {
    std::lock_guard lock(mutex_);
    Summary total = completed_summary_;
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
    completed_summary_ = {};
    automatic_epoch_ = 0;
    manual_analysis_window_open_ = false;
    epoch_analysis_policy_ = {};
    process_budget_.peak_live_bytes = process_budget_.current_live_bytes;
    return total;
  }

private:
  struct Entry {
    uint64_t reader = 0;
    void *ptr = nullptr;
    size_t size = 0;
    size_t required_size = 0;
    uint64_t generation = 0;
    ReportBufferLayout layout;
    bool fine_grained = false;
    std::string input_fingerprint;
    std::optional<AccessStaticMetadata> static_metadata;
    bool static_metadata_counted = false;
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
                  "ConSan auto report cleanup reader=%llu bytes=%zu "
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
                    "ConSan auto report cleanup reader=%llu bytes=%zu "
                    "outcome=runtime-reclaimed status=%d",
                    static_cast<unsigned long long>(entry.reader), entry.size,
                    static_cast<int>(free_status));
      }
    }
    if (freed) {
      (void)release_auto_report_bytes(process_budget_, entry.size);
      return true;
    }
    log_message(
        kLogInfo, "ConSan auto report cleanup reader=%llu bytes=%zu outcome=failed status=%d",
        static_cast<unsigned long long>(entry.reader), entry.size, static_cast<int>(free_status));
    return false;
  }

  static void accumulate_summary(Summary &total, const Summary &entry) {
    total.buffer_count += entry.buffer_count;
    total.completed_epoch_count += entry.completed_epoch_count;
    total.discarded_epoch_count += entry.discarded_epoch_count;
    total.fine_grained_snapshot_bytes += entry.fine_grained_snapshot_bytes;
    total.coarse_grained_snapshot_bytes += entry.coarse_grained_snapshot_bytes;
    total.visible_watchpoint_count += entry.visible_watchpoint_count;
    total.visible_sync_metadata_count += entry.visible_sync_metadata_count;
    total.conflict_count += entry.conflict_count;
    total.immediate_conflict_count += entry.immediate_conflict_count;
    total.claimed_window_count += entry.claimed_window_count;
    total.dropped_window_count += entry.dropped_window_count;
    total.saturated_window_count += entry.saturated_window_count;
    total.stale_snapshot_count += entry.stale_snapshot_count;
    total.incomplete_snapshot_count += entry.incomplete_snapshot_count;
    total.changed_snapshot_count += entry.changed_snapshot_count;
    total.malformed_snapshot_count += entry.malformed_snapshot_count;
    total.static_mapping_malformed_count += entry.static_mapping_malformed_count;
    total.unsupported_sync_count += entry.unsupported_sync_count;
    total.malformed_sync_count += entry.malformed_sync_count;
  }

  void record_allocation_attempt(uint64_t required_size) {
    std::lock_guard lock(mutex_);
    required_report_bytes_ = byte_accounting::saturating_add(required_report_bytes_, required_size);
  }

  static void reset_entry(Entry &entry) {
    std::memset(entry.ptr, 0, entry.size);
    *static_cast<ReportHeader *>(entry.ptr) =
        make_report_header_for_layout(entry.generation, entry.reader, entry.layout);
    std::atomic_thread_fence(std::memory_order_release);
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
        !reserve_auto_report_bytes(process_budget_, requested_size)) {
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
    (void)release_auto_report_bytes(process_budget_, requested_size);
  }

  [[nodiscard]] uint64_t current_live_bytes() const {
    std::lock_guard lock(mutex_);
    return process_budget_.current_live_bytes;
  }

  [[nodiscard]] uint64_t peak_live_bytes() const {
    std::lock_guard lock(mutex_);
    return process_budget_.peak_live_bytes;
  }

  ReportSnapshot capture_snapshot(CoreApiTable *core, const Entry &entry) const {
    return capture_report_snapshot({.source = entry.ptr,
                                    .size = entry.size,

                                    .fine_grained = entry.fine_grained},
                                   core != nullptr && core->hsa_memory_copy_fn != nullptr
                                       ? copy_coarse_report_snapshot
                                       : nullptr,
                                   core);
  }

  // All callers hold mutex_. Budget spans reports, executables, and epochs;
  // exhausting it suppresses examples only, never conflict analysis/counts.
  ReportPipelineResult process_report(const Entry &entry, const ReportSnapshot &snapshot,
                                      Summary summary = {}) {
    const auto limit = std::min(conflict_limit_, conflict_examples_remaining_);
    const auto result = hook::process_report(
        {.reader = entry.reader,
         .source_address = reinterpret_cast<uint64_t>(entry.ptr),
         .size = entry.size,
         .layout = entry.layout,
         .fine_grained = entry.fine_grained,
         .input_fingerprint = entry.input_fingerprint,
         .static_metadata = entry.static_metadata ? &*entry.static_metadata : nullptr,
         .conflict_example_limit = limit,
         .allow_uniform_lds_stores = allow_uniform_lds_stores_,
         .expected_generation = entry.generation},
        snapshot, summary);
    conflict_examples_remaining_ -=
        std::min(conflict_examples_remaining_, result.conflict_example_count);
    return result;
  }

  Summary summarize(CoreApiTable *core, const Entry &entry) {
    Summary summary;
    summary.buffer_count = 1;

    const ReportSnapshot snapshot = capture_snapshot(core, entry);
    if (!snapshot.complete()) {
      if (snapshot.failure == ReportSnapshotFailure::CopyUnavailable) {
        log_message(kLogInfo,
                    "ConSan auto report reader=%llu needs hsa_memory_copy for "
                    "coarse-grained summary",
                    static_cast<unsigned long long>(entry.reader));
      } else if (snapshot.failure == ReportSnapshotFailure::CopyFailed) {
        log_message(kLogInfo, "ConSan auto report reader=%llu hsa_memory_copy failed status=%d",
                    static_cast<unsigned long long>(entry.reader), snapshot.copy_status);
      } else {
        log_message(kLogInfo, "ConSan auto report reader=%llu has invalid snapshot source",
                    static_cast<unsigned long long>(entry.reader));
      }
      return summary;
    }
    if (entry.fine_grained)
      summary.fine_grained_snapshot_bytes = snapshot.copied_bytes;
    else
      summary.coarse_grained_snapshot_bytes = snapshot.copied_bytes;

    summary = process_report(entry, snapshot, summary).summary;
    if (entry.static_metadata_counted)
      summary.static_mapping_malformed_count = 0;
    return summary;
  }

  mutable std::mutex mutex_;
  std::array<Entry, 256> entries_{};
  size_t entry_count_ = 0;
  size_t reserved_entry_count_ = 0;
  uint64_t required_report_bytes_ = 0;
  uint64_t successful_allocated_bytes_ = 0;
  AutoReportProcessBudget process_budget_;
  uint64_t allocation_failure_count_ = 0;
  uint64_t capacity_failure_count_ = 0;
  uint64_t cleanup_failure_count_ = 0;
  Summary completed_summary_;
  HookConfig::EpochAnalysisPolicy epoch_analysis_policy_;
  bool allow_uniform_lds_stores_ = false;
  uint32_t conflict_limit_ = 8;
  uint32_t conflict_examples_remaining_ = 64;
  uint64_t automatic_epoch_ = 0;
  bool manual_analysis_window_open_ = false;
  std::atomic<uint64_t> next_generation_{0};
};

void reject_report_plan(uint64_t reader, uint64_t required_size, uint64_t configured_cap,
                        std::string_view reason) {
  AutoReportBufferRegistry::instance().reject_plan(reader, required_size, configured_cap, reason);
}

bool advance_report_generation_for_test(uint64_t generation) {
  return AutoReportBufferRegistry::instance().advance_generation_for_test(generation);
}

bool allocate_report_buffer(CoreApiTable *core, hsa_agent_t agent, uint64_t reader,
                            uint64_t required_size, uint64_t requested_size,
                            uint64_t configured_cap, const ReportBufferLayout &layout,
                            bool track_barriers, bool track_atomics, uint64_t *address,
                            uint64_t *registered_size, uint64_t *registered_generation) {
  return AutoReportBufferRegistry::instance().allocate(
      core, agent, reader, required_size, requested_size, configured_cap, layout, track_barriers,
      track_atomics, address, registered_size, registered_generation);
}

void register_report_metadata(uint64_t reader, uint64_t generation,
                              std::string_view input_fingerprint,
                              const StaticAccessMappings &static_mapping) {
  AutoReportBufferRegistry::instance().register_metadata(reader, generation, input_fingerprint,
                                                         static_mapping);
}

void bind_report_buffer_to_executable(uint64_t reader, uint64_t generation,
                                      hsa_executable_t executable) {
  AutoReportBufferRegistry::instance().bind_to_executable(reader, generation, executable);
}

void discard_report_buffer(CoreApiTable *core, uint64_t reader, uint64_t generation) {
  AutoReportBufferRegistry::instance().discard(core, reader, generation);
}

void retire_report_buffers(CoreApiTable *core, hsa_executable_t executable) {
  AutoReportBufferRegistry::instance().retire(core, executable);
}

void configure_epoch_analysis(HookConfig::EpochAnalysisPolicy policy, uint32_t conflict_limit,
                              uint32_t total_conflict_limit, bool allow_uniform_lds_stores) {
  AutoReportBufferRegistry::instance().configure_epoch_analysis(
      policy, conflict_limit, total_conflict_limit, allow_uniform_lds_stores);
}

bool begin_epoch_analysis_window() {
  return AutoReportBufferRegistry::instance().begin_epoch_analysis_window();
}

bool end_epoch_analysis_window() {
  return AutoReportBufferRegistry::instance().end_epoch_analysis_window();
}

ReportCheckpointResult checkpoint_report_buffers_after_device_synchronize(CoreApiTable *core) {
  return AutoReportBufferRegistry::instance().checkpoint_after_device_synchronize(
      core, /*automatic=*/false);
}

ReportCheckpointResult checkpoint_report_buffers_automatically(CoreApiTable *core) {
  return AutoReportBufferRegistry::instance().checkpoint_after_device_synchronize(
      core, /*automatic=*/true);
}

ReportSummary summarize_and_clear_report_buffers(CoreApiTable *core) {
  return AutoReportBufferRegistry::instance().summarize_and_clear(core);
}

} // namespace rocjitsu::consan::hook
