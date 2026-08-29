// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

namespace rocjitsu::consan_hook {

/// Typed evidence, loss, and analysis totals produced by the automatic MOI
/// report pipeline. This contract deliberately has no dependency on HSA
/// lifecycle state or raw report-buffer layouts.
struct AutoMoiReportSummary {
  uint64_t buffer_count = 0;
  uint64_t required_report_bytes = 0;
  uint64_t allocated_report_bytes = 0;
  uint64_t current_live_report_bytes = 0;
  uint64_t current_live_report_bytes_after_cleanup = 0;
  uint64_t peak_live_report_bytes = 0;
  uint64_t allocation_failure_count = 0;
  uint64_t capacity_failure_count = 0;
  uint64_t cleanup_failure_count = 0;
  uint64_t fine_grained_snapshot_bytes = 0;
  uint64_t coarse_grained_snapshot_bytes = 0;
  uint64_t visible_access_record_count = 0;
  uint64_t visible_barrier_record_count = 0;
  uint64_t visible_atomic_record_count = 0;
  uint64_t visible_fence_record_count = 0;
  uint64_t visible_diagnostic_record_count = 0;
  uint64_t visible_inline_publication_count = 0;
  uint64_t visible_exact_shadow_entry_count = 0;
  uint64_t exact_incomplete_snapshot_count = 0;
  uint64_t exact_changed_snapshot_count = 0;
  uint64_t exact_malformed_snapshot_count = 0;
  uint64_t visible_inline_atomic_release_count = 0;
  uint64_t release_incomplete_snapshot_count = 0;
  uint64_t release_changed_snapshot_count = 0;
  uint64_t release_overflow_snapshot_count = 0;
  uint64_t release_source_incomplete_snapshot_count = 0;
  uint64_t release_malformed_snapshot_count = 0;
  uint64_t visible_inline_acquired_token_count = 0;
  uint64_t token_incomplete_snapshot_count = 0;
  uint64_t token_changed_snapshot_count = 0;
  uint64_t token_malformed_snapshot_count = 0;
  uint64_t inline_undercoverage_count = 0;
  uint64_t inline_overflow_count = 0;
  uint64_t inline_unsupported_count = 0;
  uint64_t inline_malformed_count = 0;
  uint64_t visible_sampled_watchpoint_count = 0;
  uint64_t visible_sampled_sync_metadata_count = 0;
  uint64_t dropped_access_record_count = 0;
  uint64_t dropped_barrier_record_count = 0;
  uint64_t dropped_atomic_record_count = 0;
  uint64_t dropped_fence_record_count = 0;
  uint64_t dropped_diagnostic_record_count = 0;
  uint64_t record_replay_bank_saturation_count = 0;
  uint64_t record_replay_invalid_site_token_count = 0;
  uint64_t replay_conflict_count = 0;
  uint64_t replay_diagnostic_count = 0;
  uint64_t replay_dropped_access_count = 0;
  uint64_t replay_dropped_barrier_count = 0;
  uint64_t replay_unsupported_access_count = 0;
  uint64_t replay_unsupported_atomic_count = 0;
  uint64_t replay_unsupported_fence_count = 0;
  uint64_t replay_metadata_full_count = 0;
  uint64_t replay_diagnostic_capacity_exhausted_count = 0;
  uint64_t sampled_conflict_count = 0;
  uint64_t sampled_immediate_conflict_count = 0;
  uint64_t sampled_claimed_window_count = 0;
  uint64_t sampled_dropped_window_count = 0;
  uint64_t sampled_saturated_window_count = 0;
  uint64_t sampled_stale_snapshot_count = 0;
  uint64_t sampled_incomplete_snapshot_count = 0;
  uint64_t sampled_changed_snapshot_count = 0;
  uint64_t sampled_malformed_snapshot_count = 0;
  uint64_t sampled_static_mapping_malformed_count = 0;
  uint64_t sampled_unsupported_sync_count = 0;
  uint64_t sampled_malformed_sync_count = 0;

  [[nodiscard]] uint64_t sampled_unusable_snapshot_count() const;
  [[nodiscard]] uint64_t exact_unusable_snapshot_count() const;
  [[nodiscard]] uint64_t release_unusable_snapshot_count() const;
  [[nodiscard]] uint64_t token_unusable_snapshot_count() const;
};

/// Host-side trust projection derived solely from one completed report
/// summary and the caller's evidence requirement.
struct AutoMoiReportTrustEvaluation {
  uint64_t visible_evidence_count = 0;
  uint64_t dynamic_incomplete_count = 0;
  uint64_t dropped_record_count = 0;
  uint64_t inline_coverage_loss_count = 0;
  bool required_records_missing = false;
  bool dynamic_complete = false;
  bool has_diagnostics = false;
};

[[nodiscard]] AutoMoiReportTrustEvaluation
evaluate_auto_moi_report_trust(const AutoMoiReportSummary &summary, bool require_records);

} // namespace rocjitsu::consan_hook
