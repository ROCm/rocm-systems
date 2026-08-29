// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rj_hsa_dbi_moi_report_trust.h"

namespace rocjitsu::consan_hook {

uint64_t AutoMoiReportSummary::sampled_unusable_snapshot_count() const {
  return sampled_stale_snapshot_count + sampled_incomplete_snapshot_count +
         sampled_changed_snapshot_count + sampled_malformed_snapshot_count;
}

uint64_t AutoMoiReportSummary::exact_unusable_snapshot_count() const {
  return exact_incomplete_snapshot_count + exact_changed_snapshot_count +
         exact_malformed_snapshot_count;
}

uint64_t AutoMoiReportSummary::release_unusable_snapshot_count() const {
  return release_incomplete_snapshot_count + release_changed_snapshot_count +
         release_overflow_snapshot_count + release_source_incomplete_snapshot_count +
         release_malformed_snapshot_count;
}

uint64_t AutoMoiReportSummary::token_unusable_snapshot_count() const {
  return token_incomplete_snapshot_count + token_changed_snapshot_count +
         token_malformed_snapshot_count;
}

AutoMoiReportTrustEvaluation evaluate_auto_moi_report_trust(const AutoMoiReportSummary &summary,
                                                            bool require_records) {
  AutoMoiReportTrustEvaluation result;
  result.visible_evidence_count =
      summary.visible_access_record_count + summary.visible_barrier_record_count +
      summary.visible_atomic_record_count + summary.visible_fence_record_count +
      summary.visible_diagnostic_record_count + summary.visible_inline_publication_count +
      summary.visible_exact_shadow_entry_count + summary.visible_inline_atomic_release_count +
      summary.visible_inline_acquired_token_count + summary.visible_sampled_watchpoint_count;
  result.required_records_missing = require_records && result.visible_evidence_count == 0u;
  result.dropped_record_count =
      summary.dropped_access_record_count + summary.dropped_barrier_record_count +
      summary.dropped_atomic_record_count + summary.dropped_fence_record_count +
      summary.dropped_diagnostic_record_count + summary.sampled_dropped_window_count;
  result.inline_coverage_loss_count =
      summary.inline_undercoverage_count + summary.inline_overflow_count +
      summary.inline_unsupported_count + summary.inline_malformed_count;
  result.dynamic_incomplete_count =
      summary.allocation_failure_count + summary.cleanup_failure_count +
      result.dropped_record_count + summary.record_replay_bank_saturation_count +
      summary.record_replay_invalid_site_token_count + summary.sampled_unusable_snapshot_count() +
      summary.exact_unusable_snapshot_count() + summary.release_unusable_snapshot_count() +
      summary.token_unusable_snapshot_count() + result.inline_coverage_loss_count +
      summary.sampled_unsupported_sync_count + summary.sampled_malformed_sync_count +
      summary.replay_dropped_access_count + summary.replay_dropped_barrier_count +
      summary.replay_unsupported_access_count + summary.replay_unsupported_atomic_count +
      summary.replay_unsupported_fence_count + summary.replay_metadata_full_count +
      summary.replay_diagnostic_capacity_exhausted_count;
  result.dynamic_complete =
      result.dynamic_incomplete_count == 0u && !result.required_records_missing;
  result.has_diagnostics = summary.visible_diagnostic_record_count +
                               summary.replay_diagnostic_count + summary.sampled_conflict_count +
                               summary.sampled_immediate_conflict_count >
                           0u;
  return result;
}

} // namespace rocjitsu::consan_hook
