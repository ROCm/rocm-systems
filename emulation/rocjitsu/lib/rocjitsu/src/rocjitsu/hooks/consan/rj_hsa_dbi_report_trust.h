// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

namespace rocjitsu::consan::hook {

/// Typed evidence, loss, and analysis totals produced by the automatic ConSan
/// report pipeline. This contract deliberately has no dependency on HSA
/// lifecycle state or raw report-buffer layouts.
struct ReportSummary {
  uint64_t buffer_count = 0;
  uint64_t completed_epoch_count = 0;
  uint64_t discarded_epoch_count = 0;
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
  uint64_t visible_watchpoint_count = 0;
  uint64_t visible_sync_metadata_count = 0;
  uint64_t conflict_count = 0;
  uint64_t suppressed_uniform_write_conflict_count = 0;
  uint64_t immediate_conflict_count = 0;
  uint64_t claimed_window_count = 0;
  uint64_t dropped_window_count = 0;
  uint64_t saturated_window_count = 0;
  uint64_t stale_snapshot_count = 0;
  uint64_t incomplete_snapshot_count = 0;
  uint64_t changed_snapshot_count = 0;
  uint64_t malformed_snapshot_count = 0;
  uint64_t static_mapping_malformed_count = 0;
  uint64_t unsupported_sync_count = 0;
  uint64_t malformed_sync_count = 0;

  [[nodiscard]] uint64_t unusable_snapshot_count() const;
};

/// Host-side trust projection derived solely from one completed report
/// summary and the caller's evidence requirement.
struct ReportTrustEvaluation {
  uint64_t visible_evidence_count = 0;
  uint64_t dynamic_incomplete_count = 0;
  uint64_t dropped_record_count = 0;
  bool required_records_missing = false;
  bool dynamic_complete = false;
  bool has_diagnostics = false;
};

[[nodiscard]] ReportTrustEvaluation evaluate_report_trust(const ReportSummary &summary,
                                                          bool require_records);

} // namespace rocjitsu::consan::hook
