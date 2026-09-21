// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rj_hsa_dbi_report_trust.h"

namespace rocjitsu::consan::hook {

uint64_t ReportSummary::unusable_snapshot_count() const {
  return stale_snapshot_count + incomplete_snapshot_count + changed_snapshot_count +
         malformed_snapshot_count;
}

ReportTrustEvaluation evaluate_report_trust(const ReportSummary &summary, bool require_records) {
  ReportTrustEvaluation result;
  result.visible_evidence_count = summary.visible_watchpoint_count;
  result.required_records_missing = require_records && result.visible_evidence_count == 0u;
  result.dropped_record_count = summary.dropped_window_count;
  result.dynamic_incomplete_count = summary.allocation_failure_count +
                                    summary.cleanup_failure_count + result.dropped_record_count +
                                    summary.unusable_snapshot_count() +
                                    summary.unsupported_sync_count + summary.malformed_sync_count +
                                    summary.epoch_exhaustion_count;
  result.dynamic_complete =
      result.dynamic_incomplete_count == 0u && !result.required_records_missing;
  result.has_diagnostics = summary.conflict_count + summary.immediate_conflict_count > 0u;
  return result;
}

} // namespace rocjitsu::consan::hook
