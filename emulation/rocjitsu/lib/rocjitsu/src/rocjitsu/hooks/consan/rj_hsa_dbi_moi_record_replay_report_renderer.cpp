// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rj_hsa_dbi_moi_record_replay_report_renderer.h"

namespace rocjitsu::consan_hook {

AutoMoiModeRendering render_auto_moi_record_replay_report(
    const AutoMoiReportPipelineInput &input, const ConSanMoiReportHeader &header,
    const AutoMoiReportSummary &summary, const AutoMoiRecordReplayDecodedReport &mode,
    const AutoMoiRecordReplayAnalysis &analysis) {
  AutoMoiModeRendering result;
  const RecordReplayPressureTelemetry &pressure = analysis.pressure;
  const std::string_view pressure_reason =
      record_replay_pressure_unavailable_reason_name(pressure.unavailable_reason);
  result.effective_diagnostic_count = analysis.effective_diagnostic_count;
  result.summary_fields = format_auto_moi_report_text(
      " committed_records=%llu dispatch_tokens=%u/%u record_replay_flags=0x%x "
      "record_replay_bank_saturated=%s record_replay_pressure_available=%s "
      "record_replay_pressure_unavailable_reason=%.*s "
      "record_replay_access_table_occupied=%llu record_replay_access_table_capacity=%llu "
      "record_replay_observed_sites=%llu record_replay_max_site_owner_address_groups=%llu "
      "record_replay_address_group_headroom=%u record_replay_logical_access_ranges=%u "
      "record_replay_max_site_token=%u record_replay_invalid_site_tokens=%llu",
      static_cast<unsigned long long>(summary.visible_access_record_count),
      header.record_replay_dispatch_token_count, header.record_replay_dispatch_token_capacity,
      header.flags, summary.record_replay_bank_saturation_count != 0 ? "true" : "false",
      pressure.available ? "true" : "false", static_cast<int>(pressure_reason.size()),
      pressure_reason.data(),
      static_cast<unsigned long long>(pressure.occupied_access_record_count),
      static_cast<unsigned long long>(pressure.access_record_capacity),
      static_cast<unsigned long long>(pressure.observed_site_count),
      static_cast<unsigned long long>(pressure.maximum_site_owner_address_group_count),
      pressure.address_group_headroom, pressure.logical_access_range_count,
      pressure.maximum_site_token,
      static_cast<unsigned long long>(pressure.invalid_site_token_count));

  constexpr AutoMoiReportDiagnosticKind kDetail = AutoMoiReportDiagnosticKind::Detail;
  if (analysis.shadow_bounded) {
    result.details.push_back(
        {kDetail, format_auto_moi_report_text(
                      "ConSan MOI replay shadow bounded reader=%llu generation=%llu code_object=%s "
                      "required_shadow_entries=%llu limit=%llu shadow_entries=%llu",
                      static_cast<unsigned long long>(input.reader),
                      static_cast<unsigned long long>(header.generation),
                      input.input_fingerprint.empty() ? "missing" : input.input_fingerprint.data(),
                      static_cast<unsigned long long>(analysis.required_shadow_entry_count),
                      static_cast<unsigned long long>(1u << 20u),
                      static_cast<unsigned long long>(analysis.replay_shadow_entry_count))});
  }
  if (!analysis.replay_performed)
    return result;

  const ConSanMoiRecordReplayResult &replay = analysis.replay;
  result.details.push_back(
      {kDetail,
       format_auto_moi_report_text(
           "ConSan MOI auto replay reader=%llu generation=%llu code_object=%s "
           "replay_input_access=%zu published_access=%u processed_access=%u "
           "processed_barriers=%u processed_atomics=%u processed_fences=%u "
           "dropped_access=%u dropped_barriers=%u unsupported_access=%u "
           "unsupported_atomics=%u unsupported_fences=%u diagnostics=%u "
           "conflict=%s metadata_full=%s diagnostic_capacity_exhausted=%s "
           "release_metadata_max=%u acquired_metadata_max=%u diagnostic_capacity=%u "
           "replay_scratch_diagnostic_capacity=%u provenance_repaired=%u "
           "provenance_unresolved=%u disjoint_owner_suppressed=%u shadow_entries=%llu",
           static_cast<unsigned long long>(input.reader),
           static_cast<unsigned long long>(analysis.replay_header.generation),
           input.input_fingerprint.empty() ? "missing" : input.input_fingerprint.data(),
           mode.access_records.size(), replay.published_access_count, replay.processed_access_count,
           replay.processed_barrier_count, replay.processed_atomic_count,
           replay.processed_fence_count, replay.dropped_access_count, replay.dropped_barrier_count,
           replay.unsupported_access_count, replay.unsupported_atomic_count,
           replay.unsupported_fence_count, analysis.effective_diagnostic_count,
           analysis.effective_conflict ? "true" : "false", replay.metadata_full ? "true" : "false",
           replay.diagnostic_capacity_exhausted ? "true" : "false",
           replay.maximum_atomic_release_metadata_count,
           replay.maximum_acquired_epoch_metadata_count, header.diagnostic_capacity,
           analysis.replay_header.diagnostic_capacity,
           analysis.provenance.repaired_diagnostic_count,
           analysis.provenance.unresolved_diagnostic_count,
           analysis.disjoint_owner_suppressed_count,
           static_cast<unsigned long long>(analysis.replay_shadow_entry_count))});
  for (uint32_t index = 0; index < analysis.diagnostics.size(); ++index) {
    const ConSanMoiDiagnosticRecord &diagnostic = analysis.diagnostics[index];
    result.details.push_back(
        {kDetail,
         format_auto_moi_report_text(
             "ConSan MOI auto replay diagnostic reader=%llu index=%u kind=%u code_object=%s "
             "report_generation=%llu generation=%llu epoch=%u first_owner=%u second_owner=%u "
             "first_inst=0x%x second_inst=0x%x first_lds_known=%s first_lds=[%u,%u) "
             "second_lds=[%u,%u) first_kind=%u second_kind=%u first_lane_mask=0x%llx "
             "second_lane_mask=0x%llx",
             static_cast<unsigned long long>(input.reader), index, diagnostic.kind,
             input.input_fingerprint.empty() ? "missing" : input.input_fingerprint.data(),
             static_cast<unsigned long long>(analysis.replay_header.generation),
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
             static_cast<unsigned long long>(diagnostic.second_lane_mask))});
  }
  return result;
}

} // namespace rocjitsu::consan_hook
