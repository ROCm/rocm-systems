// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rj_hsa_dbi_moi_report_renderer.h"

#include "rj_hsa_dbi_moi_inline_shadow_report_renderer.h"
#include "rj_hsa_dbi_moi_record_replay_report_renderer.h"
#include "rj_hsa_dbi_moi_sampled_report_renderer.h"

#include <cstdarg>
#include <cstdio>
#include <string>
#include <type_traits>
#include <utility>

namespace rocjitsu::consan_hook {

std::string format_auto_moi_report_text(const char *format, ...) {
  std::va_list args;
  va_start(args, format);
  std::va_list size_args;
  va_copy(size_args, args);
  const int size = std::vsnprintf(nullptr, 0, format, size_args);
  va_end(size_args);
  if (size < 0) {
    va_end(args);
    return {};
  }
  std::string result(static_cast<size_t>(size), '\0');
  std::vsnprintf(result.data(), result.size() + 1u, format, args);
  va_end(args);
  return result;
}

std::vector<AutoMoiReportDiagnostic>
render_auto_moi_report(const AutoMoiReportRenderInput &render_input) {
  std::vector<AutoMoiReportDiagnostic> rendered;
  const auto append = [&rendered](AutoMoiReportDiagnosticKind kind, const char *format,
                                  auto... args) {
    rendered.push_back({kind, format_auto_moi_report_text(format, args...)});
  };
  constexpr AutoMoiReportDiagnosticKind kFailure = AutoMoiReportDiagnosticKind::Failure;
  constexpr AutoMoiReportDiagnosticKind kSummary = AutoMoiReportDiagnosticKind::Summary;
  constexpr AutoMoiReportDiagnosticKind kDetail = AutoMoiReportDiagnosticKind::Detail;

  const AutoMoiReportPipelineInput &input = render_input.pipeline;
  const AutoMoiDecodedReport &decoded = render_input.decoded;
  const AutoMoiReportSummary &summary = render_input.summary;
  if (!decoded.complete()) {
    const ConSanMoiReportHeader &invalid_header = decoded.header;
    if (decoded.failure == AutoMoiReportDecodeFailure::InvalidHeader) {
      append(kFailure,
             "ConSan MOI auto report reader=%llu has invalid header magic=0x%08x "
             "abi=%u header_size=%u",
             static_cast<unsigned long long>(input.reader), invalid_header.magic,
             invalid_header.abi_version, invalid_header.header_size);
    } else if (decoded.failure == AutoMoiReportDecodeFailure::LayoutMismatch) {
      append(kFailure, "ConSan MOI auto report reader=%llu has inconsistent ABI-v%u layout",
             static_cast<unsigned long long>(input.reader), kConSanMoiReportAbiVersion);
    } else {
      append(kFailure, "ConSan MOI auto report reader=%llu has undersized host snapshot",
             static_cast<unsigned long long>(input.reader));
    }
    return rendered;
  }
  if (render_input.mode_analysis == nullptr ||
      !auto_moi_analysis_matches_engine(*render_input.mode_analysis, input.layout.engine)) {
    append(kFailure, "ConSan MOI auto report reader=%llu has missing typed analysis",
           static_cast<unsigned long long>(input.reader));
    return rendered;
  }

  AutoMoiModeRendering mode_rendering = std::visit(
      [&](const auto &mode) -> AutoMoiModeRendering {
        using Mode = std::decay_t<decltype(mode)>;
        if constexpr (std::is_same_v<Mode, AutoMoiRecordReplayDecodedReport>) {
          return render_auto_moi_record_replay_report(
              input, decoded.header, summary, mode,
              std::get<AutoMoiRecordReplayAnalysis>(*render_input.mode_analysis));
        } else if constexpr (std::is_same_v<Mode, AutoMoiInlineShadowDecodedReport>) {
          return render_auto_moi_inline_shadow_report(input, decoded.header, summary, mode);
        } else {
          return render_auto_moi_sampled_report(
              input, decoded.header, summary, mode,
              std::get<AutoMoiSampledConflictAnalysis>(*render_input.mode_analysis));
        }
      },
      decoded.mode);
  rendered.insert(rendered.end(), std::make_move_iterator(mode_rendering.evidence.begin()),
                  std::make_move_iterator(mode_rendering.evidence.end()));

  const ConSanMoiReportHeader &header = decoded.header;
  const uint32_t visible_records = static_cast<uint32_t>(decoded.visible_access_slots.size());
  const uint32_t visible_barriers = static_cast<uint32_t>(decoded.barrier_records.size());
  const uint32_t visible_atomics = static_cast<uint32_t>(decoded.atomic_records.size());
  const uint32_t visible_fences = static_cast<uint32_t>(decoded.fence_records.size());
  const uint32_t visible_diagnostics = static_cast<uint32_t>(decoded.diagnostics.size());
  append(kSummary,
         "ConSan MOI auto report reader=%llu addr=0x%llx bytes=%zu generation=%llu "
         "code_object=%s event_counter=%u access_records=%u visible_records=%u "
         "dropped_records=%llu capacity=%u barrier_records=%u visible_barriers=%u "
         "dropped_barriers=%llu barrier_capacity=%u atomic_records=%u visible_atomics=%u "
         "dropped_atomics=%llu atomic_capacity=%u fence_records=%u visible_fences=%u "
         "dropped_fences=%llu fence_capacity=%u diagnostics=%u visible_diagnostics=%u "
         "dropped_diagnostics=%llu diagnostic_capacity=%u%s fine_grained=%s",
         static_cast<unsigned long long>(input.reader),
         static_cast<unsigned long long>(input.source_address), input.size,
         static_cast<unsigned long long>(header.generation),
         input.input_fingerprint.empty() ? "missing" : input.input_fingerprint.data(),
         header.event_counter, header.access_record_count, visible_records,
         static_cast<unsigned long long>(summary.dropped_access_record_count),
         header.access_record_capacity, header.barrier_record_count, visible_barriers,
         static_cast<unsigned long long>(summary.dropped_barrier_record_count),
         header.barrier_record_capacity, header.atomic_record_count, visible_atomics,
         static_cast<unsigned long long>(summary.dropped_atomic_record_count),
         header.atomic_record_capacity, header.fence_record_count, visible_fences,
         static_cast<unsigned long long>(summary.dropped_fence_record_count),
         input.layout.fence_record_capacity, mode_rendering.effective_diagnostic_count,
         visible_diagnostics,
         static_cast<unsigned long long>(summary.dropped_diagnostic_record_count),
         header.diagnostic_capacity, mode_rendering.summary_fields.c_str(),
         input.fine_grained ? "true" : "false");

  const uint32_t fence_sample_count = auto_moi_report_detail_count(visible_fences);
  for (uint32_t i = 0; i < fence_sample_count; ++i) {
    const ConSanMoiFenceRecord &fence = decoded.fence_records[i];
    append(kDetail,
           "ConSan MOI auto fence reader=%llu index=%u event_index=%u owner=%u "
           "generation=%llu epoch=%u workgroup=(%u,%u,%u) inst=0x%x kind=%u scope=%u "
           "semantics=%u token=0x%016llx",
           static_cast<unsigned long long>(input.reader), i, fence.event_index, fence.owner_id,
           static_cast<unsigned long long>(fence.generation), fence.epoch, fence.workgroup_x,
           fence.workgroup_y, fence.workgroup_z, fence.instruction_offset,
           static_cast<unsigned>(fence.kind), fence.scope, fence.semantics,
           static_cast<unsigned long long>(fence.communication_token));
  }
  if (visible_fences > fence_sample_count) {
    append(kDetail, "ConSan MOI auto fence reader=%llu omitted=%u after log limit=%u",
           static_cast<unsigned long long>(input.reader), visible_fences - fence_sample_count,
           kAutoMoiReportDetailLimit);
  }

  uint32_t sampled_records = 0;
  for (uint32_t i = 0; i < visible_records && sampled_records < kAutoMoiReportDetailLimit; ++i) {
    const ConSanMoiAccessRecord &record = decoded.visible_access_slots[i];
    if (record.claim_token == 0 &&
        record.access_kind == static_cast<uint32_t>(ConSanMoiShadowAccessKind::Empty))
      continue;
    append(kDetail,
           "ConSan MOI auto record reader=%llu index=%u event_index=%u kind=%u wave=%u "
           "claim_token=0x%016llx generation=%llu epoch=%u workgroup=(%u,%u,%u) "
           "inst=0x%x lds_offset=%u lds_bytes=%u cells=[%u,%u) lane_mask=0x%llx",
           static_cast<unsigned long long>(input.reader), i, record.event_index, record.access_kind,
           record.wave_id, static_cast<unsigned long long>(record.claim_token),
           static_cast<unsigned long long>(record.generation), record.epoch, record.workgroup_x,
           record.workgroup_y, record.workgroup_z, record.instruction_offset,
           record.lds_byte_offset, record.lds_byte_count, record.start_cell,
           record.start_cell + record.cell_count,
           static_cast<unsigned long long>(record.lane_mask));
    ++sampled_records;
  }

  const uint32_t barrier_sample_count = auto_moi_report_detail_count(visible_barriers);
  for (uint32_t i = 0; i < barrier_sample_count; ++i) {
    const ConSanMoiBarrierRecord &record = decoded.barrier_records[i];
    append(kDetail,
           "ConSan MOI auto barrier reader=%llu index=%u event_index=%u wave=%u "
           "inst=0x%x lane_mask=0x%llx",
           static_cast<unsigned long long>(input.reader), i, record.event_index, record.wave_id,
           record.instruction_offset, static_cast<unsigned long long>(record.lane_mask));
  }
  const uint32_t atomic_sample_count = auto_moi_report_detail_count(visible_atomics);
  for (uint32_t i = 0; i < atomic_sample_count; ++i) {
    const ConSanMoiAtomicRecord &record = decoded.atomic_records[i];
    append(kDetail,
           "ConSan MOI auto atomic reader=%llu index=%u event_index=%u kind=%u owner=%u "
           "generation=%llu epoch=%u workgroup=(%u,%u,%u) inst=0x%x address=0x%llx "
           "scope=%u semantics=%u operation=%u outcome=%u lane_mask=0x%llx "
           "success_lane_mask=0x%llx",
           static_cast<unsigned long long>(input.reader), i, record.event_index,
           static_cast<uint32_t>(record.kind), record.owner_id,
           static_cast<unsigned long long>(record.generation), record.epoch, record.workgroup_x,
           record.workgroup_y, record.workgroup_z, record.instruction_offset,
           static_cast<unsigned long long>(record.atomic_address), record.scope, record.semantics,
           static_cast<uint32_t>(record.operation), static_cast<uint32_t>(record.outcome),
           static_cast<unsigned long long>(record.lane_mask),
           static_cast<unsigned long long>(record.success_lane_mask));
  }
  const uint32_t diagnostic_sample_count = auto_moi_report_detail_count(visible_diagnostics);
  for (uint32_t i = 0; i < diagnostic_sample_count; ++i) {
    const ConSanMoiDiagnosticRecord &record = decoded.diagnostics[i];
    append(
        kDetail,
        "ConSan MOI auto diagnostic reader=%llu index=%u backend=%u kind=%u "
        "generation=%llu backend_key=%u first_epoch=%u second_epoch=%u "
        "first_owner=%u second_owner=%u first_inst=0x%x second_inst=0x%x "
        "first_kind=%u second_kind=%u first_lanes=0x%llx second_lanes=0x%llx "
        "first_lds=[%u,%u) second_lds=[%u,%u)",
        static_cast<unsigned long long>(input.reader), i, record.backend, record.kind,
        static_cast<unsigned long long>(record.generation), record.reserved, record.first_epoch,
        record.epoch, record.first_owner_id, record.second_owner_id,
        record.first_instruction_offset, record.second_instruction_offset, record.first_access_kind,
        record.second_access_kind, static_cast<unsigned long long>(record.first_lane_mask),
        static_cast<unsigned long long>(record.second_lane_mask), record.first_lds_byte_offset,
        record.first_lds_byte_offset + record.first_lds_byte_count, record.second_lds_byte_offset,
        record.second_lds_byte_offset + record.second_lds_byte_count);
  }
  rendered.insert(rendered.end(), std::make_move_iterator(mode_rendering.details.begin()),
                  std::make_move_iterator(mode_rendering.details.end()));
  return rendered;
}

} // namespace rocjitsu::consan_hook
