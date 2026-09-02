// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rj_hsa_dbi_moi_report_decoder.h"

#include <algorithm>
#include <utility>

namespace rocjitsu::consan_hook {

AutoMoiDecodedReport decode_auto_moi_report(const AutoMoiReportPipelineInput &input,
                                            const AutoMoiReportSnapshot &snapshot,
                                            AutoMoiReportSummary initial_summary) {
  AutoMoiDecodedReport result;
  result.summary = initial_summary;
  AutoMoiReportSummary &summary = result.summary;
  if (snapshot.bytes.size() < sizeof(ConSanMoiReportHeader) || snapshot.bytes.size() < input.size) {
    result.failure = AutoMoiReportDecodeFailure::SnapshotTooSmall;
    return result;
  }
  const ConSanMoiEngine expected_engine = input.layout.engine;
  const void *report_ptr = snapshot.bytes.data();

  const auto *header = static_cast<const rocjitsu::ConSanMoiReportHeader *>(report_ptr);
  if (!rocjitsu::consan_moi_report_header_is_current(*header)) {
    result.header = *header;
    result.failure = AutoMoiReportDecodeFailure::InvalidHeader;
    return result;
  }
  const rocjitsu::ConSanMoiReportBufferLayout &expected_layout = input.layout;
  if (!rocjitsu::consan_moi_report_layout_matches_header(*header, expected_layout, expected_engine,
                                                         input.size)) {
    result.header = *header;
    result.failure = AutoMoiReportDecodeFailure::LayoutMismatch;
    return result;
  }
  const uint32_t access_record_count = header->access_record_count;
  const uint32_t barrier_record_count = header->barrier_record_count;
  const uint32_t atomic_record_count = header->atomic_record_count;
  const uint32_t visible_records = std::min(access_record_count, header->access_record_capacity);
  const uint32_t visible_barriers = std::min(barrier_record_count, header->barrier_record_capacity);
  const uint32_t visible_atomics = std::min(atomic_record_count, header->atomic_record_capacity);
  const uint32_t visible_fences =
      std::min(header->fence_record_count, expected_layout.fence_record_capacity);
  const uint32_t raw_visible_diagnostics =
      std::min(header->diagnostic_count, header->diagnostic_capacity);
  const uint32_t dropped_records =
      access_record_count > visible_records ? access_record_count - visible_records : 0;
  const uint32_t dropped_barriers =
      barrier_record_count > visible_barriers ? barrier_record_count - visible_barriers : 0;
  const uint32_t dropped_atomics =
      atomic_record_count > visible_atomics ? atomic_record_count - visible_atomics : 0;
  const uint32_t dropped_fences =
      expected_layout.fence_record_capacity != 0 && header->fence_record_count > visible_fences
          ? header->fence_record_count - visible_fences
          : 0;
  const uint32_t dropped_diagnostics = header->diagnostic_count > raw_visible_diagnostics
                                           ? header->diagnostic_count - raw_visible_diagnostics
                                           : 0;
  const auto *bytes = static_cast<const uint8_t *>(report_ptr);
  const auto *records = reinterpret_cast<const ConSanMoiAccessRecord *>(
      bytes + expected_layout.access_records_offset);
  result.visible_access_slots.assign(records, records + visible_records);

  const auto *barriers = reinterpret_cast<const ConSanMoiBarrierRecord *>(
      bytes + expected_layout.barrier_records_offset);
  result.barrier_records.assign(barriers, barriers + visible_barriers);
  const auto *atomics = reinterpret_cast<const ConSanMoiAtomicRecord *>(
      bytes + expected_layout.atomic_records_offset);
  result.atomic_records.assign(atomics, atomics + visible_atomics);
  const auto *fences =
      reinterpret_cast<const ConSanMoiFenceRecord *>(bytes + expected_layout.fence_records_offset);
  result.fence_records.assign(fences, fences + visible_fences);

  const auto *raw_diagnostics = reinterpret_cast<const ConSanMoiDiagnosticRecord *>(
      bytes + expected_layout.diagnostic_records_offset);
  result.diagnostics.assign(raw_diagnostics, raw_diagnostics + raw_visible_diagnostics);

  summary.visible_barrier_record_count = visible_barriers;
  summary.visible_atomic_record_count = visible_atomics;
  summary.visible_fence_record_count = visible_fences;
  summary.dropped_access_record_count = dropped_records;
  summary.dropped_barrier_record_count = dropped_barriers;
  summary.dropped_atomic_record_count = dropped_atomics;
  summary.dropped_fence_record_count = dropped_fences;
  summary.dropped_diagnostic_record_count = dropped_diagnostics;
  result.failure = AutoMoiReportDecodeFailure::None;
  result.header = *header;
  switch (expected_engine) {
  case ConSanMoiEngine::RecordReplay:
    result.mode = decode_auto_moi_record_replay_report(result.visible_access_slots,
                                                       header->event_counter, summary);
    break;
  case ConSanMoiEngine::InlineShadow: {
    AutoMoiInlineShadowDecodeResult inline_result = decode_auto_moi_inline_shadow_report(
        input, *header, snapshot.bytes, raw_visible_diagnostics, summary);
    result.diagnostics = std::move(inline_result.diagnostics);
    result.mode = std::move(inline_result.decoded);
  } break;
  case ConSanMoiEngine::Sampled:
    result.mode = decode_auto_moi_sampled_report(input, *header, snapshot.bytes, summary);
    break;
  }
  summary.visible_diagnostic_record_count = result.diagnostics.size();
  return result;
}

} // namespace rocjitsu::consan_hook
