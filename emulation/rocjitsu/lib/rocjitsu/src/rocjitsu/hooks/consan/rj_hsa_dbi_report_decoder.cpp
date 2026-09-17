// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rj_hsa_dbi_report_decoder.h"

#include <algorithm>
#include <utility>

namespace rocjitsu::consan::hook {

DecodedReport decode_report(const ReportPipelineInput &input, const ReportSnapshot &snapshot,
                            ReportSummary initial_summary) {
  DecodedReport result;
  result.summary = initial_summary;
  ReportSummary &summary = result.summary;
  if (snapshot.bytes.size() < sizeof(ReportHeader) || snapshot.bytes.size() < input.size) {
    result.failure = ReportDecodeFailure::SnapshotTooSmall;
    return result;
  }
  const void *report_ptr = snapshot.bytes.data();

  const auto *header = static_cast<const ReportHeader *>(report_ptr);
  if (!report_header_is_current(*header)) {
    result.header = *header;
    result.failure = ReportDecodeFailure::InvalidHeader;
    return result;
  }
  const ReportBufferLayout &expected_layout = input.layout;
  if (!report_layout_matches_header(*header, expected_layout, input.size)) {
    result.header = *header;
    result.failure = ReportDecodeFailure::LayoutMismatch;
    return result;
  }
  result.failure = ReportDecodeFailure::None;
  result.header = *header;
  result.records = decode_evidence(input, *header, snapshot.bytes, summary);
  return result;
}

} // namespace rocjitsu::consan::hook
