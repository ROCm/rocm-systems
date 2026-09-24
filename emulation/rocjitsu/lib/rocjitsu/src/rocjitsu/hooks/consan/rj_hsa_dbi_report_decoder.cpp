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
    ++result.summary.incomplete_snapshot_count;
    return result;
  }
  const void *report_ptr = snapshot.bytes.data();

  const auto *header = static_cast<const ReportHeader *>(report_ptr);
  if (!report_header_is_current(*header)) {
    result.header = *header;
    result.failure = ReportDecodeFailure::InvalidHeader;
    ++result.summary.malformed_snapshot_count;
    return result;
  }
  if (input.expected_generation && header->generation != *input.expected_generation) {
    result.header = *header;
    result.failure = ReportDecodeFailure::GenerationMismatch;
    ++result.summary.stale_snapshot_count;
    return result;
  }
  const ReportBufferLayout &expected_layout = input.layout;
  if (!report_layout_matches_header(*header, expected_layout, input.size)) {
    result.header = *header;
    result.failure = ReportDecodeFailure::LayoutMismatch;
    ++result.summary.malformed_snapshot_count;
    return result;
  }
  result.failure = ReportDecodeFailure::None;
  result.header = *header;
  result.records = decode_evidence(input, *header, snapshot.bytes, summary);
  const auto *publications = reinterpret_cast<const PublicationRecord *>(
      snapshot.bytes.data() + expected_layout.publication_events_offset);
  result.publications =
      decode_publications(*header, {publications, expected_layout.publication_event_capacity});
  if (result.publications.status == PublicationDecodeStatus::Malformed ||
      result.publications.status == PublicationDecodeStatus::Incomplete) {
    result.failure = ReportDecodeFailure::PublicationEvidenceInvalid;
    if (result.publications.status == PublicationDecodeStatus::Malformed)
      ++summary.malformed_snapshot_count;
    else
      ++summary.incomplete_snapshot_count;
  }
  return result;
}

} // namespace rocjitsu::consan::hook
