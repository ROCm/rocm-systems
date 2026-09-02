// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/hooks/consan/rj_hsa_dbi_moi_inline_shadow_report_decoder.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_moi_record_replay_report_decoder.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_moi_sampled_report_decoder.h"

#include <array>
#include <cstdint>
#include <variant>
#include <vector>

namespace rocjitsu::consan_hook {

enum class AutoMoiReportDecodeFailure : uint8_t {
  None,
  SnapshotTooSmall,
  InvalidHeader,
  LayoutMismatch,
};

enum class AutoMoiReportEvidenceReason : uint8_t {
  ExactMalformed,
  ReleasePublishing,
  SampledMalformedWindow,
  SampledEmptyWatchpoint,
  SampledMalformedWatchpoint,
  CompactDiagnosticTokenUnresolved,
};

/// Typed malformed/incomplete evidence. `words` preserve bounded raw detail
/// for stable rendering without allowing the renderer to reinterpret bytes.
struct AutoMoiReportEvidenceIssue {
  AutoMoiReportEvidenceReason reason = AutoMoiReportEvidenceReason::ExactMalformed;
  uint32_t index = 0;
  std::array<uint64_t, 12> words{};
};

using AutoMoiModeDecodedReport =
    std::variant<AutoMoiRecordReplayDecodedReport, AutoMoiInlineShadowDecodedReport,
                 AutoMoiSampledDecodedReport>;

struct AutoMoiDecodedReport {
  AutoMoiReportDecodeFailure failure = AutoMoiReportDecodeFailure::None;
  ConSanMoiReportHeader header;
  AutoMoiReportSummary summary;
  AutoMoiModeDecodedReport mode;

  std::vector<ConSanMoiAccessRecord> visible_access_slots;
  std::vector<ConSanMoiBarrierRecord> barrier_records;
  std::vector<ConSanMoiAtomicRecord> atomic_records;
  std::vector<ConSanMoiFenceRecord> fence_records;
  std::vector<ConSanMoiDiagnosticRecord> diagnostics;
  std::vector<AutoMoiReportEvidenceIssue> issues;

  [[nodiscard]] bool complete() const { return failure == AutoMoiReportDecodeFailure::None; }
};

[[nodiscard]] AutoMoiDecodedReport decode_auto_moi_report(const AutoMoiReportPipelineInput &input,
                                                          const AutoMoiReportSnapshot &snapshot,
                                                          AutoMoiReportSummary initial_summary);

} // namespace rocjitsu::consan_hook
