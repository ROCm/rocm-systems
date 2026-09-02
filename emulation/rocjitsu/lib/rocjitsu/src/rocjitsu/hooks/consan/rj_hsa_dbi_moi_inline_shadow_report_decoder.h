// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/hooks/consan/rj_hsa_dbi_moi_report_pipeline.h"

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace rocjitsu::consan_hook {

struct AutoMoiExactShadowEvidence {
  uint32_t index = 0;
  ConSanMoiExactShadowEntry entry;
  ConSanMoiExactByteCellProvenance byte_provenance;
  uint64_t dispatch_id = 0;
  uint32_t version = 0;
};

struct AutoMoiInlineAtomicReleaseEvidence {
  uint32_t index = 0;
  ConSanMoiInlineAtomicReleaseSlot slot;
  ConSanMoiInlineCausalSnapshot snapshot;
};

struct AutoMoiInlineAcquiredTokenEvidence {
  uint32_t index = 0;
  ConSanMoiInlineAcquiredEpochTokenSlot token;
};

enum class AutoMoiInlineShadowEvidenceReason : uint8_t {
  ExactMalformed,
  ReleasePublishing,
  CompactDiagnosticTokenUnresolved,
};

struct AutoMoiInlineShadowEvidenceIssue {
  AutoMoiInlineShadowEvidenceReason reason = AutoMoiInlineShadowEvidenceReason::ExactMalformed;
  uint32_t index = 0;
  std::array<uint64_t, 6> words{};
};

struct AutoMoiInlineShadowDecodedReport {
  uint32_t deferred_token_qualified_diagnostic_count = 0;
  std::vector<AutoMoiExactShadowEvidence> exact_shadow;
  std::vector<AutoMoiInlineAtomicReleaseEvidence> atomic_releases;
  std::vector<AutoMoiInlineAcquiredTokenEvidence> acquired_tokens;
  std::vector<AutoMoiInlineShadowEvidenceIssue> issues;
};

struct AutoMoiInlineShadowDecodeResult {
  AutoMoiInlineShadowDecodedReport decoded;
  std::vector<ConSanMoiDiagnosticRecord> diagnostics;
};

[[nodiscard]] AutoMoiInlineShadowDecodeResult decode_auto_moi_inline_shadow_report(
    const AutoMoiReportPipelineInput &input, const ConSanMoiReportHeader &header,
    std::span<const uint8_t> report_bytes, uint32_t raw_visible_diagnostic_count,
    AutoMoiReportSummary &summary);

} // namespace rocjitsu::consan_hook
