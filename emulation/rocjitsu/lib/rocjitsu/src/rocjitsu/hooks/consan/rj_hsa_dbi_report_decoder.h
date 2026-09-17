// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/hooks/consan/rj_hsa_dbi_evidence_decoder.h"

#include <cstdint>
#include <vector>

namespace rocjitsu::consan::hook {

enum class ReportDecodeFailure : uint8_t {
  None,
  SnapshotTooSmall,
  InvalidHeader,
  LayoutMismatch,
};

struct DecodedReport {
  ReportDecodeFailure failure = ReportDecodeFailure::None;
  ReportHeader header;
  ReportSummary summary;
  DecodedEvidence records;

  [[nodiscard]] bool complete() const { return failure == ReportDecodeFailure::None; }
};

[[nodiscard]] DecodedReport decode_report(const ReportPipelineInput &input,
                                          const ReportSnapshot &snapshot,
                                          ReportSummary initial_summary);

} // namespace rocjitsu::consan::hook
