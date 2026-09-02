// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/hooks/consan/rj_hsa_dbi_moi_report_decoder.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_replay_provenance.h"

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace rocjitsu::consan_hook {

struct RecordReplayPressureTelemetry {
  bool available = false;
  bool saturated = false;
  enum class UnavailableReason : uint8_t {
    None,
    NoDispatchDirectory,
    NoAccessTable,
    NoLogicalAccessRanges,
  } unavailable_reason = UnavailableReason::None;
  uint64_t occupied_access_record_count = 0;
  uint64_t access_record_capacity = 0;
  uint64_t observed_site_count = 0;
  uint64_t maximum_site_owner_address_group_count = 0;
  uint32_t address_group_headroom = 0;
  uint32_t logical_access_range_count = 0;
  uint32_t maximum_site_token = 0;
  uint64_t invalid_site_token_count = 0;
};

[[nodiscard]] uint64_t record_replay_bank_saturation_count(const ConSanMoiReportHeader &header);
[[nodiscard]] std::string_view record_replay_pressure_unavailable_reason_name(
    RecordReplayPressureTelemetry::UnavailableReason reason);
[[nodiscard]] RecordReplayPressureTelemetry record_replay_pressure_telemetry(
    const ConSanMoiReportHeader &header, std::span<const ConSanMoiAccessRecord> records,
    uint32_t logical_access_range_count, uint32_t address_group_headroom);

struct AutoMoiRecordReplayAnalysis {
  bool replay_performed = false;
  bool shadow_bounded = false;
  bool effective_conflict = false;
  uint64_t required_shadow_entry_count = 0;
  uint64_t replay_shadow_entry_count = 0;
  uint32_t effective_diagnostic_count = 0;
  uint32_t disjoint_owner_suppressed_count = 0;
  ConSanMoiReportHeader replay_header;
  ConSanMoiRecordReplayResult replay;
  ConSanMoiReplayProvenanceRepair provenance;
  RecordReplayPressureTelemetry pressure;
  std::vector<ConSanMoiDiagnosticRecord> diagnostics;
};

[[nodiscard]] AutoMoiRecordReplayAnalysis
analyze_auto_moi_record_replay(const ConSanMoiReportHeader &header,
                               std::span<const ConSanMoiAccessRecord> access_records,
                               std::span<const ConSanMoiBarrierRecord> barrier_records,
                               std::span<const ConSanMoiAtomicRecord> atomic_records,
                               std::span<const ConSanMoiFenceRecord> fence_records,
                               std::span<const AutoMoiRecordReplayStaticMapping> static_mappings,
                               bool static_mapping_malformed, uint32_t logical_access_range_count,
                               uint32_t address_group_headroom);

void accumulate_auto_moi_record_replay_analysis(AutoMoiReportSummary &summary,
                                                const AutoMoiRecordReplayAnalysis &analysis);

} // namespace rocjitsu::consan_hook
