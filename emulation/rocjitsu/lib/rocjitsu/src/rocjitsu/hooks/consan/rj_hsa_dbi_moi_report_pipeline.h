// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/hooks/consan/rj_hsa_dbi_moi_report_snapshot.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_moi_report_trust.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace rocjitsu::consan_hook {

struct AutoMoiRecordReplayStaticMapping {
  uint32_t instruction_offset = 0;
  std::vector<uint64_t> owner_descriptor_file_offsets;
  bool owner_provenance_complete = false;
};

struct AutoMoiSampledStaticMapping {
  uint32_t first_slot = 0;
  uint32_t range_count = 0;
  uint32_t bank_count = 0;
  uint64_t instruction_offset = 0;
  uint64_t emitted_probe_offset = 0;
  uint64_t relocated_guest_offset = 0;
  std::optional<uint16_t> scratch_vgpr;
  std::vector<uint64_t> owner_descriptor_file_offsets;
  bool owner_provenance_complete = false;
};

/// Immutable runtime/static context paired with one captured report. It owns
/// no HSA handles and borrows only registry metadata for the duration of the
/// synchronous report pipeline.
struct AutoMoiReportPipelineInput {
  uint64_t reader = 0;
  uint64_t source_address = 0;
  size_t size = 0;
  ConSanMoiReportBufferLayout layout;
  bool fine_grained = false;
  std::string_view input_fingerprint;
  std::span<const AutoMoiRecordReplayStaticMapping> record_replay_static_mappings;
  std::span<const AutoMoiSampledStaticMapping> sampled_static_mappings;
  uint32_t compact_token_mapping_count = 0;
  bool compact_token_mapping_malformed = false;
  bool record_replay_static_mapping_malformed = false;
  bool sampled_static_mapping_malformed = false;
};

[[nodiscard]] AutoMoiReportSummary
summarize_auto_moi_report(const AutoMoiReportPipelineInput &input,
                          const AutoMoiReportSnapshot &snapshot,
                          AutoMoiReportSummary initial_summary);

} // namespace rocjitsu::consan_hook
