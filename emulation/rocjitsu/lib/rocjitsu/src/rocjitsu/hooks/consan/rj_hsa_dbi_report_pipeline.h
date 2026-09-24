// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/hooks/consan/rj_hsa_dbi_report_snapshot.h"
#include "rocjitsu/hooks/consan/rj_hsa_dbi_report_trust.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace rocjitsu::consan::hook {

struct AccessStaticMapping {
  uint32_t first_slot = 0;
  uint32_t range_count = 0;
  uint32_t bank_count = 0;
  uint64_t instruction_offset = 0;
  uint64_t emitted_probe_offset = 0;
  uint64_t relocated_guest_offset = 0;
  std::optional<uint16_t> scratch_vgpr;
  std::vector<uint32_t> owner_kernel_ids;
  bool owner_provenance_complete = false;
  bool uniform_lds_store = false;
};

struct AccessStaticMetadata {
  std::vector<AccessStaticMapping> mappings;
  bool malformed = false;
};

/// Immutable runtime/static context paired with one captured report. It owns
/// no HSA handles and borrows only registry metadata for the duration of the
/// synchronous report pipeline.
struct ReportPipelineInput {
  uint64_t reader = 0;
  uint64_t source_address = 0;
  size_t size = 0;
  ReportBufferLayout layout;
  bool fine_grained = false;
  std::string_view input_fingerprint;
  const AccessStaticMetadata *static_metadata = nullptr;
  uint32_t conflict_example_limit = 8;
  /// Opt-in suppression of statically proven same-instruction uniform LDS writes.
  bool allow_uniform_lds_stores = false;
  /// Allocation identity from host-owned registry state, not captured bytes.
  std::optional<uint64_t> expected_generation = std::nullopt;
  /// Host lifetime witness: no potentially interfering concurrent dispatch.
  bool publication_dispatches_isolated = true;
};

struct ReportPipelineResult {
  ReportSummary summary;
  bool complete = false;
  uint32_t conflict_example_count = 0;
};

/// Decode, analyze, and render one quiescent automatic report. `complete` is
/// false when the captured bytes cannot be decoded against the immutable
/// layout contract. Lifecycle code must not recycle such an epoch.
[[nodiscard]] ReportPipelineResult process_report(const ReportPipelineInput &input,
                                                  const ReportSnapshot &snapshot,
                                                  ReportSummary initial_summary = {});

} // namespace rocjitsu::consan::hook
