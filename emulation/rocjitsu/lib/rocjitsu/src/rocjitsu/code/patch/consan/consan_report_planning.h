// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_report_planning.h
/// @brief Shared report-layout mechanics composed by mode-owned ABI planners.

#pragma once

#include "rocjitsu/code/patch/consan/consan_report_contract.h"

#include <initializer_list>

namespace rocjitsu::consan::detail {

[[nodiscard]] bool checked_report_capacity(uint64_t count, uint32_t &capacity);

struct ReportRegionPlan {
  uint64_t count, element_size, alignment;
  uint32_t *capacity;
  size_t *offset;
};

template <typename Element>
[[nodiscard]] ReportRegionPlan report_region(uint64_t count, uint32_t &capacity, size_t &offset) {
  return {count, sizeof(Element), alignof(Element), &capacity, &offset};
}

[[nodiscard]] bool plan_report_regions(std::initializer_list<ReportRegionPlan> regions,
                                       AutoReportPlan &plan, uint64_t &cursor);

[[nodiscard]] EvidenceRequirementReason validate_evidence_intents(const ObservationPlan &plan,
                                                                  Mode expected_mode);

[[nodiscard]] std::vector<const ProbeIntent *>
accumulate_evidence_counts(const ObservationPlan &plan,
                           std::optional<uint64_t> maximum_access_probe_count,
                           AutoReportInventory &inventory);

void publish_evidence_requirements(ReportRequirements &requirements, AutoReportInventory inventory,
                                   uint64_t caller_ceiling_bytes);

[[nodiscard]] bool plan_report_layout(const AutoReportInventory &inventory, AutoReportPlan &plan,
                                      uint64_t &cursor);
[[nodiscard]] std::optional<AutoReportInventory>
reconstruct_report_inventory(const ReportBufferLayout &candidate);

} // namespace rocjitsu::consan::detail
