// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_moi_report_planning.h
/// @brief Shared report-layout mechanics composed by mode-owned ABI planners.

#pragma once

#include "rocjitsu/code/patch/consan/consan_moi.h"

#include <initializer_list>

namespace rocjitsu::consan_moi_impl {

[[nodiscard]] bool checked_moi_report_capacity(uint64_t count, uint32_t &capacity);

struct MoiReportRegionPlan {
  uint64_t count, element_size, alignment;
  uint32_t *capacity;
  size_t *offset;
};

template <typename Element>
[[nodiscard]] MoiReportRegionPlan moi_report_region(uint64_t count, uint32_t &capacity,
                                                    size_t &offset) {
  return {count, sizeof(Element), alignof(Element), &capacity, &offset};
}

[[nodiscard]] bool plan_moi_report_regions(std::initializer_list<MoiReportRegionPlan> regions,
                                           ConSanMoiAutoReportPlan &plan, uint64_t &cursor);

[[nodiscard]] ConSanEvidenceRequirementReason
validate_moi_evidence_intents(const ConSanEvidenceIntentPlan &plan,
                              ConSanCapabilityEngine expected_engine);

[[nodiscard]] std::vector<const ConSanEvidenceIntent *>
accumulate_moi_evidence_counts(const ConSanEvidenceIntentPlan &plan,
                               std::optional<uint64_t> maximum_access_probe_count,
                               ConSanMoiAutoReportInventory &inventory);

void publish_moi_evidence_requirements(ConSanMoiEvidenceRequirements &requirements,
                                       ConSanMoiAutoReportInventory inventory,
                                       uint64_t caller_ceiling_bytes);

[[nodiscard]] bool plan_record_replay_report_layout(const ConSanMoiAutoReportInventory &inventory,
                                                    ConSanMoiAutoReportPlan &plan,
                                                    uint64_t &cursor);
[[nodiscard]] std::optional<ConSanMoiAutoReportInventory>
reconstruct_record_replay_report_inventory(const ConSanMoiReportBufferLayout &candidate);

[[nodiscard]] bool plan_sampled_report_layout(const ConSanMoiAutoReportInventory &inventory,
                                              ConSanMoiAutoReportPlan &plan, uint64_t &cursor);
[[nodiscard]] std::optional<ConSanMoiAutoReportInventory>
reconstruct_sampled_report_inventory(const ConSanMoiReportBufferLayout &candidate);

[[nodiscard]] bool plan_inline_shadow_report_layout(const ConSanMoiAutoReportInventory &inventory,
                                                    ConSanMoiAutoReportPlan &plan,
                                                    uint64_t &cursor);
[[nodiscard]] std::optional<ConSanMoiAutoReportInventory>
reconstruct_inline_shadow_report_inventory(const ConSanMoiReportBufferLayout &candidate);

} // namespace rocjitsu::consan_moi_impl
