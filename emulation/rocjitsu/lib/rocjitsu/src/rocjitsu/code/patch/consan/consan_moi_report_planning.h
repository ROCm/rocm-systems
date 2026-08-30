// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_moi_report_planning.h
/// @brief Shared report-layout mechanics composed by mode-owned ABI planners.

#pragma once

#include "rocjitsu/code/patch/consan/consan_moi.h"

namespace rocjitsu::consan_moi_impl {

[[nodiscard]] bool checked_moi_report_capacity(uint64_t count, uint32_t &capacity);

[[nodiscard]] bool append_moi_report_region(uint64_t count, uint64_t element_size,
                                            uint64_t alignment, uint64_t &cursor, size_t &offset);

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
