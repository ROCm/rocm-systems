// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/patch/consan/consan_moi_report_emission.h"

#include <vector>

namespace rocjitsu::consan_moi_detail {

struct DynamicRecordLayout {
  uint32_t stride_bytes;
  uint16_t value_vgpr_offset;
};

inline constexpr DynamicRecordLayout kBarrierRecordLayout = {sizeof(ConSanMoiBarrierRecord), 5};
inline constexpr DynamicRecordLayout kAccessRecordLayout = {sizeof(ConSanMoiAccessRecord), 5};
inline constexpr DynamicRecordLayout kAtomicRecordLayout = {sizeof(ConSanMoiAtomicRecord), 4};
inline constexpr DynamicRecordLayout kFenceRecordLayout = {sizeof(ConSanMoiFenceRecord), 5};
inline constexpr DynamicRecordLayout kDiagnosticRecordLayout = {sizeof(ConSanMoiDiagnosticRecord),
                                                                4};

[[nodiscard]] bool append_atomic_fetch_add_one_u32(std::vector<uint32_t> &words,
                                                   uint64_t counter_address, uint16_t result_vgpr,
                                                   uint16_t scratch_vgpr, rj_code_arch_t arch);
[[nodiscard]] bool append_atomic_or_u32_literal(std::vector<uint32_t> &words, uint64_t address,
                                                uint32_t value, uint16_t scratch_vgpr,
                                                rj_code_arch_t arch);
[[nodiscard]] bool append_atomic_load_u32(std::vector<uint32_t> &words, uint16_t address_vgpr,
                                          uint16_t result_vgpr, rj_code_arch_t arch);
[[nodiscard]] bool
append_publish_visible_evidence_if_zero(std::vector<uint32_t> &words, uint64_t counter_address,
                                        uint16_t result_vgpr, uint16_t scratch_vgpr,
                                        uint16_t exec_save_sgpr, rj_code_arch_t arch);
[[nodiscard]] bool append_dynamic_record_address(std::vector<uint32_t> &words,
                                                 const DynamicRecordLayout &layout,
                                                 uint64_t field_address, uint16_t slot_vgpr,
                                                 uint16_t scratch_vgpr, rj_code_arch_t arch);
[[nodiscard]] bool append_dynamic_record_store_u32_vgpr(std::vector<uint32_t> &words,
                                                        const DynamicRecordLayout &layout,
                                                        uint64_t field_address, uint16_t value_vgpr,
                                                        uint16_t slot_vgpr, uint16_t scratch_vgpr,
                                                        rj_code_arch_t arch);
[[nodiscard]] bool append_dynamic_record_store_u32_literal(
    std::vector<uint32_t> &words, const DynamicRecordLayout &layout, uint64_t field_address,
    uint32_t value, uint16_t slot_vgpr, uint16_t scratch_vgpr, rj_code_arch_t arch);
[[nodiscard]] bool append_dynamic_record_store_u32_scalar_src(
    std::vector<uint32_t> &words, const DynamicRecordLayout &layout, uint64_t field_address,
    uint16_t scalar_src, uint16_t slot_vgpr, uint16_t scratch_vgpr, rj_code_arch_t arch);
[[nodiscard]] bool append_dynamic_record_store_moi_report_dispatch_id_pair(
    std::vector<uint32_t> &words, const DynamicRecordLayout &layout, uint64_t low_field_address,
    const ConSanMoiReportDispatchIdSources &sources, uint16_t slot_vgpr, uint16_t scratch_vgpr,
    rj_code_arch_t arch);
[[nodiscard]] bool append_dynamic_record_store_workgroup_source(
    std::vector<uint32_t> &words, const DynamicRecordLayout &layout, uint64_t field_address,
    const ConSanMoiWorkgroupSource &source, uint16_t slot_vgpr, uint16_t scratch_vgpr,
    rj_code_arch_t arch);
[[nodiscard]] bool append_dynamic_record_event_index_store(
    std::vector<uint32_t> &words, const DynamicRecordLayout &layout, uint64_t counter_address,
    uint64_t field_address, uint16_t slot_vgpr, uint16_t scratch_vgpr, rj_code_arch_t arch);
[[nodiscard]] bool append_dynamic_diagnostic_record_address(std::vector<uint32_t> &words,
                                                            uint64_t address, uint16_t slot,
                                                            uint16_t address_vgpr,
                                                            rj_code_arch_t arch);

} // namespace rocjitsu::consan_moi_detail
