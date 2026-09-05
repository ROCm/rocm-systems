// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/patch/consan/consan_moi_report_emission.h"
#include "rocjitsu/code/patch/instruction_sequence.h"

#include <cstddef>
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

/// Transactional serializer for one dynamically indexed report record.
///
/// The record table, slot, scratch window, and target are invariants of the
/// complete record rather than properties of each field. Keeping them here
/// makes record schemas read as field/source mappings and prevents individual
/// stores from accidentally mixing layouts or bases. Any failed field rolls
/// the complete record contribution back to its construction point.
class DynamicRecordEmitter {
public:
  DynamicRecordEmitter(std::vector<uint32_t> &words, const DynamicRecordLayout &layout,
                       uint64_t record_base, uint16_t slot_vgpr, uint16_t scratch_vgpr,
                       rj_code_arch_t arch);

  DynamicRecordEmitter(const DynamicRecordEmitter &) = delete;
  DynamicRecordEmitter &operator=(const DynamicRecordEmitter &) = delete;

  DynamicRecordEmitter &vgpr(size_t field_offset, uint16_t value_vgpr);
  DynamicRecordEmitter &scalar(size_t field_offset, uint16_t scalar_src);
  DynamicRecordEmitter &literal(size_t field_offset, uint32_t value);
  DynamicRecordEmitter &dispatch_id(size_t field_offset,
                                    const ConSanMoiReportDispatchIdSource &source);
  DynamicRecordEmitter &workgroup(size_t field_offset, const ConSanMoiWorkgroupSource &source);
  DynamicRecordEmitter &event_index(size_t field_offset, uint64_t counter_address);

  [[nodiscard]] explicit operator bool() const { return static_cast<bool>(sequence_); }
  [[nodiscard]] bool finish() const { return sequence_.finish(); }

private:
  void require(bool success);
  DynamicRecordEmitter &private_value(size_t field_offset, uint32_t private_offset);

  std::vector<uint32_t> &words_;
  const DynamicRecordLayout &layout_;
  uint64_t record_base_ = 0;
  uint16_t slot_vgpr_ = 0;
  uint16_t scratch_vgpr_ = 0;
  rj_code_arch_t arch_ = ROCJITSU_CODE_ARCH_INVALID;
  InstructionSequence sequence_;
};

enum class MoiVisibleEvidencePublicationResult : uint8_t {
  Appended,
  ElectionUnsupported,
  PublicationUnsupported,
};

[[nodiscard]] bool append_atomic_fetch_add_one_u32(std::vector<uint32_t> &words,
                                                   uint64_t counter_address, uint16_t result_vgpr,
                                                   uint16_t scratch_vgpr, rj_code_arch_t arch);
/// Reserve one monotonically numbered record slot and leave VCC set for lanes
/// whose slot is within the table capacity.
[[nodiscard]] bool append_reserve_bounded_dynamic_record_slot(
    std::vector<uint32_t> &words, uint64_t counter_address, uint32_t record_capacity,
    uint16_t slot_vgpr, uint16_t capacity_vgpr, uint16_t scratch_vgpr, rj_code_arch_t arch);
[[nodiscard]] bool append_atomic_or_u32_literal(std::vector<uint32_t> &words, uint64_t address,
                                                uint32_t value, uint16_t scratch_vgpr,
                                                rj_code_arch_t arch);
[[nodiscard]] bool append_atomic_load_u32(std::vector<uint32_t> &words, uint16_t address_vgpr,
                                          uint16_t result_vgpr, rj_code_arch_t arch);
/// Select the lowest lane named by an existing EXEC mask. The current EXEC is
/// narrowed to that lane and its incoming value is retained in
/// `saved_exec_sgpr`.
[[nodiscard]] bool append_select_first_lane_in_exec_mask(std::vector<uint32_t> &words,
                                                         uint16_t lane_rank_vgpr,
                                                         uint16_t active_exec_sgpr,
                                                         uint16_t saved_exec_sgpr,
                                                         rj_code_arch_t arch);
/// Select the first currently active lane and retain the incoming EXEC mask.
/// `active_exec_sgpr` feeds the lane-rank calculation; `saved_exec_sgpr` may
/// alias it when the caller does not need a second copy after narrowing.
[[nodiscard]] bool append_select_first_active_lane(std::vector<uint32_t> &words,
                                                   uint16_t lane_rank_vgpr,
                                                   uint16_t active_exec_sgpr,
                                                   uint16_t saved_exec_sgpr, rj_code_arch_t arch);
[[nodiscard]] MoiVisibleEvidencePublicationResult
append_publish_first_active_lane_visible_evidence_if_zero(
    std::vector<uint32_t> &words, uint64_t counter_address, uint16_t result_vgpr,
    uint16_t address_vgpr, uint16_t active_exec_sgpr, uint16_t temporary_exec_sgpr,
    rj_code_arch_t arch);
[[nodiscard]] bool append_dynamic_record_address(std::vector<uint32_t> &words,
                                                 const DynamicRecordLayout &layout,
                                                 uint64_t field_address, uint16_t slot_vgpr,
                                                 uint16_t scratch_vgpr, rj_code_arch_t arch);
/// Append one VCC-guarded dynamic record and restore the EXEC mask saved by
/// the publisher election. The optional store wait is part of the skipped
/// record body, so its branch displacement is computed from the final body.
[[nodiscard]] bool append_guarded_dynamic_record(std::vector<uint32_t> &words,
                                                 std::vector<uint32_t> &record_words,
                                                 uint16_t saved_exec_sgpr,
                                                 bool wait_for_global_stores, rj_code_arch_t arch);
[[nodiscard]] bool append_dynamic_diagnostic_record_address(std::vector<uint32_t> &words,
                                                            uint64_t address, uint16_t slot,
                                                            uint16_t address_vgpr,
                                                            rj_code_arch_t arch);

} // namespace rocjitsu::consan_moi_detail
