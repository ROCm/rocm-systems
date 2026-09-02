// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/patch/consan/consan_moi_internal.h"
#include "rocjitsu/code/patch/consan/consan_moi_mode_planning.h"
#include "rocjitsu/code/patch/consan/consan_moi_record_event_emission.h"
#include "rocjitsu/code/patch/consan/consan_moi_relocation.h"

namespace rocjitsu::consan_moi_impl {

[[nodiscard]] bool append_moi_entry_salu_write(std::vector<uint32_t> &words, uint32_t word,
                                               rj_code_arch_t arch);

/// Entry-island transaction used when a spill-backed Record/Replay probe must
/// borrow its indirect-jump scalar tuple before its ordinary body can save it.
struct MoiBorrowedRecordReplayEntry {
  uint16_t backup_vgpr = 0u;
  std::vector<uint32_t> island_words;
  std::vector<uint32_t> scalar_restore_words;
};

[[nodiscard]] std::optional<MoiBorrowedRecordReplayEntry> build_moi_borrowed_record_replay_entry(
    uint64_t island_text_offset, uint64_t body_text_offset, uint16_t backup_vgpr,
    const ConSanMoiIndirectJumpSgprs &jump_sgprs, uint32_t island_word_count, rj_code_arch_t arch);

[[nodiscard]] bool
emit_moi_local_indirect_entry_island(std::vector<uint8_t> &text, uint64_t island_text_offset,
                                     uint64_t cave_text_offset, uint64_t anchor_text_offset,
                                     const ConSanMoiIndirectJumpSgprs &jump_sgprs,
                                     std::span<const uint64_t> owner_descriptor_file_offsets,
                                     rj_code_arch_t arch, std::vector<ConSanPatchInfo> &patches,
                                     std::vector<std::string> &errors, std::string_view context);

[[nodiscard]] bool
emit_moi_local_indirect_entry_island(std::vector<uint8_t> &text, uint64_t island_text_offset,
                                     uint64_t cave_text_offset, uint64_t anchor_text_offset,
                                     const MoiRecordEventEmissionPlan &plan,
                                     std::span<const uint64_t> owner_descriptor_file_offsets,
                                     rj_code_arch_t arch, std::vector<ConSanPatchInfo> &patches,
                                     std::vector<std::string> &errors, std::string_view context);

[[nodiscard]] bool kernel_owns_patch(const ConSanKernelInfo &kernel,
                                     const ConSanPatchLoweringProduct &patch);

[[nodiscard]] bool enable_moi_full_workgroup_id_payload(rj_code_arch_t arch,
                                                        ConSanTransformArtifacts &result);

void try_apply_owner_epoch_prologue_patch(
    std::span<const uint8_t> bytes, const ConSanOptions &options,
    const ConSanMoiOperatingPoint &operating_point,
    std::span<const ConSanMoiPrologueScratchVgprAssignment> prologue_scratch_assignments,
    const MoiObjectModeSemantics &mode_semantics, const MoiPrologueModePolicy &mode_policy,
    rj_code_arch_t arch, ConSanTransformArtifacts &result);

} // namespace rocjitsu::consan_moi_impl
