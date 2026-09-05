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

[[nodiscard]] bool kernel_owns_patch(const ConSanProgramContainer &kernel,
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
