// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/patch/consan/consan_internal.h"
#include "rocjitsu/code/patch/consan/consan_lowering_plan.h"
#include "rocjitsu/code/patch/consan/consan_relocation.h"

namespace rocjitsu::consan::detail {

[[nodiscard]] bool append_entry_salu_write(std::vector<uint32_t> &words, uint32_t word,
                                           rj_code_arch_t arch);

[[nodiscard]] bool kernel_owns_patch(const ProgramContainer &kernel,
                                     const PatchLoweringProduct &patch);

[[nodiscard]] bool enable_full_workgroup_id_payload(rj_code_arch_t arch,
                                                    TransformArtifacts &result);

void try_apply_owner_epoch_prologue_patch(
    std::span<const uint8_t> bytes, const Request &request, const BoundRuntimeResources &resources,
    const OperatingPoint &operating_point,
    std::span<const PrologueScratchVgprAssignment> prologue_scratch_assignments,
    rj_code_arch_t arch, TransformArtifacts &result);

} // namespace rocjitsu::consan::detail
