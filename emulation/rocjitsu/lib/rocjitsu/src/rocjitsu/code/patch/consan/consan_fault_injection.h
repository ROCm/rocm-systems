// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_fault_injection.h
/// @brief Exact fault-mutation planning and application boundary.

#pragma once

#include "rocjitsu/code/patch/consan/consan.h"
#include "rocjitsu/code/patch/consan/consan_fault_planning.h"

#include <span>

namespace rocjitsu {

class AmdGpuCodeObject;

/// Resolve the requested faults against one pristine inventory. A dry run
/// publishes only diagnostic plans; an applicable transaction continues from
/// those exact resolutions into mutation before returning.
void resolve_consan_fault_mutations(const AmdGpuCodeObject &code_object, rj_code_arch_t arch,
                                    const ConSanOptions &options, bool apply,
                                    bool require_applicable_plan,
                                    bool require_exactly_one_applied,
                                    ConSanTransformArtifacts &result);

void try_apply_proof_nop_patch(const AmdGpuCodeObject &code_object, rj_code_arch_t arch,
                               bool force_trampoline, ConSanTransformArtifacts &result);
void try_apply_proof_endpgm_patch(const AmdGpuCodeObject &code_object, rj_code_arch_t arch,
                                  ConSanTransformArtifacts &result);
void try_apply_lds_endpgm_patch(const AmdGpuCodeObject &code_object, rj_code_arch_t arch,
                                ConSanTransformArtifacts &result);

} // namespace rocjitsu
