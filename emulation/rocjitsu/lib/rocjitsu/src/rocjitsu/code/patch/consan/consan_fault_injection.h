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

/// Validate and apply one complete set of typed fault plans. Exact mutation
/// mechanisms and their composition order are private to the fault component.
void apply_consan_fault_mutations(const AmdGpuCodeObject &code_object, rj_code_arch_t arch,
                                  const ConSanOptions &context,
                                  std::span<const ConSanFaultMutationPlan> plans,
                                  ConSanTransformArtifacts &result);

void try_apply_proof_nop_patch(const AmdGpuCodeObject &code_object, rj_code_arch_t arch,
                               bool force_trampoline, ConSanTransformArtifacts &result);
void try_apply_proof_endpgm_patch(const AmdGpuCodeObject &code_object, rj_code_arch_t arch,
                                  ConSanTransformArtifacts &result);
void try_apply_lds_endpgm_patch(const AmdGpuCodeObject &code_object, rj_code_arch_t arch,
                                ConSanTransformArtifacts &result);

} // namespace rocjitsu
