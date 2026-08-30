// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_fault_injection.h
/// @brief Exact fault-mutation planning and application boundary.

#pragma once

#include "rocjitsu/code/patch/consan/consan.h"
#include "rocjitsu/code/patch/consan/consan_fault_planning.h"

#include <cstddef>

namespace rocjitsu {

class AmdGpuCodeObject;

[[nodiscard]] size_t applied_fault_mutation_count(const ConSanTransformArtifacts &result);

void try_apply_barrier_drop_fault_patch(const AmdGpuCodeObject &code_object, rj_code_arch_t arch,
                                        const ConSanOptions &options,
                                        ConSanTransformArtifacts &result);
void try_apply_barrier_id_scope_fault_patch(const AmdGpuCodeObject &code_object,
                                            const ConSanFaultMutationPlan &plan,
                                            ConSanTransformArtifacts &result);
void try_apply_barrier_participant_fault_patch(const AmdGpuCodeObject &code_object,
                                               const ConSanFaultMutationPlan &plan,
                                               ConSanTransformArtifacts &result);
void try_apply_barrier_move_fault_patch(const AmdGpuCodeObject &code_object, rj_code_arch_t arch,
                                        const ConSanOptions &options,
                                        ConSanTransformArtifacts &result);
void try_apply_atomic_fault_patch(const AmdGpuCodeObject &code_object, rj_code_arch_t arch,
                                  const ConSanOptions &options, ConSanTransformArtifacts &result);
void try_apply_lds_fault_patch(const AmdGpuCodeObject &code_object, rj_code_arch_t arch,
                               const ConSanFaultMutationPlan &plan,
                               ConSanTransformArtifacts &result);
void try_apply_ordinary_fault_patch(const AmdGpuCodeObject &code_object,
                                    const ConSanOptions &options, ConSanTransformArtifacts &result);

void try_apply_proof_nop_patch(const AmdGpuCodeObject &code_object, rj_code_arch_t arch,
                               bool force_trampoline, ConSanTransformArtifacts &result);
void try_apply_proof_endpgm_patch(const AmdGpuCodeObject &code_object, rj_code_arch_t arch,
                                  ConSanTransformArtifacts &result);
void try_apply_lds_endpgm_patch(const AmdGpuCodeObject &code_object, rj_code_arch_t arch,
                                ConSanTransformArtifacts &result);

} // namespace rocjitsu
