// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// This translation unit owns placement, synchronization, mutation, and
// SuperCollider lowering. Program analysis remains a separate compiled
// component, so every analysis dependency crosses a declared contract.

#include "rocjitsu/code/patch/consan/consan_lowerer.h"
#include "rocjitsu/code/patch/consan/consan.h"

#include "rocjitsu/analysis/def_use_chain.h"
#include "rocjitsu/analysis/kernel_scope.h"
#include "rocjitsu/analysis/liveness.h"
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/major_image_ownership.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/patch/consan/consan_barrier_move_proof.h"
#include "rocjitsu/code/patch/consan/consan_branch_only_relay_router.h"
#include "rocjitsu/code/patch/consan/consan_cfg.h"
#include "rocjitsu/code/patch/consan/consan_descriptor.h"
#include "rocjitsu/code/patch/consan/consan_descriptor_growth.h"
#include "rocjitsu/code/patch/consan/consan_fault_selection.h"
#include "rocjitsu/code/patch/consan/consan_final_validation.h"
#include "rocjitsu/code/patch/consan/consan_growth_policy.h"
#include "rocjitsu/code/patch/consan/consan_input_layout.h"
#include "rocjitsu/code/patch/consan/consan_instruction_semantics.h"
#include "rocjitsu/code/patch/consan/consan_lowering.h"
#include "rocjitsu/code/patch/consan/consan_moi.h"
#include "rocjitsu/code/patch/consan/consan_moi_internal.h"
#include "rocjitsu/code/patch/consan/consan_perturbation_policy.h"
#include "rocjitsu/code/patch/consan/consan_physical_site_alias.h"
#include "rocjitsu/code/patch/consan/consan_program_analysis.h"
#include "rocjitsu/code/patch/consan/consan_relay_target_ops.h"
#include "rocjitsu/code/patch/consan/consan_resource.h"
#include "rocjitsu/code/patch/consan/consan_runtime_kernel.h"
#include "rocjitsu/code/patch/consan/consan_semantic_classifiers.h"
#include "rocjitsu/code/patch/consan/consan_sync_event_index.h"
#include "rocjitsu/code/patch/consan/consan_sync_metadata.h"
#include "rocjitsu/code/patch/consan/consan_validation_inventory.h"
#include "rocjitsu/code/patch/instrumentation_builder.h"
#include "rocjitsu/code/patch/instrumentor.h"
#include "rocjitsu/code/patch/spill_manager.h"
#include "rocjitsu/code/patch/trampoline_builder.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/shared/gfx12_cache_flags.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "util/bit.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/AMDHSAKernelDescriptor.h"
RJ_DIAGNOSTIC_POP

#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <initializer_list>
#include <iterator>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace rocjitsu {
namespace {

namespace kd = rocr::llvm::amdhsa;

#include "rocjitsu/code/patch/consan/consan_placement.inc"

#include "rocjitsu/code/patch/consan/consan_sync_analysis.inc"

#include "rocjitsu/code/patch/consan/consan_fault_injection.inc"

#include "rocjitsu/code/patch/consan/consan_supercollider.inc"

#include "rocjitsu/code/patch/consan/consan_composition.inc"

ConSanTransformArtifacts run_consan_lowering_core(
    std::span<const uint8_t> code_object_bytes, const MoiOptions &options,
    ConSanPerturbationPlanningState *inspected_perturbation,
    const ConSanPreappliedMutationLayout &preapplied_mutation,
    std::span<const ConSanMoiTransientSgprAssignment> initial_owner_transient_sgprs,
    ConSanLoweringExecution *execution, ConSanLoweringExtent extent,
    const ConSanLoweringObservation *observation) {
  return try_patch_consan_impl(code_object_bytes, options, {}, std::nullopt, inspected_perturbation,
                               preapplied_mutation, initial_owner_transient_sgprs, false, execution,
                               extent, observation);
}

bool install_consan_lowering_observation(const ConSanOptions &options,
                                         ConSanTransformArtifacts &result,
                                         ConSanLoweringExecution *execution,
                                         const ConSanLoweringObservation *prepared_observation) {
  return initialize_consan_lowering_observation(options, result, execution, prepared_observation);
}

void apply_consan_fault_mutation_plans(std::span<const uint8_t> code_object_bytes,
                                       const ConSanOptions &context,
                                       std::span<const ConSanFaultMutationPlan> plans,
                                       ConSanTransformArtifacts &result) {
  AmdGpuCodeObject code_object(code_object_bytes.data(), code_object_bytes.size());
  const rj_code_arch_t arch = consan_arch_for_target(code_object.target_id());
  apply_fault_mutation_to_inventory(code_object, arch, context, plans, result);
}

} // namespace rocjitsu
