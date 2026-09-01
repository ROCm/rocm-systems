// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// This translation unit owns only top-level ConSan composition. Analysis,
// placement, mutation, and engine mechanics enter through compiled contracts;
// their implementation details are not textually visible here.

#include "rocjitsu/code/patch/consan/consan_composition.h"

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
#include "rocjitsu/code/patch/consan/consan_fault_injection.h"
#include "rocjitsu/code/patch/consan/consan_fault_selection.h"
#include "rocjitsu/code/patch/consan/consan_fault_target_ops.h"
#include "rocjitsu/code/patch/consan/consan_final_validation.h"
#include "rocjitsu/code/patch/consan/consan_growth_policy.h"
#include "rocjitsu/code/patch/consan/consan_input_layout.h"
#include "rocjitsu/code/patch/consan/consan_instruction_semantics.h"
#include "rocjitsu/code/patch/consan/consan_inventory_diagnostics.h"
#include "rocjitsu/code/patch/consan/consan_moi.h"
#include "rocjitsu/code/patch/consan/consan_moi_internal.h"
#include "rocjitsu/code/patch/consan/consan_perturbation.h"
#include "rocjitsu/code/patch/consan/consan_perturbation_policy.h"
#include "rocjitsu/code/patch/consan/consan_physical_site_alias.h"
#include "rocjitsu/code/patch/consan/consan_placement.h"
#include "rocjitsu/code/patch/consan/consan_program_analysis.h"
#include "rocjitsu/code/patch/consan/consan_relay_target_ops.h"
#include "rocjitsu/code/patch/consan/consan_resource.h"
#include "rocjitsu/code/patch/consan/consan_runtime_kernel.h"
#include "rocjitsu/code/patch/consan/consan_semantic_classifiers.h"
#include "rocjitsu/code/patch/consan/consan_supercollider.h"
#include "rocjitsu/code/patch/consan/consan_supercollider_support.h"
#include "rocjitsu/code/patch/consan/consan_supercollider_target_ops.h"
#include "rocjitsu/code/patch/consan/consan_sync_analysis.h"
#include "rocjitsu/code/patch/consan/consan_sync_event_index.h"
#include "rocjitsu/code/patch/consan/consan_sync_metadata.h"
#include "rocjitsu/code/patch/instrumentation_builder.h"
#include "rocjitsu/code/patch/instrumentor.h"
#include "rocjitsu/code/patch/spill_manager.h"
#include "rocjitsu/code/patch/trampoline_builder.h"
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

namespace kd = rocr::llvm::amdhsa;

#include "rocjitsu/code/patch/consan/consan_composition.inc"

ConSanTransformArtifacts
compose_consan_lowering(std::span<const uint8_t> code_object_bytes, const ConSanOptions &options,
                        const ConSanMoiOperatingPoint &initial_operating_point,
                        ConSanPerturbationPlanningState *inspected_perturbation,
                        const ConSanPreappliedMutationLayout &preapplied_mutation,
                        ConSanLoweringExecution *execution, ConSanLoweringExtent extent,
                        const ConSanLoweringObservation *observation) {
  return try_patch_consan_impl(code_object_bytes, options, initial_operating_point, {},
                               std::nullopt, inspected_perturbation, preapplied_mutation, false,
                               execution, extent, observation);
}

bool compose_consan_observation(const ConSanOptions &options, ConSanTransformArtifacts &result,
                                ConSanLoweringExecution *execution,
                                const ConSanLoweringObservation *prepared_observation) {
  return initialize_consan_lowering_observation(options, result, execution, prepared_observation);
}

void compose_consan_fault_mutation(std::span<const uint8_t> code_object_bytes,
                                   const ConSanOptions &context,
                                   std::span<const ConSanFaultMutationPlan> plans,
                                   ConSanTransformArtifacts &result) {
  AmdGpuCodeObject code_object(code_object_bytes.data(), code_object_bytes.size());
  const rj_code_arch_t arch = consan_arch_for_target(code_object.target_id());
  apply_consan_fault_mutations(code_object, arch, context.patched_image_growth_limit,
                               context.fault_require_exactly_one, plans, result);
}

} // namespace rocjitsu
