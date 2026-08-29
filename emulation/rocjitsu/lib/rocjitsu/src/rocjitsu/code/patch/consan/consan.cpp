// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

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
#include "rocjitsu/code/patch/consan/consan_fault_selection.h"
#include "rocjitsu/code/patch/consan/consan_growth_policy.h"
#include "rocjitsu/code/patch/consan/consan_input_layout.h"
#include "rocjitsu/code/patch/consan/consan_instruction_semantics.h"
#include "rocjitsu/code/patch/consan/consan_lowering.h"
#include "rocjitsu/code/patch/consan/consan_moi.h"
#include "rocjitsu/code/patch/consan/consan_moi_internal.h"
#include "rocjitsu/code/patch/consan/consan_physical_site_alias.h"
#include "rocjitsu/code/patch/consan/consan_relay_target_ops.h"
#include "rocjitsu/code/patch/consan/consan_resource.h"
#include "rocjitsu/code/patch/consan/consan_sync_event_index.h"
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

[[nodiscard]] bool ordinary_acquire_metadata_compatible_impl(
    const ConSanSyncEvent &load, const ConSanSyncSequence &load_sequence,
    const ConSanSyncEvent &cache, const ConSanSyncSequence &cache_sequence, bool require_same_block,
    bool allow_rdna3_cache_pair_member = false) {
  const auto same_owners = [](std::span<const ConSanExecutionOwner> lhs,
                              std::span<const ConSanExecutionOwner> rhs) {
    return !lhs.empty() && lhs.size() == rhs.size() &&
           std::ranges::equal(
               lhs, rhs, [](const ConSanExecutionOwner &left, const ConSanExecutionOwner &right) {
                 return left.descriptor_file_offset == right.descriptor_file_offset &&
                        left.proof == right.proof;
               });
  };
  return load.kind == ConSanSyncEventKind::OrdinaryMemory &&
         load.operation == ConSanSyncOperation::OrdinaryLoad && load.width_bits == 32u &&
         load.confidence == ConSanSemanticConfidence::Conservative && load.raw_scope &&
         (*load.raw_scope >= 1u && *load.raw_scope <= 3u) &&
         cache.kind == ConSanSyncEventKind::Fence &&
         cache.operation == ConSanSyncOperation::Fence &&
         (cache.mnemonic == "global_inv" || cache.mnemonic == "buffer_inv" ||
          (*load.raw_scope == 1u && cache.mnemonic == "buffer_gl0_inv") ||
          (allow_rdna3_cache_pair_member &&
           (cache.mnemonic == "buffer_gl1_inv" || cache.mnemonic == "buffer_gl0_inv"))) &&
         cache.confidence == ConSanSemanticConfidence::Conservative &&
         load.code_object_fingerprint == cache.code_object_fingerprint &&
         load.container_name == cache.container_name && load.in_kernel == cache.in_kernel &&
         load_sequence.kind == ConSanSyncSequenceKind::OrdinaryMemory &&
         load_sequence.operation == ConSanSyncOperation::OrdinaryLoad &&
         cache_sequence.kind == ConSanSyncSequenceKind::Fence &&
         cache_sequence.operation == ConSanSyncOperation::Fence &&
         load_sequence.basic_block_index && cache_sequence.basic_block_index &&
         (!require_same_block ||
          load_sequence.basic_block_index == cache_sequence.basic_block_index) &&
         same_owners(load.execution_owners, cache.execution_owners) &&
         same_owners(load_sequence.execution_owners, cache_sequence.execution_owners) &&
         same_owners(load.execution_owners, load_sequence.execution_owners);
}

} // namespace

bool consan_ordinary_acquire_metadata_compatible(const ConSanSyncEvent &load,
                                                 const ConSanSyncSequence &load_sequence,
                                                 const ConSanSyncEvent &cache,
                                                 const ConSanSyncSequence &cache_sequence) {
  return ordinary_acquire_metadata_compatible_impl(load, load_sequence, cache, cache_sequence,
                                                   /*require_same_block=*/true);
}

bool consan_ordinary_release_metadata_compatible(const ConSanSyncEvent &cache,
                                                 const ConSanSyncSequence &cache_sequence,
                                                 const ConSanSyncEvent &store,
                                                 const ConSanSyncSequence &store_sequence) {
  const auto same_owners = [](std::span<const ConSanExecutionOwner> lhs,
                              std::span<const ConSanExecutionOwner> rhs) {
    return !lhs.empty() && lhs.size() == rhs.size() &&
           std::ranges::equal(
               lhs, rhs, [](const ConSanExecutionOwner &left, const ConSanExecutionOwner &right) {
                 return left.descriptor_file_offset == right.descriptor_file_offset &&
                        left.proof == right.proof;
               });
  };
  return cache.kind == ConSanSyncEventKind::Fence &&
         cache.operation == ConSanSyncOperation::Fence &&
         (cache.mnemonic == "global_wb" || cache.mnemonic == "buffer_wb" ||
          cache.mnemonic == "buffer_wbl2") &&
         cache.confidence == ConSanSemanticConfidence::Conservative &&
         store.kind == ConSanSyncEventKind::OrdinaryMemory &&
         store.operation == ConSanSyncOperation::OrdinaryStore && store.width_bits != 0u &&
         store.width_bits <= 128u && store.confidence == ConSanSemanticConfidence::Conservative &&
         store.raw_scope && (*store.raw_scope == 2u || *store.raw_scope == 3u) &&
         cache.code_object_fingerprint == store.code_object_fingerprint &&
         cache.container_name == store.container_name && cache.in_kernel == store.in_kernel &&
         cache_sequence.kind == ConSanSyncSequenceKind::Fence &&
         cache_sequence.operation == ConSanSyncOperation::Fence &&
         store_sequence.kind == ConSanSyncSequenceKind::OrdinaryMemory &&
         store_sequence.operation == ConSanSyncOperation::OrdinaryStore &&
         cache_sequence.basic_block_index &&
         cache_sequence.basic_block_index == store_sequence.basic_block_index &&
         same_owners(cache.execution_owners, store.execution_owners) &&
         same_owners(cache_sequence.execution_owners, store_sequence.execution_owners) &&
         same_owners(store.execution_owners, store_sequence.execution_owners);
}

#include "rocjitsu/code/patch/consan/consan_analysis.inc"

#include "rocjitsu/code/patch/consan/consan_placement.inc"

#include "rocjitsu/code/patch/consan/consan_sync_analysis.inc"

#include "rocjitsu/code/patch/consan/consan_fault_injection.inc"

#include "rocjitsu/code/patch/consan/consan_supercollider.inc"

#include "rocjitsu/code/patch/consan/consan_composition.inc"

#include "rocjitsu/code/patch/consan/consan_validation.inc"

ConSanTransformArtifacts retry_patch_consan_moi_from_inventory(
    ConSanTransformArtifacts inventory_artifacts, ConSanOptions options,
    std::span<const uint8_t> code_object_bytes, ConSanLoweringExecution *execution,
    const ConSanLoweringObservation *observation) {
  const major_image_ownership::ScopedOwner input_owner(major_image_ownership::OwnerKind::InputImage,
                                                       code_object_bytes.data(),
                                                       code_object_bytes.size());
  try {
    ConSanTransformArtifacts inventory = std::move(inventory_artifacts);
    if (options.flavor != ConSanFlavor::Moi)
      inventory.errors.emplace_back("ConSan MOI inventory retry requires the MOI flavor");
    if (inventory.observation_plan.engine !=
        consan_capability_engine(ConSanFlavor::Moi, options.moi_engine)) {
      inventory.errors.emplace_back(
          "ConSan MOI inventory retry does not match the requested engine");
    }
    const ConSanCodeObjectId &inventory_id = inventory.program_inventory.code_object_id();
    if (inventory_id.byte_size != code_object_bytes.size())
      inventory.errors.emplace_back(
          "ConSan MOI inventory retry does not match the original code-object size");
    if (!inventory_id.valid() || inventory_id != make_consan_code_object_id(code_object_bytes)) {
      inventory.errors.emplace_back(
          "ConSan MOI inventory retry does not match the original code-object bytes");
    }
    if (inventory.modified() || !inventory.replacement.empty() ||
        inventory.mutation.fault.applied != 0u || inventory.mutation.perturbation.applied != 0u ||
        !inventory.fault_plans.empty()) {
      inventory.errors.emplace_back(
          "ConSan MOI inventory retry requires an unmodified semantic inventory");
    }
    if (!inventory.errors.empty()) {
      inventory.outcome = ConSanTransformOutcome::Invalid;
      return finalize_consan_result(std::move(inventory), code_object_bytes, 0, false, nullptr,
                                    execution);
    }

    AmdGpuCodeObject code_object(code_object_bytes.data(), code_object_bytes.size());
    const rj_code_arch_t arch = consan_arch_for_target(code_object.target_id());
    if (arch == ROCJITSU_CODE_ARCH_INVALID ||
        inventory.program_inventory.target() != code_object.target_id()) {
      inventory.errors.emplace_back(
          "ConSan MOI inventory retry does not match the original code-object target");
    }
    if (inventory.program_inventory.arch() != arch) {
      inventory.errors.emplace_back(
          "ConSan MOI inventory retry does not match the original code-object architecture");
    }
    if (!inventory.errors.empty()) {
      inventory.outcome = ConSanTransformOutcome::Invalid;
      return finalize_consan_result(std::move(inventory), code_object_bytes, 0, false, nullptr,
                                    execution);
    }

    // The retry replaces diagnostics from the unbound sizing attempt. The
    // immutable inventory, policy, and coverage records carry its semantic
    // output without requiring warning provenance in private working state.
    inventory.warnings.clear();
    const bool has_late_fault = options.has_fault_mutation();
    if (has_late_fault && options.fault_dry_run) {
      inventory.errors.emplace_back(
          "ConSan MOI inventory retry accepts only a live late-bound fault selection");
      inventory.outcome = ConSanTransformOutcome::Invalid;
      return finalize_consan_result(std::move(inventory), code_object_bytes, 0, false, nullptr,
                                    execution);
    }
    if (has_late_fault) {
      // A pristine report-sizing inventory deliberately excludes the live
      // mutation. Rebuild for the rare late-fault path instead of coupling
      // the retained inventory to every mutation-specific analysis choice.
      ConSanTransformArtifacts result = try_patch_consan_impl(
          code_object_bytes, options, {}, std::nullopt, nullptr, {}, {}, false, execution);
      try_apply_unmatched_barrier_wait_abort(code_object_bytes, options, result);
      return finalize_consan_result(std::move(result), code_object_bytes,
                                    options.moi_report_dispatch_id, false, nullptr, execution);
    }
    if (!initialize_consan_lowering_observation(options, inventory, execution, observation)) {
      inventory.outcome = ConSanTransformOutcome::Invalid;
      return finalize_consan_result(std::move(inventory), code_object_bytes,
                                    options.moi_report_dispatch_id, false, nullptr, execution);
    }
    ConSanTransformArtifacts result =
        try_patch_consan_moi(std::move(inventory), options, code_object_bytes, arch, execution);
    try_apply_unmatched_barrier_wait_abort(code_object_bytes, options, result);
    return finalize_consan_result(std::move(result), code_object_bytes,
                                  options.moi_report_dispatch_id, false, nullptr, execution);
  } catch (const std::exception &error) {
    ConSanTransformArtifacts result;
    result.errors.emplace_back(std::string("ConSan MOI inventory retry threw an exception: ") +
                               error.what());
    return finalize_consan_result(std::move(result), code_object_bytes, 0, false, nullptr,
                                  execution);
  } catch (...) {
    ConSanTransformArtifacts result;
    result.errors.emplace_back("ConSan MOI inventory retry threw a non-standard exception");
    return finalize_consan_result(std::move(result), code_object_bytes, 0, false, nullptr,
                                  execution);
  }
}

[[nodiscard]] ConSanTransformArtifacts complete_consan_lowering_observed(
    std::span<const uint8_t> code_object_bytes, const MoiOptions &options,
    ConSanPerturbationPlanningState *inspected_perturbation,
    const ConSanPreappliedMutationLayout &preapplied_mutation,
    std::span<const ConSanMoiTransientSgprAssignment> initial_owner_transient_sgprs,
    ConSanLoweringExecution *execution, ConSanLoweringExtent extent,
    const ConSanLoweringObservation *observation) {
  const major_image_ownership::ScopedOwner input_owner(major_image_ownership::OwnerKind::InputImage,
                                                       code_object_bytes.data(),
                                                       code_object_bytes.size());
  try {
    MoiOptions effective_options = options;
    ConSanTransformArtifacts result = try_patch_consan_impl(
        code_object_bytes, effective_options, {}, std::nullopt, inspected_perturbation,
        preapplied_mutation, initial_owner_transient_sgprs, false, execution, extent, observation);
    const bool stopped_after_inventory =
        extent == ConSanLoweringExtent::ThroughProgramInventory && execution != nullptr &&
        execution->program_inventory_passes != 0u && execution->observation_plan_passes == 0u;
    const bool stopped_after_observation = extent == ConSanLoweringExtent::ThroughObservationPlan &&
                                           execution != nullptr &&
                                           execution->observation_plan_passes != 0u;
    if ((stopped_after_inventory || stopped_after_observation) && execution != nullptr &&
        execution->resource_solving_and_lowering_passes == 0u) {
      if (!result.errors.empty())
        result.outcome = ConSanTransformOutcome::Invalid;
      return result;
    }
    try_apply_unmatched_barrier_wait_abort(code_object_bytes, effective_options, result);
    result =
        finalize_consan_result(std::move(result), code_object_bytes, options.moi_report_dispatch_id,
                               false, inspected_perturbation, execution);
    return result;
  } catch (const std::exception &error) {
    ConSanTransformArtifacts result;
    result.errors.emplace_back(std::string("ConSan transform threw an exception: ") + error.what());
    return finalize_consan_result(std::move(result), code_object_bytes, 0, false, nullptr,
                                  execution);
  } catch (...) {
    ConSanTransformArtifacts result;
    result.errors.emplace_back("ConSan transform threw a non-standard exception");
    return finalize_consan_result(std::move(result), code_object_bytes, 0, false, nullptr,
                                  execution);
  }
}

ConSanTransformArtifacts complete_consan_lowering(
    std::span<const uint8_t> code_object_bytes, const MoiOptions &options,
    ConSanPerturbationPlanningState *inspected_perturbation = nullptr,
    const ConSanPreappliedMutationLayout &preapplied_mutation = {},
    std::span<const ConSanMoiTransientSgprAssignment> initial_owner_transient_sgprs = {}) {
  return complete_consan_lowering_observed(code_object_bytes, options, inspected_perturbation,
                                           preapplied_mutation, initial_owner_transient_sgprs,
                                           nullptr, ConSanLoweringExtent::Complete, nullptr);
}

ConSanTransformArtifacts test_apply_consan_fault_plans(std::span<const uint8_t> code_object_bytes,
                                                       const ConSanOptions &application_context,
                                                       ConSanTransformArtifacts planned_artifacts) {
  const major_image_ownership::ScopedOwner input_owner(major_image_ownership::OwnerKind::InputImage,
                                                       code_object_bytes.data(),
                                                       code_object_bytes.size());
  AmdGpuCodeObject code_object(code_object_bytes.data(), code_object_bytes.size());
  const rj_code_arch_t arch = consan_arch_for_target(code_object.target_id());
  apply_fault_mutation_to_inventory(code_object, arch, application_context,
                                    planned_artifacts.fault_plans, planned_artifacts);
  return finalize_consan_result(std::move(planned_artifacts), code_object_bytes,
                                application_context.moi_report_dispatch_id);
}

ConSanTransformArtifacts lower_consan(std::span<const uint8_t> code_object_bytes,
                                      const ConSanOptions &options,
                                      ConSanLoweringExecution *execution,
                                      ConSanLoweringExtent extent,
                                      const ConSanLoweringObservation *observation) {
  if (execution != nullptr)
    *execution = {};
  return complete_consan_lowering_observed(code_object_bytes, options, nullptr, {}, {}, execution,
                                           extent, observation);
}

} // namespace rocjitsu
