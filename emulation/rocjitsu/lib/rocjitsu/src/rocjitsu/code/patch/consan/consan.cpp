// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/major_image_ownership.h"
#include "rocjitsu/code/patch/consan/consan_final_validation.h"
#include "rocjitsu/code/patch/consan/consan_lowerer.h"
#include "rocjitsu/code/patch/consan/consan_lowering.h"
#include "rocjitsu/code/patch/consan/consan_moi.h"

#include <exception>
#include <string>
#include <utility>

namespace rocjitsu {

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
      ConSanTransformArtifacts result =
          run_consan_lowering_core(code_object_bytes, options, nullptr, {}, {}, execution);
      try_apply_unmatched_barrier_wait_abort(code_object_bytes, options, result);
      return finalize_consan_result(std::move(result), code_object_bytes,
                                    options.moi_report_dispatch_id, false, nullptr, execution);
    }
    if (!install_consan_lowering_observation(options, inventory, execution, observation)) {
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
    ConSanTransformArtifacts result = run_consan_lowering_core(
        code_object_bytes, effective_options, inspected_perturbation, preapplied_mutation,
        initial_owner_transient_sgprs, execution, extent, observation);
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
