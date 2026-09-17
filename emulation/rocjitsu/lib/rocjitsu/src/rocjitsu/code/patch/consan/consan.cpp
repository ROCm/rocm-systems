// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/major_image_ownership.h"
#include "rocjitsu/code/patch/consan/consan_composition.h"
#include "rocjitsu/code/patch/consan/consan_final_validation.h"
#include "rocjitsu/code/patch/consan/consan_instrumentation.h"
#include "rocjitsu/code/patch/consan/consan_lowering.h"
#include "rocjitsu/code/patch/consan/consan_unmatched_barrier_abort.h"

#include <exception>
#include <string>
#include <utility>

namespace rocjitsu::consan {

[[nodiscard]] static TransformArtifacts
finalize_exception(std::span<const uint8_t> code_object_bytes, std::string error) {
  TransformArtifacts result;
  result.errors.push_back(std::move(error));
  return finalize_result(std::move(result), code_object_bytes);
}

TransformArtifacts retry_patch_from_inventory(RetryInventory retry_inventory, Options options,
                                              std::span<const uint8_t> code_object_bytes,
                                              const LoweringObservation *observation) {
  const major_image_ownership::ScopedOwner input_owner(major_image_ownership::OwnerKind::InputImage,
                                                       code_object_bytes.data(),
                                                       code_object_bytes.size());
  try {
    const OperatingPoint initial_operating_point =
        consan::initial_operating_point(options, options);
    TransformArtifacts inventory;
    inventory.program_inventory = std::move(retry_inventory.program_inventory);
    inventory.coverage_ledger = std::move(retry_inventory.coverage_ledger);
    inventory.fault_sites = std::move(retry_inventory.fault_sites);
    inventory.barrier_move_destinations = std::move(retry_inventory.barrier_move_destinations);
    if (options.mode != Mode::Default)
      inventory.errors.emplace_back("ConSan inventory retry requires the ConSan mode");
    if (inventory.observation_plan().mode != enabled_mode(Mode::Default)) {
      inventory.errors.emplace_back("ConSan inventory retry does not match the requested mode");
    }
    const CodeObjectId &inventory_id = inventory.program_inventory.code_object_id();
    if (inventory_id.byte_size != code_object_bytes.size())
      inventory.errors.emplace_back(
          "ConSan inventory retry does not match the original code-object size");
    if (!inventory_id.valid() || inventory_id != make_code_object_id(code_object_bytes)) {
      inventory.errors.emplace_back(
          "ConSan inventory retry does not match the original code-object bytes");
    }
    if (!inventory.errors.empty()) {
      inventory.outcome = TransformOutcome::Invalid;
      return finalize_result(std::move(inventory), code_object_bytes);
    }

    AmdGpuCodeObject code_object(code_object_bytes.data(), code_object_bytes.size());
    const rj_code_arch_t arch = arch_for_target(code_object.target_id());
    if (arch == ROCJITSU_CODE_ARCH_INVALID ||
        inventory.program_inventory.target() != code_object.target_id()) {
      inventory.errors.emplace_back(
          "ConSan inventory retry does not match the original code-object target");
    }
    if (inventory.program_inventory.arch() != arch) {
      inventory.errors.emplace_back(
          "ConSan inventory retry does not match the original code-object architecture");
    }
    if (!inventory.errors.empty()) {
      inventory.outcome = TransformOutcome::Invalid;
      return finalize_result(std::move(inventory), code_object_bytes);
    }

    const bool has_late_fault = options.has_fault_mutation();
    if (has_late_fault && options.fault_dry_run) {
      inventory.errors.emplace_back(
          "ConSan inventory retry accepts only a live late-bound fault selection");
      inventory.outcome = TransformOutcome::Invalid;
      return finalize_result(std::move(inventory), code_object_bytes);
    }
    if (has_late_fault) {
      // A pristine report-sizing inventory deliberately excludes the live
      // mutation. Rebuild for the rare late-fault path instead of coupling
      // the retained inventory to every mutation-specific analysis choice.
      TransformArtifacts result =
          compose_lowering(code_object_bytes, options, initial_operating_point, nullptr, {},
                           LoweringExtent::Complete, nullptr);
      try_apply_unmatched_barrier_wait_abort(code_object_bytes, options, options, options, options,
                                             result);
      return finalize_result(std::move(result), code_object_bytes);
    }
    if (!compose_observation(options, inventory, observation)) {
      inventory.outcome = TransformOutcome::Invalid;
      return finalize_result(std::move(inventory), code_object_bytes);
    }
    TransformArtifacts result =
        try_patch(std::move(inventory), options, initial_operating_point, code_object_bytes, arch);
    try_apply_unmatched_barrier_wait_abort(code_object_bytes, options, options, options, options,
                                           result);
    return finalize_result(std::move(result), code_object_bytes);
  } catch (const std::exception &error) {
    return finalize_exception(code_object_bytes,
                              std::string("ConSan inventory retry threw an exception: ") +
                                  error.what());
  } catch (...) {
    return finalize_exception(code_object_bytes,
                              "ConSan inventory retry threw a non-standard exception");
  }
}

[[nodiscard]] TransformArtifacts complete_lowering_with_operating_point(
    std::span<const uint8_t> code_object_bytes, const Options &options,
    const OperatingPoint &initial_operating_point,
    SuperColliderPerturbationPlanningState *inspected_supercollider_perturbation,
    const PreappliedMutationLayout &preapplied_mutation, LoweringExtent extent,
    const LoweringObservation *observation) {
  const major_image_ownership::ScopedOwner input_owner(major_image_ownership::OwnerKind::InputImage,
                                                       code_object_bytes.data(),
                                                       code_object_bytes.size());
  try {
    TransformArtifacts result = compose_lowering(
        code_object_bytes, options, initial_operating_point, inspected_supercollider_perturbation,
        preapplied_mutation, extent, observation);
    if (extent != LoweringExtent::Complete) {
      if (!result.errors.empty())
        result.outcome = TransformOutcome::Invalid;
      return result;
    }
    try_apply_unmatched_barrier_wait_abort(code_object_bytes, options, options, options, options,
                                           result);
    return finalize_result(std::move(result), code_object_bytes);
  } catch (const std::exception &error) {
    return finalize_exception(code_object_bytes,
                              std::string("ConSan transform threw an exception: ") + error.what());
  } catch (...) {
    return finalize_exception(code_object_bytes, "ConSan transform threw a non-standard exception");
  }
}

TransformArtifacts lower(std::span<const uint8_t> code_object_bytes, const Options &options,
                         LoweringExtent extent, const LoweringObservation *observation) {
  return complete_lowering_with_operating_point(code_object_bytes, options,
                                                initial_operating_point(options, options), nullptr,
                                                {}, extent, observation);
}

} // namespace rocjitsu::consan
