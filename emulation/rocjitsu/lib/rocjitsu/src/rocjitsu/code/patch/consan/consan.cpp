// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan.h"

#include "rocjitsu/analysis/def_use_chain.h"
#include "rocjitsu/analysis/kernel_scope.h"
#include "rocjitsu/analysis/liveness.h"
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/patch/consan/consan_moi.h"
#include "rocjitsu/code/patch/instruction_builder.h"
#include "rocjitsu/code/patch/instrumentation_builder.h"
#include "rocjitsu/code/patch/instrumentor.h"
#include "rocjitsu/code/patch/spill_manager.h"
#include "rocjitsu/code/patch/trampoline_builder.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/rdna4/machine_insts.h"
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
    const ConSanSyncEvent &cache, const ConSanSyncSequence &cache_sequence,
    bool require_same_block) {
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
         (*load.raw_scope == 2u || *load.raw_scope == 3u) &&
         cache.kind == ConSanSyncEventKind::Fence &&
         cache.operation == ConSanSyncOperation::Fence && cache.mnemonic == "global_inv" &&
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
         cache.operation == ConSanSyncOperation::Fence && cache.mnemonic == "global_wb" &&
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

namespace {

[[nodiscard]] bool range_contains(uint64_t outer_offset, uint64_t outer_size, uint64_t inner_offset,
                                  uint64_t inner_size) {
  return inner_offset >= outer_offset && inner_offset - outer_offset <= outer_size &&
         inner_size <= outer_size - (inner_offset - outer_offset);
}

[[nodiscard]] bool section_contains(const std::vector<const Section *> &sections, uint64_t offset,
                                    uint64_t size) {
  return std::ranges::any_of(sections, [&](const Section *section) {
    return range_contains(section->sectionOffset(), section->size(), offset, size);
  });
}

[[nodiscard]] std::optional<uint64_t> add_signed_offset(uint64_t base, int64_t offset) {
  if (offset >= 0) {
    const uint64_t positive = static_cast<uint64_t>(offset);
    if (positive > UINT64_MAX - base)
      return std::nullopt;
    return base + positive;
  }
  const uint64_t magnitude = uint64_t{0} - static_cast<uint64_t>(offset);
  if (magnitude > base)
    return std::nullopt;
  return base - magnitude;
}

[[nodiscard]] std::vector<std::string>
validate_consan_input_layout(const AmdGpuCodeObject &code_object,
                             bool allow_descriptor_entry_redirect = false) {
  std::vector<std::string> errors;
  for (const auto &section : code_object.all_sections()) {
    if (!range_contains(0, code_object.image_size(), section->sectionOffset(), section->size()))
      errors.emplace_back("ConSan input section '" + section->name() + "' exceeds ELF bytes");
  }
  for (const AmdGpuKernelInfo &kernel : code_object.kernels()) {
    if (!section_contains(code_object.rodata_sections(), kernel.descriptor_file_offset,
                          sizeof(rocr::llvm::amdhsa::kernel_descriptor_t))) {
      errors.emplace_back("ConSan kernel '" + kernel.name +
                          "' descriptor is not contained in a read-only data section");
    } else if (kernel.has_text_range && !allow_descriptor_entry_redirect) {
      rocr::llvm::amdhsa::kernel_descriptor_t descriptor{};
      std::memcpy(&descriptor, code_object.image_data() + kernel.descriptor_file_offset,
                  sizeof(descriptor));
      const uint64_t descriptor_address = code_object.kernel_descriptor_offset(kernel.name);
      const std::optional<uint64_t> descriptor_entry =
          add_signed_offset(descriptor_address, descriptor.kernel_code_entry_byte_offset);
      const Section *text_section = nullptr;
      for (const Section *section : code_object.text_sections()) {
        if (section->sectionOffset() == kernel.text_file_offset) {
          text_section = section;
          break;
        }
      }
      if (!text_section || kernel.entry_text_offset > UINT64_MAX - text_section->vaddr() ||
          !descriptor_entry ||
          *descriptor_entry != text_section->vaddr() + kernel.entry_text_offset) {
        errors.emplace_back("ConSan kernel '" + kernel.name +
                            "' descriptor entry does not match its function symbol");
      }
    }
    if (kernel.has_text_range &&
        !section_contains(code_object.text_sections(), kernel.text_file_offset, kernel.text_size)) {
      errors.emplace_back("ConSan kernel '" + kernel.name +
                          "' text range is not contained in a text section");
    }
    if (kernel.code_size > kernel.text_size) {
      errors.emplace_back("ConSan kernel '" + kernel.name + "' code size exceeds its text range");
    }
    if (kernel.has_text_range && (kernel.entry_text_offset > kernel.text_size ||
                                  kernel.code_size > kernel.text_size - kernel.entry_text_offset)) {
      errors.emplace_back("ConSan kernel '" + kernel.name +
                          "' function symbol exceeds its text section");
    }
  }
  for (const AmdGpuFunctionInfo &function : code_object.functions()) {
    if (!section_contains(code_object.text_sections(), function.text_file_offset,
                          function.text_size)) {
      errors.emplace_back("ConSan function '" + function.name +
                          "' text range is not contained in a text section");
    }
    if (function.code_size > function.text_size) {
      errors.emplace_back("ConSan function '" + function.name +
                          "' code size exceeds its text range");
    }
    if (function.entry_text_offset > function.text_size ||
        function.code_size > function.text_size - function.entry_text_offset) {
      errors.emplace_back("ConSan function '" + function.name +
                          "' symbol exceeds its text section");
    }
  }
  return errors;
}

} // namespace

#include "rocjitsu/code/patch/consan/consan_analysis.inc"

#include "rocjitsu/code/patch/consan/consan_placement.inc"

#include "rocjitsu/code/patch/consan/consan_sync_analysis.inc"

#include "rocjitsu/code/patch/consan/consan_fault_injection.inc"

#include "rocjitsu/code/patch/consan/consan_supercollider.inc"

#include "rocjitsu/code/patch/consan/consan_composition.inc"

#include "rocjitsu/code/patch/consan/consan_validation.inc"

bool consan_supercollider_supports_flat_site(const ConSanFlatSite &site,
                                             ConSanFlatProvenanceMode mode) {
  return is_supported_flat_check_trap_site(site, mode);
}

namespace {

[[nodiscard]] std::optional<std::string> access_patch_kind(ConSanPatchKind kind) {
  switch (kind) {
  case ConSanPatchKind::InlineLdsLoadCheckTrap:
  case ConSanPatchKind::InlineLdsStoreCheckTrap:
  case ConSanPatchKind::LocalCaveLdsLoadCheckTrap:
  case ConSanPatchKind::LocalCaveLdsStoreCheckTrap:
    return "lds-check-trap";
  case ConSanPatchKind::InlineFlatLoadCheckTrap:
  case ConSanPatchKind::InlineFlatStoreCheckTrap:
  case ConSanPatchKind::LocalCaveFlatLoadCheckTrap:
  case ConSanPatchKind::LocalCaveFlatStoreCheckTrap:
    return "flat-check-trap";
  default:
    return std::nullopt;
  }
}

[[nodiscard]] std::optional<ConSanAccessPlan> validated_access_plan(const ConSanResult &validated) {
  const ConSanPatchInfo *selected = nullptr;
  std::string kind;
  for (const ConSanPatchInfo &patch : validated.patches) {
    const auto candidate_kind = access_patch_kind(patch.kind);
    if (!candidate_kind)
      continue;
    if (selected != nullptr)
      return std::nullopt;
    selected = &patch;
    kind = *candidate_kind;
  }
  if (selected == nullptr)
    return std::nullopt;

  ConSanAccessPlan plan;
  plan.kind = std::move(kind);
  plan.text_offset = selected->anchor_offset;
  for (const ConSanKernelInfo &kernel : validated.kernels) {
    const bool owns = std::ranges::any_of(kernel.lds_sites,
                                          [&](const ConSanLdsSite &site) {
                                            return site.text_offset == selected->anchor_offset;
                                          }) ||
                      std::ranges::any_of(kernel.flat_sites, [&](const ConSanFlatSite &site) {
                        return site.text_offset == selected->anchor_offset;
                      });
    if (!owns)
      continue;
    if (!plan.container_name.empty())
      return std::nullopt;
    plan.container_name = kernel.name;
    plan.in_kernel = true;
  }
  for (const ConSanFunctionInfo &function : validated.functions) {
    const bool owns = std::ranges::any_of(function.lds_sites,
                                          [&](const ConSanLdsSite &site) {
                                            return site.text_offset == selected->anchor_offset;
                                          }) ||
                      std::ranges::any_of(function.flat_sites, [&](const ConSanFlatSite &site) {
                        return site.text_offset == selected->anchor_offset;
                      });
    if (!owns)
      continue;
    if (!plan.container_name.empty())
      return std::nullopt;
    plan.container_name = function.name;
    plan.in_kernel = false;
  }
  if (plan.container_name.empty())
    return std::nullopt;
  plan.identity = "access:" + plan.container_name + ":" + std::to_string(plan.text_offset);
  return plan;
}

[[nodiscard]] std::optional<ConSanCompositeProof>
validated_composite_proof(const ConSanResult &dry_run, const ConSanResult &validated) {
  if (!validated.staged_composition_validated || !validated.final_validation_passed ||
      validated.applied_fault_mutations != 1u || validated.applied_perturbations != 1u)
    return std::nullopt;
  const ConSanPatchInfo *perturbation = nullptr;
  for (const ConSanPatchInfo &patch : validated.patches) {
    if (patch.kind != ConSanPatchKind::TrampolineScPerturbation)
      continue;
    if (perturbation != nullptr)
      return std::nullopt;
    perturbation = &patch;
  }
  if (perturbation == nullptr || !perturbation->perturbation_edge ||
      perturbation->perturbation_source_candidate_identity.empty() ||
      perturbation->perturbation_source_sequence_identity.empty() ||
      perturbation->perturbation_source_container_name.empty() ||
      perturbation->perturbation_source_anchor_identity.empty() ||
      !perturbation->perturbation_source_owner_descriptor_file_offset)
    return std::nullopt;

  const auto translated =
      std::ranges::find_if(validated.perturbation_plans, [&](const ConSanPerturbationPlan &plan) {
        return plan.anchor_text_offset == perturbation->anchor_offset &&
               plan.anchor_size == perturbation->original_size &&
               plan.edge == *perturbation->perturbation_edge;
      });
  if (translated == validated.perturbation_plans.end())
    return std::nullopt;

  ConSanCompositeProof proof;
  proof.pristine_identity = perturbation->perturbation_source_candidate_identity;
  proof.pristine_sequence = perturbation->perturbation_source_sequence_identity;
  proof.pristine_container = perturbation->perturbation_source_container_name;
  proof.pristine_in_kernel = perturbation->perturbation_source_in_kernel;
  proof.pristine_owner_descriptor = *perturbation->perturbation_source_owner_descriptor_file_offset;
  proof.pristine_edge = *perturbation->perturbation_edge;
  proof.pristine_anchor = perturbation->perturbation_source_anchor_identity;
  proof.pristine_anchor_text_offset = perturbation->perturbation_source_anchor_offset;
  proof.pristine_anchor_size = perturbation->perturbation_source_anchor_size;
  proof.translated_identity = translated->candidate_identity;
  proof.translated_edge = translated->edge;
  proof.translated_anchor_text_offset = perturbation->anchor_offset;
  proof.translated_anchor_size = perturbation->original_size;
  proof.anchor_relation = "unchanged";
  proof.atomic_overlap = perturbation->perturbation_composite_atomic_overlap;
  proof.removed_cache_boundary = perturbation->perturbation_composite_removed_boundary;
  proof.removed_cache_non_resurrection_applicable = proof.removed_cache_boundary;
  proof.removed_cache_non_resurrection_validated = proof.removed_cache_boundary;
  if (!dry_run.fault_plans.empty() && dry_run.fault_plans.front().companion_identity)
    proof.cache_companion_identity = *dry_run.fault_plans.front().companion_identity;

  std::vector<uint64_t> move_sources;
  std::vector<uint64_t> move_targets;
  std::string atomic_mutation_group;
  for (const ConSanPatchInfo &patch : validated.patches) {
    if (patch.kind == ConSanPatchKind::InlineBarrierMoveSourceRewrite)
      move_sources.push_back(patch.anchor_offset);
    if (patch.kind == ConSanPatchKind::InlineBarrierMoveTargetRewrite)
      move_targets.push_back(patch.anchor_offset);
    const bool atomic = patch.kind == ConSanPatchKind::InlineAtomicAddressRewrite ||
                        patch.kind == ConSanPatchKind::InlineAtomicOrderRewrite ||
                        patch.kind == ConSanPatchKind::InlineAtomicScopeRewrite;
    if (atomic) {
      if (!proof.atomic_mutation_anchor_text_offset) {
        proof.atomic_mutation_anchor_text_offset = patch.anchor_offset;
        atomic_mutation_group = patch.fault_sequence_identity;
      } else if (patch.kind != ConSanPatchKind::InlineAtomicOrderRewrite ||
                 atomic_mutation_group.empty() ||
                 patch.fault_sequence_identity != atomic_mutation_group) {
        return std::nullopt;
      }
    }
  }
  if (proof.removed_cache_boundary)
    proof.anchor_relation = "removed-cache-boundary";
  else if (!move_sources.empty()) {
    if (move_targets.size() != 1u)
      return std::nullopt;
    const uint64_t first = *std::ranges::min_element(move_sources);
    const uint64_t target = *std::ranges::min_element(move_targets);
    if (proof.pristine_anchor_text_offset < first ||
        proof.pristine_anchor_text_offset - first > sizeof(uint32_t))
      return std::nullopt;
    const uint64_t delta = proof.pristine_anchor_text_offset - first;
    proof.anchor_relation = delta == 0u ? "move-target+0" : "move-target+4";
    // Expose the logical translated boundary relative to the stable move
    // destination identity. The validated code may execute it in a trampoline.
    proof.translated_anchor_text_offset = target + delta;
  }
  if (proof.atomic_overlap && !proof.atomic_mutation_anchor_text_offset)
    return std::nullopt;
  return proof;
}

} // namespace

ConSanResult retry_patch_consan_moi_from_inventory(ConSanResult inventory,
                                                   const ConSanMoiReportRetryConfig &report,
                                                   std::span<const uint8_t> code_object_bytes) {
  const ConSanMoiEngine inventory_engine = inventory.moi_engine;
  try {
    if (!inventory.moi_inventory_options)
      inventory.errors.emplace_back("ConSan MOI inventory retry has no immutable option snapshot");
    ConSanOptions options;
    options.flavor = ConSanFlavor::Moi;
    if (inventory.moi_inventory_options)
      options = *inventory.moi_inventory_options;
    if (options.flavor != ConSanFlavor::Moi)
      inventory.errors.emplace_back("ConSan MOI inventory retry requires the MOI flavor");
    if (!inventory.visited_code_object || inventory.input_size != code_object_bytes.size())
      inventory.errors.emplace_back(
          "ConSan MOI inventory retry does not match the original code-object size");
    if (inventory.input_fingerprint.empty() ||
        inventory.input_fingerprint != code_object_fingerprint(code_object_bytes)) {
      inventory.errors.emplace_back(
          "ConSan MOI inventory retry does not match the original code-object bytes");
    }
    if (inventory.flavor != ConSanFlavor::Moi || inventory.moi_engine != options.moi_engine)
      inventory.errors.emplace_back(
          "ConSan MOI inventory retry does not match the requested engine");
    if (inventory.modified || !inventory.elf_bytes.empty() || !inventory.patches.empty() ||
        inventory.applied_fault_mutations != 0u || inventory.applied_perturbations != 0u ||
        !inventory.fault_plans.empty() || !inventory.perturbation_plans.empty()) {
      inventory.errors.emplace_back(
          "ConSan MOI inventory retry requires an unmodified semantic inventory");
    }
    if (inventory.moi_stage_warning_begin > inventory.warnings.size())
      inventory.errors.emplace_back("ConSan MOI inventory retry has an invalid warning boundary");
    if (!inventory.errors.empty()) {
      inventory.outcome = ConSanTransformOutcome::Invalid;
      return finalize_consan_result(std::move(inventory), code_object_bytes);
    }

    AmdGpuCodeObject code_object(code_object_bytes.data(), code_object_bytes.size());
    const rj_code_arch_t arch = arch_for_target(code_object.target_id());
    if (arch == ROCJITSU_CODE_ARCH_INVALID ||
        inventory.target_name != target_name(code_object.target_id())) {
      inventory.errors.emplace_back(
          "ConSan MOI inventory retry does not match the original code-object target");
    }
    if (!inventory.errors.empty()) {
      inventory.outcome = ConSanTransformOutcome::Invalid;
      return finalize_consan_result(std::move(inventory), code_object_bytes);
    }

    options.moi_report_buffer_address = report.buffer_address;
    options.moi_report_buffer_size = report.buffer_size;
    options.moi_report_layout = report.layout;
    options.moi_report_generation = report.generation;
    options.moi_report_dispatch_id = report.dispatch_id;
    ConSanResult result =
        try_patch_consan_moi(std::move(inventory), options, code_object_bytes, arch);
    try_apply_unmatched_barrier_wait_abort(code_object_bytes, options, result);
    return finalize_consan_result(std::move(result), code_object_bytes);
  } catch (const std::exception &error) {
    ConSanResult result;
    result.visited_code_object = true;
    result.flavor = ConSanFlavor::Moi;
    result.moi_engine = inventory_engine;
    result.input_size = code_object_bytes.size();
    result.errors.emplace_back(std::string("ConSan MOI inventory retry threw an exception: ") +
                               error.what());
    return finalize_consan_result(std::move(result), code_object_bytes);
  } catch (...) {
    ConSanResult result;
    result.visited_code_object = true;
    result.flavor = ConSanFlavor::Moi;
    result.moi_engine = inventory_engine;
    result.input_size = code_object_bytes.size();
    result.errors.emplace_back("ConSan MOI inventory retry threw a non-standard exception");
    return finalize_consan_result(std::move(result), code_object_bytes);
  }
}

ConSanResult try_patch_consan(std::span<const uint8_t> code_object_bytes,
                              const ConSanOptions &options) {
  try {
    ConSanResult result = try_patch_consan_impl(code_object_bytes, options);
    try_apply_unmatched_barrier_wait_abort(code_object_bytes, options, result);
    result = finalize_consan_result(std::move(result), code_object_bytes);
    const bool composite_discovery =
        options.fault_dry_run && options.flavor == ConSanFlavor::SuperCollider &&
        options.sc_perturb_kind != ConSanPerturbationKind::None &&
        (options.probe_lds_check_trap || options.probe_flat_check_trap) && result.errors.empty() &&
        result.fault_plans.size() == 1u && result.perturbation_plans.size() == 1u;
    if (composite_discovery) {
      ConSanOptions live_options = options;
      live_options.fault_dry_run = false;
      ConSanResult validated = finalize_consan_result(
          try_patch_consan_impl(code_object_bytes, live_options), code_object_bytes);
      auto access = validated_access_plan(validated);
      auto proof = validated_composite_proof(result, validated);
      if (access && proof) {
        result.access_plans.push_back(std::move(*access));
        result.composite_proof = std::move(*proof);
      }
    }
    return result;
  } catch (const std::exception &error) {
    ConSanResult result;
    result.visited_code_object = true;
    result.flavor = options.flavor;
    result.moi_engine = options.moi_engine;
    result.input_size = code_object_bytes.size();
    result.errors.emplace_back(std::string("ConSan transform threw an exception: ") + error.what());
    return finalize_consan_result(std::move(result), code_object_bytes);
  } catch (...) {
    ConSanResult result;
    result.visited_code_object = true;
    result.flavor = options.flavor;
    result.moi_engine = options.moi_engine;
    result.input_size = code_object_bytes.size();
    result.errors.emplace_back("ConSan transform threw a non-standard exception");
    return finalize_consan_result(std::move(result), code_object_bytes);
  }
}

} // namespace rocjitsu
