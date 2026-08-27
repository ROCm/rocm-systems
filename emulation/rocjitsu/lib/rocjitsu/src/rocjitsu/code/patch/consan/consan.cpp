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
#include "rocjitsu/code/patch/consan/consan_branch_only_relay_router.h"
#include "rocjitsu/code/patch/consan/consan_growth_policy.h"
#include "rocjitsu/code/patch/consan/consan_lowering.h"
#include "rocjitsu/code/patch/consan/consan_moi.h"
#include "rocjitsu/code/patch/consan/consan_moi_internal.h"
#include "rocjitsu/code/patch/consan/consan_physical_site_alias.h"
#include "rocjitsu/code/patch/consan/consan_resource.h"
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
  if (!code_object.kernel_metadata_is_trustworthy()) {
    const size_t malformed_notes = code_object.malformed_kernel_metadata_note_count();
    if (malformed_notes == 0) {
      errors.emplace_back(
          "ConSan cannot safely transform a code object with incomplete AMDGPU kernel metadata");
    } else {
      errors.emplace_back(
          "ConSan cannot safely transform a code object with " + std::to_string(malformed_notes) +
          " malformed AMDGPU kernel metadata note" + (malformed_notes == 1u ? "" : "s"));
    }
  }
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

bool consan_supercollider_supports_access(const ConSanAccessInventorySite &access,
                                          ConSanFlatProvenanceMode mode, rj_code_arch_t arch) {
  if (access.origin == ConSanAccessOrigin::Flat) {
    return is_supported_flat_check_trap_access(
        access.kind, access.instruction_size, access.decoded_width_bits, access.mnemonic,
        access.operands.destination_vgpr, access.operands.address_vgpr, access.operands.data_vgpr,
        access.flat_address_space_hint, mode);
  }
  return is_supported_lds_check_trap_access(
      access.kind, access.instruction_size, access.decoded_width_bits, access.mnemonic,
      access.operands.destination_vgpr, access.operands.destination_accvgpr,
      access.operands.address_vgpr, access.operands.data_vgpr, access.operands.second_data_vgpr,
      arch);
}

ConSanTransformArtifacts
retry_patch_consan_moi_from_inventory(ConSanTransformArtifacts inventory_artifacts,
                                      ConSanOptions options,
                                      std::span<const uint8_t> code_object_bytes) {
  const major_image_ownership::ScopedOwner input_owner(major_image_ownership::OwnerKind::InputImage,
                                                       code_object_bytes.data(),
                                                       code_object_bytes.size());
  try {
    ConSanTransformArtifacts inventory = std::move(inventory_artifacts);
    options.patched_image_growth_input_bytes = code_object_bytes.size();
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
      return finalize_consan_result(std::move(inventory), code_object_bytes);
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
      return finalize_consan_result(std::move(inventory), code_object_bytes);
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
      return finalize_consan_result(std::move(inventory), code_object_bytes);
    }
    if (has_late_fault) {
      options.faults_preapplied = false;
      // A pristine report-sizing inventory deliberately excludes the live
      // mutation. Rebuild for the rare late-fault path instead of coupling
      // the retained inventory to every mutation-specific analysis choice.
      ConSanTransformArtifacts result = try_patch_consan_impl(code_object_bytes, options);
      try_apply_unmatched_barrier_wait_abort(code_object_bytes, options, result);
      return finalize_consan_result(std::move(result), code_object_bytes,
                                    options.moi_report_dispatch_id);
    }
    ConSanTransformArtifacts result =
        try_patch_consan_moi(std::move(inventory), options, code_object_bytes, arch);
    try_apply_unmatched_barrier_wait_abort(code_object_bytes, options, result);
    return finalize_consan_result(std::move(result), code_object_bytes,
                                  options.moi_report_dispatch_id);
  } catch (const std::exception &error) {
    ConSanTransformArtifacts result;
    result.errors.emplace_back(std::string("ConSan MOI inventory retry threw an exception: ") +
                               error.what());
    return finalize_consan_result(std::move(result), code_object_bytes);
  } catch (...) {
    ConSanTransformArtifacts result;
    result.errors.emplace_back("ConSan MOI inventory retry threw a non-standard exception");
    return finalize_consan_result(std::move(result), code_object_bytes);
  }
}

ConSanTransformArtifacts
complete_consan_lowering(std::span<const uint8_t> code_object_bytes, const ConSanOptions &options,
                         ConSanPerturbationPlanningState *inspected_perturbation = nullptr) {
  const major_image_ownership::ScopedOwner input_owner(major_image_ownership::OwnerKind::InputImage,
                                                       code_object_bytes.data(),
                                                       code_object_bytes.size());
  try {
    ConSanOptions effective_options = options;
    effective_options.patched_image_growth_input_bytes = code_object_bytes.size();
    ConSanTransformArtifacts result = try_patch_consan_impl(
        code_object_bytes, effective_options, {}, std::nullopt, inspected_perturbation);
    try_apply_unmatched_barrier_wait_abort(code_object_bytes, effective_options, result);
    result = finalize_consan_result(std::move(result), code_object_bytes,
                                    options.moi_report_dispatch_id, false, inspected_perturbation);
    return result;
  } catch (const std::exception &error) {
    ConSanTransformArtifacts result;
    result.errors.emplace_back(std::string("ConSan transform threw an exception: ") + error.what());
    return finalize_consan_result(std::move(result), code_object_bytes);
  } catch (...) {
    ConSanTransformArtifacts result;
    result.errors.emplace_back("ConSan transform threw a non-standard exception");
    return finalize_consan_result(std::move(result), code_object_bytes);
  }
}

ConSanTransformArtifacts lower_consan(std::span<const uint8_t> code_object_bytes,
                                      const ConSanOptions &options) {
  return complete_consan_lowering(code_object_bytes, options);
}

} // namespace rocjitsu
