// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_program_analysis.h"

#include "rocjitsu/analysis/def_use_chain.h"
#include "rocjitsu/analysis/kernel_scope.h"
#include "rocjitsu/analysis/liveness.h"
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/major_image_ownership.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/patch/consan/consan_cfg.h"
#include "rocjitsu/code/patch/consan/consan_descriptor.h"
#include "rocjitsu/code/patch/consan/consan_input_layout.h"
#include "rocjitsu/code/patch/consan/consan_instruction_semantics.h"
#include "rocjitsu/code/patch/consan/consan_physical_site_alias.h"
#include "rocjitsu/code/patch/consan/consan_semantic_classifiers.h"
#include "rocjitsu/code/patch/consan/consan_sync_analysis.h"
#include "rocjitsu/code/patch/consan/targets/consan_program_analysis_target_ops.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "util/bit.h"

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

#include "rocjitsu/code/patch/consan/consan_analysis.inc"

void reattribute_decoded_sites_in_range(std::vector<ConSanDecodedProgramSite> &sites,
                                        uint64_t begin, uint32_t size,
                                        const ConSanProgramContainerRef &owner) {
  const uint64_t end = begin + size;
  std::set<std::pair<size_t, uint64_t>> seen;
  std::erase_if(sites, [&](ConSanDecodedProgramSite &site) {
    if (site.text_offset() < begin || site.text_offset() >= end)
      return false;
    site.container = owner;
    return !seen.emplace(site.payload.index(), site.text_offset()).second;
  });
}

void reattribute_preapplied_code_ranges(std::span<const uint8_t> code_object_bytes,
                                        Decoder &decoder, rj_code_arch_t arch,
                                        ProgramInventoryBuilder &inventory,
                                        ConSanProgramAnalysisResult &result) {
  std::span<ConSanProgramContainer> kernels = inventory.kernels();
  for (const ConSanPreappliedCodeRange &range :
       result.program_inventory.preapplied_mutation().code_ranges) {
    const auto kernel = std::ranges::find_if(kernels, [&](const ConSanProgramContainer &item) {
      return item.name == range.kernel_name;
    });
    if (kernel == kernels.end()) {
      result.errors.emplace_back("ConSan could not recover the owner of a preapplied fault cave");
      continue;
    }
    const ConSanProgramContainerRef owner = consan_program_container_ref(*kernel);
    reattribute_decoded_sites_in_range(inventory.decoded_sites(), range.text_offset, range.size,
                                       owner);

    ConSanProgramContainer decoded_range;
    decoded_range.kind = ConSanProgramContainerKind::Kernel;
    decoded_range.name = kernel->name;
    decoded_range.entry_text_offset = range.text_offset;
    decoded_range.text_file_offset = kernel->text_file_offset;
    decoded_range.code_size = range.size;
    decoded_range.has_text_range = true;
    std::vector<ConSanAccessInventorySite> decoded_accesses;
    std::vector<ConSanDecodedProgramSite> decoded_sites;
    decode_consan_kernel_inventory(code_object_bytes, decoder, arch, decoded_range,
                                   decoded_accesses, decoded_sites, result.warnings);
    inventory.reattribute_access_range(range.text_offset, range.size, *kernel, decoded_accesses,
                                       code_object_bytes);
    for (ConSanDecodedProgramSite &site : decoded_sites) {
      site.container = owner;
      const bool duplicate = std::ranges::any_of(
          inventory.decoded_sites(), [&](const ConSanDecodedProgramSite &existing) {
            return existing.container == owner &&
                   existing.payload.index() == site.payload.index() &&
                   existing.text_offset() == site.text_offset();
          });
      if (!duplicate)
        inventory.decoded_sites().push_back(std::move(site));
    }
    std::ranges::sort(inventory.decoded_sites(), {}, &ConSanDecodedProgramSite::text_offset);
  }
}

} // namespace

void decode_consan_kernel_inventory(std::span<const uint8_t> code_object_bytes, Decoder &decoder,
                                    rj_code_arch_t arch, ConSanProgramContainer &kernel,
                                    std::vector<ConSanAccessInventorySite> &accesses,
                                    std::vector<ConSanDecodedProgramSite> &decoded_sites,
                                    std::vector<std::string> &warnings) {
  kernel.kind = ConSanProgramContainerKind::Kernel;
  decode_kernel_stats(code_object_bytes, decoder, arch, kernel, accesses, decoded_sites, warnings);
}

void decode_consan_function_inventory(std::span<const uint8_t> code_object_bytes, Decoder &decoder,
                                      rj_code_arch_t arch, ConSanProgramContainer &function,
                                      std::vector<ConSanAccessInventorySite> &accesses,
                                      std::vector<ConSanDecodedProgramSite> &decoded_sites,
                                      std::vector<std::string> &warnings) {
  function.kind = ConSanProgramContainerKind::Function;
  decode_function_stats(code_object_bytes, decoder, arch, function, accesses, decoded_sites,
                        warnings);
}

void refine_consan_flat_pointer_provenance(std::span<const uint8_t> code_object_bytes,
                                           const AmdGpuCodeObject &code_object, rj_code_arch_t arch,
                                           std::span<ConSanProgramContainer> kernels,
                                           std::span<ConSanProgramContainer> functions,
                                           std::vector<ConSanAccessInventorySite> &accesses,
                                           std::vector<ConSanDecodedProgramSite> &decoded_sites,
                                           std::vector<std::string> &warnings) {
  relay_flat_pointer_provenance_across_calls(code_object_bytes, code_object, arch, kernels,
                                             functions, accesses, decoded_sites, warnings);
}

void prune_consan_unreachable_inferred_ranges(
    const AmdGpuCodeObject &code_object, Decoder &decoder, rj_code_arch_t arch,
    std::span<const ConSanPreappliedCodeRange> preapplied_ranges,
    std::span<ConSanProgramContainer> containers, std::vector<ConSanAccessInventorySite> &accesses,
    std::vector<ConSanDecodedProgramSite> &decoded_sites) {
  prune_unreachable_inferred_ranges(code_object, decoder, arch, preapplied_ranges, containers,
                                    accesses, decoded_sites);
}

void preflight_consan_kernel(ConSanProgramContainer &kernel, std::vector<std::string> &warnings) {
  preflight_kernel(kernel, warnings);
}

bool analyze_consan_program_inventory(std::span<const uint8_t> code_object_bytes,
                                      const ConSanOptions &options,
                                      std::unique_ptr<AmdGpuCodeObject> &code_object,
                                      ProgramInventoryBuilder &inventory_builder,
                                      ConSanPerturbationPlanningState &perturbation,
                                      ConSanProgramAnalysisResult &result) {
  // Even a parse failure publishes the identity-bearing empty view. Pipeline
  // stage accounting distinguishes a completed, invalid inventory attempt
  // from an analysis stage that was never entered.
  result.program_inventory = inventory_builder.view();
  if (code_object_bytes.empty()) {
    result.errors.emplace_back("ConSan received an empty code object");
    return false;
  }
  code_object =
      std::make_unique<AmdGpuCodeObject>(code_object_bytes.data(), code_object_bytes.size());
  if (!code_object->is_valid()) {
    result.errors.emplace_back("ConSan could not parse AMDGPU code object");
    return false;
  }
  const bool kernel_metadata_trustworthy = code_object->kernel_metadata_is_trustworthy();
  const size_t malformed_kernel_metadata_note_count =
      code_object->malformed_kernel_metadata_note_count();
  const rj_code_target_id_t target = code_object->target_id();
  const ConSanTargetProfile *target_profile = consan_target_profile(target);
  const rj_code_arch_t arch = target_profile ? target_profile->arch : ROCJITSU_CODE_ARCH_INVALID;
  inventory_builder.set_code_object_facts(kernel_metadata_trustworthy,
                                          malformed_kernel_metadata_note_count, arch, target);
  result.errors = validate_consan_input_layout(*code_object);
  if (!result.errors.empty())
    return false;

  for (const Section *section : code_object->text_sections()) {
    ConSanTextSection info;
    info.name = section->name();
    info.file_offset = section->sectionOffset();
    info.virtual_address = section->vaddr();
    info.size = section->size();
    inventory_builder.text_sections().push_back(std::move(info));
  }

  for (const AmdGpuKernelInfo &kernel : code_object->kernels()) {
    ConSanProgramContainer info;
    info.name = kernel.name;
    info.descriptor_file_offset = kernel.descriptor_file_offset;
    if (const auto descriptor =
            read_kernel_descriptor(code_object_bytes, kernel.descriptor_file_offset))
      info.declared_group_segment_bytes = descriptor->group_segment_fixed_size;
    info.entry_text_offset = kernel.entry_text_offset;
    info.text_file_offset = kernel.text_file_offset;
    info.code_size = kernel.code_size;
    info.code_size_inferred_from_zero = kernel.code_size_inferred_from_zero;
    info.has_text_range = kernel.has_text_range;
    info.has_dynamic_lds = kernel.has_dynamic_lds;
    info.uses_dynamic_stack = kernel.uses_dynamic_stack;
    if (kernel_metadata_trustworthy) {
      info.vgpr_count = kernel.vgpr_count;
      info.agpr_count = kernel.agpr_count;
    }
    info.sgpr_count = kernel.sgpr_count;
    info.required_workgroup_size = kernel.required_workgroup_size;
    inventory_builder.add_kernel(std::move(info));
  }

  std::unordered_set<std::string> kernel_names;
  kernel_names.reserve(code_object->kernels().size());
  for (const AmdGpuKernelInfo &kernel : code_object->kernels())
    kernel_names.insert(kernel.name);

  for (const AmdGpuFunctionInfo &function : code_object->functions()) {
    if (kernel_names.contains(function.name))
      continue;
    ConSanProgramContainer info;
    info.name = function.name;
    info.entry_text_offset = function.entry_text_offset;
    info.text_file_offset = function.text_file_offset;
    info.code_size = function.code_size;
    info.code_size_inferred_from_zero = function.code_size_inferred_from_zero;
    inventory_builder.add_function(std::move(info));
  }

  result.program_inventory = inventory_builder.view();
  const auto publish_access_inventory = [&] {
    inventory_builder.publish_decoded_accesses(code_object_bytes);
    result.program_inventory = inventory_builder.view();
  };

  if (result.program_inventory.text_sections().empty())
    result.warnings.emplace_back("ConSan found no .text sections");
  if (result.program_inventory.kernels().empty())
    result.warnings.emplace_back("ConSan found no kernel descriptor symbols");

  if (arch == ROCJITSU_CODE_ARCH_INVALID) {
    publish_access_inventory();
    result.warnings.emplace_back("ConSan does not support target '" +
                                 std::string(rj_code_target_name(target)) + "'");
    result.outcome = ConSanTransformOutcome::Unsupported;
    return false;
  }
  inventory_builder.set_semantic_arch_required(true);

  std::unique_ptr<Decoder> decoder = Decoder::create(arch);
  if (!decoder) {
    publish_access_inventory();
    result.warnings.emplace_back("ConSan could not create decoder for arch '" +
                                 std::string(rj_code_arch_name(arch)) + "'");
    result.outcome = ConSanTransformOutcome::Unsupported;
    return false;
  }

  for (ConSanProgramContainer &kernel : inventory_builder.kernels()) {
    decode_consan_kernel_inventory(code_object_bytes, *decoder, arch, kernel,
                                   inventory_builder.access_sites(),
                                   inventory_builder.decoded_sites(), result.warnings);
  }
  std::vector<ConSanAccessInventorySite> function_accesses;
  std::vector<ConSanDecodedProgramSite> function_decoded_sites;
  for (ConSanProgramContainer &function : inventory_builder.functions())
    decode_consan_function_inventory(code_object_bytes, *decoder, arch, function, function_accesses,
                                     function_decoded_sites, result.warnings);
  const auto append_function_accesses = [&] {
    inventory_builder.access_sites().insert(inventory_builder.access_sites().end(),
                                            std::make_move_iterator(function_accesses.begin()),
                                            std::make_move_iterator(function_accesses.end()));
    function_accesses.clear();
    inventory_builder.decoded_sites().insert(
        inventory_builder.decoded_sites().end(),
        std::make_move_iterator(function_decoded_sites.begin()),
        std::make_move_iterator(function_decoded_sites.end()));
    function_decoded_sites.clear();
  };
  const bool has_decode_error =
      std::ranges::any_of(result.program_inventory.kernels(),
                          [](const ConSanProgramContainer &kernel) {
                            return kernel.stats.decode_error_count != 0u;
                          }) ||
      std::ranges::any_of(result.program_inventory.functions(),
                          [](const ConSanProgramContainer &function) {
                            return function.stats.decode_error_count != 0u;
                          });
  if (has_decode_error) {
    append_function_accesses();
    publish_access_inventory();
    result.outcome = ConSanTransformOutcome::Unsupported;
    result.warnings.emplace_back(
        "ConSan stopped before CFG analysis because an instruction could not be decoded");
    return false;
  }
  refine_consan_flat_pointer_provenance(code_object_bytes, *code_object, arch,
                                        inventory_builder.kernels(), inventory_builder.functions(),
                                        function_accesses, function_decoded_sites, result.warnings);
  append_function_accesses();
  publish_access_inventory();
  if (options.flavor == ConSanFlavor::SuperCollider && !options.fault_dry_run &&
      options.sc_perturb_kind == ConSanPerturbationKind::None) {
    for (ConSanProgramContainer &kernel : inventory_builder.kernels())
      preflight_consan_kernel(kernel, result.warnings);
  }
  reattribute_preapplied_code_ranges(code_object_bytes, *decoder, arch, inventory_builder, result);
  if (!result.errors.empty())
    return false;
  prune_consan_unreachable_inferred_ranges(
      *code_object, *decoder, arch, result.program_inventory.preapplied_mutation().code_ranges,
      inventory_builder.containers(), inventory_builder.access_sites(),
      inventory_builder.decoded_sites());
  return analyze_consan_semantic_inventory(code_object_bytes, *code_object, *decoder, arch, options,
                                           inventory_builder, perturbation, result);
}

} // namespace rocjitsu
