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
#include "rocjitsu/code/patch/consan/consan_program_analysis_target_ops.h"
#include "rocjitsu/code/patch/consan/consan_semantic_classifiers.h"
#include "rocjitsu/code/patch/consan/consan_sync_analysis.h"
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

template <typename Site>
void move_sites_in_range(std::vector<Site> &source, std::vector<Site> &destination, uint64_t begin,
                         uint32_t size) {
  const uint64_t end = begin + size;
  for (auto it = source.begin(); it != source.end();) {
    if (it->text_offset < begin || it->text_offset >= end) {
      ++it;
      continue;
    }
    destination.push_back(std::move(*it));
    it = source.erase(it);
  }
}

template <typename Site>
void append_unique_sites(std::vector<Site> &source, std::vector<Site> &destination) {
  for (Site &site : source) {
    if (std::ranges::none_of(destination, [&](const Site &existing) {
          return existing.text_offset == site.text_offset;
        })) {
      destination.push_back(std::move(site));
    }
  }
  std::ranges::sort(destination, {}, &Site::text_offset);
}

void reattribute_preapplied_code_ranges(std::span<const uint8_t> code_object_bytes,
                                        Decoder &decoder, rj_code_arch_t arch,
                                        ProgramInventoryBuilder &inventory,
                                        ConSanProgramAnalysisResult &result) {
  std::span<ConSanKernelInfo> kernels = inventory.kernels();
  std::span<ConSanFunctionInfo> functions = inventory.functions();
  for (const ConSanPreappliedCodeRange &range :
       result.program_inventory.preapplied_mutation().code_ranges) {
    const auto kernel = std::ranges::find_if(
        kernels, [&](const ConSanKernelInfo &item) { return item.name == range.kernel_name; });
    if (kernel == kernels.end()) {
      result.errors.emplace_back("ConSan could not recover the owner of a preapplied fault cave");
      continue;
    }
    for (ConSanKernelInfo &other : kernels) {
      if (&other == &*kernel)
        continue;
      move_sites_in_range(other.ordinary_memory_sites, kernel->ordinary_memory_sites,
                          range.text_offset, range.size);
      move_sites_in_range(other.barrier_sites, kernel->barrier_sites, range.text_offset,
                          range.size);
      move_sites_in_range(other.fence_sites, kernel->fence_sites, range.text_offset, range.size);
      move_sites_in_range(other.atomic_sites, kernel->atomic_sites, range.text_offset, range.size);
    }
    for (ConSanFunctionInfo &function : functions) {
      move_sites_in_range(function.ordinary_memory_sites, kernel->ordinary_memory_sites,
                          range.text_offset, range.size);
      move_sites_in_range(function.barrier_sites, kernel->barrier_sites, range.text_offset,
                          range.size);
      move_sites_in_range(function.fence_sites, kernel->fence_sites, range.text_offset, range.size);
      move_sites_in_range(function.atomic_sites, kernel->atomic_sites, range.text_offset,
                          range.size);
    }

    ConSanKernelInfo decoded_range;
    decoded_range.name = kernel->name;
    decoded_range.entry_text_offset = range.text_offset;
    decoded_range.text_file_offset = kernel->text_file_offset;
    decoded_range.code_size = range.size;
    decoded_range.has_text_range = true;
    std::vector<ConSanAccessInventorySite> decoded_accesses;
    decode_consan_kernel_inventory(code_object_bytes, decoder, arch, decoded_range,
                                   decoded_accesses, result.warnings);
    inventory.reattribute_access_range(range.text_offset, range.size, *kernel, decoded_accesses,
                                       code_object_bytes);
    append_unique_sites(decoded_range.ordinary_memory_sites, kernel->ordinary_memory_sites);
    append_unique_sites(decoded_range.barrier_sites, kernel->barrier_sites);
    append_unique_sites(decoded_range.fence_sites, kernel->fence_sites);
    append_unique_sites(decoded_range.atomic_sites, kernel->atomic_sites);
  }
}

} // namespace

void decode_consan_kernel_inventory(std::span<const uint8_t> code_object_bytes, Decoder &decoder,
                                    rj_code_arch_t arch, ConSanKernelInfo &kernel,
                                    std::vector<ConSanAccessInventorySite> &accesses,
                                    std::vector<std::string> &warnings) {
  decode_kernel_stats(code_object_bytes, decoder, arch, kernel, accesses, warnings);
}

void decode_consan_function_inventory(std::span<const uint8_t> code_object_bytes, Decoder &decoder,
                                      rj_code_arch_t arch, ConSanFunctionInfo &function,
                                      std::vector<ConSanAccessInventorySite> &accesses,
                                      std::vector<std::string> &warnings) {
  decode_function_stats(code_object_bytes, decoder, arch, function, accesses, warnings);
}

void refine_consan_flat_pointer_provenance(std::span<const uint8_t> code_object_bytes,
                                           const AmdGpuCodeObject &code_object, rj_code_arch_t arch,
                                           std::vector<ConSanKernelInfo> &kernels,
                                           std::vector<ConSanFunctionInfo> &functions,
                                           std::vector<ConSanAccessInventorySite> &accesses,
                                           std::vector<std::string> &warnings) {
  relay_flat_pointer_provenance_across_calls(code_object_bytes, code_object, arch, kernels,
                                             functions, accesses, warnings);
}

void prune_consan_unreachable_inferred_ranges(
    const AmdGpuCodeObject &code_object, Decoder &decoder, rj_code_arch_t arch,
    std::span<const ConSanPreappliedCodeRange> preapplied_ranges,
    std::span<ConSanKernelInfo> kernels, std::span<ConSanFunctionInfo> functions,
    std::vector<ConSanAccessInventorySite> &accesses) {
  prune_unreachable_inferred_ranges(code_object, decoder, arch, preapplied_ranges, kernels,
                                    functions, accesses);
}

void preflight_consan_kernel(ConSanKernelInfo &kernel, std::vector<std::string> &warnings) {
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
    ConSanKernelInfo info;
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
    inventory_builder.kernels().push_back(std::move(info));
  }

  std::unordered_set<std::string> kernel_names;
  kernel_names.reserve(code_object->kernels().size());
  for (const AmdGpuKernelInfo &kernel : code_object->kernels())
    kernel_names.insert(kernel.name);

  for (const AmdGpuFunctionInfo &function : code_object->functions()) {
    if (kernel_names.contains(function.name))
      continue;
    ConSanFunctionInfo info;
    info.name = function.name;
    info.entry_text_offset = function.entry_text_offset;
    info.text_file_offset = function.text_file_offset;
    info.code_size = function.code_size;
    info.code_size_inferred_from_zero = function.code_size_inferred_from_zero;
    inventory_builder.functions().push_back(std::move(info));
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

  for (ConSanKernelInfo &kernel : inventory_builder.kernels()) {
    decode_consan_kernel_inventory(code_object_bytes, *decoder, arch, kernel,
                                   inventory_builder.access_sites(), result.warnings);
  }
  std::vector<ConSanAccessInventorySite> function_accesses;
  for (ConSanFunctionInfo &function : inventory_builder.functions())
    decode_consan_function_inventory(code_object_bytes, *decoder, arch, function, function_accesses,
                                     result.warnings);
  const auto append_function_accesses = [&] {
    inventory_builder.access_sites().insert(inventory_builder.access_sites().end(),
                                            std::make_move_iterator(function_accesses.begin()),
                                            std::make_move_iterator(function_accesses.end()));
    function_accesses.clear();
  };
  const bool has_decode_error =
      std::ranges::any_of(
          result.program_inventory.kernels(),
          [](const ConSanKernelInfo &kernel) { return kernel.stats.decode_error_count != 0u; }) ||
      std::ranges::any_of(result.program_inventory.functions(),
                          [](const ConSanFunctionInfo &function) {
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
                                        function_accesses, result.warnings);
  append_function_accesses();
  publish_access_inventory();
  if (options.flavor == ConSanFlavor::SuperCollider && !options.fault_dry_run &&
      options.sc_perturb_kind == ConSanPerturbationKind::None) {
    for (ConSanKernelInfo &kernel : inventory_builder.kernels())
      preflight_consan_kernel(kernel, result.warnings);
  }
  reattribute_preapplied_code_ranges(code_object_bytes, *decoder, arch, inventory_builder, result);
  if (!result.errors.empty())
    return false;
  prune_consan_unreachable_inferred_ranges(
      *code_object, *decoder, arch, result.program_inventory.preapplied_mutation().code_ranges,
      inventory_builder.kernels(), inventory_builder.functions(), inventory_builder.access_sites());
  return analyze_consan_semantic_inventory(code_object_bytes, *code_object, *decoder, arch, options,
                                           inventory_builder, perturbation, result);
}

} // namespace rocjitsu
