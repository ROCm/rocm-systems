// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_text_relocation.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/kernel_descriptor_scan.h"
#include "rocjitsu/code/patch/consan/consan_capability_contract.h"
#include "rocjitsu/code/patch/consan/consan_descriptor_growth.h"
#include "rocjitsu/code/patch/consan/consan_growth_policy.h"
#include "rocjitsu/code/patch/consan/consan_placement.h"

#include <algorithm>
#include <iterator>
#include <limits>
#include <string>

namespace rocjitsu {

bool stage_consan_text_rewrites(
    const AmdGpuCodeObject &code_object, rj_code_arch_t arch,
    const ConSanDescriptorMutationBatch &descriptor_mutations,
    const ConSanDescriptorMutationPolicy &descriptor_policy, std::string_view subject,
    std::vector<ConSanStagedTextRewrite> rewrites, ConSanTransformArtifacts &result) {
  const auto source = result.replacement.empty()
                          ? std::span<const uint8_t>(
                                reinterpret_cast<const uint8_t *>(code_object.image_data()),
                                code_object.image_size())
                          : std::span<const uint8_t>(result.replacement);
  std::vector<uint8_t> descriptor_image(source.begin(), source.end());
  if (!apply_consan_descriptor_mutations_to_bytes(
          descriptor_image, result.program_inventory, descriptor_mutations, descriptor_policy,
          arch, subject, result.errors))
    return false;

  result.staged_text_rewrites.insert(result.staged_text_rewrites.end(),
                                     std::make_move_iterator(rewrites.begin()),
                                     std::make_move_iterator(rewrites.end()));
  result.replacement = std::move(descriptor_image);
  result.mark_modified();
  return true;
}

std::optional<ConSanRelocatedText> relocate_consan_text(
    std::span<const uint8_t> descriptor_patched_image, rj_code_arch_t arch,
    std::span<const ConSanInlineTextRewrite> rewrites,
    const ConSanPatchedImageGrowthLimit &growth_limit, const ConSanCodeObjectId &input_id,
    std::string_view operation, std::span<const SourceTextCodeRange> additional_code_ranges,
    std::vector<std::string> &errors,
    std::optional<ConSanTransformFailureCause> *failure_cause) {
  AmdGpuCodeObject source(descriptor_patched_image.data(), descriptor_patched_image.size());
  if (!source.is_valid() || source.text_sections().size() != 1u || !input_id.valid()) {
    errors.emplace_back("ConSan " + std::string(operation) +
                        " cannot prepare one executable text relocation transaction");
    return std::nullopt;
  }
  const uint64_t source_text_size = source.text_sections().front()->size();
  BinaryTranslatorOptions options;
  options.preserve_source_text_prefix = true;
  options.preserve_source_descriptor_resources = true;
  options.source_text_code_ranges.reserve(source.functions().size());
  for (const AmdGpuFunctionInfo &function : source.functions()) {
    if (function.code_size != 0u) {
      options.source_text_code_ranges.push_back(
          {.start_offset = function.entry_text_offset, .size = function.code_size});
    }
  }
  options.source_text_code_ranges.insert(options.source_text_code_ranges.end(),
                                         additional_code_ranges.begin(),
                                         additional_code_ranges.end());
  if (const ConSanTargetProfile *target = consan_target_profile(arch);
      target != nullptr &&
      target->identity_translation_revision != ProcessorRevision::Unspecified) {
    options.input_revision = target->identity_translation_revision;
    options.output_revision = target->identity_translation_revision;
  }
  if (descriptor_patched_image.size() < sizeof(Elf64_Ehdr)) {
    errors.emplace_back("ConSan " + std::string(operation) + " has no complete ELF header");
    return std::nullopt;
  }
  const auto *header = reinterpret_cast<const Elf64_Ehdr *>(descriptor_patched_image.data());
  BinaryTranslator translator(arch, arch, header->e_flags & EF_AMDGPU_MACH, options);
  translator.set_instruction_rewrite_callback(
      [&](const Instruction &, uint64_t source_offset)
          -> std::optional<std::vector<uint32_t>> {
        const auto rewrite = std::ranges::find(rewrites, source_offset,
                                               &ConSanInlineTextRewrite::source_offset);
        if (rewrite == rewrites.end())
          return std::nullopt;
        return rewrite->words;
      });
  TranslatedCodeObject translated = translator.translate(source);
  if (!translated.dispatchable()) {
    errors.emplace_back("ConSan " + std::string(operation) +
                        " could not relocate executable text");
    for (const TranslationDiagnostic &diagnostic : translated.diagnostics) {
      if (diagnostic.severity == DiagnosticSeverity::Error)
        errors.emplace_back("ConSan text relocation: " + diagnostic.message);
    }
    return std::nullopt;
  }

  const size_t input_image_bytes = static_cast<size_t>(input_id.byte_size);
  const auto limit = consan_patched_image_growth_limit_bytes(growth_limit, input_image_bytes);
  const std::string policy =
      consan_patched_image_growth_policy_description(growth_limit, input_image_bytes);
  if (!limit) {
    errors.emplace_back("ConSan " + std::string(operation) +
                        " has an invalid patched-image growth policy (" + policy + ")");
    return std::nullopt;
  }
  const size_t required_growth = translated.elf_bytes.size() > input_image_bytes
                                     ? translated.elf_bytes.size() - input_image_bytes
                                     : 0u;
  if (required_growth > *limit) {
    if (failure_cause)
      *failure_cause = ConSanTransformFailureCause::PatchedImageGrowthLimit;
    errors.emplace_back("ConSan " + std::string(operation) +
                        " rejected patched-image file growth: required total " +
                        std::to_string(required_growth) + " bytes, limit " +
                        std::to_string(*limit) + " bytes (policy " + policy + ")");
    return std::nullopt;
  }
  AmdGpuCodeObject output(translated.elf_bytes.data(), translated.elf_bytes.size());
  if (!output.is_valid() || output.text_sections().size() != 1u ||
      output.text_sections().front()->size() < source_text_size) {
    errors.emplace_back("ConSan " + std::string(operation) +
                        " produced an invalid relocated executable image");
    return std::nullopt;
  }
  // Whole-text placement changes only executable entry coordinates. ConSan
  // has already applied its exact ABI/resource transaction to the input
  // descriptors, so retain those bytes rather than accepting DBT's general
  // cross-target descriptor normalization as a second owner of the ABI.
  for (const AmdGpuKernelInfo &source_kernel : source.kernels()) {
    const auto output_kernel = std::ranges::find(output.kernels(), source_kernel.name,
                                                 &AmdGpuKernelInfo::name);
    if (output_kernel == output.kernels().end()) {
      errors.emplace_back("ConSan " + std::string(operation) +
                          " lost a kernel while relocating executable text");
      return std::nullopt;
    }
    auto source_descriptor = read_kernel_descriptor(descriptor_patched_image,
                                                    source_kernel.descriptor_file_offset);
    const auto output_descriptor = read_kernel_descriptor(
        std::span<const uint8_t>(translated.elf_bytes), output_kernel->descriptor_file_offset);
    if (!source_descriptor || !output_descriptor) {
      errors.emplace_back("ConSan " + std::string(operation) +
                          " could not retain a relocated kernel descriptor");
      return std::nullopt;
    }
    source_descriptor->kernel_code_entry_byte_offset =
        output_descriptor->kernel_code_entry_byte_offset;
    if (!write_kernel_descriptor(std::span<uint8_t>(translated.elf_bytes),
                                 output_kernel->descriptor_file_offset, *source_descriptor)) {
      errors.emplace_back("ConSan " + std::string(operation) +
                          " could not publish a relocated kernel descriptor");
      return std::nullopt;
    }
  }
  return ConSanRelocatedText{
      .image = std::move(translated.elf_bytes),
      .placements = std::move(translated.text_placements),
      .source_text_size = source_text_size,
      .relocated_text_size = output.text_sections().front()->size() - source_text_size,
  };
}

bool finalize_consan_text_rewrites(std::span<const uint8_t> descriptor_image,
                                   rj_code_arch_t arch, const ConSanOptions &options,
                                   std::string_view subject, ConSanTransformArtifacts &result) {
  if (result.staged_text_rewrites.empty())
    return true;

  std::vector<ConSanInlineTextRewrite> rewrites;
  rewrites.reserve(result.staged_text_rewrites.size());
  for (const ConSanStagedTextRewrite &staged : result.staged_text_rewrites)
    rewrites.push_back({staged.patch.anchor_offset, staged.words});

  std::vector<SourceTextCodeRange> preexisting_patch_ranges;
  for (const ConSanPatchInfo &patch : result.patches) {
    if (patch.trampoline_size != 0u) {
      preexisting_patch_ranges.push_back(
          {.start_offset = patch.trampoline_offset, .size = patch.trampoline_size});
    }
  }

  auto relocated = relocate_consan_text(
      descriptor_image, arch, rewrites, options.patched_image_growth_limit,
      result.program_inventory.code_object_id(), subject, preexisting_patch_ranges, result.errors,
      &result.transform_failure_cause);
  if (!relocated)
    return false;
  if (relocated->source_text_size >
          std::numeric_limits<uint64_t>::max() - relocated->relocated_text_size ||
      relocated->relocated_text_size > std::numeric_limits<uint32_t>::max()) {
    result.errors.emplace_back("ConSan " + std::string(subject) +
                               " relocated text is too large");
    return false;
  }

  std::vector<ConSanPatchInfo> placed_patches;
  std::vector<ConSanCommittedLowering> commits;
  AmdGpuCodeObject relocated_object(relocated->image.data(), relocated->image.size());
  if (!relocated_object.is_valid() || relocated_object.text_sections().size() != 1u) {
    result.errors.emplace_back("ConSan " + std::string(subject) +
                               " cannot inspect relocated executable text");
    return false;
  }
  const uint64_t relocated_text_bytes = relocated_object.text_sections().front()->size();
  commits.reserve(result.staged_text_rewrites.size());
  for (const ConSanStagedTextRewrite &staged : result.staged_text_rewrites) {
    std::optional<ConSanPatchInfo> primary;
    for (const TranslatedTextPlacement &placement : relocated->placements) {
      if (placement.source_offset != staged.patch.anchor_offset || !placement.client_rewrite)
        continue;
      ConSanPatchInfo placed = staged.patch;
      placed.trampoline_offset = placement.target_offset;
      if (placed.trampoline_offset > relocated_text_bytes ||
          placed.trampoline_size > relocated_text_bytes - placed.trampoline_offset) {
        result.errors.emplace_back(
            "ConSan " + std::string(subject) + " placement exceeds relocated text: source=" +
            std::to_string(staged.patch.anchor_offset) +
            " target=" + std::to_string(placed.trampoline_offset) +
            " size=" + std::to_string(placed.trampoline_size) +
            " text=" + std::to_string(relocated_text_bytes));
        return false;
      }
      if (placed.relocated_guest_instruction_offset)
        *placed.relocated_guest_instruction_offset += placement.target_offset;
      if (!primary)
        primary = placed;
      placed_patches.push_back(std::move(placed));
    }
    if (!primary) {
      result.errors.emplace_back("ConSan " + std::string(subject) +
                                 " lost a relocated instruction placement");
      return false;
    }
    auto commit = make_consan_instrumented_patch_lowering(
        result.observation_plan(), staged.intent_ids, *primary);
    if (!commit) {
      result.errors.emplace_back("ConSan " + std::string(subject) +
                                 " produced an invalid intent-bound lowering");
      result.discard_candidate_modification();
      return false;
    }
    commits.push_back(std::move(*commit));
  }

  result.text_relocation = ConSanTextRelocationProof{
      relocated->source_text_size,
      relocated->relocated_text_size,
  };
  result.staged_text_rewrites.clear();
  return result.publish_access_lowering(std::move(relocated->image), subject, std::move(commits),
                                        std::move(placed_patches));
}

} // namespace rocjitsu
