// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_text_relocation.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/dbt/binary_translator.h"
#include "rocjitsu/code/kernel_descriptor_scan.h"
#include "rocjitsu/code/patch/consan/consan_capability_contract.h"
#include "rocjitsu/code/patch/consan/consan_descriptor_growth.h"
#include "rocjitsu/code/patch/consan/consan_growth_policy.h"
#include "rocjitsu/code/patch/consan/consan_placement.h"

#include <algorithm>
#include <cstring>
#include <iterator>
#include <string>

namespace rocjitsu {

namespace {

struct ConSanRelocatedText {
  std::vector<uint8_t> image;
  std::vector<TranslatedTextPlacement> placements;
  uint64_t text_size = 0;
  std::optional<ConSanTextRelocationProof> relocation;
};

[[nodiscard]] std::optional<ConSanRelocatedText>
try_rewrite_consan_text_in_place(std::span<const uint8_t> image, const AmdGpuCodeObject &source,
                                 rj_code_arch_t arch,
                                 std::span<const ConSanStagedTextRewrite> rewrites) {
  if (source.text_sections().size() != 1u)
    return std::nullopt;
  const Section &text = *source.text_sections().front();
  std::vector<ByteRange> ranges;
  for (const ConSanStagedTextRewrite &rewrite : rewrites) {
    const uint64_t replacement_size = rewrite.words.size() * sizeof(uint32_t);
    if (rewrite.patch.original_size == 0u || rewrite.patch.original_size % sizeof(uint32_t) != 0u ||
        replacement_size < rewrite.patch.original_size ||
        rewrite.patch.anchor_offset > text.size() ||
        replacement_size > text.size() - rewrite.patch.anchor_offset)
      return std::nullopt;
    const uint32_t padding_words =
        static_cast<uint32_t>((replacement_size - rewrite.patch.original_size) / sizeof(uint32_t));
    const uint64_t padding_file_offset =
        text.sectionOffset() + rewrite.patch.anchor_offset + rewrite.patch.original_size;
    if (count_nop_padding(image, padding_file_offset, arch, padding_words) != padding_words)
      return std::nullopt;
    const ByteRange range{rewrite.patch.anchor_offset,
                          rewrite.patch.anchor_offset + replacement_size};
    if (std::ranges::any_of(ranges, [&](ByteRange other) { return ranges_overlap(range, other); }))
      return std::nullopt;
    ranges.push_back(range);
  }

  ConSanRelocatedText result;
  result.image.assign(image.begin(), image.end());
  result.text_size = text.size();
  result.placements.reserve(rewrites.size());
  for (const ConSanStagedTextRewrite &rewrite : rewrites) {
    std::memcpy(result.image.data() + text.sectionOffset() + rewrite.patch.anchor_offset,
                rewrite.words.data(), rewrite.words.size() * sizeof(uint32_t));
    result.placements.push_back({.source_offset = rewrite.patch.anchor_offset,
                                 .target_offset = rewrite.patch.anchor_offset,
                                 .client_rewrite = true});
  }
  return result;
}

} // namespace

bool stage_consan_text_rewrites(const AmdGpuCodeObject &code_object, rj_code_arch_t arch,
                                const ConSanDescriptorMutationBatch &descriptor_mutations,
                                const ConSanDescriptorMutationPolicy &descriptor_policy,
                                std::string_view subject,
                                std::vector<ConSanStagedTextRewrite> rewrites,
                                ConSanTransformArtifacts &result) {
  const auto source =
      result.replacement.empty()
          ? std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(code_object.image_data()),
                                     code_object.image_size())
          : std::span<const uint8_t>(result.replacement);
  std::vector<uint8_t> descriptor_image(source.begin(), source.end());
  if (!apply_consan_descriptor_mutations_to_bytes(descriptor_image, result.program_inventory,
                                                  descriptor_mutations, descriptor_policy, arch,
                                                  subject, result.errors))
    return false;

  result.staged_text_rewrites.insert(result.staged_text_rewrites.end(),
                                     std::make_move_iterator(rewrites.begin()),
                                     std::make_move_iterator(rewrites.end()));
  result.replacement = std::move(descriptor_image);
  result.mark_modified();
  return true;
}

namespace {

std::optional<ConSanRelocatedText>
relocate_consan_text(std::span<const uint8_t> descriptor_patched_image, rj_code_arch_t arch,
                     const ConSanOptions &consan_options, std::string_view operation,
                     ConSanTransformArtifacts &result) {
  const auto &rewrites = result.staged_text_rewrites;
  const ConSanCodeObjectId &input_id = result.program_inventory.code_object_id();
  std::vector<std::string> &errors = result.errors;
  const std::string error_prefix = "ConSan " + std::string(operation);
  AmdGpuCodeObject source(descriptor_patched_image.data(), descriptor_patched_image.size());
  if (!source.is_valid() || source.text_sections().size() != 1u || !input_id.valid()) {
    errors.emplace_back(error_prefix +
                        " cannot prepare one executable text relocation transaction");
    return std::nullopt;
  }
  const uint64_t source_text_size = source.text_sections().front()->size();
  BinaryTranslatorOptions translator_options;
  translator_options.preserve_source_text_prefix = true;
  translator_options.preserve_source_descriptor_resources = true;
  translator_options.source_text_code_ranges.reserve(source.functions().size() +
                                                     result.patches.size());
  for (const AmdGpuFunctionInfo &function : source.functions()) {
    if (function.code_size != 0u) {
      translator_options.source_text_code_ranges.push_back(
          {.start_offset = function.entry_text_offset, .size = function.code_size});
    }
  }
  bool has_preexisting_code = false;
  for (const ConSanPatchInfo &patch : result.patches) {
    if (patch.trampoline_size == 0u)
      continue;
    translator_options.source_text_code_ranges.push_back(
        {.start_offset = patch.trampoline_offset, .size = patch.trampoline_size});
    has_preexisting_code = true;
  }
  if (const ConSanTargetProfile *target = consan_target_profile(arch);
      target != nullptr &&
      target->identity_translation_revision != ProcessorRevision::Unspecified) {
    translator_options.input_revision = target->identity_translation_revision;
    translator_options.output_revision = target->identity_translation_revision;
  }
  if (descriptor_patched_image.size() < sizeof(Elf64_Ehdr)) {
    errors.emplace_back(error_prefix + " has no complete ELF header");
    return std::nullopt;
  }
  const auto *header = reinterpret_cast<const Elf64_Ehdr *>(descriptor_patched_image.data());
  BinaryTranslator translator(arch, arch, header->e_flags & EF_AMDGPU_MACH, translator_options);
  translator.set_instruction_rewrite_callback(
      [&](const InstructionRewriteContext &context) -> std::optional<std::vector<uint32_t>> {
        const auto rewrite = std::ranges::find(
            rewrites, context.source_offset,
            [](const ConSanStagedTextRewrite &rewrite) { return rewrite.patch.anchor_offset; });
        if (rewrite == rewrites.end())
          return std::nullopt;
        return rewrite->words;
      });
  TranslatedCodeObject translated = translator.translate(source);
  if (!translated.dispatchable()) {
    errors.emplace_back(error_prefix + " could not relocate executable text");
    for (const TranslationDiagnostic &diagnostic : translated.diagnostics) {
      if (diagnostic.severity == DiagnosticSeverity::Error)
        errors.emplace_back("ConSan text relocation: " + diagnostic.message);
    }
    return std::nullopt;
  }

  const size_t input_image_bytes = static_cast<size_t>(input_id.byte_size);
  const auto limit = consan_patched_image_growth_limit_bytes(
      consan_options.patched_image_growth_limit, input_image_bytes);
  const std::string policy = consan_patched_image_growth_policy_description(
      consan_options.patched_image_growth_limit, input_image_bytes);
  if (!limit) {
    errors.emplace_back(error_prefix + " has an invalid patched-image growth policy (" + policy +
                        ")");
    return std::nullopt;
  }
  const size_t required_growth = translated.elf_bytes.size() > input_image_bytes
                                     ? translated.elf_bytes.size() - input_image_bytes
                                     : 0u;
  if (required_growth > *limit) {
    if (!has_preexisting_code) {
      auto in_place =
          try_rewrite_consan_text_in_place(descriptor_patched_image, source, arch, rewrites);
      const size_t in_place_growth = in_place && in_place->image.size() > input_image_bytes
                                         ? in_place->image.size() - input_image_bytes
                                         : 0u;
      if (in_place && in_place_growth <= *limit)
        return in_place;
    }
    result.transform_failure_cause = ConSanTransformFailureCause::PatchedImageGrowthLimit;
    errors.emplace_back(error_prefix + " rejected patched-image file growth: required total " +
                        std::to_string(required_growth) + " bytes, limit " +
                        std::to_string(*limit) + " bytes (policy " + policy + ")");
    return std::nullopt;
  }
  AmdGpuCodeObject output(translated.elf_bytes.data(), translated.elf_bytes.size());
  if (!output.is_valid() || output.text_sections().size() != 1u ||
      output.text_sections().front()->size() < source_text_size) {
    errors.emplace_back(error_prefix + " produced an invalid relocated executable image");
    return std::nullopt;
  }
  // Whole-text placement changes only executable entry coordinates. ConSan
  // has already applied its exact ABI/resource transaction to the input
  // descriptors, so retain those bytes rather than accepting DBT's general
  // cross-target descriptor normalization as a second owner of the ABI.
  for (const AmdGpuKernelInfo &source_kernel : source.kernels()) {
    const auto output_kernel =
        std::ranges::find(output.kernels(), source_kernel.name, &AmdGpuKernelInfo::name);
    if (output_kernel == output.kernels().end()) {
      errors.emplace_back(error_prefix + " lost a kernel while relocating executable text");
      return std::nullopt;
    }
    auto source_descriptor =
        read_kernel_descriptor(descriptor_patched_image, source_kernel.descriptor_file_offset);
    const auto output_descriptor = read_kernel_descriptor(
        std::span<const uint8_t>(translated.elf_bytes), output_kernel->descriptor_file_offset);
    if (!source_descriptor || !output_descriptor) {
      errors.emplace_back(error_prefix + " could not retain a relocated kernel descriptor");
      return std::nullopt;
    }
    source_descriptor->kernel_code_entry_byte_offset =
        output_descriptor->kernel_code_entry_byte_offset;
    if (!write_kernel_descriptor(std::span<uint8_t>(translated.elf_bytes),
                                 output_kernel->descriptor_file_offset, *source_descriptor)) {
      errors.emplace_back(error_prefix + " could not publish a relocated kernel descriptor");
      return std::nullopt;
    }
  }
  return ConSanRelocatedText{
      .image = std::move(translated.elf_bytes),
      .placements = std::move(translated.text_placements),
      .text_size = output.text_sections().front()->size(),
      .relocation = ConSanTextRelocationProof{source_text_size},
  };
}

} // namespace

bool finalize_consan_text_rewrites(std::span<const uint8_t> descriptor_image, rj_code_arch_t arch,
                                   const ConSanOptions &options, std::string_view subject,
                                   ConSanTransformArtifacts &result) {
  if (result.staged_text_rewrites.empty())
    return true;

  auto relocated = relocate_consan_text(descriptor_image, arch, options, subject, result);
  if (!relocated)
    return false;
  std::vector<ConSanPatchInfo> placed_patches;
  std::vector<ConSanCommittedLowering> commits;
  const bool in_place = !relocated->relocation.has_value();
  commits.reserve(result.staged_text_rewrites.size());
  for (const ConSanStagedTextRewrite &staged : result.staged_text_rewrites) {
    std::optional<ConSanPatchInfo> primary;
    for (const TranslatedTextPlacement &placement : relocated->placements) {
      if (placement.source_offset != staged.patch.anchor_offset || !placement.client_rewrite)
        continue;
      ConSanPatchInfo placed = staged.patch;
      placed.trampoline_offset = placement.target_offset;
      if (in_place) {
        placed.original_size = static_cast<uint32_t>(staged.words.size() * sizeof(uint32_t));
        placed.trampoline_size = 0u;
      }
      const uint64_t emitted_offset = in_place ? placed.anchor_offset : placed.trampoline_offset;
      const uint64_t emitted_size = in_place ? placed.original_size : placed.trampoline_size;
      if (emitted_offset > relocated->text_size ||
          emitted_size > relocated->text_size - emitted_offset) {
        result.errors.emplace_back(
            "ConSan " + std::string(subject) + " placement exceeds relocated text: source=" +
            std::to_string(staged.patch.anchor_offset) +
            " target=" + std::to_string(emitted_offset) + " size=" + std::to_string(emitted_size) +
            " text=" + std::to_string(relocated->text_size));
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
    auto commit = make_consan_instrumented_patch_lowering(result.observation_plan(),
                                                          staged.intent_ids, *primary);
    if (!commit) {
      result.errors.emplace_back("ConSan " + std::string(subject) +
                                 " produced an invalid intent-bound lowering");
      result.discard_candidate_modification();
      return false;
    }
    commits.push_back(std::move(*commit));
  }

  if (relocated->relocation)
    result.text_relocation = relocated->relocation;
  result.staged_text_rewrites.clear();
  return result.publish_access_lowering(std::move(relocated->image), subject, std::move(commits),
                                        std::move(placed_patches));
}

} // namespace rocjitsu
