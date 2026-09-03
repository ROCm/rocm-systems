// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_text_relocation.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/kernel_descriptor_scan.h"
#include "rocjitsu/code/patch/consan/consan_growth_policy.h"

#include <algorithm>
#include <limits>
#include <string>

namespace rocjitsu {

std::optional<ConSanRelocatedText> relocate_consan_text(
    std::span<const uint8_t> descriptor_patched_image, rj_code_arch_t arch,
    std::span<const ConSanInlineTextRewrite> rewrites,
    const ConSanPatchedImageGrowthLimit &growth_limit, const ConSanCodeObjectId &input_id,
    std::string_view operation, std::vector<std::string> &errors,
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
  if (arch == ROCJITSU_CODE_ARCH_CDNA5) {
    // A same-revision choice selects structural identity translation. ConSan's
    // target-specific emitters already produced words for the loaded revision;
    // no B0-to-A0 stepping rewrite is requested here.
    options.input_revision = ProcessorRevision::Gfx1250A0;
    options.output_revision = ProcessorRevision::Gfx1250A0;
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

} // namespace rocjitsu
