// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_input_layout.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/kernel_descriptor_scan.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/AMDHSAKernelDescriptor.h"
RJ_DIAGNOSTIC_POP

#include <algorithm>
#include <cstdint>
#include <optional>
#include <span>

namespace rocjitsu {
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

} // namespace

std::vector<std::string> validate_consan_input_layout(const AmdGpuCodeObject &code_object,
                                                      bool allow_descriptor_entry_redirect) {
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
      const auto descriptor = read_kernel_descriptor(
          std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(code_object.image_data()),
                                   code_object.image_size()),
          kernel.descriptor_file_offset);
      if (!descriptor) {
        errors.emplace_back("ConSan kernel '" + kernel.name + "' descriptor exceeds ELF bytes");
        continue;
      }
      const uint64_t descriptor_address = code_object.kernel_descriptor_offset(kernel.name);
      const std::optional<uint64_t> descriptor_entry =
          add_signed_offset(descriptor_address, descriptor->kernel_code_entry_byte_offset);
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

} // namespace rocjitsu
