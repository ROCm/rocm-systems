// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_descriptor_growth.h"

#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/patch/consan/consan_descriptor.h"
#include "rocjitsu/isa/register_set.h"

#include <unordered_map>

namespace rocjitsu {
[[nodiscard]] std::unordered_map<uint64_t, uint16_t>
merge_consan_descriptor_register_growths(std::span<const ConSanDescriptorRegisterGrowth> growths) {
  std::unordered_map<uint64_t, uint16_t> merged;
  for (const ConSanDescriptorRegisterGrowth &growth : growths)
    note_maximum_descriptor_extent(merged, growth.descriptor_file_offset, growth.required_count);
  return merged;
}

bool apply_consan_descriptor_sgpr_growths_to_patcher(
    CodeObjectPatcher &patcher, std::span<const ConSanDescriptorRegisterGrowth> growths,
    rj_code_arch_t arch, std::vector<std::string> &errors) {
  const std::span<const uint8_t> current_image = patcher.image_bytes();
  for (const auto &[descriptor_file_offset, required_count] :
       merge_consan_descriptor_register_growths(growths)) {
    auto descriptor = read_kernel_descriptor(current_image, descriptor_file_offset);
    if (!descriptor) {
      errors.emplace_back("ConSan relay-reservoir descriptor SGPR growth exceeds ELF bytes");
      return false;
    }
    if (!grow_descriptor_sgpr_allocation(*descriptor, required_count, arch) ||
        !patcher.patch_kernel_descriptor(descriptor_file_offset, *descriptor)) {
      errors.emplace_back("ConSan relay reservoir could not grow descriptor SGPR allocation");
      return false;
    }
  }
  return true;
}

bool apply_consan_descriptor_sgpr_growths_to_bytes(
    std::span<uint8_t> image, std::span<const ConSanDescriptorRegisterGrowth> growths,
    rj_code_arch_t arch, std::vector<std::string> &errors) {
  for (const auto &[descriptor_file_offset, required_count] :
       merge_consan_descriptor_register_growths(growths)) {
    auto descriptor = read_kernel_descriptor(image, descriptor_file_offset);
    if (!descriptor) {
      errors.emplace_back("ConSan relay-reservoir descriptor SGPR growth exceeds ELF bytes");
      return false;
    }
    if (!grow_descriptor_sgpr_allocation(*descriptor, required_count, arch)) {
      errors.emplace_back("ConSan relay reservoir could not grow descriptor SGPR allocation");
      return false;
    }
    if (!write_kernel_descriptor(image, descriptor_file_offset, *descriptor))
      return false;
  }
  return true;
}

bool apply_consan_descriptor_vgpr_growths_to_bytes(
    std::span<uint8_t> image, std::span<const ConSanDescriptorRegisterGrowth> growths,
    rj_code_arch_t arch, std::vector<std::string> &errors) {
  for (const auto &[descriptor_file_offset, required_count] :
       merge_consan_descriptor_register_growths(growths)) {
    auto descriptor = read_kernel_descriptor(image, descriptor_file_offset);
    if (!descriptor) {
      errors.emplace_back("ConSan LDS check/trap proof descriptor VGPR growth exceeds ELF bytes");
      return false;
    }
    if (!grow_descriptor_vgpr_allocation(*descriptor,
                                         {.required_ordinary_count = required_count,
                                          .maximum_ordinary_count = REGISTER_SET_MAX_VGPRS},
                                         arch)) {
      errors.emplace_back("ConSan LDS check/trap proof could not grow descriptor VGPR allocation");
      return false;
    }
    if (!write_kernel_descriptor(image, descriptor_file_offset, *descriptor))
      return false;
  }
  return true;
}

bool apply_consan_descriptor_vgpr_growths_to_patcher(
    CodeObjectPatcher &patcher, std::span<const uint8_t> original_image,
    std::span<const ConSanDescriptorRegisterGrowth> growths, rj_code_arch_t arch,
    std::vector<std::string> &errors) {
  for (const auto &[descriptor_file_offset, required_count] :
       merge_consan_descriptor_register_growths(growths)) {
    auto descriptor = read_kernel_descriptor(original_image, descriptor_file_offset);
    if (!descriptor) {
      errors.emplace_back("ConSan LDS check/trap proof descriptor VGPR growth exceeds ELF bytes");
      return false;
    }
    if (!grow_descriptor_vgpr_allocation(*descriptor,
                                         {.required_ordinary_count = required_count,
                                          .maximum_ordinary_count = REGISTER_SET_MAX_VGPRS},
                                         arch)) {
      errors.emplace_back("ConSan LDS check/trap proof could not grow descriptor VGPR allocation");
      return false;
    }
    if (!patcher.patch_kernel_descriptor(descriptor_file_offset, *descriptor)) {
      errors.emplace_back("ConSan LDS check/trap proof could not patch descriptor VGPR allocation");
      return false;
    }
  }
  return true;
}

} // namespace rocjitsu
