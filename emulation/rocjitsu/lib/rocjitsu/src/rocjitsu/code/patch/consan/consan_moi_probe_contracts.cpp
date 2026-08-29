// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_moi_probe_contracts.h"

#include "rocjitsu/code/patch/consan/consan_descriptor.h"

namespace rocjitsu::consan_moi_impl {

std::optional<uint16_t> moi_descriptor_owner_shift(std::span<const uint8_t> image,
                                                   uint64_t descriptor_file_offset,
                                                   rj_code_arch_t arch,
                                                   std::vector<std::string> &errors) {
  const auto descriptor = read_kernel_descriptor(image, descriptor_file_offset);
  if (!descriptor) {
    errors.emplace_back("ConSan MOI owner derivation descriptor exceeds ELF bytes");
    return std::nullopt;
  }
  return kernel_wavefront_size(arch, *descriptor) == 32 ? 5 : 6;
}

} // namespace rocjitsu::consan_moi_impl
