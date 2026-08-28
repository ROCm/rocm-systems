// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file elf_test_support.h
/// @brief Small, bounds-checked helpers shared by DBT ELF integration tests.

#include "rocjitsu/code/amdgpu_elf.h"

#include <cstdint>
#include <cstring>
#include <optional>
#include <span>

namespace rocjitsu::test_support {

/// @brief Map a loaded ELF virtual address back to its file offset.
[[nodiscard]] inline std::optional<uint64_t>
loaded_vaddr_to_file_offset(std::span<const uint8_t> image, uint64_t vaddr) {
  if (image.size() < sizeof(Elf64_Ehdr))
    return std::nullopt;
  Elf64_Ehdr header{};
  std::memcpy(&header, image.data(), sizeof(header));
  if (header.e_phoff == 0 || header.e_phnum == 0 || header.e_phentsize != sizeof(Elf64_Phdr)) {
    return std::nullopt;
  }
  if (header.e_phoff > image.size() ||
      header.e_phnum > (image.size() - header.e_phoff) / sizeof(Elf64_Phdr)) {
    return std::nullopt;
  }

  for (uint16_t i = 0; i < header.e_phnum; ++i) {
    Elf64_Phdr program{};
    std::memcpy(&program,
                image.data() + header.e_phoff + static_cast<uint64_t>(i) * sizeof(program),
                sizeof(program));
    if (program.p_type != PT_LOAD || vaddr < program.p_vaddr ||
        vaddr - program.p_vaddr >= program.p_filesz) {
      continue;
    }
    const uint64_t file_offset = program.p_offset + (vaddr - program.p_vaddr);
    if (file_offset >= image.size())
      return std::nullopt;
    return file_offset;
  }
  return std::nullopt;
}

/// @brief Read one POD value at a loaded ELF virtual address.
template <typename T>
[[nodiscard]] std::optional<T> read_loaded_value(std::span<const uint8_t> image, uint64_t vaddr) {
  const auto offset = loaded_vaddr_to_file_offset(image, vaddr);
  if (!offset || *offset > image.size() || sizeof(T) > image.size() - *offset)
    return std::nullopt;
  T value{};
  std::memcpy(&value, image.data() + *offset, sizeof(value));
  return value;
}

} // namespace rocjitsu::test_support
