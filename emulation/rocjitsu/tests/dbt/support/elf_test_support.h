// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/// @file elf_test_support.h
/// @brief Small, bounds-checked helpers shared by DBT ELF integration tests.

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace rocjitsu::test_support {

/// @brief Read one POD value at a file offset in an ELF fixture.
template <typename T>
[[nodiscard]] T read_elf_struct_for_test(std::span<const uint8_t> image, uint64_t offset) {
  T value{};
  assert(offset <= image.size());
  assert(sizeof(T) <= image.size() - offset);
  std::memcpy(&value, image.data() + offset, sizeof(value));
  return value;
}

/// @brief Read an array of POD values at a file offset in an ELF fixture.
template <typename T>
[[nodiscard]] std::vector<T> read_elf_array_for_test(std::span<const uint8_t> image,
                                                     uint64_t offset, size_t count) {
  std::vector<T> values(count);
  assert(offset <= image.size());
  assert(count <= (image.size() - offset) / sizeof(T));
  std::memcpy(values.data(), image.data() + offset, count * sizeof(T));
  return values;
}

/// @brief Write one POD value at a file offset in an ELF fixture.
template <typename T>
void write_elf_struct_for_test(std::span<uint8_t> image, uint64_t offset, const T &value) {
  assert(offset <= image.size());
  assert(sizeof(T) <= image.size() - offset);
  std::memcpy(image.data() + offset, &value, sizeof(value));
}

/// @brief Write one scalar value at a file offset in an ELF fixture.
template <typename T>
void write_value_for_test(std::span<uint8_t> image, uint64_t offset, T value) {
  write_elf_struct_for_test(image, offset, value);
}

/// @brief Find a named symbol in any valid symbol table in an ELF fixture.
[[nodiscard]] inline std::optional<Elf64_Sym>
find_elf_symbol_for_test(std::span<const uint8_t> image, std::string_view name) {
  const auto range_is_in_image = [&](uint64_t offset, uint64_t size) {
    const uint64_t image_size = image.size();
    return offset <= image_size && size <= image_size - offset;
  };
  if (!range_is_in_image(0, sizeof(Elf64_Ehdr)))
    return std::nullopt;

  const auto header = read_elf_struct_for_test<Elf64_Ehdr>(image, 0);
  const uint64_t section_table_size = static_cast<uint64_t>(header.e_shnum) * sizeof(Elf64_Shdr);
  if (!range_is_in_image(header.e_shoff, section_table_size))
    return std::nullopt;

  const auto sections = read_elf_array_for_test<Elf64_Shdr>(image, header.e_shoff, header.e_shnum);
  for (const Elf64_Shdr &symtab : sections) {
    if ((symtab.sh_type != SHT_SYMTAB && symtab.sh_type != SHT_DYNSYM) ||
        symtab.sh_entsize != sizeof(Elf64_Sym) || symtab.sh_size % sizeof(Elf64_Sym) != 0 ||
        symtab.sh_link >= sections.size() || !range_is_in_image(symtab.sh_offset, symtab.sh_size)) {
      continue;
    }
    const Elf64_Shdr &strtab = sections[symtab.sh_link];
    if (strtab.sh_type != SHT_STRTAB || !range_is_in_image(strtab.sh_offset, strtab.sh_size))
      continue;
    const auto symbols = read_elf_array_for_test<Elf64_Sym>(image, symtab.sh_offset,
                                                            symtab.sh_size / sizeof(Elf64_Sym));
    for (const Elf64_Sym &symbol : symbols) {
      if (symbol.st_name >= strtab.sh_size)
        continue;
      const char *begin =
          reinterpret_cast<const char *>(image.data() + strtab.sh_offset + symbol.st_name);
      const size_t remaining = strtab.sh_size - symbol.st_name;
      const auto *end = static_cast<const char *>(std::memchr(begin, '\0', remaining));
      if (end != nullptr && std::string_view(begin, static_cast<size_t>(end - begin)) == name)
        return symbol;
    }
  }
  return std::nullopt;
}

/// @brief Find a named section in a parsed code object.
[[nodiscard]] inline const Section *find_section(const CodeObject &co, std::string_view name) {
  for (const auto &section : co.all_sections()) {
    if (section->name() == name)
      return section.get();
  }
  return nullptr;
}

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
