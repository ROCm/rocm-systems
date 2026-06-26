// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file rj_dbt_translate_smoke_test.cpp
/// @brief End-to-end smoke test for the rj_dbt_translate command-line tool.

#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/rj_code.h"

#include <gtest/gtest.h>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

namespace {

std::filesystem::path g_translate_tool;

struct TempDir {
  std::filesystem::path path;

  explicit TempDir(std::filesystem::path temp_path) : path(std::move(temp_path)) {
    std::filesystem::create_directories(path);
  }

  ~TempDir() { std::filesystem::remove_all(path); }
};

uint32_t add_elf_name(std::vector<uint8_t> &names, std::string_view name) {
  const uint32_t offset = static_cast<uint32_t>(names.size());
  names.insert(names.end(), name.begin(), name.end());
  names.push_back('\0');
  return offset;
}

uint64_t align_up(uint64_t value, uint64_t alignment) {
  const uint64_t remainder = value % alignment;
  return remainder == 0 ? value : value + alignment - remainder;
}

void write_bytes(std::vector<uint8_t> &image, uint64_t offset, const void *src, size_t size) {
  assert(offset <= image.size());
  assert(size <= image.size() - offset);
  std::memcpy(image.data() + offset, src, size);
}

template <typename T> void write_value(std::vector<uint8_t> &image, uint64_t offset, T value) {
  write_bytes(image, offset, &value, sizeof(value));
}

std::vector<uint8_t> make_smoke_code_object() {
  using namespace rocjitsu;

  constexpr uint64_t text_offset = 0x100;
  constexpr uint64_t text_vaddr = 0x1100;
  constexpr uint64_t text_size = 8;
  constexpr uint64_t load_align = 0x1000;
  constexpr uint64_t kernel_descriptor_size = 64;

  std::vector<uint8_t> shstrtab{'\0'};
  const uint32_t text_name = add_elf_name(shstrtab, ".text");
  const uint32_t rodata_name = add_elf_name(shstrtab, ".rodata");
  const uint32_t symtab_name = add_elf_name(shstrtab, ".symtab");
  const uint32_t strtab_name = add_elf_name(shstrtab, ".strtab");
  const uint32_t shstrtab_name = add_elf_name(shstrtab, ".shstrtab");

  std::vector<uint8_t> strtab{'\0'};
  const uint32_t kd_symbol_name = add_elf_name(strtab, "smoke.kd");

  const uint64_t rodata_offset = text_offset + text_size;
  const uint64_t rodata_vaddr = text_vaddr + text_size + load_align;
  const uint64_t strtab_offset = rodata_offset + kernel_descriptor_size;
  const uint64_t symtab_offset = align_up(strtab_offset + strtab.size(), 8);
  constexpr size_t sym_count = 2;
  const uint64_t shstrtab_offset = symtab_offset + sym_count * sizeof(Elf64_Sym);
  const uint64_t shoff = align_up(shstrtab_offset + shstrtab.size(), 8);
  constexpr uint16_t section_count = 6;
  constexpr uint16_t phdr_count = 2;

  std::vector<uint8_t> image(shoff + section_count * sizeof(Elf64_Shdr), 0);

  write_bytes(image, offsetof(Elf64_Ehdr, e_ident), EI_MAGIC, EI_MAGIC_SIZE);
  image[offsetof(Elf64_Ehdr, e_ident) + EI_CLASS] = ELFCLASS64;
  image[offsetof(Elf64_Ehdr, e_ident) + EI_DATA] = 1;
  image[offsetof(Elf64_Ehdr, e_ident) + EI_VERSION] = 1;
  image[offsetof(Elf64_Ehdr, e_ident) + EI_OSABI] = ELFOSABI_AMDGPU_HSA;
  image[offsetof(Elf64_Ehdr, e_ident) + EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
  write_value<uint16_t>(image, offsetof(Elf64_Ehdr, e_type), ET_DYN);
  write_value<uint16_t>(image, offsetof(Elf64_Ehdr, e_machine), EM_AMDGPU);
  write_value<uint32_t>(image, offsetof(Elf64_Ehdr, e_version), 1);
  write_value<uint64_t>(image, offsetof(Elf64_Ehdr, e_phoff), sizeof(Elf64_Ehdr));
  write_value<uint64_t>(image, offsetof(Elf64_Ehdr, e_shoff), shoff);
  write_value<uint32_t>(image, offsetof(Elf64_Ehdr, e_flags), EF_AMDGPU_MACH_AMDGCN_GFX950);
  write_value<uint16_t>(image, offsetof(Elf64_Ehdr, e_ehsize), sizeof(Elf64_Ehdr));
  write_value<uint16_t>(image, offsetof(Elf64_Ehdr, e_phentsize), sizeof(Elf64_Phdr));
  write_value<uint16_t>(image, offsetof(Elf64_Ehdr, e_phnum), phdr_count);
  write_value<uint16_t>(image, offsetof(Elf64_Ehdr, e_shentsize), sizeof(Elf64_Shdr));
  write_value<uint16_t>(image, offsetof(Elf64_Ehdr, e_shnum), section_count);
  write_value<uint16_t>(image, offsetof(Elf64_Ehdr, e_shstrndx), 5);

  const uint64_t phdr0 = sizeof(Elf64_Ehdr);
  write_value<uint32_t>(image, phdr0 + offsetof(Elf64_Phdr, p_type), PT_LOAD);
  write_value<uint32_t>(image, phdr0 + offsetof(Elf64_Phdr, p_flags), 0x5); // PF_R | PF_X
  write_value<uint64_t>(image, phdr0 + offsetof(Elf64_Phdr, p_offset), text_offset);
  write_value<uint64_t>(image, phdr0 + offsetof(Elf64_Phdr, p_vaddr), text_vaddr);
  write_value<uint64_t>(image, phdr0 + offsetof(Elf64_Phdr, p_paddr), text_vaddr);
  write_value<uint64_t>(image, phdr0 + offsetof(Elf64_Phdr, p_filesz), text_size);
  write_value<uint64_t>(image, phdr0 + offsetof(Elf64_Phdr, p_memsz), text_size);
  write_value<uint64_t>(image, phdr0 + offsetof(Elf64_Phdr, p_align), load_align);

  const uint64_t phdr1 = phdr0 + sizeof(Elf64_Phdr);
  write_value<uint32_t>(image, phdr1 + offsetof(Elf64_Phdr, p_type), PT_LOAD);
  write_value<uint32_t>(image, phdr1 + offsetof(Elf64_Phdr, p_flags), 0x4); // PF_R
  write_value<uint64_t>(image, phdr1 + offsetof(Elf64_Phdr, p_offset), rodata_offset);
  write_value<uint64_t>(image, phdr1 + offsetof(Elf64_Phdr, p_vaddr), rodata_vaddr);
  write_value<uint64_t>(image, phdr1 + offsetof(Elf64_Phdr, p_paddr), rodata_vaddr);
  write_value<uint64_t>(image, phdr1 + offsetof(Elf64_Phdr, p_filesz), kernel_descriptor_size);
  write_value<uint64_t>(image, phdr1 + offsetof(Elf64_Phdr, p_memsz), kernel_descriptor_size);
  write_value<uint64_t>(image, phdr1 + offsetof(Elf64_Phdr, p_align), load_align);

  // Use a CDNA4 waitcnt that lowers to split RDNA4 wait instructions so the
  // diff report is guaranteed to contain a shown source/target pair.
  const std::array<uint32_t, 2> text_words = {0xbf8cc07fu, 0xbf810000u};
  std::memcpy(image.data() + text_offset, text_words.data(), text_size);

  // The DBT pipeline translates kernel entry offsets through AMDHSA kernel
  // descriptors. For this smoke fixture only the entry offset must be valid;
  // the remaining descriptor fields can stay zero.
  constexpr size_t kernel_code_entry_byte_offset_offset = 16;
  std::vector<uint8_t> kernel_descriptor(kernel_descriptor_size, 0);
  const int64_t entry_offset =
      static_cast<int64_t>(text_vaddr) - static_cast<int64_t>(rodata_vaddr);
  std::memcpy(kernel_descriptor.data() + kernel_code_entry_byte_offset_offset, &entry_offset,
              sizeof(entry_offset));
  std::memcpy(image.data() + rodata_offset, kernel_descriptor.data(), kernel_descriptor.size());
  std::memcpy(image.data() + strtab_offset, strtab.data(), strtab.size());

  const uint64_t sym1 = symtab_offset + sizeof(Elf64_Sym);
  write_value<uint32_t>(image, sym1 + offsetof(Elf64_Sym, st_name), kd_symbol_name);
  write_value<unsigned char>(image, sym1 + offsetof(Elf64_Sym, st_info),
                             elf_symbol_info(kElfSymbolBindGlobal, kElfSymbolTypeObject));
  write_value<uint16_t>(image, sym1 + offsetof(Elf64_Sym, st_shndx), 2);
  write_value<uint64_t>(image, sym1 + offsetof(Elf64_Sym, st_value), rodata_vaddr);
  write_value<uint64_t>(image, sym1 + offsetof(Elf64_Sym, st_size), kernel_descriptor.size());

  std::memcpy(image.data() + shstrtab_offset, shstrtab.data(), shstrtab.size());

  const auto write_shdr = [&](uint64_t index, uint32_t name, uint32_t type, uint64_t flags,
                              uint64_t addr, uint64_t offset, uint64_t size, uint32_t link,
                              uint32_t info, uint64_t addralign, uint64_t entsize) {
    const uint64_t base = shoff + index * sizeof(Elf64_Shdr);
    write_value<uint32_t>(image, base + offsetof(Elf64_Shdr, sh_name), name);
    write_value<uint32_t>(image, base + offsetof(Elf64_Shdr, sh_type), type);
    write_value<uint64_t>(image, base + offsetof(Elf64_Shdr, sh_flags), flags);
    write_value<uint64_t>(image, base + offsetof(Elf64_Shdr, sh_addr), addr);
    write_value<uint64_t>(image, base + offsetof(Elf64_Shdr, sh_offset), offset);
    write_value<uint64_t>(image, base + offsetof(Elf64_Shdr, sh_size), size);
    write_value<uint32_t>(image, base + offsetof(Elf64_Shdr, sh_link), link);
    write_value<uint32_t>(image, base + offsetof(Elf64_Shdr, sh_info), info);
    write_value<uint64_t>(image, base + offsetof(Elf64_Shdr, sh_addralign), addralign);
    write_value<uint64_t>(image, base + offsetof(Elf64_Shdr, sh_entsize), entsize);
  };
  write_shdr(1, text_name, SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR, text_vaddr, text_offset,
             text_size, 0, 0, sizeof(uint32_t), 0);
  write_shdr(2, rodata_name, SHT_PROGBITS, SHF_ALLOC, rodata_vaddr, rodata_offset,
             kernel_descriptor_size, 0, 0, 64, 0);
  write_shdr(3, symtab_name, SHT_SYMTAB, 0, 0, symtab_offset, sym_count * sizeof(Elf64_Sym), 4, 1,
             8, sizeof(Elf64_Sym));
  write_shdr(4, strtab_name, SHT_STRTAB, 0, 0, strtab_offset, strtab.size(), 0, 0, 1, 0);
  write_shdr(5, shstrtab_name, SHT_STRTAB, 0, 0, shstrtab_offset, shstrtab.size(), 0, 0, 1, 0);
  return image;
}

std::string shell_quote(std::string_view text) {
  std::string quoted = "'";
  for (const char ch : text) {
    if (ch == '\'')
      quoted += "'\\''";
    else
      quoted += ch;
  }
  quoted += "'";
  return quoted;
}

std::string read_text_file(const std::filesystem::path &path) {
  std::ifstream in(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

bool contains(std::string_view haystack, std::string_view needle) {
  return haystack.find(needle) != std::string_view::npos;
}

bool command_succeeded(int status) {
  return status != -1 && WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

} // namespace

TEST(RjDbtTranslate, Smoke) {
  const TempDir temp_dir(
      std::filesystem::temp_directory_path() /
      ("rj_dbt_translate_smoke_" + std::to_string(static_cast<long long>(getpid()))));

  const auto input = temp_dir.path / "smoke_gfx950.co";
  const auto output = temp_dir.path / "stdout.txt";
  const auto error = temp_dir.path / "stderr.txt";

  {
    const auto image = make_smoke_code_object();
    std::ofstream out(input, std::ios::binary);
    out.write(reinterpret_cast<const char *>(image.data()),
              static_cast<std::streamsize>(image.size()));
  }

  const std::string command =
      shell_quote(g_translate_tool.string()) + " " + shell_quote(input.string()) +
      " --input-target gfx950 --output-target gfx1200 --output-mode diff > " +
      shell_quote(output.string()) + " 2> " + shell_quote(error.string());

  const int status = std::system(command.c_str());
  const std::string stdout_text = read_text_file(output);
  const std::string stderr_text = read_text_file(error);

  ASSERT_TRUE(command_succeeded(status)) << "stderr:\n"
                                         << stderr_text << "\nstdout:\n"
                                         << stdout_text;
  EXPECT_TRUE(stderr_text.empty()) << stderr_text;

  const std::array<std::string_view, 5> expected = {
      "rj_dbt_translate: ok", "source_words: bf8cc07f", "source: s_waitcnt",
      "target_words:",        "target: s_wait",
  };
  for (const std::string_view needle : expected) {
    EXPECT_TRUE(contains(stdout_text, needle))
        << "missing expected output fragment: " << needle << "\noutput:\n"
        << stdout_text;
  }
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "usage: rj_dbt_translate_smoke_test <rj_dbt_translate>\n";
    return 2;
  }

  g_translate_tool = argv[1];
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
