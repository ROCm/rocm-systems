// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file rj_dbt_translate_smoke_test.cpp
/// @brief End-to-end smoke test for the rj_dbt_translate command-line tool.

#include "dbt_translate.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/rj_code.h"
#include "scoped_temp.h"

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
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

namespace {

std::filesystem::path g_translate_tool;

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

std::vector<uint8_t> make_smoke_code_object(uint32_t elf_mach,
                                            std::span<const uint32_t> text_words) {
  using namespace rocjitsu;

  constexpr uint64_t text_offset = 0x100;
  constexpr uint64_t text_vaddr = 0x1100;
  const uint64_t text_size = text_words.size_bytes();
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
  write_value<uint32_t>(image, offsetof(Elf64_Ehdr, e_flags), elf_mach);
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

std::vector<uint8_t> make_smoke_code_object() {
  // Use a CDNA4 waitcnt that lowers to split RDNA4 wait instructions so the
  // diff report is guaranteed to contain a shown source/target pair.
  constexpr std::array<uint32_t, 2> text_words = {0xbf8cc07fu, 0xbf810000u};
  return make_smoke_code_object(rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX950, text_words);
}

std::vector<uint8_t> make_gfx1250_smoke_code_object(std::span<const uint32_t> text_words) {
  return make_smoke_code_object(rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX1250, text_words);
}

std::vector<uint8_t> make_gfx1250_smoke_code_object() {
  // B0-to-A0 replaces s_clause with s_nop, giving the first pass real work while
  // leaving the translated instruction stable on the verification pass.
  constexpr std::array<uint32_t, 2> text_words = {0xbf850004u, 0xbfb00000u};
  return make_gfx1250_smoke_code_object(text_words);
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

bool command_exited_with(int status, int exit_code) {
  return status != -1 && WIFEXITED(status) && WEXITSTATUS(status) == exit_code;
}

} // namespace

TEST(RjDbtTranslateIdempotence, DescribesChangedByteAndSize) {
  constexpr std::array<uint8_t, 2> first = {0x10, 0x20};
  constexpr std::array<uint8_t, 3> second = {0x10, 0x21, 0x22};

  EXPECT_EQ(rocjitsu::tools::detail::describe_byte_difference(first, second, "section '.text'"),
            "section '.text' first differs at 0x1 (first=0x20, second=0x21); size 2 -> 3 bytes");
}

TEST(RjDbtTranslateIdempotence, DescribesPureSizeChange) {
  constexpr std::array<uint8_t, 2> first = {0x10, 0x20};
  constexpr std::array<uint8_t, 3> second = {0x10, 0x20, 0x30};

  EXPECT_EQ(rocjitsu::tools::detail::describe_byte_difference(first, second, "ELF image"),
            "ELF image size changed from 2 to 3 bytes");
}

TEST(RjDbtTranslateIdempotence, LocalizesExecutableSectionDifference) {
  using rocjitsu::tools::detail::ExecutableSectionBytes;

  constexpr std::array<uint8_t, 2> first_text = {0x10, 0x20};
  constexpr std::array<uint8_t, 2> second_text = {0x10, 0x21};
  constexpr std::array<uint8_t, 2> first_elf = {0x01, 0x02};
  constexpr std::array<uint8_t, 2> second_elf = {0x01, 0x03};
  const std::array first_sections = {ExecutableSectionBytes{".text", first_text}};
  const std::array second_sections = {ExecutableSectionBytes{".text", second_text}};

  EXPECT_EQ(rocjitsu::tools::detail::find_idempotence_difference(first_sections, second_sections,
                                                                 first_elf, second_elf),
            "section '.text' first differs at 0x1 (first=0x20, second=0x21)");
}

TEST(RjDbtTranslateIdempotence, FallsBackToWholeElfDifference) {
  using rocjitsu::tools::detail::ExecutableSectionBytes;

  constexpr std::array<uint8_t, 2> text = {0x10, 0x20};
  constexpr std::array<uint8_t, 2> first_elf = {0x01, 0x02};
  constexpr std::array<uint8_t, 2> second_elf = {0x01, 0x03};
  const std::array first_sections = {ExecutableSectionBytes{".text", text}};
  const std::array second_sections = {ExecutableSectionBytes{".text", text}};

  EXPECT_EQ(rocjitsu::tools::detail::find_idempotence_difference(first_sections, second_sections,
                                                                 first_elf, second_elf),
            "ELF image first differs at 0x1 (first=0x02, second=0x03)");
}

TEST(RjDbtTranslateIdempotence, ReportsExecutableSectionSetAndOrderChanges) {
  using rocjitsu::tools::detail::ExecutableSectionBytes;

  constexpr std::array<uint8_t, 1> bytes = {0x10};
  constexpr std::array<uint8_t, 1> first_elf = {0x01};
  constexpr std::array<uint8_t, 1> second_elf = {0x02};
  const std::array first_sections = {
      ExecutableSectionBytes{".text", bytes},
      ExecutableSectionBytes{".init", bytes},
  };
  const std::array renamed_sections = {
      ExecutableSectionBytes{".text", bytes},
      ExecutableSectionBytes{".fini", bytes},
  };
  const std::array reordered_sections = {
      ExecutableSectionBytes{".init", bytes},
      ExecutableSectionBytes{".text", bytes},
  };
  const std::array added_sections = {
      ExecutableSectionBytes{".text", bytes},
      ExecutableSectionBytes{".init", bytes},
      ExecutableSectionBytes{".fini", bytes},
  };

  EXPECT_EQ(rocjitsu::tools::detail::find_idempotence_difference(first_sections, renamed_sections,
                                                                 first_elf, second_elf),
            "executable sections changed: removed '.init', added '.fini'");
  EXPECT_EQ(rocjitsu::tools::detail::find_idempotence_difference(first_sections, reordered_sections,
                                                                 first_elf, second_elf),
            "executable sections were reordered at index 0: first='.text', second='.init'");
  EXPECT_EQ(rocjitsu::tools::detail::find_idempotence_difference(first_sections, added_sections,
                                                                 first_elf, second_elf),
            "executable section '.fini' was added");
}

TEST(RjDbtTranslateIdempotence, VerificationDiagnosticsDoNotInvalidateFirstPassOutput) {
  rocjitsu::tools::TranslateOutput output;
  rocjitsu::TranslationDiagnostic diagnostic;
  diagnostic.severity = rocjitsu::DiagnosticSeverity::Error;
  output.idempotence_diagnostics.push_back(std::move(diagnostic));

  EXPECT_TRUE(output.ok());
  EXPECT_TRUE(output.dispatchable());
}

TEST(RjDbtTranslate, Smoke) {
  const rocjitsu::test::ScopedTempDirectory temp_dir("rj_dbt_translate_smoke_");
  const std::filesystem::path temp_path(temp_dir.path());

  const auto input = temp_path / "smoke_gfx950.co";
  const auto output = temp_path / "stdout.txt";
  const auto error = temp_path / "stderr.txt";

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

TEST(RjDbtTranslate, RequiresRevisionsOnlyForGfx1250) {
  const rocjitsu::test::ScopedTempDirectory temp_dir("rj_dbt_translate_revision_");
  const std::filesystem::path temp_path(temp_dir.path());
  const auto input = temp_path / "smoke.co";
  const auto output = temp_path / "stdout.txt";
  const auto error = temp_path / "stderr.txt";

  {
    const auto image = make_smoke_code_object();
    std::ofstream out(input, std::ios::binary);
    out.write(reinterpret_cast<const char *>(image.data()),
              static_cast<std::streamsize>(image.size()));
  }

  const std::string missing_input_revision_command =
      shell_quote(g_translate_tool.string()) + " " + shell_quote(input.string()) +
      " --input-target gfx1250 --output-target gfx1250 > " + shell_quote(output.string()) + " 2> " +
      shell_quote(error.string());
  int status = std::system(missing_input_revision_command.c_str());
  EXPECT_TRUE(command_exited_with(status, 1));
  EXPECT_TRUE(contains(read_text_file(error),
                       "--input-revision is required when --input-target is gfx1250"));

  const std::string missing_output_revision_command =
      shell_quote(g_translate_tool.string()) + " " + shell_quote(input.string()) +
      " --input-target gfx1250 --input-revision b0 --output-target gfx1250 > " +
      shell_quote(output.string()) + " 2> " + shell_quote(error.string());
  status = std::system(missing_output_revision_command.c_str());
  EXPECT_TRUE(command_exited_with(status, 1));
  EXPECT_TRUE(contains(read_text_file(error),
                       "--output-revision is required when --output-target is gfx1250"));

  const std::string unsupported_input_revision_command =
      shell_quote(g_translate_tool.string()) + " " + shell_quote(input.string()) +
      " --input-target gfx950 --input-revision b0 --output-target gfx1200 > " +
      shell_quote(output.string()) + " 2> " + shell_quote(error.string());
  status = std::system(unsupported_input_revision_command.c_str());
  EXPECT_TRUE(command_exited_with(status, 1));
  EXPECT_TRUE(contains(read_text_file(error),
                       "--input-revision is only valid when --input-target is gfx1250"));

  const std::string unsupported_output_revision_command =
      shell_quote(g_translate_tool.string()) + " " + shell_quote(input.string()) +
      " --input-target gfx950 --output-target gfx1200 --output-revision a0 > " +
      shell_quote(output.string()) + " 2> " + shell_quote(error.string());
  status = std::system(unsupported_output_revision_command.c_str());
  EXPECT_TRUE(command_exited_with(status, 1));
  EXPECT_TRUE(contains(read_text_file(error),
                       "--output-revision is only valid when --output-target is gfx1250"));

  const std::string reverse_revision_command =
      shell_quote(g_translate_tool.string()) + " " + shell_quote(input.string()) +
      " --input-target gfx1250 --input-revision a0 --output-target gfx1250 "
      "--output-revision b0 > " +
      shell_quote(output.string()) + " 2> " + shell_quote(error.string());
  status = std::system(reverse_revision_command.c_str());
  EXPECT_TRUE(command_exited_with(status, 1));
  EXPECT_TRUE(contains(read_text_file(error), "gfx1250 A0-to-B0 translation is not supported"));
}

TEST(RjDbtTranslate, VerifiesGfx1250B0ToA0Idempotence) {
  const rocjitsu::test::ScopedTempDirectory temp_dir("rj_dbt_translate_idempotence_");
  const std::filesystem::path temp_path(temp_dir.path());
  const std::filesystem::path input = temp_path / "smoke_gfx1250.co";
  const std::filesystem::path output = temp_path / "stdout.txt";
  const std::filesystem::path error = temp_path / "stderr.txt";

  {
    const std::vector<uint8_t> image = make_gfx1250_smoke_code_object();
    std::ofstream out(input, std::ios::binary);
    out.write(reinterpret_cast<const char *>(image.data()),
              static_cast<std::streamsize>(image.size()));
  }

  const std::string command = shell_quote(g_translate_tool.string()) + " " +
                              shell_quote(input.string()) +
                              " --input-target gfx1250 --input-revision b0 --output-target gfx1250 "
                              "--output-revision a0 --verify-idempotence --output-mode diff > " +
                              shell_quote(output.string()) + " 2> " + shell_quote(error.string());

  const int status = std::system(command.c_str());
  const std::string stdout_text = read_text_file(output);
  const std::string stderr_text = read_text_file(error);

  ASSERT_TRUE(command_succeeded(status)) << "stderr:\n"
                                         << stderr_text << "\nstdout:\n"
                                         << stdout_text;
  EXPECT_TRUE(stderr_text.empty()) << stderr_text;
  EXPECT_TRUE(contains(stdout_text, "idempotence: verified")) << stdout_text;
  EXPECT_TRUE(contains(stdout_text, "idempotence_diagnostics: 0")) << stdout_text;
  EXPECT_TRUE(contains(stdout_text, "changed=1")) << stdout_text;
  EXPECT_TRUE(contains(stdout_text, "source: s_clause 4")) << stdout_text;
  EXPECT_TRUE(contains(stdout_text, "target: s_nop 0")) << stdout_text;
}

TEST(RjDbtTranslate, VerifiesNonGfx1250SameArchitectureTranslation) {
  const rocjitsu::test::ScopedTempDirectory temp_dir("rj_dbt_translate_non_gfx_idempotence_");
  const std::filesystem::path temp_path(temp_dir.path());
  const std::filesystem::path input = temp_path / "smoke_gfx950.co";
  const std::filesystem::path output = temp_path / "stdout.txt";
  const std::filesystem::path error = temp_path / "stderr.txt";

  {
    const std::vector<uint8_t> image = make_smoke_code_object();
    std::ofstream out(input, std::ios::binary);
    out.write(reinterpret_cast<const char *>(image.data()),
              static_cast<std::streamsize>(image.size()));
  }

  const std::string command =
      shell_quote(g_translate_tool.string()) + " " + shell_quote(input.string()) +
      " --input-target gfx950 --output-target gfx950 --verify-idempotence --output-mode diff > " +
      shell_quote(output.string()) + " 2> " + shell_quote(error.string());

  const int status = std::system(command.c_str());
  const std::string stdout_text = read_text_file(output);
  const std::string stderr_text = read_text_file(error);

  ASSERT_TRUE(command_succeeded(status)) << "stderr:\n"
                                         << stderr_text << "\nstdout:\n"
                                         << stdout_text;
  EXPECT_TRUE(stderr_text.empty()) << stderr_text;
  EXPECT_TRUE(contains(stdout_text, "idempotence: verified")) << stdout_text;
}

TEST(RjDbtTranslate, RejectsInvalidIdempotenceOptionCombinations) {
  const rocjitsu::test::ScopedTempDirectory temp_dir("rj_dbt_translate_idempotence_options_");
  const std::filesystem::path temp_path(temp_dir.path());
  const std::filesystem::path input = temp_path / "smoke_gfx950.co";
  const std::filesystem::path output = temp_path / "stdout.txt";
  const std::filesystem::path error = temp_path / "stderr.txt";

  {
    const std::vector<uint8_t> image = make_smoke_code_object();
    std::ofstream out(input, std::ios::binary);
    out.write(reinterpret_cast<const char *>(image.data()),
              static_cast<std::streamsize>(image.size()));
  }

  const std::string different_arch_command =
      shell_quote(g_translate_tool.string()) + " " + shell_quote(input.string()) +
      " --input-target gfx950 --output-target gfx1200 --verify-idempotence > " +
      shell_quote(output.string()) + " 2> " + shell_quote(error.string());
  int status = std::system(different_arch_command.c_str());
  EXPECT_TRUE(command_exited_with(status, 1));
  EXPECT_TRUE(contains(read_text_file(error),
                       "--verify-idempotence requires matching input and output architectures"));

  const std::string skip_failed_command =
      shell_quote(g_translate_tool.string()) + " " + shell_quote(input.string()) +
      " --input-target gfx950 --output-target gfx950 --verify-idempotence "
      "--skip-failed-kernels > " +
      shell_quote(output.string()) + " 2> " + shell_quote(error.string());
  status = std::system(skip_failed_command.c_str());
  EXPECT_TRUE(command_exited_with(status, 1));
  EXPECT_TRUE(contains(read_text_file(error),
                       "--verify-idempotence cannot be combined with --skip-failed-kernels"));

  const std::string list_command =
      shell_quote(g_translate_tool.string()) + " " + shell_quote(input.string()) +
      " --list-code-objects --verify-idempotence > " + shell_quote(output.string()) + " 2> " +
      shell_quote(error.string());
  status = std::system(list_command.c_str());
  EXPECT_TRUE(command_exited_with(status, 1));
  EXPECT_TRUE(contains(read_text_file(error),
                       "--verify-idempotence cannot be combined with --list-code-objects"));
}

TEST(RjDbtTranslate, CharacterizesGfx1250ClusterLoadRewrapAsNonIdempotent) {
  const rocjitsu::test::ScopedTempDirectory temp_dir("rj_dbt_translate_non_idempotent_");
  const std::filesystem::path temp_path(temp_dir.path());
  const std::filesystem::path input = temp_path / "cluster_load_gfx1250.co";
  const std::filesystem::path output = temp_path / "stdout.txt";
  const std::filesystem::path error = temp_path / "stderr.txt";

  // TODO(PR #9272 follow-up): Remove this characterization once cluster-load
  // expansion recognizes its own M0 save/restore wrapper. The permanent
  // mismatch-reporting coverage lives in the RjDbtTranslateIdempotence tests.
  // The cluster load is followed by s_endpgm. Its expansion preserves the load
  // inside an M0 save/restore wrapper, so each pass wraps it again.
  constexpr std::array<uint32_t, 4> text_words = {0xee19c07cu, 0x00000001u, 0x00000002u,
                                                  0xbfb00000u};
  {
    const std::vector<uint8_t> image = make_gfx1250_smoke_code_object(text_words);
    std::ofstream out(input, std::ios::binary);
    out.write(reinterpret_cast<const char *>(image.data()),
              static_cast<std::streamsize>(image.size()));
  }

  const std::string command = shell_quote(g_translate_tool.string()) + " " +
                              shell_quote(input.string()) +
                              " --input-target gfx1250 --input-revision b0 --output-target gfx1250 "
                              "--output-revision a0 --verify-idempotence --output-mode diff > " +
                              shell_quote(output.string()) + " 2> " + shell_quote(error.string());

  const int status = std::system(command.c_str());
  const std::string stdout_text = read_text_file(output);
  const std::string stderr_text = read_text_file(error);

  EXPECT_TRUE(command_exited_with(status, 5)) << "stderr:\n"
                                              << stderr_text << "\nstdout:\n"
                                              << stdout_text;
  EXPECT_TRUE(contains(stdout_text, "idempotence: not-verified")) << stdout_text;
  EXPECT_TRUE(contains(stdout_text, "idempotence_diagnostics: 0")) << stdout_text;
  EXPECT_TRUE(contains(stderr_text, "translation output is not byte-idempotent")) << stderr_text;
  EXPECT_TRUE(contains(stderr_text, "section '.text'")) << stderr_text;
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
