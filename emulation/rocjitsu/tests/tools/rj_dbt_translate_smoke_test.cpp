// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file rj_dbt_translate_smoke_test.cpp
/// @brief End-to-end smoke test for the rj_dbt_translate command-line tool.

#include "dbt_translate.h"
#include "dbt_translate_cli.h"
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
#include <optional>
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
std::filesystem::path g_decode_benchmark;

rocjitsu::tools::ToolResult<rocjitsu::tools::TranslateOutput>
capture_gfx1251_targets(const rocjitsu::tools::TranslateOptions &options) {
  EXPECT_EQ(options.input_target, ROCJITSU_CODE_TARGET_GFX1251);
  EXPECT_EQ(options.guest_arch, ROCJITSU_CODE_ARCH_CDNA5);
  EXPECT_EQ(options.host_arch, ROCJITSU_CODE_ARCH_CDNA5);
  EXPECT_EQ(options.target_mach, rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX1251);
  EXPECT_EQ(options.input_revision, rocjitsu::ProcessorRevision::Unspecified);
  EXPECT_EQ(options.output_revision, rocjitsu::ProcessorRevision::Unspecified);
  return {};
}

rocjitsu::tools::ToolResult<rocjitsu::tools::TranslateOutput>
synthetic_idempotence_mismatch(const rocjitsu::tools::TranslateOptions &options) {
  EXPECT_TRUE(options.verify_idempotence);

  rocjitsu::tools::ToolResult<rocjitsu::tools::TranslateOutput> result;
  result.value.idempotence_checked = true;
  result.errors.push_back(
      {.exit_code = 5,
       .message =
           "translation output is not byte-idempotent: section '.text' first differs at 0x4"});
  return result;
}

rocjitsu::tools::ToolResult<rocjitsu::tools::TranslateOutput>
synthetic_residual_rewrite(const rocjitsu::tools::TranslateOptions &options) {
  EXPECT_TRUE(options.verify_rewrite_discharge);

  rocjitsu::tools::ToolResult<rocjitsu::tools::TranslateOutput> result;
  result.value.rewrite_discharge_checked = true;
  result.value.diagnostics.push_back(
      {.severity = rocjitsu::DiagnosticSeverity::Error,
       .kind = rocjitsu::DiagnosticKind::ResidualRewrite,
       .guest_offset = std::nullopt,
       .output_offset = 4,
       .mnemonic = "s_clause",
       .message = "registered rewrite remains actionable in final output",
       .required_work = {}});
  result.errors.push_back({.exit_code = 3, .message = "translation failed"});
  return result;
}

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

void clear_smoke_kernel_descriptor_symbol(std::vector<uint8_t> &image) {
  rocjitsu::Elf64_Ehdr header{};
  std::memcpy(&header, image.data(), sizeof(header));

  auto section_offset = [&](size_t index) {
    return header.e_shoff + index * sizeof(rocjitsu::Elf64_Shdr);
  };
  rocjitsu::Elf64_Shdr symtab{};
  std::memcpy(&symtab, image.data() + section_offset(3), sizeof(symtab));
  rocjitsu::Elf64_Sym descriptor{};
  std::memcpy(&descriptor, image.data() + symtab.sh_offset + sizeof(rocjitsu::Elf64_Sym),
              sizeof(descriptor));
  descriptor.st_name = 0;
  descriptor.st_info = 0;
  descriptor.st_shndx = rocjitsu::SHN_UNDEF;
  descriptor.st_value = 0;
  descriptor.st_size = 0;
  std::memcpy(image.data() + symtab.sh_offset + sizeof(rocjitsu::Elf64_Sym), &descriptor,
              sizeof(descriptor));
}

std::vector<uint8_t> make_descriptorless_code_object() {
  constexpr std::array<uint32_t, 2> text_words = {0xbf800000u, 0};
  auto image = make_gfx1250_smoke_code_object(text_words);
  clear_smoke_kernel_descriptor_symbol(image);
  return image;
}

std::vector<uint8_t> make_empty_text_code_object() {
  auto image = make_descriptorless_code_object();

  rocjitsu::Elf64_Ehdr header{};
  std::memcpy(&header, image.data(), sizeof(header));

  rocjitsu::Elf64_Phdr executable_load{};
  std::memcpy(&executable_load, image.data() + header.e_phoff, sizeof(executable_load));
  executable_load.p_filesz = 0;
  executable_load.p_memsz = 0;
  std::memcpy(image.data() + header.e_phoff, &executable_load, sizeof(executable_load));

  const uint64_t text_section = header.e_shoff + sizeof(rocjitsu::Elf64_Shdr);
  rocjitsu::Elf64_Shdr text{};
  std::memcpy(&text, image.data() + text_section, sizeof(text));
  text.sh_size = 0;
  std::memcpy(image.data() + text_section, &text, sizeof(text));

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

/// @brief Read one `name=<count>` field out of the summary line.
///
/// @details The counters are space-separated, so a substring search for
/// "expand=1" also matches "expand=10". Anchoring on the leading space and
/// reading to the next delimiter makes the comparison numeric, so a count that
/// drifts fails instead of still matching a prefix.
std::optional<long long> summary_counter(std::string_view text, std::string_view name) {
  const std::string key = " " + std::string(name) + "=";
  const size_t at = text.find(key);
  if (at == std::string_view::npos)
    return std::nullopt;
  const size_t first = at + key.size();
  const size_t last = text.find_first_not_of("0123456789", first);
  if (last == first)
    return std::nullopt;
  long long value = 0;
  const auto digits = text.substr(first, last - first);
  for (const char digit : digits)
    value = value * 10 + (digit - '0');
  return value;
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

TEST(RjDbtTranslateIdempotence, ReportsSyntheticMismatchThroughCli) {
  std::array arguments = {
      std::string("rj_dbt_translate"),
      std::string("synthetic.co"),
      std::string("--input-target"),
      std::string("gfx1250"),
      std::string("--input-revision"),
      std::string("b0"),
      std::string("--output-target"),
      std::string("gfx1250"),
      std::string("--output-revision"),
      std::string("a0"),
      std::string("--verify-idempotence"),
      std::string("--output-mode"),
      std::string("diff"),
  };
  std::vector<char *> argv;
  argv.reserve(arguments.size());
  for (std::string &argument : arguments)
    argv.push_back(argument.data());

  testing::internal::CaptureStdout();
  testing::internal::CaptureStderr();
  const int status = rocjitsu::tools::detail::run_dbt_translate_cli(
      static_cast<int>(argv.size()), argv.data(), synthetic_idempotence_mismatch);
  const std::string stderr_text = testing::internal::GetCapturedStderr();
  const std::string stdout_text = testing::internal::GetCapturedStdout();

  EXPECT_EQ(status, 5);
  EXPECT_TRUE(contains(stdout_text, "idempotence: not-verified")) << stdout_text;
  EXPECT_TRUE(contains(stderr_text, "translation output is not byte-idempotent")) << stderr_text;
  EXPECT_TRUE(contains(stderr_text, "section '.text' first differs at 0x4")) << stderr_text;
}

TEST(RjDbtTranslateRewriteDischarge, ReportsSyntheticResidualThroughCli) {
  std::array arguments = {
      std::string("rj_dbt_translate"),
      std::string("synthetic.co"),
      std::string("--input-target"),
      std::string("gfx1250"),
      std::string("--input-revision"),
      std::string("b0"),
      std::string("--output-target"),
      std::string("gfx1250"),
      std::string("--output-revision"),
      std::string("a0"),
      std::string("--verify-rewrite-discharge"),
      std::string("--output-mode"),
      std::string("diff"),
  };
  std::vector<char *> argv;
  argv.reserve(arguments.size());
  for (std::string &argument : arguments)
    argv.push_back(argument.data());

  testing::internal::CaptureStdout();
  testing::internal::CaptureStderr();
  const int status = rocjitsu::tools::detail::run_dbt_translate_cli(
      static_cast<int>(argv.size()), argv.data(), synthetic_residual_rewrite);
  const std::string stderr_text = testing::internal::GetCapturedStderr();
  const std::string stdout_text = testing::internal::GetCapturedStdout();

  EXPECT_EQ(status, 3);
  EXPECT_TRUE(contains(stdout_text, "rewrite_discharge: not-verified")) << stdout_text;
  EXPECT_TRUE(contains(stderr_text, "residual-rewrite output:.text+0x0004 s_clause"))
      << stderr_text;
  EXPECT_TRUE(contains(stderr_text, "registered rewrite remains actionable in final output"))
      << stderr_text;
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

  const std::array<std::string_view, 6> expected = {
      "rj_dbt_translate: ok",   "source_code_object_id: fnv1a64:",
      "source_words: bf8cc07f", "source: s_waitcnt",
      "target_words:",          "target: s_wait",
  };
  for (const std::string_view needle : expected) {
    EXPECT_TRUE(contains(stdout_text, needle))
        << "missing expected output fragment: " << needle << "\noutput:\n"
        << stdout_text;
  }
}

TEST(RjDbtTranslate, EmptyTextSameArchWarnsAndCopiesInput) {
  const rocjitsu::test::ScopedTempDirectory temp_dir("rj_dbt_translate_empty_text_");
  const std::filesystem::path temp_path(temp_dir.path());

  const auto input = temp_path / "data_only_gfx1250.co";
  const auto output = temp_path / "translated.co";
  const auto error = temp_path / "stderr.txt";
  const auto image = make_empty_text_code_object();

  {
    std::ofstream out(input, std::ios::binary);
    out.write(reinterpret_cast<const char *>(image.data()),
              static_cast<std::streamsize>(image.size()));
  }

  const std::string command = shell_quote(g_translate_tool.string()) + " " +
                              shell_quote(input.string()) +
                              " --input-target gfx1250 --input-revision b0 --output-target gfx1250 "
                              "--output-revision a0 --output-mode code-object > " +
                              shell_quote(output.string()) + " 2> " + shell_quote(error.string());

  const int status = std::system(command.c_str());
  const std::string output_bytes = read_text_file(output);
  const std::string stderr_text = read_text_file(error);

  ASSERT_TRUE(command_succeeded(status)) << stderr_text;
  EXPECT_EQ(output_bytes, std::string(reinterpret_cast<const char *>(image.data()), image.size()));
  EXPECT_TRUE(contains(stderr_text, "warning: data-only: code object has no executable sections, "
                                    "segments, or callable symbols; leaving unchanged"))
      << stderr_text;
}

TEST(RjDbtTranslate, DescriptorlessExecutableWarnsAndCopiesInput) {
  const rocjitsu::test::ScopedTempDirectory temp_dir("rj_dbt_translate_descriptorless_");
  const std::filesystem::path temp_path(temp_dir.path());

  const auto input = temp_path / "descriptorless_gfx1250.co";
  const auto output = temp_path / "translated.co";
  const auto error = temp_path / "stderr.txt";
  const auto image = make_descriptorless_code_object();

  {
    std::ofstream out(input, std::ios::binary);
    out.write(reinterpret_cast<const char *>(image.data()),
              static_cast<std::streamsize>(image.size()));
  }

  const std::string command = shell_quote(g_translate_tool.string()) + " " +
                              shell_quote(input.string()) +
                              " --input-target gfx1250 --input-revision b0 --output-target gfx1250 "
                              "--output-revision a0 --output-mode code-object > " +
                              shell_quote(output.string()) + " 2> " + shell_quote(error.string());

  const int status = std::system(command.c_str());
  const std::string output_bytes = read_text_file(output);
  const std::string stderr_text = read_text_file(error);

  ASSERT_TRUE(command_succeeded(status)) << stderr_text;
  EXPECT_EQ(output_bytes, std::string(reinterpret_cast<const char *>(image.data()), image.size()));
  EXPECT_TRUE(contains(stderr_text,
                       "warning: nothing-to-translate: code object has no kernel descriptors; "
                       "leaving executable text unchanged"))
      << stderr_text;
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

TEST(RjDbtTranslate, AcceptsGfx1251WithoutGfx1250Revisions) {
  std::array arguments = {
      std::string("rj_dbt_translate"), std::string("synthetic.co"),
      std::string("--input-target"),   std::string("gfx1251"),
      std::string("--output-target"),  std::string("gfx1251"),
      std::string("--output-mode"),    std::string("diff"),
  };
  std::vector<char *> argv;
  argv.reserve(arguments.size());
  for (std::string &argument : arguments)
    argv.push_back(argument.data());

  testing::internal::CaptureStdout();
  testing::internal::CaptureStderr();
  const int status = rocjitsu::tools::detail::run_dbt_translate_cli(
      static_cast<int>(argv.size()), argv.data(), capture_gfx1251_targets);
  const std::string stderr_text = testing::internal::GetCapturedStderr();
  const std::string stdout_text = testing::internal::GetCapturedStdout();

  EXPECT_EQ(status, 0) << stderr_text;
  EXPECT_TRUE(stderr_text.empty()) << stderr_text;
  EXPECT_TRUE(contains(stdout_text, "input_target: gfx1251")) << stdout_text;
  EXPECT_TRUE(contains(stdout_text, "output_target: gfx1251")) << stdout_text;
}

TEST(RjDbtTranslate, RejectsGfx1250RevisionOptionsForGfx1251) {
  rocjitsu::tools::TranslateOptions options;
  options.input_target = ROCJITSU_CODE_TARGET_GFX1251;
  options.guest_arch = ROCJITSU_CODE_ARCH_CDNA5;
  options.target_mach = rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX1251;
  options.host_arch = ROCJITSU_CODE_ARCH_CDNA5;
  options.input_revision = rocjitsu::ProcessorRevision::Gfx1250B0;

  auto error = rocjitsu::tools::translation_request_error(options);
  ASSERT_TRUE(error.has_value());
  EXPECT_EQ(*error, "--input-revision is only valid when --input-target is gfx1250");

  options.input_revision = rocjitsu::ProcessorRevision::Unspecified;
  options.output_revision = rocjitsu::ProcessorRevision::Gfx1250A0;
  error = rocjitsu::tools::translation_request_error(options);
  ASSERT_TRUE(error.has_value());
  EXPECT_EQ(*error, "--output-revision is only valid when --output-target is gfx1250");
}

TEST(RjDbtTranslate, AppliesGfx1250RevisionContractToDefaultCdna5OutputMachine) {
  rocjitsu::tools::TranslateOptions options;
  options.input_target = ROCJITSU_CODE_TARGET_GFX1251;
  options.guest_arch = ROCJITSU_CODE_ARCH_CDNA5;
  options.host_arch = ROCJITSU_CODE_ARCH_CDNA5;
  options.target_mach = 0;

  const auto error = rocjitsu::tools::translation_request_error(options);
  ASSERT_TRUE(error.has_value());
  EXPECT_EQ(*error, "--output-revision is required when --output-target is gfx1250");
}

TEST(RjDecodeBenchmark, PreservesConcreteGfx1251Identity) {
  ASSERT_FALSE(g_decode_benchmark.empty());
  const rocjitsu::test::ScopedTempDirectory temp_dir("rj_decode_benchmark_gfx1251_");
  const std::filesystem::path temp_path(temp_dir.path());
  const auto input = temp_path / "gfx1251.co";
  const auto output = temp_path / "stdout.txt";
  const auto error = temp_path / "stderr.txt";

  constexpr std::array<uint32_t, 2> gfx1251_only_instruction = {
      0xCC4B4004u,
      0x1A021908u,
  };
  {
    const std::vector<uint8_t> image =
        make_smoke_code_object(rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX1251, gfx1251_only_instruction);
    std::ofstream out(input, std::ios::binary);
    out.write(reinterpret_cast<const char *>(image.data()),
              static_cast<std::streamsize>(image.size()));
  }

  const std::string automatic_command =
      shell_quote(g_decode_benchmark.string()) + " " + shell_quote(input.string()) +
      " --iterations 1 --invalid-limit 0 > " + shell_quote(output.string()) + " 2> " +
      shell_quote(error.string());
  int status = std::system(automatic_command.c_str());
  EXPECT_TRUE(command_succeeded(status)) << read_text_file(error);
  EXPECT_TRUE(contains(read_text_file(output), "1 valid instructions")) << read_text_file(output);

  const std::string explicit_command =
      shell_quote(g_decode_benchmark.string()) + " " + shell_quote(input.string()) +
      " --target gfx1251 --iterations 1 --invalid-limit 0 > " + shell_quote(output.string()) +
      " 2> " + shell_quote(error.string());
  status = std::system(explicit_command.c_str());
  EXPECT_TRUE(command_succeeded(status)) << read_text_file(error);

  const std::string mismatch_command =
      shell_quote(g_decode_benchmark.string()) + " " + shell_quote(input.string()) +
      " --target gfx1250 --iterations 1 --invalid-limit 0 > " + shell_quote(output.string()) +
      " 2> " + shell_quote(error.string());
  status = std::system(mismatch_command.c_str());
  EXPECT_TRUE(command_exited_with(status, 2));
  EXPECT_TRUE(contains(read_text_file(error), "does not match code-object target"))
      << read_text_file(error);
}

TEST(RjDbtTranslate, RejectsStandaloneConcreteTargetMismatch) {
  const rocjitsu::test::ScopedTempDirectory temp_dir("rj_dbt_translate_target_mismatch_");
  const std::filesystem::path temp_path(temp_dir.path());
  const auto input = temp_path / "input.co";
  const auto output = temp_path / "stdout.txt";
  const auto error = temp_path / "stderr.txt";

  constexpr std::array<uint32_t, 2> text_words = {0xBF800000u, 0xBF800000u};
  const auto run_mismatch = [&](uint32_t elf_mach, std::string_view arguments) {
    const std::vector<uint8_t> image = make_smoke_code_object(elf_mach, text_words);
    {
      std::ofstream out(input, std::ios::binary);
      out.write(reinterpret_cast<const char *>(image.data()),
                static_cast<std::streamsize>(image.size()));
    }
    const std::string command = shell_quote(g_translate_tool.string()) + " " +
                                shell_quote(input.string()) + " " + std::string(arguments) + " > " +
                                shell_quote(output.string()) + " 2> " + shell_quote(error.string());
    const int status = std::system(command.c_str());
    EXPECT_TRUE(command_exited_with(status, 2));
    EXPECT_TRUE(contains(read_text_file(error),
                         "input target does not match standalone code-object target metadata"))
        << read_text_file(error);
  };

  run_mismatch(rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX1250,
               "--input-target gfx1251 --output-target gfx1251 --output-mode diff");
  run_mismatch(rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX1251,
               "--input-target gfx1250 --input-revision b0 --output-target gfx1250 "
               "--output-revision a0 --output-mode diff");
}

TEST(RjDbtTranslate, ReportsGfx1251OnlyInstructionWithConcreteDecoders) {
  const rocjitsu::test::ScopedTempDirectory temp_dir("rj_dbt_translate_gfx1251_report_");
  const std::filesystem::path temp_path(temp_dir.path());
  const auto input = temp_path / "gfx1251.co";
  const auto output = temp_path / "stdout.txt";
  const auto error = temp_path / "stderr.txt";

  constexpr std::array<uint32_t, 3> gfx1251_only_instruction = {
      0xCC4B4004u,
      0x1A021908u,
      0xBFB00000u,
  };
  {
    const std::vector<uint8_t> image =
        make_smoke_code_object(rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX1251, gfx1251_only_instruction);
    std::ofstream out(input, std::ios::binary);
    out.write(reinterpret_cast<const char *>(image.data()),
              static_cast<std::streamsize>(image.size()));
  }

  const std::string command = shell_quote(g_translate_tool.string()) + " " +
                              shell_quote(input.string()) +
                              " --input-target gfx1251 --output-target gfx1251 --output-mode diff "
                              "--show-all-translations > " +
                              shell_quote(output.string()) + " 2> " + shell_quote(error.string());
  const int status = std::system(command.c_str());
  const std::string stdout_text = read_text_file(output);
  const std::string stderr_text = read_text_file(error);

  ASSERT_TRUE(command_succeeded(status)) << "stderr:\n"
                                         << stderr_text << "\nstdout:\n"
                                         << stdout_text;
  EXPECT_TRUE(stderr_text.empty()) << stderr_text;
  EXPECT_TRUE(contains(stdout_text, "source: v_pk_add_f64")) << stdout_text;
  EXPECT_TRUE(contains(stdout_text, "target: v_pk_add_f64")) << stdout_text;
  EXPECT_TRUE(contains(stdout_text, "section .text bytes=12 instructions=2 decode_failures=0"))
      << stdout_text;
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

TEST(RjDbtTranslate, VerifiesGfx1250IdempotenceAndRewriteDischarge) {
  const rocjitsu::test::ScopedTempDirectory temp_dir("rj_dbt_translate_rewrite_discharge_");
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

  const std::string command =
      shell_quote(g_translate_tool.string()) + " " + shell_quote(input.string()) +
      " --input-target gfx1250 --input-revision b0 --output-target gfx1250 "
      "--output-revision a0 --verify-idempotence --verify-rewrite-discharge "
      "--output-mode diff > " +
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
  EXPECT_TRUE(contains(stdout_text, "rewrite_discharge: verified")) << stdout_text;
  EXPECT_TRUE(contains(stdout_text, "changed=1")) << stdout_text;
  EXPECT_TRUE(contains(stdout_text, "source: s_clause 4")) << stdout_text;
  EXPECT_TRUE(contains(stdout_text, "target: s_nop 0")) << stdout_text;
}

TEST(RjDbtTranslate, RejectsInvalidRewriteDischargeOptionCombinations) {
  rocjitsu::tools::TranslateOptions options;
  options.verify_rewrite_discharge = true;
  auto error = rocjitsu::tools::translation_request_error(options);
  ASSERT_TRUE(error.has_value());
  EXPECT_EQ(*error, "--verify-rewrite-discharge requires gfx1250 b0-to-a0 translation");

  options.guest_arch = ROCJITSU_CODE_ARCH_CDNA5;
  options.host_arch = ROCJITSU_CODE_ARCH_CDNA5;
  options.input_target = ROCJITSU_CODE_TARGET_GFX1250;
  options.target_mach = rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX1250;
  options.input_revision = rocjitsu::ProcessorRevision::Gfx1250B0;
  options.output_revision = rocjitsu::ProcessorRevision::Gfx1250A0;
  options.skip_failed_kernels = true;
  error = rocjitsu::tools::translation_request_error(options);
  ASSERT_TRUE(error.has_value());
  EXPECT_EQ(*error, "--verify-rewrite-discharge cannot be combined with --skip-failed-kernels");
}

TEST(RjDbtTranslate, RejectsRewriteDischargeWithListCodeObjects) {
  std::array arguments = {
      std::string("rj_dbt_translate"),
      std::string("unused.co"),
      std::string("--list-code-objects"),
      std::string("--verify-rewrite-discharge"),
  };
  std::vector<char *> argv;
  argv.reserve(arguments.size());
  for (std::string &argument : arguments)
    argv.push_back(argument.data());

  testing::internal::CaptureStdout();
  testing::internal::CaptureStderr();
  const int status = rocjitsu::tools::detail::run_dbt_translate_cli(
      static_cast<int>(argv.size()), argv.data(), rocjitsu::tools::translate_code_object);
  const std::string stderr_text = testing::internal::GetCapturedStderr();
  const std::string stdout_text = testing::internal::GetCapturedStdout();

  EXPECT_EQ(status, 1);
  EXPECT_TRUE(stdout_text.empty()) << stdout_text;
  EXPECT_TRUE(contains(stderr_text,
                       "--verify-rewrite-discharge cannot be combined with --list-code-objects"));
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

TEST(RjDbtTranslate, VerifiesGfx1250ClusterLoadIdempotence) {
  const rocjitsu::test::ScopedTempDirectory temp_dir("rj_dbt_translate_cluster_load_idempotence_");
  const std::filesystem::path temp_path(temp_dir.path());
  const std::filesystem::path input = temp_path / "cluster_load_gfx1250.co";
  const std::filesystem::path output = temp_path / "stdout.txt";
  const std::filesystem::path error = temp_path / "stderr.txt";

  // The first pass wraps the cluster load with an M0 save/clear/restore. The
  // second pass recognizes the canonical same-block clear and preserves it.
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

  ASSERT_TRUE(command_succeeded(status)) << "stderr:\n"
                                         << stderr_text << "\nstdout:\n"
                                         << stdout_text;
  EXPECT_TRUE(stderr_text.empty()) << stderr_text;
  EXPECT_TRUE(contains(stdout_text, "idempotence: verified")) << stdout_text;
  EXPECT_TRUE(contains(stdout_text, "idempotence_diagnostics: 0")) << stdout_text;
}

// A rule registered in the semantic table with no legalization entry reports through a label and a
// counter branch that no other fixture reaches: the existing ones all translate through a
// legalization-classified rule. The MODE separation rule is the only one shaped this way, so
// without this the label and the partition can drift while the suite stays green.
TEST(RjDbtTranslate, ReportsUnclassifiedSemanticExpansionInTheDiff) {
  const rocjitsu::test::ScopedTempDirectory temp_dir("rj_dbt_translate_unclassified_semantic_");
  const std::filesystem::path temp_path(temp_dir.path());
  const std::filesystem::path input = temp_path / "mode_setreg_gfx1250.co";
  const std::filesystem::path output = temp_path / "stdout.txt";
  const std::filesystem::path error = temp_path / "stderr.txt";

  // s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 0, 32), 0 -- the one write the rule separates.
  constexpr std::array<uint32_t, 3> text_words = {0xb980f801u, 0x00000001u, 0xbfb00000u};
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

  ASSERT_TRUE(command_succeeded(status)) << "stderr:\n"
                                         << stderr_text << "\nstdout:\n"
                                         << stdout_text;
  EXPECT_TRUE(stderr_text.empty()) << stderr_text;

  // The label must name the rule that rewrote the instruction rather than a re-encode.
  EXPECT_TRUE(contains(stdout_text, "expand semantic")) << stdout_text;
  // The counters must still partition the total: the expansion belongs to expand, not encode.
  EXPECT_EQ(summary_counter(stdout_text, "expand"), 1) << stdout_text;
  EXPECT_EQ(summary_counter(stdout_text, "encode"), 0) << stdout_text;
  EXPECT_EQ(summary_counter(stdout_text, "semantic"), 1) << stdout_text;
  EXPECT_TRUE(contains(stdout_text, "idempotence: verified")) << stdout_text;
}

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "usage: rj_dbt_translate_smoke_test <rj_dbt_translate> "
                 "<rj_decode_benchmark>\n";
    return 2;
  }

  g_translate_tool = argv[1];
  g_decode_benchmark = argv[2];
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
