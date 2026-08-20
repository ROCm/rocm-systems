// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file translate_test_support.cpp
/// @brief Shared fixture implementations for CPU-only DBT translation tests.

#include "translate_test_support.h"
#include "decode_test_util.h"

#include "elf_test_support.h"
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/opcodes.h"
#include "rocjitsu/isa/decoder.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace rocjitsu::test_support {

static uint32_t add_elf_name(std::vector<uint8_t> &names, std::string_view name) {
  const uint32_t offset = static_cast<uint32_t>(names.size());
  names.resize(names.size() + name.size() + 1);
  if (!name.empty())
    std::memcpy(names.data() + offset, name.data(), name.size());
  return offset;
}

static uint64_t align_up_for_test(uint64_t value, uint64_t alignment) {
  const uint64_t remainder = value % alignment;
  return remainder == 0 ? value : value + alignment - remainder;
}
static void write_bytes_for_test(std::vector<uint8_t> &image, uint64_t offset, const void *src,
                                 size_t size) {
  assert(offset <= image.size());
  assert(size <= image.size() - offset);
  std::memcpy(image.data() + offset, src, size);
}

uint16_t append_elf_section_for_test(std::vector<uint8_t> &image, Elf64_Shdr section,
                                     std::span<const uint8_t> contents) {
  auto header = read_elf_struct_for_test<Elf64_Ehdr>(image, 0);
  auto sections = read_elf_array_for_test<Elf64_Shdr>(image, header.e_shoff, header.e_shnum);
  assert(header.e_shnum < std::numeric_limits<uint16_t>::max());
  assert(header.e_shoff <= image.size());

  const uint64_t contents_offset = header.e_shoff;
  image.insert(image.begin() + static_cast<std::ptrdiff_t>(contents_offset), contents.begin(),
               contents.end());
  section.sh_offset = contents_offset;
  section.sh_size = contents.size();
  sections.push_back(section);

  header.e_shoff += contents.size();
  header.e_shnum = static_cast<uint16_t>(sections.size());
  image.resize(header.e_shoff + sections.size() * sizeof(Elf64_Shdr));
  write_elf_struct_for_test(image, 0, header);
  write_bytes_for_test(image, header.e_shoff, sections.data(),
                       sections.size() * sizeof(Elf64_Shdr));
  return static_cast<uint16_t>(sections.size() - 1);
}
constexpr size_t kKernelDescriptorEntryOffset =
    offsetof(TestKernelDescriptor, kernel_code_entry_byte_offset);
void write_kernel_descriptor_entry_offset(void *descriptor, int64_t entry_offset) {
  auto *bytes = static_cast<uint8_t *>(descriptor);
  std::memcpy(bytes + kKernelDescriptorEntryOffset, &entry_offset, sizeof(entry_offset));
}

int64_t read_kernel_descriptor_entry_offset(const void *descriptor) {
  const auto *bytes = static_cast<const uint8_t *>(descriptor);
  int64_t entry_offset = 0;
  std::memcpy(&entry_offset, bytes + kKernelDescriptorEntryOffset, sizeof(entry_offset));
  return entry_offset;
}

TestKernelDescriptor read_kernel_descriptor_for_test(const void *descriptor) {
  TestKernelDescriptor kd{};
  std::memcpy(&kd, descriptor, sizeof(kd));
  return kd;
}

void write_kernel_descriptor_for_test(void *descriptor, const TestKernelDescriptor &kd) {
  std::memcpy(descriptor, &kd, sizeof(kd));
}

static std::vector<uint8_t> make_kernel_descriptor_bytes(int64_t entry_offset) {
  std::vector<uint8_t> descriptor(kKernelDescriptorSize, 0);
  write_kernel_descriptor_entry_offset(descriptor.data(), entry_offset);
  return descriptor;
}
std::vector<uint8_t> make_minimal_amdgpu_elf_with_descriptor_after_text(
    const std::vector<uint32_t> &text_words, std::optional<size_t> text_function_words,
    size_t text_function_offset_words, std::optional<size_t> function_pointer_table_target_words,
    bool name_function_pointer_table_with_symbol, bool export_text_function) {
  if (text_function_words && text_function_offset_words + *text_function_words > text_words.size())
    throw std::invalid_argument("text function extent exceeds .text fixture");
  if (function_pointer_table_target_words &&
      *function_pointer_table_target_words >= text_words.size())
    throw std::invalid_argument("function pointer target exceeds .text fixture");
  constexpr uint64_t text_offset = 0x100;
  constexpr uint64_t text_vaddr = 0x1100;
  const uint64_t text_size = text_words.size() * sizeof(uint32_t);
  constexpr uint64_t load_align = 0x1000;
  constexpr uint64_t rodata_size = kKernelDescriptorSize;
  const bool has_table = function_pointer_table_target_words.has_value();
  constexpr uint64_t table_size = sizeof(uint64_t);

  std::vector<uint8_t> shstrtab{'\0'};
  const uint32_t text_name = add_elf_name(shstrtab, ".text");
  const uint32_t rodata_name = add_elf_name(shstrtab, ".rodata");
  const uint32_t symtab_name = add_elf_name(shstrtab, ".symtab");
  const uint32_t strtab_name = add_elf_name(shstrtab, ".strtab");
  const uint32_t shstrtab_name = add_elf_name(shstrtab, ".shstrtab");
  const uint32_t table_name = has_table ? add_elf_name(shstrtab, ".data.rel.ro") : 0;
  const uint32_t rela_name = has_table ? add_elf_name(shstrtab, ".rela.dyn") : 0;

  std::vector<uint8_t> strtab{'\0'};
  const uint32_t kd_symbol_name = add_elf_name(strtab, "kernel.kd");
  const uint32_t text_symbol_name =
      text_function_words ? add_elf_name(strtab, export_text_function ? "device_fn" : "kernel") : 0;
  const uint32_t table_symbol_name = has_table ? add_elf_name(strtab, "function_table") : 0;

  // The kernel descriptor requires 8-byte alignment (tests reinterpret_cast the
  // .rodata bytes to TestKernelDescriptor). An odd text_words count leaves
  // text_offset + text_size only 4-aligned, so pad up to 8. text_offset and
  // text_vaddr are both 8-aligned and differ by a multiple of load_align, so
  // padding both keeps the PT_LOAD p_offset == p_vaddr (mod p_align) congruence.
  const uint64_t rodata_offset = align_up_for_test(text_offset + text_size, 8);
  const uint64_t rodata_vaddr = align_up_for_test(text_vaddr + text_size, 8) + load_align;
  const uint64_t table_vaddr = rodata_vaddr + load_align;
  const uint64_t strtab_offset = rodata_offset + rodata_size;
  const uint64_t symtab_offset = align_up_for_test(strtab_offset + strtab.size(), 8);
  const size_t sym_count = (text_function_words ? 3 : 2) + (has_table ? 1 : 0);
  // The table is SHF_ALLOC, so a real loader only maps it when a PT_LOAD covers it. Give the file
  // offset the same residue mod load_align as table_vaddr so the extra segment below satisfies the
  // p_offset == p_vaddr (mod p_align) congruence a loader requires.
  const uint64_t table_offset =
      align_up_for_test(symtab_offset + sym_count * sizeof(Elf64_Sym), load_align) +
      (table_vaddr % load_align);
  const uint64_t rela_offset = has_table ? table_offset + table_size : table_offset;
  const uint64_t shstrtab_offset = rela_offset + (has_table ? sizeof(Elf64_Rela) : 0);
  const uint64_t shoff = align_up_for_test(shstrtab_offset + shstrtab.size(), 8);
  const uint16_t section_count = has_table ? 8 : 6;
  const uint16_t phdr_count = has_table ? 3 : 2;

  std::vector<uint8_t> image(shoff + section_count * sizeof(Elf64_Shdr), 0);

  Elf64_Ehdr ehdr{};
  std::memcpy(ehdr.e_ident, EI_MAGIC, EI_MAGIC_SIZE);
  ehdr.e_ident[EI_CLASS] = ELFCLASS64;
  ehdr.e_ident[EI_OSABI] = ELFOSABI_AMDGPU_HSA;
  ehdr.e_type = ET_DYN;
  ehdr.e_machine = EM_AMDGPU;
  ehdr.e_version = 1;
  ehdr.e_phoff = sizeof(Elf64_Ehdr);
  ehdr.e_shoff = shoff;
  ehdr.e_flags = EF_AMDGPU_MACH_AMDGCN_GFX950;
  ehdr.e_ehsize = sizeof(Elf64_Ehdr);
  ehdr.e_phentsize = sizeof(Elf64_Phdr);
  ehdr.e_phnum = phdr_count;
  ehdr.e_shentsize = sizeof(Elf64_Shdr);
  ehdr.e_shnum = section_count;
  ehdr.e_shstrndx = 5;
  std::memcpy(image.data(), &ehdr, sizeof(ehdr));

  std::vector<Elf64_Phdr> phdrs(phdr_count);
  phdrs[0].p_type = PT_LOAD;
  phdrs[0].p_flags = 0x5; // PF_R | PF_X
  phdrs[0].p_offset = text_offset;
  phdrs[0].p_vaddr = text_vaddr;
  phdrs[0].p_paddr = text_vaddr;
  phdrs[0].p_filesz = text_size;
  phdrs[0].p_memsz = text_size;
  phdrs[0].p_align = load_align;

  phdrs[1].p_type = PT_LOAD;
  phdrs[1].p_flags = 0x4; // PF_R
  phdrs[1].p_offset = rodata_offset;
  phdrs[1].p_vaddr = rodata_vaddr;
  phdrs[1].p_paddr = rodata_vaddr;
  phdrs[1].p_filesz = rodata_size;
  phdrs[1].p_memsz = rodata_size;
  phdrs[1].p_align = load_align;
  if (has_table) {
    // Map the table itself, so these cases exercise a slot a loader would place rather than only
    // section-header discovery.
    phdrs[2].p_type = PT_LOAD;
    phdrs[2].p_flags = 0x4; // PF_R
    phdrs[2].p_offset = table_offset;
    phdrs[2].p_vaddr = table_vaddr;
    phdrs[2].p_paddr = table_vaddr;
    phdrs[2].p_filesz = table_size;
    phdrs[2].p_memsz = table_size;
    phdrs[2].p_align = load_align;
  }
  std::memcpy(image.data() + ehdr.e_phoff, phdrs.data(), phdrs.size() * sizeof(Elf64_Phdr));

  std::memcpy(image.data() + text_offset, text_words.data(), text_size);

  const auto descriptor = make_kernel_descriptor_bytes(static_cast<int64_t>(text_vaddr) -
                                                       static_cast<int64_t>(rodata_vaddr));
  std::memcpy(image.data() + rodata_offset, descriptor.data(), descriptor.size());
  std::memcpy(image.data() + strtab_offset, strtab.data(), strtab.size());

  std::vector<Elf64_Sym> syms(sym_count);
  syms[1].st_name = kd_symbol_name;
  syms[1].st_info = elf_symbol_info(kElfSymbolBindGlobal, kElfSymbolTypeObject);
  syms[1].st_shndx = 2;
  syms[1].st_value = rodata_vaddr;
  syms[1].st_size = kKernelDescriptorSize;
  if (text_function_words) {
    syms[2].st_name = text_symbol_name;
    // Real device functions are LOCAL and appear only in .symtab. Offset zero is the kernel entry,
    // which function discovery excludes; a non-zero offset names a callee body.
    syms[2].st_info = elf_symbol_info((text_function_offset_words == 0 || export_text_function)
                                          ? kElfSymbolBindGlobal
                                          : kElfSymbolBindLocal,
                                      kElfSymbolTypeFunc);
    syms[2].st_shndx = 1;
    syms[2].st_value = text_vaddr + text_function_offset_words * sizeof(uint32_t);
    syms[2].st_size = *text_function_words * sizeof(uint32_t);
  }
  if (has_table) {
    // A function-pointer table is an STT_OBJECT in a non-executable allocated section, with one
    // R_AMDGPU_RELATIVE64 slot whose addend is the callee's virtual address.
    if (name_function_pointer_table_with_symbol) {
      Elf64_Sym &table_symbol = syms.back();
      table_symbol.st_name = table_symbol_name;
      table_symbol.st_info = elf_symbol_info(kElfSymbolBindGlobal, kElfSymbolTypeObject);
      table_symbol.st_shndx = 6;
      table_symbol.st_value = table_vaddr;
      table_symbol.st_size = table_size;
    }

    Elf64_Rela rela{};
    rela.r_offset = table_vaddr;
    rela.r_info = (uint64_t{0} << 32) | rocjitsu::R_AMDGPU_RELATIVE64;
    rela.r_addend =
        static_cast<int64_t>(text_vaddr + *function_pointer_table_target_words * sizeof(uint32_t));
    std::memcpy(image.data() + rela_offset, &rela, sizeof(rela));
  }
  std::memcpy(image.data() + symtab_offset, syms.data(), syms.size() * sizeof(Elf64_Sym));

  std::memcpy(image.data() + shstrtab_offset, shstrtab.data(), shstrtab.size());

  std::vector<Elf64_Shdr> shdrs(section_count);
  shdrs[1].sh_name = text_name;
  shdrs[1].sh_type = SHT_PROGBITS;
  shdrs[1].sh_flags = SHF_ALLOC | SHF_EXECINSTR;
  shdrs[1].sh_addr = text_vaddr;
  shdrs[1].sh_offset = text_offset;
  shdrs[1].sh_size = text_size;
  shdrs[1].sh_addralign = sizeof(uint32_t);

  shdrs[2].sh_name = rodata_name;
  shdrs[2].sh_type = SHT_PROGBITS;
  shdrs[2].sh_flags = SHF_ALLOC;
  shdrs[2].sh_addr = rodata_vaddr;
  shdrs[2].sh_offset = rodata_offset;
  shdrs[2].sh_size = rodata_size;
  shdrs[2].sh_addralign = 64;

  if (has_table) {
    shdrs[6].sh_name = table_name;
    shdrs[6].sh_type = SHT_PROGBITS;
    shdrs[6].sh_flags = SHF_ALLOC;
    shdrs[6].sh_addr = table_vaddr;
    shdrs[6].sh_offset = table_offset;
    shdrs[6].sh_size = table_size;
    shdrs[6].sh_addralign = 8;

    shdrs[7].sh_name = rela_name;
    shdrs[7].sh_type = SHT_RELA;
    shdrs[7].sh_offset = rela_offset;
    shdrs[7].sh_size = sizeof(Elf64_Rela);
    shdrs[7].sh_link = 3;
    shdrs[7].sh_addralign = 8;
    shdrs[7].sh_entsize = sizeof(Elf64_Rela);
  }

  shdrs[3].sh_name = symtab_name;
  shdrs[3].sh_type = SHT_SYMTAB;
  shdrs[3].sh_offset = symtab_offset;
  shdrs[3].sh_size = syms.size() * sizeof(Elf64_Sym);
  shdrs[3].sh_link = 4;
  shdrs[3].sh_info = 1;
  shdrs[3].sh_addralign = 8;
  shdrs[3].sh_entsize = sizeof(Elf64_Sym);

  shdrs[4].sh_name = strtab_name;
  shdrs[4].sh_type = SHT_STRTAB;
  shdrs[4].sh_offset = strtab_offset;
  shdrs[4].sh_size = strtab.size();
  shdrs[4].sh_addralign = 1;

  shdrs[5].sh_name = shstrtab_name;
  shdrs[5].sh_type = SHT_STRTAB;
  shdrs[5].sh_offset = shstrtab_offset;
  shdrs[5].sh_size = shstrtab.size();
  shdrs[5].sh_addralign = 1;

  std::memcpy(image.data() + shoff, shdrs.data(), shdrs.size() * sizeof(Elf64_Shdr));
  return image;
}

std::vector<uint8_t> make_minimal_amdgpu_elf_with_descriptor_after_text() {
  return make_minimal_amdgpu_elf_with_descriptor_after_text({0xBF800000u, 0xBF800000u});
}
std::unique_ptr<Instruction> decode_one(uint32_t word, rj_code_arch_t arch) {
  auto decoder = Decoder::create(arch);
  if (!decoder)
    return nullptr;
  return std::unique_ptr<Instruction>(decode_valid(*decoder, &word));
}

bool has_error_containing(const TranslatedCodeObject &result, DiagnosticKind kind,
                          std::string_view message) {
  return std::any_of(result.diagnostics.begin(), result.diagnostics.end(),
                     [&](const TranslationDiagnostic &diagnostic) {
                       return diagnostic.severity == DiagnosticSeverity::Error &&
                              diagnostic.kind == kind &&
                              diagnostic.message.find(message) != std::string::npos;
                     });
}
bool has_warning_at(const TranslatedCodeObject &result, DiagnosticKind kind,
                    std::string_view message, uint64_t guest_offset) {
  return std::any_of(result.diagnostics.begin(), result.diagnostics.end(),
                     [&](const TranslationDiagnostic &diagnostic) {
                       return diagnostic.severity == DiagnosticSeverity::Warning &&
                              diagnostic.kind == kind && diagnostic.guest_offset == guest_offset &&
                              diagnostic.message.find(message) != std::string::npos;
                     });
}

std::vector<uint8_t> make_minimal_amdgpu_elf_with_two_kernels_and_function_pointers(
    const std::vector<uint32_t> &text_words, size_t kernel1_entry_word,
    const std::vector<TestTextFunction> &functions) {
  constexpr uint64_t text_offset = 0x100;
  constexpr uint64_t text_vaddr = 0x1100;
  const uint64_t text_size = text_words.size() * sizeof(uint32_t);
  constexpr uint64_t load_align = 0x1000;
  constexpr uint64_t rodata_size = 2 * kKernelDescriptorSize;
  const uint64_t slot_count = functions.size();
  const uint64_t table_bytes = slot_count * sizeof(uint64_t);

  std::vector<uint8_t> shstrtab{'\0'};
  const uint32_t text_name = add_elf_name(shstrtab, ".text");
  const uint32_t rodata_name = add_elf_name(shstrtab, ".rodata");
  const uint32_t symtab_name = add_elf_name(shstrtab, ".symtab");
  const uint32_t strtab_name = add_elf_name(shstrtab, ".strtab");
  const uint32_t shstrtab_name = add_elf_name(shstrtab, ".shstrtab");
  const uint32_t table_name = add_elf_name(shstrtab, ".data.rel.ro");
  const uint32_t rela_name = add_elf_name(shstrtab, ".rela.dyn");

  std::vector<uint8_t> strtab{'\0'};
  const uint32_t kernel0_name = add_elf_name(strtab, "kernel0.kd");
  const uint32_t kernel1_name = add_elf_name(strtab, "kernel1.kd");
  std::vector<uint32_t> helper_names;
  for (size_t i = 0; i < functions.size(); ++i)
    helper_names.push_back(add_elf_name(strtab, "helper" + std::to_string(i)));
  const uint32_t table_symbol_name = add_elf_name(strtab, "function_table");

  const uint64_t rodata_off = align_up_for_test(text_offset + text_size, 8);
  const uint64_t rodata_va = align_up_for_test(text_vaddr + text_size, 8) + load_align;
  const uint64_t table_va = rodata_va + load_align;
  const uint64_t strtab_off = rodata_off + rodata_size;
  const uint64_t symtab_off = align_up_for_test(strtab_off + strtab.size(), 8);
  const size_t sym_count = 4 + functions.size();
  const uint64_t table_off =
      align_up_for_test(symtab_off + sym_count * sizeof(Elf64_Sym), load_align) +
      (table_va % load_align);
  const uint64_t rela_off = table_off + table_bytes;
  const uint64_t shstrtab_off = rela_off + slot_count * sizeof(Elf64_Rela);
  const uint64_t shoff = align_up_for_test(shstrtab_off + shstrtab.size(), 8);
  constexpr uint16_t section_count = 8;
  constexpr uint16_t phdr_count = 3;

  std::vector<uint8_t> image(shoff + section_count * sizeof(Elf64_Shdr), 0);

  Elf64_Ehdr ehdr{};
  std::memcpy(ehdr.e_ident, EI_MAGIC, EI_MAGIC_SIZE);
  ehdr.e_ident[EI_CLASS] = ELFCLASS64;
  ehdr.e_ident[EI_OSABI] = ELFOSABI_AMDGPU_HSA;
  ehdr.e_type = ET_DYN;
  ehdr.e_machine = EM_AMDGPU;
  ehdr.e_version = 1;
  ehdr.e_phoff = sizeof(Elf64_Ehdr);
  ehdr.e_shoff = shoff;
  ehdr.e_flags = EF_AMDGPU_MACH_AMDGCN_GFX950;
  ehdr.e_ehsize = sizeof(Elf64_Ehdr);
  ehdr.e_phentsize = sizeof(Elf64_Phdr);
  ehdr.e_phnum = phdr_count;
  ehdr.e_shentsize = sizeof(Elf64_Shdr);
  ehdr.e_shnum = section_count;
  ehdr.e_shstrndx = 5;
  std::memcpy(image.data(), &ehdr, sizeof(ehdr));

  std::array<Elf64_Phdr, phdr_count> phdrs{};
  phdrs[0].p_type = PT_LOAD;
  phdrs[0].p_flags = 0x5; // PF_R | PF_X
  phdrs[0].p_offset = text_offset;
  phdrs[0].p_vaddr = text_vaddr;
  phdrs[0].p_paddr = text_vaddr;
  phdrs[0].p_filesz = text_size;
  phdrs[0].p_memsz = text_size;
  phdrs[0].p_align = load_align;

  phdrs[1].p_type = PT_LOAD;
  phdrs[1].p_flags = 0x4; // PF_R
  phdrs[1].p_offset = rodata_off;
  phdrs[1].p_vaddr = rodata_va;
  phdrs[1].p_paddr = rodata_va;
  phdrs[1].p_filesz = rodata_size;
  phdrs[1].p_memsz = rodata_size;
  phdrs[1].p_align = load_align;

  phdrs[2].p_type = PT_LOAD;
  phdrs[2].p_flags = 0x6; // PF_R | PF_W
  phdrs[2].p_offset = table_off;
  phdrs[2].p_vaddr = table_va;
  phdrs[2].p_paddr = table_va;
  phdrs[2].p_filesz = table_bytes;
  phdrs[2].p_memsz = table_bytes;
  phdrs[2].p_align = load_align;
  std::memcpy(image.data() + ehdr.e_phoff, phdrs.data(), phdrs.size() * sizeof(Elf64_Phdr));

  std::memcpy(image.data() + text_offset, text_words.data(), text_size);

  std::vector<uint8_t> descriptors(rodata_size, 0);
  write_kernel_descriptor_entry_offset(descriptors.data(), static_cast<int64_t>(text_vaddr) -
                                                               static_cast<int64_t>(rodata_va));
  write_kernel_descriptor_entry_offset(
      descriptors.data() + kKernelDescriptorSize,
      static_cast<int64_t>(text_vaddr + kernel1_entry_word * sizeof(uint32_t)) -
          static_cast<int64_t>(rodata_va + kKernelDescriptorSize));
  std::memcpy(image.data() + rodata_off, descriptors.data(), descriptors.size());
  std::memcpy(image.data() + strtab_off, strtab.data(), strtab.size());

  std::vector<Elf64_Sym> syms(sym_count);
  syms[1].st_name = kernel0_name;
  syms[1].st_info = elf_symbol_info(kElfSymbolBindGlobal, kElfSymbolTypeObject);
  syms[1].st_shndx = 2;
  syms[1].st_value = rodata_va;
  syms[1].st_size = kKernelDescriptorSize;
  syms[2].st_name = kernel1_name;
  syms[2].st_info = elf_symbol_info(kElfSymbolBindGlobal, kElfSymbolTypeObject);
  syms[2].st_shndx = 2;
  syms[2].st_value = rodata_va + kKernelDescriptorSize;
  syms[2].st_size = kKernelDescriptorSize;
  for (size_t i = 0; i < functions.size(); ++i) {
    Elf64_Sym &helper = syms[3 + i];
    helper.st_name = helper_names[i];
    helper.st_info = elf_symbol_info(kElfSymbolBindLocal, kElfSymbolTypeFunc);
    helper.st_shndx = 1;
    helper.st_value = text_vaddr + functions[i].offset_word * sizeof(uint32_t);
    helper.st_size = functions[i].words * sizeof(uint32_t);
  }
  Elf64_Sym &table_symbol = syms.back();
  table_symbol.st_name = table_symbol_name;
  table_symbol.st_info = elf_symbol_info(kElfSymbolBindGlobal, kElfSymbolTypeObject);
  table_symbol.st_shndx = 6;
  table_symbol.st_value = table_va;
  table_symbol.st_size = table_bytes;
  std::memcpy(image.data() + symtab_off, syms.data(), syms.size() * sizeof(Elf64_Sym));

  for (size_t i = 0; i < functions.size(); ++i) {
    Elf64_Rela rela{};
    rela.r_offset = table_va + i * sizeof(uint64_t);
    rela.r_info = (uint64_t{0} << 32) | rocjitsu::R_AMDGPU_RELATIVE64;
    rela.r_addend = static_cast<int64_t>(text_vaddr + functions[i].offset_word * sizeof(uint32_t));
    std::memcpy(image.data() + rela_off + i * sizeof(Elf64_Rela), &rela, sizeof(rela));
  }

  std::memcpy(image.data() + shstrtab_off, shstrtab.data(), shstrtab.size());

  std::array<Elf64_Shdr, section_count> shdrs{};
  shdrs[1].sh_name = text_name;
  shdrs[1].sh_type = SHT_PROGBITS;
  shdrs[1].sh_flags = SHF_ALLOC | SHF_EXECINSTR;
  shdrs[1].sh_addr = text_vaddr;
  shdrs[1].sh_offset = text_offset;
  shdrs[1].sh_size = text_size;
  shdrs[1].sh_addralign = sizeof(uint32_t);

  shdrs[2].sh_name = rodata_name;
  shdrs[2].sh_type = SHT_PROGBITS;
  shdrs[2].sh_flags = SHF_ALLOC;
  shdrs[2].sh_addr = rodata_va;
  shdrs[2].sh_offset = rodata_off;
  shdrs[2].sh_size = rodata_size;
  shdrs[2].sh_addralign = 64;

  shdrs[3].sh_name = symtab_name;
  shdrs[3].sh_type = SHT_SYMTAB;
  shdrs[3].sh_offset = symtab_off;
  shdrs[3].sh_size = syms.size() * sizeof(Elf64_Sym);
  shdrs[3].sh_link = 4;
  shdrs[3].sh_info = 1;
  shdrs[3].sh_addralign = 8;
  shdrs[3].sh_entsize = sizeof(Elf64_Sym);

  shdrs[4].sh_name = strtab_name;
  shdrs[4].sh_type = SHT_STRTAB;
  shdrs[4].sh_offset = strtab_off;
  shdrs[4].sh_size = strtab.size();
  shdrs[4].sh_addralign = 1;

  shdrs[5].sh_name = shstrtab_name;
  shdrs[5].sh_type = SHT_STRTAB;
  shdrs[5].sh_offset = shstrtab_off;
  shdrs[5].sh_size = shstrtab.size();
  shdrs[5].sh_addralign = 1;

  shdrs[6].sh_name = table_name;
  shdrs[6].sh_type = SHT_PROGBITS;
  shdrs[6].sh_flags = SHF_ALLOC | SHF_WRITE;
  shdrs[6].sh_addr = table_va;
  shdrs[6].sh_offset = table_off;
  shdrs[6].sh_size = table_bytes;
  shdrs[6].sh_addralign = 8;

  shdrs[7].sh_name = rela_name;
  shdrs[7].sh_type = SHT_RELA;
  shdrs[7].sh_offset = rela_off;
  shdrs[7].sh_size = slot_count * sizeof(Elf64_Rela);
  shdrs[7].sh_link = 3;
  shdrs[7].sh_info = 6;
  shdrs[7].sh_addralign = 8;
  shdrs[7].sh_entsize = sizeof(Elf64_Rela);

  std::memcpy(image.data() + shoff, shdrs.data(), shdrs.size() * sizeof(Elf64_Shdr));
  return image;
}

std::vector<uint8_t> make_minimal_amdgpu_elf_with_two_kernel_descriptors(
    const std::vector<uint32_t> &text_words,
    std::optional<TestRuntimeTextReference> runtime_text_reference) {
  constexpr uint64_t text_offset = 0x100;
  constexpr uint64_t text_vaddr = 0x1100;
  const uint64_t text_size = text_words.size() * sizeof(uint32_t);
  constexpr uint64_t load_align = 0x1000;
  const uint64_t rodata_size =
      2 * kKernelDescriptorSize + (runtime_text_reference ? sizeof(uint64_t) : 0);

  std::vector<uint8_t> shstrtab{'\0'};
  const uint32_t text_name = add_elf_name(shstrtab, ".text");
  const uint32_t rodata_name = add_elf_name(shstrtab, ".rodata");
  const uint32_t symtab_name = add_elf_name(shstrtab, ".symtab");
  const uint32_t strtab_name = add_elf_name(shstrtab, ".strtab");
  const uint32_t shstrtab_name = add_elf_name(shstrtab, ".shstrtab");
  const uint32_t rela_name = runtime_text_reference ? add_elf_name(shstrtab, ".rela.dyn") : 0;

  std::vector<uint8_t> strtab{'\0'};
  const uint32_t kernel0_name = add_elf_name(strtab, "kernel0.kd");
  const uint32_t kernel1_name = add_elf_name(strtab, "kernel1.kd");
  const uint32_t target_name = runtime_text_reference && runtime_text_reference->relocation ==
                                                             TestRuntimeTextRelocation::Abs64
                                   ? add_elf_name(strtab, "runtime_text_target")
                                   : 0;

  // The kernel descriptors require 8-byte alignment (tests reinterpret_cast the
  // .rodata bytes to TestKernelDescriptor). An odd text_words count leaves
  // text_offset + text_size only 4-aligned, so pad up to 8. text_offset and
  // text_vaddr are both 8-aligned and differ by a multiple of load_align, so
  // padding both keeps the PT_LOAD p_offset == p_vaddr (mod p_align) congruence.
  const uint64_t rodata_offset = align_up_for_test(text_offset + text_size, 8);
  const uint64_t rodata_vaddr = align_up_for_test(text_vaddr + text_size, 8) + load_align;
  const uint64_t strtab_offset = rodata_offset + rodata_size;
  const uint64_t symtab_offset = align_up_for_test(strtab_offset + strtab.size(), 8);
  const size_t sym_count = runtime_text_reference && runtime_text_reference->relocation ==
                                                         TestRuntimeTextRelocation::Abs64
                               ? 4
                               : 3;
  const uint64_t rela_offset = symtab_offset + sym_count * sizeof(Elf64_Sym);
  const uint64_t shstrtab_offset = rela_offset + (runtime_text_reference ? sizeof(Elf64_Rela) : 0);
  const uint64_t shoff = align_up_for_test(shstrtab_offset + shstrtab.size(), 8);
  const uint16_t section_count = runtime_text_reference ? 7 : 6;
  constexpr uint16_t phdr_count = 2;

  std::vector<uint8_t> image(shoff + section_count * sizeof(Elf64_Shdr), 0);

  Elf64_Ehdr ehdr{};
  std::memcpy(ehdr.e_ident, EI_MAGIC, EI_MAGIC_SIZE);
  ehdr.e_ident[EI_CLASS] = ELFCLASS64;
  ehdr.e_ident[EI_OSABI] = ELFOSABI_AMDGPU_HSA;
  ehdr.e_type = ET_DYN;
  ehdr.e_machine = EM_AMDGPU;
  ehdr.e_version = 1;
  ehdr.e_phoff = sizeof(Elf64_Ehdr);
  ehdr.e_shoff = shoff;
  ehdr.e_flags = EF_AMDGPU_MACH_AMDGCN_GFX950;
  ehdr.e_ehsize = sizeof(Elf64_Ehdr);
  ehdr.e_phentsize = sizeof(Elf64_Phdr);
  ehdr.e_phnum = phdr_count;
  ehdr.e_shentsize = sizeof(Elf64_Shdr);
  ehdr.e_shnum = section_count;
  ehdr.e_shstrndx = 5;
  std::memcpy(image.data(), &ehdr, sizeof(ehdr));

  std::array<Elf64_Phdr, phdr_count> phdrs{};
  phdrs[0].p_type = PT_LOAD;
  phdrs[0].p_flags = 0x5; // PF_R | PF_X
  phdrs[0].p_offset = text_offset;
  phdrs[0].p_vaddr = text_vaddr;
  phdrs[0].p_paddr = text_vaddr;
  phdrs[0].p_filesz = text_size;
  phdrs[0].p_memsz = text_size;
  phdrs[0].p_align = load_align;

  phdrs[1].p_type = PT_LOAD;
  phdrs[1].p_flags = 0x4; // PF_R
  phdrs[1].p_offset = rodata_offset;
  phdrs[1].p_vaddr = rodata_vaddr;
  phdrs[1].p_paddr = rodata_vaddr;
  phdrs[1].p_filesz = rodata_size;
  phdrs[1].p_memsz = rodata_size;
  phdrs[1].p_align = load_align;
  std::memcpy(image.data() + ehdr.e_phoff, phdrs.data(), phdrs.size() * sizeof(Elf64_Phdr));

  std::memcpy(image.data() + text_offset, text_words.data(), text_size);

  std::vector<uint8_t> descriptors(rodata_size, 0);
  write_kernel_descriptor_entry_offset(descriptors.data(), static_cast<int64_t>(text_vaddr) -
                                                               static_cast<int64_t>(rodata_vaddr));
  write_kernel_descriptor_entry_offset(
      descriptors.data() + kKernelDescriptorSize,
      static_cast<int64_t>(text_vaddr + sizeof(uint32_t)) -
          static_cast<int64_t>(rodata_vaddr + kKernelDescriptorSize));
  std::memcpy(image.data() + rodata_offset, descriptors.data(), descriptors.size());
  std::memcpy(image.data() + strtab_offset, strtab.data(), strtab.size());

  std::vector<Elf64_Sym> syms(sym_count);
  syms[1].st_name = kernel0_name;
  syms[1].st_info = elf_symbol_info(kElfSymbolBindGlobal, kElfSymbolTypeObject);
  syms[1].st_shndx = 2;
  syms[1].st_value = rodata_vaddr;
  syms[1].st_size = kKernelDescriptorSize;
  syms[2].st_name = kernel1_name;
  syms[2].st_info = elf_symbol_info(kElfSymbolBindGlobal, kElfSymbolTypeObject);
  syms[2].st_shndx = 2;
  syms[2].st_value = rodata_vaddr + kKernelDescriptorSize;
  syms[2].st_size = kKernelDescriptorSize;
  if (runtime_text_reference &&
      runtime_text_reference->relocation == TestRuntimeTextRelocation::Abs64) {
    syms[3].st_name = target_name;
    syms[3].st_info = elf_symbol_info(kElfSymbolBindGlobal, kElfSymbolTypeFunc);
    syms[3].st_shndx = 1;
    syms[3].st_value = text_vaddr + runtime_text_reference->target_text_offset;
    syms[3].st_size = sizeof(uint32_t);
  }
  std::memcpy(image.data() + symtab_offset, syms.data(), syms.size() * sizeof(Elf64_Sym));

  if (runtime_text_reference) {
    Elf64_Rela relocation{};
    relocation.r_offset = rodata_vaddr + 2 * kKernelDescriptorSize;
    if (runtime_text_reference->relocation == TestRuntimeTextRelocation::Abs64) {
      relocation.r_info = (static_cast<uint64_t>(3) << 32) |
                          static_cast<uint64_t>(runtime_text_reference->relocation_type);
    } else {
      relocation.r_info = R_AMDGPU_RELATIVE64;
      relocation.r_addend =
          static_cast<int64_t>(text_vaddr + runtime_text_reference->target_text_offset);
    }
    std::memcpy(image.data() + rela_offset, &relocation, sizeof(relocation));
  }

  std::memcpy(image.data() + shstrtab_offset, shstrtab.data(), shstrtab.size());

  std::vector<Elf64_Shdr> shdrs(section_count);
  shdrs[1].sh_name = text_name;
  shdrs[1].sh_type = SHT_PROGBITS;
  shdrs[1].sh_flags = SHF_ALLOC | SHF_EXECINSTR;
  shdrs[1].sh_addr = text_vaddr;
  shdrs[1].sh_offset = text_offset;
  shdrs[1].sh_size = text_size;
  shdrs[1].sh_addralign = sizeof(uint32_t);

  shdrs[2].sh_name = rodata_name;
  shdrs[2].sh_type = SHT_PROGBITS;
  shdrs[2].sh_flags = SHF_ALLOC;
  shdrs[2].sh_addr = rodata_vaddr;
  shdrs[2].sh_offset = rodata_offset;
  shdrs[2].sh_size = rodata_size;
  shdrs[2].sh_addralign = 64;

  shdrs[3].sh_name = symtab_name;
  shdrs[3].sh_type = SHT_SYMTAB;
  shdrs[3].sh_offset = symtab_offset;
  shdrs[3].sh_size = syms.size() * sizeof(Elf64_Sym);
  shdrs[3].sh_link = 4;
  shdrs[3].sh_info = 1;
  shdrs[3].sh_addralign = 8;
  shdrs[3].sh_entsize = sizeof(Elf64_Sym);

  shdrs[4].sh_name = strtab_name;
  shdrs[4].sh_type = SHT_STRTAB;
  shdrs[4].sh_offset = strtab_offset;
  shdrs[4].sh_size = strtab.size();
  shdrs[4].sh_addralign = 1;

  shdrs[5].sh_name = shstrtab_name;
  shdrs[5].sh_type = SHT_STRTAB;
  shdrs[5].sh_offset = shstrtab_offset;
  shdrs[5].sh_size = shstrtab.size();
  shdrs[5].sh_addralign = 1;

  if (runtime_text_reference) {
    shdrs[6].sh_name = rela_name;
    shdrs[6].sh_type = SHT_RELA;
    shdrs[6].sh_offset = rela_offset;
    shdrs[6].sh_size = sizeof(Elf64_Rela);
    shdrs[6].sh_link = 3;
    shdrs[6].sh_info = SHN_UNDEF;
    shdrs[6].sh_addralign = alignof(Elf64_Rela);
    shdrs[6].sh_entsize = sizeof(Elf64_Rela);
  }

  std::memcpy(image.data() + shoff, shdrs.data(), shdrs.size() * sizeof(Elf64_Shdr));
  return image;
}
std::vector<uint8_t> make_large_amdgpu_elf_with_waitcnt_entry() {
  constexpr uint64_t rodata_offset = 0x100;
  constexpr uint64_t rodata_vaddr = 0x100;
  constexpr uint64_t text_offset = 0x1000;
  constexpr uint64_t text_vaddr = 0x1000;
  constexpr uint64_t text_size = 0x21000;
  constexpr uint64_t rodata_size = kKernelDescriptorSize;
  constexpr uint64_t load_align = 0x1000;

  std::vector<uint8_t> shstrtab{'\0'};
  const uint32_t rodata_name = add_elf_name(shstrtab, ".rodata");
  const uint32_t text_name = add_elf_name(shstrtab, ".text");
  const uint32_t symtab_name = add_elf_name(shstrtab, ".symtab");
  const uint32_t strtab_name = add_elf_name(shstrtab, ".strtab");
  const uint32_t shstrtab_name = add_elf_name(shstrtab, ".shstrtab");

  std::vector<uint8_t> strtab{'\0'};
  const uint32_t kd_symbol_name = add_elf_name(strtab, "kernel.kd");

  const uint64_t strtab_offset = text_offset + text_size;
  const uint64_t symtab_offset = align_up_for_test(strtab_offset + strtab.size(), 8);
  constexpr size_t sym_count = 2;
  const uint64_t shstrtab_offset = symtab_offset + sym_count * sizeof(Elf64_Sym);
  const uint64_t shoff = align_up_for_test(shstrtab_offset + shstrtab.size(), 8);
  constexpr uint16_t section_count = 6;
  constexpr uint16_t phdr_count = 2;

  std::vector<uint8_t> image(shoff + section_count * sizeof(Elf64_Shdr), 0);

  write_bytes_for_test(image, offsetof(Elf64_Ehdr, e_ident), EI_MAGIC, EI_MAGIC_SIZE);
  image[offsetof(Elf64_Ehdr, e_ident) + EI_CLASS] = ELFCLASS64;
  image[offsetof(Elf64_Ehdr, e_ident) + EI_OSABI] = ELFOSABI_AMDGPU_HSA;
  write_value_for_test<uint16_t>(image, offsetof(Elf64_Ehdr, e_type), ET_DYN);
  write_value_for_test<uint16_t>(image, offsetof(Elf64_Ehdr, e_machine), EM_AMDGPU);
  write_value_for_test<uint32_t>(image, offsetof(Elf64_Ehdr, e_version), 1);
  write_value_for_test<uint64_t>(image, offsetof(Elf64_Ehdr, e_phoff), sizeof(Elf64_Ehdr));
  write_value_for_test<uint64_t>(image, offsetof(Elf64_Ehdr, e_shoff), shoff);
  write_value_for_test<uint32_t>(image, offsetof(Elf64_Ehdr, e_flags),
                                 EF_AMDGPU_MACH_AMDGCN_GFX950);
  write_value_for_test<uint16_t>(image, offsetof(Elf64_Ehdr, e_ehsize), sizeof(Elf64_Ehdr));
  write_value_for_test<uint16_t>(image, offsetof(Elf64_Ehdr, e_phentsize), sizeof(Elf64_Phdr));
  write_value_for_test<uint16_t>(image, offsetof(Elf64_Ehdr, e_phnum), phdr_count);
  write_value_for_test<uint16_t>(image, offsetof(Elf64_Ehdr, e_shentsize), sizeof(Elf64_Shdr));
  write_value_for_test<uint16_t>(image, offsetof(Elf64_Ehdr, e_shnum), section_count);
  write_value_for_test<uint16_t>(image, offsetof(Elf64_Ehdr, e_shstrndx), 5);

  const uint64_t phdr0 = sizeof(Elf64_Ehdr);
  write_value_for_test<uint32_t>(image, phdr0 + offsetof(Elf64_Phdr, p_type), PT_LOAD);
  write_value_for_test<uint32_t>(image, phdr0 + offsetof(Elf64_Phdr, p_flags), 0x4); // PF_R
  write_value_for_test<uint64_t>(image, phdr0 + offsetof(Elf64_Phdr, p_offset), rodata_offset);
  write_value_for_test<uint64_t>(image, phdr0 + offsetof(Elf64_Phdr, p_vaddr), rodata_vaddr);
  write_value_for_test<uint64_t>(image, phdr0 + offsetof(Elf64_Phdr, p_paddr), rodata_vaddr);
  write_value_for_test<uint64_t>(image, phdr0 + offsetof(Elf64_Phdr, p_filesz), rodata_size);
  write_value_for_test<uint64_t>(image, phdr0 + offsetof(Elf64_Phdr, p_memsz), rodata_size);
  write_value_for_test<uint64_t>(image, phdr0 + offsetof(Elf64_Phdr, p_align), load_align);

  const uint64_t phdr1 = phdr0 + sizeof(Elf64_Phdr);
  write_value_for_test<uint32_t>(image, phdr1 + offsetof(Elf64_Phdr, p_type), PT_LOAD);
  write_value_for_test<uint32_t>(image, phdr1 + offsetof(Elf64_Phdr, p_flags), 0x5); // PF_R | PF_X
  write_value_for_test<uint64_t>(image, phdr1 + offsetof(Elf64_Phdr, p_offset), text_offset);
  write_value_for_test<uint64_t>(image, phdr1 + offsetof(Elf64_Phdr, p_vaddr), text_vaddr);
  write_value_for_test<uint64_t>(image, phdr1 + offsetof(Elf64_Phdr, p_paddr), text_vaddr);
  write_value_for_test<uint64_t>(image, phdr1 + offsetof(Elf64_Phdr, p_filesz), text_size);
  write_value_for_test<uint64_t>(image, phdr1 + offsetof(Elf64_Phdr, p_memsz), text_size);
  write_value_for_test<uint64_t>(image, phdr1 + offsetof(Elf64_Phdr, p_align), load_align);

  const auto descriptor = make_kernel_descriptor_bytes(static_cast<int64_t>(text_vaddr) -
                                                       static_cast<int64_t>(rodata_vaddr));
  std::memcpy(image.data() + rodata_offset, descriptor.data(), descriptor.size());

  std::vector<uint32_t> text_words(text_size / sizeof(uint32_t),
                                   build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4));
  text_words[0] = cdna4::build_sopp(cdna4::kSWaitcntSopp)[0]; // Expands on RDNA4.
  std::memcpy(image.data() + text_offset, text_words.data(), text_size);
  std::memcpy(image.data() + strtab_offset, strtab.data(), strtab.size());

  const uint64_t sym1 = symtab_offset + sizeof(Elf64_Sym);
  write_value_for_test<uint32_t>(image, sym1 + offsetof(Elf64_Sym, st_name), kd_symbol_name);
  write_value_for_test<unsigned char>(image, sym1 + offsetof(Elf64_Sym, st_info),
                                      elf_symbol_info(kElfSymbolBindGlobal, kElfSymbolTypeObject));
  write_value_for_test<uint16_t>(image, sym1 + offsetof(Elf64_Sym, st_shndx), 1);
  write_value_for_test<uint64_t>(image, sym1 + offsetof(Elf64_Sym, st_value), rodata_vaddr);
  write_value_for_test<uint64_t>(image, sym1 + offsetof(Elf64_Sym, st_size), kKernelDescriptorSize);

  std::memcpy(image.data() + shstrtab_offset, shstrtab.data(), shstrtab.size());

  const auto write_shdr = [&](uint64_t index, uint32_t name, uint32_t type, uint64_t flags,
                              uint64_t addr, uint64_t offset, uint64_t size, uint32_t link,
                              uint32_t info, uint64_t addralign, uint64_t entsize) {
    const uint64_t base = shoff + index * sizeof(Elf64_Shdr);
    write_value_for_test<uint32_t>(image, base + offsetof(Elf64_Shdr, sh_name), name);
    write_value_for_test<uint32_t>(image, base + offsetof(Elf64_Shdr, sh_type), type);
    write_value_for_test<uint64_t>(image, base + offsetof(Elf64_Shdr, sh_flags), flags);
    write_value_for_test<uint64_t>(image, base + offsetof(Elf64_Shdr, sh_addr), addr);
    write_value_for_test<uint64_t>(image, base + offsetof(Elf64_Shdr, sh_offset), offset);
    write_value_for_test<uint64_t>(image, base + offsetof(Elf64_Shdr, sh_size), size);
    write_value_for_test<uint32_t>(image, base + offsetof(Elf64_Shdr, sh_link), link);
    write_value_for_test<uint32_t>(image, base + offsetof(Elf64_Shdr, sh_info), info);
    write_value_for_test<uint64_t>(image, base + offsetof(Elf64_Shdr, sh_addralign), addralign);
    write_value_for_test<uint64_t>(image, base + offsetof(Elf64_Shdr, sh_entsize), entsize);
  };
  write_shdr(1, rodata_name, SHT_PROGBITS, SHF_ALLOC, rodata_vaddr, rodata_offset, rodata_size, 0,
             0, 64, 0);
  write_shdr(2, text_name, SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR, text_vaddr, text_offset,
             text_size, 0, 0, 256, 0);
  write_shdr(3, symtab_name, SHT_SYMTAB, 0, 0, symtab_offset, sym_count * sizeof(Elf64_Sym), 4, 1,
             8, sizeof(Elf64_Sym));
  write_shdr(4, strtab_name, SHT_STRTAB, 0, 0, strtab_offset, strtab.size(), 0, 0, 1, 0);
  write_shdr(5, shstrtab_name, SHT_STRTAB, 0, 0, shstrtab_offset, shstrtab.size(), 0, 0, 1, 0);
  return image;
}
void enable_workgroup_id_x_sgpr(std::vector<uint8_t> &image) {
  using KD = rocr::llvm::amdhsa::kernel_descriptor_t;

  AmdGpuCodeObject layout(image.data(), image.size());
  ASSERT_TRUE(layout.is_valid());
  const auto *rodata = find_section(layout, ".rodata");
  ASSERT_NE(rodata, nullptr);
  ASSERT_GE(rodata->size(), sizeof(KD));

  KD kd{};
  std::memcpy(&kd, image.data() + rodata->sectionOffset(), sizeof(kd));
  kd.compute_pgm_rsrc2 |= rocr::llvm::amdhsa::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X;
  std::memcpy(image.data() + rodata->sectionOffset(), &kd, sizeof(kd));
}

} // namespace rocjitsu::test_support
