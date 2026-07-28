// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file amdgpu_code_object_test.cpp
/// @brief Unit tests for AmdGpuCodeObject queries not covered by the target-id
///        tests -- currently min_kernel_sgpr_count(), which decodes each kernel
///        descriptor's GRANULATED_WAVEFRONT_SGPR_COUNT. The RDNA sentinel branch
///        is only reachable here (no RDNA hardware in CI).

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/shared/rdna_isa_base.h"

#include "hsa/AMDHSAKernelDescriptor.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rocjitsu {
namespace {

namespace kd = rocr::llvm::amdhsa;
using KD = kd::kernel_descriptor_t;

uint32_t add_elf_name(std::vector<uint8_t> &names, std::string_view name) {
  const uint32_t offset = static_cast<uint32_t>(names.size());
  names.insert(names.end(), name.begin(), name.end());
  names.push_back('\0');
  return offset;
}

uint64_t align_up(uint64_t value, uint64_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

// A 64-byte kernel descriptor whose wavefront SGPR granulation field is
// `granulated`; everything else zero.
KD make_kd(uint32_t granulated) {
  KD desc{};
  AMDHSA_BITS_SET(desc.compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                  granulated);
  return desc;
}

// A minimal gfx950 code object exporting one `<name>.kd` object symbol per entry
// in `kernels`, each pointing at a kernel descriptor with the given granulated
// SGPR count. The descriptors live in an SHF_ALLOC .rodata section with a real
// sh_addr, and each .kd symbol's st_value is that descriptor's virtual address,
// so min_kernel_sgpr_count() can locate and decode them via Section::vaddr().
// Sections: [1]=.text [2]=.rodata [3]=.strtab [4]=.symtab [5]=.shstrtab.
std::vector<uint8_t>
make_elf_with_kds(const std::vector<std::pair<std::string, uint32_t>> &kernels) {
  constexpr uint64_t kTextAddr = 0x1000;
  constexpr uint64_t kRodataAddr = 0x2000;

  std::vector<uint8_t> shstrtab{'\0'};
  const uint32_t text_name = add_elf_name(shstrtab, ".text");
  const uint32_t rodata_name = add_elf_name(shstrtab, ".rodata");
  const uint32_t strtab_name = add_elf_name(shstrtab, ".strtab");
  const uint32_t symtab_name = add_elf_name(shstrtab, ".symtab");
  const uint32_t shstrtab_name = add_elf_name(shstrtab, ".shstrtab");

  // .rodata holds one 64-byte KD per kernel; .symtab gets a matching `.kd`
  // symbol whose st_value is the KD's virtual address.
  std::vector<uint8_t> rodata(kernels.size() * sizeof(KD), 0);
  std::vector<uint8_t> strtab{'\0'};
  std::vector<Elf64_Sym> syms(1); // mandatory null symbol
  for (size_t i = 0; i < kernels.size(); ++i) {
    const KD desc = make_kd(kernels[i].second);
    std::memcpy(rodata.data() + i * sizeof(KD), &desc, sizeof(KD));
    Elf64_Sym sym{};
    sym.st_name = add_elf_name(strtab, kernels[i].first + ".kd");
    sym.st_info = static_cast<uint8_t>((1u << 4) | kElfSymbolTypeObject); // global object
    sym.st_shndx = 2;                                                     // .rodata
    sym.st_value = kRodataAddr + i * sizeof(KD);
    sym.st_size = sizeof(KD);
    syms.push_back(sym);
  }

  const uint32_t text_word = 0xbf800000u; // s_nop 0
  const uint64_t text_offset = 0x100;
  const uint64_t text_size = sizeof(text_word);
  const uint64_t rodata_offset = align_up(text_offset + text_size, 8);
  const uint64_t strtab_offset = rodata_offset + rodata.size();
  const uint64_t symtab_offset = align_up(strtab_offset + strtab.size(), 8);
  const uint64_t shstrtab_offset = symtab_offset + syms.size() * sizeof(Elf64_Sym);
  const uint64_t shoff = align_up(shstrtab_offset + shstrtab.size(), 8);
  constexpr uint16_t section_count = 6;

  std::vector<uint8_t> image(shoff + section_count * sizeof(Elf64_Shdr), 0);

  Elf64_Ehdr ehdr{};
  std::memcpy(ehdr.e_ident, EI_MAGIC, EI_MAGIC_SIZE);
  ehdr.e_ident[EI_CLASS] = ELFCLASS64;
  ehdr.e_ident[EI_OSABI] = ELFOSABI_AMDGPU_HSA;
  ehdr.e_type = ET_REL;
  ehdr.e_machine = EM_AMDGPU;
  ehdr.e_version = 1;
  ehdr.e_shoff = shoff;
  ehdr.e_flags = EF_AMDGPU_MACH_AMDGCN_GFX950;
  ehdr.e_ehsize = sizeof(Elf64_Ehdr);
  ehdr.e_shentsize = sizeof(Elf64_Shdr);
  ehdr.e_shnum = section_count;
  ehdr.e_shstrndx = 5;
  std::memcpy(image.data(), &ehdr, sizeof(ehdr));

  std::memcpy(image.data() + text_offset, &text_word, text_size);
  if (!rodata.empty())
    std::memcpy(image.data() + rodata_offset, rodata.data(), rodata.size());
  std::memcpy(image.data() + strtab_offset, strtab.data(), strtab.size());
  std::memcpy(image.data() + symtab_offset, syms.data(), syms.size() * sizeof(Elf64_Sym));
  std::memcpy(image.data() + shstrtab_offset, shstrtab.data(), shstrtab.size());

  std::array<Elf64_Shdr, section_count> shdrs{};
  shdrs[1].sh_name = text_name;
  shdrs[1].sh_type = SHT_PROGBITS;
  shdrs[1].sh_flags = SHF_ALLOC | SHF_EXECINSTR;
  shdrs[1].sh_addr = kTextAddr;
  shdrs[1].sh_offset = text_offset;
  shdrs[1].sh_size = text_size;
  shdrs[1].sh_addralign = sizeof(uint32_t);

  shdrs[2].sh_name = rodata_name;
  shdrs[2].sh_type = SHT_PROGBITS;
  shdrs[2].sh_flags = SHF_ALLOC;
  shdrs[2].sh_addr = kRodataAddr;
  shdrs[2].sh_offset = rodata_offset;
  shdrs[2].sh_size = rodata.size();
  shdrs[2].sh_addralign = 8;

  shdrs[3].sh_name = strtab_name;
  shdrs[3].sh_type = SHT_STRTAB;
  shdrs[3].sh_offset = strtab_offset;
  shdrs[3].sh_size = strtab.size();
  shdrs[3].sh_addralign = 1;

  shdrs[4].sh_name = symtab_name;
  shdrs[4].sh_type = SHT_SYMTAB;
  shdrs[4].sh_offset = symtab_offset;
  shdrs[4].sh_size = syms.size() * sizeof(Elf64_Sym);
  shdrs[4].sh_link = 3; // .strtab
  shdrs[4].sh_info = 1; // index of first global symbol
  shdrs[4].sh_entsize = sizeof(Elf64_Sym);
  shdrs[4].sh_addralign = 8;

  shdrs[5].sh_name = shstrtab_name;
  shdrs[5].sh_type = SHT_STRTAB;
  shdrs[5].sh_offset = shstrtab_offset;
  shdrs[5].sh_size = shstrtab.size();
  shdrs[5].sh_addralign = 1;

  std::memcpy(image.data() + shoff, shdrs.data(), shdrs.size() * sizeof(Elf64_Shdr));
  return image;
}

bool set_aggregate_copied_section_bytes(std::vector<uint8_t> &image, uint64_t aggregate_bytes) {
  Elf64_Ehdr header{};
  std::memcpy(&header, image.data(), sizeof(header));
  if (header.e_shnum <= 1)
    return false;

  std::vector<Elf64_Shdr> sections(header.e_shnum);
  std::memcpy(sections.data(), image.data() + header.e_shoff, sections.size() * sizeof(Elf64_Shdr));
  uint64_t other_bytes = 0;
  for (size_t i = 2; i < sections.size(); ++i) {
    if (sections[i].sh_type != SHT_NULL && sections[i].sh_type != SHT_NOBITS &&
        sections[i].sh_name != 0) {
      other_bytes += sections[i].sh_size;
    }
  }
  if (aggregate_bytes < other_bytes || aggregate_bytes - other_bytes > image.size())
    return false;
  sections[1].sh_offset = 0;
  sections[1].sh_size = aggregate_bytes - other_bytes;
  std::memcpy(image.data() + header.e_shoff, sections.data(), sections.size() * sizeof(Elf64_Shdr));
  return true;
}

enum class SymbolNameBudget {
  AtBoundary,
  OneByteOver,
};

enum class RetainedSymbolKind {
  Functions,
  KernelDescriptors,
  KernelsAndFunctions,
};

struct SymbolBoundaryOptions {
  SymbolNameBudget budget = SymbolNameBudget::AtBoundary;
  RetainedSymbolKind retained_kind = RetainedSymbolKind::Functions;
  uint8_t symbol_type = kElfSymbolTypeFunc;
  uint16_t symbol_section_index = 1;
  size_t short_retained_name_count = 0;
  bool duplicate_symbol_table = false;
  bool include_dynamic_stack_symbol = false;
  bool dynamic_stack_symbol_first = false;
  uint32_t code_section_type = SHT_PROGBITS;
  std::string_view code_section_name = ".text";
  bool code_section_offset_out_of_image = false;
};

std::vector<uint8_t>
make_elf_at_retained_symbol_state_boundary(const SymbolBoundaryOptions &options = {}) {
  constexpr size_t kImageBytes = 4098;
  constexpr uint64_t kTextAddress = 0x1000;
  const bool retain_functions = options.retained_kind != RetainedSymbolKind::KernelDescriptors;
  const bool retain_kernels = options.retained_kind != RetainedSymbolKind::Functions;
  const size_t retained_name_count =
      options.short_retained_name_count != 0                   ? options.short_retained_name_count
      : options.retained_kind == RetainedSymbolKind::Functions ? 48
      : options.retained_kind == RetainedSymbolKind::KernelDescriptors ? 34
                                                                       : 21;
  const uint64_t function_entry_charge = options.include_dynamic_stack_symbol
                                             ? kAmdGpuCodeObjectFunctionAndTransientEntryChargeBytes
                                             : kAmdGpuCodeObjectFunctionEntryChargeBytes;
  const uint64_t kernel_entry_charge = options.include_dynamic_stack_symbol && !retain_functions
                                           ? kAmdGpuCodeObjectKernelAndTransientEntryChargeBytes
                                           : kAmdGpuCodeObjectKernelEntryChargeBytes;
  const uint64_t per_name_entry_charge =
      (retain_functions ? function_entry_charge : 0) + (retain_kernels ? kernel_entry_charge : 0);
  const uint64_t charged_name_copies = retain_functions + 2 * retain_kernels;
  constexpr uint64_t kSymbolStateBudgetBytes =
      kAmdGpuCodeObjectRetainedSymbolStateImageUnits * kImageBytes;
  size_t base_name_length = 1;
  size_t last_name_extra = 0;
  if (options.short_retained_name_count == 0) {
    const uint64_t fixed_entry_charge = retained_name_count * per_name_entry_charge;
    if (fixed_entry_charge >= kSymbolStateBudgetBytes ||
        (kSymbolStateBudgetBytes - fixed_entry_charge) % charged_name_copies != 0) {
      return {};
    }
    size_t aggregate_name_bytes =
        static_cast<size_t>((kSymbolStateBudgetBytes - fixed_entry_charge) / charged_name_copies);
    if (options.budget == SymbolNameBudget::OneByteOver)
      ++aggregate_name_bytes;
    const size_t distinct_length_delta = retained_name_count * (retained_name_count - 1) / 2;
    if (aggregate_name_bytes <= distinct_length_delta)
      return {};
    base_name_length = (aggregate_name_bytes - distinct_length_delta) / retained_name_count;
    last_name_extra = (aggregate_name_bytes - distinct_length_delta) % retained_name_count;
  } else if (retained_name_count > 64) {
    return {};
  }

  std::vector<uint8_t> shstrtab{'\0'};
  const uint32_t text_name = add_elf_name(shstrtab, options.code_section_name);
  const uint32_t strtab_name = add_elf_name(shstrtab, ".strtab");
  const uint32_t symtab_name = add_elf_name(shstrtab, ".symtab");
  const uint32_t dynsym_name = add_elf_name(shstrtab, ".dynsym");
  const uint32_t shstrtab_name = add_elf_name(shstrtab, ".shstrtab");

  std::vector<uint8_t> strtab{'\0'};
  std::vector<Elf64_Sym> symbols;
  const auto append_symbol = [&](std::string name, uint8_t symbol_type) {
    Elf64_Sym symbol{};
    symbol.st_name = add_elf_name(strtab, name);
    symbol.st_info = static_cast<uint8_t>((1u << 4) | symbol_type);
    symbol.st_shndx = options.symbol_section_index;
    symbol.st_value = kTextAddress;
    symbol.st_size = sizeof(uint32_t);
    symbols.push_back(symbol);
  };
  const auto append_dynamic_stack_symbol = [&](const std::string &name) {
    Elf64_Sym symbol{};
    symbol.st_name = add_elf_name(strtab, name + ".has_dyn_sized_stack");
    symbol.st_info = static_cast<uint8_t>((1u << 4) | kElfSymbolTypeNone);
    symbol.st_shndx = SHN_ABS;
    symbol.st_value = 1;
    symbols.push_back(symbol);
  };
  for (size_t i = 0; i < retained_name_count; ++i) {
    size_t name_length =
        options.short_retained_name_count == 0 ? base_name_length + i : base_name_length;
    if (options.short_retained_name_count == 0 && i + 1 == retained_name_count)
      name_length += last_name_extra;
    std::string name(name_length, 'x');
    name.front() = static_cast<char>('a' + i);
    if (options.include_dynamic_stack_symbol && options.dynamic_stack_symbol_first)
      append_dynamic_stack_symbol(name);
    if (retain_functions)
      append_symbol(name, options.symbol_type);
    if (retain_kernels)
      append_symbol(name + ".kd", kElfSymbolTypeObject);
    if (options.include_dynamic_stack_symbol && !options.dynamic_stack_symbol_first)
      append_dynamic_stack_symbol(name);
  }

  constexpr uint64_t text_offset = 0x100;
  const uint64_t strtab_offset = text_offset + sizeof(uint32_t);
  const uint64_t symtab_offset = align_up(strtab_offset + strtab.size(), 8);
  const uint64_t symbol_bytes = symbols.size() * sizeof(Elf64_Sym);
  const uint64_t shstrtab_offset = symtab_offset + symbol_bytes;
  const uint64_t shoff = align_up(shstrtab_offset + shstrtab.size(), 8);
  constexpr uint16_t section_count = 6;
  if (shoff + section_count * sizeof(Elf64_Shdr) > kImageBytes)
    return {};

  std::vector<uint8_t> image(kImageBytes, 0);
  Elf64_Ehdr header{};
  std::memcpy(header.e_ident, EI_MAGIC, EI_MAGIC_SIZE);
  header.e_ident[EI_CLASS] = ELFCLASS64;
  header.e_ident[EI_OSABI] = ELFOSABI_AMDGPU_HSA;
  header.e_type = ET_REL;
  header.e_machine = EM_AMDGPU;
  header.e_version = 1;
  header.e_shoff = shoff;
  header.e_flags = EF_AMDGPU_MACH_AMDGCN_GFX950;
  header.e_ehsize = sizeof(Elf64_Ehdr);
  header.e_shentsize = sizeof(Elf64_Shdr);
  header.e_shnum = section_count;
  header.e_shstrndx = 5;
  std::memcpy(image.data(), &header, sizeof(header));

  const uint32_t text_word = 0xbf800000u;
  std::memcpy(image.data() + text_offset, &text_word, sizeof(text_word));
  std::memcpy(image.data() + strtab_offset, strtab.data(), strtab.size());
  std::memcpy(image.data() + symtab_offset, symbols.data(), symbol_bytes);
  std::memcpy(image.data() + shstrtab_offset, shstrtab.data(), shstrtab.size());

  std::array<Elf64_Shdr, section_count> sections{};
  sections[1].sh_name = text_name;
  sections[1].sh_type = options.code_section_type;
  sections[1].sh_flags = SHF_ALLOC | SHF_EXECINSTR;
  sections[1].sh_addr = kTextAddress;
  sections[1].sh_offset = options.code_section_offset_out_of_image ? image.size() + 1 : text_offset;
  sections[1].sh_size = sizeof(text_word);
  sections[1].sh_addralign = alignof(uint32_t);

  sections[2].sh_name = strtab_name;
  sections[2].sh_type = SHT_STRTAB;
  sections[2].sh_offset = strtab_offset;
  sections[2].sh_size = strtab.size();
  sections[2].sh_addralign = 1;

  sections[3].sh_name = symtab_name;
  sections[3].sh_type = SHT_SYMTAB;
  sections[3].sh_offset = symtab_offset;
  sections[3].sh_size = symbol_bytes;
  sections[3].sh_link = 2;
  sections[3].sh_entsize = sizeof(Elf64_Sym);
  sections[3].sh_addralign = alignof(Elf64_Sym);

  if (options.duplicate_symbol_table) {
    sections[4] = sections[3];
    sections[4].sh_name = dynsym_name;
    sections[4].sh_type = SHT_DYNSYM;
  }

  sections[5].sh_name = shstrtab_name;
  sections[5].sh_type = SHT_STRTAB;
  sections[5].sh_offset = shstrtab_offset;
  sections[5].sh_size = shstrtab.size();
  sections[5].sh_addralign = 1;
  std::memcpy(image.data() + shoff, sections.data(), sizeof(sections));
  return image;
}

std::vector<uint8_t> make_zero_sized_section_dense_elf(size_t section_count) {
  if (section_count < 3 || section_count > std::numeric_limits<uint16_t>::max())
    return {};

  std::vector<uint8_t> shstrtab{'\0'};
  const uint32_t text_name = add_elf_name(shstrtab, ".text");
  const uint32_t rodata_name = add_elf_name(shstrtab, ".rodata");
  const uint32_t shstrtab_name = add_elf_name(shstrtab, ".shstrtab");
  const uint64_t shoff = sizeof(Elf64_Ehdr);
  const uint64_t shstrtab_offset = shoff + section_count * sizeof(Elf64_Shdr);
  std::vector<uint8_t> image(shstrtab_offset + shstrtab.size(), 0);

  Elf64_Ehdr header{};
  std::memcpy(header.e_ident, EI_MAGIC, EI_MAGIC_SIZE);
  header.e_ident[EI_CLASS] = ELFCLASS64;
  header.e_ident[EI_OSABI] = ELFOSABI_AMDGPU_HSA;
  header.e_type = ET_REL;
  header.e_machine = EM_AMDGPU;
  header.e_version = 1;
  header.e_shoff = shoff;
  header.e_flags = EF_AMDGPU_MACH_AMDGCN_GFX950;
  header.e_ehsize = sizeof(Elf64_Ehdr);
  header.e_shentsize = sizeof(Elf64_Shdr);
  header.e_shnum = static_cast<uint16_t>(section_count);
  header.e_shstrndx = static_cast<uint16_t>(section_count - 1);
  std::memcpy(image.data(), &header, sizeof(header));

  std::vector<Elf64_Shdr> sections(section_count);
  for (size_t i = 1; i + 1 < section_count; ++i) {
    sections[i].sh_name = i % 2 == 0 ? text_name : rodata_name;
    sections[i].sh_type = SHT_PROGBITS;
  }
  sections.back().sh_name = shstrtab_name;
  sections.back().sh_type = SHT_STRTAB;
  sections.back().sh_offset = shstrtab_offset;
  sections.back().sh_size = shstrtab.size();
  sections.back().sh_addralign = 1;
  std::memcpy(image.data() + shoff, sections.data(), sections.size() * sizeof(Elf64_Shdr));
  std::memcpy(image.data() + shstrtab_offset, shstrtab.data(), shstrtab.size());
  return image;
}

// CDNA: a granulated field of 0 encodes a real 8-SGPR allocation.
TEST(AmdGpuCodeObjectSgpr, CdnaGranulatedZeroIsEightSgprs) {
  const auto image = make_elf_with_kds({{"k", 0}});
  AmdGpuCodeObject obj(image.data(), image.size());
  ASSERT_TRUE(obj.is_valid());
  const auto count = obj.min_kernel_sgpr_count(ROCJITSU_CODE_ARCH_CDNA2);
  ASSERT_TRUE(count.has_value());
  EXPECT_EQ(*count, 8u);
}

// CDNA: (granulated + 1) * 8. granulated 3 -> 32, exactly enough to own s[30:31].
TEST(AmdGpuCodeObjectSgpr, CdnaGranulatedDecodesTimesEight) {
  const auto image = make_elf_with_kds({{"k", 3}});
  AmdGpuCodeObject obj(image.data(), image.size());
  ASSERT_TRUE(obj.is_valid());
  EXPECT_EQ(obj.min_kernel_sgpr_count(ROCJITSU_CODE_ARCH_CDNA2), std::optional<uint32_t>(32u));
}

// RDNA: a granulated field of 0 is a sentinel; the wave owns the fixed per-wave
// SGPR pool, not 8. This branch is unreachable on CI hardware, so this unit test
// is the only coverage for it.
TEST(AmdGpuCodeObjectSgpr, RdnaGranulatedZeroIsFixedPool) {
  const auto image = make_elf_with_kds({{"k", 0}});
  AmdGpuCodeObject obj(image.data(), image.size());
  ASSERT_TRUE(obj.is_valid());
  const auto count = obj.min_kernel_sgpr_count(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(count.has_value());
  EXPECT_EQ(*count, amdgpu::RdnaIsaBase::MAX_SGPRS_PER_WF);
  EXPECT_GE(*count, 32u); // so the fixed link pair s[30:31] always fits on RDNA
}

// Without an anchor->kernel map, the smallest kernel bounds every anchor.
TEST(AmdGpuCodeObjectSgpr, ReturnsMinAcrossKernels) {
  const auto image = make_elf_with_kds({{"big", 7}, {"small", 0}}); // 64 and 8
  AmdGpuCodeObject obj(image.data(), image.size());
  ASSERT_TRUE(obj.is_valid());
  EXPECT_EQ(obj.min_kernel_sgpr_count(ROCJITSU_CODE_ARCH_CDNA2), std::optional<uint32_t>(8u));
}

// No kernel descriptor -> nullopt, so the caller falls back permissively.
TEST(AmdGpuCodeObjectSgpr, NoKernelDescriptorReturnsNullopt) {
  const auto image = make_elf_with_kds({}); // no .kd symbols
  AmdGpuCodeObject obj(image.data(), image.size());
  ASSERT_TRUE(obj.is_valid());
  EXPECT_FALSE(obj.min_kernel_sgpr_count(ROCJITSU_CODE_ARCH_CDNA2).has_value());
}

TEST(AmdGpuCodeObjectValidation, RejectsAggregateCopiedSectionsLargerThanImage) {
  auto image = make_elf_with_kds({{"k", 0}});
  Elf64_Ehdr header{};
  std::memcpy(&header, image.data(), sizeof(header));

  // Each range is individually in bounds, but copying both overlapping
  // sections would retain twice the full input image.
  for (const size_t section_index : {size_t{1}, size_t{2}}) {
    const size_t header_offset = header.e_shoff + section_index * sizeof(Elf64_Shdr);
    Elf64_Shdr section{};
    std::memcpy(&section, image.data() + header_offset, sizeof(section));
    section.sh_offset = 0;
    section.sh_size = image.size();
    std::memcpy(image.data() + header_offset, &section, sizeof(section));
  }

  AmdGpuCodeObject obj(image.data(), image.size());
  EXPECT_FALSE(obj.is_valid());
}

TEST(AmdGpuCodeObjectValidation, AcceptsAggregateCopiedSectionsEqualToImage) {
  auto image = make_elf_with_kds({{"k", 0}});
  ASSERT_TRUE(set_aggregate_copied_section_bytes(image, image.size()));

  AmdGpuCodeObject obj(image.data(), image.size());
  EXPECT_TRUE(obj.is_valid());
}

TEST(AmdGpuCodeObjectValidation, RejectsAggregateCopiedSectionsOneByteOverImage) {
  auto image = make_elf_with_kds({{"k", 0}});
  ASSERT_TRUE(set_aggregate_copied_section_bytes(image, image.size() + 1));

  AmdGpuCodeObject obj(image.data(), image.size());
  EXPECT_FALSE(obj.is_valid());
}

TEST(AmdGpuCodeObjectValidation, RejectsAggregateSectionNameAmplificationBeforeAllocation) {
  constexpr size_t kImageBytes = 4096;
  constexpr size_t kSectionCount = 32;
  constexpr size_t kSectionTableOffset = sizeof(Elf64_Ehdr);
  constexpr size_t kStringTableOffset = kSectionTableOffset + kSectionCount * sizeof(Elf64_Shdr);
  static_assert(kStringTableOffset < kImageBytes);

  const auto seed = make_elf_with_kds({});
  Elf64_Ehdr header{};
  std::memcpy(&header, seed.data(), sizeof(header));
  header.e_shoff = kSectionTableOffset;
  header.e_shnum = kSectionCount;
  header.e_shstrndx = 0;

  std::vector<uint8_t> image(kImageBytes, 0);
  std::memcpy(image.data(), &header, sizeof(header));
  std::fill(image.begin() + kStringTableOffset, image.end(), static_cast<uint8_t>('x'));

  std::array<Elf64_Shdr, kSectionCount> sections{};
  sections[0].sh_type = SHT_STRTAB;
  sections[0].sh_offset = kStringTableOffset;
  sections[0].sh_size = kImageBytes - kStringTableOffset;
  for (Elf64_Shdr &section : sections)
    section.sh_name = 1;
  std::memcpy(image.data() + kSectionTableOffset, sections.data(), sizeof(sections));

  AmdGpuCodeObject obj(image.data(), image.size());
  EXPECT_FALSE(obj.is_valid());
}

TEST(AmdGpuCodeObjectValidation, AcceptsDenseZeroSizedSectionCollectionsAtLayoutBound) {
  constexpr size_t kSectionCount = 512;
  const auto image = make_zero_sized_section_dense_elf(kSectionCount);
  AmdGpuCodeObject obj(image.data(), image.size());

  ASSERT_TRUE(obj.is_valid());
  EXPECT_EQ(obj.all_sections().size(), kSectionCount - 1);
  EXPECT_EQ(obj.text_sections().size(), (kSectionCount - 2) / 2);
}

TEST(AmdGpuCodeObjectValidation, AcceptsRetainedFunctionStateEqualToBudget) {
  const auto image = make_elf_at_retained_symbol_state_boundary();
  AmdGpuCodeObject obj(image.data(), image.size());
  EXPECT_TRUE(obj.is_valid());
  EXPECT_EQ(obj.functions().size(), 48u);
}

TEST(AmdGpuCodeObjectValidation, RejectsRetainedFunctionStateOverBudget) {
  const auto image = make_elf_at_retained_symbol_state_boundary({
      .budget = SymbolNameBudget::OneByteOver,
  });
  AmdGpuCodeObject obj(image.data(), image.size());
  EXPECT_FALSE(obj.is_valid());
}

TEST(AmdGpuCodeObjectValidation, RejectsManyShortRetainedFunctionEntries) {
  constexpr size_t kEntryCount = kAmdGpuCodeObjectRetainedSymbolStateImageUnits * 4098 /
                                     kAmdGpuCodeObjectFunctionEntryChargeBytes +
                                 1;
  static_assert(kEntryCount <= 64);
  const auto image = make_elf_at_retained_symbol_state_boundary({
      .short_retained_name_count = kEntryCount,
  });
  AmdGpuCodeObject obj(image.data(), image.size());
  EXPECT_FALSE(obj.is_valid());
}

TEST(AmdGpuCodeObjectValidation, IgnoresUnretainedNamesAboveRetainedNameBudget) {
  const auto image = make_elf_at_retained_symbol_state_boundary({
      .budget = SymbolNameBudget::OneByteOver,
      .symbol_type = kElfSymbolTypeNone,
      .symbol_section_index = SHN_UNDEF,
  });
  AmdGpuCodeObject obj(image.data(), image.size());
  EXPECT_TRUE(obj.is_valid());
  EXPECT_TRUE(obj.functions().empty());
  EXPECT_TRUE(obj.kernels().empty());
}

TEST(AmdGpuCodeObjectValidation, DeduplicatesRetainedNamesAcrossSymbolTables) {
  const auto image = make_elf_at_retained_symbol_state_boundary({
      .duplicate_symbol_table = true,
  });
  AmdGpuCodeObject obj(image.data(), image.size());
  EXPECT_TRUE(obj.is_valid());
  EXPECT_EQ(obj.functions().size(), 48u);
}

TEST(AmdGpuCodeObjectValidation, AcceptsRetainedKernelStateEqualToBudget) {
  const auto image = make_elf_at_retained_symbol_state_boundary({
      .retained_kind = RetainedSymbolKind::KernelDescriptors,
  });
  AmdGpuCodeObject obj(image.data(), image.size());
  EXPECT_TRUE(obj.is_valid());
  EXPECT_EQ(obj.kernels().size(), 34u);
}

TEST(AmdGpuCodeObjectValidation, RejectsRetainedKernelStateOverBudget) {
  const auto image = make_elf_at_retained_symbol_state_boundary({
      .budget = SymbolNameBudget::OneByteOver,
      .retained_kind = RetainedSymbolKind::KernelDescriptors,
  });
  AmdGpuCodeObject obj(image.data(), image.size());
  EXPECT_FALSE(obj.is_valid());
}

TEST(AmdGpuCodeObjectValidation, RejectsManyShortRetainedKernelEntries) {
  constexpr size_t kEntryCount = kAmdGpuCodeObjectRetainedSymbolStateImageUnits * 4098 /
                                     kAmdGpuCodeObjectKernelEntryChargeBytes +
                                 1;
  static_assert(kEntryCount <= 64);
  const auto image = make_elf_at_retained_symbol_state_boundary({
      .retained_kind = RetainedSymbolKind::KernelDescriptors,
      .short_retained_name_count = kEntryCount,
  });
  AmdGpuCodeObject obj(image.data(), image.size());
  EXPECT_FALSE(obj.is_valid());
}

TEST(AmdGpuCodeObjectValidation, DeduplicatesRetainedKernelStateAcrossSymbolTables) {
  const auto image = make_elf_at_retained_symbol_state_boundary({
      .retained_kind = RetainedSymbolKind::KernelDescriptors,
      .duplicate_symbol_table = true,
  });
  AmdGpuCodeObject obj(image.data(), image.size());
  EXPECT_TRUE(obj.is_valid());
  EXPECT_EQ(obj.kernels().size(), 34u);
}

TEST(AmdGpuCodeObjectValidation, ChargesKernelAndFunctionRetentionIndependently) {
  const auto image = make_elf_at_retained_symbol_state_boundary({
      .retained_kind = RetainedSymbolKind::KernelsAndFunctions,
  });
  AmdGpuCodeObject obj(image.data(), image.size());
  EXPECT_TRUE(obj.is_valid());
  EXPECT_EQ(obj.kernels().size(), 21u);
  EXPECT_EQ(obj.functions().size(), 21u);
}

TEST(AmdGpuCodeObjectValidation, RejectsCombinedKernelAndFunctionStateOverBudget) {
  const auto image = make_elf_at_retained_symbol_state_boundary({
      .budget = SymbolNameBudget::OneByteOver,
      .retained_kind = RetainedSymbolKind::KernelsAndFunctions,
  });
  AmdGpuCodeObject obj(image.data(), image.size());
  EXPECT_FALSE(obj.is_valid());
}

TEST(AmdGpuCodeObjectValidation, AccountsSharedDynamicStackStateIndependentOfSymbolOrder) {
  for (const bool dynamic_stack_symbol_first : {false, true}) {
    const auto image = make_elf_at_retained_symbol_state_boundary({
        .retained_kind = RetainedSymbolKind::KernelsAndFunctions,
        .short_retained_name_count = 1,
        .include_dynamic_stack_symbol = true,
        .dynamic_stack_symbol_first = dynamic_stack_symbol_first,
    });
    AmdGpuCodeObject obj(image.data(), image.size());

    ASSERT_TRUE(obj.is_valid()) << "dynamic_stack_symbol_first=" << dynamic_stack_symbol_first;
    ASSERT_EQ(obj.kernels().size(), 1u);
    EXPECT_EQ(obj.kernels().front().uses_dynamic_stack, std::optional<bool>(true));
    EXPECT_EQ(obj.functions().size(), 1u);
  }
}

TEST(AmdGpuCodeObjectValidation, DoesNotReportFunctionsFromNoBitsTextSection) {
  const auto image = make_elf_at_retained_symbol_state_boundary({
      .code_section_type = SHT_NOBITS,
      .code_section_offset_out_of_image = true,
  });
  AmdGpuCodeObject obj(image.data(), image.size());
  ASSERT_TRUE(obj.is_valid());
  EXPECT_TRUE(obj.functions().empty());
  EXPECT_TRUE(obj.text_sections().empty());
}

TEST(AmdGpuCodeObjectValidation, DoesNotTreatOtherNoBitsSectionAsText) {
  const auto image = make_elf_at_retained_symbol_state_boundary({
      .code_section_type = SHT_NOBITS,
      .code_section_name = ".bss",
      .code_section_offset_out_of_image = true,
  });
  AmdGpuCodeObject obj(image.data(), image.size());
  ASSERT_TRUE(obj.is_valid());
  EXPECT_TRUE(obj.functions().empty());
  EXPECT_TRUE(obj.text_sections().empty());
}

} // namespace
} // namespace rocjitsu
