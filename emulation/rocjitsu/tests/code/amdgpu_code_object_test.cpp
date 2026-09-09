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
// sh_addr, and each .kd symbol's st_value is that descriptor's virtual address.
// Each descriptor's entry byte offset points into .text so the shared scanner
// (scan_kernel_descriptors) -- the discovery path min_kernel_sgpr_count() uses --
// accepts it. Sections: [1]=.text [2]=.rodata [3]=.strtab [4]=.symtab [5]=.shstrtab.
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
    KD desc = make_kd(kernels[i].second);
    const uint64_t kd_vaddr = kRodataAddr + i * sizeof(KD);
    // Point the entry into .text (one word per kernel). The offset is relative to
    // the descriptor's own vaddr and is signed, since .text sits below .rodata.
    const uint64_t entry_vaddr = kTextAddr + i * sizeof(uint32_t);
    desc.kernel_code_entry_byte_offset =
        static_cast<int64_t>(entry_vaddr) - static_cast<int64_t>(kd_vaddr);
    std::memcpy(rodata.data() + i * sizeof(KD), &desc, sizeof(KD));
    Elf64_Sym sym{};
    sym.st_name = add_elf_name(strtab, kernels[i].first + ".kd");
    sym.st_info = static_cast<uint8_t>((1u << 4) | kElfSymbolTypeObject); // global object
    sym.st_shndx = 2;                                                     // .rodata
    sym.st_value = kd_vaddr;
    sym.st_size = sizeof(KD);
    syms.push_back(sym);
  }

  const uint32_t text_word = 0xbf800000u; // s_nop 0
  const uint64_t text_offset = 0x100;
  // One word per kernel so every descriptor's entry lands inside .text.
  const uint64_t text_size = (kernels.empty() ? 1 : kernels.size()) * sizeof(text_word);
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

  std::memcpy(image.data() + text_offset, &text_word, sizeof(text_word));
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

void set_section_type_and_offset(std::vector<uint8_t> &image, size_t section_index, uint32_t type,
                                 uint64_t offset) {
  Elf64_Ehdr ehdr{};
  std::memcpy(&ehdr, image.data(), sizeof(ehdr));
  Elf64_Shdr shdr{};
  const size_t shdr_offset = ehdr.e_shoff + section_index * sizeof(Elf64_Shdr);
  std::memcpy(&shdr, image.data() + shdr_offset, sizeof(shdr));
  shdr.sh_type = type;
  shdr.sh_offset = offset;
  std::memcpy(image.data() + shdr_offset, &shdr, sizeof(shdr));
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

TEST(AmdGpuCodeObjectSections, RetainsAllocatedExecutableNobitsForLayoutValidation) {
  std::vector<uint8_t> image = make_elf_with_kds({});
  set_section_type_and_offset(image, 1, SHT_NOBITS, image.size() + 0x1000);

  AmdGpuCodeObject obj(image.data(), image.size());
  ASSERT_TRUE(obj.is_valid());
  EXPECT_TRUE(obj.text_sections().empty());
  ASSERT_EQ(obj.allocated_executable_sections().size(), 1u);
  const Section *section = obj.allocated_executable_sections().front();
  ASSERT_NE(section, nullptr);
  EXPECT_EQ(section->name(), ".text");
  EXPECT_EQ(section->sectionHeaderIndex(), std::optional<size_t>(1));
  EXPECT_EQ(section->data(), nullptr);
  EXPECT_TRUE(std::ranges::none_of(obj.all_sections(), [](const auto &candidate) {
    return candidate != nullptr && candidate->name() == ".text";
  }));
}

TEST(AmdGpuCodeObjectSections, ContinuesToSkipOrdinaryNobits) {
  std::vector<uint8_t> image = make_elf_with_kds({});
  set_section_type_and_offset(image, 2, SHT_NOBITS, image.size() + 0x1000);

  AmdGpuCodeObject obj(image.data(), image.size());
  ASSERT_TRUE(obj.is_valid());
  ASSERT_EQ(obj.text_sections().size(), 1u);
  EXPECT_TRUE(obj.rodata_sections().empty());
  EXPECT_TRUE(std::ranges::none_of(obj.all_sections(), [](const auto &section) {
    return section != nullptr && section->name() == ".rodata";
  }));
}

} // namespace
} // namespace rocjitsu
