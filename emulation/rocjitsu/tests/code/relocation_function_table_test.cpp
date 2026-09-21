// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "decode_test_util.h"
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/relocation_function_table.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/machine_insts.h"
#include "rocjitsu/isa/decoder.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <span>
#include <string_view>
#include <vector>

namespace rocjitsu {
namespace {

uint32_t add_name(std::vector<uint8_t> &names, std::string_view name) {
  const uint32_t offset = static_cast<uint32_t>(names.size());
  names.insert(names.end(), name.begin(), name.end());
  names.push_back('\0');
  return offset;
}

uint64_t align_up(uint64_t value, uint64_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

uint64_t relocation_info(uint32_t symbol, uint32_t type) {
  return (static_cast<uint64_t>(symbol) << 32) | type;
}

std::vector<uint8_t> make_relocation_function_table_elf(std::span<const uint32_t> text_words = {}) {
  constexpr uint64_t kTextVaddr = 0x1000;
  constexpr uint64_t kTableVaddr = 0x2000;
  constexpr uint64_t kGotVaddr = 0x3000;
  constexpr uint64_t kTextOffset = 0x100;
  const std::array<uint32_t, 4> default_text = {0xbf800000u, 0xbf800000u, 0xbf800000u, 0xbf800000u};
  if (text_words.empty())
    text_words = default_text;
  const uint64_t text_size = text_words.size() * sizeof(uint32_t);
  constexpr uint64_t kTableSize = 24;
  constexpr uint64_t kGotSize = 8;

  std::vector<uint8_t> shstrtab{'\0'};
  const uint32_t text_name = add_name(shstrtab, ".text");
  const uint32_t data_name = add_name(shstrtab, ".data.rel.ro");
  const uint32_t got_name = add_name(shstrtab, ".got");
  const uint32_t dynsym_name = add_name(shstrtab, ".dynsym");
  const uint32_t rela_name = add_name(shstrtab, ".rela.dyn");
  const uint32_t shstrtab_name = add_name(shstrtab, ".shstrtab");

  const uint64_t data_offset = kTextOffset + text_size;
  const uint64_t got_offset = data_offset + kTableSize;
  const uint64_t dynsym_offset = align_up(got_offset + kGotSize, 8);
  constexpr size_t kSymbolCount = 2;
  const uint64_t rela_offset = dynsym_offset + kSymbolCount * sizeof(Elf64_Sym);
  constexpr size_t kRelaCount = 3;
  const uint64_t shstrtab_offset = rela_offset + kRelaCount * sizeof(Elf64_Rela);
  const uint64_t shoff = align_up(shstrtab_offset + shstrtab.size(), 8);
  constexpr uint16_t kSectionCount = 7;
  std::vector<uint8_t> image(shoff + kSectionCount * sizeof(Elf64_Shdr), 0);

  Elf64_Ehdr ehdr{};
  std::memcpy(ehdr.e_ident, EI_MAGIC, EI_MAGIC_SIZE);
  ehdr.e_ident[EI_CLASS] = ELFCLASS64;
  ehdr.e_ident[EI_OSABI] = ELFOSABI_AMDGPU_HSA;
  ehdr.e_type = ET_DYN;
  ehdr.e_machine = EM_AMDGPU;
  ehdr.e_version = 1;
  ehdr.e_shoff = shoff;
  ehdr.e_flags = EF_AMDGPU_MACH_AMDGCN_GFX1250;
  ehdr.e_ehsize = sizeof(Elf64_Ehdr);
  ehdr.e_shentsize = sizeof(Elf64_Shdr);
  ehdr.e_shnum = kSectionCount;
  ehdr.e_shstrndx = 6;
  std::memcpy(image.data(), &ehdr, sizeof(ehdr));

  std::memcpy(image.data() + kTextOffset, text_words.data(), text_size);

  std::array<Elf64_Sym, kSymbolCount> symbols{};
  symbols[1].st_info = elf_symbol_info(kElfSymbolBindGlobal, kElfSymbolTypeObject);
  symbols[1].st_shndx = 2;
  symbols[1].st_value = kTableVaddr;
  symbols[1].st_size = kTableSize;
  std::memcpy(image.data() + dynsym_offset, symbols.data(), sizeof(symbols));

  std::array<Elf64_Rela, kRelaCount> relas{};
  relas[0].r_offset = kGotVaddr;
  relas[0].r_info = relocation_info(1, R_AMDGPU_ABS64);
  relas[1].r_offset = kTableVaddr;
  relas[1].r_info = relocation_info(0, R_AMDGPU_RELATIVE64);
  relas[1].r_addend = kTextVaddr + 4;
  relas[2].r_offset = kTableVaddr + 16; // middle slot is intentionally null.
  relas[2].r_info = relocation_info(0, R_AMDGPU_RELATIVE64);
  relas[2].r_addend = kTextVaddr + std::min<uint64_t>(12, text_size - sizeof(uint32_t));
  std::memcpy(image.data() + rela_offset, relas.data(), sizeof(relas));
  std::memcpy(image.data() + shstrtab_offset, shstrtab.data(), shstrtab.size());

  std::array<Elf64_Shdr, kSectionCount> shdrs{};
  shdrs[1] = {.sh_name = text_name,
              .sh_type = SHT_PROGBITS,
              .sh_flags = SHF_ALLOC | SHF_EXECINSTR,
              .sh_addr = kTextVaddr,
              .sh_offset = kTextOffset,
              .sh_size = text_size,
              .sh_link = 0,
              .sh_info = 0,
              .sh_addralign = 4,
              .sh_entsize = 0};
  shdrs[2] = {.sh_name = data_name,
              .sh_type = SHT_PROGBITS,
              .sh_flags = SHF_ALLOC,
              .sh_addr = kTableVaddr,
              .sh_offset = data_offset,
              .sh_size = kTableSize,
              .sh_link = 0,
              .sh_info = 0,
              .sh_addralign = 8,
              .sh_entsize = 0};
  shdrs[3] = {.sh_name = got_name,
              .sh_type = SHT_PROGBITS,
              .sh_flags = SHF_ALLOC,
              .sh_addr = kGotVaddr,
              .sh_offset = got_offset,
              .sh_size = kGotSize,
              .sh_link = 0,
              .sh_info = 0,
              .sh_addralign = 8,
              .sh_entsize = 0};
  shdrs[4] = {.sh_name = dynsym_name,
              .sh_type = SHT_DYNSYM,
              .sh_flags = 0,
              .sh_addr = 0,
              .sh_offset = dynsym_offset,
              .sh_size = sizeof(symbols),
              .sh_link = 0,
              .sh_info = 1,
              .sh_addralign = 8,
              .sh_entsize = sizeof(Elf64_Sym)};
  shdrs[5] = {.sh_name = rela_name,
              .sh_type = SHT_RELA,
              .sh_flags = 0,
              .sh_addr = 0,
              .sh_offset = rela_offset,
              .sh_size = sizeof(relas),
              .sh_link = 4,
              .sh_info = 0,
              .sh_addralign = 8,
              .sh_entsize = sizeof(Elf64_Rela)};
  shdrs[6] = {.sh_name = shstrtab_name,
              .sh_type = SHT_STRTAB,
              .sh_flags = 0,
              .sh_addr = 0,
              .sh_offset = shstrtab_offset,
              .sh_size = shstrtab.size(),
              .sh_link = 0,
              .sh_info = 0,
              .sh_addralign = 1,
              .sh_entsize = 0};
  std::memcpy(image.data() + shoff, shdrs.data(), sizeof(shdrs));
  return image;
}

std::vector<uint32_t> make_table_dispatch_text() {
  std::vector<uint32_t> words;
  auto append32 = [&](uint32_t word) { words.push_back(word); };
  auto append64 = [&](uint64_t encoding) {
    words.push_back(static_cast<uint32_t>(encoding));
    words.push_back(static_cast<uint32_t>(encoding >> 32));
  };

  auto getpc = std::bit_cast<cdna5::Sop1MachineInst>(0xbe804700u);
  getpc.sdst = 0;
  append32(std::bit_cast<uint32_t>(getpc));

  auto add = std::bit_cast<cdna5::Sop2MachineInst>(0xa9800000u);
  add.sdst = 0;
  add.ssrc0 = 0;
  add.ssrc1 = 254;
  append32(std::bit_cast<uint32_t>(add));
  append64(0x3000u - 0x1004u);

  auto got_load = std::bit_cast<cdna5::SmemMachineInst>(uint64_t{0xf4002000u});
  got_load.sbase = 0;
  got_load.sdata = 2;
  append64(std::bit_cast<uint64_t>(got_load));

  auto table_load = got_load;
  table_load.sbase = 1; // encoded in SGPR-pair units: s[2:3].
  table_load.sdata = 4;
  table_load.scale_offset = 1;
  table_load.soffset = 6;
  append64(std::bit_cast<uint64_t>(table_load));

  auto swap = std::bit_cast<cdna5::Sop1MachineInst>(0xbe804900u);
  swap.sdst = 30;
  swap.ssrc0 = 4;
  append32(std::bit_cast<uint32_t>(swap));
  append32(0xbfb00000u); // s_endpgm continuation

  auto setpc = std::bit_cast<cdna5::Sop1MachineInst>(0xbe804800u);
  setpc.ssrc0 = 30;
  append32(std::bit_cast<uint32_t>(setpc));
  return words;
}

// Direct indexed dispatch that reaches the table base by adding @p add_literal to
// the getpc result. Callers pick the literal to match whatever text base the
// analysis is told to assume, including a two's-complement negative value when the
// table sits below .text.
std::vector<uint32_t> make_direct_table_dispatch_text_with_addend(uint64_t add_literal) {
  std::vector<uint32_t> words;
  auto append32 = [&](uint32_t word) { words.push_back(word); };
  auto append64 = [&](uint64_t encoding) {
    words.push_back(static_cast<uint32_t>(encoding));
    words.push_back(static_cast<uint32_t>(encoding >> 32));
  };

  auto getpc = std::bit_cast<cdna5::Sop1MachineInst>(0xbe804700u);
  getpc.sdst = 54;
  append32(std::bit_cast<uint32_t>(getpc));

  auto add = std::bit_cast<cdna5::Sop2MachineInst>(0xa9800000u);
  add.sdst = 54;
  add.ssrc0 = 54;
  add.ssrc1 = 254;
  append32(std::bit_cast<uint32_t>(add));
  append64(add_literal);

  auto table_load = std::bit_cast<cdna5::SmemMachineInst>(uint64_t{0xf4002000u});
  table_load.sbase = 27; // encoded in SGPR-pair units: s[54:55].
  table_load.sdata = 0;
  table_load.scale_offset = 1;
  table_load.soffset = 2;
  append64(std::bit_cast<uint64_t>(table_load));

  auto swap = std::bit_cast<cdna5::Sop1MachineInst>(0xbe804900u);
  swap.sdst = 30;
  swap.ssrc0 = 0;
  append32(std::bit_cast<uint32_t>(swap));
  append32(0xbfb00000u); // s_endpgm continuation

  auto setpc = std::bit_cast<cdna5::Sop1MachineInst>(0xbe804800u);
  setpc.ssrc0 = 30;
  append32(std::bit_cast<uint32_t>(setpc));
  return words;
}

std::vector<uint32_t> make_direct_table_dispatch_text() {
  return make_direct_table_dispatch_text_with_addend(0x2000u - 0x1004u);
}

// Same direct indexed dispatch as make_direct_table_dispatch_text(), but the
// table base is built with TWO s_add_nc_u64 literal adds instead of one. The
// patcher can only rewrite a single recorded add offset, so discovery must refuse
// to resolve this (fail closed) rather than emit a dispatch whose first addend
// still executes.
std::vector<uint32_t> make_double_add_table_dispatch_text() {
  std::vector<uint32_t> words;
  auto append32 = [&](uint32_t word) { words.push_back(word); };
  auto append64 = [&](uint64_t encoding) {
    words.push_back(static_cast<uint32_t>(encoding));
    words.push_back(static_cast<uint32_t>(encoding >> 32));
  };

  auto getpc = std::bit_cast<cdna5::Sop1MachineInst>(0xbe804700u);
  getpc.sdst = 54;
  append32(std::bit_cast<uint32_t>(getpc)); // 0x00: s_get_pc_i64 s[54:55].

  auto add = std::bit_cast<cdna5::Sop2MachineInst>(0xa9800000u);
  add.sdst = 54;
  add.ssrc0 = 54;
  add.ssrc1 = 254;
  append32(std::bit_cast<uint32_t>(add)); // 0x04: s_add_nc_u64 (first add).
  append64(0x800u);                       // 0x08: literal; getpc value 0x1004 + 0x800 = 0x1804.

  append32(std::bit_cast<uint32_t>(add)); // 0x10: s_add_nc_u64 (second add).
  append64(0x2000u - 0x1804u);            // 0x14: literal; 0x1804 + 0x7FC = 0x2000 (table base).

  auto table_load = std::bit_cast<cdna5::SmemMachineInst>(uint64_t{0xf4002000u});
  table_load.sbase = 27; // s[54:55].
  table_load.sdata = 0;
  table_load.scale_offset = 1;
  table_load.soffset = 2;
  append64(std::bit_cast<uint64_t>(table_load)); // 0x18: s_load_b64.

  auto swap = std::bit_cast<cdna5::Sop1MachineInst>(0xbe804900u);
  swap.sdst = 30;
  swap.ssrc0 = 0;
  append32(std::bit_cast<uint32_t>(swap)); // 0x20: s_swap_pc_i64.
  append32(0xbfb00000u);                   // 0x24: s_endpgm continuation.

  auto setpc = std::bit_cast<cdna5::Sop1MachineInst>(0xbe804800u);
  setpc.ssrc0 = 30;
  append32(std::bit_cast<uint32_t>(setpc)); // 0x28: s_set_pc_i64.
  return words;
}

void remove_got_reference(std::vector<uint8_t> &image) {
  Elf64_Ehdr ehdr{};
  std::memcpy(&ehdr, image.data(), sizeof(ehdr));
  std::vector<Elf64_Shdr> sections(ehdr.e_shnum);
  std::memcpy(sections.data(), image.data() + ehdr.e_shoff, sections.size() * sizeof(Elf64_Shdr));
  const auto rela = std::ranges::find(sections, SHT_RELA, &Elf64_Shdr::sh_type);
  ASSERT_NE(rela, sections.end());
  Elf64_Rela first{};
  std::memcpy(&first, image.data() + rela->sh_offset, sizeof(first));
  first.r_info = relocation_info(0, 0);
  std::memcpy(image.data() + rela->sh_offset, &first, sizeof(first));
}

// Mutate the first R_AMDGPU_RELATIVE64 relocation in the .rela section via a
// caller-supplied editor, so negative tests can corrupt one table entry's slot
// offset or addend and confirm discovery rejects just that entry.
void edit_first_relative64(std::vector<uint8_t> &image,
                           const std::function<void(Elf64_Rela &)> &editor) {
  Elf64_Ehdr ehdr{};
  std::memcpy(&ehdr, image.data(), sizeof(ehdr));
  std::vector<Elf64_Shdr> sections(ehdr.e_shnum);
  std::memcpy(sections.data(), image.data() + ehdr.e_shoff, sections.size() * sizeof(Elf64_Shdr));
  const auto rela_shdr = std::ranges::find(sections, SHT_RELA, &Elf64_Shdr::sh_type);
  ASSERT_NE(rela_shdr, sections.end());
  const size_t count = rela_shdr->sh_size / sizeof(Elf64_Rela);
  for (size_t i = 0; i < count; ++i) {
    const uint64_t off = rela_shdr->sh_offset + i * sizeof(Elf64_Rela);
    Elf64_Rela rela{};
    std::memcpy(&rela, image.data() + off, sizeof(rela));
    if (elf_reloc_type(rela.r_info) != R_AMDGPU_RELATIVE64)
      continue;
    editor(rela);
    std::memcpy(image.data() + off, &rela, sizeof(rela));
    return;
  }
  FAIL() << "no R_AMDGPU_RELATIVE64 relocation found to edit";
}

void append_nonallocated_relocations(std::vector<uint8_t> &image,
                                     std::span<const Elf64_Rela> relocations) {
  Elf64_Ehdr ehdr{};
  std::memcpy(&ehdr, image.data(), sizeof(ehdr));
  std::vector<Elf64_Shdr> sections(ehdr.e_shnum);
  std::memcpy(sections.data(), image.data() + ehdr.e_shoff, sections.size() * sizeof(Elf64_Shdr));

  const uint64_t relocation_offset = ehdr.e_shoff;
  const auto *bytes = reinterpret_cast<const uint8_t *>(relocations.data());
  image.insert(image.begin() + static_cast<std::ptrdiff_t>(relocation_offset), bytes,
               bytes + relocations.size_bytes());

  Elf64_Shdr metadata_relocations{};
  metadata_relocations.sh_type = SHT_RELA;
  metadata_relocations.sh_offset = relocation_offset;
  metadata_relocations.sh_size = relocations.size_bytes();
  metadata_relocations.sh_link = 4;
  metadata_relocations.sh_info = ehdr.e_shstrndx;
  metadata_relocations.sh_addralign = alignof(Elf64_Rela);
  metadata_relocations.sh_entsize = sizeof(Elf64_Rela);
  sections.push_back(metadata_relocations);

  ehdr.e_shoff += relocations.size_bytes();
  ehdr.e_shnum = static_cast<uint16_t>(sections.size());
  image.resize(ehdr.e_shoff + sections.size() * sizeof(Elf64_Shdr));
  std::memcpy(image.data(), &ehdr, sizeof(ehdr));
  std::memcpy(image.data() + ehdr.e_shoff, sections.data(), sections.size() * sizeof(Elf64_Shdr));
}

TEST(RelocationFunctionTable, DiscoversGotReferencedRelativeTextPointers) {
  const auto image = make_relocation_function_table_elf();
  const AmdGpuCodeObject object(image.data(), image.size());
  ASSERT_TRUE(object.is_valid());

  const auto tables = discover_relocation_function_tables(object);
  ASSERT_EQ(tables.size(), 1u);
  EXPECT_EQ(tables[0].table_vaddr, 0x2000u);
  EXPECT_EQ(tables[0].table_size, 24u);
  EXPECT_EQ(tables[0].got_slot_vaddrs, std::vector<uint64_t>{0x3000u});
  ASSERT_EQ(tables[0].entries.size(), 2u);
  EXPECT_EQ(tables[0].entries[0].slot_vaddr, 0x2000u);
  EXPECT_EQ(tables[0].entries[0].target_text_offset, 4u);
  EXPECT_EQ(tables[0].entries[1].slot_vaddr, 0x2010u);
  EXPECT_EQ(tables[0].entries[1].target_text_offset, 12u);
}

TEST(RelocationFunctionTable, IgnoresExplicitNonAllocatedRelocations) {
  auto image = make_relocation_function_table_elf();
  const std::array<Elf64_Rela, 2> metadata_relocations = {
      Elf64_Rela{.r_offset = 0x2008u,
                 .r_info = relocation_info(0, R_AMDGPU_RELATIVE64),
                 .r_addend = 0x1008},
      Elf64_Rela{.r_offset = 0x3008u, .r_info = relocation_info(1, R_AMDGPU_ABS64), .r_addend = 0},
  };
  append_nonallocated_relocations(image, metadata_relocations);
  const AmdGpuCodeObject object(image.data(), image.size());
  ASSERT_TRUE(object.is_valid());

  const auto tables = discover_relocation_function_tables(object);
  ASSERT_EQ(tables.size(), 1u);
  EXPECT_EQ(tables[0].got_slot_vaddrs, std::vector<uint64_t>{0x3000u});
  ASSERT_EQ(tables[0].entries.size(), 2u);
  EXPECT_EQ(tables[0].entries[0].slot_vaddr, 0x2000u);
  EXPECT_EQ(tables[0].entries[1].slot_vaddr, 0x2010u);
}

TEST(RelocationFunctionTable, DiscoveryRejectsMisalignedTableSlot) {
  // The first table entry's RELATIVE64 relocation points at table_vaddr + 4, which
  // is not a multiple of the 8-byte pointer stride from the table base. Discovery
  // must drop that entry (it cannot be a valid function-pointer slot) while keeping
  // the well-formed second entry.
  auto image = make_relocation_function_table_elf();
  edit_first_relative64(image, [](Elf64_Rela &rela) { rela.r_offset = 0x2000u + 4u; });
  const AmdGpuCodeObject object(image.data(), image.size());
  ASSERT_TRUE(object.is_valid());

  const auto tables = discover_relocation_function_tables(object);
  ASSERT_EQ(tables.size(), 1u);
  ASSERT_EQ(tables[0].entries.size(), 1u);
  EXPECT_EQ(tables[0].entries[0].slot_vaddr, 0x2010u);
  EXPECT_EQ(tables[0].entries[0].target_text_offset, 12u);
}

TEST(RelocationFunctionTable, DiscoveryRejectsOutOfTextAddend) {
  // The first table entry's RELATIVE64 addend points outside .text (into the table
  // section itself). A function-pointer slot must resolve into .text, so discovery
  // must drop that entry while keeping the in-text second entry.
  auto image = make_relocation_function_table_elf();
  edit_first_relative64(image, [](Elf64_Rela &rela) { rela.r_addend = 0x2000; });
  const AmdGpuCodeObject object(image.data(), image.size());
  ASSERT_TRUE(object.is_valid());

  const auto tables = discover_relocation_function_tables(object);
  ASSERT_EQ(tables.size(), 1u);
  ASSERT_EQ(tables[0].entries.size(), 1u);
  EXPECT_EQ(tables[0].entries[0].slot_vaddr, 0x2010u);
  EXPECT_EQ(tables[0].entries[0].target_text_offset, 12u);
}

TEST(RelocationFunctionTable, PatcherRetargetsRelativeTextAddends) {
  const auto image = make_relocation_function_table_elf();
  const AmdGpuCodeObject object(image.data(), image.size());
  ASSERT_TRUE(object.is_valid());
  CodeObjectPatcher patcher(object);

  const std::array<uint32_t, 6> expanded_text = {0xbf800000u, 0xbf800000u, 0xbf800000u,
                                                 0xbf800000u, 0xbf800000u, 0xbf800000u};
  constexpr std::array<TextOffsetRelocation, 2> mappings = {
      TextOffsetRelocation{.source_offset = 4, .target_offset = 8},
      TextOffsetRelocation{.source_offset = 12, .target_offset = 20},
  };
  constexpr std::array<PcRelativeDataRelocation, 1> data_mappings = {
      PcRelativeDataRelocation{
          .target_getpc_offset = 0, .target_literal_offset = 8, .source_target_vaddr = 0x3000},
  };
  const auto bytes = std::span<const uint8_t>(
      reinterpret_cast<const uint8_t *>(expanded_text.data()), sizeof(expanded_text));
  ASSERT_TRUE(patcher.replace_text(bytes, mappings, data_mappings));

  const auto patched = patcher.emit();
  Elf64_Ehdr ehdr{};
  std::memcpy(&ehdr, patched.data(), sizeof(ehdr));
  ASSERT_GT(ehdr.e_shnum, 5u);
  std::vector<Elf64_Shdr> shdrs(ehdr.e_shnum);
  std::memcpy(shdrs.data(), patched.data() + ehdr.e_shoff, shdrs.size() * sizeof(Elf64_Shdr));
  ASSERT_EQ(shdrs[5].sh_type, SHT_RELA);
  ASSERT_EQ(shdrs[5].sh_size, 3 * sizeof(Elf64_Rela));
  std::array<Elf64_Rela, 3> relas{};
  std::memcpy(relas.data(), patched.data() + shdrs[5].sh_offset, sizeof(relas));
  EXPECT_EQ(relas[1].r_addend, 0x1008);
  EXPECT_EQ(relas[2].r_addend, 0x1014);
  EXPECT_EQ(relas[1].r_offset, 0x2008u);
  EXPECT_EQ(relas[2].r_offset, 0x2018u);
  uint64_t relocated_got_delta = 0;
  std::memcpy(&relocated_got_delta, patched.data() + patcher.text_offset() + 8,
              sizeof(relocated_got_delta));
  EXPECT_EQ(relocated_got_delta, 0x2004u);
}

TEST(RelocationFunctionTable, PatcherRejectsRelativeTextAddendWithoutExactOffsetMap) {
  const auto image = make_relocation_function_table_elf();
  const AmdGpuCodeObject object(image.data(), image.size());
  ASSERT_TRUE(object.is_valid());
  CodeObjectPatcher patcher(object);

  const std::array<uint32_t, 6> expanded_text = {0xbf800000u, 0xbf800000u, 0xbf800000u,
                                                 0xbf800000u, 0xbf800000u, 0xbf800000u};
  // The fixture has RELATIVE64 addends targeting source offsets 4 and 12. Omit
  // the latter deliberately: accepting the partial map would leave one stale
  // function-table pointer in the output code object.
  constexpr std::array<TextOffsetRelocation, 1> incomplete_mapping = {
      TextOffsetRelocation{.source_offset = 4, .target_offset = 8},
  };
  const auto bytes = std::span<const uint8_t>(
      reinterpret_cast<const uint8_t *>(expanded_text.data()), sizeof(expanded_text));
  EXPECT_FALSE(patcher.replace_text(bytes, incomplete_mapping));
}

TEST(RelocationFunctionTable, PatcherRejectsConflictingRelative64PlacementsAcrossScopes) {
  const auto image = make_relocation_function_table_elf();
  const AmdGpuCodeObject object(image.data(), image.size());
  ASSERT_TRUE(object.is_valid());
  CodeObjectPatcher patcher(object);

  const std::array<uint32_t, 6> expanded_text = {0xbf800000u, 0xbf800000u, 0xbf800000u,
                                                 0xbf800000u, 0xbf800000u, 0xbf800000u};
  // A shared source block emitted into more than one kernel-local scope produces
  // the SAME source offset with DIFFERENT relocated placements. A RELATIVE64
  // function pointer is dereferenced at runtime, so no single rewrite is correct:
  // the patcher must fail closed rather than collapse to one arbitrary clone. The
  // fixture dereferences source offset 4 as a RELATIVE64 addend; provide two
  // conflicting targets for it (plus the required mapping for source offset 12).
  constexpr std::array<TextOffsetRelocation, 3> conflicting = {
      TextOffsetRelocation{.source_offset = 4, .target_offset = 8},
      TextOffsetRelocation{.source_offset = 4, .target_offset = 16},
      TextOffsetRelocation{.source_offset = 12, .target_offset = 20},
  };
  const auto bytes = std::span<const uint8_t>(
      reinterpret_cast<const uint8_t *>(expanded_text.data()), sizeof(expanded_text));
  EXPECT_FALSE(patcher.replace_text(bytes, conflicting));
}

TEST(RelocationFunctionTable, ResolvesDynamicDispatchThroughGotAndTableLoads) {
  const auto text_words = make_table_dispatch_text();
  const auto image = make_relocation_function_table_elf(text_words);
  const AmdGpuCodeObject object(image.data(), image.size());
  ASSERT_TRUE(object.is_valid());
  const auto tables = discover_relocation_function_tables(object);
  ASSERT_EQ(tables.size(), 1u);

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  std::array<uint64_t, 1> leaders{40};
  const auto blocks = build_valid_blocks(object, *decoder, ROCJITSU_CODE_ARCH_CDNA5, leaders);
  const auto dispatches = analyze_relocation_pairs(blocks, tables, 0x1000).dispatches;
  ASSERT_EQ(dispatches.size(), 1u);
  EXPECT_EQ(dispatches[0].table_index, 0u);
  EXPECT_EQ(dispatches[0].source_call_offset, 32u);
  EXPECT_EQ(dispatches[0].return_sreg, 30u);
  EXPECT_EQ(dispatches[0].source_getpc_offset, 0u);
  EXPECT_EQ(dispatches[0].source_address_add_offset, 4u);
  EXPECT_EQ(dispatches[0].source_table_address_vaddr, 0x3000u);
}

TEST(RelocationFunctionTable, ResolvesRcclDirectIndexedTableDispatch) {
  const auto text_words = make_direct_table_dispatch_text();
  auto image = make_relocation_function_table_elf(text_words);
  remove_got_reference(image);
  const AmdGpuCodeObject object(image.data(), image.size());
  ASSERT_TRUE(object.is_valid());

  const auto tables = discover_relocation_function_tables(object);
  ASSERT_EQ(tables.size(), 1u);
  EXPECT_TRUE(tables[0].got_slot_vaddrs.empty());

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  std::array<uint64_t, 1> leaders{32};
  const auto blocks = build_valid_blocks(object, *decoder, ROCJITSU_CODE_ARCH_CDNA5, leaders);
  const auto analysis = analyze_relocation_pairs(blocks, tables, 0x1000);
  const auto &dispatches = analysis.dispatches;
  ASSERT_EQ(dispatches.size(), 1u);
  EXPECT_EQ(dispatches[0].table_index, 0u);
  EXPECT_EQ(dispatches[0].source_call_offset, 24u);
  EXPECT_EQ(dispatches[0].return_sreg, 30u);
  EXPECT_EQ(dispatches[0].source_getpc_offset, 0u);
  EXPECT_EQ(dispatches[0].source_address_add_offset, 4u);
  EXPECT_EQ(dispatches[0].source_table_address_vaddr, 0x2000u);

  ASSERT_EQ(analysis.address_builders.size(), 1u);
  EXPECT_EQ(analysis.address_builders[0].source_getpc_offset, 0u);
  EXPECT_EQ(analysis.address_builders[0].source_address_add_offset, 4u);
  EXPECT_EQ(analysis.address_builders[0].target_vaddr, 0x2000u);

  const auto table_free_analysis = analyze_relocation_pairs(blocks, {}, 0x1000);
  ASSERT_EQ(table_free_analysis.address_builders.size(), 1u);
  EXPECT_EQ(table_free_analysis.address_builders[0].source_getpc_offset,
            analysis.address_builders[0].source_getpc_offset);
  EXPECT_EQ(table_free_analysis.address_builders[0].source_address_add_offset,
            analysis.address_builders[0].source_address_add_offset);
  EXPECT_EQ(table_free_analysis.address_builders[0].target_vaddr,
            analysis.address_builders[0].target_vaddr);
}

TEST(RelocationFunctionTable, RejectsChainedAddressAddDispatch) {
  const auto text_words = make_double_add_table_dispatch_text();
  auto image = make_relocation_function_table_elf(text_words);
  remove_got_reference(image);
  const AmdGpuCodeObject object(image.data(), image.size());
  ASSERT_TRUE(object.is_valid());

  const auto tables = discover_relocation_function_tables(object);
  ASSERT_EQ(tables.size(), 1u);

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  std::array<uint64_t, 1> leaders{44};
  const auto blocks = build_valid_blocks(object, *decoder, ROCJITSU_CODE_ARCH_CDNA5, leaders);
  const auto dispatches = analyze_relocation_pairs(blocks, tables, 0x1000).dispatches;
  // The base is built with two literal adds; only one add offset can be relocated,
  // so the dispatch must fail closed rather than resolve to a value whose first
  // addend would still execute.
  EXPECT_TRUE(dispatches.empty());
}

TEST(RelocationFunctionTable, ResolvesBackwardTableAddress) {
  // The table sits BELOW the assumed text base, so the address is built by adding a
  // two's-complement negative literal to the getpc result, which wraps modulo
  // 2^64. Discovery must perform modulo arithmetic and still recognize the table,
  // rather than treating the expected wraparound as overflow and rejecting a valid
  // kernel. Assume text base 0x4000 (above the 0x2000 table); getpc value is
  // 0x4004, so the literal is the wrapped 0x2000 - 0x4004.
  constexpr uint64_t kAssumedTextVaddr = 0x4000u;
  const uint64_t backward_addend = 0x2000u - (kAssumedTextVaddr + 0x4u);
  const auto text_words = make_direct_table_dispatch_text_with_addend(backward_addend);
  auto image = make_relocation_function_table_elf(text_words);
  remove_got_reference(image);
  const AmdGpuCodeObject object(image.data(), image.size());
  ASSERT_TRUE(object.is_valid());

  const auto tables = discover_relocation_function_tables(object);
  ASSERT_EQ(tables.size(), 1u);

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  std::array<uint64_t, 1> leaders{32};
  const auto blocks = build_valid_blocks(object, *decoder, ROCJITSU_CODE_ARCH_CDNA5, leaders);
  const auto dispatches = analyze_relocation_pairs(blocks, tables, kAssumedTextVaddr).dispatches;
  ASSERT_EQ(dispatches.size(), 1u);
  EXPECT_EQ(dispatches[0].table_index, 0u);
  EXPECT_EQ(dispatches[0].source_table_address_vaddr, 0x2000u);
}

} // namespace
} // namespace rocjitsu
