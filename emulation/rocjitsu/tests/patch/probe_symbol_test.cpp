// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Synthetic, always-on coverage for the probe symbol resolver.
//
// NOTE: the minimal-ELF builder below is local to this file on purpose to keep
// the slice self-contained. Other patch tests (instrumentor_test.cpp,
// translate_test.cpp) hand-build similar ELFs; if a third resolver-style test
// needs symbol tables, factor a shared test-fixture header then.

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/patch/probe_symbol.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace rocjitsu {
namespace {

// Virtual address and file offset of the synthetic .text section. They differ
// on purpose so the (st_value - sh_addr) mapping is actually exercised.
constexpr uint64_t kTextAddr = 0x1000;
constexpr uint64_t kTextOffset = 0x200;
constexpr uint64_t kTextSize = 0x40;

struct TestSym {
  std::string name;
  uint8_t type = kElfSymbolTypeFunc;
  uint16_t shndx = 1; // .text by default.
  uint64_t value = kTextAddr + 0x10;
  uint64_t size = 0x8;
};

struct TestTable {
  uint32_t sym_type = SHT_SYMTAB; // SHT_SYMTAB or SHT_DYNSYM.
  std::vector<TestSym> syms;
};

uint64_t align_up(uint64_t value, uint64_t alignment) {
  const uint64_t rem = value % alignment;
  return rem == 0 ? value : value + alignment - rem;
}

uint32_t add_name(std::vector<uint8_t> &names, std::string_view name) {
  const auto offset = static_cast<uint32_t>(names.size());
  names.insert(names.end(), name.begin(), name.end());
  names.push_back('\0');
  return offset;
}

// Build a minimal gfx90a ELF with one .text section and one symbol table per
// entry in @p tables (each with its own linked string table). @p text_exec
// controls whether .text carries SHF_EXECINSTR.
std::vector<uint8_t> make_probe_elf(const std::vector<TestTable> &tables, bool text_exec = true) {
  // Section layout:
  //   0: null
  //   1: .text
  //   for each table t: strtab at (2 + 2t), symtab at (3 + 2t)
  //   last: .shstrtab
  const auto num_tables = static_cast<uint16_t>(tables.size());
  const uint16_t section_count = static_cast<uint16_t>(2 + 2 * num_tables + 1);
  const uint16_t shstrtab_index = static_cast<uint16_t>(2 + 2 * num_tables);

  // Section-header string table (names of the sections).
  std::vector<uint8_t> shstrtab{'\0'};
  const uint32_t text_name = add_name(shstrtab, ".text");
  std::vector<uint32_t> strtab_names;
  std::vector<uint32_t> symtab_names;
  for (uint16_t t = 0; t < num_tables; ++t) {
    const bool is_dyn = tables[t].sym_type == SHT_DYNSYM;
    strtab_names.push_back(add_name(shstrtab, is_dyn ? ".dynstr" : ".strtab"));
    symtab_names.push_back(add_name(shstrtab, is_dyn ? ".dynsym" : ".symtab"));
  }
  const uint32_t shstrtab_name = add_name(shstrtab, ".shstrtab");

  // Per-table string tables (symbol names) and symbol arrays. Each table gets a
  // leading null symbol, as is conventional.
  std::vector<std::vector<uint8_t>> strtabs(num_tables);
  std::vector<std::vector<Elf64_Sym>> symtabs(num_tables);
  for (uint16_t t = 0; t < num_tables; ++t) {
    strtabs[t].push_back('\0');
    symtabs[t].emplace_back(); // null symbol
    for (const auto &s : tables[t].syms) {
      Elf64_Sym sym{};
      sym.st_name = add_name(strtabs[t], s.name);
      sym.st_info = elf_symbol_info(kElfSymbolBindGlobal, s.type);
      sym.st_shndx = s.shndx;
      sym.st_value = s.value;
      sym.st_size = s.size;
      symtabs[t].push_back(sym);
    }
  }

  // Assign file offsets.
  uint64_t cursor = kTextOffset + kTextSize;
  std::vector<uint64_t> strtab_off(num_tables);
  std::vector<uint64_t> symtab_off(num_tables);
  for (uint16_t t = 0; t < num_tables; ++t) {
    strtab_off[t] = cursor;
    cursor += strtabs[t].size();
    symtab_off[t] = align_up(cursor, 8);
    cursor = symtab_off[t] + symtabs[t].size() * sizeof(Elf64_Sym);
  }
  const uint64_t shstrtab_off = cursor;
  cursor += shstrtab.size();
  const uint64_t shoff = align_up(cursor, 8);

  std::vector<uint8_t> image(shoff + section_count * sizeof(Elf64_Shdr), 0);

  Elf64_Ehdr ehdr{};
  std::memcpy(ehdr.e_ident, EI_MAGIC, EI_MAGIC_SIZE);
  ehdr.e_ident[EI_CLASS] = ELFCLASS64;
  ehdr.e_ident[EI_OSABI] = ELFOSABI_AMDGPU_HSA;
  ehdr.e_type = ET_REL;
  ehdr.e_machine = EM_AMDGPU;
  ehdr.e_version = 1;
  ehdr.e_shoff = shoff;
  ehdr.e_flags = EF_AMDGPU_MACH_AMDGCN_GFX90A;
  ehdr.e_ehsize = sizeof(Elf64_Ehdr);
  ehdr.e_shentsize = sizeof(Elf64_Shdr);
  ehdr.e_shnum = section_count;
  ehdr.e_shstrndx = shstrtab_index;
  std::memcpy(image.data(), &ehdr, sizeof(ehdr));

  // .text content: distinct dword per slot so a body range is identifiable.
  for (uint64_t off = 0; off < kTextSize; off += sizeof(uint32_t)) {
    const auto word = static_cast<uint32_t>(0xD0000000u | off);
    std::memcpy(image.data() + kTextOffset + off, &word, sizeof(word));
  }

  for (uint16_t t = 0; t < num_tables; ++t) {
    std::memcpy(image.data() + strtab_off[t], strtabs[t].data(), strtabs[t].size());
    std::memcpy(image.data() + symtab_off[t], symtabs[t].data(),
                symtabs[t].size() * sizeof(Elf64_Sym));
  }
  std::memcpy(image.data() + shstrtab_off, shstrtab.data(), shstrtab.size());

  std::vector<Elf64_Shdr> shdrs(section_count);
  shdrs[1].sh_name = text_name;
  shdrs[1].sh_type = SHT_PROGBITS;
  shdrs[1].sh_flags = SHF_ALLOC | (text_exec ? SHF_EXECINSTR : 0);
  shdrs[1].sh_addr = kTextAddr;
  shdrs[1].sh_offset = kTextOffset;
  shdrs[1].sh_size = kTextSize;
  shdrs[1].sh_addralign = sizeof(uint32_t);

  for (uint16_t t = 0; t < num_tables; ++t) {
    const uint16_t strtab_idx = static_cast<uint16_t>(2 + 2 * t);
    const uint16_t symtab_idx = static_cast<uint16_t>(3 + 2 * t);
    shdrs[strtab_idx].sh_name = strtab_names[t];
    shdrs[strtab_idx].sh_type = SHT_STRTAB;
    shdrs[strtab_idx].sh_offset = strtab_off[t];
    shdrs[strtab_idx].sh_size = strtabs[t].size();
    shdrs[strtab_idx].sh_addralign = 1;

    shdrs[symtab_idx].sh_name = symtab_names[t];
    shdrs[symtab_idx].sh_type = tables[t].sym_type;
    shdrs[symtab_idx].sh_offset = symtab_off[t];
    shdrs[symtab_idx].sh_size = symtabs[t].size() * sizeof(Elf64_Sym);
    shdrs[symtab_idx].sh_link = strtab_idx;
    shdrs[symtab_idx].sh_entsize = sizeof(Elf64_Sym);
    shdrs[symtab_idx].sh_addralign = 8;
  }

  shdrs[shstrtab_index].sh_name = shstrtab_name;
  shdrs[shstrtab_index].sh_type = SHT_STRTAB;
  shdrs[shstrtab_index].sh_offset = shstrtab_off;
  shdrs[shstrtab_index].sh_size = shstrtab.size();
  shdrs[shstrtab_index].sh_addralign = 1;

  std::memcpy(image.data() + shoff, shdrs.data(), shdrs.size() * sizeof(Elf64_Shdr));
  return image;
}

// Convenience: a single SHT_SYMTAB table holding @p syms.
std::vector<uint8_t> make_symtab_elf(std::vector<TestSym> syms, bool text_exec = true) {
  return make_probe_elf({TestTable{SHT_SYMTAB, std::move(syms)}}, text_exec);
}

// --- Image-patching helpers for the malformed-ELF cases below ---

Elf64_Ehdr get_ehdr(const std::vector<uint8_t> &image) {
  Elf64_Ehdr ehdr{};
  std::memcpy(&ehdr, image.data(), sizeof(ehdr));
  return ehdr;
}

void put_ehdr(std::vector<uint8_t> &image, const Elf64_Ehdr &ehdr) {
  std::memcpy(image.data(), &ehdr, sizeof(ehdr));
}

// Apply @p mutate to the first section header whose sh_type == type.
template <typename F>
void patch_first_shdr_of_type(std::vector<uint8_t> &image, uint32_t type, F mutate) {
  const Elf64_Ehdr ehdr = get_ehdr(image);
  for (uint16_t i = 0; i < ehdr.e_shnum; ++i) {
    const uint64_t off = ehdr.e_shoff + i * ehdr.e_shentsize;
    Elf64_Shdr shdr{};
    std::memcpy(&shdr, image.data() + off, sizeof(shdr));
    if (shdr.sh_type == type) {
      mutate(shdr);
      std::memcpy(image.data() + off, &shdr, sizeof(shdr));
      return;
    }
  }
}

TEST(ProbeSymbolTest, ResolvesFromSymtab) {
  const auto image =
      make_symtab_elf({TestSym{.name = "rj_nop_probe", .value = kTextAddr + 0x10, .size = 0x8}});
  AmdGpuCodeObject obj(image.data(), image.size());

  std::string err;
  const auto resolved = resolve_probe_symbol(obj, "rj_nop_probe", &err);
  ASSERT_TRUE(resolved.has_value()) << err;
  EXPECT_EQ(resolved->name, "rj_nop_probe");
  EXPECT_EQ(resolved->section_index, 1u);
  EXPECT_EQ(resolved->st_value, kTextAddr + 0x10);
  EXPECT_EQ(resolved->body_file_offset, kTextOffset + 0x10);
  EXPECT_EQ(resolved->body_size, 0x8u);
}

TEST(ProbeSymbolTest, ResolvesFromDynsym) {
  const auto image = make_probe_elf(
      {TestTable{SHT_DYNSYM, {TestSym{.name = "rj_nop_probe", .value = kTextAddr + 0x20}}}});
  AmdGpuCodeObject obj(image.data(), image.size());

  std::string err;
  const auto resolved = resolve_probe_symbol(obj, "rj_nop_probe", &err);
  ASSERT_TRUE(resolved.has_value()) << err;
  EXPECT_EQ(resolved->body_file_offset, kTextOffset + 0x20);
}

TEST(ProbeSymbolTest, SymtabIsPreferredAndNotDoubleCounted) {
  // Same symbol mirrored in .symtab and .dynsym must not look like a duplicate.
  const auto image = make_probe_elf({
      TestTable{SHT_SYMTAB, {TestSym{.name = "rj_nop_probe", .value = kTextAddr + 0x10}}},
      TestTable{SHT_DYNSYM, {TestSym{.name = "rj_nop_probe", .value = kTextAddr + 0x10}}},
  });
  AmdGpuCodeObject obj(image.data(), image.size());

  std::string err;
  const auto resolved = resolve_probe_symbol(obj, "rj_nop_probe", &err);
  ASSERT_TRUE(resolved.has_value()) << err;
  EXPECT_EQ(resolved->body_file_offset, kTextOffset + 0x10);
}

TEST(ProbeSymbolTest, FallsBackToDynsymWhenSymtabLacksName) {
  // .symtab present but without the probe; .dynsym carries it.
  const auto image = make_probe_elf({
      TestTable{SHT_SYMTAB, {TestSym{.name = "other_fn", .value = kTextAddr + 0x4}}},
      TestTable{SHT_DYNSYM, {TestSym{.name = "rj_nop_probe", .value = kTextAddr + 0x18}}},
  });
  AmdGpuCodeObject obj(image.data(), image.size());

  std::string err;
  const auto resolved = resolve_probe_symbol(obj, "rj_nop_probe", &err);
  ASSERT_TRUE(resolved.has_value()) << err;
  EXPECT_EQ(resolved->body_file_offset, kTextOffset + 0x18);
}

TEST(ProbeSymbolTest, RejectsMissingSymbol) {
  const auto image = make_symtab_elf({TestSym{.name = "other_fn"}});
  AmdGpuCodeObject obj(image.data(), image.size());

  std::string err;
  EXPECT_FALSE(resolve_probe_symbol(obj, "rj_nop_probe", &err).has_value());
  EXPECT_NE(err.find("not found"), std::string::npos) << err;
}

TEST(ProbeSymbolTest, RejectsDuplicateSymbol) {
  const auto image = make_symtab_elf({
      TestSym{.name = "rj_nop_probe", .value = kTextAddr + 0x8},
      TestSym{.name = "rj_nop_probe", .value = kTextAddr + 0x10},
  });
  AmdGpuCodeObject obj(image.data(), image.size());

  std::string err;
  EXPECT_FALSE(resolve_probe_symbol(obj, "rj_nop_probe", &err).has_value());
  EXPECT_NE(err.find("more than once"), std::string::npos) << err;
}

TEST(ProbeSymbolTest, RejectsUndefinedSymbol) {
  const auto image = make_symtab_elf({TestSym{.name = "rj_nop_probe", .shndx = SHN_UNDEF}});
  AmdGpuCodeObject obj(image.data(), image.size());

  std::string err;
  EXPECT_FALSE(resolve_probe_symbol(obj, "rj_nop_probe", &err).has_value());
  EXPECT_NE(err.find("undefined"), std::string::npos) << err;
}

TEST(ProbeSymbolTest, RejectsNonFunctionSymbol) {
  const auto image =
      make_symtab_elf({TestSym{.name = "rj_nop_probe", .type = kElfSymbolTypeObject}});
  AmdGpuCodeObject obj(image.data(), image.size());

  std::string err;
  EXPECT_FALSE(resolve_probe_symbol(obj, "rj_nop_probe", &err).has_value());
  EXPECT_NE(err.find("STT_FUNC"), std::string::npos) << err;
}

TEST(ProbeSymbolTest, RejectsNonExecutableSection) {
  const auto image = make_symtab_elf({TestSym{.name = "rj_nop_probe"}}, /*text_exec=*/false);
  AmdGpuCodeObject obj(image.data(), image.size());

  std::string err;
  EXPECT_FALSE(resolve_probe_symbol(obj, "rj_nop_probe", &err).has_value());
  EXPECT_NE(err.find("executable"), std::string::npos) << err;
}

TEST(ProbeSymbolTest, RejectsZeroSizeSymbol) {
  const auto image = make_symtab_elf({TestSym{.name = "rj_nop_probe", .size = 0}});
  AmdGpuCodeObject obj(image.data(), image.size());

  std::string err;
  EXPECT_FALSE(resolve_probe_symbol(obj, "rj_nop_probe", &err).has_value());
  EXPECT_NE(err.find("zero size"), std::string::npos) << err;
}

TEST(ProbeSymbolTest, RejectsUnalignedSymbol) {
  const auto image = make_symtab_elf({TestSym{.name = "rj_nop_probe", .value = kTextAddr + 0x11}});
  AmdGpuCodeObject obj(image.data(), image.size());

  std::string err;
  EXPECT_FALSE(resolve_probe_symbol(obj, "rj_nop_probe", &err).has_value());
  EXPECT_NE(err.find("aligned"), std::string::npos) << err;
}

TEST(ProbeSymbolTest, RejectsBodyPastSectionEnd) {
  // Body starts near the end of .text but claims to be larger than what's left.
  const auto image = make_symtab_elf(
      {TestSym{.name = "rj_nop_probe", .value = kTextAddr + kTextSize - 4, .size = 0x20}});
  AmdGpuCodeObject obj(image.data(), image.size());

  std::string err;
  EXPECT_FALSE(resolve_probe_symbol(obj, "rj_nop_probe", &err).has_value());
  EXPECT_NE(err.find("past"), std::string::npos) << err;
}

TEST(ProbeSymbolTest, RejectsNonElfImage) {
  std::vector<uint8_t> garbage(64, 0xAB);
  AmdGpuCodeObject obj(garbage.data(), garbage.size());

  std::string err;
  EXPECT_FALSE(resolve_probe_symbol(obj, "rj_nop_probe", &err).has_value());
  EXPECT_FALSE(err.empty());
}

TEST(ProbeSymbolTest, RejectsEmptyName) {
  const auto image = make_symtab_elf({TestSym{.name = "rj_nop_probe"}});
  AmdGpuCodeObject obj(image.data(), image.size());

  std::string err;
  EXPECT_FALSE(resolve_probe_symbol(obj, "", &err).has_value());
  EXPECT_NE(err.find("empty"), std::string::npos) << err;
}

TEST(ProbeSymbolTest, RejectsSymtabLinkNotStringTable) {
  // Point the symbol table's sh_link at .text (PROGBITS) instead of a strtab.
  auto image = make_symtab_elf({TestSym{.name = "rj_nop_probe"}});
  patch_first_shdr_of_type(image, SHT_SYMTAB, [](Elf64_Shdr &shdr) { shdr.sh_link = 1; });
  AmdGpuCodeObject obj(image.data(), image.size());

  std::string err;
  EXPECT_FALSE(resolve_probe_symbol(obj, "rj_nop_probe", &err).has_value());
  EXPECT_NE(err.find("string table"), std::string::npos) << err;
}

TEST(ProbeSymbolTest, AcceptsIdenticalSymbolInTwoSymtabs) {
  // The same definition mirrored across two SHT_SYMTAB sections is one symbol.
  const TestSym sym{.name = "rj_nop_probe", .value = kTextAddr + 0x10, .size = 0x8};
  const auto image = make_probe_elf({
      TestTable{SHT_SYMTAB, {sym}},
      TestTable{SHT_SYMTAB, {sym}},
  });
  AmdGpuCodeObject obj(image.data(), image.size());

  std::string err;
  const auto resolved = resolve_probe_symbol(obj, "rj_nop_probe", &err);
  ASSERT_TRUE(resolved.has_value()) << err;
  EXPECT_EQ(resolved->body_file_offset, kTextOffset + 0x10);
}

TEST(ProbeSymbolTest, RejectsConflictingSymbolInTwoSymtabs) {
  // Two SHT_SYMTAB sections disagreeing on the definition is a real duplicate.
  const auto image = make_probe_elf({
      TestTable{SHT_SYMTAB, {TestSym{.name = "rj_nop_probe", .value = kTextAddr + 0x8}}},
      TestTable{SHT_SYMTAB, {TestSym{.name = "rj_nop_probe", .value = kTextAddr + 0x10}}},
  });
  AmdGpuCodeObject obj(image.data(), image.size());

  std::string err;
  EXPECT_FALSE(resolve_probe_symbol(obj, "rj_nop_probe", &err).has_value());
  EXPECT_NE(err.find("more than once"), std::string::npos) << err;
}

TEST(ProbeSymbolTest, RejectsSectionHeaderTableOutOfRange) {
  // A huge e_shoff must be rejected, not wrapped into an aliased in-bounds read.
  auto image = make_symtab_elf({TestSym{.name = "rj_nop_probe"}});
  Elf64_Ehdr ehdr = get_ehdr(image);
  ehdr.e_shoff = 0xFFFFFFFFFFFFFFE0ULL;
  put_ehdr(image, ehdr);
  AmdGpuCodeObject obj(image.data(), image.size());

  std::string err;
  EXPECT_FALSE(resolve_probe_symbol(obj, "rj_nop_probe", &err).has_value());
  EXPECT_FALSE(err.empty());
}

TEST(ProbeSymbolTest, RejectsOwningSectionFileOffsetOverflow) {
  // owner.sh_offset near UINT64_MAX would wrap when added to the in-section
  // delta; the owning-section bounds check must reject it.
  auto image = make_symtab_elf({TestSym{.name = "rj_nop_probe", .value = kTextAddr + 0x10}});
  patch_first_shdr_of_type(image, SHT_PROGBITS,
                           [](Elf64_Shdr &shdr) { shdr.sh_offset = 0xFFFFFFFFFFFFFFE0ULL; });
  AmdGpuCodeObject obj(image.data(), image.size());

  std::string err;
  EXPECT_FALSE(resolve_probe_symbol(obj, "rj_nop_probe", &err).has_value());
  EXPECT_NE(err.find("past end of image"), std::string::npos) << err;
}

} // namespace
} // namespace rocjitsu
