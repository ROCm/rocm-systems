/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// ELF code objects: the image size used to copy hipModuleLoadData inputs (ElfImageSize), the
// .dynsym names (DynamicSymbolNames), which replaced comgr's symbol iteration, the note walk
// (ForEachNote) that finds the metadata notes, and the ISA named by the header
// (GetIsaFromHeader). The ELFs are built here, section by section.

// amd_hsa_elf.h first: elf_types.hpp (also included by device.hpp) defines R_AMDGPU_* and
// EM_AMDGPU as macros, which clash with its enums. It needs <cstdint> first.
#include <cstdint>
#include "hsa/amd_hsa_elf.h"

#include "device/codeobject/elf_header.hpp"
#include "device/device.hpp"
#include "elf/elfio/elf_types.hpp"
#undef EM_AMDGPU

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstring>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace amd::codeobject;
using amd::ELFIO::Elf64_Ehdr;
using amd::ELFIO::Elf64_Phdr;
using amd::ELFIO::Elf64_Shdr;
using amd::ELFIO::Elf64_Sym;
using amd::ELFIO::Elf_Word;

namespace {

//! One .dynsym entry
struct Sym {
  std::string name;
  unsigned char type;
};

template <typename T>
void Put(std::vector<char>& out, size_t offset, const T& value) {
  std::memcpy(out.data() + offset, &value, sizeof(value));
}

void Align(std::vector<char>& out, size_t alignment) {
  out.resize((out.size() + alignment - 1) / alignment * alignment, '\0');
}

/**
 * A code object v5 ELF with .shstrtab, .dynstr and .dynsym holding @p syms, then the section
 * header table; with @p loadSegment, also a program header table with one PT_LOAD covering the
 * file.
 */
std::vector<char> MakeElf(const std::vector<Sym>& syms, bool loadSegment = false) {
  std::vector<char> elf(sizeof(Elf64_Ehdr), '\0');
  const size_t phoff = elf.size();
  if (loadSegment) elf.resize(elf.size() + sizeof(Elf64_Phdr), '\0');

  const std::string shstrtab = std::string("\0.shstrtab\0.dynstr\0.dynsym\0", 27);
  const size_t shstrtabOff = elf.size();
  elf.insert(elf.end(), shstrtab.begin(), shstrtab.end());

  std::string dynstr(1, '\0');
  std::vector<Elf64_Sym> table(1, Elf64_Sym{});
  for (const Sym& sym : syms) {
    Elf64_Sym s{};
    s.st_name = sym.name.empty() ? 0 : static_cast<Elf64_Word>(dynstr.size());
    s.st_info = ELF_ST_INFO(STB_GLOBAL, sym.type);
    s.st_shndx = SHN_ABS;
    table.push_back(s);
    if (!sym.name.empty()) dynstr += sym.name + '\0';
  }
  const size_t dynstrOff = elf.size();
  elf.insert(elf.end(), dynstr.begin(), dynstr.end());
  Align(elf, 8);
  const size_t dynsymOff = elf.size();
  elf.resize(elf.size() + table.size() * sizeof(Elf64_Sym));
  std::memcpy(elf.data() + dynsymOff, table.data(), table.size() * sizeof(Elf64_Sym));
  Align(elf, 8);

  Elf64_Shdr shdrs[4] = {};
  shdrs[1].sh_name = 1;
  shdrs[1].sh_type = SHT_STRTAB;
  shdrs[1].sh_offset = shstrtabOff;
  shdrs[1].sh_size = shstrtab.size();
  shdrs[1].sh_addralign = 1;
  shdrs[2].sh_name = 11;
  shdrs[2].sh_type = SHT_STRTAB;
  shdrs[2].sh_flags = SHF_ALLOC;
  shdrs[2].sh_offset = dynstrOff;
  shdrs[2].sh_size = dynstr.size();
  shdrs[2].sh_addralign = 1;
  shdrs[3].sh_name = 19;
  shdrs[3].sh_type = SHT_DYNSYM;
  shdrs[3].sh_flags = SHF_ALLOC;
  shdrs[3].sh_offset = dynsymOff;
  shdrs[3].sh_size = table.size() * sizeof(Elf64_Sym);
  shdrs[3].sh_link = 2;
  shdrs[3].sh_info = 1;
  shdrs[3].sh_addralign = 8;
  shdrs[3].sh_entsize = sizeof(Elf64_Sym);
  const size_t shoff = elf.size();
  elf.resize(elf.size() + sizeof(shdrs));
  std::memcpy(elf.data() + shoff, shdrs, sizeof(shdrs));

  Elf64_Ehdr ehdr{};
  std::memcpy(ehdr.e_ident, "\177ELF", 4);
  ehdr.e_ident[EI_CLASS] = ELFCLASS64;
  ehdr.e_ident[EI_DATA] = ELFDATA2LSB;
  ehdr.e_ident[EI_VERSION] = EV_CURRENT;
  ehdr.e_ident[EI_OSABI] = ELF::ELFOSABI_AMDGPU_HSA;
  ehdr.e_ident[EI_ABIVERSION] = ELF::ELFABIVERSION_AMDGPU_HSA_V5;
  ehdr.e_type = ET_DYN;
  ehdr.e_machine = ELF::EM_AMDGPU;
  ehdr.e_version = EV_CURRENT;
  ehdr.e_flags = ELF::EF_AMDGPU_MACH_AMDGCN_GFX1100;
  ehdr.e_ehsize = sizeof(Elf64_Ehdr);
  ehdr.e_shoff = shoff;
  ehdr.e_shentsize = sizeof(Elf64_Shdr);
  ehdr.e_shnum = 4;
  ehdr.e_shstrndx = 1;
  if (loadSegment) {
    ehdr.e_phoff = phoff;
    ehdr.e_phentsize = sizeof(Elf64_Phdr);
    ehdr.e_phnum = 1;
    Elf64_Phdr phdr{};
    phdr.p_type = PT_LOAD;
    phdr.p_flags = PF_R;
    phdr.p_filesz = elf.size();
    phdr.p_memsz = elf.size();
    phdr.p_align = 4096;
    Put(elf, phoff, phdr);
  }
  Put(elf, 0, ehdr);
  return elf;
}

const std::vector<Sym> kSyms = {{"kernel_a", STT_FUNC},   {"global_x", STT_OBJECT},
                                {"", STT_FUNC},           {"label", STT_NOTYPE},
                                {"kernel_a.kd", STT_OBJECT}, {"section", STT_SECTION},
                                {"kernel_b", STT_FUNC}};

}  // namespace

TEST_CASE("ELF image size covers headers, sections and segments", "[elf]") {
  const std::vector<char> elf = MakeElf(kSyms);
  CHECK(ElfImageSize(elf.data(), elf.size()) == elf.size());
  CHECK(ElfImageSize(elf.data(), kUnknownSize) == elf.size());

  // Section data past the section header table counts too
  std::vector<char> tail = elf;
  tail.resize(tail.size() + 256, '\0');
  Elf64_Shdr extra{};
  std::memcpy(&extra, tail.data() + elf.size() - sizeof(Elf64_Shdr), sizeof(extra));
  extra.sh_offset = elf.size() + 128;
  extra.sh_size = 128;
  std::memcpy(tail.data() + elf.size() - sizeof(Elf64_Shdr), &extra, sizeof(extra));
  CHECK(ElfImageSize(tail.data(), tail.size()) == elf.size() + 256);

  // A segment reaching further than any section counts
  const std::vector<char> withLoad = MakeElf(kSyms, /*loadSegment=*/true);
  std::vector<char> longer = withLoad;
  longer.resize(longer.size() + 4096, '\0');
  Elf64_Phdr phdr{};
  std::memcpy(&phdr, longer.data() + sizeof(Elf64_Ehdr), sizeof(phdr));
  phdr.p_filesz = longer.size();
  std::memcpy(longer.data() + sizeof(Elf64_Ehdr), &phdr, sizeof(phdr));
  CHECK(ElfImageSize(longer.data(), longer.size()) == longer.size());

  // Without a section table the segments alone decide
  std::vector<char> noSections = longer;
  Elf64_Ehdr ehdr{};
  std::memcpy(&ehdr, noSections.data(), sizeof(ehdr));
  ehdr.e_shnum = 0;
  ehdr.e_shoff = 0;
  std::memcpy(noSections.data(), &ehdr, sizeof(ehdr));
  CHECK(ElfImageSize(noSections.data(), noSections.size()) == longer.size());
}

TEST_CASE("ELF image size rejects headers that leave the image", "[elf]") {
  const std::vector<char> elf = MakeElf(kSyms);
  Elf64_Ehdr ehdr{};
  std::memcpy(&ehdr, elf.data(), sizeof(ehdr));

  SECTION("bound shorter than the image") {
    CHECK(ElfImageSize(elf.data(), elf.size() - 1) == 0);
    CHECK(ElfImageSize(elf.data(), sizeof(Elf64_Ehdr) - 1) == 0);
  }
  SECTION("32-bit class") {
    std::vector<char> bad = elf;
    bad[EI_CLASS] = ELFCLASS32;
    CHECK(ElfImageSize(bad.data(), bad.size()) == 0);
  }
  SECTION("wrong section header size") {
    std::vector<char> bad = elf;
    Elf64_Ehdr h = ehdr;
    h.e_shentsize = sizeof(Elf64_Shdr) - 1;
    Put(bad, 0, h);
    CHECK(ElfImageSize(bad.data(), bad.size()) == 0);
  }
  SECTION("section table past the end") {
    std::vector<char> bad = elf;
    Elf64_Ehdr h = ehdr;
    h.e_shnum = 0xffff;
    Put(bad, 0, h);
    CHECK(ElfImageSize(bad.data(), bad.size()) == 0);
  }
  SECTION("section whose offset + size overflows") {
    std::vector<char> bad = elf;
    Elf64_Shdr shdr{};
    const size_t last = ehdr.e_shoff + 3 * sizeof(Elf64_Shdr);
    std::memcpy(&shdr, bad.data() + last, sizeof(shdr));
    shdr.sh_offset = ~uint64_t(0) - 4;
    shdr.sh_size = 16;
    Put(bad, last, shdr);
    CHECK(ElfImageSize(bad.data(), bad.size()) == 0);
    CHECK(ElfImageSize(bad.data(), kUnknownSize) == 0);
  }
}

TEST_CASE("Dynamic symbol names come from .dynsym", "[elf]") {
  const std::vector<char> elf = MakeElf(kSyms);
  std::vector<std::string> funcs;
  std::vector<std::string> vars;
  REQUIRE(DynamicSymbolNames(elf.data(), elf.size(), funcs, vars));
  // Functions and variables in table order; unnamed, NOTYPE and SECTION symbols are left out
  CHECK(funcs == std::vector<std::string>{"kernel_a", "kernel_b"});
  CHECK(vars == std::vector<std::string>{"global_x", "kernel_a.kd"});
}

TEST_CASE("A code object without .dynsym has no dynamic symbols", "[elf]") {
  std::vector<char> elf = MakeElf({});
  // Turn .dynsym into an unrelated section
  Elf64_Ehdr ehdr{};
  std::memcpy(&ehdr, elf.data(), sizeof(ehdr));
  Elf64_Shdr shdr{};
  const size_t dynsym = ehdr.e_shoff + 3 * sizeof(Elf64_Shdr);
  std::memcpy(&shdr, elf.data() + dynsym, sizeof(shdr));
  shdr.sh_type = SHT_PROGBITS;
  shdr.sh_link = 0;
  shdr.sh_entsize = 0;
  Put(elf, dynsym, shdr);

  std::vector<std::string> funcs;
  std::vector<std::string> vars;
  REQUIRE(DynamicSymbolNames(elf.data(), elf.size(), funcs, vars));
  CHECK(funcs.empty());
  CHECK(vars.empty());
}

TEST_CASE("A malformed .dynsym is rejected", "[elf]") {
  std::vector<char> elf = MakeElf(kSyms);
  Elf64_Ehdr ehdr{};
  std::memcpy(&ehdr, elf.data(), sizeof(ehdr));
  Elf64_Shdr shdr{};
  const size_t dynsym = ehdr.e_shoff + 3 * sizeof(Elf64_Shdr);
  std::memcpy(&shdr, elf.data() + dynsym, sizeof(shdr));
  std::vector<std::string> funcs;
  std::vector<std::string> vars;

  SECTION("wrong entry size") {
    Elf64_Shdr bad = shdr;
    bad.sh_entsize = sizeof(Elf64_Sym) - 1;
    Put(elf, dynsym, bad);
    CHECK_FALSE(DynamicSymbolNames(elf.data(), elf.size(), funcs, vars));
  }
  SECTION("string table index out of range") {
    Elf64_Shdr bad = shdr;
    bad.sh_link = 4;
    Put(elf, dynsym, bad);
    CHECK_FALSE(DynamicSymbolNames(elf.data(), elf.size(), funcs, vars));
  }
  SECTION("linked to a section that isn't a string table") {
    Elf64_Shdr bad = shdr;
    bad.sh_link = 3;
    Put(elf, dynsym, bad);
    CHECK_FALSE(DynamicSymbolNames(elf.data(), elf.size(), funcs, vars));
  }
  SECTION("name offset past .dynstr") {
    Elf64_Sym sym{};
    std::memcpy(&sym, elf.data() + shdr.sh_offset + sizeof(Elf64_Sym), sizeof(sym));
    sym.st_name = 0x10000;
    Put(elf, shdr.sh_offset + sizeof(Elf64_Sym), sym);
    CHECK_FALSE(DynamicSymbolNames(elf.data(), elf.size(), funcs, vars));
  }
  SECTION("name not terminated inside .dynstr") {
    // Drop the last name's terminator from the string table
    Elf64_Shdr dynstr{};
    const size_t dynstrHdr = ehdr.e_shoff + 2 * sizeof(Elf64_Shdr);
    std::memcpy(&dynstr, elf.data() + dynstrHdr, sizeof(dynstr));
    dynstr.sh_size -= 1;
    Put(elf, dynstrHdr, dynstr);
    CHECK_FALSE(DynamicSymbolNames(elf.data(), elf.size(), funcs, vars));
  }
}

namespace {

//! One ELF note: name (with its terminator unless @p nullTerminated is false), type, desc
std::vector<char> Note(const std::string& name, uint32_t type, const std::string& desc,
                       bool nullTerminated = true) {
  const uint32_t nameSize = static_cast<uint32_t>(name.size() + (nullTerminated ? 1 : 0));
  const uint32_t header[3] = {nameSize, static_cast<uint32_t>(desc.size()), type};
  auto align4 = [](std::string& s) { s.resize((s.size() + 3) / 4 * 4, '\0'); };
  std::string note(reinterpret_cast<const char*>(header), sizeof(header));
  note += name;
  if (nullTerminated) note += '\0';
  align4(note);
  note += desc;
  align4(note);
  return std::vector<char>(note.begin(), note.end());
}

std::vector<char> Concat(const std::vector<std::vector<char>>& parts) {
  std::vector<char> out;
  for (const auto& part : parts) out.insert(out.end(), part.begin(), part.end());
  return out;
}

//! A code object with one SHT_NOTE section per entry of @p sections, between a PROGBITS section
std::vector<char> MakeNoteElf(const std::vector<std::vector<char>>& sections) {
  std::vector<char> elf(sizeof(Elf64_Ehdr), '\0');
  const std::string shstrtab = std::string("\0.shstrtab\0.note\0.text\0", 23);
  const size_t shstrtabOff = elf.size();
  elf.insert(elf.end(), shstrtab.begin(), shstrtab.end());
  Align(elf, 8);

  std::vector<Elf64_Shdr> shdrs(2, Elf64_Shdr{});
  shdrs[1].sh_name = 1;
  shdrs[1].sh_type = SHT_STRTAB;
  shdrs[1].sh_offset = shstrtabOff;
  shdrs[1].sh_size = shstrtab.size();
  shdrs[1].sh_addralign = 1;
  for (const auto& section : sections) {
    Elf64_Shdr text{};
    text.sh_name = 17;
    text.sh_type = SHT_PROGBITS;
    text.sh_offset = elf.size();
    text.sh_size = 8;
    elf.resize(elf.size() + 8, '\x5a');
    shdrs.push_back(text);

    Elf64_Shdr note{};
    note.sh_name = 11;
    note.sh_type = SHT_NOTE;
    note.sh_offset = elf.size();
    note.sh_size = section.size();
    note.sh_addralign = 4;
    elf.insert(elf.end(), section.begin(), section.end());
    Align(elf, 8);
    shdrs.push_back(note);
  }
  const size_t shoff = elf.size();
  elf.resize(elf.size() + shdrs.size() * sizeof(Elf64_Shdr));
  std::memcpy(elf.data() + shoff, shdrs.data(), shdrs.size() * sizeof(Elf64_Shdr));

  Elf64_Ehdr ehdr{};
  std::memcpy(ehdr.e_ident, "\177ELF", 4);
  ehdr.e_ident[EI_CLASS] = ELFCLASS64;
  ehdr.e_ident[EI_DATA] = ELFDATA2LSB;
  ehdr.e_ident[EI_VERSION] = EV_CURRENT;
  ehdr.e_ident[EI_OSABI] = ELF::ELFOSABI_AMDGPU_HSA;
  ehdr.e_ident[EI_ABIVERSION] = ELF::ELFABIVERSION_AMDGPU_HSA_V5;
  ehdr.e_type = ET_DYN;
  ehdr.e_machine = ELF::EM_AMDGPU;
  ehdr.e_version = EV_CURRENT;
  ehdr.e_flags = ELF::EF_AMDGPU_MACH_AMDGCN_GFX1100;
  ehdr.e_ehsize = sizeof(Elf64_Ehdr);
  ehdr.e_shoff = shoff;
  ehdr.e_shentsize = sizeof(Elf64_Shdr);
  ehdr.e_shnum = static_cast<uint16_t>(shdrs.size());
  ehdr.e_shstrndx = 1;
  Put(elf, 0, ehdr);
  return elf;
}

//! @returns The descs of the AMDGPU metadata notes of @p elf, or false if the walk fails
bool MetadataDescs(const std::vector<char>& elf, std::vector<std::string>& descs) {
  return ForEachNote(elf.data(), elf.size(), kAmdgpuNoteName, kAmdgpuMetadataNoteType,
                     [&](const char* desc, uint32_t size) {
                       descs.emplace_back(desc, size);
                       return true;
                     });
}

}  // namespace

TEST_CASE("Metadata notes are found in every note section, in file order", "[elf][notes]") {
  // Other names and types are skipped, also when they come first
  const std::vector<char> elf = MakeNoteElf(
      {Concat({Note("GNU", kAmdgpuMetadataNoteType, "build-id"), Note("AMDGPU", 1, "other"),
               Note("AMDGPU", kAmdgpuMetadataNoteType, "first")}),
       Concat({Note("AMDGPU", kAmdgpuMetadataNoteType, "second-note")}),
       Concat({Note("AMDGPU", kAmdgpuMetadataNoteType, "third", /*nullTerminated=*/false)})});
  std::vector<std::string> descs;
  REQUIRE(MetadataDescs(elf, descs));
  CHECK(descs == std::vector<std::string>{"first", "second-note", "third"});

  // The callback stops the walk, which still succeeds
  size_t calls = 0;
  CHECK(ForEachNote(elf.data(), elf.size(), kAmdgpuNoteName, kAmdgpuMetadataNoteType,
                    [&](const char*, uint32_t) {
                      ++calls;
                      return false;
                    }));
  CHECK(calls == 1);
}

TEST_CASE("A code object without notes has no metadata", "[elf][notes]") {
  std::vector<std::string> descs;
  const std::vector<char> elf = MakeElf(kSyms);
  CHECK(MetadataDescs(elf, descs));
  CHECK(descs.empty());
  const std::vector<char> empty = MakeNoteElf(std::vector<std::vector<char>>(1));
  CHECK(MetadataDescs(empty, descs));
  CHECK(descs.empty());
}

TEST_CASE("Malformed note sections are rejected", "[elf][notes]") {
  std::vector<std::string> descs;
  SECTION("note header cut by the section end") {
    std::vector<char> section = Note("AMDGPU", kAmdgpuMetadataNoteType, "desc");
    section.resize(section.size() + 8, '\0');
    CHECK_FALSE(MetadataDescs(MakeNoteElf({section}), descs));
  }
  SECTION("desc size past the section end") {
    std::vector<char> section = Note("AMDGPU", kAmdgpuMetadataNoteType, "desc");
    const uint32_t descSize = 0x1000;
    std::memcpy(section.data() + 4, &descSize, sizeof(descSize));
    CHECK_FALSE(MetadataDescs(MakeNoteElf({section}), descs));
  }
  SECTION("name size that overflows") {
    std::vector<char> section = Note("AMDGPU", kAmdgpuMetadataNoteType, "desc");
    const uint32_t nameSize = 0xfffffffe;
    std::memcpy(section.data(), &nameSize, sizeof(nameSize));
    CHECK_FALSE(MetadataDescs(MakeNoteElf({section}), descs));
  }
  SECTION("unknown image size") {
    const std::vector<char> elf =
        MakeNoteElf({Note("AMDGPU", kAmdgpuMetadataNoteType, "desc")});
    CHECK_FALSE(ForEachNote(elf.data(), kUnknownSize, kAmdgpuNoteName, kAmdgpuMetadataNoteType,
                            [](const char*, uint32_t) { return true; }));
  }
}

TEST_CASE("Truncated ELFs never parse", "[elf]") {
  // Every prefix of a valid code object, each in a buffer of exactly that size so a read past
  // it is outside the allocation.
  const std::vector<char> elf = MakeElf(kSyms);
  const std::vector<char> noteElf =
      MakeNoteElf({Note("AMDGPU", kAmdgpuMetadataNoteType, "desc")});
  for (size_t size = 1; size < elf.size(); ++size) {
    INFO("size " << size);
    const std::vector<char> prefix(elf.begin(), elf.begin() + static_cast<std::ptrdiff_t>(size));
    CHECK(ElfImageSize(prefix.data(), prefix.size()) == 0);
    std::vector<std::string> funcs;
    std::vector<std::string> vars;
    CHECK_FALSE(DynamicSymbolNames(prefix.data(), prefix.size(), funcs, vars));
  }
  for (size_t size = 1; size < noteElf.size(); ++size) {
    INFO("size " << size);
    const std::vector<char> prefix(noteElf.begin(),
                                   noteElf.begin() + static_cast<std::ptrdiff_t>(size));
    std::vector<std::string> descs;
    CHECK_FALSE(MetadataDescs(prefix, descs));
  }
}

TEST_CASE("Code objects parse at any offset in their buffer", "[elf]") {
  // Bundle entries and files put a code object at any offset; a vector's data is aligned, so
  // copy each image to every offset up to 7. Run under -fsanitize=alignment to catch typed
  // reads of the headers.
  const std::vector<char> elf = MakeElf(kSyms, /*loadSegment=*/true);
  const std::vector<char> noteElf =
      MakeNoteElf({Note("AMDGPU", kAmdgpuMetadataNoteType, "desc")});
  for (size_t offset = 1; offset < 8; ++offset) {
    INFO("offset " << offset);
    std::vector<char> buffer(offset + elf.size());
    std::memcpy(buffer.data() + offset, elf.data(), elf.size());
    const char* image = buffer.data() + offset;
    CHECK(ElfImageSize(image, elf.size()) == elf.size());
    CHECK(ElfImageSize(image, kUnknownSize) == elf.size());
    std::vector<std::string> funcs;
    std::vector<std::string> vars;
    REQUIRE(DynamicSymbolNames(image, elf.size(), funcs, vars));
    CHECK(funcs == std::vector<std::string>{"kernel_a", "kernel_b"});
    CHECK(vars == std::vector<std::string>{"global_x", "kernel_a.kd"});

    std::vector<char> noteBuffer(offset + noteElf.size());
    std::memcpy(noteBuffer.data() + offset, noteElf.data(), noteElf.size());
    std::vector<std::string> descs;
    REQUIRE(ForEachNote(noteBuffer.data() + offset, noteElf.size(), kAmdgpuNoteName,
                        kAmdgpuMetadataNoteType, [&](const char* desc, uint32_t size) {
                          descs.emplace_back(desc, size);
                          return true;
                        }));
    CHECK(descs == std::vector<std::string>{"desc"});
  }
}

TEST_CASE("The ISA is named from the ELF header like the loader names it", "[elf][isa]") {
  auto isa = [](uint32_t abiVersion, uint32_t eFlags, uint32_t osAbi = ELF::ELFOSABI_AMDGPU_HSA,
                uint16_t machine = ELF::EM_AMDGPU, uint32_t elfClass = ELFCLASS64) {
    std::string name;
    return GetIsaFromHeader(elfClass, machine, osAbi, abiVersion, eFlags, name) ? name
                                                                                : "<none>";
  };
  const uint32_t v3 = ELF::ELFABIVERSION_AMDGPU_HSA_V3;
  const uint32_t v5 = ELF::ELFABIVERSION_AMDGPU_HSA_V5;
  const uint32_t v6 = ELF::ELFABIVERSION_AMDGPU_HSA_V6;

  CHECK(isa(v5, ELF::EF_AMDGPU_MACH_AMDGCN_GFX1100) == "amdgcn-amd-amdhsa--gfx1100");
  // v4+: a feature is named only when the code object sets it
  CHECK(isa(v5, ELF::EF_AMDGPU_MACH_AMDGCN_GFX90A) == "amdgcn-amd-amdhsa--gfx90a");
  CHECK(isa(v5, ELF::EF_AMDGPU_MACH_AMDGCN_GFX90A | ELF::EF_AMDGPU_FEATURE_SRAMECC_ON_V4 |
                    ELF::EF_AMDGPU_FEATURE_XNACK_OFF_V4) ==
        "amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-");
  CHECK(isa(v5, ELF::EF_AMDGPU_MACH_AMDGCN_GFX90A | ELF::EF_AMDGPU_FEATURE_SRAMECC_ANY_V4 |
                    ELF::EF_AMDGPU_FEATURE_XNACK_ON_V4) == "amdgcn-amd-amdhsa--gfx90a:xnack+");
  // v3: a supported feature that isn't set is off
  CHECK(isa(v3, ELF::EF_AMDGPU_MACH_AMDGCN_GFX906) == "amdgcn-amd-amdhsa--gfx906:sramecc-:xnack-");
  CHECK(isa(v3, ELF::EF_AMDGPU_MACH_AMDGCN_GFX906 | ELF::EF_AMDGPU_FEATURE_XNACK_V3) ==
        "amdgcn-amd-amdhsa--gfx906:sramecc-:xnack+");
  CHECK(isa(v3, ELF::EF_AMDGPU_MACH_AMDGCN_GFX1030) == "amdgcn-amd-amdhsa--gfx1030");
  // Generic targets, with the generic version out of the name
  CHECK(isa(v6, ELF::EF_AMDGPU_MACH_AMDGCN_GFX11_GENERIC | (1u << 24)) ==
        "amdgcn-amd-amdhsa--gfx11-generic");
  CHECK(isa(v6, ELF::EF_AMDGPU_MACH_AMDGCN_GFX12_5_GENERIC) ==
        "amdgcn-amd-amdhsa--gfx12-5-generic");
  std::string name;
  unsigned genericVersion = 0;
  REQUIRE(GetIsaFromHeader(ELFCLASS64, ELF::EM_AMDGPU, ELF::ELFOSABI_AMDGPU_HSA, v6,
                           ELF::EF_AMDGPU_MACH_AMDGCN_GFX9_4_GENERIC | (2u << 24), name,
                           &genericVersion));
  CHECK(genericVersion == 2);

  // Not loadable: code object v2, another OS ABI, machine or class, an unknown processor
  CHECK(isa(ELF::ELFABIVERSION_AMDGPU_HSA_V2, ELF::EF_AMDGPU_MACH_AMDGCN_GFX906) == "<none>");
  CHECK(isa(v5, ELF::EF_AMDGPU_MACH_AMDGCN_GFX1100, ELF::ELFOSABI_AMDGPU_HSA + 1) == "<none>");
  CHECK(isa(v5, ELF::EF_AMDGPU_MACH_AMDGCN_GFX1100, ELF::ELFOSABI_AMDGPU_HSA, 62) == "<none>");
  CHECK(isa(v5, ELF::EF_AMDGPU_MACH_AMDGCN_GFX1100, ELF::ELFOSABI_AMDGPU_HSA, ELF::EM_AMDGPU,
            ELFCLASS32) == "<none>");
  CHECK(isa(v5, ELF::EF_AMDGPU_MACH_AMDGCN_GFX940) == "<none>");
  // gfx6 and gfx7 have no amd::Isa row: no device runs their code objects
  CHECK(isa(v5, ELF::EF_AMDGPU_MACH_AMDGCN_GFX600) == "<none>");
  CHECK(isa(v5, ELF::EF_AMDGPU_MACH_AMDGCN_GFX700) == "<none>");
  CHECK(isa(v5, ELF::EF_AMDGPU_MACH_AMDGCN_RESERVED_0X27) == "<none>");
}

TEST_CASE("Every processor row has its mach value and generic target", "[elf][isa]") {
  // GetIsaFromHeader and TargetToGeneric read these amd::Isa columns: set on the row without
  // features, unique, and naming the row back; feature rows leave them unset.
  std::set<uint32_t> machs;
  for (const amd::Isa* isa = amd::Isa::begin(); isa != amd::Isa::end(); ++isa) {
    const std::string targetId = isa->targetId();
    INFO(targetId);
    if (targetId.find(':') != std::string::npos) {
      CHECK(isa->elfMach() == 0);
      CHECK(isa->genericTarget() == nullptr);
      continue;
    }
    REQUIRE(isa->elfMach() != 0);
    CHECK(machs.insert(isa->elfMach()).second);
    std::string name;
    REQUIRE(GetIsaFromHeader(ELFCLASS64, ELF::EM_AMDGPU, ELF::ELFOSABI_AMDGPU_HSA,
                             ELF::ELFABIVERSION_AMDGPU_HSA_V6, isa->elfMach(), name));
    CHECK(name == "amdgcn-amd-amdhsa--" + targetId);
    if (isa->genericTarget() != nullptr) {
      // A generic target is a generic row of the same major version with no generic itself
      const std::string generic = isa->genericTarget();
      const amd::Isa* row = amd::Isa::findIsa(("amdgcn-amd-amdhsa--" + generic).c_str());
      REQUIRE(row != nullptr);
      CHECK(generic.find("-generic") != std::string::npos);
      CHECK(row->genericTarget() == nullptr);
      CHECK(row->versionMajor() == isa->versionMajor());
    }
  }
  // The values ROCr's loader uses (amd_hsa_elf.h), for a processor of each generation
  CHECK(amd::Isa::findIsa("amdgcn-amd-amdhsa--gfx803")->elfMach() ==
        ELF::EF_AMDGPU_MACH_AMDGCN_GFX803);
  CHECK(amd::Isa::findIsa("amdgcn-amd-amdhsa--gfx90a")->elfMach() ==
        ELF::EF_AMDGPU_MACH_AMDGCN_GFX90A);
  CHECK(amd::Isa::findIsa("amdgcn-amd-amdhsa--gfx1100")->elfMach() ==
        ELF::EF_AMDGPU_MACH_AMDGCN_GFX1100);
  CHECK(amd::Isa::findIsa("amdgcn-amd-amdhsa--gfx1250")->elfMach() ==
        ELF::EF_AMDGPU_MACH_AMDGCN_GFX1250);
  CHECK(amd::Isa::findIsa("amdgcn-amd-amdhsa--gfx12-5-generic")->elfMach() ==
        ELF::EF_AMDGPU_MACH_AMDGCN_GFX12_5_GENERIC);
}
