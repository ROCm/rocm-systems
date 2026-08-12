// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Synthetic, always-on coverage for build_probe_callable().
//
// build_probe_callable() takes a ResolvedProbeSymbol by value, so these tests
// fabricate the symbol directly and only need a minimal ELF carrying the body
// words in an executable .text (plus an optional .rela.text).
//
// Instruction encodings below are gfx90a ground truth captured from
// `llvm-mc -arch=amdgcn -mcpu=gfx90a -show-encoding`.

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/patch/probe_callable.h"
#include "rocjitsu/code/patch/probe_symbol.h"
#include "rocjitsu/code/rj_code.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace rocjitsu {
namespace {

// gfx90a instruction encodings (single 32-bit word unless noted).
constexpr uint32_t kSWaitcnt0 = 0xbf8c0000;      // s_waitcnt 0
constexpr uint32_t kSNop0 = 0xbf800000;          // s_nop 0
constexpr uint32_t kSSetpcS30S31 = 0xbe801d1e;   // s_setpc_b64 s[30:31]
constexpr uint32_t kSSetpcS0S1 = 0xbe801d00;     // s_setpc_b64 s[0:1]
constexpr uint32_t kSSwappcS30S31 = 0xbe9e1e1e;  // s_swappc_b64 s[30:31], s[30:31]
constexpr uint32_t kScratchLoadLo = 0xdc504000;  // scratch_load_dword v0, off, s0 (word 0)
constexpr uint32_t kScratchLoadHi = 0x00000000;  // ... (word 1)
constexpr uint32_t kSScratchLoadLo = 0xc0140000; // s_scratch_load_dword (SMEM, word 0)
constexpr uint32_t kSScratchLoadHi = 0x00000000; // ... (word 1)

// .text placement. Virtual address and file offset differ so the body's value
// range (used by the relocation check) is distinct from its file offset.
constexpr uint64_t kTextAddr = 0x1000;
constexpr uint64_t kTextOffset = 0x100;

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

// Build a minimal gfx90a ET_DYN ELF: one executable .text holding @p body, an
// optional .rela.text whose entries' r_offset values are @p reloc_voffsets
// (in the same virtual-address space as st_value), and a .shstrtab.
std::vector<uint8_t> make_elf(const std::vector<uint32_t> &body,
                              const std::vector<uint64_t> &reloc_voffsets = {}) {
  const bool has_rela = !reloc_voffsets.empty();
  const uint16_t shstrtab_index = static_cast<uint16_t>(has_rela ? 3 : 2);
  const uint16_t section_count = static_cast<uint16_t>(has_rela ? 4 : 3);

  std::vector<uint8_t> shstrtab{'\0'};
  const uint32_t text_name = add_name(shstrtab, ".text");
  const uint32_t rela_name = has_rela ? add_name(shstrtab, ".rela.text") : 0;
  const uint32_t shstrtab_name = add_name(shstrtab, ".shstrtab");

  const uint64_t text_size = body.size() * sizeof(uint32_t);
  uint64_t cursor = kTextOffset + text_size;

  std::vector<Elf64_Rela> relas;
  uint64_t rela_off = 0;
  if (has_rela) {
    for (const uint64_t v : reloc_voffsets) {
      Elf64_Rela r{};
      r.r_offset = v;
      relas.push_back(r);
    }
    rela_off = align_up(cursor, 8);
    cursor = rela_off + relas.size() * sizeof(Elf64_Rela);
  }
  const uint64_t shstrtab_off = cursor;
  cursor += shstrtab.size();
  const uint64_t shoff = align_up(cursor, 8);

  std::vector<uint8_t> image(shoff + section_count * sizeof(Elf64_Shdr), 0);

  Elf64_Ehdr ehdr{};
  std::memcpy(ehdr.e_ident, EI_MAGIC, EI_MAGIC_SIZE);
  ehdr.e_ident[EI_CLASS] = ELFCLASS64;
  ehdr.e_ident[EI_OSABI] = ELFOSABI_AMDGPU_HSA;
  ehdr.e_type = ET_DYN;
  ehdr.e_machine = EM_AMDGPU;
  ehdr.e_version = 1;
  ehdr.e_shoff = shoff;
  ehdr.e_flags = EF_AMDGPU_MACH_AMDGCN_GFX90A;
  ehdr.e_ehsize = sizeof(Elf64_Ehdr);
  ehdr.e_shentsize = sizeof(Elf64_Shdr);
  ehdr.e_shnum = section_count;
  ehdr.e_shstrndx = shstrtab_index;
  std::memcpy(image.data(), &ehdr, sizeof(ehdr));

  std::memcpy(image.data() + kTextOffset, body.data(), text_size);
  if (has_rela)
    std::memcpy(image.data() + rela_off, relas.data(), relas.size() * sizeof(Elf64_Rela));
  std::memcpy(image.data() + shstrtab_off, shstrtab.data(), shstrtab.size());

  std::vector<Elf64_Shdr> shdrs(section_count);
  shdrs[1].sh_name = text_name;
  shdrs[1].sh_type = SHT_PROGBITS;
  shdrs[1].sh_flags = SHF_ALLOC | SHF_EXECINSTR;
  shdrs[1].sh_addr = kTextAddr;
  shdrs[1].sh_offset = kTextOffset;
  shdrs[1].sh_size = text_size;
  shdrs[1].sh_addralign = sizeof(uint32_t);

  if (has_rela) {
    shdrs[2].sh_name = rela_name;
    shdrs[2].sh_type = SHT_RELA;
    shdrs[2].sh_offset = rela_off;
    shdrs[2].sh_size = relas.size() * sizeof(Elf64_Rela);
    shdrs[2].sh_link = 0;
    shdrs[2].sh_info = 1; // applies to .text
    shdrs[2].sh_entsize = sizeof(Elf64_Rela);
    shdrs[2].sh_addralign = 8;
  }

  shdrs[shstrtab_index].sh_name = shstrtab_name;
  shdrs[shstrtab_index].sh_type = SHT_STRTAB;
  shdrs[shstrtab_index].sh_offset = shstrtab_off;
  shdrs[shstrtab_index].sh_size = shstrtab.size();
  shdrs[shstrtab_index].sh_addralign = 1;

  std::memcpy(image.data() + shoff, shdrs.data(), shdrs.size() * sizeof(Elf64_Shdr));
  return image;
}

// A ResolvedProbeSymbol covering the whole body at the start of .text.
ResolvedProbeSymbol whole_body_symbol(const std::vector<uint32_t> &body) {
  ResolvedProbeSymbol sym;
  sym.name = "rj_probe";
  sym.section_index = 1;
  sym.st_value = kTextAddr;
  sym.body_file_offset = kTextOffset;
  sym.body_size = body.size() * sizeof(uint32_t);
  return sym;
}

TEST(ProbeCallableTest, BuildsNopProbe) {
  const std::vector<uint32_t> body{kSWaitcnt0, kSSetpcS30S31};
  const auto image = make_elf(body);
  AmdGpuCodeObject obj(image.data(), image.size());

  std::string err;
  const auto callable =
      build_probe_callable(obj, whole_body_symbol(body), ROCJITSU_CODE_ARCH_CDNA2, &err);
  ASSERT_TRUE(callable.has_value()) << err;
  EXPECT_EQ(callable->symbol, "rj_probe");
  EXPECT_EQ(callable->arch, ROCJITSU_CODE_ARCH_CDNA2);
  EXPECT_EQ(callable->cc, ProbeCallingConvention::AmdGpuFuncNoArgsReturnS30S31);
  EXPECT_EQ(callable->body_words, body);
  EXPECT_EQ(callable->output_text_offset, 0u); // assigned by a later layout step.
}

TEST(ProbeCallableTest, RejectsBodyWithCall) {
  const std::vector<uint32_t> body{kSSwappcS30S31, kSSetpcS30S31};
  const auto image = make_elf(body);
  AmdGpuCodeObject obj(image.data(), image.size());

  std::string err;
  EXPECT_FALSE(build_probe_callable(obj, whole_body_symbol(body), ROCJITSU_CODE_ARCH_CDNA2, &err)
                   .has_value());
  EXPECT_NE(err.find("call"), std::string::npos) << err;
}

TEST(ProbeCallableTest, RejectsScratchAccess) {
  const std::vector<uint32_t> body{kScratchLoadLo, kScratchLoadHi, kSSetpcS30S31};
  const auto image = make_elf(body);
  AmdGpuCodeObject obj(image.data(), image.size());

  std::string err;
  EXPECT_FALSE(build_probe_callable(obj, whole_body_symbol(body), ROCJITSU_CODE_ARCH_CDNA2, &err)
                   .has_value());
  EXPECT_NE(err.find("scratch"), std::string::npos) << err;
}

TEST(ProbeCallableTest, RejectsSmemScratchAccess) {
  const std::vector<uint32_t> body{kSScratchLoadLo, kSScratchLoadHi, kSSetpcS30S31};
  const auto image = make_elf(body);
  AmdGpuCodeObject obj(image.data(), image.size());

  std::string err;
  EXPECT_FALSE(build_probe_callable(obj, whole_body_symbol(body), ROCJITSU_CODE_ARCH_CDNA2, &err)
                   .has_value());
  EXPECT_NE(err.find("scratch"), std::string::npos) << err;
}

TEST(ProbeCallableTest, RejectsMissingReturn) {
  const std::vector<uint32_t> body{kSNop0};
  const auto image = make_elf(body);
  AmdGpuCodeObject obj(image.data(), image.size());

  std::string err;
  EXPECT_FALSE(build_probe_callable(obj, whole_body_symbol(body), ROCJITSU_CODE_ARCH_CDNA2, &err)
                   .has_value());
  EXPECT_NE(err.find("s_setpc_b64"), std::string::npos) << err;
}

TEST(ProbeCallableTest, RejectsWrongReturnRegister) {
  // Ends in s_setpc_b64 but through s[0:1], not the s[30:31] link pair.
  const std::vector<uint32_t> body{kSWaitcnt0, kSSetpcS0S1};
  const auto image = make_elf(body);
  AmdGpuCodeObject obj(image.data(), image.size());

  std::string err;
  EXPECT_FALSE(build_probe_callable(obj, whole_body_symbol(body), ROCJITSU_CODE_ARCH_CDNA2, &err)
                   .has_value());
  EXPECT_NE(err.find("s_setpc_b64"), std::string::npos) << err;
}

TEST(ProbeCallableTest, RejectsRelocationInBody) {
  const std::vector<uint32_t> body{kSWaitcnt0, kSSetpcS30S31};
  // Relocation lands on the second body word (vaddr space).
  const auto image = make_elf(body, {kTextAddr + sizeof(uint32_t)});
  AmdGpuCodeObject obj(image.data(), image.size());

  std::string err;
  EXPECT_FALSE(build_probe_callable(obj, whole_body_symbol(body), ROCJITSU_CODE_ARCH_CDNA2, &err)
                   .has_value());
  EXPECT_NE(err.find("relocation"), std::string::npos) << err;
}

TEST(ProbeCallableTest, AcceptsRelocationOutsideBody) {
  const std::vector<uint32_t> body{kSWaitcnt0, kSSetpcS30S31};
  // Relocation at the byte just past the body end is not "inside" it.
  const auto image = make_elf(body, {kTextAddr + body.size() * sizeof(uint32_t)});
  AmdGpuCodeObject obj(image.data(), image.size());

  std::string err;
  const auto callable =
      build_probe_callable(obj, whole_body_symbol(body), ROCJITSU_CODE_ARCH_CDNA2, &err);
  ASSERT_TRUE(callable.has_value()) << err;
  EXPECT_EQ(callable->cc, ProbeCallingConvention::AmdGpuFuncNoArgsReturnS30S31);
}

TEST(ProbeCallableTest, RejectsNonDwordBodySize) {
  const std::vector<uint32_t> body{kSWaitcnt0, kSSetpcS30S31};
  const auto image = make_elf(body);
  AmdGpuCodeObject obj(image.data(), image.size());

  ResolvedProbeSymbol sym = whole_body_symbol(body);
  sym.body_size = 6; // not a multiple of 4.

  std::string err;
  EXPECT_FALSE(build_probe_callable(obj, sym, ROCJITSU_CODE_ARCH_CDNA2, &err).has_value());
  EXPECT_NE(err.find("multiple of 4"), std::string::npos) << err;
}

TEST(ProbeCallableTest, RejectsBodyPastImage) {
  const std::vector<uint32_t> body{kSWaitcnt0, kSSetpcS30S31};
  const auto image = make_elf(body);
  AmdGpuCodeObject obj(image.data(), image.size());

  ResolvedProbeSymbol sym = whole_body_symbol(body);
  sym.body_file_offset = image.size(); // starts at end of image.
  sym.body_size = 8;

  std::string err;
  EXPECT_FALSE(build_probe_callable(obj, sym, ROCJITSU_CODE_ARCH_CDNA2, &err).has_value());
  EXPECT_NE(err.find("past end of image"), std::string::npos) << err;
}

TEST(ProbeCallableTest, RejectsStValueOverflow) {
  // A hostile st_value that would wrap st_value + body_size must be rejected so
  // the relocation range check cannot be bypassed. body_file_offset stays valid.
  const std::vector<uint32_t> body{kSWaitcnt0, kSSetpcS30S31};
  const auto image = make_elf(body);
  AmdGpuCodeObject obj(image.data(), image.size());

  ResolvedProbeSymbol sym = whole_body_symbol(body);
  sym.st_value = UINT64_MAX; // st_value + body_size overflows.

  std::string err;
  EXPECT_FALSE(build_probe_callable(obj, sym, ROCJITSU_CODE_ARCH_CDNA2, &err).has_value());
  EXPECT_NE(err.find("overflow"), std::string::npos) << err;
}

} // namespace
} // namespace rocjitsu
