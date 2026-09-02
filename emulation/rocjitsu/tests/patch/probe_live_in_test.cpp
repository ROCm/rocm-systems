// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Synthetic, always-on coverage for analyze_probe_live_ins().
//
// The ELF builder mirrors probe_callable_test.cpp: analyze_probe_live_ins()
// takes a ResolvedProbeSymbol by value, so these tests fabricate the symbol
// directly and only need a minimal ELF carrying the body words in an
// executable .text.
//
// Instruction encodings below are gfx90a ground truth captured from
// `llvm-mc -arch=amdgcn -mcpu=gfx90a -show-encoding`.

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/patch/probe_callable.h"
#include "rocjitsu/code/patch/probe_live_in.h"
#include "rocjitsu/code/patch/probe_symbol.h"
#include "rocjitsu/code/rj_code.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace rocjitsu {
namespace {

// gfx90a instruction encodings, one 32-bit word each.
constexpr uint32_t kSWaitcnt0 = 0xbf8c0000;          // s_waitcnt 0
constexpr uint32_t kSSetpcS30S31 = 0xbe801d1e;       // s_setpc_b64 s[30:31]
constexpr uint32_t kVMovV0V31 = 0x7e00031f;          // v_mov_b32 v0, v31
constexpr uint32_t kVMovV31Zero = 0x7e3e0280;        // v_mov_b32 v31, 0
constexpr uint32_t kVMovV0S4 = 0x7e000204;           // v_mov_b32 v0, s4
constexpr uint32_t kSMovS4Zero = 0xbe840080;         // s_mov_b32 s4, 0
constexpr uint32_t kSCbranchScc0Skip1 = 0xbf840001;  // s_cbranch_scc0 over one word
constexpr uint32_t kSSetGprIdxOnS0Src0 = 0xbf110100; // s_set_gpr_idx_on s0, gpr_idx(SRC0)

constexpr uint64_t kTextAddr = 0x1000;
constexpr uint64_t kTextOffset = 0x100;

// TODO: fold this and the seven other identical test-local copies into
// util::align_up (util/bit.h). Matches its siblings rather than diverging alone.
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

// Minimal gfx90a ET_DYN ELF: one executable .text holding @p body, plus a
// .shstrtab. Section index 1 is .text, which whole_body_symbol() names.
std::vector<uint8_t> make_elf(const std::vector<uint32_t> &body,
                              uint32_t machine = EF_AMDGPU_MACH_AMDGCN_GFX90A) {
  constexpr uint16_t kShstrtabIndex = 2;
  constexpr uint16_t kSectionCount = 3;

  std::vector<uint8_t> shstrtab{'\0'};
  const uint32_t text_name = add_name(shstrtab, ".text");
  const uint32_t shstrtab_name = add_name(shstrtab, ".shstrtab");

  const uint64_t text_size = body.size() * sizeof(uint32_t);
  const uint64_t shstrtab_off = kTextOffset + text_size;
  const uint64_t shoff = align_up(shstrtab_off + shstrtab.size(), 8);

  std::vector<uint8_t> image(shoff + kSectionCount * sizeof(Elf64_Shdr), 0);

  Elf64_Ehdr ehdr{};
  std::memcpy(ehdr.e_ident, EI_MAGIC, EI_MAGIC_SIZE);
  ehdr.e_ident[EI_CLASS] = ELFCLASS64;
  ehdr.e_ident[EI_OSABI] = ELFOSABI_AMDGPU_HSA;
  ehdr.e_type = ET_DYN;
  ehdr.e_machine = EM_AMDGPU;
  ehdr.e_version = 1;
  ehdr.e_shoff = shoff;
  ehdr.e_flags = machine;
  ehdr.e_ehsize = sizeof(Elf64_Ehdr);
  ehdr.e_shentsize = sizeof(Elf64_Shdr);
  ehdr.e_shnum = kSectionCount;
  ehdr.e_shstrndx = kShstrtabIndex;
  std::memcpy(image.data(), &ehdr, sizeof(ehdr));

  std::memcpy(image.data() + kTextOffset, body.data(), text_size);
  std::memcpy(image.data() + shstrtab_off, shstrtab.data(), shstrtab.size());

  std::vector<Elf64_Shdr> shdrs(kSectionCount);
  shdrs[1].sh_name = text_name;
  shdrs[1].sh_type = SHT_PROGBITS;
  shdrs[1].sh_flags = SHF_ALLOC | SHF_EXECINSTR;
  shdrs[1].sh_addr = kTextAddr;
  shdrs[1].sh_offset = kTextOffset;
  shdrs[1].sh_size = text_size;
  shdrs[1].sh_addralign = sizeof(uint32_t);

  shdrs[kShstrtabIndex].sh_name = shstrtab_name;
  shdrs[kShstrtabIndex].sh_type = SHT_STRTAB;
  shdrs[kShstrtabIndex].sh_offset = shstrtab_off;
  shdrs[kShstrtabIndex].sh_size = shstrtab.size();
  shdrs[kShstrtabIndex].sh_addralign = 1;

  std::memcpy(image.data() + shoff, shdrs.data(), shdrs.size() * sizeof(Elf64_Shdr));
  return image;
}

// Same shape as make_elf but with a second executable section also named
// ".text", which is what CodeObject::text_sections() keys on.
std::vector<uint8_t> make_two_text_elf(const std::vector<uint32_t> &body) {
  constexpr uint16_t kShstrtabIndex = 3;
  constexpr uint16_t kSectionCount = 4;

  std::vector<uint8_t> shstrtab{'\0'};
  const uint32_t text_name = add_name(shstrtab, ".text");
  const uint32_t shstrtab_name = add_name(shstrtab, ".shstrtab");

  const uint64_t text_size = body.size() * sizeof(uint32_t);
  const uint64_t second_off = kTextOffset + text_size;
  const uint64_t shstrtab_off = second_off + text_size;
  const uint64_t shoff = align_up(shstrtab_off + shstrtab.size(), 8);

  std::vector<uint8_t> image(shoff + kSectionCount * sizeof(Elf64_Shdr), 0);

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
  ehdr.e_shnum = kSectionCount;
  ehdr.e_shstrndx = kShstrtabIndex;
  std::memcpy(image.data(), &ehdr, sizeof(ehdr));

  std::memcpy(image.data() + kTextOffset, body.data(), text_size);
  std::memcpy(image.data() + second_off, body.data(), text_size);
  std::memcpy(image.data() + shstrtab_off, shstrtab.data(), shstrtab.size());

  std::vector<Elf64_Shdr> shdrs(kSectionCount);
  for (uint16_t i = 1; i <= 2; ++i) {
    shdrs[i].sh_name = text_name;
    shdrs[i].sh_type = SHT_PROGBITS;
    shdrs[i].sh_flags = SHF_ALLOC | SHF_EXECINSTR;
    shdrs[i].sh_addr = kTextAddr + (i - 1) * text_size;
    shdrs[i].sh_offset = i == 1 ? kTextOffset : second_off;
    shdrs[i].sh_size = text_size;
    shdrs[i].sh_addralign = sizeof(uint32_t);
  }

  shdrs[kShstrtabIndex].sh_name = shstrtab_name;
  shdrs[kShstrtabIndex].sh_type = SHT_STRTAB;
  shdrs[kShstrtabIndex].sh_offset = shstrtab_off;
  shdrs[kShstrtabIndex].sh_size = shstrtab.size();
  shdrs[kShstrtabIndex].sh_addralign = 1;

  std::memcpy(image.data() + shoff, shdrs.data(), shdrs.size() * sizeof(Elf64_Shdr));
  return image;
}

ResolvedProbeSymbol whole_body_symbol(const std::vector<uint32_t> &body) {
  ResolvedProbeSymbol sym;
  sym.name = "rj_probe";
  sym.section_index = 1;
  sym.st_value = kTextAddr;
  sym.body_file_offset = kTextOffset;
  sym.body_size = body.size() * sizeof(uint32_t);
  return sym;
}

// Run the analysis over @p body under the only supported convention.
std::optional<RegisterSet> live_ins(const std::vector<uint32_t> &body, std::string *err) {
  const auto image = make_elf(body);
  const AmdGpuCodeObject obj(image.data(), image.size());
  return analyze_probe_live_ins(obj, whole_body_symbol(body), ROCJITSU_CODE_ARCH_CDNA2,
                                ProbeCallingConvention::AmdGpuFuncNoArgsReturnS30S31, err);
}

TEST(ProbeLiveInTest, NopProbeHasNoLiveIns) {
  std::string err;
  const auto set = live_ins({kSWaitcnt0, kSSetpcS30S31}, &err);
  ASSERT_TRUE(set.has_value()) << err;
  EXPECT_TRUE(set->none()) << format_register_set(*set);
}

// The concrete input the gate exists for: workitem_id_x arrives in v31, put
// there by the kernel's own prologue. No convention declares it, so the caller
// has nothing to materialize it from.
TEST(ProbeLiveInTest, ReportsWorkitemIdReadFromV31) {
  std::string err;
  const auto set = live_ins({kVMovV0V31, kSSetpcS30S31}, &err);
  ASSERT_TRUE(set.has_value()) << err;
  EXPECT_TRUE(set->contains(RegisterRef{RegClass::VGPR, 31, 1}));
  EXPECT_EQ(format_register_set(*set), "v31");
}

// The same read, but the probe defines v31 itself first, so it is not an input.
// This is what the EXEC-masked-def kill rule would otherwise get wrong: without
// LivenessAnalysisOptions::exec_masked_defs_kill the def is not a kill and v31
// shows up in the footprint here too.
TEST(ProbeLiveInTest, VgprDefinedBeforeUseIsNotALiveIn) {
  std::string err;
  const auto set = live_ins({kVMovV31Zero, kVMovV0V31, kSSetpcS30S31}, &err);
  ASSERT_TRUE(set.has_value()) << err;
  EXPECT_TRUE(set->none()) << format_register_set(*set);
}

// A linear "used before defined" scan would miss this: it sees the def at word 1
// before the use at word 2. The def is under a forward branch, so on the taken
// path v31 reaches the use undefined and is a real input.
TEST(ProbeLiveInTest, ReportsDefUnderForwardBranch) {
  std::string err;
  const auto set = live_ins({kSCbranchScc0Skip1, kVMovV31Zero, kVMovV0V31, kSSetpcS30S31}, &err);
  ASSERT_TRUE(set.has_value()) << err;
  EXPECT_EQ(format_register_set(*set), "v31");
}

// Scalars are reported the same way. Nothing the framework emits at the site
// writes s4.
TEST(ProbeLiveInTest, ReportsScalarReadBeforeDefine) {
  std::string err;
  const auto set = live_ins({kVMovV0S4, kSSetpcS30S31}, &err);
  ASSERT_TRUE(set.has_value()) << err;
  EXPECT_EQ(format_register_set(*set), "s4");
}

TEST(ProbeLiveInTest, ScalarDefinedBeforeUseIsNotALiveIn) {
  std::string err;
  const auto set = live_ins({kSMovS4Zero, kVMovV0S4, kSSetpcS30S31}, &err);
  ASSERT_TRUE(set.has_value()) << err;
  EXPECT_TRUE(set->none()) << format_register_set(*set);
}

// s[30:31] is read by the body's own s_setpc_b64, but the trampoline always
// supplies it, so it must not appear in the footprint the caller has to satisfy.
TEST(ProbeLiveInTest, LinkPairIsNotALiveIn) {
  std::string err;
  const auto set = live_ins({kSWaitcnt0, kSSetpcS30S31}, &err);
  ASSERT_TRUE(set.has_value()) << err;
  EXPECT_FALSE(set->contains(RegisterRef{RegClass::SGPR, 30, 1}));
  EXPECT_FALSE(set->contains(RegisterRef{RegClass::SGPR, 31, 1}));
}

// GPR indexing adds m0 to an encoded operand index, so the decoded operands stop
// naming every register the body reads. Missing a live-in is the direction that
// lets a bad probe through, so this fails closed rather than reporting a set.
TEST(ProbeLiveInTest, RejectsGprIndexedVgprAccess) {
  std::string err;
  EXPECT_FALSE(live_ins({kSSetGprIdxOnS0Src0, kVMovV0V31, kSSetpcS30S31}, &err).has_value());
  EXPECT_NE(err.find("GPR-indexed"), std::string::npos) << err;
}

// gfx1250 resolves an encoded VGPR operand through MODE.VGPR_MSB, whose value at
// an instrumentation anchor is unknown and which the call envelope does not set.
// Seeding a zero state anyway silently mislabels physical registers, so refuse.
TEST(ProbeLiveInTest, RejectsCdna5) {
  const std::vector<uint32_t> body{kSWaitcnt0, kSSetpcS30S31};
  const auto image = make_elf(body, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  const AmdGpuCodeObject obj(image.data(), image.size());

  std::string err;
  EXPECT_FALSE(analyze_probe_live_ins(obj, whole_body_symbol(body), ROCJITSU_CODE_ARCH_CDNA5,
                                      ProbeCallingConvention::AmdGpuFuncNoArgsReturnS30S31, &err)
                   .has_value());
  EXPECT_NE(err.find("VGPR_MSB"), std::string::npos) << err;
}

TEST(ProbeLiveInTest, RejectsUnknownCallingConvention) {
  const std::vector<uint32_t> body{kSWaitcnt0, kSSetpcS30S31};
  const auto image = make_elf(body);
  const AmdGpuCodeObject obj(image.data(), image.size());

  std::string err;
  EXPECT_FALSE(analyze_probe_live_ins(obj, whole_body_symbol(body), ROCJITSU_CODE_ARCH_CDNA2,
                                      ProbeCallingConvention::Unknown, &err)
                   .has_value());
  EXPECT_NE(err.find("calling convention"), std::string::npos) << err;
}

TEST(ProbeLiveInTest, RejectsSymbolOutsideText) {
  const std::vector<uint32_t> body{kSWaitcnt0, kSSetpcS30S31};
  const auto image = make_elf(body);
  const AmdGpuCodeObject obj(image.data(), image.size());

  ResolvedProbeSymbol sym = whole_body_symbol(body);
  sym.section_index = 2; // .shstrtab, not .text.

  std::string err;
  EXPECT_FALSE(analyze_probe_live_ins(obj, sym, ROCJITSU_CODE_ARCH_CDNA2,
                                      ProbeCallingConvention::AmdGpuFuncNoArgsReturnS30S31, &err)
                   .has_value());
  EXPECT_NE(err.find("not in the code object's .text"), std::string::npos) << err;
}

TEST(ProbeLiveInTest, RejectsEntryThatDoesNotStartABlock) {
  const std::vector<uint32_t> body{kSWaitcnt0, kSSetpcS30S31};
  const auto image = make_elf(body);
  const AmdGpuCodeObject obj(image.data(), image.size());

  ResolvedProbeSymbol sym = whole_body_symbol(body);
  sym.body_file_offset = kTextOffset + 2; // mid-instruction, so no block starts here.
  sym.body_size = 4;

  std::string err;
  EXPECT_FALSE(analyze_probe_live_ins(obj, sym, ROCJITSU_CODE_ARCH_CDNA2,
                                      ProbeCallingConvention::AmdGpuFuncNoArgsReturnS30S31, &err)
                   .has_value());
  EXPECT_NE(err.find("basic block"), std::string::npos) << err;
}

// BasicBlock::build() restarts its byte offsets per section, so a block offset
// would not name a unique body. Instrumentor's own multi-.text check covers only
// the destination object, so a multi-.text probe object reaches this from a real
// input rather than only from a fabricated one.
TEST(ProbeLiveInTest, RejectsMultipleTextSections) {
  const std::vector<uint32_t> body{kSWaitcnt0, kSSetpcS30S31};
  const auto image = make_two_text_elf(body);
  const AmdGpuCodeObject obj(image.data(), image.size());
  ASSERT_EQ(obj.text_sections().size(), 2u) << "fixture did not produce two .text sections";

  std::string err;
  EXPECT_FALSE(analyze_probe_live_ins(obj, whole_body_symbol(body), ROCJITSU_CODE_ARCH_CDNA2,
                                      ProbeCallingConvention::AmdGpuFuncNoArgsReturnS30S31, &err)
                   .has_value());
  EXPECT_NE(err.find("exactly one .text"), std::string::npos) << err;
}

// The .text-relative entry offset is body_file_offset - sh_offset; guard the
// subtraction rather than wrap it.
TEST(ProbeLiveInTest, RejectsBodyBeforeItsTextSection) {
  const std::vector<uint32_t> body{kSWaitcnt0, kSSetpcS30S31};
  const auto image = make_elf(body);
  const AmdGpuCodeObject obj(image.data(), image.size());

  ResolvedProbeSymbol sym = whole_body_symbol(body);
  sym.body_file_offset = 0; // ahead of .text at kTextOffset.

  std::string err;
  EXPECT_FALSE(analyze_probe_live_ins(obj, sym, ROCJITSU_CODE_ARCH_CDNA2,
                                      ProbeCallingConvention::AmdGpuFuncNoArgsReturnS30S31, &err)
                   .has_value());
  EXPECT_NE(err.find("starts before"), std::string::npos) << err;
}

// An arch with no registered target has no decoder factory. The object carries
// no concrete target either, so the arch overload is the one consulted.
TEST(ProbeLiveInTest, RejectsMissingDecoder) {
  const std::vector<uint32_t> body{kSWaitcnt0, kSSetpcS30S31};
  const auto image = make_elf(body, /*machine=*/0);
  const AmdGpuCodeObject obj(image.data(), image.size());
  ASSERT_EQ(obj.target_id(), ROCJITSU_CODE_TARGET_INVALID);

  std::string err;
  EXPECT_FALSE(analyze_probe_live_ins(obj, whole_body_symbol(body), ROCJITSU_CODE_ARCH_INVALID,
                                      ProbeCallingConvention::AmdGpuFuncNoArgsReturnS30S31, &err)
                   .has_value());
  EXPECT_NE(err.find("no decoder"), std::string::npos) << err;
}

// BasicBlock::build() decodes the whole section, so one bad word anywhere in
// .text fails the build -- the probe body itself need not be the bad part.
TEST(ProbeLiveInTest, RejectsUndecodableText) {
  const std::vector<uint32_t> body{0xffffffffu, kSSetpcS30S31};
  const auto image = make_elf(body);
  const AmdGpuCodeObject obj(image.data(), image.size());

  std::string err;
  EXPECT_FALSE(analyze_probe_live_ins(obj, whole_body_symbol(body), ROCJITSU_CODE_ARCH_CDNA2,
                                      ProbeCallingConvention::AmdGpuFuncNoArgsReturnS30S31, &err)
                   .has_value());
  EXPECT_NE(err.find("failed to decode"), std::string::npos) << err;
}

TEST(ProbeLiveInTest, FormatsRegisterSetAcrossClasses) {
  RegisterSet regs;
  regs.expand(RegisterRef{RegClass::SGPR, 4, 1});
  regs.expand(RegisterRef{RegClass::VGPR, 0, 2});
  EXPECT_EQ(format_register_set(regs), "s4, v0, v1");
}

} // namespace
} // namespace rocjitsu
