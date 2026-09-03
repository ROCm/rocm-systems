// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "decode_test_util.h"
#include "rocjitsu/code/patch/instrumentor.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/builders/spill_builders.h"
#include "rocjitsu/code/kernel_descriptor_scan.h"
#include "rocjitsu/code/patch/probe_clobber.h"
#include "rocjitsu/code/patch/trampoline_builder.h"
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/register_set.h"

#include "../dbi_test_util.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/AMDHSAKernelDescriptor.h"
RJ_DIAGNOSTIC_POP

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rocjitsu {
namespace {

using namespace rocjitsu::test;

//==============================================================================
// Synthetic Instruction subclass for fast validator unit tests.
//==============================================================================

class TestInstruction : public Instruction {
public:
  TestInstruction(std::string_view mnemonic, int size, uint64_t flags,
                  std::optional<int64_t> branch_delta, const uint32_t *raw_enc)
      : Instruction(mnemonic, nullptr), branch_delta_(branch_delta) {
    size_ = size;
    flags_ = flags;
    raw_encoding_ = raw_enc;
  }
  std::optional<int64_t> branch_offset_bytes() const override { return branch_delta_; }

private:
  std::optional<int64_t> branch_delta_;
};

std::vector<uint8_t> dummy_text(size_t size = 16) { return std::vector<uint8_t>(size, 0xCC); }

//==============================================================================
// Section 1: validate_anchor (synthetic)
//==============================================================================

TEST(Validator, AcceptsFourByteRelocatable) {
  static constexpr uint32_t kRaw = 0xDEADBEEFu;
  TestInstruction anchor("v_add_f32_e32", 4, /*flags=*/0, std::nullopt, &kRaw);
  auto text = dummy_text();
  InstrumentationPoint pt;

  std::string err;
  auto site =
      validate_anchor(anchor, /*anchor_offset=*/0, text, pt, ROCJITSU_CODE_ARCH_CDNA4, &err);
  ASSERT_TRUE(site.has_value()) << err;
  EXPECT_EQ(site->original_size, 4u);
  EXPECT_EQ(site->original_bytes.size(), 4u);
  EXPECT_EQ(site->mnemonic, "v_add_f32_e32");
  EXPECT_EQ(site->anchor_offset, 0u);
}

TEST(Validator, AcceptsEightByteRelocatable) {
  static constexpr uint32_t kRaw[2] = {0xDEADBEEFu, 0xCAFEF00Du};
  TestInstruction anchor("buffer_load_dword", 8, /*flags=*/0, std::nullopt, kRaw);
  auto text = dummy_text();
  InstrumentationPoint pt;

  std::string err;
  auto site = validate_anchor(anchor, 0, text, pt, ROCJITSU_CODE_ARCH_CDNA4, &err);
  ASSERT_TRUE(site.has_value()) << err;
  EXPECT_EQ(site->original_size, 8u);
  EXPECT_EQ(site->original_bytes.size(), 8u);
}

TEST(Validator, RejectsNonZeroFilterFlags) {
  static constexpr uint32_t kRaw = 0xDEADBEEFu;
  TestInstruction anchor("v_add_f32_e32", 4, 0, std::nullopt, &kRaw);
  auto text = dummy_text();
  InstrumentationPoint pt;
  pt.filter_flags = 1;
  std::string err;
  EXPECT_FALSE(validate_anchor(anchor, 0, text, pt, ROCJITSU_CODE_ARCH_CDNA4, &err).has_value());
  EXPECT_FALSE(err.empty());
}

TEST(Validator, RejectsControlFlowFlags) {
  static constexpr uint32_t kRaw = 0xCAFEBABEu;
  auto text = dummy_text();
  InstrumentationPoint pt;

  const std::array<std::pair<const char *, uint64_t>, 5> cases = {{
      {"BRANCH", BRANCH},
      {"COND_BRANCH", COND_BRANCH},
      {"INDIRECT_BRANCH", INDIRECT_BRANCH},
      {"INDIRECT_CALL", INDIRECT_CALL},
      {"PROGRAM_TERMINATOR", PROGRAM_TERMINATOR},
  }};
  for (auto [name, flag] : cases) {
    TestInstruction anchor("test_inst", 4, flag, std::nullopt, &kRaw);
    std::string err;
    EXPECT_FALSE(validate_anchor(anchor, 0, text, pt, ROCJITSU_CODE_ARCH_CDNA4, &err).has_value())
        << "flag " << name << " should be rejected";
    EXPECT_FALSE(err.empty());
  }
}

TEST(Validator, RejectsNonNullBranchOffsetBytes) {
  static constexpr uint32_t kRaw = 0xCAFEBABEu;
  TestInstruction anchor("s_cbranch_scc0", 4, /*flags=*/0, /*branch_delta=*/42, &kRaw);
  auto text = dummy_text();
  InstrumentationPoint pt;
  std::string err;
  EXPECT_FALSE(validate_anchor(anchor, 0, text, pt, ROCJITSU_CODE_ARCH_CDNA4, &err).has_value());
  EXPECT_FALSE(err.empty());
}

TEST(Validator, RejectsNullRawEncoding) {
  TestInstruction anchor("opaque", 4, /*flags=*/0, std::nullopt, /*raw_enc=*/nullptr);
  auto text = dummy_text();
  InstrumentationPoint pt;
  std::string err;
  EXPECT_FALSE(validate_anchor(anchor, 0, text, pt, ROCJITSU_CODE_ARCH_CDNA4, &err).has_value());
  EXPECT_FALSE(err.empty());
}

TEST(Validator, RejectsUnsupportedSize) {
  static constexpr uint32_t kRaw[3] = {0, 0, 0};
  TestInstruction anchor("hypothetical_12byte", 12, 0, std::nullopt, kRaw);
  auto text = dummy_text();
  InstrumentationPoint pt;
  std::string err;
  EXPECT_FALSE(validate_anchor(anchor, 0, text, pt, ROCJITSU_CODE_ARCH_CDNA4, &err).has_value());
  EXPECT_FALSE(err.empty());
}

TEST(Validator, RejectsUnalignedOffset) {
  static constexpr uint32_t kRaw = 0xDEADBEEFu;
  TestInstruction anchor("v_add_f32_e32", 4, 0, std::nullopt, &kRaw);
  auto text = dummy_text();
  InstrumentationPoint pt;
  std::string err;
  EXPECT_FALSE(
      validate_anchor(anchor, /*anchor_offset=*/2, text, pt, ROCJITSU_CODE_ARCH_CDNA4, &err)
          .has_value());
  EXPECT_FALSE(err.empty());
}

TEST(Validator, RejectsOutOfBoundsOffset) {
  static constexpr uint32_t kRaw[2] = {0, 0};
  TestInstruction anchor("buffer_load_dword", 8, 0, std::nullopt, kRaw);
  auto text = dummy_text(/*size=*/8);
  InstrumentationPoint pt;
  std::string err;
  EXPECT_FALSE(
      validate_anchor(anchor, /*anchor_offset=*/4, text, pt, ROCJITSU_CODE_ARCH_CDNA4, &err)
          .has_value());
  EXPECT_FALSE(err.empty());
}

TEST(Validator, RejectsOffsetThatWouldOverflow) {
  // Pure predicate: huge offset must fail closed, not wrap the bounds check.
  static constexpr uint32_t kRaw = 0xDEADBEEFu;
  TestInstruction anchor("v_add_f32_e32", 4, 0, std::nullopt, &kRaw);
  auto text = dummy_text();
  // UINT64_MAX - 3 is dword aligned (UINT64_MAX = 2^64 - 1 ≡ 3 mod 4) and
  // anchor_offset + 4 would wrap to 0 with the old additive check.
  const uint64_t huge_offset = std::numeric_limits<uint64_t>::max() - 3;
  std::string err;
  EXPECT_FALSE(is_relocatable_anchor(anchor, huge_offset, text, ROCJITSU_CODE_ARCH_CDNA4, &err));
  EXPECT_FALSE(err.empty());
}

TEST(Validator, RejectsNonBeforeInstKind) {
  static constexpr uint32_t kRaw = 0xDEADBEEFu;
  TestInstruction anchor("v_add_f32_e32", 4, 0, std::nullopt, &kRaw);
  auto text = dummy_text();

  for (auto kind : {InstrumentationKind::AfterInst, InstrumentationKind::BlockEntry,
                    InstrumentationKind::BlockExit}) {
    InstrumentationPoint pt;
    pt.kind = kind;
    std::string err;
    EXPECT_FALSE(validate_anchor(anchor, 0, text, pt, ROCJITSU_CODE_ARCH_CDNA4, &err).has_value())
        << "kind != BeforeInst must be rejected";
    EXPECT_FALSE(err.empty());
  }
}

TEST(Validator, RejectsProbeObjWithoutSymbol) {
  static constexpr uint32_t kRaw = 0xDEADBEEFu;
  TestInstruction anchor("v_add_f32_e32", 4, 0, std::nullopt, &kRaw);
  auto text = dummy_text();
  InstrumentationPoint pt;
  static const AmdGpuCodeObject *kSentinel = reinterpret_cast<const AmdGpuCodeObject *>(0x1);
  pt.probe_obj = kSentinel; // symbol left empty

  std::string err;
  EXPECT_FALSE(validate_anchor(anchor, 0, text, pt, ROCJITSU_CODE_ARCH_CDNA4, &err).has_value());
  EXPECT_FALSE(err.empty());
}

TEST(Validator, RejectsProbeSymbolWithoutObj) {
  static constexpr uint32_t kRaw = 0xDEADBEEFu;
  TestInstruction anchor("v_add_f32_e32", 4, 0, std::nullopt, &kRaw);
  auto text = dummy_text();
  InstrumentationPoint pt;
  pt.probe_symbol = "my_probe"; // probe_obj left null

  std::string err;
  EXPECT_FALSE(validate_anchor(anchor, 0, text, pt, ROCJITSU_CODE_ARCH_CDNA4, &err).has_value());
  EXPECT_FALSE(err.empty());
}

TEST(Validator, AcceptsProbeObjWithSymbol) {
  static constexpr uint32_t kRaw = 0xDEADBEEFu;
  TestInstruction anchor("v_add_f32_e32", 4, 0, std::nullopt, &kRaw);
  auto text = dummy_text();
  InstrumentationPoint pt;
  static const AmdGpuCodeObject *kSentinel = reinterpret_cast<const AmdGpuCodeObject *>(0x1);
  pt.probe_obj = kSentinel;
  pt.probe_symbol = "my_probe";

  std::string err;
  EXPECT_TRUE(validate_anchor(anchor, 0, text, pt, ROCJITSU_CODE_ARCH_CDNA4, &err).has_value())
      << err;
}

TEST(Validator, RejectsForceFullExec) {
  static constexpr uint32_t kRaw = 0xDEADBEEFu;
  TestInstruction anchor("v_add_f32_e32", 4, 0, std::nullopt, &kRaw);
  auto text = dummy_text();
  InstrumentationPoint pt;
  pt.force_full_exec = true;

  std::string err;
  EXPECT_FALSE(validate_anchor(anchor, 0, text, pt, ROCJITSU_CODE_ARCH_CDNA4, &err).has_value());
  EXPECT_FALSE(err.empty());
}

TEST(Validator, RejectsDenylistedMnemonic) {
  static constexpr uint32_t kRaw = 0xCAFEBABEu;
  auto text = dummy_text();
  InstrumentationPoint pt;

  // PC-relative semantics that may not surface in flags / branch_offset_bytes.
  // Includes gfx1250's renamed _i64 family (s_getpc_b64 -> s_get_pc_i64, etc.);
  // s_rfe_b64 / s_rfe_i64 exercise the s_rfe_ prefix match.
  for (const char *m :
       {"s_getpc_b64", "s_call_b64", "s_setpc_b64", "s_swappc_b64", "s_rfe_b64", "s_get_pc_i64",
        "s_call_i64", "s_set_pc_i64", "s_swap_pc_i64", "s_rfe_i64"}) {
    TestInstruction anchor(m, 4, /*flags=*/0, std::nullopt, &kRaw);
    std::string err;
    EXPECT_FALSE(validate_anchor(anchor, 0, text, pt, ROCJITSU_CODE_ARCH_CDNA4, &err).has_value())
        << "mnemonic " << m << " should be rejected";
    EXPECT_FALSE(err.empty());
  }
}

TEST(Validator, RejectsSClauseAnchor) {
  static constexpr uint32_t kRaw = 0xBF850001u;
  TestInstruction anchor("s_clause", 4, /*flags=*/0, std::nullopt, &kRaw);
  auto text = dummy_text();
  InstrumentationPoint pt;

  std::string err;
  EXPECT_FALSE(validate_anchor(anchor, 0, text, pt, ROCJITSU_CODE_ARCH_RDNA4, &err).has_value());
  EXPECT_NE(err.find("s_clause"), std::string::npos) << "error was: " << err;
}

// Instruction::size() is a signed int; the `size != 4 && size != 8` check must
// reject negative sizes (which decoders never emit in practice) rather than let
// them through to the later cast to unsigned, where they would wrap.
TEST(Validator, RejectsNegativeInstructionSize) {
  static constexpr uint32_t kRaw = 0xDEADBEEFu;
  TestInstruction anchor("v_add_f32_e32", /*size=*/-4, /*flags=*/0, std::nullopt, &kRaw);
  auto text = dummy_text();

  std::string err;
  EXPECT_FALSE(
      is_relocatable_anchor(anchor, /*anchor_offset=*/0, text, ROCJITSU_CODE_ARCH_CDNA4, &err));
  EXPECT_NE(err.find("size must be 4 or 8"), std::string::npos) << "error was: " << err;
}

//==============================================================================
// Section 1b: validate_inline_nop_plan
//
// This guardrail used to live in TrampolineBuilder. It now lives at the
// orchestrator boundary so the builder stays generic. Tests moved with it.
//==============================================================================

TEST(InlineNopGuardrail, RejectsNonCanonicalBody) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;

  auto valid_plan = [&] {
    TrampolinePlan p;
    p.arch = kArch;
    p.anchor_offset = 0x100;
    p.original_size = 4;
    p.trampoline_offset = 0x200;
    p.return_target = 0x104;
    p.original_words.assign(1, 0xDEADBEEFu);
    p.before_items = {InlineAsmItem{{build_s_nop(0, kArch)}}};
    p.emit_original = true;
    return p;
  };

  // Sanity: the unmutated plan is accepted, so each sub-case below is
  // exercising exactly one guardrail dimension.
  ASSERT_TRUE(validate_inline_nop_plan(valid_plan()));

  // Extra before-item.
  {
    auto plan = valid_plan();
    plan.before_items.push_back(InlineAsmItem{{build_s_nop(0, kArch)}});
    std::string err;
    EXPECT_FALSE(validate_inline_nop_plan(plan, &err)) << "extra before-item must be rejected";
    EXPECT_FALSE(err.empty());
  }

  // Before-item that isn't s_nop 0.
  {
    auto plan = valid_plan();
    plan.before_items = {InlineAsmItem{{build_s_branch(0, kArch)}}};
    std::string err;
    EXPECT_FALSE(validate_inline_nop_plan(plan, &err))
        << "non-placeholder before-item must be rejected";
    EXPECT_FALSE(err.empty());
  }

  // Any after-item at all.
  {
    auto plan = valid_plan();
    plan.after_items.push_back(InlineAsmItem{{build_s_nop(0, kArch)}});
    std::string err;
    EXPECT_FALSE(validate_inline_nop_plan(plan, &err)) << "non-empty after_items must be rejected";
    EXPECT_FALSE(err.empty());
  }

  // emit_original = false.
  {
    auto plan = valid_plan();
    plan.emit_original = false;
    std::string err;
    EXPECT_FALSE(validate_inline_nop_plan(plan, &err)) << "emit_original = false must be rejected";
    EXPECT_FALSE(err.empty());
  }
}

//==============================================================================
// Section 2: make_trampoline_plan
//==============================================================================

TEST(MakeTrampolinePlan, FillsCanonicalBodyAndCopiesSiteFields) {
  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  ResolvedInstrumentationSite site;
  site.anchor_offset = 0x100;
  site.original_size = 8;
  site.original_bytes = {0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88};
  site.mnemonic = "buffer_load_dword";

  const uint64_t kTrampolineOffset = 0x400;
  TrampolinePlan plan = make_trampoline_plan(site, kArch, kTrampolineOffset);

  EXPECT_EQ(plan.arch, kArch);
  EXPECT_EQ(plan.anchor_offset, 0x100u);
  EXPECT_EQ(plan.original_size, 8u);
  EXPECT_EQ(plan.trampoline_offset, kTrampolineOffset);
  EXPECT_EQ(plan.return_target, 0x108u);
  ASSERT_EQ(plan.original_words.size(), 2u);
  EXPECT_EQ(plan.original_words[0], 0x44332211u); // little-endian
  EXPECT_EQ(plan.original_words[1], 0x88776655u);
  ASSERT_EQ(plan.before_items.size(), 1u);
  ASSERT_EQ(plan.before_items[0].words.size(), 1u);
  EXPECT_EQ(plan.before_items[0].words[0], build_s_nop(0, kArch));
  EXPECT_TRUE(plan.after_items.empty());
  EXPECT_TRUE(plan.emit_original);
}

//==============================================================================
// Section 3: Instrumentor integration tests (minimal in-memory gfx950/gfx1200 ELF)
//==============================================================================

// add_elf_name, align_up_for_test, make_gfx950_kernel_elf, make_gfx950_probe_elf,
// make_gfx1200_kernel_elf, make_gfx1200_probe_elf, section_words,
// patched_private_segment_size, and the kMov*/kProbe* opcode constants are shared
// via dbi_test_util.h (namespace rocjitsu::test). The make_gfx950_elf_with_*
// builders below are unique to this integration slice.

// gfx950 ELF with a single .text of `text_size` bytes filled with `s_nop 0`
// words (so every dword-aligned offset is a valid anchor). Used for the
// forward-branch overflow test, which needs .text large enough that the
// trampoline (placed at text_size) is more than INT16_MAX dwords past offset
// 0.
// Build a minimal gfx950 ELF whose .text holds @p words verbatim. Lets a test
// embed a real multi-word instruction (e.g. an 8-byte SOP1 + literal) rather
// than only s_nop fillers.
std::vector<uint8_t>
make_gfx950_elf_with_text_words(const std::vector<uint32_t> &words,
                                uint32_t machine_flags = EF_AMDGPU_MACH_AMDGCN_GFX950) {
  const uint64_t text_offset = 0x100;
  const uint64_t text_size = words.size() * sizeof(uint32_t);
  const uint64_t rodata_size = 4;

  std::vector<uint8_t> shstrtab{'\0'};
  const uint32_t text_name = add_elf_name(shstrtab, ".text");
  const uint32_t rodata_name = add_elf_name(shstrtab, ".rodata");
  const uint32_t shstrtab_name = add_elf_name(shstrtab, ".shstrtab");

  const uint64_t rodata_offset = text_offset + text_size;
  const uint64_t shstrtab_offset = rodata_offset + rodata_size;
  const uint64_t shoff = align_up_for_test(shstrtab_offset + shstrtab.size(), 8);
  constexpr uint16_t section_count = 4;

  std::vector<uint8_t> image(shoff + section_count * sizeof(Elf64_Shdr), 0);

  Elf64_Ehdr ehdr{};
  std::memcpy(ehdr.e_ident, EI_MAGIC, EI_MAGIC_SIZE);
  ehdr.e_ident[EI_CLASS] = ELFCLASS64;
  ehdr.e_ident[EI_OSABI] = ELFOSABI_AMDGPU_HSA;
  ehdr.e_type = ET_REL;
  ehdr.e_machine = EM_AMDGPU;
  ehdr.e_version = 1;
  ehdr.e_shoff = shoff;
  ehdr.e_flags = machine_flags;
  ehdr.e_ehsize = sizeof(Elf64_Ehdr);
  ehdr.e_shentsize = sizeof(Elf64_Shdr);
  ehdr.e_shnum = section_count;
  ehdr.e_shstrndx = 3;
  std::memcpy(image.data(), &ehdr, sizeof(ehdr));

  std::memcpy(image.data() + text_offset, words.data(), text_size);

  const uint32_t rodata_word = 0xA5A55A5Au;
  std::memcpy(image.data() + rodata_offset, &rodata_word, sizeof(rodata_word));
  std::memcpy(image.data() + shstrtab_offset, shstrtab.data(), shstrtab.size());

  std::array<Elf64_Shdr, section_count> shdrs{};
  shdrs[1].sh_name = text_name;
  shdrs[1].sh_type = SHT_PROGBITS;
  shdrs[1].sh_flags = SHF_ALLOC | SHF_EXECINSTR;
  shdrs[1].sh_offset = text_offset;
  shdrs[1].sh_size = text_size;
  shdrs[1].sh_addralign = sizeof(uint32_t);

  shdrs[2].sh_name = rodata_name;
  shdrs[2].sh_type = SHT_PROGBITS;
  shdrs[2].sh_flags = SHF_ALLOC;
  shdrs[2].sh_offset = rodata_offset;
  shdrs[2].sh_size = rodata_size;
  shdrs[2].sh_addralign = sizeof(uint32_t);

  shdrs[3].sh_name = shstrtab_name;
  shdrs[3].sh_type = SHT_STRTAB;
  shdrs[3].sh_offset = shstrtab_offset;
  shdrs[3].sh_size = shstrtab.size();
  shdrs[3].sh_addralign = 1;

  std::memcpy(image.data() + shoff, shdrs.data(), shdrs.size() * sizeof(Elf64_Shdr));
  return image;
}

std::vector<uint8_t> make_gfx950_elf_with_nop_text(uint64_t text_size) {
  // Must be a multiple of 4 so the entire section is decodable as s_nop words.
  return make_gfx950_elf_with_text_words(
      std::vector<uint32_t>(text_size / sizeof(uint32_t), 0xBF800000u));
}

// gfx950 ELF with TWO `.text` sections (both SHF_ALLOC | SHF_EXECINSTR, both
// named ".text"). Used to verify the multi-text rejection path.
std::vector<uint8_t> make_gfx950_elf_with_two_text_sections() {
  constexpr uint64_t text1_offset = 0x100;
  constexpr uint64_t text1_size = 4;
  constexpr uint64_t text2_offset = text1_offset + text1_size;
  constexpr uint64_t text2_size = 4;
  constexpr uint64_t rodata_size = 4;

  std::vector<uint8_t> shstrtab{'\0'};
  const uint32_t text_name = add_elf_name(shstrtab, ".text");
  const uint32_t rodata_name = add_elf_name(shstrtab, ".rodata");
  const uint32_t shstrtab_name = add_elf_name(shstrtab, ".shstrtab");

  const uint64_t rodata_offset = text2_offset + text2_size;
  const uint64_t shstrtab_offset = rodata_offset + rodata_size;
  const uint64_t shoff = align_up_for_test(shstrtab_offset + shstrtab.size(), 8);
  constexpr uint16_t section_count = 5;

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
  ehdr.e_shstrndx = 4;
  std::memcpy(image.data(), &ehdr, sizeof(ehdr));

  const uint32_t nop = 0xBF800000u;
  std::memcpy(image.data() + text1_offset, &nop, sizeof(nop));
  std::memcpy(image.data() + text2_offset, &nop, sizeof(nop));
  const uint32_t rodata_word = 0xA5A55A5Au;
  std::memcpy(image.data() + rodata_offset, &rodata_word, sizeof(rodata_word));
  std::memcpy(image.data() + shstrtab_offset, shstrtab.data(), shstrtab.size());

  std::array<Elf64_Shdr, section_count> shdrs{};
  shdrs[1].sh_name = text_name;
  shdrs[1].sh_type = SHT_PROGBITS;
  shdrs[1].sh_flags = SHF_ALLOC | SHF_EXECINSTR;
  shdrs[1].sh_offset = text1_offset;
  shdrs[1].sh_size = text1_size;
  shdrs[1].sh_addralign = sizeof(uint32_t);

  shdrs[2].sh_name = text_name; // also named ".text"
  shdrs[2].sh_type = SHT_PROGBITS;
  shdrs[2].sh_flags = SHF_ALLOC | SHF_EXECINSTR;
  shdrs[2].sh_offset = text2_offset;
  shdrs[2].sh_size = text2_size;
  shdrs[2].sh_addralign = sizeof(uint32_t);

  shdrs[3].sh_name = rodata_name;
  shdrs[3].sh_type = SHT_PROGBITS;
  shdrs[3].sh_flags = SHF_ALLOC;
  shdrs[3].sh_offset = rodata_offset;
  shdrs[3].sh_size = rodata_size;
  shdrs[3].sh_addralign = sizeof(uint32_t);

  shdrs[4].sh_name = shstrtab_name;
  shdrs[4].sh_type = SHT_STRTAB;
  shdrs[4].sh_offset = shstrtab_offset;
  shdrs[4].sh_size = shstrtab.size();
  shdrs[4].sh_addralign = 1;

  std::memcpy(image.data() + shoff, shdrs.data(), shdrs.size() * sizeof(Elf64_Shdr));
  return image;
}

// gfx950 ELF with .text containing two `s_nop 0` words (8 bytes total). The
// two-instruction layout lets tests verify that anchor resolution finds the second
// instruction at offset 4, not just the first at offset 0.
std::vector<uint8_t> make_gfx950_elf_with_two_nops() {
  return make_gfx950_elf_with_text_words({0xBF800000u, 0xBF800000u});
}

// Like make_gfx950_elf_with_two_nops() but with a discoverable `.kd` descriptor
// (nops at .text offsets 0 and 4). Probe calls require a discovered descriptor to
// bound SGPR selection, so probe-call tests use this variant. private_bytes is 0
// because these bodies clobber nothing live, so no site spills.
std::vector<uint8_t> make_gfx950_kernel_elf_with_two_nops() {
  return make_gfx950_kernel_elf({0xBF800000u, 0xBF800000u}, /*private_bytes=*/0);
}

TEST(Instrumentor, AddPointByOffsetResolvesValidatedSite) {
  auto image = make_gfx950_elf_with_two_nops();
  AmdGpuCodeObject obj(image.data(), image.size());
  ASSERT_TRUE(obj.is_valid());

  Instrumentor instrumentor(obj, ROCJITSU_CODE_ARCH_CDNA4);
  instrumentor.add_point_by_offset(/*anchor_offset=*/4);

  auto result = instrumentor.validate_points();
  ASSERT_TRUE(result.errors.empty())
      << (result.errors.empty() ? std::string{} : result.errors.front());
  ASSERT_EQ(result.sites.size(), 1u);
  EXPECT_EQ(result.sites[0].anchor_offset, 4u);
  EXPECT_EQ(result.sites[0].original_size, 4u);
  EXPECT_EQ(result.sites[0].original_bytes.size(), 4u);
  EXPECT_NE(result.sites[0].mnemonic.find("s_nop"), std::string::npos)
      << "mnemonic was: " << result.sites[0].mnemonic;
}

TEST(Instrumentor, ResolvesAnchorAtOffsetZero) {
  // Boundary: the first instruction starts at .text offset 0. Exercises the
  // offset→Instruction map's zero key alongside the offset-4 case above.
  auto image = make_gfx950_elf_with_two_nops();
  AmdGpuCodeObject obj(image.data(), image.size());
  ASSERT_TRUE(obj.is_valid());

  Instrumentor instrumentor(obj, ROCJITSU_CODE_ARCH_CDNA4);
  instrumentor.add_point_by_offset(/*anchor_offset=*/0);

  auto result = instrumentor.validate_points();
  ASSERT_TRUE(result.errors.empty())
      << (result.errors.empty() ? std::string{} : result.errors.front());
  ASSERT_EQ(result.sites.size(), 1u);
  EXPECT_EQ(result.sites[0].anchor_offset, 0u);
}

TEST(Instrumentor, RejectsAlignedOffsetPastLastInstruction) {
  // Aligned and looks plausible, but no instruction starts at offset 8 in a
  // 2-nop (8-byte) .text. The offset→Instruction map must miss rather than
  // extrapolate past its populated keys.
  auto image = make_gfx950_elf_with_two_nops();
  AmdGpuCodeObject obj(image.data(), image.size());
  ASSERT_TRUE(obj.is_valid());

  Instrumentor instrumentor(obj, ROCJITSU_CODE_ARCH_CDNA4);
  instrumentor.add_point_by_offset(/*anchor_offset=*/8);

  auto result = instrumentor.validate_points();
  EXPECT_TRUE(result.sites.empty());
  ASSERT_FALSE(result.errors.empty());
  EXPECT_NE(result.errors.front().find("no decoded instruction"), std::string::npos)
      << "error was: " << result.errors.front();
}

TEST(Instrumentor, RejectsOffsetInteriorToMultiWordInstruction) {
  // .text: [s_mov_b32 s0, 0xDEADBEEF (8 bytes)][s_nop 0]. Offset 4 lands in the
  // *interior* of the 8-byte instruction at offset 0, so no decoded instruction
  // starts there -- a distinct miss from the past-the-end case above. The
  // offset->Instruction map has keys 0 and 8 only.
  const std::vector<uint32_t> words = {0xBE8000FFu, 0xDEADBEEFu, 0xBF800000u};
  auto image = make_gfx950_elf_with_text_words(words);
  AmdGpuCodeObject obj(image.data(), image.size());
  ASSERT_TRUE(obj.is_valid());

  Instrumentor instrumentor(obj, ROCJITSU_CODE_ARCH_CDNA4);
  instrumentor.add_point_by_offset(/*anchor_offset=*/4);

  auto result = instrumentor.validate_points();
  EXPECT_TRUE(result.sites.empty());
  ASSERT_FALSE(result.errors.empty());
  EXPECT_NE(result.errors.front().find("no decoded instruction"), std::string::npos)
      << "error was: " << result.errors.front();
}

TEST(Instrumentor, UsesGfx12SClauseCountAndAcceptsFirstInstructionAfterRun) {
  const std::vector<uint32_t> words = {
      0xBF850A01u,              // s_clause 1, BREAK_SPAN=0xa: covers two instructions.
      0xF4002200u, 0xF8000020u, // s_load_b64 s[8:9], s[0:1], 0x20
      0xF4006000u, 0xF8000000u, // s_load_b256 s[0:7], s[0:1], 0x0
      0x06040F06u,              // v_add_f32_e32 v2, v6, v7
  };
  auto image = make_gfx950_elf_with_text_words(words, EF_AMDGPU_MACH_AMDGCN_GFX1201);
  AmdGpuCodeObject obj(image.data(), image.size());
  ASSERT_TRUE(obj.is_valid());

  Instrumentor covered(obj, ROCJITSU_CODE_ARCH_RDNA4);
  covered.add_point_by_offset(/*anchor_offset=*/12);

  auto covered_result = covered.validate_points();
  EXPECT_TRUE(covered_result.sites.empty());
  ASSERT_FALSE(covered_result.errors.empty());
  EXPECT_NE(covered_result.errors.front().find("inside an s_clause run"), std::string::npos)
      << "error was: " << covered_result.errors.front();

  Instrumentor after(obj, ROCJITSU_CODE_ARCH_RDNA4);
  after.add_point_by_offset(/*anchor_offset=*/20);
  auto after_result = after.validate_points();
  EXPECT_TRUE(after_result.errors.empty())
      << (after_result.errors.empty() ? std::string{} : after_result.errors.front());
  ASSERT_EQ(after_result.sites.size(), 1u);
  EXPECT_EQ(after_result.sites.front().anchor_offset, 20u);
}

TEST(Instrumentor, CarriesSClauseCoverageAcrossBasicBlockBoundary) {
  const std::vector<uint32_t> words = {
      0xBF850001u,              // s_clause 1: covers the next two instructions.
      0xF4002200u, 0xF8000020u, // s_load_b64 s[8:9], s[0:1], 0x20
      0xF4006000u, 0xF8000000u, // s_load_b256 s[0:7], s[0:1], 0x0; branch target.
      0x06040F06u,              // v_add_f32_e32 v2, v6, v7
      0xBF82FFFCu,              // s_branch from 0x18 back to 0x0c.
  };
  auto image = make_gfx950_elf_with_text_words(words, EF_AMDGPU_MACH_AMDGCN_GFX1201);
  AmdGpuCodeObject obj(image.data(), image.size());
  ASSERT_TRUE(obj.is_valid());

  Instrumentor instrumentor(obj, ROCJITSU_CODE_ARCH_RDNA4);
  instrumentor.add_point_by_offset(/*anchor_offset=*/12);

  auto result = instrumentor.validate_points();
  EXPECT_TRUE(result.sites.empty());
  ASSERT_FALSE(result.errors.empty());
  EXPECT_NE(result.errors.front().find("inside an s_clause run"), std::string::npos)
      << "error was: " << result.errors.front();
}

TEST(Instrumentor, RejectsAvailableInstructionInTruncatedSClauseRun) {
  const std::vector<uint32_t> words = {
      0xBF85003Fu, // s_clause 63: claims 64 following instructions.
      0x06040F06u, // only one following instruction is present.
  };
  auto image = make_gfx950_elf_with_text_words(words, EF_AMDGPU_MACH_AMDGCN_GFX1201);
  AmdGpuCodeObject obj(image.data(), image.size());
  ASSERT_TRUE(obj.is_valid());

  Instrumentor instrumentor(obj, ROCJITSU_CODE_ARCH_RDNA4);
  instrumentor.add_point_by_offset(/*anchor_offset=*/4);
  auto result = instrumentor.validate_points();
  EXPECT_TRUE(result.sites.empty());
  ASSERT_FALSE(result.errors.empty());
  EXPECT_NE(result.errors.front().find("inside an s_clause run"), std::string::npos)
      << "error was: " << result.errors.front();
}

TEST(Instrumentor, PreservesOuterCoverageAcrossOverlappingSClauses) {
  const std::vector<uint32_t> words = {
      0xBF850003u, // s_clause 3: covers the next four instructions.
      0xBF850000u, // s_clause 0: covers only the next instruction.
      0x06040F06u, // v_add_f32_e32 v2, v6, v7
      0x06040F06u, // remains covered by the outer clause.
      0x06040F06u, // final instruction covered by the outer clause.
      0x06040F06u, // first instruction after both clauses.
  };
  auto image = make_gfx950_elf_with_text_words(words, EF_AMDGPU_MACH_AMDGCN_GFX1201);
  AmdGpuCodeObject obj(image.data(), image.size());
  ASSERT_TRUE(obj.is_valid());

  Instrumentor covered(obj, ROCJITSU_CODE_ARCH_RDNA4);
  covered.add_point_by_offset(/*anchor_offset=*/16);
  auto covered_result = covered.validate_points();
  EXPECT_TRUE(covered_result.sites.empty());
  ASSERT_FALSE(covered_result.errors.empty());
  EXPECT_NE(covered_result.errors.front().find("inside an s_clause run"), std::string::npos)
      << "error was: " << covered_result.errors.front();

  Instrumentor after(obj, ROCJITSU_CODE_ARCH_RDNA4);
  after.add_point_by_offset(/*anchor_offset=*/20);
  auto after_result = after.validate_points();
  EXPECT_TRUE(after_result.errors.empty())
      << (after_result.errors.empty() ? std::string{} : after_result.errors.front());
  ASSERT_EQ(after_result.sites.size(), 1u);
  EXPECT_EQ(after_result.sites.front().anchor_offset, 20u);
}

TEST(Instrumentor, UnsupportedArchReportsErrorInsteadOfCrashing) {
  // The RISC-V decoder is available through the generic target registry, but
  // the AMDGPU Instrumentor must reject it explicitly rather than attempting
  // to decode an AMDGPU code object with the wrong ISA.
  auto image = make_gfx950_elf_with_two_nops();
  AmdGpuCodeObject obj(image.data(), image.size());
  ASSERT_TRUE(obj.is_valid());

  Instrumentor instrumentor(obj, ROCJITSU_CODE_ARCH_RV64I);
  instrumentor.add_point_by_offset(/*anchor_offset=*/4);

  auto result = instrumentor.validate_points();
  EXPECT_TRUE(result.sites.empty());
  ASSERT_FALSE(result.errors.empty());
  EXPECT_NE(result.errors.front().find("does not support RISC-V"), std::string::npos)
      << "error was: " << result.errors.front();

  // patch() must surface the same error rather than crashing.
  Instrumentor instrumentor2(obj, ROCJITSU_CODE_ARCH_RV64I);
  instrumentor2.add_point_by_offset(/*anchor_offset=*/4);
  auto patched = instrumentor2.patch();
  EXPECT_TRUE(patched.elf_bytes.empty());
  ASSERT_FALSE(patched.errors.empty());
  EXPECT_NE(patched.errors.front().find("does not support RISC-V"), std::string::npos)
      << "error was: " << patched.errors.front();
}

//==============================================================================
// Section 4: Instrumentor::patch end-to-end
//
// The synthetic ELF places .text at file offset 0x100 with two s_nop 0
// instructions (size = 8 bytes total). Patching the second instruction at
// .text-relative offset 4 yields these expected layout values:
//   trampoline_offset  = patcher.text_size() = 8
//   return_target      = 4 + 4 = 8
//   forward_simm16     = (8 - (4 + 4)) / 4 = 0
//   return_branch_pc   = 8 + 4 + 4 = 16
//   return_simm16      = (8 - (16 + 4)) / 4 = -3
//==============================================================================

TEST(InstrumentorPatch, EmitsValidElfWithExpectedPatchSummary) {
  auto image = make_gfx950_elf_with_two_nops();
  AmdGpuCodeObject obj(image.data(), image.size());
  ASSERT_TRUE(obj.is_valid());

  Instrumentor instrumentor(obj, ROCJITSU_CODE_ARCH_CDNA4);
  instrumentor.add_point_by_offset(/*anchor_offset=*/4);

  auto result = instrumentor.patch_with_debug_summaries();
  ASSERT_TRUE(result.errors.empty())
      << (result.errors.empty() ? std::string{} : result.errors.front());
  EXPECT_FALSE(result.elf_bytes.empty());
  ASSERT_EQ(result.patches.size(), 1u);

  const auto &p = result.patches[0];
  EXPECT_EQ(p.anchor_offset, 4u);
  EXPECT_EQ(p.original_size, 4u);
  EXPECT_EQ(p.trampoline_offset, 8u);
  EXPECT_EQ(p.return_target, 8u);

  // Original bytes are s_nop 0 = 0xBF800000 little-endian.
  const std::vector<uint8_t> expected_original = {0x00, 0x00, 0x80, 0xBF};
  EXPECT_EQ(p.original_bytes, expected_original);

  // Patched anchor is build_s_branch(0, CDNA4): branch from anchor to the
  // trampoline that starts immediately after .text.
  ASSERT_EQ(p.patched_anchor_bytes.size(), 4u);
  uint32_t patched_word = 0;
  std::memcpy(&patched_word, p.patched_anchor_bytes.data(), sizeof(patched_word));
  EXPECT_EQ(patched_word, build_s_branch(0, ROCJITSU_CODE_ARCH_CDNA4));

  // Emitted ELF reparses cleanly.
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  EXPECT_TRUE(patched.is_valid());
}

// 8-byte anchor end-to-end. .text is [s_nop 0][s_mov_b32 s0, 0xDEADBEEF]; the
// s_mov_b32 + literal at offset 4 is a real 8-byte instruction. This exercises
// the orchestrator paths the 4-byte nop fixtures never reach: offset_to_inst_
// advancing by 8, the 8-byte splice, the s_branch + s_nop 0 tail, and a
// reparse/decode of the relocated 8-byte original from the trampoline cave.
TEST(InstrumentorPatch, PatchesEightByteAnchorEndToEnd) {
  constexpr uint32_t kSMovLitWord0 = 0xBE8000FFu; // s_mov_b32 s0, <literal>
  constexpr uint32_t kSMovLitWord1 = 0xDEADBEEFu; // the 32-bit literal
  const std::vector<uint32_t> words = {0xBF800000u, kSMovLitWord0, kSMovLitWord1};
  auto image = make_gfx950_elf_with_text_words(words);
  AmdGpuCodeObject obj(image.data(), image.size());
  ASSERT_TRUE(obj.is_valid());

  Instrumentor instrumentor(obj, ROCJITSU_CODE_ARCH_CDNA4);
  instrumentor.add_point_by_offset(/*anchor_offset=*/4);

  auto result = instrumentor.patch_with_debug_summaries();
  ASSERT_TRUE(result.errors.empty())
      << (result.errors.empty() ? std::string{} : result.errors.front());
  ASSERT_EQ(result.patches.size(), 1u);

  const auto &p = result.patches[0];
  // The decoder saw an 8-byte instruction and the orchestrator carried that
  // width through. Original .text is 3 words (12 bytes), so the trampoline cave
  // begins at offset 12.
  EXPECT_EQ(p.original_size, 8u);
  EXPECT_EQ(p.anchor_offset, 4u);
  EXPECT_EQ(p.trampoline_offset, 12u);
  EXPECT_EQ(p.return_target, 12u); // anchor_offset + original_size

  // Original 8 bytes captured verbatim.
  ASSERT_EQ(p.original_bytes.size(), 8u);
  std::array<uint32_t, 2> orig_words{};
  std::memcpy(orig_words.data(), p.original_bytes.data(), orig_words.size() * sizeof(uint32_t));
  EXPECT_EQ(orig_words[0], kSMovLitWord0);
  EXPECT_EQ(orig_words[1], kSMovLitWord1);

  // Patched anchor fills the full 8-byte slot: forward s_branch + s_nop 0 tail.
  // Forward simm16 = (trampoline_offset - (anchor_offset + 4)) / 4 = (12 - 8)/4 = 1.
  ASSERT_EQ(p.patched_anchor_bytes.size(), 8u);
  std::array<uint32_t, 2> patched_words{};
  std::memcpy(patched_words.data(), p.patched_anchor_bytes.data(),
              patched_words.size() * sizeof(uint32_t));
  EXPECT_EQ(patched_words[0], build_s_branch(1, ROCJITSU_CODE_ARCH_CDNA4));
  EXPECT_EQ(patched_words[1], build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4));

  // Reparse and decode out of the grown .text. The cave (tail of .text) for an
  // 8-byte anchor is [s_nop 0][orig word0][orig word1][s_branch return] = 16
  // bytes, so .text grows from 12 to 28 bytes.
  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_FALSE(patched.text_sections().empty());
  const Section *text = patched.text_sections().front();
  ASSERT_EQ(text->size(), 28u);

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  // Copy two words (8 bytes) so an 8-byte instruction has its literal available.
  auto decode_at = [&](uint64_t off) {
    std::array<rj_code_binary_inst_t, 2> w{};
    std::memcpy(w.data(), text->data() + off, w.size() * sizeof(rj_code_binary_inst_t));
    return std::unique_ptr<Instruction>(decode_valid(*decoder, w.data()));
  };

  // Patched anchor decodes as a forward s_branch.
  auto anchor = decode_at(4);
  ASSERT_NE(anchor, nullptr);
  EXPECT_NE(anchor->mnemonic().find("s_branch"), std::string_view::npos)
      << "mnemonic was: " << anchor->mnemonic();

  // Cave placeholder decodes as s_nop; the relocated original (cave offset 4 ->
  // .text offset 16) decodes back to the same 8-byte s_mov_b32.
  auto placeholder = decode_at(12);
  ASSERT_NE(placeholder, nullptr);
  EXPECT_NE(placeholder->mnemonic().find("s_nop"), std::string_view::npos)
      << "mnemonic was: " << placeholder->mnemonic();

  auto relocated = decode_at(16);
  ASSERT_NE(relocated, nullptr);
  EXPECT_EQ(relocated->size(), 8) << "relocated original must decode as an 8-byte instruction";
  EXPECT_NE(relocated->mnemonic().find("s_mov_b32"), std::string_view::npos)
      << "mnemonic was: " << relocated->mnemonic();
}

TEST(InstrumentorPatch, RejectsZeroQueuedPoints) {
  auto image = make_gfx950_elf_with_two_nops();
  AmdGpuCodeObject obj(image.data(), image.size());
  Instrumentor instrumentor(obj, ROCJITSU_CODE_ARCH_CDNA4);
  // No points queued.

  auto result = instrumentor.patch();
  EXPECT_TRUE(result.elf_bytes.empty());
  EXPECT_FALSE(result.errors.empty());
}

TEST(InstrumentorPatch, ValidationFailurePropagates) {
  auto image = make_gfx950_elf_with_two_nops();
  AmdGpuCodeObject obj(image.data(), image.size());
  Instrumentor instrumentor(obj, ROCJITSU_CODE_ARCH_CDNA4);
  // Unaligned offset — validator rejects.
  instrumentor.add_point_by_offset(/*anchor_offset=*/2);

  auto result = instrumentor.patch();
  EXPECT_TRUE(result.elf_bytes.empty());
  EXPECT_FALSE(result.errors.empty());
}

TEST(InstrumentorPatch, IsSingleCall) {
  auto image = make_gfx950_elf_with_two_nops();
  AmdGpuCodeObject obj(image.data(), image.size());
  Instrumentor instrumentor(obj, ROCJITSU_CODE_ARCH_CDNA4);
  instrumentor.add_point_by_offset(4);

  auto first = instrumentor.patch();
  ASSERT_TRUE(first.errors.empty()) << first.errors.front();
  EXPECT_FALSE(first.elf_bytes.empty());

  auto second = instrumentor.patch();
  EXPECT_TRUE(second.elf_bytes.empty());
  EXPECT_FALSE(second.errors.empty());
}

// Pins the documented single-attempt semantics: any call (success or failure)
// burns the budget. A caller that hits a recoverable argument error must
// construct a new Instrumentor.
TEST(InstrumentorPatch, FailedFirstCallStillBurnsSingleAttemptBudget) {
  auto image = make_gfx950_elf_with_two_nops();
  AmdGpuCodeObject obj(image.data(), image.size());
  Instrumentor instrumentor(obj, ROCJITSU_CODE_ARCH_CDNA4);
  // First call: no queued points — fatal error.
  auto first = instrumentor.patch();
  ASSERT_FALSE(first.errors.empty()) << "first call must fail with zero points";

  // Try to recover by queueing a valid point and calling again.
  instrumentor.add_point_by_offset(4);
  auto second = instrumentor.patch();
  EXPECT_TRUE(second.elf_bytes.empty()) << "single-attempt budget must already be spent";
  EXPECT_FALSE(second.errors.empty());
}

TEST(InstrumentorPatch, RejectsMultiTextCodeObject) {
  auto image = make_gfx950_elf_with_two_text_sections();
  AmdGpuCodeObject obj(image.data(), image.size());
  ASSERT_TRUE(obj.is_valid());
  // Precondition for the test to be meaningful: the loader actually produced
  // two .text sections.
  ASSERT_EQ(obj.text_sections().size(), 2u)
      << "fixture must yield multi-text; got " << obj.text_sections().size();

  Instrumentor instrumentor(obj, ROCJITSU_CODE_ARCH_CDNA4);
  instrumentor.add_point_by_offset(0);
  auto result = instrumentor.patch();
  EXPECT_TRUE(result.elf_bytes.empty());
  EXPECT_FALSE(result.errors.empty());
}

// Branch-range overflow at the orchestrator level. The trampoline is placed
// at .text_size; with anchor at offset 0 we need .text_size > 131072 to push
// forward_simm16 past INT16_MAX. .text_size = 131076 gives forward_simm16 =
// 32768 → overflow. The builder returns an error; Instrumentor::patch must
// surface it and emit no ELF.
TEST(InstrumentorPatch, BranchRangeOverflowPropagatesAsFatalError) {
  // 32769 × 4 = 131076 bytes of s_nop in .text.
  auto image = make_gfx950_elf_with_nop_text(/*text_size=*/131076);
  AmdGpuCodeObject obj(image.data(), image.size());
  ASSERT_TRUE(obj.is_valid());

  Instrumentor instrumentor(obj, ROCJITSU_CODE_ARCH_CDNA4);
  instrumentor.add_point_by_offset(/*anchor_offset=*/0);
  auto result = instrumentor.patch();

  EXPECT_TRUE(result.elf_bytes.empty()) << "overflow must not leak a half-built ELF";
  ASSERT_FALSE(result.errors.empty());
  // The builder's diagnostic mentions "forward branch ... exceeds s_branch simm16".
  EXPECT_NE(result.errors.front().find("forward"), std::string::npos)
      << "diagnostic must identify the forward branch; got: " << result.errors.front();
}

//==============================================================================
// Section 5: Patched ELF shape (reparse)
//
// Builds, instruments, and reparses once via TEST_F SetUp. Each test then
// asserts one structural property of the emitted ELF.
//==============================================================================

class InstrumentorPatchElfShape : public ::testing::Test {
protected:
  // places instrumentation points starting at anchor_offset = 4
  void Init(uint64_t num_elf_nops, uint64_t num_instrumentation_points) {
    // leaving anchor_offset = 0 empty, so will fail on num_elf_nops == num_instrumentation_points
    ASSERT_TRUE(num_elf_nops > num_instrumentation_points);
    num_elf_nops_ = num_elf_nops;
    num_points_ = num_instrumentation_points;

    image_ = make_gfx950_elf_with_nop_text(num_elf_nops * 4);
    AmdGpuCodeObject obj(image_.data(), image_.size());
    ASSERT_TRUE(obj.is_valid());

    Instrumentor instrumentor(obj, ROCJITSU_CODE_ARCH_CDNA4);
    for (uint64_t point = 0; point < num_instrumentation_points; ++point)
      instrumentor.add_point_by_offset(/*anchor_offset=*/(point + 1) * 4);

    result_ = instrumentor.patch();
    ASSERT_TRUE(result_.errors.empty())
        << (result_.errors.empty() ? std::string{} : result_.errors.front());
    ASSERT_FALSE(result_.elf_bytes.empty());

    patched_ =
        std::make_unique<AmdGpuCodeObject>(result_.elf_bytes.data(), result_.elf_bytes.size());
    ASSERT_TRUE(patched_->is_valid());
  }

  const Section *find_section(std::string_view name) const {
    for (const auto &s : patched_->all_sections())
      if (s->name() == name)
        return s.get();
    return nullptr;
  }

  // Byte size of the original .text before instrumentation.
  uint64_t original_text_size() const { return num_elf_nops_ * sizeof(uint32_t); }
  // Byte size of the appended trampoline cave: 3 words per inline-nop site.
  uint64_t cave_size() const { return num_points_ * 3 * sizeof(uint32_t); }

  // The trampoline cave lives at the tail of .text, immediately after the
  // original bytes. Reads it back as decoded words.
  std::vector<uint32_t> cave_words() const {
    const Section *text = find_section(".text");
    EXPECT_NE(text, nullptr);
    std::vector<uint32_t> words(cave_size() / sizeof(uint32_t));
    if (text != nullptr)
      std::memcpy(words.data(), text->data() + original_text_size(), cave_size());
    return words;
  }

  uint64_t num_elf_nops_ = 0;
  uint64_t num_points_ = 0;
  std::vector<uint8_t> image_;
  InstrumentedCodeObject result_;
  std::unique_ptr<AmdGpuCodeObject> patched_;
};

TEST_F(InstrumentorPatchElfShape, TrampolineCaveAppendedToText) {
  Init(2, 1);
  const Section *text = find_section(".text");
  ASSERT_NE(text, nullptr);
  EXPECT_EQ(text->size(), original_text_size() + cave_size())
      << ".text must grow by exactly the appended trampoline cave";
}

TEST_F(InstrumentorPatchElfShape, TrampolineCaveAppendedToTextMultiple) {
  Init(100, 50);
  const Section *text = find_section(".text");
  ASSERT_NE(text, nullptr);
  EXPECT_EQ(text->size(), original_text_size() + cave_size())
      << ".text must grow by exactly the appended trampoline cave";
}

TEST_F(InstrumentorPatchElfShape, TextRemainsExecutable) {
  Init(2, 1);
  // The cave is part of .text, so it inherits .text's executable flag
  const Section *text = find_section(".text");
  ASSERT_NE(text, nullptr);
  EXPECT_NE(text->flags() & SHF_EXECINSTR, 0u) << ".text must have SHF_EXECINSTR";
}

TEST_F(InstrumentorPatchElfShape, TextRemainsExecutableMultiple) {
  Init(100, 50);
  const Section *text = find_section(".text");
  ASSERT_NE(text, nullptr);
  EXPECT_NE(text->flags() & SHF_EXECINSTR, 0u) << ".text must have SHF_EXECINSTR";
}

TEST_F(InstrumentorPatchElfShape, TrampolineCaveContentsMatchExpectedWords) {
  Init(2, 1);
  // Body for the inline-nop smoke build with a 4-byte anchor:
  //   [placeholder s_nop 0, relocated original s_nop 0, return s_branch(-3)].
  // The cave's absolute offset is unchanged from the old separate-section
  // layout (still text_size()), so the branch math is identical.
  const std::vector<uint32_t> words = cave_words();
  ASSERT_EQ(words.size(), 3u);

  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  EXPECT_EQ(words[0], build_s_nop(0, kArch))
      << "trampoline body must start with the placeholder s_nop 0";
  EXPECT_EQ(words[1], 0xBF800000u) // original s_nop 0 from .text[4..8)
      << "trampoline body must contain the relocated original instruction unchanged";
  EXPECT_EQ(words[2], build_s_branch(-3, kArch))
      << "trampoline must end with s_branch back to anchor + original_size";
}

TEST_F(InstrumentorPatchElfShape, TrampolineCaveContentsMatchExpectedWordsMultiple) {
  Init(100, 50);
  const std::vector<uint32_t> words = cave_words();
  ASSERT_EQ(words.size(), 150u);

  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  for (uint64_t point = 0; point < 50; ++point) {
    EXPECT_EQ(words[point * 3], build_s_nop(0, kArch))
        << "trampoline body must start with the placeholder s_nop 0";
    EXPECT_EQ(words[point * 3 + 1], 0xBF800000u) // original s_nop 0 from .text)
        << "trampoline body must contain the relocated original instruction unchanged";
    EXPECT_EQ(words[point * 3 + 2], build_s_branch(-101 - 2 * point, kArch))
        << "trampoline must end with s_branch back to anchor + original_size";
  }
}

//==============================================================================
// Section 6: Patched code decoded back through the real Decoder
//
// Verifies that the bytes the orchestrator emits are recognized by the
// decoder as the instructions we intended, with the expected flags and
// branch-offset signs. Complements Section 5, which only checks
// byte-equality against the builder's output.
//==============================================================================

class InstrumentorPatchDecoded : public ::testing::Test {
protected:
  // places instrumentation points starting at anchor_offset = 4
  void Init(uint64_t num_elf_nops, uint64_t num_instrumentation_points) {
    // leaving anchor_offset = 0 empty, so will fail on num_elf_nops == num_instrumentation_points
    ASSERT_TRUE(num_elf_nops > num_instrumentation_points);
    num_elf_nops_ = num_elf_nops;
    num_points_ = num_instrumentation_points;

    image_ = make_gfx950_elf_with_nop_text(num_elf_nops * 4);
    AmdGpuCodeObject obj(image_.data(), image_.size());
    ASSERT_TRUE(obj.is_valid());

    Instrumentor instrumentor(obj, ROCJITSU_CODE_ARCH_CDNA4);
    for (uint64_t point = 0; point < num_instrumentation_points; ++point)
      instrumentor.add_point_by_offset(/*anchor_offset=*/(point + 1) * 4);
    result_ = instrumentor.patch();
    ASSERT_TRUE(result_.errors.empty())
        << (result_.errors.empty() ? std::string{} : result_.errors.front());
    ASSERT_FALSE(result_.elf_bytes.empty());

    patched_ =
        std::make_unique<AmdGpuCodeObject>(result_.elf_bytes.data(), result_.elf_bytes.size());
    ASSERT_TRUE(patched_->is_valid());

    decoder_ = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
    ASSERT_NE(decoder_, nullptr);
  }

  const Section *find_section(std::string_view name) const {
    for (const auto &s : patched_->all_sections())
      if (s->name() == name)
        return s.get();
    return nullptr;
  }

  std::unique_ptr<Instruction> decode_word(const Section *section, size_t byte_offset) {
    rj_code_binary_inst_t word = 0;
    std::memcpy(&word, section->data() + byte_offset, sizeof(word));
    return std::unique_ptr<Instruction>(decode_valid(*decoder_, &word));
  }

  // Byte size of the original .text before instrumentation. The trampoline cave
  // begins here, at the tail of the grown .text
  uint64_t original_text_size() const { return num_elf_nops_ * sizeof(uint32_t); }

  // Decode a word from the trampoline cave. @p cave_byte_offset is relative to
  // the start of the cave (== the old .rj_trampolines-relative offset).
  std::unique_ptr<Instruction> decode_cave_word(size_t cave_byte_offset) {
    const Section *text = find_section(".text");
    EXPECT_NE(text, nullptr);
    return decode_word(text, original_text_size() + cave_byte_offset);
  }

  uint64_t num_elf_nops_ = 0;
  uint64_t num_points_ = 0;
  std::vector<uint8_t> image_;
  InstrumentedCodeObject result_;
  std::unique_ptr<AmdGpuCodeObject> patched_;
  std::unique_ptr<Decoder> decoder_;
};

TEST_F(InstrumentorPatchDecoded, PatchedAnchorDecodesAsForwardSBranch) {
  Init(2, 1);
  const Section *text = find_section(".text");
  ASSERT_NE(text, nullptr);
  auto inst = decode_word(text, /*byte_offset=*/4);
  ASSERT_NE(inst, nullptr);

  EXPECT_NE(inst->mnemonic().find("s_branch"), std::string_view::npos)
      << "mnemonic was: " << inst->mnemonic();
  EXPECT_NE(inst->flags() & BRANCH, 0u) << "patched anchor must carry BRANCH flag";
  ASSERT_TRUE(inst->branch_offset_bytes().has_value())
      << "patched anchor must report a PC-relative branch offset";
  EXPECT_GE(*inst->branch_offset_bytes(), 0)
      << "forward branch offset must be non-negative; got " << *inst->branch_offset_bytes();
}

TEST_F(InstrumentorPatchDecoded, PatchedMultipleAnchorDecodesAsForwardSBranch) {
  Init(100, 50);
  const Section *text = find_section(".text");
  ASSERT_NE(text, nullptr);
  for (uint64_t point = 0; point < 50; ++point) {
    auto inst = decode_word(text, /*byte_offset=*/(point + 1) * 4);
    ASSERT_NE(inst, nullptr);

    EXPECT_NE(inst->mnemonic().find("s_branch"), std::string_view::npos)
        << "mnemonic was: " << inst->mnemonic();
    EXPECT_NE(inst->flags() & BRANCH, 0u) << "patched anchor must carry BRANCH flag";
    ASSERT_TRUE(inst->branch_offset_bytes().has_value())
        << "patched anchor must report a PC-relative branch offset";
    EXPECT_GE(*inst->branch_offset_bytes(), 0)
        << "forward branch offset must be non-negative; got " << *inst->branch_offset_bytes();
  }
}

TEST_F(InstrumentorPatchDecoded, UnpatchedTextInstructionIsUnchanged) {
  Init(2, 1);
  // The first s_nop at .text[0..4] was not the anchor. Decoder must still see
  // it as s_nop — proves the patcher did not accidentally touch it.
  const Section *text = find_section(".text");
  ASSERT_NE(text, nullptr);
  auto inst = decode_word(text, /*byte_offset=*/0);
  ASSERT_NE(inst, nullptr);
  EXPECT_NE(inst->mnemonic().find("s_nop"), std::string_view::npos)
      << "unpatched .text instruction must remain s_nop; got " << inst->mnemonic();
  EXPECT_EQ(inst->flags() & BRANCH, 0u);
  EXPECT_FALSE(inst->branch_offset_bytes().has_value());
}

TEST_F(InstrumentorPatchDecoded, UnpatchedMultipleTextInstructionIsUnchanged) {
  Init(100, 50);
  // The first s_nop at .text[0..4] was not the anchor. Decoder must still see
  // it as s_nop — proves the patcher did not accidentally touch it.
  const Section *text = find_section(".text");
  ASSERT_NE(text, nullptr);
  auto inst = decode_word(text, /*byte_offset=*/0);
  ASSERT_NE(inst, nullptr);
  EXPECT_NE(inst->mnemonic().find("s_nop"), std::string_view::npos)
      << "unpatched .text instruction must remain s_nop; got " << inst->mnemonic();
  EXPECT_EQ(inst->flags() & BRANCH, 0u);
  EXPECT_FALSE(inst->branch_offset_bytes().has_value());
  for (uint64_t point = 51; point < 100; ++point) {
    inst = decode_word(text, point * 4);
    ASSERT_NE(inst, nullptr);
    EXPECT_NE(inst->mnemonic().find("s_nop"), std::string_view::npos)
        << "unpatched .text instruction must remain s_nop; got " << inst->mnemonic();
    EXPECT_EQ(inst->flags() & BRANCH, 0u);
    EXPECT_FALSE(inst->branch_offset_bytes().has_value());
  }
}

TEST_F(InstrumentorPatchDecoded, TrampolinePlaceholderDecodesAsSNop) {
  Init(2, 1);
  auto inst = decode_cave_word(/*cave_byte_offset=*/0);
  ASSERT_NE(inst, nullptr);

  EXPECT_NE(inst->mnemonic().find("s_nop"), std::string_view::npos)
      << "mnemonic was: " << inst->mnemonic();
  EXPECT_EQ(inst->flags() & BRANCH, 0u) << "placeholder must not be marked as a branch";
  EXPECT_FALSE(inst->branch_offset_bytes().has_value())
      << "placeholder must not carry a branch offset";
}

TEST_F(InstrumentorPatchDecoded, TrampolinePlaceholderMultipleDecodesAsSNop) {
  Init(100, 50);
  for (uint64_t trampoline_count = 0; trampoline_count < 50; ++trampoline_count) {
    auto inst = decode_cave_word(/*cave_byte_offset=*/trampoline_count * 12);
    ASSERT_NE(inst, nullptr);

    EXPECT_NE(inst->mnemonic().find("s_nop"), std::string_view::npos)
        << "mnemonic was: " << inst->mnemonic();
    EXPECT_EQ(inst->flags() & BRANCH, 0u) << "placeholder must not be marked as a branch";
    EXPECT_FALSE(inst->branch_offset_bytes().has_value())
        << "placeholder must not carry a branch offset";
  }
}

TEST_F(InstrumentorPatchDecoded, RelocatedOriginalPreservesMnemonic) {
  Init(2, 1);
  // The original instruction at .text[4..8] was s_nop 0. After relocation it
  // lives in the trampoline body at cave offset 4. Decoded bytes must agree.
  auto relocated = decode_cave_word(/*cave_byte_offset=*/4);
  ASSERT_NE(relocated, nullptr);
  EXPECT_NE(relocated->mnemonic().find("s_nop"), std::string_view::npos)
      << "relocated original must decode to the same mnemonic as the source; got "
      << relocated->mnemonic();
}

TEST_F(InstrumentorPatchDecoded, RelocatedMultipleOriginalPreservesMnemonic) {
  Init(100, 50);
  // The original instruction at .text[4..8] was s_nop 0. After relocation it
  // lives in the trampoline body at cave offset 4. Decoded bytes must agree.
  for (uint64_t trampoline_count = 0; trampoline_count < 50; ++trampoline_count) {
    auto relocated = decode_cave_word(/*cave_byte_offset=*/trampoline_count * 12 + 4);
    ASSERT_NE(relocated, nullptr);
    EXPECT_NE(relocated->mnemonic().find("s_nop"), std::string_view::npos)
        << "relocated original must decode to the same mnemonic as the source; got "
        << relocated->mnemonic();
  }
}

TEST_F(InstrumentorPatchDecoded, TrampolineReturnDecodesAsBackwardSBranch) {
  Init(2, 1);
  auto inst = decode_cave_word(/*cave_byte_offset=*/8);
  ASSERT_NE(inst, nullptr);

  EXPECT_NE(inst->mnemonic().find("s_branch"), std::string_view::npos)
      << "mnemonic was: " << inst->mnemonic();
  EXPECT_NE(inst->flags() & BRANCH, 0u) << "return must carry BRANCH flag";
  ASSERT_TRUE(inst->branch_offset_bytes().has_value())
      << "return must report a PC-relative branch offset";
  EXPECT_LT(*inst->branch_offset_bytes(), 0)
      << "return branch offset must be negative; got " << *inst->branch_offset_bytes();
}

TEST_F(InstrumentorPatchDecoded, TrampolineMultipleReturnDecodesAsBackwardSBranch) {
  Init(100, 50);
  for (uint64_t trampoline_count = 0; trampoline_count < 50; ++trampoline_count) {
    auto inst = decode_cave_word(/*cave_byte_offset=*/trampoline_count * 12 + 8);
    ASSERT_NE(inst, nullptr);

    EXPECT_NE(inst->mnemonic().find("s_branch"), std::string_view::npos)
        << "mnemonic was: " << inst->mnemonic();
    EXPECT_NE(inst->flags() & BRANCH, 0u) << "return must carry BRANCH flag";
    ASSERT_TRUE(inst->branch_offset_bytes().has_value())
        << "return must report a PC-relative branch offset";
    EXPECT_LT(*inst->branch_offset_bytes(), 0)
        << "return branch offset must be negative; got " << *inst->branch_offset_bytes();
  }
}

// End-to-end behavioral check: decode the patched s_branch and the return
// s_branch, compute their actual landing addresses from the raw SOPP simm16,
// and assert they hit the expected .text-relative offsets. This pins the one
// promise PC01-A is fundamentally about — the round trip through the
// trampoline lands the program right back at anchor + original_size.
//
// SOPP semantics: target = branch_pc + 4 + simm16 * 4.
TEST_F(InstrumentorPatchDecoded, BranchesLandAtTheirIntendedTargets) {
  Init(2, 1);
  const Section *text = find_section(".text");
  ASSERT_NE(text, nullptr);

  auto sopp_target = [](uint64_t branch_pc, const uint32_t *word) -> uint64_t {
    const int16_t simm16 = static_cast<int16_t>(*word & 0xFFFFu);
    return branch_pc + 4 + static_cast<int64_t>(simm16) * 4;
  };

  // Forward branch lives at .text-relative offset 4 (the patched anchor). It
  // should land at the cave start == original_text_size() (== 8 in this fixture).
  auto fwd = decode_word(text, /*byte_offset=*/4);
  ASSERT_NE(fwd, nullptr);
  ASSERT_NE(fwd->raw_encoding(), nullptr);
  const uint64_t fwd_target = sopp_target(/*branch_pc=*/4, fwd->raw_encoding());
  EXPECT_EQ(fwd_target, original_text_size())
      << "forward branch must land at the trampoline cave start (= original_text_size())";

  // Return branch lives at cave-relative offset 8. In .text-relative coordinates
  // that is original_text_size() + 8 (= 16 in this fixture). It should land at
  // anchor_offset + original_size = 4 + 4 = 8.
  auto ret = decode_cave_word(/*cave_byte_offset=*/8);
  ASSERT_NE(ret, nullptr);
  ASSERT_NE(ret->raw_encoding(), nullptr);
  const uint64_t ret_pc = original_text_size() + 8;
  const uint64_t ret_target = sopp_target(ret_pc, ret->raw_encoding());
  EXPECT_EQ(ret_target, 8u) << "return branch must land at anchor + original_size";
}

TEST_F(InstrumentorPatchDecoded, BranchesMultipleLandAtTheirIntendedTargets) {
  Init(100, 50);
  const Section *text = find_section(".text");
  ASSERT_NE(text, nullptr);

  auto sopp_target = [](uint64_t branch_pc, const uint32_t *word) -> uint64_t {
    const int16_t simm16 = static_cast<int16_t>(*word & 0xFFFFu);
    return branch_pc + 4 + static_cast<int64_t>(simm16) * 4;
  };

  for (uint64_t point = 0; point < 50; ++point) {
    auto fwd = decode_word(text, /*byte_offset=*/(point + 1) * 4);
    ASSERT_NE(fwd, nullptr);
    ASSERT_NE(fwd->raw_encoding(), nullptr);
    const uint64_t fwd_target = sopp_target(/*branch_pc=*/(point + 1) * 4, fwd->raw_encoding());
    EXPECT_EQ(fwd_target, original_text_size() + point * 12)
        << "forward branch must land at its trampoline cave offset";
  }

  for (uint64_t point = 0; point < 50; ++point) {
    auto ret = decode_cave_word(/*cave_byte_offset=*/point * 12 + 8);
    ASSERT_NE(ret, nullptr);
    ASSERT_NE(ret->raw_encoding(), nullptr);
    const uint64_t ret_pc = original_text_size() + point * 12 + 8;
    const uint64_t ret_target = sopp_target(ret_pc, ret->raw_encoding());
    EXPECT_EQ(ret_target, (point + 2) * 4) << "return branch must land at anchor + original_size";
  }
}

//==============================================================================
// Spill formula and policy (orchestrator-owned).
//
// These exercise the live/clobber arithmetic and the current no-spill policy
// with explicitly-constructed register sets. The builder-envelope clobbers
// (link pair, target pair) are passed in directly here; their dead-register
// selection is deferred.
//==============================================================================

RegisterSet make_sgpr_set(std::initializer_list<uint16_t> indices) {
  RegisterSet set;
  for (uint16_t i : indices)
    set.expand(RegisterRef{RegClass::SGPR, i, 1});
  return set;
}

bool has_sgpr(const RegisterSet &set, uint16_t index) {
  return set.contains(RegisterRef{RegClass::SGPR, index, 1});
}

// instrument_clobbers = probe body clobbers | builder envelope clobbers. The
// builder set carries the link pair s[30:31] and a target pair s[8:9]; the
// probe body contributes s5. The union must contain all of them, and the
// callee summary must not be mutated to hold the envelope facts.
TEST(InstrumentorSpill, InstrumentationClobbersUnionProbeAndBuilder) {
  ProbeClobberSummary probe_summary;
  probe_summary.ordinary_clobbers.expand(RegisterRef{RegClass::SGPR, 5, 1});

  const RegisterSet builder_clobbers = make_sgpr_set({30, 31, 8, 9});
  const RegisterSet clobbers = compute_instrumentation_clobbers(probe_summary, builder_clobbers);

  EXPECT_TRUE(has_sgpr(clobbers, 5));  // from the probe body
  EXPECT_TRUE(has_sgpr(clobbers, 30)); // link pair
  EXPECT_TRUE(has_sgpr(clobbers, 31));
  EXPECT_TRUE(has_sgpr(clobbers, 8)); // target pair
  EXPECT_TRUE(has_sgpr(clobbers, 9));
  // The summary is callee-only; computing the union does not add envelope facts.
  EXPECT_FALSE(has_sgpr(probe_summary.ordinary_clobbers, 30));
  EXPECT_FALSE(has_sgpr(probe_summary.ordinary_clobbers, 8));
}

// spill_set = live_at_anchor & instrumentation_clobbers.
TEST(InstrumentorSpill, SpillSetIsLiveIntersectClobbers) {
  const RegisterSet live = make_sgpr_set({8, 30, 12});
  const RegisterSet clobbers = make_sgpr_set({8, 9, 30, 31});
  const RegisterSet spill = compute_spill_set(live, clobbers);
  EXPECT_TRUE(has_sgpr(spill, 8));
  EXPECT_TRUE(has_sgpr(spill, 30));
  EXPECT_FALSE(has_sgpr(spill, 12)); // live but not clobbered
  EXPECT_FALSE(has_sgpr(spill, 9));  // clobbered but not live
  EXPECT_FALSE(has_sgpr(spill, 31));
}

// Coarse SGPR-allocation gate for probe calls: a kernel must allocate through
// the fixed return-link pair s[link_base:link_base+1] to own it.
TEST(InstrumentorSgprGate, LinkPairFitsRequiresAllocationThroughPair) {
  // The link pair s[30:31] needs the kernel to allocate at least 32 SGPRs.
  EXPECT_FALSE(probe_link_pair_fits_in_kernel(/*kernel_sgpr_count=*/12, /*link_base=*/30));
  EXPECT_FALSE(probe_link_pair_fits_in_kernel(31, 30)); // owns s0..s30 only (s31 missing)
  EXPECT_TRUE(probe_link_pair_fits_in_kernel(32, 30));  // owns s0..s31
  EXPECT_TRUE(probe_link_pair_fits_in_kernel(64, 30));
  // The bound tracks link_base, not a hardcoded 32.
  EXPECT_FALSE(probe_link_pair_fits_in_kernel(9, 8));
  EXPECT_TRUE(probe_link_pair_fits_in_kernel(10, 8));
}

// Test that a builder's resource plan can feeds the Instrumentor's spill
// formula. The planner output flows straight into
// spill_set = live & (probe | builder) clobbers.
TEST(InstrumentorSpill, BuilderPlanFeedsSpillFormula) {
  // Create an empty probe_summary (has no clobbers)
  const ProbeClobberSummary probe_summary;

  TrampolinePlan plan;
  std::string err;
  // s5 live: the planner must avoid it; the rest are free.
  ASSERT_TRUE(TrampolineBuilder::plan_probe_call(
      plan, *derive_probe_abi(ProbeCallingConvention::AmdGpuFuncReturnS30S31), make_sgpr_set({5}),
      probe_summary.ordinary_clobbers, &err));

  const RegisterSet clobbers =
      compute_instrumentation_clobbers(probe_summary, plan.builder_clobbers);

  // A live value outside the envelope's registers does not spill.
  const RegisterSet spill_clear = compute_spill_set(make_sgpr_set({5}), clobbers);
  EXPECT_TRUE(spill_clear.none());

  // A live value that collides with the chosen target pair spills.
  const RegisterSet spill_hit = compute_spill_set(make_sgpr_set({plan.target_pair_base}), clobbers);
  EXPECT_TRUE(has_sgpr(spill_hit, plan.target_pair_base));
}

RegisterSet make_vgpr_set(std::initializer_list<uint16_t> indices) {
  RegisterSet set;
  for (uint16_t i : indices)
    set.expand(RegisterRef{RegClass::VGPR, i, 1});
  return set;
}

// plan_vgpr_spills reserves one slot per VGPR in for_each (ascending) order,
// starting at align_up(original_private_bytes, kDbiZoneAlignment).
TEST(InstrumentorSpill, PlanVgprSpillsReservesAscendingOffsets) {
  SpillManager spills(/*original_private_bytes=*/64, /*per_lane_scratch_limit=*/4096);
  std::vector<SpillSlot> out;
  std::string err;
  ASSERT_TRUE(plan_vgpr_spills(make_vgpr_set({5, 3}), spills, ROCJITSU_CODE_ARCH_CDNA4, out, &err))
      << err;
  ASSERT_EQ(out.size(), 2u);
  EXPECT_EQ(out[0].reg, 3u);
  EXPECT_EQ(out[0].byte_offset, 64u);
  EXPECT_EQ(out[1].reg, 5u);
  EXPECT_EQ(out[1].byte_offset, 68u);
  EXPECT_EQ(spills.total_private_bytes(), 72u);
}

// An empty spill set succeeds with no reservations.
TEST(InstrumentorSpill, PlanVgprSpillsEmptyIsSuccess) {
  SpillManager spills(0, 4096);
  std::vector<SpillSlot> out;
  std::string err;
  EXPECT_TRUE(plan_vgpr_spills(RegisterSet{}, spills, ROCJITSU_CODE_ARCH_CDNA4, out, &err));
  EXPECT_TRUE(out.empty());
}

// A non-VGPR register in the spill set fails closed and is named; nothing reserved.
TEST(InstrumentorSpill, PlanVgprSpillsRejectsNonVgpr) {
  RegisterSet spill = make_vgpr_set({2});
  spill.expand(RegisterRef{RegClass::SGPR, 7, 1});
  SpillManager spills(0, 4096);
  std::vector<SpillSlot> out;
  std::string err;
  EXPECT_FALSE(plan_vgpr_spills(spill, spills, ROCJITSU_CODE_ARCH_CDNA4, out, &err));
  EXPECT_TRUE(out.empty());
  EXPECT_NE(err.find("s7"), std::string::npos);
}

// An offset past the CDNA4 12-bit FLAT field fails even within the scratch limit.
TEST(InstrumentorSpill, PlanVgprSpillsFailsWhenOffsetExceedsField) {
  SpillManager spills(/*original_private_bytes=*/0x1000, /*per_lane_scratch_limit=*/0x4000);
  std::vector<SpillSlot> out;
  std::string err;
  EXPECT_FALSE(plan_vgpr_spills(make_vgpr_set({1}), spills, ROCJITSU_CODE_ARCH_CDNA4, out, &err));
  EXPECT_TRUE(out.empty());
  EXPECT_NE(err.find("offset"), std::string::npos);
}

// Reserving past the per-lane scratch limit fails closed.
TEST(InstrumentorSpill, PlanVgprSpillsFailsPastScratchLimit) {
  SpillManager spills(/*original_private_bytes=*/0, /*per_lane_scratch_limit=*/4); // one slot
  std::vector<SpillSlot> out;
  std::string err;
  EXPECT_FALSE(
      plan_vgpr_spills(make_vgpr_set({1, 2}), spills, ROCJITSU_CODE_ARCH_CDNA4, out, &err));
  EXPECT_TRUE(out.empty());
  EXPECT_NE(err.find("scratch limit"), std::string::npos);
}

// An arch with no scratch spill emitter fails closed.
TEST(InstrumentorSpill, PlanVgprSpillsRejectsUnsupportedArch) {
  SpillManager spills(0, 4096);
  std::vector<SpillSlot> out;
  std::string err;
  EXPECT_FALSE(plan_vgpr_spills(make_vgpr_set({1}), spills, ROCJITSU_CODE_ARCH_CDNA2, out, &err));
  EXPECT_TRUE(out.empty());
}

// RDNA4 uses the wide VSCRATCH offset and reserves normally.
TEST(InstrumentorSpill, PlanVgprSpillsRdna4) {
  SpillManager spills(0, 4096);
  std::vector<SpillSlot> out;
  std::string err;
  ASSERT_TRUE(plan_vgpr_spills(make_vgpr_set({0}), spills, ROCJITSU_CODE_ARCH_RDNA4, out, &err))
      << err;
  ASSERT_EQ(out.size(), 1u);
  EXPECT_EQ(out[0].reg, 0u);
  EXPECT_EQ(out[0].byte_offset, 0u);
}

// plan_sgpr_spills reserves a slot per SGPR and picks the lowest VGPR that is
// neither live nor already spilled as the bridge.
TEST(InstrumentorSpill, PlanSgprSpillsPicksLowestDeadBridge) {
  SpillManager spills(/*original_private_bytes=*/64, /*per_lane_scratch_limit=*/4096);
  std::vector<SpillSlot> out;
  uint16_t bridge = 0xFFFF;
  std::string err;
  // v0,v1,v2 live (v2 is also a VGPR spill -- a spilled reg is always live) -> the
  // lowest dead bridge is v3.
  const RegisterSet live = make_vgpr_set({0, 1, 2});
  const std::vector<SpillSlot> vgpr_spills{SpillSlot{RegClass::VGPR, 2, 0}};
  ASSERT_TRUE(plan_sgpr_spills(make_sgpr_set({7}), live, vgpr_spills, /*kernel_vgpr_count=*/8,
                               spills, ROCJITSU_CODE_ARCH_CDNA4, out, bridge, &err))
      << err;
  ASSERT_EQ(out.size(), 1u);
  EXPECT_EQ(out[0].reg, 7u);
  EXPECT_EQ(out[0].byte_offset, 64u);
  EXPECT_EQ(bridge, 3u);
}

// When no VGPR is dead within the kernel's allocation, the bridge falls back to a
// spilled VGPR: its value is already saved to scratch, so it can be clobbered by the
// writelane and its own reload restores it.
TEST(InstrumentorSpill, PlanSgprSpillsReusesSpilledBridgeWhenNoneDead) {
  SpillManager spills(/*original_private_bytes=*/64, /*per_lane_scratch_limit=*/4096);
  std::vector<SpillSlot> out;
  uint16_t bridge = 0xFFFF;
  std::string err;
  // v0,v1 both live (no dead VGPR in [0,2)); v1 is a spill -> bridge reuses v1.
  const RegisterSet live = make_vgpr_set({0, 1});
  const std::vector<SpillSlot> vgpr_spills{SpillSlot{RegClass::VGPR, 1, 0}};
  ASSERT_TRUE(plan_sgpr_spills(make_sgpr_set({7}), live, vgpr_spills, /*kernel_vgpr_count=*/2,
                               spills, ROCJITSU_CODE_ARCH_CDNA4, out, bridge, &err))
      << err;
  ASSERT_EQ(out.size(), 1u);
  EXPECT_EQ(out[0].reg, 7u);
  EXPECT_EQ(bridge, 1u);
}

// The bridge search is bounded by the kernel's allocated VGPR count: if every
// VGPR in [0, count) is live, no bridge exists and the spill fails closed.
TEST(InstrumentorSpill, PlanSgprSpillsFailsWhenNoBridgeWithinVgprCount) {
  SpillManager spills(0, 4096);
  std::vector<SpillSlot> out;
  uint16_t bridge = 0xFFFF;
  std::string err;
  const RegisterSet live = make_vgpr_set({0, 1}); // both allocated VGPRs live.
  ASSERT_FALSE(plan_sgpr_spills(make_sgpr_set({7}), live, {}, /*kernel_vgpr_count=*/2, spills,
                                ROCJITSU_CODE_ARCH_CDNA4, out, bridge, &err));
  EXPECT_TRUE(out.empty());
  EXPECT_NE(err.find("bridge VGPR"), std::string::npos) << "error was: " << err;
}

// A non-SGPR register in the spill set fails closed and is named.
TEST(InstrumentorSpill, PlanSgprSpillsRejectsNonSgpr) {
  SpillManager spills(0, 4096);
  std::vector<SpillSlot> out;
  uint16_t bridge = 0xFFFF;
  std::string err;
  ASSERT_FALSE(plan_sgpr_spills(make_vgpr_set({2}), RegisterSet{}, {}, /*kernel_vgpr_count=*/8,
                                spills, ROCJITSU_CODE_ARCH_CDNA4, out, bridge, &err));
  EXPECT_TRUE(out.empty());
  EXPECT_NE(err.find("v2"), std::string::npos) << "error was: " << err;
}

RegisterSet make_acc_set(std::initializer_list<uint16_t> indices) {
  RegisterSet set;
  for (uint16_t i : indices)
    set.expand(RegisterRef{RegClass::ACC_VGPR, i, 1});
  return set;
}

// plan_acc_spills reserves a slot per AccVGPR directly (no bridge) in ascending
// order, tagging each slot RegClass::ACC_VGPR so the emitter uses the `acc` bit.
TEST(InstrumentorSpill, PlanAccSpillsReservesSlots) {
  SpillManager spills(/*original_private_bytes=*/64, /*per_lane_scratch_limit=*/4096);
  std::vector<SpillSlot> out;
  std::string err;
  ASSERT_TRUE(plan_acc_spills(make_acc_set({5, 2}), /*acc_count=*/8, spills,
                              ROCJITSU_CODE_ARCH_CDNA4, out, &err))
      << err;
  ASSERT_EQ(out.size(), 2u);
  EXPECT_EQ(out[0].cls, RegClass::ACC_VGPR);
  EXPECT_EQ(out[0].reg, 2u);
  EXPECT_EQ(out[0].byte_offset, 64u);
  EXPECT_EQ(out[1].reg, 5u);
  EXPECT_EQ(out[1].byte_offset, 68u);
}

// A non-AccVGPR register in the spill set fails closed and is named.
TEST(InstrumentorSpill, PlanAccSpillsRejectsNonAcc) {
  SpillManager spills(0, 4096);
  std::vector<SpillSlot> out;
  std::string err;
  ASSERT_FALSE(plan_acc_spills(make_vgpr_set({2}), /*acc_count=*/8, spills,
                               ROCJITSU_CODE_ARCH_CDNA4, out, &err));
  EXPECT_TRUE(out.empty());
  EXPECT_NE(err.find("v2"), std::string::npos) << "error was: " << err;
}

// An AccVGPR index at or past the kernel's allocated AGPR count fails closed
// before anything is reserved (never spill an AGPR the kernel didn't allocate).
TEST(InstrumentorSpill, PlanAccSpillsRejectsIndexPastAllocatedCount) {
  SpillManager spills(0, 4096);
  std::vector<SpillSlot> out;
  std::string err;
  ASSERT_FALSE(plan_acc_spills(make_acc_set({8}), /*acc_count=*/8, spills, ROCJITSU_CODE_ARCH_CDNA4,
                               out, &err));
  EXPECT_TRUE(out.empty());
  EXPECT_NE(err.find("acc8"), std::string::npos) << "error was: " << err;
}

// An arch with no scratch spill emitter fails closed.
TEST(InstrumentorSpill, PlanAccSpillsRejectsUnsupportedArch) {
  SpillManager spills(0, 4096);
  std::vector<SpillSlot> out;
  std::string err;
  ASSERT_FALSE(plan_acc_spills(make_acc_set({1}), /*acc_count=*/8, spills, ROCJITSU_CODE_ARCH_CDNA2,
                               out, &err));
  EXPECT_TRUE(out.empty());
}

// AccVGPR spilling is CDNA-only. RDNA4 has a scratch emitter (nonzero offset cap)
// but no AccVGPR file, so plan_acc_spills rejects it up front with a CDNA-only
// diagnostic instead of letting emission throw UnimplementedInst later.
TEST(InstrumentorSpill, PlanAccSpillsRejectsNonCdnaArch) {
  SpillManager spills(0, 4096);
  std::vector<SpillSlot> out;
  std::string err;
  ASSERT_FALSE(plan_acc_spills(make_acc_set({0}), /*acc_count=*/8, spills, ROCJITSU_CODE_ARCH_RDNA4,
                               out, &err));
  EXPECT_TRUE(out.empty());
  EXPECT_NE(err.find("CDNA"), std::string::npos) << "error was: " << err;
}

//==============================================================================
// Section 7: Instrumentor probe-call patch end-to-end
//
// Patches a 2-nop target with a call to a probe exported from a separate code
// object. The cave is laid out as [probe body][trampoline], so for a target
// whose .text is 8 bytes and a 1-word probe body:
//   probe_target_offset = cave_start    = text_size = 8
//   trampoline_offset   = 8 + 4 (body)  = 12
//   anchor at offset 4, return_target   = 4 + 4 = 8
//==============================================================================

TEST(InstrumentorProbePatch, RejectsGfx1251ProbeForGfx1250Destination) {
  constexpr uint32_t kGfx1250Nop = 0xBF800000u;
  constexpr uint32_t kGfx1250SetPcS30 = 0xBE80481Eu;
  // Public LLVM gfx1251_asm_vop3p.s encoding for
  // v_pk_add_nc_u64 v[4:7], v[8:11], v[12:15].
  constexpr std::array<uint32_t, 2> kGfx1251OnlyInstruction{0xCC4C4004u, 0x1A021908u};

  auto target = make_amdgpu_kernel_elf({kGfx1250Nop, kGfx1250Nop}, /*private_bytes=*/0,
                                       /*granulated_sgpr_count=*/3, EF_AMDGPU_MACH_AMDGCN_GFX1250);
  auto probe = make_amdgpu_probe_elf(
      "rj_gfx1251_probe",
      {kGfx1251OnlyInstruction[0], kGfx1251OnlyInstruction[1], kGfx1250SetPcS30},
      EF_AMDGPU_MACH_AMDGCN_GFX1251);
  AmdGpuCodeObject obj(target.data(), target.size());
  AmdGpuCodeObject probe_obj(probe.data(), probe.size());
  ASSERT_TRUE(obj.is_valid());
  ASSERT_TRUE(probe_obj.is_valid());

  Instrumentor instrumentor(obj, ROCJITSU_CODE_ARCH_CDNA5);
  InstrumentationPoint point;
  point.anchor_offset = 4;
  point.probe_obj = &probe_obj;
  point.probe_symbol = "rj_gfx1251_probe";
  instrumentor.add_point(point);

  auto result = instrumentor.validate_points();
  EXPECT_TRUE(result.sites.empty());
  ASSERT_FALSE(result.errors.empty());
  EXPECT_NE(result.errors.front().find("concrete target"), std::string::npos)
      << "error was: " << result.errors.front();
}

TEST(InstrumentorProbePatch, EmitsValidElfWithProbeMetadata) {
  auto target = make_gfx950_kernel_elf_with_two_nops();
  auto probe = make_gfx950_probe_elf("rj_test_probe", {kProbeSetpcS30S31});
  AmdGpuCodeObject obj(target.data(), target.size());
  AmdGpuCodeObject probe_obj(probe.data(), probe.size());
  ASSERT_TRUE(obj.is_valid());
  ASSERT_TRUE(probe_obj.is_valid());

  Instrumentor instr(obj, ROCJITSU_CODE_ARCH_CDNA4);
  InstrumentationPoint pt;
  pt.anchor_offset = 4;
  pt.probe_obj = &probe_obj;
  pt.probe_symbol = "rj_test_probe";
  instr.add_point(pt);

  auto result = instr.patch_with_debug_summaries();
  ASSERT_TRUE(result.errors.empty())
      << (result.errors.empty() ? std::string{} : result.errors.front());
  EXPECT_FALSE(result.elf_bytes.empty());
  ASSERT_EQ(result.patches.size(), 1u);

  const auto &p = result.patches[0];
  EXPECT_TRUE(p.is_probe_call);
  EXPECT_EQ(p.probe_symbol, "rj_test_probe");
  EXPECT_EQ(p.anchor_offset, 4u);
  EXPECT_EQ(p.probe_target_offset, 8u); // cave_start == text_size.
  EXPECT_EQ(p.trampoline_offset, 12u);  // one-word probe body precedes it.
  EXPECT_EQ(p.return_target, 8u);
  EXPECT_EQ(p.link_pair_base, 30u);
  EXPECT_EQ(p.target_pair_base % 2u, 0u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  EXPECT_TRUE(patched.is_valid());
}

TEST(InstrumentorProbePatch, PatchedAnchorIsBranchToTrampoline) {
  auto target = make_gfx950_kernel_elf_with_two_nops();
  auto probe = make_gfx950_probe_elf("rj_test_probe", {kProbeSetpcS30S31});
  AmdGpuCodeObject obj(target.data(), target.size());
  AmdGpuCodeObject probe_obj(probe.data(), probe.size());

  Instrumentor instr(obj, ROCJITSU_CODE_ARCH_CDNA4);
  InstrumentationPoint pt;
  pt.anchor_offset = 4;
  pt.probe_obj = &probe_obj;
  pt.probe_symbol = "rj_test_probe";
  instr.add_point(pt);

  auto result = instr.patch_with_debug_summaries();
  ASSERT_TRUE(result.errors.empty())
      << (result.errors.empty() ? std::string{} : result.errors.front());
  ASSERT_EQ(result.patches.size(), 1u);

  // Forward branch from anchor (4) to trampoline (12): simm16 = (12-(4+4))/4 = 1.
  const auto &p = result.patches[0];
  ASSERT_EQ(p.patched_anchor_bytes.size(), 4u);
  uint32_t w = 0;
  std::memcpy(&w, p.patched_anchor_bytes.data(), sizeof(w));
  EXPECT_EQ(w, build_s_branch(1, ROCJITSU_CODE_ARCH_CDNA4));
}

TEST(InstrumentorProbePatch, CopiesProbeBodyOnceAndCallTargetsIt) {
  auto target = make_gfx950_kernel_elf_with_two_nops();
  auto probe = make_gfx950_probe_elf("rj_test_probe", {kProbeSetpcS30S31});
  AmdGpuCodeObject obj(target.data(), target.size());
  AmdGpuCodeObject probe_obj(probe.data(), probe.size());

  Instrumentor instr(obj, ROCJITSU_CODE_ARCH_CDNA4);
  InstrumentationPoint pt;
  pt.anchor_offset = 4;
  pt.probe_obj = &probe_obj;
  pt.probe_symbol = "rj_test_probe";
  instr.add_point(pt);

  auto result = instr.patch_with_debug_summaries();
  ASSERT_TRUE(result.errors.empty())
      << (result.errors.empty() ? std::string{} : result.errors.front());
  ASSERT_EQ(result.patches.size(), 1u);
  const auto &p = result.patches[0];

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  // The local-cave model grows .text in place. The appended region
  // [probe body][trampoline] lives at the tail of .text, starting at the
  // original text size (cave_start).
  const std::vector<uint32_t> text = section_words(patched, ".text");
  constexpr size_t kOriginalTextWords = 2; // make_gfx950_elf_with_two_nops(): 8 bytes.
  ASSERT_GT(text.size(), kOriginalTextWords);
  const std::vector<uint32_t> cave(text.begin() + kOriginalTextWords, text.end());
  ASSERT_FALSE(cave.empty());

  // The probe body is copied exactly once, at the front of the appended region.
  EXPECT_EQ(std::count(cave.begin(), cave.end(), kProbeSetpcS30S31), 1);
  EXPECT_EQ(cave.front(), kProbeSetpcS30S31);

  // The call is present, through the cc-derived link pair and chosen target pair.
  const uint32_t swappc =
      build_s_swappc_b64(p.link_pair_base, p.target_pair_base, ROCJITSU_CODE_ARCH_CDNA4);
  EXPECT_NE(std::find(cave.begin(), cave.end(), swappc), cave.end());

  // Re-derive the call target from the materialization delta and confirm it
  // lands on the copied body. s_getpc writes the VA of the next instruction; the
  // 64-bit add chain folds in (probe_target_offset - that VA).
  const size_t body_words = (p.trampoline_offset - p.probe_target_offset) / sizeof(uint32_t);
  const uint32_t getpc = build_s_getpc_b64(p.target_pair_base, ROCJITSU_CODE_ARCH_CDNA4);
  const auto getpc_it = std::find(cave.begin() + body_words, cave.end(), getpc);
  ASSERT_NE(getpc_it, cave.end());
  const size_t cave_idx = static_cast<size_t>(getpc_it - cave.begin());
  const size_t tramp_idx = cave_idx - body_words; // index within the trampoline.
  const uint64_t delta = (static_cast<uint64_t>(cave[cave_idx + 4]) << 32) | cave[cave_idx + 2];
  const uint64_t va_after_getpc = p.trampoline_offset + (tramp_idx + 1) * sizeof(uint32_t);
  EXPECT_EQ(va_after_getpc + delta, p.probe_target_offset); // wraps mod 2^64.
}

// Two anchors calling the SAME (probe_obj, symbol) share a single copied body:
// the body is emitted once and both trampolines target that one copy. Locks in
// the resolve_probe_index dedup so a regression that copies per site is caught.
TEST(InstrumentorProbePatch, TwoSitesSharingOneProbeCopyBodyOnce) {
  auto target = make_gfx950_kernel_elf_with_two_nops(); // anchors at offsets 0 and 4.
  auto probe = make_gfx950_probe_elf("rj_test_probe", {kProbeMarkerMovS5, kProbeSetpcS30S31});
  AmdGpuCodeObject obj(target.data(), target.size());
  AmdGpuCodeObject probe_obj(probe.data(), probe.size());

  Instrumentor instr(obj, ROCJITSU_CODE_ARCH_CDNA4);
  for (uint64_t anchor : {uint64_t{0}, uint64_t{4}}) {
    InstrumentationPoint pt;
    pt.anchor_offset = anchor;
    pt.probe_obj = &probe_obj;
    pt.probe_symbol = "rj_test_probe";
    instr.add_point(pt);
  }

  auto result = instr.patch_with_debug_summaries();
  ASSERT_TRUE(result.errors.empty())
      << (result.errors.empty() ? std::string{} : result.errors.front());
  ASSERT_EQ(result.patches.size(), 2u);

  // Both sites resolve to the same copied body at the same offset.
  EXPECT_EQ(result.patches[0].probe_target_offset, result.patches[1].probe_target_offset);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const std::vector<uint32_t> text = section_words(patched, ".text");
  constexpr size_t kOriginalTextWords = 2; // make_gfx950_elf_with_two_nops(): 8 bytes.
  constexpr uint64_t kCaveStart = kOriginalTextWords * sizeof(uint32_t);
  ASSERT_GT(text.size(), kOriginalTextWords);
  const std::vector<uint32_t> cave(text.begin() + kOriginalTextWords, text.end());

  // Deduped: the body's marker appears exactly once across the appended region.
  EXPECT_EQ(std::count(cave.begin(), cave.end(), kProbeMarkerMovS5), 1);
  EXPECT_EQ(cave.front(), kProbeMarkerMovS5); // body sits ahead of both trampolines.

  // Each trampoline emits its own call and targets the shared body's offset.
  for (const auto &p : result.patches) {
    EXPECT_EQ(p.probe_target_offset, kCaveStart);
    const uint32_t swappc =
        build_s_swappc_b64(p.link_pair_base, p.target_pair_base, ROCJITSU_CODE_ARCH_CDNA4);
    EXPECT_NE(std::find(cave.begin(), cave.end(), swappc), cave.end());
  }
}

// Two anchors calling DISTINCT probes: both bodies are copied once, in the order
// the probes were first requested, and each trampoline's probe_target_offset
// points at its own body. Catches ordering / wrong-body-target regressions.
TEST(InstrumentorProbePatch, TwoSitesDistinctProbesEachTargetsItsBody) {
  auto target = make_gfx950_kernel_elf_with_two_nops(); // anchors at offsets 0 and 4.
  // Distinguishable 2-word bodies: a unique leading marker then the return.
  auto probe_a = make_gfx950_probe_elf("rj_probe_a", {kProbeMarkerMovS5, kProbeSetpcS30S31});
  auto probe_b = make_gfx950_probe_elf("rj_probe_b", {kProbeMarkerMovS6, kProbeSetpcS30S31});
  AmdGpuCodeObject obj(target.data(), target.size());
  AmdGpuCodeObject probe_a_obj(probe_a.data(), probe_a.size());
  AmdGpuCodeObject probe_b_obj(probe_b.data(), probe_b.size());

  Instrumentor instr(obj, ROCJITSU_CODE_ARCH_CDNA4);
  InstrumentationPoint pt_a;
  pt_a.anchor_offset = 0; // requested first -> body A laid out first.
  pt_a.probe_obj = &probe_a_obj;
  pt_a.probe_symbol = "rj_probe_a";
  instr.add_point(pt_a);
  InstrumentationPoint pt_b;
  pt_b.anchor_offset = 4;
  pt_b.probe_obj = &probe_b_obj;
  pt_b.probe_symbol = "rj_probe_b";
  instr.add_point(pt_b);

  auto result = instr.patch_with_debug_summaries();
  ASSERT_TRUE(result.errors.empty())
      << (result.errors.empty() ? std::string{} : result.errors.front());
  ASSERT_EQ(result.patches.size(), 2u);

  AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  const std::vector<uint32_t> text = section_words(patched, ".text");
  constexpr size_t kOriginalTextWords = 2;
  constexpr uint64_t kCaveStart = kOriginalTextWords * sizeof(uint32_t);
  ASSERT_GT(text.size(), kOriginalTextWords);
  const std::vector<uint32_t> cave(text.begin() + kOriginalTextWords, text.end());

  // Each distinct body is copied exactly once.
  EXPECT_EQ(std::count(cave.begin(), cave.end(), kProbeMarkerMovS5), 1);
  EXPECT_EQ(std::count(cave.begin(), cave.end(), kProbeMarkerMovS6), 1);
  // ...and in request order: body A (marker mov s5) precedes body B (marker s6).
  const auto a_it = std::find(cave.begin(), cave.end(), kProbeMarkerMovS5);
  const auto b_it = std::find(cave.begin(), cave.end(), kProbeMarkerMovS6);
  ASSERT_NE(a_it, cave.end());
  ASSERT_NE(b_it, cave.end());
  EXPECT_LT(a_it, b_it);

  // Each patch's probe_target_offset points at the first word of its own body.
  auto marker_at = [&](uint64_t target_offset) -> uint32_t {
    const size_t idx = (target_offset - kCaveStart) / sizeof(uint32_t);
    return idx < cave.size() ? cave[idx] : 0u;
  };
  for (const auto &p : result.patches) {
    const uint32_t expected =
        p.probe_symbol == "rj_probe_a" ? kProbeMarkerMovS5 : kProbeMarkerMovS6;
    EXPECT_EQ(marker_at(p.probe_target_offset), expected)
        << "patch for " << p.probe_symbol << " targets the wrong body";
  }
  // The two bodies target different offsets.
  EXPECT_NE(result.patches[0].probe_target_offset, result.patches[1].probe_target_offset);
}

TEST(InstrumentorProbePatch, UnknownProbeSymbolFailsClosed) {
  auto target = make_gfx950_elf_with_two_nops();
  auto probe = make_gfx950_probe_elf("rj_test_probe", {kProbeSetpcS30S31});
  AmdGpuCodeObject obj(target.data(), target.size());
  AmdGpuCodeObject probe_obj(probe.data(), probe.size());

  Instrumentor instr(obj, ROCJITSU_CODE_ARCH_CDNA4);
  InstrumentationPoint pt;
  pt.anchor_offset = 4;
  pt.probe_obj = &probe_obj;
  pt.probe_symbol = "does_not_exist";
  instr.add_point(pt);

  auto result = instr.patch();
  EXPECT_TRUE(result.elf_bytes.empty());
  EXPECT_FALSE(result.errors.empty());
}

TEST(InstrumentorProbePatch, ProbeObjWithoutSymbolFailsClosed) {
  auto target = make_gfx950_elf_with_two_nops();
  auto probe = make_gfx950_probe_elf("rj_test_probe", {kProbeSetpcS30S31});
  AmdGpuCodeObject obj(target.data(), target.size());
  AmdGpuCodeObject probe_obj(probe.data(), probe.size());

  Instrumentor instr(obj, ROCJITSU_CODE_ARCH_CDNA4);
  InstrumentationPoint pt;
  pt.anchor_offset = 4;
  pt.probe_obj = &probe_obj; // symbol intentionally left empty
  instr.add_point(pt);

  auto result = instr.patch();
  EXPECT_TRUE(result.elf_bytes.empty());
  EXPECT_FALSE(result.errors.empty());
}

// A probe body that overwrites its own return-link pair (s30) before the closing
// s_setpc_b64 s[30:31] must be rejected: it would return through a corrupted PC.
// build_probe_callable accepts it (the final instruction is still the setpc), so
// the orchestrator's check_probe_link_pair is what fails closed.
TEST(InstrumentorProbePatch, ProbeClobberingLinkPairFailsClosed) {
  auto target = make_gfx950_kernel_elf_with_two_nops();
  auto probe = make_gfx950_probe_elf("rj_test_probe", {kProbeMovS30_0, kProbeSetpcS30S31});
  AmdGpuCodeObject obj(target.data(), target.size());
  AmdGpuCodeObject probe_obj(probe.data(), probe.size());

  Instrumentor instr(obj, ROCJITSU_CODE_ARCH_CDNA4);
  InstrumentationPoint pt;
  pt.anchor_offset = 4;
  pt.probe_obj = &probe_obj;
  pt.probe_symbol = "rj_test_probe";
  instr.add_point(pt);

  auto result = instr.patch();
  EXPECT_TRUE(result.elf_bytes.empty());
  ASSERT_FALSE(result.errors.empty());
  EXPECT_NE(result.errors.front().find("return-link pair"), std::string::npos)
      << "error was: " << result.errors.front();
}

// A probe reading v31 without defining it wants workitem_id_x, which only the
// kernel's own entry prologue produces. Nothing in the instrumentation path
// supplies it, so the site is rejected rather than silently handed whatever the
// instrumented kernel left in v31.
TEST(InstrumentorProbePatch, ProbeWithImplicitLiveInFailsClosed) {
  auto target = make_gfx950_kernel_elf_with_two_nops();
  auto probe = make_gfx950_probe_elf("rj_test_probe", {kProbeMovV0FromV31, kProbeSetpcS30S31});
  AmdGpuCodeObject obj(target.data(), target.size());
  AmdGpuCodeObject probe_obj(probe.data(), probe.size());

  Instrumentor instr(obj, ROCJITSU_CODE_ARCH_CDNA4);
  InstrumentationPoint pt;
  pt.anchor_offset = 4;
  pt.probe_obj = &probe_obj;
  pt.probe_symbol = "rj_test_probe";
  instr.add_point(pt);

  auto result = instr.patch();
  EXPECT_TRUE(result.elf_bytes.empty());
  ASSERT_FALSE(result.errors.empty());
  EXPECT_NE(result.errors.front().find("v31"), std::string::npos)
      << "error was: " << result.errors.front();
  EXPECT_NE(result.errors.front().find("before defining it"), std::string::npos)
      << "error was: " << result.errors.front();
}

// A probe call needs the kernel's SGPR allocation to bound temp selection. On a
// descriptorless object that bound is unknown, so the call fails closed rather than
// fall back to the device-wide default (which could pick temps past the allocation).
TEST(InstrumentorProbePatch, DescriptorlessProbeCallFailsClosed) {
  auto target = make_gfx950_elf_with_two_nops(); // no `.kd` descriptor
  auto probe = make_gfx950_probe_elf("rj_test_probe", {kProbeSetpcS30S31});
  AmdGpuCodeObject obj(target.data(), target.size());
  AmdGpuCodeObject probe_obj(probe.data(), probe.size());

  Instrumentor instr(obj, ROCJITSU_CODE_ARCH_CDNA4);
  InstrumentationPoint pt;
  pt.anchor_offset = 4;
  pt.probe_obj = &probe_obj;
  pt.probe_symbol = "rj_test_probe";
  instr.add_point(pt);

  auto result = instr.patch();
  EXPECT_TRUE(result.elf_bytes.empty());
  ASSERT_FALSE(result.errors.empty());
  EXPECT_NE(result.errors.front().find("discovered kernel descriptor"), std::string::npos)
      << "error was: " << result.errors.front();
}

// Section 8: probe-call register spilling (end-to-end)

// True if `cave` holds a SOP1 s_mov (opcode `mov_op`) whose scalar source (save,
// as_src=true) or scalar destination (restore, as_src=false) is `operand`. Used
// to spot the special-state save/restore s_mov pair regardless of which dead SGPR
// temp the planner happened to pick.
[[nodiscard]] bool cave_has_special_mov(const std::vector<uint32_t> &cave, uint32_t mov_op,
                                        uint16_t operand, bool as_src) {
  for (uint32_t w : cave) {
    if ((w >> 23) != kSop1EncodingPrefix)
      continue;
    if (((w >> 8) & 0xFFu) != mov_op)
      continue;
    const uint16_t field =
        as_src ? static_cast<uint16_t>(w & 0xFFu) : static_cast<uint16_t>((w >> 16) & 0x7Fu);
    if (field == operand)
      return true;
  }
  return false;
}

// DBI probe-call spill path: static e2e over CDNA3 (gfx942), CDNA4 (gfx950), and
// RDNA4 (gfx1200). CDNA shares one GFX9 FLAT-scratch bracket on wave64; RDNA4 uses
// VSCRATCH on wave32. One base fixture drives the shared patch/extract flow -- the
// arch differs only in its {kernel, probe} ELF factory pair and the setpc word.
// Per-arch derived fixtures pin the arch; TEST_F names read <Arch>ProbeSpill.<Case>.

// The arch-varying inputs: the ISA enum plus its {kernel, probe} ELF factories.
struct SpillArch {
  rj_code_arch_t arch;
  std::vector<uint8_t> (*make_kernel)(const std::vector<uint32_t> &, uint32_t, uint32_t, uint32_t,
                                      uint32_t);
  std::vector<uint8_t> (*make_probe)(std::string_view, const std::vector<uint32_t> &);
};
inline SpillArch cdna3_probe_spill_arch() {
  return {ROCJITSU_CODE_ARCH_CDNA3, &make_gfx942_kernel_elf, &make_gfx942_probe_elf};
}
inline SpillArch cdna4_probe_spill_arch() {
  return {ROCJITSU_CODE_ARCH_CDNA4, &make_gfx950_kernel_elf, &make_gfx950_probe_elf};
}
inline SpillArch rdna4_probe_spill_arch() {
  return {ROCJITSU_CODE_ARCH_RDNA4, &make_gfx1200_kernel_elf, &make_gfx1200_probe_elf};
}

class InstrProbeSpillBase : public ::testing::Test {
protected:
  explicit InstrProbeSpillBase(SpillArch a) : a_(a) {}
  rj_code_arch_t arch() const { return a_.arch; }
  uint32_t endpgm() const { return build_s_endpgm(a_.arch); }
  uint32_t setpc() const { return build_s_setpc_b64(/*s[30:31]=*/30, a_.arch); }

  // Cave (trampoline body appended after the original text) plus the patched
  // kernel's private_segment_fixed_size.
  struct Caved {
    std::vector<uint32_t> cave;
    uint32_t scratch = 0;
  };

  // Full probe-call patch: dispatch one point at offset 0 over `kernel_body` (the
  // original .text) with a probe of `probe_body`. Returns the cave (text past the
  // original words) and the patched scratch size; empty cave on failure.
  // granulated_vgpr_count / accum_offset shape the descriptor's VGPR/AccVGPR window
  // for AGPR tests; the defaults reproduce the historical CDNA shape.
  Caved patch_spill(const std::vector<uint32_t> &kernel_body,
                    const std::vector<uint32_t> &probe_body, uint32_t private_bytes,
                    uint32_t granulated_vgpr_count = 0, uint32_t accum_offset = 0) {
    auto target = a_.make_kernel(kernel_body, private_bytes, /*granulated_sgpr_count=*/3,
                                 granulated_vgpr_count, accum_offset);
    auto probe = a_.make_probe("rj_test_probe", probe_body);
    AmdGpuCodeObject obj(target.data(), target.size());
    AmdGpuCodeObject probe_obj(probe.data(), probe.size());
    EXPECT_TRUE(obj.is_valid());
    EXPECT_TRUE(probe_obj.is_valid());

    Instrumentor instr(obj, a_.arch);
    InstrumentationPoint pt;
    pt.anchor_offset = 0;
    pt.probe_obj = &probe_obj;
    pt.probe_symbol = "rj_test_probe";
    instr.add_point(pt);

    auto result = instr.patch_with_debug_summaries();
    EXPECT_TRUE(result.errors.empty())
        << (result.errors.empty() ? std::string{} : result.errors.front());
    if (!result.errors.empty())
      return {};
    EXPECT_EQ(result.patches.size(), 1u);
    EXPECT_TRUE(result.patches[0].is_probe_call);

    AmdGpuCodeObject patched(result.elf_bytes.data(), result.elf_bytes.size());
    EXPECT_TRUE(patched.is_valid());
    const std::vector<uint32_t> text = section_words(patched, ".text");
    if (text.size() <= kernel_body.size())
      return {};
    return {std::vector<uint32_t>(text.begin() + kernel_body.size(), text.end()),
            patched_private_segment_size(patched)};
  }

  // Arch's single-kernel target ELF / probe ELF for this fixture's arch.
  std::vector<uint8_t> kernel_elf(const std::vector<uint32_t> &body, uint32_t private_bytes,
                                  uint32_t granulated_vgpr_count = 0, uint32_t accum_offset = 0) {
    return a_.make_kernel(body, private_bytes, /*granulated_sgpr_count=*/3, granulated_vgpr_count,
                          accum_offset);
  }
  std::vector<uint8_t> probe_elf(const std::vector<uint32_t> &body) {
    return a_.make_probe("rj_test_probe", body);
  }

  // Fail-closed patch over prebuilt ELF bytes: returns the raw result for
  // empty-elf / error-substring checks (covers the two-kernel case too).
  InstrumentedCodeObject patch_expecting_result(const std::vector<uint8_t> &target,
                                                const std::vector<uint8_t> &probe) {
    AmdGpuCodeObject obj(target.data(), target.size());
    AmdGpuCodeObject probe_obj(probe.data(), probe.size());
    Instrumentor instr(obj, a_.arch);
    InstrumentationPoint pt;
    pt.anchor_offset = 0;
    pt.probe_obj = &probe_obj;
    pt.probe_symbol = "rj_test_probe";
    instr.add_point(pt);
    return instr.patch();
  }

  // Assert a fail-closed patch: empty ELF and an error mentioning `needle`.
  void expect_failure_containing(const InstrumentedCodeObject &r, std::string_view needle) {
    EXPECT_TRUE(r.elf_bytes.empty());
    ASSERT_FALSE(r.errors.empty());
    EXPECT_NE(r.errors.front().find(needle), std::string::npos)
        << "error was: " << r.errors.front();
  }

  // Patch a probe that clobbers one special register on a zero-scratch kernel
  // whose anchor (v_mov v3, v2) needs no spill, isolating the save/restore.
  // The compare's vsrc1 is v0, so the body defines v0 before reading it. A probe
  // that read v0 cold would be rejected as having an implicit live-in, which is
  // a different gate than the special-state preservation under test here.
  std::vector<uint32_t> patch_probe_clobbering(uint32_t probe_clobber_word) {
    Caved c = patch_spill({kMovV3V2, endpgm()}, {kMovV0Zero, probe_clobber_word, setpc()},
                          /*private_bytes=*/0);
    EXPECT_EQ(c.scratch, 0u) << "special-state preservation needs no scratch";
    return c.cave;
  }

  // The architecture-complete load drain opens the probe-call envelope, ahead of the
  // special-state saves and `first_spill_op` (the first store on the VGPR path, the
  // first writelane on the SGPR path), so a register whose producing load is still
  // pending is stored final-valued, not stale.
  void expect_drain_before_store(const std::vector<uint32_t> &cave,
                                 const std::vector<uint32_t> &first_spill_op) {
    const auto drain = build_wait_all_loads_complete(a_.arch);
    auto op = std::search(cave.begin(), cave.end(), first_spill_op.begin(), first_spill_op.end());
    ASSERT_NE(op, cave.end());
    EXPECT_NE(std::search(cave.begin(), op, drain.begin(), drain.end()), op)
        << "no load drain before the first spill op";
  }

  // The matching drain is emitted immediately after the probe call returns, ahead of
  // any restoration, so a probe's own still-in-flight load cannot retire onto a just-
  // restored register -- on RDNA4 the per-fill s_wait_loadcnt misses a scalar s_load
  // (KMCNT). It sits after the prologue op (`first_spill_op`, before the call) and
  // before `first_fill_load` (the first epilogue scratch load), i.e. after the
  // s_swappc_b64 return.
  void expect_drain_after_return(const std::vector<uint32_t> &cave,
                                 const std::vector<uint32_t> &first_spill_op,
                                 const std::vector<uint32_t> &first_fill_load) {
    const auto drain = build_wait_all_loads_complete(a_.arch);
    auto op = std::search(cave.begin(), cave.end(), first_spill_op.begin(), first_spill_op.end());
    ASSERT_NE(op, cave.end());
    auto fill = std::search(op, cave.end(), first_fill_load.begin(), first_fill_load.end());
    ASSERT_NE(fill, cave.end());
    EXPECT_NE(std::search(op, fill, drain.begin(), drain.end()), fill)
        << "no load drain after the call return, before the first spill fill";
  }

  // Assert v`vgpr`'s spill words are in `cave`: a scratch store (prologue) and load
  // (epilogue) at slot `off`. Checks the words appear, not their order or the
  // descriptor size -- callers assert the load wait, drain, and scratch growth.
  void expect_vgpr_spill_present(const std::vector<uint32_t> &cave, uint16_t vgpr, uint32_t off) {
    const auto store = build_scratch_store_dword(vgpr, off, a_.arch);
    const auto load = build_scratch_load_dword(vgpr, off, a_.arch);
    EXPECT_NE(std::search(cave.begin(), cave.end(), store.begin(), store.end()), cave.end())
        << "missing store for v" << vgpr;
    EXPECT_NE(std::search(cave.begin(), cave.end(), load.begin(), load.end()), cave.end())
        << "missing load for v" << vgpr;
  }

  // Assert s`sgpr`'s spill words are in `cave`: writelane + scratch store (prologue)
  // and scratch load + readlane (epilogue), bridged through v0 at slot `off`. Checks
  // the words appear, not their order or the descriptor size -- callers assert the
  // load wait, drain, and scratch growth.
  void expect_sgpr_spill_present(const std::vector<uint32_t> &cave, uint16_t sgpr, uint32_t off) {
    const auto writelane = build_v_writelane_b32(/*bridge=*/0, sgpr, /*lane=*/0, a_.arch);
    const auto store = build_scratch_store_dword(/*bridge=*/0, off, a_.arch);
    const auto load = build_scratch_load_dword(/*bridge=*/0, off, a_.arch);
    const auto readlane = build_v_readlane_b32(sgpr, /*bridge=*/0, /*lane=*/0, a_.arch);
    EXPECT_NE(std::search(cave.begin(), cave.end(), writelane.begin(), writelane.end()), cave.end())
        << "missing writelane for s" << sgpr;
    EXPECT_NE(std::search(cave.begin(), cave.end(), store.begin(), store.end()), cave.end())
        << "missing store for s" << sgpr;
    EXPECT_NE(std::search(cave.begin(), cave.end(), load.begin(), load.end()), cave.end())
        << "missing load for s" << sgpr;
    EXPECT_NE(std::search(cave.begin(), cave.end(), readlane.begin(), readlane.end()), cave.end())
        << "missing readlane for s" << sgpr;
  }

  // Assert AccVGPR acc`acc`'s spill words are in `cave`: a scratch store (prologue) and
  // load (epilogue) at slot `off`, both carrying the `acc` bit (a direct accumulator-file
  // spill, no bridge). Checks the words appear, not their order or the descriptor size.
  void expect_acc_spill_present(const std::vector<uint32_t> &cave, uint16_t acc, uint32_t off) {
    const auto store = build_scratch_store_dword(acc, off, a_.arch, /*acc=*/true);
    const auto load = build_scratch_load_dword(acc, off, a_.arch, /*acc=*/true);
    EXPECT_NE(std::search(cave.begin(), cave.end(), store.begin(), store.end()), cave.end())
        << "missing store for acc" << acc;
    EXPECT_NE(std::search(cave.begin(), cave.end(), load.begin(), load.end()), cave.end())
        << "missing load for acc" << acc;
  }

  // Assert the standard single-VGPR spill in `caved`: v2 stored/reloaded at slot 64,
  // the reload guarded by a load wait, the boundary drain before the store and after
  // the call return, and the descriptor grown by one slot (64 -> 68).
  void expect_vgpr_spill(const Caved &caved) {
    const std::vector<uint32_t> &cave = caved.cave;
    expect_vgpr_spill_present(cave, /*vgpr=*/2, /*off=*/64);
    EXPECT_NE(std::find(cave.begin(), cave.end(), build_wait_loads_complete(a_.arch)), cave.end());
    const auto store = build_scratch_store_dword(/*vgpr=*/2, /*off=*/64, a_.arch);
    expect_drain_before_store(cave, store);
    expect_drain_after_return(cave, store,
                              build_scratch_load_dword(/*vgpr=*/2, /*off=*/64, a_.arch));
    EXPECT_EQ(caved.scratch, 68u);
  }

  // Assert the standard single-SGPR spill in `caved`: s8 bridged through v0, stored/reloaded
  // at slot 64, the reload guarded by a load wait, the boundary drain before the first spill
  // op (the writelane) and after the call return, and the descriptor grown by one slot
  // (64 -> 68).
  void expect_sgpr_spill(const Caved &caved) {
    const std::vector<uint32_t> &cave = caved.cave;
    expect_sgpr_spill_present(cave, /*sgpr=*/8, /*off=*/64);
    EXPECT_NE(std::find(cave.begin(), cave.end(), build_wait_loads_complete(a_.arch)), cave.end());
    const auto writelane = build_v_writelane_b32(/*bridge=*/0, /*sgpr=*/8, /*lane=*/0, a_.arch);
    const std::vector<uint32_t> writelane_words{writelane.begin(), writelane.end()};
    expect_drain_before_store(cave, writelane_words);
    expect_drain_after_return(cave, writelane_words,
                              build_scratch_load_dword(/*bridge=*/0, /*off=*/64, a_.arch));
    EXPECT_EQ(caved.scratch, 68u);
  }

  // Assert the standard single-AccVGPR spill in `caved`: acc0 stored/reloaded at slot 64
  // via the `acc` bit (no bridge), the prologue drain immediately before the store and the
  // boundary drain after the call return before the fill, and the descriptor grown by one
  // slot (64 -> 68). The positional drain checks matter because on CDNA the drain, load wait,
  // and store wait all encode as the same s_waitcnt 0, so a bare "an s_waitcnt 0 exists" search
  // would pass even with the store's guarding drain gone.
  void expect_acc_spill(const Caved &caved) {
    const std::vector<uint32_t> &cave = caved.cave;
    expect_acc_spill_present(cave, /*acc=*/0, /*off=*/64);
    EXPECT_NE(std::find(cave.begin(), cave.end(), build_wait_loads_complete(a_.arch)), cave.end());
    const auto store = build_scratch_store_dword(/*acc=*/0, /*off=*/64, a_.arch, /*acc=*/true);
    expect_drain_before_store(cave, store);
    expect_drain_after_return(
        cave, store, build_scratch_load_dword(/*acc=*/0, /*off=*/64, a_.arch, /*acc=*/true));
    EXPECT_EQ(caved.scratch, 68u);
  }

  // Assert an AccVGPR tuple spill in `caved`: acc0-3 each stored/reloaded straight to
  // scratch via the `acc` bit in acc order at slots 64/68/72/76, the reload guarded by a
  // load wait, and the descriptor grown to cover all four (64 -> 80). Used by the MFMA
  // acc_cd fixtures.
  void expect_acc_tuple_spill(const Caved &caved) {
    const std::vector<uint32_t> &cave = caved.cave;
    expect_acc_spill_present(cave, /*acc=*/0, /*off=*/64);
    expect_acc_spill_present(cave, /*acc=*/1, /*off=*/68);
    expect_acc_spill_present(cave, /*acc=*/2, /*off=*/72);
    expect_acc_spill_present(cave, /*acc=*/3, /*off=*/76);
    EXPECT_NE(std::find(cave.begin(), cave.end(), build_wait_loads_complete(a_.arch)), cave.end());
    EXPECT_EQ(caved.scratch, 80u);
  }

  // Assert the cave both saves and restores `operand` via an s_mov of `mov_op`.
  void expect_special_preserved(const std::vector<uint32_t> &cave, uint32_t mov_op,
                                uint16_t operand) {
    EXPECT_TRUE(cave_has_special_mov(cave, mov_op, operand, /*as_src=*/true))
        << "missing special-state save (s_mov tmp, <reg>)";
    EXPECT_TRUE(cave_has_special_mov(cave, mov_op, operand, /*as_src=*/false))
        << "missing special-state restore (s_mov <reg>, tmp)";
  }

  // A kernel with zero fixed scratch cannot be spilled into and fails closed.
  void expect_zero_scratch_fails_closed() {
    expect_failure_containing(
        patch_expecting_result(kernel_elf({kMovV3V2, endpgm()}, /*private_bytes=*/0),
                               probe_elf({kMovV2Zero, setpc()})),
        "fixed scratch");
  }

  // A probe that clobbers EXEC is preserved: s_mov_b64 tmp, exec (save) / s_mov_b64
  // exec, tmp (restore). On RDNA4 the save/restore stays wave64-shaped even on wave32
  // (width hardcoded to 2); EXEC_LO round-trips correctly.
  void expect_preserves_clobbered_exec() {
    const uint32_t clobber =
        build_s_mov_b32(scalar_operand_exec_lo(arch()), /*inline 0=*/128, arch());
    const std::vector<uint32_t> cave = patch_probe_clobbering(clobber);
    ASSERT_FALSE(cave.empty());
    expect_special_preserved(cave, sop1_op_mov_b64(arch()), scalar_operand_exec_lo(arch()));
  }

  // VCC is preserved the same way (64-bit pair).
  void expect_preserves_clobbered_vcc() {
    const uint32_t clobber =
        build_s_mov_b32(scalar_operand_vcc_lo(arch()), /*inline 0=*/128, arch());
    const std::vector<uint32_t> cave = patch_probe_clobbering(clobber);
    ASSERT_FALSE(cave.empty());
    expect_special_preserved(cave, sop1_op_mov_b64(arch()), scalar_operand_vcc_lo(arch()));
  }

  // M0 is preserved with a single 32-bit s_mov (m0 = 124 on CDNA4, 125 on RDNA4).
  void expect_preserves_clobbered_m0() {
    const uint16_t m0 = scalar_operand_m0(arch());
    const std::vector<uint32_t> cave =
        patch_probe_clobbering(build_s_mov_b32(m0, /*inline 0=*/128, arch()));
    ASSERT_FALSE(cave.empty());
    expect_special_preserved(cave, sop1_op_mov_b32(arch()), m0);
  }

  // EXEC and VCC are preserved unconditionally, even when the probe body touches
  // neither explicitly. The clobber summary sees only explicit operands, so implicit
  // defs (v_cmp->VCC, v_cmpx->EXEC) would otherwise slip through; always saving both
  // closes that hole. Probe body here is a bare s_nop (no special-state operand).
  void expect_preserves_exec_and_vcc_unconditionally() {
    const std::vector<uint32_t> cave = patch_probe_clobbering(build_s_nop(0, arch()));
    ASSERT_FALSE(cave.empty());
    expect_special_preserved(cave, sop1_op_mov_b64(arch()), scalar_operand_exec_lo(arch()));
    expect_special_preserved(cave, sop1_op_mov_b64(arch()), scalar_operand_vcc_lo(arch()));
  }

  // Every probe-call envelope -- including a no-spill site -- opens with a load drain
  // and emits a matching drain right after the call returns, so a pre-anchor or
  // probe-issued asynchronous load cannot retire onto the saved/restored special
  // state. The probe body is a bare s_nop (no spill; special state still preserved).
  void expect_boundary_drains_at_no_spill_site() {
    const std::vector<uint32_t> cave = patch_probe_clobbering(build_s_nop(0, arch()));
    ASSERT_FALSE(cave.empty());
    const auto drain = build_wait_all_loads_complete(arch());
    size_t count = 0;
    for (auto it = std::search(cave.begin(), cave.end(), drain.begin(), drain.end());
         it != cave.end();
         it = std::search(it + drain.size(), cave.end(), drain.begin(), drain.end()))
      ++count;
    EXPECT_GE(count, 2u) << "a no-spill probe envelope must still be bracketed by boundary drains";
  }

  // FLAT_SCRATCH stays rejected (the spill store/load depend on it), failing closed.
  // gfx9 only (CDNA3, CDNA4): flat_scratch_lo/hi are writable operands (102/103) here;
  // gfx11+ removed them (OPR_SDST_SGPR_MAX=105, so dst 102 is the plain SGPR s102), so
  // the clobber is inexpressible via s_mov on RDNA4 -- no RDNA4 analog.
  void expect_rejects_clobbered_flat_scratch() {
    // FLAT_SCRATCH_LO is operand 102 on gfx9.
    const uint32_t clobber = build_s_mov_b32(/*flat_scratch_lo=*/102, /*inline 0=*/128, arch());
    expect_failure_containing(
        patch_expecting_result(kernel_elf({kMovV3V2, endpgm()}, /*private_bytes=*/0),
                               probe_elf({clobber, setpc()})),
        "FLAT_SCRATCH");
  }

  SpillArch a_;
};

class Cdna3ProbeSpill : public InstrProbeSpillBase {
protected:
  Cdna3ProbeSpill() : InstrProbeSpillBase(cdna3_probe_spill_arch()) {}
};
class Cdna4ProbeSpill : public InstrProbeSpillBase {
protected:
  Cdna4ProbeSpill() : InstrProbeSpillBase(cdna4_probe_spill_arch()) {}
};
class Rdna4ProbeSpill : public InstrProbeSpillBase {
protected:
  Rdna4ProbeSpill() : InstrProbeSpillBase(rdna4_probe_spill_arch()) {}
};

TEST_F(Cdna3ProbeSpill, SpillsLiveClobberedVgpr) {
  expect_vgpr_spill(patch_spill({kMovV3V2, endpgm()}, {kMovV2Zero, setpc()}, 64));
}

TEST_F(Cdna4ProbeSpill, SpillsLiveClobberedVgpr) {
  expect_vgpr_spill(patch_spill({kMovV3V2, endpgm()}, {kMovV2Zero, setpc()}, 64));
}

// RDNA4 spills to VSCRATCH; additionally the STORECNT fence must sit after the store
// and before the reload, since s_wait_loadcnt does not order stores on RDNA4.
TEST_F(Rdna4ProbeSpill, SpillsLiveClobberedVgpr) {
  const Caved caved = patch_spill({kMovV3V2, endpgm()}, {kMovV2Zero, setpc()}, 64);
  expect_vgpr_spill(caved);

  const std::vector<uint32_t> &cave = caved.cave;
  const auto store = build_scratch_store_dword(2, 64, arch());
  const auto load = build_scratch_load_dword(2, 64, arch());
  const auto store_it = std::search(cave.begin(), cave.end(), store.begin(), store.end());
  const auto load_it = std::search(cave.begin(), cave.end(), load.begin(), load.end());
  const uint32_t store_wait = build_wait_stores_complete(arch());
  const auto store_wait_it = std::find(cave.begin(), cave.end(), store_wait);
  ASSERT_NE(store_wait_it, cave.end()) << "missing RDNA4 store-counter fence before the reload";
  EXPECT_LT(store_it, store_wait_it) << "store-counter fence must follow the store";
  EXPECT_LT(store_wait_it, load_it) << "store-counter fence must precede the reload";
}

TEST_F(Cdna3ProbeSpill, SpillsLiveClobberedSgpr) {
  expect_sgpr_spill(patch_spill(
      {kMovV3S8, endpgm()}, {build_s_mov_b32(/*s8=*/8, /*inline 0=*/128, arch()), setpc()}, 64));
}

TEST_F(Cdna4ProbeSpill, SpillsLiveClobberedSgpr) {
  expect_sgpr_spill(patch_spill(
      {kMovV3S8, endpgm()}, {build_s_mov_b32(/*s8=*/8, /*inline 0=*/128, arch()), setpc()}, 64));
}

TEST_F(Rdna4ProbeSpill, SpillsLiveClobberedSgpr) {
  expect_sgpr_spill(patch_spill(
      {kMovV3S8, endpgm()}, {build_s_mov_b32(/*s8=*/8, /*inline 0=*/128, arch()), setpc()}, 64));
}

// A live+clobbered AccVGPR spills straight to scratch via the `acc` bit -- no bridge
// VGPR, no writelane/readlane. AGPRs are CDNA-only, so there is no RDNA4 variant. Relies
// on the decoder tagging v_accvgpr_read (src) and v_accvgpr_write (dst) operands as
// ACC_VGPR so they reach the spill set.
TEST_F(Cdna3ProbeSpill, SpillsLiveClobberedAccVgpr) {
  expect_acc_spill(patch_spill({kAccReadV3A0Lo, kAccReadV3A0Hi, endpgm()},
                               {kAccWriteA0ZeroLo, kAccWriteA0ZeroHi, setpc()}, 64));
}

TEST_F(Cdna4ProbeSpill, SpillsLiveClobberedAccVgpr) {
  expect_acc_spill(patch_spill({kAccReadV3A0Lo, kAccReadV3A0Hi, endpgm()},
                               {kAccWriteA0ZeroLo, kAccWriteA0ZeroHi, setpc()}, 64));
}

// The AccVGPR window size is descriptor-derived: acc_count = kernel_vgpr_count -
// accum_base. With ACCUM_OFFSET=1 the accumulator base rises to v8 == the 8-VGPR
// allocation, so the window is empty and a probe clobbering live acc0 must fail
// closed -- even though the same kernel with ACCUM_OFFSET=0
// (Cdna4SpillsLiveClobberedAccVgpr) spills acc0 fine. Pins both sides of the
// descriptor-derived boundary end to end.
TEST_F(Cdna4ProbeSpill, RejectsAccVgprPastAccumOffsetWindow) {
  const InstrumentedCodeObject result = patch_expecting_result(
      kernel_elf({kAccReadV3A0Lo, kAccReadV3A0Hi, endpgm()}, /*private_bytes=*/64,
                 /*granulated_vgpr_count=*/0, /*accum_offset=*/1),
      probe_elf({kAccWriteA0ZeroLo, kAccWriteA0ZeroHi, setpc()}));
  ASSERT_FALSE(result.errors.empty());
  EXPECT_NE(result.errors.front().find("allocated count"), std::string::npos)
      << result.errors.front();
}

// AccVGPR liveness must be classified through an MFMA's single acc_cd operand, not only the
// one-lane v_accvgpr_read/write forms. v_mfma_f32_16x16x16_f16 a[0:3], v[0:1], v[2:3],
// a[0:3] (acc_cd=1) reads and writes acc[0:3], so all four lanes are live at the anchor; a
// probe clobbering acc0..acc3 must spill the whole tuple. The default descriptor (8 VGPRs,
// accum_base=4) allocates exactly acc0-3 -> slots 64/68/72/76. AGPRs are CDNA-only, so no
// RDNA4 variant.
TEST_F(Cdna3ProbeSpill, SpillsMfmaAccVgprTupleLiveAcrossProbe) {
  expect_acc_tuple_spill(
      patch_spill({kMfmaF32_16x16x16F16_A0to3_Lo, kMfmaF32_16x16x16F16_A0to3_Hi, endpgm()},
                  {make_accvgpr_write_lo(0), kAccWriteA0ZeroHi, make_accvgpr_write_lo(1),
                   kAccWriteA0ZeroHi, make_accvgpr_write_lo(2), kAccWriteA0ZeroHi,
                   make_accvgpr_write_lo(3), kAccWriteA0ZeroHi, setpc()},
                  64));
}

TEST_F(Cdna4ProbeSpill, SpillsMfmaAccVgprTupleLiveAcrossProbe) {
  expect_acc_tuple_spill(
      patch_spill({kMfmaF32_16x16x16F16_A0to3_Lo, kMfmaF32_16x16x16F16_A0to3_Hi, endpgm()},
                  {make_accvgpr_write_lo(0), kAccWriteA0ZeroHi, make_accvgpr_write_lo(1),
                   kAccWriteA0ZeroHi, make_accvgpr_write_lo(2), kAccWriteA0ZeroHi,
                   make_accvgpr_write_lo(3), kAccWriteA0ZeroHi, setpc()},
                  64));
}

// A kernel with zero fixed scratch cannot be spilled into and fails closed.
TEST_F(Cdna3ProbeSpill, ZeroScratchFailsClosed) { expect_zero_scratch_fails_closed(); }
TEST_F(Cdna4ProbeSpill, ZeroScratchFailsClosed) { expect_zero_scratch_fails_closed(); }
TEST_F(Rdna4ProbeSpill, ZeroScratchFailsClosed) { expect_zero_scratch_fails_closed(); }

TEST_F(Cdna3ProbeSpill, PreservesClobberedExec) { expect_preserves_clobbered_exec(); }
TEST_F(Cdna4ProbeSpill, PreservesClobberedExec) { expect_preserves_clobbered_exec(); }
TEST_F(Rdna4ProbeSpill, PreservesClobberedExec) { expect_preserves_clobbered_exec(); }

TEST_F(Cdna3ProbeSpill, PreservesClobberedVcc) { expect_preserves_clobbered_vcc(); }
TEST_F(Cdna4ProbeSpill, PreservesClobberedVcc) { expect_preserves_clobbered_vcc(); }
TEST_F(Rdna4ProbeSpill, PreservesClobberedVcc) { expect_preserves_clobbered_vcc(); }

TEST_F(Cdna3ProbeSpill, PreservesClobberedM0) { expect_preserves_clobbered_m0(); }
TEST_F(Cdna4ProbeSpill, PreservesClobberedM0) { expect_preserves_clobbered_m0(); }
TEST_F(Rdna4ProbeSpill, PreservesClobberedM0) { expect_preserves_clobbered_m0(); }

TEST_F(Cdna3ProbeSpill, PreservesExecAndVccUnconditionally) {
  expect_preserves_exec_and_vcc_unconditionally();
}
TEST_F(Cdna4ProbeSpill, PreservesExecAndVccUnconditionally) {
  expect_preserves_exec_and_vcc_unconditionally();
}
TEST_F(Rdna4ProbeSpill, PreservesExecAndVccUnconditionally) {
  expect_preserves_exec_and_vcc_unconditionally();
}

TEST_F(Cdna3ProbeSpill, BoundaryDrainsAtNoSpillSite) { expect_boundary_drains_at_no_spill_site(); }
TEST_F(Cdna4ProbeSpill, BoundaryDrainsAtNoSpillSite) { expect_boundary_drains_at_no_spill_site(); }
TEST_F(Rdna4ProbeSpill, BoundaryDrainsAtNoSpillSite) { expect_boundary_drains_at_no_spill_site(); }

// Compare-based coverage: v_cmp writes VCC and v_cmpx writes EXEC. The always-on
// preservation brackets the call regardless of the summary, so VCC/EXEC round-trip
// even for a probe whose special-state write the summary might miss. The VOPC
// encodings are arch-specific single words, so each test passes its own.
TEST_F(Cdna3ProbeSpill, PreservesVccFromImplicitCompare) {
  const std::vector<uint32_t> cave =
      patch_probe_clobbering(/*v_cmp_eq_u32 vcc, 0, v0=*/0x7D940080u);
  ASSERT_FALSE(cave.empty());
  expect_special_preserved(cave, sop1_op_mov_b64(arch()), scalar_operand_vcc_lo(arch()));
}

TEST_F(Cdna3ProbeSpill, PreservesExecFromImplicitCmpx) {
  const std::vector<uint32_t> cave = patch_probe_clobbering(/*v_cmpx_eq_u32 0, v0=*/0x7DB40080u);
  ASSERT_FALSE(cave.empty());
  expect_special_preserved(cave, sop1_op_mov_b64(arch()), scalar_operand_exec_lo(arch()));
}

TEST_F(Cdna4ProbeSpill, PreservesVccFromImplicitCompare) {
  const std::vector<uint32_t> cave =
      patch_probe_clobbering(/*v_cmp_eq_u32 vcc, 0, v0=*/0x7D940080u);
  ASSERT_FALSE(cave.empty());
  expect_special_preserved(cave, sop1_op_mov_b64(arch()), scalar_operand_vcc_lo(arch()));
}

TEST_F(Cdna4ProbeSpill, PreservesExecFromImplicitCmpx) {
  const std::vector<uint32_t> cave = patch_probe_clobbering(/*v_cmpx_eq_u32 0, v0=*/0x7DB40080u);
  ASSERT_FALSE(cave.empty());
  expect_special_preserved(cave, sop1_op_mov_b64(arch()), scalar_operand_exec_lo(arch()));
}

TEST_F(Rdna4ProbeSpill, PreservesVccFromImplicitCompare) {
  const std::vector<uint32_t> cave =
      patch_probe_clobbering(/*v_cmp_eq_u32 vcc, 0, v0=*/0x7C940080u);
  ASSERT_FALSE(cave.empty());
  expect_special_preserved(cave, sop1_op_mov_b64(arch()), scalar_operand_vcc_lo(arch()));
}

TEST_F(Rdna4ProbeSpill, PreservesExecFromImplicitCmpx) {
  const std::vector<uint32_t> cave = patch_probe_clobbering(/*v_cmpx_eq_u32 0, v0=*/0x7D940080u);
  ASSERT_FALSE(cave.empty());
  expect_special_preserved(cave, sop1_op_mov_b64(arch()), scalar_operand_exec_lo(arch()));
}

TEST_F(Cdna3ProbeSpill, RejectsClobberedFlatScratch) { expect_rejects_clobbered_flat_scratch(); }
TEST_F(Cdna4ProbeSpill, RejectsClobberedFlatScratch) { expect_rejects_clobbered_flat_scratch(); }

// A site that must spill on a code object with more than one kernel has no
// unambiguous descriptor to grow, so it fails closed (instrumentor.cpp:768).
TEST_F(Cdna4ProbeSpill, MultiKernelSpillFailsClosed) {
  auto target = make_gfx950_two_kernel_elf({kMovV3V2, endpgm()}, /*private_bytes=*/64);
  auto probe = probe_elf({kMovV2Zero, setpc()});
  AmdGpuCodeObject obj(target.data(), target.size());
  ASSERT_TRUE(obj.is_valid());
  const Section *text = obj.text_sections().front();
  ASSERT_EQ(scan_kernel_descriptors(
                {reinterpret_cast<const uint8_t *>(obj.image_data()), obj.image_size()},
                text->sectionOffset(), text->size())
                .size(),
            2u);

  expect_failure_containing(patch_expecting_result(target, probe), "single kernel");
}

// A site whose spill set contains a live+clobbered VGPR, SGPR, and AccVGPR spills all
// three in one bracket: the VGPR and AccVGPR straight to scratch, the SGPR bridged
// through v0. Allocation is VGPR -> SGPR -> ACC, so v2 -> slot 64, s8 -> 68, acc0 -> 72;
// the descriptor grows to cover all three -> 76.
TEST_F(Cdna4ProbeSpill, SpillsLiveClobberedVgprSgprAndAccVgpr) {
  // Anchor reads v2; the next two instructions read s8 then a0, so v2, s8, and acc0 are
  // all live at the anchor and all clobbered by the probe. Distinct readback dests
  // (v5/v6/v7) so the sim e2e can share this exact kernel shape.
  // 16 VGPRs with ACCUM_OFFSET=1 puts the accumulator window at v8-v15, so v2 and
  // the v5/v6/v7 dests are genuine ordinary VGPRs and acc0 sits in a nonempty
  // window (the SGPR bridge is then bounded to the ordinary prefix v0-v7).
  const Caved caved =
      patch_spill({kMovV5V2, kMovV6S8, kAccReadV7A0Lo, kAccReadV7A0Hi, endpgm()},
                  {kMovV2Zero, kMovS8Zero, kAccWriteA0ZeroLo, kAccWriteA0ZeroHi, setpc()}, 64,
                  /*granulated_vgpr_count=*/1, /*accum_offset=*/1);
  const std::vector<uint32_t> &cave = caved.cave;

  // VGPR v2 -> slot 64, straight to scratch.
  expect_vgpr_spill_present(cave, /*vgpr=*/2, /*off=*/64);
  // SGPR s8 -> slot 68, bridged through v0.
  expect_sgpr_spill_present(cave, /*sgpr=*/8, /*off=*/68);
  // AccVGPR acc0 -> slot 72, straight to scratch via the acc bit.
  expect_acc_spill_present(cave, /*acc=*/0, /*off=*/72);
  EXPECT_NE(std::find(cave.begin(), cave.end(), build_wait_loads_complete(arch())), cave.end());

  // Three 4-byte slots on top of the 16-aligned base: 64 -> 76.
  EXPECT_EQ(caved.scratch, 76u);
}

// End-to-end guard for the orchestrator-derived bridge bound. With the default CDNA4
// descriptor (kernel_vgpr_count=8, accum_base=4) v0-v3 are the entire ordinary-VGPR
// prefix; here all four are live at the anchor and s8 must spill. The bridge is bounded
// to min(kernel_vgpr_count, accum_base)=4, so no ordinary VGPR is free and the patch
// fails closed. A regression that used kernel_vgpr_count as the bound would pick v4,
// which aliases acc0 -- silently corrupting an AGPR outside acc_spills. Exercises the
// orchestrator's own accum_base computation, unlike a unit call given a clamped bound.
TEST_F(Cdna4ProbeSpill, SgprBridgeBoundedToOrdinaryVgprPrefix) {
  // Anchor (v_mov v5,v0) reads v0; the next three read v1,v2,v3 and then s8, so v0-v3
  // and s8 are all live at the anchor. v5/v6 are dead sinks, not spill candidates.
  const InstrumentedCodeObject result = patch_expecting_result(
      kernel_elf({kMovV5V0, kMovV5V1, kMovV5V2, kMovV5V3, kMovV6S8, endpgm()},
                 /*private_bytes=*/64),
      probe_elf({kMovS8Zero, setpc()}));
  expect_failure_containing(result, "bridge VGPR");
}

// Two live+clobbered SGPRs spill through a single shared bridge VGPR (v0), each to
// its own scratch slot (s8 -> 64, s9 -> 68); descriptor grows to cover both -> 72.
TEST_F(Cdna4ProbeSpill, SpillsMultipleLiveClobberedSgprs) {
  // Anchor reads s8; the next instruction reads s9, so both are live at the anchor.
  const Caved caved =
      patch_spill({kMovV3S8, kMovV4S9, endpgm()}, {kMovS8Zero, kMovS9Zero, setpc()}, 64);
  expect_sgpr_spill_present(caved.cave, /*sgpr=*/8, /*off=*/64);
  expect_sgpr_spill_present(caved.cave, /*sgpr=*/9, /*off=*/68);
  EXPECT_NE(std::find(caved.cave.begin(), caved.cave.end(), build_wait_loads_complete(arch())),
            caved.cave.end());
  EXPECT_EQ(caved.scratch, 72u);
}

} // namespace
} // namespace rocjitsu
