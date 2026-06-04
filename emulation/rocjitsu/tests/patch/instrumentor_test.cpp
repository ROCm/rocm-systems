// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/instrumentor.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/patch/instruction_builder.h"
#include "rocjitsu/code/patch/trampoline_builder.h"
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstring>
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

TEST(Validator, RejectsNonNullProbeObj) {
  static constexpr uint32_t kRaw = 0xDEADBEEFu;
  TestInstruction anchor("v_add_f32_e32", 4, 0, std::nullopt, &kRaw);
  auto text = dummy_text();
  InstrumentationPoint pt;
  // The struct stores a const AmdGpuCodeObject*. Any non-null value triggers
  // the guardrail; we never dereference it.
  static const AmdGpuCodeObject *kSentinel = reinterpret_cast<const AmdGpuCodeObject *>(0x1);
  pt.probe_obj = kSentinel;

  std::string err;
  EXPECT_FALSE(validate_anchor(anchor, 0, text, pt, ROCJITSU_CODE_ARCH_CDNA4, &err).has_value());
  EXPECT_FALSE(err.empty());
}

TEST(Validator, RejectsNonEmptyProbeSymbol) {
  static constexpr uint32_t kRaw = 0xDEADBEEFu;
  TestInstruction anchor("v_add_f32_e32", 4, 0, std::nullopt, &kRaw);
  auto text = dummy_text();
  InstrumentationPoint pt;
  pt.probe_symbol = "my_probe";

  std::string err;
  EXPECT_FALSE(validate_anchor(anchor, 0, text, pt, ROCJITSU_CODE_ARCH_CDNA4, &err).has_value());
  EXPECT_FALSE(err.empty());
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
  for (const char *m : {"s_getpc_b64", "s_call_b64", "s_setpc_b64", "s_swappc_b64", "s_rfe_b64"}) {
    TestInstruction anchor(m, 4, /*flags=*/0, std::nullopt, &kRaw);
    std::string err;
    EXPECT_FALSE(validate_anchor(anchor, 0, text, pt, ROCJITSU_CODE_ARCH_CDNA4, &err).has_value())
        << "mnemonic " << m << " should be rejected";
    EXPECT_FALSE(err.empty());
  }
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
    p.original_words = {0xDEADBEEFu};
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
// Section 3: Instrumentor integration tests (minimal in-memory gfx950 ELF)
//==============================================================================

// NOTE: helpers below are intentionally duplicated from
//   tests/dbt/translate_test.cpp::add_elf_name
//   tests/dbt/translate_test.cpp::align_up_for_test
//   tests/dbt/translate_test.cpp::make_minimal_amdgpu_elf_with_text_and_rodata
// to keep this slice self-contained. If/when a third test file needs the same
// helpers, extract them into a shared test-fixture header (the project lead
// may also have a different preference for where this should live).

uint32_t add_elf_name(std::vector<uint8_t> &names, std::string_view name) {
  const uint32_t offset = static_cast<uint32_t>(names.size());
  names.insert(names.end(), name.begin(), name.end());
  names.push_back('\0');
  return offset;
}

uint64_t align_up_for_test(uint64_t value, uint64_t alignment) {
  const uint64_t remainder = value % alignment;
  return remainder == 0 ? value : value + alignment - remainder;
}

// gfx950 ELF with a single .text of `text_size` bytes filled with `s_nop 0`
// words (so every dword-aligned offset is a valid anchor). Used for the
// forward-branch overflow test, which needs .text large enough that the
// trampoline (placed at text_size) is more than INT16_MAX dwords past offset
// 0.
std::vector<uint8_t> make_gfx950_elf_with_nop_text(uint64_t text_size) {
  // Must be a multiple of 4 so the entire section is decodable as s_nop words.
  const uint64_t text_offset = 0x100;
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
  ehdr.e_flags = EF_AMDGPU_MACH_AMDGCN_GFX950;
  ehdr.e_ehsize = sizeof(Elf64_Ehdr);
  ehdr.e_shentsize = sizeof(Elf64_Shdr);
  ehdr.e_shnum = section_count;
  ehdr.e_shstrndx = 3;
  std::memcpy(image.data(), &ehdr, sizeof(ehdr));

  for (uint64_t off = 0; off < text_size; off += sizeof(uint32_t)) {
    const uint32_t nop = 0xBF800000u;
    std::memcpy(image.data() + text_offset + off, &nop, sizeof(nop));
  }

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

// gfx950 ELF with .text containing two `s_nop 0` words (8 bytes total).
// The two-instruction layout lets tests verify that anchor resolution finds
// the second instruction at offset 4, not just the first at offset 0.
std::vector<uint8_t> make_gfx950_elf_with_two_nops() {
  constexpr uint64_t text_offset = 0x100;
  constexpr uint64_t text_size = 8;
  constexpr uint64_t rodata_size = 4;

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
  ehdr.e_flags = EF_AMDGPU_MACH_AMDGCN_GFX950;
  ehdr.e_ehsize = sizeof(Elf64_Ehdr);
  ehdr.e_shentsize = sizeof(Elf64_Shdr);
  ehdr.e_shnum = section_count;
  ehdr.e_shstrndx = 3;
  std::memcpy(image.data(), &ehdr, sizeof(ehdr));

  // Two s_nop 0 words: SOPP encoding prefix 0x17F << 23, opcode 0, simm16 0.
  const std::array<uint32_t, 2> text_words = {0xBF800000u, 0xBF800000u};
  std::memcpy(image.data() + text_offset, text_words.data(), text_size);

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

TEST(Instrumentor, UnsupportedArchReportsErrorInsteadOfCrashing) {
  // Decoder::create returns nullptr for RV32I/RV64I/INVALID. The Instrumentor
  // must surface that as a structured ValidationResult error rather than
  // dereferencing a null decoder during block construction.
  auto image = make_gfx950_elf_with_two_nops();
  AmdGpuCodeObject obj(image.data(), image.size());
  ASSERT_TRUE(obj.is_valid());

  Instrumentor instrumentor(obj, ROCJITSU_CODE_ARCH_RV32I);
  instrumentor.add_point_by_offset(/*anchor_offset=*/4);

  auto result = instrumentor.validate_points();
  EXPECT_TRUE(result.sites.empty());
  ASSERT_FALSE(result.errors.empty());

  // patch() must surface the same error rather than crashing.
  Instrumentor instrumentor2(obj, ROCJITSU_CODE_ARCH_RV32I);
  instrumentor2.add_point_by_offset(/*anchor_offset=*/4);
  auto patched = instrumentor2.patch();
  EXPECT_TRUE(patched.elf_bytes.empty());
  ASSERT_FALSE(patched.errors.empty());
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

TEST(InstrumentorPatch, RejectsZeroQueuedPoints) {
  auto image = make_gfx950_elf_with_two_nops();
  AmdGpuCodeObject obj(image.data(), image.size());
  Instrumentor instrumentor(obj, ROCJITSU_CODE_ARCH_CDNA4);
  // No points queued.

  auto result = instrumentor.patch();
  EXPECT_TRUE(result.elf_bytes.empty());
  EXPECT_FALSE(result.errors.empty());
}

TEST(InstrumentorPatch, RejectsMoreThanOneQueuedPoint) {
  auto image = make_gfx950_elf_with_two_nops();
  AmdGpuCodeObject obj(image.data(), image.size());
  Instrumentor instrumentor(obj, ROCJITSU_CODE_ARCH_CDNA4);
  instrumentor.add_point_by_offset(0);
  instrumentor.add_point_by_offset(4);

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
  void SetUp() override {
    image_ = make_gfx950_elf_with_two_nops();
    AmdGpuCodeObject obj(image_.data(), image_.size());
    ASSERT_TRUE(obj.is_valid());

    Instrumentor instrumentor(obj, ROCJITSU_CODE_ARCH_CDNA4);
    instrumentor.add_point_by_offset(/*anchor_offset=*/4);
    result_ = instrumentor.patch();
    ASSERT_TRUE(result_.errors.empty())
        << (result_.errors.empty() ? std::string{} : result_.errors.front());
    ASSERT_FALSE(result_.elf_bytes.empty());

    patched_ =
        std::make_unique<AmdGpuCodeObject>(result_.elf_bytes.data(), result_.elf_bytes.size());
    ASSERT_TRUE(patched_->is_valid());
  }

  const Section *find_section(std::string_view name) const {
    for (const Section *s : patched_->code_sections())
      if (s->name() == name)
        return s;
    return nullptr;
  }

  std::vector<uint8_t> image_;
  InstrumentedCodeObject result_;
  std::unique_ptr<AmdGpuCodeObject> patched_;
};

TEST_F(InstrumentorPatchElfShape, RjTrampolinesSectionExists) {
  ASSERT_NE(find_section(".rj_trampolines"), nullptr)
      << ".rj_trampolines must appear in code_sections()";
}

TEST_F(InstrumentorPatchElfShape, RjTrampolinesIsExecutable) {
  const Section *tramp = find_section(".rj_trampolines");
  ASSERT_NE(tramp, nullptr);
  EXPECT_NE(tramp->flags() & SHF_EXECINSTR, 0u) << ".rj_trampolines must have SHF_EXECINSTR";
}

TEST_F(InstrumentorPatchElfShape, RjTrampolinesPlacedImmediatelyAfterText) {
  const Section *text = find_section(".text");
  const Section *tramp = find_section(".rj_trampolines");
  ASSERT_NE(text, nullptr);
  ASSERT_NE(tramp, nullptr);
  EXPECT_EQ(tramp->sectionOffset(), text->sectionOffset() + text->size())
      << ".rj_trampolines must start at the byte immediately after .text";
}

TEST_F(InstrumentorPatchElfShape, RjTrampolinesContentsMatchExpectedWords) {
  const Section *tramp = find_section(".rj_trampolines");
  ASSERT_NE(tramp, nullptr);

  // Body for the inline-nop smoke build with a 4-byte anchor:
  //   [placeholder s_nop 0, relocated original s_nop 0, return s_branch(-3)].
  ASSERT_EQ(tramp->size(), 12u);
  std::array<uint32_t, 3> words{};
  std::memcpy(words.data(), tramp->data(), tramp->size());

  constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA4;
  EXPECT_EQ(words[0], build_s_nop(0, kArch))
      << "trampoline body must start with the placeholder s_nop 0";
  EXPECT_EQ(words[1], 0xBF800000u) // original s_nop 0 from .text[4..8)
      << "trampoline body must contain the relocated original instruction unchanged";
  EXPECT_EQ(words[2], build_s_branch(-3, kArch))
      << "trampoline must end with s_branch back to anchor + original_size";
}

TEST_F(InstrumentorPatchElfShape, CodeSectionsIncludesTextAndTrampoline) {
  bool saw_text = false;
  bool saw_tramp = false;
  for (const Section *s : patched_->code_sections()) {
    if (s->name() == ".text")
      saw_text = true;
    if (s->name() == ".rj_trampolines")
      saw_tramp = true;
  }
  EXPECT_TRUE(saw_text) << "code_sections() must include .text";
  EXPECT_TRUE(saw_tramp) << "code_sections() must include .rj_trampolines";
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
  void SetUp() override {
    image_ = make_gfx950_elf_with_two_nops();
    AmdGpuCodeObject obj(image_.data(), image_.size());
    ASSERT_TRUE(obj.is_valid());

    Instrumentor instrumentor(obj, ROCJITSU_CODE_ARCH_CDNA4);
    instrumentor.add_point_by_offset(/*anchor_offset=*/4);
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
    for (const Section *s : patched_->code_sections())
      if (s->name() == name)
        return s;
    return nullptr;
  }

  std::unique_ptr<Instruction> decode_word(const Section *section, size_t byte_offset) {
    rj_code_binary_inst_t word = 0;
    std::memcpy(&word, section->data() + byte_offset, sizeof(word));
    return std::unique_ptr<Instruction>(decoder_->decode(&word));
  }

  std::vector<uint8_t> image_;
  InstrumentedCodeObject result_;
  std::unique_ptr<AmdGpuCodeObject> patched_;
  std::unique_ptr<Decoder> decoder_;
};

TEST_F(InstrumentorPatchDecoded, PatchedAnchorDecodesAsForwardSBranch) {
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

TEST_F(InstrumentorPatchDecoded, UnpatchedTextInstructionIsUnchanged) {
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

TEST_F(InstrumentorPatchDecoded, TrampolinePlaceholderDecodesAsSNop) {
  const Section *tramp = find_section(".rj_trampolines");
  ASSERT_NE(tramp, nullptr);
  auto inst = decode_word(tramp, /*byte_offset=*/0);
  ASSERT_NE(inst, nullptr);

  EXPECT_NE(inst->mnemonic().find("s_nop"), std::string_view::npos)
      << "mnemonic was: " << inst->mnemonic();
  EXPECT_EQ(inst->flags() & BRANCH, 0u) << "placeholder must not be marked as a branch";
  EXPECT_FALSE(inst->branch_offset_bytes().has_value())
      << "placeholder must not carry a branch offset";
}

TEST_F(InstrumentorPatchDecoded, RelocatedOriginalPreservesMnemonic) {
  // The original instruction at .text[4..8] was s_nop 0. After relocation it
  // lives in the trampoline body at offset 4. Decoded bytes must agree.
  const Section *tramp = find_section(".rj_trampolines");
  ASSERT_NE(tramp, nullptr);
  auto relocated = decode_word(tramp, /*byte_offset=*/4);
  ASSERT_NE(relocated, nullptr);
  EXPECT_NE(relocated->mnemonic().find("s_nop"), std::string_view::npos)
      << "relocated original must decode to the same mnemonic as the source; got "
      << relocated->mnemonic();
}

TEST_F(InstrumentorPatchDecoded, TrampolineReturnDecodesAsBackwardSBranch) {
  const Section *tramp = find_section(".rj_trampolines");
  ASSERT_NE(tramp, nullptr);
  auto inst = decode_word(tramp, /*byte_offset=*/8);
  ASSERT_NE(inst, nullptr);

  EXPECT_NE(inst->mnemonic().find("s_branch"), std::string_view::npos)
      << "mnemonic was: " << inst->mnemonic();
  EXPECT_NE(inst->flags() & BRANCH, 0u) << "return must carry BRANCH flag";
  ASSERT_TRUE(inst->branch_offset_bytes().has_value())
      << "return must report a PC-relative branch offset";
  EXPECT_LT(*inst->branch_offset_bytes(), 0)
      << "return branch offset must be negative; got " << *inst->branch_offset_bytes();
}

// End-to-end behavioral check: decode the patched s_branch and the return
// s_branch, compute their actual landing addresses from the raw SOPP simm16,
// and assert they hit the expected .text-relative offsets. This pins the one
// promise PC01-A is fundamentally about — the round trip through the
// trampoline lands the program right back at anchor + original_size.
//
// SOPP semantics: target = branch_pc + 4 + simm16 * 4.
TEST_F(InstrumentorPatchDecoded, BranchesLandAtTheirIntendedTargets) {
  const Section *text = find_section(".text");
  const Section *tramp = find_section(".rj_trampolines");
  ASSERT_NE(text, nullptr);
  ASSERT_NE(tramp, nullptr);

  auto sopp_target = [](uint64_t branch_pc, const uint32_t *word) -> uint64_t {
    const int16_t simm16 = static_cast<int16_t>(*word & 0xFFFFu);
    return branch_pc + 4 + static_cast<int64_t>(simm16) * 4;
  };

  // Forward branch lives at .text-relative offset 4 (the patched anchor).
  // It should land at trampoline_offset == text->size() (== 8 in this fixture).
  auto fwd = decode_word(text, /*byte_offset=*/4);
  ASSERT_NE(fwd, nullptr);
  ASSERT_NE(fwd->raw_encoding(), nullptr);
  const uint64_t fwd_target = sopp_target(/*branch_pc=*/4, fwd->raw_encoding());
  EXPECT_EQ(fwd_target, text->size())
      << "forward branch must land at trampoline_offset (= text->size())";

  // Return branch lives at trampoline-section-relative offset 8. In
  // .text-relative coordinates that is text->size() + 8 (= 16 in this fixture).
  // It should land at anchor_offset + original_size = 4 + 4 = 8.
  auto ret = decode_word(tramp, /*byte_offset=*/8);
  ASSERT_NE(ret, nullptr);
  ASSERT_NE(ret->raw_encoding(), nullptr);
  const uint64_t ret_pc = text->size() + 8;
  const uint64_t ret_target = sopp_target(ret_pc, ret->raw_encoding());
  EXPECT_EQ(ret_target, 8u) << "return branch must land at anchor + original_size";
}

} // namespace
} // namespace rocjitsu
