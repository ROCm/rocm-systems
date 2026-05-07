// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file translate_test.cpp
/// @brief CPU-only unit tests for the DBT translation pipeline.
///
/// Tests encoding correctness, legalization table integrity, and structural
/// properties of translated code objects — without requiring a GPU. Covers:
///   - Coherency bit remapping (GFX940→GFX12, GFX9→GFX12)
///   - Encoding field preservation across SOP1/SOP2/SOPP/SMEM/VOP3 formats
///   - Decode-encode round-trip for CDNA4→RDNA4
///   - Legalization table lookup and zero-ILLEGAL invariant across all ISA pairs
///   - Waitcnt decode/encode (GFX9 monolithic → GFX12 split counters)
///   - E2E binary translation: vector_add code object → translated ELF
///     with valid RDNA4 instructions, correct ELF flags, no GFX9 waitcnt
///
/// These tests complement the hardware tests in hsa_translate_test.cpp which
/// verify correctness on real RDNA4 GPUs.

#include "rocjitsu/analysis/liveness.h"
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/dbt/encoding_translator.h"
#include "rocjitsu/code/dbt/generated/encoding_cdna4_to_rdna4.h"
#include "rocjitsu/code/dbt/generated/encoding_fields.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna1_to_cdna2.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna1_to_cdna3.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna1_to_cdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna1_to_rdna1.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna1_to_rdna2.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna1_to_rdna3.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna1_to_rdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna2_to_cdna3.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna2_to_cdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna2_to_rdna3.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna2_to_rdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna3_to_cdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna3_to_rdna3.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna3_to_rdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna4_to_cdna3.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna4_to_rdna3.h"
#include "rocjitsu/code/dbt/generated/legalization_cdna4_to_rdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_rdna1_to_cdna3.h"
#include "rocjitsu/code/dbt/generated/legalization_rdna1_to_cdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_rdna1_to_rdna2.h"
#include "rocjitsu/code/dbt/generated/legalization_rdna1_to_rdna3.h"
#include "rocjitsu/code/dbt/generated/legalization_rdna1_to_rdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_rdna2_to_rdna3.h"
#include "rocjitsu/code/dbt/generated/legalization_rdna2_to_rdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_rdna3_5_to_rdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_rdna3_to_cdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_rdna3_to_rdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_rdna4_to_cdna4.h"
#include "rocjitsu/code/dbt/generated/legalization_types.h"
#include "rocjitsu/code/dbt/semantic_translator.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/patch/instruction_builder.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/machine_insts.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/AMDHSAKernelDescriptor.h"
RJ_DIAGNOSTIC_POP

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rocjitsu {
namespace {

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

std::vector<uint8_t> make_minimal_amdgpu_elf_with_text(const std::vector<uint32_t> &text_words) {
  constexpr uint64_t text_offset = 0x100;
  const uint64_t text_size = text_words.size() * sizeof(uint32_t);

  std::vector<uint8_t> shstrtab{'\0'};
  const uint32_t text_name = add_elf_name(shstrtab, ".text");
  const uint32_t shstrtab_name = add_elf_name(shstrtab, ".shstrtab");

  const uint64_t shstrtab_offset = text_offset + text_size;
  const uint64_t shoff = align_up_for_test(shstrtab_offset + shstrtab.size(), 8);
  constexpr uint16_t section_count = 3;

  std::vector<uint8_t> image(shoff + section_count * sizeof(Elf64_Shdr), 0);

  Elf64_Ehdr ehdr{};
  std::memcpy(ehdr.e_ident, EI_MAGIC, EI_MAGIC_SIZE);
  ehdr.e_ident[EI_CLASS] = ELFCLASS64;
  ehdr.e_ident[EI_OSABI] = ELFOSABI_AMDGPU_HSA;
  ehdr.e_type = 1; // ET_REL
  ehdr.e_machine = EM_AMDGPU;
  ehdr.e_version = 1;
  ehdr.e_shoff = shoff;
  ehdr.e_flags = EF_AMDGPU_MACH_AMDGCN_GFX950;
  ehdr.e_ehsize = sizeof(Elf64_Ehdr);
  ehdr.e_shentsize = sizeof(Elf64_Shdr);
  ehdr.e_shnum = section_count;
  ehdr.e_shstrndx = 2;
  std::memcpy(image.data(), &ehdr, sizeof(ehdr));

  if (!text_words.empty())
    std::memcpy(image.data() + text_offset, text_words.data(), text_size);
  std::memcpy(image.data() + shstrtab_offset, shstrtab.data(), shstrtab.size());

  std::array<Elf64_Shdr, section_count> shdrs{};
  shdrs[1].sh_name = text_name;
  shdrs[1].sh_type = SHT_PROGBITS;
  shdrs[1].sh_flags = 0x6; // SHF_ALLOC | SHF_EXECINSTR
  shdrs[1].sh_offset = text_offset;
  shdrs[1].sh_size = text_size;
  shdrs[1].sh_addralign = sizeof(uint32_t);

  shdrs[2].sh_name = shstrtab_name;
  shdrs[2].sh_type = SHT_STRTAB;
  shdrs[2].sh_offset = shstrtab_offset;
  shdrs[2].sh_size = shstrtab.size();
  shdrs[2].sh_addralign = 1;

  std::memcpy(image.data() + shoff, shdrs.data(), shdrs.size() * sizeof(Elf64_Shdr));
  return image;
}

std::vector<uint8_t> make_minimal_amdgpu_elf_with_text_and_rodata() {
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
  ehdr.e_type = 1; // ET_REL
  ehdr.e_machine = EM_AMDGPU;
  ehdr.e_version = 1;
  ehdr.e_shoff = shoff;
  ehdr.e_flags = EF_AMDGPU_MACH_AMDGCN_GFX950;
  ehdr.e_ehsize = sizeof(Elf64_Ehdr);
  ehdr.e_shentsize = sizeof(Elf64_Shdr);
  ehdr.e_shnum = section_count;
  ehdr.e_shstrndx = 3;
  std::memcpy(image.data(), &ehdr, sizeof(ehdr));

  const std::array<uint32_t, 2> text_words = {0xBF800000u, 0xBF800000u};
  std::memcpy(image.data() + text_offset, text_words.data(), text_size);

  const uint32_t rodata_word = 0xA5A55A5Au;
  std::memcpy(image.data() + rodata_offset, &rodata_word, sizeof(rodata_word));
  std::memcpy(image.data() + shstrtab_offset, shstrtab.data(), shstrtab.size());

  std::array<Elf64_Shdr, section_count> shdrs{};
  shdrs[1].sh_name = text_name;
  shdrs[1].sh_type = SHT_PROGBITS;
  shdrs[1].sh_flags = 0x6; // SHF_ALLOC | SHF_EXECINSTR
  shdrs[1].sh_offset = text_offset;
  shdrs[1].sh_size = text_size;
  shdrs[1].sh_addralign = sizeof(uint32_t);

  shdrs[2].sh_name = rodata_name;
  shdrs[2].sh_type = SHT_PROGBITS;
  shdrs[2].sh_flags = 0x2; // SHF_ALLOC
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

std::vector<uint8_t> make_minimal_amdgpu_elf_with_load_segments() {
  constexpr uint64_t text_offset = 0x100;
  constexpr uint64_t text_vaddr = 0x1100;
  constexpr uint64_t text_size = 8;
  constexpr uint64_t rodata_size = 4;
  constexpr uint64_t load_align = 0x1000;

  std::vector<uint8_t> shstrtab{'\0'};
  const uint32_t text_name = add_elf_name(shstrtab, ".text");
  const uint32_t rodata_name = add_elf_name(shstrtab, ".rodata");
  const uint32_t shstrtab_name = add_elf_name(shstrtab, ".shstrtab");

  const uint64_t rodata_offset = text_offset + text_size;
  const uint64_t rodata_vaddr = text_vaddr + text_size + load_align;
  const uint64_t shstrtab_offset = rodata_offset + rodata_size;
  const uint64_t shoff = align_up_for_test(shstrtab_offset + shstrtab.size(), 8);
  constexpr uint16_t section_count = 4;
  constexpr uint16_t phdr_count = 2;

  std::vector<uint8_t> image(shoff + section_count * sizeof(Elf64_Shdr), 0);

  Elf64_Ehdr ehdr{};
  std::memcpy(ehdr.e_ident, EI_MAGIC, EI_MAGIC_SIZE);
  ehdr.e_ident[EI_CLASS] = ELFCLASS64;
  ehdr.e_ident[EI_OSABI] = ELFOSABI_AMDGPU_HSA;
  ehdr.e_type = 3; // ET_DYN
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
  ehdr.e_shstrndx = 3;
  std::memcpy(image.data(), &ehdr, sizeof(ehdr));

  constexpr uint32_t kPtLoad = 1;
  std::array<Elf64_Phdr, phdr_count> phdrs{};
  phdrs[0].p_type = kPtLoad;
  phdrs[0].p_flags = 0x5; // PF_R | PF_X
  phdrs[0].p_offset = text_offset;
  phdrs[0].p_vaddr = text_vaddr;
  phdrs[0].p_paddr = text_vaddr;
  phdrs[0].p_filesz = text_size;
  phdrs[0].p_memsz = text_size;
  phdrs[0].p_align = load_align;

  phdrs[1].p_type = kPtLoad;
  phdrs[1].p_flags = 0x4; // PF_R
  phdrs[1].p_offset = rodata_offset;
  phdrs[1].p_vaddr = rodata_vaddr;
  phdrs[1].p_paddr = rodata_vaddr;
  phdrs[1].p_filesz = rodata_size;
  phdrs[1].p_memsz = rodata_size;
  phdrs[1].p_align = load_align;
  std::memcpy(image.data() + ehdr.e_phoff, phdrs.data(), phdrs.size() * sizeof(Elf64_Phdr));

  const std::array<uint32_t, 2> text_words = {0xBF800000u, 0xBF800000u};
  std::memcpy(image.data() + text_offset, text_words.data(), text_size);

  const uint32_t rodata_word = 0xA5A55A5Au;
  std::memcpy(image.data() + rodata_offset, &rodata_word, sizeof(rodata_word));
  std::memcpy(image.data() + shstrtab_offset, shstrtab.data(), shstrtab.size());

  std::array<Elf64_Shdr, section_count> shdrs{};
  shdrs[1].sh_name = text_name;
  shdrs[1].sh_type = SHT_PROGBITS;
  shdrs[1].sh_flags = 0x6; // SHF_ALLOC | SHF_EXECINSTR
  shdrs[1].sh_addr = text_vaddr;
  shdrs[1].sh_offset = text_offset;
  shdrs[1].sh_size = text_size;
  shdrs[1].sh_addralign = sizeof(uint32_t);

  shdrs[2].sh_name = rodata_name;
  shdrs[2].sh_type = SHT_PROGBITS;
  shdrs[2].sh_flags = 0x2; // SHF_ALLOC
  shdrs[2].sh_addr = rodata_vaddr;
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

const Section *find_section(const CodeObject &co, std::string_view name) {
  for (const auto &section : co.all_sections()) {
    if (section->name() == name)
      return section.get();
  }
  return nullptr;
}

constexpr uint16_t kTestA = 256 + 1;
constexpr uint16_t kTestB = 256 + 2;
constexpr uint16_t kTestC = 256 + 3;
constexpr uint16_t kInlineNeg1ForTest = 193;

enum class Bitop3Formula {
  And,
  Or,
  Xor,
  Xnor,
  Not,
  AndOr,
  Or3,
  Xor3,
};

struct Bitop3EquivalenceCase {
  const char *name;
  uint8_t truth_table;
  uint16_t src1;
  uint16_t src2;
  Bitop3Formula formula;
};

const std::array<Bitop3EquivalenceCase, 8> kBitop3EquivalenceCases = {{
    {"V_AND", 0x40, kTestB, scalar_positive_inline_u32(0), Bitop3Formula::And},
    {"V_OR", 0x54, kTestB, scalar_positive_inline_u32(0), Bitop3Formula::Or},
    {"V_XOR", 0x14, kTestB, scalar_positive_inline_u32(0), Bitop3Formula::Xor},
    {"V_XNOR", 0x41, kTestB, scalar_positive_inline_u32(0), Bitop3Formula::Xnor},
    {"V_NOT", 0x01, scalar_positive_inline_u32(0), scalar_positive_inline_u32(0),
     Bitop3Formula::Not},
    {"V_AND_OR", 0xEA, kTestB, kTestC, Bitop3Formula::AndOr},
    {"V_OR3", 0xFE, kTestB, kTestC, Bitop3Formula::Or3},
    {"V_XOR3", 0x96, kTestB, kTestC, Bitop3Formula::Xor3},
}};

std::array<uint32_t, 2> make_cdna4_bitop3_words(uint16_t opcode,
                                                const Bitop3EquivalenceCase &test_case) {
  cdna4::Vop3MachineInst inst{};
  inst.encoding = 0x34;
  inst.op = opcode;
  // Deliberately write back into A. This exercises the scratch-accumulator path:
  // the lowering must not clobber an input register before every monomial has
  // consumed the original source value.
  inst.vdst = 1;
  inst.src0 = kTestA;
  inst.src1 = test_case.src1;
  inst.src2 = test_case.src2;
  // CDNA4 V_BITOP3 stores TTBL as { OMOD[1:0], ABS[2:0], NEG[2:0] }.
  inst.omod = (test_case.truth_table >> 6) & 0x3;
  inst.abs = (test_case.truth_table >> 3) & 0x7;
  inst.neg = test_case.truth_table & 0x7;

  std::array<uint32_t, 2> words{};
  std::memcpy(words.data(), &inst, sizeof(inst));
  return words;
}

uint32_t expected_bitop3_equivalence(Bitop3Formula formula, uint32_t a, uint32_t b, uint32_t c,
                                     uint32_t mask) {
  a &= mask;
  b &= mask;
  c &= mask;

  switch (formula) {
  case Bitop3Formula::And:
    return (a & b) & mask;
  case Bitop3Formula::Or:
    return (a | b) & mask;
  case Bitop3Formula::Xor:
    return (a ^ b) & mask;
  case Bitop3Formula::Xnor:
    return ~(a ^ b) & mask;
  case Bitop3Formula::Not:
    return ~a & mask;
  case Bitop3Formula::AndOr:
    return ((a & b) | c) & mask;
  case Bitop3Formula::Or3:
    return (a | b | c) & mask;
  case Bitop3Formula::Xor3:
    return (a ^ b ^ c) & mask;
  }
  return 0;
}

std::optional<uint32_t> read_cdna3_vop3_operand(uint16_t operand,
                                                const std::array<uint32_t, 256> &vgprs) {
  if (operand >= 256 && operand < 512)
    return vgprs[operand - 256];
  if (operand >= scalar_positive_inline_u32(0) && operand <= scalar_positive_inline_u32(64))
    return operand - scalar_positive_inline_u32(0);
  if (operand == kInlineNeg1ForTest)
    return 0xFFFFFFFFu;
  return std::nullopt;
}

std::optional<uint32_t> evaluate_cdna3_bitop3_lowering(const std::vector<uint32_t> &words,
                                                       uint32_t a, uint32_t b, uint32_t c) {
  std::array<uint32_t, 256> vgprs{};
  vgprs[1] = a;
  vgprs[2] = b;
  vgprs[3] = c;

  if (words.size() % 2 != 0)
    return std::nullopt;

  for (size_t i = 0; i < words.size(); i += 2) {
    cdna3::Vop3MachineInst inst{};
    const uint32_t raw[] = {words[i], words[i + 1]};
    std::memcpy(&inst, raw, sizeof(inst));

    auto src0 = read_cdna3_vop3_operand(static_cast<uint16_t>(inst.src0), vgprs);
    auto src1 = read_cdna3_vop3_operand(static_cast<uint16_t>(inst.src1), vgprs);
    if (!src0)
      return std::nullopt;

    switch (inst.op) {
    case 321: // v_mov_b32
      vgprs[inst.vdst] = *src0;
      break;
    case 272: // v_lshrrev_b32
      if (!src1)
        return std::nullopt;
      vgprs[inst.vdst] = *src1 >> (*src0 & 31u);
      break;
    case 274: // v_lshlrev_b32
      if (!src1)
        return std::nullopt;
      vgprs[inst.vdst] = *src1 << (*src0 & 31u);
      break;
    case 275: // v_and_b32
      if (!src1)
        return std::nullopt;
      vgprs[inst.vdst] = *src0 & *src1;
      break;
    case 277: // v_xor_b32
      if (!src1)
        return std::nullopt;
      vgprs[inst.vdst] = *src0 ^ *src1;
      break;
    default:
      return std::nullopt;
    }
  }

  return vgprs[1];
}

std::vector<uint32_t>
lower_first_cdna4_instruction_for_test(const std::vector<uint32_t> &text_words) {
  auto image = make_minimal_amdgpu_elf_with_text(text_words);
  AmdGpuCodeObject co(image.data(), image.size());
  EXPECT_TRUE(co.is_valid());

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  EXPECT_NE(decoder, nullptr);
  auto blocks = BasicBlock::build(co, *decoder);
  EXPECT_EQ(blocks.size(), 1u);
  if (blocks.empty())
    return {};

  std::vector<BasicBlock *> scope_blocks;
  scope_blocks.reserve(blocks.size());
  for (auto &block : blocks)
    scope_blocks.push_back(block.get());

  LivenessAnalysis liveness(KernelBlockScope(scope_blocks.data(), scope_blocks.size()));
  const Instruction &inst = *blocks.front()->instructions().begin();

  SemanticTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_CDNA3);
  TranslationContext context;
  return translator.try_lower_expand(inst, 0, liveness, context);
}

std::vector<uint32_t> lower_cdna4_bitop3_for_test(uint16_t opcode,
                                                  const Bitop3EquivalenceCase &test_case) {
  const auto inst_words = make_cdna4_bitop3_words(opcode, test_case);
  return lower_first_cdna4_instruction_for_test({inst_words[0], inst_words[1]});
}

std::vector<uint32_t> lower_cdna4_ds_read_b64_tr_b16_for_test(uint8_t vdst, uint8_t addr,
                                                              uint8_t offset0, uint8_t offset1) {
  cdna4::DsMachineInst src{};
  src.encoding = 0x36;
  src.op = 227;
  src.vdst = vdst;
  src.addr = addr;
  src.offset0 = offset0;
  src.offset1 = offset1;

  uint32_t words[2]{};
  std::memcpy(words, &src, sizeof(src));
  return lower_first_cdna4_instruction_for_test({words[0], words[1]});
}

std::vector<uint32_t> lower_cdna4_global_load_lds_dwordx4_for_test() {
  cdna4::FlatGlblMachineInst src{};
  src.encoding = 0x37;
  src.op = 125;
  src.offset = 0;
  src.seg = 2;
  src.addr = 4;
  src.saddr = 127;
  src.vdst = 0;

  uint32_t words[2]{};
  std::memcpy(words, &src, sizeof(src));
  return lower_first_cdna4_instruction_for_test({words[0], words[1]});
}

std::vector<uint32_t> lower_cdna4_wide_k_mfma_for_test(uint16_t opcode, uint8_t vdst,
                                                       uint16_t src2) {
  cdna4::Vop3pMfmaMachineInst src{};
  src.encoding = 0x1A7;
  src.op = opcode;
  src.vdst = vdst;
  src.acc_cd = 1;
  src.src0 = 256 + 8;
  src.src1 = 256 + 16;
  src.src2 = src2;

  uint32_t words[2]{};
  std::memcpy(words, &src, sizeof(src));
  return lower_first_cdna4_instruction_for_test({words[0], words[1]});
}

void expect_cdna3_words_decode(const std::vector<uint32_t> &words) {
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA3);
  ASSERT_NE(decoder, nullptr);
  for (size_t pc = 0; pc < words.size();) {
    std::unique_ptr<Instruction> inst(decoder->decode(&words[pc]));
    ASSERT_NE(inst, nullptr);
    EXPECT_NE(std::string_view(inst->mnemonic()), "v_bitop3_b16");
    EXPECT_NE(std::string_view(inst->mnemonic()), "v_bitop3_b32");
    pc += inst->size() / sizeof(uint32_t);
  }
}

TEST(DsTransposeLowering, ReadB64TrB16Cdna4ToCdna3) {
  const auto lowered = lower_cdna4_ds_read_b64_tr_b16_for_test(8, 11, 0x80, 0);
  ASSERT_FALSE(lowered.empty());
  ASSERT_GE(lowered.size(), 2u);

  cdna3::DsMachineInst read{};
  std::memcpy(&read, lowered.data(), sizeof(read));
  EXPECT_EQ(read.encoding, 0x36u);
  EXPECT_EQ(read.op, 118u);
  EXPECT_EQ(read.addr, 11u);
  EXPECT_EQ(read.offset0, 0x80u);
  EXPECT_EQ(read.offset1, 0u);

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA3);
  ASSERT_NE(decoder, nullptr);

  int ds_read_b64_count = 0;
  int ds_bpermute_count = 0;
  int v_perm_count = 0;
  int v_pack_count = 0;
  int waitcnt_count = 0;
  for (size_t pc = 0; pc < lowered.size();) {
    std::unique_ptr<Instruction> inst(decoder->decode(&lowered[pc]));
    ASSERT_NE(inst, nullptr) << "decode failed at word " << pc;
    const std::string_view mnemonic = inst->mnemonic();
    EXPECT_NE(mnemonic, "ds_read_b64_tr_b16");
    EXPECT_NE(mnemonic, "invalid");
    if (mnemonic == "ds_read_b64")
      ++ds_read_b64_count;
    else if (mnemonic == "ds_bpermute_b32")
      ++ds_bpermute_count;
    else if (mnemonic == "v_perm_b32")
      ++v_perm_count;
    else if (mnemonic == "v_pack_b32_f16")
      ++v_pack_count;
    else if (mnemonic == "s_waitcnt")
      ++waitcnt_count;
    pc += inst->size() / sizeof(uint32_t);
  }

  EXPECT_EQ(ds_read_b64_count, 1);
  EXPECT_EQ(ds_bpermute_count, 8);
  EXPECT_EQ(v_perm_count, 4);
  EXPECT_EQ(v_pack_count, 2);
  EXPECT_GE(waitcnt_count, 1);
}

TEST(GlobalLoadLdsLowering, Dwordx4Cdna4ToCdna3) {
  const auto lowered = lower_cdna4_global_load_lds_dwordx4_for_test();
  ASSERT_FALSE(lowered.empty());

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA3);
  ASSERT_NE(decoder, nullptr);

  int global_load_count = 0;
  int ds_write_count = 0;
  int global_lds_count = 0;
  int waitcnt_count = 0;
  int mbcnt_lo_count = 0;
  int mbcnt_hi_count = 0;
  int lshl_count = 0;
  int add_count = 0;
  int global_load_saddr = -1;
  std::vector<std::string> mnemonics;
  for (size_t pc = 0; pc < lowered.size();) {
    std::unique_ptr<Instruction> inst(decoder->decode(&lowered[pc]));
    ASSERT_NE(inst, nullptr) << "decode failed at word " << pc;
    const std::string_view mnemonic = inst->mnemonic();
    mnemonics.emplace_back(mnemonic);
    EXPECT_NE(mnemonic, "invalid");
    if (mnemonic == "global_load_dwordx4") {
      ++global_load_count;
      cdna3::FlatMachineInst load{};
      std::memcpy(&load, &lowered[pc], sizeof(load));
      global_load_saddr = load.saddr;
    } else if (mnemonic == "ds_write_b128") {
      ++ds_write_count;
    } else if (mnemonic.starts_with("global_load_lds")) {
      ++global_lds_count;
    } else if (mnemonic == "s_waitcnt") {
      ++waitcnt_count;
    } else if (mnemonic == "v_mbcnt_lo_u32_b32") {
      ++mbcnt_lo_count;
    } else if (mnemonic == "v_mbcnt_hi_u32_b32") {
      ++mbcnt_hi_count;
    } else if (mnemonic == "v_lshlrev_b32") {
      ++lshl_count;
    } else if (mnemonic == "v_add_u32") {
      ++add_count;
    }
    pc += inst->size() / sizeof(uint32_t);
  }

  std::string mnemonic_list;
  for (const auto &mnemonic : mnemonics) {
    if (!mnemonic_list.empty())
      mnemonic_list += ", ";
    mnemonic_list += mnemonic;
  }

  EXPECT_EQ(global_load_count, 1) << mnemonic_list;
  EXPECT_EQ(global_load_saddr, 127) << mnemonic_list;
  EXPECT_EQ(ds_write_count, 1) << mnemonic_list;
  EXPECT_EQ(global_lds_count, 0) << mnemonic_list;
  EXPECT_GE(waitcnt_count, 2) << mnemonic_list;
  EXPECT_EQ(mbcnt_lo_count, 1) << mnemonic_list;
  EXPECT_EQ(mbcnt_hi_count, 1) << mnemonic_list;
  EXPECT_EQ(lshl_count, 1) << mnemonic_list;
  EXPECT_EQ(add_count, 1) << mnemonic_list;
}

TEST(MfmaWideKLowering, F16WideKCdna4ToCdna3) {
  struct Case {
    uint16_t wide_opcode;
    uint8_t narrow_opcode;
    const char *wide_mnemonic;
    const char *narrow_mnemonic;
  };
  const std::array<Case, 2> cases = {{
      {84, 77, "v_mfma_f32_16x16x32_f16", "v_mfma_f32_16x16x16_f16"},
      {85, 76, "v_mfma_f32_32x32x16_f16", "v_mfma_f32_32x32x8_f16"},
  }};

  constexpr uint8_t kVdst = 32;
  for (const auto &test_case : cases) {
    for (const uint16_t src2 :
         {scalar_positive_inline_u32(0), uint16_t{256}, uint16_t{256 + kVdst}}) {
      const auto lowered = lower_cdna4_wide_k_mfma_for_test(test_case.wide_opcode, kVdst, src2);
      ASSERT_FALSE(lowered.empty()) << test_case.wide_mnemonic << " src2=" << src2;

      auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA3);
      ASSERT_NE(decoder, nullptr);

      int ds_bpermute_count = 0;
      int waitcnt_count = 0;
      int narrow_mfma_count = 0;
      std::vector<uint16_t> mfma_src0;
      std::vector<uint16_t> mfma_src1;
      std::vector<uint16_t> mfma_src2;
      for (size_t pc = 0; pc < lowered.size();) {
        std::unique_ptr<Instruction> inst(decoder->decode(&lowered[pc]));
        ASSERT_NE(inst, nullptr) << "decode failed at word " << pc;
        const std::string_view mnemonic = inst->mnemonic();
        EXPECT_NE(mnemonic, "invalid");
        EXPECT_NE(mnemonic, test_case.wide_mnemonic);
        if (mnemonic == "ds_bpermute_b32")
          ++ds_bpermute_count;
        else if (mnemonic == "s_waitcnt")
          ++waitcnt_count;
        else if (mnemonic == test_case.narrow_mnemonic) {
          ++narrow_mfma_count;
          cdna3::Vop3pMfmaMachineInst mfma{};
          std::memcpy(&mfma, &lowered[pc], sizeof(mfma));
          EXPECT_EQ(mfma.op, test_case.narrow_opcode);
          EXPECT_EQ(mfma.vdst, kVdst);
          EXPECT_EQ(mfma.acc_cd, 1u);
          EXPECT_EQ(mfma.acc, 0u);
          mfma_src0.push_back(static_cast<uint16_t>(mfma.src0));
          mfma_src1.push_back(static_cast<uint16_t>(mfma.src1));
          mfma_src2.push_back(static_cast<uint16_t>(mfma.src2));
        }
        pc += inst->size() / sizeof(uint32_t);
      }

      EXPECT_EQ(ds_bpermute_count, 0) << test_case.wide_mnemonic;
      EXPECT_EQ(waitcnt_count, 0) << test_case.wide_mnemonic;
      EXPECT_EQ(narrow_mfma_count, 2) << test_case.wide_mnemonic;
      ASSERT_EQ(mfma_src0.size(), 2u) << test_case.wide_mnemonic;
      ASSERT_EQ(mfma_src1.size(), 2u) << test_case.wide_mnemonic;
      ASSERT_EQ(mfma_src2.size(), 2u) << test_case.wide_mnemonic;
      EXPECT_EQ(mfma_src0[0], 256u + 8) << test_case.wide_mnemonic;
      EXPECT_EQ(mfma_src0[1], 256u + 10) << test_case.wide_mnemonic;
      EXPECT_EQ(mfma_src1[0], 256u + 16) << test_case.wide_mnemonic;
      EXPECT_EQ(mfma_src1[1], 256u + 18) << test_case.wide_mnemonic;
      EXPECT_EQ(mfma_src2[0], src2) << test_case.wide_mnemonic;
      EXPECT_EQ(mfma_src2[1], 256u + kVdst) << test_case.wide_mnemonic;
    }
  }
}

TEST(CoherencyRemap, Gfx940ToGfx12AgentScope) {
  auto coh = remap_gfx940_to_gfx12({1, 0, 0});
  EXPECT_EQ(coh.scope, 1);
  EXPECT_EQ(coh.th, 0);
}

TEST(CoherencyRemap, Gfx940ToGfx12SystemScope) {
  auto coh = remap_gfx940_to_gfx12({1, 1, 0});
  EXPECT_EQ(coh.scope, 3);
  EXPECT_EQ(coh.th, 0);
}

TEST(CoherencyRemap, Gfx940ToGfx12NonTemporal) {
  auto coh = remap_gfx940_to_gfx12({0, 0, 1});
  EXPECT_EQ(coh.scope, 0);
  EXPECT_EQ(coh.th, 3);
}

TEST(CoherencyRemap, Gfx9GlcToGfx12) {
  auto coh_glc1 = remap_gfx9_to_gfx12({1});
  EXPECT_EQ(coh_glc1.scope, 2);
  EXPECT_EQ(coh_glc1.th, 0);

  auto coh_glc0 = remap_gfx9_to_gfx12({0});
  EXPECT_EQ(coh_glc0.scope, 0);
  EXPECT_EQ(coh_glc0.th, 0);
}

TEST(EncodingTranslator, Sop1PreservesRegisters) {
  cdna4::Sop1MachineInst src{};
  src.ssrc0 = 42;
  src.sdst = 17;
  src.op = 3;
  src.encoding = 0x17D;
  uint32_t w0 = std::bit_cast<uint32_t>(src);

  auto result = cdna4_to_rdna4::translate_encoding_cdna4_to_rdna4(kEnc_SOP1, w0, 0, 0, 5);

  ASSERT_EQ(result.word_count, 1);
  auto dst = std::bit_cast<rdna4::Sop1MachineInst>(result.words[0]);
  EXPECT_EQ(dst.ssrc0, 42);
  EXPECT_EQ(dst.sdst, 17);
  EXPECT_EQ(dst.op, 5);
  EXPECT_EQ(dst.encoding, 0x17D);
}

TEST(EncodingTranslator, Sop2PreservesRegisters) {
  cdna4::Sop2MachineInst src{};
  src.ssrc0 = 10;
  src.ssrc1 = 20;
  src.sdst = 30;
  src.op = 7;
  src.encoding = 0x2;
  uint32_t w0 = std::bit_cast<uint32_t>(src);

  auto result = cdna4_to_rdna4::translate_encoding_cdna4_to_rdna4(kEnc_SOP2, w0, 0, 0, 7);

  ASSERT_EQ(result.word_count, 1);
  auto dst = std::bit_cast<rdna4::Sop2MachineInst>(result.words[0]);
  EXPECT_EQ(dst.ssrc0, 10);
  EXPECT_EQ(dst.ssrc1, 20);
  EXPECT_EQ(dst.sdst, 30);
  EXPECT_EQ(dst.op, 7);
}

TEST(EncodingTranslator, SoppPreservesSimm16) {
  cdna4::SoppMachineInst src{};
  src.simm16 = 0xABCD;
  src.op = 12;
  src.encoding = 0x17F;
  uint32_t w0 = std::bit_cast<uint32_t>(src);

  auto result = cdna4_to_rdna4::translate_encoding_cdna4_to_rdna4(kEnc_SOPP, w0, 0, 0, 12);

  ASSERT_EQ(result.word_count, 1);
  auto dst = std::bit_cast<rdna4::SoppMachineInst>(result.words[0]);
  EXPECT_EQ(dst.simm16, 0xABCD);
  EXPECT_EQ(dst.op, 12);
}

TEST(EncodingTranslator, SmemRemapsCoherency) {
  cdna4::SmemMachineInst src{};
  src.sbase = 5;
  src.sdata = 3;
  src.glc = 1;
  src.nv = 0;
  src.op = 0;
  src.offset = 0x100;
  src.soffset = 0x7F;
  src.encoding = 0x3D;
  uint32_t words[2];
  std::memcpy(words, &src, sizeof(src));

  auto result =
      cdna4_to_rdna4::translate_encoding_cdna4_to_rdna4(kEnc_SMEM, words[0], words[1], 0, 0);

  ASSERT_EQ(result.word_count, 2);
  rdna4::SmemMachineInst dst{};
  std::memcpy(&dst, result.words, sizeof(dst));
  EXPECT_EQ(dst.sbase, 5);
  EXPECT_EQ(dst.sdata, 3);
  EXPECT_EQ(dst.scope, 2);
  EXPECT_EQ(dst.th, 0);
  EXPECT_EQ(dst.nv, 0);
  EXPECT_EQ(dst.soffset, 0x7C); // CDNA4 null (0x7F) → RDNA4 null (0x7C)
}

TEST(EncodingTranslator, Vop3PreservesModifiers) {
  cdna4::Vop3MachineInst src{};
  src.vdst = 10;
  src.src0 = 100;
  src.src1 = 200;
  src.src2 = 50;
  src.clamp = 1;
  src.omod = 2;
  src.neg = 5;
  src.abs = 3;
  src.op = 100;
  src.encoding = 0x35;
  uint32_t words[2];
  std::memcpy(words, &src, sizeof(src));

  auto result =
      cdna4_to_rdna4::translate_encoding_cdna4_to_rdna4(kEnc_VOP3, words[0], words[1], 0, 100);

  ASSERT_EQ(result.word_count, 2);
  rdna4::Vop3MachineInst dst{};
  std::memcpy(&dst, result.words, sizeof(dst));
  EXPECT_EQ(dst.vdst, 10);
  EXPECT_EQ(dst.src0, 100);
  EXPECT_EQ(dst.src1, 200);
  EXPECT_EQ(dst.src2, 50);
  EXPECT_EQ(dst.clamp, 1);
  EXPECT_EQ(dst.omod, 2);
  EXPECT_EQ(dst.neg, 5);
  EXPECT_EQ(dst.abs, 3);
}

TEST(EncodingTranslator, UnknownEncodingReturnsEmpty) {
  auto result = cdna4_to_rdna4::translate_encoding_cdna4_to_rdna4(0xFFFF, 0, 0, 0, 0);
  EXPECT_EQ(result.word_count, 0);
}

TEST(EncodingTranslator, DecodeEncodeRoundTrip) {
  cdna4::Sop1MachineInst src{};
  src.ssrc0 = 55;
  src.sdst = 33;
  src.op = 4;
  src.encoding = 0x17D;
  uint32_t w0 = std::bit_cast<uint32_t>(src);

  auto fields = cdna4_to_rdna4::decode_sop1_cdna4(w0);
  EXPECT_EQ(fields.ssrc0, 55u);
  EXPECT_EQ(fields.sdst, 33u);
  EXPECT_EQ(fields.op, 4u);

  auto result = cdna4_to_rdna4::encode_sop1_rdna4(fields, 4);
  ASSERT_EQ(result.word_count, 1);
  auto dst = std::bit_cast<rdna4::Sop1MachineInst>(result.words[0]);
  EXPECT_EQ(dst.ssrc0, 55);
  EXPECT_EQ(dst.sdst, 33);
  EXPECT_EQ(dst.op, 4);
}

TEST(LegalizationLookup, FindsKnownInstruction) {
  const auto *entry = lookup(kLegalization_cdna4_to_rdna4, 0, 0);
  EXPECT_NE(entry, nullptr);
  if (entry) {
    EXPECT_NE(entry->action, Action::Illegal);
  }
}

TEST(LegalizationLookup, ReturnsNullForUnknown) {
  const auto *entry = lookup(kLegalization_cdna4_to_rdna4, 0xFFFF, 0xFFFF);
  EXPECT_EQ(entry, nullptr);
}

TEST(LegalizationLookup, Cdna4ToCdna3UsesExactRuntimeEncodingId) {
  const auto *vop3_cmp = lookup(kLegalization_cdna4_to_cdna3, 416, 84);
  ASSERT_NE(vop3_cmp, nullptr);
  EXPECT_EQ(vop3_cmp->action, Action::Identity);

  const auto *mfma = lookup(kLegalization_cdna4_to_cdna3, 423, 84);
  ASSERT_NE(mfma, nullptr);
  EXPECT_EQ(mfma->action, Action::Expand);
}

TEST(LegalizationTable, NoIllegalEntries_Cdna4ToRdna4) {
  for (const auto &e : kLegalization_cdna4_to_rdna4) {
    EXPECT_NE(e.action, Action::Illegal)
        << "ILLEGAL at encoding_id=" << e.src_encoding_id << " opcode=" << e.src_opcode;
  }
}

#define CHECK_NO_ILLEGAL(pair)                                                                     \
  TEST(LegalizationTable, NoIllegalEntries_##pair) {                                               \
    for (const auto &e : kLegalization_##pair) {                                                   \
      EXPECT_NE(e.action, Action::Illegal)                                                         \
          << "ILLEGAL at encoding_id=" << e.src_encoding_id << " opcode=" << e.src_opcode;         \
    }                                                                                              \
    EXPECT_GT(std::size(kLegalization_##pair), 0u) << "table is empty";                            \
  }

CHECK_NO_ILLEGAL(cdna1_to_cdna2)
CHECK_NO_ILLEGAL(cdna1_to_cdna3)
CHECK_NO_ILLEGAL(cdna1_to_cdna4)
CHECK_NO_ILLEGAL(cdna1_to_rdna1)
CHECK_NO_ILLEGAL(cdna1_to_rdna2)
CHECK_NO_ILLEGAL(cdna1_to_rdna3)
CHECK_NO_ILLEGAL(cdna1_to_rdna4)
CHECK_NO_ILLEGAL(cdna2_to_cdna3)
CHECK_NO_ILLEGAL(cdna2_to_cdna4)
CHECK_NO_ILLEGAL(cdna2_to_rdna3)
CHECK_NO_ILLEGAL(cdna2_to_rdna4)
CHECK_NO_ILLEGAL(cdna3_to_cdna4)
CHECK_NO_ILLEGAL(cdna3_to_rdna3)
CHECK_NO_ILLEGAL(cdna3_to_rdna4)
CHECK_NO_ILLEGAL(cdna4_to_cdna3)
CHECK_NO_ILLEGAL(cdna4_to_rdna3)
CHECK_NO_ILLEGAL(rdna1_to_cdna3)
CHECK_NO_ILLEGAL(rdna1_to_cdna4)
CHECK_NO_ILLEGAL(rdna1_to_rdna2)
CHECK_NO_ILLEGAL(rdna1_to_rdna3)
CHECK_NO_ILLEGAL(rdna1_to_rdna4)
CHECK_NO_ILLEGAL(rdna2_to_rdna3)
CHECK_NO_ILLEGAL(rdna2_to_rdna4)
CHECK_NO_ILLEGAL(rdna3_5_to_rdna4)
CHECK_NO_ILLEGAL(rdna3_to_cdna4)
CHECK_NO_ILLEGAL(rdna3_to_rdna4)
CHECK_NO_ILLEGAL(rdna4_to_cdna4)

#undef CHECK_NO_ILLEGAL

TEST(Bitop3Lowering, EquivalenceCasesCdna4ToCdna3) {
  constexpr uint16_t kCdna4OpBitop3B16 = 563;
  constexpr uint16_t kCdna4OpBitop3B32 = 564;
  const std::array<std::array<uint32_t, 3>, 4> samples = {{
      {0xA5A55A5Au, 0x0F0FF0F0u, 0x3333CCCCu},
      {0x00000000u, 0xFFFFFFFFu, 0x13579BDFu},
      {0xFFFFFFFFu, 0x2468ACE0u, 0x00000000u},
      {0x8001007Fu, 0x7FFE0101u, 0x55AA33CCu},
  }};

  for (const auto &test_case : kBitop3EquivalenceCases) {
    for (const auto [opcode, mask] :
         {std::pair<uint16_t, uint32_t>{kCdna4OpBitop3B32, 0xFFFFFFFFu},
          std::pair<uint16_t, uint32_t>{kCdna4OpBitop3B16, 0x0000FFFFu}}) {
      const auto lowered = lower_cdna4_bitop3_for_test(opcode, test_case);
      ASSERT_FALSE(lowered.empty()) << test_case.name << " opcode " << opcode;
      expect_cdna3_words_decode(lowered);

      for (const auto &sample : samples) {
        const uint32_t a = sample[0];
        const uint32_t b = sample[1];
        const uint32_t c = sample[2];
        const auto actual = evaluate_cdna3_bitop3_lowering(lowered, a, b, c);
        ASSERT_TRUE(actual.has_value()) << test_case.name << " opcode " << opcode;
        const uint32_t expected = expected_bitop3_equivalence(test_case.formula, a, b, c, mask);
        EXPECT_EQ(*actual, expected) << test_case.name << " opcode " << opcode << " a=0x"
                                     << std::hex << a << " b=0x" << b << " c=0x" << c << std::dec;
      }
    }
  }
}

TEST(CodeObjectPatcher, CaveBodyMaterializesInRjTranslationsAfterText) {
  auto image = make_minimal_amdgpu_elf_with_text_and_rodata();
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());
  ASSERT_FALSE(co.text_sections().empty());

  CodeObjectPatcher patcher(co);
  patcher.set_cave_start(co.text_sections()[0]->size());
  const std::array<uint32_t, 2> cave_words = {0xDEADBEEFu, 0xCAFEBABEu};
  patcher.append_cave_body(cave_words);
  patcher.append_cave_section();

  auto patched_bytes = patcher.emit();
  AmdGpuCodeObject patched(patched_bytes.data(), patched_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_FALSE(patched.text_sections().empty());

  const Section *text = patched.text_sections()[0];
  ASSERT_NE(text, nullptr);
  EXPECT_EQ(text->size(), 8u) << ".text must keep its original size";

  const Section *translations = find_section(patched, ".rj_translations");
  ASSERT_NE(translations, nullptr);
  ASSERT_EQ(patched.code_sections().size(), 2u);
  EXPECT_EQ(patched.code_sections()[0]->name(), ".text");
  EXPECT_EQ(patched.code_sections()[1]->name(), ".rj_translations");
  EXPECT_EQ(translations->sectionOffset(), text->sectionOffset() + text->size())
      << ".rj_translations must be physically placed immediately after .text";
  ASSERT_EQ(translations->size(), cave_words.size() * sizeof(uint32_t));
  EXPECT_EQ(std::memcmp(translations->data(), cave_words.data(), translations->size()), 0);

  const Section *rodata = find_section(patched, ".rodata");
  ASSERT_NE(rodata, nullptr);
  EXPECT_EQ(rodata->sectionOffset(), translations->sectionOffset() + translations->size())
      << "sections following .text must be shifted after the cave section";
  uint32_t rodata_word = 0;
  std::memcpy(&rodata_word, rodata->data(), sizeof(rodata_word));
  EXPECT_EQ(rodata_word, 0xA5A55A5Au);
}

TEST(CodeObjectPatcher, CaveInsertionPreservesLoadSegmentAlignment) {
  auto image = make_minimal_amdgpu_elf_with_load_segments();
  AmdGpuCodeObject co(image.data(), image.size());
  ASSERT_TRUE(co.is_valid());
  ASSERT_FALSE(co.text_sections().empty());

  CodeObjectPatcher patcher(co);
  patcher.set_cave_start(co.text_sections()[0]->size());
  const std::array<uint32_t, 2> cave_words = {0xDEADBEEFu, 0xCAFEBABEu};
  patcher.append_cave_body(cave_words);
  patcher.append_cave_section();

  auto patched_bytes = patcher.emit();
  AmdGpuCodeObject patched(patched_bytes.data(), patched_bytes.size());
  ASSERT_TRUE(patched.is_valid());
  ASSERT_FALSE(patched.text_sections().empty());

  const Section *text = patched.text_sections()[0];
  const Section *translations = find_section(patched, ".rj_translations");
  const Section *rodata = find_section(patched, ".rodata");
  ASSERT_NE(translations, nullptr);
  ASSERT_NE(rodata, nullptr);
  EXPECT_EQ(translations->sectionOffset(), text->sectionOffset() + text->size());
  EXPECT_EQ(translations->size(), cave_words.size() * sizeof(uint32_t));

  constexpr uint64_t load_align = 0x1000;
  EXPECT_EQ(rodata->sectionOffset(), text->sectionOffset() + text->size() + load_align)
      << "file padding should preserve later PT_LOAD p_offset/p_vaddr congruence";

  const auto *ehdr = reinterpret_cast<const Elf64_Ehdr *>(patched_bytes.data());
  ASSERT_EQ(ehdr->e_phnum, 2u);
  const auto *phdrs = reinterpret_cast<const Elf64_Phdr *>(patched_bytes.data() + ehdr->e_phoff);
  for (uint16_t i = 0; i < ehdr->e_phnum; ++i) {
    ASSERT_NE(phdrs[i].p_align, 0u);
    EXPECT_EQ(phdrs[i].p_offset % phdrs[i].p_align, phdrs[i].p_vaddr % phdrs[i].p_align)
        << "PT_LOAD " << i << " must remain loader-congruent";
  }
  EXPECT_EQ(phdrs[0].p_filesz, 8u + load_align);
  EXPECT_EQ(phdrs[0].p_memsz, 8u + load_align);
}

} // namespace
} // namespace rocjitsu

// --- WaitcntTranslator tests ---
#include "rocjitsu/code/dbt/semantic_translator.h"

using rocjitsu::decode_waitcnt_gfx9;
using rocjitsu::encode_waitcnt_gfx12;
using rocjitsu::WaitcntValues;

TEST(WaitcntTranslator, DecodeVmcnt0) {
  auto v = decode_waitcnt_gfx9(0x0000);
  EXPECT_EQ(v.vmcnt, 0);
  EXPECT_EQ(v.lgkmcnt, 0);
  EXPECT_EQ(v.expcnt, 0);
}

TEST(WaitcntTranslator, DecodeAllRelaxed) {
  auto v = decode_waitcnt_gfx9(0xCF7F);
  EXPECT_EQ(v.vmcnt, 0x3F);
  EXPECT_EQ(v.lgkmcnt, 0x0F);
  EXPECT_EQ(v.expcnt, 0x07);
}

TEST(WaitcntTranslator, DecodeVmcnt15Lgkm0) {
  uint16_t simm16 = 0x000F;
  auto v = decode_waitcnt_gfx9(simm16);
  EXPECT_EQ(v.vmcnt, 15);
  EXPECT_EQ(v.lgkmcnt, 0);
  EXPECT_EQ(v.expcnt, 0);
}

TEST(WaitcntTranslator, EncodeAllZeroProducesMultipleWords) {
  WaitcntValues v{0, 0, 0};
  auto words = encode_waitcnt_gfx12(v);
  EXPECT_GE(words.size(), 3u);
}

TEST(WaitcntTranslator, EncodeAllRelaxedProducesNop) {
  WaitcntValues v{0x3F, 0x0F, 0x07};
  auto words = encode_waitcnt_gfx12(v);
  ASSERT_EQ(words.size(), 1u);
  uint8_t op = (words[0] >> 16) & 0x7F;
  EXPECT_EQ(op, 0);
}

TEST(WaitcntTranslator, EncodeVmcnt0EmitsLoadcntAndStorecnt) {
  WaitcntValues v{0, 0x0F, 0x07};
  auto words = encode_waitcnt_gfx12(v);
  EXPECT_GE(words.size(), 2u);

  bool has_loadcnt = false;
  bool has_storecnt_dscnt = false;
  for (auto w : words) {
    uint8_t op = (w >> 16) & 0x7F;
    if (op == 64)
      has_loadcnt = true;
    if (op == 73)
      has_storecnt_dscnt = true;
  }
  EXPECT_TRUE(has_loadcnt);
  EXPECT_TRUE(has_storecnt_dscnt);
}

TEST(CApi, CodeObjectCreateFromMemoryExposesImageBytes) {
  auto image = rocjitsu::make_minimal_amdgpu_elf_with_text({0xBF800000u});

  rj_code_object_t *obj = nullptr;
  ASSERT_EQ(rj_code_object_create_from_memory(image.data(), image.size(), &obj),
            ROCJITSU_STATUS_SUCCESS);
  ASSERT_NE(obj, nullptr);

  const uint8_t *data = nullptr;
  uint64_t size = 0;
  EXPECT_EQ(rj_code_object_image_data(obj, &data, &size), ROCJITSU_STATUS_SUCCESS);
  ASSERT_NE(data, nullptr);
  EXPECT_EQ(size, image.size());
  EXPECT_EQ(std::memcmp(data, image.data(), image.size()), 0);

  rj_code_object_destroy(obj);
}

// --- End-to-end BinaryTranslator integration tests ---
#ifdef HAS_DEVICE_KERNELS

#include "rocjitsu/code/dbt/binary_translator.h"
#include "rocjitsu/code/dbt/kernel_descriptor_translator.h"
#include "rocjitsu/code/executable.h"
#include "rocjitsu/code/patch/instruction_builder.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/machine_insts.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"

namespace {

std::string kernel_path(const char *name) { return std::string(KERNEL_DIR) + "/" + name + ".o"; }

} // namespace

using rocjitsu::BinaryTranslator;
using rocjitsu::Decoder;
using rocjitsu::Executable;

TEST(BinaryTranslatorE2E, TranslateVectorAddCdna4ToRdna4) {
  Executable exec(kernel_path("vector_add"));
  ASSERT_TRUE(exec.is_valid()) << "Failed to load vector_add.o";
  ASSERT_GT(exec.num_code_objects(ROCJITSU_CODE_TARGET_GFX950), 0u);

  const auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  ASSERT_NE(co, nullptr);

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA4);
  auto result = translator.translate(*co);

  EXPECT_FALSE(result.elf_bytes.empty()) << "Translation produced empty ELF";
  EXPECT_EQ(result.host_arch, ROCJITSU_CODE_ARCH_RDNA4);

  // Verify ELF machine flags contain GFX1200.
  ASSERT_GE(result.elf_bytes.size(), 48u);
  uint32_t e_flags = 0;
  std::memcpy(&e_flags, result.elf_bytes.data() + 48, sizeof(e_flags));
  constexpr uint32_t kEfAmdgpuMachGfx1200 = 0x48;
  EXPECT_EQ(e_flags & 0xFF, kEfAmdgpuMachGfx1200)
      << "ELF e_flags should contain GFX1200 machine type";
}

TEST(BinaryTranslatorE2E, ExplicitRdna4MachCanTargetGfx1201) {
  Executable exec(kernel_path("vector_add"));
  ASSERT_TRUE(exec.is_valid()) << "Failed to load vector_add.o";
  ASSERT_GT(exec.num_code_objects(ROCJITSU_CODE_TARGET_GFX950), 0u);

  const auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  ASSERT_NE(co, nullptr);

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA4,
                              ROCJITSU_CODE_AMDGPU_MACH_GFX1201);
  auto result = translator.translate(*co);
  ASSERT_FALSE(result.elf_bytes.empty()) << "Translation produced empty ELF";

  ASSERT_GE(result.elf_bytes.size(), 48u);
  uint32_t e_flags = 0;
  std::memcpy(&e_flags, result.elf_bytes.data() + 48, sizeof(e_flags));
  EXPECT_EQ(e_flags & 0xFF, ROCJITSU_CODE_AMDGPU_MACH_GFX1201)
      << "ELF e_flags should contain the exact requested GFX1201 machine type";
}

TEST(KernelDescriptorTranslator, Cdna4ToRdna4MaterializesWorkgroupIdsFromTtmpGridPayload) {
  using namespace rocr::llvm::amdhsa;

  Executable exec(kernel_path("vector_add"));
  ASSERT_TRUE(exec.is_valid());
  ASSERT_GT(exec.num_code_objects(ROCJITSU_CODE_TARGET_GFX950), 0u);

  const auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  ASSERT_NE(co, nullptr);

  std::vector<uint8_t> image(co->image_size());
  std::memcpy(image.data(), co->image_data(), image.size());

  ASSERT_FALSE(co->text_sections().empty());
  const auto *original_text = co->text_sections()[0];
  rocjitsu::KernelDescriptorTranslator translator(ROCJITSU_CODE_ARCH_CDNA4,
                                                  ROCJITSU_CODE_ARCH_RDNA4);
  const auto original_translations =
      translator.translate_image(image, original_text->sectionOffset(), original_text->size(),
                                 rocjitsu::KernelDescriptorTranslationOptions{});
  ASSERT_FALSE(original_translations.empty());
  const uint64_t kd_file_off = original_translations[0].descriptor_file_offset;
  ASSERT_LE(kd_file_off + sizeof(kernel_descriptor_t), image.size());

  auto *kd = reinterpret_cast<kernel_descriptor_t *>(image.data() + kd_file_off);
  kd->compute_pgm_rsrc2 = 0;
  AMDHSA_BITS_SET(kd->compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 12);
  AMDHSA_BITS_SET(kd->compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X, 1);
  AMDHSA_BITS_SET(kd->compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y, 1);
  AMDHSA_BITS_SET(kd->compute_pgm_rsrc2, COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z, 1);

  rocjitsu::AmdGpuCodeObject mutated(image.data(), image.size());
  ASSERT_TRUE(mutated.is_valid());
  ASSERT_FALSE(mutated.text_sections().empty());
  const auto *text = mutated.text_sections()[0];

  const auto translations = translator.translate_image(
      image, text->sectionOffset(), text->size(), rocjitsu::KernelDescriptorTranslationOptions{});
  const auto translated = std::find_if(translations.begin(), translations.end(),
                                       [kd_file_off](const auto &translation) {
                                         return translation.descriptor_file_offset == kd_file_off;
                                       });
  ASSERT_NE(translated, translations.end());

  constexpr uint16_t ttmp_base = 108;
  const uint16_t shift16 = rocjitsu::scalar_positive_inline_u32(16);
  const std::vector<uint32_t> expected = {
      rocjitsu::build_s_mov_b32(12, ttmp_base + 9, ROCJITSU_CODE_ARCH_RDNA4),
      rocjitsu::build_s_delay_alu(rocjitsu::kDelayAluSaluDep1, ROCJITSU_CODE_ARCH_RDNA4),
      rocjitsu::build_s_mov_b32(13, ttmp_base + 7, ROCJITSU_CODE_ARCH_RDNA4),
      rocjitsu::build_s_delay_alu(rocjitsu::kDelayAluSaluDep1, ROCJITSU_CODE_ARCH_RDNA4),
      rocjitsu::build_s_lshl_b32(13, 13, shift16, ROCJITSU_CODE_ARCH_RDNA4),
      rocjitsu::build_s_delay_alu(rocjitsu::kDelayAluSaluDep1, ROCJITSU_CODE_ARCH_RDNA4),
      rocjitsu::build_s_lshr_b32(13, 13, shift16, ROCJITSU_CODE_ARCH_RDNA4),
      rocjitsu::build_s_delay_alu(rocjitsu::kDelayAluSaluDep1, ROCJITSU_CODE_ARCH_RDNA4),
      rocjitsu::build_s_lshr_b32(14, ttmp_base + 7, shift16, ROCJITSU_CODE_ARCH_RDNA4),
      rocjitsu::build_s_delay_alu(rocjitsu::kDelayAluSaluDep1, ROCJITSU_CODE_ARCH_RDNA4),
  };
  EXPECT_EQ(translated->prologue_words, expected)
      << "CDNA workgroup_id SGPRs must be rebuilt from RDNA4 TTMP9 and packed TTMP7";
}

TEST(KernelDescriptorTranslator, RdnaWave64UsesAmdhsaDescriptorVgprEncoding) {
  using namespace rocr::llvm::amdhsa;

  Executable exec(kernel_path("vector_add"));
  ASSERT_TRUE(exec.is_valid());
  ASSERT_GT(exec.num_code_objects(ROCJITSU_CODE_TARGET_GFX950), 0u);

  const auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  ASSERT_NE(co, nullptr);
  ASSERT_FALSE(co->text_sections().empty());

  std::vector<uint8_t> image(co->image_size());
  std::memcpy(image.data(), co->image_data(), image.size());

  const auto *text = co->text_sections()[0];
  rocjitsu::KernelDescriptorTranslator parser(ROCJITSU_CODE_ARCH_CDNA1, ROCJITSU_CODE_ARCH_RDNA4);
  const auto parsed = parser.translate_image(image, text->sectionOffset(), text->size(),
                                             rocjitsu::KernelDescriptorTranslationOptions{});
  ASSERT_FALSE(parsed.empty());
  const uint64_t kd_file_off = parsed[0].descriptor_file_offset;
  ASSERT_LE(kd_file_off + sizeof(kernel_descriptor_t), image.size());

  auto *kd = reinterpret_cast<kernel_descriptor_t *>(image.data() + kd_file_off);
  kd->compute_pgm_rsrc1 = 0;
  kd->kernel_code_properties = 0;
  AMDHSA_BITS_SET(kd->compute_pgm_rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT, 31);

  // Use CDNA1 as the guest so the source descriptor field means exactly
  // 128 VGPRs without AccVGPR remapping. The target assertions below check the
  // AMDHSA descriptor encoding for RDNA Wave64, not the ISA manual's physical
  // allocation block size. AMDHSA GFX10-GFX12 Wave64 encodes 128 VGPRs as
  // ceil(128 / 4) - 1 == 31; using the RDNA3/RDNA4 physical block size of 8
  // would incorrectly produce 15.
  for (rj_code_arch_t host_arch :
       {ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2, ROCJITSU_CODE_ARCH_RDNA3,
        ROCJITSU_CODE_ARCH_RDNA3_5, ROCJITSU_CODE_ARCH_RDNA4}) {
    rocjitsu::KernelDescriptorTranslator translator(ROCJITSU_CODE_ARCH_CDNA1, host_arch);
    const auto translations = translator.translate_image(
        image, text->sectionOffset(), text->size(), rocjitsu::KernelDescriptorTranslationOptions{});
    const auto translated = std::find_if(translations.begin(), translations.end(),
                                         [kd_file_off](const auto &translation) {
                                           return translation.descriptor_file_offset == kd_file_off;
                                         });
    ASSERT_NE(translated, translations.end());
    EXPECT_EQ(translated->target_wave_size, 64);
    EXPECT_EQ(translated->target_vgpr_count, 128u);
    EXPECT_EQ(translated->target_vgpr_granulated, 31u);
  }
}

TEST(BinaryTranslatorE2E, DescriptorPrologueRedirectsEntryWithoutOverwritingOriginalEntry) {
  Executable exec(kernel_path("vector_add"));
  ASSERT_TRUE(exec.is_valid());
  ASSERT_GT(exec.num_code_objects(ROCJITSU_CODE_TARGET_GFX950), 0u);

  const auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  ASSERT_NE(co, nullptr);

  const auto *original_image = reinterpret_cast<const uint8_t *>(co->image_data());
  ASSERT_FALSE(co->text_sections().empty());
  const auto *original_text = co->text_sections()[0];
  rocjitsu::KernelDescriptorTranslator original_parser(ROCJITSU_CODE_ARCH_CDNA4,
                                                       ROCJITSU_CODE_ARCH_RDNA4);
  const auto original_infos = original_parser.translate_image(
      {original_image, co->image_size()}, original_text->sectionOffset(), original_text->size(),
      rocjitsu::KernelDescriptorTranslationOptions{});
  ASSERT_FALSE(original_infos.empty());
  const rocjitsu::KdTranslation *original_info = &original_infos[0];
  const uint64_t kd_file_off = original_info->descriptor_file_offset;

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA4);
  auto result = translator.translate(*co);
  ASSERT_FALSE(result.elf_bytes.empty());

  rocjitsu::AmdGpuCodeObject translated_co(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated_co.is_valid());
  ASSERT_FALSE(translated_co.text_sections().empty());
  const auto *translated_text = translated_co.text_sections()[0];
  const auto *translated_image = reinterpret_cast<const uint8_t *>(translated_co.image_data());
  rocjitsu::KernelDescriptorTranslator translated_parser(ROCJITSU_CODE_ARCH_RDNA4,
                                                         ROCJITSU_CODE_ARCH_RDNA4);
  const auto translated_infos = translated_parser.translate_image(
      {translated_image, translated_co.image_size()}, translated_text->sectionOffset(),
      translated_text->size(), rocjitsu::KernelDescriptorTranslationOptions{});
  const auto translated_info = std::find_if(
      translated_infos.begin(), translated_infos.end(),
      [kd_file_off](const auto &info) { return info.descriptor_file_offset == kd_file_off; });
  ASSERT_NE(translated_info, translated_infos.end());

  EXPECT_GT(translated_info->entry_text_offset, original_info->entry_text_offset)
      << "CDNA4 workgroup-id SGPRs must be materialized from RDNA4's TTMP launch payload";

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(decoder, nullptr);
  ASSERT_FALSE(translated_co.text_sections().empty());
  const auto *text = translated_co.text_sections()[0];
  const auto *words = reinterpret_cast<const uint32_t *>(text->data());

  ASSERT_EQ(original_info->entry_text_offset % sizeof(uint32_t), 0u);
  ASSERT_EQ(translated_info->entry_text_offset % sizeof(uint32_t), 0u);
  ASSERT_LT(original_info->entry_text_offset, text->size());

  std::unique_ptr<rocjitsu::Instruction> original_entry(
      decoder->decode(&words[original_info->entry_text_offset / sizeof(uint32_t)]));
  ASSERT_NE(original_entry, nullptr);
  EXPECT_NE(std::string_view(original_entry->mnemonic()), "s_branch")
      << "Original kernel entry should not be replaced by a prologue branch stub";

  const rocjitsu::Section *redirected_section = text;
  uint64_t redirected_section_offset = translated_info->entry_text_offset;
  if (redirected_section_offset >= text->size()) {
    redirected_section = nullptr;
    for (const auto &section : translated_co.all_sections()) {
      if (section->name() == ".rj_translations") {
        redirected_section = section.get();
        break;
      }
    }
    ASSERT_NE(redirected_section, nullptr)
        << "Descriptor ABI prologues should be materialized in .rj_translations";
    redirected_section_offset -= text->size();
  }
  ASSERT_LT(redirected_section_offset, redirected_section->size());

  const auto *redirected_words = reinterpret_cast<const uint32_t *>(redirected_section->data());
  std::unique_ptr<rocjitsu::Instruction> redirected_entry(
      decoder->decode(&redirected_words[redirected_section_offset / sizeof(uint32_t)]));
  ASSERT_NE(redirected_entry, nullptr);
  EXPECT_EQ(std::string_view(redirected_entry->mnemonic()), "s_mov_b32")
      << "Redirected kernel entry should begin with the descriptor ABI prologue";
}

TEST(CodeObjectPatcher, KernelEntryPrologueIs256ByteAligned) {
  Executable exec(kernel_path("vector_add"));
  ASSERT_TRUE(exec.is_valid());
  ASSERT_GT(exec.num_code_objects(ROCJITSU_CODE_TARGET_GFX950), 0u);

  const auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  ASSERT_NE(co, nullptr);

  rocjitsu::CodeObjectPatcher patcher(*co);
  ASSERT_FALSE(co->text_sections().empty());
  const auto *image = reinterpret_cast<const uint8_t *>(co->image_data());
  const auto *text = co->text_sections()[0];
  rocjitsu::KernelDescriptorTranslator parser(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA4);
  const auto infos =
      parser.translate_image({image, co->image_size()}, text->sectionOffset(), text->size(),
                             rocjitsu::KernelDescriptorTranslationOptions{});
  ASSERT_FALSE(infos.empty());

  patcher.set_cave_start(infos[0].entry_text_offset + sizeof(uint32_t));
  const std::array<uint32_t, 1> prologue = {rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4)};
  const uint64_t prologue_entry = patcher.append_kernel_entry_prologue(
      infos[0].entry_text_offset, prologue, ROCJITSU_CODE_ARCH_RDNA4);

  EXPECT_EQ(prologue_entry % 256, infos[0].entry_text_offset % 256)
      << "Kernel descriptor entry points are hardware launch addresses; the cave prologue must "
         "preserve the original entry's .text-relative alignment residue";
}

TEST(BinaryTranslatorE2E, OutputDecodesAsValidRdna4) {
  Executable exec(kernel_path("vector_add"));
  ASSERT_TRUE(exec.is_valid());
  ASSERT_GT(exec.num_code_objects(ROCJITSU_CODE_TARGET_GFX950), 0u);

  const auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  ASSERT_NE(co, nullptr);

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA4);
  auto result = translator.translate(*co);
  ASSERT_FALSE(result.elf_bytes.empty());

  // Construct an RDNA4 code object from the translated ELF bytes.
  rocjitsu::AmdGpuCodeObject translated_co(result.elf_bytes.data(), result.elf_bytes.size());

  // Decode every instruction with the RDNA4 decoder.
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(decoder, nullptr);

  int decode_failures = 0;
  int inst_count = 0;
  for (const auto *sec : translated_co.code_sections()) {
    const auto *data = reinterpret_cast<const uint32_t *>(sec->data());
    const size_t words = sec->size() / sizeof(uint32_t);
    size_t pc = 0;
    while (pc < words) {
      try {
        std::unique_ptr<rocjitsu::Instruction> inst(decoder->decode(&data[pc]));
        if (!inst) {
          ++decode_failures;
          ++pc;
          continue;
        }
        pc += inst->size() / 4;
        ++inst_count;
      } catch (const std::exception &e) {
        std::cerr << "  decode fail at 0x" << std::hex << pc * 4 << " word=0x" << data[pc] << ": "
                  << e.what() << "\n";
        ++decode_failures;
        ++pc;
      }
    }
  }
  EXPECT_GT(inst_count, 0) << "Text section should contain instructions";
  EXPECT_EQ(decode_failures, 0) << decode_failures << " instructions failed to decode as RDNA4";
}

TEST(BinaryTranslatorE2E, NoGfx9WaitcntInOutput) {
  Executable exec(kernel_path("vector_add"));
  ASSERT_TRUE(exec.is_valid());
  ASSERT_GT(exec.num_code_objects(ROCJITSU_CODE_TARGET_GFX950), 0u);

  const auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  ASSERT_NE(co, nullptr);

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA4);
  auto result = translator.translate(*co);
  ASSERT_FALSE(result.elf_bytes.empty());

  rocjitsu::AmdGpuCodeObject translated_co(result.elf_bytes.data(), result.elf_bytes.size());
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(decoder, nullptr);

  for (const auto *sec : translated_co.code_sections()) {
    const auto *data = reinterpret_cast<const uint32_t *>(sec->data());
    const size_t words = sec->size() / sizeof(uint32_t);
    size_t pc = 0;
    while (pc < words) {
      try {
        std::unique_ptr<rocjitsu::Instruction> inst(decoder->decode(&data[pc]));
        if (!inst) {
          ++pc;
          continue;
        }
        EXPECT_NE(std::string_view(inst->mnemonic()), "s_waitcnt")
            << "GFX9 s_waitcnt found in translated output at offset 0x" << std::hex << pc * 4;
        pc += inst->size() / 4;
      } catch (...) {
        ++pc;
      }
    }
  }
}

TEST(BinaryTranslatorE2E, TextSizesMatch) {
  Executable exec(kernel_path("vector_add"));
  ASSERT_TRUE(exec.is_valid());
  ASSERT_GT(exec.num_code_objects(ROCJITSU_CODE_TARGET_GFX950), 0u);

  const auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  ASSERT_NE(co, nullptr);

  const size_t original_text_size =
      co->text_sections().empty() ? 0 : co->text_sections()[0]->size();

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA4);
  auto result = translator.translate(*co);
  ASSERT_FALSE(result.elf_bytes.empty());

  rocjitsu::AmdGpuCodeObject translated_co(result.elf_bytes.data(), result.elf_bytes.size());
  const size_t translated_text_size =
      translated_co.text_sections().empty() ? 0 : translated_co.text_sections()[0]->size();

  // Code caves are separate from .text; the original instruction layout must
  // remain byte-for-byte sized so existing branches keep their offsets.
  EXPECT_EQ(translated_text_size, original_text_size);
}

TEST(BinaryTranslatorE2E, WriteTranslatedElfToFile) {
  Executable exec(kernel_path("vector_add"));
  ASSERT_TRUE(exec.is_valid());
  ASSERT_GT(exec.num_code_objects(ROCJITSU_CODE_TARGET_GFX950), 0u);

  const auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  ASSERT_NE(co, nullptr);

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA4);
  auto result = translator.translate(*co);
  ASSERT_FALSE(result.elf_bytes.empty());

  // Write a GFX1201 variant (same ISA, different MACH flag).
  auto elf_1201 = result.elf_bytes;
  // Patch e_flags: clear low 8 bits (MACH), set GFX1201 = 0x4E.
  uint32_t e_flags = 0;
  std::memcpy(&e_flags, elf_1201.data() + 48, 4);
  e_flags = (e_flags & ~0xFFu) | 0x4E;
  std::memcpy(elf_1201.data() + 48, &e_flags, 4);

  const char *out_path = "/tmp/vector_add_gfx1201.co";
  FILE *f = fopen(out_path, "wb");
  ASSERT_NE(f, nullptr);
  fwrite(elf_1201.data(), 1, elf_1201.size(), f);
  fclose(f);
  printf("  Wrote translated ELF to %s (%zu bytes)\n", out_path, elf_1201.size());
}

TEST(BinaryTranslatorE2E, DumpTranslation) {
  Executable exec(kernel_path("vector_add"));
  ASSERT_TRUE(exec.is_valid());
  ASSERT_GT(exec.num_code_objects(ROCJITSU_CODE_TARGET_GFX950), 0u);

  const auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  ASSERT_NE(co, nullptr);

  auto dump = [](const char *label, const uint8_t *text, size_t size, rj_code_arch_t arch) {
    auto dec = Decoder::create(arch);
    if (!dec)
      return;
    const auto *data = reinterpret_cast<const uint32_t *>(text);
    size_t words = size / 4, pc = 0;
    printf("\n--- %s (%zu bytes, %zu words) ---\n", label, size, words);
    while (pc < words) {
      try {
        std::unique_ptr<rocjitsu::Instruction> inst(dec->decode(&data[pc]));
        if (!inst) {
          printf("  0x%04zx: ???\n", pc * 4);
          ++pc;
          continue;
        }
        printf("  0x%04zx: %-45s [", pc * 4, inst->disassemble().c_str());
        for (int i = 0; i < inst->size() / 4; i++)
          printf("%s%08X", i ? " " : "", data[pc + i]);
        printf("]\n");
        pc += inst->size() / 4;
      } catch (...) {
        printf("  0x%04zx: [decode error] 0x%08X\n", pc * 4, data[pc]);
        ++pc;
      }
    }
  };

  for (const auto *sec : co->text_sections())
    dump("CDNA4 source", reinterpret_cast<const uint8_t *>(sec->data()), sec->size(),
         ROCJITSU_CODE_ARCH_CDNA4);

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA4);
  auto result = translator.translate(*co);
  ASSERT_FALSE(result.elf_bytes.empty());

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  for (const auto *sec : translated.code_sections())
    dump("RDNA4 translated", reinterpret_cast<const uint8_t *>(sec->data()), sec->size(),
         ROCJITSU_CODE_ARCH_RDNA4);

  if (!result.warnings.empty()) {
    printf("\n--- Warnings (%zu) ---\n", result.warnings.size());
    for (const auto &w : result.warnings)
      printf("  %s\n", w.c_str());
  }
}

#endif // HAS_DEVICE_KERNELS
