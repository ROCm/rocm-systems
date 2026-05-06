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
///   - E2E binary translation: representative CDNA4 code objects → RDNA3 ELF
///     with valid RDNA3 instructions and explicit fail-closed matrix gaps
///
/// These tests complement the hardware tests in hsa_translate_test.cpp which
/// verify correctness on real RDNA3/gfx1100 GPUs.

#include "rocjitsu/analysis/liveness.h"
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/dbt/binary_translator.h"
#include "rocjitsu/code/dbt/encoding_translator.h"
#include "rocjitsu/code/dbt/generated/encoding_cdna4_to_rdna3.h"
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
#include "rocjitsu/code/patch/instruction_builder.h"
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3/machine_insts.h"
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
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace rocjitsu {
namespace {

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

TEST(CoherencyRemap, Gfx940ToGfx11) {
  auto coh = remap_gfx940_to_gfx11({1, 1, 1});
  EXPECT_EQ(coh.glc, 1);
  EXPECT_EQ(coh.slc, 1);
  EXPECT_EQ(coh.dlc, 1);
}

TEST(CoherencyRemap, Gfx9GlcToGfx11) {
  auto coh_glc1 = remap_gfx9_to_gfx11({1});
  EXPECT_EQ(coh_glc1.glc, 1);
  EXPECT_EQ(coh_glc1.slc, 0);
  EXPECT_EQ(coh_glc1.dlc, 0);

  auto coh_glc0 = remap_gfx9_to_gfx11({0});
  EXPECT_EQ(coh_glc0.glc, 0);
  EXPECT_EQ(coh_glc0.slc, 0);
  EXPECT_EQ(coh_glc0.dlc, 0);
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

TEST(EncodingTranslator, Cdna4ToRdna3MubufRemapsCoherency) {
  cdna4::MubufMachineInst src{};
  src.offset = 0x123;
  src.offen = 1;
  src.idxen = 1;
  src.sc0 = 1;
  src.sc1 = 1;
  src.nt = 1;
  src.op = 0;
  src.encoding = 0x38;
  src.vaddr = 4;
  src.vdata = 5;
  src.srsrc = 6;
  src.soffset = 7;
  uint32_t words[2];
  std::memcpy(words, &src, sizeof(src));

  auto result =
      cdna4_to_rdna3::translate_encoding_cdna4_to_rdna3(kEnc_MUBUF, words[0], words[1], 0, 0);

  ASSERT_EQ(result.word_count, 2);
  rdna3::MubufMachineInst dst{};
  std::memcpy(&dst, result.words, sizeof(dst));
  EXPECT_EQ(dst.glc, 1);
  EXPECT_EQ(dst.slc, 1);
  EXPECT_EQ(dst.dlc, 1);
  EXPECT_EQ(dst.vaddr, 4);
  EXPECT_EQ(dst.vdata, 5);
  EXPECT_EQ(dst.srsrc, 6);
}

TEST(EncodingTranslator, Cdna4ToRdna3MtbufPacksFormatAndRemapsCoherency) {
  cdna4::MtbufMachineInst src{};
  src.offset = 0x321;
  src.offen = 1;
  src.idxen = 1;
  src.sc0 = 1;
  src.sc1 = 1;
  src.nt = 1;
  src.op = 0;
  src.dfmt = 4;
  src.nfmt = 7;
  src.encoding = kEnc_MTBUF >> 3;
  src.vaddr = 4;
  src.vdata = 5;
  src.srsrc = 6;
  src.soffset = 7;
  uint32_t words[2];
  std::memcpy(words, &src, sizeof(src));

  auto result =
      cdna4_to_rdna3::translate_encoding_cdna4_to_rdna3(kEnc_MTBUF, words[0], words[1], 0, 0);

  ASSERT_EQ(result.word_count, 2);
  rdna3::MtbufMachineInst dst{};
  std::memcpy(&dst, result.words, sizeof(dst));
  EXPECT_EQ(dst.glc, 1);
  EXPECT_EQ(dst.slc, 1);
  EXPECT_EQ(dst.dlc, 1);
  EXPECT_EQ(dst.format, 0x74);
  EXPECT_EQ(dst.vaddr, 4);
  EXPECT_EQ(dst.vdata, 5);
  EXPECT_EQ(dst.srsrc, 6);
  EXPECT_EQ(dst.tfe, 0);
}

TEST(EncodingTranslator, Cdna4ToRdna3SmemPreservesGlc) {
  cdna4::SmemMachineInst src{};
  src.sbase = 5;
  src.sdata = 3;
  src.soffset_en = 1;
  src.glc = 1;
  src.op = 0;
  src.offset = 0x100;
  src.soffset = 12;
  src.encoding = 0x3D;
  uint32_t words[2];
  std::memcpy(words, &src, sizeof(src));

  auto result =
      cdna4_to_rdna3::translate_encoding_cdna4_to_rdna3(kEnc_SMEM, words[0], words[1], 0, 0);

  ASSERT_EQ(result.word_count, 2);
  rdna3::SmemMachineInst dst{};
  std::memcpy(&dst, result.words, sizeof(dst));
  EXPECT_EQ(dst.glc, 1);
  EXPECT_EQ(dst.dlc, 0);
  EXPECT_EQ(dst.sbase, 5);
  EXPECT_EQ(dst.sdata, 3);
  EXPECT_EQ(dst.soffset, 12);
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

} // namespace
} // namespace rocjitsu

// --- WaitcntTranslator tests ---
#include "rocjitsu/code/dbt/semantic_translator.h"

using rocjitsu::decode_waitcnt_gfx9;
using rocjitsu::encode_waitcnt_gfx11_simm16;
using rocjitsu::encode_waitcnt_gfx12;
using rocjitsu::SemanticTranslator;
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

TEST(WaitcntTranslator, EncodeGfx11ConvertsImmediateLayout) {
  EXPECT_EQ(encode_waitcnt_gfx11_simm16(WaitcntValues{18, 3, 4}), 0x4834);
  EXPECT_EQ(encode_waitcnt_gfx11_simm16(decode_waitcnt_gfx9(0xCF7F)), 0xFCF7);
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

namespace {

struct DecodeStats {
  size_t inst_count = 0;
  size_t decode_failures = 0;
};

DecodeStats decode_words_as(rj_code_arch_t arch, std::span<const uint32_t> words) {
  DecodeStats stats;
  auto decoder = rocjitsu::Decoder::create(arch);
  if (!decoder) {
    stats.decode_failures = words.size();
    return stats;
  }

  size_t pc = 0;
  while (pc < words.size()) {
    try {
      std::unique_ptr<rocjitsu::Instruction> inst(decoder->decode(&words[pc]));
      if (!inst || inst->size() <= 0) {
        ++stats.decode_failures;
        ++pc;
        continue;
      }
      pc += static_cast<size_t>(inst->size()) / sizeof(uint32_t);
      ++stats.inst_count;
    } catch (const std::exception &) {
      ++stats.decode_failures;
      ++pc;
    }
  }
  return stats;
}

std::unique_ptr<rocjitsu::Instruction> decode_cdna4(std::span<const uint32_t> words) {
  auto decoder = rocjitsu::Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  if (!decoder)
    return nullptr;
  return std::unique_ptr<rocjitsu::Instruction>(decoder->decode(words.data()));
}

std::unique_ptr<rocjitsu::Instruction> decode_one_as(rj_code_arch_t arch,
                                                     std::span<const uint32_t> words) {
  auto decoder = rocjitsu::Decoder::create(arch);
  if (!decoder)
    return nullptr;
  return std::unique_ptr<rocjitsu::Instruction>(decoder->decode(words.data()));
}

std::vector<uint32_t> append_trailing_literal_for_test(const rocjitsu::TranslationResult &tr,
                                                       std::span<const uint32_t> source_words,
                                                       size_t source_word_count) {
  std::vector<uint32_t> translated(tr.words, tr.words + tr.word_count);
  if (source_word_count == translated.size() + 1 && translated.size() < 3)
    translated.push_back(source_words[translated.size()]);
  return translated;
}

uint32_t make_cdna4_sopp(uint16_t op, uint16_t simm16) {
  rocjitsu::cdna4::SoppMachineInst src{};
  src.simm16 = simm16;
  src.op = op;
  src.encoding = rocjitsu::kEnc_SOPP;
  return std::bit_cast<uint32_t>(src);
}

uint32_t make_cdna4_s_waitcnt(uint16_t simm16) { return make_cdna4_sopp(12, simm16); }

std::array<uint32_t, 2> make_cdna4_v_lshl_add_u64_zero_shift() {
  rocjitsu::cdna4::Vop3MachineInst src{};
  src.vdst = 20;
  src.src0 = 256 + 8;
  src.src1 = 128; // inline constant 0, so the existing add-based lowering is semantic.
  src.src2 = 256 + 12;
  src.op = 520;
  src.encoding = 0x34;

  std::array<uint32_t, 2> words{};
  std::memcpy(words.data(), &src, sizeof(src));
  return words;
}

uint32_t make_cdna4_vopc_cmp(uint8_t op) {
  rocjitsu::cdna4::VopcMachineInst src{};
  src.src0 = 0;
  src.vsrc1 = 0;
  src.op = op;
  // CDNA4 VOPC primary encodings are 0xF8..0xFB after opcode high bits join
  // the top-level decode key; the raw 7-bit encoding field is therefore 0x3E.
  src.encoding = 0x3E;
  return std::bit_cast<uint32_t>(src);
}

std::array<uint32_t, 2> make_cdna4_vop3_cmp(uint16_t op, uint8_t sdst) {
  rocjitsu::cdna4::Vop3MachineInst src{};
  src.vdst = sdst;
  src.src0 = 256 + 2;
  src.src1 = 256 + 4;
  src.op = op;
  src.encoding = 0x34;

  std::array<uint32_t, 2> words{};
  std::memcpy(words.data(), &src, sizeof(src));
  return words;
}

uint32_t make_rdna3_s_mov_b64(uint8_t sdst, uint16_t ssrc0) {
  rocjitsu::rdna3::Sop1MachineInst dst{};
  dst.ssrc0 = ssrc0 & 0xFF;
  dst.op = 1;
  dst.sdst = sdst & 0x7F;
  dst.encoding = rocjitsu::kEnc_SOP1;
  return std::bit_cast<uint32_t>(dst);
}

void expect_rdna3_s_mov_b64(uint32_t word, uint8_t sdst, uint16_t ssrc0) {
  EXPECT_EQ(word, make_rdna3_s_mov_b64(sdst, ssrc0));

  const std::array<uint32_t, 1> words{word};
  auto inst = decode_one_as(ROCJITSU_CODE_ARCH_RDNA3, words);
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(std::string_view(inst->mnemonic()), "s_mov_b64");
}

void expect_rdna3_v_mov_b32(uint32_t word, uint8_t vdst, uint16_t src0) {
  rocjitsu::rdna3::Vop1MachineInst dst{};
  static_assert(sizeof(dst) == sizeof(word));
  std::memcpy(&dst, &word, sizeof(word));
  EXPECT_EQ(dst.op, 1);
  EXPECT_EQ(dst.vdst, vdst);
  EXPECT_EQ(dst.src0, src0);

  const std::array<uint32_t, 1> words{word};
  auto inst = decode_one_as(ROCJITSU_CODE_ARCH_RDNA3, words);
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(std::string_view(inst->mnemonic()), "v_mov_b32_e32");
}

void expect_rdna3_vop2(uint32_t word, uint8_t op, uint8_t vdst, uint16_t src0, uint8_t vsrc1,
                       std::string_view mnemonic) {
  const auto dst = std::bit_cast<rocjitsu::rdna3::Vop2MachineInst>(word);
  EXPECT_EQ(dst.op, op);
  EXPECT_EQ(dst.vdst, vdst);
  EXPECT_EQ(dst.src0, src0);
  EXPECT_EQ(dst.vsrc1, vsrc1);

  const std::array<uint32_t, 1> words{word};
  auto inst = decode_one_as(ROCJITSU_CODE_ARCH_RDNA3, words);
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(std::string_view(inst->mnemonic()), mnemonic);
}

void expect_rdna3_vop3(std::span<const uint32_t> words, uint16_t op, uint8_t vdst, uint16_t src0,
                       uint16_t src1, std::string_view mnemonic) {
  SCOPED_TRACE(mnemonic);
  ASSERT_EQ(words.size(), 2u);
  rocjitsu::rdna3::Vop3MachineInst dst{};
  static_assert(sizeof(dst) == 2 * sizeof(uint32_t));
  std::memcpy(&dst, words.data(), sizeof(dst));
  EXPECT_EQ(dst.encoding, 0x28);
  EXPECT_EQ(dst.op, op);
  EXPECT_EQ(dst.vdst, vdst);
  EXPECT_EQ(dst.src0, src0);
  EXPECT_EQ(dst.src1, src1);
  EXPECT_EQ(dst.src2, 0);
  EXPECT_EQ(dst.abs, 0);
  EXPECT_EQ(dst.op_sel, 0);
  EXPECT_EQ(dst.clamp, 0);
  EXPECT_EQ(dst.omod, 0);
  EXPECT_EQ(dst.neg, 0);
}

void expect_rdna3_vop3_sdst(std::span<const uint32_t> words, uint16_t op, uint8_t vdst,
                            uint8_t sdst, uint16_t src0, uint16_t src1, uint16_t src2,
                            std::string_view mnemonic) {
  SCOPED_TRACE(mnemonic);
  ASSERT_EQ(words.size(), 2u);
  rocjitsu::rdna3::Vop3SdstEncMachineInst dst{};
  static_assert(sizeof(dst) == 2 * sizeof(uint32_t));
  std::memcpy(&dst, words.data(), sizeof(dst));
  EXPECT_EQ(dst.encoding, 0x28);
  EXPECT_EQ(dst.op, op);
  EXPECT_EQ(dst.vdst, vdst);
  EXPECT_EQ(dst.sdst, sdst);
  EXPECT_EQ(dst.src0, src0);
  EXPECT_EQ(dst.src1, src1);
  EXPECT_EQ(dst.src2, src2);
  EXPECT_EQ(dst.clamp, 0);
  EXPECT_EQ(dst.omod, 0);
  EXPECT_EQ(dst.neg, 0);
}

std::array<uint32_t, 2> make_cdna4_v_accvgpr_read() { return {0xD3D80000u, 0x00000000u}; }

std::array<uint32_t, 2> make_cdna4_v_accvgpr_write() { return {0xD3D90000u, 0x00000000u}; }

std::array<uint32_t, 1> make_cdna4_v_accvgpr_mov_b32_e32() {
  rocjitsu::cdna4::Vop1MachineInst src{};
  src.src0 = 0;
  src.op = 82;
  src.vdst = 0;
  src.encoding = rocjitsu::kEnc_VOP1 >> 2;
  return {std::bit_cast<uint32_t>(src)};
}

std::array<uint32_t, 2> make_cdna4_v_mfma_f32_16x16x16_f16() { return {0xD3CD0000u, 0x00000000u}; }

std::array<uint32_t, 2> make_cdna4_v_smfmac_f32_16x16x64_bf16() {
  return {0xD3B90000u, 0x00000000u};
}

std::array<uint32_t, 2> make_cdna4_tbuffer_load_format_x(bool acc = false) {
  rocjitsu::cdna4::MtbufMachineInst src{};
  src.op = 0;
  src.dfmt = 4;
  src.nfmt = 7;
  src.encoding = rocjitsu::kEnc_MTBUF >> 3;
  src.srsrc = 7;
  src.acc = acc ? 1 : 0;

  std::array<uint32_t, 2> words{};
  std::memcpy(words.data(), &src, sizeof(src));
  return words;
}

std::array<uint32_t, 2> make_cdna4_tbuffer_load_format_d16_x() {
  rocjitsu::cdna4::MtbufMachineInst src{};
  src.op = 8;
  src.dfmt = 4;
  src.nfmt = 7;
  src.encoding = rocjitsu::kEnc_MTBUF >> 3;
  src.srsrc = 7;

  std::array<uint32_t, 2> words{};
  std::memcpy(words.data(), &src, sizeof(src));
  return words;
}

uint32_t make_cdna4_s_mov_b32(uint8_t sdst, uint8_t ssrc0) {
  rocjitsu::cdna4::Sop1MachineInst src{};
  src.ssrc0 = ssrc0;
  src.sdst = sdst;
  src.op = 0;
  src.encoding = rocjitsu::kEnc_SOP1;
  return std::bit_cast<uint32_t>(src);
}

uint32_t make_cdna4_v_add_f32(uint8_t vdst, uint8_t vsrc1, uint16_t src0) {
  rocjitsu::cdna4::Vop2MachineInst src{};
  src.src0 = src0;
  src.vsrc1 = vsrc1;
  src.vdst = vdst;
  src.op = 1;
  src.encoding = 0;
  return std::bit_cast<uint32_t>(src);
}

uint32_t make_cdna4_v_lshrrev_b32(uint8_t vdst, uint8_t vsrc1, uint16_t src0) {
  rocjitsu::cdna4::Vop2MachineInst src{};
  src.src0 = src0;
  src.vsrc1 = vsrc1;
  src.vdst = vdst;
  src.op = 16;
  src.encoding = 0;
  return std::bit_cast<uint32_t>(src);
}

uint32_t make_cdna4_vop2(uint8_t op, uint8_t vdst, uint8_t vsrc1, uint16_t src0) {
  rocjitsu::cdna4::Vop2MachineInst src{};
  src.src0 = src0;
  src.vsrc1 = vsrc1;
  src.vdst = vdst;
  src.op = op;
  src.encoding = 0;
  return std::bit_cast<uint32_t>(src);
}

uint32_t make_cdna4_v_mov_b64_e32(uint8_t vdst, uint8_t src_vgpr) {
  rocjitsu::cdna4::Vop1MachineInst src{};
  src.src0 = 256 + src_vgpr;
  src.op = 56;
  src.vdst = vdst;
  src.encoding = rocjitsu::kEnc_VOP1 >> 2;
  return std::bit_cast<uint32_t>(src);
}

std::array<uint32_t, 2> make_cdna4_vop3(uint16_t op, uint8_t vdst, uint16_t src0, uint16_t src1) {
  rocjitsu::cdna4::Vop3MachineInst src{};
  src.vdst = vdst;
  src.src0 = src0;
  src.src1 = src1;
  src.op = op;
  src.encoding = 0x34;

  std::array<uint32_t, 2> words{};
  std::memcpy(words.data(), &src, sizeof(src));
  return words;
}

std::array<uint32_t, 2> make_cdna4_vop3_sdst(uint16_t op, uint8_t vdst, uint8_t sdst, uint16_t src0,
                                             uint16_t src1, uint16_t src2) {
  rocjitsu::cdna4::Vop3SdstEncMachineInst src{};
  src.vdst = vdst;
  src.sdst = sdst;
  src.src0 = src0;
  src.src1 = src1;
  src.src2 = src2;
  src.op = op;
  src.encoding = 0x34;

  std::array<uint32_t, 2> words{};
  std::memcpy(words.data(), &src, sizeof(src));
  return words;
}

template <typename Inst> std::array<uint32_t, 2> pack_64(const Inst &src) {
  static_assert(sizeof(Inst) == sizeof(uint32_t) * 2);
  std::array<uint32_t, 2> words{};
  std::memcpy(words.data(), &src, sizeof(src));
  return words;
}

const rocjitsu::InstructionLegalization *lookup_cdna4_to_rdna3(uint16_t encoding_id,
                                                               uint16_t opcode) {
  return rocjitsu::lookup(rocjitsu::kLegalization_cdna4_to_rdna3, encoding_id, opcode);
}

bool warnings_contain(const std::vector<std::string> &warnings, std::string_view needle) {
  return std::any_of(warnings.begin(), warnings.end(), [needle](const std::string &warning) {
    return warning.find(needle) != std::string::npos;
  });
}

size_t align_to(size_t value, size_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

std::vector<uint8_t> make_minimal_amdgpu_elf(std::span<const uint32_t> text_words,
                                             bool add_text_relocation = false,
                                             uint64_t relocation_text_offset = 0) {
  using KernelDescriptor = rocr::llvm::amdhsa::kernel_descriptor_t;
  constexpr uint64_t kTextVaddr = 0x1000;
  constexpr uint64_t kKernelDescriptorVaddr = 0x800;

  std::vector<char> shstr{'\0'};
  auto add_section_name = [&shstr](std::string_view name) {
    const auto offset = static_cast<uint32_t>(shstr.size());
    shstr.insert(shstr.end(), name.begin(), name.end());
    shstr.push_back('\0');
    return offset;
  };
  const uint32_t text_name = add_section_name(".text");
  const uint32_t rela_name = add_section_name(".rela.text");
  const uint32_t rodata_name = add_section_name(".rodata");
  const uint32_t symtab_name = add_section_name(".symtab");
  const uint32_t strtab_name = add_section_name(".strtab");
  const uint32_t shstr_name = add_section_name(".shstrtab");

  std::vector<char> strtab{'\0'};
  const uint32_t kd_sym_name = static_cast<uint32_t>(strtab.size());
  constexpr std::string_view kKernelDescriptorName = "minimal_kernel.kd";
  strtab.insert(strtab.end(), kKernelDescriptorName.begin(), kKernelDescriptorName.end());
  strtab.push_back('\0');

  constexpr uint16_t kTextSection = 1;
  const uint16_t rela_section = add_text_relocation ? 2 : 0;
  const uint16_t rodata_section = add_text_relocation ? 3 : 2;
  const uint16_t symtab_section = rodata_section + 1;
  const uint16_t strtab_section = rodata_section + 2;
  const uint16_t shstr_section = rodata_section + 3;
  const uint16_t section_count = shstr_section + 1;

  const size_t text_size = text_words.size() * sizeof(uint32_t);
  const size_t text_off = align_to(sizeof(rocjitsu::Elf64_Ehdr), alignof(uint32_t));
  const size_t rela_off = align_to(text_off + text_size, alignof(rocjitsu::Elf64_Rela));
  const size_t rela_size = add_text_relocation ? sizeof(rocjitsu::Elf64_Rela) : 0;
  const size_t rodata_off = align_to(rela_off + rela_size, alignof(KernelDescriptor));
  const size_t rodata_size = sizeof(KernelDescriptor);
  const size_t symtab_off = align_to(rodata_off + rodata_size, alignof(rocjitsu::Elf64_Sym));
  const size_t symtab_size = 2 * sizeof(rocjitsu::Elf64_Sym);
  const size_t strtab_off = symtab_off + symtab_size;
  const size_t shstr_off = strtab_off + strtab.size();
  const size_t shoff = align_to(shstr_off + shstr.size(), alignof(rocjitsu::Elf64_Shdr));

  std::vector<uint8_t> image(shoff + section_count * sizeof(rocjitsu::Elf64_Shdr), 0);

  rocjitsu::Elf64_Ehdr ehdr{};
  std::memcpy(ehdr.e_ident, rocjitsu::EI_MAGIC, rocjitsu::EI_MAGIC_SIZE);
  ehdr.e_ident[rocjitsu::EI_CLASS] = rocjitsu::ELFCLASS64;
  ehdr.e_ident[rocjitsu::EI_DATA] = 1;
  ehdr.e_ident[rocjitsu::EI_VERSION] = 1;
  ehdr.e_ident[rocjitsu::EI_OSABI] = rocjitsu::ELFOSABI_AMDGPU_HSA;
  ehdr.e_machine = rocjitsu::EM_AMDGPU;
  ehdr.e_version = 1;
  ehdr.e_shoff = shoff;
  ehdr.e_flags = rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX950;
  ehdr.e_ehsize = sizeof(ehdr);
  ehdr.e_shentsize = sizeof(rocjitsu::Elf64_Shdr);
  ehdr.e_shnum = section_count;
  ehdr.e_shstrndx = shstr_section;
  std::memcpy(image.data(), &ehdr, sizeof(ehdr));

  std::memcpy(image.data() + text_off, text_words.data(), text_size);
  if (add_text_relocation) {
    rocjitsu::Elf64_Rela rela{};
    rela.r_offset = kTextVaddr + relocation_text_offset;
    std::memcpy(image.data() + rela_off, &rela, sizeof(rela));
  }

  KernelDescriptor kd{};
  kd.kernel_code_entry_byte_offset = static_cast<int64_t>(kTextVaddr - kKernelDescriptorVaddr);
  std::memcpy(image.data() + rodata_off, &kd, sizeof(kd));

  std::array<rocjitsu::Elf64_Sym, 2> symtab{};
  symtab[1].st_name = kd_sym_name;
  symtab[1].st_shndx = rodata_section;
  symtab[1].st_value = kKernelDescriptorVaddr;
  symtab[1].st_size = sizeof(KernelDescriptor);
  std::memcpy(image.data() + symtab_off, symtab.data(), symtab_size);

  std::memcpy(image.data() + strtab_off, strtab.data(), strtab.size());
  std::memcpy(image.data() + shstr_off, shstr.data(), shstr.size());

  std::vector<rocjitsu::Elf64_Shdr> shdr(section_count);
  shdr[kTextSection].sh_name = text_name;
  shdr[kTextSection].sh_type = rocjitsu::SHT_PROGBITS;
  shdr[kTextSection].sh_flags = 0x6;
  shdr[kTextSection].sh_addr = kTextVaddr;
  shdr[kTextSection].sh_offset = text_off;
  shdr[kTextSection].sh_size = text_size;
  shdr[kTextSection].sh_addralign = alignof(uint32_t);

  if (add_text_relocation) {
    shdr[rela_section].sh_name = rela_name;
    shdr[rela_section].sh_type = rocjitsu::SHT_RELA;
    shdr[rela_section].sh_offset = rela_off;
    shdr[rela_section].sh_size = rela_size;
    shdr[rela_section].sh_link = symtab_section;
    shdr[rela_section].sh_info = kTextSection;
    shdr[rela_section].sh_addralign = alignof(rocjitsu::Elf64_Rela);
    shdr[rela_section].sh_entsize = sizeof(rocjitsu::Elf64_Rela);
  }

  shdr[rodata_section].sh_name = rodata_name;
  shdr[rodata_section].sh_type = rocjitsu::SHT_PROGBITS;
  shdr[rodata_section].sh_flags = 0x2;
  shdr[rodata_section].sh_addr = kKernelDescriptorVaddr;
  shdr[rodata_section].sh_offset = rodata_off;
  shdr[rodata_section].sh_size = rodata_size;
  shdr[rodata_section].sh_addralign = alignof(KernelDescriptor);

  shdr[symtab_section].sh_name = symtab_name;
  shdr[symtab_section].sh_type = rocjitsu::SHT_SYMTAB;
  shdr[symtab_section].sh_offset = symtab_off;
  shdr[symtab_section].sh_size = symtab_size;
  shdr[symtab_section].sh_link = strtab_section;
  shdr[symtab_section].sh_info = 1;
  shdr[symtab_section].sh_addralign = alignof(rocjitsu::Elf64_Sym);
  shdr[symtab_section].sh_entsize = sizeof(rocjitsu::Elf64_Sym);

  shdr[strtab_section].sh_name = strtab_name;
  shdr[strtab_section].sh_type = rocjitsu::SHT_STRTAB;
  shdr[strtab_section].sh_offset = strtab_off;
  shdr[strtab_section].sh_size = strtab.size();
  shdr[strtab_section].sh_addralign = 1;

  shdr[shstr_section].sh_name = shstr_name;
  shdr[shstr_section].sh_type = rocjitsu::SHT_STRTAB;
  shdr[shstr_section].sh_offset = shstr_off;
  shdr[shstr_section].sh_size = shstr.size();
  shdr[shstr_section].sh_addralign = 1;

  std::memcpy(image.data() + shoff, shdr.data(), shdr.size() * sizeof(shdr.front()));

  return image;
}

std::vector<uint32_t> first_text_words(const rocjitsu::AmdGpuCodeObject &co) {
  std::vector<uint32_t> words;
  if (co.text_sections().empty())
    return words;
  const auto *text = co.text_sections().front();
  words.resize(text->size() / sizeof(uint32_t));
  std::memcpy(words.data(), text->data(), words.size() * sizeof(uint32_t));
  return words;
}

struct SemanticInstructionContext {
  explicit SemanticInstructionContext(std::span<const uint32_t> source_words)
      : elf(make_minimal_amdgpu_elf(source_words)), code_object(elf.data(), elf.size()),
        decoder(rocjitsu::Decoder::create(ROCJITSU_CODE_ARCH_CDNA4)) {
    if (!code_object.is_valid() || !decoder)
      return;

    blocks = rocjitsu::BasicBlock::build(code_object, *decoder);
    scope.reserve(blocks.size());
    for (const auto &block : blocks) {
      if (block)
        scope.push_back(block.get());
    }

    if (!scope.empty())
      liveness = std::make_unique<rocjitsu::LivenessAnalysis>(rocjitsu::KernelBlockScope(scope));
  }

  [[nodiscard]] bool is_valid() const { return code_object.is_valid() && liveness != nullptr; }

  [[nodiscard]] const rocjitsu::Instruction *first_instruction() const {
    for (const auto &block : blocks) {
      if (!block)
        continue;
      auto it = block->instructions().begin();
      if (it != block->instructions().end())
        return &*it;
    }
    return nullptr;
  }

  [[nodiscard]] const rocjitsu::LivenessAnalysis &live() const { return *liveness; }

  std::vector<uint8_t> elf;
  rocjitsu::AmdGpuCodeObject code_object;
  std::unique_ptr<rocjitsu::Decoder> decoder;
  std::vector<std::unique_ptr<rocjitsu::BasicBlock>> blocks;
  std::vector<rocjitsu::BasicBlock *> scope;
  std::unique_ptr<rocjitsu::LivenessAnalysis> liveness;
};

void expect_unsupported_expansion_fails_closed(std::span<const uint32_t> unsupported_source,
                                               std::string_view expected_mnemonic,
                                               uint16_t expected_opcode,
                                               std::string_view expected_category) {
  auto unsupported_inst = decode_cdna4(unsupported_source);
  ASSERT_NE(unsupported_inst, nullptr);
  ASSERT_EQ(std::string_view(unsupported_inst->mnemonic()), expected_mnemonic);
  ASSERT_EQ(unsupported_inst->opcode(), expected_opcode);

  std::vector<uint32_t> source_words(unsupported_source.begin(), unsupported_source.end());
  source_words.push_back(make_cdna4_sopp(1, 0));
  source_words.push_back(rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4));
  source_words.push_back(rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4));

  const auto elf = make_minimal_amdgpu_elf(source_words);
  rocjitsu::AmdGpuCodeObject co(elf.data(), elf.size());
  ASSERT_TRUE(co.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA3);
  const auto result = translator.translate(co);
  EXPECT_TRUE(result.elf_bytes.empty());
  EXPECT_TRUE(warnings_contain(result.warnings, "unsupported expansion"));
  EXPECT_TRUE(warnings_contain(result.warnings, expected_mnemonic));
  EXPECT_TRUE(warnings_contain(result.warnings, "opcode=" + std::to_string(expected_opcode)));
  EXPECT_TRUE(warnings_contain(result.warnings, expected_category))
      << "unsupported diagnostic should identify " << expected_category
      << (result.warnings.empty() ? "" : result.warnings.front());
}

void expect_unsupported_encoding_fails_closed(std::span<const uint32_t> unsupported_source,
                                              std::string_view expected_mnemonic,
                                              std::string_view expected_category) {
  auto unsupported_inst = decode_cdna4(unsupported_source);
  ASSERT_NE(unsupported_inst, nullptr);
  ASSERT_EQ(std::string_view(unsupported_inst->mnemonic()), expected_mnemonic);

  std::vector<uint32_t> source_words(unsupported_source.begin(), unsupported_source.end());
  source_words.push_back(make_cdna4_sopp(1, 0));
  source_words.push_back(rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4));
  source_words.push_back(rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4));

  const auto elf = make_minimal_amdgpu_elf(source_words);
  rocjitsu::AmdGpuCodeObject co(elf.data(), elf.size());
  ASSERT_TRUE(co.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA3);
  const auto result = translator.translate(co);
  EXPECT_TRUE(result.elf_bytes.empty());
  EXPECT_TRUE(warnings_contain(result.warnings, "unsupported encoding translation"));
  EXPECT_TRUE(warnings_contain(result.warnings, expected_mnemonic));
  EXPECT_TRUE(warnings_contain(result.warnings,
                               "encoding_id=" + std::to_string(unsupported_inst->encoding_id())));
  EXPECT_TRUE(
      warnings_contain(result.warnings, "opcode=" + std::to_string(unsupported_inst->opcode())));
  EXPECT_TRUE(warnings_contain(result.warnings, expected_category))
      << "unsupported diagnostic should identify " << expected_category
      << (result.warnings.empty() ? "" : result.warnings.front());
}

} // namespace

TEST(Cdna4ToRdna3Legalization, DuplicateKeysResolveRepresentativeRows) {
  const auto *branch =
      rocjitsu::lookup(rocjitsu::kLegalization_cdna4_to_rdna3, rocjitsu::kEnc_SOPP, 2);
  ASSERT_NE(branch, nullptr);
  EXPECT_EQ(branch->action, rocjitsu::Action::Substitute);
  EXPECT_EQ(branch->target_opcode, 32);

  const auto *s_mov =
      rocjitsu::lookup(rocjitsu::kLegalization_cdna4_to_rdna3, rocjitsu::kEnc_SOP1, 3);
  ASSERT_NE(s_mov, nullptr);
  EXPECT_NE(s_mov->action, rocjitsu::Action::Expand);
  EXPECT_EQ(s_mov->target_opcode, 3);

  const auto *removed_false_cmp =
      rocjitsu::lookup(rocjitsu::kLegalization_cdna4_to_rdna3, rocjitsu::kEnc_VOPC, 32);
  ASSERT_NE(removed_false_cmp, nullptr);
  EXPECT_EQ(removed_false_cmp->action, rocjitsu::Action::Expand);
  EXPECT_EQ(removed_false_cmp->target_opcode, 0);

  const std::array<uint32_t, 1> cmpx_false_source{make_cdna4_vopc_cmp(176)};
  auto cmpx_false_inst = decode_cdna4(cmpx_false_source);
  ASSERT_NE(cmpx_false_inst, nullptr);
  const auto *removed_cmpx_false_cmp =
      rocjitsu::lookup(rocjitsu::kLegalization_cdna4_to_rdna3, cmpx_false_inst->encoding_id(),
                       cmpx_false_inst->opcode());
  ASSERT_NE(removed_cmpx_false_cmp, nullptr);
  EXPECT_EQ(removed_cmpx_false_cmp->action, rocjitsu::Action::Expand);
  EXPECT_EQ(removed_cmpx_false_cmp->target_opcode, 0);
}

TEST(Cdna4ToRdna3MemoryFamilies, SmemLoadPreservesCoherencyAndNullOffset) {
  rocjitsu::cdna4::SmemMachineInst src{};
  src.sbase = 6;
  src.sdata = 14;
  src.soffset_en = 0;
  src.nv = 0;
  src.glc = 1;
  src.imm = 1;
  src.op = 8; // s_buffer_load_dword -> s_buffer_load_b32
  src.encoding = 0x3D;
  src.offset = 0x12345;
  src.soffset = 17;
  const auto words = pack_64(src);

  const auto *leg = lookup_cdna4_to_rdna3(rocjitsu::kEnc_SMEM, src.op);
  ASSERT_NE(leg, nullptr);
  ASSERT_EQ(leg->action, rocjitsu::Action::Lower);

  const auto translated = rocjitsu::cdna4_to_rdna3::translate_encoding_cdna4_to_rdna3(
      rocjitsu::kEnc_SMEM, words[0], words[1], 0, leg->target_opcode);
  ASSERT_EQ(translated.word_count, 2);

  rocjitsu::rdna3::SmemMachineInst dst{};
  std::memcpy(&dst, translated.words, sizeof(dst));
  EXPECT_EQ(dst.sbase, src.sbase);
  EXPECT_EQ(dst.sdata, src.sdata);
  EXPECT_EQ(dst.offset, src.offset);
  // CDNA soffset_en=0 disables the scalar offset; RDNA3 encodes that as null.
  EXPECT_EQ(dst.soffset, 0x7Cu);
  EXPECT_EQ(dst.glc, 1u);
  EXPECT_EQ(dst.dlc, 0u);

  auto dst_inst =
      decode_one_as(ROCJITSU_CODE_ARCH_RDNA3, std::span<const uint32_t>(translated.words, 2));
  ASSERT_NE(dst_inst, nullptr);
  EXPECT_EQ(std::string_view(dst_inst->mnemonic()), "s_buffer_load_b32");
}

TEST(Cdna4ToRdna3MemoryFamilies, SmemNvRequiresResidualHandling) {
  rocjitsu::cdna4::SmemMachineInst src{};
  src.nv = 1;
  src.op = 0;
  src.encoding = 0x3D;
  const auto words = pack_64(src);

  // CDNA4 nv has no same-size RDNA3 representation, so translation must fail.
  const auto *leg = lookup_cdna4_to_rdna3(rocjitsu::kEnc_SMEM, src.op);
  ASSERT_NE(leg, nullptr);
  const auto translated = rocjitsu::cdna4_to_rdna3::translate_encoding_cdna4_to_rdna3(
      rocjitsu::kEnc_SMEM, words[0], words[1], 0, leg->target_opcode);
  EXPECT_EQ(translated.word_count, 0);
}

TEST(Cdna4ToRdna3MemoryFamilies, MubufLoadAndAtomicPreserveOperandsAndCoherency) {
  struct Case {
    uint16_t src_op;
    uint16_t target_op;
    const char *mnemonic;
  };
  const Case cases[] = {
      {20, 20, "buffer_load_b32"},
      {66, 53, "buffer_atomic_add_u32"},
  };

  for (const auto &tc : cases) {
    rocjitsu::cdna4::MubufMachineInst src{};
    src.offset = 0x321;
    src.offen = 1;
    src.idxen = 1;
    src.sc0 = 1;
    src.sc1 = 1;
    src.lds = 0;
    src.nt = 1;
    src.op = tc.src_op;
    src.encoding = 0x38;
    src.vaddr = 4;
    src.vdata = 5;
    src.srsrc = 6;
    src.acc = 0;
    src.soffset = 0x7F;
    const auto words = pack_64(src);

    const auto *leg = lookup_cdna4_to_rdna3(rocjitsu::kEnc_MUBUF, src.op);
    ASSERT_NE(leg, nullptr);
    ASSERT_EQ(leg->action, rocjitsu::Action::Lower);
    ASSERT_EQ(leg->target_opcode, tc.target_op);

    const auto translated = rocjitsu::cdna4_to_rdna3::translate_encoding_cdna4_to_rdna3(
        rocjitsu::kEnc_MUBUF, words[0], words[1], 0, leg->target_opcode);
    ASSERT_EQ(translated.word_count, 2);

    rocjitsu::rdna3::MubufMachineInst dst{};
    std::memcpy(&dst, translated.words, sizeof(dst));
    EXPECT_EQ(dst.offset, src.offset);
    EXPECT_EQ(dst.offen, src.offen);
    EXPECT_EQ(dst.idxen, src.idxen);
    EXPECT_EQ(dst.vaddr, src.vaddr);
    EXPECT_EQ(dst.vdata, src.vdata);
    EXPECT_EQ(dst.srsrc, src.srsrc);
    // The 7-bit CDNA scalar null sentinel becomes RDNA3's 0x7c sentinel.
    EXPECT_EQ(dst.soffset, 0x7Cu);
    EXPECT_EQ(dst.glc, 1u);
    EXPECT_EQ(dst.slc, 1u);
    EXPECT_EQ(dst.dlc, 1u);
    EXPECT_EQ(dst.tfe, 0u);
    EXPECT_EQ(dst.op, tc.target_op);

    auto dst_inst =
        decode_one_as(ROCJITSU_CODE_ARCH_RDNA3, std::span<const uint32_t>(translated.words, 2));
    ASSERT_NE(dst_inst, nullptr);
    EXPECT_EQ(std::string_view(dst_inst->mnemonic()), tc.mnemonic);
  }
}

TEST(Cdna4ToRdna3MemoryFamilies, MubufSourceOnlyDomainBitsRequireResidualHandling) {
  rocjitsu::cdna4::MubufMachineInst src{};
  src.op = 20;
  src.encoding = 0x38;

  const auto *leg = lookup_cdna4_to_rdna3(rocjitsu::kEnc_MUBUF, src.op);
  ASSERT_NE(leg, nullptr);

  // lds and acc change domains/register files and require residual handling.
  src.lds = 1;
  auto words = pack_64(src);
  auto translated = rocjitsu::cdna4_to_rdna3::translate_encoding_cdna4_to_rdna3(
      rocjitsu::kEnc_MUBUF, words[0], words[1], 0, leg->target_opcode);
  EXPECT_EQ(translated.word_count, 0);

  src.lds = 0;
  src.acc = 1;
  words = pack_64(src);
  translated = rocjitsu::cdna4_to_rdna3::translate_encoding_cdna4_to_rdna3(
      rocjitsu::kEnc_MUBUF, words[0], words[1], 0, leg->target_opcode);
  EXPECT_EQ(translated.word_count, 0);
}

TEST(Cdna4ToRdna3MemoryFamilies, FlatSegmentsPreserveDomainAndCoherency) {
  const auto *leg = lookup_cdna4_to_rdna3(rocjitsu::kEnc_FLAT, 20);
  ASSERT_NE(leg, nullptr);
  ASSERT_EQ(leg->action, rocjitsu::Action::Lower);

  rocjitsu::cdna4::FlatMachineInst flat{};
  flat.offset = 0x456;
  flat.seg = 0;
  flat.sc0 = 1;
  flat.nt = 1;
  flat.op = 20;
  flat.sc1 = 1;
  flat.encoding = 0x3F;
  flat.addr = 8;
  flat.data = 9;
  flat.saddr = 0x7F;
  flat.vdst = 10;
  auto words = pack_64(flat);
  auto translated = rocjitsu::cdna4_to_rdna3::translate_encoding_cdna4_to_rdna3(
      rocjitsu::kEnc_FLAT, words[0], words[1], 0, leg->target_opcode);
  ASSERT_EQ(translated.word_count, 2);

  rocjitsu::rdna3::FlatMachineInst flat_dst{};
  std::memcpy(&flat_dst, translated.words, sizeof(flat_dst));
  EXPECT_EQ(flat_dst.offset, flat.offset);
  EXPECT_EQ(flat_dst.seg, 0u);
  EXPECT_EQ(flat_dst.glc, 1u);
  EXPECT_EQ(flat_dst.slc, 1u);
  EXPECT_EQ(flat_dst.dlc, 1u);
  // Generic FLAT uses the scalar null remap when saddr carries CDNA's sentinel.
  EXPECT_EQ(flat_dst.saddr, 0x7Cu);
  EXPECT_EQ(flat_dst.addr, flat.addr);
  EXPECT_EQ(flat_dst.data, flat.data);
  EXPECT_EQ(flat_dst.vdst, flat.vdst);
  EXPECT_NE(decode_one_as(ROCJITSU_CODE_ARCH_RDNA3, std::span<const uint32_t>(translated.words, 2)),
            nullptr);

  rocjitsu::cdna4::FlatScratchMachineInst scratch{};
  scratch.offset = 0x1ABC;
  scratch.sve = 1;
  scratch.seg = 1;
  scratch.sc0 = 1;
  scratch.nt = 0;
  scratch.op = 20;
  scratch.sc1 = 1;
  scratch.encoding = 0x3F;
  scratch.addr = 11;
  scratch.data = 12;
  scratch.saddr = 13;
  scratch.vdst = 14;
  words = pack_64(scratch);
  translated = rocjitsu::cdna4_to_rdna3::translate_encoding_cdna4_to_rdna3(
      rocjitsu::kEnc_FLAT, words[0], words[1], 0, leg->target_opcode);
  ASSERT_EQ(translated.word_count, 2);

  rocjitsu::rdna3::FlatScratchMachineInst scratch_dst{};
  std::memcpy(&scratch_dst, translated.words, sizeof(scratch_dst));
  EXPECT_EQ(scratch_dst.offset, scratch.offset);
  EXPECT_EQ(scratch_dst.sve, 1u);
  EXPECT_EQ(scratch_dst.seg, 1u);
  EXPECT_EQ(scratch_dst.glc, 1u);
  EXPECT_EQ(scratch_dst.slc, 1u);
  // Scratch does not carry the CDNA non-temporal bit into RDNA3 dlc.
  EXPECT_EQ(scratch_dst.dlc, 0u);
  EXPECT_EQ(scratch_dst.saddr, scratch.saddr);
  EXPECT_NE(decode_one_as(ROCJITSU_CODE_ARCH_RDNA3, std::span<const uint32_t>(translated.words, 2)),
            nullptr);

  rocjitsu::cdna4::FlatGlblMachineInst global{};
  global.offset = 0x1FED;
  global.sve = 1;
  global.seg = 2;
  global.sc0 = 0;
  global.nt = 1;
  global.op = 20;
  global.sc1 = 1;
  global.encoding = 0x3F;
  global.addr = 15;
  global.data = 16;
  global.saddr = 17;
  global.vdst = 18;
  words = pack_64(global);
  translated = rocjitsu::cdna4_to_rdna3::translate_encoding_cdna4_to_rdna3(
      rocjitsu::kEnc_FLAT, words[0], words[1], 0, leg->target_opcode);
  ASSERT_EQ(translated.word_count, 2);

  rocjitsu::rdna3::FlatGlobalMachineInst global_dst{};
  std::memcpy(&global_dst, translated.words, sizeof(global_dst));
  EXPECT_EQ(global_dst.offset, global.offset);
  EXPECT_EQ(global_dst.sve, 1u);
  EXPECT_EQ(global_dst.seg, 2u);
  EXPECT_EQ(global_dst.glc, 0u);
  EXPECT_EQ(global_dst.slc, 1u);
  EXPECT_EQ(global_dst.dlc, 1u);
  EXPECT_EQ(global_dst.saddr, global.saddr);
  EXPECT_NE(decode_one_as(ROCJITSU_CODE_ARCH_RDNA3, std::span<const uint32_t>(translated.words, 2)),
            nullptr);
}

TEST(Cdna4ToRdna3MemoryFamilies, FlatSourceOnlyDomainBitsRequireResidualHandling) {
  const auto *leg = lookup_cdna4_to_rdna3(rocjitsu::kEnc_FLAT, 20);
  ASSERT_NE(leg, nullptr);

  // Source-only FLAT domain bits cannot be silently dropped.
  rocjitsu::cdna4::FlatMachineInst flat{};
  flat.op = 20;
  flat.lds = 1;
  flat.encoding = 0x3F;
  auto words = pack_64(flat);
  auto translated = rocjitsu::cdna4_to_rdna3::translate_encoding_cdna4_to_rdna3(
      rocjitsu::kEnc_FLAT, words[0], words[1], 0, leg->target_opcode);
  EXPECT_EQ(translated.word_count, 0);

  flat.lds = 0;
  flat.acc = 1;
  words = pack_64(flat);
  translated = rocjitsu::cdna4_to_rdna3::translate_encoding_cdna4_to_rdna3(
      rocjitsu::kEnc_FLAT, words[0], words[1], 0, leg->target_opcode);
  EXPECT_EQ(translated.word_count, 0);

  rocjitsu::cdna4::FlatGlblMachineInst global{};
  global.op = 20;
  global.seg = 2;
  global.acc = 1;
  global.encoding = 0x3F;
  words = pack_64(global);
  translated = rocjitsu::cdna4_to_rdna3::translate_encoding_cdna4_to_rdna3(
      rocjitsu::kEnc_FLAT, words[0], words[1], 0, leg->target_opcode);
  EXPECT_EQ(translated.word_count, 0);

  rocjitsu::cdna4::FlatScratchMachineInst scratch{};
  scratch.op = 20;
  scratch.seg = 1;
  scratch.acc = 1;
  scratch.encoding = 0x3F;
  words = pack_64(scratch);
  translated = rocjitsu::cdna4_to_rdna3::translate_encoding_cdna4_to_rdna3(
      rocjitsu::kEnc_FLAT, words[0], words[1], 0, leg->target_opcode);
  EXPECT_EQ(translated.word_count, 0);
}

TEST(Cdna4ToRdna3MemoryFamilies, DsAtomicPreservesOperandsAndDomain) {
  rocjitsu::cdna4::DsMachineInst src{};
  src.offset0 = 3;
  src.offset1 = 4;
  src.gds = 1;
  src.op = 32; // ds_add_rtn_u32
  src.acc = 0;
  src.encoding = 0x36;
  src.addr = 8;
  src.data0 = 9;
  src.data1 = 10;
  src.vdst = 11;
  const auto words = pack_64(src);

  const auto *leg = lookup_cdna4_to_rdna3(rocjitsu::kEnc_DS, src.op);
  ASSERT_NE(leg, nullptr);
  ASSERT_EQ(leg->action, rocjitsu::Action::Lower);

  const auto translated = rocjitsu::cdna4_to_rdna3::translate_encoding_cdna4_to_rdna3(
      rocjitsu::kEnc_DS, words[0], words[1], 0, leg->target_opcode);
  ASSERT_EQ(translated.word_count, 2);

  rocjitsu::rdna3::DsMachineInst dst{};
  std::memcpy(&dst, translated.words, sizeof(dst));
  EXPECT_EQ(dst.offset0, src.offset0);
  EXPECT_EQ(dst.offset1, src.offset1);
  EXPECT_EQ(dst.gds, src.gds);
  EXPECT_EQ(dst.op, leg->target_opcode);
  EXPECT_EQ(dst.addr, src.addr);
  EXPECT_EQ(dst.data0, src.data0);
  EXPECT_EQ(dst.data1, src.data1);
  EXPECT_EQ(dst.vdst, src.vdst);

  auto dst_inst =
      decode_one_as(ROCJITSU_CODE_ARCH_RDNA3, std::span<const uint32_t>(translated.words, 2));
  ASSERT_NE(dst_inst, nullptr);
  EXPECT_EQ(std::string_view(dst_inst->mnemonic()), "ds_add_rtn_u32");
}

TEST(Cdna4ToRdna3MemoryFamilies, DsAccRequiresResidualHandling) {
  rocjitsu::cdna4::DsMachineInst src{};
  src.op = 32;
  src.acc = 1;
  src.encoding = 0x36;
  const auto words = pack_64(src);

  // DS acc selects AccVGPR state, which this same-size path cannot encode.
  const auto *leg = lookup_cdna4_to_rdna3(rocjitsu::kEnc_DS, src.op);
  ASSERT_NE(leg, nullptr);
  const auto translated = rocjitsu::cdna4_to_rdna3::translate_encoding_cdna4_to_rdna3(
      rocjitsu::kEnc_DS, words[0], words[1], 0, leg->target_opcode);
  EXPECT_EQ(translated.word_count, 0);
}

TEST(Cdna4ToRdna3InPlaceBuckets, IdentitySoppNopDecodesAsRdna3) {
  const std::array<uint32_t, 1> source{make_cdna4_sopp(0, 0)};
  auto inst = decode_cdna4(source);
  ASSERT_NE(inst, nullptr);
  ASSERT_EQ(std::string_view(inst->mnemonic()), "s_nop");

  const auto *leg =
      rocjitsu::lookup(rocjitsu::kLegalization_cdna4_to_rdna3, inst->encoding_id(), inst->opcode());
  ASSERT_NE(leg, nullptr);
  ASSERT_EQ(leg->action, rocjitsu::Action::Identity);

  const auto translated = rocjitsu::cdna4_to_rdna3::translate_encoding_cdna4_to_rdna3(
      inst->encoding_id(), source[0], 0, 0, leg->target_opcode);
  ASSERT_EQ(translated.word_count, 1);

  auto rdna3_inst =
      decode_one_as(ROCJITSU_CODE_ARCH_RDNA3, std::span<const uint32_t>(translated.words, 1));
  ASSERT_NE(rdna3_inst, nullptr);
  EXPECT_EQ(std::string_view(rdna3_inst->mnemonic()), "s_nop");
}

TEST(Cdna4ToRdna3InPlaceBuckets, SubstituteSoppBranchDecodesAsRdna3) {
  const std::array<uint32_t, 1> source{make_cdna4_sopp(2, 7)};
  auto inst = decode_cdna4(source);
  ASSERT_NE(inst, nullptr);
  ASSERT_EQ(std::string_view(inst->mnemonic()), "s_branch");

  const auto *leg =
      rocjitsu::lookup(rocjitsu::kLegalization_cdna4_to_rdna3, inst->encoding_id(), inst->opcode());
  ASSERT_NE(leg, nullptr);
  ASSERT_EQ(leg->action, rocjitsu::Action::Substitute);
  ASSERT_EQ(leg->target_opcode, 32);

  const auto translated = rocjitsu::cdna4_to_rdna3::translate_encoding_cdna4_to_rdna3(
      inst->encoding_id(), source[0], 0, 0, leg->target_opcode);
  ASSERT_EQ(translated.word_count, 1);

  auto dst = std::bit_cast<rocjitsu::rdna3::SoppMachineInst>(translated.words[0]);
  EXPECT_EQ(dst.simm16, 7);
  EXPECT_EQ(dst.op, 32);

  auto rdna3_inst =
      decode_one_as(ROCJITSU_CODE_ARCH_RDNA3, std::span<const uint32_t>(translated.words, 1));
  ASSERT_NE(rdna3_inst, nullptr);
  EXPECT_EQ(std::string_view(rdna3_inst->mnemonic()), "s_branch");
}

TEST(Cdna4ToRdna3InPlaceBuckets, SubstituteVop2DecodesAsRdna3) {
  const std::array<uint32_t, 1> source{make_cdna4_v_add_f32(7, 9, 10)};
  auto inst = decode_cdna4(source);
  ASSERT_NE(inst, nullptr);
  ASSERT_EQ(std::string_view(inst->mnemonic()), "v_add_f32_e32");

  const auto *leg =
      rocjitsu::lookup(rocjitsu::kLegalization_cdna4_to_rdna3, inst->encoding_id(), inst->opcode());
  ASSERT_NE(leg, nullptr);
  ASSERT_EQ(leg->action, rocjitsu::Action::Substitute);
  ASSERT_EQ(leg->target_opcode, 3);

  const auto translated = rocjitsu::cdna4_to_rdna3::translate_encoding_cdna4_to_rdna3(
      inst->encoding_id(), source[0], 0, 0, leg->target_opcode);
  ASSERT_EQ(translated.word_count, 1);

  auto dst = std::bit_cast<rocjitsu::rdna3::Vop2MachineInst>(translated.words[0]);
  EXPECT_EQ(dst.src0, 10);
  EXPECT_EQ(dst.vsrc1, 9);
  EXPECT_EQ(dst.vdst, 7);
  EXPECT_EQ(dst.op, 3);

  auto rdna3_inst =
      decode_one_as(ROCJITSU_CODE_ARCH_RDNA3, std::span<const uint32_t>(translated.words, 1));
  ASSERT_NE(rdna3_inst, nullptr);
  EXPECT_EQ(std::string_view(rdna3_inst->mnemonic()), "v_add_f32_e32");
}

TEST(Cdna4ToRdna3InPlaceBuckets, LowerVop2PreservesTargetOpcode) {
  const std::array<uint32_t, 1> source{make_cdna4_v_lshrrev_b32(7, 9, 10)};
  auto inst = decode_cdna4(source);
  ASSERT_NE(inst, nullptr);
  ASSERT_EQ(std::string_view(inst->mnemonic()), "v_lshrrev_b32_e32");

  const auto *leg =
      rocjitsu::lookup(rocjitsu::kLegalization_cdna4_to_rdna3, inst->encoding_id(), inst->opcode());
  ASSERT_NE(leg, nullptr);
  ASSERT_EQ(leg->action, rocjitsu::Action::Lower);
  ASSERT_EQ(leg->target_opcode, 25);

  const auto translated = rocjitsu::cdna4_to_rdna3::translate_encoding_cdna4_to_rdna3(
      inst->encoding_id(), source[0], 0, 0, leg->target_opcode);
  ASSERT_EQ(translated.word_count, 1);

  auto dst = std::bit_cast<rocjitsu::rdna3::Vop2MachineInst>(translated.words[0]);
  EXPECT_EQ(dst.src0, 10);
  EXPECT_EQ(dst.vsrc1, 9);
  EXPECT_EQ(dst.vdst, 7);
  EXPECT_EQ(dst.op, 25);

  auto rdna3_inst =
      decode_one_as(ROCJITSU_CODE_ARCH_RDNA3, std::span<const uint32_t>(translated.words, 1));
  ASSERT_NE(rdna3_inst, nullptr);
  EXPECT_EQ(std::string_view(rdna3_inst->mnemonic()), "v_lshrrev_b32_e32");
}

TEST(Cdna4ToRdna3InPlaceBuckets, GeneratedSameSizeSop1PreservesOperands) {
  const std::array<uint32_t, 1> source{make_cdna4_s_mov_b32(17, 42)};
  auto inst = decode_cdna4(source);
  ASSERT_NE(inst, nullptr);
  ASSERT_EQ(std::string_view(inst->mnemonic()), "s_mov_b32");

  const auto *leg =
      rocjitsu::lookup(rocjitsu::kLegalization_cdna4_to_rdna3, inst->encoding_id(), inst->opcode());
  ASSERT_NE(leg, nullptr);
  ASSERT_NE(leg->action, rocjitsu::Action::Expand);
  ASSERT_EQ(leg->target_opcode, 0);

  const auto translated = rocjitsu::cdna4_to_rdna3::translate_encoding_cdna4_to_rdna3(
      inst->encoding_id(), source[0], 0, 0, leg->target_opcode);
  ASSERT_EQ(translated.word_count, 1);

  auto dst = std::bit_cast<rocjitsu::rdna3::Sop1MachineInst>(translated.words[0]);
  EXPECT_EQ(dst.ssrc0, 42);
  EXPECT_EQ(dst.sdst, 17);
  EXPECT_EQ(dst.op, 0);

  auto rdna3_inst =
      decode_one_as(ROCJITSU_CODE_ARCH_RDNA3, std::span<const uint32_t>(translated.words, 1));
  ASSERT_NE(rdna3_inst, nullptr);
  EXPECT_EQ(std::string_view(rdna3_inst->mnemonic()), "s_mov_b32");
}

TEST(Cdna4ToRdna3InPlaceBuckets, TrailingLiteralIsPreservedForSubstituteVop2) {
  constexpr uint32_t kLiteral = 0x3F800000;
  const std::array<uint32_t, 2> source{make_cdna4_v_add_f32(7, 9, 255), kLiteral};
  auto inst = decode_cdna4(source);
  ASSERT_NE(inst, nullptr);
  ASSERT_EQ(std::string_view(inst->mnemonic()), "v_add_f32_e32");
  ASSERT_EQ(inst->size(), 8);

  const auto *leg =
      rocjitsu::lookup(rocjitsu::kLegalization_cdna4_to_rdna3, inst->encoding_id(), inst->opcode());
  ASSERT_NE(leg, nullptr);
  ASSERT_EQ(leg->action, rocjitsu::Action::Substitute);

  const auto translated = rocjitsu::cdna4_to_rdna3::translate_encoding_cdna4_to_rdna3(
      inst->encoding_id(), source[0], source[1], 0, leg->target_opcode);
  ASSERT_EQ(translated.word_count, 1);

  const auto translated_words = append_trailing_literal_for_test(
      translated, std::span<const uint32_t>(source), inst->size() / sizeof(uint32_t));
  ASSERT_EQ(translated_words.size(), 2u);
  EXPECT_EQ(translated_words[1], kLiteral);

  const auto stats =
      decode_words_as(ROCJITSU_CODE_ARCH_RDNA3, std::span<const uint32_t>(translated_words));
  EXPECT_EQ(stats.decode_failures, 0u);
  EXPECT_EQ(stats.inst_count, 1u);
}

TEST(Cdna4ToRdna3InPlaceBuckets, LowerMtbufD16DecodesAsRenamedRdna3) {
  const auto source = make_cdna4_tbuffer_load_format_d16_x();
  auto inst = decode_cdna4(source);
  ASSERT_NE(inst, nullptr);
  ASSERT_EQ(std::string_view(inst->mnemonic()), "tbuffer_load_format_d16_x");

  const auto *leg =
      rocjitsu::lookup(rocjitsu::kLegalization_cdna4_to_rdna3, inst->encoding_id(), inst->opcode());
  ASSERT_NE(leg, nullptr);
  ASSERT_EQ(leg->action, rocjitsu::Action::Lower);
  ASSERT_EQ(leg->target_opcode, 8);

  const auto translated = rocjitsu::cdna4_to_rdna3::translate_encoding_cdna4_to_rdna3(
      inst->encoding_id(), source[0], source[1], 0, leg->target_opcode);
  ASSERT_EQ(translated.word_count, 2);

  rocjitsu::rdna3::MtbufMachineInst dst{};
  std::memcpy(&dst, translated.words, sizeof(dst));
  EXPECT_EQ(dst.op, 8);
  EXPECT_EQ(dst.format, 0x74);

  auto rdna3_inst =
      decode_one_as(ROCJITSU_CODE_ARCH_RDNA3, std::span<const uint32_t>(translated.words, 2));
  ASSERT_NE(rdna3_inst, nullptr);
  EXPECT_EQ(std::string_view(rdna3_inst->mnemonic()), "tbuffer_load_d16_format_x");
}

TEST(Cdna4ToRdna3SemanticTranslator, SWaitcntConvertsGfx9ToConservativeGfx11Layout) {
  const std::array<uint32_t, 1> source{make_cdna4_s_waitcnt(0x4342)};
  SemanticInstructionContext context(source);
  ASSERT_TRUE(context.is_valid());
  const auto *inst = context.first_instruction();
  ASSERT_NE(inst, nullptr);
  ASSERT_EQ(std::string_view(inst->mnemonic()), "s_waitcnt");

  SemanticTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA3,
                                lookup_cdna4_to_rdna3);
  auto replacement = translator.try_lower_expand(*inst, 0, context.live());

  ASSERT_EQ(replacement.size(), 1u);
  EXPECT_EQ(replacement[0], rocjitsu::pack_sopp(9, 0x0034));

  const auto stats =
      decode_words_as(ROCJITSU_CODE_ARCH_RDNA3, std::span<const uint32_t>(replacement));
  EXPECT_EQ(stats.decode_failures, 0u);
  EXPECT_EQ(stats.inst_count, 1u);
}

TEST(Cdna4ToRdna3SemanticTranslator, VLshlAddU64ZeroShiftOmitsRdna4WaitAlu) {
  const auto source = make_cdna4_v_lshl_add_u64_zero_shift();
  SemanticInstructionContext context(source);
  ASSERT_TRUE(context.is_valid());
  const auto *inst = context.first_instruction();
  ASSERT_NE(inst, nullptr);
  ASSERT_EQ(std::string_view(inst->mnemonic()), "v_lshl_add_u64");

  SemanticTranslator rdna3(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA3,
                           lookup_cdna4_to_rdna3);
  auto rdna3_replacement = rdna3.try_lower_expand(*inst, 0, context.live());

  constexpr uint32_t kRdna4WaitAlu = rocjitsu::pack_sopp(8, 0xFFFD);
  ASSERT_EQ(rdna3_replacement.size(), 4u);
  EXPECT_EQ(std::find(rdna3_replacement.begin(), rdna3_replacement.end(), kRdna4WaitAlu),
            rdna3_replacement.end());

  const auto rdna3_stats =
      decode_words_as(ROCJITSU_CODE_ARCH_RDNA3, std::span<const uint32_t>(rdna3_replacement));
  EXPECT_EQ(rdna3_stats.decode_failures, 0u);
  EXPECT_EQ(rdna3_stats.inst_count, 2u);

  SemanticTranslator rdna4(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA4);
  auto rdna4_replacement = rdna4.try_lower_expand(*inst, 0, context.live());
  ASSERT_EQ(rdna4_replacement.size(), 5u);
  EXPECT_NE(std::find(rdna4_replacement.begin(), rdna4_replacement.end(), kRdna4WaitAlu),
            rdna4_replacement.end());
}

TEST(Cdna4ToRdna3SemanticTranslator, RemovedVopcConstantComparisonsLowerToScalarMasks) {
  constexpr uint8_t kVccLo = 106;
  constexpr uint8_t kExecLo = 126;
  constexpr uint16_t kInlineConst0 = 128;

  SemanticTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA3,
                                lookup_cdna4_to_rdna3);

  const std::array<uint32_t, 1> false_source{make_cdna4_vopc_cmp(32)};
  SemanticInstructionContext false_context(false_source);
  ASSERT_TRUE(false_context.is_valid());
  const auto *false_inst = false_context.first_instruction();
  ASSERT_NE(false_inst, nullptr);
  ASSERT_EQ(std::string_view(false_inst->mnemonic()), "v_cmp_f_f16_e32");
  auto false_replacement = translator.try_lower_expand(*false_inst, 0, false_context.live());
  ASSERT_EQ(false_replacement.size(), 1u);
  expect_rdna3_s_mov_b64(false_replacement[0], kVccLo, kInlineConst0);

  const std::array<uint32_t, 1> true_source{make_cdna4_vopc_cmp(47)};
  SemanticInstructionContext true_context(true_source);
  ASSERT_TRUE(true_context.is_valid());
  const auto *true_inst = true_context.first_instruction();
  ASSERT_NE(true_inst, nullptr);
  ASSERT_EQ(std::string_view(true_inst->mnemonic()), "v_cmp_tru_f16_e32");
  auto true_replacement = translator.try_lower_expand(*true_inst, 0, true_context.live());
  ASSERT_EQ(true_replacement.size(), 1u);
  expect_rdna3_s_mov_b64(true_replacement[0], kVccLo, kExecLo);

  const std::array<uint32_t, 1> cmpx_false_source{make_cdna4_vopc_cmp(176)};
  SemanticInstructionContext cmpx_false_context(cmpx_false_source);
  ASSERT_TRUE(cmpx_false_context.is_valid());
  const auto *cmpx_false_inst = cmpx_false_context.first_instruction();
  ASSERT_NE(cmpx_false_inst, nullptr);
  ASSERT_EQ(std::string_view(cmpx_false_inst->mnemonic()), "v_cmpx_f_i16_e32");
  auto cmpx_false_replacement =
      translator.try_lower_expand(*cmpx_false_inst, 0, cmpx_false_context.live());
  ASSERT_EQ(cmpx_false_replacement.size(), 2u);
  expect_rdna3_s_mov_b64(cmpx_false_replacement[0], kVccLo, kInlineConst0);
  expect_rdna3_s_mov_b64(cmpx_false_replacement[1], kExecLo, kInlineConst0);

  const std::array<uint32_t, 1> substituted_source{make_cdna4_vopc_cmp(64)};
  SemanticInstructionContext substituted_context(substituted_source);
  ASSERT_TRUE(substituted_context.is_valid());
  const auto *substituted_inst = substituted_context.first_instruction();
  ASSERT_NE(substituted_inst, nullptr);
  ASSERT_EQ(std::string_view(substituted_inst->mnemonic()), "v_cmp_f_f32_e32");
  EXPECT_TRUE(
      translator.try_lower_expand(*substituted_inst, 0, substituted_context.live()).empty());
}

TEST(Cdna4ToRdna3SemanticTranslator, RemovedVop3ConstantComparisonsUseExplicitSdst) {
  constexpr uint8_t kExecLo = 126;
  constexpr uint16_t kInlineConst0 = 128;

  SemanticTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA3,
                                lookup_cdna4_to_rdna3);

  const auto false_source = make_cdna4_vop3_cmp(32, 18);
  SemanticInstructionContext false_context(false_source);
  ASSERT_TRUE(false_context.is_valid());
  const auto *false_inst = false_context.first_instruction();
  ASSERT_NE(false_inst, nullptr);
  ASSERT_EQ(std::string_view(false_inst->mnemonic()), "v_cmp_f_f16");
  auto false_replacement = translator.try_lower_expand(*false_inst, 0, false_context.live());
  ASSERT_EQ(false_replacement.size(), 1u);
  expect_rdna3_s_mov_b64(false_replacement[0], 18, kInlineConst0);

  const auto true_source = make_cdna4_vop3_cmp(47, 20);
  SemanticInstructionContext true_context(true_source);
  ASSERT_TRUE(true_context.is_valid());
  const auto *true_inst = true_context.first_instruction();
  ASSERT_NE(true_inst, nullptr);
  ASSERT_EQ(std::string_view(true_inst->mnemonic()), "v_cmp_tru_f16");
  auto true_replacement = translator.try_lower_expand(*true_inst, 0, true_context.live());
  ASSERT_EQ(true_replacement.size(), 1u);
  expect_rdna3_s_mov_b64(true_replacement[0], 20, kExecLo);

  const auto cmpx_false_source = make_cdna4_vop3_cmp(176, 22);
  SemanticInstructionContext cmpx_false_context(cmpx_false_source);
  ASSERT_TRUE(cmpx_false_context.is_valid());
  const auto *cmpx_false_inst = cmpx_false_context.first_instruction();
  ASSERT_NE(cmpx_false_inst, nullptr);
  ASSERT_EQ(std::string_view(cmpx_false_inst->mnemonic()), "v_cmpx_f_i16");
  auto cmpx_false_replacement =
      translator.try_lower_expand(*cmpx_false_inst, 0, cmpx_false_context.live());
  ASSERT_EQ(cmpx_false_replacement.size(), 2u);
  expect_rdna3_s_mov_b64(cmpx_false_replacement[0], 22, kInlineConst0);
  expect_rdna3_s_mov_b64(cmpx_false_replacement[1], kExecLo, kInlineConst0);

  const auto substituted_source = make_cdna4_vop3_cmp(64, 24);
  SemanticInstructionContext substituted_context(substituted_source);
  ASSERT_TRUE(substituted_context.is_valid());
  const auto *substituted_inst = substituted_context.first_instruction();
  ASSERT_NE(substituted_inst, nullptr);
  ASSERT_EQ(std::string_view(substituted_inst->mnemonic()), "v_cmp_f_f32");
  EXPECT_TRUE(
      translator.try_lower_expand(*substituted_inst, 0, substituted_context.live()).empty());
}

TEST(Cdna4ToRdna3Legalization, Vop2CarryAndU32RowsSubstituteToRdna3E32Forms) {
  struct Case {
    uint8_t source_op;
    std::string_view source_mnemonic;
    uint8_t target_op;
    std::string_view target_mnemonic;
  };

  const std::array<Case, 6> cases{{
      {28, "v_addc_co_u32_e32", 32, "v_add_co_ci_u32_e32"},
      {29, "v_subb_co_u32_e32", 33, "v_sub_co_ci_u32_e32"},
      {30, "v_subbrev_co_u32_e32", 34, "v_subrev_co_ci_u32_e32"},
      {52, "v_add_u32_e32", 37, "v_add_nc_u32_e32"},
      {53, "v_sub_u32_e32", 38, "v_sub_nc_u32_e32"},
      {54, "v_subrev_u32_e32", 39, "v_subrev_nc_u32_e32"},
  }};

  for (const auto &c : cases) {
    const std::array<uint32_t, 1> source{make_cdna4_vop2(c.source_op, 7, 9, 256 + 10)};
    auto inst = decode_cdna4(source);
    ASSERT_NE(inst, nullptr);
    ASSERT_EQ(std::string_view(inst->mnemonic()), c.source_mnemonic);
    const auto *leg = rocjitsu::lookup(rocjitsu::kLegalization_cdna4_to_rdna3, inst->encoding_id(),
                                       inst->opcode());
    ASSERT_NE(leg, nullptr);
    ASSERT_EQ(leg->action, rocjitsu::Action::Substitute);
    ASSERT_EQ(leg->target_opcode, c.target_op);

    // These same-width VOP2 rows should flow through generated encoding
    // translation, not the semantic expansion fallback.
    const auto translated = rocjitsu::cdna4_to_rdna3::translate_encoding_cdna4_to_rdna3(
        inst->encoding_id(), source[0], 0, 0, leg->target_opcode);
    ASSERT_EQ(translated.word_count, 1u) << c.source_mnemonic;
    expect_rdna3_vop2(translated.words[0], c.target_op, 7, 256 + 10, 9, c.target_mnemonic);
  }
}

TEST(Cdna4ToRdna3SemanticTranslator, ResidualVop2U16RowsExpandToRdna3Vop3) {
  struct Case {
    uint8_t source_op;
    std::string_view source_mnemonic;
    uint16_t target_op;
    std::string_view target_mnemonic;
    uint16_t expected_src0;
    uint16_t expected_src1;
  };

  const std::array<Case, 3> cases{{
      {38, "v_add_u16_e32", 771, "v_add_nc_u16", 266, 265},
      {39, "v_sub_u16_e32", 772, "v_sub_nc_u16", 266, 265},
      {40, "v_subrev_u16_e32", 772, "v_sub_nc_u16", 265, 266},
  }};

  SemanticTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA3,
                                lookup_cdna4_to_rdna3);
  for (const auto &c : cases) {
    const std::array<uint32_t, 1> source{make_cdna4_vop2(c.source_op, 6, 9, 256 + 10)};
    SemanticInstructionContext context(source);
    ASSERT_TRUE(context.is_valid());
    const auto *inst = context.first_instruction();
    ASSERT_NE(inst, nullptr);
    ASSERT_EQ(std::string_view(inst->mnemonic()), c.source_mnemonic);

    const auto replacement = translator.try_lower_expand(*inst, 0, context.live());
    ASSERT_EQ(replacement.size(), 2u) << c.source_mnemonic;
    expect_rdna3_vop3(replacement, c.target_op, 6, c.expected_src0, c.expected_src1,
                      c.target_mnemonic);
  }
}

TEST(Cdna4ToRdna3SemanticTranslator, ResidualVop3IntegerRowsRenameToRdna3NcForms) {
  struct Case {
    uint16_t source_op;
    std::string_view source_mnemonic;
    uint16_t target_op;
    std::string_view target_mnemonic;
    uint16_t expected_src0;
    uint16_t expected_src1;
  };

  const std::array<Case, 10> cases{{
      {294, "v_add_u16", 771, "v_add_nc_u16", 266, 267},
      {295, "v_sub_u16", 772, "v_sub_nc_u16", 266, 267},
      {296, "v_subrev_u16", 772, "v_sub_nc_u16", 267, 266},
      {308, "v_add_u32", 293, "v_add_nc_u32", 266, 267},
      {309, "v_sub_u32", 294, "v_sub_nc_u32", 266, 267},
      {310, "v_subrev_u32", 295, "v_subrev_nc_u32", 266, 267},
      {668, "v_add_i32", 806, "v_add_nc_i32", 266, 267},
      {669, "v_sub_i32", 805, "v_sub_nc_i32", 266, 267},
      {670, "v_add_i16", 781, "v_add_nc_i16", 266, 267},
      {671, "v_sub_i16", 782, "v_sub_nc_i16", 266, 267},
  }};

  SemanticTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA3,
                                lookup_cdna4_to_rdna3);
  for (const auto &c : cases) {
    const auto source = make_cdna4_vop3(c.source_op, 12, 266, 267);
    SemanticInstructionContext context(source);
    ASSERT_TRUE(context.is_valid());
    const auto *inst = context.first_instruction();
    ASSERT_NE(inst, nullptr);
    ASSERT_EQ(std::string_view(inst->mnemonic()), c.source_mnemonic);

    const auto replacement = translator.try_lower_expand(*inst, 0, context.live());
    ASSERT_EQ(replacement.size(), 2u) << c.source_mnemonic;
    expect_rdna3_vop3(replacement, c.target_op, 12, c.expected_src0, c.expected_src1,
                      c.target_mnemonic);
  }
}

TEST(Cdna4ToRdna3SemanticTranslator, ResidualVop3SdstCarryRowsRenameToRdna3CoCiForms) {
  struct Case {
    uint16_t source_op;
    std::string_view source_mnemonic;
    uint16_t target_op;
    std::string_view target_mnemonic;
  };

  const std::array<Case, 3> cases{{
      {284, "v_addc_co_u32", 288, "v_add_co_ci_u32"},
      {285, "v_subb_co_u32", 289, "v_sub_co_ci_u32"},
      {286, "v_subbrev_co_u32", 290, "v_subrev_co_ci_u32"},
  }};

  SemanticTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA3,
                                lookup_cdna4_to_rdna3);
  for (const auto &c : cases) {
    const auto source = make_cdna4_vop3_sdst(c.source_op, 13, 106, 266, 267, 106);
    SemanticInstructionContext context(source);
    ASSERT_TRUE(context.is_valid());
    const auto *inst = context.first_instruction();
    ASSERT_NE(inst, nullptr);
    ASSERT_EQ(std::string_view(inst->mnemonic()), c.source_mnemonic);

    const auto replacement = translator.try_lower_expand(*inst, 0, context.live());
    ASSERT_EQ(replacement.size(), 2u) << c.source_mnemonic;
    expect_rdna3_vop3_sdst(replacement, c.target_op, 13, 106, 266, 267, 106, c.target_mnemonic);
  }
}

TEST(Cdna4ToRdna3SemanticTranslator, ResidualVMovB64LowersToOrderedB32Moves) {
  SemanticTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA3,
                                lookup_cdna4_to_rdna3);

  const std::array<uint32_t, 1> e32_source{make_cdna4_v_mov_b64_e32(12, 10)};
  SemanticInstructionContext e32_context(e32_source);
  ASSERT_TRUE(e32_context.is_valid());
  const auto *e32_inst = e32_context.first_instruction();
  ASSERT_NE(e32_inst, nullptr);
  ASSERT_EQ(std::string_view(e32_inst->mnemonic()), "v_mov_b64_e32");
  const auto e32_replacement = translator.try_lower_expand(*e32_inst, 0, e32_context.live());
  ASSERT_EQ(e32_replacement.size(), 2u);
  expect_rdna3_v_mov_b32(e32_replacement[0], 12, 266);
  expect_rdna3_v_mov_b32(e32_replacement[1], 13, 267);

  const auto overlap_source = make_cdna4_vop3(376, 11, 266, 0);
  SemanticInstructionContext overlap_context(overlap_source);
  ASSERT_TRUE(overlap_context.is_valid());
  const auto *overlap_inst = overlap_context.first_instruction();
  ASSERT_NE(overlap_inst, nullptr);
  ASSERT_EQ(std::string_view(overlap_inst->mnemonic()), "v_mov_b64");
  const auto overlap_replacement =
      translator.try_lower_expand(*overlap_inst, 0, overlap_context.live());
  ASSERT_EQ(overlap_replacement.size(), 2u);
  expect_rdna3_v_mov_b32(overlap_replacement[0], 12, 267);
  expect_rdna3_v_mov_b32(overlap_replacement[1], 11, 266);
}

TEST(Cdna4ToRdna3SemanticTranslator, ResidualVop3ModifiedRowsRemainFailClosed) {
  auto source = make_cdna4_vop3(308, 12, 266, 267);
  rocjitsu::cdna4::Vop3MachineInst src{};
  std::memcpy(&src, source.data(), sizeof(src));
  src.clamp = 1;
  std::memcpy(source.data(), &src, sizeof(src));

  SemanticInstructionContext context(source);
  ASSERT_TRUE(context.is_valid());
  const auto *inst = context.first_instruction();
  ASSERT_NE(inst, nullptr);
  ASSERT_EQ(std::string_view(inst->mnemonic()), "v_add_u32");

  SemanticTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA3,
                                lookup_cdna4_to_rdna3);
  EXPECT_TRUE(translator.try_lower_expand(*inst, 0, context.live()).empty());
  expect_unsupported_expansion_fails_closed(source, "v_add_u32", 308,
                                            "residual non-matrix expansion");
}

TEST(BinaryTranslatorExpansion, ResidualVop2U16ExpansionUsesTrailingNopCaveAndBranchStub) {
  const std::array<uint32_t, 1> expansion_source{make_cdna4_vop2(38, 6, 9, 256 + 10)};
  const std::vector<uint32_t> source_words{
      expansion_source[0],
      make_cdna4_sopp(1, 0), // s_endpgm
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),
  };

  const auto elf = make_minimal_amdgpu_elf(source_words);
  rocjitsu::AmdGpuCodeObject co(elf.data(), elf.size());
  ASSERT_TRUE(co.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA3);
  const auto result = translator.translate(co);
  EXPECT_TRUE(result.warnings.empty()) << (result.warnings.empty() ? "" : result.warnings.front());
  ASSERT_FALSE(result.elf_bytes.empty());

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const auto words = first_text_words(translated);
  ASSERT_GE(words.size(), 5u);

  SemanticInstructionContext expansion_context(expansion_source);
  ASSERT_TRUE(expansion_context.is_valid());
  const auto *inst = expansion_context.first_instruction();
  ASSERT_NE(inst, nullptr);
  SemanticTranslator semantic(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA3,
                              lookup_cdna4_to_rdna3);
  const auto expected_expansion = semantic.try_lower_expand(*inst, 0, expansion_context.live());
  ASSERT_EQ(expected_expansion.size(), 2u);

  EXPECT_EQ(words[0], rocjitsu::build_s_branch(1, ROCJITSU_CODE_ARCH_RDNA3));
  EXPECT_TRUE(std::equal(expected_expansion.begin(), expected_expansion.end(), words.begin() + 2));
  EXPECT_EQ(words[4], rocjitsu::build_s_branch(-4, ROCJITSU_CODE_ARCH_RDNA3));
}

TEST(BinaryTranslatorExpansion, Rdna3ExpansionUsesTrailingNopCaveAndBranchStub) {
  const auto expansion_source = make_cdna4_v_lshl_add_u64_zero_shift();
  const std::vector<uint32_t> source_words{
      expansion_source[0],
      expansion_source[1],
      make_cdna4_sopp(1, 0), // s_endpgm
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),
  };

  const auto elf = make_minimal_amdgpu_elf(source_words);
  rocjitsu::AmdGpuCodeObject co(elf.data(), elf.size());
  ASSERT_TRUE(co.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA3);
  const auto result = translator.translate(co);
  EXPECT_TRUE(result.warnings.empty()) << (result.warnings.empty() ? "" : result.warnings.front());
  ASSERT_FALSE(result.elf_bytes.empty());

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const auto words = first_text_words(translated);
  ASSERT_GE(words.size(), source_words.size());

  EXPECT_EQ(words[0], rocjitsu::build_s_branch(2, ROCJITSU_CODE_ARCH_RDNA3));
  EXPECT_EQ(words[1], rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA3));

  SemanticInstructionContext expected_context(source_words);
  ASSERT_TRUE(expected_context.is_valid());
  const auto *inst = expected_context.first_instruction();
  ASSERT_NE(inst, nullptr);
  SemanticTranslator semantic(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA3,
                              lookup_cdna4_to_rdna3);
  const auto expected_expansion = semantic.try_lower_expand(*inst, 0, expected_context.live());
  ASSERT_EQ(expected_expansion.size(), 4u);

  ASSERT_GE(words.size(), 8u);
  EXPECT_TRUE(std::equal(expected_expansion.begin(), expected_expansion.end(), words.begin() + 3));
  EXPECT_EQ(words[7], rocjitsu::build_s_branch(-6, ROCJITSU_CODE_ARCH_RDNA3));

  const auto stats = decode_words_as(ROCJITSU_CODE_ARCH_RDNA3, std::span<const uint32_t>(words));
  EXPECT_EQ(stats.decode_failures, 0u);
}

TEST(BinaryTranslatorExpansion, RemovedVopcCmpxUsesTrailingNopCaveAndBranchStub) {
  constexpr uint8_t kVccLo = 106;
  constexpr uint8_t kExecLo = 126;
  constexpr uint16_t kInlineConst0 = 128;

  const std::vector<uint32_t> source_words{
      make_cdna4_vopc_cmp(176),
      make_cdna4_sopp(1, 0), // s_endpgm
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),
  };

  const auto elf = make_minimal_amdgpu_elf(source_words);
  rocjitsu::AmdGpuCodeObject co(elf.data(), elf.size());
  ASSERT_TRUE(co.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA3);
  const auto result = translator.translate(co);
  EXPECT_TRUE(result.warnings.empty()) << (result.warnings.empty() ? "" : result.warnings.front());
  ASSERT_FALSE(result.elf_bytes.empty());

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const auto words = first_text_words(translated);
  ASSERT_GE(words.size(), 5u);

  EXPECT_EQ(words[0], rocjitsu::build_s_branch(1, ROCJITSU_CODE_ARCH_RDNA3));
  expect_rdna3_s_mov_b64(words[2], kVccLo, kInlineConst0);
  expect_rdna3_s_mov_b64(words[3], kExecLo, kInlineConst0);
  EXPECT_EQ(words[4], rocjitsu::build_s_branch(-4, ROCJITSU_CODE_ARCH_RDNA3));

  const auto stats = decode_words_as(ROCJITSU_CODE_ARCH_RDNA3, std::span<const uint32_t>(words));
  EXPECT_EQ(stats.decode_failures, 0u);
}

TEST(BinaryTranslatorExpansion, ExhaustedNopPaddingFailsClosed) {
  const auto expansion_source = make_cdna4_v_lshl_add_u64_zero_shift();
  const std::vector<uint32_t> source_words{
      expansion_source[0], expansion_source[1],
      make_cdna4_sopp(1, 0), // s_endpgm leaves no trailing NOP cave capacity.
  };

  const auto elf = make_minimal_amdgpu_elf(source_words);
  rocjitsu::AmdGpuCodeObject co(elf.data(), elf.size());
  ASSERT_TRUE(co.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA3);
  const auto result = translator.translate(co);
  EXPECT_TRUE(result.elf_bytes.empty());
  EXPECT_TRUE(warnings_contain(result.warnings, "code cave exhausted"));
  EXPECT_TRUE(warnings_contain(result.warnings, "v_lshl_add_u64"));
}

TEST(BinaryTranslatorExpansion, UnsupportedExpandFailsClosed) {
  const auto unsupported_source = make_cdna4_v_accvgpr_read();
  expect_unsupported_expansion_fails_closed(unsupported_source, "v_accvgpr_read", 88, "AccVGPR");

  auto unsupported_inst = decode_cdna4(unsupported_source);
  ASSERT_NE(unsupported_inst, nullptr);
  const auto *leg = rocjitsu::lookup(rocjitsu::kLegalization_cdna4_to_rdna3,
                                     unsupported_inst->encoding_id(), unsupported_inst->opcode());
  ASSERT_NE(leg, nullptr);
  ASSERT_EQ(leg->action, rocjitsu::Action::Expand);
}

TEST(BinaryTranslatorExpansion, UnsupportedAccvgprWriteFailsClosed) {
  const auto unsupported_source = make_cdna4_v_accvgpr_write();
  expect_unsupported_expansion_fails_closed(unsupported_source, "v_accvgpr_write", 89, "AccVGPR");
}

TEST(BinaryTranslatorExpansion, UnsupportedAccvgprMovFailsClosed) {
  const auto unsupported_source = make_cdna4_v_accvgpr_mov_b32_e32();
  expect_unsupported_expansion_fails_closed(unsupported_source, "v_accvgpr_mov_b32_e32", 82,
                                            "AccVGPR");
}

TEST(BinaryTranslatorExpansion, UnsupportedMfmaLowerRowsFailClosed) {
  const auto unsupported_source = make_cdna4_v_mfma_f32_16x16x16_f16();
  expect_unsupported_expansion_fails_closed(unsupported_source, "v_mfma_f32_16x16x16_f16", 77,
                                            "dense MFMA");
}

TEST(BinaryTranslatorExpansion, UnsupportedSmfmacRowsFailClosed) {
  const auto unsupported_source = make_cdna4_v_smfmac_f32_16x16x64_bf16();
  expect_unsupported_expansion_fails_closed(unsupported_source, "v_smfmac_f32_16x16x64_bf16", 57,
                                            "sparse SMFMAC");
}

TEST(BinaryTranslatorExpansion, MtbufEncodingTranslatesAndDecodesAsRdna3) {
  const auto source = make_cdna4_tbuffer_load_format_x();
  auto inst = decode_cdna4(source);
  ASSERT_NE(inst, nullptr);
  const auto *leg =
      rocjitsu::lookup(rocjitsu::kLegalization_cdna4_to_rdna3, inst->encoding_id(), inst->opcode());
  ASSERT_NE(leg, nullptr);
  ASSERT_EQ(leg->action, rocjitsu::Action::Lower);

  const std::vector<uint32_t> source_words{
      source[0],
      source[1],
      make_cdna4_sopp(1, 0),
  };

  const auto elf = make_minimal_amdgpu_elf(source_words);
  rocjitsu::AmdGpuCodeObject co(elf.data(), elf.size());
  ASSERT_TRUE(co.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA3);
  const auto result = translator.translate(co);
  EXPECT_TRUE(result.warnings.empty()) << (result.warnings.empty() ? "" : result.warnings.front());
  ASSERT_FALSE(result.elf_bytes.empty());

  rocjitsu::AmdGpuCodeObject translated(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated.is_valid());
  const auto words = first_text_words(translated);
  ASSERT_GE(words.size(), 3u);

  auto rdna3_inst =
      decode_one_as(ROCJITSU_CODE_ARCH_RDNA3, std::span<const uint32_t>(words.data(), 2));
  ASSERT_NE(rdna3_inst, nullptr);
  EXPECT_EQ(std::string_view(rdna3_inst->mnemonic()), "tbuffer_load_format_x");
}

TEST(BinaryTranslatorExpansion, MtbufAccvgprEncodingFailsClosed) {
  const auto unsupported_source = make_cdna4_tbuffer_load_format_x(true);
  expect_unsupported_encoding_fails_closed(unsupported_source, "tbuffer_load_format_x", "AccVGPR");
}

TEST(BinaryTranslatorExpansion, RelocatedCavePaddingFailsClosed) {
  const auto expansion_source = make_cdna4_v_lshl_add_u64_zero_shift();
  const std::vector<uint32_t> source_words{
      expansion_source[0],
      expansion_source[1],
      make_cdna4_sopp(1, 0),
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),
  };

  constexpr uint64_t kCaveStartOffset = 12;
  const auto elf = make_minimal_amdgpu_elf(source_words, true, kCaveStartOffset);
  rocjitsu::AmdGpuCodeObject co(elf.data(), elf.size());
  ASSERT_TRUE(co.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA3);
  const auto result = translator.translate(co);
  EXPECT_TRUE(result.elf_bytes.empty());
  EXPECT_TRUE(warnings_contain(result.warnings, "relocation target"));
  EXPECT_TRUE(warnings_contain(result.warnings, "code cave range"));
}

TEST(BinaryTranslatorExpansion, RelocatedSourceRangeFailsClosed) {
  const auto expansion_source = make_cdna4_v_lshl_add_u64_zero_shift();
  const std::vector<uint32_t> source_words{
      expansion_source[0],
      expansion_source[1],
      make_cdna4_sopp(1, 0),
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),
      rocjitsu::build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA4),
  };

  const auto elf = make_minimal_amdgpu_elf(source_words, true, 0);
  rocjitsu::AmdGpuCodeObject co(elf.data(), elf.size());
  ASSERT_TRUE(co.is_valid());

  rocjitsu::BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA3);
  const auto result = translator.translate(co);
  EXPECT_TRUE(result.elf_bytes.empty());
  EXPECT_TRUE(warnings_contain(result.warnings, "relocation target"));
  EXPECT_TRUE(warnings_contain(result.warnings, "source range"));
}

// --- End-to-end BinaryTranslator integration tests ---
#ifdef HAS_DEVICE_KERNELS

#include "rocjitsu/code/executable.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"

namespace {

std::string kernel_path(const char *name) { return std::string(KERNEL_DIR) + "/" + name + ".o"; }

std::string format_warnings(const std::vector<std::string> &warnings) {
  std::ostringstream out;
  for (const auto &warning : warnings)
    out << "\n  " << warning;
  return out.str();
}

uint32_t read_elf_flags(std::span<const uint8_t> elf_bytes) {
  uint32_t e_flags = 0;
  std::memcpy(&e_flags, elf_bytes.data() + 48, sizeof(e_flags));
  return e_flags;
}

DecodeStats decode_code_object_text(const rocjitsu::AmdGpuCodeObject &co, rj_code_arch_t arch) {
  DecodeStats total;
  for (const auto *sec : co.text_sections()) {
    const auto *data = reinterpret_cast<const uint32_t *>(sec->data());
    const size_t words = sec->size() / sizeof(uint32_t);
    const auto stats = decode_words_as(arch, std::span<const uint32_t>(data, words));
    total.inst_count += stats.inst_count;
    total.decode_failures += stats.decode_failures;
  }
  return total;
}

struct LegalizationCoverage {
  size_t decoded = 0;
  size_t decode_failures = 0;
  size_t identity = 0;
  size_t substitute = 0;
  size_t lower = 0;
};

LegalizationCoverage
collect_cdna4_to_rdna3_no_growth_coverage(const rocjitsu::AmdGpuCodeObject &co) {
  LegalizationCoverage coverage;
  auto decoder = rocjitsu::Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  if (!decoder)
    return coverage;

  for (const auto *sec : co.text_sections()) {
    const auto *data = reinterpret_cast<const uint32_t *>(sec->data());
    const size_t words = sec->size() / sizeof(uint32_t);
    size_t pc = 0;
    while (pc < words) {
      try {
        std::unique_ptr<rocjitsu::Instruction> inst(decoder->decode(&data[pc]));
        if (!inst || inst->size() <= 0) {
          ++pc;
          continue;
        }

        ++coverage.decoded;
        if (const auto *leg = rocjitsu::lookup(rocjitsu::kLegalization_cdna4_to_rdna3,
                                               inst->encoding_id(), inst->opcode())) {
          switch (leg->action) {
          case rocjitsu::Action::Identity:
            ++coverage.identity;
            break;
          case rocjitsu::Action::Substitute:
            ++coverage.substitute;
            break;
          case rocjitsu::Action::Lower:
            ++coverage.lower;
            break;
          case rocjitsu::Action::Expand:
          case rocjitsu::Action::Illegal:
            break;
          }
        }

        pc += static_cast<size_t>(inst->size()) / sizeof(uint32_t);
      } catch (const std::exception &) {
        ++coverage.decode_failures;
        ++pc;
      }
    }
  }
  return coverage;
}

} // namespace

using rocjitsu::BinaryTranslator;
using rocjitsu::Executable;

struct LoadedCdna4Kernel {
  std::unique_ptr<Executable> executable;
  const rocjitsu::AmdGpuCodeObject *code_object = nullptr;
};

LoadedCdna4Kernel load_cdna4_kernel(const char *name) {
  LoadedCdna4Kernel loaded;
  loaded.executable = std::make_unique<Executable>(kernel_path(name));
  if (!loaded.executable->is_valid())
    return loaded;
  if (loaded.executable->num_code_objects(ROCJITSU_CODE_TARGET_GFX950) == 0)
    return loaded;
  loaded.code_object = loaded.executable->code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  return loaded;
}

rocjitsu::TranslatedCodeObject translate_cdna4_to_rdna3(const rocjitsu::AmdGpuCodeObject &co) {
  BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA3);
  return translator.translate(co);
}

rocjitsu::TranslatedCodeObject translate_cdna4_to_rdna4(const rocjitsu::AmdGpuCodeObject &co) {
  BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA4);
  return translator.translate(co);
}

void expect_successful_rdna3_translation(const rocjitsu::TranslatedCodeObject &result,
                                         std::string_view kernel_name) {
  ASSERT_FALSE(result.elf_bytes.empty())
      << kernel_name << " produced empty ELF" << format_warnings(result.warnings);
  EXPECT_EQ(result.host_arch, ROCJITSU_CODE_ARCH_RDNA3);
  EXPECT_TRUE(result.warnings.empty())
      << kernel_name << " emitted translation warnings:" << format_warnings(result.warnings);

  ASSERT_GE(result.elf_bytes.size(), sizeof(rocjitsu::Elf64_Ehdr));
  const uint32_t e_flags = read_elf_flags(result.elf_bytes);
  EXPECT_EQ(e_flags & rocjitsu::EF_AMDGPU_MACH, rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX1100)
      << "ELF e_flags should contain GFX1100 machine type";

  rocjitsu::AmdGpuCodeObject translated_co(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated_co.is_valid());
  const auto stats = decode_code_object_text(translated_co, ROCJITSU_CODE_ARCH_RDNA3);
  EXPECT_GT(stats.inst_count, 0u) << "Text section should contain instructions";
  EXPECT_EQ(stats.decode_failures, 0u)
      << stats.decode_failures << " instructions failed to decode as RDNA3";
}

void expect_successful_rdna4_translation(const rocjitsu::TranslatedCodeObject &result,
                                         std::string_view kernel_name) {
  ASSERT_FALSE(result.elf_bytes.empty())
      << kernel_name << " produced empty ELF" << format_warnings(result.warnings);
  EXPECT_EQ(result.host_arch, ROCJITSU_CODE_ARCH_RDNA4);

  ASSERT_GE(result.elf_bytes.size(), sizeof(rocjitsu::Elf64_Ehdr));
  const uint32_t e_flags = read_elf_flags(result.elf_bytes);
  EXPECT_EQ(e_flags & rocjitsu::EF_AMDGPU_MACH, rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX1200)
      << "ELF e_flags should contain GFX1200 machine type";

  rocjitsu::AmdGpuCodeObject translated_co(result.elf_bytes.data(), result.elf_bytes.size());
  ASSERT_TRUE(translated_co.is_valid());
  const auto stats = decode_code_object_text(translated_co, ROCJITSU_CODE_ARCH_RDNA4);
  EXPECT_GT(stats.inst_count, 0u) << "Text section should contain instructions";
  EXPECT_EQ(stats.decode_failures, 0u)
      << stats.decode_failures << " instructions failed to decode as RDNA4";
}

TEST(BinaryTranslatorE2E, TranslateVectorAddCdna4ToRdna3) {
  auto loaded = load_cdna4_kernel("vector_add");
  ASSERT_TRUE(loaded.executable->is_valid()) << "Failed to load vector_add.o";
  ASSERT_NE(loaded.code_object, nullptr);

  const auto result = translate_cdna4_to_rdna3(*loaded.code_object);
  expect_successful_rdna3_translation(result, "vector_add");
}

TEST(BinaryTranslatorE2E, TranslateVectorAddCdna4ToRdna4) {
  auto loaded = load_cdna4_kernel("vector_add");
  ASSERT_TRUE(loaded.executable->is_valid()) << "Failed to load vector_add.o";
  ASSERT_NE(loaded.code_object, nullptr);

  const auto result = translate_cdna4_to_rdna4(*loaded.code_object);
  expect_successful_rdna4_translation(result, "vector_add");
}

TEST(BinaryTranslatorE2E, TranslateMemoryStreamCdna4ToRdna3) {
  auto loaded = load_cdna4_kernel("memory_stream");
  ASSERT_TRUE(loaded.executable->is_valid()) << "Failed to load memory_stream.o";
  ASSERT_NE(loaded.code_object, nullptr);

  const auto result = translate_cdna4_to_rdna3(*loaded.code_object);
  expect_successful_rdna3_translation(result, "memory_stream");
}

TEST(BinaryTranslatorE2E, Cdna4ToRdna3NoGrowthBucketsProduceNoWarnings) {
  auto loaded = load_cdna4_kernel("vector_add");
  ASSERT_TRUE(loaded.executable->is_valid());
  ASSERT_NE(loaded.code_object, nullptr);

  const auto coverage = collect_cdna4_to_rdna3_no_growth_coverage(*loaded.code_object);
  EXPECT_GT(coverage.decoded, 0u);
  EXPECT_EQ(coverage.decode_failures, 0u);
  EXPECT_GT(coverage.identity, 0u) << "vector_add should cover identity no-growth rows";
  EXPECT_GT(coverage.substitute, 0u) << "vector_add should cover opcode-substitute rows";
  EXPECT_GT(coverage.lower, 0u) << "vector_add should cover in-place lower rows";

  const auto result = translate_cdna4_to_rdna3(*loaded.code_object);
  EXPECT_TRUE(result.warnings.empty())
      << "CDNA4->RDNA3 no-growth coverage emitted warnings:" << format_warnings(result.warnings);
}

TEST(BinaryTranslatorE2E, NoGfx9WaitcntInRdna4Output) {
  auto loaded = load_cdna4_kernel("vector_add");
  ASSERT_TRUE(loaded.executable->is_valid());
  ASSERT_NE(loaded.code_object, nullptr);

  const auto result = translate_cdna4_to_rdna4(*loaded.code_object);
  ASSERT_FALSE(result.elf_bytes.empty());

  rocjitsu::AmdGpuCodeObject translated_co(result.elf_bytes.data(), result.elf_bytes.size());
  auto decoder = rocjitsu::Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(decoder, nullptr);

  for (const auto *sec : translated_co.text_sections()) {
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

TEST(BinaryTranslatorE2E, Rdna3TextSizeDoesNotShrink) {
  auto loaded = load_cdna4_kernel("vector_add");
  ASSERT_TRUE(loaded.executable->is_valid());
  ASSERT_NE(loaded.code_object, nullptr);

  const size_t original_text_size = loaded.code_object->text_sections().empty()
                                        ? 0
                                        : loaded.code_object->text_sections()[0]->size();

  const auto result = translate_cdna4_to_rdna3(*loaded.code_object);
  ASSERT_FALSE(result.elf_bytes.empty());

  rocjitsu::AmdGpuCodeObject translated_co(result.elf_bytes.data(), result.elf_bytes.size());
  const size_t translated_text_size =
      translated_co.text_sections().empty() ? 0 : translated_co.text_sections()[0]->size();

  // Translation patches instructions in place and may append code caves.
  EXPECT_GE(translated_text_size, original_text_size);
}

TEST(BinaryTranslatorE2E, MatrixMfma16x16FailsClosedWithUnsupportedCategory) {
  auto loaded = load_cdna4_kernel("matmul_mfma_16x16");
  ASSERT_TRUE(loaded.executable->is_valid()) << "Failed to load matmul_mfma_16x16.o";
  ASSERT_NE(loaded.code_object, nullptr);

  const auto result = translate_cdna4_to_rdna3(*loaded.code_object);
  EXPECT_TRUE(result.elf_bytes.empty())
      << "CDNA4 MFMA unexpectedly translated for RDNA3 before matrix support landed";
  EXPECT_TRUE(warnings_contain(result.warnings, "unsupported expansion"))
      << format_warnings(result.warnings);
  const bool categorized = warnings_contain(result.warnings, "dense MFMA") ||
                           warnings_contain(result.warnings, "AccVGPR") ||
                           warnings_contain(result.warnings, "sparse SMFMAC") ||
                           warnings_contain(result.warnings, "software fallback");
  EXPECT_TRUE(categorized) << "MFMA failure should point to a matrix unsupported category"
                           << format_warnings(result.warnings);
}

#endif // HAS_DEVICE_KERNELS
