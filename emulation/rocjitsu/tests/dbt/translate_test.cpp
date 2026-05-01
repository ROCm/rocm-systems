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
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"

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

TEST(Cdna4ToRdna3SemanticTranslator, SWaitcntConvertsGfx9ToGfx11Layout) {
  const std::array<uint32_t, 1> source{make_cdna4_s_waitcnt(0x4342)};
  auto inst = decode_cdna4(source);
  ASSERT_NE(inst, nullptr);
  ASSERT_EQ(std::string_view(inst->mnemonic()), "s_waitcnt");

  rocjitsu::RegisterLiveness liveness;
  SemanticTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA3);
  auto replacement = translator.try_lower_expand(*inst, 0, liveness);

  ASSERT_EQ(replacement.size(), 1u);
  EXPECT_EQ(replacement[0], rocjitsu::pack_sopp(9, 0x4834));

  const auto stats =
      decode_words_as(ROCJITSU_CODE_ARCH_RDNA3, std::span<const uint32_t>(replacement));
  EXPECT_EQ(stats.decode_failures, 0u);
  EXPECT_EQ(stats.inst_count, 1u);
}

TEST(Cdna4ToRdna3SemanticTranslator, VLshlAddU64ZeroShiftOmitsRdna4WaitAlu) {
  const auto source = make_cdna4_v_lshl_add_u64_zero_shift();
  auto inst = decode_cdna4(source);
  ASSERT_NE(inst, nullptr);
  ASSERT_EQ(std::string_view(inst->mnemonic()), "v_lshl_add_u64");

  rocjitsu::RegisterLiveness liveness;
  SemanticTranslator rdna3(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA3);
  auto rdna3_replacement = rdna3.try_lower_expand(*inst, 0, liveness);

  constexpr uint32_t kRdna4WaitAlu = rocjitsu::pack_sopp(8, 0xFFFD);
  ASSERT_EQ(rdna3_replacement.size(), 4u);
  EXPECT_EQ(std::find(rdna3_replacement.begin(), rdna3_replacement.end(), kRdna4WaitAlu),
            rdna3_replacement.end());

  const auto rdna3_stats =
      decode_words_as(ROCJITSU_CODE_ARCH_RDNA3, std::span<const uint32_t>(rdna3_replacement));
  EXPECT_EQ(rdna3_stats.decode_failures, 0u);
  EXPECT_EQ(rdna3_stats.inst_count, 2u);

  SemanticTranslator rdna4(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA4);
  auto rdna4_replacement = rdna4.try_lower_expand(*inst, 0, liveness);
  ASSERT_EQ(rdna4_replacement.size(), 5u);
  EXPECT_NE(std::find(rdna4_replacement.begin(), rdna4_replacement.end(), kRdna4WaitAlu),
            rdna4_replacement.end());
}

// --- End-to-end BinaryTranslator integration tests ---
#ifdef HAS_DEVICE_KERNELS

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/dbt/binary_translator.h"
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

TEST(BinaryTranslatorE2E, TranslateVectorAddCdna4ToRdna3) {
  Executable exec(kernel_path("vector_add"));
  ASSERT_TRUE(exec.is_valid()) << "Failed to load vector_add.o";
  ASSERT_GT(exec.num_code_objects(ROCJITSU_CODE_TARGET_GFX950), 0u);

  const auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  ASSERT_NE(co, nullptr);

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA3);
  auto result = translator.translate(*co);

  EXPECT_FALSE(result.elf_bytes.empty()) << "Translation produced empty ELF";
  EXPECT_EQ(result.host_arch, ROCJITSU_CODE_ARCH_RDNA3);

  ASSERT_GE(result.elf_bytes.size(), 52u);
  const uint32_t e_flags = read_elf_flags(result.elf_bytes);
  constexpr uint32_t kEfAmdgpuMach = 0x0FF;
  constexpr uint32_t kEfAmdgpuMachGfx1100 = 0x41;
  EXPECT_EQ(e_flags & kEfAmdgpuMach, kEfAmdgpuMachGfx1100)
      << "ELF e_flags should contain GFX1100 machine type";
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
  for (const auto *sec : translated_co.text_sections()) {
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

TEST(BinaryTranslatorE2E, OutputDecodesAsValidRdna3) {
  Executable exec(kernel_path("vector_add"));
  ASSERT_TRUE(exec.is_valid());
  ASSERT_GT(exec.num_code_objects(ROCJITSU_CODE_TARGET_GFX950), 0u);

  const auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  ASSERT_NE(co, nullptr);

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA3);
  auto result = translator.translate(*co);
  ASSERT_FALSE(result.elf_bytes.empty());

  rocjitsu::AmdGpuCodeObject translated_co(result.elf_bytes.data(), result.elf_bytes.size());
  const auto stats = decode_code_object_text(translated_co, ROCJITSU_CODE_ARCH_RDNA3);
  EXPECT_GT(stats.inst_count, 0u) << "Text section should contain instructions";
  EXPECT_EQ(stats.decode_failures, 0u)
      << stats.decode_failures << " instructions failed to decode as RDNA3";
}

TEST(BinaryTranslatorE2E, Cdna4ToRdna3NoGrowthBucketsProduceNoWarnings) {
  Executable exec(kernel_path("vector_add"));
  ASSERT_TRUE(exec.is_valid());
  ASSERT_GT(exec.num_code_objects(ROCJITSU_CODE_TARGET_GFX950), 0u);

  const auto *co = exec.code_object(ROCJITSU_CODE_TARGET_GFX950, 0);
  ASSERT_NE(co, nullptr);

  const auto coverage = collect_cdna4_to_rdna3_no_growth_coverage(*co);
  EXPECT_GT(coverage.decoded, 0u);
  EXPECT_EQ(coverage.decode_failures, 0u);
  EXPECT_GT(coverage.identity, 0u) << "vector_add should cover identity no-growth rows";
  EXPECT_GT(coverage.substitute, 0u) << "vector_add should cover opcode-substitute rows";
  EXPECT_GT(coverage.lower, 0u) << "vector_add should cover in-place lower rows";

  BinaryTranslator translator(ROCJITSU_CODE_ARCH_CDNA4, ROCJITSU_CODE_ARCH_RDNA3);
  auto result = translator.translate(*co);
  EXPECT_TRUE(result.warnings.empty())
      << "CDNA4->RDNA3 no-growth coverage emitted warnings:" << format_warnings(result.warnings);
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

  // The translated .text must be at least as large as the original
  // (code caves may grow it, but individual instructions never shift).
  EXPECT_GE(translated_text_size, original_text_size);
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
  for (const auto *sec : translated.text_sections())
    dump("RDNA4 translated", reinterpret_cast<const uint8_t *>(sec->data()), sec->size(),
         ROCJITSU_CODE_ARCH_RDNA4);

  if (!result.warnings.empty()) {
    printf("\n--- Warnings (%zu) ---\n", result.warnings.size());
    for (const auto &w : result.warnings)
      printf("  %s\n", w.c_str());
  }
}

#endif // HAS_DEVICE_KERNELS
