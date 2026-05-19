// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file dbt_encoding_test.cpp
/// @brief CPU-only tests for generated DBT encoding and legalization tables.

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/dbt/binary_translator.h"
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
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/patch/instruction_builder.h"

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

TEST(InstructionBuilder, Sop2SetsEncodingPrefix) {
  const uint32_t word = build_s_lshl_b32(1, 2, 3, ROCJITSU_CODE_ARCH_RDNA4);
  EXPECT_EQ((word >> 30) & 0x3u, 0x2u);
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
  EXPECT_EQ(dst.soffset, 0x7C); // CDNA4 null (0x7F) -> RDNA4 null (0x7C)
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
