// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file encoding_translator_test.cpp
/// @brief CPU-only tests for cross-architecture instruction encoding.

#include "rocjitsu/code/dbt/encoding_translator.h"
#include "rocjitsu/code/dbt/generated/encoding_cdna4_to_cdna3.h"
#include "rocjitsu/code/dbt/generated/encoding_cdna4_to_rdna4.h"
#include "rocjitsu/code/dbt/generated/encoding_fields.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/encodings.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/encodings.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/opcodes.h"

#include <gtest/gtest.h>

#include <bit>
#include <cstdint>
#include <cstring>

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
  src.op = cdna4::kSCmovB64Sop1;
  src.encoding = cdna4::encoding::kSop1;
  uint32_t w0 = std::bit_cast<uint32_t>(src);

  auto result =
      cdna4_to_rdna4::translate_encoding_cdna4_to_rdna4(kEnc_SOP1, w0, 0, 0, rdna4::kSBrevB64Sop1);

  ASSERT_EQ(result.word_count, 1);
  auto dst = std::bit_cast<rdna4::Sop1MachineInst>(result.words[0]);
  EXPECT_EQ(dst.ssrc0, 42);
  EXPECT_EQ(dst.sdst, 17);
  EXPECT_EQ(dst.op, rdna4::kSBrevB64Sop1);
  EXPECT_EQ(dst.encoding, rdna4::encoding::kSop1);
}

TEST(EncodingTranslator, Sop2PreservesRegisters) {
  cdna4::Sop2MachineInst src{};
  src.ssrc0 = 10;
  src.ssrc1 = 20;
  src.sdst = 30;
  src.op = cdna4::kSMinU32Sop2;
  src.encoding = 0x2;
  uint32_t w0 = std::bit_cast<uint32_t>(src);

  auto result =
      cdna4_to_rdna4::translate_encoding_cdna4_to_rdna4(kEnc_SOP2, w0, 0, 0, rdna4::kSMinU32Sop2);

  ASSERT_EQ(result.word_count, 1);
  auto dst = std::bit_cast<rdna4::Sop2MachineInst>(result.words[0]);
  EXPECT_EQ(dst.ssrc0, 10);
  EXPECT_EQ(dst.ssrc1, 20);
  EXPECT_EQ(dst.sdst, 30);
  EXPECT_EQ(dst.op, rdna4::kSMinU32Sop2);
}

TEST(EncodingTranslator, SoppPreservesSimm16) {
  cdna4::SoppMachineInst src{};
  src.simm16 = 0xABCD;
  src.op = cdna4::kSWaitcntSopp;
  src.encoding = cdna4::encoding::kSopp;
  uint32_t w0 = std::bit_cast<uint32_t>(src);

  auto result =
      cdna4_to_rdna4::translate_encoding_cdna4_to_rdna4(kEnc_SOPP, w0, 0, 0, rdna4::kSWaitcntSopp);

  ASSERT_EQ(result.word_count, 1);
  auto dst = std::bit_cast<rdna4::SoppMachineInst>(result.words[0]);
  EXPECT_EQ(dst.simm16, 0xABCD);
  EXPECT_EQ(dst.op, rdna4::kSWaitcntSopp);
}

TEST(EncodingTranslator, SmemRemapsCoherency) {
  cdna4::SmemMachineInst src{};
  src.sbase = 5;
  src.sdata = 3;
  src.glc = 1;
  src.nv = 0;
  src.op = cdna4::kSLoadDwordSmem;
  src.offset = 0x100;
  src.soffset = 0x7F;
  src.encoding = 0x3D;
  uint32_t words[2];
  std::memcpy(words, &src, sizeof(src));

  auto result = cdna4_to_rdna4::translate_encoding_cdna4_to_rdna4(kEnc_SMEM, words[0], words[1], 0,
                                                                  rdna4::kSLoadB32Smem);

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
  src.op = cdna4::kVCmpGtF64Vop3;
  src.encoding = 0x35;
  uint32_t words[2];
  std::memcpy(words, &src, sizeof(src));

  auto result = cdna4_to_rdna4::translate_encoding_cdna4_to_rdna4(kEnc_VOP3, words[0], words[1], 0,
                                                                  rdna4::kVCmpGtF64Vop3);

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

TEST(EncodingTranslator, Cdna4ToCdna3Vop2VectorAddPreservesOperands) {
  cdna4::Vop2MachineInst src{};
  src.src0 = 3;
  src.vsrc1 = 4;
  src.vdst = 5;
  src.op = cdna4::kVAddF32Vop2;
  src.encoding = 0; // GFX9-family VOP2 prefix.
  uint32_t w0 = std::bit_cast<uint32_t>(src);

  auto result =
      cdna4_to_cdna3::translate_encoding_cdna4_to_cdna3(kEnc_VOP2, w0, 0, 0, cdna3::kVAddF32Vop2);

  ASSERT_EQ(result.word_count, 1);
  auto dst = std::bit_cast<cdna3::Vop2MachineInst>(result.words[0]);
  EXPECT_EQ(dst.src0, 3);
  EXPECT_EQ(dst.vsrc1, 4);
  EXPECT_EQ(dst.vdst, 5);
  EXPECT_EQ(dst.op, cdna3::kVAddF32Vop2);
  EXPECT_EQ(dst.encoding, 0);
}

TEST(EncodingTranslator, UnknownEncodingReturnsEmpty) {
  auto result = cdna4_to_rdna4::translate_encoding_cdna4_to_rdna4(0xFFFF, 0, 0, 0, 0);
  EXPECT_EQ(result.word_count, 0);
}

TEST(EncodingTranslator, DecodeEncodeRoundTrip) {
  cdna4::Sop1MachineInst src{};
  src.ssrc0 = 55;
  src.sdst = 33;
  src.op = cdna4::kSNotB32Sop1;
  src.encoding = cdna4::encoding::kSop1;
  uint32_t w0 = std::bit_cast<uint32_t>(src);

  auto fields = cdna4_to_rdna4::decode_sop1_cdna4(w0);
  EXPECT_EQ(fields.ssrc0, 55u);
  EXPECT_EQ(fields.sdst, 33u);
  EXPECT_EQ(fields.op, cdna4::kSNotB32Sop1);

  auto result = cdna4_to_rdna4::encode_sop1_rdna4(fields, rdna4::kSBrevB32Sop1);
  ASSERT_EQ(result.word_count, 1);
  auto dst = std::bit_cast<rdna4::Sop1MachineInst>(result.words[0]);
  EXPECT_EQ(dst.ssrc0, 55);
  EXPECT_EQ(dst.sdst, 33);
  EXPECT_EQ(dst.op, rdna4::kSBrevB32Sop1);
}

} // namespace
} // namespace rocjitsu
