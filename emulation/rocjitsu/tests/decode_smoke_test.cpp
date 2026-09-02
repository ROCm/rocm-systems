// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file decode_smoke_test.cpp
/// @brief Parameterized decode smoke test for all AMDGPU ISAs.
///
/// For each ISA we encode a known-good instruction word, call the generated
/// decoder, and assert:
///   1. Decoder::create() returns a non-null decoder.
///   2. decode() returns a non-null Instruction.
///   3. The decoded mnemonic matches the expected value.
///   4. The decoded instruction size in bytes is correct.
///
/// SOPP encoding:
///   bits[31:23] = 0x17F  (SOPP opcode base)
///   bits[22:16] = op     (per-instruction opcode field, 7 bits)
///   bits[15:0]  = simm16 (16-bit signed immediate, 0 here)
///
///   0xBF800000 = s_nop    (op = 0) — all AMDGPU ISAs
///   0xBF810000 = s_endpgm (op = 1) — CDNA1/2/3/4, RDNA1/2
///                                     (GFX9/10: op=1 is s_endpgm)
///   0xBFB00000 = s_endpgm (op = 48) — RDNA3/3.5/4
///                                     (GFX11/12: op=1 is s_setkill; s_endpgm moved to op=48)

#include "decode_test_util.h"
#include "rocjitsu/analysis/def_use_chain.h"
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna1/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna1/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna1/operand.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/operand.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/vop3.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna1/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/vop3.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/vopd.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/vopd.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/operand.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/sop2.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/vop3.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/vopd.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/except.h"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <format>
#include <memory>
#include <string>
#include <vector>

namespace {

using namespace rocjitsu;

// SOPP encodings.
// s_nop is op=0 on all AMDGPU ISAs.
// s_endpgm is op=1 on CDNA1-4 and RDNA1/2 (GFX9/GFX10).
// s_endpgm is op=48 on RDNA3/3.5/4 (GFX11/GFX12) — op=1 is s_setkill there.
constexpr uint32_t S_NOP = 0xBF800000u;          ///< s_nop    (SOPP op=0,  simm16=0)
constexpr uint32_t S_ENDPGM_GFX9 = 0xBF810000u;  ///< s_endpgm (SOPP op=1,  simm16=0): CDNA/RDNA1/2
constexpr uint32_t S_ENDPGM_GFX11 = 0xBFB00000u; ///< s_endpgm (SOPP op=48, simm16=0): RDNA3/3.5/4

constexpr uint32_t make_sopp(uint32_t op, uint32_t simm16) {
  return (0x17Fu << 23) | ((op & 0x7Fu) << 16) | (simm16 & 0xFFFFu);
}

constexpr uint32_t make_vop2(uint32_t op, uint32_t vdst, uint32_t vsrc1, uint32_t src0) {
  return ((op & 0x3Fu) << 25) | ((vdst & 0xFFu) << 17) | ((vsrc1 & 0xFFu) << 9) | (src0 & 0x1FFu);
}

constexpr uint32_t make_cdna1_sop1(uint32_t sdst, uint32_t ssrc0) {
  return 0xBE800000u | ((sdst & 0x7Fu) << 16) | (ssrc0 & 0xFFu);
}

TEST(OperandLayoutTest, DeferredSelectorStateFitsExistingPadding) {
  EXPECT_EQ(sizeof(Operand), 32u);
  EXPECT_EQ(sizeof(cdna5::Operand), 80u);
  EXPECT_EQ(sizeof(cdna5::VAddF32Vop3), 512u);
}

TEST(CodeArchApiTest, PreservesExistingPublicEnumValues) {
  EXPECT_EQ(static_cast<int>(ROCJITSU_CODE_ARCH_CDNA5), 11);
  EXPECT_EQ(static_cast<int>(ROCJITSU_CODE_ARCH_RDNA4), 8);
  EXPECT_EQ(static_cast<int>(ROCJITSU_CODE_ARCH_RV32I), 9);
  EXPECT_EQ(static_cast<int>(ROCJITSU_CODE_ARCH_RV64I), 10);
  EXPECT_EQ(static_cast<int>(ROCJITSU_CODE_ARCH_NUM_ARCHS), 12);
  EXPECT_EQ(static_cast<int>(ROCJITSU_CODE_ARCH_INVALID), 12);
}

struct DecodeCase {
  rj_code_arch_t arch;
  const char *arch_name;
  uint32_t word;
  const char *expected_mnemonic;
  int expected_size_bytes;
};

class DecoderSmokeTest : public ::testing::TestWithParam<DecodeCase> {};

TEST(DecoderDiagnosticTest, ValidEncodingEmitsNothingAndKeepsLocation) {
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  std::vector<std::string> messages;
  auto collect = [&](std::string_view message) { messages.emplace_back(message); };

  DecodeResult decoded = decoder->decode(&S_NOP, 24, DecodeErrorEmitter(collect));

  ASSERT_TRUE(decoded.succeeded());
  EXPECT_EQ(decoded.value()->src_loc(), 24u);
  EXPECT_TRUE(messages.empty());
}

TEST(DecoderDiagnosticTest, InvalidOpcodeEmitsExactlyOneDiagnostic) {
  constexpr uint32_t invalid_opcode = 0xffffffffu;
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  std::vector<std::string> messages;
  auto collect = [&](std::string_view message) { messages.emplace_back(message); };

  DecodeResult decoded = decoder->decode(&invalid_opcode, DecodeErrorEmitter(collect));

  EXPECT_TRUE(decoded.failed());
  ASSERT_EQ(messages.size(), 1u);
  EXPECT_EQ(messages.front(), "Invalid instruction opcode: FFFFFFFF");
}

TEST(DecoderDiagnosticTest, InvalidOperandEmitsExactlyOneDiagnostic) {
  constexpr auto words = cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = 209, .vdst = 0});
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  std::vector<std::string> messages;
  auto collect = [&](std::string_view message) { messages.emplace_back(message); };

  DecodeResult decoded = decoder->decode(words.data(), DecodeErrorEmitter(collect));

  EXPECT_TRUE(decoded.failed());
  ASSERT_EQ(messages.size(), 1u);
  EXPECT_EQ(messages.front(), "invalid operand selector");
}

TEST(DecoderDiagnosticTest, NoOpEmitterStillRejectsInvalidOpcode) {
  constexpr uint32_t invalid_opcode = 0xffffffffu;
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);

  EXPECT_TRUE(decoder->decode(&invalid_opcode).failed());
}

TEST_P(DecoderSmokeTest, DecodesCorrectly) {
  const DecodeCase &tc = GetParam();

  auto decoder = Decoder::create(tc.arch);
  ASSERT_NE(decoder, nullptr) << "Decoder::create() returned nullptr for arch=" << tc.arch_name;

  std::unique_ptr<Instruction> inst(decode_valid(*decoder, &tc.word));
  ASSERT_NE(inst, nullptr) << "decode() returned nullptr for arch=" << tc.arch_name << " word=0x"
                           << std::hex << tc.word;

  EXPECT_EQ(inst->mnemonic(), tc.expected_mnemonic) << "Wrong mnemonic for arch=" << tc.arch_name;
  EXPECT_EQ(inst->size(), tc.expected_size_bytes) << "Wrong size for arch=" << tc.arch_name;
}

TEST(Gfx1250DecodeTest, DisassemblesDpp8Selectors) {
  const uint32_t words[] = {
      0x7E0040EAu, // v_fract_f32 with the DPP8FI source selector.
      0x000040CCu, // v204, dpp8:[0,0,1,0,0,0,0,0].
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->size(), 8);
  EXPECT_EQ(inst->mnemonic(), "v_fract_f32_e32");
  EXPECT_EQ(inst->disassemble(), "v_fract_f32_dpp v0, v204 dpp8:[0,0,1,0,0,0,0,0] fi:1");
}

TEST(DecoderSmokeTest, Gfx1201DisassemblesDpp16Attributes) {
  const uint32_t words[] = {
      0xD6410800u, 0x000002FAu, // v_mad_u16 with a DPP16 source.
      0xFF0D0104u,              // v4, row_shl:1, full masks, bound_ctrl and fi.
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->size(), 12);
  EXPECT_EQ(inst->mnemonic(), "v_mad_u16");
  EXPECT_EQ(inst->disassemble(), "v_mad_u16_e64_dpp v0.l, v4.h, s1, s0 op_sel:[1,0,0,0] row_shl:1 "
                                 "row_mask:0xf bank_mask:0xf bound_ctrl:1 fi:1");
}

TEST(DecoderSmokeTest, Gfx950DisassemblesDpp16Attributes) {
  const uint32_t words[] = {
      0x000000FAu, // v_cndmask_b32 with a DPP16 source.
      0xFF08000Bu, // v11, quad_perm identity, full masks and bound_ctrl.
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->size(), 8);
  EXPECT_EQ(inst->mnemonic(), "v_cndmask_b32_e32");
  // CDNA4 models VCC as an implicit use, while LLVM also prints it as an
  // operand. This test intentionally covers the remaining textual fields.
  EXPECT_EQ(inst->disassemble(), "v_cndmask_b32_dpp v0, v11, v0 quad_perm:[0,0,0,0] row_mask:0xf "
                                 "bank_mask:0xf bound_ctrl:1");
}

TEST(DecoderSmokeTest, Gfx1201DppCndmaskUsesTrue16SuffixesWithoutOpSel) {
  const uint32_t words[] = {
      0xD65D5000u,
      0x01AA04FAu,
      0xFF010101u,
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->mnemonic(), "v_cndmask_b16");
  EXPECT_EQ(inst->disassemble(), "v_cndmask_b16_e64_dpp v0.h, v1.l, v2.h, vcc_lo row_shl:1 "
                                 "row_mask:0xf bank_mask:0xf");
}

TEST(DecoderSmokeTest, DppControlNamesFollowIsaDialect) {
  struct TestCase {
    rj_code_arch_t arch;
    uint32_t control;
    const char *expected_attribute;
  };
  constexpr TestCase cases[] = {
      {ROCJITSU_CODE_ARCH_CDNA4, 0x142, "row_bcast:15"},
      {ROCJITSU_CODE_ARCH_CDNA4, 0x150, "row_newbcast:0"},
      {ROCJITSU_CODE_ARCH_RDNA4, 0x150, "row_share:0"},
      {ROCJITSU_CODE_ARCH_RDNA4, 0x160, "row_xmask:0"},
  };

  for (const auto &tc : cases) {
    const uint32_t words[] = {
        0x7E0002FAu,
        0xFF000001u | (tc.control << 8),
    };
    auto decoder = Decoder::create(tc.arch);
    ASSERT_NE(decoder, nullptr);
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, words));
    ASSERT_NE(inst, nullptr);
    EXPECT_NE(inst->disassemble().find(tc.expected_attribute), std::string::npos)
        << inst->disassemble();
  }
}

TEST(DecoderSmokeTest, DppControlsReservedBySelectedIsaAreRejected) {
  struct TestCase {
    rj_code_arch_t arch;
    uint32_t control;
  };
  constexpr TestCase cases[] = {
      {ROCJITSU_CODE_ARCH_CDNA4, 0x160},
      {ROCJITSU_CODE_ARCH_RDNA4, 0x142},
  };

  for (const auto &tc : cases) {
    const uint32_t words[] = {
        0x7E0002FAu,
        0xFF000001u | (tc.control << 8),
    };
    auto decoder = Decoder::create(tc.arch);
    ASSERT_NE(decoder, nullptr);
    EXPECT_TRUE(decode_fails(*decoder, words));
  }
}

TEST(DecoderSmokeTest, Gfx1201DisassemblesVop3pAttributes) {
  const uint32_t words[] = {
      0xCC0FD3FFu,
      0x6A00D6FFu, // v_pk_add_f16 with all packed attributes set.
      0x0000FE0Bu,
      0,
  };

  for (const auto arch : {ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
    auto decoder = Decoder::create(arch);
    ASSERT_NE(decoder, nullptr);
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, words));
    ASSERT_NE(inst, nullptr);
    const char *vcc_hi = arch == ROCJITSU_CODE_ARCH_CDNA5 ? "VCC_HI" : "vcc_hi";
    EXPECT_EQ(inst->disassemble(),
              std::string("v_pk_add_f16 v255, 0xfe0b, ") + vcc_hi +
                  " op_sel:[0,1] op_sel_hi:[1,0] neg_lo:[1,1] neg_hi:[1,1] clamp");
  }
}

TEST(DecoderSmokeTest, Gfx1201NonPackedVop3pUsesZeroOpSelHiDefault) {
  const uint32_t words[] = {
      0xCC160000u,
      0x00000000u,
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->disassemble(), "v_dot4_i32_iu8 v0, s0, s0, s0");
}

TEST(DecoderSmokeTest, Gfx950DisassemblesVop3pAttributes) {
  const uint32_t words[] = {
      0xD38F4805u, 0x18020501u, // v_pk_add_f16 v5, v1, v2 op_sel:[1,0].
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->disassemble(), "v_pk_add_f16 v5, v1, v2 op_sel:[1,0]");
}

TEST(DecoderSmokeTest, Gfx1201DisassemblesVop3Attributes) {
  const uint32_t words[] = {
      0xD648C300u,
      0x2C0E0501u, // v_fma_f16 with source, true16, clamp, and output modifiers.
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->disassemble(),
            "v_fma_f16 v0.h, -|v1.l|, |v2.l|, v3.l op_sel:[0,0,0,1] clamp mul:2");
}

TEST(DecoderSmokeTest, Gfx1201CndmaskB16UsesTrue16SuffixesOnly) {
  const uint32_t words[] = {
      0xD65D5000u,
      0x01AA0501u, // v_cndmask_b16 v0.h, v1.l, v2.h, vcc_lo.
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->disassemble(), "v_cndmask_b16 v0.h, v1.l, v2.h, vcc_lo");
}

TEST(DecoderSmokeTest, Gfx1201IgnoresReservedVop3OpSelBits) {
  const uint32_t words[] = {
      0xD6134800u,
      0x040E0501u,
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->disassemble(), "v_fma_f32 v0, v1, v2, v3");
}

TEST(DecoderSmokeTest, Gfx950DisassemblesVop3Attributes) {
  const uint32_t words[] = {
      0xD1CB8300u,
      0x2C0E0501u, // v_fma_f32 with source, clamp, and output modifiers.
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->disassemble(), "v_fma_f32 v0, -|v1|, |v2|, v3 clamp mul:2");
}

TEST(DecoderSmokeTest, Gfx950Vop3OpSelDoesNotAddHalfSuffixes) {
  const uint32_t words[] = {
      0xD2050805u,
      0x040E0501u,
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->disassemble(), "v_mad_i16 v5, v1, v2, v3 op_sel:[1,0,0,0]");
}

TEST(DecoderSmokeTest, Gfx12CachePolicyUsesOperationKind) {
  struct TestCase {
    std::array<uint32_t, 3> words;
    const char *expected_disassembly;
  };
  constexpr TestCase cases[] = {
      {{{0xEE050000u, 0x001C0000u, 0x00000001u}},
       "global_load_b32 v0, v1, s[0:1] th:TH_LOAD_NT scope:SCOPE_SYS"},
      {{{0xEE068000u, 0x001C0000u, 0x00000001u}},
       "global_store_b32 v1, v0, s[0:1] th:TH_STORE_NT scope:SCOPE_SYS"},
      {{{0xEE0D4000u, 0x011C0000u, 0x00000001u}},
       "global_atomic_add_u32 v0, v1, v2, s[0:1] th:TH_ATOMIC_RETURN scope:SCOPE_SYS"},
  };

  for (const auto arch : {ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
    auto decoder = Decoder::create(arch);
    ASSERT_NE(decoder, nullptr);
    for (const auto &tc : cases) {
      std::unique_ptr<Instruction> inst(decode_valid(*decoder, tc.words.data()));
      ASSERT_NE(inst, nullptr);
      EXPECT_EQ(inst->disassemble(), tc.expected_disassembly);
    }
  }
}

TEST(DecoderSmokeTest, Gfx12VbufferAddressModifiersPrecedeCachePolicy) {
  constexpr uint32_t words[] = {
      0xC4050003u,
      0x409C1005u,
      0x00001001u,
  };

  for (const auto arch : {ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_CDNA5}) {
    auto decoder = Decoder::create(arch);
    ASSERT_NE(decoder, nullptr);
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, words));
    ASSERT_NE(inst, nullptr);
    EXPECT_EQ(inst->disassemble(),
              "buffer_load_b32 v5, v1, s[8:11], s3 offen offset:16 th:TH_LOAD_NT scope:SCOPE_SYS");
  }
}

TEST(DecoderSmokeTest, Gfx1201ImageCachePolicyUsesOperationKind) {
  struct TestCase {
    std::array<uint32_t, 4> words;
    const char *expected_disassembly;
  };
  constexpr TestCase cases[] = {
      {{{0xD0400000u, 0x003C0000u, 0x00000000u, 0x00000000u}},
       "image_load v[0:3], s[0:7] th:TH_LOAD_BYPASS scope:SCOPE_SYS"},
      {{{0xD0418000u, 0x003C0000u, 0x00000000u, 0x00000000u}},
       "image_store v[0:3], s[0:7] th:TH_STORE_BYPASS scope:SCOPE_SYS"},
      {{{0xD0430000u, 0x003C0000u, 0x00000000u, 0x00000000u}},
       "image_atomic_add_uint v[0:3], s[0:7] th:TH_ATOMIC_NT_RETURN scope:SCOPE_SYS"},
      {{{0xE446C000u, 0x003C0000u, 0x00000000u, 0x00000000u}},
       "image_sample v[0:3], s[0:7], s[0:3] th:TH_LOAD_BYPASS scope:SCOPE_SYS"},
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(decoder, nullptr);
  for (const auto &tc : cases) {
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, tc.words.data()));
    ASSERT_NE(inst, nullptr);
    EXPECT_EQ(inst->disassemble(), tc.expected_disassembly);
  }
}

TEST(DecoderSmokeTest, Gfx1201Dpp8MnemonicRequiresInstructionSupport) {
  const uint32_t words[] = {
      0xD5800000u,
      0x000000E9u,
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->disassemble(), "v_nop");
  EXPECT_EQ(inst->size(), sizeof(words));
}

TEST(DecoderSmokeTest, Gfx1201DppSpellingAndTrue16HalvesMatchEncoding) {
  struct TestCase {
    std::array<uint32_t, 3> words;
    const char *expected_disassembly;
  };
  constexpr TestCase cases[] = {
      {{{0x640A04FAu, 0xFF003981u, 0}},
       "v_add_f16_dpp v5.l, v1.h, v2.l quad_perm:[1,2,3,0] row_mask:0xf bank_mask:0xf"},
      {{{0x7F1AD4FAu, 0xFF00398Cu, 0}},
       "v_cvt_i32_i16_dpp v141, v12.h quad_perm:[1,2,3,0] row_mask:0xf bank_mask:0xf"},
      {{{0x7C73F8FAu, 0xFF00391Fu, 0}},
       "v_cmp_lt_u16 v31.l, v124.h quad_perm:[1,2,3,0] row_mask:0xf bank_mask:0xf"},
      {{{0xD5320805u, 0x000204E9u, 0xFAC68801u}},
       "v_add_f16_e64_dpp v5.l, v1.h, v2.l op_sel:[1,0,0] "
       "dpp8:[0,1,2,3,4,5,6,7]"},
      {{{0xD5325005u, 0x000204E9u, 0xFAC68801u}},
       "v_add_f16_e64_dpp v5.h, v1.l, v2.h op_sel:[0,1,1] "
       "dpp8:[0,1,2,3,4,5,6,7]"},
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(decoder, nullptr);
  for (const auto &tc : cases) {
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, tc.words.data()));
    ASSERT_NE(inst, nullptr);
    EXPECT_EQ(inst->disassemble(), tc.expected_disassembly);
  }
}

TEST(DecoderSmokeTest, Gfx1201Vop3SdstRendersOutputModifiers) {
  struct TestCase {
    std::array<uint32_t, 2> words;
    const char *expected_disassembly;
  };
  constexpr TestCase cases[] = {
      {{{0xD7008000u, 0x02020501u}}, "v_add_co_u32 v0, s[0:1], v1, v2 clamp"},
      {{{0xD6FC6A00u, 0x0C0E0501u}}, "v_div_scale_f32 v0, vcc_lo, v1, v2, v3 mul:2"},
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(decoder, nullptr);
  for (const auto &tc : cases) {
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, tc.words.data()));
    ASSERT_NE(inst, nullptr);
    EXPECT_EQ(inst->disassemble(), tc.expected_disassembly);
  }
}

TEST(PackedTrue16DisassemblyTest, SwapOmitsTiedSourceOperands) {
  constexpr std::array architectures{
      ROCJITSU_CODE_ARCH_RDNA3,
      ROCJITSU_CODE_ARCH_RDNA3_5,
      ROCJITSU_CODE_ARCH_RDNA4,
      ROCJITSU_CODE_ARCH_CDNA5,
  };
  constexpr uint32_t words[] = {0x7E0ACD81u, 0u};

  for (auto arch : architectures) {
    SCOPED_TRACE(static_cast<int>(arch));
    auto decoder = Decoder::create(arch);
    ASSERT_NE(decoder, nullptr);
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, words));
    ASSERT_NE(inst, nullptr);
    EXPECT_EQ(inst->disassemble(), "v_swap_b16 v5.l, v1.h");
  }
}

// AMDGPU ISAs × 2 instructions.
// CDNA1/2/3/4 and RDNA1/2 share the GFX9/GFX10 s_endpgm encoding (op=1).
// RDNA3/3.5/4 use the GFX11/GFX12 encoding where s_endpgm moved to op=48.
INSTANTIATE_TEST_SUITE_P(
    AllIsas, DecoderSmokeTest,
    ::testing::Values(
        DecodeCase{ROCJITSU_CODE_ARCH_CDNA1, "cdna1", S_NOP, "s_nop", 4},
        DecodeCase{ROCJITSU_CODE_ARCH_CDNA1, "cdna1", S_ENDPGM_GFX9, "s_endpgm", 4},
        DecodeCase{ROCJITSU_CODE_ARCH_CDNA2, "cdna2", S_NOP, "s_nop", 4},
        DecodeCase{ROCJITSU_CODE_ARCH_CDNA2, "cdna2", S_ENDPGM_GFX9, "s_endpgm", 4},
        DecodeCase{ROCJITSU_CODE_ARCH_CDNA3, "cdna3", S_NOP, "s_nop", 4},
        DecodeCase{ROCJITSU_CODE_ARCH_CDNA3, "cdna3", S_ENDPGM_GFX9, "s_endpgm", 4},
        DecodeCase{ROCJITSU_CODE_ARCH_CDNA4, "cdna4", S_NOP, "s_nop", 4},
        DecodeCase{ROCJITSU_CODE_ARCH_CDNA4, "cdna4", S_ENDPGM_GFX9, "s_endpgm", 4},
        DecodeCase{ROCJITSU_CODE_ARCH_RDNA1, "rdna1", S_NOP, "s_nop", 4},
        DecodeCase{ROCJITSU_CODE_ARCH_RDNA1, "rdna1", S_ENDPGM_GFX9, "s_endpgm", 4},
        DecodeCase{ROCJITSU_CODE_ARCH_RDNA2, "rdna2", S_NOP, "s_nop", 4},
        DecodeCase{ROCJITSU_CODE_ARCH_RDNA2, "rdna2", S_ENDPGM_GFX9, "s_endpgm", 4},
        DecodeCase{ROCJITSU_CODE_ARCH_RDNA3, "rdna3", S_NOP, "s_nop", 4},
        DecodeCase{ROCJITSU_CODE_ARCH_RDNA3, "rdna3", S_ENDPGM_GFX11, "s_endpgm", 4},
        DecodeCase{ROCJITSU_CODE_ARCH_RDNA3_5, "rdna3_5", S_NOP, "s_nop", 4},
        DecodeCase{ROCJITSU_CODE_ARCH_RDNA3_5, "rdna3_5", S_ENDPGM_GFX11, "s_endpgm", 4},
        DecodeCase{ROCJITSU_CODE_ARCH_RDNA4, "rdna4", S_NOP, "s_nop", 4},
        DecodeCase{ROCJITSU_CODE_ARCH_RDNA4, "rdna4", S_ENDPGM_GFX11, "s_endpgm", 4},
        DecodeCase{ROCJITSU_CODE_ARCH_CDNA5, "gfx1250", S_NOP, "s_nop", 4},
        DecodeCase{ROCJITSU_CODE_ARCH_CDNA5, "gfx1250", S_ENDPGM_GFX11, "s_endpgm", 4}),
    [](const ::testing::TestParamInfo<DecodeCase> &info) {
      std::string name = info.param.arch_name;
      name += "_";
      name += info.param.expected_mnemonic;
      return name;
    });

TEST(ScalarRegisterSelectorDecodeTest, Rdna1AndRdna2RejectReservedHole) {
  constexpr auto rdna1_lower =
      rdna1::build_vop3(rdna1::kVReadfirstlaneB32Vop3, {.vdst = 123, .src0 = 256});
  constexpr auto rdna1_hole =
      rdna1::build_vop3(rdna1::kVReadfirstlaneB32Vop3, {.vdst = 124, .src0 = 256});
  constexpr auto rdna1_upper =
      rdna1::build_vop3(rdna1::kVReadfirstlaneB32Vop3, {.vdst = 125, .src0 = 256});
  constexpr auto rdna2_lower =
      rdna2::build_vop3(rdna2::kVReadfirstlaneB32Vop3, {.vdst = 123, .src0 = 256});
  constexpr auto rdna2_hole =
      rdna2::build_vop3(rdna2::kVReadfirstlaneB32Vop3, {.vdst = 124, .src0 = 256});
  constexpr auto rdna2_upper =
      rdna2::build_vop3(rdna2::kVReadfirstlaneB32Vop3, {.vdst = 125, .src0 = 256});

  auto validate = [](rj_code_arch_t arch, const char *arch_name, const auto &lower,
                     const auto &hole, const auto &upper) {
    SCOPED_TRACE(arch_name);
    auto decoder = Decoder::create(arch);
    ASSERT_NE(decoder, nullptr);

    for (const auto &[words, expected_selector] :
         {std::pair{&lower, 123}, std::pair{&upper, 125}}) {
      std::unique_ptr<Instruction> inst;
      inst.reset(decode_valid(*decoder, words->data()));
      ASSERT_NE(inst, nullptr);
      ASSERT_EQ(inst->num_dst_operands(), 1);
      ASSERT_NE(inst->dst_operand(0), nullptr);
      EXPECT_EQ(inst->dst_operand(0)->encoding_value(), expected_selector);
    }

    EXPECT_TRUE(decode_fails(*decoder, hole.data()));
  };

  validate(ROCJITSU_CODE_ARCH_RDNA1, "rdna1", rdna1_lower, rdna1_hole, rdna1_upper);
  validate(ROCJITSU_CODE_ARCH_RDNA2, "rdna2", rdna2_lower, rdna2_hole, rdna2_upper);
}

TEST(OperandSelectorDecodeTest, Cdna1SdstRejectsReservedHole) {
  constexpr uint32_t lower = make_cdna1_sop1(124, 0);
  constexpr uint32_t hole = make_cdna1_sop1(125, 0);
  constexpr uint32_t upper = make_cdna1_sop1(126, 0);

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA1);
  ASSERT_NE(decoder, nullptr);
  for (const auto &[word, expected_selector] : {std::pair{lower, 124}, std::pair{upper, 126}}) {
    std::unique_ptr<Instruction> inst;
    inst.reset(decode_valid(*decoder, &word));
    ASSERT_NE(inst, nullptr);
    ASSERT_EQ(inst->num_dst_operands(), 1);
    ASSERT_NE(inst->dst_operand(0), nullptr);
    EXPECT_EQ(inst->dst_operand(0)->encoding_value(), expected_selector);
  }

  EXPECT_TRUE(decode_fails(*decoder, &hole));
}

TEST(OperandSelectorDecodeTest, Cdna1RestrictedScalarSourceRejectsLiteralSelector) {
  cdna1::Operand valid(32, cdna1::OperandType::OPR_SSRC_NOLIT, 253);
  EXPECT_TRUE(valid.validate_encoding().succeeded());

  cdna1::Operand invalid(32, cdna1::OperandType::OPR_SSRC_NOLIT, 255);
  EXPECT_TRUE(invalid.validate_encoding().failed());
}

TEST(OperandSelectorDecodeTest, DirectSourceAndSmemOffsetRejectReservedSelectors) {
  constexpr auto vop1 = cdna5::build_vop1(cdna5::kVMovB32Vop1, {.src0 = 209, .vdst = 0});
  auto gfx1250_decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(gfx1250_decoder, nullptr);
  EXPECT_TRUE(decode_fails(*gfx1250_decoder, vop1.data()));

  constexpr auto smem =
      cdna1::build_smem(cdna1::kSLoadDwordSmem, {.soffset_en = 1, .soffset = 125});
  auto cdna1_decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA1);
  ASSERT_NE(cdna1_decoder, nullptr);
  EXPECT_TRUE(decode_fails(*cdna1_decoder, smem.data()));
}

TEST(OperandSelectorDecodeTest, RestrictedVectorAndLaneOperandsRejectLiteralMarkers) {
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);

  const auto expect_rejected = [&](const auto &encoding) {
    std::array<uint32_t, 4> words{};
    for (size_t i = 0; i < encoding.size(); ++i)
      words[i] = encoding[i];
    words[encoding.size()] = 7;
    words[encoding.size() + 1] = 0;
    EXPECT_TRUE(decode_fails(*decoder, words.data()));
  };

  for (const uint16_t marker : {uint16_t{254}, uint16_t{255}}) {
    expect_rejected(cdna5::build_vop3(cdna5::kVReadfirstlaneB32Vop3, {.src0 = marker}));
    expect_rejected(cdna5::build_vop3(cdna5::kVReadlaneB32Vop3, {.src0 = marker, .src1 = 128}));
    expect_rejected(cdna5::build_vop3(cdna5::kVReadlaneB32Vop3, {.src0 = 256, .src1 = marker}));
  }
}

TEST(OperandSelectorDecodeTest, Gfx1250WmmaSrc2ValidatesVgprOrInlineSelector) {
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);

  const auto words_for = [](uint16_t selector) {
    const auto encoding =
        cdna5::build_vop3p(cdna5::kVWmmaF3216x16x128F8f6f4Vop3p,
                           {.vdst = 0, .src0 = 256, .src1 = 256, .src2 = selector});
    return std::array<uint32_t, 4>{encoding[0], encoding[1], 0, 0};
  };

  for (const uint16_t selector : {uint16_t{208}, uint16_t{240}, uint16_t{248}, uint16_t{256}}) {
    SCOPED_TRACE(selector);
    const auto words = words_for(selector);
    std::unique_ptr<Instruction> inst;
    inst.reset(decode_valid(*decoder, words.data()));
    ASSERT_NE(inst, nullptr);
    ASSERT_EQ(inst->num_src_operands(), 3);
    ASSERT_NE(inst->src_operand(2), nullptr);
    EXPECT_EQ(inst->src_operand(2)->encoding_value(), selector);
    EXPECT_EQ(inst->size(), 2 * static_cast<int>(sizeof(uint32_t)));
  }

  for (const uint16_t selector :
       {uint16_t{209}, uint16_t{239}, uint16_t{249}, uint16_t{253}, uint16_t{255}}) {
    SCOPED_TRACE(selector);
    const auto words = words_for(selector);
    EXPECT_TRUE(decode_fails(*decoder, words.data()));
  }

  for (const uint16_t selector : {uint16_t{0}, uint16_t{8}, uint16_t{129}, uint16_t{256}}) {
    SCOPED_TRACE(selector);
    const auto encoding =
        cdna5::build_vop3p(cdna5::kVWmmaF3216x16x128Fp8Fp8Vop3p,
                           {.vdst = 0, .src0 = 256, .src1 = 256, .src2 = selector});
    const std::array<uint32_t, 4> words{encoding[0], encoding[1], 0, 0};
    std::unique_ptr<Instruction> inst;
    inst.reset(decode_valid(*decoder, words.data()));
    ASSERT_NE(inst, nullptr);
    ASSERT_NE(inst->src_operand(2), nullptr);
    EXPECT_EQ(inst->src_operand(2)->encoding_value(), selector);
  }
}

TEST(OperandSelectorDecodeTest, Gfx1250AndRdna4ValidateBarrierIdSelectors) {
  auto validate = [](rj_code_arch_t arch, const char *arch_name, auto build, uint8_t negative_max) {
    SCOPED_TRACE(arch_name);
    auto decoder = Decoder::create(arch);
    ASSERT_NE(decoder, nullptr);

    const std::array<uint8_t, 5> valid = {125, 128, 159, 193, negative_max};
    for (const uint8_t selector : valid) {
      SCOPED_TRACE(static_cast<int>(selector));
      const auto words = build(selector);
      std::unique_ptr<Instruction> inst;
      inst.reset(decode_valid(*decoder, words.data()));
      ASSERT_NE(inst, nullptr);
      ASSERT_EQ(inst->num_src_operands(), 1);
      ASSERT_NE(inst->src_operand(0), nullptr);
      EXPECT_EQ(inst->src_operand(0)->encoding_value(), selector);
    }

    for (const uint8_t selector : std::array<uint8_t, 4>{0, 126, 160, 192}) {
      SCOPED_TRACE(static_cast<int>(selector));
      const auto words = build(selector);
      EXPECT_TRUE(decode_fails(*decoder, words.data()));
    }

    for (const uint8_t marker : {uint8_t{254}, uint8_t{255}}) {
      SCOPED_TRACE(static_cast<int>(marker));
      const auto base = build(marker);
      const std::array<uint32_t, 3> words = {base[0], 7, 0};
      EXPECT_TRUE(decode_fails(*decoder, words.data()));
    }
  };

  validate(
      ROCJITSU_CODE_ARCH_CDNA5, "cdna5",
      [](uint8_t selector) {
        return cdna5::build_sop1(cdna5::kSBarrierSignalIsfirstSop1, {.ssrc0 = selector});
      },
      196);
  validate(
      ROCJITSU_CODE_ARCH_RDNA4, "rdna4",
      [](uint8_t selector) {
        return rdna4::build_sop1(rdna4::kSBarrierSignalIsfirstSop1, {.ssrc0 = selector});
      },
      194);
}

TEST(RawEncodingTest, PreservesScalarLiteralWordsAcrossAmdgpuIsas) {
  struct Case {
    rj_code_arch_t arch;
    const char *arch_name;
    uint32_t word;
  };
  constexpr uint32_t s_mov_b32_literal = 0xBE8000FFu;
  constexpr uint32_t rdna1_s_mov_b32_literal = 0xBE8003FFu;
  constexpr Case cases[] = {
      {ROCJITSU_CODE_ARCH_CDNA1, "cdna1", s_mov_b32_literal},
      {ROCJITSU_CODE_ARCH_CDNA2, "cdna2", s_mov_b32_literal},
      {ROCJITSU_CODE_ARCH_CDNA3, "cdna3", s_mov_b32_literal},
      {ROCJITSU_CODE_ARCH_CDNA4, "cdna4", s_mov_b32_literal},
      {ROCJITSU_CODE_ARCH_RDNA1, "rdna1", rdna1_s_mov_b32_literal},
      {ROCJITSU_CODE_ARCH_RDNA2, "rdna2", rdna1_s_mov_b32_literal},
      {ROCJITSU_CODE_ARCH_RDNA3, "rdna3", s_mov_b32_literal},
      {ROCJITSU_CODE_ARCH_RDNA3_5, "rdna3_5", s_mov_b32_literal},
      {ROCJITSU_CODE_ARCH_RDNA4, "rdna4", s_mov_b32_literal},
      {ROCJITSU_CODE_ARCH_CDNA5, "gfx1250", s_mov_b32_literal},
  };

  for (const auto &tc : cases) {
    SCOPED_TRACE(tc.arch_name);
    const uint32_t words[] = {tc.word, 0x12345678u};
    auto decoder = Decoder::create(tc.arch);
    ASSERT_NE(decoder, nullptr) << tc.arch_name;
    std::unique_ptr<Instruction> inst;
    inst.reset(decode_valid(*decoder, words));
    ASSERT_NE(inst, nullptr) << tc.arch_name;
    ASSERT_EQ(inst->size(), sizeof(words)) << tc.arch_name;
    ASSERT_NE(inst->raw_encoding(), nullptr) << tc.arch_name;
    EXPECT_EQ(inst->raw_encoding()[0], words[0]) << tc.arch_name;
    EXPECT_EQ(inst->raw_encoding()[1], words[1]) << tc.arch_name;
  }
}

TEST(SendmsgReturnDecodeTest, Rdna3Selector255DoesNotConsumeLiteral) {
  constexpr uint32_t word = 0xBE804CFFu; // s_sendmsg_rtn_b32 s0, 255
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA3);
  ASSERT_NE(decoder, nullptr);

  std::unique_ptr<Instruction> inst(decode_valid(*decoder, &word));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->mnemonic(), "s_sendmsg_rtn_b32");
  EXPECT_EQ(inst->size(), sizeof(word));
  EXPECT_EQ(inst->src_operand(0)->name(), "255");
}

TEST(SendmsgReturnDecodeTest, Rdna35FormatsReturnMessageSelector) {
  constexpr uint32_t words[] = {
      0xBE804C00u, // s_sendmsg_rtn_b32 s0, sendmsg(0, 0, 0)
      0xBE804D80u, // s_sendmsg_rtn_b64 s[0:1], sendmsg(MSG_RTN_GET_DOORBELL)
  };
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA3_5);
  ASSERT_NE(decoder, nullptr);

  std::unique_ptr<Instruction> b32(decode_valid(*decoder, &words[0]));
  ASSERT_NE(b32, nullptr);
  EXPECT_EQ(b32->disassemble(), "s_sendmsg_rtn_b32 s0, sendmsg(0, 0, 0)");

  std::unique_ptr<Instruction> b64(decode_valid(*decoder, &words[1]));
  ASSERT_NE(b64, nullptr);
  EXPECT_EQ(b64->disassemble(), "s_sendmsg_rtn_b64 s[0:1], sendmsg(MSG_RTN_GET_DOORBELL)");
}

TEST(Rdna35FuzzDecodeTest, PreservesRoundTripSignificantSyntax) {
  struct TestCase {
    std::array<uint32_t, 3> words;
    int size;
    const char *disassembly;
  };
  constexpr TestCase cases[] = {
      {{0x7FE0CD74u, 0u, 0u}, 4, "v_swap_b16 v112.h, v116.l"},
      {{0xCC217FEEu, 0xBBBB00ECu, 0u},
       8,
       "v_fma_mixlo_f16 v238, -|src_shared_limit|, |v128|, -|src_private_limit| "
       "op_sel:[1,1,1] op_sel_hi:[1,1,1]"},
      {{0xCC200000u, 0x00E1FF00u, 0u}, 12, "v_fma_mix_f32 v0, v0, lit(0x0), s56"},
      {{0xCC200000u, 0x040E04FAu, 0xFF010101u},
       12,
       "v_fma_mix_f32_e64_dpp v0, v1, v2, v3 row_shl:1 row_mask:0xf bank_mask:0xf"},
      {{0xD9BE8102u, 0x03BFA100u, 0u},
       8,
       "ds_storexchg_2addr_stride64_rtn_b64 v[3:6], v0, v[161:162], v[191:192] "
       "offset0:2 offset1:129 gds"},
      {{0xF0701200u, 0x00010000u, 0u},
       8,
       "image_sample_d v0, v[0:2], s[4:11], s[0:3] dmask:0x2 "
       "dim:SQ_RSRC_IMG_1D slc"},
      {{0xF0000100u, 0x00000000u, 0u}, 8, "image_load v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D"},
      {{0xF003F380u, 0x00600000u, 0u},
       8,
       "image_load v[0:1], v0, s[0:7] dmask:0x3 dim:SQ_RSRC_IMG_1D unorm glc slc dlc "
       "r128 a16 tfe lwe d16"},
      {{0xF0340100u, 0u, 0u}, 8, "image_atomic_sub v0, v0, s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_1D"},
      {{0xF1107F01u, 0u, 0u},
       12,
       "image_sample_c_d_cl v[0:3], [v0, v0, v0, v0, v0], s[0:7], s[0:3] "
       "dmask:0xf dim:SQ_RSRC_IMG_1D glc slc dlc"},
      {{0xCC0F4205u, 0x5A00FAF0u, 0u}, 8, "v_pk_add_f16 v5, 0.5, m0 neg_lo:[0,1] neg_hi:[0,1]"},
      {{0xCC0E0005u, 0x11A8F87Fu, 0u},
       8,
       "v_pk_fma_f16 v5, exec_hi, null, vcc_lo op_sel_hi:[0,1,0]"},
      {{0xCC1A4105u, 0x3BF4D4FDu, 0u},
       8,
       "v_dot2_f32_bf16 v5, src_scc, vcc_lo, src_scc neg_lo:[1,0,0] "
       "neg_hi:[1,0,0]"},
  };
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA3_5);
  ASSERT_NE(decoder, nullptr);

  for (const auto &test : cases) {
    SCOPED_TRACE(test.disassembly);
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, test.words.data()));
    ASSERT_NE(inst, nullptr);
    EXPECT_EQ(inst->size(), test.size) << test.disassembly;
    EXPECT_EQ(inst->disassemble(), test.disassembly);
    for (int word = 0; word < test.size / 4; ++word)
      EXPECT_EQ(inst->raw_encoding()[word], test.words[word]) << test.disassembly;
  }
}

TEST(Rdna35MimgDecodeTest, PartialNsaUsesOneExtensionDword) {
  struct TestCase {
    std::array<uint32_t, 6> words;
    const char *disassembly;
  };
  constexpr TestCase cases[] = {
      {{{0xF0700F05u, 0x00000020u, 0x24232221u, S_NOP, 0u, 0u}},
       "image_sample_d v[0:3], [v32, v33, v34, v35, v[36:37]], s[0:7], s[0:3] "
       "dmask:0xf dim:SQ_RSRC_IMG_2D"},
      {{{0xF0700F09u, 0x00000020u, 0x24232221u, S_NOP, 0u, 0u}},
       "image_sample_d v[0:3], [v32, v33, v34, v35, v[36:40]], s[0:7], s[0:3] "
       "dmask:0xf dim:SQ_RSRC_IMG_3D"},
      {{{0xF1280F09u, 0x00000020u, 0x24232221u, S_NOP, 0u, 0u}},
       "image_sample_c_d_cl_o v[0:3], [v32, v33, v34, v35, v[36:43]], s[0:7], "
       "s[0:3] dmask:0xf dim:SQ_RSRC_IMG_3D"},
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA3_5);
  ASSERT_NE(decoder, nullptr);
  for (const auto &test : cases) {
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, test.words.data()));
    ASSERT_NE(inst, nullptr);
    ASSERT_EQ(inst->size(), 12);
    EXPECT_EQ(inst->disassemble(), test.disassembly);
    for (int word = 0; word < 3; ++word)
      EXPECT_EQ(inst->raw_encoding()[word], test.words[word]);

    std::unique_ptr<Instruction> next(decode_valid(*decoder, test.words.data() + 3));
    ASSERT_NE(next, nullptr);
    EXPECT_EQ(next->size(), 4);
    EXPECT_EQ(next->mnemonic(), "s_nop");
    EXPECT_EQ(next->raw_encoding()[0], S_NOP);
  }
}

TEST(Rdna35MimgDecodeTest, FixedFormWidthsDriveDefUse) {
  struct TestCase {
    std::array<uint32_t, 2> words;
    const char *disassembly;
    uint16_t vdata_base;
    uint16_t vaddr_base;
    uint8_t vaddr_words;
    uint16_t srsrc_base;
    uint8_t srsrc_words;
  };
  constexpr TestCase cases[] = {
      {{{0xF0600118u, 0x00020105u}},
       "image_msaa_load v[1:4], v[5:7], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_2D_MSAA",
       1,
       5,
       3,
       8,
       8},
      {{{0xF0648F80u, 0x00010409u}},
       "image_bvh_intersect_ray v[4:7], v[9:19], s[4:7]",
       4,
       9,
       11,
       4,
       4},
      {{{0xF0658F80u, 0x00010409u}},
       "image_bvh_intersect_ray v[4:7], v[9:16], s[4:7] a16",
       4,
       9,
       8,
       4,
       4},
      {{{0xF0688F80u, 0x00010409u}},
       "image_bvh64_intersect_ray v[4:7], v[9:20], s[4:7]",
       4,
       9,
       12,
       4,
       4},
      {{{0xF0698F80u, 0x00010409u}},
       "image_bvh64_intersect_ray v[4:7], v[9:17], s[4:7] a16",
       4,
       9,
       9,
       4,
       4},
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA3_5);
  ASSERT_NE(decoder, nullptr);
  for (const auto &test : cases) {
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, test.words.data()));
    ASSERT_NE(inst, nullptr);
    EXPECT_EQ(inst->disassemble(), test.disassembly);
    ASSERT_EQ(inst->num_dst_operands(), 1);
    ASSERT_EQ(inst->num_src_operands(), 2);
    EXPECT_EQ(inst->dst_operand(0)->size_bits(), 128);
    EXPECT_EQ(inst->src_operand(0)->size_bits(), test.vaddr_words * 32);

    InstDefUse def_use(*inst);
    EXPECT_TRUE(def_use.defs.contains({RegClass::VGPR, test.vdata_base, 4}));
    EXPECT_TRUE(def_use.uses.contains({RegClass::VGPR, test.vaddr_base, test.vaddr_words}));
    EXPECT_TRUE(def_use.uses.contains({RegClass::SGPR, test.srsrc_base, test.srsrc_words}));
  }
}

TEST(Rdna35MimgDecodeTest, FixedFormNsaUsesCanonicalGroups) {
  struct TestCase {
    std::array<uint32_t, 3> words;
    int vaddr_words;
    const char *disassembly;
  };
  constexpr TestCase cases[] = {
      {{{0xF060011Du, 0x000A0ACCu, 0x00130E0Bu}},
       4,
       "image_msaa_load v[10:13], [v204, v11, v14, v19], s[40:47] dmask:0x1 "
       "dim:SQ_RSRC_IMG_2D_MSAA_ARRAY"},
      {{{0xF0648F81u, 0x00032732u, 0x2F28142Eu}},
       11,
       "image_bvh_intersect_ray v[39:42], [v50, v46, v[20:22], v[40:42], "
       "v[47:49]], s[12:15]"},
      {{{0xF0658F81u, 0x00032732u, 0x0028142Eu}},
       8,
       "image_bvh_intersect_ray v[39:42], [v50, v46, v[20:22], v[40:42]], "
       "s[12:15] a16"},
      {{{0xF0688F81u, 0x00032732u, 0x2F28142Eu}},
       12,
       "image_bvh64_intersect_ray v[39:42], [v[50:51], v46, v[20:22], v[40:42], "
       "v[47:49]], s[12:15]"},
      {{{0xF0698F81u, 0x00032732u, 0x0028142Eu}},
       9,
       "image_bvh64_intersect_ray v[39:42], [v[50:51], v46, v[20:22], v[40:42]], "
       "s[12:15] a16"},
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA3_5);
  ASSERT_NE(decoder, nullptr);
  for (const auto &test : cases) {
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, test.words.data()));
    ASSERT_NE(inst, nullptr);
    EXPECT_EQ(inst->size(), 12);
    EXPECT_EQ(inst->dst_operand(0)->size_bits(), 128);
    EXPECT_EQ(inst->src_operand(0)->size_bits(), test.vaddr_words * 32);
    EXPECT_EQ(inst->disassemble(), test.disassembly);
  }
}

TEST(FieldlessOperandDecodeTest, SaveexecExposesInertExecAndSccOperands) {
  const uint32_t words[] = {
      0xBE802000u, // s_and_saveexec_b64 s[0:1], s[0:1]
      0x00000000u,
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);

  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words));
  ASSERT_NE(inst, nullptr);
  ASSERT_EQ(inst->mnemonic(), "s_and_saveexec_b64");
  ASSERT_EQ(inst->num_dst_operands(), 3);
  ASSERT_EQ(inst->num_src_operands(), 2);

  const Operand *sdst = inst->dst_operand(0);
  ASSERT_NE(sdst, nullptr);
  EXPECT_FALSE(sdst->is_fieldless());
  EXPECT_TRUE(sdst->to_register_ref().has_value());

  auto expect_inert_fieldless_operand = [](const Operand *op, int size_bits, int encoding_value) {
    ASSERT_NE(op, nullptr);
    EXPECT_TRUE(op->is_fieldless());
    EXPECT_EQ(op->size_bits(), size_bits);
    EXPECT_EQ(op->encoding_value(), encoding_value);
    EXPECT_FALSE(op->to_register_ref().has_value());
    // Inertness must span the register/SIMD classification surface too, not
    // just def-use (to_register_ref): a fieldless operand must not inherit
    // VGPR/SIMD behavior from its operand type by accident. The capability
    // flags applied at construction (apply_fieldless_caps) drive this.
    EXPECT_FALSE(op->is_vgpr());
    EXPECT_FALSE(op->simd_capable());
    EXPECT_FALSE(op->reads_value());
    EXPECT_FALSE(op->is_writable());
  };

  expect_inert_fieldless_operand(inst->dst_operand(1), 64, 126); // EXEC write.
  expect_inert_fieldless_operand(inst->dst_operand(2), 1, 253);  // SCC write.
  expect_inert_fieldless_operand(inst->src_operand(1), 64, 126); // EXEC read.

  EXPECT_EQ(inst->disassemble(), "s_and_saveexec_b64 s[0:1], s[0:1]");
}

// A fieldless operand must not retain register/SIMD behavior from its operand
// type by accident. Placeholder/metadata operands (the image `vaddr`, generated
// as OPR_VGPR with canonical value 0) must be inert across every register/SIMD
// API -- is_vgpr()/simd_capable()/to_register_ref(), the read/write accessors,
// and the SIMD chunk paths -- while value-bearing fieldless operands stay live.
TEST(FieldlessOperandDecodeTest, PlaceholderVaddrInertButSimm32StaysValueBearing) {
  ScopedIsaExecutionBackend execution_backend_scope{&rdna4::execution_backend()};

  // Control: a real (field-bearing) VGPR classifies and reads as a register,
  // and is a normal readable/writable operand.
  rdna4::Operand v0(128, rdna4::OperandType::OPR_VGPR, 0);
  EXPECT_TRUE(v0.is_vgpr());
  EXPECT_TRUE(v0.simd_capable());
  EXPECT_TRUE(v0.to_register_ref().has_value());
  EXPECT_TRUE(v0.reads_value());
  EXPECT_TRUE(v0.is_writable());

  // Placeholder: the fieldless image address must be inert everywhere -- it is
  // not a decoded v0, and gfx12 NSA addressing a single Operand can't express.
  // The inert capability triple is what the generator emits for it.
  rdna4::Operand vaddr(128, rdna4::OperandType::OPR_VGPR, 0);
  vaddr.apply_fieldless_caps(/*reads_value=*/false, /*writable=*/false, /*is_vgpr=*/false);
  EXPECT_TRUE(vaddr.is_fieldless());
  EXPECT_FALSE(vaddr.is_vgpr());
  EXPECT_FALSE(vaddr.simd_capable());
  EXPECT_FALSE(vaddr.to_register_ref().has_value());
  EXPECT_FALSE(vaddr.reads_value());
  EXPECT_FALSE(vaddr.is_writable());

  // Value-bearing literal: a fieldless inline literal is suppressed from
  // disasm/def-use (to_register_ref) and is not a write target, but it still
  // carries a value, so it stays readable and SIMD-capable (immediate
  // broadcast). Note, this is what keeps FMA-K / S_SETREG_IMM32_B32 literals
  // live.
  rdna4::Operand simm(32, rdna4::OperandType::OPR_SIMM32, 0x1234);
  simm.apply_fieldless_caps(/*reads_value=*/true, /*writable=*/false, /*is_vgpr=*/false);
  EXPECT_TRUE(simm.is_fieldless());
  EXPECT_FALSE(simm.is_vgpr());
  EXPECT_TRUE(simm.simd_capable());
  EXPECT_FALSE(simm.to_register_ref().has_value());
  EXPECT_TRUE(simm.reads_value());
  EXPECT_FALSE(simm.is_writable());

  // The same inertness must hold across the read/write and SIMD chunk paths:
  // a fieldless placeholder must not read or clobber a real register, while a
  // value-bearing fieldless literal must still read its value.
  amdgpu::GpuMemory gpu_mem("fieldless_io_mem");
  amdgpu::L2Cache l2("fieldless_io_l2");
  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = ROCJITSU_CODE_ARCH_RDNA4;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 106;
  cfg.vgprs_per_wf = 256;
  cfg.lds_size_kb = 64;
  auto cu = amdgpu::ComputeUnitCore::create("fieldless_io", cfg, &gpu_mem, &l2);
  ASSERT_NE(cu, nullptr);
  auto *wf = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(0xFFFFFFFFULL);

  const uint32_t vb = wf->vgpr_alloc().base;
  constexpr uint32_t kSeed = 0xC0FFEE00u;
  for (uint32_t lane = 0; lane < wf->wf_size(); ++lane)
    cu->write_vgpr(vb + 0, lane, kSeed);

  // Fieldless placeholder at encoding 0 (would resolve to v0 if not inert).
  rdna4::Operand vaddr32(32, rdna4::OperandType::OPR_VGPR, 0);
  vaddr32.apply_fieldless_caps(/*reads_value=*/false, /*writable=*/false, /*is_vgpr=*/false);

  // Value access now goes through the observed RegisterAccess facade (#8417);
  // the operand's own read/write hooks are private backend API.
  // Guarded reads return 0 (read_chunk fills 0, not a broadcast).
  EXPECT_EQ(amdgpu::RegisterAccess(*wf).read_scalar(vaddr32), 0u);
  EXPECT_EQ(amdgpu::RegisterAccess(*wf).read_lane(vaddr32, 0), 0u);
  std::array<uint32_t, 4> chunk{0x11u, 0x22u, 0x33u, 0x44u};
  amdgpu::RegisterAccess(*wf).read_chunk(vaddr32, 0, static_cast<uint32_t>(chunk.size()),
                                         chunk.data());
  for (uint32_t v : chunk)
    EXPECT_EQ(v, 0u);

  // Writes / chunk-writes leave the seeded register untouched.
  amdgpu::RegisterAccess(*wf).write_scalar(vaddr32, 0xBADBAD00u);
  amdgpu::RegisterAccess(*wf).write_lane(vaddr32, 0, 0xBADBAD01u);
  const std::array<uint32_t, 4> wvals{1u, 2u, 3u, 4u};
  amdgpu::RegisterAccess(*wf).write_chunk(vaddr32, 0, static_cast<uint32_t>(wvals.size()),
                                          wvals.data(), 0xFull);
  for (uint32_t lane = 0; lane < wf->wf_size(); ++lane)
    EXPECT_EQ(cu->read_vgpr(vb + 0, lane), kSeed) << "lane " << lane;

  // Control: an identical but field-bearing v0 DOES write, proving the guard
  // (not the fixture) is what makes the placeholder inert.
  rdna4::Operand real_v0(32, rdna4::OperandType::OPR_VGPR, 0);
  amdgpu::RegisterAccess(*wf).write_lane(real_v0, 0, 0x1234u);
  EXPECT_EQ(cu->read_vgpr(vb + 0, 0), 0x1234u);

  // Value-bearing fieldless literal still reads its carried value.
  EXPECT_EQ(amdgpu::RegisterAccess(*wf).read_scalar(simm), 0x1234u);
  EXPECT_EQ(amdgpu::RegisterAccess(*wf).read_lane(simm, 0), 0x1234u);
  // ...and its SIMD chunk path broadcasts the literal to every lane (the mirror
  // of the placeholder's zero-fill check above): a value-bearing fieldless
  // operand must stay live across the chunk path, not just read_scalar.
  std::array<uint32_t, 4> simm_chunk{0u, 0u, 0u, 0u};
  amdgpu::RegisterAccess(*wf).read_chunk(simm, 0, static_cast<uint32_t>(simm_chunk.size()),
                                         simm_chunk.data());
  for (uint32_t v : simm_chunk)
    EXPECT_EQ(v, 0x1234u);
}

TEST(Rdna4WaitcntDecodeSmokeTest, FormatsCompatWaitcntWithGfx11Layout) {
  constexpr uint32_t s_waitcnt_vmcnt1 = make_sopp(/*op=*/9, /*simm16=*/1u << 10);

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_NE(decoder, nullptr);

  std::unique_ptr<Instruction> inst(decode_valid(*decoder, &s_waitcnt_vmcnt1));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->mnemonic(), "s_waitcnt");
  EXPECT_EQ(inst->disassemble(), "s_waitcnt vmcnt(1) expcnt(0) lgkmcnt(0)");
}

TEST(CdnaF16DeclaredLiteralDecodeTest, MasksExtensionToDeclaredOperandWidth) {
  struct Case {
    rj_code_arch_t arch;
    uint32_t opcode;
    const char *mnemonic;
  };
  constexpr Case cases[] = {
      {ROCJITSU_CODE_ARCH_CDNA1, 36, "v_madmk_f16_e32"},
      {ROCJITSU_CODE_ARCH_CDNA1, 37, "v_madak_f16_e32"},
      {ROCJITSU_CODE_ARCH_CDNA2, 36, "v_madmk_f16_e32"},
      {ROCJITSU_CODE_ARCH_CDNA2, 37, "v_madak_f16_e32"},
      {ROCJITSU_CODE_ARCH_CDNA3, 36, "v_madmk_f16_e32"},
      {ROCJITSU_CODE_ARCH_CDNA3, 37, "v_madak_f16_e32"},
  };

  for (const auto &tc : cases) {
    const uint32_t words[] = {
        make_vop2(tc.opcode, /*vdst=*/0, /*vsrc1=*/0, /*src0=*/256),
        0xDEAD3E00u,
    };
    auto decoder = Decoder::create(tc.arch);
    ASSERT_NE(decoder, nullptr);
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, words));
    ASSERT_NE(inst, nullptr);
    ASSERT_EQ(inst->mnemonic(), tc.mnemonic);

    bool found_literal = false;
    for (uint8_t i = 0; i < inst->num_src_operands(); ++i) {
      const Operand *src = inst->src_operand(i);
      ASSERT_NE(src, nullptr);
      if (src->name() != "0x3e00")
        continue;
      found_literal = true;
      EXPECT_EQ(static_cast<uint32_t>(src->encoding_value()), 0x3E00u);
    }
    EXPECT_TRUE(found_literal) << tc.mnemonic;
    EXPECT_NE(inst->disassemble().find("0x3e00"), std::string::npos);
    EXPECT_EQ(inst->disassemble().find("0x-"), std::string::npos);
  }
}

TEST(LiteralDisassemblyTest, Simm32HexUsesUnsignedEncodingBits) {
  rdna4::Operand literal(32, rdna4::OperandType::OPR_SIMM32, static_cast<int>(0x80000000u));
  EXPECT_EQ(literal.name(), "0x80000000");
}

TEST(Rdna3Vop3LiteralDecodeTest, TrigPreopF64ClassifiesMixedWidthLiteralsPerOperand) {
  ScopedIsaExecutionBackend execution_backend_scope{&rdna3::execution_backend()};
  constexpr uint32_t literal = 0xaf123456u;

  amdgpu::GpuMemory gpu_mem("f64_literal_mem");
  amdgpu::L2Cache l2("f64_literal_l2");
  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = ROCJITSU_CODE_ARCH_RDNA3;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 106;
  cfg.vgprs_per_wf = 256;
  cfg.lds_size_kb = 64;
  auto cu = amdgpu::ComputeUnitCore::create("f64_literal", cfg, &gpu_mem, &l2);
  ASSERT_NE(cu, nullptr);
  auto *wf = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf);
  ASSERT_NE(wf, nullptr);
  amdgpu::RegisterAccess regs(*wf);

  rdna3::Vop3InstLiteralMachineInst raw{};
  raw.vdst = 0;
  raw.src0 = 255;
  raw.src1 = 255;
  raw.simm32 = literal;

  rdna3::VTrigPreopF64Vop3 inst(reinterpret_cast<const rdna3::MachineInst *>(&raw));

  ASSERT_EQ(inst.num_src_operands(), 2);
  const Operand *src0 = inst.src_operand(0);
  const Operand *src1 = inst.src_operand(1);
  ASSERT_NE(src0, nullptr);
  ASSERT_NE(src1, nullptr);

  EXPECT_EQ(src0->size_bits(), 64);
  EXPECT_EQ(static_cast<uint32_t>(src0->encoding_value()), literal);
  EXPECT_FALSE(src0->literal64_value().has_value());
  EXPECT_EQ(regs.read_lane64(*src0, 0), 0xaf12345600000000ULL);

  EXPECT_EQ(src1->size_bits(), 32);
  EXPECT_FALSE(src1->literal64_value().has_value());
  EXPECT_EQ(static_cast<uint32_t>(src1->encoding_value()), literal);
}

TEST(Rdna4LiteralOperandTest, SignedI64SignExtendsWithoutClaimingLiteral64Encoding) {
  ScopedIsaExecutionBackend execution_backend_scope{&rdna4::execution_backend()};
  amdgpu::GpuMemory gpu_mem("signed_i64_literal_mem");
  amdgpu::L2Cache l2("signed_i64_literal_l2");
  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = ROCJITSU_CODE_ARCH_RDNA4;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 106;
  cfg.vgprs_per_wf = 256;
  cfg.lds_size_kb = 64;
  auto cu = amdgpu::ComputeUnitCore::create("signed_i64_literal", cfg, &gpu_mem, &l2);
  ASSERT_NE(cu, nullptr);
  auto *wf = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf);
  ASSERT_NE(wf, nullptr);

  amdgpu::RegisterAccess regs(*wf);

  struct LiteralCase {
    uint32_t encoded;
    uint64_t signed_value;
  };
  constexpr std::array cases{
      LiteralCase{0x7fffffffu, 0x000000007fffffffULL},
      LiteralCase{0x80000000u, 0xffffffff80000000ULL},
      LiteralCase{0xffffffffu, 0xffffffffffffffffULL},
  };

  for (const auto &[literal, signed_value] : cases) {
    SCOPED_TRACE(::testing::Message() << "literal=" << literal);

    rdna4::Vop3InstLiteralMachineInst raw{};
    raw.vdst = 0;
    raw.src0 = 255;
    raw.src1 = 256;
    raw.simm32 = literal;

    rdna4::VCmpLtI64Vop3 signed_inst(reinterpret_cast<const rdna4::MachineInst *>(&raw));
    rdna4::VCmpLtU64Vop3 unsigned_inst(reinterpret_cast<const rdna4::MachineInst *>(&raw));

    const Operand *signed_src0 = signed_inst.src_operand(0);
    const Operand *unsigned_src0 = unsigned_inst.src_operand(0);
    ASSERT_NE(signed_src0, nullptr);
    ASSERT_NE(unsigned_src0, nullptr);

    EXPECT_EQ(signed_src0->name(), std::format("0x{:x}", literal));
    EXPECT_EQ(static_cast<uint32_t>(signed_src0->encoding_value()), literal);
    EXPECT_FALSE(signed_src0->literal64_value().has_value());
    EXPECT_EQ(regs.read_lane64(*signed_src0, 0), signed_value);

    EXPECT_EQ(static_cast<uint32_t>(unsigned_src0->encoding_value()), literal);
    EXPECT_FALSE(unsigned_src0->literal64_value().has_value());
    EXPECT_EQ(regs.read_lane64(*unsigned_src0, 0), static_cast<uint64_t>(literal));

    raw.src0 = 256;
    raw.src1 = 255;
    rdna4::VCmpLtI64Vop3 signed_src1_inst(reinterpret_cast<const rdna4::MachineInst *>(&raw));
    const Operand *signed_src1 = signed_src1_inst.src_operand(1);
    ASSERT_NE(signed_src1, nullptr);
    EXPECT_FALSE(signed_src1->literal64_value().has_value());
    EXPECT_EQ(regs.read_lane64(*signed_src1, 0), signed_value);

    rdna4::Sop2InstLiteralMachineInst scalar_raw{};
    scalar_raw.ssrc0 = 255;
    scalar_raw.ssrc1 = 128;
    scalar_raw.simm32 = literal;
    rdna4::SAshrI64Sop2 scalar_inst(reinterpret_cast<const rdna4::MachineInst *>(&scalar_raw));
    const Operand *scalar_src0 = scalar_inst.src_operand(0);
    ASSERT_NE(scalar_src0, nullptr);
    EXPECT_FALSE(scalar_src0->literal64_value().has_value());
    EXPECT_EQ(regs.read_scalar64(*scalar_src0), signed_value);
  }
}

TEST(Rdna3DecodeTest, GlobalFlatAllOnesSaddrIsNull) {
  rdna3::FlatMachineInst encoding{};
  // Flat's primary decode ID includes three lower opcode/padding bits;
  // the machine encoding field holds the upper six bits.
  encoding.encoding = rdna3::encoding::kFlat >> 3;
  encoding.op = rdna3::kFlatLoadB64Flat;
  encoding.seg = 2;
  encoding.addr = 1;
  encoding.saddr = 0x7F;
  encoding.vdst = 2;
  const auto words = std::bit_cast<std::array<uint32_t, 2>>(encoding);

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_RDNA3);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->mnemonic(), "global_load_b64");
  // vaddr + the fieldless OPR_GPUMEM memory operand (saddr is null, so it
  // is not added); gpumem is counted but currently carries no register.
  EXPECT_EQ(inst->num_src_operands(), 2);
  {
    int fieldless_srcs = 0;
    for (int i = 0; i < inst->num_src_operands(); ++i)
      if (inst->src_operand(i)->is_fieldless()) {
        ++fieldless_srcs;
        EXPECT_FALSE(inst->src_operand(i)->to_register_ref());
      }
    EXPECT_EQ(fieldless_srcs, 1);
  }

  InstDefUse def_use(*inst);
  EXPECT_TRUE(def_use.uses.contains({RegClass::VGPR, 1, 2}));
  EXPECT_FALSE(def_use.uses.contains({RegClass::SGPR, 0x7F, 1}));
  EXPECT_FALSE(def_use.uses.contains({RegClass::SGPR, 0x7F, 2}));
}

struct VopdDecodeCase {
  rj_code_arch_t arch;
  const char *arch_name;
  const char *case_name;
  std::array<uint32_t, 3> words;
  int expected_size_bytes;
  const char *expected_mnemonic;
  const char *expected_disasm_substring;
};

constexpr uint16_t vopd_src0_vgpr(uint16_t reg) { return 256 + reg; }

constexpr std::array<uint32_t, 3>
make_vopdxy_pair(uint8_t opx, uint8_t opy, uint16_t srcx0 = vopd_src0_vgpr(1), uint8_t vsrcx1 = 2,
                 uint16_t srcy0 = vopd_src0_vgpr(3), uint8_t vsrcy1 = 4, uint8_t vdstx = 0,
                 uint8_t vdsty = 1, uint32_t literal = 0) {
  return {
      (0x32u << 26) | ((static_cast<uint32_t>(opx) & 0xFu) << 22) |
          ((static_cast<uint32_t>(opy) & 0x1Fu) << 17) | (static_cast<uint32_t>(vsrcx1) << 9) |
          (srcx0 & 0x1FFu),
      (static_cast<uint32_t>(vdstx) << 24) | (static_cast<uint32_t>(vdsty >> 1) << 17) |
          (static_cast<uint32_t>(vsrcy1) << 9) | (srcy0 & 0x1FFu),
      literal,
  };
}

class RdnaVopdDecodeSmokeTest : public ::testing::TestWithParam<VopdDecodeCase> {};

TEST_P(RdnaVopdDecodeSmokeTest, DecodesDualSlotForms) {
  const auto &tc = GetParam();
  auto decoder = Decoder::create(tc.arch);
  ASSERT_NE(decoder, nullptr) << tc.arch_name;

  std::unique_ptr<Instruction> inst(decode_valid(*decoder, tc.words.data()));
  ASSERT_NE(inst, nullptr) << tc.arch_name << " " << tc.case_name;
  EXPECT_EQ(inst->mnemonic(), tc.expected_mnemonic);
  EXPECT_EQ(inst->size(), tc.expected_size_bytes);

  std::string disasm = inst->disassemble();
  EXPECT_NE(disasm.find(tc.expected_disasm_substring), std::string::npos) << disasm;
}

INSTANTIATE_TEST_SUITE_P(
    RdnaVopd, RdnaVopdDecodeSmokeTest,
    ::testing::Values(
        VopdDecodeCase{ROCJITSU_CODE_ARCH_RDNA3, "rdna3", "vopdxy", make_vopdxy_pair(9, 8), 8,
                       "v_dual_cndmask_b32 :: v_dual_mov_b32", "v_dual_cndmask_b32"},
        VopdDecodeCase{ROCJITSU_CODE_ARCH_RDNA3,
                       "rdna3",
                       "literal",
                       {0xC8D006FFu, 0x04020080u, 0x4F7FFFFEu},
                       12,
                       "v_dual_mul_f32 :: v_dual_mov_b32",
                       "0x4f7ffffe"},
        VopdDecodeCase{ROCJITSU_CODE_ARCH_RDNA3, "rdna3", "max_min", make_vopdxy_pair(10, 11), 8,
                       "v_dual_max_f32 :: v_dual_min_f32", "v_dual_min_f32"},
        VopdDecodeCase{ROCJITSU_CODE_ARCH_RDNA3, "rdna3", "dot_and", make_vopdxy_pair(12, 18), 8,
                       "v_dual_dot2acc_f32_f16 :: v_dual_and_b32", "v_dual_and_b32"},
        VopdDecodeCase{ROCJITSU_CODE_ARCH_RDNA3, "rdna3", "dot_bf16_add", make_vopdxy_pair(13, 16),
                       8, "v_dual_dot2acc_f32_bf16 :: v_dual_add_nc_u32",
                       "v_dual_dot2acc_f32_bf16"},
        VopdDecodeCase{ROCJITSU_CODE_ARCH_RDNA3_5, "rdna3_5", "vopdxy", make_vopdxy_pair(9, 8), 8,
                       "v_dual_cndmask_b32 :: v_dual_mov_b32", "v_dual_cndmask_b32"},
        VopdDecodeCase{ROCJITSU_CODE_ARCH_RDNA3_5,
                       "rdna3_5",
                       "literal",
                       {0xC8D006FFu, 0x04020080u, 0x4F7FFFFEu},
                       12,
                       "v_dual_mul_f32 :: v_dual_mov_b32",
                       "0x4f7ffffe"},
        VopdDecodeCase{ROCJITSU_CODE_ARCH_RDNA3_5, "rdna3_5", "max_min", make_vopdxy_pair(10, 11),
                       8, "v_dual_max_f32 :: v_dual_min_f32", "v_dual_min_f32"},
        VopdDecodeCase{ROCJITSU_CODE_ARCH_RDNA3_5, "rdna3_5", "dot_and", make_vopdxy_pair(12, 18),
                       8, "v_dual_dot2acc_f32_f16 :: v_dual_and_b32", "v_dual_and_b32"},
        VopdDecodeCase{ROCJITSU_CODE_ARCH_RDNA3_5, "rdna3_5", "dot_bf16_add",
                       make_vopdxy_pair(13, 16), 8, "v_dual_dot2acc_f32_bf16 :: v_dual_add_nc_u32",
                       "v_dual_dot2acc_f32_bf16"},
        VopdDecodeCase{ROCJITSU_CODE_ARCH_RDNA4, "rdna4", "vopdxy", make_vopdxy_pair(9, 8), 8,
                       "v_dual_cndmask_b32 :: v_dual_mov_b32", "v_dual_cndmask_b32"},
        VopdDecodeCase{ROCJITSU_CODE_ARCH_RDNA4,
                       "rdna4",
                       "literal",
                       {0xC8D006FFu, 0x04020080u, 0x4F7FFFFEu},
                       12,
                       "v_dual_mul_f32 :: v_dual_mov_b32",
                       "0x4f7ffffe"},
        VopdDecodeCase{ROCJITSU_CODE_ARCH_RDNA4, "rdna4", "max_min", make_vopdxy_pair(10, 11), 8,
                       "v_dual_max_num_f32 :: v_dual_min_num_f32", "v_dual_min_num_f32"},
        VopdDecodeCase{ROCJITSU_CODE_ARCH_RDNA4, "rdna4", "dot_and", make_vopdxy_pair(12, 18), 8,
                       "v_dual_dot2acc_f32_f16 :: v_dual_and_b32", "v_dual_and_b32"},
        VopdDecodeCase{ROCJITSU_CODE_ARCH_RDNA4, "rdna4", "dot_bf16_add", make_vopdxy_pair(13, 16),
                       8, "v_dual_dot2acc_f32_bf16 :: v_dual_add_nc_u32",
                       "v_dual_dot2acc_f32_bf16"}),
    [](const ::testing::TestParamInfo<VopdDecodeCase> &info) {
      std::string name = info.param.arch_name;
      name += "_";
      name += info.param.case_name;
      return name;
    });

struct InvalidVopdSlotCase {
  rj_code_arch_t arch;
  const char *arch_name;
  const char *case_name;
  std::array<uint32_t, 3> words;
};

class InvalidVopdSlotDecodeTest : public ::testing::TestWithParam<InvalidVopdSlotCase> {};

TEST_P(InvalidVopdSlotDecodeTest, RejectsOpcodeOutsideProfileSlot) {
  const auto &tc = GetParam();
  auto decoder = Decoder::create(tc.arch);
  ASSERT_NE(decoder, nullptr) << tc.arch_name;

  EXPECT_TRUE(decode_fails(*decoder, tc.words.data())) << tc.arch_name << " " << tc.case_name;
}

INSTANTIATE_TEST_SUITE_P(
    VopdSlotValidation, InvalidVopdSlotDecodeTest,
    ::testing::Values(InvalidVopdSlotCase{ROCJITSU_CODE_ARCH_RDNA3, "rdna3", "invalid_x",
                                          make_vopdxy_pair(14, 8)},
                      InvalidVopdSlotCase{ROCJITSU_CODE_ARCH_RDNA3, "rdna3", "invalid_y",
                                          make_vopdxy_pair(8, 14)},
                      InvalidVopdSlotCase{ROCJITSU_CODE_ARCH_RDNA3_5, "rdna3_5", "invalid_x",
                                          make_vopdxy_pair(14, 8)},
                      InvalidVopdSlotCase{ROCJITSU_CODE_ARCH_RDNA3_5, "rdna3_5", "invalid_y",
                                          make_vopdxy_pair(8, 14)},
                      InvalidVopdSlotCase{ROCJITSU_CODE_ARCH_RDNA4, "rdna4", "invalid_x",
                                          make_vopdxy_pair(14, 8)},
                      InvalidVopdSlotCase{ROCJITSU_CODE_ARCH_RDNA4, "rdna4", "invalid_y",
                                          make_vopdxy_pair(8, 14)},
                      InvalidVopdSlotCase{ROCJITSU_CODE_ARCH_CDNA5, "gfx1250", "invalid_x",
                                          make_vopdxy_pair(12, 8)},
                      InvalidVopdSlotCase{ROCJITSU_CODE_ARCH_CDNA5, "gfx1250",
                                          "invalid_y_defined_opcode", make_vopdxy_pair(8, 18)}),
    [](const ::testing::TestParamInfo<InvalidVopdSlotCase> &info) {
      std::string name = info.param.arch_name;
      name += "_";
      name += info.param.case_name;
      return name;
    });

struct InvalidVopdDecodeCase {
  rj_code_arch_t arch;
  const char *arch_name;
  std::array<uint32_t, 3> words;
};

class RdnaInvalidVopdDecodeSmokeTest : public ::testing::TestWithParam<InvalidVopdDecodeCase> {};

TEST_P(RdnaInvalidVopdDecodeSmokeTest, DoesNotClaimVopd3Encoding) {
  const auto &tc = GetParam();
  auto decoder = Decoder::create(tc.arch);
  ASSERT_NE(decoder, nullptr) << tc.arch_name;

  EXPECT_TRUE(decode_fails(*decoder, tc.words.data()))
      << tc.arch_name << " should reserve the 0xCF VOPD3 prefix";
}

INSTANTIATE_TEST_SUITE_P(
    RdnaVopd, RdnaInvalidVopdDecodeSmokeTest,
    ::testing::Values(
        InvalidVopdDecodeCase{
            ROCJITSU_CODE_ARCH_RDNA3, "rdna3", {0xCF455083u, 0x00000086u, 0x0A000001u}},
        InvalidVopdDecodeCase{
            ROCJITSU_CODE_ARCH_RDNA3_5, "rdna3_5", {0xCF455083u, 0x00000086u, 0x0A000001u}},
        InvalidVopdDecodeCase{
            ROCJITSU_CODE_ARCH_RDNA4, "rdna4", {0xCF455083u, 0x00000086u, 0x0A000001u}}),
    [](const ::testing::TestParamInfo<InvalidVopdDecodeCase> &info) {
      return std::string(info.param.arch_name);
    });

struct RdnaVopdExecutionCase {
  rj_code_arch_t arch;
  const char *arch_name;
};

class RdnaVopdExecutionSmokeTest : public ::testing::TestWithParam<RdnaVopdExecutionCase> {};

TEST_P(RdnaVopdExecutionSmokeTest, RejectsWave64Execution) {
  const auto &tc = GetParam();
  const auto words = make_vopdxy_pair(9, 8);

  amdgpu::GpuMemory gpu_mem(std::string(tc.arch_name) + "_vopd_wave64_mem");
  amdgpu::L2Cache l2(std::string(tc.arch_name) + "_vopd_wave64_l2");

  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = tc.arch;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 106;
  cfg.vgprs_per_wf = 256;
  cfg.lds_size_kb = 64;

  auto cu = amdgpu::ComputeUnitCore::create(std::string(tc.arch_name) + "_vopd_wave64", cfg,
                                            &gpu_mem, &l2);
  ASSERT_NE(cu, nullptr);
  auto *wf = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf, 64);
  ASSERT_NE(wf, nullptr);

  auto decoder = Decoder::create(tc.arch);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
  ASSERT_NE(inst, nullptr);

  EXPECT_THROW(cu->execute_instruction(inst.get(), *wf), util::UnimplementedInst);
}

TEST_P(RdnaVopdExecutionSmokeTest, PreservesFpRoundingAndDx9ZeroSemantics) {
  const auto &tc = GetParam();
  constexpr uint32_t kSrc0 = 0x3F800001u;
  constexpr uint32_t kSrc1 = 0x3F7FFFFFu;
  constexpr uint32_t kLiteralAddend = 0xBF800000u;
  constexpr uint32_t kQuietNan = 0x7FC00000u;
  constexpr uint32_t kPositiveZero = 0x00000000u;
  constexpr uint32_t kNegativeZero = 0x80000000u;
  constexpr uint32_t kFmaakOp = 1;
  constexpr uint32_t kMulDx9ZeroOp = 7;
  constexpr uint32_t kFmaDst = 4;
  constexpr uint32_t kDx9Dst = 5;
  constexpr uint64_t kExecMask = 0xFFFF'FFFFULL;

  const auto words = make_vopdxy_pair(kFmaakOp, kMulDx9ZeroOp, vopd_src0_vgpr(0), 1,
                                      vopd_src0_vgpr(2), 3, kFmaDst, kDx9Dst, kLiteralAddend);
  const uint32_t expected_fma =
      std::bit_cast<uint32_t>(std::fma(std::bit_cast<float>(kSrc0), std::bit_cast<float>(kSrc1),
                                       std::bit_cast<float>(kLiteralAddend)));
  ASSERT_EQ(expected_fma, 0x337FFFFEu);

  amdgpu::GpuMemory gpu_mem(std::string(tc.arch_name) + "_vopd_exec_mem");
  amdgpu::L2Cache l2(std::string(tc.arch_name) + "_vopd_exec_l2");

  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = tc.arch;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 106;
  cfg.vgprs_per_wf = 256;
  cfg.lds_size_kb = 64;

  auto cu =
      amdgpu::ComputeUnitCore::create(std::string(tc.arch_name) + "_vopd_exec", cfg, &gpu_mem, &l2);
  ASSERT_NE(cu, nullptr);
  auto *wf = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(kExecMask);

  auto decoder = Decoder::create(tc.arch);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
  ASSERT_NE(inst, nullptr);

  const uint32_t vb = wf->vgpr_alloc().base;
  for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
    cu->write_vgpr(vb + 0, lane, kSrc0);
    cu->write_vgpr(vb + 1, lane, kSrc1);
    cu->write_vgpr(vb + 2, lane, kQuietNan);
    cu->write_vgpr(vb + 3, lane, (lane & 1u) ? kNegativeZero : kPositiveZero);
    cu->write_vgpr(vb + kFmaDst, lane, 0xDEADBEEFu);
    cu->write_vgpr(vb + kDx9Dst, lane, 0xDEADBEEFu);
  }

  cu->execute_instruction(inst.get(), *wf);

  for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
    EXPECT_EQ(cu->read_vgpr(vb + kFmaDst, lane), expected_fma) << tc.arch_name << " lane " << lane;
    EXPECT_EQ(cu->read_vgpr(vb + kDx9Dst, lane), kPositiveZero) << tc.arch_name << " lane " << lane;
  }
}

TEST_P(RdnaVopdExecutionSmokeTest, DualCndmaskConsumesVccLo) {
  const auto &tc = GetParam();
  constexpr std::array<uint32_t, 2> kWords = {
      0xCA521307u, // v_dual_cndmask_b32 v7, v7, v9 :: v_dual_cndmask_b32 v6, v6, v8
      0x07061106u,
  };
  constexpr uint32_t kXDst = 7;
  constexpr uint32_t kYDst = 6;
  constexpr uint32_t kXFalse = 0x10100000u;
  constexpr uint32_t kXTrue = 0x20200000u;
  constexpr uint32_t kYFalse = 0x30300000u;
  constexpr uint32_t kYTrue = 0x40400000u;
  constexpr uint64_t kVcc = 0xAAAA'AAAAu;
  constexpr uint64_t kExecMask = 0xFFFF'FFFFULL;

  amdgpu::GpuMemory gpu_mem(std::string(tc.arch_name) + "_vopd_cndmask_mem");
  amdgpu::L2Cache l2(std::string(tc.arch_name) + "_vopd_cndmask_l2");

  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = tc.arch;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 106;
  cfg.vgprs_per_wf = 256;
  cfg.lds_size_kb = 64;

  auto cu = amdgpu::ComputeUnitCore::create(std::string(tc.arch_name) + "_vopd_cndmask", cfg,
                                            &gpu_mem, &l2);
  ASSERT_NE(cu, nullptr);
  auto *wf = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(kExecMask);
  wf->set_vcc(kVcc);

  auto decoder = Decoder::create(tc.arch);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, kWords.data()));
  ASSERT_NE(inst, nullptr);

  const uint32_t vb = wf->vgpr_alloc().base;
  for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
    cu->write_vgpr(vb + 7, lane, kXFalse | lane);
    cu->write_vgpr(vb + 9, lane, kXTrue | lane);
    cu->write_vgpr(vb + 6, lane, kYFalse | lane);
    cu->write_vgpr(vb + 8, lane, kYTrue | lane);
  }

  cu->execute_instruction(inst.get(), *wf);

  for (uint32_t lane = 0; lane < wf->wf_size(); ++lane) {
    const bool select_true = ((kVcc >> lane) & 1u) != 0;
    EXPECT_EQ(cu->read_vgpr(vb + kXDst, lane), (select_true ? kXTrue : kXFalse) | lane)
        << tc.arch_name << " lane " << lane;
    EXPECT_EQ(cu->read_vgpr(vb + kYDst, lane), (select_true ? kYTrue : kYFalse) | lane)
        << tc.arch_name << " lane " << lane;
  }
}

TEST_P(RdnaVopdExecutionSmokeTest, DualCndmaskAfterScalarVccMerge) {
  const auto &tc = GetParam();
  constexpr uint32_t kModulusMinusOne = 0xFF65CF7Au;
  constexpr uint32_t kLaneCount = 9;
  constexpr uint64_t kExecMask = (1ULL << kLaneCount) - 1;
  constexpr uint64_t kCorrectionMask = 0x1D8u;
  constexpr std::array<uint64_t, kLaneCount> kA = {
      0,
      1,
      2,
      4284862330ULL,
      4284862329ULL,
      1071215583ULL,
      2142431166ULL,
      3213646750ULL,
      4284862327ULL,
  };
  constexpr std::array<uint64_t, kLaneCount> kB = {
      0, 1, 2, 4284862330ULL, 4284862329ULL, 1071215588ULL, 2142431165ULL, 3213646748ULL, 4,
  };
  constexpr std::array<uint64_t, kLaneCount> kExpected = {
      0, 2, 4, 4284862329ULL, 4284862327ULL, 2142431171ULL, 0, 2142431167ULL, 0,
  };
  const std::array<std::array<uint32_t, 3>, 10> words = {{
      {0xBE8001FFu, kModulusMinusOne, 0u},     // s_mov_b64 s[0:1], 0xff65cf7a
      {0xD7006A06u, 0x02020D08u, 0u},          // v_add_co_u32 v6, vcc_lo, v8, v6
      {0x400E0F09u, 0u, 0u},                   // v_add_co_ci_u32_e32 v7, vcc_lo, v9, v7, vcc_lo
      {0xD4590000u, 0x02020C00u, 0u},          // v_cmp_lt_u64_e64 s0, s[0:1], v[6:7]
      {0xD7000108u, 0x02020CFFu, 0x009A3085u}, // v_add_co_u32 v8, s1, 0x9a3085, v6
      {0xD5207C09u, 0x00060EC1u, 0u},          // v_add_co_ci_u32_e64 v9, null, -1, v7, s1
      {0xD7000100u, 0x02020002u, 0u},          // v_add_co_u32 v0, s1, s2, v0
      {0x8C6A006Au, 0u, 0u},                   // s_or_b32 vcc_lo, vcc_lo, s0
      {0xD5207C01u, 0x00060203u, 0u},          // v_add_co_ci_u32_e64 v1, null, s3, v1, s1
      {0xCA521307u, 0x07061106u, 0u},          // v_dual_cndmask_b32 v7/v6
  }};

  amdgpu::GpuMemory gpu_mem(std::string(tc.arch_name) + "_vopd_vcc_merge_mem");
  amdgpu::L2Cache l2(std::string(tc.arch_name) + "_vopd_vcc_merge_l2");

  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = tc.arch;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 106;
  cfg.vgprs_per_wf = 256;
  cfg.lds_size_kb = 64;

  auto cu = amdgpu::ComputeUnitCore::create(std::string(tc.arch_name) + "_vopd_vcc_merge", cfg,
                                            &gpu_mem, &l2);
  ASSERT_NE(cu, nullptr);
  auto *wf = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(kExecMask);

  auto decoder = Decoder::create(tc.arch);
  ASSERT_NE(decoder, nullptr);
  const auto execute = [&](const std::array<uint32_t, 3> &inst_words) {
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, inst_words.data()));
    ASSERT_NE(inst, nullptr);
    cu->execute_instruction(inst.get(), *wf);
  };

  const uint32_t sb = wf->sgpr_alloc().base;
  const uint32_t vb = wf->vgpr_alloc().base;
  cu->write_sgpr(sb + 0, 0xAAAAAAAAu);
  cu->write_sgpr(sb + 1, 0xBBBBBBBBu);
  cu->write_sgpr(sb + 2, 0u);
  cu->write_sgpr(sb + 3, 0u);
  for (uint32_t lane = 0; lane < kLaneCount; ++lane) {
    cu->write_vgpr(vb + 0, lane, 0u);
    cu->write_vgpr(vb + 1, lane, 0u);
    cu->write_vgpr(vb + 6, lane, static_cast<uint32_t>(kA[lane]));
    cu->write_vgpr(vb + 7, lane, static_cast<uint32_t>(kA[lane] >> 32));
    cu->write_vgpr(vb + 8, lane, static_cast<uint32_t>(kB[lane]));
    cu->write_vgpr(vb + 9, lane, static_cast<uint32_t>(kB[lane] >> 32));
  }

  execute(words[0]);
  EXPECT_EQ(cu->read_sgpr(sb + 0), kModulusMinusOne);
  EXPECT_EQ(cu->read_sgpr(sb + 1), 0u);

  execute(words[1]);
  execute(words[2]);
  EXPECT_EQ(wf->vcc() & kExecMask, 0u);

  execute(words[3]);
  EXPECT_EQ(cu->read_sgpr(sb + 0) & kExecMask, kCorrectionMask);

  execute(words[4]);
  execute(words[5]);
  execute(words[6]);
  execute(words[7]);
  EXPECT_EQ(wf->vcc() & kExecMask, kCorrectionMask);
  execute(words[8]);
  EXPECT_EQ(wf->vcc() & kExecMask, kCorrectionMask);
  execute(words[9]);

  for (uint32_t lane = 0; lane < kLaneCount; ++lane) {
    uint64_t actual = static_cast<uint64_t>(cu->read_vgpr(vb + 6, lane)) |
                      (static_cast<uint64_t>(cu->read_vgpr(vb + 7, lane)) << 32);
    EXPECT_EQ(actual, kExpected[lane]) << tc.arch_name << " lane " << lane;
  }
}

TEST_P(RdnaVopdExecutionSmokeTest, DualCndmaskAfterScalarVccMergeViaCuStep) {
  const auto &tc = GetParam();
  constexpr uint32_t kModulusMinusOne = 0xFF65CF7Au;
  constexpr uint32_t kLaneCount = 9;
  constexpr uint64_t kExecMask = (1ULL << kLaneCount) - 1;
  constexpr uint64_t kCorrectionMask = 0x1D8u;
  constexpr std::array<uint64_t, kLaneCount> kA = {
      0,
      1,
      2,
      4284862330ULL,
      4284862329ULL,
      1071215583ULL,
      2142431166ULL,
      3213646750ULL,
      4284862327ULL,
  };
  constexpr std::array<uint64_t, kLaneCount> kB = {
      0, 1, 2, 4284862330ULL, 4284862329ULL, 1071215588ULL, 2142431165ULL, 3213646748ULL, 4,
  };
  constexpr std::array<uint64_t, kLaneCount> kExpected = {
      0, 2, 4, 4284862329ULL, 4284862327ULL, 2142431171ULL, 0, 2142431167ULL, 0,
  };
  constexpr std::array<uint32_t, 19> kWords = {
      0xBE8001FFu, kModulusMinusOne,              // s_mov_b64 s[0:1], 0xff65cf7a
      0xD7006A06u, 0x02020D08u,                   // v_add_co_u32 v6, vcc_lo, v8, v6
      0x400E0F09u,                                // v_add_co_ci_u32_e32 v7, vcc_lo, v9, v7, vcc_lo
      0xD4590000u, 0x02020C00u,                   // v_cmp_lt_u64_e64 s0, s[0:1], v[6:7]
      0xD7000108u, 0x02020CFFu,      0x009A3085u, // v_add_co_u32 v8, s1, 0x9a3085, v6
      0xD5207C09u, 0x00060EC1u,                   // v_add_co_ci_u32_e64 v9, null, -1, v7, s1
      0xD7000100u, 0x02020002u,                   // v_add_co_u32 v0, s1, s2, v0
      0x8C6A006Au,                                // s_or_b32 vcc_lo, vcc_lo, s0
      0xD5207C01u, 0x00060203u,                   // v_add_co_ci_u32_e64 v1, null, s3, v1, s1
      0xCA521307u, 0x07061106u,                   // v_dual_cndmask_b32 v7/v6
  };
  constexpr std::array<uint64_t, 10> kExpectedPc = {8, 16, 20, 28, 40, 48, 56, 60, 68, 76};

  amdgpu::GpuMemory gpu_mem(std::string(tc.arch_name) + "_vopd_vcc_merge_step_mem");
  amdgpu::L2Cache l2(std::string(tc.arch_name) + "_vopd_vcc_merge_step_l2");

  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = tc.arch;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 106;
  cfg.vgprs_per_wf = 256;
  cfg.lds_size_kb = 64;

  auto cu = amdgpu::ComputeUnitCore::create(std::string(tc.arch_name) + "_vopd_vcc_merge_step", cfg,
                                            &gpu_mem, &l2);
  ASSERT_NE(cu, nullptr);
  auto *wf = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf);
  ASSERT_NE(wf, nullptr);
  wf->set_exec(kExecMask);

  for (uint32_t i = 0; i < kWords.size(); ++i)
    gpu_mem.write32(i * sizeof(uint32_t), kWords[i]);

  const uint32_t sb = wf->sgpr_alloc().base;
  const uint32_t vb = wf->vgpr_alloc().base;
  cu->write_sgpr(sb + 0, 0xAAAAAAAAu);
  cu->write_sgpr(sb + 1, 0xBBBBBBBBu);
  cu->write_sgpr(sb + 2, 0u);
  cu->write_sgpr(sb + 3, 0u);
  for (uint32_t lane = 0; lane < kLaneCount; ++lane) {
    cu->write_vgpr(vb + 0, lane, 0u);
    cu->write_vgpr(vb + 1, lane, 0u);
    cu->write_vgpr(vb + 6, lane, static_cast<uint32_t>(kA[lane]));
    cu->write_vgpr(vb + 7, lane, static_cast<uint32_t>(kA[lane] >> 32));
    cu->write_vgpr(vb + 8, lane, static_cast<uint32_t>(kB[lane]));
    cu->write_vgpr(vb + 9, lane, static_cast<uint32_t>(kB[lane] >> 32));
  }

  for (uint32_t i = 0; i < kExpectedPc.size(); ++i) {
    cu->step();
    EXPECT_EQ(wf->pc, kExpectedPc[i]) << tc.arch_name << " step " << i;
    if (i == 0) {
      EXPECT_EQ(cu->read_sgpr(sb + 0), kModulusMinusOne);
      EXPECT_EQ(cu->read_sgpr(sb + 1), 0u);
    }
  }

  EXPECT_EQ(wf->vcc() & kExecMask, kCorrectionMask);
  for (uint32_t lane = 0; lane < kLaneCount; ++lane) {
    uint64_t actual = static_cast<uint64_t>(cu->read_vgpr(vb + 6, lane)) |
                      (static_cast<uint64_t>(cu->read_vgpr(vb + 7, lane)) << 32);
    EXPECT_EQ(actual, kExpected[lane]) << tc.arch_name << " lane " << lane;
  }
}

INSTANTIATE_TEST_SUITE_P(
    RdnaVopd, RdnaVopdExecutionSmokeTest,
    ::testing::Values(RdnaVopdExecutionCase{ROCJITSU_CODE_ARCH_RDNA3, "rdna3"},
                      RdnaVopdExecutionCase{ROCJITSU_CODE_ARCH_RDNA3_5, "rdna3_5"},
                      RdnaVopdExecutionCase{ROCJITSU_CODE_ARCH_RDNA4, "rdna4"}),
    [](const ::testing::TestParamInfo<RdnaVopdExecutionCase> &info) {
      return std::string(info.param.arch_name);
    });

void expect_sleep_yields_before_quantum_expires(rj_code_arch_t arch, uint32_t sleep_encoding) {
  constexpr uint64_t kCodeAddress = 0x2000;

  amdgpu::GpuMemory gpu_mem("functional_sleep_mem");
  amdgpu::L2Cache l2("functional_sleep_l2");
  gpu_mem.write32(kCodeAddress, sleep_encoding);
  gpu_mem.write32(kCodeAddress + sizeof(uint32_t), S_NOP);

  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = arch;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 104;
  cfg.vgprs_per_wf = 256;
  cfg.lds_size_kb = 64;

  auto cu = amdgpu::ComputeUnitCore::create("functional_sleep_cu", cfg, &gpu_mem, &l2);
  ASSERT_NE(cu, nullptr);
  auto *wf = cu->dispatch_wf(0, kCodeAddress, cfg.sgprs_per_wf, cfg.vgprs_per_wf);
  ASSERT_NE(wf, nullptr);

  EXPECT_TRUE(cu->execute_quantum());
  EXPECT_EQ(wf->pc, kCodeAddress + sizeof(uint32_t));
}

TEST(FunctionalSchedulingTest, SleepYieldsBeforeQuantumExpires) {
  constexpr uint32_t kSleep = 0xBF8E0001u; // CDNA4 s_sleep 1
  expect_sleep_yields_before_quantum_expires(ROCJITSU_CODE_ARCH_CDNA4, kSleep);
}

TEST(FunctionalSchedulingTest, SleepVarYieldsBeforeQuantumExpires) {
  constexpr uint32_t kSleepVar = 0xBE805800u; // RDNA4 s_sleep_var s0
  expect_sleep_yields_before_quantum_expires(ROCJITSU_CODE_ARCH_RDNA4, kSleepVar);
}

TEST(FunctionalSchedulingTest, DebugActiveUsesFairQuantum) {
  constexpr uint64_t kCodeAddress = 0x3000;

  amdgpu::GpuMemory gpu_mem("functional_debug_mem");
  amdgpu::L2Cache l2("functional_debug_l2");
  for (uint32_t i = 0; i <= amdgpu::ComputeUnitCore::kDebugFunctionalQuantum; ++i)
    gpu_mem.write32(kCodeAddress + i * sizeof(uint32_t), S_NOP);
  gpu_mem.write32(kCodeAddress +
                      (amdgpu::ComputeUnitCore::kDebugFunctionalQuantum + 1) * sizeof(uint32_t),
                  S_ENDPGM_GFX9);

  amdgpu::ComputeUnitCore::Config cfg{};
  cfg.arch = ROCJITSU_CODE_ARCH_CDNA4;
  cfg.num_wf_slots = 1;
  cfg.sgprs_per_wf = 104;
  cfg.vgprs_per_wf = 256;
  cfg.lds_size_kb = 64;

  auto cu = amdgpu::ComputeUnitCore::create("functional_debug_cu", cfg, &gpu_mem, &l2);
  ASSERT_NE(cu, nullptr);
  auto *wf = cu->dispatch_wf(0, kCodeAddress, cfg.sgprs_per_wf, cfg.vgprs_per_wf);
  ASSERT_NE(wf, nullptr);

  cu->set_debug_active(true);
  EXPECT_TRUE(cu->execute_quantum());
  EXPECT_EQ(wf->pc,
            kCodeAddress + amdgpu::ComputeUnitCore::kDebugFunctionalQuantum * sizeof(uint32_t));
  EXPECT_FALSE(wf->is_halted());
}

// ---------------------------------------------------------------------------
// MUBUF lds modifier test: verify that buffer_load_dword with the lds bit set
// (bit 16 of dword 0) produces a disassembly string containing " lds".
//
// MUBUF encoding (CDNA3/4):
//   dword 0: [31:26]=enc  [24:18]=op  [17]=nt  [16]=lds  [15]=sc1
//            [14]=sc0  [13]=idxen  [12]=offen  [11:0]=offset
//   dword 1: [31:24]=vdata  [20:16]=vaddr  [15:11]=srsrc  [10:8]=soffset
//
// buffer_load_dword without lds: {0xE0500000, 0x00000000}
// buffer_load_dword with    lds: {0xE0510000, 0x00000000}  (bit 16 set)
// ---------------------------------------------------------------------------

struct MubufLdsCase {
  rj_code_arch_t arch;
  const char *arch_name;
  uint32_t words[2];
  bool expect_lds;
};

class MubufLdsModifierTest : public ::testing::TestWithParam<MubufLdsCase> {};

TEST_P(MubufLdsModifierTest, LdsModifierInDisassembly) {
  const auto &tc = GetParam();
  auto decoder = Decoder::create(tc.arch);
  ASSERT_NE(decoder, nullptr);

  std::unique_ptr<Instruction> inst(decode_valid(*decoder, tc.words));
  ASSERT_NE(inst, nullptr) << "decode() returned nullptr for " << tc.arch_name;
  EXPECT_EQ(inst->mnemonic(), "buffer_load_dword");

  std::string disasm = inst->disassemble();
  if (tc.expect_lds) {
    EXPECT_NE(disasm.find(" lds"), std::string::npos)
        << "Expected ' lds' in disassembly: " << disasm;
  } else {
    EXPECT_EQ(disasm.find(" lds"), std::string::npos)
        << "Unexpected ' lds' in disassembly: " << disasm;
  }
}

INSTANTIATE_TEST_SUITE_P(
    MubufLds, MubufLdsModifierTest,
    ::testing::Values(
        // CDNA4: buffer_load_dword without lds
        MubufLdsCase{ROCJITSU_CODE_ARCH_CDNA4, "cdna4", {0xE0500000u, 0x00000000u}, false},
        // CDNA4: buffer_load_dword with lds (bit 16 set)
        MubufLdsCase{ROCJITSU_CODE_ARCH_CDNA4, "cdna4", {0xE0510000u, 0x00000000u}, true},
        // CDNA1: buffer_load_dword without lds (GFX9 MUBUF enc=0x38)
        MubufLdsCase{ROCJITSU_CODE_ARCH_CDNA1, "cdna1", {0xE0500000u, 0x00000000u}, false},
        // CDNA1: buffer_load_dword with lds (bit 16 set)
        MubufLdsCase{ROCJITSU_CODE_ARCH_CDNA1, "cdna1", {0xE0510000u, 0x00000000u}, true}),
    [](const ::testing::TestParamInfo<MubufLdsCase> &info) {
      std::string name = info.param.arch_name;
      name += info.param.expect_lds ? "_with_lds" : "_without_lds";
      return name;
    });

// SMEM IMM=0, SOE=0: OFFSET[6:0] is an SGPR and must decode as a register operand.
TEST(CdnaSmemDecodeTest, SgprOffsetFormExposesOffsetRegister) {
  const struct {
    uint32_t words[2];
    const char *text;
  } cases[] = {
      {{0xC0000101u, 0x00000005u}, "s_load_dword s4, s[2:3], s5"},
      {{0xC0140101u, 0x00000005u}, "s_scratch_load_dword s4, s[2:3], s5"},
  };
  for (rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_CDNA1, ROCJITSU_CODE_ARCH_CDNA2,
                              ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4}) {
    auto decoder = Decoder::create(arch);
    ASSERT_NE(decoder, nullptr) << arch;
    for (const auto &tc : cases) {
      std::unique_ptr<Instruction> inst(decode_valid(*decoder, tc.words));
      ASSERT_NE(inst, nullptr) << arch;
      EXPECT_EQ(inst->disassemble(), tc.text) << arch;

      InstDefUse def_use(*inst);
      EXPECT_TRUE(def_use.uses.contains({RegClass::SGPR, 5, 1})) << arch << " " << tc.text;
    }
  }
}

TEST(Cdna3DecodeTest, DsRead2st64AccDestinationUsesAccumulatorRegisterClass) {
  const uint32_t words[] = {
      0xDA704746u,
      0x3E0000F3u,
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA3);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->mnemonic(), "ds_read2st64_b32");
  EXPECT_EQ(inst->disassemble(), "ds_read2st64_b32 acc[62:63], v243 offset0:70 offset1:71");

  InstDefUse def_use(*inst);
  EXPECT_TRUE(def_use.defs.contains({RegClass::ACC_VGPR, 62, 2}));
  EXPECT_FALSE(def_use.defs.contains({RegClass::VGPR, 62, 2}));
  EXPECT_TRUE(def_use.uses.contains({RegClass::VGPR, 243, 1}));
}

TEST(Cdna3DecodeTest, DsDisassemblyPreservesEncodedOffsets) {
  struct Case {
    uint32_t words[2];
    const char *expected;
  };
  const Case cases[] = {
      {{0xD83E0800u, 0x00000A45u}, "ds_write_b16 v69, v10 offset:2048"},
      {{0xD8F00800u, 0x22000022u}, "ds_read2st64_b64 v[34:37], v34 offset1:8"},
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA3);
  ASSERT_NE(decoder, nullptr);
  for (const Case &test : cases) {
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, test.words));
    ASSERT_NE(inst, nullptr);
    EXPECT_EQ(inst->disassemble(), test.expected);
  }
}

TEST(Cdna3DecodeTest, GlobalDisassemblySignExtendsSegmentOffset) {
  const uint32_t words[] = {
      0xDC489000u,
      0x567F0002u,
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA3);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->disassemble(), "global_load_ushort v86, v[2:3] offset:-4096");
}

TEST(Cdna3DecodeTest, PkFmacF16AcceptsSdwaEncoding) {
  const uint32_t words[] = {
      0x796666F9u,
      0x66666666u,
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA3);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->mnemonic(), "v_pk_fmac_f16_sdwa");
  EXPECT_EQ(inst->size(), 8);
  EXPECT_EQ(inst->disassemble(),
            "v_pk_fmac_f16_sdwa v179, v179, v102, v51 clamp mul:2 dst_sel:DWORD "
            "dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD");
}

TEST(Cdna3DecodeTest, AccVgprMovesDoNotRenderPackedSourceModifiers) {
  const uint32_t words[] = {
      0xD3D9400Fu,
      0x18000080u,
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA3);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->disassemble(), "v_accvgpr_write acc15, 0");
}

TEST(Cdna3DecodeTest, MfmaAccCdUsesAccumulatorRegisterClassForCAndD) {
  const uint32_t words[] = {
      0xD3E08088u,
      0x1E22A554u,
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA3);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->mnemonic(), "v_mfma_f32_32x32x8_bf16");
  EXPECT_EQ(inst->disassemble(),
            "v_mfma_f32_32x32x8_bf16 acc[136:151], acc[84:85], acc[82:83], acc[136:151]");

  InstDefUse def_use(*inst);
  EXPECT_TRUE(def_use.defs.contains({RegClass::ACC_VGPR, 136, 16}));
  EXPECT_FALSE(def_use.defs.contains({RegClass::VGPR, 136, 16}));
  EXPECT_TRUE(def_use.uses.contains({RegClass::ACC_VGPR, 84, 2}));
  EXPECT_TRUE(def_use.uses.contains({RegClass::ACC_VGPR, 82, 2}));
  EXPECT_TRUE(def_use.uses.contains({RegClass::ACC_VGPR, 136, 16}));
  EXPECT_FALSE(def_use.uses.contains({RegClass::VGPR, 136, 16}));
}

TEST(Cdna3DecodeTest, MfmaAccBitsSelectIndependentMultiplicandBanks) {
  struct TestCase {
    uint32_t high_word;
    RegClass src0_class;
    RegClass src1_class;
  };
  constexpr TestCase cases[] = {
      {0x0E22A554u, RegClass::ACC_VGPR, RegClass::VGPR},
      {0x1622A554u, RegClass::VGPR, RegClass::ACC_VGPR},
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA3);
  ASSERT_NE(decoder, nullptr);
  for (const auto &tc : cases) {
    const uint32_t words[] = {0xD3E08088u, tc.high_word};
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, words));
    ASSERT_NE(inst, nullptr);
    ASSERT_EQ(inst->mnemonic(), "v_mfma_f32_32x32x8_bf16");

    InstDefUse def_use(*inst);
    EXPECT_TRUE(def_use.uses.contains({tc.src0_class, 84, 2})) << inst->disassemble();
    EXPECT_TRUE(def_use.uses.contains({tc.src1_class, 82, 2})) << inst->disassemble();
  }
}

TEST(Cdna3DecodeTest, MfmaAccCdPreservesInlineConstantSrc2) {
  const uint32_t words[] = {
      0xD3E08088u,
      0x1A02A554u,
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA3);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->mnemonic(), "v_mfma_f32_32x32x8_bf16");
  EXPECT_EQ(inst->disassemble(), "v_mfma_f32_32x32x8_bf16 acc[136:151], acc[84:85], acc[82:83], 0");
}

TEST(Cdna2DecodeTest, MemoryAccBitSelectsAccumulatorDestination) {
  struct TestCase {
    const char *mnemonic;
    std::array<uint32_t, 2> vgpr_words;
    std::array<uint32_t, 2> accvgpr_words;
    uint8_t width;
  };
  constexpr TestCase cases[] = {
      {"global_load_dword", {0xDC508000u, 0x057F0002u}, {0xDC508000u, 0x05FF0002u}, 1},
      {"scratch_load_dword", {0xDC504000u, 0x05020000u}, {0xDC504000u, 0x05820000u}, 1},
      {"image_load", {0xF0000100u, 0x00020502u}, {0xF0010100u, 0x00020502u}, 4},
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA2);
  ASSERT_NE(decoder, nullptr);
  for (const auto &tc : cases) {
    std::unique_ptr<Instruction> vgpr_inst(decode_valid(*decoder, tc.vgpr_words.data()));
    ASSERT_NE(vgpr_inst, nullptr) << tc.mnemonic;
    ASSERT_EQ(vgpr_inst->mnemonic(), tc.mnemonic);
    const auto vgpr_dst = vgpr_inst->dst_operand(0)->to_register_ref();
    ASSERT_TRUE(vgpr_dst.has_value()) << vgpr_inst->disassemble();
    EXPECT_EQ(*vgpr_dst, (RegisterRef{RegClass::VGPR, 5, tc.width})) << vgpr_inst->disassemble();

    std::unique_ptr<Instruction> accvgpr_inst(decode_valid(*decoder, tc.accvgpr_words.data()));
    ASSERT_NE(accvgpr_inst, nullptr) << tc.mnemonic;
    ASSERT_EQ(accvgpr_inst->mnemonic(), tc.mnemonic);
    const auto accvgpr_dst = accvgpr_inst->dst_operand(0)->to_register_ref();
    ASSERT_TRUE(accvgpr_dst.has_value()) << accvgpr_inst->disassemble();
    EXPECT_EQ(*accvgpr_dst, (RegisterRef{RegClass::ACC_VGPR, 5, tc.width}))
        << accvgpr_inst->disassemble();
  }
}

TEST(Gfx1250DecodeTest, FmamkF64ConsumesThreeDwords) {
  const uint32_t words[] = {
      0x46040504u, // v_fmamk_f64 v[2:3], v[4:5], 0xc1f0000000000000, v[2:3]
      0x00000000u,
      0xC1F00000u,
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->mnemonic(), "v_fmamk_f64_e32");
  EXPECT_EQ(inst->size(), sizeof(words));

  const std::string disasm = inst->disassemble();
  EXPECT_NE(disasm.find("0xc1f0000000000000"), std::string::npos) << disasm;
}

TEST(Gfx1250DecodeTest, FmaakF64ConsumesThreeDwords) {
  const uint32_t words[] = {
      0x48040504u, // v_fmaak_f64 v[2:3], v[4:5], v[2:3], 0xc1f0000000000000
      0x00000000u,
      0xC1F00000u,
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->mnemonic(), "v_fmaak_f64_e32");
  EXPECT_EQ(inst->size(), sizeof(words));
  EXPECT_EQ(inst->disassemble(), "v_fmaak_f64_e32 v[2:3], v[4:5], v[2:3], 0xc1f0000000000000");
}

TEST(Gfx1250DecodeTest, Vop3True16DestinationUsesFullEightBitVgprIndex) {
  const uint32_t words[] = {
      0xD7620086u,
      0x02030CFFu,
      0x000000FFu,
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->mnemonic(), "v_and_b16");
  EXPECT_EQ(inst->size(), sizeof(words));
  EXPECT_EQ(inst->disassemble(), "v_and_b16 v134.l, 0xff, v134.l");
}

TEST(Gfx1250DecodeTest, FlatVaddrWidthFollowsSaddrMode) {
  // LLVM disassembles these words as flat_load_b64 v[2:3], v1, s[6:7].
  const uint32_t saddr_words[] = {
      0xEC054006u,
      0x00000002u,
      0x00000001u,
  };
  // The same instruction with SADDR disabled uses a 64-bit vector address.
  const uint32_t vector_only_words[] = {
      0xEC05407Cu,
      0x00000002u,
      0x00000001u,
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);

  std::unique_ptr<Instruction> saddr_inst(decode_valid(*decoder, saddr_words));
  ASSERT_NE(saddr_inst, nullptr);
  EXPECT_EQ(saddr_inst->mnemonic(), "flat_load_b64");
  EXPECT_EQ(saddr_inst->disassemble(), "flat_load_b64 v[2:3], v1, s[6:7]");
  // vaddr + saddr + fieldless OPR_GPUMEM memory operand (counted but currently
  // to_register_ref() == nullopt).
  EXPECT_EQ(saddr_inst->num_src_operands(), 3);
  {
    int fieldless_srcs = 0;
    for (int i = 0; i < saddr_inst->num_src_operands(); ++i)
      if (saddr_inst->src_operand(i)->is_fieldless()) {
        ++fieldless_srcs;
        EXPECT_FALSE(saddr_inst->src_operand(i)->to_register_ref());
      }
    EXPECT_EQ(fieldless_srcs, 1);
  }
  InstDefUse saddr_def_use(*saddr_inst);
  EXPECT_TRUE(saddr_def_use.uses.contains({RegClass::VGPR, 1, 1}));
  EXPECT_FALSE(saddr_def_use.uses.contains({RegClass::VGPR, 1, 2}));
  EXPECT_TRUE(saddr_def_use.uses.contains({RegClass::SGPR, 6, 2}));

  std::unique_ptr<Instruction> vector_only_inst(decode_valid(*decoder, vector_only_words));
  ASSERT_NE(vector_only_inst, nullptr);
  EXPECT_EQ(vector_only_inst->mnemonic(), "flat_load_b64");
  EXPECT_EQ(vector_only_inst->num_src_operands(), 2); // vaddr + gpumem
  InstDefUse vector_only_def_use(*vector_only_inst);
  EXPECT_TRUE(vector_only_def_use.uses.contains({RegClass::VGPR, 1, 2}));
  EXPECT_FALSE(vector_only_def_use.uses.contains({RegClass::SGPR, 124, 1}));
  EXPECT_FALSE(vector_only_def_use.uses.contains({RegClass::SGPR, 124, 2}));
}

TEST(Gfx1250DecodeTest, GlobalVaddrWidthFollowsSaddrMode) {
  // LLVM disassembles these words as global_load_b64 v[2:3], v10, s[6:7].
  const uint32_t saddr_words[] = {
      0xEE054006u,
      0x00000002u,
      0x0000000Au,
  };
  // The same instruction with SADDR disabled uses a 64-bit vector address.
  const uint32_t vector_only_words[] = {
      0xEE05407Cu,
      0x00000002u,
      0x0000000Au,
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);

  std::unique_ptr<Instruction> saddr_inst(decode_valid(*decoder, saddr_words));
  ASSERT_NE(saddr_inst, nullptr);
  EXPECT_EQ(saddr_inst->mnemonic(), "global_load_b64");
  EXPECT_EQ(saddr_inst->disassemble(), "global_load_b64 v[2:3], v10, s[6:7]");
  // vaddr + saddr + fieldless OPR_GPUMEM memory operand (counted but currently
  // to_register_ref() == nullopt).
  EXPECT_EQ(saddr_inst->num_src_operands(), 3);
  {
    int fieldless_srcs = 0;
    for (int i = 0; i < saddr_inst->num_src_operands(); ++i)
      if (saddr_inst->src_operand(i)->is_fieldless()) {
        ++fieldless_srcs;
        EXPECT_FALSE(saddr_inst->src_operand(i)->to_register_ref());
      }
    EXPECT_EQ(fieldless_srcs, 1);
  }
  InstDefUse saddr_def_use(*saddr_inst);
  EXPECT_TRUE(saddr_def_use.uses.contains({RegClass::VGPR, 10, 1}));
  EXPECT_FALSE(saddr_def_use.uses.contains({RegClass::VGPR, 10, 2}));
  EXPECT_TRUE(saddr_def_use.uses.contains({RegClass::SGPR, 6, 2}));

  std::unique_ptr<Instruction> vector_only_inst(decode_valid(*decoder, vector_only_words));
  ASSERT_NE(vector_only_inst, nullptr);
  EXPECT_EQ(vector_only_inst->mnemonic(), "global_load_b64");
  EXPECT_EQ(vector_only_inst->num_src_operands(), 2); // vaddr + gpumem
  InstDefUse vector_only_def_use(*vector_only_inst);
  EXPECT_TRUE(vector_only_def_use.uses.contains({RegClass::VGPR, 10, 2}));
  EXPECT_FALSE(vector_only_def_use.uses.contains({RegClass::SGPR, 124, 1}));
  EXPECT_FALSE(vector_only_def_use.uses.contains({RegClass::SGPR, 124, 2}));
}

TEST(Gfx1250DecodeTest, GlobalStoreUsesScalarOffsetVaddrWidth) {
  // LLVM disassembles these words as global_store_b64 v10, v[2:3], s[6:7].
  const uint32_t words[] = {
      0xEE06C006u,
      0x01000000u,
      0x0000000Au,
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->mnemonic(), "global_store_b64");
  EXPECT_EQ(inst->disassemble(), "global_store_b64 v10, v[2:3], s[6:7]");
  InstDefUse def_use(*inst);
  EXPECT_TRUE(def_use.uses.contains({RegClass::VGPR, 10, 1}));
  EXPECT_FALSE(def_use.uses.contains({RegClass::VGPR, 10, 2}));
}

TEST(Gfx1250DecodeTest, Vop3CompareWritesSingleScalarMaskRegister) {
  const uint32_t words[] = {
      0xD44C0002u, // v_cmp_gt_u32 s2, s5, v12
      0x02021805u,
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->mnemonic(), "v_cmp_gt_u32");
  EXPECT_EQ(inst->size(), sizeof(words));

  const std::string disasm = inst->disassemble();
  EXPECT_NE(disasm.find("s2"), std::string::npos) << disasm;
  EXPECT_EQ(disasm.find("s[2:3]"), std::string::npos) << disasm;

  InstDefUse def_use(*inst);
  EXPECT_TRUE(def_use.defs.contains({RegClass::SGPR, 2, 1}));
  EXPECT_FALSE(def_use.defs.contains({RegClass::SGPR, 3, 1}));
}

TEST(Cdna4DecodeTest, MfmaF8f6f4DecodesStandaloneVop3pSuffix) {
  const uint32_t words[] = {
      0xD3AD0024u,
      0x0492F572u,
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->mnemonic(), "v_mfma_f32_16x16x128_f8f6f4");
  EXPECT_EQ(inst->size(), sizeof(words));

  InstDefUse def_use(*inst);
  EXPECT_TRUE(def_use.defs.contains({RegClass::VGPR, 36, 4}));
}

TEST(Cdna4DecodeTest, MfmaF8f6f4SourceWidthsFollowFormatSelectors) {
  struct FormatCase {
    uint32_t format;
    uint8_t register_count;
  };
  constexpr std::array cases = {
      FormatCase{0, 8}, // FP8
      FormatCase{2, 6}, // FP6
      FormatCase{4, 4}, // FP4
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);

  for (const auto &[format, register_count] : cases) {
    const uint32_t words[] = {
        0xD3AD0024u | (format << 8),
        0x00024100u | (format << 29),
    };
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, words));
    ASSERT_NE(inst, nullptr);
    ASSERT_EQ(inst->mnemonic(), "v_mfma_f32_16x16x128_f8f6f4");

    InstDefUse def_use(*inst);
    EXPECT_TRUE(def_use.uses.contains({RegClass::VGPR, 0, register_count}));
    EXPECT_FALSE(
        def_use.uses.contains({RegClass::VGPR, 0, static_cast<uint8_t>(register_count + 1)}));
    EXPECT_TRUE(def_use.uses.contains({RegClass::VGPR, 32, register_count}));
    EXPECT_FALSE(
        def_use.uses.contains({RegClass::VGPR, 32, static_cast<uint8_t>(register_count + 1)}));
  }
}

TEST(Cdna4DecodeTest, MfmaScaleF8f6f4ConsumesVop3px2Prefix) {
  const uint32_t words[] = {
      0xD3AC0000u,
      0x0002C360u,
      0xD3AD0C40u,
      0x84822100u,
  };

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->mnemonic(), "v_mfma_scale_f32_16x16x128_f8f6f4");
  EXPECT_EQ(inst->size(), sizeof(words));
  EXPECT_EQ(inst->num_src_operands(), 5);
}

// v_accvgpr_read's 9-bit src0 field encodes accumulator N as 256 + N, which the
// decoder shifts into the OPR_SRC_ACCVGPR range [768, 1023]. A raw field below 256 is
// malformed and must be left unshifted; adding the shift pushes it into [512, 767],
// which vgpr_index() reads directly as an out-of-range physical VGPR. word[0] is fixed
// (opcode + vdst=v3); word[1]'s low bits carry src0 (base 0x18000000 | 256 reproduces
// the known-good "a0" encoding). CDNA3 and CDNA4 both generate the shift. Reverting
// its `>= 256` guard makes these malformed cases resolve to 512, 517, 640, and 767.
TEST(AccVgprSrcCanonicalizationTest, MalformedRawSrcBelow256StaysInUnifiedRange) {
  for (rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4}) {
    auto decoder = Decoder::create(arch);
    ASSERT_NE(decoder, nullptr);
    for (uint32_t raw : {0u, 5u, 128u, 255u}) {
      const uint32_t words[] = {0xD3D84003u, 0x18000000u | raw};
      std::unique_ptr<Instruction> inst(decode_valid(*decoder, words));
      ASSERT_NE(inst, nullptr);
      EXPECT_EQ(inst->mnemonic(), "v_accvgpr_read");
      ASSERT_EQ(inst->num_src_operands(), 1);
      const Operand *src0 = inst->src_operand(0);
      ASSERT_NE(src0, nullptr);
      EXPECT_EQ(src0->encoding_value(), static_cast<int>(raw))
          << "arch=" << arch << " raw src0 " << raw << " was canonicalized in place";
      EXPECT_LT(src0->encoding_value(), 512)
          << "arch=" << arch << " raw src0 " << raw << " escaped into [512, 767]";
    }
  }
}

// A legal accumulator field (256 + N) resolves to the unified AccVGPR index 256 + N.
TEST(AccVgprSrcCanonicalizationTest, LegalRawSrcResolvesToUnifiedAccIndex) {
  for (rj_code_arch_t arch : {ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4}) {
    auto decoder = Decoder::create(arch);
    ASSERT_NE(decoder, nullptr);
    for (uint32_t n : {0u, 1u, 63u}) {
      const uint32_t words[] = {0xD3D84003u, 0x18000000u | (256u + n)};
      std::unique_ptr<Instruction> inst(decode_valid(*decoder, words));
      ASSERT_NE(inst, nullptr);
      EXPECT_EQ(inst->mnemonic(), "v_accvgpr_read");
      ASSERT_EQ(inst->num_src_operands(), 1);
      const Operand *src0 = inst->src_operand(0);
      ASSERT_NE(src0, nullptr);
      EXPECT_EQ(src0->unified_vgpr_index(), 256u + n);
    }
  }
}

TEST(Cdna4DecodeTest, RejectsVop3px2PrefixWithoutMfmaSuffix) {
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  ASSERT_NE(decoder, nullptr);

  const uint32_t unrelated_opcode[] = {
      0xD3AC0000u,
      0x00000000u,
      0xD3AC0000u,
      0x00000000u,
  };
  EXPECT_TRUE(decode_fails(*decoder, unrelated_opcode));

  const uint32_t wrong_encoding[] = {
      0xD3AC0000u,
      0x00000000u,
      0xD32D0000u,
      0x00000000u,
  };
  EXPECT_TRUE(decode_fails(*decoder, wrong_encoding));
}

} // namespace
