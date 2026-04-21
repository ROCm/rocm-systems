// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file decode_smoke_test.cpp
/// @brief Parameterized decode smoke test for all 9 AMDGPU ISAs.
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
///   0xBF800000 = s_nop    (op = 0) — all 9 ISAs
///   0xBF810000 = s_endpgm (op = 1) — CDNA1/2/3/4, RDNA1/2
///                                     (GFX9/10: op=1 is s_endpgm)
///   0xBFB00000 = s_endpgm (op = 48) — RDNA3/3.5/4
///                                     (GFX11/12: op=1 is s_setkill; s_endpgm moved to op=48)

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>

namespace {

using namespace rocjitsu;

// SOPP encodings.
// s_nop is op=0 on all 9 ISAs.
// s_endpgm is op=1 on CDNA1-4 and RDNA1/2 (GFX9/GFX10).
// s_endpgm is op=48 on RDNA3/3.5/4 (GFX11/GFX12) — op=1 is s_setkill there.
constexpr uint32_t S_NOP = 0xBF800000u;          ///< s_nop    (SOPP op=0,  simm16=0)
constexpr uint32_t S_ENDPGM_GFX9 = 0xBF810000u;  ///< s_endpgm (SOPP op=1,  simm16=0): CDNA/RDNA1/2
constexpr uint32_t S_ENDPGM_GFX11 = 0xBFB00000u; ///< s_endpgm (SOPP op=48, simm16=0): RDNA3/3.5/4

struct DecodeCase {
  rj_code_arch_t arch;
  const char *arch_name;
  uint32_t word;
  const char *expected_mnemonic;
  int expected_size_bytes;
};

class DecoderSmokeTest : public ::testing::TestWithParam<DecodeCase> {};

TEST_P(DecoderSmokeTest, DecodesCorrectly) {
  const DecodeCase &tc = GetParam();

  auto decoder = Decoder::create(tc.arch);
  ASSERT_NE(decoder, nullptr) << "Decoder::create() returned nullptr for arch=" << tc.arch_name;

  std::unique_ptr<Instruction> inst(decoder->decode(&tc.word));
  ASSERT_NE(inst, nullptr) << "decode() returned nullptr for arch=" << tc.arch_name << " word=0x"
                           << std::hex << tc.word;

  EXPECT_EQ(inst->mnemonic(), tc.expected_mnemonic) << "Wrong mnemonic for arch=" << tc.arch_name;
  EXPECT_EQ(inst->size(), tc.expected_size_bytes) << "Wrong size for arch=" << tc.arch_name;
}

// 9 ISAs × 2 instructions = 18 test cases.
// CDNA1/2/3/4 and RDNA1/2 share the GFX9/GFX10 s_endpgm encoding (op=1).
// RDNA3/3.5/4 use the GFX11/GFX12 encoding where s_endpgm moved to op=48.
INSTANTIATE_TEST_SUITE_P(
    AllIsas, DecoderSmokeTest,
    ::testing::Values(DecodeCase{ROCJITSU_CODE_ARCH_CDNA1, "cdna1", S_NOP, "s_nop", 4},
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
                      DecodeCase{ROCJITSU_CODE_ARCH_RDNA3_5, "rdna3_5", S_ENDPGM_GFX11, "s_endpgm",
                                 4},
                      DecodeCase{ROCJITSU_CODE_ARCH_RDNA4, "rdna4", S_NOP, "s_nop", 4},
                      DecodeCase{ROCJITSU_CODE_ARCH_RDNA4, "rdna4", S_ENDPGM_GFX11, "s_endpgm", 4}),
    [](const ::testing::TestParamInfo<DecodeCase> &info) {
      std::string name = info.param.arch_name;
      name += "_";
      name += info.param.expected_mnemonic;
      return name;
    });

} // namespace
