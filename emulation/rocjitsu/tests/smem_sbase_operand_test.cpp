// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file smem_sbase_operand_test.cpp
/// @brief Regression test for the SMEM SBASE operand-model ×2 scale.
///
/// SMEM encodes its SBASE field in units of 2 SGPRs: a raw field value of N
/// names the SGPR pair s[2N : 2N+1].
///
/// This test crafts an SMEM load with raw sbase = 2 and asserts that both
/// `to_register_ref()` and `InstDefUse` (the dataflow bridge that feeds
/// liveness) resolve the base to s[4:5]. It covers 10 supported AMDGPU
/// targets. There are three distinct SMEM word[0] layouts:
///   - CDNA1-4: op[25:18], encoding=0x30, s_load_dwordx2. gfx90a (CDNA2) is the
///     target the bug was originally found on.
///   - RDNA1/2/3/3.5: op[25:18], encoding=0x3D, op=1 s_load_dwordx2 (RDNA1/2),
///     s_load_b64 (RDNA3/3.5).
///   - RDNA4 + gfx1250: op[18:13], encoding=0x3D, s_load_b64.
///
/// All word[0] layouts share sbase[5:0] and sdata[12:6]; they differ in the op
/// field position and the encoding constant (see the per-word comments below).
///
/// Beyond the 64-bit scalar load, two CDNA4 cases cover the other generated
/// SBASE shapes the constructors also scale:
///   - s_buffer_load_dwordx4: SBASE is a 128-bit V# descriptor -> s[4:7].
///   - s_store_dwordx2: store family, where sdata is also a source so SBASE is
///     not the first source operand.

#include "rocjitsu/analysis/def_use_chain.h"
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/operand.h"
#include "rocjitsu/isa/register_set.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace {

using namespace rocjitsu;

// All words below set raw sbase=2 (-> s[4:5]) and sdata=s0.
// Only the fields that drive dispatch + the SBASE operand are set; the rest are
// zero. Each word is self-checked by the mnemonic assertion in the test body.

/// @brief CDNA1-4 s_load_dwordx2: sbase[5:0], op[25:18]=1, encoding[31:26]=0x30.
constexpr std::array<uint32_t, 2> kCdnaSLoadDwordx2 = {
    /*lo=*/(2u & 0x3Fu) | (1u << 18) /*op*/ | (0x30u << 26) /*encoding*/, /*hi=*/0u};

/// @brief RDNA1/2/3/3.5 64-bit SMEM load: sbase[5:0], op[25:18]=1, encoding[31:26]=0x3D.
constexpr std::array<uint32_t, 2> kRdnaGfx1011SLoad64 = {
    /*lo=*/(2u & 0x3Fu) | (1u << 18) /*op*/ | (0x3Du << 26) /*encoding*/, /*hi=*/0u};

/// @brief RDNA4 + gfx1250 s_load_b64: sbase[5:0], op[18:13]=1, encoding[31:26]=0x3D.
constexpr std::array<uint32_t, 2> kGfx12SLoadB64 = {
    /*lo=*/(2u & 0x3Fu) | (1u << 13) /*op*/ | (0x3Du << 26) /*encoding*/, /*hi=*/0u};

/// @brief CDNA s_buffer_load_dwordx4 (op[25:18]=10, encoding=0x30): SBASE is a
/// 128-bit V# descriptor; sbase(128, ...) -> s[4:7] (width 4), not a pair.
constexpr std::array<uint32_t, 2> kCdnaSBufferLoadDwordx4 = {
    /*lo=*/(2u & 0x3Fu) | (10u << 18) /*op*/ | (0x30u << 26) /*encoding*/, /*hi=*/0u};

/// @brief CDNA s_store_dwordx2 (op[25:18]=17, encoding=0x30): store family, so
/// sdata (s[0:1]) is ALSO a source and SBASE is not the first source operand.
constexpr std::array<uint32_t, 2> kCdnaSStoreDwordx2 = {
    /*lo=*/(2u & 0x3Fu) | (17u << 18) /*op*/ | (0x30u << 26) /*encoding*/, /*hi=*/0u};

/// @brief RDNA4 s_buffer_load_b128 (op[18:13]=18, encoding=0x3D): b128 naming
/// for a 128-bit V# descriptor SBASE -> s[4:7] (width 4).
constexpr std::array<uint32_t, 2> kRdna4SBufferLoadB128 = {
    /*lo=*/(2u & 0x3Fu) | (18u << 13) /*op*/ | (0x3Du << 26) /*encoding*/, /*hi=*/0u};

struct SmemCase {
  rj_code_arch_t arch;
  const char *label; ///< Unique test-instance name.
  std::array<uint32_t, 2> word;
  const char *mnemonic;
  uint8_t sbase_width; ///< SBASE register width in lanes: 2 (64-bit
                       ///< pair) or 4 (128-bit V# descriptor).
};

class SmemSbaseOperandTest : public ::testing::TestWithParam<SmemCase> {};

// SMEM with raw sbase = 2 -> SBASE register must start at s4 (units of 2 SGPRs).
TEST_P(SmemSbaseOperandTest, SbaseResolvesToScaledSgpr) {
  const SmemCase &tc = GetParam();

  auto decoder = Decoder::create(tc.arch);
  ASSERT_NE(decoder, nullptr) << "Decoder::create() failed for " << tc.label;

  std::unique_ptr<Instruction> inst(decoder->decode(tc.word.data()));
  ASSERT_NE(inst, nullptr) << "decode() returned nullptr for " << tc.label;
  ASSERT_EQ(inst->mnemonic(), tc.mnemonic) << "unexpected mnemonic for " << tc.label;

  // SBASE is in units of 2 SGPRs, so sbase=2 must decode to register s4.
  // The expected operand is {SGPR, 4, width}, and exactly one source must
  // match it. If the x2 scale regressed, SBASE would decode to s2 and match
  // nothing here, failing the EXPECT_EQ below.
  const int hi = 4 + tc.sbase_width - 1;
  const RegisterRef expected_sbase{RegClass::SGPR, 4, tc.sbase_width};
  int matches = 0;
  for (int i = 0; i < inst->num_src_operands(); ++i) {
    if (inst->src_operand(i)->to_register_ref() == expected_sbase)
      ++matches;
  }
  EXPECT_EQ(matches, 1) << "expected exactly one source operand = s[4:" << hi
                        << "] (scaled SBASE) for " << tc.label;

  // The dataflow bridge that feeds liveness must see the scaled SBASE as a use.
  // This is the exact path whose gap let the probe-call planner clobber it.
  InstDefUse du(*inst);
  EXPECT_TRUE(du.uses.contains(expected_sbase))
      << "InstDefUse.uses is missing s[4:" << hi << "] for " << tc.label;
}

INSTANTIATE_TEST_SUITE_P(
    SmemFamilies, SmemSbaseOperandTest,
    ::testing::Values(
        // 64-bit scalar loads across 10 ISAs (SBASE width 2).
        SmemCase{ROCJITSU_CODE_ARCH_CDNA1, "cdna1", kCdnaSLoadDwordx2, "s_load_dwordx2", 2},
        SmemCase{ROCJITSU_CODE_ARCH_CDNA2, "cdna2", kCdnaSLoadDwordx2, "s_load_dwordx2", 2},
        SmemCase{ROCJITSU_CODE_ARCH_CDNA3, "cdna3", kCdnaSLoadDwordx2, "s_load_dwordx2", 2},
        SmemCase{ROCJITSU_CODE_ARCH_CDNA4, "cdna4", kCdnaSLoadDwordx2, "s_load_dwordx2", 2},
        SmemCase{ROCJITSU_CODE_ARCH_RDNA1, "rdna1", kRdnaGfx1011SLoad64, "s_load_dwordx2", 2},
        SmemCase{ROCJITSU_CODE_ARCH_RDNA2, "rdna2", kRdnaGfx1011SLoad64, "s_load_dwordx2", 2},
        SmemCase{ROCJITSU_CODE_ARCH_RDNA3, "rdna3", kRdnaGfx1011SLoad64, "s_load_b64", 2},
        SmemCase{ROCJITSU_CODE_ARCH_RDNA3_5, "rdna3_5", kRdnaGfx1011SLoad64, "s_load_b64", 2},
        SmemCase{ROCJITSU_CODE_ARCH_RDNA4, "rdna4", kGfx12SLoadB64, "s_load_b64", 2},
        SmemCase{ROCJITSU_CODE_ARCH_GFX1250, "gfx1250", kGfx12SLoadB64, "s_load_b64", 2},
        // 128-bit buffer-descriptor SBASE (width 4) -> s[4:7], both naming forms.
        SmemCase{ROCJITSU_CODE_ARCH_CDNA4, "cdna4_buffer_load_x4", kCdnaSBufferLoadDwordx4,
                 "s_buffer_load_dwordx4", 4},
        SmemCase{ROCJITSU_CODE_ARCH_RDNA4, "rdna4_buffer_load_b128", kRdna4SBufferLoadB128,
                 "s_buffer_load_b128", 4},
        // Store family: SBASE is not the first source (sdata precedes it).
        SmemCase{ROCJITSU_CODE_ARCH_CDNA4, "cdna4_store_x2", kCdnaSStoreDwordx2, "s_store_dwordx2",
                 2}),
    [](const ::testing::TestParamInfo<SmemCase> &info) { return std::string(info.param.label); });

} // namespace
