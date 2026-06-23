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

struct SmemCase {
  rj_code_arch_t arch;
  const char *arch_name;
  std::array<uint32_t, 2> word;
  const char *mnemonic;
};

class SmemSbaseOperandTest : public ::testing::TestWithParam<SmemCase> {};

// SMEM load with raw sbase = 2 -> base register must be s[4:5].
TEST_P(SmemSbaseOperandTest, SbaseResolvesToScaledSgprPair) {
  const SmemCase &tc = GetParam();

  auto decoder = Decoder::create(tc.arch);
  ASSERT_NE(decoder, nullptr) << "Decoder::create() failed for " << tc.arch_name;

  std::unique_ptr<Instruction> inst(decoder->decode(tc.word.data()));
  ASSERT_NE(inst, nullptr) << "decode() returned nullptr for " << tc.arch_name;
  ASSERT_EQ(inst->mnemonic(), tc.mnemonic) << "unexpected mnemonic for " << tc.arch_name;

  // SBASE is the only SGPR-pair source operand. sdata is the destination
  // and thus not included in the following loop. Assert exactly one
  // SGPR-pair source so a future encoding cannot silently select the wrong one.
  std::optional<RegisterRef> base;
  int sgpr_pair_srcs = 0;
  for (int i = 0; i < inst->num_src_operands(); ++i) {
    auto ref = inst->src_operand(i)->to_register_ref();
    if (ref && ref->cls == RegClass::SGPR && ref->width == 2) {
      base = ref;
      ++sgpr_pair_srcs;
    }
  }
  ASSERT_EQ(sgpr_pair_srcs, 1) << "expected exactly one SGPR-pair source (SBASE) for "
                               << tc.arch_name;

  // Raw field 2 names s[4:5]; the unscaled (incorrect) operand model yields
  // s[2:3].
  EXPECT_EQ(base->index, 4u) << "SBASE operand resolved to s[" << base->index << ":"
                             << base->index + 1 << "] (expected s[4:5]) for " << tc.arch_name;

  // The dataflow bridge that feeds liveness must see s[4:5] as a use. This is
  // the exact path whose gap let the probe-call planner clobber the live base.
  InstDefUse du(*inst);
  EXPECT_TRUE(du.uses.contains(RegisterRef{RegClass::SGPR, 4, 2}))
      << "InstDefUse.uses is missing s[4:5] for " << tc.arch_name;
}

INSTANTIATE_TEST_SUITE_P(
    SmemFamilies, SmemSbaseOperandTest,
    ::testing::Values(
        SmemCase{ROCJITSU_CODE_ARCH_CDNA1, "cdna1", kCdnaSLoadDwordx2, "s_load_dwordx2"},
        SmemCase{ROCJITSU_CODE_ARCH_CDNA2, "cdna2", kCdnaSLoadDwordx2, "s_load_dwordx2"},
        SmemCase{ROCJITSU_CODE_ARCH_CDNA3, "cdna3", kCdnaSLoadDwordx2, "s_load_dwordx2"},
        SmemCase{ROCJITSU_CODE_ARCH_CDNA4, "cdna4", kCdnaSLoadDwordx2, "s_load_dwordx2"},
        SmemCase{ROCJITSU_CODE_ARCH_RDNA1, "rdna1", kRdnaGfx1011SLoad64, "s_load_dwordx2"},
        SmemCase{ROCJITSU_CODE_ARCH_RDNA2, "rdna2", kRdnaGfx1011SLoad64, "s_load_dwordx2"},
        SmemCase{ROCJITSU_CODE_ARCH_RDNA3, "rdna3", kRdnaGfx1011SLoad64, "s_load_b64"},
        SmemCase{ROCJITSU_CODE_ARCH_RDNA3_5, "rdna3_5", kRdnaGfx1011SLoad64, "s_load_b64"},
        SmemCase{ROCJITSU_CODE_ARCH_RDNA4, "rdna4", kGfx12SLoadB64, "s_load_b64"},
        SmemCase{ROCJITSU_CODE_ARCH_GFX1250, "gfx1250", kGfx12SLoadB64, "s_load_b64"}),
    [](const ::testing::TestParamInfo<SmemCase> &info) {
      return std::string(info.param.arch_name);
    });

} // namespace
